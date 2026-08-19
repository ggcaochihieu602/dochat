from __future__ import annotations

import asyncio
import html
import io
import logging
import os
import uuid
from pathlib import Path
from typing import Literal

import fitz
import httpx
import pytesseract
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError
from pydantic import BaseModel

from .helpers import (
    MAX_FILE_BYTES,
    MAX_PDF_PAGES,
    clean_text,
    load_prompt,
    split_for_tts,
    split_text,
)

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
Image.MAX_IMAGE_PIXELS = 30_000_000
logger = logging.getLogger("dochat")

app = FastAPI(title="Dochat processing backend")


class PermanentJobError(Exception):
    pass


class ExternalServiceError(Exception):
    pass


class Job(BaseModel):
    update_id: int
    chat_id: int
    user_id: int
    source_type: Literal["direct_text", "text_file", "pdf", "image"]
    mode: Literal["original_audio", "summary_text", "summary_audio"]
    file_id: str | None = None
    text: str | None = None
    status_message_id: int | None = None


def setting(name: str, local_file: str | None = None) -> str:
    value = os.getenv(name, "").strip()
    if not value and local_file:
        path = ROOT / local_file
        if path.is_file():
            value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise RuntimeError(f"Missing configuration: {name}")
    return value


def telegram_url(method: str) -> str:
    return f"https://api.telegram.org/bot{setting('TELEGRAM_BOT_TOKEN', 'Telegram_bot_api.txt')}/{method}"


async def telegram(method: str, **data):
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(telegram_url(method), data=data)
        payload = response.json()
        if response.is_error or not payload.get("ok"):
            raise ExternalServiceError("Telegram API request failed")
        return payload["result"]
    except (httpx.HTTPError, ValueError) as error:
        raise ExternalServiceError("Telegram API request failed") from error


async def edit_status(job: Job, text: str) -> None:
    if not job.status_message_id:
        return
    try:
        await telegram(
            "editMessageText",
            chat_id=job.chat_id,
            message_id=job.status_message_id,
            text=text,
        )
    except ExternalServiceError:
        pass


async def download_file(file_id: str) -> bytes:
    if not file_id:
        raise PermanentJobError("Không tìm thấy tệp cần xử lý.")
    file = await telegram("getFile", file_id=file_id)
    url = (
        "https://api.telegram.org/file/bot"
        f"{setting('TELEGRAM_BOT_TOKEN', 'Telegram_bot_api.txt')}/{file['file_path']}"
    )
    content = bytearray()
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("GET", url) as response:
                if response.is_error:
                    raise ExternalServiceError("Telegram file download failed")
                async for chunk in response.aiter_bytes():
                    content.extend(chunk)
                    if len(content) > MAX_FILE_BYTES:
                        raise PermanentJobError("Tệp vượt quá giới hạn 15 MB.")
    except httpx.HTTPError as error:
        raise ExternalServiceError("Telegram file download failed") from error
    return bytes(content)


def decode_text_file(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    return data.decode("utf-8-sig", errors="replace")


def prepare_image(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    image.thumbnail((2400, 2400))
    grayscale = ImageOps.grayscale(image)
    return ImageEnhance.Contrast(grayscale).enhance(1.5)


def ocr_image(data: bytes) -> str:
    try:
        with Image.open(io.BytesIO(data)) as image:
            return clean_text(
                pytesseract.image_to_string(prepare_image(image), lang="vie+eng")
            )
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as error:
        raise PermanentJobError("Ảnh không hợp lệ hoặc có độ phân giải quá lớn.") from error


def extract_pdf(data: bytes) -> str:
    try:
        with fitz.open(stream=data, filetype="pdf") as document:
            if document.page_count > MAX_PDF_PAGES:
                raise PermanentJobError("PDF chỉ được tối đa 2 trang.")
            pages: list[str] = []
            for page in document:
                text = clean_text(page.get_text("text"))
                if len(text) < 80:
                    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                    text = ocr_image(pixmap.tobytes("png"))
                pages.append(text)
            return clean_text("\n\n".join(pages))
    except PermanentJobError:
        raise
    except Exception as error:
        raise PermanentJobError("PDF không hợp lệ hoặc không thể đọc được.") from error


def extract(data: bytes, source_type: str) -> str:
    if source_type == "direct_text" or source_type == "text_file":
        result = clean_text(decode_text_file(data))
    elif source_type == "image":
        result = ocr_image(data)
    elif source_type == "pdf":
        result = extract_pdf(data)
    else:
        raise PermanentJobError("Loại nội dung không được hỗ trợ.")
    if len(result) < 10:
        raise PermanentJobError(
            "Trợ lý Dochat chưa đọc rõ nội dung. Ngài vui lòng gửi ảnh rõ và thẳng hơn."
        )
    return result


async def summarize(text: str) -> tuple[str, bool]:
    try:
        prompt_path = Path(os.getenv("SUMMARY_PROMPT_PATH", str(ROOT / "SUMMARY_PROMPT.md")))
        prompt = load_prompt(prompt_path).replace("{{DOCUMENT_TEXT}}", text)
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                setting("SUMMARY_PROVIDER_URL"),
                json={"prompt": prompt},
                headers={"X-Internal-Secret": setting("INTERNAL_SECRET")},
            )
        if response.is_error:
            raise ExternalServiceError("Summary provider failed")
        result = clean_text(response.json().get("text", ""))
        if not result:
            raise ExternalServiceError("Summary provider returned no text")
        return result, False
    except Exception:
        return text, True


async def synthesize(text: str) -> bytes:
    region = setting("AZURE_SPEECH_REGION", "AZURE_SPEECH_REGION.txt")
    voice = os.getenv("AZURE_VOICE_NAME", "vi-VN-NamMinhNeural")
    ssml = (
        '<speak version="1.0" xml:lang="vi-VN">'
        f'<voice name="{html.escape(voice)}">{html.escape(text)}</voice></speak>'
    )
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1",
                content=ssml.encode("utf-8"),
                headers={
                    "Ocp-Apim-Subscription-Key": setting(
                        "AZURE_SPEECH_KEY", "AZURE_SPEECH_KEY.txt"
                    ),
                    "Content-Type": "application/ssml+xml",
                    "X-Microsoft-OutputFormat": "audio-24khz-48kbitrate-mono-mp3",
                },
            )
        if response.is_error or not response.content:
            raise ExternalServiceError("Azure Speech request failed")
        return response.content
    except httpx.HTTPError as error:
        raise ExternalServiceError("Azure Speech request failed") from error


async def send_text(chat_id: int, text: str) -> None:
    for part in split_text(text):
        await telegram("sendMessage", chat_id=chat_id, text=part)


async def send_audio(chat_id: int, text: str) -> None:
    chunks = split_for_tts(
        text,
        max_words=int(os.getenv("MAX_TTS_WORDS", "600")),
        max_chars=int(os.getenv("MAX_TTS_CHARS", "5000")),
    )
    total = len(chunks)
    for index, chunk in enumerate(chunks, 1):
        audio = await synthesize(chunk)
        caption = "Trợ lý Dochat"
        if total > 1:
            caption += f" - Phần {index}/{total}"
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(
                    telegram_url("sendAudio"),
                    data={"chat_id": str(chat_id), "caption": caption},
                    files={"audio": (f"dochat-{index}.mp3", audio, "audio/mpeg")},
                )
            payload = response.json()
            if response.is_error or not payload.get("ok"):
                raise ExternalServiceError("Telegram audio upload failed")
        except (httpx.HTTPError, ValueError) as error:
            raise ExternalServiceError("Telegram audio upload failed") from error


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/process")
async def process(job: Job, x_internal_secret: str | None = Header(default=None)):
    if x_internal_secret != setting("INTERNAL_SECRET"):
        raise HTTPException(401, "Unauthorized")

    error_id = uuid.uuid4().hex[:6].upper()
    try:
        await edit_status(job, "Trợ lý Dochat đang đọc nội dung...")
        raw = (
            (job.text or "").encode("utf-8")
            if job.source_type == "direct_text"
            else await download_file(job.file_id or "")
        )
        extracted = await asyncio.to_thread(extract, raw, job.source_type)

        final_text = extracted
        summary_failed = False
        if job.mode in {"summary_text", "summary_audio"}:
            await edit_status(job, "Trợ lý Dochat đang tóm tắt nội dung...")
            final_text, summary_failed = await summarize(extracted)
            if summary_failed:
                await telegram(
                    "sendMessage",
                    chat_id=job.chat_id,
                    text=(
                        "Trợ lý Dochat chưa thể tóm tắt nội dung này. "
                        "Tôi sẽ gửi lại nội dung nguyên văn cho ngài."
                    ),
                )

        if job.mode == "summary_text":
            await send_text(job.chat_id, final_text)
        else:
            await edit_status(job, "Trợ lý Dochat đang tạo giọng nói...")
            if job.mode == "summary_audio" and not summary_failed:
                await send_text(job.chat_id, final_text)
            await send_audio(job.chat_id, final_text)

        await edit_status(job, "Trợ lý Dochat đã xử lý xong.")
        return {"status": "completed"}
    except PermanentJobError as error:
        return {
            "status": "rejected",
            "user_message": str(error),
            "error_id": error_id,
        }
    except Exception as error:
        logger.error("job_failed error_id=%s type=%s", error_id, type(error).__name__)
        raise HTTPException(503, f"Processing failed: {error_id}") from None
