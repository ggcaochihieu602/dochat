from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import re
import time
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
    chunk_text_for_normalize,
    chunk_text_for_summary,
    clean_text,
    load_prompt,
    split_for_tts,
    split_text,
    looks_like_ocr_failure,
)

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
Image.MAX_IMAGE_PIXELS = 30_000_000
logger = logging.getLogger("dochat")

app = FastAPI(title="Dochat processing backend")

_RESULT_TTL_SECONDS = 30 * 60
_RESULTS: dict[int, tuple[dict, float]] = {}
_IN_FLIGHT: set[int] = set()
_OUTPUT_SENT: set[int] = set()
_JOB_LOCK = asyncio.Lock()

_PENDING_TTL_SECONDS = 30 * 60
_PENDING: dict[int, tuple[dict, float]] = {}


def _prune_results() -> None:
    now = time.monotonic()
    stale = [key for key, (_, created) in _RESULTS.items() if now - created > _RESULT_TTL_SECONDS]
    for key in stale:
        del _RESULTS[key]


def _prune_pending() -> None:
    now = time.monotonic()
    stale = [key for key, (_, created) in _PENDING.items() if now - created > _PENDING_TTL_SECONDS]
    for key in stale:
        del _PENDING[key]


def _get_pending(update_id: int) -> dict | None:
    _prune_pending()
    pending = _PENDING.get(update_id)
    return pending[0] if pending else None


def _set_pending(update_id: int, value: dict) -> None:
    _prune_pending()
    _PENDING[update_id] = (value, time.monotonic())


def _del_pending(update_id: int) -> None:
    _PENDING.pop(update_id, None)


def _get_result(update_id: int) -> dict | None:
    _prune_results()
    cached = _RESULTS.get(update_id)
    return cached[0] if cached else None


def _set_result(update_id: int, result: dict) -> None:
    _prune_results()
    _RESULTS[update_id] = (result, time.monotonic())


async def _claim_job(update_id: int) -> bool:
    """Return True if this request should run the job, False if it must wait."""
    async with _JOB_LOCK:
        if update_id in _RESULTS:
            return False
        if update_id in _IN_FLIGHT:
            return False
        _IN_FLIGHT.add(update_id)
        return True


async def _release_job(update_id: int) -> None:
    async with _JOB_LOCK:
        _IN_FLIGHT.discard(update_id)


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
    approved: bool = False
    parent_update_id: int | None = None


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
    image.thumbnail((4500, 4500))
    # Small Telegram photos are a common source of missed Vietnamese diacritics.
    # Upscale only the short side to avoid unbounded memory use.
    short_side = min(image.size)
    if short_side < 1400:
        scale = min(2.0, 1400 / max(short_side, 1))
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    grayscale = ImageOps.grayscale(image)
    grayscale = ImageOps.autocontrast(grayscale, cutoff=1)
    return ImageEnhance.Contrast(grayscale).enhance(1.35)


def ocr_image(data: bytes) -> str:
    try:
        with Image.open(io.BytesIO(data)) as image:
            text, _ = ocr_reconstruct(prepare_image(image))
            return clean_text(text)
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as error:
        raise PermanentJobError("Ảnh không hợp lệ hoặc có độ phân giải quá lớn.") from error


def _order_items(items: list[tuple], page_width: float, group_lines: bool) -> str:
    """Reading order: top-to-bottom, left-to-right within each line."""
    items.sort(key=lambda item: (item[1], item[0]))
    if not group_lines:
        return "\n".join(item[4] for item in items)
    lines: list[list[tuple]] = []
    for item in items:
        if not lines:
            lines.append([item])
            continue
        previous = lines[-1]
        previous_bottom = max(entry[3] for entry in previous)
        if item[1] <= previous_bottom + 4:
            previous.append(item)
        else:
            lines.append([item])
    parts: list[str] = []
    for line in lines:
        line.sort(key=lambda entry: entry[0])
        parts.append(" ".join(entry[4] for entry in line))
    return "\n".join(parts)


def ocr_reconstruct(image: Image.Image) -> tuple[str, float]:
    """OCR with region-based reading-order reconstruction for multi-column documents."""
    data = pytesseract.image_to_data(image, lang="vie+eng", output_type=pytesseract.Output.DICT)
    words: list[tuple] = []
    for index, raw in enumerate(data.get("text", [])):
        text = (raw or "").strip()
        try:
            # Tesseract returns values such as "96.421234". Using int() here
            # made valid OCR words look like confidence 0 in the old pipeline.
            confidence = float(data["conf"][index] or 0)
        except (ValueError, TypeError):
            confidence = 0.0
        if not text or confidence < 12:
            continue
        left, top, width, height = (
            data["left"][index],
            data["top"][index],
            data["width"][index],
            data["height"][index],
        )
        words.append((float(left), float(top), float(left + width), float(top + height), text, confidence))
    if not words:
        return "", 0.0
    page_width = max(word[2] for word in words)
    result = _order_items(words, page_width, group_lines=True)
    average_confidence = sum(word[5] for word in words) / len(words)
    return result, average_confidence


def extract_pdf_text(page) -> str:
    """Reconstruct reading order from positioned text blocks (handles 2-column layouts)."""
    items: list[tuple] = []
    for block in page.get_text("blocks"):
        x0, y0, x1, y1, text, *_ = block
        text = clean_text(text)
        if text:
            items.append((float(x0), float(y0), float(x1), float(y1), text))
    if not items:
        return ""
    return _order_items(items, page.rect.width, group_lines=False)


def extract_pdf(data: bytes) -> str:
    try:
        with fitz.open(stream=data, filetype="pdf") as document:
            if document.page_count > MAX_PDF_PAGES:
                raise PermanentJobError("PDF chỉ được tối đa 2 trang.")
            pages: list[str] = []
            for page in document:
                text = clean_text(extract_pdf_text(page))
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


def ocr_image_with_confidence(data: bytes) -> tuple[str, float]:
    try:
        with Image.open(io.BytesIO(data)) as image:
            text, confidence = ocr_reconstruct(prepare_image(image))
            return clean_text(text), confidence
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as error:
        raise PermanentJobError("Ảnh không hợp lệ hoặc có độ phân giải quá lớn.") from error


async def vision_repair(data: bytes) -> str:
    """Ask the Worker Vision model only when local OCR is clearly unreliable."""
    provider = os.getenv("VISION_PROVIDER_URL", "").strip()
    if not provider:
        summary_provider = setting("SUMMARY_PROVIDER_URL")
        provider = summary_provider.rsplit("/", 1)[0] + "/vision-ocr"
    try:
        with Image.open(io.BytesIO(data)) as image:
            normalized = io.BytesIO()
            ImageOps.exif_transpose(image).convert("RGB").save(normalized, format="PNG", optimize=True)
            encoded = base64.b64encode(normalized.getvalue()).decode("ascii")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as error:
        raise PermanentJobError("Ảnh không hợp lệ hoặc có độ phân giải quá lớn.") from error
    prompt = (
        "Đây là ảnh tài liệu hành chính tiếng Việt. Hãy tái dựng văn bản để đọc bằng giọng nói. "
        "Giữ nguyên tiêu đề, tên riêng, ngày tháng, số tiền, số lượng và yêu cầu. "
        "Bỏ qua quốc hiệu, tiêu ngữ, số hiệu, nơi nhận và chữ ký nếu không phải nội dung chính. "
        "Nếu có bảng, chuyển thành các dòng có nhãn. Không đoán phần mờ; ghi [KHÔNG RÕ]. "
        "Chỉ trả về văn bản, không giải thích."
    )
    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post(
            provider,
            json={"image_base64": encoded, "prompt": prompt},
            headers={"X-Internal-Secret": setting("INTERNAL_SECRET")},
        )
    if response.is_error:
        raise ExternalServiceError("Vision provider failed")
    return clean_text(response.json().get("text", ""))


SUMMARY_CHUNK_CHARS = int(os.getenv("SUMMARY_CHUNK_CHARS", "18000"))
CONSOLIDATE_PROMPT = (
    "Ngài là trợ lý tóm tắt. Hãy gộp các đoạn tóm tắt phụ sau thành một bản tóm tắt "
    "duy nhất, mạch lạc, bằng tiếng Việt, phù hợp đọc giọng nói. Giữ chính xác ngày tháng, "
    "thời hạn, số tiền, số ký hiệu và các yêu cầu quan trọng; không thêm thông tin mới, "
    "không bịa đặt, không dùng bảng/Markdown. Nếu một đoạn tóm tắt có nói 'không có thông tin', "
    "giữ nguyên. Độ dài không vượt quá 650 từ.\n\nCác đoạn tóm tắt phụ:\n{{DOCUMENT_TEXT}}"
)


async def _summarize_one(prompt: str, provider: str, secret: str, max_tokens: int) -> str:
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            provider,
            json={"prompt": prompt, "max_tokens": max_tokens},
            headers={"X-Internal-Secret": secret},
        )
    if response.is_error:
        raise ExternalServiceError("Summary provider failed")
    return clean_text(response.json().get("text", ""))


async def summarize(text: str) -> tuple[str, bool]:
    try:
        template = load_prompt(
            Path(os.getenv("SUMMARY_PROMPT_PATH", str(ROOT / "SUMMARY_PROMPT.md")))
        )
        provider = setting("SUMMARY_PROVIDER_URL")
        secret = setting("INTERNAL_SECRET")
        summaries: list[str] = []
        for chunk in chunk_text_for_summary(text, SUMMARY_CHUNK_CHARS):
            partial = await _summarize_one(
                template.replace("{{DOCUMENT_TEXT}}", chunk), provider, secret, 1024
            )
            if partial:
                summaries.append(partial)
        if not summaries:
            raise ExternalServiceError("Summary provider returned no text")
        if len(summaries) == 1:
            result = summaries[0]
        else:
            combined = "\n\n".join(summaries)
            result = await _summarize_one(
                CONSOLIDATE_PROMPT.replace("{{DOCUMENT_TEXT}}", combined),
                provider,
                secret,
                4096,
            )
        if not result:
            raise ExternalServiceError("Summary provider returned no text")
        return result, False
    except Exception:
        return text, True


async def normalize_text(text: str) -> str:
    """Add missing sentence punctuation before TTS; keep content unchanged."""
    provider = os.getenv(
        "NORMALIZE_PROVIDER_URL",
        setting("SUMMARY_PROVIDER_URL").replace("/internal/summarize", "/internal/normalize"),
    )
    try:
        parts: list[str] = []
        for chunk in chunk_text_for_normalize(text):
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(
                    provider,
                    json={"text": chunk},
                    headers={"X-Internal-Secret": setting("INTERNAL_SECRET")},
                )
            if response.is_error:
                raise ExternalServiceError("Normalize provider failed")
            normalized = response.json().get("text", "").strip()
            if not normalized:
                raise ExternalServiceError("Normalize provider returned no text")
            source_tokens = re.findall(r"[\wÀ-ỹ]+", chunk, flags=re.UNICODE)
            result_tokens = re.findall(r"[\wÀ-ỹ]+", normalized, flags=re.UNICODE)
            if source_tokens != result_tokens:
                raise ExternalServiceError("Normalize provider changed document words")
            parts.append(normalized)
        result = clean_text("\n\n".join(parts))
        return result or text
    except Exception:
        return text


async def synthesize(text: str) -> bytes:
    token = setting("VIETTEL_TTS_TOKEN")
    voice = os.getenv("VIETTEL_TTS_VOICE", "hn-quynhanh")
    speed = float(os.getenv("VIETTEL_TTS_SPEED", "1.0"))
    use_filter = os.getenv("VIETTEL_TTS_FILTER", "false").lower() == "true"
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                "https://viettelai.vn/tts/speech_synthesis",
                json={
                    "text": text,
                    "voice": voice,
                    "speed": speed,
                    "tts_return_option": 3,
                    "token": token,
                    "without_filter": not use_filter,
                },
            )
        if response.is_error or not response.content:
            raise ExternalServiceError("Viettel AI TTS request failed")
        return response.content
    except httpx.HTTPError as error:
        raise ExternalServiceError("Viettel AI TTS request failed") from error


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
                    files={"audio": (f"thongbao-{index}.mp3", audio, "audio/mpeg")},
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

    if job.approved:
        parent = job.parent_update_id or job.update_id
        pending = _get_pending(parent)
        if pending is None:
            await telegram(
                "sendMessage",
                chat_id=job.chat_id,
                text="Phiên xử lý đã hết hạn. Ngài vui lòng gửi lại tài liệu.",
            )
            return {"status": "completed"}
        cached = _get_result(job.update_id)
        if cached is not None:
            return cached
        if not await _claim_job(job.update_id):
            for _ in range(240):
                cached = _get_result(job.update_id)
                if cached is not None:
                    return cached
                await asyncio.sleep(0.5)
            return {"status": "completed"}
        work = job.model_copy(
            update={
                "source_type": "direct_text",
                "text": pending["text"],
                "mode": pending["mode"],
                "status_message_id": pending["status_message_id"],
            }
        )
        error_id = uuid.uuid4().hex[:6].upper()
        try:
            await edit_status(work, "Trợ lý Dochat đang tạo giọng nói...")
            # The text was normalized before the approval message was sent.
            # Synthesize exactly what the user reviewed.
            audio_text = work.text or ""
            _OUTPUT_SENT.add(work.update_id)
            await send_audio(work.chat_id, audio_text)
            await edit_status(work, "Trợ lý Dochat đã xử lý xong.")
            _del_pending(parent)
            result = {"status": "completed"}
            _set_result(work.update_id, result)
            return result
        except PermanentJobError as error:
            result = {
                "status": "rejected",
                "user_message": str(error),
                "error_id": error_id,
            }
            _set_result(work.update_id, result)
            return result
        except Exception as error:
            logger.error("job_failed error_id=%s type=%s", error_id, type(error).__name__)
            if work.update_id in _OUTPUT_SENT:
                _set_result(work.update_id, {"status": "completed"})
                return {"status": "completed"}
            raise HTTPException(503, f"Processing failed: {error_id}") from None
        finally:
            await _release_job(work.update_id)

    cached = _get_result(job.update_id)
    if cached is not None:
        return cached

    if not await _claim_job(job.update_id):
        for _ in range(240):
            cached = _get_result(job.update_id)
            if cached is not None:
                return cached
            await asyncio.sleep(0.5)
        return {"status": "completed"}

    error_id = uuid.uuid4().hex[:6].upper()
    try:
        await edit_status(job, "Trợ lý Dochat đang đọc nội dung...")
        raw = (
            (job.text or "").encode("utf-8")
            if job.source_type == "direct_text"
            else await download_file(job.file_id or "")
        )
        if job.source_type == "image":
            extracted, ocr_confidence = await asyncio.to_thread(ocr_image_with_confidence, raw)
            # Confidence alone is not enough to replace OCR: generative Vision
            # models can produce fluent but spatially scrambled documents.
            # Keep Tesseract as the source of truth whenever it returned a
            # substantial document; use Vision only for an actual OCR failure.
            if looks_like_ocr_failure(extracted):
                await edit_status(job, "Trợ lý Dochat đang kiểm tra lại bố cục ảnh...")
                try:
                    repaired = await vision_repair(raw)
                    if len(repaired) >= max(40, len(extracted) // 3) and not looks_like_ocr_failure(repaired):
                        extracted = repaired
                except Exception as error:
                    logger.warning("vision_fallback_failed type=%s", type(error).__name__)
        else:
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

        await edit_status(job, "Trợ lý Dochat đang thêm dấu câu...")
        final_text = await normalize_text(final_text)

        if job.mode == "summary_text":
            _OUTPUT_SENT.add(job.update_id)
            await send_text(job.chat_id, final_text)
        else:
            await edit_status(job, "Trợ lý Dochat đang chuẩn bị văn bản...")
            for part in split_text(final_text):
                await telegram("sendMessage", chat_id=job.chat_id, text=part)
            await telegram(
                "sendMessage",
                chat_id=job.chat_id,
                text="Trợ lý Dochat đã đọc xong văn bản. Ngài xác nhận tạo giọng đọc?",
                reply_markup=json.dumps(
                    {
                        "inline_keyboard": [
                            [
                                {
                                    "text": "Đồng ý tạo giọng đọc",
                                    "callback_data": f"job:approve:{job.update_id}:{job.mode}",
                                },
                                {
                                    "text": "Hủy",
                                    "callback_data": f"job:reject:{job.update_id}:{job.mode}",
                                },
                            ]
                        ]
                    }
                ),
            )
            _set_pending(
                job.update_id,
                {
                    "text": final_text,
                    "mode": job.mode,
                    "status_message_id": job.status_message_id,
                },
            )
            result = {"status": "awaiting_approval"}
            _set_result(job.update_id, result)
            return result

        await edit_status(job, "Trợ lý Dochat đã xử lý xong.")
        result = {"status": "completed"}
        _set_result(job.update_id, result)
        return result
    except PermanentJobError as error:
        result = {
            "status": "rejected",
            "user_message": str(error),
            "error_id": error_id,
        }
        _set_result(job.update_id, result)
        return result
    except Exception as error:
        logger.error("job_failed error_id=%s type=%s", error_id, type(error).__name__)
        if job.update_id in _OUTPUT_SENT:
            _set_result(job.update_id, {"status": "completed"})
            return {"status": "completed"}
        raise HTTPException(503, f"Processing failed: {error_id}") from None
    finally:
        await _release_job(job.update_id)
