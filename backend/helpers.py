from __future__ import annotations

import re
import unicodedata
from pathlib import Path

MAX_FILE_BYTES = 15 * 1024 * 1024
MAX_PDF_PAGES = 2


_ADMIN_ABBREVIATIONS = (
    # Context-safe organization and signing abbreviations.
    (re.compile(r"\bHỘI\s+ND\b", re.IGNORECASE), "Hội Nông dân"),
    (re.compile(r"\bMTTQ\b", re.IGNORECASE), "Mặt trận Tổ quốc"),
    (re.compile(r"\bUBND\b", re.IGNORECASE), "Ủy ban nhân dân"),
    (re.compile(r"\bHĐND\b", re.IGNORECASE), "Hội đồng nhân dân"),
    (re.compile(r"\bĐBQH\b", re.IGNORECASE), "Đại biểu Quốc hội"),
    (re.compile(r"\bBCH\.(?=\s|$)", re.IGNORECASE), "Ban Chấp hành"),
    (re.compile(r"\bBCH\b", re.IGNORECASE), "Ban Chấp hành"),
    (re.compile(r"\bTM\.(?=\s|$)", re.IGNORECASE), "Thay mặt"),
    (re.compile(r"\bKT\.(?=\s|$)", re.IGNORECASE), "Ký thay"),
    (re.compile(r"\bTL\.(?=\s|$)", re.IGNORECASE), "Thừa lệnh"),
    (re.compile(r"\bTUQ\.(?=\s|$)", re.IGNORECASE), "Thừa ủy quyền"),
    (re.compile(r"(?i)(?<=Lưu\s)VP\b"), "Văn phòng"),
)


def expand_administrative_abbreviations(value: str) -> str:
    """Expand only unambiguous administrative abbreviations for reading/TTS."""
    result = value
    for pattern, replacement in _ADMIN_ABBREVIATIONS:
        result = pattern.sub(replacement, result)
    return result


def add_punctuation_to_document_headers(value: str) -> str:
    """Add pauses to short document-header lines without touching paragraph wraps."""
    lines = value.splitlines()
    output: list[str] = []
    header_pattern = re.compile(
        r"^(Ban Chấp hành|Hội Nông dân|CỘNG HÒA|Độc lập|Số\s*:|"
        r"[A-ZÀ-Ỹ][A-ZÀ-Ỹ .-]{5,}|[A-ZÀ-Ỹ].*ngày\s+\d+|GIẤY MỜI|"
        r"Dự Lễ|Nơi nhận|Thay mặt)",
        re.IGNORECASE,
    )
    for line in lines:
        stripped = line.strip()
        if not stripped:
            output.append("")
            continue
        if stripped.startswith("-") and not stripped.endswith((";", ".", ":")):
            stripped += ";"
        elif (
            len(stripped) <= 120
            and header_pattern.match(stripped)
            and not stripped.endswith((".", ":", ";", ",", "!", "?", "…"))
        ):
            stripped += ","
        output.append(stripped)
    return "\n".join(output)

_ALLOWED_VOICE_PUNCT = set(".,!?…;:'\"()-–—/%")


def strip_special_symbols(value: str) -> str:
    """Keep only letters, digits, combining marks, whitespace and basic punctuation."""
    cleaned = [
        char
        for char in value
        if char.isspace()
        or char.isalnum()
        or unicodedata.combining(char) != 0
        or char in _ALLOWED_VOICE_PUNCT
    ]
    result = "".join(cleaned)
    result = re.sub(r"[ \t]+", " ", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def chunk_text_for_normalize(text: str, limit: int = 6000) -> list[str]:
    """Split text on line boundaries into chunks that fit the model context."""
    text = clean_text(text)
    if not text:
        return []
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    for line in text.split("\n"):
        if not line.strip():
            continue
        if len("\n".join((*current, line))) > limit and current:
            chunks.append("\n".join(current))
            current = []
        current.append(line)
    if current:
        chunks.append("\n".join(current))
    return chunks


def chunk_text_for_summary(text: str, limit: int = 18000) -> list[str]:
    """Split text into paragraph-respecting chunks no longer than `limit` characters."""
    if not text:
        return []
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        if len(block) > limit:
            # An oversize paragraph: split by words without dropping content.
            for chunk in _split_words_to_limit(block, limit):
                chunks.append(chunk)
            continue
        if current and len("\n\n".join((*current, block))) > limit:
            chunks.append("\n\n".join(current))
            current = []
        current.append(block)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _split_words_to_limit(text: str, limit: int) -> list[str]:
    """Greedy word-boundary split so every chunk is no longer than `limit`."""
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    current: list[str] = []
    for word in words:
        if current and len(" ".join((*current, word))) > limit:
            chunks.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        chunks.append(" ".join(current))
    return chunks
    return chunks


def split_text(text: str, limit: int = 3500) -> list[str]:
    """Split text into Telegram-safe messages without losing content."""
    return _split_by_boundaries(text, max_chars=limit)


def split_for_tts(text: str, max_words: int = 600, max_chars: int = 5000) -> list[str]:
    """Split at sentence boundaries before synthesis so every MP3 is valid."""
    return _split_by_boundaries(
        strip_special_symbols(text), max_words=max_words, max_chars=max_chars
    )


def _split_by_boundaries(
    text: str,
    *,
    max_chars: int,
    max_words: int | None = None,
) -> list[str]:
    text = clean_text(text)
    if not text:
        return []

    units = re.split(r"(?<=[.!?…])\s+|\n+", text)
    parts: list[str] = []
    current: list[str] = []

    def fits(candidate: str) -> bool:
        return len(candidate) <= max_chars and (
            max_words is None or len(candidate.split()) <= max_words
        )

    def flush() -> None:
        if current:
            parts.append(" ".join(current).strip())
            current.clear()

    for unit in (item.strip() for item in units if item.strip()):
        if not fits(unit):
            flush()
            words = unit.split()
            chunk: list[str] = []
            for word in words:
                candidate = " ".join((*chunk, word))
                if chunk and not fits(candidate):
                    parts.append(" ".join(chunk))
                    chunk = [word]
                else:
                    chunk.append(word)
            if chunk:
                parts.append(" ".join(chunk))
            continue

        candidate = " ".join((*current, unit))
        if current and not fits(candidate):
            flush()
        current.append(unit)

    flush()
    return parts


def load_prompt(path: str | Path = "SUMMARY_PROMPT.md") -> str:
    prompt = Path(path).read_text(encoding="utf-8")
    if "{{DOCUMENT_TEXT}}" not in prompt:
        raise ValueError("SUMMARY_PROMPT.md must contain {{DOCUMENT_TEXT}}")
    return prompt


def clean_text(value: str) -> str:
    value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", value)
    value = re.sub(r"[ \t]+", " ", value)
    return re.sub(r"\n{3,}", "\n\n", value).strip()


def looks_like_ocr_failure(text: str) -> bool:
    """Detect obvious OCR failures without asking an LLM to repair the text."""
    text = clean_text(text)
    if len(text) < 40:
        return True
    words = re.findall(r"[^\s]+", text)
    if len(words) < 8:
        return True
    letters = sum(char.isalpha() for char in text)
    digits = sum(char.isdigit() for char in text)
    if letters == 0 or letters / max(len(text), 1) < 0.35:
        return True
    # A page that is almost entirely punctuation/noise is not usable for TTS.
    noise = len(re.findall(r"[^\w\sÀ-ỹ.,!?;:'\"()/%-]", text, re.UNICODE))
    return noise > max(12, len(text) // 12) and digits < letters // 4
