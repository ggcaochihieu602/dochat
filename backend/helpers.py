from __future__ import annotations

import re
import unicodedata
from pathlib import Path

MAX_FILE_BYTES = 15 * 1024 * 1024
MAX_PDF_PAGES = 2

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
