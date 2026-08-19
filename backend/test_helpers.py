from .helpers import (
    chunk_text_for_normalize,
    split_for_tts,
    split_text,
    strip_special_symbols,
)


def test_split_text_respects_limit():
    parts = split_text("word " * 100, 20)
    assert all(len(part) <= 20 for part in parts)
    assert " ".join(parts).replace("  ", " ").strip() == ("word " * 100).strip()


def test_tts_split_keeps_sentences_complete_and_concatenates_back():
    text = "Câu thứ nhất có nội dung. Câu thứ hai có nội dung. Câu thứ ba."
    parts = split_for_tts(text, max_words=12, max_chars=100)
    assert parts == ["Câu thứ nhất có nội dung. Câu thứ hai có nội dung.", "Câu thứ ba."]
    assert " ".join(parts) == text


def test_tts_split_breaks_one_oversized_sentence_without_losing_words():
    text = " ".join(f"từ{i}" for i in range(25))
    parts = split_for_tts(text, max_words=10, max_chars=1000)
    assert [len(part.split()) for part in parts] == [10, 10, 5]
    assert " ".join(parts) == text


def test_strip_special_symbols_removes_emoji_and_symbols_keeps_vietnamese():
    text = "Xin chào 🎉 thế giới! ✔ Đây là bài test 😀 100% đúng. @#$%^&*"
    result = strip_special_symbols(text)
    assert "🎉" not in result
    assert "✔" not in result
    assert "😀" not in result
    assert "@" not in result and "#" not in result and "&" not in result
    assert "Xin chào" in result
    assert "thế giới" in result
    assert "Đây là bài test" in result
    assert "100" in result and "đúng" in result


def test_strip_special_symbols_keeps_basic_punctuation():
    text = 'Xin chào, "thế giới"! Tôi tên là Nguyễn Văn A — rất vui được gặp ngài.'
    result = strip_special_symbols(text)
    assert "," in result and "!" in result and '"' in result
    assert "Nguyễn Văn A" in result


def test_tts_split_strips_emoji_from_voice_text():
    parts = split_for_tts("Chào bạn 👋 hôm nay trời đẹp.", max_words=10, max_chars=100)
    assert all("👋" not in part for part in parts)
    assert parts == ["Chào bạn hôm nay trời đẹp."]


def test_chunk_text_for_normalize_single_chunk_when_short():
    text = "Dòng một\nDòng hai kết thúc câu."
    chunks = chunk_text_for_normalize(text)
    assert chunks == ["Dòng một\nDòng hai kết thúc câu."]


def test_chunk_text_for_normalize_splits_long_text_without_losing_lines():
    lines = [f"Tiêu đề {i}" for i in range(50)]
    text = "\n".join(lines)
    chunks = chunk_text_for_normalize(text, limit=300)
    assert len(chunks) > 1
    assert all(len(chunk) <= 300 for chunk in chunks)
    assert sum(chunk.count("\n") + 1 for chunk in chunks) == len(lines)