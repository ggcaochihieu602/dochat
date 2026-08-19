from .helpers import split_for_tts, split_text


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