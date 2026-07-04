import pytest

from pipeline.data import FIXTURE_PROBLEMS, _extract_gsm8k_answer, grade


def test_fixture_problems_have_answers():
    assert len(FIXTURE_PROBLEMS) == 3
    for p in FIXTURE_PROBLEMS:
        assert p.answer.isdigit() or (p.answer.startswith("-") and p.answer[1:].isdigit())


def test_extract_gsm8k_answer_basic():
    raw = "She sold 48 clips.\n#### 48"
    assert _extract_gsm8k_answer(raw) == "48"


def test_extract_gsm8k_answer_strips_commas():
    raw = "Total is a lot.\n#### 1,200"
    assert _extract_gsm8k_answer(raw) == "1200"


def test_extract_gsm8k_answer_missing_marker_raises():
    with pytest.raises(ValueError):
        _extract_gsm8k_answer("no marker here")


def test_grade_exact_match():
    assert grade("72", "72")


def test_grade_normalizes_commas_and_dollar_and_trailing_zero():
    assert grade("$1,200.0", "1200")
    assert grade(" 72 ", "72")


def test_grade_mismatch():
    assert not grade("71", "72")
