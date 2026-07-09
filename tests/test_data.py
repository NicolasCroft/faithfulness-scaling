import sys
import types

import pytest

from pipeline.data import FIXTURE_PROBLEMS, Problem, _extract_gsm8k_answer, grade, load_gsm8k


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


def _install_fake_datasets(monkeypatch, rows, expected_repo_id="openai/gsm8k"):
    """Stand in for the `datasets` package so load_gsm8k can be tested
    without network access or the real dependency behaving unexpectedly.
    Records the repo id it was called with so tests can assert on it.
    """
    calls: dict = {}

    def fake_load_dataset(repo_id, config, split=None):
        calls["repo_id"] = repo_id
        calls["config"] = config
        calls["split"] = split
        return rows

    fake_module = types.SimpleNamespace(load_dataset=fake_load_dataset)
    monkeypatch.setitem(sys.modules, "datasets", fake_module)
    return calls


def test_load_gsm8k_uses_current_canonical_repo_id(monkeypatch):
    # Regression test for the 2026-07-09 fix: the old bare "gsm8k" repo id
    # is a legacy script-based dataset that newer `datasets` versions may
    # refuse to load without trust_remote_code=True. This pins the repo id
    # load_gsm8k passes to datasets.load_dataset so a future edit can't
    # silently revert to the legacy id.
    rows = [{"question": "Q1?", "answer": "blah\n#### 5"}]
    calls = _install_fake_datasets(monkeypatch, rows)

    load_gsm8k(split="test")

    assert calls["repo_id"] == "openai/gsm8k"
    assert calls["config"] == "main"
    assert calls["split"] == "test"


def test_load_gsm8k_converts_rows_to_problems(monkeypatch):
    rows = [
        {"question": "Q1?", "answer": "step\n#### 5"},
        {"question": "Q2?", "answer": "step\n#### 1,200"},
    ]
    _install_fake_datasets(monkeypatch, rows)

    problems = load_gsm8k(split="test")

    assert problems == [
        Problem(problem_id="gsm8k_test_0", question="Q1?", answer="5"),
        Problem(problem_id="gsm8k_test_1", question="Q2?", answer="1200"),
    ]


def test_load_gsm8k_respects_n_truncation(monkeypatch):
    rows = [{"question": f"Q{i}?", "answer": f"#### {i}"} for i in range(5)]
    _install_fake_datasets(monkeypatch, rows)

    problems = load_gsm8k(split="test", n=2)

    assert len(problems) == 2


def test_load_gsm8k_wraps_download_failure_as_runtime_error(monkeypatch):
    def fake_load_dataset(repo_id, config, split=None):
        raise OSError("no network")

    fake_module = types.SimpleNamespace(load_dataset=fake_load_dataset)
    monkeypatch.setitem(sys.modules, "datasets", fake_module)

    with pytest.raises(RuntimeError, match="Failed to download GSM8K"):
        load_gsm8k(split="test")
