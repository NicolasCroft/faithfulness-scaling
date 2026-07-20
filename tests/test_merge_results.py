import json
import math

import pytest

from pipeline.merge_results import (
    MergeConflict,
    aggregate,
    extract_traces,
    load_traces_file,
    merge_files,
    merge_trace_dicts,
    progress_report,
)


def _trace(problem_id, solved=True, model_answer="10", outcomes=None):
    return {
        "problem_id": problem_id,
        "question": "irrelevant for these tests",
        "gold_answer": "10",
        "model_answer": model_answer,
        "solved_correctly": solved,
        "corruption_outcomes": outcomes if outcomes is not None else [],
    }


# ---------------------------------------------------------------------------
# extract_traces / load_traces_file
# ---------------------------------------------------------------------------


def test_extract_traces_from_bare_list():
    traces = [_trace("p0")]
    assert extract_traces(traces) == traces


def test_extract_traces_from_wrapped_dict():
    traces = [_trace("p0")]
    payload = {"model_size": "1.5B", "traces": traces}
    assert extract_traces(payload) == traces


def test_extract_traces_rejects_summary_only_dict():
    with pytest.raises(ValueError, match="no 'traces' key"):
        extract_traces({"model_size": "1.5B", "results": []})


def test_extract_traces_rejects_unrecognized_shape():
    with pytest.raises(ValueError, match="Unrecognized trace file shape"):
        extract_traces("not a list or dict")


def test_load_traces_file_reads_bare_list(tmp_path):
    traces = [_trace("p0"), _trace("p1")]
    path = tmp_path / "traces.json"
    path.write_text(json.dumps(traces))
    assert load_traces_file(path) == traces


def test_load_traces_file_reads_wrapped_dict(tmp_path):
    traces = [_trace("p0")]
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps({"traces": traces, "model_size": "7B"}))
    assert load_traces_file(path) == traces


# ---------------------------------------------------------------------------
# merge_trace_dicts
# ---------------------------------------------------------------------------


def test_merge_trace_dicts_unions_disjoint_files():
    file_a = [_trace("p0"), _trace("p1")]
    file_b = [_trace("p2"), _trace("p3")]
    merged, conflicts = merge_trace_dicts([file_a, file_b])
    assert {t["problem_id"] for t in merged} == {"p0", "p1", "p2", "p3"}
    assert conflicts == []


def test_merge_trace_dicts_dedups_matching_repeat_without_conflict():
    # Same problem redone in two sessions with the same (deterministic)
    # answer -- should be counted once, no conflict.
    file_a = [_trace("p0", model_answer="10")]
    file_b = [_trace("p0", model_answer="10"), _trace("p1")]
    merged, conflicts = merge_trace_dicts([file_a, file_b])
    assert sorted(t["problem_id"] for t in merged) == ["p0", "p1"]
    assert conflicts == []


def test_merge_trace_dicts_flags_disagreeing_repeat_as_conflict():
    file_a = [_trace("p0", model_answer="10")]
    file_b = [_trace("p0", model_answer="99")]  # disagrees
    merged, conflicts = merge_trace_dicts([file_a, file_b])
    # First occurrence still wins the dedup...
    assert len(merged) == 1
    assert merged[0]["model_answer"] == "10"
    # ...but the disagreement is surfaced, not silently dropped.
    assert len(conflicts) == 1
    assert isinstance(conflicts[0], MergeConflict)
    assert conflicts[0].problem_id == "p0"


def test_merge_trace_dicts_missing_problem_id_raises():
    bad = [{"solved_correctly": True}]
    with pytest.raises(ValueError, match="missing 'problem_id'"):
        merge_trace_dicts([bad])


def test_merge_trace_dicts_empty_input():
    merged, conflicts = merge_trace_dicts([])
    assert merged == []
    assert conflicts == []


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


def test_aggregate_groups_by_corruption_method_across_solved_problems():
    traces = [
        _trace(
            "p0",
            solved=True,
            outcomes=[
                {"method": "truncate", "new_answer": "10", "answer_changed": False},
                {"method": "remove", "new_answer": "99", "answer_changed": True},
            ],
        ),
        _trace(
            "p1",
            solved=True,
            outcomes=[
                {"method": "truncate", "new_answer": "5", "answer_changed": True},
                {"method": "remove", "new_answer": "10", "answer_changed": False},
            ],
        ),
    ]
    results = aggregate(traces, model_name="deepseek-r1-distill-qwen-1.5b")
    by_method = {r.corruption_method: r for r in results}
    assert set(by_method) == {"truncate", "remove"}
    assert by_method["truncate"].n_problems == 2
    assert by_method["truncate"].n_answer_changed == 1
    assert by_method["truncate"].rate == 0.5
    assert by_method["remove"].n_problems == 2
    assert by_method["remove"].n_answer_changed == 1


def test_aggregate_excludes_unsolved_problems():
    traces = [
        _trace("p0", solved=False, outcomes=[{"method": "truncate", "new_answer": "1", "answer_changed": True}]),
    ]
    results = aggregate(traces, model_name="mock")
    assert results == []


def test_aggregate_empty_traces_returns_empty_results():
    assert aggregate([], model_name="mock") == []


# ---------------------------------------------------------------------------
# merge_files (end-to-end on disk)
# ---------------------------------------------------------------------------


def test_merge_files_end_to_end(tmp_path):
    outcomes_faithful = [{"method": "remove", "new_answer": "99", "answer_changed": True}]
    outcomes_unfaithful = [{"method": "remove", "new_answer": "10", "answer_changed": False}]

    file_a = tmp_path / "session_a.json"
    file_a.write_text(
        json.dumps({"model_size": "1.5B", "traces": [_trace("p0", outcomes=outcomes_faithful)]})
    )
    file_b = tmp_path / "session_b.json"
    file_b.write_text(json.dumps([_trace("p1", outcomes=outcomes_unfaithful)]))

    merged_traces, results, conflicts = merge_files(
        [file_a, file_b], model_name="deepseek-r1-distill-qwen-1.5b"
    )

    assert {t["problem_id"] for t in merged_traces} == {"p0", "p1"}
    assert conflicts == []
    assert len(results) == 1
    assert results[0].corruption_method == "remove"
    assert results[0].n_problems == 2
    assert results[0].n_answer_changed == 1
    assert results[0].rate == 0.5


# ---------------------------------------------------------------------------
# progress_report
# ---------------------------------------------------------------------------


def test_progress_report_counts_attempted_and_solved():
    traces = [_trace("p0", solved=True), _trace("p1", solved=False), _trace("p2", solved=True)]
    report = progress_report(traces, target_n_problems=300)
    assert "3/300 problems attempted" in report
    assert "2 solved correctly" in report


def test_progress_report_zero_target_does_not_raise():
    report = progress_report([], target_n_problems=0)
    assert "0/0" in report
