"""Integration test for the core guarantee behind the Colab
checkpoint/resume workflow (notebooks/colab_run_experiment.ipynb, section
4.5): splitting one model-size run across multiple sessions (simulating
multiple Google accounts, each contributing a partial trace file) and
merging the results must be numerically identical to running the same
problems in one uninterrupted session.

This doesn't execute the notebook itself (that needs a real GPU runtime),
but it exercises the exact same pipeline functions the notebook's cells
call -- `run_for_problem`, the trace-dict shape the notebook builds, and
`pipeline.merge_results.aggregate` -- so a break in that guarantee would
show up here.
"""

from __future__ import annotations

from pipeline.data import Problem
from pipeline.inference import MockBackend
from pipeline.merge_results import aggregate, merge_trace_dicts
from pipeline.run_experiment import run_for_problem


def _make_problems(n: int) -> list[Problem]:
    # answer="42" so MockBackend.generate (which always answers "42")
    # counts these as solved correctly, exercising the corruption/scoring
    # path -- see tests/test_run_experiment.py for why "42" is the magic
    # value.
    return [
        Problem(problem_id=f"gsm8k_test_{i}", question=f"Fake problem {i}?", answer="42")
        for i in range(n)
    ]


def _trace_to_dict(t) -> dict:
    """Mirrors the notebook's `_trace_to_dict` helper (section 5) exactly,
    so this test is checking the same transformation the real notebook
    cell performs."""
    return {
        "problem_id": t.problem.problem_id,
        "question": t.problem.question,
        "gold_answer": t.problem.answer,
        "model_answer": t.original.final_answer,
        "solved_correctly": t.solved_correctly,
        "corruption_outcomes": [
            {"method": c.method, "new_answer": a, "answer_changed": ch}
            for c, a, ch in t.corruption_outcomes
        ],
    }


def _run_traces(problems: list[Problem], seed: int) -> list[dict]:
    backend = MockBackend(model_name="mock-model")
    return [_trace_to_dict(run_for_problem(p, backend, seed=seed)) for p in problems]


def test_two_session_resume_matches_one_uninterrupted_session():
    seed = 0
    all_problems = _make_problems(6)

    # "One uninterrupted session": everything in one go.
    one_session_traces = _run_traces(all_problems, seed=seed)
    one_session_results = aggregate(one_session_traces, model_name="deepseek-r1-distill-qwen-1.5b")

    # "Two sessions, e.g. two different Google accounts": account A does
    # problems 0-2, its credits run out, account B resumes and does the
    # rest -- mirroring the notebook's skip-already-done-problem_ids logic.
    session_a_problems = all_problems[:3]
    session_a_traces = _run_traces(session_a_problems, seed=seed)

    already_done_ids = {t["problem_id"] for t in session_a_traces}
    session_b_problems = [p for p in all_problems if p.problem_id not in already_done_ids]
    assert len(session_b_problems) == 3  # sanity: resume actually skipped session A's work
    session_b_traces = _run_traces(session_b_problems, seed=seed)

    merged_traces, conflicts = merge_trace_dicts([session_a_traces, session_b_traces])
    assert conflicts == []
    merged_results = aggregate(merged_traces, model_name="deepseek-r1-distill-qwen-1.5b")

    # The merged two-session result must be identical to the one-session
    # result: same problems, same aggregate counts and rates.
    assert {t["problem_id"] for t in merged_traces} == {t["problem_id"] for t in one_session_traces}

    def as_dict(results):
        return {r.corruption_method: (r.n_problems, r.n_answer_changed) for r in results}

    assert as_dict(merged_results) == as_dict(one_session_results)


def test_three_way_split_across_accounts_still_matches():
    seed = 0
    all_problems = _make_problems(9)
    one_session_traces = _run_traces(all_problems, seed=seed)
    one_session_results = aggregate(one_session_traces, model_name="mock")

    # Three "accounts", three problems each, run independently and merged.
    shard_traces = [
        _run_traces(all_problems[0:3], seed=seed),
        _run_traces(all_problems[3:6], seed=seed),
        _run_traces(all_problems[6:9], seed=seed),
    ]
    merged_traces, conflicts = merge_trace_dicts(shard_traces)
    assert conflicts == []
    assert len(merged_traces) == 9
    merged_results = aggregate(merged_traces, model_name="mock")

    def as_dict(results):
        return {r.corruption_method: (r.n_problems, r.n_answer_changed) for r in results}

    assert as_dict(merged_results) == as_dict(one_session_results)
