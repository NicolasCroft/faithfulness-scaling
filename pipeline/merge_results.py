"""Merging partial trace files from multiple Colab sessions (and/or
multiple Google accounts).

**Why this exists:** project_overview.md's compute plan assumed one Colab
session could run a full model size's worth of problems (aim: a few hundred
solved correctly) in one sitting. In practice, free-tier Colab credits can
run out partway through a run. Nick's fix (2026-07-20): spread the work
across several Google accounts' free credits, resuming each new session
from wherever the previous one stopped. That means the traces for one model
size can arrive as several partial JSON files -- possibly downloaded from
different accounts, possibly overlapping if two sessions redid the same
problem by mistake -- instead of one clean file.

This module turns any number of those partial files into one deduplicated
set of traces and recomputes the aggregate `FaithfulnessResult` objects from
the union, using the same aggregation math as
`pipeline.run_experiment.run_experiment` (grouping corruption outcomes by
method, across every correctly-solved problem). It operates purely on the
JSON *shape* the notebook writes (dicts), not on the live
`ProblemTrace`/`Corruption` dataclasses, so it can run on data reloaded from
disk without a model backend, a GPU, or even `torch`/`transformers`
installed -- this module has zero non-stdlib dependencies beyond
`pipeline.scoring`.

File shapes this understands (both produced by
notebooks/colab_run_experiment.ipynb):
  - A bare JSON list of trace dicts (the original "traces" file shape).
  - A dict with a "traces" key holding that list, plus metadata alongside
    it (the checkpoint/cumulative file shape, which also records things
    like model_size and how many problems were targeted).

A trace dict is expected to have at least:
    {"problem_id": str, "solved_correctly": bool,
     "corruption_outcomes": [{"method": str, "answer_changed": bool}, ...]}
Problems that weren't solved correctly have solved_correctly=False and are
kept (for the record / progress counts) but excluded from aggregation, same
as in run_experiment.run_experiment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pipeline.scoring import FaithfulnessResult


def extract_traces(payload: Any) -> list[dict]:
    """Pull the flat list of trace dicts out of a parsed JSON payload,
    regardless of whether it's a bare list or a {"traces": [...]} wrapper.
    """
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if "traces" in payload:
            return payload["traces"]
        raise ValueError(
            "Dict payload has no 'traces' key -- this looks like a "
            "summary-only file (already-aggregated results, not "
            "per-problem traces). merge_results needs the per-problem "
            "traces file to recompute aggregates correctly; point it at "
            "that file instead."
        )
    raise ValueError(f"Unrecognized trace file shape: {type(payload).__name__}")


def load_traces_file(path: str | Path) -> list[dict]:
    """Load one raw/checkpoint JSON file and return its list of trace
    dicts."""
    with open(path) as f:
        payload = json.load(f)
    return extract_traces(payload)


class MergeConflict:
    """Records that the same problem_id appeared in more than one source
    file with disagreeing recorded model_answers.

    This shouldn't happen if the same deterministic (greedy, seeded) model
    run produced both copies -- the normal case is that a later session's
    resume step already skipped problem_ids present in earlier files, so a
    genuine repeat means the same problem was independently redone (e.g.
    two accounts overlapped by mistake, or a resume upload was skipped).
    A conflict doesn't block the merge -- the first occurrence is kept,
    and the problem is counted once either way -- but it's surfaced rather
    than silently swallowed, since it can also indicate a real bug (e.g.
    sampling wasn't actually deterministic, or files from two different
    model sizes got merged together).
    """

    def __init__(self, problem_id: str, sources: list[dict]):
        self.problem_id = problem_id
        self.sources = sources

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        answers = [s.get("model_answer") for s in self.sources]
        return f"MergeConflict(problem_id={self.problem_id!r}, model_answers={answers!r})"


def merge_trace_dicts(trace_lists: list[list[dict]]) -> tuple[list[dict], list[MergeConflict]]:
    """Union trace dicts across files, deduplicated by problem_id.

    Args:
        trace_lists: one list of trace dicts per source file, in the order
            they should be preferred (first occurrence of a problem_id
            wins the dedup).

    Returns:
        (merged_traces, conflicts) -- merged_traces has one entry per
        distinct problem_id; conflicts lists any problem_id that appeared
        more than once with a different model_answer recorded.
    """
    merged: dict[str, dict] = {}
    seen_by_id: dict[str, list[dict]] = {}
    for traces in trace_lists:
        for t in traces:
            pid = t.get("problem_id")
            if pid is None:
                raise ValueError(f"Trace dict missing 'problem_id': {t!r}")
            seen_by_id.setdefault(pid, []).append(t)
            if pid not in merged:
                merged[pid] = t

    conflicts = [
        MergeConflict(pid, sources)
        for pid, sources in seen_by_id.items()
        if len(sources) > 1 and len({s.get("model_answer") for s in sources}) > 1
    ]
    return list(merged.values()), conflicts


def aggregate(traces: list[dict], model_name: str) -> list[FaithfulnessResult]:
    """Recompute per-corruption-method FaithfulnessResult objects from a
    flat list of trace dicts.

    This is the same aggregation `pipeline.run_experiment.run_experiment`
    does on live ProblemTrace objects (group answer-changed outcomes by
    corruption method, across every correctly-solved problem), but on the
    JSON shape instead -- so a merged multi-session result is numerically
    identical to what one uninterrupted session covering the same problems
    would have produced.
    """
    solved = [t for t in traces if t.get("solved_correctly")]
    by_method: dict[str, list[bool]] = {}
    for t in solved:
        for outcome in t.get("corruption_outcomes", []):
            by_method.setdefault(outcome["method"], []).append(bool(outcome["answer_changed"]))

    return [
        FaithfulnessResult(
            model_name=model_name,
            corruption_method=method,
            n_problems=len(changes),
            n_answer_changed=sum(changes),
        )
        for method, changes in by_method.items()
    ]


def merge_files(
    paths: list[str | Path], model_name: str
) -> tuple[list[dict], list[FaithfulnessResult], list[MergeConflict]]:
    """Convenience wrapper: load every file, merge, and aggregate in one
    call. Returns (merged_traces, results, conflicts)."""
    trace_lists = [load_traces_file(p) for p in paths]
    merged_traces, conflicts = merge_trace_dicts(trace_lists)
    results = aggregate(merged_traces, model_name)
    return merged_traces, results, conflicts


def progress_report(merged_traces: list[dict], target_n_problems: int) -> str:
    """Human-readable one-liner: how many problems have been attempted /
    solved correctly so far across all merged sessions, vs. the target
    (project_overview.md aims for "at least a few hundred correctly-solved
    problems per size" so the trend isn't noise) -- useful for deciding
    whether another Colab session (on this account or the next free one)
    is needed before trusting a model size's result.
    """
    n_attempted = len(merged_traces)
    n_solved = sum(1 for t in merged_traces if t.get("solved_correctly"))
    pct = 100 * n_attempted / target_n_problems if target_n_problems else float("nan")
    return (
        f"{n_attempted}/{target_n_problems} problems attempted ({pct:.0f}%), "
        f"{n_solved} solved correctly with full CoT so far across all merged sessions."
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Merge two or more partial trace JSON files from separate "
            "Colab sessions (e.g. across different Google accounts, "
            "resumed after free-tier credits ran out) into one combined "
            "traces file, and recompute the aggregate faithfulness "
            "summary from the union."
        )
    )
    parser.add_argument(
        "files", nargs="+", help="paths to trace/checkpoint JSON files, all for the SAME model size"
    )
    parser.add_argument(
        "--model-name", required=True, help='e.g. "deepseek-r1-distill-qwen-1.5b"'
    )
    parser.add_argument("--out", required=True, help="output path for the merged traces JSON")
    parser.add_argument(
        "--target-n-problems", type=int, default=None, help="for a progress report, e.g. 300"
    )
    args = parser.parse_args()

    merged_traces, results, conflicts = merge_files(args.files, args.model_name)

    if conflicts:
        print(f"WARNING: {len(conflicts)} problem_id(s) had disagreeing model_answer across files:")
        for c in conflicts:
            print(f"  {c}")

    with open(args.out, "w") as f:
        json.dump(merged_traces, f, indent=2)
    print(f"Wrote {len(merged_traces)} merged traces to {args.out}")

    from pipeline.scoring import summarize

    print(summarize(results))

    if args.target_n_problems:
        print(progress_report(merged_traces, args.target_n_problems))
