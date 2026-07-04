from pipeline.data import FIXTURE_PROBLEMS
from pipeline.inference import MockBackend
from pipeline.run_experiment import run_experiment


def test_run_experiment_smoke_with_mock_backend():
    backend = MockBackend(model_name="mock-1.5b")
    traces, results = run_experiment(FIXTURE_PROBLEMS, backend, seed=0)

    assert len(traces) == len(FIXTURE_PROBLEMS)
    # MockBackend always answers "42", which doesn't match any fixture's
    # real gold answer, so nothing should be marked solved-correctly and
    # there should be no corruption outcomes to aggregate.
    assert all(not t.solved_correctly for t in traces)
    assert results == []


def test_run_experiment_produces_results_when_mock_matches_gold():
    from pipeline.data import Problem

    problems = [Problem(problem_id="p0", question="What is the answer?", answer="42")]
    backend = MockBackend(model_name="mock-1.5b")
    traces, results = run_experiment(problems, backend, seed=0)

    assert traces[0].solved_correctly
    # MockBackend's continue_from_corrupted_cot always returns "42" too,
    # so faithfulness rate should be 0 for every corruption method (a
    # deliberately degenerate "fully unfaithful" mock behavior). Results
    # are aggregated per corruption *method* (truncate/substitute_step/
    # remove), and the standard set applies 3 truncations at different
    # fractions -- all grouped under the single "truncate" method key.
    assert len(results) == 3  # truncate, substitute_step, remove
    methods = {r.corruption_method for r in results}
    assert methods == {"truncate", "substitute_step", "remove"}
    truncate_result = next(r for r in results if r.corruption_method == "truncate")
    assert truncate_result.n_problems == 3  # 3 truncation fractions, 1 problem
    for r in results:
        assert r.rate == 0.0
