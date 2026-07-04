from pipeline.corruptions import (
    remove,
    split_into_steps,
    standard_corruption_set,
    substitute_step,
    truncate,
)

SAMPLE_COT = (
    "First, Natalia sold 48 clips in April.\n\n"
    "In May, she sold half as many, so 48 / 2 = 24 clips.\n\n"
    "Altogether she sold 48 + 24 = 72 clips."
)


def test_split_into_steps_paragraphs():
    steps = split_into_steps(SAMPLE_COT)
    assert len(steps) == 3
    assert steps[0].startswith("First,")
    assert steps[-1].startswith("Altogether")


def test_split_into_steps_empty():
    assert split_into_steps("") == []
    assert split_into_steps("   ") == []


def test_truncate_keeps_correct_fraction():
    c = truncate(SAMPLE_COT, frac=0.5)
    assert c.method == "truncate"
    assert c.cut_point == 2  # round(3 * 0.5) == 2
    assert "Altogether" not in c.corrupted_cot
    assert "First," in c.corrupted_cot


def test_truncate_zero_and_one():
    c0 = truncate(SAMPLE_COT, frac=0.0)
    assert c0.corrupted_cot == ""
    assert c0.cut_point == 0

    c1 = truncate(SAMPLE_COT, frac=1.0)
    steps = split_into_steps(SAMPLE_COT)
    assert c1.cut_point == len(steps)
    assert c1.corrupted_cot == "\n\n".join(steps)


def test_truncate_rejects_out_of_range_frac():
    import pytest

    with pytest.raises(ValueError):
        truncate(SAMPLE_COT, frac=1.5)
    with pytest.raises(ValueError):
        truncate(SAMPLE_COT, frac=-0.1)


def test_substitute_step_changes_a_number():
    c = substitute_step(SAMPLE_COT, step_index=1, seed=0)
    assert c.method == "substitute_step"
    steps = split_into_steps(SAMPLE_COT)
    assert c.corrupted_cot != SAMPLE_COT
    # Step 0 and step 2 should be untouched.
    new_steps = split_into_steps(c.corrupted_cot)
    assert new_steps[0] == steps[0]
    assert new_steps[2] == steps[2]
    assert new_steps[1] != steps[1]


def test_substitute_step_custom_replacement_fn():
    c = substitute_step(SAMPLE_COT, step_index=0, replacement_fn=lambda s: "WRONG STEP")
    new_steps = split_into_steps(c.corrupted_cot)
    assert new_steps[0] == "WRONG STEP"


def test_substitute_step_random_index_is_interior_when_possible():
    steps = split_into_steps(SAMPLE_COT)
    c = substitute_step(SAMPLE_COT, seed=1)
    assert 0 <= c.cut_point < len(steps)


def test_substitute_step_no_steps():
    c = substitute_step("")
    assert c.corrupted_cot == ""
    assert "no steps" in c.detail


def test_remove_returns_empty_cot():
    c = remove(SAMPLE_COT)
    assert c.method == "remove"
    assert c.corrupted_cot == ""
    assert c.cut_point == 0


def test_standard_corruption_set_has_expected_methods():
    corruptions = standard_corruption_set(SAMPLE_COT, seed=0)
    methods = [c.method for c in corruptions]
    assert methods.count("truncate") == 3
    assert methods.count("substitute_step") == 1
    assert methods.count("remove") == 1
