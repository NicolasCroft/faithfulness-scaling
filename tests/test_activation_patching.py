from pipeline.activation_patching import (
    MockActivationPatchingBackend,
    localize_faithfulness,
    most_causal_layers,
)


def test_generate_with_cache_distinguishes_clean_and_corrupted_prompts():
    backend = MockActivationPatchingBackend()
    clean_answer, clean_cache = backend.generate_with_cache("clean version of the problem")
    corrupted_answer, _ = backend.generate_with_cache("corrupted version of the problem")

    assert clean_answer == backend.clean_answer
    assert corrupted_answer == backend.corrupted_answer
    assert clean_cache["answer"] == backend.clean_answer


def test_generate_with_patch_only_flips_at_designated_layer():
    backend = MockActivationPatchingBackend(n_layers=4, flip_at_layer=2)
    _clean_answer, clean_cache = backend.generate_with_cache("clean prompt")

    unpatched_layer = backend.generate_with_patch("corrupted prompt", 0, "resid_post", clean_cache)
    patched_layer = backend.generate_with_patch("corrupted prompt", 2, "resid_post", clean_cache)

    assert unpatched_layer == backend.corrupted_answer
    assert patched_layer == backend.clean_answer


def test_localize_faithfulness_returns_one_result_per_layer_and_component():
    backend = MockActivationPatchingBackend(n_layers=4, flip_at_layer=2)
    results = localize_faithfulness(
        backend,
        clean_prompt="clean prompt",
        corrupted_prompt="corrupted prompt",
        components=("resid_post", "attn_out"),
    )
    assert len(results) == 4 * 2
    assert {r.layer for r in results} == {0, 1, 2, 3}
    assert {r.component for r in results} == {"resid_post", "attn_out"}


def test_localize_faithfulness_flags_only_the_causal_layer():
    backend = MockActivationPatchingBackend(n_layers=4, flip_at_layer=2)
    results = localize_faithfulness(backend, "clean prompt", "corrupted prompt")

    flipped = most_causal_layers(results)
    assert len(flipped) == 1
    assert flipped[0].layer == 2
    assert flipped[0].patched_answer == backend.clean_answer
    assert flipped[0].baseline_answer == backend.corrupted_answer


def test_localize_faithfulness_with_no_causal_layer():
    backend = MockActivationPatchingBackend(n_layers=3, flip_at_layer=None)
    results = localize_faithfulness(backend, "clean prompt", "corrupted prompt")

    assert most_causal_layers(results) == []
    assert len(results) == 3
