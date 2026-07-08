"""Activation patching (causal tracing) -- the "optional deeper layer" from
project_overview.md, to be run on whichever model size(s) show the most
interesting effect in the black-box truncation test.

This is a scaffold, not a finished implementation. Unlike inference.py's
ModelBackend (text in, text out -- works fine against a cheap hosted API),
real activation patching needs access to the model's internal activations.
project_overview.md's compute plan is explicit about this: "real
activation-level work requires running the actual model weights yourself
via a library like TransformerLens or nnsight." That means:
  - it cannot run through Together/Fireworks/Groq/etc. (none of them expose
    internal activations), and
  - it cannot run inside this scheduled-task sandbox either, which has no
    GPU and (per PROGRESS_LOG.md sessions 1/3/4/5) no route to download
    model weights or reach a hosted API from here.

So `ActivationPatchingBackend` below is an interface only. A real
implementation (backed by TransformerLens or nnsight, running on a rented or
local GPU) is a separate, later piece of work that should only be built once
the core truncation-test result across 1.5B/7B/14B picks out a size worth
digging into further -- building it earlier would be guessing at an
interface before we know what's worth localizing. `MockActivationPatchingBackend`
exists so the localization logic itself (`localize_faithfulness`) can be
written, tested, and reviewed now, without blocking on GPU access.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class PatchResult:
    """Outcome of patching one layer/component during one causal-tracing run.

    Attributes:
        layer: index of the layer patched (0-indexed)
        component: name of the internal component patched (e.g.
            "resid_post", "attn_out", "mlp_out" -- exact vocabulary depends
            on the backend/library used)
        baseline_answer: the answer produced by the unpatched (corrupted)
            run, for reference
        patched_answer: the answer produced after patching in the clean
            run's activation at this layer/component
        answer_flipped: whether patching changed the answer relative to
            baseline_answer -- the signal that this layer/component is
            causally implicated
    """

    layer: int
    component: str
    baseline_answer: str
    patched_answer: str
    answer_flipped: bool


class ActivationPatchingBackend(ABC):
    """Interface a backend must implement to support layer-by-layer causal
    tracing between a "clean" run and a "corrupted" run of the same model.

    Implementations need real model weights loaded locally (e.g. via
    TransformerLens or nnsight), not just an inference API. See this
    module's docstring for why.
    """

    model_name: str
    n_layers: int

    @abstractmethod
    def generate_with_cache(self, prompt: str, max_tokens: int = 64) -> tuple[str, Any]:
        """Run the model forward on `prompt`, returning the generated answer
        text and an opaque activation cache (backend-specific object) that
        can later be passed to `generate_with_patch` as the *source* of a
        patch.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_with_patch(
        self,
        prompt: str,
        layer: int,
        component: str,
        patch_cache: Any,
        max_tokens: int = 64,
    ) -> str:
        """Run the model forward on `prompt`, but with the activation at
        `layer`/`component` overwritten with the corresponding activation
        from `patch_cache` (as produced by a prior `generate_with_cache`
        call on a *different* prompt). Returns the generated answer text
        under that intervention.
        """
        raise NotImplementedError


class MockActivationPatchingBackend(ActivationPatchingBackend):
    """Deterministic fake backend for wiring and unit tests. No real model
    weights, no GPU, no cost -- mirrors the role `MockBackend` plays in
    inference.py. Lets `localize_faithfulness` be exercised end-to-end
    before a real TransformerLens/nnsight backend exists.

    Toy behavior: `generate_with_cache` always returns a fixed "clean"
    answer for any prompt containing "clean" and a fixed "corrupted" answer
    otherwise. `generate_with_patch` only flips the answer back to the clean
    one when patching at `flip_at_layer` (if set); every other layer is a
    no-op. This gives tests/callers a single, known "causally responsible"
    layer to check for, analogous to how MockBackend's fixed "42" answer
    gives the truncation-test code a known degenerate case to check for.
    """

    def __init__(
        self,
        model_name: str = "mock-activation-model",
        n_layers: int = 4,
        flip_at_layer: int | None = 2,
        clean_answer: str = "72",
        corrupted_answer: str = "42",
    ):
        self.model_name = model_name
        self.n_layers = n_layers
        self.flip_at_layer = flip_at_layer
        self.clean_answer = clean_answer
        self.corrupted_answer = corrupted_answer

    def generate_with_cache(self, prompt: str, max_tokens: int = 64) -> tuple[str, Any]:
        answer = self.clean_answer if "clean" in prompt else self.corrupted_answer
        return answer, {"prompt": prompt, "answer": answer}

    def generate_with_patch(
        self,
        prompt: str,
        layer: int,
        component: str,
        patch_cache: Any,
        max_tokens: int = 64,
    ) -> str:
        if self.flip_at_layer is not None and layer == self.flip_at_layer:
            return patch_cache["answer"]
        return self.corrupted_answer


def localize_faithfulness(
    backend: ActivationPatchingBackend,
    clean_prompt: str,
    corrupted_prompt: str,
    components: Sequence[str] = ("resid_post",),
    max_tokens: int = 64,
) -> list[PatchResult]:
    """Layer-by-layer causal tracing between a clean and a corrupted run of
    the same problem, to localize which layer(s) causally drive the
    faithfulness effect the black-box truncation test already established
    exists (this function doesn't establish *that* an effect exists -- it
    only localizes one that run_experiment.py's black-box test already
    flagged as interesting).

    Procedure:
        1. Run `clean_prompt` (the original, uncorrupted problem/CoT),
           caching its activations.
        2. Run `corrupted_prompt` (e.g. built from one of corruptions.py's
           corrupted CoTs) to get the baseline corrupted-run answer.
        3. For each layer x component, patch the clean run's cached
           activation into the corrupted run and re-generate. If the
           patched answer reverts to the clean answer, that layer/component
           is causally implicated in the effect.

    Returns:
        One PatchResult per (layer, component) pair tested.
    """
    _clean_answer, clean_cache = backend.generate_with_cache(clean_prompt, max_tokens)
    corrupted_answer, _corrupted_cache = backend.generate_with_cache(corrupted_prompt, max_tokens)

    results: list[PatchResult] = []
    for layer in range(backend.n_layers):
        for component in components:
            patched_answer = backend.generate_with_patch(
                corrupted_prompt, layer, component, clean_cache, max_tokens
            )
            results.append(
                PatchResult(
                    layer=layer,
                    component=component,
                    baseline_answer=corrupted_answer,
                    patched_answer=patched_answer,
                    answer_flipped=(patched_answer != corrupted_answer),
                )
            )
    return results


def most_causal_layers(results: list[PatchResult]) -> list[PatchResult]:
    """Filter to the (layer, component) pairs where patching flipped the
    answer -- i.e. the ones worth reporting as "causally responsible" for
    the observed effect."""
    return [r for r in results if r.answer_flipped]
