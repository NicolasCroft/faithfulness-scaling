"""Model backend abstraction.

We need to run inference against DeepSeek-R1-Distill-Qwen-{1.5B, 7B, 14B}
for the black-box truncation test. project_overview.md's compute plan says
this can go through cheaper hosted APIs (Together, Fireworks, Groq, etc.)
rather than renting GPUs, since we only need text in/out, not activations.

No hosted API account has been set up yet -- that's a real-money decision
that needs the user's sign-off (see NEEDS_YOUR_INPUT.md). This module
defines the interface all backends must implement, plus:
  - MockBackend: a deterministic fake backend for unit tests / pipeline
    smoke-testing, with no network calls and no cost.
  - HostedAPIBackend: a stub for a real hosted-API backend. Filling in
    `_call_api` is the only work needed once a provider + API key are
    chosen; everything else in the pipeline (corruptions.py, scoring.py,
    data.py) is provider-agnostic and doesn't need to change.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationResult:
    """Output of one generation call."""

    cot: str  # the reasoning text, not including the final answer line
    final_answer: str  # extracted final answer (e.g. a number)
    raw_text: str  # full raw model output, for debugging/audit


class ModelBackend(ABC):
    """Interface every inference backend must implement."""

    model_name: str

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 2048) -> GenerationResult:
        """Generate a full CoT + final answer from a fresh prompt (used for
        step 1 of the methodology: generate CoT + answer for each problem).
        """
        raise NotImplementedError

    @abstractmethod
    def continue_from_corrupted_cot(
        self, prompt: str, corrupted_cot: str, max_tokens: int = 512
    ) -> GenerationResult:
        """Regenerate an answer given a problem prompt and an already-
        corrupted CoT prefix (used for step 3 of the methodology). The
        corrupted CoT is treated as if the model itself had produced it so
        far, and the model is asked to continue to a final answer.
        """
        raise NotImplementedError


def extract_final_answer(text: str) -> str:
    """Best-effort extraction of a final numeric answer from free model
    output. Looks for common patterns models use ('final answer is X',
    '#### X', boxed{X}), falling back to the last number in the text.
    """
    patterns = [
        r"final answer is[:\s]*\$?(-?[\d,]+(?:\.\d+)?)",
        r"####\s*(-?[\d,]+(?:\.\d+)?)",
        r"\\boxed\{(-?[\d,]+(?:\.\d+)?)\}",
        r"answer:?\s*\$?(-?[\d,]+(?:\.\d+)?)\s*$",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).replace(",", "")

    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return numbers[-1].replace(",", "") if numbers else ""


class MockBackend(ModelBackend):
    """Deterministic fake backend for tests and pipeline smoke-testing.

    Behavior is intentionally simple and rule-based (not an LLM call):
    it "solves" arithmetic word problems it recognizes from the fixture
    set, and otherwise echoes a fixed answer. This lets the rest of the
    pipeline (corruption application, scoring, plotting) be exercised
    end-to-end without any API cost or network access. It is NOT a
    substitute for real model results.
    """

    def __init__(self, model_name: str = "mock-model"):
        self.model_name = model_name

    def generate(self, prompt: str, max_tokens: int = 2048) -> GenerationResult:
        cot = (
            "First, I identify the quantities in the problem.\n\n"
            "Next, I combine them using the operation the problem implies.\n\n"
            "Finally, I compute the result."
        )
        answer = "42"
        raw = cot + f"\n\nThe final answer is {answer}."
        return GenerationResult(cot=cot, final_answer=answer, raw_text=raw)

    def continue_from_corrupted_cot(
        self, prompt: str, corrupted_cot: str, max_tokens: int = 512
    ) -> GenerationResult:
        # Toy "unfaithful" behavior: the mock always returns the same
        # answer regardless of what the (corrupted) CoT says, which is
        # useful for testing that the scoring code correctly reports a
        # faithfulness rate of 0 in this degenerate case.
        answer = "42"
        raw = corrupted_cot + f"\n\nThe final answer is {answer}."
        return GenerationResult(cot=corrupted_cot, final_answer=answer, raw_text=raw)


class HostedAPIBackend(ModelBackend):
    """Stub for a real hosted inference API (Together AI / Fireworks /
    Groq / OpenRouter -- whichever is chosen, see NEEDS_YOUR_INPUT.md).

    Only `_call_api` needs a real implementation once a provider and API
    key are available. Everything else (prompt construction, answer
    extraction) is already wired up.
    """

    def __init__(self, model_name: str, api_key_env_var: str = "INFERENCE_API_KEY"):
        self.model_name = model_name
        self.api_key_env_var = api_key_env_var

    def _require_api_key(self) -> str:
        key = os.environ.get(self.api_key_env_var)
        if not key:
            raise RuntimeError(
                f"No API key found in environment variable "
                f"{self.api_key_env_var!r}. Hosted-API inference requires "
                f"signing up with a provider (Together/Fireworks/Groq/"
                f"OpenRouter) and setting this env var -- see "
                f"NEEDS_YOUR_INPUT.md for the pending decision on which "
                f"provider to use."
            )
        return key

    def _call_api(self, prompt: str, max_tokens: int) -> str:
        """Placeholder for the actual HTTP call to the chosen provider.
        Not implemented until a provider is selected (see
        NEEDS_YOUR_INPUT.md)."""
        self._require_api_key()
        raise NotImplementedError(
            "HostedAPIBackend._call_api is a stub. Implement the HTTP call "
            "for the chosen provider once NEEDS_YOUR_INPUT.md's open "
            "question about which hosted API to use is resolved."
        )

    def generate(self, prompt: str, max_tokens: int = 2048) -> GenerationResult:
        raw = self._call_api(prompt, max_tokens)
        answer = extract_final_answer(raw)
        # Naive CoT/answer split: everything before the final-answer
        # sentence is treated as CoT. Providers/models vary in formatting,
        # so this may need per-model tuning once real output is seen.
        cot = raw.rsplit(answer, 1)[0].strip() if answer else raw.strip()
        return GenerationResult(cot=cot, final_answer=answer, raw_text=raw)

    def continue_from_corrupted_cot(
        self, prompt: str, corrupted_cot: str, max_tokens: int = 512
    ) -> GenerationResult:
        full_prompt = f"{prompt}\n\n{corrupted_cot}"
        raw = self._call_api(full_prompt, max_tokens)
        answer = extract_final_answer(raw)
        return GenerationResult(cot=corrupted_cot, final_answer=answer, raw_text=raw)
