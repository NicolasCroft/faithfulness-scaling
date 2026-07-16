"""Model backend abstraction.

We need to run inference against DeepSeek-R1-Distill-Qwen-{1.5B, 7B, 14B}
for the black-box truncation test.

**2026-07-16 update (Session 7): resolved which backend to use.** Nick asked
for a completely free option and for this to require no API key at all where
possible (see NEEDS_YOUR_INPUT.md for the full decision writeup). Two options
were evaluated:

  - Hosted "free" API tiers (OpenRouter's `:free` model variants exist for
    the 1.5B/14B/32B sizes, unclear for 7B) are $0 but rate-limited to ~50
    requests/day without ever having spent $10. This project needs on the
    order of 1,000+ generation calls per model size (a few hundred problems
    x multiple corruptions each), so the free-tier *rate limit*, not price,
    is the actual blocker -- it would take weeks per model size.
  - Running the model's own weights directly on a free Colab GPU (T4, 16GB)
    has no per-request limit at all (only a weekly GPU-hour cap, ~15-30
    hrs/week on the free tier, far more than one run needs), needs no API
    key or provider signup, and was already flagged as viable in
    project_overview.md's compute plan for 1.5B/7B, with 14B feasible via
    4-bit quantization.

  Chose the Colab route: `LocalHFBackend` below, driven from
  `notebooks/colab_run_experiment.ipynb`. `HostedAPIBackend` is kept as a
  secondary/fallback option (e.g. if Colab GPU quota runs out) but is not
  the primary path.

This module defines the interface all backends must implement, plus:
  - MockBackend: a deterministic fake backend for unit tests / pipeline
    smoke-testing, with no network calls and no cost.
  - LocalHFBackend: runs a real DeepSeek-R1-Distill-Qwen model locally via
    `transformers` (+ `bitsandbytes` for 4-bit quantization). This is the
    backend the Colab notebook uses. It has no network dependency beyond
    downloading the model weights once, and needs no API key.
  - HostedAPIBackend: a stub for a real hosted-API backend (OpenRouter,
    Together, etc.), kept as a fallback. Filling in `_call_api` is the only
    work needed if this path is used instead; everything else in the
    pipeline (corruptions.py, scoring.py, data.py) is provider-agnostic and
    doesn't need to change either way.
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


class LocalHFBackend(ModelBackend):
    """Runs a real DeepSeek-R1-Distill-Qwen model locally via `transformers`,
    intended for use on a GPU runtime (e.g. free Colab T4) rather than in
    this sandbox. This is the primary backend for the project's real runs --
    see the module docstring and NEEDS_YOUR_INPUT.md for why this was chosen
    over a hosted API.

    Deliberately has no import-time dependency on `torch`/`transformers`/
    `bitsandbytes` so that this module can still be imported (and its other
    classes unit-tested) in environments -- like this one -- that don't have
    those packages installed. The real libraries are only imported inside
    `_load` and `_generate_raw`, which only run when this backend is
    actually used for real inference.
    """

    def __init__(
        self,
        model_name: str,
        load_in_4bit: bool = False,
        device_map: str = "auto",
        dtype: str = "auto",
    ):
        """
        Args:
            model_name: a Hugging Face repo id, e.g.
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B".
            load_in_4bit: use bitsandbytes 4-bit quantization. Recommended
                for the 14B model on a 16GB GPU (full weights are ~9GB at
                4-bit vs. ~28GB at fp16, which won't fit a free-tier T4).
                Not needed for 1.5B/7B on a T4.
            device_map: passed through to `from_pretrained`; "auto" lets
                `accelerate` place the model on the available GPU.
            dtype: torch dtype to load in when not using 4-bit
                quantization ("auto" picks the checkpoint's native dtype,
                typically bfloat16).
        """
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map
        self.dtype = dtype
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise RuntimeError(
                "LocalHFBackend requires `torch` and `transformers` to be "
                "installed. This backend is meant to run on a GPU runtime "
                "(e.g. the Colab notebook at "
                "notebooks/colab_run_experiment.ipynb), not in a plain "
                "CPU-only environment. Original import error: "
                f"{e}"
            ) from e

        quantization_config = None
        torch_dtype = None
        if self.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as e:
                raise RuntimeError(
                    "load_in_4bit=True requires `bitsandbytes` to be "
                    f"installed. Original import error: {e}"
                ) from e
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        else:
            torch_dtype = "auto" if self.dtype == "auto" else getattr(torch, self.dtype)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device_map,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
        )

    def _generate_raw(self, prompt: str, max_tokens: int) -> str:
        self._load()
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
        )
        full_text = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # The decoded text includes the prompt; return only the continuation.
        return full_text[len(self._tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]

    def generate(self, prompt: str, max_tokens: int = 2048) -> GenerationResult:
        raw = self._generate_raw(prompt, max_tokens)
        answer = extract_final_answer(raw)
        cot = raw.rsplit(answer, 1)[0].strip() if answer else raw.strip()
        return GenerationResult(cot=cot, final_answer=answer, raw_text=raw)

    def continue_from_corrupted_cot(
        self, prompt: str, corrupted_cot: str, max_tokens: int = 512
    ) -> GenerationResult:
        full_prompt = f"{prompt}\n\n{corrupted_cot}"
        raw = self._generate_raw(full_prompt, max_tokens)
        answer = extract_final_answer(raw)
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
