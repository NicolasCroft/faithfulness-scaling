import sys
import types

import pytest

from pipeline.inference import (
    GenerationResult,
    HostedAPIBackend,
    LocalHFBackend,
    MockBackend,
    extract_final_answer,
)


# ---------------------------------------------------------------------------
# extract_final_answer
# ---------------------------------------------------------------------------


def test_extract_final_answer_prefers_explicit_phrase():
    assert extract_final_answer("some reasoning. The final answer is 17.") == "17"


def test_extract_final_answer_gsm8k_style_marker():
    assert extract_final_answer("steps...\n#### 1,200") == "1200"


def test_extract_final_answer_boxed():
    assert extract_final_answer(r"work work \boxed{42}") == "42"


def test_extract_final_answer_falls_back_to_last_number():
    assert extract_final_answer("we had 3 apples and 4 oranges, total 7") == "7"


def test_extract_final_answer_empty_when_no_numbers():
    assert extract_final_answer("no numbers here at all") == ""


# ---------------------------------------------------------------------------
# MockBackend (existing behavior, pinned so future edits don't silently
# change the deterministic fixture used by test_run_experiment.py)
# ---------------------------------------------------------------------------


def test_mock_backend_generate_is_deterministic():
    backend = MockBackend()
    r1 = backend.generate("any prompt")
    r2 = backend.generate("a totally different prompt")
    assert r1 == r2
    assert r1.final_answer == "42"


def test_mock_backend_continue_from_corrupted_cot_ignores_the_cot():
    backend = MockBackend()
    result = backend.continue_from_corrupted_cot("prompt", "some corrupted reasoning")
    assert result.final_answer == "42"
    assert result.cot == "some corrupted reasoning"


# ---------------------------------------------------------------------------
# HostedAPIBackend (fallback path) -- error paths that were previously
# untested
# ---------------------------------------------------------------------------


def test_hosted_api_backend_raises_clear_error_without_key(monkeypatch):
    monkeypatch.delenv("INFERENCE_API_KEY", raising=False)
    backend = HostedAPIBackend(model_name="deepseek/deepseek-r1-distill-qwen-14b:free")
    with pytest.raises(RuntimeError, match="No API key found"):
        backend.generate("solve this")


def test_hosted_api_backend_call_api_is_still_a_stub_with_key(monkeypatch):
    monkeypatch.setenv("INFERENCE_API_KEY", "fake-key-for-test")
    backend = HostedAPIBackend(model_name="deepseek/deepseek-r1-distill-qwen-14b:free")
    with pytest.raises(NotImplementedError):
        backend.generate("solve this")


# ---------------------------------------------------------------------------
# LocalHFBackend -- the new primary backend (Session 7). Mocks `torch` and
# `transformers` so this is testable without either package installed (this
# sandbox has neither; the real thing only ever runs on a GPU runtime like
# the Colab notebook).
# ---------------------------------------------------------------------------


class _FakeTensor:
    """Stands in for a torch tensor just well enough for LocalHFBackend's
    usage: `.to(device)` and indexing."""

    def __init__(self, value):
        self.value = value

    def to(self, device):
        return self

    def __getitem__(self, idx):
        return self.value[idx]


class _FakeBatchEncoding(dict):
    def to(self, device):
        return self


def _install_fake_transformers(monkeypatch, generated_continuation="The final answer is 9."):
    """Installs fake `torch` and `transformers` modules into sys.modules so
    LocalHFBackend can be exercised without the real (heavyweight, GPU-
    oriented) dependencies. Returns a dict of call-recording state.
    """
    calls: dict = {}

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, model_name):
            calls["tokenizer_model_name"] = model_name
            return cls()

        def __call__(self, prompt, return_tensors=None):
            calls["prompt"] = prompt
            return _FakeBatchEncoding(input_ids=_FakeTensor([[1, 2, 3]]))

        def decode(self, ids, skip_special_tokens=True):
            # First call decodes the prompt's own input_ids (to compute the
            # continuation-only slice); second call decodes the full output.
            if ids == [1, 2, 3]:
                return "PROMPT_TEXT"
            return "PROMPT_TEXT" + generated_continuation

    class FakeModel:
        device = "cpu"

        @classmethod
        def from_pretrained(cls, model_name, device_map=None, quantization_config=None, torch_dtype=None):
            calls["model_model_name"] = model_name
            calls["device_map"] = device_map
            calls["quantization_config"] = quantization_config
            calls["torch_dtype"] = torch_dtype
            return cls()

        def generate(self, **kwargs):
            calls["generate_kwargs"] = kwargs
            return [[1, 2, 3, 4, 5]]

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs):
            calls["bnb_config_kwargs"] = kwargs

    fake_transformers = types.SimpleNamespace(
        AutoModelForCausalLM=FakeModel,
        AutoTokenizer=FakeTokenizer,
        BitsAndBytesConfig=FakeBitsAndBytesConfig,
    )
    fake_torch = types.SimpleNamespace(bfloat16="bfloat16-sentinel", float16="float16-sentinel")

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return calls


def test_local_hf_backend_raises_clear_error_when_deps_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    backend = LocalHFBackend(model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    with pytest.raises(RuntimeError, match="requires `torch` and `transformers`"):
        backend.generate("solve this")


def test_local_hf_backend_generate_returns_continuation_only(monkeypatch):
    calls = _install_fake_transformers(monkeypatch, generated_continuation="The final answer is 9.")
    backend = LocalHFBackend(model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

    result = backend.generate("Solve: what is 4+5?")

    assert isinstance(result, GenerationResult)
    assert result.final_answer == "9"
    assert "final answer is 9" in result.raw_text.lower()
    assert calls["tokenizer_model_name"] == "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    assert calls["model_model_name"] == "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    # Not requesting 4-bit, so no quantization config should be built.
    assert calls["quantization_config"] is None


def test_local_hf_backend_loads_only_once_across_calls(monkeypatch):
    _install_fake_transformers(monkeypatch)
    backend = LocalHFBackend(model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

    backend.generate("first prompt")
    model_after_first_call = backend._model
    backend.generate("second prompt")

    # _load() is a no-op once self._model is set -- confirms the model
    # weights aren't reloaded from disk on every generate() call.
    assert backend._model is model_after_first_call


def test_local_hf_backend_4bit_builds_quantization_config(monkeypatch):
    calls = _install_fake_transformers(monkeypatch)
    backend = LocalHFBackend(
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", load_in_4bit=True
    )

    backend.generate("solve this")

    assert calls["quantization_config"] is not None
    assert calls["bnb_config_kwargs"]["load_in_4bit"] is True
    assert calls["bnb_config_kwargs"]["bnb_4bit_quant_type"] == "nf4"


def test_local_hf_backend_continue_from_corrupted_cot(monkeypatch):
    _install_fake_transformers(monkeypatch, generated_continuation="#### 12")
    backend = LocalHFBackend(model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

    result = backend.continue_from_corrupted_cot("prompt text", "corrupted reasoning so far")

    assert result.cot == "corrupted reasoning so far"
    assert result.final_answer == "12"
