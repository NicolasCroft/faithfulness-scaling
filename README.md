# Faithfulness Scaling in Distilled Reasoning Models

**Status: early build. No real model results yet — pipeline is implemented and
unit-tested against a mock backend. Inference plan is now decided (2026-07-16):
run real models for free on a Colab GPU via `pipeline/inference.py:LocalHFBackend`
and `notebooks/colab_run_experiment.ipynb`, rather than a hosted API — see
`NEEDS_YOUR_INPUT.md` for why, and the notebook for the one manual step left
(a human needs to actually click "Run" in Colab; that can't happen from this
automated sandbox).**

## Motivation

As reasoning models get distilled down to smaller sizes, does the faithfulness
of their chain-of-thought (whether the stated reasoning actually causes the
final answer) change in a predictable way with model size? This project runs
one established faithfulness metric — the truncation/corruption test from
Lanham et al. (2023) — across the DeepSeek-R1-Distill-Qwen family
(1.5B / 7B / 14B, same teacher, same distillation recipe) and reports the
resulting scaling curve.

This is narrower than "does distillation break faithfulness" in general. The
scaling-curve framing — one metric, one distilled family, multiple sizes,
plotted as a trend — doesn't appear to have been published as of this
writing.

## Method

1. Generate a chain-of-thought (CoT) and final answer for each problem in a
   GSM8K subset.
2. Keep only problems the model solves correctly with its full, uncorrupted
   CoT.
3. Apply a standard battery of corruptions to the CoT and regenerate the
   answer from the corrupted context:
   - **Truncation** at 25% / 50% / 75% of reasoning steps
   - **Step substitution**: replace one intermediate step with a
     plausible-but-wrong one
   - **Removal**: delete the reasoning entirely
4. Faithfulness score = the rate at which corrupting the reasoning changes
   the final answer. High score = faithful (answer depends on the stated
   reasoning); low score = unfaithful (answer is invariant to the reasoning).
5. Repeat across all three model sizes, plot score vs. size with 95% Wilson
   confidence intervals, and check for a trend.

## Repo layout

```
pipeline/
  data.py                - GSM8K loading, answer grading (fixture data for offline dev)
  corruptions.py         - truncate / substitute_step / remove corruption methods
  inference.py           - ModelBackend interface; MockBackend (no cost) + LocalHFBackend
                           (real inference via transformers, driven from the Colab
                           notebook below -- the primary path) + HostedAPIBackend
                           (stub, kept as a fallback -- see inference.py docstring)
  scoring.py             - faithfulness rate, Wilson confidence intervals, scaling-curve plot
  run_experiment.py      - orchestrates steps 1-4 above for one model
  activation_patching.py - optional-deeper-layer scaffold: ActivationPatchingBackend
                           interface + MockActivationPatchingBackend + layer-sweep
                           localization logic. No real backend yet (needs
                           TransformerLens/nnsight + GPU, see module docstring)
tests/              - unit tests for all of the above (54 tests, run via pytest)
notebooks/          - analysis_scaffold.ipynb runs the full run-experiment ->
                      table -> scaling-curve-plot path against a synthetic backend
                      (dry run, no real model calls). colab_run_experiment.ipynb
                      is the real thing: open it in Google Colab (free T4 GPU),
                      pick a model size, Run All -- no API key or signup needed.
results/            - output plots and result tables. mock_demo_scaling_curve.png
                      is synthetic (from analysis_scaffold.ipynb), not a real
                      result. results/raw/ is where real Colab-run output JSON
                      files should be dropped once produced.
```

## Status

- [x] Repo scaffold, corruption methods, data loading, scoring (Wilson CIs),
      inference abstraction layer — implemented and unit-tested against a
      mock backend (`pipeline/inference.py:MockBackend`), no real model
      calls yet.
- [x] Full analysis path (run experiment → aggregate → results table →
      scaling-curve plot with Wilson CIs) dry-run end-to-end in
      `notebooks/analysis_scaffold.ipynb` against a synthetic backend, so it
      needs no code changes once real inference is available — just a
      backend + data-source swap (see notebook's last cell).
- [x] Inference plan decided (2026-07-16): free Colab GPU + `LocalHFBackend`
      running real model weights locally, not a hosted API. See
      `NEEDS_YOUR_INPUT.md` for the full reasoning (every hosted provider
      tested, including free-tier OpenRouter, turned out to be either
      unreachable from this sandbox or too rate-limited for the volume of
      calls this experiment needs).
- [x] `pipeline/inference.py:LocalHFBackend` implemented and unit-tested
      against mocked `torch`/`transformers` (14 tests) — logic verified, but
      **not yet run against real GPU hardware or real model weights**, since
      that can't happen from this sandbox.
- [x] `notebooks/colab_run_experiment.ipynb` built: clones this repo, loads
      a chosen model size (1.5B/7B/14B, 4-bit for 14B), runs the real
      truncation/corruption test, saves results as JSON. Needs a human to
      actually open it in Colab and click Run — that's the one remaining
      manual step, logged in `NEEDS_YOUR_INPUT.md`.
- [ ] Truncation-test pipeline validated on one real model size (**next
      concrete step — waiting on the Colab run above**).
- [ ] Full run across 1.5B / 7B / 14B.
- [ ] Scaling-curve plot and write-up of results.
- [x] Optional-deeper-layer scaffold: `pipeline/activation_patching.py`
      (interface + mock backend + layer-sweep localization logic), unit-tested.
      Not yet backed by a real model — see Limitations.
- [ ] Optional: activation-patching deep dive on the most interesting size,
      once a real `ActivationPatchingBackend` exists.

## Literature check (2026-07-09)

Re-checked whether the scaling-curve framing in `project_overview.md` has been
published since project start. Relevant recent work found:

- Cornish & Rogers, "Examining the Faithfulness of Deepseek R1's Chain-of-Thought
  Reasoning" (ACL CHOMPS 2025) — faithfulness analysis of R1 itself, not the
  distilled family or a size sweep.
- Ye et al., "Mechanistic Evidence for Faithfulness Decay in Chain-of-Thought
  Reasoning" (arXiv 2602.11201, Feb 2026) — proposes a causal-corruption metric
  (NLDD) and a "reasoning horizon" finding, tested across three *different*
  model families (Llama, DeepSeek, Gemma) at single sizes each, not multiple
  sizes within one distilled family.
- "Mapping Faithful Reasoning in Language Models" (arXiv 2510.22362, Oct 2025) —
  activation-space faithfulness tracing on Qwen3-4B, single size, different
  method (internal activations, not truncation).
- Chen et al., "Are DeepSeek R1 And Other Reasoning Models More Faithful?"
  (arXiv 2501.08156) — compares faithfulness across different model
  *families*, not sizes within a distillation family.
- **New this check:** "Lie to Me: How Faithful Is Chain-of-Thought Reasoning in
  Open-Weight Reasoning Models?" (arXiv 2603.22582, Mar 2026) — tests 12 open-weight
  reasoning models across 9 architectural families (7B-685B) using *hint-injection*
  faithfulness (Turpin-style: does the model acknowledge a planted hint influenced
  its answer), not the truncation/corruption test this project uses. Its headline
  finding — "training methodology and model family predict faithfulness more
  strongly than parameter count" — is about cross-family comparison at ~1 size per
  family, not a within-family multi-size sweep, so it doesn't overlap with this
  project's specific gap. Worth citing in the write-up as a relevant, differently-
  scoped finding (different metric, different comparison axis), and worth revisiting
  if a follow-up applies its hint-injection method within one distilled family across
  sizes, which would be a closer overlap than anything found so far.

None of these run one faithfulness metric across multiple sizes of the same
distilled family and report the trend — the gap this project targets still
appears open as of this check. Will re-verify periodically since this is an
active research area.

## Limitations (so far)

- GSM8K was chosen over MATH as the initial task domain because its answers
  are single numbers with an unambiguous canonical format, making automatic
  grading trivial. If GSM8K turns out to be too easy for all three model
  sizes (e.g. near-100% accuracy everywhere), there won't be enough
  correctly-solved-but-corruptible problems to get a meaningful faithfulness
  signal, and a MATH subset would be the fallback — see `data.py` docstring.
- Step substitution falls back to a naive numeric perturbation (change one
  number in a step) when no model-backed replacement function is supplied.
  This produces a "wrong" step but not always a *maximally plausible* wrong
  step. `pipeline/corruptions.py:model_backed_replacement` now provides a
  provider-agnostic, unit-tested wiring for a stronger, model-generated
  substitution (any `ModelBackend` can act as the judge) — it just needs a
  real backend behind it once the hosted-API decision below is resolved.
- No real model results exist yet. Everything above has been validated only
  against `MockBackend`, a deterministic stand-in with no relationship to
  actual model behavior — it exists purely to exercise the pipeline's
  plumbing (corruption application, grading, aggregation, CI computation)
  before spending any API budget.
- The activation-patching module (`pipeline/activation_patching.py`) is an
  interface and mock backend only — no real model weights are loaded
  anywhere in this repo yet. Real activation-level work needs
  TransformerLens/nnsight running against actual weights on a GPU, which is
  a separate piece of infrastructure from the hosted-API text-in/text-out
  path used for the core truncation test (see module docstring and
  `project_overview.md`'s compute plan).
- This scheduled-task sandbox has a narrow network allowlist: `pypi.org` and
  GitHub are reachable, but `huggingface.co`, `api.together.xyz`,
  `api.fireworks.ai`, `api.groq.com`, and (confirmed 2026-07-16)
  `openrouter.ai` all fail to connect from the sandbox's proxy. This is why
  the project moved to a Colab-GPU-based plan instead of any hosted API —
  see `NEEDS_YOUR_INPUT.md`'s 2026-07-16 entry for the full reasoning. The
  real inference run itself (via `notebooks/colab_run_experiment.ipynb`)
  still needs a human to execute it in Colab; it can't run unattended from
  this sandbox either, for the same underlying reason.
- `pipeline/inference.py:LocalHFBackend` has only been tested against a
  mocked `torch`/`transformers` — it has never been exercised against real
  model weights or real GPU hardware. It's plausible (not confirmed) that
  something about the real libraries' behavior differs from what the mocks
  assume once it's actually run in Colab.

## Next steps

1. Run `notebooks/colab_run_experiment.ipynb` in Google Colab (free T4 GPU)
   for the 1.5B model first, to validate `LocalHFBackend` against real
   model output on a small sample before committing to larger runs.
2. Drop the resulting JSON files into `results/raw/`, then repeat step 1 for
   7B and 14B.
3. Aggregate the three sizes' results and generate the scaling-curve plot
   via `pipeline/scoring.py:plot_scaling_curve` (the aggregation logic
   already exists in `notebooks/analysis_scaffold.ipynb`, just pointed at
   synthetic data for now — swapping in the real `results/raw/*.json` files
   needs no code changes to the plotting/CI logic itself).
4. Write up results, limitations, and what to investigate next.
5. Optional: activation-patching deep dive on the most interesting size,
   and a cross-post to the Alignment Forum / LessWrong.
