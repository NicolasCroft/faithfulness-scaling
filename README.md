# Faithfulness Scaling in Distilled Reasoning Models

**Status: early build. No real model results yet — pipeline is implemented and
unit-tested against a mock backend; hosted-API inference is pending an
account/provider decision.**

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
  inference.py           - ModelBackend interface; MockBackend (no cost) + HostedAPIBackend (stub)
  scoring.py             - faithfulness rate, Wilson confidence intervals, scaling-curve plot
  run_experiment.py      - orchestrates steps 1-4 above for one model
  activation_patching.py - optional-deeper-layer scaffold: ActivationPatchingBackend
                           interface + MockActivationPatchingBackend + layer-sweep
                           localization logic. No real backend yet (needs
                           TransformerLens/nnsight + GPU, see module docstring)
tests/              - unit tests for all of the above (40 tests, run via pytest)
notebooks/          - analysis notebooks. analysis_scaffold.ipynb runs the full
                      run-experiment -> table -> scaling-curve-plot path against
                      a synthetic backend; swap in HostedAPIBackend + load_gsm8k
                      once real inference is unblocked (see notebook's last cell)
results/            - output plots and result tables. mock_demo_scaling_curve.png
                      is synthetic (from the notebook above), not a real result
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
- [ ] Hosted inference API selected and wired up — **blocked** on choosing
      a provider (Together AI / Fireworks / Groq / similar) and budget. A
      2026-07-09 pricing/availability check (see `NEEDS_YOUR_INPUT.md`) found
      Together AI is the only one of the three with confirmed serverless
      coverage of all three needed sizes (1.5B/7B/14B); Groq's docs only show
      32B/70B, and Fireworks' support for the smaller sizes was ambiguous.
- [ ] Truncation-test pipeline validated on one real model size.
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
- This scheduled-task sandbox has a narrow network allowlist: `pypi.org` is
  reachable, but `huggingface.co`, `api.together.xyz`, `api.fireworks.ai`,
  and `api.groq.com` all return 403 from the sandbox's proxy (confirmed
  again in Session 5). This means the real inference step likely can't run
  from inside this sandbox even after a provider/API key is chosen — see
  the open question in `NEEDS_YOUR_INPUT.md`.

## Next steps

1. Choose a hosted inference provider for DeepSeek-R1-Distill-Qwen and wire
   up `HostedAPIBackend._call_api` in `pipeline/inference.py`.
2. Validate the pipeline end-to-end against real model output on a small
   sample before committing to a full run.
3. Run all three core model sizes (1.5B / 7B / 14B) and generate the
   scaling-curve plot via `pipeline/scoring.py:plot_scaling_curve`.
4. Write up results, limitations, and what to investigate next.
5. Optional: activation-patching deep dive on the most interesting size,
   and a cross-post to the Alignment Forum / LessWrong.
