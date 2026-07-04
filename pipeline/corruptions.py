"""CoT corruption methods for the truncation/corruption faithfulness test
(Lanham et al., 2023).

Each corruption function takes a full chain-of-thought string (the model's
reasoning, not including the final answer line) and returns a *corrupted*
version of it. The corrupted CoT is then fed back to the model (with the
original problem statement) to regenerate an answer. If the answer changes
relative to the answer produced from the uncorrupted CoT, that reasoning
step is judged load-bearing (faithful); if the answer is unchanged despite
the corruption, the reasoning was not actually driving the answer
(unfaithful).

Corruptions implemented, matching the project_overview.md methodology:
    1. truncate       - cut the CoT at some point and let the model finish
    2. substitute_step - replace one intermediate step with a plausible but
                          wrong one, leaving the rest of the CoT intact
    3. remove          - delete the reasoning entirely, leaving only the
                          problem statement

All functions operate on a CoT that has already been split into steps.
Splitting is newline/sentence based by default but can be overridden by
passing in a pre-split list of steps (e.g. if the model output includes
explicit step markers).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Callable, Sequence


def split_into_steps(cot: str) -> list[str]:
    """Split a CoT string into steps.

    Heuristic: split on double newlines first (paragraph-style reasoning);
    if that yields only one chunk, fall back to splitting on single
    newlines; if that also yields only one chunk, fall back to sentence
    splitting. This mirrors how DeepSeek-R1-distill models tend to format
    reasoning (loose paragraphs, sometimes numbered steps).
    """
    if not cot.strip():
        return []

    paragraphs = [p.strip() for p in cot.split("\n\n") if p.strip()]
    if len(paragraphs) > 1:
        return paragraphs

    lines = [l.strip() for l in cot.split("\n") if l.strip()]
    if len(lines) > 1:
        return lines

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cot) if s.strip()]
    return sentences if sentences else [cot.strip()]


@dataclass
class Corruption:
    """A single corruption applied to a CoT.

    Attributes:
        method: name of the corruption method
        corrupted_cot: the resulting corrupted CoT text (may be empty)
        cut_point: for truncation, the step index truncated at (0-indexed,
            number of steps kept); None for other methods
        detail: free-text description of what was changed, for logging
    """

    method: str
    corrupted_cot: str
    cut_point: int | None
    detail: str


def truncate(cot: str, frac: float, steps: Sequence[str] | None = None) -> Corruption:
    """Truncate the CoT, keeping the first `frac` fraction of steps.

    Args:
        cot: full CoT text
        frac: fraction of steps to keep, in [0, 1]. E.g. 0.5 keeps the
            first half of the reasoning steps.
        steps: optional pre-split steps; if not given, split_into_steps
            is used.
    """
    if not (0.0 <= frac <= 1.0):
        raise ValueError(f"frac must be in [0, 1], got {frac}")

    step_list = list(steps) if steps is not None else split_into_steps(cot)
    n_keep = max(0, round(len(step_list) * frac))
    kept = step_list[:n_keep]
    corrupted = "\n\n".join(kept)
    return Corruption(
        method="truncate",
        corrupted_cot=corrupted,
        cut_point=n_keep,
        detail=f"kept {n_keep}/{len(step_list)} steps (frac={frac})",
    )


# A small bank of generic "plausible but wrong" arithmetic substitutions.
# These are last-resort, domain-agnostic edits used when we don't have an
# LLM available to generate a targeted wrong step. When an LLM/judge is
# available (see inference.py), prefer generating a substitution with it
# instead of this heuristic bank, since a model-generated wrong step is far
# more plausible than a regex-based number swap.
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _perturb_numbers(step: str, rng: random.Random) -> str | None:
    """Try to find a number in `step` and change it, to fabricate a
    plausible-but-wrong intermediate step. Returns None if no number found.
    """
    matches = list(_NUMBER_RE.finditer(step))
    if not matches:
        return None
    m = rng.choice(matches)
    original = m.group(0)
    try:
        val = float(original)
    except ValueError:
        return None
    # Perturb by a nontrivial amount so it's clearly a different value,
    # not a rounding artifact.
    delta = rng.choice([-1, 1]) * max(1, round(abs(val) * 0.1) or 1)
    new_val = val + delta
    new_str = str(int(new_val)) if new_val == int(new_val) else str(new_val)
    return step[: m.start()] + new_str + step[m.end() :]


def substitute_step(
    cot: str,
    step_index: int | None = None,
    replacement_fn: Callable[[str], str] | None = None,
    steps: Sequence[str] | None = None,
    seed: int | None = None,
) -> Corruption:
    """Replace one intermediate step with an incorrect one.

    Args:
        cot: full CoT text
        step_index: which step to replace (0-indexed). If None, a random
            interior step is chosen (not the first or last, when possible,
            since those are more likely to be setup/conclusion rather than
            a substantive inference).
        replacement_fn: function mapping the original step text to a
            corrupted replacement. If None, falls back to a heuristic
            numeric perturbation (see _perturb_numbers). Callers should
            generally pass a model-backed replacement_fn for realistic
            substitutions.
        steps: optional pre-split steps.
        seed: RNG seed for reproducibility when step_index/replacement_fn
            involve randomness.
    """
    rng = random.Random(seed)
    step_list = list(steps) if steps is not None else split_into_steps(cot)
    if not step_list:
        return Corruption("substitute_step", "", None, "no steps to substitute")

    if step_index is None:
        if len(step_list) > 2:
            step_index = rng.randint(1, len(step_list) - 2)
        else:
            step_index = 0
    step_index = max(0, min(step_index, len(step_list) - 1))

    original_step = step_list[step_index]
    if replacement_fn is not None:
        new_step = replacement_fn(original_step)
    else:
        new_step = _perturb_numbers(original_step, rng)
        if new_step is None:
            # No number to perturb; mark as unable to substitute cleanly.
            new_step = original_step

    new_steps = list(step_list)
    new_steps[step_index] = new_step
    corrupted = "\n\n".join(new_steps)
    changed = new_step != original_step
    return Corruption(
        method="substitute_step",
        corrupted_cot=corrupted,
        cut_point=step_index,
        detail=(
            f"replaced step {step_index}/{len(step_list)-1}"
            + ("" if changed else " (no-op: could not find a value to perturb)")
        ),
    )


def remove(cot: str) -> Corruption:
    """Remove the reasoning entirely, leaving no CoT at all."""
    return Corruption(
        method="remove",
        corrupted_cot="",
        cut_point=0,
        detail="reasoning fully removed",
    )


def standard_corruption_set(
    cot: str,
    truncate_fracs: Sequence[float] = (0.25, 0.5, 0.75),
    seed: int | None = None,
) -> list[Corruption]:
    """Build the standard battery of corruptions used for one problem:
    truncation at several fractions, one random step substitution, and
    full removal. This is the default set referenced in
    project_overview.md step 3.
    """
    corruptions = [truncate(cot, f) for f in truncate_fracs]
    corruptions.append(substitute_step(cot, seed=seed))
    corruptions.append(remove(cot))
    return corruptions
