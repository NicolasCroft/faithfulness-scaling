"""Loading the math problem eval set.

Default choice: GSM8K (grade-school math, 8.5K problems, free-text numeric
answers that are trivial to grade automatically by exact match). This is
one of the two options project_overview.md names ("a GSM8K or MATH subset");
GSM8K was picked over MATH for the first pass because its answers are
single numbers with a simple canonical format (`#### <answer>`), so
automatic grading has effectively zero ambiguity. MATH problems can have
answers that are expressions, sets, or intervals, which need a more careful
equivalence checker. If a harder problem distribution turns out to be
needed later (e.g. because GSM8K is too easy and all three model sizes
solve ~100% of it, leaving no meaningful faithfulness signal), switching to
a MATH subset is the natural fallback -- see NEEDS_YOUR_INPUT.md if/when
that decision point is reached.

This module deliberately has no hard dependency on network access at
import time. `load_gsm8k` will try to pull the real dataset via the
`datasets` library; if that's unavailable (no internet, package not
installed), it raises a clear error rather than silently substituting
fixture data. `FIXTURE_PROBLEMS` below is a small hand-written set used
only for unit tests and pipeline smoke-testing, not for real results.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Problem:
    """A single math problem with its ground-truth answer."""

    problem_id: str
    question: str
    answer: str  # canonical final numeric answer, as a string


def load_gsm8k(split: str = "test", n: int | None = None) -> list[Problem]:
    """Load GSM8K problems via the Hugging Face `datasets` library.

    Args:
        split: "train" or "test". Use "test" for the eval set.
        n: if given, truncate to the first n problems (deterministic order,
           not a random sample) -- useful for smoke tests before committing
           to a full run.

    Raises:
        ImportError: if the `datasets` package is not installed.
        RuntimeError: if the dataset can't be downloaded (e.g. no network).
    """
    try:
        import datasets  # type: ignore
    except ImportError as e:
        raise ImportError(
            "The `datasets` package is required to load GSM8K. "
            "Install with `pip install datasets`."
        ) from e

    try:
        ds = datasets.load_dataset("gsm8k", "main", split=split)
    except Exception as e:  # noqa: BLE001 - want a clear message either way
        raise RuntimeError(
            "Failed to download GSM8K from the Hugging Face Hub. This "
            "requires network access to huggingface.co, which is not "
            "available in this sandbox. Run this step on a machine with "
            "network access (or the eventual GPU/compute environment)."
        ) from e

    problems: list[Problem] = []
    for i, row in enumerate(ds):
        answer = _extract_gsm8k_answer(row["answer"])
        problems.append(
            Problem(problem_id=f"gsm8k_{split}_{i}", question=row["question"], answer=answer)
        )
        if n is not None and len(problems) >= n:
            break
    return problems


def _extract_gsm8k_answer(raw_answer: str) -> str:
    """GSM8K answers end with a line like '#### 42'. Extract just the
    number, stripping commas (e.g. '1,200' -> '1200')."""
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", raw_answer)
    if not match:
        raise ValueError(f"Could not find '####' final-answer marker in: {raw_answer!r}")
    return match.group(1).replace(",", "")


def grade(model_answer: str, gold_answer: str) -> bool:
    """Exact-match grading after normalizing whitespace, commas, and a
    trailing '.0' on floats that are really integers.
    """
    def normalize(s: str) -> str:
        s = s.strip().replace(",", "").replace("$", "")
        s = re.sub(r"\.0$", "", s)
        return s

    return normalize(model_answer) == normalize(gold_answer)


# ---------------------------------------------------------------------------
# Fixture data for tests / offline development. NOT used for real results.
# ---------------------------------------------------------------------------

FIXTURE_PROBLEMS: list[Problem] = [
    Problem(
        problem_id="fixture_0",
        question=(
            "Natalia sold clips to 48 of her friends in April, and then she "
            "sold half as many clips in May. How many clips did Natalia sell "
            "altogether in April and May?"
        ),
        answer="72",
    ),
    Problem(
        problem_id="fixture_1",
        question=(
            "Weng earns $12 an hour for babysitting. Yesterday, she just did "
            "50 minutes of babysitting. How much did she earn?"
        ),
        answer="10",
    ),
    Problem(
        problem_id="fixture_2",
        question=(
            "Betty is saving money for a new wallet which costs $100. Betty "
            "has only half of the money she needs. Her parents decided to "
            "give her $15 for that purpose, and her grandparents twice as "
            "much as her parents. How much more money does Betty need to buy "
            "the wallet?"
        ),
        answer="5",
    ),
]
