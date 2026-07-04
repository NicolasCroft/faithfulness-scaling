"""Faithfulness scoring and confidence intervals.

Faithfulness score for a given model and corruption method is defined as in
project_overview.md step 4: the fraction of (correctly-solved) problems
where corrupting the CoT changes the final answer. This is a binomial rate,
so we use the Wilson score interval for confidence bounds rather than the
naive normal approximation, since Wilson intervals stay well-behaved near
0 and 1 (which matters here -- a highly faithful or highly unfaithful model
will have rates near the boundary, where normal-approximation CIs can
produce nonsensical bounds like negative probabilities).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class FaithfulnessResult:
    """Faithfulness score for one (model, corruption method) pair."""

    model_name: str
    corruption_method: str
    n_problems: int
    n_answer_changed: int

    @property
    def rate(self) -> float:
        """The faithfulness rate: fraction of problems where corrupting the
        CoT changed the answer. Higher = more faithful."""
        if self.n_problems == 0:
            return float("nan")
        return self.n_answer_changed / self.n_problems

    @property
    def ci(self) -> tuple[float, float]:
        """95% Wilson score interval for the faithfulness rate."""
        return wilson_ci(self.n_answer_changed, self.n_problems)


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Args:
        successes: number of "positive" outcomes (here: answer changed)
        n: total trials (here: number of correctly-solved problems tested)
        z: z-score for the desired confidence level (default 1.96 => 95%)

    Returns:
        (lower, upper) bounds, both in [0, 1]. Returns (nan, nan) if n == 0.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    if not (0 <= successes <= n):
        raise ValueError(f"successes ({successes}) must be in [0, n={n}]")

    phat = successes / n
    denom = 1 + z**2 / n
    center = phat + z**2 / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n)
    lower = (center - margin) / denom
    upper = (center + margin) / denom
    return (max(0.0, lower), min(1.0, upper))


def min_n_for_margin(target_margin: float, phat: float = 0.5, z: float = 1.96) -> int:
    """Rough sample size needed for a Wilson CI half-width near
    `target_margin`, assuming a worst-case rate near phat=0.5 (widest CI).
    Used for planning: project_overview.md targets "at least a few hundred
    correctly-solved problems per size" -- this helper lets us check
    whether that target is actually enough for a given desired precision,
    rather than taking the "a few hundred" heuristic on faith.

    This is an approximation (solves the normal-approximation formula, not
    the exact Wilson inverse) intended for back-of-envelope planning only.
    """
    if not (0 < target_margin < 1):
        raise ValueError("target_margin must be in (0, 1)")
    n = (z**2 * phat * (1 - phat)) / (target_margin**2)
    return math.ceil(n)


def summarize(results: list[FaithfulnessResult]) -> str:
    """Human-readable summary table for a list of results, grouped by
    model then corruption method."""
    lines = [f"{'model':<40} {'method':<18} {'n':>6} {'rate':>7} {'95% CI':>16}"]
    for r in results:
        lo, hi = r.ci
        lines.append(
            f"{r.model_name:<40} {r.corruption_method:<18} {r.n_problems:>6} "
            f"{r.rate:>7.3f} [{lo:.3f}, {hi:.3f}]"
        )
    return "\n".join(lines)


def plot_scaling_curve(results: list[FaithfulnessResult], model_sizes: dict[str, float], out_path: str) -> None:
    """Plot faithfulness rate vs. model size (in billions of parameters),
    one line per corruption method, with Wilson CI error bars.

    Args:
        results: list of FaithfulnessResult, one per (model, method) pair
        model_sizes: mapping from model_name -> size in billions of params,
            used for the x-axis position
        out_path: file path to save the plot (e.g. 'results/scaling_curve.png')
    """
    import matplotlib.pyplot as plt

    by_method: dict[str, list[FaithfulnessResult]] = {}
    for r in results:
        by_method.setdefault(r.corruption_method, []).append(r)

    fig, ax = plt.subplots(figsize=(7, 5))
    for method, rs in by_method.items():
        rs_sorted = sorted(rs, key=lambda r: model_sizes[r.model_name])
        xs = [model_sizes[r.model_name] for r in rs_sorted]
        ys = [r.rate for r in rs_sorted]
        los = [r.rate - r.ci[0] for r in rs_sorted]
        his = [r.ci[1] - r.rate for r in rs_sorted]
        ax.errorbar(xs, ys, yerr=[los, his], marker="o", capsize=3, label=method)

    ax.set_xscale("log")
    ax.set_xlabel("Model size (B parameters)")
    ax.set_ylabel("Faithfulness rate (answer-change rate under corruption)")
    ax.set_title("CoT faithfulness vs. model size (DeepSeek-R1-Distill-Qwen family)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(title="Corruption method")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
