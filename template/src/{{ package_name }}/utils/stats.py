"""Statistical significance testing utilities.

Rule 25-29: No claim without evidence. This module provides helpers for
paired significance tests and result reporting.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class SignificanceResult:
    """Result of a paired significance test."""

    test_name: str
    statistic: float
    p_value: float
    significant_at_005: bool
    significant_at_001: bool
    ours_mean: float
    ours_std: float
    baseline_mean: float
    baseline_std: float
    n_seeds: int
    effect_size: float  # Cohen's d


def paired_significance_test(
    scores_ours: list[float],
    scores_baseline: list[float],
    test: str = "wilcoxon",
) -> SignificanceResult:
    """Run a paired significance test between two methods evaluated on the same seeds.

    Args:
        scores_ours: Metric values for our method, one per seed.
        scores_baseline: Metric values for the baseline, one per seed.
            Must be same length as scores_ours and use matching seeds.
        test: One of "wilcoxon" (default, non-parametric) or "ttest"
            (paired t-test, assumes normality).

    Returns:
        SignificanceResult with test statistics and summary.

    Raises:
        ValueError: If inputs have different lengths or unknown test name.
    """
    ours = np.array(scores_ours)
    base = np.array(scores_baseline)

    if len(ours) != len(base):
        raise ValueError(
            f"Mismatched lengths: ours={len(ours)}, baseline={len(base)}. "
            "Use the same seeds for paired evaluation (Rule 27)."
        )

    n = len(ours)
    diff = ours - base
    pooled_std = np.sqrt((np.std(ours, ddof=1) ** 2 + np.std(base, ddof=1) ** 2) / 2)
    effect_size = float(np.mean(diff) / pooled_std) if pooled_std > 0 else 0.0

    if test == "wilcoxon":
        if n < 6:
            # Wilcoxon needs at least 6 samples for meaningful results
            stat, p = stats.ttest_rel(ours, base)
            test_name = "paired t-test (fallback, n<6 for Wilcoxon)"
        else:
            stat, p = stats.wilcoxon(ours, base)
            test_name = "Wilcoxon signed-rank"
    elif test == "ttest":
        stat, p = stats.ttest_rel(ours, base)
        test_name = "paired t-test"
    else:
        raise ValueError(f"Unknown test: {test!r}. Use 'wilcoxon' or 'ttest'.")

    return SignificanceResult(
        test_name=test_name,
        statistic=float(stat),
        p_value=float(p),
        significant_at_005=p < 0.05,
        significant_at_001=p < 0.01,
        ours_mean=float(np.mean(ours)),
        ours_std=float(np.std(ours, ddof=1)),
        baseline_mean=float(np.mean(base)),
        baseline_std=float(np.std(base, ddof=1)),
        n_seeds=n,
        effect_size=effect_size,
    )


def report_results(result: SignificanceResult) -> str:
    """Format a SignificanceResult as a publication-ready string.

    Args:
        result: Output of paired_significance_test.

    Returns:
        Multi-line string suitable for inclusion in a paper or log.
    """
    sig = ""
    if result.significant_at_001:
        sig = " **"
    elif result.significant_at_005:
        sig = " *"

    return (
        f"Ours:     {result.ours_mean:.4f} +/- {result.ours_std:.4f}\n"
        f"Baseline: {result.baseline_mean:.4f} +/- {result.baseline_std:.4f}\n"
        f"Delta:    {result.ours_mean - result.baseline_mean:+.4f}{sig}\n"
        f"Test:     {result.test_name}\n"
        f"Stat:     {result.statistic:.4f}, p={result.p_value:.4f}\n"
        f"Effect:   Cohen's d = {result.effect_size:.3f}\n"
        f"Seeds:    {result.n_seeds}\n"
        f"{'Significant at p<0.05' if result.significant_at_005 else 'NOT significant at p<0.05'}\n"
        f"{'Significant at p<0.01' if result.significant_at_001 else 'NOT significant at p<0.01'}"
    )


def bootstrap_ci(
    scores: list[float],
    confidence: float = 0.95,
    n_resamples: int = 10000,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for the mean.

    Args:
        scores: Metric values (e.g., across seeds or test samples).
        confidence: Confidence level (default 0.95 for 95% CI).
        n_resamples: Number of bootstrap resamples.

    Returns:
        Tuple of (mean, ci_lower, ci_upper).
    """
    data = (np.array(scores),)
    result = stats.bootstrap(data, np.mean, confidence_level=confidence, n_resamples=n_resamples)
    mean = float(np.mean(scores))
    return mean, float(result.confidence_interval.low), float(result.confidence_interval.high)
