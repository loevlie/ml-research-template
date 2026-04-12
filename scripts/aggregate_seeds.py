"""Aggregate results across seeds and run significance tests (Rules 26-29).

Usage:
    python scripts/aggregate_seeds.py outputs/multi_seed_20260411_120000

Reads metrics from each seed's CSV logger output and prints summary statistics
with significance tests if a baseline is provided.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from my_project.utils.stats import bootstrap_ci, paired_significance_test, report_results


def load_seed_metrics(results_dir: Path) -> dict[int, dict[str, float]]:
    """Load final metrics from each seed directory.

    Looks for a `metrics.json` file in each seed_* subdirectory.
    You should save this file at the end of training.
    """
    metrics = {}
    for seed_dir in sorted(results_dir.glob("seed_*")):
        metrics_file = seed_dir / "metrics.json"
        if metrics_file.exists():
            seed = int(seed_dir.name.split("_")[1])
            with open(metrics_file) as f:
                metrics[seed] = json.load(f)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate multi-seed results")
    parser.add_argument("results_dir", type=Path, help="Directory with seed_* subdirectories")
    parser.add_argument("--baseline", type=Path, default=None, help="Baseline results dir for comparison")
    parser.add_argument("--metric", type=str, default="val/acc", help="Metric to aggregate")
    args = parser.parse_args()

    # Load our results
    ours = load_seed_metrics(args.results_dir)
    if not ours:
        print(f"No metrics.json files found in {args.results_dir}/seed_*/")
        print("Save a metrics.json at the end of training with your final metrics.")
        return

    scores = [m[args.metric] for m in ours.values()]
    seeds = list(ours.keys())

    print(f"Metric: {args.metric}")
    print(f"Seeds:  {seeds}")
    print(f"Mean:   {np.mean(scores):.4f}")
    print(f"Std:    {np.std(scores, ddof=1):.4f}")

    # Bootstrap CI
    mean, ci_lo, ci_hi = bootstrap_ci(scores)
    print(f"95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

    # Paired comparison if baseline provided
    if args.baseline:
        baseline = load_seed_metrics(args.baseline)
        shared_seeds = sorted(set(ours.keys()) & set(baseline.keys()))
        if len(shared_seeds) < 3:
            print(f"\nOnly {len(shared_seeds)} shared seeds -- need at least 3 for significance test.")
            return

        ours_paired = [ours[s][args.metric] for s in shared_seeds]
        base_paired = [baseline[s][args.metric] for s in shared_seeds]

        print(f"\n--- Paired Comparison (shared seeds: {shared_seeds}) ---")
        result = paired_significance_test(ours_paired, base_paired)
        print(report_results(result))


if __name__ == "__main__":
    main()
