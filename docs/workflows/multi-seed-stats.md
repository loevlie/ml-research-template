# Multi-seed statistics

Single-seed deltas are noise until proven otherwise. The template makes the honest version — N seeds, paired tests, effect sizes — a two-command workflow.

## 1. Run the seeds

```bash
# locally, sequential
bash scripts/run_seeds.sh experiment=ours seeds="42,123,456,789,1337"

# on SLURM, parallel (one array task per seed)
sbatch scripts/sbatch_seeds.sh experiment=ours
```

Each seed writes `outputs/multi_seed_<stamp>/seed_<s>/metrics.json` (train.py saves it at the end of every run).

For a **paired** comparison, run the baseline with the *same seeds*:

```bash
bash scripts/run_seeds.sh experiment=baseline seeds="42,123,456,789,1337"
```

## 2. Aggregate

```bash
uv run python scripts/aggregate_seeds.py outputs/multi_seed_ours \
    --baseline outputs/multi_seed_baseline --metric val/acc
```

```text
Metric: val/acc
Mean:   0.9234 ± 0.0045
95% CI: [0.9145, 0.9310]

--- Paired Comparison ---
Ours:     0.9234 ± 0.0045
Baseline: 0.8912 ± 0.0051
Delta:    +0.0322 **
Test:     Wilcoxon signed-rank
Stat:     0.0000, p=0.0079
Effect:   Cohen's d = 6.712
Significant at p<0.01
```

## What's underneath (`src/<pkg>/utils/stats.py`)

- **Bootstrap CI** (10k resamples) for the mean — no normality assumption.
- **Wilcoxon signed-rank** by default (non-parametric, paired); falls back to a paired t-test below 6 seeds, where Wilcoxon can't reach significance anyway.
- **Cohen's d** so reviewers see magnitude, not just a p-value.

The functions take plain lists — reuse them for any paired comparison (per-dataset benchmark scores, ablations), as the [tabular benchmark aggregator](tabular-benchmarking.md) does.

## Reporting checklist

- Report `mean ± std (N seeds)` and the test used: *"0.923 ± 0.005 vs 0.891 ± 0.005, p < 0.01, Wilcoxon signed-rank over 5 shared seeds."*
- Same seeds for both methods — that's what "paired" means.
- 5 seeds is the floor for a claim; use more when the delta is small relative to the std.
- Decide the metric you'll test on **before** running.
