---
title: Home
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

# Start every project ready to publish

A Copier template for ML research — PyTorch + Lightning Fabric + typed configs + uv,
with multi-seed significance testing and `copier update` baked in.

[Get started](getting-started.md){ .md-button .md-button--primary }
[15-minute tutorial](tutorial.md){ .md-button }
[Why this stack](stack.md){ .md-button }

</div>

```bash
uv tool install copier
copier copy --trust gh:loevlie/ml-research-template my-project
```

Answer ~10 prompts. Thirty seconds later you have a ready-to-train project: git initialized, dependencies locked, pre-commit installed, and a training run one command away.

## What you get

<div class="grid cards" markdown>

-   :material-sync:{ .lg .middle } **An explicit training loop**

    ---

    ~60 readable lines on Lightning Fabric — device placement, mixed precision,
    and DDP without a `Trainer` black box. Resume, LR scheduling, gradient
    accumulation, and clipping are wired in and config-driven.

    [:octicons-arrow-right-24: Train & experiment](workflows/training.md)

-   :material-chart-bell-curve:{ .lg .middle } **Statistical rigor on day one**

    ---

    Multi-seed launchers (local + SLURM array) and an aggregator with bootstrap
    CIs, paired Wilcoxon/t-tests, and Cohen's d. Publication-ready numbers, not
    single-seed anecdotes.

    [:octicons-arrow-right-24: Multi-seed statistics](workflows/multi-seed-stats.md)

-   :material-code-braces-box:{ .lg .middle } **Typed configs, Hydra-style CLI**

    ---

    pydantic schemas your IDE checks, with the familiar
    `experiment=x loss=contrastive model.lr=1e-3` syntax kept — typos die at
    parse time with a suggestion, not at epoch 40.

    [:octicons-arrow-right-24: Typed configs & overrides](workflows/configs.md)

-   :material-server:{ .lg .middle } **SLURM-native, preemption-proof**

    ---

    Sweeps and Optuna searches fan out as job arrays via submitit; sbatch
    scripts cover the rest. Preempted jobs requeue into the same run directory
    and continue from `last.ckpt`.

    [:octicons-arrow-right-24: Run on SLURM](workflows/slurm.md)

-   :material-table:{ .lg .middle } **Domain flavors**

    ---

    `tabular`: OpenML tasks, an sklearn-estimator wrapper, TabPFN/TabICL
    baselines, and a code-version-aware cached benchmark harness.
    `multimodal`: timm / open_clip / HF datasets and a CLIP-style objective.

    [:octicons-arrow-right-24: Tabular benchmarking](workflows/tabular-benchmarking.md)

-   :material-update:{ .lg .middle } **`copier update`**

    ---

    Pull template improvements into existing projects without clobbering your
    code — the compounding win across a PhD's worth of projects. Runtime shape
    checking (jaxtyping + beartype) rides along everywhere.

    [:octicons-arrow-right-24: Updating projects](updating.md)

</div>

## Who this is for

Researchers who value **control, rigor, and reproducibility** over framework magic — and who start more than one project. If you want a callback-driven `Trainer`, use [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) instead; if you work in JAX/TF, this isn't for you.
