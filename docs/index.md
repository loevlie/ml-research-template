---
title: Home
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

# Start every project ready to publish

A Copier template for ML research — PyTorch or JAX, typed configs, uv —
with multi-seed significance testing and `copier update` baked in.

[Get started](getting-started.md){ .md-button .md-button--primary }
[15-minute tutorial](tutorial.md){ .md-button }
[Why this stack](stack.md){ .md-button }

</div>

<!-- termynal -->

```
$ uv tool install copier
$ copier copy --trust gh:loevlie/ml-research-template my-project
🎤 project_name? Wine Quality
🎤 framework? pytorch
🎤 logger? wandb
---> 100%
"Wine Quality" is ready at my-project
$ cd my-project
$ uv run python src/wine_quality/train.py
Epoch   0 | train_loss=2.3055 | val_loss=2.2940 | val_acc=0.0850
```

Ten prompts, thirty seconds: git initialized, dependencies locked, pre-commit installed, training.

## What you get

<div class="grid cards" markdown>

-   :material-sync:{ .lg .middle } **An explicit training loop**

    ---

    ~60 readable lines on Lightning Fabric — device placement, mixed precision,
    and DDP without a `Trainer` black box. Resume, LR scheduling, gradient
    accumulation, and clipping are wired in and config-driven. Prefer JAX?
    `framework=jax` swaps the core, keeps everything else.

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

## Is it for you?

<div class="grid" markdown>

<div markdown>
**Use it if you…**

- :material-check:{ .check } start more than one project — `copier update` compounds
- :material-check:{ .check } want every paper number backed by seeds + significance tests
- :material-check:{ .check } edit training loops weekly and want to *read* yours
- :material-check:{ .check } run on a SLURM cluster (or will soon)
</div>

<div markdown>
**Skip it if you…**

- :material-close:{ .cross } want a callback-driven `Trainer` — use [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)
- :material-close:{ .cross } work in TensorFlow
- :material-close:{ .cross } don't want to own the CLI code (one ~250-line file you can read)
- :material-close:{ .cross } need multi-node pretraining scaffolding — crib [torchtitan](https://github.com/pytorch/torchtitan)
</div>

</div>
