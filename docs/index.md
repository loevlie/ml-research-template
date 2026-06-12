# ml-research-template

*A Copier template for ML research projects — PyTorch + Lightning Fabric + typed configs (tyro) + uv, with multi-seed significance testing and `copier update` baked in.*

```bash
uv tool install copier
copier copy --trust gh:loevlie/ml-research-template my-project
```

Answer ~10 prompts. Thirty seconds later you have a ready-to-train project: git initialized, dependencies locked, pre-commit installed, and a training run one command away.

## What you get

- **An explicit training loop** (~60 lines you can read) on Lightning Fabric — device placement, mixed precision, and DDP without a `Trainer` black box. Checkpoint resume, LR scheduling, and gradient accumulation are wired in and config-driven.
- **Statistical rigor on day one** — multi-seed launchers (local + SLURM array) and an aggregator that reports bootstrap CIs, paired Wilcoxon/t-tests, and Cohen's d. Publication-ready numbers, not single-seed anecdotes.
- **Typed configs, Hydra-style CLI** — pydantic schemas your IDE checks, with the familiar `experiment=x loss=contrastive model.lr=1e-3` syntax kept; typos die at parse time. Sweeps fan out to SLURM with one flag.
- **Runtime shape checking** via `jaxtyping` + `beartype` — broadcasting bugs die on the first forward pass.
- **Domain flavors** — a `tabular` flavor (OpenML tasks, sklearn-estimator wrapper, estimator × task × fold benchmark harness) for tabular foundation-model work, and a `multimodal` flavor (timm / open_clip / HF datasets, CLIP-style contrastive objective).
- **`copier update`** — pull template improvements into existing projects without clobbering your code. The compounding win across a PhD's worth of projects.

## Where to go

| You want to… | Read |
|---|---|
| Generate your first project | [Getting started](getting-started.md) |
| Understand why each tool was picked | [The stack (and why)](stack.md) |
| Train, override configs, add an objective | [Train & experiment](workflows/training.md) |
| Run seeds/sweeps on a cluster | [Run on SLURM](workflows/slurm.md) |
| Report results with significance tests | [Multi-seed statistics](workflows/multi-seed-stats.md) |
| Benchmark tabular models on OpenML | [Tabular benchmarking](workflows/tabular-benchmarking.md) |

## Who this is for

Researchers who value **control, rigor, and reproducibility** over framework magic — and who start more than one project. If you want a callback-driven `Trainer`, use [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) instead; if you work in JAX/TF, this isn't for you.
