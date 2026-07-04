# Getting started

## Prerequisites

[uv](https://docs.astral.sh/uv/) is the only hard requirement — it installs Python itself if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install copier
```

## Generate a project

```bash
copier copy --trust gh:loevlie/ml-research-template my-project
```

!!! note "`--trust`"
    The template runs four post-generation tasks: `git init`, `uv lock`, `uv sync --extra dev`, and `pre-commit install`. `--trust` allows that. Drop the flag to run them yourself.

### The prompts

| Prompt | Default | What it controls |
|---|---|---|
| `project_name` | — | Human-readable name ("Retinal OCT Classifier") |
| `package_name` | derived | Import name (`retinal_oct_classifier`) |
| `project_description` | generic | `pyproject.toml` + README |
| `author_name` / `author_email` | — | LICENSE + pyproject authors |
| `python_version` | `3.12` | Pins `.python-version`, `requires-python`, ruff/mypy targets (3.11–3.13) |
| `framework` | `pytorch` | Deep learning library: `pytorch` or `jax` — see [JAX](workflows/jax.md) |
| `cuda_version` | `cu124` | GPU wheel routing (`cpu`, `cu118`–`cu128`); for jax, any CUDA choice installs `jax[cuda12]` |
| `logger` | `wandb` | Default tracker: `wandb`, `trackio`, `tensorboard`, `csv` — see [Tracking](workflows/tracking.md) |
| `flavor` | `generic` | Domain scaffolding: `generic`, `tabular`, `multimodal` (PyTorch only — skipped for jax) |
| `include_example` | `true` | MLP reference example, smoke tests, demo, per-project docs |
| `include_dennys_rules` | `false` | A research operating manual as `DENNYS_RULES.md` |

### Pick a flavor

=== "generic"

    The base template: training loop, configs, multi-seed stats, SLURM scripts. Right for most projects.

=== "tabular"

    Adds OpenML task loading with standardized CV folds, an sklearn-estimator wrapper for your torch models, foundation-model baselines (TabPFN, TabICL) as an extra, and an estimator × task × fold benchmark harness with SLURM-array support. See [Tabular benchmarking](workflows/tabular-benchmarking.md).

=== "multimodal"

    Adds timm / open_clip / transformers / HF datasets as dependencies, an image-text dataloader with CLIP-matched preprocessing, and pairs with the CLIP-style contrastive objective. See [Multimodal](workflows/multimodal.md).

The flavors build on PyTorch libraries. Choosing `framework=jax` gives you the generic project with a flax NNX + optax training core instead — same configs, same commands ([what changes](workflows/jax.md)).

<div class="ix-card" id="ix-tree"></div>

## First training run

<!-- termynal -->

```
$ cd my-project
$ uv run python src/<package_name>/train.py
run dir   outputs/2026-06-12/10-53-13
model     ExampleModel | 22,026 params | lr=0.0003
---> 100%
Epoch   0 | train_loss=2.3055 | val_loss=2.2940 | val_acc=0.0850
best      val_loss=2.2925 | ckpts outputs/.../best.ckpt
```

That trains the reference MLP on synthetic data — a known-good baseline that proves the install, config parsing, loop, and checkpointing on your machine before your own code adds any unknowns. Then:

```bash
uv run python src/<package_name>/train.py model.lr=1e-3 data.batch_size=128   # CLI overrides
uv run python src/<package_name>/train.py experiment=example                  # named experiment
uv run pytest                                                                 # smoke tests
```

Every run lands in `outputs/<date>/<time>/` with the resolved config, logs, `best.ckpt`, `last.ckpt`, and `metrics.json`.

## Make it yours

Two factory functions are the seams between your code and the entry points —
replace their bodies and you never touch `train.py` or `eval.py`:

1. **Data** — `create_dataloaders(cfg, seed)` in `src/<pkg>/data/datamodule.py`. Swap in your dataset; new knobs (paths, transforms) become fields on `DataConfig` in `configs.py`.
2. **Model** — `build_model(cfg)` in `src/<pkg>/models/module.py`. Swap in your architecture (keep the `@jaxtyped` shape annotations); new knobs go on `ModelConfig`.
3. **Loss** — if it isn't plain cross-entropy, add an objective — see [Train & experiment](workflows/training.md).

All of these files are listed in the template's `_skip_if_exists`, so future
[`copier update`](updating.md) runs never overwrite them.

!!! tip "See it done end-to-end"
    The [tutorial](tutorial.md) walks this exact path with a real dataset —
    from `copier copy` to a multi-seed confidence interval in ~15 minutes.
