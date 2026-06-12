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
| `python_version` | `3.11` | Pins `.python-version`, `requires-python`, ruff/mypy targets |
| `cuda_version` | `cu124` | PyTorch wheel index (`cpu`, `cu118`–`cu128`) via `[tool.uv.sources]` |
| `logger` | `wandb` | Default tracker: `wandb`, `trackio`, `tensorboard`, `csv` — see [Tracking](workflows/tracking.md) |
| `flavor` | `generic` | Domain scaffolding: `generic`, `tabular`, `multimodal` |
| `include_example` | `true` | MLP reference example, smoke tests, demo, per-project docs |
| `include_dennys_rules` | `false` | A research operating manual as `DENNYS_RULES.md` |

### Pick a flavor

=== "generic"

    The base template: training loop, configs, multi-seed stats, SLURM scripts. Right for most projects.

=== "tabular"

    Adds OpenML task loading with standardized CV folds, an sklearn-estimator wrapper for your torch models, foundation-model baselines (TabPFN, TabICL) as an extra, and an estimator × task × fold benchmark harness with SLURM-array support. See [Tabular benchmarking](workflows/tabular-benchmarking.md).

=== "multimodal"

    Adds timm / open_clip / transformers / HF datasets as dependencies, an image-text dataloader with CLIP-matched preprocessing, and pairs with the CLIP-style contrastive objective. See [Multimodal](workflows/multimodal.md).

## First training run

```bash
cd my-project
uv run python src/<package_name>/train.py
```

That trains the reference MLP on synthetic data — proof the whole pipeline works before you touch anything. Then:

```bash
uv run python src/<package_name>/train.py model.lr=1e-3 data.batch_size=128   # CLI overrides
uv run python src/<package_name>/train.py experiment=example                  # named experiment
uv run pytest                                                                 # smoke tests
```

Every run lands in `outputs/<date>/<time>/` with the resolved config, logs, `best.ckpt`, `last.ckpt`, and `metrics.json`.

## Make it yours

1. Replace `src/<pkg>/data/datamodule.py` with your dataset (keep the seeded-split pattern).
2. Replace `src/<pkg>/models/module.py` with your architecture (keep the `@jaxtyped` shape annotations).
3. If your loss isn't plain cross-entropy, add an objective — see [Train & experiment](workflows/training.md).
4. Update `DataConfig` / `ModelConfig` in `src/<pkg>/configs.py` to match.

These files are listed in the template's `_skip_if_exists`, so future [`copier update`](updating.md) runs never overwrite them.
