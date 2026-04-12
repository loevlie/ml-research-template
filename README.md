# My Project

> One-sentence description of what this project does.

**Paper:** [Title](link) | **Demo:** [HF Spaces](link) | **Project Page:** [link](link)

## Key Results

| Method | Metric 1 | Metric 2 |
|--------|----------|----------|
| Baseline A | 85.2 +/- 0.3 | 72.1 +/- 0.5 |
| Baseline B | 87.4 +/- 0.4 | 74.3 +/- 0.6 |
| **Ours** | **91.0 +/- 0.2** | **79.8 +/- 0.3** |

*Mean +/- std over 5 seeds. Ours vs Baseline B: p<0.01 (Wilcoxon signed-rank).*

## Installation

```bash
git clone https://github.com/username/my-project.git
cd my-project
pip install -e ".[dev]"
pre-commit install
```

## Quick Start

```bash
# Train with defaults
python src/my_project/train.py

# Train with CLI overrides
python src/my_project/train.py model.lr=1e-3 data.batch_size=128

# Run a named experiment config
python src/my_project/train.py experiment=example

# Multi-seed run (5 seeds)
bash scripts/run_seeds.sh experiment=example seeds="42,123,456,789,1337"

# Aggregate and get significance tests
python scripts/aggregate_seeds.py outputs/multi_seed_YYYYMMDD_HHMMSS

# Evaluate a checkpoint
python src/my_project/eval.py ckpt_path=/path/to/checkpoint.ckpt

# HP search with Optuna
python src/my_project/train.py -m hparams_search=optuna

# Switch logger via config (tensorboard is default)
python src/my_project/train.py logger=wandb          # cloud — needs: pip install -e ".[tracking-wandb]"
python src/my_project/train.py logger=aim             # local — needs: pip install -e ".[tracking-aim]"
python src/my_project/train.py logger=tensorboard     # local (default)
python src/my_project/train.py logger=csv             # local, zero dependencies
```

## Project Structure

```
.
├── DENNYS_RULES.md              # Research methodology & operating manual
├── configs/                     # Hydra configs (composable YAML)
│   ├── data/                    #   Dataset & dataloader configs
│   ├── experiment/              #   Version-controlled experiment configs
│   ├── hparams_search/          #   HP search configs (Optuna)
│   ├── logger/                  #   W&B / TensorBoard / CSV logger configs
│   ├── model/                   #   Model architecture configs
│   ├── trainer/                 #   Fabric settings (accelerator, precision, epochs)
│   ├── local/                   #   Machine-specific overrides (gitignored)
│   ├── train.yaml               #   Main training config
│   └── eval.yaml                #   Evaluation config
├── data/
│   ├── raw/                     #   Immutable original data
│   └── processed/               #   Transformed, model-ready data
├── demo/
│   └── app.py                   #   Gradio demo for HF Spaces
├── docs/                        #   MkDocs documentation source
├── notebooks/                   #   Exploration notebooks (not training)
├── project_page/
│   └── index.html               #   Academic project page for GitHub Pages
├── scripts/
│   ├── run_seeds.sh             #   Multi-seed experiment launcher
│   └── aggregate_seeds.py       #   Aggregate results + significance tests
├── src/my_project/
│   ├── data/
│   │   └── datamodule.py        #   Dataset + DataLoader factory
│   ├── models/
│   │   └── module.py            #   nn.Module with jaxtyping shapes
│   ├── utils/
│   │   ├── seed.py              #   Reproducibility (seed all RNGs)
│   │   └── stats.py             #   Significance tests & reporting
│   ├── train.py                 #   Training entry point (Hydra)
│   └── eval.py                  #   Evaluation entry point (Hydra)
├── tests/                       #   Smoke tests (overfit batch, init loss, etc.)
├── .github/workflows/ci.yml     #   GitHub Actions CI
├── .pre-commit-config.yaml      #   Ruff + mypy + pre-commit hooks
├── .devcontainer/               #   VS Code dev container config
├── Dockerfile                   #   Reproducible training environment
├── mkdocs.yml                   #   Documentation config
└── pyproject.toml               #   Dependencies, ruff, mypy, pytest config
```

## Tools & Why

| Tool | Purpose | Why this one |
|------|---------|-------------|
| [Lightning Fabric](https://lightning.ai/docs/fabric/) | Device/distributed | Multi-GPU, mixed precision — no hidden training loop |
| [Hydra](https://hydra.cc/) | Config management | CLI overrides, composition, auto-snapshots per run |
| [TensorBoard](https://www.tensorflow.org/tensorboard) / [W&B](https://wandb.ai/) / [Aim](https://aimstack.io/) | Experiment tracking | Switchable via config — TensorBoard (default), W&B (cloud), Aim (local with rich UI) |
| [jaxtyping](https://github.com/patrick-kidger/jaxtyping) + [beartype](https://github.com/beartype/beartype) | Shape checking | Runtime shape verification + self-documenting signatures |
| [MkDocs](https://www.mkdocs.org/) + [mkdocstrings](https://mkdocstrings.github.io/) | Documentation | Auto-generated from Google-style docstrings |
| [Ruff](https://github.com/astral-sh/ruff) | Linting + formatting | Replaces black+isort+flake8 in one tool, millisecond speed |
| [Gradio](https://gradio.app/) | Interactive demo | ML-native widgets, one-click deploy to HF Spaces |

## Reproducing Paper Results

```bash
# Table 1
bash scripts/run_seeds.sh experiment=paper_table1 seeds="42,123,456,789,1337"

# Table 2
bash scripts/run_seeds.sh experiment=paper_table2 seeds="42,123,456,789,1337"
```

Pre-trained models: [HuggingFace Hub](link) or [Zenodo](link)

## Citation

```bibtex
@inproceedings{author2026title,
    title     = {Paper Title},
    author    = {Loevlie, Dennis},
    booktitle = {Conference},
    year      = {2026}
}
```

## License

MIT
