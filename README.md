<div align="center">

# ml-research-template

<p><i>A Copier template for ML research projects — PyTorch + Lightning Fabric + Hydra + uv,<br>with multi-seed significance testing and <code>copier update</code> baked in.</i></p>

![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Copier](https://img.shields.io/badge/templated%20with-copier-a48cdc)
![uv](https://img.shields.io/badge/managed%20by-uv-blueviolet)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)

<p>
  <a href="#quick-start"><b>Quick start</b></a>
  &nbsp;•&nbsp;
  <a href="#why-this-template">Why this template</a>
  &nbsp;•&nbsp;
  <a href="#multi-seed--significance-testing">Multi-seed demo</a>
</p>

<img src="demo.gif" alt="copier copy demo" width="800" />

<p><i>Generate a project in ~30 seconds.</i></p>

</div>

---

```bash
uv tool install copier
copier copy --trust gh:loevlie/ml-research-template my-project
```

That's it. Answer ~10 prompts, get a ready-to-train project with git initialized, deps locked, and pre-commit installed.

---

## Why this template

- ⚡ **Fabric, not Trainer.** The training loop is ~40 explicit lines you can read and edit. Modify it for custom optimizers, multi-network updates, adversarial training, RL, curriculum learning — no callbacks, no hidden machinery.

- ⚡ **Statistical rigor on day one.** `scripts/run_seeds.sh` launches N seeds; `scripts/aggregate_seeds.py` computes bootstrap CIs + paired Wilcoxon/t-tests + Cohen's d. Publication-ready out of the box.

- ⚡ **Runtime shape checking** via `@jaxtyped(typechecker=beartype)` on tensor functions. Catches broadcasting bugs on the first forward pass, not after a day of wasted training.

- ⚡ **Copier-native, not "Use this template."** Generate with prompts instead of renaming 20 files by hand. `copier update` pulls template improvements into existing projects without clobbering your work — the compounding win across multiple projects.

- ⚡ **uv + PyTorch CUDA wheels pre-wired.** Pick `cu118`/`cu124`/`cu126`/`cu128`/`cpu` at template time. 10-100× faster installs than pip.

## Why you might NOT want this

- **Skip this if you want a Lightning `Trainer` black box** (callbacks, auto-EMA, Strategy-based DeepSpeed/FSDP) — use **[lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)** instead.

- **Skip this if you're doing non-PyTorch work** (JAX, TF) — not for you.

- **Skip this if you don't want to learn Hydra or Fabric** — expect a learning curve.

---

## Quick start

```bash
# 1. Install Copier (one-time)
uv tool install copier

# 2. Generate a new project
copier copy --trust gh:loevlie/ml-research-template my-project

# 3. Train
cd my-project
uv run python src/<your_package_name>/train.py

# 4. Multi-seed run with significance tests
bash scripts/run_seeds.sh experiment=example seeds="42,123,456,789,1337"
uv run python scripts/aggregate_seeds.py outputs/multi_seed_*
```

> `--trust` lets the template run `git init`, `uv lock`, `uv sync`, and `pre-commit install` after generation. Drop the flag if you want to run those yourself.

## What you get

```
my-project/
├── configs/                   # Composable Hydra YAMLs (data/, model/, trainer/, logger/, experiment/)
├── src/my_project/
│   ├── train.py               # Explicit Fabric training loop — ~200 lines, all visible
│   ├── eval.py
│   ├── models/module.py       # @jaxtyped(typechecker=beartype) shape-checked forward
│   ├── data/datamodule.py     # Reproducible splits, seeded workers
│   └── utils/{seed,stats}.py
├── scripts/
│   ├── run_seeds.sh           # Multi-seed launcher
│   └── aggregate_seeds.py     # Bootstrap CIs + paired significance tests
├── tests/                     # Smoke tests (overfit batch, init loss, shapes)
├── demo/app.py                # Gradio → HF Spaces (optional)
├── docs/                      # MkDocs autogen from docstrings (optional)
├── .copier-answers.yml        # Enables `copier update`
└── pyproject.toml             # uv-managed deps, CUDA index routing
```

---

## Multi-seed + significance testing

```bash
# Train 5 seeds of your method
bash scripts/run_seeds.sh experiment=ours seeds="42,123,456,789,1337"

# Train 5 seeds of a baseline with same seeds (for paired comparison)
bash scripts/run_seeds.sh experiment=baseline seeds="42,123,456,789,1337"

# Aggregate + Wilcoxon signed-rank
uv run python scripts/aggregate_seeds.py outputs/multi_seed_ours_* \
  --baseline outputs/multi_seed_baseline_* --metric val/acc
```

Output:

```
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

## Updating an existing project

```bash
cd path/to/existing-project
copier update --trust
```

Pulls template improvements (new CI rules, pre-commit updates, config defaults) into your project. Files listed in `_skip_if_exists` (your model, data, README, experiment configs) are preserved. Other conflicts show up as `.rej` files or inline markers — three-way merge, not clobber.

## Projects using this template

_Submit a PR to [README.md](README.md) to add yours._

## Prompts reference

<details>
<summary>The 10 prompts you'll answer</summary>

| Prompt | Default | Notes |
|---|---|---|
| `project_name` | — | Human-readable, e.g. `"Retinal OCT Classifier"` |
| `package_name` | derived | Import name (`retinal_oct_classifier`). Validated against `^[a-z][a-z0-9_]*$` |
| `project_description` | generic | Used in `pyproject.toml` and README |
| `author_name` | — | LICENSE + pyproject authors |
| `author_email` | — | pyproject authors |
| `python_version` | `3.11` | Pins `.python-version`, `requires-python`, ruff/mypy target |
| `cuda_version` | `cu124` | `cpu`, `cu118`, `cu124`, `cu126`, `cu128` — affects `[tool.uv.sources]` |
| `logger` | `wandb` | Default tracker (`wandb`, `aim`, `tensorboard`, `csv`). Switch at runtime via `logger=aim` |
| `include_example` | `true` | Ship the MLP reference (`demo/`, `docs/`, `project_page/`, `configs/experiment/example.yaml`, `tests/test_model.py`, `mkdocs.yml`) |
| `include_dennys_rules` | `false` | Include Dennis Loevlie's research operating manual (`DENNYS_RULES.md`) |

</details>

---

## Support

⭐ **If this saved you setup time, star the repo** — it's the main way others discover it.

Issues, PRs, and `copier update` conflict reports welcome.

## License

MIT
