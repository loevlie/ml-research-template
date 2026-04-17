# ml-project-template

A [Copier](https://copier.readthedocs.io/) template for ML research projects: PyTorch + Lightning Fabric + Hydra + uv, with optional MLP reference example, demo, docs, and multi-seed/significance-testing scaffolding.

## Quick start

```bash
# Install Copier (one-time)
uv tool install copier

# Generate a new project
copier copy --trust gh:loevlie/ml-research-template path/to/new-project
cd path/to/new-project
uv run python src/<your_package_name>/train.py
```

`--trust` is required because the template runs `git init`, `uv lock`, `uv sync`, and `pre-commit install` after generation. Skip it if you want to run those steps yourself.

You can also render from a local checkout while developing:

```bash
copier copy --trust /path/to/ml-research-template /tmp/new-project
```

## What you'll be asked

| Prompt | Default | Notes |
|---|---|---|
| `project_name` | — | Human-readable, e.g. `"Retinal OCT Classifier"` |
| `package_name` | derived | Import name (`retinal_oct_classifier`). Validated against `^[a-z][a-z0-9_]*$` |
| `project_description` | generic | Used in `pyproject.toml` and README |
| `author_name` | — | LICENSE + pyproject authors |
| `author_email` | — | pyproject authors |
| `python_version` | `3.11` | Pins `.python-version`, `requires-python`, ruff/mypy target |
| `cuda_version` | `cu124` | `cpu`, `cu118`, `cu124`, `cu126`, `cu128` — affects `[tool.uv.sources]` |
| `logger` | `wandb` | Default experiment tracker (`wandb`, `aim`, `tensorboard`, `csv`). You can switch at runtime via `logger=aim` |
| `include_example` | `true` | Ship the MLP reference example (`demo/`, `docs/`, `project_page/`, `configs/experiment/example.yaml`, `tests/test_model.py`, `mkdocs.yml`) |
| `include_dennys_rules` | `false` | Include Dennis Loevlie's research operating manual (`DENNYS_RULES.md`) |

## What's in the generated project

- **Training loop:** explicit, Fabric-wrapped, Hydra-configured. No hidden callbacks.
- **Configs:** composable Hydra YAMLs for `data/`, `model/`, `trainer/`, `logger/`, `experiment/`, `hparams_search/`, `local/` (gitignored)
- **Reproducibility:** `set_seed()` seeds Python/NumPy/PyTorch/CUDA/PYTHONHASHSEED; deterministic DataLoader workers
- **Multi-seed:** `scripts/run_seeds.sh` launches N seeds; `scripts/aggregate_seeds.py` does bootstrap CIs and paired significance tests
- **Shape checking:** `jaxtyping` + `beartype` runtime verification
- **Packaging:** `uv` with PyTorch CUDA/CPU index routing
- **CI:** GitHub Actions — ruff, mypy, pytest
- **Pre-commit:** ruff format/lint, trailing-whitespace, large-files
- **Optional:** Gradio demo for HF Spaces, academic project page, MkDocs docs

## Updating an existing project

Because Copier writes a `.copier-answers.yml` file into each generated project, you can pull in template improvements later:

```bash
cd path/to/existing-project
copier update --trust
```

Files listed in `_skip_if_exists` (model/data modules, experiment configs, README) are preserved to avoid clobbering your work. Other files get three-way merged; unresolved conflicts show up as `.rej` files or inline markers.

## Template layout

```
.
├── copier.yml              # Prompts and template config
├── template/               # Everything rendered into the new project
│   ├── pyproject.toml.jinja
│   ├── src/{{ package_name }}/
│   ├── configs/
│   ├── ...
└── README.md               # This file
```

Files with a `.jinja` suffix are rendered through Jinja (strip the suffix in the output). Files and directories wrapped in `{% if cond %}...{% endif %}` are conditionally included.

## License

MIT
