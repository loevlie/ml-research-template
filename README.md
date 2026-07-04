<div align="center">

# ml-research-template

<p><i>A Copier template for ML research projects — PyTorch or JAX, typed configs, uv,<br>with multi-seed significance testing and <code>copier update</code> baked in.</i></p>

[![docs](https://github.com/loevlie/ml-research-template/actions/workflows/docs.yml/badge.svg)](https://loevlie.github.io/ml-research-template/)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Copier](https://img.shields.io/badge/templated%20with-copier-a48cdc)
![uv](https://img.shields.io/badge/managed%20by-uv-blueviolet)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)

<p>
  <a href="https://loevlie.github.io/ml-research-template/"><b>Documentation</b></a>
  &nbsp;•&nbsp;
  <a href="https://loevlie.github.io/ml-research-template/getting-started/">Getting started</a>
  &nbsp;•&nbsp;
  <a href="https://loevlie.github.io/ml-research-template/tutorial/">15-minute tutorial</a>
  &nbsp;•&nbsp;
  <a href="https://loevlie.github.io/ml-research-template/tour/">Hands-on tour</a>
  &nbsp;•&nbsp;
  <a href="https://loevlie.github.io/ml-research-template/stack/">The stack (and why)</a>
</p>

<img src="demo.gif" alt="copier copy demo" width="800" />

<p><i>Generate a project in ~30 seconds.</i></p>

</div>

---

```bash
uv tool install copier
copier copy --trust gh:loevlie/ml-research-template my-project
cd my-project
uv run python src/<your_package_name>/train.py
```

Answer ~10 prompts, get a ready-to-train project: git initialized, deps locked, pre-commit installed.

## Why this template

- ⚡ **Fabric, not Trainer.** An explicit training loop you can read and edit — with checkpoint resume, warmup+cosine LR scheduling, and gradient accumulation already wired in, config-driven. No callbacks, no hidden machinery. Prefer JAX? `-d framework=jax` generates a flax NNX + optax core with the same configs and commands.

- ⚡ **Statistical rigor on day one.** Multi-seed launchers (local + SLURM array) and an aggregator with bootstrap CIs, paired Wilcoxon/t-tests, and Cohen's d. Publication-ready out of the box.

- ⚡ **Typed configs, familiar CLI.** pydantic schemas your IDE checks, parsed with the Hydra-style `experiment=x loss=contrastive model.lr=1e-3` syntax — typos die at parse time, not at epoch 40.

- ⚡ **Domain flavors.** `tabular` adds OpenML task loading, an sklearn-estimator wrapper, TabPFN/TabICL baselines, and a code-version-aware cached benchmark harness (exca). `multimodal` adds timm/open_clip/HF-datasets plumbing and a CLIP-style contrastive objective.

- ⚡ **SLURM-native.** `sweep.py`/`tune.py` fan out as job arrays via submitit, sbatch scripts cover the rest — with preemption auto-resume (requeued jobs continue from `last.ckpt`).

- ⚡ **Runtime shape checking** via `@jaxtyped(typechecker=beartype)` — broadcasting bugs die on the first forward pass, not after a day of training.

- ⚡ **Copier-native.** `copier update` pulls template improvements into existing projects without clobbering your work — the compounding win across multiple projects.

**Learn it by doing it:** the [15-minute tutorial](https://loevlie.github.io/ml-research-template/tutorial/) takes a real dataset to a publication-grade number, and the [hands-on tour](https://loevlie.github.io/ml-research-template/tour/) continues into sweeps, Optuna, crash recovery, and a paired significance test — every output on those pages is from a real run. The tutorial also exists as an [interactive walkthrough](https://loevlie.github.io/wine-quality-example/) (a scrollable code tour of every key line) backed by a browsable repo, one commit per step: [wine-quality-example](https://github.com/loevlie/wine-quality-example). The [docs](https://loevlie.github.io/ml-research-template/) also pack a one-page [cheatsheet](https://loevlie.github.io/ml-research-template/cheatsheet/), per-workflow guides with diagrams, and interactive widgets (a live bootstrap-CI demo, a CLI command builder, a what-gets-generated explorer).

## Why you might NOT want this

- You want a Lightning `Trainer` black box → use [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template).
- You work in TensorFlow → not for you (PyTorch and JAX are).
- You don't want to own the CLI code (one ~250-line file you can read) → use a framework instead.
- You need multi-node pretraining scaffolding → crib [torchtitan](https://github.com/pytorch/torchtitan).

## Updating an existing project

```bash
cd path/to/existing-project
copier update --trust
```

Template improvements merge in three-way; files you own (model, data, configs, README) are never clobbered. [Details](https://loevlie.github.io/ml-research-template/updating/).

## License

MIT
