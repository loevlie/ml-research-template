# The stack (and why)

Every tool here was re-evaluated against the 2026 landscape. The table is the summary; the sections below are the reasoning, including the bets we're deliberately *not* taking.

| Concern | Choice | One-line why |
|---|---|---|
| Packaging | **uv** | Won the ecosystem; lockfiles + CUDA wheel routing built in |
| Config | **pydantic + tyro** (utils/cli.py) | Typed, IDE-checked configs with Hydra's CLI ergonomics kept |
| Sweeps / SLURM | **submitit + optuna** (~150 lines of owned glue) | No frozen plugins; every line readable |
| Training | **Lightning Fabric** (own loop) | Distributed/AMP plumbing without a Trainer black box |
| Caching | **exca + codever** (tabular) | Benchmark cells memoized, keyed to config *and* code version |
| Tracking | **wandb** default, **trackio** local fallback | Best academic UX; a credible local escape hatch |
| Shapes | **jaxtyping + beartype** | Runtime shape checks; bugs die at the first forward pass |
| Lint/format | **ruff** | One fast tool replaces black + isort + flake8 |
| Stats | **scipy** (+ pingouin extra) | Paired Wilcoxon, bootstrap CIs, Cohen's d out of the box |

## uv

uv is the default Python toolchain in 2026 — installs are 10–100× faster than pip, the lockfile makes every project reproducible, and `[tool.uv.sources]` routes PyTorch to the right CUDA wheel index per platform (the `cuda_version` prompt writes this for you). PyTorch's experimental *wheel variants* will eventually auto-detect accelerators and make even that config unnecessary; until it stabilizes, the index routing stays.

## Typed configs — and why we left Hydra

This template originally ran on Hydra and kept it deliberately for a while: its config groups, multirun, and submitit launcher were a uniquely integrated workflow. But Hydra's last feature release was early 2023, its plugin ecosystem is frozen, and the field moved to typed dataclass/pydantic configs (torchtitan, LeRobot, nerfstudio, Levanter). We migrated to **pydantic schemas + [tyro](https://github.com/brentyi/tyro)** — and kept the two things Hydra still did best:

- **Free-order `key=value` CLI with group swaps** — `utils/cli.py` (~150 lines) accepts the exact Hydra syntax (`experiment=example loss=contrastive model.lr=1e-3`) on top of tyro, so muscle memory and scripts survived the migration. Underneath, every override is type-checked: typos die at parse time with a suggestion.
- **`${...}` interpolation** — replaced by the derived-field pattern (`None`-defaults resolved in one visible `resolved()` method) in code, with literal `${a.b}` references still supported inside YAML override files.

What Hydra's orchestration did is now ~150 lines of owned, readable glue: `scripts/sweep.py` (itertools × submitit job arrays), `scripts/tune.py` (Optuna + JournalFileBackend on the shared filesystem), and `utils/run_dir.py` (timestamped dirs + resolved-config snapshots with git state). New dependencies — tyro, submitit, optuna — are all actively maintained; nothing in the config/orchestration path is frozen anymore.

## Lightning Fabric, not Trainer

Research code in 2026 is mostly hand-written loops with a thin distributed layer (HF Accelerate or Fabric) underneath — frameworks with callback machinery fight you on custom objectives, multi-network updates, and `torch.compile`. Fabric gives device placement, mixed precision (`trainer.precision=bf16-mixed`), DDP, and checkpoint I/O while the loop stays ~75 readable lines in `training_loop.py`, with [resume, LR scheduling, gradient accumulation, and clipping](workflows/resume-checkpointing.md) config-driven.

The `Objective` protocol (`objectives.py`) keeps the loop loss-agnostic: an objective is any callable `(model, batch) -> {"loss": ...}`, so supervised / contrastive / masked-prediction swap via `loss=<name>` without touching the loop.

## exca + codever (tabular flavor)

Benchmark cells are memoized on disk by [exca](https://github.com/facebookresearch/exca) (Meta FAIR, by the submitit author), keyed on the typed estimator config, task, fold, seed — **and the code version**. `utils/codever.py` fingerprints the package's AST-normalized source at submit time, auto-bumps `0.0.N-<hash>` on semantic edits, and appends a diff changelog. Re-running a grid recomputes only missing or code-changed cells; reverting code resurrects the old cache. Details in [Tabular benchmarking](workflows/tabular-benchmarking.md).

## Tracking: wandb + trackio

The 2025–26 tracking landscape moved a lot: Neptune was acquired by OpenAI and shut down (March 2026); W&B was acquired by CoreWeave (2025) but its **free academic tier remains the best UX** for a research lab; HuggingFace launched **trackio**, a local-first, wandb-API-compatible tracker. The template treats trackers as swappable (`logger.kind=wandb|trackio|tensorboard|csv`), so the platform risk of any one vendor stays one config key deep. Details in [Experiment tracking](workflows/tracking.md).

## jaxtyping + beartype

Shape annotations like `Float[Tensor, "batch features"]` are checked *at runtime* on every call. Despite the name, jaxtyping is framework-agnostic and is the standard answer to silent-broadcasting pain in research code. The dev extras also ship `lovely-tensors` (readable tensor reprs), `torchinfo` (model summaries), and `hypothesis-torch` (property-based tests — e.g. the tabular flavor's row-permutation-invariance test).

## What we deliberately skip

- **Lightning Trainer / Composer / HF Trainer** — wrong altitude for novel-method research.
- **DVC** — for tabular work, OpenML task IDs *are* the data versioning; for big blobs, HF Hub (Xet) is the pragmatic store.
- **Multi-node DDP scaffolding** — out of scope; crib torchtitan when you actually need it.
- **SkyPilot / snakemake** — watch list: SkyPilot ≥0.12 speaks native SLURM (cloud bursting, still preview-grade), snakemake's SLURM executor suits cached benchmark DAGs. Adopt per-project if needed.
