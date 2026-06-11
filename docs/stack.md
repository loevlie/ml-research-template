# The stack (and why)

Every tool here was re-evaluated against the 2026 landscape. The table is the summary; the sections below are the reasoning, including the bets we're deliberately *not* taking.

| Concern | Choice | One-line why |
|---|---|---|
| Packaging | **uv** | Won the ecosystem; lockfiles + CUDA wheel routing built in |
| Config | **Hydra** | Stagnant upstream but stable; its sweeps + SLURM launcher carry the workflow |
| Training | **Lightning Fabric** (own loop) | Distributed/AMP plumbing without a Trainer black box |
| Tracking | **wandb** default, **trackio** local fallback | Best academic UX; a credible local escape hatch |
| Shapes | **jaxtyping + beartype** | Runtime shape checks; bugs die at the first forward pass |
| Lint/format | **ruff** | One fast tool replaces black + isort + flake8 |
| Stats | **scipy** (+ pingouin extra) | Paired Wilcoxon, bootstrap CIs, Cohen's d out of the box |

## uv

uv is the default Python toolchain in 2026 — installs are 10–100× faster than pip, the lockfile makes every project reproducible, and `[tool.uv.sources]` routes PyTorch to the right CUDA wheel index per platform (the `cuda_version` prompt writes this for you). PyTorch's experimental *wheel variants* will eventually auto-detect accelerators and make even that config unnecessary; until it stabilizes, the index routing stays.

## Hydra — kept deliberately

Hydra's last feature release was early 2023 and Meta isn't investing in it; new codebases (torchtitan, LeRobot, nerfstudio) have moved to typed dataclass configs (tyro, draccus). We keep Hydra anyway, with eyes open:

- **What it buys us:** composable config groups with CLI overrides, version-controlled experiment files, `-m` multirun, the Optuna sweeper, and the submitit SLURM launcher — one integrated path from "idea" to "500 jobs on the cluster". No typed-config stack replicates that without hand-rolled glue.
- **What it costs:** stringly-typed configs (errors at runtime, not in your IDE) and a dependency that won't improve.
- **The escape hatch:** if Hydra ever breaks on a future Python/OmegaConf, the migration is dataclass configs + [tyro](https://github.com/brentyi/tyro) or [draccus](https://github.com/dlwh/draccus) for the CLI, and [submitit](https://github.com/facebookresearch/submitit) directly (~30 lines) for SLURM fan-out. The training code wouldn't change — config parsing is already kept out of the loop primitives.

## Lightning Fabric, not Trainer

Research code in 2026 is mostly hand-written loops with a thin distributed layer (HF Accelerate or Fabric) underneath — frameworks with callback machinery fight you on custom objectives, multi-network updates, and `torch.compile`. Fabric gives device placement, mixed precision (`precision: bf16-mixed`), DDP, and checkpoint I/O while the loop stays ~60 readable lines in `training_loop.py`. The loop already handles [resume, LR scheduling, and gradient accumulation](workflows/resume-checkpointing.md) — config-driven, no callbacks.

The `Objective` protocol (`objectives.py`) keeps the loop loss-agnostic: an objective is any callable `(model, batch) -> {"loss": ...}`, so supervised / contrastive / masked-prediction swap via `loss=<name>` without touching the loop.

## Tracking: wandb + trackio

The 2025–26 tracking landscape moved a lot: Neptune was acquired by OpenAI and shut down (March 2026); W&B was acquired by CoreWeave (2025) but its **free academic tier remains the best UX** for a research lab; HuggingFace launched **trackio**, a local-first, wandb-API-compatible tracker. The template treats trackers as swappable Fabric loggers — `logger=wandb|trackio|tensorboard|csv` at runtime — so the platform risk of any one vendor stays one config key deep. Details in [Experiment tracking](workflows/tracking.md).

## jaxtyping + beartype

Shape annotations like `Float[Tensor, "batch features"]` are checked *at runtime* on every call. Despite the name, jaxtyping is framework-agnostic and is the standard answer to silent-broadcasting pain in research code. The dev extras also ship `lovely-tensors` (readable tensor reprs), `torchinfo` (model summaries), and `hypothesis-torch` (property-based tests — e.g. the tabular flavor's row-permutation-invariance test).

## What we deliberately skip

- **Lightning Trainer / Composer / HF Trainer** — wrong altitude for novel-method research.
- **DVC** — for tabular work, OpenML task IDs *are* the data versioning; for big blobs, HF Hub (Xet) is the pragmatic store.
- **Multi-node DDP scaffolding** — out of scope; crib torchtitan when you actually need it.
- **SkyPilot / snakemake** — watch list: SkyPilot ≥0.12 speaks native SLURM (cloud bursting), snakemake's SLURM executor suits cached benchmark DAGs. Adopt per-project if needed.
