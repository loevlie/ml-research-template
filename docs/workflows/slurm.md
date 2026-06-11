# Run on SLURM

Two complementary paths ship with every project. Use the launcher for sweeps, the sbatch scripts for everything else.

## Path 1 — Hydra submitit launcher (composes with sweeps)

```bash
uv sync --extra cluster                       # one-time: hydra-submitit-launcher

# every multirun job becomes one SLURM job
uv run python src/<pkg>/train.py -m launcher=slurm seed=42,43,44,45,46
uv run python src/<pkg>/train.py -m launcher=slurm hparams_search=optuna
uv run python src/<pkg>/train.py -m launcher=slurm_debug trainer.max_epochs=2   # dry-run partition
```

Fill in `partition` and `account` in `configs/launcher/slurm.yaml` once — or keep them out of git in `configs/local/` / pass `hydra.launcher.partition=gpu` on the CLI.

## Path 2 — sbatch scripts (no extra deps, fully inspectable)

```bash
sbatch scripts/sbatch_train.sh experiment=example     # single job, single/multi-GPU via torchrun
sbatch scripts/sbatch_seeds.sh experiment=example     # array job: one seed per task
```

Both assume `.venv/` exists on a shared filesystem (`uv sync --extra dev --extra cluster` on the login node). Extra args are forwarded to `train.py`.

## Preemption survival — built in

The sbatch scripts pin each run's directory to the job ID and pass `trainer.resume=auto`:

```bash
hydra.run.dir="outputs/slurm_${SLURM_JOB_ID}" trainer.resume=auto
```

`--requeue` means a preempted job is resubmitted **with the same job ID** → it lands in the same directory → `resume=auto` picks up `last.ckpt` (saved every epoch) and continues from the last finished epoch. No babysitting. Details in [Checkpoints & resume](resume-checkpointing.md).

## Tabular flavor: array benchmarks

Tabular-FM evaluation is hundreds of small CPU jobs, not one big GPU run. `scripts/sbatch_benchmark.sh` maps an estimator × task × fold grid onto one SLURM array — see [Tabular benchmarking](tabular-benchmarking.md).

## Cluster notes

- **Logger on air-gapped compute nodes:** `WANDB_MODE=offline` + `wandb sync` from the login node, or `logger=trackio` / `logger=csv` — see [Tracking](tracking.md).
- **Multi-node DDP:** out of scope for these scripts; use the `scontrol show hostnames` + `torchrun --rdzv-backend=c10d` pattern from the [PyTorch examples](https://github.com/pytorch/examples/tree/main/distributed/ddp-tutorial-series/slurm).
- **Containers:** with Pyxis/Enroot, add `--container-image=...` to `hydra.launcher.additional_parameters`.
