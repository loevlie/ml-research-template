# Run on SLURM

Two complementary paths ship with every project: Python launchers (submitit) for sweeps and HP search, and plain sbatch scripts for everything else.

## Path 1 — submitit launchers (sweeps & tuning)

```bash
uv sync --extra cluster        # one-time: submitit

# a cross-product sweep as one SLURM job array
uv run python scripts/sweep.py --cluster slurm --partition gpu seed=42,43,44 model.lr=1e-4,1e-3

# 8 parallel Optuna workers sharing one study
uv run python scripts/tune.py --cluster slurm --partition gpu --workers 8 --n-trials 64
```

Pass `--account`, `--gpus-per-node`, `--mem-gb`, `--timeout-min` as needed — see `--help`. Details in [Sweeps & HP search](sweeps.md).

## Path 2 — sbatch scripts (no extra deps, fully inspectable)

```bash
sbatch scripts/sbatch_train.sh experiment=example     # single job, single/multi-GPU via torchrun
sbatch scripts/sbatch_seeds.sh experiment=example     # array job: one seed per task
```

Both assume `.venv/` exists on a shared filesystem (`uv sync --extra dev --extra cluster` on the login node). Extra args are forwarded to `train.py` in the same `key=value` syntax.

## Preemption survival — built in

Both paths pin each run's directory and pass `trainer.resume=auto`:

```bash
run_dir="outputs/slurm_${SLURM_JOB_ID}" trainer.resume=auto
```

`--requeue` means a preempted job is resubmitted **with the same job ID** → it lands in the same directory → `resume=auto` picks up `last.ckpt` (saved every epoch) and continues from the last finished epoch. No babysitting. Details in [Checkpoints & resume](resume-checkpointing.md).

## Tabular flavor: array benchmarks

Tabular-FM evaluation is hundreds of small CPU jobs, not one big GPU run. `scripts/sbatch_benchmark.sh` maps an estimator × task × fold grid onto one SLURM array, and exca's cache means a re-submitted grid recomputes only missing cells — see [Tabular benchmarking](tabular-benchmarking.md).

## Cluster notes

- **Logger on air-gapped compute nodes:** `WANDB_MODE=offline` + `wandb sync` from the login node, or `logger.kind=trackio` / `logger.kind=csv` — see [Tracking](tracking.md).
- **Multi-node DDP:** out of scope for these scripts; use the `scontrol show hostnames` + `torchrun --rdzv-backend=c10d` pattern from the [PyTorch examples](https://github.com/pytorch/examples/tree/main/distributed/ddp-tutorial-series/slurm).
- **Containers:** with Pyxis/Enroot, add the container flags to the sbatch scripts directly — they're plain bash.
