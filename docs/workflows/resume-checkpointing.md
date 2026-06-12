# Checkpoints & resume

## What gets saved

Every epoch, the loop writes two checkpoints to the run directory:

| File | When | Contents |
|---|---|---|
| `best.ckpt` | val loss improved | model, optimizer, scheduler, epoch, best metrics, patience |
| `last.ckpt` | every epoch | same — the resume point |

Both go through `fabric.save`, so they're DDP-safe and load anywhere (`fabric.load` restores modules/optimizers in place).

## Resuming

```bash
# pick up last.ckpt from this run dir if present (no-op otherwise)
uv run python src/<pkg>/train.py run_dir=outputs/myrun trainer.resume=auto

# resume from an explicit checkpoint
uv run python src/<pkg>/train.py trainer.resume=outputs/2026-06-11/10-30-00/last.ckpt
```

Resume restores the epoch counter, best-metric tracking, early-stopping patience, optimizer moments, and scheduler position — training continues as if never interrupted (modulo dataloader order within the epoch).

```text
Resumed from outputs/myrun/last.ckpt (epoch 1)
Epoch   2 | train_loss=2.2795 | val_loss=2.2927 | val_acc=0.0900
```

!!! warning "Keep the scheduler config consistent"
    A checkpoint saved with `scheduler.name=cosine` must be resumed with a scheduler (and vice versa) — the state dict keys are validated strictly.

## SLURM preemption: automatic

The sbatch scripts make resume hands-free by pinning the run dir to the job ID:

```bash
srun torchrun ... src/<pkg>/train.py \
    run_dir="outputs/slurm_${SLURM_JOB_ID}" \
    trainer.resume=auto "$@"
```

Preempted → requeued (same job ID) → same directory → `last.ckpt` found → continue. Granularity is one epoch; if your epochs are hours long, also wire Lightning's `SLURMEnvironment(auto_requeue=True)` for the SIGUSR1-based mid-epoch save (the scripts already send `--signal=B:USR1@90`).

## Evaluating checkpoints

```bash
uv run python src/<pkg>/eval.py ckpt_path=outputs/<run>/best.ckpt
```

`eval.py` passes only `{"model": model}` to `fabric.load`, so it works on any checkpoint regardless of the extra training state inside.
