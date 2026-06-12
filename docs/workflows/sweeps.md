# Sweeps & HP search

## Grid sweeps: `scripts/sweep.py`

Comma-separated values fan out as a cross product (the `-m` multirun replacement):

```bash
uv run python scripts/sweep.py seed=42,123,456 model.lr=1e-4,1e-3        # 6 runs, local
uv run python scripts/sweep.py experiment=example loss=contrastive seed=1,2,3
```

Add `--cluster slurm` and the same sweep becomes **one SLURM job array** via [submitit](https://github.com/facebookincubator/submitit) (`uv sync --extra cluster`):

```bash
uv run python scripts/sweep.py --cluster slurm --partition gpu --gpus-per-node 1 \
    seed=42,123,456 model.lr=1e-4,1e-3
```

Every point runs with a pinned `run_dir` and `trainer.resume=auto`, so preempted jobs requeue into the same directory and continue from `last.ckpt`.

!!! note "Seed sweeps"
    For seeds specifically, prefer `scripts/run_seeds.sh` / `scripts/sbatch_seeds.sh` — they write the `seed_<s>` layout the [significance-test aggregator](multi-seed-stats.md) expects.

## Bayesian search: `scripts/tune.py`

Optuna drives the search; the space lives in `suggest()` inside the script:

```python title="scripts/tune.py (edit me)"
def suggest(trial):
    return {
        "model": {
            "lr": trial.suggest_float("model.lr", 1e-5, 1e-2, log=True),
            "hidden_dim": trial.suggest_categorical("model.hidden_dim", [64, 128, 256, 512]),
        },
    }
```

```bash
uv run python scripts/tune.py --n-trials 20                              # local, sequential
uv run python scripts/tune.py --n-trials 64 --workers 8 \
    --cluster slurm --partition gpu                                      # 8 parallel SLURM workers
```

Workers coordinate through Optuna's **JournalFileBackend** on the shared filesystem — no database server, and no SQLite-over-NFS corruption (Optuna's docs explicitly rule that out). Trials run in-process (`tune.py` imports `train.run`), so each trial is a normal run directory under `outputs/tune_<study>/`.

```text
tune: best value 0.0850 with {'model.lr': 0.00126, 'model.hidden_dim': 128, ...}
```

## After the search

Promote the winner into a committed preset (`experiments.py`) rather than re-reading it from sweep logs — that's the version of record, and the multi-seed run for the paper starts from there.
