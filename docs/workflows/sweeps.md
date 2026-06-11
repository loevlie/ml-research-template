# Sweeps & HP search

## Grid sweeps with multirun

Hydra's `-m` flag turns comma-separated values into a job per combination:

```bash
uv run python src/<pkg>/train.py -m model.lr=1e-4,3e-4,1e-3
uv run python src/<pkg>/train.py -m model.lr=1e-4,1e-3 model.hidden_dim=128,256   # 4 jobs
uv run python src/<pkg>/train.py -m seed=42,123,456,789,1337                       # seeds (but see below)
```

Results land in `multirun/<date>/<time>/<job_idx>/`. Add `launcher=slurm` and the same sweep fans out as parallel cluster jobs — see [Run on SLURM](slurm.md).

!!! note "Seed sweeps"
    For seeds specifically, prefer `scripts/run_seeds.sh` / `scripts/sbatch_seeds.sh` — they write the directory layout the [significance-test aggregator](multi-seed-stats.md) expects.

## Bayesian search with Optuna

`configs/hparams_search/optuna.yaml` defines the search space; `train.py` returns the validation metric for Optuna to maximize:

```bash
uv run python src/<pkg>/train.py -m hparams_search=optuna
```

```yaml title="configs/hparams_search/optuna.yaml (excerpt)"
hydra:
  sweeper:
    direction: maximize
    n_trials: 50
    params:
      model.lr:
        type: float
        low: 1e-5
        high: 1e-2
        log: true
```

Edit `params` to match your model. Trials run sequentially by default (`n_jobs: 1`); raise it if you have the GPUs.

!!! warning "Optuna across SLURM jobs"
    Composing `hparams_search=optuna launcher=slurm` parallelizes trials as cluster jobs, but they must share a storage backend to coordinate: set `hydra.sweeper.storage: sqlite:////shared/fs/optuna.db`. Without it, an N-trial sweep degenerates into N independent random draws.

## After the search

Promote the winner into a committed experiment file (`configs/experiment/best_hp.yaml`) rather than re-reading it from sweep logs — that's the version of record, and the multi-seed run for the paper starts from there.
