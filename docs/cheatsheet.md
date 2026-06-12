# Cheatsheet

Every common command on one page. `<pkg>` is your package name; drop the
`uv run` prefix if your venv is activated. Everything here works identically
on PyTorch and JAX projects (`trainer.precision`/`devices` are
PyTorch-specific — JAX handles both automatically).

## Train

```bash
uv run python src/<pkg>/train.py                            # base config
uv run python src/<pkg>/train.py experiment=example         # named preset
uv run python src/<pkg>/train.py model.lr=1e-3 seed=123     # override anything
uv run python src/<pkg>/train.py loss=contrastive           # swap a config group
uv run python src/<pkg>/train.py trainer.precision=bf16-mixed
uv run python src/<pkg>/train.py run_dir=outputs/myrun trainer.resume=auto
uv run python src/<pkg>/train.py --help                     # all flags + presets + variants
```

## Evaluate

```bash
uv run python src/<pkg>/eval.py ckpt_path=outputs/<run>/best.ckpt
# training config auto-restored from the run's config.yaml; overrides still win
```

## Seeds & statistics

```bash
bash scripts/run_seeds.sh experiment=ours seeds="42,123,456,789,1337"
sbatch scripts/sbatch_seeds.sh experiment=ours              # same, as a SLURM array
uv run python scripts/aggregate_seeds.py outputs/multi_seed_<stamp> \
    --baseline outputs/multi_seed_baseline --metric val/acc
```

## Sweeps & tuning

```bash
uv run python scripts/sweep.py seed=42,123 model.lr=1e-4,1e-3      # cross product, local
uv run python scripts/sweep.py --cluster slurm --partition gpu \
    seed=42,123 model.lr=1e-4,1e-3                                  # same, one SLURM array
uv run python scripts/tune.py --n-trials 20                         # Optuna, local
uv run python scripts/tune.py --n-trials 64 --workers 8 \
    --cluster slurm --partition gpu                                 # 8 parallel workers
```

## SLURM

```bash
uv sync --extra cluster                                # one-time: submitit
sbatch scripts/sbatch_train.sh experiment=example      # single job
sbatch scripts/sbatch_seeds.sh experiment=example      # seed array
# preempted jobs requeue + resume automatically (run_dir pinned, resume=auto)
```

## Tracking

```bash
uv run python src/<pkg>/train.py logger.kind=wandb       # cloud (uv sync --extra tracking-wandb)
uv run python src/<pkg>/train.py logger.kind=trackio     # local-first; dashboard: uv run trackio show
uv run python src/<pkg>/train.py logger.kind=csv         # zero deps
WANDB_MODE=offline sbatch scripts/sbatch_train.sh ...    # air-gapped nodes; wandb sync later
```

## Tabular flavor

```bash
uv run python src/<pkg>/benchmark.py estimator=logreg task_id=31 fold=0   # one cached cell
bash scripts/run_benchmark.sh                                             # local grid
sbatch scripts/sbatch_benchmark.sh                                        # grid as SLURM array
uv run python scripts/aggregate_benchmark.py outputs/benchmark_<id> --baseline logreg
uv sync --extra tabular-baselines                                         # TabPFN + TabICL
```

## Project hygiene

```bash
uv run pytest                       # smoke tests (overfit-one-batch, init loss, shapes)
uv run ruff check . && uv run ruff format .
uv run mypy src
copier update --trust               # pull template improvements into this project
```

## Where things land

```text
outputs/<date>/<time>/              # or your run_dir=...
├── config.yaml                     # resolved config + git sha + argv (the provenance snapshot)
├── best.ckpt / last.ckpt           # model + optimizer + scheduler + counters
├── metrics.json                    # final metrics (read by aggregate_seeds.py)
└── csv_logs/ | wandb/ | ...        # logger output
```
