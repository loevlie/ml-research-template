# Configure with Hydra

## Layout

```
configs/
├── train.yaml          # composition root
├── eval.yaml
├── data/default.yaml       # one file per option in each group
├── model/default.yaml
├── trainer/default.yaml
├── loss/{supervised,contrastive}.yaml
├── logger/{wandb,trackio,tensorboard,csv}.yaml
├── experiment/         # named experiments (version-controlled)
├── hparams_search/optuna.yaml
├── launcher/{slurm,slurm_debug}.yaml
└── local/default.yaml      # machine-specific, gitignored
```

`train.yaml` declares which member of each group to compose:

```yaml
defaults:
  - _self_
  - data: default
  - model: default
  - trainer: default
  - loss: supervised
  - logger: wandb
  - experiment: null      # opt-in overrides
  - local: default        # always last → machine-specific wins
```

## The three override levels

1. **Group swap** — `loss=contrastive`, `logger=csv`: replaces a whole file.
2. **Key override** — `model.lr=1e-3`, `trainer.max_epochs=5`: surgical, from the CLI.
3. **Experiment file** — `experiment=wider_net`: a named, committed bundle of overrides. Use this for anything you might put in a paper.

## Interpolation

Configs reference each other, so shared values live in one place:

```yaml title="configs/model/default.yaml"
n_features: ${data.n_features}   # stays in sync with the dataset config
n_classes: ${data.n_classes}
```

## `local/` — machine-specific, gitignored

Anything that differs between your laptop and the cluster (worker counts, SLURM account, data roots) goes in `configs/local/default.yaml`. It composes last, wins over everything, and never pollutes the repo:

```yaml title="configs/local/default.yaml"
data:
  num_workers: 0   # laptop
```

## Reproducibility

Hydra snapshots the **resolved** config of every run into `outputs/<run>/.hydra/config.yaml` — six months later you can re-run any result exactly:

```bash
uv run python src/<pkg>/train.py --config-dir outputs/<run>/.hydra --config-name config
```

!!! tip "Debugging composition"
    `uv run python src/<pkg>/train.py --cfg job` prints the composed config without running anything. `--info defaults` shows where each value came from.
