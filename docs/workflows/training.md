# Train & experiment

## The moving parts

```
configs/train.yaml        # composition root: which data/model/trainer/loss/logger to use
src/<pkg>/train.py        # thin orchestrator: seed → fabric → data → model → loop
src/<pkg>/training_loop.py# train_epoch() + validate() — plain functions, no Hydra inside
src/<pkg>/objectives.py   # loss + forward bundled as a swappable callable
```

`train.py` builds everything from config and calls the loop; the loop never parses config, so it's testable with plain arguments. Read those two files once — they're short by design.

## Run things

```bash
uv run python src/<pkg>/train.py                          # defaults
uv run python src/<pkg>/train.py model.lr=1e-3 seed=123   # override anything
uv run python src/<pkg>/train.py experiment=example       # named experiment
uv run python src/<pkg>/train.py trainer.precision=bf16-mixed trainer.max_epochs=200
```

Each run writes to `outputs/<date>/<time>/`: resolved config (`.hydra/`), `best.ckpt`, `last.ckpt`, `metrics.json` (consumed by the [multi-seed aggregator](multi-seed-stats.md)), and logger output.

## Experiments as files

An experiment is a small YAML overriding the defaults — version-controlled, so every paper number traces to a commit:

```yaml title="configs/experiment/wider_net.yaml"
# @package _global_
model:
  hidden_dim: 512
  lr: 1e-3
trainer:
  max_epochs: 50
```

```bash
uv run python src/<pkg>/train.py experiment=wider_net
```

## Swap the objective

The loop calls `objective(model, batch)` and uses the returned `"loss"` — nothing else. To add a self-supervised / contrastive / masked objective:

```python title="src/<pkg>/objectives.py"
class MaskedColumnObjective:
    def __init__(self, mask_ratio: float = 0.15) -> None:
        self.mask_ratio = mask_ratio

    def __call__(self, model, batch):
        x, _ = batch
        x_masked, target, mask = mask_columns(x, self.mask_ratio)
        loss = F.mse_loss(model(x_masked)[mask], target[mask])
        return {"loss": loss}   # no logits/targets → val accuracy auto-skips
```

```yaml title="configs/loss/masked_column.yaml"
_target_: <pkg>.objectives.MaskedColumnObjective
mask_ratio: 0.15
```

```bash
uv run python src/<pkg>/train.py loss=masked_column
```

A CLIP-style `ContrastiveObjective` ships in every project as a worked example (`loss=contrastive`).

## Trainer knobs

```yaml title="configs/trainer/default.yaml"
precision: 32-true            # bf16-mixed for modern GPUs
max_epochs: 100
patience: 10                  # early stopping
accumulate_grad_batches: 1    # emulate larger batches
gradient_clip_val: 1.0        # max grad norm per step; null disables
resume: null                  # auto | /path/to/last.ckpt — see Checkpoints & resume
scheduler:
  name: none                  # cosine | constant — warmup + per-step stepping
  warmup_steps: 0
```

## Evaluate a checkpoint

```bash
uv run python src/<pkg>/eval.py ckpt_path=outputs/<run>/best.ckpt
```

## Built-in sanity tests

`uv run pytest` runs the research smoke tests: the model overfits a single batch, the init loss matches `-log(1/n_classes)`, shapes are right, loaders split reproducibly. Keep these passing — they catch the boring bugs that eat GPU-days.
