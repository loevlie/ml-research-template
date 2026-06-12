# JAX

The template can generate a JAX project instead of a PyTorch one:

```bash
copier copy --trust -d framework=jax gh:loevlie/ml-research-template my-project
```

You get flax NNX for the model, optax for the optimizer, and the same
everything-else: the typed configs, the `key=value` CLI, presets,
`configs/local.yaml`, sweeps, Optuna, the SLURM scripts, multi-seed stats,
and run-directory provenance are shared code — every command in the
[cheatsheet](../cheatsheet.md) works unchanged. JAX pairs with the generic
flavor; the tabular and multimodal flavors build on PyTorch libraries
(TabPFN, timm, open_clip) and stay PyTorch.

## What's different, and why it's nice

**The model takes its randomness explicitly.** No global seed state — the
init key is an argument, which is why JAX runs are reproducible by
construction:

```python
model = ExampleModel(n_features=32, hidden_dim=128, n_classes=10, rngs=nnx.Rngs(seed))
```

**Three trainer knobs become optimizer composition.** In the torch loop,
LR scheduling, gradient clipping, and gradient accumulation are loop code.
In JAX they compose into the optax chain (`train.py`), and the loop just
applies updates — same config fields, less loop to maintain:

```python
tx = optax.adamw(schedule, weight_decay=cfg.model.weight_decay)
if cfg.trainer.gradient_clip_val:                      # trainer.gradient_clip_val=1.0
    tx = optax.chain(optax.clip_by_global_norm(cfg.trainer.gradient_clip_val), tx)
if accumulate > 1:                                     # trainer.accumulate_grad_batches=4
    tx = optax.MultiSteps(tx, every_k_schedule=accumulate)
optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
```

**The training step is one jitted function.** XLA compiles it on the first
batch; for small models trained many steps (the tabular regime), this is
where JAX outruns eager PyTorch:

```python
@nnx.jit
def train_step(model, optimizer, batch):
    def loss_fn(model):
        out = objective(model, batch)          # same Objective protocol as torch
        return out["loss"], out

    (loss, _), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return loss, optax.global_norm(grads)
```

**Checkpoints are msgpack files of plain arrays** (`training_loop.py`) —
`best.ckpt`/`last.ckpt` every epoch, `trainer.resume=auto`, and SLURM
preemption recovery all behave exactly like the torch loop:

```text
Epoch   0 | train_loss=2.3743 | val_loss=2.3773 | val_acc=0.0950
Epoch   1 | train_loss=2.3152 | val_loss=2.3691 | val_acc=0.0950
# rerun with trainer.resume=auto →
Resumed from outputs/j1/last.ckpt (epoch 1)
Epoch   2 | train_loss=2.2817 | val_loss=2.3671 | val_acc=0.1000
```

The loss keeps falling smoothly across the restart because the optimizer
state (Adam moments, accumulation buffers, schedule position) is in the
checkpoint too.

## Choosing between the frameworks

Pick **JAX** when the work is small-model/many-step (XLA fusion pays),
when you have TPU access, or when the research itself is transform-shaped —
`vmap` over seeds or ensemble members, per-example gradients, meta-learning.
Pick **PyTorch** when you need the ecosystem: pretrained baselines,
domain libraries, and artifacts other people can run. The longer version is
in [The stack](../stack.md#pytorch-by-default-jax-by-choice).

!!! note "Single-device by design"
    The JAX loop runs on one device (CPU/GPU/TPU) — no `pmap`/`shard_map`
    scaffolding. When a project genuinely needs sharding, reach for
    `jax.shard_map` and switch checkpoints to orbax; `save_checkpoint`/
    `load_checkpoint` in `training_loop.py` are the only two functions
    that change.
