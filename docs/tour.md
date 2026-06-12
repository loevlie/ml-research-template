# Hands-on tour

The [tutorial](tutorial.md) ended with a number. This tour exercises
everything around it — sweeps, Optuna, crash recovery, tracker swapping, and
a paired significance test — in the same wine-quality project, ~20 minutes,
all on a laptop. Every output below is from a real run.

## 1 · Sweep a grid

Comma-separated values fan out as a cross product, each point in its own run
directory:

```bash
uv run python scripts/sweep.py experiment=wine model.lr=1e-4,1e-3 seed=42,123 trainer.max_epochs=10
grep -r "val/acc" outputs/sweep_*/run_*/metrics.json
```

```text
run_000_34788291:  "val/acc": 0.583    # lr=1e-4 seed=42
run_001_14355ae5:  "val/acc": 0.580    # lr=1e-4 seed=123
run_002_578de15c:  "val/acc": 0.624    # lr=1e-3 seed=42
run_003_76207663:  "val/acc": 0.621    # lr=1e-3 seed=123
```

Two seeds per point already tell you something a single run can't: the
lr=1e-3 advantage (~0.04) is bigger than the seed wobble (~0.006) — worth
pursuing. Add `--cluster slurm` and this exact command becomes a job array.

## 2 · Let Optuna search

The search space lives in `suggest()` inside `scripts/tune.py` — edit it
like code, then:

```bash
uv run python scripts/tune.py --n-trials 5
```

```text
tune: best value 0.6552 with {'model.lr': 9.31e-03, 'model.hidden_dim': 512,
'model.weight_decay': 1.26e-05, 'data.batch_size': 64}
```

Five trials and it beat both of our hand-picked configs — with a *wider*
model than we tried. Workers share the study through a journal file,
so `--workers 8 --cluster slurm` parallelizes it with no database server.

## 3 · Kill it, resume it

Pin a run directory and ask for auto-resume, then kill the run mid-training
(`Ctrl-C`, or a SLURM preemption — same thing):

```bash
uv run python src/wine_quality/train.py experiment=wine \
    run_dir=outputs/long trainer.resume=auto trainer.max_epochs=300
# ... training ... ^C
```

Run the **same command again**:

```text
Resumed from outputs/long/last.ckpt (epoch 51)
Epoch  52 | train_loss=0.5834 | val_loss=0.9894 | val_acc=0.6395
Epoch  53 | train_loss=0.5810 | val_loss=0.9742 | val_acc=0.6364
```

The loss continues falling smoothly because `last.ckpt` carries the
optimizer moments and scheduler position, not just weights. On SLURM the
sbatch scripts wire this up for you — preempted jobs heal themselves
([how](workflows/slurm.md)).

## 4 · Swap the tracker

```bash
uv sync --extra tracking-trackio
uv run python src/wine_quality/train.py experiment=wine logger.kind=trackio
```

```text
* Trackio project initialized: wine_quality
* Trackio metrics logged to: ~/.cache/huggingface/trackio
```

`uv run trackio show` opens a local dashboard — no account, no cloud. The
training code didn't change; trackers are one config key
([tracking](workflows/tracking.md)).

## 5 · The honest finale

Is the `wine` preset (lr=1e-3) actually better than the default config at
the same budget? Run both on the **same seeds** and let the statistics
answer:

```bash
bash scripts/run_seeds.sh experiment=wine seeds="42,123,456,789,1337"
bash scripts/run_seeds.sh experiment=base trainer.max_epochs=30 seeds="42,123,456,789,1337"
uv run python scripts/aggregate_seeds.py outputs/multi_seed_<wine> \
    --baseline outputs/multi_seed_<base> --metric val/acc
```

```text
--- Paired Comparison (shared seeds: [42, 123, 456, 789, 1337]) ---
Ours:     0.6219 +/- 0.0322
Baseline: 0.6169 +/- 0.0165
Delta:    +0.0050
Test:     paired t-test (fallback, n<6 for Wilcoxon)
Stat:     0.6610, p=0.5448
Effect:   Cohen's d = 0.196
NOT significant at p<0.05
```

A +0.5% delta that **doesn't survive the test** — at this budget, our
hand-tuned lr is seed noise, and the machinery just stopped us from
claiming otherwise. (Note the t-test fallback firing at 5 seeds — that's
the `2/2^N` floor from [Statistics, explained](workflows/stats-explained.md).)

Your move: promote Optuna's winner from step 2 into `experiments.py` and
re-run this comparison with 8 seeds. That's not an exercise anymore —
that's research, and now the whole pipeline is muscle memory.
