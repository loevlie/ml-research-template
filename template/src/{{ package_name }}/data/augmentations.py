"""Data augmentations.

Put sample-level (per-`__getitem__`) transforms here. For batch-level transforms
like MixUp, either write a custom DataLoader `collate_fn` or apply them inside
the objective.

Examples to consider adding (project-specific — keep here, don't promote to
utils unless two projects share one verbatim):

Tabular:
    - SCARF: random column corruption via intra-column resampling
    - SubTab: mask random column subsets
    - Feature dropout: randomly zero a subset of columns
    - MixUp: linear blend of row pairs + labels (batch-level; use collate_fn)

Images:
    - Prefer torchvision.transforms — don't re-implement.

Text:
    - Tokenizer-level augs (synonym replacement, span masking) belong next to
      the tokenizer, not here.
"""
