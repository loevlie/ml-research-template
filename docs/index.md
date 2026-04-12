# My Project

Brief description of what this project does.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Train with defaults
python src/my_project/train.py

# Train with overrides
python src/my_project/train.py model.lr=1e-3 data.batch_size=128

# Run a named experiment
python src/my_project/train.py experiment=example

# Multi-seed run
bash scripts/run_seeds.sh experiment=example seeds="42,123,456,789,1337"
```

## Project Structure

See the README for the full directory layout and tool descriptions.
