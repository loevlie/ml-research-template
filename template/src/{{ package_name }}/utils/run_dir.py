"""Run-directory management: timestamped dirs + resolved-config snapshots.

The two things Hydra's output management actually bought us, in ~60 lines:
every run gets its own directory, and the directory is self-describing —
config.yaml inside it records the fully-resolved config, the git state, and
the exact command line, so any result can be reproduced months later.

Reusing a run_dir (the resume flow) never clobbers the original snapshot:
identical configs are a quiet no-op; a changed config keeps the original
config.yaml, warns loudly, and records the new invocation as
config_resume<N>.yaml so provenance survives.
"""

import datetime
import subprocess
import sys
from pathlib import Path

import pydantic
import yaml


def _git_state() -> dict:
    try:
        sha = (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
            ).stdout.strip()
            or None
        )
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5
            ).stdout.strip()
        )
        return {"git_sha": sha, "git_dirty": dirty}
    except Exception:
        return {"git_sha": None, "git_dirty": False}


def _snapshot(cfg: pydantic.BaseModel) -> dict:
    return {"config": cfg.model_dump(mode="json"), **_git_state(), "argv": sys.argv}


def create_run_dir(
    cfg: pydantic.BaseModel, run_dir: str | Path | None, root: str = "outputs"
) -> Path:
    """Create (or reuse) the run directory and snapshot the resolved config.

    Args:
        cfg: The fully-resolved config to snapshot.
        run_dir: Explicit directory (pin this for SLURM requeue + resume);
            None creates ``outputs/<date>/<time>``.
        root: Parent for auto-created directories.

    Returns:
        The run directory path.
    """
    if run_dir is None:
        stamp = datetime.datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
        path = Path(root) / stamp
        suffix = 0
        while True:  # mkdir is the atomic claim — same-second launches retry
            try:
                path.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                suffix += 1
                path = Path(root) / f"{stamp}-{suffix}"
    else:
        path = Path(run_dir)
        path.mkdir(parents=True, exist_ok=True)

    snapshot = _snapshot(cfg)
    snapshot_path = path / "config.yaml"
    if snapshot_path.exists():
        existing = yaml.safe_load(snapshot_path.read_text()) or {}
        if existing.get("config") == snapshot["config"]:
            return path  # same config (a resume) — keep the original provenance
        n = 1
        while (path / f"config_resume{n}.yaml").exists():
            n += 1
        (path / f"config_resume{n}.yaml").write_text(yaml.safe_dump(snapshot, sort_keys=False))
        changed = sorted(
            key
            for key in set(existing.get("config", {})) | set(snapshot["config"])
            if existing.get("config", {}).get(key) != snapshot["config"].get(key)
        )
        print(
            f"WARNING: reusing {path} with a DIFFERENT config (changed: {', '.join(changed)}) — "
            f"original config.yaml kept, this invocation recorded as config_resume{n}.yaml",
            file=sys.stderr,
        )
        return path

    snapshot_path.write_text(yaml.safe_dump(snapshot, sort_keys=False))
    return path
