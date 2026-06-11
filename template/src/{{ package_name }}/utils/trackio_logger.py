"""Lightning Fabric logger for trackio.

trackio (https://github.com/gradio-app/trackio) is HuggingFace's local-first
experiment tracker with a wandb-compatible API: runs are stored in a local
SQLite db and viewed with `trackio show`, no account needed. Optionally syncs
dashboards to a HF Space via `space_id`.

Fabric ships no trackio logger, so this adapter implements the small
`lightning.fabric.loggers.Logger` interface on top of trackio's run API.
"""

from argparse import Namespace
from typing import Any

from lightning.fabric.loggers.logger import Logger
from lightning.fabric.utilities.rank_zero import rank_zero_only


class TrackioLogger(Logger):
    """Log metrics to a local trackio project.

    Args:
        project: trackio project name (groups runs in the dashboard).
        name: Optional run name (trackio auto-generates one if omitted).
        **init_kwargs: Forwarded to ``trackio.init`` (e.g. ``space_id`` to
            sync the dashboard to a HF Space, ``group`` to group runs).
    """

    def __init__(self, project: str, name: str | None = None, **init_kwargs: Any) -> None:
        super().__init__()
        self._project = project
        self._run_name = name
        self._init_kwargs = init_kwargs
        self._config: dict[str, Any] = {}
        self._run: Any = None

    @property
    def name(self) -> str:
        return self._project

    @property
    def version(self) -> str | None:
        return self._run_name

    @property
    def experiment(self) -> Any:
        """The underlying trackio Run, created lazily on first log."""
        if self._run is None:
            import trackio  # local import: trackio is an optional dependency

            self._run = trackio.init(
                project=self._project,
                name=self._run_name,
                config=self._config or None,
                **self._init_kwargs,
            )
        return self._run

    @rank_zero_only
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.experiment.log(dict(metrics), step=step)

    @rank_zero_only
    def log_hyperparams(
        self, params: dict[str, Any] | Namespace, *args: Any, **kwargs: Any
    ) -> None:
        # trackio takes config at run creation; stash params until then.
        if isinstance(params, Namespace):
            params = vars(params)
        self._config.update(params)

    @rank_zero_only
    def finalize(self, status: str) -> None:
        if self._run is not None:
            self._run.finish()
