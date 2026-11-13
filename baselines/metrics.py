"""W&B logging helpers that preserve environment-provided metric names."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import wandb


def log_metrics(metrics: Mapping[str, Any], *, step: int | None = None) -> None:
    """Log metrics without renaming or synthesizing metric fields."""
    wandb.log(dict(metrics), step=step)
