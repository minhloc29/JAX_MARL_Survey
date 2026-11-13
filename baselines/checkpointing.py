"""Shared checkpoint support for JAX baseline training loops."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Callable

import jax
import numpy as np
from flax import serialization


def _write_checkpoint(path: str, state: Any, update: int) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = serialization.to_bytes({"update": int(update), "state": state})
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def save_final_checkpoint(path: str | None, state: Any, update: int) -> None:
    """Save a final runner state on the host after a training run."""
    if path:
        _write_checkpoint(path, jax.device_get(state), update)


def checkpointed_scan(
    body: Callable[[Any, Any], tuple[Any, Any]],
    init: Any,
    *,
    length: int,
    path: str | None,
    interval: int,
) -> tuple[Any, Any]:
    """Run a scan and atomically save state at intervals and at the final update."""
    if not path:
        return jax.lax.scan(body, init, None, length)
    if interval < 1:
        raise ValueError("checkpoint interval must be positive")

    treedef = jax.tree_util.tree_structure(init)

    def save_callback(*values: Any) -> None:
        state = jax.tree_util.tree_unflatten(treedef, values[:-1])
        _write_checkpoint(path, state, int(np.asarray(values[-1])))

    def wrapped(carry: Any, update: Any) -> tuple[Any, Any]:
        carry, output = body(carry, update)
        update_number = update + 1
        should_save = (update_number % interval == 0) | (update_number == length)

        def save(_: None) -> None:
            jax.debug.callback(
                save_callback,
                *jax.tree_util.tree_leaves(carry),
                update_number,
                ordered=True,
            )

        jax.lax.cond(should_save, save, lambda _: None, operand=None)
        return carry, output

    return jax.lax.scan(wrapped, init, jax.numpy.arange(length))


def load_checkpoint(path: str, template: Any) -> tuple[Any, int]:
    """Restore a checkpoint using a compatible runner-state template."""
    with open(path, "rb") as file:
        payload = serialization.from_bytes(
            {"update": np.int32(0), "state": template}, file.read()
        )
    return payload["state"], int(payload["update"])
