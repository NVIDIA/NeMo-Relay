"""Lazy import helper for optional NeMo Flow integration.

NeMo Flow is an optional dependency. All functions in this module are safe to
call regardless of whether NeMo Flow is installed — they return ``None`` or
``False`` when the package is not available.
"""

from __future__ import annotations

import asyncio
import contextvars
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from typing import Any

import nemo_flow

# TODO: Move into Nemo flow's python package name it something like integration_utils.py


def get_nemo_flow() -> ModuleType | None:
    """Return the ``nemo_flow`` module, or ``None`` if not installed.

    The import is performed lazily on first call and cached thereafter.
    """
    return nemo_flow  # type: ignore[return-value]


def is_available() -> bool:
    """Return ``True`` if NeMo Flow is installed and importable."""
    return get_nemo_flow() is not None


# ---------------------------------------------------------------------------
# Sync-to-async bridge
# ---------------------------------------------------------------------------


def run_sync(coro: Any) -> Any:
    """Run *coro* synchronously, handling the case where an event loop is
    already running (e.g. Jupyter notebooks).

    When offloading to a ThreadPoolExecutor worker, this helper propagates
    both Python contextvars and the Rust thread-local scope stack so that
    NeMo Flow telemetry is preserved on the worker thread.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No loop running -- we can just use asyncio.run.
        return asyncio.run(coro)
    # Loop already running -- offload to a worker thread so we don't block.
    # Propagate contextvars and scope stack to the worker thread.
    ctx = contextvars.copy_context()
    try:
        scope_stack = nemo_flow.get_scope_stack()

        def _run_with_scope_stack() -> Any:
            nemo_flow.set_thread_scope_stack(scope_stack)
            return asyncio.run(coro)

        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(ctx.run, _run_with_scope_stack).result()
    except Exception:
        pass  # Fall through to vanilla path

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()
