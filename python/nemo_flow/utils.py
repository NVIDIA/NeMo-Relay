# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextvars
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import nemo_flow


# ---------------------------------------------------------------------------
# Sync-to-async bridge
# ---------------------------------------------------------------------------
def run_sync(coro: Any) -> Any:
    """Run *coro* synchronously, handling the case where an event loop is
    already running.

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
