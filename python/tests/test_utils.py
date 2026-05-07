# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Python utility helpers."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

from nemo_flow.utils import run_sync


def test_run_sync_allows_concurrent_running_loop_callers():
    """Concurrent running-loop callers can make progress through the shared pool."""
    barrier = threading.Barrier(2)

    def call_run_sync(value: int) -> int:
        async def inner() -> int:
            barrier.wait(timeout=2)
            return value

        async def outer() -> int:
            return run_sync(inner())

        return asyncio.run(outer())

    with ThreadPoolExecutor(max_workers=2) as callers:
        futures = [callers.submit(call_run_sync, value) for value in range(2)]
        results = [future.result(timeout=5) for future in futures]

    assert sorted(results) == [0, 1]
