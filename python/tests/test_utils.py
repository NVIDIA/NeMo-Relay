# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Python utility helpers."""

import asyncio

import pytest

import nemo_flow
from nemo_flow.utils import run_sync


@pytest.mark.parametrize("from_async", [False, True])
def test_run_sync(from_async: bool):
    """
    Test that run_sync correctly propagates the NeMo Flow scope stack to the worker thread,
    and that it can be called from inside a running loop and outside a running loop.
    """
    scope_stack = nemo_flow.get_scope_stack()
    assert scope_stack is not None

    async def coro_fn() -> int:
        thread_scope_stack = nemo_flow.get_scope_stack()
        assert thread_scope_stack is scope_stack
        return 1

    if from_async:

        async def runner() -> int:
            return run_sync(coro_fn())

        assert asyncio.run(runner()) == 1
    else:
        assert run_sync(coro_fn()) == 1
