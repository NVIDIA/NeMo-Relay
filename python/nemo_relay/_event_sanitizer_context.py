# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Track re-entrant subscriber flushes from Python event sanitizers."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import Any

_ACTIVE: ContextVar[bool] = ContextVar("nemo_relay_event_sanitizer_active", default=False)


def callback_active() -> bool:
    """Return whether the current Python context is running an event sanitizer."""
    return _ACTIVE.get()


async def _await_result(result: Awaitable[Any]) -> Any:
    token = _ACTIVE.set(True)
    try:
        return await result
    finally:
        _ACTIVE.reset(token)


async def await_result(result: Awaitable[Any]) -> Any:
    """Await an arbitrary awaitable without changing sanitizer context."""
    return await result


def invoke(callback: Callable[..., Any], *args: Any) -> Any:
    """Invoke a sanitizer while marking its sync and async execution contexts."""
    token = _ACTIVE.set(True)
    try:
        result = callback(*args)
    finally:
        _ACTIVE.reset(token)
    if inspect.isawaitable(result):
        return _await_result(result)
    return result
