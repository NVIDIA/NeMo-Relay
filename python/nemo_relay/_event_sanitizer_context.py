# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Track re-entrant subscriber flushes from Python event sanitizers."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import Any


class _CallbackState:
    """Shared liveness for contexts copied from one sanitizer invocation."""

    __slots__ = ("active",)

    def __init__(self) -> None:
        self.active = True


_ACTIVE: ContextVar[_CallbackState | None] = ContextVar("nemo_relay_event_sanitizer_active", default=None)


def callback_active() -> bool:
    """Return whether the current Python context is running an event sanitizer."""
    state = _ACTIVE.get()
    return state is not None and state.active


async def _await_result(result: Awaitable[Any], state: _CallbackState, owner: bool) -> Any:
    token = _ACTIVE.set(state)
    try:
        return await result
    finally:
        if owner:
            state.active = False
        _ACTIVE.reset(token)


async def await_result(result: Awaitable[Any]) -> Any:
    """Await an arbitrary awaitable without changing sanitizer context."""
    return await result


def invoke(callback: Callable[..., Any], *args: Any) -> Any:
    """Invoke a sanitizer while marking its sync and async execution contexts."""
    state = _ACTIVE.get()
    owner = state is None or not state.active
    if owner:
        state = _CallbackState()
    assert state is not None
    token = _ACTIVE.set(state)
    try:
        result = callback(*args)
    except BaseException:
        if owner:
            state.active = False
        raise
    finally:
        _ACTIVE.reset(token)
    if inspect.isawaitable(result):
        return _await_result(result, state, owner)
    if owner:
        state.active = False
    return result
