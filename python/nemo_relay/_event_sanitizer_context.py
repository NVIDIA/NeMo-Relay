# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Track queued publication callbacks on their originating event loop."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import Any


class _CallbackState:
    """Shared liveness for contexts copied from one publication callback."""

    __slots__ = ("active",)

    def __init__(self) -> None:
        self.active = True


_ACTIVE: ContextVar[_CallbackState | None] = ContextVar("nemo_relay_event_sanitizer_active", default=None)


def callback_active() -> bool:
    """Return whether the current context is running queued publication work."""
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


def loop_affine(callback: Callable[..., Any], *, sanitizer: bool = False) -> Callable[..., Awaitable[Any]]:
    """Defer a callback's synchronous prelude to the awaiting event-loop task."""

    async def wrapped(*args: Any) -> Any:
        result = invoke(callback, *args) if sanitizer else callback(*args)
        if inspect.isawaitable(result):
            return await result
        return result

    return wrapped


async def async_iter_next(iterator: Any) -> Any:
    """Invoke and await ``__anext__`` on the current event-loop thread."""
    return await iterator.__anext__()


async def async_iter_close(iterator: Any) -> None:
    """Invoke and await ``aclose`` on the current event-loop thread when present."""
    close = getattr(iterator, "aclose", None)
    if close is not None:
        await close()


def invoke(callback: Callable[..., Any], *args: Any) -> Any:
    """Invoke queued publication work while marking its execution context."""
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
