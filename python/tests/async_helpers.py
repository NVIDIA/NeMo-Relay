# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Async helpers shared by Python tests."""

from __future__ import annotations

from collections.abc import Awaitable
from typing import TypeVar, cast

from typing_extensions import TypeIs

T = TypeVar("T")


async def resolve_async_result(value: T | Awaitable[T]) -> T:
    """Resolve a value that Relay may return directly or through an awaitable."""
    if _is_awaitable(value):
        return cast(T, await value)
    return cast(T, value)


def _is_awaitable(value: T | Awaitable[T]) -> TypeIs[Awaitable[T]]:
    """Narrow a Relay helper's direct-or-awaitable return contract."""
    return isinstance(value, Awaitable)
