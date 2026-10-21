# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Swap an instance's class in place so the instance keeps its identity.

Each original class and method set gets one subclass.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Final, TypeVar, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

_T = TypeVar("_T")
_SWAPPED: Final = "__nemo_relay_swapped__"
_SUBCLASSES: Final[dict[tuple[type[object], Callable[..., Mapping[str, object]]], type[object]]] = {}
_LOCK: Final = threading.Lock()


def swap_class(instance: _T, methods: Callable[[type[_T]], Mapping[str, object]]) -> None:
    """Swap the instance's class for a subclass that carries the methods.

    Leave an instance whose class is already a swapped subclass unchanged.
    """
    cls = type(instance)
    if _SWAPPED not in cls.__dict__:
        instance.__class__ = _subclass(cls, methods)


def is_swapped(instance: object) -> bool:
    return _SWAPPED in type(instance).__dict__


def _subclass(cls: type[_T], methods: Callable[[type[_T]], Mapping[str, object]]) -> type[_T]:
    """Return the one subclass of the class that carries the methods.

    Create the subclass under a lock on the first request from any thread.
    """
    with _LOCK:
        key = (cls, methods)
        if (made := _SUBCLASSES.get(key)) is None:
            namespace = {
                "__module__": cls.__module__,
                "__qualname__": cls.__qualname__,
                "__doc__": cls.__doc__,
                _SWAPPED: True,
                **methods(cls),
            }
            made = _SUBCLASSES[key] = type(cls)(cls.__name__, (cls,), namespace)
        return cast("type[_T]", made)
