# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Swap an instance's class in place so the instance keeps its identity.

Each class and method set gets one subclass, and method sets layer in the order they are applied.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from typing import Concatenate, Final, Never, TypeAlias, TypeVar, cast

_T = TypeVar("_T")
Methods: TypeAlias = Mapping[str, Callable[Concatenate[_T, ...], object]]
_Builder: TypeAlias = Callable[[type[_T]], Methods[_T]]
_SWAPPED: Final = "__nemo_relay_swapped__"
_SUBCLASSES: Final[dict[tuple[type[object], _Builder[Never]], type[object]]] = {}
_LOCK: Final = threading.Lock()


def swap_class(instance: _T, methods: _Builder[_T]) -> None:
    """Swap the instance's class for a subclass that adds the methods.

    Leave the instance unchanged when its class already carries the methods. A class that carries other methods gains
    a further subclass, so the methods applied last wrap the ones applied before them.
    """
    cls = type(instance)
    if methods not in _applied(cls):
        instance.__class__ = _subclass(cls, methods)


def is_swapped(instance: object) -> bool:
    return _SWAPPED in type(instance).__dict__


def _applied(cls: type[object]) -> frozenset[_Builder[Never]]:
    return cast("frozenset[_Builder[Never]]", cls.__dict__.get(_SWAPPED, frozenset()))


def _subclass(cls: type[_T], methods: _Builder[_T]) -> type[_T]:
    """Return the one subclass of the class that adds the methods.

    Create the subclass under a lock on the first request from any thread.
    """
    with _LOCK:
        key = (cls, methods)
        if (made := _SUBCLASSES.get(key)) is None:
            namespace = {
                "__module__": cls.__module__,
                "__qualname__": cls.__qualname__,
                "__doc__": cls.__doc__,
                _SWAPPED: _applied(cls) | {methods},
                **methods(cls),
            }
            made = _SUBCLASSES[key] = type(cls)(cls.__name__, (cls,), namespace)
        return cast("type[_T]", made)
