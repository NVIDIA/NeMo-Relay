# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Global event metadata injector registration.

Injectors receive an immutable ``ScopeEvent`` or ``MarkEvent`` and return
metadata additions. Relay validates and merges accepted additions before the
event sanitizer chain runs.
"""

from __future__ import annotations

from nemo_relay import EventMetadataInjectorCallback
from nemo_relay._native import (
    deregister_event_metadata_injector as _deregister_event_metadata_injector,
)
from nemo_relay._native import (
    register_event_metadata_injector as _register_event_metadata_injector,
)


def register_injector(name: str, priority: int, injector: EventMetadataInjectorCallback) -> None:
    """Register a global event metadata injector.

    The registration applies to every subsequently published canonical Relay
    event until it is deregistered. Lower numeric priorities run first.

    Args:
        name: Unique registration name for the injector.
        priority: Injector execution order; lower values run first.
        injector: Callback that returns metadata additions for each event.

    Returns:
        None: The injector is registered for future published events.
    """
    _register_event_metadata_injector(name, priority, injector)


def deregister_injector(name: str) -> bool:
    """Remove a global event metadata injector by registration name.

    Args:
        name: Registration name previously passed to ``register_injector``.

    Returns:
        bool: ``True`` if an injector was removed; otherwise ``False``.
    """
    return _deregister_event_metadata_injector(name)


__all__ = ["register_injector", "deregister_injector"]
