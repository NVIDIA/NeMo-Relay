# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private helpers for synchronizing Python and native scope context."""

from __future__ import annotations


def ensure_scope_stack() -> None:
    """Ensure the current Python context's stack is active in the native runtime.

    A ``ContextVar`` owns the stack for asyncio tasks. Native code uses a
    thread-local fallback outside its Tokio tasks, so concurrent Python tasks
    must re-synchronize that fallback before each context-sensitive call.
    Explicit worker-thread bindings remain untouched when no ``ContextVar`` is
    present.
    """
    import nemo_relay

    stack = nemo_relay._scope_stack_var.get(None)
    if stack is not None:
        nemo_relay._sync_thread_scope_stack(stack)
        return

    if nemo_relay._native_scope_stack_active():
        return

    nemo_relay.get_scope_stack()
