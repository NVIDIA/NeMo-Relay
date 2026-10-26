# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observe the root run of each ADK invocation, which ADK cancels without calling a plugin hook."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING, TypeVar

from google.adk.agents import BaseAgent

from nemo_relay.integrations.google_adk._context import current_invocation
from nemo_relay.integrations.google_adk._swap import Methods, swap_class

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from google.adk.agents.context import Context
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.events import Event
    from google.adk.workflow import BaseNode

_logger = logging.getLogger(__name__)
_T = TypeVar("_T")


def observe_root(node: BaseNode) -> None:
    """Swap the class of the invocation's root node for a subclass whose root run reports when ADK interrupts it."""
    if isinstance(node, BaseAgent):
        swap_class(node, _agent_root_methods)
    else:
        swap_class(node, _root_methods)


def _root_methods(cls: type[BaseNode]) -> Methods[BaseNode]:
    original = cls._run_impl

    async def _run_impl(self: BaseNode, *, ctx: Context, node_input: object) -> AsyncGenerator[object, None]:
        run = _observed(ctx.invocation_id, original(self, ctx=ctx, node_input=node_input))
        async with contextlib.aclosing(run) as items:
            async for item in items:
                yield item

    return {"_run_impl": _run_impl}


def _agent_root_methods(cls: type[BaseAgent]) -> Methods[BaseAgent]:
    run_async = cls.run_async

    async def run(self: BaseAgent, parent_context: InvocationContext) -> AsyncGenerator[Event, None]:
        async with contextlib.aclosing(
            _observed(parent_context.invocation_id, run_async(self, parent_context))
        ) as events:
            async for event in events:
                yield event

    return {**_root_methods(cls), "run_async": run}


async def _observed(invocation_id: str, run: AsyncGenerator[_T, None]) -> AsyncGenerator[_T, None]:
    """Yield the run's items, and interrupt the invocation when ADK closes or cancels the invocation's root run.

    Watch the task that runs the root run as well, and log a failure to interrupt the invocation, so the close or
    cancellation reaches ADK unchanged.
    """
    invocation = current_invocation()
    root = (
        invocation
        if invocation is not None and invocation.invocation_id == invocation_id and invocation.claim_root()
        else None
    )
    if root is not None:
        root.watch()
    try:
        async with contextlib.aclosing(run) as items:
            async for item in items:
                yield item
    except (GeneratorExit, asyncio.CancelledError):
        if root is not None:
            try:
                root.interrupt()
            except Exception:
                _logger.exception("NeMo Relay: interrupting the ADK invocation %s failed", invocation_id)
        raise
