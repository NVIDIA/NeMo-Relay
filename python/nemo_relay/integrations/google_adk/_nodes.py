# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record each run of a `BaseNode` as a scope with swapped node subclasses."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, TypeAlias, TypeVar

from google.adk.agents import BaseAgent
from google.adk.events import Event, RequestInput
from google.adk.workflow import START, BaseNode, Workflow

import nemo_relay
from nemo_relay import Json, JsonObject, ScopeType
from nemo_relay.integrations.google_adk._context import (
    CURRENT_PARENT,
    OK_STATUS,
    current_invocation,
    error_status,
    integration_metadata,
    log_unrecorded_once,
    parent_handle,
)
from nemo_relay.integrations.google_adk._serialization import json_safe
from nemo_relay.integrations.google_adk._swap import Methods, swap_class

if TYPE_CHECKING:
    from google.adk.agents.context import Context

_NodeRun: TypeAlias = AsyncGenerator[object, None]
_NodeT = TypeVar("_NodeT", bound=BaseNode)


def wrap_node(node: _NodeT) -> _NodeT:
    """Swap the node's class so that each run records a scope, and return the node.

    Wrap the graph nodes of a `Workflow` the same way, and leave agents and `START` unwrapped.
    """
    if isinstance(node, BaseAgent) or node is START:
        return node
    swap_class(node, _methods)
    if isinstance(node, Workflow) and node.graph is not None:
        for child in node.graph.nodes:
            wrap_node(child)
    return node


def _methods(cls: type[BaseNode]) -> Methods[BaseNode]:
    original = cls._run_impl
    scope_type = ScopeType.Agent if issubclass(cls, Workflow) else ScopeType.Function

    async def _run_impl(self: BaseNode, *, ctx: Context, node_input: object) -> _NodeRun:
        run = original(self, ctx=ctx, node_input=node_input)
        async with contextlib.aclosing(_record_run(self, scope_type, run, ctx, node_input)) as items:
            async for item in items:
                yield item

    return {"_run_impl": _run_impl}


def _output_of(item: object) -> object:
    """Return an event's output, None for an interrupt request, and any other item unchanged."""
    if isinstance(item, Event):
        return item.output
    if isinstance(item, RequestInput):
        return None
    return item


async def _record_run(
    node: BaseNode, scope_type: ScopeType, run: _NodeRun, ctx: Context, node_input: object
) -> _NodeRun:
    """Record the node run as a scope under the current agent scope, and yield the run's items unchanged.

    Outside an observed invocation, yield the items without a scope.
    """
    invocation = current_invocation()
    if invocation is None:
        log_unrecorded_once("workflow node run")
        async with contextlib.aclosing(run) as items:
            async for item in items:
                yield item
        return
    scope_input: JsonObject = {
        "invocation_id": ctx.invocation_id,
        "branch": ctx.branch,
        "node_type": type(node).__name__,
    }
    if node_input is not None:
        scope_input["node_input"] = json_safe(node_input)
    handle = nemo_relay.scope.push(
        node.name, scope_type, handle=parent_handle(), input=scope_input, metadata=integration_metadata()
    )
    invocation.own(handle.uuid)
    invocation.recorded_nodes.add(ctx.node_path)
    previous = CURRENT_PARENT.get()
    if previous is invocation:
        invocation.watch()
    CURRENT_PARENT.set(handle)
    output: Json = None
    try:
        async with contextlib.aclosing(run) as items:
            async for item in items:
                value = _output_of(item)
                if value is not None:
                    output = json_safe(value)
                yield item
    except (GeneratorExit, asyncio.CancelledError):
        invocation.close(handle, output={"status": "cancelled"}, metadata=error_status("node cancelled"))
        raise
    except Exception as error:
        invocation.close(handle, output={"error": repr(error)}, metadata=error_status(str(error)))
        raise
    else:
        child_error = ctx.error
        if child_error is not None:
            invocation.close(
                handle,
                output={"error": repr(child_error), "error_node_path": ctx.error_node_path},
                metadata=error_status(str(child_error)),
            )
        else:
            result: JsonObject = {"status": "ok"}
            if output is not None:
                result["output"] = output
            invocation.close(handle, output=result, metadata=OK_STATUS)
    finally:
        CURRENT_PARENT.set(previous)
