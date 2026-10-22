# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provide the Google ADK plugin that records invocations and agent runs as scopes.

The plugin also wraps the models and tools that ADK calls.
"""

from __future__ import annotations

import datetime
import functools
import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, Literal, NamedTuple, ParamSpec, TypeAlias, TypeVar

from google.adk.agents import LlmAgent
from google.adk.code_executors import BuiltInCodeExecutor
from google.adk.plugins import BasePlugin
from google.adk.workflow import BaseNode
from typing_extensions import override

import nemo_relay
from nemo_relay import Json, JsonObject, ScopeType
from nemo_relay.integrations.google_adk._context import (
    CANCELLED_STATUS,
    CURRENT_INVOCATION,
    CURRENT_PARENT,
    OK_STATUS,
    Invocation,
    InvocationScope,
    current_invocation,
    current_task,
    error_status,
    integration_metadata,
    log_unrecorded_once,
    parent_handle,
)
from nemo_relay.integrations.google_adk._models import wrap_agent_models
from nemo_relay.integrations.google_adk._nodes import wrap_node
from nemo_relay.integrations.google_adk._root import observe_root
from nemo_relay.integrations.google_adk._serialization import json_safe
from nemo_relay.integrations.google_adk._server_tools import is_server_side, record_server_tools, server_tool_names
from nemo_relay.integrations.google_adk._tools import wrap_code_executor, wrap_tool

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from types import CoroutineType

    from google.adk.agents import BaseAgent
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.events.event import Event
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.adk.tools import BaseTool
    from google.adk.tools.tool_context import ToolContext
    from google.genai import types

    from nemo_relay.integrations.google_adk._server_tools import ServerTool

_logger = logging.getLogger(__name__)


_P = ParamSpec("_P")
_R = TypeVar("_R")


def _never_raises(hook: Callable[_P, Awaitable[_R]]) -> Callable[_P, CoroutineType[Any, Any, _R | None]]:
    @functools.wraps(hook)
    async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R | None:
        try:
            return await hook(*args, **kwargs)
        except Exception:
            _logger.exception("NeMo Relay: %s failed", wrapper.__name__)
            return None

    return wrapper


def _with_response(output: JsonObject, response: Json) -> JsonObject:
    return output if response is None else {**output, "response": response}


def _is_live(invocation_context: InvocationContext) -> bool:
    """Return whether the invocation runs live.

    A `run_live` invocation has a live request queue, and compositional function calling also runs live.
    """
    run_config = invocation_context.run_config
    return invocation_context.live_request_queue is not None or bool(run_config and run_config.support_cfc)


_MarkName: TypeAlias = Literal[
    "adk.agent_transfer",
    "adk.escalate",
    "adk.long_running_tools",
    "adk.node_completed",
    "adk.node_failed",
    "adk.tool_confirmation_requested",
    "adk.tool_confirmation_resumed",
]


def _mark(name: _MarkName, parent: nemo_relay.ScopeHandle, **data: Json) -> None:
    nemo_relay.scope.event(name, handle=parent, data=data, metadata=integration_metadata())


class _AgentKey(NamedTuple):
    branch: str | None
    agent: str


@dataclass(slots=True)
class _AgentRun:
    handle: nemo_relay.ScopeHandle
    previous: nemo_relay.ScopeHandle | Invocation | None
    response: Json = None


@dataclass(slots=True, kw_only=True)
class _InvocationState:
    invocation: Invocation
    runner: Annotated[
        int | None,
        "This field holds the runner's plugin manager `id` for a top-level invocation, or None for an `AgentTool` run.",
    ]
    agent_runs: dict[_AgentKey, list[_AgentRun]] = field(default_factory=dict)
    agent_names: Annotated[
        set[str], "This set holds the names of the agents whose runs the invocation recorded as scopes."
    ] = field(default_factory=set)
    response: Json = None
    server_tools: Annotated[
        dict[str, set[ServerTool]],
        "This mapping holds, for each agent name, the tools that the model provider runs itself.",
    ] = field(default_factory=dict)


class NemoRelayPlugin(BasePlugin):
    """Record ADK invocations and agent runs as scopes, and run the model and tool calls that ADK makes through Relay.

    Args:
        name: The ADK plugin name, which must be unique within an `App`.
    """

    def __init__(self, name: str = "nemo_relay") -> None:
        super().__init__(name)
        self._lock = threading.Lock()
        self._invocations: dict[str, _InvocationState] = {}
        self._shared_runner_warned = False

    @_never_raises
    @override
    async def before_run_callback(self, *, invocation_context: InvocationContext) -> types.Content | None:
        self._start_invocation(invocation_context)
        return None

    @_never_raises
    @override
    async def after_run_callback(self, *, invocation_context: InvocationContext) -> None:
        """End the invocation as ADK reports it, as cancelled for an aborted run and otherwise as completed."""
        invocation_id = invocation_context.invocation_id
        if invocation_context.is_aborted:
            self._end_invocation(invocation_id, {"status": "cancelled"}, CANCELLED_STATUS)
        else:
            self._end_invocation(invocation_id, {"status": "ok"}, OK_STATUS)
        _release(invocation_id)

    @_never_raises
    @override
    async def on_run_error_callback(self, *, invocation_context: InvocationContext, error: Exception) -> None:
        self._end_invocation(invocation_context.invocation_id, {"error": repr(error)}, error_status(str(error)))
        _release(invocation_context.invocation_id)

    @_never_raises
    @override
    async def close(self) -> None:
        """Close the scope of each invocation that is still running as cancelled."""
        with self._lock:
            open_invocations = list(self._invocations)
        for invocation_id in open_invocations:
            self._abandon_invocation(invocation_id)

    def _start_invocation(self, invocation_context: InvocationContext) -> None:
        """Track the invocation, whose scope opens under the current parent when a call first needs it.

        First close a stale invocation of the same runner in this context as cancelled. Observe the root agent's run
        when ADK names a root agent on the invocation context, and warn once when the plugin serves more than one
        runner.
        """
        stale = current_invocation()
        if stale is not None:
            stale_state = self._state(stale.invocation_id)
            if stale_state is not None and stale_state.runner == id(invocation_context.plugin_manager):
                self._abandon_invocation(stale.invocation_id)
        nested = current_invocation() is not None
        scope_input: JsonObject = {
            "invocation_id": invocation_context.invocation_id,
            "session_id": invocation_context.session.id,
            "user_id": invocation_context.user_id,
            "app_name": invocation_context.app_name,
            "user_content": json_safe(invocation_context.user_content),
        }
        root = invocation_context.agent
        if root is not None:
            scope_input["root_agent"] = root.name
        metadata = integration_metadata()
        run_config = invocation_context.run_config
        if run_config is not None and run_config.custom_metadata:
            metadata["adk_custom_metadata"] = json_safe(run_config.custom_metadata)
        invocation_id = invocation_context.invocation_id
        invocation = Invocation(
            invocation_id,
            scope=InvocationScope(
                f"adk-invocation:{invocation_context.app_name}",
                parent_handle(),
                scope_input,
                metadata,
                datetime.datetime.now(datetime.UTC),
            ),
            driver=current_task(),
            previous_parent=CURRENT_PARENT.get(),
            previous_invocation=CURRENT_INVOCATION.get(),
            end=functools.partial(self._end_invocation, invocation_id),
            abandon=functools.partial(self._abandon_invocation, invocation_id),
        )
        state = _InvocationState(
            invocation=invocation, runner=None if nested else id(invocation_context.plugin_manager)
        )
        with self._lock:
            shared = state.runner is not None and any(
                other.runner not in (None, state.runner) for other in self._invocations.values()
            )
            self._invocations[invocation_id] = state
            warn = shared and not self._shared_runner_warned
            if warn:
                self._shared_runner_warned = True
        if warn:
            _logger.warning(
                "NeMo Relay: NemoRelayPlugin %r serves more than one Runner. Create one plugin per Runner: ADK "
                "closes a plugin with the first Runner that closes and does not say which one.",
                self.name,
            )
        CURRENT_PARENT.set(invocation)
        CURRENT_INVOCATION.set(invocation)
        if root is not None:
            observe_root(root)

    def _state(self, invocation_id: str) -> _InvocationState | None:
        with self._lock:
            return self._invocations.get(invocation_id)

    def _end_invocation(self, invocation_id: str, output: JsonObject, metadata: JsonObject) -> None:
        """Record the invocation with the output and status, and open its scope first when no call opened it."""
        self._finish_invocation(invocation_id, output=output, metadata=metadata, record=True)

    def _abandon_invocation(self, invocation_id: str) -> None:
        """Close the invocation's scope as cancelled when a call opened it, and otherwise record no scope for it."""
        self._finish_invocation(invocation_id, output={"status": "cancelled"}, metadata=CANCELLED_STATUS, record=False)

    def _finish_invocation(self, invocation_id: str, *, output: JsonObject, metadata: JsonObject, record: bool) -> None:
        """Close the agent runs that ADK ended without a completion hook, and then the invocation scope.

        Close the scopes on the invocation's own stack, and restore the context variables from before the invocation
        when the invocation is current in this context. Do nothing when the invocation already finished.
        """
        with self._lock:
            state = self._invocations.pop(invocation_id, None)
            if state is None:
                return
            runs = [run for stack in reversed(state.agent_runs.values()) for run in reversed(stack)]
            state.agent_runs.clear()
        invocation = state.invocation
        for run in runs:
            invocation.close(run.handle, output=_with_response(output, run.response), metadata=metadata)
        invocation.finish(output=_with_response(output, state.response), metadata=metadata, record=record)
        invocation.release()

    @_never_raises
    @override
    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> types.Content | None:
        state = self._state(callback_context.invocation_id)
        _instrument_agent(agent, callback_context.get_invocation_context())
        if state is None:
            log_unrecorded_once("agent run")
            return None
        previous = CURRENT_PARENT.get()
        scope_input: JsonObject = {
            "invocation_id": callback_context.invocation_id,
            "branch": callback_context.branch,
            "agent_type": type(agent).__name__,
        }
        if agent.description:
            scope_input["description"] = agent.description
        handle = nemo_relay.scope.push(
            agent.name,
            ScopeType.Agent,
            handle=parent_handle(),
            input=scope_input,
            metadata=integration_metadata(),
        )
        state.invocation.own(handle.uuid)
        if previous is state.invocation:
            state.invocation.watch()
        key = _AgentKey(callback_context.branch, agent.name)
        with self._lock:
            state.agent_runs.setdefault(key, []).append(_AgentRun(handle, previous))
            state.agent_names.add(agent.name)
        CURRENT_PARENT.set(handle)
        return None

    @_never_raises
    @override
    async def after_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> types.Content | None:
        """Close the agent run, as cancelled when ADK calls this hook while the agent's task is being cancelled."""
        cancelled = current_task().cancelling() > 0
        self._finish_agent(
            callback_context.invocation_id,
            branch=callback_context.branch,
            agent_name=agent.name,
            output={"status": "cancelled"} if cancelled else {"status": "ok"},
            metadata=error_status("agent cancelled") if cancelled else OK_STATUS,
        )
        return None

    @_never_raises
    @override
    async def on_agent_error_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext, error: Exception
    ) -> None:
        self._finish_agent(
            callback_context.invocation_id,
            branch=callback_context.branch,
            agent_name=agent.name,
            output={"error": repr(error)},
            metadata=error_status(str(error)),
        )

    def _finish_agent(
        self, invocation_id: str, *, branch: str | None, agent_name: str, output: JsonObject, metadata: JsonObject
    ) -> None:
        state = self._state(invocation_id)
        if state is None:
            return
        with self._lock:
            runs = state.agent_runs.get(_AgentKey(branch, agent_name))
            run = runs.pop() if runs else None
        if run is None:
            return
        CURRENT_PARENT.set(run.previous)
        state.invocation.close(run.handle, output=_with_response(output, run.response), metadata=metadata)

    @_never_raises
    @override
    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        invocation_context = callback_context.get_invocation_context()
        state = self._state(callback_context.invocation_id)
        _instrument_agent(invocation_context.agent, invocation_context)
        if state is not None:
            state.server_tools.setdefault(callback_context.agent_name, set()).update(
                server_tool_names(llm_request.config.tools)
            )
        for tool in llm_request.tools_dict.values():
            _instrument_tool(tool)
        return None

    @_never_raises
    @override
    async def before_tool_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext
    ) -> dict[str, Any] | None:
        confirmation = tool_context.tool_confirmation
        if confirmation is not None:
            _mark(
                "adk.tool_confirmation_resumed",
                parent_handle(),
                function_call_id=tool_context.function_call_id,
                confirmed=confirmation.confirmed,
            )
        _instrument_tool(tool)
        return None

    @_never_raises
    @override
    async def on_event_callback(self, *, invocation_context: InvocationContext, event: Event) -> Event | None:
        """Record final responses and the provider-run tools that each event reports.

        Emit marks for agent transfers, tool confirmation requests, escalation, and long-running tools. Record
        provider-run tools only from events that are not partial, and mark the events of workflow nodes that no scope
        records.
        """
        actions = event.actions
        state = self._state(invocation_context.invocation_id)
        if state is not None and event.author != "user" and event.content is not None and event.is_final_response():
            self._record_response(state, event)
        if actions.transfer_to_agent:
            _mark(
                "adk.agent_transfer",
                parent_handle(),
                **{"from": event.author, "to": actions.transfer_to_agent, "reason": actions.transfer_reason},
            )
            self._finish_agent(
                invocation_context.invocation_id,
                branch=event.branch,
                agent_name=event.author,
                output={"transferred_to": actions.transfer_to_agent},
                metadata=OK_STATUS,
            )
        if actions.requested_tool_confirmations:
            _mark(
                "adk.tool_confirmation_requested",
                parent_handle(),
                function_call_ids=list[Json](sorted(actions.requested_tool_confirmations)),
            )
        if actions.escalate:
            _mark("adk.escalate", parent_handle(), author=event.author)
        if event.long_running_tool_ids:
            _mark(
                "adk.long_running_tools",
                parent_handle(),
                function_call_ids=list[Json](sorted(event.long_running_tool_ids)),
            )
        if state is not None:
            if not event.partial:
                record_server_tools(
                    event, state.server_tools.get(event.author, ()), functools.partial(self._author_scope, state, event)
                )
            _record_node_event(state, event)
        return None

    def _author_scope(self, state: _InvocationState, event: Event) -> nemo_relay.ScopeHandle:
        """Return the scope of the open agent run that authored the event, or the current parent without one."""
        with self._lock:
            runs = state.agent_runs.get(_AgentKey(event.branch, event.author))
            author = runs[-1].handle if runs else None
        return parent_handle() if author is None else author

    def _record_response(self, state: _InvocationState, event: Event) -> None:
        """Store the event content as the latest final response of the invocation and of the event's agent run."""
        response = json_safe(event.content)
        with self._lock:
            state.response = response
            runs = state.agent_runs.get(_AgentKey(event.branch, event.author))
            if runs:
                runs[-1].response = response


def _instrument_agent(agent: BaseNode | None, invocation_context: InvocationContext) -> None:
    """Wrap the models that an `LlmAgent` selects and its code executor, outside live invocations."""
    if not isinstance(agent, LlmAgent) or _is_live(invocation_context):
        return
    wrap_agent_models(agent)
    executor = agent.code_executor
    if executor is not None and not isinstance(executor, BuiltInCodeExecutor):
        wrap_code_executor(executor)


def _instrument_tool(tool: BaseTool) -> None:
    if is_server_side(tool):
        return
    wrap_tool(tool)
    if isinstance(node := getattr(tool, "node", None), BaseNode):
        wrap_node(node)


def _record_node_event(state: _InvocationState, event: Event) -> None:
    """Mark the completion or failure of an ADK workflow node that no node scope or agent scope records.

    The integration observes such a node only through the event that the node ends with.
    """
    path = event.node_info.path
    if not path or path in state.invocation.recorded_nodes:
        return
    if event.node_info.name in state.agent_names:
        return
    if event.error_code is not None or event.error_message is not None:
        _mark(
            "adk.node_failed",
            parent_handle(),
            path=path,
            error_code=event.error_code,
            error_message=event.error_message,
        )
    elif event.output is not None or event.node_info.output_for:
        _mark("adk.node_completed", parent_handle(), path=path, output=json_safe(event.output))


def _release(invocation_id: str) -> None:
    """Restore the context variables from before the invocation when it is current in this context."""
    invocation = CURRENT_INVOCATION.get()
    if invocation is not None and invocation.invocation_id == invocation_id:
        invocation.release()


__all__ = ["NemoRelayPlugin"]
