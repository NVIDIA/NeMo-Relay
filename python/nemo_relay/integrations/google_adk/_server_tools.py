# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record the tools that the model provider runs itself as Tool scopes, from the response."""

from __future__ import annotations

import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Final, Protocol, TypeVar

from google.adk.code_executors import BuiltInCodeExecutor
from google.adk.tools.enterprise_search_tool import EnterpriseWebSearchTool
from google.adk.tools.google_maps_grounding_tool import GoogleMapsGroundingTool
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.tools.url_context_tool import UrlContextTool
from google.adk.tools.vertex_ai_search_tool import VertexAiSearchTool
from google.genai import types

import nemo_relay
from nemo_relay import Json, ToolExecutionResult
from nemo_relay.integrations.google_adk._context import OK_STATUS, integration_metadata
from nemo_relay.integrations.google_adk._serialization import json_safe
from nemo_relay.integrations.google_adk._tool_codec import tool_result_json

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Mapping, Sequence

    from google.adk.code_executors import BaseCodeExecutor
    from google.adk.events.event import Event
    from google.adk.tools import BaseTool


class ServerTool(StrEnum):
    """Name the Tool scopes of the tools that the model provider runs itself."""

    GOOGLE_SEARCH = "google_search"
    ENTERPRISE_WEB_SEARCH = "enterprise_web_search"
    PARALLEL_AI_SEARCH = "parallel_ai_search"
    EXA_AI_SEARCH = "exa_ai_search"
    WEB_SEARCH = "web_search"
    VERTEX_AI_SEARCH = "vertex_ai_search"
    GOOGLE_MAPS = "google_maps"
    URL_CONTEXT = "url_context"
    CODE_EXECUTION = "code_execution"


_ADK_TYPES: Final[Mapping[str, type[BaseTool | BaseCodeExecutor]]] = {
    ServerTool.GOOGLE_SEARCH: GoogleSearchTool,
    ServerTool.ENTERPRISE_WEB_SEARCH: EnterpriseWebSearchTool,
    ServerTool.VERTEX_AI_SEARCH: VertexAiSearchTool,
    ServerTool.GOOGLE_MAPS: GoogleMapsGroundingTool,
    ServerTool.URL_CONTEXT: UrlContextTool,
    ServerTool.CODE_EXECUTION: BuiltInCodeExecutor,
}
_TOOL_TYPE_TOOLS: Final[Mapping[str, ServerTool]] = {
    types.ToolType.GOOGLE_SEARCH_WEB.value: ServerTool.GOOGLE_SEARCH,
    types.ToolType.GOOGLE_SEARCH_IMAGE.value: ServerTool.GOOGLE_SEARCH,
    types.ToolType.GOOGLE_MAPS.value: ServerTool.GOOGLE_MAPS,
    types.ToolType.URL_CONTEXT.value: ServerTool.URL_CONTEXT,
}
_SEARCH_TOOLS: Final = (
    ServerTool.GOOGLE_SEARCH,
    ServerTool.ENTERPRISE_WEB_SEARCH,
    ServerTool.PARALLEL_AI_SEARCH,
    ServerTool.EXA_AI_SEARCH,
)
_SERVER_SIDE: Final = (
    EnterpriseWebSearchTool,
    GoogleMapsGroundingTool,
    GoogleSearchTool,
    UrlContextTool,
    VertexAiSearchTool,
)


def is_server_side(tool: BaseTool) -> bool:
    return isinstance(tool, _SERVER_SIDE)


def server_tool_names(tools: Sequence[types.Tool | Callable[..., object]] | None) -> set[ServerTool]:
    """Return the web search and code execution tools that the request declares.

    The response evidence of these tools does not name the tool, so the request supplies the names. Legacy Google
    Search retrieval counts as Google Search.
    """
    names: set[ServerTool] = set()
    for tool in tools or ():
        if not isinstance(tool, types.Tool):
            continue
        present = (
            (ServerTool.GOOGLE_SEARCH, tool.google_search),
            (ServerTool.GOOGLE_SEARCH, tool.google_search_retrieval),
            (ServerTool.ENTERPRISE_WEB_SEARCH, tool.enterprise_web_search),
            (ServerTool.PARALLEL_AI_SEARCH, tool.parallel_ai_search),
            (ServerTool.EXA_AI_SEARCH, tool.exa_ai_search),
            (ServerTool.CODE_EXECUTION, tool.code_execution),
        )
        names.update(name for name, value in present if value is not None)
    return names


def _search_tool(names: Collection[ServerTool]) -> ServerTool:
    """Return the one web search tool that the request declares, or `web_search` when it declares several or none.

    Web search queries do not name the tool that ran them, and the Gemini API accepts several search tools in one
    request.
    """
    searches = [tool for tool in _SEARCH_TOOLS if tool in names]
    return searches[0] if len(searches) == 1 else ServerTool.WEB_SEARCH


class _Identified(Protocol):
    @property
    def id(self) -> str | None: ...


_Call = TypeVar("_Call", bound=_Identified)
_Result = TypeVar("_Result", bound=_Identified)


def _paired(steps: Iterable[tuple[_Call | None, _Result | None]]) -> list[tuple[_Call, _Result]]:
    """Pair each result with the earliest unpaired preceding call that has the same id, as the provider pairs them.

    Drop each call and result that finds no partner.
    """
    pending: list[_Call] = []
    pairs: list[tuple[_Call, _Result]] = []
    for call, result in steps:
        if call is not None:
            pending.append(call)
        if result is not None:
            matched = next((index for index, waiting in enumerate(pending) if waiting.id == result.id), None)
            if matched is not None:
                pairs.append((pending.pop(matched), result))
    return pairs


def _tool_call_name(tool_type: types.ToolType | None) -> str:
    """Return the `ServerTool` name for a server-side tool call's type.

    Return the lowercase type value when no `ServerTool` maps to the type. A call without a tool type counts as
    `TOOL_TYPE_UNSPECIFIED`.
    """
    value = (types.ToolType.TOOL_TYPE_UNSPECIFIED if tool_type is None else tool_type).value
    tool = _TOOL_TYPE_TOOLS.get(value)
    return value.lower() if tool is None else tool


def record_server_tools(
    event: Event, names: Collection[ServerTool], parent: Callable[[], nemo_relay.ScopeHandle]
) -> None:
    """Record each provider-run tool that the event reports as a Tool scope at the event's time.

    Resolve the parent scope from `parent` only when the event reports a tool. Pair `tool_call` parts with
    `tool_response` parts. Record grounding web search queries and Google Maps context only when those parts did not
    report that tool, and name the queries after the one web search tool in `names`, or `web_search` without one.
    Record code execution parts only when `names` includes code execution.
    """
    when = datetime.datetime.fromtimestamp(event.timestamp, datetime.UTC)
    parts = [] if event.content is None else event.content.parts or []
    tool_calls = [
        (_tool_call_name(call.tool_type), call, response)
        for call, response in _paired((part.tool_call, part.tool_response) for part in parts)
    ]
    reported = {name for name, _, _ in tool_calls}
    for name, call, response in tool_calls:
        _record(name, data=json_safe(call), output=response, tool_call_id=call.id, parent=parent, when=when)
    grounding = event.grounding_metadata
    if grounding is not None:
        chunks = {"chunks": grounding.grounding_chunks or []}
        if grounding.web_search_queries and ServerTool.GOOGLE_SEARCH not in reported:
            _record(
                _search_tool(names),
                data={"queries": list(grounding.web_search_queries)},
                output=chunks,
                tool_call_id=None,
                parent=parent,
                when=when,
            )
        if grounding.retrieval_queries:
            _record(
                ServerTool.VERTEX_AI_SEARCH,
                data={"queries": list(grounding.retrieval_queries)},
                output=chunks,
                tool_call_id=None,
                parent=parent,
                when=when,
            )
        if grounding.google_maps_widget_context_token and ServerTool.GOOGLE_MAPS not in reported:
            _record(ServerTool.GOOGLE_MAPS, data={}, output=chunks, tool_call_id=None, parent=parent, when=when)
    if ServerTool.CODE_EXECUTION in names:
        for code, result in _paired((part.executable_code, part.code_execution_result) for part in parts):
            _record(
                ServerTool.CODE_EXECUTION,
                data=json_safe(code),
                output=result,
                tool_call_id=code.id,
                parent=parent,
                when=when,
            )


def _record(
    name: str,
    *,
    data: Json,
    output: object,
    tool_call_id: str | None,
    parent: Callable[[], nemo_relay.ScopeHandle],
    when: datetime.datetime,
) -> None:
    """Open and close a Tool scope for one call that the provider ran, and project the result before the scope opens."""
    result = tool_result_json(output)
    metadata = integration_metadata(adk_server_side=True)
    if (adk_type := _ADK_TYPES.get(name)) is not None:
        metadata["adk_tool_type"] = adk_type.__name__
    handle = nemo_relay.tools.call(
        name, data, handle=parent(), metadata=metadata, tool_call_id=tool_call_id, timestamp=when
    )
    nemo_relay.tools.call_end(handle, ToolExecutionResult(result), metadata=OK_STATUS, timestamp=when)
