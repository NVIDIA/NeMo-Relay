# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run ADK tool calls and code executions through the managed tool pipeline.

Swapped `BaseTool` and `BaseCodeExecutor` subclasses route the calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from google.adk.code_executors.code_execution_utils import CodeExecutionInput, CodeExecutionResult

import nemo_relay
from nemo_relay import ToolExecutionResult
from nemo_relay.integrations.google_adk._context import integration_metadata, parent_handle, rejection_marked
from nemo_relay.integrations.google_adk._swap import Methods, swap_class
from nemo_relay.integrations.google_adk._tool_codec import AdkArgumentsCodec, AdkCodeInputCodec, ProducedResults
from nemo_relay.typed import JsonPassthrough
from nemo_relay.utils import run_sync

if TYPE_CHECKING:
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.code_executors import BaseCodeExecutor
    from google.adk.tools import BaseTool
    from google.adk.tools.tool_context import ToolContext

    from nemo_relay import Json


def wrap_tool(tool: BaseTool) -> None:
    """Swap the tool's class for a subclass whose calls run through Relay."""
    swap_class(tool, _tool_methods)


def wrap_code_executor(executor: BaseCodeExecutor) -> None:
    """Swap the executor's class for a subclass that runs each code execution as a Relay tool call."""
    swap_class(executor, _executor_methods)


def _tool_methods(cls: type[BaseTool]) -> Methods[BaseTool]:
    original = cls.run_async

    async def run_async(self: BaseTool, *, args: dict[str, Any], tool_context: ToolContext) -> Any:
        produced = ProducedResults()
        in_flight = 0

        async def call(arguments: dict[str, Any]) -> ToolExecutionResult[Json]:
            """Run the tool with the attempt's arguments.

            Write them into ADK's dictionary when no other attempt is in flight, and run the tool on a copy otherwise.
            """
            nonlocal in_flight
            if in_flight == 0:
                args.clear()
                args.update(arguments)
                attempt_args = args
            else:
                attempt_args = dict(arguments)
            in_flight += 1
            try:
                result = await original(self, args=attempt_args, tool_context=tool_context)
            finally:
                in_flight -= 1
            return ToolExecutionResult(produced.add(result))

        metadata = integration_metadata(adk_tool_type=cls.__name__)
        if self.is_long_running:
            metadata["adk_long_running"] = True
        parent = parent_handle()
        with rejection_marked(
            "adk.tool_arguments_rejected", parent, tool=self.name, tool_call_id=tool_context.function_call_id
        ):
            arguments_codec = AdkArgumentsCodec(dict(args))
        outcome = await nemo_relay.typed.tool_execute(
            self.name,
            args,
            call,
            arguments_codec,
            JsonPassthrough(),
            handle=parent,
            metadata=metadata,
            tool_call_id=tool_context.function_call_id,
        )
        with rejection_marked(
            "adk.tool_result_rejected", parent, tool=self.name, tool_call_id=tool_context.function_call_id
        ):
            return produced.restore(outcome.result, object)

    return {"run_async": run_async}


def _executor_methods(cls: type[BaseCodeExecutor]) -> Methods[BaseCodeExecutor]:
    original = cls.execute_code

    def execute_code(
        self: BaseCodeExecutor, invocation_context: InvocationContext, code_execution_input: CodeExecutionInput
    ) -> CodeExecutionResult:
        produced = ProducedResults()

        def call(code_input: CodeExecutionInput) -> ToolExecutionResult[Json]:
            return ToolExecutionResult(produced.add(original(self, invocation_context, code_input)))

        parent = parent_handle()
        outcome = run_sync(
            nemo_relay.typed.tool_execute(
                "code_execution",
                code_execution_input,
                call,
                AdkCodeInputCodec(code_execution_input),
                JsonPassthrough(),
                handle=parent,
                metadata=integration_metadata(adk_code_executor=cls.__name__),
            )
        )
        with rejection_marked("adk.tool_result_rejected", parent, tool="code_execution", tool_call_id=None):
            return produced.restore(outcome.result, CodeExecutionResult)

    return {"execute_code": execute_code}
