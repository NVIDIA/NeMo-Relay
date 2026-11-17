# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integrate an application's existing approval checks with execution middleware."""

import nemo_relay
from nemo_relay import tools


async def execute_after_application_approval(
    name, args, receipt, implementation, *, validate_existing_approval
) -> nemo_relay.ToolExecutionResult[nemo_relay.Json]:
    """Run synchronous host verification initially and immediately before the effect."""
    # The application must verify the real receipt's owner, exact operation,
    # validity and current policy before returning a HostApproval assertion.
    reviewed = validate_existing_approval(receipt, name, args)

    async def execute_reviewed(actual_args):
        # Middleware can wait or rewrite arguments. Automated review cannot
        # expand the operation that the person approved.
        validate_existing_approval(receipt, name, actual_args)
        return await implementation(actual_args)

    return await tools.execute(
        name,
        args,
        execute_reviewed,
        execution_context={"host_approval": reviewed},
    )


def register_application_review(approval_covers_call, perform_component_review) -> None:
    """Register a synchronous boolean trust matcher and an asynchronous reviewer.

    The reviewer must raise on refusal; its return value is not a decision.
    """

    async def review(context, next_call):
        supplied = context.execution_context or {}
        # This application-owned matcher must check the trusted calling host,
        # assertion source and exact tool/arguments; a source string is not proof.
        if not approval_covers_call(supplied.get("host_approval"), context.tool_name, context.args):
            # Missing or mismatched assertions take the normal review path.
            # This callback must raise when its policy forbids execution.
            # Returning False does not stop this wrapper.
            await perform_component_review(context.tool_name, context.args)

        # Skip only this component's repeated review, not the remaining chain.
        downstream = await next_call(context.args)
        return nemo_relay.ToolExecutionInterceptOutcome(result=downstream.result, annotation=downstream.annotation)

    nemo_relay.intercepts.register_tool_execution("application-review", 10, review)
