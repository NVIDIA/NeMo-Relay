# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integrate an application's existing approval checks with execution middleware."""

import nemo_relay
from nemo_relay import tools


def _approval_has_expected_shape(approval) -> bool:
    """Validate this application's example payload; this is not a Relay schema."""
    if not isinstance(approval, dict):
        return False
    version = approval.get("version")
    if isinstance(version, bool) or not isinstance(version, (int, float)) or version != 1:
        return False
    if "decision" in approval or "approved_arguments" not in approval:
        return False
    return all(isinstance(approval.get(key), str) and approval[key] for key in ("source", "approval_id", "tool_name"))


async def execute_after_application_approval(
    name, args, receipt, implementation, *, validate_existing_approval
) -> nemo_relay.ToolExecutionResult[nemo_relay.Json]:
    """Verify approval before dispatch and again at execution callback entry.

    If implementation waits before committing an effect, the application must
    recheck approval at that boundary when its policy requires continued validity.
    """
    # The application must verify the real receipt's owner, exact operation,
    # validity and current policy before returning its own approval assertion.
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
        execution_context={"data": {"host_approval": reviewed}},
    )


def register_application_review(approval_covers_call, perform_component_review) -> None:
    """Register a synchronous boolean trust matcher and an asynchronous reviewer.

    The reviewer must raise on refusal; its return value is not a decision.
    ``perform_component_review`` checks the arguments at this middleware stage.
    Later middleware can change them. If policy requires approval of the final
    operation, the application must enforce that policy in the tool implementation,
    including calls without an approval claim.
    """

    async def review(context, next_call):
        supplied = (context.execution_context or {}).get("data", {})
        if "host_approval" in supplied:
            approval = supplied["host_approval"]
            # Shape alone is not proof. The application-owned matcher checks
            # the trusted calling host, source and exact tool/arguments.
            if not _approval_has_expected_shape(approval) or not approval_covers_call(
                approval, context.tool_name, context.args
            ):
                # A supplied claim cannot downgrade to automated review.
                raise PermissionError("Application approval is invalid or does not cover this call")
        else:
            # No assertion supplied: follow this component's normal policy.
            # This callback must raise when its policy forbids execution.
            # Returning False does not stop this wrapper.
            await perform_component_review(context.tool_name, context.args)

        # Skip only this component's repeated review, not the remaining chain.
        downstream = await next_call(context.args)
        return nemo_relay.ToolExecutionInterceptOutcome(result=downstream.result, annotation=downstream.annotation)

    nemo_relay.intercepts.register_tool_execution("application-review", 10, review)
