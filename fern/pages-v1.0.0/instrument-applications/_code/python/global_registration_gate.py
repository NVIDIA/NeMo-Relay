# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Awaitable, Callable

from nemo_relay.runtime_registrations import (
    RuntimeRegistrationKind,
    deregister_conditional_middleware_guardrail,
    list_runtime_registrations,
    register_conditional_middleware_guardrail,
)


async def run_example(run_application: Callable[[], Awaitable[None]]) -> None:
    """Pause a configured OpenTelemetry subscriber during application work."""
    target = next(
        (
            registration
            for registration in list_runtime_registrations({RuntimeRegistrationKind.SUBSCRIBER})
            if registration.local_name == "opentelemetry" and registration.owner.plugin_kind == "observability"
        ),
        None,
    )
    if target is None:
        raise RuntimeError("OpenTelemetry subscriber was not found")

    paused = True
    register_conditional_middleware_guardrail(
        "pause-opentelemetry",
        {RuntimeRegistrationKind.SUBSCRIBER},
        target.effective_name,
        lambda _kinds, _name: "export paused" if paused else None,
    )
    try:
        await run_application()
    finally:
        deregister_conditional_middleware_guardrail("pause-opentelemetry")
