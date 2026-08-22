# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for dynamic runtime-registration eligibility controls."""

from uuid import uuid4

import nemo_relay._native as native
from nemo_relay import intercepts, runtime_registrations, tools
from nemo_relay.runtime_registrations import RuntimeRegistrationKind


def test_native_list_runtime_registrations_defaults_to_all_kinds():
    assert isinstance(native.list_runtime_registrations(), list)


def test_conditional_middleware_guardrail_toggles_existing_registration():
    suffix = uuid4().hex
    target_name = f"python-runtime-target-{suffix}"
    gate_name = f"python-runtime-gate-{suffix}"
    seen: list[tuple[set[RuntimeRegistrationKind], str]] = []

    intercepts.register_tool_request(
        target_name,
        0,
        False,
        lambda _name, args: {**args, "intercepted": True},
    )
    try:
        registrations = runtime_registrations.list_runtime_registrations(
            {RuntimeRegistrationKind.TOOL_REQUEST_INTERCEPT}
        )
        target = next(item for item in registrations if item.local_name == target_name)
        assert target.effective_name == target_name

        runtime_registrations.register_conditional_middleware_guardrail(
            gate_name,
            {RuntimeRegistrationKind.TOOL_REQUEST_INTERCEPT},
            target.effective_name,
            lambda kinds, name: seen.append((kinds, name)) or "timer active",
        )
        try:
            assert tools.request_intercepts("tool", {}) == {}
            assert seen == [({RuntimeRegistrationKind.TOOL_REQUEST_INTERCEPT}, target_name)]
        finally:
            assert runtime_registrations.deregister_conditional_middleware_guardrail(gate_name)

        assert tools.request_intercepts("tool", {}) == {"intercepted": True}
    finally:
        intercepts.deregister_tool_request(target_name)


def test_conditional_middleware_guardrail_callback_error_fails_open():
    suffix = uuid4().hex
    target_name = f"python-runtime-fail-open-target-{suffix}"
    gate_name = f"python-runtime-fail-open-gate-{suffix}"

    def fail(_kinds: set[RuntimeRegistrationKind], _name: str) -> None:
        raise RuntimeError("expected gate failure")

    intercepts.register_tool_request(
        target_name,
        0,
        False,
        lambda _name, args: {**args, "intercepted": True},
    )
    runtime_registrations.register_conditional_middleware_guardrail(
        gate_name,
        {RuntimeRegistrationKind.TOOL_REQUEST_INTERCEPT},
        target_name,
        fail,
    )
    try:
        assert tools.request_intercepts("tool", {}) == {"intercepted": True}
    finally:
        runtime_registrations.deregister_conditional_middleware_guardrail(gate_name)
        intercepts.deregister_tool_request(target_name)
