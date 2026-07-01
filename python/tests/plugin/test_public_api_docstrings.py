# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Documentation coverage tests for the public Python plugin API."""

from __future__ import annotations

import inspect
import os
from collections.abc import Callable
from typing import Any

import pytest

if os.environ.get("NEMO_RELAY_SKIP_PYTHON_PLUGIN_TESTS") == "1":
    pytest.skip("grpcio is unavailable for Python plugin SDK tests on this runner", allow_module_level=True)

pytest.importorskip("grpc")
import nemo_relay_plugin  # noqa: E402

_TYPE_ALIASES = {
    "AnnotatedLlmRequest",
    "Event",
    "Json",
    "LlmRequest",
    "SubscriberCallback",
    "ToolSanitizeCallback",
    "ToolConditionalCallback",
    "ToolRequestCallback",
    "ToolExecutionCallback",
    "LlmSanitizeRequestCallback",
    "LlmSanitizeResponseCallback",
    "LlmConditionalCallback",
    "LlmRequestCallback",
    "LlmExecutionCallback",
    "LlmStreamExecutionCallback",
}


def test_public_exports_have_documentation():
    package_doc = inspect.getdoc(nemo_relay_plugin) or ""

    for name in nemo_relay_plugin.__all__:
        assert f"{name}:" in package_doc, f"{name} is missing from the package module docstring"
        if name in _TYPE_ALIASES:
            continue
        value = getattr(nemo_relay_plugin, name)
        assert value.__doc__ and value.__doc__.strip(), f"{name} is missing a direct public docstring"


def test_public_class_members_have_docstrings():
    for exported_name in nemo_relay_plugin.__all__:
        exported = getattr(nemo_relay_plugin, exported_name)
        if not inspect.isclass(exported):
            continue
        for member_name, member in inspect.getmembers_static(exported):
            if member_name.startswith("_"):
                continue
            callable_member = _documented_callable(member)
            if callable_member is not None:
                assert inspect.getdoc(callable_member), f"{exported_name}.{member_name} is missing a public docstring"


def _documented_callable(member: Any) -> Callable[..., Any] | None:
    if isinstance(member, property):
        return member.fget
    if isinstance(member, (classmethod, staticmethod)):
        return member.__func__
    if inspect.isfunction(member):
        return member
    return None
