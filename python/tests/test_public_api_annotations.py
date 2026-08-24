# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for public Python API annotations and documentation."""

import inspect
from importlib import import_module
from importlib.util import find_spec
from types import ModuleType

import nemo_relay


def _public_modules() -> list[ModuleType]:
    """Return core modules plus installed optional integration modules."""
    modules: list[ModuleType] = [nemo_relay]
    modules.extend(
        value
        for value in vars(nemo_relay).values()
        if isinstance(value, ModuleType) and value.__name__.startswith("nemo_relay")
    )
    integration_dependencies = {
        "nemo_relay.integrations.langchain": ("langchain_core",),
        "nemo_relay.integrations.langgraph": ("langchain_core", "langgraph"),
        "nemo_relay.integrations.deepagents": ("langchain_core", "langgraph", "deepagents"),
    }
    for module_name, dependencies in integration_dependencies.items():
        if all(find_spec(dependency) is not None for dependency in dependencies):
            modules.append(import_module(module_name))
    return modules


def test_exported_functions_have_complete_type_annotations():
    modules = _public_modules()

    missing = []
    for module in modules:
        for name in getattr(module, "__all__", ()):
            function = getattr(module, name, None)
            if not inspect.isfunction(function):
                continue
            signature = inspect.signature(function)
            parameters = [
                parameter.name
                for parameter in signature.parameters.values()
                if parameter.annotation is inspect.Parameter.empty
            ]
            if parameters or signature.return_annotation is inspect.Signature.empty:
                missing.append(f"{module.__name__}.{name}: {', '.join(parameters) or 'return'}")

    assert not missing, "public API functions missing annotations:\n" + "\n".join(missing)


def test_exported_functions_have_comprehensive_docstrings():
    """Require public functions to document parameters and return behavior."""
    modules = _public_modules()

    missing = []
    for module in modules:
        for name in getattr(module, "__all__", ()):
            function = getattr(module, name, None)
            if not inspect.isfunction(function):
                continue
            docstring = inspect.getdoc(function) or ""
            signature = inspect.signature(function)
            requirements = ["Returns:"]
            if signature.parameters:
                requirements.append("Args:")
            absent = ["docstring"] if not docstring else []
            absent.extend(requirement for requirement in requirements if requirement not in docstring)
            if absent:
                missing.append(f"{module.__name__}.{name}: {', '.join(absent)}")

    assert not missing, "public API functions missing comprehensive docstrings:\n" + "\n".join(missing)
