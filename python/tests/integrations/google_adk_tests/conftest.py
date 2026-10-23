# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import types
from pathlib import Path

import pytest
from _helpers import ScriptedLlmFactory


@pytest.fixture(name="integration_google_adk", scope="session", autouse=True)
def integration_google_adk_fixture(integration_google_adk: types.ModuleType) -> types.ModuleType:
    """Override the `integration_google_adk` fixture so that it applies to every test in this directory."""
    return integration_google_adk


@pytest.fixture(name="scripted_llm")
def scripted_llm_fixture() -> ScriptedLlmFactory:
    return ScriptedLlmFactory()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Ignore the one-time experimental feature warnings that ADK emits in the tests of this directory."""
    here = Path(__file__).parent
    for item in items:
        if here in item.path.parents:
            item.add_marker(pytest.mark.filterwarnings(r"ignore:\[EXPERIMENTAL\] feature:UserWarning"))
