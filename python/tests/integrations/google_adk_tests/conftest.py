# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import types
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest
from _helpers import ScriptedLlmFactory

_IGNORED_WARNINGS: Final = (
    r"ignore:\[EXPERIMENTAL\] feature:UserWarning",
    r"ignore:Items? .+ on TypedDict class .+ `ReadOnly` qualifier:UserWarning",
    r"ignore:Accessing the 'model_(computed_)?fields' attribute:pydantic.PydanticDeprecatedSince211:litellm\.",
)


@pytest.fixture(name="integration_google_adk", scope="session", autouse=True)
def integration_google_adk_fixture(integration_google_adk: types.ModuleType) -> types.ModuleType:
    """Override the `integration_google_adk` fixture so that it applies to every test in this directory."""
    return integration_google_adk


@pytest.fixture(name="litellm_local_cost_map", scope="session", autouse=True)
def litellm_local_cost_map_fixture() -> Iterator[None]:
    """Make LiteLLM use its bundled model cost map instead of downloading one."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        yield


@pytest.fixture(name="scripted_llm")
def scripted_llm_fixture() -> ScriptedLlmFactory:
    return ScriptedLlmFactory()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Filter ADK experimental and LiteLLM pydantic warnings for tests in this directory."""
    here = Path(__file__).parent
    for item in items:
        if here in item.path.parents:
            item.add_marker(pytest.mark.filterwarnings(*_IGNORED_WARNINGS))
