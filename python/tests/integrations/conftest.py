# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import types

import pytest


@pytest.fixture(name="integration_langchain", scope='session')
def integration_langchain_fixture() -> types.ModuleType:
    """
    Use for integration tests that require LangChain to be installed.
    """
    try:
        import langchain
        return langchain
    except Exception:
        pytest.skip(reason="langchain must be installed to run LangChain based tests")


@pytest.fixture(name="integration_langgraph", scope='session')
def integration_langgraph_fixture(integration_langchain: types.ModuleType) -> types.ModuleType:
    """
    Use for integration tests that require LangGraph to be installed.
    """
    try:
        import langgraph
        return langgraph
    except Exception:
        pytest.skip(reason="langgraph must be installed to run LangGraph based tests")


@pytest.fixture(name="integration_deepagents", scope='session')
def integration_deepagents_fixture(integration_langgraph: types.ModuleType) -> types.ModuleType:
    """
    Use for integration tests that require Deep Agents to be installed.
    """
    try:
        import deepagents
        return deepagents
    except Exception:
        pytest.skip(reason="deepagents must be installed to run Deep Agents based tests")
