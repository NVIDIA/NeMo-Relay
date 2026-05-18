# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import types

import pytest


@pytest.fixture(name="integration_langgraph", scope="session", autouse=True)
def integration_langgraph_fixture(integration_langgraph: types.ModuleType) -> types.ModuleType:
    """
    Override the integration_langgraph fixture to make it autouse
    """
    return integration_langgraph
