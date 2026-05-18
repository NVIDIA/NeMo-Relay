# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import types

import pytest


@pytest.fixture(name="integration_deepagents", scope="session", autouse=True)
def integration_deepagents_fixture(integration_deepagents: types.ModuleType) -> types.ModuleType:
    """
    Override the integration_deepagents fixture to make it autouse
    """
    return integration_deepagents
