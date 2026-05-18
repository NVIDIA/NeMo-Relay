# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import types

import pytest


@pytest.fixture(name="integration_langchain", scope="session", autouse=True)
def integration_langchain_fixture(integration_langchain: types.ModuleType) -> types.ModuleType:
    """
    Override the integration_langchain fixture to make it autouse
    """
    yield integration_langchain
