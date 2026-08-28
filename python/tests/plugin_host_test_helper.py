# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test helpers for the owned plugin-host lifecycle."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from nemo_relay import JsonObject, plugin


def validate_plugin_config(config: plugin.PluginConfig | JsonObject) -> plugin.ConfigReport:
    """Validate one static plugin configuration through the unified host API."""
    return plugin.validate(config)["config"]


@asynccontextmanager
async def activated_plugin_host(
    config: plugin.PluginConfig | JsonObject,
) -> AsyncIterator[plugin.PluginHostActivation]:
    """Own and deterministically close a plugin host for one test."""
    activation = await plugin.initialize(config)
    try:
        yield activation
    finally:
        await activation.close()
