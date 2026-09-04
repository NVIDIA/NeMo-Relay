# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test helpers for the owned plugin-host lifecycle."""

from __future__ import annotations

from nemo_relay import JsonObject, plugin


def validate_plugin_config(config: plugin.PluginConfig | JsonObject) -> plugin.ConfigReport:
    """Validate one static plugin configuration through the unified host API."""
    return plugin.validate_exact(config)["config"]


activated_plugin_host = plugin.activate
