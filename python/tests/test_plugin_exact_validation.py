# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact plugin validation rejects settings owned by the layered host."""

import pytest

from nemo_relay import plugin


@pytest.mark.parametrize("diagnostics", [None, True, {}, {"enabled": True}, {"enabled": "invalid"}])
def test_exact_validation_rejects_host_diagnostics(diagnostics):
    with pytest.raises(ValueError, match="diagnostics.*layered validation"):
        plugin.validate_exact({"version": 1, "components": [], "diagnostics": diagnostics})


def test_exact_validation_rejects_typed_host_diagnostics():
    with pytest.raises(ValueError, match="diagnostics.*layered validation"):
        plugin.validate_exact(plugin.PluginConfig(diagnostics={"enabled": True}))


def test_exact_validation_accepts_static_plugin_config():
    report = plugin.validate_exact(plugin.PluginConfig())
    assert report["dynamic_plugins"] == []
