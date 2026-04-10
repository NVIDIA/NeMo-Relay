# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for adaptive config validation through the plugin host."""

from typing import Literal, cast

from nemo_flow import plugin
from nemo_flow.adaptive import (
    AdaptiveConfig,
    BackendSpec,
    ComponentSpec,
    ConfigPolicy,
    StateConfig,
    TelemetryConfig,
    ToolParallelismConfig,
)


class TestDynamicConfigContract:
    def test_unknown_field_warns_by_default(self):
        report = plugin.validate(
            plugin.PluginConfig(
                components=[
                    plugin.ComponentSpec(
                        kind="adaptive",
                        config={
                            "version": 1,
                            "tool_parallelism": {
                                "mode": "observe_only",
                                "future_flag": True,
                            },
                        },
                    )
                ]
            )
        )
        assert any(diag["code"] == "adaptive.unknown_field" for diag in report["diagnostics"])

    def test_invalid_known_value_can_be_made_strict(self):
        invalid_mode = cast(
            Literal["observe_only", "inject_hints", "schedule"],
            "definitely_not_supported",
        )
        report = plugin.validate(
            plugin.PluginConfig(
                components=[
                    ComponentSpec(
                        AdaptiveConfig(
                            policy=ConfigPolicy(unsupported_value="error"),
                            tool_parallelism=ToolParallelismConfig(mode=invalid_mode),
                        )
                    )
                ],
            )
        )
        assert any(diag["code"] == "adaptive.unsupported_value" for diag in report["diagnostics"])

    def test_missing_state_warns_for_telemetry(self):
        report = plugin.validate(
            plugin.PluginConfig(
                components=[
                    plugin.ComponentSpec(
                        kind="adaptive",
                        config={"version": 1, "telemetry": {}},
                    )
                ]
            )
        )
        assert any(diag["code"] == "adaptive.section_disabled_missing_state" for diag in report["diagnostics"])

    def test_in_memory_state_produces_clean_report(self):
        report = plugin.validate(
            plugin.PluginConfig(
                components=[
                    ComponentSpec(
                        AdaptiveConfig(
                            state=StateConfig(backend=BackendSpec.in_memory()),
                            telemetry=TelemetryConfig(),
                        )
                    )
                ]
            )
        )
        assert report["diagnostics"] == []
