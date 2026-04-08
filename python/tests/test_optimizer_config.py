# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility-oriented tests for the dynamic optimizer config contract."""

from nat_nexus.optimizer import (
    BackendSpec,
    ComponentSpec,
    ConfigPolicy,
    OptimizerConfig,
    StateConfig,
    validate_optimizer_config,
)


class TestDynamicConfigContract:
    def test_unknown_field_warns_by_default(self):
        report = validate_optimizer_config(
            OptimizerConfig(
                components=[
                    ComponentSpec(
                        kind="tool_parallelism",
                        config={"mode": "observe_only", "future_flag": True},
                    )
                ]
            )
        )
        assert any(diag["code"] == "optimizer.unknown_field" for diag in report["diagnostics"])

    def test_invalid_known_value_can_be_made_strict(self):
        report = validate_optimizer_config(
            OptimizerConfig(
                policy=ConfigPolicy(unsupported_value="error"),
                components=[
                    ComponentSpec(
                        kind="tool_parallelism",
                        config={"mode": "definitely_not_supported"},
                    )
                ],
            )
        )
        assert any(diag["code"] == "optimizer.unsupported_value" for diag in report["diagnostics"])

    def test_missing_state_warns_for_telemetry(self):
        report = validate_optimizer_config(OptimizerConfig(components=[ComponentSpec(kind="telemetry")]))
        assert any(diag["code"] == "optimizer.component_disabled_missing_state" for diag in report["diagnostics"])

    def test_in_memory_state_produces_clean_report(self):
        report = validate_optimizer_config(
            OptimizerConfig(
                state=StateConfig(backend=BackendSpec.in_memory()),
                components=[ComponentSpec(kind="telemetry")],
            )
        )
        assert report["diagnostics"] == []
