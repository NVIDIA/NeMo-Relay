# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for adaptive config validation through the plugin system."""

from pathlib import Path
from typing import Literal, cast

from nemo_relay import adaptive as adaptive_module
from nemo_relay import plugin
from nemo_relay.adaptive import (
    AcgConfig,
    AcgStabilityThresholds,
    AdaptiveConfig,
    BackendSpec,
    ComponentSpec,
    ConfigPolicy,
    ConvergenceConfig,
    StateConfig,
    TelemetryConfig,
    ToolParallelismConfig,
)


class TestDynamicConfigContract:
    def test_file_covers_canonical_cache_telemetry_helper(self):
        source = Path(__file__).read_text()
        helper_call = "adaptive_module" + ".build_cache_telemetry_event("
        assert helper_call in source

    def test_validate_config_exposes_native_validation_without_plugin_wrapper(self):
        report = adaptive_module.validate_config(
            {
                "version": 1,
                "telemetry": {},
            }
        )

        assert any(diag["code"] == "adaptive.section_disabled_missing_state" for diag in report["diagnostics"])

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

    def test_canonical_cache_telemetry_helper_preserves_missing_facts_diagnosis(self):
        event = adaptive_module.build_cache_telemetry_event(
            provider="anthropic",
            request_id="00000000-0000-0000-0000-000000000102",
            usage={
                "prompt_tokens": 300,
                "completion_tokens": 50,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
            },
            request_facts={
                "provider": "anthropic",
                "stable_prefix_length": 0,
                "missing_facts": ["acg_stability_unavailable"],
            },
            agent_id="test-adaptive-telemetry",
            template_version="unknown",
            toolset_hash="unknown",
            model_family="claude-sonnet-4-20250514",
            tenant_scope="default",
        )

        assert event is not None
        assert event["provider"] == "anthropic"
        assert event["miss_reason"] == {"reason": "unknown"}
        miss_diagnosis = cast(dict[str, object], event["miss_diagnosis"])
        evidence = cast(dict[str, object], miss_diagnosis["evidence"])
        assert evidence["missing_facts"] == ["acg_stability_unavailable"]

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

    def test_openai_acg_config_serializes_without_transport_fields(self):
        assert AcgConfig(provider="openai").to_dict() == {
            "provider": "openai",
            "observation_window": 100,
            "priority": 50,
            "stability_thresholds": {
                "stable_threshold": 0.95,
                "semi_stable_threshold": 0.5,
                "min_observations_for_full_confidence": 20,
            },
        }
        assert AcgConfig(convergence=ConvergenceConfig(enabled=True)).to_dict()["convergence"] == {
            "enabled": True,
            "epsilon": 0.001,
            "stability_window": 3,
        }

    def test_acg_config_allows_threshold_overrides(self):
        assert AcgConfig(
            stability_thresholds=AcgStabilityThresholds(
                stable_threshold=0.99,
                min_observations_for_full_confidence=12,
            )
        ).to_dict()["stability_thresholds"] == {
            "stable_threshold": 0.99,
            "semi_stable_threshold": 0.5,
            "min_observations_for_full_confidence": 12,
        }

    def test_canonical_cache_telemetry_helper_supports_openai_provider(self):
        event = adaptive_module.build_cache_telemetry_event(
            provider="openai",
            request_id="00000000-0000-0000-0000-000000000104",
            usage={
                "prompt_tokens": 300,
                "completion_tokens": 50,
                "cache_read_tokens": 150,
                "cache_write_tokens": 999,
            },
            request_facts={
                "provider": "openai",
                "stable_prefix_length": 0,
                "missing_facts": ["acg_stability_unavailable"],
            },
            agent_id="test-adaptive-openai-telemetry",
            template_version="unknown",
            toolset_hash="unknown",
            model_family="gpt-4.1-mini",
            tenant_scope="default",
        )

        assert event is not None
        assert event["provider"] == "openai"
        assert event["cache_read_tokens"] == 150
        assert event["cache_creation_tokens"] == 0
        assert "miss_reason" not in event
