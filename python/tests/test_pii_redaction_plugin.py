# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the built-in PII redaction plugin config helpers."""

from __future__ import annotations

from nemo_relay import plugin
from nemo_relay.pii_redaction import (
    PII_REDACTION_PLUGIN_KIND,
    BuiltinConfig,
    ComponentSpec,
    ConfigPolicy,
    LocalModelConfig,
    PiiRedactionConfig,
    PiiRedactionProfile,
    validate_config,
)


class TestPiiRedactionConfigHelpers:
    def test_defaults_and_component_wrapper(self):
        assert BuiltinConfig().to_dict() == {
            "action": "remove",
            "target_paths": [],
        }
        assert ConfigPolicy().to_dict() == {
            "unknown_component": "warn",
            "unknown_field": "warn",
            "unsupported_value": "error",
        }
        assert LocalModelConfig().to_dict() == {}
        assert LocalModelConfig(
            backend="acme.pii/detector",
            model_id="pii-model-v1",
            detector_profile="default",
            target_paths=["/message"],
            target_path_patterns=["/messages/*/content"],
            min_score=0.6,
            excluded_labels=["CITY"],
            replacement="[PRIVATE]",
            allow_network=False,
            max_latency_ms=250,
        ).to_dict() == {
            "backend": "acme.pii/detector",
            "model_id": "pii-model-v1",
            "detector_profile": "default",
            "target_paths": ["/message"],
            "target_path_patterns": ["/messages/*/content"],
            "min_score": 0.6,
            "excluded_labels": ["CITY"],
            "replacement": "[PRIVATE]",
            "allow_network": False,
            "max_latency_ms": 250,
        }

        wrapped = ComponentSpec(PiiRedactionConfig()).to_dict()
        assert wrapped["kind"] == PII_REDACTION_PLUGIN_KIND
        assert wrapped["enabled"] is True
        wrapped_config = wrapped["config"]
        assert isinstance(wrapped_config, dict)
        assert wrapped_config["version"] == 1
        assert wrapped_config["mode"] == "builtin"
        assert wrapped_config["mark"] is True

        opted_out = PiiRedactionConfig(mark=False).to_dict()
        assert opted_out["mark"] is False

    def test_profile_config_omits_legacy_top_level_fields(self):
        config = PiiRedactionConfig(
            codec="openai_chat",
            profiles=[
                PiiRedactionProfile(
                    mode="builtin",
                    builtin=BuiltinConfig(detector="email"),
                ),
                PiiRedactionProfile(
                    mode="local_model",
                    priority=110,
                    local=LocalModelConfig(
                        backend="acme.pii/detector",
                        target_path_patterns=["/messages/*/content"],
                    ),
                ),
            ],
        ).to_dict()

        profiles = config["profiles"]
        assert isinstance(profiles, list)
        local_profile = profiles[1]
        assert isinstance(local_profile, dict)
        local = local_profile["local"]
        assert isinstance(local, dict)
        assert local["backend"] == "acme.pii/detector"
        assert "mode" not in config
        assert "input" not in config
        assert validate_config(config)["diagnostics"] == []

    def test_validation_rejects_bad_values(self):
        report = validate_config(
            PiiRedactionConfig(
                input=False,
                output=False,
                builtin=BuiltinConfig(
                    action="mask",
                    detector="not_a_detector",
                ),
            )
        )
        assert any(diag.get("field") == "builtin.detector" for diag in report["diagnostics"])

    def test_component_configures_plugin_validation(self):
        report = plugin.validate(
            plugin.PluginConfig(
                components=[
                    ComponentSpec(
                        PiiRedactionConfig(
                            input=False,
                            output=False,
                            builtin=BuiltinConfig(
                                action="mask",
                                detector="email",
                            ),
                        )
                    )
                ]
            )
        )
        assert report["diagnostics"] == []

    def test_list_kinds_includes_builtin_pii_redaction(self):
        assert PII_REDACTION_PLUGIN_KIND in plugin.list_kinds()
