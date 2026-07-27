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
