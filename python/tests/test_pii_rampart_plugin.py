# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_relay import plugin
from nemo_relay.pii_rampart import (
    RAMPART_MODEL_ID,
    RAMPART_MODEL_REVISION,
    RAMPART_PII_PLUGIN_KIND,
    ComponentSpec,
    RampartPiiConfig,
    validate_config,
)


def test_rampart_config_and_component_shape() -> None:
    config = RampartPiiConfig(
        model_path="/models/rampart",
        codec="openai_chat",
        target_path_patterns=["/messages/*/content"],
    )
    value = config.to_dict()
    assert value["model_path"] == "/models/rampart"
    assert value["max_windows_per_payload"] == 4
    assert value["inference_batch_size"] == 16
    assert RAMPART_MODEL_ID == "nationaldesignstudio/rampart"
    assert RAMPART_MODEL_REVISION == "b1993e4e68b082835b80ffc65acc03325ea2e501"
    component = ComponentSpec(config).to_dict()
    assert component["kind"] == RAMPART_PII_PLUGIN_KIND
    assert component["enabled"] is True


def test_rampart_validation_and_discovery() -> None:
    report = validate_config(
        RampartPiiConfig(
            model_path="relative/model",
            target_path_patterns=["/messages/pre*fix/content"],
        )
    )
    assert {diagnostic.get("field") for diagnostic in report["diagnostics"]} == {
        "model_path",
        "target_path_patterns",
    }
    assert RAMPART_PII_PLUGIN_KIND in plugin.list_kinds()
