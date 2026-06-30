# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the built-in model pricing plugin config helpers."""

from __future__ import annotations

from pathlib import Path

from nemo_relay import plugin
from nemo_relay.model_pricing import (
    PRICING_PLUGIN_KIND,
    ComponentSpec,
    FileSource,
    InlineSource,
    ModelPricing,
    PricingCatalog,
    PricingConfig,
    PromptCachePricing,
    PromptTokenThresholdRateSchedule,
    TokenPricingRates,
    TokenRateTier,
    validate_config,
)


class TestPricingConfigHelpers:
    def test_defaults_and_component_wrapper(self, tmp_path: Path):
        pricing_file = tmp_path / "pricing.json"
        rates = TokenPricingRates(1.0, 2.0, cache_read_per_million=0.25)
        entry = ModelPricing(
            provider="test",
            model_id="priced-model",
            pricing_as_of="2026-06-15",
            pricing_source="https://example.com/pricing",
            rates=rates,
        )
        config = PricingConfig(
            sources=[
                InlineSource(PricingCatalog(entries=[entry])),
                FileSource(str(pricing_file)),
            ]
        )

        assert PromptCachePricing().to_dict() == {
            "read_accounting": "included_in_prompt_tokens",
        }
        wrapped = ComponentSpec(config).to_dict()
        assert wrapped["kind"] == PRICING_PLUGIN_KIND
        assert wrapped["enabled"] is True
        wrapped_config = wrapped["config"]
        assert isinstance(wrapped_config, dict)
        assert wrapped_config["sources"][0]["type"] == "inline"
        assert wrapped_config["sources"][1] == {
            "type": "file",
            "path": str(pricing_file),
        }

    def test_rate_schedule_serialization(self):
        schedule = PromptTokenThresholdRateSchedule(
            tiers=[
                TokenRateTier(
                    min_prompt_tokens=128,
                    rates=TokenPricingRates(1.0, 2.0),
                )
            ]
        )

        assert schedule.to_dict() == {
            "type": "prompt_token_threshold",
            "applies_to": "full_request",
            "tiers": [
                {
                    "min_prompt_tokens": 128,
                    "rates": {
                        "input_per_million": 1.0,
                        "output_per_million": 2.0,
                    },
                }
            ],
        }

    def test_component_configures_plugin_validation(self):
        report = validate_config(
            PricingConfig(
                sources=[
                    InlineSource(
                        PricingCatalog(
                            entries=[
                                ModelPricing(
                                    provider="test",
                                    model_id="priced-model",
                                    pricing_as_of="2026-06-15",
                                    pricing_source="https://example.com/pricing",
                                    rates=TokenPricingRates(1.0, 2.0),
                                )
                            ]
                        )
                    )
                ]
            )
        )
        assert report["diagnostics"] == []

    def test_validation_rejects_invalid_catalog(self):
        report = validate_config(
            PricingConfig(
                sources=[
                    InlineSource(
                        PricingCatalog(
                            entries=[
                                ModelPricing(
                                    provider="test",
                                    model_id="priced-model",
                                    pricing_as_of="2026-06-15",
                                    pricing_source="https://example.com/pricing",
                                    rates=TokenPricingRates(-1.0, 2.0),
                                )
                            ]
                        )
                    )
                ]
            )
        )
        assert any(diag["code"] == "pricing.invalid_config" for diag in report["diagnostics"])

    def test_list_kinds_includes_builtin_pricing(self):
        assert PRICING_PLUGIN_KIND in plugin.list_kinds()
