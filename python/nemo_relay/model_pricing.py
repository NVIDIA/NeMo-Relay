# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model pricing plugin configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Literal, Protocol, TypedDict, cast

from nemo_relay import Json, JsonObject
from nemo_relay import plugin as plugin_module


class _ConfigDiagnosticRequired(TypedDict):
    level: Literal["warning", "error"]
    code: str
    message: str


class ConfigDiagnostic(_ConfigDiagnosticRequired, total=False):
    """One model pricing validation diagnostic."""

    component: str
    field: str


class ConfigReport(TypedDict):
    """Validation report for model pricing configuration."""

    diagnostics: list[ConfigDiagnostic]


class _SupportsToDict(Protocol):
    def to_dict(self) -> JsonObject: ...


def _normalize(value: object) -> Json:
    if hasattr(value, "to_dict"):
        return cast(_SupportsToDict, value).to_dict()
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field_info.name: _normalize(field_value)
            for field_info in fields(value)
            if (field_value := getattr(value, field_info.name)) is not None
        }
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        return {cast(str, key): _normalize(val) for key, val in value.items() if val is not None}
    return cast(Json, value)


def _normalize_object(value: object) -> JsonObject:
    return cast(JsonObject, _normalize(value))


@dataclass(slots=True)
class TokenPricingRates:
    """Per-token model rates expressed per one million tokens."""

    input_per_million: float
    output_per_million: float
    cache_read_per_million: float | None = None
    cache_write_per_million: float | None = None

    def to_dict(self) -> JsonObject:
        """Serialize these rates to the canonical JSON object shape."""
        return _normalize_object(
            {
                "input_per_million": self.input_per_million,
                "output_per_million": self.output_per_million,
                "cache_read_per_million": self.cache_read_per_million,
                "cache_write_per_million": self.cache_write_per_million,
            }
        )


@dataclass(slots=True)
class PromptCachePricing:
    """Prompt-cache accounting settings for a model pricing entry."""

    read_accounting: Literal["included_in_prompt_tokens", "separate"] = "included_in_prompt_tokens"

    def to_dict(self) -> JsonObject:
        """Serialize this prompt-cache config to the canonical JSON object shape."""
        return {"read_accounting": self.read_accounting}


@dataclass(slots=True)
class TokenRateTier:
    """One threshold tier in a token rate schedule."""

    rates: TokenPricingRates | JsonObject
    min_prompt_tokens: int | None = None
    max_prompt_tokens: int | None = None

    def to_dict(self) -> JsonObject:
        """Serialize this rate tier to the canonical JSON object shape."""
        return _normalize_object(
            {
                "min_prompt_tokens": self.min_prompt_tokens,
                "max_prompt_tokens": self.max_prompt_tokens,
                "rates": self.rates,
            }
        )


@dataclass(slots=True)
class PromptTokenThresholdRateSchedule:
    """Rate schedule selected by full-request prompt token thresholds."""

    tiers: list[TokenRateTier | JsonObject] = field(default_factory=list)
    applies_to: Literal["full_request"] = "full_request"

    def to_dict(self) -> JsonObject:
        """Serialize this rate schedule to the canonical JSON object shape."""
        return _normalize_object(
            {
                "type": "prompt_token_threshold",
                "applies_to": self.applies_to,
                "tiers": self.tiers,
            }
        )


@dataclass(slots=True)
class ModelPricing:
    """One model pricing catalog entry."""

    provider: str
    model_id: str
    pricing_as_of: str
    pricing_source: str
    aliases: list[str] = field(default_factory=list)
    currency: str = "USD"
    unit: Literal["per_token", "per_request", "per_second", "gpu_hour"] = "per_token"
    rates: TokenPricingRates | JsonObject | None = None
    rate_schedule: PromptTokenThresholdRateSchedule | JsonObject | None = None
    prompt_cache: PromptCachePricing | JsonObject = field(default_factory=PromptCachePricing)

    def to_dict(self) -> JsonObject:
        """Serialize this catalog entry to the canonical JSON object shape."""
        return _normalize_object(
            {
                "provider": self.provider,
                "model_id": self.model_id,
                "aliases": self.aliases,
                "currency": self.currency,
                "unit": self.unit,
                "rates": self.rates,
                "rate_schedule": self.rate_schedule,
                "prompt_cache": self.prompt_cache,
                "pricing_as_of": self.pricing_as_of,
                "pricing_source": self.pricing_source,
            }
        )


@dataclass(slots=True)
class PricingCatalog:
    """Inline model pricing catalog payload."""

    entries: list[ModelPricing | JsonObject] = field(default_factory=list)
    version: int = 1

    def to_dict(self) -> JsonObject:
        """Serialize this catalog to the canonical JSON object shape."""
        return _normalize_object(
            {
                "version": self.version,
                "entries": self.entries,
            }
        )


@dataclass(slots=True)
class InlineSource:
    """Pricing source backed by an inline catalog."""

    catalog: PricingCatalog | JsonObject

    def to_dict(self) -> JsonObject:
        """Serialize this source to the canonical JSON object shape."""
        return _normalize_object(
            {
                "type": "inline",
                "catalog": self.catalog,
            }
        )


@dataclass(slots=True)
class FileSource:
    """Pricing source backed by a JSON catalog file."""

    path: str

    def to_dict(self) -> JsonObject:
        """Serialize this source to the canonical JSON object shape."""
        return {
            "type": "file",
            "path": self.path,
        }


@dataclass(slots=True)
class PricingConfig:
    """Canonical config document for the top-level model pricing component."""

    sources: list[InlineSource | FileSource | JsonObject] = field(default_factory=list)

    def to_dict(self) -> JsonObject:
        """Serialize this model pricing config to the canonical JSON object shape."""
        return _normalize_object({"sources": self.sources})


PRICING_PLUGIN_KIND = "pricing"


@dataclass(slots=True)
class ComponentSpec:
    """Top-level model pricing component wrapper."""

    config: PricingConfig | JsonObject
    enabled: bool = True

    def to_dict(self) -> JsonObject:
        """Serialize this component to the canonical plugin shape."""
        return {
            "kind": PRICING_PLUGIN_KIND,
            "enabled": self.enabled,
            "config": _normalize_object(self.config),
        }


def validate_config(config: PricingConfig | JsonObject) -> ConfigReport:
    """Validate a model pricing config document without activating it."""
    report = plugin_module.validate(
        plugin_module.PluginConfig(
            components=[ComponentSpec(config)],
        )
    )
    return cast(ConfigReport, report)


__all__ = [
    "ComponentSpec",
    "ConfigDiagnostic",
    "ConfigReport",
    "FileSource",
    "InlineSource",
    "ModelPricing",
    "PRICING_PLUGIN_KIND",
    "PromptCachePricing",
    "PromptTokenThresholdRateSchedule",
    "PricingCatalog",
    "PricingConfig",
    "TokenPricingRates",
    "TokenRateTier",
    "validate_config",
]
