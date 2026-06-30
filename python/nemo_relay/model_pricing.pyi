# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, TypedDict

from nemo_relay import JsonObject

class _ConfigDiagnosticRequired(TypedDict):
    level: Literal["warning", "error"]
    code: str
    message: str

class ConfigDiagnostic(_ConfigDiagnosticRequired, total=False):
    component: str
    field: str

class ConfigReport(TypedDict):
    diagnostics: list[ConfigDiagnostic]

class TokenPricingRates:
    input_per_million: float
    output_per_million: float
    cache_read_per_million: float | None
    cache_write_per_million: float | None

    def __init__(
        self,
        input_per_million: float,
        output_per_million: float,
        cache_read_per_million: float | None = None,
        cache_write_per_million: float | None = None,
    ) -> None: ...
    def to_dict(self) -> JsonObject: ...

class PromptCachePricing:
    read_accounting: Literal["included_in_prompt_tokens", "separate"]

    def __init__(
        self,
        read_accounting: Literal["included_in_prompt_tokens", "separate"] = "included_in_prompt_tokens",
    ) -> None: ...
    def to_dict(self) -> JsonObject: ...

class TokenRateTier:
    rates: TokenPricingRates | JsonObject
    min_prompt_tokens: int | None
    max_prompt_tokens: int | None

    def __init__(
        self,
        rates: TokenPricingRates | JsonObject,
        min_prompt_tokens: int | None = None,
        max_prompt_tokens: int | None = None,
    ) -> None: ...
    def to_dict(self) -> JsonObject: ...

class PromptTokenThresholdRateSchedule:
    tiers: list[TokenRateTier | JsonObject]
    applies_to: Literal["full_request"]

    def __init__(
        self,
        tiers: list[TokenRateTier | JsonObject] = ...,
        applies_to: Literal["full_request"] = "full_request",
    ) -> None: ...
    def to_dict(self) -> JsonObject: ...

class ModelPricing:
    provider: str
    model_id: str
    pricing_as_of: str
    pricing_source: str
    aliases: list[str]
    currency: str
    unit: Literal["per_token", "per_request", "per_second", "gpu_hour"]
    rates: TokenPricingRates | JsonObject | None
    rate_schedule: PromptTokenThresholdRateSchedule | JsonObject | None
    prompt_cache: PromptCachePricing | JsonObject

    def __init__(
        self,
        provider: str,
        model_id: str,
        pricing_as_of: str,
        pricing_source: str,
        aliases: list[str] = ...,
        currency: str = "USD",
        unit: Literal["per_token", "per_request", "per_second", "gpu_hour"] = "per_token",
        rates: TokenPricingRates | JsonObject | None = None,
        rate_schedule: PromptTokenThresholdRateSchedule | JsonObject | None = None,
        prompt_cache: PromptCachePricing | JsonObject = ...,
    ) -> None: ...
    def to_dict(self) -> JsonObject: ...

class PricingCatalog:
    entries: list[ModelPricing | JsonObject]
    version: int

    def __init__(self, entries: list[ModelPricing | JsonObject] = ..., version: int = 1) -> None: ...
    def to_dict(self) -> JsonObject: ...

class InlineSource:
    catalog: PricingCatalog | JsonObject

    def __init__(self, catalog: PricingCatalog | JsonObject) -> None: ...
    def to_dict(self) -> JsonObject: ...

class FileSource:
    path: str

    def __init__(self, path: str) -> None: ...
    def to_dict(self) -> JsonObject: ...

class PricingConfig:
    sources: list[InlineSource | FileSource | JsonObject]

    def __init__(self, sources: list[InlineSource | FileSource | JsonObject] = ...) -> None: ...
    def to_dict(self) -> JsonObject: ...

PRICING_PLUGIN_KIND: Literal["pricing"]

class ComponentSpec:
    config: PricingConfig | JsonObject
    enabled: bool

    def __init__(self, config: PricingConfig | JsonObject, enabled: bool = True) -> None: ...
    def to_dict(self) -> JsonObject: ...

def validate_config(config: PricingConfig | JsonObject) -> ConfigReport: ...
