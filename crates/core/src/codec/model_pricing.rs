// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Data-driven LLM model pricing used to layer cost estimates onto usage.
//!
//! Model pricing is deliberately separate from response normalization so adding
//! providers, aliases, or cache-accounting rules does not require editing
//! [`AnnotatedLlmResponse`](super::response::AnnotatedLlmResponse).

use std::collections::HashSet;
use std::path::PathBuf;
#[cfg(test)]
use std::sync::Mutex;
use std::sync::{Arc, LazyLock, RwLock};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::response::{AnnotatedLlmResponse, CostEstimate, CostSource, Usage};

const PRICING_CATALOG_VERSION: u32 = 1;

static ACTIVE_PRICING_RESOLVER: LazyLock<RwLock<Arc<PricingResolver>>> =
    LazyLock::new(|| RwLock::new(Arc::new(PricingResolver::default())));

#[cfg(test)]
pub(crate) fn pricing_test_mutex() -> &'static Mutex<()> {
    static PRICING_TEST_MUTEX: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));
    &PRICING_TEST_MUTEX
}

/// Errors produced while parsing or validating a model pricing catalog.
#[derive(Debug, Error)]
pub enum PricingCatalogError {
    /// The catalog was not valid JSON for the catalog schema.
    #[error("invalid model pricing catalog JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// Two entries or aliases normalize to the same model key.
    #[error("duplicate model pricing alias '{model}'")]
    DuplicateModelAlias {
        /// Normalized model key that appeared more than once.
        model: String,
    },
    /// The catalog schema version is not supported by this Relay build.
    #[error("unsupported model pricing catalog version {version}")]
    UnsupportedVersion {
        /// Version number from the catalog payload.
        version: u32,
    },
    /// A required text field was empty.
    #[error("model pricing entry {entry_index} has empty {field}")]
    EmptyField {
        /// Zero-based index of the invalid catalog entry.
        entry_index: usize,
        /// Name of the invalid field.
        field: String,
    },
    /// A price was negative or non-finite.
    #[error("model pricing entry {entry_index} has invalid {field}: {value}")]
    InvalidRate {
        /// Zero-based index of the invalid catalog entry.
        entry_index: usize,
        /// Name of the invalid rate field.
        field: String,
        /// Invalid field value.
        value: f64,
    },
    /// A model pricing catalog file could not be read.
    #[error("could not read model pricing catalog file '{}': {source}", path.display())]
    FileRead {
        /// Catalog path.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },
    /// The active model pricing resolver lock was poisoned.
    #[error("model pricing resolver lock poisoned: {0}")]
    LockPoisoned(String),
}

/// Collection of model pricing entries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PricingCatalog {
    /// Catalog schema version.
    pub version: u32,
    /// Pricing entries keyed by canonical model ID plus aliases.
    pub entries: Vec<ModelPricing>,
}

impl PricingCatalog {
    /// Parses and validates a model pricing catalog from JSON.
    pub fn from_json_str(catalog_json: &str) -> Result<Self, PricingCatalogError> {
        let catalog: Self = serde_json::from_str(catalog_json)?;
        catalog.validate()?;
        Ok(catalog)
    }

    /// Finds model pricing for a canonical model ID or alias.
    #[must_use]
    pub fn pricing_for_model(&self, model: &str) -> Option<ModelPricing> {
        self.pricing_for(None, model)
    }

    /// Finds model pricing for a provider/model pair, with model-only fallback.
    #[must_use]
    pub fn pricing_for(&self, provider: Option<&str>, model: &str) -> Option<ModelPricing> {
        let model_keys = normalized_model_lookup_keys(provider, model);
        if model_keys.is_empty() {
            return None;
        }

        model_keys.iter().find_map(|model_key| {
            self.entries
                .iter()
                .find(|entry| entry.matches_model(model_key))
                .cloned()
        })
    }

    fn validate(&self) -> Result<(), PricingCatalogError> {
        if self.version != PRICING_CATALOG_VERSION {
            return Err(PricingCatalogError::UnsupportedVersion {
                version: self.version,
            });
        }

        let mut seen = HashSet::new();

        for (entry_index, entry) in self.entries.iter().enumerate() {
            entry.validate(entry_index)?;

            for model_key in entry.provider_model_keys() {
                if !seen.insert(model_key.clone()) {
                    return Err(PricingCatalogError::DuplicateModelAlias { model: model_key });
                }
            }
        }

        Ok(())
    }
}

/// Runtime model pricing resolver configuration.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PricingConfig {
    /// Pricing sources in precedence order.
    #[serde(default)]
    pub sources: Vec<PricingSourceConfig>,
}

/// Declarative model pricing source supported by Relay configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PricingSourceConfig {
    /// Inline catalog entries from project, user, system, or plugin config.
    Inline {
        /// Inline catalog payload.
        catalog: PricingCatalog,
    },
    /// Catalog loaded from a JSON file.
    File {
        /// JSON model pricing catalog path.
        path: PathBuf,
    },
}

/// Pluggable model pricing source interface.
///
/// Database, service-backed, or enterprise-managed model pricing integrations should
/// implement this trait and return a validated catalog snapshot. The LLM hot
/// path uses [`PricingResolver`], so sources can refresh out-of-band without
/// making each response decode perform network or database I/O.
pub trait PricingSource: Send + Sync {
    /// Stable source name for diagnostics.
    fn source_name(&self) -> &str;

    /// Loads a catalog snapshot from this source.
    fn load_catalog(&self) -> Result<Option<PricingCatalog>, PricingCatalogError>;
}

/// Ordered model pricing lookup chain.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PricingResolver {
    catalogs: Vec<PricingCatalog>,
}

impl PricingResolver {
    /// Builds a resolver from already-loaded catalogs in precedence order.
    #[must_use]
    pub fn from_catalogs(catalogs: Vec<PricingCatalog>) -> Self {
        Self { catalogs }
    }

    /// Builds a resolver from declarative config.
    pub fn from_config(config: &PricingConfig) -> Result<Self, PricingCatalogError> {
        let mut catalogs = Vec::new();
        for source in &config.sources {
            match source {
                PricingSourceConfig::Inline { catalog } => {
                    catalog.validate()?;
                    catalogs.push(catalog.clone());
                }
                PricingSourceConfig::File { path } => {
                    let raw = std::fs::read_to_string(path).map_err(|source| {
                        PricingCatalogError::FileRead {
                            path: path.clone(),
                            source,
                        }
                    })?;
                    catalogs.push(PricingCatalog::from_json_str(&raw)?);
                }
            }
        }
        Ok(Self { catalogs })
    }

    /// Builds a resolver from imperative source implementations.
    pub fn from_sources(sources: Vec<Box<dyn PricingSource>>) -> Result<Self, PricingCatalogError> {
        let mut catalogs = Vec::new();
        for source in sources {
            if let Some(catalog) = source.load_catalog()? {
                catalog.validate()?;
                catalogs.push(catalog);
            }
        }
        Ok(Self { catalogs })
    }

    /// Finds model pricing for a canonical model ID or alias.
    #[must_use]
    pub fn pricing_for_model(&self, model: &str) -> Option<ModelPricing> {
        self.pricing_for(None, model)
    }

    /// Finds model pricing for a provider/model pair, with model-only fallback.
    #[must_use]
    pub fn pricing_for(&self, provider: Option<&str>, model: &str) -> Option<ModelPricing> {
        self.catalogs
            .iter()
            .find_map(|catalog| catalog.pricing_for(provider, model))
    }

    /// Estimates cost for a model/usage pair when model pricing is known.
    #[must_use]
    pub fn estimate_cost(&self, model: &str, usage: &Usage) -> Option<CostEstimate> {
        self.estimate_cost_for_provider(None, model, usage)
    }

    /// Estimates cost for a provider/model pair when model pricing is known.
    #[must_use]
    pub fn estimate_cost_for_provider(
        &self,
        provider: Option<&str>,
        model: &str,
        usage: &Usage,
    ) -> Option<CostEstimate> {
        self.pricing_for(provider, model)
            .and_then(|pricing| pricing.estimate_cost(usage))
    }
}

/// Per-token model pricing expressed in USD per one million tokens.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelPricing {
    /// Provider that owns this model pricing entry.
    pub provider: String,
    /// Canonical model ID for this model pricing entry.
    pub model_id: String,
    /// Additional model IDs that should use this model pricing.
    #[serde(default)]
    pub aliases: Vec<String>,
    /// ISO 4217 currency for this model pricing entry.
    #[serde(default = "default_pricing_currency")]
    pub currency: String,
    /// Billing unit represented by this model pricing entry.
    #[serde(default)]
    pub unit: PricingUnit,
    /// Token rates expressed as USD per one million tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rates: Option<TokenPricingRates>,
    /// Data-driven token rate schedule for threshold-based model pricing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rate_schedule: Option<TokenRateSchedule>,
    /// Prompt-cache accounting model for this provider/model.
    pub prompt_cache: PromptCachePricing,
    /// Date this model pricing entry was last verified.
    pub pricing_as_of: String,
    /// Source URL for this model pricing entry.
    pub pricing_source: String,
}

impl ModelPricing {
    /// Estimates cost for the provided token usage.
    #[must_use]
    pub fn estimate_cost(&self, usage: &Usage) -> Option<CostEstimate> {
        if self.unit != PricingUnit::PerToken {
            return None;
        }
        let prompt_tokens = usage.prompt_tokens.unwrap_or(0);
        let completion_tokens = usage.completion_tokens.unwrap_or(0);
        let cache_read_tokens = usage.cache_read_tokens.unwrap_or(0);
        let cache_write_tokens = usage.cache_write_tokens.unwrap_or(0);
        let rates = self.rates_for_usage(usage)?;

        if prompt_tokens == 0
            && completion_tokens == 0
            && cache_read_tokens == 0
            && cache_write_tokens == 0
        {
            return None;
        }

        let billable_prompt_tokens =
            if self.prompt_cache.read_accounting == CacheReadAccounting::IncludedInPromptTokens {
                prompt_tokens.saturating_sub(cache_read_tokens)
            } else {
                prompt_tokens
            };

        let input_cost = cost_component_if_nonzero(billable_prompt_tokens, rates.input_per_million);
        let output_cost = cost_component_if_nonzero(completion_tokens, rates.output_per_million);
        let cache_read_cost = rates
            .cache_read_per_million
            .and_then(|price| cost_component_if_nonzero(cache_read_tokens, price));
        let cache_write_cost = rates
            .cache_write_per_million
            .and_then(|price| cost_component_if_nonzero(cache_write_tokens, price));

        let total: f64 = [input_cost, output_cost, cache_read_cost, cache_write_cost]
            .into_iter()
            .flatten()
            .sum();

        Some(CostEstimate {
            total: Some(round_cost_amount(total)),
            currency: self.currency.clone(),
            input: input_cost,
            output: output_cost,
            cache_read: cache_read_cost,
            cache_write: cache_write_cost,
            source: CostSource::ModelPricing,
            pricing_provider: Some(self.provider.clone()),
            pricing_model: Some(self.model_id.clone()),
            pricing_as_of: Some(self.pricing_as_of.clone()),
            pricing_source: Some(self.pricing_source.clone()),
        })
    }

    fn rates_for_usage(&self, usage: &Usage) -> Option<TokenPricingRates> {
        if let Some(schedule) = &self.rate_schedule {
            return schedule.rates_for_usage(usage);
        }
        self.rates
    }

    fn matches_model(&self, lookup: &ModelLookupKey) -> bool {
        if let Some(provider) = lookup.provider.as_deref()
            && normalized_provider_name(&self.provider) != provider
        {
            return false;
        }

        self.model_keys().any(|key| key == lookup.model)
    }

    fn model_keys(&self) -> impl Iterator<Item = String> + '_ {
        std::iter::once(&self.model_id)
            .chain(self.aliases.iter())
            .map(|model| normalized_model_name(model))
            .filter(|model| !model.is_empty())
    }

    fn provider_model_keys(&self) -> impl Iterator<Item = String> + '_ {
        let provider = normalized_provider_name(&self.provider);
        self.model_keys()
            .map(move |model| format!("{provider}/{model}"))
    }

    fn validate(&self, entry_index: usize) -> Result<(), PricingCatalogError> {
        validate_nonempty(entry_index, "provider", &self.provider)?;
        validate_nonempty(entry_index, "model_id", &self.model_id)?;
        validate_nonempty(entry_index, "currency", &self.currency)?;
        validate_nonempty(entry_index, "pricing_as_of", &self.pricing_as_of)?;
        validate_nonempty(entry_index, "pricing_source", &self.pricing_source)?;

        if self.unit == PricingUnit::PerToken
            && self.rates.is_none()
            && self.rate_schedule.is_none()
        {
            return Err(PricingCatalogError::EmptyField {
                entry_index,
                field: "rates or rate_schedule".to_string(),
            });
        }
        if let Some(rates) = &self.rates {
            rates.validate(entry_index, "rates")?;
        }
        if let Some(schedule) = &self.rate_schedule {
            schedule.validate(entry_index)?;
        }

        Ok(())
    }
}

/// Billing unit represented by a model pricing entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PricingUnit {
    /// Token-based model pricing.
    #[default]
    PerToken,
    /// Request-based model pricing, reserved for future estimation.
    PerRequest,
    /// Time-based model pricing, reserved for future estimation.
    PerSecond,
    /// GPU-hour amortized model pricing for self-hosted models, reserved for future estimation.
    GpuHour,
}

/// Token rates expressed as USD per one million tokens.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TokenPricingRates {
    /// Uncached prompt/input token price.
    pub input_per_million: f64,
    /// Completion/output token price.
    pub output_per_million: f64,
    /// Cached prompt/input token read price.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_per_million: Option<f64>,
    /// Prompt cache write price.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_write_per_million: Option<f64>,
}

impl TokenPricingRates {
    fn validate(&self, entry_index: usize, field_prefix: &str) -> Result<(), PricingCatalogError> {
        validate_rate(
            entry_index,
            format!("{field_prefix}.input_per_million"),
            self.input_per_million,
        )?;
        validate_rate(
            entry_index,
            format!("{field_prefix}.output_per_million"),
            self.output_per_million,
        )?;
        if let Some(value) = self.cache_read_per_million {
            validate_rate(
                entry_index,
                format!("{field_prefix}.cache_read_per_million"),
                value,
            )?;
        }
        if let Some(value) = self.cache_write_per_million {
            validate_rate(
                entry_index,
                format!("{field_prefix}.cache_write_per_million"),
                value,
            )?;
        }
        Ok(())
    }
}

/// Data-driven token rate schedule for model pricing with request thresholds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TokenRateSchedule {
    /// Selects one full-request rate tier based on prompt/input tokens.
    PromptTokenThreshold {
        /// How selected tier rates apply to tokens.
        #[serde(default)]
        applies_to: RateScheduleApplication,
        /// Ordered threshold tiers.
        tiers: Vec<TokenRateTier>,
    },
}

impl TokenRateSchedule {
    fn rates_for_usage(&self, usage: &Usage) -> Option<TokenPricingRates> {
        match self {
            Self::PromptTokenThreshold { applies_to, tiers } => {
                if *applies_to != RateScheduleApplication::FullRequest {
                    return None;
                }
                let prompt_tokens = usage.prompt_tokens?;
                tiers
                    .iter()
                    .find(|tier| tier.matches_prompt_tokens(prompt_tokens))
                    .map(|tier| tier.rates)
            }
        }
    }

    fn validate(&self, entry_index: usize) -> Result<(), PricingCatalogError> {
        match self {
            Self::PromptTokenThreshold { tiers, .. } if tiers.is_empty() => {
                Err(PricingCatalogError::EmptyField {
                    entry_index,
                    field: "rate_schedule.tiers".to_string(),
                })
            }
            Self::PromptTokenThreshold { tiers, .. } => {
                for (tier_index, tier) in tiers.iter().enumerate() {
                    tier.validate(entry_index, tier_index)?;
                }
                Ok(())
            }
        }
    }
}

/// How a selected rate-schedule tier applies to billable usage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RateScheduleApplication {
    /// Apply the selected tier rates to the entire request.
    #[default]
    FullRequest,
}

/// A model pricing tier selected by prompt/input token count.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TokenRateTier {
    /// Inclusive lower bound for prompt tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_prompt_tokens: Option<u64>,
    /// Inclusive upper bound for prompt tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_prompt_tokens: Option<u64>,
    /// Rates to apply when this tier is selected.
    pub rates: TokenPricingRates,
}

impl TokenRateTier {
    fn matches_prompt_tokens(&self, prompt_tokens: u64) -> bool {
        self.min_prompt_tokens
            .is_none_or(|min| prompt_tokens >= min)
            && self
                .max_prompt_tokens
                .is_none_or(|max| prompt_tokens <= max)
    }

    fn validate(&self, entry_index: usize, tier_index: usize) -> Result<(), PricingCatalogError> {
        if let (Some(min), Some(max)) = (self.min_prompt_tokens, self.max_prompt_tokens)
            && min > max
        {
            return Err(PricingCatalogError::InvalidRate {
                entry_index,
                field: "rate_schedule.tiers.prompt_tokens".to_string(),
                value: min as f64,
            });
        }
        self.rates.validate(
            entry_index,
            &format!("rate_schedule.tiers[{tier_index}].rates"),
        )
    }
}

/// Prompt-cache accounting rules for a model pricing entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptCachePricing {
    /// Whether cache-read tokens are included in `prompt_tokens`.
    pub read_accounting: CacheReadAccounting,
}

/// How cache-read tokens relate to prompt token counts in provider usage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheReadAccounting {
    /// `cache_read_tokens` are already included in `prompt_tokens`.
    IncludedInPromptTokens,
    /// `cache_read_tokens` are separate from `prompt_tokens`.
    Separate,
}

/// Returns known model pricing for a model ID.
///
/// Unknown models return `None` so response handling and observability export
/// can continue without inventing a cost.
#[must_use]
pub fn pricing_for_model(model: &str) -> Option<ModelPricing> {
    active_pricing_resolver().pricing_for_model(model)
}

/// Returns known model pricing for a provider/model pair.
#[must_use]
pub fn pricing_for_provider(provider: Option<&str>, model: &str) -> Option<ModelPricing> {
    active_pricing_resolver().pricing_for(provider, model)
}

/// Estimates USD cost for a model/usage pair when model pricing is known.
#[must_use]
pub fn estimate_cost(model: &str, usage: &Usage) -> Option<CostEstimate> {
    active_pricing_resolver().estimate_cost(model, usage)
}

/// Estimates USD cost for a provider/model pair when model pricing is known.
#[must_use]
pub fn estimate_cost_for_provider(
    provider: Option<&str>,
    model: &str,
    usage: &Usage,
) -> Option<CostEstimate> {
    active_pricing_resolver().estimate_cost_for_provider(provider, model, usage)
}

/// Estimates USD cost using the provided catalog.
#[must_use]
pub fn estimate_cost_with_catalog(
    catalog: &PricingCatalog,
    model: &str,
    usage: &Usage,
) -> Option<CostEstimate> {
    catalog
        .pricing_for_model(model)
        .and_then(|pricing| pricing.estimate_cost(usage))
}

/// Estimates USD cost using the provided catalog and provider/model pair.
#[must_use]
pub fn estimate_cost_with_provider(
    catalog: &PricingCatalog,
    provider: Option<&str>,
    model: &str,
    usage: &Usage,
) -> Option<CostEstimate> {
    catalog
        .pricing_for(provider, model)
        .and_then(|pricing| pricing.estimate_cost(usage))
}

/// Returns the active process-wide model pricing resolver.
#[must_use]
pub fn active_pricing_resolver() -> Arc<PricingResolver> {
    ACTIVE_PRICING_RESOLVER
        .read()
        .map(|resolver| Arc::clone(&resolver))
        .unwrap_or_else(|_| Arc::new(PricingResolver::default()))
}

/// Replaces the active process-wide model pricing resolver.
pub fn set_active_pricing_resolver(resolver: PricingResolver) -> Result<(), PricingCatalogError> {
    let mut guard = ACTIVE_PRICING_RESOLVER
        .write()
        .map_err(|err| PricingCatalogError::LockPoisoned(err.to_string()))?;
    *guard = Arc::new(resolver);
    Ok(())
}

/// Restores the active process-wide model pricing resolver to an empty resolver.
pub fn reset_active_pricing_resolver() -> Result<(), PricingCatalogError> {
    set_active_pricing_resolver(PricingResolver::default())
}

/// Adds a model pricing estimate to a normalized response when cost is missing.
///
/// Existing provider-reported or caller-supplied costs are preserved.
pub fn attach_estimated_cost(response: &mut AnnotatedLlmResponse) {
    attach_estimated_cost_for_provider(response, None);
}

/// Adds a provider-aware model pricing estimate to a normalized response when cost is missing.
///
/// Existing provider-reported or caller-supplied costs are preserved.
pub fn attach_estimated_cost_for_provider(
    response: &mut AnnotatedLlmResponse,
    provider: Option<&str>,
) {
    if response
        .usage
        .as_ref()
        .and_then(|usage| usage.cost.as_ref())
        .is_some()
    {
        return;
    }

    let Some(model) = response.model.clone() else {
        return;
    };
    let Some(usage) = response.usage.as_mut() else {
        return;
    };

    usage.cost = estimate_cost_for_provider(provider, &model, usage);
}

fn validate_nonempty(
    entry_index: usize,
    field: &'static str,
    value: &str,
) -> Result<(), PricingCatalogError> {
    if value.trim().is_empty() {
        return Err(PricingCatalogError::EmptyField {
            entry_index,
            field: field.to_string(),
        });
    }

    Ok(())
}

fn validate_rate(
    entry_index: usize,
    field: impl Into<String>,
    value: f64,
) -> Result<(), PricingCatalogError> {
    if !value.is_finite() || value < 0.0 {
        return Err(PricingCatalogError::InvalidRate {
            entry_index,
            field: field.into(),
            value,
        });
    }

    Ok(())
}

fn default_pricing_currency() -> String {
    "USD".into()
}

fn normalized_model_name(model: &str) -> String {
    model.trim().to_ascii_lowercase()
}

fn normalized_provider_name(provider: &str) -> String {
    provider.trim().trim_matches('/').to_ascii_lowercase()
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ModelLookupKey {
    provider: Option<String>,
    model: String,
}

fn normalized_model_lookup_keys(provider: Option<&str>, model: &str) -> Vec<ModelLookupKey> {
    let normalized = normalized_model_name(model);
    if normalized.is_empty() {
        return vec![];
    }

    let parts: Vec<&str> = normalized
        .split('/')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect();
    let mut keys = Vec::with_capacity(parts.len() + 3);
    let explicit_provider = provider
        .map(normalized_provider_name)
        .filter(|provider| !provider.is_empty());
    let terminal_model = parts
        .last()
        .copied()
        .unwrap_or(normalized.as_str())
        .to_string();

    if let Some(provider) = explicit_provider {
        push_lookup_key(&mut keys, Some(provider.clone()), normalized.clone());
        push_lookup_key(&mut keys, Some(provider), terminal_model.clone());
    } else if parts.len() > 1 {
        push_lookup_key(
            &mut keys,
            Some(parts[..parts.len() - 1].join("/")),
            terminal_model,
        );
    }

    for start in 0..parts.len() {
        let key = parts[start..].join("/");
        push_lookup_key(&mut keys, None, key);
    }
    keys
}

fn push_lookup_key(keys: &mut Vec<ModelLookupKey>, provider: Option<String>, model: String) {
    let key = ModelLookupKey { provider, model };
    if !key.model.is_empty() && !keys.contains(&key) {
        keys.push(key);
    }
}

/// Infers a provider/route value for a decoded model.
#[must_use]
pub fn infer_model_provider(default_provider: &str, model: Option<&str>) -> Option<String> {
    let normalized_default = normalized_provider_name(default_provider);
    if let Some(model) = model {
        let normalized = normalized_model_name(model);
        let parts: Vec<&str> = normalized
            .split('/')
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .collect();
        if parts.len() > 1 {
            return Some(parts[..parts.len() - 1].join("/"));
        }
    }

    (!normalized_default.is_empty()).then_some(normalized_default)
}

fn cost_component(tokens: u64, price_per_million: f64) -> f64 {
    tokens as f64 * price_per_million / 1_000_000.0
}

fn cost_component_if_nonzero(tokens: u64, price_per_million: f64) -> Option<f64> {
    (tokens > 0).then(|| round_cost_amount(cost_component(tokens, price_per_million)))
}

fn round_cost_amount(cost: f64) -> f64 {
    const SCALE: f64 = 1_000_000_000_000.0;
    (cost * SCALE).round() / SCALE
}
