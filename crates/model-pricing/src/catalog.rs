// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Catalog source loading and resolution with caller-owned file access.

use crate::{ModelPricing, PricingCatalog, PricingConfig, PricingSourceConfig};
use std::path::Path;

/// One catalog and its source identity.
#[derive(Debug, Clone)]
pub struct PricingCatalogSource {
    /// Human-readable source identity.
    pub label: String,
    /// Validated catalog snapshot.
    pub catalog: PricingCatalog,
}

/// A matched entry with its source identity.
#[derive(Debug, Clone)]
pub struct ResolvedPricing {
    /// Matching source identity.
    pub source: String,
    /// Selected model pricing.
    pub pricing: ModelPricing,
}

/// Load configured sources in precedence order using the caller's file access policy.
pub fn pricing_catalog_sources_from_config<E>(
    config: &PricingConfig,
    read_catalog: impl Fn(&Path) -> Result<PricingCatalog, E>,
) -> Result<Vec<PricingCatalogSource>, E> {
    let mut sources = Vec::new();
    for (index, source) in config.sources.iter().enumerate() {
        match source {
            PricingSourceConfig::Inline { catalog } => sources.push(PricingCatalogSource {
                label: format!("inline:{index}"),
                catalog: catalog.clone(),
            }),
            PricingSourceConfig::File { path } => sources.push(PricingCatalogSource {
                label: format!("file:{}", path.display()),
                catalog: read_catalog(path)?,
            }),
        }
    }
    Ok(sources)
}

/// Resolve a provider/model pair against sources in precedence order.
pub fn resolve_pricing(
    sources: &[PricingCatalogSource],
    provider: Option<&str>,
    model: &str,
) -> Option<ResolvedPricing> {
    sources.iter().find_map(|source| {
        source
            .catalog
            .pricing_for(provider, model)
            .map(|pricing| ResolvedPricing {
                source: source.label.clone(),
                pricing,
            })
    })
}
