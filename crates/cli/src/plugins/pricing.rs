// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Model pricing catalog CLI helpers.

use std::path::Path;

use nemo_relay::codec::model_pricing::{
    ModelPricing, PricingCatalog, PricingConfig, PricingSourceConfig,
};
use nemo_relay::codec::response::Usage;
use nemo_relay::plugin::{PluginComponentSpec, PluginConfig};
use serde_json::Value;

use crate::configuration::{
    PricingAddSourceCommand, PricingInitCommand, PricingResolveCommand, PricingScopeArgs,
    PricingValidateCommand, ServerArgs, resolve_server_config,
};
use crate::error::CliError;
use crate::plugins::config_io::{PluginConfigDocument, TargetScope, target_path, validate_config};

const PRICING_PLUGIN_KIND: &str = "pricing";

pub(crate) fn validate(command: PricingValidateCommand) -> Result<(), CliError> {
    let catalog = read_pricing_catalog(&command.path)?;
    let entries = catalog.entries.len();
    println!(
        "Valid model pricing catalog: {} ({entries} {})",
        command.path.display(),
        plural(entries, "entry", "entries")
    );
    Ok(())
}

pub(crate) fn init(command: PricingInitCommand) -> Result<(), CliError> {
    let scope = target_pricing_scope(&command.scope)?;
    let path = target_path(scope)?;
    update_plugin_config_document(&path, |plugin_config| {
        let index = ensure_pricing_component(plugin_config)?;
        let pricing_config = pricing_config_from_component(&plugin_config.components[index])?;
        store_pricing_config(&mut plugin_config.components[index], &pricing_config)?;
        plugin_config.components[index].enabled = true;
        Ok(())
    })?;
    println!("Initialized model pricing config: {}", path.display());
    Ok(())
}

pub(crate) fn add_source(command: PricingAddSourceCommand) -> Result<(), CliError> {
    let source_path = std::fs::canonicalize(&command.path).map_err(|source| {
        CliError::Config(format!(
            "could not canonicalize model pricing catalog '{}': {source}",
            command.path.display()
        ))
    })?;
    read_pricing_catalog(&source_path)?;
    let scope = target_pricing_scope(&command.scope)?;
    let path = target_path(scope)?;
    let source = PricingSourceConfig::File { path: source_path };

    update_plugin_config_document(&path, |plugin_config| {
        let index = ensure_pricing_component(plugin_config)?;
        let mut pricing_config = pricing_config_from_component(&plugin_config.components[index])?;
        if !pricing_config.sources.contains(&source) {
            if command.append {
                pricing_config.sources.push(source);
            } else {
                pricing_config.sources.insert(0, source);
            }
        }
        store_pricing_config(&mut plugin_config.components[index], &pricing_config)?;
        plugin_config.components[index].enabled = true;
        Ok(())
    })?;
    println!(
        "Added model pricing source: {} -> {}",
        command.path.display(),
        path.display()
    );
    Ok(())
}

fn update_plugin_config_document(
    path: &Path,
    update: impl FnOnce(&mut PluginConfig) -> Result<(), CliError>,
) -> Result<(), CliError> {
    let mut document = PluginConfigDocument::read(path)?;
    update(document.config_mut())?;
    validate_config(document.config())?;
    document.write()
}

pub(crate) fn resolve(command: PricingResolveCommand) -> Result<(), CliError> {
    let sources = pricing_catalog_sources_from_current_config()?;
    if sources.is_empty() {
        return Err(CliError::Config(
            "no model pricing sources configured; run `nemo-relay model-pricing add-source <catalog.json>` or enable the `pricing` component".into(),
        ));
    }
    let resolved = resolve_pricing(&sources, command.provider.as_deref(), &command.model)
        .ok_or_else(|| {
            CliError::Config(format!(
                "no model pricing entry matched provider={} model={}",
                command.provider.as_deref().unwrap_or("<none>"),
                command.model
            ))
        })?;
    let pricing = resolved.pricing;

    println!("Resolved model pricing");
    println!("source = {}", resolved.source);
    println!("provider = {}", pricing.provider);
    println!("model = {}", pricing.model_id);
    println!("pricing_as_of = {}", pricing.pricing_as_of);
    println!("pricing_source = {}", pricing.pricing_source);

    let usage = Usage {
        prompt_tokens: command.prompt_tokens,
        completion_tokens: command.completion_tokens,
        total_tokens: None,
        cache_read_tokens: command.cache_read_tokens,
        cache_write_tokens: command.cache_write_tokens,
        cost: None,
    };
    if usage_has_tokens(&usage) {
        if let Some(cost) = pricing.estimate_cost(&usage) {
            if let Some(total) = cost.total {
                println!("estimated_total = {total}");
                println!("currency = {}", cost.currency);
            } else {
                println!("estimated_total = unavailable");
            }
        } else {
            println!("estimated_total = unavailable");
        }
    }
    Ok(())
}

fn read_pricing_catalog(path: &Path) -> Result<PricingCatalog, CliError> {
    let raw = std::fs::read_to_string(path).map_err(|source| {
        CliError::Config(format!(
            "could not read model pricing catalog '{}': {source}",
            path.display()
        ))
    })?;
    PricingCatalog::from_json_str(&raw).map_err(|error| {
        CliError::Config(format!(
            "invalid model pricing catalog '{}': {error}",
            path.display()
        ))
    })
}

#[derive(Debug, Clone)]
struct PricingCatalogSource {
    label: String,
    catalog: PricingCatalog,
}

#[derive(Debug, Clone)]
struct ResolvedPricing {
    source: String,
    pricing: ModelPricing,
}

fn pricing_catalog_sources_from_current_config() -> Result<Vec<PricingCatalogSource>, CliError> {
    let resolved = resolve_server_config(&ServerArgs::default())?;
    let Some(plugin_config) = resolved.gateway.plugin_config else {
        return Ok(vec![]);
    };
    let config: PluginConfig = serde_json::from_value(plugin_config)
        .map_err(|error| CliError::Config(format!("invalid plugin config: {error}")))?;
    let Some(component) = config
        .components
        .iter()
        .find(|component| component.kind == PRICING_PLUGIN_KIND && component.enabled)
    else {
        return Ok(vec![]);
    };
    let pricing_config = pricing_config_from_component(component)?;
    pricing_catalog_sources_from_config(&pricing_config)
}

fn pricing_catalog_sources_from_config(
    config: &PricingConfig,
) -> Result<Vec<PricingCatalogSource>, CliError> {
    let mut sources = Vec::new();
    for (index, source) in config.sources.iter().enumerate() {
        match source {
            PricingSourceConfig::Inline { catalog } => sources.push(PricingCatalogSource {
                label: format!("inline:{index}"),
                catalog: catalog.clone(),
            }),
            PricingSourceConfig::File { path } => sources.push(PricingCatalogSource {
                label: format!("file:{}", path.display()),
                catalog: read_pricing_catalog(path)?,
            }),
        }
    }
    Ok(sources)
}

fn resolve_pricing(
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

fn target_pricing_scope(scope: &PricingScopeArgs) -> Result<TargetScope, CliError> {
    let selected = [scope.user, scope.project, scope.global]
        .into_iter()
        .filter(|selected| *selected)
        .count();
    if selected > 1 {
        return Err(CliError::Config(
            "choose only one of --user, --project, or --global".into(),
        ));
    }
    if scope.project {
        Ok(TargetScope::Project)
    } else if scope.global {
        Ok(TargetScope::Global)
    } else {
        Ok(TargetScope::User)
    }
}

fn ensure_pricing_component(config: &mut PluginConfig) -> Result<usize, CliError> {
    if let Some(index) = config
        .components
        .iter()
        .position(|component| component.kind == PRICING_PLUGIN_KIND)
    {
        return Ok(index);
    }
    let mut component = PluginComponentSpec::new(PRICING_PLUGIN_KIND);
    store_pricing_config(&mut component, &PricingConfig::default())?;
    config.components.push(component);
    Ok(config.components.len() - 1)
}

fn pricing_config_from_component(
    component: &PluginComponentSpec,
) -> Result<PricingConfig, CliError> {
    serde_json::from_value(Value::Object(component.config.clone()))
        .map_err(|error| CliError::Config(format!("invalid model pricing config: {error}")))
}

fn store_pricing_config(
    component: &mut PluginComponentSpec,
    config: &PricingConfig,
) -> Result<(), CliError> {
    let value = serde_json::to_value(config).map_err(|error| {
        CliError::Config(format!("could not serialize model pricing config: {error}"))
    })?;
    let Value::Object(object) = value else {
        return Err(CliError::Config(
            "could not serialize model pricing config as an object".into(),
        ));
    };
    component.config = object;
    Ok(())
}

fn usage_has_tokens(usage: &Usage) -> bool {
    usage.prompt_tokens.is_some()
        || usage.completion_tokens.is_some()
        || usage.cache_read_tokens.is_some()
        || usage.cache_write_tokens.is_some()
}

fn plural<'a>(count: usize, singular: &'a str, plural: &'a str) -> &'a str {
    if count == 1 { singular } else { plural }
}

#[cfg(test)]
#[path = "../../tests/coverage/commands/model_pricing_tests.rs"]
mod tests;
