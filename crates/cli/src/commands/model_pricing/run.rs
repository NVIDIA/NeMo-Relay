// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CLI configuration-file access and presentation for model-pricing commands.

use std::path::Path;

use nemo_relay_model_pricing::catalog::{
    PricingCatalogSource, pricing_catalog_sources_from_config, resolve_pricing,
};
use nemo_relay_model_pricing::plugin::{
    PRICING_PLUGIN_KIND, ensure_pricing_component, pricing_config_from_component,
    store_pricing_config,
};
use nemo_relay_model_pricing::{PricingCatalog, PricingSourceConfig};
use nemo_relay_runtime::plugin::PluginConfig;
use nemo_relay_types::codec::response::Usage;

use crate::configuration::resolve_server_config;
use crate::server::GatewayOverrides;

use super::{
    PricingAddSourceCommand, PricingInitCommand, PricingResolveCommand, PricingValidateCommand,
};
use crate::error::CliError;
use crate::plugins::ConfigurationScope;
use crate::plugins::config_io::{PluginConfigDocument, TargetScope, target_path, validate_config};

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
    let scope = target_pricing_scope(&command.scope.clone().into())?;
    let path = target_path(scope)?;
    update_plugin_config_document(&path, |plugin_config| {
        let index = ensure_pricing_component(&mut plugin_config.components)
            .map_err(pricing_config_error)?;
        let pricing_config = pricing_config_from_component(&plugin_config.components[index])
            .map_err(pricing_config_error)?;
        store_pricing_config(&mut plugin_config.components[index], &pricing_config)
            .map_err(pricing_config_error)?;
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
    let scope = target_pricing_scope(&command.scope.clone().into())?;
    let path = target_path(scope)?;
    let source = PricingSourceConfig::File { path: source_path };

    update_plugin_config_document(&path, |plugin_config| {
        let index = ensure_pricing_component(&mut plugin_config.components)
            .map_err(pricing_config_error)?;
        let mut pricing_config = pricing_config_from_component(&plugin_config.components[index])
            .map_err(pricing_config_error)?;
        if !pricing_config.sources.contains(&source) {
            if command.append {
                pricing_config.sources.push(source);
            } else {
                pricing_config.sources.insert(0, source);
            }
        }
        store_pricing_config(&mut plugin_config.components[index], &pricing_config)
            .map_err(pricing_config_error)?;
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
        uncached_input_tokens: None,
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
    let bytes =
        crate::filesystem::bounded::read_bounded_regular_file(path, "model pricing catalog")
            .map_err(|source| {
                CliError::Config(format!(
                    "could not read model pricing catalog '{}': {source}",
                    path.display()
                ))
            })?;
    let raw = String::from_utf8(bytes).map_err(|source| {
        CliError::Config(format!(
            "model pricing catalog '{}' is not valid UTF-8: {source}",
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

fn pricing_catalog_sources_from_current_config() -> Result<Vec<PricingCatalogSource>, CliError> {
    let resolved = resolve_server_config(&GatewayOverrides::default())?;
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
    let pricing_config = pricing_config_from_component(component).map_err(pricing_config_error)?;
    pricing_catalog_sources_from_config(&pricing_config, read_pricing_catalog)
}

fn target_pricing_scope(scope: &ConfigurationScope) -> Result<TargetScope, CliError> {
    match scope {
        ConfigurationScope::Default | ConfigurationScope::User => Ok(TargetScope::User),
        ConfigurationScope::Global => Ok(TargetScope::Global),
        ConfigurationScope::Invalid => Err(CliError::Config(
            "choose only one of --user or --global".into(),
        )),
    }
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
#[path = "../../../tests/coverage/commands/model_pricing_tests.rs"]
mod tests;

fn pricing_config_error(error: nemo_relay_runtime::plugin::PluginError) -> CliError {
    CliError::Config(error.to_string())
}
