// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Built-in model pricing plugin component module.

use std::future::Future;
use std::pin::Pin;

use serde_json::Value;
use serde_json::{Map, Value as Json};

use crate::{
    PricingConfig, PricingResolver, reset_active_pricing_resolver, set_active_pricing_resolver,
};
use nemo_relay_plugin::component::{
    ConfigDiagnostic, DiagnosticLevel, Plugin, PluginComponentSpec, PluginError, PluginRegistrar,
    PluginRegistration, Result,
};

/// Plugin kind used by the model pricing component.
pub const PRICING_PLUGIN_KIND: &str = "pricing";

/// Built-in model pricing component.
pub struct PricingPlugin;

impl Plugin for PricingPlugin {
    fn plugin_kind(&self) -> &str {
        PricingPlugin::plugin_kind(self)
    }

    fn allows_multiple_components(&self) -> bool {
        PricingPlugin::allows_multiple_components(self)
    }

    fn validate(&self, plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        PricingPlugin::validate(self, plugin_config)
    }

    fn register<'a>(
        &'a self,
        plugin_config: &Map<String, Json>,
        ctx: &'a mut dyn PluginRegistrar,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        let plugin_config = plugin_config.clone();
        Box::pin(async move {
            let config: PricingConfig = serde_json::from_value(Json::Object(plugin_config))?;
            let resolver = PricingResolver::from_config(&config)
                .map_err(|error| PluginError::InvalidConfig(error.to_string()))?;
            set_active_pricing_resolver(resolver)
                .map_err(|error| PluginError::RegistrationFailed(error.to_string()))?;
            log::info!(
                target: "nemo_relay.plugin",
                event = "plugin_resource_validation_completed",
                plugin_kind = PRICING_PLUGIN_KIND,
                resource_count = 0;
                "Plugin resource validation completed"
            );
            ctx.add_registration(PluginRegistration::new(
                "plugin",
                ctx.qualify_name("pricing"),
                Box::new(|| {
                    reset_active_pricing_resolver()
                        .map_err(|error| PluginError::RegistrationFailed(error.to_string()))
                }),
            ));
            Ok(())
        })
    }
}

impl PricingPlugin {
    /// Component behavior independent of a concrete runtime host.
    pub fn plugin_kind(&self) -> &str {
        PRICING_PLUGIN_KIND
    }
    /// Component behavior independent of a concrete runtime host.
    pub fn allows_multiple_components(&self) -> bool {
        false
    }
    /// Component behavior independent of a concrete runtime host.
    pub fn validate(&self, plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        let config =
            match serde_json::from_value::<PricingConfig>(Json::Object(plugin_config.clone())) {
                Ok(config) => config,
                Err(error) => {
                    return vec![ConfigDiagnostic {
                        level: DiagnosticLevel::Error,
                        code: "pricing.invalid_config".into(),
                        component: Some(PRICING_PLUGIN_KIND.into()),
                        field: None,
                        message: format!("invalid model pricing config: {error}"),
                    }];
                }
            };
        match PricingResolver::from_config(&config) {
            Ok(_) => vec![],
            Err(error) => vec![ConfigDiagnostic {
                level: DiagnosticLevel::Error,
                code: "pricing.invalid_config".into(),
                component: Some(PRICING_PLUGIN_KIND.into()),
                field: None,
                message: format!("invalid model pricing config: {error}"),
            }],
        }
    }
}

/// Ensure a pricing component exists in the configured component list.
pub fn ensure_pricing_component(components: &mut Vec<PluginComponentSpec>) -> Result<usize> {
    if let Some(index) = components
        .iter()
        .position(|component| component.kind == PRICING_PLUGIN_KIND)
    {
        return Ok(index);
    }
    let mut component = PluginComponentSpec::new(PRICING_PLUGIN_KIND);
    store_pricing_config(&mut component, &PricingConfig::default())?;
    components.push(component);
    Ok(components.len() - 1)
}

/// Parse one pricing component's configuration.
pub fn pricing_config_from_component(component: &PluginComponentSpec) -> Result<PricingConfig> {
    serde_json::from_value(Value::Object(component.config.clone())).map_err(|error| {
        PluginError::InvalidConfig(format!("invalid model pricing config: {error}"))
    })
}

/// Store typed pricing configuration on a component.
pub fn store_pricing_config(
    component: &mut PluginComponentSpec,
    config: &PricingConfig,
) -> Result<()> {
    let value = serde_json::to_value(config).map_err(|error| {
        PluginError::InvalidConfig(format!("could not serialize model pricing config: {error}"))
    })?;
    let Value::Object(object) = value else {
        return Err(PluginError::InvalidConfig(
            "could not serialize model pricing config as an object".into(),
        ));
    };
    component.config = object;
    Ok(())
}

#[cfg(test)]
#[path = "../tests/unit/component_config_tests.rs"]
mod component_config_tests;
