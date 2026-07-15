// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Built-in model pricing plugin component module.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::{Map, Value as Json};

use crate::codec::response::{
    PricingConfig, PricingResolver, reset_active_pricing_resolver, set_active_pricing_resolver,
};
use crate::plugin::{
    ConfigDiagnostic, DiagnosticLevel, Plugin, PluginError, PluginRegistration,
    PluginRegistrationContext, Result, register_builtin_plugin,
};

/// Plugin kind used by the model pricing component.
pub const PRICING_PLUGIN_KIND: &str = "pricing";

/// Registers the built-in model pricing component.
pub fn register_pricing_component() -> Result<()> {
    register_builtin_plugin(Arc::new(PricingPlugin))
}

struct PricingPlugin;

impl Plugin for PricingPlugin {
    fn plugin_kind(&self) -> &str {
        PRICING_PLUGIN_KIND
    }

    fn allows_multiple_components(&self) -> bool {
        false
    }

    fn validate(&self, plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
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

    fn register<'a>(
        &'a self,
        plugin_config: &Map<String, Json>,
        ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        let plugin_config = plugin_config.clone();
        Box::pin(async move {
            let config: PricingConfig = serde_json::from_value(Json::Object(plugin_config))?;
            let resolver = PricingResolver::from_config(&config)
                .map_err(|error| PluginError::InvalidConfig(error.to_string()))?;
            set_active_pricing_resolver(resolver)
                .map_err(|error| PluginError::RegistrationFailed(error.to_string()))?;
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
