// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Built-in resource metrics plugin component.

pub(crate) mod config;

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, LazyLock, Mutex};

use serde_json::{Map, Value as Json};

use crate::plugin::{
    ConfigDiagnostic, DiagnosticLevel, Plugin, PluginError, PluginRegistration,
    PluginRegistrationContext, Result, register_builtin_plugin,
};
use crate::resource_metrics::manager::ResourceMetricsRuntime;
use config::ResourceMetricsConfig;

/// Plugin kind used by the resource metrics component.
pub const RESOURCE_METRICS_PLUGIN_KIND: &str = "resource_metrics";

static ACTIVE_RUNTIME: LazyLock<Mutex<Option<ResourceMetricsRuntime>>> =
    LazyLock::new(|| Mutex::new(None));

/// Registers the built-in resource metrics component.
pub fn register_resource_metrics_component() -> Result<()> {
    register_builtin_plugin(Arc::new(ResourceMetricsPlugin))
}

struct ResourceMetricsPlugin;

impl Plugin for ResourceMetricsPlugin {
    fn plugin_kind(&self) -> &str {
        RESOURCE_METRICS_PLUGIN_KIND
    }

    fn allows_multiple_components(&self) -> bool {
        false
    }

    fn validate(&self, config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        let parsed = serde_json::from_value::<ResourceMetricsConfig>(Json::Object(config.clone()));
        let result = parsed
            .map_err(|error| error.to_string())
            .and_then(|config| config.validate().map_err(|error| error.to_string()));
        result
            .err()
            .map(|error| ConfigDiagnostic {
                level: DiagnosticLevel::Error,
                code: "resource_metrics.invalid_config".into(),
                component: Some(RESOURCE_METRICS_PLUGIN_KIND.into()),
                field: None,
                message: format!("invalid resource metrics config: {error}"),
            })
            .into_iter()
            .collect()
    }

    fn register<'a>(
        &'a self,
        config: &Map<String, Json>,
        ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        let config = config.clone();
        Box::pin(async move {
            let config: ResourceMetricsConfig = serde_json::from_value(Json::Object(config))?;
            config
                .validate()
                .map_err(|error| PluginError::InvalidConfig(error.to_string()))?;
            let runtime = ResourceMetricsRuntime::configure_current_process(config)
                .map_err(|error| PluginError::RegistrationFailed(error.to_string()))?;
            let mut active = ACTIVE_RUNTIME
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if active.is_some() {
                return Err(PluginError::Conflict(
                    "resource_metrics plugin is already active".into(),
                ));
            }
            *active = Some(runtime);
            drop(active);
            ctx.add_registration(PluginRegistration::new(
                "plugin",
                ctx.qualify_name("resource_metrics"),
                Box::new(|| {
                    if crate::resource_metrics::manager::runtime_inherited_across_fork() {
                        return Ok(());
                    }
                    ACTIVE_RUNTIME
                        .lock()
                        .unwrap_or_else(|error| error.into_inner())
                        .take();
                    Ok(())
                }),
            ));
            Ok(())
        })
    }
}

#[cfg(test)]
#[path = "../../tests/unit/resource_metrics/plugin_tests.rs"]
mod tests;
