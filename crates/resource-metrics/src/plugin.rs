// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Built-in resource metrics plugin component.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, LazyLock, Mutex};

use serde_json::{Map, Value as Json};

use crate::config::ResourceMetricsConfig;
use crate::manager::ResourceMetricsRuntime;
use nemo_relay_plugin::component::{
    ConfigDiagnostic, DiagnosticLevel, Plugin, PluginError, PluginRegistration,
    PluginRegistrationContext, Result,
};

/// Plugin kind used by the resource metrics component.
pub const RESOURCE_METRICS_PLUGIN_KIND: &str = "resource_metrics";

static ACTIVE_RUNTIME: LazyLock<Mutex<Option<ResourceMetricsRuntime>>> =
    LazyLock::new(|| Mutex::new(None));

/// Built-in resource metrics component.
pub struct ResourceMetricsPlugin {
    snapshot_sink: Arc<dyn Fn() -> Result<crate::manager::SnapshotSink> + Send + Sync>,
}

impl Plugin for ResourceMetricsPlugin {
    fn plugin_kind(&self) -> &str {
        ResourceMetricsPlugin::plugin_kind(self)
    }

    fn allows_multiple_components(&self) -> bool {
        ResourceMetricsPlugin::allows_multiple_components(self)
    }

    fn validate(&self, config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        ResourceMetricsPlugin::validate(self, config)
    }

    fn register<'a>(
        &'a self,
        config: &Map<String, Json>,
        ctx: &'a mut dyn PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        let config = config.clone();
        Box::pin(async move {
            let config: ResourceMetricsConfig = serde_json::from_value(Json::Object(config))?;
            config
                .validate()
                .map_err(|error| PluginError::InvalidConfig(error.to_string()))?;
            let emit = (self.snapshot_sink)()?;
            let runtime = ResourceMetricsRuntime::configure_current_process(config, emit)
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
                    if crate::manager::runtime_inherited_across_fork() {
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

impl ResourceMetricsPlugin {
    /// Create a component whose snapshot sink is bound by the activating runtime.
    pub fn new(
        snapshot_sink: impl Fn() -> Result<crate::manager::SnapshotSink> + Send + Sync + 'static,
    ) -> Self {
        Self {
            snapshot_sink: Arc::new(snapshot_sink),
        }
    }

    /// Component behavior independent of a concrete runtime host.
    pub fn plugin_kind(&self) -> &str {
        RESOURCE_METRICS_PLUGIN_KIND
    }
    /// Component behavior independent of a concrete runtime host.
    pub fn allows_multiple_components(&self) -> bool {
        false
    }
    /// Component behavior independent of a concrete runtime host.
    pub fn validate(&self, config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
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
}

#[cfg(test)]
#[path = "../tests/unit/resource_metrics/plugin_tests.rs"]
mod tests;
