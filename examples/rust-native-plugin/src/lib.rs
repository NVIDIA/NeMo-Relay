// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod config;
mod execution;
mod observe;
mod requests;
mod runtime;

use nemo_relay_plugin::{
    ConfigDiagnostic, DiagnosticLevel, Json, NativeExecutorConfig, NativePlugin, PluginContext,
};
use serde_json::Map;

use config::ExampleConfig;

struct ExampleNativePlugin;

/// Validate the example's component-local configuration without loading a native host.
pub fn validate_example_config(plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
    let plugin = ExampleNativePlugin;
    plugin.validate(plugin_config)
}

impl NativePlugin for ExampleNativePlugin {
    fn plugin_kind(&self) -> &str {
        "examples.rust_native_policy"
    }

    fn executor_config(&self) -> NativeExecutorConfig {
        NativeExecutorConfig { worker_threads: 2 }
    }

    fn allows_multiple_components(&self) -> bool {
        false
    }

    fn validate(&self, plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        let mut diagnostics = config::validate(plugin_config);
        if let Err(error) = self.executor_config_for_component(plugin_config) {
            diagnostics.push(diagnostic(
                DiagnosticLevel::Error,
                "examples.rust_native_policy.invalid_executor",
                Some("executor.worker_threads"),
                error,
            ));
        }
        diagnostics
    }

    fn register(
        &mut self,
        plugin_config: &Map<String, Json>,
        context: &mut PluginContext<'_>,
    ) -> nemo_relay_plugin::Result<()> {
        let config = ExampleConfig::parse(plugin_config)?;
        let plugin_runtime = context.runtime();

        observe::register(context, &config, &plugin_runtime)?;
        requests::register(context, &config)?;
        execution::register(context, &config, &plugin_runtime)?;
        Ok(())
    }
}

pub(crate) fn diagnostic(
    level: DiagnosticLevel,
    code: &str,
    field: Option<&str>,
    message: impl Into<String>,
) -> ConfigDiagnostic {
    ConfigDiagnostic {
        level,
        code: code.into(),
        component: Some("examples.rust_native_policy".into()),
        field: field.map(str::to_owned),
        message: message.into(),
    }
}

nemo_relay_plugin::nemo_relay_plugin!(nemo_relay_register_plugin, || ExampleNativePlugin);
