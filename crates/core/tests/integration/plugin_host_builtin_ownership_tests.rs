// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Isolated regression coverage for builtin plugin ownership.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use nemo_relay::plugin::dynamic::{DynamicPluginActivationSpec, PluginHostActivation};
use nemo_relay::plugin::{
    ConfigDiagnostic, Plugin, PluginConfig, PluginRegistrationContext, Result, deregister_plugin,
    register_plugin,
};
use serde_json::{Map, Value as Json};

struct PreclaimedObservabilityPlugin;

impl Plugin for PreclaimedObservabilityPlugin {
    fn plugin_kind(&self) -> &str {
        "observability"
    }

    fn validate(&self, _plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        Vec::new()
    }

    fn register<'a>(
        &'a self,
        _plugin_config: &Map<String, Json>,
        _ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async { Ok(()) })
    }
}

#[tokio::test]
async fn host_rejects_a_builtin_kind_preclaimed_before_first_ensure() {
    register_plugin(Arc::new(PreclaimedObservabilityPlugin))
        .expect("the fixture must preclaim the builtin kind before first ensure");

    let error = match PluginHostActivation::activate(
        PluginConfig::default(),
        Vec::<DynamicPluginActivationSpec>::new(),
    )
    .await
    {
        Ok((activation, _)) => {
            activation
                .clear()
                .expect("unexpected host activation should clear");
            panic!("a preclaimed builtin kind must prevent host activation");
        }
        Err(error) => error.to_string(),
    };

    assert!(
        error.contains("reserved builtin plugin 'observability'"),
        "{error}"
    );
    assert!(error.contains("already registered"), "{error}");
    assert!(deregister_plugin("observability"));

    let (activation, _) = PluginHostActivation::activate(
        PluginConfig::default(),
        Vec::<DynamicPluginActivationSpec>::new(),
    )
    .await
    .expect("host activation should recover after the conflicting registration is removed");
    activation
        .clear()
        .expect("recovered host activation should clear");
}
