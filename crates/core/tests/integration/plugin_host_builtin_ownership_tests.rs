// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Isolated regression coverage for builtin plugin ownership.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use nemo_relay::plugin::dynamic::{
    DynamicPluginActivationSpec, DynamicPluginKind, PluginHostActivation,
};
use nemo_relay::plugin::{
    ConfigDiagnostic, DiagnosticLevel, Plugin, PluginComponentSpec, PluginConfig,
    PluginRegistrationContext, Result, deregister_plugin, list_plugin_kinds, lookup_plugin,
    register_plugin, validate_plugin_config,
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

    let config = PluginConfig {
        components: vec![PluginComponentSpec::new("observability")],
        ..PluginConfig::default()
    };
    let report = validate_plugin_config(&config);
    let diagnostic = report
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code == "plugin.builtin_registration_failed")
        .expect("validation must surface the builtin ownership conflict");
    assert_eq!(diagnostic.level, DiagnosticLevel::Error);
    assert!(
        diagnostic
            .message
            .contains("reserved builtin plugin 'observability'"),
        "{}",
        diagnostic.message
    );
    assert!(lookup_plugin("observability").is_none());
    assert!(list_plugin_kinds().is_empty());

    let missing_dynamic_plugin = DynamicPluginActivationSpec {
        plugin_id: "fixture_missing".into(),
        kind: DynamicPluginKind::RustDynamic,
        manifest_ref: "missing-relay-plugin.toml".into(),
        environment_ref: None,
        config: Map::new(),
    };
    let error = match PluginHostActivation::activate(
        PluginConfig::default(),
        [missing_dynamic_plugin.clone()],
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

    let report = validate_plugin_config(&config);
    assert!(!report.has_errors(), "{:#?}", report.diagnostics);
    assert!(lookup_plugin("observability").is_some());
    assert!(
        list_plugin_kinds()
            .iter()
            .any(|kind| kind == "observability")
    );

    let error = PluginHostActivation::activate(PluginConfig::default(), [missing_dynamic_plugin])
        .await
        .err()
        .expect("the missing fixture manifest should fail after builtin registration recovers")
        .to_string();
    assert!(error.contains("missing-relay-plugin.toml"), "{error}");
    assert!(!error.contains("active dynamic plugin host"), "{error}");
}
