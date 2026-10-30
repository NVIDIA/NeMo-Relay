// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Default composition, initialization order, and reserved plugin ownership.

use nemo_relay_builtins::plugin;
use std::sync::Arc;

#[test]
fn default_composition_extends_an_already_initialized_host() {
    nemo_relay_plugin_host::initialize_runtime();
    nemo_relay_plugin_host::ensure_builtin_plugins_registered().unwrap();
    assert!(nemo_relay_plugin_host::lookup_plugin("adaptive").is_none());
    // Exporter construction must also select the complete default composition.
    let _exporter = nemo_relay_builtins::exporter::atif(
        "composition".into(),
        nemo_relay_observability::atif::AtifAgentInfo {
            name: "composition".into(),
            version: "1".into(),
            model_name: None,
            tool_definitions: None,
            extra: None,
        },
    );
    plugin::ensure_builtin_plugins_registered().unwrap();
    let kinds = plugin::list_plugin_kinds();
    assert_eq!(kinds.len(), 5);
    for kind in [
        "observability",
        "pricing",
        "resource_metrics",
        "adaptive",
        "pii_redaction",
    ] {
        let implementation = plugin::lookup_plugin(kind).expect(kind);
        assert!(
            plugin::register_plugin(implementation).is_err(),
            "{kind} is reserved"
        );
    }
    // Repeated default installation remains idempotent.
    plugin::ensure_builtin_plugins_registered().unwrap();
    assert_eq!(plugin::list_plugin_kinds(), kinds);
}

#[test]
fn default_composition_reports_an_external_preclaim() {
    nemo_relay_plugin_host::register_plugin(Arc::new(ExternalAdaptive)).unwrap();
    let error = plugin::ensure_builtin_plugins_registered().unwrap_err();
    assert!(error.to_string().contains("reserved builtin"));
    assert!(error.to_string().contains("adaptive"));
}

struct ExternalAdaptive;
impl nemo_relay_plugin::component::Plugin for ExternalAdaptive {
    fn plugin_kind(&self) -> &str {
        "adaptive"
    }
    fn validate(
        &self,
        _: &serde_json::Map<String, nemo_relay_types::Json>,
    ) -> Vec<nemo_relay_plugin::component::ConfigDiagnostic> {
        Vec::new()
    }
    fn register<'a>(
        &'a self,
        _: &serde_json::Map<String, nemo_relay_types::Json>,
        _: &'a mut dyn nemo_relay_plugin::component::PluginRegistrar,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = nemo_relay_plugin::component::Result<()>> + Send + 'a>,
    > {
        Box::pin(async { Ok(()) })
    }
}
