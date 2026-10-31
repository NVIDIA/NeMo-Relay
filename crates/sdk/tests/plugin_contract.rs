// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Applications can author, configure, activate, and clean up plugins through the SDK alone.

use nemo_relay::Json;
use nemo_relay::plugin::{
    self, ConfigDiagnostic, ConfigPolicy, DiagnosticLevel, Plugin, PluginComponentSpec,
    PluginConfig, PluginRegistration, PluginRegistrationContext, Result as PluginResult,
    UnsupportedBehavior,
};
use nemo_relay::{scope, subscriber};
use serde_json::{Map, json};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct AuditPlugin {
    observed: Arc<AtomicUsize>,
    cleaned: Arc<AtomicBool>,
}

impl Plugin for AuditPlugin {
    fn plugin_kind(&self) -> &str {
        "sdk-contract.audit"
    }

    fn validate(&self, config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        if config.get("tag").and_then(Json::as_str) == Some("audit") {
            Vec::new()
        } else {
            vec![ConfigDiagnostic {
                level: DiagnosticLevel::Error,
                code: "audit.invalid_tag".into(),
                component: Some(self.plugin_kind().into()),
                field: Some("tag".into()),
                message: "tag must equal audit".into(),
            }]
        }
    }

    fn register<'a>(
        &'a self,
        _: &Map<String, Json>,
        context: &'a mut dyn PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = PluginResult<()>> + Send + 'a>> {
        let observed = self.observed.clone();
        let cleaned = self.cleaned.clone();
        Box::pin(async move {
            context.register_subscriber(
                "events",
                Arc::new(move |_| {
                    observed.fetch_add(1, Ordering::SeqCst);
                }),
            )?;
            context.add_registration(PluginRegistration::new(
                "custom",
                context.qualify_name("resource"),
                Box::new(move || {
                    cleaned.store(true, Ordering::SeqCst);
                    Ok(())
                }),
            ));
            Ok(())
        })
    }
}

fn emit_scope() {
    let handle = scope::push_scope(
        scope::PushScopeParams::builder()
            .name("sdk-plugin-contract")
            .scope_type(scope::ScopeType::Agent)
            .build(),
    )
    .unwrap();
    scope::pop_scope(
        scope::PopScopeParams::builder()
            .handle_uuid(&handle.uuid)
            .build(),
    )
    .unwrap();
    subscriber::flush_subscribers().unwrap();
}

#[tokio::test]
async fn sdk_plugin_contract_covers_policy_activation_delivery_and_cleanup() {
    let policy = ConfigPolicy {
        unknown_component: UnsupportedBehavior::Error,
        ..Default::default()
    };
    let missing = plugin::validate_static_plugin_config(&PluginConfig {
        components: vec![PluginComponentSpec::new("sdk-contract.missing")],
        policy,
        ..Default::default()
    });
    assert!(missing.has_errors());
    assert!(missing.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "plugin.unknown_component" && diagnostic.level == DiagnosticLevel::Error
    }));

    let observed = Arc::new(AtomicUsize::new(0));
    let cleaned = Arc::new(AtomicBool::new(false));
    plugin::register_plugin(Arc::new(AuditPlugin {
        observed: observed.clone(),
        cleaned: cleaned.clone(),
    }))
    .unwrap();
    let mut component = PluginComponentSpec::new("sdk-contract.audit");
    let invalid = plugin::validate_static_plugin_config(&PluginConfig {
        components: vec![component.clone()],
        policy,
        ..Default::default()
    });
    assert_eq!(invalid.diagnostics[0].code, "audit.invalid_tag");
    component.config.insert("tag".into(), json!("audit"));
    let mut activation = plugin::initialize_exact(PluginConfig {
        components: vec![component],
        policy,
        ..Default::default()
    })
    .await
    .unwrap();
    assert!(!activation.report().config.has_errors());
    emit_scope();
    assert_eq!(observed.load(Ordering::SeqCst), 2);
    assert!(!cleaned.load(Ordering::SeqCst));
    activation.close().unwrap();
    assert!(cleaned.load(Ordering::SeqCst));
    emit_scope();
    assert_eq!(observed.load(Ordering::SeqCst), 2);
    assert!(plugin::deregister_plugin("sdk-contract.audit"));
}
