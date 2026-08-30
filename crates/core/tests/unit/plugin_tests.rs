// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for plugin in the NeMo Relay core crate.

use super::*;
use std::collections::BTreeMap;
#[cfg(feature = "__skip-implicit-config")]
use std::ffi::{OsStr, OsString};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use serde_json::json;
use tokio::sync::Notify;

use crate::api::event::{BaseEvent, Event, MarkEvent};
use crate::api::llm::{LlmRequest, LlmRequestInterceptOutcome};
use crate::api::llm::{llm_conditional_execution, llm_request_intercepts};
use crate::api::registry::{
    RuntimeRegistrationKind, RuntimeRegistrationOwnerKind, list_runtime_registrations,
};
use crate::api::runtime::global_context;
use crate::api::runtime::{LlmJsonStream, NemoRelayContextState};
use crate::api::tool::tool_conditional_execution;
use crate::error::FlowError;

struct TestPlugin;
struct PolicyAwarePlugin;

struct SingletonPlugin;
struct RecordingPlugin;
struct DiscoverablePlugin;
struct ReplacementPlugin;
struct RestoreFailPlugin;
struct RestoreBreakPlugin;
struct PartialFailPlugin;
struct VanishingPlugin;
struct BlockingPlugin {
    started: Arc<Notify>,
    release: Arc<Notify>,
    registered: Arc<Notify>,
}
struct BackgroundTaskPlugin {
    release: Arc<Notify>,
    completed: Arc<Notify>,
}
struct PanickingPlugin;
struct FailingDeregisterPlugin;
struct EventMetadataPlugin;
struct PluginMutationOwnerCleanup;

impl Drop for PluginMutationOwnerCleanup {
    fn drop(&mut self) {
        let mut owner = PLUGIN_MUTATION_OWNER
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        *owner = PluginMutationOwner::Idle;
    }
}

static RECORDED_NAMES: OnceLock<Mutex<Vec<String>>> = OnceLock::new();
static PARTIAL_FAIL_ROLLBACKS: AtomicUsize = AtomicUsize::new(0);
static RESTORE_FAIL_REGISTRATIONS: AtomicUsize = AtomicUsize::new(0);
static RESTORE_BREAK_REGISTRATIONS: AtomicUsize = AtomicUsize::new(0);
static REPLACEMENT_REGISTRATIONS: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "__skip-implicit-config")]
static TEST_CONFIG_ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

#[cfg(feature = "__skip-implicit-config")]
struct TestConfigEnvironment {
    _guard: std::sync::MutexGuard<'static, ()>,
    previous: Option<OsString>,
}

#[cfg(feature = "__skip-implicit-config")]
impl TestConfigEnvironment {
    fn set(value: Option<&OsStr>) -> Self {
        let guard = TEST_CONFIG_ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let previous = std::env::var_os("NEMO_RELAY_TEST_SKIP_IMPLICIT_CONFIG");
        unsafe {
            match value {
                Some(value) => std::env::set_var("NEMO_RELAY_TEST_SKIP_IMPLICIT_CONFIG", value),
                None => std::env::remove_var("NEMO_RELAY_TEST_SKIP_IMPLICIT_CONFIG"),
            }
        }
        Self {
            _guard: guard,
            previous,
        }
    }
}

#[cfg(feature = "__skip-implicit-config")]
impl Drop for TestConfigEnvironment {
    fn drop(&mut self) {
        unsafe {
            match self.previous.take() {
                Some(value) => std::env::set_var("NEMO_RELAY_TEST_SKIP_IMPLICIT_CONFIG", value),
                None => std::env::remove_var("NEMO_RELAY_TEST_SKIP_IMPLICIT_CONFIG"),
            }
        }
    }
}

fn recorded_names() -> &'static Mutex<Vec<String>> {
    RECORDED_NAMES.get_or_init(|| Mutex::new(Vec::new()))
}

fn lock_runtime_owner() -> std::sync::MutexGuard<'static, ()> {
    crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner())
}

fn expect_registration_failed(result: Result<()>, message_fragment: &str) {
    match result {
        Err(PluginError::RegistrationFailed(message)) => {
            assert!(message.contains(message_fragment), "{message}");
        }
        Err(other) => panic!("unexpected registration failure: {other}"),
        Ok(_) => panic!("expected registration to fail"),
    }
}

fn set_conflicting_runtime_owner_for_tests() {
    unsafe {
        std::env::set_var(
            "NEMO_RELAY_RUNTIME_OWNER",
            format!(
                "pid={};binding=python;version={}",
                std::process::id(),
                env!("CARGO_PKG_VERSION")
            ),
        )
    };
}

fn observability_destination_document(destination: &str) -> Json {
    json!({
        "components": [{
            "kind": "observability",
            "config": {
                "atof": {"enabled": true, "sinks": [destination]},
                "opentelemetry": {"enabled": true, "endpoints": [destination]},
                "atif": {"enabled": true, "storage": [destination]}
            }
        }]
    })
}

#[test]
fn plugin_configuration_helpers_preserve_defaults_and_error_detection() {
    let component = PluginComponentSpec::new("coverage.plugin");
    assert_eq!(component.kind, "coverage.plugin");
    assert!(component.enabled);
    assert!(component.config.is_empty());

    let empty = ConfigReport::default();
    assert!(!empty.has_errors());
    let error = ConfigReport {
        diagnostics: vec![ConfigDiagnostic {
            level: DiagnosticLevel::Error,
            code: "coverage.error".into(),
            component: None,
            field: None,
            message: "expected".into(),
        }],
        runtime_diagnostics: vec![],
    };
    assert!(error.has_errors());
    assert!(matches!(
        apply_global_config_policy(
            ConfigPolicy::default(),
            &ConfigPolicy {
                unknown_component: UnsupportedBehavior::Error,
                ..ConfigPolicy::default()
            }
        )
        .unknown_component,
        UnsupportedBehavior::Error
    ));
}

fn programmatic_observability_config(config: Json) -> PluginConfig {
    serde_json::from_value(json!({
        "components": [{"kind": "observability", "config": config}]
    }))
    .unwrap()
}

impl Plugin for TestPlugin {
    fn plugin_kind(&self) -> &str {
        "test.plugin"
    }

    fn validate(&self, _plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        vec![ConfigDiagnostic {
            level: DiagnosticLevel::Warning,
            code: "test.warning".into(),
            component: Some("test.plugin".into()),
            field: None,
            message: "validated".into(),
        }]
    }

    fn register<'a>(
        &'a self,
        _plugin_config: &Map<String, Json>,
        ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            ctx.register_llm_request_intercept(
                "intercept",
                1,
                false,
                Arc::new(|_name, mut request, annotated| {
                    Box::pin(async move {
                        request.headers.insert("x-plugin".into(), json!(true));
                        Ok(LlmRequestInterceptOutcome::new(request, annotated))
                    })
                }),
            )
        })
    }
}

impl Plugin for EventMetadataPlugin {
    fn plugin_kind(&self) -> &str {
        "event-metadata.plugin"
    }

    fn validate(&self, _plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        Vec::new()
    }

    fn register<'a>(
        &'a self,
        plugin_config: &Map<String, Json>,
        ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        let configured = plugin_config
            .get("metadata")
            .and_then(Json::as_object)
            .expect("test plugin metadata must be an object")
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect::<BTreeMap<_, _>>();
        Box::pin(async move {
            ctx.register_event_metadata_injector(
                "configured-metadata",
                10,
                Arc::new(move |_| {
                    let additions = configured.clone();
                    Box::pin(async move { Ok(additions) })
                }),
            )
        })
    }
}

impl Plugin for PolicyAwarePlugin {
    fn plugin_kind(&self) -> &str {
        "policy-aware.plugin"
    }

    fn validate(&self, _plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        vec![]
    }

    fn validate_with_policy(
        &self,
        _plugin_config: &Map<String, Json>,
        policy: &ConfigPolicy,
    ) -> Vec<ConfigDiagnostic> {
        match policy.unsupported_value {
            UnsupportedBehavior::Ignore => vec![],
            UnsupportedBehavior::Warn => vec![ConfigDiagnostic {
                level: DiagnosticLevel::Warning,
                code: "policy-aware.unsupported_value".into(),
                component: Some(self.plugin_kind().into()),
                field: None,
                message: "unsupported value".into(),
            }],
            UnsupportedBehavior::Error => vec![ConfigDiagnostic {
                level: DiagnosticLevel::Error,
                code: "policy-aware.unsupported_value".into(),
                component: Some(self.plugin_kind().into()),
                field: None,
                message: "unsupported value".into(),
            }],
        }
    }

    fn register<'a>(
        &'a self,
        _plugin_config: &Map<String, Json>,
        _ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async { Ok(()) })
    }
}

impl Plugin for SingletonPlugin {
    fn plugin_kind(&self) -> &str {
        "singleton.plugin"
    }

    fn allows_multiple_components(&self) -> bool {
        false
    }

    fn validate(&self, _plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        vec![]
    }

    fn register<'a>(
        &'a self,
        _plugin_config: &Map<String, Json>,
        _ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async { Ok(()) })
    }
}

impl Plugin for RecordingPlugin {
    fn plugin_kind(&self) -> &str {
        "recording.plugin"
    }

    fn validate(&self, _plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        vec![]
    }

    fn register<'a>(
        &'a self,
        _plugin_config: &Map<String, Json>,
        ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        let qualified = ctx.qualify_name("subscriber");
        recorded_names().lock().unwrap().push(qualified.clone());
        Box::pin(async move {
            ctx.add_registration(PluginRegistration::new(
                "plugin",
                qualified,
                Box::new(|| Ok(())),
            ));
            Ok(())
        })
    }
}

impl Plugin for DiscoverablePlugin {
    fn plugin_kind(&self) -> &str {
        "discoverable.plugin"
    }

    fn validate(&self, _plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        vec![]
    }

    fn register<'a>(
        &'a self,
        _plugin_config: &Map<String, Json>,
        ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move { ctx.register_subscriber("subscriber", Arc::new(|_| {})) })
    }
}

impl Plugin for ReplacementPlugin {
    fn plugin_kind(&self) -> &str {
        "replacement.plugin"
    }

    fn validate(&self, _plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        vec![ConfigDiagnostic {
            level: DiagnosticLevel::Warning,
            code: "replacement.warning".into(),
            component: Some("replacement.plugin".into()),
            field: None,
            message: "replacement validated".into(),
        }]
    }

    fn register<'a>(
        &'a self,
        _plugin_config: &Map<String, Json>,
        ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            REPLACEMENT_REGISTRATIONS.fetch_add(1, Ordering::SeqCst);
            ctx.add_registration(PluginRegistration::new(
                "plugin",
                ctx.qualify_name("replacement"),
                Box::new(|| Ok(())),
            ));
            Ok(())
        })
    }
}

impl Plugin for RestoreFailPlugin {
    fn plugin_kind(&self) -> &str {
        "restore.fail.plugin"
    }

    fn validate(&self, _plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        vec![]
    }

    fn register<'a>(
        &'a self,
        _plugin_config: &Map<String, Json>,
        ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            RESTORE_FAIL_REGISTRATIONS.fetch_add(1, Ordering::SeqCst);
            ctx.add_registration(PluginRegistration::new(
                "plugin",
                ctx.qualify_name("restore-fail"),
                Box::new(|| Ok(())),
            ));
            Err(PluginError::RegistrationFailed(
                "restore.fail.plugin refused to initialize".into(),
            ))
        })
    }
}

impl Plugin for RestoreBreakPlugin {
    fn plugin_kind(&self) -> &str {
        "restore.break.plugin"
    }

    fn validate(&self, _plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        vec![]
    }

    fn register<'a>(
        &'a self,
        _plugin_config: &Map<String, Json>,
        ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            if RESTORE_BREAK_REGISTRATIONS.fetch_add(1, Ordering::SeqCst) == 0 {
                ctx.add_registration(PluginRegistration::new(
                    "plugin",
                    ctx.qualify_name("restore-break"),
                    Box::new(|| Ok(())),
                ));
                Ok(())
            } else {
                Err(PluginError::RegistrationFailed(
                    "restore.break.plugin refused to restore".into(),
                ))
            }
        })
    }
}

impl Plugin for PartialFailPlugin {
    fn plugin_kind(&self) -> &str {
        "partial.fail.plugin"
    }

    fn validate(&self, _plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        vec![]
    }

    fn register<'a>(
        &'a self,
        _plugin_config: &Map<String, Json>,
        ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            ctx.add_registration(PluginRegistration::new(
                "plugin",
                ctx.qualify_name("partial-fail"),
                Box::new(|| {
                    PARTIAL_FAIL_ROLLBACKS.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                }),
            ));
            Err(PluginError::RegistrationFailed(
                "partial.fail.plugin refused to finish initialization".into(),
            ))
        })
    }
}

impl Plugin for VanishingPlugin {
    fn plugin_kind(&self) -> &str {
        "vanishing.plugin"
    }

    fn validate(&self, _plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        let _ = deregister_plugin("vanishing.plugin");
        vec![]
    }

    fn register<'a>(
        &'a self,
        _plugin_config: &Map<String, Json>,
        _ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async { Ok(()) })
    }
}

impl Plugin for BlockingPlugin {
    fn plugin_kind(&self) -> &str {
        "blocking.plugin"
    }

    fn validate(&self, _plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        vec![ConfigDiagnostic {
            level: DiagnosticLevel::Warning,
            code: "blocking.warning".into(),
            component: Some("blocking.plugin".into()),
            field: None,
            message: "blocking plugin validated".into(),
        }]
    }

    fn register<'a>(
        &'a self,
        _plugin_config: &Map<String, Json>,
        ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        let started = Arc::clone(&self.started);
        let release = Arc::clone(&self.release);
        let registered = Arc::clone(&self.registered);
        Box::pin(async move {
            started.notify_one();
            release.notified().await;
            ctx.add_registration(PluginRegistration::new(
                "plugin",
                ctx.qualify_name("blocking"),
                Box::new(|| Ok(())),
            ));
            registered.notify_one();
            Ok(())
        })
    }
}

impl Plugin for BackgroundTaskPlugin {
    fn plugin_kind(&self) -> &str {
        "background.task.plugin"
    }

    fn validate(&self, _plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        vec![]
    }

    fn register<'a>(
        &'a self,
        _plugin_config: &Map<String, Json>,
        _ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        let release = Arc::clone(&self.release);
        let completed = Arc::clone(&self.completed);
        Box::pin(async move {
            tokio::spawn(async move {
                release.notified().await;
                completed.notify_one();
            });
            Ok(())
        })
    }
}

impl Plugin for PanickingPlugin {
    fn plugin_kind(&self) -> &str {
        "panicking.plugin"
    }

    fn validate(&self, _plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        vec![]
    }

    fn register<'a>(
        &'a self,
        _plugin_config: &Map<String, Json>,
        _ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async { panic!("fixture plugin panicked during registration") })
    }
}

impl Plugin for FailingDeregisterPlugin {
    fn plugin_kind(&self) -> &str {
        "failing.deregister.plugin"
    }

    fn validate(&self, _plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        vec![]
    }

    fn register<'a>(
        &'a self,
        _plugin_config: &Map<String, Json>,
        ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            ctx.add_registration(PluginRegistration::new(
                "fixture",
                ctx.qualify_name("refuses-deregistration"),
                Box::new(|| {
                    Err(PluginError::RegistrationFailed(
                        "fixture deregistration refused".into(),
                    ))
                }),
            ));
            Ok(())
        })
    }
}

fn reset_global() {
    let _ = spdlog::init_log_crate_proxy();
    log::set_max_level(log::LevelFilter::Info);
    crate::shared_runtime::reset_runtime_owner_for_tests();
    let ctx = global_context();
    let mut state = ctx.write().unwrap();
    *state = NemoRelayContextState::new();
    clear_plugin_configuration().unwrap();
    recorded_names().lock().unwrap().clear();
    PARTIAL_FAIL_ROLLBACKS.store(0, Ordering::SeqCst);
    RESTORE_FAIL_REGISTRATIONS.store(0, Ordering::SeqCst);
    RESTORE_BREAK_REGISTRATIONS.store(0, Ordering::SeqCst);
    REPLACEMENT_REGISTRATIONS.store(0, Ordering::SeqCst);
    let _ = deregister_plugin("test.plugin");
    let _ = deregister_plugin("singleton.plugin");
    let _ = deregister_plugin("recording.plugin");
    let _ = deregister_plugin("replacement.plugin");
    let _ = deregister_plugin("restore.fail.plugin");
    let _ = deregister_plugin("restore.break.plugin");
    let _ = deregister_plugin("partial.fail.plugin");
    let _ = deregister_plugin("vanishing.plugin");
    let _ = deregister_plugin("blocking.plugin");
    let _ = deregister_plugin("background.task.plugin");
    let _ = deregister_plugin("panicking.plugin");
    let _ = deregister_plugin("failing.deregister.plugin");
    let _ = deregister_plugin("event-metadata.plugin");
}

#[test]
fn test_layer_config_overlay_wins() {
    // The overlay is the higher-precedence layer: it overrides shared component fields, deep-merges
    // nested config objects, concatenates config lists, appends overlay-only kinds, preserves
    // base-only kinds, replaces top-level scalars, and recursively merges top-level objects
    // (policy).
    let base = json!({
        "version": 1,
        "components": [
            {
                "kind": "alpha",
                "enabled": true,
                "config": { "keep": "base", "override": "base", "nested": {"a": 1, "b": 2}, "list": [1, 2, 3] }
            },
            { "kind": "base_only", "enabled": true, "config": {} }
        ],
        "policy": { "unknown_component": "warn", "unknown_field": "warn" }
    });
    let overlay = json!({
        "version": 2,
        "components": [
            {
                "kind": "alpha",
                "enabled": false,
                "config": { "override": "overlay", "added": true, "nested": {"b": 20, "c": 30}, "list": [9] }
            },
            { "kind": "overlay_only", "enabled": true, "config": {} }
        ],
        "policy": { "unknown_component": "error" }
    });

    let mut merged = base;
    layer_config(&mut merged, overlay);
    let components = merged["components"].as_array().unwrap();

    // Ordering: base components first (in base order), then overlay-only components appended.
    let kinds: Vec<&str> = components
        .iter()
        .map(|component| component["kind"].as_str().unwrap())
        .collect();
    assert_eq!(kinds, ["alpha", "base_only", "overlay_only"]);

    let alpha = &components[0];
    assert_eq!(alpha["enabled"], json!(false), "overlay enabled wins");
    assert_eq!(
        alpha["config"]["keep"],
        json!("base"),
        "base-only key preserved"
    );
    assert_eq!(
        alpha["config"]["override"],
        json!("overlay"),
        "overlay scalar wins"
    );
    assert_eq!(alpha["config"]["added"], json!(true), "overlay key added");
    assert_eq!(
        alpha["config"]["nested"],
        json!({"a": 1, "b": 20, "c": 30}),
        "nested objects merge recursively"
    );
    assert_eq!(
        alpha["config"]["list"],
        json!([9, 1, 2, 3]),
        "config lists concatenate with higher-precedence entries first"
    );

    // Base-only component is preserved.
    assert_eq!(components[1]["kind"], json!("base_only"));

    // Top-level scalars are replaced by the overlay; objects (policy) merge recursively.
    assert_eq!(merged["version"], json!(2));
    assert_eq!(merged["policy"]["unknown_component"], json!("error"));
    assert_eq!(
        merged["policy"]["unknown_field"],
        json!("warn"),
        "base-only policy field preserved"
    );
}

#[test]
fn test_layer_config_concatenates_nested_observability_lists() {
    let base = json!({
        "components": [{
            "kind": "observability",
            "config": {
                "atof": {
                    "sinks": [{"type": "file", "output_directory": "/system"}]
                },
                "opentelemetry": {
                    "endpoints": [{"type": "full", "endpoint": "http://system:4318/v1/traces"}],
                    "logs": {
                        "endpoints": [{"endpoint": "http://system:4318/v1/logs"}]
                    },
                    "metrics": {
                        "endpoints": [{"endpoint": "http://system:4318/v1/metrics"}]
                    }
                },
                "atif": {
                    "storage": [{"type": "http", "endpoint": "http://system/trajectory"}]
                },
                "implementation": {
                    "nested": [1, 2]
                }
            }
        }]
    });
    let overlay = json!({
        "components": [{
            "kind": "observability",
            "config": {
                "atof": {
                    "sinks": [{"type": "file", "output_directory": "/user"}]
                },
                "opentelemetry": {
                    "endpoints": [
                        {"type": "gen_ai", "endpoint": "http://user:4318/v1/traces"},
                        {"type": "openinference", "endpoint": "http://user:4319/v1/traces"}
                    ],
                    "logs": {
                        "endpoints": [{"endpoint": "http://user:4318/v1/logs"}]
                    },
                    "metrics": {
                        "endpoints": [{"endpoint": "http://user:4318/v1/metrics"}]
                    }
                },
                "atif": {
                    "storage": [{"type": "http", "endpoint": "http://user/trajectory"}]
                },
                "implementation": {
                    "nested": [9]
                }
            }
        }]
    });

    let mut merged = base;
    layer_config(&mut merged, overlay);
    let config = &merged["components"][0]["config"];

    assert_eq!(
        config["atof"]["sinks"],
        json!([
            {"type": "file", "output_directory": "/user"},
            {"type": "file", "output_directory": "/system"}
        ])
    );
    assert_eq!(
        config["opentelemetry"]["endpoints"],
        json!([
            {"type": "gen_ai", "endpoint": "http://user:4318/v1/traces"},
            {"type": "openinference", "endpoint": "http://user:4319/v1/traces"},
            {"type": "full", "endpoint": "http://system:4318/v1/traces"}
        ])
    );
    assert_eq!(
        config["opentelemetry"]["logs"]["endpoints"],
        json!([
            {"endpoint": "http://user:4318/v1/logs"},
            {"endpoint": "http://system:4318/v1/logs"}
        ])
    );
    assert_eq!(
        config["opentelemetry"]["metrics"]["endpoints"],
        json!([
            {"endpoint": "http://user:4318/v1/metrics"},
            {"endpoint": "http://system:4318/v1/metrics"}
        ])
    );
    assert_eq!(
        config["atif"]["storage"],
        json!([
            {"type": "http", "endpoint": "http://user/trajectory"},
            {"type": "http", "endpoint": "http://system/trajectory"}
        ])
    );
    assert_eq!(
        config["implementation"]["nested"],
        json!([9]),
        "deeper implementation-specific lists retain replacement semantics"
    );
}

#[test]
fn test_layer_config_preserves_signal_endpoint_option_semantics() {
    let lower = json!({
        "components": [{
            "kind": "observability",
            "config": {
                "opentelemetry": {
                    "logs": {
                        "enabled": true,
                        "endpoints": [{"endpoint": "http://system:4318/v1/logs"}]
                    },
                    "metrics": {
                        "enabled": true,
                        "endpoints": [{"endpoint": "http://system:4318/v1/metrics"}]
                    }
                }
            }
        }]
    });
    let higher = json!({
        "components": [{
            "kind": "observability",
            "config": {
                "opentelemetry": {
                    "logs": {"minimum_severity": "warn"},
                    "metrics": {"endpoints": []}
                }
            }
        }]
    });

    let mut merged = lower;
    layer_config(&mut merged, higher);
    let opentelemetry = &merged["components"][0]["config"]["opentelemetry"];

    assert_eq!(
        opentelemetry["logs"]["endpoints"],
        json!([{"endpoint": "http://system:4318/v1/logs"}]),
        "an omitted higher-precedence list preserves the explicit lower-precedence list"
    );
    assert_eq!(opentelemetry["logs"]["minimum_severity"], json!("warn"));
    assert_eq!(
        opentelemetry["metrics"]["endpoints"],
        json!([]),
        "an explicit empty higher-precedence list remains explicit and empty"
    );
}

#[test]
fn test_layer_config_replaces_observability_named_nested_lists_for_other_plugins() {
    let mut merged = json!({
        "components": [{
            "kind": "third.party",
            "config": {
                "atof": {"sinks": ["base"]},
                "opentelemetry": {
                    "endpoints": ["base"],
                    "logs": {"endpoints": ["base"]},
                    "metrics": {"endpoints": ["base"]}
                },
                "atif": {"storage": ["base"]}
            }
        }]
    });
    layer_config(
        &mut merged,
        json!({
            "components": [{
                "kind": "third.party",
                "config": {
                    "atof": {"sinks": ["overlay"]},
                    "opentelemetry": {
                        "endpoints": ["overlay"],
                        "logs": {"endpoints": ["overlay"]},
                        "metrics": {"endpoints": ["overlay"]}
                    },
                    "atif": {"storage": ["overlay"]}
                }
            }]
        }),
    );

    let config = &merged["components"][0]["config"];
    assert_eq!(config["atof"]["sinks"], json!(["overlay"]));
    assert_eq!(config["opentelemetry"]["endpoints"], json!(["overlay"]));
    assert_eq!(
        config["opentelemetry"]["logs"]["endpoints"],
        json!(["overlay"])
    );
    assert_eq!(
        config["opentelemetry"]["metrics"]["endpoints"],
        json!(["overlay"])
    );
    assert_eq!(config["atif"]["storage"], json!(["overlay"]));
}

#[test]
fn test_layer_config_preserves_multi_instance_kinds() {
    // A kind used more than once (multi-instance plugins) must not collapse into the first slot.
    let base = json!({ "components": [ { "kind": "multi", "config": { "n": 0 } } ] });
    let overlay = json!({
        "components": [
            { "kind": "multi", "config": { "n": 1 } },
            { "kind": "multi", "config": { "tag": "second" } }
        ]
    });

    let mut merged = base;
    layer_config(&mut merged, overlay);
    let components = merged["components"].as_array().unwrap();

    // First overlay instance pairs with the base instance; the second is appended, not dropped.
    assert_eq!(components.len(), 2);
    assert!(
        components
            .iter()
            .all(|component| component["kind"] == json!("multi"))
    );
    assert_eq!(components[0]["config"]["n"], json!(1));
    assert_eq!(components[1]["config"]["tag"], json!("second"));
}

#[test]
fn config_layering_helpers_cover_malformed_and_scalar_shapes() {
    let mut scalar = json!("lower");
    layer_config(&mut scalar, json!("higher"));
    assert_eq!(scalar, json!("higher"));

    let mut malformed_left = json!({"not": "components"});
    merge_plugin_components(&mut malformed_left, json!([]));
    assert_eq!(malformed_left, json!([]));

    let mut malformed_right = json!([]);
    merge_plugin_components(&mut malformed_right, json!({"not": "components"}));
    assert_eq!(malformed_right, json!({"not": "components"}));

    let mut components = json!([]);
    merge_plugin_components(&mut components, json!([{"enabled": true}]));
    assert_eq!(components, json!([{"enabled": true}]));

    let mut malformed_component = json!("lower");
    merge_plugin_component(&mut malformed_component, json!({"kind": "higher"}));
    assert_eq!(malformed_component, json!({"kind": "higher"}));

    let mut object = json!({"existing": true});
    merge_json_value(&mut object, json!({"added": true}));
    assert_eq!(object, json!({"existing": true, "added": true}));
}

#[test]
fn test_config_report_has_errors() {
    let report = ConfigReport {
        diagnostics: vec![ConfigDiagnostic {
            level: DiagnosticLevel::Error,
            code: "x".into(),
            component: None,
            field: None,
            message: "boom".into(),
        }],
        ..ConfigReport::default()
    };
    assert!(report.has_errors());
}

#[test]
fn test_register_and_deregister_plugin() {
    let _guard = lock_runtime_owner();
    reset_global();
    assert!(register_plugin(Arc::new(TestPlugin)).is_ok());
    match register_plugin(Arc::new(TestPlugin)) {
        Err(PluginError::RegistrationFailed(message)) => {
            assert!(message.contains("already registered"));
        }
        Err(other) => panic!("unexpected duplicate-registration error: {other}"),
        Ok(_) => panic!("expected duplicate registration to fail"),
    }
    assert!(list_plugin_kinds().contains(&"test.plugin".to_string()));
    assert!(lookup_plugin("test.plugin").is_some());
    assert!(deregister_plugin("test.plugin"));
    assert!(!deregister_plugin("missing.plugin"));
    assert!(clear_plugin_configuration().is_ok());
    assert!(active_plugin_report().is_none());
    reset_global();
}

#[test]
fn test_plugin_registration_context_registers_and_rolls_back() {
    let _guard = lock_runtime_owner();
    reset_global();

    let mut ctx = PluginRegistrationContext::new();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime
        .block_on(TestPlugin.register(&Map::new(), &mut ctx))
        .unwrap();

    let request = runtime
        .block_on(llm_request_intercepts(
            "model",
            LlmRequest {
                headers: Map::new(),
                content: json!({"messages": []}),
            },
        ))
        .unwrap();
    assert_eq!(request.request.headers.get("x-plugin"), Some(&json!(true)));

    let mut registrations = ctx.into_registrations();
    rollback_registrations(&mut registrations);

    let request = runtime
        .block_on(llm_request_intercepts(
            "model",
            LlmRequest {
                headers: Map::new(),
                content: json!({"messages": []}),
            },
        ))
        .unwrap();
    assert_eq!(request.request.headers.get("x-plugin"), None);
    reset_global();
}

#[test]
fn test_initialize_plugins_registers_and_clears_components() {
    let _guard = lock_runtime_owner();
    reset_global();
    register_plugin(Arc::new(TestPlugin)).unwrap();

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let report = runtime
        .block_on(initialize_plugins_exact(PluginConfig {
            components: vec![PluginComponentSpec::new("test.plugin")],
            ..PluginConfig::default()
        }))
        .unwrap();
    assert!(!report.has_errors());
    assert!(active_plugin_report().is_some());

    let request = runtime
        .block_on(llm_request_intercepts(
            "model",
            LlmRequest {
                headers: Map::new(),
                content: json!({"messages": []}),
            },
        ))
        .unwrap();
    assert_eq!(request.request.headers.get("x-plugin"), Some(&json!(true)));

    clear_plugin_configuration().unwrap();
    let request = runtime
        .block_on(llm_request_intercepts(
            "model",
            LlmRequest {
                headers: Map::new(),
                content: json!({"messages": []}),
            },
        ))
        .unwrap();
    assert_eq!(request.request.headers.get("x-plugin"), None);
    reset_global();
}

#[test]
fn test_plugin_configuration_registers_event_metadata_injector() {
    let _guard = lock_runtime_owner();
    reset_global();
    register_plugin(Arc::new(EventMetadataPlugin)).unwrap();

    let mut component = PluginComponentSpec::new("event-metadata.plugin");
    component.config = serde_json::from_value(json!({
        "metadata": {"nv.test.plugin_config": "configured"}
    }))
    .unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime
        .block_on(initialize_plugins_exact(PluginConfig {
            components: vec![component],
            ..PluginConfig::default()
        }))
        .unwrap();

    let callback = {
        let context = global_context();
        let state = context.read().unwrap();
        let injectors = state.event_metadata_injectors.sorted_values();
        assert_eq!(injectors.len(), 1);
        Arc::clone(&injectors[0].payload)
    };
    let event = Arc::new(Event::Mark(MarkEvent::new(
        BaseEvent::builder().name("plugin-configured-mark").build(),
        None,
        None,
    )));
    let additions = runtime.block_on(callback(event)).unwrap();
    assert_eq!(
        additions.get("nv.test.plugin_config"),
        Some(&json!("configured"))
    );

    clear_plugin_configuration().unwrap();
    assert!(
        global_context()
            .read()
            .unwrap()
            .event_metadata_injectors
            .sorted_values()
            .is_empty()
    );
    assert!(deregister_plugin("event-metadata.plugin"));
    reset_global();
}
#[test]
fn test_validate_plugin_config_honors_policy_and_duplicate_singletons() {
    let _guard = lock_runtime_owner();
    reset_global();
    register_plugin(Arc::new(SingletonPlugin)).unwrap();

    let report = validate_plugin_config(&PluginConfig {
        components: vec![
            PluginComponentSpec::new("singleton.plugin"),
            PluginComponentSpec::new("singleton.plugin"),
            PluginComponentSpec::new("missing.plugin"),
        ],
        policy: ConfigPolicy {
            unknown_component: UnsupportedBehavior::Warn,
            unknown_field: UnsupportedBehavior::Ignore,
            unsupported_value: UnsupportedBehavior::Error,
        },
        ..PluginConfig::default()
    });

    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.code == "plugin.duplicate_component")
    );
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.code == "plugin.unknown_component"
                && diag.level == DiagnosticLevel::Warning)
    );

    let ignored = validate_plugin_config(&PluginConfig {
        components: vec![PluginComponentSpec::new("still.missing")],
        policy: ConfigPolicy {
            unknown_component: UnsupportedBehavior::Ignore,
            ..PluginConfig::default().policy
        },
        ..PluginConfig::default()
    });
    assert!(ignored.diagnostics.is_empty());

    reset_global();
}

#[test]
fn test_validate_plugin_config_passes_top_level_policy_to_plugins() {
    let _guard = lock_runtime_owner();
    reset_global();
    register_plugin(Arc::new(PolicyAwarePlugin)).unwrap();

    let warning = validate_plugin_config(&PluginConfig {
        components: vec![PluginComponentSpec::new("policy-aware.plugin")],
        policy: ConfigPolicy {
            unsupported_value: UnsupportedBehavior::Warn,
            ..ConfigPolicy::default()
        },
        ..PluginConfig::default()
    });
    assert!(warning.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "policy-aware.unsupported_value"
            && diagnostic.level == DiagnosticLevel::Warning
    }));

    let ignored = validate_plugin_config(&PluginConfig {
        components: vec![PluginComponentSpec::new("policy-aware.plugin")],
        policy: ConfigPolicy {
            unsupported_value: UnsupportedBehavior::Ignore,
            ..ConfigPolicy::default()
        },
        ..PluginConfig::default()
    });
    assert!(ignored.diagnostics.is_empty());

    reset_global();
}

#[test]
fn test_plugin_config_defaults_debug_and_invalid_config_messages() {
    let _guard = lock_runtime_owner();
    reset_global();

    let config: PluginConfig = serde_json::from_value(json!({})).unwrap();
    assert_eq!(config.version, 1);
    assert!(config.components.is_empty());
    assert_eq!(config.policy.unknown_component, UnsupportedBehavior::Warn);
    assert_eq!(config.policy.unknown_field, UnsupportedBehavior::Warn);
    assert_eq!(config.policy.unsupported_value, UnsupportedBehavior::Error);

    let component: PluginComponentSpec =
        serde_json::from_value(json!({"kind": "demo.plugin"})).unwrap();
    assert_eq!(component.kind, "demo.plugin");
    assert!(component.enabled);
    assert!(component.config.is_empty());

    let registration = PluginRegistration::new("plugin", "demo::registration", Box::new(|| Ok(())));
    let debug = format!("{registration:?}");
    assert!(debug.contains("PluginRegistration"));
    assert!(debug.contains("demo::registration"));
    assert!(debug.contains("plugin"));

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let error = runtime
        .block_on(initialize_plugins_exact(PluginConfig {
            version: 2,
            components: vec![PluginComponentSpec::new("missing.plugin")],
            policy: ConfigPolicy {
                unknown_component: UnsupportedBehavior::Error,
                ..PluginConfig::default().policy
            },
        }))
        .unwrap_err();

    match error {
        PluginError::InvalidConfig(message) => {
            assert!(message.contains("plugin config version 2 is unsupported"));
            assert!(message.contains("plugin component kind 'missing.plugin' is unsupported"));
            assert!(message.contains(";"));
        }
        other => panic!("unexpected invalid config error: {other}"),
    }

    reset_global();
}

#[test]
fn test_plugin_helper_defaults_and_policy_diagnostics() {
    let _guard = lock_runtime_owner();
    reset_global();

    assert_eq!(default_warn(), UnsupportedBehavior::Warn);
    assert_eq!(default_error(), UnsupportedBehavior::Error);
    assert_eq!(default_plugin_config_version(), 1);
    assert!(default_enabled());
    assert_eq!(UnsupportedBehavior::default(), UnsupportedBehavior::Warn);

    let mut diagnostics = Vec::new();
    push_policy_diag(
        &mut diagnostics,
        UnsupportedBehavior::Ignore,
        "ignored.code",
        None,
        None,
        "ignored".into(),
    );
    assert!(diagnostics.is_empty());

    push_policy_diag(
        &mut diagnostics,
        UnsupportedBehavior::Warn,
        "warn.code",
        Some("warn.plugin".into()),
        Some("field".into()),
        "warn".into(),
    );
    push_policy_diag(
        &mut diagnostics,
        UnsupportedBehavior::Error,
        "error.code",
        Some("error.plugin".into()),
        None,
        "error".into(),
    );

    assert_eq!(diagnostics.len(), 2);
    assert_eq!(diagnostics[0].level, DiagnosticLevel::Warning);
    assert_eq!(diagnostics[0].component.as_deref(), Some("warn.plugin"));
    assert_eq!(diagnostics[0].field.as_deref(), Some("field"));
    assert_eq!(diagnostics[1].level, DiagnosticLevel::Error);
    assert_eq!(
        join_error_messages(&ConfigReport {
            diagnostics,
            ..ConfigReport::default()
        }),
        "error"
    );

    reset_global();
}

#[test]
fn test_plugin_component_helpers_and_serialization_error_variant() {
    let _guard = lock_runtime_owner();
    reset_global();

    let config = PluginConfig {
        components: vec![
            PluginComponentSpec::new("alpha.plugin"),
            PluginComponentSpec::new("beta.plugin"),
            PluginComponentSpec::new("alpha.plugin"),
        ],
        ..PluginConfig::default()
    };

    let totals = plugin_component_totals(&config);
    assert_eq!(totals.get("alpha.plugin"), Some(&2));
    assert_eq!(totals.get("beta.plugin"), Some(&1));
    assert_eq!(
        component_namespace("alpha.plugin", 1),
        "nemo-relay-plugin.v1.alpha.plugin:1:"
    );
    assert_eq!(
        component_namespace("beta.plugin", 1),
        "nemo-relay-plugin.v1.beta.plugin:1:"
    );
    assert_eq!(
        component_namespace("alpha/plugin:π%", 2),
        "nemo-relay-plugin.v1.alpha%2Fplugin%3A%CF%80%25:2:"
    );

    let parse_error = serde_json::from_str::<PluginConfig>("{").unwrap_err();
    let wrapped: PluginError = parse_error.into();
    match wrapped {
        PluginError::Serialization(message) => {
            assert!(!message.to_string().is_empty());
        }
        other => panic!("unexpected conversion result: {other}"),
    }

    reset_global();
}

#[test]
fn test_plugin_component_effective_names_escape_and_discover_fields() {
    let _guard = lock_runtime_owner();
    reset_global();

    let namespace = component_namespace("analytics/plugin:π%", 1);
    let mut plugin_context = PluginRegistrationContext::with_plugin_component_namespace(namespace);
    let mut global_registration_context = PluginRegistrationContext::new();
    plugin_context
        .register_subscriber("cache/value:π%", Arc::new(|_| {}))
        .unwrap();
    global_registration_context
        .register_subscriber("global:subscriber", Arc::new(|_| {}))
        .unwrap();
    global_registration_context
        .register_subscriber(
            "__nemo_relay_plugin__analytics__1__subscriber",
            Arc::new(|_| {}),
        )
        .unwrap();

    let registrations =
        list_runtime_registrations(Some(&BTreeSet::from([RuntimeRegistrationKind::Subscriber])))
            .unwrap();
    let plugin_registration = registrations
        .iter()
        .find(|registration| registration.local_name == "cache/value:π%")
        .unwrap();
    assert_eq!(
        plugin_registration.effective_name,
        "nemo-relay-plugin.v1.analytics%2Fplugin%3A%CF%80%25:1:cache%2Fvalue%3A%CF%80%25"
    );
    assert_eq!(
        plugin_registration.owner.plugin_kind.as_deref(),
        Some("analytics/plugin:π%")
    );
    assert_eq!(plugin_registration.owner.component_ordinal, Some(1));

    let global_registration = registrations
        .iter()
        .find(|registration| registration.effective_name == "global:subscriber")
        .unwrap();
    assert_eq!(global_registration.local_name, "global:subscriber");
    assert_eq!(
        global_registration.owner.kind,
        RuntimeRegistrationOwnerKind::GlobalApi
    );
    assert_eq!(global_registration.owner.component_ordinal, None);

    let legacy_style_global_registration = registrations
        .iter()
        .find(|registration| {
            registration.effective_name == "__nemo_relay_plugin__analytics__1__subscriber"
        })
        .unwrap();
    assert_eq!(
        legacy_style_global_registration.owner.kind,
        RuntimeRegistrationOwnerKind::GlobalApi
    );
    assert_eq!(
        legacy_style_global_registration.owner.component_ordinal,
        None
    );

    let child_context = plugin_context.with_child_namespace("profile/π:");
    assert_eq!(
        child_context.qualify_name("subscriber%"),
        "nemo-relay-plugin.v1.analytics%2Fplugin%3A%CF%80%25:1:profile%2F%CF%80%3Asubscriber%25"
    );
    assert!(
        decode_plugin_component_effective_name(
            "nemo-relay-plugin.v1.analytics%2Fplugin%3A%CF%80%25:1:cache%2Fvalue%3A%CF%80%25"
        )
        .is_some()
    );
    assert!(
        decode_plugin_component_effective_name(
            "nemo-relay-plugin.v1.analytics/plugin:1:subscriber"
        )
        .is_none()
    );
    assert!(
        decode_plugin_component_effective_name("nemo-relay-plugin.v1.analytics:0:subscriber")
            .is_none()
    );
    assert!(
        decode_plugin_component_effective_name("__nemo_relay_plugin__analytics__1__subscriber")
            .is_none()
    );

    let mut plugin_registrations = plugin_context.into_registrations();
    rollback_registrations(&mut plugin_registrations);
    let mut global_registrations = global_registration_context.into_registrations();
    rollback_registrations(&mut global_registrations);
    reset_global();
}

#[test]
fn test_registration_context_namespace_and_manual_registration_helpers() {
    let mut ctx = PluginRegistrationContext::with_namespace("demo::");
    assert_eq!(ctx.qualify_name("subscriber"), "demo::subscriber");

    ctx.add_registration(PluginRegistration::new(
        "plugin",
        "demo::manual".to_string(),
        Box::new(|| Ok(())),
    ));
    ctx.extend_registrations(vec![PluginRegistration::new(
        "plugin",
        "demo::extra".to_string(),
        Box::new(|| Ok(())),
    )]);

    let names = ctx
        .into_registrations()
        .into_iter()
        .map(|registration| registration.name)
        .collect::<Vec<_>>();
    assert_eq!(names, vec!["demo::manual", "demo::extra"]);
}

#[test]
fn test_plugin_registration_context_covers_all_registration_helpers() {
    let _guard = lock_runtime_owner();
    reset_global();

    let mut ctx = PluginRegistrationContext::with_namespace("demo::");
    ctx.register_subscriber("subscriber", Arc::new(|_event| {}))
        .unwrap();
    ctx.register_event_metadata_injector(
        "event-metadata",
        1,
        Arc::new(|_| Box::pin(async { Ok(BTreeMap::new()) })),
    )
    .unwrap();
    ctx.register_tool_request_intercept(
        "tool-request",
        1,
        false,
        Arc::new(|_name, args| Box::pin(async move { Ok(args) })),
    )
    .unwrap();
    ctx.register_tool_execution_intercept(
        "tool-exec",
        1,
        Arc::new(|_name, args, _next| Box::pin(async move { Ok(args.into()) })),
    )
    .unwrap();
    ctx.register_llm_request_intercept(
        "llm-request",
        1,
        false,
        Arc::new(|_name, request, annotated| {
            Box::pin(async move { Ok(LlmRequestInterceptOutcome::new(request, annotated)) })
        }),
    )
    .unwrap();
    ctx.register_llm_execution_intercept(
        "llm-exec",
        1,
        Arc::new(|_name, request, _next| Box::pin(async move { Ok(request.content) })),
    )
    .unwrap();
    ctx.register_llm_stream_execution_intercept(
        "llm-stream",
        1,
        Arc::new(|_name, request, _next| {
            Box::pin(async move {
                Ok(LlmJsonStream::new(tokio_stream::iter(vec![Ok(
                    request.content
                )])))
            })
        }),
    )
    .unwrap();

    let mut registrations = ctx.into_registrations();
    let names = registrations
        .iter()
        .map(|registration| registration.name.clone())
        .collect::<Vec<_>>();
    assert_eq!(
        names,
        vec![
            "demo::subscriber",
            "demo::event-metadata",
            "demo::tool-request",
            "demo::tool-exec",
            "demo::llm-request",
            "demo::llm-exec",
            "demo::llm-stream",
        ]
    );

    rollback_registrations(&mut registrations);
    assert!(registrations.is_empty());
    reset_global();
}

#[test]
fn test_rollback_registrations_runs_in_reverse_and_ignores_failures() {
    let mut registrations = vec![];
    let call_order = Arc::new(Mutex::new(Vec::new()));

    let first_order = Arc::clone(&call_order);
    registrations.push(PluginRegistration::new(
        "plugin",
        "first",
        Box::new(move || {
            first_order.lock().unwrap().push("first");
            Ok(())
        }),
    ));

    let panic_order = Arc::clone(&call_order);
    registrations.push(PluginRegistration::new(
        "plugin",
        "panicking",
        Box::new(move || {
            panic_order.lock().unwrap().push("panicking");
            panic!("expected rollback panic")
        }),
    ));

    let second_order = Arc::clone(&call_order);
    registrations.push(PluginRegistration::new(
        "plugin",
        "second",
        Box::new(move || {
            second_order.lock().unwrap().push("second");
            Err(PluginError::RegistrationFailed(
                "expected rollback failure".into(),
            ))
        }),
    ));

    rollback_registrations(&mut registrations);

    assert!(registrations.is_empty());
    assert_eq!(
        *call_order.lock().unwrap(),
        vec!["second", "panicking", "first"]
    );
}

#[test]
fn test_initialize_plugins_restores_previous_configuration_after_failed_replacement() {
    let _guard = lock_runtime_owner();
    reset_global();
    register_plugin(Arc::new(RecordingPlugin)).unwrap();
    register_plugin(Arc::new(RestoreFailPlugin)).unwrap();

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    runtime
        .block_on(initialize_plugins_exact(PluginConfig {
            components: vec![PluginComponentSpec::new("recording.plugin")],
            ..PluginConfig::default()
        }))
        .unwrap();
    record_active_plugin_runtime_diagnostic(RuntimeDiagnostic {
        code: "atif.remote_delivery_failed".into(),
        component: "observability".into(),
        field: Some("storage[0]".into()),
        message: "HTTP 500".into(),
        session_id: Some("session-123".into()),
        count: 1,
    });

    let err = runtime
        .block_on(initialize_plugins_exact(PluginConfig {
            components: vec![PluginComponentSpec::new("restore.fail.plugin")],
            ..PluginConfig::default()
        }))
        .unwrap_err();
    match err {
        PluginError::RegistrationFailed(message) => {
            assert!(message.contains("restore.fail.plugin refused to initialize"));
        }
        other => panic!("unexpected replacement failure: {other}"),
    }

    assert_eq!(RESTORE_FAIL_REGISTRATIONS.load(Ordering::SeqCst), 1);
    let restored_report = active_plugin_report().expect("previous config should be restored");
    assert!(restored_report.diagnostics.is_empty());
    assert_eq!(restored_report.runtime_diagnostics.len(), 1);
    let restored = &restored_report.runtime_diagnostics[0];
    assert_eq!(restored.code, "atif.remote_delivery_failed");
    assert_eq!(restored.component, "observability");
    assert_eq!(restored.field.as_deref(), Some("storage[0]"));
    assert_eq!(restored.message, "HTTP 500");
    assert_eq!(restored.session_id.as_deref(), Some("session-123"));
    assert_eq!(restored.count, 1);
    let diagnostics = active_runtime_diagnostics_snapshot();
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0].code, "atif.remote_delivery_failed");
    assert_eq!(diagnostics[0].message, "HTTP 500");
    assert_eq!(diagnostics[0].count, 1);
    let names = recorded_names().lock().unwrap().clone();
    assert_eq!(
        names,
        vec![
            "nemo-relay-plugin.v1.recording.plugin:1:subscriber",
            "nemo-relay-plugin.v1.recording.plugin:1:subscriber",
        ]
    );
    reset_global();
}

#[test]
fn test_initialize_plugins_restores_previous_configuration_after_replacement_panic() {
    let _guard = lock_runtime_owner();
    reset_global();
    register_plugin(Arc::new(RecordingPlugin)).unwrap();
    register_plugin(Arc::new(PanickingPlugin)).unwrap();

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime
        .block_on(initialize_plugins_exact(PluginConfig {
            components: vec![PluginComponentSpec::new("recording.plugin")],
            ..PluginConfig::default()
        }))
        .unwrap();

    let error = runtime
        .block_on(initialize_plugins_exact(PluginConfig {
            components: vec![PluginComponentSpec::new("panicking.plugin")],
            ..PluginConfig::default()
        }))
        .unwrap_err();
    assert!(
        error.to_string().contains("fixture plugin panicked"),
        "{error}"
    );
    assert!(active_plugin_report().is_some());
    assert_eq!(
        recorded_names().lock().unwrap().as_slice(),
        [
            "nemo-relay-plugin.v1.recording.plugin:1:subscriber",
            "nemo-relay-plugin.v1.recording.plugin:1:subscriber",
        ]
    );

    reset_global();
}

#[test]
fn test_initialize_plugins_rolls_back_partial_component_registration_on_failure() {
    let _guard = lock_runtime_owner();
    reset_global();
    register_plugin(Arc::new(PartialFailPlugin)).unwrap();

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let err = runtime
        .block_on(initialize_plugins_exact(PluginConfig {
            components: vec![PluginComponentSpec::new("partial.fail.plugin")],
            ..PluginConfig::default()
        }))
        .unwrap_err();

    match err {
        PluginError::RegistrationFailed(message) => {
            assert!(message.contains("partial.fail.plugin refused to finish initialization"));
        }
        other => panic!("unexpected partial registration failure: {other}"),
    }

    assert_eq!(PARTIAL_FAIL_ROLLBACKS.load(Ordering::SeqCst), 1);
    assert!(active_plugin_report().is_none());
    reset_global();
}

#[test]
fn test_initialize_plugins_transaction_finishes_after_caller_cancellation() {
    let _guard = lock_runtime_owner();
    reset_global();
    register_plugin(Arc::new(RecordingPlugin)).unwrap();
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let registered = Arc::new(Notify::new());
    register_plugin(Arc::new(BlockingPlugin {
        started: Arc::clone(&started),
        release: Arc::clone(&release),
        registered: Arc::clone(&registered),
    }))
    .unwrap();

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        initialize_plugins_exact(PluginConfig {
            components: vec![PluginComponentSpec::new("recording.plugin")],
            ..PluginConfig::default()
        })
        .await
        .unwrap();

        let caller = tokio::spawn(initialize_plugins_exact(PluginConfig {
            components: vec![PluginComponentSpec::new("blocking.plugin")],
            ..PluginConfig::default()
        }));
        started.notified().await;
        caller.abort();
        assert!(caller.await.unwrap_err().is_cancelled());
        release.notify_one();
        registered.notified().await;

        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            loop {
                if active_plugin_report().is_some_and(|report| {
                    report
                        .diagnostics
                        .iter()
                        .any(|diagnostic| diagnostic.code == "blocking.warning")
                }) {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("owned initialization transaction did not finish after caller cancellation");

        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            loop {
                if *PLUGIN_MUTATION_OWNER.lock().unwrap() == PluginMutationOwner::Idle {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("owned initialization transaction did not release its mutation lease");
    });

    reset_global();
}

#[test]
fn test_plugin_runtime_continues_driving_background_tasks_after_initialization() {
    let _guard = lock_runtime_owner();
    reset_global();
    let release = Arc::new(Notify::new());
    let completed = Arc::new(Notify::new());
    register_plugin(Arc::new(BackgroundTaskPlugin {
        release: Arc::clone(&release),
        completed: Arc::clone(&completed),
    }))
    .unwrap();

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        initialize_plugins_exact(PluginConfig {
            components: vec![PluginComponentSpec::new("background.task.plugin")],
            ..PluginConfig::default()
        })
        .await
        .unwrap();

        release.notify_one();
        tokio::time::timeout(std::time::Duration::from_secs(1), completed.notified())
            .await
            .expect("plugin runtime should keep driving spawned background tasks");
    });

    reset_global();
}

#[test]
fn test_pending_registration_records_rollback_failures() {
    let failures = Arc::new(Mutex::new(Vec::new()));
    {
        let mut pending = PendingPluginRegistrations::new(Some(Arc::clone(&failures)));
        pending.extend(vec![PluginRegistration::new(
            "fixture",
            "failed-rollback",
            Box::new(|| {
                Err(PluginError::RegistrationFailed(
                    "rollback remained registered".into(),
                ))
            }),
        )]);
    }

    let failures = failures.lock().unwrap();
    assert_eq!(failures.len(), 1);
    assert!(failures[0].contains("failed-rollback"));
    assert!(failures[0].contains("rollback remained registered"));
}

#[test]
fn test_pending_rollbacks_ignore_delivery_only_errors() {
    let failures = Arc::new(Mutex::new(Vec::new()));
    let delivery_error = || {
        PluginRegistrationCleanupOutcome::RemovedWithError(PluginError::RegistrationFailed(
            "delivery failed".into(),
        ))
    };
    {
        let mut pending = PendingPluginRegistrations::new(Some(Arc::clone(&failures)));
        pending.extend(vec![PluginRegistration::new_with_outcome(
            "fixture",
            "delivery-only-registration",
            Box::new(delivery_error),
        )]);
    }
    {
        let mut pending =
            PendingPluginRegistrationContext::new("fixture.".into(), Some(Arc::clone(&failures)));
        pending
            .context
            .add_registration(PluginRegistration::new_with_outcome(
                "fixture",
                "delivery-only-context-registration",
                Box::new(delivery_error),
            ));
    }

    assert!(failures.lock().unwrap().is_empty());
}

#[test]
fn test_checked_teardown_reports_unremoved_registrations() {
    let _guard = lock_runtime_owner();
    reset_global();
    store_active_plugin_configuration(
        PluginConfig::default(),
        ConfigReport::default(),
        vec![PluginRegistration::new(
            "fixture",
            "stale-callback",
            Box::new(|| {
                Err(PluginError::RegistrationFailed(
                    "deregistration refused".into(),
                ))
            }),
        )],
    )
    .unwrap();

    let outcome = clear_plugin_configuration_inner();
    assert!(!outcome.callbacks_cleared);
    let error = outcome.result.unwrap_err().to_string();
    assert!(error.contains("stale-callback"), "{error}");
    assert!(error.contains("deregistration refused"), "{error}");
    assert!(active_plugin_report().is_none());
    reset_global();
}

#[test]
fn test_teardown_marker_text_does_not_imply_successful_removal() {
    let _guard = lock_runtime_owner();
    reset_global();
    store_active_plugin_configuration(
        PluginConfig::default(),
        ConfigReport::default(),
        vec![PluginRegistration::new(
            "fixture",
            "stale-marker-callback",
            Box::new(|| {
                Err(PluginError::RegistrationFailed(format!(
                    "unrelated failure mentioning {}",
                    crate::plugin::ATIF_RUNTIME_DELIVERY_FAILURE_MARKER
                )))
            }),
        )],
    )
    .unwrap();

    let outcome = clear_plugin_configuration_inner();
    assert!(!outcome.callbacks_cleared);
    let error = outcome.result.unwrap_err().to_string();
    assert!(error.contains("stale-marker-callback"), "{error}");
    assert!(error.contains("could not be removed"), "{error}");
    reset_global();
}

#[test]
fn test_teardown_runtime_diagnostics_remain_in_the_plugin_report() {
    let _guard = lock_runtime_owner();
    reset_global();
    store_active_plugin_configuration(
        PluginConfig::default(),
        ConfigReport::default(),
        vec![PluginRegistration::new_with_outcome(
            "fixture",
            "atif-shutdown",
            Box::new(|| {
                record_active_plugin_runtime_diagnostic(RuntimeDiagnostic {
                    code: "atif.remote_delivery_failed".into(),
                    component: "observability".into(),
                    field: Some("storage[0]".into()),
                    message: "HTTP 500".into(),
                    session_id: Some("session-123".into()),
                    count: 1,
                });
                let error = PluginError::RegistrationFailed(format!(
                    "{}: atif.remote_delivery_failed (1)",
                    crate::plugin::ATIF_RUNTIME_DELIVERY_FAILURE_MARKER
                ));
                PluginRegistrationCleanupOutcome::RemovedWithError(error)
            }),
        )],
    )
    .unwrap();

    let outcome = clear_plugin_configuration_inner();
    assert!(outcome.callbacks_cleared);
    let error = outcome.result.unwrap_err().to_string();
    assert!(error.contains("atif.remote_delivery_failed"), "{error}");
    assert!(!error.contains("could not be removed"), "{error}");
    let report = active_plugin_report().expect("failed teardown should retain its report");
    assert_eq!(report.runtime_diagnostics.len(), 1);
    let diagnostic = &report.runtime_diagnostics[0];
    assert_eq!(diagnostic.code, "atif.remote_delivery_failed");
    assert_eq!(diagnostic.field.as_deref(), Some("storage[0]"));
    assert_eq!(diagnostic.message, "HTTP 500");
    assert_eq!(diagnostic.session_id.as_deref(), Some("session-123"));

    clear_plugin_configuration_inner();
    assert!(active_plugin_report().is_none());
    reset_global();
}

#[test]
fn dynamic_plugin_runtime_diagnostics_are_active_only_and_aggregated_by_code() {
    let _guard = lock_runtime_owner();
    reset_global();
    store_active_plugin_configuration(PluginConfig::default(), ConfigReport::default(), vec![])
        .unwrap();

    for diagnostic in [
        RuntimeDiagnostic {
            code: "zeta".into(),
            component: "observability".into(),
            field: None,
            message: "first zeta".into(),
            session_id: None,
            count: 2,
        },
        RuntimeDiagnostic {
            code: "alpha".into(),
            component: "observability".into(),
            field: Some("logs[0]".into()),
            message: "alpha".into(),
            session_id: None,
            count: 1,
        },
        RuntimeDiagnostic {
            code: "zeta".into(),
            component: "observability".into(),
            field: Some("metrics[0]".into()),
            message: "latest zeta".into(),
            session_id: None,
            count: u64::MAX,
        },
    ] {
        record_active_plugin_runtime_diagnostic(diagnostic);
    }

    let diagnostics = active_runtime_diagnostics_snapshot();
    assert_eq!(
        diagnostics
            .iter()
            .map(|diagnostic| (
                diagnostic.code.as_str(),
                diagnostic.message.as_str(),
                diagnostic.count
            ))
            .collect::<Vec<_>>(),
        vec![("alpha", "alpha", 1), ("zeta", "latest zeta", u64::MAX)]
    );

    clear_plugin_configuration_inner();
    assert!(active_runtime_diagnostics_snapshot().is_empty());
    reset_global();
}

#[test]
fn dynamic_plugin_runtime_diagnostics_snapshot_is_bounded() {
    let _guard = lock_runtime_owner();
    reset_global();
    store_active_plugin_configuration(PluginConfig::default(), ConfigReport::default(), vec![])
        .unwrap();

    for index in 0..=MAX_DYNAMIC_PLUGIN_RUNTIME_DIAGNOSTICS {
        record_active_plugin_runtime_diagnostic(RuntimeDiagnostic {
            code: format!("diagnostic.{index:02}"),
            component: "observability".into(),
            field: None,
            message: "runtime diagnostic".into(),
            session_id: None,
            count: 1,
        });
    }

    let diagnostics = active_runtime_diagnostics_snapshot();
    assert_eq!(diagnostics.len(), MAX_DYNAMIC_PLUGIN_RUNTIME_DIAGNOSTICS);
    assert_eq!(
        diagnostics
            .first()
            .map(|diagnostic| diagnostic.code.as_str()),
        Some("diagnostic.00")
    );
    assert!(
        diagnostics
            .iter()
            .all(|diagnostic| diagnostic.code != "diagnostic.32")
    );

    clear_plugin_configuration_inner();
    reset_global();
}

#[test]
fn test_opentelemetry_delivery_failure_allows_later_plugin_configuration() {
    let _guard = lock_runtime_owner();
    reset_global();
    store_active_plugin_configuration(
        PluginConfig::default(),
        ConfigReport::default(),
        vec![PluginRegistration::new_with_outcome(
            "fixture",
            "opentelemetry-shutdown",
            Box::new(|| {
                PluginRegistrationCleanupOutcome::RemovedWithError(PluginError::RegistrationFailed(
                    format!(
                        "{}: otel.spans_dropped (2)",
                        crate::plugin::OTEL_RUNTIME_DELIVERY_FAILURE_MARKER
                    ),
                ))
            }),
        )],
    )
    .unwrap();

    let outcome = clear_plugin_configuration_inner();
    assert!(outcome.callbacks_cleared);
    assert!(outcome.result.is_err());
    reset_global();
}

#[test]
fn test_mixed_opentelemetry_shutdown_failure_blocks_later_configuration() {
    let _guard = lock_runtime_owner();
    reset_global();
    store_active_plugin_configuration(
        PluginConfig::default(),
        ConfigReport::default(),
        vec![PluginRegistration::new_with_outcome(
            "fixture",
            "opentelemetry-shutdown",
            Box::new(|| {
                PluginRegistrationCleanupOutcome::NotRemoved(PluginError::RegistrationFailed(
                    format!(
                        "OpenTelemetry shutdown failures: provider error: {}: otel.spans_dropped (2); endpoint shutdown timed out",
                        crate::plugin::OTEL_RUNTIME_DELIVERY_FAILURE_MARKER
                    ),
                ))
            }),
        )],
    )
    .unwrap();

    let outcome = clear_plugin_configuration_inner();
    assert!(!outcome.callbacks_cleared);
    let error = outcome.result.unwrap_err().to_string();
    assert!(error.contains("endpoint shutdown timed out"), "{error}");
    reset_global();
}

#[test]
fn test_replacement_teardown_runtime_diagnostics_remain_in_the_plugin_report() {
    let _guard = lock_runtime_owner();
    reset_global();
    store_active_plugin_configuration(
        PluginConfig::default(),
        ConfigReport::default(),
        vec![PluginRegistration::new_with_outcome(
            "fixture",
            "atif-shutdown",
            Box::new(|| {
                record_active_plugin_runtime_diagnostic(RuntimeDiagnostic {
                    code: "atif.remote_delivery_failed".into(),
                    component: "observability".into(),
                    field: Some("storage[0]".into()),
                    message: "HTTP 500".into(),
                    session_id: Some("session-123".into()),
                    count: 1,
                });
                PluginRegistrationCleanupOutcome::RemovedWithError(PluginError::RegistrationFailed(
                    "ATIF delivery failed".into(),
                ))
            }),
        )],
    )
    .unwrap();

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let error = runtime
        .block_on(initialize_plugins_exact(PluginConfig::default()))
        .unwrap_err();
    assert!(
        error.to_string().contains("ATIF delivery failed"),
        "{error}"
    );
    assert!(
        error
            .to_string()
            .contains("fixture registration 'atif-shutdown' reported a delivery failure"),
        "{error}"
    );
    assert!(
        !plugin_configuration_is_active().unwrap(),
        "a replacement aborted by delivery failure must not leave a configuration active"
    );
    let report = active_plugin_report().expect("failed replacement should retain its report");
    assert_eq!(report.runtime_diagnostics.len(), 1);
    assert_eq!(
        report.runtime_diagnostics[0].code,
        "atif.remote_delivery_failed"
    );
    assert_eq!(
        report.runtime_diagnostics[0].field.as_deref(),
        Some("storage[0]")
    );
    assert_eq!(report.runtime_diagnostics[0].count, 1);

    runtime
        .block_on(initialize_plugins_exact(PluginConfig::default()))
        .expect("delivery-only teardown errors must not block a later initialization");
    reset_global();
}

#[test]
fn test_legacy_clear_retains_mutation_owner_after_incomplete_teardown() {
    let _guard = lock_runtime_owner();
    let owner_cleanup = PluginMutationOwnerCleanup;
    reset_global();
    store_active_plugin_configuration(
        PluginConfig::default(),
        ConfigReport::default(),
        vec![PluginRegistration::new(
            "fixture",
            "stale-callback",
            Box::new(|| panic!("fixture deregistration panicked")),
        )],
    )
    .unwrap();

    let error = clear_plugin_configuration().unwrap_err();
    assert!(error.to_string().contains("stale-callback"), "{error}");
    assert!(
        error
            .to_string()
            .contains("fixture deregistration panicked"),
        "{error}"
    );
    assert_eq!(
        *PLUGIN_MUTATION_OWNER.lock().unwrap(),
        PluginMutationOwner::Legacy
    );
    assert!(matches!(
        clear_plugin_configuration(),
        Err(PluginError::Conflict(_))
    ));

    drop(owner_cleanup);
    reset_global();
}

#[test]
fn test_legacy_replace_retains_mutation_owner_after_incomplete_teardown() {
    let _guard = lock_runtime_owner();
    let owner_cleanup = PluginMutationOwnerCleanup;
    reset_global();
    register_plugin(Arc::new(FailingDeregisterPlugin)).unwrap();
    register_plugin(Arc::new(ReplacementPlugin)).unwrap();

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime
        .block_on(initialize_plugins_exact(PluginConfig {
            components: vec![PluginComponentSpec::new("failing.deregister.plugin")],
            ..PluginConfig::default()
        }))
        .unwrap();

    let error = runtime
        .block_on(initialize_plugins_exact(PluginConfig {
            components: vec![PluginComponentSpec::new("replacement.plugin")],
            ..PluginConfig::default()
        }))
        .unwrap_err();
    assert!(
        error.to_string().contains("fixture deregistration refused"),
        "{error}"
    );
    assert_eq!(REPLACEMENT_REGISTRATIONS.load(Ordering::SeqCst), 0);
    assert_eq!(
        *PLUGIN_MUTATION_OWNER.lock().unwrap(),
        PluginMutationOwner::Legacy
    );
    assert!(active_plugin_report().is_none());

    drop(owner_cleanup);
    reset_global();
}

#[test]
fn test_initialize_plugins_skips_disabled_components_and_namespaces_multiple_instances() {
    let _guard = lock_runtime_owner();
    reset_global();
    register_plugin(Arc::new(RecordingPlugin)).unwrap();

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime
        .block_on(initialize_plugins_exact(PluginConfig {
            components: vec![
                PluginComponentSpec::new("recording.plugin"),
                PluginComponentSpec {
                    enabled: false,
                    ..PluginComponentSpec::new("recording.plugin")
                },
                PluginComponentSpec::new("recording.plugin"),
            ],
            ..PluginConfig::default()
        }))
        .unwrap();

    let names = recorded_names().lock().unwrap().clone();
    assert_eq!(
        names,
        vec![
            "nemo-relay-plugin.v1.recording.plugin:1:subscriber",
            "nemo-relay-plugin.v1.recording.plugin:2:subscriber",
        ]
    );
    reset_global();
}

#[test]
fn test_initialize_plugins_assigns_ordinal_to_lone_enabled_component() {
    let _guard = lock_runtime_owner();
    reset_global();
    register_plugin(Arc::new(RecordingPlugin)).unwrap();

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime
        .block_on(initialize_plugins_exact(PluginConfig {
            components: vec![
                PluginComponentSpec::new("recording.plugin"),
                PluginComponentSpec {
                    enabled: false,
                    ..PluginComponentSpec::new("recording.plugin")
                },
            ],
            ..PluginConfig::default()
        }))
        .unwrap();

    assert_eq!(
        recorded_names().lock().unwrap().as_slice(),
        ["nemo-relay-plugin.v1.recording.plugin:1:subscriber"]
    );
    reset_global();
}

#[test]
fn test_singleton_plugin_registration_discovers_ordinal() {
    let _guard = lock_runtime_owner();
    reset_global();
    register_plugin(Arc::new(DiscoverablePlugin)).unwrap();

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime
        .block_on(initialize_plugins_exact(PluginConfig {
            components: vec![PluginComponentSpec::new("discoverable.plugin")],
            ..PluginConfig::default()
        }))
        .unwrap();

    let registrations =
        list_runtime_registrations(Some(&BTreeSet::from([RuntimeRegistrationKind::Subscriber])))
            .unwrap();
    let registration = registrations
        .iter()
        .find(|registration| registration.local_name == "subscriber")
        .unwrap();
    assert_eq!(
        registration.effective_name,
        "nemo-relay-plugin.v1.discoverable.plugin:1:subscriber"
    );
    assert_eq!(
        registration.owner.plugin_kind.as_deref(),
        Some("discoverable.plugin")
    );
    assert_eq!(registration.owner.component_ordinal, Some(1));

    clear_plugin_configuration().unwrap();
    reset_global();
}

#[test]
fn test_initialize_plugins_reports_missing_component_during_activation() {
    let _guard = lock_runtime_owner();
    reset_global();
    register_plugin(Arc::new(VanishingPlugin)).unwrap();

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let error = runtime
        .block_on(initialize_plugins_exact(PluginConfig {
            components: vec![PluginComponentSpec::new("vanishing.plugin")],
            ..PluginConfig::default()
        }))
        .unwrap_err();

    match error {
        PluginError::NotFound(message) => {
            assert!(message.contains("vanishing.plugin"));
            assert!(active_plugin_report().is_none());
        }
        other => panic!("unexpected activation failure: {other}"),
    }

    reset_global();
}

#[test]
fn test_plugin_registration_context_supports_guardrail_helpers() {
    let _guard = lock_runtime_owner();
    reset_global();

    let mut ctx = PluginRegistrationContext::with_namespace("plugin::");
    ctx.register_mark_sanitize_guardrail(
        "mark_sanitize",
        1,
        Arc::new(|_, fields| Box::pin(async move { Ok(fields) })),
    )
    .unwrap();
    ctx.register_scope_sanitize_start_guardrail(
        "scope_sanitize_start",
        1,
        Arc::new(|_, fields| Box::pin(async move { Ok(fields) })),
    )
    .unwrap();
    ctx.register_scope_sanitize_end_guardrail(
        "scope_sanitize_end",
        1,
        Arc::new(|_, fields| Box::pin(async move { Ok(fields) })),
    )
    .unwrap();
    ctx.register_tool_sanitize_request_guardrail(
        "tool_sanitize_request",
        1,
        Arc::new(|_, args| Box::pin(async move { Ok(args) })),
    )
    .unwrap();
    ctx.register_tool_sanitize_response_guardrail(
        "tool_sanitize_response",
        1,
        Arc::new(|_, response| Box::pin(async move { Ok(response) })),
    )
    .unwrap();
    ctx.register_tool_conditional_execution_guardrail(
        "tool_conditional",
        1,
        Arc::new(|name, _args| {
            Box::pin(
                async move { Ok((name == "blocked-tool").then(|| "blocked tool".to_string())) },
            )
        }),
    )
    .unwrap();
    ctx.register_llm_sanitize_request_guardrail(
        "llm_sanitize_request",
        1,
        Arc::new(|request, _context| Box::pin(async move { Ok(Some(request)) })),
    )
    .unwrap();
    ctx.register_llm_sanitize_response_guardrail(
        "llm_sanitize_response",
        1,
        Arc::new(|response, _context| Box::pin(async move { Ok(Some(response)) })),
    )
    .unwrap();
    ctx.register_llm_conditional_execution_guardrail(
        "llm_conditional",
        1,
        Arc::new(|request| {
            let blocked = request.headers.get("blocked") == Some(&json!(true));
            Box::pin(async move { Ok(blocked.then(|| "blocked llm".to_string())) })
        }),
    )
    .unwrap();

    let runtime = tokio::runtime::Runtime::new().unwrap();
    match runtime.block_on(tool_conditional_execution("blocked-tool", &json!({}))) {
        Err(FlowError::GuardrailRejected(message)) => assert_eq!(message, "blocked tool"),
        other => panic!("expected tool guardrail rejection, got {other:?}"),
    }

    match runtime.block_on(llm_conditional_execution(&LlmRequest {
        headers: Map::from_iter([(String::from("blocked"), json!(true))]),
        content: json!({"messages": []}),
    })) {
        Err(FlowError::GuardrailRejected(message)) => assert_eq!(message, "blocked llm"),
        other => panic!("expected llm guardrail rejection, got {other:?}"),
    }

    let mut registrations = ctx.into_registrations();
    rollback_registrations(&mut registrations);

    assert!(
        runtime
            .block_on(tool_conditional_execution("blocked-tool", &json!({})))
            .is_ok()
    );
    assert!(
        runtime
            .block_on(llm_conditional_execution(&LlmRequest {
                headers: Map::from_iter([(String::from("blocked"), json!(true))]),
                content: json!({"messages": []}),
            }))
            .is_ok()
    );

    reset_global();
}

#[test]
fn test_plugin_registration_context_maps_duplicate_registration_errors() {
    let _guard = lock_runtime_owner();
    reset_global();

    let mut ctx = PluginRegistrationContext::with_namespace("duplicate::");
    ctx.register_mark_sanitize_guardrail(
        "mark",
        1,
        Arc::new(|_, fields| Box::pin(async move { Ok(fields) })),
    )
    .unwrap();
    expect_registration_failed(
        ctx.register_mark_sanitize_guardrail(
            "mark",
            1,
            Arc::new(|_, fields| Box::pin(async move { Ok(fields) })),
        ),
        "mark sanitizer:",
    );
    ctx.register_scope_sanitize_start_guardrail(
        "scope-start",
        1,
        Arc::new(|_, fields| Box::pin(async move { Ok(fields) })),
    )
    .unwrap();
    expect_registration_failed(
        ctx.register_scope_sanitize_start_guardrail(
            "scope-start",
            1,
            Arc::new(|_, fields| Box::pin(async move { Ok(fields) })),
        ),
        "scope-start sanitizer:",
    );
    ctx.register_scope_sanitize_end_guardrail(
        "scope-end",
        1,
        Arc::new(|_, fields| Box::pin(async move { Ok(fields) })),
    )
    .unwrap();
    expect_registration_failed(
        ctx.register_scope_sanitize_end_guardrail(
            "scope-end",
            1,
            Arc::new(|_, fields| Box::pin(async move { Ok(fields) })),
        ),
        "scope-end sanitizer:",
    );
    ctx.register_llm_request_intercept(
        "llm-request",
        1,
        false,
        Arc::new(|_name, request, annotated| {
            Box::pin(async move { Ok(LlmRequestInterceptOutcome::new(request, annotated)) })
        }),
    )
    .unwrap();
    expect_registration_failed(
        ctx.register_llm_request_intercept(
            "llm-request",
            1,
            false,
            Arc::new(|_name, request, annotated| {
                Box::pin(async move { Ok(LlmRequestInterceptOutcome::new(request, annotated)) })
            }),
        ),
        "llm request intercept:",
    );

    ctx.register_tool_sanitize_request_guardrail(
        "tool-sanitize-request",
        1,
        Arc::new(|_, args| Box::pin(async move { Ok(args) })),
    )
    .unwrap();
    expect_registration_failed(
        ctx.register_tool_sanitize_request_guardrail(
            "tool-sanitize-request",
            1,
            Arc::new(|_, args| Box::pin(async move { Ok(args) })),
        ),
        "tool sanitize request guardrail:",
    );

    ctx.register_tool_sanitize_response_guardrail(
        "tool-sanitize-response",
        1,
        Arc::new(|_, response| Box::pin(async move { Ok(response) })),
    )
    .unwrap();
    expect_registration_failed(
        ctx.register_tool_sanitize_response_guardrail(
            "tool-sanitize-response",
            1,
            Arc::new(|_, response| Box::pin(async move { Ok(response) })),
        ),
        "tool sanitize response guardrail:",
    );

    ctx.register_tool_conditional_execution_guardrail(
        "tool-conditional",
        1,
        Arc::new(|_, _| Box::pin(async { Ok(None) })),
    )
    .unwrap();
    expect_registration_failed(
        ctx.register_tool_conditional_execution_guardrail(
            "tool-conditional",
            1,
            Arc::new(|_, _| Box::pin(async { Ok(None) })),
        ),
        "tool conditional execution guardrail:",
    );

    ctx.register_llm_sanitize_request_guardrail(
        "llm-sanitize-request",
        1,
        Arc::new(|request, _context| Box::pin(async move { Ok(Some(request)) })),
    )
    .unwrap();
    expect_registration_failed(
        ctx.register_llm_sanitize_request_guardrail(
            "llm-sanitize-request",
            1,
            Arc::new(|request, _context| Box::pin(async move { Ok(Some(request)) })),
        ),
        "llm sanitize request guardrail:",
    );

    ctx.register_llm_sanitize_response_guardrail(
        "llm-sanitize-response",
        1,
        Arc::new(|response, _context| Box::pin(async move { Ok(Some(response)) })),
    )
    .unwrap();
    expect_registration_failed(
        ctx.register_llm_sanitize_response_guardrail(
            "llm-sanitize-response",
            1,
            Arc::new(|response, _context| Box::pin(async move { Ok(Some(response)) })),
        ),
        "llm sanitize response guardrail:",
    );

    ctx.register_llm_conditional_execution_guardrail(
        "llm-conditional",
        1,
        Arc::new(|_| Box::pin(async { Ok(None) })),
    )
    .unwrap();
    expect_registration_failed(
        ctx.register_llm_conditional_execution_guardrail(
            "llm-conditional",
            1,
            Arc::new(|_| Box::pin(async { Ok(None) })),
        ),
        "llm conditional execution guardrail:",
    );

    ctx.register_llm_execution_intercept(
        "llm-exec",
        1,
        Arc::new(|_name, request, _next| Box::pin(async move { Ok(request.content) })),
    )
    .unwrap();
    expect_registration_failed(
        ctx.register_llm_execution_intercept(
            "llm-exec",
            1,
            Arc::new(|_name, request, _next| Box::pin(async move { Ok(request.content) })),
        ),
        "llm execution intercept:",
    );

    ctx.register_llm_stream_execution_intercept(
        "llm-stream",
        1,
        Arc::new(|_name, request, _next| {
            Box::pin(async move {
                Ok(LlmJsonStream::new(tokio_stream::iter(vec![Ok(
                    request.content
                )])))
            })
        }),
    )
    .unwrap();
    expect_registration_failed(
        ctx.register_llm_stream_execution_intercept(
            "llm-stream",
            1,
            Arc::new(|_name, request, _next| {
                Box::pin(async move {
                    Ok(LlmJsonStream::new(tokio_stream::iter(vec![Ok(
                        request.content
                    )])))
                })
            }),
        ),
        "llm stream execution intercept:",
    );

    ctx.register_tool_request_intercept(
        "tool-request",
        1,
        false,
        Arc::new(|_name, args| Box::pin(async move { Ok(args) })),
    )
    .unwrap();
    expect_registration_failed(
        ctx.register_tool_request_intercept(
            "tool-request",
            1,
            false,
            Arc::new(|_name, args| Box::pin(async move { Ok(args) })),
        ),
        "tool request intercept:",
    );

    ctx.register_tool_execution_intercept(
        "tool-exec",
        1,
        Arc::new(|_name, args, _next| Box::pin(async move { Ok(args.into()) })),
    )
    .unwrap();
    expect_registration_failed(
        ctx.register_tool_execution_intercept(
            "tool-exec",
            1,
            Arc::new(|_name, args, _next| Box::pin(async move { Ok(args.into()) })),
        ),
        "tool execution intercept:",
    );

    let mut registrations = ctx.into_registrations();
    rollback_registrations(&mut registrations);
    reset_global();
}

#[test]
fn test_plugin_registration_context_maps_deregistration_errors() {
    let _guard = lock_runtime_owner();
    reset_global();

    let mut ctx = PluginRegistrationContext::with_namespace("teardown::");
    ctx.register_mark_sanitize_guardrail(
        "mark-sanitize",
        1,
        Arc::new(|_, fields| Box::pin(async move { Ok(fields) })),
    )
    .unwrap();
    ctx.register_scope_sanitize_start_guardrail(
        "scope-sanitize-start",
        1,
        Arc::new(|_, fields| Box::pin(async move { Ok(fields) })),
    )
    .unwrap();
    ctx.register_scope_sanitize_end_guardrail(
        "scope-sanitize-end",
        1,
        Arc::new(|_, fields| Box::pin(async move { Ok(fields) })),
    )
    .unwrap();
    ctx.register_subscriber("subscriber", Arc::new(|_event| {}))
        .unwrap();
    ctx.register_llm_request_intercept(
        "llm-request",
        1,
        false,
        Arc::new(|_name, request, annotated| {
            Box::pin(async move { Ok(LlmRequestInterceptOutcome::new(request, annotated)) })
        }),
    )
    .unwrap();
    ctx.register_tool_sanitize_request_guardrail(
        "tool-sanitize-request",
        1,
        Arc::new(|_, args| Box::pin(async move { Ok(args) })),
    )
    .unwrap();
    ctx.register_tool_sanitize_response_guardrail(
        "tool-sanitize-response",
        1,
        Arc::new(|_, response| Box::pin(async move { Ok(response) })),
    )
    .unwrap();
    ctx.register_tool_conditional_execution_guardrail(
        "tool-conditional",
        1,
        Arc::new(|_, _| Box::pin(async { Ok(None) })),
    )
    .unwrap();
    ctx.register_llm_sanitize_request_guardrail(
        "llm-sanitize-request",
        1,
        Arc::new(|request, _context| Box::pin(async move { Ok(Some(request)) })),
    )
    .unwrap();
    ctx.register_llm_sanitize_response_guardrail(
        "llm-sanitize-response",
        1,
        Arc::new(|response, _context| Box::pin(async move { Ok(Some(response)) })),
    )
    .unwrap();
    ctx.register_llm_conditional_execution_guardrail(
        "llm-conditional",
        1,
        Arc::new(|_| Box::pin(async { Ok(None) })),
    )
    .unwrap();
    ctx.register_llm_execution_intercept(
        "llm-exec",
        1,
        Arc::new(|_name, request, _next| Box::pin(async move { Ok(request.content) })),
    )
    .unwrap();
    ctx.register_llm_stream_execution_intercept(
        "llm-stream",
        1,
        Arc::new(|_name, request, _next| {
            Box::pin(async move {
                Ok(LlmJsonStream::new(tokio_stream::iter(vec![Ok(
                    request.content
                )])))
            })
        }),
    )
    .unwrap();
    ctx.register_tool_request_intercept(
        "tool-request",
        1,
        false,
        Arc::new(|_name, args| Box::pin(async move { Ok(args) })),
    )
    .unwrap();
    ctx.register_tool_execution_intercept(
        "tool-exec",
        1,
        Arc::new(|_name, args, _next| Box::pin(async move { Ok(args.into()) })),
    )
    .unwrap();

    let mut registrations = ctx.into_registrations();
    let expected_messages = [
        "mark sanitizer deregistration failed:",
        "scope-start sanitizer deregistration failed:",
        "scope-end sanitizer deregistration failed:",
        "subscriber deregistration failed:",
        "llm request intercept deregistration failed:",
        "tool sanitize request guardrail deregistration failed:",
        "tool sanitize response guardrail deregistration failed:",
        "tool conditional execution guardrail deregistration failed:",
        "llm sanitize request guardrail deregistration failed:",
        "llm sanitize response guardrail deregistration failed:",
        "llm conditional execution guardrail deregistration failed:",
        "llm execution intercept deregistration failed:",
        "llm stream execution intercept deregistration failed:",
        "tool request intercept deregistration failed:",
        "tool execution intercept deregistration failed:",
    ];

    set_conflicting_runtime_owner_for_tests();
    for (registration, expected) in registrations.iter_mut().zip(expected_messages) {
        match (registration.deregister)() {
            PluginRegistrationCleanupOutcome::NotRemoved(PluginError::RegistrationFailed(
                message,
            )) => {
                assert!(message.contains(expected), "{message}");
            }
            PluginRegistrationCleanupOutcome::NotRemoved(other) => {
                panic!("unexpected deregistration failure: {other}")
            }
            PluginRegistrationCleanupOutcome::Removed
            | PluginRegistrationCleanupOutcome::RemovedWithError(_) => {
                panic!("expected deregistration to fail")
            }
        }
    }

    reset_global();
}

#[test]
fn test_initialize_plugins_replaces_previous_configuration_on_success() {
    let _guard = lock_runtime_owner();
    reset_global();
    register_plugin(Arc::new(RecordingPlugin)).unwrap();
    register_plugin(Arc::new(ReplacementPlugin)).unwrap();

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    runtime
        .block_on(initialize_plugins_exact(PluginConfig {
            components: vec![PluginComponentSpec::new("recording.plugin")],
            ..PluginConfig::default()
        }))
        .unwrap();

    let report = runtime
        .block_on(initialize_plugins_exact(PluginConfig {
            components: vec![PluginComponentSpec::new("replacement.plugin")],
            ..PluginConfig::default()
        }))
        .unwrap();

    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.code == "replacement.warning")
    );
    assert_eq!(active_plugin_report().unwrap().diagnostics.len(), 1);
    assert_eq!(REPLACEMENT_REGISTRATIONS.load(Ordering::SeqCst), 1);

    reset_global();
}

#[test]
fn test_initialize_plugins_preserves_resolution_diagnostics() {
    let _guard = lock_runtime_owner();
    reset_global();
    register_plugin(Arc::new(RecordingPlugin)).unwrap();

    let diagnostic = ConfigDiagnostic {
        level: DiagnosticLevel::Warning,
        code: "plugin.component_reenabled".to_string(),
        component: Some("recording.plugin".to_string()),
        field: Some("enabled".to_string()),
        message: "programmatic configuration re-enabled the component".to_string(),
    };
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let report = runtime
        .block_on(initialize_plugins_with_diagnostics(
            PluginConfig {
                components: vec![PluginComponentSpec::new("recording.plugin")],
                ..PluginConfig::default()
            },
            vec![diagnostic.clone()],
        ))
        .unwrap();

    assert_eq!(report.diagnostics, vec![diagnostic.clone()]);
    assert_eq!(
        active_plugin_report().unwrap().diagnostics,
        vec![diagnostic]
    );
    reset_global();
}

#[test]
fn test_initialize_plugins_reports_failed_restore_when_previous_configuration_cannot_be_restored() {
    let _guard = lock_runtime_owner();
    reset_global();
    register_plugin(Arc::new(RestoreBreakPlugin)).unwrap();
    register_plugin(Arc::new(RestoreFailPlugin)).unwrap();

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    runtime
        .block_on(initialize_plugins_exact(PluginConfig {
            components: vec![PluginComponentSpec::new("restore.break.plugin")],
            ..PluginConfig::default()
        }))
        .unwrap();

    let error = runtime
        .block_on(initialize_plugins_exact(PluginConfig {
            components: vec![PluginComponentSpec::new("restore.fail.plugin")],
            ..PluginConfig::default()
        }))
        .unwrap_err();

    match error {
        PluginError::RegistrationFailed(message) => {
            assert!(message.contains("restore.fail.plugin refused to initialize"));
            assert!(message.contains("previous plugin configuration could not be restored"));
            assert!(message.contains("restore.break.plugin refused to restore"));
        }
        other => panic!("unexpected failed-restore error: {other}"),
    }

    assert!(active_plugin_report().is_none());
    reset_global();
}

#[test]
fn test_load_plugin_config_files_merges_files_by_precedence() {
    let dir = tempfile::tempdir().unwrap();
    let lower = dir.path().join("lower.toml");
    let project = dir.path().join("project.toml");
    let system = dir.path().join("system.toml");
    std::fs::write(
        &lower,
        "version = 1\n\
         [[components]]\n\
         kind = \"observability\"\n\
         enabled = true\n\
         [components.config]\n\
         output_directory = \"/var/log\"\n\
         mode = \"append\"\n\
         values = [\"lower\"]\n",
    )
    .unwrap();
    std::fs::write(
        &project,
        "[[components]]\n\
         kind = \"observability\"\n\
         [components.config]\n\
         mode = \"project\"\n\
         values = [\"project\"]\n\
         [[components]]\n\
         kind = \"adaptive\"\n",
    )
    .unwrap();
    std::fs::write(
        &system,
        "[[components]]\n\
         kind = \"observability\"\n\
         enabled = false\n\
         [components.config]\n\
         mode = \"system\"\n\
         values = [\"system\"]\n\
         [[components]]\n\
         kind = \"system_only\"\n",
    )
    .unwrap();

    let (merged, sources) =
        load_plugin_config_files([lower.clone(), project.clone(), system.clone()])
            .unwrap()
            .expect("a file exists");
    assert_eq!(sources, vec![lower, project, system]);

    let components = merged["components"].as_array().unwrap();
    let observability = &components[0];
    assert_eq!(observability["kind"], json!("observability"));
    assert_eq!(
        observability["enabled"],
        json!(true),
        "a disabled system component does not override lower layers"
    );
    assert_eq!(
        observability["config"]["output_directory"],
        json!("/var/log"),
        "lower-only config key is inherited"
    );
    assert_eq!(
        observability["config"]["mode"],
        json!("project"),
        "a disabled system component does not contribute configuration"
    );
    assert_eq!(
        observability["config"]["values"],
        json!(["project", "lower"]),
        "a disabled system component does not contribute list entries"
    );
    assert_eq!(
        components[1]["kind"],
        json!("adaptive"),
        "a project-only component kind is preserved"
    );
    assert_eq!(
        components[2]["kind"],
        json!("system_only"),
        "a system-only component kind is appended"
    );
}

#[test]
fn test_load_plugin_config_files_omits_components_disabled_in_every_file() {
    let dir = tempfile::tempdir().unwrap();
    let lower = dir.path().join("lower.toml");
    let higher = dir.path().join("higher.toml");
    std::fs::write(
        &lower,
        "[[components]]\n\
         kind = \"observability\"\n\
         enabled = false\n",
    )
    .unwrap();
    std::fs::write(
        &higher,
        "[[components]]\n\
         kind = \"observability\"\n\
         enabled = false\n",
    )
    .unwrap();

    let (merged, sources) = load_plugin_config_files([lower.clone(), higher.clone()])
        .unwrap()
        .expect("the files exist");

    assert_eq!(sources, vec![lower, higher]);
    assert_eq!(merged["components"], json!([]));
}

#[test]
fn test_load_plugin_config_files_rejects_version_before_layering() {
    let dir = tempfile::tempdir().unwrap();
    let invalid = dir.path().join("invalid.toml");
    let higher = dir.path().join("higher.toml");
    std::fs::write(
        &invalid,
        "version = 2\n\
         [[components]]\n\
         kind = \"observability\"\n",
    )
    .unwrap();
    std::fs::write(&higher, "version = 1\n").unwrap();

    let error = load_plugin_config_files([invalid.clone(), higher])
        .expect_err("a higher-precedence version must not mask an invalid source version");

    match error {
        PluginError::InvalidConfig(message) => {
            assert!(message.contains("plugin config version 2"), "{message}");
            assert!(
                message.contains(&invalid.display().to_string()),
                "{message}"
            );
            assert!(message.contains("expected 1"), "{message}");
        }
        other => panic!("unexpected plugin config version error: {other}"),
    }
}

#[test]
fn test_plugin_config_loading_reports_read_parse_and_version_type_errors() {
    let dir = tempfile::tempdir().unwrap();
    let unreadable = dir.path().join("directory.toml");
    std::fs::create_dir(&unreadable).unwrap();
    assert!(
        load_plugin_config_files([unreadable])
            .unwrap_err()
            .to_string()
            .contains("failed to read")
    );

    let malformed = dir.path().join("malformed.toml");
    std::fs::write(&malformed, "version = [").unwrap();
    assert!(
        load_plugin_config_files([malformed])
            .unwrap_err()
            .to_string()
            .contains("invalid plugin TOML")
    );

    assert!(
        merge_plugin_config_documents([
            (dir.path().join("typed.toml"), json!({"version": "one"}),)
        ])
        .unwrap_err()
        .to_string()
        .contains("invalid plugin config version")
    );
}

#[test]
fn test_default_plugin_config_paths_order_user_system() {
    #[cfg(feature = "__skip-implicit-config")]
    let _environment = TestConfigEnvironment::set(None);

    let dir = tempfile::tempdir().unwrap();
    let user = dir.path().join("user");

    assert_eq!(
        default_plugin_config_paths(Some(user.clone())),
        vec![
            user.join("plugins.toml"),
            system_config_dir().join("plugins.toml"),
        ]
    );
}

#[cfg(feature = "__skip-implicit-config")]
#[test]
fn test_hook_skips_implicit_plugin_config_paths() {
    let _environment = TestConfigEnvironment::set(Some(OsStr::new("1")));

    assert!(default_plugin_config_paths(Some(PathBuf::from("/test/user"))).is_empty());
}

#[test]
fn test_system_config_dir_matches_platform_convention() {
    #[cfg(windows)]
    {
        let expected_base = std::env::var_os("ProgramData")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(r"C:\ProgramData"));
        assert_eq!(system_config_dir(), expected_base.join("nemo-relay"));
    }
    #[cfg(not(windows))]
    assert_eq!(system_config_dir(), PathBuf::from("/etc/nemo-relay"));
}

#[cfg(unix)]
#[test]
fn test_load_plugin_config_files_deduplicates_aliases_at_highest_precedence() {
    use std::os::unix::fs::symlink;

    let dir = tempfile::tempdir().unwrap();
    let physical = dir.path().join("system.toml");
    let alias = dir.path().join("explicit.toml");
    std::fs::write(
        &physical,
        "version = 1\n\
         [[components]]\n\
         kind = \"pricing\"\n\
         [[components.config.sources]]\n\
         type = \"file\"\n\
         path = \"/etc/nemo-relay/pricing.json\"\n",
    )
    .unwrap();
    symlink(&physical, &alias).unwrap();

    let (merged, sources) = load_plugin_config_files([alias, physical.clone()])
        .unwrap()
        .expect("the physical file exists");

    assert_eq!(sources, vec![physical]);
    assert_eq!(
        merged["components"][0]["config"]["sources"]
            .as_array()
            .unwrap()
            .len(),
        1,
        "the aliased file must not duplicate list entries"
    );
}

#[test]
fn test_plugin_config_overlay_enables_programmatically_declared_components() {
    // After each file's schema version is validated, a typed `PluginConfig` is layered over
    // the discovered file base. Default-valued policy fields inherit the file, a declared
    // component applies its enabled value, and the free-form `config` body merges.
    let file_base = json!({
        "version": 1,
        "components": [
            {
                "kind": "observability",
                "enabled": false,
                "config": { "output_directory": "/var/log", "mode": "append" }
            },
            { "kind": "adaptive", "enabled": false, "config": { "ttl": 60 } }
        ],
        "policy": {
            "unknown_component": "error",
            "unknown_field": "warn",
            "unsupported_value": "error"
        }
    });
    let code = PluginConfig {
        components: vec![PluginComponentSpec {
            config: Map::from_iter([(String::from("mode"), json!("overwrite"))]),
            ..PluginComponentSpec::new("observability")
        }],
        ..PluginConfig::default()
    };

    let mut merged = file_base;
    layer_config(&mut merged, plugin_config_overlay_value(&code).unwrap());
    let typed: PluginConfig = serde_json::from_value(merged).unwrap();

    // Typed policy defaults do not mask the file base.
    assert_eq!(typed.version, 1);
    assert_eq!(
        typed.policy.unknown_component,
        UnsupportedBehavior::Error,
        "typed default policy inherits the file value"
    );
    let observability = &typed.components[0];
    assert_eq!(observability.kind, "observability");
    assert!(
        observability.enabled,
        "a declared component is enabled by code"
    );
    // The component config body merges: code's `mode` wins, the file's `output_directory`
    // is inherited.
    assert_eq!(observability.config["mode"], json!("overwrite"));
    assert_eq!(observability.config["output_directory"], json!("/var/log"));
    // A kind the code config does not declare is inherited from the file.
    assert_eq!(typed.components[1].kind, "adaptive");
    assert!(!typed.components[1].enabled);
}

#[test]
fn test_programmatic_observability_destinations_concatenate_with_discovered_files_and_warn() {
    let lower = PathBuf::from("lower/plugins.toml");
    let higher = PathBuf::from("higher/plugins.toml");
    let discovered = resolve_discovered_plugin_config(vec![
        (
            lower.clone(),
            observability_destination_document("lower-secret-destination"),
        ),
        (
            higher.clone(),
            observability_destination_document("higher-secret-destination"),
        ),
    ])
    .unwrap();
    let programmatic = programmatic_observability_config(json!({
        "atof": {"enabled": true, "sinks": ["programmatic"]},
        "opentelemetry": {"enabled": true, "endpoints": ["programmatic"]},
        "atif": {"enabled": true, "storage": ["programmatic"]}
    }));

    let resolved = resolve_programmatic_plugin_config(discovered, programmatic).unwrap();
    let config = &resolved.config.components[0].config;
    for (section, field) in [
        ("atof", "sinks"),
        ("opentelemetry", "endpoints"),
        ("atif", "storage"),
    ] {
        assert_eq!(
            config[section][field],
            json!([
                "programmatic",
                "higher-secret-destination",
                "lower-secret-destination"
            ])
        );
    }
    assert_eq!(resolved.diagnostics.len(), 2);
    for (diagnostic, source) in resolved.diagnostics.iter().zip([lower, higher]) {
        assert_eq!(diagnostic.level, DiagnosticLevel::Warning);
        assert_eq!(diagnostic.code, "plugin.configuration_inherited");
        assert!(diagnostic.component.is_none());
        assert!(diagnostic.field.is_none());
        assert!(diagnostic.message.contains(&source.display().to_string()));
        assert!(!diagnostic.message.contains("secret-destination"));
    }
}

#[test]
fn test_no_inherited_configuration_warning_without_discovered_files() {
    let discovered = resolve_discovered_plugin_config(Vec::new()).unwrap();
    let resolved = resolve_programmatic_plugin_config(discovered, PluginConfig::default()).unwrap();
    assert!(resolved.diagnostics.is_empty());
}

#[test]
fn test_programmatic_enable_override_diagnostic_matches_positionally_and_names_source() {
    let source = PathBuf::from("/etc/nemo-relay/plugins.toml");
    let discovered = json!({
        "components": [
            { "kind": "observability", "enabled": true },
            { "kind": "observability", "enabled": false }
        ]
    });
    let enabled_sources = HashMap::from([(
        "observability".to_string(),
        ComponentEnabledSource {
            enabled: false,
            path: source.clone(),
        },
    )]);
    let programmatic = PluginConfig {
        components: vec![
            PluginComponentSpec::new("observability"),
            PluginComponentSpec::new("observability"),
        ],
        ..PluginConfig::default()
    };

    let diagnostics =
        programmatic_enable_override_diagnostics(&discovered, &enabled_sources, &programmatic);

    assert_eq!(diagnostics.len(), 1);
    let diagnostic = &diagnostics[0];
    assert_eq!(diagnostic.level, DiagnosticLevel::Warning);
    assert_eq!(diagnostic.code, "plugin.component_reenabled");
    assert_eq!(diagnostic.component.as_deref(), Some("observability"));
    assert_eq!(diagnostic.field.as_deref(), Some("enabled"));
    assert!(
        diagnostic.message.contains(&source.display().to_string()),
        "{}",
        diagnostic.message
    );
}

#[test]
fn test_programmatic_reenable_diagnostic_survives_disabled_component_normalization() {
    let source = PathBuf::from("/etc/nemo-relay/plugins.toml");
    let documents = vec![(
        source.clone(),
        json!({
            "components": [{ "kind": "observability", "enabled": false }]
        }),
    )];
    let enabled_sources = component_enabled_sources(&documents);
    let (discovered, _) = merge_plugin_config_documents(documents)
        .unwrap()
        .expect("the file-backed configuration exists");
    assert_eq!(discovered["components"], json!([]));

    let diagnostics = programmatic_enable_override_diagnostics(
        &discovered,
        &enabled_sources,
        &PluginConfig {
            components: vec![PluginComponentSpec::new("observability")],
            ..PluginConfig::default()
        },
    );

    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0].code, "plugin.component_reenabled");
    assert!(
        diagnostics[0]
            .message
            .contains(&source.display().to_string())
    );
}

#[test]
fn test_plugin_config_overlay_applies_non_default_values() {
    let mut file_base = json!({
        "version": 1,
        "components": [{ "kind": "observability", "enabled": true }],
        "policy": {
            "unknown_component": "error",
            "unknown_field": "warn",
            "unsupported_value": "error"
        }
    });
    let code = PluginConfig {
        components: vec![PluginComponentSpec {
            enabled: false,
            ..PluginComponentSpec::new("observability")
        }],
        policy: ConfigPolicy {
            unknown_field: UnsupportedBehavior::Ignore,
            ..ConfigPolicy::default()
        },
        ..PluginConfig::default()
    };

    layer_config(&mut file_base, plugin_config_overlay_value(&code).unwrap());
    let typed: PluginConfig = serde_json::from_value(file_base).unwrap();

    assert!(!typed.components[0].enabled);
    assert_eq!(
        typed.policy.unknown_component,
        UnsupportedBehavior::Error,
        "a default-valued field inherits the file"
    );
    assert_eq!(
        typed.policy.unknown_field,
        UnsupportedBehavior::Ignore,
        "a non-default field overrides the file"
    );
}
