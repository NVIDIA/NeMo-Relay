// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};

use futures::{StreamExt, stream};
use nemo_relay::api::llm::{
    LlmCallExecuteParams, LlmRequest, LlmStreamCallExecuteParams, llm_call_execute,
    llm_stream_call_execute,
};
use nemo_relay::api::runtime::callbacks::LlmJsonStream;
use nemo_relay::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use nemo_relay::api::tool::{ToolCallExecuteParams, ToolExecutionResult, tool_call_execute};
use nemo_relay::plugin::{
    ConfigReport, DiagnosticLevel, Plugin, clear_plugin_configuration, deregister_plugin,
    initialize_plugins_exact, list_plugin_kinds, register_plugin, validate_plugin_config,
};
use nemo_relay_language_binding_plugin_example::{
    DocumentationPlugin, config, config_with_enabled, observed_event_count, observed_events,
    reset_observed_events,
};
use serde_json::{Map, json};
use tokio::sync::{Mutex, MutexGuard};

static PLUGIN_TEST_LOCK: Mutex<()> = Mutex::const_new(());

struct RegisteredPlugin {
    _lock: MutexGuard<'static, ()>,
}

impl Drop for RegisteredPlugin {
    fn drop(&mut self) {
        let _ = clear_plugin_configuration();
        deregister_plugin("documentation-plugin");
    }
}

struct ActivePlugin {
    report: ConfigReport,
    _registration: RegisteredPlugin,
}

struct RegisteredSubscriber(&'static str);

impl Drop for RegisteredSubscriber {
    fn drop(&mut self) {
        let _ = deregister_subscriber(self.0);
    }
}

async fn register_only() -> RegisteredPlugin {
    let lock = PLUGIN_TEST_LOCK.lock().await;
    reset_observed_events();
    register_plugin(Arc::new(DocumentationPlugin)).expect("plugin registration should succeed");
    RegisteredPlugin { _lock: lock }
}

async fn activate_with(plugin_config: nemo_relay::plugin::PluginConfig) -> ActivePlugin {
    let registration = register_only().await;
    activate_registered(registration, plugin_config).await
}

async fn activate_registered(
    registration: RegisteredPlugin,
    plugin_config: nemo_relay::plugin::PluginConfig,
) -> ActivePlugin {
    let report = initialize_plugins_exact(plugin_config)
        .await
        .unwrap_or_else(|error| panic!("plugin activation should succeed: {error}"));
    ActivePlugin {
        report,
        _registration: registration,
    }
}

async fn activate() -> ActivePlugin {
    activate_with(config("enforce")).await
}

#[test]
fn validation_accepts_supported_mode() {
    let configuration = config("enforce");
    let diagnostics = DocumentationPlugin.validate(&configuration.components[0].config);

    assert!(diagnostics.is_empty());
}

#[test]
fn validation_rejects_unsupported_mode() {
    let configuration = config("invalid");
    let diagnostics = DocumentationPlugin.validate(&configuration.components[0].config);

    assert_eq!(diagnostics[0].code, "documentation-plugin.unsupported_mode");
}

#[test]
fn validation_rejects_wrong_type() {
    let mut configuration = config("enforce");
    configuration.components[0]
        .config
        .insert("requests".into(), json!({"priority": "high"}));

    let diagnostics = DocumentationPlugin.validate(&configuration.components[0].config);

    assert_eq!(diagnostics[0].code, "documentation-plugin.invalid_config");
}

#[test]
fn validation_reports_each_empty_required_string_at_its_field() {
    for (configuration, code, field) in [
        (
            json!({ "tag": "" }),
            "documentation-plugin.invalid_tag",
            "tag",
        ),
        (
            json!({ "requests": { "header_name": "" } }),
            "documentation-plugin.invalid_header",
            "requests.header_name",
        ),
        (
            json!({ "requests": { "header_value": "" } }),
            "documentation-plugin.invalid_header",
            "requests.header_value",
        ),
        (
            json!({ "registration_control": { "kinds": [] } }),
            "documentation-plugin.invalid_registration_control",
            "registration_control.kinds",
        ),
        (
            json!({ "registration_control": { "registration_name": "" } }),
            "documentation-plugin.invalid_registration_control",
            "registration_control.registration_name",
        ),
        (
            json!({ "registration_control": { "reason": "" } }),
            "documentation-plugin.invalid_registration_control",
            "registration_control.reason",
        ),
    ] {
        let diagnostics = DocumentationPlugin.validate(&configuration.as_object().unwrap().clone());
        assert!(diagnostics.iter().any(|diagnostic| {
            diagnostic.code == code && diagnostic.field.as_deref() == Some(field)
        }));
    }
}

#[test]
fn validation_warns_about_unknown_field() {
    let mut configuration = config("enforce");
    configuration.components[0]
        .config
        .insert("unexpected".into(), json!(true));

    let diagnostics = DocumentationPlugin.validate(&configuration.components[0].config);

    assert_eq!(diagnostics[0].level, DiagnosticLevel::Warning);
    assert_eq!(diagnostics[0].field.as_deref(), Some("unexpected"));
}

#[test]
fn implementation_registers_each_safe_plugin_surface() {
    let source = include_str!("../src/lib.rs");
    assert_eq!(
        [
            "register_subscriber",
            "register_mark_sanitize_guardrail",
            "register_scope_sanitize_start_guardrail",
            "register_scope_sanitize_end_guardrail",
            "register_tool_sanitize_request_guardrail",
            "register_tool_sanitize_response_guardrail",
            "register_tool_conditional_execution_guardrail",
            "register_tool_request_intercept",
            "register_tool_execution_intercept",
            "register_llm_sanitize_request_guardrail",
            "register_llm_sanitize_response_guardrail",
            "register_llm_conditional_execution_guardrail",
            "register_llm_request_intercept",
            "register_llm_execution_intercept",
            "register_llm_stream_execution_intercept",
            "register_conditional_middleware_guardrail",
        ]
        .iter()
        .filter(|method| source.contains(**method))
        .count(),
        16
    );
}

#[tokio::test]
async fn registration_rejects_a_duplicate_kind_and_missing_deregistration_is_false() {
    let _registered = register_only().await;
    assert!(register_plugin(Arc::new(DocumentationPlugin)).is_err());
    assert!(!deregister_plugin("missing-documentation-plugin"));
    assert!(deregister_plugin("documentation-plugin"));
}

#[tokio::test]
async fn disabled_component_is_still_validated() {
    let _registered = register_only().await;
    let report = validate_plugin_config(&config_with_enabled("invalid", false));
    assert!(deregister_plugin("documentation-plugin"));

    assert_eq!(
        report.diagnostics[0].code,
        "documentation-plugin.unsupported_mode"
    );
}

#[tokio::test]
async fn activation_reports_no_diagnostics() {
    let active = activate().await;

    assert!(active.report.diagnostics.is_empty());
}

#[tokio::test]
async fn registration_control_is_owned_by_activation() {
    const TARGET: &str = "documentation-controlled-subscriber";
    let registration = register_only().await;
    let observed = Arc::new(AtomicUsize::new(0));
    let captured = Arc::clone(&observed);
    register_subscriber(
        TARGET,
        Arc::new(move |_| {
            captured.fetch_add(1, Ordering::SeqCst);
        }),
    )
    .expect("controlled subscriber should register");
    let _subscriber = RegisteredSubscriber(TARGET);
    let mut configuration = config("enforce");
    configuration.components[0].config["registration_control"]["enabled"] = json!(true);

    let active = activate_registered(registration, configuration).await;
    flush_subscribers().expect("activation events should flush");
    let baseline = observed.load(Ordering::SeqCst);
    tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("controlled-tool")
            .args(json!({}))
            .func(Arc::new(|args| Box::pin(async move { Ok(ToolExecutionResult::new(args)) })))
            .build(),
    )
    .await
    .expect("managed call should complete");
    flush_subscribers().expect("active events should flush");
    assert_eq!(observed.load(Ordering::SeqCst), baseline);

    drop(active);
    tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("restored-tool")
            .args(json!({}))
            .func(Arc::new(|args| Box::pin(async move { Ok(ToolExecutionResult::new(args)) })))
            .build(),
    )
    .await
    .expect("managed call should complete after clear");
    flush_subscribers().expect("restored events should flush");
    assert!(observed.load(Ordering::SeqCst) > baseline);
}

#[tokio::test]
async fn tool_request_is_rewritten() {
    let _active = activate().await;

    let result = tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("safe_tool")
            .args(json!({"value": 1}))
            .func(Arc::new(|args| {
                Box::pin(async move {
                    Ok(ToolExecutionResult::annotated(args, json!({"source": "application"})))
                })
            }))
            .build(),
    )
    .await
    .expect("tool call should succeed");

    assert_eq!(result.result, json!({"value": 1, "plugin_tag": "documentation"}));
    assert_eq!(result.annotation, Some(json!({"source": "application"})));
}

#[tokio::test]
async fn tool_policy_blocks_configured_tool() {
    let _active = activate().await;

    let error = tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("dangerous_tool")
            .args(json!({"value": 1}))
            .func(Arc::new(|_args| {
                Box::pin(async move { panic!("provider must not run") })
            }))
            .build(),
    )
    .await
    .expect_err("configured tool should be blocked");

    assert!(error.to_string().contains("guardrail rejected"));
}

#[tokio::test]
async fn llm_request_is_rewritten() {
    let _active = activate().await;
    let request = LlmRequest {
        headers: Map::new(),
        content: json!({"model": "allowed-model"}),
    };

    let result = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("allowed-model")
            .request(request)
            .func(Arc::new(|request| {
                Box::pin(async move { Ok(json!({"headers": request.headers})) })
            }))
            .build(),
    )
    .await
    .expect("LLM call should succeed");

    assert_eq!(result["headers"]["x-nemo-relay-plugin"], "documentation");
}

#[tokio::test]
async fn llm_policy_blocks_configured_model() {
    let _active = activate().await;
    let request = LlmRequest {
        headers: Map::new(),
        content: json!({"model": "restricted-model"}),
    };

    let error = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("restricted-model")
            .request(request)
            .func(Arc::new(|_request| {
                Box::pin(async move { panic!("provider must not run") })
            }))
            .build(),
    )
    .await
    .expect_err("configured model should be blocked");

    assert!(error.to_string().contains("guardrail rejected"));
}

#[tokio::test]
async fn llm_stream_chunks_are_transformed() {
    let _active = activate().await;
    let request = LlmRequest {
        headers: Map::new(),
        content: json!({"model": "allowed-model"}),
    };

    let mut output = llm_stream_call_execute(
        LlmStreamCallExecuteParams::builder()
            .name("allowed-model")
            .request(request)
            .func(Arc::new(|_request| {
                Box::pin(async {
                    Ok(LlmJsonStream::new(stream::iter(vec![
                        Ok(json!({"chunk": 1})),
                        Ok(json!({"chunk": 2})),
                    ])))
                })
            }))
            .collector(Box::new(|_| Ok(())))
            .finalizer(Box::new(|| json!({"done": true})))
            .build(),
    )
    .await
    .expect("stream setup should succeed");
    let mut chunks = Vec::new();
    while let Some(chunk) = output.next().await {
        chunks.push(chunk.expect("stream chunk should succeed"));
    }

    assert_eq!(
        chunks,
        vec![
            json!({"chunk": 1, "plugin_stream": true}),
            json!({"chunk": 2, "plugin_stream": true}),
        ]
    );
}

#[tokio::test]
async fn subscriber_observes_managed_call() {
    let _active = activate().await;

    tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("safe_tool")
            .args(json!({"value": 1}))
            .func(Arc::new(|args| Box::pin(async move { Ok(ToolExecutionResult::new(args)) })))
            .build(),
    )
    .await
    .expect("tool call should succeed");
    flush_subscribers().expect("subscriber flush should succeed");

    assert!(observed_event_count() > 0);
}

#[tokio::test]
async fn configuration_controls_redaction_pending_marks_and_isolated_scope_events() {
    let _active = activate().await;

    tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("safe_tool")
            .args(json!({"value": 1}))
            .func(Arc::new(|args| Box::pin(async move { Ok(ToolExecutionResult::new(args)) })))
            .build(),
    )
    .await
    .expect("tool call should succeed");
    flush_subscribers().expect("subscriber flush should succeed");

    let events = observed_events();
    let runtime_mark = events
        .iter()
        .find(|event| event.name() == "documentation-plugin.request")
        .expect("configured runtime mark should be delivered");
    assert_eq!(
        runtime_mark.data().expect("mark data")["secret"],
        "[REDACTED]"
    );
    assert!(
        events
            .iter()
            .any(|event| event.name() == "documentation-plugin.tool-complete")
    );
    assert!(
        events
            .iter()
            .any(|event| event.name() == "documentation-plugin.isolated")
    );
}

#[tokio::test]
async fn runtime_events_do_not_depend_on_request_rewriting() {
    let mut plugin_config = config("enforce");
    plugin_config.components[0].config["requests"]["enabled"] = json!(false);
    let _active = activate_with(plugin_config).await;

    tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("safe_tool")
            .args(json!({"value": 1}))
            .func(Arc::new(|args| Box::pin(async move { Ok(ToolExecutionResult::new(args)) })))
            .build(),
    )
    .await
    .expect("tool call should succeed");
    flush_subscribers().expect("subscriber flush should succeed");

    assert!(
        observed_events()
            .iter()
            .any(|event| event.name() == "documentation-plugin.request")
    );
}

#[tokio::test]
async fn runtime_events_are_not_stopped_by_request_break_chain() {
    let mut plugin_config = config("enforce");
    plugin_config.components[0].config["requests"]["break_chain"] = json!(true);
    let _active = activate_with(plugin_config).await;

    tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("safe_tool")
            .args(json!({"value": 1}))
            .func(Arc::new(|args| Box::pin(async move { Ok(ToolExecutionResult::new(args)) })))
            .build(),
    )
    .await
    .expect("tool call should succeed");
    flush_subscribers().expect("subscriber flush should succeed");

    assert!(
        observed_events()
            .iter()
            .any(|event| event.name() == "documentation-plugin.request")
    );
}

#[tokio::test]
async fn teardown_removes_plugin_kind() {
    let _registered = register_only().await;
    initialize_plugins_exact(config("enforce"))
        .await
        .expect("plugin activation should succeed");

    clear_plugin_configuration().expect("plugin cleanup should succeed");
    assert!(deregister_plugin("documentation-plugin"));
    assert!(
        !list_plugin_kinds()
            .iter()
            .any(|kind| kind == "documentation-plugin")
    );
}
