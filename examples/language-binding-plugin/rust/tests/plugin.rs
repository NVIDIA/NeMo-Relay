// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use futures::{StreamExt, stream};
use nemo_relay::api::llm::{
    LlmCallExecuteParams, LlmRequest, LlmStreamCallExecuteParams, llm_call_execute,
    llm_stream_call_execute,
};
use nemo_relay::api::runtime::callbacks::LlmJsonStream;
use nemo_relay::api::subscriber::flush_subscribers;
use nemo_relay::api::tool::{ToolCallExecuteParams, tool_call_execute};
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

struct ActivePlugin {
    report: ConfigReport,
    _lock: MutexGuard<'static, ()>,
}

impl Drop for ActivePlugin {
    fn drop(&mut self) {
        let _ = clear_plugin_configuration();
        deregister_plugin("documentation-plugin");
    }
}

async fn activate() -> ActivePlugin {
    let lock = PLUGIN_TEST_LOCK.lock().await;
    reset_observed_events();
    register_plugin(Arc::new(DocumentationPlugin)).expect("plugin registration should succeed");
    let report = match initialize_plugins_exact(config("enforce")).await {
        Ok(report) => report,
        Err(error) => {
            deregister_plugin("documentation-plugin");
            panic!("plugin activation should succeed: {error}");
        }
    };
    ActivePlugin {
        report,
        _lock: lock,
    }
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
    for (configuration, field) in [
        (json!({ "tag": "" }), "tag"),
        (json!({ "requests": { "header_name": "" } }), "requests.header_name"),
        (
            json!({ "requests": { "header_value": "" } }),
            "requests.header_value",
        ),
    ] {
        let diagnostics = DocumentationPlugin.validate(&configuration.as_object().unwrap().clone());
        assert!(diagnostics.iter().any(|diagnostic| {
            diagnostic.code == "documentation-plugin.invalid_header"
                || (field == "tag" && diagnostic.code == "documentation-plugin.invalid_tag")
        }));
        assert!(diagnostics.iter().any(|diagnostic| diagnostic.field.as_deref() == Some(field)));
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

#[tokio::test]
async fn registration_rejects_a_duplicate_kind_and_missing_deregistration_is_false() {
    let _lock = PLUGIN_TEST_LOCK.lock().await;
    register_plugin(Arc::new(DocumentationPlugin)).expect("first registration should succeed");
    assert!(register_plugin(Arc::new(DocumentationPlugin)).is_err());
    assert!(!deregister_plugin("missing-documentation-plugin"));
    assert!(deregister_plugin("documentation-plugin"));
}

#[tokio::test]
async fn disabled_component_is_still_validated() {
    let _lock = PLUGIN_TEST_LOCK.lock().await;
    register_plugin(Arc::new(DocumentationPlugin)).expect("plugin registration should succeed");
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
async fn tool_request_is_rewritten() {
    let _active = activate().await;

    let result = tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("safe_tool")
            .args(json!({"value": 1}))
            .func(Arc::new(|args| Box::pin(async move { Ok(args) })))
            .build(),
    )
    .await
    .expect("tool call should succeed");

    assert_eq!(result, json!({"value": 1, "plugin_tag": "documentation"}));
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
            .func(Arc::new(|args| Box::pin(async move { Ok(args) })))
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
            .func(Arc::new(|args| Box::pin(async move { Ok(args) })))
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
    assert_eq!(runtime_mark.data().expect("mark data")["secret"], "[REDACTED]");
    assert!(events
        .iter()
        .any(|event| event.name() == "documentation-plugin.tool-complete"));
    assert!(events
        .iter()
        .any(|event| event.name() == "documentation-plugin.isolated"));
}

#[tokio::test]
async fn teardown_removes_plugin_kind() {
    let _lock = PLUGIN_TEST_LOCK.lock().await;
    register_plugin(Arc::new(DocumentationPlugin)).expect("plugin registration should succeed");
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
