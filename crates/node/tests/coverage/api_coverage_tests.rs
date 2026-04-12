// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use nemo_flow::context::global::global_context;
use nemo_flow::context::state::NemoFlowContextState;
use serde_json::json;
use std::sync::{Arc, Mutex, OnceLock};
use uuid::Uuid;

static TEST_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();

fn test_mutex() -> &'static Mutex<()> {
    TEST_MUTEX.get_or_init(|| Mutex::new(()))
}

fn reset_global() {
    let context = global_context();
    *context.write().unwrap() = NemoFlowContextState::new();
}

fn make_request() -> Json {
    json!({
        "headers": {"x-trace": "1"},
        "content": {"messages": [], "model": "test-model"},
    })
}

#[test]
fn test_ensure_stream_callback_queued_keeps_channel_on_success() {
    let id = 42;
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    register_stream_channel(id, tx);

    let result = ensure_stream_callback_queued(id, napi::Status::Ok);

    assert!(result.is_ok());
    assert!(push_stream_chunk(id as f64, json!({"ok": true})));
    remove_stream_channel(id);
}

#[test]
fn test_ensure_stream_callback_queued_cleans_up_channel_on_failure() {
    let id = 43;
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    register_stream_channel(id, tx);

    let result = ensure_stream_callback_queued(id, napi::Status::Closing);

    assert!(matches!(
        result,
        Err(nemo_flow::error::FlowError::Internal(_))
    ));
    assert!(!push_stream_chunk(id as f64, json!({"ok": false})));
}

#[test]
fn test_end_stream_removes_registered_channel() {
    let id = 44;
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    register_stream_channel(id, tx);

    assert!(push_stream_chunk(id as f64, json!({"before": true})));
    end_stream(id as f64);

    assert!(!push_stream_chunk(id as f64, json!({"after": true})));
}

#[test]
fn test_forward_stream_to_channel_exits_when_receiver_is_dropped() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        let stream: RustJsonStream = Box::pin(tokio_stream::iter(vec![
            Ok(json!({"chunk": 1})),
            Ok(json!({"chunk": 2})),
        ]));
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        drop(rx);

        forward_stream_to_channel(stream, tx).await;
    });
}

#[test]
fn test_promise_aware_key_scope_uuid_only_for_scope_variants() {
    let scope_uuid = "scope-uuid".to_string();

    assert_eq!(
        PromiseAwareKey::ScopeToolExecution {
            scope_uuid: scope_uuid.clone(),
            name: "tool".into(),
        }
        .scope_uuid(),
        Some("scope-uuid")
    );
    assert_eq!(
        PromiseAwareKey::ScopeLlmExecution {
            scope_uuid: scope_uuid.clone(),
            name: "llm".into(),
        }
        .scope_uuid(),
        Some("scope-uuid")
    );
    assert_eq!(
        PromiseAwareKey::ScopeLlmStreamExecution {
            scope_uuid,
            name: "stream".into(),
        }
        .scope_uuid(),
        Some("scope-uuid")
    );
    assert_eq!(
        PromiseAwareKey::GlobalToolExecution("tool".into()).scope_uuid(),
        None
    );
    assert_eq!(
        PromiseAwareKey::GlobalLlmExecution("llm".into()).scope_uuid(),
        None
    );
    assert_eq!(
        PromiseAwareKey::GlobalLlmStreamExecution("stream".into()).scope_uuid(),
        None
    );
}

#[test]
fn test_scope_stack_and_scope_lifecycle_wrappers() {
    let _guard = test_mutex().lock().unwrap();
    reset_global();

    assert!(!scope_stack_active());

    let stack = create_scope_stack();
    set_thread_scope_stack(&stack);
    assert!(scope_stack_active());

    let current = current_scope_stack();
    assert!(Arc::ptr_eq(&stack.inner, &current.inner));

    let root = get_handle().unwrap();
    assert_eq!(root.name(), "root");
    assert_eq!(root.scope_type() as i32, ScopeType::Agent as i32);

    let child = push_scope(
        "child".into(),
        ScopeType::Tool,
        Some(&root),
        Some(SCOPE_ATTR_PARALLEL),
        Some(json!({"payload": true})),
        Some(json!({"meta": true})),
    )
    .unwrap();
    assert_eq!(child.name(), "child");
    assert_eq!(child.scope_type() as i32, ScopeType::Tool as i32);
    assert_eq!(child.attributes(), SCOPE_ATTR_PARALLEL);
    assert_eq!(child.parent_uuid(), Some(root.uuid()));
    assert_eq!(child.data(), Some(json!({"payload": true})));
    assert_eq!(child.metadata(), Some(json!({"meta": true})));

    event(
        "mark".into(),
        Some(&child),
        Some(json!({"event": true})),
        Some(json!({"source": "test"})),
    )
    .unwrap();

    assert_eq!(get_handle().unwrap().name(), "child");
}

#[test]
fn test_tool_and_llm_lifecycle_wrappers() {
    let _guard = test_mutex().lock().unwrap();
    reset_global();

    let stack = create_scope_stack();
    set_thread_scope_stack(&stack);
    let parent = push_scope(
        "parent".into(),
        ScopeType::Agent,
        None,
        Some(SCOPE_ATTR_RELOCATABLE),
        None,
        None,
    )
    .unwrap();

    let tool = tool_call(
        "tool".into(),
        json!({"arg": 1}),
        Some(&parent),
        Some(TOOL_ATTR_LOCAL),
        Some(json!({"tool_data": true})),
        Some(json!({"tool_meta": true})),
        Some("tool-call-id".into()),
    )
    .unwrap();
    assert_eq!(tool.name(), "tool");
    assert_eq!(tool.attributes(), TOOL_ATTR_LOCAL);
    assert_eq!(tool.parent_uuid(), Some(parent.uuid()));
    tool_call_end(
        &tool,
        json!({"result": 2}),
        Some(json!({"done": true})),
        Some(json!({"status": "ok"})),
    )
    .unwrap();

    let llm = llm_call(
        "llm".into(),
        make_request(),
        Some(&parent),
        Some(LLM_ATTR_STATELESS | LLM_ATTR_STREAMING),
        Some(json!({"llm_data": true})),
        Some(json!({"llm_meta": true})),
        Some("model-name".into()),
    )
    .unwrap();
    assert_eq!(llm.name(), "llm");
    assert_eq!(llm.attributes(), LLM_ATTR_STATELESS | LLM_ATTR_STREAMING);
    assert_eq!(llm.parent_uuid(), Some(parent.uuid()));
    llm_call_end(
        &llm,
        json!({"response": "ok"}),
        Some(json!({"tokens": 10})),
        Some(json!({"finish_reason": "stop"})),
    )
    .unwrap();
}

#[test]
fn test_error_wrappers_for_invalid_inputs() {
    let _guard = test_mutex().lock().unwrap();
    reset_global();

    let llm_err = match llm_call(
        "bad-llm".into(),
        json!({"not": "an llm request"}),
        None,
        None,
        None,
        None,
        None,
    ) {
        Ok(_) => panic!("expected invalid LLMRequest error"),
        Err(err) => err,
    };
    assert!(llm_err.to_string().contains("invalid LLMRequest"));

    let scope_err = scope_deregister_subscriber("not-a-uuid".into(), "sub".into()).unwrap_err();
    assert!(scope_err.to_string().contains("invalid UUID"));
}

#[test]
fn test_parse_string_map_accepts_objects_and_rejects_invalid_shapes() {
    let parsed = parse_string_map(
        Some(json!({"authorization": "Bearer token", "env": "test"})),
        "headers",
    )
    .unwrap();
    assert_eq!(
        parsed.get("authorization"),
        Some(&"Bearer token".to_string())
    );
    assert_eq!(parsed.get("env"), Some(&"test".to_string()));

    let err = parse_string_map(Some(json!(["bad"])), "headers").unwrap_err();
    assert!(err.to_string().contains("headers must be an object"));

    let err = parse_string_map(Some(json!({"env": 1})), "headers").unwrap_err();
    assert!(err.to_string().contains("headers must be an object"));
}

#[test]
fn test_plugin_validate_and_lifecycle_wrappers() {
    nemo_flow_adaptive::plugin_component::register_adaptive_component().unwrap();
    let initial_kinds = list_plugin_kinds();
    assert!(initial_kinds.iter().any(|kind| kind == "adaptive"));
    assert!(!deregister_plugin("missing-plugin".into()));

    let report = validate_plugin_config(json!({
        "version": 1,
        "components": [{
            "kind": "adaptive",
            "enabled": true,
            "config": {
                "version": 1,
                "telemetry": {},
                "future_field": true
            }
        }]
    }))
    .unwrap();
    assert!(
        report["diagnostics"]
            .as_array()
            .unwrap()
            .iter()
            .any(|diag| diag["code"] == "adaptive.unknown_field")
    );

    let config = json!({
        "version": 1,
        "components": [
            {
                "kind": "adaptive",
                "enabled": true,
                "config": {
                    "version": 1,
                    "state": {
                        "backend": {
                            "kind": "in_memory",
                            "config": {}
                        }
                    },
                    "telemetry": {
                        "learners": ["latency_sensitivity"]
                    },
                    "adaptive_hints": {},
                    "tool_parallelism": {}
                }
            }
        ]
    });

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let report = initialize_plugins(config).await.unwrap();
        assert_eq!(report["diagnostics"], json!([]));
        assert_eq!(
            active_plugin_report().unwrap().unwrap()["diagnostics"],
            json!([])
        );
        assert!(list_plugin_kinds().iter().any(|kind| kind == "adaptive"));
        clear_plugin_configuration().unwrap();
        assert!(active_plugin_report().unwrap().is_none());
    });
}

#[test]
fn test_open_telemetry_subscriber_rejects_invalid_config() {
    assert!(build_otel_config(None).is_ok());

    let err = build_otel_config(Some(OpenTelemetryConfig {
        transport: Some("invalid".into()),
        ..Default::default()
    }))
    .unwrap_err();
    assert!(err.to_string().contains("transport must be"));

    let err = build_otel_config(Some(OpenTelemetryConfig {
        headers: Some(json!({"authorization": 1})),
        ..Default::default()
    }))
    .unwrap_err();
    assert!(err.to_string().contains("headers must be an object"));

    let err = build_otel_config(Some(OpenTelemetryConfig {
        resource_attributes: Some(json!({"env": 1})),
        ..Default::default()
    }))
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("resourceAttributes must be an object")
    );
}

#[test]
fn test_open_telemetry_subscriber_lifecycle_methods_work() {
    let subscriber = JsOpenTelemetrySubscriber::new(Some(OpenTelemetryConfig {
        endpoint: Some("http://localhost:4318/v1/traces".into()),
        service_name: Some("node-agent".into()),
        service_namespace: Some("agents".into()),
        service_version: Some("1.0.0".into()),
        instrumentation_scope: Some("node-tests".into()),
        timeout_millis: Some(1250),
        headers: Some(json!({"authorization": "Bearer token"})),
        resource_attributes: Some(json!({"deployment.environment": "test"})),
        ..Default::default()
    }))
    .unwrap();

    let name = format!("node_otel_{}", Uuid::now_v7().simple());
    subscriber.register(name.clone()).unwrap();
    assert!(subscriber.deregister(name.clone()).unwrap());
    assert!(!subscriber.deregister(name).unwrap());
    subscriber.force_flush().unwrap();
    subscriber.shutdown().unwrap();
}

#[test]
fn test_open_inference_subscriber_rejects_invalid_config() {
    assert!(build_openinference_config(None).is_ok());

    let err = build_openinference_config(Some(OpenInferenceConfig {
        transport: Some("invalid".into()),
        ..Default::default()
    }))
    .unwrap_err();
    assert!(err.to_string().contains("transport must be"));

    let err = build_openinference_config(Some(OpenInferenceConfig {
        headers: Some(json!({"authorization": 1})),
        ..Default::default()
    }))
    .unwrap_err();
    assert!(err.to_string().contains("headers must be an object"));

    let err = build_openinference_config(Some(OpenInferenceConfig {
        resource_attributes: Some(json!({"env": 1})),
        ..Default::default()
    }))
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("resourceAttributes must be an object")
    );
}

#[test]
fn test_open_inference_subscriber_lifecycle_methods_work() {
    let subscriber = JsOpenInferenceSubscriber::new(Some(OpenInferenceConfig {
        endpoint: Some("http://localhost:4318/v1/traces".into()),
        service_name: Some("node-agent".into()),
        service_namespace: Some("agents".into()),
        service_version: Some("1.0.0".into()),
        instrumentation_scope: Some("node-tests".into()),
        timeout_millis: Some(1250),
        headers: Some(json!({"authorization": "Bearer token"})),
        resource_attributes: Some(json!({"deployment.environment": "test"})),
        ..Default::default()
    }))
    .unwrap();

    let name = format!("node_openinference_{}", Uuid::now_v7().simple());
    subscriber.register(name.clone()).unwrap();
    assert!(subscriber.deregister(name.clone()).unwrap());
    assert!(!subscriber.deregister(name).unwrap());
    subscriber.force_flush().unwrap();
    subscriber.shutdown().unwrap();
}
