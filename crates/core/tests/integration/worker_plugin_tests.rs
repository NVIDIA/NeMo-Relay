// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration coverage for gRPC worker dynamic plugins.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex, OnceLock};

use futures::StreamExt;
use nemo_relay::api::event::{Event, ScopeCategory};
use nemo_relay::api::llm::{
    LlmCallExecuteParams, LlmRequest, LlmStreamCallExecuteParams, llm_call_execute,
    llm_stream_call_execute,
};
use nemo_relay::api::runtime::{TASK_SCOPE_STACK, create_scope_stack};
use nemo_relay::api::scope::{
    EmitMarkEventParams, PopScopeParams, PushScopeParams, ScopeType, event, pop_scope, push_scope,
};
use nemo_relay::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use nemo_relay::api::tool::{ToolCallExecuteParams, tool_call_execute, tool_request_intercepts};
use nemo_relay::codec::request::AnnotatedLlmRequest;
use nemo_relay::codec::traits::LlmCodec;
use nemo_relay::error::Result as FlowResult;
use nemo_relay::plugin::dynamic::{
    WorkerPluginActivation, WorkerPluginLoadSpec, load_worker_plugins,
};
use nemo_relay::plugin::{
    PluginComponentSpec, PluginConfig, clear_plugin_configuration, initialize_plugins_exact,
};
use serde_json::{Map, Value as Json, json};
use sha2::{Digest, Sha256};
use tempfile::TempDir;
use uuid::Uuid;

static WORKER_PLUGIN_TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

#[test]
fn worker_activation_with_no_specs_is_empty() {
    let activation = load_worker_plugins(Vec::<WorkerPluginLoadSpec>::new())
        .expect("empty worker activation should succeed");
    assert!(activation.is_empty());
    activation.clear();
}

#[tokio::test]
async fn rust_worker_registers_and_invokes_all_current_surfaces() {
    let _guard = WORKER_PLUGIN_TEST_LOCK.lock().await;
    let loaded = load_and_initialize_fixture(Map::new()).await;

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    register_subscriber(
        "worker_plugin_fixture_events",
        Arc::new(move |event| {
            captured.lock().unwrap().push(event.clone());
        }),
    )
    .expect("test subscriber should register");

    let stack = create_scope_stack();
    let (outer_uuid, rewritten, tool_result) = TASK_SCOPE_STACK
        .scope(stack, async {
            let outer = push_scope(
                PushScopeParams::builder()
                    .name("worker-plugin-test-outer")
                    .scope_type(ScopeType::Agent)
                    .build(),
            )
            .expect("outer scope should push");
            let outer_uuid = outer.uuid;
            let rewritten = tool_request_intercepts("demo_tool", json!({ "input": "value" }))
                .expect("worker request intercept should run");
            let tool_result = tool_call_execute(
                ToolCallExecuteParams::builder()
                    .name("worker-fixture-tool")
                    .args(json!({ "input": "execute" }))
                    .func(Arc::new(|args| {
                        Box::pin(async move { Ok(json!({ "tool_callback": true, "args": args })) })
                    }))
                    .build(),
            )
            .await
            .expect("worker tool middleware should run");
            pop_scope(PopScopeParams::builder().handle_uuid(&outer.uuid).build())
                .expect("outer scope should pop");
            (outer_uuid, rewritten, tool_result)
        })
        .await;

    assert_eq!(rewritten["worker_plugin"], true);
    assert_eq!(tool_result["tool_callback"], true);
    assert_eq!(tool_result["worker_plugin_tool_execution"], true);
    assert_eq!(
        tool_result["args"]["worker_plugin_tool_execution_request"],
        true
    );

    flush_subscribers().expect("worker fixture events should flush");
    let captured_events = events.lock().unwrap().clone();
    assert_parent(
        &captured_events,
        "fixture.worker.mark",
        None,
        Some(outer_uuid),
    );
    assert_eq!(
        find_event(&captured_events, "fixture.worker.mark", None)
            .metadata()
            .unwrap()["worker_plugin_mark"],
        true
    );
    assert_parent(
        &captured_events,
        "fixture.worker.scope",
        Some(ScopeCategory::Start),
        Some(outer_uuid),
    );
    assert_eq!(
        find_event(
            &captured_events,
            "fixture.worker.scope",
            Some(ScopeCategory::Start),
        )
        .metadata()
        .unwrap()["worker_plugin_scope_start"],
        true
    );
    assert_eq!(
        find_event(
            &captured_events,
            "fixture.worker.scope",
            Some(ScopeCategory::End),
        )
        .metadata()
        .unwrap()["worker_plugin_scope_end"],
        true
    );
    assert_not_parent(
        &captured_events,
        "fixture.worker.isolated.scope",
        Some(ScopeCategory::Start),
        outer_uuid,
    );
    let isolated_scope = find_event(
        &captured_events,
        "fixture.worker.isolated.scope",
        Some(ScopeCategory::Start),
    );
    let isolated_mark = find_event(&captured_events, "fixture.worker.isolated.mark", None);
    assert_eq!(isolated_mark.parent_uuid(), Some(isolated_scope.uuid()));
    assert_ne!(
        isolated_mark.parent_uuid(),
        Some(outer_uuid),
        "worker isolated mark should use the plugin-selected isolated stack"
    );
    let tool_start = find_event(
        &captured_events,
        "worker-fixture-tool",
        Some(ScopeCategory::Start),
    );
    assert_eq!(
        tool_start.input().unwrap()["worker_plugin_tool_sanitize_request"],
        true
    );
    assert_eq!(
        tool_start.metadata().unwrap()["worker_plugin_scope_start"],
        true
    );
    let tool_end = find_event(
        &captured_events,
        "worker-fixture-tool",
        Some(ScopeCategory::End),
    );
    assert_eq!(
        tool_end.output().unwrap()["worker_plugin_tool_sanitize_response"],
        true
    );
    assert_eq!(
        tool_end.metadata().unwrap()["worker_plugin_scope_end"],
        true
    );
    let tool_mark = find_event(&captured_events, "fixture.worker.tool_execution.mark", None);
    assert_eq!(tool_mark.parent_uuid(), Some(tool_start.uuid()));
    assert!(tool_mark.timestamp() > tool_end.timestamp());
    assert_eq!(tool_mark.metadata().unwrap()["worker_plugin_mark"], true);

    let llm_execute_response = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("worker-fixture-llm-execute")
            .request(LlmRequest {
                headers: Map::new(),
                content: json!({ "prompt": "managed" }),
            })
            .func(Arc::new(|request| {
                Box::pin(async move {
                    Ok(json!({
                        "id": "managed-response",
                        "request": request.content,
                        "llm_callback": true
                    }))
                })
            }))
            .build(),
    )
    .await
    .expect("worker LLM middleware should run");
    assert_eq!(llm_execute_response["llm_callback"], true);
    assert_eq!(llm_execute_response["worker_plugin_llm_execution"], true);
    assert_eq!(
        llm_execute_response["request"]["worker_plugin_llm_execution_request"],
        true
    );
    flush_subscribers().expect("worker fixture LLM events should flush");
    let captured_events = events.lock().unwrap().clone();
    find_event(&captured_events, "fixture.worker.subscriber.mark", None);
    let llm_start = find_event(
        &captured_events,
        "worker-fixture-llm-execute",
        Some(ScopeCategory::Start),
    );
    assert_eq!(
        llm_start.input().unwrap()["content"]["worker_plugin_llm_sanitize_request"],
        true
    );
    let pending_mark = find_event(&captured_events, "fixture.worker.llm_request.mark", None);
    assert_eq!(pending_mark.parent_uuid(), Some(llm_start.uuid()));
    assert_eq!(
        pending_mark.data().unwrap()["source"],
        "worker_request_intercept"
    );
    assert_eq!(pending_mark.metadata().unwrap()["fixture"], true);
    assert_eq!(pending_mark.metadata().unwrap()["worker_plugin_mark"], true);
    let llm_end = find_event(
        &captured_events,
        "worker-fixture-llm-execute",
        Some(ScopeCategory::End),
    );
    assert_eq!(
        llm_end.output().unwrap()["worker_plugin_llm_sanitize_response"],
        true
    );

    let stream_values = llm_stream_call_execute(
        LlmStreamCallExecuteParams::builder()
            .name("worker-fixture-llm-stream")
            .request(LlmRequest {
                headers: Map::new(),
                content: json!({ "prompt": "stream" }),
            })
            .func(Arc::new(|request| {
                Box::pin(async move {
                    let first = json!({
                        "chunk": 1,
                        "request": request.content,
                    });
                    Ok(Box::pin(tokio_stream::iter(vec![Ok(first)])) as _)
                })
            }))
            .collector(Box::new(|_chunk| Ok(())))
            .finalizer(Box::new(|| json!({ "done": true })))
            .build(),
    )
    .await
    .expect("worker stream middleware should start")
    .collect::<Vec<_>>()
    .await;
    let stream_value = stream_values
        .into_iter()
        .next()
        .expect("one stream chunk should be returned")
        .expect("stream chunk should succeed");
    assert_eq!(stream_value["worker_plugin_llm_stream_execution"], true);
    assert_eq!(
        stream_value["request"]["worker_plugin_llm_stream_execution_request"],
        true
    );

    loaded.clear();
}

#[tokio::test]
async fn worker_event_sanitizers_preserve_prior_field_changes() {
    let _guard = WORKER_PLUGIN_TEST_LOCK.lock().await;
    let loaded = load_and_initialize_fixture(Map::new()).await;
    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    register_subscriber(
        "worker_event_sanitizer_field_preservation",
        Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
    )
    .expect("test subscriber should register");

    TASK_SCOPE_STACK
        .scope(create_scope_stack(), async {
            event(
                EmitMarkEventParams::builder()
                    .name("worker-event-sanitizer-field-preservation")
                    .data(json!({"original_data": true}))
                    .metadata(json!({"original_metadata": true}))
                    .build(),
            )
            .expect("mark event should emit");
        })
        .await;

    flush_subscribers().expect("worker fixture events should flush");
    let captured_events = events.lock().unwrap().clone();
    let mark = find_event(
        &captured_events,
        "worker-event-sanitizer-field-preservation",
        None,
    );
    assert_eq!(mark.data(), Some(&json!({"worker_plugin_mark_data": true})));
    assert_eq!(mark.metadata().unwrap()["worker_plugin_mark"], true);
    assert_eq!(mark.metadata().unwrap()["original_metadata"], true);

    deregister_subscriber("worker_event_sanitizer_field_preservation")
        .expect("test subscriber should deregister");
    loaded.clear();
}

#[tokio::test]
async fn host_cancellation_reaches_rust_worker_invocation() {
    let _guard = WORKER_PLUGIN_TEST_LOCK.lock().await;
    let loaded = load_and_initialize_fixture(Map::new()).await;
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let (dropped_tx, dropped_rx) = tokio::sync::oneshot::channel();
    let started_tx = Arc::new(Mutex::new(Some(started_tx)));
    let dropped_tx = Arc::new(Mutex::new(Some(dropped_tx)));

    let execution = tokio::spawn(tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("worker-fixture-cancelled-tool")
            .args(json!({ "input": "cancel" }))
            .func(Arc::new(move |_| {
                let started = started_tx
                    .lock()
                    .expect("started lock")
                    .take()
                    .expect("callback should run once");
                let dropped = dropped_tx
                    .lock()
                    .expect("dropped lock")
                    .take()
                    .expect("callback should run once");
                Box::pin(async move {
                    struct DropSignal(Option<tokio::sync::oneshot::Sender<()>>);
                    impl Drop for DropSignal {
                        fn drop(&mut self) {
                            if let Some(sender) = self.0.take() {
                                let _ = sender.send(());
                            }
                        }
                    }

                    let _drop_signal = DropSignal(Some(dropped));
                    let _ = started.send(());
                    std::future::pending::<FlowResult<Json>>().await
                })
            }))
            .build(),
    ));
    tokio::time::timeout(std::time::Duration::from_secs(2), started_rx)
        .await
        .expect("worker should call the host continuation before cancellation")
        .expect("worker should call the host continuation");

    execution.abort();
    let _ = execution.await;
    tokio::time::timeout(std::time::Duration::from_secs(2), dropped_rx)
        .await
        .expect("worker cancellation should drop the host continuation")
        .expect("host continuation drop signal should be delivered");

    loaded.clear();
}

#[tokio::test]
async fn worker_request_intercept_callback_error_surfaces_to_host() {
    let _guard = WORKER_PLUGIN_TEST_LOCK.lock().await;
    let loaded =
        load_and_initialize_fixture(Map::from_iter([("tool_request_error".into(), json!(true))]))
            .await;

    let error = tool_request_intercepts("demo_tool", json!({ "input": "value" }))
        .expect_err("worker callback error should surface");
    assert!(
        error
            .to_string()
            .contains("fixture tool request error requested"),
        "{error}"
    );

    loaded.clear();
}

#[tokio::test]
async fn worker_conditional_guardrail_blocks_tool_execution() {
    let _guard = WORKER_PLUGIN_TEST_LOCK.lock().await;
    let loaded =
        load_and_initialize_fixture(Map::from_iter([("block_tool".into(), json!(true))])).await;

    let error = tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("worker-fixture-blocked-tool")
            .args(json!({ "input": "blocked" }))
            .func(Arc::new(|_| {
                Box::pin(async move { Ok(json!({ "should_not_run": true })) })
            }))
            .build(),
    )
    .await
    .expect_err("worker guardrail should block tool execution");
    assert!(
        error.to_string().contains("fixture tool blocked"),
        "{error}"
    );

    loaded.clear();
}

#[tokio::test]
async fn worker_llm_request_intercept_round_trips_annotations() {
    let _guard = WORKER_PLUGIN_TEST_LOCK.lock().await;
    let loaded = load_and_initialize_fixture(Map::new()).await;

    let response = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("worker-fixture-llm-annotated")
            .request(LlmRequest {
                headers: Map::new(),
                content: json!({ "prompt": "annotated" }),
            })
            .codec(Arc::new(FixtureCodec))
            .func(Arc::new(|request| {
                Box::pin(async move {
                    Ok(json!({
                        "request": request.content,
                        "llm_callback": true
                    }))
                })
            }))
            .build(),
    )
    .await
    .expect("worker LLM request intercept should preserve annotations");
    assert_eq!(response["llm_callback"], true);
    assert_eq!(response["request"]["worker_plugin_annotated_request"], true);

    loaded.clear();
}

#[tokio::test]
async fn worker_llm_request_intercept_callback_error_surfaces_to_host() {
    let _guard = WORKER_PLUGIN_TEST_LOCK.lock().await;
    let loaded =
        load_and_initialize_fixture(Map::from_iter([("llm_request_error".into(), json!(true))]))
            .await;

    let error = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("worker-fixture-llm-error")
            .request(LlmRequest {
                headers: Map::new(),
                content: json!({ "prompt": "error" }),
            })
            .func(Arc::new(|request| {
                Box::pin(async move {
                    Ok(json!({
                        "request": request.content,
                        "should_not_complete": true
                    }))
                })
            }))
            .build(),
    )
    .await
    .expect_err("worker LLM request intercept error should surface");
    assert!(
        error
            .to_string()
            .contains("fixture LLM request error requested"),
        "{error}"
    );

    loaded.clear();
}

#[tokio::test]
async fn worker_llm_stream_open_error_surfaces_to_host() {
    let _guard = WORKER_PLUGIN_TEST_LOCK.lock().await;
    let loaded = load_and_initialize_fixture(Map::from_iter([(
        "llm_stream_open_error".into(),
        json!(true),
    )]))
    .await;

    let mut stream = llm_stream_call_execute(
        LlmStreamCallExecuteParams::builder()
            .name("worker-fixture-llm-stream-error")
            .request(LlmRequest {
                headers: Map::new(),
                content: json!({ "prompt": "stream-error" }),
            })
            .func(Arc::new(|request| {
                Box::pin(async move {
                    let chunk = json!({ "request": request.content });
                    Ok(Box::pin(tokio_stream::iter(vec![Ok(chunk)])) as _)
                })
            }))
            .collector(Box::new(|_chunk| Ok(())))
            .finalizer(Box::new(|| json!({ "done": true })))
            .build(),
    )
    .await
    .expect("worker stream invoke should return a host stream");
    let error = stream
        .next()
        .await
        .expect("stream should yield the worker error")
        .expect_err("worker stream callback error should surface");
    assert!(
        error
            .to_string()
            .contains("fixture LLM stream open error requested"),
        "{error}"
    );

    loaded.clear();
}

#[tokio::test]
async fn worker_validation_diagnostics_prevent_initialization() {
    let _guard = WORKER_PLUGIN_TEST_LOCK.lock().await;
    let fixture = build_fixture_worker();
    let (_manifest_dir, manifest_ref) = write_manifest(fixture.binary_path());
    let config = Map::from_iter([("reject".into(), json!(true))]);

    let activation = load_worker_plugins([WorkerPluginLoadSpec {
        plugin_id: "fixture_worker".into(),
        manifest_ref: manifest_ref.to_string_lossy().into_owned(),
        environment_ref: None,
        config: config.clone(),
    }])
    .expect("worker plugin should load with validation diagnostics");

    let mut plugin_config = PluginConfig::default();
    plugin_config.components.push(PluginComponentSpec {
        kind: "fixture_worker".into(),
        enabled: true,
        config,
    });
    let error = initialize_plugins_exact(plugin_config)
        .await
        .expect_err("validation diagnostics should prevent initialization")
        .to_string();
    assert!(error.contains("fixture rejection requested"), "{error}");

    clear_plugin_configuration().expect("worker plugin config should clear");
    activation.clear();
}

#[tokio::test]
async fn worker_duplicate_component_rejected_for_single_instance_plugin() {
    let _guard = WORKER_PLUGIN_TEST_LOCK.lock().await;
    let fixture = build_fixture_worker();
    let (_manifest_dir, manifest_ref) = write_manifest(fixture.binary_path());

    let activation = load_worker_plugins([WorkerPluginLoadSpec {
        plugin_id: "fixture_worker".into(),
        manifest_ref: manifest_ref.to_string_lossy().into_owned(),
        environment_ref: None,
        config: Map::new(),
    }])
    .expect("worker plugin should load");

    let mut plugin_config = PluginConfig::default();
    plugin_config.components.push(PluginComponentSpec {
        kind: "fixture_worker".into(),
        enabled: true,
        config: Map::new(),
    });
    plugin_config.components.push(PluginComponentSpec {
        kind: "fixture_worker".into(),
        enabled: true,
        config: Map::new(),
    });
    let error = initialize_plugins_exact(plugin_config)
        .await
        .expect_err("single-instance worker plugin should reject duplicate components")
        .to_string();
    assert!(error.contains("may only appear once"), "{error}");

    clear_plugin_configuration().expect("worker plugin config should clear");
    activation.clear();
}

#[tokio::test]
async fn worker_config_mismatch_prevents_initialization() {
    let _guard = WORKER_PLUGIN_TEST_LOCK.lock().await;
    let fixture = build_fixture_worker();
    let (_manifest_dir, manifest_ref) = write_manifest(fixture.binary_path());

    let activation = load_worker_plugins([WorkerPluginLoadSpec {
        plugin_id: "fixture_worker".into(),
        manifest_ref: manifest_ref.to_string_lossy().into_owned(),
        environment_ref: None,
        config: Map::new(),
    }])
    .expect("worker plugin should load");

    let mut plugin_config = PluginConfig::default();
    plugin_config.components.push(PluginComponentSpec {
        kind: "fixture_worker".into(),
        enabled: true,
        config: Map::from_iter([("changed".into(), json!(true))]),
    });
    let error = initialize_plugins_exact(plugin_config)
        .await
        .expect_err("config drift should prevent initialization")
        .to_string();
    assert!(error.contains("config changed"), "{error}");

    clear_plugin_configuration().expect("worker plugin config should clear");
    activation.clear();
}

#[tokio::test]
async fn worker_registration_error_fails_activation() {
    let _guard = WORKER_PLUGIN_TEST_LOCK.lock().await;
    let fixture = build_fixture_worker();
    let (_manifest_dir, manifest_ref) = write_manifest(fixture.binary_path());

    let error = match load_worker_plugins([WorkerPluginLoadSpec {
        plugin_id: "fixture_worker".into(),
        manifest_ref: manifest_ref.to_string_lossy().into_owned(),
        environment_ref: None,
        config: Map::from_iter([("register_error".into(), json!(true))]),
    }]) {
        Ok(activation) => {
            activation.clear();
            panic!("worker registration error should fail activation");
        }
        Err(error) => error.to_string(),
    };
    assert!(
        error.contains("fixture registration error requested"),
        "{error}"
    );
}

#[tokio::test]
async fn worker_invalid_registration_plan_fails_activation() {
    let _guard = WORKER_PLUGIN_TEST_LOCK.lock().await;
    let fixture = build_fixture_worker();
    let (_manifest_dir, manifest_ref) = write_manifest(fixture.binary_path());

    let error = match load_worker_plugins([WorkerPluginLoadSpec {
        plugin_id: "fixture_worker".into(),
        manifest_ref: manifest_ref.to_string_lossy().into_owned(),
        environment_ref: None,
        config: Map::from_iter([("empty_registration_name".into(), json!(true))]),
    }]) {
        Ok(activation) => {
            activation.clear();
            panic!("empty registration name should fail activation");
        }
        Err(error) => error.to_string(),
    };
    assert!(error.contains("empty local_name"), "{error}");
}

#[tokio::test]
async fn worker_handshake_plugin_id_mismatch_reports_config_error() {
    let _guard = WORKER_PLUGIN_TEST_LOCK.lock().await;
    let _env = EnvVarGuard::set("FIXTURE_WORKER_PLUGIN_ID", "other_worker");
    let fixture = build_fixture_worker();
    let (_manifest_dir, manifest_ref) = write_manifest(fixture.binary_path());

    let error = match load_worker_plugins([WorkerPluginLoadSpec {
        plugin_id: "fixture_worker".into(),
        manifest_ref: manifest_ref.to_string_lossy().into_owned(),
        environment_ref: None,
        config: Map::new(),
    }]) {
        Ok(activation) => {
            activation.clear();
            panic!("worker handshake id mismatch should fail activation");
        }
        Err(error) => error.to_string(),
    };
    assert!(error.contains("returned id 'other_worker'"), "{error}");
}

#[tokio::test]
async fn worker_validation_rpc_failure_reports_activation_error() {
    let _guard = WORKER_PLUGIN_TEST_LOCK.lock().await;
    let fixture = build_fixture_worker();
    let (_manifest_dir, manifest_ref) = write_manifest(fixture.binary_path());

    let error = match load_worker_plugins([WorkerPluginLoadSpec {
        plugin_id: "fixture_worker".into(),
        manifest_ref: manifest_ref.to_string_lossy().into_owned(),
        environment_ref: None,
        config: Map::from_iter([("exit_in_validate".into(), json!(true))]),
    }]) {
        Ok(activation) => {
            activation.clear();
            panic!("worker validation process exit should fail activation");
        }
        Err(error) => error.to_string(),
    };
    assert!(error.contains("worker validation RPC failed"), "{error}");
}

#[tokio::test]
async fn worker_registration_rpc_failure_reports_activation_error() {
    let _guard = WORKER_PLUGIN_TEST_LOCK.lock().await;
    let fixture = build_fixture_worker();
    let (_manifest_dir, manifest_ref) = write_manifest(fixture.binary_path());

    let error = match load_worker_plugins([WorkerPluginLoadSpec {
        plugin_id: "fixture_worker".into(),
        manifest_ref: manifest_ref.to_string_lossy().into_owned(),
        environment_ref: None,
        config: Map::from_iter([("exit_in_register".into(), json!(true))]),
    }]) {
        Ok(activation) => {
            activation.clear();
            panic!("worker registration process exit should fail activation");
        }
        Err(error) => error.to_string(),
    };
    assert!(error.contains("worker registration RPC failed"), "{error}");
}

#[test]
fn missing_worker_executable_reports_startup_error() {
    let _guard = WORKER_PLUGIN_TEST_LOCK.blocking_lock();
    let missing_binary = std::env::temp_dir().join(format!("missing-worker-{}", Uuid::now_v7()));
    let (_manifest_dir, manifest_ref) = write_manifest(&missing_binary);

    let error = match load_worker_plugins([WorkerPluginLoadSpec {
        plugin_id: "fixture_worker".into(),
        manifest_ref: manifest_ref.to_string_lossy().into_owned(),
        environment_ref: None,
        config: Map::new(),
    }]) {
        Ok(activation) => {
            activation.clear();
            panic!("missing worker executable should fail activation");
        }
        Err(error) => error.to_string(),
    };
    assert!(error.contains("failed to spawn"), "{error}");
}

#[test]
fn worker_manifest_id_mismatch_reports_config_error() {
    let _guard = WORKER_PLUGIN_TEST_LOCK.blocking_lock();
    let missing_binary = std::env::temp_dir().join(format!("unused-worker-{}", Uuid::now_v7()));
    let (_manifest_dir, manifest_ref) = write_manifest(&missing_binary);

    let error = match load_worker_plugins([WorkerPluginLoadSpec {
        plugin_id: "different_worker".into(),
        manifest_ref: manifest_ref.to_string_lossy().into_owned(),
        environment_ref: None,
        config: Map::new(),
    }]) {
        Ok(activation) => {
            activation.clear();
            panic!("manifest id mismatch should fail");
        }
        Err(error) => error.to_string(),
    };
    assert!(error.contains("does not match expected id"), "{error}");
}

#[test]
fn worker_manifest_kind_mismatch_reports_config_error() {
    let _guard = WORKER_PLUGIN_TEST_LOCK.blocking_lock();
    let relay = supported_relay_requirement();
    let (_manifest_dir, manifest_ref) = write_manifest_text(&format!(
        r#"
manifest_version = 1

[plugin]
id = "fixture_worker"
kind = "rust_dynamic"

[compat]
relay = {relay}
native_api = "1"

[defaults]
enabled = false

[capabilities]
items = ["plugin_native"]

[load]
library = "missing"
symbol = "nemo_relay_plugin_entry"
"#,
        relay = toml_string(&relay)
    ));

    let error = match load_worker_plugins([WorkerPluginLoadSpec {
        plugin_id: "fixture_worker".into(),
        manifest_ref: manifest_ref.to_string_lossy().into_owned(),
        environment_ref: None,
        config: Map::new(),
    }]) {
        Ok(activation) => {
            activation.clear();
            panic!("manifest kind mismatch should fail");
        }
        Err(error) => error.to_string(),
    };
    assert!(
        error.contains("worker loader only supports worker"),
        "{error}"
    );
}

#[test]
fn unsupported_worker_relay_requirement_reports_compatibility_error() {
    let _guard = WORKER_PLUGIN_TEST_LOCK.blocking_lock();
    let missing_binary = std::env::temp_dir().join(format!("unused-worker-{}", Uuid::now_v7()));
    let (_manifest_dir, manifest_ref) =
        write_manifest_with_relay(&missing_binary, ">=9999.0,<10000.0");

    let error = match load_worker_plugins([WorkerPluginLoadSpec {
        plugin_id: "fixture_worker".into(),
        manifest_ref: manifest_ref.to_string_lossy().into_owned(),
        environment_ref: None,
        config: Map::new(),
    }]) {
        Ok(activation) => {
            activation.clear();
            panic!("unsupported relay requirement should fail");
        }
        Err(error) => error.to_string(),
    };
    assert!(error.contains("requires relay"), "{error}");
}

#[test]
fn invalid_worker_relay_requirement_reports_parse_error() {
    let _guard = WORKER_PLUGIN_TEST_LOCK.blocking_lock();
    let missing_binary = std::env::temp_dir().join(format!("unused-worker-{}", Uuid::now_v7()));
    let (_manifest_dir, manifest_ref) = write_manifest_with_relay(&missing_binary, "not semver");

    let error = match load_worker_plugins([WorkerPluginLoadSpec {
        plugin_id: "fixture_worker".into(),
        manifest_ref: manifest_ref.to_string_lossy().into_owned(),
        environment_ref: None,
        config: Map::new(),
    }]) {
        Ok(activation) => {
            activation.clear();
            panic!("invalid relay requirement should fail");
        }
        Err(error) => error.to_string(),
    };
    assert!(error.contains("invalid compat.relay"), "{error}");
}

#[test]
fn command_worker_entrypoint_is_resolved_relative_to_manifest() {
    let _guard = WORKER_PLUGIN_TEST_LOCK.blocking_lock();
    let relay = supported_relay_requirement();
    let (manifest_dir, manifest_ref) =
        write_worker_manifest("fixture_worker", &relay, "command", "missing-worker");

    let error = match load_worker_plugins([WorkerPluginLoadSpec {
        plugin_id: "fixture_worker".into(),
        manifest_ref: manifest_ref.to_string_lossy().into_owned(),
        environment_ref: None,
        config: Map::new(),
    }]) {
        Ok(activation) => {
            activation.clear();
            panic!("missing relative command worker should fail");
        }
        Err(error) => error.to_string(),
    };
    assert!(error.contains("failed to spawn command worker"), "{error}");
    assert_error_mentions_manifest_relative_entrypoint(
        &error,
        manifest_dir.path(),
        "missing-worker",
    );
}

#[test]
fn python_worker_without_managed_environment_is_rejected() {
    let _guard = WORKER_PLUGIN_TEST_LOCK.blocking_lock();
    let relay = supported_relay_requirement();
    let (_manifest_dir, manifest_ref) = write_worker_manifest(
        "fixture_worker",
        &relay,
        "python",
        "fixture_worker:create_plugin",
    );

    let error = match load_worker_plugins([WorkerPluginLoadSpec {
        plugin_id: "fixture_worker".into(),
        manifest_ref: manifest_ref.to_string_lossy().into_owned(),
        environment_ref: None,
        config: Map::new(),
    }]) {
        Ok(activation) => {
            activation.clear();
            panic!("direct Python worker loading should fail");
        }
        Err(error) => error.to_string(),
    };
    assert!(
        error.contains("requires a lifecycle-managed environment_ref")
            && error.contains("plugins add"),
        "{error}"
    );
}

#[test]
fn python_worker_uses_configured_environment() {
    let _guard = WORKER_PLUGIN_TEST_LOCK.blocking_lock();
    let environment_name = Sha256::digest(b"fixture_worker")
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let missing_environment = std::env::temp_dir()
        .join(format!("missing-python-environment-{}", Uuid::now_v7()))
        .join(".dynamic-plugin-environments")
        .join(environment_name);
    let relay = supported_relay_requirement();
    let (_manifest_dir, manifest_ref) = write_worker_manifest(
        "fixture_worker",
        &relay,
        "python",
        "fixture_worker:create_plugin",
    );

    let error = match load_worker_plugins([WorkerPluginLoadSpec {
        plugin_id: "fixture_worker".into(),
        manifest_ref: manifest_ref.to_string_lossy().into_owned(),
        environment_ref: Some(missing_environment.to_string_lossy().into_owned()),
        config: Map::new(),
    }]) {
        Ok(activation) => {
            activation.clear();
            panic!("missing configured Python interpreter should fail");
        }
        Err(error) => error.to_string(),
    };
    assert!(
        error.contains("configured Python worker environment interpreter")
            && error.contains("does not exist"),
        "{error}"
    );
}

#[tokio::test]
async fn python_worker_host_runtime_mark_and_mutated_request_round_trip() {
    let _guard = WORKER_PLUGIN_TEST_LOCK.lock().await;
    let Some(environment_ref) = std::env::var_os("NEMO_RELAY_PYTHON_PLUGIN_TEST_ENVIRONMENT")
    else {
        eprintln!(
            "skipping Python worker round-trip; NEMO_RELAY_PYTHON_PLUGIN_TEST_ENVIRONMENT is unset"
        );
        return;
    };
    let manifest_ref = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/python-grpc-worker-plugin/relay-plugin.toml");
    let config = Map::from_iter([("tag".into(), json!("managed-environment"))]);
    let activation = load_worker_plugins([WorkerPluginLoadSpec {
        plugin_id: "examples.python_grpc_worker".into(),
        manifest_ref: manifest_ref.to_string_lossy().into_owned(),
        environment_ref: Some(
            PathBuf::from(environment_ref)
                .to_string_lossy()
                .into_owned(),
        ),
        config: config.clone(),
    }])
    .expect("managed Python worker should load");
    let mut cleanup = PythonWorkerCleanup::new(activation);

    let mut plugin_config = PluginConfig::default();
    plugin_config.components.push(PluginComponentSpec {
        kind: "examples.python_grpc_worker".into(),
        enabled: true,
        config,
    });
    initialize_plugins_exact(plugin_config)
        .await
        .expect("managed Python worker should initialize");

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    let subscriber_name = "python_worker_host_runtime_round_trip";
    register_subscriber(
        subscriber_name,
        Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
    )
    .expect("test subscriber should register");
    cleanup.subscriber_name = Some(subscriber_name);

    let rewritten = tool_request_intercepts("lookup", json!({ "query": "relay" }))
        .expect("Python callback should emit a mark and return its mutation");
    assert_eq!(
        rewritten["_nemo_relay_plugin"]["tag"],
        "managed-environment"
    );
    flush_subscribers().expect("Python callback mark should flush");
    find_event(
        &events.lock().unwrap(),
        "examples.python_grpc_worker.tool_request",
        None,
    );

    drop(cleanup);
}

struct FixtureCodec;

impl LlmCodec for FixtureCodec {
    fn decode(&self, request: &LlmRequest) -> FlowResult<AnnotatedLlmRequest> {
        Ok(AnnotatedLlmRequest {
            messages: Vec::new(),
            model: Some("fixture-model".into()),
            params: None,
            tools: None,
            tool_choice: None,
            store: None,
            previous_response_id: None,
            truncation: None,
            reasoning: None,
            include: None,
            user: None,
            metadata: None,
            service_tier: None,
            parallel_tool_calls: None,
            max_output_tokens: None,
            max_tool_calls: None,
            top_logprobs: None,
            stream: None,
            extra: request.content.as_object().cloned().unwrap_or_default(),
        })
    }

    fn encode(
        &self,
        annotated: &AnnotatedLlmRequest,
        original: &LlmRequest,
    ) -> FlowResult<LlmRequest> {
        Ok(LlmRequest {
            headers: original.headers.clone(),
            content: Json::Object(annotated.extra.clone()),
        })
    }
}

struct LoadedWorker {
    activation: Option<WorkerPluginActivation>,
    _manifest_dir: TempDir,
}

struct PythonWorkerCleanup {
    activation: Option<WorkerPluginActivation>,
    subscriber_name: Option<&'static str>,
}

impl PythonWorkerCleanup {
    fn new(activation: WorkerPluginActivation) -> Self {
        Self {
            activation: Some(activation),
            subscriber_name: None,
        }
    }
}

impl Drop for PythonWorkerCleanup {
    fn drop(&mut self) {
        if let Some(subscriber_name) = self.subscriber_name.take() {
            let _ = deregister_subscriber(subscriber_name);
        }
        let _ = clear_plugin_configuration();
        if let Some(activation) = self.activation.take() {
            activation.clear();
        }
    }
}

impl LoadedWorker {
    fn clear(mut self) {
        clear_plugin_configuration().expect("worker plugin config should clear");
        if let Some(activation) = self.activation.take() {
            activation.clear();
        }
    }
}

impl Drop for LoadedWorker {
    fn drop(&mut self) {
        let _ = clear_plugin_configuration();
        if let Some(activation) = self.activation.take() {
            activation.clear();
        }
    }
}

async fn load_and_initialize_fixture(config: Map<String, Json>) -> LoadedWorker {
    let fixture = build_fixture_worker();
    let (manifest_dir, manifest_ref) = write_manifest(fixture.binary_path());

    let activation = load_worker_plugins([WorkerPluginLoadSpec {
        plugin_id: "fixture_worker".into(),
        manifest_ref: manifest_ref.to_string_lossy().into_owned(),
        environment_ref: None,
        config: config.clone(),
    }])
    .expect("worker plugin should load");

    let mut plugin_config = PluginConfig::default();
    plugin_config.components.push(PluginComponentSpec {
        kind: "fixture_worker".into(),
        enabled: true,
        config,
    });
    initialize_plugins_exact(plugin_config)
        .await
        .expect("worker plugin should initialize");

    LoadedWorker {
        activation: Some(activation),
        _manifest_dir: manifest_dir,
    }
}

struct BuiltWorkerFixture {
    binary_path: PathBuf,
}

impl BuiltWorkerFixture {
    fn binary_path(&self) -> &Path {
        &self.binary_path
    }
}

fn build_fixture_worker() -> BuiltWorkerFixture {
    static FIXTURE_BINARY: OnceLock<PathBuf> = OnceLock::new();
    let binary_path = FIXTURE_BINARY.get_or_init(|| {
        let fixture_dir = fixture_root();
        let target_root =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/worker-plugin-fixture");
        let target_dir = target_root.join("target");
        let manifest = fixture_dir.join("Cargo.toml");
        let status = Command::new("cargo")
            .arg("build")
            .arg("--quiet")
            .arg("--locked")
            .arg("--manifest-path")
            .arg(&manifest)
            .arg("--target-dir")
            .arg(&target_dir)
            .status()
            .expect("fixture worker build should start");
        assert!(status.success(), "fixture worker build should succeed");
        let binary_path = target_dir.join("debug").join(format!(
            "nemo-relay-worker-plugin-fixture{}",
            std::env::consts::EXE_SUFFIX
        ));
        assert!(binary_path.exists(), "fixture worker binary should exist");
        binary_path
    });
    BuiltWorkerFixture {
        binary_path: binary_path.clone(),
    }
}

fn write_manifest(binary: &Path) -> (TempDir, PathBuf) {
    let relay = supported_relay_requirement();
    write_manifest_with_relay(binary, &relay)
}

fn write_manifest_with_relay(binary: &Path, relay: &str) -> (TempDir, PathBuf) {
    write_worker_manifest(
        "fixture_worker",
        relay,
        "rust",
        binary.to_string_lossy().as_ref(),
    )
}

fn write_worker_manifest(
    plugin_id: &str,
    relay: &str,
    runtime: &str,
    entrypoint: &str,
) -> (TempDir, PathBuf) {
    write_manifest_text(&format!(
        r#"
manifest_version = 1

[plugin]
id = {plugin_id}
kind = "worker"

[compat]
relay = {relay}
worker_protocol = "grpc-v1"

[defaults]
enabled = false

[capabilities]
items = ["plugin_worker"]

[load]
runtime = {runtime}
entrypoint = {entrypoint}
"#,
        plugin_id = toml_string(plugin_id),
        relay = toml_string(relay),
        runtime = toml_string(runtime),
        entrypoint = toml_string(entrypoint)
    ))
}

fn write_manifest_text(contents: &str) -> (TempDir, PathBuf) {
    let temp = TempDir::new().expect("manifest tempdir should be created");
    let manifest = temp.path().join("relay-plugin.toml");
    std::fs::write(&manifest, contents).expect("manifest should be written");
    (temp, manifest)
}

fn toml_string(value: &str) -> String {
    format!("{value:?}")
}

fn supported_relay_requirement() -> String {
    format!("={}", env!("CARGO_PKG_VERSION"))
}

fn assert_error_mentions_manifest_relative_entrypoint(
    error: &str,
    manifest_dir: &Path,
    entrypoint: &str,
) {
    let manifest_dir_name = manifest_dir
        .file_name()
        .expect("manifest dir should have a leaf name")
        .to_string_lossy();
    let manifest_dir_pos = error.find(manifest_dir_name.as_ref()).unwrap_or_else(|| {
        panic!("error did not mention manifest dir '{manifest_dir_name}': {error}")
    });
    assert!(
        error[manifest_dir_pos + manifest_dir_name.len()..].contains(entrypoint),
        "error did not mention entrypoint '{entrypoint}' after manifest dir '{manifest_dir_name}': {error}"
    );
}

struct EnvVarGuard {
    key: &'static str,
    previous: Option<String>,
}

impl EnvVarGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var(key).ok();
        // SAFETY: this module serializes worker tests with WORKER_PLUGIN_TEST_LOCK.
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key, previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        // SAFETY: this module serializes worker tests with WORKER_PLUGIN_TEST_LOCK.
        unsafe {
            if let Some(previous) = &self.previous {
                std::env::set_var(self.key, previous);
            } else {
                std::env::remove_var(self.key);
            }
        }
    }
}

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/worker_plugin")
}

fn find_event<'a>(
    events: &'a [Event],
    name: &str,
    scope_category: Option<ScopeCategory>,
) -> &'a Event {
    events
        .iter()
        .find(|event| event.name() == name && event.scope_category() == scope_category)
        .unwrap_or_else(|| panic!("event {name:?} with category {scope_category:?} not found"))
}

fn assert_parent(
    events: &[Event],
    name: &str,
    scope_category: Option<ScopeCategory>,
    expected_parent: Option<Uuid>,
) {
    let event = find_event(events, name, scope_category);
    assert_eq!(event.parent_uuid(), expected_parent);
}

fn assert_not_parent(
    events: &[Event],
    name: &str,
    scope_category: Option<ScopeCategory>,
    excluded_parent: Uuid,
) {
    let event = find_event(events, name, scope_category);
    assert_ne!(event.parent_uuid(), Some(excluded_parent));
}
