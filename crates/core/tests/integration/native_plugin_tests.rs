// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration coverage for SDK-built native dynamic plugins.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use nemo_relay::api::event::{Event, ScopeCategory};
use nemo_relay::api::llm::{
    LlmCallEndParams, LlmCallExecuteParams, LlmCallParams, LlmRequest, LlmStreamCallExecuteParams,
    llm_call, llm_call_end, llm_call_execute, llm_stream_call_execute,
};
use nemo_relay::api::runtime::{
    LlmJsonStream, TASK_SCOPE_STACK, ThreadScopeStackBinding, capture_thread_scope_stack,
    create_scope_stack, restore_thread_scope_stack, set_thread_scope_stack,
};
use nemo_relay::api::scope::{
    EmitMarkEventParams, PopScopeParams, PushScopeParams, ScopeType, event as emit_scope_mark,
    pop_scope, push_scope,
};
use nemo_relay::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use nemo_relay::api::tool::{ToolCallExecuteParams, tool_call_execute, tool_request_intercepts};
use nemo_relay::codec::response::AnnotatedLlmResponse;
use nemo_relay::plugin::dynamic::{
    DynamicPluginActivationSpec, DynamicPluginKind, NativePluginLoadSpec, PluginHostActivation,
    load_native_plugins,
};
use nemo_relay::plugin::{
    ConfigDiagnostic, Plugin, PluginComponentSpec, PluginConfig, PluginRegistrationContext,
    Result as PluginResult, clear_plugin_configuration, deregister_plugin,
    initialize_plugins_exact, list_plugin_kinds, lookup_plugin, register_plugin,
};
use serde_json::{Map, Value as Json, json};
use sha2::{Digest, Sha256};
use tempfile::TempDir;
use tokio::sync::Notify;
use tokio_stream::StreamExt;
use uuid::Uuid;

static NATIVE_PLUGIN_TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

struct ReplacementRegistryPlugin;

const STATIC_BASE_PLUGIN_KIND: &str = "fixture_static_base";
static STATIC_BASE_REGISTRATIONS: AtomicUsize = AtomicUsize::new(0);
static STATIC_BASE_DEREGISTRATIONS: AtomicUsize = AtomicUsize::new(0);

struct StaticBasePlugin;

struct BlockingHostBasePlugin {
    started: Arc<Notify>,
    release: Arc<Notify>,
    registered: Arc<Notify>,
}

impl Plugin for StaticBasePlugin {
    fn plugin_kind(&self) -> &str {
        STATIC_BASE_PLUGIN_KIND
    }

    fn validate(&self, _plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        Vec::new()
    }

    fn register<'a>(
        &'a self,
        _plugin_config: &Map<String, Json>,
        ctx: &'a mut PluginRegistrationContext,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = PluginResult<()>> + Send + 'a>> {
        Box::pin(async move {
            STATIC_BASE_REGISTRATIONS.fetch_add(1, Ordering::SeqCst);
            ctx.add_registration(nemo_relay::plugin::PluginRegistration::new(
                "plugin",
                STATIC_BASE_PLUGIN_KIND,
                Box::new(|| {
                    STATIC_BASE_DEREGISTRATIONS.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                }),
            ));
            Ok(())
        })
    }
}

impl Plugin for BlockingHostBasePlugin {
    fn plugin_kind(&self) -> &str {
        "fixture_blocking_host_base"
    }

    fn validate(&self, _plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        Vec::new()
    }

    fn register<'a>(
        &'a self,
        _plugin_config: &Map<String, Json>,
        ctx: &'a mut PluginRegistrationContext,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = PluginResult<()>> + Send + 'a>> {
        let started = Arc::clone(&self.started);
        let release = Arc::clone(&self.release);
        let registered = Arc::clone(&self.registered);
        Box::pin(async move {
            started.notify_one();
            release.notified().await;
            ctx.add_registration(nemo_relay::plugin::PluginRegistration::new(
                "plugin",
                ctx.qualify_name("blocking-host-base"),
                Box::new(|| Ok(())),
            ));
            registered.notify_one();
            Ok(())
        })
    }
}

impl Plugin for ReplacementRegistryPlugin {
    fn plugin_kind(&self) -> &str {
        "fixture_native"
    }

    fn validate(&self, _plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        Vec::new()
    }

    fn register<'a>(
        &'a self,
        _plugin_config: &Map<String, Json>,
        _ctx: &'a mut PluginRegistrationContext,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = PluginResult<()>> + Send + 'a>> {
        Box::pin(async { Ok(()) })
    }
}

struct FixtureNativeRegistrationCleanup;

impl Drop for FixtureNativeRegistrationCleanup {
    fn drop(&mut self) {
        let _ = deregister_plugin("fixture_native");
    }
}

struct ThreadScopeStackRestore(Option<ThreadScopeStackBinding>);

impl ThreadScopeStackRestore {
    fn capture() -> Self {
        Self(Some(capture_thread_scope_stack()))
    }
}

impl Drop for ThreadScopeStackRestore {
    fn drop(&mut self) {
        if let Some(binding) = self.0.take() {
            restore_thread_scope_stack(binding);
        }
    }
}

struct NativePluginTestCleanup {
    subscriber: Option<&'static str>,
    plugin_configuration_active: bool,
}

impl NativePluginTestCleanup {
    fn new() -> Self {
        Self {
            subscriber: None,
            plugin_configuration_active: false,
        }
    }

    fn mark_plugin_configuration_active(&mut self) {
        self.plugin_configuration_active = true;
    }

    fn mark_subscriber_registered(&mut self, name: &'static str) {
        self.subscriber = Some(name);
    }
}

impl Drop for NativePluginTestCleanup {
    fn drop(&mut self) {
        if let Some(name) = self.subscriber.take() {
            let _ = deregister_subscriber(name);
        }
        if self.plugin_configuration_active {
            let _ = clear_plugin_configuration();
        }
    }
}

#[tokio::test]
async fn sdk_cdylib_registers_tool_request_intercept() {
    let _guard = NATIVE_PLUGIN_TEST_LOCK.lock().await;
    let fixture = build_fixture_plugin();
    let manifest_ref = write_manifest(&fixture);

    let activation = load_native_plugins([NativePluginLoadSpec {
        plugin_id: "fixture_native".into(),
        manifest_ref: manifest_ref.to_string_lossy().into_owned(),
    }])
    .expect("native plugin should load");
    let mut cleanup = NativePluginTestCleanup::new();

    let mut plugin_config = PluginConfig::default();
    plugin_config.components.push(PluginComponentSpec {
        kind: "fixture_native".into(),
        enabled: true,
        config: Map::new(),
    });
    initialize_plugins_exact(plugin_config)
        .await
        .expect("native plugin should initialize");
    cleanup.mark_plugin_configuration_active();

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    register_subscriber(
        "native_plugin_fixture_events",
        Arc::new(move |event| {
            captured.lock().unwrap().push(event.clone());
        }),
    )
    .expect("test subscriber should register");
    cleanup.mark_subscriber_registered("native_plugin_fixture_events");

    let stack = create_scope_stack();
    let (outer_uuid, rewritten, tool_result) = TASK_SCOPE_STACK
        .scope(stack, async {
            let outer = push_scope(
                PushScopeParams::builder()
                    .name("native-plugin-test-outer")
                    .scope_type(ScopeType::Agent)
                    .build(),
            )
            .expect("outer scope should push");
            let outer_uuid = outer.uuid;
            let rewritten = tool_request_intercepts("demo_tool", json!({ "input": "value" }))
                .expect("native request intercept should run");
            let tool_result = tool_call_execute(
                ToolCallExecuteParams::builder()
                    .name("native-fixture-tool")
                    .args(json!({ "input": "execute" }))
                    .func(Arc::new(|args| {
                        Box::pin(async move { Ok(json!({ "tool_callback": true, "args": args })) })
                    }))
                    .build(),
            )
            .await
            .expect("native tool middleware should run");
            pop_scope(PopScopeParams::builder().handle_uuid(&outer.uuid).build())
                .expect("outer scope should pop");
            (outer_uuid, rewritten, tool_result)
        })
        .await;
    assert_eq!(rewritten["input"], "value");
    assert_eq!(rewritten["native_plugin"], true);
    assert_eq!(tool_result["tool_callback"], true);
    assert_eq!(tool_result["native_plugin_tool_execution"], true);
    assert_eq!(
        tool_result["args"]["native_plugin_tool_execution_request"],
        true
    );
    assert!(tool_result.get("pending_marks").is_none());

    flush_subscribers().expect("native fixture events should flush");
    let first_events = events.lock().unwrap().clone();
    find_event(&first_events, "fixture.native.subscriber.mark", None);
    assert_parent(&first_events, "fixture.native.mark", None, Some(outer_uuid));
    assert_eq!(
        find_event(&first_events, "fixture.native.mark", None)
            .metadata()
            .unwrap()["native_plugin_mark"],
        true
    );
    assert_parent(
        &first_events,
        "fixture.native.scope",
        Some(ScopeCategory::Start),
        Some(outer_uuid),
    );
    assert_eq!(
        find_event(
            &first_events,
            "fixture.native.scope",
            Some(ScopeCategory::Start),
        )
        .metadata()
        .unwrap()["native_plugin_scope_start"],
        true
    );
    assert_eq!(
        find_event(
            &first_events,
            "fixture.native.scope",
            Some(ScopeCategory::End),
        )
        .metadata()
        .unwrap()["native_plugin_scope_end"],
        true
    );
    assert_not_parent(
        &first_events,
        "fixture.native.isolated.mark",
        None,
        outer_uuid,
    );
    assert_not_parent(
        &first_events,
        "fixture.native.isolated.scope",
        Some(ScopeCategory::Start),
        outer_uuid,
    );
    let tool_start = find_event(
        &first_events,
        "native-fixture-tool",
        Some(ScopeCategory::Start),
    );
    assert_eq!(
        tool_start.input().unwrap()["native_plugin_tool_sanitize_request"],
        true
    );
    assert_eq!(
        tool_start.metadata().unwrap()["native_plugin_scope_start"],
        true
    );
    let tool_end = find_event(
        &first_events,
        "native-fixture-tool",
        Some(ScopeCategory::End),
    );
    assert_eq!(
        tool_end.output().unwrap()["native_plugin_tool_sanitize_response"],
        true
    );
    assert_eq!(
        tool_end.metadata().unwrap()["native_plugin_scope_end"],
        true
    );
    assert!(tool_end.output().unwrap().get("pending_marks").is_none());
    let tool_mark = find_event(&first_events, "fixture.native.tool_execution.mark", None);
    assert_eq!(tool_mark.parent_uuid(), Some(tool_start.uuid()));
    assert_eq!(
        tool_mark.category().map(|category| category.as_str()),
        Some("custom")
    );
    assert_eq!(
        tool_mark
            .category_profile()
            .and_then(|profile| profile.subtype.as_deref()),
        Some("fixture.native.tool_execution")
    );
    assert_eq!(tool_mark.data().unwrap()["source"], "native_tool_execution");
    assert_eq!(tool_mark.metadata().unwrap()["fixture"], true);
    assert_eq!(tool_mark.metadata().unwrap()["native_plugin_mark"], true);
    assert!(tool_mark.timestamp() > tool_end.timestamp());
    let tool_end_index = first_events
        .iter()
        .position(|event| {
            event.name() == "native-fixture-tool"
                && event.scope_category() == Some(ScopeCategory::End)
        })
        .unwrap();
    let tool_mark_index = first_events
        .iter()
        .position(|event| event.name() == "fixture.native.tool_execution.mark")
        .unwrap();
    assert!(tool_end_index < tool_mark_index);

    events.lock().unwrap().clear();
    let isolated_next_stack = create_scope_stack();
    let isolated_next_outer_uuid = TASK_SCOPE_STACK
        .scope(isolated_next_stack, async {
            let outer = push_scope(
                PushScopeParams::builder()
                    .name("native-plugin-test-isolated-next-outer")
                    .scope_type(ScopeType::Agent)
                    .build(),
            )
            .expect("isolated next outer scope should push");
            let outer_uuid = outer.uuid;
            let result = tool_call_execute(
                ToolCallExecuteParams::builder()
                    .name("native-fixture-tool-isolated-next")
                    .args(json!({
                        "input": "isolated-next",
                        "use_isolated_next": true
                    }))
                    .func(Arc::new(|_args| {
                        Box::pin(async move {
                            emit_scope_mark(
                                EmitMarkEventParams::builder()
                                    .name("native-fixture-tool-callback-mark")
                                    .build(),
                            )?;
                            Ok(json!({ "tool_callback": true }))
                        })
                    }))
                    .build(),
            )
            .await
            .expect("native isolated next middleware should run");
            assert_eq!(result["tool_callback"], true);
            assert_eq!(result["native_plugin_tool_execution"], true);
            pop_scope(PopScopeParams::builder().handle_uuid(&outer.uuid).build())
                .expect("isolated next outer scope should pop");
            outer_uuid
        })
        .await;
    flush_subscribers().expect("isolated next native fixture events should flush");
    let isolated_next_events = events.lock().unwrap().clone();
    let isolated_next_scope = find_event(
        &isolated_next_events,
        "fixture.native.isolated.next",
        Some(ScopeCategory::Start),
    );
    let callback_mark = find_event(
        &isolated_next_events,
        "native-fixture-tool-callback-mark",
        None,
    );
    assert_eq!(
        callback_mark.parent_uuid(),
        Some(isolated_next_scope.uuid())
    );
    assert_ne!(
        callback_mark.parent_uuid(),
        Some(isolated_next_outer_uuid),
        "native next callback should use the plugin-selected isolated stack"
    );

    events.lock().unwrap().clear();
    {
        let thread_stack = create_scope_stack();
        let _thread_stack_restore = ThreadScopeStackRestore::capture();
        set_thread_scope_stack(thread_stack);
        let thread_outer = push_scope(
            PushScopeParams::builder()
                .name("native-plugin-test-thread-outer")
                .scope_type(ScopeType::Agent)
                .build(),
        )
        .expect("thread outer scope should push");
        let thread_outer_uuid = thread_outer.uuid;
        let rewritten = tool_request_intercepts("demo_tool", json!({ "input": "thread" }))
            .expect("native request intercept should run with thread stack");
        assert_eq!(rewritten["native_plugin"], true);
        pop_scope(
            PopScopeParams::builder()
                .handle_uuid(&thread_outer.uuid)
                .build(),
        )
        .expect("thread outer scope should pop");
        flush_subscribers().expect("thread-stack native fixture events should flush");
        let thread_events = events.lock().unwrap().clone();
        assert_parent(
            &thread_events,
            "fixture.native.mark",
            None,
            Some(thread_outer_uuid),
        );
        assert_not_parent(
            &thread_events,
            "fixture.native.thread_stack.mark",
            None,
            thread_outer_uuid,
        );
    }

    events.lock().unwrap().clear();
    let llm_execute_response = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("native-fixture-llm-execute")
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
    .expect("native LLM middleware should run");
    assert_eq!(llm_execute_response["llm_callback"], true);
    assert_eq!(llm_execute_response["native_plugin_llm_execution"], true);
    assert_eq!(
        llm_execute_response["request"]["native_plugin_llm_execution_request"],
        true
    );
    flush_subscribers().expect("managed LLM native fixture events should flush");
    let managed_llm_events = events.lock().unwrap().clone();
    let llm_start = find_event(
        &managed_llm_events,
        "native-fixture-llm-execute",
        Some(ScopeCategory::Start),
    );
    assert_eq!(
        llm_start.input().unwrap()["content"]["native_plugin_llm_sanitize_request"],
        true
    );
    assert_eq!(
        llm_start.input().unwrap()["content"]["native_plugin_llm_request_intercept"],
        true
    );
    let pending_mark = find_event(&managed_llm_events, "fixture.native.llm_request.mark", None);
    assert_eq!(pending_mark.parent_uuid(), Some(llm_start.uuid()));
    assert_eq!(
        pending_mark.category().map(|category| category.as_str()),
        Some("custom")
    );
    assert_eq!(
        pending_mark
            .category_profile()
            .and_then(|profile| profile.subtype.as_deref()),
        Some("fixture.native.pending")
    );
    assert_eq!(
        pending_mark.data().unwrap()["source"],
        "native_request_intercept"
    );
    assert_eq!(pending_mark.metadata().unwrap()["fixture"], true);
    assert!(pending_mark.timestamp() > llm_start.timestamp());
    let llm_end = find_event(
        &managed_llm_events,
        "native-fixture-llm-execute",
        Some(ScopeCategory::End),
    );
    assert_eq!(
        llm_end.output().unwrap()["native_plugin_llm_sanitize_response"],
        true
    );
    assert!(llm_end.annotated_response().is_none());

    events.lock().unwrap().clear();
    let collected_stream_chunks = Arc::new(Mutex::new(Vec::<Json>::new()));
    let collector_chunks = collected_stream_chunks.clone();
    let finalizer_chunks = collected_stream_chunks.clone();
    let mut stream = llm_stream_call_execute(
        LlmStreamCallExecuteParams::builder()
            .name("native-fixture-llm-stream")
            .request(LlmRequest {
                headers: Map::new(),
                content: json!({ "prompt": "stream" }),
            })
            .func(Arc::new(|request| {
                Box::pin(async move {
                    Ok(Box::pin(tokio_stream::iter(vec![
                        Ok(json!({
                            "stream_chunk": 1,
                            "request": request.content,
                        })),
                        Ok(json!({ "stream_chunk": 2 })),
                    ])) as LlmJsonStream)
                })
            }))
            .collector(Box::new(move |chunk| {
                collector_chunks.lock().unwrap().push(chunk);
                Ok(())
            }))
            .finalizer(Box::new(move || {
                Json::Array(finalizer_chunks.lock().unwrap().clone())
            }))
            .build(),
    )
    .await
    .expect("native LLM stream middleware should run");
    let mut stream_chunks = Vec::new();
    while let Some(chunk) = stream.next().await {
        stream_chunks.push(chunk.expect("native stream chunk should succeed"));
    }
    assert_eq!(stream_chunks.len(), 2);
    assert_eq!(
        stream_chunks[0]["request"]["native_plugin_llm_stream_execution_request"],
        true
    );
    assert_eq!(stream_chunks[0]["native_plugin_llm_stream_execution"], true);
    assert_eq!(stream_chunks[1]["native_plugin_llm_stream_execution"], true);
    assert_eq!(*collected_stream_chunks.lock().unwrap(), stream_chunks);
    flush_subscribers().expect("stream native fixture events should flush");
    let stream_events = events.lock().unwrap().clone();
    let stream_start = find_event(
        &stream_events,
        "native-fixture-llm-stream",
        Some(ScopeCategory::Start),
    );
    let stream_pending_mark = find_event(&stream_events, "fixture.native.llm_request.mark", None);
    assert_eq!(stream_pending_mark.parent_uuid(), Some(stream_start.uuid()));
    let stream_end = find_event(
        &stream_events,
        "native-fixture-llm-stream",
        Some(ScopeCategory::End),
    );
    assert_eq!(
        stream_end.output().unwrap()[0]["native_plugin_llm_stream_execution"],
        true
    );

    events.lock().unwrap().clear();
    let llm_request = LlmRequest {
        headers: Map::new(),
        content: json!({ "prompt": "hello" }),
    };
    let handle = llm_call(
        LlmCallParams::builder()
            .name("native-fixture-llm")
            .request(&llm_request)
            .build(),
    )
    .expect("llm start should emit");
    let mut extra = Map::new();
    extra.insert("preexisting_annotation".into(), json!("kept"));
    llm_call_end(
        LlmCallEndParams::builder()
            .handle(&handle)
            .response(json!({ "id": "response-from-test", "content": "done" }))
            .annotated_response(Arc::new(AnnotatedLlmResponse {
                id: Some("annotation-before-plugin".into()),
                model: None,
                message: None,
                tool_calls: None,
                finish_reason: None,
                usage: None,
                optimization_summary: None,
                api_specific: None,
                extra,
            }))
            .build(),
    )
    .expect("llm end should emit");
    flush_subscribers().expect("llm response annotation event should flush");
    let llm_events = events.lock().unwrap().clone();
    let llm_end = find_event(&llm_events, "native-fixture-llm", Some(ScopeCategory::End));
    let annotated = llm_end
        .annotated_response()
        .expect("native plugin should preserve response annotation");
    assert_eq!(annotated.id.as_deref(), Some("annotation-before-plugin"));
    assert_eq!(annotated.extra["preexisting_annotation"], json!("kept"));
    assert!(annotated.extra.get("native_plugin_annotation").is_none());

    drop(cleanup);
    activation.clear();
}

#[tokio::test]
async fn native_validation_diagnostics_prevent_initialization() {
    let _guard = NATIVE_PLUGIN_TEST_LOCK.lock().await;
    let fixture = build_fixture_plugin();
    let manifest_ref = write_manifest(&fixture);

    let activation = load_native_plugins([NativePluginLoadSpec {
        plugin_id: "fixture_native".into(),
        manifest_ref: manifest_ref.to_string_lossy().into_owned(),
    }])
    .expect("native plugin should load");

    let mut plugin_config = PluginConfig::default();
    plugin_config.components.push(PluginComponentSpec {
        kind: "fixture_native".into(),
        enabled: true,
        config: Map::from_iter([("reject".into(), json!(true))]),
    });
    let error = initialize_plugins_exact(plugin_config)
        .await
        .expect_err("validation diagnostics should prevent initialization")
        .to_string();
    assert!(error.contains("fixture rejection requested"), "{error}");

    clear_plugin_configuration().expect("native plugin config should clear");
    activation.clear();
}

#[tokio::test]
async fn native_tool_execution_rejects_null_malformed_and_error_outcomes() {
    let _guard = NATIVE_PLUGIN_TEST_LOCK.lock().await;
    let fixture = build_fixture_plugin();
    let manifest_ref =
        write_manifest_with_symbol(&fixture, "nemo_relay_fixture_tool_outcome_errors");
    let activation = load_native_plugins([load_spec("fixture_native", &manifest_ref)])
        .expect("raw native outcome fixture should load");
    let mut cleanup = NativePluginTestCleanup::new();

    let mut plugin_config = PluginConfig::default();
    plugin_config.components.push(PluginComponentSpec {
        kind: "fixture_native".into(),
        enabled: true,
        config: Map::new(),
    });
    initialize_plugins_exact(plugin_config)
        .await
        .expect("raw native outcome fixture should initialize");
    cleanup.mark_plugin_configuration_active();

    for (name, expected) in [
        (
            "fixture-null-outcome",
            "native tool execution returned null outcome",
        ),
        (
            "fixture-malformed-outcome",
            "invalid native tool execution outcome JSON",
        ),
        (
            "fixture-status-error-outcome",
            "fixture tool execution failed",
        ),
    ] {
        let error = tool_call_execute(
            ToolCallExecuteParams::builder()
                .name(name)
                .args(json!({ "input": true }))
                .func(Arc::new(|args| Box::pin(async move { Ok(args) })))
                .build(),
        )
        .await
        .expect_err("invalid native tool outcome should fail")
        .to_string();
        assert!(
            error.contains(expected),
            "expected {expected:?} in {error:?}"
        );
    }

    let result = tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("fixture-valid-outcome")
            .args(json!({ "input": true }))
            .func(Arc::new(|args| Box::pin(async move { Ok(args) })))
            .build(),
    )
    .await
    .expect("native loader should remain usable after rejected outcomes");
    assert_eq!(result["raw_tool_outcome"], true);
    assert!(result.get("pending_marks").is_none());

    drop(cleanup);
    activation.clear();
}

#[test]
fn native_loader_rejects_missing_library() {
    let _guard = NATIVE_PLUGIN_TEST_LOCK.blocking_lock();
    let manifest_dir = TempDir::new().expect("manifest dir");
    let missing_library = manifest_dir.path().join("libmissing_native_plugin.so");
    let manifest_ref = write_manifest_text(ManifestOptions {
        manifest_dir: manifest_dir.path(),
        plugin_id: "fixture_native",
        relay: &format!("={}", env!("CARGO_PKG_VERSION")),
        library: &missing_library.to_string_lossy(),
        symbol: "nemo_relay_fixture_native_plugin",
        integrity: None,
    });

    let error = expect_native_load_error(
        NativePluginLoadSpec {
            plugin_id: "fixture_native".into(),
            manifest_ref: manifest_ref.to_string_lossy().into_owned(),
        },
        "missing library should fail",
    );
    assert!(error.contains("does not exist"), "{error}");
}

#[test]
fn native_loader_returns_empty_activation_for_empty_specs() {
    let activation =
        load_native_plugins(std::iter::empty::<NativePluginLoadSpec>()).expect("empty load");
    assert!(activation.is_empty());
}

#[test]
fn native_activation_clear_deregisters_plugin_kind() {
    let _guard = NATIVE_PLUGIN_TEST_LOCK.blocking_lock();
    let fixture = build_fixture_plugin();
    let manifest_ref = write_manifest(&fixture);

    let activation = load_native_plugins([load_spec("fixture_native", &manifest_ref)])
        .expect("native plugin should load");
    assert!(!activation.is_empty());
    activation.clear();

    let activation = load_native_plugins([load_spec("fixture_native", &manifest_ref)])
        .expect("native plugin should reload after activation clear");
    activation.clear();
}

#[test]
fn native_loader_resolves_manifest_directory_and_relative_library_paths() {
    let _guard = NATIVE_PLUGIN_TEST_LOCK.blocking_lock();
    let fixture = build_fixture_plugin();
    let relative_dir = fixture.manifest_dir.path().join("lib");
    std::fs::create_dir_all(&relative_dir).expect("relative lib dir");
    let relative_library = Path::new("lib").join(fixture_library_name());
    std::fs::copy(
        &fixture.library_path,
        fixture.manifest_dir.path().join(&relative_library),
    )
    .expect("copy fixture library");
    write_manifest_text(ManifestOptions {
        manifest_dir: fixture.manifest_dir.path(),
        plugin_id: "fixture_native",
        relay: &format!("={}", env!("CARGO_PKG_VERSION")),
        library: &relative_library.to_string_lossy(),
        symbol: "nemo_relay_fixture_native_plugin",
        integrity: None,
    });

    let activation = load_native_plugins([NativePluginLoadSpec {
        plugin_id: "fixture_native".into(),
        manifest_ref: fixture.manifest_dir.path().to_string_lossy().into_owned(),
    }])
    .expect("native plugin should load from manifest directory");
    activation.clear();
}

#[test]
fn native_loader_rolls_back_partially_loaded_plugins() {
    let _guard = NATIVE_PLUGIN_TEST_LOCK.blocking_lock();
    let fixture = build_fixture_plugin();
    let valid_manifest = write_manifest(&fixture);
    let missing_manifest_dir = TempDir::new().expect("missing manifest dir");
    let missing_library = missing_manifest_dir
        .path()
        .join("libmissing_native_plugin.so");
    let missing_manifest = write_manifest_text(ManifestOptions {
        manifest_dir: missing_manifest_dir.path(),
        plugin_id: "fixture_native_missing",
        relay: &format!("={}", env!("CARGO_PKG_VERSION")),
        library: &missing_library.to_string_lossy(),
        symbol: "nemo_relay_fixture_native_plugin",
        integrity: None,
    });

    let error = expect_native_load_error_from_specs(
        [
            load_spec("fixture_native", &valid_manifest),
            load_spec("fixture_native_missing", &missing_manifest),
        ],
        "partial load failure should fail",
    );
    assert!(error.contains("does not exist"), "{error}");

    let activation = load_native_plugins([load_spec("fixture_native", &valid_manifest)])
        .expect("first plugin kind should be deregistered after rollback");
    activation.clear();
}

#[test]
fn native_loader_rejects_unsupported_relay_requirement_before_loading() {
    let _guard = NATIVE_PLUGIN_TEST_LOCK.blocking_lock();
    let manifest_dir = TempDir::new().expect("manifest dir");
    let manifest_ref = write_manifest_text(ManifestOptions {
        manifest_dir: manifest_dir.path(),
        plugin_id: "fixture_native",
        relay: ">=1.0,<2.0",
        library: "libdoes-not-need-to-exist.so",
        symbol: "nemo_relay_fixture_native_plugin",
        integrity: None,
    });

    let error = expect_native_load_error(
        NativePluginLoadSpec {
            plugin_id: "fixture_native".into(),
            manifest_ref: manifest_ref.to_string_lossy().into_owned(),
        },
        "unsupported relay requirement should fail",
    );
    assert!(error.contains("requires relay"), "{error}");
}

#[test]
fn native_loader_rejects_manifest_contract_errors_before_loading_library() {
    let _guard = NATIVE_PLUGIN_TEST_LOCK.blocking_lock();
    let manifest_dir = TempDir::new().expect("manifest dir");

    let mismatched_id = write_raw_manifest(
        manifest_dir.path(),
        &native_manifest_text(
            "fixture_manifest_id",
            &format!("={}", env!("CARGO_PKG_VERSION")),
            "1",
            "libdoes-not-need-to-exist.so",
            "nemo_relay_fixture_native_plugin",
        ),
    );
    let error = expect_native_load_error(
        NativePluginLoadSpec {
            plugin_id: "fixture_expected_id".into(),
            manifest_ref: mismatched_id.to_string_lossy().into_owned(),
        },
        "manifest id mismatch should fail",
    );
    assert!(error.contains("does not match expected id"), "{error}");

    let invalid_relay = write_raw_manifest(
        manifest_dir.path(),
        &native_manifest_text(
            "fixture_native",
            "not a version requirement",
            "1",
            "libdoes-not-need-to-exist.so",
            "nemo_relay_fixture_native_plugin",
        ),
    );
    let error = expect_native_load_error(
        NativePluginLoadSpec {
            plugin_id: "fixture_native".into(),
            manifest_ref: invalid_relay.to_string_lossy().into_owned(),
        },
        "invalid relay requirement should fail",
    );
    assert!(
        error.contains("invalid compat.relay version requirement"),
        "{error}"
    );

    let unsupported_native_api = write_raw_manifest(
        manifest_dir.path(),
        &native_manifest_text(
            "fixture_native",
            &format!("={}", env!("CARGO_PKG_VERSION")),
            "2",
            "libdoes-not-need-to-exist.so",
            "nemo_relay_fixture_native_plugin",
        ),
    );
    let error = expect_native_load_error(
        NativePluginLoadSpec {
            plugin_id: "fixture_native".into(),
            manifest_ref: unsupported_native_api.to_string_lossy().into_owned(),
        },
        "unsupported native API should fail",
    );
    assert!(error.contains("unsupported compat.native_api"), "{error}");

    let worker_manifest = write_raw_manifest(
        manifest_dir.path(),
        r#"
manifest_version = 1

[plugin]
id = "fixture_worker"
kind = "worker"

[compat]
relay = ">=0.5,<1.0"
worker_protocol = "grpc-v1"

[defaults]
enabled = false

[capabilities]
items = ["plugin_worker"]

[load]
runtime = "python"
entrypoint = "fixture.worker:create_plugin"
"#,
    );
    let error = expect_native_load_error(
        NativePluginLoadSpec {
            plugin_id: "fixture_worker".into(),
            manifest_ref: worker_manifest.to_string_lossy().into_owned(),
        },
        "worker manifest should fail native loading",
    );
    assert!(error.contains("only supports rust_dynamic"), "{error}");
}

#[test]
fn native_manifest_writer_escapes_toml_strings() {
    let _guard = NATIVE_PLUGIN_TEST_LOCK.blocking_lock();
    let manifest_dir = TempDir::new().expect("manifest dir");
    let windows_style_library =
        r"C:\Users\RUNNER~1\AppData\Local\Temp\.tmpPath\debug\nemo_relay_plugin_fixture.dll";
    let manifest_ref = write_manifest_text(ManifestOptions {
        manifest_dir: manifest_dir.path(),
        plugin_id: "fixture_native",
        relay: &format!("={}", env!("CARGO_PKG_VERSION")),
        library: windows_style_library,
        symbol: "nemo_relay_fixture_native_plugin",
        integrity: Some(r"sha256:abc\def"),
    });

    let manifest = std::fs::read_to_string(manifest_ref).expect("read relay-plugin.toml");
    let parsed: toml::Value = toml::from_str(&manifest).expect("manifest should parse");
    assert_eq!(
        parsed["load"]["library"].as_str(),
        Some(windows_style_library)
    );
    assert_eq!(
        parsed["integrity"]["sha256"].as_str(),
        Some(r"sha256:abc\def")
    );
}

#[test]
fn native_loader_rejects_missing_symbol_digest_mismatch_and_kind_mismatch() {
    let _guard = NATIVE_PLUGIN_TEST_LOCK.blocking_lock();
    let fixture = build_fixture_plugin();

    let missing_symbol = write_manifest_with_symbol(&fixture, "missing_native_symbol");
    let error = expect_native_load_error(
        NativePluginLoadSpec {
            plugin_id: "fixture_native".into(),
            manifest_ref: missing_symbol.to_string_lossy().into_owned(),
        },
        "missing symbol should fail",
    );
    assert!(error.contains("symbol"), "{error}");

    let digest_match = write_manifest_with_integrity(&fixture, &sha256(&fixture.library_path));
    let activation = load_native_plugins([NativePluginLoadSpec {
        plugin_id: "fixture_native".into(),
        manifest_ref: digest_match.to_string_lossy().into_owned(),
    }])
    .expect("matching digest should load");
    activation.clear();

    let digest_mismatch = write_manifest_with_integrity(&fixture, "sha256:deadbeef");
    let error = expect_native_load_error(
        NativePluginLoadSpec {
            plugin_id: "fixture_native".into(),
            manifest_ref: digest_mismatch.to_string_lossy().into_owned(),
        },
        "digest mismatch should fail",
    );
    assert!(error.contains("sha256 mismatch"), "{error}");

    let wrong_kind = write_manifest_with_plugin_id(&fixture, "fixture_native_mismatch");
    let error = expect_native_load_error(
        NativePluginLoadSpec {
            plugin_id: "fixture_native_mismatch".into(),
            manifest_ref: wrong_kind.to_string_lossy().into_owned(),
        },
        "plugin kind mismatch should fail",
    );
    assert!(error.contains("returned kind"), "{error}");
}

#[test]
fn native_loader_rejects_entry_and_descriptor_failures() {
    let _guard = NATIVE_PLUGIN_TEST_LOCK.blocking_lock();
    let fixture = build_fixture_plugin();

    for (symbol, expected) in [
        ("nemo_relay_fixture_entry_error", "fixture entry failed"),
        (
            "nemo_relay_fixture_small_descriptor",
            "incompatible plugin descriptor size",
        ),
        ("nemo_relay_fixture_null_kind", "null plugin_kind"),
        ("nemo_relay_fixture_no_register", "no register callback"),
    ] {
        let manifest_ref = write_manifest_with_symbol(&fixture, symbol);
        let error = expect_native_load_error(
            load_spec("fixture_native", &manifest_ref),
            "invalid native descriptor should fail",
        );
        assert!(
            error.contains(expected),
            "expected {expected:?} in error: {error}"
        );
    }
}

#[tokio::test]
async fn native_validate_and_register_callback_errors_are_reported() {
    let _guard = NATIVE_PLUGIN_TEST_LOCK.lock().await;
    let fixture = build_fixture_plugin();

    for (symbol, expected) in [
        (
            "nemo_relay_fixture_validate_error",
            "fixture validate failed",
        ),
        (
            "nemo_relay_fixture_invalid_diagnostics",
            "invalid diagnostics JSON",
        ),
        (
            "nemo_relay_fixture_register_error",
            "fixture register failed",
        ),
    ] {
        let manifest_ref = write_manifest_with_symbol(&fixture, symbol);
        let activation = load_native_plugins([load_spec("fixture_native", &manifest_ref)])
            .expect("native plugin should load");
        let error = initialize_fixture_native(Map::new())
            .await
            .expect_err("native plugin initialization should fail")
            .to_string();
        assert!(
            error.contains(expected),
            "expected {expected:?} in error: {error}"
        );
        clear_plugin_configuration().expect("native plugin config should clear");
        activation.clear();
    }
}

#[tokio::test]
async fn plugin_host_activation_owns_configuration_until_clear() {
    let _guard = NATIVE_PLUGIN_TEST_LOCK.lock().await;
    let fixture = build_fixture_plugin();
    let manifest_ref = write_manifest(&fixture);
    let (activation, report) = PluginHostActivation::activate(
        PluginConfig::default(),
        [host_spec("fixture_native", &manifest_ref)],
    )
    .await
    .expect("plugin host should activate");

    assert!(activation.is_active());
    assert!(!report.has_errors());
    assert!(
        list_plugin_kinds()
            .iter()
            .any(|kind| kind == "fixture_native")
    );
    let rewritten = tool_request_intercepts("host-owned-tool", json!({ "input": true }))
        .expect("host-owned intercept should run");
    assert_eq!(rewritten["native_plugin"], true);

    let initialize_error = initialize_plugins_exact(PluginConfig::default())
        .await
        .expect_err("legacy initialize must not replace a host activation")
        .to_string();
    assert!(initialize_error.contains("active dynamic plugin host"));
    let clear_error = clear_plugin_configuration()
        .expect_err("legacy clear must not clear a host activation")
        .to_string();
    assert!(clear_error.contains("active dynamic plugin host"));

    activation.clear().expect("plugin host should clear");
    assert!(
        !list_plugin_kinds()
            .iter()
            .any(|kind| kind == "fixture_native")
    );
    let unchanged = tool_request_intercepts("host-owned-tool", json!({ "input": true }))
        .expect("cleared intercept chain should be empty");
    assert_eq!(unchanged, json!({ "input": true }));
}

#[tokio::test]
async fn plugin_host_activation_combines_static_base_and_dynamic_components() {
    let _guard = NATIVE_PLUGIN_TEST_LOCK.lock().await;
    let _ = deregister_plugin(STATIC_BASE_PLUGIN_KIND);
    STATIC_BASE_REGISTRATIONS.store(0, Ordering::SeqCst);
    STATIC_BASE_DEREGISTRATIONS.store(0, Ordering::SeqCst);
    register_plugin(Arc::new(StaticBasePlugin)).expect("static base plugin should register");

    let fixture = build_fixture_plugin();
    let manifest_ref = write_manifest(&fixture);
    let mut base_config = PluginConfig::default();
    base_config.components.push(PluginComponentSpec {
        kind: STATIC_BASE_PLUGIN_KIND.into(),
        enabled: true,
        config: Map::new(),
    });
    let (activation, report) =
        PluginHostActivation::activate(base_config, [host_spec("fixture_native", &manifest_ref)])
            .await
            .expect("static and dynamic components should activate together");

    assert!(!report.has_errors());
    assert_eq!(STATIC_BASE_REGISTRATIONS.load(Ordering::SeqCst), 1);
    assert!(lookup_plugin(STATIC_BASE_PLUGIN_KIND).is_some());
    assert!(lookup_plugin("fixture_native").is_some());

    activation.clear().expect("combined host should clear");
    assert_eq!(STATIC_BASE_DEREGISTRATIONS.load(Ordering::SeqCst), 1);
    assert!(lookup_plugin(STATIC_BASE_PLUGIN_KIND).is_some());
    assert!(lookup_plugin("fixture_native").is_none());
    assert!(deregister_plugin(STATIC_BASE_PLUGIN_KIND));
}

#[tokio::test]
async fn plugin_host_clear_allows_an_in_flight_native_callback_to_finish() {
    let _guard = NATIVE_PLUGIN_TEST_LOCK.lock().await;
    let fixture = build_fixture_plugin();
    let manifest_ref = write_manifest(&fixture);
    let (activation, _) = PluginHostActivation::activate(
        PluginConfig::default(),
        [host_spec("fixture_native", &manifest_ref)],
    )
    .await
    .expect("plugin host should activate");

    let (entered_tx, entered_rx) = std::sync::mpsc::sync_channel(1);
    let (release_tx, release_rx) = std::sync::mpsc::sync_channel(1);
    let release_rx = Arc::new(Mutex::new(release_rx));
    let call_thread = std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("in-flight callback runtime should build");
        runtime.block_on(tool_call_execute(
            ToolCallExecuteParams::builder()
                .name("native-fixture-in-flight")
                .args(json!({ "input": "in-flight" }))
                .func(Arc::new(move |args| {
                    let entered_tx = entered_tx.clone();
                    let release_rx = Arc::clone(&release_rx);
                    Box::pin(async move {
                        entered_tx.send(()).map_err(|error| {
                            nemo_relay::error::FlowError::Internal(error.to_string())
                        })?;
                        release_rx
                            .lock()
                            .map_err(|error| {
                                nemo_relay::error::FlowError::Internal(error.to_string())
                            })?
                            .recv()
                            .map_err(|error| {
                                nemo_relay::error::FlowError::Internal(error.to_string())
                            })?;
                        Ok(json!({ "tool_callback": true, "args": args }))
                    })
                }))
                .build(),
        ))
    });

    entered_rx
        .recv_timeout(std::time::Duration::from_secs(10))
        .expect("native callback should enter its continuation");
    activation
        .clear()
        .expect("host should clear while a callback snapshot remains in flight");
    let unchanged = tool_request_intercepts("after-clear", json!({ "input": true }))
        .expect("new calls should observe the cleared registries");
    assert_eq!(unchanged, json!({ "input": true }));

    release_tx
        .send(())
        .expect("in-flight continuation should still be reachable");
    let result = call_thread
        .join()
        .expect("in-flight callback thread should not panic")
        .expect("in-flight callback should finish after host clear");
    assert_eq!(result["tool_callback"], true);
    assert_eq!(result["native_plugin_tool_execution"], true);
}

#[tokio::test]
async fn plugin_host_activation_cleans_up_after_caller_cancellation() {
    let _guard = NATIVE_PLUGIN_TEST_LOCK.lock().await;
    let fixture = build_fixture_plugin();
    let manifest_ref = write_manifest(&fixture);
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let registered = Arc::new(Notify::new());
    register_plugin(Arc::new(BlockingHostBasePlugin {
        started: Arc::clone(&started),
        release: Arc::clone(&release),
        registered: Arc::clone(&registered),
    }))
    .expect("blocking base plugin should register");

    let caller = tokio::spawn(PluginHostActivation::activate(
        PluginConfig {
            components: vec![PluginComponentSpec::new("fixture_blocking_host_base")],
            ..PluginConfig::default()
        },
        [host_spec("fixture_native", &manifest_ref)],
    ));
    started.notified().await;
    caller.abort();
    match caller.await {
        Err(error) => assert!(error.is_cancelled()),
        Ok(_) => panic!("activation caller should be canceled"),
    }
    release.notify_one();
    registered.notified().await;

    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            if nemo_relay::plugin::active_plugin_report().is_none()
                && lookup_plugin("fixture_native").is_none()
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("canceled activation should clear its completed host result");

    let (activation, _) = PluginHostActivation::activate(
        PluginConfig::default(),
        [host_spec("fixture_native", &manifest_ref)],
    )
    .await
    .expect("canceled activation must release process ownership");
    activation.clear().expect("recovered host should clear");
    assert!(deregister_plugin("fixture_blocking_host_base"));
}

#[tokio::test]
async fn plugin_host_rejects_empty_dynamic_specs_without_claiming_ownership() {
    let _guard = NATIVE_PLUGIN_TEST_LOCK.lock().await;
    let error = match PluginHostActivation::activate(
        PluginConfig::default(),
        Vec::<DynamicPluginActivationSpec>::new(),
    )
    .await
    {
        Ok((activation, _)) => {
            activation
                .clear()
                .expect("unexpected empty activation should clear");
            panic!("an empty dynamic activation must fail");
        }
        Err(error) => error.to_string(),
    };
    assert!(error.contains("at least one dynamic plugin"), "{error}");
    assert!(error.contains("static-only configuration"), "{error}");

    initialize_plugins_exact(PluginConfig::default())
        .await
        .expect("empty dynamic rejection must leave static initialization available");
    clear_plugin_configuration().expect("static configuration should clear");
}

#[tokio::test]
async fn plugin_host_activation_drop_releases_owner_and_plugin_kind() {
    let _guard = NATIVE_PLUGIN_TEST_LOCK.lock().await;
    let fixture = build_fixture_plugin();
    let manifest_ref = write_manifest(&fixture);

    {
        let (activation, _) = PluginHostActivation::activate(
            PluginConfig::default(),
            [host_spec("fixture_native", &manifest_ref)],
        )
        .await
        .expect("first plugin host should activate");
        assert!(activation.is_active());
    }

    assert!(
        !list_plugin_kinds()
            .iter()
            .any(|kind| kind == "fixture_native")
    );
    let (activation, _) = PluginHostActivation::activate(
        PluginConfig::default(),
        [host_spec("fixture_native", &manifest_ref)],
    )
    .await
    .expect("owner should be reusable after drop");
    activation.clear().expect("second plugin host should clear");
}

#[tokio::test]
async fn plugin_host_reserved_id_failure_does_not_poison_future_activation() {
    let _guard = NATIVE_PLUGIN_TEST_LOCK.lock().await;
    let fixture = build_fixture_plugin();
    let collision_manifest = write_manifest_with_plugin_id_and_symbol(
        &fixture,
        "observability",
        "nemo_relay_fixture_observability_collision",
    );

    let error = match PluginHostActivation::activate(
        PluginConfig::default(),
        [host_spec("observability", &collision_manifest)],
    )
    .await
    {
        Ok((activation, _)) => {
            activation
                .clear()
                .expect("unexpected collision activation should clear");
            panic!("a dynamic plugin must not replace a builtin kind");
        }
        Err(error) => error.to_string(),
    };
    assert!(error.contains("observability"), "{error}");
    assert!(error.contains("already registered"), "{error}");

    let manifest_ref = write_manifest(&fixture);
    let (activation, _) = PluginHostActivation::activate(
        PluginConfig::default(),
        [host_spec("fixture_native", &manifest_ref)],
    )
    .await
    .expect("a reserved-id failure must not poison later activation");
    activation
        .clear()
        .expect("recovered plugin host should clear");
}

#[tokio::test]
async fn plugin_host_rejects_duplicate_ids_before_loading() {
    let _guard = NATIVE_PLUGIN_TEST_LOCK.lock().await;
    let fixture = build_fixture_plugin();
    let manifest_ref = write_manifest(&fixture);
    let spec = host_spec("fixture_native", &manifest_ref);

    let error =
        match PluginHostActivation::activate(PluginConfig::default(), [spec.clone(), spec]).await {
            Ok((activation, _)) => {
                activation
                    .clear()
                    .expect("unexpected duplicate activation should clear");
                panic!("duplicate dynamic plugin ids should fail");
            }
            Err(error) => error.to_string(),
        };
    assert!(error.contains("duplicate dynamic plugin id"), "{error}");
    assert!(error.contains("fixture_native"), "{error}");
    assert!(
        !list_plugin_kinds()
            .iter()
            .any(|kind| kind == "fixture_native")
    );

    let (activation, _) = PluginHostActivation::activate(
        PluginConfig::default(),
        [host_spec("fixture_native", &manifest_ref)],
    )
    .await
    .expect("duplicate-id rejection must release the owner");
    activation.clear().expect("recovered host should clear");
}

#[tokio::test]
async fn plugin_host_clear_surfaces_missing_kind_and_releases_safe_owner() {
    let _guard = NATIVE_PLUGIN_TEST_LOCK.lock().await;
    let fixture = build_fixture_plugin();
    let manifest_ref = write_manifest(&fixture);
    let (activation, _) = PluginHostActivation::activate(
        PluginConfig::default(),
        [host_spec("fixture_native", &manifest_ref)],
    )
    .await
    .expect("plugin host should activate");

    assert!(deregister_plugin("fixture_native"));
    let error = activation
        .clear()
        .expect_err("missing plugin-kind deregistration should be surfaced")
        .to_string();
    assert!(error.contains("fixture_native"), "{error}");
    assert!(error.contains("was not registered"), "{error}");

    let (activation, _) = PluginHostActivation::activate(
        PluginConfig::default(),
        [host_spec("fixture_native", &manifest_ref)],
    )
    .await
    .expect("a safely absent plugin kind should release the owner");
    activation.clear().expect("recovered host should clear");
}

#[tokio::test]
async fn plugin_host_clear_preserves_a_replacement_registration() {
    let _guard = NATIVE_PLUGIN_TEST_LOCK.lock().await;
    let fixture = build_fixture_plugin();
    let manifest_ref = write_manifest(&fixture);
    let (activation, _) = PluginHostActivation::activate(
        PluginConfig::default(),
        [host_spec("fixture_native", &manifest_ref)],
    )
    .await
    .expect("plugin host should activate");

    assert!(deregister_plugin("fixture_native"));
    let replacement: Arc<dyn Plugin> = Arc::new(ReplacementRegistryPlugin);
    register_plugin(Arc::clone(&replacement)).expect("replacement plugin should register");
    let _cleanup = FixtureNativeRegistrationCleanup;

    let error = activation
        .clear()
        .expect_err("replacement during teardown should be surfaced")
        .to_string();
    assert!(error.contains("fixture_native"), "{error}");
    assert!(error.contains("was replaced"), "{error}");
    assert!(error.contains("left registered"), "{error}");

    let registered = lookup_plugin("fixture_native").expect("replacement must remain registered");
    assert!(Arc::ptr_eq(&registered, &replacement));
    assert!(deregister_plugin("fixture_native"));

    let (activation, _) = PluginHostActivation::activate(
        PluginConfig::default(),
        [host_spec("fixture_native", &manifest_ref)],
    )
    .await
    .expect("safe replacement detection should release the host owner");
    activation.clear().expect("recovered host should clear");
}

#[tokio::test]
async fn plugin_host_partial_load_failure_rolls_back_and_releases_owner() {
    let _guard = NATIVE_PLUGIN_TEST_LOCK.lock().await;
    let fixture = build_fixture_plugin();
    let manifest_ref = write_manifest(&fixture);
    let missing_manifest = manifest_ref.with_file_name("missing-relay-plugin.toml");

    let error = match PluginHostActivation::activate(
        PluginConfig::default(),
        [
            host_spec("fixture_native", &manifest_ref),
            host_spec("missing_native", &missing_manifest),
        ],
    )
    .await
    {
        Ok((activation, _)) => {
            activation
                .clear()
                .expect("unexpected activation should clear");
            panic!("partial load should fail");
        }
        Err(error) => error.to_string(),
    };
    assert!(error.contains("missing-relay-plugin.toml"), "{error}");
    assert!(
        !list_plugin_kinds()
            .iter()
            .any(|kind| kind == "fixture_native")
    );

    let (activation, _) = PluginHostActivation::activate(
        PluginConfig::default(),
        [host_spec("fixture_native", &manifest_ref)],
    )
    .await
    .expect("failed activation must release the owner");
    activation
        .clear()
        .expect("recovered activation should clear");
}

#[tokio::test]
async fn plugin_host_rejects_an_existing_legacy_configuration() {
    let _guard = NATIVE_PLUGIN_TEST_LOCK.lock().await;
    let fixture = build_fixture_plugin();
    let manifest_ref = write_manifest(&fixture);
    initialize_plugins_exact(PluginConfig::default())
        .await
        .expect("legacy configuration should initialize");

    let error = match PluginHostActivation::activate(
        PluginConfig::default(),
        [host_spec("fixture_native", &manifest_ref)],
    )
    .await
    {
        Ok((activation, _)) => {
            activation
                .clear()
                .expect("unexpected activation should clear");
            panic!("host activation should reject an existing configuration");
        }
        Err(error) => error.to_string(),
    };
    assert!(
        error.contains("static plugin configuration is already active"),
        "{error}"
    );
    assert!(error.contains("base configuration"), "{error}");
    clear_plugin_configuration().expect("legacy configuration should clear");
}

#[cfg(not(feature = "worker-grpc"))]
#[tokio::test]
async fn plugin_host_rejects_workers_when_worker_support_is_disabled() {
    let _guard = NATIVE_PLUGIN_TEST_LOCK.lock().await;
    let error = match PluginHostActivation::activate(
        PluginConfig::default(),
        [DynamicPluginActivationSpec {
            plugin_id: "fixture_worker".into(),
            kind: DynamicPluginKind::Worker,
            manifest_ref: "unused-worker-manifest.toml".into(),
            environment_ref: None,
            config: Map::new(),
        }],
    )
    .await
    {
        Ok((activation, _)) => {
            activation
                .clear()
                .expect("unexpected activation should clear");
            panic!("worker activation should require worker support");
        }
        Err(error) => error.to_string(),
    };
    assert!(error.contains("worker-grpc"), "{error}");

    let fixture = build_fixture_plugin();
    let manifest_ref = write_manifest(&fixture);
    let (activation, _) = PluginHostActivation::activate(
        PluginConfig::default(),
        [host_spec("fixture_native", &manifest_ref)],
    )
    .await
    .expect("failed worker activation must release the owner");
    activation.clear().expect("recovered host should clear");
}

async fn initialize_fixture_native(config: Map<String, Json>) -> nemo_relay::plugin::Result<()> {
    let mut plugin_config = PluginConfig::default();
    plugin_config.components.push(PluginComponentSpec {
        kind: "fixture_native".into(),
        enabled: true,
        config,
    });
    initialize_plugins_exact(plugin_config).await.map(|_| ())
}

fn expect_native_load_error(spec: NativePluginLoadSpec, message: &str) -> String {
    expect_native_load_error_from_specs([spec], message)
}

fn expect_native_load_error_from_specs<I>(specs: I, message: &str) -> String
where
    I: IntoIterator<Item = NativePluginLoadSpec>,
{
    match load_native_plugins(specs) {
        Ok(activation) => {
            activation.clear();
            panic!("{message}");
        }
        Err(error) => error.to_string(),
    }
}

fn load_spec(plugin_id: &str, manifest_ref: &Path) -> NativePluginLoadSpec {
    NativePluginLoadSpec {
        plugin_id: plugin_id.into(),
        manifest_ref: manifest_ref.to_string_lossy().into_owned(),
    }
}

fn host_spec(plugin_id: &str, manifest_ref: &Path) -> DynamicPluginActivationSpec {
    DynamicPluginActivationSpec {
        plugin_id: plugin_id.into(),
        kind: DynamicPluginKind::RustDynamic,
        manifest_ref: manifest_ref.to_string_lossy().into_owned(),
        environment_ref: None,
        config: Map::new(),
    }
}

fn assert_parent(
    events: &[Event],
    name: &str,
    scope_category: Option<ScopeCategory>,
    expected_parent: Option<Uuid>,
) {
    let event = find_event(events, name, scope_category);
    assert_eq!(
        event.parent_uuid(),
        expected_parent,
        "{name} parent mismatch"
    );
}

fn assert_not_parent(
    events: &[Event],
    name: &str,
    scope_category: Option<ScopeCategory>,
    unexpected_parent: Uuid,
) {
    let event = find_event(events, name, scope_category);
    assert_ne!(
        event.parent_uuid(),
        Some(unexpected_parent),
        "{name} should be emitted on an isolated stack"
    );
}

fn find_event<'a>(
    events: &'a [Event],
    name: &str,
    scope_category: Option<ScopeCategory>,
) -> &'a Event {
    events
        .iter()
        .find(|event| event.name() == name && event.scope_category() == scope_category)
        .unwrap_or_else(|| panic!("missing event {name} with scope category {scope_category:?}"))
}

struct BuiltFixture {
    _source_dir: TempDir,
    _target_dir: TempDir,
    manifest_dir: TempDir,
    library_path: PathBuf,
}

fn build_fixture_plugin() -> BuiltFixture {
    let source_dir = TempDir::new().expect("fixture source dir");
    let fixture_dir = source_dir.path().join("native_plugin");
    let fixture_src_dir = fixture_dir.join("src");
    std::fs::create_dir_all(&fixture_src_dir).expect("fixture src dir");
    let native_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../plugin");
    let fixture_manifest = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/native_plugin/Cargo.toml"),
    )
    .expect("fixture Cargo.toml template")
    .replace(
        r#"nemo-relay-plugin = { path = "../../../../plugin" }"#,
        &format!("nemo-relay-plugin = {{ path = {native_path:?} }}"),
    );
    std::fs::write(fixture_dir.join("Cargo.toml"), fixture_manifest).expect("fixture Cargo.toml");
    std::fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/native_plugin/src/lib.rs"),
        fixture_src_dir.join("lib.rs"),
    )
    .expect("fixture lib.rs");
    let target_dir = TempDir::new().expect("fixture target dir");
    let manifest_dir = TempDir::new().expect("fixture manifest dir");
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let status = Command::new(cargo)
        .arg("build")
        .arg("--quiet")
        .arg("--manifest-path")
        .arg(fixture_dir.join("Cargo.toml"))
        .arg("--target-dir")
        .arg(target_dir.path())
        .status()
        .expect("fixture cargo build should start");
    assert!(status.success(), "fixture cargo build failed: {status}");

    let library_path = target_dir.path().join("debug").join(fixture_library_name());
    assert!(
        library_path.exists(),
        "fixture library missing at {}",
        library_path.display()
    );

    BuiltFixture {
        _source_dir: source_dir,
        _target_dir: target_dir,
        manifest_dir,
        library_path,
    }
}

fn write_manifest(fixture: &BuiltFixture) -> PathBuf {
    write_manifest_text(ManifestOptions {
        manifest_dir: fixture.manifest_dir.path(),
        plugin_id: "fixture_native",
        relay: &format!("={}", env!("CARGO_PKG_VERSION")),
        library: &fixture.library_path.to_string_lossy(),
        symbol: "nemo_relay_fixture_native_plugin",
        integrity: None,
    })
}

fn write_manifest_with_symbol(fixture: &BuiltFixture, symbol: &str) -> PathBuf {
    write_manifest_text(ManifestOptions {
        manifest_dir: fixture.manifest_dir.path(),
        plugin_id: "fixture_native",
        relay: &format!("={}", env!("CARGO_PKG_VERSION")),
        library: &fixture.library_path.to_string_lossy(),
        symbol,
        integrity: None,
    })
}

fn write_manifest_with_plugin_id_and_symbol(
    fixture: &BuiltFixture,
    plugin_id: &str,
    symbol: &str,
) -> PathBuf {
    write_manifest_text(ManifestOptions {
        manifest_dir: fixture.manifest_dir.path(),
        plugin_id,
        relay: &format!("={}", env!("CARGO_PKG_VERSION")),
        library: &fixture.library_path.to_string_lossy(),
        symbol,
        integrity: None,
    })
}

fn write_manifest_with_plugin_id(fixture: &BuiltFixture, plugin_id: &str) -> PathBuf {
    write_manifest_text(ManifestOptions {
        manifest_dir: fixture.manifest_dir.path(),
        plugin_id,
        relay: &format!("={}", env!("CARGO_PKG_VERSION")),
        library: &fixture.library_path.to_string_lossy(),
        symbol: "nemo_relay_fixture_native_plugin",
        integrity: None,
    })
}

fn write_manifest_with_integrity(fixture: &BuiltFixture, integrity: &str) -> PathBuf {
    write_manifest_text(ManifestOptions {
        manifest_dir: fixture.manifest_dir.path(),
        plugin_id: "fixture_native",
        relay: &format!("={}", env!("CARGO_PKG_VERSION")),
        library: &fixture.library_path.to_string_lossy(),
        symbol: "nemo_relay_fixture_native_plugin",
        integrity: Some(integrity),
    })
}

struct ManifestOptions<'a> {
    manifest_dir: &'a Path,
    plugin_id: &'a str,
    relay: &'a str,
    library: &'a str,
    symbol: &'a str,
    integrity: Option<&'a str>,
}

fn write_manifest_text(options: ManifestOptions<'_>) -> PathBuf {
    let manifest_ref = options.manifest_dir.join("relay-plugin.toml");
    let integrity = options
        .integrity
        .map(|sha256| format!("\n[integrity]\nsha256 = {}\n", toml_string(sha256)))
        .unwrap_or_default();
    let plugin_id = toml_string(options.plugin_id);
    let relay = toml_string(options.relay);
    let library = toml_string(options.library);
    let symbol = toml_string(options.symbol);
    let manifest = format!(
        r#"
manifest_version = 1

[plugin]
id = {plugin_id}
kind = "rust_dynamic"

[compat]
relay = {relay}
native_api = "1"

[defaults]
enabled = false

[capabilities]
items = ["plugin_native"]

[load]
library = {library}
symbol = {symbol}
{integrity}
"#,
        plugin_id = plugin_id,
        relay = relay,
        library = library,
        symbol = symbol,
        integrity = integrity,
    );
    std::fs::write(&manifest_ref, manifest).expect("write relay-plugin.toml");
    manifest_ref
}

fn write_raw_manifest(manifest_dir: &Path, manifest: &str) -> PathBuf {
    let manifest_ref = manifest_dir.join("relay-plugin.toml");
    std::fs::write(&manifest_ref, manifest).expect("write relay-plugin.toml");
    manifest_ref
}

fn native_manifest_text(
    plugin_id: &str,
    relay: &str,
    native_api: &str,
    library: &str,
    symbol: &str,
) -> String {
    format!(
        r#"
manifest_version = 1

[plugin]
id = {plugin_id}
kind = "rust_dynamic"

[compat]
relay = {relay}
native_api = {native_api}

[defaults]
enabled = false

[capabilities]
items = ["plugin_native"]

[load]
library = {library}
symbol = {symbol}
"#,
        plugin_id = toml_string(plugin_id),
        relay = toml_string(relay),
        native_api = toml_string(native_api),
        library = toml_string(library),
        symbol = toml_string(symbol),
    )
}

fn toml_string(value: &str) -> String {
    serde_json::to_string(value).expect("TOML-compatible string escape should succeed")
}

fn sha256(path: &Path) -> String {
    let bytes = std::fs::read(path).expect("read file for digest");
    let digest = Sha256::digest(bytes);
    format!("sha256:{}", hex_digest(digest))
}

fn hex_digest(bytes: impl AsRef<[u8]>) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let bytes = bytes.as_ref();
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

fn fixture_library_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "nemo_relay_plugin_fixture.dll"
    } else if cfg!(target_os = "macos") {
        "libnemo_relay_plugin_fixture.dylib"
    } else {
        "libnemo_relay_plugin_fixture.so"
    }
}
