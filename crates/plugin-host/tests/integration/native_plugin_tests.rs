// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration coverage for SDK-built native dynamic plugins.

use crate::test_support::plugin_host as plugin_host_test_support;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::Poll;

use nemo_relay_plugin_host::dynamic::{
    DynamicPluginKind, NativePluginLoadSpec, VerifiedDynamicPluginSpec, initialize,
    load_native_plugins,
};
use nemo_relay_plugin_host::{
    ConfigDiagnostic, Plugin, PluginComponentSpec, PluginConfig, Result as PluginResult,
    deregister_plugin, list_plugin_kinds, lookup_plugin, register_plugin,
};
use nemo_relay_runtime::api::event::{Event, ScopeCategory};
use nemo_relay_runtime::api::llm::{
    LlmCallEndParams, LlmCallExecuteParams, LlmCallParams, LlmRequest, LlmStreamCallExecuteParams,
    llm_call, llm_call_end, llm_call_execute, llm_stream_call_execute,
};
use nemo_relay_runtime::api::runtime::{
    LlmJsonStream, TASK_SCOPE_STACK, ThreadScopeStackBinding, capture_thread_scope_stack,
    create_scope_stack, restore_thread_scope_stack, set_thread_scope_stack,
};
use nemo_relay_runtime::api::scope::{
    EmitMarkEventParams, PopScopeParams, PushScopeParams, ScopeType, event as emit_scope_mark,
    pop_scope, push_scope,
};
use nemo_relay_runtime::api::subscriber::{
    deregister_subscriber, flush_subscribers, register_subscriber,
};
use nemo_relay_runtime::api::tool::{
    ToolCallExecuteParams, ToolExecutionResult, tool_call_execute, tool_request_intercepts,
};
use nemo_relay_types::codec::response::AnnotatedLlmResponse;
use plugin_host_test_support::{test_close_plugin_host, test_initialize_plugin_host_exact};
use serde_json::{Map, Value as Json, json};
use sha2::{Digest, Sha256};
use tempfile::TempDir;
use tokio::sync::Notify;
use tokio_stream::StreamExt;
use uuid::Uuid;

static NATIVE_PLUGIN_TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
const PLUGIN_DISCOVERY_TEST_CHILD: &str = "NEMO_RELAY_PLUGIN_DISCOVERY_TEST_CHILD";

#[tokio::test]
async fn native_registration_preserves_task_scope_after_await() {
    let _guard = NATIVE_PLUGIN_TEST_LOCK.lock().await;
    let fixture = build_fixture_plugin();
    let manifest_ref = write_manifest(&fixture);
    let _activation = load_native_plugins([NativePluginLoadSpec {
        plugin_id: "fixture_native".into(),
        manifest_ref: manifest_ref.to_string_lossy().into_owned(),
    }])
    .expect("native fixture loads");
    let mut cleanup = NativePluginTestCleanup::new();
    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    register_subscriber(
        "native-registration-parent",
        Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();
    cleanup.mark_subscriber_registered("native-registration-parent");
    let parent = TASK_SCOPE_STACK
        .scope(create_scope_stack(), async {
            let parent = push_scope(
                PushScopeParams::builder()
                    .name("registration-parent")
                    .scope_type(ScopeType::Agent)
                    .build(),
            )
            .unwrap();
            let plugin = lookup_plugin("fixture_native").unwrap();
            let mut context = nemo_relay_plugin_host::HostPluginRegistrationContext::new();
            let config = Map::from_iter([("registration_mark".into(), json!(true))]);
            let result = plugin.register(&config, &mut context).await;
            let mut registrations = context.into_registrations();
            let rollback = nemo_relay_plugin_host::rollback_registrations(&mut registrations);
            assert!(rollback.callbacks_cleared(), "{:?}", rollback.errors());
            result.expect("async registration succeeds");
            pop_scope(PopScopeParams::builder().handle_uuid(&parent.uuid).build()).unwrap();
            parent.uuid
        })
        .await;
    flush_subscribers().unwrap();
    let events = events.lock().unwrap();
    let mark = events
        .iter()
        .find(|event| event.name() == "fixture-registration-mark")
        .expect("registration emitted its mark");
    assert_eq!(mark.parent_uuid(), Some(parent));
}

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
        ctx: &'a mut dyn nemo_relay_plugin::component::PluginRegistrationContext,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = PluginResult<()>> + Send + 'a>> {
        Box::pin(async move {
            STATIC_BASE_REGISTRATIONS.fetch_add(1, Ordering::SeqCst);
            ctx.add_registration(nemo_relay_plugin::component::PluginRegistration::new(
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
        ctx: &'a mut dyn nemo_relay_plugin::component::PluginRegistrationContext,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = PluginResult<()>> + Send + 'a>> {
        let started = Arc::clone(&self.started);
        let release = Arc::clone(&self.release);
        let registered = Arc::clone(&self.registered);
        Box::pin(async move {
            started.notify_one();
            release.notified().await;
            ctx.add_registration(nemo_relay_plugin::component::PluginRegistration::new(
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
        _ctx: &'a mut dyn nemo_relay_plugin::component::PluginRegistrationContext,
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

struct SubscriberRegistrations(Vec<&'static str>);

impl Drop for SubscriberRegistrations {
    fn drop(&mut self) {
        for name in self.0.drain(..) {
            let _ = deregister_subscriber(name);
        }
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
            let _ = test_close_plugin_host();
        }
    }
}

#[tokio::test]
async fn checked_sdk_preserves_all_native_callback_behavior() {
    sdk_cdylib_assert_behavior().await;
}

// Keep both transports on the same end-to-end assertions during the cutover.
#[allow(clippy::cognitive_complexity)]
async fn sdk_cdylib_assert_behavior() {
    let _guard = NATIVE_PLUGIN_TEST_LOCK.lock().await;
    let fixture = build_fixture_plugin();
    let manifest_ref = write_manifest(&fixture);

    let activation = Some(
        load_native_plugins([load_spec("fixture_native", &manifest_ref)])
            .expect("checked production activation"),
    );
    let loaded_path = activation
        .as_ref()
        .unwrap()
        .resident_libraries()
        .next()
        .unwrap()
        .to_owned();
    let fixture_library = unsafe { libloading::Library::new(&loaded_path) }
        .expect("native fixture should open for synchronization");
    let pending_entered = unsafe {
        *fixture_library
            .get::<unsafe extern "C" fn() -> bool>(b"nemo_relay_fixture_async_pending_entered\0")
            .expect("native fixture should export its pending-entry signal")
    };
    // Clear a signal left by any earlier fixture use in this process.
    let _ = unsafe { pending_entered() };
    let mut cleanup = NativePluginTestCleanup::new();

    let mut plugin_config = PluginConfig::default();
    plugin_config.components.push(PluginComponentSpec {
        kind: "fixture_native".into(),
        enabled: true,
        config: Map::from_iter([("executor".into(), json!({"worker_threads":1}))]),
    });
    test_initialize_plugin_host_exact(plugin_config)
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
    let tool_callback_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let captured_tool_callback_calls = Arc::clone(&tool_callback_calls);
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
                .await
                .expect("native request intercept should run");
            let expected = serde_json::to_value(
                nemo_relay_runtime::api::runtime::current_scope_stack_snapshot().unwrap(),
            )
            .unwrap();
            assert_eq!(rewritten["scope_stack_snapshot"], expected);
            assert_eq!(rewritten["scope_stack_snapshot_after"], expected);
            let retained = rewritten["scope_stack_snapshot"]["scopes"]
                .as_array()
                .unwrap();
            let inside = rewritten["scope_stack_snapshot_inside"]["scopes"]
                .as_array()
                .unwrap();
            assert_eq!(inside.len(), retained.len() + 1);
            assert_eq!(&inside[..retained.len()], retained.as_slice());
            assert_eq!(inside.last().unwrap()["scope_type"], "function");
            assert_eq!(retained.last().unwrap()["scope_id"], outer_uuid.to_string());
            let tool_result = tool_call_execute(
                ToolCallExecuteParams::builder()
                    .name("native-fixture-tool")
                    .args(json!({ "input": "execute", "use_concurrent_next": true }))
                    .func(Arc::new(move |args| {
                        let calls = Arc::clone(&captured_tool_callback_calls);
                        Box::pin(async move {
                            calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                            Ok(ToolExecutionResult::annotated(
                                json!({ "tool_callback": true, "args": args }),
                                json!({"source": "provider"}),
                            ))
                        })
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
    assert_eq!(tool_result.result["tool_callback"], true);
    assert_eq!(tool_result.result["native_plugin_tool_execution"], true);
    assert_eq!(tool_result.annotation, Some(json!({"source": "provider"})));
    assert_eq!(
        tool_callback_calls.load(std::sync::atomic::Ordering::SeqCst),
        2
    );
    assert_eq!(
        tool_result.result["args"]["native_plugin_tool_execution_request"],
        true
    );
    assert!(tool_result.result.get("pending_marks").is_none());

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
    let metric = find_event(&first_events, "fixture.native.metric", None);
    assert_eq!(
        metric.data_schema().expect("native metric schema").name,
        "nemo.relay.metric_measurements"
    );
    assert_eq!(
        metric.data_schema().expect("native metric schema").version,
        "1"
    );
    let measurement = &metric.data().unwrap()["measurements"][0];
    assert_eq!(measurement["name"], "fixture.native.count");
    assert_eq!(measurement["kind"], "counter");
    assert_eq!(measurement["value_type"], "u64");
    assert_eq!(measurement["value"], 1);
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
                            Ok(ToolExecutionResult::new(json!({ "tool_callback": true })))
                        })
                    }))
                    .build(),
            )
            .await
            .expect("native isolated next middleware should run");
            assert_eq!(result.result["tool_callback"], true);
            assert_eq!(result.result["native_plugin_tool_execution"], true);
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
    let result = tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("native-fixture-tool-scoped-next")
            .args(json!({
                "input": "scoped-next",
                "use_scoped_next": true
            }))
            .func(Arc::new(|_args| {
                Box::pin(async move {
                    emit_scope_mark(
                        EmitMarkEventParams::builder()
                            .name("native-fixture-tool-scoped-next-callback-mark")
                            .build(),
                    )?;
                    Ok(ToolExecutionResult::new(json!({ "tool_callback": true })))
                })
            }))
            .build(),
    )
    .await
    .expect("native scoped next middleware should run");
    assert_eq!(result.result["tool_callback"], true);
    flush_subscribers().expect("scoped next native fixture events should flush");
    let scoped_next_events = events.lock().unwrap().clone();
    let scoped_next_scope = find_event(
        &scoped_next_events,
        "fixture.native.scoped.next",
        Some(ScopeCategory::Start),
    );
    let downstream_scoped_next_scope = find_event(
        &scoped_next_events,
        "fixture.native.scoped.next.downstream",
        Some(ScopeCategory::Start),
    );
    let scoped_next_callback_mark = find_event(
        &scoped_next_events,
        "native-fixture-tool-scoped-next-callback-mark",
        None,
    );
    assert_eq!(
        downstream_scoped_next_scope.parent_uuid(),
        Some(scoped_next_scope.uuid()),
        "the downstream native callback should inherit the scope opened around next.call"
    );
    assert_eq!(
        scoped_next_callback_mark.parent_uuid(),
        Some(downstream_scoped_next_scope.uuid()),
        "work below the downstream native callback should remain nested"
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
            .await
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
    let managed_stack = create_scope_stack();
    let managed_agent = TASK_SCOPE_STACK
        .scope(managed_stack.clone(), async {
            push_scope(
                PushScopeParams::builder()
                    .name("native-plugin-managed-agent")
                    .scope_type(ScopeType::Agent)
                    .build(),
            )
            .unwrap()
        })
        .await;
    let managed_snapshot = serde_json::to_value(
        nemo_relay_runtime::api::runtime::scope_stack_snapshot(&managed_stack).unwrap(),
    )
    .unwrap();
    assert_eq!(
        managed_snapshot["scopes"][1]["scope_id"],
        managed_agent.uuid.to_string()
    );
    let llm_execute_response = TASK_SCOPE_STACK
        .scope(
            managed_stack.clone(),
            llm_call_execute(
                LlmCallExecuteParams::builder()
                    .name("native-fixture-llm-execute")
                    .request(LlmRequest {
                        headers: Map::new(),
                        content: json!({ "prompt": "managed", "inspect_scope_stack": true }),
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
            ),
        )
        .await
        .expect("native LLM middleware should run");
    assert_eq!(
        llm_execute_response["scope_stack_snapshot"],
        managed_snapshot
    );
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

    for status in [403, 404, 429, 503] {
        let error = llm_call_execute(
            LlmCallExecuteParams::builder()
                .name("native-fixture-http-error")
                .request(LlmRequest {
                    headers: Map::new(),
                    content: json!({"prompt": "fail"}),
                })
                .func(Arc::new(move |_request| {
                    Box::pin(async move {
                        Err(nemo_relay_types::error::FlowError::Upstream(
                            nemo_relay_types::error::UpstreamFailure {
                                status: Some(status),
                                body: "provider denied access".into(),
                                headers: Default::default(),
                                class: nemo_relay_types::error::UpstreamFailureClass::Other,
                            },
                        ))
                    })
                }))
                .build(),
        )
        .await
        .unwrap_err();
        assert!(
            matches!(error, nemo_relay_types::error::FlowError::Upstream(failure) if failure.status == Some(status) && failure.body == "Reported by plugin 'fixture_native': provider denied access" && !failure.is_retryable())
        );
    }
    let callback_error = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("native-fixture-originated-http-error")
            .request(LlmRequest {
                headers: Map::new(),
                content: json!({"native_http_error": true}),
            })
            .func(Arc::new(|_| {
                Box::pin(async { panic!("plugin failure must skip provider") })
            }))
            .build(),
    )
    .await
    .unwrap_err();
    assert!(
        matches!(callback_error, nemo_relay_types::error::FlowError::Upstream(failure) if failure.status == Some(403) && failure.body == "Reported by plugin 'fixture_native': provider denied access")
    );
    let stream_error = llm_stream_call_execute(
        LlmStreamCallExecuteParams::builder()
            .name("native-fixture-originated-stream-http-error")
            .request(LlmRequest {
                headers: Map::new(),
                content: json!({"native_http_error": true}),
            })
            .func(Arc::new(|_| {
                Box::pin(async { panic!("plugin failure must skip provider") })
            }))
            .collector(Box::new(|_| Ok(())))
            .finalizer(Box::new(|| Json::Null))
            .build(),
    )
    .await
    .expect("native async stream opens before its callback is polled")
    .next()
    .await
    .expect("stream reports the callback failure")
    .unwrap_err();
    assert!(
        matches!(stream_error, nemo_relay_types::error::FlowError::Upstream(failure) if failure.status == Some(403) && failure.body == "Reported by plugin 'fixture_native': provider denied access")
    );
    for successful_chunks in [0, 1] {
        let mut stream = llm_stream_call_execute(
            LlmStreamCallExecuteParams::builder()
                .name("native-fixture-stream-pull-http-error")
                .request(LlmRequest {
                    headers: Map::new(),
                    content: json!({"native_stream_http_error_after_chunks":successful_chunks}),
                })
                .func(Arc::new(|_| {
                    Box::pin(async { panic!("plugin stream must skip provider") })
                }))
                .collector(Box::new(|_| Ok(())))
                .finalizer(Box::new(|| Json::Null))
                .build(),
        )
        .await
        .expect("native stream opens before its producer is polled");
        for index in 0..successful_chunks {
            let chunk = stream.next().await.unwrap().unwrap();
            assert_eq!(chunk["before_error"], index);
        }
        let error = stream.next().await.unwrap().unwrap_err();
        assert!(
            matches!(error, nemo_relay_types::error::FlowError::Upstream(failure) if failure.status == Some(403) && failure.body == "Reported by plugin 'fixture_native': provider denied access"),
        );
        assert!(
            stream.next().await.is_none(),
            "HTTP stream errors are terminal"
        );
        stream
            .close()
            .await
            .expect("failed stream releases its producer");
    }
    flush_subscribers().expect("HTTP error events should flush");

    events.lock().unwrap().clear();
    let cancelled_unary = tokio::spawn(llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("native-fixture-cancelled-unary")
            .request(LlmRequest {
                headers: Map::new(),
                content: json!({ "prompt": "cancel" }),
            })
            .func(Arc::new(|_| Box::pin(std::future::pending())))
            .build(),
    ));
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while !unsafe { pending_entered() } {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("native unary future should start before cancellation");
    cancelled_unary.abort();
    assert!(
        cancelled_unary
            .await
            .expect_err("pending native unary call should be cancelled")
            .is_cancelled()
    );
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            flush_subscribers().expect("cancelled unary events should flush");
            if events.lock().unwrap().iter().any(|event| {
                event.name() == "fixture.native.unary.drop"
                    && event.scope_category() == Some(ScopeCategory::End)
            }) {
                break;
            }
        }
    })
    .await
    .expect("plugin unary future Drop should emit its scope after cancellation");
    let cancelled_unary_events = events.lock().unwrap().clone();
    let cancelled_unary_start = find_event(
        &cancelled_unary_events,
        "native-fixture-cancelled-unary",
        Some(ScopeCategory::Start),
    );
    assert_parent(
        &cancelled_unary_events,
        "fixture.native.unary.drop",
        Some(ScopeCategory::End),
        Some(cancelled_unary_start.uuid()),
    );
    drop(fixture_library);

    events.lock().unwrap().clear();
    let collected_stream_chunks = Arc::new(Mutex::new(Vec::<Json>::new()));
    let collector_chunks = collected_stream_chunks.clone();
    let finalizer_chunks = collected_stream_chunks.clone();
    let mut stream = TASK_SCOPE_STACK
        .scope(
            managed_stack.clone(),
            llm_stream_call_execute(
                LlmStreamCallExecuteParams::builder()
                    .name("native-fixture-llm-stream")
                    .request(LlmRequest {
                        headers: Map::new(),
                        content: json!({ "prompt": "stream", "inspect_scope_stack": true }),
                    })
                    .func(Arc::new(|request| {
                        Box::pin(async move {
                            Ok(LlmJsonStream::new(tokio_stream::iter(vec![
                                Ok(json!({
                                    "stream_chunk": 1,
                                    "request": request.content,
                                })),
                                Ok(json!({ "stream_chunk": 2 })),
                            ])))
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
            ),
        )
        .await
        .expect("native LLM stream middleware should run");
    let mut stream_chunks = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.expect("native stream chunk should succeed");
        assert_eq!(chunk["scope_stack_snapshot"], managed_snapshot);
        assert_eq!(chunk["scope_stack_snapshot_polled"], managed_snapshot);
        stream_chunks.push(chunk);
    }
    TASK_SCOPE_STACK
        .scope(managed_stack, async {
            pop_scope(
                PopScopeParams::builder()
                    .handle_uuid(&managed_agent.uuid)
                    .build(),
            )
            .unwrap();
        })
        .await;
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
    let provider_polled = Arc::new(AtomicBool::new(false));
    let polled = Arc::clone(&provider_polled);
    let cancelled_stream = llm_stream_call_execute(
        LlmStreamCallExecuteParams::builder()
            .name("native-fixture-cancelled-stream")
            .request(LlmRequest {
                headers: Map::new(),
                content: json!({ "prompt": "cancel" }),
            })
            .func(Arc::new(move |_request| {
                let polled = Arc::clone(&polled);
                Box::pin(async move {
                    Ok(LlmJsonStream::new(futures::stream::poll_fn(move |_| {
                        polled.store(true, Ordering::SeqCst);
                        Poll::Pending
                    })))
                })
            }))
            .collector(Box::new(|_| Ok(())))
            .finalizer(Box::new(|| Json::Null))
            .build(),
    )
    .await
    .expect("pending native stream should open");
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while !provider_polled.load(Ordering::SeqCst) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("provider stream should be polled before cancellation");
    drop(cancelled_stream);
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            flush_subscribers().expect("cancelled stream events should flush");
            if events.lock().unwrap().iter().any(|event| {
                event.name() == "fixture.native.stream.drop"
                    && event.scope_category() == Some(ScopeCategory::End)
            }) {
                break;
            }
        }
    })
    .await
    .expect("plugin stream Drop should emit its scope after cancellation");
    let cancelled_stream_events = events.lock().unwrap().clone();
    let cancelled_stream_start = find_event(
        &cancelled_stream_events,
        "native-fixture-cancelled-stream",
        Some(ScopeCategory::Start),
    );
    assert_parent(
        &cancelled_stream_events,
        "fixture.native.stream.drop",
        Some(ScopeCategory::End),
        Some(cancelled_stream_start.uuid()),
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
    assert_eq!(
        llm_end.output().unwrap()["native_plugin_llm_sanitize_response"],
        true
    );
    assert!(
        llm_end.annotated_response().is_none(),
        "a changed response without an active codec must discard the stale caller annotation"
    );
    let serialized = serde_json::to_string(llm_end).expect("LLM end event should serialize");
    assert!(!serialized.contains("annotation-before-plugin"));
    assert!(!serialized.contains("preexisting_annotation"));

    drop(cleanup);
    if let Some(activation) = activation {
        activation.clear();
    } else {
        assert!(deregister_plugin("fixture_native"));
    }
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
    let error = test_initialize_plugin_host_exact(plugin_config)
        .await
        .expect_err("validation diagnostics should prevent initialization")
        .to_string();
    assert!(error.contains("fixture rejection requested"), "{error}");

    test_close_plugin_host().expect("native plugin config should clear");
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
        relay: ">=1.0.0,<2.0",
        library: &missing_library.to_string_lossy(),

        integrity: None,
    });

    let error = expect_native_load_error(
        NativePluginLoadSpec {
            plugin_id: "fixture_native".into(),
            manifest_ref: manifest_ref.to_string_lossy().into_owned(),
        },
        "missing library should fail",
    );
    assert!(error.contains("native plugin library"), "{error}");
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
        relay: ">=1.0.0,<2.0",
        library: &relative_library.to_string_lossy(),

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
        relay: ">=1.0.0,<2.0",
        library: &missing_library.to_string_lossy(),

        integrity: None,
    });

    let error = expect_native_load_error_from_specs(
        [
            load_spec("fixture_native", &valid_manifest),
            load_spec("fixture_native_missing", &missing_manifest),
        ],
        "partial load failure should fail",
    );
    assert!(error.contains("native plugin library"), "{error}");

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
        relay: ">=2.0,<3.0",
        library: "libdoes-not-need-to-exist.so",

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
fn native_manifest_writer_escapes_toml_strings() {
    let _guard = NATIVE_PLUGIN_TEST_LOCK.blocking_lock();
    let manifest_dir = TempDir::new().expect("manifest dir");
    let windows_style_library =
        r"C:\Users\RUNNER~1\AppData\Local\Temp\.tmpPath\debug\nemo_relay_plugin_fixture.dll";
    let manifest_ref = write_manifest_text(ManifestOptions {
        manifest_dir: manifest_dir.path(),
        plugin_id: "fixture_native",
        relay: ">=1.0.0,<2.0",
        library: windows_style_library,

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

#[tokio::test]
async fn plugin_host_activation_owns_configuration_until_close() {
    let _guard = NATIVE_PLUGIN_TEST_LOCK.lock().await;
    let fixture = build_fixture_plugin();
    let manifest_ref = write_manifest(&fixture);
    let (mut activation, report) = crate::PluginHostActivation::initialize_with_verified_specs(
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
        .await
        .expect("host-owned intercept should run");
    assert_eq!(rewritten["native_plugin"], true);

    let initialize_error =
        match crate::PluginHostActivation::initialize_exact(PluginConfig::default()).await {
            Ok(mut unexpected) => {
                unexpected.close().expect("unexpected host should close");
                panic!("a second host replaced the active host");
            }
            Err(error) => error.to_string(),
        };
    assert!(initialize_error.contains("active dynamic plugin host"));

    activation.close().expect("plugin host should close");
    assert!(
        !list_plugin_kinds()
            .iter()
            .any(|kind| kind == "fixture_native")
    );
    let unchanged = tool_request_intercepts("host-owned-tool", json!({ "input": true }))
        .await
        .expect("cleared intercept chain should be empty");
    assert_eq!(unchanged, json!({ "input": true }));
}

#[tokio::test]
async fn native_registration_discovers_static_observability() {
    let _guard = NATIVE_PLUGIN_TEST_LOCK.lock().await;
    let fixture = build_fixture_plugin();
    let manifest_ref = write_manifest(&fixture);
    let collector = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}/v1/traces", collector.local_addr().unwrap());
    let config: PluginConfig = serde_json::from_value(json!({
        "components": [{"kind": "observability", "enabled": true, "config": {
            "opentelemetry": {"enabled": true, "endpoints": [{
                "type": "gen_ai", "endpoint": endpoint
            }]}
        }}]
    }))
    .unwrap();
    let mut spec = host_spec("fixture_native", &manifest_ref);
    spec.config = Map::from_iter([("discover_observability".into(), json!(true))]);

    let (mut activation, report) =
        crate::PluginHostActivation::initialize_with_verified_specs(config, [spec])
            .await
            .expect("native registration must discover static OpenTelemetry subscribers");
    assert!(!report.has_errors());
    activation
        .close()
        .expect("native host should close cleanly");
    assert!(
        nemo_relay_runtime::api::registry::list_runtime_registrations(None)
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn native_conditional_callbacks_block_allow_fail_open_and_clear() {
    let _guard = NATIVE_PLUGIN_TEST_LOCK.lock().await;
    let fixture = build_fixture_plugin();
    let manifest_ref = write_manifest(&fixture);
    let registrations = [
        (
            "fixture-native-blocked-subscriber",
            Arc::new(AtomicUsize::new(0)),
        ),
        (
            "fixture-native-allowed-subscriber",
            Arc::new(AtomicUsize::new(0)),
        ),
        (
            "fixture-native-failed-open-subscriber",
            Arc::new(AtomicUsize::new(0)),
        ),
    ];
    let _subscriber_cleanup =
        SubscriberRegistrations(registrations.iter().map(|(name, _)| *name).collect());
    for (name, calls) in &registrations {
        let calls = Arc::clone(calls);
        register_subscriber(
            name,
            Arc::new(move |_| {
                calls.fetch_add(1, Ordering::SeqCst);
            }),
        )
        .expect("test subscriber should register");
    }

    let mut spec = host_spec("fixture_native", &manifest_ref);
    spec.config = Map::from_iter([("conditional_gate_test".into(), json!(true))]);
    let (mut activation, report) = crate::PluginHostActivation::initialize_with_verified_specs(
        PluginConfig::default(),
        [spec],
    )
    .await
    .expect("native plugin host should activate callback gates");
    assert!(!report.has_errors());
    flush_subscribers().expect("activation events should flush");
    let baselines = registrations
        .iter()
        .map(|(_, calls)| calls.load(Ordering::SeqCst))
        .collect::<Vec<_>>();

    emit_scope_mark(
        EmitMarkEventParams::builder()
            .name("native-conditional-callback-active")
            .build(),
    )
    .expect("active callback mark should emit");
    flush_subscribers().expect("active callback mark should flush");
    assert_eq!(registrations[0].1.load(Ordering::SeqCst), baselines[0]);
    assert!(registrations[1].1.load(Ordering::SeqCst) > baselines[1]);
    assert!(registrations[2].1.load(Ordering::SeqCst) > baselines[2]);

    activation.close().expect("native plugin host should close");
    emit_scope_mark(
        EmitMarkEventParams::builder()
            .name("native-conditional-callback-cleared")
            .build(),
    )
    .expect("post-clear mark should emit");
    flush_subscribers().expect("post-clear mark should flush");
    assert!(registrations[0].1.load(Ordering::SeqCst) > baselines[0]);
}

#[tokio::test]
async fn native_event_metadata_injector_enriches_events_and_is_removed_on_clear() {
    let _guard = NATIVE_PLUGIN_TEST_LOCK.lock().await;
    let fixture = build_fixture_plugin();
    let manifest_ref = write_manifest(&fixture);
    let mut spec = host_spec("fixture_native", &manifest_ref);
    spec.config = Map::from_iter([("event_metadata_injector_only".into(), json!(true))]);
    let (mut activation, report) = crate::PluginHostActivation::initialize_with_verified_specs(
        PluginConfig::default(),
        [spec],
    )
    .await
    .expect("native plugin host should activate");
    assert!(!report.has_errors());

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    let subscriber_name = "native_event_metadata_injector_events";
    register_subscriber(
        subscriber_name,
        Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
    )
    .expect("test subscriber should register");

    emit_scope_mark(
        EmitMarkEventParams::builder()
            .name("external-plugin-event-metadata-injection-native-before-clear")
            .metadata(json!({"existing": true}))
            .build(),
    )
    .expect("mark should emit");
    flush_subscribers().expect("injected mark should flush");
    let injected = find_event(
        &events.lock().unwrap(),
        "external-plugin-event-metadata-injection-native-before-clear",
        None,
    )
    .clone();
    assert_eq!(injected.metadata().unwrap()["existing"], true);
    assert_eq!(
        injected.metadata().unwrap()["external.injector.transport"],
        "native_dynamic_rust"
    );

    activation.close().expect("native plugin host should close");
    emit_scope_mark(
        EmitMarkEventParams::builder()
            .name("external-plugin-event-metadata-injection-native-after-clear")
            .metadata(json!({"existing": true}))
            .build(),
    )
    .expect("post-clear mark should emit");
    flush_subscribers().expect("post-clear mark should flush");
    let events = events.lock().unwrap();
    let after_clear = find_event(
        &events,
        "external-plugin-event-metadata-injection-native-after-clear",
        None,
    );
    assert_eq!(after_clear.metadata().unwrap(), &json!({"existing": true}));
    drop(events);
    deregister_subscriber(subscriber_name).expect("test subscriber should deregister");
}

#[tokio::test]
async fn native_event_metadata_injector_error_preserves_event_delivery() {
    let _guard = NATIVE_PLUGIN_TEST_LOCK.lock().await;
    let fixture = build_fixture_plugin();
    let manifest_ref = write_manifest(&fixture);
    let mut spec = host_spec("fixture_native", &manifest_ref);
    spec.config = Map::from_iter([
        ("event_metadata_injector_only".into(), json!(true)),
        ("event_metadata_injector_error".into(), json!(true)),
    ]);
    let (mut activation, _) = crate::PluginHostActivation::initialize_with_verified_specs(
        PluginConfig::default(),
        [spec],
    )
    .await
    .expect("native plugin host should activate");

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    let subscriber_name = "native_event_metadata_injector_error_events";
    register_subscriber(
        subscriber_name,
        Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
    )
    .expect("test subscriber should register");
    emit_scope_mark(
        EmitMarkEventParams::builder()
            .name("external-plugin-event-metadata-injection-native-error")
            .metadata(json!({"existing": true}))
            .build(),
    )
    .expect("mark should emit");
    flush_subscribers().expect("failed injector must not block delivery");
    let events = events.lock().unwrap();
    let delivered = find_event(
        &events,
        "external-plugin-event-metadata-injection-native-error",
        None,
    );
    assert_eq!(delivered.metadata().unwrap(), &json!({"existing": true}));
    drop(events);

    deregister_subscriber(subscriber_name).expect("test subscriber should deregister");
    activation.close().expect("native plugin host should close");
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
    let (mut activation, report) = crate::PluginHostActivation::initialize_with_verified_specs(
        base_config,
        [host_spec("fixture_native", &manifest_ref)],
    )
    .await
    .expect("static and dynamic components should activate together");

    assert!(!report.has_errors());
    assert_eq!(STATIC_BASE_REGISTRATIONS.load(Ordering::SeqCst), 1);
    assert!(lookup_plugin(STATIC_BASE_PLUGIN_KIND).is_some());
    assert!(lookup_plugin("fixture_native").is_some());

    activation.close().expect("combined host should close");
    assert_eq!(STATIC_BASE_DEREGISTRATIONS.load(Ordering::SeqCst), 1);
    assert!(lookup_plugin(STATIC_BASE_PLUGIN_KIND).is_some());
    assert!(lookup_plugin("fixture_native").is_none());
    assert!(deregister_plugin(STATIC_BASE_PLUGIN_KIND));
}

#[tokio::test]
async fn plugin_host_activation_layers_discovered_static_base_with_dynamic_components() {
    if std::env::var_os(PLUGIN_DISCOVERY_TEST_CHILD).is_none() {
        let environment = TempDir::new().expect("plugin discovery environment should be created");
        let xdg_config_home = environment.path().join("xdg");
        let user_config_dir = xdg_config_home.join("nemo-relay");
        std::fs::create_dir_all(&user_config_dir)
            .expect("user plugin config directory should be created");
        let fixture = build_fixture_plugin();
        let manifest_ref = write_manifest_with_integrity(&fixture, &sha256(&fixture.library_path));
        std::fs::write(
            user_config_dir.join("plugins.toml"),
            format!(
                concat!(
                    "version = 1\n\n",
                    "[[components]]\nkind = {static_kind:?}\nenabled = true\n\n",
                    "[plugins.policy.defaults]\nattestation = \"integrity_only\"\n\n",
                    "[[plugins.dynamic]]\nmanifest = {manifest_ref:?}\n"
                ),
                static_kind = STATIC_BASE_PLUGIN_KIND,
                manifest_ref = manifest_ref.to_string_lossy(),
            ),
        )
        .expect("user plugin config should be written");
        let legacy_project_dir = environment.path().join(".nemo-relay");
        std::fs::create_dir_all(&legacy_project_dir)
            .expect("legacy project config directory should be created");
        std::fs::write(legacy_project_dir.join("plugins.toml"), "components = [\n")
            .expect("malformed legacy project config should be written");

        let output = Command::new(std::env::current_exe().expect("test executable should resolve"))
            .args([
                "--exact",
                "plugin_host_activation_layers_discovered_static_base_with_dynamic_components",
                "--nocapture",
            ])
            .current_dir(environment.path())
            .env("XDG_CONFIG_HOME", &xdg_config_home)
            .env(PLUGIN_DISCOVERY_TEST_CHILD, "1")
            .env_remove("NEMO_RELAY_TEST_SKIP_IMPLICIT_CONFIG")
            .output()
            .expect("plugin discovery child process should run");
        assert!(
            output.status.success(),
            "plugin discovery child process failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }

    let _guard = NATIVE_PLUGIN_TEST_LOCK.lock().await;
    let _ = deregister_plugin(STATIC_BASE_PLUGIN_KIND);
    STATIC_BASE_REGISTRATIONS.store(0, Ordering::SeqCst);
    STATIC_BASE_DEREGISTRATIONS.store(0, Ordering::SeqCst);
    register_plugin(Arc::new(StaticBasePlugin)).expect("static base plugin should register");

    let mut activation = initialize(PluginConfig::default(), None)
        .await
        .expect("discovered static and dynamic components should activate together");
    let report = activation.report();

    assert!(!report.config.has_errors());
    assert_eq!(STATIC_BASE_REGISTRATIONS.load(Ordering::SeqCst), 1);
    assert!(lookup_plugin(STATIC_BASE_PLUGIN_KIND).is_some());
    assert!(lookup_plugin("fixture_native").is_some());

    activation.close().expect("discovered host should close");
    assert_eq!(STATIC_BASE_DEREGISTRATIONS.load(Ordering::SeqCst), 1);
    assert!(lookup_plugin("fixture_native").is_none());
    assert!(deregister_plugin(STATIC_BASE_PLUGIN_KIND));
}

#[tokio::test]
async fn plugin_host_close_retains_uncooperative_work_for_retry() {
    let _guard = NATIVE_PLUGIN_TEST_LOCK.lock().await;
    let fixture = build_fixture_plugin();
    let manifest_ref = write_manifest(&fixture);
    let (mut activation, _) = crate::PluginHostActivation::initialize_with_verified_specs(
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
                            nemo_relay_types::error::FlowError::Internal(error.to_string())
                        })?;
                        release_rx
                            .lock()
                            .map_err(|error| {
                                nemo_relay_types::error::FlowError::Internal(error.to_string())
                            })?
                            .recv()
                            .map_err(|error| {
                                nemo_relay_types::error::FlowError::Internal(error.to_string())
                            })?;
                        Ok(ToolExecutionResult::new(
                            json!({ "tool_callback": true, "args": args }),
                        ))
                    })
                }))
                .build(),
        ))
    });

    entered_rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("native callback should enter its continuation");
    activation
        .close()
        .expect_err("blocked continuation must retain cleanup ownership");
    let unchanged = tool_request_intercepts("after-clear", json!({ "input": true }))
        .await
        .expect("new calls should observe the cleared registries");
    assert_eq!(unchanged, json!({ "input": true }));

    release_tx
        .send(())
        .expect("in-flight continuation should still be reachable");
    let result = call_thread
        .join()
        .expect("in-flight callback thread should not panic");
    assert!(result.is_err(), "shutdown cancels unfinished callbacks");
    activation
        .close()
        .expect("close retries after blocked work exits");
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

    let caller = tokio::spawn(crate::PluginHostActivation::initialize_with_verified_specs(
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
            if lookup_plugin("fixture_native").is_none() {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("canceled activation should clear its completed host result");

    let (mut activation, _) = crate::PluginHostActivation::initialize_with_verified_specs(
        PluginConfig::default(),
        [host_spec("fixture_native", &manifest_ref)],
    )
    .await
    .expect("canceled activation must release process ownership");
    activation.close().expect("recovered host should close");
    assert!(deregister_plugin("fixture_blocking_host_base"));
}

#[tokio::test]
async fn plugin_host_rejects_empty_dynamic_specs_without_claiming_ownership() {
    let _guard = NATIVE_PLUGIN_TEST_LOCK.lock().await;
    let error = match crate::PluginHostActivation::initialize_with_verified_specs(
        PluginConfig::default(),
        Vec::<VerifiedDynamicPluginSpec>::new(),
    )
    .await
    {
        Ok((mut activation, _)) => {
            activation
                .close()
                .expect("unexpected empty activation should close");
            panic!("an empty dynamic activation must fail");
        }
        Err(error) => error.to_string(),
    };
    assert!(error.contains("at least one dynamic plugin"), "{error}");
    assert!(error.contains("static-only configuration"), "{error}");

    test_initialize_plugin_host_exact(PluginConfig::default())
        .await
        .expect("empty dynamic rejection must leave static initialization available");
    test_close_plugin_host().expect("static configuration should clear");
}

#[tokio::test]
async fn plugin_host_activation_drop_releases_owner_and_plugin_kind() {
    let _guard = NATIVE_PLUGIN_TEST_LOCK.lock().await;
    let fixture = build_fixture_plugin();
    let manifest_ref = write_manifest(&fixture);

    {
        let (activation, _) = crate::PluginHostActivation::initialize_with_verified_specs(
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
    let (mut activation, _) = crate::PluginHostActivation::initialize_with_verified_specs(
        PluginConfig::default(),
        [host_spec("fixture_native", &manifest_ref)],
    )
    .await
    .expect("owner should be reusable after drop");
    activation.close().expect("second plugin host should close");
}

#[tokio::test]
async fn plugin_host_reserved_id_failure_does_not_poison_future_activation() {
    let _guard = NATIVE_PLUGIN_TEST_LOCK.lock().await;
    let fixture = build_fixture_plugin();
    let collision_manifest = write_manifest_for_kind(&fixture, "observability");

    let error = match crate::PluginHostActivation::initialize_with_verified_specs(
        PluginConfig::default(),
        [host_spec("observability", &collision_manifest)],
    )
    .await
    {
        Ok((mut activation, _)) => {
            activation
                .close()
                .expect("unexpected collision activation should close");
            panic!("a dynamic plugin must not replace a builtin kind");
        }
        Err(error) => error.to_string(),
    };
    assert!(error.contains("observability"), "{error}");
    assert!(error.contains("disagrees with binary"), "{error}");

    let manifest_ref = write_manifest(&fixture);
    let (mut activation, _) = crate::PluginHostActivation::initialize_with_verified_specs(
        PluginConfig::default(),
        [host_spec("fixture_native", &manifest_ref)],
    )
    .await
    .expect("a reserved-id failure must not poison later activation");
    activation
        .close()
        .expect("recovered plugin host should close");
}

#[tokio::test]
async fn plugin_host_rejects_duplicate_ids_before_loading() {
    let _guard = NATIVE_PLUGIN_TEST_LOCK.lock().await;
    let fixture = build_fixture_plugin();
    let manifest_ref = write_manifest(&fixture);
    let spec = host_spec("fixture_native", &manifest_ref);

    let error = match crate::PluginHostActivation::initialize_with_verified_specs(
        PluginConfig::default(),
        [spec.clone(), spec],
    )
    .await
    {
        Ok((mut activation, _)) => {
            activation
                .close()
                .expect("unexpected duplicate activation should close");
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

    let (mut activation, _) = crate::PluginHostActivation::initialize_with_verified_specs(
        PluginConfig::default(),
        [host_spec("fixture_native", &manifest_ref)],
    )
    .await
    .expect("duplicate-id rejection must release the owner");
    activation.close().expect("recovered host should close");
}

#[tokio::test]
async fn plugin_host_clear_surfaces_missing_kind_and_releases_safe_owner() {
    let _guard = NATIVE_PLUGIN_TEST_LOCK.lock().await;
    let fixture = build_fixture_plugin();
    let manifest_ref = write_manifest(&fixture);
    let (mut activation, _) = crate::PluginHostActivation::initialize_with_verified_specs(
        PluginConfig::default(),
        [host_spec("fixture_native", &manifest_ref)],
    )
    .await
    .expect("plugin host should activate");

    assert!(deregister_plugin("fixture_native"));
    let error = activation
        .close()
        .expect_err("missing plugin-kind deregistration should be surfaced")
        .to_string();
    assert!(error.contains("fixture_native"), "{error}");
    assert!(error.contains("was not registered"), "{error}");

    let (mut activation, _) = crate::PluginHostActivation::initialize_with_verified_specs(
        PluginConfig::default(),
        [host_spec("fixture_native", &manifest_ref)],
    )
    .await
    .expect("a safely absent plugin kind should release the owner");
    activation.close().expect("recovered host should close");
}

#[tokio::test]
async fn plugin_host_clear_preserves_a_replacement_registration() {
    let _guard = NATIVE_PLUGIN_TEST_LOCK.lock().await;
    let fixture = build_fixture_plugin();
    let manifest_ref = write_manifest(&fixture);
    let (mut activation, _) = crate::PluginHostActivation::initialize_with_verified_specs(
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
        .close()
        .expect_err("replacement during teardown should be surfaced")
        .to_string();
    assert!(error.contains("fixture_native"), "{error}");
    assert!(error.contains("was replaced"), "{error}");
    assert!(error.contains("left registered"), "{error}");

    let registered = lookup_plugin("fixture_native").expect("replacement must remain registered");
    assert!(Arc::ptr_eq(&registered, &replacement));
    assert!(deregister_plugin("fixture_native"));

    let (mut activation, _) = crate::PluginHostActivation::initialize_with_verified_specs(
        PluginConfig::default(),
        [host_spec("fixture_native", &manifest_ref)],
    )
    .await
    .expect("safe replacement detection should release the host owner");
    activation.close().expect("recovered host should close");
}

#[tokio::test]
async fn plugin_host_partial_load_failure_rolls_back_and_releases_owner() {
    let _guard = NATIVE_PLUGIN_TEST_LOCK.lock().await;
    let fixture = build_fixture_plugin();
    let manifest_ref = write_manifest(&fixture);
    let missing_manifest = manifest_ref.with_file_name("missing-relay-plugin.toml");

    let error = match crate::PluginHostActivation::initialize_with_verified_specs(
        PluginConfig::default(),
        [
            host_spec("fixture_native", &manifest_ref),
            host_spec("missing_native", &missing_manifest),
        ],
    )
    .await
    {
        Ok((mut activation, _)) => {
            activation
                .close()
                .expect("unexpected activation should close");
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

    let (mut activation, _) = crate::PluginHostActivation::initialize_with_verified_specs(
        PluginConfig::default(),
        [host_spec("fixture_native", &manifest_ref)],
    )
    .await
    .expect("failed activation must release the owner");
    activation
        .close()
        .expect("recovered activation should close");
}

#[tokio::test]
async fn plugin_host_rejects_a_second_active_host() {
    let _guard = NATIVE_PLUGIN_TEST_LOCK.lock().await;
    let fixture = build_fixture_plugin();
    let manifest_ref = write_manifest(&fixture);
    let mut existing = crate::PluginHostActivation::initialize_exact(PluginConfig::default())
        .await
        .expect("first host should initialize");

    let error = match crate::PluginHostActivation::initialize_with_verified_specs(
        PluginConfig::default(),
        [host_spec("fixture_native", &manifest_ref)],
    )
    .await
    {
        Ok((mut activation, _)) => {
            activation
                .close()
                .expect("unexpected activation should close");
            panic!("host activation should reject an existing configuration");
        }
        Err(error) => error.to_string(),
    };
    assert!(
        error.contains("plugin configuration is owned by an active dynamic plugin host"),
        "{error}"
    );
    existing.close().expect("first host should close");
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

fn host_spec(plugin_id: &str, manifest_ref: &Path) -> VerifiedDynamicPluginSpec {
    VerifiedDynamicPluginSpec {
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
    manifest_dir: TempDir,
    library_path: PathBuf,
}

fn checked_fixture_library(
    fixture: &BuiltFixture,
) -> crate::dynamic::native_library::CheckedNativeLibrary {
    use crate::dynamic::native_library::{CheckedNativeLibrary, NativeLibraryArtifact};
    let file = PathBuf::from(fixture.library_path.file_name().unwrap());
    let artifact = NativeLibraryArtifact::new(
        file.clone(),
        [(file, std::fs::read(&fixture.library_path).unwrap())],
    )
    .unwrap();
    CheckedNativeLibrary::load(artifact, "v1.0", "fixture_native").unwrap()
}

fn build_fixture_plugin() -> BuiltFixture {
    log::set_max_level(log::LevelFilter::Info);
    let manifest_dir = TempDir::new().expect("fixture manifest dir");
    let library_path = prepared_plugin_fixture("NEMO_RELAY_TEST_NATIVE_PLUGIN");
    assert!(
        library_path.exists(),
        "fixture library is missing; run `just build-test-plugin-fixtures`: {}",
        library_path.display()
    );

    BuiltFixture {
        manifest_dir,
        library_path,
    }
}

fn prepared_plugin_fixture(environment: &str) -> PathBuf {
    std::env::var_os(environment)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../target/test-plugin-fixtures/debug")
                .join(fixture_library_name())
        })
}

fn write_manifest(fixture: &BuiltFixture) -> PathBuf {
    write_manifest_text(ManifestOptions {
        manifest_dir: fixture.manifest_dir.path(),
        plugin_id: "fixture_native",
        relay: ">=1.0.0,<2.0",
        library: &fixture.library_path.to_string_lossy(),

        integrity: None,
    })
}

fn write_manifest_for_kind(fixture: &BuiltFixture, plugin_id: &str) -> PathBuf {
    write_manifest_text(ManifestOptions {
        manifest_dir: fixture.manifest_dir.path(),
        plugin_id,
        relay: ">=1.0.0,<2.0",
        library: &fixture.library_path.to_string_lossy(),

        integrity: None,
    })
}

fn write_manifest_with_integrity(fixture: &BuiltFixture, integrity: &str) -> PathBuf {
    write_manifest_text(ManifestOptions {
        manifest_dir: fixture.manifest_dir.path(),
        plugin_id: "fixture_native",
        relay: ">=1.0.0,<2.0",
        library: &fixture.library_path.to_string_lossy(),

        integrity: Some(integrity),
    })
}

struct ManifestOptions<'a> {
    manifest_dir: &'a Path,
    plugin_id: &'a str,
    relay: &'a str,
    library: &'a str,

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
    let manifest = format!(
        r#"
manifest_version = 1

[plugin]
id = {plugin_id}
kind = "rust_dynamic"

[compat]
relay = {relay}
native_api = "v1.0"

[defaults]
enabled = false

[capabilities]
items = ["plugin_native"]

[source]
artifact = {library}

[load]
library = {library}
{integrity}
"#,
        plugin_id = plugin_id,
        relay = relay,
        library = library,
        integrity = integrity,
    );
    std::fs::write(&manifest_ref, manifest).expect("write relay-plugin.toml");
    manifest_ref
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

#[tokio::test]
async fn sdk_cdylib_native_diagnostics_records_reads_and_closes_selected_context() {
    use nemo_relay_diagnostics::*;
    let _guard = NATIVE_PLUGIN_TEST_LOCK.lock().await;
    let fixture = build_fixture_plugin();
    let manifest = write_manifest_with_integrity(&fixture, &sha256(&fixture.library_path));
    for (enabled, grant, failure) in [
        (false, false, false),
        (true, false, false),
        (true, true, false),
        (true, false, true),
    ] {
        let context = Diagnostics::from_settings(
            SourceIdentity {
                instance_id: Uuid::now_v7(),
                process_id: None,
                process_role: "native-integration".into(),
                relay_version: "test".into(),
            },
            DiagnosticsSettings {
                enabled,
                ..Default::default()
            },
        )
        .unwrap();
        let config = fixture.manifest_dir.path().join("diagnostics-plugins.toml");
        std::fs::write(
            &config,
            format!(
                r#"
[plugins.policy.defaults]
attestation = "integrity_only"
[plugins.policy.overrides.fixture_native]
native_diagnostics_read_context = {grant}
[[plugins.dynamic]]
manifest = {manifest}
[plugins.dynamic.config]
diagnostics_fixture = true
expect_read = {grant}
registration_failure = {failure}
"#,
                manifest = toml_string(&manifest.to_string_lossy())
            ),
        )
        .unwrap();
        let activated =
            crate::initialize_with_diagnostics(PluginConfig::default(), Some(config), &context)
                .await;
        if failure {
            assert!(
                activated
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("fixture registration failure")
            );
            let page = context
                .snapshot(&DiagnosticQuery::default())
                .unwrap()
                .to_snapshot()
                .unwrap();
            let internal = page
                .components
                .iter()
                .find(|c| c.identity.key == "fixture.internal")
                .unwrap();
            assert!(internal.closed_at.is_some());
            assert_eq!(internal.lifetime.totals.occurrences_total.value.get(), 5);
            continue;
        }
        let mut host = activated.unwrap();
        let args = json!({"input": "unchanged"});
        assert_eq!(
            tool_request_intercepts("fixture.lookup", args.clone())
                .await
                .unwrap(),
            args
        );
        if enabled {
            let page = context
                .snapshot(&DiagnosticQuery::default())
                .unwrap()
                .to_snapshot()
                .unwrap();
            let internal = page
                .components
                .iter()
                .find(|c| c.identity.key == "fixture.internal")
                .unwrap();
            assert_eq!(internal.lifetime.totals.occurrences_total.value.get(), 6);
            assert_eq!(
                internal.lifetime.totals.recorded_events_total.value.get(),
                3
            );
            let evidence: Vec<_> = page
                .events
                .iter()
                .filter(|e| e.message() == Some("native-private-sentinel"))
                .collect();
            assert_eq!(evidence.len(), 3);
            for (event, session) in evidence[..2]
                .iter()
                .zip(["native-session-2", "native-session-3"])
            {
                assert_eq!(
                    event.correlation.request_id.as_deref(),
                    Some("native-request")
                );
                assert_eq!(event.correlation.session_id.as_deref(), Some(session));
            }
            assert_eq!(evidence[2].correlation, DiagnosticCorrelation::default());
            assert_eq!(
                evidence[0].context.phase.as_ref().unwrap().as_str(),
                "activation"
            );
            assert_eq!(
                evidence[2].context.phase.as_ref().unwrap().as_str(),
                "runtime"
            );
            assert!(
                !host
                    .report()
                    .config
                    .runtime_diagnostics
                    .iter()
                    .any(|d| d.message.contains("native-private-sentinel"))
            );
        } else {
            assert_eq!(context.memory_usage().reserved_bytes, 0);
        }
        host.close().unwrap();
        if enabled {
            let page = context
                .snapshot(&DiagnosticQuery::default())
                .unwrap()
                .to_snapshot()
                .unwrap();
            let internal = page
                .components
                .iter()
                .find(|c| c.identity.key == "fixture.internal")
                .unwrap();
            assert!(internal.closed_at.is_some());
            assert_eq!(internal.lifetime.totals.occurrences_total.value.get(), 6);
        }
    }
}

#[tokio::test]
async fn checked_raw_sync_callbacks_and_selected_diagnostics_preserve_contracts() {
    use crate::dynamic::native_checked::CheckedPlugin;
    use nemo_relay_diagnostics::{
        DiagnosticQuery, Diagnostics, DiagnosticsSettings, SourceIdentity,
    };
    let _guard = NATIVE_PLUGIN_TEST_LOCK.lock().await;
    let fixture = build_fixture_plugin();
    let library = checked_fixture_library(&fixture);
    for read in [false, true] {
        for fail in [false, true] {
            let context = Diagnostics::from_settings(
                SourceIdentity {
                    instance_id: Uuid::now_v7(),
                    process_id: None,
                    process_role: "checked-native-test".into(),
                    relay_version: "test".into(),
                },
                DiagnosticsSettings {
                    enabled: true,
                    ..Default::default()
                },
            )
            .unwrap();
            register_plugin(Arc::new(
                CheckedPlugin::new(&library, context.clone(), read).unwrap(),
            ))
            .unwrap();
            let config = PluginConfig {
                components: vec![PluginComponentSpec {
                    kind: "fixture_native".into(),
                    enabled: true,
                    config: Map::from_iter([
                        ("diagnostics_fixture".into(), json!(true)),
                        ("expect_read".into(), json!(read)),
                        ("registration_failure".into(), json!(fail)),
                    ]),
                }],
                ..Default::default()
            };
            let result = test_initialize_plugin_host_exact(config).await;
            if fail {
                assert!(result.is_err());
            } else {
                result.unwrap();
                let value = tool_request_intercepts("checked-diagnostic", json!({"test":true}))
                    .await
                    .unwrap();
                assert_eq!(value, json!({"test":true}));
            }
            let snapshot = context
                .reader()
                .snapshot(&DiagnosticQuery::default())
                .unwrap();
            assert!(snapshot.events().next().is_some());
            test_close_plugin_host().unwrap();
            assert!(deregister_plugin("fixture_native"));
        }
    }
    register_plugin(Arc::new(
        CheckedPlugin::new(&library, Diagnostics::process_default().clone(), false).unwrap(),
    ))
    .unwrap();
    test_initialize_plugin_host_exact(PluginConfig {
        components: vec![PluginComponentSpec {
            kind: "fixture_native".into(),
            enabled: true,
            config: Map::from_iter([("raw_sync_fixture".into(), json!(true))]),
        }],
        ..Default::default()
    })
    .await
    .unwrap();
    assert_eq!(
        tool_request_intercepts("raw", json!({})).await.unwrap()["checked_raw_sync"],
        true
    );
    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    register_subscriber(
        "checked-raw-observer",
        Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();
    emit_scope_mark(
        EmitMarkEventParams::builder()
            .name("checked-raw-sync-trigger")
            .build(),
    )
    .unwrap();
    emit_scope_mark(
        EmitMarkEventParams::builder()
            .name("checked-raw-panic")
            .build(),
    )
    .unwrap();
    flush_subscribers().unwrap();
    assert!(
        events
            .lock()
            .unwrap()
            .iter()
            .any(|event| event.name() == "checked-raw-sync-observed")
    );
    deregister_subscriber("checked-raw-observer").unwrap();
    test_close_plugin_host().unwrap();
    assert!(deregister_plugin("fixture_native"));
}

#[test]
fn native_activation_rejects_legacy_and_future_contracts_before_opening() {
    let _guard = NATIVE_PLUGIN_TEST_LOCK.blocking_lock();
    let fixture = build_fixture_plugin();
    for (api, expected) in [
        ("1", "rebuild"),
        ("v1.1", "unsupported native API"),
        ("v2.0", "unsupported native API"),
    ] {
        let manifest = write_manifest(&fixture);
        let text = std::fs::read_to_string(&manifest)
            .unwrap()
            .replace("v1.0", api)
            .replace(
                &fixture.library_path.to_string_lossy().to_string(),
                "/missing/native/library",
            );
        let text = if api == "1" {
            text.replace("relay = \">=1.0.0,<2.0\"", "relay = \">=0.8,<1.0\"")
        } else {
            text
        };
        std::fs::write(&manifest, text).unwrap();
        let error = expect_native_load_error(
            load_spec("fixture_native", &manifest),
            "unsupported API must fail",
        );
        assert!(error.contains(expected), "{error}");
    }
    let manifest = write_manifest(&fixture);
    let mut text = std::fs::read_to_string(&manifest).unwrap();
    text.push_str("\nsymbol = \"ignored_symbol\"\n");
    std::fs::write(&manifest, text).unwrap();
    let error =
        expect_native_load_error(load_spec("fixture_native", &manifest), "symbol must fail");
    assert!(error.contains("load.symbol"), "{error}");
}

#[tokio::test]
async fn raw_stream_factories_can_read_downstream_with_one_executor_worker() {
    use crate::dynamic::native_checked::CheckedPlugin;
    let _guard = NATIVE_PLUGIN_TEST_LOCK.lock().await;
    let fixture = build_fixture_plugin();
    let library = checked_fixture_library(&fixture);
    register_plugin(Arc::new(
        CheckedPlugin::new(
            &library,
            nemo_relay_diagnostics::Diagnostics::process_default().clone(),
            false,
        )
        .unwrap(),
    ))
    .unwrap();
    test_initialize_plugin_host_exact(PluginConfig {
        components: vec![PluginComponentSpec {
            kind: "fixture_native".into(),
            enabled: true,
            config: Map::from_iter([
                ("raw_stream_only_fixture".into(), json!(true)),
                ("raw_stream_peek_fixture".into(), json!(true)),
                ("executor".into(), json!({"worker_threads":1})),
            ]),
        }],
        ..Default::default()
    })
    .await
    .unwrap();
    let chunks = tokio::time::timeout(std::time::Duration::from_secs(5), async {
        let mut stream = llm_stream_call_execute(
            LlmStreamCallExecuteParams::builder()
                .name("raw-stream-peek")
                .request(LlmRequest {
                    headers: Map::new(),
                    content: json!({}),
                })
                .func(Arc::new(|_| {
                    Box::pin(async {
                        Ok(LlmJsonStream::new(tokio_stream::iter(vec![
                            Ok(json!(1)),
                            Ok(json!(2)),
                        ])))
                    })
                }))
                .collector(Box::new(|_| Ok(())))
                .finalizer(Box::new(|| Json::Null))
                .build(),
        )
        .await
        .unwrap();
        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk.unwrap());
        }
        stream.close().await.unwrap();
        chunks
    })
    .await
    .expect("a synchronous factory must allow downstream interceptors to run");
    assert_eq!(chunks, vec![json!(1), json!(2)]);
    test_close_plugin_host().unwrap();
    assert!(deregister_plugin("fixture_native"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_shutdown_unwinds_pending_plugin_scope() {
    use crate::dynamic::native_checked::CheckedPlugin;
    let _guard = NATIVE_PLUGIN_TEST_LOCK.lock().await;
    let fixture = build_fixture_plugin();
    let library = checked_fixture_library(&fixture);
    register_plugin(Arc::new(
        CheckedPlugin::new(
            &library,
            nemo_relay_diagnostics::Diagnostics::process_default().clone(),
            false,
        )
        .unwrap(),
    ))
    .unwrap();
    test_initialize_plugin_host_exact(PluginConfig {
        components: vec![PluginComponentSpec {
            kind: "fixture_native".into(),
            enabled: true,
            config: Map::new(),
        }],
        ..Default::default()
    })
    .await
    .unwrap();
    let stack = create_scope_stack();
    let (tx, rx) = tokio::sync::oneshot::channel();
    let sender = Arc::new(Mutex::new(Some(tx)));
    let task = tokio::spawn(TASK_SCOPE_STACK.scope(stack.clone(), async move {
        tool_call_execute(
            ToolCallExecuteParams::builder()
                .name("review-pending-tool")
                .args(json!({"use_scoped_next":true}))
                .func(Arc::new(move |_args| {
                    let sender = sender.clone();
                    Box::pin(async move {
                        if let Some(tx) = sender.lock().unwrap().take() {
                            let _ = tx.send(());
                        }
                        std::future::pending::<()>().await;
                        Ok(ToolExecutionResult::new(json!({})))
                    })
                }))
                .build(),
        )
        .await
    }));
    tokio::time::timeout(std::time::Duration::from_secs(10), rx)
        .await
        .unwrap()
        .unwrap();
    let close = test_close_plugin_host();
    let result = tokio::time::timeout(std::time::Duration::from_secs(10), task).await;
    deregister_plugin("fixture_native");
    let snapshot = nemo_relay_runtime::api::runtime::scope_stack_snapshot(&stack).unwrap();
    close.unwrap();
    assert!(
        result.unwrap().unwrap().is_err(),
        "shutdown cancels pending calls"
    );
    assert_eq!(snapshot.scopes.len(), 1, "only the caller root remains");
}
