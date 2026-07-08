// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end coverage for the Rust gRPC worker SDK service.

use std::future::Future;
use std::net::{SocketAddr, TcpListener};
use std::path::Path;
#[cfg(unix)]
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures_util::{Stream, StreamExt};
#[cfg(unix)]
use hyper_util::rt::TokioIo;
use nemo_relay_types::api::event::{BaseEvent, Event, MarkEvent, PendingMarkSpec};
use nemo_relay_worker::{
    Json, JsonStream, LlmNext, LlmRequest, LlmStreamNext, PluginContext, PluginRuntime, Result,
    ScopeType, ToolExecutionInterceptOutcome, ToolNext, WorkerPlugin, WorkerSdkError,
    WorkerServerConfig, serve_plugin, serve_plugin_arc, serve_plugin_arc_with_config,
};
use nemo_relay_worker_proto::v1::plugin_worker_client::PluginWorkerClient;
use nemo_relay_worker_proto::v1::relay_host_runtime_server::{
    RelayHostRuntime, RelayHostRuntimeServer,
};
use nemo_relay_worker_proto::v1::{
    CancelInvocationRequest, CreateScopeStackRequest, CreateScopeStackResponse,
    DropScopeStackRequest, EmitMarkRequest, HandshakeRequest, HealthRequest, HostAck,
    InvokeRequest, InvokeResponse, JsonEnvelope, JsonResult, LlmInvocation, LlmNextRequest,
    LlmStreamNextRequest, PopScopeRequest, PushScopeRequest, PushScopeResponse, RegisterRequest,
    RegistrationSurface, ScopeContext, ShutdownRequest, StreamChunk, ToolInvocation,
    ToolNextRequest, ValidateRequest, WorkerError,
};
use nemo_relay_worker_proto::{WORKER_PROTOCOL_GRPC_V1, decode_json_envelope, json_envelope};
use serde_json::json;
#[cfg(unix)]
use tokio::net::{UnixListener, UnixStream};
use tokio::task::JoinHandle;
#[cfg(unix)]
use tokio_stream::wrappers::UnixListenerStream;
#[cfg(unix)]
use tonic::transport::Endpoint;
use tonic::transport::{Channel, Server};
use tonic::{Request, Response, Status};
#[cfg(unix)]
use tower::service_fn;

const ACTIVATION_ID: &str = "activation-1";
const AUTH_TOKEN: &str = "secret-token";
const PLUGIN_ID: &str = "acme.worker";
const REQUIRED_WORKER_ENVS: &[&str] = &[
    "NEMO_RELAY_WORKER_SOCKET",
    "NEMO_RELAY_HOST_SOCKET",
    "NEMO_RELAY_WORKER_ID",
    "NEMO_RELAY_WORKER_TOKEN",
];
static ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

#[tokio::test(flavor = "multi_thread")]
async fn worker_service_enforces_auth_and_reports_registrations() {
    let (handle, mut client) = spawn_worker(
        Arc::new(SurfacePlugin::default()),
        "http://127.0.0.1:9".into(),
    )
    .await;

    let bad_handshake = client
        .handshake(Request::new(HandshakeRequest {
            activation_id: ACTIVATION_ID.into(),
            plugin_id: PLUGIN_ID.into(),
            relay_version: "0.5.0".into(),
            worker_protocol: WORKER_PROTOCOL_GRPC_V1.into(),
            auth_token: "bad-token".into(),
            host_endpoint: "http://127.0.0.1:9".into(),
        }))
        .await
        .expect_err("invalid token should fail");
    assert_eq!(bad_handshake.code(), tonic::Code::PermissionDenied);

    let bad_activation = client
        .handshake(Request::new(HandshakeRequest {
            activation_id: "wrong-activation".into(),
            plugin_id: PLUGIN_ID.into(),
            relay_version: "0.5.0".into(),
            worker_protocol: WORKER_PROTOCOL_GRPC_V1.into(),
            auth_token: AUTH_TOKEN.into(),
            host_endpoint: "http://127.0.0.1:9".into(),
        }))
        .await
        .expect_err("invalid activation should fail");
    assert_eq!(bad_activation.code(), tonic::Code::PermissionDenied);

    let handshake = client
        .handshake(Request::new(HandshakeRequest {
            activation_id: ACTIVATION_ID.into(),
            plugin_id: PLUGIN_ID.into(),
            relay_version: "0.5.0".into(),
            worker_protocol: WORKER_PROTOCOL_GRPC_V1.into(),
            auth_token: AUTH_TOKEN.into(),
            host_endpoint: "http://127.0.0.1:9".into(),
        }))
        .await
        .expect("handshake succeeds")
        .into_inner();
    assert_eq!(handshake.plugin_id, PLUGIN_ID);
    assert!(
        handshake
            .supported_surfaces
            .contains(&(RegistrationSurface::LlmStreamExecutionIntercept as i32))
    );

    let bad_health = client
        .health(Request::new(HealthRequest {
            activation_id: ACTIVATION_ID.into(),
            auth_token: "bad-token".into(),
        }))
        .await
        .expect_err("health auth should fail");
    assert_eq!(bad_health.code(), tonic::Code::PermissionDenied);

    let health = client
        .health(Request::new(HealthRequest {
            activation_id: ACTIVATION_ID.into(),
            auth_token: AUTH_TOKEN.into(),
        }))
        .await
        .expect("health succeeds")
        .into_inner();
    assert!(health.ok);
    assert_eq!(health.message, "ready");
    assert_eq!(health.plugin_id, PLUGIN_ID);
    assert_eq!(health.worker_protocol, WORKER_PROTOCOL_GRPC_V1);
    assert_eq!(health.sdk_name, "nemo-relay-worker");
    assert_eq!(health.runtime_name, "rust");

    let validate_err = client
        .validate(Request::new(ValidateRequest {
            activation_id: ACTIVATION_ID.into(),
            plugin_id: PLUGIN_ID.into(),
            auth_token: "bad-token".into(),
            config: Some(json_env(json!({"diagnostic": true}))),
        }))
        .await
        .expect_err("validate auth should fail");
    assert_eq!(validate_err.code(), tonic::Code::PermissionDenied);

    let invalid_validate_config = client
        .validate(Request::new(ValidateRequest {
            activation_id: ACTIVATION_ID.into(),
            plugin_id: PLUGIN_ID.into(),
            auth_token: AUTH_TOKEN.into(),
            config: Some(invalid_json_env("nemo.relay.Json@1")),
        }))
        .await
        .expect_err("invalid validate config should fail");
    assert_eq!(invalid_validate_config.code(), tonic::Code::InvalidArgument);

    let diagnostics = client
        .validate(Request::new(ValidateRequest {
            activation_id: ACTIVATION_ID.into(),
            plugin_id: PLUGIN_ID.into(),
            auth_token: AUTH_TOKEN.into(),
            config: Some(json_env(json!({"diagnostic": true}))),
        }))
        .await
        .expect("validate succeeds")
        .into_inner()
        .diagnostics
        .expect("diagnostics envelope");
    let diagnostics: Vec<nemo_relay_worker::ConfigDiagnostic> =
        decode_json_envelope(&diagnostics).expect("decode diagnostics");
    assert_eq!(diagnostics.len(), 1);

    let register_err = client
        .register(Request::new(RegisterRequest {
            activation_id: ACTIVATION_ID.into(),
            plugin_id: PLUGIN_ID.into(),
            auth_token: "bad-token".into(),
            config: Some(json_env(json!({}))),
        }))
        .await
        .expect_err("register auth should fail");
    assert_eq!(register_err.code(), tonic::Code::PermissionDenied);

    let invalid_register_config = client
        .register(Request::new(RegisterRequest {
            activation_id: ACTIVATION_ID.into(),
            plugin_id: PLUGIN_ID.into(),
            auth_token: AUTH_TOKEN.into(),
            config: Some(invalid_json_env("nemo.relay.Json@1")),
        }))
        .await
        .expect_err("invalid register config should fail");
    assert_eq!(invalid_register_config.code(), tonic::Code::InvalidArgument);

    let registrations = register_plugin(&mut client).await;
    assert_eq!(registrations.len(), 19);
    assert_eq!(
        registrations
            .iter()
            .filter(|registration| registration.local_name == "tool-sanitize")
            .count(),
        2
    );
    assert_eq!(
        registrations
            .iter()
            .filter(|registration| registration.local_name == "event-sanitize")
            .count(),
        3
    );

    let invoke_err = client
        .invoke(Request::new(InvokeRequest {
            auth_token: "bad-token".into(),
            ..tool_invoke(
                "tool-request",
                RegistrationSurface::ToolRequestIntercept,
                json!({}),
            )
        }))
        .await
        .expect_err("invoke auth should fail");
    assert_eq!(invoke_err.code(), tonic::Code::PermissionDenied);

    let cancel = client
        .cancel_invocation(Request::new(CancelInvocationRequest {
            activation_id: ACTIVATION_ID.into(),
            invocation_id: "invoke-1".into(),
            auth_token: AUTH_TOKEN.into(),
            reason: "test".into(),
        }))
        .await
        .expect("cancel returns ack")
        .into_inner();
    assert!(!cancel.accepted);
    assert!(cancel.message.contains("not active"));

    let shutdown = client
        .shutdown(Request::new(ShutdownRequest {
            activation_id: ACTIVATION_ID.into(),
            auth_token: AUTH_TOKEN.into(),
            reason: "test".into(),
        }))
        .await
        .expect("shutdown returns ack")
        .into_inner();
    assert!(!shutdown.accepted);
    assert!(shutdown.message.contains("not implemented"));

    handle.abort();
}

#[tokio::test(flavor = "multi_thread")]
async fn worker_service_rejects_duplicate_registration_names_on_one_surface() {
    let (handle, mut client) = spawn_worker(
        Arc::new(DuplicateEventSanitizerPlugin),
        "http://127.0.0.1:9".into(),
    )
    .await;

    let response = client
        .register(Request::new(RegisterRequest {
            activation_id: ACTIVATION_ID.into(),
            plugin_id: PLUGIN_ID.into(),
            auth_token: AUTH_TOKEN.into(),
            config: Some(json_env(json!({}))),
        }))
        .await
        .expect("duplicate registration should return protocol data")
        .into_inner();
    assert!(response.registrations.is_empty());
    let error = response
        .error
        .expect("duplicate registration should return an error");
    assert_eq!(error.code, "worker.error");
    assert_eq!(
        error.message,
        "invalid input: duplicate registration 'duplicate' for surface MARK_SANITIZE_GUARDRAIL"
    );

    handle.abort();
}

#[tokio::test(flavor = "multi_thread")]
async fn worker_service_cancels_unary_and_stream_invocations_by_id() {
    let timeout = Duration::from_secs(2);
    let plugin = Arc::new(CancellationPlugin::default());
    let (handle, mut client) = spawn_worker(plugin.clone(), "http://127.0.0.1:9".into()).await;
    tokio::time::timeout(timeout, register_plugin(&mut client))
        .await
        .expect("worker registration should complete");

    let mut unary_client = client.clone();
    let unary_task = tokio::spawn(async move {
        unary_client
            .invoke(Request::new(InvokeRequest {
                invocation_id: "cancel-unary".into(),
                ..tool_invoke(
                    "cancel-unary",
                    RegistrationSurface::ToolExecutionIntercept,
                    json!({}),
                )
            }))
            .await
            .expect("cancelled unary invocation should return a response")
            .into_inner()
    });
    tokio::time::timeout(timeout, plugin.unary_started.notified())
        .await
        .expect("unary invocation should start");
    let unary_ack = tokio::time::timeout(
        timeout,
        client.cancel_invocation(Request::new(CancelInvocationRequest {
            activation_id: ACTIVATION_ID.into(),
            invocation_id: "cancel-unary".into(),
            auth_token: AUTH_TOKEN.into(),
            reason: "test timeout".into(),
        })),
    )
    .await
    .expect("unary cancellation should complete")
    .expect("active unary cancellation should return an ack")
    .into_inner();
    let unary_response = tokio::time::timeout(timeout, unary_task)
        .await
        .expect("unary invocation should finish after cancellation")
        .expect("unary client task should join");
    let unary_error = match unary_response.result {
        Some(nemo_relay_worker_proto::v1::invoke_response::Result::Error(error)) => error,
        other => panic!("expected cancellation error, got {other:?}"),
    };
    assert!(unary_ack.accepted);
    assert_eq!(unary_error.code, "worker.cancelled");
    assert!(plugin.unary_cancelled.load(Ordering::SeqCst));

    let repeated = tokio::time::timeout(
        timeout,
        client.cancel_invocation(Request::new(CancelInvocationRequest {
            activation_id: ACTIVATION_ID.into(),
            invocation_id: "cancel-unary".into(),
            auth_token: AUTH_TOKEN.into(),
            reason: "repeat".into(),
        })),
    )
    .await
    .expect("repeated cancellation should complete")
    .expect("repeated cancellation should return an ack")
    .into_inner();
    assert!(!repeated.accepted);
    assert!(repeated.message.contains("not active"));

    let mut stream = tokio::time::timeout(
        timeout,
        client.invoke_stream(Request::new(InvokeRequest {
            invocation_id: "cancel-stream".into(),
            ..llm_invoke(
                "cancel-stream",
                RegistrationSurface::LlmStreamExecutionIntercept,
                llm_request(),
                None,
                None,
            )
        })),
    )
    .await
    .expect("stream invocation setup should complete")
    .expect("stream invocation should start")
    .into_inner();
    tokio::time::timeout(timeout, plugin.stream_started.notified())
        .await
        .expect("stream invocation should become active");
    let stream_ack = tokio::time::timeout(
        timeout,
        client.cancel_invocation(Request::new(CancelInvocationRequest {
            activation_id: ACTIVATION_ID.into(),
            invocation_id: "cancel-stream".into(),
            auth_token: AUTH_TOKEN.into(),
            reason: "stream abandoned".into(),
        })),
    )
    .await
    .expect("stream cancellation should complete")
    .expect("active stream cancellation should return an ack")
    .into_inner();
    let stream_error = tokio::time::timeout(timeout, async {
        loop {
            let chunk = stream
                .next()
                .await
                .expect("cancelled stream should yield a terminal error")
                .expect("cancelled stream chunk should be protocol data");
            if let Some(nemo_relay_worker_proto::v1::stream_chunk::Item::Error(error)) = chunk.item
            {
                break error;
            }
        }
    })
    .await
    .expect("cancelled stream should terminate");
    assert!(stream_ack.accepted);
    assert_eq!(stream_error.code, "worker.cancelled");
    assert!(plugin.stream_cancelled.load(Ordering::SeqCst));

    handle.abort();
}

#[tokio::test(flavor = "multi_thread")]
async fn worker_service_cancels_stream_during_async_setup() {
    let timeout = Duration::from_secs(2);
    let plugin = Arc::new(CancellationPlugin::default());
    let (handle, mut client) = spawn_worker(plugin.clone(), "http://127.0.0.1:9".into()).await;
    tokio::time::timeout(timeout, register_plugin(&mut client))
        .await
        .expect("worker registration should complete");

    let mut stream_client = client.clone();
    let stream_task = tokio::spawn(async move {
        stream_client
            .invoke_stream(Request::new(InvokeRequest {
                invocation_id: "cancel-stream-setup".into(),
                ..llm_invoke(
                    "cancel-stream-setup",
                    RegistrationSurface::LlmStreamExecutionIntercept,
                    llm_request(),
                    None,
                    None,
                )
            }))
            .await
    });
    tokio::time::timeout(timeout, plugin.stream_setup_started.notified())
        .await
        .expect("stream setup should start");
    let ack = tokio::time::timeout(
        timeout,
        client.cancel_invocation(Request::new(CancelInvocationRequest {
            activation_id: ACTIVATION_ID.into(),
            invocation_id: "cancel-stream-setup".into(),
            auth_token: AUTH_TOKEN.into(),
            reason: "cancel setup".into(),
        })),
    )
    .await
    .expect("stream setup cancellation should complete")
    .expect("stream setup cancellation should return an ack")
    .into_inner();
    let mut stream = tokio::time::timeout(timeout, stream_task)
        .await
        .expect("cancelled stream setup should finish")
        .expect("stream setup client task should join")
        .expect("cancelled stream setup should return protocol data")
        .into_inner();
    let chunk = tokio::time::timeout(timeout, stream.next())
        .await
        .expect("cancelled setup stream should terminate")
        .expect("cancelled setup stream should yield a terminal error")
        .expect("cancelled setup stream chunk should be protocol data");
    let error = match chunk.item {
        Some(nemo_relay_worker_proto::v1::stream_chunk::Item::Error(error)) => error,
        other => panic!("expected stream setup cancellation error, got {other:?}"),
    };

    assert!(ack.accepted);
    assert_eq!(error.code, "worker.cancelled");
    assert!(plugin.stream_setup_cancelled.load(Ordering::SeqCst));

    handle.abort();
}

#[tokio::test(flavor = "multi_thread")]
async fn worker_service_invokes_every_registration_surface() {
    let host = MockHost::default();
    let (host_handle, host_endpoint) = spawn_host(host.clone()).await;
    let plugin = Arc::new(SurfacePlugin::default());
    let events = plugin.events.clone();
    let (worker_handle, mut client) = spawn_worker(plugin, tcp_endpoint(&host_endpoint)).await;
    register_plugin(&mut client).await;

    let subscriber_response = client
        .invoke(Request::new(event_invoke("subscriber")))
        .await
        .expect("subscriber invoke")
        .into_inner();
    assert_empty_response(subscriber_response);
    assert_eq!(
        events.lock().expect("events lock").as_slice(),
        ["subscriber-event"]
    );

    let mark_fields = invoke_json(
        &mut client,
        event_invoke_surface("event-sanitize", RegistrationSurface::MarkSanitizeGuardrail),
    )
    .await;
    assert_eq!(mark_fields["data"]["phase"], "mark");
    let start_fields = invoke_json(
        &mut client,
        event_invoke_surface(
            "event-sanitize",
            RegistrationSurface::ScopeSanitizeStartGuardrail,
        ),
    )
    .await;
    assert_eq!(start_fields["metadata"]["phase"], "scope_start");
    let end_fields = invoke_json(
        &mut client,
        event_invoke_surface(
            "event-sanitize",
            RegistrationSurface::ScopeSanitizeEndGuardrail,
        ),
    )
    .await;
    assert_eq!(end_fields["metadata"]["phase"], "scope_end");

    assert_json_field(
        invoke_json(
            &mut client,
            tool_invoke(
                "tool-sanitize",
                RegistrationSurface::ToolSanitizeRequestGuardrail,
                json!({}),
            ),
        )
        .await,
        "phase",
        "tool_sanitize_request",
    );
    assert_json_field(
        invoke_json(
            &mut client,
            tool_invoke(
                "tool-sanitize",
                RegistrationSurface::ToolSanitizeResponseGuardrail,
                json!({}),
            ),
        )
        .await,
        "phase",
        "tool_sanitize_response",
    );
    assert_eq!(
        invoke_guardrail(
            &mut client,
            tool_invoke(
                "tool-conditional",
                RegistrationSurface::ToolConditionalExecutionGuardrail,
                json!({"block": true}),
            ),
        )
        .await,
        "blocked-tool"
    );
    assert_json_field(
        invoke_json(
            &mut client,
            tool_invoke(
                "tool-request",
                RegistrationSurface::ToolRequestIntercept,
                json!({}),
            ),
        )
        .await,
        "phase",
        "tool_request",
    );
    let tool_outcome = invoke_tool_execution(
        &mut client,
        tool_invoke(
            "tool-exec",
            RegistrationSurface::ToolExecutionIntercept,
            json!({}),
        ),
    )
    .await;
    assert_eq!(tool_outcome.pending_marks.len(), 1);
    assert_eq!(tool_outcome.pending_marks[0].name, "worker.tool.execution");
    let tool_exec = tool_outcome.result;
    assert_json_field(tool_exec.clone(), "next", "tool");
    assert_json_field(tool_exec, "phase", "tool_exec");
    assert!(
        invoke_json(
            &mut client,
            tool_invoke(
                "tool-scope-types",
                RegistrationSurface::ToolExecutionIntercept,
                json!({}),
            ),
        )
        .await
        .is_null()
    );

    let llm_sanitize = invoke_json(
        &mut client,
        llm_invoke(
            "llm-sanitize-request",
            RegistrationSurface::LlmSanitizeRequestGuardrail,
            llm_request(),
            None,
            None,
        ),
    )
    .await;
    assert_eq!(
        llm_sanitize
            .get("content")
            .and_then(|value| value.get("phase")),
        Some(&json!("llm_sanitize_request"))
    );
    assert_json_field(
        invoke_json(
            &mut client,
            llm_invoke(
                "llm-sanitize-response",
                RegistrationSurface::LlmSanitizeResponseGuardrail,
                llm_request(),
                None,
                Some(json!({})),
            ),
        )
        .await,
        "phase",
        "llm_sanitize_response",
    );
    assert_eq!(
        invoke_guardrail(
            &mut client,
            llm_invoke(
                "llm-conditional",
                RegistrationSurface::LlmConditionalExecutionGuardrail,
                llm_request_with_block(),
                None,
                None,
            ),
        )
        .await,
        "blocked-llm"
    );
    let intercepted = invoke_llm_request(
        &mut client,
        llm_invoke(
            "llm-request",
            RegistrationSurface::LlmRequestIntercept,
            llm_request(),
            None,
            None,
        ),
    )
    .await;
    assert_eq!(
        intercepted.content.get("phase"),
        Some(&json!("llm_request"))
    );
    let llm_exec = invoke_json(
        &mut client,
        llm_invoke(
            "llm-exec",
            RegistrationSurface::LlmExecutionIntercept,
            llm_request(),
            None,
            None,
        ),
    )
    .await;
    assert_json_field(llm_exec.clone(), "next", "llm");
    assert_json_field(llm_exec, "phase", "llm_exec");

    let mut stream = client
        .invoke_stream(Request::new(llm_invoke(
            "llm-stream",
            RegistrationSurface::LlmStreamExecutionIntercept,
            llm_request(),
            None,
            None,
        )))
        .await
        .expect("stream invoke")
        .into_inner();
    let first = stream_json(stream.next().await.expect("first stream item").unwrap());
    assert_json_field(first, "phase", "stream_poll");
    let second = stream_json(stream.next().await.expect("second stream item").unwrap());
    assert_json_field(second, "next", "llm_stream");
    assert!(stream.next().await.is_none());

    let stream_auth = client
        .invoke_stream(Request::new(InvokeRequest {
            auth_token: "bad-token".into(),
            ..llm_invoke(
                "llm-stream",
                RegistrationSurface::LlmStreamExecutionIntercept,
                llm_request(),
                None,
                None,
            )
        }))
        .await
        .expect_err("stream auth should fail");
    assert_eq!(stream_auth.code(), tonic::Code::PermissionDenied);

    let calls = host.calls();
    assert!(calls.contains(&"mark:tool-exec:stack-1:parent-1".into()));
    assert!(calls.contains(&"create_scope_stack".into()));
    assert!(calls.contains(&"mark:tool-exec-isolated:isolated-stack:".into()));
    assert!(calls.contains(&"mark:tool-exec-restored:stack-1:parent-1".into()));
    assert!(calls.contains(&"push:worker-scope:stack-1:parent-1".into()));
    assert!(calls.contains(&"pop:scope-handle-1".into()));
    assert!(calls.contains(&"drop:isolated-stack".into()));
    assert!(calls.contains(&"tool_next:next-1".into()));
    assert!(calls.contains(&"llm_next:next-1".into()));
    assert!(calls.contains(&"llm_stream_next:next-1".into()));
    assert!(calls.contains(&"mark:stream-poll:stack-1:parent-1".into()));
    assert!(calls.contains(&"push:scope-agent:explicit-stack:".into()));
    assert!(calls.contains(&"push:scope-unknown:explicit-stack:".into()));

    worker_handle.abort();
    host_handle.abort();
}

#[tokio::test(flavor = "multi_thread")]
async fn worker_service_reports_structured_callback_and_payload_errors() {
    let (handle, mut client) = spawn_worker(
        Arc::new(SurfacePlugin::default()),
        "http://127.0.0.1:9".into(),
    )
    .await;
    register_plugin(&mut client).await;

    assert_worker_error(
        client
            .invoke(Request::new(InvokeRequest {
                surface: 999,
                ..tool_invoke(
                    "tool-request",
                    RegistrationSurface::ToolRequestIntercept,
                    json!({}),
                )
            }))
            .await
            .expect("unknown surface returns structured error")
            .into_inner(),
        "unknown registration surface",
    );
    assert_worker_error(
        client
            .invoke(Request::new(tool_invoke(
                "missing",
                RegistrationSurface::ToolRequestIntercept,
                json!({}),
            )))
            .await
            .expect("unknown registration returns structured error")
            .into_inner(),
        "not registered",
    );
    assert_worker_error(
        client
            .invoke(Request::new(InvokeRequest {
                payload: None,
                ..tool_invoke(
                    "tool-request",
                    RegistrationSurface::ToolRequestIntercept,
                    json!({}),
                )
            }))
            .await
            .expect("invalid payload returns structured error")
            .into_inner(),
        "expected tool payload",
    );
    assert_worker_error(
        client
            .invoke(Request::new(InvokeRequest {
                payload: Some(nemo_relay_worker_proto::v1::invoke_request::Payload::Tool(
                    ToolInvocation {
                        tool_name: "tool".into(),
                        value: Some(JsonEnvelope {
                            schema: "nemo.relay.Json@1".into(),
                            json: b"{".to_vec(),
                        }),
                    },
                )),
                ..tool_invoke(
                    "tool-request",
                    RegistrationSurface::ToolRequestIntercept,
                    json!({}),
                )
            }))
            .await
            .expect("invalid JSON returns structured error")
            .into_inner(),
        "EOF while parsing",
    );
    assert_worker_error(
        client
            .invoke(Request::new(tool_invoke(
                "tool-error",
                RegistrationSurface::ToolRequestIntercept,
                json!({}),
            )))
            .await
            .expect("callback error returns structured error")
            .into_inner(),
        "boom",
    );

    let stream_err = client
        .invoke_stream(Request::new(llm_invoke(
            "llm-stream-error",
            RegistrationSurface::LlmStreamExecutionIntercept,
            llm_request(),
            None,
            None,
        )))
        .await
        .expect("stream error invoke")
        .into_inner()
        .next()
        .await
        .expect("stream item")
        .expect("stream chunk");
    match stream_err.item.expect("stream item") {
        nemo_relay_worker_proto::v1::stream_chunk::Item::Error(error) => {
            assert!(error.message.contains("stream boom"));
        }
        other => panic!("unexpected stream item: {other:?}"),
    }

    let stream_surface_err = client
        .invoke_stream(Request::new(tool_invoke(
            "tool-request",
            RegistrationSurface::ToolRequestIntercept,
            json!({}),
        )))
        .await
        .expect_err("wrong stream surface should fail transport call");
    assert_eq!(stream_surface_err.code(), tonic::Code::InvalidArgument);

    handle.abort();
}

#[test]
fn worker_plugin_defaults_and_context_construction_are_covered() {
    let plugin = MinimalPlugin;
    assert_eq!(plugin.plugin_id(), "minimal");
    assert!(!plugin.allows_multiple_components());
    assert!(plugin.validate(&Json::Null).is_empty());

    assert!(PluginContext::new().runtime().is_none());
    assert!(PluginContext::default().runtime().is_none());
}

#[tokio::test(flavor = "current_thread")]
async fn worker_service_validates_env_and_endpoints() {
    let _env_guard = ENV_LOCK.lock().await;
    let snapshot = EnvSnapshot::capture(REQUIRED_WORKER_ENVS);

    clear_required_envs();
    let err = serve_plugin(MinimalPlugin)
        .await
        .expect_err("missing worker socket env fails");
    assert_error_contains(err, "NEMO_RELAY_WORKER_SOCKET");

    for missing in REQUIRED_WORKER_ENVS {
        set_required_envs();
        remove_env(missing);
        let err = serve_plugin_arc(Arc::new(MinimalPlugin))
            .await
            .expect_err("missing env fails");
        assert_error_contains(err, missing);
    }

    set_required_envs();
    set_env("NEMO_RELAY_WORKER_SOCKET", "ftp://127.0.0.1:1");
    let err = serve_plugin_arc(Arc::new(MinimalPlugin))
        .await
        .expect_err("valid env values reach endpoint validation");
    assert_error_contains(err, "unsupported endpoint");

    snapshot.restore();

    let unsupported =
        serve_plugin_arc_with_config(Arc::new(MinimalPlugin), server_config("ftp://127.0.0.1:1"))
            .await
            .expect_err("unsupported worker endpoint fails");
    assert_error_contains(unsupported, "unsupported endpoint");

    let empty_tcp = serve_plugin_arc_with_config(Arc::new(MinimalPlugin), server_config("tcp://"))
        .await
        .expect_err("empty tcp endpoint fails");
    assert_error_contains(empty_tcp, "unsupported endpoint");

    let path_tcp = serve_plugin_arc_with_config(
        Arc::new(MinimalPlugin),
        server_config("http://127.0.0.1:1/path"),
    )
    .await
    .expect_err("tcp endpoint with path fails");
    assert_error_contains(path_tcp, "unsupported TCP endpoint");

    let invalid_host =
        serve_plugin_arc_with_config(Arc::new(MinimalPlugin), server_config("http://bad host"))
            .await
            .expect_err("invalid tcp host fails");
    assert_error_contains(invalid_host, "invalid TCP endpoint");

    let busy_listener = TcpListener::bind("127.0.0.1:0").expect("bind busy listener");
    let busy_endpoint = format!(
        "tcp://{}",
        busy_listener.local_addr().expect("busy listener addr")
    );
    let busy = serve_plugin_arc_with_config(Arc::new(MinimalPlugin), server_config(&busy_endpoint))
        .await
        .expect_err("busy tcp endpoint fails");
    assert_error_contains(busy, "transport failed");
    drop(busy_listener);

    #[cfg(unix)]
    {
        let path = unique_temp_path("nrw-file");
        std::fs::write(&path, b"not a socket").expect("write regular file");
        let endpoint = format!("unix://{}", path.display());
        let err = serve_plugin_arc_with_config(Arc::new(MinimalPlugin), server_config(&endpoint))
            .await
            .expect_err("regular file must not be removed as socket");
        assert_error_contains(err, "not a socket");
        assert!(path.exists());
        std::fs::remove_file(path).expect("remove regular file");

        let missing_parent = unique_temp_path("nrw-missing-parent").join("worker.sock");
        let missing_parent_endpoint = format!("unix://{}", missing_parent.display());
        let bind_err = serve_plugin_arc_with_config(
            Arc::new(MinimalPlugin),
            server_config(&missing_parent_endpoint),
        )
        .await
        .expect_err("unix endpoint with missing parent fails during bind");
        assert_error_contains(bind_err, "failed to bind worker socket");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn worker_service_announces_ephemeral_tcp_endpoint_file() {
    const ENVS: &[&str] = &[
        "NEMO_RELAY_WORKER_SOCKET",
        "NEMO_RELAY_HOST_SOCKET",
        "NEMO_RELAY_WORKER_ID",
        "NEMO_RELAY_WORKER_TOKEN",
        "NEMO_RELAY_WORKER_ENDPOINT_FILE",
    ];
    let _env_guard = ENV_LOCK.lock().await;
    let snapshot = EnvSnapshot::capture(ENVS);
    let endpoint_file = unique_temp_file("nrw-endpoint");
    let _ = std::fs::remove_file(&endpoint_file);

    set_required_envs();
    set_env("NEMO_RELAY_WORKER_SOCKET", "tcp://127.0.0.1:0");
    set_env(
        "NEMO_RELAY_WORKER_ENDPOINT_FILE",
        endpoint_file.to_str().expect("endpoint path utf-8"),
    );
    let handle = tokio::spawn(serve_plugin_arc(Arc::new(MinimalPlugin)));
    let endpoint = wait_for_endpoint_file(&endpoint_file).await;
    assert!(endpoint.starts_with("http://127.0.0.1:"));

    let mut client = connect_worker(&endpoint).await;
    let health = client
        .health(Request::new(HealthRequest {
            activation_id: ACTIVATION_ID.into(),
            auth_token: AUTH_TOKEN.into(),
        }))
        .await
        .expect("announced endpoint should accept connections")
        .into_inner();
    assert!(health.ok);

    handle.abort();
    let _ = std::fs::remove_file(endpoint_file);
    snapshot.restore();
}

#[cfg(unix)]
#[tokio::test(flavor = "multi_thread")]
async fn worker_service_supports_unix_socket_worker_and_host_endpoints() {
    let host = MockHost::default();
    let host_path = unique_temp_path("nrw-host");
    let worker_path = unique_temp_path("nrw-worker");
    let stale_worker_socket =
        UnixListener::bind(&worker_path).expect("bind stale worker socket before test");
    drop(stale_worker_socket);

    let host_handle = spawn_unix_host(host.clone(), host_path.clone()).await;
    let host_endpoint = format!("unix://{}", host_path.display());
    let worker_endpoint = format!("unix://{}", worker_path.display());
    let worker_handle = tokio::spawn(serve_plugin_arc_with_config(
        Arc::new(SurfacePlugin::default()),
        WorkerServerConfig {
            worker_endpoint: worker_endpoint.clone(),
            host_endpoint,
            activation_id: ACTIVATION_ID.into(),
            auth_token: AUTH_TOKEN.into(),
        },
    ));
    wait_for_unix_socket(&worker_path).await;

    let mut client = connect_worker_uds(&worker_endpoint).await;
    register_plugin(&mut client).await;
    let tool_exec = invoke_json(
        &mut client,
        tool_invoke(
            "tool-exec",
            RegistrationSurface::ToolExecutionIntercept,
            json!({}),
        ),
    )
    .await;
    assert_json_field(tool_exec, "phase", "tool_exec");
    assert!(host.calls().contains(&"tool_next:next-1".into()));

    worker_handle.abort();
    host_handle.abort();
    let _ = std::fs::remove_file(host_path);
    let _ = std::fs::remove_file(worker_path);
}

#[tokio::test(flavor = "multi_thread")]
async fn worker_service_reports_missing_handlers_and_malformed_payloads() {
    let (handle, mut client) = spawn_worker(
        Arc::new(SurfacePlugin::default()),
        "http://127.0.0.1:9".into(),
    )
    .await;
    register_plugin(&mut client).await;

    for (request, expected) in [
        (event_invoke("missing-subscriber"), "subscriber"),
        (
            event_invoke_surface(
                "missing-mark-sanitize",
                RegistrationSurface::MarkSanitizeGuardrail,
            ),
            "event sanitizer",
        ),
        (
            event_invoke_surface(
                "missing-scope-start-sanitize",
                RegistrationSurface::ScopeSanitizeStartGuardrail,
            ),
            "event sanitizer",
        ),
        (
            event_invoke_surface(
                "missing-scope-end-sanitize",
                RegistrationSurface::ScopeSanitizeEndGuardrail,
            ),
            "event sanitizer",
        ),
        (
            tool_invoke(
                "missing-tool-sanitize-request",
                RegistrationSurface::ToolSanitizeRequestGuardrail,
                json!({}),
            ),
            "tool request sanitizer",
        ),
        (
            tool_invoke(
                "missing-tool-sanitize-response",
                RegistrationSurface::ToolSanitizeResponseGuardrail,
                json!({}),
            ),
            "tool response sanitizer",
        ),
        (
            tool_invoke(
                "missing-tool-conditional",
                RegistrationSurface::ToolConditionalExecutionGuardrail,
                json!({}),
            ),
            "tool conditional",
        ),
        (
            tool_invoke(
                "missing-tool-request",
                RegistrationSurface::ToolRequestIntercept,
                json!({}),
            ),
            "tool request",
        ),
        (
            tool_invoke(
                "missing-tool-execution",
                RegistrationSurface::ToolExecutionIntercept,
                json!({}),
            ),
            "tool execution",
        ),
        (
            llm_invoke(
                "missing-llm-sanitize-request",
                RegistrationSurface::LlmSanitizeRequestGuardrail,
                llm_request(),
                None,
                None,
            ),
            "llm request sanitizer",
        ),
        (
            llm_invoke(
                "missing-llm-sanitize-response",
                RegistrationSurface::LlmSanitizeResponseGuardrail,
                llm_request(),
                None,
                Some(json!({})),
            ),
            "llm response sanitizer",
        ),
        (
            llm_invoke(
                "missing-llm-conditional",
                RegistrationSurface::LlmConditionalExecutionGuardrail,
                llm_request(),
                None,
                None,
            ),
            "llm conditional",
        ),
        (
            llm_invoke(
                "missing-llm-request",
                RegistrationSurface::LlmRequestIntercept,
                llm_request(),
                None,
                None,
            ),
            "llm request",
        ),
        (
            llm_invoke(
                "missing-llm-execution",
                RegistrationSurface::LlmExecutionIntercept,
                llm_request(),
                None,
                None,
            ),
            "llm execution",
        ),
    ] {
        assert_worker_error(
            client
                .invoke(Request::new(request))
                .await
                .expect("missing handler returns structured error")
                .into_inner(),
            expected,
        );
    }

    assert_worker_error(
        client
            .invoke(Request::new(InvokeRequest {
                payload: None,
                ..event_invoke("subscriber")
            }))
            .await
            .expect("missing event payload returns structured error")
            .into_inner(),
        "expected event payload",
    );
    assert_worker_error(
        client
            .invoke(Request::new(InvokeRequest {
                payload: None,
                ..event_invoke_surface("mark-sanitize", RegistrationSurface::MarkSanitizeGuardrail)
            }))
            .await
            .expect("missing sanitizer event payload returns structured error")
            .into_inner(),
        "expected event payload",
    );
    assert_worker_error(
        client
            .invoke(Request::new(InvokeRequest {
                payload: None,
                ..llm_invoke(
                    "llm-exec",
                    RegistrationSurface::LlmExecutionIntercept,
                    llm_request(),
                    None,
                    None,
                )
            }))
            .await
            .expect("missing llm payload returns structured error")
            .into_inner(),
        "expected llm payload",
    );
    assert_worker_error(
        client
            .invoke(Request::new(llm_invoke_without_request(
                "llm-sanitize-request",
                RegistrationSurface::LlmSanitizeRequestGuardrail,
            )))
            .await
            .expect("missing llm request returns structured error")
            .into_inner(),
        "llm request is missing",
    );
    assert_worker_error(
        client
            .invoke(Request::new(llm_invoke(
                "llm-sanitize-response",
                RegistrationSurface::LlmSanitizeResponseGuardrail,
                llm_request(),
                None,
                None,
            )))
            .await
            .expect("missing llm response returns structured error")
            .into_inner(),
        "llm response is missing",
    );
    assert_worker_error(
        client
            .invoke(Request::new(InvokeRequest {
                surface: RegistrationSurface::Unspecified as i32,
                ..tool_invoke(
                    "tool-request",
                    RegistrationSurface::ToolRequestIntercept,
                    json!({}),
                )
            }))
            .await
            .expect("unspecified surface returns structured error")
            .into_inner(),
        "unspecified",
    );
    assert_worker_error(
        client
            .invoke(Request::new(llm_invoke(
                "llm-stream",
                RegistrationSurface::LlmStreamExecutionIntercept,
                llm_request(),
                None,
                None,
            )))
            .await
            .expect("stream surface on unary invoke returns structured error")
            .into_inner(),
        "InvokeStream",
    );
    assert_worker_error(
        client
            .invoke(Request::new(llm_invoke_with_bad_annotation("llm-request")))
            .await
            .expect("bad annotation returns structured error")
            .into_inner(),
        "EOF while parsing",
    );

    let unknown_stream_surface = client
        .invoke_stream(Request::new(InvokeRequest {
            surface: 999,
            ..llm_invoke(
                "llm-stream",
                RegistrationSurface::LlmStreamExecutionIntercept,
                llm_request(),
                None,
                None,
            )
        }))
        .await
        .expect_err("unknown stream surface fails transport call");
    assert_eq!(unknown_stream_surface.code(), tonic::Code::InvalidArgument);

    let missing_stream = client
        .invoke_stream(Request::new(llm_invoke(
            "missing-llm-stream",
            RegistrationSurface::LlmStreamExecutionIntercept,
            llm_request(),
            None,
            None,
        )))
        .await
        .expect_err("missing stream handler fails transport call");
    assert_eq!(missing_stream.code(), tonic::Code::NotFound);

    let missing_stream_payload = client
        .invoke_stream(Request::new(InvokeRequest {
            payload: None,
            ..llm_invoke(
                "llm-stream",
                RegistrationSurface::LlmStreamExecutionIntercept,
                llm_request(),
                None,
                None,
            )
        }))
        .await
        .expect_err("missing stream payload fails transport call");
    assert_status_message(missing_stream_payload, "expected llm payload");

    let missing_stream_request = client
        .invoke_stream(Request::new(llm_invoke_without_request(
            "llm-stream",
            RegistrationSurface::LlmStreamExecutionIntercept,
        )))
        .await
        .expect_err("missing stream request fails transport call");
    assert_status_message(missing_stream_request, "llm request is missing");

    let open_error = client
        .invoke_stream(Request::new(llm_invoke(
            "llm-stream-open-error",
            RegistrationSurface::LlmStreamExecutionIntercept,
            llm_request(),
            None,
            None,
        )))
        .await
        .expect_err("stream open callback error fails transport call");
    assert_status_message(open_error, "stream open boom");

    handle.abort();
}

#[tokio::test(flavor = "multi_thread")]
async fn worker_service_propagates_host_runtime_errors() {
    let host = MockHost::default();
    let (host_handle, host_endpoint) = spawn_host(host.clone()).await;
    let (worker_handle, mut client) = spawn_worker(
        Arc::new(SurfacePlugin::default()),
        tcp_endpoint(&host_endpoint),
    )
    .await;
    register_plugin(&mut client).await;

    for (failures, expected) in [
        (
            MockHostFailures {
                emit_mark: HostFailure::WorkerError,
                ..Default::default()
            },
            "emit mark failed",
        ),
        (
            MockHostFailures {
                emit_mark: HostFailure::EmptyAck,
                ..Default::default()
            },
            "host call failed",
        ),
        (
            MockHostFailures {
                create_scope_stack: true,
                ..Default::default()
            },
            "create scope stack failed",
        ),
        (
            MockHostFailures {
                push_scope: true,
                ..Default::default()
            },
            "push scope failed",
        ),
        (
            MockHostFailures {
                pop_scope: HostFailure::WorkerError,
                ..Default::default()
            },
            "pop scope failed",
        ),
        (
            MockHostFailures {
                drop_scope_stack: HostFailure::WorkerError,
                ..Default::default()
            },
            "drop scope stack failed",
        ),
        (
            MockHostFailures {
                tool_next: true,
                ..Default::default()
            },
            "tool next failed",
        ),
    ] {
        host.set_failures(failures);
        assert_worker_error(
            client
                .invoke(Request::new(tool_invoke(
                    "tool-exec",
                    RegistrationSurface::ToolExecutionIntercept,
                    json!({}),
                )))
                .await
                .expect("host failure returns structured error")
                .into_inner(),
            expected,
        );
    }

    host.set_failures(MockHostFailures {
        llm_next: true,
        ..Default::default()
    });
    assert_worker_error(
        client
            .invoke(Request::new(llm_invoke(
                "llm-exec",
                RegistrationSurface::LlmExecutionIntercept,
                llm_request(),
                None,
                None,
            )))
            .await
            .expect("llm next failure returns structured error")
            .into_inner(),
        "llm next failed",
    );

    for (mode, expected) in [
        (MockStreamMode::WorkerError, "stream worker failed"),
        (MockStreamMode::EmptyChunk, "empty stream chunk"),
        (MockStreamMode::TransportError, "stream status failed"),
    ] {
        host.set_failures(MockHostFailures {
            llm_stream_mode: mode,
            ..Default::default()
        });
        let mut stream = client
            .invoke_stream(Request::new(llm_invoke(
                "llm-stream",
                RegistrationSurface::LlmStreamExecutionIntercept,
                llm_request(),
                None,
                None,
            )))
            .await
            .expect("stream invoke")
            .into_inner();
        let first = stream_json(stream.next().await.expect("stream poll item").unwrap());
        assert_json_field(first, "phase", "stream_poll");
        let second = stream.next().await.expect("host stream item").unwrap();
        assert_stream_error(second, expected);
    }

    worker_handle.abort();
    host_handle.abort();
}

struct MinimalPlugin;

impl WorkerPlugin for MinimalPlugin {
    fn plugin_id(&self) -> &str {
        "minimal"
    }

    fn register(&self, _ctx: &mut PluginContext, _config: &Json) -> Result<()> {
        Ok(())
    }
}

struct DuplicateEventSanitizerPlugin;

impl WorkerPlugin for DuplicateEventSanitizerPlugin {
    fn plugin_id(&self) -> &str {
        PLUGIN_ID
    }

    fn register(&self, ctx: &mut PluginContext, _config: &Json) -> Result<()> {
        ctx.register_mark_sanitize_guardrail("duplicate", 0, |_, fields| fields);
        ctx.register_mark_sanitize_guardrail("duplicate", 1, |_, fields| fields);
        Ok(())
    }
}

#[derive(Default)]
struct CancellationPlugin {
    unary_started: Arc<tokio::sync::Notify>,
    unary_cancelled: Arc<AtomicBool>,
    stream_started: Arc<tokio::sync::Notify>,
    stream_cancelled: Arc<AtomicBool>,
    stream_setup_started: Arc<tokio::sync::Notify>,
    stream_setup_cancelled: Arc<AtomicBool>,
}

struct CancelledOnDrop(Arc<AtomicBool>);

impl Drop for CancelledOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

impl WorkerPlugin for CancellationPlugin {
    fn plugin_id(&self) -> &str {
        "cancellation"
    }

    fn register(&self, ctx: &mut PluginContext, _config: &Json) -> Result<()> {
        let unary_started = self.unary_started.clone();
        let unary_cancelled = self.unary_cancelled.clone();
        ctx.register_tool_execution_intercept("cancel-unary", 0, move |_, _, _| {
            let unary_started = unary_started.clone();
            let unary_cancelled = unary_cancelled.clone();
            async move {
                let _cancelled = CancelledOnDrop(unary_cancelled);
                unary_started.notify_one();
                std::future::pending::<Result<ToolExecutionInterceptOutcome>>().await
            }
        });

        let stream_started = self.stream_started.clone();
        let stream_cancelled = self.stream_cancelled.clone();
        ctx.register_llm_stream_execution_intercept("cancel-stream", 0, move |_, _, _| {
            let stream_started = stream_started.clone();
            let stream_cancelled = stream_cancelled.clone();
            async move {
                Ok(Box::pin(FillThenPendingStream {
                    started: stream_started,
                    cancelled: stream_cancelled,
                    yielded: 0,
                }) as JsonStream)
            }
        });

        let stream_setup_started = self.stream_setup_started.clone();
        let stream_setup_cancelled = self.stream_setup_cancelled.clone();
        ctx.register_llm_stream_execution_intercept("cancel-stream-setup", 0, move |_, _, _| {
            let stream_setup_started = stream_setup_started.clone();
            let stream_setup_cancelled = stream_setup_cancelled.clone();
            async move {
                let _cancelled = CancelledOnDrop(stream_setup_cancelled);
                stream_setup_started.notify_one();
                std::future::pending::<Result<JsonStream>>().await
            }
        });
        Ok(())
    }
}

struct FillThenPendingStream {
    started: Arc<tokio::sync::Notify>,
    cancelled: Arc<AtomicBool>,
    yielded: usize,
}

impl Stream for FillThenPendingStream {
    type Item = Result<Json>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        if self.yielded < 17 {
            self.yielded += 1;
            if self.yielded == 17 {
                self.started.notify_one();
            }
            return Poll::Ready(Some(Ok(json!({ "chunk": self.yielded }))));
        }
        std::task::Poll::Pending
    }
}

impl Drop for FillThenPendingStream {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }
}

#[derive(Default)]
struct SurfacePlugin {
    events: Arc<Mutex<Vec<String>>>,
}

impl WorkerPlugin for SurfacePlugin {
    fn plugin_id(&self) -> &str {
        PLUGIN_ID
    }

    fn validate(&self, config: &Json) -> Vec<nemo_relay_worker::ConfigDiagnostic> {
        if config
            .get("diagnostic")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
        {
            vec![nemo_relay_worker::ConfigDiagnostic {
                level: nemo_relay_worker::DiagnosticLevel::Warning,
                code: "diagnostic.requested".into(),
                component: Some(PLUGIN_ID.into()),
                field: Some("diagnostic".into()),
                message: "diagnostic requested".into(),
            }]
        } else {
            Vec::new()
        }
    }

    fn register(&self, ctx: &mut PluginContext, _config: &Json) -> Result<()> {
        let runtime = ctx.runtime().expect("worker service provides runtime");
        let events = self.events.clone();
        ctx.register_subscriber("subscriber", move |event| {
            events
                .lock()
                .expect("events lock")
                .push(event.name().into());
        });
        ctx.register_mark_sanitize_guardrail("event-sanitize", 1, |event, mut fields| {
            fields.data = Some(json!({"name": event.name(), "phase": "mark"}));
            fields
        });
        ctx.register_scope_sanitize_start_guardrail("event-sanitize", 1, |_, mut fields| {
            fields.metadata = Some(json!({"phase": "scope_start"}));
            fields
        });
        ctx.register_scope_sanitize_end_guardrail("event-sanitize", 1, |_, mut fields| {
            fields.metadata = Some(json!({"phase": "scope_end"}));
            fields
        });
        ctx.register_tool_sanitize_request_guardrail("tool-sanitize", 1, |_, value| {
            set_json_field(value, "phase", "tool_sanitize_request")
        });
        ctx.register_tool_sanitize_response_guardrail("tool-sanitize", 1, |_, value| {
            set_json_field(value, "phase", "tool_sanitize_response")
        });
        ctx.register_tool_conditional_execution_guardrail("tool-conditional", 1, |_, value| {
            Ok(value
                .get("block")
                .and_then(serde_json::Value::as_bool)
                .and_then(|blocked| blocked.then(|| "blocked-tool".into())))
        });
        ctx.register_tool_request_intercept("tool-request", 1, false, |_, value| {
            Ok(set_json_field(value, "phase", "tool_request"))
        });
        ctx.register_tool_request_intercept("tool-error", 1, false, |_, _| {
            Err(WorkerSdkError::Callback("boom".into()))
        });

        let tool_runtime = runtime.clone();
        ctx.register_tool_execution_intercept("tool-exec", 1, move |_, value, next: ToolNext| {
            let runtime = tool_runtime.clone();
            async move {
                runtime.emit_mark("tool-exec", None, None).await?;
                let stack_id = runtime.create_scope_stack().await?;
                let isolated_runtime = runtime.clone();
                runtime
                    .with_scope_stack(&stack_id, || async move {
                        isolated_runtime
                            .emit_mark("tool-exec-isolated", None, None)
                            .await
                    })
                    .await?;
                runtime.emit_mark("tool-exec-restored", None, None).await?;
                let handle = runtime
                    .push_scope(None, "worker-scope", ScopeType::Function, None, None, None)
                    .await?;
                runtime.pop_scope(&handle, None, None).await?;
                runtime.drop_scope_stack(&stack_id).await?;
                let next_value = next.call(value).await?;
                Ok(ToolExecutionInterceptOutcome::new(set_json_field(
                    next_value,
                    "phase",
                    "tool_exec",
                ))
                .with_pending_mark(
                    PendingMarkSpec::builder()
                        .name("worker.tool.execution")
                        .build(),
                ))
            }
        });
        let scope_runtime = runtime.clone();
        ctx.register_tool_execution_intercept("tool-scope-types", 1, move |_, _, _| {
            let runtime = scope_runtime.clone();
            async move {
                for scope_type in [
                    ScopeType::Agent,
                    ScopeType::Function,
                    ScopeType::Tool,
                    ScopeType::Llm,
                    ScopeType::Retriever,
                    ScopeType::Embedder,
                    ScopeType::Reranker,
                    ScopeType::Guardrail,
                    ScopeType::Evaluator,
                    ScopeType::Custom,
                    ScopeType::Unknown,
                ] {
                    let handle = runtime
                        .push_scope(
                            Some("explicit-stack"),
                            &format!("scope-{}", scope_type.as_str()),
                            scope_type,
                            None,
                            None,
                            None,
                        )
                        .await?;
                    runtime.pop_scope(&handle, None, None).await?;
                }
                Ok(Json::Null.into())
            }
        });

        ctx.register_llm_sanitize_request_guardrail("llm-sanitize-request", 1, |request| {
            set_llm_phase(request, "llm_sanitize_request")
        });
        ctx.register_llm_sanitize_response_guardrail("llm-sanitize-response", 1, |value| {
            set_json_field(value, "phase", "llm_sanitize_response")
        });
        ctx.register_llm_conditional_execution_guardrail("llm-conditional", 1, |request| {
            Ok(request
                .content
                .get("block")
                .and_then(serde_json::Value::as_bool)
                .and_then(|blocked| blocked.then(|| "blocked-llm".into())))
        });
        ctx.register_llm_request_intercept("llm-request", 1, false, |_, request, annotated| {
            Ok(nemo_relay_worker::LlmRequestInterceptOutcome::new(
                set_llm_phase(request, "llm_request"),
                annotated,
            ))
        });

        ctx.register_llm_execution_intercept(
            "llm-exec",
            1,
            move |_, request, next: LlmNext| async move {
                let next_value = next.call(request).await?;
                Ok(set_json_field(next_value, "phase", "llm_exec"))
            },
        );

        let stream_runtime = runtime.clone();
        ctx.register_llm_stream_execution_intercept(
            "llm-stream",
            1,
            move |_, request, next: LlmStreamNext| {
                let runtime = stream_runtime.clone();
                async move {
                    let next_stream = next.call(request).await?;
                    Ok(Box::pin(RuntimeMarkThenStream::new(runtime, next_stream)) as JsonStream)
                }
            },
        );
        ctx.register_llm_stream_execution_intercept("llm-stream-error", 1, |_, _, _| async {
            let stream: JsonStream = Box::pin(tokio_stream::iter(vec![Err(
                WorkerSdkError::Callback("stream boom".into()),
            )]));
            Ok(stream)
        });
        ctx.register_llm_stream_execution_intercept("llm-stream-open-error", 1, |_, _, _| async {
            Err(WorkerSdkError::Callback("stream open boom".into()))
        });
        Ok(())
    }
}

struct RuntimeMarkThenStream {
    runtime: Option<PluginRuntime>,
    pending: Option<Pin<Box<dyn Future<Output = Result<Json>> + Send>>>,
    inner: JsonStream,
}

impl RuntimeMarkThenStream {
    fn new(runtime: PluginRuntime, inner: JsonStream) -> Self {
        Self {
            runtime: Some(runtime),
            pending: None,
            inner,
        }
    }
}

impl Unpin for RuntimeMarkThenStream {}

impl Stream for RuntimeMarkThenStream {
    type Item = Result<Json>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.runtime.is_some() || self.pending.is_some() {
            if self.pending.is_none() {
                let runtime = self.runtime.take().expect("runtime present");
                self.pending = Some(Box::pin(async move {
                    runtime.emit_mark("stream-poll", None, None).await?;
                    Ok(json!({"phase": "stream_poll"}))
                }));
            }
            let future = self.pending.as_mut().expect("pending future");
            return match future.as_mut().poll(cx) {
                Poll::Ready(result) => {
                    self.pending = None;
                    Poll::Ready(Some(result))
                }
                Poll::Pending => Poll::Pending,
            };
        }
        self.inner.as_mut().poll_next(cx)
    }
}

#[derive(Clone, Copy, Default)]
enum HostFailure {
    #[default]
    None,
    WorkerError,
    EmptyAck,
}

#[derive(Clone, Copy, Default)]
enum MockStreamMode {
    #[default]
    Value,
    WorkerError,
    TransportError,
    EmptyChunk,
}

#[derive(Clone, Copy, Default)]
struct MockHostFailures {
    emit_mark: HostFailure,
    create_scope_stack: bool,
    push_scope: bool,
    pop_scope: HostFailure,
    drop_scope_stack: HostFailure,
    tool_next: bool,
    llm_next: bool,
    llm_stream_mode: MockStreamMode,
}

#[derive(Clone, Default)]
struct MockHost {
    calls: Arc<Mutex<Vec<String>>>,
    failures: Arc<Mutex<MockHostFailures>>,
}

impl MockHost {
    fn calls(&self) -> Vec<String> {
        self.calls.lock().expect("calls lock").clone()
    }

    fn record(&self, call: impl Into<String>) {
        self.calls.lock().expect("calls lock").push(call.into());
    }

    fn failures(&self) -> MockHostFailures {
        *self.failures.lock().expect("failures lock")
    }

    fn set_failures(&self, failures: MockHostFailures) {
        *self.failures.lock().expect("failures lock") = failures;
    }
}

#[tonic::async_trait]
impl RelayHostRuntime for MockHost {
    async fn emit_mark(
        &self,
        request: Request<EmitMarkRequest>,
    ) -> std::result::Result<Response<HostAck>, Status> {
        let request = request.into_inner();
        authorize_host(&request.activation_id, &request.auth_token)?;
        let scope = request.scope.expect("scope context");
        self.record(format!(
            "mark:{}:{}:{}",
            request.name, scope.scope_stack_id, scope.parent_scope_id
        ));
        match self.failures().emit_mark {
            HostFailure::None => {}
            HostFailure::WorkerError => {
                return Ok(Response::new(host_ack_error("emit mark failed")));
            }
            HostFailure::EmptyAck => {
                return Ok(Response::new(HostAck {
                    ok: false,
                    error: None,
                }));
            }
        }
        Ok(Response::new(host_ack()))
    }

    async fn push_scope(
        &self,
        request: Request<PushScopeRequest>,
    ) -> std::result::Result<Response<PushScopeResponse>, Status> {
        let request = request.into_inner();
        authorize_host(&request.activation_id, &request.auth_token)?;
        let scope = request.scope.expect("scope context");
        self.record(format!(
            "push:{}:{}:{}",
            request.name, scope.scope_stack_id, scope.parent_scope_id
        ));
        if self.failures().push_scope {
            return Ok(Response::new(PushScopeResponse {
                scope_handle_id: String::new(),
                error: Some(worker_error("push scope failed")),
            }));
        }
        Ok(Response::new(PushScopeResponse {
            scope_handle_id: "scope-handle-1".into(),
            error: None,
        }))
    }

    async fn pop_scope(
        &self,
        request: Request<PopScopeRequest>,
    ) -> std::result::Result<Response<HostAck>, Status> {
        let request = request.into_inner();
        authorize_host(&request.activation_id, &request.auth_token)?;
        self.record(format!("pop:{}", request.scope_handle_id));
        match self.failures().pop_scope {
            HostFailure::None => {}
            HostFailure::WorkerError => {
                return Ok(Response::new(host_ack_error("pop scope failed")));
            }
            HostFailure::EmptyAck => {
                return Ok(Response::new(HostAck {
                    ok: false,
                    error: None,
                }));
            }
        }
        Ok(Response::new(host_ack()))
    }

    async fn create_scope_stack(
        &self,
        request: Request<CreateScopeStackRequest>,
    ) -> std::result::Result<Response<CreateScopeStackResponse>, Status> {
        let request = request.into_inner();
        authorize_host(&request.activation_id, &request.auth_token)?;
        self.record("create_scope_stack");
        if self.failures().create_scope_stack {
            return Ok(Response::new(CreateScopeStackResponse {
                scope_stack_id: String::new(),
                error: Some(worker_error("create scope stack failed")),
            }));
        }
        Ok(Response::new(CreateScopeStackResponse {
            scope_stack_id: "isolated-stack".into(),
            error: None,
        }))
    }

    async fn drop_scope_stack(
        &self,
        request: Request<DropScopeStackRequest>,
    ) -> std::result::Result<Response<HostAck>, Status> {
        let request = request.into_inner();
        authorize_host(&request.activation_id, &request.auth_token)?;
        self.record(format!("drop:{}", request.scope_stack_id));
        match self.failures().drop_scope_stack {
            HostFailure::None => {}
            HostFailure::WorkerError => {
                return Ok(Response::new(host_ack_error("drop scope stack failed")));
            }
            HostFailure::EmptyAck => {
                return Ok(Response::new(HostAck {
                    ok: false,
                    error: None,
                }));
            }
        }
        Ok(Response::new(host_ack()))
    }

    async fn tool_next(
        &self,
        request: Request<ToolNextRequest>,
    ) -> std::result::Result<Response<JsonResult>, Status> {
        let request = request.into_inner();
        authorize_host(&request.activation_id, &request.auth_token)?;
        self.record(format!("tool_next:{}", request.continuation_id));
        if self.failures().tool_next {
            return Ok(Response::new(JsonResult {
                value: None,
                error: Some(worker_error("tool next failed")),
            }));
        }
        Ok(Response::new(JsonResult {
            value: Some(json_env(json!({"next": "tool"}))),
            error: None,
        }))
    }

    async fn llm_next(
        &self,
        request: Request<LlmNextRequest>,
    ) -> std::result::Result<Response<JsonResult>, Status> {
        let request = request.into_inner();
        authorize_host(&request.activation_id, &request.auth_token)?;
        self.record(format!("llm_next:{}", request.continuation_id));
        if self.failures().llm_next {
            return Ok(Response::new(JsonResult {
                value: None,
                error: Some(worker_error("llm next failed")),
            }));
        }
        Ok(Response::new(JsonResult {
            value: Some(json_env(json!({"next": "llm"}))),
            error: None,
        }))
    }

    type LlmStreamNextStream =
        Pin<Box<dyn Stream<Item = std::result::Result<StreamChunk, Status>> + Send>>;

    async fn llm_stream_next(
        &self,
        request: Request<LlmStreamNextRequest>,
    ) -> std::result::Result<Response<Self::LlmStreamNextStream>, Status> {
        let request = request.into_inner();
        authorize_host(&request.activation_id, &request.auth_token)?;
        self.record(format!("llm_stream_next:{}", request.continuation_id));
        let chunks: Vec<std::result::Result<StreamChunk, Status>> =
            match self.failures().llm_stream_mode {
                MockStreamMode::Value => vec![Ok(StreamChunk {
                    item: Some(nemo_relay_worker_proto::v1::stream_chunk::Item::Value(
                        json_env(json!({"next": "llm_stream"})),
                    )),
                })],
                MockStreamMode::WorkerError => vec![Ok(StreamChunk {
                    item: Some(nemo_relay_worker_proto::v1::stream_chunk::Item::Error(
                        worker_error("stream worker failed"),
                    )),
                })],
                MockStreamMode::TransportError => {
                    vec![Err(Status::unavailable("stream status failed"))]
                }
                MockStreamMode::EmptyChunk => vec![Ok(StreamChunk { item: None })],
            };
        let stream = tokio_stream::iter(chunks);
        Ok(Response::new(Box::pin(stream)))
    }
}

async fn spawn_worker(
    plugin: Arc<dyn WorkerPlugin>,
    host_endpoint: String,
) -> (JoinHandle<Result<()>>, PluginWorkerClient<Channel>) {
    let (worker_endpoint, connect_endpoint) = unused_worker_endpoints();
    let handle = tokio::spawn(serve_plugin_arc_with_config(
        plugin,
        WorkerServerConfig {
            worker_endpoint,
            host_endpoint,
            activation_id: ACTIVATION_ID.into(),
            auth_token: AUTH_TOKEN.into(),
        },
    ));
    let client = connect_worker(&connect_endpoint).await;
    (handle, client)
}

async fn spawn_host(
    host: MockHost,
) -> (
    JoinHandle<std::result::Result<(), tonic::transport::Error>>,
    String,
) {
    let endpoint = unused_http_endpoint();
    let addr = endpoint
        .strip_prefix("http://")
        .expect("http endpoint")
        .parse::<SocketAddr>()
        .expect("socket address");
    let handle = tokio::spawn(async move {
        Server::builder()
            .add_service(RelayHostRuntimeServer::new(host))
            .serve(addr)
            .await
    });
    wait_for_port(&endpoint).await;
    (handle, endpoint)
}

#[cfg(unix)]
async fn spawn_unix_host(
    host: MockHost,
    path: PathBuf,
) -> JoinHandle<std::result::Result<(), tonic::transport::Error>> {
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path).expect("bind host unix socket");
    let handle = tokio::spawn(async move {
        Server::builder()
            .add_service(RelayHostRuntimeServer::new(host))
            .serve_with_incoming(UnixListenerStream::new(listener))
            .await
    });
    wait_for_unix_socket(&path).await;
    handle
}

async fn connect_worker(endpoint: &str) -> PluginWorkerClient<Channel> {
    for _ in 0..50 {
        match PluginWorkerClient::connect(endpoint.to_owned()).await {
            Ok(client) => return client,
            Err(_) => std::thread::sleep(Duration::from_millis(20)),
        }
    }
    panic!("worker did not start at {endpoint}");
}

#[cfg(unix)]
async fn connect_worker_uds(endpoint: &str) -> PluginWorkerClient<Channel> {
    let path = Arc::new(
        endpoint
            .strip_prefix("unix://")
            .map(PathBuf::from)
            .expect("unix endpoint"),
    );
    for _ in 0..50 {
        let path = path.clone();
        let channel = Endpoint::try_from("http://[::]:50051")
            .expect("uds placeholder endpoint")
            .connect_with_connector(service_fn(move |_| {
                let path = path.clone();
                async move {
                    let stream = UnixStream::connect(&*path).await?;
                    Ok::<_, std::io::Error>(TokioIo::new(stream))
                }
            }))
            .await;
        if let Ok(channel) = channel {
            return PluginWorkerClient::new(channel);
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    panic!("worker did not start at {endpoint}");
}

async fn wait_for_port(endpoint: &str) {
    for _ in 0..50 {
        if Channel::from_shared(endpoint.to_owned())
            .expect("valid endpoint")
            .connect()
            .await
            .is_ok()
        {
            return;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    panic!("server did not start at {endpoint}");
}

async fn wait_for_endpoint_file(path: &Path) -> String {
    for _ in 0..50 {
        match std::fs::read_to_string(path) {
            Ok(endpoint) if !endpoint.trim().is_empty() => return endpoint,
            Ok(_) | Err(_) => std::thread::sleep(Duration::from_millis(20)),
        }
    }
    panic!("endpoint file was not written at {}", path.display());
}

#[cfg(unix)]
async fn wait_for_unix_socket(path: &Path) {
    for _ in 0..50 {
        if UnixStream::connect(path).await.is_ok() {
            return;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    panic!("server did not start at {}", path.display());
}

async fn register_plugin(
    client: &mut PluginWorkerClient<Channel>,
) -> Vec<nemo_relay_worker_proto::v1::Registration> {
    client
        .register(Request::new(RegisterRequest {
            activation_id: ACTIVATION_ID.into(),
            plugin_id: PLUGIN_ID.into(),
            auth_token: AUTH_TOKEN.into(),
            config: Some(json_env(json!({}))),
        }))
        .await
        .expect("register succeeds")
        .into_inner()
        .registrations
}

fn unused_http_endpoint() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");
    drop(listener);
    format!("http://{addr}")
}

fn unused_worker_endpoints() -> (String, String) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");
    drop(listener);
    (format!("tcp://{addr}"), format!("http://{addr}"))
}

fn tcp_endpoint(http_endpoint: &str) -> String {
    format!(
        "tcp://{}",
        http_endpoint
            .strip_prefix("http://")
            .expect("http endpoint")
    )
}

fn json_env(value: Json) -> JsonEnvelope {
    json_envelope("nemo.relay.Json@1", &value).expect("encode JSON envelope")
}

fn invalid_json_env(schema: &str) -> JsonEnvelope {
    JsonEnvelope {
        schema: schema.into(),
        json: b"{".to_vec(),
    }
}

fn event_invoke(registration_name: &str) -> InvokeRequest {
    event_invoke_surface(registration_name, RegistrationSurface::Subscriber)
}

fn event_invoke_surface(registration_name: &str, surface: RegistrationSurface) -> InvokeRequest {
    let event = Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .name("subscriber-event")
            .data(json!({"raw": true}))
            .build(),
        None,
        None,
    ));
    InvokeRequest {
        activation_id: ACTIVATION_ID.into(),
        invocation_id: "invoke-1".into(),
        registration_name: registration_name.into(),
        surface: surface as i32,
        continuation_id: "next-1".into(),
        scope: Some(scope_context()),
        auth_token: AUTH_TOKEN.into(),
        payload: Some(nemo_relay_worker_proto::v1::invoke_request::Payload::Event(
            json_envelope("nemo.relay.Event@1", &event).expect("encode event"),
        )),
    }
}

fn tool_invoke(
    registration_name: &str,
    surface: RegistrationSurface,
    value: Json,
) -> InvokeRequest {
    InvokeRequest {
        activation_id: ACTIVATION_ID.into(),
        invocation_id: "invoke-1".into(),
        registration_name: registration_name.into(),
        surface: surface as i32,
        continuation_id: "next-1".into(),
        scope: Some(scope_context()),
        auth_token: AUTH_TOKEN.into(),
        payload: Some(nemo_relay_worker_proto::v1::invoke_request::Payload::Tool(
            ToolInvocation {
                tool_name: "tool".into(),
                value: Some(json_env(value)),
            },
        )),
    }
}

fn llm_invoke(
    registration_name: &str,
    surface: RegistrationSurface,
    request: LlmRequest,
    annotated_request: Option<Json>,
    response: Option<Json>,
) -> InvokeRequest {
    InvokeRequest {
        activation_id: ACTIVATION_ID.into(),
        invocation_id: "invoke-1".into(),
        registration_name: registration_name.into(),
        surface: surface as i32,
        continuation_id: "next-1".into(),
        scope: Some(scope_context()),
        auth_token: AUTH_TOKEN.into(),
        payload: Some(nemo_relay_worker_proto::v1::invoke_request::Payload::Llm(
            LlmInvocation {
                model_name: "model".into(),
                request: Some(
                    json_envelope("nemo.relay.LlmRequest@1", &request).expect("encode request"),
                ),
                annotated_request: annotated_request.map(json_env),
                response: response.map(json_env),
            },
        )),
    }
}

fn llm_invoke_without_request(
    registration_name: &str,
    surface: RegistrationSurface,
) -> InvokeRequest {
    InvokeRequest {
        activation_id: ACTIVATION_ID.into(),
        invocation_id: "invoke-1".into(),
        registration_name: registration_name.into(),
        surface: surface as i32,
        continuation_id: "next-1".into(),
        scope: Some(scope_context()),
        auth_token: AUTH_TOKEN.into(),
        payload: Some(nemo_relay_worker_proto::v1::invoke_request::Payload::Llm(
            LlmInvocation {
                model_name: "model".into(),
                request: None,
                annotated_request: None,
                response: None,
            },
        )),
    }
}

fn llm_invoke_with_bad_annotation(registration_name: &str) -> InvokeRequest {
    let mut request = llm_invoke(
        registration_name,
        RegistrationSurface::LlmRequestIntercept,
        llm_request(),
        None,
        None,
    );
    if let Some(nemo_relay_worker_proto::v1::invoke_request::Payload::Llm(payload)) =
        request.payload.as_mut()
    {
        payload.annotated_request = Some(JsonEnvelope {
            schema: "nemo.relay.AnnotatedLlmRequest@1".into(),
            json: b"{".to_vec(),
        });
    }
    request
}

fn scope_context() -> ScopeContext {
    ScopeContext {
        scope_stack_id: "stack-1".into(),
        parent_scope_id: "parent-1".into(),
    }
}

fn llm_request() -> LlmRequest {
    LlmRequest {
        headers: Default::default(),
        content: json!({"prompt": "hello"}),
    }
}

fn llm_request_with_block() -> LlmRequest {
    LlmRequest {
        headers: Default::default(),
        content: json!({"block": true}),
    }
}

fn set_json_field(mut value: Json, key: &str, field_value: &str) -> Json {
    value
        .as_object_mut()
        .expect("JSON object")
        .insert(key.into(), json!(field_value));
    value
}

fn set_llm_phase(mut request: LlmRequest, phase: &str) -> LlmRequest {
    request
        .content
        .as_object_mut()
        .expect("request content object")
        .insert("phase".into(), json!(phase));
    request
}

async fn invoke_json(client: &mut PluginWorkerClient<Channel>, request: InvokeRequest) -> Json {
    let response = client
        .invoke(Request::new(request))
        .await
        .expect("invoke succeeds")
        .into_inner();
    match response.result.expect("invoke result") {
        nemo_relay_worker_proto::v1::invoke_response::Result::Json(result) => {
            decode_json_envelope(&result.value.expect("json value")).expect("decode JSON result")
        }
        nemo_relay_worker_proto::v1::invoke_response::Result::ToolExecution(result) => {
            decode_json_envelope::<ToolExecutionInterceptOutcome>(
                &result.outcome.expect("tool execution outcome"),
            )
            .expect("decode tool execution outcome")
            .result
        }
        other => panic!("unexpected invoke result: {other:?}"),
    }
}

async fn invoke_tool_execution(
    client: &mut PluginWorkerClient<Channel>,
    request: InvokeRequest,
) -> ToolExecutionInterceptOutcome {
    let response = client
        .invoke(Request::new(request))
        .await
        .expect("invoke succeeds")
        .into_inner();
    match response.result.expect("invoke result") {
        nemo_relay_worker_proto::v1::invoke_response::Result::ToolExecution(result) => {
            let outcome = result.outcome.expect("tool execution outcome");
            assert_eq!(outcome.schema, "nemo.relay.ToolExecutionInterceptOutcome@1");
            decode_json_envelope(&outcome).expect("decode tool execution outcome")
        }
        other => panic!("unexpected invoke result: {other:?}"),
    }
}

async fn invoke_guardrail(
    client: &mut PluginWorkerClient<Channel>,
    request: InvokeRequest,
) -> String {
    let response = client
        .invoke(Request::new(request))
        .await
        .expect("invoke succeeds")
        .into_inner();
    match response.result.expect("invoke result") {
        nemo_relay_worker_proto::v1::invoke_response::Result::Guardrail(result) => {
            result.block_reason
        }
        other => panic!("unexpected invoke result: {other:?}"),
    }
}

async fn invoke_llm_request(
    client: &mut PluginWorkerClient<Channel>,
    request: InvokeRequest,
) -> LlmRequest {
    let response = client
        .invoke(Request::new(request))
        .await
        .expect("invoke succeeds")
        .into_inner();
    match response.result.expect("invoke result") {
        nemo_relay_worker_proto::v1::invoke_response::Result::LlmRequest(result) => {
            decode_json_envelope::<nemo_relay_worker::LlmRequestInterceptOutcome>(
                &result.outcome.expect("llm request outcome"),
            )
            .expect("decode LLM request outcome")
            .request
        }
        other => panic!("unexpected invoke result: {other:?}"),
    }
}

fn stream_json(chunk: StreamChunk) -> Json {
    match chunk.item.expect("stream chunk item") {
        nemo_relay_worker_proto::v1::stream_chunk::Item::Value(value) => {
            decode_json_envelope(&value).expect("decode stream value")
        }
        other => panic!("unexpected stream chunk: {other:?}"),
    }
}

fn assert_empty_response(response: InvokeResponse) {
    assert!(matches!(
        response.result.expect("invoke result"),
        nemo_relay_worker_proto::v1::invoke_response::Result::Empty(_)
    ));
}

fn assert_json_field(value: Json, key: &str, expected: &str) {
    assert_eq!(value.get(key), Some(&json!(expected)));
}

fn assert_worker_error(response: InvokeResponse, expected: &str) {
    match response.result.expect("invoke result") {
        nemo_relay_worker_proto::v1::invoke_response::Result::Error(error) => {
            assert!(
                error.message.contains(expected),
                "expected '{expected}' in '{}'",
                error.message
            );
        }
        other => panic!("unexpected invoke result: {other:?}"),
    }
}

fn assert_stream_error(chunk: StreamChunk, expected: &str) {
    match chunk.item.expect("stream item") {
        nemo_relay_worker_proto::v1::stream_chunk::Item::Error(error) => {
            assert!(
                error.message.contains(expected),
                "expected '{expected}' in '{}'",
                error.message
            );
        }
        other => panic!("unexpected stream item: {other:?}"),
    }
}

fn assert_status_message(status: Status, expected: &str) {
    assert!(
        status.message().contains(expected),
        "expected '{expected}' in '{}'",
        status.message()
    );
}

fn assert_error_contains(error: WorkerSdkError, expected: &str) {
    let message = error.to_string();
    assert!(
        message.contains(expected),
        "expected '{expected}' in '{message}'"
    );
}

fn server_config(worker_endpoint: &str) -> WorkerServerConfig {
    WorkerServerConfig {
        worker_endpoint: worker_endpoint.into(),
        host_endpoint: "http://127.0.0.1:9".into(),
        activation_id: ACTIVATION_ID.into(),
        auth_token: AUTH_TOKEN.into(),
    }
}

struct EnvSnapshot {
    values: Vec<(&'static str, Option<String>)>,
}

impl EnvSnapshot {
    fn capture(names: &'static [&'static str]) -> Self {
        Self {
            values: names
                .iter()
                .map(|name| (*name, std::env::var(name).ok()))
                .collect(),
        }
    }

    fn restore(&self) {
        for (name, value) in &self.values {
            match value {
                Some(value) => set_env(name, value),
                None => remove_env(name),
            }
        }
    }
}

impl Drop for EnvSnapshot {
    fn drop(&mut self) {
        self.restore();
    }
}

fn set_required_envs() {
    set_env("NEMO_RELAY_WORKER_SOCKET", "tcp://127.0.0.1:1");
    set_env("NEMO_RELAY_HOST_SOCKET", "http://127.0.0.1:9");
    set_env("NEMO_RELAY_WORKER_ID", ACTIVATION_ID);
    set_env("NEMO_RELAY_WORKER_TOKEN", AUTH_TOKEN);
}

fn clear_required_envs() {
    for name in REQUIRED_WORKER_ENVS {
        remove_env(name);
    }
}

fn set_env(name: &str, value: &str) {
    // Process environment mutation is guarded by ENV_LOCK in these tests.
    unsafe {
        std::env::set_var(name, value);
    }
}

fn remove_env(name: &str) {
    // Process environment mutation is guarded by ENV_LOCK in these tests.
    unsafe {
        std::env::remove_var(name);
    }
}

fn unique_temp_file(prefix: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time after epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}-{}-{nanos}", std::process::id()))
}

#[cfg(unix)]
fn unique_temp_path(prefix: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time after epoch")
        .as_nanos();
    PathBuf::from(format!("/tmp/{prefix}-{}-{nanos}.sock", std::process::id()))
}

fn host_ack() -> HostAck {
    HostAck {
        ok: true,
        error: None,
    }
}

fn host_ack_error(message: &str) -> HostAck {
    HostAck {
        ok: false,
        error: Some(worker_error(message)),
    }
}

fn worker_error(message: &str) -> WorkerError {
    WorkerError {
        code: "mock.error".into(),
        message: message.into(),
        retryable: false,
    }
}

fn authorize_host(activation_id: &str, auth_token: &str) -> std::result::Result<(), Status> {
    if activation_id != ACTIVATION_ID || auth_token != AUTH_TOKEN {
        return Err(Status::permission_denied("invalid host auth"));
    }
    Ok(())
}
