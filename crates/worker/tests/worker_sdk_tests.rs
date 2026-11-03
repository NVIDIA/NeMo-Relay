// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end coverage for the Rust gRPC worker SDK service.

use std::collections::{BTreeMap, BTreeSet};
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
    ANNOTATED_LLM_REQUEST_SCHEMA, DataSchema, EmitMarkOptions, EventCategory, Json, JsonStream,
    LlmNext, LlmRequest, LlmStreamNext, LogSeverity, MetricKind, MetricMeasurement,
    MetricValueType, PluginContext, PluginLogLevel, PluginRuntime, Result, RuntimeRegistrationKind,
    ScopeType, ToolExecutionInterceptOutcome, ToolNext, WorkerPlugin, WorkerSdkError,
    WorkerServerConfig, serve_plugin, serve_plugin_arc, serve_plugin_arc_with_config,
};
use nemo_relay_worker_proto::nvidia_worker_v1 as pb;
#[path = "support/typed_invocations.rs"]
mod fixtures;
use fixtures::{
    ExecutionCodecContext, InvocationFixture as InvokeRequest, LlmInvocation, Payload,
    PluginWorkerClient, SanitizeContext, ToolInvocation,
};
use nemo_relay_worker_proto::conversions::{self};
use nemo_relay_worker_proto::nvidia_worker_v1::relay_host_runtime_service_server::{
    RelayHostRuntimeService as RelayHostRuntime,
    RelayHostRuntimeServiceServer as RelayHostRuntimeServer,
};
use pb::WORKER_PROTOCOL;
use pb::{
    CancelInvocationRequest, CreateScopeStackRequest, CreateScopeStackResponse,
    DeregisterConditionalMiddlewareGuardrailRequest,
    DeregisterConditionalMiddlewareGuardrailResponse, DropScopeStackRequest, EmitMarkRequest,
    GetRuntimeDiagnosticsRequest, GetRuntimeDiagnosticsResponse, HandshakeRequest, HealthRequest,
    InvokeResponse, ListRuntimeRegistrationsRequest, ListRuntimeRegistrationsResponse,
    LlmNextRequest, LlmStreamNextRequest, LogRequest, LogSeverity as LogLevel, PopScopeRequest,
    PushScopeRequest, PushScopeResponse, RegisterConditionalMiddlewareGuardrailRequest,
    RegisterConditionalMiddlewareGuardrailResponse, RegisterRequest, RegistrationSurface,
    RuntimeDiagnostic as ProtoRuntimeDiagnostic, ScopeContext, ShutdownRequest,
    ToolExecutionInterceptOutcome as ProtoToolExecutionInterceptOutcome,
    ToolExecutionResult as ProtoToolExecutionResult, ToolNextRequest, ValidateRequest, WorkerError,
};
type JsonEnvelope = pb::JsonValue;
type StreamChunk = pb::InvokeStreamResponse;
type ToolExecutionResultResponse = pb::ToolNextResponse;
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

/// Every runtime registration kind paired with the wire surface it maps to.
const REGISTRATION_KIND_SURFACES: &[(RuntimeRegistrationKind, RegistrationSurface)] = &[
    (
        RuntimeRegistrationKind::Subscriber,
        RegistrationSurface::Subscriber,
    ),
    (
        RuntimeRegistrationKind::EventMetadataInjector,
        RegistrationSurface::EventMetadataInjector,
    ),
    (
        RuntimeRegistrationKind::MarkSanitizeGuardrail,
        RegistrationSurface::MarkSanitizeGuardrail,
    ),
    (
        RuntimeRegistrationKind::ScopeSanitizeStartGuardrail,
        RegistrationSurface::ScopeSanitizeStartGuardrail,
    ),
    (
        RuntimeRegistrationKind::ScopeSanitizeEndGuardrail,
        RegistrationSurface::ScopeSanitizeEndGuardrail,
    ),
    (
        RuntimeRegistrationKind::ToolSanitizeRequestGuardrail,
        RegistrationSurface::ToolSanitizeRequestGuardrail,
    ),
    (
        RuntimeRegistrationKind::ToolSanitizeResponseGuardrail,
        RegistrationSurface::ToolSanitizeResponseGuardrail,
    ),
    (
        RuntimeRegistrationKind::ToolConditionalExecutionGuardrail,
        RegistrationSurface::ToolConditionalExecutionGuardrail,
    ),
    (
        RuntimeRegistrationKind::ToolRequestIntercept,
        RegistrationSurface::ToolRequestIntercept,
    ),
    (
        RuntimeRegistrationKind::ToolExecutionIntercept,
        RegistrationSurface::ToolExecutionIntercept,
    ),
    (
        RuntimeRegistrationKind::LlmSanitizeRequestGuardrail,
        RegistrationSurface::LlmSanitizeRequestGuardrail,
    ),
    (
        RuntimeRegistrationKind::LlmSanitizeResponseGuardrail,
        RegistrationSurface::LlmSanitizeResponseGuardrail,
    ),
    (
        RuntimeRegistrationKind::LlmConditionalExecutionGuardrail,
        RegistrationSurface::LlmConditionalExecutionGuardrail,
    ),
    (
        RuntimeRegistrationKind::LlmRequestIntercept,
        RegistrationSurface::LlmRequestIntercept,
    ),
    (
        RuntimeRegistrationKind::LlmExecutionIntercept,
        RegistrationSurface::LlmExecutionIntercept,
    ),
    (
        RuntimeRegistrationKind::LlmStreamExecutionIntercept,
        RegistrationSurface::LlmStreamExecutionIntercept,
    ),
];

const ACTIVATION_ID: &str = "activation-1";
const AUTH_TOKEN: &str = "secret-token";
const PLUGIN_ID: &str = "acme.worker";
const WORKER_TEST_TIMEOUT: Duration = Duration::from_secs(10);
const REQUIRED_WORKER_ENVS: &[&str] = &[
    "NEMO_RELAY_WORKER_SOCKET",
    "NEMO_RELAY_HOST_SOCKET",
    "NEMO_RELAY_WORKER_ID",
    "NEMO_RELAY_WORKER_TOKEN",
];
static ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

#[test]
fn runtime_registration_dtos_reexport_shared_types() {
    let identity = nemo_relay_worker::RuntimeRegistrationIdentity {
        kind: nemo_relay_worker::RuntimeRegistrationKind::Subscriber,
        local_name: "subscriber".into(),
        effective_name: "subscriber".into(),
        owner: nemo_relay_worker::RuntimeRegistrationOwner {
            kind: nemo_relay_types::api::registry::RuntimeRegistrationOwnerKind::Core,
            plugin_kind: None,
            component_ordinal: None,
        },
    };
    let _: nemo_relay_types::api::registry::RuntimeRegistrationIdentity = identity;
}

#[tokio::test(flavor = "multi_thread")]
async fn worker_service_enforces_auth_and_reports_registrations() {
    let (handle, mut client) = spawn_worker(
        Arc::new(SurfacePlugin::default()),
        "http://127.0.0.1:9".into(),
    )
    .await;

    let bad_handshake = client
        .handshake(Request::new(HandshakeRequest {
            plugin_id: PLUGIN_ID.into(),
            relay_version: "0.8.0".into(),
            host_endpoint: "http://127.0.0.1:9".into(),
            credentials: Some(pb::Credentials {
                activation_id: ACTIVATION_ID.into(),
                auth_token: "bad-token".into(),
            }),
            protocol: Some(pb::ProtocolIdentity {
                name: WORKER_PROTOCOL.into(),
                version: "1.0.0".into(),
            }),
        }))
        .await
        .expect_err("invalid token should fail");
    assert_eq!(bad_handshake.code(), tonic::Code::Unauthenticated);

    let bad_activation = client
        .handshake(Request::new(HandshakeRequest {
            plugin_id: PLUGIN_ID.into(),
            relay_version: "0.8.0".into(),
            host_endpoint: "http://127.0.0.1:9".into(),
            credentials: Some(pb::Credentials {
                activation_id: "wrong-activation".into(),
                auth_token: AUTH_TOKEN.into(),
            }),
            protocol: Some(pb::ProtocolIdentity {
                name: WORKER_PROTOCOL.into(),
                version: "1.0.0".into(),
            }),
        }))
        .await
        .expect_err("invalid activation should fail");
    assert_eq!(bad_activation.code(), tonic::Code::Unauthenticated);

    let handshake = client
        .handshake(Request::new(HandshakeRequest {
            plugin_id: PLUGIN_ID.into(),
            relay_version: "0.8.0".into(),
            host_endpoint: "http://127.0.0.1:9".into(),
            credentials: Some(pb::Credentials {
                activation_id: ACTIVATION_ID.into(),
                auth_token: AUTH_TOKEN.into(),
            }),
            protocol: Some(pb::ProtocolIdentity {
                name: WORKER_PROTOCOL.into(),
                version: "1.0.0".into(),
            }),
        }))
        .await
        .expect("handshake succeeds")
        .into_inner();
    assert_eq!(handshake.identity.as_ref().unwrap().plugin_id, PLUGIN_ID);
    assert_eq!(
        handshake
            .identity
            .as_ref()
            .unwrap()
            .protocol
            .as_ref()
            .unwrap()
            .name,
        WORKER_PROTOCOL
    );
    assert!(
        handshake
            .supported_surfaces
            .contains(&(RegistrationSurface::LlmStreamExecutionIntercept as i32))
    );
    assert!(
        handshake
            .supported_surfaces
            .contains(&(RegistrationSurface::EventMetadataInjector as i32))
    );

    let bad_health = client
        .health(Request::new(HealthRequest {
            credentials: Some(pb::Credentials {
                activation_id: ACTIVATION_ID.into(),
                auth_token: "bad-token".into(),
            }),
        }))
        .await
        .expect_err("health auth should fail");
    assert_eq!(bad_health.code(), tonic::Code::Unauthenticated);

    let health = client
        .health(Request::new(HealthRequest {
            credentials: Some(pb::Credentials {
                activation_id: ACTIVATION_ID.into(),
                auth_token: AUTH_TOKEN.into(),
            }),
        }))
        .await
        .expect("health succeeds")
        .into_inner();
    assert!(health.status == pb::WorkerHealthStatus::Ready as i32);
    assert_eq!(health.message, "ready");
    assert_eq!(health.identity.as_ref().unwrap().plugin_id, PLUGIN_ID);
    assert_eq!(
        health
            .identity
            .as_ref()
            .unwrap()
            .protocol
            .as_ref()
            .unwrap()
            .name,
        WORKER_PROTOCOL
    );
    assert_eq!(
        health.identity.as_ref().unwrap().sdk_name,
        "nemo-relay-worker"
    );
    assert_eq!(health.identity.as_ref().unwrap().runtime_name, "rust");

    let validate_err = client
        .validate(Request::new(ValidateRequest {
            plugin_id: PLUGIN_ID.into(),
            config: Some(json_env(json!({"diagnostic": true}))),
            credentials: Some(pb::Credentials {
                activation_id: ACTIVATION_ID.into(),
                auth_token: "bad-token".into(),
            }),
        }))
        .await
        .expect_err("validate auth should fail");
    assert_eq!(validate_err.code(), tonic::Code::Unauthenticated);

    let invalid_validate_config = client
        .validate(Request::new(ValidateRequest {
            plugin_id: PLUGIN_ID.into(),
            config: Some(invalid_json_env("nemo.relay.Json@1")),
            credentials: Some(pb::Credentials {
                activation_id: ACTIVATION_ID.into(),
                auth_token: AUTH_TOKEN.into(),
            }),
        }))
        .await
        .expect_err("invalid validate config should fail");
    assert_eq!(invalid_validate_config.code(), tonic::Code::InvalidArgument);

    let diagnostics = client
        .validate(Request::new(ValidateRequest {
            plugin_id: PLUGIN_ID.into(),
            config: Some(json_env(json!({"diagnostic": true}))),
            credentials: Some(pb::Credentials {
                activation_id: ACTIVATION_ID.into(),
                auth_token: AUTH_TOKEN.into(),
            }),
        }))
        .await
        .expect("validate succeeds")
        .into_inner()
        .diagnostics()
        .expect("diagnostics envelope");

    assert_eq!(diagnostics.len(), 1);

    let register_err = client
        .register(Request::new(RegisterRequest {
            plugin_id: PLUGIN_ID.into(),
            config: Some(json_env(json!({}))),
            credentials: Some(pb::Credentials {
                activation_id: ACTIVATION_ID.into(),
                auth_token: "bad-token".into(),
            }),
        }))
        .await
        .expect_err("register auth should fail");
    assert_eq!(register_err.code(), tonic::Code::Unauthenticated);

    let invalid_register_config = client
        .register(Request::new(RegisterRequest {
            plugin_id: PLUGIN_ID.into(),
            config: Some(invalid_json_env("nemo.relay.Json@1")),
            credentials: Some(pb::Credentials {
                activation_id: ACTIVATION_ID.into(),
                auth_token: AUTH_TOKEN.into(),
            }),
        }))
        .await
        .expect_err("invalid register config should fail");
    assert_eq!(invalid_register_config.code(), tonic::Code::InvalidArgument);

    let registrations = register_plugin(&mut client).await;
    assert_eq!(registrations.len(), 22);
    for local_name in [
        "llm-sanitize-request",
        "llm-sanitize-response",
        "llm-sanitize-omit-request",
        "llm-sanitize-omit-response",
    ] {
        assert_eq!(
            registrations
                .iter()
                .filter(|registration| registration.local_name == local_name)
                .count(),
            1,
            "expected exactly one {local_name} registration"
        );
    }
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
            credentials: Some(pb::Credentials {
                activation_id: ACTIVATION_ID.into(),
                auth_token: "bad-token".into(),
            }),
            ..tool_invoke(
                "tool-request",
                RegistrationSurface::ToolRequestIntercept,
                json!({}),
            )
        }))
        .await
        .expect_err("invoke auth should fail");
    assert_eq!(invoke_err.code(), tonic::Code::Unauthenticated);

    let cancel = client
        .cancel_invocation(Request::new(CancelInvocationRequest {
            invocation_id: "invoke-1".into(),
            reason: "test".into(),
            credentials: Some(pb::Credentials {
                activation_id: ACTIVATION_ID.into(),
                auth_token: AUTH_TOKEN.into(),
            }),
        }))
        .await
        .expect("cancel returns ack")
        .into_inner();
    assert!(!cancel.accepted);
    assert!(cancel.message.contains("not active"));

    let shutdown = client
        .shutdown(Request::new(ShutdownRequest {
            reason: "test".into(),
            credentials: Some(pb::Credentials {
                activation_id: ACTIVATION_ID.into(),
                auth_token: AUTH_TOKEN.into(),
            }),
        }))
        .await
        .expect("shutdown returns ack")
        .into_inner();
    assert!(shutdown == pb::ShutdownResponse::default());
    assert!(shutdown == pb::ShutdownResponse::default());

    handle.abort();
}

#[tokio::test(flavor = "multi_thread")]
async fn worker_service_shutdown_closes_tcp_health_watches() {
    let endpoint = unused_http_endpoint();
    let worker = tokio::spawn(serve_plugin_arc_with_config(
        Arc::new(SurfacePlugin::default()),
        server_config(&endpoint),
    ));
    wait_for_port(&endpoint).await;
    let channel = Channel::from_shared(endpoint)
        .unwrap()
        .connect()
        .await
        .unwrap();
    assert_shutdown_closes_health_watches(worker, channel).await;
}

#[cfg(unix)]
#[tokio::test(flavor = "multi_thread")]
async fn worker_service_shutdown_closes_unix_health_watches() {
    let path = unique_temp_path("nrw-health-watch");
    let endpoint = format!("unix://{}", path.display());
    let worker = tokio::spawn(serve_plugin_arc_with_config(
        Arc::new(SurfacePlugin::default()),
        server_config(&endpoint),
    ));
    wait_for_unix_socket(&path).await;
    let connect_path = path.clone();
    let channel = Endpoint::try_from("http://[::]:50051")
        .unwrap()
        .connect_with_connector(service_fn(move |_| {
            let path = connect_path.clone();
            async move { UnixStream::connect(path).await.map(TokioIo::new) }
        }))
        .await
        .unwrap();
    assert_shutdown_closes_health_watches(worker, channel).await;
    let _ = std::fs::remove_file(path);
}

/// The server must exit while the client still holds both standard health streams.
async fn assert_shutdown_closes_health_watches(
    mut worker: JoinHandle<Result<()>>,
    channel: Channel,
) {
    use tonic_health::pb::health_check_response::ServingStatus;
    use tonic_health::pb::{HealthCheckRequest, health_client::HealthClient};

    let mut health = HealthClient::new(channel.clone());
    let mut watches = Vec::new();
    for service in ["", "nvidia.nemo.relay.worker.v1.PluginWorkerService"] {
        let mut watch = health
            .watch(HealthCheckRequest {
                service: service.into(),
            })
            .await
            .unwrap()
            .into_inner();
        assert_eq!(
            watch.message().await.unwrap().unwrap().status,
            ServingStatus::Serving as i32
        );
        watches.push(watch);
    }
    let mut client = pb::plugin_worker_service_client::PluginWorkerServiceClient::new(channel);
    let rejected = client
        .shutdown(ShutdownRequest {
            credentials: Some(pb::Credentials {
                activation_id: ACTIVATION_ID.into(),
                auth_token: "incorrect".into(),
            }),
            reason: "unauthorized probe".into(),
        })
        .await
        .unwrap_err();
    assert_eq!(rejected.code(), tonic::Code::Unauthenticated);
    assert_eq!(
        health
            .check(HealthCheckRequest { service: "".into() })
            .await
            .unwrap()
            .into_inner()
            .status,
        ServingStatus::Serving as i32
    );
    client
        .shutdown(ShutdownRequest {
            credentials: Some(pb::Credentials {
                activation_id: ACTIVATION_ID.into(),
                auth_token: AUTH_TOKEN.into(),
            }),
            reason: "test shutdown with health monitors".into(),
        })
        .await
        .unwrap();
    let exited = tokio::time::timeout(Duration::from_secs(3), &mut worker).await;
    if exited.is_err() {
        worker.abort();
    }
    exited
        .expect("health watches must not prevent worker shutdown")
        .expect("worker task must not panic")
        .expect("worker server must shut down successfully");
    for mut watch in watches {
        assert_eq!(
            watch.message().await.unwrap().unwrap().status,
            ServingStatus::NotServing as i32
        );
        assert!(watch.message().await.unwrap().is_none());
    }
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
            plugin_id: PLUGIN_ID.into(),
            config: Some(json_env(json!({}))),
            credentials: Some(pb::Credentials {
                activation_id: ACTIVATION_ID.into(),
                auth_token: AUTH_TOKEN.into(),
            }),
        }))
        .await
        .expect("duplicate registration should return protocol data")
        .into_inner();
    assert!(response.error().is_some());
    let error = response
        .error()
        .expect("duplicate registration should return an error");
    assert_eq!(error.code, "worker.error");
    assert_eq!(
        error.message,
        "invalid input: duplicate registration 'duplicate' for surface REGISTRATION_SURFACE_MARK_SANITIZE_GUARDRAIL"
    );

    handle.abort();
}

#[tokio::test(flavor = "multi_thread")]
async fn worker_service_validates_initial_conditional_middleware_guardrails() {
    for (mode, expected) in [
        (InvalidInitialGate::EmptyKinds, "has no registration kinds"),
        (InvalidInitialGate::EmptyTarget, "names must not be empty"),
        (
            InvalidInitialGate::Duplicate,
            "duplicate conditional middleware guardrail",
        ),
    ] {
        let (handle, mut client) = spawn_worker(
            Arc::new(InvalidInitialGatePlugin(mode)),
            "http://127.0.0.1:9".into(),
        )
        .await;
        let response = client
            .register(Request::new(RegisterRequest {
                plugin_id: PLUGIN_ID.into(),
                config: Some(json_env(json!({}))),
                credentials: Some(pb::Credentials {
                    activation_id: ACTIVATION_ID.into(),
                    auth_token: AUTH_TOKEN.into(),
                }),
            }))
            .await
            .expect("invalid initial gate should return protocol data")
            .into_inner();
        assert!(response.error().is_some());
        assert!(response.error().is_some());
        assert!(response.error().unwrap().message.contains(expected));
        let invoke = client
            .invoke(Request::new(InvokeRequest {
                invocation_id: "invalid-initial-gate".into(),
                registration_name: "initial-gate".into(),
                surface: RegistrationSurface::ConditionalMiddlewareGuardrail as i32,
                continuation_id: String::new(),
                scope: None,
                payload: Some(Payload::ConditionalMiddleware(
                    pb::ConditionalMiddlewareInvocation {
                        kinds: vec![RegistrationSurface::Subscriber as i32],
                        registration_name: "target-subscriber".into(),
                    },
                )),
                credentials: Some(pb::Credentials {
                    activation_id: ACTIVATION_ID.into(),
                    auth_token: AUTH_TOKEN.into(),
                }),
            }))
            .await
            .expect("failed registration callback lookup should return protocol data")
            .into_inner();
        let message = match invoke.result.expect("invoke result") {
            pb::invoke_response::Result::Error(error) => error.message,
            other => panic!("unexpected invoke result after failed registration: {other:?}"),
        };
        assert!(message.contains("is not registered"), "{message}");
        handle.abort();
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn worker_service_cancels_unary_and_stream_invocations_by_id() {
    let timeout = WORKER_TEST_TIMEOUT;
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
            invocation_id: "cancel-unary".into(),
            reason: "test timeout".into(),
            credentials: Some(pb::Credentials {
                activation_id: ACTIVATION_ID.into(),
                auth_token: AUTH_TOKEN.into(),
            }),
        })),
    )
    .await
    .expect("unary cancellation should complete")
    .expect("active unary cancellation should return an ack")
    .into_inner();
    assert!(plugin.unary_cancelled.load(Ordering::SeqCst));
    let unary_response = tokio::time::timeout(timeout, unary_task)
        .await
        .expect("unary invocation should finish after cancellation")
        .expect("unary client task should join");
    let unary_error = match unary_response.result {
        Some(pb::invoke_response::Result::Error(error)) => error,
        other => panic!("expected cancellation error, got {other:?}"),
    };
    assert!(unary_ack.accepted);
    assert_eq!(unary_error.code, "worker.cancelled");
    assert!(plugin.unary_cancelled.load(Ordering::SeqCst));

    let repeated = tokio::time::timeout(
        timeout,
        client.cancel_invocation(Request::new(CancelInvocationRequest {
            invocation_id: "cancel-unary".into(),
            reason: "repeat".into(),
            credentials: Some(pb::Credentials {
                activation_id: ACTIVATION_ID.into(),
                auth_token: AUTH_TOKEN.into(),
            }),
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
            invocation_id: "cancel-stream".into(),
            reason: "stream abandoned".into(),
            credentials: Some(pb::Credentials {
                activation_id: ACTIVATION_ID.into(),
                auth_token: AUTH_TOKEN.into(),
            }),
        })),
    )
    .await
    .expect("stream cancellation should complete")
    .expect("active stream cancellation should return an ack")
    .into_inner();
    assert!(plugin.stream_cancelled.load(Ordering::SeqCst));
    let stream_error = tokio::time::timeout(timeout, async {
        loop {
            let chunk = stream
                .next()
                .await
                .expect("cancelled stream should yield a terminal error")
                .expect("cancelled stream chunk should be protocol data");
            if let Some(pb::invoke_stream_response::Item::Error(error)) = chunk.item {
                break error;
            }
        }
    })
    .await
    .expect("cancelled stream should terminate");
    assert!(stream_ack.accepted);
    assert_eq!(stream_error.code, "worker.cancelled");
    tokio::time::timeout(timeout, plugin.stream_cancelled_notified.notified())
        .await
        .expect("cancelled stream should be dropped");
    assert!(plugin.stream_cancelled.load(Ordering::SeqCst));

    handle.abort();
}

#[tokio::test(flavor = "multi_thread")]
async fn worker_service_cancels_stream_during_async_setup() {
    let timeout = WORKER_TEST_TIMEOUT;
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
            invocation_id: "cancel-stream-setup".into(),
            reason: "cancel setup".into(),
            credentials: Some(pb::Credentials {
                activation_id: ACTIVATION_ID.into(),
                auth_token: AUTH_TOKEN.into(),
            }),
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
        Some(pb::invoke_stream_response::Item::Error(error)) => error,
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
    let registrations = register_plugin(&mut client).await;
    for name in [
        "llm-sanitize-request",
        "llm-sanitize-response",
        "llm-sanitize-omit-request",
        "llm-sanitize-omit-response",
    ] {
        assert!(
            registrations
                .iter()
                .any(|registration| registration.local_name == name),
            "{name} must be registered"
        );
    }

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

    let metadata = invoke_json(
        &mut client,
        event_invoke_surface("event-metadata", RegistrationSurface::EventMetadataInjector),
    )
    .await;
    assert_eq!(metadata, json!({"worker.event_name": "subscriber-event"}));

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
    let mut tool_execution_request = tool_invoke(
        "tool-exec",
        RegistrationSurface::ToolExecutionIntercept,
        json!({}),
    );
    let Some(Payload::Tool(invocation)) = tool_execution_request.payload.as_mut() else {
        panic!("tool execution request must contain a tool invocation");
    };
    invocation.tool_call_id = Some("call-worker-1".into());
    let tool_outcome = invoke_tool_execution(&mut client, tool_execution_request).await;
    assert_eq!(tool_outcome.pending_marks.len(), 1);
    assert_eq!(
        tool_outcome.pending_marks[0],
        PendingMarkSpec::builder()
            .name("worker.tool.execution")
            .data(json!({"checkpoint": true}))
            .build()
    );
    assert_eq!(tool_outcome.annotation, Some(json!({"source": "host"})));
    let tool_exec = tool_outcome.result;
    assert_json_field(tool_exec.clone(), "tool_call_id", "call-worker-1");
    assert_json_field(tool_exec.clone(), "next", "tool");
    assert_json_field(tool_exec, "phase", "tool_exec");
    let logs = host.logs();
    assert_eq!(logs.len(), 11);
    for (request, level) in logs.iter().take(5).zip([
        LogLevel::Trace,
        LogLevel::Debug,
        LogLevel::Info,
        LogLevel::Warn,
        LogLevel::Error,
    ]) {
        assert_eq!(request.level, level as i32);
        assert_eq!(request.target, "plugin.logging");
    }
    assert_eq!(logs[0].message, "trace message");
    assert_eq!(logs[4].message, "error message");
    assert_eq!(
        decode_json_envelope::<Json, _>(logs[0].fields.as_ref().expect("trace fields")).unwrap(),
        json!({"nested": {"ok": true}, "ordinal": 0})
    );
    assert!(logs[1].fields.is_none());
    assert_eq!(logs[5].level, LogLevel::Info as i32);
    assert_eq!(logs[5].target, "");
    assert_eq!(logs[5].message, "direct message");
    for (request, level) in logs[6..].iter().zip([
        LogLevel::Trace,
        LogLevel::Debug,
        LogLevel::Info,
        LogLevel::Warn,
        LogLevel::Error,
    ]) {
        assert_eq!(request.level, level as i32);
        assert_eq!(request.message, "method message");
        assert_eq!(request.target, "plugin.methods");
        assert_eq!(
            decode_json_envelope::<Json, _>(request.fields.as_ref().unwrap()).unwrap(),
            json!({"method": true})
        );
    }
    let conditional_middleware = client
        .invoke(Request::new(InvokeRequest {
            invocation_id: "conditional-middleware".into(),
            registration_name: "initial-gate".into(),
            surface: RegistrationSurface::ConditionalMiddlewareGuardrail as i32,
            continuation_id: String::new(),
            scope: None,
            payload: Some(Payload::ConditionalMiddleware(
                pb::ConditionalMiddlewareInvocation {
                    kinds: vec![RegistrationSurface::Subscriber as i32],
                    registration_name: "missing-target".into(),
                },
            )),
            credentials: Some(pb::Credentials {
                activation_id: ACTIVATION_ID.into(),
                auth_token: AUTH_TOKEN.into(),
            }),
        }))
        .await
        .expect("conditional middleware invoke")
        .into_inner();
    assert_eq!(guardrail_reason(conditional_middleware), "initial gate");
    let allowed_conditional_middleware = client
        .invoke(Request::new(InvokeRequest {
            invocation_id: "conditional-middleware-allowed".into(),
            registration_name: "initial-gate".into(),
            surface: RegistrationSurface::ConditionalMiddlewareGuardrail as i32,
            continuation_id: String::new(),
            scope: None,
            payload: Some(Payload::ConditionalMiddleware(
                pb::ConditionalMiddlewareInvocation {
                    kinds: vec![RegistrationSurface::Subscriber as i32],
                    registration_name: "allowed-target".into(),
                },
            )),
            credentials: Some(pb::Credentials {
                activation_id: ACTIVATION_ID.into(),
                auth_token: AUTH_TOKEN.into(),
            }),
        }))
        .await
        .expect("allowed conditional middleware invoke")
        .into_inner();
    assert_eq!(guardrail_reason(allowed_conditional_middleware), "");
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
    let omitted_request = client
        .invoke(Request::new(llm_invoke(
            "llm-sanitize-omit-request",
            RegistrationSurface::LlmSanitizeRequestGuardrail,
            llm_request(),
            None,
            None,
        )))
        .await
        .expect("request omission invoke")
        .into_inner();
    assert_empty_response(omitted_request);
    let omitted_response = client
        .invoke(Request::new(llm_invoke(
            "llm-sanitize-omit-response",
            RegistrationSurface::LlmSanitizeResponseGuardrail,
            llm_request(),
            None,
            Some(json!({})),
        )))
        .await
        .expect("response omission invoke")
        .into_inner();
    assert_empty_response(omitted_response);
    let codec_request_result = invoke_json(&mut client, llm_request_codec_invoke()).await;
    assert_eq!(
        codec_request_result,
        json!({
            "headers": {},
            "content": {"phase": "llm_sanitize_request"},
        })
    );

    let codec_response_result = invoke_json(&mut client, llm_response_codec_invoke()).await;
    assert_eq!(
        codec_response_result,
        json!({
            "value": "secret",
            "phase": "llm_sanitize_response",
        })
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
            credentials: Some(pb::Credentials {
                activation_id: ACTIVATION_ID.into(),
                auth_token: "bad-token".into(),
            }),
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
    assert_eq!(stream_auth.code(), tonic::Code::Unauthenticated);

    let calls = host.calls();
    let runtime_registration_requests = host.runtime_registration_requests();
    assert_eq!(
        runtime_registration_requests.list.len(),
        REGISTRATION_KIND_SURFACES.len()
    );
    for (index, (kind, surface)) in REGISTRATION_KIND_SURFACES.iter().enumerate() {
        let recorded = &runtime_registration_requests.list[index];
        assert_eq!(
            recorded.credentials.as_ref().unwrap().activation_id,
            ACTIVATION_ID
        );
        assert_eq!(
            recorded.credentials.as_ref().unwrap().auth_token,
            AUTH_TOKEN
        );
        assert_eq!(
            recorded.kinds,
            vec![*surface as i32],
            "{kind:?} should map to {surface:?}"
        );
    }
    assert_eq!(runtime_registration_requests.register.len(), 1);
    assert_eq!(
        runtime_registration_requests.register[0]
            .credentials
            .as_ref()
            .unwrap()
            .activation_id,
        ACTIVATION_ID
    );
    assert_eq!(
        runtime_registration_requests.register[0]
            .credentials
            .as_ref()
            .unwrap()
            .auth_token,
        AUTH_TOKEN
    );
    assert_eq!(
        runtime_registration_requests.register[0]
            .registration
            .as_ref()
            .unwrap()
            .name,
        "timer-gate"
    );
    assert_eq!(
        runtime_registration_requests.register[0]
            .registration
            .as_ref()
            .unwrap()
            .kinds,
        vec![RegistrationSurface::Subscriber as i32]
    );
    assert_eq!(
        runtime_registration_requests.register[0]
            .registration
            .as_ref()
            .unwrap()
            .registration_name,
        "target-subscriber"
    );
    assert!(
        runtime_registration_requests.register[0]
            .registration
            .as_ref()
            .unwrap()
            .callback
    );
    assert_eq!(runtime_registration_requests.deregister.len(), 1);
    assert_eq!(
        runtime_registration_requests.deregister[0]
            .credentials
            .as_ref()
            .unwrap()
            .activation_id,
        ACTIVATION_ID
    );
    assert_eq!(
        runtime_registration_requests.deregister[0]
            .credentials
            .as_ref()
            .unwrap()
            .auth_token,
        AUTH_TOKEN
    );
    assert_eq!(
        runtime_registration_requests.deregister[0].handle,
        "gate-handle"
    );
    assert!(calls.contains(&"runtime_diagnostics".into()));
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
    assert!(calls.contains(&"codec_request_decode:request-capability".into()));
    assert!(calls.contains(&"codec_request_encode:request-capability".into()));
    assert!(calls.contains(&"codec_response_decode:response-capability".into()));
    assert!(calls.contains(&"mark:stream-poll:stack-1:parent-1".into()));
    assert!(calls.contains(&"push:scope-agent:explicit-stack:".into()));
    assert!(calls.contains(&"push:scope-unknown:explicit-stack:".into()));
    let scope_requests = host.scope_requests();
    let historical_push = scope_requests
        .push
        .iter()
        .find(|request| request.name == "worker-scope")
        .expect("historical scope push");
    assert_eq!(
        historical_push
            .time
            .as_ref()
            .map(|v| v.seconds * 1_000_000 + i64::from(v.nanos) / 1000),
        Some(-2)
    );
    let historical_pop = scope_requests
        .pop
        .iter()
        .find(|request| request.scope_handle_id == "scope-handle-1")
        .expect("historical scope pop");
    assert_eq!(
        historical_pop
            .time
            .as_ref()
            .map(|v| v.seconds * 1_000_000 + i64::from(v.nanos) / 1000),
        Some(0)
    );
    let ordinary_push = scope_requests
        .push
        .iter()
        .find(|request| request.name == "scope-agent")
        .expect("ordinary scope push");
    assert_eq!(
        ordinary_push
            .time
            .as_ref()
            .map(|v| v.seconds * 1_000_000 + i64::from(v.nanos) / 1000),
        None
    );
    let telemetry_mark = host
        .marks()
        .into_iter()
        .find(|request| request.mark.as_ref().unwrap().name == "tool-exec-telemetry")
        .expect("extended mark request");
    assert_eq!(
        telemetry_mark.mark.as_ref().unwrap().severity,
        Some(pb::LogSeverity::Warn as i32)
    );
    assert_eq!(
        telemetry_mark.mark.as_ref().unwrap().category.as_deref(),
        Some("custom")
    );
    let data_schema = telemetry_mark
        .mark
        .as_ref()
        .unwrap()
        .data_schema
        .clone()
        .expect("mark data schema");
    assert_eq!(
        decode_json_envelope::<DataSchema, _>(&data_schema).unwrap(),
        DataSchema::builder()
            .name("nemo.relay.metric_measurements")
            .version("1")
            .build()
    );
    let metric_mark = host
        .marks()
        .into_iter()
        .find(|request| request.mark.as_ref().unwrap().name == "tool-exec-metric")
        .expect("metric mark request");
    assert_eq!(
        metric_mark.mark.as_ref().unwrap().category.as_deref(),
        Some("custom")
    );
    let metric_schema = metric_mark
        .mark
        .as_ref()
        .unwrap()
        .data_schema
        .clone()
        .expect("metric data schema");
    assert_eq!(
        decode_json_envelope::<DataSchema, _>(&metric_schema).unwrap(),
        DataSchema::builder()
            .name("nemo.relay.metric_measurements")
            .version("1")
            .build()
    );
    assert_eq!(
        decode_json_envelope::<Json, _>(
            &metric_mark
                .mark
                .as_ref()
                .unwrap()
                .data
                .clone()
                .expect("metric data")
        )
        .unwrap(),
        json!({
            "measurements": [{
                "name": "worker.requests",
                "kind": "counter",
                "value_type": "u64",
                "value": 1,
                "unit": null,
                "description": null,
                "attributes": null,
                "boundaries": null
            }]
        })
    );

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

    let mut invalid = tool_invoke(
        "tool-request",
        RegistrationSurface::ToolRequestIntercept,
        json!({}),
    )
    .into_wire()
    .unwrap();
    invalid.invocation = None;
    assert_eq!(
        client
            .wire()
            .invoke(Request::new(invalid))
            .await
            .unwrap_err()
            .code(),
        tonic::Code::InvalidArgument
    );
    assert_worker_error(
        client
            .invoke(Request::new(tool_invoke(
                "missing",
                RegistrationSurface::ToolRequestIntercept,
                json!({}),
            )))
            .await
            .unwrap()
            .into_inner(),
        "not registered",
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

    let mut stream_err = client
        .invoke_stream(Request::new(llm_invoke(
            "llm-stream-error",
            RegistrationSurface::LlmStreamExecutionIntercept,
            llm_request(),
            None,
            None,
        )))
        .await
        .expect("stream error invoke")
        .into_inner();
    let stream_error_chunk = stream_err
        .next()
        .await
        .expect("stream item")
        .expect("stream chunk");
    match stream_error_chunk.item.expect("stream item") {
        pb::invoke_stream_response::Item::Error(error) => {
            assert!(error.message.contains("stream boom"));
        }
        other => panic!("unexpected stream item: {other:?}"),
    }
    assert!(
        tokio::time::timeout(WORKER_TEST_TIMEOUT, stream_err.next())
            .await
            .expect("worker stream should terminate after its error")
            .is_none()
    );

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
            credentials: Some(pb::Credentials {
                activation_id: ACTIVATION_ID.into(),
                auth_token: AUTH_TOKEN.into(),
            }),
        }))
        .await
        .expect("announced endpoint should accept connections")
        .into_inner();
    assert!(health.status == pb::WorkerHealthStatus::Ready as i32);

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

    let base = tool_invoke(
        "tool-request",
        RegistrationSurface::ToolRequestIntercept,
        json!({}),
    )
    .into_wire()
    .unwrap();
    let invalid = vec![
        pb::InvokeRequest {
            invocation: None,
            ..base.clone()
        },
        pb::InvokeRequest {
            invocation: Some(pb::invoke_request::Invocation::ToolRequestIntercept(
                pb::ToolValueInvocation {
                    tool_name: "tool".into(),
                    value: None,
                },
            )),
            ..base.clone()
        },
        pb::InvokeRequest {
            invocation: Some(pb::invoke_request::Invocation::ToolRequestIntercept(
                pb::ToolValueInvocation {
                    tool_name: "tool".into(),
                    value: Some(pb::JsonValue {
                        json: b"{".to_vec(),
                    }),
                },
            )),
            ..base.clone()
        },
        pb::InvokeRequest {
            invocation: Some(pb::invoke_request::Invocation::Subscriber(
                pb::Event::default(),
            )),
            ..base.clone()
        },
        pb::InvokeRequest {
            invocation: Some(pb::invoke_request::Invocation::LlmSanitizeRequestGuardrail(
                pb::LlmSanitizeRequestInvocation::default(),
            )),
            ..base.clone()
        },
        pb::InvokeRequest {
            invocation: Some(pb::invoke_request::Invocation::LlmRequestIntercept(
                pb::LlmRequestInvocation {
                    request: Some(conversions::encode(llm_request()).unwrap()),
                    annotated_request: Some(Box::new(pb::AnnotatedLlmRequest {
                        messages: vec![pb::LlmMessage::default()],
                        ..Default::default()
                    })),
                    model_name: "model".into(),
                },
            )),
            ..base
        },
    ];
    for request in invalid {
        assert_eq!(
            client
                .wire()
                .invoke(Request::new(request))
                .await
                .unwrap_err()
                .code(),
            tonic::Code::InvalidArgument
        );
    }
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
                runtime_diagnostics_unimplemented: true,
                ..Default::default()
            },
            "runtime diagnostics require a Relay host with the worker v1 diagnostics contract",
        ),
        (
            MockHostFailures {
                emit_mark: HostFailure::WorkerError,
                ..Default::default()
            },
            "emit mark failed",
        ),
        (
            MockHostFailures {
                emit_mark: HostFailure::TransportFailure,
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
        (
            MockHostFailures {
                tool_next_missing_result: true,
                ..Default::default()
            },
            "value.result: value is required",
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

    for (failures, request, expected) in [
        (
            MockHostFailures {
                codec_request_decode: true,
                ..Default::default()
            },
            llm_request_codec_invoke(),
            "codec request decode failed",
        ),
        (
            MockHostFailures {
                codec_request_encode: true,
                ..Default::default()
            },
            llm_request_codec_invoke(),
            "codec request encode failed",
        ),
        (
            MockHostFailures {
                codec_response_decode: true,
                ..Default::default()
            },
            llm_response_codec_invoke(),
            "codec response decode failed",
        ),
    ] {
        host.set_failures(failures);
        assert_worker_error(
            client
                .invoke(Request::new(request))
                .await
                .expect("codec failure returns structured error")
                .into_inner(),
            expected,
        );
    }

    for (index, (mode, expected)) in [
        (MockStreamMode::WorkerError, "stream worker failed"),
        (MockStreamMode::EmptyChunk, "empty stream chunk"),
        (MockStreamMode::TransportError, "stream status failed"),
    ]
    .into_iter()
    .enumerate()
    {
        host.set_failures(MockHostFailures {
            llm_stream_mode: mode,
            ..Default::default()
        });
        let mut request = llm_invoke(
            "llm-stream",
            RegistrationSurface::LlmStreamExecutionIntercept,
            llm_request(),
            None,
            None,
        );
        request.invocation_id = format!("llm-stream-{index}");
        let mut stream = client
            .invoke_stream(Request::new(request))
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
        ctx.register_mark_sanitize_guardrail("duplicate", 0, |_, fields| async move { Ok(fields) });
        ctx.register_mark_sanitize_guardrail("duplicate", 1, |_, fields| async move { Ok(fields) });
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum InvalidInitialGate {
    EmptyKinds,
    EmptyTarget,
    Duplicate,
}

struct InvalidInitialGatePlugin(InvalidInitialGate);

impl WorkerPlugin for InvalidInitialGatePlugin {
    fn plugin_id(&self) -> &str {
        PLUGIN_ID
    }

    fn register(&self, ctx: &mut PluginContext, _config: &Json) -> Result<()> {
        let kinds = match self.0 {
            InvalidInitialGate::EmptyKinds => BTreeSet::new(),
            InvalidInitialGate::EmptyTarget | InvalidInitialGate::Duplicate => {
                BTreeSet::from([RuntimeRegistrationKind::Subscriber])
            }
        };
        let target = if matches!(self.0, InvalidInitialGate::EmptyTarget) {
            ""
        } else {
            "target-subscriber"
        };
        ctx.register_conditional_middleware_guardrail(
            "initial-gate",
            kinds.clone(),
            target,
            |_, _| async { Ok(Some("disabled".into())) },
        );
        if matches!(self.0, InvalidInitialGate::Duplicate) {
            ctx.register_conditional_middleware_guardrail(
                "initial-gate",
                kinds,
                target,
                |_, _| async { Ok(Some("also disabled".into())) },
            );
        }
        Ok(())
    }
}

#[derive(Default)]
struct CancellationPlugin {
    unary_started: Arc<tokio::sync::Notify>,
    unary_cancelled: Arc<AtomicBool>,
    stream_started: Arc<tokio::sync::Notify>,
    stream_cancelled: Arc<AtomicBool>,
    stream_cancelled_notified: Arc<tokio::sync::Notify>,
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
        ctx.register_tool_execution_intercept("cancel-unary", 0, move |_context, _next| {
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
        let stream_cancelled_notified = self.stream_cancelled_notified.clone();
        ctx.register_llm_stream_execution_intercept("cancel-stream", 0, move |_, _, _, _| {
            let stream_started = stream_started.clone();
            let stream_cancelled = stream_cancelled.clone();
            let stream_cancelled_notified = stream_cancelled_notified.clone();
            async move {
                Ok(Box::pin(FillThenPendingStream {
                    started: stream_started,
                    cancelled: stream_cancelled,
                    cancelled_notified: stream_cancelled_notified,
                    yielded: 0,
                }) as JsonStream)
            }
        });

        let stream_setup_started = self.stream_setup_started.clone();
        let stream_setup_cancelled = self.stream_setup_cancelled.clone();
        ctx.register_llm_stream_execution_intercept("cancel-stream-setup", 0, move |_, _, _, _| {
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
    cancelled_notified: Arc<tokio::sync::Notify>,
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
        std::thread::sleep(Duration::from_millis(50));
        self.cancelled.store(true, Ordering::SeqCst);
        self.cancelled_notified.notify_one();
    }
}

async fn assert_runtime_registration_surfaces(runtime: &PluginRuntime) -> Result<()> {
    // One request per kind: batching them would only expose the
    // resulting set, which two swapped arms leave unchanged.
    let expected_surfaces = REGISTRATION_KIND_SURFACES
        .iter()
        .map(|(_, surface)| *surface as i32)
        .collect::<BTreeSet<_>>();
    for (kind, _) in REGISTRATION_KIND_SURFACES {
        let registrations = runtime
            .list_runtime_registrations(Some(BTreeSet::from([*kind])))
            .await?;
        if registrations.len() != REGISTRATION_KIND_SURFACES.len() {
            return Err(WorkerSdkError::Callback(format!(
                "expected one registration per surface, got {}",
                registrations.len()
            )));
        }
        let mut decoded_surfaces = BTreeSet::new();
        for registration in &registrations {
            let surface: i32 = registration
                .local_name
                .strip_prefix("surface-")
                .and_then(|value| value.parse().ok())
                .ok_or_else(|| {
                    WorkerSdkError::Callback(format!(
                        "unexpected registration name {}",
                        registration.local_name
                    ))
                })?;
            let expected = REGISTRATION_KIND_SURFACES
                .iter()
                .find(|(_, candidate)| *candidate as i32 == surface)
                .map(|(kind, _)| *kind)
                .ok_or_else(|| WorkerSdkError::Callback(format!("unknown surface {surface}")))?;
            if registration.kind != expected {
                return Err(WorkerSdkError::Callback(format!(
                    "surface {surface} decoded as {:?}, expected {expected:?}",
                    registration.kind
                )));
            }
            if registration.effective_name != format!("effective-{surface}") {
                return Err(WorkerSdkError::Callback(format!(
                    "surface {surface} decoded effective name {}",
                    registration.effective_name
                )));
            }
            if registration.owner.kind != nemo_relay_worker::RuntimeRegistrationOwnerKind::Plugin
                || registration.owner.plugin_kind.as_deref() != Some("worker-sdk-tests")
                || registration.owner.component_ordinal != Some(0)
            {
                return Err(WorkerSdkError::Callback(format!(
                    "registration owner did not round-trip: {:?}",
                    registration.owner
                )));
            }
            decoded_surfaces.insert(surface);
        }
        // Set equality, so a duplicated surface standing in for a
        // missing one cannot pass the count check.
        if decoded_surfaces != expected_surfaces {
            return Err(WorkerSdkError::Callback(format!(
                "decoded surfaces {decoded_surfaces:?} do not match the table"
            )));
        }
    }
    Ok(())
}

#[derive(Default)]
struct SurfacePlugin {
    events: Arc<Mutex<Vec<String>>>,
}

struct DiscoveryPlugin;

impl WorkerPlugin for DiscoveryPlugin {
    fn plugin_id(&self) -> &str {
        PLUGIN_ID
    }

    fn register(&self, ctx: &mut PluginContext, _config: &Json) -> Result<()> {
        let runtime = ctx.runtime().unwrap();
        ctx.register_tool_request_intercept("discover", 1, false, move |_, _| {
            let runtime = runtime.clone();
            async move {
                Ok(serde_json::to_value(
                    runtime.list_runtime_registrations(None).await?,
                )?)
            }
        });
        Ok(())
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn worker_runtime_discovery_validates_host_identities_and_preserves_owner_kinds() {
    use pb::{RuntimeRegistrationIdentity as Identity, RuntimeRegistrationOwner as Owner};

    let host = MockHost::default();
    let (host_handle, host_endpoint) = spawn_host(host.clone()).await;
    let (worker_handle, mut client) =
        spawn_worker(Arc::new(DiscoveryPlugin), tcp_endpoint(&host_endpoint)).await;
    register_plugin(&mut client).await;
    let valid = Identity {
        kind: RegistrationSurface::Subscriber as i32,
        local_name: "local".into(),
        effective_name: "effective".into(),
        owner: Some(Owner {
            owner: Some(pb::runtime_registration_owner::Owner::Plugin(
                pb::PluginRegistrationOwner {
                    plugin_kind: "acme.owner".into(),
                    component_ordinal: Some(3),
                },
            )),
        }),
    };
    for (kind, expected) in [
        (pb::runtime_registration_owner::Owner::Core(()), "core"),
        (
            pb::runtime_registration_owner::Owner::GlobalApi(()),
            "global_api",
        ),
        (
            valid.owner.as_ref().unwrap().owner.clone().unwrap(),
            "plugin",
        ),
    ] {
        let mut identity = valid.clone();
        identity.owner.as_mut().unwrap().owner = Some(kind);
        *host.registration_response.lock().unwrap() = Some(ListRuntimeRegistrationsResponse {
            registrations: vec![identity],
        });
        let result = invoke_json(
            &mut client,
            tool_invoke(
                "discover",
                RegistrationSurface::ToolRequestIntercept,
                json!({}),
            ),
        )
        .await;
        assert_eq!(result[0]["kind"], "subscriber");
        assert_eq!(result[0]["local_name"], "local");
        assert_eq!(result[0]["effective_name"], "effective");
        assert_eq!(result[0]["owner"]["kind"], expected);
        if expected == "plugin" {
            assert_eq!(result[0]["owner"]["plugin_kind"], "acme.owner");
            assert_eq!(result[0]["owner"]["component_ordinal"], 3);
        }
    }
    let owner = |_| Some(Owner { owner: None });
    for (identity, expected) in [
        (
            Identity {
                kind: i32::MAX,
                ..valid.clone()
            },
            "unknown RegistrationSurface",
        ),
        (
            Identity {
                kind: RegistrationSurface::Unspecified as i32,
                ..valid.clone()
            },
            "unknown RegistrationSurface",
        ),
        (
            Identity {
                owner: None,
                ..valid.clone()
            },
            "missing required value.owner",
        ),
        (
            Identity {
                owner: owner(i32::MAX),
                ..valid.clone()
            },
            "missing required registration owner",
        ),
        (
            Identity {
                owner: owner(0),
                ..valid.clone()
            },
            "missing required registration owner",
        ),
    ] {
        *host.registration_response.lock().unwrap() = Some(ListRuntimeRegistrationsResponse {
            registrations: vec![identity],
        });
        assert_worker_error(
            client
                .invoke(Request::new(tool_invoke(
                    "discover",
                    RegistrationSurface::ToolRequestIntercept,
                    json!({}),
                )))
                .await
                .unwrap()
                .into_inner(),
            expected,
        );
    }
    *host.registration_error.lock().unwrap() = Some(Status::internal("host discovery failed"));
    assert_worker_error(
        client
            .invoke(Request::new(tool_invoke(
                "discover",
                RegistrationSurface::ToolRequestIntercept,
                json!({}),
            )))
            .await
            .unwrap()
            .into_inner(),
        "host discovery failed",
    );
    worker_handle.abort();
    host_handle.abort();
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
        ctx.register_conditional_middleware_guardrail(
            "initial-gate",
            BTreeSet::from([RuntimeRegistrationKind::Subscriber]),
            "missing-target",
            |_, registration_name| {
                let decision =
                    (registration_name == "missing-target").then(|| "initial gate".to_string());
                async move { Ok(decision) }
            },
        );
        let events = self.events.clone();
        ctx.register_subscriber("subscriber", move |event| {
            events
                .lock()
                .expect("events lock")
                .push(event.name().into());
        });
        ctx.register_event_metadata_injector("event-metadata", 1, |event| async move {
            Ok(BTreeMap::from([(
                "worker.event_name".into(),
                json!(event.name()),
            )]))
        });
        ctx.register_mark_sanitize_guardrail("event-sanitize", 1, |event, mut fields| {
            let event_name = event.name().to_owned();
            async move {
                fields.data = Some(json!({"name": event_name, "phase": "mark"}));
                Ok(fields)
            }
        });
        ctx.register_scope_sanitize_start_guardrail(
            "event-sanitize",
            1,
            |_, mut fields| async move {
                fields.metadata = Some(json!({"phase": "scope_start"}));
                Ok(fields)
            },
        );
        ctx.register_scope_sanitize_end_guardrail(
            "event-sanitize",
            1,
            |_, mut fields| async move {
                fields.metadata = Some(json!({"phase": "scope_end"}));
                Ok(fields)
            },
        );
        ctx.register_tool_sanitize_request_guardrail("tool-sanitize", 1, |_, value| async move {
            Ok(set_json_field(value, "phase", "tool_sanitize_request"))
        });
        ctx.register_tool_sanitize_response_guardrail("tool-sanitize", 1, |_, value| async move {
            Ok(set_json_field(value, "phase", "tool_sanitize_response"))
        });
        ctx.register_tool_conditional_execution_guardrail(
            "tool-conditional",
            1,
            |_, value| async move {
                tokio::task::yield_now().await;
                Ok(value
                    .get("block")
                    .and_then(serde_json::Value::as_bool)
                    .and_then(|blocked| blocked.then(|| "blocked-tool".into())))
            },
        );
        ctx.register_tool_request_intercept("tool-request", 1, false, |_, value| async move {
            tokio::task::yield_now().await;
            Ok(set_json_field(value, "phase", "tool_request"))
        });
        ctx.register_tool_request_intercept("tool-error", 1, false, |_, _| async {
            Err(WorkerSdkError::Callback("boom".into()))
        });

        let tool_runtime = runtime.clone();
        ctx.register_tool_execution_intercept("tool-exec", 1, move |context, next: ToolNext| {
            let runtime = tool_runtime.clone();
            async move {
                if context.tool_name() != "tool" || context.args() != &json!({}) {
                    return Err(WorkerSdkError::Callback(format!(
                        "unexpected tool execution context: {context:?}"
                    )));
                }
                let tool_call_id = context.tool_call_id().map(str::to_owned);
                let value = context.into_args();
                assert_runtime_registration_surfaces(&runtime).await?;
                let duplicate_error = runtime
                    .register_conditional_middleware_guardrail(
                        "initial-gate",
                        BTreeSet::from([RuntimeRegistrationKind::Subscriber]),
                        "target-subscriber",
                        |_, _| async { Ok(None) },
                    )
                    .await
                    .expect_err("activation callback names must remain reserved");
                if !matches!(duplicate_error, WorkerSdkError::InvalidInput(_)) {
                    return Err(WorkerSdkError::Callback(format!(
                        "unexpected duplicate callback error: {duplicate_error}"
                    )));
                }
                let gate = runtime
                    .register_conditional_middleware_guardrail(
                        "timer-gate",
                        BTreeSet::from([RuntimeRegistrationKind::Subscriber]),
                        "target-subscriber",
                        |_, _| async { Ok(Some("timer active".into())) },
                    )
                    .await?;
                if !runtime
                    .deregister_conditional_middleware_guardrail(&gate)
                    .await?
                {
                    return Err(WorkerSdkError::Callback(
                        "mock runtime did not remove the owned gate".into(),
                    ));
                }
                let diagnostics = runtime.runtime_diagnostics().await?;
                if diagnostics
                    .get("otel.metric_mark_invalid")
                    .map(|diagnostic| diagnostic.count)
                    != Some(3)
                {
                    return Err(WorkerSdkError::Callback(
                        "runtime diagnostics response did not contain the expected entry".into(),
                    ));
                }
                let log_fields = serde_json::Map::from_iter([(
                    "nested".into(),
                    json!({"ok": true}),
                ), ("ordinal".into(), json!(0))]);
                nemo_relay_worker::relay_trace!(runtime, target: "plugin.logging", fields: log_fields, "trace message").await?;
                nemo_relay_worker::relay_debug!(runtime, target: "plugin.logging", "debug message").await?;
                nemo_relay_worker::relay_info!(runtime, target: "plugin.logging", "info message").await?;
                nemo_relay_worker::relay_warn!(runtime, target: "plugin.logging", "warn message").await?;
                nemo_relay_worker::relay_error!(runtime, target: "plugin.logging", "error message").await?;
                runtime.log(PluginLogLevel::Info, "", "direct message", None).await?;
                let method_fields = serde_json::Map::from_iter([("method".into(), json!(true))]);
                runtime.trace("plugin.methods", "method message", Some(&method_fields)).await?;
                runtime.debug("plugin.methods", "method message", Some(&method_fields)).await?;
                runtime.info("plugin.methods", "method message", Some(&method_fields)).await?;
                runtime.warn("plugin.methods", "method message", Some(&method_fields)).await?;
                runtime.error("plugin.methods", "method message", Some(&method_fields)).await?;
                runtime.emit_mark("tool-exec", None, None).await?;
                runtime
                    .emit_mark_with_options_and_category(
                        "tool-exec-telemetry",
                        Some(json!({"measurements": []})),
                        None,
                        EmitMarkOptions {
                            data_schema: Some(
                                DataSchema::builder()
                                    .name("nemo.relay.metric_measurements")
                                    .version("1")
                                    .build(),
                            ),
                            severity: Some(LogSeverity::Warn),
                        },
                        EventCategory::custom(),
                    )
                    .await?;
                runtime
                    .emit_metric_with_category(
                        "tool-exec-metric",
                        vec![MetricMeasurement {
                            name: "worker.requests".into(),
                            kind: MetricKind::Counter,
                            value_type: MetricValueType::U64,
                            value: json!(1),
                            unit: None,
                            description: None,
                            attributes: None,
                            boundaries: None,
                        }],
                        None,
                        EventCategory::custom(),
                    )
                    .await?;
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
                    .push_scope_at(
                        None,
                        "worker-scope",
                        ScopeType::Function,
                        None,
                        None,
                        None,
                        UNIX_EPOCH - Duration::from_nanos(1_500),
                    )
                    .await?;
                runtime
                    .pop_scope_at(&handle, None, None, UNIX_EPOCH)
                    .await?;
                runtime.drop_scope_stack(&stack_id).await?;
                let mut next_value = next.call(value).await?;
                next_value.result = set_json_field(next_value.result, "phase", "tool_exec");
                next_value
                    .result
                    .as_object_mut()
                    .expect("tool execution result must be a JSON object")
                    .insert("tool_call_id".into(), json!(tool_call_id));
                Ok(
                    ToolExecutionInterceptOutcome::from(next_value).with_pending_mark(
                        PendingMarkSpec::builder()
                            .name("worker.tool.execution")
                            .data(json!({"checkpoint": true}))
                            .build(),
                    ),
                )
            }
        });
        let scope_runtime = runtime.clone();
        ctx.register_tool_execution_intercept("tool-scope-types", 1, move |_context, _next| {
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

        ctx.register_llm_sanitize_request_guardrail(
            "llm-sanitize-request",
            1,
            |mut request, context| async move {
                if let Some(codec) = context.resolve_codec() {
                    let annotated = codec.decode(&request).await?;
                    request = codec.encode(&annotated, &request).await?;
                }
                Ok(Some(set_llm_phase(request, "llm_sanitize_request")))
            },
        );
        ctx.register_llm_sanitize_response_guardrail(
            "llm-sanitize-response",
            1,
            |value, context| async move {
                if let Some(codec) = context.resolve_codec() {
                    codec.decode(&value).await?;
                }
                Ok(Some(set_json_field(
                    value,
                    "phase",
                    "llm_sanitize_response",
                )))
            },
        );
        ctx.register_llm_sanitize_request_guardrail(
            "llm-sanitize-omit-request",
            1,
            |_request, _context| async { Ok(None) },
        );
        ctx.register_llm_sanitize_response_guardrail(
            "llm-sanitize-omit-response",
            1,
            |_response, _context| async { Ok(None) },
        );
        ctx.register_llm_conditional_execution_guardrail(
            "llm-conditional",
            1,
            |request| async move {
                tokio::task::yield_now().await;
                Ok(request
                    .content
                    .get("block")
                    .and_then(serde_json::Value::as_bool)
                    .and_then(|blocked| blocked.then(|| "blocked-llm".into())))
            },
        );
        ctx.register_llm_request_intercept(
            "llm-request",
            1,
            false,
            |_, request, annotated| async move {
                tokio::task::yield_now().await;
                Ok(nemo_relay_worker::LlmRequestInterceptOutcome::new(
                    set_llm_phase(request, "llm_request"),
                    annotated,
                ))
            },
        );

        ctx.register_llm_execution_intercept(
            "llm-exec",
            1,
            move |_, request, _context, next: LlmNext| async move {
                let next_value = next.call(request).await?;
                Ok(set_json_field(next_value, "phase", "llm_exec"))
            },
        );

        let stream_runtime = runtime.clone();
        ctx.register_llm_stream_execution_intercept(
            "llm-stream",
            1,
            move |_, request, _context, next: LlmStreamNext| {
                let runtime = stream_runtime.clone();
                async move {
                    let next_stream = next.call(request).await?;
                    Ok(Box::pin(RuntimeMarkThenStream::new(runtime, next_stream)) as JsonStream)
                }
            },
        );
        ctx.register_llm_stream_execution_intercept("llm-stream-error", 1, |_, _, _, _| async {
            let stream: JsonStream = Box::pin(
                tokio_stream::once(Err(WorkerSdkError::Callback("stream boom".into())))
                    .chain(tokio_stream::pending()),
            );
            Ok(stream)
        });
        ctx.register_llm_stream_execution_intercept(
            "llm-stream-open-error",
            1,
            |_, _, _, _| async { Err(WorkerSdkError::Callback("stream open boom".into())) },
        );
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
    TransportFailure,
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
    empty_gate_handle: bool,
    empty_scope_stack_id: bool,
    empty_scope_handle_id: bool,
    runtime_diagnostics_unimplemented: bool,
    emit_mark: HostFailure,
    create_scope_stack: bool,
    push_scope: bool,
    pop_scope: HostFailure,
    drop_scope_stack: HostFailure,
    tool_next: bool,
    tool_next_missing_result: bool,
    llm_next: bool,
    llm_stream_mode: MockStreamMode,
    codec_request_decode: bool,
    codec_request_encode: bool,
    codec_response_decode: bool,
}

#[derive(Clone, Default)]
struct RuntimeRegistrationRequests {
    list: Vec<ListRuntimeRegistrationsRequest>,
    register: Vec<RegisterConditionalMiddlewareGuardrailRequest>,
    deregister: Vec<DeregisterConditionalMiddlewareGuardrailRequest>,
}

#[derive(Clone, Default)]
struct ScopeRequests {
    push: Vec<PushScopeRequest>,
    pop: Vec<PopScopeRequest>,
}

#[derive(Clone, Default)]
struct MockHost {
    calls: Arc<Mutex<Vec<String>>>,
    logs: Arc<Mutex<Vec<LogRequest>>>,
    marks: Arc<Mutex<Vec<EmitMarkRequest>>>,
    failures: Arc<Mutex<MockHostFailures>>,
    runtime_registration_requests: Arc<Mutex<RuntimeRegistrationRequests>>,
    scope_requests: Arc<Mutex<ScopeRequests>>,
    registration_response: Arc<Mutex<Option<ListRuntimeRegistrationsResponse>>>,
    registration_error: Arc<Mutex<Option<Status>>>,
    snapshot_error: Arc<Mutex<Option<Status>>>,
    snapshot_response: Arc<Mutex<Option<pb::GetCurrentScopeStackSnapshotResponse>>>,
}

impl MockHost {
    fn calls(&self) -> Vec<String> {
        self.calls.lock().expect("calls lock").clone()
    }

    fn record(&self, call: impl Into<String>) {
        self.calls.lock().expect("calls lock").push(call.into());
    }

    fn marks(&self) -> Vec<EmitMarkRequest> {
        self.marks.lock().expect("marks lock").clone()
    }

    fn logs(&self) -> Vec<LogRequest> {
        self.logs.lock().expect("logs lock").clone()
    }

    fn failures(&self) -> MockHostFailures {
        *self.failures.lock().expect("failures lock")
    }

    fn set_failures(&self, failures: MockHostFailures) {
        *self.failures.lock().expect("failures lock") = failures;
    }

    fn runtime_registration_requests(&self) -> RuntimeRegistrationRequests {
        self.runtime_registration_requests
            .lock()
            .expect("runtime registration requests lock")
            .clone()
    }

    fn scope_requests(&self) -> ScopeRequests {
        self.scope_requests
            .lock()
            .expect("scope requests lock")
            .clone()
    }
}

#[tonic::async_trait]
impl RelayHostRuntime for MockHost {
    async fn get_current_scope_stack_snapshot(
        &self,
        request: Request<pb::GetCurrentScopeStackSnapshotRequest>,
    ) -> std::result::Result<Response<pb::GetCurrentScopeStackSnapshotResponse>, Status> {
        let request = request.into_inner();
        let credentials = request.credentials.as_ref().unwrap();
        authorize_host(&credentials.activation_id, &credentials.auth_token)?;
        if let Some(error) = self.snapshot_error.lock().unwrap().clone() {
            return Err(error);
        }
        self.snapshot_response
            .lock()
            .unwrap()
            .clone()
            .map(Response::new)
            .ok_or_else(|| Status::unimplemented("snapshot extension unavailable"))
    }

    async fn log(
        &self,
        request: Request<LogRequest>,
    ) -> std::result::Result<Response<pb::LogResponse>, Status> {
        let request = request.into_inner();
        let credentials = request.credentials.as_ref().unwrap();
        authorize_host(&credentials.activation_id, &credentials.auth_token)?;
        self.logs.lock().expect("logs lock").push(request.clone());
        self.record(format!("log:{}:{}", request.target, request.message));
        Ok(Response::new(host_ack()))
    }

    async fn list_runtime_registrations(
        &self,
        request: Request<ListRuntimeRegistrationsRequest>,
    ) -> std::result::Result<Response<ListRuntimeRegistrationsResponse>, Status> {
        let request = request.into_inner();
        let credentials = request.credentials.as_ref().unwrap();
        authorize_host(&credentials.activation_id, &credentials.auth_token)?;
        self.runtime_registration_requests
            .lock()
            .expect("runtime registration requests lock")
            .list
            .push(request);
        if let Some(error) = self.registration_error.lock().unwrap().clone() {
            return Err(error);
        }
        if let Some(response) = self.registration_response.lock().unwrap().clone() {
            return Ok(Response::new(response));
        }
        Ok(Response::new(ListRuntimeRegistrationsResponse {
            registrations: REGISTRATION_KIND_SURFACES
                .iter()
                .map(|(_, surface)| pb::RuntimeRegistrationIdentity {
                    kind: *surface as i32,
                    // Names the surface that produced the record, so the
                    // decoded kind is checked against it.
                    local_name: format!("surface-{}", *surface as i32),
                    effective_name: format!("effective-{}", *surface as i32),
                    owner: Some(pb::RuntimeRegistrationOwner {
                        owner: Some(pb::runtime_registration_owner::Owner::Plugin(
                            pb::PluginRegistrationOwner {
                                plugin_kind: "worker-sdk-tests".into(),
                                component_ordinal: Some(0),
                            },
                        )),
                    }),
                })
                .collect(),
        }))
    }

    async fn register_conditional_middleware_guardrail(
        &self,
        request: Request<RegisterConditionalMiddlewareGuardrailRequest>,
    ) -> std::result::Result<Response<RegisterConditionalMiddlewareGuardrailResponse>, Status> {
        let request = request.into_inner();
        let credentials = request.credentials.as_ref().unwrap();
        authorize_host(&credentials.activation_id, &credentials.auth_token)?;
        self.runtime_registration_requests
            .lock()
            .expect("runtime registration requests lock")
            .register
            .push(request);
        Ok(Response::new(
            RegisterConditionalMiddlewareGuardrailResponse {
                handle: if self.failures().empty_gate_handle {
                    String::new()
                } else {
                    "gate-handle".into()
                },
            },
        ))
    }

    async fn deregister_conditional_middleware_guardrail(
        &self,
        request: Request<DeregisterConditionalMiddlewareGuardrailRequest>,
    ) -> std::result::Result<Response<DeregisterConditionalMiddlewareGuardrailResponse>, Status>
    {
        let request = request.into_inner();
        let credentials = request.credentials.as_ref().unwrap();
        authorize_host(&credentials.activation_id, &credentials.auth_token)?;
        self.runtime_registration_requests
            .lock()
            .expect("runtime registration requests lock")
            .deregister
            .push(request);
        Ok(Response::new(
            DeregisterConditionalMiddlewareGuardrailResponse { removed: true },
        ))
    }

    async fn get_runtime_diagnostics(
        &self,
        request: Request<GetRuntimeDiagnosticsRequest>,
    ) -> std::result::Result<Response<GetRuntimeDiagnosticsResponse>, Status> {
        let request = request.into_inner();
        let credentials = request.credentials.as_ref().unwrap();
        authorize_host(&credentials.activation_id, &credentials.auth_token)?;
        self.record("runtime_diagnostics");
        if self.failures().runtime_diagnostics_unimplemented {
            return Err(Status::unimplemented("runtime diagnostics are unavailable"));
        }
        Ok(Response::new(GetRuntimeDiagnosticsResponse {
            entries: vec![ProtoRuntimeDiagnostic {
                code: "otel.metric_mark_invalid".into(),
                message: "metric mark has an unsupported value".into(),
                count: 3,
            }],
        }))
    }

    async fn emit_mark(
        &self,
        request: Request<EmitMarkRequest>,
    ) -> std::result::Result<Response<pb::EmitMarkResponse>, Status> {
        let request = request.into_inner();
        let credentials = request.credentials.as_ref().unwrap();
        authorize_host(&credentials.activation_id, &credentials.auth_token)?;
        self.marks.lock().expect("marks lock").push(request.clone());
        let scope = request.scope.expect("scope context");
        self.record(format!(
            "mark:{}:{}:{}",
            request.mark.as_ref().unwrap().name,
            scope.scope_stack_id,
            scope.parent_scope_id
        ));
        match self.failures().emit_mark {
            HostFailure::None => {}
            HostFailure::WorkerError => {
                return Err(Status::internal("emit mark failed"));
            }
            HostFailure::TransportFailure => {
                return Err(Status::unavailable("host call failed"));
            }
        }
        Ok(Response::new(host_ack()))
    }

    async fn push_scope(
        &self,
        request: Request<PushScopeRequest>,
    ) -> std::result::Result<Response<PushScopeResponse>, Status> {
        let request = request.into_inner();
        let credentials = request.credentials.as_ref().unwrap();
        authorize_host(&credentials.activation_id, &credentials.auth_token)?;
        self.scope_requests
            .lock()
            .expect("scope requests lock")
            .push
            .push(request.clone());
        let scope = request.scope.expect("scope context");
        self.record(format!(
            "push:{}:{}:{}",
            request.name, scope.scope_stack_id, scope.parent_scope_id
        ));
        if self.failures().push_scope {
            return Err(Status::internal(
                (worker_error("push scope failed")).message,
            ));
        }
        Ok(Response::new(PushScopeResponse {
            scope_handle_id: if self.failures().empty_scope_handle_id {
                String::new()
            } else {
                "scope-handle-1".into()
            },
        }))
    }

    async fn pop_scope(
        &self,
        request: Request<PopScopeRequest>,
    ) -> std::result::Result<Response<pb::PopScopeResponse>, Status> {
        let request = request.into_inner();
        let credentials = request.credentials.as_ref().unwrap();
        authorize_host(&credentials.activation_id, &credentials.auth_token)?;
        self.scope_requests
            .lock()
            .expect("scope requests lock")
            .pop
            .push(request.clone());
        self.record(format!("pop:{}", request.scope_handle_id));
        match self.failures().pop_scope {
            HostFailure::None => {}
            HostFailure::WorkerError => {
                return Err(Status::internal("pop scope failed"));
            }
            HostFailure::TransportFailure => {
                return Err(Status::unavailable("host call failed"));
            }
        }
        Ok(Response::new(host_ack()))
    }

    async fn create_scope_stack(
        &self,
        request: Request<CreateScopeStackRequest>,
    ) -> std::result::Result<Response<CreateScopeStackResponse>, Status> {
        let request = request.into_inner();
        let credentials = request.credentials.as_ref().unwrap();
        authorize_host(&credentials.activation_id, &credentials.auth_token)?;
        self.record("create_scope_stack");
        if self.failures().create_scope_stack {
            return Err(Status::internal(
                (worker_error("create scope stack failed")).message,
            ));
        }
        Ok(Response::new(CreateScopeStackResponse {
            scope_stack_id: if self.failures().empty_scope_stack_id {
                String::new()
            } else {
                "isolated-stack".into()
            },
        }))
    }

    async fn drop_scope_stack(
        &self,
        request: Request<DropScopeStackRequest>,
    ) -> std::result::Result<Response<pb::DropScopeStackResponse>, Status> {
        let request = request.into_inner();
        let credentials = request.credentials.as_ref().unwrap();
        authorize_host(&credentials.activation_id, &credentials.auth_token)?;
        self.record(format!("drop:{}", request.scope_stack_id));
        match self.failures().drop_scope_stack {
            HostFailure::None => {}
            HostFailure::WorkerError => {
                return Err(Status::internal("drop scope stack failed"));
            }
            HostFailure::TransportFailure => {
                return Err(Status::unavailable("host call failed"));
            }
        }
        Ok(Response::new(host_ack()))
    }

    async fn tool_next(
        &self,
        request: Request<ToolNextRequest>,
    ) -> std::result::Result<Response<ToolExecutionResultResponse>, Status> {
        let request = request.into_inner();
        let credentials = request.credentials.as_ref().unwrap();
        authorize_host(&credentials.activation_id, &credentials.auth_token)?;
        self.record(format!("tool_next:{}", request.continuation_id));
        if self.failures().tool_next {
            return Err(Status::internal((worker_error("tool next failed")).message));
        }
        let value = if self.failures().tool_next_missing_result {
            ProtoToolExecutionResult {
                result: None,
                annotation: None,
            }
        } else {
            ProtoToolExecutionResult {
                result: Some(json_value(&json!({"next": "tool"})).expect("encode tool result")),
                annotation: Some(
                    json_value(&json!({"source": "host"})).expect("encode annotation"),
                ),
            }
        };
        Ok(Response::new(pb::ToolNextResponse {
            result: Some(pb::tool_next_response::Result::Value(value)),
        }))
    }

    async fn llm_next(
        &self,
        request: Request<LlmNextRequest>,
    ) -> std::result::Result<Response<pb::LlmNextResponse>, Status> {
        let request = request.into_inner();
        let credentials = request.credentials.as_ref().unwrap();
        authorize_host(&credentials.activation_id, &credentials.auth_token)?;
        self.record(format!("llm_next:{}", request.continuation_id));
        if self.failures().llm_next {
            return Ok(Response::new(pb::LlmNextResponse {
                result: Some(pb::llm_next_response::Result::Error(worker_error(
                    "llm next failed",
                ))),
            }));
        }
        Ok(Response::new(pb::LlmNextResponse {
            result: Some(pb::llm_next_response::Result::Value(json_env(
                json!({"next": "llm"}),
            ))),
        }))
    }

    type LlmStreamNextStream =
        Pin<Box<dyn Stream<Item = std::result::Result<pb::LlmStreamNextResponse, Status>> + Send>>;

    async fn llm_stream_next(
        &self,
        request: Request<LlmStreamNextRequest>,
    ) -> std::result::Result<Response<Self::LlmStreamNextStream>, Status> {
        let request = request.into_inner();
        let credentials = request.credentials.as_ref().unwrap();
        authorize_host(&credentials.activation_id, &credentials.auth_token)?;
        self.record(format!("llm_stream_next:{}", request.continuation_id));
        let chunks: Vec<std::result::Result<pb::LlmStreamNextResponse, Status>> =
            match self.failures().llm_stream_mode {
                MockStreamMode::Value => vec![Ok(pb::LlmStreamNextResponse {
                    item: Some(pb::llm_stream_next_response::Item::Value(json_env(
                        json!({"next": "llm_stream"}),
                    ))),
                })],
                MockStreamMode::WorkerError => vec![Ok(pb::LlmStreamNextResponse {
                    item: Some(pb::llm_stream_next_response::Item::Error(worker_error(
                        "stream worker failed",
                    ))),
                })],
                MockStreamMode::TransportError => {
                    vec![Err(Status::unavailable("stream status failed"))]
                }
                MockStreamMode::EmptyChunk => vec![Ok(pb::LlmStreamNextResponse { item: None })],
            };
        let stream = tokio_stream::iter(chunks);
        Ok(Response::new(Box::pin(stream)))
    }

    async fn decode_llm_codec_request(
        &self,
        request: Request<pb::DecodeLlmCodecRequestRequest>,
    ) -> std::result::Result<Response<pb::DecodeLlmCodecRequestResponse>, Status> {
        let request = request.into_inner();
        let credentials = request.credentials.as_ref().unwrap();
        authorize_host(&credentials.activation_id, &credentials.auth_token)?;
        self.record(format!(
            "codec_request_decode:{}",
            request.codec_capability_id
        ));
        if self.failures().codec_request_decode {
            return Ok(Response::new(pb::DecodeLlmCodecRequestResponse {
                result: Some(pb::decode_llm_codec_request_response::Result::Error(
                    worker_error("codec request decode failed"),
                )),
            }));
        }
        Ok(Response::new(pb::DecodeLlmCodecRequestResponse {
            result: Some(pb::decode_llm_codec_request_response::Result::Value(
                Box::default(),
            )),
        }))
    }

    async fn encode_llm_codec_request(
        &self,
        request: Request<pb::EncodeLlmCodecRequestRequest>,
    ) -> std::result::Result<Response<pb::EncodeLlmCodecRequestResponse>, Status> {
        let request = request.into_inner();
        let credentials = request.credentials.as_ref().unwrap();
        authorize_host(&credentials.activation_id, &credentials.auth_token)?;
        self.record(format!(
            "codec_request_encode:{}",
            request.codec_capability_id
        ));
        if self.failures().codec_request_encode {
            return Ok(Response::new(pb::EncodeLlmCodecRequestResponse {
                result: Some(pb::encode_llm_codec_request_response::Result::Error(
                    worker_error("codec request encode failed"),
                )),
            }));
        }
        Ok(Response::new(pb::EncodeLlmCodecRequestResponse {
            result: Some(pb::encode_llm_codec_request_response::Result::Value(
                json_envelope(
                    "nemo.relay.LlmRequest@1",
                    &json!({"headers": {}, "content": {}}),
                )
                .expect("encode LLM request"),
            )),
        }))
    }

    async fn decode_llm_codec_response(
        &self,
        request: Request<pb::DecodeLlmCodecResponseRequest>,
    ) -> std::result::Result<Response<pb::DecodeLlmCodecResponseResponse>, Status> {
        let request = request.into_inner();
        let credentials = request.credentials.as_ref().unwrap();
        authorize_host(&credentials.activation_id, &credentials.auth_token)?;
        self.record(format!(
            "codec_response_decode:{}",
            request.codec_capability_id
        ));
        if self.failures().codec_response_decode {
            return Ok(Response::new(pb::DecodeLlmCodecResponseResponse {
                result: Some(pb::decode_llm_codec_response_response::Result::Error(
                    worker_error("codec response decode failed"),
                )),
            }));
        }
        Ok(Response::new(pb::DecodeLlmCodecResponseResponse {
            result: Some(pb::decode_llm_codec_response_response::Result::Value(
                Box::default(),
            )),
        }))
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

async fn register_plugin(client: &mut PluginWorkerClient<Channel>) -> Vec<pb::Registration> {
    let identity = client
        .health(Request::new(HealthRequest {
            credentials: Some(pb::Credentials {
                activation_id: ACTIVATION_ID.into(),
                auth_token: AUTH_TOKEN.into(),
            }),
        }))
        .await
        .unwrap()
        .into_inner()
        .identity
        .unwrap();
    client
        .register(Request::new(RegisterRequest {
            plugin_id: identity.plugin_id,
            config: Some(json_env(json!({}))),
            credentials: Some(pb::Credentials {
                activation_id: ACTIVATION_ID.into(),
                auth_token: AUTH_TOKEN.into(),
            }),
        }))
        .await
        .expect("register succeeds")
        .into_inner()
        .plan()
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
    conversions::encode(value).unwrap()
}

fn invalid_json_env(_schema: &str) -> JsonEnvelope {
    JsonEnvelope {
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
    let event = match surface {
        RegistrationSurface::ScopeSanitizeStartGuardrail
        | RegistrationSurface::ScopeSanitizeEndGuardrail => {
            Event::Scope(nemo_relay_types::api::event::ScopeEvent {
                base: BaseEvent::builder()
                    .name("subscriber-event")
                    .data(json!({"raw":true}))
                    .build(),
                scope_category: if surface == RegistrationSurface::ScopeSanitizeStartGuardrail {
                    nemo_relay_types::api::event::ScopeCategory::Start
                } else {
                    nemo_relay_types::api::event::ScopeCategory::End
                },
                attributes: vec![],
                category: nemo_relay_types::api::event::EventCategory::custom(),
                category_profile: None,
            })
        }
        _ => event,
    };
    InvokeRequest {
        invocation_id: "invoke-1".into(),
        registration_name: registration_name.into(),
        surface: surface as i32,
        continuation_id: "next-1".into(),
        scope: Some(scope_context()),
        payload: Some(Payload::Event(Box::new(
            json_envelope("nemo.relay.Event@1", &event).expect("encode event"),
        ))),
        credentials: Some(pb::Credentials {
            activation_id: ACTIVATION_ID.into(),
            auth_token: AUTH_TOKEN.into(),
        }),
    }
}

fn tool_invoke(
    registration_name: &str,
    surface: RegistrationSurface,
    value: Json,
) -> InvokeRequest {
    InvokeRequest {
        invocation_id: "invoke-1".into(),
        registration_name: registration_name.into(),
        surface: surface as i32,
        continuation_id: "next-1".into(),
        scope: Some(scope_context()),
        payload: Some(Payload::Tool(ToolInvocation {
            tool_name: "tool".into(),
            value: Some(json_env(value)),
            tool_call_id: None,
        })),
        credentials: Some(pb::Credentials {
            activation_id: ACTIVATION_ID.into(),
            auth_token: AUTH_TOKEN.into(),
        }),
    }
}

fn llm_invoke(
    registration_name: &str,
    surface: RegistrationSurface,
    request: LlmRequest,
    annotated_request: Option<Json>,
    response: Option<Json>,
) -> InvokeRequest {
    let execution_codec_context = execution_codec_context_for(surface);
    InvokeRequest {
        invocation_id: "invoke-1".into(),
        registration_name: registration_name.into(),
        surface: surface as i32,
        continuation_id: "next-1".into(),
        scope: Some(scope_context()),
        payload: Some(Payload::Llm(Box::new(LlmInvocation {
            model_name: "model".into(),
            request: Some(
                json_envelope("nemo.relay.LlmRequest@1", &request).expect("encode request"),
            ),
            annotated_request: annotated_request
                .map(|v| json_envelope(ANNOTATED_LLM_REQUEST_SCHEMA, &v).unwrap()),
            response: response.map(json_env),
            sanitize_context: None,
            execution_codec_context,
        }))),
        credentials: Some(pb::Credentials {
            activation_id: ACTIVATION_ID.into(),
            auth_token: AUTH_TOKEN.into(),
        }),
    }
}

fn execution_codec_context_for(surface: RegistrationSurface) -> Option<Box<ExecutionCodecContext>> {
    match surface {
        RegistrationSurface::LlmExecutionIntercept => Some(Box::new(ExecutionCodecContext {
            request: Some(empty_request_codec_context()),
            response: Some(empty_response_codec_context()),
        })),
        RegistrationSurface::LlmStreamExecutionIntercept => Some(Box::new(ExecutionCodecContext {
            request: Some(empty_request_codec_context()),
            response: None,
        })),
        _ => None,
    }
}

fn empty_request_codec_context() -> pb::LlmCodecContext {
    pb::LlmCodecContext {
        codec: Some(pb::LlmCodecReference {
            kind: "none".into(),
            id: None,
        }),
        capability_id: None,
    }
}

fn empty_response_codec_context() -> pb::LlmCodecContext {
    pb::LlmCodecContext {
        codec: Some(pb::LlmCodecReference {
            kind: "none".into(),
            id: None,
        }),
        capability_id: None,
    }
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

fn llm_request_codec_invoke() -> InvokeRequest {
    let mut request = llm_invoke(
        "llm-sanitize-request",
        RegistrationSurface::LlmSanitizeRequestGuardrail,
        llm_request(),
        None,
        None,
    );
    if let Some(Payload::Llm(invocation)) = request.payload.as_mut() {
        invocation.sanitize_context = Some(SanitizeContext::RequestSanitizeContext(
            pb::LlmCodecContext {
                codec: Some(pb::LlmCodecReference {
                    kind: "builtin".into(),
                    id: Some("openai_chat".into()),
                }),
                capability_id: Some("request-capability".into()),
            },
        ));
    }
    request
}

fn llm_response_codec_invoke() -> InvokeRequest {
    let mut request = llm_invoke(
        "llm-sanitize-response",
        RegistrationSurface::LlmSanitizeResponseGuardrail,
        llm_request(),
        None,
        Some(json!({"value": "secret"})),
    );
    if let Some(Payload::Llm(invocation)) = request.payload.as_mut() {
        invocation.sanitize_context = Some(SanitizeContext::ResponseSanitizeContext(
            pb::LlmCodecContext {
                codec: Some(pb::LlmCodecReference {
                    kind: "opaque".into(),
                    id: None,
                }),
                capability_id: Some("response-capability".into()),
            },
        ));
    }
    request
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
        pb::invoke_response::Result::ToolValue(value)
        | pb::invoke_response::Result::LlmResponse(value) => conversions::decode(value).unwrap(),
        pb::invoke_response::Result::LlmRequest(value) => {
            serde_json::to_value(conversions::decode::<LlmRequest, _>(value).unwrap()).unwrap()
        }
        pb::invoke_response::Result::MetadataAdditions(value) => {
            conversions::decode(value).unwrap()
        }
        pb::invoke_response::Result::EventSanitize(value) => serde_json::to_value(
            conversions::decode::<nemo_relay_worker::EventSanitizeFields, _>(*value).unwrap(),
        )
        .unwrap(),
        pb::invoke_response::Result::ToolExecutionOutcome(result) => {
            decode_tool_execution_outcome(result).result
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
        pb::invoke_response::Result::ToolExecutionOutcome(result) => {
            let outcome = result;
            decode_tool_execution_outcome(outcome)
        }
        other => panic!("unexpected invoke result: {other:?}"),
    }
}

fn decode_tool_execution_outcome(
    outcome: ProtoToolExecutionInterceptOutcome,
) -> ToolExecutionInterceptOutcome {
    conversions::decode(outcome).unwrap()
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
        pb::invoke_response::Result::Guardrail(result) => match result.decision.unwrap() {
            pb::guardrail_decision::Decision::Allow(()) => String::new(),
            pb::guardrail_decision::Decision::BlockReason(reason) => reason,
        },
        other => panic!("unexpected invoke result: {other:?}"),
    }
}

fn guardrail_reason(response: InvokeResponse) -> String {
    match response.result.expect("invoke result") {
        pb::invoke_response::Result::Guardrail(result) => match result.decision.unwrap() {
            pb::guardrail_decision::Decision::Allow(()) => String::new(),
            pb::guardrail_decision::Decision::BlockReason(reason) => reason,
        },
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
        pb::invoke_response::Result::LlmRequestOutcome(result) => {
            conversions::decode::<nemo_relay_worker::LlmRequestInterceptOutcome, _>(*result)
                .unwrap()
                .request
        }
        other => panic!("unexpected invoke result: {other:?}"),
    }
}

fn stream_json(chunk: StreamChunk) -> Json {
    match chunk.item.expect("stream chunk item") {
        pb::invoke_stream_response::Item::Value(value) => {
            decode_json_envelope(&value).expect("decode stream value")
        }
        other => panic!("unexpected stream chunk: {other:?}"),
    }
}

fn assert_empty_response(response: InvokeResponse) {
    assert!(matches!(
        response.result.expect("invoke result"),
        pb::invoke_response::Result::Completed(_) | pb::invoke_response::Result::Unchanged(_)
    ));
}

fn assert_json_field(value: Json, key: &str, expected: &str) {
    assert_eq!(value.get(key), Some(&json!(expected)));
}

fn assert_worker_error(response: InvokeResponse, expected: &str) {
    match response.result.expect("invoke result") {
        pb::invoke_response::Result::Error(error) => {
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
        pb::invoke_stream_response::Item::Error(error) => {
            assert!(
                error.message.contains(expected),
                "expected '{expected}' in '{}'",
                error.message
            );
        }
        other => panic!("unexpected stream item: {other:?}"),
    }
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

fn host_ack<R: Default>() -> R {
    R::default()
}

fn worker_error(message: &str) -> WorkerError {
    WorkerError {
        code: "mock.error".into(),
        message: message.into(),
        retryable: false,
        downstream_continuation_id: None,
        http_status: None,
    }
}

fn authorize_host(activation_id: &str, auth_token: &str) -> std::result::Result<(), Status> {
    if activation_id != ACTIVATION_ID || auth_token != AUTH_TOKEN {
        return Err(Status::permission_denied("invalid host auth"));
    }
    Ok(())
}

struct SnapshotPlugin;
impl WorkerPlugin for SnapshotPlugin {
    fn plugin_id(&self) -> &str {
        PLUGIN_ID
    }
    fn register(&self, context: &mut PluginContext, _: &Json) -> Result<()> {
        let runtime = context.runtime().unwrap();
        context.register_tool_request_intercept("snapshot", 1, false, move |_, _| {
            let runtime = runtime.clone();
            async move {
                Ok(serde_json::to_value(
                    runtime.current_scope_stack_snapshot().await?,
                )?)
            }
        });
        Ok(())
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn scope_stack_snapshot_worker_decodes_fresh_values_and_explicit_errors() {
    use pb::{GetCurrentScopeStackSnapshotResponse, ScopeStackEntry, ScopeStackSnapshot};
    let host = MockHost::default();
    let (host_handle, host_endpoint) = spawn_host(host.clone()).await;
    let (worker_handle, mut client) =
        spawn_worker(Arc::new(SnapshotPlugin), tcp_endpoint(&host_endpoint)).await;
    register_plugin(&mut client).await;
    assert_worker_error(
        client
            .invoke(Request::new(tool_invoke(
                "snapshot",
                RegistrationSurface::ToolRequestIntercept,
                json!({}),
            )))
            .await
            .unwrap()
            .into_inner(),
        "unsupported",
    );
    let fixture: nemo_relay_types::api::scope::ScopeStackSnapshot = serde_json::from_str(
        include_str!("../../types/tests/fixtures/scope_stack_snapshot.json"),
    )
    .unwrap();
    let scopes = fixture
        .scopes
        .iter()
        .map(|entry| ScopeStackEntry {
            scope_id: entry.scope_id.to_string(),
            scope_type: match entry.scope_type {
                ScopeType::Agent => 1,
                ScopeType::Function => 2,
                ScopeType::Tool => 3,
                ScopeType::Llm => 4,
                _ => 11,
            },
        })
        .collect();
    *host.snapshot_response.lock().unwrap() = Some(GetCurrentScopeStackSnapshotResponse {
        result: Some(
            pb::get_current_scope_stack_snapshot_response::Result::Snapshot(ScopeStackSnapshot {
                scopes,
            }),
        ),
    });
    let first = invoke_json(
        &mut client,
        tool_invoke(
            "snapshot",
            RegistrationSurface::ToolRequestIntercept,
            json!({}),
        ),
    )
    .await;
    assert_eq!(first, serde_json::to_value(&fixture).unwrap());
    for (discriminant, expected_type) in [
        (1, "agent"),
        (2, "function"),
        (3, "tool"),
        (4, "llm"),
        (5, "retriever"),
        (6, "embedder"),
        (7, "reranker"),
        (8, "guardrail"),
        (9, "evaluator"),
        (10, "custom"),
        (11, "unknown"),
    ] {
        *host.snapshot_response.lock().unwrap() = Some(GetCurrentScopeStackSnapshotResponse {
            result: Some(
                pb::get_current_scope_stack_snapshot_response::Result::Snapshot(
                    ScopeStackSnapshot {
                        scopes: vec![ScopeStackEntry {
                            scope_id: fixture.scopes[0].scope_id.to_string(),
                            scope_type: discriminant,
                        }],
                    },
                ),
            ),
        });
        let value = invoke_json(
            &mut client,
            tool_invoke(
                "snapshot",
                RegistrationSurface::ToolRequestIntercept,
                json!({}),
            ),
        )
        .await;
        assert_eq!(value["scopes"][0]["scope_type"], expected_type);
    }
    for discriminant in [0, i32::MAX] {
        *host.snapshot_response.lock().unwrap() = Some(GetCurrentScopeStackSnapshotResponse {
            result: Some(
                pb::get_current_scope_stack_snapshot_response::Result::Snapshot(
                    ScopeStackSnapshot {
                        scopes: vec![ScopeStackEntry {
                            scope_id: fixture.scopes[0].scope_id.to_string(),
                            scope_type: discriminant,
                        }],
                    },
                ),
            ),
        });
        let response = client
            .invoke(Request::new(tool_invoke(
                "snapshot",
                RegistrationSurface::ToolRequestIntercept,
                json!({}),
            )))
            .await
            .unwrap()
            .into_inner();
        assert_worker_error(response, "unknown ScopeType");
    }
    *host.snapshot_response.lock().unwrap() = Some(GetCurrentScopeStackSnapshotResponse {
        result: Some(pb::get_current_scope_stack_snapshot_response::Result::NoContext(())),
    });
    assert_eq!(
        invoke_json(
            &mut client,
            tool_invoke(
                "snapshot",
                RegistrationSurface::ToolRequestIntercept,
                json!({})
            )
        )
        .await,
        Json::Null
    );
    *host.snapshot_response.lock().unwrap() = Some(GetCurrentScopeStackSnapshotResponse {
        result: Some(
            pb::get_current_scope_stack_snapshot_response::Result::Snapshot(ScopeStackSnapshot {
                scopes: vec![ScopeStackEntry {
                    scope_id: "invalid".into(),
                    scope_type: 1,
                }],
            }),
        ),
    });
    assert_worker_error(
        client
            .invoke(Request::new(tool_invoke(
                "snapshot",
                RegistrationSurface::ToolRequestIntercept,
                json!({}),
            )))
            .await
            .unwrap()
            .into_inner(),
        "invalid UUID",
    );
    *host.snapshot_error.lock().unwrap() =
        Some(Status::failed_precondition("expired snapshot binding"));
    assert_worker_error(
        client
            .invoke(Request::new(tool_invoke(
                "snapshot",
                RegistrationSurface::ToolRequestIntercept,
                json!({}),
            )))
            .await
            .unwrap()
            .into_inner(),
        "expired snapshot binding",
    );
    assert_eq!(first, serde_json::to_value(&fixture).unwrap());
    worker_handle.abort();
    host_handle.abort();
}

fn json_envelope<D: serde::Serialize, W: prost::Message + Default + prost::Name>(
    _schema: &str,
    value: &D,
) -> std::result::Result<W, Box<dyn std::error::Error>> {
    let json = serde_json::to_value(value)?;
    let bytes = match W::NAME {
        "Event" => prost::Message::encode_to_vec(&conversions::encode::<
            nemo_relay_types::api::event::Event,
            pb::Event,
        >(serde_json::from_value(json)?)?),
        "LlmRequest" => prost::Message::encode_to_vec(&conversions::encode::<
            LlmRequest,
            pb::LlmRequest,
        >(serde_json::from_value(json)?)?),
        "AnnotatedLlmRequest" => {
            prost::Message::encode_to_vec(&conversions::encode::<
                nemo_relay_types::codec::request::AnnotatedLlmRequest,
                pb::AnnotatedLlmRequest,
            >(serde_json::from_value(json)?)?)
        }
        "DataSchema" => prost::Message::encode_to_vec(&conversions::encode::<
            DataSchema,
            pb::DataSchema,
        >(serde_json::from_value(json)?)?),
        "JsonValue" => {
            prost::Message::encode_to_vec(&conversions::encode::<Json, pb::JsonValue>(json)?)
        }
        other => panic!("unhandled typed fixture {other}"),
    };
    Ok(W::decode(bytes.as_slice())?)
}
fn decode_json_envelope<D: serde::de::DeserializeOwned, W: prost::Message + prost::Name>(
    value: &W,
) -> std::result::Result<D, Box<dyn std::error::Error>> {
    let bytes = value.encode_to_vec();
    use prost::Message;
    let json = match W::NAME {
        "Event" => serde_json::to_value(conversions::decode::<
            nemo_relay_types::api::event::Event,
            _,
        >(pb::Event::decode(bytes.as_slice())?)?)?,
        "LlmRequest" => serde_json::to_value(conversions::decode::<LlmRequest, _>(
            pb::LlmRequest::decode(bytes.as_slice())?,
        )?)?,
        "DataSchema" => serde_json::to_value(conversions::decode::<DataSchema, _>(
            pb::DataSchema::decode(bytes.as_slice())?,
        )?)?,
        "AnnotatedLlmRequest" => serde_json::to_value(conversions::decode::<
            nemo_relay_types::codec::request::AnnotatedLlmRequest,
            _,
        >(
            pb::AnnotatedLlmRequest::decode(bytes.as_slice())?,
        )?)?,
        "JsonObject" => conversions::decode::<Json, _>(pb::JsonObject::decode(bytes.as_slice())?)?,
        "JsonValue" => conversions::decode::<Json, _>(pb::JsonValue::decode(bytes.as_slice())?)?,
        other => panic!("unhandled typed fixture {other}"),
    };
    Ok(serde_json::from_value(json)?)
}
fn json_value(value: &Json) -> std::result::Result<pb::JsonValue, conversions::ConversionError> {
    conversions::encode(value.clone())
}
trait RegistrationResult {
    fn plan(&self) -> pb::RegistrationPlan;
    fn error(&self) -> Option<WorkerError>;
}
impl RegistrationResult for pb::RegisterResponse {
    fn plan(&self) -> pb::RegistrationPlan {
        match &self.result {
            Some(pb::register_response::Result::Plan(v)) => v.clone(),
            other => panic!("unexpected registration {other:?}"),
        }
    }
    fn error(&self) -> Option<WorkerError> {
        match &self.result {
            Some(pb::register_response::Result::Error(v)) => Some(v.clone()),
            _ => None,
        }
    }
}
trait ValidationResult {
    fn diagnostics(&self) -> Option<Vec<nemo_relay_worker::ConfigDiagnostic>>;
}
impl ValidationResult for pb::ValidateResponse {
    fn diagnostics(&self) -> Option<Vec<nemo_relay_worker::ConfigDiagnostic>> {
        match &self.result {
            Some(pb::validate_response::Result::Diagnostics(v)) => Some(
                v.entries
                    .iter()
                    .cloned()
                    .map(|v| conversions::decode(v).unwrap())
                    .collect(),
            ),
            _ => None,
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn runtime_rejects_malformed_host_handles_and_allows_registration_retry() {
    struct CaptureRuntime(Arc<Mutex<Option<PluginRuntime>>>);
    impl WorkerPlugin for CaptureRuntime {
        fn plugin_id(&self) -> &str {
            "test.host-handles"
        }
        fn register(&self, context: &mut PluginContext, _: &Json) -> Result<()> {
            *self.0.lock().unwrap() = context.runtime();
            Ok(())
        }
    }
    let captured = Arc::new(Mutex::new(None));
    let host = MockHost::default();
    let (host_task, endpoint) = spawn_host(host.clone()).await;
    let (worker, mut client) =
        spawn_worker(Arc::new(CaptureRuntime(captured.clone())), endpoint).await;
    register_plugin(&mut client).await;
    let runtime = captured.lock().unwrap().take().unwrap();
    host.set_failures(MockHostFailures {
        empty_gate_handle: true,
        empty_scope_stack_id: true,
        empty_scope_handle_id: true,
        ..Default::default()
    });
    let error = runtime
        .register_conditional_middleware_guardrail(
            "retryable",
            BTreeSet::from([RuntimeRegistrationKind::Subscriber]),
            "target",
            |_, _| async { Ok(None) },
        )
        .await
        .unwrap_err();
    assert!(matches!(error, WorkerSdkError::InvalidInput(_)));
    assert!(matches!(
        runtime.create_scope_stack().await,
        Err(WorkerSdkError::InvalidInput(_))
    ));
    assert!(matches!(
        runtime
            .push_scope(
                Some("isolated-stack"),
                "child",
                ScopeType::Custom,
                None,
                None,
                None
            )
            .await,
        Err(WorkerSdkError::InvalidInput(_))
    ));
    assert!(matches!(
        runtime.drop_scope_stack("").await,
        Err(WorkerSdkError::InvalidInput(_))
    ));
    assert!(!host.calls().contains(&"drop:".to_owned()));

    host.set_failures(MockHostFailures::default());
    let gate = runtime
        .register_conditional_middleware_guardrail(
            "retryable",
            BTreeSet::from([RuntimeRegistrationKind::Subscriber]),
            "target",
            |_, _| async { Ok(None) },
        )
        .await
        .expect("invalid host handle must release the callback name");
    assert!(
        runtime
            .deregister_conditional_middleware_guardrail(&gate)
            .await
            .unwrap()
    );
    let stack = runtime.create_scope_stack().await.unwrap();
    assert_eq!(stack, "isolated-stack");
    let handle = runtime
        .push_scope(Some(&stack), "child", ScopeType::Custom, None, None, None)
        .await
        .unwrap();
    assert_eq!(handle, "scope-handle-1");
    runtime.pop_scope(&handle, None, None).await.unwrap();
    runtime.drop_scope_stack(&stack).await.unwrap();
    assert_eq!(host.runtime_registration_requests().register.len(), 2);
    worker.abort();
    host_task.abort();
}
