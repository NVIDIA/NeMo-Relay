// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end coverage for the Rust gRPC worker SDK service.

use std::future::Future;
use std::net::{SocketAddr, TcpListener};
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;

use futures_util::{Stream, StreamExt};
use nemo_relay_types::api::event::{BaseEvent, Event, MarkEvent};
use nemo_relay_worker::{
    Json, JsonStream, LlmNext, LlmRequest, LlmStreamNext, PluginContext, PluginRuntime, Result,
    ScopeType, ToolNext, WorkerPlugin, WorkerSdkError, WorkerServerConfig,
    serve_plugin_arc_with_config,
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
    ToolNextRequest, ValidateRequest,
};
use nemo_relay_worker_proto::{WORKER_PROTOCOL_GRPC_V1, decode_json_envelope, json_envelope};
use serde_json::json;
use tokio::task::JoinHandle;
use tonic::transport::{Channel, Server};
use tonic::{Request, Response, Status};

const ACTIVATION_ID: &str = "activation-1";
const AUTH_TOKEN: &str = "secret-token";
const PLUGIN_ID: &str = "acme.worker";

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

    let registrations = register_plugin(&mut client).await;
    assert_eq!(registrations.len(), 14);
    assert_eq!(
        registrations
            .iter()
            .filter(|registration| registration.local_name == "tool-sanitize")
            .count(),
        2
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
    assert!(cancel.message.contains("not implemented"));

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
async fn worker_service_invokes_every_registration_surface() {
    let host = MockHost::default();
    let (host_handle, host_endpoint) = spawn_host(host.clone()).await;
    let (worker_handle, mut client) = spawn_worker(
        Arc::new(SurfacePlugin::default()),
        tcp_endpoint(&host_endpoint),
    )
    .await;
    register_plugin(&mut client).await;

    let subscriber_response = client
        .invoke(Request::new(event_invoke("subscriber")))
        .await
        .expect("subscriber invoke")
        .into_inner();
    assert_empty_response(subscriber_response);

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
    let tool_exec = invoke_json(
        &mut client,
        tool_invoke(
            "tool-exec",
            RegistrationSurface::ToolExecutionIntercept,
            json!({}),
        ),
    )
    .await;
    assert_json_field(tool_exec.clone(), "next", "tool");
    assert_json_field(tool_exec, "phase", "tool_exec");

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

    let calls = host.calls();
    assert!(calls.contains(&"mark:tool-exec:stack-1:parent-1".into()));
    assert!(calls.contains(&"create_scope_stack".into()));
    assert!(calls.contains(&"push:worker-scope:stack-1:parent-1".into()));
    assert!(calls.contains(&"pop:scope-handle-1".into()));
    assert!(calls.contains(&"drop:isolated-stack".into()));
    assert!(calls.contains(&"tool_next:next-1".into()));
    assert!(calls.contains(&"llm_next:next-1".into()));
    assert!(calls.contains(&"llm_stream_next:next-1".into()));
    assert!(calls.contains(&"mark:stream-poll:stack-1:parent-1".into()));

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
                let handle = runtime
                    .push_scope(None, "worker-scope", ScopeType::Function, None, None, None)
                    .await?;
                runtime.pop_scope(&handle, None, None).await?;
                runtime.drop_scope_stack(&stack_id).await?;
                let next_value = next.call(value).await?;
                Ok(set_json_field(next_value, "phase", "tool_exec"))
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
            Ok((set_llm_phase(request, "llm_request"), annotated))
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

#[derive(Clone, Default)]
struct MockHost {
    calls: Arc<Mutex<Vec<String>>>,
}

impl MockHost {
    fn calls(&self) -> Vec<String> {
        self.calls.lock().expect("calls lock").clone()
    }

    fn record(&self, call: impl Into<String>) {
        self.calls.lock().expect("calls lock").push(call.into());
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
        Ok(Response::new(host_ack()))
    }

    async fn create_scope_stack(
        &self,
        request: Request<CreateScopeStackRequest>,
    ) -> std::result::Result<Response<CreateScopeStackResponse>, Status> {
        let request = request.into_inner();
        authorize_host(&request.activation_id, &request.auth_token)?;
        self.record("create_scope_stack");
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
        Ok(Response::new(host_ack()))
    }

    async fn tool_next(
        &self,
        request: Request<ToolNextRequest>,
    ) -> std::result::Result<Response<JsonResult>, Status> {
        let request = request.into_inner();
        authorize_host(&request.activation_id, &request.auth_token)?;
        self.record(format!("tool_next:{}", request.continuation_id));
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
        let stream = tokio_stream::iter(vec![Ok(StreamChunk {
            item: Some(nemo_relay_worker_proto::v1::stream_chunk::Item::Value(
                json_env(json!({"next": "llm_stream"})),
            )),
        })]);
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

async fn connect_worker(endpoint: &str) -> PluginWorkerClient<Channel> {
    for _ in 0..50 {
        match PluginWorkerClient::connect(endpoint.to_owned()).await {
            Ok(client) => return client,
            Err(_) => std::thread::sleep(Duration::from_millis(20)),
        }
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

fn event_invoke(registration_name: &str) -> InvokeRequest {
    let event = Event::Mark(MarkEvent::new(
        BaseEvent::builder().name("subscriber-event").build(),
        None,
        None,
    ));
    InvokeRequest {
        activation_id: ACTIVATION_ID.into(),
        invocation_id: "invoke-1".into(),
        registration_name: registration_name.into(),
        surface: RegistrationSurface::Subscriber as i32,
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
            decode_json_envelope(&result.request.expect("llm request")).expect("decode LLM request")
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

fn host_ack() -> HostAck {
    HostAck {
        ok: true,
        error: None,
    }
}

fn authorize_host(activation_id: &str, auth_token: &str) -> std::result::Result<(), Status> {
    if activation_id != ACTIVATION_ID || auth_token != AUTH_TOKEN {
        return Err(Status::permission_denied("invalid host auth"));
    }
    Ok(())
}
