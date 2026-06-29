// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Mutex};

use crate::api::event::{BaseEvent, MarkEvent};
use nemo_relay_worker_proto::json_envelope;
use nemo_relay_worker_proto::v1::invoke_response::Result as InvokeResult;
use nemo_relay_worker_proto::v1::plugin_worker_server::{PluginWorker, PluginWorkerServer};
use nemo_relay_worker_proto::v1::stream_chunk::Item as StreamItem;
use nemo_relay_worker_proto::v1::{
    CancelInvocationRequest, CreateScopeStackRequest, DropScopeStackRequest, EmitMarkRequest,
    EmptyResult, GuardrailResult, HandshakeRequest, HandshakeResponse, HealthRequest,
    HealthResponse, JsonEnvelope, JsonResult, LlmNextRequest, LlmRequestInterceptResult,
    LlmStreamNextRequest, PopScopeRequest, PushScopeRequest, Registration, ScopeContext,
    ScopeType as ProtoScopeType, ShutdownRequest, StreamChunk, ToolNextRequest, ValidateRequest,
    ValidateResponse, WorkerAck,
};
use serde_json::json;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::Request;
use tonic::transport::Server;

use super::*;

const ACTIVATION_ID: &str = "activation-test";
const AUTH_TOKEN: &str = "auth-test";

#[test]
fn response_helpers_cover_error_and_unexpected_shapes() {
    let worker_error = WorkerError {
        code: "worker.failed".into(),
        message: "boom".into(),
        retryable: false,
    };

    let error = json_from_invoke_response(InvokeResponse {
        result: Some(InvokeResult::Json(JsonResult {
            value: None,
            error: Some(worker_error.clone()),
        })),
    })
    .expect_err("json result worker error should surface");
    assert!(error.to_string().contains("worker.failed: boom"));

    let error = json_from_invoke_response(InvokeResponse {
        result: Some(InvokeResult::Error(worker_error.clone())),
    })
    .expect_err("top-level worker error should surface");
    assert!(error.to_string().contains("worker.failed: boom"));

    let error = json_from_invoke_response(InvokeResponse {
        result: Some(InvokeResult::Empty(EmptyResult {})),
    })
    .expect_err("unexpected JSON result shape should fail");
    assert!(error.to_string().contains("unexpected invoke result"));

    let error = json_from_invoke_response(InvokeResponse {
        result: Some(InvokeResult::Json(JsonResult {
            value: Some(JsonEnvelope {
                schema: JSON_SCHEMA.into(),
                json: b"{".to_vec(),
            }),
            error: None,
        })),
    })
    .expect_err("invalid JSON envelope should fail");
    assert!(error.to_string().contains("invalid JSON result"));

    assert_eq!(
        guardrail_from_invoke_response(InvokeResponse {
            result: Some(InvokeResult::Guardrail(GuardrailResult {
                block_reason: String::new(),
            })),
        })
        .expect("empty block reason is allowed"),
        None
    );
    assert_eq!(
        guardrail_from_invoke_response(InvokeResponse {
            result: Some(InvokeResult::Guardrail(GuardrailResult {
                block_reason: "blocked".into(),
            })),
        })
        .expect("block reason should parse"),
        Some("blocked".into())
    );
    assert!(
        guardrail_from_invoke_response(InvokeResponse {
            result: Some(InvokeResult::Error(worker_error.clone())),
        })
        .expect_err("guardrail worker error should surface")
        .to_string()
        .contains("worker.failed")
    );
    assert!(
        guardrail_from_invoke_response(InvokeResponse {
            result: Some(InvokeResult::Empty(EmptyResult {})),
        })
        .expect_err("unexpected guardrail shape should fail")
        .to_string()
        .contains("guardrail returned unexpected")
    );

    assert!(
        json_from_stream_chunk(StreamChunk {
            item: Some(StreamItem::Error(worker_error.clone())),
        })
        .expect_err("stream worker error should surface")
        .to_string()
        .contains("worker.failed")
    );
    assert!(
        json_from_stream_chunk(StreamChunk {
            item: Some(StreamItem::Value(JsonEnvelope {
                schema: JSON_SCHEMA.into(),
                json: b"{".to_vec(),
            })),
        })
        .expect_err("invalid stream JSON envelope should fail")
        .to_string()
        .contains("invalid worker stream chunk")
    );
    assert!(
        json_from_stream_chunk(StreamChunk { item: None })
            .expect_err("empty stream chunk should fail")
            .to_string()
            .contains("stream chunk was empty")
    );
}

#[test]
fn envelope_and_error_helpers_cover_failure_paths() {
    assert!(
        required_envelope(None, "required test")
            .expect_err("missing envelope should fail")
            .to_string()
            .contains("required test is missing")
    );
    assert!(
        optional_envelope_to_json(Some(JsonEnvelope {
            schema: JSON_SCHEMA.into(),
            json: b"not-json".to_vec(),
        }))
        .expect_err("invalid optional envelope should fail")
        .to_string()
        .contains("invalid JSON envelope")
    );

    let ack = host_ack(Err(FlowError::Internal("host failed".into())));
    assert!(!ack.ok);
    assert_eq!(ack.error.expect("host error").code, "host.runtime_error");

    let result = json_result(Err(FlowError::Internal("json failed".into())));
    assert!(result.value.is_none());
    assert_eq!(result.error.expect("json error").code, "host.runtime_error");

    let fallback = worker_error_to_plugin(
        WorkerError {
            code: "worker.empty".into(),
            message: String::new(),
            retryable: false,
        },
        "fallback message",
    );
    assert!(fallback.to_string().contains("fallback message"));

    let status = status_from_flow(FlowError::Internal("status failed".into()));
    assert_eq!(status.code(), tonic::Code::Internal);
    assert!(status.message().contains("status failed"));
}

#[test]
fn registration_plan_and_scope_type_helpers_validate_edges() {
    let empty_name = validate_registration_plan(
        "fixture_worker",
        &RegisterResponse {
            registrations: vec![Registration {
                local_name: " ".into(),
                surface: RegistrationSurface::Subscriber as i32,
                priority: 0,
                break_chain: false,
            }],
            error: None,
        },
    )
    .expect_err("empty registration names should fail");
    assert!(empty_name.to_string().contains("empty local_name"));

    let unsupported = validate_registration_plan(
        "fixture_worker",
        &RegisterResponse {
            registrations: vec![Registration {
                local_name: "bad".into(),
                surface: 999,
                priority: 0,
                break_chain: false,
            }],
            error: None,
        },
    )
    .expect_err("unsupported registration surfaces should fail");
    assert!(
        unsupported
            .to_string()
            .contains("unsupported registration surface")
    );

    let unspecified = validate_registration_plan(
        "fixture_worker",
        &RegisterResponse {
            registrations: vec![Registration {
                local_name: "bad".into(),
                surface: RegistrationSurface::Unspecified as i32,
                priority: 0,
                break_chain: false,
            }],
            error: None,
        },
    )
    .expect_err("unspecified registration surfaces should fail");
    assert!(
        unspecified
            .to_string()
            .contains("unspecified registration surface")
    );

    let cases = [
        (ProtoScopeType::Agent, crate::api::scope::ScopeType::Agent),
        (
            ProtoScopeType::Function,
            crate::api::scope::ScopeType::Function,
        ),
        (ProtoScopeType::Tool, crate::api::scope::ScopeType::Tool),
        (ProtoScopeType::Llm, crate::api::scope::ScopeType::Llm),
        (
            ProtoScopeType::Retriever,
            crate::api::scope::ScopeType::Retriever,
        ),
        (
            ProtoScopeType::Embedder,
            crate::api::scope::ScopeType::Embedder,
        ),
        (
            ProtoScopeType::Reranker,
            crate::api::scope::ScopeType::Reranker,
        ),
        (
            ProtoScopeType::Guardrail,
            crate::api::scope::ScopeType::Guardrail,
        ),
        (
            ProtoScopeType::Evaluator,
            crate::api::scope::ScopeType::Evaluator,
        ),
        (ProtoScopeType::Custom, crate::api::scope::ScopeType::Custom),
        (
            ProtoScopeType::Unknown,
            crate::api::scope::ScopeType::Unknown,
        ),
    ];
    for (proto, expected) in cases {
        assert_eq!(proto_scope_type(proto as i32), expected);
    }
    assert_eq!(proto_scope_type(999), crate::api::scope::ScopeType::Custom);
}

#[test]
fn relay_compatibility_and_blocking_helpers_cover_local_edges() {
    assert!(
        validate_relay_compatibility(None)
            .expect_err("missing relay compatibility should fail")
            .to_string()
            .contains("compat.relay is required")
    );
    assert!(
        validate_relay_compatibility(Some("not semver"))
            .expect_err("invalid relay compatibility should fail")
            .to_string()
            .contains("invalid compat.relay")
    );

    let runtime = RuntimeBuilder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime should build");
    assert_eq!(block_on_runtime(&runtime, async { 42 }), 42);
}

#[test]
#[cfg(unix)]
fn worker_endpoints_fail_when_host_socket_cannot_bind() {
    let activation_dir = std::env::temp_dir().join(format!("nmrw-unit-{}", Uuid::now_v7()));
    let host_socket = activation_dir.join("host.sock");
    std::fs::create_dir_all(&host_socket).expect("host socket directory should be created");

    let error = match WorkerEndpoints::new(&activation_dir) {
        Ok(_) => panic!("endpoint creation should fail when host socket path is a directory"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("failed to bind worker host runtime socket")
    );

    let _ = std::fs::remove_dir_all(&activation_dir);
}

#[tokio::test(flavor = "multi_thread")]
async fn callback_helpers_cover_worker_response_edges() {
    let worker_error = WorkerError {
        code: "worker.failed".into(),
        message: "boom".into(),
        retryable: false,
    };
    let (callback, _shutdown) = fake_callback_service({
        let worker_error = worker_error.clone();
        move |request| match request.registration_name.as_str() {
            "subscriber_error" => InvokeResponse {
                result: Some(InvokeResult::Error(worker_error.clone())),
            },
            "subscriber_unexpected" | "llm_intercept_unexpected" => InvokeResponse {
                result: Some(InvokeResult::Json(JsonResult {
                    value: Some(json_envelope(JSON_SCHEMA, &json!({})).expect("json envelope")),
                    error: None,
                })),
            },
            "llm_json_invalid" => InvokeResponse {
                result: Some(InvokeResult::Json(JsonResult {
                    value: Some(json_envelope(JSON_SCHEMA, &json!(null)).expect("json envelope")),
                    error: None,
                })),
            },
            "llm_intercept_invalid_request" => InvokeResponse {
                result: Some(InvokeResult::LlmRequest(LlmRequestInterceptResult {
                    request: Some(JsonEnvelope {
                        schema: LLM_REQUEST_SCHEMA.into(),
                        json: b"null".to_vec(),
                    }),
                    annotated_request: None,
                    has_annotated_request: false,
                })),
            },
            "llm_intercept_missing_annotated" => InvokeResponse {
                result: Some(InvokeResult::LlmRequest(LlmRequestInterceptResult {
                    request: Some(valid_llm_request_envelope()),
                    annotated_request: None,
                    has_annotated_request: true,
                })),
            },
            "llm_intercept_invalid_annotated" => InvokeResponse {
                result: Some(InvokeResult::LlmRequest(LlmRequestInterceptResult {
                    request: Some(valid_llm_request_envelope()),
                    annotated_request: Some(JsonEnvelope {
                        schema: ANNOTATED_LLM_REQUEST_SCHEMA.into(),
                        json: b"null".to_vec(),
                    }),
                    has_annotated_request: true,
                })),
            },
            "llm_intercept_error" => InvokeResponse {
                result: Some(InvokeResult::Error(worker_error.clone())),
            },
            _ => InvokeResponse {
                result: Some(InvokeResult::Empty(EmptyResult {})),
            },
        }
    })
    .await;
    let event = Event::Mark(MarkEvent::new(
        BaseEvent::builder().name("callback-edge").build(),
        None,
        None,
    ));

    let error = callback
        .invoke_subscriber("subscriber_error", &event)
        .expect_err("subscriber worker error should surface");
    assert!(error.to_string().contains("worker.failed: boom"));

    let error = callback
        .invoke_subscriber("subscriber_unexpected", &event)
        .expect_err("unexpected subscriber result should fail");
    assert!(error.to_string().contains("subscriber returned unexpected"));

    let error = callback
        .invoke_llm_request_json(
            "llm_json_invalid",
            RegistrationSurface::LlmSanitizeRequestGuardrail,
            "model",
            valid_llm_request(),
            None,
            None,
        )
        .expect_err("invalid LLM JSON result should fail");
    assert!(error.to_string().contains("invalid type"));

    let error = callback
        .invoke_llm_request_intercept(
            "llm_intercept_invalid_request",
            "model",
            valid_llm_request(),
            None,
        )
        .expect_err("invalid LLM intercept request should fail");
    assert!(error.to_string().contains("invalid LLM request"));

    let error = callback
        .invoke_llm_request_intercept(
            "llm_intercept_missing_annotated",
            "model",
            valid_llm_request(),
            None,
        )
        .expect_err("missing annotated request should fail when flagged present");
    assert!(
        error
            .to_string()
            .contains("llm request intercept annotated request is missing")
    );

    let error = callback
        .invoke_llm_request_intercept(
            "llm_intercept_invalid_annotated",
            "model",
            valid_llm_request(),
            None,
        )
        .expect_err("invalid annotated request should fail");
    assert!(error.to_string().contains("invalid annotated LLM request"));

    let error = callback
        .invoke_llm_request_intercept("llm_intercept_error", "model", valid_llm_request(), None)
        .expect_err("LLM intercept worker error should surface");
    assert!(error.to_string().contains("worker.failed: boom"));

    let error = callback
        .invoke_llm_request_intercept(
            "llm_intercept_unexpected",
            "model",
            valid_llm_request(),
            None,
        )
        .expect_err("unexpected LLM intercept result should fail");
    assert!(
        error
            .to_string()
            .contains("LLM request intercept returned unexpected")
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn callback_stream_transport_error_surfaces_to_host_stream() {
    let (callback, _shutdown) = fake_callback_service(|_| InvokeResponse {
        result: Some(InvokeResult::Empty(EmptyResult {})),
    })
    .await;

    let mut stream = callback
        .invoke_llm_stream_execution(
            "stream_transport_error",
            "model",
            valid_llm_request(),
            Arc::new(|_request| {
                Box::pin(async { Ok(Box::pin(tokio_stream::empty()) as LlmJsonStream) })
            }),
        )
        .await
        .expect("host stream should be returned");

    let error = stream
        .next()
        .await
        .expect("transport error should be yielded")
        .expect_err("stream transport error should surface");
    assert!(error.to_string().contains("worker stream transport failed"));
}

#[tokio::test(flavor = "multi_thread")]
async fn callback_stream_stops_when_host_receiver_is_dropped() {
    let (yield_tx, yield_rx) = oneshot::channel();
    let (stream_dropped_tx, stream_dropped_rx) = oneshot::channel();
    let stream_dropped_tx = Arc::new(Mutex::new(Some(stream_dropped_tx)));
    let yield_rx = Arc::new(Mutex::new(Some(yield_rx)));
    let (callback, _shutdown) = fake_callback_service_with_stream(
        |_| InvokeResponse {
            result: Some(InvokeResult::Empty(EmptyResult {})),
        },
        {
            let stream_dropped_tx = stream_dropped_tx.clone();
            let yield_rx = yield_rx.clone();
            move |_| {
                let dropped = stream_dropped_tx
                    .lock()
                    .expect("stream drop signal lock should not be poisoned")
                    .take()
                    .expect("test stream should be created once");
                let yield_rx = yield_rx
                    .lock()
                    .expect("stream yield signal lock should not be poisoned")
                    .take()
                    .expect("test stream should be created once");
                Box::pin(SignalChunkThenPendingStream {
                    yield_rx,
                    dropped: Some(dropped),
                    yielded: false,
                }) as FakeInvokeStream
            }
        },
    )
    .await;

    let stream = callback
        .invoke_llm_stream_execution(
            "stream_receiver_drop",
            "model",
            valid_llm_request(),
            Arc::new(|_request| {
                Box::pin(async { Ok(Box::pin(tokio_stream::empty()) as LlmJsonStream) })
            }),
        )
        .await
        .expect("host stream should be returned");
    drop(stream);
    yield_tx
        .send(())
        .expect("worker stream yield signal should be delivered");
    tokio::time::timeout(std::time::Duration::from_secs(1), stream_dropped_rx)
        .await
        .expect("worker stream should be dropped after host receiver is dropped")
        .expect("worker stream drop signal should be delivered");
}

#[tokio::test(flavor = "multi_thread")]
async fn install_registrations_covers_registry_error_edges() {
    for surface in [
        RegistrationSurface::Subscriber,
        RegistrationSurface::ToolSanitizeRequestGuardrail,
        RegistrationSurface::ToolSanitizeResponseGuardrail,
        RegistrationSurface::ToolConditionalExecutionGuardrail,
        RegistrationSurface::ToolRequestIntercept,
        RegistrationSurface::ToolExecutionIntercept,
        RegistrationSurface::LlmSanitizeRequestGuardrail,
        RegistrationSurface::LlmSanitizeResponseGuardrail,
        RegistrationSurface::LlmConditionalExecutionGuardrail,
        RegistrationSurface::LlmRequestIntercept,
        RegistrationSurface::LlmExecutionIntercept,
        RegistrationSurface::LlmStreamExecutionIntercept,
    ] {
        let (instance, _shutdown) = fake_worker_instance(vec![
            registration(surface, "duplicate"),
            registration(surface, "duplicate"),
        ])
        .await;
        let mut ctx = PluginRegistrationContext::new();
        let error = instance
            .install_registrations(&mut ctx)
            .expect_err("duplicate worker registration should fail");
        assert!(
            error.to_string().contains("duplicate")
                || error.to_string().contains("already registered"),
            "{surface:?}: {error}"
        );
        let mut registrations = ctx.into_registrations();
        crate::plugin::rollback_registrations(&mut registrations);
    }

    let (instance, _shutdown) = fake_worker_instance(vec![Registration {
        surface: 999,
        ..registration(RegistrationSurface::Subscriber, "bad")
    }])
    .await;
    let mut ctx = PluginRegistrationContext::new();
    assert!(
        instance
            .install_registrations(&mut ctx)
            .expect_err("unsupported registration surface should fail")
            .to_string()
            .contains("unsupported registration surface")
    );

    let (instance, _shutdown) =
        fake_worker_instance(vec![registration(RegistrationSurface::Unspecified, "bad")]).await;
    let mut ctx = PluginRegistrationContext::new();
    assert!(
        instance
            .install_registrations(&mut ctx)
            .expect_err("unspecified registration surface should fail")
            .to_string()
            .contains("unspecified registration surface")
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn adapter_register_rejects_config_drift_even_without_validation_call() {
    let (instance, _shutdown) = fake_worker_instance(Vec::new()).await;
    let adapter = WorkerPluginAdapter {
        plugin_kind: "fixture_worker".into(),
        allows_multiple_components: false,
        instance: Arc::new(instance),
    };
    let mut ctx = PluginRegistrationContext::new();
    let changed = serde_json::Map::from_iter([("changed".into(), json!(true))]);

    let error = adapter
        .register(&changed, &mut ctx)
        .await
        .expect_err("config drift should fail registration");
    assert!(error.to_string().contains("config changed"), "{error}");
}

#[tokio::test]
async fn host_runtime_service_covers_auth_scope_and_ack_errors() {
    let state = Arc::new(WorkerHostRuntimeState::new(
        ACTIVATION_ID.into(),
        AUTH_TOKEN.into(),
    ));
    let service = WorkerHostRuntimeService {
        state: state.clone(),
    };

    let auth_error = service
        .emit_mark(Request::new(EmitMarkRequest {
            activation_id: "wrong".into(),
            auth_token: AUTH_TOKEN.into(),
            name: "auth-failure".into(),
            scope: None,
            data: None,
            metadata: None,
        }))
        .await
        .expect_err("bad activation id should fail auth");
    assert_eq!(auth_error.code(), tonic::Code::PermissionDenied);

    let ack = service
        .emit_mark(Request::new(EmitMarkRequest {
            activation_id: ACTIVATION_ID.into(),
            auth_token: AUTH_TOKEN.into(),
            name: "missing-stack".into(),
            scope: Some(ScopeContext {
                scope_stack_id: "missing-stack".into(),
                parent_scope_id: String::new(),
            }),
            data: None,
            metadata: None,
        }))
        .await
        .expect("missing stack should return host ack")
        .into_inner();
    assert!(!ack.ok);
    assert!(
        ack.error
            .expect("missing stack error")
            .message
            .contains("not found")
    );

    let ack = service
        .emit_mark(Request::new(EmitMarkRequest {
            activation_id: ACTIVATION_ID.into(),
            auth_token: AUTH_TOKEN.into(),
            name: "no-scope".into(),
            scope: None,
            data: None,
            metadata: None,
        }))
        .await
        .expect("no-scope mark should succeed")
        .into_inner();
    assert!(ack.ok);

    let push = service
        .push_scope(Request::new(PushScopeRequest {
            activation_id: ACTIVATION_ID.into(),
            auth_token: AUTH_TOKEN.into(),
            scope: None,
            name: "invalid-json-scope".into(),
            scope_type: ProtoScopeType::Custom as i32,
            data: Some(JsonEnvelope {
                schema: JSON_SCHEMA.into(),
                json: b"not-json".to_vec(),
            }),
            metadata: None,
            input: None,
        }))
        .await
        .expect("invalid JSON should be structured")
        .into_inner();
    assert!(
        push.error
            .expect("push error")
            .message
            .contains("invalid JSON")
    );

    let pop_error = service
        .pop_scope(Request::new(PopScopeRequest {
            activation_id: ACTIVATION_ID.into(),
            auth_token: AUTH_TOKEN.into(),
            scope_handle_id: "missing-scope".into(),
            output: None,
            metadata: None,
        }))
        .await
        .expect_err("missing scope handle should fail");
    assert_eq!(pop_error.code(), tonic::Code::NotFound);

    let created = service
        .create_scope_stack(Request::new(CreateScopeStackRequest {
            activation_id: ACTIVATION_ID.into(),
            auth_token: AUTH_TOKEN.into(),
        }))
        .await
        .expect("scope stack should be created")
        .into_inner();
    let scope_stack_id = created.scope_stack_id.clone();
    assert!(
        state
            .stack("")
            .expect("empty stack id should be valid")
            .is_none()
    );
    let dropped = service
        .drop_scope_stack(Request::new(DropScopeStackRequest {
            activation_id: ACTIVATION_ID.into(),
            auth_token: AUTH_TOKEN.into(),
            scope_stack_id: scope_stack_id.clone(),
        }))
        .await
        .expect("scope stack should be dropped")
        .into_inner();
    assert!(dropped.ok);
    assert_eq!(
        state
            .stack(&scope_stack_id)
            .expect_err("dropped stack should be removed")
            .code(),
        tonic::Code::NotFound
    );

    assert_eq!(
        service
            .with_stack(
                Some(&ScopeContext {
                    scope_stack_id: String::new(),
                    parent_scope_id: String::new(),
                }),
                || Ok(7),
            )
            .expect("empty explicit stack id should run without binding"),
        7
    );
}

#[tokio::test]
async fn host_runtime_service_reports_poisoned_internal_locks() {
    let state = Arc::new(WorkerHostRuntimeState::new(
        ACTIVATION_ID.into(),
        AUTH_TOKEN.into(),
    ));
    poison_mutex({
        let state = state.clone();
        move || {
            let _guard = state.scope_handles.lock().expect("scope handles lock");
            panic!("poison scope handles");
        }
    });
    let service = WorkerHostRuntimeService {
        state: state.clone(),
    };
    let push_error = service
        .push_scope(Request::new(PushScopeRequest {
            activation_id: ACTIVATION_ID.into(),
            auth_token: AUTH_TOKEN.into(),
            scope: None,
            name: "poisoned".into(),
            scope_type: ProtoScopeType::Custom as i32,
            data: None,
            metadata: None,
            input: None,
        }))
        .await
        .expect_err("poisoned scope handle lock should fail");
    assert_eq!(push_error.code(), tonic::Code::Internal);

    let pop_error = service
        .pop_scope(Request::new(PopScopeRequest {
            activation_id: ACTIVATION_ID.into(),
            auth_token: AUTH_TOKEN.into(),
            scope_handle_id: "missing".into(),
            output: None,
            metadata: None,
        }))
        .await
        .expect_err("poisoned scope handle lock should fail");
    assert_eq!(pop_error.code(), tonic::Code::Internal);

    let state = Arc::new(WorkerHostRuntimeState::new(
        ACTIVATION_ID.into(),
        AUTH_TOKEN.into(),
    ));
    poison_mutex({
        let state = state.clone();
        move || {
            let _guard = state.scope_stacks.lock().expect("scope stacks lock");
            panic!("poison scope stacks");
        }
    });
    let service = WorkerHostRuntimeService { state };
    let create_error = service
        .create_scope_stack(Request::new(CreateScopeStackRequest {
            activation_id: ACTIVATION_ID.into(),
            auth_token: AUTH_TOKEN.into(),
        }))
        .await
        .expect_err("poisoned scope stack lock should fail");
    assert_eq!(create_error.code(), tonic::Code::Internal);

    let drop_error = service
        .drop_scope_stack(Request::new(DropScopeStackRequest {
            activation_id: ACTIVATION_ID.into(),
            auth_token: AUTH_TOKEN.into(),
            scope_stack_id: "stack".into(),
        }))
        .await
        .expect_err("poisoned scope stack lock should fail");
    assert_eq!(drop_error.code(), tonic::Code::Internal);
}

#[test]
fn owned_worker_runtime_drop_is_idempotent_when_runtime_already_taken() {
    drop(OwnedWorkerRuntime { runtime: None });
}

#[tokio::test]
async fn host_runtime_service_covers_continuation_errors_and_stream_items() {
    let state = Arc::new(WorkerHostRuntimeState::new(
        ACTIVATION_ID.into(),
        AUTH_TOKEN.into(),
    ));
    let service = WorkerHostRuntimeService {
        state: state.clone(),
    };

    let llm_continuation = state
        .insert_continuation(Continuation::Llm(Arc::new(|request| {
            Box::pin(async move { Ok(request.content) })
        })))
        .expect("llm continuation should insert");
    let wrong_type = service
        .tool_next(Request::new(ToolNextRequest {
            activation_id: ACTIVATION_ID.into(),
            auth_token: AUTH_TOKEN.into(),
            continuation_id: llm_continuation,
            value: Some(json_envelope(JSON_SCHEMA, &json!({})).expect("json envelope")),
        }))
        .await
        .expect_err("wrong continuation type should fail");
    assert_eq!(wrong_type.code(), tonic::Code::InvalidArgument);

    let tool_continuation = state
        .insert_continuation(Continuation::Tool(Arc::new(|value| {
            Box::pin(async move { Ok(value) })
        })))
        .expect("tool continuation should insert");
    let invalid_tool_json = service
        .tool_next(Request::new(ToolNextRequest {
            activation_id: ACTIVATION_ID.into(),
            auth_token: AUTH_TOKEN.into(),
            continuation_id: tool_continuation,
            value: Some(JsonEnvelope {
                schema: JSON_SCHEMA.into(),
                json: b"not-json".to_vec(),
            }),
        }))
        .await
        .expect_err("invalid tool next JSON should fail");
    assert_eq!(invalid_tool_json.code(), tonic::Code::InvalidArgument);

    let llm_continuation = state
        .insert_continuation(Continuation::Llm(Arc::new(|request| {
            Box::pin(async move { Ok(request.content) })
        })))
        .expect("llm continuation should insert");
    let invalid_llm_json = service
        .llm_next(Request::new(LlmNextRequest {
            activation_id: ACTIVATION_ID.into(),
            auth_token: AUTH_TOKEN.into(),
            continuation_id: llm_continuation,
            request: Some(JsonEnvelope {
                schema: LLM_REQUEST_SCHEMA.into(),
                json: b"not-json".to_vec(),
            }),
        }))
        .await
        .expect_err("invalid LLM next request should fail");
    assert_eq!(invalid_llm_json.code(), tonic::Code::InvalidArgument);

    let stream_continuation = state
        .insert_continuation(Continuation::LlmStream(Arc::new(|_request| {
            Box::pin(async move {
                Ok(Box::pin(tokio_stream::iter(vec![Err(FlowError::Internal(
                    "stream item failed".into(),
                ))])) as LlmJsonStream)
            })
        })))
        .expect("stream continuation should insert");
    let stream_response = service
        .llm_stream_next(Request::new(LlmStreamNextRequest {
            activation_id: ACTIVATION_ID.into(),
            auth_token: AUTH_TOKEN.into(),
            continuation_id: stream_continuation,
            request: Some(
                json_envelope(
                    LLM_REQUEST_SCHEMA,
                    &LlmRequest {
                        headers: serde_json::Map::new(),
                        content: json!({ "prompt": "stream" }),
                    },
                )
                .expect("llm request envelope"),
            ),
        }))
        .await
        .expect("stream next should return stream");
    let mut stream = stream_response.into_inner();
    let chunk = stream
        .next()
        .await
        .expect("stream should yield one item")
        .expect("transport should be ok");
    match chunk.item {
        Some(StreamItem::Error(error)) => {
            assert!(error.message.contains("stream item failed"));
        }
        other => panic!("expected worker stream error, got {other:?}"),
    }

    let stream_continuation = state
        .insert_continuation(Continuation::LlmStream(Arc::new(|_request| {
            Box::pin(async move { Ok(Box::pin(tokio_stream::empty()) as LlmJsonStream) })
        })))
        .expect("stream continuation should insert");
    let invalid_stream_request = match service
        .llm_stream_next(Request::new(LlmStreamNextRequest {
            activation_id: ACTIVATION_ID.into(),
            auth_token: AUTH_TOKEN.into(),
            continuation_id: stream_continuation,
            request: Some(JsonEnvelope {
                schema: LLM_REQUEST_SCHEMA.into(),
                json: b"not-json".to_vec(),
            }),
        }))
        .await
    {
        Ok(_) => panic!("invalid LLM stream request should fail"),
        Err(error) => error,
    };
    assert_eq!(invalid_stream_request.code(), tonic::Code::InvalidArgument);
}

fn valid_llm_request() -> LlmRequest {
    LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({ "prompt": "unit" }),
    }
}

fn valid_llm_request_envelope() -> JsonEnvelope {
    json_envelope(LLM_REQUEST_SCHEMA, &valid_llm_request()).expect("llm request envelope")
}

async fn fake_callback_service(
    invoke: impl Fn(InvokeRequest) -> InvokeResponse + Send + Sync + 'static,
) -> (WorkerPluginCallback, oneshot::Sender<()>) {
    let (client, shutdown_tx) = fake_worker_client(invoke).await;
    callback_for_client(client, shutdown_tx)
}

async fn fake_callback_service_with_stream(
    invoke: impl Fn(InvokeRequest) -> InvokeResponse + Send + Sync + 'static,
    invoke_stream: impl Fn(InvokeRequest) -> FakeInvokeStream + Send + Sync + 'static,
) -> (WorkerPluginCallback, oneshot::Sender<()>) {
    let (client, shutdown_tx) = fake_worker_client_with_stream(invoke, invoke_stream).await;
    callback_for_client(client, shutdown_tx)
}

fn callback_for_client(
    client: PluginWorkerClient<Channel>,
    shutdown_tx: oneshot::Sender<()>,
) -> (WorkerPluginCallback, oneshot::Sender<()>) {
    let state = Arc::new(WorkerHostRuntimeState::new(
        ACTIVATION_ID.into(),
        AUTH_TOKEN.into(),
    ));
    (
        WorkerPluginCallback {
            activation_id: ACTIVATION_ID.into(),
            runtime: tokio::runtime::Handle::current(),
            client,
            host_state: state,
        },
        shutdown_tx,
    )
}

async fn fake_worker_instance(
    registrations: Vec<Registration>,
) -> (WorkerPluginInstance, oneshot::Sender<()>) {
    let (client, shutdown_tx) = fake_worker_client(|_| InvokeResponse {
        result: Some(InvokeResult::Empty(EmptyResult {})),
    })
    .await;
    let activation_dir = std::env::temp_dir().join(format!("nmrw-unit-{}", Uuid::now_v7()));
    std::fs::create_dir_all(&activation_dir).expect("unit activation dir should be created");
    (
        WorkerPluginInstance {
            plugin_kind: "fixture_worker".into(),
            allows_multiple_components: false,
            config: serde_json::Map::new(),
            validation_diagnostics: Vec::new(),
            registrations,
            runtime: OwnedWorkerRuntime::new(
                RuntimeBuilder::new_multi_thread()
                    .enable_all()
                    .build()
                    .expect("worker runtime should build"),
            ),
            client,
            host_state: Arc::new(WorkerHostRuntimeState::new(
                ACTIVATION_ID.into(),
                AUTH_TOKEN.into(),
            )),
            shutdown: Mutex::new(None),
            process: Mutex::new(None),
            activation_dir,
        },
        shutdown_tx,
    )
}

async fn fake_worker_client(
    invoke: impl Fn(InvokeRequest) -> InvokeResponse + Send + Sync + 'static,
) -> (PluginWorkerClient<Channel>, oneshot::Sender<()>) {
    fake_worker_client_with_stream(invoke, |_| {
        Box::pin(tokio_stream::iter(vec![Err(Status::unavailable(
            "stream transport down",
        ))])) as FakeInvokeStream
    })
    .await
}

async fn fake_worker_client_with_stream(
    invoke: impl Fn(InvokeRequest) -> InvokeResponse + Send + Sync + 'static,
    invoke_stream: impl Fn(InvokeRequest) -> FakeInvokeStream + Send + Sync + 'static,
) -> (PluginWorkerClient<Channel>, oneshot::Sender<()>) {
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .expect("fake worker listener should bind");
    let addr = listener
        .local_addr()
        .expect("fake worker listener address should be available");
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    tokio::spawn(
        Server::builder()
            .add_service(PluginWorkerServer::new(FakePluginWorker {
                invoke: Arc::new(invoke),
                invoke_stream: Arc::new(invoke_stream),
            }))
            .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async {
                let _ = shutdown_rx.await;
            }),
    );
    let client = PluginWorkerClient::connect(format!("http://{addr}"))
        .await
        .expect("fake worker client should connect");
    (client, shutdown_tx)
}

fn registration(surface: RegistrationSurface, local_name: &str) -> Registration {
    Registration {
        local_name: local_name.into(),
        surface: surface as i32,
        priority: 0,
        break_chain: false,
    }
}

fn poison_mutex(f: impl FnOnce() + std::panic::UnwindSafe) {
    let _ = std::panic::catch_unwind(f);
}

struct FakePluginWorker {
    invoke: Arc<dyn Fn(InvokeRequest) -> InvokeResponse + Send + Sync>,
    invoke_stream: Arc<dyn Fn(InvokeRequest) -> FakeInvokeStream + Send + Sync>,
}

type FakeInvokeStream =
    Pin<Box<dyn tokio_stream::Stream<Item = std::result::Result<StreamChunk, Status>> + Send>>;

struct SignalChunkThenPendingStream {
    yield_rx: oneshot::Receiver<()>,
    dropped: Option<oneshot::Sender<()>>,
    yielded: bool,
}

impl tokio_stream::Stream for SignalChunkThenPendingStream {
    type Item = std::result::Result<StreamChunk, Status>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        if self.yielded {
            return std::task::Poll::Pending;
        }
        match Pin::new(&mut self.yield_rx).poll(cx) {
            std::task::Poll::Ready(_) => {
                self.yielded = true;
                std::task::Poll::Ready(Some(Ok(StreamChunk {
                    item: Some(StreamItem::Value(
                        json_envelope(JSON_SCHEMA, &json!({ "after_receiver_drop": true }))
                            .expect("test stream chunk should encode"),
                    )),
                })))
            }
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}

impl Drop for SignalChunkThenPendingStream {
    fn drop(&mut self) {
        if let Some(dropped) = self.dropped.take() {
            let _ = dropped.send(());
        }
    }
}

#[tonic::async_trait]
impl PluginWorker for FakePluginWorker {
    async fn handshake(
        &self,
        _request: Request<HandshakeRequest>,
    ) -> std::result::Result<tonic::Response<HandshakeResponse>, tonic::Status> {
        Ok(tonic::Response::new(HandshakeResponse {
            plugin_id: "fixture_worker".into(),
            plugin_kind: "fixture_worker".into(),
            allows_multiple_components: false,
            worker_protocol: WORKER_PROTOCOL_GRPC_V1.into(),
            sdk_name: "unit".into(),
            sdk_version: "0".into(),
            runtime_name: "unit".into(),
            runtime_version: "0".into(),
            supported_surfaces: Vec::new(),
        }))
    }

    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> std::result::Result<tonic::Response<HealthResponse>, tonic::Status> {
        Ok(tonic::Response::new(HealthResponse {
            ok: true,
            message: String::new(),
            plugin_id: "fixture_worker".into(),
            worker_protocol: WORKER_PROTOCOL_GRPC_V1.into(),
            sdk_name: "unit".into(),
            sdk_version: "0".into(),
            runtime_name: "unit".into(),
            runtime_version: "0".into(),
        }))
    }

    async fn validate(
        &self,
        _request: Request<ValidateRequest>,
    ) -> std::result::Result<tonic::Response<ValidateResponse>, tonic::Status> {
        Ok(tonic::Response::new(ValidateResponse {
            diagnostics: None,
            error: None,
        }))
    }

    async fn register(
        &self,
        _request: Request<RegisterRequest>,
    ) -> std::result::Result<tonic::Response<RegisterResponse>, tonic::Status> {
        Ok(tonic::Response::new(RegisterResponse {
            registrations: Vec::new(),
            error: None,
        }))
    }

    async fn invoke(
        &self,
        request: Request<InvokeRequest>,
    ) -> std::result::Result<tonic::Response<InvokeResponse>, tonic::Status> {
        Ok(tonic::Response::new((self.invoke)(request.into_inner())))
    }

    type InvokeStreamStream =
        Pin<Box<dyn tokio_stream::Stream<Item = std::result::Result<StreamChunk, Status>> + Send>>;

    async fn invoke_stream(
        &self,
        request: Request<InvokeRequest>,
    ) -> std::result::Result<tonic::Response<Self::InvokeStreamStream>, tonic::Status> {
        Ok(tonic::Response::new((self.invoke_stream)(
            request.into_inner(),
        )))
    }

    async fn cancel_invocation(
        &self,
        _request: Request<CancelInvocationRequest>,
    ) -> std::result::Result<tonic::Response<WorkerAck>, tonic::Status> {
        Ok(tonic::Response::new(WorkerAck {
            accepted: false,
            message: "not implemented".into(),
        }))
    }

    async fn shutdown(
        &self,
        _request: Request<ShutdownRequest>,
    ) -> std::result::Result<tonic::Response<WorkerAck>, tonic::Status> {
        Ok(tonic::Response::new(WorkerAck {
            accepted: false,
            message: "not implemented".into(),
        }))
    }
}
