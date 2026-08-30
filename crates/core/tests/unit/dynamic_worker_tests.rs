// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Mutex};

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

use crate::api::event::{BaseEvent, MarkEvent};
use crate::api::optimization::{
    LlmOptimizationRecorder, record_llm_optimization_contribution, scope_llm_optimization_recorder,
};
use crate::api::runtime::{
    BuiltinLlmCodec, LlmCodecIdentity, LlmSanitizeRequestContext, LlmSanitizeResponseContext,
    MiddlewareContinuationLease, NemoRelayContextState,
};
use crate::api::tool::ToolExecutionResult;
use crate::codec::openai_chat::OpenAIChatCodec;
use crate::codec::optimization::LlmOptimizationContribution;
use crate::codec::traits::{LlmCodec, LlmResponseCodec};
use nemo_relay_worker_proto::v1::invoke_response::Result as InvokeResult;
use nemo_relay_worker_proto::v1::plugin_worker_server::{PluginWorker, PluginWorkerServer};
use nemo_relay_worker_proto::v1::stream_chunk::Item as StreamItem;
use nemo_relay_worker_proto::v1::{
    CancelInvocationRequest, CreateScopeStackRequest, DropScopeStackRequest, EmitMarkRequest,
    EmptyResult, GetRuntimeDiagnosticsRequest, GuardrailResult, HandshakeRequest,
    HandshakeResponse, HealthRequest, HealthResponse, JsonEnvelope, JsonResult, JsonValue,
    LlmCodecDecodeRequest, LlmCodecDecodeResponse, LlmCodecEncodeRequest, LlmNextRequest,
    LlmRequestInterceptResult, LlmStreamNextRequest, PopScopeRequest, PushScopeRequest,
    Registration, ScopeContext, ScopeType as ProtoScopeType, ShutdownRequest, StreamChunk,
    ToolExecutionInterceptOutcome as ProtoToolExecutionInterceptOutcome,
    ToolExecutionInterceptResult, ToolExecutionResultResponse, ToolNextRequest, ValidateRequest,
    ValidateResponse, WorkerAck,
};
use nemo_relay_worker_proto::{decode_json_value, json_envelope};
use serde_json::json;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::Request;
use tonic::transport::Server;

use super::*;

const ACTIVATION_ID: &str = "activation-test";
const AUTH_TOKEN: &str = "auth-test";

#[test]
fn worker_runtime_rejects_gate_registration_after_cleanup() {
    let state = WorkerHostRuntimeState::new(ACTIVATION_ID.into(), AUTH_TOKEN.into());
    state.cleanup_conditional_middleware_guardrails();

    let error = state
        .register_owned_conditional_middleware_guardrail(
            "late-gate".into(),
            BTreeSet::from([RuntimeRegistrationKind::Subscriber]),
            "target-subscriber".into(),
            "disabled".into(),
            false,
        )
        .expect_err("gate registration after cleanup must fail");

    assert!(matches!(error, FlowError::NotFound(_)));
}

#[test]
fn worker_gate_cleanup_recovers_a_poisoned_ownership_lock() {
    let state = WorkerHostRuntimeState::new(ACTIVATION_ID.into(), AUTH_TOKEN.into());
    state
        .conditional_middleware_guardrails
        .lock()
        .expect("gate ownership lock")
        .insert(
            "gate-handle".into(),
            WorkerOwnedGate {
                local_name: "owned-gate".into(),
                qualified_name: "missing-global-gate".into(),
            },
        );
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = state
            .conditional_middleware_guardrails
            .lock()
            .expect("gate ownership lock");
        panic!("poison gate ownership lock");
    }));

    state.cleanup_conditional_middleware_guardrails();

    assert!(
        state
            .conditional_middleware_guardrails
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .is_empty()
    );
}

fn enable_operational_logs() {
    let _ = spdlog::init_log_crate_proxy();
    log::set_max_level(log::LevelFilter::Info);
}

#[tokio::test]
async fn continuation_context_preserves_optimization_recorder_across_tasks() {
    for producer in ["worker-unary-next", "worker-stream-next"] {
        let recorder = LlmOptimizationRecorder::default();
        let context = scope_llm_optimization_recorder(recorder.clone(), async {
            MiddlewareContinuationContext::capture()
        })
        .await;
        tokio::spawn(async move {
            context
                .run(async move {
                    tokio::task::yield_now().await;
                    assert!(record_llm_optimization_contribution(
                        LlmOptimizationContribution::new(producer, "worker_next")
                    ));
                })
                .await;
        })
        .await
        .unwrap();
        let contributions = recorder.unemitted();
        assert_eq!(contributions.len(), 1);
        assert_eq!(contributions[0].producer, producer);
    }
}

#[test]
fn python_environment_resolution_requires_lifecycle_managed_path() {
    enable_operational_logs();
    let plugin_id = "acme.python";
    let digest = Sha256::digest(plugin_id.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let temp = tempfile::tempdir().unwrap();
    let managed = temp.path().join(MANAGED_ENVIRONMENTS_DIR).join(digest);
    let python = resolve_python_executable(plugin_id, managed.to_str()).unwrap();
    assert!(python.starts_with(&managed));

    let outside = std::env::temp_dir().join("unmanaged-python-environment");
    let error = resolve_python_executable(plugin_id, outside.to_str())
        .expect_err("an arbitrary environment path should be rejected");
    assert!(
        error
            .to_string()
            .contains("is not the lifecycle-managed path")
    );

    #[cfg(unix)]
    {
        use std::os::unix::fs::symlink;

        let target = temp.path().join("symlink-target");
        std::fs::create_dir_all(&target).unwrap();
        std::fs::create_dir_all(managed.parent().unwrap()).unwrap();
        symlink(&target, &managed).unwrap();

        let error = resolve_python_executable(plugin_id, managed.to_str())
            .expect_err("a symlinked environment should be rejected");
        assert!(error.to_string().contains("must not be a symbolic link"));
    }
}

#[test]
fn python_worker_launch_clears_host_python_environment() {
    enable_operational_logs();
    let mut command = Command::new("python");
    clear_host_python_environment(&mut command);
    let removed = command
        .get_envs()
        .filter_map(|(key, value)| value.is_none().then_some(key))
        .collect::<Vec<_>>();
    for key in ["PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"] {
        assert!(removed.contains(&std::ffi::OsStr::new(key)));
    }
}

#[cfg(unix)]
#[test]
fn python_worker_process_launch_uses_the_managed_interpreter_and_endpoint_file() {
    let plugin_id = "acme.python.launch";
    let digest = Sha256::digest(plugin_id.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let temp = tempfile::tempdir().unwrap();
    let managed = temp.path().join(MANAGED_ENVIRONMENTS_DIR).join(digest);
    let interpreter = managed.join("bin/python");
    std::fs::create_dir_all(interpreter.parent().unwrap()).unwrap();
    let probe = temp.path().join("launch-probe");
    std::fs::write(
        &interpreter,
        format!(
            "#!/bin/sh\nprintf '%s\\n%s\\n' \"$0\" \"$NEMO_RELAY_WORKER_ENDPOINT_FILE\" > '{}'\nexit 0\n",
            probe.display()
        ),
    )
    .unwrap();
    let mut permissions = std::fs::metadata(&interpreter).unwrap().permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&interpreter, permissions).unwrap();

    let endpoint_file = temp.path().join("worker-endpoint");
    let mut child = spawn_worker_process(WorkerProcessLaunch {
        runtime: WorkerRuntime::Python,
        manifest_path: &temp.path().join("plugin.toml"),
        environment_ref: managed.to_str(),
        plugin_id,
        entrypoint: "acme_worker:create_plugin",
        activation_id: "activation",
        auth_token: "token",
        host_endpoint: "http://127.0.0.1:1",
        worker_endpoint: "http://127.0.0.1:2",
        worker_endpoint_file: Some(&endpoint_file),
    })
    .unwrap();
    assert!(child.wait().unwrap().success());
    let recorded = std::fs::read_to_string(&probe).unwrap();
    let mut lines = recorded.lines();
    assert_eq!(lines.next(), interpreter.to_str());
    assert_eq!(lines.next(), endpoint_file.to_str());
    assert_eq!(lines.next(), None);
}

#[cfg(not(unix))]
#[test]
fn empty_worker_endpoint_announcement_is_retried() {
    let temp = tempfile::tempdir().unwrap();
    let announcement = temp.path().join("worker-endpoint");
    let missing = temp.path().join("missing-endpoint");
    let directory = temp.path().join("endpoint-directory");
    std::fs::create_dir(&directory).unwrap();

    assert_eq!(
        normalize_worker_tcp_endpoint(" tcp://127.0.0.1:50051 ").unwrap(),
        "http://127.0.0.1:50051"
    );
    assert_eq!(
        normalize_worker_tcp_endpoint("http://127.0.0.1:50051").unwrap(),
        "http://127.0.0.1:50051"
    );
    assert!(normalize_worker_tcp_endpoint("tcp://").is_err());
    assert!(normalize_worker_tcp_endpoint("https://127.0.0.1").is_err());
    assert!(
        resolve_worker_connect_endpoint(&WorkerConnectEndpoint::Announced(missing))
            .unwrap()
            .is_none()
    );
    assert!(resolve_worker_connect_endpoint(&WorkerConnectEndpoint::Announced(directory)).is_err());
    std::fs::write(&announcement, "").unwrap();

    let endpoint = WorkerConnectEndpoint::Announced(announcement);
    assert!(
        resolve_worker_connect_endpoint(&endpoint)
            .unwrap()
            .is_none(),
        "an empty announcement file is a transient publication state"
    );
}

fn test_worker_error() -> WorkerError {
    WorkerError {
        code: "worker.failed".into(),
        message: "boom".into(),
        retryable: false,
    }
}

#[test]
fn worker_protocol_error_and_json_helpers_preserve_failure_semantics() {
    let error = test_worker_error();
    assert!(matches!(
        worker_error_to_flow(error.clone()),
        FlowError::Internal(message) if message.contains("worker.failed")
    ));
    assert!(matches!(
        worker_error_to_plugin(error, "fallback"),
        PluginError::RegistrationFailed(message) if message.contains("worker.failed")
    ));
    assert_eq!(
        json_from_invoke_response(InvokeResponse {
            result: Some(InvokeResult::Json(JsonResult {
                value: Some(JsonEnvelope {
                    schema: "test".into(),
                    json: b"{\"ok\":true}".to_vec(),
                }),
                error: None,
            })),
        })
        .unwrap(),
        serde_json::json!({"ok": true})
    );
    assert!(
        json_from_invoke_response(InvokeResponse {
            result: Some(InvokeResult::Error(test_worker_error())),
        })
        .is_err()
    );
}

#[test]
fn response_helpers_cover_json_and_guardrail_error_shapes() {
    enable_operational_logs();
    let worker_error = test_worker_error();

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
}

#[test]
fn response_helpers_cover_stream_optional_and_cancellation_errors() {
    enable_operational_logs();
    let worker_error = test_worker_error();

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

    assert!(
        optional_json_from_invoke_response(InvokeResponse {
            result: Some(InvokeResult::Json(JsonResult {
                value: None,
                error: Some(worker_error.clone()),
            })),
        })
        .unwrap_err()
        .to_string()
        .contains("worker.failed")
    );
    assert!(
        optional_json_from_invoke_response(InvokeResponse {
            result: Some(InvokeResult::Json(JsonResult {
                value: Some(JsonEnvelope {
                    schema: JSON_SCHEMA.into(),
                    json: b"{".to_vec(),
                }),
                error: None,
            })),
        })
        .unwrap_err()
        .to_string()
        .contains("invalid JSON result")
    );
    assert_eq!(
        optional_json_from_invoke_response(InvokeResponse {
            result: Some(InvokeResult::Empty(EmptyResult {})),
        })
        .expect("empty result must map to no replacement JSON"),
        None
    );
    assert!(
        optional_json_from_invoke_response(InvokeResponse {
            result: Some(InvokeResult::Guardrail(GuardrailResult {
                block_reason: String::new(),
            })),
        })
        .unwrap_err()
        .to_string()
        .contains("unexpected LLM sanitizer result")
    );

    let typed_error = typed_json_result::<Json>(
        JSON_SCHEMA,
        Err(FlowError::Internal("typed result failed".into())),
    );
    assert!(typed_error.value.is_none());
    assert!(
        typed_error
            .error
            .is_some_and(|error| error.message.contains("typed result failed"))
    );
    assert!(
        worker_error_to_flow(WorkerError {
            code: "worker.cancelled".into(),
            message: "cancelled by caller".into(),
            retryable: false,
        })
        .to_string()
        .contains("worker invocation cancelled")
    );
    assert!(
        worker_status_to_flow("unused", Status::cancelled("cancelled by transport"))
            .to_string()
            .contains("worker invocation cancelled")
    );
}

#[test]
fn envelope_and_error_helpers_cover_failure_paths() {
    enable_operational_logs();
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
    enable_operational_logs();
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
            conditional_middleware_guardrails: Vec::new(),
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
            conditional_middleware_guardrails: Vec::new(),
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
            conditional_middleware_guardrails: Vec::new(),
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
    enable_operational_logs();
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
fn worker_handshake_validation_accepts_v1_and_rejects_v2() {
    let mut handshake = HandshakeResponse {
        plugin_id: "fixture_worker".into(),
        plugin_kind: "fixture_worker".into(),
        allows_multiple_components: false,
        worker_protocol: "grpc-v2".into(),
        sdk_name: "unit".into(),
        sdk_version: "0".into(),
        runtime_name: "unit".into(),
        runtime_version: "0".into(),
        supported_surfaces: Vec::new(),
    };

    let error = validate_worker_handshake("fixture_worker", &handshake)
        .expect_err("a grpc-v2 handshake response must be rejected");
    assert!(
        error
            .to_string()
            .contains("unsupported worker_protocol 'grpc-v2'")
    );

    handshake.worker_protocol = WORKER_PROTOCOL_GRPC_V1.into();
    validate_worker_handshake("fixture_worker", &handshake)
        .expect("a grpc-v1 handshake response should be accepted");
}

#[test]
#[cfg(unix)]
fn worker_endpoints_fail_when_host_socket_cannot_bind() {
    enable_operational_logs();
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

#[cfg(unix)]
#[tokio::test]
async fn worker_unix_connection_reports_a_missing_socket() {
    let missing = std::env::temp_dir().join(format!("nmrw-missing-{}", Uuid::now_v7()));
    let error = connect_worker(&WorkerConnectEndpoint::Unix(missing.clone()))
        .await
        .expect_err("a missing worker socket should fail to connect");
    assert!(error.to_string().contains(&missing.display().to_string()));
}

#[tokio::test(flavor = "multi_thread")]
async fn callback_helpers_cover_worker_response_edges() {
    enable_operational_logs();
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
            "llm_json_invalid" | "event_fields_invalid" => InvokeResponse {
                result: Some(InvokeResult::Json(JsonResult {
                    value: Some(json_envelope(JSON_SCHEMA, &json!(null)).expect("json envelope")),
                    error: None,
                })),
            },
            "llm_intercept_invalid_request" => InvokeResponse {
                result: Some(InvokeResult::LlmRequest(LlmRequestInterceptResult {
                    outcome: Some(JsonEnvelope {
                        schema: "nemo.relay.LlmRequestInterceptOutcome@2".into(),
                        json: br#"{"request":null}"#.to_vec(),
                    }),
                })),
            },
            "llm_intercept_missing_annotated" => InvokeResponse {
                result: Some(InvokeResult::LlmRequest(LlmRequestInterceptResult {
                    outcome: Some(JsonEnvelope {
                        schema: "nemo.relay.LegacyLlmRequestInterceptResult@1".into(),
                        json: br#"{}"#.to_vec(),
                    }),
                })),
            },
            "llm_intercept_invalid_annotated" => InvokeResponse {
                result: Some(InvokeResult::LlmRequest(LlmRequestInterceptResult {
                    outcome: Some(JsonEnvelope {
                        schema: "nemo.relay.LlmRequestInterceptOutcome@2".into(),
                        json: serde_json::to_vec(&json!({
                            "request": valid_llm_request(),
                            "annotated_request": 3,
                            "pending_marks": [],
                        }))
                        .unwrap(),
                    }),
                })),
            },
            "llm_intercept_error" => InvokeResponse {
                result: Some(InvokeResult::Error(worker_error.clone())),
            },
            "tool_intercept_missing_result" => InvokeResponse {
                result: Some(InvokeResult::ToolExecution(ToolExecutionInterceptResult {
                    outcome: Some(ProtoToolExecutionInterceptOutcome {
                        result: None,
                        annotation: None,
                        pending_marks: None,
                    }),
                })),
            },
            "tool_intercept_invalid_result" => InvokeResponse {
                result: Some(InvokeResult::ToolExecution(ToolExecutionInterceptResult {
                    outcome: Some(ProtoToolExecutionInterceptOutcome {
                        result: Some(JsonValue {
                            json: b"{".to_vec(),
                        }),
                        annotation: None,
                        pending_marks: None,
                    }),
                })),
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
        .invoke_event_sanitize(
            "event_fields_invalid",
            RegistrationSurface::MarkSanitizeGuardrail,
            &event,
        )
        .await
        .expect_err("invalid event sanitizer fields should fail");
    assert!(
        error
            .to_string()
            .contains("worker returned invalid event sanitize fields")
    );

    let error = callback
        .invoke_llm_sanitize_request(
            "llm_json_invalid",
            valid_llm_request(),
            LlmSanitizeRequestContext::default(),
        )
        .await
        .expect_err("invalid LLM JSON result should fail");
    assert!(error.to_string().contains("invalid type"));

    let error = callback
        .invoke_llm_request_intercept(
            "llm_intercept_invalid_request",
            "model",
            valid_llm_request(),
            None,
        )
        .await
        .expect_err("invalid LLM intercept request should fail");
    assert!(
        error
            .to_string()
            .contains("invalid LLM request intercept outcome")
    );

    let error = callback
        .invoke_llm_request_intercept(
            "llm_intercept_missing_annotated",
            "model",
            valid_llm_request(),
            None,
        )
        .await
        .expect_err("legacy outcome schema should fail");
    assert!(
        error
            .to_string()
            .contains("unsupported LLM request intercept outcome schema")
    );

    let error = callback
        .invoke_llm_request_intercept(
            "llm_intercept_invalid_annotated",
            "model",
            valid_llm_request(),
            None,
        )
        .await
        .expect_err("invalid annotated request should fail");
    assert!(
        error
            .to_string()
            .contains("invalid LLM request intercept outcome")
    );

    let error = callback
        .invoke_llm_request_intercept("llm_intercept_error", "model", valid_llm_request(), None)
        .await
        .expect_err("LLM intercept worker error should surface");
    assert!(error.to_string().contains("worker.failed: boom"));

    let error = callback
        .invoke_tool_execution(
            "tool_intercept_missing_result",
            "lookup",
            json!({}),
            Arc::new(|args| Box::pin(async move { Ok(ToolExecutionResult::new(args)) })),
        )
        .await
        .expect_err("tool outcome without its required result should fail");
    assert!(
        error
            .to_string()
            .contains("tool execution intercept outcome result is missing")
    );

    let error = callback
        .invoke_tool_execution(
            "tool_intercept_invalid_result",
            "lookup",
            json!({}),
            Arc::new(|args| Box::pin(async move { Ok(ToolExecutionResult::new(args)) })),
        )
        .await
        .expect_err("tool outcome with invalid result JSON should fail");
    assert!(
        error
            .to_string()
            .contains("invalid worker tool execution result JSON")
    );

    let error = callback
        .invoke_llm_request_intercept(
            "llm_intercept_unexpected",
            "model",
            valid_llm_request(),
            None,
        )
        .await
        .expect_err("unexpected LLM intercept result should fail");
    assert!(
        error
            .to_string()
            .contains("LLM request intercept returned unexpected")
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn llm_worker_sanitizers_forward_codec_context_and_omission() {
    enable_operational_logs();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let (callback, _shutdown) = fake_callback_service({
        let seen = seen.clone();
        move |request| {
            let Some(nemo_relay_worker_proto::v1::invoke_request::Payload::Llm(invocation)) =
                request.payload
            else {
                panic!("LLM sanitizer must receive an LLM invocation");
            };
            let codec = match invocation.sanitize_context.as_ref() {
                Some(nemo_relay_worker_proto::v1::llm_invocation::SanitizeContext::RequestSanitizeContext(context)) => context.codec.as_ref(),
                Some(nemo_relay_worker_proto::v1::llm_invocation::SanitizeContext::ResponseSanitizeContext(context)) => context.codec.as_ref(),
                None => None,
            };
            seen.lock().unwrap().push((
                request.registration_name,
                codec
                    .map(|codec| codec.kind)
                    .unwrap_or(LlmCodecKind::Unspecified as i32),
                codec.and_then(|codec| codec.id.clone()),
                invocation.request.is_some(),
                invocation.response.is_some(),
            ));
            InvokeResponse {
                result: Some(InvokeResult::Empty(EmptyResult {})),
            }
        }
    })
    .await;

    let identities = [
        LlmCodecIdentity::None,
        LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OpenAiResponses),
        LlmCodecIdentity::Runtime("com.example.chat.v1".into()),
        LlmCodecIdentity::Opaque,
    ];
    for identity in identities {
        assert!(
            callback
                .invoke_llm_sanitize_request(
                    "request",
                    valid_llm_request(),
                    LlmSanitizeRequestContext::with_identity(identity.clone()),
                )
                .await
                .expect("empty worker result must represent request omission")
                .is_none()
        );
        assert!(
            callback
                .invoke_llm_sanitize_response(
                    "response",
                    json!({"secret": "value"}),
                    LlmSanitizeResponseContext::with_identity(identity),
                )
                .await
                .expect("empty worker result must represent response omission")
                .is_none()
        );
    }

    let seen = seen.lock().unwrap().clone();
    assert_eq!(
        seen,
        [
            (
                "request".into(),
                LlmCodecKind::Unspecified as i32,
                None,
                true,
                false,
            ),
            (
                "response".into(),
                LlmCodecKind::Unspecified as i32,
                None,
                false,
                true,
            ),
            (
                "request".into(),
                LlmCodecKind::Builtin as i32,
                Some("openai_responses".into()),
                true,
                false,
            ),
            (
                "response".into(),
                LlmCodecKind::Builtin as i32,
                Some("openai_responses".into()),
                false,
                true,
            ),
            (
                "request".into(),
                LlmCodecKind::Runtime as i32,
                Some("com.example.chat.v1".into()),
                true,
                false,
            ),
            (
                "response".into(),
                LlmCodecKind::Runtime as i32,
                Some("com.example.chat.v1".into()),
                false,
                true,
            ),
            (
                "request".into(),
                LlmCodecKind::Opaque as i32,
                None,
                true,
                false,
            ),
            (
                "response".into(),
                LlmCodecKind::Opaque as i32,
                None,
                false,
                true,
            ),
        ]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn llm_worker_codec_capabilities_are_active_only_during_sanitizer_invocation() {
    enable_operational_logs();
    type SeenCapability = (String, String, String);

    let host_state = Arc::new(Mutex::new(None::<Arc<WorkerHostRuntimeState>>));
    let seen = Arc::new(Mutex::new(Vec::<SeenCapability>::new()));
    let (callback, _shutdown) = fake_callback_service({
        let host_state = host_state.clone();
        let seen = seen.clone();
        move |request| {
            let invocation_id = request.invocation_id.clone();
            let registration_name = request.registration_name.clone();
            let Some(nemo_relay_worker_proto::v1::invoke_request::Payload::Llm(invocation)) =
                request.payload
            else {
                panic!("LLM sanitizer must receive an LLM invocation");
            };
            let state = host_state
                .lock()
                .unwrap()
                .clone()
                .expect("host state must be installed before invoking the worker");
            let capability_id = match invocation
                .sanitize_context
                .as_ref()
                .expect("codec context must be forwarded")
            {
                nemo_relay_worker_proto::v1::llm_invocation::SanitizeContext::RequestSanitizeContext(
                    context,
                ) => {
                    let id = context
                        .codec_capability_id
                        .as_deref()
                        .expect("active request codec must receive a capability");
                    let codec = state
                        .request_codec(id, &invocation_id)
                        .expect("request capability must resolve during invocation");
                    let request: LlmRequest = decode_json_envelope(
                        invocation.request.as_ref().expect("request payload"),
                    )
                    .expect("request payload must decode");
                    let annotated = codec
                        .decode(&request)
                        .expect("resolved request codec must be usable");
                    assert_eq!(annotated.model.as_deref(), Some("gpt-test"));
                    id.to_owned()
                }
                nemo_relay_worker_proto::v1::llm_invocation::SanitizeContext::ResponseSanitizeContext(
                    context,
                ) => {
                    let id = context
                        .codec_capability_id
                        .as_deref()
                        .expect("active response codec must receive a capability");
                    let codec = state
                        .response_codec(id, &invocation_id)
                        .expect("response capability must resolve during invocation");
                    let response: Json = decode_json_envelope(
                        invocation.response.as_ref().expect("response payload"),
                    )
                    .expect("response payload must decode");
                    let annotated = codec
                        .decode_response(&response)
                        .expect("resolved response codec must be usable");
                    assert_eq!(annotated.model.as_deref(), Some("gpt-test"));
                    id.to_owned()
                }
            };
            seen.lock().unwrap().push((
                registration_name.clone(),
                capability_id,
                invocation_id,
            ));
            if registration_name == "response-error" {
                InvokeResponse {
                    result: Some(InvokeResult::Error(
                        nemo_relay_worker_proto::v1::WorkerError {
                            code: "worker.failed".into(),
                            message: "boom".into(),
                            retryable: false,
                        },
                    )),
                }
            } else {
                InvokeResponse {
                    result: Some(InvokeResult::Empty(EmptyResult {})),
                }
            }
        }
    })
    .await;
    *host_state.lock().unwrap() = Some(callback.host_state.clone());

    let codec = Arc::new(OpenAIChatCodec);
    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "secret"}]
        }),
    };
    assert!(
        callback
            .invoke_llm_sanitize_request(
                "request-success",
                request,
                LlmSanitizeRequestContext::for_request_codec(Some(codec.clone())),
            )
            .await
            .expect("request sanitizer must succeed")
            .is_none()
    );

    let response = json!({
        "id": "chatcmpl-test",
        "model": "gpt-test",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "secret"},
            "finish_reason": "stop"
        }]
    });
    let error = callback
        .invoke_llm_sanitize_response(
            "response-error",
            response,
            LlmSanitizeResponseContext::for_response_codec(Some(codec)),
        )
        .await
        .expect_err("worker sanitizer error must surface");
    assert!(error.to_string().contains("worker.failed: boom"));

    let seen = seen.lock().unwrap().clone();
    assert_eq!(seen.len(), 2);
    let (_, request_capability, request_invocation) = &seen[0];
    assert_eq!(
        callback
            .host_state
            .request_codec(request_capability, request_invocation)
            .err()
            .expect("successful invocation must expire its request capability")
            .code(),
        tonic::Code::NotFound
    );
    let (_, response_capability, response_invocation) = &seen[1];
    assert_eq!(
        callback
            .host_state
            .response_codec(response_capability, response_invocation)
            .err()
            .expect("failed invocation must expire its response capability")
            .code(),
        tonic::Code::NotFound
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn cancelling_worker_sanitizer_expires_codec_capability() {
    enable_operational_logs();
    let (started_tx, started_rx) = oneshot::channel();
    let started_tx = Arc::new(Mutex::new(Some(started_tx)));
    let (callback, _shutdown, _cancel_rx) = fake_callback_service_with_handlers(
        {
            let started_tx = Arc::clone(&started_tx);
            move |request| {
                let started_tx = Arc::clone(&started_tx);
                Box::pin(async move {
                    let invocation_id = request.invocation_id;
                    let Some(invoke_request_payload::Payload::Llm(invocation)) = request.payload
                    else {
                        panic!("LLM sanitizer must receive an LLM invocation");
                    };
                    let Some(llm_invocation::SanitizeContext::RequestSanitizeContext(context)) =
                        invocation.sanitize_context
                    else {
                        panic!("request sanitizer context must be present");
                    };
                    let capability_id = context
                        .codec_capability_id
                        .expect("request codec capability must be present");
                    if let Some(started) = started_tx.lock().expect("started lock").take() {
                        let _ = started.send((capability_id, invocation_id));
                    }
                    std::future::pending::<InvokeResponse>().await
                })
            }
        },
        |_| Box::pin(tokio_stream::empty()),
    )
    .await;
    let callback_task = callback.clone();
    let task = tokio::spawn(async move {
        callback_task
            .invoke_llm_sanitize_request(
                "cancel-codec",
                valid_llm_request(),
                LlmSanitizeRequestContext::for_request_codec(Some(Arc::new(OpenAIChatCodec))),
            )
            .await
    });
    let (capability_id, invocation_id) =
        tokio::time::timeout(std::time::Duration::from_secs(1), started_rx)
            .await
            .expect("worker sanitizer should start")
            .expect("worker sanitizer should publish its capability");
    callback
        .host_state
        .request_codec(&capability_id, &invocation_id)
        .expect("capability must be active while the sanitizer is pending");

    task.abort();
    let _ = task.await;
    let error = match callback
        .host_state
        .request_codec(&capability_id, &invocation_id)
    {
        Ok(_) => panic!("cancelled sanitizer must expire its codec capability"),
        Err(error) => error,
    };
    assert_eq!(error.code(), tonic::Code::NotFound);
}

#[tokio::test(flavor = "multi_thread")]
async fn callback_stream_transport_error_surfaces_to_host_stream() {
    enable_operational_logs();
    let (callback, _shutdown) = fake_callback_service(|_| InvokeResponse {
        result: Some(InvokeResult::Empty(EmptyResult {})),
    })
    .await;

    let mut stream = callback
        .invoke_llm_stream_execution(
            "stream_transport_error",
            "model",
            valid_llm_request(),
            Arc::new(|_request| Box::pin(async { Ok(LlmJsonStream::new(tokio_stream::empty())) })),
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
    enable_operational_logs();
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

    let mut stream = callback
        .invoke_llm_stream_execution(
            "stream_receiver_drop",
            "model",
            valid_llm_request(),
            Arc::new(|_request| Box::pin(async { Ok(LlmJsonStream::new(tokio_stream::empty())) })),
        )
        .await
        .expect("host stream should be returned");
    yield_tx
        .send(())
        .expect("worker stream yield signal should be delivered");
    tokio::time::timeout(std::time::Duration::from_secs(1), stream.next())
        .await
        .expect("worker stream should yield before timing out")
        .expect("worker stream ended before yielding a chunk")
        .expect("worker stream chunk should be valid");
    drop(stream);
    tokio::time::timeout(std::time::Duration::from_secs(1), stream_dropped_rx)
        .await
        .expect("worker stream should be dropped after host receiver is dropped")
        .expect("worker stream drop signal should be delivered");
}

#[tokio::test(flavor = "multi_thread")]
async fn callback_timeout_sends_explicit_worker_cancellation() {
    enable_operational_logs();
    let (started_tx, started_rx) = oneshot::channel();
    let started_tx = Arc::new(Mutex::new(Some(started_tx)));
    let (callback, _shutdown, mut cancel_rx) = fake_callback_service_with_handlers(
        {
            let started_tx = started_tx.clone();
            move |_| {
                let started_tx = started_tx.clone();
                Box::pin(async move {
                    if let Some(started) = started_tx.lock().expect("started lock").take() {
                        let _ = started.send(());
                    }
                    std::future::pending::<InvokeResponse>().await
                })
            }
        },
        |_| Box::pin(tokio_stream::empty()),
    )
    .await;
    let request = callback.base_request(
        "timeout",
        RegistrationSurface::ToolRequestIntercept,
        None,
        Some(invoke_request_payload_tool("tool", json!({}))),
    );
    let invocation_id = request.invocation_id.clone();

    let callback_task = callback.clone();
    let task = tokio::spawn(async move {
        callback_task
            .invoke_async_with_timeout(request, std::time::Duration::from_millis(10))
            .await
    });
    tokio::time::timeout(std::time::Duration::from_secs(1), started_rx)
        .await
        .expect("worker invocation should start before timeout assertion")
        .expect("worker invocation should start");
    let result = tokio::time::timeout(std::time::Duration::from_secs(1), task)
        .await
        .expect("timed out invocation should complete")
        .expect("timed out invocation task should join");
    let cancellation = tokio::time::timeout(std::time::Duration::from_secs(1), cancel_rx.recv())
        .await
        .expect("host should send cancellation after timeout")
        .expect("cancellation channel should remain open");

    assert!(
        result
            .expect_err("worker invocation should time out")
            .to_string()
            .contains("worker invocation timed out")
    );
    assert_eq!(cancellation.invocation_id, invocation_id);
    assert!(cancellation.reason.contains("timed out"));
}

#[tokio::test(flavor = "multi_thread")]
async fn dropping_callback_future_cancels_worker_and_cleans_host_state() {
    enable_operational_logs();
    let (started_tx, started_rx) = oneshot::channel();
    let started_tx = Arc::new(Mutex::new(Some(started_tx)));
    let (callback, _shutdown, mut cancel_rx) = fake_callback_service_with_handlers(
        {
            let started_tx = started_tx.clone();
            move |_| {
                let started_tx = started_tx.clone();
                Box::pin(async move {
                    if let Some(started) = started_tx.lock().expect("started lock").take() {
                        let _ = started.send(());
                    }
                    std::future::pending::<InvokeResponse>().await
                })
            }
        },
        |_| Box::pin(tokio_stream::empty()),
    )
    .await;
    let continuation_id = callback
        .host_state
        .insert_continuation(Continuation::tool(Arc::new(|value| {
            Box::pin(async move { Ok(ToolExecutionResult::new(value)) })
        })))
        .expect("continuation should insert");
    let request = callback.base_request(
        "cancel",
        RegistrationSurface::ToolExecutionIntercept,
        Some(continuation_id),
        Some(invoke_request_payload_tool("tool", json!({}))),
    );
    let scope_stack_id = request
        .scope
        .as_ref()
        .expect("worker invocation should have a scope stack")
        .scope_stack_id
        .clone();
    let invocation_stack = callback
        .host_state
        .stack(&scope_stack_id)
        .expect("invocation scope stack lookup should succeed")
        .expect("invocation scope stack should exist");
    let baseline_depth = invocation_stack
        .read()
        .expect("invocation scope stack lock")
        .scopes()
        .len();
    let host_runtime = WorkerHostRuntimeService {
        state: callback.host_state.clone(),
    };
    for name in ["cancelled-outer", "cancelled-inner"] {
        let pushed = host_runtime
            .push_scope(Request::new(PushScopeRequest {
                activation_id: ACTIVATION_ID.into(),
                auth_token: AUTH_TOKEN.into(),
                scope: Some(ScopeContext {
                    scope_stack_id: scope_stack_id.clone(),
                    parent_scope_id: String::new(),
                }),
                name: name.into(),
                scope_type: ProtoScopeType::Custom as i32,
                data: None,
                metadata: None,
                input: None,
            }))
            .await
            .expect("worker scope should push")
            .into_inner();
        assert!(pushed.error.is_none());
    }
    assert_eq!(
        invocation_stack
            .read()
            .expect("invocation scope stack lock")
            .scopes()
            .len(),
        baseline_depth + 2
    );
    let overlapping_scope_stack_id = callback
        .host_state
        .insert_invocation_scope_stack(invocation_stack.clone(), None);
    let invocation_id = request.invocation_id.clone();
    let callback_task = callback.clone();
    let task = tokio::spawn(async move { callback_task.invoke_async(request).await });
    tokio::time::timeout(std::time::Duration::from_secs(1), started_rx)
        .await
        .expect("worker invocation should start before caller abort")
        .expect("worker invocation should start");

    task.abort();
    let _ = task.await;
    let cancellation = tokio::time::timeout(std::time::Duration::from_secs(1), cancel_rx.recv())
        .await
        .expect("host should send cancellation when caller drops")
        .expect("cancellation channel should remain open");

    assert_eq!(cancellation.invocation_id, invocation_id);
    assert!(cancellation.reason.contains("caller cancelled"));
    assert!(
        callback
            .host_state
            .continuations
            .lock()
            .expect("continuation lock")
            .is_empty()
    );
    assert!(
        callback
            .host_state
            .scope_handles
            .lock()
            .expect("scope handle lock")
            .is_empty()
    );
    assert_eq!(
        callback
            .host_state
            .scope_stacks
            .lock()
            .expect("scope lock")
            .len(),
        1
    );
    assert_eq!(
        invocation_stack
            .read()
            .expect("invocation scope stack lock")
            .scopes()
            .len(),
        baseline_depth + 2
    );
    callback
        .host_state
        .cleanup_invocation_scope_stack(&overlapping_scope_stack_id);
    assert!(
        callback
            .host_state
            .scope_stacks
            .lock()
            .expect("scope lock")
            .is_empty()
    );
    assert!(
        callback
            .host_state
            .pending_scope_cleanups
            .lock()
            .expect("pending cleanup lock")
            .is_empty()
    );
    assert_eq!(
        invocation_stack
            .read()
            .expect("invocation scope stack lock")
            .scopes()
            .len(),
        baseline_depth
    );
    callback
        .host_state
        .cleanup_invocation_scope_stack(&scope_stack_id);
    callback
        .host_state
        .cleanup_invocation_scope_stack(&overlapping_scope_stack_id);
    assert_eq!(
        invocation_stack
            .read()
            .expect("invocation scope stack lock")
            .scopes()
            .len(),
        baseline_depth
    );
}

#[test]
fn invocation_cleanup_releases_host_state_locks_before_unwinding() {
    enable_operational_logs();
    let state = Arc::new(WorkerHostRuntimeState::new(
        ACTIVATION_ID.into(),
        AUTH_TOKEN.into(),
    ));
    let stack = crate::api::runtime::create_scope_stack();
    let baseline_depth = stack.read().expect("scope stack lock").scopes().len();
    let scope_stack_id = state.insert_invocation_scope_stack(stack.clone(), None);
    with_scope_stack(stack.clone(), || {
        push_scope(
            PushScopeParams::builder()
                .name("cleanup-lock-test")
                .scope_type(ScopeType::Custom)
                .build(),
        )
    })
    .expect("worker scope should push");

    let stack_guard = stack.write().expect("scope stack lock");
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    let cleanup_state = state.clone();
    let cleanup = std::thread::spawn(move || {
        cleanup_state.cleanup_invocation_scope_stack(&scope_stack_id);
        let _ = done_tx.send(());
    });
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
    loop {
        let cleanup_registered = state
            .scope_stack_cleanups
            .lock()
            .expect("scope cleanup lock")
            .iter()
            .any(|handle| Arc::ptr_eq(handle, &stack));
        if cleanup_registered
            && state.scope_stacks.try_lock().is_ok()
            && state.pending_scope_cleanups.try_lock().is_ok()
        {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "scope cleanup should release host state locks before unwinding"
        );
        std::thread::yield_now();
    }

    drop(stack_guard);
    done_rx
        .recv_timeout(std::time::Duration::from_secs(1))
        .expect("scope cleanup thread should finish before timing out");
    cleanup.join().expect("scope cleanup thread should finish");

    assert!(
        state
            .scope_stack_cleanups
            .lock()
            .expect("scope cleanup lock")
            .is_empty()
    );
    assert_eq!(
        stack.read().expect("scope stack lock").scopes().len(),
        baseline_depth
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn dropping_host_stream_sends_explicit_worker_cancellation() {
    enable_operational_logs();
    let (yield_tx, yield_rx) = oneshot::channel();
    let yield_rx = Arc::new(Mutex::new(Some(yield_rx)));
    let (callback, _shutdown, mut cancel_rx) = fake_callback_service_with_handlers(
        |_| {
            Box::pin(async {
                InvokeResponse {
                    result: Some(InvokeResult::Empty(EmptyResult {})),
                }
            })
        },
        {
            let yield_rx = yield_rx.clone();
            move |_| {
                let yield_rx = yield_rx
                    .lock()
                    .expect("yield lock")
                    .take()
                    .expect("stream should be created once");
                Box::pin(SignalChunkThenPendingStream {
                    yield_rx,
                    dropped: None,
                    yielded: false,
                })
            }
        },
    )
    .await;
    let mut stream = callback
        .invoke_llm_stream_execution(
            "cancel_stream",
            "model",
            valid_llm_request(),
            Arc::new(|_request| Box::pin(async { Ok(LlmJsonStream::new(tokio_stream::empty())) })),
        )
        .await
        .expect("host stream should be returned");
    yield_tx
        .send(())
        .expect("worker stream yield signal should be delivered");
    tokio::time::timeout(std::time::Duration::from_secs(1), stream.next())
        .await
        .expect("worker stream should yield before abandonment")
        .expect("worker stream should yield before abandonment")
        .expect("worker stream chunk should be valid");
    drop(stream);

    let cancellation = tokio::time::timeout(std::time::Duration::from_secs(1), cancel_rx.recv())
        .await
        .expect("host should cancel abandoned stream")
        .expect("cancellation channel should remain open");
    assert!(cancellation.reason.contains("stopped consuming"));
}

#[tokio::test(flavor = "multi_thread")]
#[allow(clippy::await_holding_lock)] // The process-wide test mutex serializes global registrations.
async fn install_registrations_covers_registry_error_edges() {
    let _runtime_guard = crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    enable_operational_logs();
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
        let duplicate_name = format!("duplicate_worker_{surface:?}");
        let (instance, _shutdown) = fake_worker_instance(vec![
            registration(surface, &duplicate_name),
            registration(surface, &duplicate_name),
        ])
        .await;
        let mut ctx = PluginRegistrationContext::new();
        let error = match instance.install_registrations(&mut ctx) {
            Err(error) => error,
            Ok(()) => panic!("{surface:?}: duplicate worker registration should fail"),
        };
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
#[allow(clippy::await_holding_lock)] // The process-wide test mutex intentionally serializes runtime state.
async fn installed_callbacks_apply_surface_specific_fallbacks() {
    struct RuntimeCleanup {
        registrations: Option<PluginRegistrationContext>,
    }

    impl Drop for RuntimeCleanup {
        fn drop(&mut self) {
            if let Some(context) = self.registrations.take() {
                let mut registrations = context.into_registrations();
                crate::plugin::rollback_registrations(&mut registrations);
            }
            let context = crate::api::runtime::global_context();
            *context.write().unwrap_or_else(|error| error.into_inner()) =
                NemoRelayContextState::new();
            crate::shared_runtime::reset_runtime_owner_for_tests();
        }
    }

    enable_operational_logs();
    let registrations = vec![
        registration(RegistrationSurface::Subscriber, "fallback_subscriber"),
        registration(RegistrationSurface::MarkSanitizeGuardrail, "fallback_mark"),
        registration(
            RegistrationSurface::ScopeSanitizeStartGuardrail,
            "fallback_scope_start",
        ),
        registration(
            RegistrationSurface::ScopeSanitizeEndGuardrail,
            "fallback_scope_end",
        ),
        registration(
            RegistrationSurface::ToolSanitizeRequestGuardrail,
            "fallback_tool_request",
        ),
        registration(
            RegistrationSurface::ToolSanitizeResponseGuardrail,
            "fallback_tool_response",
        ),
        registration(
            RegistrationSurface::LlmSanitizeRequestGuardrail,
            "fallback_llm_request",
        ),
        registration(
            RegistrationSurface::LlmSanitizeResponseGuardrail,
            "fallback_llm_response",
        ),
    ];
    let (mut instance, _instance_shutdown) = fake_worker_instance(registrations).await;
    let (error_client, _error_shutdown) = fake_worker_client(|_| InvokeResponse {
        result: Some(InvokeResult::Error(WorkerError {
            code: "worker.failed".into(),
            message: "callback unavailable".into(),
            retryable: false,
        })),
    })
    .await;
    instance.client = error_client;

    let _runtime_guard = crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    crate::shared_runtime::reset_runtime_owner_for_tests();
    let context = crate::api::runtime::global_context();
    *context.write().unwrap() = NemoRelayContextState::new();

    let mut registration_context = PluginRegistrationContext::new();
    instance
        .install_registrations(&mut registration_context)
        .expect("worker registrations should install");
    let _cleanup = RuntimeCleanup {
        registrations: Some(registration_context),
    };

    let event = Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .name("fallback-event")
            .data(json!({"preserved": true}))
            .metadata(json!({"preserved": true}))
            .build(),
        None,
        None,
    ));
    let tool_request = json!({"request": "preserved"});
    let tool_response = json!({"response": "preserved"});
    let llm_request = valid_llm_request();
    let llm_response = json!({"response": "preserved"});

    let (
        subscribers,
        mark_entries,
        scope_start_entries,
        scope_end_entries,
        tool_request_entries,
        tool_response_entries,
        llm_request_entries,
        llm_response_entries,
    ) = {
        let state = context.read().unwrap();
        (
            state.collect_event_subscribers(&[]),
            NemoRelayContextState::event_sanitize_entries(
                &state.mark_sanitize_guardrails,
                &[],
                crate::api::registry::RuntimeRegistrationKind::MarkSanitizeGuardrail,
            ),
            NemoRelayContextState::event_sanitize_entries(
                &state.scope_sanitize_start_guardrails,
                &[],
                crate::api::registry::RuntimeRegistrationKind::ScopeSanitizeStartGuardrail,
            ),
            NemoRelayContextState::event_sanitize_entries(
                &state.scope_sanitize_end_guardrails,
                &[],
                crate::api::registry::RuntimeRegistrationKind::ScopeSanitizeEndGuardrail,
            ),
            state.tool_sanitize_request_entries(&[]),
            state.tool_sanitize_response_entries(&[]),
            state.llm_sanitize_request_entries(&[]),
            state.llm_sanitize_response_entries(&[]),
        )
    };
    NemoRelayContextState::emit_event(&event, &subscribers);

    for entries in [mark_entries, scope_start_entries, scope_end_entries] {
        let sanitized =
            NemoRelayContextState::event_sanitize_snapshot_chain(event.clone(), &entries).await;
        assert_eq!(sanitized.data(), None);
        assert_eq!(sanitized.metadata(), None);
    }

    assert_eq!(
        NemoRelayContextState::tool_sanitize_request_snapshot_chain(
            "tool",
            tool_request.clone(),
            &tool_request_entries,
        )
        .await,
        None
    );
    assert_eq!(
        NemoRelayContextState::tool_sanitize_response_snapshot_chain(
            "tool",
            tool_response.clone(),
            &tool_response_entries,
        )
        .await,
        None
    );
    assert_eq!(
        NemoRelayContextState::llm_sanitize_request_snapshot_chain(
            llm_request.clone(),
            crate::api::runtime::LlmSanitizeRequestContext::default(),
            &llm_request_entries,
        )
        .await,
        None,
    );
    assert_eq!(
        NemoRelayContextState::llm_sanitize_response_snapshot_chain(
            llm_response.clone(),
            crate::api::runtime::LlmSanitizeResponseContext::default(),
            &llm_response_entries,
        )
        .await,
        None,
    );
    crate::api::subscriber::flush_subscribers().expect("subscriber callback should flush");
}

#[tokio::test(flavor = "multi_thread")]
async fn adapter_register_rejects_config_drift_even_without_validation_call() {
    enable_operational_logs();
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
    enable_operational_logs();
    let state = Arc::new(WorkerHostRuntimeState::new(
        ACTIVATION_ID.into(),
        AUTH_TOKEN.into(),
    ));
    let service = WorkerHostRuntimeService {
        state: state.clone(),
    };

    let diagnostics_auth_error = service
        .get_runtime_diagnostics(Request::new(GetRuntimeDiagnosticsRequest {
            activation_id: "wrong".into(),
            auth_token: AUTH_TOKEN.into(),
        }))
        .await
        .expect_err("bad activation id should fail diagnostics auth");
    assert_eq!(diagnostics_auth_error.code(), tonic::Code::PermissionDenied);
    let diagnostics = service
        .get_runtime_diagnostics(Request::new(GetRuntimeDiagnosticsRequest {
            activation_id: ACTIVATION_ID.into(),
            auth_token: AUTH_TOKEN.into(),
        }))
        .await
        .expect("diagnostics request should succeed")
        .into_inner();
    assert!(diagnostics.entries.len() <= 32);

    let auth_error = service
        .emit_mark(Request::new(EmitMarkRequest {
            activation_id: "wrong".into(),
            auth_token: AUTH_TOKEN.into(),
            name: "auth-failure".into(),
            scope: None,
            data: None,
            metadata: None,
            data_schema: None,
            severity: String::new(),
            category: String::new(),
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
            data_schema: None,
            severity: String::new(),
            category: String::new(),
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
            data_schema: None,
            severity: String::new(),
            category: String::new(),
        }))
        .await
        .expect("no-scope mark should succeed")
        .into_inner();
    assert!(ack.ok);

    let ack = service
        .emit_mark(Request::new(EmitMarkRequest {
            activation_id: ACTIVATION_ID.into(),
            auth_token: AUTH_TOKEN.into(),
            name: "telemetry-options".into(),
            scope: None,
            data: Some(json_envelope("nemo.relay.Json@1", &json!({"measurements": []})).unwrap()),
            metadata: None,
            data_schema: Some(
                json_envelope(
                    "nemo.relay.DataSchema@1",
                    &json!({
                        "name": "nemo.relay.metric_measurements",
                        "version": "1"
                    }),
                )
                .unwrap(),
            ),
            severity: "warning".into(),
            category: "custom".into(),
        }))
        .await
        .expect("typed mark options should return host ack")
        .into_inner();
    assert!(ack.ok, "{:?}", ack.error);

    for request in [
        EmitMarkRequest {
            activation_id: ACTIVATION_ID.into(),
            auth_token: AUTH_TOKEN.into(),
            name: "bad-schema".into(),
            data_schema: Some(
                json_envelope(
                    "nemo.relay.DataSchema@1",
                    &json!({"name": 7, "version": "1"}),
                )
                .unwrap(),
            ),
            ..EmitMarkRequest::default()
        },
        EmitMarkRequest {
            activation_id: ACTIVATION_ID.into(),
            auth_token: AUTH_TOKEN.into(),
            name: "bad-severity".into(),
            severity: "fatal".into(),
            ..EmitMarkRequest::default()
        },
    ] {
        let ack = service
            .emit_mark(Request::new(request))
            .await
            .expect("invalid mark options should return host ack")
            .into_inner();
        assert!(!ack.ok);
        assert!(
            ack.error
                .expect("invalid mark error")
                .message
                .contains("invalid argument")
        );
    }

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
async fn host_runtime_codec_capabilities_are_directional_authorized_and_ephemeral() {
    let state = Arc::new(WorkerHostRuntimeState::new(
        ACTIVATION_ID.into(),
        AUTH_TOKEN.into(),
    ));
    let service = WorkerHostRuntimeService {
        state: state.clone(),
    };
    let codec = Arc::new(OpenAIChatCodec);
    let request_codec: Arc<dyn LlmCodec> = codec.clone();
    let response_codec: Arc<dyn LlmResponseCodec> = codec.clone();
    let invocation_id = "sanitize-invocation";
    let request_capability = state.insert_request_codec(invocation_id, request_codec);
    let response_capability = state.insert_response_codec(invocation_id, response_codec);
    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "secret"}],
            "preserve": true
        }),
    };

    let unauthorized = service
        .decode_llm_codec_request(Request::new(LlmCodecDecodeRequest {
            activation_id: ACTIVATION_ID.into(),
            auth_token: "wrong".into(),
            codec_capability_id: request_capability.clone(),
            invocation_id: invocation_id.into(),
            request: Some(json_envelope("nemo.relay.LlmRequest@1", &request).unwrap()),
        }))
        .await
        .expect_err("wrong activation credentials must be rejected");
    assert_eq!(unauthorized.code(), tonic::Code::PermissionDenied);

    let forged = service
        .decode_llm_codec_request(Request::new(LlmCodecDecodeRequest {
            activation_id: ACTIVATION_ID.into(),
            auth_token: AUTH_TOKEN.into(),
            codec_capability_id: "codec-forged".into(),
            invocation_id: invocation_id.into(),
            request: Some(json_envelope("nemo.relay.LlmRequest@1", &request).unwrap()),
        }))
        .await
        .expect_err("forged capability must be rejected");
    assert_eq!(forged.code(), tonic::Code::NotFound);

    let wrong_direction = service
        .decode_llm_codec_request(Request::new(LlmCodecDecodeRequest {
            activation_id: ACTIVATION_ID.into(),
            auth_token: AUTH_TOKEN.into(),
            codec_capability_id: response_capability.clone(),
            invocation_id: invocation_id.into(),
            request: Some(json_envelope("nemo.relay.LlmRequest@1", &request).unwrap()),
        }))
        .await
        .expect_err("response capability cannot decode requests");
    assert_eq!(wrong_direction.code(), tonic::Code::InvalidArgument);

    let wrong_invocation = service
        .decode_llm_codec_request(Request::new(LlmCodecDecodeRequest {
            activation_id: ACTIVATION_ID.into(),
            auth_token: AUTH_TOKEN.into(),
            codec_capability_id: request_capability.clone(),
            invocation_id: "another-invocation".into(),
            request: Some(json_envelope("nemo.relay.LlmRequest@1", &request).unwrap()),
        }))
        .await
        .expect_err("a capability cannot be reused by another invocation");
    assert_eq!(wrong_invocation.code(), tonic::Code::PermissionDenied);

    let wrong_invocation = state
        .response_codec(&response_capability, "another-invocation")
        .err()
        .expect("response capability must remain invocation-bound");
    assert_eq!(wrong_invocation.code(), tonic::Code::PermissionDenied);

    let wrong_direction = state
        .response_codec(&request_capability, invocation_id)
        .err()
        .expect("request capability cannot decode responses");
    assert_eq!(wrong_direction.code(), tonic::Code::InvalidArgument);

    let decoded = service
        .decode_llm_codec_request(Request::new(LlmCodecDecodeRequest {
            activation_id: ACTIVATION_ID.into(),
            auth_token: AUTH_TOKEN.into(),
            codec_capability_id: request_capability.clone(),
            invocation_id: invocation_id.into(),
            request: Some(json_envelope("nemo.relay.LlmRequest@1", &request).unwrap()),
        }))
        .await
        .expect("active request capability decodes")
        .into_inner();
    assert!(decoded.error.is_none());
    let decoded = decoded.value.expect("decoded request value");
    assert_eq!(decoded.schema, "nemo.relay.AnnotatedLlmRequest@2");
    let annotated: AnnotatedLlmRequest = decode_json_envelope(&decoded).unwrap();
    assert_eq!(annotated.model.as_deref(), Some("gpt-test"));

    let encoded = service
        .encode_llm_codec_request(Request::new(LlmCodecEncodeRequest {
            activation_id: ACTIVATION_ID.into(),
            auth_token: AUTH_TOKEN.into(),
            codec_capability_id: request_capability.clone(),
            invocation_id: invocation_id.into(),
            annotated_request: Some(
                json_envelope("nemo.relay.AnnotatedLlmRequest@2", &annotated).unwrap(),
            ),
            original_request: Some(json_envelope("nemo.relay.LlmRequest@1", &request).unwrap()),
        }))
        .await
        .expect("active request capability encodes")
        .into_inner();
    assert!(encoded.error.is_none());
    let encoded = encoded.value.expect("encoded request value");
    assert_eq!(encoded.schema, "nemo.relay.LlmRequest@1");
    let encoded: LlmRequest = decode_json_envelope(&encoded).unwrap();
    assert_eq!(encoded.content["preserve"], json!(true));

    let response = json!({
        "id": "chatcmpl-test",
        "model": "gpt-test",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "secret"},
            "finish_reason": "stop"
        }]
    });
    let decoded = service
        .decode_llm_codec_response(Request::new(LlmCodecDecodeResponse {
            activation_id: ACTIVATION_ID.into(),
            auth_token: AUTH_TOKEN.into(),
            codec_capability_id: response_capability.clone(),
            invocation_id: invocation_id.into(),
            response: Some(json_envelope(JSON_SCHEMA, &response).unwrap()),
        }))
        .await
        .expect("active response capability decodes")
        .into_inner();
    assert!(decoded.error.is_none());

    state.remove_codec(&request_capability);
    state.remove_codec(&response_capability);
    let expired = service
        .decode_llm_codec_request(Request::new(LlmCodecDecodeRequest {
            activation_id: ACTIVATION_ID.into(),
            auth_token: AUTH_TOKEN.into(),
            codec_capability_id: request_capability,
            invocation_id: invocation_id.into(),
            request: Some(json_envelope("nemo.relay.LlmRequest@1", &request).unwrap()),
        }))
        .await
        .expect_err("removed capability must expire");
    assert_eq!(expired.code(), tonic::Code::NotFound);
}

#[tokio::test]
async fn host_runtime_service_reports_poisoned_internal_locks() {
    enable_operational_logs();
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
    enable_operational_logs();
    drop(OwnedWorkerRuntime { runtime: None });
}

#[tokio::test]
async fn host_runtime_service_covers_continuation_errors_and_stream_items() {
    enable_operational_logs();
    let state = Arc::new(WorkerHostRuntimeState::new(
        ACTIVATION_ID.into(),
        AUTH_TOKEN.into(),
    ));
    let service = WorkerHostRuntimeService {
        state: state.clone(),
    };

    let llm_continuation = state
        .insert_continuation(Continuation::llm(Arc::new(|request| {
            Box::pin(async move { Ok(request.content) })
        })))
        .expect("llm continuation should insert");
    let wrong_type = service
        .tool_next(Request::new(ToolNextRequest {
            activation_id: ACTIVATION_ID.into(),
            auth_token: AUTH_TOKEN.into(),
            continuation_id: llm_continuation,
            value: Some(json_envelope(JSON_SCHEMA, &json!({})).expect("json envelope")),
            scope: None,
        }))
        .await
        .expect_err("wrong continuation type should fail");
    assert_eq!(wrong_type.code(), tonic::Code::InvalidArgument);

    let tool_continuation = state
        .insert_continuation(Continuation::tool(Arc::new(|value| {
            Box::pin(async move { Ok(ToolExecutionResult::new(value)) })
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
            scope: None,
        }))
        .await
        .expect_err("invalid tool next JSON should fail");
    assert_eq!(invalid_tool_json.code(), tonic::Code::InvalidArgument);

    let tool_continuation = state
        .insert_continuation(Continuation::tool(Arc::new(|_value| {
            Box::pin(async move {
                panic!("worker tool next panic");
            })
        })))
        .expect("panicking tool continuation should insert");
    let result = service
        .tool_next(Request::new(ToolNextRequest {
            activation_id: ACTIVATION_ID.into(),
            auth_token: AUTH_TOKEN.into(),
            continuation_id: tool_continuation,
            value: Some(json_envelope(JSON_SCHEMA, &json!({})).expect("json envelope")),
            scope: None,
        }))
        .await
        .expect("tool panic should become a structured result")
        .into_inner();
    assert!(
        result
            .error
            .is_some_and(|error| error.message.contains("worker tool next panic"))
    );

    let llm_continuation = state
        .insert_continuation(Continuation::llm(Arc::new(|request| {
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
            scope: None,
        }))
        .await
        .expect_err("invalid LLM next request should fail");
    assert_eq!(invalid_llm_json.code(), tonic::Code::InvalidArgument);

    let llm_continuation = state
        .insert_continuation(Continuation::llm(Arc::new(|_request| {
            Box::pin(async move {
                panic!("worker LLM next panic");
            })
        })))
        .expect("panicking LLM continuation should insert");
    let result = service
        .llm_next(Request::new(LlmNextRequest {
            activation_id: ACTIVATION_ID.into(),
            auth_token: AUTH_TOKEN.into(),
            continuation_id: llm_continuation,
            request: Some(
                json_envelope(LLM_REQUEST_SCHEMA, &valid_llm_request())
                    .expect("llm request envelope"),
            ),
            scope: None,
        }))
        .await
        .expect("LLM panic should become a structured result")
        .into_inner();
    assert!(
        result
            .error
            .is_some_and(|error| error.message.contains("worker LLM next panic"))
    );

    let stream_continuation = state
        .insert_continuation(Continuation::llm_stream(Arc::new(|_request| {
            Box::pin(async move {
                Ok(LlmJsonStream::new(tokio_stream::iter(vec![Err(
                    FlowError::Internal("stream item failed".into()),
                )])))
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
            scope: None,
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
        .insert_continuation(Continuation::llm_stream(Arc::new(|_request| {
            Box::pin(async move {
                Ok(LlmJsonStream::new(futures_util::stream::once(async move {
                    panic!("worker stream next panic");
                    #[allow(unreachable_code)]
                    Ok(json!({}))
                })))
            })
        })))
        .expect("panicking stream continuation should insert");
    let stream_response = service
        .llm_stream_next(Request::new(LlmStreamNextRequest {
            activation_id: ACTIVATION_ID.into(),
            auth_token: AUTH_TOKEN.into(),
            continuation_id: stream_continuation,
            request: Some(
                json_envelope(LLM_REQUEST_SCHEMA, &valid_llm_request())
                    .expect("llm request envelope"),
            ),
            scope: None,
        }))
        .await
        .expect("stream next should return a stream before polling");
    let mut stream = stream_response.into_inner();
    let chunk = stream
        .next()
        .await
        .expect("panicking stream should yield one error")
        .expect("panic should be translated into a stream item");
    match chunk.item {
        Some(StreamItem::Error(error)) => {
            assert!(error.message.contains("worker stream next panic"));
        }
        other => panic!("expected worker panic error, got {other:?}"),
    }

    let stream_continuation = state
        .insert_continuation(Continuation::llm_stream(Arc::new(|_request| {
            Box::pin(async move { Ok(LlmJsonStream::new(tokio_stream::empty())) })
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
            scope: None,
        }))
        .await
    {
        Ok(_) => panic!("invalid LLM stream request should fail"),
        Err(error) => error,
    };
    assert_eq!(invalid_stream_request.code(), tonic::Code::InvalidArgument);
}

#[tokio::test]
async fn worker_continuations_reject_calls_after_the_interceptor_settles() {
    enable_operational_logs();
    let state = Arc::new(WorkerHostRuntimeState::new(
        ACTIVATION_ID.into(),
        AUTH_TOKEN.into(),
    ));
    let service = WorkerHostRuntimeService {
        state: state.clone(),
    };
    let provider_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let (lease, guard) = MiddlewareContinuationLease::capture();
    let continuation_id = state
        .insert_continuation(Continuation::tool(Arc::new({
            let provider_calls = provider_calls.clone();
            move |value| {
                let provider_calls = provider_calls.clone();
                let invocation = lease.begin();
                Box::pin(async move {
                    invocation?
                        .invoke(|| async move {
                            provider_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                            Ok(ToolExecutionResult::new(value))
                        })
                        .await
                })
            }
        })))
        .expect("tool continuation should insert");

    drop(guard);
    let result = service
        .tool_next(Request::new(ToolNextRequest {
            activation_id: ACTIVATION_ID.into(),
            auth_token: AUTH_TOKEN.into(),
            continuation_id: continuation_id.clone(),
            value: Some(json_envelope(JSON_SCHEMA, &json!({})).expect("json envelope")),
            scope: None,
        }))
        .await
        .expect("late worker next should return a structured callback error")
        .into_inner();
    assert!(
        result.error.is_some_and(|error| {
            error
                .message
                .contains("execution continuation is no longer active")
        }),
        "late worker next should expose the shared continuation error"
    );
    assert_eq!(provider_calls.load(std::sync::atomic::Ordering::SeqCst), 0);

    state.remove_continuation(&continuation_id);
    let error = service
        .tool_next(Request::new(ToolNextRequest {
            activation_id: ACTIVATION_ID.into(),
            auth_token: AUTH_TOKEN.into(),
            continuation_id,
            value: Some(json_envelope(JSON_SCHEMA, &json!({})).expect("json envelope")),
            scope: None,
        }))
        .await
        .expect_err("cleaned-up worker continuation should no longer be addressable");
    assert_eq!(error.code(), tonic::Code::NotFound);
    assert_eq!(provider_calls.load(std::sync::atomic::Ordering::SeqCst), 0);
}

#[tokio::test]
async fn worker_continuations_use_the_scope_stack_selected_for_each_call() {
    let state = Arc::new(WorkerHostRuntimeState::new(
        ACTIVATION_ID.into(),
        AUTH_TOKEN.into(),
    ));
    let service = WorkerHostRuntimeService {
        state: state.clone(),
    };
    let mut expected = Vec::new();
    for stack_id in ["worker-next-first", "worker-next-second"] {
        let stack = crate::api::runtime::create_scope_stack();
        let scope = crate::api::runtime::with_scope_stack(stack.clone(), || {
            crate::api::scope::push_scope(
                crate::api::scope::PushScopeParams::builder()
                    .name(stack_id)
                    .scope_type(crate::api::scope::ScopeType::Agent)
                    .build(),
            )
        })
        .unwrap();
        expected.push(scope.uuid.to_string());
        state.scope_stacks.lock().unwrap().insert(
            stack_id.into(),
            StoredScopeStack {
                handle: stack,
                publication_buffer: None,
                invocation_base_depth: None,
            },
        );
    }
    let continuation_id = state
        .insert_continuation(Continuation::tool(Arc::new(|_| {
            Box::pin(async {
                Ok(ToolExecutionResult::new(json!(
                    crate::api::runtime::task_scope_top().uuid.to_string()
                )))
            })
        })))
        .expect("tool continuation should insert");
    let request = |stack_id: &str| {
        Request::new(ToolNextRequest {
            activation_id: ACTIVATION_ID.into(),
            auth_token: AUTH_TOKEN.into(),
            continuation_id: continuation_id.clone(),
            value: Some(json_envelope(JSON_SCHEMA, &json!({})).expect("json envelope")),
            scope: Some(ScopeContext {
                scope_stack_id: stack_id.into(),
                parent_scope_id: String::new(),
            }),
        })
    };

    let (first, second) = tokio::join!(
        service.tool_next(request("worker-next-first")),
        service.tool_next(request("worker-next-second"))
    );
    let decode = |response: tonic::Response<ToolExecutionResultResponse>| {
        let value = response
            .into_inner()
            .value
            .expect("tool next should return a value");
        decode_json_value::<Json>(value.result.as_ref().expect("tool next result")).unwrap()
    };

    assert_eq!(decode(first.unwrap()), json!(expected[0]));
    assert_eq!(decode(second.unwrap()), json!(expected[1]));
}

fn valid_llm_request() -> LlmRequest {
    LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({ "prompt": "unit" }),
    }
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

async fn fake_callback_service_with_handlers(
    invoke: impl Fn(InvokeRequest) -> FakeInvokeFuture + Send + Sync + 'static,
    invoke_stream: impl Fn(InvokeRequest) -> FakeInvokeStream + Send + Sync + 'static,
) -> (
    WorkerPluginCallback,
    oneshot::Sender<()>,
    mpsc::UnboundedReceiver<CancelInvocationRequest>,
) {
    let (client, shutdown_tx, cancel_rx) =
        fake_worker_client_with_handlers(invoke, invoke_stream).await;
    let (callback, shutdown_tx) = callback_for_client(client, shutdown_tx);
    (callback, shutdown_tx, cancel_rx)
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
            plugin_kind: "fixture_worker".into(),
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
    enable_operational_logs();
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
            teardown_started: AtomicBool::new(false),
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
    let invoke = Arc::new(invoke);
    let (client, shutdown_tx, _cancel_rx) = fake_worker_client_with_handlers(
        move |request| {
            let invoke = invoke.clone();
            Box::pin(async move { invoke(request) })
        },
        invoke_stream,
    )
    .await;
    (client, shutdown_tx)
}

async fn fake_worker_client_with_handlers(
    invoke: impl Fn(InvokeRequest) -> FakeInvokeFuture + Send + Sync + 'static,
    invoke_stream: impl Fn(InvokeRequest) -> FakeInvokeStream + Send + Sync + 'static,
) -> (
    PluginWorkerClient<Channel>,
    oneshot::Sender<()>,
    mpsc::UnboundedReceiver<CancelInvocationRequest>,
) {
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .expect("fake worker listener should bind");
    let addr = listener
        .local_addr()
        .expect("fake worker listener address should be available");
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let (cancel_tx, cancel_rx) = mpsc::unbounded_channel();
    tokio::spawn(
        Server::builder()
            .add_service(PluginWorkerServer::new(FakePluginWorker {
                invoke: Arc::new(invoke),
                invoke_stream: Arc::new(invoke_stream),
                cancel_tx,
            }))
            .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async {
                let _ = shutdown_rx.await;
            }),
    );
    let client = PluginWorkerClient::connect(format!("http://{addr}"))
        .await
        .expect("fake worker client should connect");
    (client, shutdown_tx, cancel_rx)
}

fn registration(surface: RegistrationSurface, local_name: &str) -> Registration {
    Registration {
        local_name: local_name.into(),
        surface: surface as i32,
        priority: 0,
        break_chain: false,
    }
}

#[test]
fn worker_loader_empty_input_returns_an_empty_activation() {
    let activation = load_worker_plugins(Vec::<WorkerPluginLoadSpec>::new()).unwrap();
    assert!(activation.is_empty());
    activation.clear();
}

fn poison_mutex(f: impl FnOnce() + std::panic::UnwindSafe) {
    let _ = std::panic::catch_unwind(f);
}

struct FakePluginWorker {
    invoke: Arc<dyn Fn(InvokeRequest) -> FakeInvokeFuture + Send + Sync>,
    invoke_stream: Arc<dyn Fn(InvokeRequest) -> FakeInvokeStream + Send + Sync>,
    cancel_tx: mpsc::UnboundedSender<CancelInvocationRequest>,
}

type FakeInvokeFuture = Pin<Box<dyn Future<Output = InvokeResponse> + Send>>;
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
            conditional_middleware_guardrails: Vec::new(),
        }))
    }

    async fn invoke(
        &self,
        request: Request<InvokeRequest>,
    ) -> std::result::Result<tonic::Response<InvokeResponse>, tonic::Status> {
        Ok(tonic::Response::new(
            (self.invoke)(request.into_inner()).await,
        ))
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
        request: Request<CancelInvocationRequest>,
    ) -> std::result::Result<tonic::Response<WorkerAck>, tonic::Status> {
        let _ = self.cancel_tx.send(request.into_inner());
        Ok(tonic::Response::new(WorkerAck {
            accepted: true,
            message: "cancelled".into(),
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
