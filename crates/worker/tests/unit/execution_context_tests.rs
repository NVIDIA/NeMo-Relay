// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn disconnected_runtime() -> PluginRuntime {
    PluginRuntime {
        activation_id: "activation".into(),
        auth_token: "token".into(),
        host_endpoint: "http://127.0.0.1:1".into(),
        host_channel: Arc::new(OnceCell::new()),
        conditional_middleware_callbacks: Arc::new(Mutex::new(HashMap::new())),
    }
}

fn llm_payload(
    execution_codec_context: Option<Box<nemo_relay_worker_proto::v1::LlmExecutionCodecContext>>,
) -> LlmPayload {
    LlmPayload {
        model_name: "model".into(),
        request: None,
        annotated_request: None,
        response: None,
        sanitize_context: None,
        execution_codec_context,
    }
}

#[test]
fn execution_registration_uses_the_existing_surface() {
    let mut context = PluginContext::new();
    context.register_llm_execution_intercept("context", 7, |_, _, _, _| async {
        Ok(serde_json::json!({"context": true}))
    });

    let registration = &context.handlers.registrations[0];
    assert_eq!(
        registration.surface,
        RegistrationSurface::LlmExecutionIntercept as i32
    );
    assert_eq!(registration.priority, 7);
}

#[test]
fn absent_execution_context_is_a_release_mismatch() {
    let payload = llm_payload(None);

    let error = payload
        .execution_context(&disconnected_runtime(), "invocation", true)
        .unwrap_err();
    assert!(error.to_string().contains("execution context is missing"));
}

#[test]
fn malformed_execution_context_fields_are_rejected() {
    let request = || nemo_relay_worker_proto::v1::LlmSanitizeRequestContext {
        codec: Some(nemo_relay_worker_proto::v1::LlmCodecIdentity::default()),
        codec_capability_id: None,
    };
    let response = || nemo_relay_worker_proto::v1::LlmSanitizeResponseContext {
        codec: Some(nemo_relay_worker_proto::v1::LlmCodecIdentity::default()),
        codec_capability_id: None,
    };
    let cases = [
        (
            nemo_relay_worker_proto::v1::LlmExecutionCodecContext {
                request: None,
                response: Some(response()),
            },
            "request context is missing",
        ),
        (
            nemo_relay_worker_proto::v1::LlmExecutionCodecContext {
                request: Some(nemo_relay_worker_proto::v1::LlmSanitizeRequestContext::default()),
                response: Some(response()),
            },
            "request codec identity is missing",
        ),
        (
            nemo_relay_worker_proto::v1::LlmExecutionCodecContext {
                request: Some(request()),
                response: Some(nemo_relay_worker_proto::v1::LlmSanitizeResponseContext::default()),
            },
            "response codec identity is missing",
        ),
        (
            nemo_relay_worker_proto::v1::LlmExecutionCodecContext {
                request: Some(request()),
                response: None,
            },
            "response context is missing",
        ),
    ];

    for (context, expected) in cases {
        let error = llm_payload(Some(Box::new(context)))
            .execution_context(&disconnected_runtime(), "invocation", true)
            .unwrap_err();
        assert!(
            error.to_string().contains(expected),
            "expected '{expected}' in '{error}'"
        );
    }
}

#[test]
fn execution_context_preserves_directional_identities_and_capabilities() {
    let codec = nemo_relay_worker_proto::v1::LlmCodecIdentity {
        kind: LlmCodecKind::Builtin as i32,
        id: Some("openai_chat".into()),
    };
    let payload = llm_payload(Some(Box::new(
        nemo_relay_worker_proto::v1::LlmExecutionCodecContext {
            request: Some(nemo_relay_worker_proto::v1::LlmSanitizeRequestContext {
                codec: Some(codec.clone()),
                codec_capability_id: Some("request-capability".into()),
            }),
            response: Some(nemo_relay_worker_proto::v1::LlmSanitizeResponseContext {
                codec: Some(codec),
                codec_capability_id: Some("response-capability".into()),
            }),
        },
    )));

    let context = payload
        .execution_context(&disconnected_runtime(), "invocation", true)
        .unwrap();
    assert_eq!(
        &context.request_codec().codec,
        &LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OpenAiChat)
    );
    assert!(context.request_codec().resolve_codec().is_some());
    let response = context.response_codec().expect("unary response context");
    assert_eq!(
        &response.codec,
        &LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OpenAiChat)
    );
    assert!(response.resolve_codec().is_some());
}

#[test]
fn execution_context_distinguishes_absent_and_resolved_opaque_codecs() {
    let absent = llm_payload(Some(Box::new(
        nemo_relay_worker_proto::v1::LlmExecutionCodecContext {
            request: Some(nemo_relay_worker_proto::v1::LlmSanitizeRequestContext {
                codec: Some(nemo_relay_worker_proto::v1::LlmCodecIdentity {
                    kind: LlmCodecKind::Unspecified as i32,
                    id: None,
                }),
                codec_capability_id: None,
            }),
            response: Some(nemo_relay_worker_proto::v1::LlmSanitizeResponseContext {
                codec: Some(nemo_relay_worker_proto::v1::LlmCodecIdentity {
                    kind: LlmCodecKind::Unspecified as i32,
                    id: None,
                }),
                codec_capability_id: None,
            }),
        },
    )))
    .execution_context(&disconnected_runtime(), "absent-invocation", true)
    .unwrap();
    assert_eq!(absent.request_codec().codec, LlmCodecIdentity::None);
    assert!(absent.request_codec().resolve_codec().is_none());
    let absent_response = absent.response_codec().expect("unary response direction");
    assert_eq!(absent_response.codec, LlmCodecIdentity::None);
    assert!(absent_response.resolve_codec().is_none());

    let opaque = llm_payload(Some(Box::new(
        nemo_relay_worker_proto::v1::LlmExecutionCodecContext {
            request: Some(nemo_relay_worker_proto::v1::LlmSanitizeRequestContext {
                codec: Some(nemo_relay_worker_proto::v1::LlmCodecIdentity {
                    kind: LlmCodecKind::Opaque as i32,
                    id: None,
                }),
                codec_capability_id: Some("opaque-request".into()),
            }),
            response: Some(nemo_relay_worker_proto::v1::LlmSanitizeResponseContext {
                codec: Some(nemo_relay_worker_proto::v1::LlmCodecIdentity {
                    kind: LlmCodecKind::Opaque as i32,
                    id: None,
                }),
                codec_capability_id: Some("opaque-response".into()),
            }),
        },
    )))
    .execution_context(&disconnected_runtime(), "opaque-invocation", true)
    .unwrap();
    assert_eq!(opaque.request_codec().codec, LlmCodecIdentity::Opaque);
    assert!(opaque.request_codec().resolve_codec().is_some());
    let opaque_response = opaque.response_codec().expect("unary response direction");
    assert_eq!(opaque_response.codec, LlmCodecIdentity::Opaque);
    assert!(opaque_response.resolve_codec().is_some());
}

#[test]
fn streaming_execution_context_has_no_response_codec() {
    let payload = llm_payload(Some(Box::new(
        nemo_relay_worker_proto::v1::LlmExecutionCodecContext {
            request: Some(nemo_relay_worker_proto::v1::LlmSanitizeRequestContext {
                codec: Some(nemo_relay_worker_proto::v1::LlmCodecIdentity {
                    kind: LlmCodecKind::Builtin as i32,
                    id: Some("openai_chat".into()),
                }),
                codec_capability_id: Some("request-capability".into()),
            }),
            response: None,
        },
    )));

    let context = payload
        .execution_context(&disconnected_runtime(), "invocation", false)
        .unwrap();
    assert!(context.request_codec().resolve_codec().is_some());
    assert!(context.response_codec().is_none());
}
