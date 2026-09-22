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
        (None, "execution context is missing"),
        (
            Some(nemo_relay_worker_proto::v1::LlmExecutionCodecContext {
                request: None,
                response: Some(response()),
            }),
            "request context is missing",
        ),
        (
            Some(nemo_relay_worker_proto::v1::LlmExecutionCodecContext {
                request: Some(nemo_relay_worker_proto::v1::LlmSanitizeRequestContext::default()),
                response: Some(response()),
            }),
            "request codec identity is missing",
        ),
        (
            Some(nemo_relay_worker_proto::v1::LlmExecutionCodecContext {
                request: Some(request()),
                response: Some(nemo_relay_worker_proto::v1::LlmSanitizeResponseContext::default()),
            }),
            "response codec identity is missing",
        ),
        (
            Some(nemo_relay_worker_proto::v1::LlmExecutionCodecContext {
                request: Some(request()),
                response: None,
            }),
            "response context is missing",
        ),
    ];

    for (context, expected) in cases {
        let error = llm_payload(context.map(Box::new))
            .execution_context(&disconnected_runtime(), "invocation", true)
            .unwrap_err();
        assert!(
            error.to_string().contains(expected),
            "expected '{expected}' in '{error}'"
        );
    }
}

#[test]
fn execution_context_preserves_codec_identities_and_capability_availability() {
    let cases = [
        (
            LlmCodecKind::Unspecified,
            None,
            None,
            LlmCodecIdentity::None,
            false,
        ),
        (
            LlmCodecKind::Opaque,
            None,
            Some("opaque"),
            LlmCodecIdentity::Opaque,
            true,
        ),
        (
            LlmCodecKind::Builtin,
            Some("openai_chat"),
            Some("builtin"),
            LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OpenAiChat),
            true,
        ),
    ];

    for (kind, id, capability, expected, resolves) in cases {
        let codec = || nemo_relay_worker_proto::v1::LlmCodecIdentity {
            kind: kind as i32,
            id: id.map(str::to_owned),
        };
        let capability = |direction: &str| capability.map(|value| format!("{value}-{direction}"));
        let context = llm_payload(Some(Box::new(
            nemo_relay_worker_proto::v1::LlmExecutionCodecContext {
                request: Some(nemo_relay_worker_proto::v1::LlmSanitizeRequestContext {
                    codec: Some(codec()),
                    codec_capability_id: capability("request"),
                }),
                response: Some(nemo_relay_worker_proto::v1::LlmSanitizeResponseContext {
                    codec: Some(codec()),
                    codec_capability_id: capability("response"),
                }),
            },
        )))
        .execution_context(&disconnected_runtime(), "invocation", true)
        .unwrap();

        assert_eq!(context.request_codec().codec, expected);
        assert_eq!(context.request_codec().resolve_codec().is_some(), resolves);
        let response = context.response_codec().expect("unary response context");
        assert_eq!(response.codec, expected);
        assert_eq!(response.resolve_codec().is_some(), resolves);
    }
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
