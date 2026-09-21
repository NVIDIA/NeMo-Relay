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
fn context_registration_is_opt_in_without_a_new_surface() {
    let mut context = PluginContext::new();
    context.register_llm_execution_intercept("legacy", 7, |_, _, _| async {
        Ok(serde_json::json!({"legacy": true}))
    });
    context.register_llm_execution_intercept_with_context("context", 7, |_, _, _, _| async {
        Ok(serde_json::json!({"context": true}))
    });

    let legacy = &context.handlers.registrations[0];
    let contextual = &context.handlers.registrations[1];
    assert_eq!(
        legacy.surface,
        RegistrationSurface::LlmExecutionIntercept as i32
    );
    assert_eq!(contextual.surface, legacy.surface);
    assert_eq!(contextual.priority, legacy.priority);
    assert!(!legacy.llm_execution_codec_context);
    assert!(contextual.llm_execution_codec_context);
}

#[test]
fn absent_execution_context_identifies_an_older_host() {
    let payload = llm_payload(None);

    let context = payload
        .execution_context(&disconnected_runtime(), "invocation")
        .unwrap();
    assert!(!context.is_available());
    assert_eq!(context.request_codec_identity(), &LlmCodecIdentity::None);
    assert_eq!(context.response_codec_identity(), &LlmCodecIdentity::None);
    assert!(context.request_codec().is_none());
    assert!(context.response_codec().is_none());
}

#[test]
fn execution_context_from_a_new_host_preserves_identities_and_capabilities() {
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
        .execution_context(&disconnected_runtime(), "invocation")
        .unwrap();
    assert!(context.is_available());
    assert_eq!(
        context.request_codec_identity(),
        &LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OpenAiChat)
    );
    assert_eq!(
        context.response_codec_identity(),
        &LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OpenAiChat)
    );
    assert!(context.request_codec().is_some());
    assert!(context.response_codec().is_some());
}
