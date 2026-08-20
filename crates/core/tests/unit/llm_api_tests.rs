// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for LLM API lifecycle behavior.

#![allow(clippy::await_holding_lock)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};

use serde_json::json;
use tokio_stream::StreamExt;

use super::{
    CreateLlmHandleParams, LlmCallEndParams, LlmCallExecuteParams, LlmCallParams, LlmHandle,
    LlmRequest, LlmStreamCallExecuteParams, create_llm_handle, emit_llm_start,
    emit_optimization_marks_with, enqueue_optimization_marks, llm_call, llm_call_end,
    llm_call_execute, llm_stream_call_execute, project_llm_request_to_current_user_turn,
    sanitize_context_for_request_codec, sanitize_context_for_response_codec,
};
use crate::api::event::{Event, ScopeCategory};
use crate::api::optimization::finalize_optimization_summary;
use crate::api::registry::{
    deregister_llm_sanitize_request_guardrail, deregister_llm_sanitize_response_guardrail,
    register_llm_sanitize_request_guardrail, register_llm_sanitize_response_guardrail,
};
use crate::api::runtime::{BuiltinLlmCodec, LlmCodecIdentity, LlmJsonStream};
use crate::api::runtime::{
    NemoRelayContextState, create_scope_stack, global_context, set_thread_scope_stack,
};
use crate::api::scope::{COMPACTION_EVENT_NAME, EmitMarkEventParams, event};
use crate::api::scope::{PopScopeParams, PushScopeParams, ScopeType, pop_scope, push_scope};
use crate::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use crate::codec::anthropic::AnthropicMessagesCodec;
use crate::codec::oci_genai::OCIGenAIChatCodec;
use crate::codec::openai_chat::OpenAIChatCodec;
use crate::codec::openai_responses::OpenAIResponsesCodec;
use crate::codec::request::{AnnotatedLlmRequest, Message, MessageContent};
use crate::codec::traits::{LlmCodec, LlmResponseCodec};
use crate::error::FlowError;
use crate::json::Json;
use crate::{
    codec::optimization::LlmOptimizationContribution,
    codec::response::{AnnotatedLlmResponse, PricingResolver},
};

fn reset_global() {
    let _ = spdlog::init_log_crate_proxy();
    log::set_max_level(log::LevelFilter::Info);
    crate::shared_runtime::reset_runtime_owner_for_tests();
    let context = global_context();
    *context.write().unwrap() = NemoRelayContextState::new();
}

fn lock_global_runtime() -> std::sync::MutexGuard<'static, ()> {
    crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner())
}

fn request() -> LlmRequest {
    LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({"messages": [], "model": "demo"}),
    }
}

fn request_with_credential_headers() -> LlmRequest {
    let mut headers = serde_json::Map::new();
    for (name, value) in [
        ("Authorization", "Bearer authorization-secret"),
        ("PrOxY-AuThOrIzAtIoN", "Basic proxy-secret"),
        ("COOKIE", "session=cookie-secret"),
        ("X-Api-Key", "x-api-key-secret"),
        ("API-KEY", "api-key-secret"),
        ("Anthropic-Api-Key", "anthropic-api-key-secret"),
        ("X-GoOg-Api-Key", "x-goog-api-key-secret"),
        ("TraceParent", "user-provided-traceparent"),
    ] {
        headers.insert(name.to_string(), json!(value));
    }
    headers.insert("x-request-id".to_string(), json!("safe-request-id"));
    LlmRequest {
        headers,
        content: json!({"messages": [], "model": "demo"}),
    }
}

fn assert_observable_credential_headers_are_removed(request: &LlmRequest) {
    assert_eq!(request.headers.len(), 2);
    assert_eq!(
        request.headers.get("x-request-id"),
        Some(&json!("safe-request-id"))
    );
    assert_eq!(
        request.headers.get("TraceParent"),
        Some(&json!("user-provided-traceparent"))
    );
}

fn multi_turn_request() -> LlmRequest {
    LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "demo",
            "messages": [
                {"role": "system", "content": "instructions"},
                {"role": "user", "content": "earlier question"},
                {"role": "assistant", "content": "earlier answer"},
                {"role": "user", "content": "latest question"}
            ]
        }),
    }
}

fn multi_turn_annotation() -> Arc<AnnotatedLlmRequest> {
    Arc::new(OpenAIChatCodec.decode(&multi_turn_request()).unwrap())
}

struct ProjectionFailingCodec {
    projection_attempts: Arc<AtomicUsize>,
}

struct RuntimeIdentityCodec;

impl LlmCodec for RuntimeIdentityCodec {
    fn codec_identity(&self) -> LlmCodecIdentity {
        LlmCodecIdentity::Runtime("com.example.chat.v1".into())
    }

    fn decode(&self, request: &LlmRequest) -> crate::error::Result<AnnotatedLlmRequest> {
        OpenAIChatCodec.decode(request)
    }

    fn encode(
        &self,
        annotated: &AnnotatedLlmRequest,
        original: &LlmRequest,
    ) -> crate::error::Result<LlmRequest> {
        OpenAIChatCodec.encode(annotated, original)
    }
}

impl LlmResponseCodec for RuntimeIdentityCodec {
    fn codec_identity(&self) -> LlmCodecIdentity {
        LlmCodecIdentity::Runtime("com.example.chat.v1".into())
    }

    fn decode_response(&self, response: &Json) -> crate::error::Result<AnnotatedLlmResponse> {
        OpenAIChatCodec.decode_response(response)
    }
}

#[test]
fn request_sanitizer_context_preserves_all_codec_identity_states() {
    let identity_only_request = crate::api::runtime::LlmSanitizeRequestContext::with_identity(
        LlmCodecIdentity::Runtime("identity-only.request.v1".into()),
    );
    assert_eq!(
        identity_only_request.codec(),
        &LlmCodecIdentity::Runtime("identity-only.request.v1".into())
    );
    assert!(identity_only_request.resolve_codec().is_none());
    assert!(format!("{identity_only_request:?}").contains("identity-only.request.v1"));

    assert_eq!(
        sanitize_context_for_request_codec(None).codec(),
        &LlmCodecIdentity::None
    );
    assert_eq!(
        sanitize_context_for_request_codec(Some(&OpenAIChatCodec)).codec(),
        &LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OpenAiChat)
    );
    assert_eq!(
        sanitize_context_for_request_codec(Some(&OpenAIResponsesCodec)).codec(),
        &LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OpenAiResponses)
    );
    assert_eq!(
        sanitize_context_for_request_codec(Some(&AnthropicMessagesCodec)).codec(),
        &LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::AnthropicMessages)
    );
    assert_eq!(
        sanitize_context_for_request_codec(Some(&OCIGenAIChatCodec)).codec(),
        &LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OCIGenAI)
    );
    assert_eq!(
        sanitize_context_for_request_codec(Some(&RuntimeIdentityCodec)).codec(),
        &LlmCodecIdentity::Runtime("com.example.chat.v1".into())
    );
    assert_eq!(
        sanitize_context_for_request_codec(Some(&ProjectionFailingCodec {
            projection_attempts: Arc::new(AtomicUsize::new(0)),
        }))
        .codec(),
        &LlmCodecIdentity::Opaque
    );
}

#[test]
fn response_sanitizer_context_preserves_all_codec_identity_states() {
    let identity_only_response = crate::api::runtime::LlmSanitizeResponseContext::with_identity(
        LlmCodecIdentity::Runtime("identity-only.response.v1".into()),
    );
    assert_eq!(
        identity_only_response.codec(),
        &LlmCodecIdentity::Runtime("identity-only.response.v1".into())
    );
    assert!(identity_only_response.resolve_codec().is_none());
    assert!(format!("{identity_only_response:?}").contains("identity-only.response.v1"));

    assert_eq!(
        sanitize_context_for_response_codec(Some(&OpenAIChatCodec as &dyn LlmResponseCodec))
            .codec(),
        &LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OpenAiChat)
    );
    assert_eq!(
        sanitize_context_for_response_codec(Some(&OpenAIResponsesCodec as &dyn LlmResponseCodec))
            .codec(),
        &LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OpenAiResponses)
    );
    assert_eq!(
        sanitize_context_for_response_codec(Some(&AnthropicMessagesCodec as &dyn LlmResponseCodec))
            .codec(),
        &LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::AnthropicMessages)
    );
    assert_eq!(
        sanitize_context_for_response_codec(Some(&OCIGenAIChatCodec as &dyn LlmResponseCodec))
            .codec(),
        &LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OCIGenAI)
    );
    assert_eq!(
        sanitize_context_for_response_codec(Some(&RuntimeIdentityCodec)).codec(),
        &LlmCodecIdentity::Runtime("com.example.chat.v1".into())
    );
    assert_eq!(
        sanitize_context_for_response_codec(None).codec(),
        &LlmCodecIdentity::None
    );
    assert_eq!(
        sanitize_context_for_response_codec(Some(&ProjectionFailingCodec {
            projection_attempts: Arc::new(AtomicUsize::new(0)),
        }))
        .codec(),
        &LlmCodecIdentity::Opaque
    );
}

impl LlmCodec for ProjectionFailingCodec {
    fn decode(&self, request: &LlmRequest) -> crate::error::Result<AnnotatedLlmRequest> {
        OpenAIChatCodec.decode(request)
    }

    fn encode(
        &self,
        annotated: &AnnotatedLlmRequest,
        original: &LlmRequest,
    ) -> crate::error::Result<LlmRequest> {
        let original_messages = original.content["messages"].as_array().map_or(0, Vec::len);
        if annotated.messages.len() < original_messages {
            self.projection_attempts.fetch_add(1, Ordering::Relaxed);
            return Err(FlowError::Internal("projection encode failed".into()));
        }
        OpenAIChatCodec.encode(annotated, original)
    }
}

impl LlmResponseCodec for ProjectionFailingCodec {
    fn decode_response(
        &self,
        response: &Json,
    ) -> crate::error::Result<crate::codec::response::AnnotatedLlmResponse> {
        OpenAIChatCodec.decode_response(response)
    }
}

fn emit_compaction() {
    event(
        EmitMarkEventParams::builder()
            .name(COMPACTION_EVENT_NAME)
            .build(),
    )
    .unwrap();
}

fn secret_request() -> LlmRequest {
    LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "SECRET"}]
        }),
    }
}

fn redacted_request() -> LlmRequest {
    LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "[REDACTED]"}]
        }),
    }
}

#[test]
fn credential_headers_are_removed_before_request_sanitizers_and_event_emission() {
    let _guard = lock_global_runtime();
    reset_global();
    set_thread_scope_stack(create_scope_stack());

    let request = request_with_credential_headers();
    let sanitizer_requests = Arc::new(Mutex::new(Vec::<LlmRequest>::new()));
    let sanitizer_capture = Arc::clone(&sanitizer_requests);
    register_llm_sanitize_request_guardrail(
        "credential-header-redaction",
        1,
        Arc::new(move |request, _context| {
            sanitizer_capture.lock().unwrap().push(request.clone());
            Box::pin(async move { Ok(Some(request)) })
        }),
    )
    .unwrap();

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let event_capture = Arc::clone(&events);
    register_subscriber(
        "credential-header-redaction",
        Arc::new(move |event| event_capture.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    llm_call(
        LlmCallParams::builder()
            .name("credential-header-manual")
            .request(&request)
            .build(),
    )
    .unwrap();

    let provider_requests = Arc::new(Mutex::new(Vec::<(String, LlmRequest)>::new()));
    let buffered_provider_requests = Arc::clone(&provider_requests);
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        llm_call_execute(
            LlmCallExecuteParams::builder()
                .name("credential-header-buffered")
                .request(request.clone())
                .func(Arc::new(move |request| {
                    buffered_provider_requests
                        .lock()
                        .unwrap()
                        .push(("credential-header-buffered".into(), request));
                    Box::pin(async { Ok(json!({"ok": true})) })
                }))
                .build(),
        )
        .await
        .unwrap();

        let streaming_provider_requests = Arc::clone(&provider_requests);
        let mut stream = llm_stream_call_execute(
            LlmStreamCallExecuteParams::builder()
                .name("credential-header-streaming")
                .request(request.clone())
                .func(Arc::new(move |request| {
                    streaming_provider_requests
                        .lock()
                        .unwrap()
                        .push(("credential-header-streaming".into(), request));
                    Box::pin(async {
                        Ok(LlmJsonStream::new(tokio_stream::iter(vec![Ok(json!({
                            "chunk": true
                        }))])))
                    })
                }))
                .collector(Box::new(|_chunk| Ok(())))
                .finalizer(Box::new(|| json!({"ok": true})))
                .build(),
        )
        .await
        .unwrap();
        while let Some(chunk) = stream.next().await {
            chunk.unwrap();
        }
    });

    flush_subscribers().unwrap();
    for sanitized in sanitizer_requests.lock().unwrap().iter() {
        assert_observable_credential_headers_are_removed(sanitized);
    }
    assert_eq!(sanitizer_requests.lock().unwrap().len(), 3);

    let events = events.lock().unwrap();
    let start_events = [
        "credential-header-manual",
        "credential-header-buffered",
        "credential-header-streaming",
    ]
    .into_iter()
    .map(|name| {
        events
            .iter()
            .find(|event| {
                event.name() == name && event.scope_category() == Some(ScopeCategory::Start)
            })
            .cloned()
            .unwrap_or_else(|| panic!("missing LLM start event {name}"))
    })
    .collect::<Vec<_>>();
    drop(events);
    assert_eq!(start_events.len(), 3);
    for event in &start_events {
        let input: LlmRequest = serde_json::from_value(event.input().cloned().unwrap()).unwrap();
        assert_observable_credential_headers_are_removed(&input);
    }

    let provider_requests = provider_requests.lock().unwrap();
    assert_eq!(provider_requests.len(), 2);
    for (name, provider) in provider_requests.iter() {
        assert_eq!(provider.content, request.content);
        assert_eq!(
            provider.headers.get("x-request-id"),
            request.headers.get("x-request-id")
        );
        let traceparent_headers = provider
            .headers
            .iter()
            .filter(|(key, _)| key.eq_ignore_ascii_case("traceparent"))
            .collect::<Vec<_>>();
        assert_eq!(traceparent_headers.len(), 1);
        assert_eq!(traceparent_headers[0].0, "traceparent");
        let traceparent = traceparent_headers[0]
            .1
            .as_str()
            .expect("managed LLM traceparent must be a string");
        assert!(traceparent.starts_with("00-"));
        assert!(traceparent.ends_with("-01"));
        assert_eq!(traceparent.len(), 55);

        let start = start_events
            .iter()
            .find(|event| event.name() == name)
            .unwrap_or_else(|| panic!("missing LLM start event {name}"));
        let event_uuid = start.uuid().simple().to_string();
        assert_eq!(&traceparent[36..52], &event_uuid[16..]);
    }

    assert!(deregister_llm_sanitize_request_guardrail("credential-header-redaction").unwrap());
    assert!(deregister_subscriber("credential-header-redaction").unwrap());
}

fn secret_response() -> Json {
    json!({
        "id": "chatcmpl-test",
        "model": "gpt-4o-mini",
        "choices": [{"message": {"role": "assistant", "content": "SECRET"}}]
    })
}

fn redacted_response() -> Json {
    json!({
        "id": "chatcmpl-test",
        "model": "gpt-4o-mini",
        "choices": [{"message": {"role": "assistant", "content": "[REDACTED]"}}]
    })
}

#[test]
fn sanitization_invalidates_manual_annotations_without_a_codec() {
    let _guard = lock_global_runtime();
    reset_global();
    set_thread_scope_stack(create_scope_stack());

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    register_subscriber(
        "manual-annotation-invalidation",
        Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();
    register_llm_sanitize_request_guardrail(
        "manual-annotation-invalidation-request",
        1,
        Arc::new(|_request, _context| Box::pin(async { Ok(Some(redacted_request())) })),
    )
    .unwrap();
    register_llm_sanitize_response_guardrail(
        "manual-annotation-invalidation-response",
        1,
        Arc::new(|_response, _context| Box::pin(async { Ok(Some(redacted_response())) })),
    )
    .unwrap();

    let request = secret_request();
    let handle = llm_call(
        LlmCallParams::builder()
            .name("manual")
            .request(&request)
            .annotated_request(Arc::new(OpenAIChatCodec.decode(&request).unwrap()))
            .build(),
    )
    .unwrap();
    let response = secret_response();
    llm_call_end(
        LlmCallEndParams::builder()
            .handle(&handle)
            .response(response.clone())
            .annotated_response(Arc::new(
                OpenAIChatCodec.decode_response(&response).unwrap(),
            ))
            .build(),
    )
    .unwrap();

    flush_subscribers().unwrap();
    let captured = events.lock().unwrap();
    assert_eq!(captured.len(), 2);
    assert!(
        captured
            .iter()
            .all(|event| !serde_json::to_string(event).unwrap().contains("SECRET"))
    );
    assert!(captured[0].annotated_request().is_none());
    assert!(captured[1].annotated_response().is_none());

    assert!(
        deregister_llm_sanitize_request_guardrail("manual-annotation-invalidation-request")
            .unwrap()
    );
    assert!(
        deregister_llm_sanitize_response_guardrail("manual-annotation-invalidation-response")
            .unwrap()
    );
    assert!(deregister_subscriber("manual-annotation-invalidation").unwrap());
}

#[test]
fn no_op_sanitizers_keep_manual_annotations() {
    let _guard = lock_global_runtime();
    reset_global();
    set_thread_scope_stack(create_scope_stack());

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    register_subscriber(
        "manual-annotation-noop",
        Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();
    register_llm_sanitize_request_guardrail(
        "manual-annotation-noop-request",
        1,
        Arc::new(|request, _context| Box::pin(async move { Ok(Some(request)) })),
    )
    .unwrap();
    register_llm_sanitize_response_guardrail(
        "manual-annotation-noop-response",
        1,
        Arc::new(|response, _context| Box::pin(async move { Ok(Some(response)) })),
    )
    .unwrap();

    let request = secret_request();
    let handle = llm_call(
        LlmCallParams::builder()
            .name("manual")
            .request(&request)
            .annotated_request(Arc::new(OpenAIChatCodec.decode(&request).unwrap()))
            .build(),
    )
    .unwrap();
    let response = secret_response();
    llm_call_end(
        LlmCallEndParams::builder()
            .handle(&handle)
            .response(response.clone())
            .annotated_response(Arc::new(
                OpenAIChatCodec.decode_response(&response).unwrap(),
            ))
            .build(),
    )
    .unwrap();

    flush_subscribers().unwrap();
    let captured = events.lock().unwrap();
    assert_eq!(captured.len(), 2);
    assert!(captured[0].annotated_request().is_some());
    assert!(captured[1].annotated_response().is_some());

    assert!(deregister_llm_sanitize_request_guardrail("manual-annotation-noop-request").unwrap());
    assert!(deregister_llm_sanitize_response_guardrail("manual-annotation-noop-response").unwrap());
    assert!(deregister_subscriber("manual-annotation-noop").unwrap());
}

#[test]
fn sanitization_regenerates_annotations_with_active_codecs() {
    let _guard = lock_global_runtime();
    reset_global();
    set_thread_scope_stack(create_scope_stack());

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    register_subscriber(
        "active-codec-annotation-regeneration",
        Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();
    register_llm_sanitize_request_guardrail(
        "active-codec-annotation-regeneration-request",
        1,
        Arc::new(|_request, _context| Box::pin(async { Ok(Some(redacted_request())) })),
    )
    .unwrap();
    register_llm_sanitize_response_guardrail(
        "active-codec-annotation-regeneration-response",
        1,
        Arc::new(|_response, _context| Box::pin(async { Ok(Some(redacted_response())) })),
    )
    .unwrap();

    let request = secret_request();
    let handle = create_llm_handle(
        CreateLlmHandleParams::builder()
            .name("manual-with-codec")
            .build(),
    )
    .unwrap();
    emit_llm_start(
        &handle,
        &request,
        Some(Arc::new(OpenAIChatCodec.decode(&request).unwrap())),
        Some(Arc::new(OpenAIChatCodec)),
    )
    .unwrap();
    let response = secret_response();
    llm_call_end(
        LlmCallEndParams::builder()
            .handle(&handle)
            .response(response.clone())
            .annotated_response(Arc::new(
                OpenAIChatCodec.decode_response(&response).unwrap(),
            ))
            .response_codec(Arc::new(OpenAIChatCodec))
            .build(),
    )
    .unwrap();

    flush_subscribers().unwrap();
    let captured = events.lock().unwrap();
    assert_eq!(captured.len(), 2);
    assert!(
        captured
            .iter()
            .all(|event| !serde_json::to_string(event).unwrap().contains("SECRET"))
    );
    assert_eq!(
        captured[0].annotated_request().unwrap().messages,
        vec![Message::User {
            content: MessageContent::Text("[REDACTED]".to_string()),
            name: None,
        }]
    );
    assert_eq!(
        captured[1].annotated_response().unwrap().message,
        Some(MessageContent::Text("[REDACTED]".to_string()))
    );

    assert!(
        deregister_llm_sanitize_request_guardrail("active-codec-annotation-regeneration-request")
            .unwrap()
    );
    assert!(
        deregister_llm_sanitize_response_guardrail("active-codec-annotation-regeneration-response")
            .unwrap()
    );
    assert!(deregister_subscriber("active-codec-annotation-regeneration").unwrap());
}

#[test]
fn buffered_null_fallback_is_sanitized_before_emission() {
    let _guard = lock_global_runtime();
    reset_global();
    set_thread_scope_stack(create_scope_stack());

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = Arc::clone(&events);
    register_subscriber(
        "buffered-null-fallback",
        Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let seen = Arc::new(Mutex::new(Vec::<Json>::new()));
    let sanitizer_inputs = Arc::clone(&seen);
    register_llm_sanitize_response_guardrail(
        "buffered-null-fallback-null",
        1,
        Arc::new(move |response, _context| {
            sanitizer_inputs.lock().unwrap().push(response);
            Box::pin(async { Ok(Some(Json::Null)) })
        }),
    )
    .unwrap();

    let handle = create_llm_handle(
        CreateLlmHandleParams::builder()
            .name("buffered-null-fallback-null")
            .build(),
    )
    .unwrap();
    let fallback = secret_response();
    llm_call_end(
        LlmCallEndParams::builder()
            .handle(&handle)
            .response(Json::Null)
            .data(fallback.clone())
            .annotated_response(Arc::new(
                OpenAIChatCodec.decode_response(&fallback).unwrap(),
            ))
            .response_codec(Arc::new(OpenAIChatCodec))
            .build(),
    )
    .unwrap();
    assert!(deregister_llm_sanitize_response_guardrail("buffered-null-fallback-null").unwrap());

    register_llm_sanitize_response_guardrail(
        "buffered-null-fallback-redacted",
        1,
        Arc::new(|_response, _context| Box::pin(async { Ok(Some(redacted_response())) })),
    )
    .unwrap();
    let handle = create_llm_handle(
        CreateLlmHandleParams::builder()
            .name("buffered-null-fallback-redacted")
            .build(),
    )
    .unwrap();
    llm_call_end(
        LlmCallEndParams::builder()
            .handle(&handle)
            .response(Json::Null)
            .data(fallback.clone())
            .annotated_response(Arc::new(
                OpenAIChatCodec.decode_response(&fallback).unwrap(),
            ))
            .response_codec(Arc::new(OpenAIChatCodec))
            .build(),
    )
    .unwrap();
    assert!(deregister_llm_sanitize_response_guardrail("buffered-null-fallback-redacted").unwrap());

    let handle = create_llm_handle(
        CreateLlmHandleParams::builder()
            .name("buffered-null-without-fallback")
            .build(),
    )
    .unwrap();
    llm_call_end(
        LlmCallEndParams::builder()
            .handle(&handle)
            .response(Json::Null)
            .build(),
    )
    .unwrap();

    let handle = create_llm_handle(
        CreateLlmHandleParams::builder()
            .name("buffered-explicit-null-fallback")
            .build(),
    )
    .unwrap();
    llm_call_end(
        LlmCallEndParams::builder()
            .handle(&handle)
            .response(Json::Null)
            .data(Json::Null)
            .build(),
    )
    .unwrap();

    flush_subscribers().unwrap();
    let captured = events.lock().unwrap();
    assert_eq!(*seen.lock().unwrap(), vec![fallback]);
    assert_eq!(captured.len(), 4);
    assert_eq!(captured[0].output(), Some(&Json::Null));
    assert!(captured[0].annotated_response().is_none());
    assert_eq!(captured[1].output(), Some(&redacted_response()));
    assert_eq!(
        captured[1].annotated_response().unwrap().message,
        Some(MessageContent::Text("[REDACTED]".to_string()))
    );
    assert!(captured[2].output().is_none());
    assert!(captured[2].annotated_response().is_none());
    assert_eq!(captured[3].output(), Some(&Json::Null));
    assert!(captured[3].annotated_response().is_none());
    assert!(
        captured
            .iter()
            .all(|event| !serde_json::to_string(event).unwrap().contains("SECRET"))
    );

    assert!(deregister_subscriber("buffered-null-fallback").unwrap());
}

#[test]
fn streaming_null_fallback_is_sanitized_before_emission() {
    let _guard = lock_global_runtime();
    reset_global();
    set_thread_scope_stack(create_scope_stack());

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = Arc::clone(&events);
    register_subscriber(
        "streaming-null-fallback",
        Arc::new(move |event| {
            if event.scope_category() == Some(ScopeCategory::End) {
                captured.lock().unwrap().push(event.clone());
            }
        }),
    )
    .unwrap();

    let seen = Arc::new(Mutex::new(Vec::<Json>::new()));
    let sanitizer_inputs = Arc::clone(&seen);
    register_llm_sanitize_response_guardrail(
        "streaming-null-fallback-null",
        1,
        Arc::new(move |response, _context| {
            sanitizer_inputs.lock().unwrap().push(response);
            Box::pin(async { Ok(Some(Json::Null)) })
        }),
    )
    .unwrap();

    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        let mut stream = llm_stream_call_execute(
            LlmStreamCallExecuteParams::builder()
                .name("streaming-null-fallback-null")
                .request(request())
                .func(Arc::new(|_request| {
                    Box::pin(async { Ok(LlmJsonStream::new(tokio_stream::empty())) })
                }))
                .collector(Box::new(|_chunk| Ok(())))
                .finalizer(Box::new(|| Json::Null))
                .data(secret_response())
                .response_codec(Arc::new(OpenAIChatCodec))
                .build(),
        )
        .await
        .unwrap();
        while let Some(chunk) = stream.next().await {
            chunk.unwrap();
        }
    });
    assert!(deregister_llm_sanitize_response_guardrail("streaming-null-fallback-null").unwrap());

    register_llm_sanitize_response_guardrail(
        "streaming-null-fallback-redacted",
        1,
        Arc::new(|_response, _context| Box::pin(async { Ok(Some(redacted_response())) })),
    )
    .unwrap();
    runtime.block_on(async {
        let mut stream = llm_stream_call_execute(
            LlmStreamCallExecuteParams::builder()
                .name("streaming-null-fallback-redacted")
                .request(request())
                .func(Arc::new(|_request| {
                    Box::pin(async { Ok(LlmJsonStream::new(tokio_stream::empty())) })
                }))
                .collector(Box::new(|_chunk| Ok(())))
                .finalizer(Box::new(|| Json::Null))
                .data(secret_response())
                .response_codec(Arc::new(OpenAIChatCodec))
                .build(),
        )
        .await
        .unwrap();
        while let Some(chunk) = stream.next().await {
            chunk.unwrap();
        }
    });
    assert!(
        deregister_llm_sanitize_response_guardrail("streaming-null-fallback-redacted").unwrap()
    );

    runtime.block_on(async {
        let mut stream = llm_stream_call_execute(
            LlmStreamCallExecuteParams::builder()
                .name("streaming-null-without-fallback")
                .request(request())
                .func(Arc::new(|_request| {
                    Box::pin(async { Ok(LlmJsonStream::new(tokio_stream::empty())) })
                }))
                .collector(Box::new(|_chunk| Ok(())))
                .finalizer(Box::new(|| Json::Null))
                .build(),
        )
        .await
        .unwrap();
        while let Some(chunk) = stream.next().await {
            chunk.unwrap();
        }
    });

    flush_subscribers().unwrap();
    let captured = events.lock().unwrap();
    assert_eq!(*seen.lock().unwrap(), vec![secret_response()]);
    assert_eq!(captured.len(), 3);
    assert_eq!(captured[0].output(), Some(&Json::Null));
    assert!(captured[0].annotated_response().is_none());
    assert_eq!(captured[1].output(), Some(&redacted_response()));
    assert_eq!(
        captured[1].annotated_response().unwrap().message,
        Some(MessageContent::Text("[REDACTED]".to_string()))
    );
    assert!(captured[2].output().is_none());
    assert!(captured[2].annotated_response().is_none());
    assert!(
        captured
            .iter()
            .all(|event| !serde_json::to_string(event).unwrap().contains("SECRET"))
    );

    assert!(deregister_subscriber("streaming-null-fallback").unwrap());
}

#[test]
fn freshness_culls_annotations_and_repeated_compactions_are_idempotent() {
    let _guard = lock_global_runtime();
    reset_global();
    set_thread_scope_stack(create_scope_stack());

    let raw_request = multi_turn_request();
    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    register_subscriber(
        "freshness-culling",
        Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let emit = |name| {
        llm_call(
            LlmCallParams::builder()
                .name(name)
                .request(&raw_request)
                .annotated_request(multi_turn_annotation())
                .build(),
        )
        .unwrap();
    };
    emit("fresh-start");
    emit("stale-start");
    // PreCompact and PostCompact both normalize to this canonical mark.
    emit_compaction();
    emit_compaction();
    emit("post-compaction-start");
    emit("post-compaction-stale");

    flush_subscribers().unwrap();
    assert!(deregister_subscriber("freshness-culling").unwrap());
    let starts = events
        .lock()
        .unwrap()
        .iter()
        .filter(|event| event.scope_category() == Some(ScopeCategory::Start))
        .map(|event| {
            assert_eq!(event.input().unwrap()["content"], raw_request.content);
            (
                event.name().to_string(),
                event.annotated_request().unwrap().messages.clone(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(starts[0].0, "fresh-start");
    assert_eq!(starts[0].1.len(), 4);
    assert_eq!(starts[1].0, "stale-start");
    assert_eq!(starts[1].1.len(), 2);
    assert!(matches!(starts[1].1[0], Message::System { .. }));
    assert!(matches!(starts[1].1[1], Message::User { .. }));
    assert_eq!(starts[2].0, "post-compaction-start");
    assert_eq!(starts[2].1.len(), 4);
    assert_eq!(starts[3].0, "post-compaction-stale");
    assert_eq!(starts[3].1.len(), 2);
}

#[test]
fn full_payload_policy_preserves_complete_repeated_request_history() {
    let _guard = lock_global_runtime();
    reset_global();
    set_thread_scope_stack(create_scope_stack());
    global_context()
        .write()
        .unwrap()
        .observability_full_payloads_enabled = true;

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    register_subscriber(
        "full-payloads",
        Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let request = multi_turn_request();
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        for name in ["full-payload-first", "full-payload-repeated"] {
            llm_call_execute(
                LlmCallExecuteParams::builder()
                    .name(name)
                    .request(request.clone())
                    .func(Arc::new(|_| Box::pin(async { Ok(json!({"done": true})) })))
                    .codec(Arc::new(OpenAIChatCodec))
                    .build(),
            )
            .await
            .unwrap();
        }
    });
    llm_call(
        LlmCallParams::builder()
            .name("full-payload-manual")
            .request(&request)
            .annotated_request(multi_turn_annotation())
            .build(),
    )
    .unwrap();

    flush_subscribers().unwrap();
    assert!(deregister_subscriber("full-payloads").unwrap());
    global_context()
        .write()
        .unwrap()
        .observability_full_payloads_enabled = false;

    let events = events.lock().unwrap();
    for name in [
        "full-payload-first",
        "full-payload-repeated",
        "full-payload-manual",
    ] {
        let start = events
            .iter()
            .find(|event| {
                event.name() == name && event.scope_category() == Some(ScopeCategory::Start)
            })
            .unwrap_or_else(|| panic!("missing LLM start event {name}"));
        assert_eq!(
            start.input().unwrap()["content"]["messages"]
                .as_array()
                .unwrap()
                .len(),
            4
        );
        assert_eq!(start.annotated_request().unwrap().messages.len(), 4);
    }
}

#[test]
fn request_projection_handles_history_edge_cases() {
    let project = |messages: Json| {
        let annotated: AnnotatedLlmRequest =
            serde_json::from_value(json!({"messages": messages})).unwrap();
        let mut annotated = Some(Arc::new(annotated));
        let mut request = LlmRequest {
            headers: serde_json::Map::new(),
            content: Json::Null,
        };
        project_llm_request_to_current_user_turn(&mut request, &mut annotated, None);
        annotated.unwrap().messages.clone()
    };

    assert!(project(json!([])).is_empty());

    let no_user = project(json!([
        {"role": "assistant", "content": "tool request"},
        {"role": "tool", "tool_call_id": "call-1", "content": "tool result"}
    ]));
    assert_eq!(no_user.len(), 1);
    assert!(matches!(no_user[0], Message::Tool { .. }));

    let instructions_only = project(json!([
        {"role": "system", "content": "first instruction"},
        {"role": "system", "content": "second instruction"}
    ]));
    assert_eq!(instructions_only.len(), 2);
    assert!(
        instructions_only
            .iter()
            .all(|message| matches!(message, Message::System { .. }))
    );

    let current_turn = project(json!([
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
        {"role": "user", "content": "latest question"},
        {"role": "assistant", "content": null, "tool_calls": [{
            "id": "call-1",
            "type": "function",
            "function": {"name": "search", "arguments": "{}"}
        }]},
        {"role": "tool", "tool_call_id": "call-1", "content": "latest result"}
    ]));
    assert_eq!(current_turn.len(), 4);
    assert!(matches!(current_turn[0], Message::System { .. }));
    assert!(matches!(current_turn[1], Message::User { .. }));
    assert!(matches!(current_turn[2], Message::Assistant { .. }));
    assert!(matches!(current_turn[3], Message::Tool { .. }));

    let original = Arc::new(
        serde_json::from_value::<AnnotatedLlmRequest>(json!({
            "messages": [{"role": "system", "content": "instructions"}]
        }))
        .unwrap(),
    );
    let mut annotated = Some(original.clone());
    let mut request = LlmRequest {
        headers: serde_json::Map::new(),
        content: Json::Null,
    };
    project_llm_request_to_current_user_turn(&mut request, &mut annotated, Some(&OpenAIChatCodec));
    assert!(Arc::ptr_eq(annotated.as_ref().unwrap(), &original));
}

#[test]
fn managed_and_streaming_calls_cull_event_inputs_and_annotations_with_real_codec() {
    let _guard = lock_global_runtime();
    reset_global();
    set_thread_scope_stack(create_scope_stack());

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    register_subscriber(
        "managed-streaming-freshness",
        Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let request = multi_turn_request();
    let codec: Arc<dyn LlmCodec> = Arc::new(OpenAIChatCodec);
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        for name in ["managed-fresh", "managed-stale"] {
            let response = llm_call_execute(
                LlmCallExecuteParams::builder()
                    .name(name)
                    .request(request.clone())
                    .func(Arc::new(|request| {
                        Box::pin(async move {
                            Ok(json!({
                                "provider_message_count": request.content["messages"]
                                    .as_array()
                                    .unwrap()
                                    .len()
                            }))
                        })
                    }))
                    .codec(codec.clone())
                    .build(),
            )
            .await
            .unwrap();
            assert_eq!(response["provider_message_count"], 4);
        }

        emit_compaction();
        for name in ["stream-fresh", "stream-stale"] {
            let mut stream = llm_stream_call_execute(
                LlmStreamCallExecuteParams::builder()
                    .name(name)
                    .request(request.clone())
                    .func(Arc::new(|request| {
                        Box::pin(async move {
                            assert_eq!(request.content["messages"].as_array().unwrap().len(), 4);
                            Ok(LlmJsonStream::new(tokio_stream::iter(vec![Ok(json!({
                                "chunk": true
                            }))])))
                        })
                    }))
                    .collector(Box::new(|_| Ok(())))
                    .finalizer(Box::new(|| json!({"done": true})))
                    .codec(codec.clone())
                    .build(),
            )
            .await
            .unwrap();
            while let Some(chunk) = stream.next().await {
                chunk.unwrap();
            }
        }
    });

    flush_subscribers().unwrap();
    assert!(deregister_subscriber("managed-streaming-freshness").unwrap());
    let events = events.lock().unwrap();
    for (name, expected_messages) in [
        ("managed-fresh", 4),
        ("managed-stale", 2),
        ("stream-fresh", 4),
        ("stream-stale", 2),
    ] {
        let start = events
            .iter()
            .find(|event| {
                event.name() == name && event.scope_category() == Some(ScopeCategory::Start)
            })
            .unwrap_or_else(|| panic!("missing LLM start event {name}"));
        assert_eq!(
            start.input().unwrap()["content"]["messages"]
                .as_array()
                .unwrap()
                .len(),
            expected_messages,
            "unexpected raw event history for {name}"
        );
        assert_eq!(
            start.annotated_request().unwrap().messages.len(),
            expected_messages,
            "unexpected annotation history for {name}"
        );
    }
}

#[test]
fn projection_encode_failures_do_not_block_managed_or_streaming_calls() {
    let _guard = lock_global_runtime();
    reset_global();
    set_thread_scope_stack(create_scope_stack());

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    register_subscriber(
        "projection-encode-failure",
        Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let projection_attempts = Arc::new(AtomicUsize::new(0));
    let codec: Arc<dyn LlmCodec> = Arc::new(ProjectionFailingCodec {
        projection_attempts: projection_attempts.clone(),
    });
    let request = multi_turn_request();
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        for name in ["managed-encode-fresh", "managed-encode-stale"] {
            let response = llm_call_execute(
                LlmCallExecuteParams::builder()
                    .name(name)
                    .request(request.clone())
                    .func(Arc::new(|request| {
                        Box::pin(async move {
                            Ok(json!({
                                "provider_message_count": request.content["messages"]
                                    .as_array()
                                    .unwrap()
                                    .len()
                            }))
                        })
                    }))
                    .codec(codec.clone())
                    .build(),
            )
            .await
            .unwrap();
            assert_eq!(response["provider_message_count"], 4);
        }

        emit_compaction();
        for name in ["stream-encode-fresh", "stream-encode-stale"] {
            let mut stream = llm_stream_call_execute(
                LlmStreamCallExecuteParams::builder()
                    .name(name)
                    .request(request.clone())
                    .func(Arc::new(|request| {
                        Box::pin(async move {
                            assert_eq!(request.content["messages"].as_array().unwrap().len(), 4);
                            Ok(LlmJsonStream::new(tokio_stream::iter(vec![Ok(json!({
                                "chunk": true
                            }))])))
                        })
                    }))
                    .collector(Box::new(|_| Ok(())))
                    .finalizer(Box::new(|| json!({"done": true})))
                    .codec(codec.clone())
                    .build(),
            )
            .await
            .unwrap();
            while let Some(chunk) = stream.next().await {
                chunk.unwrap();
            }
        }
    });

    flush_subscribers().unwrap();
    assert_eq!(projection_attempts.load(Ordering::Relaxed), 2);
    assert!(deregister_subscriber("projection-encode-failure").unwrap());
    let events = events.lock().unwrap();
    for name in [
        "managed-encode-fresh",
        "managed-encode-stale",
        "stream-encode-fresh",
        "stream-encode-stale",
    ] {
        let start = events
            .iter()
            .find(|event| {
                event.name() == name && event.scope_category() == Some(ScopeCategory::Start)
            })
            .unwrap_or_else(|| panic!("missing LLM start event {name}"));
        assert_eq!(
            start.input().unwrap()["content"]["messages"]
                .as_array()
                .unwrap()
                .len(),
            4,
            "encode failure should preserve the full event input for {name}"
        );
        assert_eq!(
            start.annotated_request().unwrap().messages.len(),
            4,
            "encode failure should preserve the full annotation for {name}"
        );
    }
}

#[test]
fn nested_agents_track_freshness_independently_end_to_end() {
    let _guard = lock_global_runtime();
    reset_global();
    set_thread_scope_stack(create_scope_stack());

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    register_subscriber(
        "nested-agent-freshness",
        Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let request = multi_turn_request();
    let emit = |name| {
        llm_call(
            LlmCallParams::builder()
                .name(name)
                .request(&request)
                .annotated_request(multi_turn_annotation())
                .build(),
        )
        .unwrap();
    };

    let parent = push_scope(
        PushScopeParams::builder()
            .name("parent-agent")
            .scope_type(ScopeType::Agent)
            .build(),
    )
    .unwrap();
    emit("parent-fresh");
    emit("parent-stale");

    let child = push_scope(
        PushScopeParams::builder()
            .name("child-agent")
            .scope_type(ScopeType::Agent)
            .build(),
    )
    .unwrap();
    emit("child-fresh");
    emit("child-stale");
    emit_compaction();
    emit_compaction();
    emit("child-after-compaction");
    emit("child-after-compaction-stale");
    pop_scope(PopScopeParams::builder().handle_uuid(&child.uuid).build()).unwrap();

    emit("parent-after-child-compaction");
    emit_compaction();
    emit("parent-after-compaction");
    pop_scope(PopScopeParams::builder().handle_uuid(&parent.uuid).build()).unwrap();

    flush_subscribers().unwrap();
    assert!(deregister_subscriber("nested-agent-freshness").unwrap());
    let events = events.lock().unwrap();
    for (name, expected_messages) in [
        ("parent-fresh", 4),
        ("parent-stale", 2),
        ("child-fresh", 4),
        ("child-stale", 2),
        ("child-after-compaction", 4),
        ("child-after-compaction-stale", 2),
        ("parent-after-child-compaction", 2),
        ("parent-after-compaction", 4),
    ] {
        let start = events
            .iter()
            .find(|event| {
                event.name() == name && event.scope_category() == Some(ScopeCategory::Start)
            })
            .unwrap_or_else(|| panic!("missing LLM start event {name}"));
        assert_eq!(
            start.annotated_request().unwrap().messages.len(),
            expected_messages,
            "unexpected annotation history for {name}"
        );
    }
}

#[test]
fn non_agent_scopes_share_the_implicit_root_freshness_budget() {
    let _guard = lock_global_runtime();
    reset_global();
    set_thread_scope_stack(create_scope_stack());

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    register_subscriber(
        "implicit-root-freshness",
        Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let request = multi_turn_request();
    for (scope_name, scope_type, event_name) in [
        ("request-a", ScopeType::Custom, "root-fresh"),
        ("request-b", ScopeType::Function, "root-stale"),
    ] {
        let scope = push_scope(
            PushScopeParams::builder()
                .name(scope_name)
                .scope_type(scope_type)
                .build(),
        )
        .unwrap();
        llm_call(
            LlmCallParams::builder()
                .name(event_name)
                .request(&request)
                .annotated_request(multi_turn_annotation())
                .build(),
        )
        .unwrap();
        pop_scope(PopScopeParams::builder().handle_uuid(&scope.uuid).build()).unwrap();
    }

    flush_subscribers().unwrap();
    assert!(deregister_subscriber("implicit-root-freshness").unwrap());
    let events = events.lock().unwrap();
    for (name, expected_messages) in [("root-fresh", 4), ("root-stale", 2)] {
        let start = events
            .iter()
            .find(|event| {
                event.name() == name && event.scope_category() == Some(ScopeCategory::Start)
            })
            .unwrap_or_else(|| panic!("missing LLM start event {name}"));
        assert_eq!(
            start.annotated_request().unwrap().messages.len(),
            expected_messages,
            "non-agent scopes should inherit the implicit root freshness for {name}"
        );
    }
}

#[test]
fn concurrent_starts_consume_freshness_exactly_once_without_ordering() {
    let _guard = lock_global_runtime();
    reset_global();

    let shared_stack = create_scope_stack();
    set_thread_scope_stack(shared_stack.clone());
    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = events.clone();
    register_subscriber(
        "concurrent-freshness",
        Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let barrier = Arc::new(Barrier::new(3));
    let mut workers = Vec::new();
    for name in ["concurrent-a", "concurrent-b"] {
        let stack = shared_stack.clone();
        let barrier = barrier.clone();
        workers.push(std::thread::spawn(move || {
            set_thread_scope_stack(stack);
            barrier.wait();
            let request = multi_turn_request();
            llm_call(
                LlmCallParams::builder()
                    .name(name)
                    .request(&request)
                    .annotated_request(multi_turn_annotation())
                    .build(),
            )
            .unwrap();
        }));
    }
    barrier.wait();
    for worker in workers {
        worker.join().unwrap();
    }

    flush_subscribers().unwrap();
    assert!(deregister_subscriber("concurrent-freshness").unwrap());
    let events = events.lock().unwrap();
    let mut history_lengths = events
        .iter()
        .filter(|event| {
            event.name().starts_with("concurrent-")
                && event.scope_category() == Some(ScopeCategory::Start)
        })
        .map(|event| event.annotated_request().unwrap().messages.len())
        .collect::<Vec<_>>();
    history_lengths.sort_unstable();
    assert_eq!(history_lengths, vec![2, 4]);
}

#[test]
fn rejected_optimization_mark_queue_keeps_cursor_and_summary_evidence() {
    let _guard = lock_global_runtime();
    reset_global();

    let handle = LlmHandle::builder().name("queue-rejection-test").build();
    assert!(
        handle
            .optimization_recorder
            .record(LlmOptimizationContribution::new("test", "queue_rejection"))
    );
    emit_optimization_marks_with(&handle, &[], Some, |_event, _subscribers| false);
    assert_eq!(handle.optimization_recorder.unemitted().len(), 1);

    let summary = finalize_optimization_summary(
        &handle.optimization_recorder,
        None,
        None,
        &PricingResolver::default(),
    )
    .expect("queue rejection must not discard close-time evidence");
    assert_eq!(summary.contributions.len(), 1);
    assert_eq!(summary.contributions[0].producer, "test");
}

#[test]
fn unavailable_runtime_owner_skips_optimization_mark_delivery() {
    let _guard = lock_global_runtime();
    reset_global();
    crate::shared_runtime::initialize_shared_runtime_binding("python").unwrap();
    let owner = format!(
        "pid={};binding=rust;version={}",
        std::process::id(),
        env!("CARGO_PKG_VERSION").split('.').next().unwrap()
    );
    // SAFETY: The runtime-owner test mutex serializes this process-global test variable.
    unsafe { std::env::set_var("NEMO_RELAY_RUNTIME_OWNER", owner) };

    let handle = LlmHandle::builder().name("owner-unavailable").build();
    assert!(
        handle
            .optimization_recorder
            .record(LlmOptimizationContribution::new(
                "test",
                "owner_unavailable"
            ))
    );
    emit_optimization_marks_with(&handle, &[], Some, |_event, _subscribers| {
        panic!("an unavailable runtime owner must skip mark delivery")
    });
    assert_eq!(handle.optimization_recorder.unemitted().len(), 1);

    reset_global();
}

#[test]
fn unavailable_mark_sanitizer_does_not_acknowledge_the_delivery_cursor() {
    let _guard = lock_global_runtime();
    reset_global();

    let handle = LlmHandle::builder().name("sanitizer-unavailable").build();
    assert!(
        handle
            .optimization_recorder
            .record(LlmOptimizationContribution::new("test", "sanitize_retry"))
    );
    handle.optimization_recorder.close_for_finalization(None);
    emit_optimization_marks_with(
        &handle,
        &[],
        |_event| None,
        |_event, _subscribers| panic!("unavailable sanitization must not enqueue"),
    );
    assert_eq!(handle.optimization_recorder.unemitted().len(), 1);

    emit_optimization_marks_with(&handle, &[], Some, |_event, _subscribers| true);
    assert!(handle.optimization_recorder.unemitted().is_empty());
}

#[test]
fn manual_optimization_mark_snapshot_failure_publishes_fail_open() {
    let _guard = lock_global_runtime();
    reset_global();
    let scope_stack = create_scope_stack();
    set_thread_scope_stack(scope_stack.clone());
    let handle = LlmHandle::builder().name("poisoned-mark-snapshot").build();
    assert!(
        handle
            .optimization_recorder
            .record(LlmOptimizationContribution::new(
                "test",
                "snapshot_fail_open"
            ))
    );
    std::thread::spawn(move || {
        let _guard = scope_stack.write().unwrap();
        panic!("poison the captured scope stack");
    })
    .join()
    .unwrap_err();

    let events = Arc::new(Mutex::new(Vec::new()));
    let captured = events.clone();
    let subscribers = vec![Arc::new(move |event: &Event| {
        captured.lock().unwrap().push(event.clone());
    }) as crate::api::runtime::EventSubscriberFn];
    enqueue_optimization_marks(&handle, &subscribers);
    flush_subscribers().unwrap();

    assert_eq!(events.lock().unwrap().len(), 1);
    assert!(handle.optimization_recorder.unemitted().is_empty());
    set_thread_scope_stack(create_scope_stack());
}

#[test]
fn close_boundary_freezes_identical_mark_and_summary_contributions() {
    let _guard = lock_global_runtime();
    reset_global();

    let handle = LlmHandle::builder().name("close-boundary").build();
    assert!(
        handle
            .optimization_recorder
            .record(LlmOptimizationContribution::new(
                "accepted",
                "close_boundary"
            ))
    );
    assert!(handle.optimization_recorder.close_for_finalization(None));
    assert!(
        !handle
            .optimization_recorder
            .record(LlmOptimizationContribution::new("late", "close_boundary"))
    );

    let mut marks = Vec::new();
    emit_optimization_marks_with(&handle, &[], Some, |event, _subscribers| {
        marks.push(event.clone());
        true
    });
    let summary = finalize_optimization_summary(
        &handle.optimization_recorder,
        None,
        None,
        &PricingResolver::default(),
    )
    .unwrap();
    assert_eq!(marks.len(), 1);
    assert_eq!(summary.contributions.len(), 1);
    assert_eq!(marks[0].data().unwrap()["producer"], "accepted");
    assert_eq!(
        marks[0].data().unwrap()["id"],
        json!(summary.contributions[0].id.unwrap())
    );
}

#[test]
fn failed_managed_calls_sanitize_fallback_end_data() {
    let _guard = lock_global_runtime();
    reset_global();
    set_thread_scope_stack(create_scope_stack());

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let captured = Arc::clone(&events);
    register_subscriber(
        "failed-managed-call-sanitization",
        Arc::new(move |event| {
            if event.scope_category() == Some(ScopeCategory::End) {
                captured.lock().unwrap().push(event.clone());
            }
        }),
    )
    .unwrap();

    let seen = Arc::new(Mutex::new(Vec::new()));
    let sanitizer_inputs = Arc::clone(&seen);
    register_llm_sanitize_response_guardrail(
        "failed-managed-call-sanitization",
        1,
        Arc::new(move |response, context| {
            let codec = context.codec().clone();
            let sanitizer_inputs = Arc::clone(&sanitizer_inputs);
            Box::pin(async move {
                sanitizer_inputs.lock().unwrap().push((response, codec));
                Ok(Some(redacted_response()))
            })
        }),
    )
    .unwrap();

    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        let buffered_error = llm_call_execute(
            LlmCallExecuteParams::builder()
                .name("failed-buffered-call")
                .request(request())
                .func(Arc::new(|_request| {
                    Box::pin(async { Err(FlowError::Internal("buffered boom".to_string())) })
                }))
                .data(secret_response())
                .response_codec(Arc::new(OpenAIChatCodec))
                .build(),
        )
        .await
        .unwrap_err();
        assert!(buffered_error.to_string().contains("buffered boom"));

        let stream_error = match llm_stream_call_execute(
            LlmStreamCallExecuteParams::builder()
                .name("failed-stream-call")
                .request(request())
                .func(Arc::new(|_request| {
                    Box::pin(async { Err(FlowError::Internal("stream setup boom".to_string())) })
                }))
                .collector(Box::new(|_chunk| Ok(())))
                .finalizer(Box::new(|| Json::Null))
                .data(secret_response())
                .response_codec(Arc::new(OpenAIChatCodec))
                .build(),
        )
        .await
        {
            Ok(_) => panic!("stream setup should fail"),
            Err(error) => error,
        };
        assert!(stream_error.to_string().contains("stream setup boom"));
    });

    flush_subscribers().unwrap();
    assert!(
        deregister_llm_sanitize_response_guardrail("failed-managed-call-sanitization").unwrap()
    );
    assert!(deregister_subscriber("failed-managed-call-sanitization").unwrap());

    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 2);
    assert!(seen.iter().all(|(response, codec)| {
        response == &secret_response()
            && codec == &LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OpenAiChat)
    }));

    let events = events.lock().unwrap();
    assert_eq!(events.len(), 2);
    assert!(events.iter().all(|event| {
        event.output() == Some(&redacted_response())
            && event.annotated_response().is_none()
            && !serde_json::to_string(event).unwrap().contains("SECRET")
    }));
}

#[test]
fn llm_call_execute_adds_otel_status_metadata_to_end_events() {
    let _guard = lock_global_runtime();
    reset_global();

    let captured_events = Arc::new(Mutex::new(Vec::<(String, Option<Json>)>::new()));
    let subscriber_events = captured_events.clone();
    register_subscriber(
        "llm-status-metadata",
        Arc::new(move |event| {
            if event.scope_category() == Some(ScopeCategory::End) {
                subscriber_events
                    .lock()
                    .unwrap()
                    .push((event.name().to_string(), event.metadata().cloned()));
            }
        }),
    )
    .unwrap();

    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        let response = llm_call_execute(
            LlmCallExecuteParams::builder()
                .name("llm-ok")
                .request(request())
                .func(Arc::new(|_request| {
                    Box::pin(async { Ok(json!({"ok": true})) })
                }))
                .metadata(json!({"caller": "llm-ok", "otel.status_code": "USER"}))
                .build(),
        )
        .await
        .unwrap();
        assert_eq!(response, json!({"ok": true}));

        let error = llm_call_execute(
            LlmCallExecuteParams::builder()
                .name("llm-error")
                .request(request())
                .func(Arc::new(|_request| {
                    Box::pin(async { Err(FlowError::Internal("llm boom".to_string())) })
                }))
                .metadata(json!({"caller": "llm-error"}))
                .build(),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("llm boom"));
    });

    flush_subscribers().unwrap();
    assert!(deregister_subscriber("llm-status-metadata").unwrap());

    let events = captured_events.lock().unwrap();
    let metadata_for = |name: &str| {
        events
            .iter()
            .find(|event| event.0 == name)
            .and_then(|event| event.1.as_ref())
            .unwrap_or_else(|| panic!("missing end event metadata for {name}"))
    };

    let success_metadata = metadata_for("llm-ok");
    assert_eq!(success_metadata["caller"], json!("llm-ok"));
    assert_eq!(success_metadata["otel.status_code"], json!("OK"));
    assert!(success_metadata.get("otel.status_description").is_none());

    let error_metadata = metadata_for("llm-error");
    assert_eq!(error_metadata["caller"], json!("llm-error"));
    assert_eq!(error_metadata["otel.status_code"], json!("ERROR"));
    assert_eq!(error_metadata["error.type"], json!("internal_error"));
    assert!(
        error_metadata["otel.status_description"]
            .as_str()
            .unwrap()
            .contains("llm boom")
    );
}

#[test]
fn llm_stream_call_execute_adds_otel_status_metadata_to_end_events() {
    let _guard = lock_global_runtime();
    reset_global();

    let captured_events = Arc::new(Mutex::new(Vec::<(String, Option<Json>)>::new()));
    let subscriber_events = captured_events.clone();
    register_subscriber(
        "llm-stream-status-metadata",
        Arc::new(move |event| {
            if event.scope_category() == Some(ScopeCategory::End) {
                subscriber_events
                    .lock()
                    .unwrap()
                    .push((event.name().to_string(), event.metadata().cloned()));
            }
        }),
    )
    .unwrap();

    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        let mut stream = llm_stream_call_execute(
            LlmStreamCallExecuteParams::builder()
                .name("llm-stream-ok")
                .request(request())
                .func(Arc::new(|_request| {
                    Box::pin(async {
                        Ok(LlmJsonStream::new(tokio_stream::iter(vec![Ok(json!({
                            "chunk": true
                        }))])))
                    })
                }))
                .collector(Box::new(|_chunk| Ok(())))
                .finalizer(Box::new(|| json!({"ok": true})))
                .metadata(json!({"caller": "llm-stream-ok", "otel.status_code": "USER"}))
                .build(),
        )
        .await
        .unwrap();

        while let Some(chunk) = stream.next().await {
            chunk.unwrap();
        }
    });

    flush_subscribers().unwrap();
    assert!(deregister_subscriber("llm-stream-status-metadata").unwrap());

    let events = captured_events.lock().unwrap();
    let success_metadata = events
        .iter()
        .find(|event| event.0 == "llm-stream-ok")
        .and_then(|event| event.1.as_ref())
        .unwrap_or_else(|| panic!("missing stream end event metadata"));
    assert_eq!(success_metadata["caller"], json!("llm-stream-ok"));
    assert_eq!(success_metadata["otel.status_code"], json!("OK"));
    assert!(success_metadata.get("otel.status_description").is_none());
}

#[test]
fn llm_stream_call_execute_adds_otel_error_metadata_to_failed_end_events() {
    let _guard = lock_global_runtime();
    reset_global();

    let captured_events = Arc::new(Mutex::new(Vec::<(String, Option<Json>)>::new()));
    let subscriber_events = captured_events.clone();
    register_subscriber(
        "llm-stream-error-status-metadata",
        Arc::new(move |event| {
            if event.scope_category() == Some(ScopeCategory::End) {
                subscriber_events
                    .lock()
                    .unwrap()
                    .push((event.name().to_string(), event.metadata().cloned()));
            }
        }),
    )
    .unwrap();

    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        let mut upstream_error_stream = llm_stream_call_execute(
            LlmStreamCallExecuteParams::builder()
                .name("llm-stream-upstream-error")
                .request(request())
                .func(Arc::new(|_request| {
                    Box::pin(async {
                        Ok(LlmJsonStream::new(tokio_stream::iter(vec![Err(
                            FlowError::Internal("stream boom".to_string()),
                        )])))
                    })
                }))
                .collector(Box::new(|_chunk| Ok(())))
                .finalizer(Box::new(|| json!({"partial": true})))
                .metadata(
                    json!({"caller": "llm-stream-upstream-error", "otel.status_code": "USER"}),
                )
                .build(),
        )
        .await
        .unwrap();
        let upstream_error = upstream_error_stream.next().await.unwrap().unwrap_err();
        assert!(upstream_error.to_string().contains("stream boom"));

        let mut collector_error_stream = llm_stream_call_execute(
            LlmStreamCallExecuteParams::builder()
                .name("llm-stream-collector-error")
                .request(request())
                .func(Arc::new(|_request| {
                    Box::pin(async {
                        Ok(LlmJsonStream::new(tokio_stream::iter(vec![Ok(json!({
                            "chunk": true
                        }))])))
                    })
                }))
                .collector(Box::new(|_chunk| {
                    Err(FlowError::Internal("collector boom".to_string()))
                }))
                .finalizer(Box::new(|| json!({"partial": true})))
                .metadata(
                    json!({"caller": "llm-stream-collector-error", "otel.status_code": "USER"}),
                )
                .build(),
        )
        .await
        .unwrap();
        let collector_error = collector_error_stream.next().await.unwrap().unwrap_err();
        assert!(collector_error.to_string().contains("collector boom"));
    });

    flush_subscribers().unwrap();
    assert!(deregister_subscriber("llm-stream-error-status-metadata").unwrap());

    let events = captured_events.lock().unwrap();
    let metadata_for = |name: &str| {
        events
            .iter()
            .find(|event| event.0 == name)
            .and_then(|event| event.1.as_ref())
            .unwrap_or_else(|| panic!("missing stream end event metadata for {name}"))
    };

    let upstream_error_metadata = metadata_for("llm-stream-upstream-error");
    assert_eq!(
        upstream_error_metadata["caller"],
        json!("llm-stream-upstream-error")
    );
    assert_eq!(upstream_error_metadata["otel.status_code"], json!("ERROR"));
    assert_eq!(
        upstream_error_metadata["error.type"],
        json!("internal_error")
    );
    assert!(
        upstream_error_metadata["otel.status_description"]
            .as_str()
            .unwrap()
            .contains("stream boom")
    );

    let collector_error_metadata = metadata_for("llm-stream-collector-error");
    assert_eq!(
        collector_error_metadata["caller"],
        json!("llm-stream-collector-error")
    );
    assert_eq!(collector_error_metadata["otel.status_code"], json!("ERROR"));
    assert_eq!(
        collector_error_metadata["error.type"],
        json!("internal_error")
    );
    assert!(
        collector_error_metadata["otel.status_description"]
            .as_str()
            .unwrap()
            .contains("collector boom")
    );
}
