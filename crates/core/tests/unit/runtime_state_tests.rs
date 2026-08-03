// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for runtime middleware snapshot chains.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde_json::{Map, json};

use super::*;
use crate::api::registry::{RegistryRecord, RequestIntercept};

#[tokio::test]
async fn sanitizer_snapshot_chains_fail_closed_on_callback_panics() {
    let event = Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .name("preserved-event")
            .data(json!({"event": "preserved"}))
            .metadata(json!({"metadata": "preserved"}))
            .build(),
        None,
        None,
    ));
    let event_sanitizer: EventSanitizeFn =
        Arc::new(|_, _| Box::pin(async { panic!("event sanitizer panic") }));
    let event_later_called = Arc::new(AtomicBool::new(false));
    let event_later_called_by_callback = Arc::clone(&event_later_called);
    let event_later_sanitizer: EventSanitizeFn = Arc::new(move |_, fields| {
        event_later_called_by_callback.store(true, Ordering::Release);
        Box::pin(async move { Ok(fields) })
    });
    let sanitized_event = NemoRelayContextState::event_sanitize_snapshot_chain(
        event.clone(),
        &[
            RegistryRecord::new("event-panic", 0, event_sanitizer),
            RegistryRecord::new("event-later", -1, event_later_sanitizer),
        ],
    )
    .await;
    assert_eq!(sanitized_event.data(), None);
    assert_eq!(sanitized_event.metadata(), None);
    assert!(!event_later_called.load(Ordering::Acquire));

    let tool_payload = json!({"tool": "preserved"});
    let tool_sanitizer: ToolSanitizeFn =
        Arc::new(|_, _| Box::pin(async { panic!("tool sanitizer panic") }));
    let tool_later_called = Arc::new(AtomicBool::new(false));
    let tool_later_called_by_callback = Arc::clone(&tool_later_called);
    let tool_later_sanitizer: ToolSanitizeFn = Arc::new(move |_, value| {
        tool_later_called_by_callback.store(true, Ordering::Release);
        Box::pin(async move { Ok(value) })
    });
    let tool_entries = vec![
        RegistryRecord::new("tool-panic", 0, tool_sanitizer),
        RegistryRecord::new("tool-later", -1, tool_later_sanitizer),
    ];
    assert_eq!(
        NemoRelayContextState::tool_sanitize_request_snapshot_chain(
            "tool",
            tool_payload.clone(),
            &tool_entries,
        )
        .await,
        None
    );
    assert!(!tool_later_called.load(Ordering::Acquire));
    let tool_response = json!({"tool_response": "preserved"});
    let tool_response_sanitizer: ToolSanitizeFn =
        Arc::new(|_, _| Box::pin(async { panic!("tool response sanitizer panic") }));
    let tool_response_later_called = Arc::new(AtomicBool::new(false));
    let tool_response_later_called_by_callback = Arc::clone(&tool_response_later_called);
    let tool_response_later_sanitizer: ToolSanitizeFn = Arc::new(move |_, value| {
        tool_response_later_called_by_callback.store(true, Ordering::Release);
        Box::pin(async move { Ok(value) })
    });
    assert_eq!(
        NemoRelayContextState::tool_sanitize_response_snapshot_chain(
            "tool",
            tool_response.clone(),
            &[
                RegistryRecord::new("tool-response-panic", 0, tool_response_sanitizer,),
                RegistryRecord::new("tool-response-later", -1, tool_response_later_sanitizer,)
            ],
        )
        .await,
        None
    );
    assert!(!tool_response_later_called.load(Ordering::Acquire));

    let request = LlmRequest {
        headers: Map::new(),
        content: json!({"llm": "preserved"}),
    };
    let llm_sanitizer: LlmSanitizeRequestFn =
        Arc::new(|_, _| Box::pin(async { panic!("LLM sanitizer panic") }));
    let llm_later_called = Arc::new(AtomicBool::new(false));
    let llm_later_called_by_callback = Arc::clone(&llm_later_called);
    let llm_later_sanitizer: LlmSanitizeRequestFn = Arc::new(move |request, _| {
        llm_later_called_by_callback.store(true, Ordering::Release);
        Box::pin(async move { Ok(Some(request)) })
    });
    let llm_entries = vec![
        RegistryRecord::new("llm-panic", 0, llm_sanitizer),
        RegistryRecord::new("llm-later", -1, llm_later_sanitizer),
    ];
    assert_eq!(
        NemoRelayContextState::llm_sanitize_request_snapshot_chain(
            request.clone(),
            LlmSanitizeRequestContext::default(),
            &llm_entries,
        )
        .await,
        None
    );
    assert!(!llm_later_called.load(Ordering::Acquire));
    let llm_response = json!({"llm_response": "preserved"});
    let llm_response_sanitizer: LlmSanitizeResponseFn =
        Arc::new(|_, _| Box::pin(async { panic!("LLM response sanitizer panic") }));
    let llm_response_later_called = Arc::new(AtomicBool::new(false));
    let llm_response_later_called_by_callback = Arc::clone(&llm_response_later_called);
    let llm_response_later_sanitizer: LlmSanitizeResponseFn = Arc::new(move |response, _| {
        llm_response_later_called_by_callback.store(true, Ordering::Release);
        Box::pin(async move { Ok(Some(response)) })
    });
    assert_eq!(
        NemoRelayContextState::llm_sanitize_response_snapshot_chain(
            llm_response.clone(),
            LlmSanitizeResponseContext::default(),
            &[
                RegistryRecord::new("llm-response-panic", 0, llm_response_sanitizer,),
                RegistryRecord::new("llm-response-later", -1, llm_response_later_sanitizer,)
            ],
        )
        .await,
        None
    );
    assert!(!llm_response_later_called.load(Ordering::Acquire));

    let tool_conditional: ToolConditionalFn =
        Arc::new(|_, _| Box::pin(async { panic!("tool conditional panic") }));
    let error = NemoRelayContextState::tool_conditional_execution_snapshot_chain(
        "tool",
        &tool_payload,
        &[RegistryRecord::new(
            "tool-conditional-panic",
            0,
            tool_conditional,
        )],
        &[],
        None,
        None,
    )
    .await
    .unwrap_err();
    assert!(matches!(
        error,
        FlowError::Internal(ref message) if message.contains("tool-conditional-panic")
    ));

    let llm_conditional: LlmConditionalFn =
        Arc::new(|_| Box::pin(async { panic!("LLM conditional panic") }));
    let error = NemoRelayContextState::llm_conditional_execution_snapshot_chain(
        &request,
        &[RegistryRecord::new(
            "llm-conditional-panic",
            0,
            llm_conditional,
        )],
        &[],
        None,
        None,
    )
    .await
    .unwrap_err();
    assert!(matches!(
        error,
        FlowError::Internal(ref message) if message.contains("llm-conditional-panic")
    ));

    let tool_intercept: ToolInterceptFn =
        Arc::new(|_, _| Box::pin(async { panic!("tool intercept panic") }));
    let error = NemoRelayContextState::tool_request_intercepts_snapshot_chain(
        "tool",
        tool_payload,
        &[RegistryRecord::new(
            "tool-intercept-panic",
            0,
            RequestIntercept::new(false, tool_intercept),
        )],
    )
    .await
    .unwrap_err();
    assert!(matches!(
        error,
        FlowError::Internal(ref message) if message.contains("tool-intercept-panic")
    ));

    let llm_intercept: LlmRequestInterceptFn =
        Arc::new(|_, _, _| Box::pin(async { panic!("LLM intercept panic") }));
    let error = NemoRelayContextState::llm_request_intercepts_snapshot_chain(
        "llm",
        request,
        None,
        &[RegistryRecord::new(
            "llm-intercept-panic",
            0,
            RequestIntercept::new(false, llm_intercept),
        )],
        false,
    )
    .await
    .unwrap_err();
    assert!(matches!(
        error,
        FlowError::Internal(ref message) if message.contains("llm-intercept-panic")
    ));
}
