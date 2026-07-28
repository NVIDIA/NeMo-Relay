// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for runtime middleware snapshot chains.

use serde_json::{Map, json};

use super::*;
use crate::api::registry::{RegistryRecord, RequestIntercept};

#[tokio::test]
async fn middleware_snapshot_chains_contain_callback_panics() {
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
    let sanitized_event = NemoRelayContextState::event_sanitize_snapshot_chain(
        event.clone(),
        &[RegistryRecord::new("event-panic", 0, event_sanitizer)],
    )
    .await;
    assert_eq!(sanitized_event.data(), event.data());
    assert_eq!(sanitized_event.metadata(), event.metadata());

    let tool_payload = json!({"tool": "preserved"});
    let tool_sanitizer: ToolSanitizeFn =
        Arc::new(|_, _| Box::pin(async { panic!("tool sanitizer panic") }));
    let tool_entries = vec![RegistryRecord::new("tool-panic", 0, tool_sanitizer)];
    assert_eq!(
        NemoRelayContextState::tool_sanitize_request_snapshot_chain(
            "tool",
            tool_payload.clone(),
            &tool_entries,
        )
        .await,
        tool_payload
    );
    let tool_response = json!({"tool_response": "preserved"});
    let tool_response_sanitizer: ToolSanitizeFn =
        Arc::new(|_, _| Box::pin(async { panic!("tool response sanitizer panic") }));
    assert_eq!(
        NemoRelayContextState::tool_sanitize_response_snapshot_chain(
            "tool",
            tool_response.clone(),
            &[RegistryRecord::new(
                "tool-response-panic",
                0,
                tool_response_sanitizer,
            )],
        )
        .await,
        tool_response
    );

    let request = LlmRequest {
        headers: Map::new(),
        content: json!({"llm": "preserved"}),
    };
    let llm_sanitizer: LlmSanitizeRequestFn =
        Arc::new(|_, _| Box::pin(async { panic!("LLM sanitizer panic") }));
    let llm_entries = vec![RegistryRecord::new("llm-panic", 0, llm_sanitizer)];
    assert_eq!(
        NemoRelayContextState::llm_sanitize_request_snapshot_chain(
            request.clone(),
            LlmSanitizeRequestContext::default(),
            &llm_entries,
        )
        .await,
        Some(request.clone())
    );
    let llm_response = json!({"llm_response": "preserved"});
    let llm_response_sanitizer: LlmSanitizeResponseFn =
        Arc::new(|_, _| Box::pin(async { panic!("LLM response sanitizer panic") }));
    assert_eq!(
        NemoRelayContextState::llm_sanitize_response_snapshot_chain(
            llm_response.clone(),
            LlmSanitizeResponseContext::default(),
            &[RegistryRecord::new(
                "llm-response-panic",
                0,
                llm_response_sanitizer,
            )],
        )
        .await,
        Some(llm_response)
    );

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
