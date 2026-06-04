// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for LLM API lifecycle behavior.

use std::sync::{Arc, Mutex};

use serde_json::json;

use super::{LlmCallExecuteParams, LlmRequest, llm_call_execute};
use crate::api::event::ScopeCategory;
use crate::api::runtime::{NemoRelayContextState, global_context};
use crate::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use crate::error::FlowError;
use crate::json::Json;

fn reset_global() {
    crate::shared_runtime::reset_runtime_owner_for_tests();
    let context = global_context();
    *context.write().unwrap() = NemoRelayContextState::new();
}

fn request() -> LlmRequest {
    LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({"messages": [], "model": "demo"}),
    }
}

#[test]
fn llm_call_execute_adds_otel_status_metadata_to_end_events() {
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
    assert!(success_metadata.get("otel.status_message").is_none());

    let error_metadata = metadata_for("llm-error");
    assert_eq!(error_metadata["caller"], json!("llm-error"));
    assert_eq!(error_metadata["otel.status_code"], json!("ERROR"));
    assert!(
        error_metadata["otel.status_message"]
            .as_str()
            .unwrap()
            .contains("llm boom")
    );
}
