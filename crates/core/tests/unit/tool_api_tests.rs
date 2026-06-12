// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for tool API lifecycle behavior.

#![allow(clippy::await_holding_lock)]

use std::sync::{Arc, Mutex};

use serde_json::json;

use super::{ToolCallExecuteParams, tool_call_execute};
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

fn lock_global_runtime() -> std::sync::MutexGuard<'static, ()> {
    crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner())
}

#[test]
fn tool_call_execute_adds_otel_status_metadata_to_end_events() {
    let _guard = lock_global_runtime();
    reset_global();

    let captured_events = Arc::new(Mutex::new(Vec::<(String, Option<Json>)>::new()));
    let subscriber_events = captured_events.clone();
    register_subscriber(
        "tool-status-metadata",
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
        let result = tool_call_execute(
            ToolCallExecuteParams::builder()
                .name("tool-ok")
                .args(json!({"value": 1}))
                .func(Arc::new(|_args| {
                    Box::pin(async { Ok(json!({"ok": true})) })
                }))
                .metadata(json!({"caller": "tool-ok", "otel.status_code": "USER"}))
                .build(),
        )
        .await
        .unwrap();
        assert_eq!(result, json!({"ok": true}));

        let error = tool_call_execute(
            ToolCallExecuteParams::builder()
                .name("tool-error")
                .args(json!({"value": 2}))
                .func(Arc::new(|_args| {
                    Box::pin(async { Err(FlowError::Internal("tool boom".to_string())) })
                }))
                .metadata(json!({"caller": "tool-error"}))
                .build(),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("tool boom"));
    });

    flush_subscribers().unwrap();
    assert!(deregister_subscriber("tool-status-metadata").unwrap());

    let events = captured_events.lock().unwrap();
    let metadata_for = |name: &str| {
        events
            .iter()
            .find(|event| event.0 == name)
            .and_then(|event| event.1.as_ref())
            .unwrap_or_else(|| panic!("missing end event metadata for {name}"))
    };

    let success_metadata = metadata_for("tool-ok");
    assert_eq!(success_metadata["caller"], json!("tool-ok"));
    assert_eq!(success_metadata["otel.status_code"], json!("OK"));
    assert!(success_metadata.get("otel.status_description").is_none());

    let error_metadata = metadata_for("tool-error");
    assert_eq!(error_metadata["caller"], json!("tool-error"));
    assert_eq!(error_metadata["otel.status_code"], json!("ERROR"));
    assert!(
        error_metadata["otel.status_description"]
            .as_str()
            .unwrap()
            .contains("tool boom")
    );
}
