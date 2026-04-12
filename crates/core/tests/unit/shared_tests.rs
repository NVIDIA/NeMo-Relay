// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::sync::Arc;

use serde_json::{Map, json};

use crate::api::registry::{deregister_llm_request_intercept, register_llm_request_intercept};
use crate::api::scope::{pop_scope, push_scope};
use crate::codec::request::{AnnotatedLLMRequest, Message, MessageContent};
use crate::codec::traits::LlmCodec;
use crate::context::global::global_context;
use crate::context::scope_stack::{create_scope_stack, set_thread_scope_stack};
use crate::context::state::NemoFlowContextState;
use crate::error::Result;
use crate::types::llm::LLMRequest;
use crate::types::scope::{ScopeAttributes, ScopeType};

struct SharedTestCodec;

impl LlmCodec for SharedTestCodec {
    fn decode(&self, request: &LLMRequest) -> Result<AnnotatedLLMRequest> {
        Ok(AnnotatedLLMRequest {
            messages: vec![Message::User {
                content: MessageContent::Text(
                    request.content["prompt"]
                        .as_str()
                        .unwrap_or_default()
                        .to_string(),
                ),
                name: None,
            }],
            model: Some("decoded-model".into()),
            params: None,
            tools: None,
            tool_choice: None,
            extra: Map::new(),
        })
    }

    fn encode(&self, annotated: &AnnotatedLLMRequest, original: &LLMRequest) -> Result<LLMRequest> {
        let mut content = original.content.clone();
        content["encoded_model"] = json!(annotated.model.clone());
        Ok(LLMRequest {
            headers: original.headers.clone(),
            content,
        })
    }
}

fn lock_runtime_owner() -> std::sync::MutexGuard<'static, ()> {
    crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner())
}

fn reset_global() {
    crate::shared_runtime::reset_runtime_owner_for_tests();
    {
        let ctx = global_context();
        let mut state = ctx.write().unwrap();
        *state = NemoFlowContextState::new();
    }
    set_thread_scope_stack(create_scope_stack());
    let _ = deregister_llm_request_intercept("shared-none");
    let _ = deregister_llm_request_intercept("shared-codec");
}

#[test]
fn test_resolve_parent_uuid_snapshot_and_runtime_owner_helpers() {
    let _guard = lock_runtime_owner();
    reset_global();

    ensure_runtime_owner().unwrap();

    let root = crate::context::scope_stack::task_scope_top();
    assert_eq!(resolve_parent_uuid(None), Some(root.uuid));

    let handle = push_scope(
        "shared-parent",
        ScopeType::Agent,
        None,
        ScopeAttributes::empty(),
        None,
        None,
    )
    .unwrap();
    assert_eq!(resolve_parent_uuid(Some(&handle)), Some(handle.uuid));

    let subscribers = snapshot_event_subscribers(vec![Arc::new(|_event| {})]).unwrap();
    assert_eq!(subscribers.len(), 1);

    pop_scope(&handle.uuid).unwrap();
    reset_global();
}

#[test]
fn test_run_request_intercepts_with_codec_none_and_codec_paths() {
    let _guard = lock_runtime_owner();
    reset_global();

    register_llm_request_intercept(
        "shared-none",
        1,
        false,
        Box::new(|_name, mut request, annotated| {
            assert!(annotated.is_none());
            request.headers.insert("x-no-codec".into(), json!(true));
            Ok((request, None))
        }),
    )
    .unwrap();

    let (request_without_codec, annotated_without_codec) = run_request_intercepts_with_codec(
        "shared",
        LLMRequest {
            headers: Map::new(),
            content: json!({"prompt": "hello"}),
        },
        None,
    )
    .unwrap();
    assert_eq!(
        request_without_codec.headers.get("x-no-codec"),
        Some(&json!(true))
    );
    assert!(annotated_without_codec.is_none());
    deregister_llm_request_intercept("shared-none").unwrap();

    register_llm_request_intercept(
        "shared-codec",
        1,
        false,
        Box::new(|_name, mut request, annotated| {
            let mut annotated = annotated.expect("codec should provide annotated request");
            annotated.model = Some("intercepted-model".into());
            request.headers.insert("x-codec".into(), json!(true));
            Ok((request, Some(annotated)))
        }),
    )
    .unwrap();

    let codec: Arc<dyn LlmCodec> = Arc::new(SharedTestCodec);
    let (request_with_codec, annotated_with_codec) = run_request_intercepts_with_codec(
        "shared",
        LLMRequest {
            headers: Map::new(),
            content: json!({"prompt": "hello"}),
        },
        Some(codec),
    )
    .unwrap();

    assert_eq!(
        request_with_codec.headers.get("x-codec"),
        Some(&json!(true))
    );
    assert_eq!(
        request_with_codec.content["encoded_model"],
        json!("intercepted-model")
    );
    assert_eq!(
        annotated_with_codec
            .as_deref()
            .and_then(|annotated| annotated.model.as_deref()),
        Some("intercepted-model")
    );

    deregister_llm_request_intercept("shared-codec").unwrap();
    reset_global();
}
