// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for shared in the NeMo Relay core crate.

use super::*;
use std::sync::Arc;

use serde_json::{Map, json};

use crate::api::llm::{LlmRequest, LlmRequestInterceptOutcome};
use crate::api::registry::{deregister_llm_request_intercept, register_llm_request_intercept};
use crate::api::runtime::NemoRelayContextState;
use crate::api::runtime::global_context;
use crate::api::runtime::{create_scope_stack, set_thread_scope_stack};
use crate::api::scope::ScopeType;
use crate::api::scope::{pop_scope, push_scope};
use crate::codec::request::{AnnotatedLlmRequest, Message, MessageContent};
use crate::codec::traits::LlmCodec;
use crate::error::Result;

struct SharedTestCodec;

impl LlmCodec for SharedTestCodec {
    fn decode(&self, request: &LlmRequest) -> Result<AnnotatedLlmRequest> {
        Ok(AnnotatedLlmRequest {
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
            store: None,
            previous_response_id: None,
            truncation: None,
            reasoning: None,
            include: None,
            user: None,
            metadata: None,
            service_tier: None,
            parallel_tool_calls: None,
            max_output_tokens: None,
            max_tool_calls: None,
            top_logprobs: None,
            stream: None,
            extra: Map::new(),
        })
    }

    fn encode(&self, annotated: &AnnotatedLlmRequest, original: &LlmRequest) -> Result<LlmRequest> {
        let mut content = original.content.clone();
        content["encoded_model"] = json!(annotated.model.clone());
        let mut headers = original.headers.clone();
        headers.insert("x-codec-encoded".into(), json!(true));
        Ok(LlmRequest { headers, content })
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
        *state = NemoRelayContextState::new();
    }
    set_thread_scope_stack(create_scope_stack());
    let _ = deregister_llm_request_intercept("shared-none");
    let _ = deregister_llm_request_intercept("shared-codec");
}

#[test]
fn test_metadata_with_otel_status_only_describes_errors() {
    let success_metadata = metadata_with_otel_status(
        Some(json!({
            "caller": "shared-ok",
            "otel.status_description": "stale status detail"
        })),
        "OK",
        Some("success detail".into()),
    )
    .unwrap();

    assert_eq!(success_metadata["caller"], json!("shared-ok"));
    assert_eq!(success_metadata["otel.status_code"], json!("OK"));
    assert!(success_metadata.get("otel.status_description").is_none());

    let error_metadata = metadata_with_otel_status(
        Some(json!({"caller": "shared-error"})),
        "ERROR",
        Some("error detail".into()),
    )
    .unwrap();

    assert_eq!(error_metadata["caller"], json!("shared-error"));
    assert_eq!(error_metadata["otel.status_code"], json!("ERROR"));
    assert_eq!(
        error_metadata["otel.status_description"],
        json!("error detail")
    );
}

#[test]
fn test_resolve_parent_uuid_snapshot_and_runtime_owner_helpers() {
    let _guard = lock_runtime_owner();
    reset_global();

    ensure_runtime_owner().unwrap();

    let root = crate::api::runtime::task_scope_top();
    assert_eq!(resolve_parent_uuid(None), Some(root.uuid));

    let handle = push_scope(
        crate::api::scope::PushScopeParams::builder()
            .name("shared-parent")
            .scope_type(ScopeType::Agent)
            .build(),
    )
    .unwrap();
    assert_eq!(resolve_parent_uuid(Some(&handle)), Some(handle.uuid));

    let subscribers = snapshot_event_subscribers(vec![Arc::new(|_event| {})]).unwrap();
    assert_eq!(subscribers.len(), 1);

    pop_scope(
        crate::api::scope::PopScopeParams::builder()
            .handle_uuid(&handle.uuid)
            .build(),
    )
    .unwrap();
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
        Arc::new(|_name, mut request, annotated| {
            assert!(annotated.is_none());
            request.headers.insert("x-no-codec".into(), json!(true));
            let mut annotated = SharedTestCodec.decode(&request)?;
            annotated.model = Some("interceptor-model".into());
            Ok(LlmRequestInterceptOutcome::new(request, Some(annotated)))
        }),
    )
    .unwrap();

    let (request_without_codec, annotated_without_codec, pending_marks_without_codec, _) =
        run_request_intercepts_with_codec(
            "shared",
            LlmRequest {
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
    assert_eq!(
        annotated_without_codec
            .as_deref()
            .and_then(|annotated| annotated.model.as_deref()),
        Some("interceptor-model")
    );
    assert!(pending_marks_without_codec.is_empty());
    deregister_llm_request_intercept("shared-none").unwrap();

    register_llm_request_intercept(
        "shared-codec",
        1,
        false,
        Arc::new(|_name, mut request, annotated| {
            let mut annotated = annotated.expect("codec should provide annotated request");
            annotated.model = Some("intercepted-model".into());
            request.headers.insert("x-codec".into(), json!(true));
            Ok(LlmRequestInterceptOutcome::new(request, Some(annotated)))
        }),
    )
    .unwrap();

    let codec: Arc<dyn LlmCodec> = Arc::new(SharedTestCodec);
    let (request_with_codec, annotated_with_codec, pending_marks_with_codec, _) =
        run_request_intercepts_with_codec(
            "shared",
            LlmRequest {
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
        request_with_codec.headers.get("x-codec-encoded"),
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
    assert!(pending_marks_with_codec.is_empty());

    deregister_llm_request_intercept("shared-codec").unwrap();
    reset_global();
}

#[test]
fn test_run_request_intercepts_injects_dynamo_agent_lineage() {
    let _guard = lock_runtime_owner();
    reset_global();

    let parent = push_scope(
        crate::api::scope::PushScopeParams::builder()
            .name("parent-name")
            .scope_type(ScopeType::Agent)
            .metadata(json!({"session_id": "parent-session"}))
            .build(),
    )
    .unwrap();
    let turn = push_scope(
        crate::api::scope::PushScopeParams::builder()
            .name("codex-turn")
            .scope_type(ScopeType::Custom)
            .metadata(json!({
                "nemo_relay_scope_role": "turn",
                "session_id": "parent-session"
            }))
            .parent(&parent)
            .build(),
    )
    .unwrap();
    let child = push_scope(
        crate::api::scope::PushScopeParams::builder()
            .name("child-name")
            .scope_type(ScopeType::Agent)
            .metadata(json!({"codex_subagent_session_id": "child-session"}))
            .parent(&turn)
            .build(),
    )
    .unwrap();

    let (request, _, _, _) = run_request_intercepts_with_codec(
        "openai.responses",
        LlmRequest {
            headers: Map::new(),
            content: json!({"prompt": "hello"}),
        },
        None,
    )
    .unwrap();
    assert_eq!(
        request.headers.get(DYNAMO_SESSION_ID_HEADER_KEY),
        Some(&json!("child-session"))
    );
    assert_eq!(
        request.headers.get(DYNAMO_PARENT_SESSION_ID_HEADER_KEY),
        Some(&json!("parent-session"))
    );

    let duplicate_id_child = push_scope(
        crate::api::scope::PushScopeParams::builder()
            .name("duplicate-id-child")
            .scope_type(ScopeType::Agent)
            .metadata(json!({"session_id": "child-session"}))
            .parent(&child)
            .build(),
    )
    .unwrap();
    let (request_with_codec, _, _, _) = run_request_intercepts_with_codec(
        "openai.responses",
        LlmRequest {
            headers: Map::new(),
            content: json!({"prompt": "hello"}),
        },
        Some(Arc::new(SharedTestCodec)),
    )
    .unwrap();
    assert_eq!(
        request_with_codec.headers.get(DYNAMO_SESSION_ID_HEADER_KEY),
        Some(&json!("child-session"))
    );
    assert_eq!(
        request_with_codec
            .headers
            .get(DYNAMO_PARENT_SESSION_ID_HEADER_KEY),
        Some(&json!("child-session"))
    );
    assert_eq!(
        request_with_codec.headers.get("x-codec-encoded"),
        Some(&json!(true))
    );
    pop_scope(
        crate::api::scope::PopScopeParams::builder()
            .handle_uuid(&duplicate_id_child.uuid)
            .build(),
    )
    .unwrap();

    pop_scope(
        crate::api::scope::PopScopeParams::builder()
            .handle_uuid(&child.uuid)
            .build(),
    )
    .unwrap();

    let (request, _, _, _) = run_request_intercepts_with_codec(
        "openai.responses",
        LlmRequest {
            headers: Map::new(),
            content: json!({"prompt": "hello"}),
        },
        None,
    )
    .unwrap();
    assert_eq!(
        request.headers.get(DYNAMO_SESSION_ID_HEADER_KEY),
        Some(&json!("parent-session"))
    );
    assert!(
        !request
            .headers
            .contains_key(DYNAMO_PARENT_SESSION_ID_HEADER_KEY)
    );
    pop_scope(
        crate::api::scope::PopScopeParams::builder()
            .handle_uuid(&turn.uuid)
            .build(),
    )
    .unwrap();
    pop_scope(
        crate::api::scope::PopScopeParams::builder()
            .handle_uuid(&parent.uuid)
            .build(),
    )
    .unwrap();

    let custom_turn = push_scope(
        crate::api::scope::PushScopeParams::builder()
            .name("custom-turn-only")
            .scope_type(ScopeType::Custom)
            .metadata(json!({
                "nemo_relay_scope_role": "turn",
                "session_id": "custom-session"
            }))
            .build(),
    )
    .unwrap();
    let (request, _, _, _) = run_request_intercepts_with_codec(
        "openai.responses",
        LlmRequest {
            headers: Map::new(),
            content: json!({"prompt": "hello"}),
        },
        None,
    )
    .unwrap();
    assert!(!request.headers.contains_key(DYNAMO_SESSION_ID_HEADER_KEY));
    assert!(
        !request
            .headers
            .contains_key(DYNAMO_PARENT_SESSION_ID_HEADER_KEY)
    );
    pop_scope(
        crate::api::scope::PopScopeParams::builder()
            .handle_uuid(&custom_turn.uuid)
            .build(),
    )
    .unwrap();
    reset_global();
}
