// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn codex_session_event(session_id: &str, payload: Value, metadata: Value) -> SessionEvent {
    SessionEvent {
        session_id: session_id.into(),
        agent_kind: AgentKind::Codex,
        event_name: "SessionStart".into(),
        payload,
        metadata,
    }
}

fn thread_spawn(parent_thread_id: &str) -> Value {
    json!({
        "source": {
            "subagent": {
                "thread_spawn": {
                    "parent_thread_id": parent_thread_id,
                    "depth": 2,
                    "agent_nickname": "Curie",
                    "agent_role": "worker"
                }
            }
        }
    })
}

#[test]
fn prompt_cache_session_id_requires_codex_responses_metadata() {
    let body = json!({
        "prompt_cache_key": "thread-1",
        "client_metadata": { "x-codex-installation-id": "install-1" }
    });

    assert_eq!(
        prompt_cache_session_id(&body, GatewayRouteKind::OpenAiResponses).as_deref(),
        Some("thread-1")
    );
    assert_eq!(
        prompt_cache_session_id(&body, GatewayRouteKind::OpenAiChatCompletions),
        None
    );
    assert_eq!(
        prompt_cache_session_id(
            &json!({ "prompt_cache_key": "plain-cache" }),
            GatewayRouteKind::OpenAiResponses,
        ),
        None
    );
    assert_eq!(
        prompt_cache_session_id(
            &json!({
                "prompt_cache_key": "",
                "client_metadata": { "x-codex-installation-id": "install-1" }
            }),
            GatewayRouteKind::OpenAiResponses,
        ),
        None
    );
}

#[test]
fn chatgpt_backend_override_requires_jwt_and_missing_replacement_key() {
    let mut headers = axum::http::HeaderMap::new();
    headers.insert(
        "authorization",
        axum::http::HeaderValue::from_static("Bearer eyJhbGciOiJIUzI1NiJ9.deadbeef.signature"),
    );

    assert_eq!(
        chatgpt_upstream_url_if_needed(
            &headers,
            GatewayRouteKind::OpenAiResponses,
            "/v1/responses",
            false,
        )
        .as_deref(),
        Some("https://chatgpt.com/backend-api/codex/responses")
    );
    assert_eq!(
        chatgpt_upstream_url_if_needed(
            &headers,
            GatewayRouteKind::AnthropicMessages,
            "/v1/messages",
            false,
        ),
        None
    );
    assert_eq!(
        chatgpt_upstream_url_if_needed(
            &headers,
            GatewayRouteKind::OpenAiResponses,
            "/v1/responses",
            true,
        ),
        None
    );
}

#[test]
fn chatgpt_oauth_strip_preserves_provider_keys_and_non_openai_routes() {
    let mut jwt_headers = axum::http::HeaderMap::new();
    jwt_headers.insert(
        "authorization",
        axum::http::HeaderValue::from_static("Bearer eyJhbGciOiJIUzI1NiJ9.deadbeef.signature"),
    );
    let stripped =
        strip_chatgpt_oauth_for_openai_route(&jwt_headers, GatewayRouteKind::OpenAiResponses, true);
    assert!(stripped.get("authorization").is_none());

    let preserved_without_replacement = strip_chatgpt_oauth_for_openai_route(
        &jwt_headers,
        GatewayRouteKind::OpenAiResponses,
        false,
    );
    assert!(preserved_without_replacement.get("authorization").is_some());

    let preserved_non_openai = strip_chatgpt_oauth_for_openai_route(
        &jwt_headers,
        GatewayRouteKind::AnthropicMessages,
        true,
    );
    assert!(preserved_non_openai.get("authorization").is_some());

    let mut key_headers = axum::http::HeaderMap::new();
    key_headers.insert(
        "authorization",
        axum::http::HeaderValue::from_static("Bearer sk-real-provider-key"),
    );
    let preserved_key =
        strip_chatgpt_oauth_for_openai_route(&key_headers, GatewayRouteKind::OpenAiResponses, true);
    assert_eq!(
        preserved_key.get("authorization").unwrap(),
        "Bearer sk-real-provider-key"
    );
}

#[tokio::test]
async fn subagent_context_reads_payload_metadata_and_rejects_non_subagents() {
    let payload_event = codex_session_event("child", thread_spawn("parent"), json!({}));
    let payload_context = subagent_context(&payload_event).await.unwrap();
    assert_eq!(payload_context.parent_session_id, "parent");
    assert_eq!(payload_context.nickname.as_deref(), Some("Curie"));
    assert_eq!(payload_context.role.as_deref(), Some("worker"));
    assert_eq!(payload_context.depth.as_deref(), Some("2"));

    let metadata_event = codex_session_event("child", json!({}), thread_spawn("parent"));
    assert_eq!(
        subagent_context(&metadata_event)
            .await
            .unwrap()
            .parent_session_id,
        "parent"
    );

    let self_parent = codex_session_event("child", thread_spawn("child"), json!({}));
    assert!(subagent_context(&self_parent).await.is_none());

    let mut non_codex = codex_session_event("child", thread_spawn("parent"), json!({}));
    non_codex.agent_kind = AgentKind::ClaudeCode;
    assert!(subagent_context(&non_codex).await.is_none());
}

#[tokio::test]
async fn subagent_context_reads_first_transcript_line() {
    let temp = tempfile::tempdir().unwrap();
    let transcript = temp.path().join("rollout.jsonl");
    std::fs::write(
        &transcript,
        serde_json::to_string(&json!({
            "session_meta": {
                "payload": {
                    "source": {
                        "subagent": {
                            "thread_spawn": {
                                "parent_thread_id": "parent-from-transcript",
                                "agent_nickname": "Noether",
                                "agent_role": "explorer"
                            }
                        }
                    }
                }
            }
        }))
        .unwrap()
            + "\n"
            + r#"{"later":true}"#,
    )
    .unwrap();
    let event = codex_session_event("child", json!({ "transcript_path": transcript }), json!({}));

    let context = subagent_context(&event).await.unwrap();
    assert_eq!(context.parent_session_id, "parent-from-transcript");
    assert_eq!(context.nickname.as_deref(), Some("Noether"));
    assert_eq!(context.role.as_deref(), Some("explorer"));
}

#[tokio::test]
async fn subagent_metadata_start_event_and_alias_share_thread_fields() {
    let event = codex_session_event("child", thread_spawn("parent"), json!({ "keep": true }));
    let context = subagent_context(&event).await.unwrap();

    let metadata = augment_subagent_metadata(event.metadata.clone(), &context);
    assert_eq!(metadata["keep"], json!(true));
    assert_eq!(metadata["thread_source"], json!("subagent"));
    assert_eq!(metadata["codex_parent_thread_id"], json!("parent"));
    assert_eq!(metadata["codex_subagent_depth"], json!("2"));
    assert_eq!(metadata["agent_nickname"], json!("Curie"));
    assert_eq!(metadata["agent_role"], json!("worker"));

    let subagent_event = subagent_start_event(&event, &context);
    assert_eq!(subagent_event.session_id, "parent");
    assert_eq!(subagent_event.subagent_id, "child");
    assert_eq!(
        subagent_event.metadata["codex_subagent_session_id"],
        json!("child")
    );

    let alias = alias_for_child_session("child".into(), &context);
    assert_eq!(alias.parent_session_id, "parent");
    assert_eq!(alias.subagent_id, "child");
    assert_eq!(alias.metadata()["thread_source"], json!("subagent"));
    assert_eq!(alias.metadata()["codex_parent_thread_id"], json!("parent"));
    assert_eq!(
        alias.metadata()["codex_subagent_session_id"],
        json!("child")
    );
}

#[test]
fn llm_owner_metadata_filters_codex_debug_fields() {
    let scope_metadata = json!({
        "thread_source": "subagent",
        "codex_parent_thread_id": "parent",
        "codex_subagent_session_id": "child",
        "codex_subagent_depth": 1,
        "agent_nickname": "Ada",
        "agent_role": null,
        "transcript_path": "/tmp/transcript.jsonl",
        "unrelated": "skip"
    });

    let metadata = llm_owner_metadata(Some(&scope_metadata));
    assert_eq!(
        metadata,
        json!({
            "thread_source": "subagent",
            "codex_parent_thread_id": "parent",
            "codex_subagent_session_id": "child",
            "codex_subagent_depth": 1,
            "agent_nickname": "Ada"
        })
    );
    assert!(metadata.get("transcript_path").is_none());
    assert_eq!(
        llm_owner_metadata(Some(&json!({ "unrelated": true }))),
        Value::Null
    );
    assert_eq!(llm_owner_metadata(Some(&json!("scalar"))), Value::Null);
    assert_eq!(llm_owner_metadata(None), Value::Null);
}

#[test]
fn owns_only_openai_responses_gateway_provider() {
    assert!(owns_gateway_provider("openai.responses"));
    assert!(!owns_gateway_provider("openai.chat_completions"));
    assert!(!owns_gateway_provider("anthropic.messages"));
}
