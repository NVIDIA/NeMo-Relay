// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::http::HeaderMap;
use nemo_flow::observability::atof::AtofExporterMode;
use serde_json::json;

use super::*;
use crate::config::{AtifExporterSettings, AtofExporterSettings, ExportersConfig};
use crate::model::{LlmEvent, LlmHintEvent, SessionEvent, ToolEvent};

#[tokio::test]
async fn nests_agent_subagent_and_tool_lifecycle() {
    let config = GatewayConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://127.0.0.1".into(),
        anthropic_base_url: "http://127.0.0.1".into(),
        exporters: ExportersConfig::default(),
        metadata: None,
        plugin_config: None,
    };
    let manager = SessionManager::new(config);
    let headers = HeaderMap::new();
    let events = vec![
        NormalizedEvent::AgentStarted(SessionEvent {
            session_id: "s1".into(),
            agent_kind: AgentKind::ClaudeCode,
            event_name: "SessionStart".into(),
            payload: json!({}),
            metadata: json!({}),
        }),
        NormalizedEvent::SubagentStarted(SubagentEvent {
            session_id: "s1".into(),
            agent_kind: AgentKind::ClaudeCode,
            event_name: "SubagentStart".into(),
            subagent_id: "worker-1".into(),
            payload: json!({}),
            metadata: json!({}),
        }),
        NormalizedEvent::ToolStarted(ToolEvent {
            session_id: "s1".into(),
            agent_kind: AgentKind::ClaudeCode,
            event_name: "PreToolUse".into(),
            tool_call_id: "t1".into(),
            tool_name: "Read".into(),
            subagent_id: Some("worker-1".into()),
            arguments: json!({ "file_path": "README.md" }),
            result: Value::Null,
            status: None,
            payload: json!({}),
            metadata: json!({}),
        }),
        NormalizedEvent::ToolEnded(ToolEvent {
            session_id: "s1".into(),
            agent_kind: AgentKind::ClaudeCode,
            event_name: "PostToolUse".into(),
            tool_call_id: "t1".into(),
            tool_name: "Read".into(),
            subagent_id: Some("worker-1".into()),
            arguments: Value::Null,
            result: json!({ "ok": true }),
            status: Some("success".into()),
            payload: json!({}),
            metadata: json!({}),
        }),
        NormalizedEvent::SubagentEnded(SubagentEvent {
            session_id: "s1".into(),
            agent_kind: AgentKind::ClaudeCode,
            event_name: "SubagentStop".into(),
            subagent_id: "worker-1".into(),
            payload: json!({}),
            metadata: json!({}),
        }),
        NormalizedEvent::AgentEnded(SessionEvent {
            session_id: "s1".into(),
            agent_kind: AgentKind::ClaudeCode,
            event_name: "SessionEnd".into(),
            payload: json!({}),
            metadata: json!({}),
        }),
    ];
    manager.apply_events(&headers, events).await.unwrap();
    assert!(manager.inner.lock().await.is_empty());
}

#[tokio::test]
async fn writes_atif_on_session_end_from_header_config() {
    let temp = tempfile::tempdir().unwrap();
    let config = GatewayConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://127.0.0.1".into(),
        anthropic_base_url: "http://127.0.0.1".into(),
        exporters: ExportersConfig::default(),
        metadata: None,
        plugin_config: None,
    };
    let manager = SessionManager::new(config);
    let mut headers = HeaderMap::new();
    headers.insert(
        "x-nemo-flow-atif-dir",
        temp.path().to_string_lossy().parse().unwrap(),
    );
    headers.insert(
        "x-nemo-flow-session-metadata",
        r#"{"team":"coverage"}"#.parse().unwrap(),
    );
    headers.insert("x-nemo-flow-gateway-mode", "required".parse().unwrap());

    manager
        .apply_events(
            &headers,
            vec![
                NormalizedEvent::AgentStarted(SessionEvent {
                    session_id: "atif-session".into(),
                    agent_kind: AgentKind::Codex,
                    event_name: "sessionStart".into(),
                    payload: json!({ "start": true }),
                    metadata: json!({ "agent": "codex" }),
                }),
                NormalizedEvent::PromptSubmitted(SessionEvent {
                    session_id: "atif-session".into(),
                    agent_kind: AgentKind::Codex,
                    event_name: "UserPromptSubmit".into(),
                    payload: json!({ "prompt": "hello" }),
                    metadata: json!({}),
                }),
                NormalizedEvent::AgentEnded(SessionEvent {
                    session_id: "atif-session".into(),
                    agent_kind: AgentKind::Codex,
                    event_name: "sessionEnd".into(),
                    payload: json!({ "done": true }),
                    metadata: json!({}),
                }),
            ],
        )
        .await
        .unwrap();

    let path = temp.path().join("atif-session.atif.json");
    let atif: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    assert_eq!(atif["agent"]["name"], json!("codex"));
}

#[tokio::test]
async fn writes_atof_with_configured_mode_and_filename_template() {
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("custom-atof-mode.jsonl");
    std::fs::write(&output, "{\"existing\":true}\n").unwrap();
    let config = GatewayConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://127.0.0.1".into(),
        anthropic_base_url: "http://127.0.0.1".into(),
        exporters: ExportersConfig {
            atof: AtofExporterSettings {
                dir: Some(temp.path().to_path_buf()),
                mode: AtofExporterMode::Overwrite,
                filename_template: "custom-{session_id}.jsonl".into(),
            },
            ..Default::default()
        },
        metadata: None,
        plugin_config: None,
    };
    let manager = SessionManager::new(config);

    manager
        .apply_events(
            &HeaderMap::new(),
            vec![
                NormalizedEvent::AgentStarted(SessionEvent {
                    session_id: "atof-mode".into(),
                    agent_kind: AgentKind::Codex,
                    event_name: "SessionStart".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::AgentEnded(SessionEvent {
                    session_id: "atof-mode".into(),
                    agent_kind: AgentKind::Codex,
                    event_name: "SessionEnd".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
            ],
        )
        .await
        .unwrap();

    let contents = std::fs::read_to_string(output).unwrap();
    assert!(!contents.contains("existing"));
    assert!(contents.contains("atof-mode"));
}

#[tokio::test]
async fn duplicate_agent_end_does_not_overwrite_atif_with_empty_session() {
    // Regression test: hermes-agent and other integrations can emit terminal hooks more than once
    // per session. Without idempotency in `end_agent`, the second AgentEnded would re-open an
    // empty agent scope via `ensure_agent_started`, close it, and `flush_observers` would write
    // an empty ATIF on top of the just-written real trajectory.
    let temp = tempfile::tempdir().unwrap();
    let config = GatewayConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://127.0.0.1".into(),
        anthropic_base_url: "http://127.0.0.1".into(),
        exporters: ExportersConfig {
            atif: AtifExporterSettings {
                dir: Some(temp.path().to_path_buf()),
            },
            ..Default::default()
        },
        metadata: None,
        plugin_config: None,
    };
    let manager = SessionManager::new(config);
    let headers = HeaderMap::new();

    manager
        .apply_events(
            &headers,
            vec![
                NormalizedEvent::AgentStarted(SessionEvent {
                    session_id: "dup-end".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "SessionStart".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::PromptSubmitted(SessionEvent {
                    session_id: "dup-end".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "UserPromptSubmit".into(),
                    payload: json!({ "prompt": "hello" }),
                    metadata: json!({}),
                }),
                NormalizedEvent::AgentEnded(SessionEvent {
                    session_id: "dup-end".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "SessionEnd".into(),
                    payload: json!({ "done": true }),
                    metadata: json!({}),
                }),
            ],
        )
        .await
        .unwrap();

    let path = temp.path().join("dup-end.atif.json");
    let first: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let first_steps = first["steps"].as_array().unwrap().len();
    assert!(
        first_steps > 0,
        "first AgentEnded should produce a non-empty ATIF"
    );

    // Second AgentEnded for the same session — must be a no-op, not overwrite with empty.
    manager
        .apply_events(
            &headers,
            vec![NormalizedEvent::AgentEnded(SessionEvent {
                session_id: "dup-end".into(),
                agent_kind: AgentKind::ClaudeCode,
                event_name: "SessionEnd".into(),
                payload: json!({ "done_again": true }),
                metadata: json!({}),
            })],
        )
        .await
        .unwrap();

    let second: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let second_steps = second["steps"].as_array().unwrap().len();
    assert_eq!(
        first_steps, second_steps,
        "duplicate AgentEnded must not change the ATIF step count"
    );
}

#[tokio::test]
async fn writes_hermes_api_hook_usage_to_atif_metrics() {
    let temp = tempfile::tempdir().unwrap();
    let config = GatewayConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://127.0.0.1".into(),
        anthropic_base_url: "http://127.0.0.1".into(),
        exporters: ExportersConfig::default(),
        metadata: None,
        plugin_config: None,
    };
    let manager = SessionManager::new(config);
    let mut headers = HeaderMap::new();
    headers.insert(
        "x-nemo-flow-atif-dir",
        temp.path().to_string_lossy().parse().unwrap(),
    );

    manager
        .apply_events(
            &headers,
            vec![
                NormalizedEvent::AgentStarted(SessionEvent {
                    session_id: "hermes-usage".into(),
                    agent_kind: AgentKind::Hermes,
                    event_name: "on_session_start".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::LlmStarted(LlmEvent {
                    session_id: "hermes-usage".into(),
                    agent_kind: AgentKind::Hermes,
                    event_name: "pre_api_request".into(),
                    api_call_id: "hermes-usage:task-1:1".into(),
                    provider: "custom".into(),
                    model_name: Some("qwen".into()),
                    request: json!({ "model": "qwen" }),
                    response: Value::Null,
                    metadata: json!({}),
                }),
                NormalizedEvent::LlmEnded(LlmEvent {
                    session_id: "hermes-usage".into(),
                    agent_kind: AgentKind::Hermes,
                    event_name: "post_api_request".into(),
                    api_call_id: "hermes-usage:task-1:1".into(),
                    provider: "custom".into(),
                    model_name: Some("qwen".into()),
                    request: json!({}),
                    response: json!({
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "prompt_tokens_details": { "cached_tokens": 3 }
                        }
                    }),
                    metadata: json!({}),
                }),
                NormalizedEvent::AgentEnded(SessionEvent {
                    session_id: "hermes-usage".into(),
                    agent_kind: AgentKind::Hermes,
                    event_name: "on_session_finalize".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
            ],
        )
        .await
        .unwrap();

    let path = temp.path().join("hermes-usage.atif.json");
    let atif: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    assert_eq!(atif["steps"][1]["metrics"]["prompt_tokens"], json!(10));
    assert_eq!(atif["steps"][1]["metrics"]["completion_tokens"], json!(5));
    assert_eq!(atif["steps"][1]["metrics"]["cached_tokens"], json!(3));
    assert_eq!(atif["final_metrics"]["total_prompt_tokens"], json!(10));
    assert_eq!(atif["final_metrics"]["total_completion_tokens"], json!(5));
    assert_eq!(atif["final_metrics"]["total_cached_tokens"], json!(3));
}

#[tokio::test]
async fn handles_out_of_order_subagent_and_tool_end_events() {
    let config = GatewayConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://127.0.0.1".into(),
        anthropic_base_url: "http://127.0.0.1".into(),
        exporters: ExportersConfig::default(),
        metadata: None,
        plugin_config: None,
    };
    let manager = SessionManager::new(config);
    let headers = HeaderMap::new();

    manager
        .apply_events(
            &headers,
            vec![
                NormalizedEvent::SubagentEnded(SubagentEvent {
                    session_id: "out-of-order".into(),
                    agent_kind: AgentKind::Cursor,
                    event_name: "subagentStop".into(),
                    subagent_id: "missing".into(),
                    payload: json!({ "reason": "missing-start" }),
                    metadata: json!({}),
                }),
                NormalizedEvent::ToolEnded(ToolEvent {
                    session_id: "out-of-order".into(),
                    agent_kind: AgentKind::Cursor,
                    event_name: "postToolUse".into(),
                    tool_call_id: "tool-without-start".into(),
                    tool_name: "Shell".into(),
                    subagent_id: None,
                    arguments: json!({ "cmd": "pwd" }),
                    result: json!({ "stdout": "/repo" }),
                    status: Some("success".into()),
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::AgentEnded(SessionEvent {
                    session_id: "out-of-order".into(),
                    agent_kind: AgentKind::Cursor,
                    event_name: "sessionEnd".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
            ],
        )
        .await
        .unwrap();

    assert!(manager.inner.lock().await.is_empty());
}

#[tokio::test]
async fn terminal_retry_for_unknown_session_is_ignored() {
    let temp = tempfile::tempdir().unwrap();
    let mut config = session_test_config();
    config.exporters.atif.dir = Some(temp.path().to_path_buf());
    let manager = SessionManager::new(config);

    manager
        .apply_events(
            &HeaderMap::new(),
            vec![NormalizedEvent::AgentEnded(SessionEvent {
                session_id: "retry-session".into(),
                agent_kind: AgentKind::Codex,
                event_name: "sessionEnd".into(),
                payload: json!({}),
                metadata: json!({}),
            })],
        )
        .await
        .unwrap();

    assert!(manager.inner.lock().await.is_empty());
    assert!(!temp.path().join("retry-session.atif.json").exists());
}

#[tokio::test]
async fn out_of_order_started_subagent_end_does_not_leak_scope() {
    let config = GatewayConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://127.0.0.1".into(),
        anthropic_base_url: "http://127.0.0.1".into(),
        exporters: ExportersConfig::default(),
        metadata: None,
        plugin_config: None,
    };
    let manager = SessionManager::new(config);
    let headers = HeaderMap::new();

    manager
        .apply_events(
            &headers,
            vec![
                NormalizedEvent::AgentStarted(SessionEvent {
                    session_id: "nested".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "SessionStart".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::SubagentStarted(SubagentEvent {
                    session_id: "nested".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "SubagentStart".into(),
                    subagent_id: "parent".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::SubagentStarted(SubagentEvent {
                    session_id: "nested".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "SubagentStart".into(),
                    subagent_id: "child".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::SubagentEnded(SubagentEvent {
                    session_id: "nested".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "SubagentStop".into(),
                    subagent_id: "parent".into(),
                    payload: json!({ "out_of_order": true }),
                    metadata: json!({}),
                }),
                NormalizedEvent::SubagentEnded(SubagentEvent {
                    session_id: "nested".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "SubagentStop".into(),
                    subagent_id: "child".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::AgentEnded(SessionEvent {
                    session_id: "nested".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "SessionEnd".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
            ],
        )
        .await
        .unwrap();

    assert!(manager.inner.lock().await.is_empty());
}

#[tokio::test]
async fn agent_end_closes_nested_active_subagents_lifo() {
    let config = GatewayConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://127.0.0.1".into(),
        anthropic_base_url: "http://127.0.0.1".into(),
        exporters: ExportersConfig::default(),
        metadata: None,
        plugin_config: None,
    };
    let manager = SessionManager::new(config);
    let headers = HeaderMap::new();

    manager
        .apply_events(
            &headers,
            vec![
                NormalizedEvent::AgentStarted(SessionEvent {
                    session_id: "cleanup".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "SessionStart".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::SubagentStarted(SubagentEvent {
                    session_id: "cleanup".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "SubagentStart".into(),
                    subagent_id: "parent".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::SubagentStarted(SubagentEvent {
                    session_id: "cleanup".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "SubagentStart".into(),
                    subagent_id: "child".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::AgentEnded(SessionEvent {
                    session_id: "cleanup".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "SessionEnd".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
            ],
        )
        .await
        .unwrap();

    assert!(manager.inner.lock().await.is_empty());
}

#[tokio::test]
async fn llm_lifecycle_starts_implicit_gateway_session() {
    let config = GatewayConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://127.0.0.1".into(),
        anthropic_base_url: "http://127.0.0.1".into(),
        exporters: ExportersConfig::default(),
        metadata: None,
        plugin_config: None,
    };
    let manager = SessionManager::new(config);
    let active = manager
        .start_llm(
            &HeaderMap::new(),
            LlmGatewayStart {
                session_id: Some("llm-session".into()),
                provider: "openai.responses".into(),
                model_name: Some("gpt-test".into()),
                subagent_id: None,
                conversation_id: None,
                generation_id: None,
                request_id: None,
                request: LlmRequest {
                    headers: Map::new(),
                    content: json!({ "model": "gpt-test", "input": "hello" }),
                },
                streaming: true,
                metadata: json!({ "gateway_path": "/v1/responses" }),
            },
        )
        .await
        .unwrap();
    manager
        .end_llm(
            active,
            json!({ "output_text": "hello" }),
            json!({ "http_status": 200 }),
        )
        .await
        .unwrap();

    let sessions = manager.inner.lock().await;
    assert!(sessions.contains_key("llm-session"));
}

#[tokio::test]
async fn agent_end_closes_in_flight_gateway_llm() {
    let temp = tempfile::tempdir().unwrap();
    let mut config = session_test_config();
    config.exporters.atif.dir = Some(temp.path().to_path_buf());
    let manager = SessionManager::new(config);
    let _active = manager
        .start_llm(
            &HeaderMap::new(),
            LlmGatewayStart {
                session_id: Some("gateway-cleanup".into()),
                provider: "openai.responses".into(),
                model_name: Some("gpt-test".into()),
                subagent_id: None,
                conversation_id: None,
                generation_id: None,
                request_id: None,
                request: LlmRequest {
                    headers: Map::new(),
                    content: json!({ "model": "gpt-test", "input": "hello" }),
                },
                streaming: true,
                metadata: json!({ "gateway_path": "/v1/responses" }),
            },
        )
        .await
        .unwrap();

    manager
        .apply_events(
            &HeaderMap::new(),
            vec![NormalizedEvent::AgentEnded(SessionEvent {
                session_id: "gateway-cleanup".into(),
                agent_kind: AgentKind::Gateway,
                event_name: "SessionEnd".into(),
                payload: json!({}),
                metadata: json!({}),
            })],
        )
        .await
        .unwrap();

    assert!(manager.inner.lock().await.is_empty());
    let atif = std::fs::read_to_string(temp.path().join("gateway-cleanup.atif.json")).unwrap();
    assert!(atif.contains("closed_by_agent_end"));
}

#[tokio::test]
async fn llm_lifecycle_uses_single_active_hook_session_when_header_is_missing() {
    let config = GatewayConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://127.0.0.1".into(),
        anthropic_base_url: "http://127.0.0.1".into(),
        exporters: ExportersConfig::default(),
        metadata: None,
        plugin_config: None,
    };
    let manager = SessionManager::new(config);
    manager
        .apply_events(
            &HeaderMap::new(),
            vec![NormalizedEvent::AgentStarted(SessionEvent {
                session_id: "hook-session".into(),
                agent_kind: AgentKind::Codex,
                event_name: "sessionStart".into(),
                payload: json!({}),
                metadata: json!({}),
            })],
        )
        .await
        .unwrap();

    let active = manager
        .start_llm(
            &HeaderMap::new(),
            LlmGatewayStart {
                session_id: None,
                provider: "openai.responses".into(),
                model_name: Some("gpt-test".into()),
                subagent_id: None,
                conversation_id: None,
                generation_id: None,
                request_id: None,
                request: LlmRequest {
                    headers: Map::new(),
                    content: json!({ "model": "gpt-test", "input": "hello" }),
                },
                streaming: false,
                metadata: json!({ "gateway_path": "/v1/responses" }),
            },
        )
        .await
        .unwrap();
    manager
        .end_llm(active, json!({ "output_text": "hello" }), json!({}))
        .await
        .unwrap();

    let sessions = manager.inner.lock().await;
    assert!(sessions.contains_key("hook-session"));
    assert!(!sessions.contains_key("gateway-gateway"));
}

#[tokio::test]
async fn single_pending_llm_hint_claims_next_gateway_llm() {
    let config = GatewayConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://127.0.0.1".into(),
        anthropic_base_url: "http://127.0.0.1".into(),
        exporters: ExportersConfig::default(),
        metadata: None,
        plugin_config: None,
    };
    let manager = SessionManager::new(config);
    manager
        .apply_events(
            &HeaderMap::new(),
            vec![
                NormalizedEvent::AgentStarted(SessionEvent {
                    session_id: "hint-session".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "SessionStart".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::SubagentStarted(SubagentEvent {
                    session_id: "hint-session".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "SubagentStart".into(),
                    subagent_id: "worker-1".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::LlmHint(LlmHintEvent {
                    session_id: "hint-session".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "UserPromptSubmit".into(),
                    subagent_id: Some("worker-1".into()),
                    agent_id: None,
                    agent_type: Some("Explore".into()),
                    conversation_id: Some("conv-1".into()),
                    generation_id: None,
                    request_id: None,
                    model: Some("gpt-test".into()),
                    payload: json!({ "prompt": "hello" }),
                    metadata: json!({}),
                }),
            ],
        )
        .await
        .unwrap();

    let subagent_uuid = {
        let sessions = manager.inner.lock().await;
        sessions
            .get("hint-session")
            .unwrap()
            .subagents
            .get("worker-1")
            .unwrap()
            .uuid
    };
    let active = manager
        .start_llm(
            &HeaderMap::new(),
            LlmGatewayStart {
                session_id: Some("hint-session".into()),
                provider: "openai.responses".into(),
                model_name: Some("gpt-test".into()),
                subagent_id: None,
                conversation_id: None,
                generation_id: None,
                request_id: None,
                request: LlmRequest {
                    headers: Map::new(),
                    content: json!({ "model": "gpt-test", "input": "hello" }),
                },
                streaming: false,
                metadata: json!({}),
            },
        )
        .await
        .unwrap();

    assert_eq!(active.handle.parent_uuid, Some(subagent_uuid));
    assert_eq!(
        active.handle.metadata.as_ref().unwrap()["llm_correlation_status"],
        json!("single_hint")
    );
    assert_eq!(
        active.handle.metadata.as_ref().unwrap()["llm_correlation_subagent_id"],
        json!("worker-1")
    );
    manager
        .end_llm(active, json!({ "output_text": "hello" }), json!({}))
        .await
        .unwrap();
}

#[tokio::test]
async fn multiple_llm_hints_resolve_by_generation_id() {
    let config = GatewayConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://127.0.0.1".into(),
        anthropic_base_url: "http://127.0.0.1".into(),
        exporters: ExportersConfig::default(),
        metadata: None,
        plugin_config: None,
    };
    let manager = SessionManager::new(config);
    manager
        .apply_events(
            &HeaderMap::new(),
            vec![
                NormalizedEvent::AgentStarted(SessionEvent {
                    session_id: "multi-session".into(),
                    agent_kind: AgentKind::Cursor,
                    event_name: "sessionStart".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::SubagentStarted(SubagentEvent {
                    session_id: "multi-session".into(),
                    agent_kind: AgentKind::Cursor,
                    event_name: "subagentStart".into(),
                    subagent_id: "worker-1".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::SubagentStarted(SubagentEvent {
                    session_id: "multi-session".into(),
                    agent_kind: AgentKind::Cursor,
                    event_name: "subagentStart".into(),
                    subagent_id: "worker-2".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::LlmHint(LlmHintEvent {
                    session_id: "multi-session".into(),
                    agent_kind: AgentKind::Cursor,
                    event_name: "afterAgentThought".into(),
                    subagent_id: Some("worker-1".into()),
                    agent_id: None,
                    agent_type: None,
                    conversation_id: Some("conv-1".into()),
                    generation_id: Some("gen-1".into()),
                    request_id: None,
                    model: Some("gpt-test".into()),
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::LlmHint(LlmHintEvent {
                    session_id: "multi-session".into(),
                    agent_kind: AgentKind::Cursor,
                    event_name: "afterAgentThought".into(),
                    subagent_id: Some("worker-2".into()),
                    agent_id: None,
                    agent_type: None,
                    conversation_id: Some("conv-1".into()),
                    generation_id: Some("gen-2".into()),
                    request_id: None,
                    model: Some("gpt-test".into()),
                    payload: json!({}),
                    metadata: json!({}),
                }),
            ],
        )
        .await
        .unwrap();

    let worker_2_uuid = {
        let sessions = manager.inner.lock().await;
        sessions
            .get("multi-session")
            .unwrap()
            .subagents
            .get("worker-2")
            .unwrap()
            .uuid
    };
    let active = manager
        .start_llm(
            &HeaderMap::new(),
            LlmGatewayStart {
                session_id: Some("multi-session".into()),
                provider: "openai.responses".into(),
                model_name: Some("gpt-test".into()),
                subagent_id: None,
                conversation_id: Some("conv-1".into()),
                generation_id: Some("gen-2".into()),
                request_id: None,
                request: LlmRequest {
                    headers: Map::new(),
                    content: json!({ "model": "gpt-test", "input": "hello" }),
                },
                streaming: false,
                metadata: json!({}),
            },
        )
        .await
        .unwrap();

    assert_eq!(active.handle.parent_uuid, Some(worker_2_uuid));
    assert_eq!(
        active.handle.metadata.as_ref().unwrap()["llm_correlation_status"],
        json!("matched_hint")
    );
    manager
        .end_llm(active, json!({ "output_text": "hello" }), json!({}))
        .await
        .unwrap();
}

#[tokio::test]
async fn ambiguous_llm_hints_fall_back_to_agent_scope() {
    let config = GatewayConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://127.0.0.1".into(),
        anthropic_base_url: "http://127.0.0.1".into(),
        exporters: ExportersConfig::default(),
        metadata: None,
        plugin_config: None,
    };
    let manager = SessionManager::new(config);
    manager
        .apply_events(
            &HeaderMap::new(),
            vec![
                NormalizedEvent::AgentStarted(SessionEvent {
                    session_id: "ambiguous-session".into(),
                    agent_kind: AgentKind::Cursor,
                    event_name: "sessionStart".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::LlmHint(LlmHintEvent {
                    session_id: "ambiguous-session".into(),
                    agent_kind: AgentKind::Cursor,
                    event_name: "afterAgentThought".into(),
                    subagent_id: None,
                    agent_id: None,
                    agent_type: None,
                    conversation_id: Some("conv-1".into()),
                    generation_id: None,
                    request_id: None,
                    model: Some("gpt-test".into()),
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::LlmHint(LlmHintEvent {
                    session_id: "ambiguous-session".into(),
                    agent_kind: AgentKind::Cursor,
                    event_name: "afterAgentResponse".into(),
                    subagent_id: None,
                    agent_id: None,
                    agent_type: None,
                    conversation_id: Some("conv-1".into()),
                    generation_id: None,
                    request_id: None,
                    model: Some("gpt-test".into()),
                    payload: json!({}),
                    metadata: json!({}),
                }),
            ],
        )
        .await
        .unwrap();

    let agent_uuid = {
        let sessions = manager.inner.lock().await;
        sessions
            .get("ambiguous-session")
            .unwrap()
            .agent_scope
            .as_ref()
            .unwrap()
            .uuid
    };
    let active = manager
        .start_llm(
            &HeaderMap::new(),
            LlmGatewayStart {
                session_id: Some("ambiguous-session".into()),
                provider: "openai.responses".into(),
                model_name: Some("gpt-test".into()),
                subagent_id: None,
                conversation_id: Some("conv-1".into()),
                generation_id: None,
                request_id: None,
                request: LlmRequest {
                    headers: Map::new(),
                    content: json!({ "model": "gpt-test", "input": "hello" }),
                },
                streaming: false,
                metadata: json!({}),
            },
        )
        .await
        .unwrap();

    assert_eq!(active.handle.parent_uuid, Some(agent_uuid));
    assert_eq!(
        active.handle.metadata.as_ref().unwrap()["llm_correlation_status"],
        json!("ambiguous_fallback")
    );
    manager
        .end_llm(active, json!({ "output_text": "hello" }), json!({}))
        .await
        .unwrap();
}

#[tokio::test]
async fn no_active_hint_reuses_last_llm_owner() {
    let config = GatewayConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://127.0.0.1".into(),
        anthropic_base_url: "http://127.0.0.1".into(),
        exporters: ExportersConfig::default(),
        metadata: None,
        plugin_config: None,
    };
    let manager = SessionManager::new(config);
    manager
        .apply_events(
            &HeaderMap::new(),
            vec![
                NormalizedEvent::AgentStarted(SessionEvent {
                    session_id: "sticky-session".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "SessionStart".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::SubagentStarted(SubagentEvent {
                    session_id: "sticky-session".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "SubagentStart".into(),
                    subagent_id: "worker-1".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::LlmHint(LlmHintEvent {
                    session_id: "sticky-session".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "UserPromptSubmit".into(),
                    subagent_id: Some("worker-1".into()),
                    agent_id: None,
                    agent_type: None,
                    conversation_id: Some("conv-1".into()),
                    generation_id: None,
                    request_id: None,
                    model: Some("gpt-test".into()),
                    payload: json!({}),
                    metadata: json!({}),
                }),
            ],
        )
        .await
        .unwrap();

    let first = manager
        .start_llm(
            &HeaderMap::new(),
            LlmGatewayStart {
                session_id: Some("sticky-session".into()),
                provider: "openai.responses".into(),
                model_name: Some("gpt-test".into()),
                subagent_id: None,
                conversation_id: None,
                generation_id: None,
                request_id: None,
                request: LlmRequest {
                    headers: Map::new(),
                    content: json!({ "model": "gpt-test", "input": "hello" }),
                },
                streaming: false,
                metadata: json!({}),
            },
        )
        .await
        .unwrap();
    let worker_uuid = first.handle.parent_uuid;
    manager
        .end_llm(first, json!({ "output_text": "hello" }), json!({}))
        .await
        .unwrap();

    let second = manager
        .start_llm(
            &HeaderMap::new(),
            LlmGatewayStart {
                session_id: Some("sticky-session".into()),
                provider: "openai.responses".into(),
                model_name: Some("gpt-test".into()),
                subagent_id: None,
                conversation_id: None,
                generation_id: None,
                request_id: None,
                request: LlmRequest {
                    headers: Map::new(),
                    content: json!({ "model": "gpt-test", "input": "again" }),
                },
                streaming: false,
                metadata: json!({}),
            },
        )
        .await
        .unwrap();

    assert_eq!(second.handle.parent_uuid, worker_uuid);
    assert_eq!(
        second.handle.metadata.as_ref().unwrap()["llm_correlation_status"],
        json!("sticky_last_owner")
    );
    manager
        .end_llm(second, json!({ "output_text": "again" }), json!({}))
        .await
        .unwrap();
}

#[tokio::test]
async fn session_marks_cover_compaction_notifications_and_hook_marks() {
    let temp = tempfile::tempdir().unwrap();
    let mut config = session_test_config();
    config.exporters.atif.dir = Some(temp.path().to_path_buf());
    let manager = SessionManager::new(config);
    let headers = HeaderMap::new();

    manager
        .apply_events(
            &headers,
            vec![
                NormalizedEvent::AgentStarted(session_event("marks", "SessionStart")),
                NormalizedEvent::Compaction(session_event("marks", "PreCompact")),
                NormalizedEvent::Notification(session_event("marks", "Notification")),
                NormalizedEvent::HookMark(session_event("marks", "CustomHook")),
                NormalizedEvent::AgentEnded(session_event("marks", "SessionEnd")),
            ],
        )
        .await
        .unwrap();

    let atif = std::fs::read_to_string(temp.path().join("marks.atif.json")).unwrap();
    assert!(atif.contains("PreCompact"));
    assert!(atif.contains("Notification"));
    assert!(atif.contains("CustomHook"));
}

#[tokio::test]
async fn agent_end_closes_active_tools_and_duplicate_starts_are_ignored() {
    let manager = SessionManager::new(session_test_config());
    let headers = HeaderMap::new();

    manager
        .apply_events(
            &headers,
            vec![
                NormalizedEvent::AgentStarted(session_event("active-tool-cleanup", "SessionStart")),
                NormalizedEvent::SubagentStarted(SubagentEvent {
                    session_id: "active-tool-cleanup".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "SubagentStart".into(),
                    subagent_id: "worker".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::SubagentStarted(SubagentEvent {
                    session_id: "active-tool-cleanup".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "SubagentStart".into(),
                    subagent_id: "worker".into(),
                    payload: json!({ "duplicate": true }),
                    metadata: json!({}),
                }),
                NormalizedEvent::ToolStarted(ToolEvent {
                    session_id: "active-tool-cleanup".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "PreToolUse".into(),
                    tool_call_id: "tool-1".into(),
                    tool_name: "Read".into(),
                    subagent_id: Some("worker".into()),
                    arguments: json!({ "file_path": "README.md" }),
                    result: Value::Null,
                    status: None,
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::ToolStarted(ToolEvent {
                    session_id: "active-tool-cleanup".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "PreToolUse".into(),
                    tool_call_id: "tool-1".into(),
                    tool_name: "Read".into(),
                    subagent_id: Some("worker".into()),
                    arguments: json!({ "file_path": "README.md" }),
                    result: Value::Null,
                    status: None,
                    payload: json!({ "duplicate": true }),
                    metadata: json!({}),
                }),
                NormalizedEvent::AgentEnded(session_event("active-tool-cleanup", "SessionEnd")),
            ],
        )
        .await
        .unwrap();

    assert!(manager.inner.lock().await.is_empty());
}

#[tokio::test]
async fn explicit_gateway_subagent_header_sets_llm_parent() {
    let manager = SessionManager::new(session_test_config());
    let headers = HeaderMap::new();
    manager
        .apply_events(
            &headers,
            vec![
                NormalizedEvent::AgentStarted(session_event("explicit-owner", "SessionStart")),
                NormalizedEvent::SubagentStarted(SubagentEvent {
                    session_id: "explicit-owner".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "SubagentStart".into(),
                    subagent_id: "worker".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
            ],
        )
        .await
        .unwrap();

    let subagent_uuid = {
        let sessions = manager.inner.lock().await;
        sessions
            .get("explicit-owner")
            .unwrap()
            .subagents
            .get("worker")
            .unwrap()
            .uuid
    };
    let active = manager
        .start_llm(
            &HeaderMap::new(),
            LlmGatewayStart {
                session_id: Some("explicit-owner".into()),
                subagent_id: Some("worker".into()),
                ..llm_start()
            },
        )
        .await
        .unwrap();

    assert_eq!(active.handle.parent_uuid, Some(subagent_uuid));
    assert_eq!(
        active.handle.metadata.as_ref().unwrap()["llm_correlation_status"],
        json!("explicit")
    );
    assert_eq!(
        active.handle.metadata.as_ref().unwrap()["llm_correlation_source"],
        json!("gateway_header")
    );
}

#[tokio::test]
async fn single_active_subagent_claims_unhinted_gateway_llm() {
    let manager = SessionManager::new(session_test_config());
    let headers = HeaderMap::new();
    manager
        .apply_events(
            &headers,
            vec![
                NormalizedEvent::AgentStarted(session_event("single-subagent", "SessionStart")),
                NormalizedEvent::SubagentStarted(SubagentEvent {
                    session_id: "single-subagent".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "SubagentStart".into(),
                    subagent_id: "worker".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
            ],
        )
        .await
        .unwrap();

    let subagent_uuid = {
        let sessions = manager.inner.lock().await;
        sessions
            .get("single-subagent")
            .unwrap()
            .subagents
            .get("worker")
            .unwrap()
            .uuid
    };
    let active = manager
        .start_llm(
            &HeaderMap::new(),
            LlmGatewayStart {
                session_id: Some("single-subagent".into()),
                ..llm_start()
            },
        )
        .await
        .unwrap();

    assert_eq!(active.handle.parent_uuid, Some(subagent_uuid));
    assert_eq!(
        active.handle.metadata.as_ref().unwrap()["llm_correlation_status"],
        json!("active_subagent")
    );
}

#[tokio::test]
async fn llm_response_tool_hint_claims_next_tool_hook() {
    let manager = SessionManager::new(session_test_config());
    manager
        .apply_events(
            &HeaderMap::new(),
            vec![
                NormalizedEvent::AgentStarted(session_event("tool-hints", "SessionStart")),
                NormalizedEvent::SubagentStarted(SubagentEvent {
                    session_id: "tool-hints".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "SubagentStart".into(),
                    subagent_id: "worker".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
            ],
        )
        .await
        .unwrap();

    let subagent_uuid = {
        let sessions = manager.inner.lock().await;
        sessions
            .get("tool-hints")
            .unwrap()
            .subagents
            .get("worker")
            .unwrap()
            .uuid
    };
    let active = manager
        .start_llm(
            &HeaderMap::new(),
            LlmGatewayStart {
                session_id: Some("tool-hints".into()),
                subagent_id: Some("worker".into()),
                ..llm_start()
            },
        )
        .await
        .unwrap();
    manager
        .end_llm(
            active,
            json!({
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call-1",
                        "name": "Read",
                        "arguments": "{\"file_path\":\"README.md\"}"
                    }
                ]
            }),
            json!({}),
        )
        .await
        .unwrap();

    manager
        .apply_events(
            &HeaderMap::new(),
            vec![NormalizedEvent::ToolStarted(ToolEvent {
                session_id: "tool-hints".into(),
                agent_kind: AgentKind::ClaudeCode,
                event_name: "PreToolUse".into(),
                tool_call_id: "call-1".into(),
                tool_name: "Read".into(),
                subagent_id: None,
                arguments: Value::Null,
                result: Value::Null,
                status: None,
                payload: json!({}),
                metadata: json!({}),
            })],
        )
        .await
        .unwrap();

    let sessions = manager.inner.lock().await;
    let handle = sessions
        .get("tool-hints")
        .unwrap()
        .tools
        .get("call-1")
        .unwrap();
    assert_eq!(handle.parent_uuid, Some(subagent_uuid));
    assert_eq!(
        handle.metadata.as_ref().unwrap()["tool_correlation_status"],
        json!("single_hint")
    );
    assert_eq!(
        handle.metadata.as_ref().unwrap()["tool_correlation_subagent_id"],
        json!("worker")
    );
}

#[test]
fn openai_response_tool_hints_ignore_non_tool_output_items() {
    let mut hints = Vec::new();

    collect_openai_response_tool_hints(
        &json!({
            "output": [
                {
                    "type": "message",
                    "id": "msg-1",
                    "name": "Read",
                    "arguments": "{\"file_path\":\"README.md\"}"
                },
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "Read",
                    "arguments": "{\"file_path\":\"README.md\"}"
                }
            ]
        }),
        Some("worker"),
        &mut hints,
    );

    assert_eq!(hints.len(), 1);
    assert_eq!(hints[0].tool_call_id.as_deref(), Some("call-1"));
}

#[test]
fn write_atif_rejects_unsafe_session_id_filename() {
    let temp = tempfile::tempdir().unwrap();
    let exporter = AtifExporter::new(
        "safe-session".to_string(),
        AtifAgentInfo {
            name: "test-agent".to_string(),
            version: "1.0.0".to_string(),
            model_name: None,
            tool_definitions: None,
            extra: None,
        },
    );

    let error = write_atif(&temp.path().to_path_buf(), "../escape", &exporter).unwrap_err();

    assert!(matches!(error, CliError::InvalidPayload(_)));
    assert!(!temp.path().join("../escape.atif.json").exists());
}

#[tokio::test]
async fn multiple_tool_hints_resolve_by_tool_call_id() {
    let manager = SessionManager::new(session_test_config());
    manager
        .apply_events(
            &HeaderMap::new(),
            vec![
                NormalizedEvent::AgentStarted(session_event("multi-tool-hints", "SessionStart")),
                NormalizedEvent::SubagentStarted(SubagentEvent {
                    session_id: "multi-tool-hints".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "SubagentStart".into(),
                    subagent_id: "worker".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
            ],
        )
        .await
        .unwrap();

    let active = manager
        .start_llm(
            &HeaderMap::new(),
            LlmGatewayStart {
                session_id: Some("multi-tool-hints".into()),
                subagent_id: Some("worker".into()),
                ..llm_start()
            },
        )
        .await
        .unwrap();
    manager
        .end_llm(
            active,
            json!({
                "choices": [{
                    "message": {
                        "tool_calls": [
                            { "id": "call-a", "function": { "name": "Read", "arguments": "{}" } },
                            { "id": "call-b", "function": { "name": "Bash", "arguments": "{\"command\":\"pwd\"}" } }
                        ]
                    }
                }]
            }),
            json!({}),
        )
        .await
        .unwrap();

    manager
        .apply_events(
            &HeaderMap::new(),
            vec![NormalizedEvent::ToolStarted(ToolEvent {
                session_id: "multi-tool-hints".into(),
                agent_kind: AgentKind::ClaudeCode,
                event_name: "PreToolUse".into(),
                tool_call_id: "call-b".into(),
                tool_name: "Bash".into(),
                subagent_id: None,
                arguments: json!({ "command": "pwd" }),
                result: Value::Null,
                status: None,
                payload: json!({}),
                metadata: json!({}),
            })],
        )
        .await
        .unwrap();

    let sessions = manager.inner.lock().await;
    let handle = sessions
        .get("multi-tool-hints")
        .unwrap()
        .tools
        .get("call-b")
        .unwrap();
    assert_eq!(
        handle.metadata.as_ref().unwrap()["tool_correlation_status"],
        json!("matched_hint")
    );
    assert_eq!(
        handle.metadata.as_ref().unwrap()["tool_correlation_tool_call_id"],
        json!("call-b")
    );
}

#[tokio::test]
async fn hint_for_missing_subagent_falls_back_to_agent_scope() {
    let manager = SessionManager::new(session_test_config());
    let headers = HeaderMap::new();
    manager
        .apply_events(
            &headers,
            vec![
                NormalizedEvent::AgentStarted(session_event("missing-hint-owner", "SessionStart")),
                NormalizedEvent::LlmHint(LlmHintEvent {
                    session_id: "missing-hint-owner".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "UserPromptSubmit".into(),
                    subagent_id: Some("missing-worker".into()),
                    agent_id: None,
                    agent_type: None,
                    conversation_id: None,
                    generation_id: None,
                    request_id: None,
                    model: Some("gpt-test".into()),
                    payload: json!({}),
                    metadata: json!({}),
                }),
            ],
        )
        .await
        .unwrap();

    let agent_uuid = {
        let sessions = manager.inner.lock().await;
        sessions
            .get("missing-hint-owner")
            .unwrap()
            .agent_scope
            .as_ref()
            .unwrap()
            .uuid
    };
    let active = manager
        .start_llm(
            &HeaderMap::new(),
            LlmGatewayStart {
                session_id: Some("missing-hint-owner".into()),
                ..llm_start()
            },
        )
        .await
        .unwrap();

    assert_eq!(active.handle.parent_uuid, Some(agent_uuid));
    assert_eq!(
        active.handle.metadata.as_ref().unwrap()["llm_correlation_status"],
        json!("single_hint")
    );
    assert!(
        active
            .handle
            .metadata
            .as_ref()
            .unwrap()
            .get("llm_correlation_subagent_id")
            .is_none()
    );
}

#[test]
fn llm_hint_scoring_and_event_accessors_cover_all_variants() {
    let hint = LlmHintEvent {
        session_id: "score".into(),
        agent_kind: AgentKind::Codex,
        event_name: "afterAgentThought".into(),
        subagent_id: Some("worker".into()),
        agent_id: None,
        agent_type: None,
        conversation_id: Some("conv".into()),
        generation_id: Some("gen".into()),
        request_id: Some("req".into()),
        model: Some("gpt-test".into()),
        payload: json!({}),
        metadata: json!({}),
    };
    let start = LlmGatewayStart {
        session_id: Some("score".into()),
        subagent_id: Some("worker".into()),
        conversation_id: Some("conv".into()),
        generation_id: Some("gen".into()),
        request_id: Some("req".into()),
        ..llm_start()
    };

    assert_eq!(hint_match_score(&hint, &start), 21);

    for event in [
        NormalizedEvent::PromptSubmitted(session_event("variant", "UserPromptSubmit")),
        NormalizedEvent::Compaction(session_event("variant", "PreCompact")),
        NormalizedEvent::Notification(session_event("variant", "Notification")),
        NormalizedEvent::HookMark(session_event("variant", "Custom")),
    ] {
        assert_eq!(event.session_id(), "variant");
        assert_eq!(event_agent_kind(&event), AgentKind::ClaudeCode);
    }
}

#[test]
fn merge_metadata_handles_objects_nulls_and_scalars() {
    assert_eq!(
        merge_metadata(json!({ "a": 1 }), json!({ "b": 2, "c": null })),
        json!({ "a": 1, "b": 2 })
    );
    assert_eq!(
        merge_metadata(Value::Null, json!({ "a": 1 })),
        json!({ "a": 1 })
    );
    assert_eq!(
        merge_metadata(json!({ "a": 1 }), Value::Null),
        json!({ "a": 1 })
    );
    assert_eq!(
        merge_metadata(json!("left"), json!("right")),
        json!({ "metadata": "left", "extra_metadata": "right" })
    );
}

fn session_test_config() -> GatewayConfig {
    GatewayConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://127.0.0.1".into(),
        anthropic_base_url: "http://127.0.0.1".into(),
        exporters: ExportersConfig::default(),
        metadata: None,
        plugin_config: None,
    }
}

// Regression: an Anthropic Messages gateway request that arrives before SessionStart used to
// freeze the session label as "gateway" (default agent_kind) for the rest of the session,
// because observer identities are baked at scope-open time. The session must instead be labeled
// `claude-code` from the provider, so ATIF and Phoenix root spans reflect the real agent.
#[tokio::test]
async fn gateway_first_anthropic_call_labels_session_as_claude_code() {
    let temp = tempfile::tempdir().unwrap();
    let config = GatewayConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://127.0.0.1".into(),
        anthropic_base_url: "http://127.0.0.1".into(),
        exporters: ExportersConfig {
            atif: AtifExporterSettings {
                dir: Some(temp.path().to_path_buf()),
            },
            ..Default::default()
        },
        metadata: None,
        plugin_config: None,
    };
    let manager = SessionManager::new(config);
    let mut start = llm_start();
    start.session_id = Some("claude-uuid".into());
    start.provider = "anthropic.messages".into();
    let active = manager.start_llm(&HeaderMap::new(), start).await.unwrap();
    manager
        .end_llm(active, json!({ "ok": true }), json!({}))
        .await
        .unwrap();
    // Drive an explicit AgentEnded so flush_observers writes ATIF.
    manager
        .apply_events(
            &HeaderMap::new(),
            vec![NormalizedEvent::AgentEnded(SessionEvent {
                session_id: "claude-uuid".into(),
                agent_kind: AgentKind::ClaudeCode,
                event_name: "SessionEnd".into(),
                payload: json!({}),
                metadata: json!({}),
            })],
        )
        .await
        .unwrap();

    let atif: Value = serde_json::from_str(
        &std::fs::read_to_string(temp.path().join("claude-uuid.atif.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        atif["agent"]["name"],
        json!("claude-code"),
        "session created from anthropic.messages gateway request must be labeled claude-code, not gateway"
    );
}

// OpenAI Responses gateway requests (codex's API path) must label the session as `codex`.
#[tokio::test]
async fn gateway_first_openai_responses_call_labels_session_as_codex() {
    let temp = tempfile::tempdir().unwrap();
    let config = GatewayConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://127.0.0.1".into(),
        anthropic_base_url: "http://127.0.0.1".into(),
        exporters: ExportersConfig {
            atif: AtifExporterSettings {
                dir: Some(temp.path().to_path_buf()),
            },
            ..Default::default()
        },
        metadata: None,
        plugin_config: None,
    };
    let manager = SessionManager::new(config);
    let mut start = llm_start();
    start.session_id = Some("codex-uuid".into());
    start.provider = "openai.responses".into();
    let active = manager.start_llm(&HeaderMap::new(), start).await.unwrap();
    manager
        .end_llm(active, json!({ "ok": true }), json!({}))
        .await
        .unwrap();
    manager
        .apply_events(
            &HeaderMap::new(),
            vec![NormalizedEvent::AgentEnded(SessionEvent {
                session_id: "codex-uuid".into(),
                agent_kind: AgentKind::Codex,
                event_name: "SessionEnd".into(),
                payload: json!({}),
                metadata: json!({}),
            })],
        )
        .await
        .unwrap();

    let atif: Value = serde_json::from_str(
        &std::fs::read_to_string(temp.path().join("codex-uuid.atif.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(atif["agent"]["name"], json!("codex"));
}

// Synthetic gateway-only sessions (pure proxy traffic, unknown provider) keep the legacy
// `gateway` label so existing observability semantics for unattributed traffic are preserved.
#[tokio::test]
async fn synthetic_gateway_session_keeps_gateway_label() {
    let temp = tempfile::tempdir().unwrap();
    let config = GatewayConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://127.0.0.1".into(),
        anthropic_base_url: "http://127.0.0.1".into(),
        exporters: ExportersConfig {
            atif: AtifExporterSettings {
                dir: Some(temp.path().to_path_buf()),
            },
            ..Default::default()
        },
        metadata: None,
        plugin_config: None,
    };
    let manager = SessionManager::new(config);
    let mut start = llm_start();
    start.session_id = None;
    start.provider = "openai.chat_completions".into(); // ambiguous → Gateway
    let active = manager.start_llm(&HeaderMap::new(), start).await.unwrap();
    manager
        .end_llm(active, json!({ "ok": true }), json!({}))
        .await
        .unwrap();
    manager
        .apply_events(
            &HeaderMap::new(),
            vec![NormalizedEvent::AgentEnded(SessionEvent {
                session_id: "gateway-gateway".into(),
                agent_kind: AgentKind::Gateway,
                event_name: "SessionEnd".into(),
                payload: json!({}),
                metadata: json!({}),
            })],
        )
        .await
        .unwrap();

    let atif: Value = serde_json::from_str(
        &std::fs::read_to_string(temp.path().join("gateway-gateway.atif.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(atif["agent"]["name"], json!("gateway"));
}

// `TurnEnded` (synthesized from per-turn `Stop` hooks) writes ATIF without closing the agent
// scope. This is the codex-0.129 workaround: codex has no `SessionEnd` hook, so per-turn
// snapshots are how its ATIF gets written. After several turns the agent scope must remain open
// and the trajectory file must reflect cumulative state.
#[tokio::test]
async fn turn_ended_snapshots_atif_without_closing_scope() {
    let temp = tempfile::tempdir().unwrap();
    let config = GatewayConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://127.0.0.1".into(),
        anthropic_base_url: "http://127.0.0.1".into(),
        exporters: ExportersConfig {
            atif: AtifExporterSettings {
                dir: Some(temp.path().to_path_buf()),
            },
            ..Default::default()
        },
        metadata: None,
        plugin_config: None,
    };
    let manager = SessionManager::new(config);
    let headers = HeaderMap::new();

    // Open a codex session.
    manager
        .apply_events(
            &headers,
            vec![NormalizedEvent::AgentStarted(SessionEvent {
                session_id: "codex-multi-turn".into(),
                agent_kind: AgentKind::Codex,
                event_name: "SessionStart".into(),
                payload: json!({}),
                metadata: json!({}),
            })],
        )
        .await
        .unwrap();
    assert_eq!(manager.open_session_count().await, 1);

    // First turn ends — ATIF should be written even though SessionEnd never arrived.
    manager
        .apply_events(
            &headers,
            vec![NormalizedEvent::TurnEnded(SessionEvent {
                session_id: "codex-multi-turn".into(),
                agent_kind: AgentKind::Codex,
                event_name: "Stop".into(),
                payload: json!({}),
                metadata: json!({}),
            })],
        )
        .await
        .unwrap();

    let atif_path = temp.path().join("codex-multi-turn.atif.json");
    assert!(
        atif_path.exists(),
        "TurnEnded must produce an ATIF file during an open session"
    );
    // Session is still open — TurnEnded must not have torn it down.
    assert_eq!(
        manager.open_session_count().await,
        1,
        "TurnEnded must NOT close the agent scope or remove the session"
    );

    // Second turn ends — file should be overwritten with a cumulative trajectory.
    manager
        .apply_events(
            &headers,
            vec![NormalizedEvent::TurnEnded(SessionEvent {
                session_id: "codex-multi-turn".into(),
                agent_kind: AgentKind::Codex,
                event_name: "Stop".into(),
                payload: json!({}),
                metadata: json!({}),
            })],
        )
        .await
        .unwrap();
    assert!(atif_path.exists());
    assert_eq!(manager.open_session_count().await, 1);

    let trajectory: Value = serde_json::from_slice(&std::fs::read(&atif_path).unwrap()).unwrap();
    assert_eq!(trajectory["session_id"], json!("codex-multi-turn"));
    assert_eq!(trajectory["agent"]["name"], json!("codex"));
}

// TurnEnded for a session that was never opened (no AgentStarted, no gateway LLM) is a no-op —
// no observers were ever installed, so there's nothing to flush.
#[tokio::test]
async fn turn_ended_is_noop_for_session_with_no_agent_scope() {
    let temp = tempfile::tempdir().unwrap();
    let config = GatewayConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://127.0.0.1".into(),
        anthropic_base_url: "http://127.0.0.1".into(),
        exporters: ExportersConfig {
            atif: AtifExporterSettings {
                dir: Some(temp.path().to_path_buf()),
            },
            ..Default::default()
        },
        metadata: None,
        plugin_config: None,
    };
    let manager = SessionManager::new(config);
    manager
        .apply_events(
            &HeaderMap::new(),
            vec![NormalizedEvent::TurnEnded(SessionEvent {
                session_id: "no-agent".into(),
                agent_kind: AgentKind::Codex,
                event_name: "Stop".into(),
                payload: json!({}),
                metadata: json!({}),
            })],
        )
        .await
        .unwrap();
    // No file should be created — the snapshot needs an active session with installed observers.
    assert!(std::fs::read_dir(temp.path()).unwrap().next().is_none());
}

fn session_event(session_id: &str, event_name: &str) -> SessionEvent {
    SessionEvent {
        session_id: session_id.into(),
        agent_kind: AgentKind::ClaudeCode,
        event_name: event_name.into(),
        payload: json!({ "event": event_name }),
        metadata: json!({}),
    }
}

fn llm_start() -> LlmGatewayStart {
    LlmGatewayStart {
        session_id: Some("llm".into()),
        provider: "openai.responses".into(),
        model_name: Some("gpt-test".into()),
        subagent_id: None,
        conversation_id: None,
        generation_id: None,
        request_id: None,
        request: LlmRequest {
            headers: Map::new(),
            content: json!({ "model": "gpt-test", "input": "hello" }),
        },
        streaming: false,
        metadata: json!({}),
    }
}
