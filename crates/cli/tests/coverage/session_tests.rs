// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::http::HeaderMap;
use nemo_relay::api::event::ScopeCategory;
use nemo_relay::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use nemo_relay::plugin::{PluginConfig, clear_plugin_configuration, initialize_plugins};
use serde_json::json;
use std::io::{Read, Write};
use std::net::TcpListener as StdTcpListener;
use std::path::Path;
use std::sync::{Arc, Mutex as StdMutex};
use std::thread;
use std::time::{Duration, Instant};

use super::*;
use crate::model::{LlmEvent, LlmHintEvent, SessionEvent, ToolEvent};

static OBSERVABILITY_PLUGIN_TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

async fn install_test_atif_plugin(output_directory: &Path) {
    let _ = clear_plugin_configuration();
    std::fs::create_dir_all(output_directory).unwrap();
    let config: PluginConfig = serde_json::from_value(json!({
        "version": 1,
        "components": [
            {
                "kind": "observability",
                "enabled": true,
                "config": {
                    "version": 1,
                    "atif": {
                        "enabled": true,
                        "output_directory": output_directory,
                        "filename_template": "trajectory-{session_id}.json"
                    }
                }
            }
        ]
    }))
    .unwrap();
    initialize_plugins(config).await.unwrap();
}

async fn install_test_openinference_plugin(endpoint: &str) {
    let _ = clear_plugin_configuration();
    let config: PluginConfig = serde_json::from_value(json!({
        "version": 1,
        "components": [
            {
                "kind": "observability",
                "enabled": true,
                "config": {
                    "version": 1,
                    "openinference": {
                        "enabled": true,
                        "transport": "http_binary",
                        "endpoint": endpoint,
                        "service_name": "codex-openinference-test",
                        "timeout_millis": 1000
                    }
                }
            }
        ]
    }))
    .unwrap();
    initialize_plugins(config).await.unwrap();
}

fn start_otlp_collector() -> (String, thread::JoinHandle<(String, Vec<u8>)>) {
    let listener = StdTcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}/v1/traces", listener.local_addr().unwrap());
    let handle = thread::spawn(move || {
        listener.set_nonblocking(true).unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        let (mut stream, _) = loop {
            match listener.accept() {
                Ok(connection) => break connection,
                Err(error)
                    if error.kind() == std::io::ErrorKind::WouldBlock
                        && Instant::now() < deadline =>
                {
                    thread::sleep(Duration::from_millis(10));
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    panic!("collector should receive an OTLP request")
                }
                Err(error) => panic!("collector failed to accept OTLP request: {error}"),
            }
        };
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut buffer = Vec::new();
        let mut content_length = None;
        let mut header_end = None;
        loop {
            let mut chunk = [0_u8; 4096];
            let read = stream.read(&mut chunk).unwrap();
            if read == 0 {
                break;
            }
            buffer.extend_from_slice(&chunk[..read]);
            if header_end.is_none()
                && let Some(position) = buffer.windows(4).position(|window| window == b"\r\n\r\n")
            {
                header_end = Some(position + 4);
                let headers = String::from_utf8_lossy(&buffer[..position]);
                content_length = headers.lines().find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())
                        .flatten()
                });
            }
            if let (Some(header_end), Some(content_length)) = (header_end, content_length)
                && buffer.len() >= header_end + content_length
            {
                break;
            }
        }

        let header_end = header_end.expect("collector should receive HTTP headers");
        let content_length = content_length.expect("collector should receive content length");
        let request = String::from_utf8_lossy(&buffer[..header_end]).to_string();
        let body = buffer[header_end..header_end + content_length].to_vec();
        stream
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
            .unwrap();
        (request, body)
    });
    (endpoint, handle)
}

async fn apply_codex_payload(manager: &SessionManager, headers: &HeaderMap, payload: Value) {
    let outcome = crate::adapters::codex::adapt(payload, headers);
    manager.apply_events(headers, outcome.events).await.unwrap();
}

async fn start_codex_prompt_turn(manager: &SessionManager, headers: &HeaderMap, session_id: &str) {
    for payload in [
        json!({
            "session_id": session_id,
            "hook_event_name": "sessionStart",
            "model": "gpt-test"
        }),
        json!({
            "session_id": session_id,
            "hook_event_name": "UserPromptSubmit",
            "prompt": "Inspect the repository."
        }),
    ] {
        apply_codex_payload(manager, headers, payload).await;
    }
}

async fn run_codex_responses_tool_activity(
    manager: &SessionManager,
    headers: &HeaderMap,
    session_id: &str,
) {
    let llm = manager
        .start_llm(
            headers,
            llm_start_with_responses_task(session_id, "Inspect the repository."),
        )
        .await
        .unwrap();
    manager
        .end_llm(
            llm,
            json!({
                "id": "resp_1",
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "tool-call-1",
                        "name": "Read",
                        "arguments": "{\"file_path\":\"README.md\"}",
                        "status": "completed"
                    }
                ]
            }),
            json!({}),
        )
        .await
        .unwrap();

    for payload in [
        json!({
            "session_id": session_id,
            "hook_event_name": "PreToolUse",
            "tool_call_id": "tool-call-1",
            "tool_name": "Read",
            "tool_input": { "file_path": "README.md" }
        }),
        json!({
            "session_id": session_id,
            "hook_event_name": "PostToolUse",
            "tool_call_id": "tool-call-1",
            "tool_name": "Read",
            "tool_output": { "content": "hello" },
            "status": "success"
        }),
    ] {
        apply_codex_payload(manager, headers, payload).await;
    }
}

async fn stop_codex_turn(manager: &SessionManager, headers: &HeaderMap, session_id: &str) {
    apply_codex_payload(
        manager,
        headers,
        json!({
            "session_id": session_id,
            "hook_event_name": "Stop",
            "response": "Done."
        }),
    )
    .await;
}

fn read_atif_for_session(output_directory: &Path, session_id: &str) -> Value {
    flush_subscribers().unwrap();
    std::fs::read_dir(output_directory)
        .unwrap()
        .filter_map(Result::ok)
        .filter_map(|entry| {
            serde_json::from_slice::<Value>(&std::fs::read(entry.path()).ok()?).ok()
        })
        .find(|trajectory| atif_matches_session(trajectory, session_id))
        .unwrap_or_else(|| panic!("expected ATIF trajectory for session {session_id}"))
}

fn atif_matches_session(trajectory: &Value, session_id: &str) -> bool {
    trajectory["session_id"] == json!(session_id)
        || trajectory["extra"]["observed_events"]
            .as_array()
            .is_some_and(|events| {
                events
                    .iter()
                    .any(|event| event_has_session_id(event, session_id))
            })
}

fn event_has_session_id(event: &Value, session_id: &str) -> bool {
    event["metadata"]["session_id"] == json!(session_id)
        || event["data"]["session_id"] == json!(session_id)
        || event["data"]["extra"]["session_id"] == json!(session_id)
}

fn active_turn_uuid(session: &Session) -> uuid::Uuid {
    active_turn_scope(session).uuid
}

fn active_turn_scope(session: &Session) -> &ScopeHandle {
    session
        .turn_scope
        .as_ref()
        .expect("expected active turn scope")
}

async fn alignment_alias(manager: &SessionManager, session_id: &str) -> Option<SessionAlias> {
    manager.alignment.lock().await.alias_for_session(session_id)
}

async fn has_alignment_alias(manager: &SessionManager, session_id: &str) -> bool {
    manager.alignment.lock().await.has_alias(session_id)
}

async fn has_pending_alignment(manager: &SessionManager, session_id: &str) -> bool {
    manager
        .alignment
        .lock()
        .await
        .has_pending_session(session_id)
}

#[tokio::test]
async fn nests_agent_subagent_and_tool_lifecycle() {
    let config = GatewayConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://127.0.0.1".into(),

        anthropic_base_url: "http://127.0.0.1".into(),
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
async fn parallel_subagents_are_siblings_under_turn_scope() {
    let manager = SessionManager::new(session_test_config());
    manager
        .apply_events(
            &HeaderMap::new(),
            vec![
                NormalizedEvent::AgentStarted(session_event("sibling-subagents", "SessionStart")),
                NormalizedEvent::SubagentStarted(SubagentEvent {
                    session_id: "sibling-subagents".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "SubagentStart".into(),
                    subagent_id: "worker-1".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::SubagentStarted(SubagentEvent {
                    session_id: "sibling-subagents".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "SubagentStart".into(),
                    subagent_id: "worker-2".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
            ],
        )
        .await
        .unwrap();

    let sessions = manager.inner.lock().await;
    let session = sessions.get("sibling-subagents").unwrap();
    assert!(session.agent_scope.is_none());
    let turn_uuid = active_turn_uuid(session);
    assert_eq!(
        session.subagents.get("worker-1").unwrap().scope_type,
        ScopeType::Agent
    );
    assert_eq!(
        session
            .subagents
            .get("worker-1")
            .unwrap()
            .metadata
            .as_ref()
            .unwrap()["nemo_relay_scope_role"],
        json!("subagent")
    );
    assert_eq!(
        session.subagents.get("worker-1").unwrap().parent_uuid,
        Some(turn_uuid)
    );
    assert_eq!(
        session.subagents.get("worker-2").unwrap().parent_uuid,
        Some(turn_uuid)
    );
}

#[tokio::test]
async fn codex_turn_is_agent_scope_with_turn_role_metadata() {
    let manager = SessionManager::new(session_test_config());
    manager
        .apply_events(
            &HeaderMap::new(),
            vec![
                NormalizedEvent::AgentStarted(codex_session_event(
                    "codex-turn-agent",
                    "SessionStart",
                    json!({ "transcript_path": "/tmp/session.jsonl" }),
                )),
                NormalizedEvent::PromptSubmitted(SessionEvent {
                    session_id: "codex-turn-agent".into(),
                    agent_kind: AgentKind::Codex,
                    event_name: "UserPromptSubmit".into(),
                    payload: json!({ "prompt": "inspect the repo" }),
                    metadata: json!({}),
                }),
            ],
        )
        .await
        .unwrap();

    let sessions = manager.inner.lock().await;
    let session = sessions.get("codex-turn-agent").unwrap();
    assert!(session.agent_scope.is_none());
    let turn = active_turn_scope(session);
    assert_eq!(turn.name, "codex-turn");
    assert_eq!(turn.scope_type, ScopeType::Agent);
    assert_eq!(
        turn.metadata.as_ref().unwrap()["nemo_relay_scope_role"],
        json!("turn")
    );
}

#[test]
fn apply_start_alias_overrides_conflicting_subagent_id() {
    let mut start = llm_start();
    start.session_id = Some("child-session".into());
    start.subagent_id = Some("stale-subagent".into());
    start.metadata = json!({ "request": "metadata" });
    let alias = SessionAlias::new(
        "parent-session".into(),
        "child-session".into(),
        json!({ "alias": "metadata" }),
    );

    apply_start_alias(&mut start, &alias);

    assert_eq!(start.session_id.as_deref(), Some("parent-session"));
    assert_eq!(start.subagent_id.as_deref(), Some("child-session"));
    assert_eq!(start.metadata["request"], json!("metadata"));
    assert_eq!(start.metadata["alias"], json!("metadata"));
}

#[tokio::test]
async fn turn_output_uses_last_root_owned_llm_response() {
    let subscriber_name = "cli-turn-output-root-llm-test";
    let _ = deregister_subscriber(subscriber_name);
    let captured_output = Arc::new(StdMutex::new(None::<Value>));
    let captured = captured_output.clone();
    register_subscriber(
        subscriber_name,
        Arc::new(move |event| {
            if event.scope_category() == Some(ScopeCategory::End)
                && event.name() == "claude-code-turn"
                && event
                    .metadata()
                    .and_then(|metadata| metadata.get("session_id"))
                    .and_then(Value::as_str)
                    == Some("turn-output")
            {
                *captured.lock().unwrap() = event.output().cloned();
            }
        }),
    )
    .unwrap();

    let manager = SessionManager::new(session_test_config());
    manager
        .apply_events(
            &HeaderMap::new(),
            vec![
                NormalizedEvent::AgentStarted(session_event("turn-output", "SessionStart")),
                NormalizedEvent::PromptSubmitted(SessionEvent {
                    session_id: "turn-output".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "UserPromptSubmit".into(),
                    payload: json!({ "prompt": "summarize" }),
                    metadata: json!({}),
                }),
                NormalizedEvent::SubagentStarted(SubagentEvent {
                    session_id: "turn-output".into(),
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

    let worker_llm = manager
        .start_llm(
            &HeaderMap::new(),
            LlmGatewayStart {
                session_id: Some("turn-output".into()),
                ..llm_start()
            },
        )
        .await
        .unwrap();
    manager
        .end_llm(
            worker_llm,
            json!({ "output_text": "worker answer" }),
            json!({}),
        )
        .await
        .unwrap();
    manager
        .apply_events(
            &HeaderMap::new(),
            vec![NormalizedEvent::SubagentEnded(SubagentEvent {
                session_id: "turn-output".into(),
                agent_kind: AgentKind::ClaudeCode,
                event_name: "SubagentStop".into(),
                subagent_id: "worker".into(),
                payload: json!({ "done": true }),
                metadata: json!({}),
            })],
        )
        .await
        .unwrap();

    let final_response = json!({ "output_text": "final answer" });
    let root_llm = manager
        .start_llm(
            &HeaderMap::new(),
            LlmGatewayStart {
                session_id: Some("turn-output".into()),
                ..llm_start()
            },
        )
        .await
        .unwrap();
    manager
        .end_llm(root_llm, final_response.clone(), json!({}))
        .await
        .unwrap();
    manager
        .apply_events(
            &HeaderMap::new(),
            vec![NormalizedEvent::AgentEnded(session_event(
                "turn-output",
                "SessionEnd",
            ))],
        )
        .await
        .unwrap();

    flush_subscribers().unwrap();
    assert_eq!(*captured_output.lock().unwrap(), Some(final_response));
    deregister_subscriber(subscriber_name).unwrap();
}

#[tokio::test]
async fn new_subagent_claims_first_unhinted_llm_when_siblings_active() {
    let manager = SessionManager::new(session_test_config());
    manager
        .apply_events(
            &HeaderMap::new(),
            vec![
                NormalizedEvent::AgentStarted(session_event("new-subagent-owner", "SessionStart")),
                NormalizedEvent::SubagentStarted(SubagentEvent {
                    session_id: "new-subagent-owner".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "SubagentStart".into(),
                    subagent_id: "worker-1".into(),
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
                session_id: Some("new-subagent-owner".into()),
                ..llm_start()
            },
        )
        .await
        .unwrap();
    manager
        .end_llm(first, json!({ "output_text": "worker-1" }), json!({}))
        .await
        .unwrap();

    manager
        .apply_events(
            &HeaderMap::new(),
            vec![NormalizedEvent::SubagentStarted(SubagentEvent {
                session_id: "new-subagent-owner".into(),
                agent_kind: AgentKind::ClaudeCode,
                event_name: "SubagentStart".into(),
                subagent_id: "worker-2".into(),
                payload: json!({}),
                metadata: json!({}),
            })],
        )
        .await
        .unwrap();

    let worker_2_uuid = {
        let sessions = manager.inner.lock().await;
        sessions
            .get("new-subagent-owner")
            .unwrap()
            .subagents
            .get("worker-2")
            .unwrap()
            .uuid
    };
    let second = manager
        .start_llm(
            &HeaderMap::new(),
            LlmGatewayStart {
                session_id: Some("new-subagent-owner".into()),
                ..llm_start()
            },
        )
        .await
        .unwrap();

    assert_eq!(second.handle.parent_uuid, Some(worker_2_uuid));
    assert_eq!(
        second.handle.metadata.as_ref().unwrap()["llm_correlation_status"],
        json!("subagent_start")
    );
    assert_eq!(
        second.handle.metadata.as_ref().unwrap()["llm_correlation_source"],
        json!("subagent_start")
    );
    manager
        .end_llm(second, json!({ "output_text": "worker-2" }), json!({}))
        .await
        .unwrap();
}

#[tokio::test]
async fn codex_subagent_session_start_uses_transcript_parent_thread() {
    let manager = SessionManager::new(session_test_config());
    let temp = tempfile::tempdir().unwrap();
    let child_transcript = temp.path().join("child.jsonl");
    std::fs::write(
        &child_transcript,
        serde_json::to_string(&json!({
            "type": "session_meta",
            "payload": {
                "id": "child-thread",
                "source": {
                    "subagent": {
                        "thread_spawn": {
                            "parent_thread_id": "parent-thread",
                            "depth": 1,
                            "agent_nickname": "Hume",
                            "agent_role": "explorer"
                        }
                    }
                },
                "thread_source": "subagent",
                "agent_nickname": "Hume",
                "agent_role": "explorer"
            }
        }))
        .unwrap()
            + "\n",
    )
    .unwrap();

    manager
        .apply_events(
            &HeaderMap::new(),
            vec![
                NormalizedEvent::AgentStarted(codex_session_event(
                    "parent-thread",
                    "SessionStart",
                    json!({}),
                )),
                NormalizedEvent::AgentStarted(codex_session_event(
                    "child-thread",
                    "SessionStart",
                    json!({ "transcript_path": child_transcript }),
                )),
            ],
        )
        .await
        .unwrap();

    let sessions = manager.inner.lock().await;
    assert!(sessions.get("child-thread").is_none());
    let parent = sessions.get("parent-thread").unwrap();
    assert!(parent.agent_scope.is_none());
    let turn_uuid = active_turn_uuid(parent);
    assert_eq!(
        parent.subagents.get("child-thread").unwrap().parent_uuid,
        Some(turn_uuid)
    );
    drop(sessions);

    let alias = alignment_alias(&manager, "child-thread").await.unwrap();
    assert_eq!(alias.parent_session_id, "parent-thread");
    assert_eq!(alias.subagent_id, "child-thread");
}

#[tokio::test]
async fn codex_subagent_agent_end_removes_alias_and_closes_scope() {
    let manager = SessionManager::new(session_test_config());
    manager
        .apply_events(
            &HeaderMap::new(),
            vec![
                NormalizedEvent::AgentStarted(codex_session_event(
                    "parent-thread",
                    "SessionStart",
                    json!({}),
                )),
                NormalizedEvent::AgentStarted(SessionEvent {
                    session_id: "child-thread".into(),
                    agent_kind: AgentKind::Codex,
                    event_name: "SessionStart".into(),
                    payload: json!({
                        "source": {
                            "subagent": {
                                "thread_spawn": {
                                    "parent_thread_id": "parent-thread"
                                }
                            }
                        }
                    }),
                    metadata: json!({}),
                }),
            ],
        )
        .await
        .unwrap();
    assert!(has_alignment_alias(&manager, "child-thread").await);

    manager
        .apply_events(
            &HeaderMap::new(),
            vec![NormalizedEvent::AgentEnded(SessionEvent {
                session_id: "child-thread".into(),
                agent_kind: AgentKind::Codex,
                event_name: "SessionEnd".into(),
                payload: json!({ "done": true }),
                metadata: json!({}),
            })],
        )
        .await
        .unwrap();

    assert!(!has_alignment_alias(&manager, "child-thread").await);
    let sessions = manager.inner.lock().await;
    let parent = sessions.get("parent-thread").unwrap();
    assert!(!parent.subagents.contains_key("child-thread"));
}

#[tokio::test]
async fn codex_parent_end_clears_alias_before_late_child_end() {
    let manager = SessionManager::new(session_test_config());
    manager
        .apply_events(
            &HeaderMap::new(),
            vec![
                NormalizedEvent::AgentStarted(codex_session_event(
                    "parent-thread",
                    "SessionStart",
                    json!({}),
                )),
                NormalizedEvent::AgentStarted(SessionEvent {
                    session_id: "child-thread".into(),
                    agent_kind: AgentKind::Codex,
                    event_name: "SessionStart".into(),
                    payload: json!({
                        "source": {
                            "subagent": {
                                "thread_spawn": {
                                    "parent_thread_id": "parent-thread"
                                }
                            }
                        }
                    }),
                    metadata: json!({}),
                }),
            ],
        )
        .await
        .unwrap();
    assert!(has_alignment_alias(&manager, "child-thread").await);

    manager
        .apply_events(
            &HeaderMap::new(),
            vec![NormalizedEvent::AgentEnded(codex_session_event(
                "parent-thread",
                "SessionEnd",
                json!({}),
            ))],
        )
        .await
        .unwrap();

    assert!(!has_alignment_alias(&manager, "child-thread").await);
    assert!(!manager.inner.lock().await.contains_key("parent-thread"));

    manager
        .apply_events(
            &HeaderMap::new(),
            vec![NormalizedEvent::AgentEnded(SessionEvent {
                session_id: "child-thread".into(),
                agent_kind: AgentKind::Codex,
                event_name: "SessionEnd".into(),
                payload: json!({ "late": true }),
                metadata: json!({}),
            })],
        )
        .await
        .unwrap();

    assert!(!has_alignment_alias(&manager, "child-thread").await);
    assert!(!manager.inner.lock().await.contains_key("parent-thread"));
}

#[tokio::test]
async fn codex_child_session_start_waits_for_parent_session() {
    let manager = SessionManager::new(session_test_config());
    manager
        .apply_events(
            &HeaderMap::new(),
            vec![NormalizedEvent::AgentStarted(SessionEvent {
                session_id: "child-thread".into(),
                agent_kind: AgentKind::Codex,
                event_name: "SessionStart".into(),
                payload: json!({
                    "source": {
                        "subagent": {
                            "thread_spawn": {
                                "parent_thread_id": "parent-thread",
                                "agent_nickname": "Late",
                                "agent_role": "worker"
                            }
                        }
                    }
                }),
                metadata: json!({}),
            })],
        )
        .await
        .unwrap();

    assert!(!manager.inner.lock().await.contains_key("child-thread"));
    assert!(has_pending_alignment(&manager, "child-thread").await);

    manager
        .apply_events(
            &HeaderMap::new(),
            vec![NormalizedEvent::AgentStarted(codex_session_event(
                "parent-thread",
                "SessionStart",
                json!({}),
            ))],
        )
        .await
        .unwrap();

    assert!(!has_pending_alignment(&manager, "child-thread").await);
    assert!(has_alignment_alias(&manager, "child-thread").await);
    let sessions = manager.inner.lock().await;
    assert!(!sessions.contains_key("child-thread"));
    assert!(
        sessions
            .get("parent-thread")
            .unwrap()
            .subagents
            .contains_key("child-thread")
    );
}

#[tokio::test]
async fn codex_pending_child_gateway_llm_promotes_parent_subagent() {
    let manager = SessionManager::new(session_test_config());
    manager
        .apply_events(
            &HeaderMap::new(),
            vec![NormalizedEvent::AgentStarted(SessionEvent {
                session_id: "child-thread".into(),
                agent_kind: AgentKind::Codex,
                event_name: "SessionStart".into(),
                payload: json!({
                    "source": {
                        "subagent": {
                            "thread_spawn": {
                                "parent_thread_id": "parent-thread"
                            }
                        }
                    }
                }),
                metadata: json!({}),
            })],
        )
        .await
        .unwrap();

    let active = manager
        .start_llm(
            &HeaderMap::new(),
            LlmGatewayStart {
                session_id: Some("child-thread".into()),
                ..llm_start()
            },
        )
        .await
        .unwrap();

    assert_eq!(active.session_id, "parent-thread");
    assert_eq!(active.owner_subagent_id.as_deref(), Some("child-thread"));
    assert!(!has_pending_alignment(&manager, "child-thread").await);
    assert!(has_alignment_alias(&manager, "child-thread").await);
    {
        let sessions = manager.inner.lock().await;
        assert!(!sessions.contains_key("child-thread"));
        assert!(
            sessions
                .get("parent-thread")
                .unwrap()
                .subagents
                .contains_key("child-thread")
        );
    }

    manager
        .end_llm(active, json!({ "output_text": "done" }), json!({}))
        .await
        .unwrap();
    manager.close_all("test_shutdown").await.unwrap();
}

#[tokio::test]
async fn codex_subagent_start_does_not_reparent_active_child_session() {
    let manager = SessionManager::new(session_test_config());
    manager
        .apply_events(
            &HeaderMap::new(),
            vec![NormalizedEvent::AgentStarted(codex_session_event(
                "parent-thread",
                "SessionStart",
                json!({}),
            ))],
        )
        .await
        .unwrap();

    let active_child_llm = manager
        .start_llm(
            &HeaderMap::new(),
            LlmGatewayStart {
                session_id: Some("child-thread".into()),
                ..llm_start()
            },
        )
        .await
        .unwrap();

    manager
        .apply_events(
            &HeaderMap::new(),
            vec![NormalizedEvent::AgentStarted(SessionEvent {
                session_id: "child-thread".into(),
                agent_kind: AgentKind::Codex,
                event_name: "SessionStart".into(),
                payload: json!({
                    "source": {
                        "subagent": {
                            "thread_spawn": {
                                "parent_thread_id": "parent-thread"
                            }
                        }
                    }
                }),
                metadata: json!({}),
            })],
        )
        .await
        .unwrap();

    assert!(!has_alignment_alias(&manager, "child-thread").await);
    {
        let sessions = manager.inner.lock().await;
        assert!(sessions.contains_key("child-thread"));
        assert!(
            !sessions
                .get("parent-thread")
                .unwrap()
                .subagents
                .contains_key("child-thread")
        );
    }

    manager
        .end_llm(
            active_child_llm,
            json!({ "output_text": "child" }),
            json!({}),
        )
        .await
        .unwrap();
    manager.close_all("test_shutdown").await.unwrap();
}

#[tokio::test]
async fn hermes_subagent_start_does_not_reparent_active_child_session() {
    let manager = SessionManager::new(session_test_config());
    let headers = HeaderMap::new();

    for payload in [
        json!({
            "hook_event_name": "on_session_start",
            "session_id": "parent-session"
        }),
        json!({
            "hook_event_name": "on_session_start",
            "session_id": "child-session"
        }),
        json!({
            "hook_event_name": "pre_api_request",
            "session_id": "child-session",
            "extra": {
                "task_id": "child-task",
                "api_call_count": 1,
                "provider": "custom",
                "model": "qwen",
                "request": { "body": { "model": "qwen" } }
            }
        }),
        json!({
            "hook_event_name": "subagent_start",
            "session_id": "parent-session",
            "extra": {
                "child_session_id": "child-session",
                "child_subagent_id": "sa-1",
                "parent_turn_id": "parent-turn-1"
            }
        }),
    ] {
        let outcome = crate::adapters::hermes::adapt(payload, &headers);
        manager
            .apply_events(&headers, outcome.events)
            .await
            .unwrap();
    }

    assert!(!has_alignment_alias(&manager, "child-session").await);
    {
        let sessions = manager.inner.lock().await;
        assert!(sessions.contains_key("child-session"));
        assert!(
            !sessions
                .get("parent-session")
                .unwrap()
                .subagents
                .contains_key("sa-1")
        );
        assert!(
            sessions
                .get("child-session")
                .unwrap()
                .llms
                .contains_key("child-session:child-task:1")
        );
    }

    manager.close_all("test_shutdown").await.unwrap();
}

#[tokio::test]
async fn codex_aliased_hook_llm_routes_to_subagent_scope() {
    let manager = SessionManager::new(session_test_config());
    manager
        .apply_events(
            &HeaderMap::new(),
            vec![
                NormalizedEvent::AgentStarted(codex_session_event(
                    "parent-thread",
                    "SessionStart",
                    json!({}),
                )),
                NormalizedEvent::AgentStarted(SessionEvent {
                    session_id: "child-thread".into(),
                    agent_kind: AgentKind::Codex,
                    event_name: "SessionStart".into(),
                    payload: json!({
                        "source": {
                            "subagent": {
                                "thread_spawn": {
                                    "parent_thread_id": "parent-thread"
                                }
                            }
                        }
                    }),
                    metadata: json!({}),
                }),
                NormalizedEvent::LlmStarted(LlmEvent {
                    session_id: "child-thread".into(),
                    agent_kind: AgentKind::Codex,
                    event_name: "PreLlm".into(),
                    api_call_id: "hook-llm".into(),
                    provider: "openai.responses".into(),
                    model_name: Some("gpt-test".into()),
                    request: json!({ "input": "hello" }),
                    response: Value::Null,
                    metadata: json!({}),
                }),
            ],
        )
        .await
        .unwrap();

    let sessions = manager.inner.lock().await;
    let parent = sessions.get("parent-thread").unwrap();
    let subagent_uuid = parent.subagents.get("child-thread").unwrap().uuid;
    let handle = parent.llms.get("hook-llm").unwrap();
    assert_eq!(handle.parent_uuid, Some(subagent_uuid));
    assert_eq!(
        handle.metadata.as_ref().unwrap()["llm_correlation_status"],
        json!("session_alias")
    );
    assert_eq!(
        handle.metadata.as_ref().unwrap()["llm_correlation_subagent_id"],
        json!("child-thread")
    );
    drop(sessions);

    manager.close_all("test_shutdown").await.unwrap();
}

#[tokio::test]
async fn codex_subagent_gateway_llm_routes_to_parent_subagent() {
    let manager = SessionManager::new(session_test_config());
    manager
        .apply_events(
            &HeaderMap::new(),
            vec![
                NormalizedEvent::AgentStarted(codex_session_event(
                    "parent-thread",
                    "SessionStart",
                    json!({}),
                )),
                NormalizedEvent::AgentStarted(SessionEvent {
                    session_id: "child-thread".into(),
                    agent_kind: AgentKind::Codex,
                    event_name: "SessionStart".into(),
                    payload: json!({
                        "source": {
                            "subagent": {
                                "thread_spawn": {
                                    "parent_thread_id": "parent-thread",
                                    "agent_nickname": "Bohr",
                                    "agent_role": "explorer"
                                }
                            }
                        }
                    }),
                    metadata: json!({}),
                }),
            ],
        )
        .await
        .unwrap();

    let subagent_uuid = {
        let sessions = manager.inner.lock().await;
        sessions
            .get("parent-thread")
            .unwrap()
            .subagents
            .get("child-thread")
            .unwrap()
            .uuid
    };

    let active = manager
        .start_llm(
            &HeaderMap::new(),
            LlmGatewayStart {
                session_id: Some("child-thread".into()),
                ..llm_start()
            },
        )
        .await
        .unwrap();

    assert_eq!(active.session_id, "parent-thread");
    assert_eq!(active.owner_subagent_id.as_deref(), Some("child-thread"));
    assert_eq!(active.handle.parent_uuid, Some(subagent_uuid));
    assert_eq!(
        active.handle.metadata.as_ref().unwrap()["llm_correlation_status"],
        json!("explicit")
    );
    assert_eq!(
        active.handle.metadata.as_ref().unwrap()["llm_correlation_subagent_id"],
        json!("child-thread")
    );
    assert_eq!(
        active.handle.metadata.as_ref().unwrap()["codex_parent_thread_id"],
        json!("parent-thread")
    );
    assert_eq!(
        active.handle.metadata.as_ref().unwrap()["codex_subagent_session_id"],
        json!("child-thread")
    );
    assert_eq!(
        active.handle.metadata.as_ref().unwrap()["agent_nickname"],
        json!("Bohr")
    );

    manager
        .end_llm(active, json!({ "output_text": "child" }), json!({}))
        .await
        .unwrap();

    let sticky = manager
        .start_llm(
            &HeaderMap::new(),
            LlmGatewayStart {
                session_id: Some("parent-thread".into()),
                ..llm_start()
            },
        )
        .await
        .unwrap();

    assert_eq!(sticky.session_id, "parent-thread");
    assert_eq!(sticky.owner_subagent_id.as_deref(), Some("child-thread"));
    assert_eq!(sticky.handle.parent_uuid, Some(subagent_uuid));
    assert_eq!(
        sticky.handle.metadata.as_ref().unwrap()["llm_correlation_status"],
        json!("sticky_last_owner")
    );
    assert_eq!(
        sticky.handle.metadata.as_ref().unwrap()["llm_correlation_subagent_id"],
        json!("child-thread")
    );
    assert_eq!(
        sticky.handle.metadata.as_ref().unwrap()["codex_parent_thread_id"],
        json!("parent-thread")
    );
    assert_eq!(
        sticky.handle.metadata.as_ref().unwrap()["codex_subagent_session_id"],
        json!("child-thread")
    );
    assert_eq!(
        sticky.handle.metadata.as_ref().unwrap()["agent_nickname"],
        json!("Bohr")
    );

    manager
        .end_llm(sticky, json!({ "output_text": "child-again" }), json!({}))
        .await
        .unwrap();

    manager
        .apply_events(
            &HeaderMap::new(),
            vec![
                NormalizedEvent::ToolStarted(ToolEvent {
                    session_id: "parent-thread".into(),
                    agent_kind: AgentKind::Codex,
                    event_name: "PreToolUse".into(),
                    tool_call_id: "tool-1".into(),
                    tool_name: "exec_command".into(),
                    subagent_id: Some("child-thread".into()),
                    arguments: json!({ "cmd": "true" }),
                    result: Value::Null,
                    status: None,
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::ToolEnded(ToolEvent {
                    session_id: "parent-thread".into(),
                    agent_kind: AgentKind::Codex,
                    event_name: "PostToolUse".into(),
                    tool_call_id: "tool-1".into(),
                    tool_name: "exec_command".into(),
                    subagent_id: Some("child-thread".into()),
                    arguments: Value::Null,
                    result: json!({ "ok": true }),
                    status: Some("success".into()),
                    payload: json!({}),
                    metadata: json!({}),
                }),
            ],
        )
        .await
        .unwrap();

    let tool_owned = manager
        .start_llm(
            &HeaderMap::new(),
            LlmGatewayStart {
                session_id: Some("parent-thread".into()),
                ..llm_start()
            },
        )
        .await
        .unwrap();

    assert_eq!(tool_owned.handle.parent_uuid, Some(subagent_uuid));
    assert_eq!(
        tool_owned.handle.metadata.as_ref().unwrap()["llm_correlation_status"],
        json!("recent_tool_owner")
    );
    assert_eq!(
        tool_owned.handle.metadata.as_ref().unwrap()["codex_parent_thread_id"],
        json!("parent-thread")
    );
    assert_eq!(
        tool_owned.handle.metadata.as_ref().unwrap()["codex_subagent_session_id"],
        json!("child-thread")
    );

    manager
        .end_llm(
            tool_owned,
            json!({ "output_text": "after-tool" }),
            json!({}),
        )
        .await
        .unwrap();

    manager
        .apply_events(
            &HeaderMap::new(),
            vec![NormalizedEvent::LlmHint(LlmHintEvent {
                session_id: "parent-thread".into(),
                agent_kind: AgentKind::Codex,
                event_name: "AgentMessageDelta".into(),
                subagent_id: Some("child-thread".into()),
                agent_id: None,
                agent_type: Some("explorer".into()),
                conversation_id: None,
                generation_id: Some("generation-1".into()),
                request_id: None,
                model: Some("gpt-test".into()),
                payload: json!({}),
                metadata: json!({}),
            })],
        )
        .await
        .unwrap();

    let hinted = manager
        .start_llm(
            &HeaderMap::new(),
            LlmGatewayStart {
                session_id: Some("parent-thread".into()),
                generation_id: Some("generation-1".into()),
                ..llm_start()
            },
        )
        .await
        .unwrap();

    assert_eq!(hinted.handle.parent_uuid, Some(subagent_uuid));
    assert_eq!(
        hinted.handle.metadata.as_ref().unwrap()["llm_correlation_status"],
        json!("single_hint")
    );
    assert_eq!(
        hinted.handle.metadata.as_ref().unwrap()["codex_parent_thread_id"],
        json!("parent-thread")
    );
    assert_eq!(
        hinted.handle.metadata.as_ref().unwrap()["codex_subagent_session_id"],
        json!("child-thread")
    );

    manager
        .end_llm(hinted, json!({ "output_text": "after-hint" }), json!({}))
        .await
        .unwrap();
}

#[tokio::test]
async fn writes_atif_on_session_end_from_plugin_config() {
    let _guard = OBSERVABILITY_PLUGIN_TEST_LOCK.lock().await;
    let temp = tempfile::tempdir().unwrap();
    let atif_dir = temp.path().join("atif");
    install_test_atif_plugin(&atif_dir).await;
    let config = GatewayConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://127.0.0.1".into(),

        anthropic_base_url: "http://127.0.0.1".into(),
        metadata: None,
        plugin_config: None,
    };
    let manager = SessionManager::new(config);
    let mut headers = HeaderMap::new();
    headers.insert(
        "x-nemo-relay-session-metadata",
        r#"{"team":"coverage"}"#.parse().unwrap(),
    );
    headers.insert("x-nemo-relay-gateway-mode", "required".parse().unwrap());

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

    clear_plugin_configuration().unwrap();
    let atif = read_atif_for_session(&atif_dir, "atif-session");
    assert!(
        atif["extra"]["observed_events"]
            .as_array()
            .is_some_and(|events| events.len() >= 2)
    );
    assert_eq!(
        atif["extra"]["observed_events"][0]["name"],
        json!("codex-turn")
    );
}

#[tokio::test]
async fn codex_stop_snapshots_atif_without_session_end() {
    let _guard = OBSERVABILITY_PLUGIN_TEST_LOCK.lock().await;
    let temp = tempfile::tempdir().unwrap();
    let atif_dir = temp.path().join("atif");
    install_test_atif_plugin(&atif_dir).await;
    let manager = SessionManager::new(session_test_config());
    let headers = HeaderMap::new();

    start_codex_prompt_turn(&manager, &headers, "codex-atif-stop").await;
    run_codex_responses_tool_activity(&manager, &headers, "codex-atif-stop").await;
    assert!(
        std::fs::read_dir(&atif_dir).unwrap().next().is_none(),
        "Codex ATIF should wait for Stop before writing a per-turn snapshot"
    );

    stop_codex_turn(&manager, &headers, "codex-atif-stop").await;

    clear_plugin_configuration().unwrap();
    let atif = read_atif_for_session(&atif_dir, "codex-atif-stop");
    assert_eq!(atif["schema_version"], json!("ATIF-v1.7"));
    assert_eq!(atif["trajectory_id"], atif["session_id"]);
    assert!(atif["subagent_trajectories"].is_null());
    assert_eq!(atif["final_metrics"]["total_steps"], json!(2));

    let observed = atif["extra"]["observed_events"].as_array().unwrap();
    assert_eq!(observed.len(), 6);
    assert!(observed.iter().all(|event| {
        event["metadata"]["hook_event_name"] != json!("sessionEnd")
            && event["metadata"]["hook_event_name"] != json!("session_end")
    }));
    let turn_start = observed
        .iter()
        .find(|event| {
            event["name"] == "codex-turn"
                && event["category"] == "agent"
                && event["scope_category"] == "start"
        })
        .expect("Codex turn start should be observed");
    let turn_end = observed
        .iter()
        .find(|event| {
            event["name"] == "codex-turn"
                && event["category"] == "agent"
                && event["scope_category"] == "end"
        })
        .expect("Codex Stop should close the turn scope");
    assert_eq!(turn_start["uuid"], atif["session_id"]);
    assert_eq!(turn_end["uuid"], atif["session_id"]);
    assert_eq!(
        turn_end["data"]["output"][0]["call_id"],
        json!("tool-call-1")
    );

    let steps = atif["steps"].as_array().unwrap();
    assert_eq!(steps.len(), 2);
    assert_eq!(steps[0]["source"], json!("user"));
    assert_eq!(steps[0]["message"], json!("Inspect the repository."));
    assert_eq!(
        steps[0]["extra"]["ancestry"]["parent_name"],
        json!("codex-turn")
    );
    assert_eq!(steps[1]["source"], json!("agent"));
    assert_eq!(steps[1]["model_name"], json!("gpt-test"));
    assert_eq!(steps[1]["llm_call_count"], json!(1));
    assert_eq!(
        steps[1]["tool_calls"][0],
        json!({
            "tool_call_id": "tool-call-1",
            "function_name": "Read",
            "arguments": { "file_path": "README.md" },
            "extra": { "status": "completed" }
        })
    );
    assert_eq!(
        steps[1]["observation"]["results"][0]["source_call_id"],
        json!("tool-call-1")
    );
    assert_eq!(
        steps[1]["observation"]["results"][0]["content"],
        json!(null)
    );
    assert_eq!(
        steps[1]["observation"]["results"][0]["extra"]["tool_result"],
        json!({ "content": "hello" })
    );
    assert_eq!(
        steps[1]["extra"]["tool_ancestry"][0]["parent_name"],
        json!("codex-turn")
    );
    assert_eq!(
        steps[1]["extra"]["tool_invocations"][0]["invocation_id"],
        json!("tool-call-1")
    );
}

#[tokio::test]
async fn codex_openinference_spans_match_shared_contract() {
    let _guard = OBSERVABILITY_PLUGIN_TEST_LOCK.lock().await;
    let (endpoint, collector) = start_otlp_collector();
    install_test_openinference_plugin(&endpoint).await;
    let manager = SessionManager::new(session_test_config());
    let headers = HeaderMap::new();

    start_codex_prompt_turn(&manager, &headers, "codex-openinference").await;
    run_codex_responses_tool_activity(&manager, &headers, "codex-openinference").await;
    stop_codex_turn(&manager, &headers, "codex-openinference").await;

    clear_plugin_configuration().unwrap();
    let (request, body) = collector.join().unwrap();
    assert!(request.starts_with("POST /v1/traces HTTP/1.1"));
    assert!(request.contains("content-type: application/x-protobuf"));

    let body = String::from_utf8_lossy(&body);
    for expected in [
        "openinference.span.kind",
        "AGENT",
        "LLM",
        "TOOL",
        "nemo_relay.uuid",
        "nemo_relay.parent_uuid",
        "codex-turn",
        "openai.responses",
        "Read",
        "gpt-test",
        "codex-openinference",
        "tool-call-1",
        "tool_call.function.arguments",
        "Requested tools: Read",
    ] {
        assert!(
            body.contains(expected),
            "expected OpenInference export body to contain {expected:?}; body: {body}"
        );
    }
    assert!(!body.contains("sessionEnd"));
}

#[tokio::test]
async fn duplicate_agent_end_does_not_overwrite_atif_with_empty_session() {
    // Regression test: hermes-agent and other integrations can emit terminal hooks more than once
    // per session. Without idempotency in `end_agent`, the second AgentEnded would re-open an
    // empty agent scope via `ensure_agent_started`, close it, and write an empty ATIF on top of
    // the just-written real trajectory.
    let _guard = OBSERVABILITY_PLUGIN_TEST_LOCK.lock().await;
    let temp = tempfile::tempdir().unwrap();
    let atif_dir = temp.path().join("atif");
    install_test_atif_plugin(&atif_dir).await;
    let config = GatewayConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://127.0.0.1".into(),

        anthropic_base_url: "http://127.0.0.1".into(),
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

    let first = read_atif_for_session(&atif_dir, "dup-end");
    let first_events = first["extra"]["observed_events"].as_array().unwrap().len();
    assert!(
        first_events > 0,
        "first AgentEnded should produce observed ATIF events"
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

    clear_plugin_configuration().unwrap();
    let second = read_atif_for_session(&atif_dir, "dup-end");
    let second_events = second["extra"]["observed_events"].as_array().unwrap().len();
    assert_eq!(
        first_events, second_events,
        "duplicate AgentEnded must not change the ATIF event count"
    );
}

#[tokio::test]
async fn writes_hermes_api_hook_usage_to_atif_metrics() {
    let _guard = OBSERVABILITY_PLUGIN_TEST_LOCK.lock().await;
    let temp = tempfile::tempdir().unwrap();
    let atif_dir = temp.path().join("atif");
    install_test_atif_plugin(&atif_dir).await;
    let config = GatewayConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://127.0.0.1".into(),

        anthropic_base_url: "http://127.0.0.1".into(),
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

    clear_plugin_configuration().unwrap();
    let atif = read_atif_for_session(&atif_dir, "hermes-usage");
    assert!(atif["subagent_trajectories"].is_null());
    assert_eq!(atif["steps"][1]["metrics"]["prompt_tokens"], json!(10));
    assert_eq!(atif["steps"][1]["metrics"]["completion_tokens"], json!(5));
    assert_eq!(atif["steps"][1]["metrics"]["cached_tokens"], json!(3));
    assert_eq!(atif["final_metrics"]["total_prompt_tokens"], json!(10));
    assert_eq!(atif["final_metrics"]["total_completion_tokens"], json!(5));
    assert_eq!(atif["final_metrics"]["total_cached_tokens"], json!(3));
}

#[tokio::test]
async fn hermes_exact_api_hooks_write_atif_request_response_and_cost() {
    let _guard = OBSERVABILITY_PLUGIN_TEST_LOCK.lock().await;
    let temp = tempfile::tempdir().unwrap();
    let atif_dir = temp.path().join("atif");
    install_test_atif_plugin(&atif_dir).await;
    let config = session_test_config();
    let manager = SessionManager::new(config);
    let headers = HeaderMap::new();

    for payload in [
        json!({
            "hook_event_name": "on_session_start",
            "session_id": "hermes-exact-atif"
        }),
        json!({
            "hook_event_name": "pre_api_request",
            "session_id": "hermes-exact-atif",
            "extra": {
                "task_id": "task-1",
                "api_call_count": 1,
                "provider": "custom",
                "model": "qwen",
                "request": {
                    "body": {
                        "model": "qwen",
                        "temperature": 0.1,
                        "messages": [
                            { "role": "user", "content": "summarize this file" }
                        ],
                        "tools": [
                            {
                                "type": "function",
                                "function": { "name": "read_file" }
                            }
                        ]
                    }
                }
            }
        }),
        json!({
            "hook_event_name": "post_api_request",
            "session_id": "hermes-exact-atif",
            "extra": {
                "task_id": "task-1",
                "api_call_count": 1,
                "provider": "custom",
                "model": "qwen",
                "response": {
                    "assistant_message": {
                        "role": "assistant",
                        "content": "summary ready"
                    },
                    "usage": {
                        "prompt_tokens": 11,
                        "completion_tokens": 7,
                        "cost": { "total": 0.0042 }
                    },
                    "finish_reason": "stop"
                }
            }
        }),
        json!({
            "hook_event_name": "on_session_finalize",
            "session_id": "hermes-exact-atif"
        }),
    ] {
        let outcome = crate::adapters::hermes::adapt(payload, &headers);
        manager
            .apply_events(&headers, outcome.events)
            .await
            .unwrap();
    }

    clear_plugin_configuration().unwrap();
    let atif = read_atif_for_session(&atif_dir, "hermes-exact-atif");
    let observed_events = atif["extra"]["observed_events"].as_array().unwrap();
    assert_eq!(atif["steps"][0]["message"], json!("summarize this file"));
    assert_eq!(
        atif["steps"][0]["extra"]["llm_request"]["temperature"],
        json!(0.1)
    );
    assert_eq!(
        atif["steps"][0]["extra"]["llm_request"]["tools"][0]["function"]["name"],
        json!("read_file")
    );
    assert_eq!(atif["steps"][1]["message"], json!("summary ready"));
    assert_eq!(
        atif["steps"][1]["extra"]["llm_response"]["content"],
        json!("summary ready")
    );
    assert_eq!(
        atif["steps"][1]["extra"]["llm_response"]["usage"]["cost"]["total"],
        json!(0.0042)
    );
    assert_eq!(atif["steps"][1]["metrics"]["prompt_tokens"], json!(11));
    assert_eq!(atif["steps"][1]["metrics"]["completion_tokens"], json!(7));
    assert_eq!(atif["steps"][1]["metrics"]["cost_usd"], json!(0.0042));
    assert_eq!(atif["final_metrics"]["total_cost_usd"], json!(0.0042));
    assert!(observed_events.iter().any(|event| {
        event["metadata"]["hook_event_name"] == json!("pre_api_request")
            && event["metadata"]["provider_payload_exact"] == json!(true)
            && event["metadata"]["fidelity_source"] == json!("hermes_api_hooks_sanitized")
    }));
    assert!(observed_events.iter().any(|event| {
        event["metadata"]["hook_event_name"] == json!("post_api_request")
            && event["metadata"]["provider_payload_exact"] == json!(true)
            && event["metadata"]["fidelity_source"] == json!("hermes_api_hooks_sanitized")
    }));
}

#[tokio::test]
async fn hermes_api_request_error_writes_atif_error_step_and_fidelity() {
    let _guard = OBSERVABILITY_PLUGIN_TEST_LOCK.lock().await;
    let temp = tempfile::tempdir().unwrap();
    let atif_dir = temp.path().join("atif");
    install_test_atif_plugin(&atif_dir).await;
    let config = session_test_config();
    let manager = SessionManager::new(config);
    let headers = HeaderMap::new();

    for payload in [
        json!({
            "hook_event_name": "on_session_start",
            "session_id": "hermes-error"
        }),
        json!({
            "hook_event_name": "pre_api_request",
            "session_id": "hermes-error",
            "extra": {
                "task_id": "task-err",
                "api_request_id": "turn-1:api:3",
                "api_call_count": 3,
                "provider": "custom",
                "model": "qwen",
                "request": {
                    "method": "POST",
                    "body": {
                        "model": "qwen",
                        "messages": [
                            { "role": "user", "content": "hello" }
                        ]
                    }
                }
            }
        }),
        json!({
            "hook_event_name": "api_request_error",
            "session_id": "hermes-error",
            "extra": {
                "task_id": "task-err",
                "api_request_id": "turn-1:api:3",
                "api_call_count": 3,
                "provider": "custom",
                "model": "qwen",
                "status_code": 502,
                "retry_count": 1,
                "max_retries": 2,
                "retryable": true,
                "reason": "upstream",
                "error": {
                    "type": "BadGateway",
                    "message": "gateway upstream error"
                }
            }
        }),
        json!({
            "hook_event_name": "on_session_finalize",
            "session_id": "hermes-error"
        }),
    ] {
        let outcome = crate::adapters::hermes::adapt(payload, &headers);
        manager
            .apply_events(&headers, outcome.events)
            .await
            .unwrap();
    }

    clear_plugin_configuration().unwrap();
    let atif = read_atif_for_session(&atif_dir, "hermes-error");
    let steps = atif["steps"].as_array().unwrap();
    assert_eq!(steps.len(), 2);
    assert_eq!(steps[0]["message"], json!("hello"));
    assert_eq!(steps[1]["source"], json!("agent"));
    assert_eq!(steps[1]["extra"]["llm_response"]["status_code"], json!(502));
    assert_eq!(steps[1]["extra"]["llm_response"]["retry_count"], json!(1));
    assert_eq!(steps[1]["extra"]["llm_response"]["retryable"], json!(true));
    assert_eq!(
        steps[1]["extra"]["llm_response"]["reason"],
        json!("upstream")
    );
    assert_eq!(
        steps[1]["extra"]["llm_response"]["error"]["message"],
        json!("gateway upstream error")
    );
    let observed_events = atif["extra"]["observed_events"].as_array().unwrap();
    assert!(
        observed_events.len() >= 4,
        "expected Hermes error trajectory to keep observed events, got {}",
        serde_json::to_string_pretty(&atif["extra"]["observed_events"]).unwrap()
    );
    let error_event = observed_events
        .iter()
        .find(|event| {
            event["scope_category"] == json!("end")
                && event["metadata"]["api_call_id"] == json!("turn-1:api:3")
        })
        .unwrap();
    let request_event = observed_events
        .iter()
        .find(|event| {
            event["scope_category"] == json!("start")
                && event["metadata"]["api_call_id"] == json!("turn-1:api:3")
        })
        .unwrap();
    assert_eq!(
        request_event["metadata"]["provider_payload_exact"],
        json!(true)
    );
    assert_eq!(
        request_event["metadata"]["fidelity_source"],
        json!("hermes_api_hooks_sanitized")
    );
    assert_eq!(
        error_event["metadata"]["provider_payload_exact"],
        json!(false)
    );
    assert_eq!(
        error_event["metadata"]["fidelity_source"],
        json!("hermes_api_hooks")
    );
}

#[tokio::test]
async fn hermes_lossy_api_hooks_write_atif_fidelity_markers() {
    let _guard = OBSERVABILITY_PLUGIN_TEST_LOCK.lock().await;
    let temp = tempfile::tempdir().unwrap();
    let atif_dir = temp.path().join("atif");
    install_test_atif_plugin(&atif_dir).await;
    let config = session_test_config();
    let manager = SessionManager::new(config);
    let headers = HeaderMap::new();

    for payload in [
        json!({
            "hook_event_name": "on_session_start",
            "session_id": "hermes-lossy-atif"
        }),
        json!({
            "hook_event_name": "pre_api_request",
            "session_id": "hermes-lossy-atif",
            "extra": {
                "task_id": "task-1",
                "api_call_count": 1,
                "provider": "custom",
                "model": "qwen",
                "message_count": 1,
                "tool_count": 0,
                "request_char_count": 42
            }
        }),
        json!({
            "hook_event_name": "post_api_request",
            "session_id": "hermes-lossy-atif",
            "extra": {
                "task_id": "task-1",
                "api_call_count": 1,
                "provider": "custom",
                "model": "qwen",
                "assistant_content_chars": 13,
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 3
                }
            }
        }),
        json!({
            "hook_event_name": "on_session_finalize",
            "session_id": "hermes-lossy-atif"
        }),
    ] {
        let outcome = crate::adapters::hermes::adapt(payload, &headers);
        manager
            .apply_events(&headers, outcome.events)
            .await
            .unwrap();
    }

    clear_plugin_configuration().unwrap();
    let atif = read_atif_for_session(&atif_dir, "hermes-lossy-atif");
    let observed_events = atif["extra"]["observed_events"].as_array().unwrap();
    assert_eq!(
        atif["steps"][0]["extra"]["llm_request"]["fidelity"]["provider_payload_exact"],
        json!(false)
    );
    assert_eq!(
        atif["steps"][0]["extra"]["llm_request"]["fidelity"]["source"],
        json!("hermes_pre_api_request")
    );
    assert_eq!(
        atif["steps"][0]["extra"]["llm_request"]["request_char_count"],
        json!(42)
    );
    assert_eq!(
        atif["steps"][1]["extra"]["llm_response"]["assistant_content_chars"],
        json!(13)
    );
    assert!(atif["steps"][1]["extra"]["llm_response"]["content"].is_null());
    assert_eq!(atif["steps"][1]["metrics"]["prompt_tokens"], json!(5));
    assert_eq!(atif["steps"][1]["metrics"]["completion_tokens"], json!(3));
    assert!(observed_events.iter().any(|event| {
        event["metadata"]["hook_event_name"] == json!("pre_api_request")
            && event["metadata"]["provider_payload_exact"] == json!(false)
            && event["metadata"]["fidelity_source"] == json!("hermes_api_hooks")
    }));
    assert!(observed_events.iter().any(|event| {
        event["metadata"]["hook_event_name"] == json!("post_api_request")
            && event["metadata"]["provider_payload_exact"] == json!(false)
            && event["metadata"]["fidelity_source"] == json!("hermes_api_hooks")
    }));
}

#[tokio::test]
async fn hermes_uncorrelatable_pre_tool_call_does_not_create_shutdown_trajectory() {
    let _guard = OBSERVABILITY_PLUGIN_TEST_LOCK.lock().await;
    let temp = tempfile::tempdir().unwrap();
    let atif_dir = temp.path().join("atif");
    install_test_atif_plugin(&atif_dir).await;
    let config = session_test_config();
    let manager = SessionManager::new(config);
    let headers = HeaderMap::new();

    for payload in [
        json!({
            "hook_event_name": "on_session_start",
            "session_id": "hermes-main"
        }),
        json!({
            "hook_event_name": "pre_tool_call",
            "task_id": "task-1",
            "tool_name": "terminal",
            "tool_input": { "command": "pwd" }
        }),
        json!({
            "hook_event_name": "on_session_finalize",
            "session_id": "hermes-main"
        }),
    ] {
        let outcome = crate::adapters::hermes::adapt(payload, &headers);
        manager
            .apply_events(&headers, outcome.events)
            .await
            .unwrap();
    }

    manager.close_all("gateway_shutdown").await.unwrap();
    clear_plugin_configuration().unwrap();

    let trajectories: Vec<Value> = std::fs::read_dir(&atif_dir)
        .unwrap()
        .filter_map(Result::ok)
        .map(|entry| serde_json::from_slice(&std::fs::read(entry.path()).unwrap()).unwrap())
        .collect();
    let serialized = serde_json::to_string(&trajectories).unwrap();
    assert!(serialized.contains("hermes-main"));
    assert!(!serialized.contains("task-1"));
    assert!(!serialized.contains("gateway_shutdown"));
}

#[tokio::test]
async fn hermes_turn_end_snapshots_atif_without_boundary_system_step() {
    let _guard = OBSERVABILITY_PLUGIN_TEST_LOCK.lock().await;
    let temp = tempfile::tempdir().unwrap();
    let atif_dir = temp.path().join("atif");
    install_test_atif_plugin(&atif_dir).await;
    let config = session_test_config();
    let manager = SessionManager::new(config);
    let headers = HeaderMap::new();

    for payload in [
        json!({
            "hook_event_name": "on_session_start",
            "session_id": "hermes-clean"
        }),
        json!({
            "hook_event_name": "pre_api_request",
            "session_id": "hermes-clean",
            "extra": {
                "task_id": "task-1",
                "api_call_count": 1,
                "provider": "custom",
                "model": "qwen",
                "request": {
                    "method": "POST",
                    "body": {
                        "model": "qwen",
                        "messages": [
                            { "role": "user", "content": "hello" }
                        ]
                    }
                }
            }
        }),
        json!({
            "hook_event_name": "post_api_request",
            "session_id": "hermes-clean",
            "extra": {
                "task_id": "task-1",
                "api_call_count": 1,
                "provider": "custom",
                "model": "qwen",
                "response": {
                    "assistant_message": {
                        "role": "assistant",
                        "content": "done"
                    },
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5
                    }
                }
            }
        }),
        json!({
            "hook_event_name": "on_session_end",
            "session_id": "hermes-clean"
        }),
    ] {
        let outcome = crate::adapters::hermes::adapt(payload, &headers);
        manager
            .apply_events(&headers, outcome.events)
            .await
            .unwrap();
    }

    clear_plugin_configuration().unwrap();
    let atif = read_atif_for_session(&atif_dir, "hermes-clean");
    assert!(atif["subagent_trajectories"].is_null());
    assert_eq!(atif["steps"].as_array().unwrap().len(), 2);
    assert_eq!(atif["steps"][0]["source"], json!("user"));
    assert_eq!(atif["steps"][1]["source"], json!("agent"));
    assert!(
        atif["steps"].as_array().unwrap().iter().all(|step| {
            step["source"] != json!("system")
                || step["message"].as_object().is_some_and(|message| {
                    !message.is_empty() && message.contains_key("hook_event_name")
                })
        }),
        "Hermes hook system steps must not be anonymous or empty: {}",
        serde_json::to_string_pretty(&atif["steps"]).unwrap()
    );
}

#[tokio::test]
async fn hermes_task_id_tool_hooks_reuse_api_session() {
    let config = session_test_config();
    let manager = SessionManager::new(config);
    let headers = HeaderMap::new();

    for payload in [
        json!({
            "hook_event_name": "on_session_start",
            "session_id": "hermes-main"
        }),
        json!({
            "hook_event_name": "pre_api_request",
            "session_id": "hermes-main",
            "extra": {
                "task_id": "task-1",
                "api_call_count": 1,
                "provider": "custom",
                "model": "qwen",
                "request": {
                    "body": {
                        "model": "qwen",
                        "messages": [
                            { "role": "user", "content": "read file" }
                        ]
                    }
                }
            }
        }),
    ] {
        let outcome = crate::adapters::hermes::adapt(payload, &headers);
        manager
            .apply_events(&headers, outcome.events)
            .await
            .unwrap();
    }

    let pre_tool = crate::adapters::hermes::adapt(
        json!({
            "hook_event_name": "pre_tool_call",
            "session_id": "hermes-main",
            "tool_name": "read_file",
            "tool_input": { "path": "README.md" },
            "extra": {
                "task_id": "task-1",
                "tool_call_id": "tool-1"
            }
        }),
        &headers,
    );
    manager
        .apply_events(&headers, pre_tool.events)
        .await
        .unwrap();

    {
        let sessions = manager.inner.lock().await;
        assert!(sessions.contains_key("hermes-main"));
        assert!(
            !sessions.contains_key("task-1"),
            "Hermes tool hooks keyed by task_id should not create a duplicate session"
        );
        let session = sessions.get("hermes-main").unwrap();
        assert!(
            !session.tools.is_empty(),
            "pre_tool_call should open an active tool before post_tool_call runs"
        );
    }

    let post_tool = crate::adapters::hermes::adapt(
        json!({
            "hook_event_name": "post_tool_call",
            "session_id": "hermes-main",
            "tool_name": "read_file",
            "tool_input": { "path": "README.md" },
            "tool_response": { "content": "hello" },
            "extra": {
                "task_id": "task-1",
                "tool_call_id": "provider-tool-1"
            }
        }),
        &headers,
    );
    manager
        .apply_events(&headers, post_tool.events)
        .await
        .unwrap();

    let sessions = manager.inner.lock().await;
    let session = sessions.get("hermes-main").unwrap();
    assert!(
        session.tools.is_empty(),
        "post_tool_call should close the matching pre_tool_call even when call IDs differ"
    );
}

#[tokio::test]
async fn hermes_post_tool_call_writes_atif_observation_with_source_call_id() {
    let _guard = OBSERVABILITY_PLUGIN_TEST_LOCK.lock().await;
    let temp = tempfile::tempdir().unwrap();
    let atif_dir = temp.path().join("atif");
    install_test_atif_plugin(&atif_dir).await;
    let config = session_test_config();
    let manager = SessionManager::new(config);
    let headers = HeaderMap::new();

    for payload in [
        json!({
            "hook_event_name": "on_session_start",
            "session_id": "hermes-tool-result"
        }),
        json!({
            "hook_event_name": "pre_api_request",
            "session_id": "hermes-tool-result",
            "extra": {
                "task_id": "task-1",
                "api_request_id": "turn-1:api:1",
                "api_call_count": 1,
                "provider": "custom",
                "model": "qwen",
                "request": {
                    "body": {
                        "model": "qwen",
                        "messages": [
                            { "role": "user", "content": "search for needle" }
                        ],
                        "tools": [
                            {
                                "type": "function",
                                "function": { "name": "search_files" }
                            }
                        ]
                    }
                }
            }
        }),
        json!({
            "hook_event_name": "post_api_request",
            "session_id": "hermes-tool-result",
            "extra": {
                "task_id": "task-1",
                "api_request_id": "turn-1:api:1",
                "api_call_count": 1,
                "provider": "custom",
                "model": "qwen",
                "response": {
                    "assistant_message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-search-1",
                                "type": "function",
                                "function": {
                                    "name": "search_files",
                                    "arguments": "{\"query\":\"needle\"}"
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls"
                }
            }
        }),
        json!({
            "hook_event_name": "pre_tool_call",
            "session_id": "hermes-tool-result",
            "tool_name": "search_files",
            "tool_input": { "query": "needle" },
            "extra": {
                "task_id": "task-1",
                "tool_call_id": "call-search-1"
            }
        }),
        json!({
            "hook_event_name": "post_tool_call",
            "session_id": "hermes-tool-result",
            "tool_name": "search_files",
            "tool_input": { "query": "needle" },
            "tool_response": { "total_count": 6 },
            "extra": {
                "task_id": "task-1",
                "tool_call_id": "call-search-1"
            }
        }),
        json!({
            "hook_event_name": "on_session_finalize",
            "session_id": "hermes-tool-result"
        }),
    ] {
        let outcome = crate::adapters::hermes::adapt(payload, &headers);
        manager
            .apply_events(&headers, outcome.events)
            .await
            .unwrap();
    }

    clear_plugin_configuration().unwrap();
    let atif = read_atif_for_session(&atif_dir, "hermes-tool-result");
    let steps = atif["steps"].as_array().unwrap();
    assert_eq!(steps.len(), 2);
    assert_eq!(steps[0]["message"], json!("search for needle"));

    let agent = &steps[1];
    assert_eq!(agent["source"], json!("agent"));
    assert_eq!(
        agent["tool_calls"][0]["tool_call_id"],
        json!("call-search-1")
    );
    assert_eq!(
        agent["tool_calls"][0]["function_name"],
        json!("search_files")
    );
    assert_eq!(
        agent["observation"]["results"][0]["source_call_id"],
        json!("call-search-1")
    );
    assert!(agent["observation"]["results"][0].get("content").is_none());
    assert_eq!(
        agent["observation"]["results"][0]["extra"]["tool_result"]["total_count"],
        json!(6)
    );
}

#[tokio::test]
async fn hermes_orphan_subagent_stop_exports_readable_mark_with_lineage() {
    let _guard = OBSERVABILITY_PLUGIN_TEST_LOCK.lock().await;
    let temp = tempfile::tempdir().unwrap();
    let atif_dir = temp.path().join("atif");
    install_test_atif_plugin(&atif_dir).await;
    let config = session_test_config();
    let manager = SessionManager::new(config);
    let headers = HeaderMap::new();

    for payload in [
        json!({
            "hook_event_name": "on_session_start",
            "session_id": "hermes-orphan"
        }),
        json!({
            "hook_event_name": "subagent_stop",
            "session_id": "hermes-orphan",
            "extra": {
                "subagent_id": "worker-1",
                "child_status": "completed"
            }
        }),
        json!({
            "hook_event_name": "on_session_finalize",
            "session_id": "hermes-orphan"
        }),
    ] {
        let outcome = crate::adapters::hermes::adapt(payload, &headers);
        manager
            .apply_events(&headers, outcome.events)
            .await
            .unwrap();
    }

    clear_plugin_configuration().unwrap();
    let atif = read_atif_for_session(&atif_dir, "hermes-orphan");
    assert!(atif["subagent_trajectories"].is_null());
    let root_steps = atif["steps"].as_array().unwrap();
    assert_eq!(root_steps.len(), 1);
    assert_eq!(root_steps[0]["source"], json!("system"));
    assert_eq!(root_steps[0]["message"], json!("subagent_stop"));
    assert_eq!(
        root_steps[0]["extra"]["event_payload"]["hook_event_name"],
        json!("subagent_stop")
    );
    assert_eq!(
        root_steps[0]["extra"]["event_payload"]["extra"]["subagent_id"],
        json!("worker-1")
    );
    assert_eq!(
        root_steps[0]["extra"]["ancestry"]["function_name"],
        json!("subagent_end_without_start")
    );
    assert_eq!(
        root_steps[0]["extra"]["ancestry"]["parent_name"],
        json!("hermes-turn")
    );
}

#[tokio::test]
async fn hermes_subagent_child_session_embeds_non_empty_atif_trajectory() {
    let _guard = OBSERVABILITY_PLUGIN_TEST_LOCK.lock().await;
    let temp = tempfile::tempdir().unwrap();
    let atif_dir = temp.path().join("atif");
    install_test_atif_plugin(&atif_dir).await;
    let config = session_test_config();
    let manager = SessionManager::new(config);
    let headers = HeaderMap::new();

    for payload in [
        json!({
            "hook_event_name": "on_session_start",
            "session_id": "parent-session"
        }),
        json!({
            "hook_event_name": "subagent_start",
            "session_id": "parent-session",
            "extra": {
                "child_goal": "read plugin yaml",
                "child_role": "leaf",
                "child_session_id": "child-session",
                "child_subagent_id": "sa-1",
                "parent_turn_id": "parent-turn-1",
                "telemetry_schema_version": "hermes.observer.v1"
            }
        }),
        json!({
            "hook_event_name": "on_session_start",
            "session_id": "child-session"
        }),
        json!({
            "hook_event_name": "pre_api_request",
            "session_id": "child-session",
            "extra": {
                "task_id": "child-task",
                "api_call_count": 1,
                "provider": "custom",
                "model": "qwen",
                "request": {
                    "body": {
                        "model": "qwen",
                        "messages": [
                            { "role": "user", "content": "read plugin yaml" }
                        ]
                    }
                }
            }
        }),
        json!({
            "hook_event_name": "post_api_request",
            "session_id": "child-session",
            "extra": {
                "task_id": "child-task",
                "api_call_count": 1,
                "provider": "custom",
                "model": "qwen",
                "response": {
                    "assistant_message": {
                        "role": "assistant",
                        "content": "name: nemo_flow"
                    },
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 2
                    }
                }
            }
        }),
        json!({
            "hook_event_name": "on_session_end",
            "session_id": "child-session"
        }),
        json!({
            "hook_event_name": "subagent_stop",
            "session_id": "parent-session",
            "extra": {
                "child_session_id": "child-session",
                "child_status": "completed"
            }
        }),
        json!({
            "hook_event_name": "on_session_finalize",
            "session_id": "parent-session"
        }),
    ] {
        let outcome = crate::adapters::hermes::adapt(payload, &headers);
        manager
            .apply_events(&headers, outcome.events)
            .await
            .unwrap();
    }

    clear_plugin_configuration().unwrap();
    let atif = read_atif_for_session(&atif_dir, "parent-session");
    assert!(
        atif["subagent_trajectories"]
            .as_array()
            .is_some_and(|trajectories| !trajectories.is_empty()),
        "parent ATIF must include at least one embedded subagent trajectory: {}",
        serde_json::to_string_pretty(&atif).unwrap()
    );
    let child = &atif["subagent_trajectories"][0];
    assert_eq!(child["session_id"], json!("child-session"));
    assert!(
        !child["steps"].as_array().unwrap().is_empty(),
        "embedded Hermes child trajectory must contain the child session work: {}",
        serde_json::to_string_pretty(child).unwrap()
    );
    assert_eq!(child["steps"][0]["source"], json!("user"));
    assert_eq!(child["steps"][1]["source"], json!("agent"));
    assert!(child["subagent_trajectories"].is_null());
    assert!(
        !serde_json::to_string(&atif)
            .unwrap()
            .contains("subagent_end_without_start")
    );
}

#[tokio::test]
async fn hermes_routed_provider_payloads_write_exact_atif_trajectory() {
    let _guard = OBSERVABILITY_PLUGIN_TEST_LOCK.lock().await;
    let temp = tempfile::tempdir().unwrap();
    let atif_dir = temp.path().join("atif");
    install_test_atif_plugin(&atif_dir).await;
    let manager = SessionManager::new(session_test_config());
    let headers = HeaderMap::new();

    manager
        .apply_events(
            &headers,
            vec![NormalizedEvent::AgentStarted(SessionEvent {
                session_id: "hermes-routed".into(),
                agent_kind: AgentKind::Hermes,
                event_name: "on_session_start".into(),
                payload: json!({}),
                metadata: json!({}),
            })],
        )
        .await
        .unwrap();

    let anthropic = manager
        .start_llm(
            &headers,
            LlmGatewayStart {
                session_id: Some("hermes-routed".into()),
                provider: "anthropic.messages".into(),
                model_name: Some("claude-sonnet-4".into()),
                subagent_id: None,
                conversation_id: None,
                generation_id: None,
                request_id: Some("msg-request".into()),
                request: LlmRequest {
                    headers: Map::new(),
                    content: json!({
                        "model": "claude-sonnet-4",
                        "messages": [{"role": "user", "content": "Find the file."}],
                        "tools": [{"name": "search", "input_schema": {"type": "object"}}]
                    }),
                },
                streaming: false,
                metadata: json!({ "gateway_path": "/v1/messages" }),
            },
        )
        .await
        .unwrap();
    manager
        .end_llm(
            anthropic,
            json!({
                "id": "msg_01",
                "type": "message",
                "content": [
                    {"type": "text", "text": "I will search."},
                    {"type": "tool_use", "id": "toolu_01", "name": "search", "input": {"query": "file"}}
                ],
                "usage": {
                    "input_tokens": 11,
                    "output_tokens": 7,
                    "cache_read_input_tokens": 3,
                    "cost": {"total": 0.0042}
                }
            }),
            json!({}),
        )
        .await
        .unwrap();

    let responses = manager
        .start_llm(
            &headers,
            LlmGatewayStart {
                session_id: Some("hermes-routed".into()),
                provider: "openai.responses".into(),
                model_name: Some("gpt-4o".into()),
                subagent_id: None,
                conversation_id: None,
                generation_id: None,
                request_id: Some("resp-request".into()),
                request: LlmRequest {
                    headers: Map::new(),
                    content: json!({
                        "model": "gpt-4o",
                        "input": "Find the weather.",
                        "tools": [{"type": "function", "name": "get_weather"}]
                    }),
                },
                streaming: false,
                metadata: json!({ "gateway_path": "/v1/responses" }),
            },
        )
        .await
        .unwrap();
    manager
        .end_llm(
            responses,
            json!({
                "id": "resp_1",
                "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": "I will check the weather."}]},
                    {"type": "function_call", "call_id": "call_weather_1", "name": "get_weather", "arguments": "{\"city\":\"SF\"}"}
                ],
                "usage": {
                    "input_tokens": 75,
                    "output_tokens": 20,
                    "total_tokens": 95,
                    "input_tokens_details": {"cached_tokens": 10},
                    "cost_usd": 0.005
                }
            }),
            json!({}),
        )
        .await
        .unwrap();

    let chat = manager
        .start_llm(
            &headers,
            LlmGatewayStart {
                session_id: Some("hermes-routed".into()),
                provider: "openai.chat_completions".into(),
                model_name: Some("gpt-4o".into()),
                subagent_id: None,
                conversation_id: None,
                generation_id: None,
                request_id: Some("chat-request".into()),
                request: LlmRequest {
                    headers: Map::new(),
                    content: json!({
                        "model": "gpt-4o",
                        "messages": [{"role": "user", "content": "Inspect the files."}],
                        "tools": [{"type": "function", "function": {"name": "read"}}]
                    }),
                },
                streaming: false,
                metadata: json!({ "gateway_path": "/v1/chat/completions" }),
            },
        )
        .await
        .unwrap();
    manager
        .end_llm(
            chat,
            json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "I will inspect.",
                        "tool_calls": [{"id": "call_read_1", "function": {"name": "read", "arguments": "{\"path\":\"api.py\"}"}}]
                    }
                }],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 4,
                    "total_tokens": 7,
                    "prompt_tokens_details": {"cached_tokens": 2},
                    "cost_usd": 0.001
                }
            }),
            json!({}),
        )
        .await
        .unwrap();

    manager
        .apply_events(
            &headers,
            vec![NormalizedEvent::AgentEnded(SessionEvent {
                session_id: "hermes-routed".into(),
                agent_kind: AgentKind::Hermes,
                event_name: "on_session_finalize".into(),
                payload: json!({}),
                metadata: json!({}),
            })],
        )
        .await
        .unwrap();

    clear_plugin_configuration().unwrap();
    let atif = read_atif_for_session(&atif_dir, "hermes-routed");
    let steps = atif["steps"].as_array().unwrap();
    assert_eq!(steps.len(), 6);

    assert_eq!(steps[0]["message"], json!("Find the file."));
    assert_eq!(steps[1]["message"], json!("I will search."));
    assert_eq!(steps[1]["tool_calls"][0]["tool_call_id"], json!("toolu_01"));
    assert_eq!(steps[1]["metrics"]["prompt_tokens"], json!(11));
    assert_eq!(steps[1]["metrics"]["cached_tokens"], json!(3));
    assert_eq!(steps[1]["metrics"]["cost_usd"], json!(0.0042));

    assert_eq!(steps[2]["message"], json!("Find the weather."));
    assert_eq!(steps[3]["message"], json!("I will check the weather."));
    assert_eq!(
        steps[3]["tool_calls"][0]["tool_call_id"],
        json!("call_weather_1")
    );
    assert_eq!(steps[3]["metrics"]["prompt_tokens"], json!(75));
    assert_eq!(steps[3]["metrics"]["cached_tokens"], json!(10));
    assert_eq!(steps[3]["metrics"]["cost_usd"], json!(0.005));

    assert_eq!(steps[4]["message"], json!("Inspect the files."));
    assert_eq!(steps[5]["message"], json!("I will inspect."));
    assert_eq!(
        steps[5]["tool_calls"][0]["tool_call_id"],
        json!("call_read_1")
    );
    assert_eq!(steps[5]["metrics"]["prompt_tokens"], json!(3));
    assert_eq!(steps[5]["metrics"]["cached_tokens"], json!(2));
    assert_eq!(steps[5]["metrics"]["cost_usd"], json!(0.001));

    assert_eq!(atif["final_metrics"]["total_prompt_tokens"], json!(89));
    assert_eq!(atif["final_metrics"]["total_completion_tokens"], json!(31));
    assert_eq!(atif["final_metrics"]["total_cached_tokens"], json!(15));
    assert_eq!(atif["final_metrics"]["total_cost_usd"], json!(0.0102));
}

#[tokio::test]
async fn empty_hook_marks_do_not_create_empty_atif_steps() {
    let _guard = OBSERVABILITY_PLUGIN_TEST_LOCK.lock().await;
    let temp = tempfile::tempdir().unwrap();
    let atif_dir = temp.path().join("atif");
    install_test_atif_plugin(&atif_dir).await;
    let config = session_test_config();
    let manager = SessionManager::new(config);

    manager
        .apply_events(
            &HeaderMap::new(),
            vec![
                NormalizedEvent::AgentStarted(SessionEvent {
                    session_id: "empty-mark".into(),
                    agent_kind: AgentKind::Hermes,
                    event_name: "on_session_start".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::HookMark(SessionEvent {
                    session_id: "empty-mark".into(),
                    agent_kind: AgentKind::Hermes,
                    event_name: "unknown".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::AgentEnded(SessionEvent {
                    session_id: "empty-mark".into(),
                    agent_kind: AgentKind::Hermes,
                    event_name: "on_session_finalize".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
            ],
        )
        .await
        .unwrap();

    clear_plugin_configuration().unwrap();
    let atif = read_atif_for_session(&atif_dir, "empty-mark");
    assert!(atif["steps"].as_array().unwrap().is_empty());
    assert!(atif["subagent_trajectories"].is_null());
}

#[tokio::test]
async fn handles_out_of_order_subagent_and_tool_end_events() {
    let config = GatewayConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://127.0.0.1".into(),

        anthropic_base_url: "http://127.0.0.1".into(),
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
    let config = session_test_config();
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
}

#[tokio::test]
async fn out_of_order_started_subagent_end_does_not_leak_scope() {
    let config = GatewayConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://127.0.0.1".into(),

        anthropic_base_url: "http://127.0.0.1".into(),
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
async fn claude_startup_probe_does_not_open_null_input_turn() {
    let subscriber_name = "cli-claude-startup-probe-turn-test";
    let _ = deregister_subscriber(subscriber_name);
    let captured_turn_starts = Arc::new(StdMutex::new(Vec::<Value>::new()));
    let captured = captured_turn_starts.clone();
    register_subscriber(
        subscriber_name,
        Arc::new(move |event| {
            if event.scope_category() == Some(ScopeCategory::Start)
                && event.name() == "claude-code-turn"
                && event
                    .metadata()
                    .and_then(|metadata| metadata.get("session_id"))
                    .and_then(Value::as_str)
                    == Some("claude-probe")
            {
                captured.lock().unwrap().push(json!({
                    "input": event.input().cloned().unwrap_or(Value::Null),
                    "metadata": event.metadata().cloned().unwrap_or(Value::Null)
                }));
            }
        }),
    )
    .unwrap();

    let manager = SessionManager::new(session_test_config());
    manager
        .apply_events(
            &HeaderMap::new(),
            vec![NormalizedEvent::AgentStarted(session_event(
                "claude-probe",
                "SessionStart",
            ))],
        )
        .await
        .unwrap();

    let prep = manager
        .prepare_gateway_call(
            &HeaderMap::new(),
            LlmGatewayStart {
                session_id: Some("claude-probe".into()),
                provider: "anthropic.messages".into(),
                model_name: Some("claude-opus-4-8[1m]".into()),
                request: LlmRequest {
                    headers: Map::from_iter([(
                        "x-claude-code-session-id".to_string(),
                        json!("claude-probe"),
                    )]),
                    content: json!({
                        "model": "claude-opus-4-8[1m]",
                        "max_tokens": 1,
                        "messages": [
                            {
                                "role": "user",
                                "content": "test"
                            }
                        ]
                    }),
                },
                ..llm_start()
            },
        )
        .await
        .unwrap();
    assert!(prep.parent.is_none());
    assert_eq!(
        prep.metadata["llm_correlation_status"],
        json!("pre_turn_probe")
    );
    assert_eq!(
        prep.metadata["llm_correlation_source"],
        json!("claude_startup_probe")
    );
    manager
        .finish_gateway_call(&prep.session_id, prep.prune_empty_session_on_finish)
        .await;

    manager
        .apply_events(
            &HeaderMap::new(),
            vec![NormalizedEvent::PromptSubmitted(SessionEvent {
                session_id: "claude-probe".into(),
                agent_kind: AgentKind::ClaudeCode,
                event_name: "UserPromptSubmit".into(),
                payload: json!({ "prompt": "list contents of this dir" }),
                metadata: json!({}),
            })],
        )
        .await
        .unwrap();

    {
        let sessions = manager.inner.lock().await;
        let session = sessions.get("claude-probe").expect("session retained");
        assert!(
            session.turn_scope.is_some(),
            "prompt should open a Claude turn after the pre-turn probe"
        );
    }

    flush_subscribers().unwrap();
    let starts = captured_turn_starts.lock().unwrap().clone();
    assert_eq!(starts.len(), 1, "expected one user-visible Claude turn");
    assert_eq!(
        starts[0]["input"],
        json!({ "prompt": "list contents of this dir" }),
        "startup probe must not create a null-input Claude turn"
    );
    assert_eq!(starts[0]["metadata"]["turn_index"], json!(1));
    assert_eq!(starts[0]["metadata"]["turn_source"], json!("user_prompt"));

    deregister_subscriber(subscriber_name).unwrap();
}

#[tokio::test]
async fn claude_startup_probe_only_session_is_pruned_after_finish() {
    let manager = SessionManager::new(session_test_config());
    let prep = manager
        .prepare_gateway_call(&HeaderMap::new(), claude_startup_probe_start("probe-only"))
        .await
        .unwrap();

    assert!(prep.bypass_managed_pipeline);
    assert!(prep.prune_empty_session_on_finish);
    assert!(manager.inner.lock().await.contains_key("probe-only"));

    manager
        .finish_gateway_call(&prep.session_id, prep.prune_empty_session_on_finish)
        .await;
    assert!(!manager.inner.lock().await.contains_key("probe-only"));

    let next = manager
        .prepare_gateway_call(
            &HeaderMap::new(),
            LlmGatewayStart {
                session_id: None,
                ..llm_start()
            },
        )
        .await
        .unwrap();
    assert_eq!(
        next.session_id, "gateway-gateway",
        "probe-only sessions must not become the single-active fallback"
    );
}

#[tokio::test]
async fn claude_orphan_subagent_stop_after_closed_turn_does_not_open_null_turn() {
    let _guard = OBSERVABILITY_PLUGIN_TEST_LOCK.lock().await;
    let temp = tempfile::tempdir().unwrap();
    let atif_dir = temp.path().join("atif");
    install_test_atif_plugin(&atif_dir).await;
    let subscriber_name = "cli-claude-orphan-subagent-stop-no-null-turn-test";
    let _ = deregister_subscriber(subscriber_name);
    let captured_events = Arc::new(StdMutex::new(Vec::<Value>::new()));
    let captured = captured_events.clone();
    register_subscriber(
        subscriber_name,
        Arc::new(move |event| {
            let event_session_id = event
                .metadata()
                .and_then(|metadata| metadata.get("session_id"))
                .and_then(Value::as_str);
            if event_session_id != Some("claude-orphan-stop") {
                return;
            }
            if event.name() == "claude-code-turn" {
                captured.lock().unwrap().push(json!({
                    "kind": "turn",
                    "scope_category": event.scope_category(),
                    "input": event.input().cloned().unwrap_or(Value::Null),
                    "output": event.output().cloned().unwrap_or(Value::Null),
                    "metadata": event.metadata().cloned().unwrap_or(Value::Null)
                }));
            } else if event.name() == "subagent_end_without_start" {
                captured.lock().unwrap().push(json!({
                    "kind": "orphan_mark",
                    "data": event.data().cloned().unwrap_or(Value::Null),
                    "metadata": event.metadata().cloned().unwrap_or(Value::Null)
                }));
            }
        }),
    )
    .unwrap();

    let manager = SessionManager::new(session_test_config());
    manager
        .apply_events(
            &HeaderMap::new(),
            vec![
                NormalizedEvent::AgentStarted(session_event("claude-orphan-stop", "SessionStart")),
                NormalizedEvent::PromptSubmitted(SessionEvent {
                    session_id: "claude-orphan-stop".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "UserPromptSubmit".into(),
                    payload: json!({ "prompt": "thanks!" }),
                    metadata: json!({}),
                }),
            ],
        )
        .await
        .unwrap();
    let active_llm = manager
        .start_llm(
            &HeaderMap::new(),
            llm_start_with_messages_task("claude-orphan-stop", "thanks!"),
        )
        .await
        .unwrap();
    manager
        .end_llm(
            active_llm,
            json!({
                "id": "msg_thanks",
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "content": [
                    {
                        "type": "text",
                        "text": "You're welcome!"
                    }
                ],
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 2,
                    "output_tokens": 4
                }
            }),
            json!({}),
        )
        .await
        .unwrap();
    manager
        .apply_events(
            &HeaderMap::new(),
            vec![
                NormalizedEvent::TurnEnded(SessionEvent {
                    session_id: "claude-orphan-stop".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "Stop".into(),
                    payload: json!({ "content": "You're welcome!" }),
                    metadata: json!({}),
                }),
                NormalizedEvent::SubagentEnded(SubagentEvent {
                    session_id: "claude-orphan-stop".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "SubagentStop".into(),
                    subagent_id: "missing-worker".into(),
                    payload: json!({
                        "hook_event_name": "SubagentStop",
                        "last_assistant_message": "add the event logs to .gitignore"
                    }),
                    metadata: json!({
                        "hook_event_name": "SubagentStop",
                        "agent_id": "missing-worker"
                    }),
                }),
            ],
        )
        .await
        .unwrap();

    let closed = manager
        .close_idle_sessions_at(
            Instant::now() + AGENT_IDLE_TIMEOUT + Duration::from_secs(1),
            AGENT_IDLE_TIMEOUT,
            "idle_timeout",
        )
        .await
        .unwrap();

    flush_subscribers().unwrap();
    clear_plugin_configuration().unwrap();
    let events = captured_events.lock().unwrap().clone();
    let turn_starts: Vec<_> = events
        .iter()
        .filter(|event| {
            event["kind"] == json!("turn") && event["scope_category"] == json!(ScopeCategory::Start)
        })
        .collect();
    let idle_turn_closes: Vec<_> = events
        .iter()
        .filter(|event| {
            event["kind"] == json!("turn")
                && event["scope_category"] == json!(ScopeCategory::End)
                && event["output"]["status"] == json!("idle_timeout")
        })
        .collect();

    assert_eq!(closed, 0, "orphan SubagentStop must not open an idle turn");
    assert_eq!(
        turn_starts.len(),
        1,
        "orphan SubagentStop must not create a second Claude turn: {events:#?}"
    );
    assert_eq!(turn_starts[0]["input"], json!({ "prompt": "thanks!" }));
    assert_eq!(
        idle_turn_closes.len(),
        0,
        "orphan SubagentStop must not create a turn later closed by idle timeout: {events:#?}"
    );
    assert!(
        events
            .iter()
            .all(|event| event["kind"] != json!("orphan_mark")),
        "uncorrelatable Claude SubagentStop should not emit a turn-scoped orphan mark: {events:#?}"
    );

    let atif = read_atif_for_session(&atif_dir, "claude-orphan-stop");
    assert_eq!(atif["steps"].as_array().unwrap().len(), 2);
    assert!(
        !serde_json::to_string(&atif)
            .unwrap()
            .contains("subagent_end_without_start"),
        "ATIF should not include uncorrelatable Claude orphan stop diagnostics: {}",
        serde_json::to_string_pretty(&atif).unwrap()
    );

    deregister_subscriber(subscriber_name).unwrap();
}

#[tokio::test]
async fn llm_lifecycle_uses_single_active_hook_session_when_header_is_missing() {
    let config = GatewayConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://127.0.0.1".into(),

        anthropic_base_url: "http://127.0.0.1".into(),
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

    let turn_uuid = {
        let sessions = manager.inner.lock().await;
        active_turn_uuid(sessions.get("ambiguous-session").unwrap())
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

    assert_eq!(active.handle.parent_uuid, Some(turn_uuid));
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
async fn root_llm_hint_does_not_stick_over_later_subagent() {
    let manager = SessionManager::new(session_test_config());
    manager
        .apply_events(
            &HeaderMap::new(),
            vec![
                NormalizedEvent::AgentStarted(session_event("root-sticky", "SessionStart")),
                NormalizedEvent::LlmHint(LlmHintEvent {
                    session_id: "root-sticky".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "UserPromptSubmit".into(),
                    subagent_id: None,
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

    let first = manager
        .start_llm(
            &HeaderMap::new(),
            LlmGatewayStart {
                session_id: Some("root-sticky".into()),
                ..llm_start()
            },
        )
        .await
        .unwrap();
    assert_eq!(
        first.handle.metadata.as_ref().unwrap()["llm_correlation_status"],
        json!("single_hint")
    );
    manager
        .end_llm(first, json!({ "output_text": "root" }), json!({}))
        .await
        .unwrap();

    manager
        .apply_events(
            &HeaderMap::new(),
            vec![NormalizedEvent::SubagentStarted(SubagentEvent {
                session_id: "root-sticky".into(),
                agent_kind: AgentKind::ClaudeCode,
                event_name: "SubagentStart".into(),
                subagent_id: "worker".into(),
                payload: json!({}),
                metadata: json!({}),
            })],
        )
        .await
        .unwrap();

    let worker_uuid = {
        let sessions = manager.inner.lock().await;
        sessions
            .get("root-sticky")
            .unwrap()
            .subagents
            .get("worker")
            .unwrap()
            .uuid
    };
    let second = manager
        .start_llm(
            &HeaderMap::new(),
            LlmGatewayStart {
                session_id: Some("root-sticky".into()),
                ..llm_start()
            },
        )
        .await
        .unwrap();

    assert_eq!(second.handle.parent_uuid, Some(worker_uuid));
    assert_eq!(
        second.handle.metadata.as_ref().unwrap()["llm_correlation_status"],
        json!("active_subagent")
    );
    manager
        .end_llm(second, json!({ "output_text": "worker" }), json!({}))
        .await
        .unwrap();
}

#[tokio::test]
async fn explicit_subagent_tool_owner_claims_next_unhinted_llm() {
    let manager = SessionManager::new(session_test_config());
    manager
        .apply_events(
            &HeaderMap::new(),
            vec![
                NormalizedEvent::AgentStarted(session_event("tool-owner", "SessionStart")),
                NormalizedEvent::SubagentStarted(SubagentEvent {
                    session_id: "tool-owner".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "SubagentStart".into(),
                    subagent_id: "worker-1".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::SubagentStarted(SubagentEvent {
                    session_id: "tool-owner".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "SubagentStart".into(),
                    subagent_id: "worker-2".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::ToolStarted(ToolEvent {
                    session_id: "tool-owner".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "PreToolUse".into(),
                    tool_call_id: "tool-1".into(),
                    tool_name: "Read".into(),
                    subagent_id: Some("worker-1".into()),
                    arguments: json!({ "file_path": "README.md" }),
                    result: Value::Null,
                    status: None,
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::ToolEnded(ToolEvent {
                    session_id: "tool-owner".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "PostToolUse".into(),
                    tool_call_id: "tool-1".into(),
                    tool_name: "Read".into(),
                    subagent_id: Some("worker-1".into()),
                    arguments: Value::Null,
                    result: json!({ "ok": true }),
                    status: Some("success".into()),
                    payload: json!({}),
                    metadata: json!({}),
                }),
            ],
        )
        .await
        .unwrap();

    let worker_uuid = {
        let sessions = manager.inner.lock().await;
        sessions
            .get("tool-owner")
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
                session_id: Some("tool-owner".into()),
                ..llm_start()
            },
        )
        .await
        .unwrap();

    assert_eq!(active.handle.parent_uuid, Some(worker_uuid));
    assert_eq!(
        active.handle.metadata.as_ref().unwrap()["llm_correlation_status"],
        json!("recent_tool_owner")
    );
    assert_eq!(
        active.handle.metadata.as_ref().unwrap()["llm_correlation_source"],
        json!("tool_owner")
    );
    manager
        .end_llm(active, json!({ "output_text": "again" }), json!({}))
        .await
        .unwrap();
}

#[tokio::test]
async fn request_affinity_pairs_parallel_subagents_across_provider_formats() {
    let manager = SessionManager::new(session_test_config());
    manager
        .apply_events(
            &HeaderMap::new(),
            vec![
                NormalizedEvent::AgentStarted(SessionEvent {
                    session_id: "parallel-affinity".into(),
                    agent_kind: AgentKind::Codex,
                    event_name: "SessionStart".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::SubagentStarted(SubagentEvent {
                    session_id: "parallel-affinity".into(),
                    agent_kind: AgentKind::Codex,
                    event_name: "SubagentStart".into(),
                    subagent_id: "python-worker".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::SubagentStarted(SubagentEvent {
                    session_id: "parallel-affinity".into(),
                    agent_kind: AgentKind::Codex,
                    event_name: "SubagentStart".into(),
                    subagent_id: "go-worker".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
            ],
        )
        .await
        .unwrap();

    let python_first = manager
        .start_llm(
            &HeaderMap::new(),
            LlmGatewayStart {
                subagent_id: Some("python-worker".into()),
                ..llm_start_with_responses_task(
                    "parallel-affinity",
                    "Very thorough analysis of the python/nemo_relay package.",
                )
            },
        )
        .await
        .unwrap();
    manager
        .end_llm(python_first, json!({ "output_text": "python" }), json!({}))
        .await
        .unwrap();

    let go_first = manager
        .start_llm(
            &HeaderMap::new(),
            LlmGatewayStart {
                subagent_id: Some("go-worker".into()),
                ..llm_start_with_messages_task(
                    "parallel-affinity",
                    "Very thorough analysis of the go/nemo_relay binding.",
                )
            },
        )
        .await
        .unwrap();
    manager
        .end_llm(go_first, json!({ "output_text": "go" }), json!({}))
        .await
        .unwrap();

    manager
        .apply_events(
            &HeaderMap::new(),
            vec![
                NormalizedEvent::ToolStarted(ToolEvent {
                    session_id: "parallel-affinity".into(),
                    agent_kind: AgentKind::Codex,
                    event_name: "PreToolUse".into(),
                    tool_call_id: "go-tool".into(),
                    tool_name: "Read".into(),
                    subagent_id: Some("go-worker".into()),
                    arguments: json!({ "file_path": "go/nemo_relay/nemo_relay.go" }),
                    result: Value::Null,
                    status: None,
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::ToolEnded(ToolEvent {
                    session_id: "parallel-affinity".into(),
                    agent_kind: AgentKind::Codex,
                    event_name: "PostToolUse".into(),
                    tool_call_id: "go-tool".into(),
                    tool_name: "Read".into(),
                    subagent_id: Some("go-worker".into()),
                    arguments: Value::Null,
                    result: json!({ "ok": true }),
                    status: Some("success".into()),
                    payload: json!({}),
                    metadata: json!({}),
                }),
            ],
        )
        .await
        .unwrap();

    let python_uuid = {
        let sessions = manager.inner.lock().await;
        sessions
            .get("parallel-affinity")
            .unwrap()
            .subagents
            .get("python-worker")
            .unwrap()
            .uuid
    };
    let python_later = manager
        .start_llm(
            &HeaderMap::new(),
            llm_start_with_chat_completion_task(
                "parallel-affinity",
                "Very thorough analysis of the python/nemo_relay package.",
            ),
        )
        .await
        .unwrap();

    assert_eq!(python_later.handle.parent_uuid, Some(python_uuid));
    assert_eq!(
        python_later.handle.metadata.as_ref().unwrap()["llm_correlation_status"],
        json!("request_affinity")
    );
    assert_eq!(
        python_later.handle.metadata.as_ref().unwrap()["llm_correlation_source"],
        json!("request_payload")
    );
}

#[tokio::test]
async fn claude_agent_tool_completion_closes_subagents_before_final_llm() {
    let manager = SessionManager::new(session_test_config());
    manager
        .apply_events(
            &HeaderMap::new(),
            vec![
                NormalizedEvent::AgentStarted(session_event("agent-tool-finish", "SessionStart")),
                NormalizedEvent::SubagentStarted(SubagentEvent {
                    session_id: "agent-tool-finish".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "SubagentStart".into(),
                    subagent_id: "worker-1".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::SubagentStarted(SubagentEvent {
                    session_id: "agent-tool-finish".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "SubagentStart".into(),
                    subagent_id: "worker-2".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::ToolEnded(ToolEvent {
                    session_id: "agent-tool-finish".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "PostToolUse".into(),
                    tool_call_id: "agent-tool-1".into(),
                    tool_name: "Agent".into(),
                    subagent_id: None,
                    arguments: Value::Null,
                    result: json!({
                        "agentId": "worker-1",
                        "status": "completed"
                    }),
                    status: Some("completed".into()),
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::ToolEnded(ToolEvent {
                    session_id: "agent-tool-finish".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "PostToolUse".into(),
                    tool_call_id: "agent-tool-2".into(),
                    tool_name: "Agent".into(),
                    subagent_id: None,
                    arguments: Value::Null,
                    result: json!({
                        "agentId": "worker-2",
                        "status": "completed"
                    }),
                    status: Some("completed".into()),
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::SubagentEnded(SubagentEvent {
                    session_id: "agent-tool-finish".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "SubagentStop".into(),
                    subagent_id: "worker-2".into(),
                    payload: json!({ "duplicate": true }),
                    metadata: json!({}),
                }),
            ],
        )
        .await
        .unwrap();

    let turn_uuid = {
        let sessions = manager.inner.lock().await;
        let session = sessions.get("agent-tool-finish").unwrap();
        assert!(session.subagents.is_empty());
        assert!(session.subagent_stacks.is_empty());
        active_turn_uuid(session)
    };
    let final_llm = manager
        .start_llm(
            &HeaderMap::new(),
            LlmGatewayStart {
                session_id: Some("agent-tool-finish".into()),
                ..llm_start()
            },
        )
        .await
        .unwrap();

    assert_eq!(final_llm.handle.parent_uuid, Some(turn_uuid));
    assert_eq!(
        final_llm.handle.metadata.as_ref().unwrap()["llm_correlation_status"],
        json!("agent_fallback")
    );
    manager
        .end_llm(final_llm, json!({ "output_text": "final" }), json!({}))
        .await
        .unwrap();
}

#[tokio::test]
async fn claude_agent_tool_async_launch_keeps_subagent_open_for_later_hooks() {
    let manager = SessionManager::new(session_test_config());
    manager
        .apply_events(
            &HeaderMap::new(),
            vec![
                NormalizedEvent::AgentStarted(session_event("agent-tool-async", "SessionStart")),
                NormalizedEvent::SubagentStarted(SubagentEvent {
                    session_id: "agent-tool-async".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "SubagentStart".into(),
                    subagent_id: "worker".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::ToolEnded(ToolEvent {
                    session_id: "agent-tool-async".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "PostToolUse".into(),
                    tool_call_id: "agent-tool".into(),
                    tool_name: "Agent".into(),
                    subagent_id: None,
                    arguments: Value::Null,
                    result: json!({
                        "agentId": "worker",
                        "status": "async_launched"
                    }),
                    status: Some("success".into()),
                    payload: json!({}),
                    metadata: json!({}),
                }),
            ],
        )
        .await
        .unwrap();

    let worker_uuid = {
        let sessions = manager.inner.lock().await;
        let session = sessions.get("agent-tool-async").unwrap();
        session.subagents.get("worker").unwrap().uuid
    };

    manager
        .apply_events(
            &HeaderMap::new(),
            vec![NormalizedEvent::ToolStarted(ToolEvent {
                session_id: "agent-tool-async".into(),
                agent_kind: AgentKind::ClaudeCode,
                event_name: "PreToolUse".into(),
                tool_call_id: "worker-tool".into(),
                tool_name: "Read".into(),
                subagent_id: Some("worker".into()),
                arguments: json!({ "file_path": "README.md" }),
                result: Value::Null,
                status: None,
                payload: json!({}),
                metadata: json!({}),
            })],
        )
        .await
        .unwrap();

    let sessions = manager.inner.lock().await;
    let tool = sessions
        .get("agent-tool-async")
        .unwrap()
        .tools
        .get("worker-tool")
        .unwrap();
    assert_eq!(tool.parent_uuid, Some(worker_uuid));
    assert_eq!(
        tool.metadata.as_ref().unwrap()["tool_correlation_status"],
        json!("explicit")
    );
    assert_eq!(
        tool.metadata.as_ref().unwrap()["tool_correlation_subagent_id"],
        json!("worker")
    );
}

#[tokio::test]
async fn active_tool_name_args_fallback_requires_matching_subagent_owner() {
    let manager = SessionManager::new(session_test_config());
    let session_id = "tool-owner-fallback";
    let same_args = json!({ "file_path": "README.md" });

    manager
        .apply_events(
            &HeaderMap::new(),
            vec![
                NormalizedEvent::AgentStarted(session_event(session_id, "SessionStart")),
                NormalizedEvent::SubagentStarted(SubagentEvent {
                    session_id: session_id.into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "SubagentStart".into(),
                    subagent_id: "worker-1".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::SubagentStarted(SubagentEvent {
                    session_id: session_id.into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "SubagentStart".into(),
                    subagent_id: "worker-2".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::ToolStarted(ToolEvent {
                    session_id: session_id.into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "PreToolUse".into(),
                    tool_call_id: "worker-1-pre".into(),
                    tool_name: "Read".into(),
                    subagent_id: Some("worker-1".into()),
                    arguments: same_args.clone(),
                    result: Value::Null,
                    status: None,
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::ToolStarted(ToolEvent {
                    session_id: session_id.into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "PreToolUse".into(),
                    tool_call_id: "worker-2-pre".into(),
                    tool_name: "Read".into(),
                    subagent_id: Some("worker-2".into()),
                    arguments: same_args.clone(),
                    result: Value::Null,
                    status: None,
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::ToolEnded(ToolEvent {
                    session_id: session_id.into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "PostToolUse".into(),
                    tool_call_id: "provider-worker-1".into(),
                    tool_name: "Read".into(),
                    subagent_id: Some("worker-1".into()),
                    arguments: same_args,
                    result: json!({ "ok": true }),
                    status: Some("success".into()),
                    payload: json!({}),
                    metadata: json!({}),
                }),
            ],
        )
        .await
        .unwrap();

    let sessions = manager.inner.lock().await;
    let tools = &sessions.get(session_id).unwrap().tools;
    assert!(!tools.contains_key("worker-1-pre"));
    assert!(tools.contains_key("worker-2-pre"));
    assert!(!tools.contains_key("provider-worker-1"));
}

#[tokio::test]
async fn active_tool_name_args_fallback_uses_unique_global_match_without_owner() {
    let manager = SessionManager::new(session_test_config());
    let session_id = "tool-global-fallback";
    let same_args = json!({ "file_path": "README.md" });

    manager
        .apply_events(
            &HeaderMap::new(),
            vec![
                NormalizedEvent::AgentStarted(session_event(session_id, "SessionStart")),
                NormalizedEvent::SubagentStarted(SubagentEvent {
                    session_id: session_id.into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "SubagentStart".into(),
                    subagent_id: "worker-1".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::ToolStarted(ToolEvent {
                    session_id: session_id.into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "PreToolUse".into(),
                    tool_call_id: "worker-1-pre".into(),
                    tool_name: "Read".into(),
                    subagent_id: Some("worker-1".into()),
                    arguments: same_args.clone(),
                    result: Value::Null,
                    status: None,
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::ToolEnded(ToolEvent {
                    session_id: session_id.into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "PostToolUse".into(),
                    tool_call_id: "provider-worker-1".into(),
                    tool_name: "Read".into(),
                    subagent_id: None,
                    arguments: same_args,
                    result: json!({ "ok": true }),
                    status: Some("success".into()),
                    payload: json!({}),
                    metadata: json!({}),
                }),
            ],
        )
        .await
        .unwrap();

    let sessions = manager.inner.lock().await;
    let tools = &sessions.get(session_id).unwrap().tools;
    assert!(tools.is_empty());
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
async fn gateway_shutdown_closes_codex_sessions_without_session_end_hook() {
    let manager = SessionManager::new(session_test_config());
    let headers = HeaderMap::new();

    manager
        .apply_events(
            &headers,
            vec![
                NormalizedEvent::AgentStarted(SessionEvent {
                    session_id: "codex-no-session-end".into(),
                    agent_kind: AgentKind::Codex,
                    event_name: "SessionStart".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::ToolStarted(ToolEvent {
                    session_id: "codex-no-session-end".into(),
                    agent_kind: AgentKind::Codex,
                    event_name: "PreToolUse".into(),
                    tool_call_id: "tool-1".into(),
                    tool_name: "shell".into(),
                    subagent_id: None,
                    arguments: json!({ "cmd": "pwd" }),
                    result: Value::Null,
                    status: None,
                    payload: json!({}),
                    metadata: json!({}),
                }),
            ],
        )
        .await
        .unwrap();

    manager.close_all("gateway_shutdown").await.unwrap();

    assert!(manager.inner.lock().await.is_empty());
}

#[tokio::test]
async fn idle_timeout_closes_codex_session_without_session_end_hook() {
    let subscriber_name = "cli-idle-timeout-close-reason-test";
    let _ = deregister_subscriber(subscriber_name);
    let close_statuses = Arc::new(StdMutex::new(Vec::<(String, String)>::new()));
    let captured = close_statuses.clone();
    register_subscriber(
        subscriber_name,
        Arc::new(move |event| {
            if event.scope_category() == Some(ScopeCategory::End)
                && event
                    .metadata()
                    .and_then(|metadata| metadata.get("session_id"))
                    .and_then(Value::as_str)
                    == Some("codex-idle")
            {
                let status = event
                    .output()
                    .and_then(|output| output.get("status"))
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                captured
                    .lock()
                    .unwrap()
                    .push((event.name().to_string(), status));
            }
        }),
    )
    .unwrap();

    let manager = SessionManager::new(session_test_config());
    manager
        .apply_events(
            &HeaderMap::new(),
            vec![
                NormalizedEvent::AgentStarted(SessionEvent {
                    session_id: "codex-idle".into(),
                    agent_kind: AgentKind::Codex,
                    event_name: "SessionStart".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::SubagentStarted(SubagentEvent {
                    session_id: "codex-idle".into(),
                    agent_kind: AgentKind::Codex,
                    event_name: "SubagentStart".into(),
                    subagent_id: "worker".into(),
                    payload: json!({}),
                    metadata: json!({ "session_id": "codex-idle" }),
                }),
            ],
        )
        .await
        .unwrap();

    let closed = manager
        .close_idle_sessions_at(
            Instant::now() + AGENT_IDLE_TIMEOUT + Duration::from_secs(1),
            AGENT_IDLE_TIMEOUT,
            "idle_timeout",
        )
        .await
        .unwrap();

    assert_eq!(closed, 1);
    {
        let sessions = manager.inner.lock().await;
        let session = sessions.get("codex-idle").unwrap();
        assert!(session.turn_scope.is_none());
        assert!(session.subagents.is_empty());
    }

    flush_subscribers().unwrap();
    let statuses = close_statuses.lock().unwrap().clone();
    assert!(
        statuses.contains(&("subagent:worker".to_string(), "idle_timeout".to_string())),
        "expected idle timeout to close the child scope, got {statuses:?}"
    );
    assert!(
        statuses.contains(&("codex-turn".to_string(), "idle_timeout".to_string())),
        "expected idle timeout to close the turn scope, got {statuses:?}"
    );

    deregister_subscriber(subscriber_name).unwrap();
}

#[tokio::test]
async fn idle_timeout_keeps_recent_claude_subagent_session_open() {
    let manager = SessionManager::new(session_test_config());
    manager
        .apply_events(
            &HeaderMap::new(),
            vec![
                NormalizedEvent::AgentStarted(session_event("claude-recent", "SessionStart")),
                NormalizedEvent::SubagentStarted(SubagentEvent {
                    session_id: "claude-recent".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "SubagentStart".into(),
                    subagent_id: "recent-worker".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
            ],
        )
        .await
        .unwrap();

    let closed = manager
        .close_idle_sessions_at(
            Instant::now() + Duration::from_secs(5),
            AGENT_IDLE_TIMEOUT,
            "idle_timeout",
        )
        .await
        .unwrap();

    assert_eq!(closed, 0);
    let sessions = manager.inner.lock().await;
    let session = sessions.get("claude-recent").unwrap();
    assert!(session.turn_scope.is_some());
    assert!(session.subagents.contains_key("recent-worker"));
}

#[tokio::test]
async fn idle_timeout_closes_claude_subagent_with_no_followup_activity() {
    let subscriber_name = "cli-claude-idle-subagent-close-reason-test";
    let _ = deregister_subscriber(subscriber_name);
    let close_statuses = Arc::new(StdMutex::new(Vec::<(String, String)>::new()));
    let captured = close_statuses.clone();
    register_subscriber(
        subscriber_name,
        Arc::new(move |event| {
            if event.scope_category() == Some(ScopeCategory::End)
                && (event.name() == "subagent:idle-worker"
                    || event
                        .metadata()
                        .and_then(|metadata| metadata.get("session_id"))
                        .and_then(Value::as_str)
                        == Some("claude-idle"))
            {
                let status = event
                    .output()
                    .and_then(|output| output.get("status"))
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                captured
                    .lock()
                    .unwrap()
                    .push((event.name().to_string(), status));
            }
        }),
    )
    .unwrap();

    let manager = SessionManager::new(session_test_config());
    manager
        .apply_events(
            &HeaderMap::new(),
            vec![
                NormalizedEvent::AgentStarted(session_event("claude-idle", "SessionStart")),
                NormalizedEvent::SubagentStarted(SubagentEvent {
                    session_id: "claude-idle".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "SubagentStart".into(),
                    subagent_id: "idle-worker".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
            ],
        )
        .await
        .unwrap();

    let closed = manager
        .close_idle_sessions_at(
            Instant::now() + AGENT_IDLE_TIMEOUT + Duration::from_secs(1),
            AGENT_IDLE_TIMEOUT,
            "idle_timeout",
        )
        .await
        .unwrap();

    assert_eq!(closed, 1);
    {
        let sessions = manager.inner.lock().await;
        let session = sessions.get("claude-idle").unwrap();
        assert!(session.turn_scope.is_none());
        assert!(session.subagents.is_empty());
    }

    flush_subscribers().unwrap();
    let statuses = close_statuses.lock().unwrap().clone();
    assert!(
        statuses.contains(&(
            "subagent:idle-worker".to_string(),
            "idle_timeout".to_string()
        )),
        "expected idle timeout to close the Claude subagent scope, got {statuses:?}"
    );
    assert!(
        statuses.contains(&("claude-code-turn".to_string(), "idle_timeout".to_string())),
        "expected idle timeout to close the Claude turn scope, got {statuses:?}"
    );

    deregister_subscriber(subscriber_name).unwrap();
}

#[tokio::test]
async fn idle_timeout_waits_for_active_claude_subagent_tool_call() {
    let manager = SessionManager::new(session_test_config());
    manager
        .apply_events(
            &HeaderMap::new(),
            vec![
                NormalizedEvent::AgentStarted(session_event("claude-active-tool", "SessionStart")),
                NormalizedEvent::SubagentStarted(SubagentEvent {
                    session_id: "claude-active-tool".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "SubagentStart".into(),
                    subagent_id: "active-tool-worker".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::ToolStarted(ToolEvent {
                    session_id: "claude-active-tool".into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "PreToolUse".into(),
                    tool_call_id: "tool-1".into(),
                    tool_name: "Read".into(),
                    subagent_id: Some("active-tool-worker".into()),
                    arguments: json!({ "file_path": "README.md" }),
                    result: Value::Null,
                    status: None,
                    payload: json!({}),
                    metadata: json!({}),
                }),
            ],
        )
        .await
        .unwrap();

    let closed = manager
        .close_idle_sessions_at(
            Instant::now() + AGENT_IDLE_TIMEOUT + Duration::from_secs(1),
            AGENT_IDLE_TIMEOUT,
            "idle_timeout",
        )
        .await
        .unwrap();

    assert_eq!(closed, 0);
    let sessions = manager.inner.lock().await;
    let session = sessions.get("claude-active-tool").unwrap();
    assert!(session.turn_scope.is_some());
    assert!(session.subagents.contains_key("active-tool-worker"));
    assert_eq!(session.tools.len(), 1);
}

#[tokio::test]
async fn idle_timeout_waits_for_active_gateway_llm_call() {
    let manager = SessionManager::new(session_test_config());
    let prep = manager
        .prepare_gateway_call(
            &HeaderMap::new(),
            LlmGatewayStart {
                session_id: Some("active-gateway-call".into()),
                ..llm_start()
            },
        )
        .await
        .unwrap();

    let closed = manager
        .close_idle_sessions_at(
            Instant::now() + AGENT_IDLE_TIMEOUT + Duration::from_secs(1),
            AGENT_IDLE_TIMEOUT,
            "idle_timeout",
        )
        .await
        .unwrap();
    assert_eq!(closed, 0);
    assert!(
        manager
            .inner
            .lock()
            .await
            .contains_key("active-gateway-call")
    );

    manager.finish_gateway_call(&prep.session_id, false).await;
    let closed = manager
        .close_idle_sessions_at(
            Instant::now() + AGENT_IDLE_TIMEOUT + Duration::from_secs(1),
            AGENT_IDLE_TIMEOUT,
            "idle_timeout",
        )
        .await
        .unwrap();

    assert_eq!(closed, 1);
    assert!(manager.inner.lock().await.is_empty());
}

#[tokio::test]
async fn gateway_shutdown_attempts_remaining_sessions_after_close_error() {
    let subscriber_name = "cli-close-all-deferred-error-test";
    let _ = deregister_subscriber(subscriber_name);

    let closed_sessions = Arc::new(StdMutex::new(Vec::<String>::new()));
    let captured = closed_sessions.clone();
    register_subscriber(
        subscriber_name,
        Arc::new(move |event| {
            if event.scope_category() == Some(ScopeCategory::End)
                && let Some(session_id) = event
                    .metadata()
                    .and_then(|metadata| metadata.get("session_id"))
                    .and_then(Value::as_str)
            {
                captured.lock().unwrap().push(session_id.to_string());
            }
        }),
    )
    .unwrap();

    let config = SessionConfig::default();
    let mut bad = Session::new("bad-shutdown".into(), AgentKind::ClaudeCode, config.clone());
    bad.agent_scope = Some(
        ScopeHandle::builder()
            .name("missing-agent-scope")
            .scope_type(ScopeType::Agent)
            .build(),
    );

    let mut good = Session::new("good-shutdown".into(), AgentKind::ClaudeCode, config);
    let stack = good.scope_stack.clone();
    TASK_SCOPE_STACK
        .scope(stack, async {
            good.open_turn(json!({}), json!({ "prompt": "close me" }), "test")
                .unwrap();
        })
        .await;

    let mut sessions = vec![bad, good];
    let error = close_sessions_for_shutdown(&mut sessions, "gateway_shutdown")
        .await
        .unwrap_err();
    assert!(error.to_string().contains("scope handle not found"));

    flush_subscribers().unwrap();
    let closed = closed_sessions.lock().unwrap().clone();
    assert!(
        closed.contains(&"good-shutdown".to_string()),
        "expected later valid session to close after first error, got {closed:?}"
    );

    deregister_subscriber(subscriber_name).unwrap();
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

#[tokio::test]
async fn single_tool_hint_does_not_claim_same_name_with_different_call_and_args() {
    for (agent_kind, label, tool_name, expected_args, actual_args) in [
        (
            AgentKind::ClaudeCode,
            "claude",
            "Read",
            json!({ "file_path": "README.md" }),
            json!({ "file_path": "Cargo.toml" }),
        ),
        (
            AgentKind::Codex,
            "codex",
            "exec_command",
            json!({ "cmd": "pwd" }),
            json!({ "cmd": "ls" }),
        ),
        (
            AgentKind::Hermes,
            "hermes",
            "shell",
            json!({ "command": "pwd" }),
            json!({ "command": "ls" }),
        ),
    ] {
        let manager = SessionManager::new(session_test_config());
        let session_id = format!("weak-tool-hint-{label}");
        manager
            .apply_events(
                &HeaderMap::new(),
                vec![
                    NormalizedEvent::AgentStarted(SessionEvent {
                        session_id: session_id.clone(),
                        agent_kind,
                        event_name: "SessionStart".into(),
                        payload: json!({}),
                        metadata: json!({}),
                    }),
                    NormalizedEvent::SubagentStarted(SubagentEvent {
                        session_id: session_id.clone(),
                        agent_kind,
                        event_name: "SubagentStart".into(),
                        subagent_id: "worker".into(),
                        payload: json!({}),
                        metadata: json!({}),
                    }),
                ],
            )
            .await
            .unwrap();

        let turn_uuid = {
            let sessions = manager.inner.lock().await;
            active_turn_uuid(sessions.get(&session_id).unwrap())
        };
        let active = manager
            .start_llm(
                &HeaderMap::new(),
                LlmGatewayStart {
                    session_id: Some(session_id.clone()),
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
                            "call_id": "expected-call",
                            "name": tool_name,
                            "arguments": serde_json::to_string(&expected_args).unwrap()
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
                    session_id: session_id.clone(),
                    agent_kind,
                    event_name: "PreToolUse".into(),
                    tool_call_id: "actual-call".into(),
                    tool_name: tool_name.into(),
                    subagent_id: None,
                    arguments: actual_args,
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
            .get(&session_id)
            .unwrap()
            .tools
            .get("actual-call")
            .unwrap();
        assert_eq!(handle.parent_uuid, Some(turn_uuid), "case {label}");
        assert_eq!(
            handle.metadata.as_ref().unwrap()["tool_correlation_status"],
            json!("ambiguous_fallback"),
            "case {label}"
        );
        assert!(
            handle.metadata.as_ref().unwrap()["tool_correlation_subagent_id"].is_null(),
            "case {label}"
        );
    }
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
fn provider_tool_hints_require_call_id_or_name_with_arguments() {
    let mut hints = Vec::new();

    collect_openai_response_tool_hints(
        &json!({
            "output": [
                {
                    "type": "function_call",
                    "name": "Read"
                },
                {
                    "type": "function_call",
                    "name": "Read",
                    "arguments": "{\"file_path\":\"README.md\"}"
                }
            ]
        }),
        Some("worker"),
        &mut hints,
    );

    assert_eq!(hints.len(), 1);
    assert_eq!(hints[0].tool_call_id.as_deref(), None);
    assert_eq!(hints[0].tool_name.as_deref(), Some("Read"));
    assert_eq!(hints[0].arguments, json!({ "file_path": "README.md" }));
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

    let turn_uuid = {
        let sessions = manager.inner.lock().await;
        active_turn_uuid(sessions.get("missing-hint-owner").unwrap())
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

    assert_eq!(active.handle.parent_uuid, Some(turn_uuid));
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
        metadata: None,
        plugin_config: None,
    }
}

#[tokio::test]
async fn turn_ended_is_noop_without_active_turn_scope() {
    let temp = tempfile::tempdir().unwrap();
    let config = GatewayConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://127.0.0.1".into(),

        anthropic_base_url: "http://127.0.0.1".into(),
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

fn codex_session_event(session_id: &str, event_name: &str, metadata: Value) -> SessionEvent {
    SessionEvent {
        session_id: session_id.into(),
        agent_kind: AgentKind::Codex,
        event_name: event_name.into(),
        payload: json!({ "event": event_name }),
        metadata,
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

fn claude_startup_probe_start(session_id: &str) -> LlmGatewayStart {
    LlmGatewayStart {
        session_id: Some(session_id.into()),
        provider: "anthropic.messages".into(),
        model_name: Some("claude-opus-4-8[1m]".into()),
        request: LlmRequest {
            headers: Map::from_iter([("x-claude-code-session-id".to_string(), json!(session_id))]),
            content: json!({
                "model": "claude-opus-4-8[1m]",
                "max_tokens": 1,
                "messages": [
                    {
                        "role": "user",
                        "content": "test"
                    }
                ]
            }),
        },
        ..llm_start()
    }
}

fn llm_start_with_messages_task(session_id: &str, task: &str) -> LlmGatewayStart {
    llm_start_with_content(
        session_id,
        "anthropic.messages",
        "claude-test",
        json!({
            "model": "claude-test",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "<system-reminder>\nToday is 2026-05-19.\n</system-reminder>"
                        },
                        {
                            "type": "text",
                            "text": task
                        }
                    ]
                }
            ]
        }),
    )
}

fn llm_start_with_responses_task(session_id: &str, task: &str) -> LlmGatewayStart {
    llm_start_with_content(
        session_id,
        "openai.responses",
        "gpt-test",
        json!({
            "model": "gpt-test",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": task
                        }
                    ]
                }
            ]
        }),
    )
}

fn llm_start_with_chat_completion_task(session_id: &str, task: &str) -> LlmGatewayStart {
    llm_start_with_content(
        session_id,
        "openai.chat_completions",
        "gpt-test",
        json!({
            "model": "gpt-test",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a coding agent."
                },
                {
                    "role": "user",
                    "content": task
                }
            ]
        }),
    )
}

fn llm_start_with_content(
    session_id: &str,
    provider: &str,
    model_name: &str,
    content: Value,
) -> LlmGatewayStart {
    LlmGatewayStart {
        session_id: Some(session_id.into()),
        provider: provider.into(),
        model_name: Some(model_name.into()),
        subagent_id: None,
        conversation_id: None,
        generation_id: None,
        request_id: None,
        request: LlmRequest {
            headers: Map::new(),
            content,
        },
        streaming: false,
        metadata: json!({}),
    }
}
