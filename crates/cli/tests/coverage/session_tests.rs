// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::http::HeaderMap;
use nemo_flow::api::event::ScopeCategory;
use nemo_flow::api::subscriber::{deregister_subscriber, register_subscriber};
use nemo_flow::plugin::{PluginConfig, clear_plugin_configuration, initialize_plugins};
use serde_json::json;
use std::path::Path;
use std::sync::{Arc, Mutex as StdMutex};

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

fn read_atif_for_session(output_directory: &Path, session_id: &str) -> Value {
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
async fn parallel_subagents_are_siblings_under_agent_scope() {
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
    let agent_uuid = session.agent_scope.as_ref().unwrap().uuid;
    assert_eq!(
        session.subagents.get("worker-1").unwrap().parent_uuid,
        Some(agent_uuid)
    );
    assert_eq!(
        session.subagents.get("worker-2").unwrap().parent_uuid,
        Some(agent_uuid)
    );
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
    let agent_uuid = parent.agent_scope.as_ref().unwrap().uuid;
    assert_eq!(
        parent.subagents.get("child-thread").unwrap().parent_uuid,
        Some(agent_uuid)
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

    clear_plugin_configuration().unwrap();
    let atif = read_atif_for_session(&atif_dir, "atif-session");
    assert!(
        atif["extra"]["observed_events"]
            .as_array()
            .is_some_and(|events| events.len() >= 3)
    );
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

    clear_plugin_configuration().unwrap();
    let second = read_atif_for_session(&atif_dir, "dup-end");
    let second_steps = second["steps"].as_array().unwrap().len();
    assert_eq!(
        first_steps, second_steps,
        "duplicate AgentEnded must not change the ATIF step count"
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
    assert_eq!(atif["steps"][1]["metrics"]["prompt_tokens"], json!(10));
    assert_eq!(atif["steps"][1]["metrics"]["completion_tokens"], json!(5));
    assert_eq!(atif["steps"][1]["metrics"]["cached_tokens"], json!(3));
    assert_eq!(atif["final_metrics"]["total_prompt_tokens"], json!(10));
    assert_eq!(atif["final_metrics"]["total_completion_tokens"], json!(5));
    assert_eq!(atif["final_metrics"]["total_cached_tokens"], json!(3));
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
    let steps = atif["steps"].as_array().unwrap();
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["source"], json!("system"));
    assert_eq!(
        steps[0]["message"]["hook_event_name"],
        json!("subagent_stop")
    );
    assert_eq!(
        steps[0]["extra"]["ancestry"]["function_name"],
        json!("subagent_end_without_start")
    );
    assert_eq!(
        steps[0]["extra"]["ancestry"]["parent_name"],
        json!("hermes")
    );
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

    let agent_uuid = {
        let sessions = manager.inner.lock().await;
        let session = sessions.get("agent-tool-finish").unwrap();
        assert!(session.subagents.is_empty());
        assert!(session.subagent_stacks.is_empty());
        session.agent_scope.as_ref().unwrap().uuid
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

    assert_eq!(final_llm.handle.parent_uuid, Some(agent_uuid));
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
    let mut bad = Session::new("bad-shutdown".into(), AgentKind::Codex, config.clone());
    bad.agent_scope = Some(
        ScopeHandle::builder()
            .name("missing-agent-scope")
            .scope_type(ScopeType::Agent)
            .build(),
    );

    let mut good = Session::new("good-shutdown".into(), AgentKind::Codex, config);
    let stack = good.scope_stack.clone();
    TASK_SCOPE_STACK
        .scope(stack, async {
            good.ensure_agent_started(json!({})).unwrap();
        })
        .await;

    let mut sessions = vec![bad, good];
    let error = close_sessions_for_shutdown(&mut sessions, "gateway_shutdown")
        .await
        .unwrap_err();
    assert!(error.to_string().contains("scope handle not found"));

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
        metadata: None,
        plugin_config: None,
    }
}

#[tokio::test]
async fn turn_ended_is_noop_for_session_with_no_agent_scope() {
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
