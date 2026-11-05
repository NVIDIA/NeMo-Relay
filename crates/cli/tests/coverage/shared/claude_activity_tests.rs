// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn activity_request(session_id: &str, request_class: &str) -> (HeaderMap, LlmGatewayStart) {
    let mut headers = HeaderMap::new();
    headers.insert(
        "x-claude-code-request-class",
        request_class.parse().unwrap(),
    );
    let mut start = llm_start_with_content(
        session_id,
        "anthropic.messages",
        "claude-test",
        json!({
            "model": "claude-test",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "inspect the activity fixture"}]
        }),
    );
    start.metadata = json!({"claude_code_request_class": "main"});
    (headers, start)
}

#[tokio::test]
async fn claude_internal_request_class_overrides_caller_metadata() {
    for request_class in ["auxiliary", "compaction"] {
        let mut config = session_test_config();
        config.metadata = Some(json!({"claude_code_request_class": "workflow"}));
        let manager = SessionManager::new(config);
        let id = format!("activity-owned-{request_class}");
        let (headers, start) = activity_request(&id, request_class);
        let prep = manager.prepare_gateway_call(&headers, start).await.unwrap();
        let observed_class = prep.metadata["claude_code_request_class"].clone();
        assert!(!prep.record_task_response);
        manager
            .finish_gateway_call(
                &prep.session_id,
                prep.owner_subagent_id.as_deref(),
                prep.session_finish,
            )
            .await;
        manager.close_all("test_shutdown").await.unwrap();

        assert_eq!(
            observed_class, request_class,
            "the original request header must own per-call classification"
        );
    }
}

#[tokio::test]
async fn claude_internal_requests_do_not_open_task_turns() {
    for request_class in ["auxiliary", "compaction"] {
        let manager = SessionManager::new(session_test_config());
        let id = format!("activity-idle-{request_class}");
        let (headers, start) = activity_request(&id, request_class);
        let prep = manager.prepare_gateway_call(&headers, start).await.unwrap();
        let opened_turn = manager.inner.lock().await[&id].turn_scope.is_some();
        assert!(!prep.record_task_response);
        manager
            .finish_gateway_call(
                &prep.session_id,
                prep.owner_subagent_id.as_deref(),
                prep.session_finish,
            )
            .await;
        manager.close_all("test_shutdown").await.unwrap();

        assert!(
            !opened_turn,
            "an idle {request_class} request must not fabricate a task turn"
        );
        assert!(prep.parent.is_none());
        assert!(!manager.inner.lock().await.contains_key(&id));
    }
}

async fn finish_activity(manager: &SessionManager, prep: &GatewayCallPrep) {
    manager
        .finish_gateway_call(
            &prep.session_id,
            prep.owner_subagent_id.as_deref(),
            prep.session_finish,
        )
        .await;
}

#[tokio::test]
async fn claude_internal_requests_preserve_parent_and_pending_conversation_evidence() {
    let manager = SessionManager::new(session_test_config());
    let id = "activity-existing-ownership";
    manager
        .apply_events(
            &HeaderMap::new(),
            vec![
                NormalizedEvent::PromptSubmitted(session_event(id, "UserPromptSubmit")),
                NormalizedEvent::SubagentStarted(SubagentEvent {
                    session_id: id.into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "SubagentStart".into(),
                    subagent_id: "worker".into(),
                    payload: json!({}),
                    metadata: json!({}),
                }),
                NormalizedEvent::LlmHint(LlmHintEvent {
                    session_id: id.into(),
                    agent_kind: AgentKind::ClaudeCode,
                    event_name: "UserPromptSubmit".into(),
                    subagent_id: Some("worker".into()),
                    agent_id: None,
                    agent_type: Some("Explore".into()),
                    conversation_id: None,
                    generation_id: None,
                    request_id: None,
                    model: None,
                    payload: json!({"prompt": "inspect the activity fixture"}),
                    metadata: json!({}),
                }),
            ],
        )
        .await
        .unwrap();
    let (turn, child, turn_index, hints) = {
        let sessions = manager.inner.lock().await;
        let session = &sessions[id];
        (
            session.turn_scope.as_ref().unwrap().uuid,
            session.subagents["worker"].uuid,
            session.turn_index,
            session.pending_llm_hints.len(),
        )
    };
    assert_eq!(hints, 1);
    for class in ["auxiliary", "compaction"] {
        for explicit_child in [false, true] {
            let (headers, mut start) = activity_request(id, class);
            if explicit_child {
                start.subagent_id = Some("worker".into());
            }
            let prep = manager.prepare_gateway_call(&headers, start).await.unwrap();
            assert_eq!(
                prep.parent.as_ref().unwrap().uuid,
                if explicit_child { child } else { turn }
            );
            assert_eq!(
                prep.owner_subagent_id.as_deref(),
                explicit_child.then_some("worker")
            );
            assert!(!prep.record_task_response);
            finish_activity(&manager, &prep).await;
            let sessions = manager.inner.lock().await;
            let session = &sessions[id];
            assert_eq!(session.turn_index, turn_index);
            assert_eq!(session.pending_llm_hints.len(), hints);
            assert!(session.last_llm_owner.is_none());
            assert!(session.llm_request_affinity.is_empty());
            assert!(session.pending_tool_hints.is_empty());
            assert!(session.last_turn_llm_output.is_none());
        }
    }
    manager.close_all("test_shutdown").await.unwrap();
}

#[tokio::test]
async fn claude_internal_session_pruning_preserves_concurrent_and_announced_work() {
    for class in ["auxiliary", "compaction"] {
        let manager = SessionManager::new(session_test_config());
        let id = format!("activity-pruning-{class}");
        let (headers, start) = activity_request(&id, class);
        let first = manager
            .prepare_gateway_call(&headers, start.clone())
            .await
            .unwrap();
        let second = manager.prepare_gateway_call(&headers, start).await.unwrap();
        finish_activity(&manager, &first).await;
        assert!(manager.inner.lock().await.contains_key(&id));
        finish_activity(&manager, &second).await;
        assert!(!manager.inner.lock().await.contains_key(&id));

        let (headers, start) = activity_request(&id, class);
        let internal = manager.prepare_gateway_call(&headers, start).await.unwrap();
        let (headers, start) = activity_request(&id, "main");
        let main = manager.prepare_gateway_call(&headers, start).await.unwrap();
        finish_activity(&manager, &internal).await;
        assert!(manager.inner.lock().await[&id].turn_scope.is_some());
        finish_activity(&manager, &main).await;
        manager.close_all("test_shutdown").await.unwrap();

        let manager = SessionManager::new(session_test_config());
        manager
            .apply_events(
                &HeaderMap::new(),
                vec![NormalizedEvent::AgentStarted(session_event(
                    &id,
                    "SessionStart",
                ))],
            )
            .await
            .unwrap();
        let (headers, start) = activity_request(&id, class);
        let prep = manager.prepare_gateway_call(&headers, start).await.unwrap();
        finish_activity(&manager, &prep).await;
        assert!(manager.inner.lock().await.contains_key(&id));
        assert!(manager.inner.lock().await[&id].turn_scope.is_none());
        manager.close_all("test_shutdown").await.unwrap();
    }
}

#[tokio::test]
async fn claude_unknown_and_conversation_classes_keep_existing_gateway_lifecycle() {
    for class in [
        None,
        Some("future-class"),
        Some("main"),
        Some("subagent"),
        Some("workflow"),
    ] {
        let manager = SessionManager::new(session_test_config());
        let (mut headers, mut start) =
            activity_request("activity-fallback", class.unwrap_or("main"));
        if class.is_none() {
            headers.clear();
        }
        start.metadata = json!({"claude_code_request_class": "auxiliary"});
        let prep = manager.prepare_gateway_call(&headers, start).await.unwrap();
        assert!(prep.record_task_response);
        assert!(prep.parent.is_some());
        assert_eq!(
            prep.metadata["claude_code_request_class"],
            class
                .filter(|value| *value != "future-class")
                .unwrap_or("unknown")
        );
        finish_activity(&manager, &prep).await;
        manager.close_all("test_shutdown").await.unwrap();
    }
    let manager = SessionManager::new(session_test_config());
    let (headers, _) = activity_request("activity-other-harness", "auxiliary");
    let prep = manager
        .prepare_gateway_call(&headers, llm_start())
        .await
        .unwrap();
    assert!(prep.record_task_response);
    assert!(prep.parent.is_some());
    finish_activity(&manager, &prep).await;
    manager.close_all("test_shutdown").await.unwrap();
}

#[tokio::test]
async fn claude_prompt_counts_exclude_internal_calls_and_idle_notifications() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let id = "activity-five-prompts";
    let events = Arc::new(StdMutex::new(Vec::<Event>::new()));
    let recorded = events.clone();
    register_filtered_session_subscriber(
        id,
        tracked_sessions(&[id]),
        Arc::new(move |event| recorded.lock().unwrap().push(event.clone())),
    );
    let mut config = session_test_config();
    config.metadata = Some(json!({"turn_prompt_observed": true}));
    let manager = SessionManager::new(config);
    manager
        .apply_events(
            &HeaderMap::new(),
            vec![NormalizedEvent::AgentStarted(session_event(
                id,
                "SessionStart",
            ))],
        )
        .await
        .unwrap();
    for index in 0..5 {
        manager
            .apply_events(
                &HeaderMap::new(),
                vec![NormalizedEvent::Notification(session_event(
                    id,
                    "Notification",
                ))],
            )
            .await
            .unwrap();
        let (headers, start) = activity_request(id, "auxiliary");
        let prep = manager.prepare_gateway_call(&headers, start).await.unwrap();
        finish_activity(&manager, &prep).await;
        assert!(manager.inner.lock().await[id].turn_scope.is_none());
        manager
            .apply_events(
                &HeaderMap::new(),
                vec![NormalizedEvent::PromptSubmitted(session_event(
                    id,
                    "UserPromptSubmit",
                ))],
            )
            .await
            .unwrap();
        let (headers, start) = activity_request(id, "compaction");
        let prep = manager.prepare_gateway_call(&headers, start).await.unwrap();
        finish_activity(&manager, &prep).await;
        if index < 4 {
            let mut event = session_event(id, "Stop");
            event.metadata = json!({"turn_prompt_observed": false});
            manager
                .apply_events(&HeaderMap::new(), vec![NormalizedEvent::TurnEnded(event)])
                .await
                .unwrap();
        }
    }
    manager.close_all("test_shutdown").await.unwrap();
    flush_subscribers().unwrap();
    deregister_subscriber(id).unwrap();
    let events = events.lock().unwrap();
    let starts = events
        .iter()
        .filter(|event| {
            event.name() == "claude-code-turn"
                && event.scope_category() == Some(ScopeCategory::Start)
        })
        .count();
    let ends = events
        .iter()
        .filter(|event| {
            event.name() == "claude-code-turn" && event.scope_category() == Some(ScopeCategory::End)
        })
        .collect::<Vec<_>>();
    assert_eq!(starts, 5);
    assert_eq!(ends.len(), 5);
    assert!(
        ends.iter()
            .all(|event| event.metadata().unwrap()["turn_prompt_observed"] == true)
    );
    assert_eq!(
        ends.iter()
            .map(|event| event.uuid())
            .collect::<HashSet<_>>()
            .len(),
        5
    );
}

#[tokio::test]
async fn claude_prompt_evidence_ignores_delegated_prompts_and_resets_between_turns() {
    let _guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let id = "activity-delegated-prompt-evidence";
    let events = Arc::new(StdMutex::new(Vec::<Event>::new()));
    let recorded = events.clone();
    register_filtered_session_subscriber(
        id,
        tracked_sessions(&[id]),
        Arc::new(move |event| recorded.lock().unwrap().push(event.clone())),
    );
    let mut config = session_test_config();
    config.metadata = Some(json!({"turn_prompt_observed": true}));
    let manager = SessionManager::new(config);
    let mut expected = Vec::new();
    for (index, observed) in [false, true, false].into_iter().enumerate() {
        if observed {
            let mut prompt = session_event(id, "UserPromptSubmit");
            prompt.metadata = json!({"turn_prompt_observed": false});
            manager
                .apply_events(
                    &HeaderMap::new(),
                    vec![NormalizedEvent::PromptSubmitted(prompt)],
                )
                .await
                .unwrap();
        } else {
            let (_, start) = activity_request(id, "main");
            let prep = manager
                .prepare_gateway_call(&HeaderMap::new(), start)
                .await
                .unwrap();
            finish_activity(&manager, &prep).await;
        }
        let turn = manager.inner.lock().await[id]
            .turn_scope
            .as_ref()
            .unwrap()
            .uuid;
        expected.push((turn, observed));
        if index == 0 {
            let mut delegated = session_event(id, "UserPromptSubmit");
            delegated.metadata =
                json!({"session_id": id, "subagent_id": "worker", "turn_prompt_observed": true});
            manager
                .apply_events(
                    &HeaderMap::new(),
                    vec![
                        NormalizedEvent::SubagentStarted(SubagentEvent {
                            session_id: id.into(),
                            agent_kind: AgentKind::ClaudeCode,
                            event_name: "SubagentStart".into(),
                            subagent_id: "worker".into(),
                            payload: json!({}),
                            metadata: json!({"session_id": id, "turn_prompt_observed": true}),
                        }),
                        NormalizedEvent::PromptSubmitted(delegated),
                    ],
                )
                .await
                .unwrap();
            let sessions = manager.inner.lock().await;
            assert_eq!(sessions[id].turn_scope.as_ref().unwrap().uuid, turn);
            assert!(!sessions[id].turn_prompt_observed);
        }
        let mut stop = session_event(id, "Stop");
        stop.metadata = json!({"turn_prompt_observed": !observed});
        manager
            .apply_events(&HeaderMap::new(), vec![NormalizedEvent::TurnEnded(stop)])
            .await
            .unwrap();
    }
    manager.close_all("test_shutdown").await.unwrap();
    flush_subscribers().unwrap();
    deregister_subscriber(id).unwrap();
    let events = events.lock().unwrap();
    for (turn, observed) in expected {
        let end = events
            .iter()
            .find(|event| {
                event.uuid() == turn && event.scope_category() == Some(ScopeCategory::End)
            })
            .unwrap();
        assert_eq!(end.metadata().unwrap()["turn_prompt_observed"], observed);
    }
    let children = events
        .iter()
        .filter(|event| event.name() == "subagent:worker" || event.name() == "codex-turn")
        .collect::<Vec<_>>();
    assert_eq!(children.len(), 4);
    assert!(children.iter().all(|event| {
        event
            .metadata()
            .unwrap()
            .get("turn_prompt_observed")
            .is_none()
    }));
}
