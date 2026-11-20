// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

async fn wait_for_idle_alias_cleanup<F: std::future::Future>(
    sweep: &mut std::pin::Pin<Box<F>>,
    manager: &SessionManager,
) {
    tokio::time::timeout(
        Duration::from_secs(2),
        std::future::poll_fn(|context| {
            assert!(std::future::Future::poll(sweep.as_mut(), context).is_pending());
            if let Ok(sessions) = manager.inner.try_lock()
                && sessions
                    .get("parent-thread")
                    .is_some_and(|session| session.subagents.is_empty())
            {
                return std::task::Poll::Ready(());
            }
            context.waker().wake_by_ref();
            std::task::Poll::Pending
        }),
    )
    .await
    .expect("sweep should restore the session before waiting for alias cleanup");
}

async fn start_child_alias(manager: &SessionManager, identifier: &str) {
    manager
        .apply_authenticated_events(
            &HeaderMap::new(),
            vec![NormalizedEvent::AgentStarted(SessionEvent {
                session_id: identifier.into(),
                agent_kind: AgentKind::Codex,
                event_name: "SessionStart".into(),
                payload: json!({"source": {"subagent": {"thread_spawn": {
                    "parent_thread_id": "parent-thread"
                }}}}),
                metadata: json!({}),
            })],
            "client-a",
        )
        .await
        .unwrap();
}

async fn apply_parent_prompt(manager: &SessionManager) {
    manager
        .apply_authenticated_events(
            &HeaderMap::new(),
            vec![NormalizedEvent::PromptSubmitted(codex_session_event(
                "parent-thread",
                "UserPromptSubmit",
                json!({}),
            ))],
            "client-a",
        )
        .await
        .unwrap();
}

async fn retirement_fixture(child_turn: bool) -> SessionManager {
    let manager = SessionManager::new(session_test_config());
    start_authenticated_codex_alias(&manager).await;
    if child_turn {
        manager
            .apply_authenticated_events(
                &HeaderMap::new(),
                vec![NormalizedEvent::PromptSubmitted(codex_session_event(
                    "child-thread",
                    "UserPromptSubmit",
                    json!({}),
                ))],
                "client-a",
            )
            .await
            .unwrap();
    }
    manager
        .apply_authenticated_events(
            &HeaderMap::new(),
            vec![NormalizedEvent::TurnEnded(codex_session_event(
                "parent-thread",
                "Stop",
                json!({}),
            ))],
            "client-a",
        )
        .await
        .unwrap();
    manager
}

async fn block_child(
    manager: &SessionManager,
    identifier: &str,
) -> (ScopeStackHandle, ScopeHandle) {
    let stack = manager.inner.lock().await["parent-thread"].subagents[identifier]
        .scope_stack
        .clone();
    let blocker = TASK_SCOPE_STACK
        .scope(stack.clone(), async {
            push_scope(
                PushScopeParams::builder()
                    .name("retirement-blocker")
                    .scope_type(ScopeType::Custom)
                    .build(),
            )
            .unwrap()
        })
        .await;
    (stack, blocker)
}

async fn unblock_child(stack: ScopeStackHandle, blocker: ScopeHandle) {
    TASK_SCOPE_STACK
        .scope(stack, async {
            nemo_relay_runtime::api::scope::pop_scope(
                PopScopeParams::builder()
                    .handle_uuid(&blocker.uuid)
                    .output(json!({}))
                    .build(),
            )
            .unwrap();
        })
        .await;
}

#[tokio::test]
async fn pending_retirement_preserves_child_turn_and_allows_root_prompts() {
    for child_turn in [false, true] {
        let captured = Arc::new(StdMutex::new(Vec::<Event>::new()));
        let capture = Arc::clone(&captured);
        let subscriber_name = "retirement-child-turn";
        register_subscriber(
            subscriber_name,
            Arc::new(move |event| capture.lock().unwrap().push(event.clone())),
        )
        .unwrap();
        let manager = retirement_fixture(child_turn).await;
        let (worker_identifier, turn_identifier, task_identifier) = {
            let sessions = manager.inner.lock().await;
            let parent = &sessions["parent-thread"];
            let child = &parent.subagents["child-thread"];
            (
                child.scope_handle.uuid,
                child.child_turn.as_ref().map(|turn| turn.uuid),
                parent.task_scope.as_ref().unwrap().uuid,
            )
        };
        let (stack, blocker) = block_child(&manager, "child-thread").await;
        let sweep_at = Instant::now() + AGENT_IDLE_TIMEOUT + Duration::from_secs(1);
        assert!(
            manager
                .close_idle_sessions_at(sweep_at, AGENT_IDLE_TIMEOUT, "idle_timeout")
                .await
                .unwrap_err()
                .to_string()
                .contains("not at the top")
        );
        // Both an initial prompt and superseding an already-open root turn must remain usable.
        for _ in 0..2 {
            apply_parent_prompt(&manager).await;
            let sessions = manager.inner.lock().await;
            let parent = &sessions["parent-thread"];
            assert!(parent.root_turn.is_open());
            assert_eq!(parent.task_scope.as_ref().unwrap().uuid, task_identifier);
            assert_eq!(
                parent.retiring_subagents["child-thread"]
                    .child_turn
                    .is_some(),
                child_turn
            );
        }
        unblock_child(stack, blocker).await;
        // An explicit completion hook must report a worker that was already retiring.
        manager
            .apply_authenticated_events(
                &HeaderMap::new(),
                vec![NormalizedEvent::SubagentEnded(SubagentEvent {
                    session_id: "parent-thread".into(),
                    agent_kind: AgentKind::Codex,
                    event_name: "SubagentStop".into(),
                    subagent_id: "child-thread".into(),
                    payload: json!({}),
                    metadata: json!({"hook_event_name": "SubagentStop"}),
                })],
                "client-a",
            )
            .await
            .unwrap();
        assert!(!has_alignment_alias(&manager, "child-thread").await);
        assert!(
            manager.inner.lock().await["parent-thread"]
                .retiring_subagents
                .is_empty()
        );
        flush_subscribers().unwrap();
        {
            let events = captured.lock().unwrap();
            let expected = turn_identifier
                .into_iter()
                .chain([worker_identifier])
                .collect::<Vec<_>>();
            let observed = events
                .iter()
                .filter(|event| {
                    expected.contains(&event.uuid())
                        && event.scope_category() == Some(ScopeCategory::End)
                })
                .map(Event::uuid)
                .collect::<Vec<_>>();
            assert_eq!(
                observed, expected,
                "child turn must end before its worker, exactly once"
            );
            for event in events.iter().filter(|event| {
                expected.contains(&event.uuid())
                    && event.scope_category() == Some(ScopeCategory::End)
            }) {
                assert_eq!(event.metadata().unwrap()["hook_event_name"], "SubagentStop");
            }
        }
        manager.close_all("test_shutdown").await.unwrap();
        deregister_subscriber(subscriber_name).unwrap();
    }
}

#[tokio::test]
async fn pending_retirement_identifier_reuse_preserves_replacement_alias() {
    let manager = retirement_fixture(false).await;
    let original_identifier = manager.inner.lock().await["parent-thread"].subagents["child-thread"]
        .scope_handle
        .uuid;
    let (stack, blocker) = block_child(&manager, "child-thread").await;
    let sweep_at = Instant::now() + AGENT_IDLE_TIMEOUT + Duration::from_secs(1);
    manager
        .close_idle_sessions_at(sweep_at, AGENT_IDLE_TIMEOUT, "idle_timeout")
        .await
        .unwrap_err();
    unblock_child(stack, blocker).await;
    manager
        .apply_authenticated_events(
            &HeaderMap::new(),
            vec![NormalizedEvent::SubagentStarted(SubagentEvent {
                session_id: "parent-thread".into(),
                agent_kind: AgentKind::Codex,
                event_name: "SubagentStart".into(),
                subagent_id: "child-thread".into(),
                payload: json!({}),
                metadata: json!({}),
            })],
            "client-a",
        )
        .await
        .unwrap();
    assert!(has_alignment_alias(&manager, "child-thread").await);
    assert_ne!(
        manager.inner.lock().await["parent-thread"].subagents["child-thread"]
            .scope_handle
            .uuid,
        original_identifier
    );
    manager.close_all("test_shutdown").await.unwrap();
}

#[tokio::test]
async fn idle_alias_cleanup_holds_parent_gate_through_success_and_partial_failure() {
    for partial_failure in [false, true] {
        let manager = Arc::new(retirement_fixture(false).await);
        if partial_failure {
            start_child_alias(&manager, "sibling-thread").await;
        }
        let sweep_at = Instant::now() + AGENT_IDLE_TIMEOUT + Duration::from_secs(1);
        let (replacement_identifier, blocked_identifier) = {
            let sessions = manager.inner.lock().await;
            let candidates =
                sessions["parent-thread"].stale_subagent_ids(sweep_at, AGENT_IDLE_TIMEOUT);
            (candidates[0].clone(), candidates.get(1).cloned())
        };
        let blocker = if let Some(identifier) = blocked_identifier.as_ref() {
            Some(block_child(&manager, identifier).await)
        } else {
            None
        };
        let alignment_guard = manager.alignment.lock().await;
        let gate = session_gate(&manager.session_gates, "parent-thread").await;
        let sweeping_manager = Arc::clone(&manager);
        let mut sweep = Box::pin(async move {
            sweeping_manager
                .close_idle_sessions_at(sweep_at, AGENT_IDLE_TIMEOUT, "idle_timeout")
                .await
        });
        wait_for_idle_alias_cleanup(&mut sweep, &manager).await;
        assert!(
            gate.try_lock().is_err(),
            "alignment cleanup must retain the parent gate"
        );
        let replacement_manager = Arc::clone(&manager);
        let replacement_child = replacement_identifier.clone();
        let replacement = tokio::spawn(async move {
            start_child_alias(&replacement_manager, &replacement_child).await;
        });
        tokio::task::yield_now().await;
        assert!(!replacement.is_finished());
        drop(alignment_guard);
        let result = sweep.await;
        assert_eq!(result.is_err(), partial_failure);
        replacement.await.unwrap();
        assert!(has_alignment_alias(&manager, &replacement_identifier).await);
        assert!(
            manager.inner.lock().await["parent-thread"]
                .subagents
                .contains_key(&replacement_identifier)
        );
        if let Some((stack, blocker)) = blocker {
            unblock_child(stack, blocker).await;
        }
        manager.close_all("test_shutdown").await.unwrap();
    }
}

#[tokio::test]
async fn cancelled_idle_alias_cleanup_restores_session_and_preserves_replacement() {
    let manager = Arc::new(retirement_fixture(false).await);
    let alignment_guard = manager.alignment.lock().await;
    let gate = session_gate(&manager.session_gates, "parent-thread").await;
    let sweep_at = Instant::now() + AGENT_IDLE_TIMEOUT + Duration::from_secs(1);
    let mut sweep =
        Box::pin(manager.close_idle_sessions_at(sweep_at, AGENT_IDLE_TIMEOUT, "idle_timeout"));
    wait_for_idle_alias_cleanup(&mut sweep, &manager).await;
    assert!(manager.inner.lock().await.contains_key("parent-thread"));
    assert!(gate.try_lock().is_err());
    drop(sweep);
    assert!(
        gate.try_lock().is_err(),
        "cancellation recovery must retain the parent gate"
    );
    drop(alignment_guard);
    let cleanup_gate = gate.lock().await;
    assert!(
        !has_alignment_alias(&manager, "child-thread").await,
        "cancelled cleanup must remove the old alias before replacement"
    );
    drop(cleanup_gate);
    start_child_alias(&manager, "child-thread").await;
    assert!(has_alignment_alias(&manager, "child-thread").await);
    assert!(
        manager.inner.lock().await["parent-thread"]
            .subagents
            .contains_key("child-thread")
    );
    manager.close_all("test_shutdown").await.unwrap();
}

#[test]
fn retirement_alias_cleanup_preserves_another_parents_replacement_route() {
    let mut alignment = SessionAlignmentState::default();
    alignment.insert_alias(
        "child-thread".into(),
        SessionAlias::new(
            "replacement-parent".into(),
            "child-thread".into(),
            json!({}),
        ),
    );
    alignment.clear_for_ended_subagent("old-parent", "child-thread");
    assert_eq!(
        alignment
            .alias_for_session("child-thread")
            .unwrap()
            .parent_session_id,
        "replacement-parent"
    );
    alignment.clear_for_ended_subagent("replacement-parent", "child-thread");
    assert!(alignment.alias_for_session("child-thread").is_none());
}

#[tokio::test]
async fn retirement_publication_failure_reports_error_and_completes_cleanup() {
    for child_turn in [false, true] {
        let captured = Arc::new(StdMutex::new(Vec::<Event>::new()));
        let capture = Arc::clone(&captured);
        let subscriber_name = "retirement-publication-failure";
        register_subscriber(
            subscriber_name,
            Arc::new(move |event| capture.lock().unwrap().push(event.clone())),
        )
        .unwrap();
        let manager = retirement_fixture(child_turn).await;
        let (stack, failing_identifier, worker_identifier) = {
            let sessions = manager.inner.lock().await;
            let child = &sessions["parent-thread"].subagents["child-thread"];
            (
                child.scope_stack.clone(),
                child.work_scope().uuid,
                child.scope_handle.uuid,
            )
        };
        let sweep_at = Instant::now() + AGENT_IDLE_TIMEOUT + Duration::from_secs(1);
        let result = RETIREMENT_PUBLICATION_FAILURE
            .scope(
                failing_identifier,
                manager.close_idle_sessions_at(sweep_at, AGENT_IDLE_TIMEOUT, "idle_timeout"),
            )
            .await;
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("injected failure to queue")
        );
        assert!(stack.read().unwrap().find(&failing_identifier).is_none());
        assert!(stack.read().unwrap().find(&worker_identifier).is_none());
        assert!(!has_alignment_alias(&manager, "child-thread").await);
        {
            let sessions = manager.inner.lock().await;
            let parent = &sessions["parent-thread"];
            assert!(parent.retiring_subagents.is_empty());
            assert!(parent.completed_subagents.contains("child-thread"));
            assert!(parent.task_scope.is_none());
        }
        assert_eq!(
            manager
                .close_idle_sessions_at(sweep_at, AGENT_IDLE_TIMEOUT, "idle_timeout")
                .await
                .unwrap(),
            0,
            "the removed scope must not be popped again"
        );
        flush_subscribers().unwrap();
        assert_eq!(
            captured
                .lock()
                .unwrap()
                .iter()
                .filter(|event| {
                    event.uuid() == failing_identifier
                        && event.scope_category() == Some(ScopeCategory::End)
                })
                .count(),
            0
        );
        apply_parent_prompt(&manager).await;
        manager.close_all("test_shutdown").await.unwrap();
        deregister_subscriber(subscriber_name).unwrap();
    }
}

#[tokio::test]
async fn explicit_completion_retry_preserves_the_synthesis_task() {
    SESSION_CLOCK_OVERRIDE
        .scope(std::cell::Cell::new(Instant::now()), async {
            let manager = retirement_fixture(false).await;
            let task_identifier = manager.inner.lock().await["parent-thread"]
                .task_scope
                .as_ref()
                .unwrap()
                .uuid;
            let (stack, blocker) = block_child(&manager, "child-thread").await;
            let ending = NormalizedEvent::SubagentEnded(SubagentEvent {
                session_id: "parent-thread".into(),
                agent_kind: AgentKind::Codex,
                event_name: "SubagentStop".into(),
                subagent_id: "child-thread".into(),
                payload: json!({}),
                metadata: json!({}),
            });
            manager
                .apply_authenticated_events(&HeaderMap::new(), vec![ending], "client-a")
                .await
                .unwrap_err();
            unblock_child(stack, blocker).await;
            SESSION_CLOCK_OVERRIDE.with(|clock| clock.set(clock.get() + Duration::from_secs(1)));
            manager
                .close_idle_sessions_at(session_now(), AGENT_IDLE_TIMEOUT, "idle_timeout")
                .await
                .unwrap();
            {
                let sessions = manager.inner.lock().await;
                let parent = &sessions["parent-thread"];
                assert!(parent.task_awaiting_synthesis);
                assert_eq!(parent.task_scope.as_ref().unwrap().uuid, task_identifier);
                assert!(parent.retiring_subagents.is_empty());
            }
            assert!(!has_alignment_alias(&manager, "child-thread").await);
            apply_parent_prompt(&manager).await;
            assert_eq!(
                manager.inner.lock().await["parent-thread"]
                    .task_scope
                    .as_ref()
                    .unwrap()
                    .uuid,
                task_identifier
            );
            manager.close_all("test_shutdown").await.unwrap();
        })
        .await;
}

#[tokio::test]
async fn removed_active_child_turn_does_not_block_the_next_child_prompt() {
    let manager = retirement_fixture(true).await;
    let old_turn = manager.inner.lock().await["parent-thread"].subagents["child-thread"]
        .child_turn
        .as_ref()
        .unwrap()
        .uuid;
    let ending = NormalizedEvent::TurnEnded(SessionEvent {
        session_id: "parent-thread".into(),
        agent_kind: AgentKind::Codex,
        event_name: "Stop".into(),
        payload: json!({}),
        metadata: json!({"agent_id": "child-thread"}),
    });
    let error = RETIREMENT_PUBLICATION_FAILURE
        .scope(
            old_turn,
            manager.apply_authenticated_events(&HeaderMap::new(), vec![ending], "client-a"),
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("injected failure to queue"));
    assert!(
        manager.inner.lock().await["parent-thread"].subagents["child-thread"]
            .child_turn
            .is_none()
    );
    let prompt = NormalizedEvent::PromptSubmitted(SessionEvent {
        session_id: "parent-thread".into(),
        agent_kind: AgentKind::Codex,
        event_name: "UserPromptSubmit".into(),
        payload: json!({}),
        metadata: json!({"agent_id": "child-thread"}),
    });
    manager
        .apply_authenticated_events(&HeaderMap::new(), vec![prompt], "client-a")
        .await
        .unwrap();
    let new_turn = manager.inner.lock().await["parent-thread"].subagents["child-thread"]
        .child_turn
        .as_ref()
        .unwrap()
        .uuid;
    assert_ne!(old_turn, new_turn);
    let call = manager
        .start_llm(
            &HeaderMap::new(),
            LlmGatewayStart {
                session_id: Some("parent-thread".into()),
                subagent_id: Some("child-thread".into()),
                ..llm_start()
            },
        )
        .await
        .unwrap();
    assert_eq!(call.handle.parent_uuid, Some(new_turn));
    manager.end_llm(call, json!({}), json!({})).await.unwrap();
    manager.close_all("test_shutdown").await.unwrap();
}

#[tokio::test]
async fn failed_hooks_clear_completed_children_without_committing_terminal_routes() {
    SESSION_CLOCK_OVERRIDE
        .scope(std::cell::Cell::new(Instant::now()), async {
            for boundary in ["prompt", "session_end"] {
                let manager = retirement_fixture(false).await;
                start_child_alias(&manager, "sibling-thread").await;
                SESSION_CLOCK_OVERRIDE.with(|clock| {
                    clock.set(clock.get() + AGENT_IDLE_TIMEOUT + Duration::from_secs(1))
                });
                let (completed_identifier, blocked_identifier) = {
                    let sessions = manager.inner.lock().await;
                    let parent = &sessions["parent-thread"];
                    let identifiers = if boundary == "prompt" {
                        parent.stale_subagent_ids(session_now(), AGENT_IDLE_TIMEOUT)
                    } else {
                        parent.subagent_stack.iter().rev().cloned().collect()
                    };
                    (identifiers[0].clone(), identifiers[1].clone())
                };
                let (stack, blocker) = block_child(&manager, &blocked_identifier).await;
                let event = if boundary == "prompt" {
                    NormalizedEvent::PromptSubmitted(codex_session_event(
                        "parent-thread",
                        "UserPromptSubmit",
                        json!({}),
                    ))
                } else {
                    NormalizedEvent::AgentEnded(codex_session_event(
                        "parent-thread",
                        "SessionEnd",
                        json!({}),
                    ))
                };
                let error = manager
                    .apply_authenticated_events(&HeaderMap::new(), vec![event], "client-a")
                    .await
                    .unwrap_err();
                assert!(error.to_string().contains("not at the top"));
                assert!(!has_alignment_alias(&manager, &completed_identifier).await);
                assert!(
                    has_alignment_alias(&manager, &blocked_identifier).await,
                    "a failed terminal hook must not clear its still-pending child route"
                );
                unblock_child(stack, blocker).await;
                manager
                    .close_idle_sessions_at(session_now(), AGENT_IDLE_TIMEOUT, "idle_timeout")
                    .await
                    .unwrap();
                assert!(!has_alignment_alias(&manager, &blocked_identifier).await);
                manager.close_all("test_shutdown").await.unwrap();
            }
        })
        .await;
}

#[tokio::test]
async fn idle_retirement_delivery_blocks_shutdown_and_rejects_new_checkouts() {
    let manager = Arc::new(retirement_fixture(false).await);
    let (stack, blocker) = block_child(&manager, "child-thread").await;
    manager
        .apply_authenticated_events(
            &HeaderMap::new(),
            vec![NormalizedEvent::SubagentEnded(SubagentEvent {
                session_id: "parent-thread".into(),
                agent_kind: AgentKind::Codex,
                event_name: "SubagentStop".into(),
                subagent_id: "child-thread".into(),
                payload: json!({}),
                metadata: json!({}),
            })],
            "client-a",
        )
        .await
        .unwrap_err();
    apply_parent_prompt(&manager).await;
    unblock_child(stack, blocker).await;
    let (worker_identifier, root_identifier) = {
        let sessions = manager.inner.lock().await;
        let parent = &sessions["parent-thread"];
        (
            parent.retiring_subagents["child-thread"].scope.handle.uuid,
            parent.root_turn.handle().unwrap().uuid,
        )
    };
    let (entered_sender, entered_receiver) = tokio::sync::oneshot::channel();
    let entered_sender = Arc::new(StdMutex::new(Some(entered_sender)));
    let (release_sender, release_receiver) = std::sync::mpsc::channel();
    let release_receiver = StdMutex::new(release_receiver);
    let captured = Arc::new(StdMutex::new(Vec::<Event>::new()));
    let capture = Arc::clone(&captured);
    let subscriber_name = "idle-shutdown-barrier";
    register_subscriber(
        subscriber_name,
        Arc::new(move |event| {
            capture.lock().unwrap().push(event.clone());
            if event.uuid() == worker_identifier
                && event.scope_category() == Some(ScopeCategory::End)
                && let Some(sender) = entered_sender.lock().unwrap().take()
            {
                let _ = sender.send(());
                let _ = release_receiver.lock().unwrap().recv();
            }
        }),
    )
    .unwrap();
    let sweeping = Arc::clone(&manager);
    let sweep = tokio::spawn(async move {
        sweeping
            .close_idle_sessions_at(Instant::now(), AGENT_IDLE_TIMEOUT, "idle_timeout")
            .await
    });
    tokio::time::timeout(Duration::from_secs(2), entered_receiver)
        .await
        .unwrap()
        .unwrap();
    assert!(!manager.inner.lock().await.contains_key("parent-thread"));
    let closing = Arc::clone(&manager);
    let shutdown = tokio::spawn(async move { closing.close_all("test_shutdown").await });
    tokio::time::timeout(Duration::from_secs(2), async {
        while !manager.session_activity.is_closing() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert!(
        !shutdown.is_finished(),
        "shutdown must wait for the checked-out root session"
    );
    assert_eq!(
        manager
            .close_idle_sessions_at(Instant::now(), AGENT_IDLE_TIMEOUT, "idle_timeout")
            .await
            .unwrap(),
        0
    );
    release_sender.send(()).unwrap();
    sweep.await.unwrap().unwrap();
    shutdown.await.unwrap().unwrap();
    assert!(manager.inner.lock().await.is_empty());
    flush_subscribers().unwrap();
    assert_eq!(
        captured
            .lock()
            .unwrap()
            .iter()
            .filter(|event| event.uuid() == root_identifier
                && event.scope_category() == Some(ScopeCategory::End))
            .count(),
        1
    );
    deregister_subscriber(subscriber_name).unwrap();
}

struct BlockedRetirementDelivery {
    entered: tokio::sync::oneshot::Receiver<()>,
    release: Option<std::sync::mpsc::Sender<()>>,
    _subscriber_cleanup: SessionSubscriberCleanup,
}

impl BlockedRetirementDelivery {
    fn new(worker_identifier: uuid::Uuid, subscriber_name: &'static str) -> Self {
        let (entered_sender, entered) = tokio::sync::oneshot::channel();
        let entered_sender = StdMutex::new(Some(entered_sender));
        let (release_sender, release_receiver) = std::sync::mpsc::channel();
        let release_receiver = StdMutex::new(release_receiver);
        register_subscriber(
            subscriber_name,
            Arc::new(move |event| {
                if event.uuid() == worker_identifier
                    && event.scope_category() == Some(ScopeCategory::End)
                    && let Some(sender) = entered_sender.lock().unwrap().take()
                {
                    let _ = sender.send(());
                    let _ = release_receiver.lock().unwrap().recv();
                }
            }),
        )
        .unwrap();
        Self {
            entered,
            release: Some(release_sender),
            _subscriber_cleanup: SessionSubscriberCleanup(subscriber_name),
        }
    }

    async fn wait_until_blocked<F: std::future::Future>(
        &mut self,
        operation: &mut std::pin::Pin<Box<F>>,
    ) {
        tokio::time::timeout(
            Duration::from_secs(2),
            std::future::poll_fn(|context| {
                assert!(std::future::Future::poll(operation.as_mut(), context).is_pending());
                match self.entered.try_recv() {
                    Ok(()) => std::task::Poll::Ready(()),
                    Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                        context.waker().wake_by_ref();
                        std::task::Poll::Pending
                    }
                    Err(error) => panic!("End delivery channel closed: {error}"),
                }
            }),
        )
        .await
        .expect("worker End should reach the blocked subscriber");
    }

    fn release(&mut self) {
        self.release.take().unwrap().send(()).unwrap();
    }
}

impl Drop for BlockedRetirementDelivery {
    fn drop(&mut self) {
        if let Some(release) = self.release.take() {
            let _ = release.send(());
        }
    }
}

#[tokio::test]
async fn cancelled_idle_end_delivery_restores_session_and_clears_completed_alias() {
    let manager = retirement_fixture(false).await;
    let (stack, blocker) = block_child(&manager, "child-thread").await;
    manager
        .apply_authenticated_events(
            &HeaderMap::new(),
            vec![NormalizedEvent::SubagentEnded(SubagentEvent {
                session_id: "parent-thread".into(),
                agent_kind: AgentKind::Codex,
                event_name: "SubagentStop".into(),
                subagent_id: "child-thread".into(),
                payload: json!({}),
                metadata: json!({}),
            })],
            "client-a",
        )
        .await
        .unwrap_err();
    apply_parent_prompt(&manager).await;
    unblock_child(stack, blocker).await;
    let worker_identifier =
        manager.inner.lock().await["parent-thread"].retiring_subagents["child-thread"]
            .scope
            .handle
            .uuid;
    let mut delivery = BlockedRetirementDelivery::new(worker_identifier, "cancelled-idle-delivery");
    let gate = session_gate(&manager.session_gates, "parent-thread").await;
    let alignment = manager.alignment.lock().await;
    let mut sweep = Box::pin(manager.close_idle_sessions_at(
        Instant::now(),
        AGENT_IDLE_TIMEOUT,
        "idle_timeout",
    ));
    delivery.wait_until_blocked(&mut sweep).await;
    assert!(!manager.inner.lock().await.contains_key("parent-thread"));
    drop(sweep);
    tokio::time::timeout(Duration::from_secs(2), async {
        while !manager.inner.lock().await.contains_key("parent-thread") {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert!(alignment.alias_for_session("child-thread").is_some());
    assert!(gate.try_lock().is_err());
    assert!(manager.session_activity.has_active());
    drop(alignment);
    let cleanup_gate = tokio::time::timeout(Duration::from_secs(2), gate.lock())
        .await
        .expect("recovery should release the parent gate after clearing aliases");
    assert!(!has_alignment_alias(&manager, "child-thread").await);
    {
        let sessions = manager.inner.lock().await;
        let parent = &sessions["parent-thread"];
        assert!(parent.root_turn.is_open());
        assert!(parent.subagents.is_empty());
        assert!(parent.retiring_subagents.is_empty());
        assert!(
            !parent
                .subagent_stack
                .iter()
                .any(|identifier| identifier == "child-thread")
        );
    }
    drop(cleanup_gate);
    delivery.release();
    flush_subscribers().unwrap();
    start_child_alias(&manager, "child-thread").await;
    assert!(has_alignment_alias(&manager, "child-thread").await);
    manager.close_all("test_shutdown").await.unwrap();
}

#[tokio::test]
async fn blocked_retirement_does_not_prevent_a_ready_sibling_from_retiring() {
    let manager = retirement_fixture(false).await;
    start_child_alias(&manager, "sibling-thread").await;
    let task_identifier = manager.inner.lock().await["parent-thread"]
        .task_scope
        .as_ref()
        .unwrap()
        .uuid;
    let mut blockers = HashMap::new();
    for identifier in ["child-thread", "sibling-thread"] {
        blockers.insert(
            identifier.to_string(),
            block_child(&manager, identifier).await,
        );
        manager
            .apply_authenticated_events(
                &HeaderMap::new(),
                vec![NormalizedEvent::SubagentEnded(SubagentEvent {
                    session_id: "parent-thread".into(),
                    agent_kind: AgentKind::Codex,
                    event_name: "SubagentStop".into(),
                    subagent_id: identifier.into(),
                    payload: json!({}),
                    metadata: json!({}),
                })],
                "client-a",
            )
            .await
            .unwrap_err();
    }
    let candidates: Vec<_> = manager.inner.lock().await["parent-thread"]
        .retiring_subagents
        .keys()
        .cloned()
        .collect();
    // Match the real sweep's stable map order, then unblock only the later candidate.
    let blocked_identifier = &candidates[0];
    let ready_identifier = &candidates[1];
    let (stack, blocker) = blockers.remove(ready_identifier).unwrap();
    unblock_child(stack, blocker).await;
    let error = manager
        .close_idle_sessions_at(Instant::now(), AGENT_IDLE_TIMEOUT, "idle_timeout")
        .await
        .unwrap_err();
    assert!(error.to_string().contains("not at the top"));
    assert!(has_alignment_alias(&manager, blocked_identifier).await);
    assert!(!has_alignment_alias(&manager, ready_identifier).await);
    {
        let sessions = manager.inner.lock().await;
        let parent = &sessions["parent-thread"];
        assert!(parent.retiring_subagents.contains_key(blocked_identifier));
        assert!(!parent.retiring_subagents.contains_key(ready_identifier));
        assert!(parent.completed_subagents.contains(ready_identifier));
        assert_eq!(parent.task_scope.as_ref().unwrap().uuid, task_identifier);
    }
    let (stack, blocker) = blockers.remove(blocked_identifier).unwrap();
    unblock_child(stack, blocker).await;
    manager
        .close_idle_sessions_at(Instant::now(), AGENT_IDLE_TIMEOUT, "idle_timeout")
        .await
        .unwrap();
    assert!(!has_alignment_alias(&manager, blocked_identifier).await);
    manager.close_all("test_shutdown").await.unwrap();
}

#[tokio::test]
async fn cancelled_hook_restoration_clears_completed_alias_before_releasing_gate() {
    let manager = retirement_fixture(true).await;
    start_child_alias(&manager, "sibling-thread").await;
    let (worker_identifier, child_turn_identifier) = {
        let sessions = manager.inner.lock().await;
        let child = &sessions["parent-thread"].subagents["child-thread"];
        (
            child.scope_handle.uuid,
            child.child_turn.as_ref().unwrap().uuid,
        )
    };
    let mut delivery = BlockedRetirementDelivery::new(worker_identifier, "cancelled-hook-restore");
    let gate = session_gate(&manager.session_gates, "parent-thread").await;
    let ending = vec![NormalizedEvent::SubagentEnded(SubagentEvent {
        session_id: "parent-thread".into(),
        agent_kind: AgentKind::Codex,
        event_name: "SubagentStop".into(),
        subagent_id: "child-thread".into(),
        payload: json!({}),
        metadata: json!({}),
    })];
    let headers = HeaderMap::new();
    let mut applying = Box::pin(RETIREMENT_PUBLICATION_FAILURE.scope(
        child_turn_identifier,
        manager.apply_authenticated_events(&headers, ending, "client-a"),
    ));
    delivery.wait_until_blocked(&mut applying).await;
    let directory = manager.inner.lock().await;
    let alignment = manager.alignment.lock().await;
    assert!(!directory.contains_key("parent-thread"));
    delivery.release();
    // The receipt is now complete; with the directory held, the next poll reaches restore().
    flush_subscribers().unwrap();
    std::future::poll_fn(|context| {
        assert!(std::future::Future::poll(applying.as_mut(), context).is_pending());
        std::task::Poll::Ready(())
    })
    .await;
    drop(applying);
    assert!(gate.try_lock().is_err());
    assert!(manager.session_activity.has_active());
    drop(directory);
    tokio::time::timeout(Duration::from_secs(2), async {
        while !manager.inner.lock().await.contains_key("parent-thread") {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert!(gate.try_lock().is_err());
    assert!(alignment.alias_for_session("child-thread").is_some());
    drop(alignment);
    let cleanup_gate = tokio::time::timeout(Duration::from_secs(2), gate.lock())
        .await
        .expect("restoration must clear completed aliases before releasing the gate");
    assert!(!has_alignment_alias(&manager, "child-thread").await);
    assert!(has_alignment_alias(&manager, "sibling-thread").await);
    {
        let sessions = manager.inner.lock().await;
        let parent = &sessions["parent-thread"];
        assert!(!parent.subagents.contains_key("child-thread"));
        assert!(!parent.retiring_subagents.contains_key("child-thread"));
        assert!(parent.completed_subagents.contains("child-thread"));
    }
    drop(cleanup_gate);
    start_child_alias(&manager, "child-thread").await;
    assert!(has_alignment_alias(&manager, "child-thread").await);
    manager.close_all("test_shutdown").await.unwrap();
}
