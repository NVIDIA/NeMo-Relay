// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for mark and metric API behavior.

use std::sync::{Arc, Mutex};

use chrono::{TimeZone, Utc};
use serde_json::json;

use super::{
    EmitMarkEventParams, EmitMetricEventParams, event, metadata_with_log_severity, metric,
};
use crate::api::event::{
    Event, LOG_SEVERITY_METADATA_KEY, LogSeverity, METRIC_DATA_SCHEMA_NAME,
    METRIC_DATA_SCHEMA_VERSION, MetricKind, MetricMeasurement, MetricValueType,
};
use crate::api::runtime::{NemoRelayContextState, global_context};
use crate::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use crate::error::FlowError;

fn reset_global() {
    crate::shared_runtime::reset_runtime_owner_for_tests();
    let context = global_context();
    *context.write().unwrap() = NemoRelayContextState::new();
}

fn lock_global_runtime() -> std::sync::MutexGuard<'static, ()> {
    crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
}

#[test]
fn typed_severity_overrides_metadata_and_requires_an_object() {
    let metadata = metadata_with_log_severity(
        Some(json!({
            (LOG_SEVERITY_METADATA_KEY): "trace",
            "source": "test"
        })),
        Some(LogSeverity::Error),
    )
    .unwrap()
    .unwrap();
    assert_eq!(metadata[LOG_SEVERITY_METADATA_KEY], "error");
    assert_eq!(metadata["source"], "test");

    let metadata = metadata_with_log_severity(None, Some(LogSeverity::Debug))
        .unwrap()
        .unwrap();
    assert_eq!(metadata[LOG_SEVERITY_METADATA_KEY], "debug");

    let error = metadata_with_log_severity(Some(json!("not-an-object")), Some(LogSeverity::Info))
        .unwrap_err();
    assert!(matches!(error, FlowError::InvalidArgument(_)));

    assert_eq!(
        metadata_with_log_severity(Some(json!("unchanged")), None).unwrap(),
        Some(json!("unchanged"))
    );
}

#[test]
fn event_and_metric_emit_canonical_mark_metadata_and_schema() {
    let _guard = lock_global_runtime();
    reset_global();

    let captured = Arc::new(Mutex::new(Vec::<Event>::new()));
    let subscriber_events = captured.clone();
    register_subscriber(
        "scope-api-mark-observer",
        Arc::new(move |event| subscriber_events.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    event(
        EmitMarkEventParams::builder()
            .name("application.warning")
            .metadata(json!({
                (LOG_SEVERITY_METADATA_KEY): "trace",
                "source": "test"
            }))
            .severity(LogSeverity::Warn)
            .build(),
    )
    .unwrap();

    let timestamp = Utc.with_ymd_and_hms(2026, 1, 2, 3, 4, 5).unwrap();
    metric(
        EmitMetricEventParams::builder()
            .name("tokenomics.sample")
            .measurements(vec![
                MetricMeasurement::builder()
                    .name("example.tokens.saved")
                    .kind(MetricKind::Counter)
                    .value_type(MetricValueType::U64)
                    .value(json!(42u64))
                    .unit("{token}")
                    .attributes(json!({"model": "example-model"}))
                    .build(),
            ])
            .metadata(json!({"source": "test"}))
            .timestamp(timestamp)
            .build(),
    )
    .unwrap();

    flush_subscribers().unwrap();
    assert!(deregister_subscriber("scope-api-mark-observer").unwrap());

    let events = captured.lock().unwrap();
    let warning = events
        .iter()
        .find(|event| event.name() == "application.warning")
        .expect("warning mark should be emitted");
    assert_eq!(
        warning.metadata().unwrap()[LOG_SEVERITY_METADATA_KEY],
        "warn"
    );
    assert_eq!(warning.metadata().unwrap()["source"], "test");

    let metric = events
        .iter()
        .find(|event| event.name() == "tokenomics.sample")
        .expect("metric mark should be emitted");
    assert_eq!(metric.timestamp(), &timestamp);
    assert_eq!(metric.data_schema().unwrap().name, METRIC_DATA_SCHEMA_NAME);
    assert_eq!(
        metric.data_schema().unwrap().version,
        METRIC_DATA_SCHEMA_VERSION
    );
    assert_eq!(metric.data().unwrap()["measurements"][0]["value"], 42);
    assert_eq!(metric.metadata().unwrap()["source"], "test");
}

#[test]
fn metric_rejects_the_entire_invalid_envelope_before_emission() {
    let _guard = lock_global_runtime();
    reset_global();

    let error = metric(
        EmitMetricEventParams::builder()
            .name("invalid.metric")
            .measurements(Vec::new())
            .build(),
    )
    .unwrap_err();
    assert!(matches!(error, FlowError::InvalidArgument(_)));

    let error = event(
        EmitMarkEventParams::builder()
            .name("invalid.severity.metadata")
            .metadata(json!(["not", "an", "object"]))
            .severity(LogSeverity::Info)
            .build(),
    )
    .unwrap_err();
    assert!(matches!(error, FlowError::InvalidArgument(_)));
}

#[test]
fn malformed_reserved_tool_completion_marks_are_rejected_before_publication() {
    let _guard = lock_global_runtime();
    reset_global();
    let captured = Arc::new(Mutex::new(Vec::<Event>::new()));
    let capture = captured.clone();
    register_subscriber(
        "invalid-completion-mark",
        Arc::new(move |event| {
            capture.lock().unwrap().push(event.clone());
        }),
    )
    .unwrap();
    for data in [
        None,
        Some(json!({"arguments": {"secret": "request"}, "result": "secret"})),
        Some(json!({"tool_name": 7, "result": "secret"})),
        Some(json!({"tool_name": "   ", "arguments": "secret"})),
    ] {
        for metadata in [
            None,
            Some(json!({"start_observed": false})),
            Some(json!({"start_observed": true})),
        ] {
            let error = event(
                EmitMarkEventParams::builder()
                    .name("tool_end_without_start")
                    .data_opt(data.clone())
                    .metadata_opt(metadata)
                    .build(),
            )
            .unwrap_err();
            assert!(matches!(error, FlowError::InvalidArgument(_)));
        }
    }
    flush_subscribers().unwrap();
    assert!(captured.lock().unwrap().is_empty());
    deregister_subscriber("invalid-completion-mark").unwrap();
}

#[test]
fn reserved_tool_completion_marks_sanitize_payloads_without_start_metadata() {
    use crate::api::registry::{
        deregister_tool_sanitize_request_guardrail, deregister_tool_sanitize_response_guardrail,
        register_tool_sanitize_request_guardrail, register_tool_sanitize_response_guardrail,
    };
    let _guard = lock_global_runtime();
    reset_global();
    register_tool_sanitize_request_guardrail(
        "completion-request",
        1,
        Arc::new(|_, _| Box::pin(async { Ok(json!({"request": "redacted"})) })),
    )
    .unwrap();
    register_tool_sanitize_response_guardrail(
        "completion-response",
        1,
        Arc::new(|_, _| Box::pin(async { Ok(json!({"response": "redacted"})) })),
    )
    .unwrap();
    let captured = Arc::new(Mutex::new(Vec::<Event>::new()));
    let capture = captured.clone();
    register_subscriber(
        "completion-sanitizer-observer",
        Arc::new(move |event| {
            capture.lock().unwrap().push(event.clone());
        }),
    )
    .unwrap();
    event(EmitMarkEventParams::builder().name("tool_end_without_start")
        .data(json!({"tool_name": "example", "arguments": "request-secret", "result": "response-secret"}))
        .build()).unwrap();
    flush_subscribers().unwrap();
    let events = captured.lock().unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!(
        events[0].data().unwrap()["arguments"],
        json!({"request": "redacted"})
    );
    assert_eq!(
        events[0].data().unwrap()["result"],
        json!({"response": "redacted"})
    );
    drop(events);
    deregister_subscriber("completion-sanitizer-observer").unwrap();
    deregister_tool_sanitize_request_guardrail("completion-request").unwrap();
    deregister_tool_sanitize_response_guardrail("completion-response").unwrap();
}

#[test]
fn scope_stack_snapshot_is_nonactivating_and_detached() {
    std::thread::spawn(|| {
        use crate::api::runtime::{
            create_scope_stack, current_scope_stack_snapshot, scope_stack_active, with_scope_stack,
        };
        use crate::api::scope::{
            PopScopeParams, PushScopeParams, ScopeType, pop_scope, push_scope,
        };
        assert!(!scope_stack_active());
        assert_eq!(current_scope_stack_snapshot().unwrap(), None);
        assert!(!scope_stack_active());
        let stack = create_scope_stack();
        let root_uuid = stack.read().unwrap().scopes()[0].uuid;
        with_scope_stack(stack, || {
            let root = current_scope_stack_snapshot().unwrap().unwrap();
            assert_eq!(root.scopes.len(), 1);
            assert_eq!(root.scopes[0].scope_id, root_uuid);
            let agent = push_scope(
                PushScopeParams::builder()
                    .name("agent")
                    .scope_type(ScopeType::Agent)
                    .build(),
            )
            .unwrap();
            let function = push_scope(
                PushScopeParams::builder()
                    .name("function")
                    .scope_type(ScopeType::Function)
                    .build(),
            )
            .unwrap();
            let tool = push_scope(
                PushScopeParams::builder()
                    .name("tool")
                    .scope_type(ScopeType::Tool)
                    .build(),
            )
            .unwrap();
            let before_pop = current_scope_stack_snapshot().unwrap().unwrap();
            assert_eq!(
                before_pop
                    .scopes
                    .iter()
                    .map(|entry| entry.scope_id)
                    .collect::<Vec<_>>(),
                vec![root_uuid, agent.uuid, function.uuid, tool.uuid]
            );
            assert_eq!(before_pop.scopes[3].scope_type, ScopeType::Tool);
            assert_eq!(current_scope_stack_snapshot().unwrap().unwrap(), before_pop);
            pop_scope(PopScopeParams::builder().handle_uuid(&tool.uuid).build()).unwrap();
            assert_eq!(
                current_scope_stack_snapshot()
                    .unwrap()
                    .unwrap()
                    .scopes
                    .len(),
                3
            );
            assert_eq!(before_pop.scopes.len(), 4);
            pop_scope(
                PopScopeParams::builder()
                    .handle_uuid(&function.uuid)
                    .build(),
            )
            .unwrap();
            pop_scope(PopScopeParams::builder().handle_uuid(&agent.uuid).build()).unwrap();
        });
        assert_eq!(current_scope_stack_snapshot().unwrap(), None);
    })
    .join()
    .unwrap();
}

#[test]
fn scope_stack_snapshot_binding_precedence_and_restoration() {
    std::thread::spawn(|| {
        use crate::api::runtime::{
            TASK_SCOPE_STACK, create_scope_stack, current_scope_stack_snapshot,
            set_thread_scope_stack, with_scope_stack,
        };
        let thread_stack = create_scope_stack();
        let thread_uuid = thread_stack.read().unwrap().scopes()[0].uuid;
        set_thread_scope_stack(thread_stack);
        assert_eq!(
            current_scope_stack_snapshot().unwrap().unwrap().scopes[0].scope_id,
            thread_uuid
        );
        let task_stack = create_scope_stack();
        let task_uuid = task_stack.read().unwrap().scopes()[0].uuid;
        let executor = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        executor.block_on(TASK_SCOPE_STACK.scope(task_stack, async {
            tokio::task::yield_now().await;
            assert_eq!(
                current_scope_stack_snapshot().unwrap().unwrap().scopes[0].scope_id,
                task_uuid
            );
            let override_stack = create_scope_stack();
            let override_uuid = override_stack.read().unwrap().scopes()[0].uuid;
            with_scope_stack(override_stack, || {
                assert_eq!(
                    current_scope_stack_snapshot().unwrap().unwrap().scopes[0].scope_id,
                    override_uuid
                )
            });
            assert_eq!(
                current_scope_stack_snapshot().unwrap().unwrap().scopes[0].scope_id,
                task_uuid
            );
        }));
        assert_eq!(
            current_scope_stack_snapshot().unwrap().unwrap().scopes[0].scope_id,
            thread_uuid
        );
    })
    .join()
    .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn scope_stack_snapshot_concurrent_agents_keep_identical_call_ids_isolated() {
    use crate::api::runtime::{
        TASK_SCOPE_STACK, create_scope_stack, current_scope_stack_snapshot, with_active_event_uuid,
    };
    use crate::api::scope::{PopScopeParams, PushScopeParams, ScopeType, pop_scope, push_scope};
    let barrier = Arc::new(tokio::sync::Barrier::new(2));
    let call_uuid = uuid::Uuid::now_v7();
    let mut tasks = Vec::new();
    for _ in 0..2 {
        let barrier = barrier.clone();
        tasks.push(tokio::spawn(TASK_SCOPE_STACK.scope(
            create_scope_stack(),
            async move {
                let agent = push_scope(
                    PushScopeParams::builder()
                        .name("agent")
                        .scope_type(ScopeType::Agent)
                        .build(),
                )
                .unwrap();
                let snapshot = with_active_event_uuid(call_uuid, async {
                    barrier.wait().await;
                    tokio::task::yield_now().await;
                    current_scope_stack_snapshot().unwrap().unwrap()
                })
                .await;
                assert_eq!(snapshot.scopes.len(), 2);
                assert_eq!(snapshot.scopes[1].scope_id, agent.uuid);
                assert!(
                    snapshot
                        .scopes
                        .iter()
                        .all(|entry| entry.scope_id != call_uuid)
                );
                pop_scope(PopScopeParams::builder().handle_uuid(&agent.uuid).build()).unwrap();
                snapshot
            },
        )));
    }
    let first = tasks.remove(0).await.unwrap();
    let second = tasks.remove(0).await.unwrap();
    assert_ne!(first.scopes[0].scope_id, second.scopes[0].scope_id);
    assert_ne!(first.scopes[1].scope_id, second.scopes[1].scope_id);
}

#[test]
fn scope_stack_snapshot_preserves_rooted_and_rootless_propagation() {
    use crate::api::runtime::{
        PropagationContext, create_scope_stack_from_propagation, scope_stack_snapshot,
    };
    use crate::api::scope::ScopeType;
    let root_uuid = uuid::Uuid::now_v7();
    let parent_uuid = uuid::Uuid::now_v7();
    for root in [Some(root_uuid), None, Some(parent_uuid)] {
        let context = PropagationContext::from_json(
            &json!({"version":1,"root_uuid":root,"parent_uuid":parent_uuid}).to_string(),
        )
        .unwrap();
        let stack = create_scope_stack_from_propagation(&context).unwrap();
        let snapshot = scope_stack_snapshot(&stack).unwrap();
        assert_eq!(snapshot.scopes[0].scope_id, root.unwrap_or(parent_uuid));
        assert_eq!(snapshot.scopes[0].scope_type, ScopeType::Agent);
        if root == Some(root_uuid) {
            assert_eq!(snapshot.scopes.len(), 2);
            assert_eq!(snapshot.scopes[1].scope_id, parent_uuid);
            assert_eq!(snapshot.scopes[1].scope_type, ScopeType::Unknown);
        } else {
            assert_eq!(snapshot.scopes.len(), 1);
        }
    }
}

#[test]
fn scope_stack_snapshot_poisoning_returns_error_without_activating_context() {
    std::thread::spawn(|| {
        use crate::api::runtime::{
            create_scope_stack, current_scope_stack_snapshot, scope_stack_active,
            scope_stack_snapshot, with_scope_stack,
        };
        let stack = create_scope_stack();
        let poisoned_stack = Arc::clone(&stack);
        assert!(
            std::thread::spawn(move || {
                let _guard = poisoned_stack.write().unwrap();
                panic!("poison the selected scope stack");
            })
            .join()
            .is_err()
        );
        assert!(scope_stack_snapshot(&stack).is_err());
        assert!(!scope_stack_active());
        with_scope_stack(stack, || {
            assert!(current_scope_stack_snapshot().is_err());
        });
        assert_eq!(current_scope_stack_snapshot().unwrap(), None);
        assert!(!scope_stack_active());
    })
    .join()
    .unwrap();
}

#[test]
fn scope_stack_snapshot_owned_values_survive_stack_drop() {
    use crate::api::runtime::{create_scope_stack, scope_stack_snapshot};
    let stack = create_scope_stack();
    let root_uuid = stack.read().unwrap().scopes()[0].uuid;
    let snapshot = scope_stack_snapshot(&stack).unwrap();
    drop(stack);
    assert_eq!(snapshot.scopes.len(), 1);
    assert_eq!(snapshot.scopes[0].scope_id, root_uuid);
    assert_eq!(
        snapshot.scopes[0].scope_type,
        crate::api::scope::ScopeType::Agent
    );
    let serialized = serde_json::to_value(&snapshot).unwrap();
    assert_eq!(serialized["scopes"][0]["scope_id"], root_uuid.to_string());
}
