// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for context in the NeMo Flow core crate.

use std::sync::{Arc, Mutex};

use serde_json::{Map, json};
use uuid::{Uuid, Version};

use crate::api::event::Event;
use crate::api::llm::LlmRequest;
use crate::api::registry::{ExecutionIntercept, GuardrailEntry, Intercept};
use crate::api::runtime::EventSubscriberFn;
use crate::api::runtime::NemoFlowContextState;
use crate::api::runtime::ScopeStack;
use crate::api::runtime::global_context;
use crate::api::scope::{ScopeAttributes, ScopeHandle, ScopeType};
use crate::api::tool::CreateToolHandleParams;
use crate::context::registries::{
    merge_execution_intercept_callables, merge_guardrail_entries, merge_intercept_entries,
};
use crate::registry::SortedRegistry;

#[test]
fn scope_stack_tracks_scope_local_registries_and_subscribers() {
    let mut stack = ScopeStack::new();
    let child = ScopeHandle::builder()
        .name("child".to_string())
        .scope_type(ScopeType::Function)
        .attributes(ScopeAttributes::PARALLEL)
        .parent_uuid(stack.root_uuid())
        .build();
    let child_uuid = child.uuid;
    stack.push(child);

    let subscriber: EventSubscriberFn = Arc::new(|_| {});
    let registries = stack.local_registries_mut(&child_uuid).unwrap();
    registries
        .event_subscribers
        .insert("sub".to_string(), subscriber.clone());
    registries
        .tool_request_intercepts
        .register(
            "tool".to_string(),
            Intercept {
                priority: 10,
                break_chain: false,
                callable: Box::new(|_, value| Ok(value)),
            },
        )
        .unwrap();

    assert_eq!(stack.collect_scope_local_subscribers().len(), 1);
    assert_eq!(
        stack
            .collect_scope_local_registries(|locals| &locals.tool_request_intercepts)
            .len(),
        1
    );
    assert!(stack.scope_registries_get(&child_uuid).is_some());

    let removed = stack.remove(&child_uuid).unwrap();
    assert_eq!(removed.uuid, child_uuid);
    assert!(stack.scope_registries_get(&child_uuid).is_none());
}

#[test]
fn scope_stack_rejects_removing_non_top_or_root_scopes() {
    let mut stack = ScopeStack::new();
    let root_uuid = stack.root_uuid();
    let parent = ScopeHandle::builder()
        .name("parent".to_string())
        .scope_type(ScopeType::Function)
        .parent_uuid(root_uuid)
        .build();
    let parent_uuid = parent.uuid;
    let child = ScopeHandle::builder()
        .name("child".to_string())
        .scope_type(ScopeType::Tool)
        .parent_uuid(parent_uuid)
        .build();

    stack.push(parent);
    stack.push(child);

    let err = stack.remove(&parent_uuid).unwrap_err();
    assert!(err.to_string().contains("not at the top"));

    let top_uuid = stack.top().uuid;
    let removed_child = stack.remove(&top_uuid).unwrap();
    assert_eq!(removed_child.parent_uuid, Some(parent_uuid));

    let removed_parent = stack.remove(&parent_uuid).unwrap();
    assert_eq!(removed_parent.parent_uuid, Some(root_uuid));

    let err = stack.remove(&root_uuid).unwrap_err();
    assert!(err.to_string().contains("root scope cannot be removed"));
}

#[test]
fn merge_helpers_preserve_global_and_scope_local_priority_order() {
    let mut global_guardrails =
        SortedRegistry::new(|entry: &GuardrailEntry<&'static str>| entry.priority);
    global_guardrails
        .register(
            "global".to_string(),
            GuardrailEntry {
                name: "global".to_string(),
                priority: 20,
                guardrail: "global",
            },
        )
        .unwrap();

    let mut local_guardrails =
        SortedRegistry::new(|entry: &GuardrailEntry<&'static str>| entry.priority);
    local_guardrails
        .register(
            "local".to_string(),
            GuardrailEntry {
                name: "local".to_string(),
                priority: 5,
                guardrail: "local",
            },
        )
        .unwrap();

    let local_guardrail_refs = [&local_guardrails];
    let merged_guardrails = merge_guardrail_entries(&global_guardrails, &local_guardrail_refs);
    assert_eq!(
        merged_guardrails
            .iter()
            .map(|entry| entry.guardrail)
            .collect::<Vec<_>>(),
        vec!["local", "global"]
    );

    let mut global_intercepts =
        SortedRegistry::new(|entry: &Intercept<&'static str>| entry.priority);
    global_intercepts
        .register(
            "global".to_string(),
            Intercept {
                priority: 40,
                break_chain: false,
                callable: "global",
            },
        )
        .unwrap();

    let mut local_intercepts =
        SortedRegistry::new(|entry: &Intercept<&'static str>| entry.priority);
    local_intercepts
        .register(
            "local".to_string(),
            Intercept {
                priority: 10,
                break_chain: false,
                callable: "local",
            },
        )
        .unwrap();

    let local_intercept_refs = [&local_intercepts];
    let merged_intercepts = merge_intercept_entries(&global_intercepts, &local_intercept_refs);
    assert_eq!(
        merged_intercepts
            .iter()
            .map(|entry| entry.callable)
            .collect::<Vec<_>>(),
        vec!["local", "global"]
    );

    let mut global_exec =
        SortedRegistry::new(|entry: &ExecutionIntercept<&'static str>| entry.priority);
    global_exec
        .register(
            "global".to_string(),
            ExecutionIntercept {
                priority: 15,
                callable: "global",
            },
        )
        .unwrap();

    let mut local_exec =
        SortedRegistry::new(|entry: &ExecutionIntercept<&'static str>| entry.priority);
    local_exec
        .register(
            "local".to_string(),
            ExecutionIntercept {
                priority: 1,
                callable: "local",
            },
        )
        .unwrap();

    let merged_exec = merge_execution_intercept_callables(&global_exec, &[&local_exec]);
    assert_eq!(merged_exec, vec![("local", 1), ("global", 15)]);
}

#[test]
fn context_state_supports_extensions_events_and_builders() {
    let mut state = NemoFlowContextState::new();
    let key = format!("ext-{}", Uuid::now_v7());
    state.set_extension(&key, vec![1_u32, 2]);
    state.get_extension_mut::<Vec<u32>>(&key).unwrap().push(3);
    assert_eq!(
        state.get_extension::<Vec<u32>>(&key).unwrap(),
        &vec![1, 2, 3]
    );
    assert!(state.remove_extension(&key));
    assert!(state.get_extension::<Vec<u32>>(&key).is_none());

    let scope = state.create_scope_handle(
        crate::api::scope::CreateScopeHandleParams::builder()
            .name("agent")
            .scope_type(ScopeType::Agent)
            .attributes(ScopeAttributes::RELOCATABLE)
            .data(json!({"phase": "start"}))
            .metadata(json!({"trace": "abc"}))
            .build(),
    );
    let scope_start = state.build_scope_start_event(&scope, Some(json!({"step": 1})));
    assert_eq!(scope_start.kind(), "scope");
    assert_eq!(scope_start.name(), "agent");
    assert_eq!(scope.uuid.get_version(), Some(Version::SortRand));

    let mut tool = state.create_tool_handle(
        CreateToolHandleParams::builder()
            .name("search")
            .parent_uuid(scope.uuid)
            .attributes(crate::api::tool::ToolAttributes::REMOTE)
            .data(json!({"base": true}))
            .metadata(json!({"m": 1}))
            .tool_call_id("tool-1")
            .build(),
    );
    tool.tool_call_id = Some("tool-1".to_string());
    let tool_end =
        state.end_tool_handle(&tool, Some(json!({"extra": true})), Some(json!({"m": 2})));
    assert_eq!(tool_end.output(), Some(&json!({"extra": true})));
    assert_eq!(tool_end.tool_call_id(), Some("tool-1"));
    assert_eq!(tool_end.data(), Some(&json!({"extra": true})));
    assert_eq!(tool_end.metadata(), Some(&json!({"m": 2})));

    let request = LlmRequest {
        headers: Map::new(),
        content: json!({"messages": []}),
    };
    let sanitized = state.llm_sanitize_request_chain(request.clone(), &[]);
    assert!(sanitized.headers.is_empty());

    let events = Arc::new(Mutex::new(Vec::<String>::new()));
    let subscriber_events = events.clone();
    state.event_subscribers.insert(
        "capture".to_string(),
        Arc::new(move |event: &Event| {
            subscriber_events
                .lock()
                .unwrap()
                .push(event.kind().to_string());
        }),
    );
    let event = state.create_event(crate::api::event::MarkEvent::new(
        crate::api::event::BaseEvent::builder().name("mark").build(),
        None,
        None,
    ));
    assert_eq!(event.uuid().get_version(), Some(Version::SortRand));
    let subscribers = state.collect_event_subscribers(&[]);
    NemoFlowContextState::emit_event(&event, &subscribers);
    assert_eq!(events.lock().unwrap().as_slice(), ["mark"]);
}

#[test]
fn global_context_is_a_singleton_handle() {
    let first = global_context();
    let second = global_context();
    assert!(Arc::ptr_eq(&first, &second));
}
