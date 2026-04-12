// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::{Map, json};
use uuid::{Uuid, Version};

use crate::types::event::Event;
use crate::types::llm::{LLMAttributes, LLMHandle, LLMRequest};
use crate::types::scope::{HandleAttributes, ScopeAttributes, ScopeHandle, ScopeType};
use crate::types::tool::{ToolAttributes, ToolHandle};

#[test]
fn handle_constructors_preserve_supplied_metadata() {
    let parent_uuid = Some(Uuid::now_v7());
    let data = Some(json!({"trace": "abc"}));
    let metadata = Some(json!({"source": "unit-test"}));

    let scope = ScopeHandle::new(
        "agent".to_string(),
        ScopeType::Agent,
        ScopeAttributes::PARALLEL,
        parent_uuid,
        data.clone(),
        metadata.clone(),
    );
    assert_eq!(scope.name, "agent");
    assert_eq!(scope.scope_type, ScopeType::Agent);
    assert_eq!(scope.attributes, ScopeAttributes::PARALLEL);
    assert_eq!(scope.parent_uuid, parent_uuid);
    assert_eq!(scope.data, data);
    assert_eq!(scope.metadata, metadata);
    assert_eq!(scope.uuid.get_version(), Some(Version::SortRand));

    let tool = ToolHandle::new(
        "search".to_string(),
        ToolAttributes::LOCAL,
        parent_uuid,
        Some(json!({"query": "rust"})),
        Some(json!({"kind": "tool"})),
    );
    assert_eq!(tool.name, "search");
    assert_eq!(tool.attributes, ToolAttributes::LOCAL);
    assert_eq!(tool.parent_uuid, parent_uuid);
    assert_eq!(tool.tool_call_id, None);
    assert_eq!(tool.uuid.get_version(), Some(Version::SortRand));

    let llm = LLMHandle::new(
        "planner".to_string(),
        LLMAttributes::STATELESS | LLMAttributes::STREAMING,
        parent_uuid,
        Some(json!({"request": 1})),
        Some(json!({"provider": "test"})),
    );
    assert_eq!(llm.name, "planner");
    assert_eq!(
        llm.attributes,
        LLMAttributes::STATELESS | LLMAttributes::STREAMING
    );
    assert_eq!(llm.parent_uuid, parent_uuid);
    assert_eq!(llm.model_name, None);
    assert_eq!(llm.uuid.get_version(), Some(Version::SortRand));
}

#[test]
fn llm_request_serializes_explicit_headers_and_content() {
    let mut headers = Map::new();
    headers.insert("x-agent".to_string(), json!("planner"));

    let request = LLMRequest {
        headers,
        content: json!({"messages": [{"role": "user", "content": "hi"}]}),
    };

    let encoded = serde_json::to_value(&request).unwrap();
    assert_eq!(encoded["headers"]["x-agent"], json!("planner"));
    assert_eq!(encoded["content"]["messages"][0]["role"], json!("user"));

    let decoded: LLMRequest = serde_json::from_value(encoded).unwrap();
    assert_eq!(decoded.headers.get("x-agent"), Some(&json!("planner")));
}

#[test]
fn event_accessors_cover_scope_tool_llm_and_mark_variants() {
    let parent_uuid = Some(Uuid::now_v7());
    let scope_uuid = Uuid::now_v7();
    let tool_uuid = Uuid::now_v7();
    let llm_uuid = Uuid::now_v7();
    let mark_uuid = Uuid::now_v7();

    let scope_event = Event::scope_start(
        parent_uuid,
        scope_uuid,
        "scope",
        Some(json!({"stage": 1})),
        Some(json!({"region": "us"})),
        ScopeAttributes::RELOCATABLE,
        ScopeType::Function,
    );
    assert_eq!(scope_event.kind(), "ScopeStart");
    assert_eq!(scope_event.parent_uuid(), parent_uuid);
    assert_eq!(scope_event.uuid(), scope_uuid);
    assert_eq!(scope_event.name(), "scope");
    assert_eq!(scope_event.data(), Some(&json!({"stage": 1})));
    assert_eq!(scope_event.metadata(), Some(&json!({"region": "us"})));
    assert_eq!(
        scope_event.attributes(),
        Some(HandleAttributes::Scope(ScopeAttributes::RELOCATABLE))
    );
    assert_eq!(scope_event.scope_type(), Some(ScopeType::Function));
    assert!(scope_event.timestamp().timestamp() > 0);

    let tool_event = Event::tool_end(
        parent_uuid,
        tool_uuid,
        "search",
        None,
        None,
        ToolAttributes::LOCAL,
        Some(json!({"answer": 42})),
        Some("tool-call-1".to_string()),
    );
    assert_eq!(tool_event.kind(), "ToolEnd");
    assert_eq!(
        tool_event.attributes(),
        Some(HandleAttributes::Tool(ToolAttributes::LOCAL))
    );
    assert_eq!(tool_event.output(), Some(&json!({"answer": 42})));
    assert_eq!(tool_event.tool_call_id(), Some("tool-call-1"));
    assert_eq!(tool_event.scope_type(), None);
    assert_eq!(tool_event.model_name(), None);

    let llm_event = Event::llm_start(
        parent_uuid,
        llm_uuid,
        "planner",
        None,
        None,
        LLMAttributes::STREAMING,
        Some(json!({"prompt": "hello"})),
        Some("gpt-test".to_string()),
        None,
    );
    assert_eq!(llm_event.kind(), "LLMStart");
    assert_eq!(
        llm_event.attributes(),
        Some(HandleAttributes::Llm(LLMAttributes::STREAMING))
    );
    assert_eq!(llm_event.input(), Some(&json!({"prompt": "hello"})));
    assert_eq!(llm_event.model_name(), Some("gpt-test"));
    assert_eq!(llm_event.output(), None);

    let mark_event = Event::mark(
        parent_uuid,
        mark_uuid,
        "checkpoint",
        Some(json!({"ok": true})),
        Some(json!({"source": "types"})),
    );
    assert_eq!(mark_event.kind(), "Mark");
    assert_eq!(mark_event.uuid(), mark_uuid);
    assert_eq!(mark_event.attributes(), None);
    assert_eq!(mark_event.scope_type(), None);
    assert_eq!(mark_event.input(), None);
    assert_eq!(mark_event.output(), None);
    assert_eq!(mark_event.tool_call_id(), None);
}
