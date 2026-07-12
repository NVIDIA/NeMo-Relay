// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hermes-specific trace alignment.
//!
//! Hermes reports subagent lifecycle through parent-session hooks, while the child worker also
//! opens its own Hermes session. The parent `subagent_start` payload carries the bridge fields
//! (`child_session_id`, `child_subagent_id`, and `parent_turn_id`) needed to route later child
//! session events back under the parent subagent scope.

use serde_json::{Map, Value, json};

use crate::agents::alignment::{SessionAlias, insert_optional, json_string_at, merge_metadata};
use crate::model::{AgentKind, SessionEvent, SubagentEvent};

#[derive(Debug, Clone)]
pub(crate) struct SubagentContext {
    pub(crate) parent_session_id: String,
    pub(crate) subagent_id: String,
    pub(crate) child_session_id: String,
    parent_turn_id: Option<String>,
    child_role: Option<String>,
    child_goal: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct ExplicitSubagentAlias {
    pub(crate) child_session_id: String,
    pub(crate) alias: SessionAlias,
    pub(crate) scope_metadata: Value,
}

pub(crate) fn subagent_context(event: &SessionEvent) -> Option<SubagentContext> {
    if event.agent_kind != AgentKind::Hermes {
        return None;
    }
    context_from_values(&event.session_id, &event.payload, &event.metadata)
        .filter(|context| context.parent_session_id != event.session_id)
}

pub(crate) fn explicit_subagent_alias(event: &SubagentEvent) -> Option<ExplicitSubagentAlias> {
    if event.agent_kind != AgentKind::Hermes {
        return None;
    }
    let context = context_from_values(&event.session_id, &event.payload, &event.metadata)?;
    if context.child_session_id == event.session_id {
        return None;
    }
    let scope_metadata = scope_metadata(event.metadata.clone(), &context);
    let alias = alias_for_context(&context);
    Some(ExplicitSubagentAlias {
        child_session_id: context.child_session_id,
        alias,
        scope_metadata,
    })
}

pub(crate) fn child_session_id_for_subagent_event(event: &SubagentEvent) -> Option<String> {
    if event.agent_kind != AgentKind::Hermes {
        return None;
    }
    context_from_values(&event.session_id, &event.payload, &event.metadata)
        .map(|context| context.child_session_id)
}

pub(crate) fn augment_subagent_metadata(metadata: Value, context: &SubagentContext) -> Value {
    scope_metadata(metadata, context)
}

pub(crate) fn subagent_start_event(
    event: &SessionEvent,
    context: &SubagentContext,
) -> SubagentEvent {
    SubagentEvent {
        session_id: context.parent_session_id.clone(),
        agent_kind: event.agent_kind,
        event_name: event.event_name.clone(),
        subagent_id: context.subagent_id.clone(),
        payload: event.payload.clone(),
        metadata: scope_metadata(event.metadata.clone(), context),
    }
}

pub(crate) fn alias_for_child_session(
    _child_session_id: String,
    context: &SubagentContext,
) -> SessionAlias {
    alias_for_context(context)
}

pub(crate) fn llm_owner_metadata(scope_metadata: Option<&Value>) -> Value {
    let Some(Value::Object(scope_metadata)) = scope_metadata else {
        return Value::Null;
    };
    let mut metadata = Map::new();
    for key in [
        "thread_source",
        "subagent_id",
        "subagent_session_id",
        "hermes_parent_session_id",
        "hermes_subagent_session_id",
        "hermes_child_subagent_id",
        "hermes_parent_turn_id",
        "child_role",
        "child_goal",
    ] {
        if let Some(value) = scope_metadata.get(key)
            && !value.is_null()
        {
            metadata.insert(key.to_string(), value.clone());
        }
    }
    if metadata.is_empty() {
        Value::Null
    } else {
        Value::Object(metadata)
    }
}

fn context_from_values(
    default_parent_session_id: &str,
    payload: &Value,
    metadata: &Value,
) -> Option<SubagentContext> {
    let child_session_id = child_session_id(payload).or_else(|| child_session_id(metadata))?;
    let parent_session_id = parent_session_id(payload)
        .or_else(|| parent_session_id(metadata))
        .unwrap_or_else(|| default_parent_session_id.to_string());
    let subagent_id = subagent_id(payload)
        .or_else(|| subagent_id(metadata))
        .unwrap_or_else(|| child_session_id.clone());
    Some(SubagentContext {
        parent_session_id,
        subagent_id,
        child_session_id,
        parent_turn_id: optional_string(payload, metadata, "parent_turn_id"),
        child_role: optional_string(payload, metadata, "child_role"),
        child_goal: optional_string(payload, metadata, "child_goal"),
    })
}

fn alias_for_context(context: &SubagentContext) -> SessionAlias {
    SessionAlias::new(
        context.parent_session_id.clone(),
        context.subagent_id.clone(),
        alias_metadata(context),
    )
}

fn alias_metadata(context: &SubagentContext) -> Value {
    Value::Object(base_metadata(context))
}

fn scope_metadata(metadata: Value, context: &SubagentContext) -> Value {
    let mut object = base_metadata(context);
    object.insert("session_id".into(), json!(context.child_session_id.clone()));
    merge_metadata(metadata, Value::Object(object))
}

fn base_metadata(context: &SubagentContext) -> Map<String, Value> {
    let mut object = Map::new();
    object.insert("thread_source".into(), json!("subagent"));
    object.insert("subagent_id".into(), json!(context.subagent_id.clone()));
    object.insert(
        "subagent_session_id".into(),
        json!(context.child_session_id.clone()),
    );
    object.insert(
        "hermes_parent_session_id".into(),
        json!(context.parent_session_id.clone()),
    );
    object.insert(
        "hermes_subagent_session_id".into(),
        json!(context.child_session_id.clone()),
    );
    object.insert(
        "hermes_child_subagent_id".into(),
        json!(context.subagent_id.clone()),
    );
    insert_optional(
        &mut object,
        "hermes_parent_turn_id",
        context.parent_turn_id.as_deref(),
    );
    insert_optional(&mut object, "child_role", context.child_role.as_deref());
    insert_optional(&mut object, "child_goal", context.child_goal.as_deref());
    object
}

fn parent_session_id(value: &Value) -> Option<String> {
    json_string_at(
        value,
        &[
            &["parent_session_id"][..],
            &["parentSessionId"][..],
            &["parent", "session_id"][..],
            &["extra", "parent_session_id"][..],
            &["extra", "parentSessionId"][..],
            &["extra", "parent", "session_id"][..],
        ],
    )
}

fn child_session_id(value: &Value) -> Option<String> {
    json_string_at(
        value,
        &[
            &["child_session_id"][..],
            &["childSessionId"][..],
            &["subagent_session_id"][..],
            &["subagentSessionId"][..],
            &["extra", "child_session_id"][..],
            &["extra", "childSessionId"][..],
            &["extra", "subagent_session_id"][..],
            &["extra", "subagentSessionId"][..],
        ],
    )
}

fn subagent_id(value: &Value) -> Option<String> {
    json_string_at(
        value,
        &[
            &["child_subagent_id"][..],
            &["childSubagentId"][..],
            &["subagent_id"][..],
            &["subagentId"][..],
            &["agent_id"][..],
            &["extra", "child_subagent_id"][..],
            &["extra", "childSubagentId"][..],
            &["extra", "subagent_id"][..],
            &["extra", "subagentId"][..],
            &["extra", "agent_id"][..],
        ],
    )
}

fn optional_string(payload: &Value, metadata: &Value, key: &str) -> Option<String> {
    json_string_at(payload, &[&[key][..], &["extra", key][..]])
        .or_else(|| json_string_at(metadata, &[&[key][..], &["extra", key][..]]))
}
