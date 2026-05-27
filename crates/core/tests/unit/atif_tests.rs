// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for atif in the NeMo Relay core crate.

use super::*;
use crate::api::event::{
    BaseEvent, CategoryProfile, Event, EventCategory, MarkEvent, ScopeCategory, ScopeEvent,
    llm_attributes_to_strings, scope_attributes_to_strings, tool_attributes_to_strings,
};
use crate::api::llm::LlmAttributes;
use crate::api::scope::{HandleAttributes, ScopeAttributes, ScopeType};
use crate::api::tool::ToolAttributes;
use serde_json::json;
use std::collections::HashSet;

#[derive(Debug, Clone, Copy)]
enum EventType {
    Start,
    End,
    Mark,
}

struct TestEventBuilder {
    uuid: Uuid,
    event_type: EventType,
    parent_uuid: Option<Uuid>,
    name: String,
    data: Option<serde_json::Value>,
    metadata: Option<serde_json::Value>,
    attributes: Option<HandleAttributes>,
    scope_type: Option<ScopeType>,
    input: Option<serde_json::Value>,
    output: Option<serde_json::Value>,
    model_name: Option<String>,
    tool_call_id: Option<String>,
}

impl TestEventBuilder {
    fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    fn parent_uuid(mut self, parent_uuid: Uuid) -> Self {
        self.parent_uuid = Some(parent_uuid);
        self
    }

    fn data(mut self, data: serde_json::Value) -> Self {
        self.data = Some(data);
        self
    }

    fn metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    fn scope_type(mut self, scope_type: ScopeType) -> Self {
        self.scope_type = Some(scope_type);
        self
    }

    fn input(mut self, input: serde_json::Value) -> Self {
        self.input = Some(input);
        self
    }

    fn output(mut self, output: serde_json::Value) -> Self {
        self.output = Some(output);
        self
    }

    fn model_name(mut self, model_name: impl Into<String>) -> Self {
        self.model_name = Some(model_name.into());
        self
    }

    fn tool_call_id(mut self, tool_call_id: impl Into<String>) -> Self {
        self.tool_call_id = Some(tool_call_id.into());
        self
    }

    fn build(self) -> Event {
        match (self.event_type, self.scope_type) {
            (EventType::Mark, _) => Event::Mark(MarkEvent::new(
                BaseEvent::builder()
                    .parent_uuid_opt(self.parent_uuid)
                    .uuid(self.uuid)
                    .name(&(self.name))
                    .data_opt(self.data)
                    .metadata_opt(self.metadata)
                    .build(),
                None,
                None,
            )),
            (EventType::Start, Some(ScopeType::Tool)) => Event::Scope(ScopeEvent::new(
                BaseEvent::builder()
                    .parent_uuid_opt(self.parent_uuid)
                    .uuid(self.uuid)
                    .name(&(self.name))
                    .data_opt(self.input.or(self.data))
                    .metadata_opt(self.metadata)
                    .build(),
                ScopeCategory::Start,
                tool_attributes_to_strings(match self.attributes {
                    Some(HandleAttributes::Tool(attributes)) => attributes,
                    _ => ToolAttributes::empty(),
                }),
                EventCategory::tool(),
                Some(
                    CategoryProfile::builder()
                        .tool_call_id_opt(self.tool_call_id)
                        .build(),
                ),
            )),
            (EventType::End, Some(ScopeType::Tool)) => Event::Scope(ScopeEvent::new(
                BaseEvent::builder()
                    .parent_uuid_opt(self.parent_uuid)
                    .uuid(self.uuid)
                    .name(&(self.name))
                    .data_opt(self.output.or(self.data))
                    .metadata_opt(self.metadata)
                    .build(),
                ScopeCategory::End,
                tool_attributes_to_strings(match self.attributes {
                    Some(HandleAttributes::Tool(attributes)) => attributes,
                    _ => ToolAttributes::empty(),
                }),
                EventCategory::tool(),
                Some(
                    CategoryProfile::builder()
                        .tool_call_id_opt(self.tool_call_id)
                        .build(),
                ),
            )),
            (EventType::Start, Some(ScopeType::Llm)) => Event::Scope(ScopeEvent::new(
                BaseEvent::builder()
                    .parent_uuid_opt(self.parent_uuid)
                    .uuid(self.uuid)
                    .name(&(self.name))
                    .data_opt(self.input.or(self.data))
                    .metadata_opt(self.metadata)
                    .build(),
                ScopeCategory::Start,
                llm_attributes_to_strings(match self.attributes {
                    Some(HandleAttributes::Llm(attributes)) => attributes,
                    _ => LlmAttributes::empty(),
                }),
                EventCategory::llm(),
                Some(
                    CategoryProfile::builder()
                        .model_name_opt(self.model_name)
                        .build(),
                ),
            )),
            (EventType::End, Some(ScopeType::Llm)) => Event::Scope(ScopeEvent::new(
                BaseEvent::builder()
                    .parent_uuid_opt(self.parent_uuid)
                    .uuid(self.uuid)
                    .name(&(self.name))
                    .data_opt(self.output.or(self.data))
                    .metadata_opt(self.metadata)
                    .build(),
                ScopeCategory::End,
                llm_attributes_to_strings(match self.attributes {
                    Some(HandleAttributes::Llm(attributes)) => attributes,
                    _ => LlmAttributes::empty(),
                }),
                EventCategory::llm(),
                Some(
                    CategoryProfile::builder()
                        .model_name_opt(self.model_name)
                        .build(),
                ),
            )),
            (EventType::Start, Some(scope_type)) => Event::Scope(ScopeEvent::new(
                BaseEvent::builder()
                    .parent_uuid_opt(self.parent_uuid)
                    .uuid(self.uuid)
                    .name(&(self.name))
                    .data_opt(self.input.or(self.data))
                    .metadata_opt(self.metadata)
                    .build(),
                ScopeCategory::Start,
                scope_attributes_to_strings(match self.attributes {
                    Some(HandleAttributes::Scope(attributes)) => attributes,
                    _ => ScopeAttributes::empty(),
                }),
                EventCategory::from(scope_type),
                None,
            )),
            (EventType::End, Some(scope_type)) => Event::Scope(ScopeEvent::new(
                BaseEvent::builder()
                    .parent_uuid_opt(self.parent_uuid)
                    .uuid(self.uuid)
                    .name(&(self.name))
                    .data_opt(self.output.or(self.data))
                    .metadata_opt(self.metadata)
                    .build(),
                ScopeCategory::End,
                scope_attributes_to_strings(match self.attributes {
                    Some(HandleAttributes::Scope(attributes)) => attributes,
                    _ => ScopeAttributes::empty(),
                }),
                EventCategory::from(scope_type),
                None,
            )),
            (event_type, None) => panic!("missing scope_type for {event_type:?} event"),
        }
    }
}

fn event_builder(uuid: Uuid, event_type: EventType) -> TestEventBuilder {
    TestEventBuilder {
        uuid,
        event_type,
        parent_uuid: None,
        name: String::new(),
        data: None,
        metadata: None,
        attributes: None,
        scope_type: None,
        input: None,
        output: None,
        model_name: None,
        tool_call_id: None,
    }
}

fn set_event_timestamp(event: &mut Event, timestamp: chrono::DateTime<chrono::Utc>) {
    match event {
        Event::Scope(inner) => inner.base.timestamp = timestamp,
        Event::Mark(inner) => inner.base.timestamp = timestamp,
    }
}

fn base_timestamp() -> chrono::DateTime<chrono::Utc> {
    chrono::DateTime::parse_from_rfc3339("2026-01-01T00:00:00Z")
        .unwrap()
        .with_timezone(&chrono::Utc)
}

fn make_agent_info() -> AtifAgentInfo {
    AtifAgentInfo {
        name: "test-agent".to_string(),
        version: "1.0.0".to_string(),
        model_name: None,
        tool_definitions: None,
        extra: None,
    }
}

fn assert_atif_v17_shape(trajectory: &AtifTrajectory) {
    assert_eq!(trajectory.schema_version, ATIF_SCHEMA_VERSION);
    assert!(!trajectory.session_id.is_empty());
    let embedded_child_ids: HashSet<&str> = trajectory
        .subagent_trajectories
        .as_deref()
        .unwrap_or_default()
        .iter()
        .filter_map(|child| child.trajectory_id.as_deref())
        .collect();

    for (index, step) in trajectory.steps.iter().enumerate() {
        assert_eq!(step.step_id, index + 1);
        assert!(
            matches!(step.source.as_str(), "system" | "user" | "agent"),
            "invalid source {}",
            step.source
        );
        assert_atif_message_value(&step.message);
        if let Some(llm_call_count) = step.llm_call_count {
            assert_eq!(step.source, "agent");
            if llm_call_count == 0 {
                assert!(step.metrics.is_none());
                assert!(step.reasoning_content.is_none());
            }
        }
        let step_tool_call_ids: HashSet<&str> = step
            .tool_calls
            .as_deref()
            .unwrap_or_default()
            .iter()
            .map(|tool_call| {
                assert!(!tool_call.tool_call_id.is_empty());
                assert!(!tool_call.function_name.is_empty());
                assert!(tool_call.arguments.is_object());
                tool_call.tool_call_id.as_str()
            })
            .collect();
        if let Some(observation) = &step.observation {
            for result in &observation.results {
                if let Some(content) = &result.content {
                    assert_atif_message_value(content);
                }
                if let Some(source_call_id) = result.source_call_id.as_deref() {
                    assert!(
                        step_tool_call_ids.contains(source_call_id),
                        "unmatched source_call_id {source_call_id}"
                    );
                }
                for reference in result
                    .subagent_trajectory_ref
                    .as_deref()
                    .unwrap_or_default()
                {
                    if let Some(trajectory_id) = reference.trajectory_id.as_deref()
                        && trajectory.subagent_trajectories.is_some()
                    {
                        assert!(
                            embedded_child_ids.contains(trajectory_id),
                            "unresolved embedded subagent reference {trajectory_id}"
                        );
                    }
                }
            }
        }
    }

    for child in trajectory
        .subagent_trajectories
        .as_deref()
        .unwrap_or_default()
    {
        assert_atif_v17_shape(child);
    }
}

fn assert_atif_message_value(value: &serde_json::Value) {
    if value.is_string() {
        return;
    }
    if let Some(parts) = value.as_array()
        && parts.iter().all(is_atif_content_part)
    {
        return;
    }
    panic!("ATIF message/content must be a string or content-part array: {value}");
}

fn is_atif_content_part(part: &serde_json::Value) -> bool {
    let Some(object) = part.as_object() else {
        return false;
    };
    match object.get("type").and_then(serde_json::Value::as_str) {
        Some("text") => object
            .get("text")
            .and_then(serde_json::Value::as_str)
            .is_some(),
        Some("image") => is_atif_image_source(object.get("source")),
        _ => false,
    }
}

fn is_atif_image_source(value: Option<&serde_json::Value>) -> bool {
    let Some(source) = value.and_then(serde_json::Value::as_object) else {
        return false;
    };
    matches!(
        source.get("media_type").and_then(serde_json::Value::as_str),
        Some("image/jpeg" | "image/png" | "image/gif" | "image/webp")
    ) && source
        .get("path")
        .and_then(serde_json::Value::as_str)
        .is_some()
}

#[test]
fn test_exporter_empty() {
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let trajectory = exporter.export();

    assert_eq!(trajectory.schema_version, ATIF_SCHEMA_VERSION);
    assert_eq!(trajectory.session_id, "session-1");
    assert_eq!(trajectory.agent.name, "test-agent");
    assert!(trajectory.steps.is_empty());
    // final_metrics is always Some now — carries total_steps even for empty trajectories
    let fm = trajectory.final_metrics.as_ref().unwrap();
    assert_eq!(fm.total_steps, Some(0));
    assert!(fm.total_prompt_tokens.is_none());
}

#[test]
fn test_exporter_schema_version() {
    assert_eq!(ATIF_SCHEMA_VERSION, "ATIF-v1.7");
}

#[test]
fn test_exporter_tool_lifecycle() {
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let tool_uuid = Uuid::now_v7();

    // Simulate tool start (should be SKIPPED — tool_calls come from LLM End)
    let start = event_builder(tool_uuid, EventType::Start)
        .name("web_search")
        .scope_type(ScopeType::Tool)
        .input(json!({"query": "test"}))
        .tool_call_id("call_123")
        .build();

    // Simulate tool end
    let end = event_builder(tool_uuid, EventType::End)
        .name("web_search")
        .scope_type(ScopeType::Tool)
        .output(json!({"results": ["result1"]}))
        .tool_call_id("call_123")
        .build();

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.push(start);
        state.events.push(end);
    }

    let trajectory = exporter.export();
    // Tool Start is skipped, only the observation step remains
    assert_eq!(trajectory.steps.len(), 1);

    let step1 = &trajectory.steps[0];
    assert_eq!(step1.step_id, 1);
    assert_eq!(step1.source, "system");
    let obs = step1.observation.as_ref().unwrap();
    assert_eq!(obs.results.len(), 1);
    assert_eq!(obs.results[0].source_call_id, None);
    let content = obs.results[0].content.as_ref().unwrap().as_str().unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(content).unwrap(),
        json!({"results": ["result1"]})
    );
    assert_eq!(
        obs.results[0].extra.as_ref().unwrap()["event_name"],
        json!("web_search")
    );
}

#[test]
fn test_exporter_llm_lifecycle() {
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let llm_uuid = Uuid::now_v7();

    // Input wrapped in LlmRequest envelope — should be unwrapped.
    let start = event_builder(llm_uuid, EventType::Start)
        .name("gpt-4")
        .scope_type(ScopeType::Llm)
        .input(json!({
            "content": {
                "messages": [{"role": "user", "content": "hello"}],
                "temperature": 0.1,
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": { "type": "string" }
                            }
                        }
                    }
                }]
            },
            "headers": {}
        }))
        .model_name("gpt-4")
        .build();

    // Output with content, token_usage, and tool_calls.
    let end = event_builder(llm_uuid, EventType::End)
        .name("gpt-4")
        .scope_type(ScopeType::Llm)
        .output(json!({
            "content": "Hi there!",
            "role": "assistant",
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            },
            "tool_calls": []
        }))
        .model_name("gpt-4")
        .build();

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.push(start);
        state.events.push(end);
    }

    let trajectory = exporter.export();
    assert_eq!(trajectory.steps.len(), 2);

    // First step: user (LLM start — unwrapped LlmRequest, then messages extracted)
    let step1 = &trajectory.steps[0];
    assert_eq!(step1.step_id, 1);
    assert_eq!(step1.source, "user");
    assert_eq!(step1.message, json!("hello"));
    assert_eq!(step1.model_name, None);
    let extra: AtifStepExtra = serde_json::from_value(step1.extra.clone().unwrap()).unwrap();
    let llm_request = extra.llm_request.unwrap();
    assert_eq!(llm_request["temperature"], json!(0.1));
    assert_eq!(
        llm_request["tools"][0]["function"]["name"],
        json!("read_file")
    );

    // Second step: agent (LLM end with extracted content + metrics)
    let step2 = &trajectory.steps[1];
    assert_eq!(step2.step_id, 2);
    assert_eq!(step2.source, "agent");
    assert_eq!(step2.message, json!("Hi there!"));
    assert_eq!(step2.model_name, Some("gpt-4".to_string()));
    assert_eq!(step2.llm_call_count, Some(1));
    // Metrics extracted from token_usage
    let metrics = step2.metrics.as_ref().unwrap();
    assert_eq!(metrics.prompt_tokens, Some(10));
    assert_eq!(metrics.completion_tokens, Some(20));
    // Empty tool_calls should not produce AtifToolCall entries
    assert!(step2.tool_calls.is_none());

    // final_metrics should aggregate using total_ prefixed fields (AtifFinalMetrics)
    let fm = trajectory.final_metrics.as_ref().unwrap();
    assert_eq!(fm.total_prompt_tokens, Some(10));
    assert_eq!(fm.total_completion_tokens, Some(20));
    assert_eq!(fm.total_steps, Some(2));
}

#[test]
fn test_extract_metrics_supports_provider_usage_payloads() {
    let openai_metrics = extract_metrics(&json!({
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "prompt_tokens_details": {
                "cached_tokens": 4
            }
        }
    }))
    .unwrap();
    assert_eq!(openai_metrics.prompt_tokens, Some(10));
    assert_eq!(openai_metrics.completion_tokens, Some(20));
    assert_eq!(openai_metrics.cached_tokens, Some(4));
    assert_eq!(
        openai_metrics.extra.as_ref().unwrap()["total_tokens"],
        json!(30)
    );

    let anthropic_metrics = extract_metrics(&json!({
        "usage": {
            "input_tokens": 11,
            "output_tokens": 22,
            "cache_read_input_tokens": 3,
            "cache_creation_input_tokens": 5
        }
    }))
    .unwrap();
    assert_eq!(anthropic_metrics.prompt_tokens, Some(11));
    assert_eq!(anthropic_metrics.completion_tokens, Some(22));
    assert_eq!(anthropic_metrics.cached_tokens, Some(8));
}

#[test]
fn test_exporter_llm_lifecycle_plain_input() {
    // Input without LlmRequest envelope — passed through unchanged.
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let llm_uuid = Uuid::now_v7();

    let start = event_builder(llm_uuid, EventType::Start)
        .name("gpt-4")
        .scope_type(ScopeType::Llm)
        .input(json!({"messages": [{"role": "user", "content": "hello"}]}))
        .model_name("gpt-4")
        .build();

    let end = event_builder(llm_uuid, EventType::End)
        .name("gpt-4")
        .scope_type(ScopeType::Llm)
        .output(json!("simple string response"))
        .model_name("gpt-4")
        .build();

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.push(start);
        state.events.push(end);
    }

    let trajectory = exporter.export();
    assert_eq!(trajectory.steps.len(), 2);

    assert_eq!(trajectory.steps[0].message, json!("hello"));
    // Non-object output is passed through as-is
    assert_eq!(trajectory.steps[1].message, json!("simple string response"));
    assert!(trajectory.steps[1].metrics.is_none());
    // No token metrics on any step — token totals are None, but total_steps is still set
    let fm = trajectory.final_metrics.as_ref().unwrap();
    assert!(fm.total_prompt_tokens.is_none());
    assert_eq!(fm.total_steps, Some(2));
}

#[test]
fn test_exporter_openai_responses_lifecycle_extracts_messages() {
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let llm_uuid = Uuid::now_v7();

    let start = event_builder(llm_uuid, EventType::Start)
        .name("gpt-test-model")
        .scope_type(ScopeType::Llm)
        .input(json!({
            "input": "Summarize the Codex worker result.",
            "model": "gpt-test-model",
            "prompt_cache_key": "codex-child-thread"
        }))
        .model_name("gpt-test-model")
        .build();

    let end = event_builder(llm_uuid, EventType::End)
        .name("gpt-test-model")
        .scope_type(ScopeType::Llm)
        .output(json!({
            "id": "resp_1",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Codex worker summary complete."
                        }
                    ]
                }
            ],
            "usage": {
                "input_tokens": 11,
                "output_tokens": 7,
                "total_tokens": 18
            }
        }))
        .model_name("gpt-test-model")
        .build();

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.push(start);
        state.events.push(end);
    }

    let trajectory = exporter.export();
    assert_eq!(trajectory.steps.len(), 2);

    let user_step = &trajectory.steps[0];
    assert_eq!(user_step.source, "user");
    assert_eq!(
        user_step.message,
        json!("Summarize the Codex worker result.")
    );
    let user_extra: AtifStepExtra =
        serde_json::from_value(user_step.extra.clone().unwrap()).unwrap();
    let llm_request = user_extra.llm_request.unwrap();
    assert_eq!(llm_request["prompt_cache_key"], json!("codex-child-thread"));

    let agent_step = &trajectory.steps[1];
    assert_eq!(agent_step.source, "agent");
    assert_eq!(agent_step.message, json!("Codex worker summary complete."));
    assert_eq!(agent_step.model_name, Some("gpt-test-model".to_string()));
    let metrics = agent_step.metrics.as_ref().unwrap();
    assert_eq!(metrics.prompt_tokens, Some(11));
    assert_eq!(metrics.completion_tokens, Some(7));
    let agent_extra: AtifStepExtra =
        serde_json::from_value(agent_step.extra.clone().unwrap()).unwrap();
    assert_eq!(agent_extra.llm_response.unwrap()["id"], json!("resp_1"));
}

#[test]
fn test_openai_responses_input_extracts_latest_user_content_block() {
    let message = extract_user_messages(&json!({
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    { "type": "input_text", "text": "Initial task" }
                ]
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    { "type": "output_text", "text": "Intermediate answer" }
                ]
            },
            {
                "type": "message",
                "role": "user",
                "content": [
                    { "type": "input_text", "text": "Follow-up task" }
                ]
            }
        ]
    }));

    assert_eq!(message, json!("Follow-up task"));
}

#[test]
fn test_exporter_openai_responses_function_calls_promoted_and_correlated() {
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let llm_uuid = Uuid::now_v7();
    let tool_uuid = Uuid::now_v7();
    let base = base_timestamp();

    let mut llm_end = event_builder(llm_uuid, EventType::End)
        .name("gpt-test-model")
        .scope_type(ScopeType::Llm)
        .output(json!({
            "id": "resp_1",
            "status": "completed",
            "output": [
                {
                    "type": "reasoning",
                    "summary": []
                },
                {
                    "type": "message",
                    "id": "msg_1",
                    "name": "not_a_tool",
                    "arguments": "{\"ignored\":true}"
                },
                {
                    "type": "function_call",
                    "call_id": "call_terminal_1",
                    "name": "terminal",
                    "arguments": "{\"command\":\"pwd\"}",
                    "status": "completed"
                }
            ]
        }))
        .model_name("gpt-test-model")
        .build();
    let mut tool_end = event_builder(tool_uuid, EventType::End)
        .name("terminal")
        .scope_type(ScopeType::Tool)
        .parent_uuid(llm_uuid)
        .tool_call_id("call_terminal_1")
        .output(json!({"stdout": "/workspace"}))
        .build();

    for (offset, event) in [&mut llm_end, &mut tool_end].into_iter().enumerate() {
        set_event_timestamp(event, base + chrono::Duration::seconds(offset as i64));
    }

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.extend([llm_end, tool_end]);
    }

    let trajectory = exporter.export();
    assert_atif_v17_shape(&trajectory);
    assert_eq!(trajectory.steps.len(), 1);

    let agent_step = &trajectory.steps[0];
    assert_eq!(agent_step.source, "agent");
    let tool_calls = agent_step.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 1);
    let tool_call = &tool_calls[0];
    assert_eq!(tool_call.tool_call_id, "call_terminal_1");
    assert_eq!(tool_call.function_name, "terminal");
    assert_eq!(tool_call.arguments, json!({"command": "pwd"}));

    let result = &agent_step.observation.as_ref().unwrap().results[0];
    assert_eq!(result.source_call_id.as_deref(), Some("call_terminal_1"));
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(
            result.content.as_ref().unwrap().as_str().unwrap()
        )
        .unwrap(),
        json!({"stdout": "/workspace"})
    );
}

#[test]
fn test_exporter_llm_tool_calls_promoted() {
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let llm_uuid = Uuid::now_v7();

    let end = event_builder(llm_uuid, EventType::End)
        .name("gpt-4")
        .scope_type(ScopeType::Llm)
        .output(json!({
            "content": null,
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "provider_data": {"trace_id": "provider-trace-1"},
                    "function": {
                        "name": "search",
                        "arguments": "{\"q\": \"test\"}",
                        "schema_version": "v1"
                    }
                }
            ]
        }))
        .build();

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.push(end);
    }

    let trajectory = exporter.export();
    assert_eq!(trajectory.steps.len(), 1);
    let step = &trajectory.steps[0];

    // tool_calls promoted from response body, string arguments parsed as JSON
    let tc = step.tool_calls.as_ref().unwrap();
    assert_eq!(tc.len(), 1);
    assert_eq!(tc[0].tool_call_id, "call_abc");
    assert_eq!(tc[0].function_name, "search");
    assert_eq!(tc[0].arguments, json!({"q": "test"}));
    assert_eq!(
        tc[0].extra.as_ref().unwrap()["provider_data"]["trace_id"],
        json!("provider-trace-1")
    );
    assert_eq!(
        tc[0].extra.as_ref().unwrap()["function"]["schema_version"],
        json!("v1")
    );

    assert_eq!(step.message, json!(""));
    assert_eq!(step.llm_call_count, Some(1));
    let extra: AtifStepExtra = serde_json::from_value(step.extra.clone().unwrap()).unwrap();
    assert_eq!(extra.llm_response.unwrap()["role"], json!("assistant"));
}

#[test]
fn test_exporter_hermes_wrapper_payload_is_atif_v17_compatible() {
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let llm_uuid = Uuid::now_v7();
    let tool_uuid = Uuid::now_v7();

    let llm_start = event_builder(llm_uuid, EventType::Start)
        .name("qwen3.6:35b")
        .scope_type(ScopeType::Llm)
        .input(json!({
            "messages": [{"role": "user", "content": "Run a terminal command"}],
            "model": "qwen3.6:35b"
        }))
        .model_name("qwen3.6:35b")
        .build();

    let llm_end = event_builder(llm_uuid, EventType::End)
        .name("qwen3.6:35b")
        .scope_type(ScopeType::Llm)
        .output(json!({
            "assistant_message": {
                "content": null,
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_terminal_1",
                        "name": "terminal",
                        "arguments": "{\"command\":\"printf hi\"}",
                        "provider_data": {"trace_id": "provider-trace"}
                    }
                ]
            },
            "raw_response": {
                "choices": [
                    {
                        "message": {
                            "content": null,
                            "tool_calls": [
                                {
                                    "id": "call_terminal_1",
                                    "type": "function",
                                    "function": {
                                        "name": "terminal",
                                        "arguments": "{\"command\":\"printf hi\"}"
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
            "usage": {
                "prompt_tokens": 14,
                "completion_tokens": 5
            }
        }))
        .model_name("qwen3.6:35b")
        .build();

    let tool_end = event_builder(tool_uuid, EventType::End)
        .name("terminal")
        .scope_type(ScopeType::Tool)
        .output(json!({
            "stdout": "hi",
            "exit_code": 0
        }))
        .tool_call_id("call_terminal_1")
        .build();

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.extend([llm_start, llm_end, tool_end]);
    }

    let trajectory = exporter.export();
    assert_atif_v17_shape(&trajectory);
    assert_eq!(trajectory.steps.len(), 2);
    assert_eq!(trajectory.steps[0].message, json!("Run a terminal command"));

    let agent_step = &trajectory.steps[1];
    assert_eq!(agent_step.message, json!(""));
    assert_eq!(agent_step.llm_call_count, Some(1));
    let tool_call = &agent_step.tool_calls.as_ref().unwrap()[0];
    assert_eq!(tool_call.tool_call_id, "call_terminal_1");
    assert_eq!(tool_call.function_name, "terminal");
    assert_eq!(tool_call.arguments, json!({"command": "printf hi"}));
    assert_eq!(
        tool_call.extra.as_ref().unwrap()["provider_data"]["trace_id"],
        json!("provider-trace")
    );

    let extra: AtifStepExtra = serde_json::from_value(agent_step.extra.clone().unwrap()).unwrap();
    assert_eq!(
        extra.llm_response.unwrap()["assistant_message"]["tool_calls"][0]["id"],
        json!("call_terminal_1")
    );

    let observation = agent_step.observation.as_ref().unwrap();
    assert_eq!(
        observation.results[0].source_call_id,
        Some("call_terminal_1".to_string())
    );
    let content = observation.results[0]
        .content
        .as_ref()
        .unwrap()
        .as_str()
        .unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(content).unwrap(),
        json!({"stdout": "hi", "exit_code": 0})
    );
}

#[test]
fn test_exporter_full_pipeline() {
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let scope_uuid = Uuid::now_v7();
    let llm_uuid = Uuid::now_v7();
    let tool_uuid = Uuid::now_v7();

    // Scope start (should be skipped)
    let scope_start = event_builder(scope_uuid, EventType::Start)
        .name("agent")
        .scope_type(ScopeType::Agent)
        .build();

    // LLM start/end
    let llm_start = event_builder(llm_uuid, EventType::Start)
        .scope_type(ScopeType::Llm)
        .input(json!({"prompt": "What is 2+2?"}))
        .build();
    let llm_end = event_builder(llm_uuid, EventType::End)
        .scope_type(ScopeType::Llm)
        .output(json!({"answer": "4"}))
        .build();

    // Tool start/end
    let tool_start = event_builder(tool_uuid, EventType::Start)
        .name("calculator")
        .scope_type(ScopeType::Tool)
        .input(json!({"expr": "2+2"}))
        .tool_call_id("call_1")
        .build();
    let tool_end = event_builder(tool_uuid, EventType::End)
        .name("calculator")
        .scope_type(ScopeType::Tool)
        .output(json!(4))
        .tool_call_id("call_1")
        .build();

    // Scope end (should be skipped)
    let scope_end = event_builder(scope_uuid, EventType::End)
        .name("agent")
        .scope_type(ScopeType::Agent)
        .build();

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.push(scope_start);
        state.events.push(llm_start);
        state.events.push(llm_end);
        state.events.push(tool_start);
        state.events.push(tool_end);
        state.events.push(scope_end);
    }

    let trajectory = exporter.export();
    // Scope events are skipped and the tool observation attaches to the agent step.
    assert_eq!(trajectory.steps.len(), 2);

    assert_eq!(trajectory.steps[0].source, "user");
    assert_eq!(trajectory.steps[1].source, "agent");
    assert!(trajectory.steps[1].observation.is_some());

    // Step IDs are 1-based
    for (i, step) in trajectory.steps.iter().enumerate() {
        assert_eq!(step.step_id, i + 1);
    }
}

#[test]
fn test_exporter_tool_call_id_linking() {
    // Tool Start is skipped; the tool_call_id comes from the event's own field.
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let tool_uuid = Uuid::now_v7();

    let start = event_builder(tool_uuid, EventType::Start)
        .name("my_tool")
        .scope_type(ScopeType::Tool)
        .input(json!({"x": 1}))
        .tool_call_id("call_abc")
        .build();

    let end = event_builder(tool_uuid, EventType::End)
        .name("my_tool")
        .scope_type(ScopeType::Tool)
        .output(json!({"y": 2}))
        .tool_call_id("call_abc")
        .build();

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.push(start);
        state.events.push(end);
    }

    let trajectory = exporter.export();
    // Only observation step (Tool Start is skipped)
    assert_eq!(trajectory.steps.len(), 1);
    let obs_result = &trajectory.steps[0].observation.as_ref().unwrap().results[0];
    assert_eq!(obs_result.source_call_id, None);
}

#[test]
fn test_exporter_mark_steps_include_hook_name_and_ancestry() {
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let agent_uuid = Uuid::now_v7();
    let mark_uuid = Uuid::now_v7();

    let agent_start = event_builder(agent_uuid, EventType::Start)
        .name("hermes")
        .scope_type(ScopeType::Agent)
        .build();
    let mark = event_builder(mark_uuid, EventType::Mark)
        .name("subagent_end_without_start")
        .parent_uuid(agent_uuid)
        .data(json!({
            "session_id": "session-1",
            "extra": {
                "subagent_id": "worker-1"
            }
        }))
        .metadata(json!({
            "hook_event_name": "subagent_stop"
        }))
        .build();

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.push(agent_start);
        state.events.push(mark);
    }

    let trajectory = exporter.export();
    assert_eq!(trajectory.steps.len(), 1);

    let step = &trajectory.steps[0];
    assert_eq!(step.source, "system");
    assert_eq!(step.message, json!("subagent_stop"));

    let extra: AtifStepExtra = serde_json::from_value(step.extra.clone().unwrap()).unwrap();
    assert_eq!(
        extra.event_payload.as_ref().unwrap()["extra"]["subagent_id"],
        json!("worker-1")
    );
    assert_eq!(extra.ancestry.function_id, mark_uuid.to_string());
    assert_eq!(extra.ancestry.function_name, "subagent_end_without_start");
    assert_eq!(extra.ancestry.parent_id, Some(agent_uuid.to_string()));
    assert_eq!(extra.ancestry.parent_name, Some("hermes".to_string()));
    assert_eq!(
        extra.invocation.as_ref().unwrap().invocation_id,
        Some(mark_uuid.to_string())
    );
}

#[test]
fn test_exporter_embeds_nested_subagent_trajectory() {
    let root_uuid = Uuid::now_v7();
    let child_uuid = Uuid::now_v7();
    let llm_uuid = Uuid::now_v7();
    let exporter = AtifExporter::new(root_uuid.to_string(), make_agent_info());
    let base = base_timestamp();

    let mut root_start = event_builder(root_uuid, EventType::Start)
        .name("root-agent")
        .scope_type(ScopeType::Agent)
        .build();
    let mut child_start = event_builder(child_uuid, EventType::Start)
        .name("worker-agent")
        .scope_type(ScopeType::Agent)
        .parent_uuid(root_uuid)
        .metadata(json!({
            "session_id": "child-session",
            "nemo_relay_scope_role": "subagent"
        }))
        .build();
    let mut llm_start = event_builder(llm_uuid, EventType::Start)
        .name("worker-llm")
        .scope_type(ScopeType::Llm)
        .parent_uuid(child_uuid)
        .input(json!({"messages": [{"role": "user", "content": "subtask"}]}))
        .build();
    let mut llm_end = event_builder(llm_uuid, EventType::End)
        .name("worker-llm")
        .scope_type(ScopeType::Llm)
        .parent_uuid(child_uuid)
        .output(json!({"content": "done"}))
        .build();
    let mut child_end = event_builder(child_uuid, EventType::End)
        .name("worker-agent")
        .scope_type(ScopeType::Agent)
        .parent_uuid(root_uuid)
        .build();
    let mut root_end = event_builder(root_uuid, EventType::End)
        .name("root-agent")
        .scope_type(ScopeType::Agent)
        .build();

    for (offset, event) in [
        &mut root_start,
        &mut child_start,
        &mut llm_start,
        &mut llm_end,
        &mut child_end,
        &mut root_end,
    ]
    .into_iter()
    .enumerate()
    {
        set_event_timestamp(event, base + chrono::Duration::seconds(offset as i64));
    }

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.extend([
            root_start,
            child_start,
            llm_start,
            llm_end,
            child_end,
            root_end,
        ]);
    }

    let trajectory = exporter.export();
    assert_atif_v17_shape(&trajectory);
    assert_eq!(trajectory.schema_version, "ATIF-v1.7");
    assert_eq!(trajectory.session_id, root_uuid.to_string());
    assert_eq!(trajectory.trajectory_id, Some(root_uuid.to_string()));
    assert_eq!(trajectory.steps.len(), 1);

    let step = &trajectory.steps[0];
    assert_eq!(step.source, "agent");
    assert_eq!(step.llm_call_count, Some(0));
    let dispatch_tool_call_id = format!("subagent:{child_uuid}");
    assert_eq!(
        step.tool_calls.as_ref().unwrap()[0].tool_call_id,
        dispatch_tool_call_id
    );
    assert_eq!(
        step.tool_calls.as_ref().unwrap()[0].function_name,
        "worker-agent"
    );
    let result = &trajectory.steps[0].observation.as_ref().unwrap().results[0];
    assert_eq!(
        result.source_call_id.as_deref(),
        Some(dispatch_tool_call_id.as_str())
    );
    assert!(result.content.is_none());
    let refs = result.subagent_trajectory_ref.as_ref().unwrap();
    assert_eq!(refs[0].trajectory_id, Some(child_uuid.to_string()));
    assert_eq!(refs[0].session_id, Some("child-session".to_string()));

    let subagents = trajectory.subagent_trajectories.as_ref().unwrap();
    assert_eq!(subagents.len(), 1);
    let child = &subagents[0];
    assert_eq!(child.trajectory_id, Some(child_uuid.to_string()));
    assert_eq!(child.session_id, "child-session");
    assert_eq!(child.steps.len(), 2);
    assert_eq!(child.steps[0].source, "user");
    assert_eq!(child.steps[1].source, "agent");

    let serialized = serde_json::to_value(&trajectory).unwrap();
    assert!(serialized["steps"][0]["observation"]["results"][0]["content"].is_null());
}

#[test]
fn test_exporter_attaches_subagent_ref_to_delegating_tool_observation() {
    let root_uuid = Uuid::now_v7();
    let child_uuid = Uuid::now_v7();
    let llm_uuid = Uuid::now_v7();
    let tool_uuid = Uuid::now_v7();
    let child_mark_uuid = Uuid::now_v7();
    let exporter = AtifExporter::new(root_uuid.to_string(), make_agent_info());
    let base = base_timestamp();

    let mut root_start = event_builder(root_uuid, EventType::Start)
        .name("root-agent")
        .scope_type(ScopeType::Agent)
        .build();
    let mut llm_start = event_builder(llm_uuid, EventType::Start)
        .name("root-llm")
        .scope_type(ScopeType::Llm)
        .parent_uuid(root_uuid)
        .input(json!({"messages": [{"role": "user", "content": "delegate"}]}))
        .build();
    let mut llm_end = event_builder(llm_uuid, EventType::End)
        .name("root-llm")
        .scope_type(ScopeType::Llm)
        .parent_uuid(root_uuid)
        .output(json!({
            "content": "launching worker",
            "tool_calls": [{
                "id": "call_delegate",
                "type": "function",
                "function": {
                    "name": "delegate_task",
                    "arguments": "{\"task\":\"subtask\"}"
                }
            }]
        }))
        .build();
    let mut tool_end = event_builder(tool_uuid, EventType::End)
        .name("delegate_task")
        .scope_type(ScopeType::Tool)
        .parent_uuid(root_uuid)
        .tool_call_id("call_delegate")
        .output(json!({"status": "launched"}))
        .build();
    let mut child_start = event_builder(child_uuid, EventType::Start)
        .name("worker-agent")
        .scope_type(ScopeType::Agent)
        .parent_uuid(root_uuid)
        .metadata(json!({
            "session_id": "child-session",
            "tool_call_id": "call_delegate"
        }))
        .build();
    let mut child_mark = event_builder(child_mark_uuid, EventType::Mark)
        .name("worker-started")
        .parent_uuid(child_uuid)
        .data(json!({"status": "started"}))
        .build();
    let mut child_end = event_builder(child_uuid, EventType::End)
        .name("worker-agent")
        .scope_type(ScopeType::Agent)
        .parent_uuid(root_uuid)
        .build();
    let mut root_end = event_builder(root_uuid, EventType::End)
        .name("root-agent")
        .scope_type(ScopeType::Agent)
        .build();

    for (offset, event) in [
        &mut root_start,
        &mut llm_start,
        &mut llm_end,
        &mut tool_end,
        &mut child_start,
        &mut child_mark,
        &mut child_end,
        &mut root_end,
    ]
    .into_iter()
    .enumerate()
    {
        set_event_timestamp(event, base + chrono::Duration::seconds(offset as i64));
    }

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.extend([
            root_start,
            llm_start,
            llm_end,
            tool_end,
            child_start,
            child_mark,
            child_end,
            root_end,
        ]);
    }

    let trajectory = exporter.export();
    assert_atif_v17_shape(&trajectory);
    assert_eq!(trajectory.steps.len(), 2);
    let agent_step = &trajectory.steps[1];
    assert_eq!(agent_step.source, "agent");
    assert_eq!(
        agent_step.tool_calls.as_ref().unwrap()[0].tool_call_id,
        "call_delegate"
    );

    let result = &agent_step.observation.as_ref().unwrap().results[0];
    assert_eq!(result.source_call_id.as_deref(), Some("call_delegate"));
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(
            result.content.as_ref().unwrap().as_str().unwrap()
        )
        .unwrap(),
        json!({"status": "launched"})
    );
    let refs = result.subagent_trajectory_ref.as_ref().unwrap();
    assert_eq!(refs[0].trajectory_id, Some(child_uuid.to_string()));
    assert_eq!(refs[0].session_id, Some("child-session".to_string()));
}

#[test]
fn test_exporter_synthesizes_tool_call_for_active_subagent_dispatch() {
    let root_uuid = Uuid::now_v7();
    let child_uuid = Uuid::now_v7();
    let llm_uuid = Uuid::now_v7();
    let tool_uuid = Uuid::now_v7();
    let child_mark_uuid = Uuid::now_v7();
    let exporter = AtifExporter::new(root_uuid.to_string(), make_agent_info());
    let base = base_timestamp();

    let mut root_start = event_builder(root_uuid, EventType::Start)
        .name("root-agent")
        .scope_type(ScopeType::Agent)
        .build();
    let mut llm_start = event_builder(llm_uuid, EventType::Start)
        .name("root-llm")
        .scope_type(ScopeType::Llm)
        .parent_uuid(root_uuid)
        .input(json!({"messages": [{"role": "user", "content": "delegate"}]}))
        .build();
    let mut llm_end = event_builder(llm_uuid, EventType::End)
        .name("root-llm")
        .scope_type(ScopeType::Llm)
        .parent_uuid(root_uuid)
        .output(json!({"content": "launching worker"}))
        .build();
    let mut tool_start = event_builder(tool_uuid, EventType::Start)
        .name("delegate_task")
        .scope_type(ScopeType::Tool)
        .parent_uuid(root_uuid)
        .input(json!({"task": "subtask"}))
        .build();
    let mut child_start = event_builder(child_uuid, EventType::Start)
        .name("worker-agent")
        .scope_type(ScopeType::Agent)
        .parent_uuid(root_uuid)
        .metadata(json!({"session_id": "child-session"}))
        .build();
    let mut child_mark = event_builder(child_mark_uuid, EventType::Mark)
        .name("worker-started")
        .parent_uuid(child_uuid)
        .data(json!({"status": "started"}))
        .build();
    let mut child_end = event_builder(child_uuid, EventType::End)
        .name("worker-agent")
        .scope_type(ScopeType::Agent)
        .parent_uuid(root_uuid)
        .build();
    let mut tool_end = event_builder(tool_uuid, EventType::End)
        .name("delegate_task")
        .scope_type(ScopeType::Tool)
        .parent_uuid(root_uuid)
        .output(json!({"status": "completed"}))
        .build();
    let mut root_end = event_builder(root_uuid, EventType::End)
        .name("root-agent")
        .scope_type(ScopeType::Agent)
        .build();

    for (offset, event) in [
        &mut root_start,
        &mut llm_start,
        &mut llm_end,
        &mut tool_start,
        &mut child_start,
        &mut child_mark,
        &mut child_end,
        &mut tool_end,
        &mut root_end,
    ]
    .into_iter()
    .enumerate()
    {
        set_event_timestamp(event, base + chrono::Duration::seconds(offset as i64));
    }

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.extend([
            root_start,
            llm_start,
            llm_end,
            tool_start,
            child_start,
            child_mark,
            child_end,
            tool_end,
            root_end,
        ]);
    }

    let trajectory = exporter.export();
    assert_atif_v17_shape(&trajectory);
    assert_eq!(trajectory.steps.len(), 2);
    let agent_step = &trajectory.steps[1];
    assert_eq!(agent_step.source, "agent");
    let tool_call_id = tool_uuid.to_string();
    let tool_call = &agent_step.tool_calls.as_ref().unwrap()[0];
    assert_eq!(tool_call.tool_call_id, tool_call_id);
    assert_eq!(tool_call.function_name, "delegate_task");
    assert_eq!(tool_call.arguments, json!({"task": "subtask"}));

    let result = &agent_step.observation.as_ref().unwrap().results[0];
    assert_eq!(
        result.source_call_id.as_deref(),
        Some(tool_call_id.as_str())
    );
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(
            result.content.as_ref().unwrap().as_str().unwrap()
        )
        .unwrap(),
        json!({"status": "completed"})
    );
    let refs = result.subagent_trajectory_ref.as_ref().unwrap();
    assert_eq!(refs[0].trajectory_id, Some(child_uuid.to_string()));
    assert_eq!(refs[0].session_id, Some("child-session".to_string()));
    assert!(!trajectory.steps.iter().any(|step| {
        step.source == "system"
            && step.observation.as_ref().is_some_and(|observation| {
                observation
                    .results
                    .iter()
                    .any(|result| result.subagent_trajectory_ref.is_some())
            })
    }));
}

#[test]
fn test_exporter_drops_empty_subagent_trajectory_and_parent_ref() {
    let root_uuid = Uuid::now_v7();
    let child_uuid = Uuid::now_v7();
    let exporter = AtifExporter::new(root_uuid.to_string(), make_agent_info());
    let base = base_timestamp();

    let mut root_start = event_builder(root_uuid, EventType::Start)
        .name("root-agent")
        .scope_type(ScopeType::Agent)
        .build();
    let mut child_start = event_builder(child_uuid, EventType::Start)
        .name("worker-agent")
        .scope_type(ScopeType::Agent)
        .parent_uuid(root_uuid)
        .metadata(json!({
            "session_id": "child-session",
            "nemo_relay_scope_role": "subagent"
        }))
        .build();
    let mut child_end = event_builder(child_uuid, EventType::End)
        .name("worker-agent")
        .scope_type(ScopeType::Agent)
        .parent_uuid(root_uuid)
        .build();
    let mut root_end = event_builder(root_uuid, EventType::End)
        .name("root-agent")
        .scope_type(ScopeType::Agent)
        .build();

    for (offset, event) in [
        &mut root_start,
        &mut child_start,
        &mut child_end,
        &mut root_end,
    ]
    .into_iter()
    .enumerate()
    {
        set_event_timestamp(event, base + chrono::Duration::seconds(offset as i64));
    }

    {
        let mut state = exporter.state.lock().unwrap();
        state
            .events
            .extend([root_start, child_start, child_end, root_end]);
    }

    let trajectory = exporter.export();
    assert_atif_v17_shape(&trajectory);
    assert!(trajectory.steps.is_empty());
    assert!(trajectory.subagent_trajectories.is_none());
    let serialized = serde_json::to_value(&trajectory).unwrap();
    assert!(!serialized.to_string().contains("subagent_trajectory_ref"));
}

#[test]
fn test_exporter_renumbers_after_pruning_empty_subagent_ref_step() {
    let root_uuid = Uuid::now_v7();
    let llm_uuid = Uuid::now_v7();
    let child_uuid = Uuid::now_v7();
    let mark_uuid = Uuid::now_v7();
    let exporter = AtifExporter::new(root_uuid.to_string(), make_agent_info());
    let base = base_timestamp();

    let mut root_start = event_builder(root_uuid, EventType::Start)
        .name("root-agent")
        .scope_type(ScopeType::Agent)
        .build();
    let mut llm_end = event_builder(llm_uuid, EventType::End)
        .name("root-llm")
        .scope_type(ScopeType::Llm)
        .parent_uuid(root_uuid)
        .output(json!({"content": "before child"}))
        .build();
    let mut child_start = event_builder(child_uuid, EventType::Start)
        .name("worker-agent")
        .scope_type(ScopeType::Agent)
        .parent_uuid(root_uuid)
        .metadata(json!({
            "session_id": "child-session",
            "nemo_relay_scope_role": "subagent"
        }))
        .build();
    let mut child_end = event_builder(child_uuid, EventType::End)
        .name("worker-agent")
        .scope_type(ScopeType::Agent)
        .parent_uuid(root_uuid)
        .build();
    let mut mark = event_builder(mark_uuid, EventType::Mark)
        .name("root-note")
        .parent_uuid(root_uuid)
        .data(json!({"status": "after child"}))
        .build();
    let mut root_end = event_builder(root_uuid, EventType::End)
        .name("root-agent")
        .scope_type(ScopeType::Agent)
        .build();

    for (offset, event) in [
        &mut root_start,
        &mut llm_end,
        &mut child_start,
        &mut child_end,
        &mut mark,
        &mut root_end,
    ]
    .into_iter()
    .enumerate()
    {
        set_event_timestamp(event, base + chrono::Duration::seconds(offset as i64));
    }

    {
        let mut state = exporter.state.lock().unwrap();
        state
            .events
            .extend([root_start, llm_end, child_start, child_end, mark, root_end]);
    }

    let trajectory = exporter.export();
    assert_atif_v17_shape(&trajectory);
    assert_eq!(trajectory.steps.len(), 2);
    assert_eq!(trajectory.steps[0].step_id, 1);
    assert_eq!(trajectory.steps[0].source, "agent");
    assert_eq!(trajectory.steps[1].step_id, 2);
    assert_eq!(trajectory.steps[1].source, "system");
    assert!(trajectory.subagent_trajectories.is_none());
    let serialized = serde_json::to_value(&trajectory).unwrap();
    assert!(!serialized.to_string().contains("subagent_trajectory_ref"));
}

#[test]
fn test_exporter_embeds_recursive_subagent_trajectories() {
    let root_uuid = Uuid::now_v7();
    let child_uuid = Uuid::now_v7();
    let grandchild_uuid = Uuid::now_v7();
    let exporter = AtifExporter::new(root_uuid.to_string(), make_agent_info());
    let base = base_timestamp();

    let mut root_start = event_builder(root_uuid, EventType::Start)
        .name("root-agent")
        .scope_type(ScopeType::Agent)
        .build();
    let mut child_start = event_builder(child_uuid, EventType::Start)
        .name("worker-agent")
        .scope_type(ScopeType::Agent)
        .parent_uuid(root_uuid)
        .metadata(json!({"session_id": "child-session"}))
        .build();
    let mut grandchild_start = event_builder(grandchild_uuid, EventType::Start)
        .name("deep-worker")
        .scope_type(ScopeType::Agent)
        .parent_uuid(child_uuid)
        .metadata(json!({"session_id": "grandchild-session"}))
        .build();
    let mut grandchild_mark = event_builder(Uuid::now_v7(), EventType::Mark)
        .name("deep-note")
        .parent_uuid(grandchild_uuid)
        .data(json!({"status": "ok"}))
        .build();
    let mut grandchild_end = event_builder(grandchild_uuid, EventType::End)
        .name("deep-worker")
        .scope_type(ScopeType::Agent)
        .parent_uuid(child_uuid)
        .build();
    let mut child_end = event_builder(child_uuid, EventType::End)
        .name("worker-agent")
        .scope_type(ScopeType::Agent)
        .parent_uuid(root_uuid)
        .build();
    let mut root_end = event_builder(root_uuid, EventType::End)
        .name("root-agent")
        .scope_type(ScopeType::Agent)
        .build();

    for (offset, event) in [
        &mut root_start,
        &mut child_start,
        &mut grandchild_start,
        &mut grandchild_mark,
        &mut grandchild_end,
        &mut child_end,
        &mut root_end,
    ]
    .into_iter()
    .enumerate()
    {
        set_event_timestamp(event, base + chrono::Duration::seconds(offset as i64));
    }

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.extend([
            root_start,
            child_start,
            grandchild_start,
            grandchild_mark,
            grandchild_end,
            child_end,
            root_end,
        ]);
    }

    let trajectory = exporter.export();
    assert_atif_v17_shape(&trajectory);
    let child = &trajectory.subagent_trajectories.as_ref().unwrap()[0];
    assert_eq!(child.session_id, "child-session");
    assert_eq!(child.steps.len(), 1);
    let child_ref = &child.steps[0].observation.as_ref().unwrap().results[0]
        .subagent_trajectory_ref
        .as_ref()
        .unwrap()[0];
    assert_eq!(child_ref.trajectory_id, Some(grandchild_uuid.to_string()));
    assert_eq!(child_ref.session_id, Some("grandchild-session".to_string()));

    let grandchild = &child.subagent_trajectories.as_ref().unwrap()[0];
    assert_eq!(grandchild.trajectory_id, Some(grandchild_uuid.to_string()));
    assert_eq!(grandchild.session_id, "grandchild-session");
    assert_eq!(grandchild.steps.len(), 1);
    assert_eq!(grandchild.steps[0].step_id, 1);
    assert_eq!(grandchild.steps[0].message, json!("deep-note"));
    let extra: AtifStepExtra =
        serde_json::from_value(grandchild.steps[0].extra.clone().unwrap()).unwrap();
    assert_eq!(extra.event_payload.as_ref().unwrap()["status"], json!("ok"));
}

#[test]
fn test_exporter_skips_empty_mark_payloads() {
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.push(
            event_builder(Uuid::now_v7(), EventType::Mark)
                .name("empty-object")
                .data(json!({}))
                .build(),
        );
        state.events.push(
            event_builder(Uuid::now_v7(), EventType::Mark)
                .name("empty-null")
                .data(json!(null))
                .build(),
        );
    }

    let trajectory = exporter.export();
    assert!(trajectory.steps.is_empty());
    assert_eq!(
        trajectory.final_metrics.as_ref().unwrap().total_steps,
        Some(0)
    );
}

#[test]
fn test_trajectory_serde_roundtrip() {
    let trajectory = AtifTrajectory {
        schema_version: ATIF_SCHEMA_VERSION.to_string(),
        session_id: "test-session".to_string(),
        trajectory_id: Some("test-session".to_string()),
        agent: AtifAgentInfo {
            name: "test".to_string(),
            version: "1.0".to_string(),
            model_name: Some("gpt-4".to_string()),
            tool_definitions: Some(vec![json!({"name": "search"})]),
            extra: None,
        },
        steps: vec![AtifStep {
            step_id: 1,
            source: "user".to_string(),
            message: json!("Hello"),
            timestamp: Some("2026-01-01T00:00:00Z".to_string()),
            model_name: None,
            reasoning_effort: None,
            reasoning_content: None,
            tool_calls: None,
            observation: None,
            metrics: Some(AtifMetrics {
                prompt_tokens: Some(10),
                completion_tokens: Some(20),
                cached_tokens: None,
                cost_usd: Some(0.001),
                prompt_token_ids: None,
                completion_token_ids: None,
                logprobs: None,
                extra: None,
            }),
            llm_call_count: None,
            is_copied_context: None,
            extra: None,
        }],
        notes: None,
        final_metrics: Some(AtifFinalMetrics {
            total_prompt_tokens: Some(100),
            total_completion_tokens: Some(200),
            total_cached_tokens: Some(50),
            total_cost_usd: Some(0.01),
            total_steps: Some(1),
            extra: None,
        }),
        continued_trajectory_ref: None,
        subagent_trajectories: None,
        extra: None,
    };

    let json_str = serde_json::to_string(&trajectory).unwrap();
    let deserialized: AtifTrajectory = serde_json::from_str(&json_str).unwrap();

    assert_eq!(deserialized.schema_version, ATIF_SCHEMA_VERSION);
    assert_eq!(deserialized.session_id, "test-session");
    assert_eq!(deserialized.agent.name, "test");
    assert_eq!(deserialized.steps.len(), 1);
    assert_eq!(deserialized.steps[0].step_id, 1);
    assert_eq!(deserialized.steps[0].source, "user");
    let metrics = deserialized.steps[0].metrics.as_ref().unwrap();
    assert_eq!(metrics.prompt_tokens, Some(10));
    let final_metrics = deserialized.final_metrics.as_ref().unwrap();
    assert_eq!(final_metrics.total_prompt_tokens, Some(100));
    assert_eq!(final_metrics.total_steps, Some(1));
}

#[test]
fn test_exporter_scope_filtering() {
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let root1 = Uuid::now_v7();
    let root2 = Uuid::now_v7();

    // Events under scope 1
    let e1 = event_builder(Uuid::now_v7(), EventType::Start)
        .scope_type(ScopeType::Llm)
        .input(json!("agent1 input"))
        .parent_uuid(root1)
        .build();
    let e2 = event_builder(e1.uuid(), EventType::End)
        .scope_type(ScopeType::Llm)
        .output(json!("agent1 output"))
        .parent_uuid(root1)
        .build();

    // Events under scope 2
    let e3 = event_builder(Uuid::now_v7(), EventType::Start)
        .scope_type(ScopeType::Llm)
        .input(json!("agent2 input"))
        .parent_uuid(root2)
        .build();
    let e4 = event_builder(e3.uuid(), EventType::End)
        .scope_type(ScopeType::Llm)
        .output(json!("agent2 output"))
        .parent_uuid(root2)
        .build();

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.push(e1);
        state.events.push(e2);
        state.events.push(e3);
        state.events.push(e4);
    }

    let traj_all = exporter.export();
    assert_eq!(traj_all.steps.len(), 4);
}

#[test]
fn test_exporter_clear() {
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.push(
            event_builder(Uuid::now_v7(), EventType::Mark)
                .data(json!("test"))
                .build(),
        );
    }

    assert_eq!(exporter.export().steps.len(), 1);
    exporter.clear();
    assert!(exporter.export().steps.is_empty());
}

#[test]
fn test_exporter_merged_tool_observations() {
    // Two consecutive tool end events should merge into one observation step.
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let llm_uuid = Uuid::now_v7();
    let tool1_uuid = Uuid::now_v7();
    let tool2_uuid = Uuid::now_v7();

    // LLM end with two promoted tool_calls
    let llm_end = event_builder(llm_uuid, EventType::End)
            .scope_type(ScopeType::Llm)
            .output(json!({
                "content": null,
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"SF\"}"}},
                    {"id": "call_2", "type": "function", "function": {"name": "get_population", "arguments": "{\"city\": \"SF\"}"}}
                ]
            }))
            .build();

    // Two tool start events (skipped)
    let tool1_start = event_builder(tool1_uuid, EventType::Start)
        .name("get_weather")
        .scope_type(ScopeType::Tool)
        .input(json!({"city": "SF"}))
        .build();
    let tool2_start = event_builder(tool2_uuid, EventType::Start)
        .name("get_population")
        .scope_type(ScopeType::Tool)
        .input(json!({"city": "SF"}))
        .build();

    // Two tool end events (should merge)
    let tool1_end = event_builder(tool1_uuid, EventType::End)
        .name("get_weather")
        .scope_type(ScopeType::Tool)
        .output(json!("62°F, foggy"))
        .tool_call_id("call_1")
        .build();
    let tool2_end = event_builder(tool2_uuid, EventType::End)
        .name("get_population")
        .scope_type(ScopeType::Tool)
        .output(json!("873,965"))
        .tool_call_id("call_2")
        .build();

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.push(llm_end);
        state.events.push(tool1_start);
        state.events.push(tool2_start);
        state.events.push(tool1_end);
        state.events.push(tool2_end);
    }

    let trajectory = exporter.export();
    // agent step with attached tool observations
    assert_eq!(trajectory.steps.len(), 1);

    // Agent step with promoted tool_calls
    let agent = &trajectory.steps[0];
    assert_eq!(agent.source, "agent");
    let tcs = agent.tool_calls.as_ref().unwrap();
    assert_eq!(tcs.len(), 2);
    // Arguments should be parsed JSON, not strings
    assert_eq!(tcs[0].arguments, json!({"city": "SF"}));
    assert_eq!(tcs[1].arguments, json!({"city": "SF"}));

    let obs = agent.observation.as_ref().unwrap();
    assert_eq!(obs.results.len(), 2);
    assert_eq!(obs.results[0].source_call_id, Some("call_1".to_string()));
    assert_eq!(obs.results[0].content, Some(json!("62°F, foggy")));
    assert_eq!(obs.results[1].source_call_id, Some("call_2".to_string()));
    assert_eq!(obs.results[1].content, Some(json!("873,965")));
}

#[test]
fn test_exporter_source_call_id_correlation_by_name() {
    // When tool_call_id is absent on the tool end event, correlate via function name
    // against the preceding LLM End's promoted tool_calls.
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let llm_uuid = Uuid::now_v7();
    let tool_uuid = Uuid::now_v7();

    let llm_end = event_builder(llm_uuid, EventType::End)
            .scope_type(ScopeType::Llm)
            .output(json!({
                "content": null,
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_xyz", "type": "function", "function": {"name": "search", "arguments": "{}"}}
                ]
            }))
            .build();

    // Tool end without tool_call_id, but with function name
    let tool_end = event_builder(tool_uuid, EventType::End)
        .name("search")
        .scope_type(ScopeType::Tool)
        .output(json!({"results": []}))
        .build();

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.push(llm_end);
        state.events.push(tool_end);
    }

    let trajectory = exporter.export();
    assert_eq!(trajectory.steps.len(), 1);

    let obs = trajectory.steps[0].observation.as_ref().unwrap();
    // Correlated by function name "search" → "call_xyz"
    assert_eq!(obs.results[0].source_call_id, Some("call_xyz".to_string()));
}

#[test]
fn test_exporter_user_message_extraction() {
    // LLM start input with max_tokens/model/tools/stream should extract just messages.
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let llm_uuid = Uuid::now_v7();

    let start = event_builder(llm_uuid, EventType::Start)
        .scope_type(ScopeType::Llm)
        .input(json!({
            "content": {
                "messages": [{"role": "user", "content": "hello"}],
                "model": "gpt-4",
                "max_tokens": 1024,
                "stream": false,
                "tools": [{"type": "function", "function": {"name": "search"}}]
            },
            "headers": {}
        }))
        .build();

    let end = event_builder(llm_uuid, EventType::End)
        .scope_type(ScopeType::Llm)
        .output(json!("response"))
        .build();

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.push(start);
        state.events.push(end);
    }

    let trajectory = exporter.export();
    assert_eq!(trajectory.steps[0].message, json!("hello"));
}

#[test]
fn test_exporter_full_agent_loop() {
    // Simulate a complete agent loop: LLM→tool_calls→observations→LLM→final answer
    // This should produce 5 steps: user, agent+tool_calls, merged obs, user, agent
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let llm1_uuid = Uuid::now_v7();
    let llm2_uuid = Uuid::now_v7();
    let t1_uuid = Uuid::now_v7();
    let t2_uuid = Uuid::now_v7();

    // First LLM start
    let llm1_start = event_builder(llm1_uuid, EventType::Start)
        .scope_type(ScopeType::Llm)
        .input(json!({
            "messages": [{"role": "user", "content": "What is the weather and population of SF?"}],
            "model": "nemotron",
            "tools": []
        }))
        .model_name("nemotron")
        .build();

    // First LLM end with tool_calls
    let llm1_end = event_builder(llm1_uuid, EventType::End)
            .scope_type(ScopeType::Llm)
            .output(json!({
                "content": null,
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"SF\"}"}},
                    {"id": "c2", "type": "function", "function": {"name": "get_population", "arguments": "{\"city\":\"SF\"}"}}
                ],
                "token_usage": {"prompt_tokens": 100, "completion_tokens": 50}
            }))
            .model_name("nemotron")
            .build();

    // Tool starts (skipped)
    let t1_start = event_builder(t1_uuid, EventType::Start)
        .name("get_weather")
        .scope_type(ScopeType::Tool)
        .input(json!({"city": "SF"}))
        .build();
    let t2_start = event_builder(t2_uuid, EventType::Start)
        .name("get_population")
        .scope_type(ScopeType::Tool)
        .input(json!({"city": "SF"}))
        .build();

    // Tool ends (merged)
    let t1_end = event_builder(t1_uuid, EventType::End)
        .name("get_weather")
        .scope_type(ScopeType::Tool)
        .output(json!("62°F, foggy"))
        .tool_call_id("c1")
        .build();
    let t2_end = event_builder(t2_uuid, EventType::End)
        .name("get_population")
        .scope_type(ScopeType::Tool)
        .output(json!("873,965"))
        .tool_call_id("c2")
        .build();

    // Second LLM start (with tool results in messages)
    let llm2_start = event_builder(llm2_uuid, EventType::Start)
        .scope_type(ScopeType::Llm)
        .input(json!({
            "messages": [
                {"role": "user", "content": "What is the weather and population of SF?"},
                {"role": "assistant", "content": null, "tool_calls": [{"id": "c1"}, {"id": "c2"}]},
                {"role": "tool", "content": "62°F, foggy", "tool_call_id": "c1"},
                {"role": "tool", "content": "873,965", "tool_call_id": "c2"}
            ],
            "model": "nemotron"
        }))
        .model_name("nemotron")
        .build();

    // Second LLM end (final answer)
    let llm2_end = event_builder(llm2_uuid, EventType::End)
        .scope_type(ScopeType::Llm)
        .output(json!({
            "content": "The weather in SF is 62°F and foggy. Population is 873,965.",
            "role": "assistant",
            "token_usage": {"prompt_tokens": 200, "completion_tokens": 30}
        }))
        .model_name("nemotron")
        .build();

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.extend([
            llm1_start, llm1_end, t1_start, t2_start, t1_end, t2_end, llm2_start, llm2_end,
        ]);
    }

    let trajectory = exporter.export();
    assert_atif_v17_shape(&trajectory);
    // Expected: user, agent+tool_calls+observations, user, agent
    assert_eq!(trajectory.steps.len(), 4);

    assert_eq!(trajectory.steps[0].source, "user");
    assert_eq!(trajectory.steps[0].step_id, 1);

    assert_eq!(trajectory.steps[1].source, "agent");
    assert_eq!(trajectory.steps[1].step_id, 2);
    let tcs = trajectory.steps[1].tool_calls.as_ref().unwrap();
    assert_eq!(tcs.len(), 2);
    assert_eq!(tcs[0].function_name, "get_weather");
    assert_eq!(tcs[1].function_name, "get_population");

    assert_eq!(trajectory.steps[2].source, "user");
    assert_eq!(trajectory.steps[2].step_id, 3);
    let obs = trajectory.steps[1].observation.as_ref().unwrap();
    assert_eq!(obs.results.len(), 2);

    assert_eq!(trajectory.steps[3].source, "agent");
    assert_eq!(trajectory.steps[3].step_id, 4);
    assert_eq!(
        trajectory.steps[3].message,
        json!("The weather in SF is 62°F and foggy. Population is 873,965.")
    );

    // Final metrics should aggregate both LLM calls
    let fm = trajectory.final_metrics.as_ref().unwrap();
    assert_eq!(fm.total_prompt_tokens, Some(300));
    assert_eq!(fm.total_completion_tokens, Some(80));
}

#[test]
fn test_reasoning_content_extracted() {
    // When an LLM End event carries output["reasoning"], the agent step
    // should have reasoning_content populated.
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let llm_uuid = Uuid::now_v7();

    let end = event_builder(llm_uuid, EventType::End)
        .scope_type(ScopeType::Llm)
        .output(json!({
            "content": "The answer is 42.",
            "role": "assistant",
            "reasoning": "Let me think step by step. The question asks for the meaning of life...",
            "token_usage": { "prompt_tokens": 10, "completion_tokens": 5 }
        }))
        .build();

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.push(end);
    }

    let trajectory = exporter.export();
    let agent_step = &trajectory.steps[0];
    assert_eq!(agent_step.source, "agent");
    assert_eq!(
        agent_step.reasoning_content,
        Some("Let me think step by step. The question asks for the meaning of life...".to_string())
    );
    // reasoning_content should not bleed into message
    assert_eq!(agent_step.message, json!("The answer is 42."));
}

#[test]
fn test_reasoning_effort_propagated() {
    // reasoning_effort is set on the LLM Start event input and must be
    // carried forward to the agent step produced by the LLM End event.
    // This tests the stateful current_reasoning_effort handoff.
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let llm_uuid = Uuid::now_v7();

    let start = event_builder(llm_uuid, EventType::Start)
        .scope_type(ScopeType::Llm)
        .input(json!({
            "messages": [{"role": "user", "content": "solve this"}],
            "reasoning_effort": "high"
        }))
        .build();

    let end = event_builder(llm_uuid, EventType::End)
        .scope_type(ScopeType::Llm)
        .output(json!({
            "content": "Done.",
            "role": "assistant"
        }))
        .build();

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.push(start);
        state.events.push(end);
    }

    let trajectory = exporter.export();
    // steps: user (LLM Start), agent (LLM End)
    let agent_step = &trajectory.steps[1];
    assert_eq!(agent_step.source, "agent");
    assert_eq!(agent_step.reasoning_effort, Some(json!("high")));
    // User step should NOT carry reasoning_effort
    assert!(trajectory.steps[0].reasoning_effort.is_none());
}

#[test]
fn test_metrics_extra_captures_unknown_token_usage_keys() {
    // Unknown keys in token_usage (e.g. reasoning_tokens) should be
    // routed to metrics.extra rather than silently dropped.
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let llm_uuid = Uuid::now_v7();

    let end = event_builder(llm_uuid, EventType::End)
        .scope_type(ScopeType::Llm)
        .output(json!({
            "content": "ok",
            "role": "assistant",
            "token_usage": {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "reasoning_tokens": 150,
                "cache_creation_input_tokens": 5
            }
        }))
        .build();

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.push(end);
    }

    let trajectory = exporter.export();
    let metrics = trajectory.steps[0].metrics.as_ref().unwrap();
    assert_eq!(metrics.prompt_tokens, Some(20));
    assert_eq!(metrics.completion_tokens, Some(10));
    // Unknown keys land in extra
    let extra = metrics.extra.as_ref().unwrap();
    assert_eq!(extra["reasoning_tokens"], json!(150));
    assert_eq!(extra["cache_creation_input_tokens"], json!(5));
    // Known keys do not appear in extra
    assert!(extra.get("prompt_tokens").is_none());
    assert!(extra.get("completion_tokens").is_none());
}

#[test]
fn test_step_extra_agent_ancestry() {
    // Agent step extra.ancestry is populated with function_id, function_name,
    // parent_id from the LLM End event.
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let agent_uuid = Uuid::now_v7();
    let llm_uuid = Uuid::now_v7();

    let llm_start = event_builder(llm_uuid, EventType::Start)
        .name("gpt-4")
        .scope_type(ScopeType::Llm)
        .parent_uuid(agent_uuid)
        .input(json!({"messages": [{"role": "user", "content": "hi"}]}))
        .build();

    let llm_end = event_builder(llm_uuid, EventType::End)
        .name("gpt-4")
        .scope_type(ScopeType::Llm)
        .parent_uuid(agent_uuid)
        .output(json!({"content": "hello", "role": "assistant"}))
        .build();

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.push(llm_start);
        state.events.push(llm_end);
    }

    let trajectory = exporter.export();
    let agent_step = &trajectory.steps[1];
    assert_eq!(agent_step.source, "agent");

    let extra: AtifStepExtra = serde_json::from_value(agent_step.extra.clone().unwrap()).unwrap();
    assert_eq!(extra.ancestry.function_id, llm_uuid.to_string());
    assert_eq!(extra.ancestry.function_name, "gpt-4");
    assert_eq!(extra.ancestry.parent_id, Some(agent_uuid.to_string()));
}

#[test]
fn test_step_extra_invocation_timestamps() {
    // Agent step extra.invocation carries paired start_timestamp and end_timestamp.
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let llm_uuid = Uuid::now_v7();

    let llm_start = event_builder(llm_uuid, EventType::Start)
        .name("gpt-4")
        .scope_type(ScopeType::Llm)
        .input(json!({"messages": []}))
        .build();

    let llm_end = event_builder(llm_uuid, EventType::End)
        .name("gpt-4")
        .scope_type(ScopeType::Llm)
        .output(json!({"content": "done", "role": "assistant"}))
        .build();

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.push(llm_start);
        state.events.push(llm_end);
    }

    let trajectory = exporter.export();
    let agent_step = &trajectory.steps[1];
    let extra: AtifStepExtra = serde_json::from_value(agent_step.extra.clone().unwrap()).unwrap();

    let inv = extra.invocation.as_ref().unwrap();
    assert!(inv.start_timestamp.is_some());
    assert!(inv.end_timestamp.is_some());
    // end must be >= start
    assert!(inv.end_timestamp.unwrap() >= inv.start_timestamp.unwrap());
    assert_eq!(inv.invocation_id, Some(llm_uuid.to_string()));
    assert_eq!(inv.framework, Some("nemo_relay".to_string()));
}

#[test]
fn test_step_extra_user_step_has_ancestry_no_invocation() {
    // User step (LLM Start) gets ancestry but invocation is None —
    // end time is unknown at the time the user step is emitted.
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let llm_uuid = Uuid::now_v7();

    let llm_start = event_builder(llm_uuid, EventType::Start)
        .name("gpt-4")
        .scope_type(ScopeType::Llm)
        .input(json!({"messages": [{"role": "user", "content": "hi"}]}))
        .build();

    let llm_end = event_builder(llm_uuid, EventType::End)
        .name("gpt-4")
        .scope_type(ScopeType::Llm)
        .output(json!({"content": "hi back", "role": "assistant"}))
        .build();

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.push(llm_start);
        state.events.push(llm_end);
    }

    let trajectory = exporter.export();
    let user_step = &trajectory.steps[0];
    assert_eq!(user_step.source, "user");

    let extra: AtifStepExtra = serde_json::from_value(user_step.extra.clone().unwrap()).unwrap();
    assert_eq!(extra.ancestry.function_id, llm_uuid.to_string());
    assert!(extra.invocation.is_none());
}

#[test]
fn test_step_extra_tool_ancestry_aligned_with_tool_calls() {
    // tool_ancestry[i] must align with tool_calls[i] on the agent step.
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let llm_uuid = Uuid::now_v7();
    let tool1_uuid = Uuid::now_v7();
    let tool2_uuid = Uuid::now_v7();

    let llm_end = event_builder(llm_uuid, EventType::End)
        .scope_type(ScopeType::Llm)
        .output(json!({
            "content": null,
            "role": "assistant",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "search", "arguments": "{}"}},
                {"id": "c2", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
            ]
        }))
        .build();

    let tool1_end = event_builder(tool1_uuid, EventType::End)
        .name("search")
        .scope_type(ScopeType::Tool)
        .output(json!("result1"))
        .tool_call_id("c1")
        .build();

    let tool2_end = event_builder(tool2_uuid, EventType::End)
        .name("lookup")
        .scope_type(ScopeType::Tool)
        .output(json!("result2"))
        .tool_call_id("c2")
        .build();

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.push(llm_end);
        state.events.push(tool1_end);
        state.events.push(tool2_end);
    }

    let trajectory = exporter.export();
    let agent_step = &trajectory.steps[0];
    let extra: AtifStepExtra = serde_json::from_value(agent_step.extra.clone().unwrap()).unwrap();

    assert_eq!(extra.tool_ancestry.len(), 2);
    assert_eq!(extra.tool_ancestry[0].function_id, tool1_uuid.to_string());
    assert_eq!(extra.tool_ancestry[0].function_name, "search");
    assert_eq!(extra.tool_ancestry[1].function_id, tool2_uuid.to_string());
    assert_eq!(extra.tool_ancestry[1].function_name, "lookup");

    let tool_invocations = extra.tool_invocations.as_ref().unwrap();
    assert_eq!(tool_invocations.len(), 2);
    assert_eq!(tool_invocations[0].invocation_id, Some("c1".to_string()));
    assert_eq!(tool_invocations[1].invocation_id, Some("c2".to_string()));
}

#[test]
fn test_step_extra_tool_ancestry_aligned_out_of_order_completion() {
    // Tools complete in reverse order (c2 before c1) but ancestry must
    // still align with tool_calls declaration order (c1=search, c2=lookup).
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let llm_uuid = Uuid::now_v7();
    let tool1_uuid = Uuid::now_v7();
    let tool2_uuid = Uuid::now_v7();

    let llm_end = event_builder(llm_uuid, EventType::End)
        .scope_type(ScopeType::Llm)
        .output(json!({
            "content": null,
            "role": "assistant",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "search", "arguments": "{}"}},
                {"id": "c2", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
            ]
        }))
        .build();

    // c2 (lookup) completes before c1 (search) — out of declaration order.
    let mut tool2_end = event_builder(tool2_uuid, EventType::End)
        .name("lookup")
        .scope_type(ScopeType::Tool)
        .output(json!("result2"))
        .tool_call_id("c2")
        .build();
    let tool2_end_ts = chrono::Utc::now();
    set_event_timestamp(&mut tool2_end, tool2_end_ts);

    let mut tool1_end = event_builder(tool1_uuid, EventType::End)
        .name("search")
        .scope_type(ScopeType::Tool)
        .output(json!("result1"))
        .tool_call_id("c1")
        .build();
    // Ensure tool1_end sorts after tool2_end by timestamp.
    set_event_timestamp(
        &mut tool1_end,
        tool2_end_ts + chrono::Duration::milliseconds(10),
    );

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.push(llm_end);
        state.events.push(tool2_end);
        state.events.push(tool1_end);
    }

    let trajectory = exporter.export();
    let agent_step = &trajectory.steps[0];
    let extra: AtifStepExtra = serde_json::from_value(agent_step.extra.clone().unwrap()).unwrap();

    // Despite out-of-order completion, ancestry aligns with tool_calls declaration order.
    assert_eq!(extra.tool_ancestry.len(), 2);
    assert_eq!(extra.tool_ancestry[0].function_name, "search"); // tool_calls[0] = c1
    assert_eq!(extra.tool_ancestry[1].function_name, "lookup"); // tool_calls[1] = c2

    let tool_invocations = extra.tool_invocations.as_ref().unwrap();
    assert_eq!(tool_invocations.len(), 2);
    assert_eq!(tool_invocations[0].invocation_id, Some("c1".to_string()));
    assert_eq!(tool_invocations[1].invocation_id, Some("c2".to_string()));
}

#[test]
fn test_step_extra_tool_ancestry_does_not_bleed_across_turns() {
    // Tool ancestry from turn 1 must not appear on the agent step of turn 2.
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let llm1_uuid = Uuid::now_v7();
    let llm2_uuid = Uuid::now_v7();
    let tool1_uuid = Uuid::now_v7();
    let tool2_uuid = Uuid::now_v7();

    // Turn 1: LLM call + one tool
    let llm1_end = event_builder(llm1_uuid, EventType::End)
        .scope_type(ScopeType::Llm)
        .output(json!({
            "content": null, "role": "assistant",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "search", "arguments": "{}"}}
            ]
        }))
        .build();
    let tool1_end = event_builder(tool1_uuid, EventType::End)
        .name("search")
        .scope_type(ScopeType::Tool)
        .output(json!("result1"))
        .tool_call_id("c1")
        .build();

    // Turn 2: new LLM call + one different tool
    let llm2_start = event_builder(llm2_uuid, EventType::Start)
        .scope_type(ScopeType::Llm)
        .input(json!({"messages": []}))
        .build();
    let llm2_end = event_builder(llm2_uuid, EventType::End)
        .scope_type(ScopeType::Llm)
        .output(json!({
            "content": null, "role": "assistant",
            "tool_calls": [
                {"id": "c2", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
            ]
        }))
        .build();
    let tool2_end = event_builder(tool2_uuid, EventType::End)
        .name("lookup")
        .scope_type(ScopeType::Tool)
        .output(json!("result2"))
        .tool_call_id("c2")
        .build();

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.push(llm1_end);
        state.events.push(tool1_end);
        state.events.push(llm2_start);
        state.events.push(llm2_end);
        state.events.push(tool2_end);
    }

    let trajectory = exporter.export();
    // steps: agent(turn1+obs1), user(turn2), agent(turn2+obs2)
    let agent1 = trajectory
        .steps
        .iter()
        .find(|s| s.source == "agent" && s.step_id == 1)
        .unwrap();
    let agent2 = trajectory
        .steps
        .iter()
        .find(|s| s.source == "agent" && s.step_id == 3)
        .unwrap();

    let extra1: AtifStepExtra = serde_json::from_value(agent1.extra.clone().unwrap()).unwrap();
    let extra2: AtifStepExtra = serde_json::from_value(agent2.extra.clone().unwrap()).unwrap();

    // Turn 1 agent step has only search
    assert_eq!(extra1.tool_ancestry.len(), 1);
    assert_eq!(extra1.tool_ancestry[0].function_name, "search");

    // Turn 2 agent step has only lookup — no bleed from turn 1
    assert_eq!(extra2.tool_ancestry.len(), 1);
    assert_eq!(extra2.tool_ancestry[0].function_name, "lookup");
}
