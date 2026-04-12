// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::types::event::Event;
use crate::types::llm::LLMAttributes;
use crate::types::scope::{HandleAttributes, ScopeAttributes, ScopeType};
use crate::types::tool::ToolAttributes;
use serde_json::json;

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

    fn parent_uuid(mut self, parent_uuid: Option<Uuid>) -> Self {
        self.parent_uuid = parent_uuid;
        self
    }

    fn data(mut self, data: Option<serde_json::Value>) -> Self {
        self.data = data;
        self
    }

    fn scope_type(mut self, scope_type: ScopeType) -> Self {
        self.scope_type = Some(scope_type);
        self
    }

    fn input(mut self, input: Option<serde_json::Value>) -> Self {
        self.input = input;
        self
    }

    fn output(mut self, output: Option<serde_json::Value>) -> Self {
        self.output = output;
        self
    }

    fn model_name(mut self, model_name: Option<String>) -> Self {
        self.model_name = model_name;
        self
    }

    fn tool_call_id(mut self, tool_call_id: Option<String>) -> Self {
        self.tool_call_id = tool_call_id;
        self
    }

    fn build(self) -> Event {
        match (self.event_type, self.scope_type) {
            (EventType::Mark, _) => Event::mark(
                self.parent_uuid,
                self.uuid,
                self.name,
                self.data,
                self.metadata,
            ),
            (EventType::Start, Some(ScopeType::Tool)) => Event::tool_start(
                self.parent_uuid,
                self.uuid,
                self.name,
                self.data,
                self.metadata,
                match self.attributes {
                    Some(HandleAttributes::Tool(attributes)) => attributes,
                    _ => ToolAttributes::empty(),
                },
                self.input,
                self.tool_call_id,
            ),
            (EventType::End, Some(ScopeType::Tool)) => Event::tool_end(
                self.parent_uuid,
                self.uuid,
                self.name,
                self.data,
                self.metadata,
                match self.attributes {
                    Some(HandleAttributes::Tool(attributes)) => attributes,
                    _ => ToolAttributes::empty(),
                },
                self.output,
                self.tool_call_id,
            ),
            (EventType::Start, Some(ScopeType::Llm)) => Event::llm_start(
                self.parent_uuid,
                self.uuid,
                self.name,
                self.data,
                self.metadata,
                match self.attributes {
                    Some(HandleAttributes::Llm(attributes)) => attributes,
                    _ => LLMAttributes::empty(),
                },
                self.input,
                self.model_name,
                None,
            ),
            (EventType::End, Some(ScopeType::Llm)) => Event::llm_end(
                self.parent_uuid,
                self.uuid,
                self.name,
                self.data,
                self.metadata,
                match self.attributes {
                    Some(HandleAttributes::Llm(attributes)) => attributes,
                    _ => LLMAttributes::empty(),
                },
                self.output,
                self.model_name,
                None,
            ),
            (EventType::Start, Some(scope_type)) => Event::scope_start(
                self.parent_uuid,
                self.uuid,
                self.name,
                self.data,
                self.metadata,
                match self.attributes {
                    Some(HandleAttributes::Scope(attributes)) => attributes,
                    _ => ScopeAttributes::empty(),
                },
                scope_type,
            ),
            (EventType::End, Some(scope_type)) => Event::scope_end(
                self.parent_uuid,
                self.uuid,
                self.name,
                self.data,
                self.metadata,
                match self.attributes {
                    Some(HandleAttributes::Scope(attributes)) => attributes,
                    _ => ScopeAttributes::empty(),
                },
                scope_type,
            ),
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
        Event::ScopeStart(inner) => inner.timestamp = timestamp,
        Event::ScopeEnd(inner) => inner.timestamp = timestamp,
        Event::ToolStart(inner) => inner.timestamp = timestamp,
        Event::ToolEnd(inner) => inner.timestamp = timestamp,
        Event::LLMStart(inner) => inner.timestamp = timestamp,
        Event::LLMEnd(inner) => inner.timestamp = timestamp,
        Event::Mark(inner) => inner.timestamp = timestamp,
    }
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
    assert_eq!(ATIF_SCHEMA_VERSION, "ATIF-v1.6");
}

#[test]
fn test_exporter_tool_lifecycle() {
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let tool_uuid = Uuid::now_v7();

    // Simulate tool start (should be SKIPPED — tool_calls come from LLM End)
    let start = event_builder(tool_uuid, EventType::Start)
        .name("web_search")
        .scope_type(ScopeType::Tool)
        .input(Some(json!({"query": "test"})))
        .tool_call_id(Some("call_123".to_string()))
        .build();

    // Simulate tool end
    let end = event_builder(tool_uuid, EventType::End)
        .name("web_search")
        .scope_type(ScopeType::Tool)
        .output(Some(json!({"results": ["result1"]})))
        .tool_call_id(Some("call_123".to_string()))
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
    assert_eq!(obs.results[0].source_call_id, Some("call_123".to_string()));
    assert_eq!(obs.results[0].content, json!({"results": ["result1"]}));
}

#[test]
fn test_exporter_llm_lifecycle() {
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let llm_uuid = Uuid::now_v7();

    // Input wrapped in LLMRequest envelope — should be unwrapped.
    let start = event_builder(llm_uuid, EventType::Start)
        .name("gpt-4")
        .scope_type(ScopeType::Llm)
        .input(Some(json!({
            "content": {"messages": [{"role": "user", "content": "hello"}]},
            "headers": {}
        })))
        .model_name(Some("gpt-4".to_string()))
        .build();

    // Output with content, token_usage, and tool_calls.
    let end = event_builder(llm_uuid, EventType::End)
        .name("gpt-4")
        .scope_type(ScopeType::Llm)
        .output(Some(json!({
            "content": "Hi there!",
            "role": "assistant",
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            },
            "tool_calls": []
        })))
        .model_name(Some("gpt-4".to_string()))
        .build();

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.push(start);
        state.events.push(end);
    }

    let trajectory = exporter.export();
    assert_eq!(trajectory.steps.len(), 2);

    // First step: user (LLM start — unwrapped LLMRequest, then messages extracted)
    let step1 = &trajectory.steps[0];
    assert_eq!(step1.step_id, 1);
    assert_eq!(step1.source, "user");
    // extract_user_messages pulls out just the messages array
    assert_eq!(step1.message, json!([{"role": "user", "content": "hello"}]));
    assert_eq!(step1.model_name, Some("gpt-4".to_string()));

    // Second step: agent (LLM end with extracted content + metrics)
    let step2 = &trajectory.steps[1];
    assert_eq!(step2.step_id, 2);
    assert_eq!(step2.source, "agent");
    assert_eq!(step2.message, json!("Hi there!"));
    assert_eq!(step2.model_name, Some("gpt-4".to_string()));
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
fn test_exporter_llm_lifecycle_plain_input() {
    // Input without LLMRequest envelope — passed through unchanged.
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let llm_uuid = Uuid::now_v7();

    let start = event_builder(llm_uuid, EventType::Start)
        .name("gpt-4")
        .scope_type(ScopeType::Llm)
        .input(Some(
            json!({"messages": [{"role": "user", "content": "hello"}]}),
        ))
        .model_name(Some("gpt-4".to_string()))
        .build();

    let end = event_builder(llm_uuid, EventType::End)
        .name("gpt-4")
        .scope_type(ScopeType::Llm)
        .output(Some(json!("simple string response")))
        .model_name(Some("gpt-4".to_string()))
        .build();

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.push(start);
        state.events.push(end);
    }

    let trajectory = exporter.export();
    assert_eq!(trajectory.steps.len(), 2);

    // Input without headers key — messages array is still extracted
    assert_eq!(
        trajectory.steps[0].message,
        json!([{"role": "user", "content": "hello"}])
    );
    // Non-object output is passed through as-is
    assert_eq!(trajectory.steps[1].message, json!("simple string response"));
    assert!(trajectory.steps[1].metrics.is_none());
    // No token metrics on any step — token totals are None, but total_steps is still set
    let fm = trajectory.final_metrics.as_ref().unwrap();
    assert!(fm.total_prompt_tokens.is_none());
    assert_eq!(fm.total_steps, Some(2));
}

#[test]
fn test_exporter_llm_tool_calls_promoted() {
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let llm_uuid = Uuid::now_v7();

    let end = event_builder(llm_uuid, EventType::End)
        .name("gpt-4")
        .scope_type(ScopeType::Llm)
        .output(Some(json!({
            "content": null,
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": "{\"q\": \"test\"}"
                    }
                }
            ]
        })))
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

    // message should be a summary (content was null)
    assert_eq!(
        step.message,
        json!({"role": "assistant", "tool_calls": [{"id": "call_abc", "type": "function", "function": {"name": "search", "arguments": "{\"q\": \"test\"}"}}]})
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
        .input(Some(json!({"prompt": "What is 2+2?"})))
        .build();
    let llm_end = event_builder(llm_uuid, EventType::End)
        .scope_type(ScopeType::Llm)
        .output(Some(json!({"answer": "4"})))
        .build();

    // Tool start/end
    let tool_start = event_builder(tool_uuid, EventType::Start)
        .name("calculator")
        .scope_type(ScopeType::Tool)
        .input(Some(json!({"expr": "2+2"})))
        .tool_call_id(Some("call_1".to_string()))
        .build();
    let tool_end = event_builder(tool_uuid, EventType::End)
        .name("calculator")
        .scope_type(ScopeType::Tool)
        .output(Some(json!(4)))
        .tool_call_id(Some("call_1".to_string()))
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
    // Scope events and Tool Start are skipped: user, agent, system(obs)
    assert_eq!(trajectory.steps.len(), 3);

    assert_eq!(trajectory.steps[0].source, "user");
    assert_eq!(trajectory.steps[1].source, "agent");
    assert_eq!(trajectory.steps[2].source, "system");
    assert!(trajectory.steps[2].observation.is_some());

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
        .input(Some(json!({"x": 1})))
        .tool_call_id(Some("call_abc".to_string()))
        .build();

    let end = event_builder(tool_uuid, EventType::End)
        .name("my_tool")
        .scope_type(ScopeType::Tool)
        .output(Some(json!({"y": 2})))
        .tool_call_id(Some("call_abc".to_string()))
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
    assert_eq!(obs_result.source_call_id, Some("call_abc".to_string()));
}

#[test]
fn test_trajectory_serde_roundtrip() {
    let trajectory = AtifTrajectory {
        schema_version: ATIF_SCHEMA_VERSION.to_string(),
        session_id: "test-session".to_string(),
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
        .input(Some(json!("agent1 input")))
        .parent_uuid(Some(root1))
        .build();
    let e2 = event_builder(e1.uuid(), EventType::End)
        .scope_type(ScopeType::Llm)
        .output(Some(json!("agent1 output")))
        .parent_uuid(Some(root1))
        .build();

    // Events under scope 2
    let e3 = event_builder(Uuid::now_v7(), EventType::Start)
        .scope_type(ScopeType::Llm)
        .input(Some(json!("agent2 input")))
        .parent_uuid(Some(root2))
        .build();
    let e4 = event_builder(e3.uuid(), EventType::End)
        .scope_type(ScopeType::Llm)
        .output(Some(json!("agent2 output")))
        .parent_uuid(Some(root2))
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
                .data(Some(json!("test")))
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
            .output(Some(json!({
                "content": null,
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"SF\"}"}},
                    {"id": "call_2", "type": "function", "function": {"name": "get_population", "arguments": "{\"city\": \"SF\"}"}}
                ]
            })))
            .build();

    // Two tool start events (skipped)
    let tool1_start = event_builder(tool1_uuid, EventType::Start)
        .name("get_weather")
        .scope_type(ScopeType::Tool)
        .input(Some(json!({"city": "SF"})))
        .build();
    let tool2_start = event_builder(tool2_uuid, EventType::Start)
        .name("get_population")
        .scope_type(ScopeType::Tool)
        .input(Some(json!({"city": "SF"})))
        .build();

    // Two tool end events (should merge)
    let tool1_end = event_builder(tool1_uuid, EventType::End)
        .name("get_weather")
        .scope_type(ScopeType::Tool)
        .output(Some(json!("62°F, foggy")))
        .tool_call_id(Some("call_1".to_string()))
        .build();
    let tool2_end = event_builder(tool2_uuid, EventType::End)
        .name("get_population")
        .scope_type(ScopeType::Tool)
        .output(Some(json!("873,965")))
        .tool_call_id(Some("call_2".to_string()))
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
    // agent step + single merged observation step
    assert_eq!(trajectory.steps.len(), 2);

    // Agent step with promoted tool_calls
    let agent = &trajectory.steps[0];
    assert_eq!(agent.source, "agent");
    let tcs = agent.tool_calls.as_ref().unwrap();
    assert_eq!(tcs.len(), 2);
    // Arguments should be parsed JSON, not strings
    assert_eq!(tcs[0].arguments, json!({"city": "SF"}));
    assert_eq!(tcs[1].arguments, json!({"city": "SF"}));

    // Merged observation step
    let obs_step = &trajectory.steps[1];
    assert_eq!(obs_step.source, "system");
    let obs = obs_step.observation.as_ref().unwrap();
    assert_eq!(obs.results.len(), 2);
    assert_eq!(obs.results[0].source_call_id, Some("call_1".to_string()));
    assert_eq!(obs.results[0].content, json!("62°F, foggy"));
    assert_eq!(obs.results[1].source_call_id, Some("call_2".to_string()));
    assert_eq!(obs.results[1].content, json!("873,965"));
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
            .output(Some(json!({
                "content": null,
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_xyz", "type": "function", "function": {"name": "search", "arguments": "{}"}}
                ]
            })))
            .build();

    // Tool end without tool_call_id, but with function name
    let tool_end = event_builder(tool_uuid, EventType::End)
        .name("search")
        .scope_type(ScopeType::Tool)
        .output(Some(json!({"results": []})))
        .build();

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.push(llm_end);
        state.events.push(tool_end);
    }

    let trajectory = exporter.export();
    assert_eq!(trajectory.steps.len(), 2);

    let obs = trajectory.steps[1].observation.as_ref().unwrap();
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
        .input(Some(json!({
            "content": {
                "messages": [{"role": "user", "content": "hello"}],
                "model": "gpt-4",
                "max_tokens": 1024,
                "stream": false,
                "tools": [{"type": "function", "function": {"name": "search"}}]
            },
            "headers": {}
        })))
        .build();

    let end = event_builder(llm_uuid, EventType::End)
        .scope_type(ScopeType::Llm)
        .output(Some(json!("response")))
        .build();

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.push(start);
        state.events.push(end);
    }

    let trajectory = exporter.export();
    // User step should contain just the messages array
    assert_eq!(
        trajectory.steps[0].message,
        json!([{"role": "user", "content": "hello"}])
    );
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
        .input(Some(json!({
            "messages": [{"role": "user", "content": "What is the weather and population of SF?"}],
            "model": "nemotron",
            "tools": []
        })))
        .model_name(Some("nemotron".to_string()))
        .build();

    // First LLM end with tool_calls
    let llm1_end = event_builder(llm1_uuid, EventType::End)
            .scope_type(ScopeType::Llm)
            .output(Some(json!({
                "content": null,
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"SF\"}"}},
                    {"id": "c2", "type": "function", "function": {"name": "get_population", "arguments": "{\"city\":\"SF\"}"}}
                ],
                "token_usage": {"prompt_tokens": 100, "completion_tokens": 50}
            })))
            .model_name(Some("nemotron".to_string()))
            .build();

    // Tool starts (skipped)
    let t1_start = event_builder(t1_uuid, EventType::Start)
        .name("get_weather")
        .scope_type(ScopeType::Tool)
        .input(Some(json!({"city": "SF"})))
        .build();
    let t2_start = event_builder(t2_uuid, EventType::Start)
        .name("get_population")
        .scope_type(ScopeType::Tool)
        .input(Some(json!({"city": "SF"})))
        .build();

    // Tool ends (merged)
    let t1_end = event_builder(t1_uuid, EventType::End)
        .name("get_weather")
        .scope_type(ScopeType::Tool)
        .output(Some(json!("62°F, foggy")))
        .tool_call_id(Some("c1".to_string()))
        .build();
    let t2_end = event_builder(t2_uuid, EventType::End)
        .name("get_population")
        .scope_type(ScopeType::Tool)
        .output(Some(json!("873,965")))
        .tool_call_id(Some("c2".to_string()))
        .build();

    // Second LLM start (with tool results in messages)
    let llm2_start = event_builder(llm2_uuid, EventType::Start)
        .scope_type(ScopeType::Llm)
        .input(Some(json!({
            "messages": [
                {"role": "user", "content": "What is the weather and population of SF?"},
                {"role": "assistant", "content": null, "tool_calls": [{"id": "c1"}, {"id": "c2"}]},
                {"role": "tool", "content": "62°F, foggy", "tool_call_id": "c1"},
                {"role": "tool", "content": "873,965", "tool_call_id": "c2"}
            ],
            "model": "nemotron"
        })))
        .model_name(Some("nemotron".to_string()))
        .build();

    // Second LLM end (final answer)
    let llm2_end = event_builder(llm2_uuid, EventType::End)
        .scope_type(ScopeType::Llm)
        .output(Some(json!({
            "content": "The weather in SF is 62°F and foggy. Population is 873,965.",
            "role": "assistant",
            "token_usage": {"prompt_tokens": 200, "completion_tokens": 30}
        })))
        .model_name(Some("nemotron".to_string()))
        .build();

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.extend([
            llm1_start, llm1_end, t1_start, t2_start, t1_end, t2_end, llm2_start, llm2_end,
        ]);
    }

    let trajectory = exporter.export();
    // Expected: user, agent+tool_calls, merged_obs, user, agent
    assert_eq!(trajectory.steps.len(), 5);

    assert_eq!(trajectory.steps[0].source, "user");
    assert_eq!(trajectory.steps[0].step_id, 1);

    assert_eq!(trajectory.steps[1].source, "agent");
    assert_eq!(trajectory.steps[1].step_id, 2);
    let tcs = trajectory.steps[1].tool_calls.as_ref().unwrap();
    assert_eq!(tcs.len(), 2);
    assert_eq!(tcs[0].function_name, "get_weather");
    assert_eq!(tcs[1].function_name, "get_population");

    assert_eq!(trajectory.steps[2].source, "system");
    assert_eq!(trajectory.steps[2].step_id, 3);
    let obs = trajectory.steps[2].observation.as_ref().unwrap();
    assert_eq!(obs.results.len(), 2);

    assert_eq!(trajectory.steps[3].source, "user");
    assert_eq!(trajectory.steps[3].step_id, 4);

    assert_eq!(trajectory.steps[4].source, "agent");
    assert_eq!(trajectory.steps[4].step_id, 5);
    assert_eq!(
        trajectory.steps[4].message,
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
        .output(Some(json!({
            "content": "The answer is 42.",
            "role": "assistant",
            "reasoning": "Let me think step by step. The question asks for the meaning of life...",
            "token_usage": { "prompt_tokens": 10, "completion_tokens": 5 }
        })))
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
        .input(Some(json!({
            "messages": [{"role": "user", "content": "solve this"}],
            "reasoning_effort": "high"
        })))
        .build();

    let end = event_builder(llm_uuid, EventType::End)
        .scope_type(ScopeType::Llm)
        .output(Some(json!({
            "content": "Done.",
            "role": "assistant"
        })))
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
        .output(Some(json!({
            "content": "ok",
            "role": "assistant",
            "token_usage": {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "reasoning_tokens": 150,
                "cache_creation_input_tokens": 5
            }
        })))
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
        .parent_uuid(Some(agent_uuid))
        .input(Some(
            json!({"messages": [{"role": "user", "content": "hi"}]}),
        ))
        .build();

    let llm_end = event_builder(llm_uuid, EventType::End)
        .name("gpt-4")
        .scope_type(ScopeType::Llm)
        .parent_uuid(Some(agent_uuid))
        .output(Some(json!({"content": "hello", "role": "assistant"})))
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
        .input(Some(json!({"messages": []})))
        .build();

    let llm_end = event_builder(llm_uuid, EventType::End)
        .name("gpt-4")
        .scope_type(ScopeType::Llm)
        .output(Some(json!({"content": "done", "role": "assistant"})))
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
    assert_eq!(inv.framework, Some("nemo_flow".to_string()));
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
        .input(Some(
            json!({"messages": [{"role": "user", "content": "hi"}]}),
        ))
        .build();

    let llm_end = event_builder(llm_uuid, EventType::End)
        .name("gpt-4")
        .scope_type(ScopeType::Llm)
        .output(Some(json!({"content": "hi back", "role": "assistant"})))
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
        .output(Some(json!({
            "content": null,
            "role": "assistant",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "search", "arguments": "{}"}},
                {"id": "c2", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
            ]
        })))
        .build();

    let tool1_end = event_builder(tool1_uuid, EventType::End)
        .name("search")
        .scope_type(ScopeType::Tool)
        .output(Some(json!("result1")))
        .tool_call_id(Some("c1".to_string()))
        .build();

    let tool2_end = event_builder(tool2_uuid, EventType::End)
        .name("lookup")
        .scope_type(ScopeType::Tool)
        .output(Some(json!("result2")))
        .tool_call_id(Some("c2".to_string()))
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
        .output(Some(json!({
            "content": null,
            "role": "assistant",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "search", "arguments": "{}"}},
                {"id": "c2", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
            ]
        })))
        .build();

    // c2 (lookup) completes before c1 (search) — out of declaration order.
    let mut tool2_end = event_builder(tool2_uuid, EventType::End)
        .name("lookup")
        .scope_type(ScopeType::Tool)
        .output(Some(json!("result2")))
        .tool_call_id(Some("c2".to_string()))
        .build();
    let tool2_end_ts = chrono::Utc::now();
    set_event_timestamp(&mut tool2_end, tool2_end_ts);

    let mut tool1_end = event_builder(tool1_uuid, EventType::End)
        .name("search")
        .scope_type(ScopeType::Tool)
        .output(Some(json!("result1")))
        .tool_call_id(Some("c1".to_string()))
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
        .output(Some(json!({
            "content": null, "role": "assistant",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "search", "arguments": "{}"}}
            ]
        })))
        .build();
    let tool1_end = event_builder(tool1_uuid, EventType::End)
        .name("search")
        .scope_type(ScopeType::Tool)
        .output(Some(json!("result1")))
        .tool_call_id(Some("c1".to_string()))
        .build();

    // Turn 2: new LLM call + one different tool
    let llm2_start = event_builder(llm2_uuid, EventType::Start)
        .scope_type(ScopeType::Llm)
        .input(Some(json!({"messages": []})))
        .build();
    let llm2_end = event_builder(llm2_uuid, EventType::End)
        .scope_type(ScopeType::Llm)
        .output(Some(json!({
            "content": null, "role": "assistant",
            "tool_calls": [
                {"id": "c2", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
            ]
        })))
        .build();
    let tool2_end = event_builder(tool2_uuid, EventType::End)
        .name("lookup")
        .scope_type(ScopeType::Tool)
        .output(Some(json!("result2")))
        .tool_call_id(Some("c2".to_string()))
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
    // steps: agent(turn1), system(obs1), user(turn2), agent(turn2), system(obs2)
    let agent1 = trajectory
        .steps
        .iter()
        .find(|s| s.source == "agent" && s.step_id == 1)
        .unwrap();
    let agent2 = trajectory
        .steps
        .iter()
        .find(|s| s.source == "agent" && s.step_id == 4)
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
