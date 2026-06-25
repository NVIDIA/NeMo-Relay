// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for atif in the NeMo Relay core crate.

use super::*;
use crate::api::event::{
    BaseEvent, CategoryProfile, Event, EventCategory, MarkEvent, ScopeCategory, ScopeEvent,
    llm_attributes_to_strings, scope_attributes_to_strings, tool_attributes_to_strings,
};
use crate::api::llm::{LlmAttributes, LlmRequest};
use crate::api::scope::{HandleAttributes, ScopeAttributes, ScopeType};
use crate::api::tool::ToolAttributes;
use crate::codec::anthropic::AnthropicMessagesCodec;
use crate::codec::openai_chat::OpenAIChatCodec;
use crate::codec::openai_responses::OpenAIResponsesCodec;
use crate::codec::pricing::pricing_test_mutex;
use crate::codec::request::AnnotatedLlmRequest;
use crate::codec::response::{
    AnnotatedLlmResponse, CostEstimate, CostSource, PricingCatalog, PricingResolver, Usage,
    reset_active_pricing_resolver, set_active_pricing_resolver,
};
use crate::codec::traits::{LlmCodec, LlmResponseCodec};
use serde_json::json;
use std::collections::HashSet;
use std::sync::Arc;

struct ResetPricingResolverGuard;

impl Drop for ResetPricingResolverGuard {
    fn drop(&mut self) {
        let _ = reset_active_pricing_resolver();
    }
}

fn install_test_pricing(model_id: &str) {
    let catalog = PricingCatalog::from_json_str(
        &json!({
            "version": 1,
            "entries": [
                {
                    "provider": "test",
                    "model_id": model_id,
                    "pricing_as_of": "2026-06-05",
                    "pricing_source": "test",
                    "rates": {
                        "input_per_million": 0.15,
                        "output_per_million": 0.60,
                        "cache_read_per_million": 0.075
                    },
                    "prompt_cache": {
                        "read_accounting": "included_in_prompt_tokens"
                    }
                }
            ]
        })
        .to_string(),
    )
    .unwrap();
    set_active_pricing_resolver(PricingResolver::from_catalogs(vec![catalog])).unwrap();
}

fn install_two_model_test_pricing(requested_model: &str, response_model: &str) {
    let catalog = PricingCatalog::from_json_str(
        &json!({
            "version": 1,
            "entries": [
                {
                    "provider": "test",
                    "model_id": requested_model,
                    "pricing_as_of": "2026-06-05",
                    "pricing_source": "test",
                    "rates": {
                        "input_per_million": 1000.0,
                        "output_per_million": 1000.0
                    },
                    "prompt_cache": {
                        "read_accounting": "included_in_prompt_tokens"
                    }
                },
                {
                    "provider": "test",
                    "model_id": response_model,
                    "pricing_as_of": "2026-06-05",
                    "pricing_source": "test",
                    "rates": {
                        "input_per_million": 0.15,
                        "output_per_million": 0.60,
                        "cache_read_per_million": 0.075
                    },
                    "prompt_cache": {
                        "read_accounting": "included_in_prompt_tokens"
                    }
                }
            ]
        })
        .to_string(),
    )
    .unwrap();
    set_active_pricing_resolver(PricingResolver::from_catalogs(vec![catalog])).unwrap();
}

fn annotated_response_with_usage(model: &str, usage: Usage) -> AnnotatedLlmResponse {
    AnnotatedLlmResponse {
        id: None,
        model: Some(model.to_string()),
        message: None,
        tool_calls: None,
        finish_reason: None,
        usage: Some(usage),
        api_specific: None,
        extra: serde_json::Map::new(),
    }
}

fn provider_reported_cost(total: f64, currency: &str, model: &str) -> CostEstimate {
    CostEstimate {
        total: Some(total),
        currency: currency.to_string(),
        input: None,
        output: None,
        cache_read: None,
        cache_write: None,
        source: CostSource::ProviderReported,
        pricing_provider: None,
        pricing_model: Some(model.to_string()),
        pricing_as_of: None,
        pricing_source: None,
    }
}

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
    annotated_request: Option<Arc<AnnotatedLlmRequest>>,
    annotated_response: Option<Arc<AnnotatedLlmResponse>>,
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

    fn annotated_request(mut self, annotated: AnnotatedLlmRequest) -> Self {
        self.annotated_request = Some(Arc::new(annotated));
        self
    }

    fn annotated_response(mut self, annotated: AnnotatedLlmResponse) -> Self {
        self.annotated_response = Some(Arc::new(annotated));
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
                        .annotated_request_opt(self.annotated_request)
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
                        .annotated_response_opt(self.annotated_response)
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
        annotated_request: None,
        annotated_response: None,
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
        assert_atif_step_shape(step, index);
        let step_tool_call_ids = assert_atif_step_tool_calls(step);
        assert_atif_step_observation_refs(
            step,
            &step_tool_call_ids,
            &embedded_child_ids,
            trajectory.subagent_trajectories.is_some(),
        );
    }

    for child in trajectory
        .subagent_trajectories
        .as_deref()
        .unwrap_or_default()
    {
        assert_atif_v17_shape(child);
    }
}

fn assert_atif_step_shape(step: &AtifStep, index: usize) {
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
}

fn assert_atif_step_tool_calls(step: &AtifStep) -> HashSet<&str> {
    step.tool_calls
        .as_deref()
        .unwrap_or_default()
        .iter()
        .map(|tool_call| {
            assert!(!tool_call.tool_call_id.is_empty());
            assert!(!tool_call.function_name.is_empty());
            assert!(tool_call.arguments.is_object());
            tool_call.tool_call_id.as_str()
        })
        .collect()
}

fn assert_atif_step_observation_refs(
    step: &AtifStep,
    step_tool_call_ids: &HashSet<&str>,
    embedded_child_ids: &HashSet<&str>,
    has_embedded_children: bool,
) {
    let Some(observation) = &step.observation else {
        return;
    };
    for result in &observation.results {
        assert_atif_observation_result_refs(
            result,
            step_tool_call_ids,
            embedded_child_ids,
            has_embedded_children,
        );
    }
}

fn assert_atif_observation_result_refs(
    result: &AtifObservationResult,
    step_tool_call_ids: &HashSet<&str>,
    embedded_child_ids: &HashSet<&str>,
    has_embedded_children: bool,
) {
    if let Some(content) = &result.content {
        assert_atif_observation_content_value(content);
    }
    if let Some(source_call_id) = result.source_call_id.as_deref() {
        assert!(
            step_tool_call_ids.contains(source_call_id),
            "unmatched source_call_id {source_call_id}"
        );
    }
    assert_atif_subagent_refs(result, embedded_child_ids, has_embedded_children);
}

fn assert_atif_subagent_refs(
    result: &AtifObservationResult,
    embedded_child_ids: &HashSet<&str>,
    has_embedded_children: bool,
) {
    for reference in result
        .subagent_trajectory_ref
        .as_deref()
        .unwrap_or_default()
    {
        if let Some(trajectory_id) = reference.trajectory_id.as_deref()
            && has_embedded_children
        {
            assert!(
                embedded_child_ids.contains(trajectory_id),
                "unresolved embedded subagent reference {trajectory_id}"
            );
        }
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

fn assert_atif_observation_content_value(value: &serde_json::Value) {
    if value.is_string() {
        return;
    }
    if let Some(parts) = value.as_array()
        && parts.iter().all(is_atif_content_part)
    {
        return;
    }
    panic!("ATIF observation content must be a string or content-part array: {value}");
}

fn assert_structured_observation_result_extra(
    result: &AtifObservationResult,
    expected: serde_json::Value,
) {
    assert_eq!(result.content, None);
    assert_eq!(result.extra.as_ref().unwrap()["tool_result"], expected);
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
    let trajectory = exporter.export().unwrap();

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

    let trajectory = exporter.export().unwrap();
    // Tool Start is skipped, only the observation step remains
    assert_eq!(trajectory.steps.len(), 1);

    let step1 = &trajectory.steps[0];
    assert_eq!(step1.step_id, 1);
    assert_eq!(step1.source, "system");
    let obs = step1.observation.as_ref().unwrap();
    assert_eq!(obs.results.len(), 1);
    assert_eq!(obs.results[0].source_call_id, None);
    assert_structured_observation_result_extra(&obs.results[0], json!({"results": ["result1"]}));
    assert_eq!(
        obs.results[0].extra.as_ref().unwrap()["event_name"],
        json!("web_search")
    );
}

#[test]
fn test_exporter_omits_null_tool_observation_content() {
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let tool_uuid = Uuid::now_v7();

    let end = event_builder(tool_uuid, EventType::End)
        .name("noop")
        .scope_type(ScopeType::Tool)
        .output(json!(null))
        .tool_call_id("call_123")
        .build();

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.push(end);
    }

    let trajectory = exporter.export().unwrap();
    let result = &trajectory.steps[0].observation.as_ref().unwrap().results[0];

    assert_eq!(result.content, None);
    assert!(
        !serde_json::to_value(result)
            .unwrap()
            .as_object()
            .unwrap()
            .contains_key("content")
    );
}

#[test]
fn test_exporter_moves_structured_tool_close_result_to_observation_extra() {
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let tool_uuid = Uuid::now_v7();
    let close_result = json!({"status": "closed_by_turn_end"});

    let end = event_builder(tool_uuid, EventType::End)
        .name("terminal")
        .scope_type(ScopeType::Tool)
        .output(close_result.clone())
        .tool_call_id("call_123")
        .build();

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.push(end);
    }

    let trajectory = exporter.export().unwrap();
    assert_atif_v17_shape(&trajectory);

    let result = &trajectory.steps[0].observation.as_ref().unwrap().results[0];
    assert_eq!(result.content, None);
    assert_eq!(result.extra.as_ref().unwrap()["tool_result"], close_result);

    let exported = serde_json::to_value(&trajectory).unwrap();
    let content = exported["steps"][0]["observation"]["results"][0].get("content");
    assert!(content.is_none());
}

#[test]
fn test_exporter_preserves_tool_content_part_array_observation_content() {
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let tool_uuid = Uuid::now_v7();
    let content_parts = json!([
        {"type": "text", "text": "screenshot captured"},
        {"type": "image", "source": {"media_type": "image/png", "path": "artifacts/screenshot.png"}},
    ]);

    let end = event_builder(tool_uuid, EventType::End)
        .name("screenshot")
        .scope_type(ScopeType::Tool)
        .output(content_parts.clone())
        .tool_call_id("call_123")
        .build();

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.push(end);
    }

    let trajectory = exporter.export().unwrap();
    assert_atif_v17_shape(&trajectory);

    let result = &trajectory.steps[0].observation.as_ref().unwrap().results[0];
    assert_eq!(result.content, Some(content_parts));
    assert!(result.extra.as_ref().unwrap().get("tool_result").is_none());
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

    let trajectory = exporter.export().unwrap();
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
fn test_exporter_uses_raw_output_model_name_without_profile() {
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let llm_uuid = Uuid::now_v7();

    let end = event_builder(llm_uuid, EventType::End)
        .name("llm-call")
        .scope_type(ScopeType::Llm)
        .output(json!({
            "model": "raw-model",
            "content": "done",
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 3
            }
        }))
        .build();

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.push(end);
    }

    let trajectory = exporter.export().unwrap();
    let agent_step = &trajectory.steps[0];
    assert_eq!(agent_step.model_name, Some("raw-model".to_string()));
    let metrics = agent_step.metrics.as_ref().unwrap();
    assert_eq!(metrics.prompt_tokens, Some(7));
    assert_eq!(metrics.completion_tokens, Some(3));
}

#[test]
fn test_exporter_prefers_explicit_end_model_over_raw_start_model() {
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let llm_uuid = Uuid::now_v7();

    let start = event_builder(llm_uuid, EventType::Start)
        .name("llm-call")
        .scope_type(ScopeType::Llm)
        .input(json!({
            "model": "raw-start-model",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .build();

    let end = event_builder(llm_uuid, EventType::End)
        .name("llm-call")
        .scope_type(ScopeType::Llm)
        .output(json!({
            "content": "done",
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 3
            }
        }))
        .model_name("profile-end-model")
        .build();

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.push(start);
        state.events.push(end);
    }

    let trajectory = exporter.export().unwrap();
    assert_eq!(
        trajectory.steps[1].model_name,
        Some("profile-end-model".to_string())
    );
}

#[test]
fn test_extract_metrics_supports_provider_usage_payloads() {
    let openai_metrics = extract_metrics(
        &json!({
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "cost_usd": 0.001,
                "prompt_tokens_details": {
                    "cached_tokens": 4
                }
            }
        }),
        None,
        None,
        None,
    )
    .unwrap();
    assert_eq!(openai_metrics.prompt_tokens, Some(10));
    assert_eq!(openai_metrics.completion_tokens, Some(20));
    assert_eq!(openai_metrics.cached_tokens, Some(4));
    assert_eq!(
        openai_metrics.extra.as_ref().unwrap()["total_tokens"],
        json!(30)
    );
    assert_eq!(openai_metrics.cost_usd, Some(0.001));

    let responses_metrics = extract_metrics(
        &json!({
            "usage": {
                "input_tokens": 75,
                "output_tokens": 20,
                "total_tokens": 95,
                "input_tokens_details": {
                    "cached_tokens": 10
                },
                "cost_usd": 0.005
            }
        }),
        None,
        None,
        None,
    )
    .unwrap();
    assert_eq!(responses_metrics.prompt_tokens, Some(75));
    assert_eq!(responses_metrics.completion_tokens, Some(20));
    assert_eq!(responses_metrics.cached_tokens, Some(10));
    assert_eq!(
        responses_metrics.extra.as_ref().unwrap()["total_tokens"],
        json!(95)
    );
    assert_eq!(responses_metrics.cost_usd, Some(0.005));

    let anthropic_metrics = extract_metrics(
        &json!({
            "usage": {
                "input_tokens": 11,
                "output_tokens": 22,
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 5,
                "cost": { "total": 0.0042, "currency": "USD" }
            }
        }),
        None,
        None,
        None,
    )
    .unwrap();
    assert_eq!(anthropic_metrics.prompt_tokens, Some(11));
    assert_eq!(anthropic_metrics.completion_tokens, Some(22));
    assert_eq!(anthropic_metrics.cached_tokens, Some(8));
    assert_eq!(anthropic_metrics.cost_usd, Some(0.0042));

    let non_usd_metrics = extract_metrics(
        &json!({
            "usage": {
                "input_tokens": 11,
                "output_tokens": 22,
                "cost": { "total": 0.0042, "currency": "EUR" }
            }
        }),
        None,
        None,
        None,
    )
    .unwrap();
    assert_eq!(non_usd_metrics.cost_usd, None);
}

#[test]
fn test_extract_metrics_merges_usage_and_token_usage_fields() {
    let metrics = extract_metrics(
        &json!({
            "usage": {
                "prompt_tokens": 10,
                "reasoning_tokens": 7
            },
            "token_usage": {
                "completion_tokens": 20,
                "total_tokens": 30,
                "audio_tokens": 4
            }
        }),
        None,
        None,
        None,
    )
    .unwrap();

    assert_eq!(metrics.prompt_tokens, Some(10));
    assert_eq!(metrics.completion_tokens, Some(20));
    assert_eq!(metrics.extra.as_ref().unwrap()["total_tokens"], json!(30));
    assert_eq!(
        metrics.extra.as_ref().unwrap()["reasoning_tokens"],
        json!(7)
    );
    assert_eq!(metrics.extra.as_ref().unwrap()["audio_tokens"], json!(4));
}

#[test]
fn test_extract_metrics_uses_shared_cache_read_precedence() {
    let metrics = extract_metrics(
        &json!({
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "cached_tokens": 12,
                "prompt_tokens_details": {"cached_tokens": 42},
                "input_tokens_details": {"cached_tokens": 24},
                "cache_read_input_tokens": 77,
                "cache_read_tokens": 99
            }
        }),
        None,
        None,
        None,
    )
    .unwrap();

    assert_eq!(metrics.cached_tokens, Some(12));
}

#[test]
fn test_extract_metrics_filters_extracted_usage_aliases_from_extra() {
    let metrics = extract_metrics(
        &json!({
            "usage": {
                "inputTokens": 10,
                "outputTokens": 20,
                "cacheReadInputTokens": 5,
                "totalTokens": 35,
                "reasoning_tokens": 3
            }
        }),
        None,
        None,
        None,
    )
    .unwrap();

    assert_eq!(metrics.prompt_tokens, Some(10));
    assert_eq!(metrics.completion_tokens, Some(20));
    assert_eq!(metrics.cached_tokens, Some(5));
    let extra = metrics.extra.as_ref().unwrap().as_object().unwrap();
    assert!(!extra.contains_key("inputTokens"));
    assert!(!extra.contains_key("outputTokens"));
    assert!(!extra.contains_key("cacheReadInputTokens"));
    assert_eq!(extra.get("totalTokens"), Some(&json!(35)));
    assert_eq!(extra.get("reasoning_tokens"), Some(&json!(3)));
}

#[test]
fn test_reported_cost_object_blocks_model_pricing_estimation() {
    let _pricing_guard = pricing_test_mutex().lock().unwrap();
    install_test_pricing("priced-model");
    let _reset_guard = ResetPricingResolverGuard;

    let non_usd_metrics = extract_metrics(
        &json!({
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "total_tokens": 1500,
                "cost": {
                    "total": 0.42,
                    "currency": "EUR"
                }
            }
        }),
        Some("test"),
        Some("priced-model"),
        None,
    )
    .unwrap();

    assert_eq!(non_usd_metrics.cost_usd, None);

    let missing_total_metrics = extract_metrics(
        &json!({
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "total_tokens": 1500,
                "cost": {
                    "currency": "USD"
                }
            }
        }),
        Some("test"),
        Some("priced-model"),
        None,
    )
    .unwrap();

    assert_eq!(missing_total_metrics.cost_usd, None);

    let legacy_missing_currency_metrics = extract_metrics(
        &json!({
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "total_tokens": 1500,
                "cost": {
                    "total": 0.42
                }
            }
        }),
        Some("test"),
        Some("priced-model"),
        None,
    )
    .unwrap();

    assert_eq!(legacy_missing_currency_metrics.cost_usd, Some(0.42));

    let component_metrics = extract_metrics(
        &json!({
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "total_tokens": 1500,
                "cost": {
                    "currency": "usd",
                    "input": 0.25,
                    "output": 0.5,
                    "cache_read": 0.125
                }
            }
        }),
        Some("test"),
        Some("priced-model"),
        None,
    )
    .unwrap();

    assert_eq!(component_metrics.cost_usd, Some(0.875));
}

#[test]
fn test_exporter_derives_llm_cost_from_model_pricing() {
    let _pricing_guard = pricing_test_mutex().lock().unwrap();
    install_test_pricing("priced-model");
    let _reset_guard = ResetPricingResolverGuard;

    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let llm_uuid = Uuid::now_v7();

    let end = event_builder(llm_uuid, EventType::End)
        .name("gpt-4o-mini")
        .scope_type(ScopeType::Llm)
        .output(json!({
            "content": "priced response",
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "total_tokens": 1500,
                "prompt_tokens_details": {"cached_tokens": 200}
            }
        }))
        .model_name("priced-model")
        .build();

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.push(end);
    }

    let trajectory = exporter.export().unwrap();
    let metrics = trajectory.steps[0].metrics.as_ref().unwrap();
    assert_eq!(metrics.cost_usd, Some(0.000_435));
    assert_eq!(
        trajectory.final_metrics.as_ref().unwrap().total_cost_usd,
        Some(0.000_435)
    );
}

#[test]
fn test_exporter_prefers_response_model_before_requested_model_pricing() {
    let _pricing_guard = pricing_test_mutex().lock().unwrap();
    install_two_model_test_pricing("requested-model", "response-model");
    let _reset_guard = ResetPricingResolverGuard;

    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let llm_uuid = Uuid::now_v7();

    let end = event_builder(llm_uuid, EventType::End)
        .name("test")
        .scope_type(ScopeType::Llm)
        .output(json!({
            "content": "priced response",
            "model": "response-model",
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "total_tokens": 1500,
                "prompt_tokens_details": {"cached_tokens": 200}
            }
        }))
        .model_name("requested-model")
        .build();

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.push(end);
    }

    let trajectory = exporter.export().unwrap();
    let metrics = trajectory.steps[0].metrics.as_ref().unwrap();
    assert_eq!(metrics.cost_usd, Some(0.000_435));
}

#[test]
fn test_exporter_falls_back_to_requested_model_when_response_model_unpriced() {
    let _pricing_guard = pricing_test_mutex().lock().unwrap();
    install_test_pricing("requested-model");
    let _reset_guard = ResetPricingResolverGuard;

    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let llm_uuid = Uuid::now_v7();

    let end = event_builder(llm_uuid, EventType::End)
        .name("test")
        .scope_type(ScopeType::Llm)
        .output(json!({
            "content": "priced response",
            "model": "api-echoed-model",
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "total_tokens": 1500,
                "prompt_tokens_details": {"cached_tokens": 200}
            }
        }))
        .model_name("requested-model")
        .build();

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.push(end);
    }

    let trajectory = exporter.export().unwrap();
    let metrics = trajectory.steps[0].metrics.as_ref().unwrap();
    assert_eq!(metrics.cost_usd, Some(0.000_435));
}

#[test]
fn test_paired_lifecycle_uses_start_model_for_end_metrics_pricing() {
    let _pricing_guard = pricing_test_mutex().lock().unwrap();
    install_test_pricing("requested-model");
    let _reset_guard = ResetPricingResolverGuard;

    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let llm_uuid = Uuid::now_v7();

    let start = event_builder(llm_uuid, EventType::Start)
        .name("test")
        .scope_type(ScopeType::Llm)
        .input(json!({"messages": [{"role": "user", "content": "price this"}]}))
        .model_name("requested-model")
        .build();
    let end = event_builder(llm_uuid, EventType::End)
        .name("test")
        .scope_type(ScopeType::Llm)
        .output(json!({
            "content": "priced response",
            "model": "api-echoed-model",
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "total_tokens": 1500
            }
        }))
        .build();

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.push(start);
        state.events.push(end);
    }

    let trajectory = exporter.export().unwrap();
    let metrics = trajectory.steps[1].metrics.as_ref().unwrap();
    assert_eq!(metrics.cost_usd, Some(0.000_45));
}

#[test]
fn test_llm_span_candidate_uses_start_payload_model_for_metrics_pricing() {
    let _pricing_guard = pricing_test_mutex().lock().unwrap();
    install_test_pricing("requested-model");
    let _reset_guard = ResetPricingResolverGuard;

    let llm_uuid = Uuid::now_v7();
    let start = event_builder(llm_uuid, EventType::Start)
        .name("test")
        .scope_type(ScopeType::Llm)
        .input(json!({
            "model": "requested-model",
            "messages": [{"role": "user", "content": "price this"}]
        }))
        .build();
    let end = event_builder(llm_uuid, EventType::End)
        .name("test")
        .scope_type(ScopeType::Llm)
        .output(json!({
            "content": "priced response",
            "model": "api-echoed-model",
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "total_tokens": 1500
            }
        }))
        .build();

    let candidate = LlmSpanCandidate::from_events(llm_uuid, &start, &end).unwrap();

    assert_eq!(candidate.end_metrics.unwrap().cost_usd, Some(0.000_45));
}

#[test]
fn test_exporter_uses_normalized_usage_cost_before_model_pricing() {
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let llm_uuid = Uuid::now_v7();

    let end = event_builder(llm_uuid, EventType::End)
        .name("unknown-model")
        .scope_type(ScopeType::Llm)
        .output(json!({
            "content": "priced response",
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "cost": {
                    "total": 0.42,
                    "source": "provider_reported",
                    "pricing_model": "external-model",
                    "pricing_as_of": "2026-06-04"
                }
            }
        }))
        .model_name("unknown-model")
        .build();

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.push(end);
    }

    let trajectory = exporter.export().unwrap();
    let metrics = trajectory.steps[0].metrics.as_ref().unwrap();
    assert_eq!(metrics.cost_usd, Some(0.42));
    assert_eq!(
        trajectory.final_metrics.as_ref().unwrap().total_cost_usd,
        Some(0.42)
    );
}

#[test]
fn test_exporter_prefers_annotated_usage_over_raw_usage_conflicts() {
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let llm_uuid = Uuid::now_v7();

    let end = event_builder(llm_uuid, EventType::End)
        .name("test")
        .scope_type(ScopeType::Llm)
        .output(json!({
            "content": "priced response",
            "model": "raw-model",
            "usage": {
                "prompt_tokens": 999,
                "completion_tokens": 888,
                "cost": {
                    "total": 9.99,
                    "currency": "USD"
                }
            }
        }))
        .annotated_response(annotated_response_with_usage(
            "annotated-model",
            Usage {
                prompt_tokens: Some(12),
                completion_tokens: Some(34),
                total_tokens: Some(46),
                cache_read_tokens: Some(5),
                cache_write_tokens: None,
                cost: Some(provider_reported_cost(0.42, "USD", "annotated-model")),
            },
        ))
        .build();

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.push(end);
    }

    let trajectory = exporter.export().unwrap();
    let metrics = trajectory.steps[0].metrics.as_ref().unwrap();
    assert_eq!(metrics.prompt_tokens, Some(12));
    assert_eq!(metrics.completion_tokens, Some(34));
    assert_eq!(metrics.cached_tokens, Some(5));
    assert_eq!(metrics.cost_usd, Some(0.42));
}

#[test]
fn test_extract_metrics_does_not_mix_raw_cost_when_annotated_cost_is_non_usd() {
    let normalized_response = annotated_response_with_usage(
        "annotated-model",
        Usage {
            prompt_tokens: Some(12),
            completion_tokens: Some(34),
            total_tokens: Some(46),
            cache_read_tokens: None,
            cache_write_tokens: None,
            cost: Some(provider_reported_cost(0.42, "EUR", "annotated-model")),
        },
    );

    let metrics = extract_metrics(
        &json!({
            "usage": {
                "prompt_tokens": 999,
                "completion_tokens": 888,
                "cost": {
                    "total": 9.99,
                    "currency": "USD"
                }
            }
        }),
        None,
        None,
        Some(&normalized_response),
    )
    .unwrap();

    assert_eq!(metrics.prompt_tokens, Some(12));
    assert_eq!(metrics.completion_tokens, Some(34));
    assert_eq!(metrics.cost_usd, None);
}

#[test]
fn test_exporter_omits_cost_for_unknown_model_pricing() {
    let _pricing_guard = pricing_test_mutex().lock().unwrap();
    reset_active_pricing_resolver().unwrap();
    let metrics = extract_metrics(
        &json!({
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500
            }
        }),
        None,
        Some("unknown-model"),
        None,
    )
    .unwrap();

    assert_eq!(metrics.cost_usd, None);
}

#[test]
fn test_final_metrics_preserve_explicit_zero_cost_without_fabricating_tokens() {
    let final_metrics = compute_final_metrics(&[AtifStep {
        step_id: 1,
        source: "assistant".to_string(),
        message: json!("done"),
        timestamp: None,
        model_name: None,
        reasoning_effort: None,
        reasoning_content: None,
        tool_calls: None,
        observation: None,
        metrics: Some(AtifMetrics {
            prompt_tokens: None,
            completion_tokens: None,
            cached_tokens: None,
            cost_usd: Some(0.0),
            prompt_token_ids: None,
            completion_token_ids: None,
            logprobs: None,
            extra: None,
        }),
        llm_call_count: None,
        is_copied_context: None,
        extra: None,
    }])
    .unwrap();

    assert_eq!(final_metrics.total_prompt_tokens, None);
    assert_eq!(final_metrics.total_completion_tokens, None);
    assert_eq!(final_metrics.total_cached_tokens, None);
    assert_eq!(final_metrics.total_cost_usd, Some(0.0));
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

    let trajectory = exporter.export().unwrap();
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

    let trajectory = exporter.export().unwrap();
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
fn test_exporter_anthropic_messages_lifecycle_promotes_tool_use_blocks() {
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let llm_uuid = Uuid::now_v7();
    let tool_uuid = Uuid::now_v7();
    let base = base_timestamp();

    let mut start = event_builder(llm_uuid, EventType::Start)
        .name("claude-sonnet-4")
        .scope_type(ScopeType::Llm)
        .input(json!({
            "model": "claude-sonnet-4",
            "system": "Be concise.",
            "messages": [{"role": "user", "content": "Find the file."}],
            "tools": [{
                "name": "search",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}}
                }
            }]
        }))
        .model_name("claude-sonnet-4")
        .build();
    let mut end = event_builder(llm_uuid, EventType::End)
        .name("claude-sonnet-4")
        .scope_type(ScopeType::Llm)
        .output(json!({
            "id": "msg_01",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4",
            "content": [
                {"type": "text", "text": "I will search for it."},
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "search",
                    "input": {"query": "file"}
                }
            ],
            "stop_reason": "tool_use",
            "usage": {
                "input_tokens": 11,
                "output_tokens": 7,
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 5
            }
        }))
        .model_name("claude-sonnet-4")
        .build();
    let mut tool_end = event_builder(tool_uuid, EventType::End)
        .name("search")
        .scope_type(ScopeType::Tool)
        .parent_uuid(llm_uuid)
        .tool_call_id("toolu_01")
        .output(json!({"matches": ["README.md"]}))
        .build();

    for (offset, event) in [&mut start, &mut end, &mut tool_end]
        .into_iter()
        .enumerate()
    {
        set_event_timestamp(event, base + chrono::Duration::milliseconds(offset as i64));
    }

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.extend([start, end, tool_end]);
    }

    let trajectory = exporter.export().unwrap();
    assert_atif_v17_shape(&trajectory);
    assert_eq!(trajectory.steps.len(), 2);

    let user_step = &trajectory.steps[0];
    assert_eq!(user_step.source, "user");
    assert_eq!(user_step.message, json!("Find the file."));
    let user_extra: AtifStepExtra =
        serde_json::from_value(user_step.extra.clone().unwrap()).unwrap();
    let llm_request = user_extra.llm_request.unwrap();
    assert_eq!(llm_request["system"], json!("Be concise."));
    assert_eq!(llm_request["tools"][0]["name"], json!("search"));

    let agent_step = &trajectory.steps[1];
    assert_eq!(agent_step.source, "agent");
    assert_eq!(agent_step.message, json!("I will search for it."));
    assert_eq!(agent_step.model_name, Some("claude-sonnet-4".to_string()));
    let metrics = agent_step.metrics.as_ref().unwrap();
    assert_eq!(metrics.prompt_tokens, Some(11));
    assert_eq!(metrics.completion_tokens, Some(7));
    assert_eq!(metrics.cached_tokens, Some(8));
    let tool_call = &agent_step.tool_calls.as_ref().unwrap()[0];
    assert_eq!(tool_call.tool_call_id, "toolu_01");
    assert_eq!(tool_call.function_name, "search");
    assert_eq!(tool_call.arguments, json!({"query": "file"}));
    let observation = agent_step.observation.as_ref().unwrap();
    assert_eq!(
        observation.results[0].source_call_id.as_deref(),
        Some("toolu_01")
    );
    assert_structured_observation_result_extra(
        &observation.results[0],
        json!({"matches": ["README.md"]}),
    );
}

#[test]
fn test_exporter_openclaw_placeholder_replay_preserves_empty_user_step_and_raw_request() {
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let llm_uuid = Uuid::now_v7();
    let base = base_timestamp();

    let mut start = event_builder(llm_uuid, EventType::Start)
        .name("openclaw-model-call")
        .scope_type(ScopeType::Llm)
        .input(json!({
            "headers": {},
            "content": {
                "provider": "nvidia-inference",
                "model": "claude-sonnet-4",
                "prompt": "",
                "messages": [],
                "imagesCount": 0,
                "placeholderRequest": true,
                "source": "openclaw.llm_output"
            }
        }))
        .model_name("claude-sonnet-4")
        .build();
    let mut end = event_builder(llm_uuid, EventType::End)
        .name("openclaw-model-call")
        .scope_type(ScopeType::Llm)
        .output(json!({
            "role": "assistant",
            "content": "I will search.",
            "assistant_texts_count": 1,
            "openclaw": {
                "assistant_tool_call_names": []
            }
        }))
        .model_name("claude-sonnet-4")
        .build();

    for (offset, event) in [&mut start, &mut end].into_iter().enumerate() {
        set_event_timestamp(event, base + chrono::Duration::milliseconds(offset as i64));
    }

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.extend([start, end]);
    }

    let trajectory = exporter.export().unwrap();
    assert_atif_v17_shape(&trajectory);
    assert_eq!(trajectory.steps.len(), 2);

    let user_step = &trajectory.steps[0];
    assert_eq!(user_step.source, "user");
    assert_eq!(user_step.message, json!(""));
    let user_extra: AtifStepExtra =
        serde_json::from_value(user_step.extra.clone().unwrap()).unwrap();
    let llm_request = user_extra.llm_request.unwrap();
    assert!(llm_request.get("headers").is_none());
    assert_eq!(llm_request["placeholderRequest"], json!(true));
    assert_eq!(llm_request["messages"], json!([]));
    assert_eq!(llm_request["prompt"], json!(""));
    assert_eq!(llm_request["source"], json!("openclaw.llm_output"));

    let agent_step = &trajectory.steps[1];
    assert_eq!(agent_step.source, "agent");
    assert_eq!(agent_step.message, json!("I will search."));
    assert_eq!(agent_step.model_name, Some("claude-sonnet-4".to_string()));
    let agent_extra: AtifStepExtra =
        serde_json::from_value(agent_step.extra.clone().unwrap()).unwrap();
    let llm_response = agent_extra.llm_response.unwrap();
    assert_eq!(llm_response["role"], json!("assistant"));
    assert_eq!(llm_response["content"], json!("I will search."));
}

#[test]
fn test_exporter_openclaw_timing_marks_become_system_steps_with_payloads() {
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let base = base_timestamp();

    let ambiguous_payload = json!({
        "runId": "run-1",
        "sessionId": "session-1",
        "provider": "openai",
        "model": "gpt-4",
        "candidateCount": 2
    });
    let unpaired_payload = json!({
        "runId": "run-1",
        "callId": "call-1",
        "provider": "openai",
        "model": "gpt-4",
        "durationMs": 42,
        "outcome": "completed"
    });

    let mut ambiguous = event_builder(Uuid::now_v7(), EventType::Mark)
        .name("openclaw.model_call_timing_ambiguous")
        .data(ambiguous_payload.clone())
        .build();
    let mut unpaired = event_builder(Uuid::now_v7(), EventType::Mark)
        .name("openclaw.model_call_timing_unpaired")
        .data(unpaired_payload.clone())
        .build();

    set_event_timestamp(&mut ambiguous, base);
    set_event_timestamp(&mut unpaired, base + chrono::Duration::milliseconds(1));

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.extend([ambiguous, unpaired]);
    }

    let trajectory = exporter.export().unwrap();
    assert_atif_v17_shape(&trajectory);
    assert_eq!(trajectory.steps.len(), 2);

    let ambiguous_step = &trajectory.steps[0];
    assert_eq!(ambiguous_step.source, "system");
    assert_eq!(
        ambiguous_step.message,
        json!("openclaw.model_call_timing_ambiguous")
    );
    let ambiguous_extra: AtifStepExtra =
        serde_json::from_value(ambiguous_step.extra.clone().unwrap()).unwrap();
    assert_eq!(
        ambiguous_extra.event_payload.as_ref(),
        Some(&ambiguous_payload)
    );

    let unpaired_step = &trajectory.steps[1];
    assert_eq!(unpaired_step.source, "system");
    assert_eq!(
        unpaired_step.message,
        json!("openclaw.model_call_timing_unpaired")
    );
    let unpaired_extra: AtifStepExtra =
        serde_json::from_value(unpaired_step.extra.clone().unwrap()).unwrap();
    assert_eq!(
        unpaired_extra.event_payload.as_ref(),
        Some(&unpaired_payload)
    );
}

#[test]
fn test_exporter_openclaw_hook_only_fallbacks_preserve_stripped_content_and_explicit_metrics() {
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let stripped_uuid = Uuid::now_v7();
    let partial_uuid = Uuid::now_v7();
    let base = base_timestamp();

    let mut stripped_start = event_builder(stripped_uuid, EventType::Start)
        .name("openclaw-model-call")
        .scope_type(ScopeType::Llm)
        .input(json!({
            "headers": {},
            "content": {
                "provider": "openai",
                "model": "gpt-4",
                "messages": [],
                "imagesCount": 1,
                "source": "openclaw.llm_output"
            }
        }))
        .model_name("gpt-4")
        .build();
    let mut stripped_end = event_builder(stripped_uuid, EventType::End)
        .name("openclaw-model-call")
        .scope_type(ScopeType::Llm)
        .output(json!({
            "role": "assistant",
            "assistant_texts_count": 1,
            "usage": {
                "cost_usd": 0.001
            },
            "openclaw": {
                "assistant_tool_call_names": []
            }
        }))
        .model_name("gpt-4")
        .build();
    let mut partial_start = event_builder(partial_uuid, EventType::Start)
        .name("openclaw-model-call")
        .scope_type(ScopeType::Llm)
        .input(json!({
            "headers": {},
            "content": {
                "provider": "openai",
                "model": "gpt-4",
                "prompt": "visible prompt",
                "messages": [{"role": "user", "content": "visible prompt"}],
                "imagesCount": 0,
                "source": "openclaw.llm_output"
            }
        }))
        .model_name("gpt-4")
        .build();
    let mut partial_end = event_builder(partial_uuid, EventType::End)
        .name("openclaw-model-call")
        .scope_type(ScopeType::Llm)
        .output(json!({
            "role": "assistant",
            "content": "visible answer",
            "usage": {
                "prompt_tokens": 42
            },
            "openclaw": {
                "assistant_tool_call_names": []
            }
        }))
        .model_name("gpt-4")
        .build();

    for (offset, event) in [
        &mut stripped_start,
        &mut stripped_end,
        &mut partial_start,
        &mut partial_end,
    ]
    .into_iter()
    .enumerate()
    {
        set_event_timestamp(event, base + chrono::Duration::milliseconds(offset as i64));
    }

    {
        let mut state = exporter.state.lock().unwrap();
        state
            .events
            .extend([stripped_start, stripped_end, partial_start, partial_end]);
    }

    let trajectory = exporter.export().unwrap();
    assert_atif_v17_shape(&trajectory);
    assert_eq!(trajectory.steps.len(), 4);

    let stripped_user = &trajectory.steps[0];
    assert_eq!(stripped_user.source, "user");
    let stripped_user_message: serde_json::Value =
        serde_json::from_str(stripped_user.message.as_str().unwrap()).unwrap();
    assert_eq!(stripped_user_message["provider"], json!("openai"));
    assert_eq!(stripped_user_message["model"], json!("gpt-4"));
    assert_eq!(stripped_user_message["messages"], json!([]));
    assert_eq!(stripped_user_message["imagesCount"], json!(1));
    assert_eq!(
        stripped_user_message["source"],
        json!("openclaw.llm_output")
    );
    assert!(stripped_user_message.get("prompt").is_none());
    assert!(stripped_user_message.get("systemPrompt").is_none());
    let stripped_user_extra: AtifStepExtra =
        serde_json::from_value(stripped_user.extra.clone().unwrap()).unwrap();
    let stripped_request = stripped_user_extra.llm_request.unwrap();
    assert!(stripped_request.get("prompt").is_none());
    assert!(stripped_request.get("systemPrompt").is_none());
    assert_eq!(stripped_request["messages"], json!([]));
    assert_eq!(stripped_request["imagesCount"], json!(1));

    let stripped_agent = &trajectory.steps[1];
    assert_eq!(stripped_agent.source, "agent");
    let stripped_message: serde_json::Value =
        serde_json::from_str(stripped_agent.message.as_str().unwrap()).unwrap();
    assert_eq!(stripped_message["assistant_texts_count"], json!(1));
    assert!(stripped_message.get("content").is_none());
    let stripped_metrics = stripped_agent.metrics.as_ref().unwrap();
    assert_eq!(stripped_metrics.prompt_tokens, None);
    assert_eq!(stripped_metrics.completion_tokens, None);
    assert_eq!(stripped_metrics.cached_tokens, None);
    assert_eq!(stripped_metrics.cost_usd, Some(0.001));
    let stripped_agent_extra: AtifStepExtra =
        serde_json::from_value(stripped_agent.extra.clone().unwrap()).unwrap();
    let stripped_response = stripped_agent_extra.llm_response.unwrap();
    assert!(stripped_response.get("content").is_none());
    assert_eq!(stripped_response["assistant_texts_count"], json!(1));

    let partial_user = &trajectory.steps[2];
    assert_eq!(partial_user.source, "user");
    assert_eq!(partial_user.message, json!("visible prompt"));
    let partial_user_extra: AtifStepExtra =
        serde_json::from_value(partial_user.extra.clone().unwrap()).unwrap();
    let partial_request = partial_user_extra.llm_request.unwrap();
    assert_eq!(partial_request["prompt"], json!("visible prompt"));
    assert_eq!(
        partial_request["messages"][0]["content"],
        json!("visible prompt")
    );

    let partial_agent = &trajectory.steps[3];
    assert_eq!(partial_agent.source, "agent");
    assert_eq!(partial_agent.message, json!("visible answer"));
    let partial_metrics = partial_agent.metrics.as_ref().unwrap();
    assert_eq!(partial_metrics.prompt_tokens, Some(42));
    assert_eq!(partial_metrics.completion_tokens, None);
    assert_eq!(partial_metrics.cached_tokens, None);
    assert_eq!(partial_metrics.cost_usd, None);

    let final_metrics = trajectory.final_metrics.as_ref().unwrap();
    assert_eq!(final_metrics.total_prompt_tokens, Some(42));
    assert_eq!(final_metrics.total_completion_tokens, None);
    assert_eq!(final_metrics.total_cached_tokens, None);
    assert_eq!(final_metrics.total_cost_usd, Some(0.001));
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

    let trajectory = exporter.export().unwrap();
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
    assert_structured_observation_result_extra(result, json!({"stdout": "/workspace"}));
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

    let trajectory = exporter.export().unwrap();
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
fn test_exporter_promotes_tool_name_alias_tool_calls() {
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());

    let end = event_builder(Uuid::now_v7(), EventType::End)
        .name("gpt-4")
        .scope_type(ScopeType::Llm)
        .output(json!({
            "role": "assistant",
            "tool_calls": [{
                "toolName": "search_docs",
                "arguments": {"query": "docs"}
            }]
        }))
        .build();

    exporter.state.lock().unwrap().events.push(end);

    let trajectory = exporter.export().unwrap();
    let tool_calls = trajectory.steps[0].tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].tool_call_id, "search_docs:1");
    assert_eq!(tool_calls[0].function_name, "search_docs");
    assert_eq!(tool_calls[0].arguments, json!({"query": "docs"}));
}

#[test]
fn test_exporter_uses_annotated_message_but_raw_tool_calls() {
    // The annotation supplies the normalized message text (winning over the raw
    // content), while tool calls stay on the raw path so provider-specific
    // extras are preserved.
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let annotated = OpenAIChatCodec
        .decode_response(&json!({
            "choices": [{
                "message": {"role": "assistant", "content": "from annotation"},
                "finish_reason": "stop"
            }]
        }))
        .unwrap();

    let end = event_builder(Uuid::now_v7(), EventType::End)
        .name("gpt-4")
        .scope_type(ScopeType::Llm)
        .output(json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "from RAW",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "provider_data": {"trace_id": "t-1"},
                        "function": {"name": "search", "arguments": "{\"q\":\"x\"}"}
                    }]
                }
            }]
        }))
        .annotated_response(annotated)
        .build();
    exporter.state.lock().unwrap().events.push(end);

    let trajectory = exporter.export().unwrap();
    let step = &trajectory.steps[0];
    // Message comes from the annotation.
    assert_eq!(step.message, json!("from annotation"));
    // Tool calls come from the raw payload, with provider extras preserved.
    let tool_calls = step.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls[0].function_name, "search");
    assert_eq!(tool_calls[0].arguments, json!({"q": "x"}));
    assert_eq!(
        tool_calls[0].extra.as_ref().unwrap()["provider_data"]["trace_id"],
        json!("t-1")
    );
}

#[test]
fn test_exporter_annotated_tool_only_response_renders_empty_message() {
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let annotated = OpenAIChatCodec
        .decode_response(&json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }))
        .unwrap();

    let end = event_builder(Uuid::now_v7(), EventType::End)
        .name("gpt-4")
        .scope_type(ScopeType::Llm)
        .output(json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"}
                    }]
                }
            }]
        }))
        .annotated_response(annotated)
        .build();
    exporter.state.lock().unwrap().events.push(end);

    let trajectory = exporter.export().unwrap();
    let step = &trajectory.steps[0];
    assert_eq!(step.message, json!(""));
    assert_eq!(step.tool_calls.as_ref().unwrap()[0].function_name, "lookup");
}

#[test]
fn test_exporter_annotated_reasoning_only_response_renders_empty_message() {
    // An OpenAI Responses reasoning-only output decodes to no assistant text.
    // The annotated path emits an empty message rather than the raw extractor's
    // stringified payload.
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let output = json!({
        "model": "gpt-4o",
        "output": [{"type": "reasoning", "summary": []}],
        "status": "completed"
    });
    let annotated = OpenAIResponsesCodec.decode_response(&output).unwrap();
    assert!(annotated.message.is_none());

    let end = event_builder(Uuid::now_v7(), EventType::End)
        .name("gpt-4o")
        .scope_type(ScopeType::Llm)
        .output(output)
        .annotated_response(annotated)
        .build();
    exporter.state.lock().unwrap().events.push(end);

    let trajectory = exporter.export().unwrap();
    assert_eq!(trajectory.steps[0].message, json!(""));
}

#[test]
fn test_exporter_annotated_thinking_only_response_renders_empty_message() {
    // An Anthropic thinking-only response decodes to no assistant text; the
    // annotated path emits an empty message rather than dumping the raw blocks.
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let output = json!({
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-sonnet",
        "content": [{"type": "thinking", "thinking": "hmm", "signature": "sig"}],
        "stop_reason": "end_turn"
    });
    let annotated = AnthropicMessagesCodec.decode_response(&output).unwrap();
    assert!(annotated.message.is_none());

    let end = event_builder(Uuid::now_v7(), EventType::End)
        .name("claude-3-5-sonnet")
        .scope_type(ScopeType::Llm)
        .output(output)
        .annotated_response(annotated)
        .build();
    exporter.state.lock().unwrap().events.push(end);

    let trajectory = exporter.export().unwrap();
    assert_eq!(trajectory.steps[0].message, json!(""));
}

#[test]
fn test_exporter_prefers_annotated_request_message() {
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hello from annotation"}]
        }),
    };
    let annotated = OpenAIChatCodec.decode(&request).unwrap();

    let start = event_builder(Uuid::now_v7(), EventType::Start)
        .name("gpt-4")
        .scope_type(ScopeType::Llm)
        .input(json!({"messages": [{"role": "user", "content": "from RAW"}]}))
        .annotated_request(annotated)
        .build();
    exporter.state.lock().unwrap().events.push(start);

    let trajectory = exporter.export().unwrap();
    let step = &trajectory.steps[0];
    assert_eq!(step.source, "user");
    assert_eq!(step.message, json!("hello from annotation"));
}

#[test]
fn test_exporter_annotated_multimodal_request_falls_back_to_raw() {
    // A multimodal (content-part) request message must not be flattened to text:
    // the annotation adapter returns None and ATIF preserves the raw content.
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let multimodal = json!({
        "model": "gpt-4",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": "https://example/img.png"}}
        ]}]
    });
    let annotated = OpenAIChatCodec
        .decode(&LlmRequest {
            headers: serde_json::Map::new(),
            content: multimodal.clone(),
        })
        .unwrap();

    let start = event_builder(Uuid::now_v7(), EventType::Start)
        .name("gpt-4")
        .scope_type(ScopeType::Llm)
        .input(multimodal)
        .annotated_request(annotated)
        .build();
    exporter.state.lock().unwrap().events.push(start);

    let trajectory = exporter.export().unwrap();
    let message = trajectory.steps[0].message.as_str().unwrap();
    // Raw fallback preserves the image part rather than flattening to "describe".
    assert!(
        message.contains("image_url"),
        "multimodal content preserved: {message}"
    );
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

    let trajectory = exporter.export().unwrap();
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
    assert_structured_observation_result_extra(
        &observation.results[0],
        json!({"stdout": "hi", "exit_code": 0}),
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

    let trajectory = exporter.export().unwrap();
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

    let trajectory = exporter.export().unwrap();
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

    let trajectory = exporter.export().unwrap();
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

    let trajectory = exporter.export().unwrap();
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

    let trajectory = exporter.export().unwrap();
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
    assert_structured_observation_result_extra(result, json!({"status": "launched"}));
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

    let trajectory = exporter.export().unwrap();
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
    assert_structured_observation_result_extra(result, json!({"status": "completed"}));
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
fn test_exporter_keeps_structured_tool_observation_on_agent_scope_tree() {
    let root_uuid = Uuid::now_v7();
    let turn_uuid = Uuid::now_v7();
    let llm_uuid = Uuid::now_v7();
    let tool_uuid = Uuid::now_v7();
    let exporter = AtifExporter::new(root_uuid.to_string(), make_agent_info());
    let base = base_timestamp();

    let mut root_start = event_builder(root_uuid, EventType::Start)
        .name("hermes")
        .scope_type(ScopeType::Agent)
        .metadata(json!({
            "nemo_relay_scope_role": "session",
            "session_id": "session-1",
        }))
        .build();
    let mut turn_start = event_builder(turn_uuid, EventType::Start)
        .name("hermes-turn")
        .parent_uuid(root_uuid)
        .scope_type(ScopeType::Agent)
        .metadata(json!({
            "nemo_relay_scope_role": "turn",
            "turn_index": 1,
            "turn_source": "implicit",
        }))
        .build();
    let mut llm_start = event_builder(llm_uuid, EventType::Start)
        .name("custom")
        .parent_uuid(turn_uuid)
        .scope_type(ScopeType::Llm)
        .input(json!({
            "messages": [{"role": "user", "content": "search for needle"}],
            "model": "qwen",
        }))
        .model_name("qwen")
        .build();
    let mut llm_end = event_builder(llm_uuid, EventType::End)
        .name("custom")
        .parent_uuid(turn_uuid)
        .scope_type(ScopeType::Llm)
        .output(json!({
            "content": "",
            "role": "assistant",
            "tool_calls": [{
                "id": "call-search-1",
                "type": "function",
                "function": {
                    "name": "search_files",
                    "arguments": "{\"query\":\"needle\"}",
                }
            }]
        }))
        .model_name("qwen")
        .build();
    let mut tool_start = event_builder(tool_uuid, EventType::Start)
        .name("search_files")
        .parent_uuid(turn_uuid)
        .scope_type(ScopeType::Tool)
        .input(json!({"query": "needle"}))
        .tool_call_id("call-search-1")
        .build();
    let mut tool_end = event_builder(tool_uuid, EventType::End)
        .name("search_files")
        .parent_uuid(turn_uuid)
        .scope_type(ScopeType::Tool)
        .output(json!({"total_count": 6}))
        .tool_call_id("call-search-1")
        .build();
    let mut turn_end = event_builder(turn_uuid, EventType::End)
        .name("hermes-turn")
        .parent_uuid(root_uuid)
        .scope_type(ScopeType::Agent)
        .build();
    let mut root_end = event_builder(root_uuid, EventType::End)
        .name("hermes")
        .scope_type(ScopeType::Agent)
        .build();

    for (offset, event) in [
        &mut root_start,
        &mut turn_start,
        &mut llm_start,
        &mut llm_end,
        &mut tool_start,
        &mut tool_end,
        &mut turn_end,
        &mut root_end,
    ]
    .into_iter()
    .enumerate()
    {
        set_event_timestamp(event, base + chrono::Duration::milliseconds(offset as i64));
    }

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.extend([
            root_start, turn_start, llm_start, llm_end, tool_start, tool_end, turn_end, root_end,
        ]);
    }

    let trajectory = exporter.export().unwrap();
    assert_atif_v17_shape(&trajectory);
    assert_eq!(trajectory.steps.len(), 2);

    let agent_step = &trajectory.steps[1];
    let observation = agent_step.observation.as_ref().unwrap();
    assert_eq!(observation.results.len(), 1);
    assert_eq!(
        observation.results[0].source_call_id.as_deref(),
        Some("call-search-1")
    );
    assert_structured_observation_result_extra(&observation.results[0], json!({"total_count": 6}));
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

    let trajectory = exporter.export().unwrap();
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

    let trajectory = exporter.export().unwrap();
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

    let trajectory = exporter.export().unwrap();
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

    let trajectory = exporter.export().unwrap();
    assert!(trajectory.steps.is_empty());
    assert_eq!(
        trajectory.final_metrics.as_ref().unwrap().total_steps,
        Some(0)
    );
}

#[test]
fn test_exporter_skips_llm_chunk_marks() {
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.push(
            event_builder(Uuid::now_v7(), EventType::Mark)
                .name("llm.chunk")
                .data(json!({"delta": "partial"}))
                .build(),
        );
        state.events.push(
            event_builder(Uuid::now_v7(), EventType::Mark)
                .name("hook_mark")
                .metadata(json!({"hook_event_name": "llm.chunk"}))
                .data(json!({"delta": "partial"}))
                .build(),
        );
        state.events.push(
            event_builder(Uuid::now_v7(), EventType::Mark)
                .name("agent.status")
                .data(json!({"status": "ok"}))
                .build(),
        );
    }

    let trajectory = exporter.export().unwrap();

    assert_eq!(trajectory.steps.len(), 1);
    assert_eq!(trajectory.steps[0].message, json!("agent.status"));
}

#[test]
fn test_exporter_dedupes_overlapping_hook_and_gateway_llm_spans() {
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let base = base_timestamp();
    let parent_uuid = Uuid::now_v7();
    let hook_uuid = Uuid::now_v7();
    let gateway_uuid = Uuid::now_v7();
    let request = json!({
        "messages": [{"role": "user", "content": "Reply with exactly dedupe_ok"}],
        "model": "test-model"
    });

    let mut hook_start = event_builder(hook_uuid, EventType::Start)
        .name("openrouter")
        .parent_uuid(parent_uuid)
        .scope_type(ScopeType::Llm)
        .model_name("test-model")
        .metadata(json!({
            "hook_event_name": "pre_api_request",
            "api_call_id": "session:task:abcd:api:1",
            "provider_payload_exact": true
        }))
        .input(json!({"content": request.clone(), "headers": {}}))
        .build();
    let mut gateway_start = event_builder(gateway_uuid, EventType::Start)
        .name("openai.chat_completions")
        .parent_uuid(parent_uuid)
        .scope_type(ScopeType::Llm)
        .model_name("test-model")
        .metadata(json!({
            "gateway_path": "/v1/chat/completions",
            "llm_correlation_source": "pre_llm_call"
        }))
        .input(request.clone())
        .build();
    let mut gateway_end = event_builder(gateway_uuid, EventType::End)
        .name("openai.chat_completions")
        .parent_uuid(parent_uuid)
        .scope_type(ScopeType::Llm)
        .model_name("test-model")
        .metadata(json!({
            "gateway_path": "/v1/chat/completions",
            "llm_correlation_source": "pre_llm_call"
        }))
        .output(json!({
            "choices": [{"message": {"content": "dedupe_ok"}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
        }))
        .build();
    let mut hook_end = event_builder(hook_uuid, EventType::End)
        .name("openrouter")
        .parent_uuid(parent_uuid)
        .scope_type(ScopeType::Llm)
        .model_name("test-model")
        .metadata(json!({
            "hook_event_name": "post_api_request",
            "api_call_id": "session:task:abcd:api:1",
            "provider_payload_exact": true
        }))
        .output(json!({"content": "dedupe_ok"}))
        .build();

    for (idx, event) in [
        &mut hook_start,
        &mut gateway_start,
        &mut gateway_end,
        &mut hook_end,
    ]
    .into_iter()
    .enumerate()
    {
        set_event_timestamp(event, base + chrono::Duration::milliseconds(idx as i64));
    }

    {
        let mut state = exporter.state.lock().unwrap();
        state
            .events
            .extend([hook_start, gateway_start, gateway_end, hook_end]);
    }

    let trajectory = exporter.export().unwrap();
    assert_eq!(trajectory.steps.len(), 2);
    assert_eq!(
        trajectory.steps[0].message,
        json!("Reply with exactly dedupe_ok")
    );
    assert_eq!(trajectory.steps[1].message, json!("dedupe_ok"));
    assert_eq!(
        trajectory.steps[0].extra.as_ref().unwrap()["ancestry"]["function_name"],
        "openrouter"
    );
    assert_eq!(
        trajectory.steps[1].extra.as_ref().unwrap()["ancestry"]["function_name"],
        "openrouter"
    );
    assert!(!trajectory.steps.iter().any(|step| {
        step.extra
            .as_ref()
            .is_some_and(|extra| extra["ancestry"]["function_name"] == "openai.chat_completions")
    }));
    let metrics = trajectory.steps[1].metrics.as_ref().unwrap();
    assert_eq!(metrics.prompt_tokens, Some(7));
    assert_eq!(metrics.completion_tokens, Some(3));
    assert_eq!(metrics.extra.as_ref().unwrap()["total_tokens"], json!(10));
}

#[test]
fn test_exporter_keeps_tool_only_llm_spans_with_different_tool_calls() {
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let base = base_timestamp();
    let parent_uuid = Uuid::now_v7();
    let first_uuid = Uuid::now_v7();
    let second_uuid = Uuid::now_v7();
    let request = json!({
        "messages": [{"role": "user", "content": "Use tools"}],
        "model": "test-model"
    });

    let mut first_start = event_builder(first_uuid, EventType::Start)
        .name("anthropic.messages")
        .parent_uuid(parent_uuid)
        .scope_type(ScopeType::Llm)
        .model_name("test-model")
        .input(request.clone())
        .build();
    let mut first_end = event_builder(first_uuid, EventType::End)
        .name("anthropic.messages")
        .parent_uuid(parent_uuid)
        .scope_type(ScopeType::Llm)
        .model_name("test-model")
        .output(json!({
            "type": "message",
            "content": [{
                "type": "tool_use",
                "id": "toolu-read",
                "name": "read_file",
                "input": { "path": "README.md" }
            }]
        }))
        .build();
    let mut second_start = event_builder(second_uuid, EventType::Start)
        .name("anthropic.messages")
        .parent_uuid(parent_uuid)
        .scope_type(ScopeType::Llm)
        .model_name("test-model")
        .input(request)
        .build();
    let mut second_end = event_builder(second_uuid, EventType::End)
        .name("anthropic.messages")
        .parent_uuid(parent_uuid)
        .scope_type(ScopeType::Llm)
        .model_name("test-model")
        .output(json!({
            "type": "message",
            "content": [{
                "type": "tool_use",
                "id": "toolu-search",
                "name": "web_search",
                "input": { "query": "release notes" }
            }]
        }))
        .build();

    for (idx, event) in [
        &mut first_start,
        &mut second_start,
        &mut first_end,
        &mut second_end,
    ]
    .into_iter()
    .enumerate()
    {
        set_event_timestamp(event, base + chrono::Duration::milliseconds(idx as i64));
    }

    {
        let mut state = exporter.state.lock().unwrap();
        state
            .events
            .extend([first_start, second_start, first_end, second_end]);
    }

    let trajectory = exporter.export().unwrap();
    let tool_call_ids = trajectory
        .steps
        .iter()
        .filter_map(|step| step.tool_calls.as_deref())
        .flat_map(|calls| calls.iter().map(|call| call.tool_call_id.as_str()))
        .collect::<HashSet<_>>();

    assert!(tool_call_ids.contains("toolu-read"));
    assert!(tool_call_ids.contains("toolu-search"));
    assert_eq!(tool_call_ids.len(), 2);
}

#[test]
fn test_exporter_prefers_gateway_span_over_non_exact_hook_summary() {
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let base = base_timestamp();
    let parent_uuid = Uuid::now_v7();
    let hook_uuid = Uuid::now_v7();
    let gateway_uuid = Uuid::now_v7();
    let request = json!({
        "messages": [{"role": "user", "content": "Reply with exactly gateway_ok"}],
        "model": "test-model"
    });

    let mut hook_start = event_builder(hook_uuid, EventType::Start)
        .name("openrouter")
        .parent_uuid(parent_uuid)
        .scope_type(ScopeType::Llm)
        .model_name("test-model")
        .metadata(json!({
            "hook_event_name": "pre_api_request",
            "api_call_id": "request-1",
            "fidelity_source": "agent_api_hooks",
            "provider_payload_exact": false
        }))
        .input(json!({
            "content": {"message_count": 2, "request_char_count": 128},
            "headers": {}
        }))
        .build();
    let mut gateway_start = event_builder(gateway_uuid, EventType::Start)
        .name("openai.chat_completions")
        .parent_uuid(parent_uuid)
        .scope_type(ScopeType::Llm)
        .model_name("test-model")
        .metadata(json!({
            "gateway_path": "/v1/chat/completions",
            "llm_correlation_request_id": "request-1"
        }))
        .input(json!({"content": request.clone(), "headers": {}}))
        .build();
    let mut gateway_end = event_builder(gateway_uuid, EventType::End)
        .name("openai.chat_completions")
        .parent_uuid(parent_uuid)
        .scope_type(ScopeType::Llm)
        .model_name("test-model")
        .metadata(json!({
            "gateway_path": "/v1/chat/completions",
            "llm_correlation_request_id": "request-1"
        }))
        .output(json!({"choices": [{"message": {"content": "gateway_ok"}}]}))
        .build();
    let mut hook_end = event_builder(hook_uuid, EventType::End)
        .name("openrouter")
        .parent_uuid(parent_uuid)
        .scope_type(ScopeType::Llm)
        .model_name("test-model")
        .metadata(json!({
            "hook_event_name": "post_api_request",
            "api_call_id": "request-1",
            "fidelity_source": "agent_api_hooks",
            "provider_payload_exact": false
        }))
        .output(json!({"assistant_content_chars": 10, "finish_reason": "stop"}))
        .build();

    for (idx, event) in [
        &mut hook_start,
        &mut gateway_start,
        &mut gateway_end,
        &mut hook_end,
    ]
    .into_iter()
    .enumerate()
    {
        set_event_timestamp(event, base + chrono::Duration::milliseconds(idx as i64));
    }

    {
        let mut state = exporter.state.lock().unwrap();
        state
            .events
            .extend([hook_start, gateway_start, gateway_end, hook_end]);
    }

    let trajectory = exporter.export().unwrap();
    assert_eq!(trajectory.steps.len(), 2);
    assert_eq!(
        trajectory.steps[0].message,
        json!("Reply with exactly gateway_ok")
    );
    assert_eq!(trajectory.steps[1].message, json!("gateway_ok"));
    assert!(trajectory.steps.iter().all(|step| {
        step.extra
            .as_ref()
            .is_some_and(|extra| extra["ancestry"]["function_name"] == "openai.chat_completions")
    }));
}

#[test]
fn test_exporter_keeps_overlapping_non_exact_hook_and_gateway_spans_without_shared_request_key() {
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let base = base_timestamp();
    let parent_uuid = Uuid::now_v7();
    let hook_uuid = Uuid::now_v7();
    let gateway_uuid = Uuid::now_v7();
    let request = json!({
        "messages": [{"role": "user", "content": "Reply with exactly gateway_distinct_ok"}],
        "model": "test-model"
    });

    let mut hook_start = event_builder(hook_uuid, EventType::Start)
        .name("openrouter")
        .parent_uuid(parent_uuid)
        .scope_type(ScopeType::Llm)
        .model_name("test-model")
        .metadata(json!({
            "hook_event_name": "pre_api_request",
            "api_call_id": "hook-request",
            "fidelity_source": "agent_api_hooks",
            "provider_payload_exact": false
        }))
        .input(json!({
            "content": {"message_count": 2, "request_char_count": 128},
            "headers": {}
        }))
        .build();
    let mut gateway_start = event_builder(gateway_uuid, EventType::Start)
        .name("openai.chat_completions")
        .parent_uuid(parent_uuid)
        .scope_type(ScopeType::Llm)
        .model_name("test-model")
        .metadata(json!({
            "gateway_path": "/v1/chat/completions",
            "llm_correlation_request_id": "gateway-request"
        }))
        .input(json!({"content": request.clone(), "headers": {}}))
        .build();
    let mut gateway_end = event_builder(gateway_uuid, EventType::End)
        .name("openai.chat_completions")
        .parent_uuid(parent_uuid)
        .scope_type(ScopeType::Llm)
        .model_name("test-model")
        .metadata(json!({
            "gateway_path": "/v1/chat/completions",
            "llm_correlation_request_id": "gateway-request"
        }))
        .output(json!({"choices": [{"message": {"content": "gateway_distinct_ok"}}]}))
        .build();
    let mut hook_end = event_builder(hook_uuid, EventType::End)
        .name("openrouter")
        .parent_uuid(parent_uuid)
        .scope_type(ScopeType::Llm)
        .model_name("test-model")
        .metadata(json!({
            "hook_event_name": "post_api_request",
            "api_call_id": "hook-request",
            "fidelity_source": "agent_api_hooks",
            "provider_payload_exact": false
        }))
        .output(json!({"assistant_content_chars": 10, "finish_reason": "stop"}))
        .build();

    for (idx, event) in [
        &mut hook_start,
        &mut gateway_start,
        &mut gateway_end,
        &mut hook_end,
    ]
    .into_iter()
    .enumerate()
    {
        set_event_timestamp(event, base + chrono::Duration::milliseconds(idx as i64));
    }

    {
        let mut state = exporter.state.lock().unwrap();
        state
            .events
            .extend([hook_start, gateway_start, gateway_end, hook_end]);
    }

    let trajectory = exporter.export().unwrap();
    assert_eq!(trajectory.steps.len(), 4);
    let function_name_count = |name: &str| {
        trajectory
            .steps
            .iter()
            .filter(|step| {
                step.extra
                    .as_ref()
                    .is_some_and(|extra| extra["ancestry"]["function_name"] == name)
            })
            .count()
    };
    assert!(function_name_count("openrouter") > 0);
    assert!(function_name_count("openai.chat_completions") > 0);
}

#[test]
fn test_exporter_keeps_sequential_same_content_llm_spans() {
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let base = base_timestamp();
    let parent_uuid = Uuid::now_v7();
    let hook_uuid = Uuid::now_v7();
    let gateway_uuid = Uuid::now_v7();
    let request = json!({
        "messages": [{"role": "user", "content": "Reply with exactly repeat_ok"}],
        "model": "test-model"
    });

    let mut hook_start = event_builder(hook_uuid, EventType::Start)
        .name("openrouter")
        .parent_uuid(parent_uuid)
        .scope_type(ScopeType::Llm)
        .metadata(json!({"hook_event_name": "pre_api_request"}))
        .input(request.clone())
        .build();
    let mut hook_end = event_builder(hook_uuid, EventType::End)
        .name("openrouter")
        .parent_uuid(parent_uuid)
        .scope_type(ScopeType::Llm)
        .metadata(json!({"hook_event_name": "post_api_request"}))
        .output(json!({"content": "repeat_ok"}))
        .build();
    let mut gateway_start = event_builder(gateway_uuid, EventType::Start)
        .name("openai.chat_completions")
        .parent_uuid(parent_uuid)
        .scope_type(ScopeType::Llm)
        .metadata(json!({"gateway_path": "/v1/chat/completions"}))
        .input(request)
        .build();
    let mut gateway_end = event_builder(gateway_uuid, EventType::End)
        .name("openai.chat_completions")
        .parent_uuid(parent_uuid)
        .scope_type(ScopeType::Llm)
        .metadata(json!({"gateway_path": "/v1/chat/completions"}))
        .output(json!({"choices": [{"message": {"content": "repeat_ok"}}]}))
        .build();

    for (idx, event) in [
        &mut hook_start,
        &mut hook_end,
        &mut gateway_start,
        &mut gateway_end,
    ]
    .into_iter()
    .enumerate()
    {
        set_event_timestamp(event, base + chrono::Duration::milliseconds(idx as i64));
    }

    {
        let mut state = exporter.state.lock().unwrap();
        state
            .events
            .extend([hook_start, hook_end, gateway_start, gateway_end]);
    }

    let trajectory = exporter.export().unwrap();
    assert_eq!(trajectory.steps.len(), 4);
    assert!(trajectory.steps.iter().any(|step| {
        step.extra
            .as_ref()
            .is_some_and(|extra| extra["ancestry"]["function_name"] == "openai.chat_completions")
    }));
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

    let traj_all = exporter.export().unwrap();
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

    assert_eq!(exporter.export().unwrap().steps.len(), 1);
    exporter.clear();
    assert!(exporter.export().unwrap().steps.is_empty());
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

    let trajectory = exporter.export().unwrap();
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

    let trajectory = exporter.export().unwrap();
    assert_eq!(trajectory.steps.len(), 1);

    let obs = trajectory.steps[0].observation.as_ref().unwrap();
    // Correlated by function name "search" → "call_xyz"
    assert_eq!(obs.results[0].source_call_id, Some("call_xyz".to_string()));
}

#[test]
fn test_exporter_correlates_hermes_style_tool_outputs_before_llm_calls() {
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let base = base_timestamp();
    let search_uuid = Uuid::now_v7();
    let read_uuid = Uuid::now_v7();
    let patch_uuid = Uuid::now_v7();
    let llm1_uuid = Uuid::now_v7();
    let llm2_uuid = Uuid::now_v7();
    let llm3_uuid = Uuid::now_v7();

    let mut search_start = event_builder(search_uuid, EventType::Start)
        .name("search_files")
        .scope_type(ScopeType::Tool)
        .input(json!({"path": ".", "pattern": "ReadOnlyPasswordHashField", "target": "content"}))
        .build();
    let mut search_end = event_builder(search_uuid, EventType::End)
        .name("search_files")
        .scope_type(ScopeType::Tool)
        .output(json!({"total_count": 6}))
        .build();
    let mut mark1 = event_builder(Uuid::now_v7(), EventType::Mark)
        .name("hermes_step")
        .data(json!({"iteration": 2, "previous_tools": ["search_files"]}))
        .build();
    let mut read_start = event_builder(read_uuid, EventType::Start)
        .name("read_file")
        .scope_type(ScopeType::Tool)
        .input(json!({"path": "./django/contrib/auth/forms.py", "offset": 50, "limit": 20}))
        .build();
    let mut read_end = event_builder(read_uuid, EventType::End)
        .name("read_file")
        .scope_type(ScopeType::Tool)
        .output(json!({"content": "class ReadOnlyPasswordHashField", "total_lines": 453}))
        .build();
    let mut patch_start = event_builder(patch_uuid, EventType::Start)
        .name("patch")
        .scope_type(ScopeType::Tool)
        .input(json!({
            "path": "./django/contrib/auth/forms.py",
            "old_string": "kwargs.setdefault(\"required\", False)",
            "new_string": "kwargs.setdefault(\"disabled\", True)"
        }))
        .build();
    let mut patch_end = event_builder(patch_uuid, EventType::End)
        .name("patch")
        .scope_type(ScopeType::Tool)
        .output(json!({"success": true, "files_modified": ["./django/contrib/auth/forms.py"]}))
        .build();

    let mut llm1_start = event_builder(llm1_uuid, EventType::Start)
        .name("hermes_assistant_message")
        .scope_type(ScopeType::Llm)
        .input(json!({"messages": [], "model": "policy_model", "projection_index": 0}))
        .build();
    let mut llm1_end = event_builder(llm1_uuid, EventType::End)
        .name("hermes_assistant_message")
        .scope_type(ScopeType::Llm)
        .output(json!({
            "content": "",
            "role": "assistant",
            "tool_calls": [{
                "id": "call-search",
                "type": "function",
                "function": {
                    "name": "search_files",
                    "arguments": "{\"path\":\".\",\"pattern\":\"ReadOnlyPasswordHashField\",\"target\":\"content\"}"
                }
            }]
        }))
        .build();
    let mut llm2_start = event_builder(llm2_uuid, EventType::Start)
        .name("hermes_assistant_message")
        .scope_type(ScopeType::Llm)
        .input(json!({"messages": [], "model": "policy_model", "projection_index": 2}))
        .build();
    let mut llm2_end = event_builder(llm2_uuid, EventType::End)
        .name("hermes_assistant_message")
        .scope_type(ScopeType::Llm)
        .output(json!({
            "content": "",
            "role": "assistant",
            "tool_calls": [{
                "id": "call-read",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": "{\"path\":\"./django/contrib/auth/forms.py\",\"offset\":50,\"limit\":20}"
                }
            }]
        }))
        .build();
    let mut llm3_start = event_builder(llm3_uuid, EventType::Start)
        .name("hermes_assistant_message")
        .scope_type(ScopeType::Llm)
        .input(json!({"messages": [], "model": "policy_model", "projection_index": 4}))
        .build();
    let mut llm3_end = event_builder(llm3_uuid, EventType::End)
        .name("hermes_assistant_message")
        .scope_type(ScopeType::Llm)
        .output(json!({
            "content": "",
            "role": "assistant",
            "tool_calls": [{
                "id": "call-patch",
                "type": "function",
                "function": {
                    "name": "patch",
                    "arguments": "{\"path\":\"./django/contrib/auth/forms.py\",\"old_string\":\"kwargs.setdefault(\\\"required\\\", False)\",\"new_string\":\"kwargs.setdefault(\\\"disabled\\\", True)\"}"
                }
            }]
        }))
        .build();

    for (idx, event) in [
        &mut search_start,
        &mut search_end,
        &mut mark1,
        &mut read_start,
        &mut read_end,
        &mut patch_start,
        &mut patch_end,
        &mut llm1_start,
        &mut llm1_end,
        &mut llm2_start,
        &mut llm2_end,
        &mut llm3_start,
        &mut llm3_end,
    ]
    .into_iter()
    .enumerate()
    {
        set_event_timestamp(event, base + chrono::Duration::milliseconds(idx as i64));
    }

    {
        let mut state = exporter.state.lock().unwrap();
        state.events.extend([
            search_start,
            search_end,
            mark1,
            read_start,
            read_end,
            patch_start,
            patch_end,
            llm1_start,
            llm1_end,
            llm2_start,
            llm2_end,
            llm3_start,
            llm3_end,
        ]);
    }

    let trajectory = exporter.export().unwrap();
    let agent_steps = trajectory
        .steps
        .iter()
        .filter(|step| step.source == "agent" && step.tool_calls.is_some())
        .collect::<Vec<_>>();
    assert_eq!(agent_steps.len(), 3);

    let expected = [
        ("call-search", json!({"total_count": 6})),
        (
            "call-read",
            json!({"content": "class ReadOnlyPasswordHashField", "total_lines": 453}),
        ),
        (
            "call-patch",
            json!({"success": true, "files_modified": ["./django/contrib/auth/forms.py"]}),
        ),
    ];
    for (step, (source_call_id, content)) in agent_steps.into_iter().zip(expected) {
        let observation = step.observation.as_ref().unwrap();
        assert_eq!(observation.results.len(), 1);
        assert_eq!(
            observation.results[0].source_call_id,
            Some(source_call_id.to_string())
        );
        assert_structured_observation_result_extra(&observation.results[0], content);
    }

    assert!(!trajectory.steps.iter().any(|step| {
        step.source == "system"
            && step
                .observation
                .as_ref()
                .is_some_and(|observation| !observation.results.is_empty())
    }));
}

#[test]
fn test_exporter_correlates_repeated_identical_tool_calls_by_ordinal() {
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let base = base_timestamp();
    let tool1_uuid = Uuid::now_v7();
    let tool2_uuid = Uuid::now_v7();
    let llm_uuid = Uuid::now_v7();

    let mut tool1_start = event_builder(tool1_uuid, EventType::Start)
        .name("read_file")
        .scope_type(ScopeType::Tool)
        .input(json!({"path": "./same.py", "offset": 0, "limit": 10}))
        .build();
    let mut tool1_end = event_builder(tool1_uuid, EventType::End)
        .name("read_file")
        .scope_type(ScopeType::Tool)
        .output(json!("first"))
        .build();
    let mut tool2_start = event_builder(tool2_uuid, EventType::Start)
        .name("read_file")
        .scope_type(ScopeType::Tool)
        .input(json!({"path": "./same.py", "offset": 0, "limit": 10}))
        .build();
    let mut tool2_end = event_builder(tool2_uuid, EventType::End)
        .name("read_file")
        .scope_type(ScopeType::Tool)
        .output(json!("second"))
        .build();
    let mut llm_end = event_builder(llm_uuid, EventType::End)
        .scope_type(ScopeType::Llm)
        .output(json!({
            "content": null,
            "role": "assistant",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\":\"./same.py\",\"offset\":0,\"limit\":10}"}},
                {"id": "c2", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\":\"./same.py\",\"offset\":0,\"limit\":10}"}}
            ]
        }))
        .build();

    for (idx, event) in [
        &mut tool1_start,
        &mut tool1_end,
        &mut tool2_start,
        &mut tool2_end,
        &mut llm_end,
    ]
    .into_iter()
    .enumerate()
    {
        set_event_timestamp(event, base + chrono::Duration::milliseconds(idx as i64));
    }

    {
        let mut state = exporter.state.lock().unwrap();
        state
            .events
            .extend([tool1_start, tool1_end, tool2_start, tool2_end, llm_end]);
    }

    let trajectory = exporter.export().unwrap();
    let observation = trajectory.steps[0].observation.as_ref().unwrap();
    assert_eq!(observation.results.len(), 2);
    assert_eq!(
        observation.results[0].source_call_id,
        Some("c1".to_string())
    );
    assert_eq!(observation.results[0].content, Some(json!("first")));
    assert_eq!(
        observation.results[1].source_call_id,
        Some("c2".to_string())
    );
    assert_eq!(observation.results[1].content, Some(json!("second")));
}

#[test]
fn test_exporter_correlates_mixed_explicit_implicit_duplicate_tool_calls() {
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let base = base_timestamp();
    let tool1_uuid = Uuid::now_v7();
    let tool2_uuid = Uuid::now_v7();
    let llm_uuid = Uuid::now_v7();
    let args = json!({"path": "./same.py", "offset": 0, "limit": 10});

    let mut tool1_start = event_builder(tool1_uuid, EventType::Start)
        .name("read_file")
        .scope_type(ScopeType::Tool)
        .input(args.clone())
        .tool_call_id("c1")
        .build();
    let mut tool1_end = event_builder(tool1_uuid, EventType::End)
        .name("read_file")
        .scope_type(ScopeType::Tool)
        .output(json!("first"))
        .build();
    let mut tool2_start = event_builder(tool2_uuid, EventType::Start)
        .name("read_file")
        .scope_type(ScopeType::Tool)
        .input(args)
        .build();
    let mut tool2_end = event_builder(tool2_uuid, EventType::End)
        .name("read_file")
        .scope_type(ScopeType::Tool)
        .output(json!("second"))
        .build();
    let mut llm_end = event_builder(llm_uuid, EventType::End)
        .scope_type(ScopeType::Llm)
        .output(json!({
            "content": null,
            "role": "assistant",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\":\"./same.py\",\"offset\":0,\"limit\":10}"}}
            ]
        }))
        .build();

    for (idx, event) in [
        &mut tool1_start,
        &mut tool1_end,
        &mut tool2_start,
        &mut tool2_end,
        &mut llm_end,
    ]
    .into_iter()
    .enumerate()
    {
        set_event_timestamp(event, base + chrono::Duration::milliseconds(idx as i64));
    }

    {
        let mut state = exporter.state.lock().unwrap();
        state
            .events
            .extend([tool1_start, tool1_end, tool2_start, tool2_end, llm_end]);
    }

    let trajectory = exporter.export().unwrap();
    let results = trajectory
        .steps
        .iter()
        .filter_map(|step| step.observation.as_ref())
        .flat_map(|observation| observation.results.iter())
        .collect::<Vec<_>>();

    assert_eq!(results.len(), 2);
    assert_eq!(
        results
            .iter()
            .find(|result| result.content == Some(json!("first")))
            .unwrap()
            .source_call_id,
        Some("c1".to_string())
    );
    assert_eq!(
        results
            .iter()
            .find(|result| result.content == Some(json!("second")))
            .unwrap()
            .source_call_id,
        None
    );
}

#[test]
fn test_exporter_does_not_guess_ambiguous_tool_calls_without_arguments() {
    let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
    let base = base_timestamp();
    let tool1_uuid = Uuid::now_v7();
    let tool2_uuid = Uuid::now_v7();
    let llm_uuid = Uuid::now_v7();

    let mut tool1_start = event_builder(tool1_uuid, EventType::Start)
        .name("lookup")
        .scope_type(ScopeType::Tool)
        .build();
    let mut tool1_end = event_builder(tool1_uuid, EventType::End)
        .name("lookup")
        .scope_type(ScopeType::Tool)
        .output(json!("first"))
        .build();
    let mut tool2_start = event_builder(tool2_uuid, EventType::Start)
        .name("lookup")
        .scope_type(ScopeType::Tool)
        .build();
    let mut tool2_end = event_builder(tool2_uuid, EventType::End)
        .name("lookup")
        .scope_type(ScopeType::Tool)
        .output(json!("second"))
        .build();
    let mut llm_end = event_builder(llm_uuid, EventType::End)
        .scope_type(ScopeType::Llm)
        .output(json!({
            "content": null,
            "role": "assistant",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}},
                {"id": "c2", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
            ]
        }))
        .build();

    for (idx, event) in [
        &mut tool1_start,
        &mut tool1_end,
        &mut tool2_start,
        &mut tool2_end,
        &mut llm_end,
    ]
    .into_iter()
    .enumerate()
    {
        set_event_timestamp(event, base + chrono::Duration::milliseconds(idx as i64));
    }

    {
        let mut state = exporter.state.lock().unwrap();
        state
            .events
            .extend([tool1_start, tool1_end, tool2_start, tool2_end, llm_end]);
    }

    let trajectory = exporter.export().unwrap();
    let agent_step = trajectory
        .steps
        .iter()
        .find(|step| step.source == "agent")
        .unwrap();
    assert!(agent_step.observation.is_none());

    let standalone = trajectory
        .steps
        .iter()
        .find(|step| step.source == "system" && step.observation.is_some())
        .unwrap();
    let observation = standalone.observation.as_ref().unwrap();
    assert_eq!(observation.results.len(), 2);
    assert!(
        observation
            .results
            .iter()
            .all(|result| result.source_call_id.is_none())
    );
}

#[test]
fn test_exporter_does_not_guess_by_name_for_active_duplicate_tool_names() {
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
                {"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}},
                {"id": "c2", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
            ]
        }))
        .build();
    let tool1_start = event_builder(tool1_uuid, EventType::Start)
        .name("lookup")
        .scope_type(ScopeType::Tool)
        .build();
    let tool1_end = event_builder(tool1_uuid, EventType::End)
        .name("lookup")
        .scope_type(ScopeType::Tool)
        .output(json!("first"))
        .build();
    let tool2_start = event_builder(tool2_uuid, EventType::Start)
        .name("lookup")
        .scope_type(ScopeType::Tool)
        .build();
    let tool2_end = event_builder(tool2_uuid, EventType::End)
        .name("lookup")
        .scope_type(ScopeType::Tool)
        .output(json!("second"))
        .build();

    {
        let mut state = exporter.state.lock().unwrap();
        state
            .events
            .extend([llm_end, tool1_start, tool1_end, tool2_start, tool2_end]);
    }

    let trajectory = exporter.export().unwrap();
    let agent_step = trajectory
        .steps
        .iter()
        .find(|step| step.source == "agent")
        .unwrap();
    assert!(agent_step.observation.is_none());

    let standalone = trajectory
        .steps
        .iter()
        .find(|step| step.source == "system" && step.observation.is_some())
        .unwrap();
    let observation = standalone.observation.as_ref().unwrap();
    assert_eq!(observation.results.len(), 2);
    assert!(
        observation
            .results
            .iter()
            .all(|result| result.source_call_id.is_none())
    );
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

    let trajectory = exporter.export().unwrap();
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

    let trajectory = exporter.export().unwrap();
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

    let trajectory = exporter.export().unwrap();
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

    let trajectory = exporter.export().unwrap();
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

    let trajectory = exporter.export().unwrap();
    let metrics = trajectory.steps[0].metrics.as_ref().unwrap();
    assert_eq!(metrics.prompt_tokens, Some(20));
    assert_eq!(metrics.completion_tokens, Some(10));
    assert_eq!(metrics.cached_tokens, Some(5));
    // Unknown keys land in extra
    let extra = metrics.extra.as_ref().unwrap();
    assert_eq!(extra["reasoning_tokens"], json!(150));
    // Known keys do not appear in extra
    assert!(extra.get("prompt_tokens").is_none());
    assert!(extra.get("completion_tokens").is_none());
    assert!(extra.get("cache_creation_input_tokens").is_none());
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

    let trajectory = exporter.export().unwrap();
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

    let trajectory = exporter.export().unwrap();
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

    let trajectory = exporter.export().unwrap();
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

    let trajectory = exporter.export().unwrap();
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

    let trajectory = exporter.export().unwrap();
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

    let trajectory = exporter.export().unwrap();
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

fn cleanup_test_agent_step(
    message: serde_json::Value,
    tool_call_ids: &[&str],
    observation_source_ids: &[&str],
) -> AtifStep {
    let observation = (!observation_source_ids.is_empty()).then(|| AtifObservation {
        results: observation_source_ids
            .iter()
            .map(|source_call_id| AtifObservationResult {
                source_call_id: Some((*source_call_id).to_string()),
                content: Some(json!("done")),
                subagent_trajectory_ref: None,
                extra: None,
            })
            .collect(),
    });

    AtifStep {
        step_id: 0,
        source: "agent".to_string(),
        message,
        timestamp: None,
        model_name: None,
        reasoning_effort: None,
        reasoning_content: None,
        tool_calls: Some(
            tool_call_ids
                .iter()
                .map(|tool_call_id| AtifToolCall {
                    tool_call_id: (*tool_call_id).to_string(),
                    function_name: "terminal".to_string(),
                    arguments: json!({"command": "pwd"}),
                    extra: None,
                })
                .collect(),
        ),
        observation,
        metrics: None,
        llm_call_count: Some(1),
        is_copied_context: None,
        extra: None,
    }
}

fn cleanup_test_user_step() -> AtifStep {
    AtifStep {
        step_id: 0,
        source: "user".to_string(),
        message: json!("next turn"),
        timestamp: None,
        model_name: None,
        reasoning_effort: None,
        reasoning_content: None,
        tool_calls: None,
        observation: None,
        metrics: None,
        llm_call_count: None,
        is_copied_context: None,
        extra: None,
    }
}

#[test]
fn test_projected_duplicate_tool_call_step_is_removed() {
    let mut steps = vec![
        cleanup_test_agent_step(empty_message(), &["call_dup"], &[]),
        cleanup_test_agent_step(empty_message(), &["call_dup"], &["call_dup"]),
    ];

    remove_projected_tool_call_duplicates(&mut steps);

    assert_eq!(steps.len(), 1);
    assert_eq!(step_tool_call_ids(&steps[0]), vec!["call_dup"]);
    assert_eq!(
        steps[0]
            .observation
            .as_ref()
            .unwrap()
            .results
            .first()
            .unwrap()
            .source_call_id
            .as_deref(),
        Some("call_dup")
    );
}

#[test]
fn test_projected_duplicate_tool_call_step_with_metrics_is_removed() {
    let mut steps = vec![
        cleanup_test_agent_step(empty_message(), &["call_dup"], &[]),
        cleanup_test_agent_step(empty_message(), &["call_dup"], &["call_dup"]),
    ];
    steps[0].metrics = Some(AtifMetrics {
        prompt_tokens: Some(10),
        completion_tokens: Some(5),
        ..Default::default()
    });

    remove_projected_tool_call_duplicates(&mut steps);

    assert_eq!(steps.len(), 1);
    assert_eq!(step_tool_call_ids(&steps[0]), vec!["call_dup"]);
    assert!(steps[0].observation.is_some());
}

#[test]
fn test_projected_duplicate_cleanup_keeps_same_id_across_turn_boundary() {
    let mut steps = vec![
        cleanup_test_agent_step(empty_message(), &["terminal:1"], &[]),
        cleanup_test_user_step(),
        cleanup_test_agent_step(empty_message(), &["terminal:1"], &["terminal:1"]),
    ];

    remove_projected_tool_call_duplicates(&mut steps);

    assert_eq!(steps.len(), 3);
    assert_eq!(step_tool_call_ids(&steps[0]), vec!["terminal:1"]);
}

#[test]
fn test_projected_duplicate_cleanup_preserves_meaningful_agent_content() {
    let mut steps = vec![
        cleanup_test_agent_step(json!("I will run terminal."), &["call_dup"], &[]),
        cleanup_test_agent_step(empty_message(), &["call_dup"], &["call_dup"]),
    ];

    remove_projected_tool_call_duplicates(&mut steps);

    assert_eq!(steps.len(), 2);
    assert_eq!(steps[0].message, json!("I will run terminal."));
}

#[test]
fn test_projected_duplicate_cleanup_preserves_partial_multi_call_step() {
    let mut steps = vec![
        cleanup_test_agent_step(empty_message(), &["call_a", "call_b"], &[]),
        cleanup_test_agent_step(empty_message(), &["call_a"], &["call_a"]),
    ];

    remove_projected_tool_call_duplicates(&mut steps);

    assert_eq!(steps.len(), 2);
    assert_eq!(step_tool_call_ids(&steps[0]), vec!["call_a", "call_b"]);
}
