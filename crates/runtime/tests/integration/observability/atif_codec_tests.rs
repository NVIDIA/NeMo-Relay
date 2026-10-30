// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Core provider-codec normalization through ATIF.

use super::*;
use crate::api::event::{
    BaseEvent, CategoryProfile, Event, EventCategory, MarkEvent, ScopeCategory, ScopeEvent,
    llm_attributes_to_strings, scope_attributes_to_strings, tool_attributes_to_strings,
};
use crate::api::llm::{LlmAttributes, LlmRequest};
use crate::api::scope::{HandleAttributes, ScopeAttributes, ScopeType};
use crate::api::tool::ToolAttributes;
use crate::test_support::pricing_test_mutex;
use nemo_relay_codecs::anthropic::AnthropicMessagesCodec;
use nemo_relay_codecs::gemini_generate_content::GeminiGenerateContentCodec;
use nemo_relay_codecs::oci_genai::OCIGenAIChatCodec;
use nemo_relay_codecs::openai_chat::OpenAIChatCodec;
use nemo_relay_codecs::openai_responses::OpenAIResponsesCodec;
use nemo_relay_model_pricing::{
    PricingCatalog, PricingResolver, reset_active_pricing_resolver, set_active_pricing_resolver,
};
use nemo_relay_types::codec::request::{AnnotatedLlmRequest, ContentPart, Message, MessageContent};
use nemo_relay_types::codec::response::{AnnotatedLlmResponse, CostEstimate, CostSource, Usage};
use nemo_relay_types::codec::traits::{LlmCodec, LlmResponseCodec};
use serde_json::json;
use std::collections::HashSet;
use std::sync::Arc;

#[derive(Debug, Clone, Copy)]
enum EventType {
    Start,
    End,
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

fn is_atif_content_part(part: &serde_json::Value) -> bool {
    let Some(object) = part.as_object() else {
        return false;
    };
    match object.get("type").and_then(serde_json::Value::as_str) {
        Some("text") => object.len() == 2 && object.get("text").is_some_and(Json::is_string),
        Some("image") => object.len() == 2 && is_atif_image_source(object.get("source")),
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
    ) && source.len() == 2
        && source
            .get("path")
            .and_then(serde_json::Value::as_str)
            .is_some()
}

#[test]
fn test_exporter_responses_image_only_keeps_empty_display_and_raw_request() {
    for part in [
        json!({"type": "input_image", "image_url": "data:image/png;base64,YQ=="}),
        json!({"type": "input_image", "image_url": "https://example.com/image.png"}),
        json!({"type": "input_image", "file_id": "file-image"}),
        json!({"type": "input_file", "file_id": "file-document"}),
    ] {
        for use_annotation in [false, true] {
            let request = json!({"input": [
                {"role": "user", "content": "Earlier message"},
                {"role": "user", "content": [part]}
            ]});
            let exporter = crate::api::exporter::atif("session-1".to_string(), make_agent_info());
            let builder = event_builder(Uuid::now_v7(), EventType::Start)
                .scope_type(ScopeType::Llm)
                .input(request.clone());
            let start = if use_annotation {
                let annotation = OpenAIResponsesCodec
                    .decode(&LlmRequest {
                        headers: Default::default(),
                        content: request.clone(),
                    })
                    .unwrap();
                builder.annotated_request(annotation).build()
            } else {
                builder.build()
            };
            exporter.state.lock().unwrap().events.push(start);

            let trajectory = exporter.export().unwrap();
            assert_atif_v17_shape(&trajectory);
            let user = &trajectory.steps[0];
            assert_eq!(user.message, json!(""));
            assert_eq!(user.extra.as_ref().unwrap()["llm_request"], request);
        }
    }
}

// Asserts that overlapping LLM lifecycles retain their matching request metadata.

// Verifies request metadata when overlapping LLM calls finish in start order.

// Verifies request metadata when overlapping LLM calls finish in reverse order.

// Finds a response in the specified trajectory without searching child trajectories.

// Verifies that overlapping responses remain isolated to their owning subagent.

// Verifies that an empty LLM end does not detach a subsequent tool result.

#[test]
fn test_exporter_uses_annotated_message_but_raw_tool_calls() {
    // The annotation supplies the normalized message text (winning over the raw
    // content), while tool calls stay on the raw path so provider-specific
    // extras are preserved.
    let exporter = crate::api::exporter::atif("session-1".to_string(), make_agent_info());
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
    let exporter = crate::api::exporter::atif("session-1".to_string(), make_agent_info());
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
    let exporter = crate::api::exporter::atif("session-1".to_string(), make_agent_info());
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
    let exporter = crate::api::exporter::atif("session-1".to_string(), make_agent_info());
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
    let exporter = crate::api::exporter::atif("session-1".to_string(), make_agent_info());
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
fn test_exporter_annotated_gemini_multimodal_display() {
    for texts in [vec![], vec!["first"], vec!["first", "second"]] {
        for image in [
            json!({"inlineData": {"mimeType": "image/png", "data": "YQ=="}}),
            json!({"fileData": {"mimeType": "image/png", "fileUri": "gs://bucket/image.png"}}),
        ] {
            let mut parts: Vec<_> = texts.iter().map(|text| json!({"text": text})).collect();
            parts.insert(usize::from(!parts.is_empty()), image);
            assert_annotated_user_display(
                &GeminiGenerateContentCodec,
                json!({"contents": [{"role": "user", "parts": parts}]}),
                json!(texts.join("\n")),
            );
        }
    }
}

#[test]
fn test_exporter_annotated_oci_multimodal_display() {
    for texts in [vec![], vec!["first"], vec!["first", "second"]] {
        let mut parts: Vec<_> = texts
            .iter()
            .map(|text| json!({"type": "TEXT", "text": text}))
            .collect();
        parts.insert(
            usize::from(!parts.is_empty()),
            json!({"type": "IMAGE", "imageUrl": {"url": "data:image/png;base64,YQ=="}}),
        );
        let body =
            json!({"apiFormat": "GENERIC", "messages": [{"role": "USER", "content": parts}]});
        for request in [
            body.clone(),
            json!({
                "servingMode": {"servingType": "ON_DEMAND", "modelId": "vision-test"},
                "compartmentId": "ocid1.compartment.oc1..example",
                "chatRequest": body
            }),
        ] {
            assert_annotated_user_display(&OCIGenAIChatCodec, request, json!(texts.join("\n")));
        }
    }
}

#[test]
fn test_exporter_annotated_multimodal_display_keeps_supported_image_paths() {
    let path_image = json!({"type": "image", "source": {
        "media_type": "image/png", "path": "shot.png"
    }});
    let request = json!({"messages": [{"role": "user", "content": [
        {"type": "text", "text": "inspect", "cache_control": {"type": "ephemeral"}},
        {"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": "YQ=="
        }},
        path_image
    ]}]});
    // Anthropic represents these images as Image; Chat retains them as ProviderNative.
    for codec in [&AnthropicMessagesCodec as &dyn LlmCodec, &OpenAIChatCodec] {
        assert_annotated_user_display(
            codec,
            request.clone(),
            json!([{"type": "text", "text": "inspect"}, path_image]),
        );
    }
}

fn assert_annotated_user_display(codec: &dyn LlmCodec, request: Json, expected: Json) {
    for wrapped in [false, true] {
        let annotation = codec
            .decode(&LlmRequest {
                headers: Default::default(),
                content: request.clone(),
            })
            .unwrap();
        let input = if wrapped {
            json!({"content": request, "headers": {}})
        } else {
            request.clone()
        };
        let exporter = crate::api::exporter::atif("session-1".to_string(), make_agent_info());
        exporter.state.lock().unwrap().events.push(
            event_builder(Uuid::now_v7(), EventType::Start)
                .scope_type(ScopeType::Llm)
                .input(input)
                .annotated_request(annotation)
                .build(),
        );
        let trajectory = exporter.export().unwrap();
        assert_atif_v17_shape(&trajectory);
        let user = &trajectory.steps[0];
        assert_eq!(user.message, expected);
        assert_eq!(user.extra.as_ref().unwrap()["llm_request"], request);
    }
}

#[test]
fn test_exporter_annotated_multimodal_request_preserves_raw_content_in_extra() {
    let exporter = crate::api::exporter::atif("session-1".to_string(), make_agent_info());
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
        .input(multimodal.clone())
        .annotated_request(annotated)
        .build();
    exporter.state.lock().unwrap().events.push(start);

    let trajectory = exporter.export().unwrap();
    let user = &trajectory.steps[0];
    assert_eq!(user.message, json!("describe"));
    assert_eq!(user.extra.as_ref().unwrap()["llm_request"], multimodal);
}

#[test]
fn test_exporter_skips_duplicate_user_step_for_openai_responses_shell_call_continuation() {
    let exporter = crate::api::exporter::atif("session-1".to_string(), make_agent_info());
    let llm1_uuid = Uuid::now_v7();
    let llm2_uuid = Uuid::now_v7();
    let codec = OpenAIResponsesCodec;

    let llm1_start_input = json!({
        "model": "switchyard",
        "input": [{
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Fix pip"}]
        }]
    });
    let llm1_start = event_builder(llm1_uuid, EventType::Start)
        .name("openai.responses")
        .scope_type(ScopeType::Llm)
        .input(llm1_start_input)
        .model_name("switchyard")
        .build();
    let llm1_end = event_builder(llm1_uuid, EventType::End)
        .name("openai.responses")
        .scope_type(ScopeType::Llm)
        .output(json!({
            "id": "resp_1",
            "model": "switchyard",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "I will inspect the environment."}]
            }]
        }))
        .model_name("switchyard")
        .build();

    let llm2_start_input = json!({
        "model": "switchyard",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Fix pip"}]
            },
            {
                "type": "shell_call",
                "id": "sh_1",
                "call_id": "sh_call",
                "action": {"commands": ["which pip"]}
            }
        ]
    });
    let llm2_start = event_builder(llm2_uuid, EventType::Start)
        .name("openai.responses")
        .scope_type(ScopeType::Llm)
        .input(llm2_start_input.clone())
        .annotated_request(
            codec
                .decode(&LlmRequest {
                    headers: serde_json::Map::new(),
                    content: llm2_start_input,
                })
                .unwrap(),
        )
        .model_name("switchyard")
        .build();
    let llm2_end = event_builder(llm2_uuid, EventType::End)
        .name("openai.responses")
        .scope_type(ScopeType::Llm)
        .output(json!({
            "id": "resp_2",
            "model": "switchyard",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "I found the missing pip binary."}]
            }]
        }))
        .model_name("switchyard")
        .build();

    {
        let mut state = exporter.state.lock().unwrap();
        state
            .events
            .extend([llm1_start, llm1_end, llm2_start, llm2_end]);
    }

    let trajectory = exporter.export().unwrap();
    assert_atif_v17_shape(&trajectory);
    assert_eq!(trajectory.steps.len(), 3);
    let user_steps = trajectory
        .steps
        .iter()
        .filter(|step| step.source == "user")
        .collect::<Vec<_>>();
    assert_eq!(user_steps.len(), 1);
    assert_eq!(user_steps[0].message, json!("Fix pip"));

    let final_extra: AtifStepExtra =
        serde_json::from_value(trajectory.steps[2].extra.clone().unwrap()).unwrap();
    let llm_request = final_extra.llm_request.unwrap();
    assert_eq!(llm_request["input"][1]["type"], json!("shell_call"));
    assert_eq!(llm_request["input"][1]["call_id"], json!("sh_call"));
}
