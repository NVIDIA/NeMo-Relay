// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenInference subscriber support for NeMo Relay.
//!
//! Projection functions used by the unified OpenTelemetry subscriber.

use super::{
    estimate_cost_for_response_or_model, estimate_cost_for_response_or_requested_model,
    input_tokens_including_cache, manual, merge_usage, model_name_for_llm_event,
    push_serialized_top_level_attributes, push_tool_result_annotation_attribute,
    push_top_level_json_attributes, total_tokens_including_cache,
};
use crate::api::event::{Event, EventNormalizationExt};
use crate::api::scope::ScopeType;
use crate::codec::request::{
    AnnotatedLlmRequest, ContentPart, Message, MessageContent, ToolDefinition,
};
use crate::codec::response::{AnnotatedLlmResponse, FinishReason, ResponseToolCall, Usage};
use crate::json::Json;
#[cfg(test)]
use chrono::{DateTime, Utc};
use opentelemetry::KeyValue;
#[cfg(test)]
use opentelemetry::trace::SpanContext;
use opentelemetry::trace::SpanKind;
use serde::Serialize;
#[cfg(test)]
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub(super) fn span_kind(event: &Event) -> SpanKind {
    match semantic_scope_type(event) {
        Some(ScopeType::Llm) => SpanKind::Client,
        Some(
            ScopeType::Tool | ScopeType::Retriever | ScopeType::Embedder | ScopeType::Reranker,
        ) => SpanKind::Client,
        _ => SpanKind::Internal,
    }
}

pub(super) fn span_name(event: &Event) -> String {
    event.name().to_string()
}

fn semantic_scope_type(event: &Event) -> Option<ScopeType> {
    event.scope_type()
}

fn scope_type_name(scope_type: Option<ScopeType>) -> &'static str {
    match scope_type {
        Some(ScopeType::Agent) => "agent",
        Some(ScopeType::Function) => "function",
        Some(ScopeType::Tool) => "tool",
        Some(ScopeType::Llm) => "llm",
        Some(ScopeType::Retriever) => "retriever",
        Some(ScopeType::Embedder) => "embedder",
        Some(ScopeType::Reranker) => "reranker",
        Some(ScopeType::Guardrail) => "guardrail",
        Some(ScopeType::Evaluator) => "evaluator",
        Some(ScopeType::Custom) => "custom",
        Some(ScopeType::Unknown) | None => "unknown",
    }
}

pub(super) fn start_attributes(event: &Event) -> Vec<KeyValue> {
    let mut attributes = common_attributes(event);
    let is_llm = event
        .category()
        .is_some_and(|category| category.as_str() == "llm")
        || semantic_scope_type(event) == Some(ScopeType::Llm);
    if is_llm {
        // Final span metadata should reflect the completed event, especially for mixed-fidelity
        // Hermes flows where the request can be exact but the terminal error is lossy.
        attributes.retain(|attribute| {
            attribute.key.as_str() != "metadata"
                && !attribute.key.as_str().starts_with("openinference.metadata")
        });
    }
    if !is_llm {
        push_serialized_top_level_attributes(
            &mut attributes,
            "nemo_relay.handle_attributes",
            event.attributes(),
        );
        push_top_level_json_attributes(&mut attributes, "nemo_relay.start.data", event.data());
        push_top_level_json_attributes(&mut attributes, "nemo_relay.start.input", event.input());
    }
    if event
        .category()
        .is_some_and(|category| category.as_str() == "tool")
    {
        attributes.push(KeyValue::new("tool.name", event.name().to_string()));
        attributes.push(KeyValue::new(
            "tool_call.function.name",
            event.name().to_string(),
        ));
    }

    if let Some((input, mime_type)) = openinference_input_value(event) {
        attributes.push(KeyValue::new("input.value", input.clone()));
        attributes.push(KeyValue::new("input.mime_type", mime_type));

        if event
            .category()
            .is_some_and(|category| category.as_str() == "tool")
        {
            attributes.push(KeyValue::new("tool.parameters", input.clone()));
            attributes.push(KeyValue::new("tool_call.function.arguments", input));
        }
    }
    if is_llm {
        push_llm_request_attributes(&mut attributes, event);
    }
    attributes
}

pub(super) fn end_attributes(event: &Event) -> Vec<KeyValue> {
    let mut attributes = Vec::new();
    let is_llm = event
        .category()
        .is_some_and(|category| category.as_str() == "llm")
        || semantic_scope_type(event) == Some(ScopeType::Llm);

    push_top_level_json_attributes(&mut attributes, "nemo_relay.end.data", event.data());
    if let Some(metadata) = event.metadata().and_then(to_json_string) {
        attributes.push(KeyValue::new("metadata", metadata));
    }
    push_top_level_json_attributes(&mut attributes, "openinference.metadata", event.metadata());
    push_top_level_json_attributes(&mut attributes, "nemo_relay.end.output", event.output());
    push_tool_result_annotation_attribute(&mut attributes, event);
    if let Some((output, mime_type)) = openinference_output_value(event) {
        attributes.push(KeyValue::new("output.value", output));
        attributes.push(KeyValue::new("output.mime_type", mime_type));
    }
    let fallback_usage = if is_llm {
        manual::usage_from_manual_llm_output(event.output())
    } else {
        None
    };
    // Combine codec-normalized usage (which carries provider-derived fields such
    // as Anthropic's computed total) with the manual scraper, preferring codec
    // values per field so neither source's coverage is lost.
    let normalized = if is_llm {
        event.normalized_llm_response()
    } else {
        None
    };
    let usage = merge_usage(
        normalized
            .as_ref()
            .and_then(|response| response.usage.as_ref()),
        fallback_usage.as_ref(),
    );
    if is_llm {
        push_llm_usage_attributes(
            &mut attributes,
            Some(event.name()),
            normalized.as_deref(),
            usage.as_ref(),
        );
    }
    if is_llm
        && let Some(cost_total) =
            cost_total_from_llm_event(event, normalized.as_deref(), fallback_usage.as_ref())
    {
        attributes.push(KeyValue::new("llm.cost.total", cost_total));
    }
    if is_llm {
        push_llm_response_attributes(&mut attributes, event, normalized.as_deref());
    }
    attributes
}

fn push_llm_usage_attributes(
    attributes: &mut Vec<KeyValue>,
    provider: Option<&str>,
    response: Option<&AnnotatedLlmResponse>,
    usage: Option<&Usage>,
) {
    let Some(usage) = usage else {
        return;
    };
    if let Some(v) = input_tokens_including_cache(provider, response, usage) {
        attributes.push(KeyValue::new("llm.token_count.prompt", v as i64));
    }
    if let Some(v) = usage.completion_tokens {
        attributes.push(KeyValue::new("llm.token_count.completion", v as i64));
    }
    if let Some(v) = total_tokens_including_cache(provider, response, usage) {
        attributes.push(KeyValue::new("llm.token_count.total", v as i64));
    }
    if let Some(v) = usage.cache_read_tokens {
        attributes.push(KeyValue::new(
            "llm.token_count.prompt_details.cache_read",
            v as i64,
        ));
    }
    if let Some(v) = usage.cache_write_tokens {
        attributes.push(KeyValue::new(
            "llm.token_count.prompt_details.cache_write",
            v as i64,
        ));
    }
}

fn push_llm_request_attributes(attributes: &mut Vec<KeyValue>, event: &Event) {
    if let Some(request) = event.annotated_request() {
        push_annotated_request_attributes(attributes, request);
        return;
    }

    // Match replay before codec detection: replay content can look
    // provider-shaped (carry `messages`) and would otherwise be misrouted.
    if let Some(input) = event.input().and_then(replay_llm_payload) {
        if let Some(provider) = input.get("provider").and_then(Json::as_str) {
            attributes.push(KeyValue::new("llm.provider", provider.to_string()));
        }
        push_replay_input_messages(attributes, input);
        return;
    }

    if let Some(request) = event.normalized_llm_request() {
        push_annotated_request_attributes(attributes, &request);
    }
}

fn push_llm_response_attributes(
    attributes: &mut Vec<KeyValue>,
    event: &Event,
    normalized: Option<&AnnotatedLlmResponse>,
) {
    if let Some(response) = event.annotated_response() {
        push_annotated_response_attributes(attributes, response);
        return;
    }

    if let Some(output) = event.output().and_then(replay_llm_response) {
        push_replay_response_attributes(attributes, output);
        return;
    }

    // Reuse the response decoded once in `end_attributes` (annotation-first;
    // falls through to codec detection) instead of decoding the payload again.
    if let Some(response) = normalized {
        push_annotated_response_attributes(attributes, response);
    }
}

fn push_annotated_request_attributes(
    attributes: &mut Vec<KeyValue>,
    request: &AnnotatedLlmRequest,
) {
    if let Some(params) = request.params.as_ref().and_then(to_json_string) {
        attributes.push(KeyValue::new("llm.invocation_parameters", params));
    }
    let mut next_index = 0usize;
    if let Some(instructions) = request.instructions.as_ref().and_then(message_content_text) {
        push_message_role(attributes, "llm.input_messages", next_index, "system");
        push_message_text_value(attributes, "llm.input_messages", next_index, instructions);
        next_index += 1;
    }
    push_annotated_input_messages(attributes, &request.messages, next_index);
    if let Some(tools) = request.tools.as_deref() {
        push_annotated_tools(attributes, tools);
    }
}

fn push_annotated_response_attributes(
    attributes: &mut Vec<KeyValue>,
    response: &AnnotatedLlmResponse,
) {
    if let Some(reason) = response.finish_reason.as_ref() {
        attributes.push(KeyValue::new(
            "llm.finish_reason",
            finish_reason_value(reason),
        ));
    }

    let has_message = response.message.is_some()
        || response
            .tool_calls
            .as_ref()
            .is_some_and(|tool_calls| !tool_calls.is_empty());
    if has_message {
        attributes.push(KeyValue::new(
            "llm.output_messages.0.message.role",
            "assistant",
        ));
    }
    if let Some(content) = response.message.as_ref().and_then(message_content_text) {
        attributes.push(KeyValue::new(
            "llm.output_messages.0.message.content",
            content,
        ));
    }
    if let Some(tool_calls) = response.tool_calls.as_deref() {
        push_response_tool_calls(attributes, 0, tool_calls);
    }
    if let Some(summary) = response.optimization_summary.as_ref() {
        push_optimization_attributes(attributes, summary);
    }
}

fn push_optimization_attributes(
    attributes: &mut Vec<KeyValue>,
    summary: &crate::codec::optimization::LlmOptimizationSummary,
) {
    crate::observability::push_common_optimization_attributes(attributes, summary);
}

fn push_annotated_input_messages(
    attributes: &mut Vec<KeyValue>,
    messages: &[Message],
    start_index: usize,
) {
    let mut next_index = start_index;
    for message in messages {
        if let Message::User { content, .. } = message
            && let Some(parts) = exclusive_tool_result_parts(content)
        {
            for part in parts {
                let ContentPart::ToolResult {
                    tool_use_id,
                    content,
                    ..
                } = part
                else {
                    continue;
                };
                push_tool_result_input_message(attributes, next_index, tool_use_id, content);
                next_index += 1;
            }
            continue;
        }

        let index = next_index;
        next_index += 1;
        let (role, content, tool_call_id) = match message {
            Message::System { content, .. } => ("system", message_content_text(content), None),
            Message::Developer { content, .. } => {
                ("developer", message_content_text(content), None)
            }
            Message::User { content, .. } => ("user", message_content_text(content), None),
            Message::Assistant { content, .. } => (
                "assistant",
                content.as_ref().and_then(message_content_text),
                None,
            ),
            Message::Tool {
                content,
                tool_call_id,
            } => (
                "tool",
                tool_message_content(content),
                Some(tool_call_id.as_str()),
            ),
            Message::Function { content, .. } => (
                "function",
                content.as_deref().and_then(display_text_from_string),
                None,
            ),
            Message::ToolCallItem { .. } => ("assistant", None, None),
            Message::ToolResultItem {
                call_id, output, ..
            } => (
                "tool",
                Some(tool_result_content(output)),
                Some(call_id.as_str()),
            ),
            Message::ProviderNative { value, .. } => (
                value
                    .get("role")
                    .and_then(Json::as_str)
                    .unwrap_or("provider_native"),
                value.get("content").and_then(display_text_from_json),
                None,
            ),
        };
        push_message_role(attributes, "llm.input_messages", index, role);
        if let Some(content) = content {
            push_message_text_value(attributes, "llm.input_messages", index, content);
        }
        if let Some(tool_call_id) = tool_call_id {
            attributes.push(KeyValue::new(
                format!("llm.input_messages.{index}.message.tool_call_id"),
                tool_call_id.to_string(),
            ));
        }
    }
}

fn exclusive_tool_result_parts(content: &MessageContent) -> Option<&[ContentPart]> {
    let MessageContent::Parts(parts) = content else {
        return None;
    };
    (!parts.is_empty()
        && parts
            .iter()
            .all(|part| matches!(part, ContentPart::ToolResult { .. })))
    .then_some(parts)
}

fn push_tool_result_input_message(
    attributes: &mut Vec<KeyValue>,
    index: usize,
    tool_call_id: &str,
    content: &Json,
) {
    push_message_role(attributes, "llm.input_messages", index, "tool");
    push_message_text_value(
        attributes,
        "llm.input_messages",
        index,
        tool_result_content(content),
    );
    attributes.push(KeyValue::new(
        format!("llm.input_messages.{index}.message.tool_call_id"),
        tool_call_id.to_string(),
    ));
}

fn tool_message_content(content: &MessageContent) -> Option<String> {
    match content {
        MessageContent::Text(text) => Some(text.clone()),
        MessageContent::Parts(parts) => to_json_string(parts),
    }
}

fn tool_result_content(content: &Json) -> String {
    match content {
        Json::String(text) => text.clone(),
        value => value.to_string(),
    }
}

fn push_annotated_tools(attributes: &mut Vec<KeyValue>, tools: &[ToolDefinition]) {
    for (index, tool) in tools.iter().enumerate() {
        if let Some(json) = to_json_string(tool) {
            attributes.push(KeyValue::new(
                format!("llm.tools.{index}.tool.json_schema"),
                json,
            ));
        }
    }
}

fn push_response_tool_calls(
    attributes: &mut Vec<KeyValue>,
    message_index: usize,
    tool_calls: &[ResponseToolCall],
) {
    for (call_index, tool_call) in tool_calls.iter().enumerate() {
        push_output_tool_call(
            attributes,
            message_index,
            call_index,
            Some(tool_call.id.as_str()),
            Some(tool_call.name.as_str()),
            to_json_string(&tool_call.arguments),
        );
    }
}

fn push_message_role(
    attributes: &mut Vec<KeyValue>,
    prefix: &'static str,
    index: usize,
    role: &str,
) {
    attributes.push(KeyValue::new(
        format!("{prefix}.{index}.message.role"),
        role.to_string(),
    ));
}

fn push_message_text_value(
    attributes: &mut Vec<KeyValue>,
    prefix: &'static str,
    index: usize,
    text: String,
) {
    attributes.push(KeyValue::new(
        format!("{prefix}.{index}.message.content"),
        text,
    ));
}

fn message_content_text(content: &MessageContent) -> Option<String> {
    match content {
        MessageContent::Text(text) => display_text_from_string(text),
        MessageContent::Parts(parts) => {
            let text = parts
                .iter()
                .filter_map(|part| match part {
                    ContentPart::Text { text, .. } => Some(text.as_str()),
                    ContentPart::Refusal { refusal, .. } => Some(refusal.as_str()),
                    ContentPart::ProviderNative { value, .. } => value
                        .get("text")
                        .and_then(Json::as_str)
                        .or_else(|| value.get("refusal").and_then(Json::as_str)),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n")
                .trim()
                .to_string();
            if text.is_empty() { None } else { Some(text) }
        }
    }
}

fn replay_llm_payload(input: &Json) -> Option<&Json> {
    let content = input.as_object().and_then(|object| object.get("content"))?;
    let content_object = content.as_object()?;
    is_openclaw_replay_payload(content_object).then_some(content)
}

fn replay_llm_response(output: &Json) -> Option<&Json> {
    output
        .as_object()
        .and_then(|object| object.get("openclaw"))
        .and_then(Json::as_object)
        .map(|_| output)
}

fn is_openclaw_replay_payload(content: &serde_json::Map<String, Json>) -> bool {
    content
        .get("source")
        .and_then(Json::as_str)
        .is_some_and(|source| source.starts_with("openclaw."))
        || content.contains_key("placeholderRequest")
}

fn push_replay_input_messages(attributes: &mut Vec<KeyValue>, input: &Json) {
    let mut next_index = 0usize;
    if let Some(system_prompt) = input.get("systemPrompt").and_then(display_text_from_json) {
        push_message_role(attributes, "llm.input_messages", next_index, "system");
        attributes.push(KeyValue::new(
            format!("llm.input_messages.{next_index}.message.content"),
            system_prompt,
        ));
        next_index += 1;
    }
    if let Some(messages) = input.get("messages").and_then(Json::as_array) {
        let first_message_index = next_index;
        for message in messages {
            if push_replay_input_message(attributes, next_index, message) {
                next_index += 1;
            }
        }
        if next_index > first_message_index {
            return;
        }
    }
    if let Some(prompt) = input.get("prompt").and_then(display_text_from_json) {
        push_message_role(attributes, "llm.input_messages", next_index, "user");
        attributes.push(KeyValue::new(
            format!("llm.input_messages.{next_index}.message.content"),
            prompt,
        ));
    }
}

fn push_replay_input_message(attributes: &mut Vec<KeyValue>, index: usize, message: &Json) -> bool {
    let Some(object) = message.as_object() else {
        return false;
    };
    let Some(role) = object.get("role").and_then(Json::as_str) else {
        return false;
    };
    let Some(text) = object.get("content").and_then(display_text_from_json) else {
        return false;
    };
    push_message_role(attributes, "llm.input_messages", index, role);
    attributes.push(KeyValue::new(
        format!("llm.input_messages.{index}.message.content"),
        text,
    ));
    true
}

fn push_replay_response_attributes(attributes: &mut Vec<KeyValue>, output: &Json) {
    if output.get("role").is_none()
        && output.get("content").is_none()
        && output.get("tool_calls").is_none()
    {
        return;
    }
    let role = output
        .get("role")
        .and_then(Json::as_str)
        .unwrap_or("assistant");
    push_message_role(attributes, "llm.output_messages", 0, role);
    if let Some(content) = output.get("content").and_then(display_text_from_json) {
        attributes.push(KeyValue::new(
            "llm.output_messages.0.message.content",
            content,
        ));
    }
    if let Some(tool_calls) = output.get("tool_calls").and_then(Json::as_array) {
        push_raw_output_tool_calls(attributes, 0, tool_calls);
    }
}

fn push_raw_output_tool_calls(
    attributes: &mut Vec<KeyValue>,
    message_index: usize,
    tool_calls: &[Json],
) {
    for (call_index, tool_call) in tool_calls.iter().enumerate() {
        push_output_tool_call(
            attributes,
            message_index,
            call_index,
            raw_tool_call_id(tool_call),
            raw_tool_call_name(tool_call),
            raw_tool_call_arguments(tool_call).and_then(|value| {
                value
                    .as_str()
                    .map(str::to_string)
                    .or_else(|| to_json_string(value))
            }),
        );
    }
}

// Raw replay payloads are an OpenInference-local fallback. Provider-shaped
// responses should use codec-normalized response tool calls instead.
fn raw_tool_call_id(tool_call: &Json) -> Option<&str> {
    tool_call
        .get("id")
        .or_else(|| tool_call.get("tool_call_id"))
        .or_else(|| tool_call.get("call_id"))
        .and_then(Json::as_str)
}

fn raw_tool_call_name(tool_call: &Json) -> Option<&str> {
    tool_call
        .get("name")
        .and_then(Json::as_str)
        .or_else(|| tool_call.get("toolName").and_then(Json::as_str))
        .or_else(|| tool_call.get("tool_name").and_then(Json::as_str))
        .or_else(|| {
            tool_call
                .get("function")
                .and_then(|function| function.get("name"))
                .and_then(Json::as_str)
        })
        .or_else(|| tool_call.get("function_name").and_then(Json::as_str))
}

fn raw_tool_call_arguments(tool_call: &Json) -> Option<&Json> {
    tool_call
        .get("function")
        .and_then(|function| function.get("arguments"))
        .or_else(|| tool_call.get("arguments"))
        .or_else(|| tool_call.get("args"))
        .or_else(|| tool_call.get("input"))
}

fn push_output_tool_call(
    attributes: &mut Vec<KeyValue>,
    message_index: usize,
    call_index: usize,
    id: Option<&str>,
    name: Option<&str>,
    arguments: Option<String>,
) {
    if let Some(id) = id {
        attributes.push(KeyValue::new(
            format!(
                "llm.output_messages.{message_index}.message.tool_calls.{call_index}.tool_call.id"
            ),
            id.to_string(),
        ));
    }
    if let Some(name) = name {
        attributes.push(KeyValue::new(
            format!(
                "llm.output_messages.{message_index}.message.tool_calls.{call_index}.tool_call.function.name"
            ),
            name.to_string(),
        ));
    }
    if let Some(arguments) = arguments {
        attributes.push(KeyValue::new(
            format!(
                "llm.output_messages.{message_index}.message.tool_calls.{call_index}.tool_call.function.arguments"
            ),
            arguments,
        ));
    }
}

fn finish_reason_value(reason: &FinishReason) -> String {
    match reason {
        FinishReason::Complete => "complete".to_string(),
        FinishReason::Length => "length".to_string(),
        FinishReason::ToolUse => "tool_use".to_string(),
        FinishReason::ContentFilter => "content_filter".to_string(),
        FinishReason::Unknown(reason) => reason.clone(),
    }
}

fn cost_total_from_llm_event(
    event: &Event,
    normalized_response: Option<&AnnotatedLlmResponse>,
    fallback_usage: Option<&Usage>,
) -> Option<f64> {
    if let Some(response) = normalized_response
        && let Some(usage) = response.usage.as_ref()
    {
        if let Some(cost) = usage.cost.as_ref() {
            return cost.total_or_component_sum_for_currency("USD");
        }
        if let Some(cost) =
            estimate_cost_for_response_or_requested_model(event, response.model.as_deref(), usage)
        {
            return cost.total_for_currency("USD");
        }
    }

    if let Some(cost) =
        manual::cost_from_manual_llm_output(event.output(), manual::ManualCostPolicy::UsdOnly)
            .map(|(total, _)| total)
    {
        return Some(cost);
    }

    let usage = fallback_usage?;
    estimate_cost_for_response_or_model(
        Some(event.name()),
        event.model_name(),
        manual::model_name_from_manual_llm_output(event.output()),
        usage,
    )
    .and_then(|cost| cost.total_for_currency("USD"))
}

pub(super) fn mark_attributes(event: &Event) -> Vec<KeyValue> {
    let mut attributes = vec![
        KeyValue::new("nemo_relay.mark.uuid", event.uuid().to_string()),
        KeyValue::new(
            "nemo_relay.mark.parent_uuid",
            event
                .parent_uuid()
                .map(|uuid| uuid.to_string())
                .unwrap_or_default(),
        ),
    ];
    push_serialized_top_level_attributes(
        &mut attributes,
        "nemo_relay.mark.attributes",
        event.attributes(),
    );
    push_top_level_json_attributes(&mut attributes, "nemo_relay.mark.data", event.data());
    push_top_level_json_attributes(
        &mut attributes,
        "nemo_relay.mark.metadata",
        event.metadata(),
    );
    if let Some(category) = event.category() {
        attributes.push(KeyValue::new(
            "nemo_relay.mark.category",
            category.as_str().to_string(),
        ));
    }
    push_serialized_top_level_attributes(
        &mut attributes,
        "nemo_relay.mark.category_profile",
        event.category_profile(),
    );
    attributes
}

fn push_projected_mark_attributes(attributes: &mut Vec<KeyValue>, event: &Event) {
    let mark_name = event.name().to_string();
    attributes.push(KeyValue::new("tool.name", mark_name.clone()));
    attributes.push(KeyValue::new("tool_call.function.name", mark_name));

    if let Some(data) = event.data().and_then(to_json_string) {
        attributes.push(KeyValue::new("output.value", data));
        attributes.push(KeyValue::new("output.mime_type", "application/json"));
    }
    if let Some(metadata) = event.metadata().and_then(to_json_string) {
        attributes.push(KeyValue::new("metadata", metadata));
    }
}

pub(super) fn remove_start_model_name(attributes: &mut Vec<KeyValue>) {
    attributes.retain(|attribute| attribute.key.as_str() != "llm.model_name");
}

pub(super) fn push_model_name(attributes: &mut Vec<KeyValue>, model_name: String) {
    attributes.push(KeyValue::new("llm.model_name", model_name));
}

pub(super) fn push_orphan_mark_attributes(attributes: &mut Vec<KeyValue>) {
    attributes.push(KeyValue::new("openinference.span.kind", "CHAIN"));
    attributes.push(KeyValue::new("nemo_relay.mark.orphan", true));
}

pub(super) fn push_tool_mark_attributes(attributes: &mut Vec<KeyValue>, event: &Event) {
    attributes.push(KeyValue::new("openinference.span.kind", "TOOL"));
    push_projected_mark_attributes(attributes, event);
}

fn common_attributes(event: &Event) -> Vec<KeyValue> {
    let mut attributes = vec![
        KeyValue::new(
            "openinference.span.kind",
            openinference_span_kind(semantic_scope_type(event)),
        ),
        KeyValue::new("nemo_relay.uuid", event.uuid().to_string()),
        KeyValue::new(
            "nemo_relay.parent_uuid",
            event
                .parent_uuid()
                .map(|uuid| uuid.to_string())
                .unwrap_or_default(),
        ),
        KeyValue::new(
            "nemo_relay.scope_type",
            scope_type_name(semantic_scope_type(event)),
        ),
    ];

    if let Some(model_name) = model_name_for_llm_event(event) {
        attributes.push(KeyValue::new("llm.model_name", model_name));
    }
    if let Some(tool_call_id) = event.tool_call_id() {
        attributes.push(KeyValue::new("tool_call.id", tool_call_id.to_string()));
    }
    if let Some(metadata) = event.metadata().and_then(to_json_string) {
        attributes.push(KeyValue::new("metadata", metadata));
    }
    push_top_level_json_attributes(&mut attributes, "openinference.metadata", event.metadata());

    attributes
}

fn openinference_span_kind(scope_type: Option<ScopeType>) -> &'static str {
    match scope_type {
        Some(ScopeType::Agent) => "AGENT",
        Some(ScopeType::Tool) => "TOOL",
        Some(ScopeType::Llm) => "LLM",
        Some(ScopeType::Retriever) => "RETRIEVER",
        Some(ScopeType::Embedder) => "EMBEDDING",
        Some(ScopeType::Reranker) => "RERANKER",
        Some(ScopeType::Guardrail) => "GUARDRAIL",
        Some(ScopeType::Evaluator) => "EVALUATOR",
        Some(ScopeType::Function | ScopeType::Custom | ScopeType::Unknown) | None => "CHAIN",
    }
}

fn openinference_input_value(event: &Event) -> Option<(String, &'static str)> {
    let input = event.input()?;

    if event
        .category()
        .is_some_and(|category| category.as_str() == "llm")
    {
        return llm_input_display_value(input)
            .map(|display| (display, "text/plain"))
            .or_else(|| sanitized_llm_input_json(input).map(|json| (json, "application/json")));
    }

    to_json_string(input).map(|json| (json, "application/json"))
}

fn openinference_output_value(event: &Event) -> Option<(String, &'static str)> {
    let output = event.output()?;
    display_text_from_json(output)
        .map(|display| (display, "text/plain"))
        .or_else(|| to_json_string(output).map(|json| (json, "application/json")))
}

fn llm_input_display_value(input: &Json) -> Option<String> {
    let content = match input {
        Json::Object(object) => object.get("content").unwrap_or(input),
        _ => input,
    };

    content
        .get("messages")
        .and_then(display_text_from_messages)
        .or_else(|| display_text_from_json(content))
}

fn sanitized_llm_input_json(input: &Json) -> Option<String> {
    match input {
        Json::Object(object) => {
            let mut sanitized = object.clone();
            sanitized.remove("headers");
            to_json_string(&Json::Object(sanitized))
        }
        _ => to_json_string(input),
    }
}

fn display_text_from_json(value: &Json) -> Option<String> {
    match value {
        Json::String(text) => display_text_from_string(text),
        Json::Object(object) => {
            for key in ["content", "summary", "message", "text", "prompt"] {
                if let Some(display) = object.get(key).and_then(display_text_from_json) {
                    return Some(display);
                }
            }
            object
                .get("output")
                .and_then(display_text_from_openai_responses_output)
                .or_else(|| {
                    object
                        .get("choices")
                        .and_then(display_text_from_chat_choices)
                })
                .or_else(|| {
                    object
                        .get("tool_calls")
                        .and_then(display_text_from_tool_calls)
                })
        }
        Json::Array(items) => display_text_from_content_blocks(items),
        _ => None,
    }
}

fn display_text_from_openai_responses_output(value: &Json) -> Option<String> {
    let items = value.as_array()?;
    let mut entries = Vec::new();
    let mut tool_names = Vec::new();
    for item in items {
        let Some(object) = item.as_object() else {
            continue;
        };
        match object.get("type").and_then(Json::as_str) {
            Some("message") => {
                if let Some(content) = object
                    .get("content")
                    .and_then(display_text_from_openai_responses_content)
                {
                    entries.push(content);
                }
            }
            Some("function_call") => {
                if let Some(name) = object.get("name").and_then(Json::as_str) {
                    tool_names.push(name.to_string());
                }
            }
            _ => {}
        }
    }
    if !tool_names.is_empty() {
        entries.push(format!("Requested tools: {}", tool_names.join(", ")));
    }
    let text = entries.join("\n").trim().to_string();
    if text.is_empty() { None } else { Some(text) }
}

fn display_text_from_openai_responses_content(value: &Json) -> Option<String> {
    let content = value.as_array()?;
    let text = content
        .iter()
        .filter_map(|part| {
            let object = part.as_object()?;
            match object.get("type").and_then(Json::as_str) {
                Some("output_text" | "text") => object.get("text").and_then(Json::as_str),
                _ => None,
            }
        })
        .collect::<Vec<_>>()
        .join("\n\n")
        .trim()
        .to_string();
    if text.is_empty() { None } else { Some(text) }
}

fn display_text_from_messages(value: &Json) -> Option<String> {
    let messages = value.as_array()?;
    let text = messages
        .iter()
        .filter_map(display_text_from_message)
        .collect::<Vec<_>>()
        .join("\n\n")
        .trim()
        .to_string();
    if text.is_empty() { None } else { Some(text) }
}

fn display_text_from_message(value: &Json) -> Option<String> {
    let role = value
        .get("role")
        .and_then(Json::as_str)
        .unwrap_or("message");
    if role == "tool" {
        return Some("tool: Tool result omitted".to_string());
    }
    let display = value
        .get("content")
        .and_then(display_text_from_json)
        .or_else(|| {
            value
                .get("tool_calls")
                .and_then(display_text_from_tool_calls)
        })?;
    Some(format!("{role}: {display}"))
}

fn display_text_from_string(text: &str) -> Option<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Ok(parsed) = serde_json::from_str::<Json>(trimmed)
        && let Some(display) = display_text_from_json(&parsed)
    {
        return Some(display);
    }
    Some(trimmed.to_string())
}

fn display_text_from_chat_choices(value: &Json) -> Option<String> {
    let choices = value.as_array()?;
    for choice in choices {
        let Some(message) = choice.get("message") else {
            continue;
        };
        let content = message.get("content").and_then(display_text_from_json);
        let tool_calls = message
            .get("tool_calls")
            .and_then(display_text_from_tool_calls);
        match (content, tool_calls) {
            (Some(content), Some(tool_calls)) => return Some(format!("{content}\n{tool_calls}")),
            (Some(content), None) => return Some(content),
            (None, Some(tool_calls)) => return Some(tool_calls),
            (None, None) => {}
        }
    }
    None
}

fn display_text_from_content_blocks(items: &[Json]) -> Option<String> {
    let mut entries = items
        .iter()
        .filter_map(content_block_display_text)
        .collect::<Vec<_>>();
    let tool_calls = items.iter().filter_map(tool_call_name).collect::<Vec<_>>();
    if !tool_calls.is_empty() {
        entries.push(format!("Requested tools: {}", tool_calls.join(", ")));
    }
    let text = entries
        .into_iter()
        .filter(|item| !item.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string();
    if text.is_empty() { None } else { Some(text) }
}

fn content_block_display_text(item: &Json) -> Option<String> {
    if let Some(text) = item.as_str() {
        return Some(text.to_string());
    }
    if item.get("stripped").and_then(Json::as_bool) == Some(true) {
        return None;
    }
    if let Some("thinking" | "reasoning" | "toolResult" | "tool_result") =
        item.get("type").and_then(Json::as_str)
    {
        return None;
    }
    item.get("text").and_then(Json::as_str).map(str::to_string)
}

fn display_text_from_tool_calls(value: &Json) -> Option<String> {
    let calls = value.as_array()?;
    let names = calls.iter().filter_map(tool_call_name).collect::<Vec<_>>();
    if names.is_empty() {
        None
    } else {
        Some(format!("Requested tools: {}", names.join(", ")))
    }
}

fn tool_call_name(value: &Json) -> Option<String> {
    value
        .get("name")
        .and_then(Json::as_str)
        .or_else(|| value.get("toolName").and_then(Json::as_str))
        .or_else(|| {
            value
                .get("function")
                .and_then(|function| function.get("name"))
                .and_then(Json::as_str)
        })
        .map(str::to_string)
}

fn to_json_string<T: Serialize>(value: &T) -> Option<String> {
    serde_json::to_string(value).ok()
}

#[cfg(test)]
fn local_parent_span_context(span_context: &SpanContext) -> SpanContext {
    SpanContext::new(
        span_context.trace_id(),
        span_context.span_id(),
        span_context.trace_flags(),
        false,
        span_context.trace_state().clone(),
    )
}

#[cfg(test)]
fn to_system_time(timestamp: DateTime<Utc>) -> SystemTime {
    let seconds = timestamp.timestamp();
    let nanos = timestamp.timestamp_subsec_nanos();
    if seconds >= 0 {
        UNIX_EPOCH + Duration::new(seconds as u64, nanos)
    } else if nanos == 0 {
        UNIX_EPOCH - Duration::new(seconds.unsigned_abs(), 0)
    } else {
        UNIX_EPOCH - Duration::new(seconds.unsigned_abs() - 1, 1_000_000_000 - nanos)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/observability/openinference_tests.rs"]
mod tests;
