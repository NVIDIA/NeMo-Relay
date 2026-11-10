// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Export-only provider views; original codec annotations remain lossless.

use nemo_relay_types::Json;
use nemo_relay_types::codec::normalization::{TelemetryToolCall, tool_call_array};
use nemo_relay_types::codec::request::AnnotatedLlmRequest;
use nemo_relay_types::codec::response::AnnotatedLlmResponse;
use std::borrow::Cow;

pub(crate) fn request_view(
    mut request: Cow<'_, AnnotatedLlmRequest>,
) -> Cow<'_, AnnotatedLlmRequest> {
    for index in 0..request.messages.len() {
        if let Some(message) =
            crate::openai_responses::custom_tool_message(&request.messages[index])
        {
            request.to_mut().messages[index] = message;
        }
    }
    request
}

pub(crate) fn tool_calls(
    annotation: Option<&AnnotatedLlmResponse>,
    output: Option<&Json>,
) -> Option<Vec<TelemetryToolCall>> {
    if let Some(calls) = annotation.and_then(|response| response.tool_calls.as_ref()) {
        let raw_calls = output.and_then(raw_tool_calls).unwrap_or_default();
        return Some(
            calls
                .iter()
                .map(|call| TelemetryToolCall {
                    id: call.id.clone(),
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                    extra: raw_calls
                        .iter()
                        .find(|raw| raw.id == call.id && raw.name == call.name)
                        .and_then(|raw| raw.extra.clone()),
                })
                .collect(),
        );
    }
    raw_tool_calls(output?)
}

/// Try to promote provider tool calls from a raw LLM response into `TelemetryToolCall` entries.
///
/// Supports generic/OpenAI Chat `tool_calls`, OpenAI Responses function and custom call
/// output items, and Anthropic `tool_use` content blocks. Provider-specific ID
/// semantics are preserved so later observations can correlate to the call.
///
/// Legacy function arguments keep their existing telemetry representation;
/// free-form tool input remains a string even when it contains valid JSON.
///
/// Returns `None` if there are no tool calls or the structure is unrecognized.
fn raw_tool_calls(output: &Json) -> Option<Vec<TelemetryToolCall>> {
    let (arr, origin) = if let Some(tool_calls) = tool_call_array(output).filter(|a| !a.is_empty())
    {
        (
            tool_calls.iter().collect::<Vec<_>>(),
            ToolCallOrigin::Generic,
        )
    } else if let Some(function_calls) = openai_responses_tool_call_items(output) {
        (function_calls, ToolCallOrigin::OpenAiResponses)
    } else {
        (
            anthropic_messages_tool_use_items(output)?,
            ToolCallOrigin::AnthropicMessages,
        )
    };
    let mut calls = Vec::with_capacity(arr.len());
    for (index, tc) in arr.iter().enumerate() {
        let Some(tc_obj) = tc.as_object() else {
            continue;
        };
        let mut id = tool_call_id_for_origin(tc_obj, origin)
            .unwrap_or("")
            .to_string();
        let is_custom = matches!(origin, ToolCallOrigin::OpenAiResponses)
            && tc_obj.get("type").and_then(Json::as_str) == Some("custom_tool_call");
        let func = (!is_custom)
            .then(|| tc_obj.get("function").and_then(Json::as_object))
            .flatten();
        let name = func
            .and_then(|f| f.get("name"))
            .or_else(|| tc_obj.get("name"))
            .or_else(|| tc_obj.get("toolName"))
            .or_else(|| tc_obj.get("tool_name"))
            .or_else(|| tc_obj.get("function_name"))
            .and_then(Json::as_str)
            .unwrap_or("")
            .to_string();
        if id.is_empty() && !name.is_empty() {
            id = format!("{name}:{}", index + 1);
        }
        let raw_arguments = func
            .and_then(|f| f.get("arguments"))
            .or_else(|| tc_obj.get("arguments"))
            .or_else(|| tc_obj.get("args"))
            .or_else(|| tc_obj.get("input"));
        let arguments = if is_custom {
            tc_obj.get("input").cloned().unwrap_or(Json::Null)
        } else {
            normalize_tool_arguments(raw_arguments)
        };
        // Skip entries with no id and no name — they are not meaningful.
        if id.is_empty() && name.is_empty() {
            continue;
        }
        calls.push(TelemetryToolCall {
            id,
            name,
            arguments,
            extra: tool_call_extra(tc),
        });
    }
    if calls.is_empty() { None } else { Some(calls) }
}

/// Provider payload shape governing tool-call identifier semantics.
#[derive(Clone, Copy)]
enum ToolCallOrigin {
    Generic,
    OpenAiResponses,
    AnthropicMessages,
}

/// Select the first usable correlation identifier for the provider payload shape.
fn tool_call_id_for_origin(
    tool_call: &serde_json::Map<String, Json>,
    origin: ToolCallOrigin,
) -> Option<&str> {
    match origin {
        // Responses distinguishes the output item `id` (`fc_*`) from the
        // invocation `call_id` used by function_call_output and tool events.
        ToolCallOrigin::OpenAiResponses => {
            ["call_id", "id", "tool_call_id"].iter().find_map(|field| {
                tool_call
                    .get(*field)
                    .and_then(Json::as_str)
                    .filter(|value| !value.is_empty())
            })
        }
        ToolCallOrigin::Generic | ToolCallOrigin::AnthropicMessages => {
            ["id", "tool_call_id", "call_id"].iter().find_map(|field| {
                tool_call
                    .get(*field)
                    .and_then(Json::as_str)
                    .filter(|value| !value.is_empty())
            })
        }
    }
}

fn openai_responses_tool_call_items(output: &Json) -> Option<Vec<&Json>> {
    let items = output
        .as_object()
        .and_then(|object| object.get("output"))
        .and_then(Json::as_array)?;
    let function_call_items = items
        .iter()
        .filter(|item| {
            item.get("type").and_then(Json::as_str) == Some("function_call")
                || crate::openai_responses::custom_tool_call_fields(item).is_some()
        })
        .collect::<Vec<_>>();
    (!function_call_items.is_empty()).then_some(function_call_items)
}

fn anthropic_messages_tool_use_items(output: &Json) -> Option<Vec<&Json>> {
    let object = output.as_object()?;
    if object.get("type").and_then(Json::as_str) != Some("message") {
        return None;
    }
    let content_blocks = object.get("content").and_then(Json::as_array)?;
    let tool_use_items = content_blocks
        .iter()
        .filter(|item| item.get("type").and_then(Json::as_str) == Some("tool_use"))
        .collect::<Vec<_>>();
    (!tool_use_items.is_empty()).then_some(tool_use_items)
}

pub(crate) fn normalize_tool_arguments(raw_arguments: Option<&Json>) -> Json {
    let Some(raw_arguments) = raw_arguments else {
        return serde_json::json!({});
    };
    match raw_arguments {
        Json::Object(_) => raw_arguments.clone(),
        Json::String(arguments) => match serde_json::from_str::<Json>(arguments) {
            Ok(Json::Object(object)) => Json::Object(object),
            Ok(value) => serde_json::json!({ "value": value }),
            Err(_) => serde_json::json!({ "raw": arguments }),
        },
        Json::Null => serde_json::json!({}),
        value => serde_json::json!({ "value": value }),
    }
}

fn tool_call_extra(tool_call: &Json) -> Option<Json> {
    let object = tool_call.as_object()?;
    let mut extra = serde_json::Map::new();

    for (key, value) in object {
        if !matches!(
            key.as_str(),
            "id" | "tool_call_id"
                | "call_id"
                | "type"
                | "function"
                | "name"
                | "tool_name"
                | "function_name"
                | "arguments"
                | "args"
                | "input"
        ) {
            extra.insert(key.clone(), value.clone());
        }
    }

    if let Some(function) = object.get("function").and_then(Json::as_object) {
        let mut function_extra = serde_json::Map::new();
        for (key, value) in function {
            if key != "name" && key != "arguments" {
                function_extra.insert(key.clone(), value.clone());
            }
        }
        if !function_extra.is_empty() {
            extra.insert("function".to_string(), Json::Object(function_extra));
        }
    }

    (!extra.is_empty()).then_some(Json::Object(extra))
}
