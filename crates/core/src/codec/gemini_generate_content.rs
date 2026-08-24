// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Built-in codec for the Gemini generateContent API.
//!
//! Implements [`LlmCodec`] (request decode/encode) and [`LlmResponseCodec`]
//! (response decode) for the Gemini generateContent API format.
//!
//! Gemini function-call IDs are normalized as tool-call IDs and echoed in
//! `functionResponse.id`. For legacy responses that omit an ID, the codec uses
//! the function name only when it is unambiguous; duplicate effective IDs are
//! rejected because their function responses cannot be correlated safely.

use std::collections::{HashMap, HashSet};

use serde::Deserialize;

use crate::api::llm::LlmRequest;
use crate::api::runtime::{BuiltinLlmCodec, LlmCodecIdentity};
use crate::error::{FlowError, Result};
use crate::json::Json;

use super::request::{
    AnnotatedLlmRequest, ContentPart, FunctionCall, FunctionDefinition, GenerationParams, Message,
    MessageContent, ToolCall, ToolDefinition,
};
use super::resolve::{ProviderSurface, ProviderSurfaceDescriptor};
use super::response::{
    AnnotatedLlmResponse, FinishReason, ResponseToolCall, Usage, estimate_cost_for_provider,
    infer_model_provider,
};
use super::traits::{LlmCodec, LlmResponseCodec};

const GEMINI_PROVIDER: &str = "gemini";

// ---------------------------------------------------------------------------
// Public codec struct
// ---------------------------------------------------------------------------

/// Built-in codec for the Gemini generateContent API.
pub struct GeminiGenerateContentCodec;

pub(crate) const PROVIDER_SURFACE: ProviderSurfaceDescriptor = ProviderSurfaceDescriptor {
    surface: ProviderSurface::GeminiGenerateContent,
    detect_request: |obj, _hint| obj.contains_key("contents"),
    detect_response: detect_gemini_response,
    decode_request: |request| GeminiGenerateContentCodec.decode(request),
    decode_response: |raw| GeminiGenerateContentCodec.decode_response(raw),
    codec_name: "gemini_generate_content",
    request_codec: || std::sync::Arc::new(GeminiGenerateContentCodec),
    response_codec: || std::sync::Arc::new(GeminiGenerateContentCodec),
    streaming_codec: || Box::new(GeminiGenerateContentStreamingCodec::new()),
};

// ---------------------------------------------------------------------------
// Private serde intermediates for response decode
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct RawGeminiGenerateContentResponse {
    candidates: Option<Vec<RawCandidate>>,
    #[serde(rename = "usageMetadata")]
    usage_metadata: Option<RawUsageMetadata>,
    #[serde(rename = "modelVersion")]
    model_version: Option<String>,
    #[serde(rename = "responseId")]
    response_id: Option<String>,
    #[serde(flatten)]
    extra: serde_json::Map<String, Json>,
}

#[derive(Deserialize)]
struct RawCandidate {
    content: Option<RawContent>,
    #[serde(rename = "finishReason")]
    finish_reason: Option<String>,
    #[serde(flatten)]
    extra: serde_json::Map<String, Json>,
}

#[derive(Deserialize)]
struct RawContent {
    parts: Option<Vec<Json>>,
}

#[derive(Deserialize)]
struct RawUsageMetadata {
    #[serde(rename = "promptTokenCount")]
    prompt_token_count: Option<u64>,
    #[serde(rename = "candidatesTokenCount")]
    candidates_token_count: Option<u64>,
    #[serde(rename = "totalTokenCount")]
    total_token_count: Option<u64>,
    #[serde(rename = "cachedContentTokenCount")]
    cached_content_token_count: Option<u64>,
    #[serde(rename = "thoughtsTokenCount")]
    thoughts_token_count: Option<u64>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Top-level request keys modeled by this codec; all others go into `extra`.
const MODELED_REQUEST_KEYS: &[&str] = &[
    "contents",
    "systemInstruction",
    "tools",
    "generationConfig",
    "model",
];

/// Map Gemini `finishReason` (and presence of `functionCall` parts) to [`FinishReason`].
///
/// Explicit provider reasons are always honored first. The `has_tool_calls` heuristic
/// is only applied when the provider either omits the reason entirely or signals
/// successful generation (`STOP` / `TOOL_CODE` / `FINISH_REASON_UNSPECIFIED` / empty).
/// This prevents `MAX_TOKENS`, safety reasons, or error codes like
/// `MALFORMED_FUNCTION_CALL` from being obscured by the presence of function-call parts.
///
/// Reference (GenerateContent API):
/// - absent / empty / `FINISH_REASON_UNSPECIFIED` → derive from content (ToolUse or None)
/// - `STOP` → Complete; or ToolUse when function-call parts are present
/// - `TOOL_CODE` → ToolUse
/// - `MAX_TOKENS` → Length
/// - `SAFETY` / `RECITATION` / `BLOCKLIST` / `PROHIBITED_CONTENT` / `SPII` /
///   `LANGUAGE` / `IMAGE_SAFETY` / `IMAGE_PROHIBITED_CONTENT` / `IMAGE_RECITATION` /
///   `ESCALATION` → ContentFilter
/// - anything else → Unknown (e.g. `MALFORMED_FUNCTION_CALL`, `UNEXPECTED_TOOL_CALL`)
fn map_finish_reason(reason: Option<&str>, has_tool_calls: bool) -> Option<FinishReason> {
    match reason {
        // Absent / unspecified: derive from response content.
        None | Some("") | Some("FINISH_REASON_UNSPECIFIED") => {
            if has_tool_calls {
                Some(FinishReason::ToolUse)
            } else {
                None
            }
        }
        Some(r) => Some(match r {
            // Successful stop: ToolUse when tool calls are present, Complete otherwise.
            "STOP" => {
                if has_tool_calls {
                    FinishReason::ToolUse
                } else {
                    FinishReason::Complete
                }
            }
            "TOOL_CODE" => FinishReason::ToolUse,
            "MAX_TOKENS" => FinishReason::Length,
            // All policy / safety terminations map to ContentFilter, regardless of whether
            // any function-call parts happened to be present.
            "SAFETY"
            | "RECITATION"
            | "BLOCKLIST"
            | "PROHIBITED_CONTENT"
            | "SPII"
            | "LANGUAGE"
            | "IMAGE_SAFETY"
            | "IMAGE_PROHIBITED_CONTENT"
            | "IMAGE_RECITATION"
            | "ESCALATION" => FinishReason::ContentFilter,
            // Unknown / future codes (e.g. MALFORMED_FUNCTION_CALL, UNEXPECTED_TOOL_CALL).
            other => FinishReason::Unknown(other.to_string()),
        }),
    }
}

fn map_prompt_block_reason(reason: Option<&str>) -> Option<FinishReason> {
    match reason {
        None | Some("") | Some("BLOCK_REASON_UNSPECIFIED") => None,
        Some("SAFETY" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "IMAGE_SAFETY") => {
            Some(FinishReason::ContentFilter)
        }
        Some(other) => Some(FinishReason::Unknown(other.to_string())),
    }
}

fn detect_gemini_response(obj: &serde_json::Map<String, Json>) -> bool {
    obj.get("candidates").is_some_and(Json::is_array)
        || obj.get("promptFeedback").is_some_and(|feedback| {
            feedback.as_object().is_some_and(|feedback| {
                feedback.get("blockReason").is_some() || feedback.get("safetyRatings").is_some()
            })
        })
}

fn prompt_feedback_block_reason(extra: &serde_json::Map<String, Json>) -> Result<Option<&str>> {
    let Some(feedback) = extra.get("promptFeedback") else {
        return Ok(None);
    };
    let Some(feedback) = feedback.as_object() else {
        return Err(FlowError::InvalidArgument(
            "Gemini response promptFeedback must be an object".into(),
        ));
    };
    match feedback.get("blockReason") {
        Some(Json::String(reason)) => Ok(Some(reason.as_str())),
        Some(_) => Err(FlowError::InvalidArgument(
            "Gemini response promptFeedback.blockReason must be a string".into(),
        )),
        None => Ok(None),
    }
}

/// Extract a `tool_call_id` from a serialized `Message::Tool` JSON object.
///
/// Returns `Err` when the field is absent, empty, or non-string.
fn extract_tool_call_id(obj: &serde_json::Map<String, Json>) -> Result<&str> {
    match obj.get("tool_call_id") {
        Some(Json::String(s)) if !s.is_empty() => Ok(s.as_str()),
        Some(Json::String(_)) => Err(FlowError::InvalidArgument(
            "Gemini encoder: Message::Tool has an empty tool_call_id".into(),
        )),
        Some(_) => Err(FlowError::InvalidArgument(
            "Gemini encoder: Message::Tool tool_call_id must be a string".into(),
        )),
        None => Err(FlowError::Internal(
            "Message::Tool has no tool_call_id".into(),
        )),
    }
}

/// Parse an optional Gemini `id` field: absent → `Ok(None)`; present non-empty string
/// → `Ok(Some(s))`; present empty string or non-string → `Err`.
fn parse_optional_id(obj: &serde_json::Map<String, Json>, context: &str) -> Result<Option<String>> {
    match obj.get("id") {
        None => Ok(None),
        Some(Json::String(s)) if !s.is_empty() => Ok(Some(s.clone())),
        Some(Json::String(_)) => Err(FlowError::InvalidArgument(format!(
            "Gemini generateContent {context}.id must be a non-empty string"
        ))),
        Some(_) => Err(FlowError::InvalidArgument(format!(
            "Gemini generateContent {context}.id must be a string"
        ))),
    }
}

/// Reject an ID that would make function-call correlation ambiguous in one Gemini content item.
fn ensure_unique_gemini_function_id(
    seen_ids: &mut HashSet<String>,
    id: &str,
    part_kind: &str,
) -> Result<()> {
    if seen_ids.insert(id.to_string()) {
        return Ok(());
    }
    Err(FlowError::InvalidArgument(format!(
        "Gemini {part_kind} parts have duplicate effective id '{id}'; distinct ids are required to correlate function responses"
    )))
}

fn is_gemini_part_data_key(key: &str) -> bool {
    matches!(
        key,
        "text"
            | "inlineData"
            | "fileData"
            | "functionCall"
            | "functionResponse"
            | "executableCode"
            | "codeExecutionResult"
    )
}

fn gemini_part_data_keys(obj: &serde_json::Map<String, Json>) -> Vec<&str> {
    obj.keys()
        .map(String::as_str)
        .filter(|key| is_gemini_part_data_key(key))
        .collect()
}

fn validate_single_gemini_part_data_field<'a>(
    obj: &'a serde_json::Map<String, Json>,
    context: &str,
) -> Result<Option<&'a str>> {
    let keys = gemini_part_data_keys(obj);
    if keys.len() > 1 {
        return Err(FlowError::InvalidArgument(format!(
            "Gemini generateContent {context} part must not contain multiple data fields: {}",
            keys.join(", ")
        )));
    }
    Ok(keys.first().copied())
}

/// Convert visible Gemini content parts into normalized message content.
///
/// Thought parts and tool-call/tool-response parts are not user-visible message
/// content. Text-only content stays as `MessageContent::Text` for compatibility;
/// mixed or metadata-bearing Gemini parts are exposed as provider-native content
/// blocks so middleware can inspect and sanitize them.
fn gemini_parts_to_message_content(
    parts: &[Json],
    context: &str,
) -> Result<Option<MessageContent>> {
    let mut texts: Vec<String> = Vec::new();
    let mut content_parts: Vec<ContentPart> = Vec::new();
    let mut requires_parts = false;

    for part in parts {
        let obj = part.as_object().ok_or_else(|| {
            FlowError::InvalidArgument(format!(
                "Gemini generateContent {context} parts entry must be an object"
            ))
        })?;
        if obj.get("thought").and_then(Json::as_bool) == Some(true) {
            continue;
        }

        let Some(data_key) = validate_single_gemini_part_data_field(obj, context)? else {
            requires_parts = true;
            content_parts.push(ContentPart::ProviderNative {
                provider: GEMINI_PROVIDER.into(),
                kind: "unknown".into(),
                value: part.clone(),
            });
            continue;
        };

        match data_key {
            "functionCall" | "functionResponse" => continue,
            "text" => {
                let text = obj.get("text").and_then(Json::as_str).ok_or_else(|| {
                    FlowError::InvalidArgument(format!(
                        "Gemini generateContent {context} parts[].text must be a string"
                    ))
                })?;
                let extra: serde_json::Map<String, Json> = obj
                    .iter()
                    .filter(|(key, _)| key.as_str() != "text")
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect();
                if !extra.is_empty() {
                    requires_parts = true;
                }
                texts.push(text.to_string());
                content_parts.push(ContentPart::Text {
                    text: text.to_string(),
                    extra,
                });
            }
            native_key => {
                requires_parts = true;
                content_parts.push(ContentPart::ProviderNative {
                    provider: GEMINI_PROVIDER.into(),
                    kind: native_key.to_string(),
                    value: part.clone(),
                });
            }
        }
    }

    Ok(if content_parts.is_empty() {
        None
    } else if requires_parts {
        Some(MessageContent::Parts(content_parts))
    } else {
        Some(MessageContent::Text(texts.join("\n")))
    })
}

fn validate_gemini_nested_function_response_part(part: &Json) -> Result<&str> {
    let obj = part.as_object().ok_or_else(|| {
        FlowError::InvalidArgument("Gemini functionResponse.parts entry must be an object".into())
    })?;
    let data_key = validate_single_gemini_part_data_field(obj, "functionResponse.parts")?;
    if matches!(data_key, Some("functionCall" | "functionResponse")) {
        return Err(FlowError::InvalidArgument(
            "Gemini functionResponse.parts must not contain nested functionCall/functionResponse"
                .into(),
        ));
    }
    if data_key == Some("text") && part.get("text").is_some_and(|v| !v.is_string()) {
        return Err(FlowError::InvalidArgument(
            "Gemini functionResponse.parts[].text must be a string".into(),
        ));
    }
    Ok(data_key.unwrap_or("unknown"))
}

fn gemini_function_response_to_message_content(fr: &Json) -> Result<MessageContent> {
    let response = fr
        .get("response")
        .ok_or_else(|| {
            FlowError::InvalidArgument(
                "Gemini functionResponse is missing required 'response'".into(),
            )
        })?
        .clone();
    let content_str = serde_json::to_string(&response).unwrap_or_else(|_| "{}".into());
    let Some(parts_value) = fr.get("parts") else {
        return Ok(MessageContent::Text(content_str));
    };
    let nested_parts = parts_value.as_array().ok_or_else(|| {
        FlowError::InvalidArgument("Gemini functionResponse.parts must be an array".into())
    })?;

    let mut content_parts = Vec::with_capacity(nested_parts.len() + 1);
    content_parts.push(ContentPart::Text {
        text: content_str,
        extra: Default::default(),
    });
    for nested_part in nested_parts {
        let kind = validate_gemini_nested_function_response_part(nested_part)?;
        content_parts.push(ContentPart::ProviderNative {
            provider: GEMINI_PROVIDER.into(),
            kind: kind.to_string(),
            value: nested_part.clone(),
        });
    }
    Ok(MessageContent::Parts(content_parts))
}

/// Validate and extract content from a Gemini response `parts` array.
///
/// Skips functionCall and thought parts. The Gemini API spec defines `thought`
/// as a boolean; a non-boolean value is treated as an ordinary field, not a
/// thought part.
fn extract_parts_message_content(parts: &[Json]) -> Result<Option<MessageContent>> {
    for part in parts {
        if part.get("functionResponse").is_some() {
            return Err(FlowError::InvalidArgument(
                "Gemini response parts must not contain functionResponse".into(),
            ));
        }
    }
    gemini_parts_to_message_content(parts, "response")
}

/// Extract `functionCall` parts from a Gemini `parts` array as [`ResponseToolCall`]s.
///
/// Uses the `id` field when the model supplies one; falls back to the function name
/// when absent (older models omit it). Returns `Err` for malformed functionCall
/// shapes (non-object, missing name, empty id, non-object args). Non-functionCall
/// parts are skipped; callers must validate text+functionCall conflicts separately.
fn extract_parts_tool_calls(parts: &[Json]) -> Result<Option<Vec<ResponseToolCall>>> {
    let mut calls: Vec<ResponseToolCall> = Vec::new();
    let mut seen_ids = HashSet::new();
    for p in parts {
        let Some(fc) = p.get("functionCall") else {
            continue;
        };
        let fc_obj = fc.as_object().ok_or_else(|| {
            FlowError::InvalidArgument("Gemini response functionCall must be an object".into())
        })?;

        let name = match fc_obj.get("name") {
            Some(Json::String(s)) if !s.is_empty() => s.clone(),
            Some(Json::String(_)) => {
                return Err(FlowError::InvalidArgument(
                    "Gemini response functionCall.name must be a non-empty string".into(),
                ));
            }
            Some(_) => {
                return Err(FlowError::InvalidArgument(
                    "Gemini response functionCall.name must be a string".into(),
                ));
            }
            None => {
                return Err(FlowError::InvalidArgument(
                    "Gemini response functionCall is missing 'name'".into(),
                ));
            }
        };

        let id =
            parse_optional_id(fc_obj, "response functionCall")?.unwrap_or_else(|| name.clone());
        ensure_unique_gemini_function_id(&mut seen_ids, &id, "functionCall")?;

        let arguments = match fc_obj.get("args") {
            None => Json::Object(Default::default()),
            Some(v) if v.is_object() => v.clone(),
            Some(_) => {
                return Err(FlowError::InvalidArgument(
                    "Gemini response functionCall.args must be an object".into(),
                ));
            }
        };

        calls.push(ResponseToolCall {
            id,
            name,
            arguments,
        });
    }
    Ok(if calls.is_empty() { None } else { Some(calls) })
}

/// Map Gemini `usageMetadata` to a normalized [`Usage`] and the raw thinking-token count.
///
/// Returns `(usage, thoughts_token_count)`. The thoughts token count is kept
/// separate because it belongs in `ApiSpecificResponse::GeminiGenerateContent`, not in the
/// provider-neutral `Usage` struct.
fn map_usage(meta: Option<RawUsageMetadata>, model: Option<&str>) -> (Option<Usage>, Option<u64>) {
    let model_provider = infer_model_provider("google", model);
    let Some(m) = meta else {
        return (None, None);
    };
    let thoughts_token_count = m.thoughts_token_count;
    let prompt = m.prompt_token_count;
    let completion = m.candidates_token_count;
    // When totalTokenCount is absent, compute a fallback from every count that is
    // available.  Thinking tokens count as billable output and must be included even
    // when candidatesTokenCount is missing (e.g. thinking-only partial responses).
    let total = m.total_token_count.or_else(|| {
        let has_any = prompt.is_some() || completion.is_some() || thoughts_token_count.is_some();
        has_any.then(|| {
            prompt.unwrap_or(0) + completion.unwrap_or(0) + thoughts_token_count.unwrap_or(0)
        })
    });
    // For cost estimation: treat thinking tokens as additional completion tokens
    // so the pricing table applies the output-token rate to them even when
    // candidatesTokenCount is absent.
    let completion_for_cost = match (completion, thoughts_token_count) {
        (Some(c), Some(t)) => Some(c + t),
        (Some(c), None) => Some(c),
        (None, Some(t)) => Some(t),
        (None, None) => None,
    };
    let usage_for_cost = Usage {
        prompt_tokens: prompt,
        completion_tokens: completion_for_cost,
        total_tokens: total,
        cache_read_tokens: m.cached_content_token_count,
        cache_write_tokens: None,
        cost: None,
    };
    let cost = model
        .and_then(|m| estimate_cost_for_provider(model_provider.as_deref(), m, &usage_for_cost));
    let usage = Usage {
        prompt_tokens: prompt,
        completion_tokens: completion,
        total_tokens: total,
        cache_read_tokens: m.cached_content_token_count,
        cache_write_tokens: None,
        cost,
    };
    (Some(usage), thoughts_token_count)
}

/// Validate the shape of a Gemini `systemInstruction` value.
fn validate_system_instruction(val: &Json) -> Result<()> {
    let obj = val.as_object().ok_or_else(|| {
        FlowError::InvalidArgument("Gemini systemInstruction must be an object".into())
    })?;
    if obj.get("role").is_some_and(|v| !v.is_string()) {
        return Err(FlowError::InvalidArgument(
            "Gemini systemInstruction.role must be a string".into(),
        ));
    }
    let parts = obj
        .get("parts")
        .ok_or_else(|| {
            FlowError::InvalidArgument("Gemini systemInstruction must have a 'parts' field".into())
        })?
        .as_array()
        .ok_or_else(|| {
            FlowError::InvalidArgument("Gemini systemInstruction.parts must be an array".into())
        })?;
    for part in parts {
        let part_obj = part.as_object().ok_or_else(|| {
            FlowError::InvalidArgument(
                "Gemini systemInstruction.parts entry must be an object".into(),
            )
        })?;
        let data_key = validate_single_gemini_part_data_field(part_obj, "systemInstruction")?;
        if data_key != Some("text") {
            return Err(FlowError::InvalidArgument(
                "Gemini systemInstruction.parts entries must be text parts".into(),
            ));
        }
        if part.get("text").is_some_and(|v| !v.is_string()) {
            return Err(FlowError::InvalidArgument(
                "Gemini systemInstruction.parts[].text must be a string".into(),
            ));
        }
    }
    Ok(())
}

/// Extract the text content from a Gemini `systemInstruction` value.
fn system_instruction_text(val: &Json) -> Option<String> {
    let parts = val.get("parts")?.as_array()?;
    let text = parts
        .iter()
        .filter_map(|p| p.get("text")?.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    if text.is_empty() { None } else { Some(text) }
}

/// Convert a single Gemini `contents` item to zero or more normalized [`Message`]s.
///
/// - `functionResponse` parts (user role) → one `Message::Tool` per part, each with
///   `tool_call_id` set to the `id` field when present, falling back to `name`.
/// - `functionCall` parts (model role) → `Message::Assistant { tool_calls: Some([…]) }`
/// - text parts → plain message content
fn gemini_content_to_messages(content: &Json) -> Result<Vec<Message>> {
    let obj = content.as_object().ok_or_else(|| {
        FlowError::InvalidArgument("Gemini contents item must be an object".into())
    })?;
    let role = gemini_content_role(obj)?;
    let parts = obj.get("parts").and_then(Json::as_array).ok_or_else(|| {
        FlowError::InvalidArgument("Gemini contents item must have an array 'parts' field".into())
    })?;
    let (fr_parts, fn_call_parts) = validate_gemini_content_parts(parts)?;
    validate_gemini_content_roles(role, &fr_parts, &fn_call_parts)?;
    if !fr_parts.is_empty() {
        return gemini_function_response_messages(parts, &fr_parts);
    }
    let content = gemini_parts_to_message_content(parts, "request")?;
    if !fn_call_parts.is_empty() {
        return gemini_function_call_messages(content, &fn_call_parts);
    }
    Ok(vec![gemini_plain_message(role, content)])
}

fn gemini_content_role(obj: &serde_json::Map<String, Json>) -> Result<&str> {
    match obj.get("role") {
        None => Ok("user"),
        Some(Json::String(role)) if role == "user" || role == "model" => Ok(role),
        Some(Json::String(other)) => Err(FlowError::InvalidArgument(format!(
            "Gemini contents item has unsupported role '{other}'; expected 'user' or 'model'"
        ))),
        Some(_) => Err(FlowError::InvalidArgument(
            "Gemini contents item 'role' must be a string".into(),
        )),
    }
}

fn validate_gemini_content_parts(parts: &[Json]) -> Result<(Vec<&Json>, Vec<&Json>)> {
    let mut responses = Vec::new();
    let mut calls = Vec::new();
    for part in parts {
        let part_obj = part.as_object().ok_or_else(|| {
            FlowError::InvalidArgument("Gemini parts item must be an object".into())
        })?;
        match validate_single_gemini_part_data_field(part_obj, "request")? {
            Some("functionResponse") => {
                validate_gemini_function_response_part(part)?;
                responses.push(part);
            }
            Some("functionCall") => {
                validate_gemini_function_call_part(part)?;
                calls.push(part);
            }
            Some("text") if part.get("text").is_some_and(|value| !value.is_string()) => {
                return Err(FlowError::InvalidArgument(
                    "Gemini parts item 'text' must be a string".into(),
                ));
            }
            _ => {}
        }
    }
    if !responses.is_empty() && !calls.is_empty() {
        return Err(FlowError::InvalidArgument(
            "Gemini contents item must not contain both functionResponse and functionCall parts"
                .into(),
        ));
    }
    Ok((responses, calls))
}

fn validate_gemini_function_response_part(part: &Json) -> Result<()> {
    let response = part
        .get("functionResponse")
        .and_then(Json::as_object)
        .ok_or_else(|| {
            FlowError::InvalidArgument("Gemini functionResponse must be an object".into())
        })?;
    if response
        .get("name")
        .and_then(Json::as_str)
        .is_none_or(str::is_empty)
    {
        return Err(FlowError::InvalidArgument(
            "Gemini functionResponse is missing a non-empty 'name'".into(),
        ));
    }
    match response.get("response") {
        None => {
            return Err(FlowError::InvalidArgument(
                "Gemini functionResponse is missing required 'response'".into(),
            ));
        }
        Some(value) if !value.is_object() => {
            return Err(FlowError::InvalidArgument(
                "Gemini functionResponse.response must be an object".into(),
            ));
        }
        Some(_) => {}
    }
    if let Some(parts) = response.get("parts") {
        for part in parts.as_array().ok_or_else(|| {
            FlowError::InvalidArgument("Gemini functionResponse.parts must be an array".into())
        })? {
            validate_gemini_nested_function_response_part(part)?;
        }
    }
    parse_optional_id(response, "functionResponse")?;
    Ok(())
}

fn validate_gemini_function_call_part(part: &Json) -> Result<()> {
    let call = part
        .get("functionCall")
        .and_then(Json::as_object)
        .ok_or_else(|| {
            FlowError::InvalidArgument("Gemini functionCall must be an object".into())
        })?;
    if call
        .get("name")
        .and_then(Json::as_str)
        .is_none_or(str::is_empty)
    {
        return Err(FlowError::InvalidArgument(
            "Gemini functionCall is missing a non-empty 'name'".into(),
        ));
    }
    if call.get("args").is_some_and(|args| !args.is_object()) {
        return Err(FlowError::InvalidArgument(
            "Gemini functionCall.args must be an object".into(),
        ));
    }
    Ok(())
}

fn validate_gemini_content_roles(role: &str, responses: &[&Json], calls: &[&Json]) -> Result<()> {
    if !responses.is_empty() && role != "user" {
        return Err(FlowError::InvalidArgument(format!(
            "Gemini functionResponse parts must be in a 'user' role content item, got '{role}'"
        )));
    }
    if !calls.is_empty() && role != "model" {
        return Err(FlowError::InvalidArgument(format!(
            "Gemini functionCall parts must be in a 'model' role content item, got '{role}'"
        )));
    }
    Ok(())
}

fn gemini_function_response_messages(parts: &[Json], responses: &[&Json]) -> Result<Vec<Message>> {
    if parts.iter().any(|part| {
        part.get("functionResponse").is_none()
            && part.get("thought").and_then(Json::as_bool) != Some(true)
    }) {
        return Err(FlowError::InvalidArgument(
            "Gemini contents item must not mix functionResponse with visible/native parts".into(),
        ));
    }
    let mut seen_ids = HashSet::new();
    responses
        .iter()
        .map(|part| {
            let response = part
                .get("functionResponse")
                .and_then(Json::as_object)
                .unwrap();
            let name = response
                .get("name")
                .and_then(Json::as_str)
                .unwrap()
                .to_string();
            let id = parse_optional_id(response, "functionResponse")?.unwrap_or(name);
            ensure_unique_gemini_function_id(&mut seen_ids, &id, "functionResponse")?;
            Ok(Message::Tool {
                content: gemini_function_response_to_message_content(
                    part.get("functionResponse").unwrap(),
                )?,
                tool_call_id: id,
            })
        })
        .collect()
}

fn gemini_function_call_messages(
    content: Option<MessageContent>,
    calls: &[&Json],
) -> Result<Vec<Message>> {
    let mut seen_ids = HashSet::new();
    let tool_calls = calls
        .iter()
        .map(|part| {
            let call = part.get("functionCall").and_then(Json::as_object).unwrap();
            let name = call.get("name").and_then(Json::as_str).unwrap().to_string();
            let id = parse_optional_id(call, "functionCall")?.unwrap_or_else(|| name.clone());
            ensure_unique_gemini_function_id(&mut seen_ids, &id, "functionCall")?;
            let args = call
                .get("args")
                .cloned()
                .unwrap_or_else(|| Json::Object(Default::default()));
            Ok(ToolCall {
                id,
                call_type: "function".into(),
                function: FunctionCall {
                    name,
                    arguments: serde_json::to_string(&args).unwrap_or_else(|_| "{}".into()),
                },
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(vec![Message::Assistant {
        content,
        tool_calls: Some(tool_calls),
        name: None,
    }])
}

fn gemini_plain_message(role: &str, content: Option<MessageContent>) -> Message {
    let content = content.unwrap_or_else(|| MessageContent::Text(String::new()));
    if role == "model" {
        Message::Assistant {
            content: Some(content),
            tool_calls: None,
            name: None,
        }
    } else {
        Message::User {
            content,
            name: None,
        }
    }
}

/// Build the `functionCall` JSON object for a single normalized tool call.
///
/// Returns `Err` when the tool call has a missing or empty function name,
/// an invalid `id`, or non-object arguments.
/// Used by both the fresh-encode path and the patch path; the patch path
/// additionally merges this into an original part to preserve part-level metadata.
fn tool_call_to_fc_obj(tc: &Json) -> Result<serde_json::Map<String, Json>> {
    let fn_obj = tc.get("function");
    let name = fn_obj
        .and_then(|f| f.get("name"))
        .and_then(Json::as_str)
        .unwrap_or("");
    if name.is_empty() {
        return Err(FlowError::InvalidArgument(
            "Gemini encoder: tool call is missing a non-empty function name".into(),
        ));
    }
    let id = match tc.get("id") {
        None => None,
        Some(Json::String(s)) if !s.is_empty() => Some(s.as_str()),
        Some(Json::String(_)) => {
            return Err(FlowError::InvalidArgument(format!(
                "Gemini encoder: tool call for '{name}' has an empty 'id'"
            )));
        }
        Some(_) => {
            return Err(FlowError::InvalidArgument(format!(
                "Gemini encoder: tool call for '{name}' has a non-string 'id'"
            )));
        }
    };
    let arguments = fn_obj
        .and_then(|f| f.get("arguments"))
        .and_then(Json::as_str);
    // If arguments is present but not valid JSON, surface the error rather than
    // silently replacing it with {} which would hide model or interceptor output.
    let args: Json = match arguments {
        None => Json::Object(Default::default()),
        Some(a) => {
            let v: Json = serde_json::from_str(a).map_err(|_| {
                FlowError::InvalidArgument(format!(
                    "Gemini encoder: function call '{name}' has arguments that are not valid JSON: {a}"
                ))
            })?;
            if !v.is_object() {
                return Err(FlowError::InvalidArgument(format!(
                    "Gemini encoder: function call '{name}' arguments must be a JSON object, \
                     not {v}"
                )));
            }
            v
        }
    };
    let mut fc_obj = serde_json::Map::new();
    fc_obj.insert("name".into(), Json::String(name.to_string()));
    if let Some(id) = id {
        fc_obj.insert("id".into(), Json::String(id.to_string()));
    }
    fc_obj.insert("args".into(), args);
    Ok(fc_obj)
}

fn gemini_content_parts_from_normalized(content: &Json) -> Result<(Vec<Json>, bool)> {
    match content {
        Json::Null => Ok((Vec::new(), false)),
        Json::String(s) => {
            if s.is_empty() {
                Ok((Vec::new(), false))
            } else {
                Ok((vec![serde_json::json!({"text": s})], false))
            }
        }
        Json::Array(parts) => {
            let mut out = Vec::with_capacity(parts.len());
            for part in parts {
                let obj = part.as_object().ok_or_else(|| {
                    FlowError::InvalidArgument(
                        "Gemini encoder: content parts must be objects".into(),
                    )
                })?;
                let part_type = match obj.get("type") {
                    None => "text",
                    Some(Json::String(s)) => s.as_str(),
                    Some(other) => {
                        return Err(FlowError::InvalidArgument(format!(
                            "Gemini encoder: content part 'type' must be a string, got: {other}"
                        )));
                    }
                };
                match part_type {
                    "text" => {
                        let text = obj.get("text").and_then(Json::as_str).ok_or_else(|| {
                            FlowError::InvalidArgument(
                                "Gemini encoder: text content part must have string 'text'".into(),
                            )
                        })?;
                        let mut gemini_part: serde_json::Map<String, Json> = obj
                            .iter()
                            .filter(|(key, _)| key.as_str() != "type")
                            .map(|(key, value)| (key.clone(), value.clone()))
                            .collect();
                        gemini_part.insert("text".into(), Json::String(text.to_string()));
                        out.push(Json::Object(gemini_part));
                    }
                    "provider_native" => {
                        out.push(provider_native_gemini_part_value(obj)?);
                    }
                    other => {
                        return Err(FlowError::InvalidArgument(format!(
                            "Gemini encoder: normalized content part type '{other}' cannot be \
                             encoded into Gemini content"
                        )));
                    }
                }
            }
            Ok((out, true))
        }
        other => Err(FlowError::InvalidArgument(format!(
            "Gemini encoder: message content must be a string, null, or array, got: {other}"
        ))),
    }
}

/// Convert a serialized normalized message to a Gemini `contents` item.
///
/// Returns `Ok(None)` for system messages (handled via `systemInstruction`).
/// Returns `Err(FlowError::InvalidArgument)` for message roles that have no
/// valid Gemini mapping — callers must surface these rather than silently drop
/// the message, which would be data loss.
///
/// `call_id_to_name` is used to resolve the function name for `Message::Tool`
/// when the caller set `tool_call_id` to the actual call ID. If the ID is not
/// found in the map, `tool_call_id` is used as the name (backward compat with
/// payloads where the name was stored as the ID).
fn normalized_to_gemini_content(
    msg_json: &Json,
    call_id_to_name: &HashMap<String, String>,
) -> Result<Option<Json>> {
    let obj = msg_json
        .as_object()
        .ok_or_else(|| FlowError::Internal("message is not an object".into()))?;
    let role = obj
        .get("role")
        .and_then(Json::as_str)
        .ok_or_else(|| FlowError::Internal("message has no role".into()))?;

    if role == "system" {
        return Ok(None);
    }

    // Message::Tool → functionResponse in a user turn.
    // tool_call_id carries the actual call ID (or function name as fallback).
    if role == "tool" {
        let call_id = extract_tool_call_id(obj)?;
        let fn_name = call_id_to_name
            .get(call_id)
            .map(String::as_str)
            .unwrap_or(call_id);
        let content_val = obj.get("content").unwrap_or(&Json::Null);
        let payload = function_response_payload_from_tool_content(content_val)?;
        let mut fr = serde_json::Map::new();
        fr.insert("id".into(), Json::String(call_id.to_string()));
        fr.insert("name".into(), Json::String(fn_name.to_string()));
        fr.insert("response".into(), payload.response);
        if let Some(parts) = payload.parts {
            fr.insert("parts".into(), Json::Array(parts));
        }
        return Ok(Some(serde_json::json!({
            "role": "user",
            "parts": [{"functionResponse": Json::Object(fr)}]
        })));
    }

    // Gemini only accepts "user" and "model". Return an error for anything else
    // so callers surface the data loss rather than silently dropping the message.
    let gemini_role = match role {
        "assistant" => "model",
        "user" => "user",
        other => {
            return Err(FlowError::InvalidArgument(format!(
                "Gemini encoder: role '{other}' has no Gemini equivalent \
                 (only 'user' and 'assistant' are supported)"
            )));
        }
    };

    let content_val = obj.get("content").unwrap_or(&Json::Null);
    let (content_parts, _) = gemini_content_parts_from_normalized(content_val)?;

    // Message::Assistant with tool_calls → functionCall parts.
    if let Some(tool_calls) = obj.get("tool_calls").and_then(Json::as_array) {
        let mut parts = content_parts.clone();
        for tc in tool_calls {
            let fc_obj = tool_call_to_fc_obj(tc)?;
            parts.push(serde_json::json!({"functionCall": Json::Object(fc_obj)}));
        }
        if !parts.is_empty() {
            return Ok(Some(
                serde_json::json!({"role": gemini_role, "parts": parts}),
            ));
        }
    }

    // Plain content message.
    let mut parts = content_parts;
    if parts.is_empty() {
        parts.push(serde_json::json!({"text": ""}));
    }
    Ok(Some(
        serde_json::json!({"role": gemini_role, "parts": parts}),
    ))
}

/// Reject normalized message content that contains non-text parts.
///
/// Used for normalized surfaces that Gemini can only encode as text
/// (`systemInstruction`). User/assistant and tool content have their own helpers
/// because Gemini-native parts are representable there.
fn reject_non_text_content_parts(content: &Json) -> Result<()> {
    let Json::Array(parts) = content else {
        return Ok(()); // String or Null — always encodable as text
    };
    for part in parts {
        let obj = part.as_object().ok_or_else(|| {
            FlowError::InvalidArgument(
                "Gemini encoder: normalized content parts must be objects".into(),
            )
        })?;
        let part_type = match obj.get("type") {
            None => "text",
            Some(Json::String(s)) => s.as_str(),
            Some(other) => {
                return Err(FlowError::InvalidArgument(format!(
                    "Gemini encoder: content part 'type' must be a string, got: {other}"
                )));
            }
        };
        if part_type != "text" {
            return Err(FlowError::InvalidArgument(format!(
                "Gemini encoder: normalized content part type '{part_type}' cannot be \
                 encoded into a Gemini text part; use a provider-native extra field \
                 for non-text content"
            )));
        }
        match obj.get("text") {
            Some(Json::String(_)) => {}
            Some(_) => {
                return Err(FlowError::InvalidArgument(
                    "Gemini encoder: text content part must have string 'text'".into(),
                ));
            }
            None => {
                return Err(FlowError::InvalidArgument(
                    "Gemini encoder: text content part is missing 'text'".into(),
                ));
            }
        }
    }
    Ok(())
}

/// Parse a tool-result string into a Gemini `functionResponse.response` object.
///
/// Gemini requires `response` to be a JSON object.  If the string parses as an
/// object it is used directly; any other JSON value (or non-JSON text) is wrapped
/// in `{"output": <value>}` so the response field is always object-shaped.
fn ensure_object_response(content_str: String) -> Json {
    match serde_json::from_str::<Json>(&content_str) {
        Ok(Json::Object(m)) => Json::Object(m),
        Ok(other) => serde_json::json!({"output": other}),
        Err(_) => serde_json::json!({"output": content_str}),
    }
}

struct GeminiGenerateContentFunctionResponsePayload {
    response: Json,
    parts: Option<Vec<Json>>,
}

fn provider_native_gemini_part_value(obj: &serde_json::Map<String, Json>) -> Result<Json> {
    let provider = obj.get("provider").and_then(Json::as_str).ok_or_else(|| {
        FlowError::InvalidArgument(
            "Gemini encoder: provider_native provider must be a string".into(),
        )
    })?;
    if provider != GEMINI_PROVIDER {
        return Err(FlowError::InvalidArgument(format!(
            "Gemini encoder: provider_native content part for provider \
             '{provider}' cannot be encoded by the Gemini generateContent codec"
        )));
    }
    match obj.get("kind") {
        Some(Json::String(s)) if !s.is_empty() => {}
        Some(Json::String(_)) => {
            return Err(FlowError::InvalidArgument(
                "Gemini encoder: provider_native content part kind must be non-empty".into(),
            ));
        }
        _ => {
            return Err(FlowError::InvalidArgument(
                "Gemini encoder: provider_native kind must be a string".into(),
            ));
        }
    }
    let value = obj.get("value").ok_or_else(|| {
        FlowError::InvalidArgument(
            "Gemini encoder: provider_native content part is missing 'value'".into(),
        )
    })?;
    let value_obj = value.as_object().ok_or_else(|| {
        FlowError::InvalidArgument("Gemini encoder: provider_native value must be an object".into())
    })?;
    let data_key = validate_single_gemini_part_data_field(value_obj, "provider_native")?;
    if matches!(data_key, Some("functionCall" | "functionResponse")) {
        return Err(FlowError::InvalidArgument(
            "Gemini encoder: provider_native content parts must not encode \
             functionCall or functionResponse; use tool_calls or tool messages"
                .into(),
        ));
    }
    if data_key == Some("text") && value.get("text").is_some_and(|v| !v.is_string()) {
        return Err(FlowError::InvalidArgument(
            "Gemini encoder: provider_native text part must have string 'text'".into(),
        ));
    }
    Ok(value.clone())
}

fn function_response_payload_from_tool_content(
    content: &Json,
) -> Result<GeminiGenerateContentFunctionResponsePayload> {
    match content {
        Json::Null | Json::String(_) => Ok(GeminiGenerateContentFunctionResponsePayload {
            response: ensure_object_response(extract_content_text(content)),
            parts: None,
        }),
        Json::Array(parts) => {
            let mut response_texts = Vec::new();
            let mut native_parts = Vec::new();
            for part in parts {
                let obj = part.as_object().ok_or_else(|| {
                    FlowError::InvalidArgument(
                        "Gemini encoder: normalized tool content parts must be objects".into(),
                    )
                })?;
                let part_type = match obj.get("type") {
                    None => "text",
                    Some(Json::String(s)) => s.as_str(),
                    Some(other) => {
                        return Err(FlowError::InvalidArgument(format!(
                            "Gemini encoder: content part 'type' must be a string, got: {other}"
                        )));
                    }
                };
                match part_type {
                    "text" => match obj.get("text") {
                        Some(Json::String(text)) => response_texts.push(text.clone()),
                        Some(_) => {
                            return Err(FlowError::InvalidArgument(
                                "Gemini encoder: text content part must have string 'text'".into(),
                            ));
                        }
                        None => {
                            return Err(FlowError::InvalidArgument(
                                "Gemini encoder: text content part is missing 'text'".into(),
                            ));
                        }
                    },
                    "provider_native" => {
                        let value = provider_native_gemini_part_value(obj)?;
                        validate_gemini_nested_function_response_part(&value)?;
                        native_parts.push(value);
                    }
                    other => {
                        return Err(FlowError::InvalidArgument(format!(
                            "Gemini encoder: normalized tool content part type '{other}' cannot be \
                             encoded into functionResponse.parts"
                        )));
                    }
                }
            }
            Ok(GeminiGenerateContentFunctionResponsePayload {
                response: ensure_object_response(response_texts.join("\n")),
                parts: Some(native_parts),
            })
        }
        other => Err(FlowError::InvalidArgument(format!(
            "Gemini encoder: message content must be a string, null, or array, got: {other}"
        ))),
    }
}

/// Extract a plain text string from a normalized `content` field.
fn extract_content_text(content: &Json) -> String {
    match content {
        Json::String(s) => s.clone(),
        Json::Array(parts) => parts
            .iter()
            .filter_map(|p| {
                if p.get("type")
                    .map(|ty| ty.as_str() == Some("text"))
                    .unwrap_or(true)
                {
                    p.get("text")?.as_str().map(str::to_string)
                } else {
                    p.as_str().map(str::to_string)
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn insert_serialized<T: serde::Serialize>(
    obj: &mut serde_json::Map<String, Json>,
    key: &str,
    value: &T,
    context: &str,
) -> Result<()> {
    let json = serde_json::to_value(value).map_err(|e| {
        FlowError::Internal(format!("Gemini generateContent {context} encode: {e}"))
    })?;
    obj.insert(key.into(), json);
    Ok(())
}

fn json_f64(v: f64, field: &str) -> Result<Json> {
    serde_json::Number::from_f64(v)
        .map(Json::Number)
        .ok_or_else(|| {
            FlowError::InvalidArgument(format!(
                "Gemini encoder: '{field}' value {v} is not a finite number"
            ))
        })
}

fn gemini_native_tool_fields(group: &serde_json::Map<String, Json>) -> Option<Json> {
    let native: serde_json::Map<String, Json> = group
        .iter()
        .filter(|(key, _)| key.as_str() != "functionDeclarations")
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    if native.is_empty() {
        None
    } else {
        Some(Json::Object(native))
    }
}

fn gemini_native_tool_kind(value: &Json) -> String {
    value
        .as_object()
        .and_then(|group| group.keys().next())
        .cloned()
        .unwrap_or_else(|| "unknown".to_string())
}

fn gemini_native_tool_keys(value: &Json) -> Vec<String> {
    let mut keys: Vec<String> = value
        .as_object()
        .map(|group| {
            group
                .keys()
                .filter(|key| key.as_str() != "functionDeclarations")
                .cloned()
                .collect()
        })
        .unwrap_or_default();
    keys.sort();
    keys
}

fn take_matching_native_group(
    native_groups: &[Json],
    native_used: &mut [bool],
    expected_keys: &[String],
) -> Option<Json> {
    for (idx, group) in native_groups.iter().enumerate() {
        if native_used[idx] {
            continue;
        }
        if gemini_native_tool_keys(group) == expected_keys {
            native_used[idx] = true;
            return Some(group.clone());
        }
    }
    None
}

// ---------------------------------------------------------------------------
// LlmResponseCodec
// ---------------------------------------------------------------------------

impl LlmResponseCodec for GeminiGenerateContentCodec {
    fn codec_identity(&self) -> LlmCodecIdentity {
        LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::GeminiGenerateContent)
    }

    fn decode_response(&self, response: &Json) -> Result<AnnotatedLlmResponse> {
        let raw: RawGeminiGenerateContentResponse = serde_json::from_value(response.clone())
            .map_err(|e| FlowError::Internal(format!("Gemini response decode: {e}")))?;

        let candidate = raw.candidates.as_ref().and_then(|c| c.first());

        let (message, tool_calls, has_tool_calls) = if let Some(c) = candidate {
            let parts = c.content.as_ref().and_then(|ct| ct.parts.as_deref());
            let msg = parts
                .map(extract_parts_message_content)
                .transpose()?
                .flatten();
            let tcs = parts.map(extract_parts_tool_calls).transpose()?.flatten();
            let has = tcs.is_some();
            (msg, tcs, has)
        } else {
            (None, None, false)
        };

        let prompt_block_reason = prompt_feedback_block_reason(&raw.extra)?;
        let finish_reason = candidate
            .and_then(|c| map_finish_reason(c.finish_reason.as_deref(), has_tool_calls))
            .or_else(|| map_prompt_block_reason(prompt_block_reason));

        let model = raw.model_version.clone();
        let (usage, thoughts_tokens) = map_usage(raw.usage_metadata, model.as_deref());

        // Capture candidate-level metadata (safetyRatings, groundingMetadata,
        // citationMetadata, thinking token count, …) that cannot be normalized
        // across providers.  thoughts_tokens lives here, not in shared Usage.
        let api_specific = {
            let mut extra = candidate.map(|c| c.extra.clone()).unwrap_or_default();
            extra.remove("api");
            let has_candidate_data = !extra.is_empty() || thoughts_tokens.is_some();
            if !has_candidate_data {
                None
            } else {
                let safety_ratings = extra.remove("safetyRatings");
                let grounding_metadata = extra.remove("groundingMetadata");
                let citation_metadata = extra.remove("citationMetadata");
                Some(
                    super::response::ApiSpecificResponse::GeminiGenerateContent {
                        thoughts_tokens,
                        safety_ratings,
                        grounding_metadata,
                        citation_metadata,
                        extra,
                    },
                )
            }
        };

        Ok(AnnotatedLlmResponse {
            id: raw.response_id,
            model,
            message,
            tool_calls,
            finish_reason,
            usage,
            optimization_summary: None,
            api_specific,
            extra: raw.extra,
        })
    }
}

// ---------------------------------------------------------------------------
// LlmCodec
// ---------------------------------------------------------------------------

impl LlmCodec for GeminiGenerateContentCodec {
    fn codec_identity(&self) -> LlmCodecIdentity {
        LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::GeminiGenerateContent)
    }

    fn decode(&self, request: &LlmRequest) -> Result<AnnotatedLlmRequest> {
        let obj = request
            .content
            .as_object()
            .ok_or_else(|| FlowError::Internal("request content is not an object".into()))?;

        let messages = decode_gemini_messages(obj)?;
        let params = decode_gemini_generation_params(obj)?;

        let tools = decode_gemini_tools(obj)?;

        // All unrecognized top-level keys go into extra.
        let extra: serde_json::Map<String, Json> = obj
            .iter()
            .filter(|(k, _)| !MODELED_REQUEST_KEYS.contains(&k.as_str()))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        let model = decode_gemini_model(obj)?;

        Ok(AnnotatedLlmRequest {
            messages,
            instructions: None,
            model,
            params,
            tools,
            tool_choice: None,
            store: None,
            previous_response_id: None,
            truncation: None,
            reasoning: None,
            include: None,
            user: None,
            metadata: None,
            service_tier: None,
            parallel_tool_calls: None,
            max_output_tokens: None,
            max_tool_calls: None,
            top_logprobs: None,
            stream: None,
            api_specific: None,
            extra,
        })
    }

    fn encode(&self, annotated: &AnnotatedLlmRequest, original: &LlmRequest) -> Result<LlmRequest> {
        let baseline = self.decode(original)?;
        validate_gemini_supported_fields(annotated, &baseline)?;
        let mut content = original.content.clone();
        let obj = content
            .as_object_mut()
            .ok_or_else(|| FlowError::Internal("original content is not an object".into()))?;

        if annotated.messages != baseline.messages {
            patch_gemini_messages(obj, &annotated.messages, &baseline.messages)?;
        }

        if annotated.params != baseline.params {
            patch_gemini_params(obj, annotated.params.as_ref())?;
        }

        if annotated.tools != baseline.tools {
            patch_gemini_tools(obj, annotated.tools.as_ref())?;
        }

        if annotated.model != baseline.model {
            match &annotated.model {
                Some(m) => {
                    obj.insert("model".into(), Json::String(m.clone()));
                }
                None => {
                    obj.remove("model");
                }
            }
        }

        patch_extra_fields(obj, &baseline.extra, &annotated.extra);

        Ok(LlmRequest {
            headers: original.headers.clone(),
            content,
        })
    }
}

// ---------------------------------------------------------------------------
// Baseline-aware patch helpers
// ---------------------------------------------------------------------------

/// Return `InvalidArgument` if the interceptor changed a field that the Gemini
/// encoder cannot represent.  These fields are always `None` in the Gemini
/// baseline because the Gemini request format has no equivalent concept.
/// Silently ignoring them would mean data the interceptor intended to act on
/// is lost without any signal.
fn validate_gemini_supported_fields(
    annotated: &AnnotatedLlmRequest,
    baseline: &AnnotatedLlmRequest,
) -> Result<()> {
    for message in &annotated.messages {
        match message {
            Message::System { name: Some(_), .. } => {
                return Err(FlowError::InvalidArgument(
                    "Gemini encoder: Message::System.name is not representable".into(),
                ));
            }
            Message::User { name: Some(_), .. } => {
                return Err(FlowError::InvalidArgument(
                    "Gemini encoder: Message::User.name is not representable".into(),
                ));
            }
            Message::Assistant { name: Some(_), .. } => {
                return Err(FlowError::InvalidArgument(
                    "Gemini encoder: Message::Assistant.name is not representable".into(),
                ));
            }
            _ => {}
        }
    }

    macro_rules! reject_if_changed {
        ($field:ident) => {
            if annotated.$field != baseline.$field {
                return Err(FlowError::InvalidArgument(format!(
                    "Gemini encoder: field '{}' is not representable in the \
                     Gemini generateContent API; use a provider-native extra field instead",
                    stringify!($field)
                )));
            }
        };
    }
    reject_if_changed!(instructions);
    reject_if_changed!(tool_choice);
    reject_if_changed!(store);
    reject_if_changed!(previous_response_id);
    reject_if_changed!(truncation);
    reject_if_changed!(reasoning);
    reject_if_changed!(include);
    reject_if_changed!(user);
    reject_if_changed!(metadata);
    reject_if_changed!(service_tier);
    reject_if_changed!(parallel_tool_calls);
    reject_if_changed!(max_output_tokens);
    reject_if_changed!(max_tool_calls);
    reject_if_changed!(top_logprobs);
    reject_if_changed!(stream);
    reject_if_changed!(api_specific);
    Ok(())
}

/// Patch `systemInstruction` and `contents` using prefix/suffix-aligned merging.
///
/// Computes the longest matching prefix and suffix between the annotated and
/// baseline non-system message lists, then maps each annotated position to its
/// corresponding original `contents` item. Unchanged positions are preserved
/// byte-identically. Pure insertions in the gap between prefix and suffix are
/// encoded fresh. Equal-size gaps (pure edits) pair 1-to-1 with their original
/// item so that `patch_changed_gemini_content` can carry over metadata.
fn patch_gemini_messages(
    obj: &mut serde_json::Map<String, Json>,
    messages: &[Message],
    baseline: &[Message],
) -> Result<()> {
    let msgs_json = serde_json::to_value(messages)
        .map_err(|e| FlowError::Internal(format!("Gemini messages encode: {e}")))?;
    let base_json = serde_json::to_value(baseline)
        .map_err(|e| FlowError::Internal(format!("Gemini baseline encode: {e}")))?;

    let msgs_arr = msgs_json.as_array().map(Vec::as_slice).unwrap_or(&[]);
    let base_arr = base_json.as_array().map(Vec::as_slice).unwrap_or(&[]);

    // Collect all system messages.  Gemini has one systemInstruction so multiple
    // system messages are merged by joining their text content with newlines.
    let is_sys = |m: &&Json| m.get("role").and_then(Json::as_str) == Some("system");
    let ann_sys_msgs: Vec<&Json> = msgs_arr.iter().filter(is_sys).collect();
    let base_sys_msgs: Vec<&Json> = base_arr.iter().filter(is_sys).collect();
    patch_gemini_system_instruction(obj, &ann_sys_msgs, &base_sys_msgs)?;

    let is_non_sys = |m: &&Json| m.get("role").and_then(Json::as_str) != Some("system");
    let ann_non_sys: Vec<&Json> = msgs_arr.iter().filter(is_non_sys).collect();
    let base_non_sys: Vec<&Json> = base_arr.iter().filter(is_non_sys).collect();
    patch_gemini_non_system_contents(obj, messages, &ann_non_sys, &base_non_sys)
}

fn patch_gemini_system_instruction(
    obj: &mut serde_json::Map<String, Json>,
    ann_sys_msgs: &[&Json],
    base_sys_msgs: &[&Json],
) -> Result<()> {
    if ann_sys_msgs == base_sys_msgs {
        return Ok(());
    }

    for m in ann_sys_msgs {
        reject_non_text_content_parts(m.get("content").unwrap_or(&Json::Null))?;
    }
    let text = ann_sys_msgs
        .iter()
        .filter_map(|m| {
            let t = extract_content_text(m.get("content").unwrap_or(&Json::Null));
            if t.is_empty() { None } else { Some(t) }
        })
        .collect::<Vec<_>>()
        .join("\n");
    if text.is_empty() {
        obj.remove("systemInstruction");
        return Ok(());
    }

    let mut si: serde_json::Map<String, Json> = obj
        .get("systemInstruction")
        .and_then(Json::as_object)
        .cloned()
        .unwrap_or_default();
    if si.get("role").is_some_and(|v| !v.is_string()) {
        return Err(FlowError::InvalidArgument(
            "Gemini systemInstruction.role must be a string".into(),
        ));
    }

    let orig_parts = si
        .get("parts")
        .and_then(Json::as_array)
        .cloned()
        .unwrap_or_default();
    validate_editable_gemini_system_parts(&orig_parts)?;
    si.insert(
        "parts".into(),
        Json::Array(rebuild_gemini_system_parts(&orig_parts, text)),
    );
    obj.insert("systemInstruction".into(), Json::Object(si));
    Ok(())
}

fn validate_editable_gemini_system_parts(orig_parts: &[Json]) -> Result<()> {
    let text_part_count = orig_parts
        .iter()
        .filter(|p| {
            p.get("thought").and_then(Json::as_bool) != Some(true) && p.get("text").is_some()
        })
        .count();
    let has_non_text_non_thought = orig_parts
        .iter()
        .any(|p| p.get("thought").and_then(Json::as_bool) != Some(true) && p.get("text").is_none());
    if text_part_count > 1 || has_non_text_non_thought {
        return Err(FlowError::InvalidArgument(
            "Gemini systemInstruction with multiple text parts or non-text parts \
             cannot be edited via the normalized layer; edit the raw provider payload directly"
                .into(),
        ));
    }
    Ok(())
}

fn rebuild_gemini_system_parts(orig_parts: &[Json], text: String) -> Vec<Json> {
    let mut new_parts = Vec::with_capacity(orig_parts.len().max(1));
    let mut text_part_placed = false;
    for orig_part in orig_parts {
        if orig_part.get("thought").and_then(Json::as_bool) == Some(true) {
            new_parts.push(orig_part.clone());
        } else if orig_part.get("text").is_some() && !text_part_placed {
            let mut p = orig_part.as_object().cloned().unwrap_or_default();
            p.insert("text".into(), Json::String(text.clone()));
            new_parts.push(Json::Object(p));
            text_part_placed = true;
        }
    }
    if !text_part_placed {
        new_parts.push(serde_json::json!({"text": text}));
    }
    new_parts
}

#[derive(Debug)]
struct GeminiGenerateContentMessageAlignment {
    prefix_len: usize,
    ann_gap_end: usize,
    base_gap_end: usize,
    ann_gap_len: usize,
    base_gap_len: usize,
}

impl GeminiGenerateContentMessageAlignment {
    fn new(ann_non_sys: &[&Json], base_non_sys: &[&Json]) -> Self {
        let prefix_len = ann_non_sys
            .iter()
            .zip(base_non_sys.iter())
            .take_while(|(a, b)| a == b)
            .count();
        let max_suffix = ann_non_sys
            .len()
            .saturating_sub(prefix_len)
            .min(base_non_sys.len().saturating_sub(prefix_len));
        let suffix_len = ann_non_sys[ann_non_sys.len().saturating_sub(max_suffix)..]
            .iter()
            .rev()
            .zip(
                base_non_sys[base_non_sys.len().saturating_sub(max_suffix)..]
                    .iter()
                    .rev(),
            )
            .take_while(|(a, b)| a == b)
            .count();
        let ann_gap_end = ann_non_sys.len().saturating_sub(suffix_len);
        let base_gap_end = base_non_sys.len().saturating_sub(suffix_len);
        Self {
            prefix_len,
            ann_gap_end,
            base_gap_end,
            ann_gap_len: ann_gap_end.saturating_sub(prefix_len),
            base_gap_len: base_gap_end.saturating_sub(prefix_len),
        }
    }

    fn base_idx_for(&self, i: usize) -> Option<usize> {
        if i < self.prefix_len {
            Some(i)
        } else if i >= self.ann_gap_end {
            Some(self.base_gap_end + (i - self.ann_gap_end))
        } else if self.ann_gap_len == self.base_gap_len {
            Some(self.prefix_len + (i - self.prefix_len))
        } else {
            None
        }
    }
}

fn gemini_content_idx_of_base_msg(
    orig_contents: &[Json],
    base_non_sys_len: usize,
) -> Result<Vec<usize>> {
    let mut mapping = Vec::with_capacity(base_non_sys_len);
    for (cidx, content) in orig_contents.iter().enumerate() {
        let n = gemini_content_to_messages(content)?.len().max(1);
        for _ in 0..n {
            if mapping.len() < base_non_sys_len {
                mapping.push(cidx);
            }
        }
    }
    let last = orig_contents.len().saturating_sub(1);
    while mapping.len() < base_non_sys_len {
        mapping.push(last);
    }
    Ok(mapping)
}

fn gemini_content_msg_counts(mapping: &[usize], orig_content_count: usize) -> Vec<usize> {
    let mut counts = vec![0usize; orig_content_count];
    for &cidx in mapping {
        if cidx < counts.len() {
            counts[cidx] += 1;
        }
    }
    counts
}

fn gemini_call_id_to_name(messages: &[Message]) -> HashMap<String, String> {
    messages
        .iter()
        .filter_map(|m| match m {
            Message::Assistant {
                tool_calls: Some(tcs),
                ..
            } => Some(
                tcs.iter()
                    .map(|tc| (tc.id.clone(), tc.function.name.clone())),
            ),
            _ => None,
        })
        .flatten()
        .collect()
}

fn push_fresh_gemini_content(
    new_contents: &mut Vec<Json>,
    message: &Json,
    call_id_to_name: &HashMap<String, String>,
) -> Result<()> {
    if let Some(item) = normalized_to_gemini_content(message, call_id_to_name)? {
        new_contents.push(item);
    }
    Ok(())
}

fn gemini_run_end(
    start: usize,
    ann_len: usize,
    orig_cidx: usize,
    alignment: &GeminiGenerateContentMessageAlignment,
    content_idx_of_base_msg: &[usize],
) -> usize {
    let mut run_end = start + 1;
    while run_end < ann_len {
        match alignment.base_idx_for(run_end) {
            Some(bidx) if content_idx_of_base_msg[bidx] == orig_cidx => run_end += 1,
            _ => break,
        }
    }
    run_end
}

fn gemini_run_is_unchanged(
    start: usize,
    run_end: usize,
    expected_len: usize,
    alignment: &GeminiGenerateContentMessageAlignment,
    ann_non_sys: &[&Json],
    base_non_sys: &[&Json],
) -> bool {
    let run_len = run_end - start;
    run_len == expected_len
        && (start..run_end).all(|j| {
            alignment
                .base_idx_for(j)
                .and_then(|bidx| base_non_sys.get(bidx))
                .map(|bm| *bm == ann_non_sys[j])
                .unwrap_or(false)
        })
}

fn patch_gemini_missing_content_run(
    run: &[&Json],
    call_id_to_name: &HashMap<String, String>,
) -> Result<Option<Json>> {
    if run.len() > 1 {
        return Err(FlowError::Internal(
            "Gemini encode: multiple messages map to a missing content item".into(),
        ));
    }
    match run.first() {
        Some(m) => normalized_to_gemini_content(m, call_id_to_name),
        None => Ok(None),
    }
}

fn patch_gemini_non_system_contents(
    obj: &mut serde_json::Map<String, Json>,
    messages: &[Message],
    ann_non_sys: &[&Json],
    base_non_sys: &[&Json],
) -> Result<()> {
    if ann_non_sys == base_non_sys {
        return Ok(());
    }
    let call_id_to_name = gemini_call_id_to_name(messages);
    let orig_contents: Vec<Json> = obj
        .get("contents")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let alignment = GeminiGenerateContentMessageAlignment::new(ann_non_sys, base_non_sys);
    let content_idx_of_base_msg =
        gemini_content_idx_of_base_msg(&orig_contents, base_non_sys.len())?;
    let content_msg_count =
        gemini_content_msg_counts(&content_idx_of_base_msg, orig_contents.len());

    let mut new_contents = Vec::new();
    let mut processed_cidxs = std::collections::HashSet::<usize>::new();
    let mut i = 0;
    while i < ann_non_sys.len() {
        let Some(base_idx) = alignment.base_idx_for(i) else {
            push_fresh_gemini_content(&mut new_contents, ann_non_sys[i], &call_id_to_name)?;
            i += 1;
            continue;
        };
        let orig_cidx = content_idx_of_base_msg[base_idx];
        if processed_cidxs.contains(&orig_cidx) {
            push_fresh_gemini_content(&mut new_contents, ann_non_sys[i], &call_id_to_name)?;
            i += 1;
            continue;
        }

        let run_end = gemini_run_end(
            i,
            ann_non_sys.len(),
            orig_cidx,
            &alignment,
            &content_idx_of_base_msg,
        );
        processed_cidxs.insert(orig_cidx);
        let expected_len = content_msg_count.get(orig_cidx).copied().unwrap_or(1);
        if gemini_run_is_unchanged(
            i,
            run_end,
            expected_len,
            &alignment,
            ann_non_sys,
            base_non_sys,
        ) {
            if let Some(orig) = orig_contents.get(orig_cidx) {
                new_contents.push(orig.clone());
            }
        } else {
            let run: Vec<&Json> = ann_non_sys[i..run_end].to_vec();
            let item = orig_contents
                .get(orig_cidx)
                .map(|orig| patch_changed_gemini_content(orig, &run, &call_id_to_name))
                .unwrap_or_else(|| patch_gemini_missing_content_run(&run, &call_id_to_name))?;
            if let Some(item) = item {
                new_contents.push(item);
            }
        }

        i = run_end;
    }
    obj.insert("contents".into(), Json::Array(new_contents));
    Ok(())
}

/// Rebuild a Gemini `contents` item for a position that changed from baseline.
///
/// Applies one or more normalized-message changes to an original Gemini content
/// item in a single pass over the original parts, preserving native metadata
/// (`thoughtSignature`, `thought`, `inlineData`, unmodeled fields). Surrounding
/// non-call parts keep their original positions; function-call slots follow the
/// annotated call order so reordered calls keep the right signatures.
///
/// `ann_msgs` is the slice of annotated messages that map to this content item.
/// For most content items it is a single message; for parallel `functionResponse`
/// items it may be multiple consecutive `Message::Tool` entries.
fn patch_changed_gemini_content(
    original_item: &Json,
    ann_msgs: &[&Json],
    call_id_to_name: &HashMap<String, String>,
) -> Result<Option<Json>> {
    let orig_obj = original_item
        .as_object()
        .ok_or_else(|| FlowError::Internal("original contents item is not an object".into()))?;
    // Missing role is treated as "user" on decode; mirror that here so valid roleless
    // content items are not silently dropped when edited.
    let orig_role = orig_obj
        .get("role")
        .and_then(Json::as_str)
        .unwrap_or("user");
    let orig_parts = orig_obj
        .get("parts")
        .and_then(|v| v.as_array())
        .ok_or_else(|| FlowError::Internal("original contents item has no parts array".into()))?;

    // For multi-message runs (parallel functionResponse), apply changes to one
    // representative message first, then chain the rest.
    let ann_msg = match ann_msgs.first() {
        Some(m) => m,
        None => return Ok(None),
    };
    let msg_obj = match ann_msg.as_object() {
        Some(o) => o,
        None => return Ok(None),
    };

    let ann_role = msg_obj
        .get("role")
        .and_then(Json::as_str)
        .unwrap_or(orig_role);
    // Apply the same role validation as normalized_to_gemini_content to catch
    // unsupported roles injected by interceptors on the edit path.
    let gemini_role = match ann_role {
        "assistant" => "model",
        "tool" => "user",
        "user" => "user",
        other => {
            return Err(FlowError::InvalidArgument(format!(
                "Gemini encoder: role '{other}' has no Gemini equivalent \
                 (only 'user' and 'assistant' are supported)"
            )));
        }
    };

    if ann_role == "tool" {
        return patch_gemini_tool_response_content(orig_parts, ann_msgs, call_id_to_name);
    }

    patch_gemini_visible_content(orig_parts, msg_obj, gemini_role)
}

fn gemini_function_response_updates<'a>(
    ann_msgs: &'a [&Json],
) -> Result<(
    HashMap<&'a str, GeminiGenerateContentFunctionResponsePayload>,
    Vec<&'a str>,
)> {
    let mut updates = HashMap::new();
    let mut update_order = Vec::new();
    for am in ann_msgs {
        let mo = am
            .as_object()
            .ok_or_else(|| FlowError::Internal("tool message is not an object".into()))?;
        let call_id = extract_tool_call_id(mo)?;
        let content_val = mo.get("content").unwrap_or(&Json::Null);
        let payload = function_response_payload_from_tool_content(content_val)?;
        if updates.insert(call_id, payload).is_some() {
            return Err(FlowError::InvalidArgument(format!(
                "Gemini encoder: duplicate Message::Tool tool_call_id '{call_id}'"
            )));
        }
        update_order.push(call_id);
    }
    Ok((updates, update_order))
}

fn function_response_call_id(fr: &Json) -> Option<&str> {
    fr.get("id")
        .and_then(|v| v.as_str())
        .or_else(|| fr.get("name").and_then(|v| v.as_str()))
}

fn build_gemini_function_response_part(
    call_id: &str,
    name: &str,
    payload: GeminiGenerateContentFunctionResponsePayload,
    original_fr: Option<&Json>,
) -> Json {
    let mut new_fr = original_fr
        .and_then(Json::as_object)
        .cloned()
        .unwrap_or_default();
    new_fr.insert("id".into(), Json::String(call_id.to_string()));
    new_fr.insert("name".into(), Json::String(name.to_string()));
    new_fr.insert("response".into(), payload.response);
    if let Some(parts) = payload.parts {
        new_fr.insert("parts".into(), Json::Array(parts));
    }
    serde_json::json!({"functionResponse": Json::Object(new_fr)})
}

fn patch_gemini_tool_response_content(
    orig_parts: &[Json],
    ann_msgs: &[&Json],
    call_id_to_name: &HashMap<String, String>,
) -> Result<Option<Json>> {
    let (mut updates, update_order) = gemini_function_response_updates(ann_msgs)?;
    let known_ids: std::collections::HashSet<&str> = updates.keys().copied().collect();
    let mut new_parts = Vec::new();

    for orig_part in orig_parts {
        let Some(fr) = orig_part.get("functionResponse") else {
            new_parts.push(orig_part.clone());
            continue;
        };
        let Some(call_id) = function_response_call_id(fr) else {
            new_parts.push(orig_part.clone());
            continue;
        };
        if !known_ids.contains(call_id) {
            continue;
        }
        if let Some(payload) = updates.remove(call_id) {
            let name = fr.get("name").and_then(Json::as_str).unwrap_or(call_id);
            new_parts.push(build_gemini_function_response_part(
                call_id,
                name,
                payload,
                Some(fr),
            ));
        } else {
            new_parts.push(orig_part.clone());
        }
    }

    for call_id in update_order {
        let Some(payload) = updates.remove(call_id) else {
            continue;
        };
        let fn_name = call_id_to_name
            .get(call_id)
            .map(String::as_str)
            .unwrap_or(call_id);
        new_parts.push(build_gemini_function_response_part(
            call_id, fn_name, payload, None,
        ));
    }

    Ok(Some(
        serde_json::json!({"role": "user", "parts": new_parts}),
    ))
}

fn original_function_call_entries(orig_parts: &[Json]) -> Vec<(usize, Option<&str>, &str)> {
    orig_parts
        .iter()
        .enumerate()
        .filter_map(|(i, p)| {
            let fc = p.get("functionCall")?;
            let name = fc.get("name")?.as_str()?;
            let id = fc.get("id").and_then(Json::as_str);
            Some((i, id, name))
        })
        .collect()
}

fn matching_function_call_entry(
    entries: &[(usize, Option<&str>, &str)],
    consumed: &std::collections::HashSet<usize>,
    fn_id: Option<&str>,
    fn_name: &str,
) -> Option<(usize, Option<String>)> {
    let matched = fn_id
        .and_then(|id| {
            entries
                .iter()
                .find(|(idx, orig_id, _)| !consumed.contains(idx) && *orig_id == Some(id))
        })
        .or_else(|| {
            entries.iter().find(|(idx, orig_id, name)| {
                !consumed.contains(idx) && orig_id.is_none() && *name == fn_name
            })
        })?;
    Some((matched.0, matched.1.map(str::to_string)))
}

fn rebuilt_gemini_function_call_parts(
    orig_parts: &[Json],
    msg_obj: &serde_json::Map<String, Json>,
) -> Result<Vec<Json>> {
    let entries = original_function_call_entries(orig_parts);
    let mut consumed = std::collections::HashSet::<usize>::new();
    let mut rebuilt = Vec::new();
    let Some(tool_calls) = msg_obj.get("tool_calls").and_then(Json::as_array) else {
        return Ok(rebuilt);
    };

    for tc in tool_calls {
        let mut fc_obj = tool_call_to_fc_obj(tc)?;
        let fn_name = fc_obj
            .get("name")
            .and_then(Json::as_str)
            .unwrap_or("")
            .to_string();
        let fn_id = fc_obj.get("id").and_then(Json::as_str).map(str::to_string);
        if let Some((orig_idx, orig_id)) =
            matching_function_call_entry(&entries, &consumed, fn_id.as_deref(), fn_name.as_str())
        {
            consumed.insert(orig_idx);
            if orig_id.is_none() && fn_id.as_deref() == Some(fn_name.as_str()) {
                fc_obj.remove("id");
            }
            let mut part_obj = orig_parts[orig_idx]
                .as_object()
                .cloned()
                .unwrap_or_default();
            part_obj.insert("functionCall".into(), Json::Object(fc_obj));
            rebuilt.push(Json::Object(part_obj));
        } else {
            rebuilt.push(serde_json::json!({"functionCall": Json::Object(fc_obj)}));
        }
    }
    Ok(rebuilt)
}

fn replacement_content_part(
    orig_part: &Json,
    replacement: Json,
    content_is_parts_form: bool,
) -> Json {
    if !content_is_parts_form
        && orig_part.get("text").is_some()
        && replacement.get("text").is_some()
    {
        let mut obj = orig_part.as_object().cloned().unwrap_or_default();
        obj.insert("text".into(), replacement.get("text").unwrap().clone());
        Json::Object(obj)
    } else {
        replacement
    }
}

fn merge_gemini_original_parts(
    orig_parts: &[Json],
    new_content_parts: Vec<Json>,
    rebuilt_fn_calls: Vec<Json>,
    content_is_parts_form: bool,
) -> Vec<Json> {
    let mut parts = Vec::new();
    let mut new_content_parts = new_content_parts.into_iter();
    let mut rebuilt_fn_calls = rebuilt_fn_calls.into_iter();
    let mut content_emitted = false;

    for orig_part in orig_parts {
        content_emitted |= merge_gemini_original_part(
            orig_part,
            &mut new_content_parts,
            &mut rebuilt_fn_calls,
            content_is_parts_form,
            &mut parts,
        );
    }

    let remaining_content_parts: Vec<Json> = new_content_parts.collect();
    if !content_emitted && !remaining_content_parts.is_empty() {
        for part in remaining_content_parts.into_iter().rev() {
            parts.insert(0, part);
        }
    } else {
        parts.extend(remaining_content_parts);
    }
    parts.extend(rebuilt_fn_calls);
    if parts.is_empty() {
        parts.push(serde_json::json!({"text": ""}));
    }
    parts
}

fn merge_gemini_original_part(
    orig_part: &Json,
    new_content_parts: &mut impl Iterator<Item = Json>,
    rebuilt_fn_calls: &mut impl Iterator<Item = Json>,
    content_is_parts_form: bool,
    parts: &mut Vec<Json>,
) -> bool {
    if orig_part.get("functionCall").is_some() {
        if let Some(rebuilt) = rebuilt_fn_calls.next() {
            parts.push(rebuilt);
        }
        return false;
    }
    let is_thought = orig_part.get("thought").and_then(Json::as_bool) == Some(true);
    let is_content_part = !is_thought && orig_part.get("functionResponse").is_none();
    if !is_content_part {
        parts.push(orig_part.clone());
        return false;
    }
    if let Some(replacement) = new_content_parts.next() {
        parts.push(replacement_content_part(
            orig_part,
            replacement,
            content_is_parts_form,
        ));
        return true;
    }
    if !content_is_parts_form
        && (orig_part.get("text").is_none()
            || (orig_part.get("text").is_some() && orig_part.get("thoughtSignature").is_some()))
    {
        parts.push(orig_part.clone());
    }
    false
}

fn decode_gemini_messages(obj: &serde_json::Map<String, Json>) -> Result<Vec<Message>> {
    let mut messages = Vec::new();
    if let Some(system) = obj.get("systemInstruction") {
        validate_system_instruction(system)?;
        if let Some(text) = system_instruction_text(system) {
            messages.push(
                serde_json::from_value(serde_json::json!({
                    "role": "system",
                    "content": text,
                }))
                .map_err(|e| {
                    FlowError::Internal(format!("Gemini system instruction decode: {e}"))
                })?,
            );
        }
    }
    let contents = obj
        .get("contents")
        .ok_or_else(|| FlowError::InvalidArgument("Gemini request is missing contents".into()))?;
    let contents = contents.as_array().ok_or_else(|| {
        FlowError::InvalidArgument("Gemini request contents must be an array".into())
    })?;
    for content in contents {
        messages.extend(gemini_content_to_messages(content)?);
    }
    Ok(messages)
}

fn decode_gemini_generation_params(
    obj: &serde_json::Map<String, Json>,
) -> Result<Option<GenerationParams>> {
    let config = match obj.get("generationConfig") {
        Some(value) if !value.is_object() => {
            return Err(FlowError::InvalidArgument(
                "Gemini generationConfig must be an object".into(),
            ));
        }
        value => value,
    };
    let temperature =
        decode_gemini_f64(config, "temperature", "Gemini temperature must be a number")?;
    let top_p = decode_gemini_f64(config, "topP", "Gemini topP must be a number")?;
    let max_tokens = config
        .and_then(|value| value.get("maxOutputTokens"))
        .map(|value| {
            value.as_u64().ok_or_else(|| {
                FlowError::InvalidArgument(
                    "Gemini maxOutputTokens must be a non-negative integer".into(),
                )
            })
        })
        .transpose()?;
    let stop = config
        .and_then(|value| value.get("stopSequences"))
        .map(|value| {
            serde_json::from_value::<Vec<String>>(value.clone()).map_err(|_| {
                FlowError::InvalidArgument(
                    "Gemini stopSequences must be an array of strings".into(),
                )
            })
        })
        .transpose()?;
    if temperature.is_none() && top_p.is_none() && max_tokens.is_none() && stop.is_none() {
        Ok(None)
    } else {
        Ok(Some(GenerationParams {
            temperature,
            max_tokens,
            top_p,
            stop,
        }))
    }
}

fn decode_gemini_f64(config: Option<&Json>, key: &str, error: &str) -> Result<Option<f64>> {
    config
        .and_then(|value| value.get(key))
        .map(|value| {
            value
                .as_f64()
                .ok_or_else(|| FlowError::InvalidArgument(error.into()))
        })
        .transpose()
}

fn decode_gemini_tools(obj: &serde_json::Map<String, Json>) -> Result<Option<Vec<ToolDefinition>>> {
    let Some(value) = obj.get("tools") else {
        return Ok(None);
    };
    let groups = value
        .as_array()
        .ok_or_else(|| FlowError::InvalidArgument("Gemini tools must be an array".into()))?;
    let mut definitions = Vec::new();
    for group in groups {
        decode_gemini_tool_group(group, &mut definitions)?;
    }
    Ok((!definitions.is_empty()).then_some(definitions))
}

fn decode_gemini_tool_group(group: &Json, definitions: &mut Vec<ToolDefinition>) -> Result<()> {
    let group_obj = group.as_object().ok_or_else(|| {
        FlowError::InvalidArgument("Gemini tools[] entry must be an object".into())
    })?;
    let has_declarations = group_obj.contains_key("functionDeclarations");
    if let Some(value) = group_obj.get("functionDeclarations") {
        let declarations = value.as_array().ok_or_else(|| {
            FlowError::InvalidArgument("Gemini functionDeclarations must be an array".into())
        })?;
        for declaration in declarations {
            definitions.push(decode_gemini_function_definition(declaration)?);
        }
    }
    let native_value = has_declarations
        .then(|| gemini_native_tool_fields(group_obj))
        .flatten()
        .or_else(|| (!has_declarations).then(|| group.clone()));
    if let Some(value) = native_value {
        definitions.push(ToolDefinition::ProviderNative {
            provider: GEMINI_PROVIDER.into(),
            kind: gemini_native_tool_kind(&value),
            value,
        });
    }
    Ok(())
}

fn decode_gemini_function_definition(fd: &Json) -> Result<ToolDefinition> {
    const MODELED_FUNCTION_DEFINITION_KEYS: &[&str] = &["name", "description", "parameters"];
    let fd_obj = fd.as_object().ok_or_else(|| {
        FlowError::InvalidArgument("Gemini functionDeclaration entry must be an object".into())
    })?;
    let name = fd_obj
        .get("name")
        .and_then(Json::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            FlowError::InvalidArgument(
                "Gemini functionDeclaration must have a non-empty 'name'".into(),
            )
        })?;
    let description = match fd_obj.get("description") {
        None => None,
        Some(Json::String(value)) => Some(value.clone()),
        Some(_) => {
            return Err(FlowError::InvalidArgument(
                "Gemini functionDeclaration.description must be a string".into(),
            ));
        }
    };
    let extra = fd_obj
        .iter()
        .filter(|(key, _)| !MODELED_FUNCTION_DEFINITION_KEYS.contains(&key.as_str()))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    Ok(ToolDefinition::Function {
        function: FunctionDefinition {
            name: name.to_string(),
            description,
            parameters: fd_obj.get("parameters").cloned(),
            strict: None,
            extra,
        },
        extra: Default::default(),
    })
}

fn decode_gemini_model(obj: &serde_json::Map<String, Json>) -> Result<Option<String>> {
    match obj.get("model") {
        None => Ok(None),
        Some(Json::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(FlowError::InvalidArgument(
            "Gemini request 'model' must be a string".into(),
        )),
    }
}

fn patch_gemini_visible_content(
    orig_parts: &[Json],
    msg_obj: &serde_json::Map<String, Json>,
    gemini_role: &str,
) -> Result<Option<Json>> {
    let content_val = msg_obj.get("content").unwrap_or(&Json::Null);
    let (new_content_parts, content_is_parts_form) =
        gemini_content_parts_from_normalized(content_val)?;
    let rebuilt_fn_calls = rebuilt_gemini_function_call_parts(orig_parts, msg_obj)?;
    let parts = merge_gemini_original_parts(
        orig_parts,
        new_content_parts,
        rebuilt_fn_calls,
        content_is_parts_form,
    );

    Ok(Some(
        serde_json::json!({"role": gemini_role, "parts": parts}),
    ))
}

/// Patch `generationConfig` with the modeled params, preserving unmodeled keys.
///
/// Modeled keys: `temperature`, `topP`, `maxOutputTokens`, `stopSequences`.
/// All other keys (e.g. `responseMimeType`, `responseSchema`, `thinkingConfig`)
/// are preserved regardless of whether `params` is `Some` or `None`.
const MODELED_GEN_CONFIG_KEYS: &[&str] =
    &["temperature", "topP", "maxOutputTokens", "stopSequences"];

fn patch_gemini_params(
    obj: &mut serde_json::Map<String, Json>,
    params: Option<&GenerationParams>,
) -> Result<()> {
    let Some(params) = params else {
        // Params cleared: remove only modeled keys; keep provider-native fields.
        if let Some(gc) = obj
            .get_mut("generationConfig")
            .and_then(|v| v.as_object_mut())
        {
            for key in MODELED_GEN_CONFIG_KEYS {
                gc.remove(*key);
            }
        }
        // If generationConfig is now empty (or was absent), drop the key entirely.
        if obj
            .get("generationConfig")
            .and_then(|v| v.as_object())
            .map(|m| m.is_empty())
            .unwrap_or(false)
        {
            obj.remove("generationConfig");
        }
        return Ok(());
    };

    let mut gen_config: serde_json::Map<String, Json> = obj
        .get("generationConfig")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();

    match params.temperature {
        Some(t) => {
            gen_config.insert("temperature".into(), json_f64(t, "temperature")?);
        }
        None => {
            gen_config.remove("temperature");
        }
    }
    match params.top_p {
        Some(p) => {
            gen_config.insert("topP".into(), json_f64(p, "topP")?);
        }
        None => {
            gen_config.remove("topP");
        }
    }
    match params.max_tokens {
        Some(n) => {
            gen_config.insert("maxOutputTokens".into(), Json::from(n));
        }
        None => {
            gen_config.remove("maxOutputTokens");
        }
    }
    match &params.stop {
        Some(stop) => {
            insert_serialized(&mut gen_config, "stopSequences", stop, "stopSequences")?;
        }
        None => {
            gen_config.remove("stopSequences");
        }
    }

    if gen_config.is_empty() {
        obj.remove("generationConfig");
    } else {
        obj.insert("generationConfig".into(), Json::Object(gen_config));
    }
    Ok(())
}

/// Patch the `functionDeclarations` group inside `tools`, preserving original
/// group order, group-level sibling fields, and all other tool groups
/// (googleSearch, codeExecution, …) in their original positions.
///
/// Provider-native fields captured in `FunctionDefinition.extra`
/// (parametersJsonSchema, responseJsonSchema, response, behavior, …) are merged
/// back into the encoded functionDeclaration, with modeled fields (name,
/// description, parameters) taking precedence.
fn patch_gemini_tools(
    obj: &mut serde_json::Map<String, Json>,
    tools: Option<&Vec<ToolDefinition>>,
) -> Result<()> {
    let fn_declarations = gemini_function_declarations(tools)?;
    let native_groups = gemini_native_tool_groups(tools)?;

    // Walk the original tools array in order: replace the FIRST functionDeclarations
    // group with the rebuilt list, merge any native sibling fields through the
    // normalized ProviderNative item, replace native-only groups from the normalized
    // list, and append any newly added native groups. If there was no original
    // functionDeclarations group, append the new one at the end.
    let orig_tools = obj
        .get("tools")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    // If the original request had multiple functionDeclarations groups, the decode path
    // already flattened them into a single normalized list so there is no way to know
    // which function belongs in which group after an edit.  Return an error rather than
    // silently collapsing all functions into the first group and losing the rest.
    let fn_decl_group_count = orig_tools
        .iter()
        .filter(|g| g.get("functionDeclarations").is_some())
        .count();
    if fn_decl_group_count > 1 {
        return Err(FlowError::InvalidArgument(format!(
            "Gemini encoder: the original request has {fn_decl_group_count} \
             functionDeclarations groups; editing tools with multiple groups is not \
             supported because the decode path flattens them and grouping cannot be \
             recovered. Use provider-native extra fields to manage multi-group tools."
        )));
    }
    let new_groups = rebuild_gemini_tool_groups(&orig_tools, &fn_declarations, &native_groups);

    if new_groups.is_empty() {
        obj.remove("tools");
    } else {
        obj.insert("tools".into(), Json::Array(new_groups));
    }
    Ok(())
}

fn gemini_function_declarations(tools: Option<&Vec<ToolDefinition>>) -> Result<Vec<Json>> {
    tools.into_iter().flatten().filter_map(|tool| match tool {
        ToolDefinition::Function { function, .. } => Some(gemini_function_declaration(function)),
        ToolDefinition::ProviderNative { provider, .. } if provider == GEMINI_PROVIDER => None,
        ToolDefinition::ProviderNative { provider, kind, .. } => Some(Err(FlowError::InvalidArgument(
            format!("Gemini encoder: ProviderNative tool '{kind}' (provider '{provider}') cannot be represented on the Gemini surface")
        ))),
    }).collect()
}

fn gemini_function_declaration(fd: &FunctionDefinition) -> Result<Json> {
    if fd.name.is_empty() {
        return Err(FlowError::InvalidArgument(
            "Gemini encoder: FunctionDefinition.name must be non-empty".into(),
        ));
    }
    if fd.strict.is_some() {
        return Err(FlowError::InvalidArgument(
            "Gemini encoder: FunctionDefinition.strict is not supported; remove it or use a provider-native extra field".into(),
        ));
    }
    let mut object = fd.extra.clone();
    object.insert("name".into(), Json::String(fd.name.clone()));
    if let Some(description) = &fd.description {
        object.insert("description".into(), Json::String(description.clone()));
    }
    if let Some(parameters) = &fd.parameters {
        object.insert("parameters".into(), parameters.clone());
    }
    Ok(Json::Object(object))
}

fn gemini_native_tool_groups(tools: Option<&Vec<ToolDefinition>>) -> Result<Vec<Json>> {
    tools
        .into_iter()
        .flatten()
        .filter_map(|tool| match tool {
            ToolDefinition::ProviderNative {
                provider, value, ..
            } if provider == GEMINI_PROVIDER => Some(validate_gemini_native_tool_group(value)),
            _ => None,
        })
        .collect()
}

fn validate_gemini_native_tool_group(value: &Json) -> Result<Json> {
    if !value.is_object() {
        return Err(FlowError::InvalidArgument(
            "Gemini encoder: ProviderNative tool value must be an object".into(),
        ));
    }
    if value.get("functionDeclarations").is_some() {
        return Err(FlowError::InvalidArgument(
            "Gemini encoder: ProviderNative tool value must not contain functionDeclarations; use ToolDefinition::Function instead".into(),
        ));
    }
    Ok(value.clone())
}

fn rebuild_gemini_tool_groups(
    orig_tools: &[Json],
    fn_declarations: &[Json],
    native_groups: &[Json],
) -> Vec<Json> {
    let mut new_groups = Vec::with_capacity(orig_tools.len());
    let mut fn_group_placed = false;
    let mut native_used = vec![false; native_groups.len()];
    for orig_group in orig_tools {
        let (placed, group) = rebuild_gemini_tool_group(
            orig_group,
            fn_declarations,
            native_groups,
            &mut native_used,
            fn_group_placed,
        );
        fn_group_placed |= placed;
        if let Some(group) = group {
            new_groups.push(group);
        }
    }
    if !fn_group_placed && !fn_declarations.is_empty() {
        new_groups.push(serde_json::json!({"functionDeclarations": fn_declarations}));
    }
    new_groups.extend(
        native_groups
            .iter()
            .enumerate()
            .filter(|(index, _)| !native_used[*index])
            .map(|(_, group)| group.clone()),
    );
    new_groups
}

fn rebuild_gemini_tool_group(
    orig_group: &Json,
    fn_declarations: &[Json],
    native_groups: &[Json],
    native_used: &mut [bool],
    fn_group_placed: bool,
) -> (bool, Option<Json>) {
    if orig_group.get("functionDeclarations").is_some() {
        if fn_group_placed {
            return (false, None);
        }
        let mut group = serde_json::Map::new();
        if !fn_declarations.is_empty() {
            group.insert(
                "functionDeclarations".into(),
                Json::Array(fn_declarations.to_vec()),
            );
        }
        merge_gemini_native_sibling_group(orig_group, native_groups, native_used, &mut group);
        return (true, (!group.is_empty()).then_some(Json::Object(group)));
    }
    let group = take_matching_native_group(
        native_groups,
        native_used,
        &gemini_native_tool_keys(orig_group),
    );
    (false, group)
}

fn merge_gemini_native_sibling_group(
    orig_group: &Json,
    native_groups: &[Json],
    native_used: &mut [bool],
    group: &mut serde_json::Map<String, Json>,
) {
    let keys = gemini_native_tool_keys(orig_group);
    if let Some(native_group) = take_matching_native_group(native_groups, native_used, &keys)
        && let Some(native_obj) = native_group.as_object()
    {
        group.extend(native_obj.clone());
    }
}

/// Overlay extra-field changes from `annotated` onto `obj`, guided by `baseline`.
fn patch_extra_fields(
    obj: &mut serde_json::Map<String, Json>,
    baseline: &serde_json::Map<String, Json>,
    annotated: &serde_json::Map<String, Json>,
) {
    for key in baseline.keys().filter(|k| !annotated.contains_key(*k)) {
        obj.remove(key);
    }
    for (key, value) in annotated {
        if baseline.get(key) != Some(value) {
            obj.insert(key.clone(), value.clone());
        }
    }
}

// ---------------------------------------------------------------------------
// Streaming codec
// ---------------------------------------------------------------------------

/// Streaming counterpart to [`GeminiGenerateContentCodec`].
///
/// Accumulates Gemini server-sent event chunks and assembles a complete response
/// that [`GeminiGenerateContentCodec::decode_response`] can consume.
pub struct GeminiGenerateContentStreamingCodec {
    state: std::sync::Arc<std::sync::Mutex<GeminiGenerateContentStreamingState>>,
}

impl GeminiGenerateContentStreamingCodec {
    /// Creates a fresh streaming codec with empty accumulator state.
    pub fn new() -> Self {
        Self {
            state: std::sync::Arc::new(std::sync::Mutex::new(
                GeminiGenerateContentStreamingState::default(),
            )),
        }
    }
}

impl Default for GeminiGenerateContentStreamingCodec {
    fn default() -> Self {
        Self::new()
    }
}

impl super::streaming::StreamingCodec for GeminiGenerateContentStreamingCodec {
    fn collector(&self) -> crate::api::runtime::LlmCollectorFn {
        let state = std::sync::Arc::clone(&self.state);
        Box::new(move |event: Json| -> Result<()> {
            let mut guard = state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            guard.observe(&event)?;
            Ok(())
        })
    }

    fn finalizer(&self) -> crate::api::runtime::LlmFinalizerFn {
        let state = std::sync::Arc::clone(&self.state);
        Box::new(move || -> Json {
            let mut guard = state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            std::mem::take(&mut *guard).finalize()
        })
    }
}

#[derive(Debug, Default)]
struct GeminiGenerateContentStreamingState {
    parts: Vec<Json>,
    candidate_index: Option<u64>,
    finish_reason: Option<String>,
    usage_metadata: Option<Json>,
    model_version: Option<String>,
    response_id: Option<String>,
    /// Merged candidate-level extra fields (safetyRatings, groundingMetadata,
    /// citationMetadata, avgLogprobs, etc.) accumulated across SSE chunks.
    candidate_extra: serde_json::Map<String, Json>,
}

impl GeminiGenerateContentStreamingState {
    fn push_text_part(&mut self, text: &str, part_obj: &serde_json::Map<String, Json>) {
        let mut extra: serde_json::Map<String, Json> = part_obj
            .iter()
            .filter(|(k, _)| k.as_str() != "text")
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        if let Some(last_obj) = self.parts.last_mut().and_then(Json::as_object_mut) {
            let current_has_metadata = !extra.is_empty();
            let previous_has_metadata = last_obj.keys().any(|k| k != "text");
            let can_merge = last_obj.get("text").is_some_and(Json::is_string)
                && !current_has_metadata
                && !previous_has_metadata;
            if can_merge {
                let Some(Json::String(existing)) = last_obj.get_mut("text") else {
                    return;
                };
                existing.push_str(text);
                for (k, v) in extra {
                    last_obj.insert(k, v);
                }
                return;
            }
        }

        extra.insert("text".into(), Json::String(text.to_string()));
        self.parts.push(Json::Object(extra));
    }

    fn observe(&mut self, event: &Json) -> Result<()> {
        if let Some(candidates) = event.get("candidates").and_then(Json::as_array) {
            self.observe_candidate(candidates)?;
        }
        if let Some(usage) = event.get("usageMetadata") {
            self.usage_metadata = Some(usage.clone());
        }
        if let Some(mv_val) = event.get("modelVersion") {
            match mv_val.as_str() {
                Some(s) => self.model_version = Some(s.to_string()),
                None => {
                    return Err(FlowError::InvalidArgument(
                        "Gemini streaming event modelVersion must be a string".into(),
                    ));
                }
            }
        }
        if let Some(rid_val) = event.get("responseId") {
            match rid_val.as_str() {
                Some(s) => self.response_id = Some(s.to_string()),
                None => {
                    return Err(FlowError::InvalidArgument(
                        "Gemini streaming event responseId must be a string".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    fn observe_candidate(&mut self, candidates: &[Json]) -> Result<()> {
        let Some(candidate) = candidates.first() else {
            return Ok(());
        };
        if candidates.len() > 1 {
            return Err(FlowError::InvalidArgument(
                "Gemini streaming chunks with multiple candidates are not supported".into(),
            ));
        }
        let candidate_obj = candidate.as_object().ok_or_else(|| {
            FlowError::InvalidArgument("Gemini streaming candidate must be an object".into())
        })?;
        self.observe_candidate_index(candidate_obj)?;
        if let Some(parts) = candidate_obj
            .get("content")
            .and_then(|content| content.get("parts"))
            .and_then(Json::as_array)
        {
            self.observe_parts(parts)?;
        }
        self.observe_finish_reason(candidate)?;
        for (key, value) in candidate_obj {
            if !matches!(key.as_str(), "content" | "finishReason" | "index") {
                self.candidate_extra.insert(key.clone(), value.clone());
            }
        }
        Ok(())
    }

    fn observe_candidate_index(&mut self, candidate: &serde_json::Map<String, Json>) -> Result<()> {
        let index = candidate
            .get("index")
            .ok_or_else(|| {
                FlowError::InvalidArgument("Gemini streaming candidate index is required".into())
            })?
            .as_u64()
            .ok_or_else(|| {
                FlowError::InvalidArgument(
                    "Gemini streaming candidate index must be an unsigned integer".into(),
                )
            })?;
        match self.candidate_index {
            Some(previous) if previous != index => Err(FlowError::InvalidArgument(
                "Gemini streaming candidate index changed across chunks".into(),
            )),
            Some(_) => Ok(()),
            None if index == 0 => {
                self.candidate_index = Some(index);
                Ok(())
            }
            None => Err(FlowError::InvalidArgument(
                "Gemini streaming only supports candidate index 0".into(),
            )),
        }
    }

    fn observe_parts(&mut self, parts: &[Json]) -> Result<()> {
        for part in parts {
            let part_obj = part.as_object().ok_or_else(|| {
                FlowError::InvalidArgument("Gemini streaming parts entry must be an object".into())
            })?;
            let data_key = validate_single_gemini_part_data_field(part_obj, "streaming")?;
            if part.get("thought").and_then(Json::as_bool) == Some(true) {
                self.parts.push(part.clone());
            } else {
                self.observe_part(part, part_obj, data_key)?;
            }
        }
        Ok(())
    }

    fn observe_part(
        &mut self,
        part: &Json,
        part_obj: &serde_json::Map<String, Json>,
        data_key: Option<&str>,
    ) -> Result<()> {
        match data_key {
            Some("text") => {
                let text = part.get("text").and_then(Json::as_str).ok_or_else(|| {
                    FlowError::InvalidArgument(
                        "Gemini streaming parts[].text must be a string".into(),
                    )
                })?;
                self.push_text_part(text, part_obj);
            }
            Some("functionResponse") => {
                return Err(FlowError::InvalidArgument(
                    "Gemini streaming response parts must not contain functionResponse".into(),
                ));
            }
            _ => self.parts.push(part.clone()),
        }
        Ok(())
    }

    fn observe_finish_reason(&mut self, candidate: &Json) -> Result<()> {
        if let Some(reason) = candidate.get("finishReason") {
            self.finish_reason = Some(
                reason
                    .as_str()
                    .ok_or_else(|| {
                        FlowError::InvalidArgument(
                            "Gemini streaming candidate finishReason must be a string".into(),
                        )
                    })?
                    .to_string(),
            );
        }
        Ok(())
    }

    fn finalize(self) -> Json {
        let mut candidate_obj = serde_json::Map::new();
        candidate_obj.insert(
            "content".into(),
            serde_json::json!({"role": "model", "parts": self.parts}),
        );
        if let Some(reason) = self.finish_reason {
            candidate_obj.insert("finishReason".into(), Json::String(reason));
        }
        candidate_obj.insert(
            "index".into(),
            Json::from(self.candidate_index.unwrap_or(0)),
        );
        // Merge accumulated candidate-level metadata so that decode_response can
        // populate ApiSpecificResponse::GeminiGenerateContent with the same fields as non-streaming.
        for (k, v) in self.candidate_extra {
            candidate_obj.entry(k).or_insert(v);
        }
        let candidate = Json::Object(candidate_obj);

        let mut output = serde_json::Map::new();
        output.insert("candidates".to_string(), Json::Array(vec![candidate]));
        if let Some(usage) = self.usage_metadata {
            output.insert("usageMetadata".to_string(), usage);
        }
        if let Some(mv) = self.model_version {
            output.insert("modelVersion".to_string(), Json::String(mv));
        }
        if let Some(rid) = self.response_id {
            output.insert("responseId".to_string(), Json::String(rid));
        }
        Json::Object(output)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "../../tests/unit/codec/gemini_generate_content_tests.rs"]
mod tests;
