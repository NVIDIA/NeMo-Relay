// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Built-in codec for the Oracle Cloud Infrastructure (OCI) Generative AI chat API.
//!
//! Implements [`LlmResponseCodec`] (response decode) for the OCI Generative AI
//! chat format.
//!
//! # OCI-specific patterns handled
//!
//! - **Three API formats** selected by the response `apiFormat`: `GENERIC`
//!   (`choices`-based, OpenAI-style), `COHERE` (`text`-based), and `COHEREV2`
//!   (single `message` with typed content parts and nested `function` tool
//!   calls, per the OCI `CohereChatResponseV2` schema).
//! - **Responses**: `ChatResult` payloads (`modelId`, `chatResponse`); `usage`
//!   counters are `promptTokens`/`completionTokens`/`totalTokens`.
//! - **Unmodeled fields are preserved**: envelope and chat-response fields the
//!   normalized shape does not model (`timeCreated`, future provider fields)
//!   are carried in `extra` rather than discarded, consistent with the other
//!   response codecs. Unmodeled fields of the decoded choice and assistant
//!   message — `logprobs`, `serviceTier`, `groundingMetadata`,
//!   `reasoningContent`, `refusal` (GENERIC) and `toolPlan`, `citations`
//!   (COHEREV2) per the OCI schema — are namespaced in `extra` under
//!   `"choice"` and `"message"`.
//!
//! The codec accepts the REST wire format only: camelCase keys, as documented
//! in the OCI API reference. Alternate renderings produced by Oracle tooling
//! (the CLI's kebab-case `data` envelope, `oci.util.to_dict()` snake_case
//! dicts) are the caller's responsibility to convert.

use crate::error::{FlowError, Result};
use crate::json::Json;

use super::request::{ContentPart, MessageContent, ProviderNativeComponent};
use super::response::{
    AnnotatedLlmResponse, ApiSpecificResponse, FinishReason, ResponseToolCall, Usage,
};
use super::traits::LlmResponseCodec;

// ---------------------------------------------------------------------------
// Public codec struct
// ---------------------------------------------------------------------------

/// Built-in codec for the OCI Generative AI chat API.
pub struct OCIGenAIChatCodec;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Map an OCI finish reason string to normalized [`FinishReason`].
///
/// GENERIC responses use OpenAI-style lowercase reasons (Gemini models emit
/// `max_tokens` for the length stop); COHERE and COHEREV2 responses use
/// UPPERCASE Cohere reasons (`TOOL_CALL` and `STOP_SEQUENCE` are V2-only).
fn map_oci_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "stop" | "COMPLETE" | "STOP_SEQUENCE" => FinishReason::Complete,
        "length" | "max_tokens" | "MAX_TOKENS" => FinishReason::Length,
        "tool_calls" | "TOOL_CALL" => FinishReason::ToolUse,
        "content_filter" => FinishReason::ContentFilter,
        other => FinishReason::Unknown(other.to_string()),
    }
}

/// Collect the fields of `obj` that are not in `modeled` for `extra` carriage.
fn unmodeled_fields(
    obj: &serde_json::Map<String, Json>,
    modeled: &[&str],
) -> serde_json::Map<String, Json> {
    obj.iter()
        .filter(|(key, _)| !modeled.contains(&key.as_str()))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

fn native_component(value: &Json) -> ProviderNativeComponent {
    ProviderNativeComponent {
        provider: "oci_genai".to_string(),
        kind: value
            .get("type")
            .and_then(Json::as_str)
            .unwrap_or("unknown")
            .to_string(),
        value: value.clone(),
    }
}

// ---------------------------------------------------------------------------
// GENERIC content conversion
// ---------------------------------------------------------------------------

/// Flatten a GENERIC content value into normalized [`MessageContent`].
///
/// A content-part list whose parts are all `{"type": "TEXT", "text": ...}` is
/// flattened to plain text; lists carrying any non-text part are preserved as
/// typed parts so image or future block types survive losslessly.
fn decode_generic_content(value: Option<&Json>) -> Result<Option<MessageContent>> {
    let value = match value {
        None | Some(Json::Null) => return Ok(None),
        Some(value) => value,
    };
    if let Some(text) = value.as_str() {
        return Ok(Some(MessageContent::Text(text.to_string())));
    }
    let parts = value.as_array().ok_or_else(|| {
        FlowError::InvalidArgument(
            "OCI GenAI GENERIC message content must be a string, an array, or null".into(),
        )
    })?;
    if parts.is_empty() {
        // Tool-call-only messages carry `"content": []`; there is no content.
        return Ok(None);
    }
    if let Some(text) = flatten_all_text_parts(parts) {
        return Ok(Some(MessageContent::Text(text)));
    }
    let parts = parts
        .iter()
        .map(decode_generic_content_part)
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(MessageContent::Parts(parts)))
}

/// Join a part list into plain text when every part is a `TEXT` part.
fn flatten_all_text_parts(parts: &[Json]) -> Option<String> {
    let mut text = String::new();
    for part in parts {
        let obj = part.as_object()?;
        if obj.get("type").and_then(Json::as_str) != Some("TEXT") {
            return None;
        }
        match obj.get("text") {
            None | Some(Json::Null) => {}
            Some(Json::String(part_text)) => text.push_str(part_text),
            Some(_) => return None,
        }
    }
    Some(text)
}

fn decode_generic_content_part(value: &Json) -> Result<ContentPart> {
    let Some(obj) = value.as_object() else {
        return Err(FlowError::InvalidArgument(
            "OCI GenAI GENERIC content part must be an object".into(),
        ));
    };
    match obj.get("type").and_then(Json::as_str) {
        Some("TEXT") => Ok(ContentPart::Text {
            text: obj
                .get("text")
                .and_then(Json::as_str)
                .unwrap_or_default()
                .to_string(),
            extra: obj
                .iter()
                .filter(|(key, _)| !matches!(key.as_str(), "type" | "text"))
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        }),
        _ => {
            let native = native_component(value);
            Ok(ContentPart::ProviderNative {
                provider: native.provider,
                kind: native.kind,
                value: native.value,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// LlmResponseCodec implementation
// ---------------------------------------------------------------------------

impl LlmResponseCodec for OCIGenAIChatCodec {
    fn decode_response(&self, response: &Json) -> Result<AnnotatedLlmResponse> {
        let Some(obj) = response.as_object() else {
            // Non-object responses are preserved raw so observability still
            // captures whatever the provider path produced.
            let mut extra = serde_json::Map::new();
            extra.insert("raw".to_string(), response.clone());
            return Ok(AnnotatedLlmResponse {
                extra,
                ..AnnotatedLlmResponse::default()
            });
        };

        let (envelope, chat_response) = match obj.get("chatResponse").and_then(Json::as_object) {
            Some(chat_response) => (Some(obj), chat_response),
            None => (None, obj),
        };

        let model = envelope
            .and_then(|envelope| envelope.get("modelId"))
            .and_then(Json::as_str)
            .map(str::to_string);
        let model_version = envelope
            .and_then(|envelope| envelope.get("modelVersion"))
            .and_then(Json::as_str)
            .map(str::to_string);
        let api_format = chat_response
            .get("apiFormat")
            .and_then(Json::as_str)
            .unwrap_or("GENERIC")
            .to_uppercase();

        let (message, tool_calls, finish_reason, nested_extra) = match api_format.as_str() {
            "COHERE" => decode_cohere_response_body(chat_response),
            "COHEREV2" => decode_cohere_v2_response_body(chat_response)?,
            _ => decode_generic_response_body(chat_response)?,
        };

        let id = if api_format == "COHEREV2" {
            chat_response
                .get("id")
                .and_then(Json::as_str)
                .map(str::to_string)
        } else {
            None
        };

        let usage = chat_response
            .get("usage")
            .and_then(Json::as_object)
            .map(decode_oci_usage);

        // Preserve fields the normalized shape does not model so observability
        // keeps timeCreated, service tiers, grounding metadata, and future
        // provider fields.
        let modeled_response_keys: &[&str] = match api_format.as_str() {
            "COHERE" => &["apiFormat", "text", "finishReason", "toolCalls", "usage"],
            "COHEREV2" => &["apiFormat", "id", "message", "finishReason", "usage"],
            _ => &["apiFormat", "choices", "usage"],
        };
        let mut extra = match envelope {
            Some(envelope) => {
                unmodeled_fields(envelope, &["chatResponse", "modelId", "modelVersion"])
            }
            None => serde_json::Map::new(),
        };
        extra.extend(unmodeled_fields(chat_response, modeled_response_keys));
        extra.extend(nested_extra);

        Ok(AnnotatedLlmResponse {
            id,
            model,
            message,
            tool_calls,
            finish_reason: finish_reason.as_deref().map(map_oci_finish_reason),
            usage,
            optimization_summary: None,
            api_specific: Some(ApiSpecificResponse::OCIGenAI {
                api_format: Some(api_format),
                model_version,
            }),
            extra,
        })
    }
}

type ResponseBody = (
    Option<MessageContent>,
    Option<Vec<ResponseToolCall>>,
    Option<String>,
    serde_json::Map<String, Json>,
);

/// Keys of the decoded GENERIC choice consumed by the normalized shape.
///
/// `index` is excluded from `extra` carriage as well: it is positional trivia
/// (always `0` for the single decoded choice) rather than provider data.
const MODELED_CHOICE_KEYS: &[&str] = &["message", "finishReason", "index"];

/// Keys of a decoded assistant message consumed by the normalized shape.
const MODELED_MESSAGE_KEYS: &[&str] = &["role", "content", "toolCalls"];

/// Namespace unmodeled fields of a decoded nested container under `key`.
///
/// The choice-level fields of GENERIC responses (`logprobs`, `usage`,
/// `groundingMetadata`, `serviceTier`) and the message-level fields of
/// GENERIC (`refusal`, `annotations`, `reasoningContent`) and COHEREV2
/// (`toolPlan`, `citations`) responses are documented in the OCI schema but
/// not normalized; they are carried in `extra` under the container's wire key
/// so their origin stays unambiguous.
fn nest_unmodeled_fields(
    extra: &mut serde_json::Map<String, Json>,
    key: &str,
    obj: &serde_json::Map<String, Json>,
    modeled: &[&str],
) {
    let unmodeled = unmodeled_fields(obj, modeled);
    if !unmodeled.is_empty() {
        extra.insert(key.to_string(), Json::Object(unmodeled));
    }
}

fn decode_generic_response_body(
    chat_response: &serde_json::Map<String, Json>,
) -> Result<ResponseBody> {
    let mut nested_extra = serde_json::Map::new();
    let Some(first_choice) = chat_response
        .get("choices")
        .and_then(Json::as_array)
        .and_then(|choices| choices.first())
        .and_then(Json::as_object)
    else {
        return Ok((None, None, None, nested_extra));
    };
    nest_unmodeled_fields(
        &mut nested_extra,
        "choice",
        first_choice,
        MODELED_CHOICE_KEYS,
    );
    let finish_reason = first_choice
        .get("finishReason")
        .and_then(Json::as_str)
        .map(str::to_string);
    let Some(raw_message) = first_choice.get("message").and_then(Json::as_object) else {
        return Ok((None, None, finish_reason, nested_extra));
    };
    nest_unmodeled_fields(
        &mut nested_extra,
        "message",
        raw_message,
        MODELED_MESSAGE_KEYS,
    );
    let message = decode_generic_content(raw_message.get("content"))?;
    let tool_calls = raw_message
        .get("toolCalls")
        .and_then(Json::as_array)
        .map(|calls| decode_response_tool_calls(calls))
        .filter(|calls: &Vec<ResponseToolCall>| !calls.is_empty());
    Ok((message, tool_calls, finish_reason, nested_extra))
}

fn decode_cohere_response_body(chat_response: &serde_json::Map<String, Json>) -> ResponseBody {
    let message = chat_response
        .get("text")
        .and_then(Json::as_str)
        .map(|text| MessageContent::Text(text.to_string()));
    let tool_calls = chat_response
        .get("toolCalls")
        .and_then(Json::as_array)
        .map(|calls| decode_response_tool_calls(calls))
        .filter(|calls| !calls.is_empty());
    let finish_reason = chat_response
        .get("finishReason")
        .and_then(Json::as_str)
        .map(str::to_string);
    // COHERE (v1) is flat: unmodeled fields live directly on the chat
    // response and are already carried by the chat-response-level pass.
    (message, tool_calls, finish_reason, serde_json::Map::new())
}

/// Decode a COHEREV2 (`CohereChatResponseV2`) body: a single assistant
/// `message` whose `content` is a typed part list (`TEXT`, `THINKING`,
/// `IMAGE_URL`, `DOCUMENT`) and whose tool calls nest an OpenAI-style
/// `function` object.
fn decode_cohere_v2_response_body(
    chat_response: &serde_json::Map<String, Json>,
) -> Result<ResponseBody> {
    let mut nested_extra = serde_json::Map::new();
    let finish_reason = chat_response
        .get("finishReason")
        .and_then(Json::as_str)
        .map(str::to_string);
    let Some(raw_message) = chat_response.get("message").and_then(Json::as_object) else {
        return Ok((None, None, finish_reason, nested_extra));
    };
    nest_unmodeled_fields(
        &mut nested_extra,
        "message",
        raw_message,
        MODELED_MESSAGE_KEYS,
    );
    let message = decode_generic_content(raw_message.get("content"))?;
    let tool_calls = raw_message
        .get("toolCalls")
        .and_then(Json::as_array)
        .map(|calls| decode_response_tool_calls(calls))
        .filter(|calls: &Vec<ResponseToolCall>| !calls.is_empty());
    Ok((message, tool_calls, finish_reason, nested_extra))
}

/// Convert an OCI response tool-call list into [`ResponseToolCall`]s.
fn decode_response_tool_calls(calls: &[Json]) -> Vec<ResponseToolCall> {
    calls
        .iter()
        .enumerate()
        .filter_map(|(index, call)| decode_response_tool_call(index, call))
        .collect()
}

/// Convert an OCI response tool call into [`ResponseToolCall`].
///
/// GENERIC calls are flat (`{id, type, name, arguments}`) with `arguments` as a
/// JSON-encoded string; COHERE calls carry `name` plus parsed `parameters` and
/// no `id`, so a positional `call_{index}` id is synthesized to keep parallel
/// calls distinguishable; COHEREV2 calls nest `name`/`arguments` under an
/// OpenAI-style `function` object next to the `id`.
fn decode_response_tool_call(index: usize, value: &Json) -> Option<ResponseToolCall> {
    let obj = value.as_object()?;
    let body = obj.get("function").and_then(Json::as_object).unwrap_or(obj);
    let name = body.get("name")?.as_str()?.to_string();
    let arguments = match body.get("arguments") {
        Some(Json::String(text)) => {
            // CRITICAL: GENERIC arguments arrive JSON-encoded; parse for the
            // normalized shape, preserving the raw string when unparseable.
            serde_json::from_str::<Json>(text).unwrap_or_else(|_| Json::String(text.clone()))
        }
        Some(other) => other.clone(),
        None => body.get("parameters").cloned().unwrap_or(Json::Null),
    };
    let id = match obj.get("id").and_then(Json::as_str) {
        Some(id) => id.to_string(),
        None => format!("call_{index}"),
    };
    Some(ResponseToolCall {
        id,
        name,
        arguments,
    })
}

/// Map OCI usage counters onto the normalized [`Usage`] field names.
///
/// OpenAI and xAI models report cache hits under
/// `promptTokensDetails.cachedTokens`.
fn decode_oci_usage(usage: &serde_json::Map<String, Json>) -> Usage {
    let cache_read_tokens = usage
        .get("promptTokensDetails")
        .and_then(Json::as_object)
        .and_then(|details| details.get("cachedTokens"))
        .and_then(Json::as_u64);
    Usage {
        prompt_tokens: usage.get("promptTokens").and_then(Json::as_u64),
        completion_tokens: usage.get("completionTokens").and_then(Json::as_u64),
        total_tokens: usage.get("totalTokens").and_then(Json::as_u64),
        cache_read_tokens,
        cache_write_tokens: None,
        cost: None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "../../tests/unit/codec/oci_genai_tests.rs"]
mod tests;
