// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Built-in codec for the Oracle Cloud Infrastructure (OCI) Generative AI chat API.
//!
//! Implements [`LlmResponseCodec`] (response decode) for the OCI Generative AI
//! chat format.
//!
//! # OCI-specific patterns handled
//!
//! - **Two API formats** selected by the response `apiFormat`: `GENERIC`
//!   (`choices`-based, OpenAI-style) and `COHERE` (`text`-based).
//! - **Responses**: `ChatResult` payloads (`modelId`, `chatResponse`) where the
//!   chat response is `choices`-based for `GENERIC` and `text`-based for
//!   `COHERE`; `usage` counters are `promptTokens`/`completionTokens`/`totalTokens`.
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
/// `max_tokens` for the length stop); COHERE responses use UPPERCASE Cohere
/// reasons.
fn map_oci_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "stop" | "COMPLETE" => FinishReason::Complete,
        "length" | "max_tokens" | "MAX_TOKENS" => FinishReason::Length,
        "tool_calls" => FinishReason::ToolUse,
        other => FinishReason::Unknown(other.to_string()),
    }
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

        let (message, tool_calls, finish_reason) = if api_format == "COHERE" {
            decode_cohere_response_body(chat_response)
        } else {
            decode_generic_response_body(chat_response)?
        };

        let usage = chat_response
            .get("usage")
            .and_then(Json::as_object)
            .map(decode_oci_usage);

        Ok(AnnotatedLlmResponse {
            id: None,
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
            extra: serde_json::Map::new(),
        })
    }
}

type ResponseBody = (
    Option<MessageContent>,
    Option<Vec<ResponseToolCall>>,
    Option<String>,
);

fn decode_generic_response_body(
    chat_response: &serde_json::Map<String, Json>,
) -> Result<ResponseBody> {
    let Some(first_choice) = chat_response
        .get("choices")
        .and_then(Json::as_array)
        .and_then(|choices| choices.first())
        .and_then(Json::as_object)
    else {
        return Ok((None, None, None));
    };
    let finish_reason = first_choice
        .get("finishReason")
        .and_then(Json::as_str)
        .map(str::to_string);
    let Some(raw_message) = first_choice.get("message").and_then(Json::as_object) else {
        return Ok((None, None, finish_reason));
    };
    let message = decode_generic_content(raw_message.get("content"))?;
    let tool_calls = raw_message
        .get("toolCalls")
        .and_then(Json::as_array)
        .map(|calls| decode_response_tool_calls(calls))
        .filter(|calls: &Vec<ResponseToolCall>| !calls.is_empty());
    Ok((message, tool_calls, finish_reason))
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
    (message, tool_calls, finish_reason)
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
/// calls distinguishable.
fn decode_response_tool_call(index: usize, value: &Json) -> Option<ResponseToolCall> {
    let obj = value.as_object()?;
    let name = obj.get("name")?.as_str()?.to_string();
    let arguments = match obj.get("arguments") {
        Some(Json::String(text)) => {
            // CRITICAL: GENERIC arguments arrive JSON-encoded; parse for the
            // normalized shape, preserving the raw string when unparseable.
            serde_json::from_str::<Json>(text).unwrap_or_else(|_| Json::String(text.clone()))
        }
        Some(other) => other.clone(),
        None => obj.get("parameters").cloned().unwrap_or(Json::Null),
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
