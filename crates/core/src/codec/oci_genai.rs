// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Built-in codec for the Oracle Cloud Infrastructure (OCI) Generative AI chat API.
//!
//! Implements [`LlmCodec`] (request decode/encode) and [`LlmResponseCodec`]
//! (response decode) for the OCI Generative AI chat format.
//!
//! # OCI-specific patterns handled
//!
//! - **ChatDetails envelope**: Requests may arrive as a full envelope
//!   (`compartmentId`, `servingMode`, `chatRequest`) or as a bare `chatRequest`
//!   payload; both are accepted and the envelope is preserved on encode.
//! - **API formats** selected by `apiFormat`:
//!   - `GENERIC`: OpenAI-style `messages` with UPPERCASE roles
//!     (`USER`/`ASSISTANT`/`SYSTEM`/`TOOL`) whose `content` is a list of typed
//!     parts (`{"type": "TEXT", "text": ...}`), flat `toolCalls`
//!     (`{id, type: "FUNCTION", name, arguments}`), and `toolCallId` on tool
//!     messages. Used by Meta Llama, Google, xAI, OpenAI, and imported
//!     open-weights models hosted on dedicated AI clusters.
//!   - `COHERE`: a single `message` string plus `chatHistory` turns with
//!     `USER`/`CHATBOT`/`SYSTEM` roles and an optional `preambleOverride`.
//!     Used by Cohere Command models.
//!   - `COHEREV2`: responses are a single assistant `message` with typed
//!     content parts and nested `function` tool calls, per the OCI
//!     `CohereChatResponseV2` schema. Requests follow the GENERIC `messages`
//!     path with COHERE-style `stopSequences`; V2-only request fields the
//!     normalized shape does not model (`citationOptions`, `documents`, ...)
//!     ride along in `extra` and survive edits untouched.
//! - **Model identity**: Carried in `servingMode.modelId` (on-demand) or
//!   `servingMode.endpointId` (dedicated), not in the chat request body.
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

use crate::api::llm::LlmRequest;
use crate::api::runtime::{BuiltinLlmCodec, LlmCodecIdentity};
use crate::error::{FlowError, Result};
use crate::json::Json;

use super::request::{
    AnnotatedLlmRequest, ApiSpecificRequest, ContentPart, FunctionCall, GenerationParams, Message,
    MessageContent, ProviderNativeComponent, ToolCall, ToolChoice, ToolDefinition,
};
use super::resolve::{ProviderSurface, ProviderSurfaceDescriptor};
use super::response::{
    AnnotatedLlmResponse, ApiSpecificResponse, FinishReason, ResponseToolCall, Usage,
};
use super::traits::{LlmCodec, LlmResponseCodec};

// ---------------------------------------------------------------------------
// Public codec struct
// ---------------------------------------------------------------------------

/// Built-in codec for the OCI Generative AI chat API.
pub struct OCIGenAIChatCodec;

pub(crate) const PROVIDER_SURFACE: ProviderSurfaceDescriptor = ProviderSurfaceDescriptor {
    surface: ProviderSurface::OCIGenAI,
    detect_request: |obj, hint| {
        // The ChatDetails envelope (chatRequest + servingMode/compartmentId) and
        // the apiFormat discriminator are unique to OCI Generative AI; a bare
        // chatRequest without apiFormat needs the provider hint to classify.
        let has_chat_request = obj.get("chatRequest").is_some_and(Json::is_object);
        let has_envelope_marker =
            obj.get("servingMode").is_some() || obj.get("compartmentId").is_some();
        let hinted_oci =
            hint.is_some_and(|hint_value| hint_value == "oci" || hint_value == "oci.genai");
        (has_chat_request && has_envelope_marker)
            || obj.get("apiFormat").is_some()
            || (hinted_oci && has_chat_request)
    },
    detect_response: |obj| match obj.get("chatResponse") {
        Some(Json::Object(chat_response)) => chat_response.get("apiFormat").is_some(),
        _ => obj.get("apiFormat").is_some(),
    },
    decode_request: |request| OCIGenAIChatCodec.decode(request),
    decode_response: |raw| OCIGenAIChatCodec.decode_response(raw),
    codec_name: "oci_genai",
    request_codec: || std::sync::Arc::new(OCIGenAIChatCodec),
    response_codec: || std::sync::Arc::new(OCIGenAIChatCodec),
    streaming_codec: || Box::new(OCIGenAIStreamingCodec::new()),
};

// ---------------------------------------------------------------------------
// Optional-field helpers
// ---------------------------------------------------------------------------

/// Lookup of an optional list of strings (stop sequences).
fn optional_string_list(
    obj: &serde_json::Map<String, Json>,
    key: &str,
    surface: &str,
) -> Result<Option<Vec<String>>> {
    let Some(value) = obj.get(key) else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    serde_json::from_value::<Vec<String>>(value.clone())
        .map(Some)
        .map_err(|error| {
            FlowError::InvalidArgument(format!("{surface} {key} must be a string array: {error}"))
        })
}

// ---------------------------------------------------------------------------
// Modeled-key bookkeeping
// ---------------------------------------------------------------------------

/// Chat-request keys modeled in [`AnnotatedLlmRequest`] for the GENERIC format.
const MODELED_GENERIC_REQUEST_KEYS: &[&str] = &[
    "apiFormat",
    "messages",
    "maxTokens",
    "temperature",
    "topP",
    "stop",
    "tools",
    "toolChoice",
];

/// Chat-request keys modeled in [`AnnotatedLlmRequest`] for the COHERE format.
const MODELED_COHERE_REQUEST_KEYS: &[&str] = &[
    "apiFormat",
    "message",
    "chatHistory",
    "preambleOverride",
    "maxTokens",
    "temperature",
    "topP",
    "stopSequences",
    "tools",
    "toolChoice",
];

/// Chat-request keys modeled in [`AnnotatedLlmRequest`] for the COHEREV2
/// format: GENERIC-style `messages` with COHERE-style `stopSequences`.
const MODELED_COHERE_V2_REQUEST_KEYS: &[&str] = &[
    "apiFormat",
    "messages",
    "maxTokens",
    "temperature",
    "topP",
    "stopSequences",
    "tools",
    "toolChoice",
];

/// Whether `key` is one of the modeled keys.
fn is_modeled_key(key: &str, modeled: &[&str]) -> bool {
    modeled.contains(&key)
}

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

/// Helper to construct a [`Json`] number from an `f64`.
fn json_f64(v: f64) -> Json {
    serde_json::Number::from_f64(v)
        .map(Json::Number)
        .unwrap_or(Json::Null)
}

fn insert_json(obj: &mut serde_json::Map<String, Json>, key: &str, value: Json) {
    obj.insert(key.to_string(), value);
}

fn set_or_remove_json(obj: &mut serde_json::Map<String, Json>, key: &str, value: Option<Json>) {
    if let Some(value) = value {
        obj.insert(key.into(), value);
    } else {
        obj.remove(key);
    }
}

fn patch_extra_fields(
    obj: &mut serde_json::Map<String, Json>,
    baseline: &serde_json::Map<String, Json>,
    edited: &serde_json::Map<String, Json>,
) {
    for key in baseline.keys().filter(|key| !edited.contains_key(*key)) {
        obj.remove(key);
    }
    for (key, value) in edited {
        if baseline.get(key) != Some(value) {
            obj.insert(key.clone(), value.clone());
        }
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
        // A TEXT part whose `text` is not a string falls through to the
        // provider-native branch so the raw value survives the round trip
        // instead of collapsing to an empty string.
        Some("TEXT") if obj.get("text").is_none_or(Json::is_string) => Ok(ContentPart::Text {
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

/// Wrap normalized content back into the GENERIC typed content-part list.
fn encode_generic_content(content: &MessageContent) -> Result<Json> {
    match content {
        MessageContent::Text(text) => Ok(serde_json::json!([{"type": "TEXT", "text": text}])),
        MessageContent::Parts(parts) => Ok(Json::Array(
            parts
                .iter()
                .map(encode_generic_content_part)
                .collect::<Result<Vec<_>>>()?,
        )),
    }
}

fn encode_generic_content_part(part: &ContentPart) -> Result<Json> {
    match part {
        ContentPart::Text { text, extra } => {
            let mut obj = extra.clone();
            obj.insert("type".into(), Json::String("TEXT".into()));
            obj.insert("text".into(), Json::String(text.clone()));
            Ok(Json::Object(obj))
        }
        ContentPart::ProviderNative {
            provider, value, ..
        } if provider == "oci_genai" => Ok(value.clone()),
        other => Err(FlowError::InvalidArgument(format!(
            "content part {other:?} cannot be encoded for OCI GenAI"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Tool call conversion
// ---------------------------------------------------------------------------

/// Convert a flat OCI `toolCalls` entry into the normalized nested [`ToolCall`].
fn decode_oci_tool_call(value: &Json) -> Result<ToolCall> {
    let obj = value.as_object().ok_or_else(|| {
        FlowError::InvalidArgument("OCI GenAI toolCalls entry must be an object".into())
    })?;
    // A nested `function` object means the entry is already normalized.
    let function = obj.get("function").and_then(Json::as_object);
    let (name, arguments) = match function {
        Some(function) => (function.get("name"), function.get("arguments")),
        None => (obj.get("name"), obj.get("arguments")),
    };
    Ok(ToolCall {
        id: obj
            .get("id")
            .and_then(Json::as_str)
            .unwrap_or_default()
            .to_string(),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: name.and_then(Json::as_str).unwrap_or_default().to_string(),
            arguments: match arguments {
                Some(Json::String(text)) => text.clone(),
                Some(other) => other.to_string(),
                None => String::new(),
            },
        },
    })
}

/// Convert a normalized [`ToolCall`] into the COHEREV2 nested-function shape.
fn encode_oci_tool_call_nested(tool_call: &ToolCall) -> Json {
    let mut function = serde_json::Map::new();
    function.insert("name".into(), Json::String(tool_call.function.name.clone()));
    function.insert(
        "arguments".into(),
        Json::String(tool_call.function.arguments.clone()),
    );
    let mut obj = serde_json::Map::new();
    if !tool_call.id.is_empty() {
        obj.insert("id".into(), Json::String(tool_call.id.clone()));
    }
    obj.insert("type".into(), Json::String("FUNCTION".into()));
    obj.insert("function".into(), Json::Object(function));
    Json::Object(obj)
}

/// Convert a normalized nested [`ToolCall`] back into the flat OCI shape.
///
/// A missing wire `id` decodes to an empty string, so an empty id is omitted
/// on re-encode rather than materializing an `"id": ""` field.
fn encode_oci_tool_call(tool_call: &ToolCall) -> Json {
    let mut obj = serde_json::Map::new();
    if !tool_call.id.is_empty() {
        obj.insert("id".into(), Json::String(tool_call.id.clone()));
    }
    obj.insert("type".into(), Json::String("FUNCTION".into()));
    obj.insert("name".into(), Json::String(tool_call.function.name.clone()));
    obj.insert(
        "arguments".into(),
        Json::String(tool_call.function.arguments.clone()),
    );
    Json::Object(obj)
}

// ---------------------------------------------------------------------------
// GENERIC message decode/encode
// ---------------------------------------------------------------------------

fn decode_generic_message(value: &Json) -> Result<Message> {
    let obj = value.as_object().ok_or_else(|| {
        FlowError::InvalidArgument("OCI GenAI GENERIC message must be an object".into())
    })?;
    let role = obj
        .get("role")
        .and_then(Json::as_str)
        .unwrap_or("USER")
        .to_lowercase();
    let content = decode_generic_content(obj.get("content"))?;
    let tool_calls = match obj.get("toolCalls") {
        None | Some(Json::Null) => None,
        Some(Json::Array(calls)) => Some(
            calls
                .iter()
                .map(decode_oci_tool_call)
                .collect::<Result<Vec<_>>>()?,
        ),
        Some(_) => {
            return Err(FlowError::InvalidArgument(
                "OCI GenAI GENERIC toolCalls must be an array".into(),
            ));
        }
    };
    let tool_call_id = obj
        .get("toolCallId")
        .and_then(Json::as_str)
        .map(str::to_string);
    match role.as_str() {
        "system" => match content {
            Some(content) => Ok(Message::System {
                content,
                name: None,
            }),
            None => Ok(provider_native_message(&role, value)),
        },
        "user" => match content {
            Some(content) => Ok(Message::User {
                content,
                name: None,
            }),
            None => Ok(provider_native_message(&role, value)),
        },
        "assistant" => Ok(Message::Assistant {
            content,
            tool_calls,
            name: None,
        }),
        "tool" => match (content, tool_call_id) {
            (Some(content), Some(tool_call_id)) => Ok(Message::Tool {
                content,
                tool_call_id,
            }),
            _ => Ok(provider_native_message(&role, value)),
        },
        _ => Ok(provider_native_message(&role, value)),
    }
}

fn provider_native_message(kind: &str, value: &Json) -> Message {
    Message::ProviderNative {
        provider: "oci_genai".into(),
        kind: kind.to_string(),
        value: value.clone(),
    }
}

fn encode_generic_message(message: &Message, api_format: &str) -> Result<Json> {
    let mut obj = serde_json::Map::new();
    match message {
        Message::System { content, .. } => {
            obj.insert("role".into(), Json::String("SYSTEM".into()));
            obj.insert("content".into(), encode_generic_content(content)?);
        }
        Message::User { content, .. } => {
            obj.insert("role".into(), Json::String("USER".into()));
            obj.insert("content".into(), encode_generic_content(content)?);
        }
        Message::Assistant {
            content,
            tool_calls,
            ..
        } => {
            obj.insert("role".into(), Json::String("ASSISTANT".into()));
            // `None` round-trips a tool-call-only message: an empty part list
            // decodes to `None`, so re-encode as `[]` rather than `null` to
            // keep the OCI typed-part-list shape.
            obj.insert(
                "content".into(),
                match content {
                    Some(content) => encode_generic_content(content)?,
                    None => Json::Array(Vec::new()),
                },
            );
            if let Some(tool_calls) = tool_calls {
                // COHEREV2 nests name/arguments under `function`; GENERIC is flat.
                let encode = if api_format == "COHEREV2" {
                    encode_oci_tool_call_nested
                } else {
                    encode_oci_tool_call
                };
                obj.insert(
                    "toolCalls".into(),
                    Json::Array(tool_calls.iter().map(encode).collect()),
                );
            }
        }
        Message::Tool {
            content,
            tool_call_id,
        } => {
            obj.insert("role".into(), Json::String("TOOL".into()));
            obj.insert("content".into(), encode_generic_content(content)?);
            obj.insert("toolCallId".into(), Json::String(tool_call_id.clone()));
        }
        Message::ProviderNative {
            provider, value, ..
        } if provider == "oci_genai" => return Ok(value.clone()),
        other => {
            return Err(FlowError::InvalidArgument(format!(
                "message {other:?} cannot be encoded for OCI GenAI"
            )));
        }
    }
    Ok(Json::Object(obj))
}

/// Rewrite only the GENERIC messages that intercepts actually changed.
///
/// Unchanged messages are carried over from the raw payload verbatim so
/// per-message provider fields without a normalized equivalent survive.
fn patch_generic_messages(
    chat_request: &mut serde_json::Map<String, Json>,
    edited: &[Message],
    baseline: &[Message],
    api_format: &str,
) -> Result<()> {
    let raw_messages: Vec<Json> = chat_request
        .get("messages")
        .and_then(Json::as_array)
        .cloned()
        .unwrap_or_default();
    let patched = edited
        .iter()
        .enumerate()
        .map(|(index, message)| {
            let unchanged = baseline.get(index) == Some(message);
            match raw_messages.get(index) {
                Some(raw) if unchanged => Ok(raw.clone()),
                _ => encode_generic_message(message, api_format),
            }
        })
        .collect::<Result<Vec<_>>>()?;
    insert_json(chat_request, "messages", Json::Array(patched));
    Ok(())
}

// ---------------------------------------------------------------------------
// COHERE message decode/encode
// ---------------------------------------------------------------------------

fn decode_cohere_messages(chat_request: &serde_json::Map<String, Json>) -> Result<Vec<Message>> {
    let mut messages = Vec::new();

    if let Some(preamble) = chat_request.get("preambleOverride").and_then(Json::as_str)
        && !preamble.is_empty()
    {
        messages.push(Message::System {
            content: MessageContent::Text(preamble.to_string()),
            name: None,
        });
    }

    if let Some(history) = chat_request.get("chatHistory") {
        let turns = history.as_array().ok_or_else(|| {
            FlowError::InvalidArgument("OCI GenAI COHERE chatHistory must be an array".into())
        })?;
        for turn in turns {
            messages.push(decode_cohere_turn(turn)?);
        }
    }

    if let Some(current) = chat_request.get("message").and_then(Json::as_str) {
        messages.push(Message::User {
            content: MessageContent::Text(current.to_string()),
            name: None,
        });
    }

    Ok(messages)
}

fn decode_cohere_turn(turn: &Json) -> Result<Message> {
    let obj = turn.as_object().ok_or_else(|| {
        FlowError::InvalidArgument("OCI GenAI COHERE chatHistory turn must be an object".into())
    })?;
    let role = obj
        .get("role")
        .and_then(Json::as_str)
        .unwrap_or("USER")
        .to_uppercase();
    let Some(text) = obj.get("message").and_then(Json::as_str) else {
        return Ok(provider_native_message(&role, turn));
    };
    let content = MessageContent::Text(text.to_string());
    match role.as_str() {
        "USER" => Ok(Message::User {
            content,
            name: None,
        }),
        "CHATBOT" => Ok(Message::Assistant {
            content: Some(content),
            tool_calls: None,
            name: None,
        }),
        "SYSTEM" => Ok(Message::System {
            content,
            name: None,
        }),
        _ => Ok(provider_native_message(&role, turn)),
    }
}

/// Extract the plain-text body of a normalized message for COHERE encoding.
fn cohere_text(content: &MessageContent) -> Result<String> {
    match content {
        MessageContent::Text(text) => Ok(text.clone()),
        MessageContent::Parts(_) => Err(FlowError::InvalidArgument(
            "multimodal content cannot be encoded for the OCI GenAI COHERE format".into(),
        )),
    }
}

/// Rebuild the COHERE `preambleOverride`/`chatHistory`/`message` fields from
/// edited messages. COHERE turns are plain strings, so edits rebuild the
/// modeled fields rather than patching individual turns.
fn encode_cohere_messages(
    chat_request: &mut serde_json::Map<String, Json>,
    messages: &[Message],
) -> Result<()> {
    let mut remaining = messages;

    if let Some(Message::System { content, .. }) = remaining.first() {
        insert_json(
            chat_request,
            "preambleOverride",
            Json::String(cohere_text(content)?),
        );
        remaining = &remaining[1..];
    } else {
        // The encoder merges into the raw request, so without this removal a
        // preamble deleted (or re-roled) by an intercept would survive on the
        // wire while the normalized annotation no longer contains it.
        chat_request.remove("preambleOverride");
    }

    // The COHERE chat request requires a non-empty `message` prompt; an edited
    // list without a trailing user turn would otherwise silently send `""`.
    let Some(Message::User { content, .. }) = remaining.last() else {
        return Err(FlowError::InvalidArgument(
            "OCI GenAI COHERE requests require the last message to be a user message".into(),
        ));
    };
    let current = cohere_text(content)?;
    remaining = &remaining[..remaining.len() - 1];

    let history = remaining
        .iter()
        .map(encode_cohere_turn)
        .collect::<Result<Vec<_>>>()?;

    chat_request.insert("message".into(), Json::String(current));
    if !history.is_empty() || chat_request.contains_key("chatHistory") {
        chat_request.insert("chatHistory".into(), Json::Array(history));
    }
    Ok(())
}

fn encode_cohere_turn(message: &Message) -> Result<Json> {
    let (role, content) = match message {
        Message::User { content, .. } => ("USER", content),
        Message::Assistant {
            content: Some(content),
            ..
        } => ("CHATBOT", content),
        Message::System { content, .. } => ("SYSTEM", content),
        // OCI's CohereToolMessage carries structured `toolResults`, not a plain
        // `message` string, and has no field for the normalized tool_call_id.
        // Reject rather than silently dropping the identifier; wire-shaped TOOL
        // turns survive untouched as ProviderNative messages below.
        Message::Tool { .. } => {
            return Err(FlowError::InvalidArgument(
                "OCI GenAI COHERE tool results cannot be rebuilt from a normalized tool message"
                    .into(),
            ));
        }
        Message::ProviderNative {
            provider, value, ..
        } if provider == "oci_genai" => return Ok(value.clone()),
        other => {
            return Err(FlowError::InvalidArgument(format!(
                "message {other:?} cannot be encoded as an OCI GenAI COHERE chatHistory turn"
            )));
        }
    };
    Ok(serde_json::json!({"role": role, "message": cohere_text(content)?}))
}

// ---------------------------------------------------------------------------
// Params, tools, and envelope helpers
// ---------------------------------------------------------------------------

/// Decode the normalized generation params for one API format.
fn decode_params(
    chat_request: &serde_json::Map<String, Json>,
    api_format: &str,
) -> Result<Option<GenerationParams>> {
    const SURFACE: &str = "OCI GenAI";
    let temperature = super::optional_f64(chat_request, "temperature", SURFACE)?;
    let max_tokens = super::optional_u64(chat_request, "maxTokens", SURFACE)?;
    let top_p = super::optional_f64(chat_request, "topP", SURFACE)?;
    // Both Cohere formats spell the stop list `stopSequences`.
    let stop_key = if api_format.starts_with("COHERE") {
        "stopSequences"
    } else {
        "stop"
    };
    let stop = optional_string_list(chat_request, stop_key, SURFACE)?;
    if temperature.is_some() || max_tokens.is_some() || top_p.is_some() || stop.is_some() {
        Ok(Some(GenerationParams {
            temperature,
            max_tokens,
            top_p,
            stop,
        }))
    } else {
        Ok(None)
    }
}

/// Patch only the generation params an intercept actually changed.
///
/// A param cleared to `None` removes the raw key, matching the set-or-remove
/// semantics of the other provider codecs.
fn patch_params(
    chat_request: &mut serde_json::Map<String, Json>,
    edited: Option<&GenerationParams>,
    baseline: Option<&GenerationParams>,
    api_format: &str,
) {
    if edited == baseline {
        return;
    }
    let temperature = edited.and_then(|params| params.temperature);
    if temperature != baseline.and_then(|params| params.temperature) {
        set_or_remove_json(chat_request, "temperature", temperature.map(json_f64));
    }
    let top_p = edited.and_then(|params| params.top_p);
    if top_p != baseline.and_then(|params| params.top_p) {
        set_or_remove_json(chat_request, "topP", top_p.map(json_f64));
    }
    let max_tokens = edited.and_then(|params| params.max_tokens);
    if max_tokens != baseline.and_then(|params| params.max_tokens) {
        set_or_remove_json(chat_request, "maxTokens", max_tokens.map(Json::from));
    }
    let stop = edited.and_then(|params| params.stop.as_ref());
    if stop != baseline.and_then(|params| params.stop.as_ref()) {
        let stop_key = if api_format.starts_with("COHERE") {
            "stopSequences"
        } else {
            "stop"
        };
        set_or_remove_json(
            chat_request,
            stop_key,
            stop.map(|values| serde_json::json!(values)),
        );
    }
}

fn decode_tools(
    chat_request: &serde_json::Map<String, Json>,
) -> Result<Option<Vec<ToolDefinition>>> {
    match chat_request.get("tools") {
        None | Some(Json::Null) => Ok(None),
        Some(Json::Array(tools)) => Ok(Some(
            tools
                .iter()
                .map(|tool| {
                    let native = native_component(tool);
                    ToolDefinition::ProviderNative {
                        provider: native.provider,
                        kind: native.kind,
                        value: native.value,
                    }
                })
                .collect(),
        )),
        Some(_) => Err(FlowError::InvalidArgument(
            "OCI GenAI tools must be an array".into(),
        )),
    }
}

fn encode_oci_tool(tool: &ToolDefinition) -> Result<Json> {
    match tool {
        ToolDefinition::ProviderNative {
            provider, value, ..
        } if provider == "oci_genai" => Ok(value.clone()),
        ToolDefinition::Function { function, extra } => {
            let mut obj = extra.clone();
            obj.insert("type".into(), Json::String("FUNCTION".into()));
            obj.insert("name".into(), Json::String(function.name.clone()));
            if let Some(description) = &function.description {
                obj.insert("description".into(), Json::String(description.clone()));
            }
            if let Some(parameters) = &function.parameters {
                obj.insert("parameters".into(), parameters.clone());
            }
            obj.extend(function.extra.clone());
            Ok(Json::Object(obj))
        }
        other => Err(FlowError::InvalidArgument(format!(
            "tool {other:?} cannot be encoded for OCI GenAI"
        ))),
    }
}

fn encode_oci_tool_choice(tool_choice: &ToolChoice) -> Result<Json> {
    match tool_choice {
        ToolChoice::ProviderNative(native) if native.provider == "oci_genai" => {
            Ok(native.value.clone())
        }
        other => Err(FlowError::InvalidArgument(format!(
            "tool choice {other:?} cannot be encoded for OCI GenAI"
        ))),
    }
}

/// Extract the model identity from the `servingMode` envelope object.
fn model_from_envelope(envelope: &serde_json::Map<String, Json>) -> Option<String> {
    let serving_mode = envelope.get("servingMode")?.as_object()?;
    serving_mode
        .get("modelId")
        .or_else(|| serving_mode.get("endpointId"))
        .and_then(Json::as_str)
        .map(str::to_string)
}

/// Split the request content into the optional ChatDetails envelope and the
/// chat request object.
fn split_envelope(
    obj: &serde_json::Map<String, Json>,
) -> (
    Option<&serde_json::Map<String, Json>>,
    &serde_json::Map<String, Json>,
) {
    match obj.get("chatRequest").and_then(Json::as_object) {
        Some(chat_request) => (Some(obj), chat_request),
        None => (None, obj),
    }
}

/// Resolve the request API format (uppercased), defaulting to `GENERIC`.
fn request_api_format(chat_request: &serde_json::Map<String, Json>) -> String {
    chat_request
        .get("apiFormat")
        .and_then(Json::as_str)
        .unwrap_or("GENERIC")
        .to_uppercase()
}

fn validate_oci_supported_fields(
    annotated: &AnnotatedLlmRequest,
    baseline: &AnnotatedLlmRequest,
) -> Result<()> {
    let unsupported = [
        annotated.model != baseline.model,
        annotated.instructions != baseline.instructions,
        annotated.store != baseline.store,
        annotated.previous_response_id != baseline.previous_response_id,
        annotated.truncation != baseline.truncation,
        annotated.reasoning != baseline.reasoning,
        annotated.include != baseline.include,
        annotated.user != baseline.user,
        annotated.metadata != baseline.metadata,
        annotated.service_tier != baseline.service_tier,
        annotated.parallel_tool_calls != baseline.parallel_tool_calls,
        annotated.max_output_tokens != baseline.max_output_tokens,
        annotated.max_tool_calls != baseline.max_tool_calls,
        annotated.top_logprobs != baseline.top_logprobs,
        annotated.stream != baseline.stream,
    ]
    .into_iter()
    .any(|changed| changed);
    if unsupported {
        return Err(FlowError::InvalidArgument(
            "request contains fields that cannot be encoded for OCI GenAI".into(),
        ));
    }
    Ok(())
}

/// Patch envelope-level fields (`compartmentId`, `servingMode`) when the
/// api-specific annotation changed them.
///
/// `api_format` is read-only: the encoder patches the raw payload in place, so
/// switching formats cannot rebuild the body without leaving the other
/// format's modeled fields behind, and an edit is rejected instead.
fn patch_oci_api_specific(
    envelope: Option<&mut serde_json::Map<String, Json>>,
    edited: &Option<ApiSpecificRequest>,
    baseline: &Option<ApiSpecificRequest>,
) -> Result<()> {
    let (compartment_id, serving_mode, old_compartment_id, old_serving_mode) =
        match (edited, baseline) {
            (
                Some(ApiSpecificRequest::OCIGenAI {
                    compartment_id,
                    serving_mode,
                    api_format,
                }),
                Some(ApiSpecificRequest::OCIGenAI {
                    compartment_id: old_compartment_id,
                    serving_mode: old_serving_mode,
                    api_format: old_api_format,
                }),
            ) => {
                if api_format != old_api_format {
                    return Err(FlowError::InvalidArgument(
                        "the OCI GenAI api_format cannot be edited".into(),
                    ));
                }
                (
                    compartment_id,
                    serving_mode,
                    old_compartment_id,
                    old_serving_mode,
                )
            }
            // A dropped api_specific annotation leaves the envelope untouched;
            // the raw payload keeps serving as the source of truth.
            (None, _) => return Ok(()),
            (Some(_), _) => {
                return Err(FlowError::InvalidArgument(
                    "api_specific provider does not match OCI GenAI".into(),
                ));
            }
        };
    if compartment_id == old_compartment_id && serving_mode == old_serving_mode {
        return Ok(());
    }
    let Some(envelope) = envelope else {
        return Err(FlowError::InvalidArgument(
            "compartmentId and servingMode edits require a ChatDetails envelope".into(),
        ));
    };
    if compartment_id != old_compartment_id {
        set_or_remove_json(
            envelope,
            "compartmentId",
            compartment_id.clone().map(Json::String),
        );
    }
    if serving_mode != old_serving_mode {
        set_or_remove_json(envelope, "servingMode", serving_mode.clone());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// LlmCodec implementation
// ---------------------------------------------------------------------------

impl LlmCodec for OCIGenAIChatCodec {
    fn codec_identity(&self) -> LlmCodecIdentity {
        LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OCIGenAI)
    }

    fn decode(&self, request: &LlmRequest) -> Result<AnnotatedLlmRequest> {
        let obj = request
            .content
            .as_object()
            .ok_or_else(|| FlowError::Internal("request content is not an object".into()))?;
        let (envelope, chat_request) = split_envelope(obj);
        let api_format = request_api_format(chat_request);

        let messages = if api_format == "COHERE" {
            decode_cohere_messages(chat_request)?
        } else {
            match chat_request.get("messages") {
                None | Some(Json::Null) => Vec::new(),
                Some(Json::Array(messages)) => messages
                    .iter()
                    .map(decode_generic_message)
                    .collect::<Result<Vec<_>>>()?,
                Some(_) => {
                    return Err(FlowError::InvalidArgument(
                        "OCI GenAI GENERIC messages must be an array".into(),
                    ));
                }
            }
        };
        let params = decode_params(chat_request, &api_format)?;
        let tools = decode_tools(chat_request)?;
        let tool_choice = chat_request
            .get("toolChoice")
            .filter(|value| !value.is_null())
            .map(|value| ToolChoice::ProviderNative(native_component(value)));

        let modeled = match api_format.as_str() {
            "COHERE" => MODELED_COHERE_REQUEST_KEYS,
            "COHEREV2" => MODELED_COHERE_V2_REQUEST_KEYS,
            _ => MODELED_GENERIC_REQUEST_KEYS,
        };
        let extra: serde_json::Map<String, Json> = chat_request
            .iter()
            .filter(|(key, _)| !is_modeled_key(key, modeled))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();

        Ok(AnnotatedLlmRequest {
            messages,
            instructions: None,
            model: envelope.and_then(model_from_envelope),
            params,
            tools,
            tool_choice,
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
            api_specific: Some(ApiSpecificRequest::OCIGenAI {
                compartment_id: envelope
                    .and_then(|envelope| envelope.get("compartmentId"))
                    .and_then(Json::as_str)
                    .map(str::to_string),
                serving_mode: envelope
                    .and_then(|envelope| envelope.get("servingMode"))
                    .cloned(),
                api_format: Some(api_format),
            }),
            extra,
        })
    }

    fn encode(&self, annotated: &AnnotatedLlmRequest, original: &LlmRequest) -> Result<LlmRequest> {
        let baseline = self.decode(original)?;
        let mut content = original.content.clone();
        let obj = content
            .as_object_mut()
            .ok_or_else(|| FlowError::Internal("original content is not an object".into()))?;

        // Split the mutable envelope from a working copy of the chat request.
        let chat_request_key = obj
            .get("chatRequest")
            .is_some_and(Json::is_object)
            .then(|| "chatRequest".to_string());
        let mut chat_request = match &chat_request_key {
            Some(key) => obj
                .get(key)
                .and_then(Json::as_object)
                .cloned()
                .unwrap_or_default(),
            None => obj.clone(),
        };
        let api_format = request_api_format(&chat_request);

        validate_oci_supported_fields(annotated, &baseline)?;

        if annotated.messages != baseline.messages {
            if api_format == "COHERE" {
                encode_cohere_messages(&mut chat_request, &annotated.messages)?;
            } else {
                patch_generic_messages(
                    &mut chat_request,
                    &annotated.messages,
                    &baseline.messages,
                    &api_format,
                )?;
            }
        }

        patch_params(
            &mut chat_request,
            annotated.params.as_ref(),
            baseline.params.as_ref(),
            &api_format,
        );

        if annotated.tools != baseline.tools {
            let tools = annotated
                .tools
                .as_deref()
                .map(|tools| {
                    tools
                        .iter()
                        .map(encode_oci_tool)
                        .collect::<Result<Vec<_>>>()
                })
                .transpose()?
                .map(Json::Array);
            set_or_remove_json(&mut chat_request, "tools", tools);
        }
        if annotated.tool_choice != baseline.tool_choice {
            let tool_choice = annotated
                .tool_choice
                .as_ref()
                .map(encode_oci_tool_choice)
                .transpose()?;
            set_or_remove_json(&mut chat_request, "toolChoice", tool_choice);
        }

        patch_extra_fields(&mut chat_request, &baseline.extra, &annotated.extra);

        match chat_request_key {
            Some(key) => {
                obj.insert(key, Json::Object(chat_request));
                patch_oci_api_specific(Some(obj), &annotated.api_specific, &baseline.api_specific)?;
                Ok(LlmRequest {
                    headers: original.headers.clone(),
                    content,
                })
            }
            None => {
                patch_oci_api_specific(None, &annotated.api_specific, &baseline.api_specific)?;
                Ok(LlmRequest {
                    headers: original.headers.clone(),
                    content: Json::Object(chat_request),
                })
            }
        }
    }
}

// ---------------------------------------------------------------------------
// LlmResponseCodec implementation
// ---------------------------------------------------------------------------

impl LlmResponseCodec for OCIGenAIChatCodec {
    fn codec_identity(&self) -> LlmCodecIdentity {
        LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OCIGenAI)
    }

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
// Streaming codec
// ---------------------------------------------------------------------------

/// Streaming counterpart to [`OCIGenAIChatCodec`].
///
/// Replays the OCI Generative AI SSE event sequence into the same JSON shape a
/// non-streaming `ChatResult` carries (`{modelId, chatResponse: {apiFormat,
/// ...}}`). Once finalized, the assembled JSON can be fed back through
/// [`OCIGenAIChatCodec::decode_response`] to produce an
/// [`AnnotatedLlmResponse`] — meaning streaming and non-streaming OCI requests
/// converge on the same observability output.
///
/// # Strategy
///
/// OCI streams untagged chat-response deltas. `GENERIC` events carry
/// `{index, message: {role, content: [{type: "TEXT", text}], toolCalls}, finishReason}`
/// fragments whose text and tool-call `arguments` accumulate per choice index;
/// `COHERE` events carry incremental `{apiFormat: "COHERE", text}` fragments
/// with `finishReason` on the terminal event. Events wrapped in a
/// `chatResponse` envelope are unwrapped first, and `modelId`/`usage` are
/// captured whenever a chunk supplies them.
///
/// Internal state lives behind `Arc<Mutex<...>>` so the `&self`-produced
/// collector and finalizer closures share access. Each instance is single-use
/// because [`LlmFinalizerFn`] consumes the finalize step.
///
/// [`LlmFinalizerFn`]: crate::api::runtime::LlmFinalizerFn
pub struct OCIGenAIStreamingCodec {
    state: std::sync::Arc<std::sync::Mutex<OCIGenAIStreamingState>>,
}

impl OCIGenAIStreamingCodec {
    /// Creates a fresh streaming codec with empty accumulator state.
    pub fn new() -> Self {
        Self {
            state: std::sync::Arc::new(std::sync::Mutex::new(OCIGenAIStreamingState::default())),
        }
    }
}

impl Default for OCIGenAIStreamingCodec {
    fn default() -> Self {
        Self::new()
    }
}

impl super::streaming::StreamingCodec for OCIGenAIStreamingCodec {
    fn collector(&self) -> crate::api::runtime::LlmCollectorFn {
        let state = std::sync::Arc::clone(&self.state);
        Box::new(move |event: Json| -> Result<()> {
            let mut guard = state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            guard.observe(&event);
            Ok(())
        })
    }

    fn finalizer(&self) -> crate::api::runtime::LlmFinalizerFn {
        let state = std::sync::Arc::clone(&self.state);
        Box::new(move || -> Json {
            let mut guard = state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            // Move state out so finalize can consume it; the codec is single-use, so leaving a
            // default behind is intentional and never observed by another caller.
            std::mem::take(&mut *guard).finalize()
        })
    }
}

#[derive(Debug, Default)]
struct OCIGenAIStreamingState {
    model_id: Option<String>,
    /// Resolved from the first event's `apiFormat`, or inferred from the event
    /// shape (`message`/`index` => GENERIC, bare `text` => COHERE).
    api_format: Option<String>,
    /// Latest non-null usage snapshot; the terminal event's counters win.
    usage: Option<Json>,
    /// Per-choice accumulators keyed by `index`. BTreeMap so finalize emits
    /// choices in stable order.
    choices: std::collections::BTreeMap<u64, OCIChoiceState>,
    cohere_text: String,
    cohere_finish_reason: Option<String>,
}

#[derive(Debug, Default)]
struct OCIChoiceState {
    role: Option<String>,
    /// Typed content parts in arrival order; consecutive TEXT fragments merge
    /// into one part, non-TEXT typed parts (THINKING, IMAGE_URL, DOCUMENT,
    /// future kinds) are preserved verbatim.
    parts: Vec<Json>,
    /// Tool-call accumulators in first-seen order. Fragments that carry an
    /// `id` are matched to the accumulator with that id (OCI provides no
    /// per-call `index`, and parallel calls can each arrive at event-local
    /// position 0); id-less fragments fall back to their array position.
    tool_calls: Vec<OCIToolCallState>,
    finish_reason: Option<String>,
}

#[derive(Debug, Default)]
struct OCIToolCallState {
    id: Option<String>,
    type_: Option<String>,
    name: Option<String>,
    arguments: String,
}

impl OCIGenAIStreamingState {
    fn observe(&mut self, event: &Json) {
        let Some(obj) = event.as_object() else {
            return;
        };
        // Some transports wrap each delta in the ChatResult envelope; unwrap it.
        let inner = obj.get("chatResponse").and_then(Json::as_object);
        if let Some(model_id) = obj.get("modelId").and_then(Json::as_str) {
            self.model_id = Some(model_id.to_string());
        }
        let obj = inner.unwrap_or(obj);

        if self.api_format.is_none() {
            self.api_format = obj
                .get("apiFormat")
                .and_then(Json::as_str)
                .map(str::to_uppercase)
                .or_else(|| self.infer_api_format(obj));
        }
        if let Some(usage) = obj.get("usage")
            && !usage.is_null()
        {
            self.usage = Some(usage.clone());
        }

        if self.api_format.as_deref() == Some("COHERE") {
            let finish_reason = obj.get("finishReason").and_then(Json::as_str);
            if let Some(text) = obj.get("text").and_then(Json::as_str) {
                if finish_reason.is_some() && !text.is_empty() {
                    // The live service's terminal COHERE event repeats the
                    // complete response text (alongside chatHistory and
                    // finishReason); take it as authoritative rather than
                    // appending, which would double the assembled text.
                    self.cohere_text = text.to_string();
                } else {
                    self.cohere_text.push_str(text);
                }
            }
            if let Some(reason) = finish_reason {
                self.cohere_finish_reason = Some(reason.to_string());
            }
            return;
        }

        // GENERIC: the event is either a bare choice delta or carries a
        // `choices` array of deltas.
        match obj.get("choices").and_then(Json::as_array) {
            Some(choices) => {
                for choice in choices {
                    if let Some(choice) = choice.as_object() {
                        self.observe_generic_choice(choice);
                    }
                }
            }
            None => self.observe_generic_choice(obj),
        }
    }

    fn infer_api_format(&self, obj: &serde_json::Map<String, Json>) -> Option<String> {
        if obj.get("message").is_some()
            || obj.get("choices").is_some()
            || obj.get("index").is_some()
        {
            Some("GENERIC".to_string())
        } else if obj.get("text").is_some() {
            Some("COHERE".to_string())
        } else {
            None
        }
    }

    fn observe_generic_choice(&mut self, choice: &serde_json::Map<String, Json>) {
        let index = choice.get("index").and_then(Json::as_u64).unwrap_or(0);
        let entry = self.choices.entry(index).or_default();
        if let Some(reason) = choice.get("finishReason").and_then(Json::as_str) {
            entry.finish_reason = Some(reason.to_string());
        }
        let Some(message) = choice.get("message").and_then(Json::as_object) else {
            return;
        };
        if let Some(role) = message.get("role").and_then(Json::as_str) {
            entry.role = Some(role.to_string());
        }
        if let Some(parts) = message.get("content").and_then(Json::as_array) {
            for part in parts {
                entry.observe_content_part(part);
            }
        }
        if let Some(tool_calls) = message.get("toolCalls").and_then(Json::as_array) {
            for (position, tool_call) in tool_calls.iter().enumerate() {
                if let Some(tool_call) = tool_call.as_object() {
                    entry.observe_tool_call(position, tool_call);
                }
            }
        }
    }

    fn finalize(self) -> Json {
        let api_format = self.api_format.unwrap_or_else(|| "GENERIC".to_string());
        let mut chat_response = serde_json::Map::new();
        chat_response.insert("apiFormat".to_string(), Json::String(api_format.clone()));
        if api_format == "COHERE" {
            chat_response.insert("text".to_string(), Json::String(self.cohere_text));
            if let Some(reason) = self.cohere_finish_reason {
                chat_response.insert("finishReason".to_string(), Json::String(reason));
            }
        } else if api_format == "COHEREV2" {
            // The COHEREV2 response decoder reads a single root-level
            // `message` (nested-function tool calls, root finishReason)
            // rather than a `choices` array.
            let choice = self.choices.into_values().next().unwrap_or_default();
            let finish_reason = choice.finish_reason.clone();
            chat_response.insert("message".to_string(), choice.finalize_v2_message());
            if let Some(reason) = finish_reason {
                chat_response.insert("finishReason".to_string(), Json::String(reason));
            }
        } else {
            let choices: Vec<Json> = self
                .choices
                .into_iter()
                .map(|(index, choice)| choice.finalize(index))
                .collect();
            chat_response.insert("choices".to_string(), Json::Array(choices));
        }
        if let Some(usage) = self.usage {
            chat_response.insert("usage".to_string(), usage);
        }
        let mut output = serde_json::Map::new();
        if let Some(model_id) = self.model_id {
            output.insert("modelId".to_string(), Json::String(model_id));
        }
        output.insert("chatResponse".to_string(), Json::Object(chat_response));
        Json::Object(output)
    }
}

impl OCIChoiceState {
    fn observe_content_part(&mut self, part: &Json) {
        if part.get("type").and_then(Json::as_str) == Some("TEXT") {
            let Some(text) = part.get("text").and_then(Json::as_str) else {
                return;
            };
            if let Some(Json::Object(last)) = self.parts.last_mut()
                && last.get("type").and_then(Json::as_str) == Some("TEXT")
            {
                let merged = format!(
                    "{}{}",
                    last.get("text").and_then(Json::as_str).unwrap_or_default(),
                    text
                );
                last.insert("text".to_string(), Json::String(merged));
                return;
            }
            self.parts
                .push(serde_json::json!({"type": "TEXT", "text": text}));
            return;
        }
        if part.is_object() {
            self.parts.push(part.clone());
        }
    }

    /// The accumulated parts with empty TEXT placeholders removed; an
    /// all-empty stream yields `[]` so the response decode reports no
    /// assistant message, matching the non-streaming path.
    fn content_parts(parts: Vec<Json>) -> Vec<Json> {
        parts
            .into_iter()
            .filter(|part| {
                part.get("type").and_then(Json::as_str) != Some("TEXT")
                    || part
                        .get("text")
                        .and_then(Json::as_str)
                        .is_some_and(|text| !text.is_empty())
            })
            .collect()
    }

    fn observe_tool_call(&mut self, position: usize, tool_call: &serde_json::Map<String, Json>) {
        let slot = match tool_call.get("id").and_then(Json::as_str) {
            Some(id) => self
                .tool_calls
                .iter()
                .position(|state| state.id.as_deref() == Some(id))
                .unwrap_or_else(|| {
                    self.tool_calls.push(OCIToolCallState::default());
                    self.tool_calls.len() - 1
                }),
            None if position < self.tool_calls.len() => position,
            None => {
                self.tool_calls.push(OCIToolCallState::default());
                self.tool_calls.len() - 1
            }
        };
        let state = &mut self.tool_calls[slot];
        if let Some(id) = tool_call.get("id").and_then(Json::as_str) {
            state.id = Some(id.to_string());
        }
        if let Some(type_) = tool_call.get("type").and_then(Json::as_str) {
            state.type_ = Some(type_.to_string());
        }
        // COHEREV2 fragments nest name/arguments under a `function` object;
        // GENERIC fragments are flat.
        let body = tool_call
            .get("function")
            .and_then(Json::as_object)
            .unwrap_or(tool_call);
        if let Some(name) = body.get("name").and_then(Json::as_str) {
            state.name = Some(name.to_string());
        }
        if let Some(arguments) = body.get("arguments").and_then(Json::as_str) {
            state.arguments.push_str(arguments);
        }
    }

    /// Assemble the accumulated choice as a COHEREV2 root `message`
    /// (typed content parts, nested-function tool calls).
    fn finalize_v2_message(self) -> Json {
        let mut message = serde_json::Map::new();
        message.insert(
            "role".to_string(),
            Json::String(self.role.unwrap_or_else(|| "ASSISTANT".to_string())),
        );
        message.insert(
            "content".to_string(),
            Json::Array(Self::content_parts(self.parts)),
        );
        if !self.tool_calls.is_empty() {
            let tool_calls: Vec<Json> = self
                .tool_calls
                .into_iter()
                .map(OCIToolCallState::finalize_v2)
                .collect();
            message.insert("toolCalls".to_string(), Json::Array(tool_calls));
        }
        Json::Object(message)
    }

    fn finalize(self, index: u64) -> Json {
        let mut message = serde_json::Map::new();
        message.insert(
            "role".to_string(),
            Json::String(self.role.unwrap_or_else(|| "ASSISTANT".to_string())),
        );
        message.insert(
            "content".to_string(),
            Json::Array(Self::content_parts(self.parts)),
        );
        if !self.tool_calls.is_empty() {
            let tool_calls: Vec<Json> = self
                .tool_calls
                .into_iter()
                .map(OCIToolCallState::finalize)
                .collect();
            message.insert("toolCalls".to_string(), Json::Array(tool_calls));
        }
        let mut choice = serde_json::Map::new();
        choice.insert("index".to_string(), Json::Number(index.into()));
        choice.insert("message".to_string(), Json::Object(message));
        if let Some(reason) = self.finish_reason {
            choice.insert("finishReason".to_string(), Json::String(reason));
        }
        Json::Object(choice)
    }
}

impl OCIToolCallState {
    /// Assemble the call in the COHEREV2 nested-function wire shape.
    fn finalize_v2(self) -> Json {
        let mut function = serde_json::Map::new();
        function.insert(
            "name".to_string(),
            Json::String(self.name.unwrap_or_default()),
        );
        function.insert("arguments".to_string(), Json::String(self.arguments));
        let mut call = serde_json::Map::new();
        if let Some(id) = self.id {
            call.insert("id".to_string(), Json::String(id));
        }
        call.insert(
            "type".to_string(),
            Json::String(self.type_.unwrap_or_else(|| "FUNCTION".to_string())),
        );
        call.insert("function".to_string(), Json::Object(function));
        Json::Object(call)
    }

    fn finalize(self) -> Json {
        let mut call = serde_json::Map::new();
        if let Some(id) = self.id {
            call.insert("id".to_string(), Json::String(id));
        }
        call.insert(
            "type".to_string(),
            Json::String(self.type_.unwrap_or_else(|| "FUNCTION".to_string())),
        );
        call.insert(
            "name".to_string(),
            Json::String(self.name.unwrap_or_default()),
        );
        call.insert("arguments".to_string(), Json::String(self.arguments));
        Json::Object(call)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "../../tests/unit/codec/oci_genai_tests.rs"]
mod tests;
