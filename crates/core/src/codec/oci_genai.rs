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
//! - **Two API formats** selected by `chatRequest.apiFormat`:
//!   - `GENERIC`: OpenAI-style `messages` with UPPERCASE roles
//!     (`USER`/`ASSISTANT`/`SYSTEM`/`TOOL`) whose `content` is a list of typed
//!     parts (`{"type": "TEXT", "text": ...}`), flat `toolCalls`
//!     (`{id, type: "FUNCTION", name, arguments}`), and `toolCallId` on tool
//!     messages. Used by Meta Llama, Google, xAI, OpenAI, and imported
//!     open-weights models hosted on dedicated AI clusters.
//!   - `COHERE`: a single `message` string plus `chatHistory` turns with
//!     `USER`/`CHATBOT`/`SYSTEM` roles and an optional `preambleOverride`.
//!     Used by Cohere Command models.
//! - **Key conventions**: The OCI SDKs emit camelCase JSON while the OCI CLI
//!   emits kebab-case; decode tolerates camelCase, kebab-case, and snake_case.
//! - **Model identity**: Carried in `servingMode.modelId` (on-demand) or
//!   `servingMode.endpointId` (dedicated), not in the chat request body.
//! - **Responses**: `ChatResult` payloads (`modelId`, `chatResponse`) where the
//!   chat response is `choices`-based for `GENERIC` and `text`-based for
//!   `COHERE`; `usage` counters are `promptTokens`/`completionTokens`/`totalTokens`.

use crate::api::llm::LlmRequest;
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
        let has_chat_request = get_first(obj, "chatRequest").is_some_and(Json::is_object);
        let has_envelope_marker =
            get_first(obj, "servingMode").is_some() || get_first(obj, "compartmentId").is_some();
        let hinted_oci =
            hint.is_some_and(|hint_value| hint_value == "oci" || hint_value == "oci.genai");
        (has_chat_request && has_envelope_marker)
            || get_first(obj, "apiFormat").is_some()
            || (hinted_oci && has_chat_request)
    },
    detect_response: |obj| match get_first(obj, "chatResponse") {
        Some(Json::Object(chat_response)) => get_first(chat_response, "apiFormat").is_some(),
        _ => get_first(obj, "apiFormat").is_some(),
    },
    decode_request: |request| OCIGenAIChatCodec.decode(request),
    decode_response: |raw| OCIGenAIChatCodec.decode_response(raw),
    codec_name: "oci_genai",
    request_codec: || std::sync::Arc::new(OCIGenAIChatCodec),
    response_codec: || std::sync::Arc::new(OCIGenAIChatCodec),
    streaming_codec: || Box::new(OCIGenAIStreamingCodec::new()),
};

// ---------------------------------------------------------------------------
// Key-convention helpers
// ---------------------------------------------------------------------------

/// Convert a camelCase key to its kebab-case form (`maxTokens` -> `max-tokens`).
fn camel_to_kebab(key: &str) -> String {
    camel_with_separator(key, '-')
}

/// Convert a camelCase key to its snake_case form (`maxTokens` -> `max_tokens`).
fn camel_to_snake(key: &str) -> String {
    camel_with_separator(key, '_')
}

fn camel_with_separator(key: &str, separator: char) -> String {
    let mut out = String::with_capacity(key.len() + 4);
    for c in key.chars() {
        if c.is_ascii_uppercase() {
            out.push(separator);
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

/// Return the value for the first present key spelling across naming conventions.
///
/// The OCI SDKs emit camelCase JSON while the OCI CLI emits kebab-case; callers
/// pass camelCase keys and kebab-case/snake_case fallbacks are derived.
fn get_first<'a>(obj: &'a serde_json::Map<String, Json>, key: &str) -> Option<&'a Json> {
    present_key(obj, key).and_then(|present| obj.get(&present))
}

/// Return the concrete key spelling present in `obj` for a camelCase `key`.
fn present_key(obj: &serde_json::Map<String, Json>, key: &str) -> Option<String> {
    if obj.contains_key(key) {
        return Some(key.to_string());
    }
    let kebab = camel_to_kebab(key);
    if obj.contains_key(&kebab) {
        return Some(kebab);
    }
    let snake = camel_to_snake(key);
    if obj.contains_key(&snake) {
        return Some(snake);
    }
    None
}

/// Multi-convention wrapper around [`super::optional_f64`].
fn optional_f64_any(
    obj: &serde_json::Map<String, Json>,
    key: &str,
    surface: &str,
) -> Result<Option<f64>> {
    match present_key(obj, key) {
        Some(present) => super::optional_f64(obj, &present, surface),
        None => Ok(None),
    }
}

/// Multi-convention wrapper around [`super::optional_u64`].
fn optional_u64_any(
    obj: &serde_json::Map<String, Json>,
    key: &str,
    surface: &str,
) -> Result<Option<u64>> {
    match present_key(obj, key) {
        Some(present) => super::optional_u64(obj, &present, surface),
        None => Ok(None),
    }
}

/// Multi-convention lookup of an optional list of strings (stop sequences).
fn optional_string_list_any(
    obj: &serde_json::Map<String, Json>,
    key: &str,
    surface: &str,
) -> Result<Option<Vec<String>>> {
    let Some(value) = get_first(obj, key) else {
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

/// Whether `key` (in any supported naming convention) is one of the modeled keys.
fn is_modeled_key(key: &str, modeled: &[&str]) -> bool {
    modeled.iter().any(|candidate| {
        key == *candidate || key == camel_to_kebab(candidate) || key == camel_to_snake(candidate)
    })
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Map an OCI finish reason string to normalized [`FinishReason`].
///
/// GENERIC responses use OpenAI-style lowercase reasons; COHERE responses use
/// UPPERCASE Cohere reasons.
fn map_oci_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "stop" | "COMPLETE" => FinishReason::Complete,
        "length" | "MAX_TOKENS" => FinishReason::Length,
        "tool_calls" => FinishReason::ToolUse,
        other => FinishReason::Unknown(other.to_string()),
    }
}

/// Helper to construct a [`Json`] number from an `f64`.
fn json_f64(v: f64) -> Json {
    serde_json::Number::from_f64(v)
        .map(Json::Number)
        .unwrap_or(Json::Null)
}

fn set_or_remove_json(obj: &mut serde_json::Map<String, Json>, key: &str, value: Option<Json>) {
    if let Some(value) = value {
        obj.insert(key.into(), value);
    } else {
        obj.remove(key);
    }
}

/// Insert `value` under whichever spelling of `key` the payload already uses,
/// falling back to camelCase for new keys.
fn insert_preserving_convention(obj: &mut serde_json::Map<String, Json>, key: &str, value: Json) {
    let target = present_key(obj, key).unwrap_or_else(|| key.to_string());
    obj.insert(target, value);
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
        if get_first(obj, "type").and_then(Json::as_str) != Some("TEXT") {
            return None;
        }
        match get_first(obj, "text") {
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
    match get_first(obj, "type").and_then(Json::as_str) {
        Some("TEXT") => Ok(ContentPart::Text {
            text: get_first(obj, "text")
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
    let function = get_first(obj, "function").and_then(Json::as_object);
    let (name, arguments) = match function {
        Some(function) => (
            get_first(function, "name"),
            get_first(function, "arguments"),
        ),
        None => (get_first(obj, "name"), get_first(obj, "arguments")),
    };
    Ok(ToolCall {
        id: get_first(obj, "id")
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

/// Convert a normalized nested [`ToolCall`] back into the flat OCI shape.
fn encode_oci_tool_call(tool_call: &ToolCall) -> Json {
    serde_json::json!({
        "id": tool_call.id,
        "type": "FUNCTION",
        "name": tool_call.function.name,
        "arguments": tool_call.function.arguments,
    })
}

// ---------------------------------------------------------------------------
// GENERIC message decode/encode
// ---------------------------------------------------------------------------

fn decode_generic_message(value: &Json) -> Result<Message> {
    let obj = value.as_object().ok_or_else(|| {
        FlowError::InvalidArgument("OCI GenAI GENERIC message must be an object".into())
    })?;
    let role = get_first(obj, "role")
        .and_then(Json::as_str)
        .unwrap_or("USER")
        .to_lowercase();
    let content = decode_generic_content(get_first(obj, "content"))?;
    let tool_calls = match get_first(obj, "toolCalls") {
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
    let tool_call_id = get_first(obj, "toolCallId")
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

fn encode_generic_message(message: &Message) -> Result<Json> {
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
            obj.insert(
                "content".into(),
                match content {
                    Some(content) => encode_generic_content(content)?,
                    None => Json::Null,
                },
            );
            if let Some(tool_calls) = tool_calls {
                obj.insert(
                    "toolCalls".into(),
                    Json::Array(tool_calls.iter().map(encode_oci_tool_call).collect()),
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
) -> Result<()> {
    let raw_messages: Vec<Json> = get_first(chat_request, "messages")
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
                _ => encode_generic_message(message),
            }
        })
        .collect::<Result<Vec<_>>>()?;
    insert_preserving_convention(chat_request, "messages", Json::Array(patched));
    Ok(())
}

// ---------------------------------------------------------------------------
// COHERE message decode/encode
// ---------------------------------------------------------------------------

fn decode_cohere_messages(chat_request: &serde_json::Map<String, Json>) -> Result<Vec<Message>> {
    let mut messages = Vec::new();

    if let Some(preamble) = get_first(chat_request, "preambleOverride").and_then(Json::as_str)
        && !preamble.is_empty()
    {
        messages.push(Message::System {
            content: MessageContent::Text(preamble.to_string()),
            name: None,
        });
    }

    if let Some(history) = get_first(chat_request, "chatHistory") {
        let turns = history.as_array().ok_or_else(|| {
            FlowError::InvalidArgument("OCI GenAI COHERE chatHistory must be an array".into())
        })?;
        for turn in turns {
            messages.push(decode_cohere_turn(turn)?);
        }
    }

    if let Some(current) = get_first(chat_request, "message").and_then(Json::as_str) {
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
    let role = get_first(obj, "role")
        .and_then(Json::as_str)
        .unwrap_or("USER")
        .to_uppercase();
    let Some(text) = get_first(obj, "message").and_then(Json::as_str) else {
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
        insert_preserving_convention(
            chat_request,
            "preambleOverride",
            Json::String(cohere_text(content)?),
        );
        remaining = &remaining[1..];
    }

    let mut current = String::new();
    if let Some(Message::User { content, .. }) = remaining.last() {
        current = cohere_text(content)?;
        remaining = &remaining[..remaining.len() - 1];
    }

    let history = remaining
        .iter()
        .map(encode_cohere_turn)
        .collect::<Result<Vec<_>>>()?;

    insert_preserving_convention(chat_request, "message", Json::String(current));
    if !history.is_empty() || present_key(chat_request, "chatHistory").is_some() {
        insert_preserving_convention(chat_request, "chatHistory", Json::Array(history));
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
        Message::Tool { content, .. } => ("TOOL", content),
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
    let temperature = optional_f64_any(chat_request, "temperature", SURFACE)?;
    let max_tokens = optional_u64_any(chat_request, "maxTokens", SURFACE)?;
    let top_p = optional_f64_any(chat_request, "topP", SURFACE)?;
    let stop_key = if api_format == "COHERE" {
        "stopSequences"
    } else {
        "stop"
    };
    let stop = optional_string_list_any(chat_request, stop_key, SURFACE)?;
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
/// Setting a param to `None` does not remove the raw key: the raw value keeps
/// serving as the source of truth for anything the annotation stops modeling.
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
    if temperature.is_some() && temperature != baseline.and_then(|params| params.temperature) {
        insert_preserving_convention(
            chat_request,
            "temperature",
            temperature.map(json_f64).unwrap_or(Json::Null),
        );
    }
    let top_p = edited.and_then(|params| params.top_p);
    if top_p.is_some() && top_p != baseline.and_then(|params| params.top_p) {
        insert_preserving_convention(
            chat_request,
            "topP",
            top_p.map(json_f64).unwrap_or(Json::Null),
        );
    }
    let max_tokens = edited.and_then(|params| params.max_tokens);
    if max_tokens.is_some() && max_tokens != baseline.and_then(|params| params.max_tokens) {
        insert_preserving_convention(
            chat_request,
            "maxTokens",
            max_tokens.map(Json::from).unwrap_or(Json::Null),
        );
    }
    let stop = edited.and_then(|params| params.stop.as_ref());
    if stop.is_some() && stop != baseline.and_then(|params| params.stop.as_ref()) {
        let stop_key = if api_format == "COHERE" {
            "stopSequences"
        } else {
            "stop"
        };
        insert_preserving_convention(
            chat_request,
            stop_key,
            stop.map(|values| serde_json::json!(values))
                .unwrap_or(Json::Null),
        );
    }
}

fn decode_tools(
    chat_request: &serde_json::Map<String, Json>,
) -> Result<Option<Vec<ToolDefinition>>> {
    match get_first(chat_request, "tools") {
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
    let serving_mode = get_first(envelope, "servingMode")?.as_object()?;
    get_first(serving_mode, "modelId")
        .or_else(|| get_first(serving_mode, "endpointId"))
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
    match get_first(obj, "chatRequest").and_then(Json::as_object) {
        Some(chat_request) => (Some(obj), chat_request),
        None => (None, obj),
    }
}

/// Resolve the request API format (uppercased), defaulting to `GENERIC`.
fn request_api_format(chat_request: &serde_json::Map<String, Json>) -> String {
    get_first(chat_request, "apiFormat")
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
                    ..
                }),
                Some(ApiSpecificRequest::OCIGenAI {
                    compartment_id: old_compartment_id,
                    serving_mode: old_serving_mode,
                    ..
                }),
            ) => (
                compartment_id,
                serving_mode,
                old_compartment_id,
                old_serving_mode,
            ),
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
        let key = present_key(envelope, "compartmentId").unwrap_or_else(|| "compartmentId".into());
        set_or_remove_json(envelope, &key, compartment_id.clone().map(Json::String));
    }
    if serving_mode != old_serving_mode {
        let key = present_key(envelope, "servingMode").unwrap_or_else(|| "servingMode".into());
        set_or_remove_json(envelope, &key, serving_mode.clone());
    }
    Ok(())
}

/// The API format the encoder should target: an edited api-specific annotation
/// wins over the raw payload, mirroring the decode default of `GENERIC`.
fn encode_api_format(
    annotated: &AnnotatedLlmRequest,
    chat_request: &serde_json::Map<String, Json>,
) -> String {
    if let Some(ApiSpecificRequest::OCIGenAI {
        api_format: Some(api_format),
        ..
    }) = &annotated.api_specific
    {
        return api_format.to_uppercase();
    }
    request_api_format(chat_request)
}

// ---------------------------------------------------------------------------
// LlmCodec implementation
// ---------------------------------------------------------------------------

impl LlmCodec for OCIGenAIChatCodec {
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
            match get_first(chat_request, "messages") {
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
        let tool_choice = get_first(chat_request, "toolChoice")
            .filter(|value| !value.is_null())
            .map(|value| ToolChoice::ProviderNative(native_component(value)));

        let modeled = if api_format == "COHERE" {
            MODELED_COHERE_REQUEST_KEYS
        } else {
            MODELED_GENERIC_REQUEST_KEYS
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
                    .and_then(|envelope| get_first(envelope, "compartmentId"))
                    .and_then(Json::as_str)
                    .map(str::to_string),
                serving_mode: envelope
                    .and_then(|envelope| get_first(envelope, "servingMode"))
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
        let chat_request_key =
            present_key(obj, "chatRequest").filter(|key| obj.get(key).is_some_and(Json::is_object));
        let mut chat_request = match &chat_request_key {
            Some(key) => obj
                .get(key)
                .and_then(Json::as_object)
                .cloned()
                .unwrap_or_default(),
            None => obj.clone(),
        };
        let api_format = encode_api_format(annotated, &chat_request);

        validate_oci_supported_fields(annotated, &baseline)?;

        if annotated.messages != baseline.messages {
            if api_format == "COHERE" {
                encode_cohere_messages(&mut chat_request, &annotated.messages)?;
            } else {
                patch_generic_messages(&mut chat_request, &annotated.messages, &baseline.messages)?;
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
            let key = present_key(&chat_request, "tools").unwrap_or_else(|| "tools".into());
            set_or_remove_json(&mut chat_request, &key, tools);
        }
        if annotated.tool_choice != baseline.tool_choice {
            let tool_choice = annotated
                .tool_choice
                .as_ref()
                .map(encode_oci_tool_choice)
                .transpose()?;
            let key =
                present_key(&chat_request, "toolChoice").unwrap_or_else(|| "toolChoice".into());
            set_or_remove_json(&mut chat_request, &key, tool_choice);
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

        let (envelope, chat_response) =
            match get_first(obj, "chatResponse").and_then(Json::as_object) {
                Some(chat_response) => (Some(obj), chat_response),
                None => (None, obj),
            };

        let model = envelope
            .and_then(|envelope| get_first(envelope, "modelId"))
            .and_then(Json::as_str)
            .map(str::to_string);
        let model_version = envelope
            .and_then(|envelope| get_first(envelope, "modelVersion"))
            .and_then(Json::as_str)
            .map(str::to_string);
        let api_format = get_first(chat_response, "apiFormat")
            .and_then(Json::as_str)
            .unwrap_or("GENERIC")
            .to_uppercase();

        let (message, tool_calls, finish_reason) = if api_format == "COHERE" {
            decode_cohere_response_body(chat_response)
        } else {
            decode_generic_response_body(chat_response)?
        };

        let usage = get_first(chat_response, "usage")
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
    let Some(first_choice) = get_first(chat_response, "choices")
        .and_then(Json::as_array)
        .and_then(|choices| choices.first())
        .and_then(Json::as_object)
    else {
        return Ok((None, None, None));
    };
    let finish_reason = get_first(first_choice, "finishReason")
        .and_then(Json::as_str)
        .map(str::to_string);
    let Some(raw_message) = get_first(first_choice, "message").and_then(Json::as_object) else {
        return Ok((None, None, finish_reason));
    };
    let message = decode_generic_content(get_first(raw_message, "content"))?;
    let tool_calls = get_first(raw_message, "toolCalls")
        .and_then(Json::as_array)
        .map(|calls| calls.iter().filter_map(decode_response_tool_call).collect())
        .filter(|calls: &Vec<ResponseToolCall>| !calls.is_empty());
    Ok((message, tool_calls, finish_reason))
}

fn decode_cohere_response_body(chat_response: &serde_json::Map<String, Json>) -> ResponseBody {
    let message = get_first(chat_response, "text")
        .and_then(Json::as_str)
        .map(|text| MessageContent::Text(text.to_string()));
    let tool_calls = get_first(chat_response, "toolCalls")
        .and_then(Json::as_array)
        .map(|calls| {
            calls
                .iter()
                .filter_map(decode_response_tool_call)
                .collect::<Vec<_>>()
        })
        .filter(|calls| !calls.is_empty());
    let finish_reason = get_first(chat_response, "finishReason")
        .and_then(Json::as_str)
        .map(str::to_string);
    (message, tool_calls, finish_reason)
}

/// Convert an OCI response tool call into [`ResponseToolCall`].
///
/// GENERIC calls are flat (`{id, type, name, arguments}`) with `arguments` as a
/// JSON-encoded string; COHERE calls carry `name` plus parsed `parameters`.
fn decode_response_tool_call(value: &Json) -> Option<ResponseToolCall> {
    let obj = value.as_object()?;
    let name = get_first(obj, "name")?.as_str()?.to_string();
    let arguments = match get_first(obj, "arguments") {
        Some(Json::String(text)) => {
            // CRITICAL: GENERIC arguments arrive JSON-encoded; parse for the
            // normalized shape, preserving the raw string when unparseable.
            serde_json::from_str::<Json>(text).unwrap_or_else(|_| Json::String(text.clone()))
        }
        Some(other) => other.clone(),
        None => get_first(obj, "parameters").cloned().unwrap_or(Json::Null),
    };
    Some(ResponseToolCall {
        id: get_first(obj, "id")
            .and_then(Json::as_str)
            .unwrap_or_default()
            .to_string(),
        name,
        arguments,
    })
}

/// Map OCI usage counters onto the normalized [`Usage`] field names.
fn decode_oci_usage(usage: &serde_json::Map<String, Json>) -> Usage {
    Usage {
        prompt_tokens: get_first(usage, "promptTokens").and_then(Json::as_u64),
        completion_tokens: get_first(usage, "completionTokens").and_then(Json::as_u64),
        total_tokens: get_first(usage, "totalTokens").and_then(Json::as_u64),
        cache_read_tokens: None,
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
    text: String,
    /// Tool calls keyed by their array position; `arguments` fragments
    /// accumulate, identity fields last-write-win.
    tool_calls: std::collections::BTreeMap<usize, OCIToolCallState>,
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
        let inner = get_first(obj, "chatResponse").and_then(Json::as_object);
        if let Some(model_id) = get_first(obj, "modelId").and_then(Json::as_str) {
            self.model_id = Some(model_id.to_string());
        }
        let obj = inner.unwrap_or(obj);

        if self.api_format.is_none() {
            self.api_format = get_first(obj, "apiFormat")
                .and_then(Json::as_str)
                .map(str::to_uppercase)
                .or_else(|| self.infer_api_format(obj));
        }
        if let Some(usage) = get_first(obj, "usage")
            && !usage.is_null()
        {
            self.usage = Some(usage.clone());
        }

        if self.api_format.as_deref() == Some("COHERE") {
            if let Some(text) = get_first(obj, "text").and_then(Json::as_str) {
                self.cohere_text.push_str(text);
            }
            if let Some(reason) = get_first(obj, "finishReason").and_then(Json::as_str) {
                self.cohere_finish_reason = Some(reason.to_string());
            }
            return;
        }

        // GENERIC: the event is either a bare choice delta or carries a
        // `choices` array of deltas.
        match get_first(obj, "choices").and_then(Json::as_array) {
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
        if get_first(obj, "message").is_some()
            || get_first(obj, "choices").is_some()
            || get_first(obj, "index").is_some()
        {
            Some("GENERIC".to_string())
        } else if get_first(obj, "text").is_some() {
            Some("COHERE".to_string())
        } else {
            None
        }
    }

    fn observe_generic_choice(&mut self, choice: &serde_json::Map<String, Json>) {
        let index = get_first(choice, "index")
            .and_then(Json::as_u64)
            .unwrap_or(0);
        let entry = self.choices.entry(index).or_default();
        if let Some(reason) = get_first(choice, "finishReason").and_then(Json::as_str) {
            entry.finish_reason = Some(reason.to_string());
        }
        let Some(message) = get_first(choice, "message").and_then(Json::as_object) else {
            return;
        };
        if let Some(role) = get_first(message, "role").and_then(Json::as_str) {
            entry.role = Some(role.to_string());
        }
        if let Some(parts) = get_first(message, "content").and_then(Json::as_array) {
            for part in parts {
                let Some(part) = part.as_object() else {
                    continue;
                };
                if get_first(part, "type").and_then(Json::as_str) == Some("TEXT")
                    && let Some(text) = get_first(part, "text").and_then(Json::as_str)
                {
                    entry.text.push_str(text);
                }
            }
        }
        if let Some(tool_calls) = get_first(message, "toolCalls").and_then(Json::as_array) {
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
    fn observe_tool_call(&mut self, position: usize, tool_call: &serde_json::Map<String, Json>) {
        let state = self.tool_calls.entry(position).or_default();
        if let Some(id) = get_first(tool_call, "id").and_then(Json::as_str) {
            state.id = Some(id.to_string());
        }
        if let Some(type_) = get_first(tool_call, "type").and_then(Json::as_str) {
            state.type_ = Some(type_.to_string());
        }
        if let Some(name) = get_first(tool_call, "name").and_then(Json::as_str) {
            state.name = Some(name.to_string());
        }
        if let Some(arguments) = get_first(tool_call, "arguments").and_then(Json::as_str) {
            state.arguments.push_str(arguments);
        }
    }

    fn finalize(self, index: u64) -> Json {
        let mut message = serde_json::Map::new();
        message.insert(
            "role".to_string(),
            Json::String(self.role.unwrap_or_else(|| "ASSISTANT".to_string())),
        );
        message.insert(
            "content".to_string(),
            serde_json::json!([{"type": "TEXT", "text": self.text}]),
        );
        if !self.tool_calls.is_empty() {
            let tool_calls: Vec<Json> = self
                .tool_calls
                .into_values()
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
