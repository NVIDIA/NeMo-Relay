// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Built-in codec for the OpenAI Chat Completions API.
//!
//! Implements [`LlmCodec`] (request decode/encode) and [`LlmResponseCodec`]
//! (response decode) for the OpenAI Chat Completions format.

use serde::Deserialize;

use crate::api::llm::LlmRequest;
use crate::api::runtime::{BuiltinLlmCodec, LlmCodecIdentity};
use crate::error::{FlowError, Result};
use crate::json::Json;

use super::request::{
    AnnotatedLlmRequest, ApiSpecificRequest, ContentPart, FunctionCall, FunctionDefinition,
    GenerationParams, Message, MessageContent, OpenAiImageUrl, ProviderNativeComponent, ToolCall,
    ToolChoice, ToolDefinition,
};
use super::resolve::{ProviderSurface, ProviderSurfaceDescriptor};
use super::response::{
    AnnotatedLlmResponse, ApiSpecificResponse, FinishReason, RawUsageCost, ResponseToolCall, Usage,
    estimate_cost_for_provider, infer_model_provider, provider_reported_cost,
};
use super::traits::{LlmCodec, LlmResponseCodec};

// ---------------------------------------------------------------------------
// Public codec struct
// ---------------------------------------------------------------------------

/// Built-in codec for the OpenAI Chat Completions API.
pub struct OpenAIChatCodec;

pub(crate) const PROVIDER_SURFACE: ProviderSurfaceDescriptor = ProviderSurfaceDescriptor {
    surface: ProviderSurface::OpenAIChat,
    detect_request: |obj, _| obj.contains_key("messages"),
    detect_response: |obj| obj.get("choices").is_some_and(Json::is_array),
    decode_request: |request| OpenAIChatCodec.decode(request),
    decode_response: |raw| OpenAIChatCodec.decode_response(raw),
    codec_name: "openai_chat",
    request_codec: || std::sync::Arc::new(OpenAIChatCodec),
    response_codec: || std::sync::Arc::new(OpenAIChatCodec),
    streaming_codec: || Box::new(OpenAIChatStreamingCodec::new()),
};

// ---------------------------------------------------------------------------
// Private intermediate serde structs for response decode
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct RawChatCompletion {
    id: Option<String>,
    model: Option<String>,
    choices: Option<Vec<RawChoice>>,
    usage: Option<RawChatUsage>,
    system_fingerprint: Option<String>,
    service_tier: Option<String>,
    #[serde(flatten)]
    extra: serde_json::Map<String, Json>,
}

#[derive(Deserialize)]
struct RawChoice {
    message: Option<RawMessage>,
    finish_reason: Option<String>,
    logprobs: Option<Json>,
}

#[derive(Deserialize)]
struct RawMessage {
    content: Option<String>,
    tool_calls: Option<Vec<RawToolCall>>,
}

#[derive(Deserialize)]
struct RawToolCall {
    id: Option<String>,
    function: Option<RawFunction>,
}

#[derive(Deserialize)]
struct RawFunction {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Deserialize)]
struct RawChatUsage {
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    total_tokens: Option<u64>,
    prompt_tokens_details: Option<RawPromptTokensDetails>,
    #[serde(rename = "cost_usd")]
    provider_cost: Option<f64>,
    cost: Option<RawUsageCost>,
}

#[derive(Deserialize)]
struct RawPromptTokensDetails {
    cached_tokens: Option<u64>,
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Map OpenAI Chat finish_reason string to normalized [`FinishReason`].
fn map_chat_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "stop" => FinishReason::Complete,
        "length" => FinishReason::Length,
        "tool_calls" | "function_call" => FinishReason::ToolUse,
        "content_filter" => FinishReason::ContentFilter,
        other => FinishReason::Unknown(other.to_string()),
    }
}

/// Parse OpenAI tool call arguments from JSON string to [`Json`] value.
///
/// Falls back to [`Json::String`] if parsing fails (malformed model output).
fn parse_arguments(arguments: &str) -> Json {
    serde_json::from_str(arguments).unwrap_or_else(|_| Json::String(arguments.to_string()))
}

/// Keys that are modeled in [`AnnotatedLlmRequest`] and should NOT go into `extra`.
const MODELED_REQUEST_KEYS: &[&str] = &[
    "messages",
    "model",
    "temperature",
    "max_tokens",
    "max_completion_tokens",
    "top_p",
    "stop",
    "tools",
    "tool_choice",
    "store",
    "user",
    "metadata",
    "service_tier",
    "parallel_tool_calls",
    "top_logprobs",
    "stream",
    "audio",
    "frequency_penalty",
    "function_call",
    "functions",
    "logit_bias",
    "logprobs",
    "modalities",
    "moderation",
    "n",
    "prediction",
    "presence_penalty",
    "prompt_cache_key",
    "prompt_cache_options",
    "prompt_cache_retention",
    "reasoning_effort",
    "response_format",
    "safety_identifier",
    "seed",
    "stream_options",
    "verbosity",
    "web_search_options",
];

fn chat_native(kind: &str, value: &Json) -> ProviderNativeComponent {
    ProviderNativeComponent {
        provider: "openai_chat".into(),
        kind: kind.to_string(),
        value: value.clone(),
    }
}

fn decode_chat_content(value: &Json) -> Result<MessageContent> {
    if let Some(text) = value.as_str() {
        return Ok(MessageContent::Text(text.to_string()));
    }
    let parts = value.as_array().ok_or_else(|| {
        FlowError::InvalidArgument("OpenAI Chat message content must be a string or array".into())
    })?;
    Ok(MessageContent::Parts(
        parts
            .iter()
            .map(decode_chat_content_part)
            .collect::<Result<Vec<_>>>()?,
    ))
}

fn decode_chat_content_part(value: &Json) -> Result<ContentPart> {
    let obj = value.as_object().ok_or_else(|| {
        FlowError::InvalidArgument("OpenAI Chat content part must be an object".into())
    })?;
    let kind = obj.get("type").and_then(Json::as_str).unwrap_or("unknown");
    match kind {
        "text" => Ok(ContentPart::Text {
            text: obj
                .get("text")
                .and_then(Json::as_str)
                .ok_or_else(|| {
                    FlowError::InvalidArgument("OpenAI Chat text part is missing text".into())
                })?
                .to_string(),
            extra: obj
                .iter()
                .filter(|(key, _)| !matches!(key.as_str(), "type" | "text"))
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        }),
        "image_url" => {
            let image_url: OpenAiImageUrl =
                serde_json::from_value(obj.get("image_url").cloned().ok_or_else(|| {
                    FlowError::InvalidArgument("OpenAI Chat image part is missing image_url".into())
                })?)
                .map_err(|error| {
                    FlowError::InvalidArgument(format!("invalid OpenAI Chat image_url: {error}"))
                })?;
            Ok(ContentPart::ImageUrl {
                image_url,
                extra: obj
                    .iter()
                    .filter(|(key, _)| !matches!(key.as_str(), "type" | "image_url"))
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect(),
            })
        }
        "input_audio" => Ok(ContentPart::Audio {
            audio: obj.get("input_audio").cloned().ok_or_else(|| {
                FlowError::InvalidArgument("OpenAI Chat audio part is missing input_audio".into())
            })?,
            extra: obj
                .iter()
                .filter(|(key, _)| !matches!(key.as_str(), "type" | "input_audio"))
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        }),
        "file" => Ok(ContentPart::File {
            file: obj.get("file").cloned().ok_or_else(|| {
                FlowError::InvalidArgument("OpenAI Chat file part is missing file".into())
            })?,
            extra: obj
                .iter()
                .filter(|(key, _)| !matches!(key.as_str(), "type" | "file"))
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        }),
        "refusal" => Ok(ContentPart::Refusal {
            refusal: obj
                .get("refusal")
                .and_then(Json::as_str)
                .ok_or_else(|| {
                    FlowError::InvalidArgument("OpenAI Chat refusal part is missing refusal".into())
                })?
                .to_string(),
            extra: obj
                .iter()
                .filter(|(key, _)| !matches!(key.as_str(), "type" | "refusal"))
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        }),
        _ => Ok(ContentPart::ProviderNative {
            provider: "openai_chat".into(),
            kind: kind.to_string(),
            value: value.clone(),
        }),
    }
}

fn optional_chat_string(
    obj: &serde_json::Map<String, Json>,
    key: &str,
    context: &str,
) -> Result<Option<String>> {
    match obj.get(key) {
        Some(Json::Null) | None => Ok(None),
        Some(Json::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(FlowError::InvalidArgument(format!(
            "OpenAI Chat {context} {key} must be a string or null"
        ))),
    }
}

fn decode_chat_tool_call(value: &Json) -> Result<Option<ToolCall>> {
    let obj = value.as_object().ok_or_else(|| {
        FlowError::InvalidArgument("OpenAI Chat tool call must be an object".into())
    })?;
    if obj.get("type").and_then(Json::as_str) != Some("function") {
        return Ok(None);
    }
    if obj
        .keys()
        .any(|key| !matches!(key.as_str(), "id" | "type" | "function"))
    {
        return Ok(None);
    }
    let function = obj
        .get("function")
        .and_then(Json::as_object)
        .ok_or_else(|| {
            FlowError::InvalidArgument("OpenAI Chat function tool call is missing function".into())
        })?;
    if function
        .keys()
        .any(|key| !matches!(key.as_str(), "name" | "arguments"))
    {
        return Ok(None);
    }
    let id = obj.get("id").and_then(Json::as_str).ok_or_else(|| {
        FlowError::InvalidArgument("OpenAI Chat function tool call is missing id".into())
    })?;
    let name = function.get("name").and_then(Json::as_str).ok_or_else(|| {
        FlowError::InvalidArgument("OpenAI Chat function tool call is missing name".into())
    })?;
    let arguments = function
        .get("arguments")
        .and_then(Json::as_str)
        .ok_or_else(|| {
            FlowError::InvalidArgument("OpenAI Chat function tool call is missing arguments".into())
        })?;
    Ok(Some(ToolCall {
        id: id.to_string(),
        call_type: "function".into(),
        function: FunctionCall {
            name: name.to_string(),
            arguments: arguments.to_string(),
        },
    }))
}

fn decode_chat_message(value: &Json) -> Result<Message> {
    let obj = value.as_object().ok_or_else(|| {
        FlowError::InvalidArgument("OpenAI Chat message must be an object".into())
    })?;
    let role = obj
        .get("role")
        .and_then(Json::as_str)
        .ok_or_else(|| FlowError::InvalidArgument("OpenAI Chat message is missing role".into()))?;
    let native = || Message::ProviderNative {
        provider: "openai_chat".into(),
        kind: role.to_string(),
        value: value.clone(),
    };
    match role {
        "system" | "developer" | "user" => {
            if obj
                .keys()
                .any(|key| !matches!(key.as_str(), "role" | "content" | "name"))
            {
                return Ok(native());
            }
            let content = decode_chat_content(obj.get("content").ok_or_else(|| {
                FlowError::InvalidArgument("OpenAI Chat message is missing content".into())
            })?)?;
            let name = optional_chat_string(obj, "name", "message")?;
            Ok(match role {
                "system" => Message::System { content, name },
                "developer" => Message::Developer { content, name },
                _ => Message::User { content, name },
            })
        }
        "assistant" => {
            if obj
                .keys()
                .any(|key| !matches!(key.as_str(), "role" | "content" | "tool_calls" | "name"))
            {
                return Ok(native());
            }
            let content = obj
                .get("content")
                .filter(|content| !content.is_null())
                .map(decode_chat_content)
                .transpose()?;
            let tool_calls = match obj.get("tool_calls") {
                Some(Json::Null) | None => None,
                Some(Json::Array(calls)) => {
                    let decoded = calls
                        .iter()
                        .map(decode_chat_tool_call)
                        .collect::<Result<Vec<_>>>()?;
                    let Some(decoded) = decoded.into_iter().collect::<Option<Vec<_>>>() else {
                        return Ok(native());
                    };
                    Some(decoded)
                }
                Some(_) => {
                    return Err(FlowError::InvalidArgument(
                        "OpenAI Chat assistant tool_calls must be an array or null".into(),
                    ));
                }
            };
            Ok(Message::Assistant {
                content,
                tool_calls,
                name: optional_chat_string(obj, "name", "assistant message")?,
            })
        }
        "tool"
            if obj
                .keys()
                .all(|key| matches!(key.as_str(), "role" | "content" | "tool_call_id")) =>
        {
            Ok(Message::Tool {
                content: decode_chat_content(obj.get("content").ok_or_else(|| {
                    FlowError::InvalidArgument("OpenAI Chat tool message is missing content".into())
                })?)?,
                tool_call_id: obj
                    .get("tool_call_id")
                    .and_then(Json::as_str)
                    .ok_or_else(|| {
                        FlowError::InvalidArgument(
                            "OpenAI Chat tool message is missing tool_call_id".into(),
                        )
                    })?
                    .to_string(),
            })
        }
        "function"
            if obj
                .keys()
                .all(|key| matches!(key.as_str(), "role" | "content" | "name")) =>
        {
            Ok(Message::Function {
                content: optional_chat_string(obj, "content", "function message")?,
                name: obj
                    .get("name")
                    .and_then(Json::as_str)
                    .ok_or_else(|| {
                        FlowError::InvalidArgument(
                            "OpenAI Chat function message is missing name".into(),
                        )
                    })?
                    .to_string(),
            })
        }
        _ => Ok(native()),
    }
}

fn encode_chat_content(content: &MessageContent) -> Result<Json> {
    match content {
        MessageContent::Text(text) => Ok(Json::String(text.clone())),
        MessageContent::Parts(parts) => Ok(Json::Array(
            parts
                .iter()
                .map(|part| match part {
                    ContentPart::Text { text, extra } => {
                        let mut obj = extra.clone();
                        obj.insert("type".into(), Json::String("text".into()));
                        obj.insert("text".into(), Json::String(text.clone()));
                        Ok(Json::Object(obj))
                    }
                    ContentPart::ImageUrl { image_url, extra } => {
                        let mut obj = extra.clone();
                        obj.insert("type".into(), Json::String("image_url".into()));
                        obj.insert(
                            "image_url".into(),
                            serde_json::to_value(image_url).map_err(|error| {
                                FlowError::Internal(format!(
                                    "OpenAI Chat image URL encode: {error}"
                                ))
                            })?,
                        );
                        Ok(Json::Object(obj))
                    }
                    ContentPart::Audio { audio, extra } => {
                        let mut obj = extra.clone();
                        obj.insert("type".into(), Json::String("input_audio".into()));
                        obj.insert("input_audio".into(), audio.clone());
                        Ok(Json::Object(obj))
                    }
                    ContentPart::File { file, extra } => {
                        let mut obj = extra.clone();
                        obj.insert("type".into(), Json::String("file".into()));
                        obj.insert("file".into(), file.clone());
                        Ok(Json::Object(obj))
                    }
                    ContentPart::Refusal { refusal, extra } => {
                        let mut obj = extra.clone();
                        obj.insert("type".into(), Json::String("refusal".into()));
                        obj.insert("refusal".into(), Json::String(refusal.clone()));
                        Ok(Json::Object(obj))
                    }
                    ContentPart::ProviderNative {
                        provider, value, ..
                    } if provider == "openai_chat" => Ok(value.clone()),
                    other => Err(FlowError::InvalidArgument(format!(
                        "content part {other:?} cannot be encoded for OpenAI Chat"
                    ))),
                })
                .collect::<Result<Vec<_>>>()?,
        )),
    }
}

fn encode_chat_message(message: &Message) -> Result<Json> {
    let message_with_content =
        |role: &str, content: &MessageContent, name: &Option<String>| -> Result<Json> {
            let mut obj = serde_json::Map::new();
            obj.insert("role".into(), Json::String(role.into()));
            obj.insert("content".into(), encode_chat_content(content)?);
            if let Some(name) = name {
                obj.insert("name".into(), Json::String(name.clone()));
            }
            Ok(Json::Object(obj))
        };
    match message {
        Message::System { content, name } => message_with_content("system", content, name),
        Message::Developer { content, name } => message_with_content("developer", content, name),
        Message::User { content, name } => message_with_content("user", content, name),
        Message::Assistant {
            content,
            tool_calls,
            name,
        } => {
            let mut obj = serde_json::Map::new();
            obj.insert("role".into(), Json::String("assistant".into()));
            if let Some(content) = content {
                obj.insert("content".into(), encode_chat_content(content)?);
            }
            if let Some(tool_calls) = tool_calls {
                obj.insert(
                    "tool_calls".into(),
                    serde_json::to_value(tool_calls).map_err(|error| {
                        FlowError::Internal(format!("OpenAI Chat tool calls encode: {error}"))
                    })?,
                );
            }
            if let Some(name) = name {
                obj.insert("name".into(), Json::String(name.clone()));
            }
            Ok(Json::Object(obj))
        }
        Message::Tool {
            content,
            tool_call_id,
        } => {
            let mut obj = serde_json::Map::new();
            obj.insert("role".into(), Json::String("tool".into()));
            obj.insert("content".into(), encode_chat_content(content)?);
            obj.insert("tool_call_id".into(), Json::String(tool_call_id.clone()));
            Ok(Json::Object(obj))
        }
        Message::Function { content, name } => {
            let mut obj = serde_json::Map::new();
            obj.insert("role".into(), Json::String("function".into()));
            obj.insert(
                "content".into(),
                content.clone().map(Json::String).unwrap_or(Json::Null),
            );
            obj.insert("name".into(), Json::String(name.clone()));
            Ok(Json::Object(obj))
        }
        Message::ProviderNative {
            provider, value, ..
        } if provider == "openai_chat" => Ok(value.clone()),
        other => Err(FlowError::InvalidArgument(format!(
            "message {other:?} cannot be encoded for OpenAI Chat"
        ))),
    }
}

fn decode_chat_tool(value: &Json) -> Result<ToolDefinition> {
    let obj = value
        .as_object()
        .ok_or_else(|| FlowError::InvalidArgument("OpenAI Chat tool must be an object".into()))?;
    if obj.get("type").and_then(Json::as_str) != Some("function") {
        let native = chat_native(
            obj.get("type").and_then(Json::as_str).unwrap_or("unknown"),
            value,
        );
        return Ok(ToolDefinition::ProviderNative {
            provider: native.provider,
            kind: native.kind,
            value: native.value,
        });
    }
    let function = obj
        .get("function")
        .and_then(Json::as_object)
        .ok_or_else(|| {
            FlowError::InvalidArgument("OpenAI Chat function tool is missing function".into())
        })?;
    let name = function.get("name").and_then(Json::as_str).ok_or_else(|| {
        FlowError::InvalidArgument("OpenAI Chat function tool is missing name".into())
    })?;
    let description = super::optional_string(function, "description", "OpenAI Chat function tool")?;
    let strict = super::optional_bool(function, "strict", "OpenAI Chat function tool")?;
    Ok(ToolDefinition::Function {
        function: FunctionDefinition {
            name: name.to_string(),
            description,
            parameters: function.get("parameters").cloned(),
            strict,
            extra: function
                .iter()
                .filter(|(key, _)| {
                    !matches!(
                        key.as_str(),
                        "name" | "description" | "parameters" | "strict"
                    )
                })
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        },
        extra: obj
            .iter()
            .filter(|(key, _)| !matches!(key.as_str(), "type" | "function"))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
    })
}

fn encode_chat_tool(tool: &ToolDefinition) -> Result<Json> {
    match tool {
        ToolDefinition::Function { function, extra } => {
            let mut function_obj = function.extra.clone();
            function_obj.insert("name".into(), Json::String(function.name.clone()));
            if let Some(description) = &function.description {
                function_obj.insert("description".into(), Json::String(description.clone()));
            }
            if let Some(parameters) = &function.parameters {
                function_obj.insert("parameters".into(), parameters.clone());
            }
            if let Some(strict) = function.strict {
                function_obj.insert("strict".into(), Json::Bool(strict));
            }
            let mut obj = extra.clone();
            obj.insert("type".into(), Json::String("function".into()));
            obj.insert("function".into(), Json::Object(function_obj));
            Ok(Json::Object(obj))
        }
        ToolDefinition::ProviderNative {
            provider, value, ..
        } if provider == "openai_chat" => Ok(value.clone()),
        other => Err(FlowError::InvalidArgument(format!(
            "tool {other:?} cannot be encoded for OpenAI Chat"
        ))),
    }
}

fn decode_chat_tool_choice(value: &Json) -> ToolChoice {
    match value.as_str() {
        Some("auto") => ToolChoice::Auto,
        Some("none") => ToolChoice::None,
        Some("required") => ToolChoice::Required,
        _ => {
            if let Some(function) = value
                .as_object()
                .filter(|obj| obj.get("type").and_then(Json::as_str) == Some("function"))
                .and_then(|obj| obj.get("function"))
                .and_then(Json::as_object)
                .and_then(|function| function.get("name"))
                .and_then(Json::as_str)
            {
                ToolChoice::Specific(super::request::ToolChoiceFunction {
                    choice_type: "function".into(),
                    function: super::request::ToolChoiceFunctionName {
                        name: function.to_string(),
                    },
                })
            } else {
                ToolChoice::ProviderNative(chat_native("tool_choice", value))
            }
        }
    }
}

fn encode_chat_tool_choice(choice: &ToolChoice) -> Result<Json> {
    match choice {
        ToolChoice::Auto => Ok(Json::String("auto".into())),
        ToolChoice::None => Ok(Json::String("none".into())),
        ToolChoice::Required => Ok(Json::String("required".into())),
        ToolChoice::Specific(choice) => Ok(serde_json::json!({
            "type":"function",
            "function":{"name":choice.function.name}
        })),
        ToolChoice::ProviderNative(native) if native.provider == "openai_chat" => {
            Ok(native.value.clone())
        }
        ToolChoice::ProviderNative(native) => Err(FlowError::InvalidArgument(format!(
            "tool choice for {} cannot be encoded for OpenAI Chat",
            native.provider
        ))),
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

fn set_or_remove_json(obj: &mut serde_json::Map<String, Json>, key: &str, value: Option<Json>) {
    if let Some(value) = value {
        obj.insert(key.into(), value);
    } else {
        obj.remove(key);
    }
}

fn patch_chat_messages_and_validate(
    obj: &mut serde_json::Map<String, Json>,
    annotated: &AnnotatedLlmRequest,
    baseline: &AnnotatedLlmRequest,
) -> Result<()> {
    if annotated.messages != baseline.messages {
        let original_messages = obj.get("messages").and_then(Json::as_array);
        obj.insert(
            "messages".into(),
            Json::Array(super::encode_changed_items(
                &annotated.messages,
                &baseline.messages,
                original_messages.map(Vec::as_slice),
                encode_chat_message,
            )?),
        );
    }
    let unsupported = [
        annotated.instructions != baseline.instructions,
        annotated.previous_response_id != baseline.previous_response_id,
        annotated.truncation != baseline.truncation,
        annotated.reasoning != baseline.reasoning,
        annotated.include != baseline.include,
        annotated.max_output_tokens != baseline.max_output_tokens,
        annotated.max_tool_calls != baseline.max_tool_calls,
    ]
    .into_iter()
    .any(|changed| changed);
    if unsupported {
        return Err(FlowError::InvalidArgument(
            "request contains fields that cannot be encoded for OpenAI Chat".into(),
        ));
    }
    Ok(())
}

fn patch_chat_model_and_params(
    obj: &mut serde_json::Map<String, Json>,
    annotated: &AnnotatedLlmRequest,
    baseline: &AnnotatedLlmRequest,
) {
    if annotated.model != baseline.model {
        set_or_remove_json(obj, "model", annotated.model.clone().map(Json::String));
    }
    if annotated.params == baseline.params {
        return;
    }
    let edited = annotated.params.as_ref();
    let before = baseline.params.as_ref();
    patch_chat_optional_params(obj, edited, before);
    patch_chat_max_tokens(obj, edited, before);
    patch_chat_stop_sequences(obj, edited, before);
}

fn patch_chat_optional_params(
    obj: &mut serde_json::Map<String, Json>,
    edited: Option<&GenerationParams>,
    before: Option<&GenerationParams>,
) {
    for (key, value, old_value) in [
        (
            "temperature",
            edited.and_then(|params| params.temperature),
            before.and_then(|params| params.temperature),
        ),
        (
            "top_p",
            edited.and_then(|params| params.top_p),
            before.and_then(|params| params.top_p),
        ),
    ] {
        if value != old_value {
            set_or_remove_json(obj, key, value.map(json_f64));
        }
    }
}

fn patch_chat_max_tokens(
    obj: &mut serde_json::Map<String, Json>,
    edited: Option<&GenerationParams>,
    before: Option<&GenerationParams>,
) {
    let max_tokens = edited.and_then(|params| params.max_tokens);
    if max_tokens == before.and_then(|params| params.max_tokens) {
        return;
    }
    let max_tokens = max_tokens.map(Json::from);
    if max_tokens.is_none() {
        obj.remove("max_completion_tokens");
        obj.remove("max_tokens");
        return;
    }
    let key = if obj.contains_key("max_completion_tokens") || !obj.contains_key("max_tokens") {
        "max_completion_tokens"
    } else {
        "max_tokens"
    };
    set_or_remove_json(obj, key, max_tokens);
}

fn patch_chat_stop_sequences(
    obj: &mut serde_json::Map<String, Json>,
    edited: Option<&GenerationParams>,
    before: Option<&GenerationParams>,
) {
    let stop = edited.and_then(|params| params.stop.as_ref());
    if stop == before.and_then(|params| params.stop.as_ref()) {
        return;
    }
    let stop = stop.map(|values| {
        if obj.get("stop").is_some_and(Json::is_string) && values.len() == 1 {
            Json::String(values[0].clone())
        } else {
            serde_json::json!(values)
        }
    });
    set_or_remove_json(obj, "stop", stop);
}

fn patch_chat_tools(
    obj: &mut serde_json::Map<String, Json>,
    annotated: &AnnotatedLlmRequest,
    baseline: &AnnotatedLlmRequest,
) -> Result<()> {
    if annotated.tools != baseline.tools {
        let tools = annotated
            .tools
            .as_deref()
            .map(|tools| {
                super::encode_changed_items(
                    tools,
                    baseline.tools.as_deref().unwrap_or(&[]),
                    obj.get("tools").and_then(Json::as_array).map(Vec::as_slice),
                    encode_chat_tool,
                )
            })
            .transpose()?
            .map(Json::Array);
        set_or_remove_json(obj, "tools", tools);
    }
    if annotated.tool_choice != baseline.tool_choice {
        let tool_choice = match (&annotated.tool_choice, &baseline.tool_choice) {
            (Some(edited), Some(before)) => {
                let edited = encode_chat_tool_choice(edited)?;
                let before = encode_chat_tool_choice(before)?;
                Some(match obj.get("tool_choice") {
                    Some(original) => super::patch_changed_json(original, &before, &edited)?,
                    None => edited,
                })
            }
            (Some(edited), None) => Some(encode_chat_tool_choice(edited)?),
            (None, _) => None,
        };
        set_or_remove_json(obj, "tool_choice", tool_choice);
    }
    Ok(())
}

fn patch_chat_common_fields(
    obj: &mut serde_json::Map<String, Json>,
    annotated: &AnnotatedLlmRequest,
    baseline: &AnnotatedLlmRequest,
) {
    if annotated.metadata != baseline.metadata {
        set_or_remove_json(obj, "metadata", annotated.metadata.clone());
    }
    for (key, edited, before) in [
        ("store", annotated.store, baseline.store),
        (
            "parallel_tool_calls",
            annotated.parallel_tool_calls,
            baseline.parallel_tool_calls,
        ),
        ("stream", annotated.stream, baseline.stream),
    ] {
        if edited != before {
            set_or_remove_json(obj, key, edited.map(Json::Bool));
        }
    }
    for (key, edited, before) in [
        ("user", &annotated.user, &baseline.user),
        (
            "service_tier",
            &annotated.service_tier,
            &baseline.service_tier,
        ),
    ] {
        if edited != before {
            set_or_remove_json(obj, key, edited.clone().map(Json::String));
        }
    }
    if annotated.top_logprobs != baseline.top_logprobs {
        set_or_remove_json(obj, "top_logprobs", annotated.top_logprobs.map(Json::from));
    }
}

fn patch_optional_json_fields(
    obj: &mut serde_json::Map<String, Json>,
    fields: &[(&str, &Option<Json>, &Option<Json>)],
) {
    for (key, edited, before) in fields {
        if edited != before {
            set_or_remove_json(obj, key, (*edited).clone());
        }
    }
}

fn patch_optional_string_fields(
    obj: &mut serde_json::Map<String, Json>,
    fields: &[(&str, &Option<String>, &Option<String>)],
) {
    for (key, edited, before) in fields {
        if edited != before {
            set_or_remove_json(obj, key, (*edited).clone().map(Json::String));
        }
    }
}

fn patch_optional_f64_fields(
    obj: &mut serde_json::Map<String, Json>,
    fields: &[(&str, &Option<f64>, &Option<f64>)],
) {
    for (key, edited, before) in fields {
        if edited != before {
            set_or_remove_json(obj, key, (*edited).map(json_f64));
        }
    }
}

fn patch_chat_api_specific(
    obj: &mut serde_json::Map<String, Json>,
    edited: &Option<ApiSpecificRequest>,
    baseline: &Option<ApiSpecificRequest>,
) -> Result<()> {
    match (edited, baseline) {
        (
            Some(edited @ ApiSpecificRequest::OpenAIChat { .. }),
            Some(baseline @ ApiSpecificRequest::OpenAIChat { .. }),
        ) => {
            patch_chat_api_fields(obj, edited, baseline);
            Ok(())
        }
        (None, Some(ApiSpecificRequest::OpenAIChat { .. })) => {
            for key in [
                "audio",
                "frequency_penalty",
                "function_call",
                "functions",
                "logit_bias",
                "logprobs",
                "modalities",
                "moderation",
                "n",
                "prediction",
                "presence_penalty",
                "prompt_cache_key",
                "prompt_cache_options",
                "prompt_cache_retention",
                "reasoning_effort",
                "response_format",
                "safety_identifier",
                "seed",
                "stream_options",
                "verbosity",
                "web_search_options",
            ] {
                obj.remove(key);
            }
            Ok(())
        }
        (Some(_), _) | (None, Some(_)) => Err(FlowError::InvalidArgument(
            "api_specific provider does not match OpenAI Chat".into(),
        )),
        (None, None) => Ok(()),
    }
}

fn patch_chat_api_fields(
    obj: &mut serde_json::Map<String, Json>,
    edited: &ApiSpecificRequest,
    baseline: &ApiSpecificRequest,
) {
    let (
        ApiSpecificRequest::OpenAIChat {
            audio,
            frequency_penalty,
            function_call,
            functions,
            logit_bias,
            logprobs,
            modalities,
            moderation,
            n,
            prediction,
            presence_penalty,
            prompt_cache_key,
            prompt_cache_options,
            prompt_cache_retention,
            reasoning_effort,
            response_format,
            safety_identifier,
            seed,
            stream_options,
            verbosity,
            web_search_options,
        },
        ApiSpecificRequest::OpenAIChat {
            audio: old_audio,
            frequency_penalty: old_frequency_penalty,
            function_call: old_function_call,
            functions: old_functions,
            logit_bias: old_logit_bias,
            logprobs: old_logprobs,
            modalities: old_modalities,
            moderation: old_moderation,
            n: old_n,
            prediction: old_prediction,
            presence_penalty: old_presence_penalty,
            prompt_cache_key: old_prompt_cache_key,
            prompt_cache_options: old_prompt_cache_options,
            prompt_cache_retention: old_prompt_cache_retention,
            reasoning_effort: old_reasoning_effort,
            response_format: old_response_format,
            safety_identifier: old_safety_identifier,
            seed: old_seed,
            stream_options: old_stream_options,
            verbosity: old_verbosity,
            web_search_options: old_web_search_options,
        },
    ) = (edited, baseline)
    else {
        unreachable!("OpenAI Chat variants checked by caller");
    };
    patch_optional_json_fields(
        obj,
        &[
            ("audio", audio, old_audio),
            ("function_call", function_call, old_function_call),
            ("logit_bias", logit_bias, old_logit_bias),
            ("moderation", moderation, old_moderation),
            ("prediction", prediction, old_prediction),
            (
                "prompt_cache_options",
                prompt_cache_options,
                old_prompt_cache_options,
            ),
            ("response_format", response_format, old_response_format),
            ("stream_options", stream_options, old_stream_options),
            (
                "web_search_options",
                web_search_options,
                old_web_search_options,
            ),
        ],
    );
    patch_optional_f64_fields(
        obj,
        &[
            (
                "frequency_penalty",
                frequency_penalty,
                old_frequency_penalty,
            ),
            ("presence_penalty", presence_penalty, old_presence_penalty),
        ],
    );
    patch_optional_string_fields(
        obj,
        &[
            ("prompt_cache_key", prompt_cache_key, old_prompt_cache_key),
            (
                "prompt_cache_retention",
                prompt_cache_retention,
                old_prompt_cache_retention,
            ),
            ("reasoning_effort", reasoning_effort, old_reasoning_effort),
            (
                "safety_identifier",
                safety_identifier,
                old_safety_identifier,
            ),
            ("verbosity", verbosity, old_verbosity),
        ],
    );
    if functions != old_functions {
        set_or_remove_json(obj, "functions", functions.clone().map(Json::Array));
    }
    if modalities != old_modalities {
        set_or_remove_json(
            obj,
            "modalities",
            modalities.as_ref().map(|value| serde_json::json!(value)),
        );
    }
    if logprobs != old_logprobs {
        set_or_remove_json(obj, "logprobs", logprobs.map(Json::Bool));
    }
    if n != old_n {
        set_or_remove_json(obj, "n", n.map(Json::from));
    }
    if seed != old_seed {
        set_or_remove_json(obj, "seed", seed.map(Json::from));
    }
}

// ---------------------------------------------------------------------------
// LlmResponseCodec implementation
// ---------------------------------------------------------------------------

impl LlmResponseCodec for OpenAIChatCodec {
    fn codec_identity(&self) -> LlmCodecIdentity {
        LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OpenAiChat)
    }

    fn decode_response(&self, response: &Json) -> Result<AnnotatedLlmResponse> {
        let raw: RawChatCompletion = serde_json::from_value(response.clone())
            .map_err(|e| FlowError::Internal(format!("OpenAI Chat response decode: {e}")))?;

        // Extract first choice (if any).
        let choice = raw.choices.as_ref().and_then(|c| c.first());

        // Map message content.
        let message = choice
            .and_then(|c| c.message.as_ref())
            .and_then(|m| m.content.as_ref())
            .map(|s| super::request::MessageContent::Text(s.clone()));

        // Map tool calls, skipping entries that lack a usable function body.
        // Some providers (proxies, vLLM, NIM) may return partial tool_calls
        // entries where `function` or `function.name` is absent or null.
        let tool_calls = choice
            .and_then(|c| c.message.as_ref())
            .and_then(|m| m.tool_calls.as_ref())
            .map(|tcs| {
                tcs.iter()
                    .filter_map(|tc| {
                        let func = tc.function.as_ref()?;
                        let name = func.name.as_ref()?;
                        Some(ResponseToolCall {
                            id: tc.id.clone().unwrap_or_default(),
                            name: name.clone(),
                            arguments: func
                                .arguments
                                .as_deref()
                                .map(parse_arguments)
                                .unwrap_or(Json::Object(Default::default())),
                        })
                    })
                    .collect::<Vec<_>>()
            });

        // Map finish reason.
        let finish_reason = choice
            .and_then(|c| c.finish_reason.as_deref())
            .map(map_chat_finish_reason);

        // Map usage.
        let model_for_pricing = raw.model.as_deref();
        let model_provider = infer_model_provider("openai", model_for_pricing);
        let usage = raw.usage.map(|u| {
            let mut usage = Usage {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
                cache_read_tokens: u.prompt_tokens_details.and_then(|d| d.cached_tokens),
                cache_write_tokens: None,
                cost: provider_reported_cost(u.provider_cost, u.cost),
            };
            if usage.cost.is_none() {
                usage.cost = model_for_pricing.and_then(|model| {
                    estimate_cost_for_provider(model_provider.as_deref(), model, &usage)
                });
            }
            usage
        });

        // Build API-specific fields.
        let logprobs = choice.and_then(|c| c.logprobs.clone());
        let api_specific = Some(ApiSpecificResponse::OpenAIChat {
            logprobs,
            system_fingerprint: raw.system_fingerprint,
            service_tier: raw.service_tier,
        });

        Ok(AnnotatedLlmResponse {
            id: raw.id,
            model: raw.model,
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
// LlmCodec implementation
// ---------------------------------------------------------------------------

impl LlmCodec for OpenAIChatCodec {
    fn codec_identity(&self) -> LlmCodecIdentity {
        LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OpenAiChat)
    }

    fn decode(&self, request: &LlmRequest) -> Result<AnnotatedLlmRequest> {
        let obj = request
            .content
            .as_object()
            .ok_or_else(|| FlowError::Internal("request content is not an object".into()))?;
        let messages = obj
            .get("messages")
            .ok_or_else(|| {
                FlowError::InvalidArgument("OpenAI Chat request is missing messages".into())
            })?
            .as_array()
            .ok_or_else(|| {
                FlowError::InvalidArgument("OpenAI Chat messages must be an array".into())
            })?
            .iter()
            .map(decode_chat_message)
            .collect::<Result<Vec<_>>>()?;
        let model = super::optional_string(obj, "model", "OpenAI Chat")?;
        let temperature = super::optional_f64(obj, "temperature", "OpenAI Chat")?;
        let top_p = super::optional_f64(obj, "top_p", "OpenAI Chat")?;
        let stop = match obj.get("stop") {
            Some(Json::String(stop)) => Some(vec![stop.clone()]),
            Some(Json::Array(_)) => Some(
                serde_json::from_value::<Vec<String>>(obj["stop"].clone()).map_err(|error| {
                    FlowError::InvalidArgument(format!("invalid OpenAI Chat stop value: {error}"))
                })?,
            ),
            Some(Json::Null) | None => None,
            Some(_) => {
                return Err(FlowError::InvalidArgument(
                    "OpenAI Chat stop must be a string, array, or null".into(),
                ));
            }
        };
        let max_tokens = super::optional_u64(obj, "max_completion_tokens", "OpenAI Chat")?
            .or(super::optional_u64(obj, "max_tokens", "OpenAI Chat")?);
        let params =
            if temperature.is_some() || max_tokens.is_some() || top_p.is_some() || stop.is_some() {
                Some(GenerationParams {
                    temperature,
                    max_tokens,
                    top_p,
                    stop,
                })
            } else {
                None
            };
        let tools = obj
            .get("tools")
            .map(|value| {
                value
                    .as_array()
                    .ok_or_else(|| {
                        FlowError::InvalidArgument("OpenAI Chat tools must be an array".into())
                    })?
                    .iter()
                    .map(decode_chat_tool)
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        let tool_choice = obj.get("tool_choice").map(decode_chat_tool_choice);
        let store = super::optional_bool(obj, "store", "OpenAI Chat")?;
        let user = super::optional_string(obj, "user", "OpenAI Chat")?;
        let service_tier = super::optional_string(obj, "service_tier", "OpenAI Chat")?;
        let parallel_tool_calls = super::optional_bool(obj, "parallel_tool_calls", "OpenAI Chat")?;
        let top_logprobs = super::optional_u64(obj, "top_logprobs", "OpenAI Chat")?;
        let stream = super::optional_bool(obj, "stream", "OpenAI Chat")?;
        let frequency_penalty = super::optional_f64(obj, "frequency_penalty", "OpenAI Chat")?;
        let functions = match obj.get("functions") {
            Some(Json::Null) | None => None,
            Some(Json::Array(functions)) => Some(functions.clone()),
            Some(_) => {
                return Err(FlowError::InvalidArgument(
                    "OpenAI Chat functions must be an array or null".into(),
                ));
            }
        };
        let logprobs = super::optional_bool(obj, "logprobs", "OpenAI Chat")?;
        let modalities = match obj.get("modalities") {
            Some(Json::Null) | None => None,
            Some(value) => Some(serde_json::from_value(value.clone()).map_err(|error| {
                FlowError::InvalidArgument(format!("invalid OpenAI Chat modalities: {error}"))
            })?),
        };
        let n = super::optional_u64(obj, "n", "OpenAI Chat")?;
        let presence_penalty = super::optional_f64(obj, "presence_penalty", "OpenAI Chat")?;
        let prompt_cache_key = super::optional_string(obj, "prompt_cache_key", "OpenAI Chat")?;
        let prompt_cache_retention =
            super::optional_string(obj, "prompt_cache_retention", "OpenAI Chat")?;
        let reasoning_effort = super::optional_string(obj, "reasoning_effort", "OpenAI Chat")?;
        let safety_identifier = super::optional_string(obj, "safety_identifier", "OpenAI Chat")?;
        let seed = super::optional_i64(obj, "seed", "OpenAI Chat")?;
        let verbosity = super::optional_string(obj, "verbosity", "OpenAI Chat")?;
        let metadata = super::optional_object(obj, "metadata", "OpenAI Chat")?;
        let audio = super::optional_object(obj, "audio", "OpenAI Chat")?;
        let logit_bias = super::optional_object(obj, "logit_bias", "OpenAI Chat")?;
        let moderation = super::optional_object(obj, "moderation", "OpenAI Chat")?;
        let prediction = super::optional_object(obj, "prediction", "OpenAI Chat")?;
        let prompt_cache_options =
            super::optional_object(obj, "prompt_cache_options", "OpenAI Chat")?;
        let response_format = super::optional_object(obj, "response_format", "OpenAI Chat")?;
        let stream_options = super::optional_object(obj, "stream_options", "OpenAI Chat")?;
        let web_search_options = super::optional_object(obj, "web_search_options", "OpenAI Chat")?;
        let extra: serde_json::Map<String, Json> = obj
            .iter()
            .filter(|(k, _)| !MODELED_REQUEST_KEYS.contains(&k.as_str()))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        Ok(AnnotatedLlmRequest {
            messages,
            instructions: None,
            model,
            params,
            tools,
            tool_choice,
            store,
            previous_response_id: None,
            truncation: None,
            reasoning: None,
            include: None,
            user,
            metadata,
            service_tier,
            parallel_tool_calls,
            max_output_tokens: None,
            max_tool_calls: None,
            top_logprobs,
            stream,
            api_specific: Some(ApiSpecificRequest::OpenAIChat {
                audio,
                frequency_penalty,
                function_call: obj.get("function_call").cloned(),
                functions,
                logit_bias,
                logprobs,
                modalities,
                moderation,
                n,
                prediction,
                presence_penalty,
                prompt_cache_key,
                prompt_cache_options,
                prompt_cache_retention,
                reasoning_effort,
                response_format,
                safety_identifier,
                seed,
                stream_options,
                verbosity,
                web_search_options,
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
        patch_chat_messages_and_validate(obj, annotated, &baseline)?;
        patch_chat_model_and_params(obj, annotated, &baseline);
        patch_chat_tools(obj, annotated, &baseline)?;
        patch_chat_common_fields(obj, annotated, &baseline);
        patch_chat_api_specific(obj, &annotated.api_specific, &baseline.api_specific)?;
        patch_extra_fields(obj, &baseline.extra, &annotated.extra);

        Ok(LlmRequest {
            headers: original.headers.clone(),
            content,
        })
    }
}

/// Helper to construct a [`Json`] number from an `f64`.
fn json_f64(v: f64) -> Json {
    serde_json::Number::from_f64(v)
        .map(Json::Number)
        .unwrap_or(Json::Null)
}

// ---------------------------------------------------------------------------
// Streaming codec
// ---------------------------------------------------------------------------

/// Streaming counterpart to [`OpenAIChatCodec`].
///
/// Replays the OpenAI Chat Completions SSE chunk sequence into the same JSON shape returned for a
/// non-streaming request (`{id, object, created, model, choices: [{message, finish_reason}],
/// usage}`). Once finalized, the assembled JSON can be fed back through
/// [`OpenAIChatCodec::decode_response`] to produce the canonical
/// [`AnnotatedLlmResponse`].
///
/// # Strategy
///
/// Chat Completions streams untyped SSE chunks of `{choices: [{index, delta: {...},
/// finish_reason: ...}]}`. Each delta may carry a `role` (typically only on the first chunk),
/// incremental `content` text, or partial `tool_calls` whose `function.arguments` stream as a
/// JSON-encoded string fragment-by-fragment. Top-level fields (`id`, `model`, `created`) are
/// repeated on every chunk; we capture them once. Final-chunk `usage` is preserved when emitted
/// (only sent when `stream_options.include_usage` is set on the request).
///
/// The OpenAI `[DONE]` end-of-stream sentinel is dropped by the SSE event decoder before
/// reaching the collector, so this codec never sees it.
///
/// Internal state lives behind `Arc<Mutex<...>>` so the `&self`-produced collector and finalizer
/// closures share access. Each instance is single-use because [`LlmFinalizerFn`] consumes the
/// finalize step.
///
/// [`LlmFinalizerFn`]: crate::api::runtime::LlmFinalizerFn
pub struct OpenAIChatStreamingCodec {
    state: std::sync::Arc<std::sync::Mutex<OpenAIChatStreamingState>>,
}

impl OpenAIChatStreamingCodec {
    /// Creates a fresh streaming codec with empty accumulator state.
    pub fn new() -> Self {
        Self {
            state: std::sync::Arc::new(std::sync::Mutex::new(OpenAIChatStreamingState::default())),
        }
    }
}

impl Default for OpenAIChatStreamingCodec {
    fn default() -> Self {
        Self::new()
    }
}

impl super::streaming::StreamingCodec for OpenAIChatStreamingCodec {
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
            std::mem::take(&mut *guard).finalize()
        })
    }
}

#[derive(Debug, Default)]
struct OpenAIChatStreamingState {
    id: Option<String>,
    object: Option<String>,
    created: Option<u64>,
    model: Option<String>,
    /// Per-choice accumulator keyed by `choice.index`. BTreeMap so finalize emits choices in
    /// stable order.
    choices: std::collections::BTreeMap<u64, ChoiceState>,
    /// Top-level usage from the final chunk (when `stream_options.include_usage` is set).
    usage: Option<Json>,
}

#[derive(Debug, Default)]
struct ChoiceState {
    role: Option<String>,
    content: String,
    has_content: bool,
    /// Tool calls keyed by their `index` within the choice. Each tool call's `arguments` is
    /// streamed as a JSON-encoded string accumulated fragment-by-fragment.
    tool_calls: std::collections::BTreeMap<u64, ToolCallState>,
    finish_reason: Option<String>,
}

#[derive(Debug, Default)]
struct ToolCallState {
    id: Option<String>,
    type_: Option<String>,
    name: Option<String>,
    arguments: String,
}

impl OpenAIChatStreamingState {
    fn observe(&mut self, chunk: &Json) {
        // Top-level fields (id, object, created, model) are repeated on every chunk; capture once
        // each so unrelated later chunks can't overwrite the canonical values.
        if self.id.is_none()
            && let Some(id) = chunk.get("id").and_then(Json::as_str)
        {
            self.id = Some(id.to_string());
        }
        if self.object.is_none()
            && let Some(obj) = chunk.get("object").and_then(Json::as_str)
        {
            self.object = Some(obj.to_string());
        }
        if self.created.is_none()
            && let Some(c) = chunk.get("created").and_then(Json::as_u64)
        {
            self.created = Some(c);
        }
        if self.model.is_none()
            && let Some(m) = chunk.get("model").and_then(Json::as_str)
        {
            self.model = Some(m.to_string());
        }
        if let Some(usage) = chunk.get("usage") {
            // Some streams emit `usage: null` on every chunk and the real usage only on the
            // final chunk; only capture non-null usage objects.
            if !usage.is_null() {
                self.usage = Some(usage.clone());
            }
        }
        let Some(choices) = chunk.get("choices").and_then(Json::as_array) else {
            return;
        };
        for choice in choices {
            self.observe_choice(choice);
        }
    }

    fn observe_choice(&mut self, choice: &Json) {
        let index = choice.get("index").and_then(Json::as_u64).unwrap_or(0);
        let entry = self.choices.entry(index).or_default();
        entry.observe_finish_reason(choice);
        entry.observe_delta(choice.get("delta"));
    }

    fn finalize(self) -> Json {
        let mut output = serde_json::Map::new();
        if let Some(id) = self.id {
            output.insert("id".to_string(), Json::String(id));
        }
        // After streaming, the final shape is `chat.completion`, not `chat.completion.chunk`.
        // Strip the `.chunk` suffix so the assembled JSON round-trips through
        // OpenAIChatCodec::decode_response with the same `object` field a non-streaming response
        // would carry.
        if let Some(object) = self.object {
            let normalized = object
                .strip_suffix(".chunk")
                .map(str::to_string)
                .unwrap_or(object);
            output.insert("object".to_string(), Json::String(normalized));
        }
        if let Some(created) = self.created {
            output.insert("created".to_string(), Json::Number(created.into()));
        }
        if let Some(model) = self.model {
            output.insert("model".to_string(), Json::String(model));
        }
        let choices: Vec<Json> = self
            .choices
            .into_iter()
            .map(|(index, choice)| choice.finalize(index))
            .collect();
        output.insert("choices".to_string(), Json::Array(choices));
        if let Some(usage) = self.usage {
            output.insert("usage".to_string(), usage);
        }
        Json::Object(output)
    }
}

impl ChoiceState {
    fn observe_finish_reason(&mut self, choice: &Json) {
        if let Some(reason) = choice.get("finish_reason").and_then(Json::as_str) {
            self.finish_reason = Some(reason.to_string());
        }
    }

    fn observe_delta(&mut self, delta: Option<&Json>) {
        let Some(delta) = delta else {
            return;
        };
        if let Some(role) = delta.get("role").and_then(Json::as_str) {
            self.role = Some(role.to_string());
        }
        if let Some(content) = delta.get("content").and_then(Json::as_str) {
            self.content.push_str(content);
            self.has_content = true;
        }
        self.observe_tool_calls(delta);
    }

    fn observe_tool_calls(&mut self, delta: &Json) {
        if let Some(tool_calls) = delta.get("tool_calls").and_then(Json::as_array) {
            for tool_call in tool_calls {
                self.observe_tool_call(tool_call);
            }
        }
    }

    fn observe_tool_call(&mut self, tool_call: &Json) {
        let index = tool_call.get("index").and_then(Json::as_u64).unwrap_or(0);
        let state = self.tool_calls.entry(index).or_default();
        if let Some(id) = tool_call.get("id").and_then(Json::as_str) {
            state.id = Some(id.to_string());
        }
        if let Some(type_) = tool_call.get("type").and_then(Json::as_str) {
            state.type_ = Some(type_.to_string());
        }
        if let Some(function) = tool_call.get("function") {
            state.observe_function(function);
        }
    }

    fn finalize(self, index: u64) -> Json {
        let mut message = serde_json::Map::new();
        message.insert(
            "role".to_string(),
            Json::String(self.role.unwrap_or_else(|| "assistant".to_string())),
        );
        // OpenAI's wire format uses `content: null` when the model only emitted tool calls.
        // Preserve that distinction: empty-string content when the model said something, null
        // when it didn't.
        if self.has_content {
            message.insert("content".to_string(), Json::String(self.content));
        } else {
            message.insert("content".to_string(), Json::Null);
        }
        if !self.tool_calls.is_empty() {
            let tool_calls: Vec<Json> = self
                .tool_calls
                .into_values()
                .map(ToolCallState::finalize)
                .collect();
            message.insert("tool_calls".to_string(), Json::Array(tool_calls));
        }
        let mut choice = serde_json::Map::new();
        choice.insert("index".to_string(), Json::Number(index.into()));
        choice.insert("message".to_string(), Json::Object(message));
        if let Some(reason) = self.finish_reason {
            choice.insert("finish_reason".to_string(), Json::String(reason));
        } else {
            choice.insert("finish_reason".to_string(), Json::Null);
        }
        Json::Object(choice)
    }
}

impl ToolCallState {
    fn observe_function(&mut self, function: &Json) {
        if let Some(name) = function.get("name").and_then(Json::as_str) {
            self.name = Some(name.to_string());
        }
        if let Some(args) = function.get("arguments").and_then(Json::as_str) {
            self.arguments.push_str(args);
        }
    }

    fn finalize(self) -> Json {
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
            Json::String(self.type_.unwrap_or_else(|| "function".to_string())),
        );
        call.insert("function".to_string(), Json::Object(function));
        Json::Object(call)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "../../tests/unit/codec/openai_chat_tests.rs"]
mod tests;
