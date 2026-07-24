// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Built-in codec for the Anthropic Messages API.
//!
//! Implements [`LlmCodec`] (request decode/encode) and [`LlmResponseCodec`]
//! (response decode) for the Anthropic Messages API format.
//!
//! # Anthropic-specific patterns handled
//!
//! - **Content blocks**: Heterogeneous array of `text`, `tool_use`, `thinking`,
//!   `redacted_thinking`, `mcp_tool_use`, `server_tool_use` blocks
//! - **Top-level system**: System prompt is a top-level field, not inside messages
//! - **stop_reason**: Maps to [`FinishReason`] (not `finish_reason`)
//! - **Tool definitions**: Uses `input_schema` instead of `parameters`
//! - **Tool choice**: `{"type":"auto"}` / `{"type":"any"}` / `{"type":"tool","name":"..."}`
//! - **Cache tokens**: `cache_read_input_tokens` / `cache_creation_input_tokens`

use serde::Deserialize;

use crate::api::llm::LlmRequest;
use crate::api::runtime::{BuiltinLlmCodec, LlmCodecIdentity};
use crate::error::{FlowError, Result};
use crate::json::Json;

use super::request::{
    AnnotatedLlmRequest, ApiSpecificRequest, ContentPart, FunctionDefinition, GenerationParams,
    Message, MessageContent, ProviderNativeComponent, ToolChoice, ToolChoiceFunction,
    ToolChoiceFunctionName, ToolDefinition,
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

/// Built-in codec for the Anthropic Messages API.
pub struct AnthropicMessagesCodec;

pub(crate) const PROVIDER_SURFACE: ProviderSurfaceDescriptor = ProviderSurfaceDescriptor {
    surface: ProviderSurface::AnthropicMessages,
    detect_request: |obj, hint| {
        // A system-less Anthropic request is shape-identical to OpenAI Chat;
        // a recognized Anthropic provider hint disambiguates it.
        let hinted_anthropic = hint.is_some_and(|hint_value| {
            hint_value == "anthropic" || hint_value == "anthropic.messages"
        });
        obj.contains_key("system") || (hinted_anthropic && obj.contains_key("messages"))
    },
    detect_response: |obj| {
        obj.get("type").and_then(Json::as_str) == Some("message")
            && obj.get("content").is_some_and(Json::is_array)
    },
    decode_request: |request| AnthropicMessagesCodec.decode(request),
    decode_response: |raw| AnthropicMessagesCodec.decode_response(raw),
    codec_name: "anthropic_messages",
    request_codec: || std::sync::Arc::new(AnthropicMessagesCodec),
    response_codec: || std::sync::Arc::new(AnthropicMessagesCodec),
    streaming_codec: || Box::new(AnthropicMessagesStreamingCodec::new()),
};

// ---------------------------------------------------------------------------
// Private intermediate serde structs for response decode
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct RawAnthropicResponse {
    id: Option<String>,
    #[serde(rename = "type")]
    object_type: Option<String>,
    role: Option<String>,
    model: Option<String>,
    content: Option<Vec<Json>>,
    stop_reason: Option<String>,
    stop_sequence: Option<String>,
    service_tier: Option<String>,
    container: Option<Json>,
    usage: Option<RawAnthropicUsage>,
    #[serde(flatten)]
    extra: serde_json::Map<String, Json>,
}

#[derive(Deserialize)]
struct RawAnthropicUsage {
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    cache_read_input_tokens: Option<u64>,
    cache_creation_input_tokens: Option<u64>,
    #[serde(rename = "cost_usd")]
    provider_cost: Option<f64>,
    cost: Option<RawUsageCost>,
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Map Anthropic `stop_reason` string to normalized [`FinishReason`].
fn map_anthropic_stop_reason(reason: &str) -> FinishReason {
    match reason {
        "end_turn" => FinishReason::Complete,
        "max_tokens" => FinishReason::Length,
        "tool_use" => FinishReason::ToolUse,
        other => FinishReason::Unknown(other.to_string()),
    }
}

/// Helper to construct a [`Json`] number from an `f64`.
fn json_f64(v: f64) -> Json {
    serde_json::Number::from_f64(v)
        .map(Json::Number)
        .unwrap_or(Json::Null)
}

/// Keys that are modeled in [`AnnotatedLlmRequest`] and should NOT go into `extra`.
const MODELED_REQUEST_KEYS: &[&str] = &[
    "system",
    "messages",
    "model",
    "max_tokens",
    "temperature",
    "top_p",
    "stop_sequences",
    "tools",
    "tool_choice",
    "metadata",
    "service_tier",
    "stream",
    "cache_control",
    "container",
    "inference_geo",
    "output_config",
    "thinking",
    "top_k",
    "anthropic-user-profile-id",
];

/// Decode the Anthropic `tool_choice` JSON value into a normalized [`ToolChoice`].
///
/// Anthropic format:
/// - `{"type": "auto"}` -> `ToolChoice::Auto`
/// - `{"type": "any"}` -> `ToolChoice::Required`
/// - `{"type": "none"}` -> `ToolChoice::None`
/// - `{"type": "tool", "name": "X"}` -> `ToolChoice::Specific`
fn decode_anthropic_tool_choice(val: &Json) -> Option<ToolChoice> {
    let obj = val.as_object()?;
    let tc_type = obj.get("type")?.as_str()?;
    match tc_type {
        "auto" => Some(ToolChoice::Auto),
        "any" => Some(ToolChoice::Required),
        "none" => Some(ToolChoice::None),
        "tool" => {
            let name = obj.get("name")?.as_str()?.to_string();
            Some(ToolChoice::Specific(ToolChoiceFunction {
                choice_type: "function".into(),
                function: ToolChoiceFunctionName { name },
            }))
        }
        _ => None,
    }
}

/// Extract Anthropic `disable_parallel_tool_use` from tool_choice and map
/// to normalized `parallel_tool_calls` semantics.
fn decode_parallel_tool_calls(val: &Json) -> Result<Option<bool>> {
    let Some(obj) = val.as_object() else {
        return Ok(None);
    };
    Ok(super::optional_bool(
        obj,
        "disable_parallel_tool_use",
        "Anthropic Messages tool_choice",
    )?
    .map(|disabled| !disabled))
}

/// Encode a normalized [`ToolChoice`] back into Anthropic JSON format.
fn encode_anthropic_tool_choice(tc: &ToolChoice) -> Result<Json> {
    match tc {
        ToolChoice::Auto => Ok(serde_json::json!({"type": "auto"})),
        ToolChoice::Required => Ok(serde_json::json!({"type": "any"})),
        ToolChoice::None => Ok(serde_json::json!({"type": "none"})),
        ToolChoice::Specific(func) => {
            Ok(serde_json::json!({"type": "tool", "name": func.function.name}))
        }
        ToolChoice::ProviderNative(native) if native.provider == "anthropic_messages" => {
            Ok(native.value.clone())
        }
        ToolChoice::ProviderNative(native) => Err(FlowError::InvalidArgument(format!(
            "tool choice for {} cannot be encoded for Anthropic Messages",
            native.provider
        ))),
    }
}

fn encode_tool_choice_with_parallel_hint(
    tc: &ToolChoice,
    parallel_tool_calls: Option<bool>,
) -> Result<Json> {
    let mut value = encode_anthropic_tool_choice(tc)?;
    if let (Some(parallel), Some(obj)) = (parallel_tool_calls, value.as_object_mut()) {
        obj.insert("disable_parallel_tool_use".into(), Json::Bool(!parallel));
    }
    Ok(value)
}

fn native_component(provider: &str, value: &Json) -> ProviderNativeComponent {
    ProviderNativeComponent {
        provider: provider.to_string(),
        kind: value
            .get("type")
            .and_then(Json::as_str)
            .unwrap_or("unknown")
            .to_string(),
        value: value.clone(),
    }
}

fn decode_anthropic_content(value: &Json) -> Result<MessageContent> {
    if let Some(text) = value.as_str() {
        return Ok(MessageContent::Text(text.to_string()));
    }
    let blocks = value.as_array().ok_or_else(|| {
        FlowError::InvalidArgument("Anthropic Messages content must be a string or an array".into())
    })?;
    let parts = blocks
        .iter()
        .map(decode_anthropic_content_part)
        .collect::<Result<Vec<_>>>()?;
    Ok(MessageContent::Parts(parts))
}

fn decode_anthropic_content_part(value: &Json) -> Result<ContentPart> {
    let obj = value.as_object().ok_or_else(|| {
        FlowError::InvalidArgument("Anthropic Messages content block must be an object".into())
    })?;
    let kind = obj.get("type").and_then(Json::as_str).unwrap_or("unknown");
    match kind {
        "text" => {
            let text = obj.get("text").and_then(Json::as_str).ok_or_else(|| {
                FlowError::InvalidArgument("Anthropic text block is missing text".into())
            })?;
            Ok(ContentPart::Text {
                text: text.to_string(),
                extra: obj
                    .iter()
                    .filter(|(key, _)| !matches!(key.as_str(), "type" | "text"))
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect(),
            })
        }
        "image" => Ok(ContentPart::Image {
            image: Json::Object(
                obj.iter()
                    .filter(|(key, _)| key.as_str() != "type")
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect(),
            ),
            extra: serde_json::Map::new(),
        }),
        "document" => Ok(ContentPart::File {
            file: Json::Object(
                obj.iter()
                    .filter(|(key, _)| key.as_str() != "type")
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect(),
            ),
            extra: serde_json::Map::new(),
        }),
        "tool_use" => {
            let id = obj.get("id").and_then(Json::as_str).ok_or_else(|| {
                FlowError::InvalidArgument("Anthropic tool_use block is missing id".into())
            })?;
            let name = obj.get("name").and_then(Json::as_str).ok_or_else(|| {
                FlowError::InvalidArgument("Anthropic tool_use block is missing name".into())
            })?;
            let input = obj.get("input").ok_or_else(|| {
                FlowError::InvalidArgument("Anthropic tool_use block is missing input".into())
            })?;
            let extra = obj
                .iter()
                .filter(|(key, _)| !matches!(key.as_str(), "type" | "id" | "name" | "input"))
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect();
            Ok(ContentPart::ToolUse {
                id: id.to_string(),
                name: name.to_string(),
                input: input.clone(),
                extra,
            })
        }
        "tool_result" => {
            let tool_use_id = obj
                .get("tool_use_id")
                .and_then(Json::as_str)
                .ok_or_else(|| {
                    FlowError::InvalidArgument(
                        "Anthropic tool_result block is missing tool_use_id".into(),
                    )
                })?;
            let content = obj.get("content").ok_or_else(|| {
                FlowError::InvalidArgument("Anthropic tool_result block is missing content".into())
            })?;
            let is_error = match obj.get("is_error") {
                Some(Json::Null) | None => None,
                Some(value) => Some(value.as_bool().ok_or_else(|| {
                    FlowError::InvalidArgument(
                        "Anthropic tool_result is_error must be a boolean".into(),
                    )
                })?),
            };
            let extra = obj
                .iter()
                .filter(|(key, _)| {
                    !matches!(
                        key.as_str(),
                        "type" | "tool_use_id" | "content" | "is_error"
                    )
                })
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect();
            Ok(ContentPart::ToolResult {
                tool_use_id: tool_use_id.to_string(),
                content: content.clone(),
                is_error,
                extra,
            })
        }
        _ => {
            let native = native_component("anthropic_messages", value);
            Ok(ContentPart::ProviderNative {
                provider: native.provider,
                kind: native.kind,
                value: native.value,
            })
        }
    }
}

fn decode_anthropic_message(value: &Json) -> Result<Message> {
    let obj = value.as_object().ok_or_else(|| {
        FlowError::InvalidArgument("Anthropic Messages message must be an object".into())
    })?;
    let role = obj.get("role").and_then(Json::as_str).ok_or_else(|| {
        FlowError::InvalidArgument("Anthropic Messages message is missing role".into())
    })?;
    let content = obj.get("content").ok_or_else(|| {
        FlowError::InvalidArgument("Anthropic Messages message is missing content".into())
    })?;
    if obj
        .keys()
        .any(|key| !matches!(key.as_str(), "role" | "content"))
    {
        return Ok(Message::ProviderNative {
            provider: "anthropic_messages".into(),
            kind: role.to_string(),
            value: value.clone(),
        });
    }
    let content = decode_anthropic_content(content)?;
    match role {
        "user" => Ok(Message::User {
            content,
            name: None,
        }),
        "assistant" => Ok(Message::Assistant {
            content: Some(content),
            tool_calls: None,
            name: None,
        }),
        "system" => Ok(Message::System {
            content,
            name: None,
        }),
        _ => Ok(Message::ProviderNative {
            provider: "anthropic_messages".into(),
            kind: role.to_string(),
            value: value.clone(),
        }),
    }
}

fn encode_anthropic_content(content: &MessageContent) -> Result<Json> {
    match content {
        MessageContent::Text(text) => Ok(Json::String(text.clone())),
        MessageContent::Parts(parts) => Ok(Json::Array(
            parts
                .iter()
                .map(encode_anthropic_content_part)
                .collect::<Result<Vec<_>>>()?,
        )),
    }
}

fn encode_anthropic_content_part(part: &ContentPart) -> Result<Json> {
    match part {
        ContentPart::Text { text, extra } => {
            let mut obj = extra.clone();
            obj.insert("type".into(), Json::String("text".into()));
            obj.insert("text".into(), Json::String(text.clone()));
            Ok(Json::Object(obj))
        }
        ContentPart::Image { image, extra } => {
            let mut obj = image.as_object().cloned().ok_or_else(|| {
                FlowError::InvalidArgument("Anthropic image payload must be an object".into())
            })?;
            obj.extend(extra.clone());
            obj.insert("type".into(), Json::String("image".into()));
            Ok(Json::Object(obj))
        }
        ContentPart::File { file, extra } => {
            let mut obj = file.as_object().cloned().ok_or_else(|| {
                FlowError::InvalidArgument("Anthropic document payload must be an object".into())
            })?;
            obj.extend(extra.clone());
            obj.insert("type".into(), Json::String("document".into()));
            Ok(Json::Object(obj))
        }
        ContentPart::ToolUse {
            id,
            name,
            input,
            extra,
        } => {
            let mut obj = extra.clone();
            obj.insert("type".into(), Json::String("tool_use".into()));
            obj.insert("id".into(), Json::String(id.clone()));
            obj.insert("name".into(), Json::String(name.clone()));
            obj.insert("input".into(), input.clone());
            Ok(Json::Object(obj))
        }
        ContentPart::ToolResult {
            tool_use_id,
            content,
            is_error,
            extra,
        } => {
            let mut obj = extra.clone();
            obj.insert("type".into(), Json::String("tool_result".into()));
            obj.insert("tool_use_id".into(), Json::String(tool_use_id.clone()));
            obj.insert("content".into(), content.clone());
            if let Some(is_error) = is_error {
                obj.insert("is_error".into(), Json::Bool(*is_error));
            }
            Ok(Json::Object(obj))
        }
        ContentPart::ProviderNative {
            provider, value, ..
        } if provider == "anthropic_messages" => Ok(value.clone()),
        other => Err(FlowError::InvalidArgument(format!(
            "content part {other:?} cannot be encoded for Anthropic Messages"
        ))),
    }
}

fn encode_anthropic_message(message: &Message) -> Result<Json> {
    match message {
        Message::User { content, .. } | Message::System { content, .. } => {
            let role = if matches!(message, Message::User { .. }) {
                "user"
            } else {
                "system"
            };
            let mut obj = serde_json::Map::new();
            obj.insert("role".into(), Json::String(role.into()));
            obj.insert("content".into(), encode_anthropic_content(content)?);
            Ok(Json::Object(obj))
        }
        Message::Assistant { content, .. } => {
            let mut obj = serde_json::Map::new();
            obj.insert("role".into(), Json::String("assistant".into()));
            obj.insert(
                "content".into(),
                match content {
                    Some(content) => encode_anthropic_content(content)?,
                    None => Json::Array(Vec::new()),
                },
            );
            Ok(Json::Object(obj))
        }
        Message::ProviderNative {
            provider, value, ..
        } if provider == "anthropic_messages" => Ok(value.clone()),
        other => Err(FlowError::InvalidArgument(format!(
            "message {other:?} cannot be encoded for Anthropic Messages"
        ))),
    }
}

fn encode_anthropic_tool(tool: &ToolDefinition) -> Result<Json> {
    match tool {
        ToolDefinition::Function { function, extra } => {
            let mut obj = extra.clone();
            obj.insert("name".into(), Json::String(function.name.clone()));
            if let Some(description) = &function.description {
                obj.insert("description".into(), Json::String(description.clone()));
            }
            if let Some(parameters) = &function.parameters {
                obj.insert("input_schema".into(), parameters.clone());
            }
            if let Some(strict) = function.strict {
                obj.insert("strict".into(), Json::Bool(strict));
            }
            obj.extend(function.extra.clone());
            Ok(Json::Object(obj))
        }
        ToolDefinition::ProviderNative {
            provider, value, ..
        } if provider == "anthropic_messages" => Ok(value.clone()),
        other => Err(FlowError::InvalidArgument(format!(
            "tool {other:?} cannot be encoded for Anthropic Messages"
        ))),
    }
}

fn decode_anthropic_tool(value: &Json) -> Result<ToolDefinition> {
    let obj = value.as_object().ok_or_else(|| {
        FlowError::InvalidArgument("Anthropic Messages tool must be an object".into())
    })?;
    let is_client_tool =
        obj.get("type").is_none() && obj.contains_key("name") && obj.contains_key("input_schema");
    if !is_client_tool {
        let native = native_component("anthropic_messages", value);
        return Ok(ToolDefinition::ProviderNative {
            provider: native.provider,
            kind: native.kind,
            value: native.value,
        });
    }

    let function_extra = serde_json::Map::new();
    let wrapper_extra = obj
        .iter()
        .filter(|(key, _)| {
            !matches!(
                key.as_str(),
                "name" | "description" | "input_schema" | "strict"
            )
        })
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    let name =
        super::optional_string(obj, "name", "Anthropic Messages tool")?.ok_or_else(|| {
            FlowError::InvalidArgument("Anthropic Messages tool is missing name".into())
        })?;
    let description = super::optional_string(obj, "description", "Anthropic Messages tool")?;
    let strict = super::optional_bool(obj, "strict", "Anthropic Messages tool")?;
    Ok(ToolDefinition::Function {
        function: FunctionDefinition {
            name,
            description,
            parameters: obj.get("input_schema").cloned(),
            strict,
            extra: function_extra,
        },
        extra: wrapper_extra,
    })
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

fn patch_anthropic_messages_and_model(
    obj: &mut serde_json::Map<String, Json>,
    annotated: &AnnotatedLlmRequest,
    baseline: &AnnotatedLlmRequest,
) -> Result<()> {
    if annotated.messages != baseline.messages {
        let original_messages = obj.get("messages").and_then(Json::as_array);
        let messages = super::encode_changed_items(
            &annotated.messages,
            &baseline.messages,
            original_messages.map(Vec::as_slice),
            encode_anthropic_message,
        )?;
        obj.insert("messages".into(), Json::Array(messages));
    }
    if annotated.instructions != baseline.instructions {
        set_or_remove_json(
            obj,
            "system",
            annotated
                .instructions
                .as_ref()
                .map(encode_anthropic_content)
                .transpose()?,
        );
    }
    if annotated.model != baseline.model {
        set_or_remove_json(obj, "model", annotated.model.clone().map(Json::String));
    }
    Ok(())
}

fn patch_anthropic_params(
    obj: &mut serde_json::Map<String, Json>,
    annotated: &AnnotatedLlmRequest,
    baseline: &AnnotatedLlmRequest,
) {
    if annotated.params == baseline.params {
        return;
    }
    let edited = annotated.params.as_ref();
    let before = baseline.params.as_ref();
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
    let max_tokens = edited.and_then(|params| params.max_tokens);
    if max_tokens != before.and_then(|params| params.max_tokens) {
        set_or_remove_json(obj, "max_tokens", max_tokens.map(Json::from));
    }
    let stop = edited.and_then(|params| params.stop.as_ref());
    if stop != before.and_then(|params| params.stop.as_ref()) {
        set_or_remove_json(
            obj,
            "stop_sequences",
            stop.map(|values| serde_json::json!(values)),
        );
    }
}

fn patch_anthropic_tools(
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
                    encode_anthropic_tool,
                )
            })
            .transpose()?
            .map(Json::Array);
        set_or_remove_json(obj, "tools", tools);
    }
    if annotated.tool_choice != baseline.tool_choice
        || annotated.parallel_tool_calls != baseline.parallel_tool_calls
    {
        let tool_choice = match (&annotated.tool_choice, &baseline.tool_choice) {
            (Some(edited), Some(before)) => {
                let edited =
                    encode_tool_choice_with_parallel_hint(edited, annotated.parallel_tool_calls)?;
                let before =
                    encode_tool_choice_with_parallel_hint(before, baseline.parallel_tool_calls)?;
                Some(match obj.get("tool_choice") {
                    Some(original) => super::patch_changed_json(original, &before, &edited)?,
                    None => edited,
                })
            }
            (Some(edited), None) => Some(encode_tool_choice_with_parallel_hint(
                edited,
                annotated.parallel_tool_calls,
            )?),
            (None, _) => annotated
                .parallel_tool_calls
                .map(|parallel| {
                    encode_tool_choice_with_parallel_hint(&ToolChoice::Auto, Some(parallel))
                })
                .transpose()?,
        };
        set_or_remove_json(obj, "tool_choice", tool_choice);
    }
    Ok(())
}

fn patch_anthropic_common_fields(
    obj: &mut serde_json::Map<String, Json>,
    annotated: &AnnotatedLlmRequest,
    baseline: &AnnotatedLlmRequest,
) {
    if annotated.metadata != baseline.metadata {
        set_or_remove_json(obj, "metadata", annotated.metadata.clone());
    }
    if annotated.service_tier != baseline.service_tier {
        set_or_remove_json(
            obj,
            "service_tier",
            annotated.service_tier.clone().map(Json::String),
        );
    }
    if annotated.stream != baseline.stream {
        set_or_remove_json(obj, "stream", annotated.stream.map(Json::Bool));
    }
}

fn validate_anthropic_supported_fields(
    annotated: &AnnotatedLlmRequest,
    baseline: &AnnotatedLlmRequest,
) -> Result<()> {
    let unsupported = [
        annotated.store != baseline.store,
        annotated.previous_response_id != baseline.previous_response_id,
        annotated.truncation != baseline.truncation,
        annotated.reasoning != baseline.reasoning,
        annotated.include != baseline.include,
        annotated.user != baseline.user,
        annotated.max_output_tokens != baseline.max_output_tokens,
        annotated.max_tool_calls != baseline.max_tool_calls,
        annotated.top_logprobs != baseline.top_logprobs,
    ]
    .into_iter()
    .any(|changed| changed);
    if unsupported {
        return Err(FlowError::InvalidArgument(
            "request contains fields that cannot be encoded for Anthropic Messages".into(),
        ));
    }
    Ok(())
}

fn patch_anthropic_api_specific(
    obj: &mut serde_json::Map<String, Json>,
    edited: &Option<ApiSpecificRequest>,
    baseline: &Option<ApiSpecificRequest>,
) -> Result<()> {
    match (edited, baseline) {
        (
            Some(ApiSpecificRequest::AnthropicMessages {
                cache_control,
                container,
                inference_geo,
                output_config,
                thinking,
                top_k,
                user_profile_id,
            }),
            Some(ApiSpecificRequest::AnthropicMessages {
                cache_control: old_cache_control,
                container: old_container,
                inference_geo: old_inference_geo,
                output_config: old_output_config,
                thinking: old_thinking,
                top_k: old_top_k,
                user_profile_id: old_user_profile_id,
            }),
        ) => {
            if cache_control != old_cache_control {
                set_or_remove_json(obj, "cache_control", cache_control.clone());
            }
            patch_anthropic_api_strings(
                obj,
                &[
                    ("container", container, old_container),
                    ("inference_geo", inference_geo, old_inference_geo),
                    (
                        "anthropic-user-profile-id",
                        user_profile_id,
                        old_user_profile_id,
                    ),
                ],
            );
            patch_anthropic_api_json(
                obj,
                &[
                    ("output_config", output_config, old_output_config),
                    ("thinking", thinking, old_thinking),
                ],
            );
            if top_k != old_top_k {
                set_or_remove_json(obj, "top_k", top_k.map(Json::from));
            }
            Ok(())
        }
        (None, Some(ApiSpecificRequest::AnthropicMessages { .. })) => {
            for key in [
                "cache_control",
                "container",
                "inference_geo",
                "output_config",
                "thinking",
                "top_k",
                "anthropic-user-profile-id",
            ] {
                obj.remove(key);
            }
            Ok(())
        }
        (Some(_), _) | (None, Some(_)) => Err(FlowError::InvalidArgument(
            "api_specific provider does not match Anthropic Messages".into(),
        )),
        (None, None) => Ok(()),
    }
}

fn patch_anthropic_api_strings(
    obj: &mut serde_json::Map<String, Json>,
    fields: &[(&str, &Option<String>, &Option<String>)],
) {
    for (key, edited, before) in fields {
        if edited != before {
            set_or_remove_json(obj, key, (*edited).clone().map(Json::String));
        }
    }
}

fn patch_anthropic_api_json(
    obj: &mut serde_json::Map<String, Json>,
    fields: &[(&str, &Option<Json>, &Option<Json>)],
) {
    for (key, edited, before) in fields {
        if edited != before {
            set_or_remove_json(obj, key, (*edited).clone());
        }
    }
}

fn anthropic_text_message(content_blocks: Option<&[Json]>) -> Option<MessageContent> {
    let text_parts: Vec<&str> = content_blocks
        .map(|blocks| blocks.iter().filter_map(anthropic_text_block).collect())
        .unwrap_or_default();

    (!text_parts.is_empty()).then(|| MessageContent::Text(text_parts.join("\n")))
}

fn anthropic_text_block(block: &Json) -> Option<&str> {
    if block.get("type")?.as_str()? != "text" {
        return None;
    }
    block.get("text")?.as_str()
}

fn anthropic_tool_calls(content_blocks: Option<&[Json]>) -> Option<Vec<ResponseToolCall>> {
    let tool_calls: Vec<ResponseToolCall> = content_blocks
        .map(|blocks| {
            blocks
                .iter()
                .filter_map(anthropic_tool_call_block)
                .collect()
        })
        .unwrap_or_default();

    (!tool_calls.is_empty()).then_some(tool_calls)
}

fn anthropic_tool_call_block(block: &Json) -> Option<ResponseToolCall> {
    if block.get("type")?.as_str()? != "tool_use" {
        return None;
    }
    Some(ResponseToolCall {
        id: block.get("id")?.as_str()?.to_string(),
        name: block.get("name")?.as_str()?.to_string(),
        // CRITICAL: input is already parsed JSON -- clone directly.
        arguments: block.get("input")?.clone(),
    })
}

fn anthropic_usage(
    raw_usage: Option<RawAnthropicUsage>,
    model_for_pricing: Option<&str>,
) -> Option<Usage> {
    let model_provider = infer_model_provider("anthropic", model_for_pricing);
    raw_usage.map(|u| {
        let prompt = u.input_tokens;
        let completion = u.output_tokens;
        let mut usage = Usage {
            prompt_tokens: prompt,
            completion_tokens: completion,
            // Anthropic does not supply total_tokens; compute it.
            total_tokens: match (prompt, completion) {
                (Some(p), Some(c)) => Some(p + c),
                _ => None,
            },
            cache_read_tokens: u.cache_read_input_tokens,
            cache_write_tokens: u.cache_creation_input_tokens,
            cost: provider_reported_cost(u.provider_cost, u.cost),
        };
        if usage.cost.is_none() {
            usage.cost = model_for_pricing.and_then(|model| {
                estimate_cost_for_provider(model_provider.as_deref(), model, &usage)
            });
        }
        usage
    })
}

// ---------------------------------------------------------------------------
// LlmResponseCodec implementation
// ---------------------------------------------------------------------------

impl LlmResponseCodec for AnthropicMessagesCodec {
    fn codec_identity(&self) -> LlmCodecIdentity {
        LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::AnthropicMessages)
    }

    fn decode_response(&self, response: &Json) -> Result<AnnotatedLlmResponse> {
        let raw: RawAnthropicResponse = serde_json::from_value(response.clone())
            .map_err(|e| FlowError::Internal(format!("Anthropic Messages response decode: {e}")))?;

        let content_blocks = raw.content.as_deref();
        let message = anthropic_text_message(content_blocks);
        // Extract tool_use blocks (only "tool_use" type, NOT mcp_tool_use or server_tool_use).
        let tool_calls = anthropic_tool_calls(content_blocks);

        // Map stop_reason to FinishReason.
        let finish_reason = raw.stop_reason.as_deref().map(map_anthropic_stop_reason);

        // Map usage.
        let usage = anthropic_usage(raw.usage, raw.model.as_deref());

        // Build API-specific fields: all content blocks + stop_sequence.
        let api_specific_content_blocks = raw.content.clone();
        let api_specific = Some(ApiSpecificResponse::AnthropicMessages {
            object_type: raw.object_type,
            role: raw.role,
            stop_reason: raw.stop_reason,
            stop_sequence: raw.stop_sequence,
            service_tier: raw.service_tier,
            container: raw.container,
            content_blocks: api_specific_content_blocks,
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

impl LlmCodec for AnthropicMessagesCodec {
    fn codec_identity(&self) -> LlmCodecIdentity {
        LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::AnthropicMessages)
    }

    fn decode(&self, request: &LlmRequest) -> Result<AnnotatedLlmRequest> {
        let obj = request
            .content
            .as_object()
            .ok_or_else(|| FlowError::Internal("request content is not an object".into()))?;
        let raw_messages = obj.get("messages").ok_or_else(|| {
            FlowError::InvalidArgument("Anthropic Messages request is missing messages".into())
        })?;
        let messages = raw_messages
            .as_array()
            .ok_or_else(|| {
                FlowError::InvalidArgument("Anthropic Messages messages must be an array".into())
            })?
            .iter()
            .map(decode_anthropic_message)
            .collect::<Result<Vec<_>>>()?;
        let instructions = obj
            .get("system")
            .map(decode_anthropic_content)
            .transpose()?;
        let model = super::optional_string(obj, "model", "Anthropic Messages")?;
        let temperature = super::optional_f64(obj, "temperature", "Anthropic Messages")?;
        let top_p = super::optional_f64(obj, "top_p", "Anthropic Messages")?;
        let max_tokens = super::optional_u64(obj, "max_tokens", "Anthropic Messages")?;
        let stop = obj
            .get("stop_sequences")
            .filter(|value| !value.is_null())
            .map(|value| {
                serde_json::from_value::<Vec<String>>(value.clone()).map_err(|error| {
                    FlowError::InvalidArgument(format!(
                        "invalid Anthropic stop_sequences value: {error}"
                    ))
                })
            })
            .transpose()?;
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
                        FlowError::InvalidArgument(
                            "Anthropic Messages tools must be an array".into(),
                        )
                    })?
                    .iter()
                    .map(decode_anthropic_tool)
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        let tool_choice = obj.get("tool_choice").map(|value| {
            decode_anthropic_tool_choice(value).unwrap_or_else(|| {
                ToolChoice::ProviderNative(native_component("anthropic_messages", value))
            })
        });
        let parallel_tool_calls = obj
            .get("tool_choice")
            .map(decode_parallel_tool_calls)
            .transpose()?
            .flatten();
        let service_tier = super::optional_string(obj, "service_tier", "Anthropic Messages")?;
        let stream = super::optional_bool(obj, "stream", "Anthropic Messages")?;
        let container = super::optional_string(obj, "container", "Anthropic Messages")?;
        let inference_geo = super::optional_string(obj, "inference_geo", "Anthropic Messages")?;
        let top_k = super::optional_u64(obj, "top_k", "Anthropic Messages")?;
        let user_profile_id =
            super::optional_string(obj, "anthropic-user-profile-id", "Anthropic Messages")?;
        let metadata = super::optional_object(obj, "metadata", "Anthropic Messages")?;
        let cache_control = super::optional_object(obj, "cache_control", "Anthropic Messages")?;
        let output_config = super::optional_object(obj, "output_config", "Anthropic Messages")?;
        let thinking = super::optional_object(obj, "thinking", "Anthropic Messages")?;
        let extra: serde_json::Map<String, Json> = obj
            .iter()
            .filter(|(k, _)| !MODELED_REQUEST_KEYS.contains(&k.as_str()))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        Ok(AnnotatedLlmRequest {
            messages,
            instructions,
            model,
            params,
            tools,
            tool_choice,
            store: None,
            previous_response_id: None,
            truncation: None,
            reasoning: None,
            include: None,
            user: None,
            metadata,
            service_tier,
            parallel_tool_calls,
            max_output_tokens: None,
            max_tool_calls: None,
            top_logprobs: None,
            stream,
            api_specific: Some(ApiSpecificRequest::AnthropicMessages {
                cache_control,
                container,
                inference_geo,
                output_config,
                thinking,
                top_k,
                user_profile_id,
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
        patch_anthropic_messages_and_model(obj, annotated, &baseline)?;
        patch_anthropic_params(obj, annotated, &baseline);
        patch_anthropic_tools(obj, annotated, &baseline)?;
        patch_anthropic_common_fields(obj, annotated, &baseline);
        validate_anthropic_supported_fields(annotated, &baseline)?;
        patch_anthropic_api_specific(obj, &annotated.api_specific, &baseline.api_specific)?;
        patch_extra_fields(obj, &baseline.extra, &annotated.extra);

        Ok(LlmRequest {
            headers: original.headers.clone(),
            content,
        })
    }
}

// ---------------------------------------------------------------------------
// Streaming codec
// ---------------------------------------------------------------------------

/// Streaming counterpart to [`AnthropicMessagesCodec`].
///
/// Replays the Anthropic Messages SSE event sequence into the same JSON shape Anthropic returns
/// for a non-streaming request (`{id, type, role, model, content, stop_reason, stop_sequence,
/// usage}`). Once finalized, the assembled JSON can be fed back through
/// [`AnthropicMessagesCodec::decode_response`] to produce an
/// [`AnnotatedLlmResponse`] — meaning streaming and
/// non-streaming Anthropic requests converge on the same observability output.
///
/// Internal state lives behind `Arc<Mutex<...>>` so the `&self`-produced collector and finalizer
/// closures share access. Each instance is single-use because [`LlmFinalizerFn`] consumes the
/// finalize step.
///
/// [`LlmFinalizerFn`]: crate::api::runtime::LlmFinalizerFn
pub struct AnthropicMessagesStreamingCodec {
    state: std::sync::Arc<std::sync::Mutex<AnthropicMessagesStreamingState>>,
}

impl AnthropicMessagesStreamingCodec {
    /// Creates a fresh streaming codec with empty accumulator state.
    pub fn new() -> Self {
        Self {
            state: std::sync::Arc::new(std::sync::Mutex::new(
                AnthropicMessagesStreamingState::default(),
            )),
        }
    }
}

impl Default for AnthropicMessagesStreamingCodec {
    fn default() -> Self {
        Self::new()
    }
}

impl super::streaming::StreamingCodec for AnthropicMessagesStreamingCodec {
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
struct AnthropicMessagesStreamingState {
    id: Option<String>,
    type_: Option<String>,
    role: Option<String>,
    model: Option<String>,
    /// Latest usage snapshot. `message_start` carries an initial value (input tokens, zero output
    /// so far); `message_delta` updates it cumulatively. Last write wins.
    usage: Option<Json>,
    stop_reason: Option<String>,
    /// Stored as raw `Json` to preserve `null` (Anthropic's wire shape) versus omitted.
    stop_sequence: Option<Json>,
    /// Indexed by the SSE event's `index` field. `None` slots accommodate sparse indices though
    /// Anthropic emits them in order today.
    blocks: Vec<Option<StreamingBlock>>,
}

#[derive(Debug, Default, Clone)]
struct StreamingBlock {
    /// The `content_block` JSON captured at `content_block_start`. Deltas mutate fields directly
    /// for blocks Anthropic delivers incrementally (text, tool_use input, citations); other block
    /// types (server_tool_use results) ship complete at start and pass through unchanged.
    skeleton: serde_json::Map<String, Json>,
    text: String,
    has_text: bool,
    partial_json: String,
    has_partial_json: bool,
    citations: Vec<Json>,
    has_citations: bool,
}

impl AnthropicMessagesStreamingState {
    fn observe(&mut self, event: &Json) {
        let event_type = event.get("type").and_then(Json::as_str).unwrap_or("");
        match event_type {
            "message_start" => self.observe_message_start(event),
            "content_block_start" => self.observe_content_block_start(event),
            "content_block_delta" => self.observe_content_block_delta(event),
            "message_delta" => self.observe_message_delta(event),
            // content_block_stop, message_stop, ping, and any unknown event type carry no
            // accumulator-relevant payload. Unknown types are ignored rather than erroring so a
            // future Anthropic event addition does not break observability.
            _ => {}
        }
    }

    fn observe_message_start(&mut self, event: &Json) {
        let Some(message) = event.get("message") else {
            return;
        };
        if let Some(id) = message.get("id").and_then(Json::as_str) {
            self.id = Some(id.to_string());
        }
        if let Some(model) = message.get("model").and_then(Json::as_str) {
            self.model = Some(model.to_string());
        }
        if let Some(role) = message.get("role").and_then(Json::as_str) {
            self.role = Some(role.to_string());
        }
        if let Some(t) = message.get("type").and_then(Json::as_str) {
            self.type_ = Some(t.to_string());
        }
        if let Some(usage) = message.get("usage") {
            self.usage = Some(usage.clone());
        }
    }

    fn observe_content_block_start(&mut self, event: &Json) {
        let Some(index) = event.get("index").and_then(Json::as_u64) else {
            return;
        };
        let Some(content_block) = event.get("content_block") else {
            return;
        };
        let skeleton = match content_block {
            Json::Object(map) => map.clone(),
            _ => return,
        };
        let index = index as usize;
        while self.blocks.len() <= index {
            self.blocks.push(None);
        }
        self.blocks[index] = Some(StreamingBlock {
            skeleton,
            ..StreamingBlock::default()
        });
    }

    fn observe_content_block_delta(&mut self, event: &Json) {
        let Some(index) = event.get("index").and_then(Json::as_u64) else {
            return;
        };
        let index = index as usize;
        let Some(delta) = event.get("delta") else {
            return;
        };
        let delta_type = delta.get("type").and_then(Json::as_str).unwrap_or("");
        let Some(slot) = self.blocks.get_mut(index) else {
            return;
        };
        let Some(block) = slot.as_mut() else { return };
        match delta_type {
            "text_delta" => {
                if let Some(text) = delta.get("text").and_then(Json::as_str) {
                    block.text.push_str(text);
                    block.has_text = true;
                }
            }
            "input_json_delta" => {
                if let Some(partial) = delta.get("partial_json").and_then(Json::as_str) {
                    block.partial_json.push_str(partial);
                    block.has_partial_json = true;
                }
            }
            "citations_delta" => {
                if let Some(citation) = delta.get("citation") {
                    block.citations.push(citation.clone());
                    block.has_citations = true;
                }
            }
            // thinking_delta, signature_delta, and any future delta types fall through; the block
            // skeleton retains whatever shape was set at content_block_start.
            _ => {}
        }
    }

    fn observe_message_delta(&mut self, event: &Json) {
        if let Some(delta) = event.get("delta") {
            if let Some(reason) = delta.get("stop_reason").and_then(Json::as_str) {
                self.stop_reason = Some(reason.to_string());
            }
            if let Some(seq) = delta.get("stop_sequence") {
                self.stop_sequence = Some(seq.clone());
            }
        }
        if let Some(usage) = event.get("usage") {
            self.usage = Some(usage.clone());
        }
    }

    fn finalize(self) -> Json {
        let mut output = serde_json::Map::new();
        if let Some(id) = self.id {
            output.insert("id".to_string(), Json::String(id));
        }
        if let Some(t) = self.type_ {
            output.insert("type".to_string(), Json::String(t));
        }
        if let Some(role) = self.role {
            output.insert("role".to_string(), Json::String(role));
        }
        if let Some(model) = self.model {
            output.insert("model".to_string(), Json::String(model));
        }
        let content: Vec<Json> = self
            .blocks
            .into_iter()
            .filter_map(|block| block.map(StreamingBlock::finalize))
            .collect();
        output.insert("content".to_string(), Json::Array(content));
        if let Some(reason) = self.stop_reason {
            output.insert("stop_reason".to_string(), Json::String(reason));
        }
        if let Some(seq) = self.stop_sequence {
            output.insert("stop_sequence".to_string(), seq);
        }
        if let Some(usage) = self.usage {
            output.insert("usage".to_string(), usage);
        }
        Json::Object(output)
    }
}

impl StreamingBlock {
    fn finalize(mut self) -> Json {
        if self.has_text {
            self.skeleton
                .insert("text".to_string(), Json::String(self.text));
        }
        if self.has_partial_json {
            // Concatenated `partial_json` fragments are expected to parse as a JSON object — that's
            // the assembled tool input. If parsing fails (Anthropic emits malformed deltas, stream
            // truncated mid-block), surface the raw concatenation so observability still captures
            // something rather than dropping the call.
            let parsed = match serde_json::from_str::<Json>(&self.partial_json) {
                Ok(value) => value,
                Err(_) => Json::String(self.partial_json),
            };
            self.skeleton.insert("input".to_string(), parsed);
        }
        if self.has_citations {
            self.skeleton
                .insert("citations".to_string(), Json::Array(self.citations));
        }
        Json::Object(self.skeleton)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "../../tests/unit/codec/anthropic_tests.rs"]
mod tests;
