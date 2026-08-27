// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Built-in codec for Amazon Bedrock Converse and ConverseStream payloads.
//!
//! This codec operates on the JSON-compatible request and response dictionaries used by AWS SDK
//! calls. It does not implement SigV4 signing, HTTP transport, or conversion of SDK-native binary
//! values into JSON.

use crate::api::llm::LlmRequest;
use crate::api::runtime::{BuiltinLlmCodec, LlmCodecIdentity};
use crate::error::{FlowError, Result, UpstreamFailure, UpstreamFailureClass};
use crate::json::Json;

use super::request::{
    AnnotatedLlmRequest, ContentPart, FunctionDefinition, GenerationParams, Message,
    MessageContent, ProviderNativeComponent, ToolCall, ToolChoice, ToolChoiceFunction,
    ToolChoiceFunctionName, ToolDefinition,
};
use super::resolve::{ProviderSurface, ProviderSurfaceDescriptor};
use super::response::{
    AnnotatedLlmResponse, ApiSpecificResponse, FinishReason, ResponseToolCall, Usage,
};
use super::traits::{LlmCodec, LlmResponseCodec};

const BEDROCK_PROVIDER: &str = "bedrock_converse";
const MAX_UPSTREAM_ERROR_BODY_BYTES: usize = 64 * 1024;
const MODELED_REQUEST_KEYS: &[&str] = &[
    "modelId",
    "messages",
    "system",
    "inferenceConfig",
    "toolConfig",
];

/// Built-in codec for Amazon Bedrock's model-independent Converse API.
pub struct BedrockConverseCodec;

pub(crate) const PROVIDER_SURFACE: ProviderSurfaceDescriptor = ProviderSurfaceDescriptor {
    surface: ProviderSurface::BedrockConverse,
    detect_request: |obj, hint| {
        let hinted = matches!(hint, Some("aws.bedrock.converse" | "bedrock.converse"));
        let converse_shape = [
            "messages",
            "system",
            "promptVariables",
            "inferenceConfig",
            "toolConfig",
            "additionalModelRequestFields",
        ]
        .into_iter()
        .any(|key| obj.contains_key(key));
        obj.contains_key("modelId") && (hinted || converse_shape)
    },
    detect_response: |obj| {
        obj.get("output")
            .and_then(|output| output.get("message"))
            .is_some_and(Json::is_object)
            && (obj.contains_key("stopReason")
                || obj.contains_key("usage")
                || obj.contains_key("metrics"))
    },
    decode_request: |request| BedrockConverseCodec.decode(request),
    decode_response: |raw| BedrockConverseCodec.decode_response(raw),
    codec_name: BEDROCK_PROVIDER,
    request_codec: || std::sync::Arc::new(BedrockConverseCodec),
    response_codec: || std::sync::Arc::new(BedrockConverseCodec),
    streaming_codec: || Box::new(BedrockConverseStreamingCodec::new()),
};

fn native_component(value: &Json) -> ProviderNativeComponent {
    ProviderNativeComponent {
        provider: BEDROCK_PROVIDER.into(),
        kind: value
            .as_object()
            .and_then(|obj| obj.keys().next())
            .cloned()
            .unwrap_or_else(|| "unknown".into()),
        value: value.clone(),
    }
}

fn required_object<'a>(
    value: &'a Json,
    context: &str,
) -> Result<&'a serde_json::Map<String, Json>> {
    value.as_object().ok_or_else(|| {
        FlowError::InvalidArgument(format!("Bedrock Converse {context} must be an object"))
    })
}

fn required_array<'a>(value: &'a Json, context: &str) -> Result<&'a [Json]> {
    value.as_array().map(Vec::as_slice).ok_or_else(|| {
        FlowError::InvalidArgument(format!("Bedrock Converse {context} must be an array"))
    })
}

fn required_string<'a>(
    obj: &'a serde_json::Map<String, Json>,
    key: &str,
    context: &str,
) -> Result<&'a str> {
    obj.get(key).and_then(Json::as_str).ok_or_else(|| {
        FlowError::InvalidArgument(format!(
            "Bedrock Converse {context} is missing string field {key}"
        ))
    })
}

fn provider_native_content(value: &Json) -> ContentPart {
    let native = native_component(value);
    ContentPart::ProviderNative {
        provider: native.provider,
        kind: native.kind,
        value: native.value,
    }
}

fn decode_content_block(value: &Json) -> Result<ContentPart> {
    let obj = required_object(value, "content block")?;
    let data_keys = [
        "text",
        "image",
        "audio",
        "document",
        "toolUse",
        "toolResult",
        "video",
        "guardContent",
        "cachePoint",
        "reasoningContent",
        "citationsContent",
        "searchResult",
        "toolAddition",
        "toolRemoval",
    ]
    .into_iter()
    .filter(|key| obj.contains_key(*key))
    .collect::<Vec<_>>();
    if data_keys.len() != 1 {
        return Ok(provider_native_content(value));
    }

    match data_keys[0] {
        "text" => {
            let text = required_string(obj, "text", "text content block")?;
            Ok(ContentPart::Text {
                text: text.into(),
                extra: obj
                    .iter()
                    .filter(|(key, _)| key.as_str() != "text")
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect(),
            })
        }
        "image" => Ok(ContentPart::Image {
            image: obj["image"].clone(),
            extra: obj
                .iter()
                .filter(|(key, _)| key.as_str() != "image")
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        }),
        "document" => Ok(ContentPart::File {
            file: obj["document"].clone(),
            extra: obj
                .iter()
                .filter(|(key, _)| key.as_str() != "document")
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        }),
        "toolUse" => {
            let tool = required_object(&obj["toolUse"], "toolUse block")?;
            Ok(ContentPart::ToolUse {
                id: required_string(tool, "toolUseId", "toolUse block")?.into(),
                name: required_string(tool, "name", "toolUse block")?.into(),
                input: tool.get("input").cloned().ok_or_else(|| {
                    FlowError::InvalidArgument(
                        "Bedrock Converse toolUse block is missing input".into(),
                    )
                })?,
                extra: tool
                    .iter()
                    .filter(|(key, _)| !matches!(key.as_str(), "toolUseId" | "name" | "input"))
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect(),
            })
        }
        "toolResult" => {
            let result = required_object(&obj["toolResult"], "toolResult block")?;
            let status = match result.get("status") {
                None | Some(Json::Null) => None,
                Some(Json::String(status)) if status == "success" => Some(false),
                Some(Json::String(status)) if status == "error" => Some(true),
                Some(_) => return Ok(provider_native_content(value)),
            };
            Ok(ContentPart::ToolResult {
                tool_use_id: required_string(result, "toolUseId", "toolResult block")?.into(),
                content: result.get("content").cloned().ok_or_else(|| {
                    FlowError::InvalidArgument(
                        "Bedrock Converse toolResult block is missing content".into(),
                    )
                })?,
                is_error: status,
                extra: result
                    .iter()
                    .filter(|(key, _)| !matches!(key.as_str(), "toolUseId" | "content" | "status"))
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect(),
            })
        }
        _ => Ok(provider_native_content(value)),
    }
}

fn decode_content(value: &Json, context: &str) -> Result<MessageContent> {
    let parts = required_array(value, context)?
        .iter()
        .map(decode_content_block)
        .collect::<Result<Vec<_>>>()?;
    Ok(MessageContent::Parts(parts))
}

fn encode_content_part(part: &ContentPart) -> Result<Json> {
    match part {
        ContentPart::Text { text, extra } => {
            let mut obj = extra.clone();
            obj.insert("text".into(), Json::String(text.clone()));
            Ok(Json::Object(obj))
        }
        ContentPart::Image { image, extra } => {
            let mut obj = extra.clone();
            obj.insert("image".into(), image.clone());
            Ok(Json::Object(obj))
        }
        ContentPart::File { file, extra } => {
            let mut obj = extra.clone();
            obj.insert("document".into(), file.clone());
            Ok(Json::Object(obj))
        }
        ContentPart::ToolUse {
            id,
            name,
            input,
            extra,
        } => {
            let mut tool = extra.clone();
            tool.insert("toolUseId".into(), Json::String(id.clone()));
            tool.insert("name".into(), Json::String(name.clone()));
            tool.insert("input".into(), input.clone());
            Ok(serde_json::json!({"toolUse": tool}))
        }
        ContentPart::ToolResult {
            tool_use_id,
            content,
            is_error,
            extra,
        } => {
            let mut result = extra.clone();
            result.insert("toolUseId".into(), Json::String(tool_use_id.clone()));
            result.insert("content".into(), content.clone());
            if let Some(is_error) = is_error {
                result.insert(
                    "status".into(),
                    Json::String(if *is_error { "error" } else { "success" }.into()),
                );
            }
            Ok(serde_json::json!({"toolResult": result}))
        }
        ContentPart::ProviderNative {
            provider, value, ..
        } if provider == BEDROCK_PROVIDER => Ok(value.clone()),
        other => Err(FlowError::InvalidArgument(format!(
            "content part {other:?} cannot be encoded for Bedrock Converse"
        ))),
    }
}

fn encode_content(content: &MessageContent) -> Result<Json> {
    let parts = match content {
        MessageContent::Text(text) => vec![serde_json::json!({"text": text})],
        MessageContent::Parts(parts) => parts
            .iter()
            .map(encode_content_part)
            .collect::<Result<Vec<_>>>()?,
    };
    Ok(Json::Array(parts))
}

fn decode_message(value: &Json) -> Result<Message> {
    let obj = required_object(value, "message")?;
    let role = required_string(obj, "role", "message")?;
    let content = obj
        .get("content")
        .ok_or_else(|| {
            FlowError::InvalidArgument("Bedrock Converse message is missing content".into())
        })
        .and_then(|value| decode_content(value, "message content"))?;

    if obj
        .keys()
        .any(|key| !matches!(key.as_str(), "role" | "content"))
    {
        return Ok(Message::ProviderNative {
            provider: BEDROCK_PROVIDER.into(),
            kind: role.into(),
            value: value.clone(),
        });
    }
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
        _ => Ok(Message::ProviderNative {
            provider: BEDROCK_PROVIDER.into(),
            kind: role.into(),
            value: value.clone(),
        }),
    }
}

fn encode_tool_call(call: &ToolCall) -> Result<Json> {
    if call.call_type != "function" {
        return Err(FlowError::InvalidArgument(format!(
            "Bedrock Converse cannot encode tool call type {}",
            call.call_type
        )));
    }
    let input = serde_json::from_str::<Json>(&call.function.arguments).map_err(|error| {
        FlowError::InvalidArgument(format!(
            "Bedrock Converse tool call {} has invalid JSON arguments: {error}",
            call.id
        ))
    })?;
    Ok(serde_json::json!({
        "toolUse": {
            "toolUseId": call.id,
            "name": call.function.name,
            "input": input,
        }
    }))
}

fn encode_message(message: &Message) -> Result<Json> {
    match message {
        Message::User {
            content,
            name: None,
        } => Ok(serde_json::json!({
            "role": "user",
            "content": encode_content(content)?,
        })),
        Message::Assistant {
            content,
            tool_calls,
            name: None,
        } => {
            let mut blocks = match content {
                Some(content) => match encode_content(content)? {
                    Json::Array(blocks) => blocks,
                    _ => unreachable!("Bedrock content encoder always returns an array"),
                },
                None => Vec::new(),
            };
            if let Some(calls) = tool_calls {
                blocks.extend(
                    calls
                        .iter()
                        .map(encode_tool_call)
                        .collect::<Result<Vec<_>>>()?,
                );
            }
            Ok(serde_json::json!({"role": "assistant", "content": blocks}))
        }
        Message::Tool {
            content,
            tool_call_id,
        } => Ok(serde_json::json!({
            "role": "user",
            "content": [{
                "toolResult": {
                    "toolUseId": tool_call_id,
                    "content": encode_content(content)?,
                }
            }],
        })),
        Message::ProviderNative {
            provider, value, ..
        } if provider == BEDROCK_PROVIDER => Ok(value.clone()),
        other => Err(FlowError::InvalidArgument(format!(
            "message {other:?} cannot be encoded for Bedrock Converse"
        ))),
    }
}

fn decode_tool(value: &Json) -> Result<ToolDefinition> {
    let obj = required_object(value, "tool definition")?;
    let Some(spec_value) = obj.get("toolSpec") else {
        let native = native_component(value);
        return Ok(ToolDefinition::ProviderNative {
            provider: native.provider,
            kind: native.kind,
            value: native.value,
        });
    };
    let spec = required_object(spec_value, "toolSpec")?;
    let parameters = spec
        .get("inputSchema")
        .and_then(Json::as_object)
        .and_then(|schema| schema.get("json"))
        .cloned();
    Ok(ToolDefinition::Function {
        function: FunctionDefinition {
            name: required_string(spec, "name", "toolSpec")?.into(),
            description: super::optional_string(spec, "description", "Bedrock Converse toolSpec")?,
            parameters,
            strict: super::optional_bool(spec, "strict", "Bedrock Converse toolSpec")?,
            extra: spec
                .iter()
                .filter(|(key, _)| {
                    !matches!(
                        key.as_str(),
                        "name" | "description" | "inputSchema" | "strict"
                    )
                })
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        },
        extra: obj
            .iter()
            .filter(|(key, _)| key.as_str() != "toolSpec")
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
    })
}

fn encode_tool(tool: &ToolDefinition) -> Result<Json> {
    match tool {
        ToolDefinition::Function { function, extra } => {
            let mut spec = function.extra.clone();
            spec.insert("name".into(), Json::String(function.name.clone()));
            if let Some(description) = &function.description {
                spec.insert("description".into(), Json::String(description.clone()));
            }
            if let Some(parameters) = &function.parameters {
                spec.insert(
                    "inputSchema".into(),
                    serde_json::json!({"json": parameters}),
                );
            }
            if let Some(strict) = function.strict {
                spec.insert("strict".into(), Json::Bool(strict));
            }
            let mut obj = extra.clone();
            obj.insert("toolSpec".into(), Json::Object(spec));
            Ok(Json::Object(obj))
        }
        ToolDefinition::ProviderNative {
            provider, value, ..
        } if provider == BEDROCK_PROVIDER => Ok(value.clone()),
        other => Err(FlowError::InvalidArgument(format!(
            "tool {other:?} cannot be encoded for Bedrock Converse"
        ))),
    }
}

fn decode_tool_choice(value: &Json) -> ToolChoice {
    let Some(obj) = value.as_object() else {
        return ToolChoice::ProviderNative(native_component(value));
    };
    if obj.len() == 1 && obj.get("auto").is_some_and(Json::is_object) {
        ToolChoice::Auto
    } else if obj.len() == 1 && obj.get("any").is_some_and(Json::is_object) {
        ToolChoice::Required
    } else if obj.len() == 1 {
        if let Some(name) = obj
            .get("tool")
            .and_then(Json::as_object)
            .and_then(|tool| tool.get("name"))
            .and_then(Json::as_str)
        {
            return ToolChoice::Specific(ToolChoiceFunction {
                choice_type: "function".into(),
                function: ToolChoiceFunctionName { name: name.into() },
            });
        }
        ToolChoice::ProviderNative(native_component(value))
    } else {
        ToolChoice::ProviderNative(native_component(value))
    }
}

fn encode_tool_choice(choice: &ToolChoice) -> Result<Json> {
    match choice {
        ToolChoice::Auto => Ok(serde_json::json!({"auto": {}})),
        ToolChoice::Required => Ok(serde_json::json!({"any": {}})),
        ToolChoice::Specific(choice) if choice.choice_type == "function" => {
            Ok(serde_json::json!({"tool": {"name": choice.function.name}}))
        }
        ToolChoice::ProviderNative(native) if native.provider == BEDROCK_PROVIDER => {
            Ok(native.value.clone())
        }
        ToolChoice::None => Err(FlowError::InvalidArgument(
            "Bedrock Converse has no tool choice equivalent for none".into(),
        )),
        other => Err(FlowError::InvalidArgument(format!(
            "tool choice {other:?} cannot be encoded for Bedrock Converse"
        ))),
    }
}

fn decode_generation_params(
    obj: &serde_json::Map<String, Json>,
) -> Result<Option<GenerationParams>> {
    let Some(config) = obj.get("inferenceConfig") else {
        return Ok(None);
    };
    if config.is_null() {
        return Ok(None);
    }
    let config = required_object(config, "inferenceConfig")?;
    let temperature =
        super::optional_f64(config, "temperature", "Bedrock Converse inferenceConfig")?;
    let max_tokens = super::optional_u64(config, "maxTokens", "Bedrock Converse inferenceConfig")?;
    let top_p = super::optional_f64(config, "topP", "Bedrock Converse inferenceConfig")?;
    let stop = config
        .get("stopSequences")
        .filter(|value| !value.is_null())
        .map(|value| {
            serde_json::from_value::<Vec<String>>(value.clone()).map_err(|error| {
                FlowError::InvalidArgument(format!(
                    "invalid Bedrock Converse inferenceConfig.stopSequences: {error}"
                ))
            })
        })
        .transpose()?;
    let params = GenerationParams {
        temperature,
        max_tokens,
        top_p,
        stop,
    };
    Ok((params != GenerationParams::default()).then_some(params))
}

fn json_number(value: f64, field: &str) -> Result<Json> {
    serde_json::Number::from_f64(value)
        .map(Json::Number)
        .ok_or_else(|| {
            FlowError::InvalidArgument(format!(
                "Bedrock Converse inferenceConfig.{field} must be finite"
            ))
        })
}

fn set_or_remove(obj: &mut serde_json::Map<String, Json>, key: &str, value: Option<Json>) {
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

fn validate_supported_fields(
    annotated: &AnnotatedLlmRequest,
    baseline: &AnnotatedLlmRequest,
) -> Result<()> {
    let unsupported_changed = annotated.store != baseline.store
        || annotated.previous_response_id != baseline.previous_response_id
        || annotated.truncation != baseline.truncation
        || annotated.reasoning != baseline.reasoning
        || annotated.include != baseline.include
        || annotated.user != baseline.user
        || annotated.metadata != baseline.metadata
        || annotated.service_tier != baseline.service_tier
        || annotated.parallel_tool_calls != baseline.parallel_tool_calls
        || annotated.max_output_tokens != baseline.max_output_tokens
        || annotated.max_tool_calls != baseline.max_tool_calls
        || annotated.top_logprobs != baseline.top_logprobs
        || annotated.stream != baseline.stream
        || annotated.api_specific != baseline.api_specific;
    if unsupported_changed {
        return Err(FlowError::InvalidArgument(
            "request contains fields that cannot be encoded for Bedrock Converse".into(),
        ));
    }
    Ok(())
}

fn validate_prompt_resource_request(obj: &serde_json::Map<String, Json>) -> Result<()> {
    let is_prompt_resource = obj
        .get("modelId")
        .and_then(Json::as_str)
        .is_some_and(|model| model.starts_with("arn:") && model.contains(":prompt/"));
    if !is_prompt_resource {
        return Ok(());
    }
    let has_forbidden_fields = [
        "additionalModelRequestFields",
        "inferenceConfig",
        "system",
        "toolConfig",
    ]
    .into_iter()
    .any(|key| obj.contains_key(key));
    if has_forbidden_fields {
        return Err(FlowError::InvalidArgument(
            "Bedrock Converse prompt resource ARNs cannot be combined with system, inferenceConfig, toolConfig, or additionalModelRequestFields"
                .into(),
        ));
    }
    Ok(())
}

impl LlmCodec for BedrockConverseCodec {
    fn codec_identity(&self) -> LlmCodecIdentity {
        LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::BedrockConverse)
    }

    fn decode(&self, request: &LlmRequest) -> Result<AnnotatedLlmRequest> {
        let obj = request.content.as_object().ok_or_else(|| {
            FlowError::InvalidArgument("Bedrock Converse request content must be an object".into())
        })?;
        let model =
            super::optional_string(obj, "modelId", "Bedrock Converse")?.ok_or_else(|| {
                FlowError::InvalidArgument("Bedrock Converse request is missing modelId".into())
            })?;
        let messages = obj
            .get("messages")
            .map(|value| {
                required_array(value, "messages")?
                    .iter()
                    .map(decode_message)
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default();
        let instructions = obj
            .get("system")
            .filter(|value| !value.is_null())
            .map(|value| decode_content(value, "system"))
            .transpose()?;
        let params = decode_generation_params(obj)?;

        let (tools, tool_choice) = match obj.get("toolConfig") {
            None | Some(Json::Null) => (None, None),
            Some(value) => {
                let config = required_object(value, "toolConfig")?;
                let tools = config
                    .get("tools")
                    .map(|value| {
                        required_array(value, "toolConfig.tools")?
                            .iter()
                            .map(decode_tool)
                            .collect::<Result<Vec<_>>>()
                    })
                    .transpose()?;
                let choice = config.get("toolChoice").map(decode_tool_choice);
                (tools, choice)
            }
        };

        let extra = obj
            .iter()
            .filter(|(key, _)| !MODELED_REQUEST_KEYS.contains(&key.as_str()))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        Ok(AnnotatedLlmRequest {
            messages,
            instructions,
            model: Some(model),
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
            api_specific: None,
            extra,
        })
    }

    fn encode(&self, annotated: &AnnotatedLlmRequest, original: &LlmRequest) -> Result<LlmRequest> {
        let baseline = self.decode(original)?;
        validate_supported_fields(annotated, &baseline)?;
        let mut content = original.content.clone();
        let obj = content.as_object_mut().ok_or_else(|| {
            FlowError::InvalidArgument("Bedrock Converse request content must be an object".into())
        })?;

        if annotated.model != baseline.model {
            set_or_remove(obj, "modelId", annotated.model.clone().map(Json::String));
        }
        if annotated.messages != baseline.messages {
            let messages = super::encode_changed_items(
                &annotated.messages,
                &baseline.messages,
                obj.get("messages")
                    .and_then(Json::as_array)
                    .map(Vec::as_slice),
                encode_message,
            )?;
            obj.insert("messages".into(), Json::Array(messages));
        }
        if annotated.instructions != baseline.instructions {
            let encoded = annotated
                .instructions
                .as_ref()
                .map(encode_content)
                .transpose()?;
            let encoded = match (
                obj.get("system"),
                baseline
                    .instructions
                    .as_ref()
                    .map(encode_content)
                    .transpose()?,
                encoded,
            ) {
                (Some(original), Some(before), Some(edited)) => {
                    Some(super::patch_changed_json(original, &before, &edited)?)
                }
                (_, _, edited) => edited,
            };
            set_or_remove(obj, "system", encoded);
        }
        if annotated.params != baseline.params {
            patch_inference_config(obj, annotated.params.as_ref(), baseline.params.as_ref())?;
        }
        if annotated.tools != baseline.tools || annotated.tool_choice != baseline.tool_choice {
            patch_tool_config(obj, annotated, &baseline)?;
        }
        patch_extra_fields(obj, &baseline.extra, &annotated.extra);
        // Validate the final provider envelope, not only normalized fields:
        // unknown members inside an otherwise-unmodeled inferenceConfig or
        // toolConfig are still forbidden by Bedrock prompt resources.
        validate_prompt_resource_request(obj)?;

        Ok(LlmRequest {
            headers: original.headers.clone(),
            content,
        })
    }
}

fn patch_inference_config(
    obj: &mut serde_json::Map<String, Json>,
    edited: Option<&GenerationParams>,
    baseline: Option<&GenerationParams>,
) -> Result<()> {
    let mut config = match obj.get("inferenceConfig") {
        Some(Json::Object(config)) => config.clone(),
        Some(Json::Null) | None => serde_json::Map::new(),
        Some(_) => {
            return Err(FlowError::InvalidArgument(
                "Bedrock Converse inferenceConfig must be an object".into(),
            ));
        }
    };
    let fields = [
        (
            "temperature",
            edited.and_then(|params| params.temperature),
            baseline.and_then(|params| params.temperature),
        ),
        (
            "topP",
            edited.and_then(|params| params.top_p),
            baseline.and_then(|params| params.top_p),
        ),
    ];
    for (key, value, before) in fields {
        if value != before {
            set_or_remove(
                &mut config,
                key,
                value.map(|value| json_number(value, key)).transpose()?,
            );
        }
    }
    let max_tokens = edited.and_then(|params| params.max_tokens);
    if max_tokens != baseline.and_then(|params| params.max_tokens) {
        set_or_remove(&mut config, "maxTokens", max_tokens.map(Json::from));
    }
    let stop = edited.and_then(|params| params.stop.as_ref());
    if stop != baseline.and_then(|params| params.stop.as_ref()) {
        set_or_remove(
            &mut config,
            "stopSequences",
            stop.map(|values| serde_json::json!(values)),
        );
    }
    if config.is_empty() {
        obj.remove("inferenceConfig");
    } else {
        obj.insert("inferenceConfig".into(), Json::Object(config));
    }
    Ok(())
}

fn patch_tool_config(
    obj: &mut serde_json::Map<String, Json>,
    annotated: &AnnotatedLlmRequest,
    baseline: &AnnotatedLlmRequest,
) -> Result<()> {
    let mut config = match obj.get("toolConfig") {
        Some(Json::Object(config)) => config.clone(),
        Some(Json::Null) | None => serde_json::Map::new(),
        Some(_) => {
            return Err(FlowError::InvalidArgument(
                "Bedrock Converse toolConfig must be an object".into(),
            ));
        }
    };
    if annotated.tools != baseline.tools {
        let tools = annotated
            .tools
            .as_deref()
            .map(|tools| {
                super::encode_changed_items(
                    tools,
                    baseline.tools.as_deref().unwrap_or(&[]),
                    config
                        .get("tools")
                        .and_then(Json::as_array)
                        .map(Vec::as_slice),
                    encode_tool,
                )
            })
            .transpose()?
            .map(Json::Array);
        set_or_remove(&mut config, "tools", tools);
    }
    if annotated.tool_choice != baseline.tool_choice {
        let edited = annotated
            .tool_choice
            .as_ref()
            .map(encode_tool_choice)
            .transpose()?;
        let before = baseline
            .tool_choice
            .as_ref()
            .map(encode_tool_choice)
            .transpose()?;
        let edited = match (config.get("toolChoice"), before, edited) {
            (Some(original), Some(before), Some(edited)) => {
                Some(super::patch_changed_json(original, &before, &edited)?)
            }
            (_, _, edited) => edited,
        };
        set_or_remove(&mut config, "toolChoice", edited);
    }
    if config.is_empty() {
        obj.remove("toolConfig");
    } else {
        obj.insert("toolConfig".into(), Json::Object(config));
    }
    Ok(())
}

fn map_stop_reason(reason: &str) -> FinishReason {
    match reason {
        "end_turn" | "stop_sequence" => FinishReason::Complete,
        "max_tokens" => FinishReason::Length,
        "tool_use" => FinishReason::ToolUse,
        "guardrail_intervened" | "content_filtered" => FinishReason::ContentFilter,
        other => FinishReason::Unknown(other.into()),
    }
}

fn response_tool_calls(content: &[Json]) -> Result<Option<Vec<ResponseToolCall>>> {
    let calls = content
        .iter()
        .filter_map(|block| block.get("toolUse"))
        .map(|value| {
            let tool = required_object(value, "response toolUse block")?;
            Ok(ResponseToolCall {
                id: required_string(tool, "toolUseId", "response toolUse block")?.into(),
                name: required_string(tool, "name", "response toolUse block")?.into(),
                arguments: tool.get("input").cloned().ok_or_else(|| {
                    FlowError::InvalidArgument(
                        "Bedrock Converse response toolUse block is missing input".into(),
                    )
                })?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((!calls.is_empty()).then_some(calls))
}

fn decode_usage(value: Option<&Json>) -> Result<Option<Usage>> {
    let Some(value) = value else { return Ok(None) };
    let obj = required_object(value, "response usage")?;
    let usage = Usage {
        prompt_tokens: super::optional_u64(obj, "inputTokens", "Bedrock Converse usage")?,
        completion_tokens: super::optional_u64(obj, "outputTokens", "Bedrock Converse usage")?,
        total_tokens: super::optional_u64(obj, "totalTokens", "Bedrock Converse usage")?,
        cache_read_tokens: super::optional_u64(
            obj,
            "cacheReadInputTokens",
            "Bedrock Converse usage",
        )?,
        cache_write_tokens: super::optional_u64(
            obj,
            "cacheWriteInputTokens",
            "Bedrock Converse usage",
        )?,
        cost: None,
    };
    Ok(Some(usage))
}

impl LlmResponseCodec for BedrockConverseCodec {
    fn codec_identity(&self) -> LlmCodecIdentity {
        LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::BedrockConverse)
    }

    fn decode_response(&self, response: &Json) -> Result<AnnotatedLlmResponse> {
        let obj = required_object(response, "response")?;
        let output = obj.get("output").ok_or_else(|| {
            FlowError::InvalidArgument("Bedrock Converse response is missing output".into())
        })?;
        let message = required_object(output, "response output")?
            .get("message")
            .ok_or_else(|| {
                FlowError::InvalidArgument(
                    "Bedrock Converse response output is missing message".into(),
                )
            })?;
        let message = required_object(message, "response message")?;
        let content = message
            .get("content")
            .map(|value| required_array(value, "response message content"))
            .transpose()?
            .unwrap_or_default();
        let normalized_message = (!content.is_empty())
            .then(|| {
                content
                    .iter()
                    .map(decode_content_block)
                    .collect::<Result<Vec<_>>>()
                    .map(MessageContent::Parts)
            })
            .transpose()?;
        let tool_calls = response_tool_calls(content)?;
        let finish_reason = obj
            .get("stopReason")
            .filter(|value| !value.is_null())
            .map(|value| {
                value.as_str().map(map_stop_reason).ok_or_else(|| {
                    FlowError::InvalidArgument(
                        "Bedrock Converse response stopReason must be a string".into(),
                    )
                })
            })
            .transpose()?;
        let usage = decode_usage(obj.get("usage"))?;
        let extra = obj
            .iter()
            .filter(|(key, _)| !matches!(key.as_str(), "output" | "stopReason" | "usage"))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        let mut api_specific_data = serde_json::Map::new();
        if let Some(stop_reason) = obj.get("stopReason") {
            api_specific_data.insert("stopReason".into(), stop_reason.clone());
        }
        // Keep the complete usage object so newly added Bedrock counters
        // (for example cacheDetails) are not lost when the common Usage shape
        // extracts only the counters Relay understands.
        if let Some(usage) = obj.get("usage") {
            api_specific_data.insert("usage".into(), usage.clone());
        }
        let api_specific = (!api_specific_data.is_empty()).then(|| ApiSpecificResponse::Custom {
            api_name: BEDROCK_PROVIDER.into(),
            data: Json::Object(api_specific_data),
        });

        Ok(AnnotatedLlmResponse {
            id: None,
            // Converse responses do not identify the invoked model. Integrations
            // must provide the request model through the managed LLM call.
            model: None,
            message: normalized_message,
            tool_calls,
            finish_reason,
            usage,
            optimization_summary: None,
            api_specific,
            extra,
        })
    }
}

/// Stateful accumulator for decoded `ConverseStream` event dictionaries.
pub struct BedrockConverseStreamingCodec {
    state: std::sync::Arc<std::sync::Mutex<BedrockConverseStreamingState>>,
}

impl BedrockConverseStreamingCodec {
    /// Creates an empty, single-use ConverseStream accumulator.
    pub fn new() -> Self {
        Self {
            state: std::sync::Arc::new(std::sync::Mutex::new(
                BedrockConverseStreamingState::default(),
            )),
        }
    }
}

impl Default for BedrockConverseStreamingCodec {
    fn default() -> Self {
        Self::new()
    }
}

impl super::streaming::StreamingCodec for BedrockConverseStreamingCodec {
    fn collector(&self) -> crate::api::runtime::LlmCollectorFn {
        let state = std::sync::Arc::clone(&self.state);
        Box::new(move |event: Json| -> Result<()> {
            state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .observe(&event)
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

#[derive(Debug)]
struct BedrockConverseStreamingState {
    aggregate: Option<Json>,
    aggregation_supported: bool,
    role: Option<String>,
    blocks: std::collections::BTreeMap<u64, StreamingBlock>,
    stop_reason: Option<String>,
    usage: Option<Json>,
    metrics: Option<Json>,
    extra: serde_json::Map<String, Json>,
}

impl Default for BedrockConverseStreamingState {
    fn default() -> Self {
        Self {
            aggregate: None,
            aggregation_supported: true,
            role: None,
            blocks: std::collections::BTreeMap::new(),
            stop_reason: None,
            usage: None,
            metrics: None,
            extra: serde_json::Map::new(),
        }
    }
}

#[derive(Debug, Default)]
struct StreamingBlock {
    value: serde_json::Map<String, Json>,
    text: String,
    has_text: bool,
    tool_input: String,
    has_tool_input: bool,
}

fn bedrock_stream_failure(kind: &str, payload: &Json) -> FlowError {
    let payload = payload.as_object();
    let default_message = "provider stream exception";
    let message = payload
        .and_then(|payload| payload.get("message"))
        .and_then(Json::as_str)
        .unwrap_or(default_message);
    let original_message = payload
        .and_then(|payload| payload.get("originalMessage"))
        .and_then(Json::as_str);
    let body = match original_message.filter(|original| *original != message) {
        Some(original) => {
            format!("Bedrock ConverseStream {kind}: {message}; original: {original}")
        }
        None => format!("Bedrock ConverseStream {kind}: {message}"),
    };
    let body = bounded_upstream_error_body(body);
    let (default_status, class) = match kind {
        "throttlingException" => (Some(429), UpstreamFailureClass::RetryableStatus),
        "serviceUnavailableException" => (Some(503), UpstreamFailureClass::ModelUnavailable),
        "internalServerException" => (Some(500), UpstreamFailureClass::RetryableStatus),
        "modelStreamErrorException" => (Some(424), UpstreamFailureClass::RetryableStatus),
        "modelTimeoutException" => (Some(408), UpstreamFailureClass::Timeout),
        "validationException" => (Some(400), UpstreamFailureClass::InvalidRequest),
        "accessDeniedException" => (Some(403), UpstreamFailureClass::Authentication),
        _ => (None, UpstreamFailureClass::Other),
    };
    let status = payload
        .and_then(|payload| {
            payload
                .get("originalStatusCode")
                .or_else(|| payload.get("statusCode"))
        })
        .and_then(Json::as_u64)
        .and_then(|status| u16::try_from(status).ok())
        .or(default_status);
    FlowError::Upstream(UpstreamFailure {
        status,
        body,
        headers: std::collections::BTreeMap::new(),
        class,
    })
}

fn bounded_upstream_error_body(mut body: String) -> String {
    if body.len() <= MAX_UPSTREAM_ERROR_BODY_BYTES {
        return body;
    }
    let mut end = MAX_UPSTREAM_ERROR_BODY_BYTES;
    while !body.is_char_boundary(end) {
        end -= 1;
    }
    body.truncate(end);
    body
}

impl BedrockConverseStreamingState {
    fn observe(&mut self, event: &Json) -> Result<()> {
        let obj = required_object(event, "stream event")?;
        let member_count = obj
            .keys()
            .filter(|key| {
                matches!(
                    key.as_str(),
                    "output"
                        | "messageStart"
                        | "contentBlockStart"
                        | "contentBlockDelta"
                        | "contentBlockStop"
                        | "messageStop"
                        | "metadata"
                ) || key.ends_with("Exception")
            })
            .count();
        if member_count > 1 {
            return Err(FlowError::InvalidArgument(
                "Bedrock ConverseStream event contains multiple recognized union members".into(),
            ));
        }
        if member_count == 0 {
            // AWS SDKs intentionally surface future union members rather than
            // discarding them. Forward the raw stream item unchanged, but do
            // not synthesize a misleading partial final response.
            self.mark_aggregation_unsupported("unknown top-level stream union member");
            return Ok(());
        }
        if let Some((kind, payload)) = obj.iter().find(|(key, _)| key.ends_with("Exception")) {
            return Err(bedrock_stream_failure(kind, payload));
        }
        if !self.aggregation_supported {
            return Ok(());
        }
        if obj.contains_key("output") {
            self.aggregate = Some(event.clone());
            return Ok(());
        }
        if let Some(start) = obj.get("messageStart") {
            let start = required_object(start, "messageStart event")?;
            if let Some(role) = start.get("role") {
                self.role = Some(
                    role.as_str()
                        .ok_or_else(|| {
                            FlowError::InvalidArgument(
                                "Bedrock ConverseStream messageStart.role must be a string".into(),
                            )
                        })?
                        .into(),
                );
            }
        }
        if let Some(start) = obj.get("contentBlockStart") {
            self.observe_block_start(start)?;
        }
        if let Some(delta) = obj.get("contentBlockDelta") {
            self.observe_block_delta(delta)?;
        }
        if let Some(stop) = obj.get("contentBlockStop") {
            self.observe_block_stop(stop)?;
        }
        if let Some(stop) = obj.get("messageStop") {
            if self.aggregation_supported && !self.tool_inputs_are_valid() {
                self.mark_aggregation_unsupported("toolUse.input is not complete JSON");
            }
            let stop = required_object(stop, "messageStop event")?;
            if let Some(reason) = stop.get("stopReason") {
                self.stop_reason = Some(
                    reason
                        .as_str()
                        .ok_or_else(|| {
                            FlowError::InvalidArgument(
                                "Bedrock ConverseStream messageStop.stopReason must be a string"
                                    .into(),
                            )
                        })?
                        .into(),
                );
            }
            for (key, value) in stop.iter().filter(|(key, _)| key.as_str() != "stopReason") {
                self.extra.insert(key.clone(), value.clone());
            }
        }
        if let Some(metadata) = obj.get("metadata") {
            let metadata = required_object(metadata, "metadata event")?;
            if let Some(usage) = metadata.get("usage") {
                self.usage = Some(usage.clone());
            }
            if let Some(metrics) = metadata.get("metrics") {
                self.metrics = Some(metrics.clone());
            }
            for (key, value) in metadata
                .iter()
                .filter(|(key, _)| !matches!(key.as_str(), "usage" | "metrics"))
            {
                self.extra.insert(key.clone(), value.clone());
            }
        }
        Ok(())
    }

    fn mark_aggregation_unsupported(&mut self, reason: &str) {
        if self.aggregation_supported {
            log::warn!(
                target: "nemo_relay.codec",
                event = "bedrock_stream_aggregation_unsupported",
                provider = BEDROCK_PROVIDER,
                reason = reason;
                "omitting normalized ConverseStream response for unsupported content"
            );
        }
        self.aggregation_supported = false;
        self.aggregate = None;
        self.role = None;
        self.blocks.clear();
        self.stop_reason = None;
        self.usage = None;
        self.metrics = None;
        self.extra.clear();
    }

    fn tool_inputs_are_valid(&self) -> bool {
        for block in self.blocks.values() {
            if block.has_tool_input && serde_json::from_str::<Json>(&block.tool_input).is_err() {
                return false;
            }
        }
        true
    }

    fn block_mut(&mut self, index: u64) -> &mut StreamingBlock {
        self.blocks.entry(index).or_default()
    }

    fn observe_block_start(&mut self, event: &Json) -> Result<()> {
        let event = required_object(event, "contentBlockStart event")?;
        let index = super::optional_u64(
            event,
            "contentBlockIndex",
            "Bedrock ConverseStream contentBlockStart",
        )?
        .ok_or_else(|| {
            FlowError::InvalidArgument(
                "Bedrock ConverseStream contentBlockStart is missing contentBlockIndex".into(),
            )
        })?;
        let start = event
            .get("start")
            .map(|value| required_object(value, "contentBlockStart.start"))
            .transpose()?
            .cloned()
            .unwrap_or_default();
        if start.keys().any(|key| !matches!(key.as_str(), "toolUse")) {
            self.mark_aggregation_unsupported("unsupported contentBlockStart union member");
            return Ok(());
        }
        self.block_mut(index).value = start;
        Ok(())
    }

    fn observe_block_delta(&mut self, event: &Json) -> Result<()> {
        let event = required_object(event, "contentBlockDelta event")?;
        let index = super::optional_u64(
            event,
            "contentBlockIndex",
            "Bedrock ConverseStream contentBlockDelta",
        )?
        .ok_or_else(|| {
            FlowError::InvalidArgument(
                "Bedrock ConverseStream contentBlockDelta is missing contentBlockIndex".into(),
            )
        })?;
        let delta = event
            .get("delta")
            .ok_or_else(|| {
                FlowError::InvalidArgument(
                    "Bedrock ConverseStream contentBlockDelta is missing delta".into(),
                )
            })
            .and_then(|value| required_object(value, "contentBlockDelta.delta"))?;
        let delta_members = delta
            .keys()
            .filter(|key| {
                matches!(
                    key.as_str(),
                    "text" | "toolUse" | "toolResult" | "reasoningContent" | "citation" | "image"
                )
            })
            .count();
        if delta_members != 1 {
            self.mark_aggregation_unsupported("unknown or malformed contentBlockDelta union");
            return Ok(());
        }
        if delta
            .keys()
            .any(|key| !matches!(key.as_str(), "text" | "toolUse"))
        {
            self.mark_aggregation_unsupported("unsupported contentBlockDelta union member");
            return Ok(());
        }
        let block = self.block_mut(index);
        if let Some(text) = delta.get("text") {
            block.text.push_str(text.as_str().ok_or_else(|| {
                FlowError::InvalidArgument(
                    "Bedrock ConverseStream text delta must be a string".into(),
                )
            })?);
            block.has_text = true;
        }
        if let Some(tool_delta) = delta.get("toolUse") {
            let tool_delta = required_object(tool_delta, "toolUse delta")?;
            if let Some(input) = tool_delta.get("input") {
                block.tool_input.push_str(input.as_str().ok_or_else(|| {
                    FlowError::InvalidArgument(
                        "Bedrock ConverseStream toolUse.input delta must be a string".into(),
                    )
                })?);
                block.has_tool_input = true;
            }
        }
        Ok(())
    }

    fn observe_block_stop(&mut self, event: &Json) -> Result<()> {
        let event = required_object(event, "contentBlockStop event")?;
        let index = super::optional_u64(
            event,
            "contentBlockIndex",
            "Bedrock ConverseStream contentBlockStop",
        )?
        .ok_or_else(|| {
            FlowError::InvalidArgument(
                "Bedrock ConverseStream contentBlockStop is missing contentBlockIndex".into(),
            )
        })?;
        if !self.aggregation_supported {
            return Ok(());
        }
        let block = self.blocks.get(&index).ok_or_else(|| {
            FlowError::InvalidArgument(format!(
                "Bedrock ConverseStream contentBlockStop references unknown block {index}"
            ))
        })?;
        if block.has_tool_input && serde_json::from_str::<Json>(&block.tool_input).is_err() {
            self.mark_aggregation_unsupported("toolUse.input is not complete JSON");
        }
        Ok(())
    }

    fn finalize(self) -> Json {
        if !self.aggregation_supported {
            return Json::Null;
        }
        if let Some(aggregate) = self.aggregate {
            return aggregate;
        }
        let content = self
            .blocks
            .into_values()
            .map(StreamingBlock::finalize)
            .collect::<Vec<_>>();
        let mut output = self.extra;
        output.insert(
            "output".into(),
            serde_json::json!({
                "message": {
                    "role": self.role.unwrap_or_else(|| "assistant".into()),
                    "content": content,
                }
            }),
        );
        if let Some(stop_reason) = self.stop_reason {
            output.insert("stopReason".into(), Json::String(stop_reason));
        }
        if let Some(usage) = self.usage {
            output.insert("usage".into(), usage);
        }
        if let Some(metrics) = self.metrics {
            output.insert("metrics".into(), metrics);
        }
        Json::Object(output)
    }
}

impl StreamingBlock {
    fn finalize(mut self) -> Json {
        if self.has_text {
            self.value.insert("text".into(), Json::String(self.text));
        }
        if self.has_tool_input {
            let input = serde_json::from_str::<Json>(&self.tool_input)
                .unwrap_or(Json::String(self.tool_input));
            let tool = self
                .value
                .entry("toolUse")
                .or_insert_with(|| Json::Object(serde_json::Map::new()));
            if let Some(tool) = tool.as_object_mut() {
                tool.insert("input".into(), input);
            }
        }
        Json::Object(self.value)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/codec/bedrock_converse_tests.rs"]
mod tests;
