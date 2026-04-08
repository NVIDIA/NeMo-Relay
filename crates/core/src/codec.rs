// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LLM request codec types and trait.
//!
//! This module defines the [`AnnotatedLLMRequest`] type system for structured
//! LLM request representation and the [`LlmCodec`] trait for bidirectional
//! translation between opaque [`LLMRequest`] payloads and typed form.

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::json::Json;
use crate::types::LLMRequest;

// ---------------------------------------------------------------------------
// AnnotatedLLMRequest type hierarchy
// ---------------------------------------------------------------------------

/// Structured view of an LLM request, produced by a Codec from opaque
/// [`LLMRequest`](crate::types::LLMRequest) content.
///
/// The `extra` field captures any provider-specific keys not modeled by the
/// known fields, ensuring lossless round-trip through `decode`/`encode`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnnotatedLLMRequest {
    /// Parsed conversation messages.
    pub messages: Vec<Message>,
    /// Model identifier (e.g., `"gpt-4"`, `"claude-sonnet-4-20250514"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Common generation parameters, normalized.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<GenerationParams>,
    /// Tool definitions (function schemas) available to the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,
    /// Tool choice control.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    /// Extensible key-value pairs for unmodeled provider-specific fields.
    /// Merged back into the request body during encode via `serde(flatten)`.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Json>,
}

/// A single message in a conversation, tagged by role.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    /// A system instruction message.
    System {
        /// The message content.
        content: MessageContent,
        /// Optional sender name.
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    /// A user message.
    User {
        /// The message content.
        content: MessageContent,
        /// Optional sender name.
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    /// An assistant response, optionally containing tool calls.
    Assistant {
        /// The message content (optional — may be absent when tool calls are present).
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<MessageContent>,
        /// Tool calls requested by the assistant.
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<ToolCall>>,
        /// Optional sender name.
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    /// A tool result message.
    Tool {
        /// The tool execution result.
        content: MessageContent,
        /// The ID of the tool call this result corresponds to.
        tool_call_id: String,
    },
}

/// Message content: either a plain string or multimodal parts array.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Plain text content.
    Text(String),
    /// Multimodal content parts.
    Parts(Vec<ContentPart>),
}

/// A single content part within a multimodal message.
///
/// v1 supports text only. Future versions may add `ImageUrl`, `Audio`, etc.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// A text content part.
    Text {
        /// The text content.
        text: String,
    },
}

/// A tool call requested by the assistant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique identifier for this tool call.
    pub id: String,
    /// The type of tool call (typically `"function"`).
    #[serde(rename = "type")]
    pub call_type: String,
    /// The function to call.
    pub function: FunctionCall,
}

/// A function call within a tool call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionCall {
    /// The name of the function to call.
    pub name: String,
    /// The function arguments as a JSON string (per OpenAI convention).
    pub arguments: String,
}

/// A tool definition (function schema) available to the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// The type of tool (typically `"function"`).
    #[serde(rename = "type")]
    pub tool_type: String,
    /// The function definition.
    pub function: FunctionDefinition,
}

/// A function definition within a tool definition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionDefinition {
    /// The name of the function.
    pub name: String,
    /// A description of what the function does.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The JSON Schema for the function parameters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Json>,
}

/// Tool choice control: how the model should use available tools.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoice {
    /// Let the model decide whether to call a tool.
    Auto,
    /// Do not call any tools.
    None,
    /// The model must call at least one tool.
    Required,
    /// Force a specific function by name.
    #[serde(untagged)]
    Specific(ToolChoiceFunction),
}

/// A specific tool choice that forces a named function.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolChoiceFunction {
    /// The type (typically `"function"`).
    #[serde(rename = "type")]
    pub choice_type: String,
    /// The function to call.
    pub function: ToolChoiceFunctionName,
}

/// The name component of a specific tool choice.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolChoiceFunctionName {
    /// The function name.
    pub name: String,
}

/// Normalized generation parameters across providers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct GenerationParams {
    /// Sampling temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Maximum number of tokens to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    /// Nucleus sampling probability.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Stop sequences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
}

// ---------------------------------------------------------------------------
// LlmCodec trait
// ---------------------------------------------------------------------------

/// A bidirectional translator between opaque [`LLMRequest`] content and
/// structured [`AnnotatedLLMRequest`].
///
/// Codecs are implemented by integration patches (LangChain, LangChain-NVIDIA,
/// LangGraph, etc.) since each SDK has its own request format. They are
/// registered by name in the global codec registry.
///
/// # Design
///
/// - **Synchronous**: `decode`/`encode` are pure data transforms (JSON
///   restructuring), not I/O operations. This matches existing guardrails
///   and request intercepts.
/// - **`Send + Sync`**: Required because [`NatNexusContextState`](crate::context::NatNexusContextState)
///   is behind `Arc<RwLock<>>` and accessed from async contexts.
/// - **Trait object**: Codecs are registered at runtime (e.g., by Python
///   patches), so the Rust core cannot know concrete types at compile time.
///   Store as `Arc<dyn LlmCodec>`.
pub trait LlmCodec: Send + Sync {
    /// Parse opaque request content into structured form.
    fn decode(&self, request: &LLMRequest) -> Result<AnnotatedLLMRequest>;

    /// Merge structured changes back into the opaque request.
    ///
    /// The `original` parameter is the pre-intercept [`LLMRequest`], used to
    /// preserve fields that the Codec does not structurally model. Implementations
    /// MUST use merge-not-replace semantics: overlay structured changes onto
    /// the original content, do not construct a fresh content object.
    fn encode(&self, annotated: &AnnotatedLLMRequest, original: &LLMRequest) -> Result<LLMRequest>;
}

// ---------------------------------------------------------------------------
// Helper methods
// ---------------------------------------------------------------------------

impl AnnotatedLLMRequest {
    /// Extract the text content of the first system message, if any.
    ///
    /// For [`MessageContent::Text`], returns the string directly.
    /// For [`MessageContent::Parts`], returns the text of the first
    /// [`ContentPart::Text`] part.
    pub fn system_prompt(&self) -> Option<&str> {
        self.messages.iter().find_map(|m| match m {
            Message::System { content, .. } => match content {
                MessageContent::Text(s) => Some(s.as_str()),
                MessageContent::Parts(parts) => parts
                    .iter()
                    .map(|p| {
                        let ContentPart::Text { text } = p;
                        text.as_str()
                    })
                    .next(),
            },
            _ => None,
        })
    }

    /// Get the text content of the last user message, if any.
    ///
    /// Searches messages in reverse order and returns the first user
    /// message found. For [`MessageContent::Parts`], returns the text of
    /// the first [`ContentPart::Text`] part.
    pub fn last_user_message(&self) -> Option<&str> {
        self.messages.iter().rev().find_map(|m| match m {
            Message::User { content, .. } => match content {
                MessageContent::Text(s) => Some(s.as_str()),
                MessageContent::Parts(parts) => parts
                    .iter()
                    .map(|p| {
                        let ContentPart::Text { text } = p;
                        text.as_str()
                    })
                    .next(),
            },
            _ => None,
        })
    }

    /// Check if any assistant message in the conversation contains tool calls.
    ///
    /// Returns `true` if at least one [`Message::Assistant`] variant has a
    /// non-empty `tool_calls` field.
    pub fn has_tool_calls(&self) -> bool {
        self.messages.iter().any(|m| {
            matches!(
                m,
                Message::Assistant { tool_calls: Some(calls), .. } if !calls.is_empty()
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -------------------------------------------------------------------
    // AnnotatedLLMRequest serialization round-trip
    // -------------------------------------------------------------------

    #[test]
    fn test_annotated_llm_request_round_trip() {
        let req = AnnotatedLLMRequest {
            messages: vec![Message::User {
                content: MessageContent::Text("Hello".into()),
                name: None,
            }],
            model: Some("gpt-4".into()),
            params: Some(GenerationParams {
                temperature: Some(0.7),
                max_tokens: Some(100),
                top_p: None,
                stop: None,
            }),
            tools: None,
            tool_choice: None,
            extra: serde_json::Map::new(),
        };
        let json_val = serde_json::to_value(&req).unwrap();
        let deserialized: AnnotatedLLMRequest = serde_json::from_value(json_val).unwrap();
        assert_eq!(req, deserialized);
    }

    // -------------------------------------------------------------------
    // Message role serialization
    // -------------------------------------------------------------------

    #[test]
    fn test_message_system_serialization() {
        let msg = Message::System {
            content: MessageContent::Text("Be helpful".into()),
            name: None,
        };
        let json_val = serde_json::to_value(&msg).unwrap();
        assert_eq!(json_val, json!({"role": "system", "content": "Be helpful"}));
        let deserialized: Message = serde_json::from_value(json_val).unwrap();
        assert_eq!(msg, deserialized);
    }

    #[test]
    fn test_message_user_serialization() {
        let msg = Message::User {
            content: MessageContent::Text("Hello".into()),
            name: None,
        };
        let json_val = serde_json::to_value(&msg).unwrap();
        assert_eq!(json_val, json!({"role": "user", "content": "Hello"}));
        let deserialized: Message = serde_json::from_value(json_val).unwrap();
        assert_eq!(msg, deserialized);
    }

    #[test]
    fn test_message_assistant_with_tool_calls() {
        let msg = Message::Assistant {
            content: None,
            tool_calls: Some(vec![ToolCall {
                id: "call_123".into(),
                call_type: "function".into(),
                function: FunctionCall {
                    name: "get_weather".into(),
                    arguments: r#"{"city":"NYC"}"#.into(),
                },
            }]),
            name: None,
        };
        let json_val = serde_json::to_value(&msg).unwrap();
        assert_eq!(
            json_val,
            json!({
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"city\":\"NYC\"}"
                    }
                }]
            })
        );
        let deserialized: Message = serde_json::from_value(json_val).unwrap();
        assert_eq!(msg, deserialized);
    }

    #[test]
    fn test_message_tool_serialization() {
        let msg = Message::Tool {
            content: MessageContent::Text("72F, sunny".into()),
            tool_call_id: "call_123".into(),
        };
        let json_val = serde_json::to_value(&msg).unwrap();
        assert_eq!(
            json_val,
            json!({"role": "tool", "content": "72F, sunny", "tool_call_id": "call_123"})
        );
        let deserialized: Message = serde_json::from_value(json_val).unwrap();
        assert_eq!(msg, deserialized);
    }

    // -------------------------------------------------------------------
    // MessageContent serialization
    // -------------------------------------------------------------------

    #[test]
    fn test_message_content_text_serialization() {
        let content = MessageContent::Text("hello".into());
        let json_val = serde_json::to_value(&content).unwrap();
        assert_eq!(json_val, json!("hello"));
    }

    #[test]
    fn test_message_content_parts_serialization() {
        let content = MessageContent::Parts(vec![ContentPart::Text {
            text: "Hello world".into(),
        }]);
        let json_val = serde_json::to_value(&content).unwrap();
        assert_eq!(json_val, json!([{"type": "text", "text": "Hello world"}]));
    }

    // -------------------------------------------------------------------
    // ToolCall serialization
    // -------------------------------------------------------------------

    #[test]
    fn test_tool_call_serialization() {
        let tc = ToolCall {
            id: "tc_1".into(),
            call_type: "function".into(),
            function: FunctionCall {
                name: "search".into(),
                arguments: r#"{"q":"test"}"#.into(),
            },
        };
        let json_val = serde_json::to_value(&tc).unwrap();
        assert_eq!(
            json_val,
            json!({
                "id": "tc_1",
                "type": "function",
                "function": {"name": "search", "arguments": "{\"q\":\"test\"}"}
            })
        );
    }

    // -------------------------------------------------------------------
    // ToolDefinition serialization
    // -------------------------------------------------------------------

    #[test]
    fn test_tool_definition_serialization() {
        let td = ToolDefinition {
            tool_type: "function".into(),
            function: FunctionDefinition {
                name: "get_weather".into(),
                description: Some("Get current weather".into()),
                parameters: Some(
                    json!({"type": "object", "properties": {"city": {"type": "string"}}}),
                ),
            },
        };
        let json_val = serde_json::to_value(&td).unwrap();
        assert_eq!(
            json_val,
            json!({
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
                }
            })
        );
    }

    // -------------------------------------------------------------------
    // ToolChoice serialization
    // -------------------------------------------------------------------

    #[test]
    fn test_tool_choice_auto_serialization() {
        let tc = ToolChoice::Auto;
        let json_val = serde_json::to_value(&tc).unwrap();
        assert_eq!(json_val, json!("auto"));
    }

    #[test]
    fn test_tool_choice_none_serialization() {
        let tc = ToolChoice::None;
        let json_val = serde_json::to_value(&tc).unwrap();
        assert_eq!(json_val, json!("none"));
    }

    #[test]
    fn test_tool_choice_required_serialization() {
        let tc = ToolChoice::Required;
        let json_val = serde_json::to_value(&tc).unwrap();
        assert_eq!(json_val, json!("required"));
    }

    #[test]
    fn test_tool_choice_specific_serialization() {
        let tc = ToolChoice::Specific(ToolChoiceFunction {
            choice_type: "function".into(),
            function: ToolChoiceFunctionName {
                name: "my_func".into(),
            },
        });
        let json_val = serde_json::to_value(&tc).unwrap();
        assert_eq!(
            json_val,
            json!({"type": "function", "function": {"name": "my_func"}})
        );
    }

    // -------------------------------------------------------------------
    // GenerationParams serialization
    // -------------------------------------------------------------------

    #[test]
    fn test_generation_params_all_none_serializes_to_empty() {
        let params = GenerationParams::default();
        let json_val = serde_json::to_value(&params).unwrap();
        assert_eq!(json_val, json!({}));
    }

    // -------------------------------------------------------------------
    // Extra / flatten field
    // -------------------------------------------------------------------

    #[test]
    fn test_annotated_llm_request_extra_flatten() {
        let json_val = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
            "custom_field": "value"
        });
        let req: AnnotatedLLMRequest = serde_json::from_value(json_val).unwrap();
        assert_eq!(req.extra.get("stream"), Some(&json!(true)));
        assert_eq!(req.extra.get("custom_field"), Some(&json!("value")));
        // Round-trip: extra fields should appear as top-level keys
        let serialized = serde_json::to_value(&req).unwrap();
        assert_eq!(serialized["stream"], json!(true));
        assert_eq!(serialized["custom_field"], json!("value"));
    }

    // -------------------------------------------------------------------
    // Clone trait
    // -------------------------------------------------------------------

    #[test]
    fn test_all_types_clone() {
        let req = AnnotatedLLMRequest {
            messages: vec![
                Message::System {
                    content: MessageContent::Text("system".into()),
                    name: None,
                },
                Message::User {
                    content: MessageContent::Parts(vec![ContentPart::Text {
                        text: "user part".into(),
                    }]),
                    name: Some("alice".into()),
                },
            ],
            model: Some("gpt-4".into()),
            params: Some(GenerationParams {
                temperature: Some(0.5),
                max_tokens: None,
                top_p: Some(0.9),
                stop: Some(vec!["END".into()]),
            }),
            tools: Some(vec![ToolDefinition {
                tool_type: "function".into(),
                function: FunctionDefinition {
                    name: "test".into(),
                    description: None,
                    parameters: None,
                },
            }]),
            tool_choice: Some(ToolChoice::Auto),
            extra: serde_json::Map::new(),
        };
        let cloned = req.clone();
        assert_eq!(req, cloned);
    }

    // -------------------------------------------------------------------
    // PartialEq trait
    // -------------------------------------------------------------------

    #[test]
    fn test_all_types_partial_eq() {
        let msg1 = Message::User {
            content: MessageContent::Text("hello".into()),
            name: None,
        };
        let msg2 = Message::User {
            content: MessageContent::Text("hello".into()),
            name: None,
        };
        let msg3 = Message::User {
            content: MessageContent::Text("world".into()),
            name: None,
        };
        assert_eq!(msg1, msg2);
        assert_ne!(msg1, msg3);

        let tc1 = ToolChoice::Auto;
        let tc2 = ToolChoice::Auto;
        let tc3 = ToolChoice::None;
        assert_eq!(tc1, tc2);
        assert_ne!(tc1, tc3);
    }

    // -------------------------------------------------------------------
    // Helper method: system_prompt()
    // -------------------------------------------------------------------

    #[test]
    fn test_system_prompt_returns_text() {
        let req = AnnotatedLLMRequest {
            messages: vec![
                Message::System {
                    content: MessageContent::Text("Be helpful".into()),
                    name: None,
                },
                Message::User {
                    content: MessageContent::Text("Hi".into()),
                    name: None,
                },
            ],
            model: None,
            params: None,
            tools: None,
            tool_choice: None,
            extra: serde_json::Map::new(),
        };
        assert_eq!(req.system_prompt(), Some("Be helpful"));
    }

    #[test]
    fn test_system_prompt_returns_none_when_absent() {
        let req = AnnotatedLLMRequest {
            messages: vec![Message::User {
                content: MessageContent::Text("Hi".into()),
                name: None,
            }],
            model: None,
            params: None,
            tools: None,
            tool_choice: None,
            extra: serde_json::Map::new(),
        };
        assert_eq!(req.system_prompt(), None);
    }

    #[test]
    fn test_system_prompt_from_parts() {
        let req = AnnotatedLLMRequest {
            messages: vec![Message::System {
                content: MessageContent::Parts(vec![ContentPart::Text {
                    text: "Be concise".into(),
                }]),
                name: None,
            }],
            model: None,
            params: None,
            tools: None,
            tool_choice: None,
            extra: serde_json::Map::new(),
        };
        assert_eq!(req.system_prompt(), Some("Be concise"));
    }

    // -------------------------------------------------------------------
    // Helper method: last_user_message()
    // -------------------------------------------------------------------

    #[test]
    fn test_last_user_message_returns_last() {
        let req = AnnotatedLLMRequest {
            messages: vec![
                Message::User {
                    content: MessageContent::Text("first".into()),
                    name: None,
                },
                Message::Assistant {
                    content: None,
                    tool_calls: None,
                    name: None,
                },
                Message::User {
                    content: MessageContent::Text("last".into()),
                    name: None,
                },
            ],
            model: None,
            params: None,
            tools: None,
            tool_choice: None,
            extra: serde_json::Map::new(),
        };
        assert_eq!(req.last_user_message(), Some("last"));
    }

    #[test]
    fn test_last_user_message_returns_none_when_absent() {
        let req = AnnotatedLLMRequest {
            messages: vec![Message::System {
                content: MessageContent::Text("system".into()),
                name: None,
            }],
            model: None,
            params: None,
            tools: None,
            tool_choice: None,
            extra: serde_json::Map::new(),
        };
        assert_eq!(req.last_user_message(), None);
    }

    // -------------------------------------------------------------------
    // Helper method: has_tool_calls()
    // -------------------------------------------------------------------

    #[test]
    fn test_has_tool_calls_true() {
        let req = AnnotatedLLMRequest {
            messages: vec![Message::Assistant {
                content: None,
                tool_calls: Some(vec![ToolCall {
                    id: "tc_1".into(),
                    call_type: "function".into(),
                    function: FunctionCall {
                        name: "search".into(),
                        arguments: "{}".into(),
                    },
                }]),
                name: None,
            }],
            model: None,
            params: None,
            tools: None,
            tool_choice: None,
            extra: serde_json::Map::new(),
        };
        assert!(req.has_tool_calls());
    }

    #[test]
    fn test_has_tool_calls_false_no_assistant() {
        let req = AnnotatedLLMRequest {
            messages: vec![Message::User {
                content: MessageContent::Text("hi".into()),
                name: None,
            }],
            model: None,
            params: None,
            tools: None,
            tool_choice: None,
            extra: serde_json::Map::new(),
        };
        assert!(!req.has_tool_calls());
    }

    #[test]
    fn test_has_tool_calls_false_empty_vec() {
        let req = AnnotatedLLMRequest {
            messages: vec![Message::Assistant {
                content: Some(MessageContent::Text("hello".into())),
                tool_calls: Some(vec![]),
                name: None,
            }],
            model: None,
            params: None,
            tools: None,
            tool_choice: None,
            extra: serde_json::Map::new(),
        };
        assert!(!req.has_tool_calls());
    }
}
