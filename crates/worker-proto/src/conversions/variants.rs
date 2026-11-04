// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Explicit conversions for domain enums and protobuf structural differences.
//!
//! These mappings preserve alternatives, flattened fields, and native ownership
//! metadata that cannot be represented by a direct struct-field mapping.

use super::*;

impl WireValue<pb::LlmMessage> for nemo_relay_types::codec::request::Message {
    fn to_wire(self) -> Result<pb::LlmMessage> {
        Ok(pb::LlmMessage {
            role: Some(match self {
                Self::System { content, name } => pb::llm_message::Role::System(pb::NamedMessage {
                    content: field_encode!(content, required),
                    name: field_encode!(name, optional),
                }),
                Self::User { content, name } => pb::llm_message::Role::User(pb::NamedMessage {
                    content: field_encode!(content, required),
                    name: field_encode!(name, optional),
                }),
                Self::Developer { content, name } => {
                    pb::llm_message::Role::Developer(pb::NamedMessage {
                        content: field_encode!(content, required),
                        name: field_encode!(name, optional),
                    })
                }
                Self::Assistant {
                    content,
                    tool_calls,
                    name,
                } => pb::llm_message::Role::Assistant(pb::AssistantMessage {
                    content: field_encode!(content, optional),
                    tool_calls: field_encode!(tool_calls, optional),
                    name: field_encode!(name, optional),
                }),
                Self::Tool {
                    content,
                    tool_call_id,
                } => pb::llm_message::Role::Tool(pb::ToolMessage {
                    content: field_encode!(content, required),
                    tool_call_id: field_encode!(tool_call_id, scalar),
                }),
                Self::Function { content, name } => {
                    pb::llm_message::Role::Function(pb::FunctionMessage {
                        content: field_encode!(content, optional),
                        name: field_encode!(name, scalar),
                    })
                }
                Self::ToolCallItem {
                    id,
                    call_id,
                    name,
                    arguments,
                    extra,
                } => pb::llm_message::Role::ToolCall(pb::ToolCallItem {
                    id: field_encode!(id, optional),
                    call_id: field_encode!(call_id, scalar),
                    name: field_encode!(name, scalar),
                    arguments: field_encode!(arguments, required),
                    extra: encode_extra(extra, &["role", "id", "call_id", "name", "arguments"])?,
                }),
                Self::ToolResultItem {
                    id,
                    call_id,
                    output,
                    extra,
                } => pb::llm_message::Role::ToolResult(pb::ToolResultItem {
                    id: field_encode!(id, optional),
                    call_id: field_encode!(call_id, scalar),
                    output: field_encode!(output, required),
                    extra: encode_extra(extra, &["role", "id", "call_id", "output"])?,
                }),
                Self::ProviderNative {
                    provider,
                    kind,
                    value,
                } => pb::llm_message::Role::ProviderNative(pb::ProviderNativeComponent {
                    provider: field_encode!(provider, scalar),
                    kind: field_encode!(kind, scalar),
                    value: field_encode!(value, required),
                }),
            }),
        })
    }
    fn from_wire(value: pb::LlmMessage) -> Result<Self> {
        match required(value.role, "LlmMessage.role")? {
            pb::llm_message::Role::System(value) => Ok(Self::System {
                content: field_decode!(value.content, required),
                name: field_decode!(value.name, optional),
            }),
            pb::llm_message::Role::User(value) => Ok(Self::User {
                content: field_decode!(value.content, required),
                name: field_decode!(value.name, optional),
            }),
            pb::llm_message::Role::Developer(value) => Ok(Self::Developer {
                content: field_decode!(value.content, required),
                name: field_decode!(value.name, optional),
            }),
            pb::llm_message::Role::Assistant(value) => Ok(Self::Assistant {
                content: field_decode!(value.content, optional),
                tool_calls: field_decode!(value.tool_calls, optional),
                name: field_decode!(value.name, optional),
            }),
            pb::llm_message::Role::Tool(value) => Ok(Self::Tool {
                content: field_decode!(value.content, required),
                tool_call_id: field_decode!(value.tool_call_id, scalar),
            }),
            pb::llm_message::Role::Function(value) => Ok(Self::Function {
                content: field_decode!(value.content, optional),
                name: field_decode!(value.name, scalar),
            }),
            pb::llm_message::Role::ToolCall(value) => Ok(Self::ToolCallItem {
                id: field_decode!(value.id, optional),
                call_id: field_decode!(value.call_id, scalar),
                name: field_decode!(value.name, scalar),
                arguments: field_decode!(value.arguments, required),
                extra: decode_extra(value.extra, &["role", "id", "call_id", "name", "arguments"])?,
            }),
            pb::llm_message::Role::ToolResult(value) => Ok(Self::ToolResultItem {
                id: field_decode!(value.id, optional),
                call_id: field_decode!(value.call_id, scalar),
                output: field_decode!(value.output, required),
                extra: decode_extra(value.extra, &["role", "id", "call_id", "output"])?,
            }),
            pb::llm_message::Role::ProviderNative(value) => Ok(Self::ProviderNative {
                provider: field_decode!(value.provider, scalar),
                kind: field_decode!(value.kind, scalar),
                value: field_decode!(value.value, required),
            }),
        }
    }
}

impl WireValue<pb::ContentPart> for nemo_relay_types::codec::request::ContentPart {
    fn to_wire(self) -> Result<pb::ContentPart> {
        Ok(pb::ContentPart {
            part: Some(match self {
                Self::Text { text, extra } => pb::content_part::Part::Text(pb::TextContentPart {
                    text: field_encode!(text, scalar),
                    extra: encode_extra(extra, &["type", "text"])?,
                }),
                Self::ImageUrl { image_url, extra } => {
                    pb::content_part::Part::ImageUrl(pb::ImageUrlContentPart {
                        image_url: field_encode!(image_url, required),
                        extra: encode_extra(extra, &["type", "image_url"])?,
                    })
                }
                Self::Image { image, extra } => {
                    pb::content_part::Part::Image(pb::JsonContentPart {
                        value: field_encode!(image, required),
                        extra: encode_extra(extra, &["type", "image"])?,
                    })
                }
                Self::Audio { audio, extra } => {
                    pb::content_part::Part::Audio(pb::JsonContentPart {
                        value: field_encode!(audio, required),
                        extra: encode_extra(extra, &["type", "audio"])?,
                    })
                }
                Self::File { file, extra } => pb::content_part::Part::File(pb::JsonContentPart {
                    value: field_encode!(file, required),
                    extra: encode_extra(extra, &["type", "file"])?,
                }),
                Self::Refusal { refusal, extra } => {
                    pb::content_part::Part::Refusal(pb::TextContentPart {
                        text: field_encode!(refusal, scalar),
                        extra: encode_extra(extra, &["type", "refusal"])?,
                    })
                }
                Self::ToolUse {
                    id,
                    name,
                    input,
                    extra,
                } => pb::content_part::Part::ToolUse(pb::ToolUseContentPart {
                    id: field_encode!(id, scalar),
                    name: field_encode!(name, scalar),
                    input: field_encode!(input, required),
                    extra: encode_extra(extra, &["type", "id", "name", "input"])?,
                }),
                Self::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                    extra,
                } => pb::content_part::Part::ToolResult(pb::ToolResultContentPart {
                    tool_use_id: field_encode!(tool_use_id, scalar),
                    content: field_encode!(content, required),
                    is_error: field_encode!(is_error, optional),
                    extra: encode_extra(extra, &["type", "tool_use_id", "content", "is_error"])?,
                }),
                Self::ProviderNative {
                    provider,
                    kind,
                    value,
                } => pb::content_part::Part::ProviderNative(pb::ProviderNativeComponent {
                    provider: field_encode!(provider, scalar),
                    kind: field_encode!(kind, scalar),
                    value: field_encode!(value, required),
                }),
            }),
        })
    }
    fn from_wire(value: pb::ContentPart) -> Result<Self> {
        match required(value.part, "ContentPart.part")? {
            pb::content_part::Part::Text(value) => Ok(Self::Text {
                text: field_decode!(value.text, scalar),
                extra: decode_extra(value.extra, &["type", "text"])?,
            }),
            pb::content_part::Part::ImageUrl(value) => Ok(Self::ImageUrl {
                image_url: field_decode!(value.image_url, required),
                extra: decode_extra(value.extra, &["type", "image_url"])?,
            }),
            pb::content_part::Part::Image(value) => Ok(Self::Image {
                image: field_decode!(value.value, required),
                extra: decode_extra(value.extra, &["type", "image"])?,
            }),
            pb::content_part::Part::Audio(value) => Ok(Self::Audio {
                audio: field_decode!(value.value, required),
                extra: decode_extra(value.extra, &["type", "audio"])?,
            }),
            pb::content_part::Part::File(value) => Ok(Self::File {
                file: field_decode!(value.value, required),
                extra: decode_extra(value.extra, &["type", "file"])?,
            }),
            pb::content_part::Part::Refusal(value) => Ok(Self::Refusal {
                refusal: field_decode!(value.text, scalar),
                extra: decode_extra(value.extra, &["type", "refusal"])?,
            }),
            pb::content_part::Part::ToolUse(value) => Ok(Self::ToolUse {
                id: field_decode!(value.id, scalar),
                name: field_decode!(value.name, scalar),
                input: field_decode!(value.input, required),
                extra: decode_extra(value.extra, &["type", "id", "name", "input"])?,
            }),
            pb::content_part::Part::ToolResult(value) => Ok(Self::ToolResult {
                tool_use_id: field_decode!(value.tool_use_id, scalar),
                content: field_decode!(value.content, required),
                is_error: field_decode!(value.is_error, optional),
                extra: decode_extra(value.extra, &["type", "tool_use_id", "content", "is_error"])?,
            }),
            pb::content_part::Part::ProviderNative(value) => Ok(Self::ProviderNative {
                provider: field_decode!(value.provider, scalar),
                kind: field_decode!(value.kind, scalar),
                value: field_decode!(value.value, required),
            }),
        }
    }
}

impl WireValue<pb::ToolDefinition> for nemo_relay_types::codec::request::ToolDefinition {
    fn to_wire(self) -> Result<pb::ToolDefinition> {
        Ok(pb::ToolDefinition {
            definition: Some(match self {
                Self::Function { function, extra } => {
                    pb::tool_definition::Definition::Function(pb::FunctionToolDefinition {
                        function: field_encode!(function, required),
                        extra: encode_extra(extra, &["type", "function"])?,
                    })
                }
                Self::ProviderNative {
                    provider,
                    kind,
                    value,
                } => pb::tool_definition::Definition::ProviderNative(pb::ProviderNativeComponent {
                    provider: field_encode!(provider, scalar),
                    kind: field_encode!(kind, scalar),
                    value: field_encode!(value, required),
                }),
            }),
        })
    }
    fn from_wire(value: pb::ToolDefinition) -> Result<Self> {
        match required(value.definition, "ToolDefinition.definition")? {
            pb::tool_definition::Definition::Function(value) => Ok(Self::Function {
                function: field_decode!(value.function, required),
                extra: decode_extra(value.extra, &["type", "function"])?,
            }),
            pb::tool_definition::Definition::ProviderNative(value) => Ok(Self::ProviderNative {
                provider: field_decode!(value.provider, scalar),
                kind: field_decode!(value.kind, scalar),
                value: field_decode!(value.value, required),
            }),
        }
    }
}

// Codec-owned provider data uses the same Serde representation as the native SDK.
macro_rules! codec_data {
    ($($domain:ty),+ $(,)?) => {$(
        impl WireValue<pb::JsonValue> for $domain {
            fn to_wire(self) -> Result<pb::JsonValue> {
                Ok(pb::JsonValue {
                    json: serde_json::to_vec(&self).map_err(|e| ConversionError(e.to_string()))?,
                })
            }
            fn from_wire(value: pb::JsonValue) -> Result<Self> {
                serde_json::from_slice(&value.json).map_err(|e| ConversionError(e.to_string()))
            }
        }
    )+};
}
codec_data!(
    nemo_relay_types::codec::request::ApiSpecificRequest,
    nemo_relay_types::codec::response::ApiSpecificResponse,
);

impl WireValue<pb::Event> for nemo_relay_types::api::event::Event {
    fn to_wire(self) -> Result<pb::Event> {
        Ok(pb::Event {
            kind: Some(match self {
                Self::Scope(inner) => pb::event::Kind::Scope(encode(inner)?),
                Self::Mark(inner) => pb::event::Kind::Mark(encode(inner)?),
            }),
        })
    }
    fn from_wire(value: pb::Event) -> Result<Self> {
        match required(value.kind, "Event.kind")? {
            pb::event::Kind::Scope(inner) => Ok(Self::Scope(decode(inner)?)),
            pb::event::Kind::Mark(inner) => Ok(Self::Mark(decode(inner)?)),
        }
    }
}

impl WireValue<pb::FinishReason> for nemo_relay_types::codec::response::FinishReason {
    fn to_wire(self) -> Result<pb::FinishReason> {
        Ok(pb::FinishReason {
            reason: Some(match self {
                Self::Complete => pb::finish_reason::Reason::Complete(()),
                Self::Length => pb::finish_reason::Reason::Length(()),
                Self::ToolUse => pb::finish_reason::Reason::ToolUse(()),
                Self::ContentFilter => pb::finish_reason::Reason::ContentFilter(()),
                Self::Unknown(inner) => pb::finish_reason::Reason::Unknown(encode(inner)?),
            }),
        })
    }
    fn from_wire(value: pb::FinishReason) -> Result<Self> {
        match required(value.reason, "FinishReason.reason")? {
            pb::finish_reason::Reason::Complete(_) => Ok(Self::Complete),
            pb::finish_reason::Reason::Length(_) => Ok(Self::Length),
            pb::finish_reason::Reason::ToolUse(_) => Ok(Self::ToolUse),
            pb::finish_reason::Reason::ContentFilter(_) => Ok(Self::ContentFilter),
            pb::finish_reason::Reason::Unknown(inner) => Ok(Self::Unknown(decode(inner)?)),
        }
    }
}

impl WireValue<pb::ToolChoice> for nemo_relay_types::codec::request::ToolChoice {
    fn to_wire(self) -> Result<pb::ToolChoice> {
        Ok(pb::ToolChoice {
            choice: Some(match self {
                Self::Auto => pb::tool_choice::Choice::Auto(()),
                Self::None => pb::tool_choice::Choice::None(()),
                Self::Required => pb::tool_choice::Choice::Required(()),
                Self::Specific(inner) => pb::tool_choice::Choice::Specific(encode(inner)?),
                Self::ProviderNative(inner) => {
                    pb::tool_choice::Choice::ProviderNative(encode(inner)?)
                }
            }),
        })
    }
    fn from_wire(value: pb::ToolChoice) -> Result<Self> {
        match required(value.choice, "ToolChoice.choice")? {
            pb::tool_choice::Choice::Auto(_) => Ok(Self::Auto),
            pb::tool_choice::Choice::None(_) => Ok(Self::None),
            pb::tool_choice::Choice::Required(_) => Ok(Self::Required),
            pb::tool_choice::Choice::Specific(inner) => Ok(Self::Specific(decode(inner)?)),
            pb::tool_choice::Choice::ProviderNative(inner) => {
                Ok(Self::ProviderNative(decode(inner)?))
            }
        }
    }
}

impl WireValue<pb::ResourceMemoryNumber> for nemo_relay_types::diagnostics::ResourceMemoryNumber {
    fn to_wire(self) -> Result<pb::ResourceMemoryNumber> {
        Ok(pb::ResourceMemoryNumber {
            kind: Some(match self {
                Self::Integer(inner) => pb::resource_memory_number::Kind::Integer(encode(inner)?),
                Self::Decimal(inner) => pb::resource_memory_number::Kind::Decimal(encode(inner)?),
                Self::NonFinite => pb::resource_memory_number::Kind::NonFinite(()),
            }),
        })
    }
    fn from_wire(value: pb::ResourceMemoryNumber) -> Result<Self> {
        match required(value.kind, "ResourceMemoryNumber.kind")? {
            pb::resource_memory_number::Kind::Integer(inner) => Ok(Self::Integer(decode(inner)?)),
            pb::resource_memory_number::Kind::Decimal(inner) => Ok(Self::Decimal(decode(inner)?)),
            pb::resource_memory_number::Kind::NonFinite(_) => Ok(Self::NonFinite),
        }
    }
}

impl WireValue<pb::ResourceMemoryUnit> for nemo_relay_types::diagnostics::ResourceMemoryUnit {
    fn to_wire(self) -> Result<pb::ResourceMemoryUnit> {
        Ok(pb::ResourceMemoryUnit {
            dimension: Some(match self {
                Self::Capacity(inner) => {
                    pb::resource_memory_unit::Dimension::Capacity(encode(inner)?)
                }
                Self::Duration(inner) => {
                    pb::resource_memory_unit::Dimension::Duration(encode(inner)?)
                }
                Self::Count(inner) => pb::resource_memory_unit::Dimension::Count(encode(inner)?),
            }),
        })
    }
    fn from_wire(value: pb::ResourceMemoryUnit) -> Result<Self> {
        match required(value.dimension, "ResourceMemoryUnit.dimension")? {
            pb::resource_memory_unit::Dimension::Capacity(inner) => {
                Ok(Self::Capacity(decode(inner)?))
            }
            pb::resource_memory_unit::Dimension::Duration(inner) => {
                Ok(Self::Duration(decode(inner)?))
            }
            pb::resource_memory_unit::Dimension::Count(inner) => Ok(Self::Count(decode(inner)?)),
        }
    }
}

impl WireValue<pb::MessageContent> for nemo_relay_types::codec::request::MessageContent {
    fn to_wire(self) -> Result<pb::MessageContent> {
        use pb::message_content::Content;
        Ok(pb::MessageContent {
            content: Some(match self {
                Self::Text(text) => Content::Text(text),
                Self::Parts(parts) => Content::Parts(encode(parts)?),
            }),
        })
    }
    fn from_wire(value: pb::MessageContent) -> Result<Self> {
        Ok(match required(value.content, "MessageContent.content")? {
            pb::message_content::Content::Text(text) => Self::Text(text),
            pb::message_content::Content::Parts(parts) => Self::Parts(decode(parts)?),
        })
    }
}

impl WireValue<pb::RuntimeRegistrationOwner>
    for nemo_relay_types::api::registry::RuntimeRegistrationOwner
{
    fn to_wire(self) -> Result<pb::RuntimeRegistrationOwner> {
        use nemo_relay_types::api::registry::RuntimeRegistrationOwnerKind as Kind;
        use pb::runtime_registration_owner::Owner;
        Ok(pb::RuntimeRegistrationOwner {
            owner: Some(match self.kind {
                Kind::Core => Owner::Core(()),
                Kind::GlobalApi => Owner::GlobalApi(()),
                Kind::Plugin => Owner::Plugin(pb::PluginRegistrationOwner {
                    plugin_kind: required(self.plugin_kind, "plugin_kind")?,
                    component_ordinal: self.component_ordinal,
                }),
            }),
        })
    }
    fn from_wire(value: pb::RuntimeRegistrationOwner) -> Result<Self> {
        use nemo_relay_types::api::registry::RuntimeRegistrationOwnerKind as Kind;
        use pb::runtime_registration_owner::Owner;
        let (kind, plugin_kind, component_ordinal) =
            match required(value.owner, "registration owner")? {
                Owner::Core(_) => (Kind::Core, None, None),
                Owner::GlobalApi(_) => (Kind::GlobalApi, None, None),
                Owner::Plugin(plugin) => (
                    Kind::Plugin,
                    Some(plugin.plugin_kind),
                    plugin.component_ordinal,
                ),
            };
        Ok(Self {
            kind,
            plugin_kind,
            component_ordinal,
        })
    }
}

impl WireValue<pb::DiagnosticClassification> for nemo_relay_types::diagnostics::Classification {
    fn to_wire(self) -> Result<pb::DiagnosticClassification> {
        let (code, class) = self.into_names();
        Ok(pb::DiagnosticClassification {
            canonical_code: code.into_owned(),
            failure_class: class.map(|v| v.into_owned()),
        })
    }
    fn from_wire(value: pb::DiagnosticClassification) -> Result<Self> {
        Ok(Self::from_opaque_names(
            value.canonical_code.into(),
            value.failure_class.map(Into::into),
        ))
    }
}

impl<'a> WireValue<pb::DiagnosticCorrelation>
    for nemo_relay_types::diagnostics::DiagnosticCorrelation<'a>
{
    fn to_wire(self) -> Result<pb::DiagnosticCorrelation> {
        Ok(pb::DiagnosticCorrelation {
            session_id: self.session_id.map(|v| v.into_owned()),
            request_id: self.request_id.map(|v| v.into_owned()),
        })
    }
    fn from_wire(value: pb::DiagnosticCorrelation) -> Result<Self> {
        Ok(Self {
            session_id: value.session_id.map(Into::into),
            request_id: value.request_id.map(Into::into),
        })
    }
}

impl WireValue<pb::LlmCodecReference> for nemo_relay_types::codec::identity::LlmCodecIdentity {
    fn to_wire(self) -> Result<pb::LlmCodecReference> {
        let (kind, id) = match self {
            Self::None => ("none", None),
            Self::BuiltIn(value) => ("builtin", Some(value.id().to_owned())),
            Self::Runtime(value) => ("runtime", Some(value)),
            Self::Opaque => ("opaque", None),
        };
        Ok(pb::LlmCodecReference {
            kind: kind.into(),
            id,
        })
    }
    fn from_wire(value: pb::LlmCodecReference) -> Result<Self> {
        // References are descriptive. Unknown implementations still use the
        // same capability RPCs; their metadata must not prevent codec access.
        Ok(match (value.kind.as_str(), value.id) {
            ("none", _) => Self::None,
            ("builtin", Some(id)) => {
                nemo_relay_types::codec::identity::BuiltinLlmCodec::from_id(&id)
                    .map(Self::BuiltIn)
                    .unwrap_or(Self::Runtime(id))
            }
            (_, Some(id)) => Self::Runtime(id),
            (_, None) => Self::Opaque,
        })
    }
}

impl WireValue<pb::ToolExecutionInterceptOutcome>
    for nemo_relay_types::api::tool::ToolExecutionInterceptOutcome
{
    fn to_wire(self) -> Result<pb::ToolExecutionInterceptOutcome> {
        Ok(pb::ToolExecutionInterceptOutcome {
            value: Some(encode(nemo_relay_types::api::tool::ToolExecutionResult {
                result: self.result,
                annotation: self.annotation.filter(|v| !v.is_null()),
            })?),
            pending_marks: self
                .pending_marks
                .into_iter()
                .map(encode)
                .collect::<Result<_>>()?,
        })
    }
    fn from_wire(value: pb::ToolExecutionInterceptOutcome) -> Result<Self> {
        let tool: nemo_relay_types::api::tool::ToolExecutionResult =
            decode(required(value.value, "tool execution outcome")?)?;
        Ok(Self {
            result: tool.result,
            annotation: tool.annotation.filter(|v| !v.is_null()),
            pending_marks: value
                .pending_marks
                .into_iter()
                .map(decode)
                .collect::<Result<_>>()?,
        })
    }
}
