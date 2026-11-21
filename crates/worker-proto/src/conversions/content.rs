// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Closed content-inspection wire projections; application payloads remain lossless JSON.

use super::*;
use nemo_relay_types::codec::content::{
    ContentField, ContentLocation, ContentRoot, ContentSelection, ContentSnapshot, SnapshotEntry,
    SnapshotKind, SnapshotPayload, SnapshotToolCall, SnapshotToolResult,
};
use nemo_relay_types::codec::request::ProviderNativeComponent;

impl WireValue<pb::LlmContentSelection> for ContentSelection {
    fn to_wire(self) -> Result<pb::LlmContentSelection> {
        use pb::llm_content_selection::Kind;
        Ok(pb::LlmContentSelection {
            kind: Some(match self {
                Self::LastMessage => Kind::LastMessage(()),
                Self::Message(index) => Kind::MessageIndex(index),
                Self::Instructions => Kind::Instructions(()),
                Self::SystemPrompt => Kind::SystemPrompt(()),
                Self::LastUserMessage => Kind::LastUserMessage(()),
                Self::Output => Kind::Output(()),
                Self::All => Kind::All(()),
            }),
        })
    }
    fn from_wire(value: pb::LlmContentSelection) -> Result<Self> {
        use pb::llm_content_selection::Kind;
        Ok(match required(value.kind, "content selection")? {
            Kind::LastMessage(()) => Self::LastMessage,
            Kind::MessageIndex(index) => Self::Message(index),
            Kind::Instructions(()) => Self::Instructions,
            Kind::SystemPrompt(()) => Self::SystemPrompt,
            Kind::LastUserMessage(()) => Self::LastUserMessage,
            Kind::Output(()) => Self::Output,
            Kind::All(()) => Self::All,
        })
    }
}

impl WireValue<pb::LlmContentLocation> for ContentLocation {
    fn to_wire(self) -> Result<pb::LlmContentLocation> {
        let (root, message_index) = match self.root {
            ContentRoot::RequestMessage(index) => {
                (pb::LlmContentRoot::RequestMessage, Some(encode(index)?))
            }
            ContentRoot::Message => (pb::LlmContentRoot::Message, None),
            ContentRoot::Response => (pb::LlmContentRoot::Response, None),
            ContentRoot::Content => (pb::LlmContentRoot::Content, None),
        };
        let field = match self.field {
            ContentField::Content => pb::LlmContentField::Content,
            ContentField::Message => pb::LlmContentField::Message,
            ContentField::ToolCalls => pb::LlmContentField::ToolCalls,
            ContentField::ToolCallItem => pb::LlmContentField::ToolCallItem,
            ContentField::Output => pb::LlmContentField::Output,
            ContentField::Value => pb::LlmContentField::Value,
            ContentField::NativeContent => pb::LlmContentField::NativeContent,
        };
        Ok(pb::LlmContentLocation {
            root: root as i32,
            message_index,
            field: field as i32,
            index: self.index.map(encode).transpose()?,
        })
    }
    fn from_wire(value: pb::LlmContentLocation) -> Result<Self> {
        let root = match pb::LlmContentRoot::try_from(value.root).ok() {
            Some(pb::LlmContentRoot::RequestMessage) => ContentRoot::RequestMessage(decode(
                required(value.message_index, "request message index")?,
            )?),
            Some(pb::LlmContentRoot::Message) => ContentRoot::Message,
            Some(pb::LlmContentRoot::Response) => ContentRoot::Response,
            Some(pb::LlmContentRoot::Content) => ContentRoot::Content,
            _ => return Err(ConversionError("invalid content root".into())),
        };
        if !matches!(root, ContentRoot::RequestMessage(_)) && value.message_index.is_some() {
            return Err(ConversionError(
                "message_index requires a request_message root".into(),
            ));
        }
        let field = match pb::LlmContentField::try_from(value.field).ok() {
            Some(pb::LlmContentField::Content) => ContentField::Content,
            Some(pb::LlmContentField::Message) => ContentField::Message,
            Some(pb::LlmContentField::ToolCalls) => ContentField::ToolCalls,
            Some(pb::LlmContentField::ToolCallItem) => ContentField::ToolCallItem,
            Some(pb::LlmContentField::Output) => ContentField::Output,
            Some(pb::LlmContentField::Value) => ContentField::Value,
            Some(pb::LlmContentField::NativeContent) => ContentField::NativeContent,
            _ => return Err(ConversionError("invalid content field".into())),
        };
        Ok(Self {
            root,
            field,
            index: value.index.map(decode).transpose()?,
        })
    }
}

impl WireValue<pb::LlmContentPayload> for SnapshotPayload {
    fn to_wire(self) -> Result<pb::LlmContentPayload> {
        use pb::llm_content_payload::Kind;
        Ok(pb::LlmContentPayload {
            kind: Some(match self {
                Self::Null => Kind::Null(()),
                Self::String(value) => Kind::String(value),
                Self::Json(value) => Kind::Json(encode(value)?),
                Self::Content(value) => Kind::Content(encode(value)?),
            }),
        })
    }
    fn from_wire(value: pb::LlmContentPayload) -> Result<Self> {
        use pb::llm_content_payload::Kind;
        Ok(
            match required(value.kind, "content payload representation")? {
                Kind::Null(()) => Self::Null,
                Kind::String(value) => Self::String(value),
                Kind::Json(value) => Self::Json(decode(value)?),
                Kind::Content(value) => Self::Content(decode(value)?),
            },
        )
    }
}

dto!(SnapshotToolCall => LlmContentToolCall {
    id: optional, call_id: optional, name: scalar, arguments: required,
});
dto!(SnapshotToolResult => LlmContentToolResult {
    id: optional, call_id: optional, name: optional, is_error: optional, payload: required,
});
dto!(ContentSnapshot => LlmContentSnapshot {
    parts: repeated, text_parts: repeated,
});

impl WireValue<pb::LlmContentEntry> for SnapshotEntry {
    fn to_wire(self) -> Result<pb::LlmContentEntry> {
        use pb::llm_content_entry::Kind;
        let kind = match self.kind {
            SnapshotKind::Text(value) => Kind::Text(value),
            SnapshotKind::Refusal(value) => Kind::Refusal(value),
            SnapshotKind::ToolCall(value) => Kind::ToolCall(encode(value)?),
            SnapshotKind::ToolResult(value) => Kind::ToolResult(encode(value)?),
            SnapshotKind::ImageUrl(value) => Kind::ImageUrl(encode(value)?),
            SnapshotKind::Image(value) => Kind::Image(encode(value)?),
            SnapshotKind::Audio(value) => Kind::Audio(encode(value)?),
            SnapshotKind::File(value) => Kind::File(encode(value)?),
            SnapshotKind::ProviderNative {
                provider,
                kind,
                value,
            } => Kind::ProviderNative(encode(ProviderNativeComponent {
                provider,
                kind,
                value,
            })?),
        };
        Ok(pb::LlmContentEntry {
            location: Some(encode(self.location)?),
            kind: Some(kind),
        })
    }
    fn from_wire(value: pb::LlmContentEntry) -> Result<Self> {
        use pb::llm_content_entry::Kind;
        let kind = match required(value.kind, "content entry kind")? {
            Kind::Text(value) => SnapshotKind::Text(value),
            Kind::Refusal(value) => SnapshotKind::Refusal(value),
            Kind::ToolCall(value) => SnapshotKind::ToolCall(decode(value)?),
            Kind::ToolResult(value) => SnapshotKind::ToolResult(decode(value)?),
            Kind::ImageUrl(value) => SnapshotKind::ImageUrl(decode(value)?),
            Kind::Image(value) => SnapshotKind::Image(decode(value)?),
            Kind::Audio(value) => SnapshotKind::Audio(decode(value)?),
            Kind::File(value) => SnapshotKind::File(decode(value)?),
            Kind::ProviderNative(value) => {
                let ProviderNativeComponent {
                    provider,
                    kind,
                    value,
                } = decode(value)?;
                SnapshotKind::ProviderNative {
                    provider,
                    kind,
                    value,
                }
            }
        };
        Ok(Self {
            location: decode(required(value.location, "content entry location")?)?,
            kind,
        })
    }
}
