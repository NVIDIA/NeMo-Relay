// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Owned content projection shared by transports and local readers.

use super::content::{ContentKind, ContentLocation, MessageView, PayloadRef};
use super::request::{AnnotatedLlmRequest, Message, MessageContent, OpenAiImageUrl};
use super::response::AnnotatedLlmResponse;
use crate::Json;

/// Borrowed normalized input for content inspection; no provider decoding occurs.
#[derive(Clone, Copy)]
pub enum ContentSource<'a> {
    /// Supplied normalized request history and instructions.
    Request(&'a AnnotatedLlmRequest),
    /// Current normalized response output.
    Response(&'a AnnotatedLlmResponse),
    /// Standalone normalized message.
    Message(&'a Message),
    /// Standalone content, including instructions.
    Content(&'a MessageContent),
}

/// Selection applied before extraction. Invalid source/selection pairs are errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentSelection {
    /// Exactly the final supplied request item.
    LastMessage,
    /// Request history index; negative values count from the end.
    Message(i64),
    /// Top-level instructions only.
    Instructions,
    /// Instructions, otherwise the first system/developer item.
    SystemPrompt,
    /// Last portable or recognized native user item.
    LastUserMessage,
    /// Current response message and separate tool calls.
    Output,
    /// All parts of standalone message/content input.
    All,
}

/// Invalid content-inspection selection, independent of any transport.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ContentSelectionError {
    /// The selection is not defined for this normalized source category.
    #[error("content selection is incompatible with source")]
    InvalidSource,
    /// Explicit history index is outside the supplied request.
    #[error("message index out of range")]
    IndexOutOfRange,
}

pub(super) fn request_system_view(request: &AnnotatedLlmRequest) -> Option<MessageView<'_>> {
    if let Some(instructions) = &request.instructions {
        return Some(instructions.content_view());
    }
    request
        .messages
        .iter()
        .enumerate()
        .find(|(_, message)| {
            matches!(message, Message::System { .. } | Message::Developer { .. })
                || matches!(message.native_message_role(), Some("system" | "developer"))
        })
        .map(|(index, message)| MessageView::request_message(index, message))
}

pub(super) fn request_user_view(request: &AnnotatedLlmRequest) -> Option<MessageView<'_>> {
    request
        .messages
        .iter()
        .enumerate()
        .rev()
        .find(|(_, message)| {
            matches!(message, Message::User { .. }) || message.native_message_role() == Some("user")
        })
        .map(|(index, message)| MessageView::request_message(index, message))
}

impl ContentSelection {
    /// Select borrowed normalized content without searching past a selected textless item.
    ///
    /// # Errors
    /// Returns an error for incompatible sources or an out-of-range explicit index.
    pub fn select(
        self,
        source: ContentSource<'_>,
    ) -> Result<Option<MessageView<'_>>, ContentSelectionError> {
        match (self, source) {
            (Self::LastMessage, ContentSource::Request(request)) => Ok(request.last_message()),
            (Self::Message(index), ContentSource::Request(request)) => {
                let resolved = if index < 0 {
                    usize::try_from(index.unsigned_abs())
                        .ok()
                        .and_then(|distance| request.messages.len().checked_sub(distance))
                } else {
                    usize::try_from(index).ok()
                };
                let index = resolved
                    .filter(|index| *index < request.messages.len())
                    .ok_or(ContentSelectionError::IndexOutOfRange)?;
                Ok(Some(MessageView::request_message(
                    index,
                    &request.messages[index],
                )))
            }
            (Self::Instructions, ContentSource::Request(request)) => Ok(request
                .instructions
                .as_ref()
                .map(MessageContent::content_view)),
            (Self::SystemPrompt, ContentSource::Request(request)) => {
                Ok(request_system_view(request))
            }
            (Self::LastUserMessage, ContentSource::Request(request)) => {
                Ok(request_user_view(request))
            }
            (Self::Output, ContentSource::Response(response)) => Ok(Some(response.output())),
            (Self::All, ContentSource::Message(message)) => Ok(Some(message.content_view())),
            (Self::All, ContentSource::Content(content)) => Ok(Some(content.content_view())),
            _ => Err(ContentSelectionError::InvalidSource),
        }
    }
}

/// Owned snapshot; materialization copies selected content for transport or independent retention.
#[derive(Debug, Clone, PartialEq)]
pub struct ContentSnapshot {
    /// Ordered typed entries, including tool/media/native payloads.
    pub parts: Vec<SnapshotEntry>,
    /// Authoritative eligible text strings from shared extraction, preserving empty parts.
    pub text_parts: Vec<String>,
}

/// One owned typed entry and its read-time normalized location.
#[derive(Debug, Clone, PartialEq)]
pub struct SnapshotEntry {
    /// Normalized field and array position, not an edit handle.
    pub location: ContentLocation,
    /// Owned typed content.
    pub kind: SnapshotKind,
}

/// Owned counterpart of borrowed content kinds.
#[derive(Debug, Clone, PartialEq)]
pub enum SnapshotKind {
    /// Explicit text.
    Text(String),
    /// Explicit refusal.
    Refusal(String),
    /// Tool invocation.
    ToolCall(SnapshotToolCall),
    /// Tool result, excluded from default text.
    ToolResult(SnapshotToolResult),
    /// Image URL.
    ImageUrl(OpenAiImageUrl),
    /// Image payload.
    Image(Json),
    /// Audio payload.
    Audio(Json),
    /// File payload.
    File(Json),
    /// Opaque native data; extraction eligibility is supplied separately by the host.
    ProviderNative {
        /// Provider identity.
        provider: String,
        /// Native discriminator.
        kind: String,
        /// Original normalized native value.
        value: Json,
    },
}

/// Tool payload in its existing normalized representation.
#[derive(Debug, Clone, PartialEq)]
pub enum SnapshotPayload {
    /// Explicit absent legacy result.
    Null,
    /// String-valued arguments/result; never parsed.
    String(String),
    /// Parsed JSON, including JSON strings/nulls.
    Json(Json),
    /// Complete normalized tool-result content.
    Content(MessageContent),
}

/// Owned invocation identifiers, name, and arguments.
#[derive(Debug, Clone, PartialEq)]
pub struct SnapshotToolCall {
    /// Item/tool-call identifier.
    pub id: Option<String>,
    /// Separate correlation identifier.
    pub call_id: Option<String>,
    /// Function/tool name.
    pub name: String,
    /// Original argument representation.
    pub arguments: SnapshotPayload,
}

/// Owned result identifiers and complete payload.
#[derive(Debug, Clone, PartialEq)]
pub struct SnapshotToolResult {
    /// Item identifier.
    pub id: Option<String>,
    /// Corresponding tool-call identifier.
    pub call_id: Option<String>,
    /// Legacy function-result name.
    pub name: Option<String>,
    /// Explicit error flag.
    pub is_error: Option<bool>,
    /// Complete result payload.
    pub payload: SnapshotPayload,
}

impl From<PayloadRef<'_>> for SnapshotPayload {
    fn from(payload: PayloadRef<'_>) -> Self {
        match payload {
            PayloadRef::Null => Self::Null,
            PayloadRef::String(value) => Self::String(value.into()),
            PayloadRef::Json(value) => Self::Json(value.clone()),
            PayloadRef::Content(value) => Self::Content(value.clone()),
        }
    }
}

impl MessageView<'_> {
    /// Copy selected typed content and eligible text into an owned transport snapshot.
    pub fn snapshot(self) -> ContentSnapshot {
        let parts = self
            .parts()
            .map(|entry| {
                let kind = match entry.kind {
                    ContentKind::Text(text) => SnapshotKind::Text(text.into()),
                    ContentKind::Refusal(text) => SnapshotKind::Refusal(text.into()),
                    ContentKind::ToolCall(call) => SnapshotKind::ToolCall(SnapshotToolCall {
                        id: call.id.map(Into::into),
                        call_id: call.call_id.map(Into::into),
                        name: call.name.into(),
                        arguments: call.arguments.into(),
                    }),
                    ContentKind::ToolResult(result) => {
                        SnapshotKind::ToolResult(SnapshotToolResult {
                            id: result.id.map(Into::into),
                            call_id: result.call_id.map(Into::into),
                            name: result.name.map(Into::into),
                            is_error: result.is_error,
                            payload: result.payload.into(),
                        })
                    }
                    ContentKind::ImageUrl(value) => SnapshotKind::ImageUrl(value.clone()),
                    ContentKind::Image(value) => SnapshotKind::Image(value.clone()),
                    ContentKind::Audio(value) => SnapshotKind::Audio(value.clone()),
                    ContentKind::File(value) => SnapshotKind::File(value.clone()),
                    ContentKind::ProviderNative {
                        provider,
                        kind,
                        value,
                    } => SnapshotKind::ProviderNative {
                        provider: provider.into(),
                        kind: kind.into(),
                        value: value.clone(),
                    },
                };
                SnapshotEntry {
                    location: entry.location,
                    kind,
                }
            })
            .collect();
        ContentSnapshot {
            parts,
            text_parts: self.text_parts().map(Into::into).collect(),
        }
    }
}
