// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Borrowed access to normalized message content.
//!
//! Views do not reconstruct provider wire order, parse tool arguments, or retain
//! conversation state. Locations refer to the current normalized source only.

use std::borrow::Cow;

pub use super::content_snapshot::{
    ContentSelection, ContentSelectionError, ContentSnapshot, ContentSource, SnapshotEntry,
    SnapshotKind, SnapshotPayload, SnapshotToolCall, SnapshotToolResult,
};

use crate::Json;

use super::request::{ContentPart, Message, MessageContent, OpenAiImageUrl};
use super::response::AnnotatedLlmResponse;

/// The normalized object against which a content location is interpreted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "root", content = "message_index", rename_all = "snake_case")]
pub enum ContentRoot {
    /// A request's message at this index.
    RequestMessage(usize),
    /// A standalone message.
    Message,
    /// A normalized response.
    Response,
    /// Standalone content, including request instructions.
    Content,
}

/// Field containing an entry, relative to its normalized root.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ContentField {
    /// Message content, or standalone content itself.
    Content,
    /// The response's `message` field.
    Message,
    /// A separate `tool_calls` array.
    ToolCalls,
    /// A top-level tool-call message item itself.
    ToolCallItem,
    /// A top-level tool-result item's `output` field.
    Output,
    /// An opaque native message's `value` field.
    Value,
    /// Recognized native message content at `value.content`.
    NativeContent,
    /// Separate refusal at `value.refusal` on a native Chat assistant.
    NativeRefusal,
    /// Tool calls at `value.tool_calls` on a native Chat assistant.
    NativeToolCalls,
}

/// Read-time location; not a stable edit handle or provider-wire offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct ContentLocation {
    /// Owning normalized object.
    #[serde(flatten)]
    pub root: ContentRoot,
    /// Field within that object.
    pub field: ContentField,
    /// Content-part or tool-call array index; `None` for a scalar or whole payload.
    pub index: Option<usize>,
}

/// A payload in its existing representation. No parsing or rendering is performed.
#[derive(Debug, Clone, Copy)]
pub enum PayloadRef<'a> {
    /// An explicitly absent legacy function-result value.
    Null,
    /// String-valued arguments or results.
    String(&'a str),
    /// Parsed arguments or an arbitrary result payload.
    Json(&'a Json),
    /// A tool-result message's complete content, including multipart content.
    Content(&'a MessageContent),
}

/// Borrowed tool-call data, retaining the source's identifier distinctions.
#[derive(Debug, Clone, Copy)]
pub struct ToolCallView<'a> {
    /// Item or tool-call ID, when present in the source.
    pub id: Option<&'a str>,
    /// Separate correlation ID, when present in the source.
    pub call_id: Option<&'a str>,
    /// Tool/function name.
    pub name: &'a str,
    /// Arguments in their original normalized representation.
    pub arguments: PayloadRef<'a>,
}

/// Borrowed tool result. Its payload is excluded from default text extraction.
#[derive(Debug, Clone, Copy)]
pub struct ToolResultView<'a> {
    /// Item ID, when present.
    pub id: Option<&'a str>,
    /// Corresponding tool-call ID, when present.
    pub call_id: Option<&'a str>,
    /// Legacy function-result name, when present.
    pub name: Option<&'a str>,
    /// Result payload, without recursive text extraction.
    pub payload: PayloadRef<'a>,
    /// Explicit failure flag, when present.
    pub is_error: Option<bool>,
}

/// Typed content without cloning the normalized source.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum ContentKind<'a> {
    /// Explicit text.
    Text(&'a str),
    /// Explicit refusal text.
    Refusal(&'a str),
    /// A tool invocation.
    ToolCall(ToolCallView<'a>),
    /// A tool result, including string-valued results.
    ToolResult(ToolResultView<'a>),
    /// An image URL.
    ImageUrl(&'a OpenAiImageUrl),
    /// Portable image data.
    Image(&'a Json),
    /// Portable audio data.
    Audio(&'a Json),
    /// Portable file data.
    File(&'a Json),
    /// A provider-native payload retained without recursive interpretation.
    ProviderNative {
        /// Owning provider surface.
        provider: &'a str,
        /// Native discriminator.
        kind: &'a str,
        /// Original payload.
        value: &'a Json,
    },
}

/// One typed entry and its location in the normalized source.
#[derive(Debug, Clone, Copy)]
pub struct ContentEntry<'a> {
    /// Read-time source location.
    pub location: ContentLocation,
    /// Borrowed content.
    pub kind: ContentKind<'a>,
}

#[derive(Debug, Clone, Copy)]
enum Source<'a> {
    Message(&'a Message),
    Response(&'a AnnotatedLlmResponse),
    Content(&'a MessageContent),
}

/// Shared read-only view of a request item, response output, or instructions.
#[derive(Debug, Clone, Copy)]
pub struct MessageView<'a> {
    root: ContentRoot,
    source: Source<'a>,
}

/// Owned backing for lazy traversal after the original annotation is dropped.
pub enum RetainedContent {
    /// A selected history item (with its original index) or a standalone message.
    Message {
        /// The selected normalized message.
        message: Message,
        /// Index in the original request, if selected from history.
        request_index: Option<usize>,
    },
    /// Standalone content or request instructions.
    Content(MessageContent),
    /// Normalized response backing the output view.
    Response(Box<AnnotatedLlmResponse>),
}

impl RetainedContent {
    /// Borrow the selection with its original root, including for an empty message.
    pub fn view(&self) -> MessageView<'_> {
        match self {
            Self::Message {
                message,
                request_index: Some(index),
            } => MessageView::request_message(*index, message),
            Self::Message {
                message,
                request_index: None,
            } => message.content_view(),
            Self::Content(content) => content.content_view(),
            Self::Response(response) => response.output(),
        }
    }

    /// Read an entry from a saved cursor position without advancing the caller's cursor.
    pub fn entry(&self, mut cursor: ContentCursor) -> Option<ContentEntry<'_>> {
        cursor.next_part(self.view())
    }
}

/// Forward-only traversal state, independent of the borrowed source lifetime.
///
/// Keep one cursor per immutable selected view. Bindings can retain this state
/// beside owned snapshot storage without a self-referential iterator or copying
/// content. A copied cursor records a position for constant-time entry access.
#[derive(Debug, Default, Clone, Copy)]
pub struct ContentCursor {
    index: usize,
    calls: bool,
    done: bool,
}

impl ContentCursor {
    /// Read the next typed part. Exhaustion is permanent for this cursor.
    pub fn next_part<'a>(&mut self, view: MessageView<'a>) -> Option<ContentEntry<'a>> {
        if self.done {
            return None;
        }
        if !self.calls {
            if let Some(entry) = view.primary(self.index) {
                self.index += 1;
                return Some(entry);
            }
            self.calls = true;
            self.index = 0;
            if let Some(refusal) = view
                .native_chat_assistant()
                .and_then(|value| value.get("refusal"))
                .and_then(Json::as_str)
            {
                return Some(view.entry(
                    ContentField::NativeRefusal,
                    None,
                    ContentKind::Refusal(refusal),
                ));
            }
        }
        let entry = view.tool_call(self.index);
        if entry.is_some() {
            self.index += 1;
        } else {
            self.done = true;
        }
        entry
    }

    /// Read the next eligible borrowed string using the shared extraction policy.
    pub fn next_text<'a>(&mut self, view: MessageView<'a>) -> Option<&'a str> {
        while let Some(entry) = self.next_part(view) {
            if let Some(text) = view.entry_text(entry) {
                return Some(text);
            }
        }
        None
    }
}

impl<'a> MessageView<'a> {
    /// Normalized root of this selection, including when it contains no parts.
    pub fn root(self) -> ContentRoot {
        self.root
    }

    /// Copy the selected source for lazy traversal. For output, retain only the
    /// response message and tool calls. Parts and eligible text are read on demand.
    pub fn retain(self) -> RetainedContent {
        match self.source {
            Source::Message(message) => RetainedContent::Message {
                message: message.clone(),
                request_index: match self.root {
                    ContentRoot::RequestMessage(index) => Some(index),
                    _ => None,
                },
            },
            Source::Content(content) => RetainedContent::Content(content.clone()),
            Source::Response(response) => {
                RetainedContent::Response(Box::new(AnnotatedLlmResponse {
                    message: response.message.clone(),
                    tool_calls: response.tool_calls.clone(),
                    ..Default::default()
                }))
            }
        }
    }

    /// Borrow the backing normalized item so callers can retain selected storage.
    pub fn source(self) -> ContentSource<'a> {
        match self.source {
            Source::Message(message) => ContentSource::Message(message),
            Source::Response(response) => ContentSource::Response(response),
            Source::Content(content) => ContentSource::Content(content),
        }
    }

    pub(super) fn request_message(index: usize, message: &'a Message) -> Self {
        Self {
            root: ContentRoot::RequestMessage(index),
            source: Source::Message(message),
        }
    }

    pub(super) fn response(response: &'a AnnotatedLlmResponse) -> Self {
        Self {
            root: ContentRoot::Response,
            source: Source::Response(response),
        }
    }

    /// Enumerate content in normalized array order, then separate tool calls.
    ///
    /// Tool-result messages yield one result containing the entire payload.
    /// Original provider ordering between separate fields is not recoverable.
    pub fn parts(self) -> impl Iterator<Item = ContentEntry<'a>> + 'a {
        let mut cursor = ContentCursor::default();
        std::iter::from_fn(move || cursor.next_part(self))
    }

    /// Iterate explicit text/refusal strings, including known native fallbacks.
    ///
    /// Tool payloads, media, reasoning, and arbitrary nested JSON are excluded.
    pub fn text_parts(self) -> impl Iterator<Item = &'a str> + 'a {
        let mut cursor = ContentCursor::default();
        std::iter::from_fn(move || cursor.next_text(self))
    }

    fn entry_text(self, entry: ContentEntry<'a>) -> Option<&'a str> {
        // Envelopes and native tool calls can carry diagnostic text in extension fields.
        if matches!(
            entry.location.field,
            ContentField::Value | ContentField::NativeToolCalls
        ) {
            return None;
        }
        match entry.kind {
            ContentKind::Text(text) | ContentKind::Refusal(text) => Some(text),
            ContentKind::ProviderNative { kind, value, .. } => native_text(kind, value),
            _ => None,
        }
    }

    /// Borrow a single text part, or join multiple parts with newlines.
    ///
    /// Empty strings and whitespace are preserved. No eligible parts yields `None`.
    pub fn text(self) -> Option<Cow<'a, str>> {
        let mut parts = self.text_parts();
        let first = parts.next()?;
        let Some(second) = parts.next() else {
            return Some(Cow::Borrowed(first));
        };
        let mut joined = String::with_capacity(first.len() + 1 + second.len());
        joined.push_str(first);
        joined.push('\n');
        joined.push_str(second);
        for part in parts {
            joined.push('\n');
            joined.push_str(part);
        }
        Some(Cow::Owned(joined))
    }

    fn entry(
        self,
        field: ContentField,
        index: Option<usize>,
        kind: ContentKind<'a>,
    ) -> ContentEntry<'a> {
        ContentEntry {
            location: ContentLocation {
                root: self.root,
                field,
                index,
            },
            kind,
        }
    }

    fn content(
        self,
        content: &'a MessageContent,
        field: ContentField,
        index: usize,
    ) -> Option<ContentEntry<'a>> {
        match content {
            MessageContent::Text(text) => {
                (index == 0).then(|| self.entry(field, None, ContentKind::Text(text)))
            }
            MessageContent::Parts(parts) => {
                Some(self.entry(field, Some(index), part_kind(parts.get(index)?)))
            }
        }
    }

    fn tool_call(self, index: usize) -> Option<ContentEntry<'a>> {
        let call = match self.source {
            Source::Message(Message::Assistant {
                tool_calls: Some(calls),
                ..
            }) => calls.get(index).map(|call| ToolCallView {
                id: Some(&call.id),
                call_id: None,
                name: &call.function.name,
                arguments: PayloadRef::String(&call.function.arguments),
            }),
            Source::Response(response) => response
                .tool_calls
                .as_ref()
                .and_then(|calls| calls.get(index))
                .map(|call| ToolCallView {
                    id: Some(&call.id),
                    call_id: None,
                    name: &call.name,
                    arguments: PayloadRef::Json(&call.arguments),
                }),
            _ => {
                let call = self
                    .native_chat_assistant()?
                    .get("tool_calls")?
                    .as_array()?
                    .get(index)?;
                return Some(self.entry(
                    ContentField::NativeToolCalls,
                    Some(index),
                    native_chat_call_kind(call),
                ));
            }
        }?;
        Some(self.entry(
            ContentField::ToolCalls,
            Some(index),
            ContentKind::ToolCall(call),
        ))
    }

    fn native_chat_assistant(self) -> Option<&'a Json> {
        match self.source {
            Source::Message(
                message @ Message::ProviderNative {
                    provider, value, ..
                },
            ) if provider == "openai_chat"
                && message.native_message_role() == Some("assistant") =>
            {
                Some(value)
            }
            _ => None,
        }
    }

    fn primary(self, index: usize) -> Option<ContentEntry<'a>> {
        match self.source {
            Source::Content(content) => self.content(content, ContentField::Content, index),
            Source::Response(response) => {
                self.content(response.message.as_ref()?, ContentField::Message, index)
            }
            Source::Message(message) => match message {
                Message::System { content, .. }
                | Message::Developer { content, .. }
                | Message::User { content, .. } => {
                    self.content(content, ContentField::Content, index)
                }
                Message::Assistant { content, .. } => {
                    self.content(content.as_ref()?, ContentField::Content, index)
                }
                Message::Tool {
                    content,
                    tool_call_id,
                } => (index == 0).then(|| {
                    self.entry(
                        ContentField::Content,
                        None,
                        ContentKind::ToolResult(ToolResultView {
                            id: None,
                            call_id: Some(tool_call_id),
                            name: None,
                            payload: PayloadRef::Content(content),
                            is_error: None,
                        }),
                    )
                }),
                Message::Function { content, name } => {
                    let payload = content
                        .as_deref()
                        .map_or(PayloadRef::Null, PayloadRef::String);
                    (index == 0).then(|| {
                        self.entry(
                            ContentField::Content,
                            None,
                            ContentKind::ToolResult(ToolResultView {
                                id: None,
                                call_id: None,
                                name: Some(name),
                                payload,
                                is_error: None,
                            }),
                        )
                    })
                }
                Message::ToolCallItem {
                    id,
                    call_id,
                    name,
                    arguments,
                    ..
                } => (index == 0).then(|| {
                    self.entry(
                        ContentField::ToolCallItem,
                        None,
                        ContentKind::ToolCall(ToolCallView {
                            id: id.as_deref(),
                            call_id: Some(call_id),
                            name,
                            arguments: PayloadRef::Json(arguments),
                        }),
                    )
                }),
                Message::ToolResultItem {
                    id,
                    call_id,
                    output,
                    ..
                } => (index == 0).then(|| {
                    self.entry(
                        ContentField::Output,
                        None,
                        ContentKind::ToolResult(ToolResultView {
                            id: id.as_deref(),
                            call_id: Some(call_id),
                            name: None,
                            payload: PayloadRef::Json(output),
                            is_error: None,
                        }),
                    )
                }),
                Message::ProviderNative {
                    provider,
                    kind,
                    value,
                } => {
                    if message.native_message_role().is_some() {
                        match value.get("content") {
                            Some(Json::String(text)) => {
                                return (index == 0).then(|| {
                                    self.entry(
                                        ContentField::NativeContent,
                                        None,
                                        ContentKind::Text(text),
                                    )
                                });
                            }
                            Some(Json::Array(parts)) => {
                                let part = parts.get(index)?;
                                return Some(self.entry(
                                    ContentField::NativeContent,
                                    Some(index),
                                    ContentKind::ProviderNative {
                                        provider,
                                        kind:
                                            part.get("type").and_then(Json::as_str).unwrap_or(kind),
                                        value: part,
                                    },
                                ));
                            }
                            None | Some(Json::Null)
                                if provider == "openai_chat"
                                    && kind == "assistant"
                                    && (value.get("refusal").is_some_and(Json::is_string)
                                        || value
                                            .get("tool_calls")
                                            .and_then(Json::as_array)
                                            .is_some_and(|calls| !calls.is_empty())) =>
                            {
                                return None;
                            }
                            _ => {}
                        }
                    }
                    (index == 0).then(|| {
                        self.entry(
                            ContentField::Value,
                            None,
                            ContentKind::ProviderNative {
                                provider,
                                kind,
                                value,
                            },
                        )
                    })
                }
            },
        }
    }
}

impl Message {
    /// Return the role of a recognized provider-native message envelope.
    ///
    /// Keep selection independent of content availability so textless native
    /// messages still establish the same boundary as portable messages.
    pub fn native_message_role(&self) -> Option<&str> {
        let Self::ProviderNative {
            provider,
            kind,
            value,
        } = self
        else {
            return None;
        };
        let role = value.get("role")?.as_str()?;
        if value
            .get("type")
            .is_some_and(|value| value.as_str() != Some("message"))
        {
            return None;
        }
        let recognized = match provider.as_str() {
            "openai_chat" => {
                kind == role && matches!(role, "system" | "developer" | "user" | "assistant")
            }
            "anthropic_messages" => kind == role && matches!(role, "system" | "user" | "assistant"),
            "openai_responses" => {
                kind == "message" && matches!(role, "system" | "developer" | "user" | "assistant")
            }
            _ => false,
        };
        recognized.then_some(role)
    }

    /// Inspect every part of this item without copying its content.
    pub fn content_view(&self) -> MessageView<'_> {
        MessageView {
            root: ContentRoot::Message,
            source: Source::Message(self),
        }
    }
}

impl MessageContent {
    /// Inspect standalone content, including provider-level instructions.
    pub fn content_view(&self) -> MessageView<'_> {
        MessageView {
            root: ContentRoot::Content,
            source: Source::Content(self),
        }
    }
}

fn native_chat_call_kind(call: &Json) -> ContentKind<'_> {
    let kind = call.get("type").and_then(Json::as_str);
    let function = call.get("function");
    if let (Some("function"), Some(id), Some(name), Some(arguments)) = (
        kind,
        call.get("id").and_then(Json::as_str),
        function.and_then(|f| f.get("name")).and_then(Json::as_str),
        function
            .and_then(|f| f.get("arguments"))
            .and_then(Json::as_str),
    ) {
        return ContentKind::ToolCall(ToolCallView {
            id: Some(id),
            call_id: None,
            name,
            arguments: PayloadRef::String(arguments),
        });
    }
    ContentKind::ProviderNative {
        provider: "openai_chat",
        kind: kind.unwrap_or("tool_call"),
        value: call,
    }
}

fn native_text<'a>(kind: &str, value: &'a Json) -> Option<&'a str> {
    // Preserve the existing shallow fallback, but do not expose reasoning or
    // tool payloads merely because a native block also contains a text field.
    if value.get("thought").and_then(Json::as_bool) == Some(true)
        || native_non_text(kind)
        || value
            .get("type")
            .and_then(Json::as_str)
            .is_some_and(native_non_text)
    {
        return None;
    }
    value
        .get("text")
        .and_then(Json::as_str)
        .or_else(|| value.get("refusal").and_then(Json::as_str))
}

fn native_non_text(kind: &str) -> bool {
    matches!(
        kind,
        "reasoning"
            | "thinking"
            | "redacted_thinking"
            | "tool"
            | "function"
            | "tool_use"
            | "mcp_tool_use"
            | "server_tool_use"
            | "tool_result"
            | "function_call"
            | "function_call_output"
            | "custom_tool_call"
            | "custom_tool_call_output"
            | "image"
            | "image_url"
            | "input_image"
            | "audio"
            | "input_audio"
            | "file"
            | "input_file"
            | "document"
    )
}

fn part_kind(part: &ContentPart) -> ContentKind<'_> {
    match part {
        ContentPart::Text { text, .. } => ContentKind::Text(text),
        ContentPart::Refusal { refusal, .. } => ContentKind::Refusal(refusal),
        ContentPart::ImageUrl { image_url, .. } => ContentKind::ImageUrl(image_url),
        ContentPart::Image { image, .. } => ContentKind::Image(image),
        ContentPart::Audio { audio, .. } => ContentKind::Audio(audio),
        ContentPart::File { file, .. } => ContentKind::File(file),
        ContentPart::ToolUse {
            id, name, input, ..
        } => ContentKind::ToolCall(ToolCallView {
            id: Some(id),
            call_id: None,
            name,
            arguments: PayloadRef::Json(input),
        }),
        ContentPart::ToolResult {
            tool_use_id,
            content,
            is_error,
            ..
        } => ContentKind::ToolResult(ToolResultView {
            id: None,
            call_id: Some(tool_use_id),
            name: None,
            payload: PayloadRef::Json(content),
            is_error: *is_error,
        }),
        ContentPart::ProviderNative {
            provider,
            kind,
            value,
        } => ContentKind::ProviderNative {
            provider,
            kind,
            value,
        },
    }
}
