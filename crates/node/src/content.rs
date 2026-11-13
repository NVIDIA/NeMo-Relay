// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust-backed snapshots used by the TypeScript content facade.

use napi::{Env, bindgen_prelude::Unknown};
use napi_derive::napi;
use nemo_relay_types::codec::content::{
    ContentEntry, ContentField, ContentKind, ContentRoot, ContentSelection, ContentSelectionError,
    ContentSource, MessageView, PayloadRef,
};
use nemo_relay_types::codec::request::{AnnotatedLlmRequest, Message, MessageContent};
use nemo_relay_types::codec::response::AnnotatedLlmResponse;
use serde_json::{Value as Json, json};

use crate::convert::{checked_json, parse_json};
use crate::error::{Error, Result};

enum Source {
    Message(Message),
    Content(MessageContent),
    Response(Box<AnnotatedLlmResponse>),
}

/// Internal native storage for the public codecs.MessageView facade.
#[napi(js_name = "_ContentSnapshot")]
pub struct ContentSnapshot {
    source: Source,
    request_index: Option<usize>,
}

impl ContentSnapshot {
    fn view(&self) -> MessageView<'_> {
        match &self.source {
            Source::Message(message) => message.content_view(),
            Source::Content(content) => content.content_view(),
            Source::Response(response) => response.output(),
        }
    }

    fn entry(&self, index: u32) -> Option<ContentEntry<'_>> {
        let mut entry = self.view().parts().nth(index as usize)?;
        if let Some(index) = self.request_index {
            entry.location.root = ContentRoot::RequestMessage(index);
        }
        Some(entry)
    }

    fn payload(&self, index: u32) -> Result<PayloadRef<'_>> {
        match self.entry(index).map(|entry| entry.kind) {
            Some(ContentKind::ToolCall(call)) => Ok(call.arguments),
            Some(ContentKind::ToolResult(result)) => Ok(result.payload),
            _ => Err(Error::invalid_argument("entry has no tool payload")),
        }
    }
}

fn payload_kind(payload: PayloadRef<'_>) -> &'static str {
    match payload {
        PayloadRef::Null => "null",
        PayloadRef::String(_) => "string",
        PayloadRef::Json(_) => "json",
        PayloadRef::Content(_) => "content",
    }
}

#[napi]
impl ContentSnapshot {
    /// Internal constructor for standalone content and current response output.
    #[napi(constructor, catch_unwind)]
    pub fn new(
        env: Env,
        mode: String,
        #[napi(ts_arg_type = "JsonValue")] value: Unknown,
        request_index: Option<u32>,
    ) -> Result<Self> {
        let value = checked_json(&env, value)?;
        let source = match mode.as_str() {
            "message" => Source::Message(parse_json(value, "normalized message")?),
            "content" => Source::Content(parse_json(value, "normalized content")?),
            "response" => {
                Source::Response(Box::new(parse_json(value, "normalized response output")?))
            }
            _ => return Err(Error::invalid_argument("unknown content snapshot source")),
        };
        Ok(Self {
            source,
            request_index: request_index.map(|index| index as usize),
        })
    }

    /// Select request content with the same Rust policy used by worker SDKs.
    #[napi(catch_unwind)]
    pub fn select_request(
        env: Env,
        mode: String,
        #[napi(ts_arg_type = "JsonValue")] value: Unknown,
        index: Option<f64>,
    ) -> Result<Option<Json>> {
        // The facade supplies role/envelope metadata only. Never deserialize
        // unselected tool/media payloads just to determine the selected index.
        let value = checked_json(&env, value)?;
        let request = AnnotatedLlmRequest {
            messages: value["messages"]
                .as_array()
                .ok_or_else(|| Error::invalid_argument("expected history metadata"))?
                .iter()
                .map(|item| {
                    serde_json::from_value(item.clone()).unwrap_or(Message::Assistant {
                        content: None,
                        tool_calls: None,
                        name: None,
                    })
                })
                .collect(),
            instructions: (!value["instructions"].is_null()).then(|| MessageContent::Parts(vec![])),
            ..Default::default()
        };
        let selection = match mode.as_str() {
            "last_message" => ContentSelection::LastMessage,
            "message" => {
                let index = index
                    .filter(|value| {
                        value.is_finite()
                            && value.fract() == 0.0
                            && value.abs() <= crate::convert::JAVASCRIPT_MAX_SAFE_INTEGER as f64
                    })
                    .ok_or_else(|| {
                        Error::invalid_argument("message index must be a safe integer")
                    })?;
                ContentSelection::Message(index as i64)
            }
            "instructions" => ContentSelection::Instructions,
            "system_prompt" => ContentSelection::SystemPrompt,
            "last_user_message" => ContentSelection::LastUserMessage,
            _ => return Err(Error::invalid_argument("unknown request content selection")),
        };
        let selected = selection
            .select(ContentSource::Request(&request))
            .map_err(|error| {
                if error == ContentSelectionError::IndexOutOfRange {
                    // Preserve the JavaScript RangeError contract for explicit indexing.
                    let _ = env.throw_range_error("Message index out of range", None);
                    Error::Napi(napi::Error::new(
                        napi::Status::PendingException,
                        error.to_string(),
                    ))
                } else {
                    Error::invalid_argument(error.to_string())
                }
            })?;
        Ok(selected.map(|view| match view.root() {
            ContentRoot::RequestMessage(index) => json!({"messageIndex": index}),
            ContentRoot::Content => json!({"instructions": true}),
            _ => unreachable!("request selection yields a message or instructions"),
        }))
    }

    /// Internal typed metadata without materializing media or tool payloads.
    #[napi(catch_unwind)]
    pub fn entry_metadata(&self, index: u32) -> Result<Json> {
        let Some(entry) = self.entry(index) else {
            return Ok(Json::Null);
        };
        let (root, message_index) = match entry.location.root {
            ContentRoot::RequestMessage(index) => ("request_message", Some(index)),
            ContentRoot::Message => ("message", None),
            ContentRoot::Response => ("response", None),
            ContentRoot::Content => ("content", None),
        };
        let field = match entry.location.field {
            ContentField::Content => "content",
            ContentField::Message => "message",
            ContentField::ToolCalls => "tool_calls",
            ContentField::ToolCallItem => "tool_call_item",
            ContentField::Output => "output",
            ContentField::Value => "value",
            ContentField::NativeContent => "native_content",
        };
        let mut meta = json!({"location": {"root": root, "messageIndex": message_index, "field": field, "index": entry.location.index}});
        let kind = match entry.kind {
            ContentKind::Text(text) => {
                meta["text"] = json!(text);
                "text"
            }
            ContentKind::Refusal(text) => {
                meta["text"] = json!(text);
                "refusal"
            }
            ContentKind::ToolCall(call) => {
                meta["toolCall"] = json!({"id": call.id, "callId": call.call_id, "name": call.name, "payloadKind": payload_kind(call.arguments)});
                "tool_call"
            }
            ContentKind::ToolResult(result) => {
                meta["toolResult"] = json!({"id": result.id, "callId": result.call_id, "name": result.name, "isError": result.is_error, "payloadKind": payload_kind(result.payload)});
                "tool_result"
            }
            ContentKind::ImageUrl(_) => "image_url",
            ContentKind::Image(_) => "image",
            ContentKind::Audio(_) => "audio",
            ContentKind::File(_) => "file",
            ContentKind::ProviderNative { provider, kind, .. } => {
                meta["provider"] = json!(provider);
                meta["nativeKind"] = json!(kind);
                "provider_native"
            }
            _ => {
                return Err(Error::invalid_argument(
                    "unsupported normalized content kind",
                ));
            }
        };
        meta["kind"] = json!(kind);
        Ok(meta)
    }

    /// Internal payload conversion directly from borrowed Rust data to a fresh JavaScript value.
    #[napi(ts_return_type = "JsonValue", catch_unwind)]
    pub fn payload_value(&self, env: Env, index: u32) -> Result<Unknown<'_>> {
        Ok(match self.payload(index)? {
            PayloadRef::Null => env.to_js_value(&())?,
            PayloadRef::String(text) => env.to_js_value(&text)?,
            PayloadRef::Json(value) => env.to_js_value(value)?,
            PayloadRef::Content(content) => env.to_js_value(content)?,
        })
    }

    /// Internal media/native conversion; returned values do not share mutable Rust storage.
    #[napi(ts_return_type = "JsonValue", catch_unwind)]
    pub fn value(&self, env: Env, index: u32) -> Result<Unknown<'_>> {
        Ok(match self.entry(index).map(|entry| entry.kind) {
            Some(ContentKind::ImageUrl(value)) => env.to_js_value(value)?,
            Some(
                ContentKind::Image(value)
                | ContentKind::Audio(value)
                | ContentKind::File(value)
                | ContentKind::ProviderNative { value, .. },
            ) => env.to_js_value(value)?,
            _ => env.to_js_value(&())?,
        })
    }

    /// Internal extraction using the shared Rust policy, preserving empty strings.
    #[napi(ts_return_type = "string | null", catch_unwind)]
    pub fn text(&self, env: Env) -> Result<Unknown<'_>> {
        Ok(env.to_js_value(&self.view().text())?)
    }

    /// Internal single text part conversion without an intermediate owned Rust string.
    #[napi(ts_return_type = "string | null", catch_unwind)]
    pub fn text_part(&self, env: Env, index: u32) -> Result<Unknown<'_>> {
        Ok(env.to_js_value(&self.view().text_parts().nth(index as usize))?)
    }
}
