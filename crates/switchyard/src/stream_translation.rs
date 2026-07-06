// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Incremental provider-neutral SSE transcoding.

use std::collections::BTreeMap;

use nemo_relay::codec::streaming::NormalizedStreamEvent;
use nemo_relay::error::{FlowError, Result};
use serde_json::{Value as Json, json};

use crate::component::WireProtocol;

pub(crate) struct StreamTranscoder {
    source: WireProtocol,
    target: WireProtocol,
    source_tools: BTreeMap<usize, String>,
    target_tools: BTreeMap<String, usize>,
    target_tool_names: BTreeMap<String, String>,
    target_tool_arguments: BTreeMap<String, String>,
    next_index: usize,
    text_index: Option<usize>,
    open_block: Option<usize>,
    started: bool,
    usage: Option<Json>,
    text: String,
}

impl StreamTranscoder {
    pub(crate) fn new(source: WireProtocol, target: WireProtocol) -> Self {
        Self {
            source,
            target,
            source_tools: BTreeMap::new(),
            target_tools: BTreeMap::new(),
            target_tool_names: BTreeMap::new(),
            target_tool_arguments: BTreeMap::new(),
            next_index: 0,
            text_index: None,
            open_block: None,
            started: false,
            usage: None,
            text: String::new(),
        }
    }

    pub(crate) fn transcode(&mut self, chunk: &Json) -> Result<Vec<Json>> {
        if unsupported_stream_chunk(self.source, chunk) {
            return Err(FlowError::InvalidArgument(
                "provider-specific streaming extension cannot be translated safely".into(),
            ));
        }
        let events = match self.source {
            WireProtocol::OpenaiChat => self.decode_chat(chunk),
            WireProtocol::OpenaiResponses => self.decode_responses(chunk),
            WireProtocol::AnthropicMessages => self.decode_anthropic(chunk),
        };
        let mut output = Vec::new();
        for event in events {
            match self.target {
                WireProtocol::OpenaiChat => self.encode_chat(event, &mut output),
                WireProtocol::OpenaiResponses => self.encode_responses(event, &mut output),
                WireProtocol::AnthropicMessages => self.encode_anthropic(event, &mut output),
            }
        }
        Ok(output)
    }

    fn decode_chat(&mut self, chunk: &Json) -> Vec<NormalizedStreamEvent> {
        if let Some(error) = chunk.get("error") {
            return vec![NormalizedStreamEvent::Error {
                error: error.clone(),
            }];
        }
        let mut events = Vec::new();
        if let Some(choice) = chunk["choices"]
            .as_array()
            .and_then(|choices| choices.first())
        {
            let delta = &choice["delta"];
            if let Some(text) = delta.get("content").and_then(Json::as_str) {
                events.push(NormalizedStreamEvent::TextDelta { text: text.into() });
            }
            for call in delta["tool_calls"].as_array().into_iter().flatten() {
                let index = call.get("index").and_then(Json::as_u64).unwrap_or(0) as usize;
                let id = call
                    .get("id")
                    .and_then(Json::as_str)
                    .map(ToOwned::to_owned)
                    .or_else(|| self.source_tools.get(&index).cloned())
                    .unwrap_or_else(|| format!("call-{index}"));
                if let Some(name) = call["function"].get("name").and_then(Json::as_str) {
                    self.source_tools.insert(index, id.clone());
                    events.push(NormalizedStreamEvent::ToolCallStart {
                        id: id.clone(),
                        name: name.into(),
                    });
                }
                if let Some(arguments) = call["function"].get("arguments").and_then(Json::as_str)
                    && !arguments.is_empty()
                {
                    events.push(NormalizedStreamEvent::ToolCallArgumentsDelta {
                        id,
                        delta: arguments.into(),
                    });
                }
            }
            if let Some(reason) = choice.get("finish_reason").and_then(Json::as_str) {
                events.push(NormalizedStreamEvent::Finish {
                    reason: Some(reason.into()),
                });
            }
        }
        if let Some(usage) = chunk.get("usage").filter(|usage| !usage.is_null()) {
            events.push(NormalizedStreamEvent::Usage {
                usage: normalize_usage(WireProtocol::OpenaiChat, usage),
            });
        }
        events
    }

    fn decode_anthropic(&mut self, chunk: &Json) -> Vec<NormalizedStreamEvent> {
        let mut events = Vec::new();
        let index = chunk.get("index").and_then(Json::as_u64).unwrap_or(0) as usize;
        match chunk.get("type").and_then(Json::as_str) {
            Some("content_block_start") if chunk["content_block"]["type"] == "tool_use" => {
                let block = &chunk["content_block"];
                let id = block["id"].as_str().unwrap_or("call-relay").to_string();
                self.source_tools.insert(index, id.clone());
                events.push(NormalizedStreamEvent::ToolCallStart {
                    id: id.clone(),
                    name: block["name"].as_str().unwrap_or("tool").into(),
                });
                if let Some(input) = block.get("input").filter(|input| {
                    !input.is_null() && input.as_object().is_none_or(|value| !value.is_empty())
                }) {
                    events.push(NormalizedStreamEvent::ToolCallArgumentsDelta {
                        id,
                        delta: input.to_string(),
                    });
                }
            }
            Some("content_block_start") => {
                if let Some(text) = chunk["content_block"].get("text").and_then(Json::as_str)
                    && !text.is_empty()
                {
                    events.push(NormalizedStreamEvent::TextDelta { text: text.into() });
                }
            }
            Some("content_block_delta") if chunk["delta"]["type"] == "text_delta" => {
                if let Some(text) = chunk["delta"].get("text").and_then(Json::as_str) {
                    events.push(NormalizedStreamEvent::TextDelta { text: text.into() });
                }
            }
            Some("content_block_delta") if chunk["delta"]["type"] == "input_json_delta" => {
                if let Some(delta) = chunk["delta"].get("partial_json").and_then(Json::as_str) {
                    events.push(NormalizedStreamEvent::ToolCallArgumentsDelta {
                        id: self
                            .source_tools
                            .get(&index)
                            .cloned()
                            .unwrap_or_else(|| format!("call-{index}")),
                        delta: delta.into(),
                    });
                }
            }
            Some("message_delta") => {
                if let Some(usage) = chunk.get("usage") {
                    events.push(NormalizedStreamEvent::Usage {
                        usage: normalize_usage(WireProtocol::AnthropicMessages, usage),
                    });
                }
                if let Some(reason) = chunk["delta"].get("stop_reason").and_then(Json::as_str) {
                    events.push(NormalizedStreamEvent::Finish {
                        reason: Some(reason.into()),
                    });
                }
            }
            Some("error") => events.push(NormalizedStreamEvent::Error {
                error: chunk.get("error").cloned().unwrap_or_else(|| chunk.clone()),
            }),
            _ => {}
        }
        events
    }

    fn decode_responses(&mut self, chunk: &Json) -> Vec<NormalizedStreamEvent> {
        let mut events = Vec::new();
        let index = chunk
            .get("output_index")
            .and_then(Json::as_u64)
            .unwrap_or(0) as usize;
        match chunk.get("type").and_then(Json::as_str) {
            Some("response.output_text.delta") => {
                if let Some(text) = chunk.get("delta").and_then(Json::as_str) {
                    events.push(NormalizedStreamEvent::TextDelta { text: text.into() });
                }
            }
            Some("response.output_item.added") if chunk["item"]["type"] == "function_call" => {
                let item = &chunk["item"];
                let id = item
                    .get("call_id")
                    .or_else(|| item.get("id"))
                    .and_then(Json::as_str)
                    .unwrap_or("call-relay")
                    .to_string();
                self.source_tools.insert(index, id.clone());
                events.push(NormalizedStreamEvent::ToolCallStart {
                    id: id.clone(),
                    name: item["name"].as_str().unwrap_or("tool").into(),
                });
                if let Some(arguments) = item.get("arguments").and_then(Json::as_str)
                    && !arguments.is_empty()
                {
                    events.push(NormalizedStreamEvent::ToolCallArgumentsDelta {
                        id,
                        delta: arguments.into(),
                    });
                }
            }
            Some("response.function_call_arguments.delta") => {
                if let Some(delta) = chunk.get("delta").and_then(Json::as_str) {
                    events.push(NormalizedStreamEvent::ToolCallArgumentsDelta {
                        id: chunk
                            .get("item_id")
                            .and_then(Json::as_str)
                            .map(ToOwned::to_owned)
                            .or_else(|| self.source_tools.get(&index).cloned())
                            .unwrap_or_else(|| format!("call-{index}")),
                        delta: delta.into(),
                    });
                }
            }
            Some("response.completed" | "response.incomplete") => {
                if let Some(usage) = chunk["response"].get("usage") {
                    events.push(NormalizedStreamEvent::Usage {
                        usage: normalize_usage(WireProtocol::OpenaiResponses, usage),
                    });
                }
                events.push(NormalizedStreamEvent::Finish {
                    reason: chunk["response"]
                        .get("status")
                        .and_then(Json::as_str)
                        .map(ToOwned::to_owned),
                });
            }
            Some("response.failed" | "error") => events.push(NormalizedStreamEvent::Error {
                error: chunk
                    .get("error")
                    .or_else(|| chunk["response"].get("error"))
                    .cloned()
                    .unwrap_or_else(|| chunk.clone()),
            }),
            _ => {}
        }
        events
    }

    fn encode_chat(&mut self, event: NormalizedStreamEvent, output: &mut Vec<Json>) {
        match event {
            NormalizedStreamEvent::TextDelta { text } => {
                let role = (!self.started).then_some("assistant");
                self.started = true;
                output.push(json!({
                    "id": "chatcmpl-relay", "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"role": role, "content": text}, "finish_reason": null}]
                }));
            }
            NormalizedStreamEvent::ToolCallStart { id, name } => {
                let index = self.next_index;
                self.next_index += 1;
                self.target_tools.insert(id.clone(), index);
                output.push(json!({
                    "id": "chatcmpl-relay", "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"tool_calls": [{"index": index, "id": id, "type": "function", "function": {"name": name, "arguments": ""}}]}, "finish_reason": null}]
                }));
            }
            NormalizedStreamEvent::ToolCallArgumentsDelta { id, delta } => {
                let index = self.target_tools.get(&id).copied().unwrap_or(0);
                output.push(json!({
                    "id": "chatcmpl-relay", "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"tool_calls": [{"index": index, "function": {"arguments": delta}}]}, "finish_reason": null}]
                }));
            }
            NormalizedStreamEvent::Finish { reason } => output.push(json!({
                "id": "chatcmpl-relay", "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": chat_finish(reason.as_deref())}]
            })),
            NormalizedStreamEvent::Usage { usage } => output.push(json!({
                "id": "chatcmpl-relay", "object": "chat.completion.chunk", "choices": [], "usage": usage_for_target(WireProtocol::OpenaiChat, &usage)
            })),
            NormalizedStreamEvent::Error { error } => output.push(json!({"error": error})),
        }
    }

    fn ensure_anthropic_started(&mut self, output: &mut Vec<Json>) {
        if !self.started {
            self.started = true;
            output.push(json!({
                "type": "message_start",
                "message": {"id": "msg_relay", "type": "message", "role": "assistant", "content": [], "stop_reason": null, "usage": {"input_tokens": 0, "output_tokens": 0}}
            }));
        }
    }

    fn close_anthropic_block(&mut self, output: &mut Vec<Json>) {
        if let Some(index) = self.open_block.take() {
            output.push(json!({"type": "content_block_stop", "index": index}));
        }
    }

    fn encode_anthropic(&mut self, event: NormalizedStreamEvent, output: &mut Vec<Json>) {
        self.ensure_anthropic_started(output);
        match event {
            NormalizedStreamEvent::TextDelta { text } => {
                self.text.push_str(&text);
                let index = if let Some(index) = self.text_index {
                    index
                } else {
                    self.close_anthropic_block(output);
                    let index = self.next_index;
                    self.next_index += 1;
                    self.text_index = Some(index);
                    self.open_block = Some(index);
                    output.push(json!({"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}}));
                    index
                };
                output.push(json!({"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": text}}));
            }
            NormalizedStreamEvent::ToolCallStart { id, name } => {
                self.close_anthropic_block(output);
                self.text_index = None;
                let index = self.next_index;
                self.next_index += 1;
                self.target_tools.insert(id.clone(), index);
                self.target_tool_names.insert(id.clone(), name.clone());
                self.target_tool_arguments.insert(id.clone(), String::new());
                self.open_block = Some(index);
                output.push(json!({"type": "content_block_start", "index": index, "content_block": {"type": "tool_use", "id": id, "name": name, "input": {}}}));
            }
            NormalizedStreamEvent::ToolCallArgumentsDelta { id, delta } => {
                let index = self.target_tools.get(&id).copied().unwrap_or(0);
                output.push(json!({"type": "content_block_delta", "index": index, "delta": {"type": "input_json_delta", "partial_json": delta}}));
            }
            NormalizedStreamEvent::Usage { usage } => self.usage = Some(usage),
            NormalizedStreamEvent::Finish { reason } => {
                self.close_anthropic_block(output);
                output.push(json!({
                    "type": "message_delta", "delta": {"stop_reason": anthropic_finish(reason.as_deref())},
                    "usage": self.usage.as_ref().map(|usage| usage_for_target(WireProtocol::AnthropicMessages, usage)).unwrap_or_else(|| json!({"output_tokens": 0}))
                }));
                output.push(json!({"type": "message_stop"}));
            }
            NormalizedStreamEvent::Error { error } => {
                output.push(json!({"type": "error", "error": error}));
            }
        }
    }

    fn ensure_responses_started(&mut self, output: &mut Vec<Json>) {
        if !self.started {
            self.started = true;
            output.push(json!({
                "type": "response.created",
                "response": {"id": "resp_relay", "object": "response", "status": "in_progress", "output": []}
            }));
        }
    }

    fn encode_responses(&mut self, event: NormalizedStreamEvent, output: &mut Vec<Json>) {
        self.ensure_responses_started(output);
        match event {
            NormalizedStreamEvent::TextDelta { text } => {
                self.text.push_str(&text);
                let index = if let Some(index) = self.text_index {
                    index
                } else {
                    let index = self.next_index;
                    self.next_index += 1;
                    self.text_index = Some(index);
                    output.push(json!({"type": "response.output_item.added", "output_index": index, "item": {"id": "msg_relay", "type": "message", "role": "assistant", "status": "in_progress", "content": []}}));
                    index
                };
                output.push(json!({"type": "response.output_text.delta", "output_index": index, "content_index": 0, "delta": text}));
            }
            NormalizedStreamEvent::ToolCallStart { id, name } => {
                let index = self.next_index;
                self.next_index += 1;
                self.target_tools.insert(id.clone(), index);
                self.target_tool_names.insert(id.clone(), name.clone());
                self.target_tool_arguments.insert(id.clone(), String::new());
                output.push(json!({"type": "response.output_item.added", "output_index": index, "item": {"id": id, "type": "function_call", "call_id": id, "name": name, "arguments": "", "status": "in_progress"}}));
            }
            NormalizedStreamEvent::ToolCallArgumentsDelta { id, delta } => {
                let index = self.target_tools.get(&id).copied().unwrap_or(0);
                self.target_tool_arguments
                    .entry(id.clone())
                    .or_default()
                    .push_str(&delta);
                output.push(json!({"type": "response.function_call_arguments.delta", "output_index": index, "item_id": id, "delta": delta}));
            }
            NormalizedStreamEvent::Usage { usage } => self.usage = Some(usage),
            NormalizedStreamEvent::Finish { .. } => {
                let mut items = Vec::new();
                if let Some(index) = self.text_index {
                    let item = json!({
                        "id": "msg_relay", "type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": self.text, "annotations": []}]
                    });
                    output.push(json!({"type": "response.output_text.done", "output_index": index, "content_index": 0, "text": self.text}));
                    output.push(json!({"type": "response.output_item.done", "output_index": index, "item": item}));
                    items.push(item);
                }
                for (id, index) in &self.target_tools {
                    let arguments = self
                        .target_tool_arguments
                        .get(id)
                        .cloned()
                        .unwrap_or_default();
                    let item = json!({
                        "id": id, "type": "function_call", "call_id": id,
                        "name": self.target_tool_names.get(id).cloned().unwrap_or_else(|| "tool".into()),
                        "arguments": arguments, "status": "completed"
                    });
                    output.push(json!({"type": "response.function_call_arguments.done", "output_index": index, "item_id": id, "arguments": arguments}));
                    output.push(json!({"type": "response.output_item.done", "output_index": index, "item": item}));
                    items.push(item);
                }
                output.push(json!({
                    "type": "response.completed",
                    "response": {"id": "resp_relay", "object": "response", "status": "completed", "output": items, "usage": self.usage.as_ref().map(|usage| usage_for_target(WireProtocol::OpenaiResponses, usage))}
                }));
            }
            NormalizedStreamEvent::Error { error } => output.push(json!({
                "type": "response.failed", "response": {"id": "resp_relay", "status": "failed", "error": error}
            })),
        }
    }
}

fn unsupported_stream_chunk(source: WireProtocol, chunk: &Json) -> bool {
    match source {
        WireProtocol::OpenaiChat => {
            chunk["choices"]
                .as_array()
                .into_iter()
                .flatten()
                .any(|choice| {
                    choice["delta"].get("audio").is_some()
                        || choice["delta"].get("reasoning_content").is_some()
                })
        }
        WireProtocol::OpenaiResponses => {
            chunk.get("type").and_then(Json::as_str) == Some("response.output_item.added")
                && matches!(
                    chunk["item"].get("type").and_then(Json::as_str),
                    Some("reasoning" | "computer_call" | "web_search_call")
                )
        }
        WireProtocol::AnthropicMessages => {
            chunk.get("type").and_then(Json::as_str) == Some("content_block_start")
                && !matches!(
                    chunk["content_block"].get("type").and_then(Json::as_str),
                    Some("text" | "tool_use")
                )
        }
    }
}

fn chat_finish(reason: Option<&str>) -> &str {
    match reason {
        Some("tool_use") => "tool_calls",
        Some("max_tokens" | "incomplete") => "length",
        Some("content_filter") => "content_filter",
        _ => "stop",
    }
}

fn anthropic_finish(reason: Option<&str>) -> &str {
    match reason {
        Some("tool_calls") => "tool_use",
        Some("length" | "incomplete") => "max_tokens",
        _ => "end_turn",
    }
}

fn normalize_usage(source: WireProtocol, usage: &Json) -> Json {
    match source {
        WireProtocol::OpenaiChat => usage.clone(),
        WireProtocol::OpenaiResponses | WireProtocol::AnthropicMessages => {
            let prompt = usage.get("input_tokens").and_then(Json::as_u64);
            let completion = usage.get("output_tokens").and_then(Json::as_u64);
            json!({
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": usage.get("total_tokens").and_then(Json::as_u64).or_else(|| prompt.zip(completion).map(|(prompt, completion)| prompt + completion)),
            })
        }
    }
}

fn usage_for_target(target: WireProtocol, usage: &Json) -> Json {
    match target {
        WireProtocol::OpenaiChat => usage.clone(),
        WireProtocol::OpenaiResponses | WireProtocol::AnthropicMessages => json!({
            "input_tokens": usage.get("prompt_tokens").cloned(),
            "output_tokens": usage.get("completion_tokens").cloned(),
            "total_tokens": usage.get("total_tokens").cloned(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nemo_relay::codec::anthropic::AnthropicMessagesStreamingCodec;
    use nemo_relay::codec::openai_chat::OpenAIChatStreamingCodec;
    use nemo_relay::codec::openai_responses::OpenAIResponsesStreamingCodec;
    use nemo_relay::codec::streaming::StreamingCodec;

    fn source_chunks(source: WireProtocol) -> Vec<Json> {
        match source {
            WireProtocol::OpenaiChat => vec![
                json!({"choices": [{"delta": {"content": "hello"}, "finish_reason": null}]}),
                json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call-1", "function": {"name": "lookup", "arguments": "{\"key\":1}"}}]}, "finish_reason": null}]}),
                json!({"choices": [{"delta": {}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}),
            ],
            WireProtocol::AnthropicMessages => vec![
                json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
                json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hello"}}),
                json!({"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "call-1", "name": "lookup", "input": {}}}),
                json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"key\":1}"}}),
                json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"input_tokens": 3, "output_tokens": 2}}),
            ],
            WireProtocol::OpenaiResponses => vec![
                json!({"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": "hello"}),
                json!({"type": "response.output_item.added", "output_index": 1, "item": {"id": "fc-1", "type": "function_call", "call_id": "call-1", "name": "lookup", "arguments": ""}}),
                json!({"type": "response.function_call_arguments.delta", "output_index": 1, "item_id": "call-1", "delta": "{\"key\":1}"}),
                json!({"type": "response.completed", "response": {"id": "resp-1", "status": "completed", "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}}}),
            ],
        }
    }

    #[test]
    fn chat_text_delta_is_emitted_immediately_for_each_cross_protocol_target() {
        let chunk = json!({
            "id": "chat-1", "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": "hello"}, "finish_reason": null}]
        });
        for target in [
            WireProtocol::OpenaiResponses,
            WireProtocol::AnthropicMessages,
        ] {
            let mut transcoder = StreamTranscoder::new(WireProtocol::OpenaiChat, target);
            let output = transcoder.transcode(&chunk).unwrap();
            assert!(!output.is_empty(), "target={target:?}");
            assert!(output.iter().any(|item| item.to_string().contains("hello")));
        }
    }

    #[test]
    fn tool_argument_deltas_keep_their_call_identity() {
        let mut transcoder =
            StreamTranscoder::new(WireProtocol::OpenaiChat, WireProtocol::AnthropicMessages);
        let start = json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call-1", "function": {"name": "lookup", "arguments": ""}}]}, "finish_reason": null}]});
        let delta = json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"key\":"}}]}, "finish_reason": null}]});
        transcoder.transcode(&start).unwrap();
        let output = transcoder.transcode(&delta).unwrap();
        assert_eq!(output.last().unwrap()["delta"]["partial_json"], "{\"key\":");
    }

    #[test]
    fn every_cross_protocol_stream_pair_is_incremental_and_collectable() {
        let protocols = [
            WireProtocol::OpenaiChat,
            WireProtocol::OpenaiResponses,
            WireProtocol::AnthropicMessages,
        ];
        for source in protocols {
            for target in protocols {
                if source == target {
                    continue;
                }
                let mut transcoder = StreamTranscoder::new(source, target);
                let mut output = Vec::new();
                for source_chunk in source_chunks(source) {
                    let translated = transcoder.transcode(&source_chunk).unwrap();
                    if source_chunk.to_string().contains("hello") {
                        assert!(
                            translated
                                .iter()
                                .any(|chunk| chunk.to_string().contains("hello")),
                            "source={source:?} target={target:?}"
                        );
                    }
                    output.extend(translated);
                }
                let codec: Box<dyn StreamingCodec> = match target {
                    WireProtocol::OpenaiChat => Box::new(OpenAIChatStreamingCodec::new()),
                    WireProtocol::OpenaiResponses => Box::new(OpenAIResponsesStreamingCodec::new()),
                    WireProtocol::AnthropicMessages => {
                        Box::new(AnthropicMessagesStreamingCodec::new())
                    }
                };
                let mut collector = codec.collector();
                let finalizer = codec.finalizer();
                for chunk in output {
                    collector(chunk).unwrap();
                }
                let aggregate = finalizer();
                let annotated = crate::translation::decode_response(target, &aggregate).unwrap();
                assert!(
                    annotated.message.is_some(),
                    "source={source:?} target={target:?}"
                );
                assert_eq!(
                    annotated.tool_calls.as_ref().map(Vec::len),
                    Some(1),
                    "source={source:?} target={target:?} aggregate={aggregate}"
                );
            }
        }
    }
}
