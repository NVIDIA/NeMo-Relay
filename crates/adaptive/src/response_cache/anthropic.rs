// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Anthropic compaction cache eligibility and replay contracts.

use std::collections::BTreeMap;

use nemo_relay::api::llm::LlmRequest;
use serde::Serialize;
use serde_json::{Value as Json, json};

use crate::response_cache::mark::CacheReason;

const THRESHOLD_BETA: &str = "compact-2026-01-12";
const ANTHROPIC_API_VERSION: &str = "2023-06-01";
const MIN_THRESHOLD_TOKENS: u64 = 50_000;
const MAX_INSTRUCTIONS_CHARS: usize = 16_384;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(crate) enum AnthropicProtocol {
    ThresholdV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(crate) enum AnthropicOperation {
    PausedThreshold,
    ThresholdContinuation,
    ThresholdRecompact,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AnthropicResponseKind {
    Ordinary,
    ThresholdCompaction,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AnthropicCacheContext {
    pub(crate) version: String,
    pub(crate) beta_tokens: Vec<String>,
    pub(crate) protocol: AnthropicProtocol,
    pub(crate) operation: AnthropicOperation,
}

impl AnthropicCacheContext {
    pub(crate) fn key_value(&self) -> Json {
        json!({
            "schema_version": 1,
            "version": self.version,
            "betas": self.beta_tokens,
            "protocol": self.protocol,
            "operation": self.operation,
        })
    }
}

pub(crate) fn cache_context(
    request: &LlmRequest,
) -> Result<Option<AnthropicCacheContext>, CacheReason> {
    let beta_mentions_compaction = request.headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case("anthropic-beta")
            && value.as_str().is_some_and(|value| {
                value.split(',').any(|token| {
                    token
                        .trim()
                        .get(..8)
                        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("compact-"))
                })
            })
    });
    let Some(body) = request.content.as_object() else {
        return beta_mentions_compaction
            .then_some(CacheReason::AnthropicCompaction)
            .map_or(Ok(None), Err);
    };
    if body.get("messages").and_then(Json::as_array).is_none() {
        return beta_mentions_compaction
            .then_some(CacheReason::AnthropicCompaction)
            .map_or(Ok(None), Err);
    }

    let threshold_related = body
        .get("context_management")
        .is_some_and(context_management_is_compaction_related);
    let on_demand_related = body.contains_key("compaction");
    let continuation = contains_compaction_block(body);
    if !threshold_related && !on_demand_related && !continuation && !beta_mentions_compaction {
        return Ok(None);
    }

    let (version, beta_tokens) = protocol_headers(request)?;
    let compact_tokens: Vec<_> = beta_tokens
        .iter()
        .filter(|token| token.starts_with("compact-"))
        .map(String::as_str)
        .collect();
    let protocol = match compact_tokens.as_slice() {
        [token] if *token == THRESHOLD_BETA && !on_demand_related => AnthropicProtocol::ThresholdV1,
        _ => return Err(CacheReason::AnthropicCompaction),
    };

    if threshold_related
        && !body
            .get("context_management")
            .is_some_and(is_supported_threshold_request)
    {
        return Err(CacheReason::AnthropicCompaction);
    }
    if continuation && !has_supported_continuation(body) {
        return Err(CacheReason::AnthropicCompaction);
    }

    let operation = match (protocol, threshold_related, on_demand_related, continuation) {
        (AnthropicProtocol::ThresholdV1, true, false, false) => AnthropicOperation::PausedThreshold,
        (AnthropicProtocol::ThresholdV1, false, false, true) => {
            AnthropicOperation::ThresholdContinuation
        }
        (AnthropicProtocol::ThresholdV1, true, false, true) => {
            AnthropicOperation::ThresholdRecompact
        }
        _ => return Err(CacheReason::AnthropicCompaction),
    };

    Ok(Some(AnthropicCacheContext {
        version,
        beta_tokens,
        protocol,
        operation,
    }))
}

fn protocol_headers(request: &LlmRequest) -> Result<(String, Vec<String>), CacheReason> {
    let versions: Vec<_> = request
        .headers
        .iter()
        .filter(|(name, _)| name.eq_ignore_ascii_case("anthropic-version"))
        .collect();
    let [(_, version)] = versions.as_slice() else {
        return Err(CacheReason::AnthropicCompaction);
    };
    let version = version
        .as_str()
        .filter(|value| !value.trim().is_empty())
        .map(str::to_owned)
        .ok_or(CacheReason::AnthropicCompaction)?;
    if version != ANTHROPIC_API_VERSION {
        return Err(CacheReason::AnthropicCompaction);
    }

    let betas: Vec<_> = request
        .headers
        .iter()
        .filter(|(name, _)| name.eq_ignore_ascii_case("anthropic-beta"))
        .collect();
    let [(_, beta)] = betas.as_slice() else {
        return Err(CacheReason::AnthropicCompaction);
    };
    let beta = beta.as_str().ok_or(CacheReason::AnthropicCompaction)?;
    let beta_tokens: Vec<_> = beta.split(',').map(str::trim).map(str::to_owned).collect();
    if beta_tokens.is_empty() || beta_tokens.iter().any(String::is_empty) {
        return Err(CacheReason::AnthropicCompaction);
    }
    Ok((version, beta_tokens))
}

fn is_supported_threshold_request(value: &Json) -> bool {
    let Some(context) = value.as_object() else {
        return false;
    };
    if context.len() != 1 {
        return false;
    }
    let Some(edits) = context.get("edits").and_then(Json::as_array) else {
        return false;
    };
    let [edit] = edits.as_slice() else {
        return false;
    };
    let Some(edit) = edit.as_object() else {
        return false;
    };
    edit.keys().all(|key| {
        matches!(
            key.as_str(),
            "type" | "pause_after_compaction" | "trigger" | "instructions"
        )
    }) && edit.get("type").and_then(Json::as_str) == Some("compact_20260112")
        && edit.get("pause_after_compaction").and_then(Json::as_bool) == Some(true)
        && edit.get("trigger").is_none_or(valid_threshold_trigger)
        && edit.get("instructions").is_none_or(valid_instructions)
}

fn valid_threshold_trigger(value: &Json) -> bool {
    value.as_object().is_some_and(|trigger| {
        trigger.len() == 2
            && trigger.get("type").and_then(Json::as_str) == Some("input_tokens")
            && trigger
                .get("value")
                .and_then(Json::as_u64)
                .is_some_and(|value| value >= MIN_THRESHOLD_TOKENS)
    })
}

fn valid_instructions(value: &Json) -> bool {
    value.as_str().is_some_and(|value| {
        !value.trim().is_empty() && value.chars().count() <= MAX_INSTRUCTIONS_CHARS
    })
}

fn context_management_is_compaction_related(value: &Json) -> bool {
    let Some(object) = value.as_object() else {
        return true;
    };
    let Some(edits) = object.get("edits").and_then(Json::as_array) else {
        return true;
    };
    edits.iter().any(|edit| {
        edit.get("type")
            .and_then(Json::as_str)
            .is_none_or(|kind| kind == "compact" || kind.starts_with("compact_"))
    })
}

fn compaction_blocks(body: &serde_json::Map<String, Json>) -> Vec<&Json> {
    body.get("messages")
        .and_then(Json::as_array)
        .into_iter()
        .flatten()
        .filter_map(|message| message.get("content").and_then(Json::as_array))
        .flatten()
        .filter(|block| block.get("type").and_then(Json::as_str) == Some("compaction"))
        .collect()
}

fn contains_compaction_block(body: &serde_json::Map<String, Json>) -> bool {
    !compaction_blocks(body).is_empty()
}

fn has_supported_continuation(body: &serde_json::Map<String, Json>) -> bool {
    let blocks = compaction_blocks(body);
    !blocks.is_empty()
        && blocks
            .iter()
            .all(|block| threshold_request_block_is_valid(block))
}

fn cache_control_is_valid(value: &Json) -> bool {
    value.as_object().is_some_and(|object| {
        object.len() == 1 && object.get("type").and_then(Json::as_str) == Some("ephemeral")
    })
}

fn threshold_request_block_is_valid(block: &Json) -> bool {
    has_only_fields(
        block,
        &["type", "content", "encrypted_content", "cache_control"],
    ) && block.get("content").is_some_and(Json::is_string)
        && block
            .get("encrypted_content")
            .is_none_or(|value| value.is_null() || value.is_string())
        && block
            .get("cache_control")
            .is_none_or(cache_control_is_valid)
}

pub(crate) fn classify_aggregate(
    response: &Json,
    context: &AnthropicCacheContext,
) -> Option<AnthropicResponseKind> {
    if !has_only_fields(
        response,
        &[
            "id",
            "type",
            "role",
            "model",
            "content",
            "stop_reason",
            "stop_sequence",
            "usage",
            "context_management",
            "container",
            "stop_details",
            "diagnostics",
            "service_tier",
        ],
    ) || !response.get("id").is_some_and(Json::is_string)
        || response.get("type").and_then(Json::as_str) != Some("message")
        || response.get("role").and_then(Json::as_str) != Some("assistant")
        || !response.get("model").is_some_and(Json::is_string)
        || !response.get("stop_reason").is_some_and(Json::is_string)
        || !response
            .get("stop_sequence")
            .is_none_or(|value| value.is_null() || value.is_string())
        || !response
            .get("context_management")
            .is_none_or(context_management_response_is_valid)
        || !nullable_response_metadata_is_valid(
            response,
            &["container", "stop_details", "diagnostics"],
        )
        || !response
            .get("service_tier")
            .is_none_or(service_tier_is_valid)
    {
        return None;
    }
    let content = response.get("content").and_then(Json::as_array)?;
    if content.is_empty() || !response.get("usage").is_some_and(Json::is_object) {
        return None;
    }
    let compaction_count = content
        .iter()
        .filter(|block| block.get("type").and_then(Json::as_str) == Some("compaction"))
        .count();
    if compaction_count == 0 {
        return operation_allows_ordinary(context.operation)
            .then_some(AnthropicResponseKind::Ordinary)
            .filter(|_| response.get("stop_reason").and_then(Json::as_str) != Some("compaction"));
    }
    if content.len() != 1
        || compaction_count != 1
        || response.get("stop_reason").and_then(Json::as_str) != Some("compaction")
        || !response
            .pointer("/usage/iterations")
            .is_some_and(valid_compaction_iterations)
    {
        return None;
    }
    let block = &content[0];
    match context.protocol {
        AnthropicProtocol::ThresholdV1
            if operation_allows_threshold_compaction(context.operation)
                && threshold_response_block_is_valid(block) =>
        {
            Some(AnthropicResponseKind::ThresholdCompaction)
        }
        _ => None,
    }
}

fn operation_allows_ordinary(operation: AnthropicOperation) -> bool {
    matches!(
        operation,
        AnthropicOperation::PausedThreshold
            | AnthropicOperation::ThresholdContinuation
            | AnthropicOperation::ThresholdRecompact
    )
}

fn operation_allows_threshold_compaction(operation: AnthropicOperation) -> bool {
    matches!(
        operation,
        AnthropicOperation::PausedThreshold | AnthropicOperation::ThresholdRecompact
    )
}

fn valid_compaction_iterations(value: &Json) -> bool {
    value.as_array().is_some_and(|iterations| {
        !iterations.is_empty()
            && iterations
                .iter()
                .any(|iteration| iteration.get("type").and_then(Json::as_str) == Some("compaction"))
    })
}

fn context_management_response_is_valid(value: &Json) -> bool {
    value.is_null()
        || value.as_object().is_some_and(|object| {
            object.len() == 1 && object.get("applied_edits").is_some_and(Json::is_array)
        })
}

fn threshold_response_block_is_valid(block: &Json) -> bool {
    has_only_fields(block, &["type", "content", "encrypted_content"])
        && block.get("content").is_some_and(Json::is_string)
        && block
            .get("encrypted_content")
            .is_none_or(|value| value.is_null() || value.is_string())
}

fn has_only_fields(value: &Json, allowed: &[&str]) -> bool {
    value
        .as_object()
        .is_some_and(|object| object.keys().all(|key| allowed.contains(&key.as_str())))
}

fn nullable_response_metadata_is_valid(value: &Json, fields: &[&str]) -> bool {
    fields
        .iter()
        .all(|field| value.get(*field).is_none_or(Json::is_null))
}

fn service_tier_is_valid(value: &Json) -> bool {
    value.is_null() || matches!(value.as_str(), Some("standard" | "priority" | "batch"))
}

pub(crate) struct AnthropicStreamValidator {
    protocol: AnthropicProtocol,
    blocks: BTreeMap<u64, BlockState>,
    active_block: Option<u64>,
    next_block_index: u64,
    message_started: bool,
    message_delta_seen: bool,
    message_stopped: bool,
    valid: bool,
}

#[derive(Default)]
struct BlockState {
    kind: Option<String>,
    compaction_delta_count: u8,
    stopped: bool,
}

impl AnthropicStreamValidator {
    pub(crate) fn new(context: &AnthropicCacheContext) -> Self {
        Self {
            protocol: context.protocol,
            blocks: BTreeMap::new(),
            active_block: None,
            next_block_index: 0,
            message_started: false,
            message_delta_seen: false,
            message_stopped: false,
            valid: true,
        }
    }

    pub(crate) fn observe(&mut self, chunk: &Json) {
        if !self.valid {
            return;
        }
        let Some(kind) = chunk.get("type").and_then(Json::as_str) else {
            self.valid = false;
            return;
        };
        self.valid = match kind {
            "ping" => has_only_fields(chunk, &["type"]) && !self.message_stopped,
            "message_start" => self.observe_message_start(chunk),
            "message_delta" => self.observe_message_delta(chunk),
            "message_stop" => self.observe_message_stop(chunk),
            "content_block_start" => self.observe_block_start(chunk),
            "content_block_delta" => self.observe_block_delta(chunk),
            "content_block_stop" => self.observe_block_stop(chunk),
            _ => false,
        };
    }

    pub(crate) fn is_valid_for(&self, kind: AnthropicResponseKind) -> bool {
        if !self.valid
            || self.blocks.is_empty()
            || !self.message_started
            || !self.message_delta_seen
            || !self.message_stopped
            || self.active_block.is_some()
            || self.blocks.values().any(|block| !block.stopped)
        {
            return false;
        }
        let compaction: Vec<_> = self
            .blocks
            .values()
            .filter(|block| block.kind.as_deref() == Some("compaction"))
            .collect();
        match kind {
            AnthropicResponseKind::Ordinary => compaction.is_empty(),
            AnthropicResponseKind::ThresholdCompaction => {
                self.protocol == AnthropicProtocol::ThresholdV1
                    && self.blocks.len() == 1
                    && compaction.len() == 1
                    && compaction[0].compaction_delta_count == 1
            }
        }
    }

    fn observe_message_start(&mut self, chunk: &Json) -> bool {
        if self.message_started || !has_only_fields(chunk, &["type", "message"]) {
            return false;
        }
        let Some(message) = chunk.get("message") else {
            return false;
        };
        if !has_only_fields(
            message,
            &[
                "id",
                "type",
                "role",
                "model",
                "content",
                "stop_reason",
                "stop_sequence",
                "usage",
                "container",
                "stop_details",
                "diagnostics",
                "service_tier",
            ],
        ) || !message.get("id").is_some_and(Json::is_string)
            || message.get("type").and_then(Json::as_str) != Some("message")
            || message.get("role").and_then(Json::as_str) != Some("assistant")
            || !message.get("model").is_some_and(Json::is_string)
            || !message
                .get("content")
                .is_some_and(|content| content.as_array().is_some_and(Vec::is_empty))
            || !message.get("stop_reason").is_some_and(Json::is_null)
            || !message.get("stop_sequence").is_some_and(Json::is_null)
            || !message.get("usage").is_some_and(Json::is_object)
            || !nullable_response_metadata_is_valid(
                message,
                &["container", "stop_details", "diagnostics"],
            )
            || !message
                .get("service_tier")
                .is_none_or(service_tier_is_valid)
        {
            return false;
        }
        self.message_started = true;
        true
    }

    fn observe_block_start(&mut self, chunk: &Json) -> bool {
        if !self.message_started
            || self.message_delta_seen
            || self.active_block.is_some()
            || !has_only_fields(chunk, &["type", "index", "content_block"])
        {
            return false;
        }
        let Some(index) = chunk.get("index").and_then(Json::as_u64) else {
            return false;
        };
        if index != self.next_block_index {
            return false;
        }
        let Some(block) = chunk.get("content_block") else {
            return false;
        };
        let Some(kind) = block.get("type").and_then(Json::as_str) else {
            return false;
        };
        if kind == "compaction" {
            let valid = has_only_fields(block, &["type", "content"])
                && block.get("content").is_some_and(Json::is_null);
            if !valid {
                return false;
            }
        }
        let state = BlockState {
            kind: Some(kind.to_string()),
            ..BlockState::default()
        };
        if self.blocks.insert(index, state).is_some() {
            return false;
        }
        self.active_block = Some(index);
        self.next_block_index = self.next_block_index.saturating_add(1);
        true
    }

    fn observe_block_delta(&mut self, chunk: &Json) -> bool {
        if !has_only_fields(chunk, &["type", "index", "delta"]) {
            return false;
        }
        let Some(index) = chunk.get("index").and_then(Json::as_u64) else {
            return false;
        };
        if self.active_block != Some(index) {
            return false;
        }
        let Some(delta) = chunk.get("delta") else {
            return false;
        };
        let Some(block) = self.blocks.get_mut(&index) else {
            return false;
        };
        if block.kind.as_deref() != Some("compaction") {
            return ordinary_delta_is_replayable(block.kind.as_deref(), delta);
        }
        if self.protocol != AnthropicProtocol::ThresholdV1
            || !has_only_fields(delta, &["type", "content", "encrypted_content"])
            || delta.get("type").and_then(Json::as_str) != Some("compaction_delta")
            || !delta.get("content").is_some_and(Json::is_string)
            || !delta
                .get("encrypted_content")
                .is_none_or(|value| value.is_null() || value.is_string())
        {
            return false;
        }
        block.compaction_delta_count = block.compaction_delta_count.saturating_add(1);
        block.compaction_delta_count == 1
    }

    fn observe_block_stop(&mut self, chunk: &Json) -> bool {
        if !has_only_fields(chunk, &["type", "index"]) {
            return false;
        }
        let Some(index) = chunk.get("index").and_then(Json::as_u64) else {
            return false;
        };
        if self.active_block != Some(index) {
            return false;
        }
        let Some(block) = self.blocks.get_mut(&index) else {
            return false;
        };
        if block.stopped
            || (block.kind.as_deref() == Some("compaction")
                && self.protocol == AnthropicProtocol::ThresholdV1
                && block.compaction_delta_count != 1)
        {
            return false;
        }
        block.stopped = true;
        self.active_block = None;
        true
    }

    fn observe_message_delta(&mut self, chunk: &Json) -> bool {
        if !self.message_started
            || self.message_delta_seen
            || self.active_block.is_some()
            || !has_only_fields(chunk, &["type", "delta", "usage", "context_management"])
            || !chunk.get("delta").is_some_and(|delta| {
                has_only_fields(
                    delta,
                    &["stop_reason", "stop_sequence", "container", "stop_details"],
                ) && nullable_response_metadata_is_valid(delta, &["container", "stop_details"])
            })
            || !chunk.get("usage").is_some_and(Json::is_object)
            || !chunk
                .get("context_management")
                .is_none_or(context_management_response_is_valid)
        {
            return false;
        }
        self.message_delta_seen = true;
        true
    }

    fn observe_message_stop(&mut self, chunk: &Json) -> bool {
        if !self.message_started
            || !self.message_delta_seen
            || self.message_stopped
            || self.active_block.is_some()
            || !has_only_fields(chunk, &["type"])
        {
            return false;
        }
        self.message_stopped = true;
        true
    }
}

fn ordinary_delta_is_replayable(block_kind: Option<&str>, delta: &Json) -> bool {
    match (block_kind, delta.get("type").and_then(Json::as_str)) {
        (Some("text"), Some("text_delta")) => {
            has_only_fields(delta, &["type", "text"])
                && delta.get("text").is_some_and(Json::is_string)
        }
        (Some("text"), Some("citations_delta")) => {
            has_only_fields(delta, &["type", "citation"])
                && delta.get("citation").is_some_and(Json::is_object)
        }
        (Some("tool_use"), Some("input_json_delta")) => {
            has_only_fields(delta, &["type", "partial_json"])
                && delta.get("partial_json").is_some_and(Json::is_string)
        }
        _ => false,
    }
}

#[cfg(test)]
#[path = "../../tests/unit/response_cache/anthropic_tests.rs"]
mod tests;
