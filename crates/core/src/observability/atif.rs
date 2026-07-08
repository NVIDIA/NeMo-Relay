// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Agent Trajectory Interchange Format (ATIF) exporter.
//!
//! This module provides types and an exporter that collects lifecycle events
//! from the NeMo Relay runtime and converts them into ATIF trajectories conforming
//! to the ATIF v1.7 schema.
//!
//! # Overview
//!
//! The [`AtifExporter`] registers as an event subscriber, collects all events,
//! and can export them as an [`AtifTrajectory`] via [`AtifExporter::export`].
//!
//! # Event-to-Step Mapping
//!
//! The core conversion from NeMo Relay events to ATIF steps follows these rules:
//!
//! | NeMo Relay Event     | ATIF Step               | Notes                                |
//! |-----------------|-------------------------|--------------------------------------|
//! | LLM Start       | `user` step             | Messages extracted from LlmRequest   |
//! | LLM End         | `agent` step            | Response content, tool_calls promoted|
//! | Tool Start      | *(skipped)*             | tool_calls come from LLM End instead |
//! | Tool End        | agent observation         | Correlated by `source_call_id`       |
//! | Mark            | *(skipped)*             | Point-in-time telemetry is not a step|
//! | Scope Start/End | *(skipped)*             | Structural events, not trajectory    |
//!
//! The exporter serializes the full collected event stream into a single ATIF
//! trajectory.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::api::event::{Event, EventNormalizationExt};
use crate::api::runtime::EventSubscriberFn;
use crate::api::subscriber::flush_subscribers;
use crate::codec::request::{AnnotatedLlmRequest, Message, MessageContent};
use crate::codec::response::AnnotatedLlmResponse;
use crate::error::Result;
use crate::json::Json;

use super::{estimate_cost_for_response_or_model, manual, merge_usage, model_name_for_llm_event};

/// The ATIF schema version string embedded in all exported trajectories.
///
/// Currently `"ATIF-v1.7"`. This constant is used by [`AtifTrajectory`]
/// serialization and verified by downstream consumers to ensure compatibility.
pub const ATIF_SCHEMA_VERSION: &str = "ATIF-v1.7";

// ---------------------------------------------------------------------------
// ATIF types
// ---------------------------------------------------------------------------

/// Information about the agent that produced the trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtifAgentInfo {
    /// Human-readable agent name.
    pub name: String,
    /// Agent version string.
    pub version: String,
    /// Default LLM model name used by the agent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_name: Option<String>,
    /// Tool definitions available to the agent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_definitions: Option<Vec<Json>>,
    /// Extra metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra: Option<Json>,
}

/// A single step in an ATIF trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtifStep {
    /// 1-based ordinal step ID.
    pub step_id: usize,
    /// Source of the step: `"system"`, `"user"`, or `"agent"`.
    pub source: String,
    /// The message content (string or array of content parts).
    pub message: Json,
    /// ISO 8601 timestamp.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
    /// LLM model name, if applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_name: Option<String>,
    /// Qualitative or quantitative measure of reasoning effort.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<Json>,
    /// The agent's explicit internal reasoning.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// Tool calls made by the agent in this step.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<AtifToolCall>>,
    /// Observation (tool results) for this step.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observation: Option<AtifObservation>,
    /// Token usage and cost metrics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics: Option<AtifMetrics>,
    /// Number of LLM calls represented by this step.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm_call_count: Option<u64>,
    /// Whether this step was copied from a previous trajectory for context.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_copied_context: Option<bool>,
    /// Extra metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra: Option<Json>,
}

/// Token usage and cost metrics for a single step.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AtifMetrics {
    /// Number of prompt tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u64>,
    /// Number of completion tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u64>,
    /// Number of cached tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u64>,
    /// Cost in USD.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost_usd: Option<f64>,
    /// Token IDs for prompt (input) tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_token_ids: Option<Vec<u64>>,
    /// Token IDs for completion (response) tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_token_ids: Option<Vec<u64>>,
    /// Log probability assigned to each generated token.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Vec<f64>>,
    /// Other metrics (e.g. reasoning_tokens, cache_creation_input_tokens).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra: Option<Json>,
}

/// Aggregate statistics for the entire trajectory (ATIF final_metrics).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AtifFinalMetrics {
    /// Sum of all prompt tokens across all steps, including cached tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_prompt_tokens: Option<u64>,
    /// Sum of all completion tokens across all steps.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_completion_tokens: Option<u64>,
    /// Sum of all cached tokens across all steps.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_cached_tokens: Option<u64>,
    /// Total real monetary cost for the entire trajectory.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_cost_usd: Option<f64>,
    /// Total number of steps. If not equivalent to steps.len(), document in notes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_steps: Option<u64>,
    /// Custom aggregate metrics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra: Option<Json>,
}

/// A tool call made by the agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtifToolCall {
    /// Correlation ID linking this call to its observation result.
    pub tool_call_id: String,
    /// Name of the tool/function called.
    pub function_name: String,
    /// Arguments passed to the tool.
    pub arguments: Json,
    /// Provider or host-specific metadata for this tool call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra: Option<Json>,
}

/// Observation results from tool execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtifObservation {
    /// List of observation results (one per tool call).
    pub results: Vec<AtifObservationResult>,
}

/// A single observation result from a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtifObservationResult {
    /// Correlation ID linking to the originating tool call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_call_id: Option<String>,
    /// The tool's output content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Json>,
    /// References to delegated subagent trajectories.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subagent_trajectory_ref: Option<Vec<AtifSubagentTrajectoryRef>>,
    /// Provider or host-specific metadata for this observation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra: Option<Json>,
}

/// Reference to a delegated subagent trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtifSubagentTrajectoryRef {
    /// Embedded trajectory identifier, resolved against `subagent_trajectories`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trajectory_id: Option<String>,
    /// Run identity for debug/search/display correlation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Extra metadata about the subagent execution.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra: Option<Json>,
}

/// Lineage node identifying a callable within an ATIF step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtifAncestry {
    /// Unique identifier for the callable node (scope UUID).
    pub function_id: String,
    /// Human-readable name of the callable node.
    pub function_name: String,
    /// Optional parent callable identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    /// Optional parent callable name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_name: Option<String>,
}

/// Invocation timing and correlation metadata for one execution occurrence.
///
/// `start_timestamp` and `end_timestamp` are always emitted together or not
/// at all.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtifInvocationInfo {
    /// Invocation start timestamp in Unix epoch seconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_timestamp: Option<f64>,
    /// Invocation end timestamp in Unix epoch seconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_timestamp: Option<f64>,
    /// Stable invocation identifier for correlation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub invocation_id: Option<String>,
    /// Terminal status of the invocation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    /// Runtime or framework label.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub framework: Option<String>,
}

/// Lineage payload serialized into ATIF `Step.extra`.
///
/// `tool_ancestry[i]` and `tool_invocations[i]` align by index with
/// `Step.tool_calls[i]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtifStepExtra {
    /// Step-level callable lineage.
    pub ancestry: AtifAncestry,
    /// Step-level invocation timing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub invocation: Option<AtifInvocationInfo>,
    /// Full unwrapped LLM request payload for request-level fidelity.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm_request: Option<Json>,
    /// Full raw LLM response payload for response-level fidelity.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm_response: Option<Json>,
    /// Legacy event payload field retained for source compatibility.
    ///
    /// The ATIF exporter does not translate point-in-time mark events into
    /// trajectory steps, so exporter-produced steps leave this field unset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event_payload: Option<Json>,
    /// Per-tool callable lineage, aligned with `tool_calls`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_ancestry: Vec<AtifAncestry>,
    /// Per-tool invocation timing, aligned with `tool_calls`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_invocations: Option<Vec<AtifInvocationInfo>>,
}

/// A complete ATIF trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtifTrajectory {
    /// Schema version (e.g., `"ATIF-v1.7"`).
    pub schema_version: String,
    /// Unique session identifier.
    pub session_id: String,
    /// Canonical per-trajectory-document identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trajectory_id: Option<String>,
    /// Information about the agent.
    pub agent: AtifAgentInfo,
    /// Ordered list of trajectory steps.
    pub steps: Vec<AtifStep>,
    /// Custom information, design notes, or explanations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
    /// Aggregate metrics for the entire trajectory.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_metrics: Option<AtifFinalMetrics>,
    /// Reference to the continuation trajectory file if continued elsewhere.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub continued_trajectory_ref: Option<String>,
    /// Embedded subagent trajectories.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subagent_trajectories: Option<Vec<AtifTrajectory>>,
    /// Extra metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra: Option<Json>,
}

// ---------------------------------------------------------------------------
// AtifExporter
// ---------------------------------------------------------------------------

struct AtifExporterState {
    session_id: String,
    agent_info: AtifAgentInfo,
    events: Vec<Event>,
}

/// Collects lifecycle events and exports them as ATIF trajectories.
///
/// Register this exporter as an event subscriber via [`AtifExporter::subscriber`],
/// then call [`AtifExporter::export`] to produce an [`AtifTrajectory`].
#[derive(Clone)]
pub struct AtifExporter {
    state: Arc<Mutex<AtifExporterState>>,
}

impl AtifExporter {
    /// Create a new exporter with the given session metadata.
    ///
    /// # Parameters
    /// - `session_id`: Stable identifier for the trajectory being collected.
    /// - `agent_info`: Metadata describing the emitting agent.
    ///
    /// # Returns
    /// A new [`AtifExporter`] with an empty in-memory event buffer.
    pub fn new(session_id: String, agent_info: AtifAgentInfo) -> Self {
        Self {
            state: Arc::new(Mutex::new(AtifExporterState {
                session_id,
                agent_info,
                events: Vec::new(),
            })),
        }
    }

    /// Return an event subscriber function that records NeMo Relay events.
    ///
    /// The returned callback can be registered with
    /// [`register_subscriber`](crate::api::subscriber::register_subscriber).
    ///
    /// # Returns
    /// An [`EventSubscriberFn`] that appends compatible lifecycle events to
    /// this exporter's internal buffer. Point-in-time marks are ignored.
    pub fn subscriber(&self) -> EventSubscriberFn {
        let state = self.state.clone();
        Arc::new(move |event: &Event| {
            if matches!(event, Event::Mark(_)) {
                return;
            }
            if let Ok(mut s) = state.lock() {
                s.events.push(event.clone());
            }
        })
    }

    /// Export the collected event history as an [`AtifTrajectory`].
    ///
    /// # Returns
    /// An [`AtifTrajectory`] synthesized from the events observed so far.
    ///
    /// # Errors
    /// Returns an error if queued subscriber delivery cannot be flushed before
    /// the trajectory is cloned.
    ///
    /// # Notes
    /// Exporting does not clear the buffered events. Call [`AtifExporter::clear`]
    /// when you need to reset the exporter between trajectories.
    pub fn export(&self) -> Result<AtifTrajectory> {
        self.try_export()
    }

    /// Try to export the collected event history as an [`AtifTrajectory`].
    ///
    /// This is equivalent to [`AtifExporter::export`] and is retained for
    /// callers that prefer an explicitly fallible method name.
    pub fn try_export(&self) -> Result<AtifTrajectory> {
        flush_subscribers()?;
        let (session_id, agent_info, events) = {
            let state = self.state.lock().unwrap();
            (
                state.session_id.clone(),
                state.agent_info.clone(),
                state.events.clone(),
            )
        };
        let collected_events: Vec<&Event> = events.iter().collect();
        Ok(events_to_trajectory(
            &session_id,
            agent_info,
            &collected_events,
        ))
    }

    /// Clear all collected events from the internal buffer.
    ///
    /// # Returns
    /// `()`.
    pub fn clear(&self) {
        let mut state = self.state.lock().unwrap();
        state.events.clear();
    }
}

// ---------------------------------------------------------------------------
// Safe JSON extraction helpers
// ---------------------------------------------------------------------------

/// If `input` looks like an `LlmRequest` envelope (`{"content": ..., "headers": ...}`),
/// return the inner `content` value. Otherwise return the input unchanged.
///
/// This avoids leaking the NeMo Relay transport wrapper into the trajectory.
fn unwrap_llm_request(input: &Json) -> Json {
    if let Some(obj) = input.as_object()
        && obj.contains_key("content")
        && obj.contains_key("headers")
    {
        return obj.get("content").cloned().unwrap_or_else(|| input.clone());
    }
    input.clone()
}

/// Extract the user-facing message content from a raw LLM response.
///
/// Looks for provider response content fields that can be represented as an
/// ATIF agent message.
/// Tool-call-only responses use an empty string message and keep the full
/// response under `Step.extra.llm_response`.
fn extract_llm_response_message(output: &Json) -> Json {
    let Some(obj) = output.as_object() else {
        return atif_content_value(output);
    };

    if let Some(message) = extract_object_llm_response_message(output, obj) {
        return message;
    }

    atif_content_value(output)
}

fn extract_object_llm_response_message(
    output: &Json,
    obj: &serde_json::Map<String, Json>,
) -> Option<Json> {
    if let Some(content) = non_null_object_field(obj, "content") {
        return Some(extract_content_message(output, &content));
    }
    if let Some(content) = assistant_message_content(obj) {
        return Some(atif_content_value(&content));
    }
    if let Some(content) = raw_response_message_field(output, "content")
        && !content.is_null()
    {
        return Some(atif_content_value(content));
    }
    if let Some(answer) = non_null_object_field(obj, "answer") {
        return Some(atif_content_value(&answer));
    }
    if let Some(content) = openai_responses_output_message(output) {
        return Some(content);
    }
    tool_call_array(output).map(|_| empty_message())
}

fn extract_content_message(output: &Json, content: &Json) -> Json {
    anthropic_messages_content_message(output, content)
        .unwrap_or_else(|| atif_content_value(content))
}

fn assistant_message_content(obj: &serde_json::Map<String, Json>) -> Option<Json> {
    obj.get("assistant_message")
        .and_then(Json::as_object)
        .and_then(|assistant| non_null_object_field(assistant, "content"))
}

fn non_null_object_field(obj: &serde_json::Map<String, Json>, key: &str) -> Option<Json> {
    obj.get(key).filter(|value| !value.is_null()).cloned()
}

fn empty_message() -> Json {
    Json::String(String::new())
}

fn atif_content_value(value: &Json) -> Json {
    match value {
        Json::String(_) => value.clone(),
        Json::Array(_) if is_atif_content_parts(value) => value.clone(),
        Json::Null => empty_message(),
        _ => Json::String(json_to_string(value)),
    }
}

fn anthropic_messages_content_message(output: &Json, content: &Json) -> Option<Json> {
    let object = output.as_object()?;
    if object.get("type").and_then(Json::as_str) != Some("message") {
        return None;
    }
    let blocks = content.as_array()?;
    let mut text_parts = Vec::new();
    let mut has_tool_use = false;
    for block in blocks {
        let Some(block_object) = block.as_object() else {
            continue;
        };
        match block_object.get("type").and_then(Json::as_str) {
            Some("text") => {
                if let Some(text) = block_object.get("text").and_then(Json::as_str)
                    && !text.trim().is_empty()
                {
                    text_parts.push(text.to_string());
                }
            }
            Some("tool_use") => has_tool_use = true,
            _ => {}
        }
    }
    match text_parts.as_slice() {
        [] if has_tool_use => Some(empty_message()),
        [] => None,
        [text] => Some(Json::String(text.clone())),
        _ => Some(Json::String(text_parts.join("\n"))),
    }
}

fn observation_content_value(value: &Json) -> Option<Json> {
    match value {
        Json::String(_) => Some(value.clone()),
        Json::Array(_) if is_atif_content_parts(value) => Some(value.clone()),
        _ => None,
    }
}

fn observation_extra(event: &Event, output: &Json) -> Json {
    let mut extra = event_extra(event);
    if let Some(tool_result) = observation_tool_result_extra(output)
        && let Json::Object(extra_object) = &mut extra
    {
        extra_object.insert("tool_result".to_string(), tool_result);
    }
    extra
}

fn observation_tool_result_extra(value: &Json) -> Option<Json> {
    match value {
        Json::Null | Json::String(_) => None,
        Json::Array(_) if is_atif_content_parts(value) => None,
        _ => Some(value.clone()),
    }
}

fn is_atif_content_parts(value: &Json) -> bool {
    let Some(parts) = value.as_array() else {
        return false;
    };
    parts.iter().all(|part| {
        let Some(object) = part.as_object() else {
            return false;
        };
        match object.get("type").and_then(Json::as_str) {
            Some("text") => object.get("text").and_then(Json::as_str).is_some(),
            Some("image") => is_atif_image_source(object.get("source")),
            _ => false,
        }
    })
}

fn is_atif_image_source(value: Option<&Json>) -> bool {
    let Some(source) = value.and_then(Json::as_object) else {
        return false;
    };
    matches!(
        source.get("media_type").and_then(Json::as_str),
        Some("image/jpeg" | "image/png" | "image/gif" | "image/webp")
    ) && source.get("path").and_then(Json::as_str).is_some()
}

fn json_to_string(value: &Json) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| value.to_string())
}

fn raw_response_message_field<'a>(output: &'a Json, field: &str) -> Option<&'a Json> {
    let object = output.as_object()?;
    object
        .get("raw_response")
        .or(Some(output))
        .and_then(|raw_response| raw_response.as_object())
        .and_then(|raw_response| raw_response.get("choices"))
        .and_then(Json::as_array)
        .and_then(|choices| choices.first())
        .and_then(Json::as_object)
        .and_then(|choice| choice.get("message"))
        .and_then(Json::as_object)
        .and_then(|message| message.get(field))
}

fn openai_responses_output_message(output: &Json) -> Option<Json> {
    let object = output.as_object()?;
    if let Some(output_text) = object.get("output_text").and_then(Json::as_str) {
        return Some(Json::String(output_text.to_string()));
    }

    let output_items = object.get("output").and_then(Json::as_array)?;
    let mut text_parts = Vec::new();
    for item in output_items {
        collect_openai_responses_output_text(item, &mut text_parts);
    }

    match text_parts.as_slice() {
        [] => None,
        [text] => Some(Json::String(text.clone())),
        _ => Some(Json::String(text_parts.join("\n"))),
    }
}

fn collect_openai_responses_output_text(item: &Json, text_parts: &mut Vec<String>) {
    let Some(item_obj) = item.as_object() else {
        return;
    };
    match item_obj.get("type").and_then(Json::as_str) {
        Some("message") => {
            if let Some(content) = item_obj.get("content").and_then(Json::as_array) {
                collect_openai_responses_content_text(content, "output_text", text_parts);
            }
        }
        Some("output_text") => {
            if let Some(text) = item_obj.get("text").and_then(Json::as_str) {
                text_parts.push(text.to_string());
            }
        }
        _ => {}
    }
}

fn collect_openai_responses_content_text(
    content: &[Json],
    block_type: &str,
    text_parts: &mut Vec<String>,
) {
    for block in content {
        let Some(block_obj) = block.as_object() else {
            continue;
        };
        if block_obj.get("type").and_then(Json::as_str) == Some(block_type)
            && let Some(text) = block_obj.get("text").and_then(Json::as_str)
        {
            text_parts.push(text.to_string());
        }
    }
}

/// Known keys in token_usage that we extract to dedicated fields.
const TOKEN_USAGE_KNOWN_KEYS: &[&str] = &[
    "prompt_tokens",
    "input_tokens",
    "inputTokens",
    "input",
    "completion_tokens",
    "output_tokens",
    "completionTokens",
    "outputTokens",
    "output",
    "cached_tokens",
    "cachedTokens",
    "cache_read_tokens",
    "cacheReadTokens",
    "cache_read_input_tokens",
    "cacheReadInputTokens",
    "cacheRead",
    "cache_creation_input_tokens",
    "cacheCreationInputTokens",
    "cache_write_tokens",
    "cacheWriteTokens",
    "cacheWrite",
    "cost_usd",
    "cost",
    "prompt_tokens_details",
    "input_tokens_details",
    "prompt_token_ids",
    "completion_token_ids",
    "logprobs",
];

/// Try to extract `AtifMetrics` from a `token_usage` object in the LLM response.
///
/// Supports NeMo Relay `token_usage` and provider-native `usage` payloads.
/// Populates `extra` with any unknown usage keys (e.g. reasoning_tokens or total_tokens).
/// Returns `None` if the response has no recognized token or cost metrics.
fn extract_metrics(
    output: &Json,
    provider: Option<&str>,
    model_name: Option<&str>,
    normalized_response: Option<&AnnotatedLlmResponse>,
) -> Option<AtifMetrics> {
    let raw_usage = token_usage_object(output);
    let fallback_usage = manual::usage_from_manual_llm_output(Some(output));
    let normalized_usage = normalized_response.and_then(|response| response.usage.as_ref());
    let merged_usage = merge_usage(normalized_usage, fallback_usage.as_ref());
    let prompt = merged_usage.as_ref().and_then(|usage| usage.prompt_tokens);
    let completion = merged_usage
        .as_ref()
        .and_then(|usage| usage.completion_tokens);
    let cache_read = merged_usage
        .as_ref()
        .and_then(|usage| usage.cache_read_tokens);
    let cache_write = merged_usage
        .as_ref()
        .and_then(|usage| usage.cache_write_tokens);
    let cached = sum_options(cache_read, cache_write);
    let normalized_cost_source = normalized_usage.and_then(|usage| usage.cost.as_ref());
    let normalized_cost =
        normalized_cost_source.and_then(|cost| cost.total_or_component_sum_for_currency("USD"));
    let manual_cost =
        manual::cost_from_manual_llm_output(Some(output), manual::ManualCostPolicy::AtifUsdOnly)
            .map(|(total, _)| total);
    let explicit_cost = if normalized_cost_source.is_some() {
        normalized_cost
    } else {
        manual_cost
    };
    let has_reported_cost = normalized_cost_source.is_some()
        || raw_usage.is_some_and(|usage| usage.get("cost").is_some());
    let cost = if has_reported_cost {
        explicit_cost
    } else {
        explicit_cost.or_else(|| {
            let usage = merged_usage.as_ref()?;
            estimate_cost_for_response_or_model(
                provider,
                model_name,
                normalized_response
                    .and_then(|response| response.model.as_deref())
                    .or_else(|| response_model_name(output)),
                usage,
            )
            .and_then(|cost| cost.total_for_currency("USD"))
        })
    };
    let prompt_ids = raw_usage
        .and_then(|usage| usage.get("prompt_token_ids"))
        .and_then(Json::as_array)
        .map(|a| a.iter().filter_map(Json::as_u64).collect());
    let completion_ids = raw_usage
        .and_then(|usage| usage.get("completion_token_ids"))
        .and_then(Json::as_array)
        .map(|a| a.iter().filter_map(Json::as_u64).collect());
    let logprobs = raw_usage
        .and_then(|usage| usage.get("logprobs"))
        .and_then(Json::as_array)
        .map(|a| a.iter().filter_map(Json::as_f64).collect());
    let known: std::collections::HashSet<&str> = TOKEN_USAGE_KNOWN_KEYS.iter().copied().collect();
    let extra_map: serde_json::Map<String, Json> = output
        .as_object()
        .into_iter()
        .flat_map(|output| {
            ["usage", "token_usage"]
                .into_iter()
                .filter_map(|key| output.get(key).and_then(Json::as_object))
        })
        .flat_map(|usage| usage.iter())
        .filter(|(k, _)| !known.contains(k.as_str()))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let extra = if extra_map.is_empty() {
        None
    } else {
        Some(Json::Object(extra_map))
    };
    if prompt.is_none() && completion.is_none() && cached.is_none() && cost.is_none() {
        return None;
    }
    Some(AtifMetrics {
        prompt_tokens: prompt,
        completion_tokens: completion,
        cached_tokens: cached,
        cost_usd: cost,
        prompt_token_ids: prompt_ids,
        completion_token_ids: completion_ids,
        logprobs,
        extra,
    })
}

fn merge_metrics(
    primary: Option<AtifMetrics>,
    supplemental: Option<&AtifMetrics>,
) -> Option<AtifMetrics> {
    match (primary, supplemental) {
        (None, None) => None,
        (Some(metrics), None) => Some(metrics),
        (None, Some(supplemental)) => Some(supplemental.clone()),
        (Some(mut metrics), Some(supplemental)) => {
            merge_metrics_fields(&mut metrics, supplemental);
            Some(metrics)
        }
    }
}

fn merge_metrics_fields(target: &mut AtifMetrics, supplemental: &AtifMetrics) {
    if target.prompt_tokens.is_none() {
        target.prompt_tokens = supplemental.prompt_tokens;
    }
    if target.completion_tokens.is_none() {
        target.completion_tokens = supplemental.completion_tokens;
    }
    if target.cached_tokens.is_none() {
        target.cached_tokens = supplemental.cached_tokens;
    }
    if target.cost_usd.is_none() {
        target.cost_usd = supplemental.cost_usd;
    }
    if target.prompt_token_ids.is_none() {
        target.prompt_token_ids = supplemental.prompt_token_ids.clone();
    }
    if target.completion_token_ids.is_none() {
        target.completion_token_ids = supplemental.completion_token_ids.clone();
    }
    if target.logprobs.is_none() {
        target.logprobs = supplemental.logprobs.clone();
    }
    merge_metrics_extra(&mut target.extra, &supplemental.extra);
}

fn merge_metrics_extra(target: &mut Option<Json>, supplemental: &Option<Json>) {
    let Some(supplemental) = supplemental else {
        return;
    };
    match (target.as_mut(), supplemental) {
        (Some(Json::Object(target_object)), Json::Object(supplemental_object)) => {
            for (key, value) in supplemental_object {
                target_object
                    .entry(key.clone())
                    .or_insert_with(|| value.clone());
            }
        }
        (None, _) => *target = Some(supplemental.clone()),
        _ => {}
    }
}

fn token_usage_object(output: &Json) -> Option<&serde_json::Map<String, Json>> {
    let output = output.as_object()?;
    output
        .get("token_usage")
        .or_else(|| output.get("usage"))
        .and_then(Json::as_object)
}

fn response_model_name(output: &Json) -> Option<&str> {
    output
        .as_object()
        .and_then(|object| object.get("model").and_then(Json::as_str))
}

fn sum_options(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left + right),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

/// Extract `reasoning_effort` from an LLM request (string or number).
///
/// The request content may have `reasoning_effort` (e.g. `"high"`, `"medium"`,
/// or a numeric value). Returns the value as Json for flexibility.
fn extract_reasoning_effort(input: &Json) -> Option<Json> {
    if let Some(obj) = input.as_object()
        && let Some(v) = obj.get("reasoning_effort")
        && !v.is_null()
    {
        return Some(v.clone());
    }
    None
}

/// Extract `reasoning` (reasoning_content) from an LLM response output.
///
/// The agent's explicit internal reasoning may appear in the response under the
/// `"reasoning"` key. Returns `None` if absent or not a string.
fn extract_reasoning_content(output: &Json) -> Option<String> {
    if let Some(obj) = output.as_object()
        && let Some(r) = obj.get("reasoning")
    {
        return r.as_str().map(String::from);
    }
    None
}

/// Extract the latest user-facing message from an LLM request payload.
///
/// LLM start inputs typically contain `{ "messages": [...], "model": "...",
/// "max_tokens": ..., "tools": [...], "stream": ... }`. For the ATIF user step
/// we emit a schema-compatible message value (string or content-part array)
/// and preserve the full LLM request in `Step.extra.llm_request`.
///
/// Returns the latest user message content if present, a prompt if present, or
/// a stringified representation of the input as a last resort.
fn extract_user_messages(input: &Json) -> Json {
    if let Some(obj) = input.as_object()
        && let Some(messages) = obj.get("messages").and_then(Json::as_array)
        && let Some(message) = messages
            .iter()
            .rev()
            .filter_map(Json::as_object)
            .find(|message| match message.get("role").and_then(Json::as_str) {
                Some(role) => role == "user",
                None => true,
            })
            .and_then(|message| message.get("content"))
    {
        return atif_content_value(message);
    }
    if let Some(obj) = input.as_object()
        && let Some(message) = obj.get("input").and_then(openai_responses_input_message)
    {
        return message;
    }
    if let Some(obj) = input.as_object()
        && let Some(prompt) = obj.get("prompt")
    {
        return atif_content_value(prompt);
    }
    atif_content_value(input)
}

fn openai_responses_input_message(input: &Json) -> Option<Json> {
    if input.is_string() {
        return Some(atif_content_value(input));
    }

    let items = input.as_array()?;
    items
        .iter()
        .rev()
        .find_map(openai_responses_input_item_message)
}

fn openai_responses_input_item_message(item: &Json) -> Option<Json> {
    let item_obj = item.as_object()?;
    if item_obj.get("role").and_then(Json::as_str) != Some("user") {
        return None;
    }
    let content = item_obj.get("content")?;
    openai_responses_input_content_message(content)
}

fn openai_responses_input_content_message(content: &Json) -> Option<Json> {
    if content.is_string() {
        return Some(atif_content_value(content));
    }

    if let Some(content_parts) = content.as_array() {
        let mut text_parts = Vec::new();
        collect_openai_responses_content_text(content_parts, "input_text", &mut text_parts);
        if text_parts.is_empty() {
            collect_openai_responses_content_text(content_parts, "text", &mut text_parts);
        }
        return match text_parts.as_slice() {
            [] => is_atif_content_parts(content).then(|| content.clone()),
            [text] => Some(Json::String(text.clone())),
            _ => Some(Json::String(text_parts.join("\n"))),
        };
    }

    None
}

/// Try to promote `tool_calls` from the raw LLM response into `AtifToolCall` entries.
///
/// Expected shape per OpenAI convention:
/// ```json
/// "tool_calls": [{ "id": "...", "type": "function", "function": { "name": "...", "arguments": "..." } }]
/// ```
///
/// String `arguments` are parsed into JSON for consistency with NeMo Relay tool events
/// which always provide parsed arguments.
///
/// Returns `None` if there are no tool calls or the structure is unrecognized.
fn extract_tool_calls(output: &Json) -> Option<Vec<AtifToolCall>> {
    let arr = tool_call_array(output)
        .filter(|arr| !arr.is_empty())
        .map(|arr| arr.iter().collect::<Vec<_>>())
        .or_else(|| openai_responses_function_call_items(output))
        .or_else(|| anthropic_messages_tool_use_items(output))?;
    let mut calls = Vec::with_capacity(arr.len());
    for (index, tc) in arr.iter().enumerate() {
        let tc_obj = tc.as_object()?;
        let mut id = tc_obj
            .get("id")
            .or_else(|| tc_obj.get("tool_call_id"))
            .or_else(|| tc_obj.get("call_id"))
            .and_then(Json::as_str)
            .unwrap_or("")
            .to_string();
        let func = tc_obj.get("function").and_then(Json::as_object);
        let name = func
            .and_then(|f| f.get("name"))
            .or_else(|| tc_obj.get("name"))
            .or_else(|| tc_obj.get("toolName"))
            .or_else(|| tc_obj.get("tool_name"))
            .or_else(|| tc_obj.get("function_name"))
            .and_then(Json::as_str)
            .unwrap_or("")
            .to_string();
        if id.is_empty() && !name.is_empty() {
            id = format!("{name}:{}", index + 1);
        }
        let raw_arguments = func
            .and_then(|f| f.get("arguments"))
            .or_else(|| tc_obj.get("arguments"))
            .or_else(|| tc_obj.get("args"))
            .or_else(|| tc_obj.get("input"));
        let arguments = normalize_tool_arguments(raw_arguments);
        // Skip entries with no id and no name — they are not meaningful.
        if id.is_empty() && name.is_empty() {
            continue;
        }
        calls.push(AtifToolCall {
            tool_call_id: id,
            function_name: name,
            arguments,
            extra: tool_call_extra(tc),
        });
    }
    if calls.is_empty() { None } else { Some(calls) }
}

// Annotation adapters: read the normalized message from an annotation, returning
// None for multimodal content so the caller falls back to the raw extractor.

fn atif_message_from_annotated_request(request: &AnnotatedLlmRequest) -> Option<Json> {
    let content = request
        .messages
        .iter()
        .rev()
        .find_map(|message| match message {
            Message::User { content, .. } => Some(content),
            _ => None,
        })?;
    match content {
        MessageContent::Text(text) => Some(Json::String(text.clone())),
        MessageContent::Parts(_) => None,
    }
}

fn atif_message_from_annotated_response(response: &AnnotatedLlmResponse) -> Option<Json> {
    match &response.message {
        Some(MessageContent::Text(text)) => Some(Json::String(text.clone())),
        // Multimodal: defer to the raw extractor (returns None to fall back).
        Some(MessageContent::Parts(_)) => None,
        // No assistant text (tool-call-only, or reasoning/thinking-only): emit an
        // empty message rather than the raw extractor's stringified payload. Tool
        // calls and metrics are still recovered from the raw output downstream.
        None => Some(empty_message()),
    }
}

fn tool_call_array(output: &Json) -> Option<&Vec<Json>> {
    output
        .as_object()
        .and_then(|object| object.get("tool_calls"))
        .and_then(Json::as_array)
        .or_else(|| {
            output
                .as_object()
                .and_then(|object| object.get("assistant_message"))
                .and_then(Json::as_object)
                .and_then(|assistant| assistant.get("tool_calls"))
                .and_then(Json::as_array)
        })
        .or_else(|| raw_response_message_field(output, "tool_calls").and_then(Json::as_array))
}

fn openai_responses_function_call_items(output: &Json) -> Option<Vec<&Json>> {
    let items = output
        .as_object()
        .and_then(|object| object.get("output"))
        .and_then(Json::as_array)?;
    let function_call_items = items
        .iter()
        .filter(|item| item.get("type").and_then(Json::as_str) == Some("function_call"))
        .collect::<Vec<_>>();
    (!function_call_items.is_empty()).then_some(function_call_items)
}

fn anthropic_messages_tool_use_items(output: &Json) -> Option<Vec<&Json>> {
    let object = output.as_object()?;
    if object.get("type").and_then(Json::as_str) != Some("message") {
        return None;
    }
    let content_blocks = object.get("content").and_then(Json::as_array)?;
    let tool_use_items = content_blocks
        .iter()
        .filter(|item| item.get("type").and_then(Json::as_str) == Some("tool_use"))
        .collect::<Vec<_>>();
    (!tool_use_items.is_empty()).then_some(tool_use_items)
}

fn normalize_tool_arguments(raw_arguments: Option<&Json>) -> Json {
    let Some(raw_arguments) = raw_arguments else {
        return serde_json::json!({});
    };
    match raw_arguments {
        Json::Object(_) => raw_arguments.clone(),
        Json::String(arguments) => match serde_json::from_str::<Json>(arguments) {
            Ok(Json::Object(object)) => Json::Object(object),
            Ok(value) => serde_json::json!({ "value": value }),
            Err(_) => serde_json::json!({ "raw": arguments }),
        },
        Json::Null => serde_json::json!({}),
        value => serde_json::json!({ "value": value }),
    }
}

fn tool_call_extra(tool_call: &Json) -> Option<Json> {
    let object = tool_call.as_object()?;
    let mut extra = serde_json::Map::new();

    for (key, value) in object {
        if !matches!(
            key.as_str(),
            "id" | "tool_call_id"
                | "call_id"
                | "type"
                | "function"
                | "name"
                | "tool_name"
                | "function_name"
                | "arguments"
                | "args"
                | "input"
        ) {
            extra.insert(key.clone(), value.clone());
        }
    }

    if let Some(function) = object.get("function").and_then(Json::as_object) {
        let mut function_extra = serde_json::Map::new();
        for (key, value) in function {
            if key != "name" && key != "arguments" {
                function_extra.insert(key.clone(), value.clone());
            }
        }
        if !function_extra.is_empty() {
            extra.insert("function".to_string(), Json::Object(function_extra));
        }
    }

    (!extra.is_empty()).then_some(Json::Object(extra))
}

fn event_extra(event: &Event) -> Json {
    let mut extra = serde_json::Map::new();
    extra.insert(
        "event_uuid".to_string(),
        Json::String(event.uuid().to_string()),
    );
    extra.insert(
        "event_name".to_string(),
        Json::String(event.name().to_string()),
    );
    if let Some(parent_uuid) = event.parent_uuid() {
        extra.insert(
            "parent_event_uuid".to_string(),
            Json::String(parent_uuid.to_string()),
        );
    }
    if let Some(metadata) = event.metadata()
        && !metadata.is_null()
    {
        extra.insert("metadata".to_string(), metadata.clone());
    }
    Json::Object(extra)
}

/// Compute aggregate `final_metrics` by summing metrics across all steps.
///
/// Always returns `Some(AtifFinalMetrics)` with `total_steps` set. Each token
/// or cost total is populated only when at least one step provides that field.
fn compute_final_metrics(steps: &[AtifStep]) -> Option<AtifFinalMetrics> {
    let mut totals = FinalMetricsTotals::default();
    for metrics in steps.iter().filter_map(|step| step.metrics.as_ref()) {
        totals.add(metrics);
    }
    Some(totals.into_final_metrics(steps.len()))
}

#[derive(Default)]
struct FinalMetricsTotals {
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    cached_tokens: Option<u64>,
    cost_usd: Option<f64>,
}

impl FinalMetricsTotals {
    fn add(&mut self, metrics: &AtifMetrics) {
        add_u64_total(&mut self.prompt_tokens, metrics.prompt_tokens);
        add_u64_total(&mut self.completion_tokens, metrics.completion_tokens);
        add_u64_total(&mut self.cached_tokens, metrics.cached_tokens);
        add_f64_total(&mut self.cost_usd, metrics.cost_usd);
    }

    fn into_final_metrics(self, step_count: usize) -> AtifFinalMetrics {
        AtifFinalMetrics {
            total_prompt_tokens: self.prompt_tokens,
            total_completion_tokens: self.completion_tokens,
            total_cached_tokens: self.cached_tokens,
            total_cost_usd: self.cost_usd,
            total_steps: Some(step_count as u64),
            extra: None,
        }
    }
}

fn add_u64_total(total: &mut Option<u64>, value: Option<u64>) {
    if let Some(value) = value {
        *total = Some(total.unwrap_or(0) + value);
    }
}

fn add_f64_total(total: &mut Option<f64>, value: Option<f64>) {
    if let Some(value) = value {
        *total = Some(total.unwrap_or(0.0) + value);
    }
}

// ---------------------------------------------------------------------------
// AtifStepExtra helpers
// ---------------------------------------------------------------------------

/// Build an [`AtifAncestry`] from a NeMo Relay [`Event`].
///
/// `name_map` is a pre-pass uuid → name lookup used to resolve `parent_name`.
fn build_ancestry(
    event: &Event,
    name_map: &std::collections::HashMap<Uuid, String>,
) -> AtifAncestry {
    AtifAncestry {
        function_id: event.uuid().to_string(),
        function_name: event.name().to_string(),
        parent_id: event.parent_uuid().map(|u| u.to_string()),
        parent_name: event.parent_uuid().and_then(|u| name_map.get(&u)).cloned(),
    }
}

/// Build an [`AtifInvocationInfo`] from start/end timestamps.
///
/// If `start_ts` is `None`, both timestamps are omitted to preserve the
/// requirement that they are always emitted together or not at all.
fn build_invocation_info(
    start_ts: Option<DateTime<Utc>>,
    end_ts: DateTime<Utc>,
    invocation_id: Option<String>,
    framework: &str,
) -> AtifInvocationInfo {
    AtifInvocationInfo {
        start_timestamp: start_ts.map(|s| s.timestamp_millis() as f64 / 1000.0),
        end_timestamp: start_ts.map(|_| end_ts.timestamp_millis() as f64 / 1000.0),
        invocation_id,
        status: Some("completed".to_string()),
        framework: Some(framework.to_string()),
    }
}

fn delegation_tool_call_id(value: &Json) -> Option<String> {
    [
        &["tool_call_id"][..],
        &["toolCallId"],
        &["source_call_id"],
        &["sourceCallId"],
        &["delegation_tool_call_id"],
        &["delegationToolCallId"],
        &["parent_tool_call_id"],
        &["parentToolCallId"],
        &["extra", "tool_call_id"],
        &["extra", "toolCallId"],
        &["extra", "source_call_id"],
        &["extra", "sourceCallId"],
        &["extra", "delegation_tool_call_id"],
        &["extra", "delegationToolCallId"],
        &["extra", "parent_tool_call_id"],
        &["extra", "parentToolCallId"],
    ]
    .into_iter()
    .find_map(|path| json_string_at(value, path))
}

fn json_string_at(value: &Json, path: &[&str]) -> Option<String> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    current.as_str().map(ToString::to_string)
}

struct EventLookupMaps {
    name_map: std::collections::HashMap<Uuid, String>,
    start_ts_map: std::collections::HashMap<Uuid, DateTime<Utc>>,
    llm_start_model_names: HashMap<Uuid, String>,
    tool_call_ids: std::collections::HashMap<Uuid, String>,
    suppressed_llm_events: HashSet<Uuid>,
    supplemental_llm_metrics: HashMap<Uuid, AtifMetrics>,
}

impl EventLookupMaps {
    fn from_events(events: &[&Event]) -> Self {
        Self::from_events_with_correlation_events(events, events, events)
    }

    fn from_events_for_agent(events: &[&Event], tree: &AgentScopeTree, agent_uuid: Uuid) -> Self {
        let tool_correlation_events = events
            .iter()
            .copied()
            .filter(|event| tree.owner_agent(event) == Some(agent_uuid))
            .collect::<Vec<_>>();
        Self::from_events_with_correlation_events(events, events, &tool_correlation_events)
    }

    fn from_events_with_correlation_events(
        events: &[&Event],
        llm_dedupe_events: &[&Event],
        tool_correlation_events: &[&Event],
    ) -> Self {
        let mut name_map = std::collections::HashMap::new();
        let mut start_ts_map = std::collections::HashMap::new();
        let mut llm_start_model_names = HashMap::new();
        for event in events {
            if is_start_event(event) {
                name_map.insert(event.uuid(), event.name().to_string());
                start_ts_map.insert(event.uuid(), *event.timestamp());
                if event.category().map(|category| category.as_str()) == Some("llm")
                    && let Some(model_name) = model_name_for_llm_event(event)
                {
                    llm_start_model_names.insert(event.uuid(), model_name);
                }
            }
        }
        let llm_dedupe = build_llm_dedupe(llm_dedupe_events);
        Self {
            name_map,
            start_ts_map,
            llm_start_model_names,
            tool_call_ids: build_tool_call_correlations(tool_correlation_events),
            suppressed_llm_events: llm_dedupe.suppressed_events,
            supplemental_llm_metrics: llm_dedupe.supplemental_metrics,
        }
    }

    fn should_suppress_llm_event(&self, event: &Event) -> bool {
        event.category().map(|category| category.as_str()) == Some("llm")
            && self.suppressed_llm_events.contains(&event.uuid())
    }
}

#[derive(Default)]
struct LlmDedupeLookups {
    suppressed_events: HashSet<Uuid>,
    supplemental_metrics: HashMap<Uuid, AtifMetrics>,
}

#[derive(Default)]
struct LlmSpanParts<'a> {
    start: Option<&'a Event>,
    end: Option<&'a Event>,
}

#[derive(Debug, Clone)]
struct LlmSpanCandidate {
    uuid: Uuid,
    parent_uuid: Option<Uuid>,
    start_ts: DateTime<Utc>,
    end_ts: DateTime<Utc>,
    request_signature: String,
    request_correlation_keys: HashSet<String>,
    response_signature: String,
    model_name: Option<String>,
    fidelity_score: u8,
    end_metrics: Option<AtifMetrics>,
    hook_instrumentation: bool,
    gateway_instrumentation: bool,
    non_exact_provider_payload: bool,
}

fn build_llm_dedupe(events: &[&Event]) -> LlmDedupeLookups {
    let candidates = collect_llm_span_candidates(events);
    let mut lookups = LlmDedupeLookups::default();

    for (left_idx, left) in candidates.iter().enumerate() {
        for right in candidates.iter().skip(left_idx + 1) {
            if same_physical_llm_request(left, right) {
                suppress_lower_fidelity_llm_span(left, right, &mut lookups);
            }
        }
    }

    lookups
}

fn collect_llm_span_candidates(events: &[&Event]) -> Vec<LlmSpanCandidate> {
    let mut spans: HashMap<Uuid, LlmSpanParts<'_>> = HashMap::new();
    for event in events {
        if event.category().map(|category| category.as_str()) != Some("llm") {
            continue;
        }
        let parts = spans.entry(event.uuid()).or_default();
        match event.scope_category() {
            Some(crate::api::event::ScopeCategory::Start) => parts.start = Some(event),
            Some(crate::api::event::ScopeCategory::End) => parts.end = Some(event),
            None => {}
        }
    }

    spans
        .into_iter()
        .filter_map(|(uuid, parts)| LlmSpanCandidate::from_events(uuid, parts.start?, parts.end?))
        .collect()
}

impl LlmSpanCandidate {
    fn from_events(uuid: Uuid, start: &Event, end: &Event) -> Option<Self> {
        let request_signature = start.data().map(llm_request_signature)?;
        let response_signature = end.data().map(llm_response_signature)?;
        Some(Self {
            uuid,
            parent_uuid: start.parent_uuid().or_else(|| end.parent_uuid()),
            start_ts: *start.timestamp(),
            end_ts: *end.timestamp(),
            request_signature,
            request_correlation_keys: llm_request_correlation_keys(start, end),
            response_signature,
            model_name: start
                .model_name()
                .or_else(|| end.model_name())
                .map(ToOwned::to_owned)
                .or_else(|| model_name_for_llm_event(start))
                .or_else(|| model_name_for_llm_event(end)),
            fidelity_score: llm_event_fidelity_score(start).max(llm_event_fidelity_score(end)),
            end_metrics: end.data().and_then(|output| {
                let normalized_response = end.normalized_llm_response();
                let requested_model = start
                    .model_name()
                    .map(ToOwned::to_owned)
                    .or_else(|| model_name_for_llm_event(start))
                    .or_else(|| end.model_name().map(ToOwned::to_owned));
                extract_metrics(
                    output,
                    Some(end.name()),
                    requested_model.as_deref(),
                    normalized_response.as_deref(),
                )
            }),
            hook_instrumentation: is_hook_instrumented_llm_event(start)
                || is_hook_instrumented_llm_event(end),
            gateway_instrumentation: is_gateway_instrumented_llm_event(start)
                || is_gateway_instrumented_llm_event(end),
            non_exact_provider_payload: has_non_exact_provider_payload(start)
                || has_non_exact_provider_payload(end),
        })
    }
}

fn llm_request_signature(input: &Json) -> String {
    let content = unwrap_llm_request(input);
    json_to_string(&extract_user_messages(&content))
}

fn llm_response_signature(output: &Json) -> String {
    json_to_string(&serde_json::json!({
        "message": extract_llm_response_message(output),
        "tool_calls": extract_tool_calls(output),
    }))
}

fn llm_request_correlation_keys(start: &Event, end: &Event) -> HashSet<String> {
    let mut keys = HashSet::new();
    collect_llm_request_correlation_keys(start, &mut keys);
    collect_llm_request_correlation_keys(end, &mut keys);
    keys
}

fn collect_llm_request_correlation_keys(event: &Event, keys: &mut HashSet<String>) {
    if let Some(metadata) = event.metadata() {
        collect_request_correlation_values(metadata, keys);
    }
    if let Some(data) = event.data() {
        collect_request_correlation_values(data, keys);
        collect_request_correlation_values(&unwrap_llm_request(data), keys);
    }
}

fn collect_request_correlation_values(value: &Json, keys: &mut HashSet<String>) {
    for path in [
        &["api_call_id"][..],
        &["apiCallId"],
        &["request_id"],
        &["requestId"],
        &["request", "id"],
        &["metadata", "request_id"],
        &["metadata", "requestId"],
        &["extra", "api_call_id"],
        &["extra", "apiCallId"],
        &["extra", "request_id"],
        &["extra", "requestId"],
        &["llm_correlation_request_id"],
    ] {
        insert_correlation_key(keys, "request", json_string_at(value, path));
    }

    for path in [
        &["generation_id"][..],
        &["generationId"],
        &["generation", "id"],
        &["metadata", "generation_id"],
        &["metadata", "generationId"],
        &["extra", "generation_id"],
        &["extra", "generationId"],
        &["llm_correlation_generation_id"],
    ] {
        insert_correlation_key(keys, "generation", json_string_at(value, path));
    }
}

fn insert_correlation_key(keys: &mut HashSet<String>, kind: &str, value: Option<String>) {
    if let Some(value) = value.filter(|value| !value.is_empty()) {
        keys.insert(format!("{kind}:{value}"));
    }
}

fn same_physical_llm_request(left: &LlmSpanCandidate, right: &LlmSpanCandidate) -> bool {
    same_parent(left, right)
        && compatible_model_names(left, right)
        && llm_spans_overlap(left, right)
        && (same_llm_payload_signatures(left, right)
            || complementary_hook_and_gateway_spans(left, right))
}

fn same_llm_payload_signatures(left: &LlmSpanCandidate, right: &LlmSpanCandidate) -> bool {
    left.request_signature == right.request_signature
        && left.response_signature == right.response_signature
}

fn complementary_hook_and_gateway_spans(left: &LlmSpanCandidate, right: &LlmSpanCandidate) -> bool {
    let complementary_polarity = (left.non_exact_provider_payload
        && left.hook_instrumentation
        && right.gateway_instrumentation)
        || (right.non_exact_provider_payload
            && right.hook_instrumentation
            && left.gateway_instrumentation);

    complementary_polarity
        && (left.request_signature == right.request_signature
            || shared_llm_request_correlation_key(left, right))
}

fn shared_llm_request_correlation_key(left: &LlmSpanCandidate, right: &LlmSpanCandidate) -> bool {
    !left
        .request_correlation_keys
        .is_disjoint(&right.request_correlation_keys)
}

fn same_parent(left: &LlmSpanCandidate, right: &LlmSpanCandidate) -> bool {
    left.parent_uuid.is_some() && left.parent_uuid == right.parent_uuid
}

fn compatible_model_names(left: &LlmSpanCandidate, right: &LlmSpanCandidate) -> bool {
    match (&left.model_name, &right.model_name) {
        (Some(left_model), Some(right_model)) => left_model == right_model,
        _ => true,
    }
}

fn llm_spans_overlap(left: &LlmSpanCandidate, right: &LlmSpanCandidate) -> bool {
    left.start_ts <= right.end_ts && right.start_ts <= left.end_ts
}

fn suppress_lower_fidelity_llm_span(
    left: &LlmSpanCandidate,
    right: &LlmSpanCandidate,
    lookups: &mut LlmDedupeLookups,
) {
    match left.fidelity_score.cmp(&right.fidelity_score) {
        std::cmp::Ordering::Greater => suppress_llm_span(right, left, lookups),
        std::cmp::Ordering::Less => suppress_llm_span(left, right, lookups),
        std::cmp::Ordering::Equal => {}
    }
}

fn suppress_llm_span(
    suppressed: &LlmSpanCandidate,
    canonical: &LlmSpanCandidate,
    lookups: &mut LlmDedupeLookups,
) {
    lookups.suppressed_events.insert(suppressed.uuid);
    if let Some(metrics) = &suppressed.end_metrics {
        let entry = lookups
            .supplemental_metrics
            .entry(canonical.uuid)
            .or_default();
        merge_metrics_fields(entry, metrics);
    }
}

fn llm_event_fidelity_score(event: &Event) -> u8 {
    let Some(metadata) = event.metadata().and_then(Json::as_object) else {
        return 50;
    };
    if metadata
        .get("projection")
        .and_then(Json::as_bool)
        .unwrap_or(false)
    {
        return 10;
    }
    if has_non_exact_provider_payload(event) {
        return 30;
    }
    if metadata
        .get("provider_payload_exact")
        .and_then(Json::as_bool)
        .unwrap_or(false)
    {
        return 100;
    }
    if metadata.contains_key("fidelity_source") || metadata.contains_key("api_call_id") {
        return 95;
    }
    if metadata.contains_key("hook_event_name") {
        return 90;
    }
    if is_gateway_instrumented_llm_event(event) {
        return 50;
    }
    50
}

fn is_hook_instrumented_llm_event(event: &Event) -> bool {
    event
        .metadata()
        .and_then(Json::as_object)
        .is_some_and(|metadata| metadata.contains_key("hook_event_name"))
}

fn is_gateway_instrumented_llm_event(event: &Event) -> bool {
    event
        .metadata()
        .and_then(Json::as_object)
        .is_some_and(|metadata| {
            metadata.contains_key("gateway_path") || metadata.contains_key("llm_correlation_source")
        })
}

fn has_non_exact_provider_payload(event: &Event) -> bool {
    event
        .metadata()
        .and_then(Json::as_object)
        .and_then(|metadata| metadata.get("provider_payload_exact"))
        .and_then(Json::as_bool)
        == Some(false)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ToolCallMatchKey {
    name: String,
    arguments: String,
}

#[derive(Debug, Clone)]
struct ToolExecutionRecord {
    uuid: Uuid,
    explicit_call_id: Option<String>,
    key: Option<ToolCallMatchKey>,
}

#[derive(Debug, Clone)]
struct LlmToolCallRecord {
    tool_call_id: String,
    key: Option<ToolCallMatchKey>,
}

fn build_tool_call_correlations(events: &[&Event]) -> HashMap<Uuid, String> {
    let (mut correlations, executions, tool_calls) = collect_tool_correlation_inputs(events);
    let consumed_tool_call_ids = consumed_tool_call_ids(&correlations);
    let executions_by_key = group_unmatched_executions_by_key(executions, &correlations);
    let tool_calls_by_key = group_tool_calls_by_key(tool_calls, &consumed_tool_call_ids);
    apply_keyed_tool_correlations(&mut correlations, executions_by_key, &tool_calls_by_key);
    correlations
}

fn consumed_tool_call_ids(correlations: &HashMap<Uuid, String>) -> HashSet<String> {
    correlations.values().cloned().collect()
}

fn collect_tool_correlation_inputs(
    events: &[&Event],
) -> (
    HashMap<Uuid, String>,
    Vec<ToolExecutionRecord>,
    Vec<LlmToolCallRecord>,
) {
    let mut explicit = HashMap::new();
    let mut executions = Vec::new();
    let mut tool_calls = Vec::new();

    for event in events {
        collect_tool_correlation_event(event, &mut explicit, &mut executions, &mut tool_calls);
    }

    (explicit, executions, tool_calls)
}

fn collect_tool_correlation_event(
    event: &Event,
    explicit: &mut HashMap<Uuid, String>,
    executions: &mut Vec<ToolExecutionRecord>,
    tool_calls: &mut Vec<LlmToolCallRecord>,
) {
    match event_signature(event) {
        ("scope", Some(crate::api::event::ScopeCategory::Start), Some("tool")) => {
            collect_tool_execution_start(event, explicit, executions)
        }
        ("scope", Some(crate::api::event::ScopeCategory::End), Some("tool")) => {
            collect_explicit_tool_call_id(event, explicit)
        }
        ("scope", Some(crate::api::event::ScopeCategory::End), Some("llm")) => {
            collect_llm_tool_calls(event, tool_calls)
        }
        _ => {}
    }
}

fn event_signature(
    event: &Event,
) -> (&str, Option<crate::api::event::ScopeCategory>, Option<&str>) {
    (
        event.kind(),
        event.scope_category(),
        event.category().map(|category| category.as_str()),
    )
}

fn collect_tool_execution_start(
    event: &Event,
    explicit: &mut HashMap<Uuid, String>,
    executions: &mut Vec<ToolExecutionRecord>,
) {
    let record = ToolExecutionRecord {
        uuid: event.uuid(),
        explicit_call_id: event.tool_call_id().map(ToOwned::to_owned),
        key: tool_execution_match_key(event),
    };
    if let Some(tool_call_id) = &record.explicit_call_id {
        explicit.insert(record.uuid, tool_call_id.clone());
    }
    executions.push(record);
}

fn collect_explicit_tool_call_id(event: &Event, explicit: &mut HashMap<Uuid, String>) {
    if let Some(tool_call_id) = event.tool_call_id() {
        explicit.insert(event.uuid(), tool_call_id.to_string());
    }
}

fn collect_llm_tool_calls(event: &Event, tool_calls: &mut Vec<LlmToolCallRecord>) {
    let Some(calls) = event.data().and_then(extract_tool_calls) else {
        return;
    };
    tool_calls.extend(calls.into_iter().map(|tool_call| LlmToolCallRecord {
        key: tool_call_match_key(&tool_call.function_name, &tool_call.arguments),
        tool_call_id: tool_call.tool_call_id,
    }));
}

fn group_unmatched_executions_by_key(
    executions: Vec<ToolExecutionRecord>,
    correlations: &HashMap<Uuid, String>,
) -> HashMap<ToolCallMatchKey, Vec<Uuid>> {
    let mut grouped: HashMap<ToolCallMatchKey, Vec<Uuid>> = HashMap::new();
    for execution in executions {
        if correlations.contains_key(&execution.uuid) {
            continue;
        }
        if let Some(key) = execution.key {
            grouped.entry(key).or_default().push(execution.uuid);
        }
    }
    grouped
}

fn group_tool_calls_by_key(
    tool_calls: Vec<LlmToolCallRecord>,
    consumed_tool_call_ids: &HashSet<String>,
) -> HashMap<ToolCallMatchKey, Vec<String>> {
    let mut grouped: HashMap<ToolCallMatchKey, Vec<String>> = HashMap::new();
    for tool_call in tool_calls {
        if consumed_tool_call_ids.contains(&tool_call.tool_call_id) {
            continue;
        }
        if let Some(key) = tool_call.key {
            grouped.entry(key).or_default().push(tool_call.tool_call_id);
        }
    }
    grouped
}

fn apply_keyed_tool_correlations(
    correlations: &mut HashMap<Uuid, String>,
    executions_by_key: HashMap<ToolCallMatchKey, Vec<Uuid>>,
    tool_calls_by_key: &HashMap<ToolCallMatchKey, Vec<String>>,
) {
    for (key, execution_uuids) in executions_by_key {
        let Some(tool_call_ids) = tool_calls_by_key.get(&key) else {
            continue;
        };
        if execution_uuids.len() == tool_call_ids.len() {
            insert_keyed_tool_correlations(correlations, execution_uuids, tool_call_ids);
        }
    }
}

fn insert_keyed_tool_correlations(
    correlations: &mut HashMap<Uuid, String>,
    execution_uuids: Vec<Uuid>,
    tool_call_ids: &[String],
) {
    for (uuid, tool_call_id) in execution_uuids.into_iter().zip(tool_call_ids) {
        correlations.insert(uuid, tool_call_id.clone());
    }
}

fn tool_execution_match_key(event: &Event) -> Option<ToolCallMatchKey> {
    let arguments = event
        .data()
        .map(|data| normalize_tool_arguments(Some(data)))?;
    tool_call_match_key(event.name(), &arguments)
}

fn tool_call_match_key(name: &str, arguments: &Json) -> Option<ToolCallMatchKey> {
    if name.is_empty() {
        return None;
    }
    Some(ToolCallMatchKey {
        name: name.to_string(),
        arguments: json_to_string(arguments),
    })
}

#[derive(Default)]
struct PendingAgentStep {
    step_idx: Option<usize>,
    ancestry: Option<AtifAncestry>,
    invocation: Option<AtifInvocationInfo>,
    llm_response: Option<Json>,
    tool_ancestry: Vec<AtifAncestry>,
    tool_invocations: Vec<AtifInvocationInfo>,
    tool_call_order: Vec<String>,
}

impl PendingAgentStep {
    fn finalize_into(&mut self, steps: &mut [AtifStep]) {
        let (Some(step_idx), Some(ancestry)) = (self.step_idx.take(), self.ancestry.take()) else {
            return;
        };
        let Some(step) = steps.get_mut(step_idx) else {
            return;
        };

        self.sort_tool_metadata();
        let extra = AtifStepExtra {
            ancestry,
            invocation: self.invocation.take(),
            llm_request: None,
            llm_response: self.llm_response.take(),
            event_payload: None,
            tool_ancestry: std::mem::take(&mut self.tool_ancestry),
            tool_invocations: if self.tool_invocations.is_empty() {
                None
            } else {
                Some(std::mem::take(&mut self.tool_invocations))
            },
        };
        step.extra = serde_json::to_value(&extra).ok();
    }

    fn set_current_agent(
        &mut self,
        step_idx: usize,
        ancestry: AtifAncestry,
        invocation: AtifInvocationInfo,
        tool_call_order: Vec<String>,
        llm_response: Json,
    ) {
        self.step_idx = Some(step_idx);
        self.ancestry = Some(ancestry);
        self.invocation = Some(invocation);
        self.llm_response = Some(llm_response);
        self.tool_ancestry.clear();
        self.tool_invocations.clear();
        self.tool_call_order = tool_call_order;
    }

    fn push_tool_metadata(&mut self, ancestry: AtifAncestry, invocation: AtifInvocationInfo) {
        self.tool_ancestry.push(ancestry);
        self.tool_invocations.push(invocation);
    }

    fn push_tool_call_id(&mut self, tool_call_id: String) {
        if !self
            .tool_call_order
            .iter()
            .any(|known_id| known_id == &tool_call_id)
        {
            self.tool_call_order.push(tool_call_id);
        }
    }

    fn has_active_step(&self) -> bool {
        self.step_idx.is_some()
    }

    fn has_tool_call_id(&self, tool_call_id: &str) -> bool {
        self.tool_call_order
            .iter()
            .any(|known_id| known_id == tool_call_id)
    }

    fn sort_tool_metadata(&mut self) {
        if self.tool_call_order.is_empty() || self.tool_ancestry.is_empty() {
            return;
        }

        let mut pairs: Vec<(AtifAncestry, AtifInvocationInfo)> =
            std::mem::take(&mut self.tool_ancestry)
                .into_iter()
                .zip(std::mem::take(&mut self.tool_invocations))
                .collect();
        pairs.sort_by_key(|(_, invocation)| {
            invocation
                .invocation_id
                .as_deref()
                .and_then(|id| self.tool_call_order.iter().position(|entry| entry == id))
                .unwrap_or(usize::MAX)
        });
        let (sorted_ancestry, sorted_invocations): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
        self.tool_ancestry = sorted_ancestry;
        self.tool_invocations = sorted_invocations;
    }
}

#[derive(Default)]
struct StepConversionState {
    steps: Vec<AtifStep>,
    last_tool_call_map: std::collections::HashMap<String, String>,
    tool_scope_call_ids: std::collections::HashMap<Uuid, String>,
    active_tool_call_id: Option<String>,
    pending_observations: Vec<AtifObservationResult>,
    pending_obs_timestamp: Option<String>,
    deferred_observations: HashMap<String, Vec<DeferredToolObservation>>,
    deferred_tool_metadata: HashMap<String, Vec<(AtifAncestry, AtifInvocationInfo)>>,
    current_reasoning_effort: Option<Json>,
    current_agent: PendingAgentStep,
}

struct DeferredToolObservation {
    result: AtifObservationResult,
    timestamp: Option<String>,
}

impl StepConversionState {
    fn handle_event(&mut self, event: &Event, lookups: &EventLookupMaps) {
        if lookups.should_suppress_llm_event(event) {
            return;
        }
        match (
            event.kind(),
            event.scope_category(),
            event.category().map(|category| category.as_str()),
        ) {
            ("scope", Some(crate::api::event::ScopeCategory::Start), Some("llm")) => {
                self.handle_llm_start(event, lookups)
            }
            ("scope", Some(crate::api::event::ScopeCategory::End), Some("llm")) => {
                self.handle_llm_end(event, lookups)
            }
            ("scope", Some(crate::api::event::ScopeCategory::Start), Some("tool")) => {
                self.handle_tool_start(event, lookups)
            }
            ("scope", Some(crate::api::event::ScopeCategory::End), Some("tool")) => {
                self.handle_tool_end(event, lookups)
            }
            _ => {}
        }
    }

    fn flush_observations(&mut self) {
        if self.pending_observations.is_empty() {
            return;
        }

        let timestamp = self.pending_obs_timestamp.take();
        let observations = std::mem::take(&mut self.pending_observations);
        let (attached, standalone) = self.route_observations(observations, timestamp.clone());
        self.attach_observations_to_current_step(attached);
        self.push_standalone_observation_step(standalone, timestamp);
    }

    fn route_observations(
        &mut self,
        observations: Vec<AtifObservationResult>,
        timestamp: Option<String>,
    ) -> (Vec<AtifObservationResult>, Vec<AtifObservationResult>) {
        let mut attached = Vec::new();
        let mut standalone = Vec::new();
        for mut result in observations {
            match result.source_call_id.clone() {
                Some(source_call_id) => {
                    self.route_correlated_observation(
                        source_call_id,
                        result,
                        timestamp.clone(),
                        &mut attached,
                    );
                }
                None => {
                    result.source_call_id = None;
                    standalone.push(result);
                }
            }
        }
        (attached, standalone)
    }

    fn route_correlated_observation(
        &mut self,
        source_call_id: String,
        result: AtifObservationResult,
        timestamp: Option<String>,
        attached: &mut Vec<AtifObservationResult>,
    ) {
        if self.current_step_has_tool_call(&source_call_id) {
            attached.push(result);
        } else {
            self.defer_observation(source_call_id, result, timestamp);
        }
    }

    fn attach_observations_to_current_step(&mut self, attached: Vec<AtifObservationResult>) {
        if attached.is_empty() {
            return;
        }
        let Some(step_idx) = self.current_agent.step_idx else {
            return;
        };
        let Some(step) = self.steps.get_mut(step_idx) else {
            return;
        };
        let observation = step.observation.get_or_insert_with(|| AtifObservation {
            results: Vec::new(),
        });
        for result in attached {
            merge_observation_result(observation, result);
        }
    }

    fn push_standalone_observation_step(
        &mut self,
        mut observations: Vec<AtifObservationResult>,
        timestamp: Option<String>,
    ) {
        if observations.is_empty() {
            return;
        }

        for result in &mut observations {
            if result.source_call_id.is_some() {
                result.source_call_id = None;
            }
        }

        self.steps.push(AtifStep {
            step_id: 0,
            source: "system".to_string(),
            message: empty_message(),
            timestamp,
            model_name: None,
            reasoning_effort: None,
            reasoning_content: None,
            tool_calls: None,
            observation: Some(AtifObservation {
                results: observations,
            }),
            metrics: None,
            llm_call_count: None,
            is_copied_context: None,
            extra: None,
        });
    }

    fn finalize_agent_extra(&mut self) {
        self.current_agent.finalize_into(&mut self.steps);
    }

    fn current_step_has_tool_call(&self, source_call_id: &str) -> bool {
        let Some(step_idx) = self.current_agent.step_idx else {
            return false;
        };
        self.steps
            .get(step_idx)
            .and_then(|step| step.tool_calls.as_deref())
            .unwrap_or_default()
            .iter()
            .any(|tool_call| tool_call.tool_call_id == source_call_id)
    }

    fn defer_observation(
        &mut self,
        source_call_id: String,
        result: AtifObservationResult,
        timestamp: Option<String>,
    ) {
        self.deferred_observations
            .entry(source_call_id)
            .or_default()
            .push(DeferredToolObservation { result, timestamp });
    }

    fn attach_deferred_to_current_agent(&mut self) {
        let Some(step_idx) = self.current_agent.step_idx else {
            return;
        };
        let tool_call_ids = self.tool_call_ids_for_step(step_idx);

        for source_call_id in tool_call_ids {
            self.attach_deferred_observations(step_idx, &source_call_id);
            self.attach_deferred_tool_metadata(&source_call_id);
        }
    }

    fn tool_call_ids_for_step(&self, step_idx: usize) -> Vec<String> {
        self.steps
            .get(step_idx)
            .and_then(|step| step.tool_calls.as_deref())
            .unwrap_or_default()
            .iter()
            .map(|tool_call| tool_call.tool_call_id.clone())
            .collect()
    }

    fn attach_deferred_observations(&mut self, step_idx: usize, source_call_id: &str) {
        let Some(observations) = self.deferred_observations.remove(source_call_id) else {
            return;
        };
        let Some(step) = self.steps.get_mut(step_idx) else {
            return;
        };
        let observation = step.observation.get_or_insert_with(|| AtifObservation {
            results: Vec::new(),
        });
        for deferred in observations {
            merge_observation_result(observation, deferred.result);
        }
    }

    fn attach_deferred_tool_metadata(&mut self, source_call_id: &str) {
        let Some(metadata) = self.deferred_tool_metadata.remove(source_call_id) else {
            return;
        };
        for (ancestry, invocation) in metadata {
            self.current_agent.push_tool_metadata(ancestry, invocation);
        }
    }

    fn flush_deferred_observations_as_standalone(&mut self) {
        if self.deferred_observations.is_empty() {
            return;
        }
        let mut deferred = std::mem::take(&mut self.deferred_observations)
            .into_values()
            .flatten()
            .collect::<Vec<_>>();
        deferred.sort_by_key(|entry| entry.timestamp.clone());
        let timestamp = deferred.iter().find_map(|entry| entry.timestamp.clone());
        let mut observations = deferred
            .into_iter()
            .map(|mut entry| {
                entry.result.source_call_id = None;
                entry.result
            })
            .collect::<Vec<_>>();
        if observations.is_empty() {
            return;
        }
        self.deferred_tool_metadata.clear();
        self.steps.push(AtifStep {
            step_id: 0,
            source: "system".to_string(),
            message: empty_message(),
            timestamp,
            model_name: None,
            reasoning_effort: None,
            reasoning_content: None,
            tool_calls: None,
            observation: Some(AtifObservation {
                results: std::mem::take(&mut observations),
            }),
            metrics: None,
            llm_call_count: None,
            is_copied_context: None,
            extra: None,
        });
    }

    fn handle_llm_start(&mut self, event: &Event, lookups: &EventLookupMaps) {
        self.flush_observations();
        self.finalize_agent_extra();
        self.tool_scope_call_ids.clear();
        self.active_tool_call_id = None;

        let Some(input) = event.data() else {
            return;
        };
        let content = unwrap_llm_request(input);
        self.current_reasoning_effort = extract_reasoning_effort(&content);
        let extra = AtifStepExtra {
            ancestry: build_ancestry(event, &lookups.name_map),
            invocation: None,
            llm_request: Some(content.clone()),
            llm_response: None,
            event_payload: None,
            tool_ancestry: Vec::new(),
            tool_invocations: None,
        };
        self.steps.push(AtifStep {
            step_id: 0,
            source: "user".to_string(),
            message: event
                .annotated_request()
                .and_then(|request| atif_message_from_annotated_request(request))
                .unwrap_or_else(|| extract_user_messages(&content)),
            timestamp: Some(event.timestamp().to_rfc3339()),
            model_name: None,
            reasoning_effort: None,
            reasoning_content: None,
            tool_calls: None,
            observation: None,
            metrics: None,
            llm_call_count: None,
            is_copied_context: None,
            extra: serde_json::to_value(&extra).ok(),
        });
    }

    fn handle_llm_end(&mut self, event: &Event, lookups: &EventLookupMaps) {
        self.flush_observations();

        let Some(output) = event.data() else {
            return;
        };
        let tool_calls = extract_tool_calls(output);
        let tool_call_order = refresh_tool_call_lookup(&mut self.last_tool_call_map, &tool_calls);
        let reasoning_effort = self.current_reasoning_effort.take();
        let reasoning_content = extract_reasoning_content(output);
        let start_ts = lookups.start_ts_map.get(&event.uuid()).cloned();
        let paired_start_model = lookups
            .llm_start_model_names
            .get(&event.uuid())
            .map(String::as_str);
        let ancestry = build_ancestry(event, &lookups.name_map);
        let invocation = build_invocation_info(
            start_ts,
            *event.timestamp(),
            Some(event.uuid().to_string()),
            "nemo_relay",
        );

        let metrics = merge_metrics(
            {
                let normalized_response = event.normalized_llm_response();
                extract_metrics(
                    output,
                    Some(event.name()),
                    paired_start_model.or_else(|| event.model_name()),
                    normalized_response.as_deref(),
                )
            },
            lookups.supplemental_llm_metrics.get(&event.uuid()),
        );

        self.steps.push(AtifStep {
            step_id: 0,
            source: "agent".to_string(),
            message: event
                .annotated_response()
                .and_then(|response| atif_message_from_annotated_response(response))
                .unwrap_or_else(|| extract_llm_response_message(output)),
            timestamp: Some(event.timestamp().to_rfc3339()),
            model_name: model_name_for_llm_event(event),
            reasoning_effort,
            reasoning_content,
            tool_calls,
            observation: None,
            metrics,
            llm_call_count: Some(1),
            is_copied_context: None,
            extra: None,
        });
        self.current_agent.set_current_agent(
            self.steps.len() - 1,
            ancestry,
            invocation,
            tool_call_order,
            output.clone(),
        );
        self.attach_deferred_to_current_agent();
    }

    fn handle_tool_start(&mut self, event: &Event, lookups: &EventLookupMaps) {
        let Some(source_call_id) = self.source_call_id_for_tool_start(event, lookups) else {
            return;
        };
        self.tool_scope_call_ids
            .insert(event.uuid(), source_call_id.clone());
        if !self.current_agent.has_active_step() {
            return;
        }
        if !self.ensure_tool_call_on_current_agent(event, &source_call_id) {
            return;
        }
        self.active_tool_call_id = Some(source_call_id);
    }

    fn source_call_id_for_tool_start(
        &self,
        event: &Event,
        lookups: &EventLookupMaps,
    ) -> Option<String> {
        event
            .tool_call_id()
            .map(ToOwned::to_owned)
            .or_else(|| self.tool_scope_call_ids.get(&event.uuid()).cloned())
            .or_else(|| lookups.tool_call_ids.get(&event.uuid()).cloned())
            .or_else(|| self.current_step_tool_call_id_by_name(event.name()))
            .or_else(|| self.synthetic_tool_call_id_for_start(event))
    }

    fn current_step_tool_call_id_by_name(&self, name: &str) -> Option<String> {
        let step_idx = self.current_agent.step_idx?;
        let matches = self.current_step_tool_call_ids_by_name(step_idx, name);
        match matches.as_slice() {
            [tool_call_id] => Some(tool_call_id.clone()),
            _ => None,
        }
    }

    fn current_step_tool_call_ids_by_name(&self, step_idx: usize, name: &str) -> Vec<String> {
        self.steps
            .get(step_idx)
            .and_then(|step| step.tool_calls.as_deref())
            .unwrap_or_default()
            .iter()
            .filter(|tool_call| tool_call.function_name == name)
            .map(|tool_call| tool_call.tool_call_id.clone())
            .collect()
    }

    fn synthetic_tool_call_id_for_start(&self, event: &Event) -> Option<String> {
        if self.current_step_has_duplicate_tool_name(event.name()) {
            return None;
        }
        self.current_agent
            .has_active_step()
            .then(|| event.uuid().to_string())
    }

    fn current_step_has_duplicate_tool_name(&self, name: &str) -> bool {
        let Some(step_idx) = self.current_agent.step_idx else {
            return false;
        };
        self.current_step_tool_call_ids_by_name(step_idx, name)
            .len()
            > 1
    }

    fn ensure_tool_call_on_current_agent(&mut self, event: &Event, source_call_id: &str) -> bool {
        if self.current_agent.has_tool_call_id(source_call_id) {
            self.tool_scope_call_ids
                .insert(event.uuid(), source_call_id.to_string());
            return true;
        }
        let Some(step_idx) = self.current_agent.step_idx else {
            return false;
        };
        let Some(step) = self.steps.get_mut(step_idx) else {
            return false;
        };
        let tool_calls = step.tool_calls.get_or_insert_with(Vec::new);
        if tool_calls
            .iter()
            .any(|tool_call| tool_call.tool_call_id == source_call_id)
        {
            self.current_agent
                .push_tool_call_id(source_call_id.to_string());
            return true;
        }
        tool_calls.push(AtifToolCall {
            tool_call_id: source_call_id.to_string(),
            function_name: event.name().to_string(),
            arguments: event
                .data()
                .cloned()
                .unwrap_or_else(|| serde_json::json!({})),
            extra: Some(event_extra(event)),
        });
        self.current_agent
            .push_tool_call_id(source_call_id.to_string());
        if !event.name().is_empty() {
            self.last_tool_call_map
                .insert(event.name().to_string(), source_call_id.to_string());
        }
        true
    }

    fn resolve_source_call_id(&self, event: &Event, lookups: &EventLookupMaps) -> Option<String> {
        if let Some(tool_call_id) = event.tool_call_id() {
            return Some(tool_call_id.to_string());
        }
        if let Some(tool_call_id) = self.tool_scope_call_ids.get(&event.uuid()) {
            return Some(tool_call_id.clone());
        }
        if let Some(tool_call_id) = lookups.tool_call_ids.get(&event.uuid()) {
            return Some(tool_call_id.clone());
        }

        let candidate = self.current_step_tool_call_id_by_name(event.name())?;

        if self.current_agent.has_tool_call_id(&candidate)
            || self
                .last_tool_call_map
                .values()
                .any(|known_id| known_id == &candidate)
            || self.deferred_observations.contains_key(&candidate)
            || self.deferred_tool_metadata.contains_key(&candidate)
        {
            Some(candidate)
        } else {
            None
        }
    }

    fn handle_tool_end(&mut self, event: &Event, lookups: &EventLookupMaps) {
        let source_call_id = self.resolve_source_call_id(event, lookups);
        if let Some(output) = event.data() {
            if self.pending_obs_timestamp.is_none() {
                self.pending_obs_timestamp = Some(event.timestamp().to_rfc3339());
            }
            self.pending_observations.push(AtifObservationResult {
                source_call_id: source_call_id.clone(),
                content: observation_content_value(output),
                subagent_trajectory_ref: None,
                extra: Some(observation_extra(event, output)),
            });
        }

        if self.active_tool_call_id.as_deref() == source_call_id.as_deref() {
            self.active_tool_call_id = None;
        }

        let Some(source_call_id) = source_call_id else {
            return;
        };
        let start_ts = lookups.start_ts_map.get(&event.uuid()).cloned();
        let invocation = build_invocation_info(
            start_ts,
            *event.timestamp(),
            Some(source_call_id.clone()),
            "nemo_relay",
        );
        let ancestry = build_ancestry(event, &lookups.name_map);
        if self.current_agent.has_active_step()
            && self.current_agent.has_tool_call_id(&source_call_id)
        {
            self.current_agent.push_tool_metadata(ancestry, invocation);
        } else {
            self.deferred_tool_metadata
                .entry(source_call_id)
                .or_default()
                .push((ancestry, invocation));
        }
    }

    fn resolve_subagent_source_call_id(&self, event: &Event) -> Option<String> {
        let candidate = event
            .metadata()
            .and_then(delegation_tool_call_id)
            .or_else(|| event.data().and_then(delegation_tool_call_id))
            .or_else(|| self.active_tool_call_id.clone())?;

        self.current_agent
            .has_tool_call_id(&candidate)
            .then_some(candidate)
    }

    fn subagent_reference_result(
        child: &AgentScopeNode,
        event: &Event,
        source_call_id: Option<String>,
    ) -> AtifObservationResult {
        AtifObservationResult {
            source_call_id,
            content: None,
            subagent_trajectory_ref: Some(vec![AtifSubagentTrajectoryRef {
                trajectory_id: Some(child.uuid.to_string()),
                session_id: child.session_id.clone(),
                extra: Some(serde_json::json!({
                    "name": child.name.clone(),
                    "scope_uuid": child.uuid.to_string(),
                })),
            }]),
            extra: Some(event_extra(event)),
        }
    }

    fn attach_subagent_ref_to_agent_step(
        &mut self,
        child: &AgentScopeNode,
        event: &Event,
        source_call_id: &str,
    ) -> bool {
        let Some(step_idx) = self.current_agent.step_idx else {
            return false;
        };
        let Some(step) = self.steps.get_mut(step_idx) else {
            return false;
        };
        let Some(tool_calls) = step.tool_calls.as_deref() else {
            return false;
        };
        if !tool_calls
            .iter()
            .any(|tool_call| tool_call.tool_call_id == source_call_id)
        {
            return false;
        }

        let observation = step.observation.get_or_insert_with(|| AtifObservation {
            results: Vec::new(),
        });
        if let Some(result) = observation
            .results
            .iter_mut()
            .find(|result| result.source_call_id.as_deref() == Some(source_call_id))
        {
            let refs = result.subagent_trajectory_ref.get_or_insert_with(Vec::new);
            refs.push(AtifSubagentTrajectoryRef {
                trajectory_id: Some(child.uuid.to_string()),
                session_id: child.session_id.clone(),
                extra: Some(serde_json::json!({
                    "name": child.name.clone(),
                    "scope_uuid": child.uuid.to_string(),
                })),
            });
            return true;
        }

        observation.results.push(Self::subagent_reference_result(
            child,
            event,
            Some(source_call_id.to_string()),
        ));
        true
    }

    fn handle_subagent_start(&mut self, child: &AgentScopeNode, event: &Event) {
        let source_call_id = self.resolve_subagent_source_call_id(event);
        self.flush_observations();
        if let Some(source_call_id) = source_call_id
            && self.attach_subagent_ref_to_agent_step(child, event, &source_call_id)
        {
            return;
        }
        self.finalize_agent_extra();

        let source_call_id = format!("subagent:{}", child.uuid);
        self.steps.push(AtifStep {
            step_id: 0,
            source: "agent".to_string(),
            message: empty_message(),
            timestamp: Some(event.timestamp().to_rfc3339()),
            model_name: None,
            reasoning_effort: None,
            reasoning_content: None,
            tool_calls: Some(vec![AtifToolCall {
                tool_call_id: source_call_id.clone(),
                function_name: child.name.clone(),
                arguments: subagent_dispatch_arguments(child, event),
                extra: Some(event_extra(event)),
            }]),
            observation: Some(AtifObservation {
                results: vec![Self::subagent_reference_result(
                    child,
                    event,
                    Some(source_call_id),
                )],
            }),
            metrics: None,
            llm_call_count: Some(0),
            is_copied_context: None,
            extra: None,
        });
    }

    fn finish(mut self) -> Vec<AtifStep> {
        self.flush_observations();
        self.flush_deferred_observations_as_standalone();
        self.finalize_agent_extra();
        remove_projected_tool_call_duplicates(&mut self.steps);
        renumber_steps(&mut self.steps);
        self.steps
    }
}

fn remove_projected_tool_call_duplicates(steps: &mut Vec<AtifStep>) {
    let mut observed_later = HashSet::new();
    let mut keep = vec![true; steps.len()];

    for (idx, step) in steps.iter().enumerate().rev() {
        if step.source != "agent" {
            observed_later.clear();
            continue;
        }
        if projected_tool_call_duplicate(step, &observed_later) {
            keep[idx] = false;
        }
        extend_observed_tool_call_keys(&mut observed_later, step);
    }

    let mut idx = 0;
    steps.retain(|_| {
        let retain_step = keep[idx];
        idx += 1;
        retain_step
    });
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ToolCallDedupeKey {
    tool_call_id: String,
    function_name: String,
    arguments: String,
}

fn projected_tool_call_duplicate(
    step: &AtifStep,
    observed_later: &HashSet<ToolCallDedupeKey>,
) -> bool {
    if !projected_tool_call_candidate(step) {
        return false;
    }
    let tool_call_keys = step_tool_call_dedupe_keys(step);
    !tool_call_keys.is_empty()
        && tool_call_keys
            .iter()
            .all(|tool_call_key| observed_later.contains(tool_call_key))
}

fn projected_tool_call_candidate(step: &AtifStep) -> bool {
    step.source == "agent"
        && step.message == empty_message()
        && step.observation.is_none()
        && step.reasoning_content.is_none()
        && step.reasoning_effort.is_none()
        && step.llm_call_count == Some(1)
        && step
            .tool_calls
            .as_ref()
            .is_some_and(|tool_calls| !tool_calls.is_empty())
}

fn extend_observed_tool_call_keys(
    observed_later: &mut HashSet<ToolCallDedupeKey>,
    step: &AtifStep,
) {
    for tool_call_key in observed_tool_call_keys(step) {
        observed_later.insert(tool_call_key);
    }
}

fn observed_tool_call_keys(step: &AtifStep) -> Vec<ToolCallDedupeKey> {
    let step_tool_call_keys = step_tool_call_dedupe_keys(step);
    step.observation
        .as_ref()
        .map(|observation| matching_observation_tool_call_keys(observation, &step_tool_call_keys))
        .unwrap_or_default()
}

fn matching_observation_tool_call_keys(
    observation: &AtifObservation,
    step_tool_call_keys: &[ToolCallDedupeKey],
) -> Vec<ToolCallDedupeKey> {
    observation
        .results
        .iter()
        .filter_map(|result| result.source_call_id.as_ref())
        .flat_map(|source_call_id| matching_tool_call_keys(source_call_id, step_tool_call_keys))
        .collect()
}

fn matching_tool_call_keys(
    source_call_id: &str,
    step_tool_call_keys: &[ToolCallDedupeKey],
) -> Vec<ToolCallDedupeKey> {
    step_tool_call_keys
        .iter()
        .filter(|tool_call_key| tool_call_key.tool_call_id == source_call_id)
        .cloned()
        .collect()
}

fn step_tool_call_dedupe_keys(step: &AtifStep) -> Vec<ToolCallDedupeKey> {
    step.tool_calls
        .as_deref()
        .unwrap_or_default()
        .iter()
        .filter(|tool_call| !tool_call.tool_call_id.is_empty())
        .map(|tool_call| ToolCallDedupeKey {
            tool_call_id: tool_call.tool_call_id.clone(),
            function_name: tool_call.function_name.clone(),
            arguments: json_to_string(&tool_call.arguments),
        })
        .collect()
}

#[cfg(test)]
fn step_tool_call_ids(step: &AtifStep) -> Vec<String> {
    step.tool_calls
        .as_deref()
        .unwrap_or_default()
        .iter()
        .map(|tool_call| tool_call.tool_call_id.clone())
        .filter(|tool_call_id| !tool_call_id.is_empty())
        .collect()
}

fn merge_observation_result(observation: &mut AtifObservation, mut result: AtifObservationResult) {
    if let Some(source_call_id) = result.source_call_id.as_deref()
        && let Some(existing) = observation
            .results
            .iter_mut()
            .find(|existing| existing.source_call_id.as_deref() == Some(source_call_id))
    {
        if existing.content.is_none() {
            existing.content = result.content.take();
        }
        if let Some(mut refs) = result.subagent_trajectory_ref.take() {
            existing
                .subagent_trajectory_ref
                .get_or_insert_with(Vec::new)
                .append(&mut refs);
        }
        if let Some(extra) = result.extra.take() {
            merge_observation_extra(&mut existing.extra, extra);
        }
        return;
    }

    observation.results.push(result);
}

fn merge_observation_extra(existing: &mut Option<Json>, incoming: Json) {
    let Some(existing_extra) = existing.as_mut() else {
        *existing = Some(incoming);
        return;
    };
    let (Json::Object(existing_object), Json::Object(incoming_object)) = (existing_extra, incoming)
    else {
        return;
    };
    for (key, value) in incoming_object {
        existing_object.entry(key).or_insert(value);
    }
}

fn subagent_dispatch_arguments(child: &AgentScopeNode, event: &Event) -> Json {
    let mut arguments = serde_json::Map::new();
    arguments.insert("name".to_string(), Json::String(child.name.clone()));
    if let Some(session_id) = &child.session_id {
        arguments.insert("session_id".to_string(), Json::String(session_id.clone()));
    }
    if let Some(data) = event.data()
        && !data.is_null()
    {
        arguments.insert("payload".to_string(), data.clone());
    }
    Json::Object(arguments)
}

fn prune_subagent_refs(steps: &mut Vec<AtifStep>, child_trajectory_ids: &HashSet<String>) {
    for step in steps.iter_mut() {
        let Some(observation) = &mut step.observation else {
            continue;
        };
        observation.results.retain_mut(|result| {
            if let Some(refs) = &mut result.subagent_trajectory_ref {
                refs.retain(|reference| {
                    reference
                        .trajectory_id
                        .as_ref()
                        .is_some_and(|trajectory_id| child_trajectory_ids.contains(trajectory_id))
                });
                if refs.is_empty() {
                    result.subagent_trajectory_ref = None;
                }
            }
            result.content.is_some()
                || result.subagent_trajectory_ref.is_some()
                || observation_result_has_tool_result_extra(result)
        });
        if observation.results.is_empty() {
            step.observation = None;
        }
    }
    steps.retain(|step| {
        !(step.source == "system"
            && step.observation.is_none()
            && step.message == empty_message()
            && step.extra.is_none())
            && !(step.source == "agent"
                && step.llm_call_count == Some(0)
                && step.observation.is_none()
                && step.message == empty_message()
                && step.extra.is_none())
    });
    renumber_steps(steps);
}

fn renumber_steps(steps: &mut [AtifStep]) {
    for (index, step) in steps.iter_mut().enumerate() {
        step.step_id = index + 1;
    }
}

fn observation_result_has_tool_result_extra(result: &AtifObservationResult) -> bool {
    result
        .extra
        .as_ref()
        .and_then(|extra| extra.as_object())
        .is_some_and(|extra| extra.contains_key("tool_result"))
}

fn refresh_tool_call_lookup(
    last_tool_call_map: &mut std::collections::HashMap<String, String>,
    tool_calls: &Option<Vec<AtifToolCall>>,
) -> Vec<String> {
    last_tool_call_map.clear();
    let mut tool_call_order = Vec::new();
    if let Some(tool_calls) = tool_calls {
        for tool_call in tool_calls {
            if !tool_call.function_name.is_empty() {
                last_tool_call_map.insert(
                    tool_call.function_name.clone(),
                    tool_call.tool_call_id.clone(),
                );
            }
            tool_call_order.push(tool_call.tool_call_id.clone());
        }
    }
    tool_call_order
}

#[derive(Debug, Clone)]
struct AgentScopeNode {
    uuid: Uuid,
    name: String,
    session_id: Option<String>,
    referenced_by_parent: bool,
    parent_agent: Option<Uuid>,
    children: Vec<Uuid>,
    start_timestamp: DateTime<Utc>,
}

struct AgentScopeTree {
    nodes: HashMap<Uuid, AgentScopeNode>,
    roots: Vec<Uuid>,
    scope_parent_map: HashMap<Uuid, Uuid>,
    agent_uuids: HashSet<Uuid>,
}

type AgentScopeRoles = HashMap<Uuid, Option<String>>;

impl AgentScopeTree {
    fn from_events(events: &[&Event]) -> Self {
        let (scope_parent_map, agent_scope_roles) = agent_scope_maps(events);
        let agent_uuids = agent_uuids_from_events(events, &scope_parent_map, &agent_scope_roles);
        let mut nodes = agent_scope_nodes(events, &scope_parent_map, &agent_uuids);
        let mut roots = link_agent_children(&mut nodes);
        sort_agent_tree(&mut roots, &mut nodes);

        Self {
            nodes,
            roots,
            scope_parent_map,
            agent_uuids,
        }
    }

    fn choose_root(&self, session_id: &str) -> Option<Uuid> {
        Uuid::parse_str(session_id)
            .ok()
            .filter(|uuid| self.nodes.contains_key(uuid))
            .or_else(|| (self.roots.len() == 1).then(|| self.roots[0]))
    }

    fn owner_agent(&self, event: &Event) -> Option<Uuid> {
        if event.scope_type() == Some(crate::api::scope::ScopeType::Agent) {
            return Some(event.uuid()).filter(|uuid| self.agent_uuids.contains(uuid));
        }
        nearest_agent_parent(
            event.parent_uuid(),
            &self.scope_parent_map,
            &self.agent_uuids,
            None,
        )
    }

    fn direct_child_for_start(&self, parent: Uuid, event: &Event) -> Option<&AgentScopeNode> {
        if !is_start_event(event) || event.scope_type() != Some(crate::api::scope::ScopeType::Agent)
        {
            return None;
        }
        let child = self.nodes.get(&event.uuid())?;
        (child.parent_agent == Some(parent)).then_some(child)
    }
}

fn agent_scope_maps(events: &[&Event]) -> (HashMap<Uuid, Uuid>, AgentScopeRoles) {
    let mut scope_parent_map = HashMap::new();
    let mut agent_scope_roles = HashMap::new();

    for event in events.iter().copied().filter(|event| is_start_event(event)) {
        if let Some(parent_uuid) = event.parent_uuid() {
            scope_parent_map.insert(event.uuid(), parent_uuid);
        }
        if event.scope_type() == Some(crate::api::scope::ScopeType::Agent) {
            agent_scope_roles.insert(event.uuid(), agent_scope_role(event).map(str::to_string));
        }
    }
    (scope_parent_map, agent_scope_roles)
}

fn agent_uuids_from_events(
    events: &[&Event],
    scope_parent_map: &HashMap<Uuid, Uuid>,
    agent_scope_roles: &AgentScopeRoles,
) -> HashSet<Uuid> {
    events
        .iter()
        .copied()
        .filter(|event| should_include_agent_scope(event, scope_parent_map, agent_scope_roles))
        .map(Event::uuid)
        .collect()
}

fn should_include_agent_scope(
    event: &Event,
    scope_parent_map: &HashMap<Uuid, Uuid>,
    agent_scope_roles: &AgentScopeRoles,
) -> bool {
    if !is_start_event(event) || event.scope_type() != Some(crate::api::scope::ScopeType::Agent) {
        return false;
    }
    agent_scope_role(event) != Some("turn")
        || nearest_non_turn_agent_parent(
            event.parent_uuid(),
            scope_parent_map,
            agent_scope_roles,
            Some(event.uuid()),
        )
        .is_none()
}

fn agent_scope_nodes(
    events: &[&Event],
    scope_parent_map: &HashMap<Uuid, Uuid>,
    agent_uuids: &HashSet<Uuid>,
) -> HashMap<Uuid, AgentScopeNode> {
    events
        .iter()
        .copied()
        .filter(|event| is_included_agent_scope(event, agent_uuids))
        .map(|event| {
            let uuid = event.uuid();
            (
                uuid,
                AgentScopeNode {
                    uuid,
                    name: event.name().to_string(),
                    session_id: agent_session_id(event),
                    referenced_by_parent: is_subagent_reference_event(event),
                    parent_agent: nearest_agent_parent(
                        event.parent_uuid(),
                        scope_parent_map,
                        agent_uuids,
                        Some(uuid),
                    ),
                    children: Vec::new(),
                    start_timestamp: *event.timestamp(),
                },
            )
        })
        .collect()
}

fn is_included_agent_scope(event: &Event, agent_uuids: &HashSet<Uuid>) -> bool {
    is_start_event(event)
        && event.scope_type() == Some(crate::api::scope::ScopeType::Agent)
        && agent_uuids.contains(&event.uuid())
}

fn agent_session_id(event: &Event) -> Option<String> {
    event
        .metadata()
        .and_then(|metadata| metadata.get("session_id"))
        .and_then(Json::as_str)
        .map(ToString::to_string)
}

fn link_agent_children(nodes: &mut HashMap<Uuid, AgentScopeNode>) -> Vec<Uuid> {
    let mut child_links = Vec::new();
    let mut roots = Vec::new();
    for node in nodes.values() {
        if let Some(parent_agent) = node.parent_agent {
            child_links.push((parent_agent, node.uuid));
        } else {
            roots.push(node.uuid);
        }
    }
    for (parent_agent, child) in child_links {
        if let Some(parent) = nodes.get_mut(&parent_agent) {
            parent.children.push(child);
        }
    }
    roots
}

fn sort_agent_tree(roots: &mut [Uuid], nodes: &mut HashMap<Uuid, AgentScopeNode>) {
    let start_timestamps = nodes
        .iter()
        .map(|(uuid, node)| (*uuid, node.start_timestamp))
        .collect::<HashMap<_, _>>();
    roots.sort_by_key(|uuid| start_timestamps.get(uuid).copied());
    for node in nodes.values_mut() {
        node.children
            .sort_by_key(|uuid| start_timestamps.get(uuid).copied());
    }
}

fn agent_scope_role(event: &Event) -> Option<&str> {
    event
        .metadata()
        .and_then(|metadata| metadata.get("nemo_relay_scope_role"))
        .and_then(Json::as_str)
}

fn is_agent_scope_event(event: &Event) -> bool {
    event.scope_type() == Some(crate::api::scope::ScopeType::Agent)
}

fn is_subagent_reference_event(event: &Event) -> bool {
    agent_scope_role(event) == Some("subagent")
        || event.metadata().and_then(delegation_tool_call_id).is_some()
        || event.data().and_then(delegation_tool_call_id).is_some()
}

fn nearest_agent_parent(
    mut current: Option<Uuid>,
    scope_parent_map: &HashMap<Uuid, Uuid>,
    agent_uuids: &HashSet<Uuid>,
    excluded_uuid: Option<Uuid>,
) -> Option<Uuid> {
    while let Some(uuid) = current {
        if Some(uuid) != excluded_uuid && agent_uuids.contains(&uuid) {
            return Some(uuid);
        }
        current = scope_parent_map.get(&uuid).copied();
    }
    None
}

fn nearest_non_turn_agent_parent(
    mut current: Option<Uuid>,
    scope_parent_map: &HashMap<Uuid, Uuid>,
    agent_scope_roles: &HashMap<Uuid, Option<String>>,
    excluded_uuid: Option<Uuid>,
) -> Option<Uuid> {
    while let Some(uuid) = current {
        if Some(uuid) != excluded_uuid
            && let Some(role) = agent_scope_roles.get(&uuid)
            && role.as_deref() != Some("turn")
        {
            return Some(uuid);
        }
        current = scope_parent_map.get(&uuid).copied();
    }
    None
}

// ---------------------------------------------------------------------------
// Event-to-step mapping
// ---------------------------------------------------------------------------

fn events_to_trajectory(
    session_id: &str,
    agent_info: AtifAgentInfo,
    events: &[&Event],
) -> AtifTrajectory {
    let mut sorted: Vec<&Event> = events.to_vec();
    sorted.sort_by_key(|event| *event.timestamp());
    let tree = AgentScopeTree::from_events(&sorted);

    if let Some(root_uuid) = tree.choose_root(session_id)
        && can_use_agent_scope_tree(&tree, &sorted)
    {
        return agent_scope_to_trajectory(&tree, root_uuid, session_id, &agent_info, &sorted, true);
    }

    let steps = events_to_steps(&sorted);
    trajectory_from_parts(
        session_id.to_string(),
        Some(session_id.to_string()),
        agent_info,
        steps,
        None,
    )
}

fn can_use_agent_scope_tree(tree: &AgentScopeTree, events: &[&Event]) -> bool {
    events.iter().all(|event| {
        is_agent_scope_event(event) || !is_step_event(event) || tree.owner_agent(event).is_some()
    })
}

fn is_step_event(event: &Event) -> bool {
    matches!(
        (
            event.kind(),
            event.scope_category(),
            event.category().map(|category| category.as_str()),
        ),
        (
            "scope",
            Some(crate::api::event::ScopeCategory::Start),
            Some("llm")
        ) | (
            "scope",
            Some(crate::api::event::ScopeCategory::End),
            Some("llm")
        ) | (
            "scope",
            Some(crate::api::event::ScopeCategory::End),
            Some("tool")
        )
    )
}

fn agent_scope_to_trajectory(
    tree: &AgentScopeTree,
    agent_uuid: Uuid,
    session_id: &str,
    agent_info: &AtifAgentInfo,
    sorted_events: &[&Event],
    is_root: bool,
) -> AtifTrajectory {
    let mut steps = events_to_steps_for_agent(sorted_events, tree, agent_uuid);
    let subagent_trajectories = tree
        .nodes
        .get(&agent_uuid)
        .map(|node| {
            node.children
                .iter()
                .map(|child_uuid| {
                    agent_scope_to_trajectory(
                        tree,
                        *child_uuid,
                        session_id,
                        agent_info,
                        sorted_events,
                        false,
                    )
                })
                .filter(|trajectory| {
                    !trajectory.steps.is_empty()
                        || trajectory
                            .trajectory_id
                            .as_deref()
                            .and_then(|trajectory_id| Uuid::parse_str(trajectory_id).ok())
                            .and_then(|uuid| tree.nodes.get(&uuid))
                            .is_some_and(|child| !child.referenced_by_parent)
                })
                .collect::<Vec<_>>()
        })
        .filter(|children| !children.is_empty());
    let child_trajectory_ids = subagent_trajectories
        .as_deref()
        .unwrap_or_default()
        .iter()
        .filter_map(|trajectory| trajectory.trajectory_id.clone())
        .collect::<HashSet<_>>();
    prune_subagent_refs(&mut steps, &child_trajectory_ids);
    let trajectory_id = if is_root {
        session_id.to_string()
    } else {
        agent_uuid.to_string()
    };
    let trajectory_session_id = if is_root {
        session_id.to_string()
    } else {
        tree.nodes
            .get(&agent_uuid)
            .and_then(|node| node.session_id.clone())
            .unwrap_or_else(|| session_id.to_string())
    };

    trajectory_from_parts(
        trajectory_session_id,
        Some(trajectory_id),
        agent_info.clone(),
        steps,
        subagent_trajectories,
    )
}

fn trajectory_from_parts(
    session_id: String,
    trajectory_id: Option<String>,
    agent: AtifAgentInfo,
    steps: Vec<AtifStep>,
    subagent_trajectories: Option<Vec<AtifTrajectory>>,
) -> AtifTrajectory {
    let final_metrics = compute_final_metrics(&steps);

    AtifTrajectory {
        schema_version: ATIF_SCHEMA_VERSION.to_string(),
        session_id,
        trajectory_id,
        agent,
        steps,
        notes: None,
        final_metrics,
        continued_trajectory_ref: None,
        subagent_trajectories,
        extra: None,
    }
}

fn events_to_steps_for_agent(
    events: &[&Event],
    tree: &AgentScopeTree,
    agent_uuid: Uuid,
) -> Vec<AtifStep> {
    let lookups = EventLookupMaps::from_events_for_agent(events, tree, agent_uuid);
    let mut state = StepConversionState::default();

    for event in events {
        if let Some(child) = tree.direct_child_for_start(agent_uuid, event) {
            state.handle_subagent_start(child, event);
            continue;
        }

        if tree.owner_agent(event) != Some(agent_uuid) {
            continue;
        }

        state.handle_event(event, &lookups);
    }

    state.finish()
}

/// Converts a slice of events into ATIF steps.
///
/// Mapping logic:
/// 1. Sort events by timestamp.
/// 2. For each LLM pair:
///    - Start event → user step (message = extracted `messages` array from
///      unwrapped LlmRequest content, stripping `max_tokens`/`model`/etc.)
///    - End event → agent step (message = extracted content, metrics from
///      token_usage, tool_calls promoted to AtifToolCall entries with parsed
///      JSON arguments)
/// 3. For Tool events:
///    - Start events are **skipped** (tool_calls come from LLM End promotion)
///    - Consecutive End events are attached to the matching agent step
///      step with multiple results
/// 4. Tool End observation results are correlated with the preceding LLM End's
///    promoted tool_calls by function name → `source_call_id`.
/// 5. Mark and other Scope Start/End events → skipped.
fn events_to_steps(events: &[&Event]) -> Vec<AtifStep> {
    let mut sorted: Vec<&Event> = events.to_vec();
    sorted.sort_by_key(|e| *e.timestamp());
    let lookups = EventLookupMaps::from_events(&sorted);
    let mut state = StepConversionState::default();

    for event in &sorted {
        state.handle_event(event, &lookups);
    }

    state.finish()
}

fn is_start_event(event: &Event) -> bool {
    event.scope_category() == Some(crate::api::event::ScopeCategory::Start)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "../../tests/unit/atif_tests.rs"]
mod tests;
