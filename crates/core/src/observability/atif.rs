// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ATIF (Agent Trajectory Interchange Format) exporter.
//!
//! This module provides types and an exporter that collects lifecycle events
//! from the NeMo Flow runtime and converts them into ATIF trajectories conforming
//! to the ATIF v1.6 schema.
//!
//! # Overview
//!
//! The [`AtifExporter`] registers as an event subscriber, collects all events,
//! and can export them as an [`AtifTrajectory`] via [`AtifExporter::export`].
//!
//! # Event-to-Step Mapping
//!
//! The core conversion from NeMo Flow events to ATIF steps follows these rules:
//!
//! | NeMo Flow Event     | ATIF Step               | Notes                                |
//! |-----------------|-------------------------|--------------------------------------|
//! | LLM Start       | `user` step             | Messages extracted from LlmRequest   |
//! | LLM End         | `agent` step            | Response content, tool_calls promoted|
//! | Tool Start      | *(skipped)*             | tool_calls come from LLM End instead |
//! | Tool End        | `system` observation     | Consecutive tool ends merged         |
//! | Mark (with data)| `system` step           | Custom event data preserved          |
//! | Scope Start/End | *(skipped)*             | Structural events, not trajectory    |
//!
//! The exporter serializes the full collected event stream into a single ATIF
//! trajectory.

use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::api::event::Event;
use crate::api::runtime::EventSubscriberFn;
use crate::json::Json;

/// The ATIF schema version string embedded in all exported trajectories.
///
/// Currently `"ATIF-v1.6"`. This constant is used by [`AtifTrajectory`]
/// serialization and verified by downstream consumers to ensure compatibility.
pub const ATIF_SCHEMA_VERSION: &str = "ATIF-v1.6";

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

/// Aggregate statistics for the entire trajectory (ATIF v1.6 final_metrics).
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
    pub content: Json,
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
    /// Schema version (e.g., `"ATIF-v1.6"`).
    pub schema_version: String,
    /// Unique session identifier.
    pub session_id: String,
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

    /// Return an event subscriber function that records NeMo Flow events.
    ///
    /// The returned callback can be registered with
    /// [`register_subscriber`](crate::api::subscriber::register_subscriber).
    ///
    /// # Returns
    /// An [`EventSubscriberFn`] that appends each observed event to this
    /// exporter's internal buffer.
    pub fn subscriber(&self) -> EventSubscriberFn {
        let state = self.state.clone();
        Arc::new(move |event: &Event| {
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
    /// # Notes
    /// Exporting does not clear the buffered events. Call [`AtifExporter::clear`]
    /// when you need to reset the exporter between trajectories.
    pub fn export(&self) -> AtifTrajectory {
        let state = self.state.lock().unwrap();
        let collected_events: Vec<&Event> = state.events.iter().collect();
        let steps = events_to_steps(&collected_events);
        let final_metrics = compute_final_metrics(&steps);

        AtifTrajectory {
            schema_version: ATIF_SCHEMA_VERSION.to_string(),
            session_id: state.session_id.clone(),
            agent: state.agent_info.clone(),
            steps,
            notes: None,
            final_metrics,
            continued_trajectory_ref: None,
            extra: None,
        }
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
/// This avoids leaking the NeMo Flow transport wrapper into the trajectory.
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
/// Looks for a `"content"` field (string or structured) on the response object.
/// Falls back to the full response if the field is absent or not an object.
fn extract_llm_response_message(output: &Json) -> Json {
    if let Some(obj) = output.as_object() {
        if let Some(content) = non_null_object_field(obj, "content") {
            return content;
        }
        if let Some(summary) = llm_response_summary(obj) {
            return summary;
        }
    }
    // Not a recognized object structure — return as-is.
    output.clone()
}

fn non_null_object_field(obj: &serde_json::Map<String, Json>, key: &str) -> Option<Json> {
    obj.get(key).filter(|value| !value.is_null()).cloned()
}

fn llm_response_summary(obj: &serde_json::Map<String, Json>) -> Option<Json> {
    if !obj.contains_key("tool_calls") && !obj.contains_key("role") {
        return None;
    }

    let mut summary = serde_json::Map::new();
    if let Some(role) = obj.get("role") {
        summary.insert("role".to_string(), role.clone());
    }
    if let Some(tool_calls) = obj.get("tool_calls") {
        summary.insert("tool_calls".to_string(), tool_calls.clone());
    }
    if let Some(reasoning) = non_null_object_field(obj, "reasoning") {
        summary.insert("reasoning".to_string(), reasoning);
    }

    (!summary.is_empty()).then_some(Json::Object(summary))
}

/// Known keys in token_usage that we extract to dedicated fields.
const TOKEN_USAGE_KNOWN_KEYS: &[&str] = &[
    "prompt_tokens",
    "completion_tokens",
    "cached_tokens",
    "cost_usd",
    "prompt_token_ids",
    "completion_token_ids",
    "logprobs",
];

/// Try to extract `AtifMetrics` from a `token_usage` object in the LLM response.
///
/// Supports NeMo Flow `token_usage` and provider-native `usage` payloads.
/// Populates `extra` with any unknown usage keys (e.g. reasoning_tokens or total_tokens).
/// Returns `None` if the response has no recognized token counts.
fn extract_metrics(output: &Json) -> Option<AtifMetrics> {
    let usage = token_usage_object(output)?;
    let prompt = usage_u64(usage, &["prompt_tokens", "input_tokens"]);
    let completion = usage_u64(usage, &["completion_tokens", "output_tokens"]);
    let cached = usage_u64(usage, &["cached_tokens"])
        .or_else(|| prompt_tokens_detail_u64(usage, "cached_tokens"))
        .or_else(|| {
            sum_usage_u64(
                usage,
                &["cache_read_input_tokens", "cache_creation_input_tokens"],
            )
        });
    let cost = usage.get("cost_usd").and_then(Json::as_f64);
    let prompt_ids = usage
        .get("prompt_token_ids")
        .and_then(Json::as_array)
        .map(|a| a.iter().filter_map(Json::as_u64).collect());
    let completion_ids = usage
        .get("completion_token_ids")
        .and_then(Json::as_array)
        .map(|a| a.iter().filter_map(Json::as_u64).collect());
    let logprobs = usage
        .get("logprobs")
        .and_then(Json::as_array)
        .map(|a| a.iter().filter_map(Json::as_f64).collect());
    let known: std::collections::HashSet<&str> = TOKEN_USAGE_KNOWN_KEYS.iter().copied().collect();
    let extra_map: serde_json::Map<String, Json> = usage
        .iter()
        .filter(|(k, _)| !known.contains(k.as_str()))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let extra = if extra_map.is_empty() {
        None
    } else {
        Some(Json::Object(extra_map))
    };
    if prompt.is_none() && completion.is_none() && cached.is_none() {
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

fn token_usage_object(output: &Json) -> Option<&serde_json::Map<String, Json>> {
    let output = output.as_object()?;
    output
        .get("token_usage")
        .or_else(|| output.get("usage"))
        .and_then(Json::as_object)
}

fn usage_u64(usage: &serde_json::Map<String, Json>, keys: &[&str]) -> Option<u64> {
    keys.iter()
        .find_map(|key| usage.get(*key).and_then(Json::as_u64))
}

fn sum_usage_u64(usage: &serde_json::Map<String, Json>, keys: &[&str]) -> Option<u64> {
    let mut total = 0;
    let mut found = false;
    for key in keys {
        if let Some(value) = usage.get(*key).and_then(Json::as_u64) {
            total += value;
            found = true;
        }
    }
    found.then_some(total)
}

fn prompt_tokens_detail_u64(usage: &serde_json::Map<String, Json>, key: &str) -> Option<u64> {
    usage
        .get("prompt_tokens_details")
        .and_then(Json::as_object)
        .and_then(|details| details.get(key))
        .and_then(Json::as_u64)
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

/// Extract just the `messages` array from an LLM request payload.
///
/// LLM start inputs typically contain `{ "messages": [...], "model": "...",
/// "max_tokens": ..., "tools": [...], "stream": ... }`. For the user step we
/// only want the `messages` array — the rest is LLM configuration noise.
///
/// Returns the `messages` value if present, otherwise the full input.
fn extract_user_messages(input: &Json) -> Json {
    if let Some(obj) = input.as_object()
        && let Some(messages) = obj.get("messages")
    {
        return messages.clone();
    }
    input.clone()
}

/// Try to promote `tool_calls` from the raw LLM response into `AtifToolCall` entries.
///
/// Expected shape per OpenAI convention:
/// ```json
/// "tool_calls": [{ "id": "...", "type": "function", "function": { "name": "...", "arguments": "..." } }]
/// ```
///
/// String `arguments` are parsed into JSON for consistency with NeMo Flow tool events
/// which always provide parsed arguments.
///
/// Returns `None` if there are no tool calls or the structure is unrecognized.
fn extract_tool_calls(output: &Json) -> Option<Vec<AtifToolCall>> {
    let arr = output.as_object()?.get("tool_calls")?.as_array()?;
    if arr.is_empty() {
        return None;
    }
    let mut calls = Vec::with_capacity(arr.len());
    for tc in arr {
        let tc_obj = tc.as_object()?;
        let id = tc_obj
            .get("id")
            .and_then(Json::as_str)
            .unwrap_or("")
            .to_string();
        // The function details live under "function".
        let func = tc_obj.get("function").and_then(Json::as_object);
        let name = func
            .and_then(|f| f.get("name"))
            .and_then(Json::as_str)
            .unwrap_or("")
            .to_string();
        let raw_arguments = func
            .and_then(|f| f.get("arguments"))
            .cloned()
            .unwrap_or(Json::Null);
        // Parse string arguments as JSON for consistency.
        let arguments = if let Some(s) = raw_arguments.as_str() {
            serde_json::from_str(s).unwrap_or(raw_arguments)
        } else {
            raw_arguments
        };
        // Skip entries with no id and no name — they are not meaningful.
        if id.is_empty() && name.is_empty() {
            continue;
        }
        calls.push(AtifToolCall {
            tool_call_id: id,
            function_name: name,
            arguments,
        });
    }
    if calls.is_empty() { None } else { Some(calls) }
}

/// Compute aggregate `final_metrics` by summing token counts across all steps.
///
/// Always returns `Some(AtifFinalMetrics)` with `total_steps` set. Token/cost
/// fields are populated when at least one step carries metrics.
fn compute_final_metrics(steps: &[AtifStep]) -> Option<AtifFinalMetrics> {
    let mut total_prompt: u64 = 0;
    let mut total_completion: u64 = 0;
    let mut total_cached: u64 = 0;
    let mut total_cost: f64 = 0.0;
    let mut has_any = false;

    for step in steps {
        if let Some(m) = &step.metrics {
            has_any = true;
            total_prompt += m.prompt_tokens.unwrap_or(0);
            total_completion += m.completion_tokens.unwrap_or(0);
            total_cached += m.cached_tokens.unwrap_or(0);
            total_cost += m.cost_usd.unwrap_or(0.0);
        }
    }

    Some(AtifFinalMetrics {
        total_prompt_tokens: if has_any { Some(total_prompt) } else { None },
        total_completion_tokens: if has_any {
            Some(total_completion)
        } else {
            None
        },
        total_cached_tokens: if has_any && total_cached > 0 {
            Some(total_cached)
        } else {
            None
        },
        total_cost_usd: if has_any && total_cost > 0.0 {
            Some(total_cost)
        } else {
            None
        },
        total_steps: Some(steps.len() as u64),
        extra: None,
    })
}

// ---------------------------------------------------------------------------
// AtifStepExtra helpers
// ---------------------------------------------------------------------------

/// Build an [`AtifAncestry`] from a NeMo Flow [`Event`].
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

struct EventLookupMaps {
    name_map: std::collections::HashMap<Uuid, String>,
    start_ts_map: std::collections::HashMap<Uuid, DateTime<Utc>>,
}

impl EventLookupMaps {
    fn from_events(events: &[&Event]) -> Self {
        let mut name_map = std::collections::HashMap::new();
        let mut start_ts_map = std::collections::HashMap::new();
        for event in events {
            if is_start_event(event) {
                name_map.insert(event.uuid(), event.name().to_string());
                start_ts_map.insert(event.uuid(), *event.timestamp());
            }
        }
        Self {
            name_map,
            start_ts_map,
        }
    }
}

#[derive(Default)]
struct PendingAgentStep {
    step_idx: Option<usize>,
    ancestry: Option<AtifAncestry>,
    invocation: Option<AtifInvocationInfo>,
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
    ) {
        self.step_idx = Some(step_idx);
        self.ancestry = Some(ancestry);
        self.invocation = Some(invocation);
        self.tool_ancestry.clear();
        self.tool_invocations.clear();
        self.tool_call_order = tool_call_order;
    }

    fn push_tool_metadata(&mut self, ancestry: AtifAncestry, invocation: AtifInvocationInfo) {
        self.tool_ancestry.push(ancestry);
        self.tool_invocations.push(invocation);
    }

    fn has_active_step(&self) -> bool {
        self.step_idx.is_some()
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
    pending_observations: Vec<AtifObservationResult>,
    pending_obs_timestamp: Option<String>,
    current_reasoning_effort: Option<Json>,
    current_agent: PendingAgentStep,
}

impl StepConversionState {
    fn flush_observations(&mut self) {
        if self.pending_observations.is_empty() {
            return;
        }

        self.steps.push(AtifStep {
            step_id: 0,
            source: "system".to_string(),
            message: Json::Null,
            timestamp: self.pending_obs_timestamp.take(),
            model_name: None,
            reasoning_effort: None,
            reasoning_content: None,
            tool_calls: None,
            observation: Some(AtifObservation {
                results: std::mem::take(&mut self.pending_observations),
            }),
            metrics: None,
            is_copied_context: None,
            extra: None,
        });
    }

    fn finalize_agent_extra(&mut self) {
        self.current_agent.finalize_into(&mut self.steps);
    }

    fn handle_llm_start(&mut self, event: &Event, lookups: &EventLookupMaps) {
        self.flush_observations();
        self.finalize_agent_extra();

        let Some(input) = event.data() else {
            return;
        };
        let content = unwrap_llm_request(input);
        self.current_reasoning_effort = extract_reasoning_effort(&content);
        let extra = AtifStepExtra {
            ancestry: build_ancestry(event, &lookups.name_map),
            invocation: None,
            llm_request: Some(content.clone()),
            tool_ancestry: Vec::new(),
            tool_invocations: None,
        };
        self.steps.push(AtifStep {
            step_id: 0,
            source: "user".to_string(),
            message: extract_user_messages(&content),
            timestamp: Some(event.timestamp().to_rfc3339()),
            model_name: None,
            reasoning_effort: None,
            reasoning_content: None,
            tool_calls: None,
            observation: None,
            metrics: None,
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
        let ancestry = build_ancestry(event, &lookups.name_map);
        let invocation = build_invocation_info(
            start_ts,
            *event.timestamp(),
            Some(event.uuid().to_string()),
            "nemo_flow",
        );

        self.steps.push(AtifStep {
            step_id: 0,
            source: "agent".to_string(),
            message: extract_llm_response_message(output),
            timestamp: Some(event.timestamp().to_rfc3339()),
            model_name: event.model_name().map(ToOwned::to_owned),
            reasoning_effort,
            reasoning_content,
            tool_calls,
            observation: None,
            metrics: extract_metrics(output),
            is_copied_context: None,
            extra: None,
        });
        self.current_agent.set_current_agent(
            self.steps.len() - 1,
            ancestry,
            invocation,
            tool_call_order,
        );
    }

    fn handle_tool_end(&mut self, event: &Event, lookups: &EventLookupMaps) {
        if let Some(output) = event.data() {
            if self.pending_obs_timestamp.is_none() {
                self.pending_obs_timestamp = Some(event.timestamp().to_rfc3339());
            }
            self.pending_observations.push(AtifObservationResult {
                source_call_id: event
                    .tool_call_id()
                    .map(ToOwned::to_owned)
                    .or_else(|| self.last_tool_call_map.get(event.name()).cloned()),
                content: output.clone(),
            });
        }

        if !self.current_agent.has_active_step() {
            return;
        }
        let start_ts = lookups.start_ts_map.get(&event.uuid()).cloned();
        let invocation = build_invocation_info(
            start_ts,
            *event.timestamp(),
            event
                .tool_call_id()
                .map(ToOwned::to_owned)
                .or_else(|| Some(event.uuid().to_string())),
            "nemo_flow",
        );
        self.current_agent
            .push_tool_metadata(build_ancestry(event, &lookups.name_map), invocation);
    }

    fn handle_mark(&mut self, mark: &Event, lookups: &EventLookupMaps) {
        self.flush_observations();
        let Some(data) = mark.data() else {
            return;
        };
        if is_empty_mark_payload(data) {
            return;
        }
        let extra = AtifStepExtra {
            ancestry: build_ancestry(mark, &lookups.name_map),
            invocation: Some(AtifInvocationInfo {
                start_timestamp: None,
                end_timestamp: None,
                invocation_id: Some(mark.uuid().to_string()),
                status: Some("completed".to_string()),
                framework: Some("nemo_flow".to_string()),
            }),
            llm_request: None,
            tool_ancestry: Vec::new(),
            tool_invocations: None,
        };
        self.steps.push(AtifStep {
            step_id: 0,
            source: "system".to_string(),
            message: mark_message(mark, data),
            timestamp: Some(mark.timestamp().to_rfc3339()),
            model_name: None,
            reasoning_effort: None,
            reasoning_content: None,
            tool_calls: None,
            observation: None,
            metrics: None,
            is_copied_context: None,
            extra: serde_json::to_value(&extra).ok(),
        });
    }

    fn finish(mut self) -> Vec<AtifStep> {
        self.finalize_agent_extra();
        self.flush_observations();
        for (index, step) in self.steps.iter_mut().enumerate() {
            step.step_id = index + 1;
        }
        self.steps
    }
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

// ---------------------------------------------------------------------------
// Event-to-step mapping
// ---------------------------------------------------------------------------

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
///    - Consecutive End events are **merged** into a single system observation
///      step with multiple results
/// 4. Tool End observation results are correlated with the preceding LLM End's
///    promoted tool_calls by function name → `source_call_id`.
/// 5. Mark events → system steps if they carry data.
/// 6. Scope Start/End → skipped.
fn events_to_steps(events: &[&Event]) -> Vec<AtifStep> {
    let mut sorted: Vec<&Event> = events.to_vec();
    sorted.sort_by_key(|e| *e.timestamp());
    let lookups = EventLookupMaps::from_events(&sorted);
    let mut state = StepConversionState::default();

    for event in &sorted {
        match (
            event.kind(),
            event.scope_category(),
            event.category().map(|category| category.as_str()),
        ) {
            ("scope", Some(crate::api::event::ScopeCategory::Start), Some("llm")) => {
                state.handle_llm_start(event, &lookups)
            }
            ("scope", Some(crate::api::event::ScopeCategory::End), Some("llm")) => {
                state.handle_llm_end(event, &lookups)
            }
            ("scope", Some(crate::api::event::ScopeCategory::End), Some("tool")) => {
                state.handle_tool_end(event, &lookups)
            }
            ("mark", _, _) => state.handle_mark(event, &lookups),
            _ => {}
        }
    }

    state.finish()
}

fn is_empty_mark_payload(data: &Json) -> bool {
    data.is_null() || data.as_object().is_some_and(|object| object.is_empty())
}

// A runtime mark is point-in-time telemetry rather than a scoped call with start/end events. Agent
// hook adapters use marks for lifecycle notifications that do not map to first-class ATIF step
// types, for example hook-only status updates or synthetic fallback events. Preserve the original
// mark payload, but surface the hook name in a stable `hook_event_name` field so trajectory readers
// can label system steps without knowing adapter-specific metadata conventions.
fn mark_message(mark: &Event, data: &Json) -> Json {
    let Some(object) = data.as_object() else {
        return data.clone();
    };
    let mut message = object.clone();
    if !message.contains_key("hook_event_name")
        && let Some(hook_event_name) = mark_hook_event_name(mark)
    {
        message.insert("hook_event_name".to_string(), Json::String(hook_event_name));
    }
    Json::Object(message)
}

// Prefer the adapter-provided hook name because the runtime mark name may be a generic bucket such
// as `hook_mark` or a synthetic fallback like `subagent_end_without_start`. Falling back to the mark
// name keeps non-hook marks readable without making this exporter depend on any one agent adapter.
fn mark_hook_event_name(mark: &Event) -> Option<String> {
    mark.metadata()
        .and_then(Json::as_object)
        .and_then(|metadata| metadata.get("hook_event_name"))
        .and_then(Json::as_str)
        .filter(|name| !name.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| Some(mark.name().to_string()).filter(|name| !name.is_empty()))
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
