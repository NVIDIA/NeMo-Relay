// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ATIF (Agent Trajectory Interchange Format) exporter.
//!
//! This module provides types and an exporter that collects lifecycle events
//! from the Nexus runtime and converts them into ATIF trajectories conforming
//! to the ATIF v1.6 schema.
//!
//! # Overview
//!
//! The [`AtifExporter`] registers as an event subscriber, collects all events,
//! and can export them as an [`AtifTrajectory`] via [`AtifExporter::export`].
//!
//! # Event-to-Step Mapping
//!
//! The core conversion from Nexus events to ATIF steps follows these rules:
//!
//! | Nexus Event     | ATIF Step               | Notes                                |
//! |-----------------|-------------------------|--------------------------------------|
//! | LLM Start       | `user` step             | Messages extracted from LLMRequest   |
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

use crate::context::EventSubscriberFn;
use crate::json::Json;
use crate::types::Event;

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
    /// Creates a new exporter with the given session ID and agent info.
    pub fn new(session_id: String, agent_info: AtifAgentInfo) -> Self {
        Self {
            state: Arc::new(Mutex::new(AtifExporterState {
                session_id,
                agent_info,
                events: Vec::new(),
            })),
        }
    }

    /// Returns an event subscriber function that can be registered with
    /// [`nat_nexus_register_subscriber`](crate::api::nat_nexus_register_subscriber).
    pub fn subscriber(&self) -> EventSubscriberFn {
        let state = self.state.clone();
        Arc::new(move |event: &Event| {
            if let Ok(mut s) = state.lock() {
                s.events.push(event.clone());
            }
        })
    }

    /// Exports collected events as an [`AtifTrajectory`].
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

    /// Clears all collected events.
    pub fn clear(&self) {
        let mut state = self.state.lock().unwrap();
        state.events.clear();
    }
}

// ---------------------------------------------------------------------------
// Safe JSON extraction helpers
// ---------------------------------------------------------------------------

/// If `input` looks like an `LLMRequest` envelope (`{"content": ..., "headers": ...}`),
/// return the inner `content` value. Otherwise return the input unchanged.
///
/// This avoids leaking the Nexus transport wrapper into the trajectory.
fn unwrap_llm_request(input: &Json) -> Json {
    if let Some(obj) = input.as_object() {
        if obj.contains_key("content") && obj.contains_key("headers") {
            return obj.get("content").cloned().unwrap_or_else(|| input.clone());
        }
    }
    input.clone()
}

/// Extract the user-facing message content from a raw LLM response.
///
/// Looks for a `"content"` field (string or structured) on the response object.
/// Falls back to the full response if the field is absent or not an object.
fn extract_llm_response_message(output: &Json) -> Json {
    if let Some(obj) = output.as_object() {
        // Prefer the "content" field if it carries actual content.
        if let Some(content) = obj.get("content") {
            if !content.is_null() {
                return content.clone();
            }
        }
        // If content is null (e.g. tool_calls-only response), fall back to the
        // role + tool_calls summary so the step is still meaningful.
        if obj.contains_key("tool_calls") || obj.contains_key("role") {
            let mut summary = serde_json::Map::new();
            if let Some(role) = obj.get("role") {
                summary.insert("role".to_string(), role.clone());
            }
            if let Some(tc) = obj.get("tool_calls") {
                summary.insert("tool_calls".to_string(), tc.clone());
            }
            if let Some(r) = obj.get("reasoning") {
                if !r.is_null() {
                    summary.insert("reasoning".to_string(), r.clone());
                }
            }
            if !summary.is_empty() {
                return Json::Object(summary);
            }
        }
    }
    // Not a recognized object structure — return as-is.
    output.clone()
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
/// Populates `extra` with any unknown token_usage keys (e.g. reasoning_tokens).
/// Returns `None` if the response has no `token_usage` or it is not an object.
fn extract_metrics(output: &Json) -> Option<AtifMetrics> {
    let usage = output.as_object()?.get("token_usage")?.as_object()?;
    let prompt = usage.get("prompt_tokens").and_then(Json::as_u64);
    let completion = usage.get("completion_tokens").and_then(Json::as_u64);
    let cached = usage.get("cached_tokens").and_then(Json::as_u64);
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

/// Extract `reasoning_effort` from an LLM request (string or number).
///
/// The request content may have `reasoning_effort` (e.g. `"high"`, `"medium"`,
/// or a numeric value). Returns the value as Json for flexibility.
fn extract_reasoning_effort(input: &Json) -> Option<Json> {
    if let Some(obj) = input.as_object() {
        if let Some(v) = obj.get("reasoning_effort") {
            if !v.is_null() {
                return Some(v.clone());
            }
        }
    }
    None
}

/// Extract `reasoning` (reasoning_content) from an LLM response output.
///
/// The agent's explicit internal reasoning may appear in the response under the
/// `"reasoning"` key. Returns `None` if absent or not a string.
fn extract_reasoning_content(output: &Json) -> Option<String> {
    if let Some(obj) = output.as_object() {
        if let Some(r) = obj.get("reasoning") {
            return r.as_str().map(String::from);
        }
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
    if let Some(obj) = input.as_object() {
        if let Some(messages) = obj.get("messages") {
            return messages.clone();
        }
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
/// String `arguments` are parsed into JSON for consistency with Nexus tool events
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
    if calls.is_empty() {
        None
    } else {
        Some(calls)
    }
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

/// Build an [`AtifAncestry`] from a Nexus [`Event`].
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

// ---------------------------------------------------------------------------
// Event-to-step mapping
// ---------------------------------------------------------------------------

/// Converts a slice of events into ATIF steps.
///
/// Mapping logic:
/// 1. Sort events by timestamp.
/// 2. For each LLM pair:
///    - Start event → user step (message = extracted `messages` array from
///      unwrapped LLMRequest content, stripping `max_tokens`/`model`/etc.)
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

    // Pre-pass: build uuid → name and uuid → start_timestamp maps so that
    // build_ancestry can resolve parent_name and build_invocation_info can
    // emit paired start/end timestamps on End events.
    let mut name_map: std::collections::HashMap<Uuid, String> = std::collections::HashMap::new();
    let mut start_ts_map: std::collections::HashMap<Uuid, DateTime<Utc>> =
        std::collections::HashMap::new();
    for event in &sorted {
        if is_start_event(event) {
            name_map.insert(event.uuid(), event.name().to_string());
            start_ts_map.insert(event.uuid(), *event.timestamp());
        }
    }

    let mut steps = Vec::new();
    let mut last_tool_call_map: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    let mut pending_observations: Vec<AtifObservationResult> = Vec::new();
    let mut pending_obs_timestamp: Option<String> = None;
    let mut current_reasoning_effort: Option<Json> = None;
    // Deferred extra state for the current agent step. Written once when the
    // next LLM Start arrives (or at end of loop), after all Tool End events
    // for this turn have been accumulated.
    let mut current_agent_step_idx: Option<usize> = None;
    let mut current_agent_ancestry: Option<AtifAncestry> = None;
    let mut current_agent_invocation: Option<AtifInvocationInfo> = None;
    let mut pending_tool_ancestry: Vec<AtifAncestry> = Vec::new();
    let mut pending_tool_invocations: Vec<AtifInvocationInfo> = Vec::new();
    // Declaration order of tool_call_ids from the most recent LLM End event.
    // Used to sort pending_tool_ancestry/invocations to match tool_calls[i].
    let mut last_tool_call_order: Vec<String> = Vec::new();

    // Write the accumulated AtifStepExtra to the current agent step. Called
    // when a new LLM Start arrives (closing the previous turn) and after the
    // event loop ends.
    let finalize_agent_extra = |steps: &mut Vec<AtifStep>,
                                idx: &mut Option<usize>,
                                ancestry: &mut Option<AtifAncestry>,
                                invocation: &mut Option<AtifInvocationInfo>,
                                tool_ancestry: &mut Vec<AtifAncestry>,
                                tool_invocations: &mut Vec<AtifInvocationInfo>,
                                tool_call_order: &[String]| {
        if let (Some(i), Some(anc)) = (idx.take(), ancestry.take()) {
            if let Some(step) = steps.get_mut(i) {
                // Sort ancestry/invocations to match tool_calls declaration
                // order. Tools may complete out of order (concurrent execution)
                // but tool_ancestry[i] must align with tool_calls[i] by spec.
                if !tool_call_order.is_empty() && !tool_ancestry.is_empty() {
                    let mut pairs: Vec<(AtifAncestry, AtifInvocationInfo)> =
                        std::mem::take(tool_ancestry)
                            .into_iter()
                            .zip(std::mem::take(tool_invocations))
                            .collect();
                    pairs.sort_by_key(|(_, inv)| {
                        inv.invocation_id
                            .as_deref()
                            .and_then(|id| tool_call_order.iter().position(|o| o == id))
                            .unwrap_or(usize::MAX)
                    });
                    let (sorted_anc, sorted_inv): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
                    *tool_ancestry = sorted_anc;
                    *tool_invocations = sorted_inv;
                }
                let extra = AtifStepExtra {
                    ancestry: anc,
                    invocation: invocation.take(),
                    tool_ancestry: std::mem::take(tool_ancestry),
                    tool_invocations: if tool_invocations.is_empty() {
                        None
                    } else {
                        Some(std::mem::take(tool_invocations))
                    },
                };
                step.extra = serde_json::to_value(&extra).ok();
            }
        }
    };

    let flush_observations = |steps: &mut Vec<AtifStep>,
                              obs: &mut Vec<AtifObservationResult>,
                              ts: &mut Option<String>| {
        if obs.is_empty() {
            return;
        }
        steps.push(AtifStep {
            step_id: 0,
            source: "system".to_string(),
            message: Json::Null,
            timestamp: ts.take(),
            model_name: None,
            reasoning_effort: None,
            reasoning_content: None,
            tool_calls: None,
            observation: Some(AtifObservation {
                results: std::mem::take(obs),
            }),
            metrics: None,
            is_copied_context: None,
            extra: None,
        });
    };

    for event in &sorted {
        match event {
            Event::LLMStart(llm_start) => {
                flush_observations(
                    &mut steps,
                    &mut pending_observations,
                    &mut pending_obs_timestamp,
                );
                // Finalize the previous agent step now that all its
                // Tool End events have been seen.
                // Sort ancestry/invocations to match tool_calls declaration order.
                finalize_agent_extra(
                    &mut steps,
                    &mut current_agent_step_idx,
                    &mut current_agent_ancestry,
                    &mut current_agent_invocation,
                    &mut pending_tool_ancestry,
                    &mut pending_tool_invocations,
                    &last_tool_call_order,
                );
                if let Some(input) = &llm_start.input {
                    let content = unwrap_llm_request(input);
                    current_reasoning_effort = extract_reasoning_effort(&content);
                    let ancestry = build_ancestry(event, &name_map);
                    let extra = AtifStepExtra {
                        ancestry,
                        invocation: None, // user step: end time unknown
                        tool_ancestry: Vec::new(),
                        tool_invocations: None,
                    };
                    steps.push(AtifStep {
                        step_id: 0,
                        source: "user".to_string(),
                        message: extract_user_messages(&content),
                        timestamp: Some(llm_start.timestamp.to_rfc3339()),
                        model_name: llm_start.model_name.clone(),
                        reasoning_effort: None,
                        reasoning_content: None,
                        tool_calls: None,
                        observation: None,
                        metrics: None,
                        is_copied_context: None,
                        extra: serde_json::to_value(&extra).ok(),
                    });
                }
            }
            Event::ToolStart(_) | Event::ScopeStart(_) => {
                // Tool and scope start events do not become ATIF steps.
            }
            Event::LLMEnd(llm_end) => {
                flush_observations(
                    &mut steps,
                    &mut pending_observations,
                    &mut pending_obs_timestamp,
                );
                if let Some(output) = &llm_end.output {
                    let tool_calls = extract_tool_calls(output);
                    last_tool_call_map.clear();
                    last_tool_call_order.clear();
                    if let Some(ref tcs) = tool_calls {
                        for tc in tcs {
                            if !tc.function_name.is_empty() {
                                last_tool_call_map
                                    .insert(tc.function_name.clone(), tc.tool_call_id.clone());
                            }
                            last_tool_call_order.push(tc.tool_call_id.clone());
                        }
                    }
                    let reasoning_effort = current_reasoning_effort.take();
                    let reasoning_content = extract_reasoning_content(output);
                    let start_ts = start_ts_map.get(&llm_end.uuid).cloned();
                    // Save ancestry and invocation for deferred write —
                    // tool_ancestry is not yet known at this point.
                    current_agent_ancestry = Some(build_ancestry(event, &name_map));
                    current_agent_invocation = Some(build_invocation_info(
                        start_ts,
                        llm_end.timestamp,
                        Some(llm_end.uuid.to_string()),
                        "nexus",
                    ));
                    steps.push(AtifStep {
                        step_id: 0,
                        source: "agent".to_string(),
                        message: extract_llm_response_message(output),
                        timestamp: Some(llm_end.timestamp.to_rfc3339()),
                        model_name: llm_end.model_name.clone(),
                        reasoning_effort,
                        reasoning_content,
                        tool_calls,
                        observation: None,
                        metrics: extract_metrics(output),
                        is_copied_context: None,
                        extra: None,
                    });
                    current_agent_step_idx = Some(steps.len() - 1);
                    pending_tool_ancestry.clear();
                    pending_tool_invocations.clear();
                }
            }
            Event::ToolEnd(tool_end) => {
                // Tool end -> buffer as observation result for merging,
                // and append ancestry/invocation to the current agent step.
                if let Some(output) = &tool_end.output {
                    // Correlate: prefer event's own tool_call_id, then
                    // look up by function_name in the last LLM End's promoted calls.
                    let source_call_id = tool_end
                        .tool_call_id
                        .clone()
                        .or_else(|| last_tool_call_map.get(tool_end.name.as_str()).cloned());
                    if pending_obs_timestamp.is_none() {
                        pending_obs_timestamp = Some(tool_end.timestamp.to_rfc3339());
                    }
                    pending_observations.push(AtifObservationResult {
                        source_call_id: source_call_id.clone(),
                        content: output.clone(),
                    });
                }
                // Accumulate tool ancestry/invocation for the current
                // agent step. Written to step.extra when finalized.
                if current_agent_step_idx.is_some() {
                    let start_ts = start_ts_map.get(&tool_end.uuid).cloned();
                    pending_tool_ancestry.push(build_ancestry(event, &name_map));
                    pending_tool_invocations.push(build_invocation_info(
                        start_ts,
                        tool_end.timestamp,
                        tool_end
                            .tool_call_id
                            .clone()
                            .or_else(|| Some(tool_end.uuid.to_string())),
                        "nexus",
                    ));
                }
            }
            Event::ScopeEnd(_) => {
                // Scope end events do not become ATIF steps.
            }
            Event::Mark(mark) => {
                flush_observations(
                    &mut steps,
                    &mut pending_observations,
                    &mut pending_obs_timestamp,
                );
                if let Some(data) = &mark.data {
                    steps.push(AtifStep {
                        step_id: 0,
                        source: "system".to_string(),
                        message: data.clone(),
                        timestamp: Some(mark.timestamp.to_rfc3339()),
                        model_name: None,
                        reasoning_effort: None,
                        reasoning_content: None,
                        tool_calls: None,
                        observation: None,
                        metrics: None,
                        is_copied_context: None,
                        extra: None,
                    });
                }
            }
        }
    }

    // Finalize the last agent step's extra (no subsequent LLM Start to trigger it).
    finalize_agent_extra(
        &mut steps,
        &mut current_agent_step_idx,
        &mut current_agent_ancestry,
        &mut current_agent_invocation,
        &mut pending_tool_ancestry,
        &mut pending_tool_invocations,
        &last_tool_call_order,
    );

    // Flush any remaining observations
    flush_observations(
        &mut steps,
        &mut pending_observations,
        &mut pending_obs_timestamp,
    );

    // Assign 1-based step IDs
    for (i, step) in steps.iter_mut().enumerate() {
        step.step_id = i + 1;
    }

    steps
}

fn is_start_event(event: &Event) -> bool {
    matches!(
        event,
        Event::ScopeStart(_) | Event::ToolStart(_) | Event::LLMStart(_)
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;
    use serde_json::json;

    #[derive(Debug, Clone, Copy)]
    enum EventType {
        Start,
        End,
        Mark,
    }

    struct TestEventBuilder {
        uuid: Uuid,
        event_type: EventType,
        parent_uuid: Option<Uuid>,
        name: String,
        data: Option<serde_json::Value>,
        metadata: Option<serde_json::Value>,
        attributes: Option<HandleAttributes>,
        scope_type: Option<ScopeType>,
        input: Option<serde_json::Value>,
        output: Option<serde_json::Value>,
        model_name: Option<String>,
        tool_call_id: Option<String>,
    }

    impl TestEventBuilder {
        fn name(mut self, name: impl Into<String>) -> Self {
            self.name = name.into();
            self
        }

        fn parent_uuid(mut self, parent_uuid: Option<Uuid>) -> Self {
            self.parent_uuid = parent_uuid;
            self
        }

        fn data(mut self, data: Option<serde_json::Value>) -> Self {
            self.data = data;
            self
        }

        fn scope_type(mut self, scope_type: ScopeType) -> Self {
            self.scope_type = Some(scope_type);
            self
        }

        fn input(mut self, input: Option<serde_json::Value>) -> Self {
            self.input = input;
            self
        }

        fn output(mut self, output: Option<serde_json::Value>) -> Self {
            self.output = output;
            self
        }

        fn model_name(mut self, model_name: Option<String>) -> Self {
            self.model_name = model_name;
            self
        }

        fn tool_call_id(mut self, tool_call_id: Option<String>) -> Self {
            self.tool_call_id = tool_call_id;
            self
        }

        fn build(self) -> Event {
            match (self.event_type, self.scope_type) {
                (EventType::Mark, _) => Event::mark(
                    self.parent_uuid,
                    self.uuid,
                    self.name,
                    self.data,
                    self.metadata,
                ),
                (EventType::Start, Some(ScopeType::Tool)) => Event::tool_start(
                    self.parent_uuid,
                    self.uuid,
                    self.name,
                    self.data,
                    self.metadata,
                    match self.attributes {
                        Some(HandleAttributes::Tool(attributes)) => attributes,
                        _ => ToolAttributes::empty(),
                    },
                    self.input,
                    self.tool_call_id,
                ),
                (EventType::End, Some(ScopeType::Tool)) => Event::tool_end(
                    self.parent_uuid,
                    self.uuid,
                    self.name,
                    self.data,
                    self.metadata,
                    match self.attributes {
                        Some(HandleAttributes::Tool(attributes)) => attributes,
                        _ => ToolAttributes::empty(),
                    },
                    self.output,
                    self.tool_call_id,
                ),
                (EventType::Start, Some(ScopeType::Llm)) => Event::llm_start(
                    self.parent_uuid,
                    self.uuid,
                    self.name,
                    self.data,
                    self.metadata,
                    match self.attributes {
                        Some(HandleAttributes::Llm(attributes)) => attributes,
                        _ => LLMAttributes::empty(),
                    },
                    self.input,
                    self.model_name,
                ),
                (EventType::End, Some(ScopeType::Llm)) => Event::llm_end(
                    self.parent_uuid,
                    self.uuid,
                    self.name,
                    self.data,
                    self.metadata,
                    match self.attributes {
                        Some(HandleAttributes::Llm(attributes)) => attributes,
                        _ => LLMAttributes::empty(),
                    },
                    self.output,
                    self.model_name,
                ),
                (EventType::Start, Some(scope_type)) => Event::scope_start(
                    self.parent_uuid,
                    self.uuid,
                    self.name,
                    self.data,
                    self.metadata,
                    match self.attributes {
                        Some(HandleAttributes::Scope(attributes)) => attributes,
                        _ => ScopeAttributes::empty(),
                    },
                    scope_type,
                ),
                (EventType::End, Some(scope_type)) => Event::scope_end(
                    self.parent_uuid,
                    self.uuid,
                    self.name,
                    self.data,
                    self.metadata,
                    match self.attributes {
                        Some(HandleAttributes::Scope(attributes)) => attributes,
                        _ => ScopeAttributes::empty(),
                    },
                    scope_type,
                ),
                (event_type, None) => panic!("missing scope_type for {event_type:?} event"),
            }
        }
    }

    fn event_builder(uuid: Uuid, event_type: EventType) -> TestEventBuilder {
        TestEventBuilder {
            uuid,
            event_type,
            parent_uuid: None,
            name: String::new(),
            data: None,
            metadata: None,
            attributes: None,
            scope_type: None,
            input: None,
            output: None,
            model_name: None,
            tool_call_id: None,
        }
    }

    fn set_event_timestamp(event: &mut Event, timestamp: chrono::DateTime<chrono::Utc>) {
        match event {
            Event::ScopeStart(inner) => inner.timestamp = timestamp,
            Event::ScopeEnd(inner) => inner.timestamp = timestamp,
            Event::ToolStart(inner) => inner.timestamp = timestamp,
            Event::ToolEnd(inner) => inner.timestamp = timestamp,
            Event::LLMStart(inner) => inner.timestamp = timestamp,
            Event::LLMEnd(inner) => inner.timestamp = timestamp,
            Event::Mark(inner) => inner.timestamp = timestamp,
        }
    }

    fn make_agent_info() -> AtifAgentInfo {
        AtifAgentInfo {
            name: "test-agent".to_string(),
            version: "1.0.0".to_string(),
            model_name: None,
            tool_definitions: None,
            extra: None,
        }
    }

    #[test]
    fn test_exporter_empty() {
        let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
        let trajectory = exporter.export();

        assert_eq!(trajectory.schema_version, ATIF_SCHEMA_VERSION);
        assert_eq!(trajectory.session_id, "session-1");
        assert_eq!(trajectory.agent.name, "test-agent");
        assert!(trajectory.steps.is_empty());
        // final_metrics is always Some now — carries total_steps even for empty trajectories
        let fm = trajectory.final_metrics.as_ref().unwrap();
        assert_eq!(fm.total_steps, Some(0));
        assert!(fm.total_prompt_tokens.is_none());
    }

    #[test]
    fn test_exporter_schema_version() {
        assert_eq!(ATIF_SCHEMA_VERSION, "ATIF-v1.6");
    }

    #[test]
    fn test_exporter_tool_lifecycle() {
        let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
        let tool_uuid = Uuid::new_v4();

        // Simulate tool start (should be SKIPPED — tool_calls come from LLM End)
        let start = event_builder(tool_uuid, EventType::Start)
            .name("web_search")
            .scope_type(ScopeType::Tool)
            .input(Some(json!({"query": "test"})))
            .tool_call_id(Some("call_123".to_string()))
            .build();

        // Simulate tool end
        let end = event_builder(tool_uuid, EventType::End)
            .name("web_search")
            .scope_type(ScopeType::Tool)
            .output(Some(json!({"results": ["result1"]})))
            .tool_call_id(Some("call_123".to_string()))
            .build();

        {
            let mut state = exporter.state.lock().unwrap();
            state.events.push(start);
            state.events.push(end);
        }

        let trajectory = exporter.export();
        // Tool Start is skipped, only the observation step remains
        assert_eq!(trajectory.steps.len(), 1);

        let step1 = &trajectory.steps[0];
        assert_eq!(step1.step_id, 1);
        assert_eq!(step1.source, "system");
        let obs = step1.observation.as_ref().unwrap();
        assert_eq!(obs.results.len(), 1);
        assert_eq!(obs.results[0].source_call_id, Some("call_123".to_string()));
        assert_eq!(obs.results[0].content, json!({"results": ["result1"]}));
    }

    #[test]
    fn test_exporter_llm_lifecycle() {
        let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
        let llm_uuid = Uuid::new_v4();

        // Input wrapped in LLMRequest envelope — should be unwrapped.
        let start = event_builder(llm_uuid, EventType::Start)
            .name("gpt-4")
            .scope_type(ScopeType::Llm)
            .input(Some(json!({
                "content": {"messages": [{"role": "user", "content": "hello"}]},
                "headers": {}
            })))
            .model_name(Some("gpt-4".to_string()))
            .build();

        // Output with content, token_usage, and tool_calls.
        let end = event_builder(llm_uuid, EventType::End)
            .name("gpt-4")
            .scope_type(ScopeType::Llm)
            .output(Some(json!({
                "content": "Hi there!",
                "role": "assistant",
                "token_usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30
                },
                "tool_calls": []
            })))
            .model_name(Some("gpt-4".to_string()))
            .build();

        {
            let mut state = exporter.state.lock().unwrap();
            state.events.push(start);
            state.events.push(end);
        }

        let trajectory = exporter.export();
        assert_eq!(trajectory.steps.len(), 2);

        // First step: user (LLM start — unwrapped LLMRequest, then messages extracted)
        let step1 = &trajectory.steps[0];
        assert_eq!(step1.step_id, 1);
        assert_eq!(step1.source, "user");
        // extract_user_messages pulls out just the messages array
        assert_eq!(step1.message, json!([{"role": "user", "content": "hello"}]));
        assert_eq!(step1.model_name, Some("gpt-4".to_string()));

        // Second step: agent (LLM end with extracted content + metrics)
        let step2 = &trajectory.steps[1];
        assert_eq!(step2.step_id, 2);
        assert_eq!(step2.source, "agent");
        assert_eq!(step2.message, json!("Hi there!"));
        assert_eq!(step2.model_name, Some("gpt-4".to_string()));
        // Metrics extracted from token_usage
        let metrics = step2.metrics.as_ref().unwrap();
        assert_eq!(metrics.prompt_tokens, Some(10));
        assert_eq!(metrics.completion_tokens, Some(20));
        // Empty tool_calls should not produce AtifToolCall entries
        assert!(step2.tool_calls.is_none());

        // final_metrics should aggregate using total_ prefixed fields (AtifFinalMetrics)
        let fm = trajectory.final_metrics.as_ref().unwrap();
        assert_eq!(fm.total_prompt_tokens, Some(10));
        assert_eq!(fm.total_completion_tokens, Some(20));
        assert_eq!(fm.total_steps, Some(2));
    }

    #[test]
    fn test_exporter_llm_lifecycle_plain_input() {
        // Input without LLMRequest envelope — passed through unchanged.
        let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
        let llm_uuid = Uuid::new_v4();

        let start = event_builder(llm_uuid, EventType::Start)
            .name("gpt-4")
            .scope_type(ScopeType::Llm)
            .input(Some(
                json!({"messages": [{"role": "user", "content": "hello"}]}),
            ))
            .model_name(Some("gpt-4".to_string()))
            .build();

        let end = event_builder(llm_uuid, EventType::End)
            .name("gpt-4")
            .scope_type(ScopeType::Llm)
            .output(Some(json!("simple string response")))
            .model_name(Some("gpt-4".to_string()))
            .build();

        {
            let mut state = exporter.state.lock().unwrap();
            state.events.push(start);
            state.events.push(end);
        }

        let trajectory = exporter.export();
        assert_eq!(trajectory.steps.len(), 2);

        // Input without headers key — messages array is still extracted
        assert_eq!(
            trajectory.steps[0].message,
            json!([{"role": "user", "content": "hello"}])
        );
        // Non-object output is passed through as-is
        assert_eq!(trajectory.steps[1].message, json!("simple string response"));
        assert!(trajectory.steps[1].metrics.is_none());
        // No token metrics on any step — token totals are None, but total_steps is still set
        let fm = trajectory.final_metrics.as_ref().unwrap();
        assert!(fm.total_prompt_tokens.is_none());
        assert_eq!(fm.total_steps, Some(2));
    }

    #[test]
    fn test_exporter_llm_tool_calls_promoted() {
        let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
        let llm_uuid = Uuid::new_v4();

        let end = event_builder(llm_uuid, EventType::End)
            .name("gpt-4")
            .scope_type(ScopeType::Llm)
            .output(Some(json!({
                "content": null,
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": "{\"q\": \"test\"}"
                        }
                    }
                ]
            })))
            .build();

        {
            let mut state = exporter.state.lock().unwrap();
            state.events.push(end);
        }

        let trajectory = exporter.export();
        assert_eq!(trajectory.steps.len(), 1);
        let step = &trajectory.steps[0];

        // tool_calls promoted from response body, string arguments parsed as JSON
        let tc = step.tool_calls.as_ref().unwrap();
        assert_eq!(tc.len(), 1);
        assert_eq!(tc[0].tool_call_id, "call_abc");
        assert_eq!(tc[0].function_name, "search");
        assert_eq!(tc[0].arguments, json!({"q": "test"}));

        // message should be a summary (content was null)
        assert_eq!(
            step.message,
            json!({"role": "assistant", "tool_calls": [{"id": "call_abc", "type": "function", "function": {"name": "search", "arguments": "{\"q\": \"test\"}"}}]})
        );
    }

    #[test]
    fn test_exporter_full_pipeline() {
        let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
        let scope_uuid = Uuid::new_v4();
        let llm_uuid = Uuid::new_v4();
        let tool_uuid = Uuid::new_v4();

        // Scope start (should be skipped)
        let scope_start = event_builder(scope_uuid, EventType::Start)
            .name("agent")
            .scope_type(ScopeType::Agent)
            .build();

        // LLM start/end
        let llm_start = event_builder(llm_uuid, EventType::Start)
            .scope_type(ScopeType::Llm)
            .input(Some(json!({"prompt": "What is 2+2?"})))
            .build();
        let llm_end = event_builder(llm_uuid, EventType::End)
            .scope_type(ScopeType::Llm)
            .output(Some(json!({"answer": "4"})))
            .build();

        // Tool start/end
        let tool_start = event_builder(tool_uuid, EventType::Start)
            .name("calculator")
            .scope_type(ScopeType::Tool)
            .input(Some(json!({"expr": "2+2"})))
            .tool_call_id(Some("call_1".to_string()))
            .build();
        let tool_end = event_builder(tool_uuid, EventType::End)
            .name("calculator")
            .scope_type(ScopeType::Tool)
            .output(Some(json!(4)))
            .tool_call_id(Some("call_1".to_string()))
            .build();

        // Scope end (should be skipped)
        let scope_end = event_builder(scope_uuid, EventType::End)
            .name("agent")
            .scope_type(ScopeType::Agent)
            .build();

        {
            let mut state = exporter.state.lock().unwrap();
            state.events.push(scope_start);
            state.events.push(llm_start);
            state.events.push(llm_end);
            state.events.push(tool_start);
            state.events.push(tool_end);
            state.events.push(scope_end);
        }

        let trajectory = exporter.export();
        // Scope events and Tool Start are skipped: user, agent, system(obs)
        assert_eq!(trajectory.steps.len(), 3);

        assert_eq!(trajectory.steps[0].source, "user");
        assert_eq!(trajectory.steps[1].source, "agent");
        assert_eq!(trajectory.steps[2].source, "system");
        assert!(trajectory.steps[2].observation.is_some());

        // Step IDs are 1-based
        for (i, step) in trajectory.steps.iter().enumerate() {
            assert_eq!(step.step_id, i + 1);
        }
    }

    #[test]
    fn test_exporter_tool_call_id_linking() {
        // Tool Start is skipped; the tool_call_id comes from the event's own field.
        let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
        let tool_uuid = Uuid::new_v4();

        let start = event_builder(tool_uuid, EventType::Start)
            .name("my_tool")
            .scope_type(ScopeType::Tool)
            .input(Some(json!({"x": 1})))
            .tool_call_id(Some("call_abc".to_string()))
            .build();

        let end = event_builder(tool_uuid, EventType::End)
            .name("my_tool")
            .scope_type(ScopeType::Tool)
            .output(Some(json!({"y": 2})))
            .tool_call_id(Some("call_abc".to_string()))
            .build();

        {
            let mut state = exporter.state.lock().unwrap();
            state.events.push(start);
            state.events.push(end);
        }

        let trajectory = exporter.export();
        // Only observation step (Tool Start is skipped)
        assert_eq!(trajectory.steps.len(), 1);
        let obs_result = &trajectory.steps[0].observation.as_ref().unwrap().results[0];
        assert_eq!(obs_result.source_call_id, Some("call_abc".to_string()));
    }

    #[test]
    fn test_trajectory_serde_roundtrip() {
        let trajectory = AtifTrajectory {
            schema_version: ATIF_SCHEMA_VERSION.to_string(),
            session_id: "test-session".to_string(),
            agent: AtifAgentInfo {
                name: "test".to_string(),
                version: "1.0".to_string(),
                model_name: Some("gpt-4".to_string()),
                tool_definitions: Some(vec![json!({"name": "search"})]),
                extra: None,
            },
            steps: vec![AtifStep {
                step_id: 1,
                source: "user".to_string(),
                message: json!("Hello"),
                timestamp: Some("2026-01-01T00:00:00Z".to_string()),
                model_name: None,
                reasoning_effort: None,
                reasoning_content: None,
                tool_calls: None,
                observation: None,
                metrics: Some(AtifMetrics {
                    prompt_tokens: Some(10),
                    completion_tokens: Some(20),
                    cached_tokens: None,
                    cost_usd: Some(0.001),
                    prompt_token_ids: None,
                    completion_token_ids: None,
                    logprobs: None,
                    extra: None,
                }),
                is_copied_context: None,
                extra: None,
            }],
            notes: None,
            final_metrics: Some(AtifFinalMetrics {
                total_prompt_tokens: Some(100),
                total_completion_tokens: Some(200),
                total_cached_tokens: Some(50),
                total_cost_usd: Some(0.01),
                total_steps: Some(1),
                extra: None,
            }),
            continued_trajectory_ref: None,
            extra: None,
        };

        let json_str = serde_json::to_string(&trajectory).unwrap();
        let deserialized: AtifTrajectory = serde_json::from_str(&json_str).unwrap();

        assert_eq!(deserialized.schema_version, ATIF_SCHEMA_VERSION);
        assert_eq!(deserialized.session_id, "test-session");
        assert_eq!(deserialized.agent.name, "test");
        assert_eq!(deserialized.steps.len(), 1);
        assert_eq!(deserialized.steps[0].step_id, 1);
        assert_eq!(deserialized.steps[0].source, "user");
        let metrics = deserialized.steps[0].metrics.as_ref().unwrap();
        assert_eq!(metrics.prompt_tokens, Some(10));
        let final_metrics = deserialized.final_metrics.as_ref().unwrap();
        assert_eq!(final_metrics.total_prompt_tokens, Some(100));
        assert_eq!(final_metrics.total_steps, Some(1));
    }

    #[test]
    fn test_exporter_scope_filtering() {
        let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
        let root1 = Uuid::new_v4();
        let root2 = Uuid::new_v4();

        // Events under scope 1
        let e1 = event_builder(Uuid::new_v4(), EventType::Start)
            .scope_type(ScopeType::Llm)
            .input(Some(json!("agent1 input")))
            .parent_uuid(Some(root1))
            .build();
        let e2 = event_builder(e1.uuid(), EventType::End)
            .scope_type(ScopeType::Llm)
            .output(Some(json!("agent1 output")))
            .parent_uuid(Some(root1))
            .build();

        // Events under scope 2
        let e3 = event_builder(Uuid::new_v4(), EventType::Start)
            .scope_type(ScopeType::Llm)
            .input(Some(json!("agent2 input")))
            .parent_uuid(Some(root2))
            .build();
        let e4 = event_builder(e3.uuid(), EventType::End)
            .scope_type(ScopeType::Llm)
            .output(Some(json!("agent2 output")))
            .parent_uuid(Some(root2))
            .build();

        {
            let mut state = exporter.state.lock().unwrap();
            state.events.push(e1);
            state.events.push(e2);
            state.events.push(e3);
            state.events.push(e4);
        }

        let traj_all = exporter.export();
        assert_eq!(traj_all.steps.len(), 4);
    }

    #[test]
    fn test_exporter_clear() {
        let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());

        {
            let mut state = exporter.state.lock().unwrap();
            state.events.push(
                event_builder(Uuid::new_v4(), EventType::Mark)
                    .data(Some(json!("test")))
                    .build(),
            );
        }

        assert_eq!(exporter.export().steps.len(), 1);
        exporter.clear();
        assert!(exporter.export().steps.is_empty());
    }

    #[test]
    fn test_exporter_merged_tool_observations() {
        // Two consecutive tool end events should merge into one observation step.
        let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
        let llm_uuid = Uuid::new_v4();
        let tool1_uuid = Uuid::new_v4();
        let tool2_uuid = Uuid::new_v4();

        // LLM end with two promoted tool_calls
        let llm_end = event_builder(llm_uuid, EventType::End)
            .scope_type(ScopeType::Llm)
            .output(Some(json!({
                "content": null,
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"SF\"}"}},
                    {"id": "call_2", "type": "function", "function": {"name": "get_population", "arguments": "{\"city\": \"SF\"}"}}
                ]
            })))
            .build();

        // Two tool start events (skipped)
        let tool1_start = event_builder(tool1_uuid, EventType::Start)
            .name("get_weather")
            .scope_type(ScopeType::Tool)
            .input(Some(json!({"city": "SF"})))
            .build();
        let tool2_start = event_builder(tool2_uuid, EventType::Start)
            .name("get_population")
            .scope_type(ScopeType::Tool)
            .input(Some(json!({"city": "SF"})))
            .build();

        // Two tool end events (should merge)
        let tool1_end = event_builder(tool1_uuid, EventType::End)
            .name("get_weather")
            .scope_type(ScopeType::Tool)
            .output(Some(json!("62°F, foggy")))
            .tool_call_id(Some("call_1".to_string()))
            .build();
        let tool2_end = event_builder(tool2_uuid, EventType::End)
            .name("get_population")
            .scope_type(ScopeType::Tool)
            .output(Some(json!("873,965")))
            .tool_call_id(Some("call_2".to_string()))
            .build();

        {
            let mut state = exporter.state.lock().unwrap();
            state.events.push(llm_end);
            state.events.push(tool1_start);
            state.events.push(tool2_start);
            state.events.push(tool1_end);
            state.events.push(tool2_end);
        }

        let trajectory = exporter.export();
        // agent step + single merged observation step
        assert_eq!(trajectory.steps.len(), 2);

        // Agent step with promoted tool_calls
        let agent = &trajectory.steps[0];
        assert_eq!(agent.source, "agent");
        let tcs = agent.tool_calls.as_ref().unwrap();
        assert_eq!(tcs.len(), 2);
        // Arguments should be parsed JSON, not strings
        assert_eq!(tcs[0].arguments, json!({"city": "SF"}));
        assert_eq!(tcs[1].arguments, json!({"city": "SF"}));

        // Merged observation step
        let obs_step = &trajectory.steps[1];
        assert_eq!(obs_step.source, "system");
        let obs = obs_step.observation.as_ref().unwrap();
        assert_eq!(obs.results.len(), 2);
        assert_eq!(obs.results[0].source_call_id, Some("call_1".to_string()));
        assert_eq!(obs.results[0].content, json!("62°F, foggy"));
        assert_eq!(obs.results[1].source_call_id, Some("call_2".to_string()));
        assert_eq!(obs.results[1].content, json!("873,965"));
    }

    #[test]
    fn test_exporter_source_call_id_correlation_by_name() {
        // When tool_call_id is absent on the tool end event, correlate via function name
        // against the preceding LLM End's promoted tool_calls.
        let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
        let llm_uuid = Uuid::new_v4();
        let tool_uuid = Uuid::new_v4();

        let llm_end = event_builder(llm_uuid, EventType::End)
            .scope_type(ScopeType::Llm)
            .output(Some(json!({
                "content": null,
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_xyz", "type": "function", "function": {"name": "search", "arguments": "{}"}}
                ]
            })))
            .build();

        // Tool end without tool_call_id, but with function name
        let tool_end = event_builder(tool_uuid, EventType::End)
            .name("search")
            .scope_type(ScopeType::Tool)
            .output(Some(json!({"results": []})))
            .build();

        {
            let mut state = exporter.state.lock().unwrap();
            state.events.push(llm_end);
            state.events.push(tool_end);
        }

        let trajectory = exporter.export();
        assert_eq!(trajectory.steps.len(), 2);

        let obs = trajectory.steps[1].observation.as_ref().unwrap();
        // Correlated by function name "search" → "call_xyz"
        assert_eq!(obs.results[0].source_call_id, Some("call_xyz".to_string()));
    }

    #[test]
    fn test_exporter_user_message_extraction() {
        // LLM start input with max_tokens/model/tools/stream should extract just messages.
        let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
        let llm_uuid = Uuid::new_v4();

        let start = event_builder(llm_uuid, EventType::Start)
            .scope_type(ScopeType::Llm)
            .input(Some(json!({
                "content": {
                    "messages": [{"role": "user", "content": "hello"}],
                    "model": "gpt-4",
                    "max_tokens": 1024,
                    "stream": false,
                    "tools": [{"type": "function", "function": {"name": "search"}}]
                },
                "headers": {}
            })))
            .build();

        let end = event_builder(llm_uuid, EventType::End)
            .scope_type(ScopeType::Llm)
            .output(Some(json!("response")))
            .build();

        {
            let mut state = exporter.state.lock().unwrap();
            state.events.push(start);
            state.events.push(end);
        }

        let trajectory = exporter.export();
        // User step should contain just the messages array
        assert_eq!(
            trajectory.steps[0].message,
            json!([{"role": "user", "content": "hello"}])
        );
    }

    #[test]
    fn test_exporter_full_agent_loop() {
        // Simulate a complete agent loop: LLM→tool_calls→observations→LLM→final answer
        // This should produce 5 steps: user, agent+tool_calls, merged obs, user, agent
        let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
        let llm1_uuid = Uuid::new_v4();
        let llm2_uuid = Uuid::new_v4();
        let t1_uuid = Uuid::new_v4();
        let t2_uuid = Uuid::new_v4();

        // First LLM start
        let llm1_start = event_builder(llm1_uuid, EventType::Start)
            .scope_type(ScopeType::Llm)
            .input(Some(json!({
                "messages": [{"role": "user", "content": "What is the weather and population of SF?"}],
                "model": "nemotron",
                "tools": []
            })))
            .model_name(Some("nemotron".to_string()))
            .build();

        // First LLM end with tool_calls
        let llm1_end = event_builder(llm1_uuid, EventType::End)
            .scope_type(ScopeType::Llm)
            .output(Some(json!({
                "content": null,
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"SF\"}"}},
                    {"id": "c2", "type": "function", "function": {"name": "get_population", "arguments": "{\"city\":\"SF\"}"}}
                ],
                "token_usage": {"prompt_tokens": 100, "completion_tokens": 50}
            })))
            .model_name(Some("nemotron".to_string()))
            .build();

        // Tool starts (skipped)
        let t1_start = event_builder(t1_uuid, EventType::Start)
            .name("get_weather")
            .scope_type(ScopeType::Tool)
            .input(Some(json!({"city": "SF"})))
            .build();
        let t2_start = event_builder(t2_uuid, EventType::Start)
            .name("get_population")
            .scope_type(ScopeType::Tool)
            .input(Some(json!({"city": "SF"})))
            .build();

        // Tool ends (merged)
        let t1_end = event_builder(t1_uuid, EventType::End)
            .name("get_weather")
            .scope_type(ScopeType::Tool)
            .output(Some(json!("62°F, foggy")))
            .tool_call_id(Some("c1".to_string()))
            .build();
        let t2_end = event_builder(t2_uuid, EventType::End)
            .name("get_population")
            .scope_type(ScopeType::Tool)
            .output(Some(json!("873,965")))
            .tool_call_id(Some("c2".to_string()))
            .build();

        // Second LLM start (with tool results in messages)
        let llm2_start = event_builder(llm2_uuid, EventType::Start)
            .scope_type(ScopeType::Llm)
            .input(Some(json!({
                "messages": [
                    {"role": "user", "content": "What is the weather and population of SF?"},
                    {"role": "assistant", "content": null, "tool_calls": [{"id": "c1"}, {"id": "c2"}]},
                    {"role": "tool", "content": "62°F, foggy", "tool_call_id": "c1"},
                    {"role": "tool", "content": "873,965", "tool_call_id": "c2"}
                ],
                "model": "nemotron"
            })))
            .model_name(Some("nemotron".to_string()))
            .build();

        // Second LLM end (final answer)
        let llm2_end = event_builder(llm2_uuid, EventType::End)
            .scope_type(ScopeType::Llm)
            .output(Some(json!({
                "content": "The weather in SF is 62°F and foggy. Population is 873,965.",
                "role": "assistant",
                "token_usage": {"prompt_tokens": 200, "completion_tokens": 30}
            })))
            .model_name(Some("nemotron".to_string()))
            .build();

        {
            let mut state = exporter.state.lock().unwrap();
            state.events.extend([
                llm1_start, llm1_end, t1_start, t2_start, t1_end, t2_end, llm2_start, llm2_end,
            ]);
        }

        let trajectory = exporter.export();
        // Expected: user, agent+tool_calls, merged_obs, user, agent
        assert_eq!(trajectory.steps.len(), 5);

        assert_eq!(trajectory.steps[0].source, "user");
        assert_eq!(trajectory.steps[0].step_id, 1);

        assert_eq!(trajectory.steps[1].source, "agent");
        assert_eq!(trajectory.steps[1].step_id, 2);
        let tcs = trajectory.steps[1].tool_calls.as_ref().unwrap();
        assert_eq!(tcs.len(), 2);
        assert_eq!(tcs[0].function_name, "get_weather");
        assert_eq!(tcs[1].function_name, "get_population");

        assert_eq!(trajectory.steps[2].source, "system");
        assert_eq!(trajectory.steps[2].step_id, 3);
        let obs = trajectory.steps[2].observation.as_ref().unwrap();
        assert_eq!(obs.results.len(), 2);

        assert_eq!(trajectory.steps[3].source, "user");
        assert_eq!(trajectory.steps[3].step_id, 4);

        assert_eq!(trajectory.steps[4].source, "agent");
        assert_eq!(trajectory.steps[4].step_id, 5);
        assert_eq!(
            trajectory.steps[4].message,
            json!("The weather in SF is 62°F and foggy. Population is 873,965.")
        );

        // Final metrics should aggregate both LLM calls
        let fm = trajectory.final_metrics.as_ref().unwrap();
        assert_eq!(fm.total_prompt_tokens, Some(300));
        assert_eq!(fm.total_completion_tokens, Some(80));
    }

    #[test]
    fn test_reasoning_content_extracted() {
        // When an LLM End event carries output["reasoning"], the agent step
        // should have reasoning_content populated.
        let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
        let llm_uuid = Uuid::new_v4();

        let end = event_builder(llm_uuid, EventType::End)
            .scope_type(ScopeType::Llm)
            .output(Some(json!({
                "content": "The answer is 42.",
                "role": "assistant",
                "reasoning": "Let me think step by step. The question asks for the meaning of life...",
                "token_usage": { "prompt_tokens": 10, "completion_tokens": 5 }
            })))
            .build();

        {
            let mut state = exporter.state.lock().unwrap();
            state.events.push(end);
        }

        let trajectory = exporter.export();
        let agent_step = &trajectory.steps[0];
        assert_eq!(agent_step.source, "agent");
        assert_eq!(
            agent_step.reasoning_content,
            Some(
                "Let me think step by step. The question asks for the meaning of life..."
                    .to_string()
            )
        );
        // reasoning_content should not bleed into message
        assert_eq!(agent_step.message, json!("The answer is 42."));
    }

    #[test]
    fn test_reasoning_effort_propagated() {
        // reasoning_effort is set on the LLM Start event input and must be
        // carried forward to the agent step produced by the LLM End event.
        // This tests the stateful current_reasoning_effort handoff.
        let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
        let llm_uuid = Uuid::new_v4();

        let start = event_builder(llm_uuid, EventType::Start)
            .scope_type(ScopeType::Llm)
            .input(Some(json!({
                "messages": [{"role": "user", "content": "solve this"}],
                "reasoning_effort": "high"
            })))
            .build();

        let end = event_builder(llm_uuid, EventType::End)
            .scope_type(ScopeType::Llm)
            .output(Some(json!({
                "content": "Done.",
                "role": "assistant"
            })))
            .build();

        {
            let mut state = exporter.state.lock().unwrap();
            state.events.push(start);
            state.events.push(end);
        }

        let trajectory = exporter.export();
        // steps: user (LLM Start), agent (LLM End)
        let agent_step = &trajectory.steps[1];
        assert_eq!(agent_step.source, "agent");
        assert_eq!(agent_step.reasoning_effort, Some(json!("high")));
        // User step should NOT carry reasoning_effort
        assert!(trajectory.steps[0].reasoning_effort.is_none());
    }

    #[test]
    fn test_metrics_extra_captures_unknown_token_usage_keys() {
        // Unknown keys in token_usage (e.g. reasoning_tokens) should be
        // routed to metrics.extra rather than silently dropped.
        let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
        let llm_uuid = Uuid::new_v4();

        let end = event_builder(llm_uuid, EventType::End)
            .scope_type(ScopeType::Llm)
            .output(Some(json!({
                "content": "ok",
                "role": "assistant",
                "token_usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 10,
                    "reasoning_tokens": 150,
                    "cache_creation_input_tokens": 5
                }
            })))
            .build();

        {
            let mut state = exporter.state.lock().unwrap();
            state.events.push(end);
        }

        let trajectory = exporter.export();
        let metrics = trajectory.steps[0].metrics.as_ref().unwrap();
        assert_eq!(metrics.prompt_tokens, Some(20));
        assert_eq!(metrics.completion_tokens, Some(10));
        // Unknown keys land in extra
        let extra = metrics.extra.as_ref().unwrap();
        assert_eq!(extra["reasoning_tokens"], json!(150));
        assert_eq!(extra["cache_creation_input_tokens"], json!(5));
        // Known keys do not appear in extra
        assert!(extra.get("prompt_tokens").is_none());
        assert!(extra.get("completion_tokens").is_none());
    }

    #[test]
    fn test_step_extra_agent_ancestry() {
        // Agent step extra.ancestry is populated with function_id, function_name,
        // parent_id from the LLM End event.
        let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
        let agent_uuid = Uuid::new_v4();
        let llm_uuid = Uuid::new_v4();

        let llm_start = event_builder(llm_uuid, EventType::Start)
            .name("gpt-4")
            .scope_type(ScopeType::Llm)
            .parent_uuid(Some(agent_uuid))
            .input(Some(
                json!({"messages": [{"role": "user", "content": "hi"}]}),
            ))
            .build();

        let llm_end = event_builder(llm_uuid, EventType::End)
            .name("gpt-4")
            .scope_type(ScopeType::Llm)
            .parent_uuid(Some(agent_uuid))
            .output(Some(json!({"content": "hello", "role": "assistant"})))
            .build();

        {
            let mut state = exporter.state.lock().unwrap();
            state.events.push(llm_start);
            state.events.push(llm_end);
        }

        let trajectory = exporter.export();
        let agent_step = &trajectory.steps[1];
        assert_eq!(agent_step.source, "agent");

        let extra: AtifStepExtra =
            serde_json::from_value(agent_step.extra.clone().unwrap()).unwrap();
        assert_eq!(extra.ancestry.function_id, llm_uuid.to_string());
        assert_eq!(extra.ancestry.function_name, "gpt-4");
        assert_eq!(extra.ancestry.parent_id, Some(agent_uuid.to_string()));
    }

    #[test]
    fn test_step_extra_invocation_timestamps() {
        // Agent step extra.invocation carries paired start_timestamp and end_timestamp.
        let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
        let llm_uuid = Uuid::new_v4();

        let llm_start = event_builder(llm_uuid, EventType::Start)
            .name("gpt-4")
            .scope_type(ScopeType::Llm)
            .input(Some(json!({"messages": []})))
            .build();

        let llm_end = event_builder(llm_uuid, EventType::End)
            .name("gpt-4")
            .scope_type(ScopeType::Llm)
            .output(Some(json!({"content": "done", "role": "assistant"})))
            .build();

        {
            let mut state = exporter.state.lock().unwrap();
            state.events.push(llm_start);
            state.events.push(llm_end);
        }

        let trajectory = exporter.export();
        let agent_step = &trajectory.steps[1];
        let extra: AtifStepExtra =
            serde_json::from_value(agent_step.extra.clone().unwrap()).unwrap();

        let inv = extra.invocation.as_ref().unwrap();
        assert!(inv.start_timestamp.is_some());
        assert!(inv.end_timestamp.is_some());
        // end must be >= start
        assert!(inv.end_timestamp.unwrap() >= inv.start_timestamp.unwrap());
        assert_eq!(inv.invocation_id, Some(llm_uuid.to_string()));
        assert_eq!(inv.framework, Some("nexus".to_string()));
    }

    #[test]
    fn test_step_extra_user_step_has_ancestry_no_invocation() {
        // User step (LLM Start) gets ancestry but invocation is None —
        // end time is unknown at the time the user step is emitted.
        let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
        let llm_uuid = Uuid::new_v4();

        let llm_start = event_builder(llm_uuid, EventType::Start)
            .name("gpt-4")
            .scope_type(ScopeType::Llm)
            .input(Some(
                json!({"messages": [{"role": "user", "content": "hi"}]}),
            ))
            .build();

        let llm_end = event_builder(llm_uuid, EventType::End)
            .name("gpt-4")
            .scope_type(ScopeType::Llm)
            .output(Some(json!({"content": "hi back", "role": "assistant"})))
            .build();

        {
            let mut state = exporter.state.lock().unwrap();
            state.events.push(llm_start);
            state.events.push(llm_end);
        }

        let trajectory = exporter.export();
        let user_step = &trajectory.steps[0];
        assert_eq!(user_step.source, "user");

        let extra: AtifStepExtra =
            serde_json::from_value(user_step.extra.clone().unwrap()).unwrap();
        assert_eq!(extra.ancestry.function_id, llm_uuid.to_string());
        assert!(extra.invocation.is_none());
    }

    #[test]
    fn test_step_extra_tool_ancestry_aligned_with_tool_calls() {
        // tool_ancestry[i] must align with tool_calls[i] on the agent step.
        let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
        let llm_uuid = Uuid::new_v4();
        let tool1_uuid = Uuid::new_v4();
        let tool2_uuid = Uuid::new_v4();

        let llm_end = event_builder(llm_uuid, EventType::End)
            .scope_type(ScopeType::Llm)
            .output(Some(json!({
                "content": null,
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "search", "arguments": "{}"}},
                    {"id": "c2", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
                ]
            })))
            .build();

        let tool1_end = event_builder(tool1_uuid, EventType::End)
            .name("search")
            .scope_type(ScopeType::Tool)
            .output(Some(json!("result1")))
            .tool_call_id(Some("c1".to_string()))
            .build();

        let tool2_end = event_builder(tool2_uuid, EventType::End)
            .name("lookup")
            .scope_type(ScopeType::Tool)
            .output(Some(json!("result2")))
            .tool_call_id(Some("c2".to_string()))
            .build();

        {
            let mut state = exporter.state.lock().unwrap();
            state.events.push(llm_end);
            state.events.push(tool1_end);
            state.events.push(tool2_end);
        }

        let trajectory = exporter.export();
        let agent_step = &trajectory.steps[0];
        let extra: AtifStepExtra =
            serde_json::from_value(agent_step.extra.clone().unwrap()).unwrap();

        assert_eq!(extra.tool_ancestry.len(), 2);
        assert_eq!(extra.tool_ancestry[0].function_id, tool1_uuid.to_string());
        assert_eq!(extra.tool_ancestry[0].function_name, "search");
        assert_eq!(extra.tool_ancestry[1].function_id, tool2_uuid.to_string());
        assert_eq!(extra.tool_ancestry[1].function_name, "lookup");

        let tool_invocations = extra.tool_invocations.as_ref().unwrap();
        assert_eq!(tool_invocations.len(), 2);
        assert_eq!(tool_invocations[0].invocation_id, Some("c1".to_string()));
        assert_eq!(tool_invocations[1].invocation_id, Some("c2".to_string()));
    }

    #[test]
    fn test_step_extra_tool_ancestry_aligned_out_of_order_completion() {
        // Tools complete in reverse order (c2 before c1) but ancestry must
        // still align with tool_calls declaration order (c1=search, c2=lookup).
        let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
        let llm_uuid = Uuid::new_v4();
        let tool1_uuid = Uuid::new_v4();
        let tool2_uuid = Uuid::new_v4();

        let llm_end = event_builder(llm_uuid, EventType::End)
            .scope_type(ScopeType::Llm)
            .output(Some(json!({
                "content": null,
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "search", "arguments": "{}"}},
                    {"id": "c2", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
                ]
            })))
            .build();

        // c2 (lookup) completes before c1 (search) — out of declaration order.
        let mut tool2_end = event_builder(tool2_uuid, EventType::End)
            .name("lookup")
            .scope_type(ScopeType::Tool)
            .output(Some(json!("result2")))
            .tool_call_id(Some("c2".to_string()))
            .build();
        let tool2_end_ts = chrono::Utc::now();
        set_event_timestamp(&mut tool2_end, tool2_end_ts);

        let mut tool1_end = event_builder(tool1_uuid, EventType::End)
            .name("search")
            .scope_type(ScopeType::Tool)
            .output(Some(json!("result1")))
            .tool_call_id(Some("c1".to_string()))
            .build();
        // Ensure tool1_end sorts after tool2_end by timestamp.
        set_event_timestamp(
            &mut tool1_end,
            tool2_end_ts + chrono::Duration::milliseconds(10),
        );

        {
            let mut state = exporter.state.lock().unwrap();
            state.events.push(llm_end);
            state.events.push(tool2_end);
            state.events.push(tool1_end);
        }

        let trajectory = exporter.export();
        let agent_step = &trajectory.steps[0];
        let extra: AtifStepExtra =
            serde_json::from_value(agent_step.extra.clone().unwrap()).unwrap();

        // Despite out-of-order completion, ancestry aligns with tool_calls declaration order.
        assert_eq!(extra.tool_ancestry.len(), 2);
        assert_eq!(extra.tool_ancestry[0].function_name, "search"); // tool_calls[0] = c1
        assert_eq!(extra.tool_ancestry[1].function_name, "lookup"); // tool_calls[1] = c2

        let tool_invocations = extra.tool_invocations.as_ref().unwrap();
        assert_eq!(tool_invocations.len(), 2);
        assert_eq!(tool_invocations[0].invocation_id, Some("c1".to_string()));
        assert_eq!(tool_invocations[1].invocation_id, Some("c2".to_string()));
    }

    #[test]
    fn test_step_extra_tool_ancestry_does_not_bleed_across_turns() {
        // Tool ancestry from turn 1 must not appear on the agent step of turn 2.
        let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
        let llm1_uuid = Uuid::new_v4();
        let llm2_uuid = Uuid::new_v4();
        let tool1_uuid = Uuid::new_v4();
        let tool2_uuid = Uuid::new_v4();

        // Turn 1: LLM call + one tool
        let llm1_end = event_builder(llm1_uuid, EventType::End)
            .scope_type(ScopeType::Llm)
            .output(Some(json!({
                "content": null, "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "search", "arguments": "{}"}}
                ]
            })))
            .build();
        let tool1_end = event_builder(tool1_uuid, EventType::End)
            .name("search")
            .scope_type(ScopeType::Tool)
            .output(Some(json!("result1")))
            .tool_call_id(Some("c1".to_string()))
            .build();

        // Turn 2: new LLM call + one different tool
        let llm2_start = event_builder(llm2_uuid, EventType::Start)
            .scope_type(ScopeType::Llm)
            .input(Some(json!({"messages": []})))
            .build();
        let llm2_end = event_builder(llm2_uuid, EventType::End)
            .scope_type(ScopeType::Llm)
            .output(Some(json!({
                "content": null, "role": "assistant",
                "tool_calls": [
                    {"id": "c2", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
                ]
            })))
            .build();
        let tool2_end = event_builder(tool2_uuid, EventType::End)
            .name("lookup")
            .scope_type(ScopeType::Tool)
            .output(Some(json!("result2")))
            .tool_call_id(Some("c2".to_string()))
            .build();

        {
            let mut state = exporter.state.lock().unwrap();
            state.events.push(llm1_end);
            state.events.push(tool1_end);
            state.events.push(llm2_start);
            state.events.push(llm2_end);
            state.events.push(tool2_end);
        }

        let trajectory = exporter.export();
        // steps: agent(turn1), system(obs1), user(turn2), agent(turn2), system(obs2)
        let agent1 = trajectory
            .steps
            .iter()
            .find(|s| s.source == "agent" && s.step_id == 1)
            .unwrap();
        let agent2 = trajectory
            .steps
            .iter()
            .find(|s| s.source == "agent" && s.step_id == 4)
            .unwrap();

        let extra1: AtifStepExtra = serde_json::from_value(agent1.extra.clone().unwrap()).unwrap();
        let extra2: AtifStepExtra = serde_json::from_value(agent2.extra.clone().unwrap()).unwrap();

        // Turn 1 agent step has only search
        assert_eq!(extra1.tool_ancestry.len(), 1);
        assert_eq!(extra1.tool_ancestry[0].function_name, "search");

        // Turn 2 agent step has only lookup — no bleed from turn 1
        assert_eq!(extra2.tool_ancestry.len(), 1);
        assert_eq!(extra2.tool_ancestry[0].function_name, "lookup");
    }
}
