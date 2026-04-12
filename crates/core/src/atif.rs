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

use crate::context::callbacks::EventSubscriberFn;
use crate::json::Json;
use crate::types::event::Event;

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
    /// [`register_subscriber`](crate::api::register_subscriber).
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
        // Prefer the "content" field if it carries actual content.
        if let Some(content) = obj.get("content")
            && !content.is_null()
        {
            return content.clone();
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
            if let Some(r) = obj.get("reasoning")
                && !r.is_null()
            {
                summary.insert("reasoning".to_string(), r.clone());
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
        if let (Some(i), Some(anc)) = (idx.take(), ancestry.take())
            && let Some(step) = steps.get_mut(i)
        {
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
                        "nemo_flow",
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
                        "nemo_flow",
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
#[path = "../tests/unit/atif_tests.rs"]
mod tests;
