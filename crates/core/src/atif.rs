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
//! # Concurrent Agent Isolation
//!
//! When exporting, pass a `scope_uuid` to [`AtifExporter::export`] to filter
//! events to a specific agent's root scope and its descendants. This enables
//! concurrent agents to produce independent trajectory files.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::context::EventSubscriberFn;
use crate::json::Json;
use crate::types::{Event, EventType, ScopeType};

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
    /// Tool calls made by the agent in this step.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<AtifToolCall>>,
    /// Observation (tool results) for this step.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observation: Option<AtifObservation>,
    /// Token usage and cost metrics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics: Option<AtifMetrics>,
    /// Extra metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra: Option<Json>,
}

/// Token usage and cost metrics for a step or trajectory.
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
    /// Extra metrics.
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
    /// Aggregate metrics for the entire trajectory.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_metrics: Option<AtifMetrics>,
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
        Box::new(move |event: &Event| {
            if let Ok(mut s) = state.lock() {
                s.events.push(event.clone());
            }
        })
    }

    /// Exports collected events as an [`AtifTrajectory`].
    ///
    /// When `scope_uuid` is provided, only events belonging to that scope or
    /// any of its descendants are included. This allows filtering by any scope
    /// in the hierarchy — not just the auto-created root scope. When `None`,
    /// all events are exported.
    pub fn export(&self, scope_uuid: Option<Uuid>) -> AtifTrajectory {
        let state = self.state.lock().unwrap();
        let filtered_events: Vec<&Event> = if let Some(target) = scope_uuid {
            // Build a set of UUIDs that are the target or descendants of it.
            // An event belongs to the target scope if:
            //   - Its parent_uuid is the target, OR
            //   - Its parent_uuid is already a known descendant
            // Also match events whose root_uuid equals the target (backwards compat).
            let mut included: HashSet<Uuid> = HashSet::new();
            included.insert(target);

            // First pass: discover descendant scope UUIDs by walking Start events
            // in timestamp order. A Start event with parent_uuid in the included
            // set means its uuid is also a descendant.
            let mut sorted: Vec<&Event> = state.events.iter().collect();
            sorted.sort_by_key(|e| e.timestamp);
            for event in &sorted {
                if event.event_type == EventType::Start {
                    if let Some(parent) = event.parent_uuid {
                        if included.contains(&parent) {
                            included.insert(event.uuid);
                        }
                    }
                }
            }

            // Second pass: include events whose parent_uuid or uuid is in the
            // descendant set, OR whose root_uuid matches the target.
            state
                .events
                .iter()
                .filter(|e| {
                    e.root_uuid == Some(target)
                        || included.contains(&e.uuid)
                        || e.parent_uuid
                            .map(|p| included.contains(&p))
                            .unwrap_or(false)
                })
                .collect()
        } else {
            state.events.iter().collect()
        };

        let steps = events_to_steps(&filtered_events);
        let final_metrics = compute_final_metrics(&steps);

        AtifTrajectory {
            schema_version: ATIF_SCHEMA_VERSION.to_string(),
            session_id: state.session_id.clone(),
            agent: state.agent_info.clone(),
            steps,
            final_metrics,
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

/// Try to extract `AtifMetrics` from a `token_usage` object in the LLM response.
///
/// Returns `None` if the response has no `token_usage` or it is not an object.
fn extract_metrics(output: &Json) -> Option<AtifMetrics> {
    let usage = output.as_object()?.get("token_usage")?.as_object()?;
    // Only produce metrics if at least one field is present.
    let prompt = usage.get("prompt_tokens").and_then(Json::as_u64);
    let completion = usage.get("completion_tokens").and_then(Json::as_u64);
    let cached = usage.get("cached_tokens").and_then(Json::as_u64);
    if prompt.is_none() && completion.is_none() && cached.is_none() {
        return None;
    }
    Some(AtifMetrics {
        prompt_tokens: prompt,
        completion_tokens: completion,
        cached_tokens: cached,
        cost_usd: None,
        extra: None,
    })
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
/// Returns `None` if no steps carry metrics.
fn compute_final_metrics(steps: &[AtifStep]) -> Option<AtifMetrics> {
    let mut total_prompt: u64 = 0;
    let mut total_completion: u64 = 0;
    let mut total_cached: u64 = 0;
    let mut has_any = false;

    for step in steps {
        if let Some(m) = &step.metrics {
            has_any = true;
            total_prompt += m.prompt_tokens.unwrap_or(0);
            total_completion += m.completion_tokens.unwrap_or(0);
            total_cached += m.cached_tokens.unwrap_or(0);
        }
    }

    if !has_any {
        return None;
    }

    Some(AtifMetrics {
        prompt_tokens: Some(total_prompt),
        completion_tokens: Some(total_completion),
        cached_tokens: if total_cached > 0 {
            Some(total_cached)
        } else {
            None
        },
        cost_usd: None,
        extra: None,
    })
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
    sorted.sort_by_key(|e| e.timestamp);

    let mut steps = Vec::new();
    // Track the most recent LLM End's promoted tool_calls for source_call_id correlation.
    // Maps function_name → tool_call_id for the most recent LLM End step.
    let mut last_tool_call_map: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    // Buffer for merging consecutive Tool End observations.
    let mut pending_observations: Vec<AtifObservationResult> = Vec::new();
    let mut pending_obs_timestamp: Option<String> = None;

    // Flush any buffered observations into a single merged system step.
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
            tool_calls: None,
            observation: Some(AtifObservation {
                results: std::mem::take(obs),
            }),
            metrics: None,
            extra: None,
        });
    };

    for event in &sorted {
        match event.event_type {
            EventType::Start => {
                match event.scope_type {
                    Some(ScopeType::Llm) => {
                        // Flush pending observations before a new LLM turn
                        flush_observations(
                            &mut steps,
                            &mut pending_observations,
                            &mut pending_obs_timestamp,
                        );
                        // LLM start -> user step (extract just the messages array)
                        if let Some(input) = &event.input {
                            let content = unwrap_llm_request(input);
                            steps.push(AtifStep {
                                step_id: 0,
                                source: "user".to_string(),
                                message: extract_user_messages(&content),
                                timestamp: Some(event.timestamp.to_rfc3339()),
                                model_name: event.model_name.clone(),
                                tool_calls: None,
                                observation: None,
                                metrics: None,
                                extra: None,
                            });
                        }
                    }
                    Some(ScopeType::Tool) => {
                        // Tool start: SKIP — tool_calls come from LLM End promotion
                    }
                    _ => {
                        // Scope events: skip
                    }
                }
            }
            EventType::End => {
                match event.scope_type {
                    Some(ScopeType::Llm) => {
                        // Flush pending observations before a new LLM end
                        flush_observations(
                            &mut steps,
                            &mut pending_observations,
                            &mut pending_obs_timestamp,
                        );
                        // LLM end -> agent step
                        if let Some(output) = &event.output {
                            let tool_calls = extract_tool_calls(output);
                            // Build correlation map for upcoming tool observations
                            last_tool_call_map.clear();
                            if let Some(ref tcs) = tool_calls {
                                for tc in tcs {
                                    if !tc.function_name.is_empty() {
                                        last_tool_call_map.insert(
                                            tc.function_name.clone(),
                                            tc.tool_call_id.clone(),
                                        );
                                    }
                                }
                            }
                            steps.push(AtifStep {
                                step_id: 0,
                                source: "agent".to_string(),
                                message: extract_llm_response_message(output),
                                timestamp: Some(event.timestamp.to_rfc3339()),
                                model_name: event.model_name.clone(),
                                tool_calls,
                                observation: None,
                                metrics: extract_metrics(output),
                                extra: None,
                            });
                        }
                    }
                    Some(ScopeType::Tool) => {
                        // Tool end -> buffer as observation result for merging
                        if let Some(output) = &event.output {
                            // Correlate: prefer event's own tool_call_id, then
                            // look up by function_name in the last LLM End's promoted calls.
                            let source_call_id = event.tool_call_id.clone().or_else(|| {
                                event
                                    .name
                                    .as_ref()
                                    .and_then(|n| last_tool_call_map.get(n).cloned())
                            });
                            if pending_obs_timestamp.is_none() {
                                pending_obs_timestamp = Some(event.timestamp.to_rfc3339());
                            }
                            pending_observations.push(AtifObservationResult {
                                source_call_id,
                                content: output.clone(),
                            });
                        }
                    }
                    _ => {
                        // Scope end: skip
                    }
                }
            }
            EventType::Mark => {
                // Flush pending observations before a mark event
                flush_observations(
                    &mut steps,
                    &mut pending_observations,
                    &mut pending_obs_timestamp,
                );
                // Mark events: include if they have data
                if let Some(data) = &event.data {
                    steps.push(AtifStep {
                        step_id: 0,
                        source: "system".to_string(),
                        message: data.clone(),
                        timestamp: Some(event.timestamp.to_rfc3339()),
                        model_name: None,
                        tool_calls: None,
                        observation: None,
                        metrics: None,
                        extra: None,
                    });
                }
            }
        }
    }

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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;
    use serde_json::json;

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
        let trajectory = exporter.export(None);

        assert_eq!(trajectory.schema_version, ATIF_SCHEMA_VERSION);
        assert_eq!(trajectory.session_id, "session-1");
        assert_eq!(trajectory.agent.name, "test-agent");
        assert!(trajectory.steps.is_empty());
        assert!(trajectory.final_metrics.is_none());
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
        let start = Event::builder(tool_uuid, EventType::Start)
            .name("web_search")
            .scope_type(ScopeType::Tool)
            .input(Some(json!({"query": "test"})))
            .tool_call_id(Some("call_123".to_string()))
            .build();

        // Simulate tool end
        let end = Event::builder(tool_uuid, EventType::End)
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

        let trajectory = exporter.export(None);
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
        let start = Event::builder(llm_uuid, EventType::Start)
            .name("gpt-4")
            .scope_type(ScopeType::Llm)
            .input(Some(json!({
                "content": {"messages": [{"role": "user", "content": "hello"}]},
                "headers": {}
            })))
            .model_name(Some("gpt-4".to_string()))
            .build();

        // Output with content, token_usage, and tool_calls.
        let end = Event::builder(llm_uuid, EventType::End)
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

        let trajectory = exporter.export(None);
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

        // final_metrics should aggregate
        let fm = trajectory.final_metrics.as_ref().unwrap();
        assert_eq!(fm.prompt_tokens, Some(10));
        assert_eq!(fm.completion_tokens, Some(20));
    }

    #[test]
    fn test_exporter_llm_lifecycle_plain_input() {
        // Input without LLMRequest envelope — passed through unchanged.
        let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
        let llm_uuid = Uuid::new_v4();

        let start = Event::builder(llm_uuid, EventType::Start)
            .name("gpt-4")
            .scope_type(ScopeType::Llm)
            .input(Some(
                json!({"messages": [{"role": "user", "content": "hello"}]}),
            ))
            .model_name(Some("gpt-4".to_string()))
            .build();

        let end = Event::builder(llm_uuid, EventType::End)
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

        let trajectory = exporter.export(None);
        assert_eq!(trajectory.steps.len(), 2);

        // Input without headers key — messages array is still extracted
        assert_eq!(
            trajectory.steps[0].message,
            json!([{"role": "user", "content": "hello"}])
        );
        // Non-object output is passed through as-is
        assert_eq!(trajectory.steps[1].message, json!("simple string response"));
        assert!(trajectory.steps[1].metrics.is_none());
        assert!(trajectory.final_metrics.is_none());
    }

    #[test]
    fn test_exporter_llm_tool_calls_promoted() {
        let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
        let llm_uuid = Uuid::new_v4();

        let end = Event::builder(llm_uuid, EventType::End)
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

        let trajectory = exporter.export(None);
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
        let scope_start = Event::builder(scope_uuid, EventType::Start)
            .name("agent")
            .scope_type(ScopeType::Agent)
            .build();

        // LLM start/end
        let llm_start = Event::builder(llm_uuid, EventType::Start)
            .scope_type(ScopeType::Llm)
            .input(Some(json!({"prompt": "What is 2+2?"})))
            .build();
        let llm_end = Event::builder(llm_uuid, EventType::End)
            .scope_type(ScopeType::Llm)
            .output(Some(json!({"answer": "4"})))
            .build();

        // Tool start/end
        let tool_start = Event::builder(tool_uuid, EventType::Start)
            .name("calculator")
            .scope_type(ScopeType::Tool)
            .input(Some(json!({"expr": "2+2"})))
            .tool_call_id(Some("call_1".to_string()))
            .build();
        let tool_end = Event::builder(tool_uuid, EventType::End)
            .name("calculator")
            .scope_type(ScopeType::Tool)
            .output(Some(json!(4)))
            .tool_call_id(Some("call_1".to_string()))
            .build();

        // Scope end (should be skipped)
        let scope_end = Event::builder(scope_uuid, EventType::End)
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

        let trajectory = exporter.export(None);
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

        let start = Event::builder(tool_uuid, EventType::Start)
            .name("my_tool")
            .scope_type(ScopeType::Tool)
            .input(Some(json!({"x": 1})))
            .tool_call_id(Some("call_abc".to_string()))
            .build();

        let end = Event::builder(tool_uuid, EventType::End)
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

        let trajectory = exporter.export(None);
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
                tool_calls: None,
                observation: None,
                metrics: Some(AtifMetrics {
                    prompt_tokens: Some(10),
                    completion_tokens: Some(20),
                    cached_tokens: None,
                    cost_usd: Some(0.001),
                    extra: None,
                }),
                extra: None,
            }],
            final_metrics: Some(AtifMetrics {
                prompt_tokens: Some(100),
                completion_tokens: Some(200),
                cached_tokens: Some(50),
                cost_usd: Some(0.01),
                extra: None,
            }),
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
        assert_eq!(final_metrics.prompt_tokens, Some(100));
    }

    #[test]
    fn test_exporter_root_uuid_filtering() {
        let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
        let root1 = Uuid::new_v4();
        let root2 = Uuid::new_v4();

        // Events from agent 1
        let e1 = Event::builder(Uuid::new_v4(), EventType::Start)
            .scope_type(ScopeType::Llm)
            .input(Some(json!("agent1 input")))
            .root_uuid(Some(root1))
            .build();
        let e2 = Event::builder(e1.uuid, EventType::End)
            .scope_type(ScopeType::Llm)
            .output(Some(json!("agent1 output")))
            .root_uuid(Some(root1))
            .build();

        // Events from agent 2
        let e3 = Event::builder(Uuid::new_v4(), EventType::Start)
            .scope_type(ScopeType::Llm)
            .input(Some(json!("agent2 input")))
            .root_uuid(Some(root2))
            .build();
        let e4 = Event::builder(e3.uuid, EventType::End)
            .scope_type(ScopeType::Llm)
            .output(Some(json!("agent2 output")))
            .root_uuid(Some(root2))
            .build();

        {
            let mut state = exporter.state.lock().unwrap();
            state.events.push(e1);
            state.events.push(e2);
            state.events.push(e3);
            state.events.push(e4);
        }

        // Export with root1 filter
        let traj1 = exporter.export(Some(root1));
        assert_eq!(traj1.steps.len(), 2);
        assert_eq!(traj1.steps[0].message, json!("agent1 input"));
        assert_eq!(traj1.steps[1].message, json!("agent1 output"));

        // Export with root2 filter
        let traj2 = exporter.export(Some(root2));
        assert_eq!(traj2.steps.len(), 2);
        assert_eq!(traj2.steps[0].message, json!("agent2 input"));
        assert_eq!(traj2.steps[1].message, json!("agent2 output"));

        // Export without filter
        let traj_all = exporter.export(None);
        assert_eq!(traj_all.steps.len(), 4);
    }

    #[test]
    fn test_exporter_hierarchy_filtering() {
        // Filtering by an agent scope UUID should include events from that scope
        // and all its descendants (LLM calls, tool calls parented under it).
        let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());

        let root_uuid = Uuid::new_v4(); // auto-created root scope
        let agent_uuid = Uuid::new_v4(); // user-pushed agent scope

        // Agent scope Start (parent = root)
        let agent_start = Event::builder(agent_uuid, EventType::Start)
            .name("my-agent")
            .scope_type(ScopeType::Agent)
            .parent_uuid(Some(root_uuid))
            .root_uuid(Some(root_uuid))
            .build();

        // LLM call under agent (parent = agent)
        let llm_uuid = Uuid::new_v4();
        let llm_start = Event::builder(llm_uuid, EventType::Start)
            .scope_type(ScopeType::Llm)
            .input(Some(
                json!({"messages": [{"role": "user", "content": "hi"}]}),
            ))
            .parent_uuid(Some(agent_uuid))
            .root_uuid(Some(root_uuid))
            .build();
        let llm_end = Event::builder(llm_uuid, EventType::End)
            .scope_type(ScopeType::Llm)
            .output(Some(json!({"content": "hello!", "role": "assistant"})))
            .parent_uuid(Some(agent_uuid))
            .root_uuid(Some(root_uuid))
            .build();

        // Tool call under LLM (parent = llm)
        let tool_uuid = Uuid::new_v4();
        let tool_start = Event::builder(tool_uuid, EventType::Start)
            .name("search")
            .scope_type(ScopeType::Tool)
            .input(Some(json!({"q": "test"})))
            .parent_uuid(Some(llm_uuid))
            .root_uuid(Some(root_uuid))
            .build();
        let tool_end = Event::builder(tool_uuid, EventType::End)
            .name("search")
            .scope_type(ScopeType::Tool)
            .output(Some(json!("result")))
            .tool_call_id(Some("c1".to_string()))
            .parent_uuid(Some(llm_uuid))
            .root_uuid(Some(root_uuid))
            .build();

        // Unrelated event from a different agent (parent = different root)
        let other_uuid = Uuid::new_v4();
        let other_root = Uuid::new_v4();
        let unrelated = Event::builder(other_uuid, EventType::Start)
            .scope_type(ScopeType::Llm)
            .input(Some(json!("other agent")))
            .parent_uuid(Some(other_root))
            .root_uuid(Some(other_root))
            .build();

        {
            let mut state = exporter.state.lock().unwrap();
            state.events.extend([
                agent_start,
                llm_start,
                llm_end,
                tool_start,
                tool_end,
                unrelated,
            ]);
        }

        // Filter by agent_uuid — should get LLM + tool events (descendants),
        // plus the agent Start itself
        let traj = exporter.export(Some(agent_uuid));
        // agent Start is scope (skipped), LLM start → user, LLM end → agent,
        // tool start → skipped, tool end → observation
        assert_eq!(traj.steps.len(), 3);
        assert_eq!(traj.steps[0].source, "user");
        assert_eq!(traj.steps[1].source, "agent");
        assert_eq!(traj.steps[2].source, "system");

        // Filter by root_uuid — should get everything except the unrelated event
        let traj_root = exporter.export(Some(root_uuid));
        // Same 3 steps + agent Start/End scope events (skipped) = still 3
        assert_eq!(traj_root.steps.len(), 3);

        // No filter — should get all events including unrelated
        let traj_all = exporter.export(None);
        assert_eq!(traj_all.steps.len(), 4); // +1 user step from unrelated LLM start
    }

    #[test]
    fn test_exporter_clear() {
        let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());

        {
            let mut state = exporter.state.lock().unwrap();
            state.events.push(
                Event::builder(Uuid::new_v4(), EventType::Mark)
                    .data(Some(json!("test")))
                    .build(),
            );
        }

        assert_eq!(exporter.export(None).steps.len(), 1);
        exporter.clear();
        assert!(exporter.export(None).steps.is_empty());
    }

    #[test]
    fn test_exporter_merged_tool_observations() {
        // Two consecutive tool end events should merge into one observation step.
        let exporter = AtifExporter::new("session-1".to_string(), make_agent_info());
        let llm_uuid = Uuid::new_v4();
        let tool1_uuid = Uuid::new_v4();
        let tool2_uuid = Uuid::new_v4();

        // LLM end with two promoted tool_calls
        let llm_end = Event::builder(llm_uuid, EventType::End)
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
        let tool1_start = Event::builder(tool1_uuid, EventType::Start)
            .name("get_weather")
            .scope_type(ScopeType::Tool)
            .input(Some(json!({"city": "SF"})))
            .build();
        let tool2_start = Event::builder(tool2_uuid, EventType::Start)
            .name("get_population")
            .scope_type(ScopeType::Tool)
            .input(Some(json!({"city": "SF"})))
            .build();

        // Two tool end events (should merge)
        let tool1_end = Event::builder(tool1_uuid, EventType::End)
            .name("get_weather")
            .scope_type(ScopeType::Tool)
            .output(Some(json!("62°F, foggy")))
            .tool_call_id(Some("call_1".to_string()))
            .build();
        let tool2_end = Event::builder(tool2_uuid, EventType::End)
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

        let trajectory = exporter.export(None);
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

        let llm_end = Event::builder(llm_uuid, EventType::End)
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
        let tool_end = Event::builder(tool_uuid, EventType::End)
            .name("search")
            .scope_type(ScopeType::Tool)
            .output(Some(json!({"results": []})))
            .build();

        {
            let mut state = exporter.state.lock().unwrap();
            state.events.push(llm_end);
            state.events.push(tool_end);
        }

        let trajectory = exporter.export(None);
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

        let start = Event::builder(llm_uuid, EventType::Start)
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

        let end = Event::builder(llm_uuid, EventType::End)
            .scope_type(ScopeType::Llm)
            .output(Some(json!("response")))
            .build();

        {
            let mut state = exporter.state.lock().unwrap();
            state.events.push(start);
            state.events.push(end);
        }

        let trajectory = exporter.export(None);
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
        let llm1_start = Event::builder(llm1_uuid, EventType::Start)
            .scope_type(ScopeType::Llm)
            .input(Some(json!({
                "messages": [{"role": "user", "content": "What is the weather and population of SF?"}],
                "model": "nemotron",
                "tools": []
            })))
            .model_name(Some("nemotron".to_string()))
            .build();

        // First LLM end with tool_calls
        let llm1_end = Event::builder(llm1_uuid, EventType::End)
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
        let t1_start = Event::builder(t1_uuid, EventType::Start)
            .name("get_weather")
            .scope_type(ScopeType::Tool)
            .input(Some(json!({"city": "SF"})))
            .build();
        let t2_start = Event::builder(t2_uuid, EventType::Start)
            .name("get_population")
            .scope_type(ScopeType::Tool)
            .input(Some(json!({"city": "SF"})))
            .build();

        // Tool ends (merged)
        let t1_end = Event::builder(t1_uuid, EventType::End)
            .name("get_weather")
            .scope_type(ScopeType::Tool)
            .output(Some(json!("62°F, foggy")))
            .tool_call_id(Some("c1".to_string()))
            .build();
        let t2_end = Event::builder(t2_uuid, EventType::End)
            .name("get_population")
            .scope_type(ScopeType::Tool)
            .output(Some(json!("873,965")))
            .tool_call_id(Some("c2".to_string()))
            .build();

        // Second LLM start (with tool results in messages)
        let llm2_start = Event::builder(llm2_uuid, EventType::Start)
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
        let llm2_end = Event::builder(llm2_uuid, EventType::End)
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

        let trajectory = exporter.export(None);
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
        assert_eq!(fm.prompt_tokens, Some(300));
        assert_eq!(fm.completion_tokens, Some(80));
    }
}
