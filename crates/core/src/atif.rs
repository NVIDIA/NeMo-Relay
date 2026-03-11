// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ATIF (Agent Trajectory Interchange Format) exporter.
//!
//! This module provides types and an exporter that collects lifecycle events
//! from the NVMagic runtime and converts them into ATIF trajectories.
//!
//! The [`AtifExporter`] registers as an event subscriber, collects all events,
//! and can export them as an [`AtifTrajectory`] via [`AtifExporter::export`].

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::context::EventSubscriberFn;
use crate::json::Json;
use crate::types::{Event, EventType, ScopeType};

/// The ATIF schema version produced by this exporter.
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
    /// [`nvmagic_register_subscriber`](crate::api::nvmagic_register_subscriber).
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
    /// When `root_uuid` is provided, only events matching that root are included
    /// (for concurrent agent isolation). When `None`, all events are exported.
    pub fn export(&self, root_uuid: Option<Uuid>) -> AtifTrajectory {
        let state = self.state.lock().unwrap();
        let filtered_events: Vec<&Event> = if let Some(root) = root_uuid {
            state
                .events
                .iter()
                .filter(|e| e.root_uuid == Some(root))
                .collect()
        } else {
            state.events.iter().collect()
        };

        let steps = events_to_steps(&filtered_events);

        AtifTrajectory {
            schema_version: ATIF_SCHEMA_VERSION.to_string(),
            session_id: state.session_id.clone(),
            agent: state.agent_info.clone(),
            steps,
            final_metrics: None,
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
// Event-to-step mapping
// ---------------------------------------------------------------------------

/// Converts a slice of events into ATIF steps.
///
/// Mapping logic:
/// 1. Sort events by timestamp.
/// 2. Group by UUID: pair Start events with their End events.
/// 3. For each LLM pair:
///    - Start event -> user step (source="user", message=input)
///    - End event -> agent step (source="agent", message=output, model_name)
/// 4. For each Tool pair:
///    - Start event -> part of an agent step's tool_calls
///    - End event -> system step with observation
/// 5. Mark events -> included as steps if they have data.
/// 6. Scope events -> skipped unless they carry input data.
fn events_to_steps(events: &[&Event]) -> Vec<AtifStep> {
    let mut sorted: Vec<&Event> = events.to_vec();
    sorted.sort_by_key(|e| e.timestamp);

    // Group start/end events by UUID
    let mut start_events: HashMap<Uuid, &Event> = HashMap::new();
    let mut end_events: HashMap<Uuid, &Event> = HashMap::new();
    let mut mark_events: Vec<&Event> = Vec::new();

    for event in &sorted {
        match event.event_type {
            EventType::Start => {
                start_events.insert(event.uuid, event);
            }
            EventType::End => {
                end_events.insert(event.uuid, event);
            }
            EventType::Mark => {
                mark_events.push(event);
            }
        }
    }

    let mut steps = Vec::new();

    // Process events in timestamp order
    for event in &sorted {
        match event.event_type {
            EventType::Start => {
                match event.scope_type {
                    Some(ScopeType::Llm) => {
                        // LLM start -> user step
                        if let Some(input) = &event.input {
                            steps.push(AtifStep {
                                step_id: 0, // assigned below
                                source: "user".to_string(),
                                message: input.clone(),
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
                        // Tool start -> agent step with tool_calls
                        let tool_call_id = event
                            .tool_call_id
                            .clone()
                            .unwrap_or_else(|| event.uuid.to_string());
                        let function_name = event.name.clone().unwrap_or_default();
                        let arguments = event.input.clone().unwrap_or(Json::Null);

                        steps.push(AtifStep {
                            step_id: 0,
                            source: "agent".to_string(),
                            message: Json::Null,
                            timestamp: Some(event.timestamp.to_rfc3339()),
                            model_name: None,
                            tool_calls: Some(vec![AtifToolCall {
                                tool_call_id,
                                function_name,
                                arguments,
                            }]),
                            observation: None,
                            metrics: None,
                            extra: None,
                        });
                    }
                    _ => {
                        // Scope events: skip
                    }
                }
            }
            EventType::End => {
                match event.scope_type {
                    Some(ScopeType::Llm) => {
                        // LLM end -> agent step
                        if let Some(output) = &event.output {
                            steps.push(AtifStep {
                                step_id: 0,
                                source: "agent".to_string(),
                                message: output.clone(),
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
                        // Tool end -> system step with observation
                        if let Some(output) = &event.output {
                            let source_call_id = event.tool_call_id.clone();
                            steps.push(AtifStep {
                                step_id: 0,
                                source: "system".to_string(),
                                message: Json::Null,
                                timestamp: Some(event.timestamp.to_rfc3339()),
                                model_name: None,
                                tool_calls: None,
                                observation: Some(AtifObservation {
                                    results: vec![AtifObservationResult {
                                        source_call_id,
                                        content: output.clone(),
                                    }],
                                }),
                                metrics: None,
                                extra: None,
                            });
                        }
                    }
                    _ => {
                        // Scope end: skip
                    }
                }
            }
            EventType::Mark => {
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

        // Simulate tool start
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
        assert_eq!(trajectory.steps.len(), 2);

        // First step: agent with tool_calls
        let step1 = &trajectory.steps[0];
        assert_eq!(step1.step_id, 1);
        assert_eq!(step1.source, "agent");
        let tool_calls = step1.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].tool_call_id, "call_123");
        assert_eq!(tool_calls[0].function_name, "web_search");
        assert_eq!(tool_calls[0].arguments, json!({"query": "test"}));

        // Second step: system with observation
        let step2 = &trajectory.steps[1];
        assert_eq!(step2.step_id, 2);
        assert_eq!(step2.source, "system");
        let obs = step2.observation.as_ref().unwrap();
        assert_eq!(obs.results.len(), 1);
        assert_eq!(obs.results[0].source_call_id, Some("call_123".to_string()));
        assert_eq!(obs.results[0].content, json!({"results": ["result1"]}));
    }

    #[test]
    fn test_exporter_llm_lifecycle() {
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
            .output(Some(json!({"content": "Hi there!"})))
            .model_name(Some("gpt-4".to_string()))
            .build();

        {
            let mut state = exporter.state.lock().unwrap();
            state.events.push(start);
            state.events.push(end);
        }

        let trajectory = exporter.export(None);
        assert_eq!(trajectory.steps.len(), 2);

        // First step: user (from LLM start)
        let step1 = &trajectory.steps[0];
        assert_eq!(step1.step_id, 1);
        assert_eq!(step1.source, "user");
        assert_eq!(
            step1.message,
            json!({"messages": [{"role": "user", "content": "hello"}]})
        );
        assert_eq!(step1.model_name, Some("gpt-4".to_string()));

        // Second step: agent (from LLM end)
        let step2 = &trajectory.steps[1];
        assert_eq!(step2.step_id, 2);
        assert_eq!(step2.source, "agent");
        assert_eq!(step2.message, json!({"content": "Hi there!"}));
        assert_eq!(step2.model_name, Some("gpt-4".to_string()));
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
        // Scope events are skipped, so we should have: user, agent, agent(tool), system(obs)
        assert_eq!(trajectory.steps.len(), 4);

        assert_eq!(trajectory.steps[0].source, "user");
        assert_eq!(trajectory.steps[1].source, "agent");
        assert_eq!(trajectory.steps[2].source, "agent");
        assert!(trajectory.steps[2].tool_calls.is_some());
        assert_eq!(trajectory.steps[3].source, "system");
        assert!(trajectory.steps[3].observation.is_some());

        // Step IDs are 1-based
        for (i, step) in trajectory.steps.iter().enumerate() {
            assert_eq!(step.step_id, i + 1);
        }
    }

    #[test]
    fn test_exporter_tool_call_id_linking() {
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
        let tool_call = &trajectory.steps[0].tool_calls.as_ref().unwrap()[0];
        let obs_result = &trajectory.steps[1].observation.as_ref().unwrap().results[0];

        // The tool_call_id should match between the tool call and observation
        assert_eq!(tool_call.tool_call_id, "call_abc");
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
}
