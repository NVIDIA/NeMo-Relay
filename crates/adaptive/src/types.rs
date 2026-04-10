// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Core data types for the nemo-flow-adaptive crate.
//!
//! This module defines the vocabulary types used by the adaptive:
//! [`MetadataEnvelope`], [`ParallelHint`], [`RunRecord`], [`CallRecord`],
//! [`CallKind`], [`ExecutionPlan`], and [`ParallelGroup`].
//!
//! All types derive [`serde::Serialize`] and [`serde::Deserialize`] so they
//! can be persisted by any [`StorageBackend`](crate::storage) implementation
//! and round-tripped through JSON without loss.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Type alias for [`serde_json::Value`], matching the workspace convention.
pub type Json = serde_json::Value;

/// Per-request metadata injected by the LLM request intercept.
///
/// Carries the run identifier, agent identity, parallelism hints, and an
/// open-ended extensions map for user-defined data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataEnvelope {
    /// Unique identifier for the current run.
    pub run_id: Uuid,
    /// Identifier of the agent that owns this run.
    pub agent_id: String,
    /// Parallel execution hints attached to this request.
    pub parallel_hints: Vec<ParallelHint>,
    /// Open-ended extensions map for user-defined data.
    pub extensions: Json,
}

/// Annotates a tool with a parallel execution group.
///
/// When `explicit` is `true`, the hint was provided by an annotation;
/// when `false`, it was learned from historical execution data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelHint {
    /// Name of the tool this hint applies to.
    pub tool_name: String,
    /// Identifier of the parallel group this tool belongs to.
    pub group_id: String,
    /// Whether this hint was explicitly annotated (`true`) or learned (`false`).
    pub explicit: bool,
}

/// Distinguishes LLM calls from tool calls in a [`CallRecord`].
///
/// Serializes to/from lowercase strings: `"llm"` and `"tool"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CallKind {
    /// An LLM (large language model) call.
    Llm,
    /// A tool invocation.
    Tool,
}

/// Captures a single LLM or tool call within a run.
///
/// Each record carries the call kind, timing information, and an optional
/// snapshot of the [`MetadataEnvelope`] that was active at call time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallRecord {
    /// Whether this was an LLM or tool call.
    pub kind: CallKind,
    /// Name of the LLM model or tool that was called.
    pub name: String,
    /// When the call started.
    pub started_at: DateTime<Utc>,
    /// When the call ended, if it has completed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ended_at: Option<DateTime<Utc>>,
    /// Snapshot of the metadata envelope at call time, if available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata_snapshot: Option<MetadataEnvelope>,
    /// Output token count for LLM calls, if available.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub output_tokens: Option<u32>,
    /// Prompt token count, populated from annotated_response.usage.prompt_tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub prompt_tokens: Option<u32>,
    /// Total token count, populated from annotated_response.usage.total_tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub total_tokens: Option<u32>,
    /// Actual serving model name, populated from annotated_response.model.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub model_name: Option<String>,
    /// Count of tool calls in the LLM response.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub tool_call_count: Option<u32>,
}

/// Captures a complete agent execution run.
///
/// A run record holds the full sequence of [`CallRecord`]s that occurred
/// during the run, along with timing information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRecord {
    /// Unique identifier for this run.
    pub id: Uuid,
    /// Identifier of the agent that executed this run.
    pub agent_id: String,
    /// Ordered sequence of calls made during this run.
    pub calls: Vec<CallRecord>,
    /// When the run started.
    pub started_at: DateTime<Utc>,
    /// When the run ended, if it has completed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ended_at: Option<DateTime<Utc>>,
}

/// Groups tool names that can execute concurrently.
///
/// Each group has a unique identifier and a list of tool names that belong
/// to the group.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelGroup {
    /// Unique identifier for this parallel group.
    pub group_id: String,
    /// Tool names that belong to this group and can run concurrently.
    pub tool_names: Vec<String>,
}

/// Hot-cache state holding current parallel groups and a metadata template.
///
/// An `ExecutionPlan` is a data structure only — it holds the information
/// needed by intercepts to inject metadata and evaluate parallelism, but
/// contains no update logic or methods beyond construction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPlan {
    /// Identifier of the agent this plan belongs to.
    pub agent_id: String,
    /// Parallel groups currently active for this agent.
    pub parallel_groups: Vec<ParallelGroup>,
    /// Template envelope used to stamp outgoing requests.
    pub metadata_template: MetadataEnvelope,
}

/// Typed struct for agent hints injected into LLM request headers.
///
/// Field names match NAT's `nvext.agent_hints` dict keys. Serialized as
/// the `x-nemo-flow-adaptive-agent-hints` header value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentHints {
    /// Output Sequence Length (tokens), from prediction output_tokens p90.
    pub osl: u32,
    /// Inter-Arrival Time (ms), from prediction interarrival_ms mean.
    pub iat: u32,
    /// Engine scheduler priority = max_sensitivity - latency_sensitivity.
    pub priority: i32,
    /// Sensitivity score kept as a float for downstream scheduler compatibility.
    pub latency_sensitivity: f64,
    /// KV cache prefix identity, e.g. "{agent_id}-d{scope_depth}".
    pub prefix_id: String,
    /// Expected total requests = prediction remaining_calls mean + call_index.
    pub total_requests: u32,
}

/// Hot-cache state holding plan, prediction trie, and pre-computed hints.
///
/// Replaces `Option<ExecutionPlan>` with a single struct under one
/// `Arc<RwLock<>>` for atomic reads on the intercept hot path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotCache {
    /// v1 execution plan data (unchanged).
    pub plan: Option<ExecutionPlan>,
    /// Prediction trie built by the learner pipeline.
    pub trie: Option<crate::trie::PredictionTrieNode>,
    /// Pre-computed default hints from the trie root node.
    pub agent_hints_default: Option<AgentHints>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_metadata_envelope_serde_roundtrip() {
        let envelope = MetadataEnvelope {
            run_id: Uuid::new_v4(),
            agent_id: "agent-1".to_string(),
            parallel_hints: vec![ParallelHint {
                tool_name: "search".to_string(),
                group_id: "g1".to_string(),
                explicit: true,
            }],
            extensions: json!({"key": "value"}),
        };

        let serialized = serde_json::to_string(&envelope).unwrap();
        let deserialized: MetadataEnvelope = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.run_id, envelope.run_id);
        assert_eq!(deserialized.agent_id, envelope.agent_id);
        assert_eq!(deserialized.parallel_hints.len(), 1);
        assert_eq!(deserialized.extensions, json!({"key": "value"}));
    }

    #[test]
    fn test_parallel_hint_serde_roundtrip() {
        let hint = ParallelHint {
            tool_name: "weather".to_string(),
            group_id: "grp-42".to_string(),
            explicit: false,
        };

        let serialized = serde_json::to_string(&hint).unwrap();
        let deserialized: ParallelHint = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.tool_name, "weather");
        assert_eq!(deserialized.group_id, "grp-42");
        assert!(!deserialized.explicit);
    }

    #[test]
    fn test_call_kind_serde_values() {
        let llm_json = serde_json::to_string(&CallKind::Llm).unwrap();
        let tool_json = serde_json::to_string(&CallKind::Tool).unwrap();

        assert_eq!(llm_json, "\"llm\"");
        assert_eq!(tool_json, "\"tool\"");
    }

    #[test]
    fn test_call_record_serde_roundtrip() {
        let now = Utc::now();
        let record = CallRecord {
            kind: CallKind::Tool,
            name: "calculator".to_string(),
            started_at: now,
            ended_at: Some(now),
            metadata_snapshot: Some(MetadataEnvelope {
                run_id: Uuid::new_v4(),
                agent_id: "agent-2".to_string(),
                parallel_hints: vec![],
                extensions: json!({}),
            }),
            output_tokens: None,
            prompt_tokens: None,
            total_tokens: None,
            model_name: None,
            tool_call_count: None,
        };

        let serialized = serde_json::to_string(&record).unwrap();
        let deserialized: CallRecord = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.kind, CallKind::Tool);
        assert_eq!(deserialized.name, "calculator");
        assert!(deserialized.metadata_snapshot.is_some());
    }

    #[test]
    fn test_call_record_skip_none_metadata() {
        let now = Utc::now();
        let record = CallRecord {
            kind: CallKind::Llm,
            name: "gpt-4".to_string(),
            started_at: now,
            ended_at: None,
            metadata_snapshot: None,
            output_tokens: None,
            prompt_tokens: None,
            total_tokens: None,
            model_name: None,
            tool_call_count: None,
        };

        let serialized = serde_json::to_string(&record).unwrap();
        assert!(
            !serialized.contains("metadata_snapshot"),
            "serialized JSON should not contain 'metadata_snapshot' when None, got: {serialized}"
        );
    }

    #[test]
    fn test_run_record_serde_roundtrip() {
        let now = Utc::now();
        let run = RunRecord {
            id: Uuid::new_v4(),
            agent_id: "agent-3".to_string(),
            calls: vec![
                CallRecord {
                    kind: CallKind::Llm,
                    name: "gpt-4".to_string(),
                    started_at: now,
                    ended_at: Some(now),
                    metadata_snapshot: None,
                    output_tokens: None,
                    prompt_tokens: None,
                    total_tokens: None,
                    model_name: None,
                    tool_call_count: None,
                },
                CallRecord {
                    kind: CallKind::Tool,
                    name: "search".to_string(),
                    started_at: now,
                    ended_at: None,
                    metadata_snapshot: None,
                    output_tokens: None,
                    prompt_tokens: None,
                    total_tokens: None,
                    model_name: None,
                    tool_call_count: None,
                },
            ],
            started_at: now,
            ended_at: Some(now),
        };

        let serialized = serde_json::to_string(&run).unwrap();
        let deserialized: RunRecord = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.id, run.id);
        assert_eq!(deserialized.agent_id, "agent-3");
        assert_eq!(deserialized.calls.len(), 2);
        assert_eq!(deserialized.started_at, run.started_at);
    }

    #[test]
    fn test_run_record_ended_at_none() {
        let now = Utc::now();
        let run = RunRecord {
            id: Uuid::new_v4(),
            agent_id: "agent-4".to_string(),
            calls: vec![],
            started_at: now,
            ended_at: None,
        };

        let serialized = serde_json::to_string(&run).unwrap();
        let deserialized: RunRecord = serde_json::from_str(&serialized).unwrap();

        assert!(deserialized.ended_at.is_none());
    }

    #[test]
    fn test_parallel_group_serde_roundtrip() {
        let group = ParallelGroup {
            group_id: "pg-1".to_string(),
            tool_names: vec!["search".to_string(), "fetch".to_string()],
        };

        let serialized = serde_json::to_string(&group).unwrap();
        let deserialized: ParallelGroup = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.group_id, "pg-1");
        assert_eq!(deserialized.tool_names, vec!["search", "fetch"]);
    }

    #[test]
    fn test_execution_plan_serde_roundtrip() {
        let plan = ExecutionPlan {
            agent_id: "agent-5".to_string(),
            parallel_groups: vec![ParallelGroup {
                group_id: "pg-2".to_string(),
                tool_names: vec!["a".to_string(), "b".to_string()],
            }],
            metadata_template: MetadataEnvelope {
                run_id: Uuid::new_v4(),
                agent_id: "agent-5".to_string(),
                parallel_hints: vec![],
                extensions: json!({"version": 1}),
            },
        };

        let serialized = serde_json::to_string(&plan).unwrap();
        let deserialized: ExecutionPlan = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.agent_id, "agent-5");
        assert_eq!(deserialized.parallel_groups.len(), 1);
        assert_eq!(deserialized.metadata_template.agent_id, "agent-5");
    }

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn test_types_are_send_sync() {
        assert_send_sync::<MetadataEnvelope>();
        assert_send_sync::<ParallelHint>();
        assert_send_sync::<RunRecord>();
        assert_send_sync::<CallRecord>();
        assert_send_sync::<ExecutionPlan>();
        assert_send_sync::<ParallelGroup>();
    }

    // -----------------------------------------------------------------------
    // AgentHints tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_agent_hints_serde_roundtrip() {
        let hints = AgentHints {
            osl: 256,
            iat: 150,
            priority: 2,
            latency_sensitivity: 0.8,
            prefix_id: "agent-1-d2".to_string(),
            total_requests: 5,
        };

        let value = serde_json::to_value(&hints).unwrap();
        let restored: AgentHints = serde_json::from_value(value).unwrap();

        assert_eq!(restored.osl, 256);
        assert_eq!(restored.iat, 150);
        assert_eq!(restored.priority, 2);
        assert!((restored.latency_sensitivity - 0.8).abs() < f64::EPSILON);
        assert_eq!(restored.prefix_id, "agent-1-d2");
        assert_eq!(restored.total_requests, 5);
    }

    #[test]
    fn test_agent_hints_send_sync() {
        assert_send_sync::<AgentHints>();
    }

    // -----------------------------------------------------------------------
    // HotCache tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_hot_cache_default_construction() {
        let cache = HotCache {
            plan: None,
            trie: None,
            agent_hints_default: None,
        };

        assert!(cache.plan.is_none());
        assert!(cache.trie.is_none());
        assert!(cache.agent_hints_default.is_none());
    }

    #[test]
    fn test_hot_cache_send_sync() {
        assert_send_sync::<HotCache>();
    }

    // -----------------------------------------------------------------------
    // CallRecord output_tokens tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_call_record_output_tokens_some() {
        let now = Utc::now();
        let record = CallRecord {
            kind: CallKind::Llm,
            name: "gpt-4".to_string(),
            started_at: now,
            ended_at: None,
            metadata_snapshot: None,
            output_tokens: Some(256),
            prompt_tokens: None,
            total_tokens: None,
            model_name: None,
            tool_call_count: None,
        };

        let value = serde_json::to_value(&record).unwrap();
        assert_eq!(value["output_tokens"], json!(256));
    }

    #[test]
    fn test_call_record_output_tokens_none_omitted() {
        let now = Utc::now();
        let record = CallRecord {
            kind: CallKind::Llm,
            name: "gpt-4".to_string(),
            started_at: now,
            ended_at: None,
            metadata_snapshot: None,
            output_tokens: None,
            prompt_tokens: None,
            total_tokens: None,
            model_name: None,
            tool_call_count: None,
        };

        let serialized = serde_json::to_string(&record).unwrap();
        assert!(
            !serialized.contains("output_tokens"),
            "serialized JSON should not contain 'output_tokens' when None, got: {serialized}"
        );
    }

    #[test]
    fn test_call_record_backward_compat() {
        // Simulate an old JSON without output_tokens field
        let old_json = json!({
            "kind": "llm",
            "name": "gpt-4",
            "started_at": "2026-03-31T12:00:00Z",
            "ended_at": null,
            "metadata_snapshot": null
        });

        let record: CallRecord =
            serde_json::from_value(old_json).expect("should deserialize old format");
        assert!(
            record.output_tokens.is_none(),
            "missing output_tokens should default to None"
        );
    }

    // -----------------------------------------------------------------------
    // New telemetry field tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_call_record_new_fields_serde_roundtrip() {
        let now = Utc::now();
        let record = CallRecord {
            kind: CallKind::Llm,
            name: "gpt-4".to_string(),
            started_at: now,
            ended_at: Some(now),
            metadata_snapshot: None,
            output_tokens: Some(100),
            prompt_tokens: Some(100),
            total_tokens: Some(150),
            model_name: Some("gpt-4".to_string()),
            tool_call_count: Some(2),
        };

        let serialized = serde_json::to_string(&record).unwrap();
        let deserialized: CallRecord = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.prompt_tokens, Some(100));
        assert_eq!(deserialized.total_tokens, Some(150));
        assert_eq!(deserialized.model_name, Some("gpt-4".to_string()));
        assert_eq!(deserialized.tool_call_count, Some(2));
    }

    #[test]
    fn test_call_record_backward_compat_new_fields() {
        // Old format JSON missing all new fields
        let old_json = r#"{"kind":"llm","name":"gpt-4","started_at":"2026-03-31T12:00:00Z","ended_at":null,"metadata_snapshot":null}"#;

        let record: CallRecord =
            serde_json::from_str(old_json).expect("should deserialize old format");
        assert!(
            record.prompt_tokens.is_none(),
            "missing prompt_tokens should default to None"
        );
        assert!(
            record.total_tokens.is_none(),
            "missing total_tokens should default to None"
        );
        assert!(
            record.model_name.is_none(),
            "missing model_name should default to None"
        );
        assert!(
            record.tool_call_count.is_none(),
            "missing tool_call_count should default to None"
        );
    }

    #[test]
    fn test_call_record_new_fields_none_omitted() {
        let now = Utc::now();
        let record = CallRecord {
            kind: CallKind::Llm,
            name: "gpt-4".to_string(),
            started_at: now,
            ended_at: None,
            metadata_snapshot: None,
            output_tokens: None,
            prompt_tokens: None,
            total_tokens: None,
            model_name: None,
            tool_call_count: None,
        };

        let serialized = serde_json::to_string(&record).unwrap();
        assert!(
            !serialized.contains("prompt_tokens"),
            "prompt_tokens should be omitted when None, got: {serialized}"
        );
        assert!(
            !serialized.contains("total_tokens"),
            "total_tokens should be omitted when None, got: {serialized}"
        );
        assert!(
            !serialized.contains("model_name"),
            "model_name should be omitted when None, got: {serialized}"
        );
        assert!(
            !serialized.contains("tool_call_count"),
            "tool_call_count should be omitted when None, got: {serialized}"
        );
    }
}
