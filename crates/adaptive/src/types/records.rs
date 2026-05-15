// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Run and call record types collected by the adaptive telemetry pipeline.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::types::metadata::MetadataEnvelope;

/// Kind of runtime call captured in adaptive telemetry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CallKind {
    /// LLM or model-provider invocation.
    Llm,
    /// Tool invocation.
    Tool,
}

/// Telemetry record for a single tool or LLM call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallRecord {
    /// Category of call represented by this record.
    pub kind: CallKind,
    /// Logical tool or provider name.
    pub name: String,
    /// Timestamp when the call began.
    pub started_at: DateTime<Utc>,
    /// Timestamp when the call finished, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ended_at: Option<DateTime<Utc>>,
    /// Adaptive metadata snapshot associated with the call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata_snapshot: Option<MetadataEnvelope>,
    /// Output token count reported by the provider, when available.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub output_tokens: Option<u32>,
    /// Prompt token count reported by the provider, when available.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub prompt_tokens: Option<u32>,
    /// Total token count reported by the provider, when available.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub total_tokens: Option<u32>,
    /// Normalized model name associated with the call, when available.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub model_name: Option<String>,
    /// Number of tool calls issued by the provider, when available.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub tool_call_count: Option<u32>,
    /// Annotated request captured for Adaptive Cache Governor (ACG) analysis,
    /// when available.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub annotated_request: Option<Arc<nemo_flow::codec::request::AnnotatedLlmRequest>>,
    /// Annotated response captured for Adaptive Cache Governor (ACG) analysis,
    /// when available.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub annotated_response: Option<Arc<nemo_flow::codec::response::AnnotatedLlmResponse>>,
}

/// Telemetry record for one observed agent run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRecord {
    /// Unique run identifier.
    pub id: Uuid,
    /// Agent identifier that produced the run.
    pub agent_id: String,
    /// Calls observed during the run.
    pub calls: Vec<CallRecord>,
    /// Timestamp when the run began.
    pub started_at: DateTime<Utc>,
    /// Timestamp when the run finished, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ended_at: Option<DateTime<Utc>>,
}
