// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub type Json = serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataEnvelope {
    pub run_id: Uuid,
    pub agent_id: String,
    pub parallel_hints: Vec<ParallelHint>,
    pub extensions: Json,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelHint {
    pub tool_name: String,
    pub group_id: String,
    pub explicit: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentHints {
    pub osl: u32,
    pub iat: u32,
    pub priority: i32,
    pub latency_sensitivity: f64,
    pub prefix_id: String,
    pub total_requests: u32,
}
