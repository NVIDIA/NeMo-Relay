// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

use crate::types::metadata::AgentHints;
use crate::types::plan::ExecutionPlan;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotCache {
    pub plan: Option<ExecutionPlan>,
    pub trie: Option<crate::trie::data_models::PredictionTrieNode>,
    pub agent_hints_default: Option<AgentHints>,
}
