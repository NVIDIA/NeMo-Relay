// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use bitflags::bitflags;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::types::llm::LLMAttributes;
use crate::types::tool::ToolAttributes;

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct ScopeAttributes: u32 {
        const PARALLEL    = 0b01;
        const RELOCATABLE = 0b10;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ScopeType {
    Agent,
    Function,
    Tool,
    Llm,
    Retriever,
    Embedder,
    Reranker,
    Guardrail,
    Evaluator,
    Custom,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HandleAttributes {
    Scope(ScopeAttributes),
    Tool(ToolAttributes),
    Llm(LLMAttributes),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopeHandle {
    pub uuid: Uuid,
    pub scope_type: ScopeType,
    pub name: String,
    pub data: Option<crate::json::Json>,
    pub metadata: Option<crate::json::Json>,
    pub attributes: ScopeAttributes,
    pub parent_uuid: Option<Uuid>,
}

impl ScopeHandle {
    pub fn new(
        name: String,
        scope_type: ScopeType,
        attributes: ScopeAttributes,
        parent_uuid: Option<Uuid>,
        data: Option<crate::json::Json>,
        metadata: Option<crate::json::Json>,
    ) -> Self {
        Self {
            uuid: Uuid::now_v7(),
            scope_type,
            name,
            data,
            metadata,
            attributes,
            parent_uuid,
        }
    }
}
