// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared scope data types.

use bitflags::bitflags;
use serde::{Deserialize, Serialize};

use crate::api::llm::LlmAttributes;
use crate::api::tool::ToolAttributes;

bitflags! {
    /// Bitflags that modify scope behavior and observability.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct ScopeAttributes: u32 {
        /// Marks the scope as running in parallel with sibling work.
        const PARALLEL    = 0b01;
        /// Marks the scope as safe to move across execution contexts.
        const RELOCATABLE = 0b10;
    }
}

/// Semantic category attached to a scope lifecycle span.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ScopeType {
    /// A top-level agent or workflow scope.
    Agent,
    /// A generic function or application step.
    Function,
    /// A tool lifecycle scope.
    Tool,
    /// An LLM lifecycle scope.
    Llm,
    /// A retrieval step such as document search.
    Retriever,
    /// An embedding generation step.
    Embedder,
    /// A reranking step.
    Reranker,
    /// A guardrail or validation step.
    Guardrail,
    /// An evaluation or scoring step.
    Evaluator,
    /// A caller-defined custom scope category.
    Custom,
    /// A fallback for unknown or unsupported scope categories.
    Unknown,
}

impl ScopeType {
    /// Return the stable lowercase string form used for encoded scope types.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Agent => "agent",
            Self::Function => "function",
            Self::Tool => "tool",
            Self::Llm => "llm",
            Self::Retriever => "retriever",
            Self::Embedder => "embedder",
            Self::Reranker => "reranker",
            Self::Guardrail => "guardrail",
            Self::Evaluator => "evaluator",
            Self::Custom => "custom",
            Self::Unknown => "unknown",
        }
    }
}

/// Attribute bitflags attached to a concrete handle kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HandleAttributes {
    /// Scope-specific attributes.
    Scope(ScopeAttributes),
    /// Tool-specific attributes.
    Tool(ToolAttributes),
    /// LLM-specific attributes.
    Llm(LlmAttributes),
}
