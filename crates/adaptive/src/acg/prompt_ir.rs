// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prompt Intermediate Representation (IR) types for the Adaptive Cache
//! Governor (ACG) system.
//!
//! The Prompt IR decomposes LLM conversations into addressable blocks
//! with structural metadata for cache analysis and prompt rewriting.
//! This is deliberately different from the message-oriented
//! `AnnotatedLlmRequest` in core -- the IR flattens the hierarchy
//! into a sequence of blocks, each carrying provenance, sensitivity,
//! and stability metadata.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ===================================================================
// Span identifiers
// ===================================================================

/// Stable span identifier for addressable prompt blocks.
///
/// A newtype wrapper around `String` that provides `Hash` and `Eq`
/// so `SpanId` values can be used as keys in `HashMap` / `HashSet`.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct SpanId(pub String);

// ===================================================================
// Provenance, sensitivity, and role labels
// ===================================================================

/// Origin label for a prompt block.
///
/// Tracks where the content came from so downstream phases can apply
/// provenance-specific caching and sharing rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProvenanceLabel {
    /// System-level instructions (e.g., system prompt).
    System,
    /// Developer-authored prompt templates.
    Developer,
    /// End-user input.
    User,
    /// Tool output / function call results.
    Tool,
    /// Retrieved context (RAG).
    Retrieval,
    /// Agent memory / conversation history.
    Memory,
}

/// Sensitivity classification for a prompt block.
///
/// Gates downstream sharing decisions. Defaults to `Public` -- promotion
/// to `Private` or `Restricted` requires explicit assignment (T-04-02).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SensitivityLabel {
    /// Content may be shared freely.
    #[default]
    Public,
    /// Content contains private information.
    Private,
    /// Content is restricted and must not leave its originating scope.
    Restricted,
}

/// Role of a prompt block within the conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptRole {
    /// System prompt.
    System,
    /// User message.
    User,
    /// Assistant (model) message.
    Assistant,
    /// Tool / function call result.
    Tool,
}

/// Content type discriminant for a prompt block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockContentType {
    /// Plain text content.
    Text,
    /// Tool/function definition schema (JSON Schema).
    ToolSchema,
    /// Tool/function call result.
    ToolResult,
    /// Structured output (e.g., JSON output).
    StructuredOutput,
    /// Image content (base64 or URL reference).
    Image,
}

// ===================================================================
// Tokenization metadata
// ===================================================================

/// Token count metadata for a prompt block.
///
/// Records the model family and token count so that downstream phases
/// can compute cache-aware token budgets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenizationMetadata {
    /// Model family used for tokenization (e.g., "claude", "gpt").
    pub model_family: String,
    /// Number of tokens in the block content.
    pub token_count: u32,
}

// ===================================================================
// PromptBlock
// ===================================================================

/// A single addressable block within the Prompt IR.
///
/// Each block carries provenance, sensitivity, and content type metadata
/// along with an optional token count. Blocks are sequenced by
/// `sequence_index` within the parent [`PromptIR`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromptBlock {
    /// Stable span identifier for this block.
    pub span_id: SpanId,
    /// Zero-based index in the prompt sequence.
    pub sequence_index: u32,
    /// Conversation role of this block.
    pub role: PromptRole,
    /// Raw content of the block.
    pub content: String,
    /// Content type discriminant.
    pub content_type: BlockContentType,
    /// Origin of the content.
    pub provenance: ProvenanceLabel,
    /// Sensitivity classification (defaults to `Public`).
    #[serde(default)]
    pub sensitivity: SensitivityLabel,
    /// Optional tokenization metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub token_metadata: Option<TokenizationMetadata>,
}

// ===================================================================
// ToolSchemaHash
// ===================================================================

/// Hash fingerprint of a tool schema definition.
///
/// Used to detect when the active toolset changes across requests,
/// which invalidates tool-schema blocks in the cache.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ToolSchemaHash {
    /// Name of the tool.
    pub tool_name: String,
    /// Hash of the tool's JSON Schema definition.
    pub schema_hash: String,
}

// ===================================================================
// PromptIR
// ===================================================================

/// Prompt Intermediate Representation -- the full decomposed prompt.
///
/// A `PromptIR` is produced by the IR construction phase (Phase 6) from
/// an `AnnotatedLlmRequest`. It flattens the message hierarchy into an
/// ordered sequence of [`PromptBlock`]s, each carrying structural metadata
/// for cache analysis and rewriting.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromptIR {
    /// Unique identifier for this IR instance.
    pub ir_id: Uuid,
    /// Ordered sequence of prompt blocks.
    pub blocks: Vec<PromptBlock>,
    /// Hashes of tool schemas active at IR creation time.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub tool_schema_hashes: Option<Vec<ToolSchemaHash>>,
    /// Identifier of the structured output schema, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub structured_output_schema_id: Option<String>,
    /// Optional hash of the source `AnnotatedLlmRequest` for traceability.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub source_request_hash: Option<String>,
    /// When this IR was created.
    pub created_at: DateTime<Utc>,
}

#[cfg(test)]
#[path = "../../tests/unit/acg/prompt_ir_tests.rs"]
mod tests;
