// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared runtime-registration discovery data types.

use serde::{Deserialize, Serialize};

/// A global runtime registration surface that can be selected by a
/// conditional middleware guardrail.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeRegistrationKind {
    /// Lifecycle event subscriber.
    Subscriber,
    /// Event metadata injector.
    EventMetadataInjector,
    /// Mark event sanitizer.
    MarkSanitizeGuardrail,
    /// Scope-start event sanitizer.
    ScopeSanitizeStartGuardrail,
    /// Scope-end event sanitizer.
    ScopeSanitizeEndGuardrail,
    /// Tool request observability sanitizer.
    ToolSanitizeRequestGuardrail,
    /// Tool response observability sanitizer.
    ToolSanitizeResponseGuardrail,
    /// Tool execution gate.
    ToolConditionalExecutionGuardrail,
    /// Tool request intercept.
    ToolRequestIntercept,
    /// Tool execution intercept.
    ToolExecutionIntercept,
    /// LLM request observability sanitizer.
    LlmSanitizeRequestGuardrail,
    /// LLM response observability sanitizer.
    LlmSanitizeResponseGuardrail,
    /// LLM execution gate.
    LlmConditionalExecutionGuardrail,
    /// LLM request intercept.
    LlmRequestIntercept,
    /// Non-streaming LLM execution intercept.
    LlmExecutionIntercept,
    /// Streaming LLM execution intercept.
    LlmStreamExecutionIntercept,
}

impl RuntimeRegistrationKind {
    /// Return the stable snake-case binding name for this kind.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Subscriber => "subscriber",
            Self::EventMetadataInjector => "event_metadata_injector",
            Self::MarkSanitizeGuardrail => "mark_sanitize_guardrail",
            Self::ScopeSanitizeStartGuardrail => "scope_sanitize_start_guardrail",
            Self::ScopeSanitizeEndGuardrail => "scope_sanitize_end_guardrail",
            Self::ToolSanitizeRequestGuardrail => "tool_sanitize_request_guardrail",
            Self::ToolSanitizeResponseGuardrail => "tool_sanitize_response_guardrail",
            Self::ToolConditionalExecutionGuardrail => "tool_conditional_execution_guardrail",
            Self::ToolRequestIntercept => "tool_request_intercept",
            Self::ToolExecutionIntercept => "tool_execution_intercept",
            Self::LlmSanitizeRequestGuardrail => "llm_sanitize_request_guardrail",
            Self::LlmSanitizeResponseGuardrail => "llm_sanitize_response_guardrail",
            Self::LlmConditionalExecutionGuardrail => "llm_conditional_execution_guardrail",
            Self::LlmRequestIntercept => "llm_request_intercept",
            Self::LlmExecutionIntercept => "llm_execution_intercept",
            Self::LlmStreamExecutionIntercept => "llm_stream_execution_intercept",
        }
    }
}

/// The owner category reported for a global runtime registration.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeRegistrationOwnerKind {
    /// Registration installed directly by Relay core.
    Core,
    /// Registration installed through a process-global public API.
    GlobalApi,
    /// Registration installed by a Relay plugin component.
    Plugin,
}

/// Discovery metadata describing the owner of a runtime registration.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RuntimeRegistrationOwner {
    /// Owner category.
    pub kind: RuntimeRegistrationOwnerKind,
    /// Plugin kind for plugin-owned registrations.
    pub plugin_kind: Option<String>,
    /// One-based ordinal for a Relay-created plugin component, when applicable.
    pub component_ordinal: Option<u32>,
}

/// Structured identity for a global gateable runtime registration.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RuntimeRegistrationIdentity {
    /// Registration surface.
    pub kind: RuntimeRegistrationKind,
    /// Name authored by the registration owner.
    pub local_name: String,
    /// Runtime-qualified name used for gate matching.
    pub effective_name: String,
    /// Registration owner metadata.
    pub owner: RuntimeRegistrationOwner,
}
