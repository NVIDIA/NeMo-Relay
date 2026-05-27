// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Scope-local middleware registries and registry-merging helpers.
//!
//! Scope-local middleware behaves like an overlay on top of the process-global
//! runtime state. The types and helpers in this module store those per-scope
//! registrations and resolve the merged ordering used by the execution layer.

use std::collections::HashMap;

use crate::api::registry::{ExecutionIntercept, Guardrail, Intercept};
use crate::api::runtime::{
    EventSubscriberFn, LlmConditionalFn, LlmExecutionFn, LlmRequestInterceptFn,
    LlmSanitizeRequestFn, LlmSanitizeResponseFn, LlmStreamExecutionFn, ToolConditionalFn,
    ToolExecutionFn, ToolInterceptFn, ToolSanitizeFn,
};
use crate::registry::SortedRegistry;
use uuid::Uuid;

/// Scope-owned middleware registries and subscribers.
///
/// Each active scope can own its own set of guardrails, intercepts, and event
/// subscribers. These registrations are merged with the global runtime
/// registries when the runtime resolves the effective middleware chain for a
/// tool or LLM call executed inside that scope.
pub(crate) struct ScopeLocalRegistries {
    /// Tool request sanitizers applied to emitted tool-start payloads.
    pub(crate) tool_sanitize_request_guardrails: SortedRegistry<Guardrail<ToolSanitizeFn>>,
    /// Tool response sanitizers applied to emitted tool-end payloads.
    pub(crate) tool_sanitize_response_guardrails: SortedRegistry<Guardrail<ToolSanitizeFn>>,
    /// Tool guardrails that can reject execution before the callback runs.
    pub(crate) tool_conditional_execution_guardrails: SortedRegistry<Guardrail<ToolConditionalFn>>,
    /// Tool request intercepts that can rewrite arguments before execution.
    pub(crate) tool_request_intercepts: SortedRegistry<Intercept<ToolInterceptFn>>,
    /// Tool execution intercepts that wrap or replace callback execution.
    pub(crate) tool_execution_intercepts: SortedRegistry<ExecutionIntercept<ToolExecutionFn>>,
    /// LLM request sanitizers applied to emitted LLM-start payloads.
    pub(crate) llm_sanitize_request_guardrails: SortedRegistry<Guardrail<LlmSanitizeRequestFn>>,
    /// LLM response sanitizers applied to emitted LLM-end payloads.
    pub(crate) llm_sanitize_response_guardrails: SortedRegistry<Guardrail<LlmSanitizeResponseFn>>,
    /// LLM guardrails that can reject execution before the provider callback runs.
    pub(crate) llm_conditional_execution_guardrails: SortedRegistry<Guardrail<LlmConditionalFn>>,
    /// LLM request intercepts that can rewrite or annotate requests.
    pub(crate) llm_request_intercepts: SortedRegistry<Intercept<LlmRequestInterceptFn>>,
    /// Non-streaming LLM execution intercepts that wrap callback execution.
    pub(crate) llm_execution_intercepts: SortedRegistry<ExecutionIntercept<LlmExecutionFn>>,
    /// Streaming LLM execution intercepts that wrap stream-producing callbacks.
    pub(crate) llm_stream_execution_intercepts:
        SortedRegistry<ExecutionIntercept<LlmStreamExecutionFn>>,
    /// Scope-local lifecycle subscribers visible while the owning scope is active.
    pub(crate) event_subscribers: HashMap<String, EventSubscriberFn>,
    /// Anonymous scope-local lifecycle subscribers owned by closeable handles.
    pub(crate) anonymous_event_subscribers: HashMap<Uuid, EventSubscriberFn>,
}

impl ScopeLocalRegistries {
    /// Create an empty set of scope-local registries.
    ///
    /// # Returns
    /// A [`ScopeLocalRegistries`] value with no registered guardrails,
    /// intercepts, or subscribers.
    pub(crate) fn new() -> Self {
        Self {
            tool_sanitize_request_guardrails: SortedRegistry::new(),
            tool_sanitize_response_guardrails: SortedRegistry::new(),
            tool_conditional_execution_guardrails: SortedRegistry::new(),
            tool_request_intercepts: SortedRegistry::new(),
            tool_execution_intercepts: SortedRegistry::new(),
            llm_sanitize_request_guardrails: SortedRegistry::new(),
            llm_sanitize_response_guardrails: SortedRegistry::new(),
            llm_conditional_execution_guardrails: SortedRegistry::new(),
            llm_request_intercepts: SortedRegistry::new(),
            llm_execution_intercepts: SortedRegistry::new(),
            llm_stream_execution_intercepts: SortedRegistry::new(),
            event_subscribers: HashMap::new(),
            anonymous_event_subscribers: HashMap::new(),
        }
    }
}

impl Default for ScopeLocalRegistries {
    fn default() -> Self {
        Self::new()
    }
}

/// Merge global and scope-local guardrail entries into one priority-sorted list.
///
/// This helper snapshots the guardrail entries visible to the current
/// execution, including the process-global registry and any scope-local
/// overlays collected from the active scope stack.
///
/// # Parameters
/// - `global`: Process-global guardrail registry.
/// - `scope_locals`: Scope-local registries collected from active scopes.
///
/// # Returns
/// A vector of guardrail entries sorted by ascending priority.
pub(crate) fn merge_guardrail_entries<'a, F>(
    global: &'a SortedRegistry<Guardrail<F>>,
    scope_locals: &'a [&'a SortedRegistry<Guardrail<F>>],
) -> Vec<&'a Guardrail<F>> {
    let mut all = Vec::new();
    all.extend(global.sorted_values());
    for registry in scope_locals {
        all.extend(registry.sorted_values());
    }
    all.sort_by_key(|entry| entry.priority);
    all
}

/// Merge global and scope-local intercept entries into one priority-sorted list.
///
/// # Parameters
/// - `global`: Process-global intercept registry.
/// - `scope_locals`: Scope-local registries collected from active scopes.
///
/// # Returns
/// A vector of intercept entries sorted by ascending priority.
pub(crate) fn merge_intercept_entries<'a, F>(
    global: &'a SortedRegistry<Intercept<F>>,
    scope_locals: &'a [&'a SortedRegistry<Intercept<F>>],
) -> Vec<&'a Intercept<F>> {
    let mut all = Vec::new();
    all.extend(global.sorted_values());
    for registry in scope_locals {
        all.extend(registry.sorted_values());
    }
    all.sort_by_key(|entry| entry.priority);
    all
}

/// Collect execution intercept callables with their resolved priorities.
///
/// Execution intercepts are cloned out of their registries because the runtime
/// builds a composed continuation chain from owned callables.
///
/// # Parameters
/// - `global`: Process-global execution intercept registry.
/// - `scope_locals`: Scope-local registries collected from active scopes.
///
/// # Returns
/// A vector of `(callable, priority)` pairs sorted by ascending priority.
pub(crate) fn merge_execution_intercept_callables<F: Clone>(
    global: &SortedRegistry<ExecutionIntercept<F>>,
    scope_locals: &[&SortedRegistry<ExecutionIntercept<F>>],
) -> Vec<(F, i32)> {
    let mut all = Vec::new();
    for entry in global.sorted_values() {
        all.push((entry.payload.clone(), entry.priority));
    }
    for registry in scope_locals {
        for entry in registry.sorted_values() {
            all.push((entry.payload.clone(), entry.priority));
        }
    }
    all.sort_by_key(|(_, priority)| *priority);
    all
}
