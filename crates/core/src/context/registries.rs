// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use crate::context::callbacks::{
    EventSubscriberFn, LlmConditionalFn, LlmExecutionFn, LlmRequestInterceptFn,
    LlmSanitizeRequestFn, LlmSanitizeResponseFn, LlmStreamExecutionFn, ToolConditionalFn,
    ToolExecutionFn, ToolInterceptFn, ToolSanitizeFn,
};
use crate::registry::SortedRegistry;
use crate::types::middleware::{ExecutionIntercept, GuardrailEntry, Intercept};

pub struct ScopeLocalRegistries {
    pub tool_sanitize_request_guardrails: SortedRegistry<GuardrailEntry<ToolSanitizeFn>>,
    pub tool_sanitize_response_guardrails: SortedRegistry<GuardrailEntry<ToolSanitizeFn>>,
    pub tool_conditional_execution_guardrails: SortedRegistry<GuardrailEntry<ToolConditionalFn>>,
    pub tool_request_intercepts: SortedRegistry<Intercept<ToolInterceptFn>>,
    pub tool_execution_intercepts: SortedRegistry<ExecutionIntercept<ToolExecutionFn>>,
    pub llm_sanitize_request_guardrails: SortedRegistry<GuardrailEntry<LlmSanitizeRequestFn>>,
    pub llm_sanitize_response_guardrails: SortedRegistry<GuardrailEntry<LlmSanitizeResponseFn>>,
    pub llm_conditional_execution_guardrails: SortedRegistry<GuardrailEntry<LlmConditionalFn>>,
    pub llm_request_intercepts: SortedRegistry<Intercept<LlmRequestInterceptFn>>,
    pub llm_execution_intercepts: SortedRegistry<ExecutionIntercept<LlmExecutionFn>>,
    pub llm_stream_execution_intercepts: SortedRegistry<ExecutionIntercept<LlmStreamExecutionFn>>,
    pub event_subscribers: HashMap<String, EventSubscriberFn>,
}

impl ScopeLocalRegistries {
    pub fn new() -> Self {
        Self {
            tool_sanitize_request_guardrails: SortedRegistry::new(|entry| entry.priority),
            tool_sanitize_response_guardrails: SortedRegistry::new(|entry| entry.priority),
            tool_conditional_execution_guardrails: SortedRegistry::new(|entry| entry.priority),
            tool_request_intercepts: SortedRegistry::new(|entry| entry.priority),
            tool_execution_intercepts: SortedRegistry::new(|entry| entry.priority),
            llm_sanitize_request_guardrails: SortedRegistry::new(|entry| entry.priority),
            llm_sanitize_response_guardrails: SortedRegistry::new(|entry| entry.priority),
            llm_conditional_execution_guardrails: SortedRegistry::new(|entry| entry.priority),
            llm_request_intercepts: SortedRegistry::new(|entry| entry.priority),
            llm_execution_intercepts: SortedRegistry::new(|entry| entry.priority),
            llm_stream_execution_intercepts: SortedRegistry::new(|entry| entry.priority),
            event_subscribers: HashMap::new(),
        }
    }
}

impl Default for ScopeLocalRegistries {
    fn default() -> Self {
        Self::new()
    }
}

pub fn merge_guardrail_entries<'a, F>(
    global: &'a SortedRegistry<GuardrailEntry<F>>,
    scope_locals: &'a [&'a SortedRegistry<GuardrailEntry<F>>],
) -> Vec<&'a GuardrailEntry<F>> {
    let mut all = Vec::new();
    all.extend(global.sorted_values());
    for registry in scope_locals {
        all.extend(registry.sorted_values());
    }
    all.sort_by_key(|entry| entry.priority);
    all
}

pub fn merge_intercept_entries<'a, F>(
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

pub fn merge_execution_intercept_callables<F: Clone>(
    global: &SortedRegistry<ExecutionIntercept<F>>,
    scope_locals: &[&SortedRegistry<ExecutionIntercept<F>>],
) -> Vec<(F, i32)> {
    let mut all = Vec::new();
    for entry in global.sorted_values() {
        all.push((entry.callable.clone(), entry.priority));
    }
    for registry in scope_locals {
        for entry in registry.sorted_values() {
            all.push((entry.callable.clone(), entry.priority));
        }
    }
    all.sort_by_key(|(_, priority)| *priority);
    all
}
