// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::api::shared::ensure_runtime_owner;
use crate::context::callbacks::{
    LlmConditionalFn, LlmExecutionFn, LlmRequestInterceptFn, LlmSanitizeRequestFn,
    LlmSanitizeResponseFn, LlmStreamExecutionFn, ToolConditionalFn, ToolExecutionFn,
    ToolInterceptFn, ToolSanitizeFn,
};
use crate::context::global::global_context;
use crate::context::scope_stack::current_scope_stack;
use crate::error::{FlowError, Result};
use crate::types::middleware::{ExecutionIntercept, GuardrailEntry, Intercept};

macro_rules! global_guardrail_registry_api {
    ($(#[$meta:meta])* $register_name:ident, $deregister_name:ident, $field:ident, $fn_type:ty) => {
        $(#[$meta])*
        pub fn $register_name(name: &str, priority: i32, guardrail: $fn_type) -> Result<()> {
            ensure_runtime_owner()?;
            let context = global_context();
            let mut state = context
                .write()
                .map_err(|error| FlowError::Internal(error.to_string()))?;
            state
                .$field
                .register(name.to_string(), GuardrailEntry { priority, guardrail })
                .map_err(FlowError::AlreadyExists)
        }

        pub fn $deregister_name(name: &str) -> Result<bool> {
            ensure_runtime_owner()?;
            let context = global_context();
            let mut state = context
                .write()
                .map_err(|error| FlowError::Internal(error.to_string()))?;
            Ok(state.$field.deregister(name))
        }
    };
}

macro_rules! global_intercept_registry_api {
    ($(#[$meta:meta])* $register_name:ident, $deregister_name:ident, $field:ident, $fn_type:ty) => {
        $(#[$meta])*
        pub fn $register_name(
            name: &str,
            priority: i32,
            break_chain: bool,
            callable: $fn_type,
        ) -> Result<()> {
            ensure_runtime_owner()?;
            let context = global_context();
            let mut state = context
                .write()
                .map_err(|error| FlowError::Internal(error.to_string()))?;
            state
                .$field
                .register(
                    name.to_string(),
                    Intercept {
                        priority,
                        break_chain,
                        callable,
                    },
                )
                .map_err(FlowError::AlreadyExists)
        }

        pub fn $deregister_name(name: &str) -> Result<bool> {
            ensure_runtime_owner()?;
            let context = global_context();
            let mut state = context
                .write()
                .map_err(|error| FlowError::Internal(error.to_string()))?;
            Ok(state.$field.deregister(name))
        }
    };
}

macro_rules! global_execution_registry_api {
    ($(#[$meta:meta])* $register_name:ident, $deregister_name:ident, $field:ident, $fn_type:ty) => {
        $(#[$meta])*
        pub fn $register_name(name: &str, priority: i32, callable: $fn_type) -> Result<()> {
            ensure_runtime_owner()?;
            let context = global_context();
            let mut state = context
                .write()
                .map_err(|error| FlowError::Internal(error.to_string()))?;
            state
                .$field
                .register(name.to_string(), ExecutionIntercept { priority, callable })
                .map_err(FlowError::AlreadyExists)
        }

        pub fn $deregister_name(name: &str) -> Result<bool> {
            ensure_runtime_owner()?;
            let context = global_context();
            let mut state = context
                .write()
                .map_err(|error| FlowError::Internal(error.to_string()))?;
            Ok(state.$field.deregister(name))
        }
    };
}

macro_rules! scope_guardrail_registry_api {
    ($(#[$meta:meta])* $register_name:ident, $deregister_name:ident, $field:ident, $fn_type:ty) => {
        $(#[$meta])*
        pub fn $register_name(
            scope_uuid: &uuid::Uuid,
            name: &str,
            priority: i32,
            guardrail: $fn_type,
        ) -> Result<()> {
            ensure_runtime_owner()?;
            let scope_stack = current_scope_stack();
            let mut guard = scope_stack.write().expect("scope stack lock poisoned");
            let registries = guard
                .local_registries_mut(scope_uuid)
                .ok_or_else(|| FlowError::NotFound(format!("scope {scope_uuid} not found")))?;
            registries
                .$field
                .register(name.to_string(), GuardrailEntry { priority, guardrail })
                .map_err(FlowError::AlreadyExists)
        }

        pub fn $deregister_name(scope_uuid: &uuid::Uuid, name: &str) -> Result<bool> {
            ensure_runtime_owner()?;
            let scope_stack = current_scope_stack();
            let mut guard = scope_stack.write().expect("scope stack lock poisoned");
            let registries = guard
                .local_registries_mut(scope_uuid)
                .ok_or_else(|| FlowError::NotFound(format!("scope {scope_uuid} not found")))?;
            Ok(registries.$field.deregister(name))
        }
    };
}

macro_rules! scope_intercept_registry_api {
    ($(#[$meta:meta])* $register_name:ident, $deregister_name:ident, $field:ident, $fn_type:ty) => {
        $(#[$meta])*
        pub fn $register_name(
            scope_uuid: &uuid::Uuid,
            name: &str,
            priority: i32,
            break_chain: bool,
            callable: $fn_type,
        ) -> Result<()> {
            ensure_runtime_owner()?;
            let scope_stack = current_scope_stack();
            let mut guard = scope_stack.write().expect("scope stack lock poisoned");
            let registries = guard
                .local_registries_mut(scope_uuid)
                .ok_or_else(|| FlowError::NotFound(format!("scope {scope_uuid} not found")))?;
            registries
                .$field
                .register(
                    name.to_string(),
                    Intercept {
                        priority,
                        break_chain,
                        callable,
                    },
                )
                .map_err(FlowError::AlreadyExists)
        }

        pub fn $deregister_name(scope_uuid: &uuid::Uuid, name: &str) -> Result<bool> {
            ensure_runtime_owner()?;
            let scope_stack = current_scope_stack();
            let mut guard = scope_stack.write().expect("scope stack lock poisoned");
            let registries = guard
                .local_registries_mut(scope_uuid)
                .ok_or_else(|| FlowError::NotFound(format!("scope {scope_uuid} not found")))?;
            Ok(registries.$field.deregister(name))
        }
    };
}

macro_rules! scope_execution_registry_api {
    ($(#[$meta:meta])* $register_name:ident, $deregister_name:ident, $field:ident, $fn_type:ty) => {
        $(#[$meta])*
        pub fn $register_name(
            scope_uuid: &uuid::Uuid,
            name: &str,
            priority: i32,
            callable: $fn_type,
        ) -> Result<()> {
            ensure_runtime_owner()?;
            let scope_stack = current_scope_stack();
            let mut guard = scope_stack.write().expect("scope stack lock poisoned");
            let registries = guard
                .local_registries_mut(scope_uuid)
                .ok_or_else(|| FlowError::NotFound(format!("scope {scope_uuid} not found")))?;
            registries
                .$field
                .register(name.to_string(), ExecutionIntercept { priority, callable })
                .map_err(FlowError::AlreadyExists)
        }

        pub fn $deregister_name(scope_uuid: &uuid::Uuid, name: &str) -> Result<bool> {
            ensure_runtime_owner()?;
            let scope_stack = current_scope_stack();
            let mut guard = scope_stack.write().expect("scope stack lock poisoned");
            let registries = guard
                .local_registries_mut(scope_uuid)
                .ok_or_else(|| FlowError::NotFound(format!("scope {scope_uuid} not found")))?;
            Ok(registries.$field.deregister(name))
        }
    };
}

global_guardrail_registry_api!(
    register_tool_sanitize_request_guardrail,
    deregister_tool_sanitize_request_guardrail,
    tool_sanitize_request_guardrails,
    ToolSanitizeFn
);
global_guardrail_registry_api!(
    register_tool_sanitize_response_guardrail,
    deregister_tool_sanitize_response_guardrail,
    tool_sanitize_response_guardrails,
    ToolSanitizeFn
);
global_guardrail_registry_api!(
    register_tool_conditional_execution_guardrail,
    deregister_tool_conditional_execution_guardrail,
    tool_conditional_execution_guardrails,
    ToolConditionalFn
);
global_intercept_registry_api!(
    register_tool_request_intercept,
    deregister_tool_request_intercept,
    tool_request_intercepts,
    ToolInterceptFn
);
global_execution_registry_api!(
    register_tool_execution_intercept,
    deregister_tool_execution_intercept,
    tool_execution_intercepts,
    ToolExecutionFn
);

global_guardrail_registry_api!(
    register_llm_sanitize_request_guardrail,
    deregister_llm_sanitize_request_guardrail,
    llm_sanitize_request_guardrails,
    LlmSanitizeRequestFn
);
global_guardrail_registry_api!(
    register_llm_sanitize_response_guardrail,
    deregister_llm_sanitize_response_guardrail,
    llm_sanitize_response_guardrails,
    LlmSanitizeResponseFn
);
global_guardrail_registry_api!(
    register_llm_conditional_execution_guardrail,
    deregister_llm_conditional_execution_guardrail,
    llm_conditional_execution_guardrails,
    LlmConditionalFn
);
global_intercept_registry_api!(
    register_llm_request_intercept,
    deregister_llm_request_intercept,
    llm_request_intercepts,
    LlmRequestInterceptFn
);
global_execution_registry_api!(
    register_llm_execution_intercept,
    deregister_llm_execution_intercept,
    llm_execution_intercepts,
    LlmExecutionFn
);
global_execution_registry_api!(
    register_llm_stream_execution_intercept,
    deregister_llm_stream_execution_intercept,
    llm_stream_execution_intercepts,
    LlmStreamExecutionFn
);

scope_guardrail_registry_api!(
    scope_register_tool_sanitize_request_guardrail,
    scope_deregister_tool_sanitize_request_guardrail,
    tool_sanitize_request_guardrails,
    ToolSanitizeFn
);
scope_guardrail_registry_api!(
    scope_register_tool_sanitize_response_guardrail,
    scope_deregister_tool_sanitize_response_guardrail,
    tool_sanitize_response_guardrails,
    ToolSanitizeFn
);
scope_guardrail_registry_api!(
    scope_register_tool_conditional_execution_guardrail,
    scope_deregister_tool_conditional_execution_guardrail,
    tool_conditional_execution_guardrails,
    ToolConditionalFn
);
scope_intercept_registry_api!(
    scope_register_tool_request_intercept,
    scope_deregister_tool_request_intercept,
    tool_request_intercepts,
    ToolInterceptFn
);
scope_execution_registry_api!(
    scope_register_tool_execution_intercept,
    scope_deregister_tool_execution_intercept,
    tool_execution_intercepts,
    ToolExecutionFn
);

scope_guardrail_registry_api!(
    scope_register_llm_sanitize_request_guardrail,
    scope_deregister_llm_sanitize_request_guardrail,
    llm_sanitize_request_guardrails,
    LlmSanitizeRequestFn
);
scope_guardrail_registry_api!(
    scope_register_llm_sanitize_response_guardrail,
    scope_deregister_llm_sanitize_response_guardrail,
    llm_sanitize_response_guardrails,
    LlmSanitizeResponseFn
);
scope_guardrail_registry_api!(
    scope_register_llm_conditional_execution_guardrail,
    scope_deregister_llm_conditional_execution_guardrail,
    llm_conditional_execution_guardrails,
    LlmConditionalFn
);
scope_intercept_registry_api!(
    scope_register_llm_request_intercept,
    scope_deregister_llm_request_intercept,
    llm_request_intercepts,
    LlmRequestInterceptFn
);
scope_execution_registry_api!(
    scope_register_llm_execution_intercept,
    scope_deregister_llm_execution_intercept,
    llm_execution_intercepts,
    LlmExecutionFn
);
scope_execution_registry_api!(
    scope_register_llm_stream_execution_intercept,
    scope_deregister_llm_stream_execution_intercept,
    llm_stream_execution_intercepts,
    LlmStreamExecutionFn
);
