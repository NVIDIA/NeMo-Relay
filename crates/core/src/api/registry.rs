// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Middleware registry helpers for global and scope-local guardrails,
//! intercepts, and subscribers.

use std::sync::Arc;

use crate::api::llm::LlmRequest;
use crate::api::runtime::callbacks::{LlmConditionalSharedFn, ToolConditionalSharedFn};
use crate::api::runtime::{
    LlmConditionalFn, LlmExecutionFn, LlmRequestInterceptFn, LlmSanitizeRequestFn,
    LlmSanitizeResponseFn, LlmStreamExecutionFn, ToolConditionalFn, ToolExecutionFn,
    ToolInterceptFn, ToolSanitizeFn,
};
use crate::api::runtime::{current_scope_stack, global_context};
use crate::api::shared::ensure_runtime_owner;
use crate::error::{FlowError, Result};
use crate::json::Json;

/// A priority-ordered request intercept registration entry.
pub struct Intercept<F> {
    /// Lower values run earlier in the chain.
    pub priority: i32,
    /// Whether this intercept stops later request intercepts after it returns.
    pub break_chain: bool,
    /// The caller-provided intercept callback.
    pub callable: F,
}

/// A priority-ordered execution intercept registration entry.
pub struct ExecutionIntercept<F> {
    /// Lower values run earlier in the chain.
    pub priority: i32,
    /// The caller-provided execution intercept callback.
    pub callable: F,
}

/// A priority-ordered guardrail registration entry.
pub struct GuardrailEntry<F> {
    /// Lower values run earlier in the chain.
    pub priority: i32,
    /// The caller-provided guardrail callback.
    pub guardrail: F,
}

fn share_tool_conditional_guardrail(
    guardrail: ToolConditionalFn,
) -> (ToolConditionalFn, ToolConditionalSharedFn) {
    let guardrail = Arc::new(guardrail);
    let boxed_guardrail = {
        let guardrail = guardrail.clone();
        Box::new(move |name: &str, args: &Json| (guardrail.as_ref())(name, args))
            as ToolConditionalFn
    };
    let shared_guardrail = Arc::new(move |name: &str, args: &Json| (guardrail.as_ref())(name, args))
        as ToolConditionalSharedFn;
    (boxed_guardrail, shared_guardrail)
}

fn share_llm_conditional_guardrail(
    guardrail: LlmConditionalFn,
) -> (LlmConditionalFn, LlmConditionalSharedFn) {
    let guardrail = Arc::new(guardrail);
    let boxed_guardrail = {
        let guardrail = guardrail.clone();
        Box::new(move |request: &LlmRequest| (guardrail.as_ref())(request)) as LlmConditionalFn
    };
    let shared_guardrail = Arc::new(move |request: &LlmRequest| (guardrail.as_ref())(request))
        as LlmConditionalSharedFn;
    (boxed_guardrail, shared_guardrail)
}

macro_rules! global_guardrail_registry_api {
    (
        $(#[$register_meta:meta])*
        $register_name:ident,
        $(#[$deregister_meta:meta])*
        $deregister_name:ident,
        $field:ident,
        $fn_type:ty
    ) => {
        $(#[$register_meta])*
        ///
        /// # Parameters
        /// - `name`: Unique middleware name in the global registry.
        /// - `priority`: Lower values run earlier in the chain.
        /// - `guardrail`: Guardrail callback stored under `name`.
        ///
        /// # Returns
        /// A [`Result`] that is `Ok(())` when the guardrail was registered.
        ///
        /// # Errors
        /// Returns [`FlowError::AlreadyExists`] when the name is already in
        /// use or an internal error if the runtime state cannot be updated.
        pub fn $register_name(name: &str, priority: i32, guardrail: $fn_type) -> Result<()> {
            ensure_runtime_owner()?;
            let context = global_context();
            let mut state = context
                .write()
                .map_err(|error| FlowError::Internal(error.to_string()))?;
            state
                .$field
                .register(
                    name.to_string(),
                    GuardrailEntry {
                        priority,
                        guardrail: guardrail.into(),
                    },
                )
                .map_err(FlowError::AlreadyExists)
        }

        $(#[$deregister_meta])*
        ///
        /// # Parameters
        /// - `name`: Global middleware name to remove.
        ///
        /// # Returns
        /// A [`Result`] containing `true` when a guardrail was removed and
        /// `false` when the name was not registered.
        ///
        /// # Errors
        /// Returns an internal error if the runtime state cannot be updated.
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
    (
        $(#[$register_meta:meta])*
        $register_name:ident,
        $(#[$deregister_meta:meta])*
        $deregister_name:ident,
        $field:ident,
        $fn_type:ty
    ) => {
        $(#[$register_meta])*
        ///
        /// # Parameters
        /// - `name`: Unique middleware name in the global registry.
        /// - `priority`: Lower values run earlier in the chain.
        /// - `break_chain`: Whether the intercept should stop later request
        ///   intercepts after it returns.
        /// - `callable`: Intercept callback stored under `name`.
        ///
        /// # Returns
        /// A [`Result`] that is `Ok(())` when the intercept was registered.
        ///
        /// # Errors
        /// Returns [`FlowError::AlreadyExists`] when the name is already in
        /// use or an internal error if the runtime state cannot be updated.
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

        $(#[$deregister_meta])*
        ///
        /// # Parameters
        /// - `name`: Global middleware name to remove.
        ///
        /// # Returns
        /// A [`Result`] containing `true` when an intercept was removed and
        /// `false` when the name was not registered.
        ///
        /// # Errors
        /// Returns an internal error if the runtime state cannot be updated.
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
    (
        $(#[$register_meta:meta])*
        $register_name:ident,
        $(#[$deregister_meta:meta])*
        $deregister_name:ident,
        $field:ident,
        $fn_type:ty
    ) => {
        $(#[$register_meta])*
        ///
        /// # Parameters
        /// - `name`: Unique middleware name in the global registry.
        /// - `priority`: Lower values run earlier in the chain.
        /// - `callable`: Execution intercept callback stored under `name`.
        ///
        /// # Returns
        /// A [`Result`] that is `Ok(())` when the intercept was registered.
        ///
        /// # Errors
        /// Returns [`FlowError::AlreadyExists`] when the name is already in
        /// use or an internal error if the runtime state cannot be updated.
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

        $(#[$deregister_meta])*
        ///
        /// # Parameters
        /// - `name`: Global middleware name to remove.
        ///
        /// # Returns
        /// A [`Result`] containing `true` when an execution intercept was
        /// removed and `false` when the name was not registered.
        ///
        /// # Errors
        /// Returns an internal error if the runtime state cannot be updated.
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
    (
        $(#[$register_meta:meta])*
        $register_name:ident,
        $(#[$deregister_meta:meta])*
        $deregister_name:ident,
        $field:ident,
        $fn_type:ty
    ) => {
        $(#[$register_meta])*
        ///
        /// # Parameters
        /// - `scope_uuid`: UUID of the active scope that owns the middleware.
        /// - `name`: Unique middleware name within that scope.
        /// - `priority`: Lower values run earlier in the chain.
        /// - `guardrail`: Guardrail callback stored under `name`.
        ///
        /// # Returns
        /// A [`Result`] that is `Ok(())` when the guardrail was registered.
        ///
        /// # Errors
        /// Returns [`FlowError::NotFound`] when the scope is not active,
        /// [`FlowError::AlreadyExists`] when the name is already in use on
        /// that scope, or an internal error if the runtime owner check fails.
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
                .register(
                    name.to_string(),
                    GuardrailEntry {
                        priority,
                        guardrail: guardrail.into(),
                    },
                )
                .map_err(FlowError::AlreadyExists)
        }

        $(#[$deregister_meta])*
        ///
        /// # Parameters
        /// - `scope_uuid`: UUID of the active scope that owns the middleware.
        /// - `name`: Scope-local middleware name to remove.
        ///
        /// # Returns
        /// A [`Result`] containing `true` when a guardrail was removed and
        /// `false` when the name was not registered on that scope.
        ///
        /// # Errors
        /// Returns [`FlowError::NotFound`] when the scope is not active or an
        /// internal error if the runtime owner check fails.
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
    (
        $(#[$register_meta:meta])*
        $register_name:ident,
        $(#[$deregister_meta:meta])*
        $deregister_name:ident,
        $field:ident,
        $fn_type:ty
    ) => {
        $(#[$register_meta])*
        ///
        /// # Parameters
        /// - `scope_uuid`: UUID of the active scope that owns the middleware.
        /// - `name`: Unique middleware name within that scope.
        /// - `priority`: Lower values run earlier in the chain.
        /// - `break_chain`: Whether the intercept should stop later request
        ///   intercepts after it returns.
        /// - `callable`: Intercept callback stored under `name`.
        ///
        /// # Returns
        /// A [`Result`] that is `Ok(())` when the intercept was registered.
        ///
        /// # Errors
        /// Returns [`FlowError::NotFound`] when the scope is not active,
        /// [`FlowError::AlreadyExists`] when the name is already in use on
        /// that scope, or an internal error if the runtime owner check fails.
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

        $(#[$deregister_meta])*
        ///
        /// # Parameters
        /// - `scope_uuid`: UUID of the active scope that owns the middleware.
        /// - `name`: Scope-local middleware name to remove.
        ///
        /// # Returns
        /// A [`Result`] containing `true` when an intercept was removed and
        /// `false` when the name was not registered on that scope.
        ///
        /// # Errors
        /// Returns [`FlowError::NotFound`] when the scope is not active or an
        /// internal error if the runtime owner check fails.
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
    (
        $(#[$register_meta:meta])*
        $register_name:ident,
        $(#[$deregister_meta:meta])*
        $deregister_name:ident,
        $field:ident,
        $fn_type:ty
    ) => {
        $(#[$register_meta])*
        ///
        /// # Parameters
        /// - `scope_uuid`: UUID of the active scope that owns the middleware.
        /// - `name`: Unique middleware name within that scope.
        /// - `priority`: Lower values run earlier in the chain.
        /// - `callable`: Execution intercept callback stored under `name`.
        ///
        /// # Returns
        /// A [`Result`] that is `Ok(())` when the intercept was registered.
        ///
        /// # Errors
        /// Returns [`FlowError::NotFound`] when the scope is not active,
        /// [`FlowError::AlreadyExists`] when the name is already in use on
        /// that scope, or an internal error if the runtime owner check fails.
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

        $(#[$deregister_meta])*
        ///
        /// # Parameters
        /// - `scope_uuid`: UUID of the active scope that owns the middleware.
        /// - `name`: Scope-local middleware name to remove.
        ///
        /// # Returns
        /// A [`Result`] containing `true` when an execution intercept was
        /// removed and `false` when the name was not registered on that scope.
        ///
        /// # Errors
        /// Returns [`FlowError::NotFound`] when the scope is not active or an
        /// internal error if the runtime owner check fails.
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
    /// Register a global tool sanitize-request guardrail.
    /// The guardrail rewrites only the tool input recorded on emitted start
    /// events.
    register_tool_sanitize_request_guardrail,
    /// Deregister a global tool sanitize-request guardrail.
    deregister_tool_sanitize_request_guardrail,
    tool_sanitize_request_guardrails,
    ToolSanitizeFn
);
global_guardrail_registry_api!(
    /// Register a global tool sanitize-response guardrail.
    /// The guardrail rewrites only the tool output recorded on emitted end
    /// events.
    register_tool_sanitize_response_guardrail,
    /// Deregister a global tool sanitize-response guardrail.
    deregister_tool_sanitize_response_guardrail,
    tool_sanitize_response_guardrails,
    ToolSanitizeFn
);
/// Register a global tool conditional-execution guardrail.
/// The guardrail can block tool execution before intercepts or the tool
/// callback run.
///
/// # Parameters
/// - `name`: Unique middleware name in the global registry.
/// - `priority`: Lower values run earlier in the chain.
/// - `guardrail`: Guardrail callback stored under `name`.
///
/// # Returns
/// A [`Result`] that is `Ok(())` when the guardrail was registered.
///
/// # Errors
/// Returns [`FlowError::AlreadyExists`] when the name is already in use or an
/// internal error if the runtime state cannot be updated.
pub fn register_tool_conditional_execution_guardrail(
    name: &str,
    priority: i32,
    guardrail: ToolConditionalFn,
) -> Result<()> {
    ensure_runtime_owner()?;
    let context = global_context();
    let mut state = context
        .write()
        .map_err(|error| FlowError::Internal(error.to_string()))?;
    let (boxed_guardrail, shared_guardrail) = share_tool_conditional_guardrail(guardrail);
    state
        .tool_conditional_execution_guardrails
        .register(
            name.to_string(),
            GuardrailEntry {
                priority,
                guardrail: boxed_guardrail,
            },
        )
        .map_err(FlowError::AlreadyExists)?;
    if let Err(error) = state.tool_conditional_shared_guardrails_mut().register(
        name.to_string(),
        GuardrailEntry {
            priority,
            guardrail: shared_guardrail,
        },
    ) {
        state.tool_conditional_execution_guardrails.deregister(name);
        return Err(FlowError::AlreadyExists(error));
    }
    Ok(())
}

/// Deregister a global tool conditional-execution guardrail.
///
/// # Parameters
/// - `name`: Global middleware name to remove.
///
/// # Returns
/// A [`Result`] containing `true` when a guardrail was removed and `false`
/// when the name was not registered.
///
/// # Errors
/// Returns an internal error if the runtime state cannot be updated.
pub fn deregister_tool_conditional_execution_guardrail(name: &str) -> Result<bool> {
    ensure_runtime_owner()?;
    let context = global_context();
    let mut state = context
        .write()
        .map_err(|error| FlowError::Internal(error.to_string()))?;
    let removed = state.tool_conditional_execution_guardrails.deregister(name);
    let _ = state
        .tool_conditional_shared_guardrails_mut()
        .deregister(name);
    Ok(removed)
}
global_intercept_registry_api!(
    /// Register a global tool request intercept.
    /// Request intercepts can rewrite tool arguments before execution.
    register_tool_request_intercept,
    /// Deregister a global tool request intercept.
    deregister_tool_request_intercept,
    tool_request_intercepts,
    ToolInterceptFn
);
global_execution_registry_api!(
    /// Register a global tool execution intercept.
    /// Execution intercepts can wrap or replace the tool callback.
    register_tool_execution_intercept,
    /// Deregister a global tool execution intercept.
    deregister_tool_execution_intercept,
    tool_execution_intercepts,
    ToolExecutionFn
);

global_guardrail_registry_api!(
    /// Register a global LLM sanitize-request guardrail.
    /// The guardrail rewrites only the request payload recorded on emitted
    /// start events.
    register_llm_sanitize_request_guardrail,
    /// Deregister a global LLM sanitize-request guardrail.
    deregister_llm_sanitize_request_guardrail,
    llm_sanitize_request_guardrails,
    LlmSanitizeRequestFn
);
global_guardrail_registry_api!(
    /// Register a global LLM sanitize-response guardrail.
    /// The guardrail rewrites only the response payload recorded on emitted
    /// end events.
    register_llm_sanitize_response_guardrail,
    /// Deregister a global LLM sanitize-response guardrail.
    deregister_llm_sanitize_response_guardrail,
    llm_sanitize_response_guardrails,
    LlmSanitizeResponseFn
);
/// Register a global LLM conditional-execution guardrail.
/// The guardrail can block LLM execution before intercepts or the provider
/// callback run.
///
/// # Parameters
/// - `name`: Unique middleware name in the global registry.
/// - `priority`: Lower values run earlier in the chain.
/// - `guardrail`: Guardrail callback stored under `name`.
///
/// # Returns
/// A [`Result`] that is `Ok(())` when the guardrail was registered.
///
/// # Errors
/// Returns [`FlowError::AlreadyExists`] when the name is already in use or an
/// internal error if the runtime state cannot be updated.
pub fn register_llm_conditional_execution_guardrail(
    name: &str,
    priority: i32,
    guardrail: LlmConditionalFn,
) -> Result<()> {
    ensure_runtime_owner()?;
    let context = global_context();
    let mut state = context
        .write()
        .map_err(|error| FlowError::Internal(error.to_string()))?;
    let (boxed_guardrail, shared_guardrail) = share_llm_conditional_guardrail(guardrail);
    state
        .llm_conditional_execution_guardrails
        .register(
            name.to_string(),
            GuardrailEntry {
                priority,
                guardrail: boxed_guardrail,
            },
        )
        .map_err(FlowError::AlreadyExists)?;
    if let Err(error) = state.llm_conditional_shared_guardrails_mut().register(
        name.to_string(),
        GuardrailEntry {
            priority,
            guardrail: shared_guardrail,
        },
    ) {
        state.llm_conditional_execution_guardrails.deregister(name);
        return Err(FlowError::AlreadyExists(error));
    }
    Ok(())
}

/// Deregister a global LLM conditional-execution guardrail.
///
/// # Parameters
/// - `name`: Global middleware name to remove.
///
/// # Returns
/// A [`Result`] containing `true` when a guardrail was removed and `false`
/// when the name was not registered.
///
/// # Errors
/// Returns an internal error if the runtime state cannot be updated.
pub fn deregister_llm_conditional_execution_guardrail(name: &str) -> Result<bool> {
    ensure_runtime_owner()?;
    let context = global_context();
    let mut state = context
        .write()
        .map_err(|error| FlowError::Internal(error.to_string()))?;
    let removed = state.llm_conditional_execution_guardrails.deregister(name);
    let _ = state
        .llm_conditional_shared_guardrails_mut()
        .deregister(name);
    Ok(removed)
}
global_intercept_registry_api!(
    /// Register a global LLM request intercept.
    /// Request intercepts can rewrite or annotate the outgoing LLM request.
    register_llm_request_intercept,
    /// Deregister a global LLM request intercept.
    deregister_llm_request_intercept,
    llm_request_intercepts,
    LlmRequestInterceptFn
);
global_execution_registry_api!(
    /// Register a global LLM execution intercept.
    /// Execution intercepts can wrap or replace the non-streaming provider
    /// callback.
    register_llm_execution_intercept,
    /// Deregister a global LLM execution intercept.
    deregister_llm_execution_intercept,
    llm_execution_intercepts,
    LlmExecutionFn
);
global_execution_registry_api!(
    /// Register a global streaming LLM execution intercept.
    /// Execution intercepts can wrap or replace the streaming provider
    /// callback.
    register_llm_stream_execution_intercept,
    /// Deregister a global streaming LLM execution intercept.
    deregister_llm_stream_execution_intercept,
    llm_stream_execution_intercepts,
    LlmStreamExecutionFn
);

scope_guardrail_registry_api!(
    /// Register a scope-local tool sanitize-request guardrail.
    /// The guardrail rewrites only tool input emitted under the owning scope.
    scope_register_tool_sanitize_request_guardrail,
    /// Deregister a scope-local tool sanitize-request guardrail.
    scope_deregister_tool_sanitize_request_guardrail,
    tool_sanitize_request_guardrails,
    ToolSanitizeFn
);
scope_guardrail_registry_api!(
    /// Register a scope-local tool sanitize-response guardrail.
    /// The guardrail rewrites only tool output emitted under the owning scope.
    scope_register_tool_sanitize_response_guardrail,
    /// Deregister a scope-local tool sanitize-response guardrail.
    scope_deregister_tool_sanitize_response_guardrail,
    tool_sanitize_response_guardrails,
    ToolSanitizeFn
);
/// Register a scope-local tool conditional-execution guardrail.
/// The guardrail can block tool execution inside the owning scope.
///
/// # Parameters
/// - `scope_uuid`: UUID of the active scope that owns the middleware.
/// - `name`: Unique middleware name on the scope-local registry.
/// - `priority`: Lower values run earlier in the chain.
/// - `guardrail`: Guardrail callback stored under `name`.
///
/// # Returns
/// A [`Result`] that is `Ok(())` when the guardrail was registered.
///
/// # Errors
/// Returns [`FlowError::NotFound`] when the scope is not active,
/// [`FlowError::AlreadyExists`] when the name is already in use on that scope,
/// or an internal error if the runtime owner check fails.
pub fn scope_register_tool_conditional_execution_guardrail(
    scope_uuid: &uuid::Uuid,
    name: &str,
    priority: i32,
    guardrail: ToolConditionalFn,
) -> Result<()> {
    ensure_runtime_owner()?;
    let scope_stack = current_scope_stack();
    let mut guard = scope_stack.write().expect("scope stack lock poisoned");
    let registries = guard
        .local_registries_mut(scope_uuid)
        .ok_or_else(|| FlowError::NotFound(format!("scope {scope_uuid} not found")))?;
    let (boxed_guardrail, shared_guardrail) = share_tool_conditional_guardrail(guardrail);
    registries
        .tool_conditional_execution_guardrails
        .register(
            name.to_string(),
            GuardrailEntry {
                priority,
                guardrail: boxed_guardrail,
            },
        )
        .map_err(FlowError::AlreadyExists)?;
    if let Err(error) = registries
        .tool_conditional_execution_shared_guardrails
        .register(
            name.to_string(),
            GuardrailEntry {
                priority,
                guardrail: shared_guardrail,
            },
        )
    {
        registries
            .tool_conditional_execution_guardrails
            .deregister(name);
        return Err(FlowError::AlreadyExists(error));
    }
    Ok(())
}

/// Deregister a scope-local tool conditional-execution guardrail.
///
/// # Parameters
/// - `scope_uuid`: UUID of the active scope that owns the middleware.
/// - `name`: Scope-local middleware name to remove.
///
/// # Returns
/// A [`Result`] containing `true` when a guardrail was removed and `false`
/// when the name was not registered on that scope.
///
/// # Errors
/// Returns [`FlowError::NotFound`] when the scope is not active or an internal
/// error if the runtime owner check fails.
pub fn scope_deregister_tool_conditional_execution_guardrail(
    scope_uuid: &uuid::Uuid,
    name: &str,
) -> Result<bool> {
    ensure_runtime_owner()?;
    let scope_stack = current_scope_stack();
    let mut guard = scope_stack.write().expect("scope stack lock poisoned");
    let registries = guard
        .local_registries_mut(scope_uuid)
        .ok_or_else(|| FlowError::NotFound(format!("scope {scope_uuid} not found")))?;
    let removed = registries
        .tool_conditional_execution_guardrails
        .deregister(name);
    let _ = registries
        .tool_conditional_execution_shared_guardrails
        .deregister(name);
    Ok(removed)
}
scope_intercept_registry_api!(
    /// Register a scope-local tool request intercept.
    /// Request intercepts can rewrite tool arguments inside the owning scope.
    scope_register_tool_request_intercept,
    /// Deregister a scope-local tool request intercept.
    scope_deregister_tool_request_intercept,
    tool_request_intercepts,
    ToolInterceptFn
);
scope_execution_registry_api!(
    /// Register a scope-local tool execution intercept.
    /// Execution intercepts can wrap or replace the tool callback inside the
    /// owning scope.
    scope_register_tool_execution_intercept,
    /// Deregister a scope-local tool execution intercept.
    scope_deregister_tool_execution_intercept,
    tool_execution_intercepts,
    ToolExecutionFn
);

scope_guardrail_registry_api!(
    /// Register a scope-local LLM sanitize-request guardrail.
    /// The guardrail rewrites only request payloads emitted under the owning
    /// scope.
    scope_register_llm_sanitize_request_guardrail,
    /// Deregister a scope-local LLM sanitize-request guardrail.
    scope_deregister_llm_sanitize_request_guardrail,
    llm_sanitize_request_guardrails,
    LlmSanitizeRequestFn
);
scope_guardrail_registry_api!(
    /// Register a scope-local LLM sanitize-response guardrail.
    /// The guardrail rewrites only response payloads emitted under the owning
    /// scope.
    scope_register_llm_sanitize_response_guardrail,
    /// Deregister a scope-local LLM sanitize-response guardrail.
    scope_deregister_llm_sanitize_response_guardrail,
    llm_sanitize_response_guardrails,
    LlmSanitizeResponseFn
);
/// Register a scope-local LLM conditional-execution guardrail.
/// The guardrail can block LLM execution inside the owning scope.
///
/// # Parameters
/// - `scope_uuid`: UUID of the active scope that owns the middleware.
/// - `name`: Unique middleware name on the scope-local registry.
/// - `priority`: Lower values run earlier in the chain.
/// - `guardrail`: Guardrail callback stored under `name`.
///
/// # Returns
/// A [`Result`] that is `Ok(())` when the guardrail was registered.
///
/// # Errors
/// Returns [`FlowError::NotFound`] when the scope is not active,
/// [`FlowError::AlreadyExists`] when the name is already in use on that scope,
/// or an internal error if the runtime owner check fails.
pub fn scope_register_llm_conditional_execution_guardrail(
    scope_uuid: &uuid::Uuid,
    name: &str,
    priority: i32,
    guardrail: LlmConditionalFn,
) -> Result<()> {
    ensure_runtime_owner()?;
    let scope_stack = current_scope_stack();
    let mut guard = scope_stack.write().expect("scope stack lock poisoned");
    let registries = guard
        .local_registries_mut(scope_uuid)
        .ok_or_else(|| FlowError::NotFound(format!("scope {scope_uuid} not found")))?;
    let (boxed_guardrail, shared_guardrail) = share_llm_conditional_guardrail(guardrail);
    registries
        .llm_conditional_execution_guardrails
        .register(
            name.to_string(),
            GuardrailEntry {
                priority,
                guardrail: boxed_guardrail,
            },
        )
        .map_err(FlowError::AlreadyExists)?;
    if let Err(error) = registries
        .llm_conditional_execution_shared_guardrails
        .register(
            name.to_string(),
            GuardrailEntry {
                priority,
                guardrail: shared_guardrail,
            },
        )
    {
        registries
            .llm_conditional_execution_guardrails
            .deregister(name);
        return Err(FlowError::AlreadyExists(error));
    }
    Ok(())
}

/// Deregister a scope-local LLM conditional-execution guardrail.
///
/// # Parameters
/// - `scope_uuid`: UUID of the active scope that owns the middleware.
/// - `name`: Scope-local middleware name to remove.
///
/// # Returns
/// A [`Result`] containing `true` when a guardrail was removed and `false`
/// when the name was not registered on that scope.
///
/// # Errors
/// Returns [`FlowError::NotFound`] when the scope is not active or an internal
/// error if the runtime owner check fails.
pub fn scope_deregister_llm_conditional_execution_guardrail(
    scope_uuid: &uuid::Uuid,
    name: &str,
) -> Result<bool> {
    ensure_runtime_owner()?;
    let scope_stack = current_scope_stack();
    let mut guard = scope_stack.write().expect("scope stack lock poisoned");
    let registries = guard
        .local_registries_mut(scope_uuid)
        .ok_or_else(|| FlowError::NotFound(format!("scope {scope_uuid} not found")))?;
    let removed = registries
        .llm_conditional_execution_guardrails
        .deregister(name);
    let _ = registries
        .llm_conditional_execution_shared_guardrails
        .deregister(name);
    Ok(removed)
}
scope_intercept_registry_api!(
    /// Register a scope-local LLM request intercept.
    /// Request intercepts can rewrite or annotate LLM requests inside the
    /// owning scope.
    scope_register_llm_request_intercept,
    /// Deregister a scope-local LLM request intercept.
    scope_deregister_llm_request_intercept,
    llm_request_intercepts,
    LlmRequestInterceptFn
);
scope_execution_registry_api!(
    /// Register a scope-local LLM execution intercept.
    /// Execution intercepts can wrap or replace the non-streaming provider
    /// callback inside the owning scope.
    scope_register_llm_execution_intercept,
    /// Deregister a scope-local LLM execution intercept.
    scope_deregister_llm_execution_intercept,
    llm_execution_intercepts,
    LlmExecutionFn
);
scope_execution_registry_api!(
    /// Register a scope-local streaming LLM execution intercept.
    /// Execution intercepts can wrap or replace the streaming provider
    /// callback inside the owning scope.
    scope_register_llm_stream_execution_intercept,
    /// Deregister a scope-local streaming LLM execution intercept.
    scope_deregister_llm_stream_execution_intercept,
    llm_stream_execution_intercepts,
    LlmStreamExecutionFn
);
