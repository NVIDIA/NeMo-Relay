// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Middleware registry helpers for global and scope-local guardrails,
//! intercepts, and subscribers.

use crate::api::runtime::{
    ConditionalMiddlewareGuardrailFn, EventMetadataInjectorFn, EventSanitizeFn, LlmConditionalFn,
    LlmExecutionFn, LlmRequestInterceptFn, LlmSanitizeRequestFn, LlmSanitizeResponseFn,
    LlmStreamExecutionFn, ToolConditionalFn, ToolExecutionFn, ToolInterceptFn, ToolSanitizeFn,
};
use crate::api::runtime::{current_scope_stack, global_context};
use crate::api::shared::ensure_runtime_owner;
use crate::error::{FlowError, Result};
use crate::registry::RegistryEntry;
pub use nemo_relay_types::api::registry::{
    RuntimeRegistrationIdentity, RuntimeRegistrationKind, RuntimeRegistrationOwner,
    RuntimeRegistrationOwnerKind,
};
use std::collections::{BTreeMap, BTreeSet};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{OnceLock, RwLock};

#[derive(Clone)]
struct ConditionalMiddlewareGuardrail {
    kinds: BTreeSet<RuntimeRegistrationKind>,
    registration_name: String,
    callback: ConditionalMiddlewareGuardrailFn,
}

static CONDITIONAL_MIDDLEWARE_GUARDRAILS: OnceLock<
    RwLock<BTreeMap<String, ConditionalMiddlewareGuardrail>>,
> = OnceLock::new();

fn conditional_middleware_guardrails()
-> &'static RwLock<BTreeMap<String, ConditionalMiddlewareGuardrail>> {
    CONDITIONAL_MIDDLEWARE_GUARDRAILS.get_or_init(|| RwLock::new(BTreeMap::new()))
}

/// Register a global conditional middleware guardrail.
pub fn register_conditional_middleware_guardrail(
    name: &str,
    kinds: BTreeSet<RuntimeRegistrationKind>,
    registration_name: &str,
    guardrail: ConditionalMiddlewareGuardrailFn,
) -> Result<()> {
    ensure_runtime_owner()?;
    if kinds.is_empty() {
        return Err(FlowError::InvalidArgument(
            "conditional middleware guardrail kinds must not be empty".to_string(),
        ));
    }
    let mut gates = conditional_middleware_guardrails()
        .write()
        .map_err(|error| FlowError::Internal(error.to_string()))?;
    if gates.contains_key(name) {
        return Err(FlowError::AlreadyExists(format!(
            "{name} conditional middleware guardrail already exists"
        )));
    }
    gates.insert(
        name.to_string(),
        ConditionalMiddlewareGuardrail {
            kinds,
            registration_name: registration_name.to_string(),
            callback: guardrail,
        },
    );
    Ok(())
}

/// Deregister a global conditional middleware guardrail.
pub fn deregister_conditional_middleware_guardrail(name: &str) -> Result<bool> {
    ensure_runtime_owner()?;
    let mut gates = conditional_middleware_guardrails()
        .write()
        .map_err(|error| FlowError::Internal(error.to_string()))?;
    Ok(gates.remove(name).is_some())
}

/// Return whether one global runtime registration is enabled by every
/// matching conditional middleware guardrail.
pub(crate) fn runtime_registration_is_enabled(
    kind: RuntimeRegistrationKind,
    effective_name: &str,
) -> bool {
    let Some(registry) = CONDITIONAL_MIDDLEWARE_GUARDRAILS.get() else {
        return true;
    };
    let matching = match registry.read() {
        Ok(gates) => gates
            .iter()
            .filter(|(_, gate)| {
                gate.kinds.contains(&kind) && gate.registration_name == effective_name
            })
            .map(|(name, gate)| (name.clone(), gate.kinds.clone(), gate.callback.clone()))
            .collect::<Vec<_>>(),
        Err(error) => {
            log::error!(
                target: "nemo_relay.runtime",
                event = "conditional_middleware_guardrail_registry_failed",
                registration_kind = kind.as_str(),
                registration_name = effective_name;
                "Conditional middleware guardrail registry read failed; enabling target: {error}"
            );
            return true;
        }
    };

    for (gate_name, kinds, callback) in matching {
        match catch_unwind(AssertUnwindSafe(|| callback(&kinds, effective_name))) {
            Ok(None) => {}
            Ok(Some(reason)) => {
                log::debug!(
                    target: "nemo_relay.runtime",
                    event = "runtime_registration_disabled",
                    gate = gate_name.as_str(),
                    registration_kind = kind.as_str(),
                    registration_name = effective_name,
                    reason = reason.as_str();
                    "Conditional middleware guardrail disabled runtime registration"
                );
                return false;
            }
            Err(_) => log::error!(
                target: "nemo_relay.runtime",
                event = "conditional_middleware_guardrail_panicked",
                gate = gate_name.as_str(),
                registration_kind = kind.as_str(),
                registration_name = effective_name;
                "Conditional middleware guardrail panicked; enabling target"
            ),
        }
    }
    true
}

fn registration_identity(
    kind: RuntimeRegistrationKind,
    effective_name: &str,
) -> RuntimeRegistrationIdentity {
    let Some((plugin_kind, component_ordinal, local_name)) =
        crate::plugin::decode_plugin_component_effective_name(effective_name)
    else {
        return RuntimeRegistrationIdentity {
            kind,
            local_name: effective_name.to_string(),
            effective_name: effective_name.to_string(),
            owner: RuntimeRegistrationOwner {
                kind: RuntimeRegistrationOwnerKind::GlobalApi,
                plugin_kind: None,
                component_ordinal: None,
            },
        };
    };
    RuntimeRegistrationIdentity {
        kind,
        local_name,
        effective_name: effective_name.to_string(),
        owner: RuntimeRegistrationOwner {
            kind: RuntimeRegistrationOwnerKind::Plugin,
            plugin_kind: Some(plugin_kind),
            component_ordinal: Some(component_ordinal),
        },
    }
}

/// List a deterministic snapshot of global gateable runtime registrations.
pub fn list_runtime_registrations(
    kinds: Option<&BTreeSet<RuntimeRegistrationKind>>,
) -> Result<Vec<RuntimeRegistrationIdentity>> {
    ensure_runtime_owner()?;
    let context = global_context();
    let state = context
        .read()
        .map_err(|error| FlowError::Internal(error.to_string()))?;
    let mut registrations = Vec::new();
    macro_rules! collect_registry {
        ($kind:expr, $field:ident) => {
            if kinds.is_none_or(|selected| selected.contains(&$kind)) {
                registrations.extend(
                    state
                        .$field
                        .values()
                        .map(|entry| registration_identity($kind, &entry.name)),
                );
            }
        };
    }
    if kinds.is_none_or(|selected| selected.contains(&RuntimeRegistrationKind::Subscriber)) {
        registrations.extend(
            state
                .event_subscribers
                .keys()
                .map(|name| registration_identity(RuntimeRegistrationKind::Subscriber, name)),
        );
    }
    collect_registry!(
        RuntimeRegistrationKind::EventMetadataInjector,
        event_metadata_injectors
    );
    collect_registry!(
        RuntimeRegistrationKind::MarkSanitizeGuardrail,
        mark_sanitize_guardrails
    );
    collect_registry!(
        RuntimeRegistrationKind::ScopeSanitizeStartGuardrail,
        scope_sanitize_start_guardrails
    );
    collect_registry!(
        RuntimeRegistrationKind::ScopeSanitizeEndGuardrail,
        scope_sanitize_end_guardrails
    );
    collect_registry!(
        RuntimeRegistrationKind::ToolSanitizeRequestGuardrail,
        tool_sanitize_request_guardrails
    );
    collect_registry!(
        RuntimeRegistrationKind::ToolSanitizeResponseGuardrail,
        tool_sanitize_response_guardrails
    );
    collect_registry!(
        RuntimeRegistrationKind::ToolConditionalExecutionGuardrail,
        tool_conditional_execution_guardrails
    );
    collect_registry!(
        RuntimeRegistrationKind::ToolRequestIntercept,
        tool_request_intercepts
    );
    collect_registry!(
        RuntimeRegistrationKind::ToolExecutionIntercept,
        tool_execution_intercepts
    );
    collect_registry!(
        RuntimeRegistrationKind::LlmSanitizeRequestGuardrail,
        llm_sanitize_request_guardrails
    );
    collect_registry!(
        RuntimeRegistrationKind::LlmSanitizeResponseGuardrail,
        llm_sanitize_response_guardrails
    );
    collect_registry!(
        RuntimeRegistrationKind::LlmConditionalExecutionGuardrail,
        llm_conditional_execution_guardrails
    );
    collect_registry!(
        RuntimeRegistrationKind::LlmRequestIntercept,
        llm_request_intercepts
    );
    collect_registry!(
        RuntimeRegistrationKind::LlmExecutionIntercept,
        llm_execution_intercepts
    );
    collect_registry!(
        RuntimeRegistrationKind::LlmStreamExecutionIntercept,
        llm_stream_execution_intercepts
    );
    registrations.sort_by(|left, right| {
        left.kind
            .cmp(&right.kind)
            .then_with(|| left.effective_name.cmp(&right.effective_name))
    });
    Ok(registrations)
}

/// A priority-ordered registration record.
///
/// Registry records carry the common entry metadata used by all middleware
/// registries plus the caller-provided payload.
#[derive(Clone)]
pub(crate) struct RegistryRecord<T> {
    /// Unique middleware name within its registry.
    pub(crate) name: String,
    /// Lower values run earlier in the chain.
    pub(crate) priority: i32,
    /// The caller-provided registry payload.
    pub(crate) payload: T,
}

impl<T> RegistryRecord<T> {
    /// Create a new priority-ordered registry record.
    pub(crate) fn new(name: impl Into<String>, priority: i32, payload: T) -> Self {
        Self {
            name: name.into(),
            priority,
            payload,
        }
    }
}

impl<T> RegistryEntry for RegistryRecord<T> {
    fn name(&self) -> &str {
        &self.name
    }

    fn priority(&self) -> i32 {
        self.priority
    }
}

/// Request-intercept-specific registration payload.
///
/// Request intercepts carry one extra chain-control flag that does not apply
/// to guardrails or execution intercepts.
#[derive(Clone)]
pub(crate) struct RequestIntercept<F> {
    /// Whether this intercept stops later request intercepts after it returns.
    pub(crate) break_chain: bool,
    /// The caller-provided request intercept callback.
    pub(crate) callable: F,
}

impl<F> RequestIntercept<F> {
    /// Create a new request intercept payload.
    pub(crate) fn new(break_chain: bool, callable: F) -> Self {
        Self {
            break_chain,
            callable,
        }
    }
}

/// A priority-ordered guardrail registration record.
pub(crate) type Guardrail<F> = RegistryRecord<F>;

/// A priority-ordered Event metadata injector registration record.
pub(crate) type EventMetadataInjector = RegistryRecord<EventMetadataInjectorFn>;

/// A priority-ordered request intercept registration record.
pub(crate) type Intercept<F> = RegistryRecord<RequestIntercept<F>>;

/// A priority-ordered execution intercept registration record.
pub(crate) type ExecutionIntercept<F> = RegistryRecord<F>;

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
                    Guardrail::new(name, priority, guardrail.into()),
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
                    Intercept::new(name, priority, RequestIntercept::new(break_chain, callable)),
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
                .register(ExecutionIntercept::new(name, priority, callable))
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
                    Guardrail::new(name, priority, guardrail.into()),
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
                    Intercept::new(name, priority, RequestIntercept::new(break_chain, callable)),
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
                .register(ExecutionIntercept::new(name, priority, callable))
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
    /// Register a global mark event sanitizer.
    register_mark_sanitize_guardrail,
    /// Deregister a global mark event sanitizer.
    deregister_mark_sanitize_guardrail,
    mark_sanitize_guardrails,
    EventSanitizeFn
);

/// Register a global Event metadata injector.
///
/// Injectors run in ascending priority order on every Event before Event
/// sanitizers. Returned metadata is insert-only.
pub fn register_event_metadata_injector(
    name: &str,
    priority: i32,
    injector: EventMetadataInjectorFn,
) -> Result<()> {
    ensure_runtime_owner()?;
    let context = global_context();
    let mut state = context
        .write()
        .map_err(|error| FlowError::Internal(error.to_string()))?;
    state
        .event_metadata_injectors
        .register(EventMetadataInjector::new(name, priority, injector))
        .map_err(FlowError::AlreadyExists)
}

/// Deregister a global Event metadata injector.
pub fn deregister_event_metadata_injector(name: &str) -> Result<bool> {
    ensure_runtime_owner()?;
    let context = global_context();
    let mut state = context
        .write()
        .map_err(|error| FlowError::Internal(error.to_string()))?;
    Ok(state.event_metadata_injectors.deregister(name))
}

global_guardrail_registry_api!(
    /// Register a global scope-start event sanitizer.
    register_scope_sanitize_start_guardrail,
    /// Deregister a global scope-start event sanitizer.
    deregister_scope_sanitize_start_guardrail,
    scope_sanitize_start_guardrails,
    EventSanitizeFn
);
global_guardrail_registry_api!(
    /// Register a global scope-end event sanitizer.
    register_scope_sanitize_end_guardrail,
    /// Deregister a global scope-end event sanitizer.
    deregister_scope_sanitize_end_guardrail,
    scope_sanitize_end_guardrails,
    EventSanitizeFn
);

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
global_guardrail_registry_api!(
    /// Register a global tool conditional-execution guardrail.
    /// The guardrail can block tool execution before intercepts or the tool
    /// callback run.
    register_tool_conditional_execution_guardrail,
    /// Deregister a global tool conditional-execution guardrail.
    deregister_tool_conditional_execution_guardrail,
    tool_conditional_execution_guardrails,
    ToolConditionalFn
);
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
    /// Execution intercepts can wrap or replace the tool callback. Each
    /// callback returns a canonical tool execution outcome, while its
    /// continuation resolves to the downstream
    /// [`ToolExecutionResult`](crate::api::tool::ToolExecutionResult).
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
global_guardrail_registry_api!(
    /// Register a global LLM conditional-execution guardrail.
    /// The guardrail can block LLM execution before intercepts or the provider
    /// callback run.
    register_llm_conditional_execution_guardrail,
    /// Deregister a global LLM conditional-execution guardrail.
    deregister_llm_conditional_execution_guardrail,
    llm_conditional_execution_guardrails,
    LlmConditionalFn
);
global_intercept_registry_api!(
    /// Register a global LLM request intercept.
    /// Request intercepts can rewrite or annotate the outgoing LLM request and
    /// schedule lifecycle marks for the resulting LLM scope.
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
    /// Register a scope-local mark event sanitizer.
    scope_register_mark_sanitize_guardrail,
    /// Deregister a scope-local mark event sanitizer.
    scope_deregister_mark_sanitize_guardrail,
    mark_sanitize_guardrails,
    EventSanitizeFn
);

/// Register an Event metadata injector owned by an active scope.
pub fn scope_register_event_metadata_injector(
    scope_uuid: &uuid::Uuid,
    name: &str,
    priority: i32,
    injector: EventMetadataInjectorFn,
) -> Result<()> {
    ensure_runtime_owner()?;
    let scope_stack = current_scope_stack();
    let mut guard = scope_stack.write().expect("scope stack lock poisoned");
    let registries = guard
        .local_registries_mut(scope_uuid)
        .ok_or_else(|| FlowError::NotFound(format!("scope {scope_uuid} not found")))?;
    registries
        .event_metadata_injectors
        .register(EventMetadataInjector::new(name, priority, injector))
        .map_err(FlowError::AlreadyExists)
}

/// Deregister an Event metadata injector owned by an active scope.
pub fn scope_deregister_event_metadata_injector(
    scope_uuid: &uuid::Uuid,
    name: &str,
) -> Result<bool> {
    ensure_runtime_owner()?;
    let scope_stack = current_scope_stack();
    let mut guard = scope_stack.write().expect("scope stack lock poisoned");
    let registries = guard
        .local_registries_mut(scope_uuid)
        .ok_or_else(|| FlowError::NotFound(format!("scope {scope_uuid} not found")))?;
    Ok(registries.event_metadata_injectors.deregister(name))
}
scope_guardrail_registry_api!(
    /// Register a scope-local scope-start event sanitizer.
    scope_register_scope_sanitize_start_guardrail,
    /// Deregister a scope-local scope-start event sanitizer.
    scope_deregister_scope_sanitize_start_guardrail,
    scope_sanitize_start_guardrails,
    EventSanitizeFn
);
scope_guardrail_registry_api!(
    /// Register a scope-local scope-end event sanitizer.
    scope_register_scope_sanitize_end_guardrail,
    /// Deregister a scope-local scope-end event sanitizer.
    scope_deregister_scope_sanitize_end_guardrail,
    scope_sanitize_end_guardrails,
    EventSanitizeFn
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
scope_guardrail_registry_api!(
    /// Register a scope-local tool conditional-execution guardrail.
    /// The guardrail can block tool execution inside the owning scope.
    scope_register_tool_conditional_execution_guardrail,
    /// Deregister a scope-local tool conditional-execution guardrail.
    scope_deregister_tool_conditional_execution_guardrail,
    tool_conditional_execution_guardrails,
    ToolConditionalFn
);
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
    /// owning scope. Each callback returns a canonical tool execution outcome,
    /// while its continuation resolves to the downstream
    /// [`ToolExecutionResult`](crate::api::tool::ToolExecutionResult).
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
scope_guardrail_registry_api!(
    /// Register a scope-local LLM conditional-execution guardrail.
    /// The guardrail can block LLM execution inside the owning scope.
    scope_register_llm_conditional_execution_guardrail,
    /// Deregister a scope-local LLM conditional-execution guardrail.
    scope_deregister_llm_conditional_execution_guardrail,
    llm_conditional_execution_guardrails,
    LlmConditionalFn
);
scope_intercept_registry_api!(
    /// Register a scope-local LLM request intercept.
    /// Request intercepts can rewrite or annotate LLM requests inside the
    /// owning scope and schedule lifecycle marks for the resulting LLM scope.
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
