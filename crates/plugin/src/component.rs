// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-process plugin contracts shared by built-in components and runtime hosts.

use nemo_relay_types::api::{event::Event, registry::RuntimeRegistrationKind, runtime::*};
use nemo_relay_types::diagnostics::Classification;
pub use nemo_relay_types::plugin::{ConfigDiagnostic, DiagnosticLevel};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as Json};
use std::collections::BTreeSet;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use thiserror::Error;

/// Result of validating, activating, or cleaning up a component.
pub type Result<T> = std::result::Result<T, PluginError>;

/// Error type for generic plugin operations.
#[derive(Debug, Error)]
pub enum PluginError {
    /// Configuration validation failed.
    #[error("invalid config: {0}")]
    InvalidConfig(String),

    /// The requested mutation conflicts with current plugin state.
    #[error("conflict: {0}")]
    Conflict(String),

    /// The requested plugin resource was not found.
    #[error("not found: {0}")]
    NotFound(String),

    /// A serialization or deserialization operation failed.
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// An internal plugin-system error occurred.
    #[error("internal error: {0}")]
    Internal(String),

    /// A runtime middleware/subscriber registration failed.
    #[error("registration failed: {0}")]
    RegistrationFailed(String),
}

/// One configured plugin component.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct PluginComponentSpec {
    /// Registered plugin kind string.
    pub kind: String,
    /// Whether the component should be activated.
    ///
    /// Disabled components are still validated but skipped during runtime
    /// registration.
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    /// Component-local JSON config object passed to the plugin.
    #[serde(default)]
    pub config: Map<String, Json>,
}

impl PluginComponentSpec {
    /// Creates a new enabled component spec with empty config.
    pub fn new(kind: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            enabled: true,
            config: Map::new(),
        }
    }
}

/// Bounded aggregate for a runtime failure observed by an active plugin.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct RuntimeDiagnostic {
    /// Stable failure classification.
    pub code: String,
    /// Plugin component that reported the failure.
    pub component: String,
    /// Optional configuration field associated with the failure.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub field: Option<String>,
    /// Latest human-readable failure detail.
    pub message: String,
    /// Latest affected trajectory session identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Number of failures aggregated into this entry.
    pub count: u64,
}

/// Policy for how unsupported plugin/runtime config is handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ConfigPolicy {
    /// Policy applied when a component kind is unknown to the plugin registry.
    #[serde(default = "default_warn")]
    pub unknown_component: UnsupportedBehavior,
    /// Policy applied when a known component contains an unknown field.
    #[serde(default = "default_warn")]
    pub unknown_field: UnsupportedBehavior,
    /// Policy applied when a known field contains an unsupported value.
    #[serde(default = "default_error")]
    pub unsupported_value: UnsupportedBehavior,
}

impl Default for ConfigPolicy {
    fn default() -> Self {
        Self {
            unknown_component: default_warn(),
            unknown_field: default_warn(),
            unsupported_value: default_error(),
        }
    }
}

/// Applies non-default global policy fields to a component policy.
///
/// Component-specific policy remains in effect when the corresponding global
/// setting is left at its default value.
pub fn apply_global_config_policy(
    component_policy: ConfigPolicy,
    global_policy: &ConfigPolicy,
) -> ConfigPolicy {
    let default_policy = ConfigPolicy::default();
    ConfigPolicy {
        unknown_component: if global_policy.unknown_component != default_policy.unknown_component {
            global_policy.unknown_component
        } else {
            component_policy.unknown_component
        },
        unknown_field: if global_policy.unknown_field != default_policy.unknown_field {
            global_policy.unknown_field
        } else {
            component_policy.unknown_field
        },
        unsupported_value: if global_policy.unsupported_value != default_policy.unsupported_value {
            global_policy.unsupported_value
        } else {
            component_policy.unsupported_value
        },
    }
}

nemo_relay_types::editor_config! {
    impl ConfigPolicy {
        unknown_component => {
            label: "unknown_component",
            kind: Enum,
            values: ["warn", "ignore", "error"],
        },
        unknown_field => {
            label: "unknown_field",
            kind: Enum,
            values: ["warn", "ignore", "error"],
        },
        unsupported_value => {
            label: "unsupported_value",
            kind: Enum,
            values: ["warn", "ignore", "error"],
        },
    }
}

/// Per-policy behavior for unsupported configuration.
#[derive(
    Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema,
)]
#[serde(rename_all = "lowercase")]
pub enum UnsupportedBehavior {
    /// Suppress the diagnostic entirely.
    Ignore,
    /// Emit a warning diagnostic.
    #[default]
    Warn,
    /// Emit an error diagnostic.
    Error,
}

fn default_warn() -> UnsupportedBehavior {
    UnsupportedBehavior::Warn
}

fn default_error() -> UnsupportedBehavior {
    UnsupportedBehavior::Error
}

fn default_enabled() -> bool {
    true
}

/// Bookkeeping for one middleware/subscriber registration.
pub struct PluginRegistration {
    /// Registration kind used for bookkeeping.
    pub kind: String,
    /// Runtime-qualified registration name.
    pub name: String,
    deregister: Box<dyn FnMut() -> PluginRegistrationCleanupOutcome + Send>,
}

/// Outcome of a tracked resource cleanup operation.
pub enum PluginRegistrationCleanupOutcome {
    /// The resource was removed.
    Removed,
    /// The resource was removed but final delivery failed.
    RemovedWithError(PluginError),
    /// The resource remains registered.
    NotRemoved(PluginError),
}

impl fmt::Debug for PluginRegistration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PluginRegistration")
            .field("kind", &self.kind)
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl PluginRegistration {
    /// Creates a new registration bookkeeping entry.
    pub fn new(
        kind: impl Into<String>,
        name: impl Into<String>,
        mut deregister: Box<dyn FnMut() -> Result<()> + Send>,
    ) -> Self {
        Self {
            kind: kind.into(),
            name: name.into(),
            deregister: Box::new(move || match deregister() {
                Ok(()) => PluginRegistrationCleanupOutcome::Removed,
                Err(error) => PluginRegistrationCleanupOutcome::NotRemoved(error),
            }),
        }
    }

    /// Construct a cleanup entry that reports whether the resource was removed.
    pub fn new_with_outcome(
        kind: impl Into<String>,
        name: impl Into<String>,
        deregister: Box<dyn FnMut() -> PluginRegistrationCleanupOutcome + Send>,
    ) -> Self {
        Self {
            kind: kind.into(),
            name: name.into(),
            deregister,
        }
    }
}

/// Implemented by custom plugins that register runtime middleware.
pub trait Plugin: Send + Sync + 'static {
    /// Returns the unique plugin kind string.
    fn plugin_kind(&self) -> &str;

    /// Returns whether the plugin kind can appear multiple times in the config.
    ///
    /// Return `false` for singleton components such as the built-in adaptive
    /// component.
    fn allows_multiple_components(&self) -> bool {
        true
    }

    /// Validates one plugin component config.
    ///
    /// Returning error-level diagnostics prevents plugin-host activation
    /// from activating the configuration.
    fn validate(&self, plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic>;

    /// Validates one plugin component config using the host's global policy.
    ///
    /// The default preserves the validation behavior of existing custom
    /// plugins. Plugins that emit policy-controlled diagnostics should
    /// override this method so host validation honors non-default fields in
    /// [`ConfigPolicy`].
    fn validate_with_policy(
        &self,
        plugin_config: &Map<String, Json>,
        _policy: &ConfigPolicy,
    ) -> Vec<ConfigDiagnostic> {
        self.validate(plugin_config)
    }

    /// Registers runtime middleware/subscribers for one plugin component.
    ///
    /// The provided [`PluginRegistrationContext`] is component-scoped. Any
    /// error aborts the current initialization and triggers rollback of
    /// registrations created during the failed activation attempt.
    fn register<'a>(
        &'a self,
        plugin_config: &Map<String, Json>,
        ctx: &'a mut dyn PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;
}

impl PluginRegistration {
    /// Execute this registration's tracked cleanup operation.
    pub fn cleanup(&mut self) -> PluginRegistrationCleanupOutcome {
        (self.deregister)()
    }
}

/// Host capabilities available to plugins during component activation.
///
/// Hosts retain ownership of registry mutation, namespacing, rollback, and
/// activation diagnostics. Implementations do not depend on a runtime crate.
pub trait PluginRegistrationContext: Send {
    /// Optional diagnostic producer context supplied by the embedding runtime.
    ///
    /// Runtime-specific components may downcast this handle to their runtime's
    /// diagnostic facade. Shared plugins must tolerate an unavailable context.
    fn diagnostics_context(&self) -> Option<&(dyn std::any::Any + Send + Sync)> {
        None
    }

    /// Qualify a component-local registration name.
    fn qualify_name(&self, name: &str) -> String;
    /// Install a subscriber and track its rollback.
    fn register_subscriber(
        &mut self,
        name: &str,
        callback: Arc<dyn Fn(&Event) + Send + Sync>,
    ) -> Result<()>;
    /// Track an already-created resource's cleanup operation.
    fn add_registration(&mut self, registration: PluginRegistration);
    /// Record an activation diagnostic.
    fn record_activation_diagnostic(&mut self, diagnostic: ConfigDiagnostic);
    /// Record an optional exporter initialization failure without aborting activation.
    fn record_nonfatal_activation_failure(
        &mut self,
        component_key: &'static str,
        classification: Classification,
        diagnostic: ConfigDiagnostic,
    );
    /// Read activation diagnostics recorded so far.
    fn activation_diagnostics(&self) -> &[ConfigDiagnostic];
    /// Install middleware and track its rollback.
    fn register_conditional_middleware_guardrail(
        &mut self,
        name: &str,
        kinds: BTreeSet<RuntimeRegistrationKind>,
        registration_name: &str,
        guardrail: ConditionalMiddlewareGuardrailFn,
    ) -> Result<()>;

    /// Install middleware and track its rollback.
    fn register_event_metadata_injector(
        &mut self,
        name: &str,
        priority: i32,
        callback: EventMetadataInjectorFn,
    ) -> Result<()>;

    /// Install middleware and track its rollback.
    fn register_mark_sanitize_guardrail(
        &mut self,
        name: &str,
        priority: i32,
        callback: EventSanitizeFn,
    ) -> Result<()>;

    /// Install middleware and track its rollback.
    fn register_scope_sanitize_start_guardrail(
        &mut self,
        name: &str,
        priority: i32,
        callback: EventSanitizeFn,
    ) -> Result<()>;

    /// Install middleware and track its rollback.
    fn register_scope_sanitize_end_guardrail(
        &mut self,
        name: &str,
        priority: i32,
        callback: EventSanitizeFn,
    ) -> Result<()>;

    /// Install middleware and track its rollback.
    fn register_llm_request_intercept(
        &mut self,
        name: &str,
        priority: i32,
        break_chain: bool,
        callback: LlmRequestInterceptFn,
    ) -> Result<()>;

    /// Install middleware and track its rollback.
    fn register_tool_sanitize_request_guardrail(
        &mut self,
        name: &str,
        priority: i32,
        callback: ToolSanitizeFn,
    ) -> Result<()>;

    /// Install middleware and track its rollback.
    fn register_tool_sanitize_response_guardrail(
        &mut self,
        name: &str,
        priority: i32,
        callback: ToolSanitizeFn,
    ) -> Result<()>;

    /// Install middleware and track its rollback.
    fn register_tool_conditional_execution_guardrail(
        &mut self,
        name: &str,
        priority: i32,
        callback: ToolConditionalFn,
    ) -> Result<()>;

    /// Install middleware and track its rollback.
    fn register_llm_sanitize_request_guardrail(
        &mut self,
        name: &str,
        priority: i32,
        callback: LlmSanitizeRequestFn,
    ) -> Result<()>;

    /// Install middleware and track its rollback.
    fn register_llm_sanitize_response_guardrail(
        &mut self,
        name: &str,
        priority: i32,
        callback: LlmSanitizeResponseFn,
    ) -> Result<()>;

    /// Install middleware and track its rollback.
    fn register_llm_conditional_execution_guardrail(
        &mut self,
        name: &str,
        priority: i32,
        callback: LlmConditionalFn,
    ) -> Result<()>;

    /// Install middleware and track its rollback.
    fn register_llm_execution_intercept(
        &mut self,
        name: &str,
        priority: i32,
        callback: LlmExecutionFn,
    ) -> Result<()>;

    /// Install middleware and track its rollback.
    fn register_llm_stream_execution_intercept(
        &mut self,
        name: &str,
        priority: i32,
        callback: LlmStreamExecutionFn,
    ) -> Result<()>;

    /// Install middleware and track its rollback.
    fn register_tool_request_intercept(
        &mut self,
        name: &str,
        priority: i32,
        break_chain: bool,
        callback: ToolInterceptFn,
    ) -> Result<()>;

    /// Install middleware and track its rollback.
    fn register_tool_execution_intercept(
        &mut self,
        name: &str,
        priority: i32,
        callback: ToolExecutionFn,
    ) -> Result<()>;

    /// Whether registration names belong to a plugin component namespace.
    fn uses_plugin_component_namespace(&self) -> bool;
    /// Create an independent registration context nested under this namespace.
    fn with_child_namespace(&self, name: &str) -> Box<dyn PluginRegistrationContext>;
    /// Consume this context's cleanup entries.
    fn into_registrations(self: Box<Self>) -> Vec<PluginRegistration>;
    /// Track cleanup entries returned by a nested context or language adapter.
    fn extend_registrations(&mut self, registrations: Vec<PluginRegistration>) {
        for registration in registrations {
            self.add_registration(registration);
        }
    }
}
