// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generic plugin infrastructure for NeMo Relay runtimes.
//!
//! This module owns:
//! - config diagnostics and policy enums used by plugin systems
//! - a global plugin registry
//! - plugin registration contexts for middleware/subscriber installation
//! - rollback bookkeeping for registrations created during plugin setup

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;
use std::future::Future;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, OnceLock, RwLock};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as Json};
use thiserror::Error;

use crate::api::registry::{
    RuntimeRegistrationKind, deregister_conditional_middleware_guardrail,
    deregister_event_metadata_injector, deregister_llm_conditional_execution_guardrail,
    deregister_llm_execution_intercept, deregister_llm_request_intercept,
    deregister_llm_sanitize_request_guardrail, deregister_llm_sanitize_response_guardrail,
    deregister_llm_stream_execution_intercept, deregister_mark_sanitize_guardrail,
    deregister_scope_sanitize_end_guardrail, deregister_scope_sanitize_start_guardrail,
    deregister_tool_conditional_execution_guardrail, deregister_tool_execution_intercept,
    deregister_tool_request_intercept, deregister_tool_sanitize_request_guardrail,
    deregister_tool_sanitize_response_guardrail, register_conditional_middleware_guardrail,
    register_event_metadata_injector, register_llm_conditional_execution_guardrail,
    register_llm_execution_intercept, register_llm_request_intercept,
    register_llm_sanitize_request_guardrail, register_llm_sanitize_response_guardrail,
    register_llm_stream_execution_intercept, register_mark_sanitize_guardrail,
    register_scope_sanitize_end_guardrail, register_scope_sanitize_start_guardrail,
    register_tool_conditional_execution_guardrail, register_tool_execution_intercept,
    register_tool_request_intercept, register_tool_sanitize_request_guardrail,
    register_tool_sanitize_response_guardrail,
};
use crate::api::runtime::{
    ConditionalMiddlewareGuardrailFn, EventMetadataInjectorFn, EventSanitizeFn, EventSubscriberFn,
    LlmConditionalFn, LlmExecutionFn, LlmRequestInterceptFn, LlmSanitizeRequestFn,
    LlmSanitizeResponseFn, LlmStreamExecutionFn, ToolConditionalFn, ToolExecutionFn,
    ToolInterceptFn, ToolSanitizeFn,
};
use crate::api::subscriber::{deregister_subscriber, register_subscriber};
pub use nemo_relay_types::plugin::{ConfigDiagnostic, DiagnosticLevel};

pub mod dynamic;
pub use dynamic::*;

type PluginMap = HashMap<String, RegisteredPlugin>;

struct RegisteredPlugin {
    registration_id: u64,
    owner: PluginRegistrationOwner,
    plugin: Arc<dyn Plugin>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PluginRegistrationOwner {
    Builtin,
    External,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PluginDeregistrationOutcome {
    Removed,
    Missing,
    Replaced,
}

static PLUGIN_HANDLERS: LazyLock<RwLock<PluginMap>> = LazyLock::new(|| RwLock::new(HashMap::new()));
static ACTIVE_PLUGIN_CONFIGURATION: LazyLock<Mutex<Option<ActivePluginConfiguration>>> =
    LazyLock::new(|| Mutex::new(None));
static LAST_FAILED_RUNTIME_DIAGNOSTICS_REPORT: LazyLock<Mutex<Option<ConfigReport>>> =
    LazyLock::new(|| Mutex::new(None));
static PLUGIN_MUTATION_OWNER: LazyLock<Mutex<PluginMutationOwner>> =
    LazyLock::new(|| Mutex::new(PluginMutationOwner::Idle));
static NEXT_PLUGIN_REGISTRATION_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_PLUGIN_HOST_OWNER_ID: AtomicU64 = AtomicU64::new(1);
static PLUGIN_MUTATION_EXECUTOR: OnceLock<PluginMutationSender> = OnceLock::new();

type PluginMutationJob = Pin<Box<dyn Future<Output = ()> + Send + 'static>>;
type PluginMutationSender = tokio::sync::mpsc::UnboundedSender<PluginMutationJob>;

thread_local! {
    static IN_PLUGIN_MUTATION_EXECUTOR: Cell<bool> = const { Cell::new(false) };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PluginMutationOwner {
    Idle,
    Legacy,
    Host(u64),
}

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

/// Specialized [`Result`](std::result::Result) type for plugin operations.
pub type Result<T> = std::result::Result<T, PluginError>;

/// Identifies teardown errors caused by recoverable ATIF delivery failures.
pub(crate) const ATIF_RUNTIME_DELIVERY_FAILURE_MARKER: &str = "ATIF runtime delivery failures";
/// Identifies teardown errors caused by recoverable OpenTelemetry delivery failures.
pub(crate) const OTEL_RUNTIME_DELIVERY_FAILURE_MARKER: &str =
    "OpenTelemetry runtime delivery failures";

/// Canonical plugin configuration document.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct PluginConfig {
    /// Plugin config schema version.
    #[serde(default = "default_plugin_config_version")]
    pub version: u32,
    /// Ordered list of top-level plugin components to validate and activate.
    #[serde(default)]
    pub components: Vec<PluginComponentSpec>,
    /// Plugin-level policy for unsupported plugin kinds, fields, and values.
    ///
    /// Non-default field values override the corresponding component policy.
    #[serde(default)]
    pub policy: ConfigPolicy,
}

impl Default for PluginConfig {
    fn default() -> Self {
        Self {
            version: default_plugin_config_version(),
            components: vec![],
            policy: ConfigPolicy::default(),
        }
    }
}

/// One configured plugin component.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
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

/// Structured validation report.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ConfigReport {
    /// Validation and compatibility diagnostics in evaluation order.
    #[serde(default)]
    pub diagnostics: Vec<ConfigDiagnostic>,
    /// Runtime delivery diagnostics recorded after activation.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub runtime_diagnostics: Vec<RuntimeDiagnostic>,
}

/// Bounded aggregate for a runtime failure observed by an active plugin.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
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

/// Read-only projection of runtime diagnostics exposed to dynamic plugins.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct RuntimeDiagnosticsSnapshotEntry {
    pub(crate) code: String,
    pub(crate) message: String,
    pub(crate) count: u64,
}

const MAX_DYNAMIC_PLUGIN_RUNTIME_DIAGNOSTICS: usize = 32;

impl ConfigReport {
    /// Returns `true` when the report contains at least one error diagnostic.
    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|diag| diag.level == DiagnosticLevel::Error)
    }
}

/// Policy for how unsupported plugin/runtime config is handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
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

crate::editor_config! {
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
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
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

fn default_plugin_config_version() -> u32 {
    1
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

pub(crate) enum PluginRegistrationCleanupOutcome {
    Removed,
    RemovedWithError(PluginError),
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

    pub(crate) fn new_with_outcome(
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

/// Context provided to plugin handlers during runtime registration.
///
/// Each `register_*` call both installs the middleware/subscriber into the
/// NeMo Relay runtime and records the inverse deregistration closure so the host
/// can roll back partial setup on failure.
#[derive(Default)]
pub struct PluginRegistrationContext {
    registrations: Vec<PluginRegistration>,
    namespace: Option<PluginRegistrationNamespace>,
}

enum PluginRegistrationNamespace {
    Plain(String),
    PluginComponent(String),
}

impl PluginRegistrationContext {
    /// Creates an empty plugin registration context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a plugin registration context that namespaces all registration names.
    pub fn with_namespace(namespace: impl Into<String>) -> Self {
        Self {
            registrations: vec![],
            namespace: Some(PluginRegistrationNamespace::Plain(namespace.into())),
        }
    }

    fn with_plugin_component_namespace(namespace: String) -> Self {
        Self {
            registrations: vec![],
            namespace: Some(PluginRegistrationNamespace::PluginComponent(namespace)),
        }
    }

    /// Creates a child context that extends this context's namespace.
    ///
    /// Child contexts preserve Relay-created plugin component qualification so
    /// their local namespace segments and registration names remain encoded as
    /// one effective component name.
    pub fn with_child_namespace(&self, local_namespace: &str) -> Self {
        let namespace = match &self.namespace {
            Some(PluginRegistrationNamespace::Plain(namespace)) => {
                PluginRegistrationNamespace::Plain(format!("{namespace}{local_namespace}"))
            }
            Some(PluginRegistrationNamespace::PluginComponent(namespace)) => {
                PluginRegistrationNamespace::PluginComponent(format!(
                    "{namespace}{}",
                    encode_plugin_component_field(local_namespace)
                ))
            }
            None => PluginRegistrationNamespace::Plain(local_namespace.to_string()),
        };
        Self {
            registrations: vec![],
            namespace: Some(namespace),
        }
    }

    /// Returns the runtime-qualified name for a plugin-local registration.
    ///
    /// Plugin handlers should pass stable component-local names such as
    /// `"tool"` or `"subscriber"`. The host applies the namespace so users do
    /// not have to provide component instance ids.
    pub fn qualify_name(&self, name: &str) -> String {
        match &self.namespace {
            Some(PluginRegistrationNamespace::Plain(namespace)) => format!("{namespace}{name}"),
            Some(PluginRegistrationNamespace::PluginComponent(namespace)) => {
                format!("{namespace}{}", encode_plugin_component_field(name))
            }
            None => name.to_string(),
        }
    }

    pub(crate) fn uses_plugin_component_namespace(&self) -> bool {
        matches!(
            &self.namespace,
            Some(PluginRegistrationNamespace::PluginComponent(_))
        )
    }

    /// Registers an event subscriber and records its rollback closure.
    pub fn register_subscriber(&mut self, name: &str, callback: EventSubscriberFn) -> Result<()> {
        let qualified_name = self.qualify_name(name);
        register_subscriber(&qualified_name, callback)
            .map_err(|err| PluginError::RegistrationFailed(format!("subscriber: {err}")))?;

        let name_owned = qualified_name;
        self.registrations.push(PluginRegistration::new(
            "plugin",
            name_owned.clone(),
            Box::new(move || {
                deregister_subscriber(&name_owned)
                    .map(|_| ())
                    .map_err(|err| {
                        PluginError::RegistrationFailed(format!(
                            "subscriber deregistration failed: {err}"
                        ))
                    })
            }),
        ));
        Ok(())
    }

    /// Registers a global conditional middleware guardrail and records its
    /// rollback closure.
    pub fn register_conditional_middleware_guardrail(
        &mut self,
        name: &str,
        kinds: BTreeSet<RuntimeRegistrationKind>,
        registration_name: &str,
        guardrail: ConditionalMiddlewareGuardrailFn,
    ) -> Result<()> {
        let qualified_name = self.qualify_name(name);
        register_conditional_middleware_guardrail(
            &qualified_name,
            kinds,
            registration_name,
            guardrail,
        )
        .map_err(|err| {
            PluginError::RegistrationFailed(format!("conditional middleware guardrail: {err}"))
        })?;

        let name_owned = qualified_name;
        self.registrations.push(PluginRegistration::new(
            "plugin",
            name_owned.clone(),
            Box::new(move || {
                deregister_conditional_middleware_guardrail(&name_owned)
                    .map(|_| ())
                    .map_err(|err| {
                        PluginError::RegistrationFailed(format!(
                            "conditional middleware guardrail deregistration failed: {err}"
                        ))
                    })
            }),
        ));
        Ok(())
    }

    /// Registers an Event metadata injector and records its rollback closure.
    pub fn register_event_metadata_injector(
        &mut self,
        name: &str,
        priority: i32,
        callback: EventMetadataInjectorFn,
    ) -> Result<()> {
        let qualified_name = self.qualify_name(name);
        register_event_metadata_injector(&qualified_name, priority, callback).map_err(|err| {
            PluginError::RegistrationFailed(format!("event metadata injector: {err}"))
        })?;

        let name_owned = qualified_name;
        self.registrations.push(PluginRegistration::new(
            "plugin",
            name_owned.clone(),
            Box::new(move || {
                deregister_event_metadata_injector(&name_owned)
                    .map(|_| ())
                    .map_err(|err| {
                        PluginError::RegistrationFailed(format!(
                            "event metadata injector deregistration failed: {err}"
                        ))
                    })
            }),
        ));
        Ok(())
    }
    /// Registers a mark event sanitizer and records its rollback closure.
    pub fn register_mark_sanitize_guardrail(
        &mut self,
        name: &str,
        priority: i32,
        callback: EventSanitizeFn,
    ) -> Result<()> {
        let qualified_name = self.qualify_name(name);
        register_mark_sanitize_guardrail(&qualified_name, priority, callback)
            .map_err(|err| PluginError::RegistrationFailed(format!("mark sanitizer: {err}")))?;
        let name_owned = qualified_name;
        self.registrations.push(PluginRegistration::new(
            "plugin",
            name_owned.clone(),
            Box::new(move || {
                deregister_mark_sanitize_guardrail(&name_owned)
                    .map(|_| ())
                    .map_err(|err| {
                        PluginError::RegistrationFailed(format!(
                            "mark sanitizer deregistration failed: {err}"
                        ))
                    })
            }),
        ));
        Ok(())
    }

    /// Registers a scope-start event sanitizer and records its rollback closure.
    pub fn register_scope_sanitize_start_guardrail(
        &mut self,
        name: &str,
        priority: i32,
        callback: EventSanitizeFn,
    ) -> Result<()> {
        let qualified_name = self.qualify_name(name);
        register_scope_sanitize_start_guardrail(&qualified_name, priority, callback).map_err(
            |err| PluginError::RegistrationFailed(format!("scope-start sanitizer: {err}")),
        )?;
        let name_owned = qualified_name;
        self.registrations.push(PluginRegistration::new(
            "plugin",
            name_owned.clone(),
            Box::new(move || {
                deregister_scope_sanitize_start_guardrail(&name_owned)
                    .map(|_| ())
                    .map_err(|err| {
                        PluginError::RegistrationFailed(format!(
                            "scope-start sanitizer deregistration failed: {err}"
                        ))
                    })
            }),
        ));
        Ok(())
    }

    /// Registers a scope-end event sanitizer and records its rollback closure.
    pub fn register_scope_sanitize_end_guardrail(
        &mut self,
        name: &str,
        priority: i32,
        callback: EventSanitizeFn,
    ) -> Result<()> {
        let qualified_name = self.qualify_name(name);
        register_scope_sanitize_end_guardrail(&qualified_name, priority, callback).map_err(
            |err| PluginError::RegistrationFailed(format!("scope-end sanitizer: {err}")),
        )?;
        let name_owned = qualified_name;
        self.registrations.push(PluginRegistration::new(
            "plugin",
            name_owned.clone(),
            Box::new(move || {
                deregister_scope_sanitize_end_guardrail(&name_owned)
                    .map(|_| ())
                    .map_err(|err| {
                        PluginError::RegistrationFailed(format!(
                            "scope-end sanitizer deregistration failed: {err}"
                        ))
                    })
            }),
        ));
        Ok(())
    }

    /// Registers an LLM request intercept and records its rollback closure.
    pub fn register_llm_request_intercept(
        &mut self,
        name: &str,
        priority: i32,
        break_chain: bool,
        callback: LlmRequestInterceptFn,
    ) -> Result<()> {
        let qualified_name = self.qualify_name(name);
        register_llm_request_intercept(&qualified_name, priority, break_chain, callback).map_err(
            |err| PluginError::RegistrationFailed(format!("llm request intercept: {err}")),
        )?;

        let name_owned = qualified_name;
        self.registrations.push(PluginRegistration::new(
            "plugin",
            name_owned.clone(),
            Box::new(move || {
                deregister_llm_request_intercept(&name_owned)
                    .map(|_| ())
                    .map_err(|err| {
                        PluginError::RegistrationFailed(format!(
                            "llm request intercept deregistration failed: {err}"
                        ))
                    })
            }),
        ));
        Ok(())
    }

    /// Registers a tool sanitize-request guardrail and records its rollback closure.
    pub fn register_tool_sanitize_request_guardrail(
        &mut self,
        name: &str,
        priority: i32,
        callback: ToolSanitizeFn,
    ) -> Result<()> {
        let qualified_name = self.qualify_name(name);
        register_tool_sanitize_request_guardrail(&qualified_name, priority, callback).map_err(
            |err| {
                PluginError::RegistrationFailed(format!("tool sanitize request guardrail: {err}"))
            },
        )?;

        let name_owned = qualified_name;
        self.registrations.push(PluginRegistration::new(
            "plugin",
            name_owned.clone(),
            Box::new(move || {
                deregister_tool_sanitize_request_guardrail(&name_owned)
                    .map(|_| ())
                    .map_err(|err| {
                        PluginError::RegistrationFailed(format!(
                            "tool sanitize request guardrail deregistration failed: {err}"
                        ))
                    })
            }),
        ));
        Ok(())
    }

    /// Registers a tool sanitize-response guardrail and records its rollback closure.
    pub fn register_tool_sanitize_response_guardrail(
        &mut self,
        name: &str,
        priority: i32,
        callback: ToolSanitizeFn,
    ) -> Result<()> {
        let qualified_name = self.qualify_name(name);
        register_tool_sanitize_response_guardrail(&qualified_name, priority, callback).map_err(
            |err| {
                PluginError::RegistrationFailed(format!("tool sanitize response guardrail: {err}"))
            },
        )?;

        let name_owned = qualified_name;
        self.registrations.push(PluginRegistration::new(
            "plugin",
            name_owned.clone(),
            Box::new(move || {
                deregister_tool_sanitize_response_guardrail(&name_owned)
                    .map(|_| ())
                    .map_err(|err| {
                        PluginError::RegistrationFailed(format!(
                            "tool sanitize response guardrail deregistration failed: {err}"
                        ))
                    })
            }),
        ));
        Ok(())
    }

    /// Registers a tool conditional-execution guardrail and records its rollback closure.
    pub fn register_tool_conditional_execution_guardrail(
        &mut self,
        name: &str,
        priority: i32,
        callback: ToolConditionalFn,
    ) -> Result<()> {
        let qualified_name = self.qualify_name(name);
        register_tool_conditional_execution_guardrail(&qualified_name, priority, callback)
            .map_err(|err| {
                PluginError::RegistrationFailed(format!(
                    "tool conditional execution guardrail: {err}"
                ))
            })?;

        let name_owned = qualified_name;
        self.registrations.push(PluginRegistration::new(
            "plugin",
            name_owned.clone(),
            Box::new(move || {
                deregister_tool_conditional_execution_guardrail(&name_owned)
                    .map(|_| ())
                    .map_err(|err| {
                        PluginError::RegistrationFailed(format!(
                            "tool conditional execution guardrail deregistration failed: {err}"
                        ))
                    })
            }),
        ));
        Ok(())
    }

    /// Registers an LLM sanitize-request guardrail and records its rollback closure.
    pub fn register_llm_sanitize_request_guardrail(
        &mut self,
        name: &str,
        priority: i32,
        callback: LlmSanitizeRequestFn,
    ) -> Result<()> {
        let qualified_name = self.qualify_name(name);
        register_llm_sanitize_request_guardrail(&qualified_name, priority, callback).map_err(
            |err| PluginError::RegistrationFailed(format!("llm sanitize request guardrail: {err}")),
        )?;

        let name_owned = qualified_name;
        self.registrations.push(PluginRegistration::new(
            "plugin",
            name_owned.clone(),
            Box::new(move || {
                deregister_llm_sanitize_request_guardrail(&name_owned)
                    .map(|_| ())
                    .map_err(|err| {
                        PluginError::RegistrationFailed(format!(
                            "llm sanitize request guardrail deregistration failed: {err}"
                        ))
                    })
            }),
        ));
        Ok(())
    }

    /// Registers an LLM sanitize-response guardrail and records its rollback closure.
    pub fn register_llm_sanitize_response_guardrail(
        &mut self,
        name: &str,
        priority: i32,
        callback: LlmSanitizeResponseFn,
    ) -> Result<()> {
        let qualified_name = self.qualify_name(name);
        register_llm_sanitize_response_guardrail(&qualified_name, priority, callback).map_err(
            |err| {
                PluginError::RegistrationFailed(format!("llm sanitize response guardrail: {err}"))
            },
        )?;

        let name_owned = qualified_name;
        self.registrations.push(PluginRegistration::new(
            "plugin",
            name_owned.clone(),
            Box::new(move || {
                deregister_llm_sanitize_response_guardrail(&name_owned)
                    .map(|_| ())
                    .map_err(|err| {
                        PluginError::RegistrationFailed(format!(
                            "llm sanitize response guardrail deregistration failed: {err}"
                        ))
                    })
            }),
        ));
        Ok(())
    }

    /// Registers an LLM conditional-execution guardrail and records its rollback closure.
    pub fn register_llm_conditional_execution_guardrail(
        &mut self,
        name: &str,
        priority: i32,
        callback: LlmConditionalFn,
    ) -> Result<()> {
        let qualified_name = self.qualify_name(name);
        register_llm_conditional_execution_guardrail(&qualified_name, priority, callback).map_err(
            |err| {
                PluginError::RegistrationFailed(format!(
                    "llm conditional execution guardrail: {err}"
                ))
            },
        )?;

        let name_owned = qualified_name;
        self.registrations.push(PluginRegistration::new(
            "plugin",
            name_owned.clone(),
            Box::new(move || {
                deregister_llm_conditional_execution_guardrail(&name_owned)
                    .map(|_| ())
                    .map_err(|err| {
                        PluginError::RegistrationFailed(format!(
                            "llm conditional execution guardrail deregistration failed: {err}"
                        ))
                    })
            }),
        ));
        Ok(())
    }

    /// Registers an LLM execution intercept and records its rollback closure.
    pub fn register_llm_execution_intercept(
        &mut self,
        name: &str,
        priority: i32,
        callback: LlmExecutionFn,
    ) -> Result<()> {
        let qualified_name = self.qualify_name(name);
        register_llm_execution_intercept(&qualified_name, priority, callback).map_err(|err| {
            PluginError::RegistrationFailed(format!("llm execution intercept: {err}"))
        })?;

        let name_owned = qualified_name;
        self.registrations.push(PluginRegistration::new(
            "plugin",
            name_owned.clone(),
            Box::new(move || {
                deregister_llm_execution_intercept(&name_owned)
                    .map(|_| ())
                    .map_err(|err| {
                        PluginError::RegistrationFailed(format!(
                            "llm execution intercept deregistration failed: {err}"
                        ))
                    })
            }),
        ));
        Ok(())
    }

    /// Registers an LLM stream execution intercept and records its rollback closure.
    pub fn register_llm_stream_execution_intercept(
        &mut self,
        name: &str,
        priority: i32,
        callback: LlmStreamExecutionFn,
    ) -> Result<()> {
        let qualified_name = self.qualify_name(name);
        register_llm_stream_execution_intercept(&qualified_name, priority, callback).map_err(
            |err| PluginError::RegistrationFailed(format!("llm stream execution intercept: {err}")),
        )?;

        let name_owned = qualified_name;
        self.registrations.push(PluginRegistration::new(
            "plugin",
            name_owned.clone(),
            Box::new(move || {
                deregister_llm_stream_execution_intercept(&name_owned)
                    .map(|_| ())
                    .map_err(|err| {
                        PluginError::RegistrationFailed(format!(
                            "llm stream execution intercept deregistration failed: {err}"
                        ))
                    })
            }),
        ));
        Ok(())
    }

    /// Registers a tool request intercept and records its rollback closure.
    pub fn register_tool_request_intercept(
        &mut self,
        name: &str,
        priority: i32,
        break_chain: bool,
        callback: ToolInterceptFn,
    ) -> Result<()> {
        let qualified_name = self.qualify_name(name);
        register_tool_request_intercept(&qualified_name, priority, break_chain, callback).map_err(
            |err| PluginError::RegistrationFailed(format!("tool request intercept: {err}")),
        )?;

        let name_owned = qualified_name;
        self.registrations.push(PluginRegistration::new(
            "plugin",
            name_owned.clone(),
            Box::new(move || {
                deregister_tool_request_intercept(&name_owned)
                    .map(|_| ())
                    .map_err(|err| {
                        PluginError::RegistrationFailed(format!(
                            "tool request intercept deregistration failed: {err}"
                        ))
                    })
            }),
        ));
        Ok(())
    }

    /// Registers a tool execution intercept and records its rollback closure.
    pub fn register_tool_execution_intercept(
        &mut self,
        name: &str,
        priority: i32,
        callback: ToolExecutionFn,
    ) -> Result<()> {
        let qualified_name = self.qualify_name(name);
        register_tool_execution_intercept(&qualified_name, priority, callback).map_err(|err| {
            PluginError::RegistrationFailed(format!("tool execution intercept: {err}"))
        })?;

        let name_owned = qualified_name;
        self.registrations.push(PluginRegistration::new(
            "plugin",
            name_owned.clone(),
            Box::new(move || {
                deregister_tool_execution_intercept(&name_owned)
                    .map(|_| ())
                    .map_err(|err| {
                        PluginError::RegistrationFailed(format!(
                            "tool execution intercept deregistration failed: {err}"
                        ))
                    })
            }),
        ));
        Ok(())
    }

    /// Adds a prebuilt registration to the context.
    pub fn add_registration(&mut self, registration: PluginRegistration) {
        self.registrations.push(registration);
    }

    /// Extends the context with prebuilt registrations.
    pub fn extend_registrations(&mut self, registrations: Vec<PluginRegistration>) {
        self.registrations.extend(registrations);
    }

    /// Consumes the context and returns the recorded registrations.
    pub fn into_registrations(self) -> Vec<PluginRegistration> {
        self.registrations
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
    /// Returning error-level diagnostics prevents `initialize_plugins(...)`
    /// from activating the configuration.
    fn validate(&self, plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic>;

    /// Validates one plugin component config using the host's global policy.
    ///
    /// The default preserves the validation behavior of existing custom
    /// plugins. Plugins that emit policy-controlled diagnostics should
    /// override this method so host validation honors non-default fields in
    /// [`PluginConfig::policy`].
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
        ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;
}

/// Registers a plugin by kind.
///
/// Registering the same kind twice returns [`PluginError::RegistrationFailed`].
/// Register a plugin kind with the global plugin registry.
///
/// Registered plugins can then participate in validation and initialization of
/// [`PluginConfig`] documents.
///
/// # Parameters
/// - `plugin`: Plugin implementation to register.
///
/// # Returns
/// A plugin [`Result`] that is `Ok(())` when the plugin kind was added
/// to the registry.
///
/// # Errors
/// Returns an error when a plugin with the same kind is already registered or
/// when the registry lock is poisoned.
///
/// # Notes
/// Registration affects future validation and initialization only.
pub fn register_plugin(plugin: Arc<dyn Plugin>) -> Result<()> {
    register_plugin_with_owner(plugin, PluginRegistrationOwner::External).map(|_| ())
}

pub(crate) fn register_plugin_tracked(plugin: Arc<dyn Plugin>) -> Result<u64> {
    register_plugin_with_owner(plugin, PluginRegistrationOwner::External)
}

pub(crate) fn register_builtin_plugin(plugin: Arc<dyn Plugin>) -> Result<()> {
    let plugin_kind = plugin.plugin_kind();
    {
        let guard = PLUGIN_HANDLERS.read().map_err(|err| {
            PluginError::Internal(format!("plugin registry lock poisoned: {err}"))
        })?;
        if let Some(existing) = guard.get(plugin_kind) {
            if existing.owner == PluginRegistrationOwner::Builtin {
                return Ok(());
            }
            return Err(plugin_already_registered_error(
                plugin_kind,
                PluginRegistrationOwner::Builtin,
            ));
        }
    }

    register_plugin_with_owner(plugin, PluginRegistrationOwner::Builtin).map(|_| ())
}

fn plugin_already_registered_error(
    plugin_kind: &str,
    owner: PluginRegistrationOwner,
) -> PluginError {
    let ownership = if owner == PluginRegistrationOwner::Builtin {
        "reserved builtin "
    } else {
        ""
    };
    PluginError::RegistrationFailed(format!(
        "{ownership}plugin '{plugin_kind}' is already registered"
    ))
}

fn register_plugin_with_owner(
    plugin: Arc<dyn Plugin>,
    owner: PluginRegistrationOwner,
) -> Result<u64> {
    let mut guard = PLUGIN_HANDLERS
        .write()
        .map_err(|err| PluginError::Internal(format!("plugin registry lock poisoned: {err}")))?;
    let plugin_kind = plugin.plugin_kind().to_string();
    if let Some(existing) = guard.get(&plugin_kind) {
        if owner == PluginRegistrationOwner::Builtin
            && existing.owner == PluginRegistrationOwner::Builtin
        {
            return Ok(existing.registration_id);
        }
        return Err(plugin_already_registered_error(&plugin_kind, owner));
    }
    let registration_id = NEXT_PLUGIN_REGISTRATION_ID.fetch_add(1, Ordering::Relaxed);
    guard.insert(
        plugin_kind.clone(),
        RegisteredPlugin {
            registration_id,
            owner,
            plugin,
        },
    );
    log::info!(
        target: "nemo_relay.plugin",
        event = "plugin_registered",
        plugin_kind = plugin_kind.as_str(),
        registration_id = registration_id;
        "Plugin kind registered"
    );
    Ok(registration_id)
}

/// Registers core-provided plugin kinds.
///
/// Built-in plugins are available to validation and initialization without a
/// binding or application-specific registration call.
#[allow(
    deprecated,
    reason = "the host must register the built-in Guardrails plugin until its scheduled removal"
)]
pub fn ensure_builtin_plugins_registered() -> Result<()> {
    let all_registered = {
        let guard = PLUGIN_HANDLERS.read().map_err(|err| {
            PluginError::Internal(format!("plugin registry lock poisoned: {err}"))
        })?;
        [
            crate::observability::plugin_component::OBSERVABILITY_PLUGIN_KIND,
            crate::plugins::nemo_guardrails::component::NEMO_GUARDRAILS_PLUGIN_KIND,
            crate::plugins::model_pricing::PRICING_PLUGIN_KIND,
        ]
        .iter()
        .all(|kind| {
            guard
                .get(*kind)
                .is_some_and(|plugin| plugin.owner == PluginRegistrationOwner::Builtin)
        })
    };
    if all_registered {
        return Ok(());
    }

    // Registration is idempotent for genuine built-ins. Revalidate on every
    // call so a removed built-in is restored, a replacement is rejected, and
    // a corrected ownership conflict can be retried without restarting Relay.
    crate::observability::plugin_component::register_observability_component()?;
    crate::plugins::nemo_guardrails::component::register_nemo_guardrails_component()?;
    crate::plugins::model_pricing::register_pricing_component()
}

/// Removes a previously registered plugin.
///
/// This affects future validation and initialization only. Active runtime
/// registrations remain until cleared or replaced.
///
/// # Parameters
/// - `plugin_kind`: Plugin kind to remove from the registry.
///
/// # Returns
/// `true` when a plugin was removed from the registry and `false` when the
/// kind was not registered.
///
/// # Notes
/// Active component registrations created by previous initialization calls are
/// not removed by this function.
pub fn deregister_plugin(plugin_kind: &str) -> bool {
    deregister_plugin_checked(plugin_kind).unwrap_or(false)
}

pub(crate) fn deregister_plugin_checked(plugin_kind: &str) -> Result<bool> {
    let removed = PLUGIN_HANDLERS
        .write()
        .map(|mut guard| guard.remove(plugin_kind).is_some())
        .map_err(|err| PluginError::Internal(format!("plugin registry lock poisoned: {err}")))?;
    if removed {
        log::info!(
            target: "nemo_relay.plugin",
            event = "plugin_deregistered",
            plugin_kind = plugin_kind;
            "Plugin kind deregistered"
        );
    }
    Ok(removed)
}

pub(crate) fn deregister_plugin_registration_checked(
    plugin_kind: &str,
    expected_registration_id: u64,
) -> Result<PluginDeregistrationOutcome> {
    let mut guard = PLUGIN_HANDLERS
        .write()
        .map_err(|err| PluginError::Internal(format!("plugin registry lock poisoned: {err}")))?;
    match guard.get(plugin_kind) {
        Some(plugin) if plugin.registration_id == expected_registration_id => {
            guard.remove(plugin_kind);
            log::info!(
                target: "nemo_relay.plugin",
                event = "plugin_deregistered",
                plugin_kind = plugin_kind,
                registration_id = expected_registration_id;
                "Plugin kind deregistered"
            );
            Ok(PluginDeregistrationOutcome::Removed)
        }
        Some(_) => Ok(PluginDeregistrationOutcome::Replaced),
        None => Ok(PluginDeregistrationOutcome::Missing),
    }
}

/// Lists registered plugin kinds in sorted order.
///
/// This returns the currently registered plugin kinds without inspecting the
/// active runtime configuration.
///
/// # Returns
/// A sorted [`Vec<String>`] of registered plugin kinds.
///
/// # Notes
/// Disabled or inactive components still appear here when their plugin kind is
/// registered. An empty list is returned when built-in registration fails.
pub fn list_plugin_kinds() -> Vec<String> {
    if ensure_builtin_plugins_registered().is_err() {
        return Vec::new();
    }
    let mut kinds = PLUGIN_HANDLERS
        .read()
        .map(|guard| guard.keys().cloned().collect::<Vec<_>>())
        .unwrap_or_default();
    kinds.sort();
    kinds
}

/// Looks up a registered plugin by kind.
///
/// # Parameters
/// - `plugin_kind`: Plugin kind to resolve.
///
/// # Returns
/// The registered plugin implementation for `plugin_kind`, or `None` when the
/// kind is unknown or built-in registration fails.
///
/// # Notes
/// The returned plugin is shared by [`Arc`], so callers receive a cheap clone.
pub fn lookup_plugin(plugin_kind: &str) -> Option<Arc<dyn Plugin>> {
    ensure_builtin_plugins_registered().ok()?;
    lookup_registered_plugin(plugin_kind)
}

fn lookup_registered_plugin(plugin_kind: &str) -> Option<Arc<dyn Plugin>> {
    PLUGIN_HANDLERS.read().ok().and_then(|guard| {
        guard
            .get(plugin_kind)
            .map(|registered| Arc::clone(&registered.plugin))
    })
}

/// Validates a plugin configuration document.
///
/// This is a pure validation pass. It does not mutate the active runtime
/// configuration.
///
/// # Parameters
/// - `config`: Plugin configuration to validate.
///
/// # Returns
/// A [`ConfigReport`] describing warnings and errors discovered during
/// validation.
///
/// # Notes
/// Validation checks host policy, plugin multiplicity rules, unknown component
/// kinds, and plugin-provided validation hooks.
pub fn validate_plugin_config(config: &PluginConfig) -> ConfigReport {
    let mut report = ConfigReport::default();
    if let Err(error) = ensure_builtin_plugins_registered() {
        report.diagnostics.push(ConfigDiagnostic {
            level: DiagnosticLevel::Error,
            code: "plugin.builtin_registration_failed".to_string(),
            component: None,
            field: None,
            message: format!("built-in plugin registration failed: {error}"),
        });
        return report;
    }

    if config.version != 1 {
        push_policy_diag(
            &mut report.diagnostics,
            config.policy.unsupported_value,
            "plugin.unsupported_config_version",
            None,
            Some("version".to_string()),
            format!("plugin config version {} is unsupported", config.version),
        );
    }

    validate_plugin_multiplicity(&mut report, config);

    for component in &config.components {
        let Some(plugin) = lookup_registered_plugin(&component.kind) else {
            push_policy_diag(
                &mut report.diagnostics,
                config.policy.unknown_component,
                "plugin.unknown_component",
                Some(component.kind.clone()),
                None,
                format!("plugin component kind '{}' is unsupported", component.kind),
            );
            continue;
        };
        report
            .diagnostics
            .extend(plugin.validate_with_policy(&component.config, &config.policy));
    }

    report
}

/// Layers `right` (higher precedence) onto `left` in place.
///
/// Objects merge recursively and arrays/scalars are replaced by `right`, except:
///
/// - the top-level `components` array pairs entries by `kind` in order of appearance so
///   multi-instance kinds are not collapsed;
/// - lists inside a component's `config` concatenate with higher-precedence entries first.
///
/// Internal helper shared by plugin initialization and `plugins.toml` discovery.
fn layer_config(left: &mut Json, right: Json) {
    match (left, right) {
        (Json::Object(left), Json::Object(right)) => {
            for (key, value) in right {
                match (key.as_str(), left.get_mut(&key)) {
                    ("components", Some(existing)) => merge_plugin_components(existing, value),
                    (_, Some(existing)) => merge_json_value(existing, value),
                    (_, _) => {
                        left.insert(key, value);
                    }
                }
            }
        }
        (left, right) => *left = right,
    }
}

/// Merges `right` components into `left` by `kind`, pairing repeated kinds positionally.
fn merge_plugin_components(left: &mut Json, right: Json) {
    let Json::Array(left_components) = left else {
        *left = right;
        return;
    };
    let Json::Array(right_components) = right else {
        *left = right;
        return;
    };
    let base_component_count = left_components.len();
    let mut consumed: HashMap<String, usize> = HashMap::new();
    for component in right_components {
        let Some(kind) = component_kind(&component).map(str::to_owned) else {
            left_components.push(component);
            continue;
        };
        let nth = consumed.entry(kind.clone()).or_insert(0);
        let slot = nth_component_by_kind(&left_components[..base_component_count], &kind, *nth);
        *nth += 1;
        match slot {
            Some(index) => merge_plugin_component(&mut left_components[index], component),
            None => left_components.push(component),
        }
    }
}

/// Merges one higher-precedence component into its lower-precedence match.
///
/// Direct list fields in a component's `config` object concatenate with
/// higher-precedence entries first. Declared observability collections do the
/// same, except that an explicit empty log or metric endpoint list clears the
/// lower-precedence list so it remains distinguishable from an omitted list.
/// Deeper implementation-specific lists retain replacement semantics.
fn merge_plugin_component(existing: &mut Json, higher_priority: Json) {
    let is_observability = component_kind(&higher_priority).or_else(|| component_kind(existing))
        == Some("observability");
    match (existing, higher_priority) {
        (Json::Object(existing), Json::Object(higher_priority)) => {
            for (key, value) in higher_priority {
                match (key.as_str(), existing.get_mut(&key)) {
                    ("config", Some(existing_config)) => {
                        merge_plugin_config_value(
                            existing_config,
                            value,
                            &mut Vec::new(),
                            is_observability,
                        );
                    }
                    (_, Some(existing_value)) => merge_json_value(existing_value, value),
                    (_, None) => {
                        existing.insert(key, value);
                    }
                }
            }
        }
        (existing, higher_priority) => *existing = higher_priority,
    }
}

/// Recursively merges plugin component config with scoped list concatenation.
fn merge_plugin_config_value(
    lower_priority: &mut Json,
    higher_priority: Json,
    path: &mut Vec<String>,
    is_observability: bool,
) {
    match (lower_priority, higher_priority) {
        (Json::Object(lower_priority), Json::Object(higher_priority)) => {
            for (key, value) in higher_priority {
                path.push(key.clone());
                match lower_priority.get_mut(&key) {
                    Some(existing) => {
                        merge_plugin_config_value(existing, value, path, is_observability)
                    }
                    None => {
                        lower_priority.insert(key, value);
                    }
                }
                path.pop();
            }
        }
        (Json::Array(lower_priority), Json::Array(mut higher_priority))
            if plugin_config_list_concatenates(path, is_observability) =>
        {
            if higher_priority.is_empty()
                && plugin_config_empty_list_replaces(path, is_observability)
            {
                lower_priority.clear();
            } else {
                higher_priority.append(lower_priority);
                *lower_priority = higher_priority;
            }
        }
        (lower_priority, higher_priority) => *lower_priority = higher_priority,
    }
}

fn plugin_config_list_concatenates(path: &[String], is_observability: bool) -> bool {
    path.len() == 1
        || (is_observability
            && (matches!(
                path,
                [section, field]
                    if (section == "atof" && field == "sinks")
                        || (section == "opentelemetry" && matches!(field.as_str(), "traces" | "endpoints"))
                        || (section == "atif" && field == "storage")
            ) || matches!(
                path,
                [section, signal, field]
                    if section == "opentelemetry"
                        && matches!(signal.as_str(), "logs" | "metrics")
                        && field == "endpoints"
            )))
}

fn plugin_config_empty_list_replaces(path: &[String], is_observability: bool) -> bool {
    is_observability
        && matches!(
            path,
            [section, signal, field]
                if section == "opentelemetry"
                    && matches!(signal.as_str(), "logs" | "metrics")
                    && field == "endpoints"
        )
}

/// Recursively merges `right` into a `left` JSON object; arrays and scalars are replaced.
fn merge_json_value(left: &mut Json, right: Json) {
    match (left, right) {
        (Json::Object(left), Json::Object(right)) => {
            for (key, value) in right {
                match left.get_mut(&key) {
                    Some(existing) => merge_json_value(existing, value),
                    None => {
                        left.insert(key, value);
                    }
                }
            }
        }
        (left, right) => *left = right,
    }
}

fn component_kind(component: &Json) -> Option<&str> {
    component.get("kind").and_then(Json::as_str)
}

fn nth_component_by_kind(components: &[Json], kind: &str, nth: usize) -> Option<usize> {
    components
        .iter()
        .enumerate()
        .filter(|(_index, component)| component_kind(component) == Some(kind))
        .nth(nth)
        .map(|(index, _component)| index)
}

/// Returns the JSON Schema for the canonical plugin configuration document.
#[cfg(feature = "schema")]
pub fn plugin_config_schema() -> Json {
    serde_json::to_value(schemars::schema_for!(PluginConfig))
        .expect("plugin config schema should serialize")
}

/// Configures the active global plugin components.
///
/// Initialization validates the supplied config, replaces the active
/// configuration, and rolls back partial registration on failure. If a
/// previous configuration was active, the host attempts to restore it when the
/// new activation fails.
///
/// # Parameters
/// - `config`: Plugin configuration to validate and activate.
///
/// # Returns
/// A plugin [`Result`] containing the successful [`ConfigReport`].
///
/// # Errors
/// Returns an error when validation fails, when plugin registration fails, or
/// when the previous configuration cannot be restored after a failed replace.
///
/// # Notes
/// Initialization is replace-with-rollback: the previous active configuration
/// is removed before the new configuration is activated.
#[doc(hidden)]
pub async fn initialize_plugins_exact(config: PluginConfig) -> Result<ConfigReport> {
    initialize_plugins_with_diagnostics(config, Vec::new()).await
}

async fn initialize_plugins_with_diagnostics(
    config: PluginConfig,
    diagnostics: Vec<ConfigDiagnostic>,
) -> Result<ConfigReport> {
    run_owned_plugin_mutation("plugin initialization", move || async move {
        let lease = LegacyPluginMutationLease::acquire()?;
        let rollback_failures = Arc::new(Mutex::new(Vec::new()));
        let initialization = tokio::spawn(initialize_plugins_exact_inner(
            config,
            Some(Arc::clone(&rollback_failures)),
            diagnostics,
        ))
        .await
        .map_err(|error| {
            PluginError::Internal(format!("plugin initialization task failed: {error}"))
        });
        let result = initialization.and_then(|result| result);
        let failures = rollback_failures
            .lock()
            .map(|failures| failures.clone())
            .unwrap_or_else(|lock_error| {
                vec![format!("rollback failure lock poisoned: {lock_error}")]
            });
        match result {
            Err(error) if !failures.is_empty() => {
                std::mem::forget(lease);
                Err(PluginError::RegistrationFailed(format!(
                    concat!(
                        "{}; initialization rollback was incomplete: {}; plugin ",
                        "configuration mutations are disabled for this process because callbacks ",
                        "may remain registered"
                    ),
                    error,
                    failures.join("; ")
                )))
            }
            result => result,
        }
    })
    .await
}

pub(crate) async fn run_owned_plugin_mutation<T, F, Fut>(
    operation_name: &'static str,
    operation: F,
) -> Result<T>
where
    T: Send + 'static,
    F: FnOnce() -> Fut + Send + 'static,
    Fut: Future<Output = Result<T>> + Send + 'static,
{
    if IN_PLUGIN_MUTATION_EXECUTOR.get() {
        return operation().await;
    }

    let (result_tx, result_rx) = tokio::sync::oneshot::channel();
    plugin_mutation_executor()?
        .send(Box::pin(async move {
            let result = tokio::spawn(operation())
                .await
                .map_err(|error| {
                    PluginError::Internal(format!("{operation_name} task failed: {error}"))
                })
                .and_then(|result| result);
            let _ = result_tx.send(result);
        }))
        .map_err(|_| {
            PluginError::Internal(format!(
                "failed to queue {operation_name}: executor stopped"
            ))
        })?;
    result_rx.await.map_err(|_| {
        PluginError::Internal(format!(
            "{operation_name} task stopped before returning a result"
        ))
    })?
}

fn plugin_mutation_executor() -> Result<&'static PluginMutationSender> {
    if let Some(sender) = PLUGIN_MUTATION_EXECUTOR.get() {
        return Ok(sender);
    }

    let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel::<PluginMutationJob>();
    let (startup_tx, startup_rx) = std::sync::mpsc::sync_channel(1);
    std::thread::Builder::new()
        .name("nemo-relay-plugin-host".into())
        .spawn(move || {
            let runtime = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(runtime) => runtime,
                Err(error) => {
                    let _ = startup_tx.send(Err(error.to_string()));
                    return;
                }
            };
            let _ = startup_tx.send(Ok(()));
            IN_PLUGIN_MUTATION_EXECUTOR.set(true);
            runtime.block_on(async move {
                while let Some(job) = receiver.recv().await {
                    job.await;
                }
            });
        })
        .map_err(|error| {
            PluginError::Internal(format!("failed to start plugin host executor: {error}"))
        })?;
    startup_rx
        .recv()
        .map_err(|error| {
            PluginError::Internal(format!(
                "plugin host executor stopped during startup: {error}"
            ))
        })?
        .map_err(|error| {
            PluginError::Internal(format!("failed to start plugin host runtime: {error}"))
        })?;

    Ok(PLUGIN_MUTATION_EXECUTOR.get_or_init(|| sender))
}

pub(crate) async fn initialize_plugins_exact_for_host(
    config: PluginConfig,
    owner_id: u64,
    rollback_failures: Arc<Mutex<Vec<String>>>,
    diagnostics: Vec<ConfigDiagnostic>,
) -> Result<ConfigReport> {
    verify_plugin_host_owner(owner_id)?;
    initialize_plugins_exact_inner(config, Some(rollback_failures), diagnostics).await
}

async fn initialize_plugins_exact_inner(
    config: PluginConfig,
    rollback_failures: Option<Arc<Mutex<Vec<String>>>>,
    diagnostics: Vec<ConfigDiagnostic>,
) -> Result<ConfigReport> {
    let enabled_component_count = config
        .components
        .iter()
        .filter(|component| component.enabled)
        .count();
    log::info!(
        target: "nemo_relay.plugin",
        event = "plugin_configuration_activation_started",
        component_count = enabled_component_count;
        "Plugin configuration activation started"
    );
    let mut report = ConfigReport {
        diagnostics,
        ..ConfigReport::default()
    };
    report
        .diagnostics
        .extend(validate_plugin_config(&config).diagnostics);
    if report.has_errors() {
        return Err(PluginError::InvalidConfig(join_error_messages(&report)));
    }

    let previous = {
        let mut guard = ACTIVE_PLUGIN_CONFIGURATION.lock().map_err(|err| {
            PluginError::Internal(format!("active plugin configuration lock poisoned: {err}"))
        })?;
        guard.take()
    };

    match previous {
        Some(previous_state) => {
            replace_plugin_configuration(
                config,
                report,
                previous_state,
                rollback_failures,
                enabled_component_count,
            )
            .await
        }
        None => {
            activate_initial_plugin_configuration(
                config,
                report,
                rollback_failures,
                enabled_component_count,
            )
            .await
        }
    }
}

async fn activate_initial_plugin_configuration(
    config: PluginConfig,
    report: ConfigReport,
    rollback_failures: Option<Arc<Mutex<Vec<String>>>>,
    enabled_component_count: usize,
) -> Result<ConfigReport> {
    let registrations =
        initialize_plugin_components_catching_panics(config.clone(), rollback_failures).await?;
    store_active_plugin_configuration(config, report.clone(), registrations)?;
    log::info!(
        target: "nemo_relay.plugin",
        event = "plugin_configuration_activated",
        component_count = enabled_component_count;
        "Plugin configuration activated"
    );
    Ok(report)
}

async fn replace_plugin_configuration(
    config: PluginConfig,
    report: ConfigReport,
    mut previous_state: ActivePluginConfiguration,
    rollback_failures: Option<Arc<Mutex<Vec<String>>>>,
    enabled_component_count: usize,
) -> Result<ConfigReport> {
    install_previous_configuration_for_teardown(&previous_state)?;
    let teardown = rollback_registrations_checked(&mut previous_state.registrations);
    let teardown_report = take_active_runtime_diagnostics_report()?;
    if !teardown.errors.is_empty() {
        record_failed_teardown(&teardown, teardown_report, rollback_failures.as_ref());
        return Err(PluginError::RegistrationFailed(format!(
            "previous plugin configuration could not be cleared: {}",
            teardown.errors.join("; ")
        )));
    }
    activate_replacement_or_restore(
        config,
        report,
        previous_state,
        rollback_failures,
        enabled_component_count,
    )
    .await
}

fn install_previous_configuration_for_teardown(
    previous_state: &ActivePluginConfiguration,
) -> Result<()> {
    let mut guard = ACTIVE_PLUGIN_CONFIGURATION.lock().map_err(|err| {
        PluginError::Internal(format!("active plugin configuration lock poisoned: {err}"))
    })?;
    *guard = Some(ActivePluginConfiguration {
        config: previous_state.config.clone(),
        report: previous_state.report.clone(),
        runtime_diagnostics: previous_state.runtime_diagnostics.clone(),
        registrations: Vec::new(),
    });
    Ok(())
}

fn take_active_runtime_diagnostics_report() -> Result<Option<ConfigReport>> {
    Ok(ACTIVE_PLUGIN_CONFIGURATION
        .lock()
        .map_err(|err| {
            PluginError::Internal(format!("active plugin configuration lock poisoned: {err}"))
        })?
        .take()
        .map(|state| state.report))
}

fn record_failed_teardown(
    teardown: &PluginRollbackOutcome,
    teardown_report: Option<ConfigReport>,
    rollback_failures: Option<&Arc<Mutex<Vec<String>>>>,
) {
    if let Some(report) = teardown_report.filter(|report| !report.runtime_diagnostics.is_empty())
        && let Ok(mut guard) = LAST_FAILED_RUNTIME_DIAGNOSTICS_REPORT.lock()
    {
        *guard = Some(report);
    }
    if !teardown.callbacks_cleared {
        record_rollback_failures(rollback_failures, teardown.errors.clone());
    }
}

async fn activate_replacement_or_restore(
    config: PluginConfig,
    report: ConfigReport,
    previous_state: ActivePluginConfiguration,
    rollback_failures: Option<Arc<Mutex<Vec<String>>>>,
    enabled_component_count: usize,
) -> Result<ConfigReport> {
    match initialize_plugin_components_catching_panics(config.clone(), rollback_failures.clone())
        .await
    {
        Ok(registrations) => {
            store_active_plugin_configuration(config, report.clone(), registrations)?;
            log::info!(
                target: "nemo_relay.plugin",
                event = "plugin_configuration_replaced",
                component_count = enabled_component_count;
                "Plugin configuration replaced"
            );
            Ok(report)
        }
        Err(err) => {
            restore_previous_plugin_configuration(previous_state, rollback_failures, err).await
        }
    }
}

async fn restore_previous_plugin_configuration(
    previous_state: ActivePluginConfiguration,
    rollback_failures: Option<Arc<Mutex<Vec<String>>>>,
    err: PluginError,
) -> Result<ConfigReport> {
    match initialize_plugin_components_catching_panics(
        previous_state.config.clone(),
        rollback_failures,
    )
    .await
    {
        Ok(registrations) => {
            store_active_plugin_configuration_with_runtime_diagnostics(
                previous_state.config,
                previous_state.report,
                previous_state.runtime_diagnostics,
                registrations,
            )?;
            log::warn!(
                target: "nemo_relay.plugin",
                event = "plugin_configuration_restored",
                recovery = "previous_configuration";
                "Plugin activation failed; previous configuration restored"
            );
            Err(err)
        }
        Err(restore_err) => {
            log::error!(
                target: "nemo_relay.plugin",
                event = "plugin_rollback_failed",
                recovery = "previous_configuration";
                "Plugin activation failed and the previous configuration could not be restored"
            );
            Err(PluginError::RegistrationFailed(format!(
                "{err}; previous plugin configuration could not be restored: {restore_err}"
            )))
        }
    }
}

async fn initialize_plugin_components_catching_panics(
    config: PluginConfig,
    rollback_failures: Option<Arc<Mutex<Vec<String>>>>,
) -> Result<Vec<PluginRegistration>> {
    tokio::spawn(async move { initialize_plugin_components(&config, rollback_failures).await })
        .await
        .map_err(|error| {
            PluginError::Internal(format!(
                "plugin component initialization task failed: {error}"
            ))
        })?
}

/// Validates and activates `config` layered on top of the discovered
/// `plugins.toml` configuration, so a direct integration sees the same file
/// layering as the gateway. Each file's schema version is validated before
/// layering. Declaring a component in `config` applies its `enabled` value,
/// while default policy values inherit from discovered files and component
/// `config` bodies merge field-by-field. The resolved configuration and
/// diagnostics are passed to the shared `initialize_plugins_with_diagnostics`
/// helper. Call [`initialize_plugins_exact`] directly when `config` is already
/// fully resolved and every value must be applied exactly.
pub async fn initialize_plugins(config: PluginConfig) -> Result<ConfigReport> {
    let resolved = resolve_plugin_config(config)?;
    initialize_plugins_with_diagnostics(resolved.config, resolved.diagnostics).await
}

/// Layers `config` over the default discovered `plugins.toml` files.
///
/// This is crate-visible so owned dynamic-plugin activation can use the same
/// one-time configuration resolution as regular harness-native initialization.
pub(crate) fn resolve_plugin_config(config: PluginConfig) -> Result<ResolvedPluginConfig> {
    let discovered = resolve_default_file_plugin_config()?;
    resolve_programmatic_plugin_config(discovered, config)
}

fn resolve_programmatic_plugin_config(
    discovered: DiscoveredPluginConfig,
    config: PluginConfig,
) -> Result<ResolvedPluginConfig> {
    let mut diagnostics = inherited_plugin_config_diagnostics(&discovered.sources);
    diagnostics.extend(programmatic_enable_override_diagnostics(
        &discovered.value,
        &discovered.enabled_sources,
        &config,
    ));
    let mut base = discovered.value;
    layer_config(&mut base, plugin_config_overlay_value(&config)?);
    Ok(ResolvedPluginConfig {
        config: serde_json::from_value(base)?,
        diagnostics,
    })
}

pub(crate) struct ResolvedPluginConfig {
    pub(crate) config: PluginConfig,
    pub(crate) diagnostics: Vec<ConfigDiagnostic>,
}

/// Serializes a typed configuration as a discovery overlay.
///
/// A [`PluginConfig`] cannot record whether a default-valued field was supplied
/// explicitly or filled by serde. Treating those defaults as overlay values
/// would mask discovered file settings on every library initialization. A
/// declared component is an activation intent, so its `enabled` value remains
/// in the overlay. Exact callers bypass discovery through
/// [`initialize_plugins_exact`].
fn plugin_config_overlay_value(config: &PluginConfig) -> Result<Json> {
    let mut overlay = serde_json::to_value(config)?;
    let Json::Object(root) = &mut overlay else {
        return Ok(overlay);
    };

    if config.version == default_plugin_config_version() {
        root.remove("version");
    }

    remove_default_policy_overlay(root, &config.policy);

    Ok(overlay)
}

fn remove_default_policy_overlay(root: &mut Map<String, Json>, config: &ConfigPolicy) {
    let Some(Json::Object(policy)) = root.get_mut("policy") else {
        return;
    };
    let defaults = ConfigPolicy::default();
    for (field, is_default) in [
        (
            "unknown_component",
            config.unknown_component == defaults.unknown_component,
        ),
        (
            "unknown_field",
            config.unknown_field == defaults.unknown_field,
        ),
        (
            "unsupported_value",
            config.unsupported_value == defaults.unsupported_value,
        ),
    ] {
        if is_default {
            policy.remove(field);
        }
    }
    if policy.is_empty() {
        root.remove("policy");
    }
}

/// Resolves the default `plugins.toml` layering into one JSON document, or an
/// empty object when no plugin file exists.
fn resolve_default_file_plugin_config() -> Result<DiscoveredPluginConfig> {
    let paths = default_plugin_config_paths(user_config_dir());
    let documents = read_plugin_config_files(paths)?;
    resolve_discovered_plugin_config(documents)
}

fn resolve_discovered_plugin_config(
    documents: Vec<(PathBuf, Json)>,
) -> Result<DiscoveredPluginConfig> {
    let enabled_sources = component_enabled_sources(&documents);
    let (value, sources) = merge_plugin_config_documents(documents)?
        .unwrap_or_else(|| (Json::Object(Map::new()), Vec::new()));
    Ok(DiscoveredPluginConfig {
        value,
        enabled_sources,
        sources,
    })
}

struct DiscoveredPluginConfig {
    value: Json,
    enabled_sources: HashMap<String, ComponentEnabledSource>,
    sources: Vec<PathBuf>,
}

struct ComponentEnabledSource {
    enabled: bool,
    path: PathBuf,
}

use std::path::{Path, PathBuf};

/// Reads, parses, and merges the `plugins.toml` files at `paths` (lowest
/// precedence first) into one JSON document with its source paths, or `None`
/// when none exist. Internal: `pub` only for cross-crate reuse by the gateway.
#[doc(hidden)]
pub fn load_plugin_config_files<I>(paths: I) -> Result<Option<(Json, Vec<PathBuf>)>>
where
    I: IntoIterator<Item = PathBuf>,
{
    merge_plugin_config_documents(read_plugin_config_files(paths)?)
}

fn read_plugin_config_files<I>(paths: I) -> Result<Vec<(PathBuf, Json)>>
where
    I: IntoIterator<Item = PathBuf>,
{
    let mut documents = Vec::new();
    for path in deduplicate_plugin_config_paths(paths) {
        if !path.exists() {
            continue;
        }
        let raw = std::fs::read_to_string(&path).map_err(|err| {
            PluginError::InvalidConfig(format!("failed to read {}: {err}", path.display()))
        })?;
        let parsed = raw.parse::<toml::Table>().map_err(|err| {
            PluginError::InvalidConfig(format!("invalid plugin TOML in {}: {err}", path.display()))
        })?;
        documents.push((path, serde_json::to_value(parsed)?));
    }
    Ok(documents)
}

fn component_enabled_sources(
    documents: &[(PathBuf, Json)],
) -> HashMap<String, ComponentEnabledSource> {
    let mut sources = HashMap::new();
    for (path, document) in documents {
        let Some(components) = document.get("components").and_then(Json::as_array) else {
            continue;
        };
        for component in components {
            let Some(kind) = component_kind(component) else {
                continue;
            };
            if let Some(enabled) = component.get("enabled").and_then(Json::as_bool) {
                sources.insert(
                    kind.to_string(),
                    ComponentEnabledSource {
                        enabled,
                        path: path.clone(),
                    },
                );
            }
        }
    }
    sources
}

fn inherited_plugin_config_diagnostics(sources: &[PathBuf]) -> Vec<ConfigDiagnostic> {
    sources
        .iter()
        .map(|source| {
            let source = source.display().to_string();
            log::warn!(
                target: "nemo_relay.plugin",
                event = "plugin_configuration_inherited",
                config_path = source.as_str();
                "Inherited plugin configuration from discovered file"
            );
            ConfigDiagnostic {
                level: DiagnosticLevel::Warning,
                code: "plugin.configuration_inherited".to_string(),
                component: None,
                field: None,
                message: format!("inherited plugin configuration from discovered file: {source}"),
            }
        })
        .collect()
}

fn programmatic_enable_override_diagnostics(
    discovered: &Json,
    enabled_sources: &HashMap<String, ComponentEnabledSource>,
    programmatic: &PluginConfig,
) -> Vec<ConfigDiagnostic> {
    let Some(discovered_components) = discovered.get("components").and_then(Json::as_array) else {
        return Vec::new();
    };
    let mut consumed = HashMap::new();
    let mut diagnostics = Vec::new();
    for component in &programmatic.components {
        let nth = consumed.entry(component.kind.as_str()).or_insert(0usize);
        let discovered_component =
            nth_component_by_kind(discovered_components, &component.kind, *nth)
                .and_then(|index| discovered_components.get(index));
        *nth += 1;
        let discovered_enabled = discovered_component
            .and_then(|component| component.get("enabled"))
            .and_then(Json::as_bool);
        let file_disabled = discovered_enabled == Some(false)
            || (discovered_enabled.is_none()
                && enabled_sources
                    .get(&component.kind)
                    .is_some_and(|source| !source.enabled));
        if !component.enabled || !file_disabled {
            continue;
        }

        let source = enabled_sources
            .get(&component.kind)
            .map(|source| format!(" from {}", source.path.display()))
            .unwrap_or_default();
        diagnostics.push(ConfigDiagnostic {
            level: DiagnosticLevel::Warning,
            code: "plugin.component_reenabled".to_string(),
            component: Some(component.kind.clone()),
            field: Some("enabled".to_string()),
            message: format!(
                "programmatic configuration enabled plugin component '{}' and overrode enabled = false{source}",
                component.kind
            ),
        });
    }
    diagnostics
}

/// Removes physical duplicates while preserving the highest-precedence path.
/// Internal: `pub` only for cross-crate reuse by the gateway.
#[doc(hidden)]
pub fn deduplicate_plugin_config_paths<I>(paths: I) -> Vec<PathBuf>
where
    I: IntoIterator<Item = PathBuf>,
{
    let paths = paths.into_iter().collect::<Vec<_>>();
    let mut seen = HashSet::new();
    let mut unique = Vec::with_capacity(paths.len());
    for path in paths.into_iter().rev() {
        let identity = path.canonicalize().unwrap_or_else(|_| path.clone());
        if seen.insert(identity) {
            unique.push(path);
        }
    }
    unique.reverse();
    unique
}

/// Merges pre-parsed `plugins.toml` JSON documents (lowest precedence first) using the canonical
/// plugin-config layering rules. Internal: `pub` only so the CLI can preprocess dynamic-plugin
/// refs while still sharing one merge semantics implementation with core.
#[doc(hidden)]
pub fn merge_plugin_config_documents<I>(documents: I) -> Result<Option<(Json, Vec<PathBuf>)>>
where
    I: IntoIterator<Item = (PathBuf, Json)>,
{
    let mut merged = Json::Object(Map::new());
    let mut sources = Vec::new();
    for (path, mut document) in documents {
        validate_plugin_config_version(&path, &document)?;
        validate_unique_component_kinds(&path, &document)?;

        filter_disabled_plugin_components(&mut document);
        layer_config(&mut merged, document);
        sources.push(path);
    }
    Ok((!sources.is_empty()).then_some((merged, sources)))
}

/// Removes disabled components from one discovered plugin document before layering.
fn filter_disabled_plugin_components(document: &mut Json) {
    let Some(components) = document.get_mut("components").and_then(Json::as_array_mut) else {
        return;
    };
    components.retain(|component| component.get("enabled").and_then(Json::as_bool) != Some(false));
}

/// Rejects a file with an unsupported top-level plugin config version before layering can
/// overwrite it with a higher-precedence source or typed default.
fn validate_plugin_config_version(path: &Path, document: &Json) -> Result<()> {
    let Some(raw_version) = document.get("version") else {
        return Ok(());
    };
    let version = serde_json::from_value::<u32>(raw_version.clone()).map_err(|error| {
        PluginError::InvalidConfig(format!(
            "invalid plugin config version in {}: {error}",
            path.display()
        ))
    })?;
    if version == default_plugin_config_version() {
        return Ok(());
    }
    Err(PluginError::InvalidConfig(format!(
        "plugin config version {version} in {} is unsupported; expected {}",
        path.display(),
        default_plugin_config_version()
    )))
}

/// Rejects a single file that declares the same component `kind` more than once.
fn validate_unique_component_kinds(path: &Path, document: &Json) -> Result<()> {
    let Some(components) = document.get("components").and_then(Json::as_array) else {
        return Ok(());
    };
    let mut seen = HashSet::new();
    let mut duplicates = Vec::new();
    for component in components {
        if let Some(kind) = component_kind(component)
            && !seen.insert(kind)
        {
            duplicates.push(kind.to_string());
        }
    }
    if duplicates.is_empty() {
        return Ok(());
    }
    duplicates.sort();
    duplicates.dedup();
    Err(PluginError::InvalidConfig(format!(
        "duplicate plugin component kind in {}: {}; declare each kind once per plugins.toml",
        path.display(),
        duplicates.join(", ")
    )))
}

/// Default `plugins.toml` search path (lowest precedence first): user, then
/// system — mirroring the gateway's discovery. `pub` only for cross-crate reuse
/// by the gateway.
#[doc(hidden)]
pub fn default_plugin_config_paths(user_dir: Option<PathBuf>) -> Vec<PathBuf> {
    if skip_implicit_plugin_config() {
        return Vec::new();
    }
    let mut paths = Vec::new();
    if let Some(dir) = user_dir {
        paths.push(dir.join("plugins.toml"));
    }
    paths.push(system_config_dir().join("plugins.toml"));
    paths
}

#[cfg(feature = "__skip-implicit-config")]
fn skip_implicit_plugin_config() -> bool {
    std::env::var("NEMO_RELAY_TEST_SKIP_IMPLICIT_CONFIG")
        .ok()
        .as_deref()
        == Some("1")
}

#[cfg(not(feature = "__skip-implicit-config"))]
fn skip_implicit_plugin_config() -> bool {
    false
}

/// Resolves the platform system configuration directory.
#[doc(hidden)]
pub fn system_config_dir() -> PathBuf {
    #[cfg(windows)]
    {
        std::env::var_os("ProgramData")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(r"C:\ProgramData"))
            .join("nemo-relay")
    }
    #[cfg(not(windows))]
    {
        PathBuf::from("/etc/nemo-relay")
    }
}

/// Resolves the nemo-relay user config directory from `XDG_CONFIG_HOME`, then
/// `HOME`/`USERPROFILE`. `pub` only for cross-crate reuse by the gateway.
#[doc(hidden)]
pub fn user_config_dir() -> Option<PathBuf> {
    if let Some(base) = std::env::var_os("XDG_CONFIG_HOME") {
        return Some(PathBuf::from(base).join("nemo-relay"));
    }
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(|home| PathBuf::from(home).join(".config/nemo-relay"))
}

/// Deregisters and clears all configured plugin components.
///
/// Registered plugin kinds remain available for future validation and
/// initialization.
///
/// # Returns
/// A plugin [`Result`] that is `Ok(())` when the active configuration
/// has been cleared.
///
/// # Errors
/// Returns an error when the active configuration lock is poisoned.
///
/// # Notes
/// Clearing active configuration does not remove plugin kinds from the global
/// registry.
pub fn clear_plugin_configuration() -> Result<()> {
    let lease = LegacyPluginMutationLease::acquire()?;
    let outcome = clear_plugin_configuration_inner();
    if !outcome.callbacks_cleared {
        // Deregistration callbacks are single-use. If one failed, the process
        // can no longer prove that replacing configuration is safe.
        std::mem::forget(lease);
        log::error!(
            target: "nemo_relay.plugin",
            event = "plugin_cleanup_failed",
            callbacks_cleared = false;
            "Plugin configuration cleanup was incomplete"
        );
        return Err(PluginError::RegistrationFailed(format!(
            concat!(
                "{}; plugin configuration mutations are disabled for this process because ",
                "callbacks may remain registered"
            ),
            outcome
                .result
                .err()
                .map(|error| error.to_string())
                .unwrap_or_else(|| "plugin teardown was incomplete".into())
        )));
    }
    if outcome.result.is_ok() {
        log::info!(
            target: "nemo_relay.plugin",
            event = "plugin_configuration_cleared";
            "Plugin configuration cleared"
        );
    }
    outcome.result
}

pub(crate) fn clear_plugin_configuration_for_host(owner_id: u64) -> PluginHostClearOutcome {
    if let Err(error) = verify_plugin_host_owner(owner_id) {
        return PluginHostClearOutcome {
            result: Err(error),
            callbacks_cleared: false,
        };
    }
    clear_plugin_configuration_inner()
}

pub(crate) struct PluginHostClearOutcome {
    pub(crate) result: Result<()>,
    pub(crate) callbacks_cleared: bool,
}

fn clear_plugin_configuration_inner() -> PluginHostClearOutcome {
    let flush_error = crate::api::runtime::subscriber_dispatcher::flush_queued_subscribers()
        .err()
        .map(|error| error.to_string());
    let mut registrations = {
        let mut guard = match ACTIVE_PLUGIN_CONFIGURATION.lock() {
            Ok(guard) => guard,
            Err(err) => {
                return PluginHostClearOutcome {
                    result: Err(PluginError::Internal(format!(
                        "active plugin configuration lock poisoned: {err}"
                    ))),
                    callbacks_cleared: false,
                };
            }
        };
        guard
            .as_mut()
            .map(|state| std::mem::take(&mut state.registrations))
    };
    // Keep the report installed while callbacks run so runtime diagnostics
    // emitted by teardown work can be recorded against it.
    let deregistration = registrations
        .as_mut()
        .map(rollback_registrations_checked)
        .unwrap_or_default();
    let teardown_report = match ACTIVE_PLUGIN_CONFIGURATION.lock() {
        Ok(mut guard) => guard.take().map(|state| state.report),
        Err(err) => {
            return PluginHostClearOutcome {
                result: Err(PluginError::Internal(format!(
                    "active plugin configuration lock poisoned: {err}"
                ))),
                callbacks_cleared: false,
            };
        }
    };
    let deregistration_error = (!deregistration.errors.is_empty()).then(|| {
        PluginError::RegistrationFailed(format!(
            "plugin teardown failed: {}",
            deregistration.errors.join("; ")
        ))
    });
    let result = match (flush_error, deregistration_error) {
        (None, None) => Ok(()),
        (Some(flush), None) => Err(PluginError::Internal(flush)),
        (None, Some(deregister)) => Err(deregister),
        (Some(flush), Some(deregister)) => Err(PluginError::RegistrationFailed(format!(
            "{deregister}; subscriber flush also failed: {flush}"
        ))),
    };
    if result.is_ok() {
        if let Ok(mut guard) = LAST_FAILED_RUNTIME_DIAGNOSTICS_REPORT.lock() {
            *guard = None;
        }
    } else if let Some(report) =
        teardown_report.filter(|report| !report.runtime_diagnostics.is_empty())
        && let Ok(mut guard) = LAST_FAILED_RUNTIME_DIAGNOSTICS_REPORT.lock()
    {
        *guard = Some(report);
    }
    PluginHostClearOutcome {
        result,
        callbacks_cleared: deregistration.callbacks_cleared,
    }
}

pub(crate) fn plugin_configuration_is_active() -> Result<bool> {
    ACTIVE_PLUGIN_CONFIGURATION
        .lock()
        .map(|guard| guard.is_some())
        .map_err(|err| {
            PluginError::Internal(format!("active plugin configuration lock poisoned: {err}"))
        })
}

pub(crate) struct PluginHostLease {
    owner_id: u64,
}

impl PluginHostLease {
    pub(crate) fn owner_id(&self) -> u64 {
        self.owner_id
    }
}

impl Drop for PluginHostLease {
    fn drop(&mut self) {
        if let Ok(mut owner) = PLUGIN_MUTATION_OWNER.lock()
            && *owner == PluginMutationOwner::Host(self.owner_id)
        {
            *owner = PluginMutationOwner::Idle;
        }
    }
}

pub(crate) fn acquire_plugin_host_lease() -> Result<PluginHostLease> {
    let mut owner = PLUGIN_MUTATION_OWNER.lock().map_err(|err| {
        PluginError::Internal(format!("plugin mutation owner lock poisoned: {err}"))
    })?;
    if *owner != PluginMutationOwner::Idle {
        return Err(plugin_mutation_conflict(*owner));
    }
    if plugin_configuration_is_active()? {
        return Err(PluginError::Conflict(
            concat!(
                "a static plugin configuration is already active; to combine static and ",
                "dynamic plugins, provide the static components as the base configuration to ",
                "dynamic plugin activation before calling plugin initialization"
            )
            .into(),
        ));
    }
    let owner_id = NEXT_PLUGIN_HOST_OWNER_ID.fetch_add(1, Ordering::Relaxed);
    *owner = PluginMutationOwner::Host(owner_id);
    Ok(PluginHostLease { owner_id })
}

fn verify_plugin_host_owner(owner_id: u64) -> Result<()> {
    let owner = PLUGIN_MUTATION_OWNER.lock().map_err(|err| {
        PluginError::Internal(format!("plugin mutation owner lock poisoned: {err}"))
    })?;
    if *owner == PluginMutationOwner::Host(owner_id) {
        Ok(())
    } else {
        Err(PluginError::Conflict(
            "dynamic plugin host no longer owns plugin configuration".into(),
        ))
    }
}

struct LegacyPluginMutationLease;

impl LegacyPluginMutationLease {
    fn acquire() -> Result<Self> {
        let mut owner = PLUGIN_MUTATION_OWNER.lock().map_err(|err| {
            PluginError::Internal(format!("plugin mutation owner lock poisoned: {err}"))
        })?;
        if *owner != PluginMutationOwner::Idle {
            return Err(plugin_mutation_conflict(*owner));
        }
        *owner = PluginMutationOwner::Legacy;
        Ok(Self)
    }
}

impl Drop for LegacyPluginMutationLease {
    fn drop(&mut self) {
        if let Ok(mut owner) = PLUGIN_MUTATION_OWNER.lock()
            && *owner == PluginMutationOwner::Legacy
        {
            *owner = PluginMutationOwner::Idle;
        }
    }
}

fn plugin_mutation_conflict(owner: PluginMutationOwner) -> PluginError {
    let message = match owner {
        PluginMutationOwner::Idle => "plugin configuration is available",
        PluginMutationOwner::Legacy => "another plugin configuration mutation is in progress",
        PluginMutationOwner::Host(_) => {
            "plugin configuration is owned by an active dynamic plugin host"
        }
    };
    PluginError::Conflict(message.into())
}

/// Returns the active plugin report or a report retained after a failed teardown.
///
/// `None` indicates that no plugin configuration is active and no failed
/// teardown report is retained.
///
/// # Returns
/// The active [`ConfigReport`], or the report containing runtime diagnostics
/// from the last failed teardown.
///
/// # Notes
/// This is a snapshot of the last successful activation and does not re-run
/// validation.
pub fn active_plugin_report() -> Option<ConfigReport> {
    let active_report = ACTIVE_PLUGIN_CONFIGURATION
        .lock()
        .ok()
        .and_then(|guard| guard.as_ref().map(|state| state.report.clone()));
    active_report.or_else(|| {
        LAST_FAILED_RUNTIME_DIAGNOSTICS_REPORT
            .lock()
            .ok()
            .and_then(|guard| guard.clone())
    })
}

/// Record a bounded runtime diagnostic against the active plugin report.
pub fn record_active_plugin_runtime_diagnostic(diagnostic: RuntimeDiagnostic) {
    let Ok(mut guard) = ACTIVE_PLUGIN_CONFIGURATION.lock() else {
        return;
    };
    let Some(state) = guard.as_mut() else {
        return;
    };
    if let Some(existing) = state.runtime_diagnostics.get_mut(&diagnostic.code) {
        existing.message = diagnostic.message.clone();
        existing.count = existing.count.saturating_add(diagnostic.count);
    } else if state.runtime_diagnostics.len() < MAX_DYNAMIC_PLUGIN_RUNTIME_DIAGNOSTICS {
        state.runtime_diagnostics.insert(
            diagnostic.code.clone(),
            RuntimeDiagnosticsSnapshotEntry {
                code: diagnostic.code.clone(),
                message: diagnostic.message.clone(),
                count: diagnostic.count,
            },
        );
    }
    if let Some(existing) = state
        .report
        .runtime_diagnostics
        .iter_mut()
        .find(|existing| {
            existing.code == diagnostic.code
                && existing.component == diagnostic.component
                && existing.field == diagnostic.field
        })
    {
        existing.message = diagnostic.message;
        existing.session_id = diagnostic.session_id;
        existing.count = existing.count.saturating_add(diagnostic.count);
    } else {
        state.report.runtime_diagnostics.push(diagnostic);
    }
}

/// Return a bounded, active-only runtime-diagnostics snapshot for dynamic plugins.
pub(crate) fn active_runtime_diagnostics_snapshot() -> Vec<RuntimeDiagnosticsSnapshotEntry> {
    ACTIVE_PLUGIN_CONFIGURATION
        .lock()
        .ok()
        .and_then(|guard| {
            guard
                .as_ref()
                .map(|state| state.runtime_diagnostics.values().cloned().collect())
        })
        .unwrap_or_default()
}

/// Rolls back registrations in reverse order, ignoring rollback failures.
///
/// This is used internally during failed initialization and by
/// [`clear_plugin_configuration`].
pub fn rollback_registrations(registrations: &mut Vec<PluginRegistration>) {
    let _ = rollback_registrations_checked(registrations);
}

struct PluginRollbackOutcome {
    errors: Vec<String>,
    callbacks_cleared: bool,
}

impl Default for PluginRollbackOutcome {
    fn default() -> Self {
        Self {
            errors: Vec::new(),
            callbacks_cleared: true,
        }
    }
}

fn rollback_registrations_checked(
    registrations: &mut Vec<PluginRegistration>,
) -> PluginRollbackOutcome {
    let mut outcome = PluginRollbackOutcome::default();
    for registration in registrations.iter_mut().rev() {
        match catch_unwind(AssertUnwindSafe(|| (registration.deregister)())) {
            Ok(PluginRegistrationCleanupOutcome::Removed) => {}
            Ok(PluginRegistrationCleanupOutcome::RemovedWithError(error)) => {
                outcome.errors.push(format!(
                    "{} registration '{}' reported a delivery failure: {error}",
                    registration.kind, registration.name
                ));
            }
            Ok(PluginRegistrationCleanupOutcome::NotRemoved(error)) => {
                outcome.callbacks_cleared = false;
                outcome.errors.push(format!(
                    "{} registration '{}' could not be removed: {error}",
                    registration.kind, registration.name
                ));
            }
            Err(payload) => {
                outcome.callbacks_cleared = false;
                outcome.errors.push(format!(
                    "{} registration '{}' could not be removed: deregistration panicked: {}",
                    registration.kind,
                    registration.name,
                    panic_payload_message(payload)
                ));
            }
        }
    }
    registrations.clear();
    outcome
}

fn panic_payload_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|message| (*message).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic payload".into())
}

struct ActivePluginConfiguration {
    config: PluginConfig,
    report: ConfigReport,
    runtime_diagnostics: BTreeMap<String, RuntimeDiagnosticsSnapshotEntry>,
    registrations: Vec<PluginRegistration>,
}

async fn initialize_plugin_components(
    config: &PluginConfig,
    rollback_failures: Option<Arc<Mutex<Vec<String>>>>,
) -> Result<Vec<PluginRegistration>> {
    ensure_builtin_plugins_registered()?;
    let mut ordinals: HashMap<&str, usize> = HashMap::new();
    let mut registrations = PendingPluginRegistrations::new(rollback_failures.clone());

    for component in config
        .components
        .iter()
        .filter(|component| component.enabled)
    {
        let Some(plugin) = lookup_registered_plugin(&component.kind) else {
            return Err(PluginError::NotFound(format!(
                "plugin component '{}' is not registered",
                component.kind
            )));
        };

        let ordinal = ordinals
            .entry(component.kind.as_str())
            .and_modify(|value| *value += 1)
            .or_insert(1);
        let namespace = component_namespace(&component.kind, *ordinal);

        let mut pending =
            PendingPluginRegistrationContext::new(namespace, rollback_failures.clone());
        plugin
            .register(&component.config, &mut pending.context)
            .await?;
        registrations.extend(pending.take());
    }

    Ok(registrations.take())
}

struct PendingPluginRegistrations {
    registrations: Vec<PluginRegistration>,
    rollback_failures: Option<Arc<Mutex<Vec<String>>>>,
}

impl PendingPluginRegistrations {
    fn new(rollback_failures: Option<Arc<Mutex<Vec<String>>>>) -> Self {
        Self {
            registrations: Vec::new(),
            rollback_failures,
        }
    }

    fn extend(&mut self, registrations: Vec<PluginRegistration>) {
        self.registrations.extend(registrations);
    }

    fn take(&mut self) -> Vec<PluginRegistration> {
        std::mem::take(&mut self.registrations)
    }
}

impl Drop for PendingPluginRegistrations {
    fn drop(&mut self) {
        let outcome = rollback_registrations_checked(&mut self.registrations);
        if !outcome.callbacks_cleared {
            record_rollback_failures(self.rollback_failures.as_ref(), outcome.errors);
        }
    }
}

struct PendingPluginRegistrationContext {
    context: PluginRegistrationContext,
    rollback_failures: Option<Arc<Mutex<Vec<String>>>>,
}

impl PendingPluginRegistrationContext {
    fn new(namespace: String, rollback_failures: Option<Arc<Mutex<Vec<String>>>>) -> Self {
        Self {
            context: PluginRegistrationContext::with_plugin_component_namespace(namespace),
            rollback_failures,
        }
    }

    fn take(&mut self) -> Vec<PluginRegistration> {
        std::mem::take(&mut self.context.registrations)
    }
}

impl Drop for PendingPluginRegistrationContext {
    fn drop(&mut self) {
        let outcome = rollback_registrations_checked(&mut self.context.registrations);
        if !outcome.callbacks_cleared {
            record_rollback_failures(self.rollback_failures.as_ref(), outcome.errors);
        }
    }
}

fn record_rollback_failures(
    rollback_failures: Option<&Arc<Mutex<Vec<String>>>>,
    errors: Vec<String>,
) {
    if errors.is_empty() {
        return;
    }
    if let Some(rollback_failures) = rollback_failures
        && let Ok(mut recorded) = rollback_failures.lock()
    {
        recorded.extend(errors);
    }
}

fn store_active_plugin_configuration(
    config: PluginConfig,
    report: ConfigReport,
    registrations: Vec<PluginRegistration>,
) -> Result<()> {
    store_active_plugin_configuration_with_runtime_diagnostics(
        config,
        report,
        BTreeMap::new(),
        registrations,
    )
}

fn store_active_plugin_configuration_with_runtime_diagnostics(
    config: PluginConfig,
    report: ConfigReport,
    runtime_diagnostics: BTreeMap<String, RuntimeDiagnosticsSnapshotEntry>,
    registrations: Vec<PluginRegistration>,
) -> Result<()> {
    let mut guard = ACTIVE_PLUGIN_CONFIGURATION.lock().map_err(|err| {
        PluginError::Internal(format!("active plugin configuration lock poisoned: {err}"))
    })?;
    *guard = Some(ActivePluginConfiguration {
        config,
        report,
        runtime_diagnostics,
        registrations,
    });
    if let Ok(mut guard) = LAST_FAILED_RUNTIME_DIAGNOSTICS_REPORT.lock() {
        *guard = None;
    }
    Ok(())
}

fn plugin_component_totals(config: &PluginConfig) -> HashMap<&str, usize> {
    let mut totals = HashMap::new();
    for component in &config.components {
        *totals.entry(component.kind.as_str()).or_insert(0) += 1;
    }
    totals
}

fn component_namespace(kind: &str, ordinal: usize) -> String {
    assert!(ordinal > 0, "plugin component ordinals are one-based");
    format!(
        "nemo-relay-plugin.v1.{}:{ordinal}:",
        encode_plugin_component_field(kind)
    )
}

pub(crate) fn encode_plugin_component_field(value: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";

    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
            encoded.push(char::from(byte));
        } else {
            encoded.push('%');
            encoded.push(char::from(HEX[(byte >> 4) as usize]));
            encoded.push(char::from(HEX[(byte & 0x0f) as usize]));
        }
    }
    encoded
}

pub(crate) fn decode_plugin_component_effective_name(
    effective_name: &str,
) -> Option<(String, u32, String)> {
    const PREFIX: &str = "nemo-relay-plugin.v1.";

    let rest = effective_name.strip_prefix(PREFIX)?;
    let (encoded_kind, rest) = rest.split_once(':')?;
    let (ordinal_text, encoded_local_name) = rest.split_once(':')?;
    let ordinal = ordinal_text
        .parse::<u32>()
        .ok()
        .filter(|ordinal| *ordinal > 0)?;
    if ordinal.to_string() != ordinal_text {
        return None;
    }
    let plugin_kind = decode_plugin_component_field(encoded_kind)?;
    let local_name = decode_plugin_component_field(encoded_local_name)?;
    (!plugin_kind.is_empty() && !local_name.is_empty()).then_some((
        plugin_kind,
        ordinal,
        local_name,
    ))
}

fn decode_plugin_component_field(encoded: &str) -> Option<String> {
    let mut bytes = Vec::with_capacity(encoded.len());
    let encoded = encoded.as_bytes();
    let mut index = 0;

    while index < encoded.len() {
        let byte = encoded[index];
        if byte == b'%' {
            let high = *encoded.get(index + 1)?;
            let low = *encoded.get(index + 2)?;
            bytes.push((hex_value(high)? << 4) | hex_value(low)?);
            index += 3;
        } else if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
            bytes.push(byte);
            index += 1;
        } else {
            return None;
        }
    }

    let decoded = String::from_utf8(bytes).ok()?;
    (encode_plugin_component_field(&decoded) == std::str::from_utf8(encoded).ok()?)
        .then_some(decoded)
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

fn validate_plugin_multiplicity(report: &mut ConfigReport, config: &PluginConfig) {
    let totals = plugin_component_totals(config);
    let mut emitted = HashSet::new();

    for component in &config.components {
        let count = totals
            .get(component.kind.as_str())
            .copied()
            .unwrap_or_default();
        if count <= 1 || !emitted.insert(component.kind.clone()) {
            continue;
        }

        let allows_multiple = lookup_registered_plugin(&component.kind)
            .map(|plugin| plugin.allows_multiple_components())
            .unwrap_or(true);
        if !allows_multiple {
            report.diagnostics.push(ConfigDiagnostic {
                level: DiagnosticLevel::Error,
                code: "plugin.duplicate_component".to_string(),
                component: Some(component.kind.clone()),
                field: None,
                message: format!(
                    "plugin component kind '{}' may only appear once",
                    component.kind
                ),
            });
        }
    }
}

fn push_policy_diag(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    behavior: UnsupportedBehavior,
    code: &str,
    component: Option<String>,
    field: Option<String>,
    message: String,
) {
    let level = match behavior {
        UnsupportedBehavior::Ignore => return,
        UnsupportedBehavior::Warn => DiagnosticLevel::Warning,
        UnsupportedBehavior::Error => DiagnosticLevel::Error,
    };

    diagnostics.push(ConfigDiagnostic {
        level,
        code: code.to_string(),
        component,
        field,
        message,
    });
}

fn join_error_messages(report: &ConfigReport) -> String {
    report
        .diagnostics
        .iter()
        .filter(|diag| diag.level == DiagnosticLevel::Error)
        .map(|diag| diag.message.as_str())
        .collect::<Vec<_>>()
        .join("; ")
}

#[cfg(test)]
#[path = "../tests/unit/plugin_tests.rs"]
mod tests;
