// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generic plugin infrastructure for NeMo Relay runtimes.
//!
//! This module owns:
//! - config diagnostics and policy enums used by plugin systems
//! - a global plugin registry
//! - plugin registration contexts for middleware/subscriber installation
//! - rollback bookkeeping for registrations created during plugin setup

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, LazyLock, Mutex, OnceLock, RwLock};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as Json};
use thiserror::Error;

use crate::api::registry::{
    deregister_llm_conditional_execution_guardrail, deregister_llm_execution_intercept,
    deregister_llm_request_intercept, deregister_llm_sanitize_request_guardrail,
    deregister_llm_sanitize_response_guardrail, deregister_llm_stream_execution_intercept,
    deregister_tool_conditional_execution_guardrail, deregister_tool_execution_intercept,
    deregister_tool_request_intercept, deregister_tool_sanitize_request_guardrail,
    deregister_tool_sanitize_response_guardrail, register_llm_conditional_execution_guardrail,
    register_llm_execution_intercept, register_llm_request_intercept,
    register_llm_sanitize_request_guardrail, register_llm_sanitize_response_guardrail,
    register_llm_stream_execution_intercept, register_tool_conditional_execution_guardrail,
    register_tool_execution_intercept, register_tool_request_intercept,
    register_tool_sanitize_request_guardrail, register_tool_sanitize_response_guardrail,
};
use crate::api::runtime::{
    EventSubscriberFn, LlmConditionalFn, LlmExecutionFn, LlmRequestInterceptFn,
    LlmSanitizeRequestFn, LlmSanitizeResponseFn, LlmStreamExecutionFn, ToolConditionalFn,
    ToolExecutionFn, ToolInterceptFn, ToolSanitizeFn,
};
use crate::api::subscriber::{deregister_subscriber, register_subscriber};

pub mod dynamic;
pub use dynamic::*;

type PluginMap = HashMap<String, Arc<dyn Plugin>>;

static PLUGIN_HANDLERS: LazyLock<RwLock<PluginMap>> = LazyLock::new(|| RwLock::new(HashMap::new()));
static ACTIVE_PLUGIN_CONFIGURATION: LazyLock<Mutex<Option<ActivePluginConfiguration>>> =
    LazyLock::new(|| Mutex::new(None));
static BUILTIN_PLUGIN_REGISTRATION: OnceLock<Result<()>> = OnceLock::new();

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
}

impl ConfigReport {
    /// Returns `true` when the report contains at least one error diagnostic.
    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|diag| diag.level == DiagnosticLevel::Error)
    }
}

/// One validation or compatibility diagnostic.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ConfigDiagnostic {
    /// Severity level for the diagnostic.
    pub level: DiagnosticLevel,
    /// Stable diagnostic code suitable for machine checks.
    pub code: String,
    /// Optional component identifier associated with the diagnostic.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub component: Option<String>,
    /// Optional field path associated with the diagnostic.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub field: Option<String>,
    /// Human-readable diagnostic message.
    pub message: String,
}

/// Diagnostic severity.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum DiagnosticLevel {
    /// Non-fatal compatibility or validation issue.
    Warning,
    /// Fatal validation issue that blocks initialization.
    Error,
}

/// Policy for how unsupported plugin/runtime config is handled.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
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
    deregister: Box<dyn FnMut() -> Result<()> + Send>,
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
        deregister: Box<dyn FnMut() -> Result<()> + Send>,
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
    namespace: Option<String>,
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
            namespace: Some(namespace.into()),
        }
    }

    /// Returns the runtime-qualified name for a plugin-local registration.
    ///
    /// Plugin handlers should pass stable component-local names such as
    /// `"tool"` or `"subscriber"`. The host applies the namespace so users do
    /// not have to provide component instance ids.
    pub fn qualify_name(&self, name: &str) -> String {
        match &self.namespace {
            Some(namespace) => format!("{namespace}{name}"),
            None => name.to_string(),
        }
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
    let mut guard = PLUGIN_HANDLERS
        .write()
        .map_err(|err| PluginError::Internal(format!("plugin registry lock poisoned: {err}")))?;
    let plugin_kind = plugin.plugin_kind().to_string();
    if guard.contains_key(&plugin_kind) {
        return Err(PluginError::RegistrationFailed(format!(
            "plugin '{plugin_kind}' is already registered"
        )));
    }
    guard.insert(plugin_kind, plugin);
    Ok(())
}

/// Registers core-provided plugin kinds.
///
/// Built-in plugins are available to validation and initialization without a
/// binding or application-specific registration call.
pub fn ensure_builtin_plugins_registered() -> Result<()> {
    let register_builtins = || {
        crate::observability::plugin_component::register_observability_component()?;
        crate::plugins::nemo_guardrails::component::register_nemo_guardrails_component()?;
        crate::plugins::pricing::register_pricing_component()
    };
    match BUILTIN_PLUGIN_REGISTRATION.get_or_init(register_builtins) {
        Ok(()) => Ok(()),
        Err(err) => Err(clone_cached_plugin_error(err)),
    }
}

fn clone_cached_plugin_error(err: &PluginError) -> PluginError {
    match err {
        PluginError::InvalidConfig(message) => PluginError::InvalidConfig(message.clone()),
        PluginError::Conflict(message) => PluginError::Conflict(message.clone()),
        PluginError::NotFound(message) => PluginError::NotFound(message.clone()),
        PluginError::Serialization(err) => PluginError::Internal(err.to_string()),
        PluginError::Internal(message) => PluginError::Internal(message.clone()),
        PluginError::RegistrationFailed(message) => {
            PluginError::RegistrationFailed(message.clone())
        }
    }
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
    PLUGIN_HANDLERS
        .write()
        .ok()
        .and_then(|mut guard| guard.remove(plugin_kind))
        .is_some()
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
/// registered.
pub fn list_plugin_kinds() -> Vec<String> {
    let _ = ensure_builtin_plugins_registered();
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
/// kind is unknown.
///
/// # Notes
/// The returned plugin is shared by [`Arc`], so callers receive a cheap clone.
pub fn lookup_plugin(plugin_kind: &str) -> Option<Arc<dyn Plugin>> {
    let _ = ensure_builtin_plugins_registered();
    PLUGIN_HANDLERS
        .read()
        .ok()
        .and_then(|guard| guard.get(plugin_kind).cloned())
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
    let _ = ensure_builtin_plugins_registered();
    let mut report = ConfigReport::default();

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
        let Some(plugin) = lookup_plugin(&component.kind) else {
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
            .extend(plugin.validate(&component.config));
    }

    report
}

/// Layers `right` (higher precedence) onto `left` in place.
///
/// Objects merge recursively and arrays/scalars are replaced by `right`, except the
/// top-level `components` array, whose entries pair by `kind` in order of appearance so
/// multi-instance kinds are not collapsed. Internal helper shared by plugin
/// initialization and `plugins.toml` discovery.
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
    let mut base_slots: HashMap<String, Vec<usize>> = HashMap::new();
    for (index, component) in left_components.iter().enumerate() {
        if let Some(kind) = component_kind(component) {
            base_slots.entry(kind.to_string()).or_default().push(index);
        }
    }
    let mut consumed: HashMap<String, usize> = HashMap::new();
    for component in right_components {
        let Some(kind) = component_kind(&component).map(str::to_owned) else {
            left_components.push(component);
            continue;
        };
        let nth = consumed.entry(kind.clone()).or_insert(0);
        let slot = base_slots
            .get(&kind)
            .and_then(|slots| slots.get(*nth))
            .copied();
        *nth += 1;
        match slot {
            Some(index) if kind == "pricing" => {
                merge_pricing_component(&mut left_components[index], component)
            }
            Some(index) => merge_json_value(&mut left_components[index], component),
            None => left_components.push(component),
        }
    }
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

/// Like `merge_json_value`, but concatenates a `pricing` component's `config.sources`
/// (higher-precedence first) instead of replacing them, so lower-precedence fallback sources survive.
fn merge_pricing_component(existing: &mut Json, higher_priority: Json) {
    let lower_priority_sources = pricing_component_sources(existing).cloned();
    let higher_priority_sources = pricing_component_sources(&higher_priority).cloned();
    merge_json_value(existing, higher_priority);

    let Some(mut sources) = higher_priority_sources else {
        return;
    };
    if let Some(lower_priority_sources) = lower_priority_sources {
        sources.extend(lower_priority_sources);
    }
    set_pricing_component_sources(existing, sources);
}

fn pricing_component_sources(component: &Json) -> Option<&Vec<Json>> {
    component
        .get("config")
        .and_then(|config| config.get("sources"))
        .and_then(Json::as_array)
}

fn set_pricing_component_sources(component: &mut Json, sources: Vec<Json>) {
    if let Some(config) = component.get_mut("config").and_then(Json::as_object_mut) {
        config.insert("sources".into(), Json::Array(sources));
    }
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
    let report = validate_plugin_config(&config);
    if report.has_errors() {
        return Err(PluginError::InvalidConfig(join_error_messages(&report)));
    }

    let previous = {
        let mut guard = ACTIVE_PLUGIN_CONFIGURATION.lock().map_err(|err| {
            PluginError::Internal(format!("active plugin configuration lock poisoned: {err}"))
        })?;
        guard.take()
    };

    if let Some(mut previous_state) = previous {
        rollback_registrations(&mut previous_state.registrations);
        match initialize_plugin_components(&config).await {
            Ok(registrations) => {
                store_active_plugin_configuration(config, report.clone(), registrations)?;
                Ok(report)
            }
            Err(err) => match initialize_plugin_components(&previous_state.config).await {
                Ok(registrations) => {
                    let previous_report = validate_plugin_config(&previous_state.config);
                    store_active_plugin_configuration(
                        previous_state.config,
                        previous_report,
                        registrations,
                    )?;
                    Err(err)
                }
                Err(restore_err) => Err(PluginError::RegistrationFailed(format!(
                    "{err}; previous plugin configuration could not be restored: {restore_err}"
                ))),
            },
        }
    } else {
        let registrations = initialize_plugin_components(&config).await?;
        store_active_plugin_configuration(config, report.clone(), registrations)?;
        Ok(report)
    }
}

/// Validates and activates `config` layered on top of the discovered
/// `plugins.toml` configuration, so a direct integration sees the same file
/// layering as the gateway. `config` wins on conflicts; as a typed document its
/// default `version`/`policy`/`enabled` override the file, while `config` bodies
/// merge field-by-field. Delegates to [`initialize_plugins_exact`].
pub async fn initialize_plugins(config: PluginConfig) -> Result<ConfigReport> {
    let mut base = resolve_default_file_plugin_config()?;
    layer_config(&mut base, serde_json::to_value(config)?);
    let config: PluginConfig = serde_json::from_value(base)?;
    initialize_plugins_exact(config).await
}

/// Resolves the default `plugins.toml` layering into one JSON document, or an
/// empty object when no plugin file exists.
fn resolve_default_file_plugin_config() -> Result<Json> {
    let paths =
        default_plugin_config_paths(std::env::current_dir().ok().as_deref(), user_config_dir());
    Ok(load_plugin_config_files(paths)?
        .map(|(value, _sources)| value)
        .unwrap_or_else(|| Json::Object(Map::new())))
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
    let mut documents = Vec::new();
    for path in paths {
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
    merge_plugin_config_documents(documents)
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
    for (path, document) in documents {
        validate_unique_component_kinds(&path, &document)?;
        layer_config(&mut merged, document);
        sources.push(path);
    }
    Ok((!sources.is_empty()).then_some((merged, sources)))
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

/// Default `plugins.toml` search path (lowest precedence first): system, nearest
/// project file, then user file — mirroring the gateway's discovery. `pub` only
/// for cross-crate reuse by the gateway.
#[doc(hidden)]
pub fn default_plugin_config_paths(cwd: Option<&Path>, user_dir: Option<PathBuf>) -> Vec<PathBuf> {
    let mut paths = vec![PathBuf::from("/etc/nemo-relay/plugins.toml")];
    if let Some(cwd) = cwd
        && let Some(project) = nearest_project_plugin_config(cwd)
    {
        paths.push(project);
    }
    if let Some(dir) = user_dir {
        paths.push(dir.join("plugins.toml"));
    }
    paths
}

/// Walks upward from `start` for the nearest `.nemo-relay/plugins.toml`. `pub`
/// only for cross-crate reuse by the gateway.
#[doc(hidden)]
pub fn nearest_project_plugin_config(start: &Path) -> Option<PathBuf> {
    start
        .ancestors()
        .map(|ancestor| ancestor.join(".nemo-relay").join("plugins.toml"))
        .find(|path| path.exists())
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
    let flush_error = crate::api::runtime::flush_subscribers()
        .err()
        .map(|error| error.to_string());
    let previous = {
        let mut guard = ACTIVE_PLUGIN_CONFIGURATION.lock().map_err(|err| {
            PluginError::Internal(format!("active plugin configuration lock poisoned: {err}"))
        })?;
        guard.take()
    };
    if let Some(mut previous_state) = previous {
        rollback_registrations(&mut previous_state.registrations);
    }
    if let Some(message) = flush_error {
        return Err(PluginError::Internal(message));
    }
    Ok(())
}

/// Returns the last successfully configured plugin report.
///
/// `None` indicates that no plugin configuration is currently active.
///
/// # Returns
/// The last successful [`ConfigReport`], or `None` when no configuration is
/// active.
///
/// # Notes
/// This is a snapshot of the last successful activation and does not re-run
/// validation.
pub fn active_plugin_report() -> Option<ConfigReport> {
    ACTIVE_PLUGIN_CONFIGURATION
        .lock()
        .ok()
        .and_then(|guard| guard.as_ref().map(|state| state.report.clone()))
}

/// Rolls back registrations in reverse order, ignoring rollback failures.
///
/// This is used internally during failed initialization and by
/// [`clear_plugin_configuration`].
pub fn rollback_registrations(registrations: &mut Vec<PluginRegistration>) {
    for registration in registrations.iter_mut().rev() {
        let _ = (registration.deregister)();
    }
    registrations.clear();
}

struct ActivePluginConfiguration {
    config: PluginConfig,
    report: ConfigReport,
    registrations: Vec<PluginRegistration>,
}

async fn initialize_plugin_components(config: &PluginConfig) -> Result<Vec<PluginRegistration>> {
    ensure_builtin_plugins_registered()?;
    let totals = plugin_component_totals(config);
    let mut ordinals: HashMap<&str, usize> = HashMap::new();
    let mut registrations = vec![];

    for component in config
        .components
        .iter()
        .filter(|component| component.enabled)
    {
        let Some(plugin) = lookup_plugin(&component.kind) else {
            rollback_registrations(&mut registrations);
            return Err(PluginError::NotFound(format!(
                "plugin component '{}' is not registered",
                component.kind
            )));
        };

        let ordinal = ordinals
            .entry(component.kind.as_str())
            .and_modify(|value| *value += 1)
            .or_insert(1);
        let namespace = component_namespace(
            &component.kind,
            *ordinal,
            totals.get(component.kind.as_str()).copied().unwrap_or(1),
        );

        let mut ctx = PluginRegistrationContext::with_namespace(namespace);
        if let Err(err) = plugin.register(&component.config, &mut ctx).await {
            let mut just_registered = ctx.into_registrations();
            rollback_registrations(&mut just_registered);
            rollback_registrations(&mut registrations);
            return Err(err);
        }
        registrations.extend(ctx.into_registrations());
    }

    Ok(registrations)
}

fn store_active_plugin_configuration(
    config: PluginConfig,
    report: ConfigReport,
    registrations: Vec<PluginRegistration>,
) -> Result<()> {
    let mut guard = ACTIVE_PLUGIN_CONFIGURATION.lock().map_err(|err| {
        PluginError::Internal(format!("active plugin configuration lock poisoned: {err}"))
    })?;
    *guard = Some(ActivePluginConfiguration {
        config,
        report,
        registrations,
    });
    Ok(())
}

fn plugin_component_totals(config: &PluginConfig) -> HashMap<&str, usize> {
    let mut totals = HashMap::new();
    for component in &config.components {
        *totals.entry(component.kind.as_str()).or_insert(0) += 1;
    }
    totals
}

fn component_namespace(kind: &str, ordinal: usize, total: usize) -> String {
    if total > 1 {
        format!("__nemo_relay_plugin__{kind}__{ordinal}__")
    } else {
        format!("__nemo_relay_plugin__{kind}__")
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

        let allows_multiple = lookup_plugin(&component.kind)
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
