// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generic plugin infrastructure for NeMo Flow hosts.
//!
//! This module owns:
//! - config diagnostics and policy enums used by plugin hosts
//! - a global plugin handler registry
//! - plugin registration contexts for middleware/subscriber installation
//! - rollback bookkeeping for registrations created during plugin setup

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, LazyLock, Mutex, RwLock};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as Json};
use thiserror::Error;

use crate::api::{
    nemo_flow_deregister_llm_execution_intercept, nemo_flow_deregister_llm_request_intercept,
    nemo_flow_deregister_llm_stream_execution_intercept, nemo_flow_deregister_subscriber,
    nemo_flow_deregister_tool_execution_intercept, nemo_flow_deregister_tool_request_intercept,
    nemo_flow_register_llm_execution_intercept, nemo_flow_register_llm_request_intercept,
    nemo_flow_register_llm_stream_execution_intercept, nemo_flow_register_subscriber,
    nemo_flow_register_tool_execution_intercept, nemo_flow_register_tool_request_intercept,
};
use crate::context::{
    EventSubscriberFn, LlmExecutionFn, LlmRequestInterceptFn, LlmStreamExecutionFn,
    ToolExecutionFn, ToolInterceptFn,
};

type PluginMap = HashMap<String, Arc<dyn PluginHandler>>;

static PLUGIN_HANDLERS: LazyLock<RwLock<PluginMap>> = LazyLock::new(|| RwLock::new(HashMap::new()));
static ACTIVE_PLUGIN_CONFIGURATION: LazyLock<Mutex<Option<ActivePluginConfiguration>>> =
    LazyLock::new(|| Mutex::new(None));

/// Error type for generic plugin operations.
#[derive(Debug, Error)]
pub enum PluginError {
    /// Configuration validation failed.
    #[error("invalid config: {0}")]
    InvalidConfig(String),

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

/// Canonical plugin host configuration document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginConfig {
    /// Plugin-host config schema version.
    #[serde(default = "default_plugin_config_version")]
    pub version: u32,
    /// Ordered list of top-level plugin components to validate and activate.
    #[serde(default)]
    pub components: Vec<PluginComponentSpec>,
    /// Host-level policy for unsupported plugin kinds, fields, and values.
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
pub struct PluginComponentSpec {
    /// Registered plugin kind string.
    pub kind: String,
    /// Whether the component should be activated.
    ///
    /// Disabled components are still validated but skipped during runtime
    /// registration.
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    /// Component-local JSON config object passed to the plugin handler.
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
#[serde(rename_all = "lowercase")]
pub enum DiagnosticLevel {
    /// Non-fatal compatibility or validation issue.
    Warning,
    /// Fatal validation issue that blocks initialization.
    Error,
}

/// Policy for how unsupported plugin/runtime config is handled.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct ConfigPolicy {
    /// Policy applied when a component kind is unknown to the handler registry.
    #[serde(default = "default_warn")]
    pub unknown_component: UnsupportedBehavior,
    /// Policy applied when a known component contains an unknown field.
    #[serde(default = "default_warn")]
    pub unknown_field: UnsupportedBehavior,
    /// Policy applied when a known field contains an unsupported value.
    #[serde(default = "default_error")]
    pub unsupported_value: UnsupportedBehavior,
}

/// Per-policy behavior for unsupported configuration.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
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
/// NeMo Flow runtime and records the inverse deregistration closure so the host
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
        nemo_flow_register_subscriber(&qualified_name, callback)
            .map_err(|err| PluginError::RegistrationFailed(format!("subscriber: {err}")))?;

        let name_owned = qualified_name;
        self.registrations.push(PluginRegistration::new(
            "plugin",
            name_owned.clone(),
            Box::new(move || {
                nemo_flow_deregister_subscriber(&name_owned)
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
        nemo_flow_register_llm_request_intercept(&qualified_name, priority, break_chain, callback)
            .map_err(|err| {
                PluginError::RegistrationFailed(format!("llm request intercept: {err}"))
            })?;

        let name_owned = qualified_name;
        self.registrations.push(PluginRegistration::new(
            "plugin",
            name_owned.clone(),
            Box::new(move || {
                nemo_flow_deregister_llm_request_intercept(&name_owned)
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

    /// Registers an LLM execution intercept and records its rollback closure.
    pub fn register_llm_execution_intercept(
        &mut self,
        name: &str,
        priority: i32,
        callback: LlmExecutionFn,
    ) -> Result<()> {
        let qualified_name = self.qualify_name(name);
        nemo_flow_register_llm_execution_intercept(&qualified_name, priority, callback).map_err(
            |err| PluginError::RegistrationFailed(format!("llm execution intercept: {err}")),
        )?;

        let name_owned = qualified_name;
        self.registrations.push(PluginRegistration::new(
            "plugin",
            name_owned.clone(),
            Box::new(move || {
                nemo_flow_deregister_llm_execution_intercept(&name_owned)
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
        nemo_flow_register_llm_stream_execution_intercept(&qualified_name, priority, callback)
            .map_err(|err| {
                PluginError::RegistrationFailed(format!("llm stream execution intercept: {err}"))
            })?;

        let name_owned = qualified_name;
        self.registrations.push(PluginRegistration::new(
            "plugin",
            name_owned.clone(),
            Box::new(move || {
                nemo_flow_deregister_llm_stream_execution_intercept(&name_owned)
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
        nemo_flow_register_tool_request_intercept(&qualified_name, priority, break_chain, callback)
            .map_err(|err| {
                PluginError::RegistrationFailed(format!("tool request intercept: {err}"))
            })?;

        let name_owned = qualified_name;
        self.registrations.push(PluginRegistration::new(
            "plugin",
            name_owned.clone(),
            Box::new(move || {
                nemo_flow_deregister_tool_request_intercept(&name_owned)
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
        nemo_flow_register_tool_execution_intercept(&qualified_name, priority, callback).map_err(
            |err| PluginError::RegistrationFailed(format!("tool execution intercept: {err}")),
        )?;

        let name_owned = qualified_name;
        self.registrations.push(PluginRegistration::new(
            "plugin",
            name_owned.clone(),
            Box::new(move || {
                nemo_flow_deregister_tool_execution_intercept(&name_owned)
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

/// Implemented by hosted plugins that register runtime middleware.
pub trait PluginHandler: Send + Sync + 'static {
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

/// Registers a plugin handler by kind.
///
/// Registering the same kind twice returns
/// [`PluginError::RegistrationFailed`].
pub fn register_plugin_handler(handler: Arc<dyn PluginHandler>) -> Result<()> {
    let mut guard = PLUGIN_HANDLERS
        .write()
        .map_err(|err| PluginError::Internal(format!("plugin registry lock poisoned: {err}")))?;
    let plugin_kind = handler.plugin_kind().to_string();
    if guard.contains_key(&plugin_kind) {
        return Err(PluginError::RegistrationFailed(format!(
            "plugin handler '{plugin_kind}' is already registered"
        )));
    }
    guard.insert(plugin_kind, handler);
    Ok(())
}

/// Removes a previously registered plugin handler.
///
/// This affects future validation and initialization only. Active runtime
/// registrations remain until cleared or replaced.
pub fn deregister_plugin_handler(plugin_kind: &str) -> bool {
    PLUGIN_HANDLERS
        .write()
        .ok()
        .and_then(|mut guard| guard.remove(plugin_kind))
        .is_some()
}

/// Lists registered plugin kinds in sorted order.
pub fn list_plugin_kinds() -> Vec<String> {
    let mut kinds = PLUGIN_HANDLERS
        .read()
        .map(|guard| guard.keys().cloned().collect::<Vec<_>>())
        .unwrap_or_default();
    kinds.sort();
    kinds
}

/// Looks up a registered plugin handler by kind.
pub fn plugin_handler(plugin_kind: &str) -> Option<Arc<dyn PluginHandler>> {
    PLUGIN_HANDLERS
        .read()
        .ok()
        .and_then(|guard| guard.get(plugin_kind).cloned())
}

/// Validates a plugin host configuration document.
///
/// This is a pure validation pass. It does not mutate the active runtime
/// configuration.
pub fn validate_plugin_config(config: &PluginConfig) -> ConfigReport {
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
        let Some(handler) = plugin_handler(&component.kind) else {
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
            .extend(handler.validate(&component.config));
    }

    report
}

/// Configures the active global plugin components.
///
/// Initialization validates the supplied config, replaces the active
/// configuration, and rolls back partial registration on failure. If a
/// previous configuration was active, the host attempts to restore it when the
/// new activation fails.
pub async fn initialize_plugins(config: PluginConfig) -> Result<ConfigReport> {
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

/// Deregisters and clears all configured plugin components.
///
/// Registered plugin kinds remain available for future validation and
/// initialization.
pub fn clear_plugin_configuration() -> Result<()> {
    let previous = {
        let mut guard = ACTIVE_PLUGIN_CONFIGURATION.lock().map_err(|err| {
            PluginError::Internal(format!("active plugin configuration lock poisoned: {err}"))
        })?;
        guard.take()
    };
    if let Some(mut previous_state) = previous {
        rollback_registrations(&mut previous_state.registrations);
    }
    Ok(())
}

/// Returns the last successfully configured plugin host report.
///
/// `None` indicates that no plugin configuration is currently active.
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
    let totals = plugin_component_totals(config);
    let mut ordinals: HashMap<&str, usize> = HashMap::new();
    let mut registrations = vec![];

    for component in config
        .components
        .iter()
        .filter(|component| component.enabled)
    {
        let Some(handler) = plugin_handler(&component.kind) else {
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
        if let Err(err) = handler.register(&component.config, &mut ctx).await {
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
        format!("__nemo_flow_plugin__{kind}__{ordinal}__")
    } else {
        format!("__nemo_flow_plugin__{kind}__")
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

        let allows_multiple = plugin_handler(&component.kind)
            .map(|handler| handler.allows_multiple_components())
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
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex, OnceLock};

    use serde_json::json;

    use crate::{NemoFlowContextState, global_context, nemo_flow_llm_request_intercepts};

    struct TestPluginHandler;

    static TEST_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();

    fn test_mutex() -> &'static Mutex<()> {
        TEST_MUTEX.get_or_init(|| Mutex::new(()))
    }

    impl PluginHandler for TestPluginHandler {
        fn plugin_kind(&self) -> &str {
            "test.plugin"
        }

        fn validate(&self, _plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
            vec![ConfigDiagnostic {
                level: DiagnosticLevel::Warning,
                code: "test.warning".into(),
                component: Some("test.plugin".into()),
                field: None,
                message: "validated".into(),
            }]
        }

        fn register<'a>(
            &'a self,
            _plugin_config: &Map<String, Json>,
            ctx: &'a mut PluginRegistrationContext,
        ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
            Box::pin(async move {
                ctx.register_llm_request_intercept(
                    "intercept",
                    1,
                    false,
                    Box::new(|_name, mut request, annotated| {
                        request.headers.insert("x-plugin".into(), json!(true));
                        Ok((request, annotated))
                    }),
                )
            })
        }
    }

    fn reset_global() {
        let ctx = global_context();
        let mut state = ctx.write().unwrap();
        *state = NemoFlowContextState::new();
        clear_plugin_configuration().unwrap();
        let _ = deregister_plugin_handler("test.plugin");
    }

    #[test]
    fn test_config_report_has_errors() {
        let report = ConfigReport {
            diagnostics: vec![ConfigDiagnostic {
                level: DiagnosticLevel::Error,
                code: "x".into(),
                component: None,
                field: None,
                message: "boom".into(),
            }],
        };
        assert!(report.has_errors());
    }

    #[test]
    fn test_register_and_deregister_plugin_handler() {
        let _guard = test_mutex().lock().unwrap();
        reset_global();
        assert!(register_plugin_handler(Arc::new(TestPluginHandler)).is_ok());
        assert!(list_plugin_kinds().contains(&"test.plugin".to_string()));
        assert!(plugin_handler("test.plugin").is_some());
        assert!(deregister_plugin_handler("test.plugin"));
    }

    #[test]
    fn test_plugin_registration_context_registers_and_rolls_back() {
        let _guard = test_mutex().lock().unwrap();
        reset_global();

        let mut ctx = PluginRegistrationContext::new();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime
            .block_on(TestPluginHandler.register(&Map::new(), &mut ctx))
            .unwrap();

        let request = nemo_flow_llm_request_intercepts(
            "model",
            crate::LLMRequest {
                headers: Map::new(),
                content: json!({"messages": []}),
            },
        )
        .unwrap();
        assert_eq!(request.headers.get("x-plugin"), Some(&json!(true)));

        let mut registrations = ctx.into_registrations();
        rollback_registrations(&mut registrations);

        let request = nemo_flow_llm_request_intercepts(
            "model",
            crate::LLMRequest {
                headers: Map::new(),
                content: json!({"messages": []}),
            },
        )
        .unwrap();
        assert_eq!(request.headers.get("x-plugin"), None);
    }

    #[test]
    fn test_initialize_plugins_registers_and_clears_components() {
        let _guard = test_mutex().lock().unwrap();
        reset_global();
        register_plugin_handler(Arc::new(TestPluginHandler)).unwrap();

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let report = runtime
            .block_on(initialize_plugins(PluginConfig {
                components: vec![PluginComponentSpec::new("test.plugin")],
                ..PluginConfig::default()
            }))
            .unwrap();
        assert!(!report.has_errors());
        assert!(active_plugin_report().is_some());

        let request = nemo_flow_llm_request_intercepts(
            "model",
            crate::LLMRequest {
                headers: Map::new(),
                content: json!({"messages": []}),
            },
        )
        .unwrap();
        assert_eq!(request.headers.get("x-plugin"), Some(&json!(true)));

        clear_plugin_configuration().unwrap();
        let request = nemo_flow_llm_request_intercepts(
            "model",
            crate::LLMRequest {
                headers: Map::new(),
                content: json!({"messages": []}),
            },
        )
        .unwrap();
        assert_eq!(request.headers.get("x-plugin"), None);
    }
}
