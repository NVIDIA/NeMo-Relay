// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Core plugin integration for the adaptive runtime.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use nemo_relay::plugin::{
    ConfigDiagnostic, ConfigPolicy, DiagnosticLevel, Plugin, PluginComponentSpec, PluginError,
    PluginRegistration, PluginRegistrationContext, Result, UnsupportedBehavior, deregister_plugin,
    lookup_plugin, register_plugin,
};
use serde_json::{Map, Value as Json};

use crate::config::AdaptiveConfig;
use crate::error::AdaptiveError;
use crate::runtime::features::AdaptiveRuntime;

/// The plugin kind registered by the adaptive crate.
pub const ADAPTIVE_PLUGIN_KIND: &str = "adaptive";

/// One configured adaptive component.
#[derive(Debug, Clone)]
pub struct ComponentSpec {
    /// Whether the adaptive component should be activated.
    pub enabled: bool,
    /// Adaptive config for this top-level component.
    pub config: AdaptiveConfig,
}

impl ComponentSpec {
    /// Creates an enabled adaptive component spec.
    pub fn new(config: AdaptiveConfig) -> Self {
        Self {
            enabled: true,
            config,
        }
    }
}

impl From<ComponentSpec> for PluginComponentSpec {
    fn from(value: ComponentSpec) -> Self {
        let Json::Object(config) =
            serde_json::to_value(value.config).expect("adaptive config should serialize to object")
        else {
            unreachable!("adaptive config must serialize to object");
        };

        PluginComponentSpec {
            kind: ADAPTIVE_PLUGIN_KIND.to_string(),
            enabled: value.enabled,
            config,
        }
    }
}

struct AdaptivePlugin;

impl Plugin for AdaptivePlugin {
    fn plugin_kind(&self) -> &str {
        ADAPTIVE_PLUGIN_KIND
    }

    fn allows_multiple_components(&self) -> bool {
        false
    }

    fn validate(&self, plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        validate_adaptive_plugin_config(plugin_config)
    }

    fn register<'a>(
        &'a self,
        plugin_config: &Map<String, Json>,
        ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        let plugin_config = plugin_config.clone();
        Box::pin(async move {
            let config = parse_adaptive_config(&plugin_config)?;
            let mut runtime = AdaptiveRuntime::new(config)
                .await
                .map_err(adaptive_to_plugin_error)?;
            runtime.register().await.map_err(adaptive_to_plugin_error)?;

            let runtime = Arc::new(Mutex::new(Some(runtime)));
            ctx.add_registration(PluginRegistration::new(
                ADAPTIVE_PLUGIN_KIND,
                ADAPTIVE_PLUGIN_KIND,
                Box::new(move || {
                    let mut guard = runtime.lock().map_err(|err| {
                        PluginError::Internal(format!(
                            "adaptive runtime registration lock poisoned: {err}"
                        ))
                    })?;
                    if let Some(mut runtime) = guard.take() {
                        runtime.deregister().map_err(adaptive_to_plugin_error)?;
                    }
                    Ok(())
                }),
            ));
            Ok(())
        })
    }
}

/// Registers the adaptive component kind in the core plugin registry.
///
/// Call this during startup before validating or initializing plugin configs
/// that contain adaptive components.
///
/// # Returns
/// A core plugin [`Result`] that is `Ok(())` when the adaptive component kind
/// is available in the registry.
///
/// # Errors
/// Returns an error when registration fails for a reason other than an already
/// registered adaptive component.
///
/// # Notes
/// Re-registering the adaptive component is treated as success when the
/// existing registration already resolves to the adaptive plugin kind.
pub fn register_adaptive_component() -> Result<()> {
    match register_plugin(Arc::new(AdaptivePlugin)) {
        Ok(()) => Ok(()),
        Err(PluginError::RegistrationFailed(message))
            if message.contains("already registered")
                && lookup_plugin(ADAPTIVE_PLUGIN_KIND).is_some() =>
        {
            Ok(())
        }
        Err(err) => Err(err),
    }
}

/// Deregisters the adaptive component kind from the core plugin registry.
///
/// This affects future validation and initialization only. Active adaptive
/// runtime registrations remain until cleared or replaced.
///
/// # Returns
/// `true` when the adaptive component kind was removed from the registry and
/// `false` when it was not registered.
///
/// # Notes
/// Active adaptive runtime registrations are not torn down by this function.
pub fn deregister_adaptive_component() -> bool {
    deregister_plugin(ADAPTIVE_PLUGIN_KIND)
}

fn parse_adaptive_config(plugin_config: &Map<String, Json>) -> Result<AdaptiveConfig> {
    serde_json::from_value(Json::Object(plugin_config.clone()))
        .map_err(|err| PluginError::InvalidConfig(format!("invalid adaptive plugin config: {err}")))
}

fn validate_adaptive_plugin_config(plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
    let config = match parse_adaptive_config(plugin_config) {
        Ok(config) => config,
        Err(err) => {
            return vec![ConfigDiagnostic {
                level: DiagnosticLevel::Error,
                code: "adaptive.invalid_plugin_config".to_string(),
                component: Some(ADAPTIVE_PLUGIN_KIND.to_string()),
                field: None,
                message: err.to_string(),
            }];
        }
    };

    let mut diagnostics = vec![];
    validate_unknown_fields(
        &mut diagnostics,
        &config.policy,
        Some(ADAPTIVE_PLUGIN_KIND.to_string()),
        plugin_config,
        &[
            "version",
            "agent_id",
            "state",
            "telemetry",
            "adaptive_hints",
            "tool_parallelism",
            "acg",
            "policy",
        ],
    );

    if let Some(policy_json) = plugin_config.get("policy").and_then(Json::as_object) {
        validate_unknown_fields(
            &mut diagnostics,
            &config.policy,
            Some("policy".to_string()),
            policy_json,
            &["unknown_component", "unknown_field", "unsupported_value"],
        );
    }

    if let Some(state_json) = plugin_config.get("state").and_then(Json::as_object) {
        validate_unknown_fields(
            &mut diagnostics,
            &config.policy,
            Some("state".to_string()),
            state_json,
            &["backend"],
        );
        if let Some(backend_json) = state_json.get("backend").and_then(Json::as_object) {
            validate_unknown_fields(
                &mut diagnostics,
                &config.policy,
                Some("backend".to_string()),
                backend_json,
                &["kind", "config"],
            );
            let backend_kind = backend_json
                .get("kind")
                .and_then(Json::as_str)
                .unwrap_or_default();
            if let Some(backend_config_json) = backend_json.get("config").and_then(Json::as_object)
            {
                validate_backend_config_fields(
                    &mut diagnostics,
                    &config.policy,
                    backend_kind,
                    backend_config_json,
                );
            }
        }
    }

    if let Some(telemetry_json) = plugin_config.get("telemetry").and_then(Json::as_object) {
        validate_unknown_fields(
            &mut diagnostics,
            &config.policy,
            Some("telemetry".to_string()),
            telemetry_json,
            &["subscriber_name", "learners"],
        );
    }

    if let Some(adaptive_hints_json) = plugin_config
        .get("adaptive_hints")
        .and_then(Json::as_object)
    {
        validate_unknown_fields(
            &mut diagnostics,
            &config.policy,
            Some("adaptive_hints".to_string()),
            adaptive_hints_json,
            &[
                "priority",
                "break_chain",
                "inject_header",
                "inject_body_path",
            ],
        );
    }

    if let Some(tool_parallelism_json) = plugin_config
        .get("tool_parallelism")
        .and_then(Json::as_object)
    {
        validate_unknown_fields(
            &mut diagnostics,
            &config.policy,
            Some("tool_parallelism".to_string()),
            tool_parallelism_json,
            &["priority", "mode"],
        );
    }

    if let Some(acg_json) = plugin_config.get("acg").and_then(Json::as_object) {
        validate_unknown_fields(
            &mut diagnostics,
            &config.policy,
            Some("acg".to_string()),
            acg_json,
            &[
                "provider",
                "observation_window",
                "priority",
                "stability_thresholds",
            ],
        );
    }

    diagnostics.extend(AdaptiveRuntime::validate_config(&config).diagnostics);
    diagnostics
}

fn validate_backend_config_fields(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    backend_kind: &str,
    backend_config: &Map<String, Json>,
) {
    let known_fields: &[&str] = match backend_kind {
        "in_memory" => &[],
        "redis" => &["url", "key_prefix"],
        _ => return,
    };
    validate_unknown_fields(
        diagnostics,
        policy,
        Some(backend_kind.to_string()),
        backend_config,
        known_fields,
    );
}

fn validate_unknown_fields(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    component: Option<String>,
    config: &Map<String, Json>,
    known_fields: &[&str],
) {
    for field in config.keys() {
        if !known_fields.contains(&field.as_str()) {
            push_policy_diag(
                diagnostics,
                policy.unknown_field,
                "adaptive.unknown_field",
                component.clone(),
                Some(field.clone()),
                format!(
                    "field '{}' is not recognized for '{}'",
                    field,
                    component.as_deref().unwrap_or("unknown")
                ),
            );
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

fn adaptive_to_plugin_error(err: AdaptiveError) -> PluginError {
    match err {
        AdaptiveError::InvalidConfig(message) => PluginError::InvalidConfig(message),
        AdaptiveError::NotFound(message) => PluginError::NotFound(message),
        AdaptiveError::Storage(message) => PluginError::Internal(message),
        AdaptiveError::Serialization(err) => PluginError::Serialization(err),
        AdaptiveError::Internal(message) => PluginError::Internal(message),
        AdaptiveError::RegistrationFailed(message) => PluginError::RegistrationFailed(message),
        AdaptiveError::ChannelClosed(message) => PluginError::Internal(message),
        #[cfg(feature = "redis-backend")]
        AdaptiveError::Redis(err) => PluginError::Internal(err.to_string()),
    }
}

#[cfg(test)]
#[path = "../tests/unit/plugin_component_tests.rs"]
mod tests;
