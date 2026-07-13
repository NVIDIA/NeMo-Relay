// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `nemo-relay doctor` — environment + config + agent + observability health check.
//!
//! Split into three layers so the data path can be unit-tested without real I/O:
//!
//! - `collect_report()` does the I/O (env probes, $PATH scans, network checks, fs writability).
//! - `DoctorReport` is the resulting pure data shape.
//! - `format_human(&report)` / `format_json(&report)` render the report.

use std::path::{Path, PathBuf};
use std::time::Duration;

use futures_util::SinkExt;
use nemo_relay::api::event::{BaseEvent, Event, MarkEvent};
use nemo_relay::codec::model_pricing::{PricingCatalog, PricingConfig, PricingSourceConfig};
use nemo_relay::observability::plugin_component::OBSERVABILITY_PLUGIN_KIND;
use nemo_relay::plugin::{DiagnosticLevel, PluginConfig, validate_plugin_config};
use nemo_relay_adaptive::plugin_component::register_adaptive_component;
use nemo_relay_pii_redaction::component::register_pii_redaction_component;
use serde::Serialize;
use serde_json::{Value, json};
use tokio::time::timeout;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use uuid::Uuid;

use crate::configuration::{
    AgentConfigs, CodingAgent, DynamicPluginHostConfigStatus, GatewayConfig, ResolvedConfig,
    ServerArgs, default_plugin_config_paths, effective_plugin_toml_sources, resolve_server_config,
};
use crate::error::CliError;

const NETWORK_TIMEOUT: Duration = Duration::from_secs(2);
const PRICING_PLUGIN_KIND: &str = "pricing";

/// Outcome of one check inside the doctor report. The `details` field carries human-readable
/// supplementary text; the `status` is the bottom-line signal callers (and CI) use to decide
/// pass/fail.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct Check {
    pub name: &'static str,
    pub status: Status,
    pub details: String,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Status {
    Pass,
    Warn,
    Fail,
    /// The check ran but no relevant state was detected — purely informational (e.g. an agent
    /// not on $PATH). Renders as a dot; not counted toward exit code.
    Info,
}

/// Snapshot of the running system that the doctor renders. Stable schema, versioned via
/// `schema_version`. Adding fields is non-breaking; removing or renaming requires a bump.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct DoctorReport {
    pub schema_version: u32,
    pub binary_version: &'static str,
    pub target_agent: Option<String>,
    pub environment: EnvironmentInfo,
    pub configuration: ConfigurationInfo,
    pub agents: Vec<AgentInfo>,
    pub host_plugins: Vec<crate::agents::install::HostPluginReadiness>,
    pub observability: Vec<Check>,
    pub completions: Vec<Check>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct EnvironmentInfo {
    pub os: String,
    pub arch: &'static str,
    pub shell: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ConfigurationInfo {
    pub workspace: ConfigLayer,
    pub global: ConfigLayer,
    pub system: ConfigLayer,
    pub plugin_configs: Vec<ConfigLayer>,
    pub plugin_resolution: Check,
    pub resolution: Check,
    pub default_agent: Option<String>,
    pub configured_agents: Vec<String>,
    pub dynamic_plugins: Vec<DynamicPluginReferenceInfo>,
}

struct PluginConfigurationDiagnostics {
    sources: Vec<PathBuf>,
    error: Option<String>,
    resolution: Check,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct DynamicPluginReferenceInfo {
    pub plugin_id: String,
    pub manifest_ref: String,
    pub source: PathBuf,
    pub host_config_status: DynamicPluginHostConfigStatus,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ConfigLayer {
    pub path: PathBuf,
    pub status: Status,
    pub active: bool,
    pub details: String,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct AgentInfo {
    pub name: &'static str,
    pub status: Status,
    pub configured: bool,
    pub command: String,
    pub path: Option<PathBuf>,
    pub version: Option<String>,
    /// Free-form annotation, e.g. "hooks: installed" once we wire up hook detection.
    pub annotation: String,
}

/// Drives all checks and produces a single `DoctorReport`. Network probes are bounded by a
/// short timeout so the command always returns quickly. Filesystem checks short-circuit on
/// the first missing directory.
pub(crate) async fn collect_report(
    target_agent: Option<CodingAgent>,
) -> Result<DoctorReport, CliError> {
    let (resolved, resolution) = match resolve_server_config(&ServerArgs::default()) {
        Ok(resolved) => (
            resolved,
            Check {
                name: "Resolution",
                status: Status::Pass,
                details: "valid".into(),
            },
        ),
        Err(err) => (
            ResolvedConfig::default(),
            Check {
                name: "Resolution",
                status: Status::Fail,
                details: format!("could not resolve merged config: {err}"),
            },
        ),
    };
    let cwd = std::env::current_dir().ok();
    let home = home_dir();
    let configured_agents = configured_agent_names(&resolved.agents);
    let (plugin_sources, plugin_error) = match effective_plugin_toml_sources() {
        Ok(sources) => (sources, None),
        Err(error) => (Vec::new(), Some(error.to_string())),
    };
    let plugin_resolution =
        plugin_resolution_check(&resolved, &resolution, plugin_error.as_deref());
    let plugin_diagnostics = PluginConfigurationDiagnostics {
        sources: plugin_sources,
        error: plugin_error,
        resolution: plugin_resolution,
    };

    Ok(DoctorReport {
        schema_version: 1,
        binary_version: env!("CARGO_PKG_VERSION"),
        target_agent: target_agent.map(|agent| agent.as_arg().to_string()),
        environment: collect_environment(),
        configuration: collect_configuration(
            cwd.as_deref(),
            home.as_deref(),
            resolution,
            configured_agents,
            &resolved.dynamic_plugins,
            &plugin_diagnostics,
        ),
        agents: collect_agents(target_agent, &resolved).await,
        host_plugins: crate::agents::install::collect_default_host_plugin_readiness(),
        observability: collect_observability(&resolved.gateway).await,
        completions: collect_completions(home.as_deref()),
    })
}

fn collect_environment() -> EnvironmentInfo {
    EnvironmentInfo {
        os: format!("{} {}", std::env::consts::OS, os_version()),
        arch: std::env::consts::ARCH,
        shell: std::env::var("SHELL").ok().and_then(|path| {
            std::path::Path::new(&path)
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
        }),
    }
}

fn os_version() -> String {
    // `uname -r` works on macOS/Linux; on Windows we just report the OS name with no detail.
    if cfg!(windows) {
        return String::new();
    }
    match std::process::Command::new("uname").arg("-r").output() {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).trim().to_string(),
        _ => String::new(),
    }
}

fn collect_configuration(
    cwd: Option<&Path>,
    home: Option<&Path>,
    resolution: Check,
    configured_agents: Vec<String>,
    dynamic_plugins: &[crate::configuration::ResolvedDynamicPluginConfig],
    plugin_diagnostics: &PluginConfigurationDiagnostics,
) -> ConfigurationInfo {
    let workspace_path = cwd
        .map(|p| p.join(".nemo-relay").join("config.toml"))
        .unwrap_or_else(|| PathBuf::from(".nemo-relay/config.toml"));
    // Use the same XDG-aware resolver the config loader uses, so doctor reports the path the
    // runtime would actually read instead of a hard-coded `$HOME/.config/nemo-relay`.
    let global_path = crate::configuration::user_config_dir()
        .map(|dir| dir.join("config.toml"))
        .or_else(|| home.map(|h| h.join(".config").join("nemo-relay").join("config.toml")))
        .unwrap_or_else(|| PathBuf::from("~/.config/nemo-relay/config.toml"));
    let system_path = PathBuf::from("/etc/nemo-relay/config.toml");

    ConfigurationInfo {
        workspace: layer_status(&workspace_path),
        global: layer_status(&global_path),
        system: layer_status(&system_path),
        plugin_configs: default_plugin_config_paths()
            .iter()
            .map(|path| {
                plugin_layer_status(
                    path,
                    &plugin_diagnostics.sources,
                    plugin_diagnostics.error.as_deref(),
                )
            })
            .collect(),
        plugin_resolution: plugin_diagnostics.resolution.clone(),
        resolution,
        // `default_agent` is reserved in the design for Phase 2 dispatch; not currently parsed
        // out of FileConfig. Doctor reports `None` until that lands.
        default_agent: None,
        configured_agents,
        dynamic_plugins: dynamic_plugins
            .iter()
            .map(|plugin| DynamicPluginReferenceInfo {
                plugin_id: plugin.plugin_id.clone(),
                manifest_ref: plugin.manifest_ref.clone(),
                source: plugin.source.clone(),
                host_config_status: plugin.host_config_status(),
            })
            .collect(),
    }
}

fn plugin_resolution_check(
    resolved: &ResolvedConfig,
    resolution: &Check,
    plugin_error: Option<&str>,
) -> Check {
    if let Some(error) = plugin_error {
        return Check {
            name: "Plugin resolution",
            status: Status::Fail,
            details: format!(
                "could not resolve plugins.toml: {error}; update the named source and run `nemo-relay plugins edit`"
            ),
        };
    }
    if matches!(resolution.status, Status::Fail) {
        return Check {
            name: "Plugin resolution",
            status: Status::Fail,
            details: resolution.details.clone(),
        };
    }
    if resolved.gateway.plugin_config.is_some() {
        Check {
            name: "Plugin resolution",
            status: Status::Info,
            details: "effective plugin configuration loaded; see Plugin validation below".into(),
        }
    } else if !resolved.dynamic_plugins.is_empty() {
        Check {
            name: "Plugin resolution",
            status: Status::Info,
            details: "dynamic plugin configuration loaded; see Dynamic plugin checks below".into(),
        }
    } else {
        Check {
            name: "Plugin resolution",
            status: Status::Info,
            details:
                "plugins.toml not configured; run `nemo-relay plugins edit` to configure plugins"
                    .into(),
        }
    }
}

fn dynamic_plugin_reference_check(plugin: &DynamicPluginReferenceInfo) -> Check {
    Check {
        name: "Dynamic plugin",
        status: Status::Pass,
        details: format!("{} resolved from {}", plugin.plugin_id, plugin.manifest_ref),
    }
}

fn dynamic_plugin_host_config_check(plugin: &DynamicPluginReferenceInfo) -> Check {
    let details = match plugin.host_config_status {
        DynamicPluginHostConfigStatus::Absent => {
            format!(
                "{} discovered via host config only; not enabled by config alone",
                plugin.plugin_id
            )
        }
        DynamicPluginHostConfigStatus::Present => format!(
            "{} discovered via host config; host-owned config present; not enabled by config alone",
            plugin.plugin_id
        ),
    };
    Check {
        name: "Dynamic plugin",
        status: Status::Info,
        details,
    }
}

fn layer_status(path: &Path) -> ConfigLayer {
    if !path.exists() {
        return ConfigLayer {
            path: path.to_path_buf(),
            status: Status::Info,
            active: false,
            details: "not present".into(),
        };
    }
    match std::fs::read_to_string(path) {
        // Parse as `toml::Table` to match the rest of the loader (config.rs::load_shared_config).
        // `toml::Value` parsing in `toml = 0.9` treats multi-section docs as a single Value and
        // chokes on the second section header, so `Table` is the right top-level shape.
        Ok(text) => match text.parse::<toml::Table>() {
            Ok(_) => ConfigLayer {
                path: path.to_path_buf(),
                status: Status::Pass,
                active: true,
                details: "valid".into(),
            },
            Err(err) => ConfigLayer {
                path: path.to_path_buf(),
                status: Status::Fail,
                active: false,
                details: format!("invalid TOML: {err}"),
            },
        },
        Err(err) => ConfigLayer {
            path: path.to_path_buf(),
            status: Status::Fail,
            active: false,
            details: format!("unreadable: {err}"),
        },
    }
}

fn plugin_layer_status(
    path: &Path,
    contributing_paths: &[PathBuf],
    plugin_error: Option<&str>,
) -> ConfigLayer {
    let mut layer = layer_status(path);
    if let Some(error) = plugin_error.filter(|error| error.contains(&path.display().to_string()))
        && matches!(layer.status, Status::Pass)
    {
        layer.status = Status::Fail;
        layer.active = false;
        layer.details = format!("invalid plugin configuration: {error}");
        return layer;
    }
    if layer.active && contributing_paths.iter().any(|source| source == path) {
        layer.details = "discovered and contributes to plugin resolution".into();
    } else if layer.active {
        layer.active = false;
        layer.details = "valid but does not contribute effective plugin configuration".into();
    }
    layer
}

async fn collect_agents(
    target_agent: Option<CodingAgent>,
    resolved: &ResolvedConfig,
) -> Vec<AgentInfo> {
    let mut out = Vec::with_capacity(CodingAgent::ALL.len());
    for agent in CodingAgent::ALL {
        if target_agent.is_some_and(|target| target != agent) {
            continue;
        }
        let configured = agent_configured(agent, &resolved.agents);
        let target_requested = target_agent == Some(agent);
        let command = agent_command(agent, &resolved.agents);
        let argv = crate::process::command_argv(&command);
        let exec = argv.first().map(String::as_str).unwrap_or_default();
        let path = crate::process::resolve_executable(exec);
        let version = match &path {
            Some(_) => {
                let probe = crate::process::version_probe_argv(agent, &argv);
                probe_version(&probe).await
            }
            None => None,
        };
        let mut status = agent_command_status(path.as_deref(), configured, target_requested);
        let (hook_status, hook_details) = hook_status(agent, &resolved.agents);
        status = combine_status(status, hook_status, configured || target_requested);
        let mut details = Vec::new();
        details.push(if configured {
            "configured".to_string()
        } else if target_requested {
            "not configured; first run will launch setup".to_string()
        } else {
            "not configured".to_string()
        });
        if path.is_none() {
            details.push(format!("command `{exec}` not found"));
        }
        if !hook_details.is_empty() {
            details.push(hook_details);
        }
        let version_required = configured || target_requested;
        match version.as_deref() {
            Some(version) => {
                if let Err(error) = agent.validate_version_output(version) {
                    status = combine_status(
                        status,
                        if version_required {
                            Status::Fail
                        } else {
                            Status::Warn
                        },
                        true,
                    );
                    details.push(error);
                }
            }
            None if path.is_some() => {
                status = combine_status(
                    status,
                    if version_required {
                        Status::Fail
                    } else {
                        Status::Warn
                    },
                    true,
                );
                details.push(format!(
                    "could not determine version; NeMo Relay requires {}",
                    agent.version_requirement()
                ));
            }
            None => {}
        }
        out.push(AgentInfo {
            name: agent.as_arg(),
            status,
            configured,
            command,
            path,
            version,
            annotation: details.join("; "),
        });
    }
    out
}

fn agent_command(agent: CodingAgent, agents: &AgentConfigs) -> String {
    configured_agent_command(agent, agents)
        .cloned()
        .unwrap_or_else(|| agent.executable().to_string())
}

fn configured_agent_command(agent: CodingAgent, agents: &AgentConfigs) -> Option<&String> {
    agents.get(agent).command.as_ref()
}

fn agent_configured(agent: CodingAgent, agents: &AgentConfigs) -> bool {
    configured_agent_command(agent, agents).is_some()
        || (matches!(agent, CodingAgent::Hermes) && agents.hermes.hooks_path.is_some())
}

fn configured_agent_names(agents: &AgentConfigs) -> Vec<String> {
    CodingAgent::ALL
        .into_iter()
        .filter_map(|agent| agent_configured(agent, agents).then_some(agent.as_arg().to_string()))
        .collect()
}

fn agent_command_status(path: Option<&Path>, configured: bool, target_requested: bool) -> Status {
    match (path.is_some(), configured, target_requested) {
        (true, false, true) => Status::Warn,
        (true, _, _) => Status::Pass,
        (false, true, _) | (false, _, true) => Status::Fail,
        (false, false, false) => Status::Info,
    }
}

fn combine_status(base: Status, hook: Status, readiness_required: bool) -> Status {
    if matches!(base, Status::Fail) || matches!(hook, Status::Fail) {
        return Status::Fail;
    }
    if matches!(base, Status::Warn) || (readiness_required && matches!(hook, Status::Warn)) {
        return Status::Warn;
    }
    base
}

fn hook_status(agent: CodingAgent, agents: &AgentConfigs) -> (Status, String) {
    match agent {
        CodingAgent::ClaudeCode | CodingAgent::Codex => {
            (Status::Pass, "hooks: injected during run".into())
        }
        CodingAgent::Hermes => match agents.hermes.hooks_path.as_deref() {
            Some(path) => match crate::agents::hermes::diagnose_persistent(path) {
                Ok(details) => (Status::Pass, details),
                Err(error) => (
                    Status::Fail,
                    format!(
                        "persistent MCP/hooks: {error}; run `nemo-relay install hermes --force`"
                    ),
                ),
            },
            None => (
                Status::Pass,
                "hooks: injected through an isolated HERMES_HOME during run".into(),
            ),
        },
    }
}

async fn probe_version(argv: &[String]) -> Option<String> {
    // Run the shared wrapper-preserving probe and read the first line of stdout. Bounded by the network
    // timeout (re-used as a generic short timeout) so a misbehaving binary doesn't hang doctor.
    let mut cmd = crate::process::tokio_command(argv);
    cmd.stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .stdin(std::process::Stdio::null())
        // Ensure the child gets killed if our future is dropped on timeout. Without this a
        // misbehaving agent binary that exceeds NETWORK_TIMEOUT would leak as an orphan
        // process for the lifetime of the doctor invocation (and beyond).
        .kill_on_drop(true);
    let child = cmd.spawn().ok()?;
    let output = timeout(NETWORK_TIMEOUT, child.wait_with_output())
        .await
        .ok()?
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let first_line = stdout.lines().next()?.trim();
    if first_line.is_empty() {
        None
    } else {
        Some(first_line.to_string())
    }
}

async fn collect_observability(gateway: &GatewayConfig) -> Vec<Check> {
    let mut checks = Vec::new();

    let Some(plugin_value) = &gateway.plugin_config else {
        checks.push(Check {
            name: "Plugin validation",
            status: Status::Info,
            details: "plugins.toml not configured".into(),
        });
        return checks;
    };

    let plugin_config = match serde_json::from_value::<PluginConfig>(plugin_value.clone()) {
        Ok(config) => config,
        Err(err) => {
            checks.push(Check {
                name: "Plugin validation",
                status: Status::Fail,
                details: format!("invalid plugin config: {err}"),
            });
            return checks;
        }
    };
    if let Err(error) = register_adaptive_component() {
        checks.push(Check {
            name: "Adaptive plugin",
            status: Status::Fail,
            details: format!("registration failed: {error}"),
        });
        return checks;
    }
    if let Err(error) = register_pii_redaction_component() {
        checks.push(Check {
            name: "PII redaction plugin",
            status: Status::Fail,
            details: format!("registration failed: {error}"),
        });
        return checks;
    }
    let report = validate_plugin_config(&plugin_config);
    if report.diagnostics.is_empty() {
        checks.push(Check {
            name: "Plugin validation",
            status: Status::Pass,
            details: "validation passed".into(),
        });
    } else {
        for diagnostic in report.diagnostics {
            checks.push(Check {
                name: "Plugin diagnostic",
                status: if diagnostic.level == DiagnosticLevel::Error {
                    Status::Fail
                } else {
                    Status::Warn
                },
                details: format!("{}: {}", diagnostic.code, diagnostic.message),
            });
        }
    }

    if let Some(config) = observability_component_config(plugin_value) {
        collect_observability_component_checks(&mut checks, config).await;
    } else {
        checks.push(Check {
            name: "Observability plugin",
            status: Status::Info,
            details: "component not configured".into(),
        });
    }
    collect_pricing_component_checks(&mut checks, &plugin_config);

    checks
}

async fn collect_observability_component_checks(checks: &mut Vec<Check>, config: &Value) {
    for section in ["atof", "atif"] {
        if let Some(check) = observability_file_exporter_check(config, section) {
            checks.push(check);
        }
    }
    for section in ["opentelemetry", "openinference"] {
        if let Some(check) = observability_http_exporter_check(config, section).await {
            checks.push(check);
        }
    }
    if section_enabled(config, "atof") && atof_endpoint_count(config) > 0 {
        if atof_streaming_supported() {
            checks.extend(observability_atof_endpoint_checks(config).await);
        } else {
            checks.push(Check {
                name: "ATOF endpoint",
                status: Status::Fail,
                details: "ATOF streaming endpoints are not available in this binary".into(),
            });
        }
    }
}

fn observability_file_exporter_check(config: &Value, section: &str) -> Option<Check> {
    if !section_enabled(config, section) {
        return None;
    }
    let label = if section == "atof" {
        "ATOF dir"
    } else {
        "ATIF dir"
    };
    Some(match section_output_directory(config, section) {
        Some(path) => check_directory(label, &path),
        None => Check {
            name: label,
            status: Status::Info,
            details: "enabled; using runtime default output directory".into(),
        },
    })
}

async fn observability_http_exporter_check(config: &Value, section: &str) -> Option<Check> {
    if !section_enabled(config, section) {
        return None;
    }
    let label = if section == "opentelemetry" {
        "OpenTelemetry endpoint"
    } else {
        "OpenInference endpoint"
    };
    Some(match section_endpoint(config, section) {
        Some(endpoint) => probe_http_named(label, &endpoint).await,
        None => Check {
            name: label,
            status: Status::Info,
            details: "enabled; using exporter default endpoint".into(),
        },
    })
}

fn observability_component_config(plugin_value: &Value) -> Option<&Value> {
    plugin_value
        .get("components")
        .and_then(Value::as_array)
        .and_then(|components| {
            components.iter().find(|component| {
                component
                    .get("kind")
                    .and_then(Value::as_str)
                    .is_some_and(|kind| kind == OBSERVABILITY_PLUGIN_KIND)
            })
        })
        .and_then(|component| component.get("config"))
}

fn collect_pricing_component_checks(checks: &mut Vec<Check>, plugin_config: &PluginConfig) {
    let Some(component) = plugin_config
        .components
        .iter()
        .find(|component| component.kind == PRICING_PLUGIN_KIND)
    else {
        checks.push(Check {
            name: "Model pricing",
            status: Status::Info,
            details: "component not configured".into(),
        });
        return;
    };

    if !component.enabled {
        checks.push(Check {
            name: "Model pricing",
            status: Status::Info,
            details: "component disabled".into(),
        });
        return;
    }

    let config =
        match serde_json::from_value::<PricingConfig>(Value::Object(component.config.clone())) {
            Ok(config) => config,
            Err(error) => {
                checks.push(Check {
                    name: "Model pricing",
                    status: Status::Fail,
                    details: format!("invalid config: {error}"),
                });
                return;
            }
        };

    if config.sources.is_empty() {
        checks.push(Check {
            name: "Model pricing",
            status: Status::Info,
            details: "component configured with no sources".into(),
        });
        return;
    }

    for (index, source) in config.sources.iter().enumerate() {
        checks.push(pricing_source_check(index, source));
    }
}

fn pricing_source_check(index: usize, source: &PricingSourceConfig) -> Check {
    match source {
        PricingSourceConfig::Inline { catalog } => Check {
            name: "Model pricing source",
            status: Status::Pass,
            details: format!("inline:{index} valid ({} entries)", catalog.entries.len()),
        },
        PricingSourceConfig::File { path } => match std::fs::read_to_string(path) {
            Ok(raw) => match PricingCatalog::from_json_str(&raw) {
                Ok(catalog) => Check {
                    name: "Model pricing source",
                    status: Status::Pass,
                    details: format!(
                        "file:{} valid ({} entries)",
                        path.display(),
                        catalog.entries.len()
                    ),
                },
                Err(error) => Check {
                    name: "Model pricing source",
                    status: Status::Fail,
                    details: format!("file:{} invalid catalog: {error}", path.display()),
                },
            },
            Err(error) => Check {
                name: "Model pricing source",
                status: Status::Fail,
                details: format!("file:{} unreadable: {error}", path.display()),
            },
        },
    }
}

fn section_enabled(config: &Value, section: &str) -> bool {
    config
        .get(section)
        .and_then(|section| section.get("enabled"))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

fn section_output_directory(config: &Value, section: &str) -> Option<PathBuf> {
    config
        .get(section)
        .and_then(|section| section.get("output_directory"))
        .and_then(Value::as_str)
        .map(PathBuf::from)
}

fn section_endpoint(config: &Value, section: &str) -> Option<String> {
    config
        .get(section)
        .and_then(|section| section.get("endpoint"))
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn atof_endpoint_count(config: &Value) -> usize {
    config
        .get("atof")
        .and_then(|section| section.get("endpoints"))
        .and_then(Value::as_array)
        .map_or(0, Vec::len)
}

fn atof_streaming_supported() -> bool {
    cfg!(feature = "atof-streaming")
}

async fn observability_atof_endpoint_checks(config: &Value) -> Vec<Check> {
    let Some(endpoints) = config
        .get("atof")
        .and_then(|section| section.get("endpoints"))
        .and_then(Value::as_array)
    else {
        return Vec::new();
    };
    let mut checks = Vec::with_capacity(endpoints.len());
    for (index, endpoint) in endpoints.iter().enumerate() {
        checks.push(probe_atof_endpoint(index, endpoint).await);
    }
    checks
}

async fn probe_atof_endpoint(index: usize, endpoint: &Value) -> Check {
    let name = "ATOF endpoint";
    let Some(url) = endpoint.get("url").and_then(Value::as_str) else {
        return Check {
            name,
            status: Status::Fail,
            details: format!("endpoints[{index}]: missing url"),
        };
    };
    let transport = endpoint
        .get("transport")
        .and_then(Value::as_str)
        .unwrap_or("http_post");
    let timeout_millis = endpoint
        .get("timeout_millis")
        .and_then(Value::as_u64)
        .unwrap_or(3_000);
    if timeout_millis == 0 {
        return Check {
            name,
            status: Status::Fail,
            details: format!("endpoints[{index}] {transport} {url}: timeout_millis must be > 0"),
        };
    }
    let headers = match endpoint_headers(endpoint) {
        Ok(headers) => headers,
        Err(err) => {
            return Check {
                name,
                status: Status::Fail,
                details: format!("endpoints[{index}] {transport} {url}: {err}"),
            };
        }
    };
    let payload = match doctor_atof_probe_payload() {
        Ok(payload) => payload,
        Err(err) => {
            return Check {
                name,
                status: Status::Fail,
                details: format!("endpoints[{index}] {transport} {url}: {err}"),
            };
        }
    };
    let timeout_duration = Duration::from_millis(timeout_millis);
    match transport {
        "http_post" => probe_atof_http_post(url, headers, payload, timeout_duration, index).await,
        "websocket" => probe_atof_websocket(url, headers, payload, timeout_duration, index).await,
        "ndjson" => probe_atof_ndjson(url, headers, payload, timeout_duration, index).await,
        _ => Check {
            name,
            status: Status::Fail,
            details: format!("endpoints[{index}] {transport} {url}: unsupported transport"),
        },
    }
}

fn endpoint_headers(endpoint: &Value) -> Result<Vec<(String, String)>, String> {
    let Some(headers) = endpoint.get("headers") else {
        return Ok(Vec::new());
    };
    let Some(object) = headers.as_object() else {
        return Err("headers must be an object of string values".into());
    };
    let mut out = Vec::with_capacity(object.len());
    for (key, value) in object {
        let Some(value) = value.as_str() else {
            return Err(format!("headers.{key} must be a string"));
        };
        out.push((key.clone(), value.to_string()));
    }
    Ok(out)
}

fn doctor_atof_probe_payload() -> Result<String, String> {
    let event = Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .uuid(Uuid::now_v7())
            .name("nemo_relay.doctor.atof_probe")
            .data(json!({"doctor": true}))
            .metadata(json!({"source": "nemo-relay doctor"}))
            .build(),
        None,
        None,
    ));
    event
        .try_to_json_value()
        .and_then(|value| serde_json::to_string(&value))
        .map_err(|error| error.to_string())
}

async fn probe_atof_http_post(
    url: &str,
    headers: Vec<(String, String)>,
    payload: String,
    timeout_duration: Duration,
    index: usize,
) -> Check {
    probe_atof_http_upload(url, headers, payload, timeout_duration, index, "http_post").await
}

async fn probe_atof_ndjson(
    url: &str,
    headers: Vec<(String, String)>,
    payload: String,
    timeout_duration: Duration,
    index: usize,
) -> Check {
    probe_atof_http_upload(url, headers, payload, timeout_duration, index, "ndjson").await
}

async fn probe_atof_http_upload(
    url: &str,
    headers: Vec<(String, String)>,
    payload: String,
    timeout_duration: Duration,
    index: usize,
    transport: &str,
) -> Check {
    let client = match reqwest::Client::builder().timeout(timeout_duration).build() {
        Ok(client) => client,
        Err(err) => {
            return Check {
                name: "ATOF endpoint",
                status: Status::Fail,
                details: format!(
                    "endpoints[{index}] {transport} {url}: could not build client: {err}"
                ),
            };
        }
    };
    let mut request = client
        .post(url)
        .header(reqwest::header::CONTENT_TYPE, "application/x-ndjson")
        .body(format!("{payload}\n"));
    for (key, value) in headers {
        request = request.header(key, value);
    }
    match request.send().await {
        Ok(response) if response.status().is_success() => Check {
            name: "ATOF endpoint",
            status: Status::Pass,
            details: format!(
                "endpoints[{index}] {transport} {url} (HTTP {})",
                response.status()
            ),
        },
        Ok(response) => Check {
            name: "ATOF endpoint",
            status: Status::Fail,
            details: format!(
                "endpoints[{index}] {transport} {url} (HTTP {})",
                response.status()
            ),
        },
        Err(err) => Check {
            name: "ATOF endpoint",
            status: Status::Fail,
            details: format!("endpoints[{index}] {transport} {url}: {err}"),
        },
    }
}

async fn probe_atof_websocket(
    url: &str,
    headers: Vec<(String, String)>,
    payload: String,
    timeout_duration: Duration,
    index: usize,
) -> Check {
    match reqwest::Url::parse(url) {
        Ok(parsed) if matches!(parsed.scheme(), "ws" | "wss") => {}
        Ok(_) => {
            return Check {
                name: "ATOF endpoint",
                status: Status::Fail,
                details: format!(
                    "endpoints[{index}] websocket {url}: invalid scheme (must be ws or wss)"
                ),
            };
        }
        Err(err) => {
            return Check {
                name: "ATOF endpoint",
                status: Status::Fail,
                details: format!("endpoints[{index}] websocket {url}: {err}"),
            };
        }
    }
    let mut request = match url.into_client_request() {
        Ok(request) => request,
        Err(err) => {
            return Check {
                name: "ATOF endpoint",
                status: Status::Fail,
                details: format!("endpoints[{index}] websocket {url}: {err}"),
            };
        }
    };
    for (key, value) in headers {
        let name = match tokio_tungstenite::tungstenite::http::header::HeaderName::from_bytes(
            key.as_bytes(),
        ) {
            Ok(name) => name,
            Err(err) => {
                return Check {
                    name: "ATOF endpoint",
                    status: Status::Fail,
                    details: format!("endpoints[{index}] websocket {url}: {err}"),
                };
            }
        };
        let value =
            match tokio_tungstenite::tungstenite::http::header::HeaderValue::from_str(&value) {
                Ok(value) => value,
                Err(err) => {
                    return Check {
                        name: "ATOF endpoint",
                        status: Status::Fail,
                        details: format!("endpoints[{index}] websocket {url}: {err}"),
                    };
                }
            };
        request.headers_mut().insert(name, value);
    }
    match timeout(timeout_duration, tokio_tungstenite::connect_async(request)).await {
        Ok(Ok((mut socket, _))) => {
            let send = timeout(
                timeout_duration,
                socket.send(tokio_tungstenite::tungstenite::Message::Text(
                    payload.into(),
                )),
            )
            .await;
            let _ = timeout(timeout_duration, socket.close(None)).await;
            match send {
                Ok(Ok(())) => Check {
                    name: "ATOF endpoint",
                    status: Status::Pass,
                    details: format!("endpoints[{index}] websocket {url}"),
                },
                Ok(Err(err)) => Check {
                    name: "ATOF endpoint",
                    status: Status::Fail,
                    details: format!("endpoints[{index}] websocket {url}: {err}"),
                },
                Err(_) => Check {
                    name: "ATOF endpoint",
                    status: Status::Fail,
                    details: format!(
                        "endpoints[{index}] websocket {url}: timed out sending probe payload"
                    ),
                },
            }
        }
        Ok(Err(err)) => Check {
            name: "ATOF endpoint",
            status: Status::Fail,
            details: format!("endpoints[{index}] websocket {url}: {err}"),
        },
        Err(_) => Check {
            name: "ATOF endpoint",
            status: Status::Fail,
            details: format!("endpoints[{index}] websocket {url}: timed out"),
        },
    }
}

fn check_directory(name: &'static str, path: &Path) -> Check {
    match check_dir_writable(path) {
        Ok(()) => Check {
            name,
            status: Status::Pass,
            details: format!("{} (appears writable)", path.display()),
        },
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Check {
            name,
            status: Status::Warn,
            details: format!("{}: not present; runtime will create it", path.display()),
        },
        Err(err) => Check {
            name,
            status: Status::Fail,
            details: format!("{}: {err}", path.display()),
        },
    }
}

fn check_dir_writable(dir: &Path) -> Result<(), std::io::Error> {
    let metadata = std::fs::metadata(dir)?;
    if !metadata.is_dir() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "path is not a directory",
        ));
    }
    if metadata.permissions().readonly() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "directory is read-only",
        ));
    }
    Ok(())
}

async fn probe_http_named(name: &'static str, url: &str) -> Check {
    let client = match reqwest::Client::builder().timeout(NETWORK_TIMEOUT).build() {
        Ok(c) => c,
        Err(err) => {
            return Check {
                name,
                status: Status::Fail,
                details: format!("could not build HTTP client: {err}"),
            };
        }
    };
    match client.get(url).send().await {
        Ok(resp) => Check {
            name,
            status: if resp.status().is_success() || resp.status().is_redirection() {
                Status::Pass
            } else {
                Status::Warn
            },
            details: format!("{} (HTTP {})", url, resp.status().as_u16()),
        },
        Err(err) => Check {
            name,
            status: Status::Fail,
            details: format!("{url}: {err}"),
        },
    }
}

fn collect_completions(home: Option<&std::path::Path>) -> Vec<Check> {
    let mut checks = Vec::new();
    let shell = std::env::var("SHELL").ok().and_then(|s| {
        std::path::Path::new(&s)
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
    });
    let Some(shell_name) = shell else {
        checks.push(Check {
            name: "Completions",
            status: Status::Info,
            details: "no $SHELL set; cannot infer install location".into(),
        });
        return checks;
    };
    let Some(home) = home else {
        checks.push(Check {
            name: "Completions",
            status: Status::Info,
            details: format!("$SHELL={shell_name}; could not resolve home dir"),
        });
        return checks;
    };
    let likely_path = match shell_name.as_str() {
        "zsh" => Some(home.join(".zfunc").join("_nemo-relay")),
        "bash" => Some(home.join(".bash_completion.d").join("nemo-relay")),
        "fish" => Some(
            home.join(".config")
                .join("fish")
                .join("completions")
                .join("nemo-relay.fish"),
        ),
        _ => None,
    };
    match likely_path {
        Some(path) if path.exists() => checks.push(Check {
            name: "Completions",
            status: Status::Pass,
            details: format!("{shell_name}: {}", path.display()),
        }),
        Some(path) => checks.push(Check {
            name: "Completions",
            status: Status::Info,
            details: format!(
                "{shell_name}: not installed (run `nemo-relay completions {shell_name} > {}`)",
                path.display()
            ),
        }),
        None => checks.push(Check {
            name: "Completions",
            status: Status::Info,
            details: format!("{shell_name}: no known completion path; run `nemo-relay completions <shell>` to generate"),
        }),
    }
    checks
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

/// Aggregate exit code: 1 if any check is Fail, 0 otherwise. Warnings do not fail.
pub(crate) fn exit_code(report: &DoctorReport) -> u8 {
    let any_fail = report
        .observability
        .iter()
        .chain(report.completions.iter())
        .any(|c| matches!(c.status, Status::Fail))
        || report
            .agents
            .iter()
            .any(|agent| matches!(agent.status, Status::Fail))
        || report.host_plugins.iter().any(|plugin| !plugin.ok())
        || matches!(report.configuration.workspace.status, Status::Fail)
        || matches!(report.configuration.global.status, Status::Fail)
        || matches!(report.configuration.system.status, Status::Fail)
        || matches!(report.configuration.plugin_resolution.status, Status::Fail)
        || matches!(report.configuration.resolution.status, Status::Fail);
    u8::from(any_fail)
}

// Returns true if any check in the report carries a `Warn` status. Used by the human footer to
// distinguish a fully-green report from one where everything passed but some checks issued
// warnings — both exit 0, but the wording shouldn't.
fn report_has_warn(report: &DoctorReport) -> bool {
    report
        .observability
        .iter()
        .chain(report.completions.iter())
        .any(|c| matches!(c.status, Status::Warn))
        || report
            .agents
            .iter()
            .any(|agent| matches!(agent.status, Status::Warn))
        || report.host_plugins.iter().any(|plugin| !plugin.ok())
        || matches!(report.configuration.workspace.status, Status::Warn)
        || matches!(report.configuration.global.status, Status::Warn)
        || matches!(report.configuration.system.status, Status::Warn)
        || matches!(report.configuration.plugin_resolution.status, Status::Warn)
        || matches!(report.configuration.resolution.status, Status::Warn)
}

/// Renders the doctor report in the fixed human-readable layout the design doc shows. Sections
/// stay in the same order across runs so users can diff across machines. The banner header lives
/// in `crate::banner::print_doctor_header` (called from `run_doctor` before this renders) so the
/// pure formatter stays banner-free for tests.
pub(crate) fn format_human(report: &DoctorReport) -> String {
    let mut out = String::new();
    format_human_header(&mut out, report);
    format_human_environment(&mut out, report);
    format_human_configuration(&mut out, report);
    format_human_plugin_configuration(&mut out, report);
    format_human_agents(&mut out, report);
    format_human_host_plugins(&mut out, report);
    format_human_checks(&mut out, "Observability", &report.observability);
    format_human_completion_checks(&mut out, &report.completions);
    format_human_conclusion(&mut out, report);
    out
}

fn format_human_header(out: &mut String, report: &DoctorReport) {
    out.push_str(&format!("\n  NeMo Relay {}\n", report.binary_version));
    out.push_str("  ─────────────────────────────────────────────\n");
    if let Some(agent) = &report.target_agent {
        out.push_str(&format!("  Target agent  {agent}\n\n"));
    }
}

fn format_human_environment(out: &mut String, report: &DoctorReport) {
    out.push_str("  Environment\n");
    out.push_str(&format!(
        "    OS         {}\n",
        report.environment.os.trim()
    ));
    out.push_str(&format!("    Arch       {}\n", report.environment.arch));
    if let Some(shell) = &report.environment.shell {
        out.push_str(&format!("    Shell      {shell}\n"));
    }
    out.push('\n');
}

fn format_human_configuration(out: &mut String, report: &DoctorReport) {
    out.push_str("  Configuration\n");
    out.push_str(&format!(
        "    Workspace  {}\n",
        format_layer(&report.configuration.workspace)
    ));
    out.push_str(&format!(
        "    Global     {}\n",
        format_layer(&report.configuration.global)
    ));
    out.push_str(&format!(
        "    System     {}\n",
        format_layer(&report.configuration.system)
    ));
    if !matches!(report.configuration.resolution.status, Status::Pass) {
        out.push_str(&format!(
            "    Resolution {} {}\n",
            format_status(report.configuration.resolution.status),
            report.configuration.resolution.details
        ));
    }
    if !report.configuration.configured_agents.is_empty() {
        out.push_str(&format!(
            "    Agents     {}\n",
            report.configuration.configured_agents.join(", ")
        ));
    }
    out.push('\n');
}

fn format_human_plugin_configuration(out: &mut String, report: &DoctorReport) {
    out.push_str("  Plugin configuration\n");
    for plugin in &report.configuration.dynamic_plugins {
        let config_suffix = if matches!(
            plugin.host_config_status,
            DynamicPluginHostConfigStatus::Present
        ) {
            "; host config"
        } else {
            ""
        };
        out.push_str(&format!(
            "    Dynamic    {} ({}){}\n",
            plugin.plugin_id, plugin.manifest_ref, config_suffix
        ));
    }
    if !report.configuration.plugin_configs.is_empty() {
        for (index, layer) in report.configuration.plugin_configs.iter().enumerate() {
            let label = if index == 0 { "Plugin files" } else { "" };
            out.push_str(&format!("    {label:<13}{}\n", format_layer(layer)));
        }
    }
    out.push_str(&format!(
        "    Plugins    {} {}\n",
        format_status(report.configuration.plugin_resolution.status),
        report.configuration.plugin_resolution.details
    ));
    for plugin in &report.configuration.dynamic_plugins {
        for check in [
            dynamic_plugin_reference_check(plugin),
            dynamic_plugin_host_config_check(plugin),
        ] {
            out.push_str(&format!(
                "    Dynamic    {} {}\n",
                format_status(check.status),
                check.details
            ));
        }
    }
    out.push('\n');
}

fn format_human_agents(out: &mut String, report: &DoctorReport) {
    out.push_str("  Agents detected\n");
    for agent in &report.agents {
        let status = format_status(agent.status);
        match &agent.path {
            Some(path) => {
                let version = agent.version.as_deref().unwrap_or("(unknown version)");
                out.push_str(&format!(
                    "    {}  {:<8} {}\n          command  {}\n          path     {}\n          {}\n",
                    status,
                    agent.name,
                    version,
                    agent.command,
                    path.display(),
                    agent.annotation
                ));
            }
            None => {
                out.push_str(&format!(
                    "    {}  {:<8} not on $PATH\n          command  {}\n          {}\n",
                    status, agent.name, agent.command, agent.annotation
                ));
            }
        }
    }
    out.push('\n');
}

fn format_human_host_plugins(out: &mut String, report: &DoctorReport) {
    out.push_str("  Persistent integrations\n");
    if report.host_plugins.is_empty() {
        out.push_str("    ·  none installed; run `nemo-relay install <host>` to enable one\n");
    } else {
        for plugin in &report.host_plugins {
            out.push_str(&format!(
                "    {}  {}\n",
                if plugin.ok() { "✓" } else { "✗" },
                plugin.host
            ));
            for check in &plugin.checks {
                out.push_str(&format!(
                    "          {}  {}: {}\n",
                    if check.ok { "✓" } else { "✗" },
                    check.name,
                    check.details
                ));
            }
            if !plugin.ok() {
                out.push_str(&format!("          repair: {}\n", plugin.remediation));
            }
        }
    }
    out.push('\n');
}

fn format_human_checks(out: &mut String, title: &str, checks: &[Check]) {
    out.push_str(&format!("  {title}\n"));
    for check in checks {
        out.push_str(&format!("    {:<22}  {}\n", check.name, check.details));
    }
    out.push('\n');
}

fn format_human_completion_checks(out: &mut String, checks: &[Check]) {
    out.push_str("  Completions\n");
    for check in checks {
        out.push_str(&format!("    {}\n", check.details));
    }
    out.push('\n');
}

fn format_human_conclusion(out: &mut String, report: &DoctorReport) {
    if exit_code(report) == 0 {
        if report_has_warn(report) {
            out.push_str("  All checks passed, but some issued warnings; see details above.\n");
        } else {
            out.push_str("  All checks passed.\n");
        }
    } else {
        out.push_str("  Some checks FAILED; see details above.\n");
    }
}

fn format_layer(layer: &ConfigLayer) -> String {
    let active = if layer.active { " (loaded)" } else { "" };
    format!("{}   {}{}", layer.path.display(), layer.details, active)
}

fn format_status(status: Status) -> &'static str {
    match status {
        Status::Pass => "✓",
        Status::Warn => "!",
        Status::Fail => "✗",
        Status::Info => "·",
    }
}

/// Renders the doctor report as machine-readable JSON. Versioned via `schema_version` so
/// downstream consumers (CI dashboards, eval harnesses) can detect schema changes.
pub(crate) fn format_json(report: &DoctorReport) -> Result<String, CliError> {
    serde_json::to_string_pretty(report)
        .map_err(|err| CliError::Config(format!("could not serialize doctor report: {err}")))
}

/// Runs `agents` — a thin wrapper over `collect_agents` that emits only the agent list. Shares
/// the same JSON schema as `doctor.agents` for consistency.
pub(crate) async fn agents_report() -> Vec<AgentInfo> {
    let resolved = resolve_server_config(&ServerArgs::default()).unwrap_or_default();
    collect_agents(None, &resolved).await
}

/// Renders the agents listing in human form.
pub(crate) fn format_agents_human(agents: &[AgentInfo]) -> String {
    let mut out = String::new();
    out.push_str("\n  Supported\n");
    for agent in agents {
        out.push_str(&format!("    {}\n", agent.name));
    }
    out.push('\n');
    out.push_str("  Detected on this machine\n");
    let detected: Vec<&AgentInfo> = agents.iter().filter(|a| a.path.is_some()).collect();
    if detected.is_empty() {
        out.push_str("    (none)\n");
    } else {
        for agent in detected {
            let version = agent.version.as_deref().unwrap_or("(unknown version)");
            let path = agent
                .path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_default();
            out.push_str(&format!(
                "    {}  {:<8} {}\n               {}\n               {}\n",
                format_status(agent.status),
                agent.name,
                version,
                path,
                agent.annotation
            ));
        }
    }
    out.push('\n');
    out
}

/// Renders the agents listing as JSON. Same shape as `DoctorReport.agents`.
pub(crate) fn format_agents_json(agents: &[AgentInfo]) -> Result<String, CliError> {
    serde_json::to_string_pretty(agents)
        .map_err(|err| CliError::Config(format!("could not serialize agents report: {err}")))
}

/// Top-level entry point invoked by `nemo-relay doctor`. Emits to stdout and returns the
/// appropriate process exit code (0 on pass-or-warn, 1 on any failure).
pub(crate) async fn run_doctor(
    target_agent: Option<CodingAgent>,
    json: bool,
) -> Result<std::process::ExitCode, CliError> {
    let report = collect_report(target_agent).await?;
    if json {
        print!("{}", format_json(&report)?);
    } else {
        // Banner first, then the static report. JSON mode skips both so callers parsing the
        // output don't have to strip ANSI/decorations.
        crate::banner::print_doctor_header();
        print!("{}", format_human(&report));
    }
    match exit_code(&report) {
        0 => Ok(std::process::ExitCode::SUCCESS),
        _ => Ok(std::process::ExitCode::FAILURE),
    }
}

/// Top-level entry point invoked by `nemo-relay agents`. Always exits 0; the data drives caller
/// decisions (e.g., CI gating on JSON output).
pub(crate) async fn run_agents(json: bool) -> Result<std::process::ExitCode, CliError> {
    let agents = agents_report().await;
    let output = if json {
        format_agents_json(&agents)?
    } else {
        format_agents_human(&agents)
    };
    print!("{output}");
    Ok(std::process::ExitCode::SUCCESS)
}

// `ResolvedConfig` defaults to "no settings" when no config file is present. Trait kept here
// so `unwrap_or_default()` works on the resolved config without leaking optionality into the
// rest of the doctor surface. The Default impl on `ResolvedConfig` is provided by its derive.
const _: fn() = || {
    let _: ResolvedConfig = ResolvedConfig::default();
};

#[cfg(test)]
#[path = "../../tests/coverage/shared/doctor_tests.rs"]
mod tests;
