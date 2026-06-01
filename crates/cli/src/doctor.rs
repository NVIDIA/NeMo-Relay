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
use std::process::Stdio;
use std::time::Duration;

use nemo_relay::observability::plugin_component::OBSERVABILITY_PLUGIN_KIND;
use nemo_relay::plugin::{DiagnosticLevel, PluginConfig, validate_plugin_config};
use nemo_relay_adaptive::plugin_component::register_adaptive_component;
use serde::Serialize;
use serde_json::Value;
use tokio::time::timeout;

use crate::config::{
    AgentConfigs, CodingAgent, GatewayConfig, ResolvedConfig, ServerArgs, resolve_server_config,
};
use crate::error::CliError;

const NETWORK_TIMEOUT: Duration = Duration::from_secs(2);

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
    pub resolution: Check,
    pub default_agent: Option<String>,
    pub configured_agents: Vec<String>,
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
        ),
        agents: collect_agents(target_agent, &resolved).await,
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
) -> ConfigurationInfo {
    let workspace_path = cwd
        .map(|p| p.join(".nemo-relay").join("config.toml"))
        .unwrap_or_else(|| PathBuf::from(".nemo-relay/config.toml"));
    // Use the same XDG-aware resolver the config loader uses, so doctor reports the path the
    // runtime would actually read instead of a hard-coded `$HOME/.config/nemo-relay`.
    let global_path = crate::config::user_config_dir()
        .map(|dir| dir.join("config.toml"))
        .or_else(|| home.map(|h| h.join(".config").join("nemo-relay").join("config.toml")))
        .unwrap_or_else(|| PathBuf::from("~/.config/nemo-relay/config.toml"));
    let system_path = PathBuf::from("/etc/nemo-relay/config.toml");

    ConfigurationInfo {
        workspace: layer_status(&workspace_path),
        global: layer_status(&global_path),
        system: layer_status(&system_path),
        resolution,
        // `default_agent` is reserved in the design for Phase 2 dispatch; not currently parsed
        // out of FileConfig. Doctor reports `None` until that lands.
        default_agent: None,
        configured_agents,
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

async fn collect_agents(
    target_agent: Option<CodingAgent>,
    resolved: &ResolvedConfig,
) -> Vec<AgentInfo> {
    let supported = [
        (CodingAgent::ClaudeCode, "claude", "claude"),
        (CodingAgent::Codex, "codex", "codex"),
        (CodingAgent::Cursor, "cursor", "cursor-agent"),
        (CodingAgent::Hermes, "hermes", "hermes"),
    ];
    let mut out = Vec::with_capacity(supported.len());
    for (agent, display_name, default_exec) in supported {
        if target_agent.is_some_and(|target| target != agent) {
            continue;
        }
        let configured = agent_configured(agent, &resolved.agents);
        let target_requested = target_agent == Some(agent);
        let command = agent_command(agent, &resolved.agents, default_exec);
        let exec = command_executable(&command);
        let path = which_command(exec);
        let version = match &path {
            Some(p) => probe_version(p).await,
            None => None,
        };
        let mut status = agent_command_status(path.as_deref(), configured, target_requested);
        let (hook_status, hook_details) =
            hook_status(agent, &resolved.agents, configured || target_requested);
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
        out.push(AgentInfo {
            name: display_name,
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

fn which_on_path(exec: &str) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    std::env::split_paths(&path_var)
        .map(|dir| dir.join(exec))
        .find(|candidate| candidate.is_file())
}

fn which_command(exec: &str) -> Option<PathBuf> {
    let candidate = Path::new(exec);
    if candidate.components().count() > 1 || candidate.is_absolute() {
        return candidate.is_file().then(|| candidate.to_path_buf());
    }
    which_on_path(exec)
}

fn command_executable(command: &str) -> &str {
    command.split_whitespace().next().unwrap_or(command)
}

fn agent_command(agent: CodingAgent, agents: &AgentConfigs, default_exec: &str) -> String {
    configured_agent_command(agent, agents)
        .cloned()
        .unwrap_or_else(|| default_exec.to_string())
}

fn configured_agent_command(agent: CodingAgent, agents: &AgentConfigs) -> Option<&String> {
    match agent {
        CodingAgent::ClaudeCode => agents.claude.command.as_ref(),
        CodingAgent::Codex => agents.codex.command.as_ref(),
        CodingAgent::Cursor => agents.cursor.command.as_ref(),
        CodingAgent::Hermes => agents.hermes.command.as_ref(),
    }
}

fn agent_configured(agent: CodingAgent, agents: &AgentConfigs) -> bool {
    configured_agent_command(agent, agents).is_some()
        || (matches!(agent, CodingAgent::Hermes) && agents.hermes.hooks_path.is_some())
}

fn configured_agent_names(agents: &AgentConfigs) -> Vec<String> {
    [
        (CodingAgent::ClaudeCode, "claude"),
        (CodingAgent::Codex, "codex"),
        (CodingAgent::Cursor, "cursor"),
        (CodingAgent::Hermes, "hermes"),
    ]
    .into_iter()
    .filter_map(|(agent, name)| agent_configured(agent, agents).then_some(name.to_string()))
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

fn hook_status(
    agent: CodingAgent,
    agents: &AgentConfigs,
    readiness_required: bool,
) -> (Status, String) {
    match agent {
        CodingAgent::ClaudeCode | CodingAgent::Codex => {
            (Status::Pass, "hooks: injected during run".into())
        }
        CodingAgent::Cursor if agents.cursor.patch_restore_hooks => {
            (Status::Pass, "hooks: patched during run".into())
        }
        CodingAgent::Cursor => hook_file_status(
            cursor_hooks_path(),
            CodingAgent::Cursor,
            readiness_required,
            "hooks: user-managed",
        ),
        CodingAgent::Hermes => match agents.hermes.hooks_path.as_deref() {
            Some(path) => hook_file_status(
                Ok(path.to_path_buf()),
                CodingAgent::Hermes,
                readiness_required,
                "hooks",
            ),
            None if readiness_required => (
                Status::Fail,
                "hooks: not installed; run `nemo-relay config hermes`".into(),
            ),
            None => (Status::Info, "hooks: not configured".into()),
        },
    }
}

fn hook_file_status(
    path: Result<PathBuf, CliError>,
    agent: CodingAgent,
    readiness_required: bool,
    label: &str,
) -> (Status, String) {
    let path = match path {
        Ok(path) => path,
        Err(err) => {
            return (
                Status::Fail,
                format!("{label}: could not resolve path: {err}"),
            );
        }
    };
    match std::fs::read_to_string(&path) {
        Ok(raw) if matches!(agent, CodingAgent::Cursor) => {
            cursor_hook_file_status(&raw, &path, readiness_required, label)
        }
        Ok(raw) if raw.contains(&format!("hook-forward {}", agent.as_arg())) => (
            Status::Pass,
            format!("{label}: installed at {}", path.display()),
        ),
        Ok(_) if readiness_required => (
            Status::Fail,
            format!("{label}: missing NeMo Relay hook in {}", path.display()),
        ),
        Ok(_) => (
            Status::Info,
            format!("{label}: no NeMo Relay hook in {}", path.display()),
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound && readiness_required => {
            (Status::Fail, format!("{label}: missing {}", path.display()))
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            (Status::Info, format!("{label}: missing {}", path.display()))
        }
        Err(error) => (
            Status::Fail,
            format!("{label}: could not read {}: {error}", path.display()),
        ),
    }
}

fn cursor_hook_file_status(
    raw: &str,
    path: &Path,
    readiness_required: bool,
    label: &str,
) -> (Status, String) {
    let has_nemo_hook = raw.contains("hook-forward cursor");
    if !has_nemo_hook {
        if readiness_required {
            return (
                Status::Fail,
                format!("{label}: missing NeMo Relay hook in {}", path.display()),
            );
        }
        return (
            Status::Info,
            format!("{label}: no NeMo Relay hook in {}", path.display()),
        );
    }

    let parsed: Value = match serde_json::from_str(raw) {
        Ok(parsed) => parsed,
        Err(err) => {
            return (
                Status::Fail,
                format!(
                    "{label}: invalid Cursor hooks JSON in {}: {err}",
                    path.display()
                ),
            );
        }
    };

    if parsed.get("version").and_then(Value::as_u64) != Some(1) {
        return (
            Status::Fail,
            format!(
                "{label}: Cursor hook file {} must set top-level `version` to 1",
                path.display()
            ),
        );
    }

    let Some(hooks) = parsed.get("hooks").and_then(Value::as_object) else {
        return (
            Status::Fail,
            format!(
                "{label}: Cursor hook file {} has no hooks object",
                path.display()
            ),
        );
    };
    let has_direct_nemo_hook = hooks.values().any(cursor_event_has_direct_nemo_hook);
    if has_nested_hook_group(&parsed) {
        return (
            Status::Fail,
            format!(
                "{label}: Cursor hook file {} uses nested hook groups; Cursor CLI requires direct command entries",
                path.display()
            ),
        );
    }
    if !has_direct_nemo_hook {
        return (
            Status::Fail,
            format!(
                "{label}: Cursor hook file {} has no direct NeMo Relay command entries",
                path.display()
            ),
        );
    }

    (
        Status::Pass,
        format!("{label}: installed at {}", path.display()),
    )
}

fn cursor_event_has_direct_nemo_hook(event_hooks: &Value) -> bool {
    event_hooks.as_array().is_some_and(|entries| {
        entries.iter().any(|entry| {
            entry
                .get("command")
                .and_then(Value::as_str)
                .is_some_and(|command| command.contains("hook-forward cursor"))
        })
    })
}

fn has_nested_hook_group(value: &Value) -> bool {
    match value {
        Value::Object(object) => {
            let nested_here = object.get("hooks").is_some_and(Value::is_array);
            nested_here || object.values().any(has_nested_hook_group)
        }
        Value::Array(items) => items.iter().any(has_nested_hook_group),
        _ => false,
    }
}

fn cursor_hooks_path() -> Result<PathBuf, CliError> {
    let cwd = std::env::current_dir()?;
    let project = cwd
        .ancestors()
        .find(|ancestor| ancestor.join(".cursor").is_dir())
        .unwrap_or(cwd.as_path());
    Ok(project.join(".cursor/hooks.json"))
}

async fn probe_version(binary: &Path) -> Option<String> {
    // Spawn `<binary> --version` and read the first line of stdout. Bounded by the network
    // timeout (re-used as a generic short timeout) so a misbehaving binary doesn't hang doctor.
    let mut cmd = tokio::process::Command::new(binary);
    cmd.arg("--version")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .stdin(Stdio::null())
        // Ensure the child gets killed if our future is dropped on timeout. Without this a
        // misbehaving agent binary that exceeds NETWORK_TIMEOUT would leak as an orphan
        // process for the lifetime of the doctor invocation (and beyond).
        .kill_on_drop(true);
    let child = cmd.spawn().ok()?;
    let output = timeout(NETWORK_TIMEOUT, child.wait_with_output())
        .await
        .ok()?
        .ok()?;
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
            name: "Plugins",
            status: Status::Info,
            details: "plugin config not configured".into(),
        });
        return checks;
    };
    let source = gateway
        .plugin_config_source
        .as_deref()
        .unwrap_or("plugin config");

    let plugin_config = match serde_json::from_value::<PluginConfig>(plugin_value.clone()) {
        Ok(config) => config,
        Err(err) => {
            checks.push(Check {
                name: "Plugins",
                status: Status::Fail,
                details: format!("invalid plugin config from {source}: {err}"),
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
    let report = validate_plugin_config(&plugin_config);
    if report.diagnostics.is_empty() {
        checks.push(Check {
            name: "Plugins",
            status: Status::Pass,
            details: format!("validation passed from {source}"),
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
                details: format!("{source}: {}: {}", diagnostic.code, diagnostic.message),
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
    crate::tls::install_rustls_crypto_provider();
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
        || matches!(report.configuration.workspace.status, Status::Fail)
        || matches!(report.configuration.global.status, Status::Fail)
        || matches!(report.configuration.system.status, Status::Fail)
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
        || matches!(report.configuration.workspace.status, Status::Warn)
        || matches!(report.configuration.global.status, Status::Warn)
        || matches!(report.configuration.system.status, Status::Warn)
        || matches!(report.configuration.resolution.status, Status::Warn)
}

/// Renders the doctor report in the fixed human-readable layout the design doc shows. Sections
/// stay in the same order across runs so users can diff across machines. The banner header lives
/// in `crate::banner::print_doctor_header` (called from `run_doctor` before this renders) so the
/// pure formatter stays banner-free for tests.
pub(crate) fn format_human(report: &DoctorReport) -> String {
    let mut out = String::new();
    out.push_str(&format!("\n  NeMo Relay {}\n", report.binary_version));
    out.push_str("  ─────────────────────────────────────────────\n");
    if let Some(agent) = &report.target_agent {
        out.push_str(&format!("  Target agent  {agent}\n\n"));
    }
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

    out.push_str("  Observability\n");
    for check in &report.observability {
        out.push_str(&format!("    {:<22}  {}\n", check.name, check.details));
    }
    out.push('\n');

    out.push_str("  Completions\n");
    for check in &report.completions {
        out.push_str(&format!("    {}\n", check.details));
    }
    out.push('\n');

    if exit_code(report) == 0 {
        if report_has_warn(report) {
            out.push_str("  All checks passed, but some issued warnings; see details above.\n");
        } else {
            out.push_str("  All checks passed.\n");
        }
    } else {
        out.push_str("  Some checks FAILED; see details above.\n");
    }
    out
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
#[path = "../tests/coverage/doctor_tests.rs"]
mod tests;
