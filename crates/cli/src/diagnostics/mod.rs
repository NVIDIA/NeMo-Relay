// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `nemo-relay doctor` — environment + config + agent + observability health check.
//!
//! Split into three layers so the data path can be unit-tested without real I/O:
//!
//! - `collect_report()` does the I/O (env probes, $PATH scans, network checks, fs writability).
//! - `DoctorReport` is the resulting pure data shape.
//! - `format_human(&report)` / `format_json(&report)` render the report.

mod environment;
mod model;
mod probes;
mod render;

use environment::collect_environment;
pub(crate) use model::*;
use probes::*;
use render::*;

use std::path::{Path, PathBuf};
use std::time::Duration;

use futures_util::{SinkExt, future::join_all};
use nemo_relay::api::event::{BaseEvent, Event, MarkEvent};
use nemo_relay::codec::model_pricing::{PricingCatalog, PricingConfig, PricingSourceConfig};
use nemo_relay::observability::otel::resolve_http_trace_endpoint;
use nemo_relay::observability::plugin_component::OBSERVABILITY_PLUGIN_KIND;
use nemo_relay::plugin::{DiagnosticLevel, PluginConfig, validate_plugin_config};
use nemo_relay_adaptive::plugin_component::ADAPTIVE_PLUGIN_KIND;
use nemo_relay_adaptive::{ResponseCacheConfig, response_cache};
use serde_json::{Value, json};
use tokio::time::timeout;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use uuid::Uuid;

use crate::agents::CodingAgent;
use crate::configuration::{
    AgentConfigs, DynamicPluginHostConfigStatus, GatewayConfig, ResolvedConfig,
    diagnostic_plugin_config_paths, effective_plugin_toml_sources, resolve_server_config,
};
use crate::error::CliError;
use crate::server::{GatewayOverrides, register_and_validate_plugin_components};

const NETWORK_TIMEOUT: Duration = Duration::from_secs(2);
const PRICING_PLUGIN_KIND: &str = "pricing";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DoctorProbeMode {
    Live,
    Offline,
}

impl DoctorProbeMode {
    pub(crate) fn from_offline_flag(offline: bool) -> Self {
        if offline { Self::Offline } else { Self::Live }
    }

    fn is_offline(self) -> bool {
        matches!(self, Self::Offline)
    }
}

struct PluginConfigurationDiagnostics {
    sources: Vec<PathBuf>,
    error: Option<String>,
    resolution: Check,
}

/// Drives all checks and produces a single `DoctorReport`. Network probes are bounded by a
/// short timeout so the command always returns quickly. Filesystem checks short-circuit on
/// the first missing directory.
pub(crate) async fn collect_report(
    target_agent: Option<CodingAgent>,
    probe_mode: DoctorProbeMode,
    gateway_overrides: &GatewayOverrides,
) -> Result<DoctorReport, CliError> {
    let (resolved, resolution) = match resolve_server_config(gateway_overrides) {
        Ok(resolved) => (
            resolved,
            Check {
                name: "Resolution",
                status: Status::Pass,
                details: "valid".into(),
            },
        ),
        Err(err) => {
            log::warn!(
                target: "nemo_relay.diagnostics",
                event = "diagnostic_probe_failed",
                probe = "configuration",
                error_kind = err.log_kind();
                "Diagnostic probe failed"
            );
            (
                ResolvedConfig::default(),
                Check {
                    name: "Resolution",
                    status: Status::Fail,
                    details: format!(
                        "could not resolve merged config: {err}; repair or recreate the named configuration file, or rerun the setup or installer that manages it"
                    ),
                },
            )
        }
    };
    let cwd = std::env::current_dir().ok();
    let home = home_dir();
    let configured_agents = configured_agent_names(&resolved.agents);
    let (plugin_sources, plugin_error) = match effective_plugin_toml_sources(
        gateway_overrides.config.as_ref(),
        gateway_overrides.plugin_config_path.as_ref(),
    ) {
        Ok(sources) => (sources, None),
        Err(error) => {
            log::warn!(
                target: "nemo_relay.diagnostics",
                event = "diagnostic_probe_failed",
                probe = "plugin_configuration",
                error_kind = error.log_kind();
                "Diagnostic probe failed"
            );
            (Vec::new(), Some(error.to_string()))
        }
    };
    let plugin_resolution =
        plugin_resolution_check(&resolved, &resolution, plugin_error.as_deref());
    let plugin_diagnostics = PluginConfigurationDiagnostics {
        sources: plugin_sources,
        error: plugin_error,
        resolution: plugin_resolution,
    };

    Ok(DoctorReport {
        schema_version: 2,
        binary_version: env!("CARGO_PKG_VERSION"),
        target_agent: target_agent.map(|agent| agent.as_arg().to_string()),
        environment: collect_environment(),
        configuration: collect_configuration(
            cwd.as_deref(),
            home.as_deref(),
            &resolved,
            gateway_overrides,
            resolution,
            configured_agents,
            &plugin_diagnostics,
        ),
        agents: collect_agents(target_agent, probe_mode, &resolved).await,
        host_plugins: crate::agents::collect_default_integration_readiness(),
        observability: collect_observability(&resolved.gateway, probe_mode).await,
        completions: collect_completions(home.as_deref()),
    })
}

fn collect_configuration(
    cwd: Option<&Path>,
    home: Option<&Path>,
    resolved: &ResolvedConfig,
    gateway_overrides: &GatewayOverrides,
    resolution: Check,
    configured_agents: Vec<String>,
    plugin_diagnostics: &PluginConfigurationDiagnostics,
) -> ConfigurationInfo {
    let explicit_config = gateway_overrides.config.is_some();
    // Use the same XDG-aware resolver the config loader uses, so doctor reports the path the
    // runtime would actually read instead of a hard-coded `$HOME/.config/nemo-relay`.
    let global_path = crate::configuration::user_config_dir()
        .map(|dir| dir.join("config.toml"))
        .or_else(|| home.map(|h| h.join(".config").join("nemo-relay").join("config.toml")))
        .unwrap_or_else(|| PathBuf::from("~/.config/nemo-relay/config.toml"));
    let system_path = crate::configuration::system_config_dir().join("config.toml");
    let explicit = gateway_overrides.config.as_deref().map(layer_status);
    let global = if explicit_config {
        replaced_user_layer_status(&global_path)
    } else {
        layer_status(&global_path)
    };
    let system = layer_status(&system_path);
    let upstream_auth = if matches!(resolution.status, Status::Fail) {
        UpstreamAuthInfo::unknown()
    } else {
        UpstreamAuthInfo::from_effective_gateway_auth(&resolved.gateway)
    };

    ConfigurationInfo {
        explicit,
        global,
        system,
        unsupported_project_files: unsupported_project_files(cwd),
        upstream_auth,
        plugin_configs: diagnostic_plugin_config_paths(
            gateway_overrides.config.as_ref(),
            gateway_overrides.plugin_config_path.as_ref(),
        )
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
        dynamic_plugins: resolved
            .dynamic_plugins
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

fn unsupported_project_files(cwd: Option<&Path>) -> Vec<ConfigLayer> {
    let Some(cwd) = cwd else {
        return Vec::new();
    };
    cwd.ancestors()
        .flat_map(|ancestor| {
            let directory = ancestor.join(".nemo-relay");
            ["config.toml", "plugins.toml", ".dynamic-plugins.json"]
                .map(move |filename| directory.join(filename))
        })
        .filter(|path| path.is_file())
        .map(|path| ConfigLayer {
            path,
            status: Status::Warn,
            active: false,
            details: "unsupported project file; ignored by Relay; move configuration to a user or system location, or select a config file explicitly".into(),
        })
        .collect()
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
    let mut layer = toml_layer_status(path);
    if !matches!(layer.status, Status::Pass) {
        return layer;
    }
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(err) => {
            layer.status = Status::Fail;
            layer.active = false;
            layer.details = format!("unreadable: {err}");
            return layer;
        }
    };
    let table = match text.parse::<toml::Table>() {
        Ok(table) => table,
        Err(err) => {
            layer.status = Status::Fail;
            layer.active = false;
            layer.details = format!("invalid TOML: {err}");
            return layer;
        }
    };
    match crate::configuration::validate_shared_config_shape(toml::Value::Table(table)) {
        Ok(()) => layer,
        Err(err) => ConfigLayer {
            path: path.to_path_buf(),
            status: Status::Fail,
            active: false,
            details: err.to_string(),
        },
    }
}

fn toml_layer_status(path: &Path) -> ConfigLayer {
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

fn replaced_user_layer_status(path: &Path) -> ConfigLayer {
    ConfigLayer {
        path: path.to_path_buf(),
        status: Status::Info,
        active: false,
        details: "replaced by explicit --config".into(),
    }
}

fn plugin_layer_status(
    path: &Path,
    contributing_paths: &[PathBuf],
    plugin_error: Option<&str>,
) -> ConfigLayer {
    let mut layer = toml_layer_status(path);
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
    probe_mode: DoctorProbeMode,
    resolved: &ResolvedConfig,
) -> Vec<AgentInfo> {
    let mut out = Vec::with_capacity(CodingAgent::ALL.len());
    for agent in CodingAgent::ALL {
        if target_agent.is_some_and(|target| target != agent) {
            continue;
        }
        out.push(collect_agent(agent, target_agent == Some(agent), probe_mode, resolved).await);
    }
    out
}

async fn collect_agent(
    agent: CodingAgent,
    target_requested: bool,
    probe_mode: DoctorProbeMode,
    resolved: &ResolvedConfig,
) -> AgentInfo {
    let configured = agent_configured(agent, &resolved.agents);
    let command = agent_command(agent, &resolved.agents);
    let argv = crate::process::command_argv(&command);
    let exec = argv.first().map(String::as_str).unwrap_or_default();
    let path = crate::process::resolve_executable(exec);
    let version = if path.is_some() {
        probe_version(&crate::process::version_probe_argv(agent, &argv)).await
    } else {
        None
    };
    let mut status = agent_command_status(path.as_deref(), configured, target_requested);
    let (hook_status, hook_details) = hook_status(agent, &resolved.agents);
    status = combine_status(status, hook_status, configured || target_requested);
    let mut details = agent_details(
        configured,
        target_requested,
        path.as_deref(),
        exec,
        hook_details,
    );
    apply_agent_version_status(
        agent,
        version.as_deref(),
        path.is_some(),
        configured || target_requested,
        &mut status,
        &mut details,
    );
    let checks =
        agent_preflight_checks(agent, probe_mode, configured || target_requested, resolved).await;
    status = fold_preflight_checks(status, &checks);
    AgentInfo {
        name: agent.as_arg(),
        status,
        configured,
        command,
        path,
        version,
        annotation: details.join("; "),
        checks,
    }
}

/// Preflight checks that depend on how the *agent* is set up rather than on
/// anything NeMo Relay wrote.
///
/// Only pi has any. Codex reads a generated hook config and Claude Code reads a
/// settings file, both written by `nemo-relay install`, so their setup is
/// already fully described by `hook_status`. pi's hooks live in an extension the
/// user installs wherever they like, and pi's own trust rules decide whether it
/// loads -- neither of which is visible from anything NeMo Relay owns.
async fn agent_preflight_checks(
    agent: CodingAgent,
    probe_mode: DoctorProbeMode,
    relevant: bool,
    resolved: &ResolvedConfig,
) -> Vec<Check> {
    if agent != CodingAgent::Pi {
        return Vec::new();
    }
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut checks = vec![pi_extension_trust_check(&cwd)];
    checks.extend(pi_managed_install_check());
    // The filesystem check is free and always worth running -- a stray project-scoped
    // extension is worth knowing about whether or not pi is set up yet. The network probe
    // is not: a bare `nemo-relay doctor` on a machine that does not use pi would spend the
    // timeout budget dialling a gateway nobody asked about, and warn about it.
    if relevant {
        checks.push(pi_gateway_reachability_check(probe_mode, resolved).await);
    }
    checks
}

/// Report a Relay-managed install that is older than the binary driving it.
///
/// pi ships breaking changes through *minor* releases, and the extension is what
/// translates pi's hook shapes for the gateway -- so an extension written by an older
/// Relay is the same silent failure the version floor exists to catch, arriving from the
/// other side. Only a managed install can be checked: a copy the user placed themselves
/// records no version, and inferring one from its contents would report drift on an edit.
///
/// Nothing is reported when there is no managed install, because there is nothing to be
/// stale -- the other pi checks already cover "installed, somewhere else" and "not
/// installed at all".
fn pi_managed_install_check() -> Option<Check> {
    const NAME: &str = "pi extension install";
    let installed = crate::agents::pi::install::installed_version()?;
    let current = crate::agents::pi::assets::EXTENSION_VERSION;
    if installed == current {
        return Some(Check {
            name: NAME,
            status: Status::Pass,
            details: format!("NeMo Relay-managed, version {installed}"),
        });
    }
    Some(Check {
        name: NAME,
        status: Status::Warn,
        details: format!(
            "the installed NeMo Relay pi extension is version {installed} and this build \
             ships {current}. pi can change a hook shape in a minor release, and the \
             symptom is missing spans rather than an error. Run `nemo-relay install pi` \
             to update it"
        ),
    })
}

/// Warn when an extension sits somewhere pi will silently ignore.
///
/// The danger is specific: pi adds project-scoped extensions to its candidate
/// set only for a trusted project, and its non-interactive modes never prompt
/// for trust. The skip is a bare conditional rather than an error path, so pi
/// does not treat it as a failure and nothing surfaces it -- and the extension
/// cannot surface it either, because it is not running. This check is the only
/// place a user finds out before wondering why NeMo Relay "does nothing".
fn pi_extension_trust_check(cwd: &Path) -> Check {
    const NAME: &str = "pi extension load path";
    let sites = crate::agents::pi::doctor::relay_extension_sites(cwd);
    let project_sites: Vec<&crate::agents::pi::doctor::ExtensionSite> = sites
        .iter()
        .filter(|site| site.scope == crate::agents::pi::doctor::ExtensionScope::Project)
        .collect();

    // Ahead of the project warning, because a project copy must not hide it: the two ungated
    // copies load whether or not the project is trusted, and that is the louder problem.
    // Asked of the copy the launcher would choose, not of whichever site sorts first: a copy pi's
    // settings switch off is not one of the two that load, and naming it here would report a
    // duplicate that does not exist.
    if let Some(launched) = crate::agents::pi::doctor::launchable_extension_path(cwd)
        && let Some(duplicate) =
            crate::agents::pi::doctor::conflicting_extension_site(cwd, &launched)
    {
        return Check {
            name: NAME,
            status: Status::Warn,
            details: format!(
                "two copies would load: {} and {}. pi de-duplicates by path, not by package, \
                 so both register hooks and every event is reported twice -- each turn is \
                 closed as superseded by its own duplicate, and the inline-shell gate decides \
                 one command twice. Keep one copy",
                launched.display(),
                duplicate.display()
            ),
        };
    }

    if let Some(project) = project_sites.first() {
        return Check {
            name: NAME,
            status: Status::Warn,
            details: format!(
                "{} is project-scoped, so pi loads it only when the project is trusted, and \
                 `-p`, `--mode json` and `--mode rpc` never prompt -- it is silently skipped \
                 there, with nothing reporting it. Install at user scope \
                 (`~/.pi/agent/extensions/`, or `pi install` without `--local`), or set {} to \
                 it. `nemo-relay run --agent pi` passes `-e`, which is never trust-gated, but it \
                 resolves only those two routes and refuses a project-scoped copy for the same \
                 reason this warns about one",
                project.path.display(),
                crate::agents::pi::launch::PI_EXTENSION_PATH_ENV
            ),
        };
    }

    use crate::agents::pi::doctor::SettingsFilter;
    match sites.first() {
        // Installed, and switched off in pi's own settings. Same silent drop as the trust gate,
        // from the other direction -- and `-e` ignores those filters, so the launcher still
        // instruments a session the user's own `pi` runs are missing.
        Some(site) if site.filter == SettingsFilter::Excluded => Check {
            name: NAME,
            status: Status::Warn,
            details: format!(
                "{} is recorded in pi's settings with its extensions filtered off, so pi does not \
                 load it -- remove the `extensions` filter, or the `autoload: false`, on that \
                 entry. `nemo-relay run --agent pi` is unaffected: it passes `-e`, which applies \
                 no settings filter",
                site.path.display()
            ),
        },
        // Installed, with a filter this check cannot evaluate: pi matches glob patterns with
        // `minimatch`, which is not reimplemented here. Reporting Pass would claim pi loads it,
        // and that is the claim this whole check exists to stop being made on no evidence.
        Some(site) if site.filter == SettingsFilter::Undecided => Check {
            name: NAME,
            status: Status::Warn,
            details: format!(
                "{} is recorded in pi's settings with a glob filter on its extensions, and \
                 whether pi loads it cannot be decided from here. Check that filter if NeMo \
                 Relay appears to do nothing. `nemo-relay run --agent pi` is unaffected: it \
                 passes `-e`, which applies no settings filter",
                site.path.display()
            ),
        },
        Some(site) => Check {
            name: NAME,
            status: Status::Pass,
            details: format!("{} ({})", site.path.display(), site.scope.describe()),
        },
        None => Check {
            name: NAME,
            status: Status::Info,
            details: format!(
                "the NeMo Relay pi extension was not found; run `nemo-relay install pi`, \
                 or set {} to a copy you manage yourself",
                crate::agents::pi::launch::PI_EXTENSION_PATH_ENV
            ),
        },
    }
}

/// Probe the gateway the pi extension will post to.
///
/// Deliberately a warning rather than a failure when it does not answer: the
/// gateway is a separate process a user normally starts alongside pi, so doctor
/// running before it is the common case rather than a broken one. What makes
/// this worth probing at all is that the extension defaults to **failing open**,
/// so an unreachable gateway does not stop pi -- it silently stops enforcing.
async fn pi_gateway_reachability_check(
    probe_mode: DoctorProbeMode,
    resolved: &ResolvedConfig,
) -> Check {
    const NAME: &str = "pi gateway reachability";
    let url = crate::agents::pi::doctor::gateway_url(Some(resolved.gateway.bind));
    if probe_mode.is_offline() {
        return Check {
            name: NAME,
            status: Status::Info,
            details: format!("{url} (live reachability probe skipped (--offline))"),
        };
    }
    // The shared probe classifies rather than just connecting, which is the difference
    // between "your gateway is down" and "something else owns that port" -- two problems
    // with nothing in common. It is blocking and loopback-only, hence the offload; a pi
    // gateway pointed somewhere else falls back to a plain reachability check below.
    let probed = if is_loopback(&url) {
        let probe_url = url.clone();
        timeout(
            NETWORK_TIMEOUT,
            tokio::task::spawn_blocking(move || crate::gateway::client::probe(&probe_url, None)),
        )
        .await
        .ok()
        .and_then(Result::ok)
    } else {
        None
    };

    let unreachable = |detail: &str| Check {
        name: NAME,
        status: Status::Warn,
        details: format!(
            "{url} {detail}; start the gateway before pi, or every hook will fault -- and \
             because the extension fails open by default, pi keeps running with no policy applied"
        ),
    };

    match probed {
        Some(crate::gateway::client::RelayHealth::Compatible) => Check {
            name: NAME,
            status: Status::Pass,
            details: format!("{url} is running a compatible NeMo Relay gateway"),
        },
        Some(crate::gateway::client::RelayHealth::Incompatible) => Check {
            name: NAME,
            status: Status::Warn,
            details: format!(
                "{url} is a NeMo Relay gateway of an incompatible version; the pi extension \
                 posts hooks here"
            ),
        },
        Some(crate::gateway::client::RelayHealth::Foreign) => Check {
            name: NAME,
            status: Status::Warn,
            details: format!(
                "{url} is answering, but it is not a NeMo Relay gateway -- something else owns \
                 that port, so every hook goes somewhere unexpected"
            ),
        },
        Some(crate::gateway::client::RelayHealth::Unavailable) => unreachable("is not answering"),
        // Non-loopback, or the blocking probe did not finish in budget.
        None => match reqwest::Client::builder()
            .timeout(NETWORK_TIMEOUT)
            .build()
            .ok()
        {
            Some(client) => match client.get(format!("{url}/healthz")).send().await {
                Ok(response) if response.status().is_success() => Check {
                    name: NAME,
                    status: Status::Pass,
                    details: format!("{url} answered /healthz"),
                },
                Ok(response) => Check {
                    name: NAME,
                    status: Status::Warn,
                    details: format!(
                        "{url} answered /healthz with HTTP {}; the pi extension posts hooks here",
                        response.status().as_u16()
                    ),
                },
                Err(_) => unreachable("did not answer /healthz"),
            },
            None => unreachable("could not be probed"),
        },
    }
}

/// Whether a URL names the local machine, which is all the shared probe supports.
fn is_loopback(url: &str) -> bool {
    reqwest::Url::parse(url)
        .ok()
        .and_then(|parsed| parsed.host_str().map(str::to_owned))
        .is_some_and(|host| {
            host == "localhost"
                || host
                    .trim_matches(['[', ']'])
                    .parse::<std::net::IpAddr>()
                    .is_ok_and(|ip| ip.is_loopback())
        })
}

fn agent_details(
    configured: bool,
    target_requested: bool,
    path: Option<&Path>,
    exec: &str,
    hook_details: String,
) -> Vec<String> {
    let mut details = vec![if configured {
        "configured".to_string()
    } else if target_requested {
        "not configured; first run will launch setup".to_string()
    } else {
        "not configured".to_string()
    }];
    if path.is_none() {
        details.push(format!("command `{exec}` not found"));
    }
    if !hook_details.is_empty() {
        details.push(hook_details);
    }
    details
}

fn apply_agent_version_status(
    agent: CodingAgent,
    version: Option<&str>,
    executable_found: bool,
    version_required: bool,
    status: &mut Status,
    details: &mut Vec<String>,
) {
    // Three outcomes, not two. A host whose minor releases can move a hook shape has a
    // band above the floor that is untested rather than supported, and reporting it as a
    // plain pass is what let a newer pi look verified.
    let mut unverified = None;
    let problem = match version {
        Some(version) => match agent.validate_version_output(version) {
            Ok(parsed) => {
                unverified = agent.unverified_version(&parsed);
                None
            }
            Err(problem) => Some(problem),
        },
        None if executable_found => Some(format!(
            "could not determine version; NeMo Relay requires {}",
            agent.version_requirement()
        )),
        None => None,
    };
    if let Some(unverified) = unverified {
        // Warn even when the agent is not the requested target: an untested host is a
        // property of the machine, not of what the user asked about.
        *status = combine_status(*status, Status::Warn, true);
        details.push(unverified);
    }
    if let Some(problem) = problem {
        *status = combine_status(
            *status,
            if version_required {
                Status::Fail
            } else {
                Status::Warn
            },
            true,
        );
        details.push(problem);
    }
}

fn agent_command(agent: CodingAgent, agents: &AgentConfigs) -> String {
    configured_agent_command(agent, agents)
        .cloned()
        .unwrap_or_else(|| agent.executable().to_string())
}

fn configured_agent_command(agent: CodingAgent, agents: &AgentConfigs) -> Option<&String> {
    crate::agents::config(agent, agents).command.as_ref()
}

fn agent_configured(agent: CodingAgent, agents: &AgentConfigs) -> bool {
    crate::agents::configured(agent, agents)
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

/// Fold preflight findings into the agent status, configured or not.
///
/// The readiness gate `combine_status` applies elsewhere asks whether the user has
/// told Relay to use this agent, which is the right question for "the hook config
/// is missing" -- nobody wants a bare `doctor` complaining about an agent they do
/// not run. It is the wrong question here, because a preflight finding is
/// *evidence* rather than readiness: every warning branch already requires the
/// extension to be installed on this machine, and a machine without one reports
/// `Info`, which never folds either way.
///
/// Gating on `configured` meant `agents --json` reported `status: "pass"` for a pi
/// whose only install is project-scoped -- the exact silent skip the check exists
/// to catch -- while its own nested check said `warn`.
fn fold_preflight_checks(status: Status, checks: &[Check]) -> Status {
    checks.iter().fold(status, |status, check| {
        combine_status(status, check.status, true)
    })
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
    match crate::agents::hook_status(agent, agents) {
        Ok(details) => (Status::Pass, details),
        Err(details) => (Status::Fail, details),
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

async fn collect_observability(gateway: &GatewayConfig, probe_mode: DoctorProbeMode) -> Vec<Check> {
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
    let component_errors = register_and_validate_plugin_components(&plugin_config);
    if !component_errors.is_empty() {
        checks.extend(component_errors.into_iter().map(|error| Check {
            name: error.check_name(),
            status: Status::Fail,
            details: error.diagnostic_details(),
        }));
        return checks;
    }
    let report = validate_plugin_config(&plugin_config);
    let response_cache_invalid = report.diagnostics.iter().any(|diagnostic| {
        diagnostic.level == DiagnosticLevel::Error
            && diagnostic.component.as_deref().is_some_and(|component| {
                // Backend diagnostics are tagged `response_cache.backend[.kind]`.
                component == "response_cache" || component.starts_with("response_cache.")
            })
    });
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
        collect_observability_component_checks(&mut checks, config, probe_mode).await;
    } else {
        checks.push(Check {
            name: "Observability plugin",
            status: Status::Info,
            details: "component not configured".into(),
        });
    }
    collect_pricing_component_checks(&mut checks, &plugin_config);
    collect_response_cache_component_checks(
        &mut checks,
        &plugin_config,
        response_cache_invalid,
        probe_mode,
    )
    .await;

    checks
}

async fn collect_response_cache_component_checks(
    checks: &mut Vec<Check>,
    plugin_config: &PluginConfig,
    config_invalid: bool,
    probe_mode: DoctorProbeMode,
) {
    // The response cache is a section of the adaptive component, not its own
    // plugin kind: find the adaptive component and look for `response_cache`.
    let Some(component) = plugin_config
        .components
        .iter()
        .find(|component| component.kind == ADAPTIVE_PLUGIN_KIND)
    else {
        checks.push(Check {
            name: "Response cache",
            status: Status::Info,
            details: "not configured".into(),
        });
        return;
    };
    let Some(response_cache_value) = component
        .config
        .get("response_cache")
        .filter(|value| !value.is_null())
    else {
        checks.push(Check {
            name: "Response cache",
            status: Status::Info,
            details: "not configured".into(),
        });
        return;
    };
    if !component.enabled {
        checks.push(Check {
            name: "Response cache",
            status: Status::Info,
            details: "configured but disabled (adaptive plugin disabled)".into(),
        });
        return;
    }
    // Don't probe the backend for a config validation already rejected — that
    // would report a misleading "reachable" Pass next to the validation failure.
    if config_invalid {
        checks.push(Check {
            name: "Response cache",
            status: Status::Fail,
            details: "config invalid (see plugin diagnostics above)".into(),
        });
        return;
    }
    let config: ResponseCacheConfig = match serde_json::from_value(response_cache_value.clone()) {
        Ok(config) => config,
        Err(error) => {
            checks.push(Check {
                name: "Response cache",
                status: Status::Fail,
                details: format!("invalid response_cache config: {error}"),
            });
            return;
        }
    };
    if probe_mode.is_offline()
        && let Err(error) = response_cache::store::validate_backend_target(&config)
    {
        checks.push(Check {
            name: "Response cache",
            status: Status::Fail,
            details: format!("invalid backend target: {error}"),
        });
        return;
    }
    if let Some(tools) = config.tools.as_ref() {
        let details = if tools.enabled {
            let cacheable_classes = tools
                .classes
                .values()
                .filter(|class| class.cacheable)
                .count();
            let cacheable_overrides = tools
                .overrides
                .values()
                .filter(|override_| override_.cacheable == Some(true))
                .count();
            format!(
                "on; {cacheable_classes} cacheable class(es); {cacheable_overrides} cacheable override(s); default {}",
                if tools.default.cacheable {
                    "cacheable"
                } else {
                    "uncached"
                }
            )
        } else {
            "configured but disabled".to_string()
        };
        checks.push(Check {
            name: "Response cache (tools)",
            status: Status::Info,
            details,
        });
    }
    if probe_mode.is_offline() && config.backend.kind != "in_memory" {
        checks.push(Check {
            name: "Response cache",
            status: Status::Info,
            details: format!(
                "configured; live {} backend probe skipped (--offline)",
                config.backend.kind
            ),
        });
        return;
    }
    checks.push(response_cache_backend_check(response_cache::check_backend_health(&config)).await);
}

async fn response_cache_backend_check(
    health_check: impl std::future::Future<Output = Result<String, String>>,
) -> Check {
    match timeout(NETWORK_TIMEOUT, health_check).await {
        Ok(Ok(backend)) => Check {
            name: "Response cache",
            status: Status::Pass,
            details: format!("on; backend '{backend}' reachable"),
        },
        Ok(Err(error)) => Check {
            name: "Response cache",
            status: Status::Fail,
            details: format!("backend not reachable: {error}"),
        },
        Err(_) => Check {
            name: "Response cache",
            status: Status::Fail,
            details: format!(
                "backend not reachable: probe timed out after {}s",
                NETWORK_TIMEOUT.as_secs()
            ),
        },
    }
}

async fn collect_observability_component_checks(
    checks: &mut Vec<Check>,
    config: &Value,
    probe_mode: DoctorProbeMode,
) {
    checks.extend(observability_atof_file_checks(config));
    if let Some(check) = observability_file_exporter_check(config, "atif") {
        checks.push(check);
    }
    checks.extend(observability_http_exporter_checks(config, probe_mode).await);
    if section_enabled(config, "atof") && !atof_stream_sinks(config).is_empty() {
        if atof_streaming_supported() {
            checks.extend(observability_atof_stream_checks(config, probe_mode).await);
        } else {
            checks.push(Check {
                name: "ATOF stream sink",
                status: Status::Fail,
                details: "ATOF stream sinks are not available in this binary".into(),
            });
        }
    }
}

fn observability_atof_file_checks(config: &Value) -> Vec<Check> {
    if !section_enabled(config, "atof") {
        return Vec::new();
    }
    let sinks = config
        .get("atof")
        .and_then(|section| section.get("sinks"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
        .filter(|(_, sink)| sink.get("type").and_then(Value::as_str) == Some("file"));
    let checks = sinks
        .map(
            |(index, sink)| match sink.get("output_directory").and_then(Value::as_str) {
                Some(path) => {
                    let mut check = check_directory("ATOF file sink", Path::new(path));
                    check.details = format!("sinks[{index}]: {}", check.details);
                    check
                }
                None => Check {
                    name: "ATOF file sink",
                    status: Status::Info,
                    details: format!("sinks[{index}] uses the runtime default output directory"),
                },
            },
        )
        .collect::<Vec<_>>();
    if checks.is_empty() {
        vec![Check {
            name: "ATOF file sink",
            status: Status::Info,
            details: "no file sinks configured".into(),
        }]
    } else {
        checks
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

async fn observability_http_exporter_checks(
    config: &Value,
    probe_mode: DoctorProbeMode,
) -> Vec<Check> {
    if !section_enabled(config, "opentelemetry") {
        return Vec::new();
    }
    let Some(endpoints) = config
        .get("opentelemetry")
        .and_then(|section| section.get("endpoints"))
        .and_then(Value::as_array)
    else {
        return Vec::new();
    };
    join_all(
        endpoints.iter().enumerate().map(|(index, endpoint)| {
            observability_http_exporter_check(index, endpoint, probe_mode)
        }),
    )
    .await
}

async fn observability_http_exporter_check(
    index: usize,
    endpoint: &Value,
    probe_mode: DoctorProbeMode,
) -> Check {
    let endpoint_type = endpoint
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let label = "OpenTelemetry endpoint";
    let Some(url) = endpoint.get("endpoint").and_then(Value::as_str) else {
        return Check {
            name: label,
            status: Status::Fail,
            details: format!("endpoints[{index}] ({endpoint_type}): endpoint is required"),
        };
    };
    let transport = endpoint
        .get("transport")
        .and_then(Value::as_str)
        .unwrap_or("http_binary");
    let mut check = if transport == "grpc" {
        probe_grpc_endpoint(label, url, probe_mode).await
    } else {
        probe_http_endpoint(label, url, probe_mode).await
    };
    check.details = format!("endpoints[{index}] ({endpoint_type}): {}", check.details);
    check
}

async fn probe_grpc_endpoint(label: &'static str, url: &str, mode: DoctorProbeMode) -> Check {
    if mode.is_offline() {
        return match validate_grpc_endpoint(url) {
            Ok(_) => Check {
                name: label,
                status: Status::Info,
                details: "live network probe skipped (--offline)".into(),
            },
            Err(details) => Check {
                name: label,
                status: Status::Fail,
                details,
            },
        };
    }
    probe_tcp_named(label, url).await
}

async fn probe_http_endpoint(label: &'static str, url: &str, mode: DoctorProbeMode) -> Check {
    let effective_url = resolve_http_trace_endpoint(url);
    if mode.is_offline() {
        return match validate_otlp_http_endpoint(effective_url.as_ref()) {
            Ok(()) => Check {
                name: label,
                status: Status::Info,
                details: "live network probe skipped (--offline)".into(),
            },
            Err(details) => Check {
                name: label,
                status: Status::Fail,
                details,
            },
        };
    }
    probe_otlp_http_named(label, effective_url.as_ref()).await
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

fn atof_stream_sinks(config: &Value) -> Vec<(usize, &Value)> {
    config
        .get("atof")
        .and_then(|section| section.get("sinks"))
        .and_then(Value::as_array)
        .map(|sinks| {
            sinks
                .iter()
                .enumerate()
                .filter(|(_, sink)| sink.get("type").and_then(Value::as_str) == Some("stream"))
                .collect()
        })
        .unwrap_or_default()
}

fn atof_streaming_supported() -> bool {
    cfg!(feature = "atof-streaming")
}

async fn observability_atof_stream_checks(
    config: &Value,
    probe_mode: DoctorProbeMode,
) -> Vec<Check> {
    let streams = atof_stream_sinks(config);
    let mut checks = Vec::with_capacity(streams.len());
    for (index, sink) in streams {
        checks.push(probe_atof_stream_sink(index, sink, probe_mode).await);
    }
    checks
}

async fn probe_atof_stream_sink(
    index: usize,
    endpoint: &Value,
    probe_mode: DoctorProbeMode,
) -> Check {
    let name = "ATOF stream sink";
    let Some(url) = endpoint.get("url").and_then(Value::as_str) else {
        return Check {
            name,
            status: Status::Fail,
            details: format!("sinks[{index}]: missing url"),
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
            details: format!("sinks[{index}] {transport} {url}: timeout_millis must be > 0"),
        };
    }
    let headers = match endpoint_headers(endpoint) {
        Ok(headers) => headers,
        Err(err) => {
            return Check {
                name,
                status: Status::Fail,
                details: format!("sinks[{index}] {transport} {url}: {err}"),
            };
        }
    };
    if let Err(details) = validate_atof_stream_probe_target(index, transport, url) {
        return Check {
            name,
            status: Status::Fail,
            details,
        };
    }
    if probe_mode.is_offline() {
        return Check {
            name,
            status: Status::Info,
            details: format!(
                "sinks[{index}] {transport} {url}: live network probe skipped (--offline)"
            ),
        };
    }
    let payload = match doctor_atof_probe_payload() {
        Ok(payload) => payload,
        Err(err) => {
            return Check {
                name,
                status: Status::Fail,
                details: format!("sinks[{index}] {transport} {url}: {err}"),
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
            details: format!("sinks[{index}] {transport} {url}: unsupported transport"),
        },
    }
}

#[cfg(test)]
async fn probe_atof_endpoint(index: usize, endpoint: &Value) -> Check {
    probe_atof_stream_sink(index, endpoint, DoctorProbeMode::Live).await
}

fn validate_atof_stream_probe_target(
    index: usize,
    transport: &str,
    url: &str,
) -> Result<(), String> {
    let parsed = reqwest::Url::parse(url)
        .map_err(|error| format!("sinks[{index}] {transport} {url}: {error}"))?;
    if parsed.host_str().is_none() {
        return Err(format!("sinks[{index}] {transport} {url}: missing host"));
    }
    let valid_scheme = match transport {
        "http_post" | "ndjson" => matches!(parsed.scheme(), "http" | "https"),
        "websocket" => matches!(parsed.scheme(), "ws" | "wss"),
        _ => {
            return Err(format!(
                "sinks[{index}] {transport} {url}: unsupported transport"
            ));
        }
    };
    if valid_scheme {
        Ok(())
    } else {
        let expected = match transport {
            "http_post" | "ndjson" => "http or https",
            "websocket" => "ws or wss",
            _ => unreachable!("unsupported transports return earlier"),
        };
        Err(format!(
            "sinks[{index}] {transport} {url}: invalid scheme (must be {expected})"
        ))
    }
}

fn endpoint_headers(endpoint: &Value) -> Result<Vec<(String, String)>, String> {
    let mut out = Vec::new();
    let mut names = std::collections::HashSet::new();
    append_configured_headers(endpoint.get("headers"), &mut names, &mut out)?;
    append_environment_headers(endpoint.get("header_env"), &mut names, &mut out)?;
    Ok(out)
}

fn append_configured_headers(
    headers: Option<&Value>,
    names: &mut std::collections::HashSet<reqwest::header::HeaderName>,
    out: &mut Vec<(String, String)>,
) -> Result<(), String> {
    let Some(headers) = headers else {
        return Ok(());
    };
    let Some(object) = headers.as_object() else {
        return Err("headers must be an object of string values".into());
    };
    for (key, value) in object {
        let name = reqwest::header::HeaderName::from_bytes(key.as_bytes())
            .map_err(|error| error.to_string())?;
        let Some(value) = value.as_str() else {
            return Err(format!("headers.{key} must be a string"));
        };
        if value.trim().is_empty() {
            return Err(format!("headers.{key} must not be blank"));
        }
        reqwest::header::HeaderValue::from_bytes(value.as_bytes())
            .map_err(|error| format!("headers.{key} invalid: {error}"))?;
        if !names.insert(name) {
            return Err(format!("header {key:?} appears more than once"));
        }
        out.push((key.clone(), value.to_string()));
    }
    Ok(())
}

fn append_environment_headers(
    header_env: Option<&Value>,
    names: &mut std::collections::HashSet<reqwest::header::HeaderName>,
    out: &mut Vec<(String, String)>,
) -> Result<(), String> {
    let Some(header_env) = header_env else {
        return Ok(());
    };
    let Some(object) = header_env.as_object() else {
        return Err("header_env must be an object of string values".into());
    };
    for (key, variable) in object {
        let name = reqwest::header::HeaderName::from_bytes(key.as_bytes())
            .map_err(|error| error.to_string())?;
        if names.contains(&name) {
            return Err(format!(
                "header {key:?} cannot appear in both headers and header_env"
            ));
        }
        let Some(variable) = variable.as_str() else {
            return Err(format!("header_env.{key} must be a string"));
        };
        let value = std::env::var(variable)
            .map_err(|_| format!("environment variable {variable:?} is not set"))?;
        if value.trim().is_empty() {
            return Err(format!("environment variable {variable:?} is blank"));
        }
        reqwest::header::HeaderValue::from_bytes(value.as_bytes())
            .map_err(|error| format!("header_env.{key} invalid: {error}"))?;
        names.insert(name);
        out.push((key.clone(), value));
    }
    Ok(())
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
                name: "ATOF stream sink",
                status: Status::Fail,
                details: format!("sinks[{index}] {transport} {url}: could not build client: {err}"),
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
            name: "ATOF stream sink",
            status: Status::Pass,
            details: format!(
                "sinks[{index}] {transport} {url} (HTTP {})",
                response.status()
            ),
        },
        Ok(response) => Check {
            name: "ATOF stream sink",
            status: Status::Fail,
            details: format!(
                "sinks[{index}] {transport} {url} (HTTP {})",
                response.status()
            ),
        },
        Err(err) => Check {
            name: "ATOF stream sink",
            status: Status::Fail,
            details: format!("sinks[{index}] {transport} {url}: {err}"),
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
                name: "ATOF stream sink",
                status: Status::Fail,
                details: format!(
                    "sinks[{index}] websocket {url}: invalid scheme (must be ws or wss)"
                ),
            };
        }
        Err(err) => {
            return Check {
                name: "ATOF stream sink",
                status: Status::Fail,
                details: format!("sinks[{index}] websocket {url}: {err}"),
            };
        }
    }
    let mut request = match url.into_client_request() {
        Ok(request) => request,
        Err(err) => {
            return Check {
                name: "ATOF stream sink",
                status: Status::Fail,
                details: format!("sinks[{index}] websocket {url}: {err}"),
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
                    name: "ATOF stream sink",
                    status: Status::Fail,
                    details: format!("sinks[{index}] websocket {url}: {err}"),
                };
            }
        };
        let value =
            match tokio_tungstenite::tungstenite::http::header::HeaderValue::from_str(&value) {
                Ok(value) => value,
                Err(err) => {
                    return Check {
                        name: "ATOF stream sink",
                        status: Status::Fail,
                        details: format!("sinks[{index}] websocket {url}: {err}"),
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
                    name: "ATOF stream sink",
                    status: Status::Pass,
                    details: format!("sinks[{index}] websocket {url}"),
                },
                Ok(Err(err)) => Check {
                    name: "ATOF stream sink",
                    status: Status::Fail,
                    details: format!("sinks[{index}] websocket {url}: {err}"),
                },
                Err(_) => Check {
                    name: "ATOF stream sink",
                    status: Status::Fail,
                    details: format!(
                        "sinks[{index}] websocket {url}: timed out sending probe payload"
                    ),
                },
            }
        }
        Ok(Err(err)) => Check {
            name: "ATOF stream sink",
            status: Status::Fail,
            details: format!("sinks[{index}] websocket {url}: {err}"),
        },
        Err(_) => Check {
            name: "ATOF stream sink",
            status: Status::Fail,
            details: format!("sinks[{index}] websocket {url}: timed out"),
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
pub(crate) fn format_agents_json(agents: &[AgentInfo]) -> Result<String, CliError> {
    serde_json::to_string_pretty(agents)
        .map_err(|err| CliError::Config(format!("could not serialize agents report: {err}")))
}

/// Top-level entry point invoked by `nemo-relay doctor`. Emits to stdout and returns the
/// appropriate process exit code (0 on pass-or-warn, 1 on any failure).
pub(crate) async fn run_doctor(
    target_agent: Option<CodingAgent>,
    json: bool,
    probe_mode: DoctorProbeMode,
    gateway_overrides: &GatewayOverrides,
    logging_fallback_error: Option<&CliError>,
) -> Result<std::process::ExitCode, CliError> {
    let mut report = collect_report(target_agent, probe_mode, gateway_overrides).await?;
    if let Some(error) = logging_fallback_error {
        let logging_details = format!(
            "could not resolve logging configuration: {error}; repair or recreate the named logging configuration file"
        );
        if report.configuration.resolution.status == Status::Pass {
            report.configuration.resolution.status = Status::Fail;
            report.configuration.resolution.details = logging_details;
        } else {
            report
                .configuration
                .resolution
                .details
                .push_str(&format!("; additionally, {logging_details}"));
        }
    }
    log::info!(
        target: "nemo_relay.diagnostics",
        event = "diagnostics_completed",
        agent_count = report.agents.len(),
        host_plugin_count = report.host_plugins.len(),
        observability_check_count = report.observability.len(),
        completion_check_count = report.completions.len();
        "Diagnostics completed"
    );
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
    let agents = agents_report().await?;
    log::info!(
        target: "nemo_relay.diagnostics",
        event = "diagnostics_completed",
        agent_count = agents.len(),
        report = "agents";
        "Agent diagnostics completed"
    );
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
