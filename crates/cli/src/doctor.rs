// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `nemo-flow doctor` — environment + config + agent + observability health check.
//!
//! Split into three layers so the data path can be unit-tested without real I/O:
//!
//! - `collect_report()` does the I/O (env probes, $PATH scans, network checks, fs writability).
//! - `DoctorReport` is the resulting pure data shape.
//! - `format_human(&report)` / `format_json(&report)` render the report.

use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use serde::Serialize;
use tokio::time::timeout;

use crate::config::{
    CodingAgent, GatewayConfig, ResolvedConfig, ServerArgs, resolve_server_config,
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
    /// not on $PATH). Renders as a dim dot; not counted toward exit code.
    Info,
}

/// Snapshot of the running system that the doctor renders. Stable schema, versioned via
/// `schema_version`. Adding fields is non-breaking; removing or renaming requires a bump.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct DoctorReport {
    pub schema_version: u32,
    pub binary_version: &'static str,
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
    pub default_agent: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ConfigLayer {
    pub path: PathBuf,
    pub status: Status,
    pub details: String,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct AgentInfo {
    pub name: &'static str,
    pub path: Option<PathBuf>,
    pub version: Option<String>,
    /// Free-form annotation, e.g. "hooks: installed" once we wire up hook detection.
    pub annotation: String,
}

/// Drives all checks and produces a single `DoctorReport`. Network probes are bounded by a
/// short timeout so the command always returns quickly. Filesystem checks short-circuit on
/// the first missing directory.
pub(crate) async fn collect_report() -> Result<DoctorReport, CliError> {
    let resolved = resolve_server_config(&ServerArgs::default()).unwrap_or_default();
    let cwd = std::env::current_dir().ok();
    let home = home_dir();

    Ok(DoctorReport {
        schema_version: 1,
        binary_version: env!("CARGO_PKG_VERSION"),
        environment: collect_environment(),
        configuration: collect_configuration(cwd.as_deref(), home.as_deref()),
        agents: collect_agents().await,
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
    cwd: Option<&std::path::Path>,
    home: Option<&std::path::Path>,
) -> ConfigurationInfo {
    let workspace_path = cwd
        .map(|p| p.join(".nemo-flow").join("config.toml"))
        .unwrap_or_else(|| PathBuf::from(".nemo-flow/config.toml"));
    let global_path = home
        .map(|h| h.join(".config").join("nemo-flow").join("config.toml"))
        .unwrap_or_else(|| PathBuf::from("~/.config/nemo-flow/config.toml"));
    let system_path = PathBuf::from("/etc/nemo-flow/config.toml");

    ConfigurationInfo {
        workspace: layer_status(&workspace_path),
        global: layer_status(&global_path),
        system: layer_status(&system_path),
        // `default_agent` is reserved in the design for Phase 2 dispatch; not currently parsed
        // out of FileConfig. Doctor reports `None` until that lands.
        default_agent: None,
    }
}

fn layer_status(path: &std::path::Path) -> ConfigLayer {
    if !path.exists() {
        return ConfigLayer {
            path: path.to_path_buf(),
            status: Status::Info,
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
                details: "valid".into(),
            },
            Err(err) => ConfigLayer {
                path: path.to_path_buf(),
                status: Status::Fail,
                details: format!("invalid TOML: {err}"),
            },
        },
        Err(err) => ConfigLayer {
            path: path.to_path_buf(),
            status: Status::Fail,
            details: format!("unreadable: {err}"),
        },
    }
}

async fn collect_agents() -> Vec<AgentInfo> {
    let supported = [
        (CodingAgent::ClaudeCode, "claude", "claude"),
        (CodingAgent::Codex, "codex", "codex"),
        (CodingAgent::Cursor, "cursor", "cursor-agent"),
        (CodingAgent::Hermes, "hermes", "hermes"),
    ];
    let mut out = Vec::with_capacity(supported.len());
    for (_, display_name, exec) in supported {
        let path = which_on_path(exec);
        let version = match &path {
            Some(p) => probe_version(p).await,
            None => None,
        };
        out.push(AgentInfo {
            name: display_name,
            path,
            version,
            annotation: String::new(),
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

async fn probe_version(binary: &std::path::Path) -> Option<String> {
    // Spawn `<binary> --version` and read the first line of stdout. Bounded by the network
    // timeout (re-used as a generic short timeout) so a misbehaving binary doesn't hang doctor.
    let mut cmd = tokio::process::Command::new(binary);
    cmd.arg("--version")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .stdin(Stdio::null());
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

    checks.push(match &gateway.atif_dir {
        None => Check {
            name: "ATIF dir",
            status: Status::Info,
            details: "not configured".into(),
        },
        Some(path) => match check_dir_writable(path) {
            Ok(()) => Check {
                name: "ATIF dir",
                status: Status::Pass,
                details: format!("{} (writable)", path.display()),
            },
            Err(err) => Check {
                name: "ATIF dir",
                status: Status::Fail,
                details: format!("{}: {err}", path.display()),
            },
        },
    });

    checks.push(match &gateway.openinference_endpoint {
        None => Check {
            name: "OpenInference endpoint",
            status: Status::Info,
            details: "not configured".into(),
        },
        Some(url) => probe_http(url).await,
    });

    checks
}

fn check_dir_writable(dir: &std::path::Path) -> Result<(), std::io::Error> {
    use std::fs::OpenOptions;
    std::fs::create_dir_all(dir)?;
    // PID-suffixed name + create_new=true so we can never overwrite a real user file even if
    // they happen to have a `.nemo-flow-write-probe` of their own. The probe is removed
    // immediately; the file just witnesses that we have write access here.
    let probe = dir.join(format!(".nemo-flow-write-probe-{}", std::process::id()));
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&probe)?;
    std::fs::remove_file(&probe).ok();
    Ok(())
}

async fn probe_http(url: &str) -> Check {
    let client = match reqwest::Client::builder().timeout(NETWORK_TIMEOUT).build() {
        Ok(c) => c,
        Err(err) => {
            return Check {
                name: "OpenInference endpoint",
                status: Status::Fail,
                details: format!("could not build HTTP client: {err}"),
            };
        }
    };
    match client.get(url).send().await {
        Ok(resp) => Check {
            name: "OpenInference endpoint",
            status: if resp.status().is_success() || resp.status().is_redirection() {
                Status::Pass
            } else {
                Status::Warn
            },
            details: format!("{} (HTTP {})", url, resp.status().as_u16()),
        },
        Err(err) => Check {
            name: "OpenInference endpoint",
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
        "zsh" => Some(home.join(".zfunc").join("_nemo-flow")),
        "bash" => Some(home.join(".bash_completion.d").join("nemo-flow")),
        "fish" => Some(
            home.join(".config")
                .join("fish")
                .join("completions")
                .join("nemo-flow.fish"),
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
                "{shell_name}: not installed (run `nemo-flow completions {shell_name} > {}`)",
                path.display()
            ),
        }),
        None => checks.push(Check {
            name: "Completions",
            status: Status::Info,
            details: format!("{shell_name}: no known completion path; run `nemo-flow completions <shell>` to generate"),
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
        || matches!(report.configuration.workspace.status, Status::Fail)
        || matches!(report.configuration.global.status, Status::Fail)
        || matches!(report.configuration.system.status, Status::Fail);
    u8::from(any_fail)
}

/// Renders the doctor report in the fixed human-readable layout the design doc shows. Sections
/// stay in the same order across runs so users can diff across machines. The banner header lives
/// in `crate::banner::print_doctor_header` (called from `run_doctor` before this renders) so the
/// pure formatter stays banner-free for tests.
pub(crate) fn format_human(report: &DoctorReport) -> String {
    let mut out = String::new();
    out.push_str(&format!("\n  NeMo Flow {}\n", report.binary_version));
    out.push_str("  ─────────────────────────────────────────────\n");
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
    out.push('\n');

    out.push_str("  Agents detected\n");
    for agent in &report.agents {
        match &agent.path {
            Some(path) => {
                let version = agent.version.as_deref().unwrap_or("(unknown version)");
                out.push_str(&format!(
                    "    {:<8} {}\n               {}\n",
                    agent.name,
                    version,
                    path.display()
                ));
            }
            None => {
                out.push_str(&format!("    {:<8} not on $PATH\n", agent.name));
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
        // Don't say "All checks passed" — `Warn` results still map to exit code 0, so a clean
        // exit just means nothing is failing, not that everything is green. This wording keeps
        // the footer accurate when the report carries warnings.
        out.push_str("  No failing checks.\n");
    } else {
        out.push_str("  Some checks FAILED; see details above.\n");
    }
    out
}

fn format_layer(layer: &ConfigLayer) -> String {
    format!("{}   {}", layer.path.display(), layer.details)
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
    collect_agents().await
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
                "    {:<8} {}\n               {}\n",
                agent.name, version, path
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

/// Top-level entry point invoked by `nemo-flow doctor`. Emits to stdout and returns the
/// appropriate process exit code (0 on pass-or-warn, 1 on any failure).
pub(crate) async fn run_doctor(json: bool) -> Result<std::process::ExitCode, CliError> {
    let report = collect_report().await?;
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

/// Top-level entry point invoked by `nemo-flow agents`. Always exits 0; the data drives caller
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
