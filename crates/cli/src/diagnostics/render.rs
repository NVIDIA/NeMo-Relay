// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Doctor and agent-report presentation.

use super::*;

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
pub(super) fn report_has_warn(report: &DoctorReport) -> bool {
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

pub(super) fn format_human_header(out: &mut String, report: &DoctorReport) {
    out.push_str(&format!("\n  NeMo Relay {}\n", report.binary_version));
    out.push_str("  ─────────────────────────────────────────────\n");
    if let Some(agent) = &report.target_agent {
        out.push_str(&format!("  Target agent  {agent}\n\n"));
    }
}

pub(super) fn format_human_environment(out: &mut String, report: &DoctorReport) {
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

pub(super) fn format_human_configuration(out: &mut String, report: &DoctorReport) {
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

pub(super) fn format_human_plugin_configuration(out: &mut String, report: &DoctorReport) {
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

pub(super) fn format_human_agents(out: &mut String, report: &DoctorReport) {
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

pub(super) fn format_human_host_plugins(out: &mut String, report: &DoctorReport) {
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

pub(super) fn format_human_checks(out: &mut String, title: &str, checks: &[Check]) {
    out.push_str(&format!("  {title}\n"));
    for check in checks {
        out.push_str(&format!(
            "    {}  {:<22}  {}\n",
            format_status(check.status),
            check.name,
            check.details
        ));
    }
    out.push('\n');
}

pub(super) fn format_human_completion_checks(out: &mut String, checks: &[Check]) {
    out.push_str("  Completions\n");
    for check in checks {
        out.push_str(&format!("    {}\n", check.details));
    }
    out.push('\n');
}

pub(super) fn format_human_conclusion(out: &mut String, report: &DoctorReport) {
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

pub(super) fn format_layer(layer: &ConfigLayer) -> String {
    let active = if layer.active { " (loaded)" } else { "" };
    format!("{}   {}{}", layer.path.display(), layer.details, active)
}

pub(super) fn format_status(status: Status) -> &'static str {
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
    let resolved = resolve_server_config(&GatewayOverrides::default()).unwrap_or_default();
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
