// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical coding-agent identity and compatibility policy.

pub(crate) mod claude;
pub(crate) mod codex;
pub(crate) mod pi;
pub(crate) mod shared;

use semver::Version;

/// Coding-agent hosts supported by the CLI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum CodingAgent {
    /// `claude-code` remains an input alias for older Relay configuration.
    ClaudeCode,
    Codex,
    Pi,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct AgentDescriptor {
    argument: &'static str,
    install_argument: &'static str,
    label: &'static str,
    executable: &'static str,
    hook_path: &'static str,
    version_product: &'static str,
    minimum_version: (u64, u64, u64),
    /// The last `major.minor` this integration was actually verified against, for a
    /// host whose minor releases can break the hook contract.
    ///
    /// `None` means the floor behaves like a floor: newer minors are additive and
    /// nothing above it needs reporting. `Some` means the opposite -- above it, the
    /// integration is untested rather than known-good, and saying so is the whole
    /// point, because the failure it guards against is silent.
    verified_through: Option<(u64, u64)>,
    hook_events: &'static [&'static str],
}

/// Why pi has no marketplace-plugin implementation.
///
/// Codex and Claude Code both install NeMo Relay through a plugin marketplace
/// (`codex plugin add`, `claude plugin install`) backed by a generated manifest.
/// pi has none of that: it has no `pi plugin` verb, no marketplace, and no MCP
/// client for a plugin-owned server to serve. Extensions are installed with
/// `pi install <source>` or by placing a file in an auto-discovered directory.
///
/// Rather than synthesize manifests pi will never read, the marketplace surface
/// rejects pi explicitly and the gateway surface (hooks, launch, doctor) is
/// implemented for real.
///
/// "No marketplace" is not "nothing to install into", and this constant used to
/// imply otherwise. `nemo-relay install pi` writes a package directory into pi's
/// own auto-discovery location (`agents::pi::install`), which is a real
/// persistent install by a route that has no marketplace in it. What stays
/// unsupported is everything below: a generated plugin manifest, a host
/// registration verb, and a plugin-owned MCP server, none of which pi has a
/// consumer for.
pub(crate) const PI_MARKETPLACE_UNSUPPORTED: &str = "pi has no plugin marketplace; \
     `nemo-relay install pi` writes the extension into pi's own auto-discovery directory \
     instead, and `nemo-relay pi` runs pi against it";

/// Reached only if a marketplace code path forgets to reject pi first.
macro_rules! pi_marketplace_unreachable {
    () => {
        unreachable!("{}", PI_MARKETPLACE_UNSUPPORTED)
    };
}

impl CodingAgent {
    pub(crate) const ALL: [Self; 3] = [Self::ClaudeCode, Self::Codex, Self::Pi];

    /// Every agent Relay installs *through the plugin marketplace*.
    ///
    /// Not the same set as [`Self::ALL`], and the difference is load-bearing: pi has no
    /// marketplace, so `PluginLayout::new` and everything reached from it is
    /// `pi_marketplace_unreachable!()` for it. Anything that writes or reads marketplace state
    /// for a set of hosts wants this constant, not `ALL` -- iterating `ALL` there panics.
    ///
    /// Named rather than filtered at each call site because the failure mode is silent until
    /// it is loud: `ALL` compiles fine and reads naturally, and the panic only appears once a
    /// host without a marketplace reaches the layout.
    pub(crate) const MARKETPLACE_HOSTS: [Self; 2] = [Self::ClaudeCode, Self::Codex];

    const fn descriptor(self) -> AgentDescriptor {
        match self {
            Self::ClaudeCode => claude::DESCRIPTOR,
            Self::Codex => codex::DESCRIPTOR,
            Self::Pi => pi::DESCRIPTOR,
        }
    }

    /// Canonical CLI spelling used in generated commands and configuration.
    pub(crate) const fn as_arg(self) -> &'static str {
        self.descriptor().argument
    }

    /// Canonical spelling accepted by persistent integration commands.
    pub(crate) const fn install_arg(self) -> &'static str {
        self.descriptor().install_argument
    }

    /// Human-readable product name used in diagnostics.
    pub(crate) const fn label(self) -> &'static str {
        self.descriptor().label
    }

    /// Default executable name used for discovery and transparent launch.
    pub(crate) const fn executable(self) -> &'static str {
        self.descriptor().executable
    }

    /// Stable gateway endpoint used by lifecycle hooks.
    pub(crate) const fn hook_path(self) -> &'static str {
        self.descriptor().hook_path
    }

    /// Complete lifecycle event set installed for this host.
    pub(crate) const fn hook_events(self) -> &'static [&'static str] {
        self.descriptor().hook_events
    }

    pub(crate) fn minimum_version(self) -> Version {
        let (major, minor, patch) = self.descriptor().minimum_version;
        Version::new(major, minor, patch)
    }

    pub(crate) fn version_requirement(self) -> String {
        let descriptor = self.descriptor();
        format!(
            "{} {} or newer",
            descriptor.version_product,
            self.minimum_version()
        )
    }

    /// Parses and validates the first version line emitted by the host CLI.
    pub(crate) fn validate_version_output(self, raw: &str) -> Result<Version, String> {
        let first_line = raw.lines().next().unwrap_or_default().trim();
        let version = self.parse_version(first_line).ok_or_else(|| {
            format!(
                "could not parse `{} --version` output {:?}; NeMo Relay requires {}",
                self.executable(),
                raw.trim(),
                self.version_requirement()
            )
        })?;
        if version < self.minimum_version() || !version.pre.is_empty() {
            return Err(format!(
                "{} {version} is unsupported; NeMo Relay requires {}",
                self.descriptor().version_product,
                self.version_requirement()
            ));
        }
        Ok(version)
    }

    /// Why a version passes the floor and is still not one we have run against.
    ///
    /// Deliberately not folded into `validate_version_output`: that returns an error,
    /// and an error blocks the launch (`process::launcher::validate_agent_version`).
    /// Refusing to start on a version that probably works would be worse than the
    /// silence it replaces -- the user would have to downgrade the agent to use
    /// Relay at all. This reports; it does not gate.
    pub(crate) fn unverified_version(self, version: &Version) -> Option<String> {
        let (major, minor) = self.descriptor().verified_through?;
        if (version.major, version.minor) <= (major, minor) {
            return None;
        }
        Some(format!(
            "{product} {version} is newer than the {product} {major}.{minor}.x this integration \
             was verified against; {product} can change hook shapes in a minor release, and the \
             symptom is missing spans rather than an error. Re-verify, or pin {product} {major}.\
             {minor}.x",
            product = self.descriptor().version_product,
        ))
    }

    fn parse_version(self, raw: &str) -> Option<Version> {
        match self {
            Self::ClaudeCode => claude::parse_version(raw),
            Self::Codex => codex::parse_version(raw),
            Self::Pi => pi::parse_version(raw),
        }
    }

    /// Infers a host from an executable basename.
    pub(crate) fn infer(command: &str) -> Option<Self> {
        let command = command.trim_matches(['"', '\'']);
        if command.starts_with('@') {
            return None;
        }
        let name = command
            .rsplit(['/', '\\'])
            .next()
            .unwrap_or(command)
            .to_ascii_lowercase();
        let name = [".exe", ".cmd", ".bat", ".com"]
            .into_iter()
            .find_map(|suffix| name.strip_suffix(suffix))
            .unwrap_or(&name);
        match name {
            "claude" | "claude-code" => Some(Self::ClaudeCode),
            "codex" => Some(Self::Codex),
            "pi" => Some(Self::Pi),
            _ => None,
        }
    }
}

impl crate::installation::marketplace::MarketplaceHost for CodingAgent {
    fn install_arg(self) -> &'static str {
        self.install_arg()
    }

    fn label(self) -> &'static str {
        self.label()
    }

    fn executable(self) -> &'static str {
        self.executable()
    }

    fn validate_version_output(self, output: &str) -> Result<(), String> {
        self.validate_version_output(output).map(|_| ())
    }

    fn version_requirement(self) -> String {
        self.version_requirement()
    }

    fn marketplace_manifest_relative(self) -> &'static [&'static str] {
        match self {
            Self::Codex => &[".agents", "plugins", "marketplace.json"],
            Self::ClaudeCode => &[".claude-plugin", "marketplace.json"],
            Self::Pi => pi_marketplace_unreachable!(),
        }
    }

    fn plugin_manifest_relative(self) -> &'static [&'static str] {
        match self {
            Self::Codex => &[".codex-plugin", "plugin.json"],
            Self::ClaudeCode => &[".claude-plugin", "plugin.json"],
            Self::Pi => pi_marketplace_unreachable!(),
        }
    }

    fn marketplace_manifest(self, marketplace: &str, plugin: &str) -> serde_json::Value {
        marketplace_manifest(self, marketplace, plugin)
    }

    fn plugin_manifest(self, plugin: &str) -> serde_json::Value {
        plugin_manifest(self, plugin)
    }

    fn plugin_mcp_config(self, server: serde_json::Value) -> Result<serde_json::Value, String> {
        plugin_mcp_config(self, server)
    }

    fn plugin_hooks(
        self,
        relay: &std::path::Path,
        generation_fence: &std::path::Path,
        generation_token: &str,
    ) -> Result<serde_json::Value, String> {
        let commands = crate::hooks::persistent_hook_forward_commands(
            relay,
            self,
            generation_fence,
            generation_token,
        )?;
        Ok(crate::hooks::generated_policy_hooks(self, &commands))
    }

    fn plugin_registration_args(self, plugin_id: &str) -> Vec<String> {
        match self {
            Self::Codex => vec!["plugin".into(), "add".into(), plugin_id.into()],
            Self::ClaudeCode => vec![
                "plugin".into(),
                "install".into(),
                plugin_id.into(),
                "--scope".into(),
                "user".into(),
            ],
            Self::Pi => pi_marketplace_unreachable!(),
        }
    }

    fn plugin_removal_args(self, plugin_name: &str, plugin_id: &str) -> Vec<String> {
        match self {
            Self::Codex => vec!["plugin".into(), "remove".into(), plugin_id.into()],
            Self::ClaudeCode => vec!["plugin".into(), "uninstall".into(), plugin_name.into()],
            Self::Pi => pi_marketplace_unreachable!(),
        }
    }

    fn registration_report(
        self,
        options: &crate::installation::marketplace::state::PluginInstallOptions,
        runner: &dyn crate::installation::marketplace::host::CommandRunner,
    ) -> Result<crate::installation::marketplace::host::HostRegistrationReport, String> {
        match self {
            Self::Codex => {
                crate::installation::marketplace::host::codex_registration_report(options, runner)
            }
            Self::ClaudeCode => {
                crate::installation::marketplace::host::claude_registration_report(options, runner)
            }
            Self::Pi => Err(PI_MARKETPLACE_UNSUPPORTED.to_string()),
        }
    }

    fn setup_may_mutate_before_success(self) -> bool {
        !matches!(self, Self::Codex)
    }

    fn unsafe_generation_fence_error(self, problem: &str) -> String {
        match self {
            Self::Codex => format!(
                "cannot safely replace or uninstall an existing Codex plugin because its MCP generation marker {problem}; close all Codex clients and standalone `nemo-relay mcp` processes, run `codex plugin remove nemo-relay-plugin@nemo-relay-local` and `codex plugin marketplace remove nemo-relay-local`, remove the stale marketplace and state from the selected install directory, then run `nemo-relay install codex --force` to create a fenced install (and `nemo-relay uninstall codex` afterward if removal was intended)"
            ),
            Self::ClaudeCode => format!(
                "cannot safely replace or uninstall an existing Claude Code plugin because its MCP generation marker {problem}; close all Claude Code clients and standalone `nemo-relay mcp` processes, run `claude plugin uninstall nemo-relay-plugin` and `claude plugin marketplace remove nemo-relay-local`, remove the stale marketplace and state from the selected install directory, then run `nemo-relay install claude-code --force` to create a fenced install (and `nemo-relay uninstall claude-code` afterward if removal was intended)"
            ),
            Self::Pi => pi_marketplace_unreachable!(),
        }
    }

    fn accepts_legacy_hook_only_plugin(self) -> bool {
        matches!(self, Self::ClaudeCode)
    }

    fn accepts_mcp_environment_superset(self) -> bool {
        matches!(self, Self::Codex)
    }

    fn local_install_exists(
        self,
        marketplace_root: &std::path::Path,
        plugin_root: &std::path::Path,
        plugin_manifest: &std::path::Path,
        generation_fence: &std::path::Path,
    ) -> bool {
        match self {
            Self::Codex => marketplace_root.exists(),
            Self::ClaudeCode => {
                plugin_manifest.exists()
                    || plugin_root.join(".mcp.json").exists()
                    || generation_fence.exists()
            }
            // pi installs nothing through this surface, so nothing to detect.
            Self::Pi => false,
        }
    }

    fn setup_action_description(self, action: &str) -> String {
        setup_action_description(self, action)
    }

    fn snapshot_setup(
        self,
    ) -> Result<Option<crate::installation::marketplace::PluginSetupSnapshot>, String> {
        let snapshot = snapshot_setup(self)?;
        Ok(Some(
            crate::installation::marketplace::PluginSetupSnapshot::new(move || {
                restore_setup_snapshot(&snapshot)
            }),
        ))
    }

    fn setup_plugin(
        self,
        gateway_url: &str,
        plugin_root: &std::path::Path,
        generation_token: Option<&str>,
    ) -> Result<(), String> {
        setup_marketplace_plugin(self, gateway_url, plugin_root, generation_token)
    }

    fn uninstall_plugin(
        self,
        gateway_url: &str,
        plugin_root: &std::path::Path,
    ) -> Result<(), String> {
        uninstall_marketplace_plugin(self, gateway_url, plugin_root)
    }

    fn doctor_plugin(
        self,
        gateway_url: &str,
        plugin_root: &std::path::Path,
        generation_token: Option<&str>,
    ) -> Result<(), String> {
        doctor_marketplace_plugin(self, gateway_url, plugin_root, generation_token)
    }

    fn doctor_plugin_json(
        self,
        gateway_url: &str,
        plugin_root: &std::path::Path,
    ) -> Result<serde_json::Value, String> {
        doctor_marketplace_plugin_json(self, gateway_url, plugin_root)
    }
}

pub(crate) fn marketplace_manifest(
    agent: CodingAgent,
    marketplace: &str,
    plugin: &str,
) -> serde_json::Value {
    match agent {
        CodingAgent::Codex => codex::assets::marketplace_manifest(marketplace, plugin),
        CodingAgent::ClaudeCode => claude::assets::marketplace_manifest(marketplace, plugin),
        CodingAgent::Pi => pi_marketplace_unreachable!(),
    }
}

pub(crate) fn plugin_manifest(agent: CodingAgent, plugin: &str) -> serde_json::Value {
    match agent {
        CodingAgent::Codex => codex::assets::plugin_manifest(plugin),
        CodingAgent::ClaudeCode => claude::assets::plugin_manifest(plugin),
        CodingAgent::Pi => pi_marketplace_unreachable!(),
    }
}

pub(crate) fn plugin_mcp_config(
    agent: CodingAgent,
    server: serde_json::Value,
) -> Result<serde_json::Value, String> {
    match agent {
        CodingAgent::Codex => codex::assets::mcp_config(server),
        CodingAgent::ClaudeCode => Ok(claude::assets::mcp_config(server)),
        // pi ships no MCP client, so a plugin-owned server would have no consumer.
        CodingAgent::Pi => Err(PI_MARKETPLACE_UNSUPPORTED.to_string()),
    }
}

#[cfg(test)]
pub(crate) fn codex_mcp_env_vars_from(
    environment: impl IntoIterator<Item = String>,
    config: Option<&serde_json::Value>,
) -> Vec<String> {
    codex::assets::mcp_env_vars_from(environment, config)
}

pub(crate) fn prepare_launch(
    agent: CodingAgent,
    launch: &mut crate::process::PreparedAgentLaunch,
    gateway_url: &str,
    resolved: &crate::configuration::ResolvedConfig,
    proxy_credential: &crate::provider_auth::TransparentProxyCredential,
    dry_run: bool,
) -> Result<(), crate::error::CliError> {
    launch.set_secret_env(
        crate::provider_auth::TRANSPARENT_PROXY_CREDENTIAL_ENV,
        proxy_credential.expose(),
    );
    match agent {
        CodingAgent::Codex => codex::launch::prepare(launch, gateway_url),
        CodingAgent::ClaudeCode => {
            claude::launch::prepare(launch, gateway_url, proxy_credential, dry_run)
        }
        // pi is the only agent whose launcher needs the gateway's upstream configuration: its
        // extension redirects model traffic only when the selected model already targets that
        // upstream. See `pi::launch::prepare`.
        CodingAgent::Pi => pi::launch::prepare(launch, gateway_url, &resolved.gateway),
    }
}

pub(crate) fn configured(agent: CodingAgent, configs: &crate::configuration::AgentConfigs) -> bool {
    config(agent, configs).command.is_some()
}

pub(crate) const fn config(
    agent: CodingAgent,
    configs: &crate::configuration::AgentConfigs,
) -> &crate::configuration::AgentCommandConfig {
    match agent {
        CodingAgent::ClaudeCode => &configs.claude,
        CodingAgent::Codex => &configs.codex,
        CodingAgent::Pi => &configs.pi,
    }
}

pub(crate) fn hook_status(
    agent: CodingAgent,
    _configs: &crate::configuration::AgentConfigs,
) -> Result<String, String> {
    match agent {
        CodingAgent::Codex => codex::doctor::hook_status(),
        CodingAgent::ClaudeCode => claude::doctor::hook_status(),
        CodingAgent::Pi => pi::doctor::hook_status(),
    }
}

pub(crate) enum SetupSnapshot {
    Codex(CodexSetupSnapshot),
    Claude(ClaudeSetupSnapshot),
}

pub(crate) fn setup_action_description(agent: CodingAgent, action: &str) -> String {
    match (agent, action) {
        (CodingAgent::Codex, "configure") => {
            "configure Codex provider and trust plugin-owned hooks".into()
        }
        (CodingAgent::Codex, "restore") => "remove Codex provider and plugin hook trust".into(),
        (CodingAgent::Codex, "doctor") => "check Codex provider and plugin-owned hooks".into(),
        (CodingAgent::ClaudeCode, "configure") => {
            "enable Claude Code provider routing through NeMo Relay".into()
        }
        (CodingAgent::ClaudeCode, "restore") => {
            "restore Claude Code provider routing from NeMo Relay backup".into()
        }
        (CodingAgent::ClaudeCode, "doctor") => "check Claude Code provider routing".into(),
        _ => unreachable!("unsupported setup action"),
    }
}

pub(crate) fn snapshot_setup(agent: CodingAgent) -> Result<SetupSnapshot, String> {
    match agent {
        CodingAgent::Codex => snapshot_codex_setup().map(SetupSnapshot::Codex),
        CodingAgent::ClaudeCode => snapshot_claude_setup().map(SetupSnapshot::Claude),
        CodingAgent::Pi => Err(PI_MARKETPLACE_UNSUPPORTED.to_string()),
    }
}

pub(crate) fn restore_setup_snapshot(snapshot: &SetupSnapshot) -> Result<(), String> {
    match snapshot {
        SetupSnapshot::Codex(snapshot) => restore_codex_setup(snapshot),
        SetupSnapshot::Claude(snapshot) => restore_claude_setup(snapshot),
    }
}

pub(crate) fn setup_marketplace_plugin(
    agent: CodingAgent,
    gateway_url: &str,
    plugin_root: &Path,
    generation_token: Option<&str>,
) -> Result<(), String> {
    match agent {
        CodingAgent::Codex => {
            install_codex_plugin_with_generation(gateway_url, plugin_root, generation_token)
        }
        CodingAgent::ClaudeCode => enable_claude_provider(gateway_url),
        CodingAgent::Pi => Err(PI_MARKETPLACE_UNSUPPORTED.to_string()),
    }
}

pub(crate) fn uninstall_marketplace_plugin(
    agent: CodingAgent,
    gateway_url: &str,
    plugin_root: &Path,
) -> Result<(), String> {
    match agent {
        CodingAgent::Codex => uninstall_codex_plugin(gateway_url, plugin_root),
        CodingAgent::ClaudeCode => restore_claude_provider(gateway_url),
        CodingAgent::Pi => Err(PI_MARKETPLACE_UNSUPPORTED.to_string()),
    }
}

pub(crate) fn doctor_marketplace_plugin(
    agent: CodingAgent,
    gateway_url: &str,
    plugin_root: &Path,
    generation_token: Option<&str>,
) -> Result<(), String> {
    match agent {
        CodingAgent::Codex => doctor_plugin_with_generation(
            CodingAgent::Codex,
            gateway_url,
            plugin_root,
            generation_token,
        ),
        CodingAgent::ClaudeCode => doctor_plugin(CodingAgent::ClaudeCode, gateway_url, plugin_root),
        CodingAgent::Pi => Err(PI_MARKETPLACE_UNSUPPORTED.to_string()),
    }
}

pub(crate) fn doctor_marketplace_plugin_json(
    agent: CodingAgent,
    gateway_url: &str,
    plugin_root: &Path,
) -> Result<Value, String> {
    match agent {
        CodingAgent::Codex => doctor_plugin_json(CodingAgent::Codex, gateway_url, plugin_root),
        CodingAgent::ClaudeCode => {
            doctor_plugin_json(CodingAgent::ClaudeCode, gateway_url, plugin_root)
        }
        CodingAgent::Pi => Err(PI_MARKETPLACE_UNSUPPORTED.to_string()),
    }
}

pub(crate) fn install_integration(
    agent: CodingAgent,
    command: crate::installation::InstallRequest,
) -> Result<std::process::ExitCode, crate::error::CliError> {
    match agent {
        CodingAgent::Codex => codex::install::install(command),
        CodingAgent::ClaudeCode => claude::install::install(command),
        // Not a marketplace install: pi auto-discovers a package directory under its own
        // agent directory, so Relay writes one there instead of synthesizing a manifest.
        CodingAgent::Pi => pi::install::install(command),
    }
}

pub(crate) fn uninstall_integration(
    agent: CodingAgent,
    command: crate::installation::UninstallRequest,
) -> Result<std::process::ExitCode, crate::error::CliError> {
    match agent {
        CodingAgent::Codex => codex::install::uninstall(command),
        CodingAgent::ClaudeCode => claude::install::uninstall(command),
        CodingAgent::Pi => pi::install::uninstall(command),
    }
}

pub(crate) fn detected_install_integrations(candidates: &[CodingAgent]) -> Vec<CodingAgent> {
    candidates
        .iter()
        .copied()
        .filter(|agent| crate::process::resolve_executable(agent.executable()).is_some())
        .collect()
}

/// Hosts with **marketplace plugin state** under `install_dir`.
///
/// With `include_local_install`, force-cleanup targets count as state too, so a stale local
/// installation is still reachable by `uninstall --force`.
///
/// ⚠️ **pi is never in this list, and putting it there panics the doctor.** Every consumer
/// feeds the result into marketplace-only code -- `PluginLayout::new` calls
/// `marketplace_manifest_relative`, which is `unreachable!()` for pi -- so a pi here is not
/// a wrong answer, it is an abort. That regression shipped once: teaching this function
/// about pi so `uninstall all` could see a managed install also put pi in front of the
/// readiness collector, and a single `nemo-relay install pi` then panicked `doctor` for
/// *every* agent.
///
/// Use [`uninstallable_integrations`] for "what can Relay remove", which is the question
/// that has a pi answer.
pub(crate) fn installed_integrations(
    candidates: &[CodingAgent],
    install_dir: Option<&Path>,
    include_local_install: bool,
) -> Vec<CodingAgent> {
    let install_dir = install_dir
        .map(Path::to_path_buf)
        .unwrap_or_else(crate::installation::marketplace::default_marketplace_install_dir);
    candidates
        .iter()
        .copied()
        .filter(|agent| !matches!(agent, CodingAgent::Pi))
        .filter(|agent| {
            crate::installation::marketplace::persisted_state_exists(*agent, &install_dir)
                || (include_local_install
                    && crate::installation::marketplace::force_cleanup_target_exists(
                        *agent,
                        &install_dir,
                    ))
        })
        .collect()
}

/// Hosts with integration state Relay can remove, which is a wider question.
///
/// Marketplace state for Codex and Claude Code, and a Relay-managed extension directory for
/// pi. Separate from [`installed_integrations`] because only one of the two questions has a
/// pi answer, and answering both from one function is what broke `doctor`.
///
/// `include_local_install` is forwarded to [`installed_integrations`]. It does not gate pi:
/// a Relay-managed extension directory *is* the local install, so it is always removable.
pub(crate) fn uninstallable_integrations(
    candidates: &[CodingAgent],
    install_dir: Option<&Path>,
    include_local_install: bool,
) -> Vec<CodingAgent> {
    let mut agents = installed_integrations(candidates, install_dir, include_local_install);
    if candidates.contains(&CodingAgent::Pi) && pi::install::is_installed() {
        agents.push(CodingAgent::Pi);
    }
    agents
}

pub(crate) fn doctor_integration(
    agent: CodingAgent,
    options: &crate::installation::marketplace::state::PluginInstallOptions,
) -> Result<(), crate::error::CliError> {
    crate::installation::marketplace::doctor_marketplace_integration(agent, options)
}

pub(crate) fn doctor_integration_report(
    agent: CodingAgent,
    options: &crate::installation::marketplace::state::PluginInstallOptions,
) -> Result<serde_json::Value, crate::error::CliError> {
    crate::installation::marketplace::doctor_marketplace_report(agent, options)
}

struct PendingIntegrationReadiness {
    agent: CodingAgent,
    state_path: PathBuf,
    receiver: std::sync::mpsc::Receiver<crate::installation::marketplace::HostPluginReadiness>,
}

pub(crate) fn collect_default_integration_readiness()
-> Vec<crate::installation::marketplace::HostPluginReadiness> {
    const TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

    let install_dir = crate::installation::marketplace::default_marketplace_install_dir();
    let agents = installed_integrations(&CodingAgent::ALL, Some(&install_dir), false);
    let pending = agents
        .into_iter()
        .map(|agent| spawn_integration_readiness(agent, install_dir.clone()))
        .collect::<Vec<_>>();
    let deadline = std::time::Instant::now() + TIMEOUT;
    pending
        .into_iter()
        .map(|pending| {
            receive_integration_readiness(
                pending,
                deadline.saturating_duration_since(std::time::Instant::now()),
                &install_dir,
            )
        })
        .collect()
}

fn receive_integration_readiness(
    pending: PendingIntegrationReadiness,
    timeout: std::time::Duration,
    install_dir: &Path,
) -> crate::installation::marketplace::HostPluginReadiness {
    match pending.receiver.recv_timeout(timeout) {
        Ok(readiness) => readiness,
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => failed_integration_readiness(
            pending.agent,
            pending.state_path,
            install_dir,
            "timed out while collecting persistent-integration readiness",
        ),
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => failed_integration_readiness(
            pending.agent,
            pending.state_path,
            install_dir,
            "persistent-integration readiness collector stopped unexpectedly",
        ),
    }
}

#[cfg(test)]
pub(crate) fn receive_integration_readiness_for_test(
    agent: CodingAgent,
    state_path: PathBuf,
    receiver: std::sync::mpsc::Receiver<crate::installation::marketplace::HostPluginReadiness>,
    install_dir: &Path,
    timeout: std::time::Duration,
) -> crate::installation::marketplace::HostPluginReadiness {
    receive_integration_readiness(
        PendingIntegrationReadiness {
            agent,
            state_path,
            receiver,
        },
        timeout,
        install_dir,
    )
}

fn spawn_integration_readiness(
    agent: CodingAgent,
    install_dir: PathBuf,
) -> PendingIntegrationReadiness {
    let state_path = crate::installation::marketplace::marketplace_state_path(agent, &install_dir);
    let worker_install_dir = install_dir.clone();
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    std::thread::spawn(move || {
        let options = crate::installation::marketplace::state::PluginInstallOptions {
            install_dir: worker_install_dir,
            operation_lock_dir: PathBuf::new(),
            force: false,
            dry_run: false,
            skip_doctor: true,
        };
        let runner = crate::installation::marketplace::host::RealCommandRunner;
        let readiness = crate::installation::marketplace::collect_marketplace_readiness(
            agent, &options, &runner,
        );
        let _ = sender.send(readiness);
    });
    PendingIntegrationReadiness {
        agent,
        state_path,
        receiver,
    }
}

fn failed_integration_readiness(
    agent: CodingAgent,
    state_path: PathBuf,
    install_dir: &Path,
    details: &str,
) -> crate::installation::marketplace::HostPluginReadiness {
    let (marketplace, plugin) =
        crate::installation::marketplace::marketplace_install_roots(agent, install_dir);
    let mut readiness = crate::installation::marketplace::HostPluginReadiness {
        host: agent.install_arg().to_string(),
        remediation: format!("nemo-relay install {} --force", agent.install_arg()),
        state_path,
        marketplace: Some(marketplace),
        plugin: Some(plugin),
        checks: Vec::new(),
        relay: None,
        host_plugin_registered: None,
        host_marketplace_registered: None,
        plugin_setup: None,
    };
    readiness.push("Host readiness", Err(details.to_string()));
    readiness
}

pub(crate) use crate::process::portable_executable_path;
#[cfg(any(not(windows), test))]
pub(crate) use crate::process::shell_quote_arg_for_platform;
#[cfg(test)]
pub(crate) use crate::process::strip_windows_verbatim_prefix;
pub(crate) use claude::host::{ClaudeSetupSnapshot, restore_claude_setup, snapshot_claude_setup};
pub(crate) use codex::host::{CodexSetupSnapshot, restore_codex_setup, snapshot_codex_setup};

use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use claude::host::claude_settings_base_url;
use codex::host::{
    codex_hook_trust_report, codex_hook_trust_report_with_generation, codex_hooks_installed,
    codex_hooks_installed_with_generation, codex_provider_installed, empty_codex_hook_trust_report,
    install_codex_with_generation, uninstall_codex,
};
use shared::host::{current_exe, healthz, print_check, print_info};

#[cfg(test)]
pub(super) use crate::bootstrap::DEFAULT_URL;

pub(crate) fn install_codex_plugin_with_generation(
    gateway_url: &str,
    plugin_root: &Path,
    generation_token: Option<&str>,
) -> Result<(), String> {
    install_codex_with_generation(
        gateway_url,
        &plugin_root.join("hooks").join("hooks.json"),
        generation_token,
    )
    .map(|_| ())
}

pub(crate) fn uninstall_codex_plugin(gateway_url: &str, plugin_root: &Path) -> Result<(), String> {
    uninstall_codex(gateway_url, &plugin_root.join("hooks").join("hooks.json")).map(|_| ())
}

pub(crate) fn enable_claude_provider(gateway_url: &str) -> Result<(), String> {
    claude::host::enable_claude_provider(gateway_url)
}

pub(crate) fn restore_claude_provider(gateway_url: &str) -> Result<(), String> {
    claude::host::restore_claude_provider(gateway_url)
}

pub(crate) fn doctor_plugin(
    agent: CodingAgent,
    gateway_url: &str,
    plugin_root: &Path,
) -> Result<(), String> {
    doctor_plugin_with_generation(agent, gateway_url, plugin_root, None)
}

pub(crate) fn doctor_plugin_with_generation(
    agent: CodingAgent,
    gateway_url: &str,
    plugin_root: &Path,
    generation_token: Option<&str>,
) -> Result<(), String> {
    if doctor_ok(
        agent,
        gateway_url,
        Some(&plugin_root.join("hooks").join("hooks.json")),
        generation_token,
    )? {
        Ok(())
    } else {
        Err(format!("{} plugin doctor checks failed", agent.as_arg()))
    }
}

pub(crate) fn doctor_plugin_json(
    agent: CodingAgent,
    gateway_url: &str,
    plugin_root: &Path,
) -> Result<Value, String> {
    let plugin_binary = current_exe().ok().is_some_and(|path| path.exists());
    let sidecar_running = healthz(gateway_url);
    let (checks, ok, codex_trust) = match agent {
        CodingAgent::ClaudeCode => {
            let provider = claude_settings_base_url().as_deref() == Some(gateway_url);
            (
                json!({
                    "plugin_binary": plugin_binary,
                    "sidecar_running": sidecar_running,
                    "claude_provider_routing": provider
                }),
                plugin_binary && provider,
                None,
            )
        }
        CodingAgent::Codex => {
            let plugin_hooks_path = plugin_root.join("hooks").join("hooks.json");
            let provider = codex_provider_installed(gateway_url);
            let hooks = codex_hooks_installed(&plugin_hooks_path)?;
            let trust = if hooks {
                codex_hook_trust_report(&plugin_hooks_path)?
            } else {
                empty_codex_hook_trust_report()
            };
            let hooks_trusted = trust.ready();
            (
                json!({
                    "plugin_binary": plugin_binary,
                    "sidecar_running": sidecar_running,
                    "codex_provider_alias": provider,
                    "codex_hooks": hooks,
                    "codex_hooks_trusted": hooks_trusted
                }),
                plugin_binary && provider && hooks && hooks_trusted,
                Some(trust),
            )
        }
        // pi's hooks live inside a user-loaded extension rather than in a
        // NeMo Relay-managed plugin root, so the only checkable fact here is
        // whether that extension is discoverable.
        CodingAgent::Pi => {
            let extension = pi::doctor::extension_configured();
            (
                json!({
                    "plugin_binary": plugin_binary,
                    "sidecar_running": sidecar_running,
                    "pi_extension_located": extension
                }),
                plugin_binary && extension,
                None,
            )
        }
    };
    let mut report = json!({
        "ok": ok,
        "sidecar_health": if sidecar_running {
            "running"
        } else {
            "not_running_mcp_start"
        },
        "checks": checks
    });
    if let Some(trust) = codex_trust {
        report["codex_hook_trust"] = trust.to_json();
    }
    Ok(report)
}

fn doctor_ok(
    agent: CodingAgent,
    gateway_url: &str,
    plugin_hooks_path: Option<&Path>,
    generation_token: Option<&str>,
) -> Result<bool, String> {
    let mut ok = true;
    ok &= print_check(
        "plugin binary",
        current_exe().ok().is_some_and(|path| path.exists()),
    );
    if healthz(gateway_url) {
        print_info("sidecar health", "running");
    } else {
        print_info(
            "sidecar health",
            "not running; the plugin MCP starts it when the host launches",
        );
    }
    match agent {
        CodingAgent::ClaudeCode => {
            ok &= print_check(
                "claude provider routing",
                claude_settings_base_url().as_deref() == Some(gateway_url),
            );
        }
        CodingAgent::Codex => {
            let plugin_hooks_path = plugin_hooks_path
                .ok_or_else(|| "Codex plugin hooks path is required for doctor".to_string())?;
            let provider = codex_provider_installed(gateway_url);
            let hooks = codex_hooks_installed_with_generation(plugin_hooks_path, generation_token)?;
            ok &= print_check("codex provider alias", provider);
            ok &= print_check("codex hooks", hooks);
            let trust = if hooks {
                codex_hook_trust_report_with_generation(plugin_hooks_path, generation_token)?
            } else {
                empty_codex_hook_trust_report()
            };
            ok &= print_check("codex hooks trusted and enabled", trust.ready());
            if !trust.ready() {
                print_info("codex hook trust", &trust.summary());
            }
        }
        CodingAgent::Pi => {
            ok &= print_check("pi extension located", pi::doctor::extension_configured());
            print_info("pi hooks", &pi::doctor::hook_status()?);
        }
    }
    Ok(ok)
}

#[cfg(test)]
use crate::bootstrap::*;
#[cfg(test)]
use crate::hooks::generated_hooks;
#[cfg(test)]
use claude::host::*;
#[cfg(test)]
use codex::app_server::*;
#[cfg(test)]
use codex::host::*;
#[cfg(test)]
use shared::host::*;

#[cfg(test)]
#[path = "../../tests/coverage/agents/plugin_host_tests.rs"]
mod host_tests;

#[cfg(test)]
#[path = "../../tests/coverage/agents/coding_agent_tests.rs"]
mod tests;
