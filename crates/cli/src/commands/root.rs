// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use clap::{Parser, Subcommand, ValueEnum};

use super::completions::CompletionsCommand;
use super::configure::ConfigCommand;
use super::diagnostics::{AgentsCommand, DoctorCommand};
use super::hook_forward::HookForwardCommand;
use super::install::{InstallCommand, UninstallCommand};
use super::logging::LoggingArgs;
use super::model_pricing::PricingCommand;
use super::plugins::PluginsCommand;
use super::run::{EasyPathCommand, RunCommand};
use super::serve::ServerArgs;
use crate::agents::CodingAgent;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub(crate) enum AgentArg {
    #[value(name = "claude", alias = "claude-code")]
    Claude,
    Codex,
    Hermes,
}

impl From<AgentArg> for CodingAgent {
    fn from(value: AgentArg) -> Self {
        match value {
            AgentArg::Claude => Self::ClaudeCode,
            AgentArg::Codex => Self::Codex,
            AgentArg::Hermes => Self::Hermes,
        }
    }
}

#[derive(Debug, Clone, Parser)]
#[command(name = "nemo-relay")]
#[command(about = "Coding-agent gateway for NeMo Relay observability")]
#[command(version)]
pub(crate) struct Cli {
    #[command(flatten)]
    pub(crate) server: ServerArgs,
    #[command(flatten)]
    pub(super) logging: LoggingArgs,
    #[command(subcommand)]
    pub(crate) command: Option<Command>,
}

#[derive(Debug, Clone, Subcommand)]
pub(crate) enum Command {
    /// Run Claude Code with observability (setup on first use)
    #[command(
        long_about = "Run Anthropic's `claude` CLI under an ephemeral NeMo Relay gateway. \
                      Observability (ATIF + OpenInference) is wired in transparently via \
                      ANTHROPIC_BASE_URL. First-time use launches the setup wizard so the \
                      `[agents.claude]` block lands in `.nemo-relay/config.toml` and observation \
                      starts on the next invocation without prompts.",
        after_help = "Examples:\n  \
                      nemo-relay claude\n  \
                      nemo-relay claude -- chat \"refactor the launcher\"\n  \
                      nemo-relay claude -- --resume <session-id>"
    )]
    Claude(EasyPathCommand),
    /// Run Codex with observability (setup on first use)
    #[command(
        long_about = "Run OpenAI's `codex` CLI under an ephemeral NeMo Relay gateway. NeMo Relay \
                      injects a `nemo-relay-openai` provider override so codex points at the \
                      gateway; the gateway then forwards to `--openai-base-url` (defaults to \
                      api.openai.com) with `OPENAI_API_KEY` injected on the codex route (see \
                      NMF-86 — codex's own auth.json JWT is stripped). The supported host version \
                      is validated before launch.",
        after_help = "Examples:\n  \
                      nemo-relay codex\n  \
                      nemo-relay codex -- exec \"fix the bug in foo.rs\"\n  \
                      nemo-relay --openai-base-url https://inference-api.nvidia.com codex"
    )]
    Codex(EasyPathCommand),
    /// Run Hermes with observability (setup on first use)
    #[command(
        long_about = "Run Hermes Agent under an ephemeral NeMo Relay gateway. The wrapper uses a \
                      process-private HERMES_HOME overlay for dynamic hooks, without rewriting \
                      the user's Hermes configuration. Use `nemo-relay install hermes` when bare \
                      Hermes processes should load the shared native Relay gateway on \
                      127.0.0.1:47632 through MCP.",
        after_help = "Examples:\n  \
                      nemo-relay hermes\n  \
                      nemo-relay hermes -- chat --provider custom"
    )]
    Hermes(EasyPathCommand),
    /// Keep a shared Relay gateway ready for an MCP client.
    #[command(
        long_about = "Start or reuse a shared native NeMo Relay gateway for an MCP stdio \
                      connection. The command acquires the gateway immediately, before reading \
                      MCP protocol frames. The gateway binds 127.0.0.1:47632 by default and MCP \
                      initialization completes only after Relay identity and readiness are \
                      verified. Multiple MCP clients share the gateway; it remains available \
                      until its idle timeout after the final client closes. This command \
                      advertises no MCP tools.",
        after_help = "Examples:\n  \
                      nemo-relay mcp\n  \
                      nemo-relay --bind 127.0.0.1:4041 mcp  # explicit standalone/test bind"
    )]
    Mcp,
    /// Run the interactive setup (writes `.nemo-relay/config.toml`)
    Config(ConfigCommand),
    /// Create or edit plugin configuration (writes `plugins.toml`)
    Plugins(PluginsCommand),
    /// Install coding-agent plugins from the local nemo-relay CLI.
    Install(InstallCommand),
    /// Uninstall coding-agent plugins installed by `nemo-relay install`.
    Uninstall(UninstallCommand),
    /// Validate and configure model pricing catalogs.
    ModelPricing(PricingCommand),
    /// Diagnose env, agents, config, observability (optionally scoped to one agent)
    Doctor(DoctorCommand),
    /// List supported and locally-detected agents (use `--json` for machine output)
    Agents(AgentsCommand),
    /// Print shell completion script (e.g. `nemo-relay completions zsh > ~/.zfunc/_nemo-relay`)
    Completions(CompletionsCommand),
    /// Run an agent deterministically (no wizard; errors if config is missing)
    Run(RunCommand),
    /// Internal: subprocess used by installed hooks to forward events. Not typed by humans.
    #[command(hide = true)]
    HookForward(HookForwardCommand),
}

impl Command {
    pub(crate) fn log_name(&self) -> &'static str {
        match self {
            Self::Claude(_) => "claude",
            Self::Codex(_) => "codex",
            Self::Hermes(_) => "hermes",
            Self::Mcp => "mcp",
            Self::Config(_) => "config",
            Self::Plugins(_) => "plugins",
            Self::Install(_) => "install",
            Self::Uninstall(_) => "uninstall",
            Self::ModelPricing(_) => "model_pricing",
            Self::Doctor(_) => "doctor",
            Self::Agents(_) => "agents",
            Self::Completions(_) => "completions",
            Self::Run(_) => "run",
            Self::HookForward(_) => "hook_forward",
        }
    }

    /// Configuration-editing commands remain available even when operational logging settings are
    /// invalid, so users can repair their configuration.
    pub(crate) fn skips_logging(&self) -> bool {
        matches!(self, Self::Config(_))
            || matches!(self, Self::Plugins(command) if command.is_edit())
            || matches!(self, Self::HookForward(command) if transparent_hook_is_inert(command))
    }
}

fn transparent_hook_is_inert(command: &HookForwardCommand) -> bool {
    !command.transparent_run
        && std::env::var(crate::configuration::TRANSPARENT_RUN_ENV)
            .ok()
            .as_deref()
            == Some("1")
}
