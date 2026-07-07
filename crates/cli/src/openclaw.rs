// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenClaw transparent-launch preparation.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value, json};

use crate::config::GatewayConfig;
use crate::error::CliError;

const OPENCLAW_CONFIG_PATH: &str = "OPENCLAW_CONFIG_PATH";
const RELAY_OPENCLAW_PLUGIN_ID: &str = "nemo-relay";
const RELAY_OPENCLAW_BRIDGE_ID: &str = "nemo-relay-cli-bridge";
const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_ANTHROPIC_BASE_URL: &str = "https://api.anthropic.com";

const BRIDGE_PACKAGE_JSON: &str = r#"{
  "name": "nemo-relay-cli-openclaw-bridge",
  "version": "0.0.0",
  "private": true,
  "type": "module",
  "openclaw": {
    "extensions": ["./index.js"]
  }
}
"#;

const BRIDGE_MANIFEST_JSON: &str = r#"{
  "id": "nemo-relay-cli-bridge",
  "name": "NeMo Relay CLI Bridge",
  "description": "Temporary hook bridge owned by the NeMo Relay CLI wrapper.",
  "activation": {"onStartup": true},
  "configSchema": {
    "type": "object",
    "additionalProperties": false,
    "properties": {}
  }
}
"#;

const BRIDGE_INDEX_JS: &str = r#"const HOOK_EVENT_NAMES = {
  session_start: "session_start",
  session_end: "session_end",
  llm_input: "preLlmCall",
  before_tool_call: "tool_start",
  after_tool_call: "tool_end",
  subagent_spawned: "subagent_start",
  subagent_ended: "subagent_end",
};

let warned = false;

function firstString(...values) {
  return values.find((value) => typeof value === "string" && value.length > 0);
}

function payloadFor(hookName, event = {}, ctx = {}) {
  const sessionId = firstString(
    event.sessionId,
    ctx.sessionId,
    event.sessionKey,
    ctx.sessionKey,
    event.requesterSessionKey,
    ctx.requesterSessionKey,
    event.runId,
    ctx.runId,
  );
  const toolCallId = firstString(event.toolCallId, ctx.toolCallId);
  const toolName = firstString(event.toolName, ctx.toolName);
  return {
    ...event,
    hook_event_name: HOOK_EVENT_NAMES[hookName] ?? hookName,
    session_id: sessionId,
    conversation_id: firstString(event.sessionKey, ctx.sessionKey, sessionId),
    request_id: firstString(event.callId, event.requestId, event.runId, ctx.runId),
    generation_id: firstString(event.callId, event.runId, ctx.runId),
    agent_id: firstString(event.agentId, ctx.agentId),
    subagent_id: firstString(event.childSessionKey, event.targetSessionKey),
    tool_call_id: toolCallId,
    tool_name: toolName,
    tool_input: event.params,
    tool_output: event.result,
    status: event.error ? "error" : undefined,
  };
}

async function forward(api, hookName, event, ctx) {
  const baseUrl = process.env.NEMO_RELAY_GATEWAY_URL;
  if (!baseUrl) return;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 2000);
  timer.unref?.();
  try {
    const response = await fetch(`${baseUrl.replace(/\/$/, "")}/hooks/openclaw`, {
      method: "POST",
      headers: {"content-type": "application/json"},
      body: JSON.stringify(payloadFor(hookName, event, ctx)),
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
  } catch (error) {
    if (!warned) {
      warned = true;
      api.logger.warn?.(`NeMo Relay CLI hook forwarding degraded: ${String(error)}`);
    }
  } finally {
    clearTimeout(timer);
  }
}

export default {
  id: "nemo-relay-cli-bridge",
  name: "NeMo Relay CLI Bridge",
  register(api) {
    for (const hookName of Object.keys(HOOK_EVENT_NAMES)) {
      api.on(hookName, (event, ctx) => forward(api, hookName, event, ctx));
    }
  },
};
"#;

/// Records which Relay upstream flags were explicitly supplied by the caller.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ExplicitUpstreams {
    pub(crate) openai: bool,
    pub(crate) anthropic: bool,
}

/// Files, environment, and status notes added to an OpenClaw child process.
#[derive(Debug, Default)]
pub(crate) struct Preparation {
    pub(crate) env: Vec<(String, String)>,
    pub(crate) temp_dirs: Vec<PathBuf>,
    pub(crate) temp_files: Vec<PathBuf>,
    pub(crate) notes: Vec<String>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum AuthProfileEvidence {
    /// The read-only OpenClaw probe found no saved profiles for this provider.
    #[default]
    None,
    /// Every profile OpenClaw may select is an API key.
    ApiKeyOnly,
    /// Some agents have API-key profiles and the others have no saved profile.
    ApiKeyOrNone,
    /// At least one profile is OAuth/token-based, so selection or rotation is not deterministic.
    NonApiKeyOrMixed,
    /// The profile inventory could not be read safely.
    Unknown,
}

#[derive(Debug, Clone, Copy, Default)]
struct AuthProfileEvidenceSet {
    openai: AuthProfileEvidence,
    anthropic: AuthProfileEvidence,
}

#[derive(Debug, Clone, Copy, Default)]
struct TrustedApiKeySet {
    openai: bool,
    anthropic: bool,
}

impl TrustedApiKeySet {
    const fn get(self, family: ProviderFamily) -> bool {
        match family {
            ProviderFamily::OpenAi => self.openai,
            ProviderFamily::Anthropic => self.anthropic,
        }
    }
}

impl AuthProfileEvidenceSet {
    const fn get(self, family: ProviderFamily) -> AuthProfileEvidence {
        match family {
            ProviderFamily::OpenAi => self.openai,
            ProviderFamily::Anthropic => self.anthropic,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProviderFamily {
    OpenAi,
    Anthropic,
}

impl ProviderFamily {
    const fn id(self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::Anthropic => "anthropic",
        }
    }

    const fn default_api(self) -> &'static str {
        match self {
            Self::OpenAi => "openai-responses",
            Self::Anthropic => "anthropic-messages",
        }
    }

    const fn default_base_url(self) -> &'static str {
        match self {
            Self::OpenAi => DEFAULT_OPENAI_BASE_URL,
            Self::Anthropic => DEFAULT_ANTHROPIC_BASE_URL,
        }
    }

    fn is_api_key_name(self, name: &str) -> bool {
        let exact = match self {
            Self::OpenAi => [
                "OPENAI_API_KEY",
                "OPENAI_API_KEYS",
                "OPENCLAW_LIVE_OPENAI_KEY",
            ],
            Self::Anthropic => [
                "ANTHROPIC_API_KEY",
                "ANTHROPIC_API_KEYS",
                "OPENCLAW_LIVE_ANTHROPIC_KEY",
            ],
        };
        exact.contains(&name)
            || match self {
                Self::OpenAi => name.starts_with("OPENAI_API_KEY_"),
                Self::Anthropic => name.starts_with("ANTHROPIC_API_KEY_"),
            }
    }

    fn has_api_key_environment(self) -> bool {
        std::env::vars_os().any(|(name, value)| {
            self.is_api_key_name(&name.to_string_lossy())
                && !value.to_string_lossy().trim().is_empty()
        })
    }
}

#[derive(Debug)]
struct RoutingPlan {
    provider_overrides: Map<String, Value>,
    openai_upstream: Option<String>,
    anthropic_upstream: Option<String>,
    notes: Vec<String>,
}

/// Prepare an OpenClaw process that owns the model request path.
///
/// OpenClaw's `agent` and `tui` commands normally attach to a separately running
/// gateway. Those modes would leave the temporary config in the client process,
/// not the model-serving process, so this function forces their embedded local
/// modes. A foreground `gateway run` is also supported. Remote and detached
/// gateway modes fail before Relay claims interception.
pub(crate) fn prepare(
    argv: &mut Vec<String>,
    gateway_url: &str,
    gateway: &mut GatewayConfig,
    explicit_upstreams: ExplicitUpstreams,
    dry_run: bool,
) -> Result<Preparation, CliError> {
    normalize_foreground_argv(argv)?;
    add_invocation_metadata(gateway);

    let source_path = resolve_config_path(argv)?;
    let (source_config, source_exists) = read_source_config(&source_path)?;
    let disable_embedded_relay = source_config.as_ref().is_some_and(|config| {
        config
            .pointer("/plugins/entries")
            .and_then(Value::as_object)
            .is_some_and(|entries| entries.contains_key(RELAY_OPENCLAW_PLUGIN_ID))
    }) || probe_relay_plugin_installed(argv, &source_path);
    let auth_profiles = AuthProfileEvidenceSet {
        openai: probe_auth_profiles(
            argv,
            &source_path,
            source_config.as_ref(),
            ProviderFamily::OpenAi,
        ),
        anthropic: probe_auth_profiles(
            argv,
            &source_path,
            source_config.as_ref(),
            ProviderFamily::Anthropic,
        ),
    };
    let trusted_api_keys = trusted_api_key_sources(argv, source_config.as_ref());
    let mut routing = build_routing_plan_with_evidence(
        source_config.as_ref(),
        gateway_url,
        auth_profiles,
        trusted_api_keys,
    );
    if let Some(upstream) = routing.openai_upstream.take() {
        if explicit_upstreams.openai {
            routing.notes.push(
                "explicit Relay OpenAI upstream overrides the OpenClaw provider endpoint".into(),
            );
        } else {
            gateway.openai_base_url = upstream;
        }
    }
    if let Some(upstream) = routing.anthropic_upstream.take() {
        if explicit_upstreams.anthropic {
            routing.notes.push(
                "explicit Relay Anthropic upstream overrides the OpenClaw provider endpoint".into(),
            );
        } else {
            gateway.anthropic_base_url = upstream;
        }
    }

    if dry_run {
        let mut notes = routing.notes;
        notes.push(format!(
            "would generate a temporary OpenClaw JSON5 overlay and CLI hook bridge for {}",
            source_path.display()
        ));
        return Ok(Preparation {
            env: vec![(
                OPENCLAW_CONFIG_PATH.into(),
                "<temporary-openclaw-config-overlay>".into(),
            )],
            temp_dirs: Vec::new(),
            temp_files: Vec::new(),
            notes,
        });
    }

    let bridge_dir = write_bridge_plugin()?;
    let overlay = overlay_document(
        source_exists.then_some(source_path.as_path()),
        source_config.as_ref(),
        routing.provider_overrides,
        &bridge_dir,
        disable_embedded_relay,
    );
    let overlay_path = match temporary_overlay_path(&source_path) {
        Ok(path) => path,
        Err(error) => {
            let _ = std::fs::remove_dir_all(&bridge_dir);
            return Err(error);
        }
    };
    let write_result = serde_json::to_vec_pretty(&overlay)
        .map_err(|error| CliError::Launch(format!("could not serialize OpenClaw overlay: {error}")))
        .and_then(|contents| std::fs::write(&overlay_path, contents).map_err(CliError::from));
    if let Err(error) = write_result {
        let _ = std::fs::remove_file(&overlay_path);
        let _ = std::fs::remove_dir_all(&bridge_dir);
        return Err(error);
    }

    let env = vec![(
        OPENCLAW_CONFIG_PATH.into(),
        overlay_path.display().to_string(),
    )];
    Ok(Preparation {
        env,
        temp_dirs: vec![bridge_dir],
        temp_files: vec![overlay_path],
        notes: routing
            .notes
            .into_iter()
            .chain(["OpenClaw hooks forwarded to the CLI-owned Relay runtime".into()])
            .collect(),
    })
}

fn normalize_foreground_argv(argv: &mut Vec<String>) -> Result<(), CliError> {
    let executable = openclaw_executable_index(argv).ok_or_else(|| {
        CliError::Launch(
            "could not locate the OpenClaw executable in the configured launch prefix; use `openclaw`, `npx openclaw`, or a supported package-manager exec form"
                .into(),
        )
    })?;
    if has_container_option(&argv[executable + 1..]) {
        return Err(CliError::Launch(
            "OpenClaw --container execution cannot use Relay's host-side temporary provider overlay; run OpenClaw directly or launch a foreground host process"
                .into(),
        ));
    }
    let command = first_openclaw_positional(argv, executable)
        .map(|(index, value)| (index, value.to_string()));

    let Some((command_index, command)) = command else {
        if argv.len() == executable + 1 || contains_only_root_options(&argv[executable + 1..]) {
            argv.extend(["tui".into(), "--local".into()]);
        }
        return Ok(());
    };

    match command.as_str() {
        "agent" => insert_local_flag(argv, command_index),
        "tui" | "chat" | "terminal" => {
            if has_remote_tui_options(&argv[command_index + 1..]) {
                return Err(CliError::Launch(
                    "OpenClaw remote TUI options (--url, --token, or --password) bypass the \
                     process carrying Relay's temporary provider overlay; use local mode or run \
                     the target OpenClaw gateway in the foreground under Relay"
                        .into(),
                ));
            }
            insert_local_flag(argv, command_index);
        }
        "gateway" => {
            let action = argv[command_index + 1..]
                .iter()
                .find(|value| !value.starts_with('-'))
                .map(String::as_str);
            if action != Some("run") {
                return Err(CliError::Launch(
                    "OpenClaw gateway interception requires the foreground `gateway run` \
                     command; detached, service-managed, or remote gateways cannot retain \
                     Relay's temporary configuration overlay"
                        .into(),
                ));
            }
        }
        other => {
            return Err(CliError::Launch(format!(
                "OpenClaw command `{other}` is not a supported Relay launch target; run configuration, plugin, model-management, and onboarding commands directly so their writes are not redirected into Relay's temporary overlay"
            )));
        }
    }
    Ok(())
}

fn contains_only_root_options(args: &[String]) -> bool {
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--dev" => index += 1,
            "--profile" | "--log-level" if args.get(index + 1).is_some() => index += 2,
            value if value.starts_with("--profile=") || value.starts_with("--log-level=") => {
                index += 1;
            }
            _ => return false,
        }
    }
    true
}

fn has_container_option(args: &[String]) -> bool {
    args.iter()
        .any(|value| value == "--container" || value.starts_with("--container="))
}

fn insert_local_flag(argv: &mut Vec<String>, command_index: usize) {
    if !argv[command_index + 1..]
        .iter()
        .any(|value| value == "--local")
    {
        argv.insert(command_index + 1, "--local".into());
    }
}

fn has_remote_tui_options(args: &[String]) -> bool {
    args.iter().any(|value| {
        matches!(value.as_str(), "--url" | "--token" | "--password")
            || value.starts_with("--url=")
            || value.starts_with("--token=")
            || value.starts_with("--password=")
    })
}

fn is_openclaw_executable(value: &str) -> bool {
    let name = Path::new(value)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or(value);
    matches!(name, "openclaw" | "openclaw.exe")
}

#[cfg(test)]
fn build_routing_plan(config: Option<&Value>, gateway_url: &str) -> RoutingPlan {
    build_routing_plan_with_auth(config, gateway_url, AuthProfileEvidenceSet::default())
}

#[cfg(test)]
fn build_routing_plan_with_auth(
    config: Option<&Value>,
    gateway_url: &str,
    auth_profiles: AuthProfileEvidenceSet,
) -> RoutingPlan {
    build_routing_plan_with_evidence(
        config,
        gateway_url,
        auth_profiles,
        TrustedApiKeySet {
            openai: ProviderFamily::OpenAi.has_api_key_environment()
                || config
                    .is_some_and(|config| config_env_has_api_key(config, ProviderFamily::OpenAi)),
            anthropic: ProviderFamily::Anthropic.has_api_key_environment()
                || config.is_some_and(|config| {
                    config_env_has_api_key(config, ProviderFamily::Anthropic)
                }),
        },
    )
}

fn build_routing_plan_with_evidence(
    config: Option<&Value>,
    gateway_url: &str,
    auth_profiles: AuthProfileEvidenceSet,
    trusted_api_keys: TrustedApiKeySet,
) -> RoutingPlan {
    let mut plan = RoutingPlan {
        provider_overrides: Map::new(),
        openai_upstream: None,
        anthropic_upstream: None,
        notes: Vec::new(),
    };
    if config.is_some_and(contains_provider_include) {
        plan.notes.push(
            "OpenClaw provider routing left unchanged because its provider configuration uses \
             `$include`; Relay will not guess at an unresolved upstream"
                .into(),
        );
        return plan;
    }

    let providers = config.and_then(provider_map);
    if let Some(providers) = providers {
        let custom = providers
            .keys()
            .filter(|id| id.as_str() != "openai" && id.as_str() != "anthropic")
            .cloned()
            .collect::<Vec<_>>();
        if !custom.is_empty() {
            plan.notes.push(format!(
                "custom OpenClaw providers left unchanged and not intercepted: {}",
                custom.join(", ")
            ));
        }
    }

    for family in [ProviderFamily::OpenAi, ProviderFamily::Anthropic] {
        if config.is_some_and(|config| has_non_http_agent_model_runtime(config, family)) {
            plan.notes.push(format!(
                "OpenClaw provider `{}` left unchanged and not intercepted: agent model policy selects a non-HTTP runtime",
                family.id()
            ));
            continue;
        }
        let provider = providers.and_then(|providers| providers.get(family.id()));
        if let Some(reason) = unsupported_provider_reason(family, provider) {
            plan.notes.push(format!(
                "OpenClaw provider `{}` left unchanged and not intercepted: {reason}",
                family.id()
            ));
            continue;
        }
        if let Err(reason) = provider_uses_api_key(
            provider,
            auth_profiles.get(family),
            trusted_api_keys.get(family),
        ) {
            plan.notes.push(format!(
                "OpenClaw provider `{}` left unchanged: {reason}",
                family.id(),
            ));
            continue;
        }

        let api = provider
            .and_then(Value::as_object)
            .and_then(|provider| provider.get("api"))
            .and_then(Value::as_str)
            .unwrap_or_else(|| family.default_api());
        let upstream = provider
            .and_then(Value::as_object)
            .and_then(|provider| provider.get("baseUrl"))
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| family.default_base_url())
            .to_string();
        let relay_base_url = match family {
            ProviderFamily::OpenAi => format!("{}/v1", gateway_url.trim_end_matches('/')),
            ProviderFamily::Anthropic => gateway_url.trim_end_matches('/').to_string(),
        };
        plan.provider_overrides.insert(
            family.id().into(),
            json!({
                "baseUrl": relay_base_url,
                "agentRuntime": {"id": "openclaw"}
            }),
        );
        match family {
            ProviderFamily::OpenAi => plan.openai_upstream = Some(upstream.clone()),
            ProviderFamily::Anthropic => plan.anthropic_upstream = Some(upstream.clone()),
        }
        plan.notes.push(format!(
            "OpenClaw provider `{}` routed through Relay using {api}; original upstream preserved",
            family.id()
        ));
    }
    plan
}

fn has_non_http_agent_model_runtime(config: &Value, family: ProviderFamily) -> bool {
    let provider_prefix = format!("{}/", family.id());
    let has_override = |models: &Map<String, Value>| {
        models.iter().any(|(model_ref, model)| {
            model_ref.starts_with(&provider_prefix)
                && model.get("agentRuntime").is_some()
                && model.pointer("/agentRuntime/id").and_then(Value::as_str) != Some("openclaw")
        })
    };

    config
        .pointer("/agents/defaults/models")
        .and_then(Value::as_object)
        .is_some_and(&has_override)
        || config
            .pointer("/agents/list")
            .and_then(Value::as_array)
            .is_some_and(|agents| {
                agents.iter().any(|agent| {
                    agent
                        .get("models")
                        .and_then(Value::as_object)
                        .is_some_and(&has_override)
                })
            })
}

fn unsupported_provider_reason(family: ProviderFamily, provider: Option<&Value>) -> Option<String> {
    let provider_value = provider?;
    let Some(provider) = provider_value.as_object() else {
        return Some("provider configuration is not an object".into());
    };
    if provider.contains_key("baseUrl")
        && provider
            .get("baseUrl")
            .and_then(Value::as_str)
            .is_none_or(|value| value.trim().is_empty())
    {
        return Some("baseUrl is not a non-empty string".into());
    }
    if provider
        .get("baseUrl")
        .and_then(Value::as_str)
        .is_some_and(|value| value.contains("${"))
    {
        return Some(
            "environment-substituted baseUrl cannot be preserved safely by the launcher".into(),
        );
    }
    if provider.contains_key("agentRuntime")
        && provider
            .get("agentRuntime")
            .and_then(|runtime| runtime.get("id"))
            .and_then(Value::as_str)
            != Some("openclaw")
    {
        return Some("its explicit agent runtime is not the direct OpenClaw HTTP runtime".into());
    }
    if provider
        .get("models")
        .and_then(Value::as_array)
        .is_some_and(|models| {
            models.iter().any(|model| {
                model.as_object().is_some_and(|model| {
                    model.contains_key("api")
                        || model.contains_key("baseUrl")
                        || (model.contains_key("agentRuntime")
                            && model
                                .get("agentRuntime")
                                .and_then(|runtime| runtime.get("id"))
                                .and_then(Value::as_str)
                                != Some("openclaw"))
                })
            })
        })
    {
        return Some(
            "model-level API, base URL, or agent-runtime overrides cannot be routed safely".into(),
        );
    }
    let api = provider.get("api").and_then(Value::as_str)?;
    let supported = match family {
        ProviderFamily::OpenAi => matches!(api, "openai-completions" | "openai-responses"),
        ProviderFamily::Anthropic => api == "anthropic-messages",
    };
    (!supported).then(|| format!("unsupported API adapter `{api}`"))
}

fn provider_uses_api_key(
    provider: Option<&Value>,
    profiles: AuthProfileEvidence,
    trusted_api_key: bool,
) -> Result<(), &'static str> {
    let provider = provider.and_then(Value::as_object);
    let explicit_api_key = provider.is_some_and(|provider| {
        provider.get("auth").and_then(Value::as_str) == Some("api-key")
            || provider.get("apiKey").is_some_and(|value| match value {
                Value::String(value) => !value.trim().is_empty(),
                Value::Object(_) => true,
                _ => false,
            })
    });
    if provider
        .and_then(|provider| provider.get("auth"))
        .and_then(Value::as_str)
        .is_some_and(|auth| auth != "api-key")
    {
        return Err("its explicit authentication mode is not API-key based");
    }
    match profiles {
        AuthProfileEvidence::ApiKeyOnly => Ok(()),
        AuthProfileEvidence::ApiKeyOrNone if explicit_api_key || trusted_api_key => Ok(()),
        AuthProfileEvidence::ApiKeyOrNone => Err(
            "some OpenClaw agents have no saved API-key profile and no shared API key was detected",
        ),
        AuthProfileEvidence::NonApiKeyOrMixed => Err(
            "OpenClaw auth profiles include non-API-key or mixed credential types, so selection is ambiguous",
        ),
        AuthProfileEvidence::Unknown => {
            Err("OpenClaw auth-profile state could not be verified as API-key-only")
        }
        AuthProfileEvidence::None if explicit_api_key || trusted_api_key => Ok(()),
        AuthProfileEvidence::None => {
            Err("no API-key-backed configuration or auth profile was detected")
        }
    }
}

fn provider_map(config: &Value) -> Option<&Map<String, Value>> {
    config
        .get("models")
        .and_then(Value::as_object)
        .and_then(|models| models.get("providers"))
        .and_then(Value::as_object)
}

fn contains_provider_include(config: &Value) -> bool {
    let Some(root) = config.as_object() else {
        return false;
    };
    root.contains_key("$include")
        || ["models", "agents", "auth", "env"]
            .into_iter()
            .filter_map(|key| root.get(key))
            .any(contains_nested_include)
}

fn contains_nested_include(config: &Value) -> bool {
    match config {
        Value::Object(object) => {
            object.contains_key("$include") || object.values().any(contains_nested_include)
        }
        Value::Array(values) => values.iter().any(contains_nested_include),
        _ => false,
    }
}

fn trusted_api_key_sources(argv: &[String], config: Option<&Value>) -> TrustedApiKeySet {
    let state_dir = probe_state_dir(argv).ok();
    let has_key = |family: ProviderFamily| {
        family.has_api_key_environment()
            || config.is_some_and(|config| config_env_has_api_key(config, family))
            || state_dir
                .as_ref()
                .is_some_and(|state| dotenv_has_api_key(&state.join(".env"), family))
    };
    TrustedApiKeySet {
        openai: has_key(ProviderFamily::OpenAi),
        anthropic: has_key(ProviderFamily::Anthropic),
    }
}

fn config_env_has_api_key(config: &Value, family: ProviderFamily) -> bool {
    let Some(env) = config.get("env").and_then(Value::as_object) else {
        return false;
    };
    env.iter()
        .filter(|(name, _)| name.as_str() != "vars")
        .any(|(name, value)| family.is_api_key_name(name) && nonempty_string(value))
        || env
            .get("vars")
            .and_then(Value::as_object)
            .is_some_and(|vars| {
                vars.iter()
                    .any(|(name, value)| family.is_api_key_name(name) && nonempty_string(value))
            })
}

fn dotenv_has_api_key(path: &Path, family: ProviderFamily) -> bool {
    let Ok(contents) = std::fs::read_to_string(path) else {
        return false;
    };
    contents.lines().any(|line| {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            return false;
        }
        let line = line.strip_prefix("export ").unwrap_or(line).trim_start();
        let Some((name, value)) = line.split_once('=') else {
            return false;
        };
        family.is_api_key_name(name.trim()) && nonempty_dotenv_value(value)
    })
}

fn nonempty_string(value: &Value) -> bool {
    value.as_str().is_some_and(|value| !value.trim().is_empty())
}

fn nonempty_dotenv_value(value: &str) -> bool {
    let value = value.trim();
    if value.starts_with('#') {
        return false;
    }
    let value = if value.starts_with('\'') || value.starts_with('"') {
        value
    } else {
        value.split(" #").next().unwrap_or(value).trim_end()
    };
    !value.is_empty() && value != "\"\"" && value != "''"
}

fn overlay_document(
    source: Option<&Path>,
    source_config: Option<&Value>,
    providers: Map<String, Value>,
    bridge_dir: &Path,
    disable_embedded_relay: bool,
) -> Value {
    let mut overlay = Map::new();
    if let Some(source) = source {
        overlay.insert(
            "$include".into(),
            Value::String(source.display().to_string()),
        );
    }
    if !providers.is_empty() {
        overlay.insert("models".into(), json!({"providers": providers}));
    }
    let mut load_paths = source_config
        .and_then(|config| config.pointer("/plugins/load/paths"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let bridge_path = bridge_dir.display().to_string();
    if !load_paths
        .iter()
        .any(|path| path.as_str() == Some(bridge_path.as_str()))
    {
        load_paths.push(Value::String(bridge_path));
    }
    let mut entries = Map::from_iter([(
        RELAY_OPENCLAW_BRIDGE_ID.to_string(),
        json!({
            "enabled": true,
            "config": {},
            "hooks": {"allowConversationAccess": true}
        }),
    )]);
    if disable_embedded_relay {
        entries.insert(
            RELAY_OPENCLAW_PLUGIN_ID.to_string(),
            json!({"enabled": false}),
        );
    }
    overlay.insert(
        "plugins".into(),
        json!({
            "enabled": true,
            "load": {"paths": load_paths},
            "entries": entries
        }),
    );
    Value::Object(overlay)
}

fn probe_relay_plugin_installed(argv: &[String], source_path: &Path) -> bool {
    let args = ["plugins", "inspect", RELAY_OPENCLAW_PLUGIN_ID, "--json"].map(str::to_string);
    run_read_only_probe(argv, source_path, &args).is_some_and(|output| output.status.success())
}

fn write_bridge_plugin() -> Result<PathBuf, CliError> {
    let root = std::env::temp_dir().join(format!(
        "nemo-relay-openclaw-bridge-{}-{}",
        std::process::id(),
        unique_stamp()?
    ));
    let result = std::fs::create_dir_all(&root)
        .and_then(|()| std::fs::write(root.join("package.json"), BRIDGE_PACKAGE_JSON))
        .and_then(|()| std::fs::write(root.join("openclaw.plugin.json"), BRIDGE_MANIFEST_JSON))
        .and_then(|()| std::fs::write(root.join("index.js"), BRIDGE_INDEX_JS));
    if let Err(error) = result {
        let _ = std::fs::remove_dir_all(&root);
        return Err(CliError::Launch(format!(
            "could not create temporary OpenClaw CLI bridge: {error}"
        )));
    }
    Ok(root)
}

fn add_invocation_metadata(gateway: &mut GatewayConfig) {
    let mut metadata = match gateway.metadata.take() {
        Some(Value::Object(metadata)) => metadata,
        Some(metadata) => Map::from_iter([("nemo_relay_original_metadata".into(), metadata)]),
        None => Map::new(),
    };
    metadata.insert(
        "nemo_relay_invocation".into(),
        json!({"agent": "openclaw", "integration": "cli_launcher"}),
    );
    gateway.metadata = Some(Value::Object(metadata));
}

fn read_source_config(path: &Path) -> Result<(Option<Value>, bool), CliError> {
    let raw = match std::fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok((None, false)),
        Err(error) => {
            return Err(CliError::Launch(format!(
                "could not read OpenClaw config {}: {error}",
                path.display()
            )));
        }
    };
    let value: Value = json5::from_str(&raw).map_err(|error| {
        CliError::Launch(format!(
            "invalid OpenClaw JSON5 config {}: {error}",
            path.display()
        ))
    })?;
    if !value.is_object() {
        return Err(CliError::Launch(format!(
            "OpenClaw config {} must contain a JSON5 object",
            path.display()
        )));
    }
    Ok((Some(value), true))
}

fn resolve_config_path(argv: &[String]) -> Result<PathBuf, CliError> {
    if let Some(path) = nonempty_env_os(OPENCLAW_CONFIG_PATH) {
        return resolve_openclaw_path(path);
    }
    let home = effective_home()?;
    if let Some(state) = nonempty_env_os("OPENCLAW_STATE_DIR") {
        return Ok(resolve_openclaw_path(state)?.join("openclaw.json"));
    }
    if let Some(profile) = command_profile(argv)? {
        return Ok(home
            .join(format!(".openclaw-{profile}"))
            .join("openclaw.json"));
    }
    let primary = home.join(".openclaw").join("openclaw.json");
    Ok(primary)
}

fn command_profile(argv: &[String]) -> Result<Option<String>, CliError> {
    let mut profile = None;
    let executable = openclaw_executable_index(argv).ok_or_else(|| {
        CliError::Launch("could not locate the OpenClaw executable in the launch prefix".into())
    })?;
    let command = first_openclaw_positional(argv, executable)
        .map(|(index, _)| index)
        .unwrap_or(argv.len());
    let mut index = executable + 1;
    while index < command {
        match argv[index].as_str() {
            "--dev" if profile.is_none() => profile = Some("dev".to_string()),
            "--profile" => {
                let value = argv.get(index + 1).ok_or_else(|| {
                    CliError::Launch("OpenClaw --profile requires a profile name".into())
                })?;
                profile = Some(value.clone());
                index += 1;
            }
            value if value.starts_with("--profile=") => {
                profile = Some(value.trim_start_matches("--profile=").to_string());
            }
            _ => {}
        }
        index += 1;
    }
    if let Some(value) = profile.as_deref()
        && (value.is_empty()
            || !value.chars().all(|character| {
                character.is_ascii_alphanumeric() || matches!(character, '-' | '_')
            }))
    {
        return Err(CliError::Launch(format!(
            "invalid OpenClaw profile name `{value}`"
        )));
    }
    Ok(profile)
}

fn openclaw_executable_index(argv: &[String]) -> Option<usize> {
    if argv
        .first()
        .is_some_and(|value| is_openclaw_executable(value))
    {
        return Some(0);
    }
    let launcher = argv
        .first()
        .and_then(|value| Path::new(value).file_name())
        .and_then(|value| value.to_str())?;
    let start = match launcher {
        "npx" | "npx.exe" | "bunx" | "bunx.exe" | "pnpx" | "pnpx.exe" => 1,
        "npm" | "npm.exe" if matches!(argv.get(1).map(String::as_str), Some("exec" | "x")) => 2,
        "pnpm" | "pnpm.exe" | "yarn" | "yarn.exe"
            if matches!(argv.get(1).map(String::as_str), Some("exec" | "dlx")) =>
        {
            2
        }
        _ => return None,
    };
    for (index, value) in argv.iter().enumerate().skip(start) {
        if value == "--" || matches!(value.as_str(), "--yes" | "-y" | "--quiet") {
            continue;
        }
        // The first package/command token in a supported wrapper must be OpenClaw; never scan
        // later user arguments for a matching basename.
        return is_openclaw_executable(value).then_some(index);
    }
    None
}

fn first_openclaw_positional(argv: &[String], executable: usize) -> Option<(usize, &str)> {
    let mut index = executable + 1;
    while index < argv.len() {
        match argv[index].as_str() {
            "--profile" | "--log-level" | "--container" if argv.get(index + 1).is_some() => {
                index += 2;
            }
            value
                if value.starts_with("--profile=")
                    || value.starts_with("--log-level=")
                    || value.starts_with("--container=") =>
            {
                index += 1;
            }
            value if value.starts_with('-') => index += 1,
            value => return Some((index, value)),
        }
    }
    None
}

fn effective_home() -> Result<PathBuf, CliError> {
    let value = nonempty_env_os("OPENCLAW_HOME")
        .or_else(|| nonempty_env_os("HOME"))
        .or_else(|| nonempty_env_os("USERPROFILE"));
    match value {
        Some(value) => resolve_openclaw_path(value),
        None => std::env::current_dir().map_err(CliError::from),
    }
}

fn resolve_openclaw_path(value: OsString) -> Result<PathBuf, CliError> {
    let text = value.to_string_lossy();
    let path = if text == "~" || text.starts_with("~/") || text.starts_with("~\\") {
        let home = nonempty_env_os("HOME")
            .or_else(|| nonempty_env_os("USERPROFILE"))
            .ok_or_else(|| {
                CliError::Launch("cannot expand OpenClaw path without a home directory".into())
            })?;
        PathBuf::from(home).join(text.trim_start_matches('~').trim_start_matches(['/', '\\']))
    } else {
        PathBuf::from(value)
    };
    if path.is_absolute() {
        Ok(path)
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

fn probe_auth_profiles(
    argv: &[String],
    source_path: &Path,
    config: Option<&Value>,
    family: ProviderFamily,
) -> AuthProfileEvidence {
    let mut aggregate = None;
    for agent in auth_probe_agents(argv, config) {
        let evidence = probe_one_auth_profile_set(argv, source_path, family, agent.as_deref());
        aggregate = Some(match aggregate {
            None => evidence,
            Some(current) => merge_auth_profile_evidence(current, evidence),
        });
    }
    aggregate.unwrap_or(AuthProfileEvidence::Unknown)
}

fn probe_one_auth_profile_set(
    argv: &[String],
    source_path: &Path,
    family: ProviderFamily,
    agent: Option<&str>,
) -> AuthProfileEvidence {
    let mut args = vec!["models".into(), "auth".into()];
    if let Some(agent) = agent {
        args.extend(["--agent".into(), agent.to_string()]);
    }
    args.push("list".into());
    args.extend(["--provider".into(), family.id().into(), "--json".into()]);
    let Some(output) = run_read_only_probe(argv, source_path, &args) else {
        return AuthProfileEvidence::Unknown;
    };
    if !output.status.success() {
        return AuthProfileEvidence::Unknown;
    }
    let Ok(value) = serde_json::from_slice::<Value>(&output.stdout) else {
        return AuthProfileEvidence::Unknown;
    };
    let Some(profiles) = value.get("profiles").and_then(Value::as_array) else {
        return AuthProfileEvidence::Unknown;
    };
    if profiles.is_empty() {
        return AuthProfileEvidence::None;
    }
    if profiles
        .iter()
        .all(|profile| profile.get("type").and_then(Value::as_str) == Some("api_key"))
    {
        AuthProfileEvidence::ApiKeyOnly
    } else {
        AuthProfileEvidence::NonApiKeyOrMixed
    }
}

fn auth_probe_agents(argv: &[String], config: Option<&Value>) -> Vec<Option<String>> {
    if let Some(agent) = selected_agent_id(argv) {
        return vec![Some(agent.to_string())];
    }
    // Session keys and workspace selection can choose a non-default agent without an explicit
    // `--agent`. In that ambiguous case, aggregate every configured auth store rather than
    // redirecting globally based only on the default agent.
    let mut agents = vec![None];
    if let Some(configured) = config
        .and_then(|config| config.pointer("/agents/list"))
        .and_then(Value::as_array)
    {
        for agent in configured {
            let Some(id) = agent.get("id").and_then(Value::as_str) else {
                continue;
            };
            if !id.trim().is_empty() && !agents.iter().flatten().any(|existing| existing == id) {
                agents.push(Some(id.to_string()));
            }
        }
    }
    agents
}

fn merge_auth_profile_evidence(
    left: AuthProfileEvidence,
    right: AuthProfileEvidence,
) -> AuthProfileEvidence {
    use AuthProfileEvidence::{ApiKeyOnly, ApiKeyOrNone, NonApiKeyOrMixed, None, Unknown};
    match (left, right) {
        (Unknown, _) | (_, Unknown) => Unknown,
        (NonApiKeyOrMixed, _) | (_, NonApiKeyOrMixed) => NonApiKeyOrMixed,
        (ApiKeyOnly, ApiKeyOnly) => ApiKeyOnly,
        (None, None) => None,
        (ApiKeyOrNone, ApiKeyOnly | None | ApiKeyOrNone)
        | (ApiKeyOnly | None, ApiKeyOrNone)
        | (ApiKeyOnly, None)
        | (None, ApiKeyOnly) => ApiKeyOrNone,
    }
}

fn selected_agent_id(argv: &[String]) -> Option<&str> {
    let executable = openclaw_executable_index(argv)?;
    let (command_index, command) = first_openclaw_positional(argv, executable)?;
    if command != "agent" {
        return None;
    }
    let mut index = command_index + 1;
    while index < argv.len() {
        match argv[index].as_str() {
            "--agent" => return argv.get(index + 1).map(String::as_str),
            value if value.starts_with("--agent=") => {
                return Some(value.trim_start_matches("--agent="));
            }
            _ => index += 1,
        }
    }
    None
}

fn run_read_only_probe(argv: &[String], source_path: &Path, args: &[String]) -> Option<Output> {
    let executable = openclaw_executable_index(argv)?;
    let mut command = Command::new(argv.first()?);
    if executable > 0 {
        command.args(&argv[1..=executable]);
    }
    command
        .args(args)
        .env(OPENCLAW_CONFIG_PATH, source_path)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if std::env::var_os("OPENCLAW_STATE_DIR").is_none()
        && let Ok(state_dir) = probe_state_dir(argv)
    {
        command.env("OPENCLAW_STATE_DIR", state_dir);
    }
    command.output().ok()
}

fn probe_state_dir(argv: &[String]) -> Result<PathBuf, CliError> {
    if let Some(path) = nonempty_env_os("OPENCLAW_STATE_DIR") {
        return resolve_openclaw_path(path);
    }
    let home = effective_home()?;
    if let Some(profile) = command_profile(argv)? {
        return Ok(home.join(format!(".openclaw-{profile}")));
    }
    let primary = home.join(".openclaw");
    Ok(primary)
}

fn temporary_overlay_path(source_path: &Path) -> Result<PathBuf, CliError> {
    let stamp = unique_stamp()?;
    let parent = source_path.parent().ok_or_else(|| {
        CliError::Launch(format!(
            "OpenClaw config path {} has no parent directory",
            source_path.display()
        ))
    })?;
    std::fs::create_dir_all(parent)?;
    Ok(parent.join(format!(
        ".openclaw.nemo-relay-{}-{stamp}.json5",
        std::process::id()
    )))
}

fn unique_stamp() -> Result<u128, CliError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| CliError::Launch(error.to_string()))
        .map(|duration| duration.as_nanos())
}

fn nonempty_env_os(name: &str) -> Option<OsString> {
    std::env::var_os(name).filter(|value| {
        let value = value.to_string_lossy();
        let value = value.trim();
        !value.is_empty() && value != "undefined" && value != "null"
    })
}

#[cfg(test)]
#[path = "../tests/coverage/openclaw_tests.rs"]
mod tests;
