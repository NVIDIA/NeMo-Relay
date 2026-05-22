// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io::Read;
use std::path::Path;
use std::time::Duration;

use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde_json::{Value, json};

use crate::config::{CodingAgent, GatewayMode, HookForwardCommand};
use crate::error::CliError;

// Claude Code's hook loader strictly whitelists event names — any unknown event causes the
// entire hooks file to be rejected (no hooks register). Only events present in Claude Code's
// whitelist as of 2.1.x belong here. Codex 0.129 has a smaller subset (SessionStart,
// UserPromptSubmit, PreToolUse, PostToolUse, Stop, PreCompact, PostCompact, PermissionRequest)
// and silently ignores events it doesn't recognize, so the union list is safe for both agents.
const HOOK_EVENTS: &[&str] = &[
    "SessionStart",
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "PostToolUseFailure",
    "PermissionRequest",
    "SubagentStart",
    "SubagentStop",
    "Notification",
    "Stop",
    "PreCompact",
    "PostCompact",
    "SessionEnd",
];

const CURSOR_HOOK_EVENTS: &[&str] = &[
    "sessionStart",
    "beforeSubmitPrompt",
    "preToolUse",
    "beforeShellExecution",
    "beforeMCPExecution",
    "postToolUse",
    "afterShellExecution",
    "afterMCPExecution",
    "subagentStart",
    "subagentStop",
    "afterAgentResponse",
    "afterAgentThought",
    "preCompact",
    "stop",
    "sessionEnd",
];
const HOOK_FORWARD_TIMEOUT: Duration = Duration::from_secs(2);

const HERMES_HOOK_EVENTS: &[&str] = &[
    "on_session_start",
    "on_session_end",
    "on_session_finalize",
    "on_session_reset",
    "pre_llm_call",
    "post_llm_call",
    "pre_api_request",
    "post_api_request",
    // Observer-only failure telemetry. Older Hermes versions ignore unknown hook names during
    // install, while newer versions use this to close failed provider attempts.
    "api_request_error",
    "pre_tool_call",
    "post_tool_call",
    "subagent_start",
    "subagent_stop",
];

/// Forwards a hook payload from an installed shell command to a running gateway.
///
/// Empty stdin is normalized to `{}` so hooks that provide no payload still generate observable
/// marks. Delivery failures are fail-open by default to avoid blocking coding agents, but
/// `--fail-closed` converts missing URLs, HTTP failures, and upstream errors into process errors.
pub(crate) async fn hook_forward(command: HookForwardCommand) -> Result<(), CliError> {
    validate_optional_json("session metadata", command.session_metadata.as_deref())?;
    validate_optional_json("plugin config", command.plugin_config.as_deref())?;

    let input = read_hook_payload()?;
    let Some(url) = hook_forward_url(&command)? else {
        return Ok(());
    };
    let response = send_hook_forward_request(&command, url, input).await?;
    handle_hook_forward_response(response, command.fail_closed).await
}

// Reads the native hook payload from stdin and normalizes empty payloads to JSON object syntax.
// This keeps hook commands observable even for agents or events that invoke hooks without input.
fn read_hook_payload() -> Result<String, CliError> {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;
    if input.trim().is_empty() {
        Ok("{}".to_string())
    } else {
        Ok(input)
    }
}

// Builds the target gateway hook URL and applies fail-open/fail-closed behavior for missing
// gateway discovery. Returning `Ok(None)` is the fail-open path used by default hook commands.
fn hook_forward_url(command: &HookForwardCommand) -> Result<Option<String>, CliError> {
    let Some(gateway_url) = resolve_hook_gateway_url(
        command.agent,
        command.gateway_url.clone(),
        std::env::var("NEMO_RELAY_GATEWAY_URL").ok(),
    ) else {
        eprintln!(
            "nemo-relay hook forward failed: missing gateway URL; pass --gateway-url or set NEMO_RELAY_GATEWAY_URL"
        );
        if command.fail_closed {
            return Err(CliError::Install(
                "missing gateway URL; pass --gateway-url or set NEMO_RELAY_GATEWAY_URL".into(),
            ));
        }
        return Ok(None);
    };
    Ok(Some(format!(
        "{}{}",
        gateway_url.trim_end_matches('/'),
        command.agent.hook_path()
    )))
}

// Sends the hook payload with gateway-specific headers translated from CLI flags. The reqwest
// transport result is returned separately so response handling can preserve fail-open semantics.
async fn send_hook_forward_request(
    command: &HookForwardCommand,
    url: String,
    input: String,
) -> Result<Result<reqwest::Response, reqwest::Error>, CliError> {
    crate::tls::install_rustls_crypto_provider();
    Ok(reqwest::Client::builder()
        .timeout(HOOK_FORWARD_TIMEOUT)
        .build()?
        .post(url)
        .headers(gateway_headers(
            command.profile.as_deref(),
            command.session_metadata.as_deref(),
            command.plugin_config.as_deref(),
            command.gateway_mode,
        )?)
        .header(CONTENT_TYPE, "application/json")
        .body(input)
        .send()
        .await)
}

// Handles hook delivery results without changing agent control flow unless `--fail-closed` was
// requested. Successful non-empty endpoint bodies are printed verbatim for the invoking hook API.
async fn handle_hook_forward_response(
    response: Result<reqwest::Response, reqwest::Error>,
    fail_closed: bool,
) -> Result<(), CliError> {
    match response {
        Ok(response) => {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            if !status.is_success() {
                if let Some(reason) = guardrail_rejection_reason(&body) {
                    return Err(CliError::GuardrailRejected(reason));
                }
                eprintln!("nemo-relay hook forward failed with HTTP {status}");
                if fail_closed {
                    return Err(CliError::Install(format!(
                        "hook forward failed with HTTP {status}"
                    )));
                }
                return Ok(());
            }
            if !body.is_empty() {
                println!("{body}");
            }
            Ok(())
        }
        Err(error) => {
            eprintln!("nemo-relay hook forward failed: {error}");
            if fail_closed {
                Err(CliError::Upstream(error))
            } else {
                Ok(())
            }
        }
    }
}

fn guardrail_rejection_reason(body: &str) -> Option<String> {
    let value: Value = serde_json::from_str(body).ok()?;
    let error = value.get("error")?;
    (error.get("type").and_then(Value::as_str) == Some("nemo_relay_guardrail_rejected"))
        .then(|| {
            error
                .get("reason")
                .and_then(Value::as_str)
                .or_else(|| error.get("message").and_then(Value::as_str))
                .map(ToOwned::to_owned)
        })
        .flatten()
}

// Chooses the gateway URL for hook-forward. Hermes prefers the runtime environment URL because
// its hooks are installed persistently by setup but reused under `nemo-relay hermes` with an
// ephemeral gateway; other agents prefer the installed command URL for stable configuration.
fn resolve_hook_gateway_url(
    agent: CodingAgent,
    command_url: Option<String>,
    env_url: Option<String>,
) -> Option<String> {
    match agent {
        CodingAgent::Hermes => env_url.or(command_url),
        _ => command_url.or(env_url),
    }
}

/// Generates native hook configuration for the selected agent.
///
/// The returned value always has a top-level `hooks` object. Claude/Codex use command hook
/// groups with optional tool matchers, while Cursor and Hermes use direct command entries.
pub(crate) fn generated_hooks(agent: CodingAgent, command: &str) -> Value {
    match agent {
        CodingAgent::ClaudeCode => claude_hooks(command),
        CodingAgent::Codex => codex_hooks(command),
        CodingAgent::Cursor => cursor_hooks(command),
        CodingAgent::Hermes => hermes_hooks(command),
    }
}

// Returns the shell command a hook should run to forward an event to the gateway. Callers must
// pass the executable they want hooks to invoke. Transparent-run callers should pass the absolute
// path of the currently running gateway binary so spawned hook subprocesses do not depend on the
// user's `PATH` (which Codex/Claude/Cursor inherit but which typically does not include
// `target/debug` or other dev locations); persistent-install callers can pass the bare name
// `"nemo-relay"` because the user is expected to have the binary on `PATH` after install.
pub(crate) fn hook_forward_command(executable: &str, agent: CodingAgent) -> String {
    format!("{executable} hook-forward {}", agent.as_arg())
}

fn claude_hooks(command: &str) -> Value {
    hooks_for_events(HOOK_EVENTS, command, true)
}

fn codex_hooks(command: &str) -> Value {
    hooks_for_events(HOOK_EVENTS, command, true)
}

fn cursor_hooks(command: &str) -> Value {
    direct_command_hooks_for_events(CURSOR_HOOK_EVENTS, command)
}

// Generates Hermes YAML-compatible hook groups. Hermes expects direct command entries rather than
// the nested `type = command` group format used by Claude and Codex.
pub(crate) fn hermes_hooks(command: &str) -> Value {
    let hooks: serde_json::Map<String, Value> = HERMES_HOOK_EVENTS
        .iter()
        .map(|event| {
            (
                (*event).to_string(),
                json!([{
                    "command": command,
                    "timeout": 30
                }]),
            )
        })
        .collect();
    json!({ "hooks": Value::Object(hooks) })
}

// Generates hook groups for Claude/Codex events and adds a wildcard matcher to tool events when
// the target agent requires matcher-scoped tool hooks. Non-tool events omit matchers so they fire
// for the full lifecycle.
fn hooks_for_events(events: &[&str], command: &str, matcher_for_tools: bool) -> Value {
    let hooks: serde_json::Map<String, Value> = events
        .iter()
        .map(|event| {
            let mut group = serde_json::Map::new();
            if matcher_for_tools && event_matches_tools(event) {
                group.insert("matcher".into(), json!("*"));
            }
            group.insert(
                "hooks".into(),
                json!([{
                    "type": "command",
                    "command": command,
                    "timeout": 30
                }]),
            );
            (
                (*event).to_string(),
                Value::Array(vec![Value::Object(group)]),
            )
        })
        .collect();
    json!({ "hooks": Value::Object(hooks) })
}

// Cursor CLI 2026.05 accepts direct command entries in `.cursor/hooks.json`; it does not execute
// the nested hook-group shape used by Claude Code and Codex.
fn direct_command_hooks_for_events(events: &[&str], command: &str) -> Value {
    let hooks: serde_json::Map<String, Value> = events
        .iter()
        .map(|event| {
            (
                (*event).to_string(),
                json!([{
                    "command": command,
                    "timeout": 30
                }]),
            )
        })
        .collect();
    json!({
        "version": 1,
        "hooks": Value::Object(hooks)
    })
}

// Identifies hook events that should receive wildcard tool matchers. The list includes current
// Claude/Codex spellings. Cursor uses direct command hooks and does not call this helper.
fn event_matches_tools(event: &str) -> bool {
    matches!(
        event,
        "PreToolUse" | "PostToolUse" | "PostToolUseFailure" | "PermissionRequest"
    )
}

/// Merges generated hook groups into an existing hook configuration without duplicating groups.
///
/// Missing files are represented by `Null` and become empty objects. Existing non-object roots,
/// non-object `hooks`, non-array event hooks, or malformed generated hooks fail closed because
/// writing through those shapes would corrupt user configuration.
pub(crate) fn merge_hooks(existing: Value, generated: Value) -> Result<Value, CliError> {
    let mut root = hook_config_root(existing)?;
    let hooks = hooks_object_mut(&mut root)?;
    let generated_hooks = generated_hooks_object(&generated)?;
    for (event, groups) in generated_hooks {
        merge_event_hook_groups(hooks, event, groups)?;
    }
    Ok(root)
}

// Normalizes an existing hook config root. Missing files arrive as `Null`, valid JSON/YAML config
// roots remain objects, and other shapes are rejected before any write can occur.
fn hook_config_root(existing: Value) -> Result<Value, CliError> {
    match existing {
        Value::Null => Ok(json!({})),
        Value::Object(object) => Ok(Value::Object(object)),
        _ => Err(CliError::Install(
            "hook config must be a JSON object".into(),
        )),
    }
}

// Returns the mutable `hooks` object from a config root, creating it when absent. A non-object
// `hooks` field is considered user config corruption and is not overwritten.
fn hooks_object_mut(root: &mut Value) -> Result<&mut serde_json::Map<String, Value>, CliError> {
    root.as_object_mut()
        .expect("root checked as object")
        .entry("hooks")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .ok_or_else(|| CliError::Install("hooks must be a JSON object".into()))
}

// Validates generated hook shape before merging. Generated hooks are internal data, but checking
// here keeps test failures localized if an agent bundle generator regresses.
fn generated_hooks_object(generated: &Value) -> Result<&serde_json::Map<String, Value>, CliError> {
    generated
        .get("hooks")
        .and_then(Value::as_object)
        .ok_or_else(|| CliError::Install("generated hooks were malformed".into()))
}

// Appends missing generated groups for one hook event. Equality comparison is exact so repeated
// writes are idempotent without trying to interpret vendor-specific hook group schemas.
fn merge_event_hook_groups(
    hooks: &mut serde_json::Map<String, Value>,
    event: &str,
    groups: &Value,
) -> Result<(), CliError> {
    let groups = groups
        .as_array()
        .ok_or_else(|| CliError::Install("generated hook groups were malformed".into()))?;
    let event_groups = hooks.entry(event.to_string()).or_insert_with(|| json!([]));
    let event_groups = event_groups
        .as_array_mut()
        .ok_or_else(|| CliError::Install(format!("{event} hooks must be an array")))?;
    for group in groups {
        if !event_groups.iter().any(|existing| existing == group) {
            event_groups.push(group.clone());
        }
    }
    Ok(())
}

/// Parses Hermes YAML, merges generated hooks through the shared JSON hook merger, and serializes
/// back to YAML. Empty input is treated as no existing configuration.
pub(crate) fn merge_hermes_config(existing: &str, generated: Value) -> Result<String, CliError> {
    let existing = if existing.trim().is_empty() {
        Value::Null
    } else {
        serde_yaml::from_str(existing)
            .map_err(|error| CliError::Install(format!("invalid YAML in Hermes config: {error}")))?
    };
    let merged = merge_hooks(existing, generated)?;
    serde_yaml::to_string(&merged).map_err(|error| CliError::Install(error.to_string()))
}

/// Reads a JSON config file, returning `Null` for missing files.
///
/// Missing hook files are normal during first install and are merged as empty configs; malformed
/// JSON fails closed with the path in the error so callers do not overwrite bad input.
pub(crate) fn read_json_file(path: &Path) -> Result<Value, CliError> {
    match std::fs::read_to_string(path) {
        Ok(raw) => serde_json::from_str(&raw).map_err(|error| {
            CliError::Install(format!("invalid JSON in {}: {error}", path.display()))
        }),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Value::Null),
        Err(error) => Err(CliError::Io(error)),
    }
}

// Validates optional JSON strings before they are embedded into hook-forward headers. Catches
// quoting/config mistakes at hook-fire time rather than after the request reaches the gateway.
fn validate_optional_json(name: &str, value: Option<&str>) -> Result<(), CliError> {
    if let Some(value) = value {
        serde_json::from_str::<Value>(value)
            .map_err(|error| CliError::Install(format!("invalid {name}: {error}")))?;
    }
    Ok(())
}

// Converts optional session/export/gateway settings into gateway headers for hook-forward. Each
// absent value is omitted so the server can fall back to file, environment, or default config.
fn gateway_headers(
    profile: Option<&str>,
    session_metadata: Option<&str>,
    plugin_config: Option<&str>,
    gateway_mode: Option<GatewayMode>,
) -> Result<HeaderMap, CliError> {
    let mut headers = HeaderMap::new();
    insert_header(&mut headers, "x-nemo-relay-config-profile", profile)?;
    insert_header(
        &mut headers,
        "x-nemo-relay-session-metadata",
        session_metadata,
    )?;
    insert_header(&mut headers, "x-nemo-relay-plugin-config", plugin_config)?;
    insert_header(
        &mut headers,
        "x-nemo-relay-gateway-mode",
        gateway_mode.map(GatewayMode::as_arg),
    )?;
    Ok(headers)
}

// Inserts one optional header after validating it is legal HTTP header text. Invalid values are
// reported as installer errors because they came from generated or user-provided hook options.
fn insert_header(
    headers: &mut HeaderMap,
    name: &'static str,
    value: Option<&str>,
) -> Result<(), CliError> {
    if let Some(value) = value {
        headers.insert(
            HeaderName::from_static(name),
            HeaderValue::from_str(value)
                .map_err(|error| CliError::Install(format!("invalid header {name}: {error}")))?,
        );
    }
    Ok(())
}

#[cfg(test)]
#[path = "../tests/coverage/installer_tests.rs"]
mod tests;
