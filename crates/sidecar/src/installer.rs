// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde_json::{Value, json};
use toml_edit::{DocumentMut, table, value};

use crate::config::{
    CodingAgent, GatewayMode, HookForwardCommand, InstallCommand, InstallScope, InstallTarget,
};
use crate::error::SidecarError;

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
    "pre_tool_call",
    "post_tool_call",
    "subagent_start",
    "subagent_stop",
];

#[derive(Debug, Clone)]
struct PlannedFile {
    path: PathBuf,
    contents: String,
}

/// Plans and optionally writes persistent hook configuration for the selected agent.
///
/// Structured JSON options are validated before any filesystem writes, `--print` shows the exact
/// planned contents, and `--dry-run` stops before mutation. Existing files are merged rather than
/// replaced, with per-file backups created by `write_planned_file`.
pub(crate) fn install(command: InstallCommand) -> Result<(), SidecarError> {
    validate_optional_json("session metadata", command.session_metadata.as_deref())?;
    validate_optional_json("plugin config", command.plugin_config.as_deref())?;
    let files = planned_files(&command)?;
    if command.print {
        print_planned_files(&files);
    }
    if command.dry_run {
        print_dry_run_summary(&command);
        return Ok(());
    }
    write_planned_files(&files)?;
    print_target_note(command.agent, command.target);
    Ok(())
}

// Prints planned file contents in the same format used by installer dry-run tests. The trailing
// newline fix keeps concatenated file previews readable even when serialized contents lack one.
fn print_planned_files(files: &[PlannedFile]) {
    for file in files {
        println!("--- {}", file.path.display());
        print!("{}", file.contents);
        if !file.contents.ends_with('\n') {
            println!();
        }
    }
}

// Prints the install summary without touching the filesystem. Keeping this separate from the write
// path makes the `install` control flow read as validate, plan, preview, then mutate-or-return.
fn print_dry_run_summary(command: &InstallCommand) {
    println!(
        "Dry run: would install {} integration for {:?} {:?}.",
        command.agent.as_arg(),
        command.scope,
        command.target
    );
}

// Writes every planned file with backup behavior handled by `write_planned_file`. This helper
// centralizes the success output so per-file write semantics stay consistent across agents.
fn write_planned_files(files: &[PlannedFile]) -> Result<(), SidecarError> {
    for file in files {
        write_planned_file(file)?;
        println!("Installed {}", file.path.display());
    }
    Ok(())
}

/// Forwards a hook payload from an installed shell command to a running sidecar.
///
/// Empty stdin is normalized to `{}` so hooks that provide no payload still generate observable
/// marks. Delivery failures are fail-open by default to avoid blocking coding agents, but
/// `--fail-closed` converts missing URLs, HTTP failures, and upstream errors into process errors.
pub(crate) async fn hook_forward(command: HookForwardCommand) -> Result<(), SidecarError> {
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
fn read_hook_payload() -> Result<String, SidecarError> {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;
    if input.trim().is_empty() {
        Ok("{}".to_string())
    } else {
        Ok(input)
    }
}

// Builds the target sidecar hook URL and applies fail-open/fail-closed behavior for missing
// sidecar discovery. Returning `Ok(None)` is the fail-open path used by default hook commands.
fn hook_forward_url(command: &HookForwardCommand) -> Result<Option<String>, SidecarError> {
    let Some(sidecar_url) = resolve_hook_sidecar_url(
        command.agent,
        command.sidecar_url.clone(),
        std::env::var("NEMO_FLOW_SIDECAR_URL").ok(),
    ) else {
        eprintln!(
            "nemo-flow-sidecar hook forward failed: missing sidecar URL; pass --sidecar-url or set NEMO_FLOW_SIDECAR_URL"
        );
        if command.fail_closed {
            return Err(SidecarError::Install(
                "missing sidecar URL; pass --sidecar-url or set NEMO_FLOW_SIDECAR_URL".into(),
            ));
        }
        return Ok(None);
    };
    Ok(Some(format!(
        "{}{}",
        sidecar_url.trim_end_matches('/'),
        command.agent.hook_path()
    )))
}

// Sends the hook payload with sidecar-specific headers translated from CLI flags. The reqwest
// transport result is returned separately so response handling can preserve fail-open semantics.
async fn send_hook_forward_request(
    command: &HookForwardCommand,
    url: String,
    input: String,
) -> Result<Result<reqwest::Response, reqwest::Error>, SidecarError> {
    Ok(reqwest::Client::builder()
        .timeout(HOOK_FORWARD_TIMEOUT)
        .build()?
        .post(url)
        .headers(sidecar_headers(
            command.atif_dir.as_deref(),
            command.openinference_endpoint.as_deref(),
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
) -> Result<(), SidecarError> {
    match response {
        Ok(response) => {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            if !status.is_success() {
                eprintln!("nemo-flow-sidecar hook forward failed with HTTP {status}");
                if fail_closed {
                    return Err(SidecarError::Install(format!(
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
            eprintln!("nemo-flow-sidecar hook forward failed: {error}");
            if fail_closed {
                Err(SidecarError::Upstream(error))
            } else {
                Ok(())
            }
        }
    }
}

// Chooses the sidecar URL for hook-forward. Hermes prefers the runtime environment URL because
// its hooks are commonly installed persistently but reused by `run --agent hermes` with an
// ephemeral sidecar; other agents prefer the installed command URL for stable configuration.
fn resolve_hook_sidecar_url(
    agent: CodingAgent,
    command_url: Option<String>,
    env_url: Option<String>,
) -> Option<String> {
    match agent {
        // Hermes shell hooks are installed persistently, but `run --agent hermes`
        // starts an ephemeral sidecar and passes the live URL through env.
        CodingAgent::Hermes => env_url.or(command_url),
        _ => command_url.or(env_url),
    }
}

// Builds the exact files that would be written for an install command. Each agent keeps its native
// config format: Claude/Cursor/Codex hook JSON, Codex feature TOML, and Hermes YAML translated
// through the shared JSON hook merge logic.
fn planned_files(command: &InstallCommand) -> Result<Vec<PlannedFile>, SidecarError> {
    let base = install_base(command)?;
    match command.agent {
        CodingAgent::ClaudeCode => planned_claude_file(command, &base),
        CodingAgent::Codex => planned_codex_files(command, &base),
        CodingAgent::Cursor => planned_cursor_file(command, &base),
        CodingAgent::Hermes => planned_hermes_file(command, &base),
    }
}

// Plans the Claude settings file by merging generated hook groups into existing JSON settings.
// Claude's plugin-dir transparent mode uses a separate temporary file path handled by launcher.
fn planned_claude_file(
    command: &InstallCommand,
    base: &Path,
) -> Result<Vec<PlannedFile>, SidecarError> {
    let path = base.join(".claude/settings.json");
    Ok(vec![planned_json_hooks_file(
        path,
        claude_hooks(&hook_command(command, CodingAgent::ClaudeCode)),
    )?])
}

// Plans both Codex files: feature enablement in TOML and generated hook groups in JSON. The TOML
// merge intentionally leaves unrelated provider configuration untouched.
fn planned_codex_files(
    command: &InstallCommand,
    base: &Path,
) -> Result<Vec<PlannedFile>, SidecarError> {
    let config_path = base.join(".codex/config.toml");
    let hooks_path = base.join(".codex/hooks.json");
    let existing_config = read_optional_text_file(&config_path)?;
    Ok(vec![
        PlannedFile {
            path: config_path.clone(),
            contents: merge_codex_config(&existing_config)?,
        },
        planned_json_hooks_file(
            hooks_path,
            codex_hooks(&hook_command(command, CodingAgent::Codex)),
        )?,
    ])
}

// Plans Cursor's project hook file using the shared JSON hook merge behavior. Cursor transparent
// runs patch and restore this same path dynamically instead of writing persistent config.
fn planned_cursor_file(
    command: &InstallCommand,
    base: &Path,
) -> Result<Vec<PlannedFile>, SidecarError> {
    let path = base.join(".cursor/hooks.json");
    Ok(vec![planned_json_hooks_file(
        path,
        cursor_hooks(&hook_command(command, CodingAgent::Cursor)),
    )?])
}

// Plans Hermes YAML config by translating through the shared hook map format. Missing files are
// treated as empty config, while unreadable files fail rather than overwriting user state.
fn planned_hermes_file(
    command: &InstallCommand,
    base: &Path,
) -> Result<Vec<PlannedFile>, SidecarError> {
    let path = base.join(".hermes/config.yaml");
    let existing = read_optional_text_file(&path)?;
    let contents = merge_hermes_config(
        &existing,
        hermes_hooks(&hook_command(command, CodingAgent::Hermes)),
    )?;
    Ok(vec![PlannedFile { path, contents }])
}

// Reads an optional text file for config formats where missing files are valid install targets.
// Non-not-found I/O errors still propagate to avoid losing existing user configuration.
fn read_optional_text_file(path: &Path) -> Result<String, SidecarError> {
    match std::fs::read_to_string(path) {
        Ok(raw) => Ok(raw),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(String::new()),
        Err(error) => Err(SidecarError::Io(error)),
    }
}

// Produces a planned JSON hook file by reading existing JSON, merging generated hooks, and
// formatting the result consistently with the package hook bundles.
fn planned_json_hooks_file(path: PathBuf, generated: Value) -> Result<PlannedFile, SidecarError> {
    let existing = read_json_file(&path)?;
    let contents = serde_json::to_string_pretty(&merge_hooks(existing, generated)?)
        .map_err(|error| SidecarError::Install(error.to_string()))?;
    Ok(PlannedFile { path, contents })
}

// Resolves the installation root according to user or project scope. Hidden test-only overrides
// take precedence so coverage can avoid touching real home/project directories.
fn install_base(command: &InstallCommand) -> Result<PathBuf, SidecarError> {
    match command.scope {
        InstallScope::User => command
            .home_dir
            .clone()
            .or_else(home_dir)
            .ok_or_else(|| SidecarError::Install("could not resolve home directory".into())),
        InstallScope::Project => command
            .project_dir
            .clone()
            .map(Ok)
            .unwrap_or_else(std::env::current_dir)
            .map_err(SidecarError::from),
    }
}

// Builds the shell command persisted into hook configuration. Optional sidecar settings are turned
// into hook-forward flags and every argument is shell-quoted because most target hook systems store
// the command as a single shell string.
fn hook_command(command: &InstallCommand, agent: CodingAgent) -> String {
    let mut args = vec![
        "nemo-flow-sidecar".to_string(),
        "hook-forward".to_string(),
        agent.as_arg().to_string(),
        "--sidecar-url".to_string(),
        command.sidecar_url.clone(),
    ];
    push_optional_path(&mut args, "--atif-dir", command.atif_dir.as_deref());
    push_optional(
        &mut args,
        "--openinference-endpoint",
        command.openinference_endpoint.as_deref(),
    );
    push_optional(&mut args, "--profile", command.profile.as_deref());
    push_optional(
        &mut args,
        "--session-metadata",
        command.session_metadata.as_deref(),
    );
    push_optional(
        &mut args,
        "--plugin-config",
        command.plugin_config.as_deref(),
    );
    push_optional_gateway_mode(&mut args, command.gateway_mode);
    args.into_iter()
        .map(|arg| shell_quote(&arg))
        .collect::<Vec<_>>()
        .join(" ")
}

// Appends a flag/value pair only when a string option is present, preserving omission semantics in
// generated hook commands instead of serializing empty values.
fn push_optional(args: &mut Vec<String>, flag: &str, value: Option<&str>) {
    if let Some(value) = value {
        args.push(flag.to_string());
        args.push(value.to_string());
    }
}

// Appends optional path flags using display formatting because installed commands are read by a
// shell, not by Rust path parsers.
fn push_optional_path(args: &mut Vec<String>, flag: &str, value: Option<&Path>) {
    if let Some(value) = value {
        args.push(flag.to_string());
        args.push(value.display().to_string());
    }
}

// Serializes the gateway-mode enum into the generated hook-forward command only when explicitly
// configured, leaving default runtime behavior under the sidecar's normal config resolution.
fn push_optional_gateway_mode(args: &mut Vec<String>, gateway_mode: Option<GatewayMode>) {
    if let Some(gateway_mode) = gateway_mode {
        args.push("--gateway-mode".to_string());
        args.push(gateway_mode.as_arg().to_string());
    }
}

// Quotes a shell argument only when necessary. The safe character set is intentionally small so
// paths and URLs remain readable while whitespace, quotes, and shell metacharacters are protected.
fn shell_quote(value: &str) -> String {
    if value
        .chars()
        .all(|character| character.is_ascii_alphanumeric() || "-_./:=,".contains(character))
    {
        value.to_string()
    } else {
        format!("'{}'", value.replace('\'', "'\\''"))
    }
}

/// Generates native hook configuration for the selected agent.
///
/// The returned value always has a top-level `hooks` object, but Hermes uses its simpler command
/// group shape while Claude/Codex/Cursor use command hook groups with optional tool matchers.
pub(crate) fn generated_hooks(agent: CodingAgent, command: &str) -> Value {
    match agent {
        CodingAgent::ClaudeCode => claude_hooks(command),
        CodingAgent::Codex => codex_hooks(command),
        CodingAgent::Cursor => cursor_hooks(command),
        CodingAgent::Hermes => hermes_hooks(command),
    }
}

// Returns the shell command a hook should run to forward an event to the sidecar. Callers must
// pass the executable they want hooks to invoke. Transparent-run callers should pass the absolute
// path of the currently running sidecar binary so spawned hook subprocesses do not depend on the
// user's `PATH` (which Codex/Claude/Cursor inherit but which typically does not include
// `target/debug` or other dev locations); persistent-install callers can pass the bare name
// `"nemo-flow-sidecar"` because the user is expected to have the binary on `PATH` after install.
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
    hooks_for_events(CURSOR_HOOK_EVENTS, command, true)
}

// Generates Hermes YAML-compatible hook groups. Hermes expects direct command entries rather than
// the nested `type = command` group format used by Claude, Codex, and Cursor.
fn hermes_hooks(command: &str) -> Value {
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

// Generates hook groups for all requested events and adds a wildcard matcher to tool events when
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

// Identifies hook events that should receive wildcard tool matchers. The list includes current
// Claude/Codex spellings plus Cursor shell/MCP names so generated config stays agent-compatible.
fn event_matches_tools(event: &str) -> bool {
    matches!(
        event,
        "PreToolUse"
            | "PostToolUse"
            | "PostToolUseFailure"
            | "PermissionRequest"
            | "preToolUse"
            | "postToolUse"
            | "beforeShellExecution"
            | "afterShellExecution"
            | "beforeMCPExecution"
            | "afterMCPExecution"
    )
}

/// Merges generated hook groups into an existing hook configuration without duplicating groups.
///
/// Missing files are represented by `Null` and become empty objects. Existing non-object roots,
/// non-object `hooks`, non-array event hooks, or malformed generated hooks fail closed because
/// writing through those shapes would corrupt user configuration.
pub(crate) fn merge_hooks(existing: Value, generated: Value) -> Result<Value, SidecarError> {
    let mut root = hook_config_root(existing)?;
    let hooks = hooks_object_mut(&mut root)?;
    let generated_hooks = generated_hooks_object(&generated)?;
    for (event, groups) in generated_hooks {
        merge_event_hook_groups(hooks, event, groups)?;
    }
    Ok(root)
}

// Normalizes an existing hook config root. Missing files arrive as `Null`, valid JSON/YAML config
// roots remain objects, and other shapes are rejected before any install write can occur.
fn hook_config_root(existing: Value) -> Result<Value, SidecarError> {
    match existing {
        Value::Null => Ok(json!({})),
        Value::Object(object) => Ok(Value::Object(object)),
        _ => Err(SidecarError::Install(
            "hook config must be a JSON object".into(),
        )),
    }
}

// Returns the mutable `hooks` object from a config root, creating it when absent. A non-object
// `hooks` field is considered user config corruption and is not overwritten.
fn hooks_object_mut(root: &mut Value) -> Result<&mut serde_json::Map<String, Value>, SidecarError> {
    root.as_object_mut()
        .expect("root checked as object")
        .entry("hooks")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .ok_or_else(|| SidecarError::Install("hooks must be a JSON object".into()))
}

// Validates generated hook shape before merging. Generated hooks are internal data, but checking
// here keeps test failures localized if an agent bundle generator regresses.
fn generated_hooks_object(
    generated: &Value,
) -> Result<&serde_json::Map<String, Value>, SidecarError> {
    generated
        .get("hooks")
        .and_then(Value::as_object)
        .ok_or_else(|| SidecarError::Install("generated hooks were malformed".into()))
}

// Appends missing generated groups for one hook event. Equality comparison is exact so repeated
// installs are idempotent without trying to interpret vendor-specific hook group schemas.
fn merge_event_hook_groups(
    hooks: &mut serde_json::Map<String, Value>,
    event: &str,
    groups: &Value,
) -> Result<(), SidecarError> {
    let groups = groups
        .as_array()
        .ok_or_else(|| SidecarError::Install("generated hook groups were malformed".into()))?;
    let event_groups = hooks.entry(event.to_string()).or_insert_with(|| json!([]));
    let event_groups = event_groups
        .as_array_mut()
        .ok_or_else(|| SidecarError::Install(format!("{event} hooks must be an array")))?;
    for group in groups {
        if !event_groups.iter().any(|existing| existing == group) {
            event_groups.push(group.clone());
        }
    }
    Ok(())
}

// Enables Codex hook support in TOML without rewriting unrelated config. Empty config creates a
// new document; malformed TOML fails before any install writes occur.
fn merge_codex_config(existing: &str) -> Result<String, SidecarError> {
    let mut document = if existing.trim().is_empty() {
        DocumentMut::new()
    } else {
        existing
            .parse::<DocumentMut>()
            .map_err(|error| SidecarError::Install(format!("invalid TOML: {error}")))?
    };
    if !document.as_table().contains_key("features") {
        document["features"] = table();
    }
    document["features"]["codex_hooks"] = value(true);
    Ok(document.to_string())
}

// Parses Hermes YAML, merges generated hooks through the shared JSON hook merger, and serializes
// back to YAML. Empty files are treated as no existing configuration.
fn merge_hermes_config(existing: &str, generated: Value) -> Result<String, SidecarError> {
    let existing = if existing.trim().is_empty() {
        Value::Null
    } else {
        serde_yaml::from_str(existing).map_err(|error| {
            SidecarError::Install(format!("invalid YAML in Hermes config: {error}"))
        })?
    };
    let merged = merge_hooks(existing, generated)?;
    serde_yaml::to_string(&merged).map_err(|error| SidecarError::Install(error.to_string()))
}

/// Reads a JSON config file, returning `Null` for missing files.
///
/// Missing hook files are normal during first install and are merged as empty configs; malformed
/// JSON fails closed with the path in the error so the installer does not overwrite bad input.
pub(crate) fn read_json_file(path: &Path) -> Result<Value, SidecarError> {
    match std::fs::read_to_string(path) {
        Ok(raw) => serde_json::from_str(&raw).map_err(|error| {
            SidecarError::Install(format!("invalid JSON in {}: {error}", path.display()))
        }),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Value::Null),
        Err(error) => Err(SidecarError::Io(error)),
    }
}

// Writes one planned file, creating parents and backing up any existing file first. Backup naming
// is delegated to `backup_path` so the original extension is preserved in the backup filename.
fn write_planned_file(file: &PlannedFile) -> Result<(), SidecarError> {
    if let Some(parent) = file.path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if file.path.exists() {
        std::fs::copy(&file.path, backup_path(&file.path)?)?;
    }
    std::fs::write(&file.path, &file.contents)?;
    Ok(())
}

// Builds a timestamped backup path beside the original file. If a file has no extension, `config`
// is used so backup names remain recognizable.
fn backup_path(path: &Path) -> Result<PathBuf, SidecarError> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| SidecarError::Install(error.to_string()))?
        .as_secs();
    Ok(path.with_extension(format!(
        "{}.bak.{timestamp}",
        path.extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or("config")
    )))
}

// Resolves a cross-platform home directory from environment variables only, matching config
// resolution and keeping installer tests isolated through env/test overrides.
fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

// Validates optional JSON strings before they are embedded into generated hook-forward commands or
// headers. This catches quoting/config mistakes during install rather than during a later hook run.
fn validate_optional_json(name: &str, value: Option<&str>) -> Result<(), SidecarError> {
    if let Some(value) = value {
        serde_json::from_str::<Value>(value)
            .map_err(|error| SidecarError::Install(format!("invalid {name}: {error}")))?;
    }
    Ok(())
}

// Converts optional session/export/gateway settings into sidecar headers for hook-forward. Each
// absent value is omitted so the server can fall back to file, environment, or default config.
fn sidecar_headers(
    atif_dir: Option<&Path>,
    openinference_endpoint: Option<&str>,
    profile: Option<&str>,
    session_metadata: Option<&str>,
    plugin_config: Option<&str>,
    gateway_mode: Option<GatewayMode>,
) -> Result<HeaderMap, SidecarError> {
    let mut headers = HeaderMap::new();
    insert_header_path(&mut headers, "x-nemo-flow-atif-dir", atif_dir)?;
    insert_header(
        &mut headers,
        "x-nemo-flow-openinference-endpoint",
        openinference_endpoint,
    )?;
    insert_header(&mut headers, "x-nemo-flow-config-profile", profile)?;
    insert_header(
        &mut headers,
        "x-nemo-flow-session-metadata",
        session_metadata,
    )?;
    insert_header(&mut headers, "x-nemo-flow-plugin-config", plugin_config)?;
    insert_header(
        &mut headers,
        "x-nemo-flow-gateway-mode",
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
) -> Result<(), SidecarError> {
    if let Some(value) = value {
        headers.insert(
            HeaderName::from_static(name),
            HeaderValue::from_str(value).map_err(|error| {
                SidecarError::Install(format!("invalid header {name}: {error}"))
            })?,
        );
    }
    Ok(())
}

// Converts an optional filesystem path to a header value using loss-tolerant display text. This
// mirrors installed shell command behavior, where paths are passed as strings.
fn insert_header_path(
    headers: &mut HeaderMap,
    name: &'static str,
    value: Option<&Path>,
) -> Result<(), SidecarError> {
    if let Some(value) = value {
        let value = value.to_string_lossy();
        insert_header(headers, name, Some(value.as_ref()))
    } else {
        Ok(())
    }
}

// Prints agent/target-specific follow-up notes for limitations that cannot be encoded directly in
// hook files, such as GUI/cloud behavior or Hermes consent requirements.
fn print_target_note(agent: CodingAgent, target: InstallTarget) {
    match (agent, target) {
        (CodingAgent::ClaudeCode, InstallTarget::Gui | InstallTarget::Both) => {
            println!(
                "Note: Claude application/web sessions are not configured by Claude Code hooks."
            );
        }
        (CodingAgent::Codex, InstallTarget::Gui | InstallTarget::Both) => {
            println!(
                "Note: Codex GUI local sessions can use local config; cloud tasks need separate gateway support."
            );
        }
        (CodingAgent::Cursor, InstallTarget::Cli | InstallTarget::Both) => {
            println!(
                "Note: run the Cursor CLI smoke test to confirm cursor-agent loads hooks in your version."
            );
        }
        (CodingAgent::Hermes, InstallTarget::Cli | InstallTarget::Both) => {
            println!(
                "Note: Hermes shell hooks prefer NEMO_FLOW_SIDECAR_URL at runtime when set; otherwise they use the installed sidecar URL. Hook consent is still required unless approved interactively or through Hermes configuration."
            );
        }
        _ => {}
    }
}

#[cfg(test)]
#[path = "../tests/coverage/installer_tests.rs"]
mod tests;
