// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io::Read;
use std::path::Path;
use std::time::{Duration, Instant};

use base64::Engine;
use futures_util::StreamExt;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde_json::{Value, json};

use crate::config::{
    CodingAgent, GATEWAY_URL_ENV, GatewayMode, HookForwardCommand, TRANSPARENT_RUN_ENV,
};
use crate::error::CliError;
use crate::install_generation::InstallGeneration;

const HOOK_FORWARD_TIMEOUT: Duration = Duration::from_secs(2);
const HOOK_GATEWAY_RETRY_TIMEOUT: Duration = Duration::from_secs(20);
const MAX_HOOK_RESPONSE_BYTES: usize = 1024 * 1024;

/// Forwards a hook payload from an installed shell command to a running gateway.
///
/// Empty stdin is normalized to `{}` so hooks that provide no payload still generate observable
/// marks. Delivery failures are fail-open by default to avoid blocking coding agents, but
/// `--fail-closed` converts missing URLs, HTTP failures, and upstream errors into process errors.
pub(crate) async fn hook_forward(command: HookForwardCommand) -> Result<(), CliError> {
    // A transparent wrapper can coexist with any installed Relay plugin. Its process marker makes
    // persistent plugin hooks inert, while only the wrapper-owned command carries
    // `--transparent-run` and forwards to the process-private gateway. This avoids rewriting host
    // plugin settings and works for both installer and source-marketplace plugin identities.
    if transparent_run_active() && !command.transparent_run {
        return Ok(());
    }
    validate_optional_json("session metadata", command.session_metadata.as_deref())?;
    let fail_closed =
        command.fail_closed || std::env::var("NEMO_RELAY_FAIL_CLOSED").ok().as_deref() == Some("1");
    let destination = hook_destination(&command);
    let persistent = match (destination.lifecycle != HookGatewayLifecycle::Transparent)
        .then(|| recovery_plan(&destination.gateway_url))
        .transpose()
    {
        Ok(persistent) => persistent,
        Err(error) => return handle_hook_error(error, fail_closed),
    };
    let transparent_gateway = match command
        .transparent_run
        .then(|| transparent_gateway_spec(&destination.gateway_url))
        .transpose()
    {
        Ok(gateway) => gateway,
        Err(error) => return handle_hook_error(error, fail_closed),
    };
    let _generation_guard = if destination.lifecycle == HookGatewayLifecycle::Existing
        && !command.forward_only
    {
        let install_host = command.agent.install_arg();
        let Some(generation_file) = command.generation_file.clone() else {
            return handle_hook_error(
                CliError::Launch(format!(
                    "persistent {} hook is missing its install-generation fence; run `nemo-relay install {install_host} --force`",
                    command.agent.label()
                )),
                fail_closed,
            );
        };
        let Some(generation_token) = command.generation_token.as_deref() else {
            return handle_hook_error(
                CliError::Launch(format!(
                    "persistent {} hook is missing its expected install-generation identity; run `nemo-relay install {install_host} --force`",
                    command.agent.label()
                )),
                fail_closed,
            );
        };
        match InstallGeneration::capture_guarded_expected(generation_file, generation_token) {
            Ok((_generation, guard)) => Some(guard),
            Err(error) => return handle_hook_error(CliError::Launch(error), fail_closed),
        }
    } else {
        None
    };
    let input = match read_hook_payload(
        persistent
            .as_ref()
            .map_or(crate::config::DEFAULT_MAX_HOOK_PAYLOAD_BYTES, |launch| {
                launch.max_hook_payload_bytes
            }),
    ) {
        Ok(input) => input,
        Err(error) => return handle_hook_error(error, fail_closed),
    };
    if destination.lifecycle == HookGatewayLifecycle::Existing {
        let gateway = persistent
            .as_ref()
            .expect("existing persistent destinations resolve a gateway")
            .gateway
            .clone();
        if let Err(error) =
            wait_for_existing_gateway(gateway, destination.gateway_url.clone()).await
        {
            return handle_hook_error(error, fail_closed);
        }
    }
    let verified_gateway = persistent
        .as_ref()
        .map(|launch| &launch.gateway)
        .or(transparent_gateway.as_ref());
    if let Some(gateway) = verified_gateway {
        let response =
            send_verified_hook_forward_request(&command, gateway, &destination.gateway_url, input)
                .await?;
        return handle_verified_hook_forward_response(response, fail_closed);
    }

    let url = format!(
        "{}{}",
        destination.gateway_url.trim_end_matches('/'),
        command.agent.hook_path()
    );
    let response = send_hook_forward_request(&command, &url, input).await?;
    handle_hook_forward_response(response, fail_closed).await
}

fn handle_hook_error(error: CliError, fail_closed: bool) -> Result<(), CliError> {
    eprintln!("nemo-relay hook forward failed: {error}");
    if fail_closed { Err(error) } else { Ok(()) }
}

// Reads the native hook payload from stdin and normalizes empty payloads to JSON object syntax.
// This keeps hook commands observable even for agents or events that invoke hooks without input.
fn read_hook_payload(limit: usize) -> Result<String, CliError> {
    read_hook_payload_from(std::io::stdin(), limit)
}

fn read_hook_payload_from(reader: impl Read, limit: usize) -> Result<String, CliError> {
    let mut bytes = Vec::new();
    reader
        .take(limit.saturating_add(1) as u64)
        .read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        return Err(CliError::Install(format!(
            "hook payload exceeds the {limit}-byte limit"
        )));
    }
    let input = String::from_utf8(bytes)
        .map_err(|error| CliError::Install(format!("hook payload is not valid UTF-8: {error}")))?;
    if input.trim().is_empty() {
        Ok("{}".to_string())
    } else {
        Ok(input)
    }
}

struct HookDestination {
    gateway_url: String,
    lifecycle: HookGatewayLifecycle,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HookGatewayLifecycle {
    /// A transparent run owns the dynamic gateway and passes its URL through the environment.
    Transparent,
    /// Persistent hooks use the authenticated gateway started and maintained by MCP.
    Existing,
}

// Installed hooks use the shared fixed gateway that MCP owns. Transparent runs set the dynamic
// environment URL and already own that gateway's lifecycle.
fn hook_destination(command: &HookForwardCommand) -> HookDestination {
    resolve_hook_destination(
        command.gateway_url.clone(),
        std::env::var(GATEWAY_URL_ENV).ok(),
        command.forward_only,
        command.transparent_run,
    )
}

fn transparent_run_active() -> bool {
    std::env::var(TRANSPARENT_RUN_ENV).ok().as_deref() == Some("1")
}

fn resolve_hook_destination(
    command_url: Option<String>,
    environment_url: Option<String>,
    forward_only: bool,
    transparent_run: bool,
) -> HookDestination {
    if transparent_run {
        return HookDestination {
            gateway_url: command_url
                .or(environment_url)
                .unwrap_or_else(|| crate::sidecar::DEFAULT_URL.into()),
            lifecycle: HookGatewayLifecycle::Transparent,
        };
    }
    if forward_only {
        return HookDestination {
            gateway_url: command_url.unwrap_or_else(|| crate::sidecar::DEFAULT_URL.into()),
            lifecycle: HookGatewayLifecycle::Existing,
        };
    }
    if let Some(gateway_url) = command_url {
        return HookDestination {
            gateway_url,
            lifecycle: HookGatewayLifecycle::Existing,
        };
    }
    if let Some(gateway_url) = environment_url {
        return HookDestination {
            gateway_url,
            lifecycle: HookGatewayLifecycle::Transparent,
        };
    }
    HookDestination {
        gateway_url: crate::sidecar::DEFAULT_URL.into(),
        lifecycle: HookGatewayLifecycle::Existing,
    }
}

async fn wait_for_existing_gateway(
    gateway: crate::sidecar::GatewaySpec,
    gateway_url: String,
) -> Result<(), CliError> {
    tokio::task::spawn_blocking(move || {
        let deadline = Instant::now() + HOOK_GATEWAY_RETRY_TIMEOUT;
        loop {
            match gateway.existing_healthy_instance(&gateway_url) {
                Ok(Some(_instance_id)) => return Ok(()),
                Ok(None) if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(50));
                }
                Ok(None) => {
                    return Err(format!(
                        "no compatible Relay gateway became ready at {gateway_url}; ensure the host started `nemo-relay mcp`"
                    ));
                }
                Err(error) => return Err(error),
            }
        }
    })
    .await
    .map_err(|error| CliError::Launch(format!("hook gateway verification task failed: {error}")))?
    .map_err(CliError::Launch)
}

fn recovery_plan(gateway_url: &str) -> Result<crate::sidecar::PluginGatewaySpec, CliError> {
    let bind = crate::sidecar::loopback_bind(gateway_url).map_err(CliError::Install)?;
    crate::sidecar::resolve_plugin_gateway(&Default::default(), bind)
}

fn transparent_gateway_spec(gateway_url: &str) -> Result<crate::sidecar::GatewaySpec, CliError> {
    let bind = crate::sidecar::loopback_bind(gateway_url).map_err(CliError::Install)?;
    Ok(crate::sidecar::GatewaySpec::new(bind)
        .with_fingerprint(crate::config::transparent_gateway_fingerprint(gateway_url)))
}

async fn send_verified_hook_forward_request(
    command: &HookForwardCommand,
    gateway: &crate::sidecar::GatewaySpec,
    gateway_url: &str,
    input: String,
) -> Result<Result<crate::sidecar::VerifiedHttpResponse, crate::sidecar::VerifiedHttpError>, CliError>
{
    let headers = gateway_headers(
        command.profile.as_deref(),
        command.session_metadata.as_deref(),
        command.gateway_mode,
    )?
    .iter()
    .map(|(name, value)| {
        value
            .to_str()
            .map(|value| (name.as_str().to_string(), value.to_string()))
            .map_err(|error| {
                CliError::Install(format!(
                    "hook header {name} is not valid HTTP text: {error}"
                ))
            })
    })
    .collect::<Result<Vec<_>, _>>()?;
    let gateway = gateway.clone();
    let gateway_url = gateway_url.to_string();
    let path = command.agent.hook_path().to_string();
    tokio::task::spawn_blocking(move || {
        gateway.post_verified(
            &gateway_url,
            &path,
            &headers,
            input.as_bytes(),
            HOOK_FORWARD_TIMEOUT,
            MAX_HOOK_RESPONSE_BYTES,
        )
    })
    .await
    .map_err(|error| CliError::Launch(format!("verified hook request task failed: {error}")))
}

// Sends the hook payload with gateway-specific headers translated from CLI flags. The reqwest
// transport result is returned separately so response handling can preserve fail-open semantics.
async fn send_hook_forward_request(
    command: &HookForwardCommand,
    url: &str,
    input: String,
) -> Result<Result<reqwest::Response, reqwest::Error>, CliError> {
    Ok(reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(HOOK_FORWARD_TIMEOUT)
        .build()?
        .post(url)
        .headers(gateway_headers(
            command.profile.as_deref(),
            command.session_metadata.as_deref(),
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
            let body = match read_hook_response(response).await {
                Ok(body) => body,
                Err(error) if fail_closed => return Err(error),
                Err(error) => {
                    eprintln!("nemo-relay hook forward failed: {error}");
                    return Ok(());
                }
            };
            handle_hook_forward_status(status, body, fail_closed)
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

fn handle_verified_hook_forward_response(
    response: Result<crate::sidecar::VerifiedHttpResponse, crate::sidecar::VerifiedHttpError>,
    fail_closed: bool,
) -> Result<(), CliError> {
    match response {
        Ok(response) => {
            let status = reqwest::StatusCode::from_u16(response.status).map_err(|error| {
                CliError::Install(format!(
                    "verified hook response had an invalid status: {error}"
                ))
            })?;
            handle_hook_forward_status(
                status,
                String::from_utf8_lossy(&response.body).into_owned(),
                fail_closed,
            )
        }
        Err(error) => {
            eprintln!("nemo-relay hook forward failed: {error}");
            if fail_closed {
                Err(CliError::Install(format!(
                    "verified hook forward failed: {error}"
                )))
            } else {
                Ok(())
            }
        }
    }
}

fn handle_hook_forward_status(
    status: reqwest::StatusCode,
    body: String,
    fail_closed: bool,
) -> Result<(), CliError> {
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

async fn read_hook_response(response: reqwest::Response) -> Result<String, CliError> {
    let mut stream = response.bytes_stream();
    let mut body = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if body.len().saturating_add(chunk.len()) > MAX_HOOK_RESPONSE_BYTES {
            return Err(CliError::Install(format!(
                "hook forward response exceeds the {MAX_HOOK_RESPONSE_BYTES}-byte limit"
            )));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(String::from_utf8_lossy(&body).into_owned())
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

/// Generates native hook configuration for the selected agent.
///
/// The returned value always has a top-level `hooks` object. Claude/Codex use command hook
/// groups with optional tool matchers, while Hermes uses direct command entries.
pub(crate) fn generated_hooks(agent: CodingAgent, command: &str) -> Value {
    if agent.uses_direct_hook_entries() {
        direct_hooks(agent.hook_events(), command)
    } else {
        grouped_hooks(agent.hook_events(), command)
    }
}

/// Canonical persistent hook command used by every supported host.
pub(crate) fn persistent_hook_forward_command(
    relay: &Path,
    agent: CodingAgent,
    generation_file: &Path,
    generation_token: &str,
) -> Result<String, String> {
    hook_command(
        relay,
        &persistent_hook_arguments(agent, generation_file, generation_token),
    )
}

/// Canonical transparent hook command. It embeds the process-private dynamic gateway so hook hosts
/// that filter inherited environment variables cannot redirect delivery to the fixed endpoint.
pub(crate) fn transparent_hook_forward_command(
    relay: &Path,
    agent: CodingAgent,
    gateway_url: &str,
) -> Result<String, String> {
    hook_command(relay, &transparent_hook_arguments(agent, gateway_url))
}

#[cfg(test)]
pub(crate) fn transparent_hook_forward_command_for_platform(
    relay: &Path,
    agent: CodingAgent,
    gateway_url: &str,
    windows: bool,
) -> String {
    hook_command_for_platform(
        relay,
        &transparent_hook_arguments(agent, gateway_url),
        windows,
    )
}

#[cfg(test)]
pub(crate) fn persistent_hook_forward_command_for_platform(
    relay: &Path,
    agent: CodingAgent,
    generation_file: &Path,
    generation_token: &str,
    windows: bool,
) -> String {
    hook_command_for_platform(
        relay,
        &persistent_hook_arguments(agent, generation_file, generation_token),
        windows,
    )
}

fn transparent_hook_arguments(agent: CodingAgent, gateway_url: &str) -> Vec<String> {
    vec![
        "hook-forward".into(),
        agent.as_arg().into(),
        "--gateway-url".into(),
        gateway_url.into(),
        "--transparent-run".into(),
    ]
}

fn persistent_hook_arguments(
    agent: CodingAgent,
    generation_file: &Path,
    generation_token: &str,
) -> Vec<String> {
    vec![
        "hook-forward".into(),
        agent.as_arg().into(),
        "--gateway-url".into(),
        crate::sidecar::DEFAULT_URL.into(),
        "--generation-file".into(),
        generation_file.display().to_string(),
        "--generation-token".into(),
        generation_token.into(),
    ]
}

fn hook_command(relay: &Path, arguments: &[String]) -> Result<String, String> {
    #[cfg(windows)]
    {
        return encoded_windows_hook_command(&windows_powershell_launcher()?, relay, arguments);
    }
    #[cfg(not(windows))]
    {
        Ok(posix_hook_command(relay, arguments))
    }
}

#[cfg(test)]
fn hook_command_for_platform(relay: &Path, arguments: &[String], windows: bool) -> String {
    if windows {
        return encoded_windows_hook_command(
            "C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
            relay,
            arguments,
        )
        .expect("test hook command must fit within the Windows command-line limit");
    }
    posix_hook_command(relay, arguments)
}

#[cfg(any(not(windows), test))]
fn posix_hook_command(relay: &Path, arguments: &[String]) -> String {
    std::iter::once(relay.display().to_string())
        .chain(arguments.iter().cloned())
        .map(|argument| crate::plugin_host::shell_quote_arg_for_platform(&argument, false))
        .collect::<Vec<_>>()
        .join(" ")
}

// `cmd.exe` accepts at most 8,191 characters. Leave room for `/C` and the executable path added
// by the hook host instead of generating a command that will be truncated at runtime.
const MAX_WINDOWS_HOOK_COMMAND_UTF16_UNITS: usize = 8_000;

/// Encode a native Relay invocation so Windows hook hosts can pass it through `cmd.exe /C` as one
/// argument without corrupting quotes in canonical paths. Windows PowerShell is part of the
/// supported Windows platform; it only launches the Rust binary and preserves its standard I/O.
#[cfg(any(windows, test))]
fn encoded_windows_hook_command(
    powershell: &str,
    relay: &Path,
    arguments: &[String],
) -> Result<String, String> {
    const PREFIX: &str = "$ErrorActionPreference='Stop'; & ";
    const SUFFIX: &str = "; if ($null -eq $LASTEXITCODE) { exit 1 }; exit $LASTEXITCODE";

    let invocation = std::iter::once(relay.display().to_string())
        .chain(arguments.iter().cloned())
        .map(|argument| format!("'{}'", argument.replace('\'', "''")))
        .collect::<Vec<_>>()
        .join(" ");
    let script = format!("{PREFIX}{invocation}{SUFFIX}");
    let bytes = script
        .encode_utf16()
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
    let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
    let command =
        format!("{powershell} -NoLogo -NoProfile -NonInteractive -EncodedCommand {encoded}");
    if command.encode_utf16().count() > MAX_WINDOWS_HOOK_COMMAND_UTF16_UNITS {
        return Err(format!(
            "generated Windows coding-agent hook command exceeds the {MAX_WINDOWS_HOOK_COMMAND_UTF16_UNITS}-character safety limit; shorten the Relay or plugin installation path"
        ));
    }
    Ok(command)
}

#[cfg(windows)]
fn windows_powershell_launcher() -> Result<String, String> {
    let powershell = windows_powershell_path()?;
    if !Path::new(&powershell).is_file() {
        return Err(format!(
            "trusted Windows PowerShell launcher is missing at {powershell}; install Windows PowerShell before configuring coding-agent hooks"
        ));
    }
    Ok(powershell)
}

#[cfg(windows)]
fn windows_powershell_path() -> Result<String, String> {
    use std::os::windows::ffi::OsStringExt;
    use windows_sys::Win32::System::SystemInformation::GetSystemDirectoryW;

    let mut buffer = vec![0_u16; 260];
    let length = loop {
        // SAFETY: `buffer` is writable for its declared length and remains live for the call.
        let length = unsafe { GetSystemDirectoryW(buffer.as_mut_ptr(), buffer.len() as u32) };
        if length == 0 {
            return Err(format!(
                "failed to resolve the trusted Windows system directory: {}",
                std::io::Error::last_os_error()
            ));
        }
        if (length as usize) < buffer.len() {
            break length as usize;
        }
        buffer.resize(length as usize + 1, 0);
    };
    let system = std::path::PathBuf::from(std::ffi::OsString::from_wide(&buffer[..length]));
    let powershell = system.join("WindowsPowerShell/v1.0/powershell.exe");
    let powershell = powershell
        .into_os_string()
        .into_string()
        .map_err(|_| "trusted Windows PowerShell path is not valid Unicode".to_string())?
        .replace('\\', "/");
    if !safe_windows_launcher_token(&powershell) {
        return Err(format!(
            "trusted Windows PowerShell path {powershell} contains characters that cannot be represented safely in coding-agent hook commands"
        ));
    }
    Ok(powershell)
}

fn safe_windows_launcher_token(launcher: &str) -> bool {
    !launcher.is_empty()
        && launcher.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '/' | ':' | '.' | '_' | '-')
        })
        && launcher
            .to_ascii_lowercase()
            .ends_with("/system32/windowspowershell/v1.0/powershell.exe")
}

/// Decode only the exact PowerShell envelope emitted by [`encoded_windows_hook_command`].
///
/// Hermes uses this to migrate and replace Relay-owned hooks whose generation arguments change.
pub(crate) fn decode_windows_hook_command(command: &str) -> Option<Vec<String>> {
    const COMMAND_SEPARATOR: &str = " -NoLogo -NoProfile -NonInteractive -EncodedCommand ";
    const SCRIPT_PREFIX: &str = "$ErrorActionPreference='Stop'; & ";
    const SCRIPT_SUFFIX: &str = "; if ($null -eq $LASTEXITCODE) { exit 1 }; exit $LASTEXITCODE";

    if command.encode_utf16().count() > MAX_WINDOWS_HOOK_COMMAND_UTF16_UNITS {
        return None;
    }
    let (launcher, encoded) = command.split_once(COMMAND_SEPARATOR)?;
    if !safe_windows_launcher_token(launcher) {
        return None;
    }
    #[cfg(windows)]
    if !launcher.eq_ignore_ascii_case(&windows_powershell_path().ok()?) {
        return None;
    }
    if encoded.is_empty() || encoded.chars().any(char::is_whitespace) {
        return None;
    }
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .ok()?;
    let pairs = bytes.chunks_exact(2);
    if !pairs.remainder().is_empty() {
        return None;
    }
    let script = String::from_utf16(
        &pairs
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect::<Vec<_>>(),
    )
    .ok()?;
    let invocation = script
        .strip_prefix(SCRIPT_PREFIX)?
        .strip_suffix(SCRIPT_SUFFIX)?;
    parse_powershell_single_quoted_arguments(invocation)
}

fn parse_powershell_single_quoted_arguments(mut raw: &str) -> Option<Vec<String>> {
    let mut arguments = Vec::new();
    while !raw.is_empty() {
        raw = raw.strip_prefix('\'')?;
        let mut argument = String::new();
        loop {
            let quote = raw.find('\'')?;
            argument.push_str(&raw[..quote]);
            raw = &raw[quote + 1..];
            if let Some(rest) = raw.strip_prefix('\'') {
                argument.push('\'');
                raw = rest;
            } else {
                break;
            }
        }
        arguments.push(argument);
        if raw.is_empty() {
            break;
        }
        raw = raw.strip_prefix(' ')?;
        if raw.is_empty() {
            return None;
        }
    }
    (!arguments.is_empty()).then_some(arguments)
}

fn direct_hooks(events: &[&str], command: &str) -> Value {
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
    json!({ "hooks": Value::Object(hooks) })
}

// Generates hook groups for Claude/Codex events and adds a wildcard matcher to tool events when
// the target agent requires matcher-scoped tool hooks. Non-tool events omit matchers so they fire
// for the full lifecycle.
fn grouped_hooks(events: &[&str], command: &str) -> Value {
    let hooks: serde_json::Map<String, Value> = events
        .iter()
        .map(|event| {
            let mut group = serde_json::Map::new();
            if event_matches_tools(event) {
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
// Claude/Codex spellings.
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
    gateway_mode: Option<GatewayMode>,
) -> Result<HeaderMap, CliError> {
    let mut headers = HeaderMap::new();
    insert_header(&mut headers, "x-nemo-relay-config-profile", profile)?;
    insert_header(
        &mut headers,
        "x-nemo-relay-session-metadata",
        session_metadata,
    )?;
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
