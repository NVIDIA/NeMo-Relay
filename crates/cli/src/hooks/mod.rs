// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hook delivery, command encoding, generated definitions, and configuration merging.

mod encoding;
mod merging;
mod response;

#[cfg(test)]
pub(crate) use encoding::{
    decode_windows_hook_command, encoded_windows_hook_command, event_matches_tools,
    persistent_hook_forward_command_for_platform, transparent_hook_forward_command_for_platform,
};
pub(crate) use encoding::{
    generated_hooks, persistent_hook_forward_command, transparent_hook_forward_command,
};
pub(crate) use merging::merge_hooks;
use response::*;

use std::io::Read;
use std::path::Path;
use std::time::{Duration, Instant};

#[cfg(any(windows, test))]
use base64::Engine;
use futures_util::StreamExt;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde_json::{Value, json};

use crate::configuration::{
    CodingAgent, GATEWAY_URL_ENV, GatewayMode, HookForwardRequest, TRANSPARENT_RUN_ENV,
};
use crate::error::CliError;
use crate::installation::generation::InstallGeneration;

const HOOK_FORWARD_TIMEOUT: Duration = Duration::from_secs(2);
const HOOK_GATEWAY_RETRY_TIMEOUT: Duration = Duration::from_secs(20);
const MAX_HOOK_RESPONSE_BYTES: usize = 1024 * 1024;

/// Forwards a hook payload from an installed shell command to a running gateway.
///
/// Empty stdin is normalized to `{}` so hooks that provide no payload still generate observable
/// marks. Delivery failures are fail-open by default to avoid blocking coding agents, but
/// `--fail-closed` converts missing URLs, HTTP failures, and upstream errors into process errors.
pub(crate) async fn hook_forward(command: HookForwardRequest) -> Result<(), CliError> {
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
    let input = match read_hook_payload(persistent.as_ref().map_or(
        crate::configuration::DEFAULT_MAX_HOOK_PAYLOAD_BYTES,
        |launch| launch.max_hook_payload_bytes,
    )) {
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
fn hook_destination(command: &HookForwardRequest) -> HookDestination {
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
                .unwrap_or_else(|| crate::bootstrap::DEFAULT_URL.into()),
            lifecycle: HookGatewayLifecycle::Transparent,
        };
    }
    if forward_only {
        return HookDestination {
            gateway_url: command_url.unwrap_or_else(|| crate::bootstrap::DEFAULT_URL.into()),
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
        gateway_url: crate::bootstrap::DEFAULT_URL.into(),
        lifecycle: HookGatewayLifecycle::Existing,
    }
}

async fn wait_for_existing_gateway(
    gateway: crate::bootstrap::GatewaySpec,
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

fn recovery_plan(gateway_url: &str) -> Result<crate::bootstrap::PluginGatewaySpec, CliError> {
    let bind = crate::gateway::client::loopback_bind(gateway_url).map_err(CliError::Install)?;
    crate::bootstrap::resolve_plugin_gateway(&Default::default(), bind)
}

fn transparent_gateway_spec(gateway_url: &str) -> Result<crate::bootstrap::GatewaySpec, CliError> {
    let bind = crate::gateway::client::loopback_bind(gateway_url).map_err(CliError::Install)?;
    Ok(crate::bootstrap::GatewaySpec::new(bind).with_fingerprint(
        crate::configuration::transparent_gateway_fingerprint(gateway_url),
    ))
}

async fn send_verified_hook_forward_request(
    command: &HookForwardRequest,
    gateway: &crate::bootstrap::GatewaySpec,
    gateway_url: &str,
    input: String,
) -> Result<
    Result<crate::gateway::client::VerifiedHttpResponse, crate::gateway::client::VerifiedHttpError>,
    CliError,
> {
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
    command: &HookForwardRequest,
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
#[path = "../../tests/coverage/shared/installer_tests.rs"]
mod tests;
