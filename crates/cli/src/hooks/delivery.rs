// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io::Read;
use std::time::Duration;

use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde_json::Value;

use crate::error::CliError;
use crate::installation::generation::InstallGeneration;

use super::destination::{
    HookGatewayLifecycle, hook_destination, recovery_plan, transparent_gateway_spec,
    transparent_run_active, wait_for_existing_gateway,
};
use super::response::MAX_HOOK_RESPONSE_BYTES;
use super::response::{handle_hook_forward_response, handle_verified_hook_forward_response};
use super::{GatewayMode, HookForwardRequest};

const HOOK_FORWARD_TIMEOUT: Duration = Duration::from_secs(2);

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
        let response = match send_verified_hook_forward_request(
            &command,
            gateway,
            &destination.gateway_url,
            input,
        )
        .await
        {
            Ok(response) => response,
            Err(error) => return handle_hook_error(error, fail_closed),
        };
        return handle_verified_hook_forward_response(response, fail_closed);
    }

    let url = format!(
        "{}{}",
        destination.gateway_url.trim_end_matches('/'),
        command.agent.hook_path()
    );
    let response = match send_hook_forward_request(&command, &url, input).await {
        Ok(response) => response,
        Err(error) => return handle_hook_error(error, fail_closed),
    };
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

pub(crate) fn read_hook_payload_from(reader: impl Read, limit: usize) -> Result<String, CliError> {
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

pub(crate) async fn send_verified_hook_forward_request(
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
pub(crate) fn gateway_headers(
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
pub(crate) fn insert_header(
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
