// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io::Read;

#[cfg(test)]
use reqwest::header::CONTENT_TYPE;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde_json::Value;

use crate::error::CliError;
use crate::installation::generation::{ActiveGenerationGuard, InstallGeneration};
use crate::operational::{self, OperationalContext};

use super::destination::{
    HookGatewayLifecycle, hook_destination, recovery_plan, transparent_gateway_spec,
    transparent_run_active, wait_for_existing_gateway,
};
use super::response::MAX_HOOK_RESPONSE_BYTES;
use super::response::{HookDeliveryOutcome, handle_verified_hook_forward_response_with_context};
use super::{GatewayMode, HOOK_FORWARD_TIMEOUT, HookForwardRequest};

/// Pi invokes the same credentialed sender as native command hooks, but consumes the raw
/// HTTP verdict itself so transport faults retain the extension's existing failure policy.
pub(crate) async fn hook_forward_pi(command: HookForwardRequest) -> Result<(), CliError> {
    let operational = OperationalContext::new();
    let destination = hook_destination(&command);
    // The challenge accepts any caller fingerprint, including on a standalone gateway.
    // Reuse the pinned TLS sender so the hook payload reaches only an authenticated Relay.
    let gateway = transparent_gateway_spec(&destination.gateway_url)?;
    let input = read_hook_payload(
        crate::configuration::DEFAULT_MAX_HOOK_PAYLOAD_BYTES,
        &operational,
    )?;
    let response = send_verified_hook_forward_request_with_context(
        &command,
        &gateway,
        &destination.gateway_url,
        input,
        &operational,
        false,
    )
    .await?
    .map_err(|error| CliError::Launch(error.to_string()))?;
    let body = String::from_utf8(response.body)
        .map_err(|error| CliError::Install(format!("pi hook response is not UTF-8: {error}")))?;
    println!(
        "{}",
        serde_json::json!({"status": response.status, "body": body})
    );
    Ok(())
}

pub(crate) async fn hook_forward(mut command: HookForwardRequest) -> Result<(), CliError> {
    let operational = OperationalContext::new();
    operational::hook_started(&operational, "hook_forward");
    let fail_closed = super::effective_fail_closed(command.failure_policy, None);
    // Persistent hooks do not carry this marker, so they can become inert without reading a
    // stale configuration file while a process-private transparent gateway is active. Generated
    // transparent wrappers do carry it and still load their private configuration fail-closed.
    if transparent_hook_is_inert(&command) {
        operational::hook_completed(&operational, "hook_forward", "inert");
        return Ok(());
    }
    if let Some(path) = command.hook_config.clone()
        && let Err(error) =
            super::HookCommandConfig::load(&path).and_then(|config| config.apply(&mut command))
    {
        return handle_hook_error(CliError::Launch(error), fail_closed, &operational);
    }
    if let Err(error) =
        validate_optional_json("session metadata", command.session_metadata.as_deref())
    {
        return handle_hook_error(error, fail_closed, &operational);
    }
    let destination = hook_destination(&command);
    let persistent = match persistent_gateway(&destination) {
        Ok(persistent) => persistent,
        Err(error) => return handle_hook_error(error, fail_closed, &operational),
    };
    let transparent_gateway = match (destination.lifecycle == HookGatewayLifecycle::Transparent)
        .then(|| transparent_gateway_spec(&destination.gateway_url))
        .transpose()
    {
        Ok(gateway) => gateway,
        Err(error) => return handle_hook_error(error, fail_closed, &operational),
    };
    let _generation_guard = match capture_generation_guard(&command, destination.lifecycle) {
        Ok(guard) => guard,
        Err(error) => return handle_hook_error(error, fail_closed, &operational),
    };
    let payload_limit = persistent.as_ref().map_or(
        crate::configuration::DEFAULT_MAX_HOOK_PAYLOAD_BYTES,
        |launch| launch.max_hook_payload_bytes,
    );
    let input = match read_hook_payload(payload_limit, &operational) {
        Ok(input) => input,
        Err(error) => return handle_hook_error(error, fail_closed, &operational),
    };
    let fail_closed = super::effective_fail_closed(command.failure_policy, Some(input.as_bytes()));
    let verified_gateway = persistent
        .as_ref()
        .map(|launch| &launch.gateway)
        .or(transparent_gateway.as_ref());
    if let Some(gateway) = verified_gateway {
        let response = match send_verified_hook_forward_request_with_context(
            &command,
            gateway,
            &destination.gateway_url,
            input,
            &operational,
            destination.lifecycle == HookGatewayLifecycle::Existing,
        )
        .await
        {
            Ok(response) => response,
            Err(error) => return handle_hook_error(error, fail_closed, &operational),
        };
        return match handle_verified_hook_forward_response_with_context(
            response,
            fail_closed,
            &operational,
        ) {
            Ok(HookDeliveryOutcome::Completed) => {
                operational::hook_completed(&operational, "hook_forward", "completed");
                Ok(())
            }
            Ok(HookDeliveryOutcome::FailedOpen) => Ok(()),
            Err(error) if error.guardrail_rejection_reason().is_some() => {
                operational::hook_completed(&operational, "hook_forward", "rejected");
                Err(error)
            }
            Err(error) => Err(error),
        };
    }

    handle_hook_error(
        CliError::Launch("hook destination has no authenticated gateway".into()),
        fail_closed,
        &operational,
    )
}

pub(crate) fn transparent_hook_is_inert(command: &HookForwardRequest) -> bool {
    transparent_run_active() && !command.transparent_run
}

fn persistent_gateway(
    destination: &super::destination::HookDestination,
) -> Result<Option<crate::bootstrap::PluginGatewaySpec>, CliError> {
    (destination.lifecycle != HookGatewayLifecycle::Transparent)
        .then(|| recovery_plan(&destination.gateway_url))
        .transpose()
}

fn capture_generation_guard(
    command: &HookForwardRequest,
    lifecycle: HookGatewayLifecycle,
) -> Result<Option<ActiveGenerationGuard>, CliError> {
    if lifecycle != HookGatewayLifecycle::Existing || command.forward_only {
        return Ok(None);
    }
    let install_host = command.agent.install_arg();
    let generation_file = command.generation_file.clone().ok_or_else(|| {
        CliError::Launch(format!(
            "persistent {} hook is missing its install-generation fence; run `nemo-relay install {install_host} --force`",
            command.agent.label()
        ))
    })?;
    let generation_token = command.generation_token.as_deref().ok_or_else(|| {
        CliError::Launch(format!(
            "persistent {} hook is missing its expected install-generation identity; run `nemo-relay install {install_host} --force`",
            command.agent.label()
        ))
    })?;
    InstallGeneration::capture_guarded_expected(generation_file, generation_token)
        .map(|(_generation, guard)| Some(guard))
        .map_err(CliError::Launch)
}

fn handle_hook_error(
    error: CliError,
    fail_closed: bool,
    operational: &OperationalContext,
) -> Result<(), CliError> {
    operational::hook_failed(operational, "hook_forward", error.log_kind(), fail_closed);
    if fail_closed {
        log::error!(
            target: "nemo_relay.hook",
            event = "hook_delivery_failed",
            mode = "fail_closed",
            error_kind = error.log_kind();
            "Hook delivery failed"
        );
        Err(CliError::HookDelivery {
            source: Box::new(error),
        })
    } else {
        log::warn!(
            target: "nemo_relay.hook",
            event = "hook_delivery_failed",
            mode = "fail_open",
            error_kind = error.log_kind();
            "Hook delivery failed open"
        );
        eprintln!("nemo-relay hook forward failed: {error}");
        Ok(())
    }
}

// Reads the native hook payload from stdin and normalizes empty payloads to JSON object syntax.
// This keeps hook commands observable even for agents or events that invoke hooks without input.
fn read_hook_payload(limit: usize, operational: &OperationalContext) -> Result<String, CliError> {
    read_hook_payload_from_with_context(std::io::stdin(), limit, operational)
}

#[cfg(test)]
pub(crate) fn read_hook_payload_from(reader: impl Read, limit: usize) -> Result<String, CliError> {
    read_hook_payload_from_with_context(reader, limit, &OperationalContext::new())
}

fn read_hook_payload_from_with_context(
    reader: impl Read,
    limit: usize,
    operational: &OperationalContext,
) -> Result<String, CliError> {
    let mut bytes = Vec::new();
    reader
        .take(limit.saturating_add(1) as u64)
        .read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        operational::limit_exceeded(operational, "hook_forward", "max_hook_payload_bytes", limit);
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

#[cfg(test)]
pub(crate) async fn send_verified_hook_forward_request(
    command: &HookForwardRequest,
    gateway: &crate::bootstrap::GatewaySpec,
    gateway_url: &str,
    input: String,
) -> Result<
    Result<crate::gateway::client::VerifiedHttpResponse, crate::gateway::client::VerifiedHttpError>,
    CliError,
> {
    let operational = OperationalContext::new();
    send_verified_hook_forward_request_with_context(
        command,
        gateway,
        gateway_url,
        input,
        &operational,
        false,
    )
    .await
}

async fn send_verified_hook_forward_request_with_context(
    command: &HookForwardRequest,
    gateway: &crate::bootstrap::GatewaySpec,
    gateway_url: &str,
    input: String,
    operational: &OperationalContext,
    wait_for_ready: bool,
) -> Result<
    Result<crate::gateway::client::VerifiedHttpResponse, crate::gateway::client::VerifiedHttpError>,
    CliError,
> {
    let mut headers = gateway_headers(
        command.profile.as_deref(),
        command.session_metadata.as_deref(),
        command.gateway_mode,
    )?;
    attach_internal_hook_credentials(command, &mut headers)?;
    operational.attach_header(&mut headers);
    let headers = headers
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
        let deadline = std::time::Instant::now() + super::destination::HOOK_GATEWAY_RETRY_TIMEOUT;
        if wait_for_ready {
            match gateway.post_verified_preflight(
                &gateway_url,
                &path,
                &headers,
                input.as_bytes(),
                HOOK_FORWARD_TIMEOUT,
                MAX_HOOK_RESPONSE_BYTES,
            ) {
                Ok(response) => return Ok(Ok(response)),
                Err(error) if error.readiness_failed() => {
                    wait_for_existing_gateway(&gateway, &gateway_url, deadline)?;
                }
                Err(error) => return Ok(Err(error)),
            }
        }
        Ok(gateway.post_verified(
            &gateway_url,
            &path,
            &headers,
            input.as_bytes(),
            HOOK_FORWARD_TIMEOUT,
            MAX_HOOK_RESPONSE_BYTES,
        ))
    })
    .await
    .map_err(|error| CliError::Launch(format!("verified hook request task failed: {error}")))?
}

// Sends the hook payload with gateway-specific headers translated from CLI flags. The reqwest
// transport result is returned separately so response handling can preserve fail-open semantics.
#[cfg(test)]
pub(super) async fn send_hook_forward_request(
    command: &HookForwardRequest,
    url: &str,
    input: String,
    operational: &OperationalContext,
) -> Result<Result<reqwest::Response, reqwest::Error>, CliError> {
    let mut headers = gateway_headers(
        command.profile.as_deref(),
        command.session_metadata.as_deref(),
        command.gateway_mode,
    )?;
    attach_internal_hook_credentials(command, &mut headers)?;
    operational.attach_header(&mut headers);
    Ok(reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(HOOK_FORWARD_TIMEOUT)
        .build()?
        .post(url)
        .headers(headers)
        .header(CONTENT_TYPE, "application/json")
        .body(input)
        .send()
        .await)
}

fn attach_internal_hook_credentials(
    command: &HookForwardRequest,
    headers: &mut HeaderMap,
) -> Result<(), CliError> {
    let key = crate::configuration::BootstrapChallengeKey::load()?;
    let client_token = key.client_token();
    insert_header(
        headers,
        crate::configuration::BOOTSTRAP_CLIENT_TOKEN_HEADER,
        Some(&client_token),
    )?;
    if command.transparent_run {
        let credential = std::env::var(crate::provider_auth::TRANSPARENT_PROXY_CREDENTIAL_ENV)
            .map_err(|_| {
                CliError::Launch(
                    "transparent hook forwarding is missing its invocation credential".into(),
                )
            })?;
        insert_header(
            headers,
            crate::provider_auth::TRANSPARENT_PROXY_CREDENTIAL_HEADER,
            Some(&credential),
        )?;
    }
    if let Some(generation) = command.generation_token.as_deref() {
        let token = key.hook_client_token(generation);
        insert_header(
            headers,
            crate::configuration::HOOK_CLIENT_TOKEN_HEADER,
            Some(&token),
        )?;
    }
    Ok(())
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
