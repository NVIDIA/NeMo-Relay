// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Managed hook forwarding through an explicitly selected daemon.

use std::io::{Read, Write};
use std::time::Duration;

use futures_util::StreamExt;
use reqwest::header::{CONTENT_TYPE, HeaderValue};
use serde_json::Value;

use crate::agents::CodingAgent;
use crate::daemon::common::state::{ROUTE_TOKEN_ENV, RouteCredential};
use crate::error::CliError;
use crate::hooks::HookFailurePolicy;

pub(crate) const CLIENT_TOKEN_ENV: &str = ROUTE_TOKEN_ENV;
const CLIENT_TOKEN_HEADER: &str = crate::configuration::BOOTSTRAP_CLIENT_TOKEN_HEADER;
const HOOK_FORWARD_TIMEOUT: Duration = Duration::from_secs(2);
const MAX_HOOK_RESPONSE_BYTES: usize = 1024 * 1024;

#[derive(Debug, Clone)]
pub(crate) struct Options {
    pub(crate) agent: CodingAgent,
    pub(crate) daemon_address: String,
    pub(crate) failure_policy: HookFailurePolicy,
}

/// Reads one native hook payload, sends it to the daemon, and relays the response to stdout.
pub(crate) async fn run(options: Options) -> Result<(), CliError> {
    let payload = read_hook_payload(std::io::stdin());
    let fail_closed = effective_fail_closed(options.failure_policy, payload.as_deref().ok());
    let result: Result<(), CliError> = async {
        let token = route_token_from_environment()?;
        let payload = payload?;
        let body = forward(&options, payload, token).await?;
        if !body.is_empty() {
            std::io::stdout().write_all(&body)?;
        }
        Ok(())
    }
    .await;

    match result {
        Ok(()) => Ok(()),
        Err(error) if error.guardrail_rejection_reason().is_some() => Err(error),
        Err(error) => handle_delivery_failure(error, fail_closed),
    }
}

fn effective_fail_closed(policy: HookFailurePolicy, payload: Option<&[u8]>) -> bool {
    match policy {
        HookFailurePolicy::FailOpen => false,
        HookFailurePolicy::FailClosed => true,
        HookFailurePolicy::Default => {
            if policy.fail_closed() {
                return true;
            }
            payload
                .and_then(|payload| serde_json::from_slice::<Value>(payload).ok())
                .and_then(|payload| {
                    ["hook_event_name", "event_name", "event", "type"]
                        .into_iter()
                        .find_map(|name| payload.get(name).and_then(Value::as_str))
                        .map(crate::hooks::event_requires_fail_closed)
                })
                .unwrap_or(false)
        }
    }
}

fn read_hook_payload(mut reader: impl Read) -> Result<Vec<u8>, CliError> {
    let limit = crate::configuration::DEFAULT_MAX_HOOK_PAYLOAD_BYTES;
    let mut payload = Vec::new();
    reader
        .by_ref()
        .take(limit.saturating_add(1) as u64)
        .read_to_end(&mut payload)?;
    if payload.len() > limit {
        return Err(CliError::PayloadTooLarge(format!(
            "hook payload exceeds the {limit}-byte limit"
        )));
    }
    std::str::from_utf8(&payload)
        .map_err(|error| CliError::InvalidPayload(format!("hook payload is not UTF-8: {error}")))?;
    if payload.iter().all(u8::is_ascii_whitespace) {
        Ok(b"{}".to_vec())
    } else {
        Ok(payload)
    }
}

fn route_token_from_environment() -> Result<HeaderValue, CliError> {
    let credential = RouteCredential::from_environment()?;
    HeaderValue::from_str(credential.expose())
        .map_err(|_| CliError::Config(format!("{CLIENT_TOKEN_ENV} is not valid HTTP header text")))
}

#[cfg(test)]
fn route_token(value: &str) -> Result<HeaderValue, CliError> {
    let credential = RouteCredential::parse(value.to_owned())?;
    HeaderValue::from_str(credential.expose())
        .map_err(|_| CliError::Config(format!("{CLIENT_TOKEN_ENV} is not valid HTTP header text")))
}

async fn forward(
    options: &Options,
    payload: Vec<u8>,
    token: HeaderValue,
) -> Result<Vec<u8>, CliError> {
    let endpoint = hook_endpoint(&options.daemon_address, options.agent)?;
    let response = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(HOOK_FORWARD_TIMEOUT)
        .build()?
        .post(endpoint)
        .header(CONTENT_TYPE, "application/json")
        .header(CLIENT_TOKEN_HEADER, token)
        .body(payload)
        .send()
        .await?;
    let status = response.status();
    let body = read_response(response).await?;
    if status.is_success() {
        return Ok(body);
    }
    if let Some(reason) = guardrail_rejection_reason(&body) {
        return Err(CliError::GuardrailRejected(reason));
    }
    Err(CliError::Install(format!(
        "daemon hook forward failed with HTTP {status}"
    )))
}

fn hook_endpoint(daemon_address: &str, agent: CodingAgent) -> Result<reqwest::Url, CliError> {
    let mut url = reqwest::Url::parse(daemon_address)
        .map_err(|error| CliError::Config(format!("invalid daemon address: {error}")))?;
    if !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || !matches!(url.path(), "" | "/")
    {
        return Err(CliError::Config(
            "daemon address must be a root URL without credentials, query, or fragment".into(),
        ));
    }
    url.set_path(agent.hook_path());
    Ok(url)
}

async fn read_response(response: reqwest::Response) -> Result<Vec<u8>, CliError> {
    let mut stream = response.bytes_stream();
    let mut body = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if body.len().saturating_add(chunk.len()) > MAX_HOOK_RESPONSE_BYTES {
            return Err(CliError::PayloadTooLarge(format!(
                "daemon hook response exceeds the {MAX_HOOK_RESPONSE_BYTES}-byte limit"
            )));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

fn guardrail_rejection_reason(body: &[u8]) -> Option<String> {
    let value: Value = serde_json::from_slice(body).ok()?;
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

fn handle_delivery_failure(error: CliError, fail_closed: bool) -> Result<(), CliError> {
    let mode = if fail_closed {
        "fail_closed"
    } else {
        "fail_open"
    };
    if fail_closed {
        log::error!(
            target: "nemo_relay.hook",
            event = "daemon_hook_delivery_failed",
            mode,
            error_kind = error.log_kind();
            "Managed daemon hook delivery failed"
        );
        Err(CliError::HookDelivery {
            source: Box::new(error),
        })
    } else {
        log::warn!(
            target: "nemo_relay.hook",
            event = "daemon_hook_delivery_failed",
            mode,
            error_kind = error.log_kind();
            "Managed daemon hook delivery failed open"
        );
        eprintln!("nemo-relay daemon hook failed: {error}");
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../../tests/coverage/daemon/hook_tests.rs"]
mod tests;
