// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Gateway response handling for forwarded lifecycle hooks.

use futures_util::StreamExt;
use serde_json::Value;

use crate::error::CliError;
use crate::operational::{self, OperationalContext};

pub(super) const MAX_HOOK_RESPONSE_BYTES: usize = 1024 * 1024;

pub(super) async fn handle_hook_forward_response_with_context(
    response: Result<reqwest::Response, reqwest::Error>,
    fail_closed: bool,
    operational: &OperationalContext,
) -> Result<HookDeliveryOutcome, CliError> {
    match response {
        Ok(response) => {
            let status = response.status();
            let body = match read_hook_response(response, operational).await {
                Ok(body) => body,
                Err(error) => {
                    return handle_hook_failure(
                        error,
                        fail_closed,
                        "response_read",
                        None,
                        operational,
                    );
                }
            };
            handle_hook_forward_status_with_context(status, body, fail_closed, operational)
        }
        Err(error) => handle_hook_failure(
            CliError::Upstream(error),
            fail_closed,
            "transport",
            None,
            operational,
        ),
    }
}

#[cfg(test)]
pub(crate) fn handle_verified_hook_forward_response(
    response: Result<
        crate::gateway::client::VerifiedHttpResponse,
        crate::gateway::client::VerifiedHttpError,
    >,
    fail_closed: bool,
) -> Result<(), CliError> {
    let operational = OperationalContext::new();
    handle_verified_hook_forward_response_with_context(response, fail_closed, &operational)
        .map(|_| ())
}

pub(super) fn handle_verified_hook_forward_response_with_context(
    response: Result<
        crate::gateway::client::VerifiedHttpResponse,
        crate::gateway::client::VerifiedHttpError,
    >,
    fail_closed: bool,
    operational: &OperationalContext,
) -> Result<HookDeliveryOutcome, CliError> {
    match response {
        Ok(response) => {
            let status = match reqwest::StatusCode::from_u16(response.status) {
                Ok(status) => status,
                Err(error) => {
                    let message = format!("verified hook response had an invalid status: {error}");
                    return handle_hook_failure(
                        CliError::Install(message),
                        fail_closed,
                        "invalid_status",
                        None,
                        operational,
                    );
                }
            };
            handle_hook_forward_status_with_context(
                status,
                String::from_utf8_lossy(&response.body).into_owned(),
                fail_closed,
                operational,
            )
        }
        Err(error) => handle_hook_failure(
            CliError::Install(format!("verified hook forward failed: {error}")),
            fail_closed,
            "verified_transport",
            None,
            operational,
        ),
    }
}

#[cfg(test)]
pub(crate) fn handle_hook_forward_status(
    status: reqwest::StatusCode,
    body: String,
    fail_closed: bool,
) -> Result<(), CliError> {
    let operational = OperationalContext::new();
    handle_hook_forward_status_with_context(status, body, fail_closed, &operational).map(|_| ())
}

pub(super) fn handle_hook_forward_status_with_context(
    status: reqwest::StatusCode,
    body: String,
    fail_closed: bool,
    operational: &OperationalContext,
) -> Result<HookDeliveryOutcome, CliError> {
    if !status.is_success() {
        if let Some(reason) = guardrail_rejection_reason(&body) {
            return Err(CliError::GuardrailRejected(reason));
        }
        return handle_hook_failure(
            CliError::Install(format!("hook forward failed with HTTP {status}")),
            fail_closed,
            "http_status",
            Some(status.as_u16()),
            operational,
        );
    }
    if !body.is_empty() {
        println!("{body}");
    }
    Ok(HookDeliveryOutcome::Completed)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum HookDeliveryOutcome {
    Completed,
    FailedOpen,
}

fn handle_hook_failure(
    error: CliError,
    fail_closed: bool,
    reason: &'static str,
    status_code: Option<u16>,
    operational: &OperationalContext,
) -> Result<HookDeliveryOutcome, CliError> {
    let mode = if fail_closed {
        "fail_closed"
    } else {
        "fail_open"
    };
    if fail_closed {
        operational::hook_failed_with_status(
            operational,
            "hook_forward",
            reason,
            true,
            status_code,
        );
        log::error!(
            target: "nemo_relay.hook",
            event = "hook_delivery_failed",
            mode,
            reason,
            status_code,
            error_kind = error.log_kind();
            "Hook delivery failed"
        );
        Err(CliError::HookDelivery {
            source: Box::new(error),
        })
    } else {
        operational::hook_failed_with_status(
            operational,
            "hook_forward",
            reason,
            false,
            status_code,
        );
        log::warn!(
            target: "nemo_relay.hook",
            event = "hook_delivery_failed",
            mode,
            reason,
            status_code,
            error_kind = error.log_kind();
            "Hook delivery failed open"
        );
        eprintln!("nemo-relay hook forward failed: {error}");
        Ok(HookDeliveryOutcome::FailedOpen)
    }
}

pub(super) async fn read_hook_response(
    response: reqwest::Response,
    operational: &OperationalContext,
) -> Result<String, CliError> {
    let mut stream = response.bytes_stream();
    let mut body = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if body.len().saturating_add(chunk.len()) > MAX_HOOK_RESPONSE_BYTES {
            operational::limit_exceeded(
                operational,
                "hook_forward",
                "max_hook_response_bytes",
                MAX_HOOK_RESPONSE_BYTES,
            );
            return Err(CliError::Install(format!(
                "hook forward response exceeds the {MAX_HOOK_RESPONSE_BYTES}-byte limit"
            )));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(String::from_utf8_lossy(&body).into_owned())
}

pub(super) fn guardrail_rejection_reason(body: &str) -> Option<String> {
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
