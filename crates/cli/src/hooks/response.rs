// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Gateway response handling for forwarded lifecycle hooks.

use futures_util::StreamExt;
use serde_json::Value;

use crate::error::CliError;

pub(super) const MAX_HOOK_RESPONSE_BYTES: usize = 1024 * 1024;

pub(super) async fn handle_hook_forward_response(
    response: Result<reqwest::Response, reqwest::Error>,
    fail_closed: bool,
) -> Result<(), CliError> {
    match response {
        Ok(response) => {
            let status = response.status();
            let body = match read_hook_response(response).await {
                Ok(body) => body,
                Err(error) => {
                    return handle_hook_failure(error, fail_closed, "response_read", None);
                }
            };
            handle_hook_forward_status(status, body, fail_closed)
        }
        Err(error) => {
            handle_hook_failure(CliError::Upstream(error), fail_closed, "transport", None)
        }
    }
}

pub(crate) fn handle_verified_hook_forward_response(
    response: Result<
        crate::gateway::client::VerifiedHttpResponse,
        crate::gateway::client::VerifiedHttpError,
    >,
    fail_closed: bool,
) -> Result<(), CliError> {
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
                    );
                }
            };
            handle_hook_forward_status(
                status,
                String::from_utf8_lossy(&response.body).into_owned(),
                fail_closed,
            )
        }
        Err(error) => handle_hook_failure(
            CliError::Install(format!("verified hook forward failed: {error}")),
            fail_closed,
            "verified_transport",
            None,
        ),
    }
}

pub(crate) fn handle_hook_forward_status(
    status: reqwest::StatusCode,
    body: String,
    fail_closed: bool,
) -> Result<(), CliError> {
    if !status.is_success() {
        if let Some(reason) = guardrail_rejection_reason(&body) {
            return Err(CliError::GuardrailRejected(reason));
        }
        return handle_hook_failure(
            CliError::Install(format!("hook forward failed with HTTP {status}")),
            fail_closed,
            "http_status",
            Some(status.as_u16()),
        );
    }
    if !body.is_empty() {
        println!("{body}");
    }
    Ok(())
}

fn handle_hook_failure(
    error: CliError,
    fail_closed: bool,
    reason: &'static str,
    status_code: Option<u16>,
) -> Result<(), CliError> {
    let mode = if fail_closed {
        "fail_closed"
    } else {
        "fail_open"
    };
    if fail_closed {
        log::error!(
            target: "nemo_relay.hook",
            event = "hook_delivery_failed",
            mode,
            reason,
            status_code,
            error_kind = error.log_kind();
            "Hook delivery failed"
        );
        Err(CliError::Install(format!(
            "nemo-relay hook forward failed: {error}"
        )))
    } else {
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
        Ok(())
    }
}

pub(super) async fn read_hook_response(response: reqwest::Response) -> Result<String, CliError> {
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
