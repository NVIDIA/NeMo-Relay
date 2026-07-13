// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Gateway response handling for forwarded lifecycle hooks.

use super::*;

pub(super) async fn handle_hook_forward_response(
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

pub(super) fn handle_verified_hook_forward_response(
    response: Result<
        crate::gateway::client::VerifiedHttpResponse,
        crate::gateway::client::VerifiedHttpError,
    >,
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

pub(super) fn handle_hook_forward_status(
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
