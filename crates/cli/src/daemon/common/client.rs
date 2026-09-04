// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared authenticated control-plane client used by MCP and worker processes.

use std::future::Future;
use std::time::Duration;

use bytes::{Bytes, BytesMut};
use futures_util::StreamExt;
use reqwest::{Client, Response, StatusCode};
use serde::Serialize;
use serde::de::DeserializeOwned;

use super::address::daemon_url;
use super::control::{
    CHALLENGE_PATH, CLIENT_TOKEN_HEADER, ChallengeRequest, ChallengeResponse, RegistrationProof,
    descriptor, fresh_nonce,
};
use super::identity::{MachineIdentity, TokenDigest};
use super::protocol::{ComponentRole, HandshakeTranscript};
use super::state::verify_or_store_daemon_pin;
use crate::error::CliError;

const CONTROL_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const CONTROL_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_CONTROL_RESPONSE_BYTES: usize = 256 * 1024;

/// Limits retries for one idempotent, session-authenticated control request.
///
/// The request is serialized once before the first attempt. Every retry therefore carries the
/// same session sequence, request ID, payload hash, and JSON bytes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ControlRetryPolicy {
    attempt_timeout: Duration,
    total_timeout: Duration,
    retry_delay: Duration,
}

impl ControlRetryPolicy {
    pub(crate) const fn new(
        attempt_timeout: Duration,
        total_timeout: Duration,
        retry_delay: Duration,
    ) -> Self {
        Self {
            attempt_timeout,
            total_timeout,
            retry_delay,
        }
    }
}

struct ControlAttemptError {
    error: CliError,
    transient: bool,
}

impl ControlAttemptError {
    fn permanent(error: CliError) -> Self {
        Self {
            error,
            transient: false,
        }
    }

    fn transient(error: CliError) -> Self {
        Self {
            error,
            transient: true,
        }
    }
}

pub(crate) struct ClientHandshake {
    pub(crate) proof: RegistrationProof,
    daemon_origin: String,
}

impl ClientHandshake {
    /// Verifies the daemon's signature before TOFU-pinning its public identity.
    pub(crate) fn authenticate_daemon(
        &self,
        proof: &super::protocol::HandshakeProof,
    ) -> Result<(), CliError> {
        if proof.signer != ComponentRole::Daemon {
            return Err(CliError::Unauthorized(
                "daemon registration proof used the wrong role".into(),
            ));
        }
        self.proof
            .transcript
            .verify(proof)
            .map_err(|error| CliError::Unauthorized(error.to_string()))?;
        verify_or_store_daemon_pin(
            &self.daemon_origin,
            self.proof.transcript.responder_public_identity,
        )
    }
}

pub(crate) fn control_client() -> Result<Client, CliError> {
    Client::builder()
        .connect_timeout(CONTROL_CONNECT_TIMEOUT)
        .timeout(CONTROL_REQUEST_TIMEOUT)
        .redirect(reqwest::redirect::Policy::none())
        .http2_keep_alive_interval(Duration::from_secs(15))
        .build()
        .map_err(CliError::Upstream)
}

pub(crate) async fn begin_handshake(
    client: &Client,
    daemon_address: &str,
    role: ComponentRole,
    identity: &MachineIdentity,
    instance_id: &str,
    route_token_digest: Option<TokenDigest>,
) -> Result<ClientHandshake, CliError> {
    if role == ComponentRole::Daemon {
        return Err(CliError::Config(
            "a daemon cannot initiate a daemon client handshake".into(),
        ));
    }
    let daemon = daemon_url(daemon_address)?;
    let daemon_origin = daemon.as_str().trim_end_matches('/').to_owned();
    let initiator = descriptor(role);
    let initiator_nonce = fresh_nonce()?;
    let request = ChallengeRequest {
        initiator: initiator.clone(),
        initiator_instance_id: instance_id.to_owned(),
        initiator_public_identity: identity.public_identity(),
        initiator_fingerprint: identity.fingerprint(),
        initiator_nonce,
    };
    let challenge: ChallengeResponse = post_json(
        client,
        &format!("{daemon_origin}{CHALLENGE_PATH}"),
        &request,
        None,
    )
    .await?;
    challenge
        .daemon
        .validate()
        .map_err(|error| CliError::Unauthorized(error.to_string()))?;
    if challenge.daemon.role != ComponentRole::Daemon
        || challenge.daemon_public_identity.fingerprint() != challenge.daemon_fingerprint
        || challenge.daemon_instance_id.is_empty()
    {
        return Err(CliError::Unauthorized(
            "daemon returned an invalid service identity".into(),
        ));
    }
    challenge.verify_attestation(&request)?;
    // Authenticate and TOFU-pin the daemon before a subsequent registration request can disclose
    // the reusable route credential. First contact retains the normal limitations of TOFU.
    verify_or_store_daemon_pin(&daemon_origin, challenge.daemon_public_identity)?;
    let selected_protocol = initiator
        .protocol
        .negotiate(challenge.daemon.protocol)
        .map_err(|error| CliError::Unauthorized(error.to_string()))?;
    let transcript = HandshakeTranscript {
        daemon_target: daemon_origin.clone(),
        initiator,
        responder: challenge.daemon,
        initiator_instance_id: instance_id.to_owned(),
        responder_instance_id: challenge.daemon_instance_id,
        selected_protocol,
        initiator_public_identity: identity.public_identity(),
        responder_public_identity: challenge.daemon_public_identity,
        initiator_fingerprint: identity.fingerprint(),
        responder_fingerprint: challenge.daemon_fingerprint,
        challenge_id: challenge.challenge.id,
        initiator_nonce,
        responder_nonce: challenge.challenge.nonce,
        route_token_digest,
    };
    let initiator_proof = transcript
        .sign(role, identity)
        .map_err(|error| CliError::Unauthorized(error.to_string()))?;
    Ok(ClientHandshake {
        proof: RegistrationProof {
            transcript,
            initiator_proof,
        },
        daemon_origin,
    })
}

pub(crate) async fn post_json<T, R>(
    client: &Client,
    url: &str,
    payload: &T,
    route_token: Option<&str>,
) -> Result<R, CliError>
where
    T: Serialize + ?Sized,
    R: DeserializeOwned,
{
    let body = encode_control_request(payload)?;
    post_json_encoded(client, url, body, route_token)
        .await
        .map_err(|failure| failure.error)
}

pub(crate) async fn post_json_idempotent<T, R>(
    client: &Client,
    url: &str,
    payload: &T,
    route_token: Option<&str>,
    policy: ControlRetryPolicy,
) -> Result<R, CliError>
where
    T: Serialize + ?Sized,
    R: DeserializeOwned,
{
    let body = encode_control_request(payload)?;
    retry_control(policy, || {
        post_json_encoded(client, url, body.clone(), route_token)
    })
    .await
}

pub(crate) async fn post_empty_idempotent<T: Serialize + ?Sized>(
    client: &Client,
    url: &str,
    payload: &T,
    policy: ControlRetryPolicy,
) -> Result<(), CliError> {
    let body = encode_control_request(payload)?;
    retry_control(policy, || post_empty_encoded(client, url, body.clone())).await
}

fn encode_control_request<T: Serialize + ?Sized>(payload: &T) -> Result<Bytes, CliError> {
    serde_json::to_vec(payload)
        .map(Bytes::from)
        .map_err(|error| {
            CliError::Launch(format!("failed to encode daemon control request: {error}"))
        })
}

async fn post_json_encoded<R: DeserializeOwned>(
    client: &Client,
    url: &str,
    body: Bytes,
    route_token: Option<&str>,
) -> Result<R, ControlAttemptError> {
    let response = send_control_request(client, url, body, route_token).await?;
    let status = response.status();
    let bytes = read_bounded_control_response(response).await?;
    if !status.is_success() {
        return Err(status_error(status, &bytes));
    }
    serde_json::from_slice(&bytes).map_err(|error| {
        ControlAttemptError::permanent(CliError::Launch(format!(
            "invalid daemon control response: {error}"
        )))
    })
}

async fn post_empty_encoded(
    client: &Client,
    url: &str,
    body: Bytes,
) -> Result<(), ControlAttemptError> {
    let response = send_control_request(client, url, body, None).await?;
    let status = response.status();
    if status.is_success() {
        return Ok(());
    }
    Err(if status == StatusCode::UNAUTHORIZED {
        ControlAttemptError::permanent(CliError::Unauthorized(
            "daemon rejected the control session credential".into(),
        ))
    } else {
        let error = CliError::Launch(format!("daemon control request failed with HTTP {status}"));
        if is_transient_status(status) {
            ControlAttemptError::transient(error)
        } else {
            ControlAttemptError::permanent(error)
        }
    })
}

async fn send_control_request(
    client: &Client,
    url: &str,
    body: Bytes,
    route_token: Option<&str>,
) -> Result<Response, ControlAttemptError> {
    let mut request = client
        .post(url)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(body);
    if let Some(token) = route_token {
        request = request.header(CLIENT_TOKEN_HEADER, token);
    }
    request
        .send()
        .await
        .map_err(|error| ControlAttemptError::transient(CliError::Upstream(error)))
}

async fn read_bounded_control_response(response: Response) -> Result<Bytes, ControlAttemptError> {
    if response
        .content_length()
        .is_some_and(|length| length > MAX_CONTROL_RESPONSE_BYTES as u64)
    {
        return Err(response_too_large());
    }
    let initial_capacity = response
        .content_length()
        .and_then(|length| usize::try_from(length).ok())
        .unwrap_or(0)
        .min(MAX_CONTROL_RESPONSE_BYTES);
    let mut bytes = BytesMut::with_capacity(initial_capacity);
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk =
            chunk.map_err(|error| ControlAttemptError::transient(CliError::Upstream(error)))?;
        if chunk.len() > MAX_CONTROL_RESPONSE_BYTES.saturating_sub(bytes.len()) {
            return Err(response_too_large());
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes.freeze())
}

fn response_too_large() -> ControlAttemptError {
    ControlAttemptError::permanent(CliError::Launch(format!(
        "daemon control response exceeded {MAX_CONTROL_RESPONSE_BYTES} bytes"
    )))
}

fn status_error(status: StatusCode, bytes: &[u8]) -> ControlAttemptError {
    let message = serde_json::from_slice::<serde_json::Value>(bytes)
        .ok()
        .and_then(|value| {
            value
                .pointer("/error/message")
                .and_then(serde_json::Value::as_str)
                .map(ToOwned::to_owned)
        })
        .unwrap_or_else(|| "daemon rejected the control request".into());
    if status == StatusCode::UNAUTHORIZED {
        return ControlAttemptError::permanent(CliError::Unauthorized(message));
    }
    let error = CliError::Launch(format!(
        "daemon control request failed with HTTP {status}: {message}"
    ));
    if is_transient_status(status) {
        ControlAttemptError::transient(error)
    } else {
        ControlAttemptError::permanent(error)
    }
}

fn is_transient_status(status: StatusCode) -> bool {
    matches!(
        status,
        StatusCode::REQUEST_TIMEOUT
            | StatusCode::TOO_MANY_REQUESTS
            | StatusCode::INTERNAL_SERVER_ERROR
            | StatusCode::BAD_GATEWAY
            | StatusCode::SERVICE_UNAVAILABLE
            | StatusCode::GATEWAY_TIMEOUT
    ) || status.as_u16() == 425
}

async fn retry_control<T, Operation, Attempt>(
    policy: ControlRetryPolicy,
    mut operation: Operation,
) -> Result<T, CliError>
where
    Operation: FnMut() -> Attempt,
    Attempt: Future<Output = Result<T, ControlAttemptError>>,
{
    let deadline = tokio::time::Instant::now() + policy.total_timeout;
    loop {
        let now = tokio::time::Instant::now();
        let attempt_deadline = deadline.min(now + policy.attempt_timeout);
        let result = tokio::time::timeout_at(attempt_deadline, operation()).await;
        let error = match result {
            Ok(Ok(value)) => return Ok(value),
            Ok(Err(failure)) if !failure.transient => return Err(failure.error),
            Ok(Err(failure)) => failure.error,
            Err(_) => CliError::Launch("daemon control request attempt timed out".into()),
        };
        let now = tokio::time::Instant::now();
        if now >= deadline {
            return Err(error);
        }
        tokio::time::sleep_until(deadline.min(now + policy.retry_delay)).await;
    }
}

#[cfg(test)]
#[path = "../../../tests/coverage/daemon/client_tests.rs"]
mod tests;
