// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-side authenticated daemon control session.

use std::time::Duration;

use reqwest::Client;

use super::super::common::client::{
    ControlRetryPolicy, begin_handshake, control_client, post_empty_idempotent, post_json,
};
use super::super::common::control::{
    SessionRequest, WORKER_HEARTBEAT_PATH, WORKER_READY_PATH, WORKER_RECOVER_PATH,
    WORKER_REGISTER_PATH, WorkerBootstrap, WorkerGenerationGrant, WorkerHeartbeatPayload,
    WorkerReadyPayload, WorkerRecoverRequest, WorkerRegisterRequest, WorkerRegisterResponse,
};
use super::super::common::identity::{MachineIdentity, TokenDigest};
use super::super::common::protocol::{ComponentRole, SensitiveString};
use crate::error::CliError;

const MIN_HEARTBEAT_INTERVAL: Duration = Duration::from_millis(100);
const MAX_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(20);
const CONTROL_RETRY_POLICY: ControlRetryPolicy = ControlRetryPolicy::new(
    Duration::from_secs(1),
    Duration::from_secs(14),
    Duration::from_millis(100),
);
const REGISTRATION_RETRY_MAX: Duration = Duration::from_secs(30);
const REGISTRATION_RETRY_DELAY: Duration = Duration::from_millis(100);

pub(super) struct Registration {
    client: Client,
    session_token: SensitiveString,
    data_token: SensitiveString,
    heartbeat_interval: Duration,
    next_sequence: u64,
    pending_ready: Option<SessionRequest<WorkerReadyPayload>>,
    pending_heartbeat: Option<SessionRequest<WorkerHeartbeatPayload>>,
    generation_grant: WorkerGenerationGrant,
}

impl Registration {
    pub(super) fn data_token_digest(&self) -> TokenDigest {
        TokenDigest::from_token(self.data_token.expose().as_bytes())
    }

    pub(super) fn session_token_digest(&self) -> TokenDigest {
        TokenDigest::from_token(self.session_token.expose().as_bytes())
    }

    pub(super) const fn heartbeat_interval(&self) -> Duration {
        self.heartbeat_interval
    }

    pub(super) const fn generation_grant(&self) -> &WorkerGenerationGrant {
        &self.generation_grant
    }

    pub(super) async fn ready(
        &mut self,
        daemon_origin: &str,
        worker_id: &str,
    ) -> Result<(), CliError> {
        if self.pending_ready.is_none() {
            self.pending_ready = Some(SessionRequest::new(
                worker_id.to_owned(),
                self.session_token.clone(),
                self.next_sequence,
                WorkerReadyPayload {
                    worker_id: worker_id.to_owned(),
                },
            )?);
        }
        let request = self
            .pending_ready
            .as_ref()
            .expect("pending readiness message was initialized");
        post_empty_idempotent(
            &self.client,
            &format!("{daemon_origin}{WORKER_READY_PATH}"),
            request,
            CONTROL_RETRY_POLICY,
        )
        .await?;
        self.pending_ready = None;
        self.advance_sequence()
    }

    pub(super) async fn heartbeat(
        &mut self,
        daemon_origin: &str,
        worker_id: &str,
    ) -> Result<(), CliError> {
        if self.pending_heartbeat.is_none() {
            self.pending_heartbeat = Some(SessionRequest::new(
                worker_id.to_owned(),
                self.session_token.clone(),
                self.next_sequence,
                WorkerHeartbeatPayload {
                    worker_id: worker_id.to_owned(),
                },
            )?);
        }
        let request = self
            .pending_heartbeat
            .as_ref()
            .expect("pending heartbeat was initialized");
        post_empty_idempotent(
            &self.client,
            &format!("{daemon_origin}{WORKER_HEARTBEAT_PATH}"),
            request,
            CONTROL_RETRY_POLICY,
        )
        .await?;
        self.pending_heartbeat = None;
        self.advance_sequence()
    }

    fn advance_sequence(&mut self) -> Result<(), CliError> {
        self.next_sequence = self.next_sequence.checked_add(1).ok_or_else(|| {
            CliError::Launch("daemon worker control sequence was exhausted".into())
        })?;
        Ok(())
    }
}

pub(super) async fn register(
    daemon_origin: &str,
    identity: &MachineIdentity,
    worker_id: &str,
    endpoint: &str,
    bootstrap: WorkerBootstrap,
    tls_root_certificate: Option<String>,
) -> Result<Registration, CliError> {
    let deadline = tokio::time::Instant::now() + REGISTRATION_RETRY_MAX;
    loop {
        match register_once(
            daemon_origin,
            identity,
            worker_id,
            endpoint,
            bootstrap.clone(),
            tls_root_certificate.clone(),
        )
        .await
        {
            Ok(registration) => return Ok(registration),
            Err(error @ CliError::Upstream(_)) => {
                let now = tokio::time::Instant::now();
                if now >= deadline {
                    return Err(error);
                }
                tokio::time::sleep_until(deadline.min(now + REGISTRATION_RETRY_DELAY)).await;
            }
            Err(error) => return Err(error),
        }
    }
}

async fn register_once(
    daemon_origin: &str,
    identity: &MachineIdentity,
    worker_id: &str,
    endpoint: &str,
    bootstrap: WorkerBootstrap,
    tls_root_certificate: Option<String>,
) -> Result<Registration, CliError> {
    let client = control_client()?;
    let handshake = begin_handshake(
        &client,
        daemon_origin,
        ComponentRole::Worker,
        identity,
        worker_id,
        None,
    )
    .await?;
    let request = WorkerRegisterRequest {
        proof: handshake.proof.clone(),
        worker_id: worker_id.to_owned(),
        endpoint: endpoint.to_owned(),
        activation_id: bootstrap.activation_id,
        activation_token: bootstrap.activation_token,
        tls_root_certificate,
    };
    let response: WorkerRegisterResponse = post_json(
        &client,
        &format!("{daemon_origin}{WORKER_REGISTER_PATH}"),
        &request,
        None,
    )
    .await?;
    handshake.authenticate_daemon(&response.daemon_proof)?;
    registration(client, response)
}

pub(super) async fn recover(
    daemon_origin: &str,
    identity: &MachineIdentity,
    worker_id: &str,
    endpoint: &str,
    tls_root_certificate: Option<&str>,
    generation_grant: WorkerGenerationGrant,
) -> Result<Registration, CliError> {
    let deadline = tokio::time::Instant::now() + REGISTRATION_RETRY_MAX;
    loop {
        match recover_once(
            daemon_origin,
            identity,
            worker_id,
            endpoint,
            tls_root_certificate,
            generation_grant.clone(),
        )
        .await
        {
            Ok(registration) => return Ok(registration),
            Err(error @ CliError::Upstream(_)) => {
                let now = tokio::time::Instant::now();
                if now >= deadline {
                    return Err(error);
                }
                tokio::time::sleep_until(deadline.min(now + REGISTRATION_RETRY_DELAY)).await;
            }
            Err(error) => return Err(error),
        }
    }
}

async fn recover_once(
    daemon_origin: &str,
    identity: &MachineIdentity,
    worker_id: &str,
    endpoint: &str,
    tls_root_certificate: Option<&str>,
    generation_grant: WorkerGenerationGrant,
) -> Result<Registration, CliError> {
    let client = control_client()?;
    let handshake = begin_handshake(
        &client,
        daemon_origin,
        ComponentRole::Worker,
        identity,
        worker_id,
        None,
    )
    .await?;
    let request = WorkerRecoverRequest {
        proof: handshake.proof.clone(),
        worker_id: worker_id.to_owned(),
        endpoint: endpoint.to_owned(),
        tls_root_certificate: tls_root_certificate.map(ToOwned::to_owned),
        generation_grant,
    };
    let response: WorkerRegisterResponse = post_json(
        &client,
        &format!("{daemon_origin}{WORKER_RECOVER_PATH}"),
        &request,
        None,
    )
    .await?;
    handshake.authenticate_daemon(&response.daemon_proof)?;
    registration(client, response)
}

fn registration(
    client: Client,
    response: WorkerRegisterResponse,
) -> Result<Registration, CliError> {
    let heartbeat_interval = validate_heartbeat_interval(response.heartbeat_interval_ms)?;
    Ok(Registration {
        client,
        session_token: response.session_token,
        data_token: response.data_token,
        heartbeat_interval,
        next_sequence: 1,
        pending_ready: None,
        pending_heartbeat: None,
        generation_grant: response.generation_grant,
    })
}

fn validate_heartbeat_interval(milliseconds: u64) -> Result<Duration, CliError> {
    let interval = Duration::from_millis(milliseconds);
    if !(MIN_HEARTBEAT_INTERVAL..=MAX_HEARTBEAT_INTERVAL).contains(&interval) {
        return Err(CliError::Unauthorized(
            "daemon returned an invalid worker heartbeat interval".into(),
        ));
    }
    Ok(interval)
}

#[cfg(test)]
pub(super) fn test_registration(data_token: &str, session_token: &str) -> Registration {
    let identity = MachineIdentity::generate().expect("test identity").identity;
    let generation_grant = WorkerGenerationGrant::issue(
        "worker-one",
        identity.fingerprint(),
        "http://127.0.0.1:1",
        None,
        &identity,
    )
    .expect("test generation grant");
    Registration {
        client: control_client().expect("test control client"),
        session_token: SensitiveString::new(session_token).expect("test session token"),
        data_token: SensitiveString::new(data_token).expect("test data token"),
        heartbeat_interval: Duration::from_secs(5),
        next_sequence: 1,
        pending_ready: None,
        pending_heartbeat: None,
        generation_grant,
    }
}

#[cfg(test)]
#[path = "../../../tests/coverage/daemon/worker_control_tests.rs"]
mod tests;
