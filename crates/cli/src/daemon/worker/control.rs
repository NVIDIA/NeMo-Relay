// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-side authenticated daemon control session.

use super::super::common::socket::{Client, Command, Event};

use super::super::common::client::{begin_handshake, control_client};
use super::super::common::control::{
    SessionRequest, WorkerBootstrap, WorkerGenerationGrant, WorkerReadyPayload,
    WorkerRecoverRequest, WorkerRegisterRequest, WorkerRegisterResponse,
};
use super::super::common::identity::{MachineIdentity, TokenDigest};
use super::super::common::protocol::{ComponentRole, SensitiveString};
use crate::error::CliError;

pub(super) struct Registration {
    client: Client,
    session_token: SensitiveString,
    data_token: SensitiveString,
    next_sequence: u64,
    pending_ready: Option<SessionRequest<WorkerReadyPayload>>,
    generation_grant: WorkerGenerationGrant,
}

impl Registration {
    pub(super) fn data_token_digest(&self) -> TokenDigest {
        TokenDigest::from_token(self.data_token.expose().as_bytes())
    }

    pub(super) fn session_token_digest(&self) -> TokenDigest {
        TokenDigest::from_token(self.session_token.expose().as_bytes())
    }

    pub(super) const fn generation_grant(&self) -> &WorkerGenerationGrant {
        &self.generation_grant
    }

    pub(super) async fn ready(
        &mut self,
        _daemon_origin: &str,
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
        self.client
            .request::<()>(Command::Ready(request.clone()))
            .await?;
        self.pending_ready = None;
        self.advance_sequence()
    }

    #[cfg(test)]
    pub(super) async fn test_connect(&self, origin: &str) {
        self.client
            .connect(origin, ComponentRole::Worker)
            .await
            .unwrap();
    }
    pub(super) async fn recovery_deadline(&self) -> tokio::time::Instant {
        self.client.recovery_deadline().await
    }
    pub(super) async fn next(&self) -> Result<Event, CliError> {
        self.client.next().await
    }
    pub(super) async fn pending_event(&self) -> Option<Event> {
        self.client.pending_event().await
    }
    pub(super) async fn acknowledge(&self, id: String) -> Result<(), CliError> {
        self.client.acknowledge(id).await
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
    // Startup has no deadline. Loss of the broker still obeys the control recovery window,
    // including children whose protected bootstrap was transferred before their MCP exited.
    let recovery_deadline = tokio::time::Instant::now()
        + std::time::Duration::from_millis(crate::daemon::common::control::RECOVERY_LIFETIME_MS);
    let mut delay = std::time::Duration::from_millis(250);
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
            Err(error @ (CliError::Unauthorized(_) | CliError::Config(_))) => return Err(error),
            Err(error) => {
                if tokio::time::Instant::now() >= recovery_deadline {
                    return Err(error);
                }
                tokio::time::sleep(delay).await;
                delay = (delay * 2).min(std::time::Duration::from_secs(2));
            }
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
    let response: WorkerRegisterResponse = client.request(Command::RegisterWorker(request)).await?;
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
    recover_once(
        daemon_origin,
        identity,
        worker_id,
        endpoint,
        tls_root_certificate,
        generation_grant.clone(),
    )
    .await
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
    let response: WorkerRegisterResponse = client.request(Command::RecoverWorker(request)).await?;
    handshake.authenticate_daemon(&response.daemon_proof)?;
    registration(client, response)
}

fn registration(
    client: Client,
    response: WorkerRegisterResponse,
) -> Result<Registration, CliError> {
    Ok(Registration {
        client,
        session_token: response.session_token,
        data_token: response.data_token,
        next_sequence: 1,
        pending_ready: None,
        generation_grant: response.generation_grant,
    })
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
        next_sequence: 1,
        pending_ready: None,
        generation_grant,
    }
}

#[cfg(test)]
#[path = "../../../tests/coverage/daemon/worker_control_tests.rs"]
mod tests;
