// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Versioned control-plane wire messages shared by daemon, MCP, and worker processes.

use std::net::Ipv4Addr;
use std::time::{SystemTime, UNIX_EPOCH};

use base64::Engine;
use ring::rand::{SecureRandom, SystemRandom};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::identity::{
    Challenge, ChallengeId, ChallengeNonce, Ed25519Signature, Fingerprint, MachineIdentity,
    PublicIdentity, encode_transcript,
};
use super::protocol::{
    BrokerDirective, ComponentDescriptor, ComponentRole, HandshakeProof, HandshakeTranscript,
    SensitiveString,
};
use crate::error::CliError;

pub(crate) const CHALLENGE_PATH: &str = "/_nemo-relay/control/v1/challenge";
pub(crate) const MCP_REGISTER_PATH: &str = "/_nemo-relay/control/v1/mcp/register";
pub(crate) const MCP_HEARTBEAT_PATH: &str = "/_nemo-relay/control/v1/mcp/heartbeat";
pub(crate) const MCP_RELEASE_PATH: &str = "/_nemo-relay/control/v1/mcp/release";
pub(crate) const MCP_ACTIVATION_FAILED_PATH: &str = "/_nemo-relay/control/v1/mcp/activation-failed";
pub(crate) const WORKER_REGISTER_PATH: &str = "/_nemo-relay/control/v1/worker/register";
pub(crate) const WORKER_RECOVER_PATH: &str = "/_nemo-relay/control/v1/worker/recover";
pub(crate) const WORKER_READY_PATH: &str = "/_nemo-relay/control/v1/worker/ready";
pub(crate) const WORKER_HEARTBEAT_PATH: &str = "/_nemo-relay/control/v1/worker/heartbeat";
pub(crate) const WORKER_DRAIN_PATH: &str = "/_nemo-relay/control/v1/worker/drain";
pub(crate) const WORKER_PROBE_PATH: &str = "/_nemo-relay/worker/v1/ready";

pub(crate) const CLIENT_TOKEN_HEADER: &str = "x-nemo-relay-client-token";
pub(crate) const WORKER_TOKEN_HEADER: &str = "x-nemo-relay-worker-token";
/// Private worker-to-daemon signal that a route-wide invariant failed after authentication.
/// The daemon consumes this field and never exposes it on the public response.
pub(crate) const WORKER_ROUTE_FAILURE_HEADER: &str = "x-nemo-relay-worker-route-failure";
pub(crate) const MAX_CONTROL_BODY_BYTES: usize = 256 * 1024;
pub(crate) const CHALLENGE_LIFETIME_MS: u64 = 15_000;
pub(crate) const MCP_HEARTBEAT_INTERVAL_MS: u64 = 10_000;
pub(crate) const MCP_LEASE_MS: u64 = 30_000;
pub(crate) const WORKER_HEARTBEAT_INTERVAL_MS: u64 = 5_000;
pub(crate) const WORKER_LEASE_MS: u64 = 20_000;
pub(crate) const ACTIVATION_LIFETIME_MS: u64 = 15_000;
pub(crate) const DRAIN_LIFETIME_MS: u64 = 120_000;
pub(crate) const RECOVERY_LIFETIME_MS: u64 = 120_000;
const WORKER_NETWORK_HINT_DOMAIN: &[u8] = b"nemo-relay/worker-network-hint/v1";
const WORKER_GENERATION_DOMAIN: &[u8] = b"nemo-relay/worker-generation/v1";
const DAEMON_CHALLENGE_DOMAIN: &[u8] = b"nemo-relay/daemon-challenge/v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ChallengeRequest {
    pub(crate) initiator: ComponentDescriptor,
    pub(crate) initiator_instance_id: String,
    pub(crate) initiator_public_identity: PublicIdentity,
    pub(crate) initiator_fingerprint: Fingerprint,
    pub(crate) initiator_nonce: ChallengeNonce,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ChallengeResponse {
    pub(crate) daemon: ComponentDescriptor,
    pub(crate) daemon_instance_id: String,
    pub(crate) daemon_public_identity: PublicIdentity,
    pub(crate) daemon_fingerprint: Fingerprint,
    pub(crate) challenge: Challenge,
    /// Daemon signature over this challenge and the complete initiating request.
    pub(crate) daemon_challenge_proof: Ed25519Signature,
}

impl ChallengeResponse {
    pub(crate) fn verify_attestation(&self, request: &ChallengeRequest) -> Result<(), CliError> {
        let canonical = daemon_challenge_bytes(request, self)?;
        self.daemon_public_identity
            .verify(&canonical, &self.daemon_challenge_proof)
            .map_err(|_| CliError::Unauthorized("daemon challenge signature did not verify".into()))
    }
}

pub(crate) fn daemon_challenge_bytes(
    request: &ChallengeRequest,
    response: &ChallengeResponse,
) -> Result<Vec<u8>, CliError> {
    let initiator = serde_json::to_vec(&request.initiator).map_err(|error| {
        CliError::Launch(format!("failed to encode initiator descriptor: {error}"))
    })?;
    let daemon = serde_json::to_vec(&response.daemon).map_err(|error| {
        CliError::Launch(format!("failed to encode daemon descriptor: {error}"))
    })?;
    let issued_at = response.challenge.issued_at_unix_ms.to_be_bytes();
    let expires_at = response.challenge.expires_at_unix_ms.to_be_bytes();
    encode_transcript(
        DAEMON_CHALLENGE_DOMAIN,
        &[
            ("initiator", initiator.as_slice()),
            (
                "initiator_instance_id",
                request.initiator_instance_id.as_bytes(),
            ),
            (
                "initiator_public_identity",
                request.initiator_public_identity.as_bytes().as_slice(),
            ),
            (
                "initiator_fingerprint",
                request.initiator_fingerprint.as_bytes().as_slice(),
            ),
            (
                "initiator_nonce",
                request.initiator_nonce.as_bytes().as_slice(),
            ),
            ("daemon", daemon.as_slice()),
            ("daemon_instance_id", response.daemon_instance_id.as_bytes()),
            (
                "daemon_public_identity",
                response.daemon_public_identity.as_bytes().as_slice(),
            ),
            (
                "daemon_fingerprint",
                response.daemon_fingerprint.as_bytes().as_slice(),
            ),
            ("challenge_id", response.challenge.id.as_bytes().as_slice()),
            (
                "challenge_nonce",
                response.challenge.nonce.as_bytes().as_slice(),
            ),
            ("issued_at_unix_ms", issued_at.as_slice()),
            ("expires_at_unix_ms", expires_at.as_slice()),
        ],
    )
    .map_err(|error| CliError::Launch(format!("failed to encode daemon challenge: {error}")))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RegistrationProof {
    pub(crate) transcript: HandshakeTranscript,
    pub(crate) initiator_proof: HandshakeProof,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct McpRegisterRequest {
    pub(crate) proof: RegistrationProof,
    pub(crate) worker_network: WorkerNetworkHintProof,
}

/// The MCP machine's daemon-reachable IPv4 address and optional prescribed worker port.
///
/// The daemon validates this signed hint and remains authoritative over the resulting bind and
/// advertise arguments in [`BrokerDirective::LaunchWorker`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct WorkerNetworkHint {
    pub(crate) advertised_host: String,
    pub(crate) port: Option<u16>,
}

impl WorkerNetworkHint {
    pub(crate) fn new(
        advertised_host: impl Into<String>,
        port: Option<u16>,
    ) -> Result<Self, CliError> {
        let hint = Self {
            advertised_host: advertised_host.into().to_ascii_lowercase(),
            port,
        };
        hint.validate()?;
        Ok(hint)
    }

    pub(crate) fn validate(&self) -> Result<(), CliError> {
        let host = self.advertised_host.as_str();
        let ipv4 = host.parse::<Ipv4Addr>().ok();
        let valid_hostname = host.len() <= 253
            && !host.is_empty()
            && host.is_ascii()
            && !host.contains(['/', ':', '@', '[', ']', '?', '#'])
            && host.split('.').all(|label| {
                !label.is_empty()
                    && label.len() <= 63
                    && !label.starts_with('-')
                    && !label.ends_with('-')
                    && label
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
            });
        if ipv4.is_some_and(|address| address.is_unspecified())
            || (ipv4.is_none() && !valid_hostname)
            || self.port == Some(0)
        {
            return Err(CliError::Config(
                "worker network hint requires a concrete hostname or IPv4 address and a nonzero prescribed port"
                    .into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn is_loopback(&self) -> bool {
        self.advertised_host.eq_ignore_ascii_case("localhost")
            || self
                .advertised_host
                .parse::<Ipv4Addr>()
                .is_ok_and(|address| address.is_loopback())
    }
}

/// A worker network hint bound to the authenticated MCP registration challenge.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct WorkerNetworkHintProof {
    pub(crate) hint: WorkerNetworkHint,
    pub(crate) signature: Ed25519Signature,
}

impl WorkerNetworkHintProof {
    pub(crate) fn sign(
        hint: WorkerNetworkHint,
        daemon_target: &str,
        mcp_instance_id: &str,
        challenge_id: &ChallengeId,
        fingerprint: &Fingerprint,
        identity: &MachineIdentity,
    ) -> Result<Self, CliError> {
        hint.validate()?;
        let canonical = worker_network_hint_bytes(
            &hint,
            daemon_target,
            mcp_instance_id,
            challenge_id,
            fingerprint,
        )?;
        Ok(Self {
            hint,
            signature: identity.sign(&canonical),
        })
    }

    pub(crate) fn verify(
        &self,
        daemon_target: &str,
        mcp_instance_id: &str,
        challenge_id: &ChallengeId,
        fingerprint: &Fingerprint,
        identity: &PublicIdentity,
    ) -> Result<(), CliError> {
        self.hint.validate()?;
        let canonical = worker_network_hint_bytes(
            &self.hint,
            daemon_target,
            mcp_instance_id,
            challenge_id,
            fingerprint,
        )?;
        identity
            .verify(&canonical, &self.signature)
            .map_err(|_| CliError::Unauthorized("invalid signed worker network hint".into()))
    }
}

fn worker_network_hint_bytes(
    hint: &WorkerNetworkHint,
    daemon_target: &str,
    mcp_instance_id: &str,
    challenge_id: &ChallengeId,
    fingerprint: &Fingerprint,
) -> Result<Vec<u8>, CliError> {
    let port_present = [u8::from(hint.port.is_some())];
    let port = hint.port.unwrap_or_default().to_be_bytes();
    encode_transcript(
        WORKER_NETWORK_HINT_DOMAIN,
        &[
            ("daemon_target", daemon_target.as_bytes()),
            ("mcp_instance_id", mcp_instance_id.as_bytes()),
            ("challenge_id", challenge_id.as_bytes().as_slice()),
            ("fingerprint", fingerprint.as_bytes().as_slice()),
            ("advertised_host", hint.advertised_host.as_bytes()),
            ("port_present", port_present.as_slice()),
            ("port", port.as_slice()),
        ],
    )
    .map_err(|error| CliError::Unauthorized(error.to_string()))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct McpRegisterResponse {
    pub(crate) daemon_proof: HandshakeProof,
    pub(crate) session_token: SensitiveString,
    pub(crate) heartbeat_interval_ms: u64,
    pub(crate) directive: BrokerDirective,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct WorkerBootstrap {
    pub(crate) activation_id: String,
    pub(crate) activation_token: SensitiveString,
    pub(crate) deadline_unix_ms: u64,
    pub(crate) bind_ip: Ipv4Addr,
    pub(crate) port: u16,
    pub(crate) advertise_address: Option<String>,
}

impl WorkerBootstrap {
    pub(crate) fn from_directive(directive: BrokerDirective) -> Option<Self> {
        let BrokerDirective::LaunchWorker {
            activation_id,
            activation_token,
            deadline_unix_ms,
            bind_ip,
            port,
            advertise_address,
        } = directive
        else {
            return None;
        };
        Some(Self {
            activation_id,
            activation_token,
            deadline_unix_ms,
            bind_ip,
            port,
            advertise_address,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct WorkerRegisterRequest {
    pub(crate) proof: RegistrationProof,
    pub(crate) worker_id: String,
    pub(crate) endpoint: String,
    pub(crate) activation_id: String,
    pub(crate) activation_token: SensitiveString,
    pub(crate) tls_root_certificate: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct WorkerRecoverRequest {
    pub(crate) proof: RegistrationProof,
    pub(crate) worker_id: String,
    pub(crate) endpoint: String,
    pub(crate) tls_root_certificate: Option<String>,
    pub(crate) generation_grant: WorkerGenerationGrant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct WorkerRegisterResponse {
    pub(crate) daemon_proof: HandshakeProof,
    pub(crate) session_token: SensitiveString,
    pub(crate) data_token: SensitiveString,
    pub(crate) heartbeat_interval_ms: u64,
    pub(crate) generation_grant: WorkerGenerationGrant,
}

/// A daemon-signed proof binding one worker generation to its endpoint and TLS trust anchor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct WorkerGenerationGrant {
    pub(crate) generation_id: String,
    pub(crate) worker_id: String,
    pub(crate) fingerprint: Fingerprint,
    pub(crate) endpoint: String,
    pub(crate) tls_root_digest: Option<[u8; 32]>,
    pub(crate) signature: Ed25519Signature,
}

impl WorkerGenerationGrant {
    pub(crate) fn issue(
        worker_id: &str,
        fingerprint: Fingerprint,
        endpoint: &str,
        tls_root_certificate: Option<&str>,
        daemon_identity: &MachineIdentity,
    ) -> Result<Self, CliError> {
        let generation_id = random_secret(16)?;
        let tls_root_digest =
            tls_root_certificate.map(|root| Sha256::digest(root.as_bytes()).into());
        let canonical = worker_generation_bytes(
            &generation_id,
            worker_id,
            &fingerprint,
            endpoint,
            tls_root_digest.as_ref(),
        )?;
        Ok(Self {
            generation_id,
            worker_id: worker_id.to_owned(),
            fingerprint,
            endpoint: endpoint.to_owned(),
            tls_root_digest,
            signature: daemon_identity.sign(&canonical),
        })
    }

    pub(crate) fn verify(
        &self,
        worker_id: &str,
        fingerprint: Fingerprint,
        endpoint: &str,
        tls_root_certificate: Option<&str>,
        daemon_identity: &PublicIdentity,
    ) -> Result<(), CliError> {
        let expected_root = tls_root_certificate.map(|root| Sha256::digest(root.as_bytes()).into());
        if self.generation_id.is_empty()
            || self.worker_id != worker_id
            || self.fingerprint != fingerprint
            || self.endpoint != endpoint
            || self.tls_root_digest != expected_root
        {
            return Err(CliError::Unauthorized(
                "worker generation grant does not match recovery".into(),
            ));
        }
        let canonical = worker_generation_bytes(
            &self.generation_id,
            &self.worker_id,
            &self.fingerprint,
            &self.endpoint,
            self.tls_root_digest.as_ref(),
        )?;
        daemon_identity
            .verify(&canonical, &self.signature)
            .map_err(|_| CliError::Unauthorized("invalid worker generation grant".into()))
    }
}

fn worker_generation_bytes(
    generation_id: &str,
    worker_id: &str,
    fingerprint: &Fingerprint,
    endpoint: &str,
    tls_root_digest: Option<&[u8; 32]>,
) -> Result<Vec<u8>, CliError> {
    let root_present = [u8::from(tls_root_digest.is_some())];
    encode_transcript(
        WORKER_GENERATION_DOMAIN,
        &[
            ("generation_id", generation_id.as_bytes()),
            ("worker_id", worker_id.as_bytes()),
            ("fingerprint", fingerprint.as_bytes().as_slice()),
            ("endpoint", endpoint.as_bytes()),
            ("tls_root_present", root_present.as_slice()),
            (
                "tls_root_digest",
                tls_root_digest.map_or(&[][..], |digest| digest.as_slice()),
            ),
        ],
    )
    .map_err(|error| CliError::Unauthorized(error.to_string()))
}

/// A session-authenticated message. Sequence numbers are strictly increasing per session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SessionRequest<T> {
    pub(crate) session_id: String,
    pub(crate) session_token: SensitiveString,
    pub(crate) request_id: String,
    pub(crate) sequence: u64,
    pub(crate) payload_sha256: [u8; 32],
    pub(crate) payload: T,
}

impl<T: Serialize> SessionRequest<T> {
    pub(crate) fn new(
        session_id: String,
        session_token: SensitiveString,
        sequence: u64,
        payload: T,
    ) -> Result<Self, CliError> {
        let encoded = serde_json::to_vec(&payload).map_err(|error| {
            CliError::Launch(format!("failed to encode daemon control payload: {error}"))
        })?;
        Ok(Self {
            session_id,
            session_token,
            request_id: random_secret(16)?,
            sequence,
            payload_sha256: Sha256::digest(encoded).into(),
            payload,
        })
    }

    pub(crate) fn validate_payload_hash(&self) -> bool {
        serde_json::to_vec(&self.payload)
            .map(|encoded| {
                let actual: [u8; 32] = Sha256::digest(encoded).into();
                subtle::ConstantTimeEq::ct_eq(actual.as_slice(), self.payload_sha256.as_slice())
                    .into()
            })
            .unwrap_or(false)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct EmptyPayload {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ActivationFailedPayload {
    pub(crate) activation_id: String,
    pub(crate) reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct McpHeartbeatResponse {
    pub(crate) directive: Option<BrokerDirective>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct WorkerHeartbeatPayload {
    pub(crate) worker_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct WorkerReadyPayload {
    pub(crate) worker_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct WorkerDrainRequest {
    pub(crate) worker_id: String,
    /// Daemon wall-clock deadline retained for protocol-v1 compatibility and audit logs.
    pub(crate) deadline_unix_ms: u64,
    /// Relative lifetime enforced against the worker's local monotonic clock.
    #[serde(default)]
    pub(crate) timeout_ms: Option<u64>,
}

pub(crate) fn random_secret(bytes: usize) -> Result<String, CliError> {
    let mut value = vec![0_u8; bytes];
    SystemRandom::new()
        .fill(&mut value)
        .map_err(|_| CliError::Launch("failed to generate daemon session credential".into()))?;
    Ok(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(value))
}

pub(crate) fn fresh_nonce() -> Result<ChallengeNonce, CliError> {
    let record = super::identity::ChallengeRecord::generate(now_unix_ms(), 1).map_err(|error| {
        CliError::Launch(format!("failed to generate handshake nonce: {error}"))
    })?;
    Ok(record.challenge().nonce)
}

pub(crate) fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

pub(crate) fn descriptor(role: ComponentRole) -> ComponentDescriptor {
    ComponentDescriptor::nemo_relay(
        role,
        super::protocol::ProtocolRange::default(),
        super::protocol::Capabilities::streaming_transport(),
        env!("CARGO_PKG_VERSION"),
    )
}

#[cfg(test)]
#[path = "../../../tests/coverage/daemon/control_tests.rs"]
mod tests;
