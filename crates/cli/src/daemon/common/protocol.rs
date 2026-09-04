// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;
use std::fmt;
use std::net::Ipv4Addr;

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize};
use subtle::ConstantTimeEq;
use thiserror::Error;

use super::identity::{
    ChallengeId, ChallengeNonce, Ed25519Signature, Fingerprint, IdentityError, MachineIdentity,
    PublicIdentity, TokenDigest, encode_transcript,
};

pub(crate) const SERVICE_NAME: &str = "nemo-relay";
pub(crate) const PROTOCOL_V1: u16 = 1;
const HANDSHAKE_DOMAIN: &[u8] = b"nemo-relay/daemon-handshake/v1";
const MAX_CAPABILITIES: usize = 64;
const MAX_CAPABILITY_BYTES: usize = 128;
const MAX_BINARY_VERSION_BYTES: usize = 256;
const MAX_INSTANCE_ID_BYTES: usize = 256;
const MAX_DAEMON_TARGET_BYTES: usize = 2_048;

/// The authenticated role of a daemon-protocol participant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ComponentRole {
    Daemon,
    Mcp,
    Worker,
}

impl ComponentRole {
    const fn transcript_name(self) -> &'static str {
        match self {
            Self::Daemon => "daemon",
            Self::Mcp => "mcp",
            Self::Worker => "worker",
        }
    }
}

/// An inclusive range of daemon-protocol versions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ProtocolRange {
    pub(crate) minimum: u16,
    pub(crate) maximum: u16,
}

impl ProtocolRange {
    /// Constructs a validated inclusive protocol range.
    #[cfg(test)]
    pub(crate) fn new(minimum: u16, maximum: u16) -> Result<Self, ProtocolError> {
        let range = Self { minimum, maximum };
        range.validate()?;
        Ok(range)
    }

    /// Returns the highest mutually supported protocol version.
    pub(crate) fn negotiate(self, peer: Self) -> Result<u16, ProtocolError> {
        self.validate()?;
        peer.validate()?;
        let minimum = self.minimum.max(peer.minimum);
        let maximum = self.maximum.min(peer.maximum);
        (minimum <= maximum)
            .then_some(maximum)
            .ok_or(ProtocolError::NoProtocolOverlap)
    }

    /// Reports whether this range contains one protocol version.
    pub(crate) const fn contains(self, version: u16) -> bool {
        version >= self.minimum && version <= self.maximum
    }

    fn validate(self) -> Result<(), ProtocolError> {
        if self.minimum == 0 || self.minimum > self.maximum {
            return Err(ProtocolError::InvalidProtocolRange);
        }
        Ok(())
    }
}

impl Default for ProtocolRange {
    fn default() -> Self {
        Self {
            minimum: PROTOCOL_V1,
            maximum: PROTOCOL_V1,
        }
    }
}

/// A forward-compatible, deterministically ordered set of protocol capabilities.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub(crate) struct Capabilities(BTreeSet<String>);

impl Capabilities {
    /// Constructs a capability set and validates every capability name.
    pub(crate) fn new(
        names: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<Self, ProtocolError> {
        let capabilities = Self(names.into_iter().map(Into::into).collect());
        capabilities.validate()?;
        Ok(capabilities)
    }

    /// Returns the baseline lossless HTTP transport capabilities.
    pub(crate) fn streaming_transport() -> Self {
        Self::new([
            "http1",
            "http2",
            "streaming_body_frames",
            "sse_passthrough",
            "trailers",
        ])
        .expect("built-in capability names are valid")
    }

    #[cfg(test)]
    pub(crate) fn contains(&self, name: &str) -> bool {
        self.0.contains(name)
    }

    /// Reports whether this set includes every required capability.
    pub(crate) fn includes(&self, required: &Self) -> bool {
        required.0.is_subset(&self.0)
    }

    fn validate(&self) -> Result<(), ProtocolError> {
        if self.0.len() > MAX_CAPABILITIES
            || self
                .0
                .iter()
                .any(|name| name.len() > MAX_CAPABILITY_BYTES || !valid_capability_name(name))
        {
            return Err(ProtocolError::InvalidCapability);
        }
        Ok(())
    }

    fn canonical_bytes(&self) -> Result<Vec<u8>, ProtocolError> {
        self.validate()?;
        let mut encoded = Vec::new();
        let count = u32::try_from(self.0.len()).map_err(|_| ProtocolError::FieldTooLarge)?;
        encoded.extend_from_slice(&count.to_be_bytes());
        for capability in &self.0 {
            let length =
                u32::try_from(capability.len()).map_err(|_| ProtocolError::FieldTooLarge)?;
            encoded.extend_from_slice(&length.to_be_bytes());
            encoded.extend_from_slice(capability.as_bytes());
        }
        Ok(encoded)
    }
}

/// Authenticated metadata describing one protocol component.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ComponentDescriptor {
    pub(crate) service: String,
    pub(crate) role: ComponentRole,
    pub(crate) protocol: ProtocolRange,
    pub(crate) capabilities: Capabilities,
    pub(crate) binary_version: String,
}

impl ComponentDescriptor {
    /// Constructs a descriptor for a real NeMo Relay component.
    pub(crate) fn nemo_relay(
        role: ComponentRole,
        protocol: ProtocolRange,
        capabilities: Capabilities,
        binary_version: impl Into<String>,
    ) -> Self {
        Self {
            service: SERVICE_NAME.to_owned(),
            role,
            protocol,
            capabilities,
            binary_version: binary_version.into(),
        }
    }

    /// Validates invariants that must hold regardless of binary release version.
    pub(crate) fn validate(&self) -> Result<(), ProtocolError> {
        if self.service != SERVICE_NAME {
            return Err(ProtocolError::WrongService);
        }
        self.protocol.validate()?;
        self.capabilities.validate()?;
        if self.binary_version.is_empty() || self.binary_version.len() > MAX_BINARY_VERSION_BYTES {
            return Err(ProtocolError::MissingBinaryVersion);
        }
        Ok(())
    }
}

/// A sensitive wire value whose debug output is always redacted.
#[derive(Clone, Serialize)]
#[serde(transparent)]
pub(crate) struct SensitiveString(String);

impl SensitiveString {
    /// Constructs a non-empty sensitive string.
    pub(crate) fn new(value: impl Into<String>) -> Result<Self, ProtocolError> {
        let value = value.into();
        if value.is_empty() {
            return Err(ProtocolError::MissingSensitiveValue);
        }
        Ok(Self(value))
    }

    /// Exposes the value only at the protocol boundary that consumes it.
    pub(crate) fn expose(&self) -> &str {
        &self.0
    }
}

impl PartialEq for SensitiveString {
    fn eq(&self, other: &Self) -> bool {
        bool::from(self.0.as_bytes().ct_eq(other.0.as_bytes()))
    }
}

impl Eq for SensitiveString {}

impl<'de> Deserialize<'de> for SensitiveString {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(D::Error::custom)
    }
}

impl fmt::Debug for SensitiveString {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("[REDACTED]")
    }
}

/// The canonical transcript signed by both sides of MCP or worker registration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct HandshakeTranscript {
    pub(crate) daemon_target: String,
    pub(crate) initiator: ComponentDescriptor,
    pub(crate) responder: ComponentDescriptor,
    pub(crate) initiator_instance_id: String,
    pub(crate) responder_instance_id: String,
    pub(crate) selected_protocol: u16,
    pub(crate) initiator_public_identity: PublicIdentity,
    pub(crate) responder_public_identity: PublicIdentity,
    pub(crate) initiator_fingerprint: Fingerprint,
    pub(crate) responder_fingerprint: Fingerprint,
    pub(crate) challenge_id: ChallengeId,
    pub(crate) initiator_nonce: ChallengeNonce,
    pub(crate) responder_nonce: ChallengeNonce,
    pub(crate) route_token_digest: Option<TokenDigest>,
}

impl HandshakeTranscript {
    /// Validates identities, service names, roles, and negotiated protocol values.
    pub(crate) fn validate(&self) -> Result<(), ProtocolError> {
        self.initiator.validate()?;
        self.responder.validate()?;
        if self.initiator.role == ComponentRole::Daemon
            || self.responder.role != ComponentRole::Daemon
        {
            return Err(ProtocolError::InvalidRolePair);
        }
        if self.daemon_target.is_empty()
            || self.daemon_target.len() > MAX_DAEMON_TARGET_BYTES
            || self.initiator_instance_id.is_empty()
            || self.initiator_instance_id.len() > MAX_INSTANCE_ID_BYTES
            || self.responder_instance_id.is_empty()
            || self.responder_instance_id.len() > MAX_INSTANCE_ID_BYTES
        {
            return Err(ProtocolError::MissingTranscriptIdentity);
        }
        if !self.initiator.protocol.contains(self.selected_protocol)
            || !self.responder.protocol.contains(self.selected_protocol)
        {
            return Err(ProtocolError::InvalidSelectedProtocol);
        }
        if self.initiator_public_identity.fingerprint() != self.initiator_fingerprint
            || self.responder_public_identity.fingerprint() != self.responder_fingerprint
        {
            return Err(ProtocolError::FingerprintMismatch);
        }
        if self.initiator.role == ComponentRole::Mcp && self.route_token_digest.is_none() {
            return Err(ProtocolError::MissingRouteTokenDigest);
        }
        Ok(())
    }

    /// Encodes all signed fields deterministically and independently of JSON serialization.
    pub(crate) fn canonical_bytes(&self) -> Result<Vec<u8>, ProtocolError> {
        self.validate()?;
        let initiator_protocol_minimum = self.initiator.protocol.minimum.to_be_bytes();
        let initiator_protocol_maximum = self.initiator.protocol.maximum.to_be_bytes();
        let responder_protocol_minimum = self.responder.protocol.minimum.to_be_bytes();
        let responder_protocol_maximum = self.responder.protocol.maximum.to_be_bytes();
        let selected_protocol = self.selected_protocol.to_be_bytes();
        let initiator_capabilities = self.initiator.capabilities.canonical_bytes()?;
        let responder_capabilities = self.responder.capabilities.canonical_bytes()?;
        let route_token_present = [u8::from(self.route_token_digest.is_some())];
        let route_token_digest = self
            .route_token_digest
            .as_ref()
            .map_or(&[][..], |digest| digest.as_bytes().as_slice());
        let fields = [
            ("daemon_target", self.daemon_target.as_bytes()),
            ("initiator_service", self.initiator.service.as_bytes()),
            (
                "initiator_role",
                self.initiator.role.transcript_name().as_bytes(),
            ),
            (
                "initiator_protocol_minimum",
                initiator_protocol_minimum.as_slice(),
            ),
            (
                "initiator_protocol_maximum",
                initiator_protocol_maximum.as_slice(),
            ),
            ("initiator_capabilities", initiator_capabilities.as_slice()),
            (
                "initiator_binary_version",
                self.initiator.binary_version.as_bytes(),
            ),
            ("responder_service", self.responder.service.as_bytes()),
            (
                "responder_role",
                self.responder.role.transcript_name().as_bytes(),
            ),
            (
                "responder_protocol_minimum",
                responder_protocol_minimum.as_slice(),
            ),
            (
                "responder_protocol_maximum",
                responder_protocol_maximum.as_slice(),
            ),
            ("responder_capabilities", responder_capabilities.as_slice()),
            (
                "responder_binary_version",
                self.responder.binary_version.as_bytes(),
            ),
            (
                "initiator_instance_id",
                self.initiator_instance_id.as_bytes(),
            ),
            (
                "responder_instance_id",
                self.responder_instance_id.as_bytes(),
            ),
            ("selected_protocol", selected_protocol.as_slice()),
            (
                "initiator_public_identity",
                self.initiator_public_identity.as_bytes().as_slice(),
            ),
            (
                "responder_public_identity",
                self.responder_public_identity.as_bytes().as_slice(),
            ),
            (
                "initiator_fingerprint",
                self.initiator_fingerprint.as_bytes().as_slice(),
            ),
            (
                "responder_fingerprint",
                self.responder_fingerprint.as_bytes().as_slice(),
            ),
            ("challenge_id", self.challenge_id.as_bytes().as_slice()),
            (
                "initiator_nonce",
                self.initiator_nonce.as_bytes().as_slice(),
            ),
            (
                "responder_nonce",
                self.responder_nonce.as_bytes().as_slice(),
            ),
            ("route_token_present", route_token_present.as_slice()),
            ("route_token_digest", route_token_digest),
        ];
        encode_transcript(HANDSHAKE_DOMAIN, &fields).map_err(ProtocolError::Transcript)
    }

    /// Signs the canonical transcript for one of its declared participants.
    pub(crate) fn sign(
        &self,
        signer: ComponentRole,
        identity: &MachineIdentity,
    ) -> Result<HandshakeProof, ProtocolError> {
        let expected_identity = self.identity_for_role(signer)?;
        if identity.public_identity() != expected_identity {
            return Err(ProtocolError::SignerIdentityMismatch);
        }
        Ok(HandshakeProof {
            signer,
            signature: identity.sign(&self.canonical_bytes()?),
        })
    }

    /// Verifies that a proof signs this exact canonical transcript.
    pub(crate) fn verify(&self, proof: &HandshakeProof) -> Result<(), ProtocolError> {
        self.identity_for_role(proof.signer)?
            .verify(&self.canonical_bytes()?, &proof.signature)
            .map_err(ProtocolError::Identity)
    }

    fn identity_for_role(&self, role: ComponentRole) -> Result<PublicIdentity, ProtocolError> {
        if self.initiator.role == role {
            return Ok(self.initiator_public_identity);
        }
        if self.responder.role == role {
            return Ok(self.responder_public_identity);
        }
        Err(ProtocolError::UnknownSignerRole)
    }
}

/// A participant's signature over a complete handshake transcript.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct HandshakeProof {
    pub(crate) signer: ComponentRole,
    pub(crate) signature: Ed25519Signature,
}

/// A daemon-issued plan for launching one worker.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct WorkerLaunch {
    pub(crate) activation_id: String,
    pub(crate) activation_token: SensitiveString,
    pub(crate) deadline_unix_ms: u64,
    pub(crate) bind_ip: Ipv4Addr,
    pub(crate) port: u16,
    pub(crate) advertise_address: Option<String>,
}

impl WorkerLaunch {
    /// Converts the launch plan into its wire directive.
    pub(crate) fn into_directive(self) -> BrokerDirective {
        BrokerDirective::LaunchWorker {
            activation_id: self.activation_id,
            activation_token: self.activation_token,
            deadline_unix_ms: self.deadline_unix_ms,
            bind_ip: self.bind_ip,
            port: self.port,
            advertise_address: self.advertise_address,
        }
    }
}

/// The daemon's authoritative instruction for an MCP session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "directive", rename_all = "snake_case")]
pub(crate) enum BrokerDirective {
    ReuseWorker {
        endpoint: String,
    },
    WaitForWorker {
        retry_after_ms: u64,
    },
    LaunchWorker {
        activation_id: String,
        activation_token: SensitiveString,
        deadline_unix_ms: u64,
        bind_ip: Ipv4Addr,
        port: u16,
        advertise_address: Option<String>,
    },
    UsePassThrough,
}

/// Protocol construction or verification failures.
#[derive(Debug, Error, PartialEq, Eq)]
pub(crate) enum ProtocolError {
    #[error("the peer did not identify itself as nemo-relay")]
    WrongService,
    #[error("the protocol version range is invalid")]
    InvalidProtocolRange,
    #[error("the peers have no overlapping daemon protocol version")]
    NoProtocolOverlap,
    #[error("the selected protocol version is not supported by both peers")]
    InvalidSelectedProtocol,
    #[error("the component binary version is missing")]
    MissingBinaryVersion,
    #[error("the capability set contains an invalid name")]
    InvalidCapability,
    #[error("the handshake role pair must be MCP/daemon or worker/daemon")]
    InvalidRolePair,
    #[error("the handshake is missing a daemon target or component instance ID")]
    MissingTranscriptIdentity,
    #[error("a public identity does not match its advertised fingerprint")]
    FingerprintMismatch,
    #[error("an MCP handshake is missing its route-token digest")]
    MissingRouteTokenDigest,
    #[error("the signing key does not match the transcript participant")]
    SignerIdentityMismatch,
    #[error("the proof signer is not a participant in this handshake")]
    UnknownSignerRole,
    #[error("a required sensitive protocol value is empty")]
    MissingSensitiveValue,
    #[error("a protocol field is too large")]
    FieldTooLarge,
    #[error(transparent)]
    Identity(IdentityError),
    #[error("failed to encode the signed transcript: {0}")]
    Transcript(IdentityError),
}

fn valid_capability_name(name: &str) -> bool {
    !name.is_empty()
        && name.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || b"._-".contains(&byte)
        })
}

#[cfg(test)]
#[path = "../../../tests/coverage/daemon/protocol_tests.rs"]
mod tests;
