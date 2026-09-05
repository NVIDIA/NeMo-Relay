// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use ring::rand::{SecureRandom, SystemRandom};
use ring::signature::{ED25519, Ed25519KeyPair, KeyPair, UnparsedPublicKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use thiserror::Error;

const ED25519_PUBLIC_KEY_BYTES: usize = 32;
const CHALLENGE_ID_BYTES: usize = 16;
const CHALLENGE_NONCE_BYTES: usize = 32;
const TRANSCRIPT_MAGIC: &[u8] = b"NEMO-RELAY-SIGNED-TRANSCRIPT\0";

/// An Ed25519 identity used by one daemon component.
#[derive(Clone)]
pub(crate) struct MachineIdentity {
    key_pair: Arc<Ed25519KeyPair>,
}

impl MachineIdentity {
    /// Generates an identity and returns its PKCS#8 document for owner-private storage.
    pub(crate) fn generate() -> Result<GeneratedMachineIdentity, IdentityError> {
        let random = SystemRandom::new();
        let document =
            Ed25519KeyPair::generate_pkcs8(&random).map_err(|_| IdentityError::KeyGeneration)?;
        let identity = Self::from_pkcs8(document.as_ref())?;
        Ok(GeneratedMachineIdentity {
            identity,
            pkcs8: document.as_ref().to_vec(),
        })
    }

    /// Loads an identity from an unencrypted PKCS#8 Ed25519 document.
    pub(crate) fn from_pkcs8(pkcs8: &[u8]) -> Result<Self, IdentityError> {
        let key_pair =
            Ed25519KeyPair::from_pkcs8(pkcs8).map_err(|_| IdentityError::InvalidPrivateKey)?;
        Ok(Self {
            key_pair: Arc::new(key_pair),
        })
    }

    /// Returns the public half of this identity.
    pub(crate) fn public_identity(&self) -> PublicIdentity {
        let bytes = self
            .key_pair
            .public_key()
            .as_ref()
            .try_into()
            .expect("ring Ed25519 public keys have a fixed length");
        PublicIdentity(bytes)
    }

    /// Returns the stable SHA-256 fingerprint of the public identity.
    pub(crate) fn fingerprint(&self) -> Fingerprint {
        self.public_identity().fingerprint()
    }

    /// Signs already-canonical transcript bytes.
    pub(crate) fn sign(&self, transcript: &[u8]) -> Ed25519Signature {
        Ed25519Signature(self.key_pair.sign(transcript).as_ref().to_vec())
    }
}

impl fmt::Debug for MachineIdentity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MachineIdentity")
            .field("fingerprint", &self.fingerprint())
            .finish_non_exhaustive()
    }
}

/// A newly generated identity and the private document that must be persisted securely.
pub(crate) struct GeneratedMachineIdentity {
    pub(crate) identity: MachineIdentity,
    pub(crate) pkcs8: Vec<u8>,
}

/// An Ed25519 public identity suitable for control-protocol serialization.
#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct PublicIdentity([u8; ED25519_PUBLIC_KEY_BYTES]);

impl PublicIdentity {
    /// Parses an Ed25519 public key.
    #[cfg(test)]
    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self, IdentityError> {
        let bytes = bytes
            .try_into()
            .map_err(|_| IdentityError::InvalidPublicKey)?;
        Ok(Self(bytes))
    }

    /// Returns the raw Ed25519 public-key bytes.
    pub(crate) const fn as_bytes(&self) -> &[u8; ED25519_PUBLIC_KEY_BYTES] {
        &self.0
    }

    /// Returns the stable SHA-256 fingerprint of this public key.
    pub(crate) fn fingerprint(&self) -> Fingerprint {
        Fingerprint(sha256(&self.0))
    }

    /// Verifies a signature over already-canonical transcript bytes.
    pub(crate) fn verify(
        &self,
        transcript: &[u8],
        signature: &Ed25519Signature,
    ) -> Result<(), IdentityError> {
        UnparsedPublicKey::new(&ED25519, self.0)
            .verify(transcript, signature.as_bytes())
            .map_err(|_| IdentityError::SignatureVerification)
    }
}

impl fmt::Debug for PublicIdentity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("PublicIdentity")
            .field(&self.fingerprint())
            .finish()
    }
}

/// A serialized Ed25519 signature.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub(crate) struct Ed25519Signature(Vec<u8>);

impl Ed25519Signature {
    /// Returns the signature bytes.
    pub(crate) fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

impl fmt::Debug for Ed25519Signature {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Ed25519Signature")
            .field("length", &self.0.len())
            .finish_non_exhaustive()
    }
}

/// A stable public-key fingerprint used as the broker route key.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub(crate) struct Fingerprint([u8; 32]);

impl Fingerprint {
    /// Returns the raw SHA-256 digest.
    pub(crate) const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Debug for Fingerprint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Fingerprint({self})")
    }
}

impl fmt::Display for Fingerprint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_hex(formatter, &self.0)
    }
}

/// The SHA-256 digest of the per-user-machine route token.
#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(transparent)]
pub(crate) struct TokenDigest([u8; 32]);

impl TokenDigest {
    /// Hashes the exact token bytes received from the environment or HTTP header.
    pub(crate) fn from_token(token: &[u8]) -> Self {
        Self(sha256(token))
    }

    /// Returns the raw SHA-256 digest.
    pub(crate) const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Compares two token digests in constant time.
    pub(crate) fn matches(&self, other: &Self) -> bool {
        bool::from(self.0.ct_eq(&other.0))
    }
}

impl PartialEq for TokenDigest {
    fn eq(&self, other: &Self) -> bool {
        self.matches(other)
    }
}

impl Eq for TokenDigest {}

impl Hash for TokenDigest {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl fmt::Debug for TokenDigest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "TokenDigest({self})")
    }
}

impl fmt::Display for TokenDigest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_hex(formatter, &self.0)
    }
}

/// A random identifier for one daemon challenge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub(crate) struct ChallengeId([u8; CHALLENGE_ID_BYTES]);

impl ChallengeId {
    /// Returns the identifier bytes.
    pub(crate) const fn as_bytes(&self) -> &[u8; CHALLENGE_ID_BYTES] {
        &self.0
    }
}

/// A random challenge nonce.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub(crate) struct ChallengeNonce([u8; CHALLENGE_NONCE_BYTES]);

impl ChallengeNonce {
    /// Returns the nonce bytes.
    pub(crate) const fn as_bytes(&self) -> &[u8; CHALLENGE_NONCE_BYTES] {
        &self.0
    }
}

/// The wire-safe portion of a challenge record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Challenge {
    pub(crate) id: ChallengeId,
    pub(crate) nonce: ChallengeNonce,
    pub(crate) issued_at_unix_ms: u64,
    pub(crate) expires_at_unix_ms: u64,
}

/// A one-use local challenge record with explicit expiry handling.
#[derive(Debug)]
pub(crate) struct ChallengeRecord {
    challenge: Challenge,
    consumed: bool,
}

impl ChallengeRecord {
    /// Creates a random challenge using caller-supplied wall-clock values.
    pub(crate) fn generate(
        issued_at_unix_ms: u64,
        lifetime_ms: u64,
    ) -> Result<Self, IdentityError> {
        let expires_at_unix_ms = issued_at_unix_ms
            .checked_add(lifetime_ms)
            .ok_or(IdentityError::ChallengeLifetimeOverflow)?;
        let random = SystemRandom::new();
        let mut id = [0_u8; CHALLENGE_ID_BYTES];
        let mut nonce = [0_u8; CHALLENGE_NONCE_BYTES];
        random
            .fill(&mut id)
            .and_then(|()| random.fill(&mut nonce))
            .map_err(|_| IdentityError::ChallengeGeneration)?;
        Ok(Self::from_challenge(Challenge {
            id: ChallengeId(id),
            nonce: ChallengeNonce(nonce),
            issued_at_unix_ms,
            expires_at_unix_ms,
        }))
    }

    /// Wraps a challenge for tracking. Primarily useful when restoring an issued challenge.
    pub(crate) fn from_challenge(challenge: Challenge) -> Self {
        Self {
            challenge,
            consumed: false,
        }
    }

    /// Returns the challenge sent to the peer.
    pub(crate) const fn challenge(&self) -> Challenge {
        self.challenge
    }

    /// Consumes this challenge exactly once before its expiry time.
    pub(crate) fn consume(
        &mut self,
        presented_id: &ChallengeId,
        now_unix_ms: u64,
    ) -> Result<Challenge, ChallengeError> {
        if !bool::from(self.challenge.id.0.ct_eq(&presented_id.0)) {
            return Err(ChallengeError::IdentifierMismatch);
        }
        if self.consumed {
            return Err(ChallengeError::Replay);
        }
        if now_unix_ms >= self.challenge.expires_at_unix_ms {
            self.consumed = true;
            return Err(ChallengeError::Expired);
        }
        self.consumed = true;
        Ok(self.challenge)
    }
}

/// Identity and transcript construction failures.
#[derive(Debug, Error, PartialEq, Eq)]
pub(crate) enum IdentityError {
    #[error("failed to generate an Ed25519 identity")]
    KeyGeneration,
    #[error("the Ed25519 private key is invalid")]
    InvalidPrivateKey,
    #[cfg(test)]
    #[error("the Ed25519 public key is invalid")]
    InvalidPublicKey,
    #[error("the Ed25519 signature did not verify")]
    SignatureVerification,
    #[error("failed to generate a handshake challenge")]
    ChallengeGeneration,
    #[error("the handshake challenge lifetime overflowed")]
    ChallengeLifetimeOverflow,
    #[error("a signed transcript field is too large")]
    TranscriptFieldTooLarge,
}

/// Challenge rejection reasons that callers can map to typed protocol errors.
#[derive(Debug, Error, PartialEq, Eq)]
pub(crate) enum ChallengeError {
    #[error("the challenge identifier does not match")]
    IdentifierMismatch,
    #[error("the challenge has already been consumed")]
    Replay,
    #[error("the challenge has expired")]
    Expired,
}

/// Encodes a signed transcript without relying on JSON map order or host endianness.
pub(crate) fn encode_transcript(
    domain: &[u8],
    fields: &[(&str, &[u8])],
) -> Result<Vec<u8>, IdentityError> {
    let mut encoded = Vec::with_capacity(
        TRANSCRIPT_MAGIC.len()
            + domain.len()
            + fields
                .iter()
                .map(|(name, value)| name.len() + value.len() + 16)
                .sum::<usize>(),
    );
    encoded.extend_from_slice(TRANSCRIPT_MAGIC);
    append_length_prefixed(&mut encoded, domain)?;
    let field_count =
        u32::try_from(fields.len()).map_err(|_| IdentityError::TranscriptFieldTooLarge)?;
    encoded.extend_from_slice(&field_count.to_be_bytes());
    for (name, value) in fields {
        append_length_prefixed(&mut encoded, name.as_bytes())?;
        append_length_prefixed(&mut encoded, value)?;
    }
    Ok(encoded)
}

fn append_length_prefixed(encoded: &mut Vec<u8>, value: &[u8]) -> Result<(), IdentityError> {
    let length = u64::try_from(value.len()).map_err(|_| IdentityError::TranscriptFieldTooLarge)?;
    encoded.extend_from_slice(&length.to_be_bytes());
    encoded.extend_from_slice(value);
    Ok(())
}

fn sha256(value: &[u8]) -> [u8; 32] {
    Sha256::digest(value).into()
}

fn write_hex(formatter: &mut fmt::Formatter<'_>, bytes: &[u8]) -> fmt::Result {
    for byte in bytes {
        write!(formatter, "{byte:02x}")?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../../tests/coverage/daemon/identity_tests.rs"]
mod tests;
