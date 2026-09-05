// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn generated_identity_round_trips_and_verifies() {
    let generated = MachineIdentity::generate().expect("identity");
    let restored = MachineIdentity::from_pkcs8(&generated.pkcs8).expect("restored identity");
    assert_eq!(generated.identity.fingerprint(), restored.fingerprint());

    let transcript = b"canonical transcript";
    let signature = generated.identity.sign(transcript);
    restored
        .public_identity()
        .verify(transcript, &signature)
        .expect("valid signature");
    assert_eq!(
        restored
            .public_identity()
            .verify(b"changed transcript", &signature),
        Err(IdentityError::SignatureVerification)
    );
}

#[test]
fn public_identity_rejects_the_wrong_length() {
    assert_eq!(
        PublicIdentity::from_bytes(&[0_u8; 31]),
        Err(IdentityError::InvalidPublicKey)
    );
}

#[test]
fn fingerprint_and_token_digest_are_stable() {
    let public = PublicIdentity::from_bytes(&[7_u8; 32]).expect("public key");
    assert_eq!(
        public.fingerprint().to_string(),
        "4bb06f8e4e3a7715d201d573d0aa423762e55dabd61a2c02278fa56cc6d294e0"
    );
    assert_eq!(
        TokenDigest::from_token(b"route token").to_string(),
        "fdd50053ddd4f9762b19d688e79add7403e4c354bb81430aaf25d3041f5c84e3"
    );
}

#[test]
fn transcript_encoding_is_domain_separated_and_length_prefixed() {
    let encoded = encode_transcript(b"test", &[("a", b"b"), ("cd", b"ef")]).expect("transcript");
    let mut expected = TRANSCRIPT_MAGIC.to_vec();
    expected.extend_from_slice(&4_u64.to_be_bytes());
    expected.extend_from_slice(b"test");
    expected.extend_from_slice(&2_u32.to_be_bytes());
    expected.extend_from_slice(&1_u64.to_be_bytes());
    expected.extend_from_slice(b"a");
    expected.extend_from_slice(&1_u64.to_be_bytes());
    expected.extend_from_slice(b"b");
    expected.extend_from_slice(&2_u64.to_be_bytes());
    expected.extend_from_slice(b"cd");
    expected.extend_from_slice(&2_u64.to_be_bytes());
    expected.extend_from_slice(b"ef");
    assert_eq!(encoded, expected);

    let other_domain =
        encode_transcript(b"other", &[("a", b"b"), ("cd", b"ef")]).expect("transcript");
    assert_ne!(encoded, other_domain);
}

#[test]
fn challenge_is_single_use_and_expires_at_the_boundary() {
    let challenge = Challenge {
        id: ChallengeId([1; CHALLENGE_ID_BYTES]),
        nonce: ChallengeNonce([2; CHALLENGE_NONCE_BYTES]),
        issued_at_unix_ms: 100,
        expires_at_unix_ms: 200,
    };
    let mut record = ChallengeRecord::from_challenge(challenge);
    assert_eq!(record.consume(&challenge.id, 199), Ok(challenge));
    assert_eq!(
        record.consume(&challenge.id, 199),
        Err(ChallengeError::Replay)
    );

    let mut expired = ChallengeRecord::from_challenge(challenge);
    assert_eq!(
        expired.consume(&challenge.id, 200),
        Err(ChallengeError::Expired)
    );
    assert_eq!(
        expired.consume(&challenge.id, 199),
        Err(ChallengeError::Replay)
    );
}

#[test]
fn mismatched_challenge_does_not_consume_record() {
    let challenge = Challenge {
        id: ChallengeId([1; CHALLENGE_ID_BYTES]),
        nonce: ChallengeNonce([2; CHALLENGE_NONCE_BYTES]),
        issued_at_unix_ms: 100,
        expires_at_unix_ms: 200,
    };
    let mut record = ChallengeRecord::from_challenge(challenge);
    assert_eq!(
        record.consume(&ChallengeId([3; CHALLENGE_ID_BYTES]), 150),
        Err(ChallengeError::IdentifierMismatch)
    );
    assert_eq!(record.consume(&challenge.id, 150), Ok(challenge));
}
