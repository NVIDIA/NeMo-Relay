// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::daemon::common::identity::{ChallengeRecord, TokenDigest};

fn sample_transcript() -> (HandshakeTranscript, MachineIdentity, MachineIdentity) {
    let initiator = MachineIdentity::generate().expect("initiator").identity;
    let responder = MachineIdentity::generate().expect("responder").identity;
    let challenge = ChallengeRecord::generate(10, 100)
        .expect("challenge")
        .challenge();
    (
        HandshakeTranscript {
            daemon_target: "https://relay.example:443".to_owned(),
            initiator: ComponentDescriptor::nemo_relay(
                ComponentRole::Mcp,
                ProtocolRange::default(),
                Capabilities::streaming_transport(),
                "0.9.0",
            ),
            responder: ComponentDescriptor::nemo_relay(
                ComponentRole::Daemon,
                ProtocolRange::default(),
                Capabilities::streaming_transport(),
                "2.0.0",
            ),
            initiator_instance_id: "mcp-1".to_owned(),
            responder_instance_id: "daemon-1".to_owned(),
            selected_protocol: PROTOCOL_V1,
            initiator_public_identity: initiator.public_identity(),
            responder_public_identity: responder.public_identity(),
            initiator_fingerprint: initiator.fingerprint(),
            responder_fingerprint: responder.fingerprint(),
            challenge_id: challenge.id,
            initiator_nonce: challenge.nonce,
            responder_nonce: challenge.nonce,
            route_token_digest: Some(TokenDigest::from_token(b"token")),
        },
        initiator,
        responder,
    )
}

#[test]
fn negotiation_selects_highest_overlap_without_using_binary_version() {
    assert_eq!(
        ProtocolRange::new(1, 4)
            .expect("range")
            .negotiate(ProtocolRange::new(2, 3).expect("range")),
        Ok(3)
    );
    assert_eq!(
        ProtocolRange::new(1, 2)
            .expect("range")
            .negotiate(ProtocolRange::new(3, 4).expect("range")),
        Err(ProtocolError::NoProtocolOverlap)
    );
}

#[test]
fn capability_serialization_and_transcript_order_are_deterministic() {
    let first = Capabilities::new(["trailers", "http2", "http1"]).expect("capabilities");
    let second = Capabilities::new(["http1", "trailers", "http2"]).expect("capabilities");
    assert_eq!(first, second);
    assert_eq!(first.canonical_bytes(), second.canonical_bytes());
    assert!(first.contains("trailers"));
    assert!(first.includes(&Capabilities::new(["http1", "http2"]).expect("required")));
}

#[test]
fn both_participants_sign_the_same_transcript() {
    let (transcript, initiator, responder) = sample_transcript();
    let initiator_proof = transcript
        .sign(ComponentRole::Mcp, &initiator)
        .expect("initiator proof");
    let responder_proof = transcript
        .sign(ComponentRole::Daemon, &responder)
        .expect("responder proof");
    transcript.verify(&initiator_proof).expect("initiator");
    transcript.verify(&responder_proof).expect("responder");
}

#[test]
fn any_signed_field_mutation_invalidates_the_proof() {
    let (mut transcript, initiator, _) = sample_transcript();
    let proof = transcript
        .sign(ComponentRole::Mcp, &initiator)
        .expect("proof");
    transcript.initiator.binary_version = "different-binary".to_owned();
    assert!(matches!(
        transcript.verify(&proof),
        Err(ProtocolError::Identity(
            IdentityError::SignatureVerification
        ))
    ));
}

#[test]
fn service_and_fingerprint_are_validated_before_signing() {
    let (mut wrong_service, initiator, _) = sample_transcript();
    wrong_service.initiator.service = "impostor".to_owned();
    assert_eq!(
        wrong_service.sign(ComponentRole::Mcp, &initiator),
        Err(ProtocolError::WrongService)
    );

    let (mut wrong_fingerprint, initiator, _) = sample_transcript();
    wrong_fingerprint.initiator_fingerprint = wrong_fingerprint.responder_fingerprint;
    assert_eq!(
        wrong_fingerprint.sign(ComponentRole::Mcp, &initiator),
        Err(ProtocolError::FingerprintMismatch)
    );
}

#[test]
fn descriptors_reject_oversized_untrusted_fields() {
    let oversized_capability = "a".repeat(129);
    assert!(Capabilities::new([oversized_capability]).is_err());
    let too_many = (0..65).map(|index| format!("capability-{index}"));
    assert!(Capabilities::new(too_many).is_err());

    let descriptor = ComponentDescriptor::nemo_relay(
        ComponentRole::Mcp,
        ProtocolRange::default(),
        Capabilities::streaming_transport(),
        "v".repeat(257),
    );
    assert_eq!(
        descriptor.validate(),
        Err(ProtocolError::MissingBinaryVersion)
    );
}

#[test]
fn activation_token_is_redacted_from_debug_but_serialized() {
    let directive = WorkerLaunch {
        activation_id: "activation-1".to_owned(),
        activation_token: SensitiveString::new("secret-value").expect("token"),
        deadline_unix_ms: 100,
        bind_ip: Ipv4Addr::LOCALHOST,
        port: 0,
        advertise_address: None,
    }
    .into_directive();
    assert!(!format!("{directive:?}").contains("secret-value"));
    assert!(
        serde_json::to_string(&directive)
            .expect("serialize")
            .contains("secret-value")
    );
    assert!(serde_json::from_str::<SensitiveString>("\"\"").is_err());
}
