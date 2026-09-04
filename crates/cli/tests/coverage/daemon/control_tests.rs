// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn daemon_challenge_signature_binds_the_request_before_token_disclosure() {
    let daemon = MachineIdentity::generate().expect("daemon").identity;
    let mcp = MachineIdentity::generate().expect("mcp").identity;
    let request = ChallengeRequest {
        initiator: descriptor(ComponentRole::Mcp),
        initiator_instance_id: "mcp-one".into(),
        initiator_public_identity: mcp.public_identity(),
        initiator_fingerprint: mcp.fingerprint(),
        initiator_nonce: fresh_nonce().expect("nonce"),
    };
    let challenge = super::super::identity::ChallengeRecord::generate(1, 10)
        .expect("challenge")
        .challenge();
    let mut response = ChallengeResponse {
        daemon: descriptor(ComponentRole::Daemon),
        daemon_instance_id: "daemon-one".into(),
        daemon_public_identity: daemon.public_identity(),
        daemon_fingerprint: daemon.fingerprint(),
        challenge,
        daemon_challenge_proof: daemon.sign(b"placeholder"),
    };
    response.daemon_challenge_proof =
        daemon.sign(&daemon_challenge_bytes(&request, &response).expect("canonical challenge"));
    response
        .verify_attestation(&request)
        .expect("signed challenge");

    let mut substituted = request;
    substituted.initiator_instance_id = "mcp-two".into();
    assert!(response.verify_attestation(&substituted).is_err());
}

#[test]
fn session_request_hash_covers_the_payload_and_sensitive_values_are_redacted() {
    let request = SessionRequest::new(
        "mcp-1".into(),
        SensitiveString::new("session-secret").expect("secret"),
        1,
        ActivationFailedPayload {
            activation_id: "activation-1".into(),
            reason: "bind failed".into(),
        },
    )
    .expect("request");
    assert!(request.validate_payload_hash());
    assert!(!format!("{request:?}").contains("session-secret"));

    let mut changed = request;
    changed.payload.reason = "different".into();
    assert!(!changed.validate_payload_hash());
}

#[test]
fn launch_directive_becomes_worker_bootstrap_without_reencoding_fields() {
    let directive = BrokerDirective::LaunchWorker {
        activation_id: "activation-1".into(),
        activation_token: SensitiveString::new("activation-secret").expect("secret"),
        deadline_unix_ms: 42,
        bind_ip: Ipv4Addr::LOCALHOST,
        port: 0,
        advertise_address: None,
    };
    let bootstrap = WorkerBootstrap::from_directive(directive).expect("launch directive");
    assert_eq!(bootstrap.activation_id, "activation-1");
    assert_eq!(bootstrap.activation_token.expose(), "activation-secret");
    assert_eq!(bootstrap.port, 0);
}

#[test]
fn worker_network_hint_is_signed_and_accepts_concrete_hostnames() {
    let identity = MachineIdentity::generate().expect("identity").identity;
    let challenge = super::super::identity::ChallengeRecord::generate(1, 10)
        .expect("challenge")
        .challenge();
    let hint = WorkerNetworkHint::new("Worker.Example.COM", Some(443)).expect("hint");
    assert_eq!(hint.advertised_host, "worker.example.com");
    let proof = WorkerNetworkHintProof::sign(
        hint,
        "https://daemon.example.com:443",
        "mcp-one",
        &challenge.id,
        &identity.fingerprint(),
        &identity,
    )
    .expect("signed hint");
    proof
        .verify(
            "https://daemon.example.com:443",
            "mcp-one",
            &challenge.id,
            &identity.fingerprint(),
            &identity.public_identity(),
        )
        .expect("valid hint");

    let mut changed = proof;
    changed.hint.advertised_host = "attacker.example.com".into();
    assert!(
        changed
            .verify(
                "https://daemon.example.com:443",
                "mcp-one",
                &challenge.id,
                &identity.fingerprint(),
                &identity.public_identity(),
            )
            .is_err()
    );
    assert!(WorkerNetworkHint::new("https://worker.example.com", None).is_err());
    assert!(WorkerNetworkHint::new("0.0.0.0", None).is_err());
}

#[test]
fn worker_generation_grant_binds_endpoint_and_tls_root() {
    let daemon = MachineIdentity::generate().expect("daemon").identity;
    let worker = MachineIdentity::generate().expect("worker").identity;
    let grant = WorkerGenerationGrant::issue(
        "worker-one",
        worker.fingerprint(),
        "https://worker.example.com:9443",
        Some("root-certificate"),
        &daemon,
    )
    .expect("generation grant");
    grant
        .verify(
            "worker-one",
            worker.fingerprint(),
            "https://worker.example.com:9443",
            Some("root-certificate"),
            &daemon.public_identity(),
        )
        .expect("valid generation");
    assert!(
        grant
            .verify(
                "worker-one",
                worker.fingerprint(),
                "https://attacker.example.com:9443",
                Some("root-certificate"),
                &daemon.public_identity(),
            )
            .is_err()
    );
}
