// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Executable request and response annotation validation.

use nemo_relay_worker_proto::{nvidia_worker_v1 as pb, validation::validate};

#[test]
fn validates_required_fields_and_json_syntax() {
    assert!(validate(&pb::HandshakeRequest::default()).is_err());
    assert!(
        validate(&pb::JsonValue {
            json: b"null".to_vec()
        })
        .is_ok()
    );
    for json in [b"NaN".as_slice(), b"{", b""] {
        assert!(
            validate(&pb::JsonValue {
                json: json.to_vec()
            })
            .is_err()
        );
    }
    let request = pb::HandshakeRequest {
        credentials: Some(pb::Credentials {
            activation_id: "act".into(),
            auth_token: "token".into(),
        }),
        plugin_id: "test".into(),
        relay_version: "0.11.0".into(),
        host_endpoint: "unix:///tmp/host".into(),
        protocol: Some(pb::ProtocolIdentity {
            name: "nvidia.nemo.relay.worker.v1".into(),
            version: "1.0.0".into(),
        }),
    };
    validate(&request).unwrap();
    let mut invalid = request;
    invalid.protocol.as_mut().unwrap().version = "2.0.0".into();
    assert!(validate(&invalid).is_err());
}

#[test]
fn diagnostic_observations_require_positive_handles() {
    for (component, code) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
        let observation = pb::DiagnosticObservation {
            component,
            code,
            severity: pb::DiagnosticObservationSeverity::Info as i32,
            count: 1,
            ..Default::default()
        };
        assert_eq!(validate(&observation).is_ok(), component > 0 && code > 0);
    }
}

#[test]
fn enforces_result_oneofs_and_enum_constraints() {
    assert!(validate(&pb::InvokeResponse::default()).is_err());
    validate(&pb::InvokeResponse {
        result: Some(pb::invoke_response::Result::Completed(())),
    })
    .unwrap();
    assert!(
        validate(&pb::HealthResponse {
            status: 999,
            ..Default::default()
        })
        .is_err()
    );
}

#[test]
fn diagnostic_receipt_evidence_matches_recording_status() {
    for status in [
        pb::DiagnosticReceiptStatus::Recorded,
        pb::DiagnosticReceiptStatus::Disabled,
        pb::DiagnosticReceiptStatus::Rejected,
        pb::DiagnosticReceiptStatus::Registered,
        pb::DiagnosticReceiptStatus::Closed,
    ] {
        for evidence in [
            pb::DiagnosticEvidenceRetention::Unspecified,
            pb::DiagnosticEvidenceRetention::Retained,
            pb::DiagnosticEvidenceRetention::Truncated,
            pb::DiagnosticEvidenceRetention::Dropped,
        ] {
            let receipt = pb::DiagnosticReceipt {
                submission_sequence: 1,
                status: status as i32,
                evidence: evidence as i32,
                component: 1,
                code_count: 1,
                ..Default::default()
            };
            let expected = (status == pb::DiagnosticReceiptStatus::Recorded)
                == (evidence != pb::DiagnosticEvidenceRetention::Unspecified);
            let result = validate(&receipt);
            assert_eq!(
                result.is_ok(),
                expected,
                "{status:?}, {evidence:?}: {result:?}"
            );
        }
        assert!(
            validate(&pb::DiagnosticReceipt {
                status: status as i32,
                evidence: 999,
                ..Default::default()
            })
            .is_err()
        );
    }
}

#[test]
fn rejects_noncanonical_well_known_times() {
    validate(&prost_types::Timestamp {
        seconds: -1,
        nanos: 999_999_999,
    })
    .unwrap();
    for time in [
        prost_types::Timestamp {
            seconds: 253402300800,
            nanos: 0,
        },
        prost_types::Timestamp {
            seconds: 0,
            nanos: -1,
        },
    ] {
        assert!(validate(&time).is_err());
    }
    validate(&prost_types::Duration {
        seconds: -1,
        nanos: -1,
    })
    .unwrap();
    assert!(
        validate(&prost_types::Duration {
            seconds: 1,
            nanos: -1
        })
        .is_err()
    );
}

#[test]
fn registered_receipts_require_positive_component_and_code_count() {
    for (component, code_count) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
        let receipt = pb::DiagnosticReceipt {
            submission_sequence: 1,
            status: pb::DiagnosticReceiptStatus::Registered as i32,
            component,
            code_count,
            ..Default::default()
        };
        assert_eq!(validate(&receipt).is_ok(), component > 0 && code_count > 0);
    }
}

#[test]
fn isolated_scope_stack_operations_require_nonempty_handles() {
    assert!(validate(&pb::CreateScopeStackResponse::default()).is_err());
    validate(&pb::CreateScopeStackResponse {
        scope_stack_id: "isolated-stack".into(),
    })
    .unwrap();
    for scope_stack_id in ["", "isolated-stack"] {
        let request = pb::DropScopeStackRequest {
            credentials: Some(pb::Credentials {
                activation_id: "activation".into(),
                auth_token: "token".into(),
            }),
            scope_stack_id: scope_stack_id.into(),
        };
        assert_eq!(validate(&request).is_ok(), !scope_stack_id.is_empty());
    }
}
