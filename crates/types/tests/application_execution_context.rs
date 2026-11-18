// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Standard assertions validate without sacrificing extension data or privacy.

use nemo_relay_types::Json;
use nemo_relay_types::api::tool::{ApplicationExecutionContext, HostApproval};
use serde_json::json;

fn assertion() -> Json {
    json!({"host_approval": {
        "version": 1, "source": "example.application",
        "approval_id": "approval-123", "tool_name": "files.write",
        "approved_arguments": {"path": "notes.txt", "content": "hello"}
    }})
}

#[test]
fn standard_claim_and_extensions_round_trip_without_authorization_interpretation() {
    for arguments in [
        Json::Null,
        json!([1, true]),
        json!("text"),
        json!({"path":"notes.txt"}),
    ] {
        let mut input = assertion();
        input["host_approval"]["approved_arguments"] = arguments;
        input["host_approval"]["example.receipt"] = json!({"opaque": [1, false]});
        input["example.application"] = json!({"request_id":"request-456", "integer":u64::MAX});
        let typed: ApplicationExecutionContext = serde_json::from_value(input.clone()).unwrap();
        assert_eq!(typed.to_json(), input);
        assert_eq!(typed.host_approval.as_ref().unwrap().version, 1);
        typed.validate().unwrap();
        let debug = format!("{typed:?} {:?}", typed.host_approval.as_ref().unwrap());
        assert!(!debug.contains("approval-123"));
        assert!(!debug.contains("request-456"));
    }
    assert_eq!(
        ApplicationExecutionContext::try_from(json!({}))
            .unwrap()
            .to_json(),
        json!({})
    );
}

#[test]
fn malformed_reserved_claims_fail_with_non_sensitive_errors() {
    let mut invalid = vec![
        Json::Null,
        json!([]),
        json!({"host_approval":null}),
        json!({"host_approval":{}}),
    ];
    for field in [
        "version",
        "source",
        "approval_id",
        "tool_name",
        "approved_arguments",
    ] {
        let mut value = assertion();
        value["host_approval"]
            .as_object_mut()
            .unwrap()
            .remove(field);
        invalid.push(value);
    }
    for (field, bad) in [
        ("version", json!(true)),
        ("version", json!(2)),
        ("version", json!("1")),
        ("source", json!("")),
        ("source", json!(42)),
        ("decision", json!("approved")),
        ("decision", json!("denied")),
        ("decision", Json::Null),
        ("decision", json!("pending")),
        ("approval_id", json!(null)),
        ("tool_name", json!("")),
    ] {
        let mut value = assertion();
        value["host_approval"][field] = bad;
        invalid.push(value);
    }
    for input in invalid {
        let error = ApplicationExecutionContext::try_from(input)
            .unwrap_err()
            .to_string();
        assert!(error.contains("JSON Schema"));
        assert!(!error.contains("approval-123"));
        assert!(!error.contains("notes.txt"));
    }
}

#[test]
fn numeric_version_uses_json_schema_integer_semantics() {
    for version in ["1", "1.0", "1e0"] {
        let mut value = assertion();
        value["host_approval"]["version"] = serde_json::from_str(version).unwrap();
        let approval: HostApproval =
            serde_json::from_value(value["host_approval"].clone()).unwrap();
        assert_eq!(approval.version, 1);
        assert_eq!(
            ApplicationExecutionContext::try_from(value)
                .unwrap()
                .host_approval
                .unwrap()
                .version,
            1
        );
    }
}

/// Reserved decisions cannot turn a legacy denied claim into an approval assertion.
#[test]
fn direct_host_approval_rejects_reserved_decision_field() {
    for decision in [
        json!("approved"),
        json!("denied"),
        json!("pending"),
        Json::Null,
        json!(false),
        json!(1),
    ] {
        let mut input = assertion()["host_approval"].clone();
        input["decision"] = decision;
        assert!(serde_json::from_value::<HostApproval>(input).is_err());
    }
}

/// A standalone claim has the same resource budget as when carried by an invocation.
#[test]
fn direct_host_approval_preserves_context_resource_limits() {
    let mut input = assertion()["host_approval"].clone();
    input["example.extra"] = json!("x".repeat(16_384));
    assert!(
        serde_json::from_value::<HostApproval>(input)
            .unwrap_err()
            .to_string()
            .contains("16384")
    );
    let mut input = assertion()["host_approval"].clone();
    let mut nested = Json::Null;
    for _ in 0..160 {
        nested = Json::Array(vec![nested]);
    }
    input["example.extra"] = nested;
    assert!(
        serde_json::from_value::<HostApproval>(input)
            .unwrap_err()
            .to_string()
            .contains("64 container levels")
    );
}

#[test]
fn direct_rust_changes_cannot_bypass_validation_or_shadow_known_properties() {
    let mut value = ApplicationExecutionContext::try_from(assertion()).unwrap();
    value.host_approval.as_mut().unwrap().approval_id.clear();
    assert!(value.validate().is_err());
    value.host_approval.as_mut().unwrap().approval_id = "approval-123".into();
    value
        .additional_properties
        .insert("host_approval".into(), Json::Null);
    assert!(value.validate().is_err());
    value.additional_properties.clear();
    value
        .host_approval
        .as_mut()
        .unwrap()
        .additional_properties
        .insert("decision".into(), json!("approved"));
    assert!(value.validate().is_err());
}

#[test]
fn extensions_share_depth_and_serialized_size_limits() {
    assert!(ApplicationExecutionContext::try_from(json!({"extra":"x".repeat(16_384)})).is_err());
    for levels in [64, 65, 161] {
        let mut nested = Json::Null;
        for _ in 1..levels {
            nested = json!([nested]);
        }
        let input = json!({"extra":nested});
        let valid = ApplicationExecutionContext::try_from(input).is_ok();
        assert_eq!(valid, levels == 64);
    }
    let mut typed = ApplicationExecutionContext::default();
    let mut nested = Json::Null;
    for _ in 0..160 {
        nested = json!([nested]);
    }
    typed.additional_properties.insert("extra".into(), nested);
    assert!(
        typed
            .validate()
            .unwrap_err()
            .to_string()
            .contains("64 container levels")
    );
}

#[test]
fn published_schema_conformance_vectors_match_typed_decoding() {
    let cases: Vec<Json> =
        serde_json::from_str(include_str!("fixtures/application_execution_context.json")).unwrap();
    for case in cases {
        let input = &case["value"];
        let expected = case["valid"].as_bool().unwrap();
        assert_eq!(
            ApplicationExecutionContext::validate_json(input).is_ok(),
            expected,
            "{}",
            case["name"]
        );
        assert_eq!(
            ApplicationExecutionContext::try_from(input.clone()).is_ok(),
            expected,
            "{}",
            case["name"]
        );
    }
}

/// Direct approval decoding must enforce the same contract as its enclosing context.
#[test]
fn direct_host_approval_decoding_matches_shared_conformance_vectors() {
    let cases: Vec<Json> =
        serde_json::from_str(include_str!("fixtures/application_execution_context.json")).unwrap();
    for case in cases {
        let Some(input) = case["value"].get("host_approval") else {
            continue;
        };
        let decoded = serde_json::from_value::<HostApproval>(input.clone());
        assert_eq!(
            decoded.is_ok(),
            case["valid"].as_bool().unwrap(),
            "{}",
            case["name"]
        );
        match decoded {
            Ok(approval) => {
                let mut expected = input.clone();
                expected["version"] = json!(1);
                assert_eq!(serde_json::to_value(approval).unwrap(), expected);
            }
            Err(error) => {
                assert!(error.to_string().contains("JSON Schema"));
                assert!(!error.to_string().contains("approval-123"));
            }
        }
    }
    for input in [Json::Null, json!([]), json!("private-receipt"), json!({})] {
        let error = serde_json::from_value::<HostApproval>(input).unwrap_err();
        assert!(!error.to_string().contains("private-receipt"));
    }
}
