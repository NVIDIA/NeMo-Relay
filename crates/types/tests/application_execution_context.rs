// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Relay owns the context envelope; applications own the JSON data and its policy.

use nemo_relay_types::Json;
use nemo_relay_types::api::tool::ApplicationExecutionContext;
use serde_json::json;

#[test]
fn concrete_context_and_application_data_round_trip_without_interpretation() {
    for input in [
        json!({}),
        json!({"data": {}}),
        json!({"data": {"integer": u64::MAX, "unicode": "π", "nul": "\u{0000}"}}),
        json!({"data": {"host_approval": {"decision": "denied"}, "private-reference": "private-receipt"}}),
        json!({"data": {"host_approval": null, "future_relay_field": [false, null, 3]}}),
    ] {
        let typed: ApplicationExecutionContext = serde_json::from_value(input.clone()).unwrap();
        assert_eq!(typed.to_json(), input);
        typed.validate().unwrap();
        let debug = format!("{typed:?}");
        assert!(!debug.contains("private-receipt"));
        assert!(!debug.contains("host_approval"));
    }
    let omitted_data = ApplicationExecutionContext::default();
    let empty_data = ApplicationExecutionContext::try_from(json!({"data": {}})).unwrap();
    assert_ne!(omitted_data, empty_data);
    assert!(omitted_data.data.is_none());
    assert!(empty_data.data.unwrap().is_empty());
}

#[test]
fn unknown_top_level_keys_and_non_object_data_are_rejected_without_values() {
    for input in [
        Json::Null,
        json!([]),
        json!("private-receipt"),
        json!({"host_approval": {"approval_id": "private-receipt"}}),
        json!({"custom": null}),
        json!({"data": {}, "custom": {}}),
        json!({"data": null}),
        json!({"data": []}),
        json!({"data": false}),
        json!({"data": "private-receipt"}),
        json!({"data": 1}),
    ] {
        let error = ApplicationExecutionContext::try_from(input)
            .unwrap_err()
            .to_string();
        assert!(error.contains("JSON Schema"));
        assert!(!error.contains("private-receipt"));
    }
}

#[test]
fn application_data_shares_envelope_depth_and_byte_limits() {
    assert!(
        ApplicationExecutionContext::try_from(json!({"data": {"extra": "x".repeat(16_384)}}))
            .is_err()
    );
    for levels in [64, 65, 161] {
        let mut nested = Json::Null;
        for _ in 2..levels {
            nested = json!([nested]);
        }
        let input = json!({"data": {"extra": nested}});
        assert_eq!(
            ApplicationExecutionContext::try_from(input).is_ok(),
            levels == 64
        );
    }
}

#[test]
fn typed_mutation_is_bounded_before_serialization() {
    let mut context = ApplicationExecutionContext::default();
    let mut nested = Json::Null;
    for _ in 0..160 {
        nested = Json::Array(vec![nested]);
    }
    context.data = Some(std::collections::BTreeMap::from([("extra".into(), nested)]));
    assert!(
        context
            .validate()
            .unwrap_err()
            .to_string()
            .contains("64 container levels")
    );
    context
        .data
        .as_mut()
        .unwrap()
        .insert("extra".into(), json!("x".repeat(16_384)));
    assert!(
        context
            .validate()
            .unwrap_err()
            .to_string()
            .contains("16384")
    );
}

#[test]
fn context_resource_budget_counts_exact_serialized_bytes() {
    use nemo_relay_types::api::tool::application_context::ApplicationExecutionContextError;

    let overhead_bytes = serde_json::to_vec(&json!({"data": {"value": ""}}))
        .unwrap()
        .len();
    let available_bytes = 16_384 - overhead_bytes;
    let unicode_boundary = "雪".repeat(available_bytes / 3) + &"x".repeat(available_bytes % 3);
    for (text, expected) in [
        ("x".repeat(available_bytes), Ok(())),
        (unicode_boundary, Ok(())),
        (
            "x".repeat(available_bytes + 1),
            Err(ApplicationExecutionContextError::TooLarge),
        ),
        (
            "\0".repeat(3_000),
            Err(ApplicationExecutionContextError::TooLarge),
        ),
    ] {
        let mut context = ApplicationExecutionContext::default();
        context.data = Some(std::collections::BTreeMap::from([(
            "value".to_owned(),
            Json::String(text),
        )]));
        let raw = context.to_json();
        if expected.is_ok() {
            assert_eq!(serde_json::to_vec(&raw).unwrap().len(), 16_384);
        }
        assert_eq!(ApplicationExecutionContext::validate_json(&raw), expected);
        assert_eq!(context.validate(), expected);
    }
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
