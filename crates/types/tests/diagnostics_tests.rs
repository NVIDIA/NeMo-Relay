// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Portable diagnostics wire-contract coverage.

use chrono::{DateTime, Utc};
use nemo_relay_types::diagnostics::{
    ComponentIdentity, ComponentSnapshot, DIAGNOSTIC_SCHEMA_VERSION, DiagnosticAvailability,
    DiagnosticCondition, DiagnosticConditionState, DiagnosticCorrelation, DiagnosticCount,
    DiagnosticCounter, DiagnosticCoverage, DiagnosticEvent, DiagnosticNumber, DiagnosticOutcome,
    DiagnosticRetention, DiagnosticSeverity, DiagnosticSnapshot, SourceIdentity,
};
use serde_json::{Value, json};
use uuid::Uuid;

fn observed_at() -> DateTime<Utc> {
    "2026-10-15T09:00:00Z".parse().unwrap()
}

fn snapshot() -> DiagnosticSnapshot {
    let component = ComponentIdentity {
        key: "exporter.otlp.logs".into(),
        instance_id: Uuid::parse_str("0199942f-1000-7000-8000-000000000002").unwrap(),
        generation: DiagnosticNumber::new(2),
    };
    let counter = DiagnosticCounter {
        code: "otel.logs_export_failed".into(),
        epoch: component.instance_id,
        occurrences: DiagnosticCount {
            value: DiagnosticNumber::new(u64::MAX),
            saturated: true,
        },
        last_observed_at: observed_at(),
        last_message: "Collector is unavailable".into(),
    };
    let condition = DiagnosticCondition {
        code: counter.code.clone(),
        severity: DiagnosticSeverity::Error,
        message: counter.last_message.clone(),
        first_observed_at: observed_at(),
        last_observed_at: observed_at(),
        state: DiagnosticConditionState::Active,
    };
    DiagnosticSnapshot {
        schema_version: DIAGNOSTIC_SCHEMA_VERSION,
        source: SourceIdentity {
            instance_id: Uuid::parse_str("0199942f-1000-7000-8000-000000000001").unwrap(),
            process_id: Some(42),
            process_role: "gateway".into(),
            relay_version: "0.11.0".into(),
        },
        captured_at: observed_at(),
        sequence: DiagnosticNumber::new(u64::MAX),
        components: vec![ComponentSnapshot {
            identity: component.clone(),
            registered_at: observed_at(),
            closed_at: Some(observed_at()),
            conditions: vec![condition],
            counters: vec![counter.clone()],
        }],
        events: vec![DiagnosticEvent {
            sequence: DiagnosticNumber::new(u64::MAX),
            observed_at: observed_at(),
            component,
            code: counter.code,
            severity: DiagnosticSeverity::Error,
            outcome: DiagnosticOutcome::Failure,
            message: counter.last_message,
            count: DiagnosticNumber::new(9_007_199_254_740_993),
            correlation: DiagnosticCorrelation {
                session_id: Some("session-1".into()),
                request_id: Some("request-1".into()),
            },
        }],
        coverage: DiagnosticCoverage {
            retention: DiagnosticRetention {
                oldest_event_sequence: Some(DiagnosticNumber::new(u64::MAX)),
                latest_event_sequence: Some(DiagnosticNumber::new(u64::MAX)),
                events_retained: DiagnosticNumber::new(1),
                components_retained: DiagnosticNumber::new(1),
                query_truncated: true,
            },
            ..Default::default()
        },
    }
}

#[test]
fn all_u64_diagnostic_values_round_trip_as_decimal_strings() {
    for value in [0, 1, 9_007_199_254_740_993, u64::MAX] {
        let number = DiagnosticNumber::from(value);
        let wire = serde_json::to_value(number).unwrap();
        assert_eq!(wire, Value::String(value.to_string()));
        assert_eq!(
            serde_json::from_value::<DiagnosticNumber>(wire)
                .unwrap()
                .get(),
            value
        );
    }
}

#[test]
fn diagnostic_numbers_reject_lossy_or_noncanonical_encodings() {
    for wire in [
        json!(0),
        json!(9_007_199_254_740_993_u64),
        json!(1.5),
        json!(-1),
        json!(true),
        Value::Null,
        json!(""),
        json!("00"),
        json!("01"),
        json!("+1"),
        json!("-1"),
        json!("1.0"),
        json!("1e3"),
        json!(" 1"),
        json!("1 "),
        json!("١"),
        json!("18446744073709551616"),
        json!("100000000000000000000"),
    ] {
        assert!(
            serde_json::from_value::<DiagnosticNumber>(wire.clone()).is_err(),
            "unexpectedly accepted {wire}"
        );
    }
}

#[test]
fn snapshots_preserve_identity_epochs_correlation_and_lossless_counts() {
    let expected = snapshot();
    let mut wire = serde_json::to_value(&expected).unwrap();
    assert_eq!(wire["sequence"], json!(u64::MAX.to_string()));
    assert_eq!(
        wire["components"][0]["counters"][0]["occurrences"]["value"],
        json!(u64::MAX.to_string())
    );
    assert_eq!(wire["events"][0]["count"], json!("9007199254740993"));
    assert_eq!(
        wire["components"][0]["conditions"][0]["state"],
        json!({"status": "active"})
    );
    // A closed component can retain an unresolved condition.
    assert!(wire["components"][0]["closed_at"].is_string());
    wire["future_additive_field"] = json!({"producer": "newer"});
    let actual: DiagnosticSnapshot = serde_json::from_value(wire).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn recovery_requires_a_valid_observed_timestamp() {
    let recovered = DiagnosticConditionState::Recovered {
        recovered_at: observed_at(),
    };
    let wire = serde_json::to_value(&recovered).unwrap();
    assert_eq!(wire["status"], json!("recovered"));
    assert_eq!(
        serde_json::from_value::<DiagnosticConditionState>(wire).unwrap(),
        recovered
    );
    for invalid in [
        json!({"status": "recovered"}),
        json!({"status": "recovered", "recovered_at": "not-a-time"}),
        json!({"status": "healthy"}),
    ] {
        assert!(serde_json::from_value::<DiagnosticConditionState>(invalid).is_err());
    }
}

#[test]
fn malformed_source_identity_and_observation_time_are_rejected() {
    for (field, invalid) in [
        ("instance_id", json!("not-a-uuid")),
        ("process_id", json!(-1)),
    ] {
        let mut wire = serde_json::to_value(snapshot()).unwrap();
        wire["source"][field] = invalid;
        assert!(serde_json::from_value::<DiagnosticSnapshot>(wire).is_err());
    }
    let mut wire = serde_json::to_value(snapshot()).unwrap();
    wire["captured_at"] = json!("not-a-time");
    assert!(serde_json::from_value::<DiagnosticSnapshot>(wire).is_err());
}

#[test]
fn evidence_availability_is_explicit_and_not_inferred_from_an_empty_list() {
    for (availability, wire_name) in [
        (DiagnosticAvailability::Available, "available"),
        (DiagnosticAvailability::Unavailable, "unavailable"),
        (DiagnosticAvailability::Unsupported, "unsupported"),
        (DiagnosticAvailability::NeverObserved, "never_observed"),
    ] {
        let mut expected = snapshot();
        expected.events.clear();
        expected.components.clear();
        expected.coverage = DiagnosticCoverage {
            availability,
            ..Default::default()
        };
        let wire = serde_json::to_value(&expected).unwrap();
        assert_eq!(wire["coverage"]["availability"], json!(wire_name));
        assert_eq!(
            serde_json::from_value::<DiagnosticSnapshot>(wire).unwrap(),
            expected
        );
    }
}

#[cfg(feature = "schema")]
#[test]
fn generated_schema_keeps_diagnostic_numbers_as_strings() {
    let schema = serde_json::to_value(schemars::schema_for!(DiagnosticSnapshot)).unwrap();
    let number = &schema["definitions"]["DiagnosticNumber"];
    assert_eq!(number["type"], json!("string"));
    assert_eq!(number["minLength"], json!(1));
    assert_eq!(number["maxLength"], json!(20));
    assert!(number["pattern"].as_str().unwrap().starts_with('^'));
    assert!(number["pattern"].as_str().unwrap().ends_with('$'));
    // Schemars may wrap a documented reference in allOf to retain its description.
    let sequence = &schema["properties"]["sequence"];
    let number_ref = json!("#/definitions/DiagnosticNumber");
    assert!(
        sequence["$ref"] == number_ref
            || sequence["allOf"]
                .as_array()
                .is_some_and(|items| { items.iter().any(|item| item["$ref"] == number_ref) })
    );
}

#[cfg(feature = "schema")]
#[test]
fn numeric_schema_and_deserializer_agree_at_every_u64_maximum_prefix() {
    let schema = serde_json::to_value(schemars::schema_for!(DiagnosticNumber)).unwrap();
    let pattern = regex::Regex::new(schema["pattern"].as_str().unwrap()).unwrap();
    let maximum = u64::MAX.to_string();
    let mut values = vec![
        "0".to_owned(),
        "00".to_owned(),
        "01".to_owned(),
        "+1".to_owned(),
        "-1".to_owned(),
        "1.0".to_owned(),
        "1e3".to_owned(),
        "1\n".to_owned(),
        " 1".to_owned(),
        "18446744073709551616".to_owned(),
        "99999999999999999999".to_owned(),
        "100000000000000000000".to_owned(),
    ];
    for length in 1..20 {
        values.push("9".repeat(length));
    }
    for index in 0..maximum.len() {
        for digit in '0'..='9' {
            values.push(format!(
                "{}{}{}",
                &maximum[..index],
                digit,
                &maximum[index + 1..],
            ));
        }
    }
    for value in values {
        let deserialized = serde_json::from_value::<DiagnosticNumber>(Value::String(value.clone()));
        assert_eq!(
            pattern.is_match(&value),
            deserialized.is_ok(),
            "schema and deserializer disagree for {value:?}",
        );
    }
}

#[cfg(feature = "schema")]
#[test]
fn identity_and_observation_schemas_preserve_wire_formats() {
    let schema = serde_json::to_value(schemars::schema_for!(DiagnosticSnapshot)).unwrap();
    assert_eq!(
        schema["definitions"]["SourceIdentity"]["properties"]["instance_id"]["format"],
        json!("uuid"),
    );
    assert_eq!(
        schema["definitions"]["ComponentIdentity"]["properties"]["instance_id"]["format"],
        json!("uuid"),
    );
    assert_eq!(
        schema["definitions"]["DiagnosticCounter"]["properties"]["epoch"]["format"],
        json!("uuid"),
    );
    assert_eq!(
        schema["properties"]["captured_at"]["format"],
        json!("date-time"),
    );
    assert_eq!(
        schema["definitions"]["DiagnosticEvent"]["properties"]["observed_at"]["format"],
        json!("date-time"),
    );
}
