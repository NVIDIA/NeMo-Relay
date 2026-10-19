// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Portable diagnostics wire-contract coverage.

use chrono::{DateTime, Utc};
use nemo_relay_types::diagnostics::*;
use serde_json::{Value, json};
use uuid::Uuid;

fn observed_at() -> DateTime<Utc> {
    "2026-10-15T09:00:00Z".parse().unwrap()
}

fn snapshot() -> DiagnosticSnapshot {
    let component = ComponentIdentity {
        key: "logging.file_sink".into(),
        instance_id: Uuid::parse_str("0199942f-1000-7000-8000-000000000002").unwrap(),
        generation: DiagnosticNumber::new(2),
    };
    let ownership = ComponentOwnership {
        configuration_id: Some("configuration-a".into()),
        responsibility: Some("operational_file_output".into()),
        ..Default::default()
    };
    let totals = DiagnosticTotals {
        recorded_events_total: DiagnosticCount {
            value: 7.into(),
            saturated: false,
        },
        occurrences_total: DiagnosticCount {
            value: u64::MAX.into(),
            saturated: true,
        },
        history_evictions_total: DiagnosticCount {
            value: 3.into(),
            saturated: false,
        },
        ..Default::default()
    };
    let lifetime = DiagnosticLifetimeSummary {
        totals,
        elapsed_millis: 50.into(),
        last_failure_at: Some(observed_at()),
        since_last_failure_millis: Some(10.into()),
    };
    let classification = Classification::failure(
        FailureCode::Logging(LoggingFailureCode::SinkWriteFailed),
        FailureClass::Unknown,
    )
    .unwrap();
    DiagnosticSnapshot {
        schema_version: DIAGNOSTIC_SCHEMA_VERSION,
        catalogue_version: DIAGNOSTIC_CATALOGUE_VERSION,
        source: SourceIdentity {
            instance_id: Uuid::parse_str("0199942f-1000-7000-8000-000000000001").unwrap(),
            process_id: Some(42),
            process_role: "gateway".into(),
            relay_version: "0.11.0".into(),
        },
        captured_at: observed_at(),
        sequence: DiagnosticNumber::new(u64::MAX),
        lifetime,
        components: vec![ComponentSnapshot {
            identity: component.clone(),
            registered_at: observed_at(),
            closed_at: Some(observed_at()),
            lifetime,
            since_refresh: DiagnosticRefreshSummary {
                generation: u64::MAX.into(),
                started_at: observed_at(),
                elapsed_millis: 30.into(),
                ownership: ownership.clone(),
                totals,
            },
            counters: vec![DiagnosticCounter {
                code: classification.code().canonical_name().into(),
                occurrences: DiagnosticCount {
                    value: u64::MAX.into(),
                    saturated: true,
                },
                last_observed_at: observed_at(),
            }],
            windows: vec![],
        }],
        events: vec![DiagnosticEvent {
            sequence: DiagnosticNumber::new(u64::MAX),
            observed_at: observed_at(),
            component,
            origin: DiagnosticOrigin {
                refresh_generation: u64::MAX.into(),
                ownership,
            },
            classification: classification.into(),
            context: PluginObservationContext::default(),
            severity: DiagnosticSeverity::Error,
            outcome: DiagnosticOutcome::Failure,
            detail: Some(DiagnosticDetail {
                message: "The file delivery attempt failed".into(),
                truncated: false,
            }),
            count: DiagnosticNumber::new(9_007_199_254_740_993),
            correlation: DiagnosticCorrelation {
                session_id: Some("session-1".into()),
                request_id: Some("request-1".into()),
            },
        }],
        coverage: DiagnosticCoverage {
            retention: DiagnosticRetention {
                oldest_event_sequence: Some(u64::MAX.into()),
                latest_event_sequence: Some(u64::MAX.into()),
                events_retained: 1.into(),
                components_retained: 1.into(),
                query_truncated: true,
            },
            ..Default::default()
        },
        continuation: Some(DiagnosticCursor {
            source_instance_id: Uuid::parse_str("0199942f-1000-7000-8000-000000000001").unwrap(),
            sequence: u64::MAX.into(),
            component_offset: 1.into(),
            event_offset: 1.into(),
            scope: 99.into(),
        }),
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
fn snapshots_preserve_identity_origin_correlation_and_lossless_counts() {
    let expected = snapshot();
    let mut wire = serde_json::to_value(&expected).unwrap();
    assert_eq!(wire["sequence"], json!(u64::MAX.to_string()));
    assert_eq!(
        wire["components"][0]["counters"][0]["occurrences"]["value"],
        json!(u64::MAX.to_string())
    );
    assert_eq!(wire["events"][0]["count"], json!("9007199254740993"));
    assert!(wire["components"][0].get("conditions").is_none());
    // A closed component retains observed evidence and cumulative counters.
    assert!(wire["components"][0]["closed_at"].is_string());
    wire["future_additive_field"] = json!({"producer": "newer"});
    let actual: DiagnosticSnapshot = serde_json::from_value(wire).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn outcomes_describe_observed_operations_without_condition_state() {
    for (outcome, wire_name) in [
        (DiagnosticOutcome::Unknown, "unknown"),
        (DiagnosticOutcome::Success, "success"),
        (DiagnosticOutcome::Failure, "failure"),
    ] {
        let wire = serde_json::to_value(outcome).unwrap();
        assert_eq!(wire, json!(wire_name));
        assert_eq!(
            serde_json::from_value::<DiagnosticOutcome>(wire).unwrap(),
            outcome
        );
    }
    for invalid in ["recovery", "healthy", "degraded"] {
        assert!(serde_json::from_value::<DiagnosticOutcome>(json!(invalid)).is_err());
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
        (DiagnosticAvailability::Disabled, "disabled"),
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
    assert!(schema["definitions"].get("CounterEpoch").is_none());
    assert!(schema["definitions"].get("CounterEpochId").is_none());
    assert!(
        schema["definitions"]["DiagnosticEvent"]["properties"]
            .get("counter_epoch")
            .is_none()
    );
    let origin_generation =
        &schema["definitions"]["DiagnosticOrigin"]["properties"]["refresh_generation"];
    let number_ref = json!("#/definitions/DiagnosticNumber");
    assert!(
        origin_generation["$ref"] == number_ref
            || origin_generation["allOf"]
                .as_array()
                .is_some_and(|items| items.iter().any(|item| item["$ref"] == number_ref))
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

#[test]
fn lifetime_refresh_origin_and_cursor_values_remain_lossless() {
    let expected = snapshot();
    let wire = serde_json::to_value(&expected).unwrap();
    assert_eq!(
        wire["catalogue_version"],
        json!(DIAGNOSTIC_CATALOGUE_VERSION)
    );
    assert_eq!(
        wire["components"][0]["lifetime"]["totals"]["recorded_events_total"]["value"],
        json!("7")
    );
    assert_eq!(
        wire["components"][0]["lifetime"]["totals"]["occurrences_total"]["value"],
        json!(u64::MAX.to_string())
    );
    assert_eq!(
        wire["components"][0]["lifetime"]["totals"]["history_evictions_total"]["value"],
        json!("3")
    );
    assert_eq!(
        wire["components"][0]["since_refresh"]["generation"],
        wire["events"][0]["origin"]["refresh_generation"]
    );
    assert_eq!(
        wire["continuation"]["sequence"],
        json!(u64::MAX.to_string())
    );
    assert_eq!(
        wire["components"][0]["since_refresh"]["ownership"]["configuration_id"],
        json!("configuration-a")
    );
    assert_eq!(
        wire["coverage"]["summary_scope"],
        json!("whole_component_lifetime")
    );
    assert!(
        wire["components"][0]["counters"][0]
            .get("last_message")
            .is_none()
    );
    assert!(wire["events"][0].get("code").is_none());
    assert_eq!(
        wire["events"][0]["classification"]["canonical_code"],
        json!("logging.sink.write_failed")
    );
    assert_eq!(
        serde_json::from_value::<DiagnosticSnapshot>(wire).unwrap(),
        expected
    );
}

#[test]
fn future_wire_classifications_remain_opaque_without_reclassifying_them() {
    let mut expected = snapshot();
    expected.catalogue_version = DIAGNOSTIC_CATALOGUE_VERSION + 1;
    expected.events[0].classification = ClassificationWire {
        canonical_code: "vendor.extension.new_failure".into(),
        failure_class: Some("new_remote_cause".into()),
    };
    let wire = serde_json::to_value(&expected).unwrap();
    let decoded: DiagnosticSnapshot = serde_json::from_value(wire).unwrap();
    assert_eq!(decoded, expected);
    assert!(BuiltinDiagnosticCode::from_canonical_name(decoded.events[0].code()).is_none());
}

#[test]
fn unsupported_snapshot_schema_is_rejected_independently_of_catalogue_version() {
    for version in [0, DIAGNOSTIC_SCHEMA_VERSION + 1, u32::MAX] {
        let mut wire = serde_json::to_value(snapshot()).unwrap();
        wire["schema_version"] = json!(version);
        assert!(serde_json::from_value::<DiagnosticSnapshot>(wire).is_err());
    }
    let mut wire = serde_json::to_value(snapshot()).unwrap();
    wire["catalogue_version"] = json!(u32::MAX);
    assert!(serde_json::from_value::<DiagnosticSnapshot>(wire).is_ok());
}

#[test]
fn count_saturation_distinguishes_exact_maximum_from_an_overflowing_increment() {
    let mut count = DiagnosticCount {
        value: (u64::MAX - 1).into(),
        saturated: false,
    };
    count.add(1);
    assert_eq!(count.value.get(), u64::MAX);
    assert!(!count.saturated);
    count.add(1);
    assert_eq!(count.value.get(), u64::MAX);
    assert!(count.saturated);
    count.add(0);
    assert_eq!(
        count,
        DiagnosticCount {
            value: u64::MAX.into(),
            saturated: true
        }
    );
}

#[test]
fn authoring_catalogue_is_fixed_size_and_has_unique_round_trippable_names() {
    fn requires_copy<T: Copy>() {}
    requires_copy::<BuiltinDiagnosticCode>();
    requires_copy::<Classification>();
    assert!(!std::mem::needs_drop::<Classification>());
    assert!(std::mem::size_of::<Classification>() <= 16);
    let mut names = std::collections::BTreeSet::new();
    for code in BuiltinDiagnosticCode::all() {
        assert!(
            names.insert(code.canonical_name()),
            "duplicate canonical code {}",
            code.canonical_name()
        );
        assert_eq!(
            BuiltinDiagnosticCode::from_canonical_name(code.canonical_name()),
            Some(*code)
        );
        assert!(!code.canonical_name().is_empty());
    }
    assert!(BuiltinDiagnosticCode::from_canonical_name("worker.error").is_none());
    assert!(
        BuiltinDiagnosticCode::from_canonical_name("logging.sink.write_failed.extra").is_none()
    );
}

#[test]
fn classification_constructors_enforce_entry_specific_class_contracts() {
    for code in BuiltinDiagnosticCode::all() {
        match code {
            BuiltinDiagnosticCode::Failure(_) => {
                assert!(!code.allowed_classes().is_empty());
                assert!(Classification::new(*code, None).is_err());
                for class in code.allowed_classes() {
                    let classification = Classification::new(*code, Some(*class)).unwrap();
                    assert_eq!(classification.code(), *code);
                    assert_eq!(classification.failure_class(), Some(*class));
                }
            }
            _ => {
                assert!(Classification::new(*code, None).is_ok());
                assert!(Classification::new(*code, Some(FailureClass::Unknown)).is_err());
            }
        }
    }
    let abi = BuiltinDiagnosticCode::Failure(FailureCode::Plugin(PluginFailureCode::Native(
        NativePluginFailureCode::AbiMismatch,
    )));
    assert!(Classification::new(abi, Some(FailureClass::Incompatible)).is_ok());
    assert!(Classification::new(abi, Some(FailureClass::Timeout)).is_err());
    assert!(Classification::new(abi, Some(FailureClass::Unknown)).is_err());
}

#[test]
fn nested_catalogue_selectors_derive_parent_ownership_without_another_record() {
    let worker = BuiltinDiagnosticCode::Failure(FailureCode::Plugin(PluginFailureCode::Worker(
        WorkerPluginFailureCode::ProtocolMismatch,
    )));
    assert_eq!(worker.domain(), DiagnosticDomain::Plugin);
    assert_eq!(worker.plugin_family(), Some(PluginDiagnosticFamily::Worker));
    assert_eq!(worker.canonical_name(), "plugin.worker.protocol_mismatch");
    let upstream =
        BuiltinDiagnosticCode::Failure(FailureCode::Upstream(UpstreamFailureCode::TimedOut));
    assert_eq!(upstream.domain(), DiagnosticDomain::Upstream);
    assert_eq!(upstream.plugin_family(), None);
    assert_eq!(upstream.canonical_name(), "upstream.timed_out");
}

#[test]
fn classifier_trait_supports_concrete_and_borrowed_dynamic_producers_without_a_runtime() {
    struct ProducerClassifier;
    impl DiagnosticClassifier<u32> for ProducerClassifier {
        fn classify(&self, _error: &u32) -> Classification {
            Classification::failure(
                FailureCode::Runtime(RuntimeFailureCode::Internal),
                FailureClass::Internal,
            )
            .unwrap()
        }
    }
    let error = 42;
    let concrete = ProducerClassifier.classify(&error);
    let selected: &dyn DiagnosticClassifier<u32> = &ProducerClassifier;
    assert_eq!(selected.classify(&error), concrete);
    assert_eq!(concrete.code().canonical_name(), "runtime.internal");
}

#[test]
fn typed_plugin_context_round_trips_without_requiring_unknown_mechanism_inference() {
    for (execution, phase) in [
        (
            Some(PluginExecution::NativeAbi),
            Some(PluginPhase::Validation),
        ),
        (
            Some(PluginExecution::WorkerProtocol),
            Some(PluginPhase::Runtime),
        ),
        (None, Some(PluginPhase::Activation)),
    ] {
        let mut expected = snapshot();
        expected.events[0].context = PluginObservationContext { execution, phase };
        let wire = serde_json::to_value(&expected).unwrap();
        assert_eq!(
            serde_json::from_value::<DiagnosticSnapshot>(wire).unwrap(),
            expected
        );
    }
    let expected = snapshot();
    let mut wire = serde_json::to_value(&expected).unwrap();
    wire["events"][0].as_object_mut().unwrap().remove("context");
    let decoded: DiagnosticSnapshot = serde_json::from_value(wire).unwrap();
    assert_eq!(
        decoded.events[0].context,
        PluginObservationContext::default()
    );
}

#[test]
fn lifetime_and_refresh_totals_are_fixed_copy_values_with_explicit_scopes() {
    fn requires_copy<T: Copy>() {}
    requires_copy::<DiagnosticTotals>();
    requires_copy::<DiagnosticLifetimeSummary>();
    assert!(!std::mem::needs_drop::<DiagnosticTotals>());
    let mut totals = DiagnosticTotals::default();
    totals.record(DiagnosticOutcome::Failure, 2);
    totals.record(DiagnosticOutcome::Failure, 3);
    totals.record(DiagnosticOutcome::Success, 4);
    totals.record(DiagnosticOutcome::Unknown, 1);
    assert_eq!(totals.recorded_events_total.value.get(), 4);
    assert_eq!(totals.occurrences_total.value.get(), 10);
    assert_eq!(totals.failures_total.value.get(), 5);
    assert_eq!(totals.successes_total.value.get(), 4);
    assert_eq!(totals.unspecified_total.value.get(), 1);

    let mut expected = snapshot();
    expected.lifetime.totals = totals;
    expected.components[0].since_refresh.totals = DiagnosticTotals::default();
    expected.components[0]
        .since_refresh
        .ownership
        .configuration_id = Some("configuration-b".into());
    // Original event attribution and lifetime accounting remain independent of
    // the replacement configuration's one current summary.
    let wire = serde_json::to_value(&expected).unwrap();
    assert_eq!(wire["lifetime"]["totals"]["failures_total"]["value"], "5");
    assert_eq!(
        wire["components"][0]["since_refresh"]["totals"]["failures_total"]["value"],
        "0"
    );
    assert_eq!(
        wire["events"][0]["origin"]["ownership"]["configuration_id"],
        "configuration-a"
    );
    assert_eq!(
        serde_json::from_value::<DiagnosticSnapshot>(wire).unwrap(),
        expected
    );
}

#[test]
fn origin_generation_query_is_lossless_without_a_historical_summary_selector() {
    let query = DiagnosticQuery {
        origin_generation: Some(u64::MAX.into()),
        ..Default::default()
    };
    let wire = serde_json::to_value(&query).unwrap();
    assert_eq!(wire["origin_generation"], u64::MAX.to_string());
    assert!(wire.get("counter_epoch").is_none());
    assert_eq!(
        serde_json::from_value::<DiagnosticQuery>(wire).unwrap(),
        query
    );
}
