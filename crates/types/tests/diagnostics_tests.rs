// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Portable diagnostics wire-contract coverage.

use std::borrow::Cow;

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
        failures_total: DiagnosticCount {
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
                code: classification.code().into(),
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
            remote: None,
            observed_at: observed_at(),
            component,
            origin: DiagnosticOrigin {
                refresh_generation: u64::MAX.into(),
                ownership,
            },
            classification,
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
            loss: DiagnosticLoss {
                events_evicted: DiagnosticCount {
                    value: 3.into(),
                    saturated: false,
                },
                ..Default::default()
            },
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
fn remote_provenance_is_optional_and_preserves_lossless_sequences() {
    let mut view = snapshot();
    let local = serde_json::to_value(&view).unwrap();
    assert!(local["events"][0].get("remote").is_none());
    assert_eq!(
        serde_json::from_value::<DiagnosticSnapshot>(local).unwrap(),
        view
    );
    view.events[0].remote = Some(RemoteDiagnosticOrigin {
        source_instance_id: Uuid::now_v7(),
        activation_instance_id: Uuid::now_v7(),
        submission_sequence: u64::MAX.into(),
        observed_at: None,
        received_at: observed_at(),
        accepted_elapsed_millis: u64::MAX.into(),
    });
    let wire = serde_json::to_value(&view).unwrap();
    assert_eq!(
        wire["events"][0]["remote"]["submission_sequence"],
        u64::MAX.to_string()
    );
    assert_eq!(
        serde_json::from_value::<DiagnosticSnapshot>(wire).unwrap(),
        view
    );
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
fn future_classifications_remain_opaque_without_reclassifying_them() {
    let mut expected = snapshot();
    expected.catalogue_version = DIAGNOSTIC_CATALOGUE_VERSION + 1;
    expected.events[0].classification = Classification::from_opaque_names(
        "vendor.extension.new_failure".into(),
        Some("new_remote_cause".into()),
    );
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
    assert!(!std::mem::needs_drop::<BuiltinDiagnosticCode>());
    assert!(std::mem::size_of::<BuiltinDiagnosticCode>() <= 16);
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
    assert!(BuiltinDiagnosticCode::from_canonical_name("vendor.unregistered_failure").is_none());
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
                    assert_eq!(classification.code(), code.canonical_name());
                    assert_eq!(classification.failure_class(), Some(class.as_str()));
                }
            }
            _ => {
                assert!(Classification::new(*code, None).is_ok());
                assert!(Classification::new(*code, Some(FailureClass::Unknown)).is_err());
            }
        }
    }
    let dropped =
        BuiltinDiagnosticCode::Failure(FailureCode::Logging(LoggingFailureCode::SinkRecordDropped));
    assert!(Classification::new(dropped, Some(FailureClass::ResourceExhausted)).is_ok());
    assert!(Classification::new(dropped, Some(FailureClass::Timeout)).is_err());
    assert!(Classification::new(dropped, Some(FailureClass::Unknown)).is_err());
}

#[test]
fn nested_catalogue_selectors_derive_parent_ownership_without_another_record() {
    let plugin = BuiltinDiagnosticCode::Failure(FailureCode::Plugin(PluginFailureCode::Common(
        CommonPluginFailureCode::RegistrationFailed,
    )));
    assert_eq!(plugin.domain(), DiagnosticDomain::Plugin);
    assert_eq!(plugin.plugin_family(), Some(PluginDiagnosticFamily::Common));
    assert_eq!(plugin.canonical_name(), "plugin.registration_failed");
    let logging =
        BuiltinDiagnosticCode::Failure(FailureCode::Logging(LoggingFailureCode::SinkWriteFailed));
    assert_eq!(logging.domain(), DiagnosticDomain::Logging);
    assert_eq!(logging.plugin_family(), None);
    assert_eq!(logging.canonical_name(), "logging.sink.write_failed");
    for (success, name) in [
        (SuccessCode::LoggingInitialized, "logging.initialized"),
        (
            SuccessCode::LoggingShutdownCompleted,
            "logging.shutdown_completed",
        ),
        (SuccessCode::SinkInitialized, "logging.sink.initialized"),
    ] {
        let code = BuiltinDiagnosticCode::Success(success);
        assert_eq!(code.canonical_name(), name);
        assert_eq!(code.domain(), DiagnosticDomain::Logging);
        assert_eq!(code.plugin_family(), None);
    }
    let inherited = BuiltinDiagnosticCode::Notice(NoticeCode::ConfigurationInherited);
    assert_eq!(inherited.canonical_name(), "plugin.configuration_inherited");
    assert_eq!(inherited.domain(), DiagnosticDomain::Plugin);
    assert_eq!(
        inherited.plugin_family(),
        Some(PluginDiagnosticFamily::Common)
    );
}

#[test]
fn classifier_trait_supports_concrete_and_borrowed_dynamic_producers_without_a_runtime() {
    struct ProducerClassifier;
    impl DiagnosticClassifier<u32> for ProducerClassifier {
        fn classify(&self, _error: &u32) -> Classification {
            Classification::failure(
                FailureCode::Plugin(PluginFailureCode::Common(CommonPluginFailureCode::Internal)),
                FailureClass::Internal,
            )
            .unwrap()
        }
    }
    let error = 42;
    let concrete = ProducerClassifier.classify(&error);
    let selected: &dyn DiagnosticClassifier<u32> = &ProducerClassifier;
    assert_eq!(selected.classify(&error), concrete);
    assert_eq!(concrete.code(), "plugin.internal");
}

#[test]
fn typed_plugin_context_round_trips_without_requiring_unknown_mechanism_inference() {
    for (execution, phase, execution_name, phase_name) in [
        (
            Some(PluginExecution::NativeAbi),
            Some(PluginPhase::Validation),
            Some("native_abi"),
            "validation",
        ),
        (
            Some(PluginExecution::WorkerProtocol),
            Some(PluginPhase::Runtime),
            Some("worker_protocol"),
            "runtime",
        ),
        (None, Some(PluginPhase::Activation), None, "activation"),
        (
            Some(PluginExecution::NativeAbi),
            Some(PluginPhase::Policy),
            Some("native_abi"),
            "policy",
        ),
    ] {
        let mut expected = snapshot();
        expected.events[0].context = PluginObservationContext {
            execution: execution.map(Into::into),
            phase: phase.map(Into::into),
        };
        let wire = serde_json::to_value(&expected).unwrap();
        assert_eq!(
            wire["events"][0]["context"],
            json!({ "execution": execution_name, "phase": phase_name }),
        );
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
fn future_plugin_context_names_survive_complete_snapshot_round_trips() {
    for (execution_name, phase_name) in [
        ("wasm_v2", "activation"),
        ("worker_protocol", "quiescing"),
        ("wasm_v2", "quiescing"),
        ("unrecognized", "unrecognized"),
    ] {
        let mut wire = serde_json::to_value(snapshot()).unwrap();
        wire["events"][0]["context"] = json!({
            "execution": execution_name,
            "phase": phase_name,
        });
        let decoded: DiagnosticSnapshot = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(
            decoded.events[0]
                .context
                .execution
                .as_ref()
                .map(PluginExecutionName::as_str),
            Some(execution_name)
        );
        assert_eq!(
            decoded.events[0]
                .context
                .phase
                .as_ref()
                .map(PluginPhaseName::as_str),
            Some(phase_name)
        );
        assert!(matches!(
            decoded.events[0]
                .context
                .execution
                .clone()
                .map(PluginExecutionName::into_name),
            Some(Cow::Owned(_))
        ));
        assert!(matches!(
            decoded.events[0]
                .context
                .phase
                .clone()
                .map(PluginPhaseName::into_name),
            Some(Cow::Owned(_))
        ));
        assert_eq!(serde_json::to_value(decoded).unwrap(), wire);
    }
}

#[test]
fn missing_and_null_plugin_context_fields_remain_absent() {
    for (context, expected) in [
        (json!({}), PluginObservationContext::default()),
        (
            json!({ "execution": null, "phase": null }),
            PluginObservationContext::default(),
        ),
        (
            json!({ "execution": "native_abi" }),
            PluginObservationContext {
                execution: Some(PluginExecution::NativeAbi.into()),
                phase: None,
            },
        ),
        (
            json!({ "phase": "activation" }),
            PluginObservationContext {
                execution: None,
                phase: Some(PluginPhase::Activation.into()),
            },
        ),
    ] {
        let mut wire = serde_json::to_value(snapshot()).unwrap();
        wire["events"][0]["context"] = context;
        let decoded: DiagnosticSnapshot = serde_json::from_value(wire).unwrap();
        assert_eq!(decoded.events[0].context, expected);
    }
}

#[test]
fn shared_payload_names_reject_non_string_values() {
    for (payload, field) in [
        ("classification", "canonical_code"),
        ("classification", "failure_class"),
        ("context", "execution"),
        ("context", "phase"),
        ("correlation", "session_id"),
        ("correlation", "request_id"),
    ] {
        for value in [
            json!(true),
            json!(1),
            json!(1.5),
            json!([]),
            json!(["native_abi"]),
            json!({}),
            json!({ "native_abi": null }),
            json!({ "activation": null }),
        ] {
            let mut wire = serde_json::to_value(snapshot()).unwrap();
            wire["events"][0][payload][field] = value.clone();
            assert!(
                serde_json::from_value::<DiagnosticSnapshot>(wire).is_err(),
                "non-string {payload}.{field} unexpectedly decoded: {value}",
            );
        }
    }
}

#[test]
fn shared_payloads_borrow_producer_values_and_own_decoded_values() {
    let classification = Classification::failure(
        FailureCode::Logging(LoggingFailureCode::SinkWriteFailed),
        FailureClass::Io,
    )
    .unwrap();
    assert!(matches!(
        &classification.clone().into_names().0,
        Cow::Borrowed(_)
    ));
    assert!(matches!(
        &classification.clone().into_names().1,
        Some(Cow::Borrowed(_))
    ));
    let decoded: Classification =
        serde_json::from_value(serde_json::to_value(&classification).unwrap()).unwrap();
    assert_eq!(decoded, classification);
    assert!(matches!(&decoded.clone().into_names().0, Cow::Owned(_)));
    assert!(matches!(
        &decoded.clone().into_names().1,
        Some(Cow::Owned(_))
    ));

    let session = String::from("session-dynamic");
    let request = String::from("request-dynamic");
    let correlation = DiagnosticCorrelation {
        session_id: Some(Cow::Borrowed(session.as_str())),
        request_id: Some(Cow::Borrowed(request.as_str())),
    };
    let decoded: DiagnosticCorrelation<'static> =
        serde_json::from_value(serde_json::to_value(&correlation).unwrap()).unwrap();
    assert_eq!(decoded, correlation);
    assert!(matches!(&decoded.session_id, Some(Cow::Owned(_))));
    assert!(matches!(&decoded.request_id, Some(Cow::Owned(_))));
}

#[test]
fn typed_context_helpers_use_canonical_borrowed_names() {
    for execution in [PluginExecution::NativeAbi, PluginExecution::WorkerProtocol] {
        let name: Cow<'static, str> = execution.into();
        assert!(matches!(&name, Cow::Borrowed(_)));
        assert_eq!(name.as_ref(), execution.as_str());
        assert_eq!(<&'static str>::from(execution), execution.as_str());
        assert_eq!(serde_json::to_value(execution).unwrap(), json!(name));
        assert_eq!(
            PluginExecution::from_canonical_name(execution.as_str()),
            Some(execution)
        );
    }
    for phase in [
        PluginPhase::Validation,
        PluginPhase::Activation,
        PluginPhase::Runtime,
        PluginPhase::Policy,
    ] {
        let name: Cow<'static, str> = phase.into();
        assert!(matches!(&name, Cow::Borrowed(_)));
        assert_eq!(name.as_ref(), phase.as_str());
        assert_eq!(<&'static str>::from(phase), phase.as_str());
        assert_eq!(serde_json::to_value(phase).unwrap(), json!(name));
        assert_eq!(
            PluginPhase::from_canonical_name(phase.as_str()),
            Some(phase)
        );
    }
}

#[test]
fn failure_class_names_match_serialization_and_lookup() {
    for class in [
        FailureClass::InvalidInput,
        FailureClass::InvalidState,
        FailureClass::NotFound,
        FailureClass::Conflict,
        FailureClass::Authentication,
        FailureClass::PermissionDenied,
        FailureClass::Unsupported,
        FailureClass::Incompatible,
        FailureClass::Timeout,
        FailureClass::Unavailable,
        FailureClass::ResourceExhausted,
        FailureClass::Io,
        FailureClass::ExecutionFailed,
        FailureClass::PolicyRejected,
        FailureClass::Internal,
        FailureClass::Unknown,
    ] {
        assert_eq!(serde_json::to_value(class).unwrap(), json!(class.as_str()));
        assert_eq!(<&'static str>::from(class), class.as_str());
        assert_eq!(
            FailureClass::from_canonical_name(class.as_str()),
            Some(class)
        );
    }
}

#[test]
fn canonical_name_helpers_reject_unrecognized_names() {
    assert!(PluginExecution::from_canonical_name("wasm_v2").is_none());
    assert!(PluginPhase::from_canonical_name("quiescing").is_none());
    assert!(FailureClass::from_canonical_name("new_remote_cause").is_none());
    let long_name = "unrecognized".repeat(4096);
    for name in [
        "",
        " native_abi",
        "ACTIVATION",
        "invalid_input ",
        long_name.as_str(),
    ] {
        assert!(PluginExecution::from_canonical_name(name).is_none());
        assert!(PluginPhase::from_canonical_name(name).is_none());
        assert!(FailureClass::from_canonical_name(name).is_none());
    }
}

#[cfg(feature = "schema")]
#[test]
fn shared_payload_schema_admits_names_as_strings() {
    let schema = serde_json::to_value(schemars::schema_for!(DiagnosticSnapshot)).unwrap();
    let event = &schema["definitions"]["DiagnosticEvent"]["properties"];
    for (field, name) in [
        ("classification", "Classification"),
        ("context", "PluginObservationContext"),
        ("correlation", "DiagnosticCorrelation"),
    ] {
        assert!(
            serde_json::to_string(&event[field])
                .unwrap()
                .contains(&format!("#/definitions/{name}"))
        );
    }
    let classification = &schema["definitions"]["Classification"]["properties"];
    assert_eq!(classification["canonical_code"]["type"], json!("string"));
    for (name, fields) in [
        ("Classification", &["failure_class"][..]),
        ("PluginObservationContext", &["execution", "phase"][..]),
        ("DiagnosticCorrelation", &["session_id", "request_id"][..]),
    ] {
        for field in fields {
            let property = &schema["definitions"][name]["properties"][field];
            assert_eq!(property["type"], json!(["string", "null"]));
            assert!(property.get("enum").is_none());
        }
    }
}

#[test]
fn lifetime_and_refresh_totals_are_fixed_copy_values_with_explicit_scopes() {
    fn requires_copy<T: Copy>() {}
    requires_copy::<DiagnosticTotals>();
    requires_copy::<DiagnosticLifetimeSummary>();
    requires_copy::<PluginExecution>();
    requires_copy::<PluginPhase>();
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

#[cfg(feature = "schema")]
#[test]
fn snapshot_schema_version_matches_decoder_and_numeric_wire_notation() {
    let schema = serde_json::to_value(schemars::schema_for!(DiagnosticSnapshot)).unwrap();
    let version = &schema["properties"]["schema_version"];
    assert_eq!(version["type"], json!("integer"));
    assert_eq!(version["enum"], json!([DIAGNOSTIC_SCHEMA_VERSION]));
    for candidate in [
        json!(0),
        json!(1),
        json!(2),
        json!(u32::MAX),
        json!("1"),
        json!(null),
    ] {
        let mut wire = serde_json::to_value(snapshot()).unwrap();
        wire["schema_version"] = candidate.clone();
        let schema_accepts = version["enum"].as_array().unwrap().contains(&candidate);
        assert_eq!(
            serde_json::from_value::<DiagnosticSnapshot>(wire).is_ok(),
            schema_accepts,
            "schema and decoder disagree for {candidate}",
        );
    }
}

#[test]
fn canonical_enum_conversions_keep_const_names_and_reject_noncanonical_input() {
    const CAUSE: &str = FailureClass::ResourceExhausted.as_str();
    const EXECUTION: &str = PluginExecution::WorkerProtocol.as_str();
    const PHASE: &str = PluginPhase::Activation.as_str();
    assert_eq!(
        (CAUSE, EXECUTION, PHASE),
        ("resource_exhausted", "worker_protocol", "activation")
    );
    for invalid in ["", "Activation", " activation", "future_value"] {
        assert_eq!(FailureClass::from_canonical_name(invalid), None);
        assert_eq!(PluginExecution::from_canonical_name(invalid), None);
        assert_eq!(PluginPhase::from_canonical_name(invalid), None);
    }
}

#[test]
fn distinct_context_names_keep_known_values_borrowed_and_future_names_opaque() {
    let execution = PluginExecutionName::from(PluginExecution::NativeAbi);
    let phase = PluginPhaseName::from(PluginPhase::Activation);
    assert_eq!(execution.known(), Some(PluginExecution::NativeAbi));
    assert_eq!(phase.known(), Some(PluginPhase::Activation));
    assert!(matches!(execution.into_name(), Cow::Borrowed("native_abi")));
    assert!(matches!(phase.into_name(), Cow::Borrowed("activation")));
    let execution: PluginExecutionName = serde_json::from_value(json!("future_vm")).unwrap();
    let phase: PluginPhaseName = serde_json::from_value(json!("future_stage")).unwrap();
    assert_eq!(execution.known(), None);
    assert_eq!(phase.known(), None);
    assert_eq!(serde_json::to_value(execution).unwrap(), json!("future_vm"));
    assert_eq!(serde_json::to_value(phase).unwrap(), json!("future_stage"));
}

// Reviewed wire fixture: changing a model cannot silently regenerate expectations.
#[test]
fn versioned_fixture_matches_the_public_model_and_evolution_rules() {
    let wire: Value = serde_json::from_str(include_str!(
        "fixtures/diagnostics/diagnostic-snapshot-schema-1.json"
    ))
    .unwrap();
    assert_eq!(serde_json::to_value(snapshot()).unwrap(), wire);
    let actual: DiagnosticSnapshot = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(actual, snapshot());
    let mut future = wire.clone();
    future["catalogue_version"] = json!(99);
    future["future_field"] = json!(true);
    future["events"][0]["classification"] =
        json!({"canonical_code":"future.failure", "failure_class":"future.cause"});
    future["events"][0]["context"] = json!({"execution":"future_engine", "phase":"future_phase"});
    let decoded: DiagnosticSnapshot = serde_json::from_value(future).unwrap();
    assert_eq!(decoded.events[0].classification.code(), "future.failure");
    assert_eq!(
        decoded.events[0].context.phase.as_ref().unwrap().as_str(),
        "future_phase"
    );
    for version in [0, 2, u32::MAX] {
        let mut unsupported = wire.clone();
        unsupported["schema_version"] = json!(version);
        assert!(serde_json::from_value::<DiagnosticSnapshot>(unsupported).is_err());
    }
}

#[test]
fn read_reason_wire_names_match_language_error_spelling() {
    use DiagnosticReadReason::*;
    for reason in [
        Disabled,
        QueryDisabled,
        UnsupportedSchema,
        InvalidQuery,
        InvalidCursor,
        Busy,
        SnapshotCapacity,
        InvalidLimits,
        AllocationFailed,
        Internal,
    ] {
        assert_eq!(serde_json::to_value(reason).unwrap(), reason.as_str());
        assert_eq!(
            serde_json::from_value::<DiagnosticReadReason>(json!(reason.as_str())).unwrap(),
            reason
        );
    }
}

#[test]
fn memory_page_wire_quantities_are_lossless_and_versioned() {
    let page = DiagnosticMemoryPage {
        schema_version: DIAGNOSTIC_MEMORY_VERSION,
        source: snapshot().source,
        captured_at: observed_at(),
        sequence: u64::MAX.into(),
        aggregate: DiagnosticMemoryUsage {
            reserved_bytes: usize::MAX,
            ..Default::default()
        },
        component_occupied_bytes: 0.into(),
        shared_or_unassigned_reserved_bytes: (usize::MAX as u64).into(),
        loss: DiagnosticLoss::default(),
        selection: MemorySelectionTotals::default(),
        page: MemorySelectionTotals::default(),
        components: vec![],
        entries: vec![],
        resource: ResourceMemoryView::disabled(),
        continuation: None,
    };
    let mut wire = serde_json::to_value(&page).unwrap();
    assert_eq!(wire["aggregate"]["reserved_bytes"], usize::MAX.to_string());
    assert_eq!(wire["sequence"], u64::MAX.to_string());
    assert_eq!(
        serde_json::from_value::<DiagnosticMemoryPage>(wire.clone()).unwrap(),
        page
    );
    wire["schema_version"] = json!(2);
    assert!(serde_json::from_value::<DiagnosticMemoryPage>(wire).is_err());
}

#[cfg(feature = "schema")]
#[test]
fn memory_schema_uses_versioned_envelope_and_lossless_aggregate_fields() {
    let schema = serde_json::to_value(schemars::schema_for!(DiagnosticMemoryPage)).unwrap();
    assert_eq!(schema["properties"]["schema_version"]["enum"], json!([1]));
    let fields = schema["definitions"]["DiagnosticMemoryUsage"]["properties"]
        .as_object()
        .unwrap();
    assert!(!fields.is_empty());
    for value in fields.values() {
        let reference = json!("#/definitions/DiagnosticNumber");
        assert!(
            value["$ref"] == reference
                || value["allOf"]
                    .as_array()
                    .is_some_and(|items| { items.iter().any(|item| item["$ref"] == reference) })
        );
    }
}

#[test]
fn canonical_memory_fixture_preserves_complete_wire_shape_and_precision() {
    let value: Value = serde_json::from_str(include_str!(
        "fixtures/diagnostics/diagnostic-memory-schema-1.json"
    ))
    .unwrap();
    let page: DiagnosticMemoryPage = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(page.sequence.get(), u64::MAX);
    assert_eq!(page.aggregate.reserved_bytes as u64, 9_007_199_254_740_993);
    assert_eq!(page.components[0].component.key, "9007199254740993");
    let sample = page.resource.sample.unwrap();
    assert_eq!(sample.ordinal.get(), u64::MAX);
    assert_eq!(
        sample.memory.system_used.unwrap().value,
        ResourceMemoryNumber::Integer(u64::MAX.into())
    );
    assert_eq!(
        sample.memory.private.unwrap().value,
        ResourceMemoryNumber::Decimal(1.25)
    );
    assert_eq!(
        sample.memory.virtual_memory.unwrap().value,
        ResourceMemoryNumber::NonFinite
    );
    assert!(sample.memory.system_available.is_none());
    assert_eq!(serde_json::to_value(page).unwrap(), value);
}

#[test]
fn exporter_initialization_failure_has_plugin_common_ownership_and_explicit_causes() {
    let code = BuiltinDiagnosticCode::Failure(FailureCode::Plugin(PluginFailureCode::Common(
        CommonPluginFailureCode::ExporterInitializationFailed,
    )));
    assert_eq!(
        code.canonical_name(),
        "plugin.exporter_initialization_failed"
    );
    assert_eq!(
        BuiltinDiagnosticCode::from_canonical_name(code.canonical_name()),
        Some(code)
    );
    assert_eq!(code.domain(), DiagnosticDomain::Plugin);
    assert_eq!(code.plugin_family(), Some(PluginDiagnosticFamily::Common));
    for class in [
        FailureClass::InvalidInput,
        FailureClass::PolicyRejected,
        FailureClass::Io,
        FailureClass::Unknown,
    ] {
        assert!(code.allows_class(Some(class)));
    }
    assert!(!code.allows_class(None));
    assert!(!code.allows_class(Some(FailureClass::Internal)));
}
