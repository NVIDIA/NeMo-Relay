// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::diagnostics::*;
use std::num::NonZeroU64;

fn service() -> EmbeddedService {
    let context = Diagnostics::from_settings(
        SourceIdentity {
            instance_id: uuid::Uuid::now_v7(),
            process_id: Some(std::process::id()),
            process_role: "test".into(),
            relay_version: "test".into(),
        },
        DiagnosticsSettings {
            enabled: true,
            ..Default::default()
        },
    )
    .unwrap();
    EmbeddedService::new(context).unwrap()
}

#[test]
fn copy_budget_releases_core_leases_and_recovers_after_failure() {
    let service = service();
    let first = service.reader.snapshot_json("{}").unwrap();
    let second = service.reader.clone().snapshot_json("{}").unwrap();
    assert_eq!(service.context.memory_usage().query_workspaces, 0);
    assert!(matches!(
        service.reader.snapshot_json("{}"),
        Err(EmbeddedDiagnosticError::Core(
            DiagnosticError::QueryCapacity
        ))
    ));
    drop(first);
    assert!(
        service
            .reader
            .snapshot_json("{\"max_response_bytes\":0}")
            .is_err()
    );
    // Failed core reads also return adapter capacity.
    let recovered = service.reader.snapshot_json("{}").unwrap();
    assert!(recovered.as_str().contains("schema_version"));
    drop((second, recovered));
    assert_eq!(
        service.reader.copies.lock().unwrap().len(),
        BINDING_COPY_CONCURRENCY
    );
    assert_eq!(
        service.reader.info().unwrap().binding_reserved_bytes,
        2 * service.context.limits().max_snapshot_bytes
    );
}

#[test]
fn embedded_query_is_bounded_and_rejects_unknown_fields() {
    assert_eq!(decode_query("{}").unwrap(), DiagnosticQuery::default());
    assert!(decode_query("{\"health\":true}").is_err());
    assert!(decode_query("[]").is_err());
    for query in [r#"{"ownership":true}"#, r#"{"continuation":true}"#] {
        assert_eq!(decode_query(query).unwrap_err().reason(), "invalid_query");
    }
    assert!(decode_query(&" ".repeat(MAX_BINDING_QUERY_BYTES + 1)).is_err());
    assert!(decode_query("{\"after_sequence\":9007199254740993}").is_err());
    assert_eq!(
        decode_query("{\"after_sequence\":\"18446744073709551615\"}")
            .unwrap()
            .after_sequence
            .unwrap()
            .get(),
        u64::MAX
    );
}

#[test]
fn nested_query_selectors_reject_unknown_fields_without_changing_response_models() {
    let ownership = ComponentOwnership {
        subscriber_instance_id: Some(uuid::Uuid::now_v7()),
        plugin_instance_id: Some(uuid::Uuid::now_v7()),
        configuration_id: Some("selected-config".into()),
        responsibility: Some("selected-runtime".into()),
    };
    let cursor = DiagnosticCursor {
        source_instance_id: uuid::Uuid::now_v7(),
        sequence: 1.into(),
        component_offset: 0.into(),
        event_offset: 0.into(),
        scope: 1.into(),
    };
    for (selector, value) in [
        ("ownership", serde_json::to_value(&ownership).unwrap()),
        ("continuation", serde_json::to_value(&cursor).unwrap()),
    ] {
        let query = serde_json::json!({ (selector): value });
        let decoded = decode_query(&query.to_string()).unwrap();
        if selector == "ownership" {
            assert_eq!(decoded.ownership, ownership);
        } else {
            assert_eq!(decoded.continuation, Some(cursor.clone()));
        }
        let mut unknown_only = serde_json::json!({ (selector): {"configuration_typo": "PRIVATE"} });
        assert_eq!(
            decode_query(&unknown_only.to_string())
                .unwrap_err()
                .reason(),
            "invalid_query"
        );
        unknown_only[selector] = query[selector].clone();
        unknown_only[selector]["future_field"] = serde_json::json!("PRIVATE");
        let error = decode_query(&unknown_only.to_string()).unwrap_err();
        assert_eq!(error.reason(), "invalid_query");
        assert!(!error.to_string().contains("PRIVATE"));
        // The same models remain extensible as snapshot response fields.
        if selector == "ownership" {
            assert_eq!(
                serde_json::from_value::<ComponentOwnership>(unknown_only[selector].clone())
                    .unwrap(),
                ownership
            );
        } else {
            assert_eq!(
                serde_json::from_value::<DiagnosticCursor>(unknown_only[selector].clone()).unwrap(),
                cursor
            );
        }
    }
    assert_eq!(
        decode_query(r#"{"ownership":{},"continuation":null}"#).unwrap(),
        DiagnosticQuery::default()
    );
}

#[test]
fn ownership_query_typos_cannot_broaden_embedded_snapshot_selection() {
    let service = service();
    let mut registration = ComponentRegistration::new("example", &[]);
    registration.ownership.configuration_id = Some("actual-config".into());
    let _component = service.context.register_component(registration).unwrap();
    for query in [
        r#"{"ownership":{"configuration_typo":"other-config"}}"#,
        r#"{"ownership":{"configuration_id":"actual-config","configuration_typo":"other-config"}}"#,
    ] {
        assert_eq!(
            service.reader.snapshot_json(query).err().unwrap().reason(),
            "invalid_query"
        );
    }
    for (query, expected) in [
        ("{}", 1),
        (r#"{"ownership":{}}"#, 1),
        (r#"{"ownership":{"configuration_id":"actual-config"}}"#, 1),
        (r#"{"ownership":{"configuration_id":"other-config"}}"#, 0),
    ] {
        let page = service.reader.snapshot_json(query).unwrap();
        let snapshot: DiagnosticSnapshot = serde_json::from_str(page.as_str()).unwrap();
        assert_eq!(snapshot.components.len(), expected, "{query}");
    }
}

#[test]
fn core_query_descriptor_matches_shared_contract_fixture() {
    let fixture: Json =
        serde_json::from_str(include_str!("../fixtures/diagnostics_query_contract.json")).unwrap();
    assert_eq!(fixture["max_query_bytes"], MAX_BINDING_QUERY_BYTES);
    assert_eq!(
        fixture["numeric_query_fields"],
        serde_json::json!(DIAGNOSTIC_QUERY_INTEGER_FIELDS)
    );
}

#[test]
fn disabled_copies_have_no_payload_and_fork_checks_precede_locks() {
    let mut service = EmbeddedService::new(Diagnostics::default()).unwrap();
    assert_eq!(service.reader.info().unwrap().binding_reserved_bytes, 0);
    assert!(matches!(
        service.reader.snapshot_json("{}"),
        Err(EmbeddedDiagnosticError::Core(DiagnosticError::Disabled))
    ));
    service.reader.pid = std::process::id().wrapping_add(1);
    let _locked = service.reader.copies.lock().unwrap();
    assert!(matches!(
        service.reader.snapshot_json("{}"),
        Err(EmbeddedDiagnosticError::ForkRequiresExec)
    ));
    assert!(matches!(
        service.reader.info(),
        Err(EmbeddedDiagnosticError::ForkRequiresExec)
    ));
}

#[test]
fn canonical_copy_keeps_large_counts_restricted_text_and_closed_evidence() {
    let service = service();
    let code = BuiltinDiagnosticCode::Success(SuccessCode::SinkInitialized);
    let (owner, producer) = service
        .context
        .register_owned_component(ComponentRegistration::new("example", &[code]))
        .unwrap();
    producer
        .record_without_projection_lazy(|| {
            Observation::new(
                Classification::success(SuccessCode::SinkInitialized),
                "18446744073709551615",
                NonZeroU64::new(u64::MAX).unwrap(),
            )
        })
        .unwrap();
    producer
        .record_without_projection_lazy(|| {
            Observation::new(
                Classification::success(SuccessCode::SinkInitialized),
                "restricted-detail",
                NonZeroU64::MIN,
            )
        })
        .unwrap();
    owner.close().unwrap();
    let page = service.reader.snapshot_json("{}").unwrap();
    let snapshot: DiagnosticSnapshot = serde_json::from_str(page.as_str()).unwrap();
    assert_eq!(
        snapshot.lifetime.totals.occurrences_total.value.get(),
        u64::MAX
    );
    assert!(snapshot.lifetime.totals.occurrences_total.saturated);
    assert!(snapshot.components[0].closed_at.is_some());
    assert_eq!(snapshot.events[0].message(), Some("18446744073709551615"));
}

#[tokio::test]
async fn embedded_host_selects_service_before_plugin_failure_and_reuses_it() {
    // nextest runs this process-global scenario in its own process.
    assert!(matches!(
        reader(),
        Err(EmbeddedDiagnosticError::NotInitialized)
    ));
    crate::logging::initialize_default_logging().unwrap();
    assert!(Diagnostics::selected_process_default().is_none());
    let config: crate::plugin::dynamic::PluginHostConfig =
        serde_json::from_value(serde_json::json!({
            "diagnostics": {"enabled": true},
            "components": [{"kind": "missing.phase4.component"}],
            "policy": {"unknown_component": "error"}
        }))
        .unwrap();
    assert!(
        crate::plugin::dynamic::initialize(config, None)
            .await
            .is_err()
    );
    let retained = reader().unwrap();
    assert!(retained.info().unwrap().discovery.collection_enabled);
    let source = retained.info().unwrap().discovery.source.instance_id;
    let config: crate::plugin::dynamic::PluginHostConfig =
        serde_json::from_value(serde_json::json!({"diagnostics":{"enabled":true}})).unwrap();
    let mut activation = crate::plugin::dynamic::initialize(config, None)
        .await
        .unwrap();
    activation.close().unwrap();
    assert_eq!(
        reader()
            .unwrap()
            .info()
            .unwrap()
            .discovery
            .source
            .instance_id,
        source
    );
    assert!(start(DiagnosticsSettings::default()).is_err());
    assert!(retained.snapshot_json("{}").is_ok());
    crate::logging::shutdown_default_logging().unwrap();
    assert!(retained.snapshot_json("{}").is_ok());
}

#[cfg(feature = "schema")]
#[test]
fn native_integer_field_inventory_covers_the_canonical_schema() {
    fn inspect(value: &Json, field: &str) {
        if value.get("$ref").and_then(Json::as_str) == Some("#/definitions/DiagnosticNumber") {
            assert!(
                is_diagnostic_number_field(field),
                "missing native integer field {field}"
            );
        }
        if let Some(properties) = value.get("properties").and_then(Json::as_object) {
            for (field, value) in properties {
                inspect(value, field);
            }
        }
        if let Some(fields) = value.as_object() {
            for (key, child) in fields {
                if key != "properties" {
                    inspect(child, field);
                }
            }
        } else if let Some(values) = value.as_array() {
            for child in values {
                inspect(child, field);
            }
        }
    }
    inspect(
        &serde_json::to_value(schemars::schema_for!(DiagnosticSnapshot)).unwrap(),
        "",
    );
    inspect(
        &serde_json::to_value(schemars::schema_for!(DiagnosticMemoryPage)).unwrap(),
        "",
    );
}

#[test]
fn shared_query_fixture_and_duplicate_selectors_have_safe_reasons() {
    let cases: Json = serde_json::from_str(include_str!(
        "../../../types/tests/fixtures/diagnostics/read-queries.json"
    ))
    .unwrap();
    for case in cases.as_array().unwrap() {
        let result = decode_query(&case["query"].to_string());
        assert_eq!(
            result.err().map(|error| error.reason()),
            case["reason"].as_str()
        );
    }
    for input in [
        r#"{"schema_version":1,"schema_version":2}"#,
        r#"{"ownership":{"responsibility":"a","responsibility":"b"}}"#,
    ] {
        assert_eq!(decode_query(input).unwrap_err().reason(), "invalid_query");
    }
}

#[test]
fn direct_and_embedded_reads_preserve_facts_and_return_matching_capacity_reasons() {
    let service = service();
    let failure =
        BuiltinDiagnosticCode::Failure(FailureCode::Logging(LoggingFailureCode::SinkWriteFailed));
    let success = BuiltinDiagnosticCode::Success(SuccessCode::SinkInitialized);
    for (name, delta) in [("example.a", 2), ("example.b", 3)] {
        let recorder = service
            .context
            .register_component(ComponentRegistration::new(name, &[failure, success]))
            .unwrap();
        recorder
            .record_lazy(|| {
                Observation::new(
                    Classification::failure(
                        FailureCode::Logging(LoggingFailureCode::SinkWriteFailed),
                        FailureClass::Io,
                    )
                    .unwrap(),
                    "restricted-fact",
                    NonZeroU64::new(delta).unwrap(),
                )
            })
            .unwrap();
        recorder
            .record_without_projection_lazy(|| {
                Observation::new(
                    Classification::success(SuccessCode::SinkInitialized),
                    "success",
                    NonZeroU64::MIN,
                )
            })
            .unwrap();
    }
    let direct = service
        .context
        .snapshot(&DiagnosticQuery::default())
        .unwrap()
        .to_snapshot()
        .unwrap();
    let copied: DiagnosticSnapshot =
        serde_json::from_str(service.reader.snapshot_json("{}").unwrap().as_str()).unwrap();
    assert_eq!(direct.source, copied.source);
    assert_eq!(direct.sequence, copied.sequence);
    assert_eq!(direct.events, copied.events);
    assert_eq!(direct.lifetime.totals, copied.lifetime.totals);
    assert_eq!(copied.lifetime.totals.failures_total.value.get(), 5);
    assert_eq!(copied.lifetime.totals.successes_total.value.get(), 2);
    assert_eq!(copied.lifetime.totals.recorded_events_total.value.get(), 4);
    for (query, expected) in [
        (
            DiagnosticQuery {
                schema_version: Some(2),
                ..Default::default()
            },
            "unsupported_schema",
        ),
        (
            DiagnosticQuery {
                max_events: Some(u32::MAX),
                ..Default::default()
            },
            "invalid_query",
        ),
        (
            DiagnosticQuery {
                max_response_bytes: Some(1),
                ..Default::default()
            },
            "snapshot_capacity",
        ),
    ] {
        assert_eq!(
            service
                .context
                .snapshot(&query)
                .err()
                .unwrap()
                .read_reason()
                .as_str(),
            expected
        );
        assert_eq!(
            service.reader.snapshot(&query).err().unwrap().reason(),
            expected
        );
    }
}

#[test]
fn discovery_is_independent_of_query_permits_and_matches_core() {
    let context = Diagnostics::from_settings(
        Diagnostics::default().source().clone(),
        DiagnosticsSettings {
            enabled: true,
            limits: DiagnosticLimits {
                max_queries: 0,
                ..Default::default()
            },
        },
    )
    .unwrap();
    let service = EmbeddedService::new(context).unwrap();
    let info = service.reader.info().unwrap();
    assert_eq!(info.discovery, service.context.reader().discovery());
    assert!(info.discovery.collection_enabled);
    assert!(!info.discovery.queries_enabled);
    assert_eq!(
        service.reader.snapshot_json("{}").err().unwrap().reason(),
        "query_disabled"
    );
    assert_eq!(info.binding_reserved_bytes, 0);
    assert!(
        info.discovery
            .producer_support
            .iter()
            .any(|p| p.name == "otel" && !p.supported)
    );
    assert!(info.discovery.producer_support.iter().any(
        |p| p.name == "plugin.worker.emission" && p.supported == cfg!(feature = "worker-grpc")
    ));
    assert!(info.discovery.detailed_memory_inspection && info.discovery.resource_correlation);
}

#[test]
fn memory_and_snapshot_copies_share_admission_and_release_core_before_conversion() {
    let service = service();
    let memory = service.reader.inspect_memory_json("{}").unwrap();
    let snapshot = service.reader.snapshot_json("{}").unwrap();
    assert_eq!(service.context.memory_usage().query_workspaces, 0);
    assert!(matches!(
        service.reader.inspect_memory_json("{}"),
        Err(EmbeddedDiagnosticError::Core(
            DiagnosticError::QueryCapacity
        ))
    ));
    drop(snapshot);
    drop(memory);
    assert!(service.reader.inspect_memory_json("{}").is_ok());
}
