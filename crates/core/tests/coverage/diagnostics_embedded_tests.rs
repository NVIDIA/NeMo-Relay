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
}
