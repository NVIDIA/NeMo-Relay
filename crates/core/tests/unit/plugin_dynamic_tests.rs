// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the dynamic plugin control-plane model.

use super::*;
use crate::plugin::PluginError;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_dir(prefix: &str) -> PathBuf {
    let id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("nemo-relay-{prefix}-{id}"));
    fs::create_dir_all(&path).unwrap();
    path
}

fn sample_record() -> DynamicPluginRecord {
    DynamicPluginRecord {
        metadata: DynamicPluginMetadata {
            id: "acme.guardrails.pii".into(),
            name: Some("PII Guardrails".into()),
            version: Some("0.1.0".into()),
            kind: DynamicPluginKind::Worker,
            generation: 3,
            created_at: Some("2026-06-16T00:00:00Z".into()),
            updated_at: Some("2026-06-16T00:00:00Z".into()),
        },
        capabilities: vec![
            DynamicPluginCapability::PluginWorker,
            DynamicPluginCapability::MiddlewareGuardrail,
        ],
        source: DynamicPluginSource {
            manifest_ref: Some("/plugins/pii/relay-plugin.toml".into()),
            artifact_ref: Some("/plugins/pii/dist/pii.whl".into()),
            environment_ref: Some("/plugins/pii/.venv".into()),
            artifact_digest: Some("sha256:abc123".into()),
        },
        spec: DynamicPluginSpec {
            present: true,
            enabled: true,
            config_ref: Some("plugins.acme.guardrails.pii".into()),
        },
        compatibility: DynamicPluginCompatibility {
            relay: Some(">=0.1.0,<0.2.0".into()),
            native_api: None,
            worker_protocol: Some("1".into()),
        },
        load: DynamicPluginLoadContract::Worker(DynamicPluginWorkerLoadContract {
            runtime: WorkerRuntime::Python,
            entrypoint: "acme_guardrails.plugin:register".into(),
        }),
        status: DynamicPluginStatus {
            validation: DynamicPluginValidationStatus {
                manifest: DynamicPluginCheckState::Valid,
                compatibility: DynamicPluginCheckState::Valid,
                integrity: DynamicPluginCheckState::Unknown,
                environment: DynamicPluginCheckState::Valid,
                authenticity: DynamicPluginCheckState::Unknown,
                policy_satisfied: DynamicPluginCheckState::Valid,
                checked_at: Some("2026-06-16T00:00:01Z".into()),
                message: Some("ready".into()),
            },
            runtime: DynamicPluginRuntimeStatus {
                state: DynamicPluginRuntimeState::Running,
                observed_generation: 3,
                started_at: Some("2026-06-16T00:00:02Z".into()),
                updated_at: Some("2026-06-16T00:00:02Z".into()),
                message: Some("running".into()),
            },
            startup_class: Some(DynamicPluginStartupClass::Optional),
            attestation_mode: Some(DynamicPluginAttestationMode::IntegrityOnly),
            last_error: None,
        },
    }
}

#[test]
fn dynamic_plugin_spec_defaults_to_present_but_disabled() {
    let spec = DynamicPluginSpec::default();
    assert!(spec.present);
    assert!(!spec.enabled);
    assert!(spec.config_ref.is_none());
}

#[test]
fn dynamic_plugin_status_defaults_to_unknown_validation_and_stopped_runtime() {
    let status = DynamicPluginStatus::default();
    assert_eq!(status.validation.manifest, DynamicPluginCheckState::Unknown);
    assert_eq!(
        status.validation.policy_satisfied,
        DynamicPluginCheckState::Unknown
    );
    assert_eq!(status.runtime.state, DynamicPluginRuntimeState::Stopped);
    assert_eq!(status.runtime.observed_generation, 0);
    assert!(status.last_error.is_none());
}

#[test]
fn dynamic_plugin_record_reports_reconciliation_from_generation() {
    let record = sample_record();
    assert!(record.is_reconciled());

    let mut stale = record.clone();
    stale.status.runtime.observed_generation = 2;
    assert!(!stale.is_reconciled());
}

#[test]
fn dynamic_plugin_record_tombstone_tracks_presence_in_spec() {
    let record = sample_record();
    assert!(!record.is_tombstoned());

    let mut removed = record.clone();
    removed.spec.present = false;
    assert!(removed.is_tombstoned());
}

#[test]
fn dynamic_plugin_record_round_trips_through_json() {
    let record = sample_record();
    let json = serde_json::to_value(&record).expect("serialize dynamic plugin record");
    let decoded: DynamicPluginRecord =
        serde_json::from_value(json).expect("deserialize dynamic plugin record");
    assert_eq!(decoded, record);
}

#[test]
fn registry_adds_record_and_lists_live_entries_only_by_default() {
    let mut registry = DynamicPluginRegistry::new();
    registry.add(sample_record()).expect("register plugin");

    let live = registry.list(false);
    assert_eq!(live.len(), 1);
    assert_eq!(live[0].metadata.id, "acme.guardrails.pii");
    assert_eq!(live[0].metadata.created_at, live[0].metadata.updated_at);

    let all = registry.list(true);
    assert_eq!(all.len(), 1);
}

#[test]
fn registry_rejects_duplicate_live_plugin_ids() {
    let mut registry = DynamicPluginRegistry::new();
    registry.add(sample_record()).expect("register plugin");

    let err = registry
        .add(sample_record())
        .expect_err("duplicate id should fail");
    match err {
        PluginError::Conflict(message) => {
            assert!(message.contains("already registered"), "{message}");
        }
        other => panic!("unexpected duplicate error: {other}"),
    }
}

#[test]
fn registry_rejects_invalid_raw_record_shapes() {
    let mut registry = DynamicPluginRegistry::new();
    let mut record = sample_record();
    record.capabilities = vec![DynamicPluginCapability::PluginNative];

    let err = registry
        .add(record)
        .expect_err("invalid raw record shape should fail");
    match err {
        PluginError::InvalidConfig(message) => {
            assert!(message.contains("capability shape"), "{message}");
        }
        other => panic!("unexpected invalid raw record error: {other}"),
    }
}

#[test]
fn registry_rejects_invalid_raw_record_load_shapes() {
    let mut registry = DynamicPluginRegistry::new();
    let mut record = sample_record();
    record.load = DynamicPluginLoadContract::Worker(DynamicPluginWorkerLoadContract {
        runtime: WorkerRuntime::Python,
        entrypoint: String::new(),
    });

    let err = registry
        .add(record)
        .expect_err("invalid raw worker load shape should fail");
    match err {
        PluginError::InvalidConfig(message) => {
            assert!(message.contains("load shape"), "{message}");
        }
        other => panic!("unexpected invalid raw load error: {other}"),
    }
}

#[test]
fn registry_rejects_invalid_raw_record_compatibility_shapes() {
    let mut registry = DynamicPluginRegistry::new();
    let mut record = sample_record();
    record.compatibility.worker_protocol = None;

    let err = registry
        .add(record)
        .expect_err("invalid raw worker compatibility shape should fail");
    match err {
        PluginError::InvalidConfig(message) => {
            assert!(message.contains("compatibility shape"), "{message}");
        }
        other => panic!("unexpected invalid raw compatibility error: {other}"),
    }
}

#[test]
fn registry_rejects_empty_required_lane_specific_compatibility_strings() {
    let mut registry = DynamicPluginRegistry::new();
    let mut worker_record = sample_record();
    worker_record.compatibility.worker_protocol = Some("   ".into());

    let err = registry
        .add(worker_record)
        .expect_err("empty worker_protocol should fail");
    match err {
        PluginError::InvalidConfig(message) => {
            assert!(message.contains("compatibility shape"), "{message}");
        }
        other => panic!("unexpected empty worker compatibility error: {other}"),
    }

    let mut rust_record = sample_record();
    rust_record.metadata.kind = DynamicPluginKind::RustDynamic;
    rust_record.capabilities = vec![
        DynamicPluginCapability::PluginNative,
        DynamicPluginCapability::MiddlewareInterceptor,
    ];
    rust_record.compatibility.native_api = Some("   ".into());
    rust_record.compatibility.worker_protocol = None;
    rust_record.load = DynamicPluginLoadContract::RustDynamic(DynamicPluginRustLoadContract {
        library: "target/release/libswitchyard.dylib".into(),
        symbol: "nemo_relay_register_plugin".into(),
    });

    let err = registry
        .add(rust_record)
        .expect_err("empty native_api should fail");
    match err {
        PluginError::InvalidConfig(message) => {
            assert!(message.contains("compatibility shape"), "{message}");
        }
        other => panic!("unexpected empty native compatibility error: {other}"),
    }
}

#[test]
fn registry_rejects_missing_raw_record_relay_compatibility() {
    let mut registry = DynamicPluginRegistry::new();
    let mut record = sample_record();
    record.compatibility.relay = None;

    let err = registry
        .add(record)
        .expect_err("missing compat.relay should fail");
    match err {
        PluginError::InvalidConfig(message) => {
            assert!(message.contains("compat.relay"), "{message}");
        }
        other => panic!("unexpected missing-relay compatibility error: {other}"),
    }
}

#[test]
fn registry_add_canonicalizes_required_record_strings_before_storage() {
    let mut registry = DynamicPluginRegistry::new();
    let mut record = sample_record();
    record.metadata.id = " acme.guardrails.pii ".into();
    record.compatibility.relay = Some(" >=0.1.0,<0.2.0 ".into());
    record.compatibility.worker_protocol = Some(" 1 ".into());
    record.load = DynamicPluginLoadContract::Worker(DynamicPluginWorkerLoadContract {
        runtime: WorkerRuntime::Python,
        entrypoint: " acme_guardrails.plugin:register ".into(),
    });

    let stored = registry.add(record).expect("register canonicalized plugin");
    assert_eq!(stored.metadata.id, "acme.guardrails.pii");
    assert_eq!(
        stored.compatibility.relay.as_deref(),
        Some(">=0.1.0,<0.2.0")
    );
    assert_eq!(stored.compatibility.worker_protocol.as_deref(), Some("1"));
    assert_eq!(
        stored.load,
        DynamicPluginLoadContract::Worker(DynamicPluginWorkerLoadContract {
            runtime: WorkerRuntime::Python,
            entrypoint: "acme_guardrails.plugin:register".into(),
        })
    );
}

#[test]
fn registry_enable_disable_and_remove_are_generation_bumping_spec_mutations() {
    let mut registry = DynamicPluginRegistry::new();
    let mut record = sample_record();
    record.metadata.generation = 0;
    record.spec.enabled = false;
    record.status.runtime.observed_generation = 0;
    registry.add(record).expect("register plugin");

    assert!(
        registry
            .enable("acme.guardrails.pii")
            .expect("enable plugin")
    );
    let enabled = registry.get("acme.guardrails.pii").expect("enabled record");
    assert!(enabled.spec.enabled);
    assert_eq!(enabled.metadata.generation, 1);

    assert!(
        !registry
            .enable("acme.guardrails.pii")
            .expect("idempotent enable")
    );
    let still_enabled = registry.get("acme.guardrails.pii").expect("enabled record");
    assert_eq!(still_enabled.metadata.generation, 1);

    assert!(
        registry
            .disable("acme.guardrails.pii")
            .expect("disable plugin")
    );
    let disabled = registry
        .get("acme.guardrails.pii")
        .expect("disabled record");
    assert!(!disabled.spec.enabled);
    assert_eq!(disabled.metadata.generation, 2);

    assert!(
        registry
            .remove("acme.guardrails.pii")
            .expect("remove plugin")
    );
    let removed = registry.get("acme.guardrails.pii").expect("removed record");
    assert!(removed.is_tombstoned());
    assert!(!removed.spec.enabled);
    assert_eq!(removed.metadata.generation, 3);

    assert_eq!(registry.list(false).len(), 0);
    assert_eq!(registry.list(true).len(), 1);
}

#[test]
fn registry_can_revive_tombstoned_ids_and_preserve_logical_lineage() {
    let mut registry = DynamicPluginRegistry::new();
    let mut original = sample_record();
    original.metadata.generation = 4;
    original.spec.enabled = false;
    original.metadata.created_at = Some("2026-06-01T00:00:00Z".into());
    registry.add(original).expect("register plugin");
    registry
        .remove("acme.guardrails.pii")
        .expect("tombstone plugin");

    let mut revived = sample_record();
    revived.metadata.generation = 0;
    revived.metadata.created_at = None;
    revived.spec.enabled = false;
    registry.add(revived).expect("revive plugin");

    let record = registry.get("acme.guardrails.pii").expect("revived record");
    assert!(!record.is_tombstoned());
    assert_eq!(record.metadata.generation, 6);
    assert_eq!(
        record.metadata.created_at.as_deref(),
        Some("2026-06-01T00:00:00Z")
    );
}

#[test]
fn registry_status_updates_do_not_change_desired_state_generation() {
    let mut registry = DynamicPluginRegistry::new();
    let mut record = sample_record();
    record.metadata.generation = 7;
    registry.add(record).expect("register plugin");

    registry
        .update_validation_status(
            "acme.guardrails.pii",
            DynamicPluginValidationStatus {
                manifest: DynamicPluginCheckState::Valid,
                compatibility: DynamicPluginCheckState::Invalid,
                integrity: DynamicPluginCheckState::Unknown,
                environment: DynamicPluginCheckState::Valid,
                authenticity: DynamicPluginCheckState::Unknown,
                policy_satisfied: DynamicPluginCheckState::Invalid,
                checked_at: Some("2026-06-16T01:00:00Z".into()),
                message: Some("compatibility failed".into()),
            },
        )
        .expect("update validation status");
    let checked_at = registry
        .get("acme.guardrails.pii")
        .and_then(|record| record.status.validation.checked_at.clone())
        .expect("registry should stamp validation checked_at");

    registry
        .update_runtime_status(
            "acme.guardrails.pii",
            DynamicPluginRuntimeStatus {
                state: DynamicPluginRuntimeState::Failed,
                observed_generation: 6,
                started_at: None,
                updated_at: None,
                message: Some("worker crashed".into()),
            },
        )
        .expect("update runtime status");
    let runtime_updated_at = registry
        .get("acme.guardrails.pii")
        .and_then(|record| record.status.runtime.updated_at.clone())
        .expect("registry should stamp runtime updated_at");
    registry
        .update_last_error(
            "acme.guardrails.pii",
            Some(DynamicPluginFailure {
                phase: DynamicPluginFailurePhase::Runtime,
                code: "worker.crash".into(),
                message: "worker crashed".into(),
            }),
        )
        .expect("update failure");

    let record = registry.get("acme.guardrails.pii").expect("updated record");
    assert_eq!(record.metadata.generation, 7);
    assert_eq!(
        record.status.validation.compatibility,
        DynamicPluginCheckState::Invalid
    );
    assert!(!checked_at.trim().is_empty());
    assert_eq!(
        record.status.runtime.state,
        DynamicPluginRuntimeState::Failed
    );
    assert!(!runtime_updated_at.trim().is_empty());
    assert_eq!(
        record
            .status
            .last_error
            .as_ref()
            .map(|error| error.code.as_str()),
        Some("worker.crash")
    );
}

fn valid_worker_manifest_toml() -> &'static str {
    r#"
manifest_version = 1
description = "PII guardrail worker"

[plugin]
id = "acme.guardrails.pii"
name = "PII Guardrails"
version = "0.1.0"
kind = "worker"

[compat]
relay = ">=0.1.0,<0.2.0"
worker_protocol = "1"

[defaults]
enabled = false

[capabilities]
items = ["plugin_worker", "middleware_guardrail", "config_schema"]

[load]
runtime = "python"
entrypoint = "acme_guardrails.plugin:register"

[source]
manifest_root = "."
artifact = "dist/acme_guardrails.whl"

[integrity]
sha256 = "sha256:abc123"
"#
}

fn valid_rust_manifest_toml() -> &'static str {
    r#"
manifest_version = 1

[plugin]
id = "acme.native.switchyard"
kind = "rust_dynamic"

[compat]
relay = ">=0.1.0,<0.2.0"
native_api = "1"

[defaults]
enabled = false

[capabilities]
items = ["plugin_native", "middleware_interceptor"]

[load]
library = "target/release/libswitchyard.dylib"
symbol = "nemo_relay_register_plugin"
"#
}

#[test]
fn manifest_parse_and_conversion_supports_worker_lane() {
    let manifest =
        DynamicPluginManifest::parse_toml(valid_worker_manifest_toml()).expect("parse manifest");
    assert_eq!(manifest.plugin.kind, DynamicPluginKind::Worker);
    assert_eq!(
        manifest.load,
        DynamicPluginManifestLoad::Worker(DynamicPluginManifestWorkerLoad {
            runtime: Some(WorkerRuntime::Python),
            entrypoint: Some("acme_guardrails.plugin:register".into()),
        })
    );

    let record = manifest
        .into_record(Some("/plugins/pii/relay-plugin.toml".into()))
        .expect("manifest converts into record");
    assert_eq!(record.metadata.id, "acme.guardrails.pii");
    assert_eq!(record.metadata.version.as_deref(), Some("0.1.0"));
    assert!(!record.spec.enabled);
    assert_eq!(
        record.load,
        DynamicPluginLoadContract::Worker(DynamicPluginWorkerLoadContract {
            runtime: WorkerRuntime::Python,
            entrypoint: "acme_guardrails.plugin:register".into()
        })
    );
    assert_eq!(
        record.source.manifest_ref.as_deref(),
        Some("/plugins/pii/relay-plugin.toml")
    );
    assert_eq!(
        record.source.artifact_ref.as_deref(),
        Some("dist/acme_guardrails.whl")
    );
    assert_eq!(
        record.source.artifact_digest.as_deref(),
        Some("sha256:abc123")
    );
    assert_eq!(
        record.status.validation.manifest,
        DynamicPluginCheckState::Valid
    );
    assert_eq!(
        record.status.validation.compatibility,
        DynamicPluginCheckState::Unknown
    );
    assert_eq!(
        record.status.validation.message.as_deref(),
        Some("manifest validated")
    );
}

#[test]
fn manifest_conversion_canonicalizes_required_strings_in_record_state() {
    let manifest = DynamicPluginManifest::parse_toml(
        r#"
manifest_version = 1

[plugin]
id = " acme.guardrails.trimmed "
kind = "worker"

[compat]
relay = " >=0.1.0,<0.2.0 "
worker_protocol = " 1 "

[defaults]
enabled = false

[capabilities]
items = ["plugin_worker"]

[load]
runtime = "python"
entrypoint = " acme_guardrails.plugin:register "
"#,
    )
    .expect("parse manifest");

    let record = manifest
        .into_record(None)
        .expect("manifest converts into record");
    assert_eq!(record.metadata.id, "acme.guardrails.trimmed");
    assert_eq!(
        record.compatibility.relay.as_deref(),
        Some(">=0.1.0,<0.2.0")
    );
    assert_eq!(record.compatibility.worker_protocol.as_deref(), Some("1"));
    assert_eq!(
        record.load,
        DynamicPluginLoadContract::Worker(DynamicPluginWorkerLoadContract {
            runtime: WorkerRuntime::Python,
            entrypoint: "acme_guardrails.plugin:register".into(),
        })
    );
}

#[test]
fn manifest_parse_and_conversion_supports_rust_dynamic_lane() {
    let manifest =
        DynamicPluginManifest::parse_toml(valid_rust_manifest_toml()).expect("parse manifest");
    let record = manifest
        .into_record(None)
        .expect("manifest converts into record");
    assert_eq!(record.metadata.kind, DynamicPluginKind::RustDynamic);
    assert_eq!(
        record.capabilities,
        vec![
            DynamicPluginCapability::PluginNative,
            DynamicPluginCapability::MiddlewareInterceptor
        ]
    );
    assert_eq!(
        record.load,
        DynamicPluginLoadContract::RustDynamic(DynamicPluginRustLoadContract {
            library: "target/release/libswitchyard.dylib".into(),
            symbol: "nemo_relay_register_plugin".into(),
        })
    );
    assert_eq!(record.compatibility.native_api.as_deref(), Some("1"));
}

#[test]
fn manifest_requires_kind_specific_worker_compatibility_and_load_fields() {
    let err = DynamicPluginManifest::parse_toml(
        r#"
manifest_version = 1

[plugin]
id = "acme.guardrails.bad"
kind = "worker"

[compat]
relay = ">=0.1.0,<0.2.0"

[defaults]
enabled = false

[capabilities]
items = ["plugin_worker"]

[load]
runtime = "python"
"#,
    )
    .expect_err("worker manifest without required fields should fail");

    match err {
        PluginError::InvalidConfig(message) => {
            assert!(
                message.contains("compat.worker_protocol") || message.contains("load.entrypoint"),
                "{message}"
            );
        }
        other => panic!("unexpected worker validation error: {other}"),
    }
}

#[test]
fn manifest_rejects_capability_kind_mismatch() {
    let err = DynamicPluginManifest::parse_toml(
        r#"
manifest_version = 1

[plugin]
id = "acme.native.bad"
kind = "rust_dynamic"

[compat]
relay = ">=0.1.0,<0.2.0"
native_api = "1"

[defaults]
enabled = false

[capabilities]
items = ["plugin_worker"]

[load]
library = "target/release/libbad.dylib"
symbol = "nemo_relay_register_plugin"
"#,
    )
    .expect_err("capability mismatch should fail");

    match err {
        PluginError::InvalidConfig(message) => {
            assert!(message.contains("plugin_native"), "{message}");
        }
        other => panic!("unexpected capability validation error: {other}"),
    }
}

#[test]
fn manifest_rejects_empty_plugin_id() {
    let err = DynamicPluginManifest::parse_toml(
        r#"
manifest_version = 1

[plugin]
id = "   "
kind = "worker"

[compat]
relay = ">=0.1.0,<0.2.0"
worker_protocol = "1"

[defaults]
enabled = false

[capabilities]
items = ["plugin_worker"]

[load]
runtime = "python"
entrypoint = "acme_guardrails.plugin:register"
"#,
    )
    .expect_err("empty plugin id should fail");

    match err {
        PluginError::InvalidConfig(message) => {
            assert!(message.contains("plugin.id"), "{message}");
        }
        other => panic!("unexpected empty-id validation error: {other}"),
    }
}

#[test]
fn manifest_requires_native_api_for_rust_dynamic_plugins() {
    let err = DynamicPluginManifest::parse_toml(
        r#"
manifest_version = 1

[plugin]
id = "acme.native.missing-api"
kind = "rust_dynamic"

[compat]
relay = ">=0.1.0,<0.2.0"

[defaults]
enabled = false

[capabilities]
items = ["plugin_native"]

[load]
library = "target/release/libmissing.dylib"
symbol = "nemo_relay_register_plugin"
"#,
    )
    .expect_err("native manifest without compat.native_api should fail");

    match err {
        PluginError::InvalidConfig(message) => {
            assert!(message.contains("compat.native_api"), "{message}");
        }
        other => panic!("unexpected native compatibility validation error: {other}"),
    }
}

#[test]
fn manifest_rejects_worker_fields_for_rust_dynamic_plugins() {
    let err = DynamicPluginManifest::parse_toml(
        r#"
manifest_version = 1

[plugin]
id = "acme.native.bad-load"
kind = "rust_dynamic"

[compat]
relay = ">=0.1.0,<0.2.0"
native_api = "1"

[defaults]
enabled = false

[capabilities]
items = ["plugin_native"]

[load]
runtime = "python"
entrypoint = "bad.plugin:register"
library = "target/release/libbad.dylib"
symbol = "nemo_relay_register_plugin"
"#,
    )
    .expect_err("native manifest with worker load fields should fail");

    match err {
        PluginError::InvalidConfig(message) => {
            assert!(
                message.contains("invalid relay-plugin.toml")
                    || message.contains("load.runtime")
                    || message.contains("load.entrypoint"),
                "{message}"
            );
        }
        other => panic!("unexpected native load validation error: {other}"),
    }
}

#[test]
fn manifest_load_from_file_path_returns_manifest_and_manifest_ref() {
    let dir = temp_dir("dynamic-plugin-manifest-file");
    let path = dir.join("custom-plugin.toml");
    fs::write(&path, valid_worker_manifest_toml()).expect("write manifest");
    let canonical = fs::canonicalize(&path).expect("canonicalize manifest path");

    let (manifest, manifest_ref) =
        DynamicPluginManifest::load_from_path(&path).expect("load manifest from file");
    assert_eq!(manifest.plugin.id, "acme.guardrails.pii");
    assert_eq!(manifest_ref, canonical.to_string_lossy());
}

#[test]
fn manifest_load_from_directory_resolves_canonical_filename() {
    let dir = temp_dir("dynamic-plugin-manifest-dir");
    let path = dir.join(DYNAMIC_PLUGIN_MANIFEST_FILENAME);
    fs::write(&path, valid_rust_manifest_toml()).expect("write manifest");
    let canonical = fs::canonicalize(&path).expect("canonicalize manifest path");

    let (manifest, manifest_ref) =
        DynamicPluginManifest::load_from_path(&dir).expect("load manifest from dir");
    assert_eq!(manifest.plugin.id, "acme.native.switchyard");
    assert_eq!(manifest_ref, canonical.to_string_lossy());
}

#[test]
fn manifest_load_from_directory_errors_when_manifest_is_missing() {
    let dir = temp_dir("dynamic-plugin-manifest-missing");
    let err = DynamicPluginManifest::load_from_path(&dir)
        .expect_err("missing relay-plugin.toml should fail");

    match err {
        PluginError::NotFound(message) => {
            assert!(
                message.contains(DYNAMIC_PLUGIN_MANIFEST_FILENAME),
                "{message}"
            );
        }
        other => panic!("unexpected missing-manifest error: {other}"),
    }
}

#[test]
fn manifest_load_from_missing_file_path_errors_with_not_found() {
    let dir = temp_dir("dynamic-plugin-manifest-missing-file");
    let missing = dir.join("relay-plugin.toml");
    let err = DynamicPluginManifest::load_from_path(&missing)
        .expect_err("missing explicit manifest file should fail");

    match err {
        PluginError::NotFound(message) => {
            assert!(message.contains("does not exist"), "{message}");
        }
        other => panic!("unexpected missing-file error: {other}"),
    }
}

#[test]
fn registry_add_manifest_converts_and_registers_validated_record() {
    let manifest =
        DynamicPluginManifest::parse_toml(valid_worker_manifest_toml()).expect("parse manifest");
    let mut registry = DynamicPluginRegistry::new();

    let record = registry
        .add_manifest(manifest, Some("/plugins/pii/relay-plugin.toml".into()))
        .expect("add manifest");
    assert_eq!(record.metadata.id, "acme.guardrails.pii");
    assert_eq!(
        record.status.validation.manifest,
        DynamicPluginCheckState::Valid
    );
    assert_eq!(
        record.status.validation.compatibility,
        DynamicPluginCheckState::Unknown
    );
    assert_eq!(
        record.source.manifest_ref.as_deref(),
        Some("/plugins/pii/relay-plugin.toml")
    );
}

#[test]
fn registry_add_manifest_preserves_canonicalized_manifest_path() {
    let dir = temp_dir("dynamic-plugin-add-manifest-path");
    let path = dir.join(DYNAMIC_PLUGIN_MANIFEST_FILENAME);
    fs::write(&path, valid_worker_manifest_toml()).expect("write manifest");
    let (manifest, manifest_ref) =
        DynamicPluginManifest::load_from_path(&dir).expect("load manifest from dir");
    let canonical = fs::canonicalize(&path).expect("canonicalize manifest path");

    let mut registry = DynamicPluginRegistry::new();
    let record = registry
        .add_manifest(manifest, Some(manifest_ref))
        .expect("add manifest");

    assert_eq!(
        record.source.manifest_ref.as_deref(),
        Some(canonical.to_string_lossy().as_ref())
    );
}

#[test]
fn manifest_rejects_defaults_enabled_true_for_dynamic_plugins() {
    let err = DynamicPluginManifest::parse_toml(
        r#"
manifest_version = 1

[plugin]
id = "acme.guardrails.enabled"
kind = "worker"

[compat]
relay = ">=0.1.0,<0.2.0"
worker_protocol = "1"

[defaults]
enabled = true

[capabilities]
items = ["plugin_worker"]

[load]
runtime = "python"
entrypoint = "acme_guardrails.plugin:register"
"#,
    )
    .expect_err("defaults.enabled=true should fail");

    match err {
        PluginError::InvalidConfig(message) => {
            assert!(message.contains("added disabled"), "{message}");
        }
        other => panic!("unexpected defaults.enabled error: {other}"),
    }
}

#[test]
fn manifest_rejects_empty_optional_source_and_integrity_strings() {
    let err = DynamicPluginManifest::parse_toml(
        r#"
manifest_version = 1

[plugin]
id = "acme.guardrails.empty-paths"
kind = "worker"

[compat]
relay = ">=0.1.0,<0.2.0"
worker_protocol = "1"

[defaults]
enabled = false

[capabilities]
items = ["plugin_worker"]

[load]
runtime = "python"
entrypoint = "acme_guardrails.plugin:register"

[source]
manifest_root = "   "
artifact = "   "

[integrity]
sha256 = "   "
signature = "   "
"#,
    )
    .expect_err("empty source/integrity strings should fail");

    match err {
        PluginError::InvalidConfig(message) => {
            assert!(
                message.contains("source.manifest_root")
                    || message.contains("source.artifact")
                    || message.contains("integrity.sha256")
                    || message.contains("integrity.signature"),
                "{message}"
            );
        }
        other => panic!("unexpected source/integrity validation error: {other}"),
    }
}

#[test]
fn registry_add_manifest_rejects_duplicate_live_ids() {
    let manifest =
        DynamicPluginManifest::parse_toml(valid_worker_manifest_toml()).expect("parse manifest");
    let manifest_ref = Some("/plugins/pii/relay-plugin.toml".into());
    let mut registry = DynamicPluginRegistry::new();
    registry
        .add_manifest(manifest.clone(), manifest_ref.clone())
        .expect("initial add_manifest succeeds");

    let err = registry
        .add_manifest(manifest, manifest_ref)
        .expect_err("duplicate manifest add should fail");
    match err {
        PluginError::Conflict(message) => {
            assert!(message.contains("already registered"), "{message}");
        }
        other => panic!("unexpected duplicate manifest error: {other}"),
    }
}

#[test]
fn registry_enable_and_disable_reject_tombstoned_records() {
    let mut registry = DynamicPluginRegistry::new();
    registry.add(sample_record()).expect("register plugin");
    registry
        .remove("acme.guardrails.pii")
        .expect("tombstone plugin");

    let enable_err = registry
        .enable("acme.guardrails.pii")
        .expect_err("cannot enable tombstoned plugin");
    match enable_err {
        PluginError::Conflict(message) => {
            assert!(message.contains("has been removed"), "{message}");
        }
        other => panic!("unexpected tombstone enable error: {other}"),
    }

    let disable_err = registry
        .disable("acme.guardrails.pii")
        .expect_err("cannot disable tombstoned plugin");
    match disable_err {
        PluginError::Conflict(message) => {
            assert!(message.contains("has been removed"), "{message}");
        }
        other => panic!("unexpected tombstone disable error: {other}"),
    }
}

#[test]
fn registry_status_updates_require_existing_plugin_ids() {
    let mut registry = DynamicPluginRegistry::new();

    let validation_err = registry
        .update_validation_status("missing.plugin", DynamicPluginValidationStatus::default())
        .expect_err("missing id should fail validation update");
    match validation_err {
        PluginError::NotFound(message) => {
            assert!(message.contains("missing.plugin"), "{message}");
        }
        other => panic!("unexpected missing-id validation error: {other}"),
    }

    let runtime_err = registry
        .update_runtime_status("missing.plugin", DynamicPluginRuntimeStatus::default())
        .expect_err("missing id should fail runtime update");
    match runtime_err {
        PluginError::NotFound(message) => {
            assert!(message.contains("missing.plugin"), "{message}");
        }
        other => panic!("unexpected missing-id runtime error: {other}"),
    }

    let error_err = registry
        .update_last_error("missing.plugin", None)
        .expect_err("missing id should fail last-error update");
    match error_err {
        PluginError::NotFound(message) => {
            assert!(message.contains("missing.plugin"), "{message}");
        }
        other => panic!("unexpected missing-id last-error error: {other}"),
    }
}

#[test]
fn manifest_rejects_unsupported_manifest_version() {
    let err = DynamicPluginManifest::parse_toml(
        r#"
manifest_version = 2

[plugin]
id = "acme.guardrails.future"
kind = "worker"

[compat]
relay = ">=0.1.0,<0.2.0"
worker_protocol = "1"

[defaults]
enabled = false

[capabilities]
items = ["plugin_worker"]

[load]
runtime = "python"
entrypoint = "future.plugin:register"
"#,
    )
    .expect_err("unsupported manifest version should fail");

    match err {
        PluginError::InvalidConfig(message) => {
            assert!(message.contains("manifest_version"), "{message}");
        }
        other => panic!("unexpected manifest version error: {other}"),
    }
}

#[test]
fn manifest_rejects_duplicate_capabilities() {
    let err = DynamicPluginManifest::parse_toml(
        r#"
manifest_version = 1

[plugin]
id = "acme.guardrails.dupe-cap"
kind = "worker"

[compat]
relay = ">=0.1.0,<0.2.0"
worker_protocol = "1"

[defaults]
enabled = false

[capabilities]
items = ["plugin_worker", "plugin_worker"]

[load]
runtime = "python"
entrypoint = "acme_guardrails.plugin:register"
"#,
    )
    .expect_err("duplicate capabilities should fail");

    match err {
        PluginError::InvalidConfig(message) => {
            assert!(message.contains("duplicate capability"), "{message}");
        }
        other => panic!("unexpected duplicate capability error: {other}"),
    }
}

#[test]
fn manifest_rejects_empty_optional_strings_when_present() {
    let err = DynamicPluginManifest::parse_toml(
        r#"
manifest_version = 1
description = "   "

[plugin]
id = "acme.guardrails.empty-strings"
name = "   "
version = "   "
kind = "worker"

[compat]
relay = ">=0.1.0,<0.2.0"
worker_protocol = "1"

[defaults]
enabled = false

[capabilities]
items = ["plugin_worker"]

[load]
runtime = "python"
entrypoint = "acme_guardrails.plugin:register"
"#,
    )
    .expect_err("empty optional strings should fail");

    match err {
        PluginError::InvalidConfig(message) => {
            assert!(
                message.contains("plugin.name")
                    || message.contains("plugin.version")
                    || message.contains("description"),
                "{message}"
            );
        }
        other => panic!("unexpected empty optional string error: {other}"),
    }
}
