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
        compatibility: DynamicPluginCompatibility::Worker(DynamicPluginWorkerCompatibility {
            relay: ">=0.1.0,<0.2.0".into(),
            worker_protocol: "grpc-v1".into(),
        }),
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
    record.compatibility =
        DynamicPluginCompatibility::RustDynamic(DynamicPluginRustCompatibility {
            relay: ">=0.1.0,<0.2.0".into(),
            native_api: "1".into(),
        });

    let err = registry
        .add(record)
        .expect_err("invalid raw record shape should fail");
    match err {
        PluginError::InvalidConfig(message) => {
            assert!(message.contains("compatibility shape"), "{message}");
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
    record.compatibility =
        DynamicPluginCompatibility::RustDynamic(DynamicPluginRustCompatibility {
            relay: ">=0.1.0,<0.2.0".into(),
            native_api: "1".into(),
        });

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
    worker_record.compatibility =
        DynamicPluginCompatibility::Worker(DynamicPluginWorkerCompatibility {
            relay: ">=0.1.0,<0.2.0".into(),
            worker_protocol: "   ".into(),
        });

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
    rust_record.compatibility =
        DynamicPluginCompatibility::RustDynamic(DynamicPluginRustCompatibility {
            relay: ">=0.1.0,<0.2.0".into(),
            native_api: "   ".into(),
        });
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
    record.compatibility = DynamicPluginCompatibility::Worker(DynamicPluginWorkerCompatibility {
        relay: String::new(),
        worker_protocol: "grpc-v1".into(),
    });

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
    record.compatibility = DynamicPluginCompatibility::Worker(DynamicPluginWorkerCompatibility {
        relay: " >=0.1.0,<0.2.0 ".into(),
        worker_protocol: " grpc-v1 ".into(),
    });
    record.load = DynamicPluginLoadContract::Worker(DynamicPluginWorkerLoadContract {
        runtime: WorkerRuntime::Python,
        entrypoint: " acme_guardrails.plugin:register ".into(),
    });

    let stored = registry.add(record).expect("register canonicalized plugin");
    assert_eq!(stored.metadata.id, "acme.guardrails.pii");
    assert_eq!(
        stored.compatibility,
        DynamicPluginCompatibility::Worker(DynamicPluginWorkerCompatibility {
            relay: ">=0.1.0,<0.2.0".into(),
            worker_protocol: "grpc-v1".into(),
        })
    );
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

#[test]
fn registry_environment_updates_stamp_validation_time() {
    let mut registry = DynamicPluginRegistry::new();
    let mut record = sample_record();
    record.status.validation.checked_at = None;
    registry.add(record).expect("register plugin");

    registry
        .update_environment(
            "acme.guardrails.pii",
            Some("/managed/environment".into()),
            DynamicPluginCheckState::Valid,
        )
        .expect("update environment");

    let record = registry
        .get("acme.guardrails.pii")
        .expect("updated plugin record");
    assert_eq!(
        record.source.environment_ref.as_deref(),
        Some("/managed/environment")
    );
    assert_eq!(
        record.status.validation.environment,
        DynamicPluginCheckState::Valid
    );
    assert!(record.status.validation.checked_at.is_some());
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
worker_protocol = "grpc-v1"

[defaults]
enabled = false

[capabilities]
items = ["plugin_worker", "config_schema"]

[config_schema]
path = "config.schema.json"

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
items = ["plugin_native"]

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
fn manifest_supports_declared_capabilities() {
    let manifest = DynamicPluginManifest::parse_toml(
        r#"
manifest_version = 1

[plugin]
id = "acme.guardrails.rich"
kind = "worker"

[compat]
relay = ">=0.1.0,<0.2.0"
worker_protocol = "grpc-v1"

[defaults]
enabled = false

[capabilities]
items = [
  "plugin_worker",
  "config_schema",
]

[config_schema]
path = "schemas/config.schema.json"

[load]
runtime = "python"
entrypoint = "acme_guardrails.plugin:register"
"#,
    )
    .expect("parse manifest with functional surface capabilities");

    assert_eq!(
        manifest.capabilities.items,
        vec![
            DynamicPluginCapability::PluginWorker,
            DynamicPluginCapability::ConfigSchema,
        ]
    );
    assert_eq!(
        manifest.config_schema,
        Some(DynamicPluginManifestConfigSchema {
            path: "schemas/config.schema.json".into(),
        })
    );
}

#[test]
fn manifest_requires_config_schema_section_for_capability() {
    let err = DynamicPluginManifest::parse_toml(
        r#"
manifest_version = 1

[plugin]
id = "acme.guardrails.missing-schema"
kind = "worker"

[compat]
relay = ">=0.1.0,<0.2.0"
worker_protocol = "grpc-v1"

[defaults]
enabled = false

[capabilities]
items = ["plugin_worker", "config_schema"]

[load]
runtime = "python"
entrypoint = "acme_guardrails.plugin:register"
"#,
    )
    .expect_err("config_schema capability without section should fail");

    match err {
        PluginError::InvalidConfig(message) => {
            assert!(
                message.contains("requires a [config_schema] section"),
                "{message}"
            );
        }
        other => panic!("unexpected missing config schema error: {other}"),
    }
}

#[test]
fn manifest_requires_config_schema_capability_for_section() {
    let err = DynamicPluginManifest::parse_toml(
        r#"
manifest_version = 1

[plugin]
id = "acme.guardrails.missing-capability"
kind = "worker"

[compat]
relay = ">=0.1.0,<0.2.0"
worker_protocol = "grpc-v1"

[defaults]
enabled = false

[capabilities]
items = ["plugin_worker"]

[config_schema]
path = "config.schema.json"

[load]
runtime = "python"
entrypoint = "acme_guardrails.plugin:register"
"#,
    )
    .expect_err("config_schema section without capability should fail");

    match err {
        PluginError::InvalidConfig(message) => {
            assert!(message.contains("requires capabilities.items"), "{message}");
        }
        other => panic!("unexpected missing config schema capability error: {other}"),
    }
}

#[test]
fn manifest_rejects_empty_config_schema_path() {
    let err = DynamicPluginManifest::parse_toml(
        &valid_worker_manifest_toml().replace("path = \"config.schema.json\"", "path = \"   \""),
    )
    .expect_err("empty config schema path should fail");

    match err {
        PluginError::InvalidConfig(message) => {
            assert!(message.contains("config_schema.path"), "{message}");
        }
        other => panic!("unexpected empty config schema path error: {other}"),
    }
}

#[test]
fn manifest_rejects_uri_config_schema_paths() {
    for path in [
        "https://example.com/config.schema.json",
        "http://example.com/config.schema.json",
        "file:///tmp/config.schema.json",
        "schema+custom://example/config.schema.json",
    ] {
        let err = DynamicPluginManifest::parse_toml(&valid_worker_manifest_toml().replace(
            "path = \"config.schema.json\"",
            &format!("path = \"{path}\""),
        ))
        .expect_err("URI config schema path should fail");

        match err {
            PluginError::InvalidConfig(message) => {
                assert!(message.contains("local filesystem path"), "{message}");
            }
            other => panic!("unexpected URI config schema path error: {other}"),
        }
    }
}

#[test]
fn manifest_rejects_unc_config_schema_paths() {
    for path_declaration in [
        r#"path = '\\server\share\config.schema.json'"#,
        r#"path = '\\?\UNC\server\share\config.schema.json'"#,
        r#"path = "//server/share/config.schema.json""#,
    ] {
        let err = DynamicPluginManifest::parse_toml(
            &valid_worker_manifest_toml()
                .replace(r#"path = "config.schema.json""#, path_declaration),
        )
        .expect_err("UNC config schema path should fail");

        match err {
            PluginError::InvalidConfig(message) => {
                assert!(message.contains("local filesystem path"), "{message}");
            }
            other => panic!("unexpected UNC config schema path error: {other}"),
        }
    }
}

#[test]
fn manifest_accepts_windows_drive_config_schema_paths() {
    for path_declaration in [
        r#"path = "C:/schemas/config.schema.json""#,
        r#"path = 'C:\schemas\config.schema.json'"#,
    ] {
        DynamicPluginManifest::parse_toml(
            &valid_worker_manifest_toml()
                .replace(r#"path = "config.schema.json""#, path_declaration),
        )
        .expect("Windows drive schema path should be accepted");
    }
}

#[test]
fn manifest_accepts_local_config_schema_paths_containing_colons() {
    for path_declaration in [
        r#"path = "schemas:v2/config.schema.json""#,
        r#"path = "C:relative.schema.json""#,
    ] {
        DynamicPluginManifest::parse_toml(
            &valid_worker_manifest_toml()
                .replace(r#"path = "config.schema.json""#, path_declaration),
        )
        .expect("local schema paths containing colons should be accepted");
    }
}

#[test]
fn manifest_resolves_relative_config_schema_path_from_manifest_parent() {
    let dir = temp_dir("dynamic-plugin-config-schema-relative");
    let manifest_path = dir.join(DYNAMIC_PLUGIN_MANIFEST_FILENAME);
    fs::write(&manifest_path, valid_worker_manifest_toml()).expect("write manifest");
    let canonical_manifest_path = fs::canonicalize(&manifest_path).expect("canonicalize manifest");
    let manifest =
        DynamicPluginManifest::parse_toml(valid_worker_manifest_toml()).expect("parse manifest");

    assert_eq!(
        manifest
            .resolve_config_schema_path(&canonical_manifest_path)
            .expect("resolve schema path"),
        Some(
            canonical_manifest_path
                .parent()
                .expect("canonical manifest parent")
                .join("config.schema.json"),
        )
    );
}

#[test]
fn manifest_resolver_rejects_mutated_non_local_config_schema_paths() {
    let mut manifest =
        DynamicPluginManifest::parse_toml(valid_worker_manifest_toml()).expect("parse manifest");

    for path in [
        "https://example.com/config.schema.json",
        r"\\server\share\config.schema.json",
        "//server/share/config.schema.json",
    ] {
        manifest.config_schema.as_mut().unwrap().path = path.to_owned();
        let error = manifest
            .resolve_config_schema_path("/plugins/native/relay-plugin.toml")
            .expect_err("public resolver must reject non-local paths");
        match error {
            PluginError::InvalidConfig(message) => {
                assert!(message.contains("local filesystem path"), "{message}");
            }
            other => panic!("unexpected config schema path error: {other}"),
        }
    }
}

#[test]
fn manifest_resolves_absolute_config_schema_path_without_filesystem_access() {
    let dir = temp_dir("dynamic-plugin-config-schema-absolute");
    let manifest_path = dir.join(DYNAMIC_PLUGIN_MANIFEST_FILENAME);
    fs::write(&manifest_path, valid_worker_manifest_toml()).expect("write manifest");
    let canonical_manifest_path = fs::canonicalize(&manifest_path).expect("canonicalize manifest");
    let absolute_schema_path = dir.join("missing.schema.json");
    let manifest_source = valid_worker_manifest_toml().replace(
        "path = \"config.schema.json\"",
        &format!("path = {:?}", absolute_schema_path.to_string_lossy()),
    );
    let manifest = DynamicPluginManifest::parse_toml(&manifest_source).expect("parse manifest");

    assert_eq!(
        manifest
            .resolve_config_schema_path(&canonical_manifest_path)
            .expect("resolve schema path"),
        Some(absolute_schema_path)
    );
}

#[test]
fn manifest_without_config_schema_resolves_to_none() {
    let manifest =
        DynamicPluginManifest::parse_toml(valid_rust_manifest_toml()).expect("parse manifest");

    assert_eq!(
        manifest
            .resolve_config_schema_path("/plugins/native/relay-plugin.toml")
            .expect("resolve absent schema path"),
        None
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
worker_protocol = " grpc-v1 "

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
        record.compatibility,
        DynamicPluginCompatibility::Worker(DynamicPluginWorkerCompatibility {
            relay: ">=0.1.0,<0.2.0".into(),
            worker_protocol: "grpc-v1".into(),
        })
    );
    assert_eq!(
        record.load,
        DynamicPluginLoadContract::Worker(DynamicPluginWorkerLoadContract {
            runtime: WorkerRuntime::Python,
            entrypoint: "acme_guardrails.plugin:register".into(),
        })
    );
}

#[test]
fn manifest_parse_accepts_rust_and_command_worker_runtimes() {
    for (runtime, expected) in [
        ("rust", WorkerRuntime::Rust),
        ("command", WorkerRuntime::Command),
    ] {
        let manifest = DynamicPluginManifest::parse_toml(&format!(
            r#"
manifest_version = 1

[plugin]
id = "acme.guardrails.{runtime}"
kind = "worker"

[compat]
relay = ">=0.1.0,<0.2.0"
worker_protocol = "grpc-v1"

[defaults]
enabled = false

[capabilities]
items = ["plugin_worker"]

[load]
runtime = "{runtime}"
entrypoint = "acme_guardrails.plugin:register"
"#
        ))
        .expect("parse worker manifest");

        let record = manifest
            .into_record(None)
            .expect("manifest converts into record");
        assert_eq!(
            record.load,
            DynamicPluginLoadContract::Worker(DynamicPluginWorkerLoadContract {
                runtime: expected,
                entrypoint: "acme_guardrails.plugin:register".into(),
            })
        );
    }
}

#[test]
fn manifest_rejects_unsupported_worker_protocol() {
    let err = DynamicPluginManifest::parse_toml(
        r#"
manifest_version = 1

[plugin]
id = "acme.guardrails.future-worker"
kind = "worker"

[compat]
relay = ">=0.1.0,<0.2.0"
worker_protocol = "grpc-v2"

[defaults]
enabled = false

[capabilities]
items = ["plugin_worker"]

[load]
runtime = "python"
entrypoint = "acme_guardrails.plugin:register"
"#,
    )
    .expect_err("unsupported worker protocol should fail");

    match err {
        PluginError::InvalidConfig(message) => {
            assert!(message.contains("grpc-v1"), "{message}");
        }
        other => panic!("unexpected worker protocol error: {other}"),
    }
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
        record.load,
        DynamicPluginLoadContract::RustDynamic(DynamicPluginRustLoadContract {
            library: "target/release/libswitchyard.dylib".into(),
            symbol: "nemo_relay_register_plugin".into(),
        })
    );
    assert_eq!(
        record.compatibility,
        DynamicPluginCompatibility::RustDynamic(DynamicPluginRustCompatibility {
            relay: ">=0.1.0,<0.2.0".into(),
            native_api: "1".into(),
        })
    );
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
worker_protocol = "grpc-v1"

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
worker_protocol = "grpc-v1"

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
worker_protocol = "grpc-v1"

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
worker_protocol = "grpc-v1"

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
worker_protocol = "grpc-v1"

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
worker_protocol = "grpc-v1"

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

#[test]
fn manifest_load_contract_serializes_both_execution_lanes() {
    let worker = DynamicPluginManifest::parse_toml(valid_worker_manifest_toml()).unwrap();
    let worker_toml = toml::to_string(&worker).expect("serialize worker manifest");
    assert!(worker_toml.contains("runtime = \"python\""));
    assert!(worker_toml.contains("entrypoint = \"acme_guardrails.plugin:register\""));
    assert!(!worker_toml.contains("library ="));

    let native = DynamicPluginManifest::parse_toml(valid_rust_manifest_toml()).unwrap();
    let native_toml = toml::to_string(&native).expect("serialize native manifest");
    assert!(native_toml.contains("library = \"target/release/libswitchyard.dylib\""));
    assert!(native_toml.contains("symbol = \"nemo_relay_register_plugin\""));
    assert!(!native_toml.contains("runtime ="));

    assert_eq!(
        DynamicPluginManifest::parse_toml(&worker_toml).unwrap(),
        worker
    );
    assert_eq!(
        DynamicPluginManifest::parse_toml(&native_toml).unwrap(),
        native
    );
}

#[test]
fn manifest_load_contract_rejects_mixed_and_empty_shapes() {
    let mixed = valid_worker_manifest_toml().replace(
        "entrypoint = \"acme_guardrails.plugin:register\"",
        "entrypoint = \"acme_guardrails.plugin:register\"\nlibrary = \"plugin.so\"",
    );
    let error = DynamicPluginManifest::parse_toml(&mixed).unwrap_err();
    assert!(error.to_string().contains("not both"));

    let empty = valid_worker_manifest_toml()
        .replace("runtime = \"python\"\n", "")
        .replace("entrypoint = \"acme_guardrails.plugin:register\"\n", "");
    let error = DynamicPluginManifest::parse_toml(&empty).unwrap_err();
    assert!(error.to_string().contains("load must declare either"));
}

#[test]
fn manifest_validation_covers_forbidden_cross_lane_fields() {
    let mut native = DynamicPluginManifest::parse_toml(valid_rust_manifest_toml()).unwrap();
    native
        .capabilities
        .items
        .push(DynamicPluginCapability::PluginWorker);
    assert!(
        native
            .validate()
            .unwrap_err()
            .to_string()
            .contains("must not declare plugin_worker")
    );

    let mut worker = DynamicPluginManifest::parse_toml(valid_worker_manifest_toml()).unwrap();
    worker.capabilities.items = vec![DynamicPluginCapability::ConfigSchema];
    assert!(
        worker
            .validate()
            .unwrap_err()
            .to_string()
            .contains("must declare capabilities.items containing plugin_worker")
    );

    let mut worker = DynamicPluginManifest::parse_toml(valid_worker_manifest_toml()).unwrap();
    worker
        .capabilities
        .items
        .push(DynamicPluginCapability::PluginNative);
    assert!(
        worker
            .validate()
            .unwrap_err()
            .to_string()
            .contains("must not declare plugin_native")
    );

    let mut native = DynamicPluginManifest::parse_toml(valid_rust_manifest_toml()).unwrap();
    native.load = DynamicPluginManifestLoad::Worker(DynamicPluginManifestWorkerLoad {
        runtime: Some(WorkerRuntime::Rust),
        entrypoint: Some("worker".into()),
    });
    assert!(
        native
            .validate()
            .unwrap_err()
            .to_string()
            .contains("must not declare load.runtime")
    );

    let mut worker = DynamicPluginManifest::parse_toml(valid_worker_manifest_toml()).unwrap();
    worker.load = DynamicPluginManifestLoad::RustDynamic(DynamicPluginManifestRustDynamicLoad {
        library: Some("plugin.so".into()),
        symbol: Some("register".into()),
    });
    assert!(
        worker
            .validate()
            .unwrap_err()
            .to_string()
            .contains("must not declare load.library")
    );

    let mut worker = DynamicPluginManifest::parse_toml(valid_worker_manifest_toml()).unwrap();
    let DynamicPluginManifestLoad::Worker(load) = &mut worker.load else {
        panic!("expected worker load contract");
    };
    load.runtime = None;
    assert!(
        worker
            .validate()
            .unwrap_err()
            .to_string()
            .contains("must declare load.runtime")
    );

    let mut native = DynamicPluginManifest::parse_toml(valid_rust_manifest_toml()).unwrap();
    native.compat.worker_protocol = Some("grpc-v1".into());
    assert!(
        native
            .validate()
            .unwrap_err()
            .to_string()
            .contains("must not declare compat.worker_protocol")
    );

    let mut worker = DynamicPluginManifest::parse_toml(valid_worker_manifest_toml()).unwrap();
    worker.compat.native_api = Some("1".into());
    assert!(
        worker
            .validate()
            .unwrap_err()
            .to_string()
            .contains("must not declare compat.native_api")
    );
}

#[test]
fn manifest_validation_checks_each_optional_source_and_integrity_field() {
    for field in ["source.artifact", "integrity.sha256", "integrity.signature"] {
        let mut manifest = DynamicPluginManifest::parse_toml(valid_worker_manifest_toml()).unwrap();
        match field {
            "source.artifact" => manifest.source.as_mut().unwrap().artifact = Some(" ".into()),
            "integrity.sha256" => manifest.integrity.as_mut().unwrap().sha256 = Some(" ".into()),
            "integrity.signature" => {
                manifest.integrity.as_mut().unwrap().signature = Some(" ".into())
            }
            _ => unreachable!(),
        }
        assert!(manifest.validate().unwrap_err().to_string().contains(field));
    }

    let manifest = DynamicPluginManifest::parse_toml(valid_worker_manifest_toml()).unwrap();
    assert!(
        manifest
            .resolve_config_schema_path("")
            .unwrap_err()
            .to_string()
            .contains("has no parent directory")
    );
}

#[test]
fn registry_reconstruction_and_raw_native_validation_cover_rejections() {
    let record = sample_record();
    let error = DynamicPluginRegistry::from_records(vec![record.clone(), record]).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("duplicated in persisted registry state")
    );

    let mut empty_id = sample_record();
    empty_id.metadata.id = " ".into();
    assert!(
        DynamicPluginRegistry::from_records(vec![empty_id])
            .unwrap_err()
            .to_string()
            .contains("id must not be empty")
    );

    let mut native = sample_record();
    native.metadata.kind = DynamicPluginKind::RustDynamic;
    native.compatibility = DynamicPluginCompatibility::Worker(DynamicPluginWorkerCompatibility {
        relay: ">=0.5,<1.0".into(),
        worker_protocol: "grpc-v1".into(),
    });
    native.load = DynamicPluginLoadContract::RustDynamic(DynamicPluginRustLoadContract {
        library: "plugin.so".into(),
        symbol: "register".into(),
    });
    assert!(
        DynamicPluginRegistry::from_records(vec![native])
            .unwrap_err()
            .to_string()
            .contains("compatibility shape")
    );

    let mut native = sample_record();
    native.metadata.kind = DynamicPluginKind::RustDynamic;
    native.compatibility =
        DynamicPluginCompatibility::RustDynamic(DynamicPluginRustCompatibility {
            relay: ">=0.5,<1.0".into(),
            native_api: "1".into(),
        });
    native.load = DynamicPluginLoadContract::Worker(DynamicPluginWorkerLoadContract {
        runtime: WorkerRuntime::Rust,
        entrypoint: "worker".into(),
    });
    assert!(
        DynamicPluginRegistry::from_records(vec![native])
            .unwrap_err()
            .to_string()
            .contains("load shape")
    );

    let mut registry = DynamicPluginRegistry::new();
    let mut disabled = sample_record();
    disabled.spec.enabled = false;
    registry.add(disabled).unwrap();
    assert!(!registry.disable("acme.guardrails.pii").unwrap());
}

#[test]
fn annotated_request_consumers_must_exclude_relay_zero_five() {
    let error = validate_annotated_request_consumer_compatibility(
        ">=0.5,<1.0",
        "example.request_interceptor",
    )
    .unwrap_err();
    assert!(error.to_string().contains(">=0.6,<1.0"));

    validate_annotated_request_consumer_compatibility(">=0.6,<1.0", "example.request_interceptor")
        .unwrap();
}
