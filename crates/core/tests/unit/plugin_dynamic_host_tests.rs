// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::plugin::{
    PLUGIN_HANDLERS, PLUGIN_MUTATION_OWNER, PluginHostValidationRequest,
    PluginHostValidationTarget, PluginMutationOwner,
};
use serde_json::{Map, Value as Json};

struct PoisonedRegistryCleanup;

impl Drop for PoisonedRegistryCleanup {
    fn drop(&mut self) {
        PLUGIN_HANDLERS.clear_poison();
        if let Ok(mut owner) = PLUGIN_MUTATION_OWNER.lock() {
            *owner = PluginMutationOwner::Idle;
        }
    }
}

#[test]
fn unsafe_kind_deregistration_keeps_activation_retryable() {
    let _guard = crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let _cleanup = PoisonedRegistryCleanup;
    let claim = acquire_plugin_host_lease().expect("fixture host should acquire the owner");
    let owner_id = claim.owner_id();
    let mut activation = PluginHostActivation {
        active: true,
        report: PluginHostReport::default(),
        native: Some(NativePluginActivation::with_plugin_kind_for_test(
            "fixture.poisoned",
        )),
        #[cfg(feature = "worker-grpc")]
        worker: None,
        claim: Some(claim),
    };

    std::thread::spawn(|| {
        let _registry = PLUGIN_HANDLERS.write().unwrap();
        panic!("poison plugin registry for teardown test");
    })
    .join()
    .expect_err("fixture registry writer should panic");

    let error = activation
        .clear_inner()
        .expect_err("an uncertain kind deregistration must retain the activation")
        .to_string();
    assert!(error.contains("plugin registry lock poisoned"), "{error}");
    assert!(error.contains("activation owner were retained"), "{error}");
    assert!(activation.is_active());
    assert_eq!(
        *PLUGIN_MUTATION_OWNER.lock().unwrap(),
        PluginMutationOwner::Host(owner_id)
    );
    assert!(matches!(
        acquire_plugin_host_lease(),
        Err(crate::plugin::PluginError::Conflict(_))
    ));

    PLUGIN_HANDLERS.clear_poison();
    let retry_error = activation
        .clear_inner()
        .expect_err("the missing fixture registration should still be reported")
        .to_string();
    assert!(retry_error.contains("was not registered"), "{retry_error}");
    assert!(!activation.is_active());
    assert_eq!(
        *PLUGIN_MUTATION_OWNER.lock().unwrap(),
        PluginMutationOwner::Idle
    );
}

#[test]
fn dynamic_plugin_specs_require_unique_nonempty_input() {
    let empty = validate_dynamic_plugin_specs(&[]).unwrap_err().to_string();
    assert!(
        empty.contains("requires at least one dynamic plugin"),
        "{empty}"
    );

    let duplicate = VerifiedDynamicPluginSpec {
        plugin_id: "fixture.duplicate".into(),
        kind: DynamicPluginKind::RustDynamic,
        manifest_ref: "relay-plugin.toml".into(),
        environment_ref: None,
        config: Map::new(),
    };
    let error = validate_dynamic_plugin_specs(&[duplicate.clone(), duplicate])
        .unwrap_err()
        .to_string();
    assert!(error.contains("duplicate dynamic plugin id"), "{error}");
}

#[test]
fn plugin_error_context_preserves_each_error_class() {
    use crate::plugin::PluginError;

    let serialization = serde_json::from_str::<Json>("{").unwrap_err();
    let errors = [
        PluginError::InvalidConfig("invalid".into()),
        PluginError::Conflict("conflict".into()),
        PluginError::NotFound("missing".into()),
        PluginError::Serialization(serialization),
        PluginError::Internal("internal".into()),
        PluginError::RegistrationFailed("registration".into()),
    ];

    for error in errors {
        let message = plugin_error_context("dynamic load", error).to_string();
        assert!(message.contains("dynamic load"), "{message}");
    }
}

#[test]
fn retained_runtime_errors_include_cleanup_details_when_available() {
    let default_error = retained_runtime_error(Vec::new()).to_string();
    assert!(
        default_error.contains("teardown was incomplete"),
        "{default_error}"
    );

    let detailed_error =
        retained_runtime_error(vec!["registry remained active".into()]).to_string();
    assert!(
        detailed_error.contains("registry remained active"),
        "{detailed_error}"
    );
}

#[tokio::test]
#[allow(clippy::await_holding_lock)] // Serializes process-wide host ownership for this test.
async fn static_only_host_is_owned_conflict_checked_and_reinitializable() {
    let _guard = crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let _ = crate::plugin::test_close_plugin_host();

    let mut activation =
        PluginHostActivation::initialize_exact(crate::plugin::PluginConfig::default())
            .await
            .expect("static host should activate");
    assert!(activation.is_active());
    assert!(activation.report().dynamic_plugins.is_empty());
    crate::plugin::record_active_plugin_runtime_diagnostic(crate::plugin::RuntimeDiagnostic {
        code: "plugin.host.live_report".into(),
        component: "fixture".into(),
        field: None,
        message: "runtime diagnostic".into(),
        session_id: None,
        count: 1,
    });
    assert_eq!(
        activation.report().config.runtime_diagnostics[0].code,
        "plugin.host.live_report"
    );
    assert!(matches!(
        PluginHostActivation::initialize_exact(crate::plugin::PluginConfig::default()).await,
        Err(crate::plugin::PluginError::Conflict(_))
    ));

    activation.close().expect("static host should close");
    let mut activation =
        PluginHostActivation::initialize_exact(crate::plugin::PluginConfig::default())
            .await
            .expect("host lease should be reusable after close");
    activation.close().expect("reinitialized host should close");
}

#[test]
fn standalone_validation_targets_use_a_stable_tagged_wire_shape() {
    let request = PluginHostValidationRequest {
        config: crate::plugin::PluginConfig::default(),
        additional_plugins_toml: None,
        target: PluginHostValidationTarget::PluginId("fixture.plugin".into()),
    };
    let value = serde_json::to_value(request).expect("request should serialize");
    assert_eq!(value["target"], "plugin_id");
    assert_eq!(value["value"], "fixture.plugin");
}

#[test]
fn standalone_validation_defaults_an_omitted_target_to_all() {
    let request: PluginHostValidationRequest =
        serde_json::from_value(serde_json::json!({"config": {}}))
            .expect("omitted target should deserialize");
    assert_eq!(request.target, PluginHostValidationTarget::All);
}

// Exercise the same resolution/activation boundary as initialize(), but swap
// source files deterministically between those phases instead of racing threads.
#[test]
fn embedding_activation_loads_verified_bytes_after_source_replacement() {
    verified_native_activation_survives_source_swap(false);
}

#[cfg(unix)]
#[test]
fn embedding_activation_loads_verified_bytes_after_source_symlink_swap() {
    verified_native_activation_survives_source_swap(true);
}

fn verified_native_activation_survives_source_swap(symlink_swap: bool) {
    use base64::Engine;
    use ring::signature::{Ed25519KeyPair, KeyPair};
    use sha2::{Digest, Sha256};
    use std::fs;

    let _guard = crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let library_name = if cfg!(target_os = "windows") {
        "nemo_relay_plugin_fixture.dll"
    } else if cfg!(target_os = "macos") {
        "libnemo_relay_plugin_fixture.dylib"
    } else {
        "libnemo_relay_plugin_fixture.so"
    };
    let fixture = std::env::var_os("NEMO_RELAY_TEST_NATIVE_PLUGIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../target/test-plugin-fixtures/debug")
                .join(library_name)
        });
    let bytes = fs::read(&fixture).expect("run just build-test-plugin-fixtures first");
    let temp = tempfile::tempdir().unwrap();
    let library = temp.path().join(library_name);
    fs::write(&library, &bytes).unwrap();
    let key = Ed25519KeyPair::from_seed_unchecked(&[42; 32]).unwrap();
    let encode = |bytes: &[u8]| base64::engine::general_purpose::STANDARD.encode(bytes);
    fs::write(
        temp.path().join("artifact.sig"),
        encode(key.sign(&bytes).as_ref()),
    )
    .unwrap();
    let manifest_ref = temp.path().join("relay-plugin.toml");
    fs::write(
        &manifest_ref,
        format!(
            r#"
manifest_version = 1
[plugin]
id = "fixture_native"
kind = "rust_dynamic"
[compat]
relay = "={version}"
native_api = "1"
[defaults]
enabled = false
[capabilities]
items = ["plugin_native"]
[source]
artifact = {library_name:?}
[integrity]
sha256 = "sha256:{digest}"
signature = "artifact.sig"
[load]
library = {library_name:?}
symbol = "nemo_relay_fixture_native_plugin"
"#,
            version = env!("CARGO_PKG_VERSION"),
            digest = Sha256::digest(&bytes)
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        ),
    )
    .unwrap();
    let config_path = temp.path().join("plugins.toml");
    fs::write(
        &config_path,
        format!(
            r#"
version = 1
[plugins.policy.defaults]
startup = "required"
attestation = "signature_required"
trusted_public_keys = ["ed25519:{public_key}"]
[[plugins.dynamic]]
manifest = {manifest_ref:?}
"#,
            public_key = encode(key.public_key().as_ref()),
            manifest_ref = manifest_ref.to_string_lossy()
        ),
    )
    .unwrap();
    let resolved = resolve_plugin_host_config(PluginConfig::default(), Some(&config_path)).unwrap();
    let report = &resolved
        .dynamic_reports
        .iter()
        .find(|entry| entry.report.plugin_id == "fixture_native")
        .unwrap()
        .report;
    assert_eq!(
        report.status.authenticity,
        super::super::DynamicPluginCheckState::Valid,
        "{report:?}"
    );
    assert!(report.selected);

    // Supply a matching digest for the replacement, as in the reported attack.
    let replacement_bytes = b"unverified library bytes";
    let digest = |bytes: &[u8]| {
        Sha256::digest(bytes)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    };
    let replacement_manifest = fs::read_to_string(&manifest_ref)
        .unwrap()
        .replace(&digest(&bytes), &digest(replacement_bytes));
    fs::remove_file(&manifest_ref).unwrap();
    fs::remove_file(&library).unwrap();
    if symlink_swap {
        #[cfg(unix)]
        {
            let replacement = temp.path().join("replacement-library");
            fs::write(&replacement, replacement_bytes).unwrap();
            std::os::unix::fs::symlink(replacement, &library).unwrap();
            let replacement = temp.path().join("replacement-manifest");
            fs::write(&replacement, &replacement_manifest).unwrap();
            std::os::unix::fs::symlink(replacement, &manifest_ref).unwrap();
        }
    } else {
        fs::write(&library, replacement_bytes).unwrap();
        fs::write(&manifest_ref, &replacement_manifest).unwrap();
    }
    fs::write(temp.path().join("artifact.sig"), "unverified signature").unwrap();
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        let (mut activation, _) = PluginHostActivation::activate_validated(
            resolved.config,
            resolved.dynamic_plugins,
            resolved.diagnostics,
            resolved.load_trust,
            resolved.native_artifacts,
        )
        .await
        .expect("activation must load the verified fixture, without reopening source paths");
        assert!(crate::plugin::lookup_plugin("fixture_native").is_some());
        activation.close().unwrap();
        assert!(crate::plugin::lookup_plugin("fixture_native").is_none());
    });
}

#[tokio::test]
#[allow(clippy::await_holding_lock)] // Serializes process-wide host ownership for this test.
async fn optional_load_time_trust_failure_skips_component_and_preserves_host() {
    use crate::plugin::dynamic::{
        DynamicPluginAttestationMode, DynamicPluginHostPolicy, DynamicPluginManifest,
        DynamicPluginStartupClass, evaluate_dynamic_plugin_host_policy,
    };
    let _guard = crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let _ = crate::plugin::test_close_plugin_host();
    let temp = tempfile::tempdir().unwrap();
    for kind in [DynamicPluginKind::RustDynamic, DynamicPluginKind::Worker] {
        if kind == DynamicPluginKind::Worker && !cfg!(feature = "worker-grpc") {
            continue;
        }
        let (compat, load) = if kind == DynamicPluginKind::RustDynamic {
            (
                "native_api = \"1\"",
                "library = \"never-load.so\"\nsymbol = \"never_call\"",
            )
        } else {
            (
                "worker_protocol = \"grpc-v1\"",
                "runtime = \"command\"\nentrypoint = \"never-launch\"",
            )
        };
        let capability = if kind == DynamicPluginKind::RustDynamic {
            "plugin_native"
        } else {
            "plugin_worker"
        };
        let manifest = DynamicPluginManifest::parse_toml(&format!(
            r#"
manifest_version = 1
[plugin]
id = "fixture.skipped"
kind = "{kind}"
[compat]
relay = ">=0.8.0,<1.0"
{compat}
[defaults]
enabled = false
[capabilities]
items = ["{capability}"]
[source]
artifact = "missing-artifact"
[integrity]
sha256 = "sha256:0000000000000000000000000000000000000000000000000000000000000000"
[load]
{load}
"#
        ))
        .unwrap();
        let mut policy = DynamicPluginHostPolicy::default();
        policy.defaults.startup = Some(DynamicPluginStartupClass::Optional);
        policy.defaults.attestation = Some(DynamicPluginAttestationMode::IntegrityOnly);
        let context = DynamicPluginLoadTrust {
            policy: evaluate_dynamic_plugin_host_policy(&policy, &manifest),
            manifest,
        };
        let manifest_ref = temp
            .path()
            .join("relay-plugin.toml")
            .to_string_lossy()
            .into_owned();
        let spec = VerifiedDynamicPluginSpec {
            plugin_id: "fixture.skipped".into(),
            kind,
            manifest_ref,
            environment_ref: None,
            config: Map::new(),
        };
        let (mut activation, _) = PluginHostActivation::activate_validated(
            PluginConfig::default(),
            vec![spec],
            Vec::new(),
            HashMap::from([("fixture.skipped".into(), context)]),
            Vec::new(),
        )
        .await
        .unwrap();
        assert!(activation.is_active());
        let report = activation.report();
        assert_eq!(report.dynamic_plugins.len(), 1);
        assert!(!report.dynamic_plugins[0].selected);
        assert_eq!(
            report.dynamic_plugins[0].failure.as_ref().unwrap().code,
            "integrity_failed"
        );
        assert!(
            !report
                .config
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.component.as_deref() == Some("fixture.skipped"))
        );
        activation.close().unwrap();
    }
}
