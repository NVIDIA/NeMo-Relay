// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::plugin::{PLUGIN_HANDLERS, PLUGIN_MUTATION_OWNER, PluginMutationOwner};
use serde_json::{Map, Value as Json};
use std::sync::atomic::{AtomicUsize, Ordering};

fn invalid_native_library_fixture(plugin_id: &str) -> (tempfile::TempDir, String) {
    let directory = tempfile::tempdir().expect("native plugin fixture directory should create");
    std::fs::write(
        directory.path().join("invalid-library"),
        b"not a native library",
    )
    .expect("invalid native plugin library should write");
    let manifest = directory.path().join("relay-plugin.toml");
    std::fs::write(
        &manifest,
        format!(
            r#"manifest_version = 1

[plugin]
id = {plugin_id:?}
kind = "rust_dynamic"

[compat]
relay = ">=0.5,<1.0"
native_api = "1"

[defaults]
enabled = false

[capabilities]
items = ["plugin_native"]

[load]
library = "invalid-library"
symbol = "nemo_relay_register_plugin"
"#
        ),
    )
    .expect("native plugin manifest should write");
    (directory, manifest.to_string_lossy().into_owned())
}

struct TrackingActivationResource {
    verify_count: Arc<AtomicUsize>,
    drop_count: Arc<AtomicUsize>,
    fail_verification: bool,
}

struct PanickingActivationResource {
    verify_count: Arc<AtomicUsize>,
    drop_count: Arc<AtomicUsize>,
}

struct TrackingPartialRuntime {
    drop_count: Arc<AtomicUsize>,
}

impl Drop for TrackingPartialRuntime {
    fn drop(&mut self) {
        self.drop_count.fetch_add(1, Ordering::SeqCst);
    }
}

impl DynamicPluginActivationResource for TrackingActivationResource {
    fn verify(&self) -> crate::plugin::Result<()> {
        self.verify_count.fetch_add(1, Ordering::SeqCst);
        if self.fail_verification {
            Err(crate::plugin::PluginError::InvalidConfig(
                "activation snapshot changed".into(),
            ))
        } else {
            Ok(())
        }
    }
}

impl DynamicPluginActivationResource for PanickingActivationResource {
    fn verify(&self) -> crate::plugin::Result<()> {
        self.verify_count.fetch_add(1, Ordering::SeqCst);
        panic!("injected activation resource verification panic");
    }
}

impl Drop for TrackingActivationResource {
    fn drop(&mut self) {
        self.drop_count.fetch_add(1, Ordering::SeqCst);
    }
}

impl Drop for PanickingActivationResource {
    fn drop(&mut self) {
        self.drop_count.fetch_add(1, Ordering::SeqCst);
    }
}

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
fn unsafe_kind_deregistration_retains_runtime_and_owner() {
    let _guard = crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let _cleanup = PoisonedRegistryCleanup;
    let claim = acquire_plugin_host_lease().expect("fixture host should acquire the owner");
    let owner_id = claim.owner_id();
    let mut activation = PluginHostActivation {
        active: true,
        native: Some(NativePluginActivation::with_plugin_kind_for_test(
            "fixture.poisoned",
        )),
        #[cfg(feature = "worker-grpc")]
        worker: None,
        resource_anchors: Vec::new(),
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
    assert!(!activation.is_active());
    assert_eq!(
        *PLUGIN_MUTATION_OWNER.lock().unwrap(),
        PluginMutationOwner::Host(owner_id)
    );
    assert!(matches!(
        acquire_plugin_host_lease(),
        Err(crate::plugin::PluginError::Conflict(_))
    ));
}

#[test]
fn unsafe_partial_load_rollback_retains_runtime_and_owner() {
    let _guard = crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let _cleanup = PoisonedRegistryCleanup;
    if let Ok(mut owner) = PLUGIN_MUTATION_OWNER.lock() {
        *owner = PluginMutationOwner::Idle;
    }
    let drop_count = Arc::new(AtomicUsize::new(0));
    let mut rollback = DynamicPluginTeardownOutcome::success();
    rollback.record_error("plugin registry lock poisoned", false);
    let failure = super::super::finish_partial_load_rollback(
        TrackingPartialRuntime {
            drop_count: Arc::clone(&drop_count),
        },
        crate::plugin::PluginError::NotFound("second plugin manifest".into()),
        rollback,
    );
    assert_eq!(drop_count.load(Ordering::SeqCst), 0);

    let mut claim = Some(acquire_plugin_host_lease().expect("fixture host should own activation"));
    let owner_id = claim.as_ref().unwrap().owner_id();
    let error = finalize_load_failure("native plugin load failed", &mut claim, failure).to_string();

    assert!(claim.is_none());
    assert!(
        error.contains("partially loaded runtime was retained"),
        "{error}"
    );
    assert!(error.contains("activation owner was retained"), "{error}");
    assert_eq!(
        *PLUGIN_MUTATION_OWNER.lock().unwrap(),
        PluginMutationOwner::Host(owner_id)
    );
    assert!(matches!(
        acquire_plugin_host_lease(),
        Err(crate::plugin::PluginError::Conflict(_))
    ));
    assert_eq!(drop_count.load(Ordering::SeqCst), 0);
}

#[test]
fn activation_panic_retains_planned_resource_and_owner() {
    let _guard = crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let _cleanup = PoisonedRegistryCleanup;
    if let Ok(mut owner) = PLUGIN_MUTATION_OWNER.lock() {
        *owner = PluginMutationOwner::Idle;
    }
    let verify_count = Arc::new(AtomicUsize::new(0));
    let drop_count = Arc::new(AtomicUsize::new(0));
    let (_fixture, manifest_ref) = invalid_native_library_fixture("fixture.panicking-resource");

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("file-backed activation runtime should build");
    let result = runtime.block_on(PluginHostActivation::activate_plan(
        PluginHostActivationPlan {
            config: PluginConfig::default(),
            dynamic_plugins: vec![PlannedDynamicPluginActivation {
                spec: DynamicPluginActivationSpec {
                    plugin_id: "fixture.panicking-resource".into(),
                    kind: DynamicPluginKind::RustDynamic,
                    manifest_ref,
                    environment_ref: None,
                    config: Map::new(),
                },
                resource: Arc::new(PanickingActivationResource {
                    verify_count: Arc::clone(&verify_count),
                    drop_count: Arc::clone(&drop_count),
                }),
            }],
            diagnostics: Vec::new(),
        },
    ));
    let error = match result {
        Ok((activation, _)) => {
            std::mem::forget(activation);
            panic!("the injected activation panic should fail the plan");
        }
        Err(error) => error.to_string(),
    };

    assert!(
        error.contains("file-backed plugin activation task failed"),
        "{error}"
    );
    assert_eq!(verify_count.load(Ordering::SeqCst), 1);
    assert_eq!(drop_count.load(Ordering::SeqCst), 0);
    assert!(matches!(
        *PLUGIN_MUTATION_OWNER.lock().unwrap(),
        PluginMutationOwner::Host(_)
    ));
    assert!(matches!(
        acquire_plugin_host_lease(),
        Err(crate::plugin::PluginError::Conflict(_))
    ));
}

#[test]
fn teardown_panic_after_deactivation_retains_runtime_resource_and_owner() {
    let _guard = crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let _cleanup = PoisonedRegistryCleanup;
    if let Ok(mut owner) = PLUGIN_MUTATION_OWNER.lock() {
        *owner = PluginMutationOwner::Idle;
    }
    let claim = acquire_plugin_host_lease().expect("fixture host should acquire the owner");
    let owner_id = claim.owner_id();
    let resource_drop_count = Arc::new(AtomicUsize::new(0));
    let mut activation = PluginHostActivation {
        active: true,
        native: Some(NativePluginActivation::with_resource_for_test(Arc::new(
            TrackingActivationResource {
                verify_count: Arc::new(AtomicUsize::new(0)),
                drop_count: Arc::clone(&resource_drop_count),
                fail_verification: false,
            },
        ))),
        #[cfg(feature = "worker-grpc")]
        worker: None,
        resource_anchors: Vec::new(),
        claim: Some(claim),
    };
    PANIC_PLUGIN_HOST_CLEAR_AFTER_DEACTIVATION.store(true, Ordering::SeqCst);

    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| activation.clear_inner()));

    assert!(panic.is_err(), "the injected teardown panic should unwind");
    assert!(!activation.is_active());
    assert!(activation.native.is_none());
    assert!(activation.claim.is_none());
    assert_eq!(resource_drop_count.load(Ordering::SeqCst), 0);
    assert_eq!(
        *PLUGIN_MUTATION_OWNER.lock().unwrap(),
        PluginMutationOwner::Host(owner_id)
    );
    assert!(matches!(
        acquire_plugin_host_lease(),
        Err(crate::plugin::PluginError::Conflict(_))
    ));
}

#[test]
fn runtime_unload_panic_retains_activation_resource_and_owner() {
    let _guard = crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let _cleanup = PoisonedRegistryCleanup;
    if let Ok(mut owner) = PLUGIN_MUTATION_OWNER.lock() {
        *owner = PluginMutationOwner::Idle;
    }
    let claim = acquire_plugin_host_lease().expect("fixture host should acquire the owner");
    let owner_id = claim.owner_id();
    let resource_drop_count = Arc::new(AtomicUsize::new(0));
    let resource: Arc<dyn DynamicPluginActivationResource> = Arc::new(TrackingActivationResource {
        verify_count: Arc::new(AtomicUsize::new(0)),
        drop_count: Arc::clone(&resource_drop_count),
        fail_verification: false,
    });
    let mut activation = PluginHostActivation {
        active: true,
        native: Some(
            NativePluginActivation::with_panicking_unload_resource_for_test(Arc::clone(&resource)),
        ),
        #[cfg(feature = "worker-grpc")]
        worker: None,
        resource_anchors: vec![resource],
        claim: Some(claim),
    };

    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| activation.clear_inner()));

    assert!(panic.is_err(), "the injected unload panic should unwind");
    assert!(!activation.is_active());
    assert!(activation.native.is_none());
    assert!(activation.resource_anchors.is_empty());
    assert!(activation.claim.is_none());
    assert_eq!(resource_drop_count.load(Ordering::SeqCst), 0);
    assert_eq!(
        *PLUGIN_MUTATION_OWNER.lock().unwrap(),
        PluginMutationOwner::Host(owner_id)
    );
    assert!(matches!(
        acquire_plugin_host_lease(),
        Err(crate::plugin::PluginError::Conflict(_))
    ));
}

#[test]
fn panic_retention_guard_keeps_partial_runtime_alive_during_unwind() {
    let drop_count = Arc::new(AtomicUsize::new(0));
    let panic = std::panic::catch_unwind({
        let drop_count = Arc::clone(&drop_count);
        move || {
            let _activation =
                super::super::PanicRetentionGuard::new(TrackingPartialRuntime { drop_count });
            panic!("injected partial runtime panic");
        }
    });

    assert!(panic.is_err());
    assert_eq!(drop_count.load(Ordering::SeqCst), 0);
}

#[test]
fn component_failure_uses_checked_runtime_rollback_and_retains_owner_when_unsafe() {
    let _guard = crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let _cleanup = PoisonedRegistryCleanup;
    if let Ok(mut owner) = PLUGIN_MUTATION_OWNER.lock() {
        *owner = PluginMutationOwner::Idle;
    }
    let mut claim = Some(acquire_plugin_host_lease().expect("fixture host should acquire owner"));
    let owner_id = claim.as_ref().unwrap().owner_id();
    let mut native = Some(NativePluginActivation::with_plugin_kind_for_test(
        "fixture.partial-component",
    ));
    #[cfg(feature = "worker-grpc")]
    let mut worker = None;

    std::thread::spawn(|| {
        let _registry = PLUGIN_HANDLERS.write().unwrap();
        panic!("poison plugin registry during component failure rollback");
    })
    .join()
    .expect_err("fixture registry writer should panic");

    let error = finalize_configuration_failure(
        crate::plugin::PluginError::RegistrationFailed(
            "injected component initialization failure".into(),
        ),
        Vec::new(),
        &mut native,
        #[cfg(feature = "worker-grpc")]
        &mut worker,
        &mut claim,
    )
    .to_string();

    assert!(native.is_none());
    assert!(claim.is_none());
    assert!(
        error.contains("injected component initialization failure"),
        "{error}"
    );
    assert!(
        error.contains("dynamic runtime rollback was incomplete"),
        "{error}"
    );
    assert!(error.contains("activation owner were retained"), "{error}");
    assert_eq!(
        *PLUGIN_MUTATION_OWNER.lock().unwrap(),
        PluginMutationOwner::Host(owner_id)
    );
    assert!(matches!(
        acquire_plugin_host_lease(),
        Err(crate::plugin::PluginError::Conflict(_))
    ));
}

#[test]
fn dynamic_plugin_specs_require_unique_nonempty_input() {
    let empty = validate_dynamic_plugin_specs(&[]).unwrap_err().to_string();
    assert!(
        empty.contains("requires at least one dynamic plugin"),
        "{empty}"
    );

    let duplicate = DynamicPluginActivationSpec {
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
fn activation_plan_allows_owned_static_only_configuration() {
    let _guard = crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if let Ok(mut owner) = PLUGIN_MUTATION_OWNER.lock() {
        *owner = PluginMutationOwner::Idle;
    }

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("file-backed activation runtime should build");
    let (activation, report) = runtime
        .block_on(PluginHostActivation::activate_plan(
            PluginHostActivationPlan {
                config: PluginConfig::default(),
                dynamic_plugins: Vec::new(),
                diagnostics: Vec::new(),
            },
        ))
        .expect("a file-backed static-only plan should activate");

    assert!(activation.is_active());
    assert!(!report.has_errors());
    assert!(crate::plugin::active_plugin_report().is_some());
    activation
        .clear()
        .expect("the static-only file-backed owner should clear");
    assert!(crate::plugin::active_plugin_report().is_none());
    assert_eq!(
        *PLUGIN_MUTATION_OWNER.lock().unwrap(),
        PluginMutationOwner::Idle
    );
}

#[test]
fn activation_plan_verifies_resources_before_loading() {
    let _guard = crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if let Ok(mut owner) = PLUGIN_MUTATION_OWNER.lock() {
        *owner = PluginMutationOwner::Idle;
    }
    let verify_count = Arc::new(AtomicUsize::new(0));
    let drop_count = Arc::new(AtomicUsize::new(0));
    let resource = Arc::new(TrackingActivationResource {
        verify_count: Arc::clone(&verify_count),
        drop_count: Arc::clone(&drop_count),
        fail_verification: true,
    });
    let (_fixture, manifest_ref) = invalid_native_library_fixture("fixture.snapshot");

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("file-backed activation runtime should build");
    let result = runtime.block_on(PluginHostActivation::activate_plan(
        PluginHostActivationPlan {
            config: PluginConfig::default(),
            dynamic_plugins: vec![PlannedDynamicPluginActivation {
                spec: DynamicPluginActivationSpec {
                    plugin_id: "fixture.snapshot".into(),
                    kind: DynamicPluginKind::RustDynamic,
                    manifest_ref,
                    environment_ref: None,
                    config: Map::new(),
                },
                resource,
            }],
            diagnostics: Vec::new(),
        },
    ));
    let error = match result {
        Ok((activation, _)) => {
            activation
                .clear()
                .expect("unexpected resource activation should clear");
            panic!("resource verification failure should prevent loading");
        }
        Err(error) => error.to_string(),
    };

    assert!(error.contains("activation snapshot changed"), "{error}");
    assert_eq!(verify_count.load(Ordering::SeqCst), 1);
    assert_eq!(drop_count.load(Ordering::SeqCst), 1);
    assert_eq!(
        *PLUGIN_MUTATION_OWNER.lock().unwrap(),
        PluginMutationOwner::Idle
    );
}

#[test]
fn unpolled_activation_plan_drops_resources_without_enqueuing() {
    let _guard = crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if let Ok(mut owner) = PLUGIN_MUTATION_OWNER.lock() {
        *owner = PluginMutationOwner::Idle;
    }
    let verify_count = Arc::new(AtomicUsize::new(0));
    let drop_count = Arc::new(AtomicUsize::new(0));
    let resource = Arc::new(TrackingActivationResource {
        verify_count: Arc::clone(&verify_count),
        drop_count: Arc::clone(&drop_count),
        fail_verification: false,
    });

    let activation = PluginHostActivation::activate_plan(PluginHostActivationPlan {
        config: PluginConfig::default(),
        dynamic_plugins: vec![PlannedDynamicPluginActivation {
            spec: DynamicPluginActivationSpec {
                plugin_id: "fixture.unpolled".into(),
                kind: DynamicPluginKind::RustDynamic,
                manifest_ref: "unpolled-relay-plugin.toml".into(),
                environment_ref: None,
                config: Map::new(),
            },
            resource,
        }],
        diagnostics: Vec::new(),
    });
    drop(activation);

    assert_eq!(verify_count.load(Ordering::SeqCst), 0);
    assert_eq!(drop_count.load(Ordering::SeqCst), 1);
    assert_eq!(
        *PLUGIN_MUTATION_OWNER.lock().unwrap(),
        PluginMutationOwner::Idle
    );
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
