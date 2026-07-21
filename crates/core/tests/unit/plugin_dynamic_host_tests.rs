// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::plugin::{PLUGIN_HANDLERS, PLUGIN_MUTATION_OWNER, PluginMutationOwner};
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
