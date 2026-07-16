// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Owned activation lifecycle for dynamically loaded plugin components.
//!
//! Activation transactions run on Relay's process-wide plugin lifecycle
//! executor. This keeps registration cancellation-resistant and gives native
//! and worker plugins a stable Tokio runtime independent of the embedding
//! caller. Plugin registration therefore must not depend on caller-thread
//! affinity; the lifecycle executor remains available for the process lifetime.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as Json};

use crate::plugin::{
    ConfigReport, PluginComponentSpec, PluginConfig, PluginHostLease, Result,
    acquire_plugin_host_lease, clear_plugin_configuration_for_host,
    ensure_builtin_plugins_registered, initialize_plugins_exact_for_host, resolve_plugin_config,
    run_owned_plugin_mutation,
};

use super::{
    DynamicPluginKind, DynamicPluginTeardownOutcome, NativePluginActivation, NativePluginLoadSpec,
    load_native_plugins,
};

#[cfg(feature = "worker-grpc")]
use super::{WorkerPluginActivation, WorkerPluginLoadSpec, load_worker_plugins};

/// One dynamic plugin component to load and activate in an embedding host.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DynamicPluginActivationSpec {
    /// Expected plugin identifier from the authored manifest.
    pub plugin_id: String,
    /// Plugin execution lane.
    pub kind: DynamicPluginKind,
    /// Path or reference to the authored `relay-plugin.toml`.
    pub manifest_ref: String,
    /// Relay-managed runtime environment used by Python workers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub environment_ref: Option<String>,
    /// Component-local configuration passed to the dynamically loaded plugin.
    #[serde(default)]
    pub config: Map<String, Json>,
}

/// Owns one process-wide dynamic plugin configuration and its loaded runtimes.
///
/// The activation keeps native libraries and worker processes alive until after
/// all callbacks and subscribers registered from them have been removed. Only
/// one activation may exist in a process at a time.
#[must_use = "dropping the activation clears and unloads its dynamic plugins"]
pub struct PluginHostActivation {
    active: bool,
    native: Option<NativePluginActivation>,
    #[cfg(feature = "worker-grpc")]
    worker: Option<WorkerPluginActivation>,
    claim: Option<PluginHostLease>,
}

impl PluginHostActivation {
    /// Load dynamic plugins and activate them with `config` as one transaction.
    ///
    /// The supplied base configuration may contain statically registered
    /// components. Dynamic components are appended after them in specification
    /// order. At least one dynamic plugin is required; static-only callers
    /// should use the regular plugin initialization API. The returned activation
    /// must remain alive for as long as code may invoke plugin-provided callbacks.
    pub async fn activate<I>(
        config: PluginConfig,
        dynamic_plugins: I,
    ) -> Result<(Self, ConfigReport)>
    where
        I: IntoIterator<Item = DynamicPluginActivationSpec>,
    {
        let dynamic_plugins = dynamic_plugins.into_iter().collect::<Vec<_>>();
        validate_dynamic_plugin_specs(&dynamic_plugins)?;
        Self::activate_validated(config, dynamic_plugins).await
    }

    /// Load dynamic plugins after layering `config` over discovered `plugins.toml` files.
    ///
    /// This is the harness-native entrypoint for language and FFI bindings. File
    /// discovery and merging happen once before activation, and the explicit
    /// `config` has higher precedence. Hosts such as the Relay CLI that already
    /// resolved plugin configuration should call [`Self::activate`] instead.
    pub async fn activate_with_discovered_config<I>(
        config: PluginConfig,
        dynamic_plugins: I,
    ) -> Result<(Self, ConfigReport)>
    where
        I: IntoIterator<Item = DynamicPluginActivationSpec>,
    {
        let dynamic_plugins = dynamic_plugins.into_iter().collect::<Vec<_>>();
        validate_dynamic_plugin_specs(&dynamic_plugins)?;
        let config = resolve_plugin_config(config)?;
        Self::activate_validated(config, dynamic_plugins).await
    }

    async fn activate_validated(
        config: PluginConfig,
        dynamic_plugins: Vec<DynamicPluginActivationSpec>,
    ) -> Result<(Self, ConfigReport)> {
        run_owned_plugin_mutation("dynamic plugin activation", move || async move {
            Self::activate_inner(config, dynamic_plugins).await
        })
        .await
    }

    async fn activate_inner(
        mut config: PluginConfig,
        dynamic_plugins: Vec<DynamicPluginActivationSpec>,
    ) -> Result<(Self, ConfigReport)> {
        let dynamic_plugin_count = dynamic_plugins.len();
        log::info!(
            target: "nemo_relay.plugin",
            event = "dynamic_plugin_activation_started",
            plugin_count = dynamic_plugin_count;
            "Dynamic plugin activation started"
        );
        let claim = acquire_plugin_host_lease()?;

        #[cfg(not(feature = "worker-grpc"))]
        if let Some(plugin) = dynamic_plugins
            .iter()
            .find(|plugin| plugin.kind == DynamicPluginKind::Worker)
        {
            return Err(crate::plugin::PluginError::InvalidConfig(format!(
                "worker dynamic plugin '{}' requires the 'worker-grpc' feature",
                plugin.plugin_id
            )));
        }

        // Builtin registration is cached process-wide. It must complete before
        // a dynamic plugin can claim a reserved builtin kind and permanently
        // cache a failed builtin registration attempt.
        ensure_builtin_plugins_registered()?;

        let native_specs = dynamic_plugins
            .iter()
            .filter(|plugin| plugin.kind == DynamicPluginKind::RustDynamic)
            .map(|plugin| NativePluginLoadSpec {
                plugin_id: plugin.plugin_id.clone(),
                manifest_ref: plugin.manifest_ref.clone(),
            })
            .collect::<Vec<_>>();
        let native = (!native_specs.is_empty())
            .then(|| {
                load_native_plugins(native_specs)
                    .map_err(|error| plugin_error_context("native plugin load failed", error))
            })
            .transpose()?;

        #[cfg(feature = "worker-grpc")]
        let worker = {
            let worker_specs = dynamic_plugins
                .iter()
                .filter(|plugin| plugin.kind == DynamicPluginKind::Worker)
                .map(|plugin| WorkerPluginLoadSpec {
                    plugin_id: plugin.plugin_id.clone(),
                    manifest_ref: plugin.manifest_ref.clone(),
                    environment_ref: plugin.environment_ref.clone(),
                    config: plugin.config.clone(),
                })
                .collect::<Vec<_>>();
            (!worker_specs.is_empty())
                .then(|| {
                    load_worker_plugins(worker_specs)
                        .map_err(|error| plugin_error_context("worker plugin load failed", error))
                })
                .transpose()?
        };

        config.components.extend(
            dynamic_plugins
                .into_iter()
                .map(|plugin| PluginComponentSpec {
                    kind: plugin.plugin_id,
                    enabled: true,
                    config: plugin.config,
                }),
        );
        let rollback_failures = Arc::new(Mutex::new(Vec::new()));
        let owner_id = claim.owner_id();
        let initialization = tokio::spawn(initialize_plugins_exact_for_host(
            config,
            owner_id,
            Arc::clone(&rollback_failures),
        ))
        .await
        .map_err(|error| {
            crate::plugin::PluginError::Internal(format!(
                "dynamic plugin initialization task failed: {error}"
            ))
        });
        let report = match initialization.and_then(|result| result) {
            Ok(report) => report,
            Err(error) => {
                let failures = rollback_failures
                    .lock()
                    .map(|failures| failures.clone())
                    .unwrap_or_else(|lock_error| {
                        vec![format!("rollback failure lock poisoned: {lock_error}")]
                    });
                if failures.is_empty() {
                    return Err(error);
                }
                log::error!(
                    target: "nemo_relay.plugin",
                    event = "plugin_rollback_failed",
                    plugin_count = dynamic_plugin_count,
                    failure_count = failures.len();
                    "Dynamic plugin activation rollback was incomplete"
                );
                if let Some(native) = native {
                    std::mem::forget(native);
                }
                #[cfg(feature = "worker-grpc")]
                if let Some(worker) = worker {
                    std::mem::forget(worker);
                }
                std::mem::forget(claim);
                return Err(crate::plugin::PluginError::RegistrationFailed(format!(
                    concat!(
                        "{}; activation rollback was incomplete: {}; the loaded runtimes ",
                        "were retained because callbacks may remain registered"
                    ),
                    error,
                    failures.join("; ")
                )));
            }
        };

        log::info!(
            target: "nemo_relay.plugin",
            event = "dynamic_plugin_activated",
            plugin_count = dynamic_plugin_count;
            "Dynamic plugins activated"
        );
        Ok((
            Self {
                active: true,
                native,
                #[cfg(feature = "worker-grpc")]
                worker,
                claim: Some(claim),
            },
            report,
        ))
    }

    /// Returns whether this activation handle has not begun teardown.
    ///
    /// `false` means the handle is no longer reusable. It does not guarantee
    /// that another process-wide activation can start: failed teardown may
    /// intentionally retain the loaded runtimes and activation owner.
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Clear registered callbacks before unloading libraries and workers.
    pub fn clear(mut self) -> Result<()> {
        self.clear_inner()
    }

    fn clear_inner(&mut self) -> Result<()> {
        if !self.active {
            return Ok(());
        }
        self.active = false;
        let outcome = self
            .claim
            .as_ref()
            .map(|claim| clear_plugin_configuration_for_host(claim.owner_id()))
            .unwrap_or(crate::plugin::PluginHostClearOutcome {
                result: Ok(()),
                callbacks_cleared: true,
            });
        let mut errors = outcome
            .result
            .err()
            .map(|error| vec![error.to_string()])
            .unwrap_or_default();
        if !outcome.callbacks_cleared {
            // If core could not prove callbacks were removed, intentionally
            // retain their code and owner for process lifetime rather than
            // unload a library or worker that may still be referenced.
            self.retain_loaded_runtimes();
            return Err(retained_runtime_error(errors));
        }

        let mut runtime_outcome = DynamicPluginTeardownOutcome::success();
        if let Some(native) = &mut self.native {
            runtime_outcome.merge(native.deregister_plugin_kinds_checked());
        }
        #[cfg(feature = "worker-grpc")]
        if let Some(worker) = &mut self.worker {
            runtime_outcome.merge(worker.deregister_plugin_kinds_checked());
        }

        // A worker cannot be stopped while its registry adapter might still be
        // callable. Only begin process shutdown once every kind is known to be
        // absent from the registry.
        #[cfg(feature = "worker-grpc")]
        if runtime_outcome.safe_to_unload
            && let Some(worker) = &self.worker
        {
            runtime_outcome.merge(worker.shutdown_plugins_checked());
        }
        errors.extend(runtime_outcome.errors);

        if !runtime_outcome.safe_to_unload {
            self.retain_loaded_runtimes();
            return Err(retained_runtime_error(errors));
        }

        // Callback removal and kind deregistration are now complete. Dropping
        // the activations unloads libraries and runtimes before releasing the
        // process-wide host claim.
        self.native.take();
        #[cfg(feature = "worker-grpc")]
        self.worker.take();
        self.claim.take();

        if errors.is_empty() {
            log::info!(
                target: "nemo_relay.plugin",
                event = "dynamic_plugin_cleared";
                "Dynamic plugin activation cleared"
            );
            Ok(())
        } else {
            Err(crate::plugin::PluginError::RegistrationFailed(format!(
                "dynamic plugin teardown failed: {}",
                errors.join("; ")
            )))
        }
    }

    fn retain_loaded_runtimes(&mut self) {
        if let Some(native) = self.native.take() {
            std::mem::forget(native);
        }
        #[cfg(feature = "worker-grpc")]
        if let Some(worker) = self.worker.take() {
            std::mem::forget(worker);
        }
        if let Some(claim) = self.claim.take() {
            std::mem::forget(claim);
        }
    }
}

fn validate_dynamic_plugin_specs(dynamic_plugins: &[DynamicPluginActivationSpec]) -> Result<()> {
    if dynamic_plugins.is_empty() {
        return Err(crate::plugin::PluginError::InvalidConfig(
            concat!(
                "dynamic plugin activation requires at least one dynamic plugin; ",
                "use plugin initialization for a static-only configuration"
            )
            .into(),
        ));
    }
    let mut plugin_ids = HashSet::with_capacity(dynamic_plugins.len());
    for plugin in dynamic_plugins {
        if !plugin_ids.insert(plugin.plugin_id.as_str()) {
            return Err(crate::plugin::PluginError::InvalidConfig(format!(
                "duplicate dynamic plugin id '{}'",
                plugin.plugin_id
            )));
        }
    }
    Ok(())
}

fn retained_runtime_error(errors: Vec<String>) -> crate::plugin::PluginError {
    crate::plugin::PluginError::RegistrationFailed(format!(
        concat!(
            "{}; the loaded runtimes and activation owner were retained because safe ",
            "unloading could not be proven"
        ),
        if errors.is_empty() {
            "dynamic plugin teardown was incomplete".into()
        } else {
            errors.join("; ")
        }
    ))
}

fn plugin_error_context(
    prefix: &str,
    error: crate::plugin::PluginError,
) -> crate::plugin::PluginError {
    use crate::plugin::PluginError;

    match error {
        PluginError::InvalidConfig(message) => {
            PluginError::InvalidConfig(format!("{prefix}: {message}"))
        }
        PluginError::Conflict(message) => PluginError::Conflict(format!("{prefix}: {message}")),
        PluginError::NotFound(message) => PluginError::NotFound(format!("{prefix}: {message}")),
        PluginError::Serialization(error) => {
            PluginError::Internal(format!("{prefix}: serialization error: {error}"))
        }
        PluginError::Internal(message) => PluginError::Internal(format!("{prefix}: {message}")),
        PluginError::RegistrationFailed(message) => {
            PluginError::RegistrationFailed(format!("{prefix}: {message}"))
        }
    }
}

impl Drop for PluginHostActivation {
    fn drop(&mut self) {
        if self.clear_inner().is_err() {
            log::error!(
                target: "nemo_relay.plugin",
                event = "plugin_cleanup_failed",
                cleanup = "dynamic_activation_drop";
                "Dynamic plugin activation cleanup failed during drop"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::{PLUGIN_HANDLERS, PLUGIN_MUTATION_OWNER, PluginMutationOwner};

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
}
