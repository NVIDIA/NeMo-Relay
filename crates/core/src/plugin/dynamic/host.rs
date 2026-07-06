// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Owned activation lifecycle for dynamically loaded plugin components.

use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as Json};

use crate::plugin::{
    ConfigReport, PluginComponentSpec, PluginConfig, PluginHostLease, Result,
    acquire_plugin_host_lease, clear_plugin_configuration_for_host,
    initialize_plugins_exact_for_host, run_owned_plugin_mutation,
};

use super::{DynamicPluginKind, NativePluginActivation, NativePluginLoadSpec, load_native_plugins};

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
    /// Dynamic components are appended to the supplied base configuration in
    /// specification order. The returned activation must remain alive for as
    /// long as code may invoke plugin-provided callbacks.
    pub async fn activate<I>(
        config: PluginConfig,
        dynamic_plugins: I,
    ) -> Result<(Self, ConfigReport)>
    where
        I: IntoIterator<Item = DynamicPluginActivationSpec>,
    {
        let dynamic_plugins = dynamic_plugins.into_iter().collect::<Vec<_>>();
        run_owned_plugin_mutation("dynamic plugin activation", move || async move {
            Self::activate_inner(config, dynamic_plugins).await
        })
        .await
    }

    async fn activate_inner(
        mut config: PluginConfig,
        dynamic_plugins: Vec<DynamicPluginActivationSpec>,
    ) -> Result<(Self, ConfigReport)> {
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

    /// Returns whether this activation still owns an active plugin host.
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
        if outcome.callbacks_cleared {
            self.native.take();
            #[cfg(feature = "worker-grpc")]
            self.worker.take();
            self.claim.take();
        } else {
            // If core could not prove callbacks were removed, intentionally
            // retain their code and owner for process lifetime rather than
            // unload a library or worker that may still be referenced.
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
        if outcome.callbacks_cleared {
            outcome.result
        } else {
            Err(crate::plugin::PluginError::RegistrationFailed(format!(
                "{}; the loaded runtimes were retained because callbacks may remain registered",
                outcome
                    .result
                    .err()
                    .map(|error| error.to_string())
                    .unwrap_or_else(|| "plugin teardown was incomplete".into())
            )))
        }
    }
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
        let _ = self.clear_inner();
    }
}
