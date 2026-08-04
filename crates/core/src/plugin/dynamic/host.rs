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
#[cfg(test)]
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as Json};

use crate::plugin::{
    ConfigDiagnostic, ConfigReport, PluginComponentSpec, PluginConfig, PluginHostLease, Result,
    acquire_plugin_host_lease, clear_plugin_configuration_for_host,
    ensure_builtin_plugins_registered, initialize_plugins_exact_for_host, resolve_plugin_config,
    run_owned_plugin_mutation,
};

use super::{
    DynamicPluginKind, DynamicPluginLoadFailure, DynamicPluginTeardownOutcome,
    NativePluginActivation, NativePluginLoadSpec, load_native_plugins_with_resources,
};

#[cfg(feature = "worker-grpc")]
use super::{WorkerPluginActivation, WorkerPluginLoadSpec, load_worker_plugins_with_resources};

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

/// Owns resources that must remain stable for one planned dynamic plugin.
///
/// File-backed hosts use this contract to retain verified activation snapshots
/// alongside the native library or worker runtime that consumes them.
#[doc(hidden)]
pub trait DynamicPluginActivationResource: Send + Sync {
    /// Verify that the retained resource is still safe to load.
    fn verify(&self) -> Result<()>;
}

/// One dynamic plugin and its retained file-backed activation resource.
#[doc(hidden)]
pub struct PlannedDynamicPluginActivation {
    /// Dynamic plugin activation details resolved by the embedding host.
    pub spec: DynamicPluginActivationSpec,
    /// Resource retained for the complete runtime and callback lifetime.
    pub resource: Arc<dyn DynamicPluginActivationResource>,
}

/// Fully resolved static and dynamic configuration for an owned plugin host.
#[doc(hidden)]
pub struct PluginHostActivationPlan {
    /// Resolved static plugin configuration.
    pub config: PluginConfig,
    /// Enabled dynamic plugins and their retained activation resources.
    pub dynamic_plugins: Vec<PlannedDynamicPluginActivation>,
    /// Configuration diagnostics produced while resolving physical sources.
    pub diagnostics: Vec<ConfigDiagnostic>,
}

struct PreparedDynamicPluginActivation {
    spec: DynamicPluginActivationSpec,
    resource: Option<Arc<dyn DynamicPluginActivationResource>>,
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
    resource_anchors: Vec<Arc<dyn DynamicPluginActivationResource>>,
    claim: Option<PluginHostLease>,
}

struct PluginHostActivationTransaction {
    native: Option<NativePluginActivation>,
    #[cfg(feature = "worker-grpc")]
    worker: Option<WorkerPluginActivation>,
    claim: Option<PluginHostLease>,
    resource_anchors: Vec<Arc<dyn DynamicPluginActivationResource>>,
}

impl PluginHostActivationTransaction {
    fn new(claim: PluginHostLease, dynamic_plugins: &[PreparedDynamicPluginActivation]) -> Self {
        Self {
            native: None,
            #[cfg(feature = "worker-grpc")]
            worker: None,
            claim: Some(claim),
            resource_anchors: dynamic_plugins
                .iter()
                .filter_map(|plugin| plugin.resource.clone())
                .collect(),
        }
    }

    fn into_activation(mut self) -> PluginHostActivation {
        PluginHostActivation {
            active: true,
            native: self.native.take(),
            #[cfg(feature = "worker-grpc")]
            worker: self.worker.take(),
            resource_anchors: std::mem::take(&mut self.resource_anchors),
            claim: self.claim.take(),
        }
    }

    fn retain_for_process_lifetime(&mut self) {
        retain_loaded_runtimes(
            &mut self.native,
            #[cfg(feature = "worker-grpc")]
            &mut self.worker,
        );
        retain_plugin_host_claim(&mut self.claim);
        let resources = std::mem::take(&mut self.resource_anchors);
        if !resources.is_empty() {
            std::mem::forget(resources);
        }
    }
}

impl Drop for PluginHostActivationTransaction {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.retain_for_process_lifetime();
        }
    }
}

struct PluginHostClearUnwindGuard<'a> {
    activation: &'a mut PluginHostActivation,
}

impl Drop for PluginHostClearUnwindGuard<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.activation.retain_loaded_runtimes();
        }
    }
}

#[cfg(test)]
static PANIC_PLUGIN_HOST_CLEAR_AFTER_DEACTIVATION: AtomicBool = AtomicBool::new(false);

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
        Self::activate_validated(
            config,
            prepare_explicit_dynamic_plugins(dynamic_plugins),
            Vec::new(),
        )
        .await
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
        let resolved = resolve_plugin_config(config)?;
        Self::activate_validated(
            resolved.config,
            prepare_explicit_dynamic_plugins(dynamic_plugins),
            resolved.diagnostics,
        )
        .await
    }

    /// Activate a fully resolved file-backed plugin host plan.
    ///
    /// Unlike the explicit dynamic-spec entrypoints, a plan may contain no
    /// dynamic plugins so one owner can manage a static-only file-backed
    /// configuration through the same lifecycle.
    #[doc(hidden)]
    pub async fn activate_plan(plan: PluginHostActivationPlan) -> Result<(Self, ConfigReport)> {
        validate_planned_dynamic_plugins(&plan.dynamic_plugins)?;
        run_owned_plugin_mutation("file-backed plugin activation", move || async move {
            let PluginHostActivationPlan {
                config,
                dynamic_plugins,
                diagnostics,
            } = plan;
            let dynamic_plugins = dynamic_plugins
                .into_iter()
                .map(|plugin| PreparedDynamicPluginActivation {
                    spec: plugin.spec,
                    resource: Some(plugin.resource),
                })
                .collect();
            Self::activate_inner(config, dynamic_plugins, diagnostics).await
        })
        .await
    }

    async fn activate_validated(
        config: PluginConfig,
        dynamic_plugins: Vec<PreparedDynamicPluginActivation>,
        diagnostics: Vec<ConfigDiagnostic>,
    ) -> Result<(Self, ConfigReport)> {
        run_owned_plugin_mutation("dynamic plugin activation", move || async move {
            Self::activate_inner(config, dynamic_plugins, diagnostics).await
        })
        .await
    }

    async fn activate_inner(
        mut config: PluginConfig,
        dynamic_plugins: Vec<PreparedDynamicPluginActivation>,
        diagnostics: Vec<ConfigDiagnostic>,
    ) -> Result<(Self, ConfigReport)> {
        let dynamic_plugin_count = dynamic_plugins.len();
        log::info!(
            target: "nemo_relay.plugin",
            event = "dynamic_plugin_activation_started",
            plugin_count = dynamic_plugin_count;
            "Dynamic plugin activation started"
        );
        let claim = acquire_plugin_host_lease()?;
        let mut transaction = PluginHostActivationTransaction::new(claim, &dynamic_plugins);

        #[cfg(not(feature = "worker-grpc"))]
        if let Some(plugin) = dynamic_plugins
            .iter()
            .find(|plugin| plugin.spec.kind == DynamicPluginKind::Worker)
        {
            return Err(crate::plugin::PluginError::InvalidConfig(format!(
                "worker dynamic plugin '{}' requires the 'worker-grpc' feature",
                plugin.spec.plugin_id
            )));
        }

        // Builtin registration is cached process-wide. It must complete before
        // a dynamic plugin can claim a reserved builtin kind and permanently
        // cache a failed builtin registration attempt.
        ensure_builtin_plugins_registered()?;

        let native_specs = dynamic_plugins
            .iter()
            .filter(|plugin| plugin.spec.kind == DynamicPluginKind::RustDynamic)
            .map(|plugin| {
                (
                    NativePluginLoadSpec {
                        plugin_id: plugin.spec.plugin_id.clone(),
                        manifest_ref: plugin.spec.manifest_ref.clone(),
                    },
                    plugin.resource.clone(),
                )
            })
            .collect::<Vec<_>>();
        transaction.native = if native_specs.is_empty() {
            None
        } else {
            match load_native_plugins_with_resources(native_specs) {
                Ok(native) => Some(native),
                Err(failure) => {
                    return Err(finalize_load_failure(
                        "native plugin load failed",
                        &mut transaction.claim,
                        failure,
                    ));
                }
            }
        };

        #[cfg(feature = "worker-grpc")]
        {
            let worker_specs = dynamic_plugins
                .iter()
                .filter(|plugin| plugin.spec.kind == DynamicPluginKind::Worker)
                .map(|plugin| {
                    (
                        WorkerPluginLoadSpec {
                            plugin_id: plugin.spec.plugin_id.clone(),
                            manifest_ref: plugin.spec.manifest_ref.clone(),
                            environment_ref: plugin.spec.environment_ref.clone(),
                            config: plugin.spec.config.clone(),
                        },
                        plugin.resource.clone(),
                    )
                })
                .collect::<Vec<_>>();
            if worker_specs.is_empty() {
                transaction.worker = None;
            } else {
                transaction.worker = match load_worker_plugins_with_resources(worker_specs) {
                    Ok(worker) => Some(worker),
                    Err(mut failure) => {
                        if let Some(native_activation) = transaction.native.as_mut() {
                            let mut native_rollback =
                                native_activation.deregister_plugin_kinds_checked();
                            if native_rollback.safe_to_unload {
                                native_rollback.merge(native_activation.prepare_unload_checked());
                            }
                            let native_safe_to_unload = native_rollback.safe_to_unload;
                            failure.merge_rollback(native_rollback);
                            if !native_safe_to_unload
                                && let Some(native_activation) = transaction.native.take()
                            {
                                std::mem::forget(native_activation);
                            }
                        }
                        return Err(finalize_load_failure(
                            "worker plugin load failed",
                            &mut transaction.claim,
                            failure,
                        ));
                    }
                };
            }
        }

        config.components.extend(
            dynamic_plugins
                .into_iter()
                .map(|plugin| PluginComponentSpec {
                    kind: plugin.spec.plugin_id,
                    enabled: true,
                    config: plugin.spec.config,
                }),
        );
        let rollback_failures = Arc::new(Mutex::new(Vec::new()));
        let owner_id = transaction
            .claim
            .as_ref()
            .expect("active plugin host must retain its owner lease")
            .owner_id();
        let initialization = tokio::spawn(initialize_plugins_exact_for_host(
            config,
            owner_id,
            Arc::clone(&rollback_failures),
            diagnostics,
        ))
        .await;
        let report = match initialization {
            Ok(Ok(report)) => report,
            Ok(Err(error)) => {
                let failures = rollback_failures
                    .lock()
                    .map(|failures| failures.clone())
                    .unwrap_or_else(|lock_error| {
                        vec![format!("rollback failure lock poisoned: {lock_error}")]
                    });
                return Err(finalize_configuration_failure(
                    error,
                    failures,
                    &mut transaction.native,
                    #[cfg(feature = "worker-grpc")]
                    &mut transaction.worker,
                    &mut transaction.claim,
                ));
            }
            Err(join_error) => {
                retain_loaded_runtimes(
                    &mut transaction.native,
                    #[cfg(feature = "worker-grpc")]
                    &mut transaction.worker,
                );
                retain_plugin_host_claim(&mut transaction.claim);
                return Err(crate::plugin::PluginError::RegistrationFailed(format!(
                    concat!(
                        "dynamic plugin initialization task failed: {}; the loaded runtimes ",
                        "and activation owner were retained because callback rollback could not be proven"
                    ),
                    join_error
                )));
            }
        };

        log::info!(
            target: "nemo_relay.plugin",
            event = "dynamic_plugin_activated",
            plugin_count = dynamic_plugin_count;
            "Dynamic plugins activated"
        );
        Ok((transaction.into_activation(), report))
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
        let unwind_guard = PluginHostClearUnwindGuard { activation: self };
        #[cfg(test)]
        if PANIC_PLUGIN_HOST_CLEAR_AFTER_DEACTIVATION.swap(false, Ordering::SeqCst) {
            panic!("injected plugin host teardown panic after deactivation");
        }
        let outcome = unwind_guard
            .activation
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
            unwind_guard.activation.retain_loaded_runtimes();
            return Err(retained_runtime_error(errors));
        }

        let mut runtime_outcome = DynamicPluginTeardownOutcome::success();
        if let Some(native) = &mut unwind_guard.activation.native {
            runtime_outcome.merge(native.deregister_plugin_kinds_checked());
        }
        #[cfg(feature = "worker-grpc")]
        if let Some(worker) = &mut unwind_guard.activation.worker {
            runtime_outcome.merge(worker.deregister_plugin_kinds_checked());
        }

        // A worker cannot be stopped while its registry adapter might still be
        // callable. Only begin process shutdown once every kind is known to be
        // absent from the registry.
        #[cfg(feature = "worker-grpc")]
        if runtime_outcome.safe_to_unload
            && let Some(worker) = &unwind_guard.activation.worker
        {
            runtime_outcome.merge(worker.shutdown_plugins_checked());
        }
        if runtime_outcome.safe_to_unload
            && let Some(native) = &mut unwind_guard.activation.native
        {
            runtime_outcome.merge(native.prepare_unload_checked());
        }
        #[cfg(feature = "worker-grpc")]
        if runtime_outcome.safe_to_unload
            && let Some(worker) = &mut unwind_guard.activation.worker
        {
            runtime_outcome.merge(worker.prepare_unload_checked());
        }
        errors.extend(runtime_outcome.errors);

        if !runtime_outcome.safe_to_unload {
            unwind_guard.activation.retain_loaded_runtimes();
            return Err(retained_runtime_error(errors));
        }

        // Callback removal and kind deregistration are now complete. Dropping
        // the activations unloads libraries and runtimes before releasing the
        // process-wide host claim.
        unwind_guard.activation.native.take();
        #[cfg(feature = "worker-grpc")]
        unwind_guard.activation.worker.take();
        unwind_guard.activation.resource_anchors.clear();
        unwind_guard.activation.claim.take();

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
        let resources = std::mem::take(&mut self.resource_anchors);
        if !resources.is_empty() {
            std::mem::forget(resources);
        }
    }
}

fn prepare_explicit_dynamic_plugins(
    dynamic_plugins: Vec<DynamicPluginActivationSpec>,
) -> Vec<PreparedDynamicPluginActivation> {
    dynamic_plugins
        .into_iter()
        .map(|spec| PreparedDynamicPluginActivation {
            spec,
            resource: None,
        })
        .collect()
}

fn validate_planned_dynamic_plugins(
    dynamic_plugins: &[PlannedDynamicPluginActivation],
) -> Result<()> {
    validate_unique_dynamic_plugin_ids(dynamic_plugins.iter().map(|plugin| &plugin.spec))
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
    validate_unique_dynamic_plugin_ids(dynamic_plugins.iter())
}

fn validate_unique_dynamic_plugin_ids<'a>(
    dynamic_plugins: impl IntoIterator<Item = &'a DynamicPluginActivationSpec>,
) -> Result<()> {
    let mut plugin_ids = HashSet::new();
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

fn finalize_configuration_failure(
    error: crate::plugin::PluginError,
    callback_failures: Vec<String>,
    native: &mut Option<NativePluginActivation>,
    #[cfg(feature = "worker-grpc")] worker: &mut Option<WorkerPluginActivation>,
    claim: &mut Option<PluginHostLease>,
) -> crate::plugin::PluginError {
    if !callback_failures.is_empty() {
        log::error!(
            target: "nemo_relay.plugin",
            event = "plugin_rollback_failed",
            failure_count = callback_failures.len();
            "Dynamic plugin callback rollback was incomplete"
        );
        retain_loaded_runtimes(
            native,
            #[cfg(feature = "worker-grpc")]
            worker,
        );
        retain_plugin_host_claim(claim);
        return crate::plugin::PluginError::RegistrationFailed(format!(
            concat!(
                "{}; activation rollback was incomplete: {}; the loaded runtimes and ",
                "activation owner were retained because callbacks may remain registered"
            ),
            error,
            callback_failures.join("; ")
        ));
    }

    let runtime_rollback = rollback_loaded_runtimes(
        native,
        #[cfg(feature = "worker-grpc")]
        worker,
    );
    if runtime_rollback.errors.is_empty() {
        return error;
    }
    if !runtime_rollback.safe_to_unload {
        retain_plugin_host_claim(claim);
        return crate::plugin::PluginError::RegistrationFailed(format!(
            concat!(
                "{}; dynamic runtime rollback was incomplete: {}; the loaded runtimes and ",
                "activation owner were retained because safe unloading could not be proven"
            ),
            error,
            runtime_rollback.errors.join("; ")
        ));
    }
    crate::plugin::PluginError::RegistrationFailed(format!(
        "{}; dynamic runtime rollback reported: {}; all loaded runtimes were removed",
        error,
        runtime_rollback.errors.join("; ")
    ))
}

#[cfg(not(feature = "worker-grpc"))]
fn rollback_loaded_runtimes(
    native: &mut Option<NativePluginActivation>,
) -> DynamicPluginTeardownOutcome {
    let mut outcome = DynamicPluginTeardownOutcome::success();
    if let Some(native) = native.as_mut() {
        outcome.merge(native.deregister_plugin_kinds_checked());
    }
    if outcome.safe_to_unload
        && let Some(native) = native.as_mut()
    {
        outcome.merge(native.prepare_unload_checked());
    }
    finish_runtime_rollback(native, &outcome);
    outcome
}

#[cfg(feature = "worker-grpc")]
fn rollback_loaded_runtimes(
    native: &mut Option<NativePluginActivation>,
    worker: &mut Option<WorkerPluginActivation>,
) -> DynamicPluginTeardownOutcome {
    let mut outcome = DynamicPluginTeardownOutcome::success();
    if let Some(native) = native.as_mut() {
        outcome.merge(native.deregister_plugin_kinds_checked());
    }
    if let Some(worker) = worker.as_mut() {
        outcome.merge(worker.deregister_plugin_kinds_checked());
    }
    if outcome.safe_to_unload
        && let Some(worker) = worker.as_ref()
    {
        outcome.merge(worker.shutdown_plugins_checked());
    }
    if outcome.safe_to_unload
        && let Some(native) = native.as_mut()
    {
        outcome.merge(native.prepare_unload_checked());
    }
    if outcome.safe_to_unload
        && let Some(worker) = worker.as_mut()
    {
        outcome.merge(worker.prepare_unload_checked());
    }
    finish_runtime_rollback(native, &outcome);
    finish_runtime_rollback(worker, &outcome);
    outcome
}

fn finish_runtime_rollback<T>(activation: &mut Option<T>, outcome: &DynamicPluginTeardownOutcome) {
    if outcome.safe_to_unload {
        drop(activation.take());
    } else if let Some(activation) = activation.take() {
        std::mem::forget(activation);
    }
}

fn retain_loaded_runtimes(
    native: &mut Option<NativePluginActivation>,
    #[cfg(feature = "worker-grpc")] worker: &mut Option<WorkerPluginActivation>,
) {
    if let Some(native) = native.take() {
        std::mem::forget(native);
    }
    #[cfg(feature = "worker-grpc")]
    if let Some(worker) = worker.take() {
        std::mem::forget(worker);
    }
}

fn retain_plugin_host_claim(claim: &mut Option<PluginHostLease>) {
    if let Some(claim) = claim.take() {
        std::mem::forget(claim);
    }
}

fn finalize_load_failure(
    prefix: &str,
    claim: &mut Option<PluginHostLease>,
    failure: DynamicPluginLoadFailure,
) -> crate::plugin::PluginError {
    let retain_owner = !failure.safe_to_unload();
    let error = plugin_error_context(prefix, failure.into_plugin_error());
    if !retain_owner {
        return error;
    }

    retain_plugin_host_claim(claim);
    crate::plugin::PluginError::RegistrationFailed(format!(
        "{error}; the plugin host activation owner was retained because a partially loaded runtime remains reachable"
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
#[path = "../../../tests/unit/plugin_dynamic_host_tests.rs"]
mod tests;
