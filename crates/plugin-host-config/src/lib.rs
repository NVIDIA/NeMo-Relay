// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared resolution and lifecycle preparation for file-backed dynamic plugin hosts.
//!
//! This crate contains the lifecycle-reconciling host side of Relay's dynamic-plugin control
//! plane. It resolves the same physical `plugins.toml` layers as the CLI, durably refreshes their
//! sibling lifecycle registries, verifies policy and trust, and produces an owned activation
//! plan. It intentionally does not install plugins, create environments, or change desired
//! enablement.

mod activation;
mod environment;
mod error;
mod io;
mod lifecycle;
mod policy;
mod resolver;
mod snapshot;
mod state;
mod trust;

pub use activation::{PluginFileActivation, initialize_from_plugins_toml};
#[doc(hidden)]
pub use environment::{
    ENVIRONMENT_ATTESTATION_FILE, MANAGED_ENVIRONMENTS_DIR, environment_state,
    read_environment_attestation, validate_environment_state, validate_python_entrypoint_artifact,
    verify_environment_attestation,
};
pub use error::{PluginHostConfigError, Result};
pub use lifecycle::{
    ReconciledDynamicPlugin, ReconciledPluginLifecycle, prepare_plugin_host_activation,
    reconcile_plugin_lifecycle,
};
pub use policy::{
    DynamicPluginHostPolicy, DynamicPluginHostPolicyEffect, DynamicPluginHostPolicyFailure,
    DynamicPluginHostPolicyRule, EvaluatedDynamicPluginHostPolicy, FileDynamicPluginHostPolicy,
    evaluate_dynamic_plugin_host_policy,
};
pub use resolver::{
    PluginFileResolveOptions, ResolvedDynamicPluginConfig, ResolvedPluginFileConfiguration,
    resolve_plugin_files, resolve_plugin_files_from_paths,
};
pub use snapshot::DynamicPluginActivationSnapshot;
#[doc(hidden)]
pub use state::{
    DynamicPluginLifecycleState, LifecycleStateLock, lock_lifecycle_state, pin_plugin_config_path,
    read_lifecycle_registry, read_lifecycle_state, read_locked_lifecycle_registry,
    read_locked_lifecycle_state, save_locked_lifecycle_registry, save_locked_lifecycle_state,
    sibling_lifecycle_state_path,
};
#[doc(hidden)]
pub use trust::{
    DynamicPluginTrustFailure, DynamicPluginTrustFailureDisplay, EvaluatedDynamicPluginTrust,
    evaluate_dynamic_plugin_trust,
};
