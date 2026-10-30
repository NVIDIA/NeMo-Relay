// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CLI compatibility re-exports for core-owned dynamic-plugin policy.

pub(crate) use {
    nemo_relay_runtime::plugin::dynamic::DynamicPluginHostPolicy,
    nemo_relay_runtime::plugin::dynamic::EvaluatedDynamicPluginHostPolicy,
    nemo_relay_runtime::plugin::dynamic::FileDynamicPluginHostPolicy,
    nemo_relay_runtime::plugin::dynamic::evaluate_dynamic_plugin_host_policy,
};

#[cfg(test)]
pub(crate) use {
    nemo_relay_runtime::plugin::dynamic::DynamicPluginHostPolicyEffect,
    nemo_relay_runtime::plugin::dynamic::DynamicPluginHostPolicyFailure,
    nemo_relay_runtime::plugin::dynamic::DynamicPluginHostPolicyRule,
};

/// Applies the core production defaults at the CLI server boundary.
pub(crate) fn apply_secure_runtime_defaults(policy: &mut DynamicPluginHostPolicy) {
    policy.apply_secure_defaults();
}
