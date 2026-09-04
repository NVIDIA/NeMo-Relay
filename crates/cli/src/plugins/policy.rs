// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CLI compatibility re-exports for core-owned dynamic-plugin policy.

pub(crate) use nemo_relay::plugin::dynamic::{
    DynamicPluginHostPolicy, EvaluatedDynamicPluginHostPolicy, FileDynamicPluginHostPolicy,
    evaluate_dynamic_plugin_host_policy,
};

#[cfg(test)]
pub(crate) use nemo_relay::plugin::dynamic::{
    DynamicPluginHostPolicyEffect, DynamicPluginHostPolicyFailure, DynamicPluginHostPolicyRule,
};

/// Applies the core production defaults at the CLI server boundary.
pub(crate) fn apply_secure_runtime_defaults(policy: &mut DynamicPluginHostPolicy) {
    policy.apply_secure_defaults();
}
