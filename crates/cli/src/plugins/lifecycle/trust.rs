// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Trust verification is shared with embedded file-backed hosts. Keep this module as a compatibility
// adapter until the stacked CLI runtime-owner migration removes the private lifecycle facade.
#[cfg(test)]
pub(super) use nemo_relay_plugin_host_config::DynamicPluginTrustFailure;
pub(super) use nemo_relay_plugin_host_config::{
    EvaluatedDynamicPluginTrust, evaluate_dynamic_plugin_trust,
};
