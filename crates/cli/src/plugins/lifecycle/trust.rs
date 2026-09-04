// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CLI compatibility re-exports for core-owned artifact trust verification.

pub(super) use nemo_relay::plugin::dynamic::{
    EvaluatedDynamicPluginTrust, evaluate_dynamic_plugin_trust,
};

#[cfg(test)]
pub(super) use nemo_relay::plugin::dynamic::DynamicPluginTrustFailure;
