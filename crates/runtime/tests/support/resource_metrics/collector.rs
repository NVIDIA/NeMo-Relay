// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub(crate) use nemo_relay_resource_metrics::collector::*;

#[cfg(test)]
pub(crate) fn collect(
    target: &CollectionTarget,
    config: &crate::api::resource_metrics::ResourceMetricsConfig,
) -> CollectedSnapshot {
    collect_with_state(target, config, &mut SamplingState::default())
}
