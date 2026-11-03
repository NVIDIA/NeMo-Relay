// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fresh and managed system resource metric observation.

use nemo_relay_types::api::resource_metrics::ResourceMetricsSnapshot;

use crate::error::Result;
pub use nemo_relay_resource_metrics::config::{
    ResourceMetricsConfig, ResourceMetricsCpuConfig, ResourceMetricsCpuUnits,
    ResourceMetricsDiskConfig, ResourceMetricsDiskUnits, ResourceMetricsGpuConfig,
    ResourceMetricsGpuUnits, ResourceMetricsMeasurementScope, ResourceMetricsMemoryConfig,
    ResourceMetricsMemoryUnits, ResourceMetricsNetworkConfig, ResourceMetricsNetworkTrafficUnits,
    ResourceMetricsNetworkUnits, ResourceMetricsPollingConfig, ResourceMetricsProcessConfig,
    ResourceMetricsUnits,
};
#[doc(hidden)]
pub use nemo_relay_resource_metrics::manager::CliResourceMetricsLaunchGuard;
pub use nemo_relay_resource_metrics::manager::ResourceMetricsTargetGuard;

/// Acquire a fresh snapshot without requiring managed polling.
///
/// Await this operation on a Tokio runtime. OS and vendor queries run on a
/// blocking worker. Dropping the future stops waiting; a query already running
/// may still finish and update the on-demand sampling baseline.
pub async fn collect() -> Result<ResourceMetricsSnapshot> {
    crate::composition::ensure_runtime_services();
    nemo_relay_resource_metrics::manager::collect_on_demand().await
}

/// Prepare a CLI launch before activating a process-tree resource metrics component.
///
/// The guard keeps process-tree polling paused until the launched child becomes
/// the collection target. The CLI holds it until gateway shutdown.
#[doc(hidden)]
pub fn prepare_cli_resource_metrics_process_tree() -> Result<CliResourceMetricsLaunchGuard> {
    crate::composition::ensure_runtime_services();
    nemo_relay_resource_metrics::manager::prepare_cli_owned_process_tree()
}

/// Direct the active resource metrics plugin to a Relay-owned child process tree.
///
/// This lifecycle hook is used by the CLI while supervising a launched agent.
/// The returned guard restores application-process collection when supervision
/// ends. Plugin configuration remains exclusively on the plugin-host API.
#[doc(hidden)]
#[cfg(not(windows))]
pub fn target_resource_metrics_to_owned_process_tree(
    process_id: u32,
) -> Result<Option<ResourceMetricsTargetGuard>> {
    crate::composition::ensure_runtime_services();
    nemo_relay_resource_metrics::manager::target_owned_process_tree(process_id)
}

/// Direct the active resource metrics plugin to a Relay-owned Windows Job Object.
///
/// # Safety
/// `job_handle` must be a live Job Object handle owned by this process. Relay
/// duplicates the handle before returning and never closes the caller's handle.
#[doc(hidden)]
#[cfg(windows)]
pub unsafe fn target_resource_metrics_to_owned_process_tree(
    process_id: u32,
    job_handle: isize,
) -> Result<Option<ResourceMetricsTargetGuard>> {
    nemo_relay_resource_metrics::manager::target_owned_process_tree_with_job_handle(
        process_id, job_handle,
    )
}
