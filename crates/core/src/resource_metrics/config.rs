// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::error::{FlowError, Result};

/// Optional managed polling configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ResourceMetricsPollingConfig {
    /// Whether Relay should collect snapshots periodically.
    pub enabled: bool,
    /// Delay between polling attempts.
    pub interval_millis: u64,
    /// Maximum number of successful polling snapshots retained in memory.
    pub retained_snapshots: usize,
}

impl Default for ResourceMetricsPollingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            interval_millis: 5_000,
            retained_snapshots: 120,
        }
    }
}

/// Optional rotating structured-file output for successful polling snapshots.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ResourceMetricsFileConfig {
    /// Whether successful polling snapshots should be appended to a JSONL file.
    pub enabled: bool,
    /// Active JSONL file path.
    pub path: PathBuf,
    /// Maximum active-file size before rotation.
    pub max_file_size_bytes: u64,
    /// Maximum number of rotated files retained alongside the active file.
    pub retained_files: usize,
}

impl Default for ResourceMetricsFileConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            path: PathBuf::from("resource-metrics.jsonl"),
            max_file_size_bytes: 10 * 1024 * 1024,
            retained_files: 5,
        }
    }
}

/// Process-wide resource metrics subsystem configuration.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ResourceMetricsConfig {
    /// Managed polling behavior.
    pub polling: ResourceMetricsPollingConfig,
    /// Structured-file output for successful polling snapshots.
    pub file: ResourceMetricsFileConfig,
}

impl ResourceMetricsConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.polling.enabled && self.polling.interval_millis == 0 {
            return Err(FlowError::InvalidArgument(
                "resource_metrics.polling.interval_millis must be greater than zero".into(),
            ));
        }
        if self.polling.enabled && self.polling.retained_snapshots == 0 {
            return Err(FlowError::InvalidArgument(
                "resource_metrics.polling.retained_snapshots must be greater than zero".into(),
            ));
        }
        if self.file.enabled && !self.polling.enabled {
            return Err(FlowError::InvalidArgument(
                "resource_metrics.file requires resource_metrics.polling.enabled = true".into(),
            ));
        }
        if self.file.enabled && self.file.path.as_os_str().is_empty() {
            return Err(FlowError::InvalidArgument(
                "resource_metrics.file.path must not be empty".into(),
            ));
        }
        if self.file.enabled && self.file.max_file_size_bytes == 0 {
            return Err(FlowError::InvalidArgument(
                "resource_metrics.file.max_file_size_bytes must be greater than zero".into(),
            ));
        }
        if self.file.enabled && self.file.retained_files == 0 {
            return Err(FlowError::InvalidArgument(
                "resource_metrics.file.retained_files must be greater than zero".into(),
            ));
        }
        Ok(())
    }
}
