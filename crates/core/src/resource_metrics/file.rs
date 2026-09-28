// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io::Write;

use nemo_relay_types::api::resource_metrics::ResourceMetricsSnapshot;

use crate::error::{FlowError, Result};
use crate::logging::rotation::SizeRotatingFileWriter;

use super::config::ResourceMetricsFileConfig;

pub(crate) struct ResourceMetricsFile {
    writer: SizeRotatingFileWriter,
}

impl ResourceMetricsFile {
    pub(crate) fn open(config: &ResourceMetricsFileConfig) -> Result<Self> {
        SizeRotatingFileWriter::new(
            config.path.clone(),
            config.max_file_size_bytes,
            config.retained_files,
        )
        .map(|writer| Self { writer })
        .map_err(|error| {
            FlowError::Internal(format!(
                "failed to open resource metrics file {}: {error}",
                config.path.display()
            ))
        })
    }

    pub(crate) fn append(&mut self, snapshot: &ResourceMetricsSnapshot) -> Result<()> {
        let mut encoded = serde_json::to_vec(snapshot).map_err(|error| {
            FlowError::Internal(format!(
                "failed to serialize resource metrics snapshot: {error}"
            ))
        })?;
        encoded.push(b'\n');
        self.writer.write_all(&encoded).map_err(|error| {
            FlowError::Internal(format!(
                "failed to write resource metrics snapshot: {error}"
            ))
        })?;
        self.writer.flush().map_err(|error| {
            FlowError::Internal(format!(
                "failed to flush resource metrics snapshot: {error}"
            ))
        })
    }
}

impl Drop for ResourceMetricsFile {
    fn drop(&mut self) {
        if let Err(error) = self.writer.flush() {
            log::warn!(
                target: "nemo_relay.resource_metrics",
                event = "resource_metrics_file_flush_failed",
                error_kind = "io";
                "Resource metrics file flush failed: {error}"
            );
        }
    }
}
