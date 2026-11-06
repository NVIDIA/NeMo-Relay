// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust configuration for component-owned native executors.

use crate::{Json, Result};
use serde_json::Map;

/// Configuration for the executor owned by one exported native plugin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeExecutorConfig {
    /// Number of Tokio worker threads dedicated to the plugin.
    pub worker_threads: usize,
}

impl Default for NativeExecutorConfig {
    fn default() -> Self {
        Self { worker_threads: 2 }
    }
}

impl NativeExecutorConfig {
    /// Applies an optional component-local executor override.
    ///
    /// Relay passes `[plugins.dynamic.config.executor]` as the `executor`
    /// object in `plugin_config`. The only supported setting is the positive
    /// integer `worker_threads`.
    pub fn with_component_config(mut self, plugin_config: &Map<String, Json>) -> Result<Self> {
        let Some(executor) = plugin_config.get("executor") else {
            return self.validate();
        };
        let executor = executor
            .as_object()
            .ok_or_else(|| "executor configuration must be an object".to_string())?;
        if let Some(worker_threads) = executor.get("worker_threads") {
            let worker_threads = worker_threads
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| "executor.worker_threads must be a positive integer".to_string())?;
            self.worker_threads = worker_threads;
        }
        self.validate()
    }

    /// Reject a zero executor worker-thread count.
    fn validate(self) -> Result<Self> {
        if self.worker_threads == 0 {
            Err("executor.worker_threads must be greater than zero".into())
        } else {
            Ok(self)
        }
    }
}
