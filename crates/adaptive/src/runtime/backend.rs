// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use crate::config::BackendSpec;
use crate::error::{AdaptiveError, Result};
#[cfg(feature = "redis-backend")]
use crate::redis::RedisBackend;
use crate::storage::memory::InMemoryBackend;
use crate::storage::traits::StorageBackendDyn;

pub async fn build_backend(
    backend: &BackendSpec,
) -> Result<Arc<dyn StorageBackendDyn + Send + Sync>> {
    match backend.kind.as_str() {
        "in_memory" => {
            log::info!(
                target: "nemo_relay.plugin",
                event = "plugin_resource_validation_completed",
                plugin_kind = "adaptive",
                resource_count = 0;
                "Plugin resource validation completed"
            );
            Ok(Arc::new(InMemoryBackend::new()))
        }
        #[cfg(feature = "redis-backend")]
        "redis" => {
            let url = backend
                .config
                .get("url")
                .and_then(|value| value.as_str())
                .ok_or_else(|| AdaptiveError::InvalidConfig("redis backend missing url".into()))?;
            let key_prefix = backend
                .config
                .get("key_prefix")
                .and_then(|value| value.as_str())
                .unwrap_or("nemo_relay:");
            Ok(Arc::new(RedisBackend::new(url, key_prefix).await.map_err(
                |error| AdaptiveError::Storage(error.to_string()),
            )?))
        }
        #[cfg(not(feature = "redis-backend"))]
        "redis" => Err(AdaptiveError::InvalidConfig(
            "redis backend is not enabled in this build".into(),
        )),
        other => Err(AdaptiveError::InvalidConfig(format!(
            "unsupported backend '{other}'"
        ))),
    }
}
