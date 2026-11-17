// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay_adaptive::{AdaptiveConfig, AdaptiveRuntime, ResponseCacheConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let adaptive = AdaptiveConfig {
        response_cache: Some(ResponseCacheConfig {
            namespace: "dev-harness".into(),
            ..ResponseCacheConfig::default()
        }),
        ..AdaptiveConfig::default()
    };

    let mut runtime = AdaptiveRuntime::new(adaptive).await?;
    runtime.register().await?;

    // Run instrumented application work here.

    runtime.wait_for_idle();
    runtime.shutdown().await?;
    Ok(())
}
