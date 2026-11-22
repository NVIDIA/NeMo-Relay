// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay::plugin::initialize;
use nemo_relay::plugin::{PluginComponentSpec, PluginConfig};
use nemo_relay::resource_metrics;

pub async fn collect_snapshot()
-> Result<resource_metrics::ResourceMetricsSnapshot, Box<dyn std::error::Error>> {
    let config = PluginConfig {
        components: vec![PluginComponentSpec::new("resource_metrics")],
        ..PluginConfig::default()
    };
    let mut activation = initialize(config, None).await?;

    let result = resource_metrics::collect().await;
    activation.close()?;
    let snapshot = result?;
    println!("{snapshot:#?}");
    Ok(snapshot)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    collect_snapshot().await?;
    Ok(())
}
