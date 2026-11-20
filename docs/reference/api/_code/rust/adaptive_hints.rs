// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use nemo_relay_adaptive::{
        AdaptiveConfig, AdaptiveHintsComponentConfig, AdaptiveRuntime, BackendSpec, StateConfig,
        TelemetryComponentConfig, set_latency_sensitivity,
    };

    let adaptive = AdaptiveConfig {
        agent_id: Some("planner".into()),
        state: Some(StateConfig {
            backend: BackendSpec::in_memory(),
        }),
        telemetry: Some(TelemetryComponentConfig {
            learners: vec!["tool_parallelism".into()],
            ..TelemetryComponentConfig::default()
        }),
        adaptive_hints: Some(AdaptiveHintsComponentConfig {
            inject_body_path: "nvext.agent_hints".into(),
            ..AdaptiveHintsComponentConfig::default()
        }),
        ..AdaptiveConfig::default()
    };

    let mut runtime = AdaptiveRuntime::new(adaptive).await?;
    runtime.register().await?;

    // Run instrumented application work here.
    set_latency_sensitivity(8)?;

    runtime.shutdown().await?;
    Ok(())
}
