// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use nemo_relay_adaptive::{
        AcgComponentConfig, AdaptiveConfig, AdaptiveHintsComponentConfig, AdaptiveRuntime,
        BackendSpec, StateConfig, TelemetryComponentConfig, ToolParallelismComponentConfig,
    };

    let adaptive = AdaptiveConfig {
        agent_id: Some("planner".into()),
        state: Some(StateConfig {
            backend: BackendSpec::in_memory(),
        }),
        telemetry: Some(TelemetryComponentConfig {
            subscriber_name: Some("adaptive.telemetry".into()),
            learners: vec!["tool_parallelism".into()],
        }),
        tool_parallelism: Some(ToolParallelismComponentConfig::default()),
        adaptive_hints: Some(AdaptiveHintsComponentConfig {
            inject_body_path: "nvext.agent_hints".into(),
            ..AdaptiveHintsComponentConfig::default()
        }),
        acg: Some(AcgComponentConfig {
            provider: "passthrough".into(),
            ..AcgComponentConfig::default()
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
