// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay::exporter::atif::{AtifExporter, AtifTrajectory};
use nemo_relay::subscriber::{deregister_subscriber, register_subscriber};
use nemo_relay_observability::atif::AtifAgentInfo;

pub fn run_example()
-> Result<nemo_relay::exporter::atif::AtifTrajectory, Box<dyn std::error::Error>> {
    let exporter = nemo_relay::exporter::atif(
        "session-1".to_string(),
        AtifAgentInfo {
            name: "agent".to_string(),
            version: "1.0.0".to_string(),
            model_name: Some("demo-model".to_string()),
            tool_definitions: None,
            extra: None,
        },
    );
    register_subscriber("atif-exporter", exporter.subscriber())?;

    // Run instrumented application work here.

    complete_export(&exporter, exporter.export())
}

fn complete_export(
    exporter: &AtifExporter,
    trajectory_result: Result<AtifTrajectory, nemo_relay::FlowError>,
) -> Result<AtifTrajectory, Box<dyn std::error::Error>> {
    let deregistration_result = deregister_subscriber("atif-exporter");
    exporter.clear();
    let trajectory = trajectory_result?;
    deregistration_result?;
    let trajectory_json = serde_json::to_string_pretty(&trajectory)?;
    println!("{trajectory_json}");
    Ok(trajectory)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    run_example()?;
    Ok(())
}
