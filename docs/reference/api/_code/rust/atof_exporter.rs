// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay_observability::atof::{
    AtofEndpointTransport, AtofExporterConfig, AtofStreamSinkConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = AtofExporterConfig::new().with_stream_sink(AtofStreamSinkConfig::new(
        "http://localhost:8080/events",
        AtofEndpointTransport::HttpPost,
    ));
    let exporter = nemo_relay::exporter::atof(config)?;
    exporter.register("atof-exporter")?;

    // Run instrumented application work here.

    exporter.force_flush()?;
    let _ = exporter.deregister("atof-exporter")?;
    exporter.shutdown()?;
    Ok(())
}
