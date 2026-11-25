// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay::exporter::otel::{OpenTelemetrySubscriber, Result};

pub fn create_subscriber() -> Result<OpenTelemetrySubscriber> {
    use nemo_relay_observability::OpenTelemetryType;
    use nemo_relay_observability::otel::OpenTelemetryConfig;

    let config = OpenTelemetryConfig::new(
        OpenTelemetryType::OpenInference,
        "http://localhost:6006/v1/traces",
    )
    .with_service_name("agent-service")
    .with_header_env("authorization", "OTEL_AUTHORIZATION");
    nemo_relay::exporter::opentelemetry(config)
}
