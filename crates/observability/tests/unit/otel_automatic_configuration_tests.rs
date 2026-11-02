// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn automatic_trace_configuration_uses_gen_ai_projection() {
    assert_eq!(
        OpenTelemetryConfig::from_automatic_configuration()
            .shared
            .otel_type,
        OpenTelemetryType::GenAi
    );
}
