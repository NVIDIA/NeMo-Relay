// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn test_adaptive_config_defaults() {
    let config = AdaptiveConfig::default();
    assert_eq!(config.version, 1);
    assert!(config.telemetry.is_none());
    assert!(config.adaptive_hints.is_none());
    assert!(config.tool_parallelism.is_none());
    assert_eq!(
        config.policy.unknown_component,
        nemo_flow::plugin::UnsupportedBehavior::Warn
    );
}

#[test]
fn test_typed_section_helpers_default() {
    let adaptive_hints = AdaptiveHintsComponentConfig::default();
    assert_eq!(adaptive_hints.priority, 100);
    assert!(adaptive_hints.inject_header);

    let tool_parallelism = ToolParallelismComponentConfig::default();
    assert_eq!(tool_parallelism.mode, "observe_only");
}
