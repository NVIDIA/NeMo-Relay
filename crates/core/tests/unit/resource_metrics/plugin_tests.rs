// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn resource_metrics_plugin_rejects_multiple_components_and_reports_invalid_configuration() {
    let plugin = ResourceMetricsPlugin;
    assert!(!plugin.allows_multiple_components());
    for config in [
        serde_json::json!({"polling": {"interval_millis": 0}}),
        serde_json::json!({"network": {"interfaces": [""]}}),
        serde_json::json!({"cpu": {"enabled": "yes"}}),
    ] {
        let diagnostics = plugin.validate(config.as_object().unwrap());
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].level, DiagnosticLevel::Error);
        assert_eq!(diagnostics[0].code, "resource_metrics.invalid_config");
        assert_eq!(
            diagnostics[0].component.as_deref(),
            Some("resource_metrics")
        );
        assert!(
            diagnostics[0]
                .message
                .starts_with("invalid resource metrics config:")
        );
    }
    assert!(plugin.validate(&Map::new()).is_empty());
}
