// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Producer classification preserves structured evidence without changing execution.

use nemo_relay::diagnostics::{
    Classification, DiagnosticClassifier, FailureClass, PluginErrorClassifier,
};
use nemo_relay::plugin::PluginError;

fn assert_classification(actual: Classification, code: &str, class: Option<FailureClass>) {
    assert_eq!(actual.code().canonical_name(), code);
    assert_eq!(actual.failure_class(), class);
    assert!(actual.code().allows_class(actual.failure_class()));
}

#[test]
fn plugin_error_classifiers_do_not_infer_a_cause_from_flattened_strings() {
    let json_error = serde_json::from_str::<serde_json::Value>("{").unwrap_err();
    let cases = [
        (
            PluginError::InvalidConfig("timeout".into()),
            "plugin.configuration_invalid",
            FailureClass::InvalidInput,
        ),
        (
            PluginError::Conflict("not found".into()),
            "plugin.state_conflict",
            FailureClass::Conflict,
        ),
        (
            PluginError::NotFound("access denied".into()),
            "plugin.resource_not_found",
            FailureClass::NotFound,
        ),
        (
            PluginError::Serialization(json_error),
            "plugin.serialization_failed",
            FailureClass::Unknown,
        ),
        (
            PluginError::Internal("permission denied".into()),
            "plugin.internal",
            FailureClass::Internal,
        ),
        (
            PluginError::RegistrationFailed("duplicate conflict: invalid config".into()),
            "plugin.registration_failed",
            FailureClass::Unknown,
        ),
    ];
    for (error, code, class) in cases {
        assert_classification(PluginErrorClassifier.classify(&error), code, Some(class));
    }
}
