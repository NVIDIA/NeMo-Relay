// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Producer classification preserves structured evidence without changing execution.

use nemo_relay::diagnostics::{
    BuiltinDiagnosticCode, Classification, CommonPluginFailureCode, DiagnosticClassifier,
    FailureClass, FailureCode, PluginErrorClassifier, PluginFailureCode,
};
use nemo_relay::plugin::{PluginError, load_plugin_config_files};

fn assert_classification(actual: Classification, code: &str, class: Option<FailureClass>) {
    assert_eq!(actual.code(), code);
    assert_eq!(actual.failure_class(), class.map(FailureClass::as_str));
    assert!(
        BuiltinDiagnosticCode::from_canonical_name(actual.code())
            .unwrap()
            .allows_class(class)
    );
}

#[test]
fn plugin_error_classifiers_do_not_infer_a_cause_from_flattened_strings() {
    let json_error = serde_json::from_str::<serde_json::Value>("{").unwrap_err();
    let cases = [
        (
            PluginError::InvalidConfig("timeout".into()),
            "plugin.configuration_invalid",
            FailureClass::Unknown,
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

#[test]
fn flattened_configuration_read_and_validation_errors_have_unknown_cause() {
    let directory = tempfile::tempdir().unwrap();
    let unreadable = directory.path().join("directory.toml");
    std::fs::create_dir(&unreadable).unwrap();
    let invalid_utf8 = directory.path().join("invalid-utf8.toml");
    std::fs::write(&invalid_utf8, [0xff]).unwrap();
    let malformed = directory.path().join("malformed.toml");
    std::fs::write(&malformed, "version = [").unwrap();
    let invalid_version = directory.path().join("invalid-version.toml");
    std::fs::write(&invalid_version, "version = \"one\"").unwrap();

    for path in [unreadable, invalid_utf8, malformed, invalid_version] {
        let error = load_plugin_config_files([path.clone()]).unwrap_err();
        assert!(
            matches!(error, PluginError::InvalidConfig(_)),
            "{path:?}: {error}"
        );
        assert_classification(
            PluginErrorClassifier.classify(&error),
            "plugin.configuration_invalid",
            Some(FailureClass::Unknown),
        );
    }
    assert!(
        load_plugin_config_files([directory.path().join("missing.toml")])
            .unwrap()
            .is_none()
    );
}

#[test]
fn structured_configuration_validation_can_still_report_invalid_input() {
    let classification = Classification::failure(
        FailureCode::Plugin(PluginFailureCode::Common(
            CommonPluginFailureCode::ConfigurationInvalid,
        )),
        FailureClass::InvalidInput,
    )
    .unwrap();
    assert_classification(
        classification,
        "plugin.configuration_invalid",
        Some(FailureClass::InvalidInput),
    );
}
