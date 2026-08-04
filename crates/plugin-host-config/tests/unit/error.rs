// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

use nemo_relay::plugin::PluginError;

use super::{PluginHostConfigError, sanitize_parser_reason};

#[test]
fn parser_reason_preserves_schema_identifiers_but_redacts_values() {
    let reason =
        "unknown field `enabled`, expected `manifest` or `config`; invalid value \"secret\"";

    let sanitized = sanitize_parser_reason(reason);

    assert!(sanitized.contains("`enabled`"));
    assert!(sanitized.contains("`manifest`"));
    assert!(sanitized.contains("`config`"));
    assert!(!sanitized.contains("secret"));
    assert!(sanitized.contains("\"<redacted>\""));
}

#[test]
fn parser_reason_redacts_escaped_and_single_quoted_values_on_the_first_line() {
    let sanitized = sanitize_parser_reason(
        "invalid value 'first\\'secret' for `field`; second line is ignored\ncredential-leak",
    );

    assert!(sanitized.contains("'<redacted>'"));
    assert!(sanitized.contains("`field`"));
    assert!(!sanitized.contains("secret"));
    assert!(!sanitized.contains("credential-leak"));
}

#[test]
fn every_host_configuration_error_maps_to_the_public_plugin_taxonomy() {
    assert!(matches!(
        PluginHostConfigError::InvalidConfig("bad config".into()).into_plugin_error(),
        PluginError::InvalidConfig(message) if message == "bad config"
    ));
    assert!(matches!(
        PluginHostConfigError::NotFound {
            path: PathBuf::from("missing/manifest.toml"),
            message: "absent".into(),
        }
        .into_plugin_error(),
        PluginError::NotFound(message)
            if message.contains("missing/manifest.toml") && message.contains("absent")
    ));
    assert!(matches!(
        PluginHostConfigError::io(
            "read fixture",
            "fixture.toml",
            std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied"),
        )
        .into_plugin_error(),
        PluginError::InvalidConfig(message)
            if message.contains("read fixture") && message.contains("denied")
    ));
    assert!(matches!(
        PluginHostConfigError::Relay(PluginError::Internal("relay failure".into()))
            .into_plugin_error(),
        PluginError::Internal(message) if message == "relay failure"
    ));

    let json_error = serde_json::from_str::<serde_json::Value>("{").unwrap_err();
    assert!(matches!(
        PluginHostConfigError::Json(json_error).into_plugin_error(),
        PluginError::Serialization(_)
    ));
}
