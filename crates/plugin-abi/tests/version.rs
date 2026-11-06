// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical native version syntax and compatibility policy.

use nemo_relay_plugin_abi::{
    NativeApiVersion as Api, ParseNativeApiVersionError as Error, PluginDescriptor,
    validate_descriptor,
};

/// The textual and typed SDK identifiers share the explicit contract source.
#[test]
fn sdk_identifiers_match_the_explicit_contract() {
    use nemo_relay_plugin_abi::{NATIVE_API, NATIVE_API_VERSION};
    assert_eq!(NATIVE_API.parse::<Api>().unwrap(), NATIVE_API_VERSION);
}

#[test]
fn canonical_dotted_versions_round_trip() {
    for (text, version) in [
        ("v1.0", Api::new(1, 0)),
        ("v1.1", Api::new(1, 1)),
        ("v2.0", Api::new(2, 0)),
        ("v12.34", Api::new(12, 34)),
    ] {
        assert_eq!(text.parse::<Api>().unwrap(), version);
        assert_eq!(version.to_string(), text);
    }
}

#[test]
fn reject_legacy_and_ambiguous_versions() {
    assert_eq!("1".parse::<Api>(), Err(Error::Legacy));
    for text in [
        "v1_0",
        "1.0",
        "v0.1",
        "v01.0",
        "v1.00",
        "v1.0.0",
        "v1.0-alpha",
        " v1.0",
        "v1.0 ",
        "v1.-1",
        "v1.+1",
        "v4294967296.0",
    ] {
        assert_eq!(text.parse::<Api>(), Err(Error::Invalid), "{text}");
    }
}

#[test]
fn sdk_minor_is_the_minimum_host_version() {
    let cases = [
        (Api::new(1, 0), Api::new(1, 0), true),
        (Api::new(1, 1), Api::new(1, 0), true),
        (Api::new(1, 1), Api::new(1, 1), true),
        (Api::new(1, 0), Api::new(1, 1), false),
        (Api::new(2, 0), Api::new(1, 0), false),
        (Api::new(1, 9), Api::new(2, 0), false),
    ];
    for (host, plugin, expected) in cases {
        assert_eq!(
            host.supports(plugin),
            expected,
            "host {host}, plugin {plugin}"
        );
    }
}

#[test]
fn binary_and_manifest_must_agree_before_factory_initialization() {
    let descriptor = PluginDescriptor {
        native_api: Api::new(1, 0),
        plugin_kind: "example".into(),
    };
    assert!(validate_descriptor(&descriptor, Api::new(1, 0), "example", Api::new(1, 1)).is_ok());
    assert!(
        validate_descriptor(&descriptor, Api::new(1, 1), "example", Api::new(1, 1))
            .unwrap_err()
            .contains("disagrees")
    );
    assert!(
        validate_descriptor(&descriptor, Api::new(1, 0), "other", Api::new(1, 1))
            .unwrap_err()
            .contains("kind")
    );
    let future = PluginDescriptor {
        native_api: Api::new(1, 1),
        plugin_kind: "example".into(),
    };
    assert!(
        validate_descriptor(&future, Api::new(1, 1), "example", Api::new(1, 0))
            .unwrap_err()
            .contains("requires")
    );
}
