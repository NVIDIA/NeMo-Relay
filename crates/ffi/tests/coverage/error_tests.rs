// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Coverage tests for error in the NeMo Relay FFI crate.

use super::*;
use std::ffi::{CStr, CString};

use nemo_relay::plugin::PluginError;

#[test]
fn test_last_error_round_trip_and_clear() {
    clear_last_error();
    assert_eq!(last_error_message(), None);
    assert!(nemo_relay_last_error().is_null());

    set_last_error("ffi failure");
    assert_eq!(last_error_message(), Some("ffi failure".into()));

    let raw = nemo_relay_last_error();
    assert_eq!(
        unsafe { CStr::from_ptr(raw) }.to_str().unwrap(),
        "ffi failure"
    );

    clear_last_error();
    assert_eq!(last_error_message(), None);
    assert!(nemo_relay_last_error().is_null());
}

#[test]
fn test_set_last_error_message_handles_null_and_invalid_utf8() {
    unsafe { nemo_relay_set_last_error_message(std::ptr::null()) };
    assert_eq!(last_error_message(), Some("unknown callback error".into()));

    let invalid_utf8 = [0xffu8, 0];
    unsafe {
        nemo_relay_set_last_error_message(invalid_utf8.as_ptr() as *const c_char);
    }
    assert_eq!(
        last_error_message(),
        Some("callback error was not valid UTF-8".into())
    );

    let valid = CString::new("callback failed").unwrap();
    unsafe { nemo_relay_set_last_error_message(valid.as_ptr()) };
    assert_eq!(last_error_message(), Some("callback failed".into()));
}

#[test]
fn test_status_from_error_maps_variants_and_sets_message() {
    let cases = [
        (
            FlowError::AlreadyExists("dup".into()),
            NemoRelayStatus::AlreadyExists,
        ),
        (
            FlowError::NotFound("missing".into()),
            NemoRelayStatus::NotFound,
        ),
        (
            FlowError::InvalidArgument("bad arg".into()),
            NemoRelayStatus::InvalidArg,
        ),
        (
            FlowError::GuardrailRejected("blocked".into()),
            NemoRelayStatus::GuardrailRejected,
        ),
        (
            FlowError::Internal("boom".into()),
            NemoRelayStatus::Internal,
        ),
        (FlowError::ScopeStackEmpty, NemoRelayStatus::ScopeStackEmpty),
    ];

    for (error, expected_status) in cases {
        clear_last_error();
        let status = status_from_error(&error);
        assert_eq!(status, expected_status);
        assert_eq!(NemoRelayStatus::from(&error), expected_status);
        assert!(last_error_message().unwrap().contains(&error.to_string()));
    }
}

#[test]
fn test_status_from_plugin_error_maps_variants_and_sets_message() {
    let serialization_error = serde_json::from_str::<serde_json::Value>("{").unwrap_err();
    let cases = [
        (
            PluginError::NotFound("missing plugin".into()),
            NemoRelayStatus::NotFound,
            "missing plugin",
        ),
        (
            PluginError::InvalidConfig("bad config".into()),
            NemoRelayStatus::InvalidArg,
            "bad config",
        ),
        (
            PluginError::Conflict("duplicate plugin".into()),
            NemoRelayStatus::AlreadyExists,
            "duplicate plugin",
        ),
        (
            PluginError::Serialization(serialization_error),
            NemoRelayStatus::InvalidArg,
            "serialization error",
        ),
        (
            PluginError::Internal("plugin blew up".into()),
            NemoRelayStatus::Internal,
            "plugin blew up",
        ),
        (
            PluginError::RegistrationFailed("register failed".into()),
            NemoRelayStatus::Internal,
            "register failed",
        ),
    ];

    for (error, expected_status, message_fragment) in cases {
        clear_last_error();
        let status = status_from_plugin_error(&error);
        assert_eq!(status, expected_status);
        assert!(
            last_error_message()
                .unwrap_or_default()
                .contains(message_fragment)
        );
    }
}
