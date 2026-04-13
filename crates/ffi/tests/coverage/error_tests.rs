// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::ffi::{CStr, CString};

use nemo_flow::plugin::PluginError;

#[test]
fn test_last_error_round_trip_and_clear() {
    clear_last_error();
    assert_eq!(last_error_message(), None);
    assert!(nemo_flow_last_error().is_null());

    set_last_error("ffi failure");
    assert_eq!(last_error_message(), Some("ffi failure".into()));

    let raw = nemo_flow_last_error();
    assert_eq!(
        unsafe { CStr::from_ptr(raw) }.to_str().unwrap(),
        "ffi failure"
    );

    clear_last_error();
    assert_eq!(last_error_message(), None);
    assert!(nemo_flow_last_error().is_null());
}

#[test]
fn test_set_last_error_message_handles_null_and_invalid_utf8() {
    unsafe { nemo_flow_set_last_error_message(std::ptr::null()) };
    assert_eq!(last_error_message(), Some("unknown callback error".into()));

    let invalid_utf8 = [0xffu8, 0];
    unsafe {
        nemo_flow_set_last_error_message(invalid_utf8.as_ptr() as *const c_char);
    }
    assert_eq!(
        last_error_message(),
        Some("callback error was not valid UTF-8".into())
    );

    let valid = CString::new("callback failed").unwrap();
    unsafe { nemo_flow_set_last_error_message(valid.as_ptr()) };
    assert_eq!(last_error_message(), Some("callback failed".into()));
}

#[test]
fn test_status_from_error_maps_variants_and_sets_message() {
    let cases = [
        (
            FlowError::AlreadyExists("dup".into()),
            NemoFlowStatus::AlreadyExists,
        ),
        (
            FlowError::NotFound("missing".into()),
            NemoFlowStatus::NotFound,
        ),
        (
            FlowError::InvalidArgument("bad arg".into()),
            NemoFlowStatus::InvalidArg,
        ),
        (
            FlowError::GuardrailRejected("blocked".into()),
            NemoFlowStatus::GuardrailRejected,
        ),
        (FlowError::Internal("boom".into()), NemoFlowStatus::Internal),
        (FlowError::ScopeStackEmpty, NemoFlowStatus::ScopeStackEmpty),
    ];

    for (error, expected_status) in cases {
        clear_last_error();
        let status = status_from_error(&error);
        assert_eq!(status, expected_status);
        assert_eq!(NemoFlowStatus::from(&error), expected_status);
        assert!(last_error_message().unwrap().contains(&error.to_string()));
    }
}

#[test]
fn test_status_from_plugin_error_maps_variants_and_sets_message() {
    let serialization_error = serde_json::from_str::<serde_json::Value>("{").unwrap_err();
    let cases = [
        (
            PluginError::NotFound("missing plugin".into()),
            NemoFlowStatus::NotFound,
            "missing plugin",
        ),
        (
            PluginError::InvalidConfig("bad config".into()),
            NemoFlowStatus::InvalidArg,
            "bad config",
        ),
        (
            PluginError::Serialization(serialization_error),
            NemoFlowStatus::InvalidArg,
            "serialization error",
        ),
        (
            PluginError::Internal("plugin blew up".into()),
            NemoFlowStatus::Internal,
            "plugin blew up",
        ),
        (
            PluginError::RegistrationFailed("register failed".into()),
            NemoFlowStatus::Internal,
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
