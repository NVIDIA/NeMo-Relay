// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Coverage tests for convert in the NeMo Relay FFI crate.

use super::*;
use std::ffi::CString;

use super::error::{clear_last_error, last_error_message};
use serde_json::json;

#[test]
fn test_c_str_json_conversions_and_errors() {
    clear_last_error();
    assert_eq!(c_str_to_json(std::ptr::null()), Some(Json::Null));
    assert_eq!(c_str_to_opt_json(std::ptr::null()), Some(None));

    let valid = CString::new(r#"{"value":1}"#).unwrap();
    assert_eq!(c_str_to_json(valid.as_ptr()), Some(json!({"value": 1})));
    assert_eq!(
        c_str_to_opt_json(valid.as_ptr()),
        Some(Some(json!({"value": 1})))
    );

    let invalid_json = CString::new("{").unwrap();
    assert_eq!(c_str_to_json(invalid_json.as_ptr()), None);
    assert!(last_error_message().unwrap().contains("invalid JSON"));

    let invalid_utf8 = [0xffu8, 0];
    assert_eq!(c_str_to_json(invalid_utf8.as_ptr() as *const c_char), None);
    assert!(last_error_message().unwrap().contains("invalid UTF-8"));
}

#[test]
fn test_string_to_c_string_round_trip_and_validation() {
    let json_ptr = json_to_c_string(&json!({"ok": true}));
    let json_text = unsafe { CStr::from_ptr(json_ptr) }
        .to_str()
        .unwrap()
        .to_string();
    assert_eq!(
        serde_json::from_str::<Json>(&json_text).unwrap(),
        json!({"ok": true})
    );
    unsafe { nemo_relay_string_free(json_ptr) };

    let string_ptr = str_to_c_string("ffi-string");
    assert_eq!(
        unsafe { CStr::from_ptr(string_ptr) }.to_str().unwrap(),
        "ffi-string"
    );
    unsafe { nemo_relay_string_free(string_ptr) };

    clear_last_error();
    assert_eq!(
        c_str_to_string(CString::new("plain-text").unwrap().as_ptr()).unwrap(),
        "plain-text"
    );
    assert_eq!(
        c_str_to_string(std::ptr::null()),
        Err(NemoRelayStatus::NullPointer)
    );
    assert_eq!(last_error_message(), Some("null string pointer".into()));

    let invalid_utf8 = [0xffu8, 0];
    assert_eq!(
        c_str_to_string(invalid_utf8.as_ptr() as *const c_char),
        Err(NemoRelayStatus::InvalidUtf8)
    );
    assert!(last_error_message().unwrap().contains("invalid UTF-8"));

    unsafe { nemo_relay_string_free(std::ptr::null_mut()) };
}
