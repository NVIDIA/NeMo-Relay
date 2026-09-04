// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared owned plugin-host helpers for FFI tests.

use super::*;

static TEST_PLUGIN_HOST: Mutex<Option<usize>> = Mutex::new(None);

pub(super) unsafe fn validate_test_plugin_config(
    config_json: *const c_char,
    out_report_json: *mut *mut c_char,
) -> NemoRelayStatus {
    if config_json.is_null() || out_report_json.is_null() {
        return unsafe {
            api::nemo_relay_plugin_validate(config_json, ptr::null(), out_report_json)
        };
    }
    let config = unsafe { CStr::from_ptr(config_json) }
        .to_str()
        .ok()
        .and_then(|value| serde_json::from_str::<Json>(value).ok());
    let Some(config) = config else {
        return unsafe {
            api::nemo_relay_plugin_validate(config_json, ptr::null(), out_report_json)
        };
    };
    let config = cstring(&config.to_string());
    let mut host_report = ptr::null_mut();
    let status =
        unsafe { api::nemo_relay_plugin_validate(config.as_ptr(), ptr::null(), &mut host_report) };
    if status != NemoRelayStatus::Ok {
        return status;
    }
    let report = unsafe { returned_json(host_report) };
    let config_report = report.get("config").cloned().unwrap_or(Json::Null);
    unsafe { *out_report_json = CString::new(config_report.to_string()).unwrap().into_raw() };
    NemoRelayStatus::Ok
}

pub(super) unsafe fn activate_test_plugin_config(
    config_json: *const c_char,
    out_report_json: *mut *mut c_char,
) -> NemoRelayStatus {
    let close_status = close_test_plugin_host();
    if close_status != NemoRelayStatus::Ok {
        return close_status;
    }
    let mut activation = ptr::null_mut();
    if out_report_json.is_null() {
        let status = unsafe {
            api::nemo_relay_plugin_initialize(
                config_json,
                ptr::null(),
                &mut activation,
                out_report_json,
            )
        };
        if status == NemoRelayStatus::Ok {
            *TEST_PLUGIN_HOST
                .lock()
                .unwrap_or_else(|error| error.into_inner()) = Some(activation as usize);
        }
        return status;
    }
    let mut host_report = ptr::null_mut();
    let status = unsafe {
        api::nemo_relay_plugin_initialize(
            config_json,
            ptr::null(),
            &mut activation,
            &mut host_report,
        )
    };
    if status != NemoRelayStatus::Ok {
        return status;
    }
    let report = unsafe { returned_json(host_report) };
    let config_report = report.get("config").cloned().unwrap_or(Json::Null);
    unsafe { *out_report_json = CString::new(config_report.to_string()).unwrap().into_raw() };
    *TEST_PLUGIN_HOST
        .lock()
        .unwrap_or_else(|error| error.into_inner()) = Some(activation as usize);
    NemoRelayStatus::Ok
}

pub(super) unsafe fn test_plugin_host_report_json(
    out_report_json: *mut *mut c_char,
) -> NemoRelayStatus {
    if out_report_json.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    let guard = TEST_PLUGIN_HOST
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let Some(activation) = *guard else {
        unsafe { *out_report_json = CString::new("null").unwrap().into_raw() };
        return NemoRelayStatus::Ok;
    };
    let mut host_report = ptr::null_mut();
    let status = unsafe {
        api::nemo_relay_plugin_host_activation_report_json(
            activation as *mut FfiPluginHostActivation,
            &mut host_report,
        )
    };
    if status != NemoRelayStatus::Ok {
        return status;
    }
    let report = unsafe { returned_json(host_report) };
    let config_report = report.get("config").cloned().unwrap_or(Json::Null);
    unsafe { *out_report_json = CString::new(config_report.to_string()).unwrap().into_raw() };
    NemoRelayStatus::Ok
}

pub(super) fn close_test_plugin_host() -> NemoRelayStatus {
    let activation = TEST_PLUGIN_HOST
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .take();
    let Some(activation) = activation else {
        return NemoRelayStatus::Ok;
    };
    let mut activation = activation as *mut FfiPluginHostActivation;
    let status = unsafe { api::nemo_relay_plugin_host_activation_close(activation) };
    if status == NemoRelayStatus::Ok {
        unsafe { nemo_relay_plugin_host_activation_free(&mut activation) };
    } else {
        *TEST_PLUGIN_HOST
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = Some(activation as usize);
    }
    status
}
