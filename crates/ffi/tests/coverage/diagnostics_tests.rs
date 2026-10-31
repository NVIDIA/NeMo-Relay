// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::api::*;
use crate::error::NemoRelayStatus;
use nemo_relay_diagnostics::DiagnosticsSettings;
use std::ffi::{CStr, CString};
use {
    nemo_relay_plugin_host::dynamic::PluginHostConfig, nemo_relay_plugin_host::dynamic::initialize,
};

#[test]
fn query_byte_limit_matches_core_contract() {
    assert_eq!(
        nemo_relay_diagnostics_query_max_bytes(),
        nemo_relay_diagnostics::embedded::MAX_BINDING_QUERY_BYTES as u32
    );
}

#[test]
fn initialization_and_validation_preserve_configuration_parse_details() {
    let config = CString::new(r#"{"diagnostics":{"enabled":"true"}}"#).unwrap();
    let mut activation = std::ptr::null_mut();
    let mut report = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            nemo_relay_plugin_initialize(
                config.as_ptr(),
                std::ptr::null(),
                &mut activation,
                &mut report,
            )
        },
        NemoRelayStatus::InvalidJson
    );
    assert!(activation.is_null());
    assert!(report.is_null());
    let initialization_error = unsafe { CStr::from_ptr(crate::error::nemo_relay_last_error()) }
        .to_str()
        .unwrap()
        .to_owned();
    assert!(initialization_error.starts_with("invalid plugin config:"));
    assert!(initialization_error.contains("expected a boolean"));
    assert_eq!(
        unsafe { nemo_relay_plugin_validate(config.as_ptr(), std::ptr::null(), &mut report) },
        NemoRelayStatus::InvalidJson
    );
    assert!(report.is_null());
    assert_eq!(
        unsafe { CStr::from_ptr(crate::error::nemo_relay_last_error()) }
            .to_str()
            .unwrap(),
        initialization_error
    );
}

#[tokio::test]
async fn reader_owns_strings_and_survives_plugin_close() {
    let mut reader = std::ptr::null_mut();
    assert_eq!(
        unsafe { nemo_relay_diagnostics_reader(&mut reader) },
        NemoRelayStatus::NotFound
    );
    assert!(reader.is_null());
    let mut activation = initialize(
        PluginHostConfig {
            diagnostics: Some(DiagnosticsSettings {
                enabled: true,
                ..Default::default()
            }),
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    assert_eq!(
        unsafe { nemo_relay_diagnostics_reader(&mut reader) },
        NemoRelayStatus::Ok
    );
    activation.close().unwrap();
    let mut out = std::ptr::null_mut();
    assert_eq!(
        unsafe { nemo_relay_diagnostics_snapshot_json(reader, std::ptr::null(), &mut out) },
        NemoRelayStatus::Ok
    );
    let snapshot: nemo_relay_diagnostics::DiagnosticSnapshot =
        serde_json::from_str(unsafe { CStr::from_ptr(out) }.to_str().unwrap()).unwrap();
    assert_eq!(snapshot.schema_version, 1);
    unsafe {
        crate::convert::nemo_relay_string_free(out);
    }
    assert_eq!(
        unsafe { nemo_relay_diagnostics_memory_json(reader, std::ptr::null(), &mut out) },
        NemoRelayStatus::Ok
    );
    let memory: nemo_relay_diagnostics::DiagnosticMemoryPage =
        serde_json::from_str(unsafe { CStr::from_ptr(out) }.to_str().unwrap()).unwrap();
    assert_eq!(
        memory.component_occupied_bytes.get() + memory.shared_or_unassigned_reserved_bytes.get(),
        memory.aggregate.reserved_bytes as u64
    );
    unsafe {
        crate::convert::nemo_relay_string_free(out);
    }
    for (query, status, reason) in [
        (
            r#"{"schema_version":2}"#,
            NemoRelayStatus::InvalidArg,
            "unsupported_schema",
        ),
        (
            r#"{"max_response_bytes":1}"#,
            NemoRelayStatus::Internal,
            "snapshot_capacity",
        ),
    ] {
        let input = CString::new(query).unwrap();
        assert_eq!(
            unsafe { nemo_relay_diagnostics_snapshot_json(reader, input.as_ptr(), &mut out) },
            status
        );
        assert!(out.is_null());
        assert_eq!(
            unsafe { CStr::from_ptr(crate::error::nemo_relay_last_error()) }
                .to_str()
                .unwrap(),
            reason
        );
    }
    let oversized = CString::new("x".repeat(4097)).unwrap();
    assert_eq!(
        unsafe { nemo_relay_diagnostics_snapshot_json(reader, oversized.as_ptr(), &mut out) },
        NemoRelayStatus::InvalidArg
    );
    assert!(out.is_null());
    unsafe {
        nemo_relay_diagnostics_reader_free(&mut reader);
        nemo_relay_diagnostics_reader_free(&mut reader);
    }
    assert!(reader.is_null());
}
