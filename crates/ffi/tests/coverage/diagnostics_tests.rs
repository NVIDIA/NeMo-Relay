// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::api::*;
use crate::error::NemoRelayStatus;
use nemo_relay::diagnostics::DiagnosticsSettings;
use nemo_relay::plugin::dynamic::{PluginHostConfig, initialize};
use std::ffi::{CStr, CString};

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
    let snapshot: nemo_relay::diagnostics::DiagnosticSnapshot =
        serde_json::from_str(unsafe { CStr::from_ptr(out) }.to_str().unwrap()).unwrap();
    assert_eq!(snapshot.schema_version, 1);
    unsafe {
        crate::convert::nemo_relay_string_free(out);
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
