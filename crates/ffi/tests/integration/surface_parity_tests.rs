// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::ptr;

#[test]
fn atif_constructor_preserves_complete_agent_configuration() {
    unsafe {
        let mut exporter = ptr::null_mut();
        assert_eq!(
            nemo_relay_atif_exporter_create(
                c"session".as_ptr(),
                c"agent".as_ptr(),
                c"1".as_ptr(),
                c"model".as_ptr(),
                c"trajectory".as_ptr(),
                c"[{\"name\":\"search\"}]".as_ptr(),
                c"{\"custom\":true}".as_ptr(),
                &mut exporter,
            ),
            NemoRelayStatus::Ok
        );
        let mut output = ptr::null_mut();
        assert_eq!(
            nemo_relay_atif_exporter_export(exporter, &mut output),
            NemoRelayStatus::Ok
        );
        let trajectory: Json =
            serde_json::from_str(CStr::from_ptr(output).to_str().unwrap()).unwrap();
        nemo_relay_string_free(output);
        nemo_relay_atif_exporter_free(exporter);
        assert_eq!(trajectory["session_id"], "session");
        assert_eq!(trajectory["trajectory_id"], "trajectory");
        assert_eq!(trajectory["agent"]["model_name"], "model");
        assert_eq!(
            trajectory["agent"]["tool_definitions"],
            json!([{"name":"search"}])
        );
        assert_eq!(trajectory["agent"]["extra"], json!({"custom":true}));
    }
}

#[test]
fn atif_constructor_rejects_invalid_optional_json() {
    unsafe {
        for (definitions, extra) in [
            (c"{}".as_ptr(), ptr::null()),
            (c"[".as_ptr(), ptr::null()),
            (ptr::null(), c"{".as_ptr()),
        ] {
            let mut exporter = ptr::null_mut();
            assert_eq!(
                nemo_relay_atif_exporter_create(
                    c"session".as_ptr(),
                    c"agent".as_ptr(),
                    c"1".as_ptr(),
                    ptr::null(),
                    ptr::null(),
                    definitions,
                    extra,
                    &mut exporter,
                ),
                NemoRelayStatus::InvalidJson
            );
            assert!(exporter.is_null());
            assert!(!nemo_relay_last_error().is_null());
        }
    }
}
