// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for debug in the NeMo Relay adaptive crate.

use std::process::Command;

use serde_json::json;

use super::*;

#[test]
fn debug_env_flag_enabled_recognizes_truthy_and_falsey_values() {
    for value in ["", "0", "false", "off", "no", " FALSE "] {
        assert!(!env_flag_enabled(value), "{value:?} should disable debug");
    }

    for value in ["1", "true", "yes", "debug"] {
        assert!(env_flag_enabled(value), "{value:?} should enable debug");
    }
}

#[test]
fn debug_emit_emits_object_and_scalar_payloads_when_enabled_in_child_process() {
    if std::env::var_os("NEMO_RELAY_ACG_DEBUG_CHILD").is_some() {
        emit("object", json!({"value": 1}));
        emit("scalar", json!("payload"));
        return;
    }

    let output = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "acg::debug::tests::debug_emit_emits_object_and_scalar_payloads_when_enabled_in_child_process",
            "--nocapture",
        ])
        .env("NEMO_RELAY_ACG_DEBUG_CHILD", "1")
        .env("NEMO_RELAY_ACG_DEBUG", "1")
        .output()
        .unwrap();

    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("nemo-relay-adaptive acg-debug"));
    assert!(stderr.contains("\"event\":\"object\""));
    assert!(stderr.contains("\"value\":1"));
    assert!(stderr.contains("\"event\":\"scalar\""));
    assert!(stderr.contains("\"payload\":\"payload\""));
}
