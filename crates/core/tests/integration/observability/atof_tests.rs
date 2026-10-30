// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the ATOF JSONL exporter.

use super::*;
use crate::api::runtime::{NemoRelayContextState, global_context};
use crate::api::scope::{EmitMarkEventParams, PopScopeParams, PushScopeParams, ScopeType};
use serde_json::json;
use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};
use uuid::Uuid;

fn temp_dir(prefix: &str) -> PathBuf {
    let id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("nemo-relay-{prefix}-{id}"));
    fs::create_dir_all(&path).unwrap();
    path.canonicalize().unwrap()
}

fn reset_global() {
    crate::shared_runtime::reset_runtime_owner_for_tests();
    let context = global_context();
    *context.write().unwrap() = NemoRelayContextState::new();
}

fn read_jsonl(path: &Path) -> Vec<serde_json::Value> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[test]
fn register_deregister_flush_and_shutdown_work_with_runtime_events() {
    let _guard = crate::observability_test_support::test_mutex()
        .lock()
        .unwrap();
    reset_global();

    let dir = temp_dir("atof-runtime");
    let exporter = crate::api::exporter::atof(
        AtofExporterConfig::new()
            .with_output_directory(&dir)
            .with_filename("events.jsonl"),
    )
    .unwrap();
    let name = format!("atof_exporter_{}", Uuid::now_v7());

    exporter.register(&name).unwrap();
    let handle = crate::api::scope::push_scope(
        PushScopeParams::builder()
            .name("atof_scope")
            .scope_type(ScopeType::Agent)
            .input(json!({"scope": true}))
            .build(),
    )
    .unwrap();
    crate::api::scope::event(
        EmitMarkEventParams::builder()
            .name("atof_mark")
            .parent(&handle)
            .data(json!({"mark": true}))
            .build(),
    )
    .unwrap();
    crate::api::scope::pop_scope(
        PopScopeParams::builder()
            .handle_uuid(&handle.uuid)
            .output(json!({"done": true}))
            .build(),
    )
    .unwrap();

    assert!(exporter.deregister(&name).unwrap());
    assert!(!exporter.deregister(&name).unwrap());
    exporter.force_flush().unwrap();
    exporter.shutdown().unwrap();
    exporter.shutdown().unwrap();

    let lines = read_jsonl(exporter.path().expect("file sink path"));
    assert_eq!(lines.len(), 3);
    assert_eq!(lines[0]["name"], "atof_scope");
    assert_eq!(lines[1]["name"], "atof_mark");
    assert_eq!(lines[2]["scope_category"], "end");
}
