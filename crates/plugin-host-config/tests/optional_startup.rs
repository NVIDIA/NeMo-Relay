// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Optional dynamic-plugin preflight failures remain activation attempts.

use std::fs;
use std::path::{Path, PathBuf};

use nemo_relay_plugin_host_config::{
    prepare_plugin_host_activation, reconcile_plugin_lifecycle, resolve_plugin_files_from_paths,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tempfile::tempdir;

fn write_command_worker(root: &Path, plugin_id: &str) -> PathBuf {
    fs::create_dir_all(root).unwrap();
    let artifact = b"#!/bin/sh\nexit 0\n";
    fs::write(root.join("worker.sh"), artifact).unwrap();
    let digest = Sha256::digest(artifact)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let manifest = root.join("relay-plugin.toml");
    fs::write(
        &manifest,
        format!(
            r#"manifest_version = 1
[plugin]
id = "{plugin_id}"
kind = "worker"
[compat]
relay = ">=0.5,<1.0"
worker_protocol = "grpc-v1"
[capabilities]
items = ["plugin_worker"]
[defaults]
enabled = false
[source]
manifest_root = "."
artifact = "worker.sh"
[integrity]
sha256 = "sha256:{digest}"
[load]
runtime = "command"
entrypoint = "worker.sh"
"#
        ),
    )
    .unwrap();
    manifest
}

fn enable_record(state_path: &Path) {
    let mut state: Value = serde_json::from_slice(&fs::read(state_path).unwrap()).unwrap();
    state["records"][0]["spec"]["enabled"] = true.into();
    fs::write(state_path, serde_json::to_vec_pretty(&state).unwrap()).unwrap();
}

fn optional_plan(policy: &str, plugin_id: &str) -> (usize, Value) {
    let temp = tempdir().unwrap();
    let manifest = write_command_worker(&temp.path().join("plugin"), plugin_id);
    let config = temp.path().join("plugins.toml");
    fs::write(
        &config,
        format!(
            "[[plugins.dynamic]]\nmanifest = {:?}\n\n{policy}\n",
            manifest.to_string_lossy()
        ),
    )
    .unwrap();
    let resolved = resolve_plugin_files_from_paths([config], None).unwrap();
    reconcile_plugin_lifecycle(&resolved).unwrap();
    let state_path = temp.path().join(".dynamic-plugins.json");
    enable_record(&state_path);

    let plan = prepare_plugin_host_activation(resolved).unwrap();
    let persisted = serde_json::from_slice(&fs::read(state_path).unwrap()).unwrap();
    (plan.dynamic_plugins.len(), persisted)
}

#[test]
fn optional_policy_failure_is_recorded_but_still_planned() {
    let (planned, state) = optional_plan(
        r#"[plugins.policy.defaults]
startup = "optional"
allowed = false"#,
        "optional-policy",
    );

    assert_eq!(planned, 1);
    assert_eq!(
        state["records"][0]["status"]["validation"]["policy_satisfied"],
        "invalid"
    );
    assert_eq!(state["records"][0]["status"]["startup_class"], "optional");
}

#[test]
fn optional_trust_failure_is_recorded_but_still_planned() {
    let (planned, state) = optional_plan(
        r#"[plugins.policy.defaults]
startup = "optional"
attestation = "signature_required""#,
        "optional-trust",
    );

    assert_eq!(planned, 1);
    assert_eq!(
        state["records"][0]["status"]["validation"]["authenticity"],
        "invalid"
    );
    assert_eq!(state["records"][0]["status"]["startup_class"], "optional");
}
