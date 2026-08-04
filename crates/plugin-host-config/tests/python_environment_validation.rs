// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Regression coverage for lifecycle-managed Python environment validation.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use nemo_relay_plugin_host_config::{
    ENVIRONMENT_ATTESTATION_FILE, prepare_plugin_host_activation, reconcile_plugin_lifecycle,
    resolve_plugin_files_from_paths, verify_environment_attestation,
};
use ring::hmac;
use serde_json::json;
use sha2::{Digest, Sha256};
use tempfile::tempdir;

const LIFECYCLE_STATE_FILE: &str = ".dynamic-plugins.json";
const ATTESTATION_DOMAIN: &[u8] = b"nemo-relay/python-environment-attestation/v1\0";
static ENVIRONMENT_LOCK: Mutex<()> = Mutex::new(());

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn write_python_manifest(root: &Path, plugin_id: &str) -> (PathBuf, String) {
    fs::create_dir_all(root).unwrap();
    let artifact = b"def main():\n    return None\n";
    fs::write(root.join("plugin.py"), artifact).unwrap();
    let source_digest = format!("sha256:{}", sha256_hex(artifact));
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
artifact = "plugin.py"
[integrity]
sha256 = "{source_digest}"
[load]
runtime = "python"
entrypoint = "plugin:main"
"#
        ),
    )
    .unwrap();
    (manifest, source_digest)
}

fn update_tree_digest(digest: &mut Sha256, entry_type: u8, path: &Path, payload: &[u8]) {
    let path = raw_path_bytes(path);
    digest.update([entry_type]);
    digest.update((path.len() as u64).to_le_bytes());
    digest.update(path);
    digest.update((payload.len() as u64).to_le_bytes());
    digest.update(payload);
}

#[cfg(unix)]
fn raw_path_bytes(path: &Path) -> Vec<u8> {
    use std::os::unix::ffi::OsStrExt;
    path.as_os_str().as_bytes().to_vec()
}

#[cfg(windows)]
fn raw_path_bytes(path: &Path) -> Vec<u8> {
    use std::os::windows::ffi::OsStrExt;
    path.as_os_str()
        .encode_wide()
        .flat_map(u16::to_le_bytes)
        .collect()
}

fn write_attested_environment(environment: &Path, source_digest: &str, key_bytes: &[u8; 32]) {
    let (interpreter_dir, interpreter_name) = if cfg!(windows) {
        ("Scripts", "python.exe")
    } else {
        ("bin", "python")
    };
    let interpreter_bytes = b"managed Python interpreter fixture";
    fs::create_dir_all(environment.join(interpreter_dir)).unwrap();
    fs::write(
        environment.join(interpreter_dir).join(interpreter_name),
        interpreter_bytes,
    )
    .unwrap();

    let mut digest = Sha256::new();
    update_tree_digest(&mut digest, b'd', Path::new(interpreter_dir), &[]);
    update_tree_digest(
        &mut digest,
        b'f',
        &Path::new(interpreter_dir).join(interpreter_name),
        interpreter_bytes,
    );
    let environment_digest = digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();

    let mut message = Vec::new();
    message.extend_from_slice(ATTESTATION_DOMAIN);
    message.extend_from_slice(source_digest.trim().as_bytes());
    message.push(0);
    message.extend_from_slice(environment_digest.as_bytes());
    let key = hmac::Key::new(hmac::HMAC_SHA256, key_bytes);
    let authentication = format!(
        "hmac-sha256:{}",
        hmac::sign(&key, &message)
            .as_ref()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    );
    fs::write(
        environment.join(ENVIRONMENT_ATTESTATION_FILE),
        serde_json::to_vec_pretty(&json!({
            "version": 1,
            "source_artifact_sha256": source_digest,
            "environment_sha256": environment_digest,
            "authentication": authentication,
        }))
        .unwrap(),
    )
    .unwrap();
}

fn lifecycle_environment(root: &Path, plugin_id: &str) -> PathBuf {
    root.join(".dynamic-plugin-environments")
        .join(sha256_hex(plugin_id.as_bytes()))
}

fn enable_with_environment(state_path: &Path, environment: &Path) {
    let mut state: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(state_path).unwrap()).unwrap();
    state["records"][0]["spec"]["enabled"] = true.into();
    state["records"][0]["source"]["environment_ref"] =
        environment.to_string_lossy().as_ref().into();
    fs::write(state_path, serde_json::to_vec_pretty(&state).unwrap()).unwrap();
}

#[test]
fn enabled_python_worker_rejects_an_attested_environment_outside_its_lifecycle_path() {
    let _environment = ENVIRONMENT_LOCK.lock().unwrap();
    let temp = tempdir().unwrap();
    let xdg_config_home = temp.path().join("xdg");
    // SAFETY: environment-sensitive tests in this binary hold `ENVIRONMENT_LOCK`.
    unsafe { std::env::set_var("XDG_CONFIG_HOME", &xdg_config_home) };
    let key_bytes = [0x5a; 32];
    let key_directory = xdg_config_home.join("nemo-relay").join("bootstrap");
    fs::create_dir_all(&key_directory).unwrap();
    fs::write(key_directory.join("fingerprint-hmac.key"), key_bytes).unwrap();

    let (manifest, source_digest) =
        write_python_manifest(&temp.path().join("plugin"), "wrong-environment-location");
    let config = temp.path().join("plugins.toml");
    fs::write(
        &config,
        format!(
            "[[plugins.dynamic]]\nmanifest = {:?}\n",
            manifest.to_string_lossy()
        ),
    )
    .unwrap();
    let resolved = resolve_plugin_files_from_paths([config.clone()], None).unwrap();
    reconcile_plugin_lifecycle(&resolved).unwrap();

    let wrong_environment = temp.path().join("attested-but-not-lifecycle-managed");
    write_attested_environment(&wrong_environment, &source_digest, &key_bytes);
    verify_environment_attestation(&wrong_environment, &source_digest).unwrap();

    let state_path = temp.path().join(LIFECYCLE_STATE_FILE);
    let mut state: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&state_path).unwrap()).unwrap();
    state["records"][0]["spec"]["enabled"] = true.into();
    state["records"][0]["source"]["environment_ref"] =
        wrong_environment.to_string_lossy().as_ref().into();
    fs::write(&state_path, serde_json::to_vec_pretty(&state).unwrap()).unwrap();

    let error = match prepare_plugin_host_activation(resolved) {
        Ok(_) => {
            panic!("wrong-location Python environment unexpectedly reached an activation plan")
        }
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("invalid lifecycle-managed environment")
    );
    let persisted: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(state_path).unwrap()).unwrap();
    assert_eq!(
        persisted["records"][0]["status"]["validation"]["environment"],
        "invalid"
    );
}

#[test]
fn enabled_python_worker_reports_a_missing_lifecycle_environment_contextually() {
    let temp = tempdir().unwrap();
    let (manifest, _) = write_python_manifest(&temp.path().join("plugin"), "missing-environment");
    let config = temp.path().join("plugins.toml");
    fs::write(
        &config,
        format!(
            "[[plugins.dynamic]]\nmanifest = {:?}\n",
            manifest.to_string_lossy()
        ),
    )
    .unwrap();
    let resolved = resolve_plugin_files_from_paths([config], None).unwrap();
    reconcile_plugin_lifecycle(&resolved).unwrap();

    let state_path = temp.path().join(LIFECYCLE_STATE_FILE);
    let environment = lifecycle_environment(temp.path(), "missing-environment");
    enable_with_environment(&state_path, &environment);

    let error = match prepare_plugin_host_activation(resolved) {
        Ok(_) => panic!("missing Python environment unexpectedly reached an activation plan"),
        Err(error) => error.to_string(),
    };
    assert!(
        error.contains("failed to inspect lifecycle-managed Python environment"),
        "{error}"
    );
    assert!(
        error.contains(
            dunce::canonicalize(temp.path())
                .unwrap()
                .join(environment.strip_prefix(temp.path()).unwrap())
                .to_string_lossy()
                .as_ref()
        ),
        "{error}"
    );
}

#[cfg(unix)]
#[test]
fn enabled_python_worker_rejects_a_symlinked_lifecycle_environment_slot() {
    use std::os::unix::fs::symlink;

    let _environment = ENVIRONMENT_LOCK.lock().unwrap();
    let temp = tempdir().unwrap();
    let xdg_config_home = temp.path().join("xdg");
    // SAFETY: environment-sensitive tests in this binary hold `ENVIRONMENT_LOCK`.
    unsafe { std::env::set_var("XDG_CONFIG_HOME", &xdg_config_home) };
    let key_bytes = [0x5b; 32];
    let key_directory = xdg_config_home.join("nemo-relay").join("bootstrap");
    fs::create_dir_all(&key_directory).unwrap();
    fs::write(key_directory.join("fingerprint-hmac.key"), key_bytes).unwrap();

    let plugin_id = "symlinked-environment";
    let (manifest, source_digest) = write_python_manifest(&temp.path().join("plugin"), plugin_id);
    let config = temp.path().join("plugins.toml");
    fs::write(
        &config,
        format!(
            "[[plugins.dynamic]]\nmanifest = {:?}\n",
            manifest.to_string_lossy()
        ),
    )
    .unwrap();
    let resolved = resolve_plugin_files_from_paths([config], None).unwrap();
    reconcile_plugin_lifecycle(&resolved).unwrap();

    let external = temp.path().join("external-environment");
    write_attested_environment(&external, &source_digest, &key_bytes);
    let managed = lifecycle_environment(temp.path(), plugin_id);
    fs::create_dir_all(managed.parent().unwrap()).unwrap();
    symlink(&external, &managed).unwrap();
    let state_path = temp.path().join(LIFECYCLE_STATE_FILE);
    enable_with_environment(&state_path, &external);

    let error = match prepare_plugin_host_activation(resolved) {
        Ok(_) => panic!("symlinked Python environment unexpectedly reached an activation plan"),
        Err(error) => error.to_string(),
    };
    assert!(
        error.contains("lifecycle-managed Python environment")
            && error.contains("not a symbolic link"),
        "{error}"
    );
}
