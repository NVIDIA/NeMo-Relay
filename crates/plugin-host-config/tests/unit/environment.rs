// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fs;

use nemo_relay::plugin::dynamic::{
    DynamicPluginCheckState, DynamicPluginManifest, DynamicPluginManifestLoad, WorkerRuntime,
};
use sha2::Sha256;
use tempfile::tempdir;

use super::*;

fn python_manifest(root: &Path, id: &str) -> (DynamicPluginManifest, PathBuf) {
    fs::create_dir_all(root).unwrap();
    fs::write(root.join("plugin.py"), b"def main():\n    return None\n").unwrap();
    let manifest_path = root.join("relay-plugin.toml");
    let manifest = DynamicPluginManifest::parse_toml(&format!(
        r#"manifest_version = 1
[plugin]
id = "{id}"
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
sha256 = "sha256:placeholder"
[load]
runtime = "python"
entrypoint = "plugin:main"
"#,
    ))
    .unwrap();
    (manifest, manifest_path)
}

#[test]
fn python_entrypoint_contract_accepts_exact_artifacts_and_rejects_ambiguous_execution() {
    let temp = tempdir().unwrap();
    let (mut manifest, manifest_path) = python_manifest(temp.path(), "entrypoint");
    validate_python_entrypoint_artifact(&manifest, manifest_path.to_string_lossy().as_ref())
        .unwrap();

    if let DynamicPluginManifestLoad::Worker(load) = &mut manifest.load {
        load.runtime = Some(WorkerRuntime::Command);
    }
    validate_python_entrypoint_artifact(&manifest, manifest_path.to_string_lossy().as_ref())
        .unwrap();

    if let DynamicPluginManifestLoad::Worker(load) = &mut manifest.load {
        load.runtime = Some(WorkerRuntime::Python);
        load.entrypoint = Some("plugin".into());
    }
    let error =
        validate_python_entrypoint_artifact(&manifest, manifest_path.to_string_lossy().as_ref())
            .unwrap_err();
    assert!(error.contains("module:function"));

    if let DynamicPluginManifestLoad::Worker(load) = &mut manifest.load {
        load.entrypoint = Some("plugin:main:extra".into());
    }
    assert!(
        validate_python_entrypoint_artifact(&manifest, manifest_path.to_string_lossy().as_ref())
            .unwrap_err()
            .contains("module:function")
    );

    if let DynamicPluginManifestLoad::Worker(load) = &mut manifest.load {
        load.entrypoint = Some("plugin:main".into());
    }
    fs::create_dir(temp.path().join("plugin")).unwrap();
    fs::write(
        temp.path().join("plugin/__init__.py"),
        b"def main(): pass\n",
    )
    .unwrap();
    assert!(
        validate_python_entrypoint_artifact(&manifest, manifest_path.to_string_lossy().as_ref())
            .unwrap_err()
            .contains("exactly one source module")
    );

    fs::remove_dir_all(temp.path().join("plugin")).unwrap();
    fs::write(temp.path().join("other.py"), b"def main(): pass\n").unwrap();
    manifest.source.as_mut().unwrap().artifact = Some("other.py".into());
    assert!(
        validate_python_entrypoint_artifact(&manifest, manifest_path.to_string_lossy().as_ref())
            .unwrap_err()
            .contains("executed entrypoint module must be the integrity-checked artifact")
    );

    manifest.source = None;
    assert!(
        validate_python_entrypoint_artifact(&manifest, manifest_path.to_string_lossy().as_ref())
            .unwrap_err()
            .contains("source.manifest_root and source.artifact")
    );
}

#[test]
fn environment_state_reports_invalid_layouts_without_provisioning() {
    let temp = tempdir().unwrap();
    let (mut manifest, _) = python_manifest(&temp.path().join("plugin"), "managed-layout");
    let state_path = temp.path().join(".dynamic-plugins.json");

    assert_eq!(
        environment_state(&manifest, &state_path, None),
        DynamicPluginCheckState::Invalid
    );

    let expected = managed_environment_path(&state_path, "managed-layout").unwrap();
    fs::create_dir_all(expected.parent().unwrap()).unwrap();
    fs::write(&expected, b"not a directory").unwrap();
    let error = validate_environment_state(
        &manifest,
        &state_path,
        Some(expected.to_string_lossy().as_ref()),
    )
    .unwrap_err();
    assert!(error.to_string().contains("must be a directory"));

    fs::remove_file(&expected).unwrap();
    fs::create_dir(&expected).unwrap();
    let configured_file = temp.path().join("configured-file");
    fs::write(&configured_file, b"not a directory").unwrap();
    let error = validate_environment_state(
        &manifest,
        &state_path,
        Some(configured_file.to_string_lossy().as_ref()),
    )
    .unwrap_err();
    assert!(error.to_string().contains("configured Python environment"));

    let error = validate_environment_state(
        &manifest,
        &state_path,
        Some(expected.to_string_lossy().as_ref()),
    )
    .unwrap_err();
    assert!(error.to_string().contains("has no Python launcher"));

    let python = environment_python_path(&expected);
    fs::create_dir_all(python.parent().unwrap()).unwrap();
    fs::write(&python, b"python").unwrap();
    manifest.integrity = None;
    let error = validate_environment_state(
        &manifest,
        &state_path,
        Some(expected.to_string_lossy().as_ref()),
    )
    .unwrap_err();
    assert!(error.to_string().contains("requires integrity.sha256"));

    let DynamicPluginManifestLoad::Worker(load) = &mut manifest.load else {
        unreachable!();
    };
    load.runtime = Some(WorkerRuntime::Command);
    assert_eq!(
        validate_environment_state(&manifest, &state_path, None).unwrap(),
        DynamicPluginCheckState::Unknown
    );
}

#[test]
fn attestation_parser_rejects_unauthenticated_documents_before_tree_verification() {
    let temp = tempdir().unwrap();
    let attestation_path = temp.path().join(ENVIRONMENT_ATTESTATION_FILE);
    fs::write(&attestation_path, "not-json").unwrap();
    assert!(
        read_environment_attestation(temp.path(), "sha256:source")
            .unwrap_err()
            .to_string()
            .contains("attestation")
    );

    fs::write(
        &attestation_path,
        serde_json::json!({
            "version": 1,
            "source_artifact_sha256": "sha256:source",
            "environment_sha256": "a".repeat(64),
            "authentication": "not-authenticated"
        })
        .to_string(),
    )
    .unwrap();
    let error = read_environment_attestation(temp.path(), "sha256:source").unwrap_err();
    assert!(error.to_string().contains("failed authentication"));

    assert!(!verify_environment_authentication("source", "environment", "plain").unwrap());
    assert!(
        !verify_environment_authentication("source", "environment", "hmac-sha256:not-hex").unwrap()
    );
    assert_eq!(decode_fixed_hex::<2>("00ff"), Some([0, 255]));
    assert_eq!(decode_fixed_hex::<2>("0"), None);
}

#[test]
fn environment_digest_rejects_cycles_depth_overflow_entry_overflow_and_special_files() {
    let temp = tempdir().unwrap();
    let environment = temp.path().join("environment");
    fs::create_dir(&environment).unwrap();
    fs::write(environment.join("module.py"), b"fixture").unwrap();
    fs::create_dir(environment.join("__pycache__")).unwrap();
    fs::write(environment.join("cached.pyc"), b"ignored").unwrap();
    assert_eq!(environment_tree_digest(&environment).unwrap().len(), 64);

    let missing = temp.path().join("missing-environment");
    assert!(
        environment_tree_digest(&missing)
            .unwrap_err()
            .to_string()
            .contains("normalize environment directory")
    );
    let not_directory = temp.path().join("not-a-directory");
    fs::write(&not_directory, b"file").unwrap();
    assert!(
        environment_tree_digest(&not_directory)
            .unwrap_err()
            .to_string()
            .contains("read environment directory")
    );

    let absolute = resolve_relative_path(Path::new("ignored"), &environment.to_string_lossy());
    assert_eq!(absolute, environment);
    assert!(
        absolute_path(Path::new("relative-environment"))
            .unwrap()
            .is_absolute()
    );

    let mut digest = Sha256::new();
    let mut entries = MAX_ENVIRONMENT_FILES;
    let error = digest_environment_directory(
        &environment,
        Path::new(""),
        &mut vec![PathBuf::new(); MAX_ENVIRONMENT_DEPTH],
        &mut digest,
        &mut 0,
        &mut 0,
    )
    .unwrap_err();
    assert!(error.to_string().contains("traversal depth"));

    let canonical = environment.canonicalize().unwrap();
    let error = digest_environment_directory(
        &environment,
        Path::new(""),
        &mut vec![canonical],
        &mut Sha256::new(),
        &mut 0,
        &mut 0,
    )
    .unwrap_err();
    assert!(error.to_string().contains("symlink cycle"));

    let error = digest_environment_directory(
        &environment,
        Path::new(""),
        &mut Vec::new(),
        &mut Sha256::new(),
        &mut 0,
        &mut entries,
    )
    .unwrap_err();
    assert!(error.to_string().contains("entry attestation budget"));

    #[cfg(unix)]
    {
        use std::ffi::CString;
        use std::os::unix::fs::symlink;

        let dangling = environment.join("dangling");
        symlink(environment.join("missing"), &dangling).unwrap();
        assert!(
            resolve_environment_entry(&dangling)
                .unwrap_err()
                .to_string()
                .contains("resolve environment symlink")
        );
        fs::remove_file(dangling).unwrap();

        let fifo = environment.join("worker.pipe");
        let encoded = CString::new(fifo.as_os_str().as_encoded_bytes()).unwrap();
        // SAFETY: `encoded` is a valid NUL-terminated path and the mode is permission bits only.
        assert_eq!(unsafe { libc::mkfifo(encoded.as_ptr(), 0o600) }, 0);
        let error = environment_tree_digest(&environment).unwrap_err();
        assert!(error.to_string().contains("regular file or directory"));
    }
}

#[cfg(not(windows))]
#[test]
fn bootstrap_hmac_key_is_created_reused_and_invalid_lengths_are_rejected() {
    let temp = tempdir().unwrap();
    let key_path = temp.path().join("bootstrap/fingerprint-hmac.key");

    let created = load_or_create_hmac_key_at(&key_path).unwrap();
    assert_eq!(created.len(), HMAC_KEY_BYTES);
    assert_eq!(load_or_create_hmac_key_at(&key_path).unwrap(), created);

    fs::write(&key_path, b"short").unwrap();
    let error = load_or_create_hmac_key_at(&key_path).unwrap_err();
    assert!(error.to_string().contains("invalid length"));

    let blocked_config_root = temp.path().join("blocked-config-root");
    fs::write(&blocked_config_root, b"file").unwrap();
    let error = load_or_create_hmac_key_at(&blocked_config_root.join("key")).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("create bootstrap state directory")
    );
}
