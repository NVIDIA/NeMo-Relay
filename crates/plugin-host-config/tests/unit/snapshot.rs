// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use tempfile::tempdir;

use super::*;

#[cfg(target_os = "macos")]
#[test]
fn macos_snapshot_copies_only_the_authenticated_venvs_matching_runtime_library() {
    let temp = tempdir().unwrap();
    let mutable_base = temp.path().join("mutable-base");
    let authenticated_base = temp.path().join("authenticated-base");
    fs::create_dir_all(mutable_base.join("bin")).unwrap();
    fs::create_dir_all(mutable_base.join("lib")).unwrap();
    fs::create_dir_all(authenticated_base.join("bin")).unwrap();
    fs::create_dir_all(authenticated_base.join("lib")).unwrap();
    fs::write(
        mutable_base.join("lib/libpython3.11.dylib"),
        b"mutable runtime",
    )
    .unwrap();
    fs::write(
        authenticated_base.join("lib/libpython3.11.dylib"),
        b"authenticated runtime",
    )
    .unwrap();
    fs::write(
        authenticated_base.join("lib/libpython3.12.dylib"),
        b"wrong version",
    )
    .unwrap();
    fs::write(
        authenticated_base.join("lib/libpython3.11.dylib.backup"),
        b"backup",
    )
    .unwrap();

    let mutable_environment = temp.path().join("mutable-environment");
    fs::create_dir(&mutable_environment).unwrap();
    fs::write(
        mutable_environment.join("pyvenv.cfg"),
        format!(
            "home = {}\nversion_info = 3.11.14\n",
            mutable_base.join("bin").display()
        ),
    )
    .unwrap();
    let copied_environment = temp.path().join("snapshot-environment");
    fs::create_dir(&copied_environment).unwrap();
    fs::write(
        copied_environment.join("pyvenv.cfg"),
        format!(
            "home = {}\nversion_info = 3.11.14\n",
            authenticated_base.join("bin").display()
        ),
    )
    .unwrap();
    let mut copied_files = HashMap::new();
    let mut budget = SnapshotBudget::default();

    snapshot_macos_python_runtime_library(&copied_environment, &mut copied_files, &mut budget)
        .unwrap();

    assert_eq!(
        fs::read(copied_environment.join("lib/libpython3.11.dylib")).unwrap(),
        b"authenticated runtime"
    );
    assert!(!copied_environment.join("lib/libpython3.12.dylib").exists());
    assert!(
        !copied_environment
            .join("lib/libpython3.11.dylib.backup")
            .exists()
    );
    assert_eq!(budget.entries, 1);
    assert_eq!(budget.bytes, b"authenticated runtime".len() as u64);
}

#[cfg(target_os = "macos")]
#[test]
fn macos_snapshot_preserves_an_attested_runtime_library_and_avoids_empty_directories() {
    let temp = tempdir().unwrap();
    let base = temp.path().join("base");
    fs::create_dir_all(base.join("bin")).unwrap();
    fs::create_dir_all(base.join("lib")).unwrap();
    fs::write(base.join("lib/libpython3.11.dylib"), b"base runtime").unwrap();
    let copied_environment = temp.path().join("snapshot-environment");
    fs::create_dir_all(copied_environment.join("lib")).unwrap();
    fs::write(
        copied_environment.join("pyvenv.cfg"),
        format!("home = {}\nversion = 3.11.14\n", base.join("bin").display()),
    )
    .unwrap();
    let destination = copied_environment.join("lib/libpython3.11.dylib");
    fs::write(&destination, b"attested runtime").unwrap();
    let mut budget = SnapshotBudget::default();

    snapshot_macos_python_runtime_library(&copied_environment, &mut HashMap::new(), &mut budget)
        .unwrap();

    assert_eq!(fs::read(destination).unwrap(), b"attested runtime");
    assert_eq!(budget.entries, 0);
    assert_eq!(budget.bytes, 0);

    let static_base = temp.path().join("static-base");
    fs::create_dir_all(static_base.join("bin")).unwrap();
    let static_environment = temp.path().join("static-environment");
    fs::create_dir(&static_environment).unwrap();
    fs::write(
        static_environment.join("pyvenv.cfg"),
        format!(
            "home = {}\nversion_info = 3.12.0\n",
            static_base.join("bin").display()
        ),
    )
    .unwrap();
    snapshot_macos_python_runtime_library(
        &static_environment,
        &mut HashMap::new(),
        &mut SnapshotBudget::default(),
    )
    .unwrap();
    assert!(!static_environment.join("lib").exists());
}

#[cfg(unix)]
#[test]
fn snapshot_directory_copy_materializes_worker_launcher_and_preserves_versioned_aliases() {
    use std::os::unix::fs::symlink;

    let temp = tempdir().unwrap();
    let source = temp.path().join("environment");
    let bin = source.join("bin");
    let lib = source.join("lib");
    fs::create_dir_all(&bin).unwrap();
    fs::create_dir_all(&lib).unwrap();
    let interpreter = temp.path().join("managed-python");
    fs::write(&interpreter, b"python").unwrap();
    symlink(&interpreter, bin.join("python")).unwrap();
    symlink(&interpreter, bin.join("python3.11")).unwrap();
    symlink(&interpreter, bin.join("pip")).unwrap();
    symlink(&interpreter, lib.join("python3.11")).unwrap();
    let destination = temp.path().join("snapshot");

    copy_snapshot_directory(
        &source,
        &destination,
        &mut HashMap::new(),
        &mut SnapshotBudget::default(),
        false,
        &mut Vec::new(),
    )
    .unwrap();

    assert!(
        fs::symlink_metadata(destination.join("bin/python"))
            .unwrap()
            .is_file()
    );
    assert_eq!(fs::read(destination.join("bin/python")).unwrap(), b"python");
    assert!(
        fs::symlink_metadata(destination.join("bin/python3.11"))
            .unwrap()
            .file_type()
            .is_symlink()
    );
    assert!(
        fs::symlink_metadata(destination.join("bin/pip"))
            .unwrap()
            .is_file()
    );
    assert!(
        fs::symlink_metadata(destination.join("lib/python3.11"))
            .unwrap()
            .is_file()
    );
}

#[cfg(unix)]
#[test]
fn snapshot_worker_launcher_isolated_from_post_copy_symlink_target_substitution() {
    use std::os::unix::fs::{PermissionsExt, symlink};
    use std::process::Command;

    let temp = tempdir().unwrap();
    let source = temp.path().join("environment");
    let bin = source.join("bin");
    fs::create_dir_all(&bin).unwrap();
    let interpreter = temp.path().join("external-python");
    fs::write(&interpreter, b"#!/bin/sh\nprintf original-interpreter").unwrap();
    fs::set_permissions(&interpreter, fs::Permissions::from_mode(0o755)).unwrap();
    symlink(&interpreter, bin.join("python")).unwrap();
    let destination = temp.path().join("snapshot");

    copy_snapshot_directory(
        &source,
        &destination,
        &mut HashMap::new(),
        &mut SnapshotBudget::default(),
        true,
        &mut Vec::new(),
    )
    .unwrap();
    let launcher = destination.join("bin/python");
    let before = snapshot_tree_digest(&destination, false).unwrap();

    let replacement = temp.path().join("replacement-python");
    fs::write(&replacement, b"#!/bin/sh\nprintf substituted-interpreter").unwrap();
    fs::set_permissions(&replacement, fs::Permissions::from_mode(0o755)).unwrap();
    fs::rename(replacement, &interpreter).unwrap();

    assert_eq!(before, snapshot_tree_digest(&destination, false).unwrap());
    assert!(fs::symlink_metadata(&launcher).unwrap().is_file());
    let output = Command::new(&launcher).output().unwrap();
    assert!(output.status.success());
    assert_eq!(output.stdout, b"original-interpreter");
}

#[cfg(unix)]
#[test]
fn materialized_worker_launcher_preserves_python_virtual_environment() {
    use std::process::Command;

    let temp = tempdir().unwrap();
    let source = temp.path().join("source-venv");
    let status = Command::new("python3")
        .args(["-m", "venv", "--without-pip"])
        .arg(&source)
        .status()
        .expect("Python 3 is required to test virtual-environment snapshots");
    assert!(
        status.success(),
        "Python virtual environment creation failed"
    );
    let destination = temp.path().join("snapshot-venv");

    copy_snapshot_directory(
        &source,
        &destination,
        &mut HashMap::new(),
        &mut SnapshotBudget::default(),
        true,
        &mut Vec::new(),
    )
    .unwrap();

    let launcher = destination.join("bin/python");
    assert!(fs::symlink_metadata(&launcher).unwrap().is_file());
    let output = Command::new(&launcher)
        .args(["-c", "import sys; print(sys.prefix)"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "snapshotted virtual environment failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let observed_prefix = PathBuf::from(String::from_utf8(output.stdout).unwrap().trim());
    assert_eq!(
        fs::canonicalize(observed_prefix).unwrap(),
        fs::canonicalize(destination).unwrap()
    );
}

#[cfg(unix)]
#[test]
fn snapshot_protection_does_not_follow_python_launcher_symlink() {
    use std::os::unix::fs::{PermissionsExt, symlink};

    let temp = tempdir().unwrap();
    let target = temp.path().join("external-python");
    fs::write(&target, b"python").unwrap();
    fs::set_permissions(&target, fs::Permissions::from_mode(0o755)).unwrap();
    let root = temp.path().join("snapshot");
    let bin = root.join("bin");
    fs::create_dir_all(&bin).unwrap();
    symlink(&target, bin.join("python")).unwrap();

    protect_snapshot_tree(&root).unwrap();

    assert!(
        fs::symlink_metadata(bin.join("python"))
            .unwrap()
            .file_type()
            .is_symlink()
    );
    assert_eq!(
        fs::metadata(&target).unwrap().permissions().mode() & 0o777,
        0o755
    );
    make_snapshot_removable(&root);
}

#[cfg(unix)]
#[test]
fn snapshot_digest_hashes_python_launcher_symlink_without_following_it() {
    use std::os::unix::fs::symlink;

    let temp = tempdir().unwrap();
    let root = temp.path().join("snapshot");
    let bin = root
        .join(MANAGED_ENVIRONMENTS_DIR)
        .join("environment")
        .join("bin");
    fs::create_dir_all(&bin).unwrap();
    let launcher = bin.join("python");
    symlink("/missing/python-a", &launcher).unwrap();

    let first_verification = snapshot_tree_digest(&root, false).unwrap();
    let first_identity = snapshot_tree_digest(&root, true).unwrap();

    fs::remove_file(&launcher).unwrap();
    fs::write(&launcher, b"/missing/python-a").unwrap();
    assert_ne!(
        first_verification,
        snapshot_tree_digest(&root, false).unwrap(),
        "a regular file must not collide with an equivalent symlink target"
    );

    fs::remove_file(&launcher).unwrap();
    symlink("/missing/python-b", &launcher).unwrap();

    assert_ne!(
        first_verification,
        snapshot_tree_digest(&root, false).unwrap(),
        "verification must include the exact launcher target"
    );
    assert_eq!(
        first_identity,
        snapshot_tree_digest(&root, true).unwrap(),
        "managed environment contents are excluded from stable gateway identity"
    );
}

#[cfg(unix)]
#[test]
fn snapshot_directory_copy_rejects_special_entries_and_write_failures() {
    use std::ffi::CString;

    let temp = tempdir().unwrap();
    let fifo_source = temp.path().join("fifo-source");
    fs::create_dir(&fifo_source).unwrap();
    let fifo = fifo_source.join("worker.pipe");
    let fifo_c = CString::new(fifo.as_os_str().as_encoded_bytes()).unwrap();
    // SAFETY: `fifo_c` is a valid NUL-terminated path and the mode contains only permission bits.
    assert_eq!(unsafe { libc::mkfifo(fifo_c.as_ptr(), 0o600) }, 0);

    let special_error = copy_snapshot_directory(
        &fifo_source,
        &temp.path().join("fifo-snapshot"),
        &mut HashMap::new(),
        &mut SnapshotBudget::default(),
        false,
        &mut Vec::new(),
    )
    .unwrap_err()
    .to_string();
    assert!(
        special_error.contains("regular file or directory"),
        "{special_error}"
    );

    let regular = temp.path().join("regular");
    fs::write(&regular, b"regular").unwrap();
    let destination_directory = temp.path().join("destination-directory");
    fs::create_dir(&destination_directory).unwrap();
    let write_error = copy_snapshot_regular_file(
        &regular,
        &destination_directory,
        &mut HashMap::new(),
        &mut SnapshotBudget::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(write_error.contains("write snapshot file"), "{write_error}");
}

#[cfg(unix)]
#[test]
fn snapshot_directory_walk_rejects_missing_cycles_dangling_links_and_depth() {
    use std::os::unix::fs::symlink;

    let temp = tempdir().unwrap();
    let missing = temp.path().join("missing");
    let normalization_error = copy_snapshot_directory(
        &missing,
        &temp.path().join("destination"),
        &mut HashMap::new(),
        &mut SnapshotBudget::default(),
        false,
        &mut Vec::new(),
    )
    .unwrap_err()
    .to_string();
    assert!(
        normalization_error.contains("normalize runtime directory"),
        "{normalization_error}"
    );

    let source = temp.path().join("source");
    fs::create_dir(&source).unwrap();
    let canonical = source.canonicalize().unwrap();
    let cycle_error = copy_snapshot_directory_contents(
        &source,
        &temp.path().join("cycle-destination"),
        &mut HashMap::new(),
        &mut SnapshotBudget::default(),
        false,
        &mut vec![canonical],
    )
    .unwrap_err()
    .to_string();
    assert!(cycle_error.contains("symlink cycle"), "{cycle_error}");

    let destination_file = temp.path().join("destination-file");
    fs::write(&destination_file, b"file").unwrap();
    let destination_error = copy_snapshot_directory_contents(
        &source,
        &destination_file,
        &mut HashMap::new(),
        &mut SnapshotBudget::default(),
        false,
        &mut Vec::new(),
    )
    .unwrap_err()
    .to_string();
    assert!(
        destination_error.contains("create snapshot directory"),
        "{destination_error}"
    );

    symlink(temp.path().join("absent-target"), source.join("dangling")).unwrap();
    let dangling_error = copy_snapshot_directory(
        &source,
        &temp.path().join("dangling-destination"),
        &mut HashMap::new(),
        &mut SnapshotBudget::default(),
        false,
        &mut Vec::new(),
    )
    .unwrap_err()
    .to_string();
    assert!(
        dangling_error.contains("resolve runtime symlink"),
        "{dangling_error}"
    );

    let depth_error = collect_snapshot_files(
        &source,
        &source,
        &mut Vec::new(),
        MAX_SNAPSHOT_DEPTH,
        &mut 0,
    )
    .unwrap_err()
    .to_string();
    assert!(depth_error.contains("traversal depth"), "{depth_error}");
}

#[test]
fn snapshot_file_and_verification_helpers_cover_external_and_invalid_sources() {
    let temp = tempdir().unwrap();
    let plugin_dir = temp.path().join("plugin");
    fs::create_dir(&plugin_dir).unwrap();
    let manifest = plugin_dir.join("relay-plugin.toml");
    fs::write(&manifest, b"fixture").unwrap();
    let root = temp.path().join("snapshot");
    fs::create_dir(&root).unwrap();

    let missing_error = copy_snapshot_file(
        &root,
        &manifest,
        "missing.bin",
        "artifact",
        &mut HashMap::new(),
        &mut SnapshotBudget::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(
        missing_error.contains("normalize dynamic plugin artifact"),
        "{missing_error}"
    );

    let root_error = copy_snapshot_file(
        &root,
        &manifest,
        "/",
        "library",
        &mut HashMap::new(),
        &mut SnapshotBudget::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(
        root_error.contains("has no parent directory"),
        "{root_error}"
    );

    let external = temp.path().join("external-artifact.bin");
    fs::write(&external, b"external artifact").unwrap();
    let (logical, canonical, copied) = copy_snapshot_file(
        &root,
        &manifest,
        external.to_string_lossy().as_ref(),
        "artifact",
        &mut HashMap::new(),
        &mut SnapshotBudget::default(),
    )
    .unwrap();
    assert_eq!(logical, external);
    assert_eq!(canonical, external.canonicalize().unwrap());
    assert_eq!(fs::read(copied).unwrap(), b"external artifact");

    let one_file = temp.path().join("one-file");
    fs::create_dir(&one_file).unwrap();
    fs::write(one_file.join("entry"), b"entry").unwrap();
    let mut entries = MAX_SNAPSHOT_FILES;
    let entry_error =
        collect_snapshot_files(&one_file, &one_file, &mut Vec::new(), 0, &mut entries)
            .unwrap_err()
            .to_string();
    assert!(entry_error.contains("verification budget"), "{entry_error}");

    make_snapshot_removable(&temp.path().join("already-removed"));
}

#[test]
fn snapshot_budget_rejects_entry_and_byte_overflow() {
    let path = Path::new("fixture");
    let mut entry_budget = SnapshotBudget {
        entries: MAX_SNAPSHOT_FILES,
        ..SnapshotBudget::default()
    };
    let entry_error = entry_budget.record_entry(path).unwrap_err().to_string();
    assert!(entry_error.contains("entry activation snapshot budget"));

    let mut byte_budget = SnapshotBudget::default();
    let byte_error = byte_budget
        .record_bytes(path, usize::try_from(MAX_BOUNDED_FILE_BYTES).unwrap() + 1)
        .unwrap_err()
        .to_string();
    assert!(byte_error.contains("byte activation snapshot budget"));
}
