// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::OsString;

use super::*;

fn token(byte: u8) -> String {
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([byte; 32])
}

fn write_token_file(path: &Path, contents: &str) {
    // Owner-private on every platform, including the protected Windows DACL the reader requires.
    crate::filesystem::atomic_write_private(path, contents.as_bytes()).expect("write token file");
}

#[test]
fn resolver_prefers_environment_then_file_then_none() {
    let directory = tempfile::tempdir().expect("tempdir");
    let path = directory.path().join(CLIENT_TOKEN_FILENAME);
    write_token_file(&path, &format!("{}\n", token(2)));

    let resolved = resolve_route_credential_from(Some(OsString::from(token(1))), Some(&path))
        .expect("environment credential");
    assert_eq!(resolved.source, CredentialSource::Environment);
    assert_eq!(resolved.credential.expose(), token(1));

    for environment in [None, Some(OsString::new())] {
        let resolved =
            resolve_route_credential_from(environment, Some(&path)).expect("file credential");
        assert_eq!(resolved.source, CredentialSource::File);
        assert_eq!(resolved.credential.expose(), token(2));
    }

    let missing = directory.path().join("missing");
    assert!(resolve_route_credential_from(None, Some(&missing)).is_none());
    assert!(resolve_route_credential_from(None, None).is_none());
}

#[test]
fn invalid_environment_credential_falls_back_to_the_file() {
    let directory = tempfile::tempdir().expect("tempdir");
    let path = directory.path().join(CLIENT_TOKEN_FILENAME);
    write_token_file(&path, &token(3));

    for invalid in ["short", " padded ", "!invalid!"] {
        let resolved = resolve_route_credential_from(Some(OsString::from(invalid)), Some(&path))
            .expect("file fallback");
        assert_eq!(resolved.source, CredentialSource::File);
        assert_eq!(resolved.credential.expose(), token(3));
    }
    std::fs::remove_file(&path).expect("remove token file");
    assert!(resolve_route_credential_from(Some(OsString::from("short")), Some(&path)).is_none());
}

#[test]
fn token_file_accepts_one_trailing_line_ending_and_rejects_other_contents() {
    let directory = tempfile::tempdir().expect("tempdir");
    let path = directory.path().join(CLIENT_TOKEN_FILENAME);
    for contents in [
        token(4),
        format!("{}\n", token(4)),
        format!("{}\r\n", token(4)),
    ] {
        write_token_file(&path, &contents);
        assert_eq!(
            read_client_token_file(&path)
                .expect("valid file")
                .expect("present")
                .expose(),
            token(4)
        );
    }
    for contents in [
        String::new(),
        format!("{}\n\n", token(4)),
        format!(" {}", token(4)),
        "not-a-token\n".to_owned(),
    ] {
        write_token_file(&path, &contents);
        let error = read_client_token_file(&path).expect_err("invalid contents");
        assert_eq!(error.problem, TokenFileProblem::InvalidContents);
        assert!(!error.to_string().contains(&token(4)), "{error}");
    }
    write_token_file(&path, &"A".repeat(MAX_CLIENT_TOKEN_FILE_BYTES as usize + 1));
    assert_eq!(
        read_client_token_file(&path).unwrap_err().problem,
        TokenFileProblem::TooLarge
    );
    assert!(resolve_route_credential_from(None, Some(&path)).is_none());
}

#[test]
fn token_file_must_be_a_regular_file() {
    let directory = tempfile::tempdir().expect("tempdir");
    let path = directory.path().join(CLIENT_TOKEN_FILENAME);
    std::fs::create_dir(&path).expect("directory in place of file");
    assert_eq!(
        read_client_token_file(&path).unwrap_err().problem,
        TokenFileProblem::NotRegularFile
    );
}

#[cfg(unix)]
#[test]
fn token_file_rejects_group_or_other_permissions_and_symlinks() {
    use std::os::unix::fs::{PermissionsExt, symlink};

    let directory = tempfile::tempdir().expect("tempdir");
    let path = directory.path().join(CLIENT_TOKEN_FILENAME);
    write_token_file(&path, &token(5));
    for mode in [0o640, 0o604, 0o660, 0o644] {
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).expect("chmod");
        let error = read_client_token_file(&path).expect_err("insecure mode");
        assert_eq!(error.problem, TokenFileProblem::InsecurePermissions);
        assert!(error.to_string().contains(&path.display().to_string()));
        assert!(!error.to_string().contains(&token(5)));
        assert!(resolve_route_credential_from(None, Some(&path)).is_none());
    }
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o400)).expect("chmod");
    assert!(
        read_client_token_file(&path)
            .expect("owner read-only")
            .is_some()
    );

    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).expect("chmod");
    let link = directory.path().join("linked-token");
    symlink(&path, &link).expect("symlink");
    assert_eq!(
        read_client_token_file(&link).unwrap_err().problem,
        TokenFileProblem::Symlink
    );
    assert!(resolve_route_credential_from(None, Some(&link)).is_none());
}

#[cfg(unix)]
#[test]
fn token_file_owned_by_another_user_is_rejected() {
    // Ownership cannot be changed without privileges; a root-owned system file stands in for a
    // file planted by another account.
    if current_euid() == 0 {
        return;
    }
    let path = Path::new("/etc/hosts");
    let metadata = std::fs::metadata(path).expect("system file");
    assert!(metadata.is_file());
    let file = std::fs::File::open(path).expect("open system file");
    assert_eq!(
        validate_token_file_handle(&file).unwrap_err(),
        TokenFileProblem::NotOwnedByCurrentUser
    );
}

#[test]
fn ensure_creates_a_private_token_once_and_then_reports_existing() {
    let directory = tempfile::tempdir().expect("tempdir");
    let parent = directory.path().join("config").join("nemo-relay");
    let path = parent.join(CLIENT_TOKEN_FILENAME);

    assert_eq!(
        ensure_client_token_unprivileged(&path).expect("create"),
        EnsureStatus::Created
    );
    let created = std::fs::read_to_string(&path).expect("token contents");
    assert!(created.ends_with('\n'));
    let credential = read_client_token_file(&path)
        .expect("valid token")
        .expect("present");
    assert_eq!(format!("{}\n", credential.expose()), created);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = |path: &Path| std::fs::metadata(path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode(&path), 0o600);
        assert_eq!(mode(&parent), 0o700);
    }

    assert_eq!(
        ensure_client_token_unprivileged(&path).expect("existing"),
        EnsureStatus::Existing
    );
    assert_eq!(std::fs::read_to_string(&path).unwrap(), created);
    let leftovers = std::fs::read_dir(&parent)
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect::<Vec<_>>();
    assert_eq!(leftovers, [std::ffi::OsString::from(CLIENT_TOKEN_FILENAME)]);
}

#[test]
fn ensure_never_overwrites_an_invalid_file_or_reveals_its_contents() {
    let directory = tempfile::tempdir().expect("tempdir");
    let path = directory.path().join(CLIENT_TOKEN_FILENAME);
    let secret_like = "not-a-valid-token-but-still-private";
    write_token_file(&path, secret_like);

    let error = ensure_client_token_unprivileged(&path)
        .expect_err("invalid file")
        .to_string();
    assert!(error.contains(&path.display().to_string()), "{error}");
    assert!(error.contains("left unchanged"), "{error}");
    assert!(!error.contains(secret_like), "{error}");
    assert_eq!(std::fs::read_to_string(&path).unwrap(), secret_like);
}

#[cfg(unix)]
#[test]
fn ensure_never_repairs_an_unsafe_existing_file() {
    use std::os::unix::fs::PermissionsExt;

    let directory = tempfile::tempdir().expect("tempdir");
    let path = directory.path().join(CLIENT_TOKEN_FILENAME);
    write_token_file(&path, &token(6));
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).expect("chmod");

    let error = ensure_client_token_unprivileged(&path)
        .expect_err("unsafe file")
        .to_string();
    assert!(error.contains("accessible by other users"), "{error}");
    assert!(!error.contains(&token(6)), "{error}");
    assert_eq!(
        std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o644
    );
    assert_eq!(std::fs::read_to_string(&path).unwrap(), token(6));
}

#[cfg(unix)]
#[test]
fn ensure_rejects_an_existing_directory_other_users_can_write_or_own() {
    use std::os::unix::fs::PermissionsExt;

    let directory = tempfile::tempdir().expect("tempdir");
    let shared = directory.path().join("shared");
    std::fs::create_dir(&shared).expect("create directory");
    std::fs::set_permissions(&shared, std::fs::Permissions::from_mode(0o775)).expect("chmod");
    let path = shared.join(CLIENT_TOKEN_FILENAME);

    let error = ensure_client_token_unprivileged(&path)
        .expect_err("group-writable directory")
        .to_string();
    assert!(error.contains("writable by other users"), "{error}");
    assert!(!path.exists());
    assert_eq!(
        std::fs::metadata(&shared).unwrap().permissions().mode() & 0o777,
        0o775
    );

    // Ownership cannot be changed without privileges; a root-owned system directory stands in.
    if current_euid() != 0 {
        let foreign = Path::new("/usr").join(format!("{CLIENT_TOKEN_FILENAME}.test"));
        let error = ensure_client_token_unprivileged(&foreign)
            .expect_err("directory owned by another user")
            .to_string();
        assert!(error.contains("not owned by the current user"), "{error}");
        assert!(!foreign.exists());
    }
}

#[test]
fn concurrent_ensure_calls_converge_on_one_token() {
    let directory = tempfile::tempdir().expect("tempdir");
    let path = directory
        .path()
        .join("nemo-relay")
        .join(CLIENT_TOKEN_FILENAME);
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(8));
    let handles = (0..8)
        .map(|_| {
            let path = path.clone();
            let barrier = std::sync::Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                ensure_client_token_unprivileged(&path).expect("ensure")
            })
        })
        .collect::<Vec<_>>();
    let statuses = handles
        .into_iter()
        .map(|handle| handle.join().expect("thread"))
        .collect::<Vec<_>>();
    assert_eq!(
        statuses
            .iter()
            .filter(|status| **status == EnsureStatus::Created)
            .count(),
        1,
        "{statuses:?}"
    );
    assert!(read_client_token_file(&path).expect("valid").is_some());
    let entries = std::fs::read_dir(path.parent().unwrap()).unwrap().count();
    assert_eq!(entries, 1, "temporary files must not be left behind");
}

#[cfg(unix)]
#[test]
fn ensure_refuses_root() {
    let result = refuse_privileged_user();
    if current_euid() == 0 {
        assert!(result.unwrap_err().to_string().contains("as root"));
    } else {
        result.expect("unprivileged user");
    }
}
