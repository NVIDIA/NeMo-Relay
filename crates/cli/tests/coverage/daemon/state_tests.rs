// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn route_credential_is_exactly_256_bits() {
    let value = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([7_u8; 32]);
    let credential = RouteCredential::parse(value.clone()).unwrap();
    assert_eq!(credential.expose(), value);
    assert!(RouteCredential::parse("short".into()).is_err());
    assert!(RouteCredential::parse(format!(" {value}")).is_err());
    assert!(RouteCredential::parse("!invalid!".into()).is_err());
    assert!(RouteCredential::parse(format!("{value}\n")).is_err());
}

#[cfg(unix)]
#[test]
fn identity_and_lock_files_reject_symlinks_and_repair_owner_private_modes() {
    use std::os::unix::fs::{PermissionsExt, symlink};

    let directory = tempfile::tempdir().expect("tempdir");
    let identity = directory.path().join("identity.pk8");
    load_or_create_identity(&identity).expect("create identity");
    std::fs::set_permissions(&identity, std::fs::Permissions::from_mode(0o644))
        .expect("loosen identity mode");
    load_or_create_identity(&identity).expect("repair identity mode");
    assert_eq!(
        std::fs::metadata(&identity)
            .expect("identity metadata")
            .permissions()
            .mode()
            & 0o777,
        0o600
    );

    let target = directory.path().join("target");
    std::fs::write(&target, b"not-an-identity").expect("target");
    let linked_identity = directory.path().join("linked.pk8");
    symlink(&target, &linked_identity).expect("identity symlink");
    assert!(load_or_create_identity(&linked_identity).is_err());

    let lock_target = directory.path().join("lock-target");
    std::fs::write(&lock_target, b"").expect("lock target");
    let lock_identity = directory.path().join("lock-linked.pk8");
    symlink(&lock_target, lock_identity.with_extension("lock")).expect("lock symlink");
    assert!(load_or_create_identity(&lock_identity).is_err());
}

#[test]
fn active_generation_survives_restart_but_revoked_generation_does_not() {
    let directory = tempfile::tempdir().expect("tempdir");
    let path = directory.path().join(ACTIVE_WORKER_GENERATIONS_FILENAME);
    let fingerprint = MachineIdentity::generate()
        .expect("machine identity")
        .identity
        .fingerprint();
    let generations =
        ActiveWorkerGenerations::load_for_test(path.clone()).expect("load empty state");

    assert!(!generations.matches(fingerprint, "generation-one").unwrap());
    assert_eq!(
        generations
            .publish(fingerprint, "generation-one", Some("activation-one"))
            .unwrap(),
        None
    );
    assert!(generations.matches(fingerprint, "generation-one").unwrap());
    assert!(!generations.matches(fingerprint, "generation-two").unwrap());

    let reloaded = ActiveWorkerGenerations::load_for_test(path.clone()).expect("reload state");
    assert!(reloaded.matches(fingerprint, "generation-one").unwrap());
    assert_eq!(
        reloaded
            .launch_activation_id(fingerprint, "generation-one")
            .unwrap(),
        Some("activation-one".into())
    );
    assert!(
        reloaded
            .publish(fingerprint, "generation-two", None)
            .unwrap()
            .is_some()
    );
    assert!(
        !reloaded
            .revoke_if_matches(fingerprint, "generation-one")
            .unwrap()
    );
    assert!(reloaded.matches(fingerprint, "generation-two").unwrap());
    assert!(
        reloaded
            .revoke_if_matches(fingerprint, "generation-two")
            .unwrap()
    );

    let after_revoke = ActiveWorkerGenerations::load_for_test(path).expect("reload revoked state");
    assert!(!after_revoke.matches(fingerprint, "generation-two").unwrap());
}

#[test]
fn active_generation_state_without_activation_id_remains_compatible() {
    let directory = tempfile::tempdir().expect("tempdir");
    let path = directory.path().join(ACTIVE_WORKER_GENERATIONS_FILENAME);
    let fingerprint = MachineIdentity::generate()
        .expect("machine identity")
        .identity
        .fingerprint();
    let legacy = serde_json::json!({
        "schema_version": ACTIVE_WORKER_GENERATIONS_SCHEMA_VERSION,
        "generations": [{
            "fingerprint": fingerprint,
            "generation_id": "legacy-generation"
        }]
    });
    std::fs::write(&path, serde_json::to_vec(&legacy).unwrap()).expect("write legacy state");

    let generations = ActiveWorkerGenerations::load_for_test(path).expect("load legacy state");
    assert!(
        generations
            .matches(fingerprint, "legacy-generation")
            .unwrap()
    );
    assert_eq!(
        generations
            .launch_activation_id(fingerprint, "legacy-generation")
            .unwrap(),
        None
    );
}

#[test]
fn active_worker_generation_restore_is_compare_and_set() {
    let directory = tempfile::tempdir().expect("tempdir");
    let path = directory.path().join(ACTIVE_WORKER_GENERATIONS_FILENAME);
    let fingerprint = MachineIdentity::generate()
        .expect("machine identity")
        .identity
        .fingerprint();
    let generations = ActiveWorkerGenerations::load_for_test(path).expect("load state");
    generations
        .publish(fingerprint, "generation-old", Some("activation-old"))
        .expect("publish old generation");
    let previous = generations
        .publish(
            fingerprint,
            "generation-candidate",
            Some("activation-candidate"),
        )
        .expect("publish candidate");

    assert!(
        !generations
            .restore_if_matches(fingerprint, "different-candidate", previous.as_ref(),)
            .unwrap()
    );
    assert!(
        generations
            .restore_if_matches(fingerprint, "generation-candidate", previous.as_ref(),)
            .unwrap()
    );
    assert!(generations.matches(fingerprint, "generation-old").unwrap());
    assert_eq!(
        generations
            .launch_activation_id(fingerprint, "generation-old")
            .unwrap(),
        Some("activation-old".into())
    );
    assert!(
        generations
            .restore_if_matches(fingerprint, "generation-old", None)
            .unwrap()
    );
    assert!(!generations.matches(fingerprint, "generation-old").unwrap());
}

#[test]
fn active_worker_generation_ids_are_validated_on_every_public_operation() {
    let directory = tempfile::tempdir().expect("tempdir");
    let generations = ActiveWorkerGenerations::load_for_test(
        directory.path().join(ACTIVE_WORKER_GENERATIONS_FILENAME),
    )
    .expect("load generation state");
    let fingerprint = MachineIdentity::generate().unwrap().identity.fingerprint();
    for invalid in ["", &"x".repeat(MAX_GENERATION_ID_BYTES + 1)] {
        assert!(generations.matches(fingerprint, invalid).is_err());
        assert!(generations.publish(fingerprint, invalid, None).is_err());
        assert!(generations.revoke_if_matches(fingerprint, invalid).is_err());
        assert!(
            generations
                .restore_if_matches(fingerprint, invalid, None)
                .is_err()
        );
    }
    generations.publish(fingerprint, "current", None).unwrap();
    let invalid_previous = ActiveWorkerGeneration {
        generation_id: "x".repeat(MAX_GENERATION_ID_BYTES + 1),
        activation_id: None,
    };
    assert!(
        generations
            .restore_if_matches(fingerprint, "current", Some(&invalid_previous))
            .is_err()
    );
    assert!(generations.publish(fingerprint, "next", Some("")).is_err());
}

#[test]
fn identity_and_daemon_pin_state_fail_closed_on_corruption_or_replacement() {
    let directory = tempfile::tempdir().expect("tempdir");
    let invalid = directory.path().join("invalid.pk8");
    std::fs::write(&invalid, b"not a private key").unwrap();
    assert!(load_or_create_identity(&invalid).is_err());

    let oversized = directory.path().join("oversized.pk8");
    std::fs::File::create(&oversized)
        .unwrap()
        .set_len(MAX_IDENTITY_BYTES + 1)
        .unwrap();
    assert!(load_or_create_identity(&oversized).is_err());

    let _environment = crate::test_support::EnvScope::set(&[(
        "XDG_CONFIG_HOME",
        Some(directory.path().as_os_str()),
    )]);
    let first = MachineIdentity::generate()
        .unwrap()
        .identity
        .public_identity();
    let second = MachineIdentity::generate()
        .unwrap()
        .identity
        .public_identity();
    verify_or_store_daemon_pin("https://daemon.example:443", first).unwrap();
    verify_or_store_daemon_pin("https://daemon.example:443", first).unwrap();
    assert!(verify_or_store_daemon_pin("https://daemon.example:443", second).is_err());
    load_or_create_machine_identity().unwrap();
    load_or_create_daemon_identity().unwrap();
}

#[cfg(unix)]
#[test]
fn private_state_helpers_reject_non_directories_and_non_regular_files() {
    use std::os::unix::fs::symlink;

    let directory = tempfile::tempdir().expect("tempdir");
    let ordinary = directory.path().join("ordinary");
    std::fs::write(&ordinary, b"value").unwrap();
    assert!(create_private_directory(&ordinary).is_err());

    let linked = directory.path().join("linked-directory");
    symlink(directory.path(), &linked).unwrap();
    assert!(create_private_directory(&linked).is_err());

    let opened_directory = std::fs::File::open(directory.path()).unwrap();
    assert!(validate_private_file(&opened_directory, directory.path()).is_err());
    assert!(
        read_bounded(&directory.path().join("missing"), 8, "missing")
            .unwrap()
            .is_none()
    );
    assert!(read_bounded(directory.path(), 8, "directory").is_err());
}

#[test]
fn active_worker_generation_state_corruption_fails_closed() {
    let directory = tempfile::tempdir().expect("tempdir");
    let path = directory.path().join(ACTIVE_WORKER_GENERATIONS_FILENAME);
    std::fs::write(&path, b"{").expect("write corrupt state");

    let error = ActiveWorkerGenerations::load_for_test(path)
        .expect_err("corrupt state must fail")
        .to_string();

    assert!(error.contains("corrupt"), "{error}");
}

#[test]
fn active_worker_generation_state_enforces_its_file_bound() {
    let directory = tempfile::tempdir().expect("tempdir");
    let path = directory.path().join(ACTIVE_WORKER_GENERATIONS_FILENAME);
    let file = std::fs::File::create(&path).expect("create oversized state");
    file.set_len(MAX_ACTIVE_WORKER_GENERATIONS_BYTES + 1)
        .expect("extend oversized state");

    let error = ActiveWorkerGenerations::load_for_test(path)
        .expect_err("oversized state must fail")
        .to_string();

    assert!(error.contains("exceeds"), "{error}");
}

#[test]
fn active_worker_generation_state_rejects_duplicate_routes_and_unknown_schema() {
    let directory = tempfile::tempdir().expect("tempdir");
    let path = directory.path().join(ACTIVE_WORKER_GENERATIONS_FILENAME);
    let fingerprint = MachineIdentity::generate()
        .expect("machine identity")
        .identity
        .fingerprint();
    let duplicate = serde_json::json!({
        "schema_version": ACTIVE_WORKER_GENERATIONS_SCHEMA_VERSION,
        "generations": [
            {"fingerprint": fingerprint, "generation_id": "generation-one"},
            {"fingerprint": fingerprint, "generation_id": "generation-two"}
        ]
    });
    std::fs::write(&path, serde_json::to_vec(&duplicate).unwrap()).expect("write duplicate state");
    assert!(
        ActiveWorkerGenerations::load_for_test(path.clone())
            .unwrap_err()
            .to_string()
            .contains("duplicate fingerprint")
    );

    let too_many = serde_json::json!({
        "schema_version": ACTIVE_WORKER_GENERATIONS_SCHEMA_VERSION,
        "generations": (0..=MAX_ACTIVE_WORKER_GENERATIONS)
            .map(|index| {
                let mut distinct = [0_u8; 32];
                distinct[..std::mem::size_of::<usize>()]
                    .copy_from_slice(&index.to_le_bytes());
                serde_json::json!({
                    "fingerprint": distinct,
                    "generation_id": format!("generation-{index}")
                })
            })
            .collect::<Vec<_>>()
    });
    std::fs::write(&path, serde_json::to_vec(&too_many).unwrap()).expect("write oversized map");
    assert!(
        ActiveWorkerGenerations::load_for_test(path.clone())
            .unwrap_err()
            .to_string()
            .contains("routes")
    );

    let unknown = serde_json::json!({"schema_version": 2, "generations": []});
    std::fs::write(&path, serde_json::to_vec(&unknown).unwrap()).expect("write unknown state");
    assert!(
        ActiveWorkerGenerations::load_for_test(path)
            .unwrap_err()
            .to_string()
            .contains("unsupported schema version")
    );
}

#[cfg(unix)]
#[test]
fn active_worker_generation_state_is_owner_private_and_rejects_symlinks() {
    use std::os::unix::fs::{PermissionsExt, symlink};

    let directory = tempfile::tempdir().expect("tempdir");
    let path = directory.path().join(ACTIVE_WORKER_GENERATIONS_FILENAME);
    let fingerprint = MachineIdentity::generate()
        .expect("machine identity")
        .identity
        .fingerprint();
    let generations =
        ActiveWorkerGenerations::load_for_test(path.clone()).expect("load generation state");
    generations
        .publish(fingerprint, "generation", None)
        .expect("publish generation");
    assert_eq!(
        std::fs::metadata(&path)
            .expect("state metadata")
            .permissions()
            .mode()
            & 0o777,
        0o600
    );
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644))
        .expect("loosen state mode");
    ActiveWorkerGenerations::load_for_test(path.clone()).expect("repair state mode");
    assert_eq!(
        std::fs::metadata(&path)
            .expect("repaired state metadata")
            .permissions()
            .mode()
            & 0o777,
        0o600
    );

    let target = directory.path().join("target.json");
    std::fs::write(&target, b"{}").expect("write target");
    let linked = directory.path().join("linked.json");
    symlink(&target, &linked).expect("create state symlink");
    assert!(ActiveWorkerGenerations::load_for_test(linked).is_err());
}
