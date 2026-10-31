// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Coverage tests for shared runtime in the NeMo Relay core crate.

use super::*;

use crate::api::scope::get_handle;

fn acquire_test_lock() -> std::sync::MutexGuard<'static, ()> {
    runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn owner_token_for(pid: u32, binding_kind: &str, version: &str) -> String {
    format!("pid={pid};binding={binding_kind};version={version}")
}

#[test]
fn test_initialize_binding_claims_process_owner() {
    let _lock = acquire_test_lock();
    reset_runtime_owner_for_tests();

    initialize_shared_runtime_binding("python").unwrap();

    let owner = read_process_runtime_owner().unwrap().unwrap();
    assert_eq!(owner.pid, std::process::id());
    assert_eq!(owner.binding_kind, "python");
    assert_eq!(
        owner.major_version,
        current_compatibility_version().unwrap().to_string()
    );
}

#[test]
fn test_same_binding_same_major_reuses_process_owner() {
    let _lock = acquire_test_lock();
    reset_runtime_owner_for_tests();

    let version = format!("{}.999.999", current_compatibility_version().unwrap());
    unsafe {
        std::env::set_var(
            OWNER_TOKEN_ENV,
            owner_token_for(std::process::id(), "python", &version),
        )
    };

    initialize_shared_runtime_binding("python").unwrap();

    let owner = read_process_runtime_owner().unwrap().unwrap();
    assert_eq!(owner.binding_kind, "python");
    assert_eq!(
        owner.major_version,
        current_compatibility_version().unwrap().to_string()
    );
}

#[test]
fn test_conflicting_binding_is_rejected() {
    let _lock = acquire_test_lock();
    reset_runtime_owner_for_tests();

    unsafe {
        std::env::set_var(
            OWNER_TOKEN_ENV,
            owner_token_for(
                std::process::id(),
                "python",
                current_compatibility_version().unwrap(),
            ),
        )
    };

    let error = initialize_shared_runtime_binding("node").unwrap_err();
    let message = error.to_string();
    assert!(message.contains("NeMo Relay does not support multiple bindings in one process"));
    assert!(message.contains("existing owner=python@"));
    assert!(message.contains("attempted=node@"));
}

#[test]
fn test_stale_pid_owner_is_replaced() {
    let _lock = acquire_test_lock();
    reset_runtime_owner_for_tests();

    unsafe {
        std::env::set_var(
            OWNER_TOKEN_ENV,
            owner_token_for(
                std::process::id() + 1,
                "python",
                current_compatibility_version().unwrap(),
            ),
        )
    };

    initialize_shared_runtime_binding("node").unwrap();

    let owner = read_process_runtime_owner().unwrap().unwrap();
    assert_eq!(owner.pid, std::process::id());
    assert_eq!(owner.binding_kind, "node");
}

#[test]
fn test_api_use_claims_default_rust_owner() {
    let _lock = acquire_test_lock();
    reset_runtime_owner_for_tests();

    let handle = get_handle().unwrap();
    assert_eq!(handle.name, "root");

    let owner = read_process_runtime_owner().unwrap().unwrap();
    assert_eq!(owner.binding_kind, "rust");
}

#[test]
fn test_api_use_rejects_conflicting_owner() {
    let _lock = acquire_test_lock();
    reset_runtime_owner_for_tests();

    unsafe {
        std::env::set_var(
            OWNER_TOKEN_ENV,
            owner_token_for(
                std::process::id(),
                "python",
                current_compatibility_version().unwrap(),
            ),
        )
    };

    let error = get_handle().unwrap_err();
    let message = error.to_string();
    assert!(message.contains("NeMo Relay does not support multiple bindings in one process"));
    assert!(message.contains("existing owner=python@"));
    assert!(message.contains("attempted=rust@"));
}

#[test]
fn test_runtime_owner_parse_and_display_cover_invalid_tokens() {
    let _lock = acquire_test_lock();
    reset_runtime_owner_for_tests();

    let owner = RuntimeOwner::current("rust".into()).unwrap();
    assert!(owner.to_string().contains("rust@"));
    assert!(owner.token().contains("binding=rust"));
    assert!(owner.same_owner(&RuntimeOwner::parse(&owner.token()).unwrap()));

    let invalid_pid = RuntimeOwner::parse("pid=abc;binding=rust;version=1.2.3").unwrap_err();
    assert!(
        invalid_pid
            .to_string()
            .contains("invalid NeMo Relay owner token pid")
    );

    let empty_binding = RuntimeOwner::parse("pid=1;binding=;version=1.2.3").unwrap_err();
    assert!(empty_binding.to_string().contains("binding kind is empty"));

    let missing_pid = RuntimeOwner::parse("binding=rust;version=1.2.3").unwrap_err();
    assert!(missing_pid.to_string().contains("missing pid"));

    let missing_binding = RuntimeOwner::parse("pid=1;version=1.2.3").unwrap_err();
    assert!(missing_binding.to_string().contains("missing binding"));

    let missing_version = RuntimeOwner::parse("pid=1;binding=rust").unwrap_err();
    assert!(missing_version.to_string().contains("missing version"));
}

#[test]
fn test_version_validation_binding_resolution_and_invalid_owner_cleanup() {
    let _lock = acquire_test_lock();
    reset_runtime_owner_for_tests();

    assert_eq!(compatibility_major_version("1.2.3").unwrap(), "1");
    assert!(compatibility_major_version("").is_err());
    assert!(compatibility_major_version("beta").is_err());

    unsafe { std::env::set_var(BINDING_KIND_ENV, "node") };
    assert_eq!(resolve_binding_kind(None), "node");
    unsafe { std::env::remove_var(BINDING_KIND_ENV) };
    assert_eq!(resolve_binding_kind(Some("python".into())), "python");
    assert_eq!(resolve_binding_kind(None), "rust");

    unsafe { std::env::set_var(OWNER_TOKEN_ENV, "pid=broken;binding=rust;version=1.2.3") };
    assert!(read_process_runtime_owner().unwrap().is_none());
    assert!(std::env::var(OWNER_TOKEN_ENV).is_err());
}

#[test]
fn test_initialize_binding_rejects_second_binding_identity() {
    let _lock = acquire_test_lock();
    reset_runtime_owner_for_tests();

    initialize_shared_runtime_binding("python").unwrap();

    let error = initialize_shared_runtime_binding("node").unwrap_err();
    let message = error.to_string();
    assert!(message.contains("already initialized as python"));
    assert!(message.contains("attempted=node"));
}
