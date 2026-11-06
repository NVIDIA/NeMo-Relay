// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Binary checks against separately compiled, frozen interface generations.

extern crate nemo_relay_plugin_abi as abi;
#[path = "fixtures/behavior_host.rs"]
mod behavior;

use abi_stable::library::{LibHeader, LibraryError, lib_header_from_path};
use nemo_relay_plugin_abi::{Bootstrap_Ref, NativeApiVersion as Api, v1_0, validate_descriptor};
use std::path::PathBuf;

fn fixture_path(name: &str) -> PathBuf {
    let directory = std::env::var_os("NEMO_RELAY_TEST_NATIVE_ABI_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/native-abi-fixtures/debug")
        });
    let path = directory.join(format!(
        "{}nemo_relay_native_abi_fixture_{name}{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    ));
    assert!(
        path.is_file(),
        "{} is missing; run just build-native-abi-test-fixtures before these tests",
        path.display()
    );
    path
}

fn fixture(name: &str) -> &'static LibHeader {
    lib_header_from_path(&fixture_path(name))
        .expect("fixture must export a compatible abi_stable header")
}

#[test]
fn v1_0_plugin_loads_on_current_and_future_hosts() {
    let header = fixture("baseline");
    let module = header.init_root_module::<Bootstrap_Ref>().unwrap();
    let descriptor = (module.descriptor())();
    for host in [Api::new(1, 0), Api::new(1, 1)] {
        validate_descriptor(&descriptor, Api::new(1, 0), "fixture", host).unwrap();
        let factory = (module.query_module())(v1_0::FACTORY_MODULE.into())
            .into_option()
            .unwrap();
        let factory = factory
            .get()
            .init_root_module::<v1_0::Factory_Ref>()
            .unwrap();
        let plugin = (factory.create())().into_result().unwrap();
        assert!(plugin.allows_multiple_components());
        assert!(
            plugin
                .validate(nemo_relay_plugin_abi::Json::Null)
                .is_empty()
        );
        behavior::assert_behavior(plugin);
    }
    assert_future_host_accepts("baseline");
}

#[test]
fn newer_sdk_requires_newer_host_and_selects_its_own_module() {
    let module = fixture("future")
        .init_root_module::<Bootstrap_Ref>()
        .unwrap();
    let descriptor = (module.descriptor())();
    assert!(validate_descriptor(&descriptor, Api::new(1, 1), "fixture", Api::new(1, 0)).is_err());
    validate_descriptor(&descriptor, Api::new(1, 1), "fixture", Api::new(1, 1)).unwrap();
    assert_future_host_accepts("future");
}

/// Unchanged v1.0 contracts load in both directions across Cargo package major and prerelease versions.
#[test]
fn package_release_versions_do_not_define_binary_compatibility() {
    let bootstrap = fixture("package_next")
        .init_root_module::<Bootstrap_Ref>()
        .unwrap();
    let descriptor = (bootstrap.descriptor())();
    validate_descriptor(&descriptor, Api::new(1, 0), "fixture", Api::new(1, 0)).unwrap();
    let module = (bootstrap.query_module())(v1_0::FACTORY_MODULE.into())
        .into_option()
        .unwrap();
    let factory = module
        .get()
        .init_root_module::<v1_0::Factory_Ref>()
        .unwrap();
    behavior::assert_behavior((factory.create())().into_result().unwrap());
    assert_future_host_accepts("package_next");
    let directory = fixture_path("baseline").parent().unwrap().to_owned();
    let host = directory.join(format!(
        "nemo-relay-native-abi-fixture-package-next-host{}",
        std::env::consts::EXE_SUFFIX
    ));
    for plugin in ["baseline", "package_next"] {
        let output = std::process::Command::new(&host)
            .arg(fixture_path(plugin))
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "package-version host failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn reject_different_major_and_missing_module_before_factory_use() {
    let module = fixture("major")
        .init_root_module::<Bootstrap_Ref>()
        .unwrap();
    let descriptor = (module.descriptor())();
    assert!(validate_descriptor(&descriptor, Api::new(2, 0), "fixture", Api::new(1, 0)).is_err());
    assert!((module.query_module())("not.a.published.module".into()).is_none());
}

#[test]
fn reject_changed_function_signature_and_marker_traits() {
    for name in ["signature", "marker", "reordered"] {
        let module = fixture(name).init_root_module::<Bootstrap_Ref>().unwrap();
        let factory = (module.query_module())(v1_0::FACTORY_MODULE.into())
            .into_option()
            .unwrap();
        assert!(
            factory
                .get()
                .init_root_module::<v1_0::Factory_Ref>()
                .is_err(),
            "{name} must be rejected by layout checking"
        );
    }
}

#[test]
fn incompatible_dependency_generation_fails_before_module_discovery() {
    assert!(matches!(
        lib_header_from_path(&fixture_path("dependency")),
        Err(LibraryError::InvalidAbiHeader { .. })
    ));
}

#[test]
fn declared_contract_requires_its_factory_module() {
    let module = fixture("missing")
        .init_root_module::<Bootstrap_Ref>()
        .unwrap();
    let descriptor = (module.descriptor())();
    validate_descriptor(&descriptor, Api::new(1, 0), "fixture", Api::new(1, 0)).unwrap();
    assert!((module.query_module())(v1_0::FACTORY_MODULE.into()).is_none());
}

#[test]
fn independent_loading_does_not_return_the_first_library_for_every_path() {
    let old = fixture("baseline")
        .init_root_module::<Bootstrap_Ref>()
        .unwrap();
    let new = fixture("future")
        .init_root_module::<Bootstrap_Ref>()
        .unwrap();
    assert_eq!((old.descriptor())().native_api, Api::new(1, 0));
    assert_eq!((new.descriptor())().native_api, Api::new(1, 1));
}

fn assert_future_host_accepts(name: &str) {
    let directory = std::env::var_os("NEMO_RELAY_TEST_NATIVE_ABI_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/native-abi-fixtures/debug")
        });
    let host = directory.join(format!(
        "nemo-relay-native-abi-fixture-future-host{}",
        std::env::consts::EXE_SUFFIX
    ));
    let plugin = directory.join(format!(
        "{}nemo_relay_native_abi_fixture_{name}{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    ));
    let output = std::process::Command::new(host)
        .arg(plugin)
        .output()
        .expect("build the future host fixture before running tests");
    assert!(
        output.status.success(),
        "future host failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}
