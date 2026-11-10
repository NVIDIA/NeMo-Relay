// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Binary checks against separately compiled, frozen interface generations.
extern crate nemo_relay_plugin_abi as abi;
#[path = "fixtures/behavior_host.rs"]
mod behavior;
use abi::{NativeApiVersion as Api, PluginDescriptor, bootstrap, validate_descriptor, xabi};
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
        "{} is missing; run just build-native-abi-test-fixtures",
        path.display()
    );
    path
}
fn raw_fixture(name: &str) -> *const bootstrap::Bootstrap {
    let library = unsafe { xabi::ModuleHandle::load(fixture_path(name)) }.unwrap();
    let symbol = unsafe {
        library.get::<unsafe extern "C" fn() -> *const bootstrap::Bootstrap>(bootstrap::SYMBOL)
    }
    .unwrap();
    let raw = unsafe { symbol() };
    std::mem::forget(library);
    raw
}
fn fixture(name: &str) -> Result<(PluginDescriptor, &'static xabi::XabiManifest), String> {
    unsafe { bootstrap::check(raw_fixture(name)) }
}
fn assert_host_accepts(host: &str, name: &str) {
    let directory = fixture_path(name).parent().unwrap().to_owned();
    let host = directory.join(format!(
        "nemo-relay-native-abi-fixture-{host}{}",
        std::env::consts::EXE_SUFFIX
    ));
    let output = std::process::Command::new(host)
        .arg(fixture_path(name))
        .output()
        .expect("build the host fixture first");
    assert!(
        output.status.success(),
        "host failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}
#[test]
fn v1_0_plugin_loads_on_current_and_future_hosts() {
    let (descriptor, manifest) = fixture("baseline").unwrap();
    for host in [Api::new(1, 0), Api::new(1, 1)] {
        validate_descriptor(&descriptor, Api::new(1, 0), "fixture", host).unwrap();
        let factory = unsafe { bootstrap::factory(manifest) }.unwrap();
        let plugin = factory.create().unwrap();
        assert!(plugin.allows_multiple_components());
        assert!(plugin.validate(abi::Json::Null).is_empty());
        behavior::assert_behavior(plugin);
    }
    assert_host_accepts("future-host", "baseline");
}
#[test]
fn newer_sdk_requires_newer_host_and_selects_its_own_module() {
    let error = fixture("future").err().unwrap();
    assert!(error.contains("requires native API v1.1"), "{error}");
    assert_host_accepts("future-host", "future");
}
#[test]
fn package_release_versions_do_not_define_binary_compatibility() {
    let (descriptor, manifest) = fixture("package_next").unwrap();
    validate_descriptor(&descriptor, Api::new(1, 0), "fixture", Api::new(1, 0)).unwrap();
    let factory = unsafe { bootstrap::factory(manifest) }.unwrap();
    behavior::assert_behavior(factory.create().unwrap());
    assert_host_accepts("future-host", "package_next");
    for plugin in ["baseline", "package_next"] {
        assert_host_accepts("package-next-host", plugin);
    }
}
#[test]
fn reject_different_major_and_missing_module_before_factory_use() {
    let error = fixture("major").err().unwrap();
    assert!(error.contains("requires native API v2.0"), "{error}");
    let (_, manifest) = fixture("missing").unwrap();
    assert!(unsafe { bootstrap::factory(manifest) }.is_err());
}
#[test]
fn reject_changed_function_signature_and_marker_traits() {
    for name in ["signature", "marker"] {
        let error = fixture(name)
            .err()
            .expect("incompatible binary must fail before construction");
        assert!(
            error.contains("incompatible native contract"),
            "{name}: {error}"
        );
    }
}
#[test]
fn incompatible_dependency_generation_fails_before_module_discovery() {
    let error = fixture("dependency").err().unwrap();
    assert!(error.contains("dependency ABI"), "{error}");
}
#[test]
fn declared_contract_requires_its_factory_module() {
    let (descriptor, manifest) = fixture("missing").unwrap();
    validate_descriptor(&descriptor, Api::new(1, 0), "fixture", Api::new(1, 0)).unwrap();
    let error = unsafe { bootstrap::factory(manifest) }.err().unwrap();
    assert!(error.contains("missing required factory module"), "{error}");
}
#[test]
fn independent_loading_does_not_return_the_first_library_for_every_path() {
    let old = unsafe { &*raw_fixture("baseline") };
    let new = unsafe { &*raw_fixture("future") };
    assert_eq!((old.major, old.minor), (1, 0));
    assert_eq!((new.major, new.minor), (1, 1));
}

#[test]
fn changed_generated_field_offsets_fail_even_when_source_size_and_alignment_match() {
    let baseline = unsafe { &*raw_fixture("baseline") };
    let reordered = unsafe { &*raw_fixture("reordered") };
    let baseline_schema = unsafe { baseline.schema.as_str() }.unwrap();
    let reordered_schema = unsafe { reordered.schema.as_str() }.unwrap();
    // The macro invocations and Rust authoring declarations remain identical.
    assert_eq!(
        baseline_schema.split_once("\npointer:").unwrap().0,
        reordered_schema.split_once("\npointer:").unwrap().0,
    );
    let record = |schema: &str| {
        schema
            .split(';')
            .find(|entry| entry.starts_with("FieldWire:"))
            .unwrap()
            .to_owned()
    };
    assert_eq!(record(baseline_schema), record(reordered_schema));
    assert_ne!(baseline_schema, reordered_schema);
    let error = unsafe { bootstrap::check(reordered) }.err().unwrap();
    assert!(error.contains("incompatible native contract"), "{error}");
}
