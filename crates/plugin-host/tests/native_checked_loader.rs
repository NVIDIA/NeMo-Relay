// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Checked loader integration against independently compiled plugin binaries.

use nemo_relay_plugin_host::dynamic::native_library::{
    CheckedNativeLibrary, NativeLibraryArtifact, resident_library_path,
};
use std::{collections::BTreeMap, path::PathBuf};

fn artifact(name: &str, dependency: &[u8]) -> NativeLibraryArtifact {
    let directory = std::env::var_os("NEMO_RELAY_TEST_NATIVE_ABI_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/native-abi-fixtures/debug")
        });
    let filename = format!(
        "{}nemo_relay_native_abi_fixture_{name}{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    );
    let bytes = std::fs::read(directory.join(&filename))
        .expect("run just build-native-abi-test-fixtures first");
    NativeLibraryArtifact::new(
        PathBuf::from(&filename),
        [
            (PathBuf::from(filename), bytes),
            (PathBuf::from("deps/closure-marker"), dependency.to_vec()),
        ],
    )
    .unwrap()
}

#[test]
fn checked_library_resides_without_retaining_component_state() {
    let library =
        CheckedNativeLibrary::load(artifact("baseline", b"baseline"), "v1.0", "fixture").unwrap();
    assert_eq!(library.descriptor().native_api.to_string(), "v1.0");
    let identity = library.identity();
    let path = library.resident_path().to_owned();
    let plugin = library.create_plugin().unwrap();
    assert!(plugin.allows_multiple_components());
    drop(plugin);
    drop(library);
    assert!(path.is_file());
    assert_eq!(
        std::fs::read(path.parent().unwrap().join("deps/closure-marker")).unwrap(),
        b"baseline"
    );
    let again =
        CheckedNativeLibrary::load(artifact("baseline", b"baseline"), "v1.0", "fixture").unwrap();
    assert_eq!(again.identity(), identity);
    assert_eq!(again.resident_path(), path);
    let different = CheckedNativeLibrary::load(
        artifact("baseline", b"changed dependency"),
        "v1.0",
        "fixture",
    )
    .unwrap();
    assert_ne!(different.identity(), identity);
    assert_ne!(different.resident_path(), path);
    assert!(
        different
            .create_plugin()
            .unwrap()
            .allows_multiple_components()
    );
}

#[test]
fn descriptor_disagreement_does_not_poison_resident_artifact() {
    let error = CheckedNativeLibrary::load(artifact("baseline", b"mismatch"), "v1.0", "wrong-kind")
        .err()
        .unwrap();
    assert!(error.contains("disagrees with binary"), "{error}");
    let valid =
        CheckedNativeLibrary::load(artifact("baseline", b"mismatch"), "v1.0", "fixture").unwrap();
    assert!(valid.create_plugin().unwrap().allows_multiple_components());
    let error = CheckedNativeLibrary::load(artifact("future", b"mismatch"), "v1.0", "fixture")
        .err()
        .unwrap();
    assert!(error.contains("disagrees with binary"), "{error}");
}

#[test]
fn incompatible_layouts_and_unsupported_versions_fail_before_factory() {
    for name in ["signature", "marker", "reordered"] {
        let artifact = artifact(name, b"layout");
        let identity = artifact.identity();
        let error = CheckedNativeLibrary::load(artifact, "v1.0", "fixture")
            .err()
            .unwrap();
        assert!(resident_library_path(identity).unwrap().is_file());
        assert!(
            error.contains("incompatible native module"),
            "{name}: {error}"
        );
    }
    for (version, diagnostic) in [
        ("1", "rebuild"),
        ("v1.1", "host supports v1.0"),
        ("v2.0", "host supports v1.0"),
    ] {
        let artifact = artifact("baseline", b"api");
        let identity = artifact.identity();
        let error = CheckedNativeLibrary::load(artifact, version, "fixture")
            .err()
            .unwrap();
        assert!(resident_library_path(identity).is_none());
        assert!(error.contains(diagnostic), "{error}");
    }
}

#[test]
fn missing_modules_and_incompatible_dependency_generations_are_rejected() {
    for (name, diagnostic) in [
        ("missing", "missing required module"),
        ("dependency", "bootstrap"),
    ] {
        let error = CheckedNativeLibrary::load(
            artifact(name, b"negative-compatibility"),
            "v1.0",
            "fixture",
        )
        .err()
        .unwrap();
        assert!(error.contains(diagnostic), "{name}: {error}");
    }
}

#[test]
fn artifact_paths_cannot_escape_staging_and_identity_includes_complete_closure() {
    for path in ["../escape", "/absolute", "a/../escape", ""] {
        assert!(
            NativeLibraryArtifact::new(PathBuf::from(path), [(PathBuf::from(path), vec![])])
                .is_err()
        );
    }
    assert!(
        NativeLibraryArtifact::new(
            PathBuf::from("library"),
            [
                (PathBuf::from("library"), vec![]),
                (PathBuf::from("library"), vec![])
            ]
        )
        .is_err()
    );
    assert!(NativeLibraryArtifact::new(PathBuf::from("library"), BTreeMap::new()).is_err());
    let one = NativeLibraryArtifact::new(
        PathBuf::from("library"),
        [
            (PathBuf::from("library"), vec![1]),
            (PathBuf::from("dep"), vec![2]),
        ],
    )
    .unwrap();
    let reordered = NativeLibraryArtifact::new(
        PathBuf::from("library"),
        [
            (PathBuf::from("dep"), vec![2]),
            (PathBuf::from("library"), vec![1]),
        ],
    )
    .unwrap();
    assert_eq!(one.identity(), reordered.identity());
}

use nemo_relay_plugin::abi::{
    Json,
    abi_stable::{sabi_trait::prelude::TD_Opaque, std_types::RString},
    async_ffi::{FfiFuture, FutureExt},
    v1_0::{Error, Result as AbiResult, Runtime, Runtime_TO, RuntimeBox},
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Default)]
struct Probe {
    started: AtomicUsize,
    dropped: AtomicUsize,
    runtime_dropped: AtomicUsize,
}
struct ProbeRuntime(Arc<Probe>);
impl Drop for ProbeRuntime {
    fn drop(&mut self) {
        self.0.runtime_dropped.fetch_add(1, Ordering::AcqRel);
    }
}
impl Runtime for ProbeRuntime {
    fn diagnostics(&self) -> nemo_relay_plugin::abi::diagnostics::DiagnosticsBox {
        nemo_relay_plugin::abi::diagnostics::Diagnostics_TO::from_value(EmptyDiagnostics, TD_Opaque)
    }
    fn bind_component(&self, _: nemo_relay_plugin::abi::v1_0::ComponentBox) -> AbiResult<()> {
        AbiResult::RErr(Error::new("fixture", "no component binding"))
    }
    fn register_callback(&self, _: nemo_relay_plugin::abi::v1_0::Registration) -> AbiResult<()> {
        AbiResult::RErr(Error::new("fixture", "no registration"))
    }
    fn capture_scope(&self) -> AbiResult<nemo_relay_plugin::abi::v1_0::ScopeBox> {
        AbiResult::RErr(Error::new("fixture", "no scope"))
    }
    fn register_gate(&self, _: nemo_relay_plugin::abi::v1_0::Registration) -> AbiResult<RString> {
        AbiResult::RErr(Error::new("fixture", "no gate"))
    }

    fn call_sync(&self, operation: RString, _: Json) -> AbiResult<Json> {
        match operation.as_str() {
            "fixture.task_started" => {
                self.0.started.fetch_add(1, Ordering::AcqRel);
            }
            "fixture.task_dropped" => {
                self.0.dropped.fetch_add(1, Ordering::AcqRel);
            }
            _ => return AbiResult::RErr(Error::new("unknown", "unknown fixture operation")),
        }
        AbiResult::ROk(Json::Null)
    }
    fn call(&self, _: RString, _: Json) -> FfiFuture<AbiResult<Json>> {
        async { AbiResult::RErr(Error::new("unknown", "unknown fixture operation")) }.into_ffi()
    }
}
fn runtime(probe: &Arc<Probe>) -> RuntimeBox {
    Runtime_TO::from_value(ProbeRuntime(probe.clone()), TD_Opaque)
}
fn sdk_artifact() -> NativeLibraryArtifact {
    let path = std::env::var_os("NEMO_RELAY_TEST_NATIVE_PLUGIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../target/test-plugin-fixtures/debug")
                .join(format!(
                    "{}nemo_relay_plugin_fixture{}",
                    std::env::consts::DLL_PREFIX,
                    std::env::consts::DLL_SUFFIX
                ))
        });
    let name = PathBuf::from(path.file_name().unwrap());
    NativeLibraryArtifact::new(
        name.clone(),
        [(
            name,
            std::fs::read(path).expect("run just build-test-plugin-fixtures"),
        )],
    )
    .unwrap()
}

#[tokio::test]
async fn sdk_components_shutdown_independently_across_the_library_boundary() {
    // OS image validation is setup, outside the asynchronous lifecycle deadline.
    let library = CheckedNativeLibrary::load(sdk_artifact(), "v1.0", "fixture_native").unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(10), async {
        let path = library.resident_path().to_owned();
        let plugin = library.create_plugin().unwrap();
        let first = Arc::new(Probe::default());
        let second = Arc::new(Probe::default());
        let (one, two) = tokio::join!(
            plugin.register(Json::Null, runtime(&first)),
            plugin.register(Json::Null, runtime(&second))
        );
        let one = one.into_result().unwrap();
        let two = two.into_result().unwrap();
        assert!(one.registrations.is_empty());
        while first.started.load(Ordering::Acquire) != 1
            || second.started.load(Ordering::Acquire) != 1
        {
            tokio::task::yield_now().await;
        }
        drop(plugin);
        drop(library);
        assert!(path.is_file());
        let (a, b) = tokio::join!(one.component.shutdown(), one.component.shutdown());
        a.into_result().unwrap();
        b.into_result().unwrap();
        assert_eq!(first.dropped.load(Ordering::Acquire), 1);
        assert_eq!(first.runtime_dropped.load(Ordering::Acquire), 1);
        assert_eq!(second.dropped.load(Ordering::Acquire), 0);
        two.component.shutdown().await.into_result().unwrap();
        assert_eq!(second.dropped.load(Ordering::Acquire), 1);
        assert_eq!(second.runtime_dropped.load(Ordering::Acquire), 1);
        drop(one);
        drop(two);
        let library = CheckedNativeLibrary::load(sdk_artifact(), "v1.0", "fixture_native").unwrap();
        assert_eq!(library.resident_path(), path);
        let fresh = Arc::new(Probe::default());
        let component = library
            .create_plugin()
            .unwrap()
            .register(Json::Null, runtime(&fresh))
            .await
            .into_result()
            .unwrap();
        component.component.shutdown().await.into_result().unwrap();
        assert_eq!(fresh.dropped.load(Ordering::Acquire), 1);
        assert_eq!(fresh.runtime_dropped.load(Ordering::Acquire), 1);
    })
    .await
    .expect("component teardown must not block its own executor");
}

#[tokio::test]
async fn dropping_unpolled_registration_releases_foreign_component_and_runtime() {
    // OS image validation is setup, outside the asynchronous lifecycle deadline.
    let library = CheckedNativeLibrary::load(sdk_artifact(), "v1.0", "fixture_native").unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(10), async {
        let plugin = library.create_plugin().unwrap();
        for _ in 0..20 {
            let probe = Arc::new(Probe::default());
            let registration = plugin.register(Json::Null, runtime(&probe));
            // Registration work has already been admitted in the SDK library,
            // but no foreign poll has happened in this host library.
            drop(registration);
            while probe.runtime_dropped.load(Ordering::Acquire) != 1 {
                tokio::task::yield_now().await;
            }
            assert_eq!(probe.dropped.load(Ordering::Acquire), 1);
        }
    })
    .await
    .expect("abandoned registration must release its component executor");
}

struct EmptyDiagnostics;
impl nemo_relay_plugin::abi::diagnostics::Diagnostics for EmptyDiagnostics {
    fn capabilities(
        &self,
    ) -> nemo_relay_plugin::abi::abi_stable::std_types::RResult<
        nemo_relay_plugin::abi::diagnostics::DiagnosticCapabilities,
        u32,
    > {
        Err(2).into()
    }
    fn register(
        &self,
        _: nemo_relay_plugin::abi::diagnostics::DiagnosticRegistration<'_>,
    ) -> nemo_relay_plugin::abi::abi_stable::std_types::RResult<u64, u32> {
        Err(2).into()
    }
    fn record(
        &self,
        _: u64,
        _: nemo_relay_plugin::abi::diagnostics::DiagnosticObservation<'_>,
    ) -> nemo_relay_plugin::abi::abi_stable::std_types::RResult<
        nemo_relay_plugin::abi::diagnostics::DiagnosticReceipt,
        u32,
    > {
        Err(2).into()
    }
    fn close(&self, _: u64) -> nemo_relay_plugin::abi::abi_stable::std_types::RResult<(), u32> {
        Err(2).into()
    }
    fn read(
        &self,
        _: u32,
        _: nemo_relay_plugin::abi::abi_stable::std_types::RStr<'_>,
        _: nemo_relay_plugin::abi::abi_stable::std_types::RSliceMut<'_, u8>,
    ) -> nemo_relay_plugin::abi::abi_stable::std_types::RResult<usize, u32> {
        Err(2).into()
    }
}
