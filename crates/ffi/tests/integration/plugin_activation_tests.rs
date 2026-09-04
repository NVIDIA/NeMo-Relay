// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::ptr;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use nemo_relay_ffi::types::{FfiPluginHostActivation, nemo_relay_plugin_host_activation_free};
use tempfile::TempDir;

const DISCOVERY_CHILD_ENV: &str = "NEMO_RELAY_FFI_DISCOVERY_CHILD";
const DISCOVERED_STATIC_PLUGIN_KIND: &str = "ffi_discovered_static";
static DISCOVERED_STATIC_REGISTRATIONS: AtomicUsize = AtomicUsize::new(0);
static DISCOVERED_STATIC_CALLBACKS: AtomicUsize = AtomicUsize::new(0);
static DISCOVERED_STATIC_CONFIG: Mutex<Option<Json>> = Mutex::new(None);

struct PluginDiscoveryTestEnv {
    previous_cwd: PathBuf,
    previous_xdg_config_home: Option<std::ffi::OsString>,
}

impl PluginDiscoveryTestEnv {
    fn enter(cwd: &Path, xdg_config_home: &Path) -> Self {
        let guard = Self {
            previous_cwd: std::env::current_dir().expect("current directory"),
            previous_xdg_config_home: std::env::var_os("XDG_CONFIG_HOME"),
        };
        std::env::set_current_dir(cwd).expect("set project directory");
        // SAFETY: this runs in a dedicated child test process and Drop restores
        // the environment before that process exits.
        unsafe { std::env::set_var("XDG_CONFIG_HOME", xdg_config_home) };
        guard
    }
}

impl Drop for PluginDiscoveryTestEnv {
    fn drop(&mut self) {
        let _ = std::env::set_current_dir(&self.previous_cwd);
        // SAFETY: see PluginDiscoveryTestEnv::enter.
        unsafe {
            match &self.previous_xdg_config_home {
                Some(value) => std::env::set_var("XDG_CONFIG_HOME", value),
                None => std::env::remove_var("XDG_CONFIG_HOME"),
            }
        }
    }
}

#[test]
fn ffi_activation_explicit_config_replaces_discovered_user_config() {
    if std::env::var_os(DISCOVERY_CHILD_ENV).is_some() {
        run_explicit_config_replacement_test();
        return;
    }

    let output = Command::new(std::env::current_exe().expect("current test executable"))
        .arg("--exact")
        .arg(
            "plugin_activation_tests::ffi_activation_explicit_config_replaces_discovered_user_config",
        )
        .arg("--nocapture")
        .env(DISCOVERY_CHILD_ENV, "1")
        .env_remove("NEMO_RELAY_TEST_SKIP_IMPLICIT_CONFIG")
        .output()
        .expect("discovery child test should start");
    assert!(
        output.status.success(),
        "discovery child test failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn run_explicit_config_replacement_test() {
    DISCOVERED_STATIC_REGISTRATIONS.store(0, Ordering::SeqCst);
    DISCOVERED_STATIC_CALLBACKS.store(0, Ordering::SeqCst);
    *DISCOVERED_STATIC_CONFIG.lock().unwrap() = None;

    let environment = TempDir::new().expect("plugin discovery environment");
    let xdg_config_home = environment.path().join("xdg");
    let user_config_dir = xdg_config_home.join("nemo-relay");
    let project_config_dir = environment.path().join(".nemo-relay");
    std::fs::create_dir_all(&user_config_dir).expect("isolated user config directory");
    std::fs::create_dir_all(&project_config_dir).expect("legacy project config directory");
    let plugins_toml = user_config_dir.join("plugins.toml");
    std::fs::write(project_config_dir.join("plugins.toml"), "invalid = [")
        .expect("write ignored project plugin config");
    let _environment = PluginDiscoveryTestEnv::enter(environment.path(), &xdg_config_home);

    std::fs::write(
        &plugins_toml,
        format!(
            r#"version = 1

[[components]]
kind = {DISCOVERED_STATIC_PLUGIN_KIND:?}
enabled = true

[components.config]
source = "user-file"
"#
        ),
    )
    .expect("write user plugin config");

    let plugin_kind = cstring(DISCOVERED_STATIC_PLUGIN_KIND);
    assert_eq!(
        unsafe {
            api::nemo_relay_register_plugin(
                plugin_kind.as_ptr(),
                None,
                discovered_static_register,
                ptr::null_mut(),
                None,
            )
        },
        NemoRelayStatus::Ok
    );

    let manifest_dir = TempDir::new().expect("native manifest tempdir");
    let manifest = write_native_manifest(manifest_dir.path(), build_native_fixture());
    let (mut activation, report) =
        initialize_test_plugin_host_from_manifests(&[manifest.as_path()]);

    let mut active = false;
    assert_eq!(
        unsafe { api::nemo_relay_plugin_host_activation_is_active(activation, &mut active) },
        NemoRelayStatus::Ok
    );
    assert!(active);

    assert_eq!(report["config"]["diagnostics"], json!([]));
    assert_eq!(DISCOVERED_STATIC_REGISTRATIONS.load(Ordering::SeqCst), 0);
    assert_eq!(DISCOVERED_STATIC_CONFIG.lock().unwrap().as_ref(), None);
    assert!(plugin_kinds().iter().any(|kind| kind == "fixture_native"));

    let intercepted = tool_request_intercepts("ffi-layered-tool", json!({"input": true}));
    assert!(intercepted.get("file_static").is_none());
    assert_eq!(intercepted["native_plugin"], true);
    assert_eq!(DISCOVERED_STATIC_CALLBACKS.load(Ordering::SeqCst), 0);

    unsafe {
        assert_eq!(
            api::nemo_relay_plugin_host_activation_close(activation),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            api::nemo_relay_plugin_host_activation_is_active(activation, &mut active),
            NemoRelayStatus::Ok
        );
        assert!(!active);
        nemo_relay_plugin_host_activation_free(&mut activation);
    }
    assert!(!plugin_kinds().iter().any(|kind| kind == "fixture_native"));
    assert_eq!(
        tool_request_intercepts("ffi-layered-tool", json!({"input": true})),
        json!({"input": true})
    );
    assert_eq!(DISCOVERED_STATIC_CALLBACKS.load(Ordering::SeqCst), 0);
    assert_eq!(
        unsafe { api::nemo_relay_deregister_plugin(plugin_kind.as_ptr()) },
        NemoRelayStatus::Ok
    );
}

unsafe extern "C" fn discovered_static_register(
    _user_data: *mut libc::c_void,
    plugin_config_json: *const c_char,
    ctx: *mut FfiPluginContext,
) -> NemoRelayStatus {
    let config = unsafe { CStr::from_ptr(plugin_config_json) }
        .to_str()
        .ok()
        .and_then(|value| serde_json::from_str(value).ok());
    *DISCOVERED_STATIC_CONFIG.lock().unwrap() = config;
    DISCOVERED_STATIC_REGISTRATIONS.fetch_add(1, Ordering::SeqCst);
    let name = cstring("project_file_intercept");
    unsafe {
        api::nemo_relay_plugin_context_register_tool_request_intercept(
            ctx,
            name.as_ptr(),
            -1,
            false,
            discovered_static_tool_request,
            ptr::null_mut(),
            None,
        )
    }
}

unsafe extern "C" fn discovered_static_tool_request(
    _user_data: *mut libc::c_void,
    _name: *const c_char,
    args_json: *const c_char,
) -> *mut c_char {
    DISCOVERED_STATIC_CALLBACKS.fetch_add(1, Ordering::SeqCst);
    let mut args: Json = serde_json::from_str(
        unsafe { CStr::from_ptr(args_json) }
            .to_str()
            .unwrap_or("null"),
    )
    .unwrap_or_else(|_| json!({}));
    args["static_saw_dynamic"] = json!(args.get("native_plugin").is_some());
    args["file_static"] = json!(true);
    CString::new(args.to_string()).unwrap().into_raw()
}

#[test]
fn ffi_activation_loads_native_callbacks_and_removes_them_before_free() {
    let _guard = TEST_MUTEX.lock().unwrap();

    let manifest_dir = TempDir::new().expect("native manifest tempdir");
    let manifest = write_native_manifest(manifest_dir.path(), build_native_fixture());
    let (mut activation, report) =
        initialize_test_plugin_host_from_manifests(&[manifest.as_path()]);
    assert_eq!(report["config"]["diagnostics"], json!([]));
    assert_eq!(report["dynamic_plugins"][0]["plugin_id"], "fixture_native");
    assert!(plugin_kinds().iter().any(|kind| kind == "fixture_native"));

    assert_eq!(
        tool_request_intercepts("ffi-native-tool", json!({"input": true}))["native_plugin"],
        true
    );

    unsafe {
        assert_eq!(
            api::nemo_relay_plugin_host_activation_close(activation),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            api::nemo_relay_plugin_host_activation_close(activation),
            NemoRelayStatus::Ok
        );
        let mut closed_report = ptr::null_mut();
        assert_eq!(
            api::nemo_relay_plugin_host_activation_report_json(activation, &mut closed_report),
            NemoRelayStatus::Ok
        );
        assert_eq!(returned_json(closed_report), report);
        nemo_relay_plugin_host_activation_free(&mut activation);
    }
    assert!(!plugin_kinds().iter().any(|kind| kind == "fixture_native"));
    assert_eq!(
        tool_request_intercepts("ffi-native-tool", json!({"input": true})),
        json!({"input": true})
    );

    let (mut drop_activation, _) =
        initialize_test_plugin_host_from_manifests(&[manifest.as_path()]);
    assert_eq!(
        tool_request_intercepts("ffi-native-tool", json!({"input": true}))["native_plugin"],
        true
    );
    unsafe { nemo_relay_plugin_host_activation_free(&mut drop_activation) };
    assert_eq!(
        tool_request_intercepts("ffi-native-tool", json!({"input": true})),
        json!({"input": true})
    );
}

#[test]
fn ffi_activation_loads_worker_callbacks_and_stops_worker_on_clear() {
    let _guard = TEST_MUTEX.lock().unwrap();

    let manifest_dir = TempDir::new().expect("worker manifest tempdir");
    let manifest = write_worker_manifest(manifest_dir.path(), build_worker_fixture());
    let (mut activation, report) =
        initialize_test_plugin_host_from_manifests(&[manifest.as_path()]);
    assert_eq!(report["config"]["diagnostics"], json!([]));
    assert_eq!(report["dynamic_plugins"][0]["plugin_id"], "fixture_worker");
    assert!(plugin_kinds().iter().any(|kind| kind == "fixture_worker"));
    assert_eq!(
        tool_request_intercepts("ffi-worker-tool", json!({"input": true}))["worker_plugin"],
        true
    );

    unsafe {
        assert_eq!(
            api::nemo_relay_plugin_host_activation_close(activation),
            NemoRelayStatus::Ok
        );
        nemo_relay_plugin_host_activation_free(&mut activation);
    }
    assert!(!plugin_kinds().iter().any(|kind| kind == "fixture_worker"));
    assert_eq!(
        tool_request_intercepts("ffi-worker-tool", json!({"input": true})),
        json!({"input": true})
    );
}

#[test]
fn ffi_activation_rolls_back_an_earlier_native_load_when_a_later_load_fails() {
    let _guard = TEST_MUTEX.lock().unwrap();

    let manifest_dir = TempDir::new().expect("native manifest tempdir");
    let manifest = write_native_manifest(manifest_dir.path(), build_native_fixture());
    let missing_manifest = manifest_dir.path().join("missing-relay-plugin.toml");
    let (_config_dir, plugins_toml) =
        write_test_plugin_host_config(&[manifest.as_path(), missing_manifest.as_path()]);
    let config = cstring(r#"{"version":1,"components":[]}"#);
    let plugins_toml = cstring(&plugins_toml.to_string_lossy());
    let mut activation = ptr::null_mut();
    let mut report = ptr::null_mut();
    let status = unsafe {
        api::nemo_relay_plugin_initialize(
            config.as_ptr(),
            plugins_toml.as_ptr(),
            &mut activation,
            &mut report,
        )
    };
    assert_eq!(status, NemoRelayStatus::NotFound);
    assert!(activation.is_null());
    assert!(report.is_null());
    assert!(!plugin_kinds().iter().any(|kind| kind == "fixture_native"));
    assert_eq!(
        tool_request_intercepts("ffi-native-tool", json!({"input": true})),
        json!({"input": true})
    );

    let (mut activation, _) = initialize_test_plugin_host_from_manifests(&[manifest.as_path()]);
    unsafe {
        assert_eq!(
            api::nemo_relay_plugin_host_activation_close(activation),
            NemoRelayStatus::Ok
        );
        nemo_relay_plugin_host_activation_free(&mut activation);
    }
}

fn initialize_test_plugin_host_from_manifests(
    manifests: &[&Path],
) -> (*mut FfiPluginHostActivation, Json) {
    let (_config_dir, plugins_toml) = write_test_plugin_host_config(manifests);
    let config = cstring(r#"{"version":1,"components":[]}"#);
    let plugins_toml = cstring(&plugins_toml.to_string_lossy());
    let mut activation = ptr::null_mut();
    let mut report = ptr::null_mut();
    let status = unsafe {
        api::nemo_relay_plugin_initialize(
            config.as_ptr(),
            plugins_toml.as_ptr(),
            &mut activation,
            &mut report,
        )
    };
    assert_eq!(
        status,
        NemoRelayStatus::Ok,
        "activation failed: {:?}",
        unsafe { read_last_error() }
    );
    assert!(!activation.is_null());
    (activation, unsafe { returned_json(report) })
}

fn write_test_plugin_host_config(manifests: &[&Path]) -> (TempDir, PathBuf) {
    let directory = TempDir::new().expect("plugin host config tempdir");
    let path = directory.path().join("plugins.toml");
    let declarations = manifests
        .iter()
        .map(|manifest| {
            format!(
                "[[plugins.dynamic]]\nmanifest = {}\n",
                serde_json::to_string(&manifest.to_string_lossy())
                    .expect("manifest path JSON string")
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(
        &path,
        format!(
            "version = 1\n\n[plugins.policy.defaults]\nattestation = \"integrity_only\"\n\n{declarations}"
        ),
    )
    .expect("write plugin host config");
    (directory, path)
}

fn cstring(value: &str) -> CString {
    CString::new(value).expect("C string")
}

unsafe fn read_last_error() -> Option<String> {
    let pointer = nemo_relay_last_error();
    (!pointer.is_null()).then(|| {
        unsafe { CStr::from_ptr(pointer) }
            .to_string_lossy()
            .into_owned()
    })
}

unsafe fn returned_json(pointer: *mut c_char) -> Json {
    assert!(!pointer.is_null(), "expected returned JSON string");
    let json = unsafe { CStr::from_ptr(pointer) }
        .to_string_lossy()
        .into_owned();
    unsafe { nemo_relay_string_free(pointer) };
    serde_json::from_str(&json).expect("returned JSON")
}

fn tool_request_intercepts(name: &str, args: Json) -> Json {
    let name = cstring(name);
    let args = cstring(&args.to_string());
    let mut output = ptr::null_mut();
    let status = unsafe {
        api::nemo_relay_tool_request_intercepts(name.as_ptr(), args.as_ptr(), &mut output)
    };
    assert_eq!(
        status,
        NemoRelayStatus::Ok,
        "tool request intercept failed: {:?}",
        unsafe { read_last_error() }
    );
    unsafe { returned_json(output) }
}

fn plugin_kinds() -> Vec<String> {
    let mut output = ptr::null_mut();
    assert_eq!(
        unsafe { api::nemo_relay_list_plugin_kinds_json(&mut output) },
        NemoRelayStatus::Ok
    );
    serde_json::from_value(unsafe { returned_json(output) }).expect("plugin kinds JSON")
}

fn build_native_fixture() -> &'static Path {
    prepared_fixture("NEMO_RELAY_TEST_NATIVE_PLUGIN")
}

fn build_worker_fixture() -> &'static Path {
    prepared_fixture("NEMO_RELAY_TEST_WORKER_PLUGIN")
}

fn prepared_fixture(environment: &str) -> &'static Path {
    let path = std::env::var_os(environment)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let filename = if environment == "NEMO_RELAY_TEST_NATIVE_PLUGIN" {
                if cfg!(target_os = "windows") {
                    "nemo_relay_plugin_fixture.dll".into()
                } else if cfg!(target_os = "macos") {
                    "libnemo_relay_plugin_fixture.dylib".into()
                } else {
                    "libnemo_relay_plugin_fixture.so".into()
                }
            } else {
                format!(
                    "nemo-relay-worker-plugin-fixture{}",
                    std::env::consts::EXE_SUFFIX
                )
            };
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../target/test-plugin-fixtures/debug")
                .join(filename)
        });
    assert!(
        path.exists(),
        "plugin test fixture is missing; run `just build-test-plugin-fixtures`: {}",
        path.display()
    );
    Box::leak(path.into_boxed_path())
}

fn write_native_manifest(directory: &Path, library: &Path) -> PathBuf {
    let manifest = directory.join("relay-plugin.toml");
    std::fs::write(
        &manifest,
        format!(
            r#"
manifest_version = 1

[plugin]
id = "fixture_native"
kind = "rust_dynamic"

[compat]
relay = "={version}"
native_api = "1"

[defaults]
enabled = false

[capabilities]
items = ["plugin_native"]

[source]
artifact = {library:?}

[integrity]
sha256 = "sha256:{digest}"

[load]
library = {library:?}
symbol = "nemo_relay_fixture_native_plugin"
"#,
            version = env!("CARGO_PKG_VERSION"),
            library = library.to_string_lossy(),
            digest = sha256_file(library),
        ),
    )
    .expect("write native fixture manifest");
    manifest
}

fn write_worker_manifest(directory: &Path, binary: &Path) -> PathBuf {
    let manifest = directory.join("relay-plugin.toml");
    std::fs::write(
        &manifest,
        format!(
            r#"
manifest_version = 1

[plugin]
id = "fixture_worker"
kind = "worker"

[compat]
relay = "={version}"
worker_protocol = "grpc-v1"

[defaults]
enabled = false

[capabilities]
items = ["plugin_worker"]

[source]
artifact = {entrypoint:?}

[integrity]
sha256 = "sha256:{digest}"

[load]
runtime = "rust"
entrypoint = {entrypoint:?}
"#,
            version = env!("CARGO_PKG_VERSION"),
            entrypoint = binary.to_string_lossy(),
            digest = sha256_file(binary),
        ),
    )
    .expect("write worker fixture manifest");
    manifest
}

fn sha256_file(path: &Path) -> String {
    use sha2::{Digest, Sha256};

    Sha256::digest(std::fs::read(path).expect("read plugin fixture"))
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
