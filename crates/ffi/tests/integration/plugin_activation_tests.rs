// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use nemo_relay_ffi::types::{FfiPluginActivation, nemo_relay_plugin_activation_free};
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
fn ffi_activation_layers_discovered_static_and_explicit_dynamic_plugins() {
    if std::env::var_os(DISCOVERY_CHILD_ENV).is_some() {
        run_discovered_config_activation_test();
        return;
    }

    let output = Command::new(std::env::current_exe().expect("current test executable"))
        .arg("--exact")
        .arg(
            "plugin_activation_tests::ffi_activation_layers_discovered_static_and_explicit_dynamic_plugins",
        )
        .arg("--nocapture")
        .env(DISCOVERY_CHILD_ENV, "1")
        .output()
        .expect("discovery child test should start");
    assert!(
        output.status.success(),
        "discovery child test failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn run_discovered_config_activation_test() {
    let _ = nemo_relay_clear_plugin_configuration();
    DISCOVERED_STATIC_REGISTRATIONS.store(0, Ordering::SeqCst);
    DISCOVERED_STATIC_CALLBACKS.store(0, Ordering::SeqCst);
    *DISCOVERED_STATIC_CONFIG.lock().unwrap() = None;

    let environment = TempDir::new().expect("plugin discovery environment");
    let project_config_dir = environment.path().join(".nemo-relay");
    let xdg_config_home = environment.path().join("xdg");
    std::fs::create_dir_all(&project_config_dir).expect("project config directory");
    std::fs::create_dir_all(&xdg_config_home).expect("isolated user config directory");
    let plugins_toml = project_config_dir.join("plugins.toml");
    std::fs::write(&plugins_toml, "invalid = [").expect("write invalid plugin config");
    let _environment = PluginDiscoveryTestEnv::enter(environment.path(), &xdg_config_home);

    // Empty specifications fail before discovery or ownership. The malformed
    // file would otherwise produce a TOML error, and the successful activation
    // below proves this attempt did not retain the process-wide host claim.
    let config = cstring(r#"{"version":1,"components":[]}"#);
    let empty_specs = cstring("[]");
    let mut empty_activation = ptr::null_mut();
    let mut empty_report = ptr::null_mut();
    assert_eq!(
        unsafe {
            api::nemo_relay_activate_dynamic_plugins(
                config.as_ptr(),
                empty_specs.as_ptr(),
                &mut empty_activation,
                &mut empty_report,
            )
        },
        NemoRelayStatus::InvalidArg
    );
    assert!(empty_activation.is_null());
    assert!(empty_report.is_null());
    assert!(
        unsafe { read_last_error() }
            .unwrap_or_default()
            .contains("at least one dynamic plugin")
    );

    std::fs::write(
        &plugins_toml,
        format!(
            r#"version = 999

[[components]]
kind = {DISCOVERED_STATIC_PLUGIN_KIND:?}
enabled = true

[components.config]
source = "project-file"
"#
        ),
    )
    .expect("write project plugin config");

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
    let (mut activation, report) = activate_plugins(json!([{
        "plugin_id": "fixture_native",
        "kind": "rust_dynamic",
        "manifest_ref": manifest,
        "config": {}
    }]));

    // The explicit version 1 must override the discovered version 999. The
    // file-only component and its config must still survive the merge.
    assert_eq!(report["diagnostics"], json!([]));
    assert_eq!(DISCOVERED_STATIC_REGISTRATIONS.load(Ordering::SeqCst), 1);
    assert_eq!(
        DISCOVERED_STATIC_CONFIG.lock().unwrap().as_ref(),
        Some(&json!({"source": "project-file"}))
    );
    assert!(plugin_kinds().iter().any(|kind| kind == "fixture_native"));

    // Mutating the file after startup has no effect: discovery is one-shot.
    std::fs::write(&plugins_toml, "invalid = [").expect("mutate plugin config after startup");
    let intercepted = tool_request_intercepts("ffi-layered-tool", json!({"input": true}));
    assert_eq!(intercepted["file_static"], true);
    assert_eq!(intercepted["static_saw_dynamic"], false);
    assert_eq!(intercepted["native_plugin"], true);
    assert_eq!(DISCOVERED_STATIC_CALLBACKS.load(Ordering::SeqCst), 1);

    unsafe {
        assert_eq!(
            api::nemo_relay_plugin_activation_clear(activation),
            NemoRelayStatus::Ok
        );
        nemo_relay_plugin_activation_free(&mut activation);
    }
    assert!(!plugin_kinds().iter().any(|kind| kind == "fixture_native"));
    assert_eq!(
        tool_request_intercepts("ffi-layered-tool", json!({"input": true})),
        json!({"input": true})
    );
    assert_eq!(DISCOVERED_STATIC_CALLBACKS.load(Ordering::SeqCst), 1);
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
    let _ = nemo_relay_clear_plugin_configuration();

    let manifest_dir = TempDir::new().expect("native manifest tempdir");
    let manifest = write_native_manifest(manifest_dir.path(), build_native_fixture());
    let (mut activation, report) = activate_plugins(json!([{
        "plugin_id": "fixture_native",
        "kind": "rust_dynamic",
        "manifest_ref": manifest,
        "config": {}
    }]));
    assert_eq!(report["diagnostics"], json!([]));
    assert!(plugin_kinds().iter().any(|kind| kind == "fixture_native"));

    assert_eq!(
        tool_request_intercepts("ffi-native-tool", json!({"input": true}))["native_plugin"],
        true
    );

    unsafe {
        assert_eq!(
            api::nemo_relay_plugin_activation_clear(activation),
            NemoRelayStatus::Ok
        );
        assert_eq!(
            api::nemo_relay_plugin_activation_clear(activation),
            NemoRelayStatus::Ok
        );
        nemo_relay_plugin_activation_free(&mut activation);
    }
    assert!(!plugin_kinds().iter().any(|kind| kind == "fixture_native"));
    assert_eq!(
        tool_request_intercepts("ffi-native-tool", json!({"input": true})),
        json!({"input": true})
    );

    let (mut drop_activation, _) = activate_plugins(json!([{
        "plugin_id": "fixture_native",
        "kind": "rust_dynamic",
        "manifest_ref": manifest,
        "config": {}
    }]));
    assert_eq!(
        tool_request_intercepts("ffi-native-tool", json!({"input": true}))["native_plugin"],
        true
    );
    unsafe { nemo_relay_plugin_activation_free(&mut drop_activation) };
    assert_eq!(
        tool_request_intercepts("ffi-native-tool", json!({"input": true})),
        json!({"input": true})
    );
}

#[test]
fn ffi_activation_rejects_overlapping_outputs_without_claiming_host() {
    let _guard = TEST_MUTEX.lock().unwrap();
    let _ = nemo_relay_clear_plugin_configuration();

    let manifest_dir = TempDir::new().expect("native manifest tempdir");
    let manifest = write_native_manifest(manifest_dir.path(), build_native_fixture());
    let config = cstring(r#"{"version":1,"components":[]}"#);
    let specs_value = json!([{
        "plugin_id": "fixture_native",
        "kind": "rust_dynamic",
        "manifest_ref": manifest,
        "config": {}
    }]);
    let specs = cstring(&specs_value.to_string());
    let mut aliased_output = std::ptr::dangling_mut::<std::ffi::c_void>();
    let output_slot = &mut aliased_output as *mut *mut std::ffi::c_void;
    let status = unsafe {
        api::nemo_relay_activate_dynamic_plugins(
            config.as_ptr(),
            specs.as_ptr(),
            output_slot.cast::<*mut FfiPluginActivation>(),
            output_slot.cast::<*mut c_char>(),
        )
    };
    assert_eq!(status, NemoRelayStatus::InvalidArg);
    assert!(aliased_output.is_null());
    assert!(
        unsafe { read_last_error() }
            .unwrap_or_default()
            .contains("must not overlap")
    );

    let (mut activation, _) = activate_plugins(specs_value);
    unsafe {
        assert_eq!(
            api::nemo_relay_plugin_activation_clear(activation),
            NemoRelayStatus::Ok
        );
        nemo_relay_plugin_activation_free(&mut activation);
    }
}

#[test]
fn ffi_activation_loads_worker_callbacks_and_stops_worker_on_clear() {
    let _guard = TEST_MUTEX.lock().unwrap();
    let _ = nemo_relay_clear_plugin_configuration();

    let manifest_dir = TempDir::new().expect("worker manifest tempdir");
    let manifest = write_worker_manifest(manifest_dir.path(), build_worker_fixture());
    let (mut activation, report) = activate_plugins(json!([{
        "plugin_id": "fixture_worker",
        "kind": "worker",
        "manifest_ref": manifest,
        "config": {}
    }]));
    assert_eq!(report["diagnostics"], json!([]));
    assert!(plugin_kinds().iter().any(|kind| kind == "fixture_worker"));
    assert_eq!(
        tool_request_intercepts("ffi-worker-tool", json!({"input": true}))["worker_plugin"],
        true
    );

    unsafe {
        assert_eq!(
            api::nemo_relay_plugin_activation_clear(activation),
            NemoRelayStatus::Ok
        );
        nemo_relay_plugin_activation_free(&mut activation);
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
    let _ = nemo_relay_clear_plugin_configuration();

    let manifest_dir = TempDir::new().expect("native manifest tempdir");
    let manifest = write_native_manifest(manifest_dir.path(), build_native_fixture());
    let missing_manifest = manifest_dir.path().join("missing-relay-plugin.toml");
    let config = cstring(r#"{"version":1,"components":[]}"#);
    let specs = cstring(
        &json!([
            {
                "plugin_id": "fixture_native",
                "kind": "rust_dynamic",
                "manifest_ref": manifest,
                "config": {}
            },
            {
                "plugin_id": "fixture_missing",
                "kind": "rust_dynamic",
                "manifest_ref": missing_manifest,
                "config": {}
            }
        ])
        .to_string(),
    );
    let mut activation = ptr::null_mut();
    let mut report = ptr::null_mut();
    let status = unsafe {
        api::nemo_relay_activate_dynamic_plugins(
            config.as_ptr(),
            specs.as_ptr(),
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

    let (mut activation, _) = activate_plugins(json!([{
        "plugin_id": "fixture_native",
        "kind": "rust_dynamic",
        "manifest_ref": manifest,
        "config": {}
    }]));
    unsafe {
        assert_eq!(
            api::nemo_relay_plugin_activation_clear(activation),
            NemoRelayStatus::Ok
        );
        nemo_relay_plugin_activation_free(&mut activation);
    }
}

fn activate_plugins(specs: Json) -> (*mut FfiPluginActivation, Json) {
    let config = cstring(r#"{"version":1,"components":[]}"#);
    let specs = cstring(&specs.to_string());
    let mut activation = ptr::null_mut();
    let mut report = ptr::null_mut();
    let status = unsafe {
        api::nemo_relay_activate_dynamic_plugins(
            config.as_ptr(),
            specs.as_ptr(),
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
    static FIXTURE: OnceLock<PathBuf> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let source_dir = TempDir::new().expect("native fixture source tempdir");
        let fixture_dir = source_dir.path().join("native_plugin");
        let source = fixture_dir.join("src");
        std::fs::create_dir_all(&source).expect("native fixture src dir");
        let plugin_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../plugin");
        let manifest_template = std::fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../core/tests/fixtures/native_plugin/Cargo.toml"),
        )
        .expect("native fixture Cargo.toml");
        let manifest = manifest_template.replace(
            r#"nemo-relay-plugin = { path = "../../../../plugin" }"#,
            &format!("nemo-relay-plugin = {{ path = {plugin_path:?} }}"),
        );
        std::fs::write(fixture_dir.join("Cargo.toml"), manifest)
            .expect("write native fixture Cargo.toml");
        std::fs::copy(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../core/tests/fixtures/native_plugin/src/lib.rs"),
            source.join("lib.rs"),
        )
        .expect("copy native fixture source");

        let target =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/ffi-native-plugin-fixture");
        let status = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()))
            .arg("build")
            .arg("--quiet")
            .arg("--manifest-path")
            .arg(fixture_dir.join("Cargo.toml"))
            .arg("--target-dir")
            .arg(&target)
            .status()
            .expect("native fixture build should start");
        assert!(status.success(), "native fixture build failed: {status}");
        let library = target.join("debug").join(native_library_name());
        assert!(
            library.exists(),
            "missing native fixture: {}",
            library.display()
        );
        library
    })
}

fn build_worker_fixture() -> &'static Path {
    static FIXTURE: OnceLock<PathBuf> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../core/tests/fixtures/worker_plugin/Cargo.toml");
        let target =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/ffi-worker-plugin-fixture");
        let status = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()))
            .arg("build")
            .arg("--quiet")
            .arg("--locked")
            .arg("--manifest-path")
            .arg(manifest)
            .arg("--target-dir")
            .arg(&target)
            .status()
            .expect("worker fixture build should start");
        assert!(status.success(), "worker fixture build failed: {status}");
        let binary = target.join("debug").join(format!(
            "nemo-relay-worker-plugin-fixture{}",
            std::env::consts::EXE_SUFFIX
        ));
        assert!(
            binary.exists(),
            "missing worker fixture: {}",
            binary.display()
        );
        binary
    })
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

[load]
library = {library:?}
symbol = "nemo_relay_fixture_native_plugin"
"#,
            version = env!("CARGO_PKG_VERSION"),
            library = library.to_string_lossy(),
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

[load]
runtime = "rust"
entrypoint = {entrypoint:?}
"#,
            version = env!("CARGO_PKG_VERSION"),
            entrypoint = binary.to_string_lossy(),
        ),
    )
    .expect("write worker fixture manifest");
    manifest
}

fn native_library_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "nemo_relay_plugin_fixture.dll"
    } else if cfg!(target_os = "macos") {
        "libnemo_relay_plugin_fixture.dylib"
    } else {
        "libnemo_relay_plugin_fixture.so"
    }
}
