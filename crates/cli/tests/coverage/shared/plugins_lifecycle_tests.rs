// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};
use std::{
    ffi::{OsStr, OsString},
    sync::{Mutex, MutexGuard},
};

use super::*;
use crate::configuration::{
    ConfigurationScope, GatewayOverrides, PluginsAddRequest, PluginsDisableRequest,
    PluginsEnableRequest, PluginsInspectRequest, PluginsListRequest, PluginsRemoveRequest,
    PluginsValidateRequest,
};
use crate::error::PluginLifecycleFailureKind;
use base64::Engine;
use nemo_relay::plugin::dynamic::{
    DynamicPluginFailurePhase, WorkerPluginLoadSpec, load_worker_plugins,
};
use ring::rand::SystemRandom;
use ring::signature::{Ed25519KeyPair, KeyPair};
use sha2::{Digest, Sha256};

#[cfg(unix)]
#[test]
fn python_venv_launcher_detection_only_preserves_bin_python_links() {
    assert!(is_python_venv_launcher(Path::new("env/bin/python")));
    assert!(is_python_venv_launcher(Path::new("env/bin/python3.11")));
    assert!(!is_python_venv_launcher(Path::new("env/bin/pip")));
    assert!(!is_python_venv_launcher(Path::new("env/lib/python3.11")));
}

#[cfg(unix)]
#[test]
fn snapshot_protection_does_not_follow_python_launcher_symlink() {
    use std::os::unix::fs::{PermissionsExt, symlink};

    let temp = tempfile::tempdir().unwrap();
    let target = temp.path().join("external-python");
    std::fs::write(&target, b"python").unwrap();
    std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o755)).unwrap();
    let root = temp.path().join("snapshot");
    let bin = root.join("bin");
    std::fs::create_dir_all(&bin).unwrap();
    symlink(&target, bin.join("python")).unwrap();

    protect_snapshot_tree(&root).unwrap();

    assert!(
        std::fs::symlink_metadata(bin.join("python"))
            .unwrap()
            .file_type()
            .is_symlink()
    );
    assert_eq!(
        std::fs::metadata(&target).unwrap().permissions().mode() & 0o777,
        0o755
    );
    make_snapshot_removable(&root);
}

#[cfg(unix)]
#[test]
fn snapshot_digest_hashes_python_launcher_symlink_without_following_it() {
    use std::os::unix::fs::symlink;

    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("snapshot");
    let bin = root
        .join(MANAGED_ENVIRONMENTS_DIR)
        .join("environment")
        .join("bin");
    std::fs::create_dir_all(&bin).unwrap();
    let launcher = bin.join("python");
    symlink("/missing/python-a", &launcher).unwrap();

    let first_verification = snapshot_tree_digest(&root, false).unwrap();
    let first_identity = snapshot_tree_digest(&root, true).unwrap();

    std::fs::remove_file(&launcher).unwrap();
    std::fs::write(&launcher, b"/missing/python-a").unwrap();
    assert_ne!(
        first_verification,
        snapshot_tree_digest(&root, false).unwrap(),
        "a regular file must not collide with an equivalent symlink target"
    );

    std::fs::remove_file(&launcher).unwrap();
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

struct CurrentDirGuard {
    original: PathBuf,
}

impl CurrentDirGuard {
    fn enter(path: &Path) -> Self {
        let original = std::env::current_dir().unwrap();
        std::env::set_current_dir(path).unwrap();
        Self { original }
    }
}

impl Drop for CurrentDirGuard {
    fn drop(&mut self) {
        std::env::set_current_dir(&self.original).unwrap();
    }
}

struct EnvScope {
    _guard: MutexGuard<'static, ()>,
    _cwd_guard: crate::test_support::CwdTestScope,
    values: Vec<(&'static str, Option<OsString>)>,
}

impl EnvScope {
    fn hermetic(temp: &tempfile::TempDir) -> Self {
        let xdg = temp.path().join("xdg");
        std::fs::create_dir_all(&xdg).unwrap();
        Self::set(&[
            ("HOME", Some(temp.path().as_os_str())),
            ("XDG_CONFIG_HOME", Some(xdg.as_os_str())),
            ("NEMO_RELAY_PYTHON", None),
        ])
    }

    fn set(values: &[(&'static str, Option<&std::ffi::OsStr>)]) -> Self {
        // Acquire process-global locks in the same CWD-then-environment order as
        // the rest of the CLI test suite. Lifecycle tests also change CWD while
        // this scope is alive.
        let cwd_guard = crate::test_support::CwdTestScope::locked();
        let guard = crate::test_support::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let previous = values
            .iter()
            .map(|(key, _)| (*key, std::env::var_os(key)))
            .collect::<Vec<_>>();
        for (key, value) in values {
            unsafe {
                match value {
                    Some(value) => std::env::set_var(key, value),
                    None => std::env::remove_var(key),
                }
            }
        }
        Self {
            _guard: guard,
            _cwd_guard: cwd_guard,
            values: previous,
        }
    }
}

impl Drop for EnvScope {
    fn drop(&mut self) {
        for (key, value) in self.values.drain(..) {
            unsafe {
                match value {
                    Some(value) => std::env::set_var(key, value),
                    None => std::env::remove_var(key),
                }
            }
        }
    }
}

fn write_dynamic_manifest(dir: &Path, plugin_id: &str) -> PathBuf {
    write_dynamic_manifest_with_options(dir, plugin_id, &["plugin_worker"], None)
}

fn write_dynamic_manifest_with_options(
    dir: &Path,
    plugin_id: &str,
    capabilities: &[&str],
    signature_ref: Option<&str>,
) -> PathBuf {
    let artifact_body = format!("def register():\n    return {plugin_id:?}\n");
    std::fs::write(dir.join("plugin.py"), &artifact_body).unwrap();
    let digest = format!(
        "sha256:{}",
        Sha256::digest(artifact_body.as_bytes())
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    );
    let capabilities = capabilities
        .iter()
        .map(|capability| format!("\"{capability}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let signature_line = signature_ref
        .map(|signature_ref| format!("signature = \"{signature_ref}\"\n"))
        .unwrap_or_default();
    let manifest_path = dir.join("relay-plugin.toml");
    std::fs::write(
        &manifest_path,
        format!(
            r#"
manifest_version = 1

[plugin]
id = "{plugin_id}"
kind = "worker"

[compat]
relay = "0.5"
worker_protocol = "grpc-v1"

[defaults]
enabled = false

[capabilities]
items = [{capabilities}]

[source]
artifact = "plugin.py"

[integrity]
sha256 = "{digest}"
{signature_line}

[load]
runtime = "command"
entrypoint = "plugin.py"
"#,
            capabilities = capabilities,
            signature_line = signature_line,
        ),
    )
    .unwrap();
    manifest_path
}

fn write_dynamic_manifest_with_config_schema(
    dir: &Path,
    plugin_id: &str,
    schema: &serde_json::Value,
) -> PathBuf {
    let manifest_path = write_dynamic_manifest_with_options(
        dir,
        plugin_id,
        &["plugin_worker", "config_schema"],
        None,
    );
    let mut manifest = std::fs::read_to_string(&manifest_path).unwrap();
    manifest.push_str(
        r#"

[config_schema]
path = "config.schema.json"
"#,
    );
    std::fs::write(&manifest_path, manifest).unwrap();
    std::fs::write(
        dir.join("config.schema.json"),
        serde_json::to_vec_pretty(schema).unwrap(),
    )
    .unwrap();
    manifest_path
}

fn write_python_dynamic_manifest(dir: &Path, plugin_id: &str) -> PathBuf {
    let artifact_body = "def main():\n    return None\n";
    std::fs::write(dir.join("plugin.py"), artifact_body).unwrap();
    std::fs::write(
        dir.join("pyproject.toml"),
        r#"[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "nemo-relay-lifecycle-test-plugin"
version = "0.1.0"

[tool.setuptools]
py-modules = ["plugin"]
"#,
    )
    .unwrap();
    let digest = format!(
        "sha256:{}",
        Sha256::digest(artifact_body.as_bytes())
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    );
    let manifest_path = dir.join("relay-plugin.toml");
    std::fs::write(
        &manifest_path,
        format!(
            r#"
manifest_version = 1

[plugin]
id = "{plugin_id}"
kind = "worker"

[compat]
relay = "0.5"
worker_protocol = "grpc-v1"

[defaults]
enabled = false

[capabilities]
items = ["plugin_worker"]

[source]
manifest_root = "."
artifact = "plugin.py"

[integrity]
sha256 = "{digest}"

[load]
runtime = "python"
entrypoint = "plugin:main"
"#,
        ),
    )
    .unwrap();
    manifest_path
}

#[derive(Default)]
struct FakePythonEnvironmentRunner {
    fail_install: bool,
    calls: Mutex<Vec<(OsString, Vec<OsString>)>>,
}

impl FakePythonEnvironmentRunner {
    fn failing_install() -> Self {
        Self {
            fail_install: true,
            ..Self::default()
        }
    }

    fn calls(&self) -> Vec<(OsString, Vec<OsString>)> {
        self.calls.lock().unwrap().clone()
    }
}

impl PythonEnvironmentCommandRunner for FakePythonEnvironmentRunner {
    fn run(&self, program: &OsStr, args: &[OsString]) -> Result<(), String> {
        self.calls
            .lock()
            .unwrap()
            .push((program.to_owned(), args.to_vec()));
        if args.get(1).is_some_and(|arg| arg == "venv") {
            let environment = PathBuf::from(args.last().expect("venv environment path"));
            let python = environment::environment_python_path(&environment);
            std::fs::create_dir_all(python.parent().unwrap()).unwrap();
            std::fs::write(python, b"fake python").unwrap();
            return Ok(());
        }
        if self.fail_install && args.get(1).is_some_and(|arg| arg == "pip") {
            return Err("fixture pip failure".into());
        }
        Ok(())
    }
}

fn write_detached_ed25519_signature(dir: &Path, signature_name: &str) -> String {
    std::fs::create_dir_all(dir).unwrap();
    let artifact = std::fs::read(dir.join("plugin.py")).unwrap();
    let pkcs8 =
        Ed25519KeyPair::generate_pkcs8(&SystemRandom::new()).expect("generate ed25519 keypair");
    let key_pair = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).expect("parse ed25519 keypair");
    let signature = key_pair.sign(&artifact);
    let signature_text = format!(
        "ed25519:{}\n",
        base64::engine::general_purpose::STANDARD.encode(signature.as_ref())
    );
    std::fs::write(dir.join(signature_name), signature_text).unwrap();
    format!(
        "ed25519:{}",
        base64::engine::general_purpose::STANDARD.encode(key_pair.public_key().as_ref())
    )
}

fn generate_ed25519_public_key() -> String {
    let pkcs8 =
        Ed25519KeyPair::generate_pkcs8(&SystemRandom::new()).expect("generate ed25519 keypair");
    let key_pair = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).expect("parse ed25519 keypair");
    format!(
        "ed25519:{}",
        base64::engine::general_purpose::STANDARD.encode(key_pair.public_key().as_ref())
    )
}

#[test]
fn trust_failure_messages_and_codes_cover_all_variants() {
    let path = PathBuf::from("/tmp/plugin.py");
    let signature_path = PathBuf::from("/tmp/plugin.py.sig");
    let cases = vec![
        (
            trust::DynamicPluginTrustFailure::MissingArtifact,
            "integrity_failed",
            "missing source.artifact",
        ),
        (
            trust::DynamicPluginTrustFailure::MissingIntegrityDigest,
            "integrity_failed",
            "missing integrity.sha256",
        ),
        (
            trust::DynamicPluginTrustFailure::ArtifactRead {
                path: path.clone(),
                error: "boom".into(),
            },
            "integrity_failed",
            "could not be read for trust verification",
        ),
        (
            trust::DynamicPluginTrustFailure::IntegrityMismatch {
                path: path.clone(),
                expected: "sha256:expected".into(),
                actual: "sha256:actual".into(),
            },
            "integrity_failed",
            "failed integrity verification",
        ),
        (
            trust::DynamicPluginTrustFailure::MissingSignature,
            "attestation_failed",
            "requires integrity.signature",
        ),
        (
            trust::DynamicPluginTrustFailure::MissingTrustedKeys,
            "attestation_failed",
            "no trusted_public_keys",
        ),
        (
            trust::DynamicPluginTrustFailure::SignatureRead {
                path: signature_path.clone(),
                error: "nope".into(),
            },
            "attestation_failed",
            "signature /tmp/plugin.py.sig could not be read",
        ),
        (
            trust::DynamicPluginTrustFailure::InvalidTrustedKey {
                key: "ed25519:bad".into(),
                error: "invalid".into(),
            },
            "attestation_failed",
            "invalid trusted public key",
        ),
        (
            trust::DynamicPluginTrustFailure::SignatureVerification {
                path: signature_path,
                parse_errors: vec!["bad key".into()],
            },
            "attestation_failed",
            "key parse errors: bad key",
        ),
    ];

    for (failure, code, snippet) in cases {
        assert_eq!(failure.refusal_code(), code);
        let rendered = failure.display("acme.coverage").to_string();
        assert!(rendered.contains("acme.coverage"), "{rendered}");
        assert!(rendered.contains(snippet), "{rendered}");
    }
}

#[test]
fn trust_last_error_preserves_integrity_code_under_signature_policy() {
    let trust = EvaluatedDynamicPluginTrust {
        integrity: DynamicPluginCheckState::Invalid,
        authenticity: DynamicPluginCheckState::Unknown,
        failure: Some(trust::DynamicPluginTrustFailure::IntegrityMismatch {
            path: PathBuf::from("/tmp/plugin.py"),
            expected: "sha256:expected".into(),
            actual: "sha256:actual".into(),
        }),
    };

    let error = trust
        .last_error("acme.coverage")
        .expect("integrity mismatch should persist an error");
    assert_eq!(error.phase, DynamicPluginFailurePhase::Validation);
    assert_eq!(error.code, "integrity_failed");
    assert!(error.message.contains("acme.coverage"));
    assert!(error.message.contains("failed integrity verification"));
}

#[test]
fn trust_evaluation_short_circuits_when_policy_is_blocked() {
    let temp = tempfile::tempdir().unwrap();
    let plugin_dir = temp.path().join("plugins").join("acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    let manifest_path = write_dynamic_manifest(&plugin_dir, "acme.blocked-short-circuit");

    let (manifest, manifest_ref) = DynamicPluginManifest::load_from_path(&manifest_path)
        .map_err(|error| CliError::Config(error.to_string()))
        .unwrap();
    let blocked_policy = crate::plugins::policy::EvaluatedDynamicPluginHostPolicy {
        policy_satisfied: false,
        startup_class: nemo_relay::plugin::dynamic::DynamicPluginStartupClass::Required,
        attestation_mode:
            nemo_relay::plugin::dynamic::DynamicPluginAttestationMode::SignatureRequired,
        trusted_public_keys: Vec::new(),
        failure: Some(crate::plugins::policy::DynamicPluginHostPolicyFailure::Blocked),
    };

    let trust = evaluate_dynamic_plugin_trust(&manifest, &manifest_ref, &blocked_policy);

    assert_eq!(trust.integrity, DynamicPluginCheckState::Unknown);
    assert_eq!(trust.authenticity, DynamicPluginCheckState::Unknown);
    assert!(trust.failure().is_none());
}

fn write_native_dynamic_manifest(dir: &Path, plugin_id: &str) -> PathBuf {
    let artifact_body = b"native plugin fixture";
    std::fs::write(dir.join("libfixture_native.so"), artifact_body).unwrap();
    let digest = format!(
        "sha256:{}",
        Sha256::digest(artifact_body)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    );
    let manifest_path = dir.join("relay-plugin.toml");
    std::fs::write(
        &manifest_path,
        format!(
            r#"
manifest_version = 1

[plugin]
id = "{plugin_id}"
kind = "rust_dynamic"

[compat]
relay = "0.5"
native_api = "1"

[defaults]
enabled = false

[capabilities]
items = ["plugin_native"]

[source]
artifact = "libfixture_native.so"

[integrity]
sha256 = "{digest}"

[load]
library = "libfixture_native.so"
symbol = "nemo_relay_fixture_native_plugin"
"#,
            digest = digest,
        ),
    )
    .unwrap();
    manifest_path
}

fn materialize_native_example_manifest(dir: &Path) -> (PathBuf, PathBuf) {
    let artifact_name = format!(
        "{}nemo_relay_rust_native_plugin_example{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    );
    let artifact_relative = Path::new("target").join("debug").join(&artifact_name);
    let artifact_path = dir.join(&artifact_relative);
    std::fs::create_dir_all(artifact_path.parent().unwrap()).unwrap();
    let artifact_body = b"native plugin example fixture";
    std::fs::write(&artifact_path, artifact_body).unwrap();
    let digest = Sha256::digest(artifact_body)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();

    let repository_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let template = std::fs::read_to_string(
        repository_root.join("examples/rust-native-plugin/relay-plugin.toml"),
    )
    .unwrap();
    let config_schema =
        std::fs::read(repository_root.join("examples/rust-native-plugin/config.schema.json"))
            .unwrap();
    let manifest = template
        .replace("<platform-library-file>", &artifact_name)
        .replace("<artifact-sha256>", &digest);
    let manifest_path = dir.join("relay-plugin.toml");
    std::fs::write(&manifest_path, manifest).unwrap();
    std::fs::write(dir.join("config.schema.json"), config_schema).unwrap();
    (manifest_path, artifact_path)
}

#[test]
fn tracked_native_plugin_example_satisfies_default_trust_policy() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("native-example");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    materialize_native_example_manifest(&plugin_dir);

    add(
        PluginsAddRequest {
            scope: ConfigurationScope {
                project: true,
                ..ConfigurationScope::default()
            },
            path: plugin_dir,
        },
        &GatewayOverrides::default(),
    )
    .unwrap();

    let resolved = resolve_plugins_config(None).unwrap();
    assert_eq!(resolved.dynamic_plugins.len(), 1);
    assert_eq!(
        resolved.dynamic_plugins[0].plugin_id,
        "examples.rust_native_policy"
    );
}

#[test]
fn tracked_native_plugin_example_rejects_tampered_artifact() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("native-example");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    let (_, artifact_path) = materialize_native_example_manifest(&plugin_dir);
    std::fs::write(artifact_path, b"tampered native plugin example fixture").unwrap();

    let error = add(
        PluginsAddRequest {
            scope: ConfigurationScope {
                project: true,
                ..ConfigurationScope::default()
            },
            path: plugin_dir,
        },
        &GatewayOverrides::default(),
    )
    .unwrap_err();

    match error {
        CliError::PluginLifecycle {
            kind: PluginLifecycleFailureKind::Refused,
            code: Some("integrity_failed"),
            message,
            ..
        } => assert!(message.contains("failed integrity verification")),
        other => panic!("unexpected integrity add error: {other}"),
    }
    assert!(
        resolve_plugins_config(None)
            .unwrap()
            .dynamic_plugins
            .is_empty()
    );
}

#[cfg(unix)]
#[test]
fn activation_snapshot_never_rereads_replaced_or_oversized_worker_code() {
    use std::os::unix::fs::PermissionsExt;

    let temp = tempfile::tempdir().unwrap();
    let plugin_dir = temp.path().join("plugin");
    let worker_dir = temp.path().join("worker-runtime");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    std::fs::create_dir_all(&worker_dir).unwrap();
    let artifact_path = worker_dir.join("worker.sh");
    let safe_worker = "#!/bin/sh\nexit 1\n".to_string();
    std::fs::write(&artifact_path, &safe_worker).unwrap();
    std::fs::write(worker_dir.join("resource.txt"), b"expected\n").unwrap();
    std::fs::set_permissions(&artifact_path, std::fs::Permissions::from_mode(0o755)).unwrap();
    let digest = Sha256::digest(safe_worker.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let manifest_path = plugin_dir.join("relay-plugin.toml");
    std::fs::write(
        &manifest_path,
        format!(
            r#"manifest_version = 1

[plugin]
id = "acme.snapshot-race"
kind = "worker"

[compat]
relay = "0.5"
worker_protocol = "grpc-v1"

[defaults]
enabled = false

[capabilities]
items = ["plugin_worker"]

[source]
artifact = "../worker-runtime/worker.sh"

[integrity]
sha256 = "sha256:{digest}"

[load]
runtime = "command"
entrypoint = "../worker-runtime/worker.sh"
"#
        ),
    )
    .unwrap();
    let snapshot = DynamicPluginActivationSnapshot::create(
        manifest_path.to_string_lossy().as_ref(),
        "acme.snapshot-race",
        DynamicPluginKind::Worker,
        None,
        &crate::plugins::policy::DynamicPluginHostPolicy::default(),
    )
    .unwrap();
    assert_eq!(
        std::fs::read(snapshot.root.join("external-entrypoint/resource.txt")).unwrap(),
        b"expected\n"
    );
    let (activation_manifest, _) =
        DynamicPluginManifest::load_from_path(PathBuf::from(snapshot.activation_manifest_ref()))
            .unwrap();
    let activation_artifact = activation_manifest
        .source
        .as_ref()
        .and_then(|source| source.artifact.as_deref())
        .unwrap();
    let DynamicPluginManifestLoad::Worker(activation_load) = &activation_manifest.load else {
        panic!("command activation must retain a worker load contract");
    };
    assert_eq!(
        Some(activation_artifact),
        activation_load.entrypoint.as_deref(),
        "the integrity-checked artifact and executed entrypoint must be one snapshot file"
    );
    let source_closure_digest =
        dynamic_plugin_runtime_closure_digest(manifest_path.to_string_lossy().as_ref(), None)
            .unwrap();
    assert_eq!(snapshot.closure_digest(), source_closure_digest);

    let marker = temp.path().join("replaced-worker-executed");
    std::fs::write(
        &artifact_path,
        format!("#!/bin/sh\ntouch {}\nexit 1\n", marker.display()),
    )
    .unwrap();
    std::fs::OpenOptions::new()
        .write(true)
        .open(&artifact_path)
        .unwrap()
        .set_len(crate::configuration::MAX_BOOTSTRAP_IDENTITY_FILE_BYTES + 1)
        .unwrap();
    std::fs::set_permissions(&artifact_path, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::fs::write(&manifest_path, b"not valid TOML").unwrap();
    std::fs::OpenOptions::new()
        .write(true)
        .open(&manifest_path)
        .unwrap()
        .set_len(crate::configuration::MAX_BOOTSTRAP_IDENTITY_FILE_BYTES + 1)
        .unwrap();

    let error = match load_worker_plugins(vec![WorkerPluginLoadSpec {
        plugin_id: "acme.snapshot-race".into(),
        manifest_ref: snapshot.activation_manifest_ref(),
        environment_ref: None,
        config: Map::new(),
    }]) {
        Ok(_) => panic!("safe snapshot worker unexpectedly activated"),
        Err(error) => error.to_string(),
    };

    assert!(
        !marker.exists(),
        "the replaced original worker was executed"
    );
    assert!(!error.contains("invalid relay-plugin.toml"), "{error}");
    assert!(!error.contains("exceeds the"), "{error}");
}

#[cfg(unix)]
#[test]
fn activation_snapshot_detects_mutation_of_the_runtime_copy() {
    use std::os::unix::fs::PermissionsExt;

    let temp = tempfile::tempdir().unwrap();
    let manifest_path = write_native_dynamic_manifest(temp.path(), "acme.snapshot-mutation");
    let snapshot = DynamicPluginActivationSnapshot::create(
        manifest_path.to_string_lossy().as_ref(),
        "acme.snapshot-mutation",
        DynamicPluginKind::RustDynamic,
        None,
        &crate::plugins::policy::DynamicPluginHostPolicy::default(),
    )
    .unwrap();
    std::fs::set_permissions(&snapshot.root, std::fs::Permissions::from_mode(0o700)).unwrap();
    std::fs::set_permissions(
        snapshot.activation_manifest.parent().unwrap(),
        std::fs::Permissions::from_mode(0o700),
    )
    .unwrap();
    std::fs::set_permissions(
        &snapshot.activation_manifest,
        std::fs::Permissions::from_mode(0o600),
    )
    .unwrap();
    std::fs::write(&snapshot.activation_manifest, b"replaced").unwrap();

    let error = snapshot.verify_current().unwrap_err().to_string();
    assert!(error.contains("changed before code load"), "{error}");
}

#[test]
fn activation_snapshot_keeps_adjacent_native_dependencies_for_external_load_target() {
    let temp = tempfile::tempdir().unwrap();
    let manifest_dir = temp.path().join("plugin");
    let native_dir = temp.path().join("native-runtime");
    std::fs::create_dir_all(&manifest_dir).unwrap();
    std::fs::create_dir_all(&native_dir).unwrap();
    let library = native_dir.join("libfixture_native.so");
    let library_bytes = b"native plugin fixture";
    std::fs::write(&library, library_bytes).unwrap();
    std::fs::write(
        native_dir.join("libadjacent_dependency.so"),
        b"adjacent dependency",
    )
    .unwrap();
    let digest = Sha256::digest(library_bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let manifest_path = manifest_dir.join("relay-plugin.toml");
    std::fs::write(
        &manifest_path,
        format!(
            r#"manifest_version = 1

[plugin]
id = "acme.external-native-closure"
kind = "rust_dynamic"

[compat]
relay = "0.5"
native_api = "1"

[defaults]
enabled = false

[capabilities]
items = ["plugin_native"]

[source]
artifact = "../native-runtime/libfixture_native.so"

[integrity]
sha256 = "sha256:{digest}"

[load]
library = "../native-runtime/libfixture_native.so"
symbol = "nemo_relay_fixture_native_plugin"
"#,
        ),
    )
    .unwrap();

    let snapshot = DynamicPluginActivationSnapshot::create(
        manifest_path.to_string_lossy().as_ref(),
        "acme.external-native-closure",
        DynamicPluginKind::RustDynamic,
        None,
        &crate::plugins::policy::DynamicPluginHostPolicy::default(),
    )
    .unwrap();

    assert!(
        snapshot
            .root
            .join("external-library/libadjacent_dependency.so")
            .is_file()
    );
    assert_eq!(
        snapshot.closure_digest(),
        dynamic_plugin_runtime_closure_digest(manifest_path.to_string_lossy().as_ref(), None)
            .unwrap()
    );
}

#[test]
fn activation_snapshot_and_python_attestation_enforce_exact_directory_depth_boundary() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let plugin_dir = temp.path().join("deep-plugin");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    let manifest_path = write_native_dynamic_manifest(&plugin_dir, "acme.deep-closure");
    let mut deep_plugin_path = plugin_dir.clone();
    for _ in 1..MAX_SNAPSHOT_DEPTH {
        deep_plugin_path.push("d");
    }
    std::fs::create_dir_all(&deep_plugin_path).unwrap();

    dynamic_plugin_runtime_closure_digest(manifest_path.to_string_lossy().as_ref(), None).unwrap();
    DynamicPluginActivationSnapshot::create(
        manifest_path.to_string_lossy().as_ref(),
        "acme.deep-closure",
        DynamicPluginKind::RustDynamic,
        None,
        &crate::plugins::policy::DynamicPluginHostPolicy::default(),
    )
    .unwrap();

    deep_plugin_path.push("too-deep");
    std::fs::create_dir(&deep_plugin_path).unwrap();

    let error =
        dynamic_plugin_runtime_closure_digest(manifest_path.to_string_lossy().as_ref(), None)
            .unwrap_err()
            .to_string();
    assert!(error.contains("traversal depth"), "{error}");

    let error = DynamicPluginActivationSnapshot::create(
        manifest_path.to_string_lossy().as_ref(),
        "acme.deep-closure",
        DynamicPluginKind::RustDynamic,
        None,
        &crate::plugins::policy::DynamicPluginHostPolicy::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("traversal depth"), "{error}");

    let environment_path = temp.path().join("deep-environment");
    let mut deep_environment_path = environment_path.clone();
    for _ in 1..environment::MAX_ENVIRONMENT_DEPTH {
        deep_environment_path.push("d");
    }
    std::fs::create_dir_all(&deep_environment_path).unwrap();
    environment::write_environment_attestation(&environment_path, "sha256:fixture-source-artifact")
        .unwrap();

    deep_environment_path.push("too-deep");
    std::fs::create_dir(&deep_environment_path).unwrap();
    let error = environment::write_environment_attestation(
        &environment_path,
        "sha256:fixture-source-artifact",
    )
    .unwrap_err();
    assert!(error.contains("traversal depth"), "{error}");
}

#[test]
fn runtime_directory_collection_rejects_before_exceeding_bounded_sort_capacity() {
    let temp = tempfile::tempdir().unwrap();
    for name in ["one", "two", "three"] {
        std::fs::write(temp.path().join(name), name).unwrap();
    }

    let error = bounded_runtime_directory_entries(temp.path(), 2)
        .unwrap_err()
        .to_string();

    assert!(
        error.contains("entry activation snapshot budget"),
        "{error}"
    );
}

#[test]
fn activation_snapshot_budgets_reject_entry_and_byte_overflow() {
    let path = Path::new("fixture");
    let mut entry_budget = SnapshotBudget::default();
    let entry_error = entry_budget
        .record_entries(path, MAX_SNAPSHOT_FILES + 1)
        .unwrap_err()
        .to_string();
    assert!(
        entry_error.contains("entry activation snapshot budget"),
        "{entry_error}"
    );

    let mut byte_budget = SnapshotBudget::default();
    let byte_error = byte_budget
        .record_bytes(
            path,
            usize::try_from(crate::configuration::MAX_BOOTSTRAP_IDENTITY_FILE_BYTES).unwrap() + 1,
        )
        .unwrap_err()
        .to_string();
    assert!(
        byte_error.contains("byte activation snapshot budget"),
        "{byte_error}"
    );

    let mut closure = RuntimeClosureSources {
        entries: MAX_SNAPSHOT_FILES,
        ..RuntimeClosureSources::default()
    };
    let closure_error = closure.record_entry().unwrap_err().to_string();
    assert!(
        closure_error.contains("entry activation snapshot budget"),
        "{closure_error}"
    );
}

#[test]
fn activation_snapshot_rejects_identity_mismatch_missing_files_and_policy_denial() {
    let temp = tempfile::tempdir().unwrap();
    let plugin_dir = temp.path().join("plugin");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    let manifest = write_dynamic_manifest(&plugin_dir, "acme.snapshot-contracts");

    let identity_error = DynamicPluginActivationSnapshot::create(
        manifest.to_string_lossy().as_ref(),
        "acme.other-plugin",
        DynamicPluginKind::Worker,
        None,
        &crate::plugins::policy::DynamicPluginHostPolicy::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(
        identity_error.contains("identity changed"),
        "{identity_error}"
    );

    let contents = std::fs::read_to_string(&manifest).unwrap();
    std::fs::write(
        &manifest,
        contents.replace("entrypoint = \"plugin.py\"", "entrypoint = \"other.py\""),
    )
    .unwrap();
    std::fs::write(plugin_dir.join("other.py"), b"print('other')\n").unwrap();
    let entrypoint_error = DynamicPluginActivationSnapshot::create(
        manifest.to_string_lossy().as_ref(),
        "acme.snapshot-contracts",
        DynamicPluginKind::Worker,
        None,
        &crate::plugins::policy::DynamicPluginHostPolicy::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(
        entrypoint_error.contains("integrity-checked source.artifact"),
        "{entrypoint_error}"
    );
    let closure_error =
        dynamic_plugin_runtime_closure_digest(manifest.to_string_lossy().as_ref(), None)
            .unwrap_err()
            .to_string();
    assert!(
        closure_error.contains("integrity-checked source.artifact"),
        "{closure_error}"
    );

    std::fs::write(&manifest, contents).unwrap();
    let blocked = crate::plugins::policy::DynamicPluginHostPolicy {
        defaults: crate::plugins::policy::DynamicPluginHostPolicyEffect {
            allowed: Some(false),
            ..crate::plugins::policy::DynamicPluginHostPolicyEffect::default()
        },
        ..crate::plugins::policy::DynamicPluginHostPolicy::default()
    };
    let policy_error = DynamicPluginActivationSnapshot::create(
        manifest.to_string_lossy().as_ref(),
        "acme.snapshot-contracts",
        DynamicPluginKind::Worker,
        None,
        &blocked,
    )
    .unwrap_err()
    .to_string();
    assert!(
        policy_error.contains("violates host policy"),
        "{policy_error}"
    );

    std::fs::remove_file(plugin_dir.join("plugin.py")).unwrap();
    let missing_error = DynamicPluginActivationSnapshot::create(
        manifest.to_string_lossy().as_ref(),
        "acme.snapshot-contracts",
        DynamicPluginKind::Worker,
        None,
        &crate::plugins::policy::DynamicPluginHostPolicy::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(
        missing_error.contains("failed to normalize"),
        "{missing_error}"
    );
}

#[test]
fn activation_snapshot_copies_declared_signature_into_stable_identity() {
    let temp = tempfile::tempdir().unwrap();
    let plugin_dir = temp.path().join("plugin");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    let manifest = write_dynamic_manifest_with_options(
        &plugin_dir,
        "acme.signed-snapshot",
        &["plugin_worker"],
        Some("plugin.py.sig"),
    );
    std::fs::write(plugin_dir.join("plugin.py.sig"), b"fixture signature\n").unwrap();

    let snapshot = DynamicPluginActivationSnapshot::create(
        manifest.to_string_lossy().as_ref(),
        "acme.signed-snapshot",
        DynamicPluginKind::Worker,
        None,
        &crate::plugins::policy::DynamicPluginHostPolicy::default(),
    )
    .unwrap();

    let signature_logical = snapshot
        .identity_files
        .keys()
        .find(|path| path.ends_with("plugin.py.sig"))
        .cloned()
        .expect("signature is part of the stable snapshot identity");
    assert!(snapshot.identity_file(&signature_logical).is_some());
    assert_eq!(
        snapshot.closure_digest(),
        dynamic_plugin_runtime_closure_digest(manifest.to_string_lossy().as_ref(), None).unwrap()
    );
}

#[test]
fn python_snapshot_contract_requires_environment_and_trusted_source_digest() {
    let temp = tempfile::tempdir().unwrap();
    let plugin_dir = temp.path().join("plugin");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    let manifest = write_python_dynamic_manifest(&plugin_dir, "acme.python-contract");

    let snapshot_error = DynamicPluginActivationSnapshot::create(
        manifest.to_string_lossy().as_ref(),
        "acme.python-contract",
        DynamicPluginKind::Worker,
        None,
        &crate::plugins::policy::DynamicPluginHostPolicy::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(
        snapshot_error.contains("no managed environment"),
        "{snapshot_error}"
    );
    let closure_error =
        dynamic_plugin_runtime_closure_digest(manifest.to_string_lossy().as_ref(), None)
            .unwrap_err()
            .to_string();
    assert!(
        closure_error.contains("no managed environment"),
        "{closure_error}"
    );

    let contents = std::fs::read_to_string(&manifest).unwrap();
    let integrity = contents
        .find("[integrity]")
        .expect("fixture has integrity section");
    let load = contents.find("[load]").expect("fixture has load section");
    std::fs::write(
        &manifest,
        format!("{}{}", &contents[..integrity], &contents[load..]),
    )
    .unwrap();
    let digest_error = dynamic_plugin_runtime_closure_digest(
        manifest.to_string_lossy().as_ref(),
        Some(temp.path().join("environment").to_string_lossy().as_ref()),
    )
    .unwrap_err()
    .to_string();
    assert!(
        digest_error.contains("requires integrity.sha256"),
        "{digest_error}"
    );
}

#[cfg(unix)]
#[test]
fn snapshot_directory_copy_preserves_python_launcher_and_rejects_special_entries() {
    use std::ffi::CString;
    use std::os::unix::fs::symlink;

    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("environment");
    let bin = source.join("bin");
    std::fs::create_dir_all(&bin).unwrap();
    let interpreter = temp.path().join("managed-python");
    std::fs::write(&interpreter, b"python").unwrap();
    symlink(&interpreter, bin.join("python3.11")).unwrap();
    let destination = temp.path().join("snapshot");
    let mut copied = HashMap::new();
    let mut budget = SnapshotBudget::default();

    copy_snapshot_directory(
        &source,
        &destination,
        &mut copied,
        &mut budget,
        false,
        &mut Vec::new(),
    )
    .unwrap();

    assert!(
        std::fs::symlink_metadata(destination.join("bin/python3.11"))
            .unwrap()
            .file_type()
            .is_symlink()
    );
    assert!(!is_python_venv_launcher(Path::new("/")));

    let fifo_source = temp.path().join("fifo-source");
    std::fs::create_dir(&fifo_source).unwrap();
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
    std::fs::write(&regular, b"regular").unwrap();
    let destination_directory = temp.path().join("destination-directory");
    std::fs::create_dir(&destination_directory).unwrap();
    let write_error = copy_snapshot_regular_file(
        &regular,
        &destination_directory,
        &mut HashMap::new(),
        &mut SnapshotBudget::default(),
        "fixture",
    )
    .unwrap_err()
    .to_string();
    assert!(write_error.contains("failed to write dynamic plugin snapshot file"));
}

#[cfg(unix)]
#[test]
fn snapshot_directory_walk_rejects_missing_cycles_dangling_links_and_depth() {
    use std::os::unix::fs::symlink;

    let temp = tempfile::tempdir().unwrap();
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
        normalization_error.contains("failed to normalize"),
        "{normalization_error}"
    );

    let source = temp.path().join("source");
    std::fs::create_dir(&source).unwrap();
    let canonical = source.canonicalize().unwrap();
    let cycle_error = copy_snapshot_directory_contents(
        &source,
        &temp.path().join("cycle-destination"),
        &mut HashMap::new(),
        &mut SnapshotBudget::default(),
        false,
        &mut vec![canonical.clone()],
    )
    .unwrap_err()
    .to_string();
    assert!(cycle_error.contains("symlink cycle"), "{cycle_error}");

    let destination_file = temp.path().join("destination-file");
    std::fs::write(&destination_file, b"file").unwrap();
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
        destination_error.contains("failed to create"),
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
        dangling_error.contains("failed to resolve"),
        "{dangling_error}"
    );

    let closure_cycle = collect_runtime_closure_directory_contents(
        &source,
        Path::new("runtime"),
        false,
        &mut vec![canonical],
        &mut RuntimeClosureSources::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(closure_cycle.contains("symlink cycle"), "{closure_cycle}");

    let depth_error = collect_snapshot_files(
        &source,
        &source,
        &mut Vec::new(),
        Some(MAX_SNAPSHOT_DEPTH),
        &mut 0,
    )
    .unwrap_err()
    .to_string();
    assert!(depth_error.contains("traversal depth"), "{depth_error}");
}

#[cfg(unix)]
#[test]
fn snapshot_file_and_closure_helpers_cover_external_and_invalid_sources() {
    use std::ffi::CString;
    use std::os::unix::fs::symlink;

    let temp = tempfile::tempdir().unwrap();
    let plugin_dir = temp.path().join("plugin");
    std::fs::create_dir(&plugin_dir).unwrap();
    let manifest = plugin_dir.join("relay-plugin.toml");
    std::fs::write(&manifest, b"fixture").unwrap();
    let root = temp.path().join("snapshot");
    std::fs::create_dir(&root).unwrap();

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
        missing_error.contains("failed to normalize"),
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
    std::fs::write(&external, b"external artifact").unwrap();
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
    assert_eq!(std::fs::read(copied).unwrap(), b"external artifact");

    let mut closure = RuntimeClosureSources::default();
    let closure_missing =
        collect_declared_runtime_closure_file(&manifest, "missing.bin", "artifact", &mut closure)
            .unwrap_err()
            .to_string();
    assert!(
        closure_missing.contains("failed to normalize"),
        "{closure_missing}"
    );
    let closure_root =
        collect_declared_runtime_closure_file(&manifest, "/", "library", &mut closure)
            .unwrap_err()
            .to_string();
    assert!(
        closure_root.contains("has no parent directory"),
        "{closure_root}"
    );
    collect_declared_runtime_closure_file(
        &manifest,
        external.to_string_lossy().as_ref(),
        "artifact",
        &mut closure,
    )
    .unwrap();
    assert!(
        closure
            .files
            .keys()
            .any(|path| path.ends_with("external-artifact.bin"))
    );

    let missing_directory_error = collect_runtime_closure_directory_contents(
        &temp.path().join("missing-directory"),
        Path::new("runtime"),
        false,
        &mut Vec::new(),
        &mut RuntimeClosureSources::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(
        missing_directory_error.contains("failed to normalize"),
        "{missing_directory_error}"
    );

    let walk = temp.path().join("walk");
    std::fs::create_dir(&walk).unwrap();
    std::fs::create_dir(walk.join("__pycache__")).unwrap();
    std::fs::write(walk.join("cached.pyc"), b"cache").unwrap();
    std::fs::write(walk.join("module.py"), b"module").unwrap();
    let mut skipped = RuntimeClosureSources::default();
    collect_runtime_closure_directory(
        &walk,
        Path::new("runtime"),
        true,
        &mut Vec::new(),
        &mut skipped,
    )
    .unwrap();
    assert_eq!(skipped.files.len(), 1);
    assert!(skipped.files.contains_key(Path::new("runtime/module.py")));

    symlink(temp.path().join("absent"), walk.join("dangling")).unwrap();
    let dangling_error = collect_runtime_closure_directory(
        &walk,
        Path::new("runtime"),
        false,
        &mut Vec::new(),
        &mut RuntimeClosureSources::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(
        dangling_error.contains("failed to resolve"),
        "{dangling_error}"
    );
    std::fs::remove_file(walk.join("dangling")).unwrap();

    let fifo = walk.join("worker.pipe");
    let fifo_c = CString::new(fifo.as_os_str().as_encoded_bytes()).unwrap();
    // SAFETY: `fifo_c` is a valid NUL-terminated path and the mode contains only permission bits.
    assert_eq!(unsafe { libc::mkfifo(fifo_c.as_ptr(), 0o600) }, 0);
    let fifo_error = collect_runtime_closure_directory(
        &walk,
        Path::new("runtime"),
        false,
        &mut Vec::new(),
        &mut RuntimeClosureSources::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(
        fifo_error.contains("regular file or directory"),
        "{fifo_error}"
    );

    let one_file = temp.path().join("one-file");
    std::fs::create_dir(&one_file).unwrap();
    std::fs::write(one_file.join("entry"), b"entry").unwrap();
    let mut entries = MAX_SNAPSHOT_FILES;
    let entry_error =
        collect_snapshot_files(&one_file, &one_file, &mut Vec::new(), None, &mut entries)
            .unwrap_err()
            .to_string();
    assert!(entry_error.contains("verification budget"), "{entry_error}");

    make_snapshot_removable(&temp.path().join("already-removed"));
}

#[test]
fn python_environment_entry_budget_counts_skipped_cache_entries() {
    let temp = tempfile::tempdir().unwrap();
    std::fs::write(temp.path().join("ignored.pyc"), b"cache").unwrap();
    std::fs::write(temp.path().join("module.py"), b"module").unwrap();
    std::fs::write(temp.path().join("metadata.txt"), b"metadata").unwrap();

    let error =
        environment::test_environment_tree_digest_with_entry_limit(temp.path(), 2).unwrap_err();

    assert!(error.contains("2-entry attestation budget"), "{error}");
}

#[test]
fn python_activation_snapshot_is_attested_copied_and_tamper_evident() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let plugin_dir = temp.path().join("python-plugin");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    let manifest_path = write_python_dynamic_manifest(&plugin_dir, "acme.python-snapshot");
    let environment_name = Sha256::digest(b"acme.python-snapshot")
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let environment_path = temp
        .path()
        .join(environment::MANAGED_ENVIRONMENTS_DIR)
        .join(environment_name);
    let interpreter = environment::environment_python_path(&environment_path);
    std::fs::create_dir_all(interpreter.parent().unwrap()).unwrap();
    std::fs::write(&interpreter, b"attested interpreter").unwrap();
    let installed_module = environment_path
        .join("site-packages")
        .join("plugin-data.txt");
    std::fs::create_dir_all(installed_module.parent().unwrap()).unwrap();
    std::fs::write(&installed_module, b"safe installed module").unwrap();
    let (manifest, _) = DynamicPluginManifest::load_from_path(&manifest_path).unwrap();
    let source_artifact_sha256 = manifest
        .integrity
        .as_ref()
        .and_then(|integrity| integrity.sha256.as_deref())
        .unwrap();
    environment::write_environment_attestation(&environment_path, source_artifact_sha256).unwrap();

    let snapshot = DynamicPluginActivationSnapshot::create(
        manifest_path.to_string_lossy().as_ref(),
        "acme.python-snapshot",
        DynamicPluginKind::Worker,
        Some(environment_path.to_string_lossy().as_ref()),
        &crate::plugins::policy::DynamicPluginHostPolicy::default(),
    )
    .unwrap();
    let source_closure_digest = dynamic_plugin_runtime_closure_digest(
        manifest_path.to_string_lossy().as_ref(),
        Some(environment_path.to_string_lossy().as_ref()),
    )
    .unwrap();
    assert_eq!(snapshot.closure_digest(), source_closure_digest);
    let copied_environment = PathBuf::from(snapshot.activation_environment_ref().unwrap());
    assert_ne!(copied_environment, environment_path);
    assert_eq!(
        copied_environment.parent().unwrap().file_name(),
        Some(OsStr::new(environment::MANAGED_ENVIRONMENTS_DIR))
    );
    assert_eq!(copied_environment.file_name(), environment_path.file_name());
    assert_eq!(
        std::fs::read(copied_environment.join("site-packages/plugin-data.txt")).unwrap(),
        b"safe installed module"
    );
    let load_error = match load_worker_plugins(vec![WorkerPluginLoadSpec {
        plugin_id: "acme.python-snapshot".into(),
        manifest_ref: snapshot.activation_manifest_ref(),
        environment_ref: snapshot.activation_environment_ref().map(ToOwned::to_owned),
        config: Map::new(),
    }]) {
        Ok(_) => panic!("fixture Python worker unexpectedly activated"),
        Err(error) => error.to_string(),
    };
    assert!(
        !load_error.contains("not the lifecycle-managed path"),
        "{load_error}"
    );

    std::fs::write(&installed_module, b"tampered installed module").unwrap();

    assert!(
        environment::verify_environment_attestation(&environment_path, source_artifact_sha256)
            .is_err()
    );
    assert_eq!(
        std::fs::read(copied_environment.join("site-packages/plugin-data.txt")).unwrap(),
        b"safe installed module"
    );
    snapshot.verify_current().unwrap();
    dynamic_plugin_runtime_closure_digest(
        manifest_path.to_string_lossy().as_ref(),
        Some(environment_path.to_string_lossy().as_ref()),
    )
    .expect("hook preflight should authenticate the attestation without rehashing the environment");
    let changed_error = DynamicPluginActivationSnapshot::create(
        manifest_path.to_string_lossy().as_ref(),
        "acme.python-snapshot",
        DynamicPluginKind::Worker,
        Some(environment_path.to_string_lossy().as_ref()),
        &crate::plugins::policy::DynamicPluginHostPolicy::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(
        changed_error.contains("changed after provisioning"),
        "{changed_error}"
    );
    let attestation_path = environment_path.join(environment::ENVIRONMENT_ATTESTATION_FILE);
    let mut forged: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&attestation_path).unwrap()).unwrap();
    forged["environment_sha256"] =
        serde_json::json!(environment::environment_tree_digest(&environment_path).unwrap());
    std::fs::write(
        &attestation_path,
        serde_json::to_vec_pretty(&forged).unwrap(),
    )
    .unwrap();
    let error = DynamicPluginActivationSnapshot::create(
        manifest_path.to_string_lossy().as_ref(),
        "acme.python-snapshot",
        DynamicPluginKind::Worker,
        Some(environment_path.to_string_lossy().as_ref()),
        &crate::plugins::policy::DynamicPluginHostPolicy::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("failed authentication"), "{error}");
}

#[test]
fn python_entrypoint_validation_reports_each_authored_contract_error() {
    let temp = tempfile::tempdir().unwrap();
    let plugin_dir = temp.path().join("plugin");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    let manifest_path = write_python_dynamic_manifest(&plugin_dir, "acme.python-validation");
    let (manifest, manifest_ref) = DynamicPluginManifest::load_from_path(&manifest_path).unwrap();

    let mut missing_source = manifest.clone();
    missing_source.source = None;
    let error = environment::validate_python_entrypoint_artifact(&missing_source, &manifest_ref)
        .unwrap_err();
    assert!(
        error.contains("must declare source.manifest_root"),
        "{error}"
    );

    let mut missing_separator = manifest.clone();
    let DynamicPluginManifestLoad::Worker(load) = &mut missing_separator.load else {
        panic!("fixture must be a worker plugin");
    };
    load.entrypoint = Some("plugin".into());
    let error = environment::validate_python_entrypoint_artifact(&missing_separator, &manifest_ref)
        .unwrap_err();
    assert!(error.contains("module:function form"), "{error}");

    let mut empty_module = manifest.clone();
    let DynamicPluginManifestLoad::Worker(load) = &mut empty_module.load else {
        panic!("fixture must be a worker plugin");
    };
    load.entrypoint = Some(":main".into());
    let error =
        environment::validate_python_entrypoint_artifact(&empty_module, &manifest_ref).unwrap_err();
    assert!(error.contains("module:function form"), "{error}");

    let mut missing_root = manifest;
    missing_root
        .source
        .as_mut()
        .expect("fixture declares source")
        .manifest_root = Some("missing-root".into());
    let error =
        environment::validate_python_entrypoint_artifact(&missing_root, &manifest_ref).unwrap_err();
    assert!(
        error.contains("could not resolve Python plugin source.manifest_root"),
        "{error}"
    );
}

#[test]
fn python_environment_attestation_rejects_invalid_json_and_source_identity_drift() {
    let temp = tempfile::tempdir().unwrap();
    let environment_path = temp.path().join("environment");
    std::fs::create_dir_all(&environment_path).unwrap();
    let attestation_path = environment_path.join(environment::ENVIRONMENT_ATTESTATION_FILE);

    std::fs::write(&attestation_path, "{not-json").unwrap();
    let error =
        environment::read_environment_attestation(&environment_path, "expected").unwrap_err();
    assert!(error.contains("attestation"), "{error}");
    assert!(error.contains("is invalid"), "{error}");

    std::fs::write(
        &attestation_path,
        serde_json::to_vec(&serde_json::json!({
            "version": 1,
            "source_artifact_sha256": "different",
            "environment_sha256": "0".repeat(64),
            "authentication": "unused"
        }))
        .unwrap(),
    )
    .unwrap();
    let error =
        environment::read_environment_attestation(&environment_path, "expected").unwrap_err();
    assert!(
        error.contains("does not match the trusted source artifact"),
        "{error}"
    );
}

#[test]
fn add_registers_dynamic_plugin_in_project_plugins_toml() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    write_dynamic_manifest(&plugin_dir, "acme.guardrail");

    add(
        PluginsAddRequest {
            scope: ConfigurationScope {
                project: true,
                ..ConfigurationScope::default()
            },
            path: plugin_dir.clone(),
        },
        &crate::server::GatewayOverrides::default(),
    )
    .unwrap();

    let plugins_toml = temp.path().join(".nemo-relay").join("plugins.toml");
    let rendered = std::fs::read_to_string(&plugins_toml).unwrap();
    assert!(rendered.contains("[[plugins.dynamic]]"));
    assert!(rendered.contains("relay-plugin.toml"));

    let resolved = resolve_plugins_config(None).unwrap();
    assert_eq!(resolved.dynamic_plugins.len(), 1);
    assert_eq!(resolved.dynamic_plugins[0].plugin_id, "acme.guardrail");
}

#[test]
fn add_rejects_unreadable_declared_config_schema() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("schema-missing");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    write_dynamic_manifest_with_config_schema(
        &plugin_dir,
        "acme.schema-missing",
        &serde_json::json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object"
        }),
    );
    std::fs::remove_file(plugin_dir.join("config.schema.json")).unwrap();

    let error = add(
        PluginsAddRequest {
            scope: ConfigurationScope {
                project: true,
                ..ConfigurationScope::default()
            },
            path: plugin_dir,
        },
        &GatewayOverrides::default(),
    )
    .unwrap_err()
    .to_string();

    assert!(error.contains("acme.schema-missing"), "{error}");
    assert!(error.contains("config.schema.json"), "{error}");
    assert!(error.contains("failed to read schema"), "{error}");
}

#[test]
fn validate_path_rejects_invalid_declared_config_schema() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("schema-invalid");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    write_dynamic_manifest_with_config_schema(
        &plugin_dir,
        "acme.schema-invalid",
        &serde_json::json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": 42
        }),
    );

    let error = validate(
        PluginsValidateRequest {
            target: plugin_dir.to_string_lossy().into_owned(),
            json: false,
        },
        &GatewayOverrides::default(),
    )
    .unwrap_err()
    .to_string();

    assert!(error.contains("acme.schema-invalid"), "{error}");
    assert!(error.contains("schema is invalid"), "{error}");
}

#[test]
fn validate_id_checks_resolved_host_config_against_declared_schema() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("schema-config");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    write_dynamic_manifest_with_config_schema(
        &plugin_dir,
        "acme.schema-config",
        &serde_json::json!({
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {"port": {"type": "integer"}}
        }),
    );
    let server = GatewayOverrides::default();
    add(
        PluginsAddRequest {
            scope: ConfigurationScope {
                project: true,
                ..ConfigurationScope::default()
            },
            path: plugin_dir,
        },
        &server,
    )
    .unwrap();

    let plugins_toml = temp.path().join(".nemo-relay").join("plugins.toml");
    let mut rendered = std::fs::read_to_string(&plugins_toml).unwrap();
    rendered.push_str(
        r#"
[plugins.dynamic.config]
port = "not-an-integer"
"#,
    );
    std::fs::write(&plugins_toml, rendered).unwrap();

    let error = validate(
        PluginsValidateRequest {
            target: "acme.schema-config".into(),
            json: false,
        },
        &server,
    )
    .unwrap_err()
    .to_string();

    assert!(error.contains("acme.schema-config"), "{error}");
    assert!(error.contains("JSON pointer '/port'"), "{error}");
}

#[test]
fn add_provisions_persists_and_removes_managed_python_environment() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("python");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    write_python_dynamic_manifest(&plugin_dir, "  acme.python  ");
    let runner = FakePythonEnvironmentRunner::default();
    let server = GatewayOverrides::default();

    add_with_environment_runner(
        PluginsAddRequest {
            scope: ConfigurationScope {
                project: true,
                ..ConfigurationScope::default()
            },
            path: plugin_dir.clone(),
        },
        &server,
        &runner,
    )
    .unwrap();

    let scopes = load_scoped_registries(None).unwrap();
    let added = find_record_by_id(&scopes, "acme.python")
        .unwrap()
        .expect("Python record should exist");
    let environment_ref = added
        .record
        .source
        .environment_ref
        .as_deref()
        .expect("managed environment should be persisted");
    let environment_path = PathBuf::from(environment_ref);
    assert!(environment_path.is_absolute());
    let expected_environment_name = Sha256::digest(b"acme.python")
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    assert_eq!(
        environment_path.file_name(),
        Some(OsStr::new(&expected_environment_name))
    );
    assert!(
        environment_path
            .parent()
            .is_some_and(|parent| parent.ends_with(".dynamic-plugin-environments"))
    );
    assert!(environment::environment_python_path(&environment_path).is_file());
    assert_eq!(
        added.record.status.validation.environment,
        DynamicPluginCheckState::Valid
    );
    let (manifest, manifest_ref) =
        DynamicPluginManifest::load_from_path(plugin_dir.join("relay-plugin.toml")).unwrap();
    let inspect = serde_json::to_value(responses::inspect_success(
        "plugins inspect",
        "acme.python",
        &added,
        &manifest,
        &manifest_ref,
        None,
    ))
    .unwrap();
    assert_eq!(
        inspect["data"]["environment_state"],
        serde_json::json!("valid")
    );
    assert_eq!(
        inspect["data"]["source"]["environment_ref"],
        serde_json::json!(environment_ref)
    );
    let calls = runner.calls();
    assert_eq!(calls.len(), 2);
    assert_eq!(
        calls[0].0,
        OsString::from(if cfg!(windows) { "python" } else { "python3" })
    );
    assert_eq!(
        calls[0].1,
        vec![
            OsString::from("-m"),
            OsString::from("venv"),
            environment_path.as_os_str().to_owned(),
        ]
    );
    assert_eq!(
        PathBuf::from(&calls[1].0),
        environment::environment_python_path(&environment_path)
    );
    assert_eq!(
        calls[1].1,
        vec![
            OsString::from("-m"),
            OsString::from("pip"),
            OsString::from("install"),
            plugin_dir.canonicalize().unwrap().into_os_string(),
        ]
    );
    assert!(
        !calls[1]
            .1
            .iter()
            .any(|arg| arg == "-e" || arg == "--editable")
    );
    enable(
        PluginsEnableRequest {
            id: "acme.python".into(),
        },
        &server,
    )
    .unwrap();
    let resolved = resolve_plugins_config(None).unwrap();
    let active = active_dynamic_plugin_components(None, &resolved).unwrap();
    assert_eq!(active.len(), 1);
    assert_eq!(active[0].environment_ref.as_deref(), Some(environment_ref));

    let stale_marker = environment_path.join("stale-marker");
    std::fs::write(&stale_marker, b"stale").unwrap();
    remove(
        PluginsRemoveRequest {
            id: "acme.python".into(),
        },
        &server,
    )
    .unwrap();
    assert!(!environment_path.exists());
    let scopes = load_scoped_registries(None).unwrap();
    let removed = find_record_by_id(&scopes, "acme.python")
        .unwrap()
        .expect("tombstone should remain");
    assert!(removed.record.is_tombstoned());
    assert_eq!(removed.record.source.environment_ref, None);

    add_with_environment_runner(
        PluginsAddRequest {
            scope: ConfigurationScope {
                project: true,
                ..ConfigurationScope::default()
            },
            path: plugin_dir,
        },
        &server,
        &runner,
    )
    .unwrap();
    assert!(environment::environment_python_path(&environment_path).is_file());
    assert!(!stale_marker.exists());
}

#[test]
fn add_rolls_back_python_environment_when_installation_fails() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("python");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    write_python_dynamic_manifest(&plugin_dir, "acme.python-failure");
    let runner = FakePythonEnvironmentRunner::failing_install();

    let error = add_with_environment_runner(
        PluginsAddRequest {
            scope: ConfigurationScope {
                project: true,
                ..ConfigurationScope::default()
            },
            path: plugin_dir,
        },
        &GatewayOverrides::default(),
        &runner,
    )
    .expect_err("pip failure should abort plugin registration");

    let (_, _, kind, code, message) = error
        .as_plugin_lifecycle_error_context()
        .expect("environment failure should be structured");
    assert_eq!(kind, PluginLifecycleFailureKind::Failed);
    assert_eq!(code, Some("environment_failed"));
    assert!(message.contains("fixture pip failure"));
    assert_eq!(runner.calls().len(), 2);
    let managed_root = temp
        .path()
        .join(".nemo-relay")
        .join(".dynamic-plugin-environments");
    assert!(!managed_root.exists() || std::fs::read_dir(managed_root).unwrap().next().is_none());
    assert!(
        resolve_plugins_config(None)
            .unwrap()
            .dynamic_plugins
            .is_empty()
    );
}

#[test]
fn enable_rejects_missing_managed_python_environment() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("python");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    write_python_dynamic_manifest(&plugin_dir, "acme.python-missing");
    let runner = FakePythonEnvironmentRunner::default();
    let server = GatewayOverrides::default();
    add_with_environment_runner(
        PluginsAddRequest {
            scope: ConfigurationScope {
                project: true,
                ..ConfigurationScope::default()
            },
            path: plugin_dir,
        },
        &server,
        &runner,
    )
    .unwrap();
    let scopes = load_scoped_registries(None).unwrap();
    let environment_ref = find_record_by_id(&scopes, "acme.python-missing")
        .unwrap()
        .unwrap()
        .record
        .source
        .environment_ref
        .clone()
        .unwrap();
    std::fs::remove_dir_all(&environment_ref).unwrap();

    let error = enable(
        PluginsEnableRequest {
            id: "acme.python-missing".into(),
        },
        &server,
    )
    .expect_err("missing managed environment should prevent activation");

    let (_, _, kind, code, message) = error
        .as_plugin_lifecycle_error_context()
        .expect("environment failure should be structured");
    assert_eq!(kind, PluginLifecycleFailureKind::Refused);
    assert_eq!(code, Some("environment_failed"));
    assert!(message.contains("is unavailable"));
    let scopes = load_scoped_registries(None).unwrap();
    let record = find_record_by_id(&scopes, "acme.python-missing")
        .unwrap()
        .unwrap()
        .record;
    assert!(!record.spec.enabled);
    assert_eq!(
        record.status.validation.environment,
        DynamicPluginCheckState::Invalid
    );
}

#[test]
fn enable_rejects_python_environment_outside_managed_location() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("python");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    write_python_dynamic_manifest(&plugin_dir, "acme.python-outside");
    let runner = FakePythonEnvironmentRunner::default();
    let server = GatewayOverrides::default();
    add_with_environment_runner(
        PluginsAddRequest {
            scope: ConfigurationScope {
                project: true,
                ..ConfigurationScope::default()
            },
            path: plugin_dir,
        },
        &server,
        &runner,
    )
    .unwrap();

    let outside = temp.path().join("outside");
    let outside_python = environment::environment_python_path(&outside);
    std::fs::create_dir_all(outside_python.parent().unwrap()).unwrap();
    std::fs::write(&outside_python, b"not managed by Relay").unwrap();
    let mut scopes = load_scoped_registries(None).unwrap();
    let scope = scopes
        .iter_mut()
        .find(|scope| scope.registry.get("acme.python-outside").is_some())
        .unwrap();
    scope
        .registry
        .update_environment(
            "acme.python-outside",
            Some(outside.display().to_string()),
            DynamicPluginCheckState::Valid,
        )
        .unwrap();
    scope.save().unwrap();

    let error = enable(
        PluginsEnableRequest {
            id: "acme.python-outside".into(),
        },
        &server,
    )
    .expect_err("an unmanaged Python environment should prevent activation");

    let (_, _, kind, code, message) = error
        .as_plugin_lifecycle_error_context()
        .expect("environment refusal should be structured");
    assert_eq!(kind, PluginLifecycleFailureKind::Refused);
    assert_eq!(code, Some("environment_failed"));
    assert!(message.contains("is unavailable"));
    assert!(outside.exists());
    let scopes = load_scoped_registries(None).unwrap();
    let record = find_record_by_id(&scopes, "acme.python-outside")
        .unwrap()
        .unwrap()
        .record;
    assert!(!record.spec.enabled);
    assert_eq!(
        record.status.validation.environment,
        DynamicPluginCheckState::Invalid
    );
}

#[test]
fn add_requires_manifest_root_for_python_workers() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("python");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    let manifest = write_python_dynamic_manifest(&plugin_dir, "acme.no-root");
    let contents = std::fs::read_to_string(&manifest)
        .unwrap()
        .replace("manifest_root = \".\"\n", "");
    std::fs::write(&manifest, contents).unwrap();
    let runner = FakePythonEnvironmentRunner::default();

    let error = add_with_environment_runner(
        PluginsAddRequest {
            scope: ConfigurationScope {
                project: true,
                ..ConfigurationScope::default()
            },
            path: plugin_dir,
        },
        &GatewayOverrides::default(),
        &runner,
    )
    .expect_err("Python plugins without manifest_root should fail");

    let (_, _, kind, code, message) = error
        .as_plugin_lifecycle_error_context()
        .expect("environment failure should be structured");
    assert_eq!(kind, PluginLifecycleFailureKind::Failed);
    assert_eq!(code, Some("environment_failed"));
    assert!(message.contains("source.manifest_root"));
    assert!(runner.calls().is_empty());
}

#[test]
fn add_rejects_python_entrypoint_module_that_is_not_integrity_checked_artifact() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("python");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    let manifest = write_python_dynamic_manifest(&plugin_dir, "acme.unsigned-entrypoint");
    std::fs::write(
        plugin_dir.join("unsigned_sibling.py"),
        b"def main(): pass\n",
    )
    .unwrap();
    let contents = std::fs::read_to_string(&manifest).unwrap().replace(
        "entrypoint = \"plugin:main\"",
        "entrypoint = \"unsigned_sibling:main\"",
    );
    std::fs::write(&manifest, contents).unwrap();
    let runner = FakePythonEnvironmentRunner::default();

    let error = add_with_environment_runner(
        PluginsAddRequest {
            scope: ConfigurationScope {
                project: true,
                ..ConfigurationScope::default()
            },
            path: plugin_dir,
        },
        &GatewayOverrides::default(),
        &runner,
    )
    .expect_err("an unsigned sibling module must not become the executed entrypoint");

    let (_, _, kind, code, message) = error
        .as_plugin_lifecycle_error_context()
        .expect("environment refusal should be structured");
    assert_eq!(kind, PluginLifecycleFailureKind::Failed);
    assert_eq!(code, Some("environment_failed"));
    assert!(message.contains("executed entrypoint module"), "{message}");
    assert!(message.contains("integrity-checked artifact"), "{message}");
    assert!(runner.calls().is_empty());
}

#[test]
fn activation_snapshot_rejects_ambiguous_python_entrypoint_module() {
    let temp = tempfile::tempdir().unwrap();
    let plugin_dir = temp.path().join("python-plugin");
    std::fs::create_dir_all(plugin_dir.join("plugin")).unwrap();
    let manifest = write_python_dynamic_manifest(&plugin_dir, "acme.ambiguous-entrypoint");
    std::fs::write(plugin_dir.join("plugin/__init__.py"), b"def main(): pass\n").unwrap();

    let error = DynamicPluginActivationSnapshot::create(
        manifest.to_string_lossy().as_ref(),
        "acme.ambiguous-entrypoint",
        DynamicPluginKind::Worker,
        None,
        &crate::plugins::policy::DynamicPluginHostPolicy::default(),
    )
    .unwrap_err()
    .to_string();

    assert!(error.contains("exactly one source module"), "{error}");
    assert!(error.contains("plugin.py"), "{error}");
    assert!(error.contains("__init__.py"), "{error}");
}

#[test]
fn managed_environment_cleanup_refuses_paths_outside_lifecycle_directory() {
    let temp = tempfile::tempdir().unwrap();
    let state_path = temp.path().join(".dynamic-plugins.json");
    let outside = temp.path().join("outside");
    std::fs::create_dir_all(&outside).unwrap();

    let error = environment::remove_managed_environment(
        &state_path,
        "acme.python",
        outside.to_string_lossy().as_ref(),
    )
    .expect_err("unmanaged environment must not be removed");

    assert!(error.contains("refusing to delete Python environment"));
    assert!(outside.exists());
}

#[cfg(unix)]
#[test]
fn managed_environment_cleanup_refuses_symlinked_environment() {
    use std::os::unix::fs::symlink;

    let temp = tempfile::tempdir().unwrap();
    let state_path = temp.path().join(".dynamic-plugins.json");
    let outside = temp.path().join("outside");
    std::fs::create_dir_all(&outside).unwrap();
    let environment_name = Sha256::digest(b"acme.python")
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let environment = temp
        .path()
        .join(".dynamic-plugin-environments")
        .join(environment_name);
    std::fs::create_dir_all(environment.parent().unwrap()).unwrap();
    symlink(&outside, &environment).unwrap();

    let error = environment::remove_managed_environment(
        &state_path,
        "acme.python",
        environment.to_string_lossy().as_ref(),
    )
    .expect_err("symlinked environment must not be removed");

    assert!(error.contains("is not a directory"));
    assert!(environment.is_symlink());
    assert!(outside.exists());
}

#[test]
fn remove_can_retry_after_guarded_environment_cleanup_failure() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("python");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    write_python_dynamic_manifest(&plugin_dir, "acme.python-retry");
    let runner = FakePythonEnvironmentRunner::default();
    let server = GatewayOverrides::default();
    add_with_environment_runner(
        PluginsAddRequest {
            scope: ConfigurationScope {
                project: true,
                ..ConfigurationScope::default()
            },
            path: plugin_dir,
        },
        &server,
        &runner,
    )
    .unwrap();

    let mut scopes = load_scoped_registries(None).unwrap();
    let scope = scopes
        .iter_mut()
        .find(|scope| scope.registry.get("acme.python-retry").is_some())
        .unwrap();
    let expected_environment = scope
        .registry
        .get("acme.python-retry")
        .unwrap()
        .source
        .environment_ref
        .clone()
        .unwrap();
    let outside = temp.path().join("outside");
    std::fs::create_dir_all(&outside).unwrap();
    scope
        .registry
        .update_environment(
            "acme.python-retry",
            Some(outside.display().to_string()),
            DynamicPluginCheckState::Valid,
        )
        .unwrap();
    scope.save().unwrap();

    let error = remove(
        PluginsRemoveRequest {
            id: "acme.python-retry".into(),
        },
        &server,
    )
    .expect_err("guarded cleanup should preserve unmanaged paths");
    assert!(error.to_string().contains("refusing to delete"));
    assert!(outside.exists());
    let mut scopes = load_scoped_registries(None).unwrap();
    let scope = scopes
        .iter_mut()
        .find(|scope| scope.registry.get("acme.python-retry").is_some())
        .unwrap();
    assert!(
        scope
            .registry
            .get("acme.python-retry")
            .unwrap()
            .is_tombstoned()
    );
    scope
        .registry
        .update_environment(
            "acme.python-retry",
            Some(expected_environment.clone()),
            DynamicPluginCheckState::Valid,
        )
        .unwrap();
    scope.save().unwrap();

    remove(
        PluginsRemoveRequest {
            id: "acme.python-retry".into(),
        },
        &server,
    )
    .unwrap();
    assert!(!Path::new(&expected_environment).exists());
    assert!(outside.exists());
    let scopes = load_scoped_registries(None).unwrap();
    let record = find_record_by_id(&scopes, "acme.python-retry")
        .unwrap()
        .unwrap()
        .record;
    assert!(record.is_tombstoned());
    assert_eq!(record.source.environment_ref, None);
}

#[test]
fn active_dynamic_plugin_components_project_enabled_native_records_only() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("native");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    write_native_dynamic_manifest(&plugin_dir, "acme.native");
    let server = crate::server::GatewayOverrides::default();

    add(
        PluginsAddRequest {
            scope: ConfigurationScope {
                project: true,
                ..ConfigurationScope::default()
            },
            path: plugin_dir,
        },
        &server,
    )
    .unwrap();

    let resolved = resolve_plugins_config(None).unwrap();
    let inactive = active_dynamic_plugin_components(None, &resolved).unwrap();
    assert!(inactive.is_empty());

    enable(
        PluginsEnableRequest {
            id: "acme.native".into(),
        },
        &server,
    )
    .unwrap();
    let resolved = resolve_plugins_config(None).unwrap();
    let active = active_dynamic_plugin_components(None, &resolved).unwrap();
    assert_eq!(active.len(), 1);
    assert_eq!(active[0].plugin_id, "acme.native");
    assert_eq!(active[0].kind, DynamicPluginKind::RustDynamic);
    assert!(
        active[0]
            .manifest_ref
            .as_deref()
            .is_some_and(|manifest_ref| manifest_ref.contains("relay-plugin.toml"))
    );
    assert!(active[0].config.is_empty());
}

#[test]
fn active_dynamic_plugin_components_accept_enabled_worker_records() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("worker");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    write_dynamic_manifest(&plugin_dir, "acme.worker");
    let server = crate::server::GatewayOverrides::default();

    add(
        PluginsAddRequest {
            scope: ConfigurationScope {
                project: true,
                ..ConfigurationScope::default()
            },
            path: plugin_dir,
        },
        &server,
    )
    .unwrap();
    enable(
        PluginsEnableRequest {
            id: "acme.worker".into(),
        },
        &server,
    )
    .unwrap();

    let resolved = resolve_plugins_config(None).unwrap();
    let active = active_dynamic_plugin_components(None, &resolved).unwrap();
    assert_eq!(active.len(), 1);
    assert_eq!(active[0].plugin_id, "acme.worker");
    assert_eq!(active[0].kind, DynamicPluginKind::Worker);
    assert!(
        active[0]
            .manifest_ref
            .as_deref()
            .is_some_and(|manifest_ref| manifest_ref.contains("relay-plugin.toml"))
    );
    assert!(active[0].config.is_empty());
}

#[test]
fn active_dynamic_plugin_components_accept_worker_records_without_manifest_ref() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("worker");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    write_dynamic_manifest(&plugin_dir, "acme.worker");
    let server = crate::server::GatewayOverrides::default();

    add(
        PluginsAddRequest {
            scope: ConfigurationScope {
                project: true,
                ..ConfigurationScope::default()
            },
            path: plugin_dir,
        },
        &server,
    )
    .unwrap();
    enable(
        PluginsEnableRequest {
            id: "acme.worker".into(),
        },
        &server,
    )
    .unwrap();

    let mut scopes = load_scoped_registries(server.config.as_ref()).unwrap();
    let scope = scopes
        .iter_mut()
        .find(|scope| scope.registry.get("acme.worker").is_some())
        .expect("worker record should exist");
    let mut records = scope.registry.cloned_records(true);
    records
        .iter_mut()
        .find(|record| record.metadata.id == "acme.worker")
        .expect("worker record should exist")
        .source
        .manifest_ref = None;
    scope.registry = nemo_relay::plugin::dynamic::DynamicPluginRegistry::from_records(records)
        .expect("registry should accept worker without manifest_ref");
    scope.save().unwrap();

    let resolved = resolve_plugins_config(None).unwrap();
    let active = active_dynamic_plugin_components(None, &resolved).unwrap();
    assert_eq!(active.len(), 1);
    assert_eq!(active[0].plugin_id, "acme.worker");
    assert_eq!(active[0].kind, DynamicPluginKind::Worker);
    assert_eq!(active[0].manifest_ref, None);
}

#[test]
fn add_rejects_duplicate_dynamic_plugin_ids() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    write_dynamic_manifest(&plugin_dir, "acme.guardrail");

    add(
        PluginsAddRequest {
            scope: ConfigurationScope {
                project: true,
                ..ConfigurationScope::default()
            },
            path: plugin_dir.clone(),
        },
        &crate::server::GatewayOverrides::default(),
    )
    .unwrap();

    let error = add(
        PluginsAddRequest {
            scope: ConfigurationScope {
                project: true,
                ..ConfigurationScope::default()
            },
            path: plugin_dir,
        },
        &crate::server::GatewayOverrides::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("already registered"));
}

#[test]
fn add_rejects_scope_flags_when_explicit_config_is_set() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let plugin_dir = temp.path().join("plugins").join("acme");
    let config_dir = temp.path().join("custom-config");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    std::fs::create_dir_all(&config_dir).unwrap();
    write_dynamic_manifest(&plugin_dir, "acme.explicit-conflict");
    let config_path = config_dir.join("gateway.toml");
    std::fs::write(&config_path, "").unwrap();

    let server = GatewayOverrides {
        config: Some(config_path),
        ..GatewayOverrides::default()
    };

    let error = add(
        PluginsAddRequest {
            scope: ConfigurationScope {
                project: true,
                ..ConfigurationScope::default()
            },
            path: plugin_dir,
        },
        &server,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("--config cannot be combined"));
}

#[test]
fn add_refuses_dynamic_plugins_blocked_by_host_policy() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    let config_dir = temp.path().join(".nemo-relay");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    std::fs::create_dir_all(&config_dir).unwrap();
    write_dynamic_manifest(&plugin_dir, "acme.blocked");
    std::fs::write(
        config_dir.join("plugins.toml"),
        r#"
[plugins.policy.defaults]
allowed = false
"#,
    )
    .unwrap();

    let error = add(
        PluginsAddRequest {
            scope: ConfigurationScope {
                project: true,
                ..ConfigurationScope::default()
            },
            path: plugin_dir,
        },
        &crate::server::GatewayOverrides::default(),
    )
    .unwrap_err();

    match error {
        CliError::PluginLifecycle {
            kind: PluginLifecycleFailureKind::Refused,
            message,
            ..
        } => assert!(message.contains("blocked by host policy")),
        other => panic!("unexpected policy add error: {other}"),
    }

    let rendered = std::fs::read_to_string(config_dir.join("plugins.toml")).unwrap();
    assert!(!rendered.contains("[[plugins.dynamic]]"));
}

#[test]
fn validate_path_reports_integrity_mismatch() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    let manifest_path = write_dynamic_manifest(&plugin_dir, "acme.integrity");
    std::fs::write(
        plugin_dir.join("plugin.py"),
        "def register():\n    return 'tampered'\n",
    )
    .unwrap();
    let (manifest, manifest_ref) = DynamicPluginManifest::load_from_path(&manifest_path)
        .map_err(|error| CliError::Config(error.to_string()))
        .unwrap();
    let policy = evaluate_dynamic_plugin_host_policy(
        &ResolvedConfig::default().dynamic_plugin_policy,
        &manifest,
    );
    let trust = evaluate_dynamic_plugin_trust(&manifest, &manifest_ref, &policy);
    let summary = PluginValidationSummaryView {
        manifest: &manifest,
        manifest_ref: &manifest_ref,
        entry: None,
        host_config: None,
        policy: &policy,
        trust: &trust,
    }
    .to_string();

    assert_eq!(trust.integrity, DynamicPluginCheckState::Invalid);
    assert!(summary.contains("trust verification blocks it"));
    assert!(summary.contains("integrity_state: invalid"));
    assert!(
        summary
            .contains("trust_error: dynamic plugin 'acme.integrity' failed integrity verification")
    );
}

#[test]
fn list_and_inspect_render_discovered_dynamic_plugins() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    write_dynamic_manifest(&plugin_dir, "acme.guardrail");

    add(
        PluginsAddRequest {
            scope: ConfigurationScope {
                project: true,
                ..ConfigurationScope::default()
            },
            path: plugin_dir,
        },
        &crate::server::GatewayOverrides::default(),
    )
    .unwrap();

    let resolved = resolve_plugins_config(None).unwrap();
    let host_config_by_id = host_config_by_id(&resolved);
    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let records = collect_records(&scopes, false);
    let list = PluginListView {
        records: &records,
        host_config_by_id: &host_config_by_id,
    }
    .to_string();
    assert!(list.contains("POLICY"));
    assert!(list.contains("acme.guardrail"));
    assert!(list.contains("absent"));
    assert!(list.contains("false"));
    assert!(
        list.lines()
            .any(|line| line.contains("acme.guardrail") && line.contains(" valid "))
    );

    let entry = find_record_by_id(&scopes, "acme.guardrail")
        .unwrap()
        .expect("plugin record");
    let (manifest, manifest_ref) =
        DynamicPluginManifest::load_from_path(entry.record.source.manifest_ref.clone().unwrap())
            .map_err(|error| CliError::Config(error.to_string()))
            .unwrap();
    let inspect = PluginInspectView {
        entry: &entry,
        manifest: &manifest,
        manifest_ref: &manifest_ref,
        host_config: host_config_by_id.get("acme.guardrail"),
    }
    .to_string();
    let inspect_value: serde_yaml::Value = serde_yaml::from_str(&inspect).unwrap();
    assert_eq!(
        inspect_value["metadata"]["id"].as_str(),
        Some("acme.guardrail")
    );
    assert_eq!(inspect_value["metadata"]["kind"].as_str(), Some("worker"));
    assert_eq!(inspect_value["host_config_status"].as_str(), Some("absent"));
    assert!(
        inspect_value["source"]["manifest_ref"]
            .as_str()
            .unwrap()
            .contains("relay-plugin.toml")
    );
    assert_eq!(
        inspect_value["load"]["entrypoint"].as_str(),
        Some("plugin.py")
    );
}

#[test]
fn validate_renders_summary_for_path_and_id_targets() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    let manifest_path = write_dynamic_manifest(&plugin_dir, "acme.guardrail");

    add(
        PluginsAddRequest {
            scope: ConfigurationScope {
                project: true,
                ..ConfigurationScope::default()
            },
            path: plugin_dir,
        },
        &crate::server::GatewayOverrides::default(),
    )
    .unwrap();

    let (manifest, manifest_ref) = DynamicPluginManifest::load_from_path(&manifest_path)
        .map_err(|error| CliError::Config(error.to_string()))
        .unwrap();
    let default_policy = evaluate_dynamic_plugin_host_policy(
        &ResolvedConfig::default().dynamic_plugin_policy,
        &manifest,
    );
    let default_trust = evaluate_dynamic_plugin_trust(&manifest, &manifest_ref, &default_policy);
    let path_summary = PluginValidationSummaryView {
        manifest: &manifest,
        manifest_ref: &manifest_ref,
        entry: None,
        host_config: None,
        policy: &default_policy,
        trust: &default_trust,
    }
    .to_string();
    assert!(path_summary.contains("Dynamic plugin 'acme.guardrail' is valid."));
    assert!(path_summary.contains("policy_state: valid"));

    let resolved = resolve_plugins_config(None).unwrap();
    let host_config_by_id = host_config_by_id(&resolved);
    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let entry = find_record_by_id(&scopes, "acme.guardrail")
        .unwrap()
        .expect("plugin record");
    let policy = evaluate_dynamic_plugin_host_policy(&resolved.dynamic_plugin_policy, &manifest);
    let trust = evaluate_dynamic_plugin_trust(&manifest, &manifest_ref, &policy);
    let id_summary = PluginValidationSummaryView {
        manifest: &manifest,
        manifest_ref: &manifest_ref,
        entry: Some(&entry),
        host_config: host_config_by_id.get("acme.guardrail"),
        policy: &policy,
        trust: &trust,
    }
    .to_string();
    assert!(id_summary.contains("host_config: absent"));
    assert!(id_summary.contains("desired.enabled: false"));

    let missing_validate = validate(
        PluginsValidateRequest {
            target: "missing.plugin".into(),
            json: false,
        },
        &crate::server::GatewayOverrides::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(missing_validate.contains("not registered"));

    let missing_inspect = inspect(
        PluginsInspectRequest {
            id: "missing.plugin".into(),
            json: false,
        },
        &crate::server::GatewayOverrides::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(missing_inspect.contains("not registered"));

    assert_eq!(
        list(
            PluginsListRequest::default(),
            &crate::server::GatewayOverrides::default()
        )
        .unwrap(),
        ()
    );
}

#[test]
fn enable_disable_and_remove_persist_lifecycle_state() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    write_dynamic_manifest(&plugin_dir, "acme.guardrail");
    let server = crate::server::GatewayOverrides::default();

    add(
        PluginsAddRequest {
            scope: ConfigurationScope {
                project: true,
                ..ConfigurationScope::default()
            },
            path: plugin_dir,
        },
        &server,
    )
    .unwrap();

    enable(
        PluginsEnableRequest {
            id: "acme.guardrail".into(),
        },
        &server,
    )
    .unwrap();

    let resolved = resolve_plugins_config(None).unwrap();
    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let enabled = find_record_by_id(&scopes, "acme.guardrail")
        .unwrap()
        .expect("enabled record");
    assert!(enabled.record.spec.enabled);

    disable(
        PluginsDisableRequest {
            id: "acme.guardrail".into(),
        },
        &server,
    )
    .unwrap();
    let resolved = resolve_plugins_config(None).unwrap();
    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let disabled = find_record_by_id(&scopes, "acme.guardrail")
        .unwrap()
        .expect("disabled record");
    assert!(!disabled.record.spec.enabled);

    remove(
        PluginsRemoveRequest {
            id: "acme.guardrail".into(),
        },
        &server,
    )
    .unwrap();
    let resolved = resolve_plugins_config(None).unwrap();
    assert!(resolved.dynamic_plugins.is_empty());
    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let removed = find_record_by_id(&scopes, "acme.guardrail")
        .unwrap()
        .expect("removed record");
    assert!(removed.record.is_tombstoned());

    let all_records = collect_records(&scopes, true);
    let host_config_by_id = host_config_by_id(&resolved);
    let all_list = PluginListView {
        records: &all_records,
        host_config_by_id: &host_config_by_id,
    }
    .to_string();
    assert!(all_list.contains("acme.guardrail"));
    assert!(all_list.contains("tombstoned"));

    let error = enable(
        PluginsEnableRequest {
            id: "acme.guardrail".into(),
        },
        &server,
    )
    .expect_err("tombstoned plugin should not enable");
    match error {
        CliError::PluginLifecycle {
            kind: PluginLifecycleFailureKind::Refused,
            message,
            ..
        } => assert!(message.contains("tombstoned")),
        other => panic!("unexpected tombstone enable error: {other}"),
    }
}

#[test]
fn add_with_explicit_config_uses_sibling_plugins_and_state_files() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let plugin_dir = temp.path().join("plugins").join("acme");
    let config_dir = temp.path().join("custom-config");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    std::fs::create_dir_all(&config_dir).unwrap();
    write_dynamic_manifest(&plugin_dir, "acme.explicit");
    let config_path = config_dir.join("gateway.toml");
    std::fs::write(&config_path, "").unwrap();

    let server = GatewayOverrides {
        config: Some(config_path),
        ..GatewayOverrides::default()
    };

    add(
        PluginsAddRequest {
            scope: ConfigurationScope::default(),
            path: plugin_dir,
        },
        &server,
    )
    .unwrap();

    let plugins_toml = config_dir.join("plugins.toml");
    let state_path = config_dir.join(".dynamic-plugins.json");
    assert!(plugins_toml.exists());
    assert!(state_path.exists());

    let resolved = resolve_plugins_config(server.config.as_ref()).unwrap();
    assert_eq!(resolved.dynamic_plugins.len(), 1);
    assert_eq!(resolved.dynamic_plugins[0].plugin_id, "acme.explicit");

    let scopes = load_and_hydrate_scopes(server.config.as_ref(), &resolved).unwrap();
    let entry = find_record_by_id(&scopes, "acme.explicit")
        .unwrap()
        .expect("explicit-scope record");
    assert_eq!(entry.scope.to_string(), "explicit");
    assert_eq!(entry.plugins_toml_path, plugins_toml);
    assert_eq!(entry.state_path, state_path);
}

#[test]
fn hydrate_bootstraps_registry_records_from_existing_dynamic_plugin_refs() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    let config_dir = temp.path().join(".nemo-relay");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    std::fs::create_dir_all(&config_dir).unwrap();
    let manifest_path = write_dynamic_manifest(&plugin_dir, "acme.bootstrap");

    std::fs::write(
        config_dir.join("plugins.toml"),
        format!(
            "[[plugins.dynamic]]\nmanifest = {:?}\n",
            manifest_path.to_string_lossy()
        ),
    )
    .unwrap();

    let resolved = resolve_plugins_config(None).unwrap();
    assert_eq!(resolved.dynamic_plugins.len(), 1);

    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let entry = find_record_by_id(&scopes, "acme.bootstrap")
        .unwrap()
        .expect("hydrated record");
    assert_eq!(entry.scope.to_string(), "project");
    assert_eq!(entry.record.metadata.id, "acme.bootstrap");
    assert!(entry.record.spec.present);
    assert!(!entry.record.spec.enabled);
    let canonical_manifest_path = std::fs::canonicalize(&manifest_path).unwrap();
    assert_eq!(
        entry.record.source.manifest_ref.as_deref(),
        Some(canonical_manifest_path.to_string_lossy().as_ref())
    );
}

#[test]
fn manually_configured_python_worker_cannot_enable_without_lifecycle_add() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("python");
    let config_dir = temp.path().join(".nemo-relay");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    std::fs::create_dir_all(&config_dir).unwrap();
    let manifest_path = write_python_dynamic_manifest(&plugin_dir, "acme.python-direct");
    std::fs::write(
        config_dir.join("plugins.toml"),
        format!(
            "[[plugins.dynamic]]\nmanifest = {:?}\n",
            manifest_path.to_string_lossy()
        ),
    )
    .unwrap();

    let resolved = resolve_plugins_config(None).unwrap();
    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let entry = find_record_by_id(&scopes, "acme.python-direct")
        .unwrap()
        .unwrap();
    assert_eq!(entry.record.source.environment_ref, None);
    assert_eq!(
        entry.record.status.validation.environment,
        DynamicPluginCheckState::Invalid
    );
    let (manifest, manifest_ref) = DynamicPluginManifest::load_from_path(&manifest_path).unwrap();
    let policy = evaluate_dynamic_plugin_host_policy(&resolved.dynamic_plugin_policy, &manifest);
    let trust = evaluate_dynamic_plugin_trust(&manifest, &manifest_ref, &policy);
    let summary = PluginValidationSummaryView {
        manifest: &manifest,
        manifest_ref: &manifest_ref,
        entry: Some(&entry),
        host_config: None,
        policy: &policy,
        trust: &trust,
    }
    .to_string();
    assert!(summary.contains("runtime environment is unavailable"));

    let error = enable(
        PluginsEnableRequest {
            id: "acme.python-direct".into(),
        },
        &GatewayOverrides::default(),
    )
    .expect_err("manually configured Python workers must not activate");
    let (_, _, kind, code, message) = error
        .as_plugin_lifecycle_error_context()
        .expect("direct Python activation error should be structured");
    assert_eq!(kind, PluginLifecycleFailureKind::Refused);
    assert_eq!(code, Some("environment_failed"));
    assert!(message.contains("plugins remove acme.python-direct"));
    assert!(message.contains("plugins add"));
}

#[test]
fn hydrate_applies_host_policy_status_to_discovered_dynamic_plugins() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    let config_dir = temp.path().join(".nemo-relay");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    std::fs::create_dir_all(&config_dir).unwrap();
    let manifest_path = write_dynamic_manifest(&plugin_dir, "acme.policy");

    std::fs::write(
        config_dir.join("plugins.toml"),
        format!(
            concat!(
                "[[plugins.dynamic]]\n",
                "manifest = {:?}\n\n",
                "[plugins.policy.defaults]\n",
                "startup = \"required\"\n",
                "attestation = \"signature_required\"\n"
            ),
            manifest_path.to_string_lossy()
        ),
    )
    .unwrap();

    let resolved = resolve_plugins_config(None).unwrap();
    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let entry = find_record_by_id(&scopes, "acme.policy")
        .unwrap()
        .expect("hydrated record");

    assert_eq!(
        entry.record.status.validation.policy_satisfied,
        DynamicPluginCheckState::Valid
    );
    assert_eq!(
        entry
            .record
            .status
            .startup_class
            .map(|value| value.to_string()),
        Some("required".into())
    );
    assert_eq!(
        entry
            .record
            .status
            .attestation_mode
            .map(|value| value.to_string()),
        Some("signature_required".into())
    );
    assert_eq!(
        entry.record.status.validation.authenticity,
        DynamicPluginCheckState::Invalid
    );
    assert!(
        entry
            .record
            .status
            .last_error
            .as_ref()
            .unwrap()
            .message
            .contains("signature verification")
            || entry
                .record
                .status
                .last_error
                .as_ref()
                .unwrap()
                .message
                .contains("integrity.signature")
    );
}

#[test]
fn hydrate_persists_updated_policy_and_error_state() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    let config_dir = temp.path().join(".nemo-relay");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    std::fs::create_dir_all(&config_dir).unwrap();
    write_dynamic_manifest(&plugin_dir, "acme.persist-blocked");

    add(
        PluginsAddRequest {
            scope: ConfigurationScope {
                project: true,
                ..ConfigurationScope::default()
            },
            path: plugin_dir.clone(),
        },
        &GatewayOverrides::default(),
    )
    .unwrap();

    std::fs::write(
        config_dir.join("plugins.toml"),
        format!(
            concat!(
                "[[plugins.dynamic]]\n",
                "manifest = {:?}\n\n",
                "[plugins.policy.defaults]\n",
                "allowed = false\n"
            ),
            plugin_dir.join("relay-plugin.toml").to_string_lossy()
        ),
    )
    .unwrap();

    let resolved = resolve_plugins_config(None).unwrap();
    let _ = load_and_hydrate_scopes(None, &resolved).unwrap();

    let state_path = config_dir.join(".dynamic-plugins.json");
    let state: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&state_path).unwrap()).unwrap();
    let record = &state["records"][0];
    assert_eq!(
        record["metadata"]["id"],
        serde_json::json!("acme.persist-blocked")
    );
    assert_eq!(
        record["status"]["validation"]["policy_satisfied"],
        serde_json::json!("invalid")
    );
    assert_eq!(
        record["status"]["last_error"]["phase"],
        serde_json::json!("policy")
    );
}

#[test]
fn hydrate_verifies_signatures_when_host_policy_provides_trusted_keys() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    let config_dir = temp.path().join(".nemo-relay");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    std::fs::create_dir_all(&config_dir).unwrap();
    let manifest_path = write_dynamic_manifest_with_options(
        &plugin_dir,
        "acme.signed",
        &["plugin_worker"],
        Some("plugin.py.sig"),
    );
    let trusted_public_key = write_detached_ed25519_signature(&plugin_dir, "plugin.py.sig");

    std::fs::write(
        config_dir.join("plugins.toml"),
        format!(
            concat!(
                "[[plugins.dynamic]]\n",
                "manifest = {:?}\n\n",
                "[plugins.policy.defaults]\n",
                "startup = \"required\"\n",
                "attestation = \"signature_required\"\n",
                "trusted_public_keys = [{:?}]\n"
            ),
            manifest_path.to_string_lossy(),
            trusted_public_key
        ),
    )
    .unwrap();

    let resolved = resolve_plugins_config(None).unwrap();
    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let entry = find_record_by_id(&scopes, "acme.signed")
        .unwrap()
        .expect("hydrated signed record");

    assert_eq!(
        entry.record.status.validation.integrity,
        DynamicPluginCheckState::Valid
    );
    assert_eq!(
        entry.record.status.validation.authenticity,
        DynamicPluginCheckState::Valid
    );
    assert!(entry.record.status.last_error.is_none());
}

#[test]
fn hydrate_marks_signature_required_plugins_invalid_without_trusted_keys() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    let config_dir = temp.path().join(".nemo-relay");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    std::fs::create_dir_all(&config_dir).unwrap();
    let manifest_path = write_dynamic_manifest_with_options(
        &plugin_dir,
        "acme.signed-without-trust",
        &["plugin_worker"],
        Some("plugin.py.sig"),
    );
    write_detached_ed25519_signature(&plugin_dir, "plugin.py.sig");

    std::fs::write(
        config_dir.join("plugins.toml"),
        format!(
            concat!(
                "[[plugins.dynamic]]\n",
                "manifest = {:?}\n\n",
                "[plugins.policy.defaults]\n",
                "attestation = \"signature_required\"\n"
            ),
            manifest_path.to_string_lossy()
        ),
    )
    .unwrap();

    let resolved = resolve_plugins_config(None).unwrap();
    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let entry = find_record_by_id(&scopes, "acme.signed-without-trust")
        .unwrap()
        .expect("hydrated signed record");

    assert_eq!(
        entry.record.status.validation.authenticity,
        DynamicPluginCheckState::Invalid
    );
    assert!(
        entry
            .record
            .status
            .last_error
            .as_ref()
            .unwrap()
            .message
            .contains("no trusted_public_keys")
    );
}

#[test]
fn hydrate_marks_signature_required_plugins_invalid_with_wrong_trusted_key() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    let config_dir = temp.path().join(".nemo-relay");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    std::fs::create_dir_all(&config_dir).unwrap();
    let manifest_path = write_dynamic_manifest_with_options(
        &plugin_dir,
        "acme.signed-wrong-key",
        &["plugin_worker"],
        Some("plugin.py.sig"),
    );
    write_detached_ed25519_signature(&plugin_dir, "plugin.py.sig");
    let wrong_public_key = generate_ed25519_public_key();

    std::fs::write(
        config_dir.join("plugins.toml"),
        format!(
            concat!(
                "[[plugins.dynamic]]\n",
                "manifest = {:?}\n\n",
                "[plugins.policy.defaults]\n",
                "attestation = \"signature_required\"\n",
                "trusted_public_keys = [{:?}]\n"
            ),
            manifest_path.to_string_lossy(),
            wrong_public_key
        ),
    )
    .unwrap();

    let resolved = resolve_plugins_config(None).unwrap();
    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let entry = find_record_by_id(&scopes, "acme.signed-wrong-key")
        .unwrap()
        .expect("hydrated signed record");

    assert_eq!(
        entry.record.status.validation.authenticity,
        DynamicPluginCheckState::Invalid
    );
    assert!(
        entry
            .record
            .status
            .last_error
            .as_ref()
            .unwrap()
            .message
            .contains("failed signature verification")
    );
}

#[test]
fn hydrate_marks_malformed_signature_files_invalid_when_signature_is_present() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    let config_dir = temp.path().join(".nemo-relay");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    std::fs::create_dir_all(&config_dir).unwrap();
    let manifest_path = write_dynamic_manifest_with_options(
        &plugin_dir,
        "acme.signed-malformed",
        &["plugin_worker"],
        Some("plugin.py.sig"),
    );
    std::fs::write(plugin_dir.join("plugin.py.sig"), "ed25519:not-base64\n").unwrap();
    let trusted_public_key = generate_ed25519_public_key();

    std::fs::write(
        config_dir.join("plugins.toml"),
        format!(
            concat!(
                "[[plugins.dynamic]]\n",
                "manifest = {:?}\n\n",
                "[plugins.policy.defaults]\n",
                "attestation = \"signature_if_present\"\n",
                "trusted_public_keys = [{:?}]\n"
            ),
            manifest_path.to_string_lossy(),
            trusted_public_key
        ),
    )
    .unwrap();

    let resolved = resolve_plugins_config(None).unwrap();
    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let entry = find_record_by_id(&scopes, "acme.signed-malformed")
        .unwrap()
        .expect("hydrated signed record");

    assert_eq!(
        entry.record.status.validation.authenticity,
        DynamicPluginCheckState::Invalid
    );
    assert!(
        entry
            .record
            .status
            .last_error
            .as_ref()
            .unwrap()
            .message
            .contains("invalid base64 signature")
    );
}

#[test]
fn enable_refuses_dynamic_plugins_blocked_by_host_policy_and_persists_status() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    let config_dir = temp.path().join(".nemo-relay");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    std::fs::create_dir_all(&config_dir).unwrap();
    let manifest_path = write_dynamic_manifest(&plugin_dir, "acme.enable-blocked");
    let server = crate::server::GatewayOverrides::default();

    add(
        PluginsAddRequest {
            scope: ConfigurationScope {
                project: true,
                ..ConfigurationScope::default()
            },
            path: plugin_dir,
        },
        &server,
    )
    .unwrap();

    std::fs::write(
        config_dir.join("plugins.toml"),
        format!(
            concat!(
                "[[plugins.dynamic]]\n",
                "manifest = {:?}\n\n",
                "[plugins.policy.defaults]\n",
                "allowed = false\n"
            ),
            manifest_path.to_string_lossy()
        ),
    )
    .unwrap();

    let error = enable(
        PluginsEnableRequest {
            id: "acme.enable-blocked".into(),
        },
        &server,
    )
    .unwrap_err();

    match error {
        CliError::PluginLifecycle {
            kind: PluginLifecycleFailureKind::Refused,
            ref message,
            ..
        } => assert!(message.contains("blocked by host policy")),
        other => panic!("unexpected enable policy error: {other}"),
    }
    let (command, target, kind, code, _) = error
        .as_plugin_lifecycle_error_context()
        .expect("plugin lifecycle error context");
    assert_eq!(command, "plugins enable");
    assert_eq!(target, Some("acme.enable-blocked"));
    assert_eq!(kind, PluginLifecycleFailureKind::Refused);
    assert_eq!(code, Some("policy_blocked"));

    let resolved = resolve_plugins_config(None).unwrap();
    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let entry = find_record_by_id(&scopes, "acme.enable-blocked")
        .unwrap()
        .expect("policy-updated record");
    assert!(!entry.record.spec.enabled);
    assert_eq!(
        entry.record.status.validation.policy_satisfied,
        DynamicPluginCheckState::Invalid
    );
    assert_eq!(
        entry
            .record
            .status
            .last_error
            .as_ref()
            .map(|error| error.phase.to_string()),
        Some("policy".into())
    );
}

#[test]
fn disable_succeeds_when_registered_plugin_manifest_is_unreadable() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    write_dynamic_manifest(&plugin_dir, "acme.guardrail");
    let server = crate::server::GatewayOverrides::default();

    add(
        PluginsAddRequest {
            scope: ConfigurationScope {
                project: true,
                ..ConfigurationScope::default()
            },
            path: plugin_dir.clone(),
        },
        &server,
    )
    .unwrap();

    enable(
        PluginsEnableRequest {
            id: "acme.guardrail".into(),
        },
        &server,
    )
    .unwrap();

    std::fs::remove_file(plugin_dir.join("relay-plugin.toml")).unwrap();

    disable(
        PluginsDisableRequest {
            id: "acme.guardrail".into(),
        },
        &server,
    )
    .unwrap();

    let scopes = load_scoped_registries(None).unwrap();
    let entry = find_record_by_id(&scopes, "acme.guardrail")
        .unwrap()
        .expect("disabled plugin record");
    assert!(!entry.record.spec.enabled);
}

#[test]
fn validate_marks_registered_plugins_invalid_when_host_policy_blocks_them() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    let config_dir = temp.path().join(".nemo-relay");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    std::fs::create_dir_all(&config_dir).unwrap();
    let manifest_path = write_dynamic_manifest(&plugin_dir, "acme.validate-blocked");
    let server = crate::server::GatewayOverrides::default();

    add(
        PluginsAddRequest {
            scope: ConfigurationScope {
                project: true,
                ..ConfigurationScope::default()
            },
            path: plugin_dir,
        },
        &server,
    )
    .unwrap();

    std::fs::write(
        config_dir.join("plugins.toml"),
        format!(
            concat!(
                "[[plugins.dynamic]]\n",
                "manifest = {:?}\n\n",
                "[plugins.policy.defaults]\n",
                "startup = \"required\"\n",
                "attestation = \"signature_required\"\n",
                "allowed = false\n"
            ),
            manifest_path.to_string_lossy()
        ),
    )
    .unwrap();

    validate(
        PluginsValidateRequest {
            target: "acme.validate-blocked".into(),
            json: false,
        },
        &server,
    )
    .unwrap();

    let resolved = resolve_plugins_config(None).unwrap();
    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let entry = find_record_by_id(&scopes, "acme.validate-blocked")
        .unwrap()
        .expect("policy-updated record");

    assert_eq!(
        entry.record.status.validation.policy_satisfied,
        DynamicPluginCheckState::Invalid
    );
    assert_eq!(
        entry.record.status.validation.message.as_deref(),
        Some("validated by CLI")
    );
    let (blocked_manifest, blocked_manifest_ref) =
        DynamicPluginManifest::load_from_path(&manifest_path)
            .map_err(|error| CliError::Config(error.to_string()))
            .unwrap();
    let blocked_policy =
        evaluate_dynamic_plugin_host_policy(&resolved.dynamic_plugin_policy, &blocked_manifest);
    let blocked_trust =
        evaluate_dynamic_plugin_trust(&blocked_manifest, &blocked_manifest_ref, &blocked_policy);
    let blocked_summary = PluginValidationSummaryView {
        manifest: &blocked_manifest,
        manifest_ref: &blocked_manifest_ref,
        entry: Some(&entry),
        host_config: None,
        policy: &blocked_policy,
        trust: &blocked_trust,
    }
    .to_string();
    assert!(blocked_summary.contains("host policy blocks it"));
    assert!(blocked_summary.contains("policy_state: invalid"));
    let blocked_list = PluginListView {
        records: std::slice::from_ref(&entry),
        host_config_by_id: &std::collections::HashMap::new(),
    }
    .to_string();
    assert!(blocked_list.contains("POLICY"));
    assert!(blocked_list.contains("invalid"));
    let blocked_validate_value = serde_json::to_value(responses::validate_success(
        responses::ValidateResponseInput {
            command: "plugins validate",
            target: Some("acme.validate-blocked"),
            target_kind: "plugin_id",
            resolved_plugin_id: Some("acme.validate-blocked"),
            manifest: &blocked_manifest,
            manifest_ref: &blocked_manifest_ref,
            entry: Some(&entry),
            host_config: None,
            policy: &blocked_policy,
            trust: &blocked_trust,
        },
    ))
    .unwrap();
    assert_eq!(
        blocked_validate_value["data"]["valid"],
        serde_json::json!(false)
    );
    assert_eq!(
        blocked_validate_value["data"]["policy_state"],
        serde_json::json!("invalid")
    );
    assert!(
        blocked_validate_value["data"]["errors"][0]
            .as_str()
            .unwrap()
            .contains("blocked by host policy")
    );
    assert_eq!(
        entry
            .record
            .status
            .startup_class
            .map(|value| value.to_string()),
        Some("required".into())
    );
    assert_eq!(
        entry
            .record
            .status
            .attestation_mode
            .map(|value| value.to_string()),
        Some("signature_required".into())
    );
    assert_eq!(
        entry
            .record
            .status
            .last_error
            .as_ref()
            .map(|error| error.phase.to_string()),
        Some("policy".into())
    );
}

#[test]
fn add_can_revive_tombstoned_records() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    write_dynamic_manifest(&plugin_dir, "acme.revive");
    let server = crate::server::GatewayOverrides::default();

    add(
        PluginsAddRequest {
            scope: ConfigurationScope {
                project: true,
                ..ConfigurationScope::default()
            },
            path: plugin_dir.clone(),
        },
        &server,
    )
    .unwrap();

    remove(
        PluginsRemoveRequest {
            id: "acme.revive".into(),
        },
        &server,
    )
    .unwrap();

    add(
        PluginsAddRequest {
            scope: ConfigurationScope {
                project: true,
                ..ConfigurationScope::default()
            },
            path: plugin_dir,
        },
        &server,
    )
    .unwrap();

    let resolved = resolve_plugins_config(None).unwrap();
    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let revived = find_record_by_id(&scopes, "acme.revive")
        .unwrap()
        .expect("revived record");
    assert!(!revived.record.is_tombstoned());
    assert!(revived.record.spec.present);
}

#[test]
fn json_helpers_emit_stable_success_and_failure_shapes() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    let manifest_path = write_dynamic_manifest(&plugin_dir, "acme.json");
    let server = GatewayOverrides::default();

    add(
        PluginsAddRequest {
            scope: ConfigurationScope {
                project: true,
                ..ConfigurationScope::default()
            },
            path: plugin_dir,
        },
        &server,
    )
    .unwrap();

    let resolved = resolve_plugins_config(None).unwrap();
    let host_config_by_id = host_config_by_id(&resolved);
    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let records = collect_records(&scopes, false);
    let entry = find_record_by_id(&scopes, "acme.json")
        .unwrap()
        .expect("json record");
    let (manifest, manifest_ref) = DynamicPluginManifest::load_from_path(&manifest_path)
        .map_err(|error| CliError::Config(error.to_string()))
        .unwrap();

    let list_value = serde_json::to_value(responses::list_success(
        "plugins list",
        None,
        &records,
        &host_config_by_id,
    ))
    .unwrap();
    assert_eq!(list_value["schema_version"], serde_json::json!(1));
    assert_eq!(list_value["ok"], serde_json::json!(true));
    assert_eq!(list_value["data"][0]["id"], serde_json::json!("acme.json"));

    let inspect_value = serde_json::to_value(responses::inspect_success(
        "plugins inspect",
        "acme.json",
        &entry,
        &manifest,
        &manifest_ref,
        host_config_by_id.get("acme.json"),
    ))
    .unwrap();
    assert_eq!(inspect_value["data"]["id"], serde_json::json!("acme.json"));
    assert_eq!(
        inspect_value["data"]["source"]["manifest_ref"],
        serde_json::json!(manifest_ref)
    );
    assert_eq!(
        inspect_value["data"]["environment_state"],
        serde_json::json!("unknown")
    );
    assert_eq!(
        inspect_value["data"]["host_config"],
        serde_json::Value::Null
    );

    let validate_policy =
        evaluate_dynamic_plugin_host_policy(&resolved.dynamic_plugin_policy, &manifest);
    let validate_trust = evaluate_dynamic_plugin_trust(&manifest, &manifest_ref, &validate_policy);
    let validate_value = serde_json::to_value(responses::validate_success(
        responses::ValidateResponseInput {
            command: "plugins validate",
            target: Some("acme.json"),
            target_kind: "plugin_id",
            resolved_plugin_id: Some("acme.json"),
            manifest: &manifest,
            manifest_ref: &manifest_ref,
            entry: Some(&entry),
            host_config: host_config_by_id.get("acme.json"),
            policy: &validate_policy,
            trust: &validate_trust,
        },
    ))
    .unwrap();
    assert_eq!(
        validate_value["data"]["target_kind"],
        serde_json::json!("plugin_id")
    );
    assert_eq!(validate_value["data"]["valid"], serde_json::json!(true));
    assert_eq!(
        validate_value["data"]["environment_state"],
        serde_json::json!("unknown")
    );
    assert_eq!(
        validate_value["data"]["policy_state"],
        serde_json::json!("valid")
    );
    assert_eq!(
        validate_value["data"]["startup_class"],
        serde_json::json!("optional")
    );
    assert_eq!(
        validate_value["data"]["attestation_mode"],
        serde_json::json!("integrity_only")
    );

    let failure = serde_json::to_value(responses::failure(
        "plugins inspect",
        Some("missing.plugin"),
        PluginLifecycleFailureKind::NotFound,
        None,
        "missing plugin",
    ))
    .unwrap();
    assert_eq!(failure["ok"], serde_json::json!(false));
    assert_eq!(failure["error"]["code"], serde_json::json!("not_found"));

    let refused = serde_json::to_value(responses::failure(
        "plugins add",
        Some("acme.blocked"),
        PluginLifecycleFailureKind::Refused,
        Some("policy_blocked"),
        "blocked by host policy",
    ))
    .unwrap();
    assert_eq!(
        refused["error"]["code"],
        serde_json::json!("policy_blocked")
    );
}

#[test]
fn remove_tolerates_unreadable_non_target_manifest_entries() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    let broken_dir = temp.path().join("plugins").join("broken");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    std::fs::create_dir_all(&broken_dir).unwrap();
    let manifest_path = write_dynamic_manifest(&plugin_dir, "acme.guardrail");
    let server = GatewayOverrides::default();

    add(
        PluginsAddRequest {
            scope: ConfigurationScope {
                project: true,
                ..ConfigurationScope::default()
            },
            path: plugin_dir,
        },
        &server,
    )
    .unwrap();

    let plugins_toml = temp.path().join(".nemo-relay").join("plugins.toml");
    std::fs::write(
        &plugins_toml,
        format!(
            "[[plugins.dynamic]]\nmanifest = {:?}\n\n[[plugins.dynamic]]\nmanifest = {:?}\n",
            manifest_path.to_string_lossy(),
            broken_dir.join("missing.toml").to_string_lossy()
        ),
    )
    .unwrap();

    remove(
        PluginsRemoveRequest {
            id: "acme.guardrail".into(),
        },
        &server,
    )
    .unwrap();

    let rendered = std::fs::read_to_string(&plugins_toml).unwrap();
    assert!(!rendered.contains("acme.guardrail"));
    assert!(rendered.contains("missing.toml"));
}

#[test]
fn remove_reports_malformed_dynamic_plugin_containers() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let plugins_toml = temp.path().join("plugins.toml");

    std::fs::write(&plugins_toml, "[plugins]\ndynamic = \"oops\"\n").unwrap();
    let error = remove_dynamic_plugin_reference(&plugins_toml, "acme.guardrail", None)
        .unwrap_err()
        .to_string();
    assert!(error.contains("plugins.dynamic must be an array of tables"));

    std::fs::write(&plugins_toml, "plugins = \"oops\"\n").unwrap();
    let error = remove_dynamic_plugin_reference(&plugins_toml, "acme.guardrail", None)
        .unwrap_err()
        .to_string();
    assert!(error.contains("[plugins] must be a table"));
}

#[test]
fn append_reports_malformed_dynamic_plugin_containers() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let plugins_toml = temp.path().join("plugins.toml");

    std::fs::write(&plugins_toml, "plugins = \"oops\"\n").unwrap();
    let error = append_dynamic_plugin_reference(&plugins_toml, "/tmp/plugin/relay-plugin.toml")
        .unwrap_err()
        .to_string();
    assert!(error.contains("[plugins] must be a table"));

    std::fs::write(&plugins_toml, "[plugins]\ndynamic = \"oops\"\n").unwrap();
    let error = append_dynamic_plugin_reference(&plugins_toml, "/tmp/plugin/relay-plugin.toml")
        .unwrap_err()
        .to_string();
    assert!(error.contains("plugins.dynamic must be an array of tables"));
}

#[test]
fn remove_matches_relative_target_manifest_refs_without_loading_manifest() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let config_dir = temp.path().join(".nemo-relay");
    let plugin_dir = temp.path().join("plugins").join("acme");
    std::fs::create_dir_all(&config_dir).unwrap();
    std::fs::create_dir_all(&plugin_dir).unwrap();

    let manifest_path = plugin_dir.join("relay-plugin.toml");
    std::fs::write(
        &manifest_path,
        r#"
manifest_version = 1

[plugin]
id = "acme.guardrail"
kind = "worker"
"#,
    )
    .unwrap();

    let plugins_toml = config_dir.join("plugins.toml");
    std::fs::write(
        &plugins_toml,
        "[[plugins.dynamic]]\nmanifest = \"../plugins/acme/relay-plugin.toml\"\n",
    )
    .unwrap();

    std::fs::remove_file(&manifest_path).unwrap();

    let removed = remove_dynamic_plugin_reference(
        &plugins_toml,
        "acme.guardrail",
        Some("../plugins/acme/relay-plugin.toml"),
    )
    .unwrap();
    assert!(removed);
    let rendered = std::fs::read_to_string(&plugins_toml).unwrap();
    assert!(!rendered.contains("relay-plugin.toml"));
}

#[test]
fn inspect_redacts_host_config_values() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    let manifest_path = write_dynamic_manifest(&plugin_dir, "acme.redacted");
    let server = GatewayOverrides::default();

    add(
        PluginsAddRequest {
            scope: ConfigurationScope {
                project: true,
                ..ConfigurationScope::default()
            },
            path: plugin_dir,
        },
        &server,
    )
    .unwrap();

    let plugins_toml = temp.path().join(".nemo-relay").join("plugins.toml");
    std::fs::write(
        &plugins_toml,
        format!(
            "[[plugins.dynamic]]\nmanifest = {:?}\nconfig = {{ api_key = \"secret-token\", region = \"us-west-2\" }}\n",
            manifest_path.to_string_lossy()
        ),
    )
    .unwrap();

    let resolved = resolve_plugins_config(None).unwrap();
    let host_config_by_id = host_config_by_id(&resolved);
    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let entry = find_record_by_id(&scopes, "acme.redacted")
        .unwrap()
        .expect("redacted record");
    let (manifest, manifest_ref) = DynamicPluginManifest::load_from_path(&manifest_path)
        .map_err(|error| CliError::Config(error.to_string()))
        .unwrap();

    let inspect_output = PluginInspectView {
        entry: &entry,
        manifest: &manifest,
        manifest_ref: &manifest_ref,
        host_config: host_config_by_id.get("acme.redacted"),
    }
    .to_string();
    assert!(!inspect_output.contains("secret-token"));
    let inspect_output: serde_yaml::Value = serde_yaml::from_str(&inspect_output).unwrap();
    assert_eq!(
        inspect_output["host_config"]["api_key"].as_str(),
        Some("<redacted>")
    );
    assert_eq!(
        inspect_output["host_config"]["region"].as_str(),
        Some("<redacted>")
    );

    let inspect_value = serde_json::to_value(responses::inspect_success(
        "plugins inspect",
        "acme.redacted",
        &entry,
        &manifest,
        &manifest_ref,
        host_config_by_id.get("acme.redacted"),
    ))
    .unwrap();
    assert_eq!(
        inspect_value["data"]["host_config"]["api_key"],
        serde_json::json!("<redacted>")
    );
    assert_eq!(
        inspect_value["data"]["host_config"]["region"],
        serde_json::json!("<redacted>")
    );
    assert_eq!(inspect_value["data"]["host_config_status"], "present");
}

#[test]
fn inspect_distinguishes_empty_host_config_from_missing_host_config() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let _cwd = CurrentDirGuard::enter(temp.path());
    let plugin_dir = temp.path().join("plugins").join("acme");
    std::fs::create_dir_all(&plugin_dir).unwrap();
    let manifest_path = write_dynamic_manifest(&plugin_dir, "acme.empty-config");
    let server = GatewayOverrides::default();

    add(
        PluginsAddRequest {
            scope: ConfigurationScope {
                project: true,
                ..ConfigurationScope::default()
            },
            path: plugin_dir,
        },
        &server,
    )
    .unwrap();

    let plugins_toml = temp.path().join(".nemo-relay").join("plugins.toml");
    std::fs::write(
        &plugins_toml,
        format!(
            "[[plugins.dynamic]]\nmanifest = {:?}\nconfig = {{}}\n",
            manifest_path.to_string_lossy()
        ),
    )
    .unwrap();

    let resolved = resolve_plugins_config(None).unwrap();
    let host_config_by_id = host_config_by_id(&resolved);
    let scopes = load_and_hydrate_scopes(None, &resolved).unwrap();
    let entry = find_record_by_id(&scopes, "acme.empty-config")
        .unwrap()
        .expect("empty-config record");
    let (manifest, manifest_ref) = DynamicPluginManifest::load_from_path(&manifest_path)
        .map_err(|error| CliError::Config(error.to_string()))
        .unwrap();

    let inspect_output = PluginInspectView {
        entry: &entry,
        manifest: &manifest,
        manifest_ref: &manifest_ref,
        host_config: host_config_by_id.get("acme.empty-config"),
    }
    .to_string();
    let inspect_output: serde_yaml::Value = serde_yaml::from_str(&inspect_output).unwrap();
    assert_eq!(
        inspect_output["host_config_status"].as_str(),
        Some("present")
    );
    assert_eq!(
        inspect_output["host_config"]
            .as_mapping()
            .expect("empty host config should render as an object")
            .len(),
        0
    );

    let inspect_value = serde_json::to_value(responses::inspect_success(
        "plugins inspect",
        "acme.empty-config",
        &entry,
        &manifest,
        &manifest_ref,
        host_config_by_id.get("acme.empty-config"),
    ))
    .unwrap();
    assert_eq!(inspect_value["data"]["host_config_status"], "present");
    assert_eq!(
        inspect_value["data"]["host_config"]
            .as_object()
            .expect("empty host config should serialize as an object")
            .len(),
        0
    );
}
