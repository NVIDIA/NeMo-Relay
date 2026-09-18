// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::OsStr;

use base64::Engine;
use tempfile::tempdir;

use super::*;
use crate::test_support::EnvScope;

fn credential() -> String {
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([0x42_u8; 32])
}

fn spec(agents: impl IntoIterator<Item = ManagedAgent>) -> ManagedBundleSpec {
    ManagedBundleSpec::new(
        "https://relay.example.com:443",
        "/opt/nvidia/bin/nemo-relay-dispatch",
        ManagedPlatform::Linux,
        agents,
    )
    .unwrap()
}

fn managed_environment(token: &str) -> EnvScope {
    let header = format!("x-enterprise-context: fixed\n{ROUTE_TOKEN_HEADER}: {token}");
    EnvScope::set(&[
        (ROUTE_TOKEN_ENV, Some(OsStr::new(token))),
        (CLAUDE_CUSTOM_HEADERS_ENV, Some(OsStr::new(&header))),
    ])
}

fn make_test_artifact_writable(path: &std::path::Path) {
    let mut permissions = std::fs::metadata(path).unwrap().permissions();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        permissions.set_mode(0o600);
    }
    #[cfg(not(unix))]
    permissions.set_readonly(false);
    std::fs::set_permissions(path, permissions).unwrap();
}

#[test]
fn v1_render_is_deterministic_and_contains_only_deployment_constants() {
    let spec = spec([
        ManagedAgent::Codex,
        ManagedAgent::ClaudeCode,
        ManagedAgent::Pi,
    ]);
    let first = render_bundle(&spec).unwrap();
    let second = render_bundle(&spec).unwrap();
    assert_eq!(first.manifest, second.manifest);
    assert_eq!(first.artifacts.len(), 13);
    for (left, right) in first.artifacts.iter().zip(&second.artifacts) {
        assert_eq!(left.path, right.path);
        assert_eq!(left.bytes, right.bytes);
        let text = String::from_utf8(left.bytes.clone()).unwrap();
        if left.path != "pi/extension-v1/package.json" && env!("CARGO_PKG_VERSION") != "1.0.0" {
            assert!(!text.contains(env!("CARGO_PKG_VERSION")));
        }
        for forbidden in [
            "machine-identity",
            "generation_token",
            "/Users/",
            "C:\\Users\\",
        ] {
            assert!(
                !text.contains(forbidden),
                "{} contained {forbidden}",
                left.path
            );
        }
    }

    let codex_settings = first
        .artifacts
        .iter()
        .find(|artifact| artifact.path == "codex/settings-v1/config.toml")
        .unwrap();
    let codex_settings = String::from_utf8_lossy(&codex_settings.bytes);
    assert!(codex_settings.contains("https://relay.example.com:443/v1"));
    assert!(codex_settings.contains(ROUTE_TOKEN_HEADER));
    assert!(codex_settings.contains(ROUTE_TOKEN_ENV));
    assert!(!codex_settings.contains(&credential()));

    for artifact in first
        .artifacts
        .iter()
        .filter(|artifact| artifact.path.ends_with("hooks/hooks.json"))
    {
        let hooks = String::from_utf8_lossy(&artifact.bytes);
        assert!(hooks.contains("/opt/nvidia/bin/nemo-relay-dispatch daemon hook"));
        assert!(hooks.contains("--daemon-address https://relay.example.com:443"));
        assert!(!hooks.contains("hook-forward"));
    }

    assert_pi_bundle_artifacts(&first);
}

fn assert_pi_bundle_artifacts(bundle: &RenderedBundle) {
    let pi_config = bundle
        .artifacts
        .iter()
        .find(|artifact| artifact.path == "pi/extension-v1/managed-config.json")
        .unwrap();
    let pi_config: serde_json::Value = serde_json::from_slice(&pi_config.bytes).unwrap();
    assert_eq!(pi_config["daemonAddress"], "https://relay.example.com:443");
    assert_eq!(
        pi_config["dispatcherCommand"],
        "/opt/nvidia/bin/nemo-relay-dispatch"
    );

    for (path, expected) in [
        (
            "pi/extension-v1/README.md",
            canonical_embedded_text(include_str!(
                "../../../src/daemon/managed/pi_extension/README.md"
            ))
            .into_bytes(),
        ),
        (
            "pi/extension-v1/index.ts",
            canonical_embedded_text(include_str!(
                "../../../src/daemon/managed/pi_extension/index.ts"
            ))
            .into_bytes(),
        ),
        (
            "pi/extension-v1/package.json",
            canonical_embedded_text(include_str!(
                "../../../src/daemon/managed/pi_extension/package.json"
            ))
            .into_bytes(),
        ),
        (
            "pi/extension-v1/tsconfig.json",
            canonical_embedded_text(include_str!(
                "../../../src/daemon/managed/pi_extension/tsconfig.json"
            ))
            .into_bytes(),
        ),
    ] {
        let rendered = bundle
            .artifacts
            .iter()
            .find(|artifact| artifact.path == path)
            .unwrap();
        assert_eq!(rendered.bytes, expected, "{path}");
    }
}

#[test]
fn managed_pi_launch_disables_discovered_extensions() {
    const ISOLATED_LAUNCH: &str = "pi --no-extensions -e ";
    const NON_ISOLATED_LAUNCH: &str = "pi -e ";

    let rendered = render_bundle(&spec([ManagedAgent::Pi])).unwrap();
    let readme = rendered
        .artifacts
        .iter()
        .find(|artifact| artifact.path == "pi/extension-v1/README.md")
        .unwrap();
    let readme = std::str::from_utf8(&readme.bytes).unwrap();
    assert!(
        readme.contains(ISOLATED_LAUNCH),
        "rendered managed Pi README must suppress all discovered Pi extensions"
    );
    assert!(
        !readme.contains(NON_ISOLATED_LAUNCH),
        "rendered managed Pi README must not document a non-isolated managed Pi launch"
    );
}

#[test]
fn managed_pi_extension_forwards_custom_provider_endpoints_without_route_urls() {
    let source = include_str!("../../../src/daemon/managed/pi_extension/index.ts");

    for contract in [
        "['daemon', 'mcp', '--daemon-address'",
        "/hooks/pi",
        "pi.registerProvider(model.provider",
        "'openai-completions'",
        "'openai-responses'",
        "'anthropic-messages'",
        "pi.on('session_start'",
        "pi.on('session_before_compact'",
        "pi.on('session_compact'",
        "pi.on('tool_call'",
        "pi.on('user_bash'",
        "typeof toolCall.tool_call_id !== 'string'",
        "shapeViolation(current, toolCall.input)",
        "'tool_arguments_transformed'",
        "code: 'model-registry-unavailable'",
        "function toolResultText(content: unknown)",
        "function sliceAtCodePointBoundary(value: string",
        "const MCP_READY_TIMEOUT_MS = 180_000",
        "[CLIENT_TOKEN_HEADER]: active.credential",
        "const UPSTREAM_BASE_URL_HEADER = 'x-nemo-relay-upstream-base-url'",
        "[UPSTREAM_BASE_URL_HEADER]: decision.upstream",
    ] {
        assert!(source.contains(contract), "missing Pi contract: {contract}");
    }

    assert_eq!(
        source.matches("const CLIENT_TOKEN_HEADER =").count(),
        1,
        "the managed Pi extension must define one Relay-specific public header"
    );
    for forbidden in [
        "x-nemo-relay-session-id",
        "x-nemo-relay-fingerprint",
        "x-nemo-relay-generation",
    ] {
        assert!(
            !source.contains(forbidden),
            "managed Pi source contains forbidden routing metadata: {forbidden}"
        );
    }
}

#[test]
fn managed_pi_config_json_encodes_cross_platform_deployment_values() {
    let spec = ManagedBundleSpec::new(
        "https://relay.example.com:443",
        "C:\\ProgramData\\NVIDIA\\nemo-relay-dispatch.exe",
        ManagedPlatform::Windows,
        [ManagedAgent::Pi],
    )
    .unwrap();
    let rendered = render_bundle(&spec).unwrap();
    let config = rendered
        .artifacts
        .iter()
        .find(|artifact| artifact.path == "pi/extension-v1/managed-config.json")
        .unwrap();
    let text = String::from_utf8(config.bytes.clone()).unwrap();
    assert!(!text.contains("__NEMO_RELAY_"));
    let value: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(
        value["dispatcherCommand"],
        "C:\\ProgramData\\NVIDIA\\nemo-relay-dispatch.exe"
    );
}

#[test]
fn v1_artifact_bytes_do_not_depend_on_binary_or_platform_versioning() {
    let source = include_str!("../../../src/daemon/managed/mod.rs");
    assert!(!source.contains("env!(\"CARGO_PKG_VERSION\")"));

    let linux = render_bundle(&spec([ManagedAgent::Codex])).unwrap();
    let macos_spec = ManagedBundleSpec::new(
        "https://relay.example.com:443",
        "/opt/nvidia/bin/nemo-relay-dispatch",
        ManagedPlatform::Macos,
        [ManagedAgent::Codex],
    )
    .unwrap();
    let macos = render_bundle(&macos_spec).unwrap();
    let linux_artifacts = linux
        .artifacts
        .into_iter()
        .map(|artifact| (artifact.path, artifact.bytes))
        .collect::<Vec<_>>();
    let macos_artifacts = macos
        .artifacts
        .into_iter()
        .map(|artifact| (artifact.path, artifact.bytes))
        .collect::<Vec<_>>();
    assert_eq!(linux_artifacts, macos_artifacts);
}

#[test]
fn canonical_v1_bundle_matches_the_release_frozen_golden_digest() {
    // This digest pins the release-candidate v1 manifest and every artifact byte for a canonical
    // deployment. After v1 is published, behavior changes must use a separately named v2 family.
    const GOLDEN_SHA256: &str = "de9788ebe6c84b377f7cc583f1d725ae93fd82ba714382e11ffc034d9a642777";
    let rendered = render_bundle(&spec([
        ManagedAgent::Codex,
        ManagedAgent::ClaudeCode,
        ManagedAgent::Pi,
    ]))
    .unwrap();
    assert_eq!(rendered_bundle_digest(&rendered).to_string(), GOLDEN_SHA256);
}

#[test]
fn embedded_managed_text_has_platform_independent_line_endings() {
    assert_eq!(
        canonical_embedded_text("first\r\nsecond\r\n"),
        "first\nsecond\n"
    );
    assert_eq!(
        canonical_embedded_text("first\nsecond\n"),
        "first\nsecond\n"
    );
}

#[test]
fn write_is_create_only_and_existing_exact_bundle_is_not_rewritten() {
    let directory = tempdir().unwrap();
    let root = directory.path().join("bundle");
    let spec = spec([ManagedAgent::Codex]);
    let first_digest = write_new_bundle(&root, &spec).unwrap();
    let manifest = root.join(MANIFEST_FILE);
    let before = std::fs::metadata(&manifest).unwrap().modified().unwrap();
    let second_digest = write_new_bundle(&root, &spec).unwrap();
    assert_eq!(first_digest, second_digest);
    let after = std::fs::metadata(&manifest).unwrap().modified().unwrap();
    assert_eq!(before, after);

    let other_deployment = ManagedBundleSpec::new(
        "https://other-relay.example.com:443",
        "/opt/nvidia/bin/nemo-relay-dispatch",
        ManagedPlatform::Linux,
        [ManagedAgent::Codex],
    )
    .unwrap();
    let error = write_new_bundle(&root, &other_deployment)
        .unwrap_err()
        .to_string();
    assert!(error.contains("different deployment bytes"), "{error}");
    assert_eq!(
        before,
        std::fs::metadata(&manifest).unwrap().modified().unwrap()
    );

    let config = root.join("codex/settings-v1/config.toml");
    make_test_artifact_writable(&config);
    std::fs::write(&config, "changed\n").unwrap();
    let error = write_new_bundle(&root, &spec).unwrap_err().to_string();
    assert!(error.contains("exact canonical bytes"), "{error}");
    assert_eq!(
        std::fs::read_to_string(root.join("codex/settings-v1/config.toml")).unwrap(),
        "changed\n"
    );
}

#[test]
fn managed_refresh_is_validation_only() {
    let directory = tempdir().unwrap();
    let root = directory.path().join("bundle");
    let digest = write_new_bundle(&root, &spec([ManagedAgent::Codex])).unwrap();
    let before = std::fs::read(root.join(MANIFEST_FILE)).unwrap();
    let token = credential();
    let _environment = EnvScope::set(&[(ROUTE_TOKEN_ENV, Some(OsStr::new(&token)))]);

    refresh_bundle(&root, &digest).unwrap();

    assert_eq!(std::fs::read(root.join(MANIFEST_FILE)).unwrap(), before);
}

#[test]
fn doctor_validation_checks_exact_bytes_and_managed_environment() {
    let directory = tempdir().unwrap();
    let root = directory.path().join("bundle");
    let digest = write_new_bundle(
        &root,
        &spec([ManagedAgent::Codex, ManagedAgent::ClaudeCode]),
    )
    .unwrap();
    let token = credential();
    let _environment = managed_environment(&token);

    let validation = refresh_bundle(&root, &digest).unwrap();
    assert_eq!(validation.artifact_count, 8);
    assert_eq!(validation.daemon_address, "https://relay.example.com:443");
    assert_eq!(validation.sha256, digest);

    let hooks = root.join("claude-code/plugin-v1/hooks/hooks.json");
    make_test_artifact_writable(&hooks);
    let mut changed = std::fs::read(&hooks).unwrap();
    changed.push(b' ');
    std::fs::write(&hooks, changed).unwrap();
    let error = refresh_bundle(&root, &digest).unwrap_err().to_string();
    assert!(
        error.contains("expected size limit") || error.contains("exact canonical bytes"),
        "{error}"
    );
}

#[test]
fn doctor_requires_the_separately_provisioned_bundle_digest() {
    let directory = tempdir().unwrap();
    let root = directory.path().join("bundle");
    let digest = write_new_bundle(&root, &spec([ManagedAgent::Codex])).unwrap();
    let wrong: ManagedBundleDigest =
        "0000000000000000000000000000000000000000000000000000000000000000"
            .parse()
            .unwrap();
    let _environment = EnvScope::set(&[(ROUTE_TOKEN_ENV, None)]);

    let error = refresh_bundle(&root, &wrong).unwrap_err().to_string();
    assert!(error.contains("SHA-256 mismatch"), "{error}");
    assert!(error.contains(&digest.to_string()), "{error}");
    assert!(!error.contains(ROUTE_TOKEN_ENV), "{error}");

    assert!(digest.to_string().parse::<ManagedBundleDigest>().is_ok());
    for malformed in [
        "0",
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        "gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg",
    ] {
        assert!(malformed.parse::<ManagedBundleDigest>().is_err());
    }
}

#[test]
fn claude_environment_must_bind_the_custom_header_to_the_route_token() {
    let directory = tempdir().unwrap();
    let root = directory.path().join("bundle");
    let digest = write_new_bundle(&root, &spec([ManagedAgent::ClaudeCode])).unwrap();
    let token = credential();
    let wrong = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([0x24_u8; 32]);
    let header = format!("{ROUTE_TOKEN_HEADER}: {wrong}");
    let _environment = EnvScope::set(&[
        (ROUTE_TOKEN_ENV, Some(OsStr::new(&token))),
        (CLAUDE_CUSTOM_HEADERS_ENV, Some(OsStr::new(&header))),
    ]);

    let error = refresh_bundle(&root, &digest).unwrap_err().to_string();
    assert!(error.contains("must contain exactly one"), "{error}");
    assert!(!error.contains(&token));
    assert!(!error.contains(&wrong));
}

#[test]
fn managed_environment_requires_the_enterprise_provisioned_credential() {
    let directory = tempdir().unwrap();
    let root = directory.path().join("bundle");
    let digest = write_new_bundle(&root, &spec([ManagedAgent::Codex])).unwrap();
    let _environment = EnvScope::set(&[(ROUTE_TOKEN_ENV, None)]);

    let error = refresh_bundle(&root, &digest).unwrap_err().to_string();
    assert!(error.contains(ROUTE_TOKEN_ENV), "{error}");
    assert!(
        error.contains("enterprise") || error.contains("managed"),
        "{error}"
    );
}

#[test]
fn doctor_rejects_extra_files_and_noncanonical_manifest_bytes() {
    let directory = tempdir().unwrap();
    let root = directory.path().join("bundle");
    let digest = write_new_bundle(&root, &spec([ManagedAgent::Pi])).unwrap();
    let token = credential();
    let _environment = EnvScope::set(&[(ROUTE_TOKEN_ENV, Some(OsStr::new(&token)))]);

    std::fs::write(root.join("unmanaged.json"), "{}\n").unwrap();
    let error = refresh_bundle(&root, &digest).unwrap_err().to_string();
    assert!(error.contains("unexpected artifact"), "{error}");
    std::fs::remove_file(root.join("unmanaged.json")).unwrap();

    let manifest = root.join(MANIFEST_FILE);
    make_test_artifact_writable(&manifest);
    let value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest).unwrap()).unwrap();
    std::fs::write(&manifest, serde_json::to_vec(&value).unwrap()).unwrap();
    let error = refresh_bundle(&root, &digest).unwrap_err().to_string();
    assert!(error.contains("canonical"), "{error}");
}

#[test]
fn doctor_rejects_missing_oversized_and_non_file_artifacts() {
    let directory = tempdir().unwrap();
    let token = credential();
    let _environment = EnvScope::set(&[(ROUTE_TOKEN_ENV, Some(OsStr::new(&token)))]);

    let missing_root = directory.path().join("missing-root");
    let digest: ManagedBundleDigest =
        "0000000000000000000000000000000000000000000000000000000000000000"
            .parse()
            .unwrap();
    assert!(refresh_bundle(&missing_root, &digest).is_err());

    let regular_root = directory.path().join("regular-root");
    std::fs::write(&regular_root, b"not a directory").unwrap();
    assert!(refresh_bundle(&regular_root, &digest).is_err());

    let root = directory.path().join("bundle");
    let digest = write_new_bundle(&root, &spec([ManagedAgent::Codex])).unwrap();
    let artifact = root.join("codex/settings-v1/config.toml");
    std::fs::remove_file(&artifact).unwrap();
    assert!(refresh_bundle(&root, &digest).is_err());

    write_new_bundle(&root, &spec([ManagedAgent::Codex])).unwrap_err();
    std::fs::create_dir(&artifact).unwrap();
    assert!(refresh_bundle(&root, &digest).is_err());
}

#[cfg(unix)]
#[test]
fn doctor_rejects_symlinked_manifest_and_nested_artifact() {
    let directory = tempdir().unwrap();
    let token = credential();
    let _environment = EnvScope::set(&[(ROUTE_TOKEN_ENV, Some(OsStr::new(&token)))]);

    for relative in [MANIFEST_FILE, "codex/settings-v1/config.toml"] {
        let root = directory.path().join(relative.replace('/', "-"));
        let digest = write_new_bundle(&root, &spec([ManagedAgent::Codex])).unwrap();
        let target = root.join(relative);
        let original = directory
            .path()
            .join(format!("{}.original", relative.replace('/', "-")));
        std::fs::rename(&target, &original).unwrap();
        std::os::unix::fs::symlink(&original, &target).unwrap();
        let error = refresh_bundle(&root, &digest).unwrap_err().to_string();
        assert!(error.contains("symlink"), "{relative}: {error}");
    }
}

#[test]
fn pi_placeholder_replacement_requires_exactly_one_json_string() {
    assert!(replace_json_string_value("{}", "missing", "value").is_err());
    assert!(replace_json_string_value(r#"["same","same"]"#, "same", "value").is_err());
    assert_eq!(
        replace_json_string_value(
            r#"{"key":"placeholder"}"#,
            "placeholder",
            "quoted \"value\""
        )
        .unwrap(),
        r#"{"key":"quoted \"value\""}"#
    );
}

#[cfg(unix)]
#[test]
fn doctor_rejects_a_symlinked_bundle_root() {
    let directory = tempdir().unwrap();
    let root = directory.path().join("bundle");
    let alias = directory.path().join("bundle-alias");
    let digest = write_new_bundle(&root, &spec([ManagedAgent::Pi])).unwrap();
    std::os::unix::fs::symlink(&root, &alias).unwrap();

    let error = refresh_bundle(&alias, &digest).unwrap_err().to_string();
    assert!(
        error.contains("must be a directory, not a symlink"),
        "{error}"
    );
}

#[test]
fn spec_rejects_mutable_or_unsafe_inputs() {
    for address in [
        "https://relay.example.com",
        "http://relay.example.com:80",
        "https://relay.example.com:443/tenant/alice",
    ] {
        assert!(
            ManagedBundleSpec::new(
                address,
                "C:\\ProgramData\\NVIDIA\\nemo-relay-dispatch.exe",
                ManagedPlatform::Windows,
                [ManagedAgent::Codex]
            )
            .is_err(),
            "accepted {address}"
        );
    }
    for dispatcher in [
        "",
        "nemo relay",
        "nemo-relay-dispatch",
        "nemo-relay;malicious",
        "/Users/alice/bin/nemo-relay",
        "/home/alice/bin/nemo-relay",
        "/root/bin/nemo-relay",
        "/tmp/nemo-relay",
        "/opt/nvidia/../alice/nemo-relay",
        "C:\\ProgramData\\NVIDIA\\nemo-relay.exe",
    ] {
        assert!(
            ManagedBundleSpec::new(
                "https://relay.example.com:443",
                dispatcher,
                ManagedPlatform::Macos,
                [ManagedAgent::Codex]
            )
            .is_err(),
            "accepted {dispatcher}"
        );
    }

    for (platform, dispatcher) in [
        (ManagedPlatform::Windows, "nemo-relay-dispatch"),
        (ManagedPlatform::Windows, "C:\\Users\\alice\\nemo-relay.exe"),
        (
            ManagedPlatform::Windows,
            "C:\\Windows\\Temp\\nemo-relay.exe",
        ),
        (ManagedPlatform::Windows, "/opt/nvidia/nemo-relay"),
    ] {
        assert!(
            ManagedBundleSpec::new(
                "https://relay.example.com:443",
                dispatcher,
                platform,
                [ManagedAgent::Codex]
            )
            .is_err(),
            "accepted {dispatcher}"
        );
    }

    for (platform, dispatcher) in [
        (
            ManagedPlatform::Linux,
            "/opt/nvidia/bin/nemo-relay-dispatch",
        ),
        (
            ManagedPlatform::Macos,
            "/Library/NVIDIA/bin/nemo-relay-dispatch",
        ),
        (
            ManagedPlatform::Windows,
            "C:\\ProgramData\\NVIDIA\\nemo-relay-dispatch.exe",
        ),
        (
            ManagedPlatform::Windows,
            "\\\\relay.example.com\\nvidia\\nemo-relay-dispatch.exe",
        ),
    ] {
        ManagedBundleSpec::new(
            "https://relay.example.com:443",
            dispatcher,
            platform,
            [ManagedAgent::Codex],
        )
        .unwrap();
    }
}

#[test]
fn managed_bundle_value_types_and_empty_agent_sets_are_strict() {
    assert_eq!(ManagedPlatform::Linux.as_str(), "linux");
    assert_eq!(ManagedPlatform::Macos.as_str(), "macos");
    assert_eq!(ManagedPlatform::Windows.as_str(), "windows");
    assert_eq!(ManagedAgent::Codex.hook_argument(), "codex");
    assert_eq!(ManagedAgent::ClaudeCode.hook_argument(), "claude");
    assert_eq!(ManagedAgent::Pi.hook_argument(), "pi");
    assert!(
        ManagedBundleSpec::new(
            "https://relay.example.com:443",
            "/opt/nvidia/bin/nemo-relay",
            ManagedPlatform::Linux,
            [],
        )
        .is_err()
    );
    for invalid in [
        "",
        "ABCDEF0000000000000000000000000000000000000000000000000000000000",
        "g000000000000000000000000000000000000000000000000000000000000000",
    ] {
        assert!(invalid.parse::<ManagedBundleDigest>().is_err());
    }
}

#[test]
fn managed_bundle_rejects_invalid_manifest_family_and_artifact_capacity() {
    let directory = tempdir().unwrap();
    let root = directory.path().join("bundle");
    let _digest = write_new_bundle(&root, &spec([ManagedAgent::Codex])).unwrap();
    let manifest_path = root.join(MANIFEST_FILE);
    make_test_artifact_writable(&manifest_path);
    let original = std::fs::read(&manifest_path).unwrap();
    let mut manifest: serde_json::Value = serde_json::from_slice(&original).unwrap();
    manifest["schema_version"] = json!(SCHEMA_VERSION + 1);
    std::fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert!(validate_bundle_files(&root, false, None).is_err());

    manifest = serde_json::from_slice(&original).unwrap();
    manifest["artifacts"] = Value::Array(
        (0..=MAX_ARTIFACTS)
            .map(|index| {
                json!({
                    "agent": "codex",
                    "path": format!("artifact-{index}"),
                    "byte_length": 0,
                    "sha256": "0".repeat(64),
                })
            })
            .collect(),
    );
    std::fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert!(validate_bundle_files(&root, false, None).is_err());
}

#[test]
fn managed_file_helpers_reject_overwrite_oversize_and_excessive_entries() {
    let directory = tempdir().unwrap();
    let file = directory.path().join("artifact");
    write_new_file(&file, b"first").unwrap();
    assert!(write_new_file(&file, b"second").is_err());
    assert_eq!(read_bounded(&file, 5).unwrap(), b"first");
    assert!(read_bounded(&file, 4).is_err());
    assert!(read_bounded(&directory.path().join("missing"), 4).is_err());

    let root = directory.path().join("bundle");
    std::fs::create_dir(&root).unwrap();
    for index in 0..=(MAX_ARTIFACTS * 4) {
        std::fs::create_dir(root.join(format!("empty-{index}"))).unwrap();
    }
    assert!(bundle_files(&root).is_err());
}
