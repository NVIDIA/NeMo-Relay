// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cell::Cell;
use std::ffi::OsString;
use std::path::Path;
use std::sync::MutexGuard;
use std::time::{Duration, UNIX_EPOCH};

use serde_json::{Value, json};

use super::*;
use crate::config::CodingAgent;

const TEST_GENERATION_TOKEN: &str = "test-generation";

fn relay_binary(root: &Path) -> PathBuf {
    let path = root.join("NeMo Relay's bin").join("nemo-relay");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, b"relay").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    path
}

fn paths(root: &Path) -> PersistentPaths {
    PersistentPaths::for_config(root.join("config.yaml")).unwrap()
}

fn yaml(path: &Path) -> Value {
    serde_yaml::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn json_file(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

struct XdgConfigHomeScope {
    _guard: MutexGuard<'static, ()>,
    previous: Option<OsString>,
}

impl XdgConfigHomeScope {
    fn enter(path: &Path) -> Self {
        let guard = crate::test_support::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let previous = std::env::var_os("XDG_CONFIG_HOME");
        // SAFETY: This scope holds the process-wide environment mutex.
        unsafe { std::env::set_var("XDG_CONFIG_HOME", path) };
        Self {
            _guard: guard,
            previous,
        }
    }
}

impl Drop for XdgConfigHomeScope {
    fn drop(&mut self) {
        // SAFETY: This restores the process environment while the mutex is still held.
        unsafe {
            match self.previous.take() {
                Some(value) => std::env::set_var("XDG_CONFIG_HOME", value),
                None => std::env::remove_var("XDG_CONFIG_HOME"),
            }
        }
    }
}

#[test]
fn user_config_path_uses_hermes_home_or_platform_home() {
    let default_home = Path::new("/users/relay");
    assert_eq!(
        user_config_path_with_override(default_home, None),
        default_home.join(".hermes/config.yaml")
    );
    assert_eq!(
        user_config_path_with_override(default_home, Some("/profiles/hermes".into())),
        Path::new("/profiles/hermes/config.yaml")
    );
    assert_eq!(
        user_config_path_with_override(default_home, Some("".into())),
        default_home.join(".hermes/config.yaml")
    );
}

#[test]
fn install_lock_serializes_concurrent_hermes_config_updates() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("config.yaml");
    let _first = acquire_install_lock(&config, Duration::from_millis(10)).unwrap();

    let error = acquire_install_lock(&config, Duration::ZERO).unwrap_err();

    assert!(
        error.contains("another Hermes integration update"),
        "{error}"
    );
}

#[test]
fn install_uses_the_native_hermes_allowlist_lock() {
    let temp = tempfile::tempdir().unwrap();
    let allowlist = temp.path().join("shell-hooks-allowlist.json");
    let _first = acquire_allowlist_lock(&allowlist, Duration::from_millis(10)).unwrap();

    let error = acquire_allowlist_lock(&allowlist, Duration::ZERO).unwrap_err();

    assert!(error.contains("shell-hook approval update"), "{error}");
    assert!(temp.path().join("shell-hooks-allowlist.json.lock").exists());
}

#[test]
fn hook_command_round_trips_paths_and_recognizes_owned_legacy_spellings() {
    let relay = Path::new("/tmp/NeMo $Relay`test'/bin/nemo-relay");
    let generation = Path::new("/tmp/generation");
    assert_eq!(
        persistent_hook_command_for_platform(relay, generation, TEST_GENERATION_TOKEN, false),
        "'/tmp/NeMo $Relay`test'\\''/bin/nemo-relay' hook-forward hermes --gateway-url http://127.0.0.1:47632 --generation-file /tmp/generation --generation-token test-generation"
    );
    assert_eq!(
        crate::installer::decode_windows_hook_command(&persistent_hook_command_for_platform(
            Path::new(r"C:\Program Files\NeMo 100%\bin\nemo-relay.exe"),
            Path::new(r"C:\Temp\generation"),
            TEST_GENERATION_TOKEN,
            true,
        ))
        .unwrap(),
        vec![
            r"C:\Program Files\NeMo 100%\bin\nemo-relay.exe",
            "hook-forward",
            "hermes",
            "--gateway-url",
            crate::sidecar::DEFAULT_URL,
            "--generation-file",
            r"C:\Temp\generation",
            "--generation-token",
            TEST_GENERATION_TOKEN,
        ]
    );
    assert_eq!(
        crate::installer::transparent_hook_forward_command_for_platform(
            relay,
            CodingAgent::Hermes,
            "http://127.0.0.1:1234",
            false,
        ),
        "'/tmp/NeMo $Relay`test'\\''/bin/nemo-relay' hook-forward hermes --gateway-url http://127.0.0.1:1234 --transparent-run"
    );
    let encoded = persistent_hook_command_for_platform(
        Path::new(r"C:\Program Files\NeMo 100%\bin\nemo-relay.exe"),
        Path::new(r"C:\Temp\generation"),
        TEST_GENERATION_TOKEN,
        true,
    );
    let encoded_codex = crate::installer::persistent_hook_forward_command_for_platform(
        Path::new(r"C:\Program Files\NeMo 100%\bin\nemo-relay.exe"),
        CodingAgent::Codex,
        Path::new(r"C:\Temp\generation"),
        TEST_GENERATION_TOKEN,
        true,
    );
    for command in [
        "nemo-relay hook-forward hermes",
        "/old/path/nemo-relay plugin-shim hook hermes",
        "'/old/plugin-shim hook hermes/nemo-relay' plugin-shim hook hermes",
        "'/old install/nemo-relay' plugin-shim hook hermes --gateway-url http://127.0.0.1:47632",
        r#""C:\Program Files\NeMo 100%%\nemo-relay.exe" plugin-shim hook hermes"#,
        &encoded,
    ] {
        assert!(
            is_managed_hook_command(command),
            "not recognized: {command}"
        );
    }
    for command in [
        "relay-helper hook-forward hermes",
        "echo nemo-relay hook-forward hermes",
        "nemo-relay plugin-shim hook codex",
        "nemo-relay-safe plugin-shim hook hermes",
        &encoded_codex,
    ] {
        assert!(!is_managed_hook_command(command), "overmatched: {command}");
    }
}

#[test]
fn forwarded_environment_includes_static_dynamic_and_config_referenced_names() {
    let environment = vec![
        "AWS_REGION".into(),
        "NEMO_RELAY_CUSTOM".into(),
        "NEMO_RELAY_WORKER_TOKEN".into(),
        "UNRELATED_SECRET".into(),
    ];
    let config = json!({
        "header_env": {"Authorization": "CUSTOM_EXPORT_TOKEN"},
        "secret_access_key_var": "AWS_PRIVATE_SECRET",
        "session_token_var": "NEMO_RELAY_WORKER_TOKEN"
    });
    let names = forwarded_environment_names(&environment, Some(&config));

    assert!(names.contains(&"ANTHROPIC_API_KEY".into()));
    assert!(names.contains(&"OPENAI_API_KEY".into()));
    assert!(names.contains(&"AWS_REGION".into()));
    assert!(names.contains(&"NEMO_RELAY_CUSTOM".into()));
    assert!(names.contains(&"CUSTOM_EXPORT_TOKEN".into()));
    assert!(names.contains(&"AWS_PRIVATE_SECRET".into()));
    assert!(names.contains(&"AWS_PROFILE".into()));
    assert!(names.contains(&"OTEL_EXPORTER_OTLP_ENDPOINT".into()));
    assert!(!names.contains(&"NEMO_RELAY_WORKER_TOKEN".into()));
    assert!(!names.contains(&"UNRELATED_SECRET".into()));
}

#[test]
fn persistent_config_migrates_owned_state_and_preserves_unrelated_config() {
    let temp = tempfile::tempdir().unwrap();
    let relay = relay_binary(temp.path());
    let generation = temp.path().join(GENERATION_FILE_NAME);
    let command = persistent_hook_command(&relay, &generation, TEST_GENERATION_TOKEN).unwrap();
    let existing = r#"
model: keep-me
mcp_servers:
  filesystem:
    command: fs-mcp
  nemo-relay:
    command: /old/bin/nemo-relay
    args: [mcp, --agent, hermes]
hooks:
  on_session_start:
    - command: custom-hook
      timeout: 9
    - command: nemo-relay hook-forward hermes
      timeout: 30
  legacy_event:
    - command: /old/bin/nemo-relay plugin-shim hook hermes
  custom_event:
    - command: keep-custom
"#;
    let merged = persistent_config(
        Some(existing),
        &relay,
        &command,
        &generation,
        TEST_GENERATION_TOKEN,
        &["AWS_REGION".into()],
    )
    .unwrap();

    assert_eq!(merged["model"], json!("keep-me"));
    assert_eq!(
        merged["mcp_servers"]["filesystem"]["command"],
        json!("fs-mcp")
    );
    assert_eq!(
        merged["mcp_servers"][MCP_SERVER_NAME],
        expected_mcp_server(
            &relay,
            &generation,
            TEST_GENERATION_TOKEN,
            &["AWS_REGION".into()]
        )
    );
    assert_eq!(
        merged["mcp_servers"][MCP_SERVER_NAME]["env"]["AWS_REGION"],
        json!("${AWS_REGION}")
    );
    assert_eq!(
        merged["hooks"]["on_session_start"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        merged["hooks"]["on_session_start"][0]["command"],
        json!("custom-hook")
    );
    assert_eq!(
        merged["hooks"]["on_session_start"][1]["command"],
        json!(command)
    );
    assert!(merged["hooks"].get("legacy_event").is_none());
    assert_eq!(
        merged["hooks"]["custom_event"][0]["command"],
        json!("keep-custom")
    );
    for event in CodingAgent::Hermes.hook_events() {
        let groups = merged["hooks"][event].as_array().unwrap();
        assert_eq!(
            groups
                .iter()
                .filter(|group| group["command"] == json!(command))
                .count(),
            1,
            "event {event}"
        );
    }
}

#[test]
fn persistent_config_rejects_a_foreign_server_with_the_reserved_name() {
    let temp = tempfile::tempdir().unwrap();
    let relay = relay_binary(temp.path());
    let generation = temp.path().join(GENERATION_FILE_NAME);
    let command = persistent_hook_command(&relay, &generation, TEST_GENERATION_TOKEN).unwrap();
    let existing = r#"
model: keep-me
mcp_servers:
  nemo-relay:
    command: foreign-mcp
    args: [serve]
"#;

    let error = persistent_config(
        Some(existing),
        &relay,
        &command,
        &generation,
        TEST_GENERATION_TOKEN,
        &[],
    )
    .unwrap_err()
    .to_string();

    assert!(error.contains("not managed by Relay"), "{error}");
    assert!(error.contains("rename or remove"), "{error}");
}

#[test]
fn foreign_reserved_server_aborts_install_before_any_file_changes() {
    let temp = tempfile::tempdir().unwrap();
    let relay = relay_binary(temp.path());
    let paths = paths(&temp.path().join("hermes"));
    std::fs::create_dir_all(paths.config.parent().unwrap()).unwrap();
    let config =
        b"# preserve\nmcp_servers:\n  nemo-relay:\n    command: foreign-mcp\n    args: [serve]\n";
    let allowlist = b"{\"approvals\":[{\"event\":\"custom\",\"command\":\"custom-hook\"}]}\n";
    std::fs::write(&paths.config, config).unwrap();
    std::fs::write(&paths.allowlist, allowlist).unwrap();

    let error = install_persistent_with(paths.clone(), &relay, &[], None, UNIX_EPOCH, atomic_write)
        .unwrap_err()
        .to_string();

    assert!(error.contains("not managed by Relay"), "{error}");
    assert_eq!(std::fs::read(&paths.config).unwrap(), config);
    assert_eq!(std::fs::read(&paths.allowlist).unwrap(), allowlist);
    assert!(!paths.generation.exists());
}

#[test]
fn trusted_hooks_migrates_only_relay_approvals_and_records_every_event() {
    let temp = tempfile::tempdir().unwrap();
    let relay = relay_binary(temp.path());
    let generation = temp.path().join(GENERATION_FILE_NAME);
    let command = persistent_hook_command(&relay, &generation, TEST_GENERATION_TOKEN).unwrap();
    let existing = json!({
        "schema": 7,
        "approvals": [
            {"event": "custom", "command": "custom-hook", "approved_at": "keep"},
            {"event": "on_session_start", "command": "nemo-relay hook-forward hermes"},
            {"event": "on_session_end", "command": "/old/nemo-relay plugin-shim hook hermes"}
        ]
    });
    let now = UNIX_EPOCH + Duration::from_secs(1_700_000_000);
    let merged = trusted_hooks(
        Some(&serde_json::to_string(&existing).unwrap()),
        &command,
        &relay,
        now,
    )
    .unwrap();
    let approvals = merged["approvals"].as_array().unwrap();

    assert_eq!(merged["schema"], json!(7));
    assert!(
        approvals
            .iter()
            .any(|entry| entry["command"] == json!("custom-hook"))
    );
    assert_eq!(approvals.len(), CodingAgent::Hermes.hook_events().len() + 1);
    for event in CodingAgent::Hermes.hook_events() {
        let entries = approvals
            .iter()
            .filter(|entry| entry["event"] == json!(event) && entry["command"] == json!(command))
            .collect::<Vec<_>>();
        assert_eq!(entries.len(), 1, "event {event}");
        assert_eq!(
            entries[0]["approved_at"],
            json!("2023-11-14T22:13:20.000000Z")
        );
        assert!(entries[0].get("script_mtime_at_approval").is_some());
    }
}

#[test]
fn verification_rejects_relay_handlers_and_approvals_on_unexpected_events() {
    let temp = tempfile::tempdir().unwrap();
    let relay = relay_binary(temp.path());
    let generation = temp.path().join(GENERATION_FILE_NAME);
    let command = persistent_hook_command(&relay, &generation, TEST_GENERATION_TOKEN).unwrap();
    let mut config = persistent_config(
        None,
        &relay,
        &command,
        &generation,
        TEST_GENERATION_TOKEN,
        &[],
    )
    .unwrap();
    config["hooks"]["unexpected_event"] = json!([{"command": command, "timeout": 30}]);
    let error = verify_hook_definitions(&config, &command).unwrap_err();
    assert!(error.contains("unexpected Relay hook"));
    let mut malformed = persistent_config(
        None,
        &relay,
        &command,
        &generation,
        TEST_GENERATION_TOKEN,
        &[],
    )
    .unwrap();
    malformed["hooks"]["unexpected_event"] = json!({"command": command});
    let error = verify_hook_definitions(&malformed, &command).unwrap_err();
    assert!(error.contains("must be an array"));

    let mut allowlist = trusted_hooks(None, &command, &relay, UNIX_EPOCH).unwrap();
    allowlist["approvals"].as_array_mut().unwrap().push(json!({
        "event": "unexpected_event",
        "command": command,
        "approved_at": "1970-01-01T00:00:00.000000Z"
    }));
    let path = temp.path().join("shell-hooks-allowlist.json");
    std::fs::write(&path, serde_json::to_vec(&allowlist).unwrap()).unwrap();
    let error = verify_trust(&path, &command).unwrap_err();
    assert!(error.contains("unexpected Relay hook approval"));

    let mut missing_event = trusted_hooks(None, &command, &relay, UNIX_EPOCH).unwrap();
    missing_event["approvals"]
        .as_array_mut()
        .unwrap()
        .push(json!({
            "command": command,
            "approved_at": "1970-01-01T00:00:00.000000Z"
        }));
    std::fs::write(&path, serde_json::to_vec(&missing_event).unwrap()).unwrap();
    let error = verify_trust(&path, &command).unwrap_err();
    assert!(error.contains("missing its event"));
}

#[test]
fn hermes_structure_and_trust_validation_cover_exact_failure_shapes() {
    let temp = tempfile::tempdir().unwrap();
    let relay = relay_binary(temp.path());
    let generation = temp.path().join(GENERATION_FILE_NAME);
    let command = persistent_hook_command(&relay, &generation, TEST_GENERATION_TOKEN).unwrap();

    let error = trusted_hooks(Some(r#"{"approvals": {}}"#), &command, &relay, UNIX_EPOCH)
        .unwrap_err()
        .to_string();
    assert!(error.contains("approvals must be an array"), "{error}");

    let error = parse_json_object(Some("[]"), "test allowlist")
        .unwrap_err()
        .to_string();
    assert!(error.contains("must contain a JSON object"), "{error}");

    let mut malformed_hooks = json!({"hooks": {"on_session_start": {}}});
    let error = strip_managed_hooks(&mut malformed_hooks)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("on_session_start hooks must be an array"),
        "{error}"
    );

    let error = parse_yaml_object(Some("[]"), "test config")
        .unwrap_err()
        .to_string();
    assert!(error.contains("must contain an object"), "{error}");
    assert!(is_managed_hook_command(
        "nemo-relay hook-forward hermes --fail-closed"
    ));

    let path = temp.path().join("shell-hooks-allowlist.json");
    let mut missing = trusted_hooks(None, &command, &relay, UNIX_EPOCH).unwrap();
    missing["approvals"].as_array_mut().unwrap().remove(0);
    std::fs::write(&path, serde_json::to_vec(&missing).unwrap()).unwrap();
    let error = verify_trust(&path, &command).unwrap_err();
    assert!(
        error.contains("expected exactly one trust approval"),
        "{error}"
    );

    let mut with_opaque_entry = trusted_hooks(None, &command, &relay, UNIX_EPOCH).unwrap();
    with_opaque_entry["approvals"]
        .as_array_mut()
        .unwrap()
        .push(json!({"metadata": "unrelated"}));
    std::fs::write(&path, serde_json::to_vec(&with_opaque_entry).unwrap()).unwrap();
    verify_trust(&path, &command).unwrap();
}

#[test]
fn install_is_verified_idempotent_and_rotates_the_generation() {
    let temp = tempfile::tempdir().unwrap();
    let relay = relay_binary(temp.path());
    let paths = paths(&temp.path().join("hermes"));
    let environment = vec!["OTEL_SERVICE_NAME".into()];
    let now = UNIX_EPOCH + Duration::from_secs(1_700_000_000);

    let written =
        install_persistent_with(paths.clone(), &relay, &environment, None, now, atomic_write)
            .unwrap();
    assert_eq!(written, paths.all());
    let first_generation =
        crate::install_generation::InstallGeneration::capture(paths.generation.clone())
            .unwrap()
            .token()
            .to_owned();
    let first_config = yaml(&paths.config);
    let first_command = first_config["hooks"]["on_session_start"][0]["command"]
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(
        first_config["mcp_servers"][MCP_SERVER_NAME]["env"][GENERATION_TOKEN_ENV],
        json!(first_generation)
    );
    assert!(crate::hook_assertions::command_has_arguments(
        &first_command,
        &["--generation-token", &first_generation]
    ));

    install_persistent_with(paths.clone(), &relay, &environment, None, now, atomic_write).unwrap();
    let second_generation =
        crate::install_generation::InstallGeneration::capture(paths.generation.clone())
            .unwrap()
            .token()
            .to_owned();
    assert_ne!(first_generation, second_generation);

    let config = yaml(&paths.config);
    assert_eq!(
        config["hooks"]["on_session_start"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|group| is_managed_hook_command(group["command"].as_str().unwrap()))
            .count(),
        1
    );
    assert_eq!(
        config["mcp_servers"][MCP_SERVER_NAME]["env"][GENERATION_FILE_ENV],
        json!(paths.generation.display().to_string())
    );
    assert_eq!(
        config["mcp_servers"][MCP_SERVER_NAME]["env"][GENERATION_TOKEN_ENV],
        json!(second_generation)
    );
    assert_ne!(
        first_config["mcp_servers"][MCP_SERVER_NAME]["env"][GENERATION_TOKEN_ENV],
        config["mcp_servers"][MCP_SERVER_NAME]["env"][GENERATION_TOKEN_ENV]
    );
    assert!(crate::hook_assertions::command_has_arguments(
        &first_command,
        &["--generation-token", &first_generation]
    ));
    assert!(!crate::hook_assertions::command_has_arguments(
        &first_command,
        &["--generation-token", &second_generation]
    ));
    assert_eq!(
        config["mcp_servers"][MCP_SERVER_NAME]["env"]["OTEL_SERVICE_NAME"],
        json!("${OTEL_SERVICE_NAME}")
    );
    assert_eq!(
        json_file(&paths.allowlist)["approvals"]
            .as_array()
            .unwrap()
            .len(),
        CodingAgent::Hermes.hook_events().len()
    );
}

#[test]
fn reinstall_verifies_generation_through_the_existing_retirement_transaction() {
    let temp = tempfile::tempdir().unwrap();
    let relay = relay_binary(temp.path());
    let paths = paths(&temp.path().join("hermes"));
    install_persistent_with(paths.clone(), &relay, &[], None, UNIX_EPOCH, atomic_write).unwrap();
    let first_token = InstallGeneration::capture(paths.generation.clone())
        .unwrap()
        .token()
        .to_owned();
    let mut retirement = GenerationRetirement::acquire(&paths.generation)
        .unwrap()
        .unwrap();
    retirement.invalidate_for_replacement().unwrap();

    let result = install_persistent_with_generation(
        paths.clone(),
        &relay,
        &[],
        None,
        Some(&retirement),
        UNIX_EPOCH,
        atomic_write,
    );
    finish_generation_mutation(result, Some(&mut retirement), "install").unwrap();
    drop(retirement);

    let second_token = InstallGeneration::capture(paths.generation)
        .unwrap()
        .token()
        .to_owned();
    assert_ne!(first_token, second_token);
}

#[test]
fn diagnosis_rejects_a_stale_mcp_generation_identity() {
    let temp = tempfile::tempdir().unwrap();
    let relay = relay_binary(temp.path());
    let paths = paths(&temp.path().join("hermes"));
    install_persistent_with(paths.clone(), &relay, &[], None, UNIX_EPOCH, atomic_write).unwrap();
    let mut config = yaml(&paths.config);
    config["mcp_servers"][MCP_SERVER_NAME]["env"][GENERATION_TOKEN_ENV] = json!("stale-generation");
    std::fs::write(&paths.config, serde_yaml::to_string(&config).unwrap()).unwrap();

    let error = diagnose_persistent(&paths.config).unwrap_err();

    assert!(
        error.contains("expected generation identity is stale"),
        "{error}"
    );
}

#[test]
fn install_rolls_back_config_allowlist_and_generation_after_write_failure() {
    let temp = tempfile::tempdir().unwrap();
    let relay = relay_binary(temp.path());
    let paths = paths(&temp.path().join("hermes"));
    std::fs::create_dir_all(paths.config.parent().unwrap()).unwrap();
    let originals = [
        (&paths.config, b"model: original\n".as_slice()),
        (
            &paths.allowlist,
            b"{\"approvals\":[{\"event\":\"x\",\"command\":\"custom\"}]}\n".as_slice(),
        ),
        (&paths.generation, b"original-generation\n".as_slice()),
    ];
    for (path, bytes) in originals {
        std::fs::write(path, bytes).unwrap();
    }
    let before = paths.all().map(|path| std::fs::read(path).unwrap());
    let writes = Cell::new(0);

    let error = install_persistent_with(
        paths.clone(),
        &relay,
        &[],
        None,
        UNIX_EPOCH,
        |path, bytes| {
            let write = writes.get() + 1;
            writes.set(write);
            if write == 3 {
                return Err("injected config write failure".into());
            }
            atomic_write(path, bytes)
        },
    )
    .unwrap_err()
    .to_string();

    assert!(error.contains("injected config write failure"), "{error}");
    for (index, path) in paths.all().iter().enumerate() {
        assert_eq!(
            std::fs::read(path).unwrap(),
            before[index],
            "{}",
            path.display()
        );
    }
}

#[cfg(unix)]
#[test]
fn install_rollback_restores_original_file_permissions() {
    use std::os::unix::fs::PermissionsExt;

    let temp = tempfile::tempdir().unwrap();
    let relay = relay_binary(temp.path());
    let paths = paths(&temp.path().join("hermes"));
    std::fs::create_dir_all(paths.config.parent().unwrap()).unwrap();
    let originals = [
        (&paths.config, b"model: original\n".as_slice(), 0o640),
        (
            &paths.allowlist,
            b"{\"approvals\":[{\"event\":\"x\",\"command\":\"custom\"}]}\n".as_slice(),
            0o644,
        ),
        (
            &paths.generation,
            b"original-generation\n".as_slice(),
            0o600,
        ),
    ];
    for (path, bytes, mode) in originals {
        std::fs::write(path, bytes).unwrap();
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).unwrap();
    }
    let expected_modes = paths
        .all()
        .map(|path| std::fs::metadata(path).unwrap().permissions().mode() & 0o777);
    let writes = Cell::new(0);

    install_persistent_with(
        paths.clone(),
        &relay,
        &[],
        None,
        UNIX_EPOCH,
        |path, bytes| {
            let write = writes.get() + 1;
            writes.set(write);
            if write == 3 {
                return Err("injected config write failure".into());
            }
            atomic_write(path, bytes)?;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
                .map_err(|error| error.to_string())
        },
    )
    .unwrap_err();

    for (index, path) in paths.all().iter().enumerate() {
        assert_eq!(
            std::fs::metadata(path).unwrap().permissions().mode() & 0o777,
            expected_modes[index],
            "{}",
            path.display()
        );
    }
}

#[test]
fn composed_install_rollback_restores_the_visible_preexisting_generation() {
    let temp = tempfile::tempdir().unwrap();
    let relay = relay_binary(temp.path());
    let paths = paths(&temp.path().join("hermes"));
    std::fs::create_dir_all(paths.config.parent().unwrap()).unwrap();
    install_persistent_with(paths.clone(), &relay, &[], None, UNIX_EPOCH, atomic_write).unwrap();
    let previous =
        crate::install_generation::InstallGeneration::capture(paths.generation.clone()).unwrap();
    let mut retirement = GenerationRetirement::acquire(&paths.generation)
        .unwrap()
        .unwrap();
    retirement.invalidate_for_replacement().unwrap();
    let writes = Cell::new(0);

    let result = install_persistent_with(
        paths.clone(),
        &relay,
        &[],
        None,
        UNIX_EPOCH,
        |path, bytes| {
            let write = writes.get() + 1;
            writes.set(write);
            if write == 3 {
                return Err("injected composed install failure".into());
            }
            atomic_write(path, bytes)
        },
    );
    let error = finish_generation_mutation(result, Some(&mut retirement), "install")
        .unwrap_err()
        .to_string();

    assert!(
        error.contains("injected composed install failure"),
        "{error}"
    );
    previous.verify_current().unwrap();
    crate::install_generation::InstallGeneration::capture(paths.generation).unwrap();
}

#[test]
fn install_rolls_back_after_post_write_verification_failure() {
    let temp = tempfile::tempdir().unwrap();
    let relay = relay_binary(temp.path());
    let paths = paths(&temp.path().join("hermes"));
    std::fs::create_dir_all(paths.config.parent().unwrap()).unwrap();
    std::fs::write(&paths.config, "model: original\n").unwrap();
    std::fs::write(&paths.allowlist, "{\"approvals\":[]}\n").unwrap();
    std::fs::write(&paths.generation, "old\n").unwrap();
    let before = paths.all().map(|path| std::fs::read(path).unwrap());
    let corrupted = Cell::new(false);

    let error = install_persistent_with(
        paths.clone(),
        &relay,
        &[],
        None,
        UNIX_EPOCH,
        |path, bytes| {
            if path == paths.config && !corrupted.replace(true) {
                return atomic_write(path, b"hooks: invalid-shape\n");
            }
            atomic_write(path, bytes)
        },
    )
    .unwrap_err()
    .to_string();

    assert!(
        error.contains("Hermes MCP server did not persist exactly"),
        "{error}"
    );
    for (index, path) in paths.all().iter().enumerate() {
        assert_eq!(
            std::fs::read(path).unwrap(),
            before[index],
            "{}",
            path.display()
        );
    }
}

#[test]
fn uninstall_removes_only_relay_owned_hermes_state() {
    let temp = tempfile::tempdir().unwrap();
    let relay = relay_binary(temp.path());
    let paths = paths(&temp.path().join("hermes"));
    std::fs::create_dir_all(paths.config.parent().unwrap()).unwrap();
    std::fs::write(
        &paths.config,
        "model: keep\nmcp_servers:\n  filesystem:\n    command: fs-mcp\nhooks:\n  custom_event:\n  - command: custom-hook\n",
    )
    .unwrap();
    std::fs::write(
        &paths.allowlist,
        "{\"owner\":\"user\",\"approvals\":[{\"event\":\"custom_event\",\"command\":\"custom-hook\"}]}\n",
    )
    .unwrap();
    install_persistent_with(paths.clone(), &relay, &[], None, UNIX_EPOCH, atomic_write).unwrap();

    let removed = uninstall_persistent_with(paths.clone(), atomic_write).unwrap();

    assert_eq!(removed, paths.all());
    assert!(!paths.generation.exists());
    let config = yaml(&paths.config);
    assert_eq!(config["model"], json!("keep"));
    assert_eq!(
        config["mcp_servers"]["filesystem"]["command"],
        json!("fs-mcp")
    );
    assert!(config["mcp_servers"].get(MCP_SERVER_NAME).is_none());
    assert_eq!(
        config["hooks"]["custom_event"][0]["command"],
        json!("custom-hook")
    );
    let allowlist = json_file(&paths.allowlist);
    assert_eq!(allowlist["owner"], json!("user"));
    assert_eq!(allowlist["approvals"].as_array().unwrap().len(), 1);
    assert_eq!(allowlist["approvals"][0]["command"], json!("custom-hook"));
}

#[test]
fn uninstall_rolls_back_every_file_when_commit_fails() {
    let temp = tempfile::tempdir().unwrap();
    let relay = relay_binary(temp.path());
    let paths = paths(&temp.path().join("hermes"));
    std::fs::create_dir_all(paths.config.parent().unwrap()).unwrap();
    std::fs::write(&paths.config, "model: keep\n").unwrap();
    std::fs::write(&paths.allowlist, "{\"owner\":\"keep\"}\n").unwrap();
    install_persistent_with(paths.clone(), &relay, &[], None, UNIX_EPOCH, atomic_write).unwrap();
    let before = paths.all().map(|path| std::fs::read(path).unwrap());
    let writes = Cell::new(0);

    let error = uninstall_persistent_with(paths.clone(), |path, bytes| {
        let write = writes.get() + 1;
        writes.set(write);
        if write == 2 {
            return Err("injected uninstall config failure".into());
        }
        atomic_write(path, bytes)
    })
    .unwrap_err()
    .to_string();

    assert!(
        error.contains("injected uninstall config failure"),
        "{error}"
    );
    for (index, path) in paths.all().iter().enumerate() {
        assert_eq!(
            std::fs::read(path).unwrap(),
            before[index],
            "{}",
            path.display()
        );
    }
}

#[test]
fn composed_uninstall_rollback_restores_the_visible_preexisting_generation() {
    let temp = tempfile::tempdir().unwrap();
    let relay = relay_binary(temp.path());
    let paths = paths(&temp.path().join("hermes"));
    std::fs::create_dir_all(paths.config.parent().unwrap()).unwrap();
    std::fs::write(&paths.config, "model: keep\n").unwrap();
    std::fs::write(&paths.allowlist, "{\"owner\":\"keep\"}\n").unwrap();
    install_persistent_with(paths.clone(), &relay, &[], None, UNIX_EPOCH, atomic_write).unwrap();
    let previous =
        crate::install_generation::InstallGeneration::capture(paths.generation.clone()).unwrap();
    let mut retirement = GenerationRetirement::acquire(&paths.generation)
        .unwrap()
        .unwrap();
    retirement.invalidate_for_replacement().unwrap();
    let writes = Cell::new(0);

    let result = uninstall_persistent_with(paths.clone(), |path, bytes| {
        let write = writes.get() + 1;
        writes.set(write);
        if write == 2 {
            return Err("injected composed uninstall failure".into());
        }
        atomic_write(path, bytes)
    });
    let error = finish_generation_mutation(result, Some(&mut retirement), "uninstall")
        .unwrap_err()
        .to_string();

    assert!(
        error.contains("injected composed uninstall failure"),
        "{error}"
    );
    previous.verify_current().unwrap();
    crate::install_generation::InstallGeneration::capture(paths.generation).unwrap();
}

#[test]
fn uninstall_noops_without_creating_a_hermes_home() {
    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("missing-hermes-home");
    let config = home.join("config.yaml");

    assert!(uninstall_persistent(&config).unwrap().is_empty());
    assert!(!home.exists());
}

#[test]
fn unrelated_hermes_files_are_not_owned_or_rewritten_by_uninstall() {
    let temp = tempfile::tempdir().unwrap();
    let paths = paths(&temp.path().join("hermes"));
    std::fs::create_dir_all(paths.config.parent().unwrap()).unwrap();
    let config = b"# preserve this exact formatting\nmodel: custom\nmcp_servers:\n  nemo-relay:\n    command: foreign-mcp\n    args: [serve]\n";
    let allowlist = b"{ \"approvals\": [{\"event\":\"custom\",\"command\":\"custom-hook\"}] }\n";
    std::fs::write(&paths.config, config).unwrap();
    std::fs::write(&paths.allowlist, allowlist).unwrap();

    assert!(!persistent_state_exists(&paths.config));
    assert!(uninstall_persistent(&paths.config).unwrap().is_empty());
    assert_eq!(std::fs::read(&paths.config).unwrap(), config);
    assert_eq!(std::fs::read(&paths.allowlist).unwrap(), allowlist);
    assert!(!paths.generation.exists());
}

#[test]
fn persistent_state_detection_recognizes_each_relay_owned_surface() {
    let temp = tempfile::tempdir().unwrap();
    let relay = relay_binary(temp.path());
    let roots = ["generation", "mcp", "hook", "approval"].map(|name| {
        let paths = paths(&temp.path().join(name));
        std::fs::create_dir_all(paths.config.parent().unwrap()).unwrap();
        paths
    });

    std::fs::write(&roots[0].generation, "active\n").unwrap();
    std::fs::write(
        &roots[1].config,
        serde_yaml::to_string(&json!({
            "mcp_servers": {MCP_SERVER_NAME: expected_mcp_server(
                &relay,
                &roots[1].generation,
                TEST_GENERATION_TOKEN,
                &[]
            )}
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        &roots[2].config,
        serde_yaml::to_string(&json!({
            "hooks": {
                "on_session_start": [{"command": persistent_hook_command(
                    &relay,
                    &roots[2].generation,
                    TEST_GENERATION_TOKEN
                ).unwrap()}]
            }
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        &roots[3].allowlist,
        serde_json::to_vec(&json!({
            "approvals": [{
                "event": "on_session_start",
                "command": persistent_hook_command(
                    &relay,
                    &roots[3].generation,
                    TEST_GENERATION_TOKEN
                ).unwrap()
            }]
        }))
        .unwrap(),
    )
    .unwrap();

    for paths in roots {
        assert!(
            persistent_state_exists(&paths.config),
            "managed state at {} was not detected",
            paths.config.display()
        );
    }
}

#[test]
fn transparent_config_suppresses_only_the_managed_mcp_and_uses_one_relay_hook() {
    let temp = tempfile::tempdir().unwrap();
    let relay = relay_binary(temp.path());
    let command = crate::installer::transparent_hook_forward_command(
        &relay,
        CodingAgent::Hermes,
        "http://127.0.0.1:1234",
    )
    .unwrap();
    let existing = format!(
        r#"
mcp_servers:
  nemo-relay:
    command: {relay}
    args: [mcp, --agent, hermes]
  filesystem:
    command: fs-mcp
hooks:
  on_session_start:
    - command: nemo-relay hook-forward hermes
    - command: custom-hook
"#,
        relay = relay.display()
    );
    let patched: Value = serde_yaml::from_str(
        &transparent_config(&existing, &relay, "http://127.0.0.1:1234").unwrap(),
    )
    .unwrap();

    assert!(patched["mcp_servers"].get(MCP_SERVER_NAME).is_none());
    assert_eq!(
        patched["mcp_servers"]["filesystem"]["command"],
        json!("fs-mcp")
    );
    for event in CodingAgent::Hermes.hook_events() {
        let groups = patched["hooks"][event].as_array().unwrap();
        assert_eq!(
            groups
                .iter()
                .filter_map(|group| group.get("command").and_then(Value::as_str))
                .filter(|candidate| is_managed_hook_command(candidate))
                .count(),
            1,
            "event {event}"
        );
        assert!(
            groups
                .iter()
                .any(|group| group["command"] == json!(command))
        );
    }
    assert!(
        patched["hooks"]["on_session_start"]
            .as_array()
            .unwrap()
            .iter()
            .any(|group| group["command"] == json!("custom-hook"))
    );
}

#[test]
fn malformed_user_files_fail_before_any_state_is_replaced() {
    let temp = tempfile::tempdir().unwrap();
    let relay = relay_binary(temp.path());
    let paths = paths(&temp.path().join("hermes"));
    std::fs::create_dir_all(paths.config.parent().unwrap()).unwrap();
    std::fs::write(&paths.config, "hooks: [not-an-object]\n").unwrap();
    std::fs::write(&paths.allowlist, "{\"approvals\":[]}").unwrap();
    std::fs::write(&paths.generation, "old\n").unwrap();
    let before = paths.all().map(|path| std::fs::read(path).unwrap());

    assert!(
        install_persistent_with(paths.clone(), &relay, &[], None, UNIX_EPOCH, atomic_write,)
            .is_err()
    );
    for (index, path) in paths.all().iter().enumerate() {
        assert_eq!(std::fs::read(path).unwrap(), before[index]);
    }
}

#[test]
fn hermes_entrypoints_reject_missing_or_foreign_relay_binaries() {
    let temp = tempfile::tempdir().unwrap();
    let config_path = temp.path().join("hermes/config.yaml");
    let missing_relay = temp.path().join("missing/nemo-relay");

    let error = install_persistent(&config_path, &missing_relay)
        .unwrap_err()
        .to_string();
    assert!(error.contains("missing or not executable"), "{error}");

    std::fs::create_dir_all(config_path.parent().unwrap()).unwrap();
    std::fs::write(
        &config_path,
        format!(
            "mcp_servers:\n  {MCP_SERVER_NAME}:\n    command: {}\n    args: [mcp]\n",
            missing_relay.display()
        ),
    )
    .unwrap();
    let error = configured_relay_executable(&config_path).unwrap_err();
    assert!(error.contains("missing or not executable"), "{error}");

    let foreign = json!({
        "mcp_servers": {
            MCP_SERVER_NAME: {
                "command": "foreign-mcp",
                "args": ["serve"]
            }
        }
    });
    let error = relay_executable_from_config(&foreign).unwrap_err();
    assert!(error.contains("not a managed Relay MCP client"), "{error}");
}

#[test]
fn hermes_diagnosis_validates_binary_bind_generation_and_environment() {
    let temp = tempfile::tempdir().unwrap();
    let _config_home = XdgConfigHomeScope::enter(&temp.path().join("xdg"));
    let relay = relay_binary(temp.path());
    let paths = paths(&temp.path().join("hermes"));
    install_persistent_with(paths.clone(), &relay, &[], None, UNIX_EPOCH, atomic_write).unwrap();

    let original = yaml(&paths.config);
    std::fs::remove_file(&relay).unwrap();
    let error = diagnose_persistent(&paths.config).unwrap_err();
    assert!(error.contains("missing or not executable"), "{error}");

    let relay = relay_binary(temp.path());
    let mut wrong_bind = original.clone();
    wrong_bind["mcp_servers"][MCP_SERVER_NAME]["env"]["NEMO_RELAY_GATEWAY_BIND"] =
        json!("127.0.0.1:1");
    std::fs::write(&paths.config, serde_yaml::to_string(&wrong_bind).unwrap()).unwrap();
    let error = diagnose_persistent(&paths.config).unwrap_err();
    assert!(error.contains("shared gateway bind"), "{error}");

    let mut wrong_generation = original.clone();
    wrong_generation["mcp_servers"][MCP_SERVER_NAME]["env"][GENERATION_FILE_ENV] =
        json!(temp.path().join("wrong-generation").display().to_string());
    std::fs::write(
        &paths.config,
        serde_yaml::to_string(&wrong_generation).unwrap(),
    )
    .unwrap();
    let error = diagnose_persistent(&paths.config).unwrap_err();
    assert!(error.contains("points at the wrong file"), "{error}");

    let mut missing_environment = original;
    assert!(
        missing_environment["mcp_servers"][MCP_SERVER_NAME]["env"]
            .as_object_mut()
            .unwrap()
            .remove("OPENAI_API_KEY")
            .is_some()
    );
    std::fs::write(
        &paths.config,
        serde_yaml::to_string(&missing_environment).unwrap(),
    )
    .unwrap();
    let error = diagnose_persistent(&paths.config).unwrap_err();
    assert!(error.contains("missing environment names"), "{error}");
    assert!(error.contains("OPENAI_API_KEY"), "{error}");
    assert!(error.contains("install hermes --force"), "{error}");

    assert!(relay.exists());
}

#[test]
fn hermes_generation_finish_preserves_primary_errors_and_reports_restore_failures() {
    let primary = CliError::Install("primary failure".into());
    let error = finish_generation_mutation::<()>(Err(primary), None, "install")
        .unwrap_err()
        .to_string();
    assert!(error.contains("primary failure"), "{error}");

    let temp = tempfile::tempdir().unwrap();
    let generation = temp.path().join(GENERATION_FILE_NAME);
    crate::install_generation::write_new_generation(&generation).unwrap();
    let mut retirement = GenerationRetirement::acquire(&generation).unwrap().unwrap();
    retirement.invalidate_for_replacement().unwrap();
    std::fs::write(&generation, "foreign-generation\n").unwrap();

    let error = finish_generation_mutation::<()>(
        Err(CliError::Install("mutation failed".into())),
        Some(&mut retirement),
        "install",
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("mutation failed"), "{error}");
    assert!(error.contains("additionally failed to restore"), "{error}");
}

#[test]
fn hermes_uninstall_and_verification_reject_malformed_or_residual_state() {
    let temp = tempfile::tempdir().unwrap();
    let relay = relay_binary(temp.path());
    let hermes_paths = paths(&temp.path().join("hermes"));
    install_persistent_with(
        hermes_paths.clone(),
        &relay,
        &[],
        None,
        UNIX_EPOCH,
        atomic_write,
    )
    .unwrap();

    let config = yaml(&hermes_paths.config);
    let command = config["hooks"]["on_session_start"][0]["command"]
        .as_str()
        .unwrap()
        .to_string();
    let token = InstallGeneration::capture(hermes_paths.generation.clone())
        .unwrap()
        .token()
        .to_owned();
    let expected_environment = forwarded_environment_names(&[], None);

    let mut duplicate_hook = config.clone();
    duplicate_hook["hooks"]["on_session_start"]
        .as_array_mut()
        .unwrap()
        .push(json!({"command": command}));
    let error = verify_hook_definitions(&duplicate_hook, &command).unwrap_err();
    assert!(
        error.contains("exactly one trusted Relay handler"),
        "{error}"
    );

    let mut harmless_missing_command = config.clone();
    harmless_missing_command["hooks"]
        .as_object_mut()
        .unwrap()
        .insert("custom".into(), json!([{"timeout": 1}]));
    verify_hook_definitions(&harmless_missing_command, &command).unwrap();

    verify_install(
        &hermes_paths,
        &relay,
        &command,
        &expected_environment,
        &token,
        None,
    )
    .unwrap();

    install_persistent_with(
        hermes_paths.clone(),
        &relay,
        &expected_environment,
        None,
        UNIX_EPOCH,
        atomic_write,
    )
    .unwrap();
    let config = yaml(&hermes_paths.config);
    let command = config["hooks"]["on_session_start"][0]["command"]
        .as_str()
        .unwrap()
        .to_string();
    let expected_token = InstallGeneration::capture(hermes_paths.generation.clone())
        .unwrap()
        .token()
        .to_owned();
    crate::install_generation::write_new_generation(&hermes_paths.generation).unwrap();
    let error = verify_install(
        &hermes_paths,
        &relay,
        &command,
        &expected_environment,
        &expected_token,
        None,
    )
    .unwrap_err();
    assert!(
        error.contains("generation did not persist exactly"),
        "{error}"
    );

    let malformed_paths = paths(&temp.path().join("malformed"));
    std::fs::create_dir_all(malformed_paths.config.parent().unwrap()).unwrap();
    std::fs::write(&malformed_paths.allowlist, r#"{"approvals":{}}"#).unwrap();
    let error = uninstall_persistent_with(malformed_paths, atomic_write)
        .unwrap_err()
        .to_string();
    assert!(error.contains("approvals must be an array"), "{error}");
}

#[test]
fn hermes_uninstall_verifier_identifies_each_residual_owned_surface() {
    let temp = tempfile::tempdir().unwrap();
    let relay = relay_binary(temp.path());
    let paths = paths(&temp.path().join("hermes"));
    install_persistent_with(paths.clone(), &relay, &[], None, UNIX_EPOCH, atomic_write).unwrap();

    let error = verify_uninstall(&paths).unwrap_err();
    assert!(error.contains("generation fence still exists"), "{error}");

    std::fs::remove_file(&paths.generation).unwrap();
    let error = verify_uninstall(&paths).unwrap_err();
    assert!(
        error.contains("managed Hermes Relay config still exists"),
        "{error}"
    );

    std::fs::remove_file(&paths.config).unwrap();
    let error = verify_uninstall(&paths).unwrap_err();
    assert!(
        error.contains("managed Hermes Relay trust approval still exists"),
        "{error}"
    );
}

#[test]
fn hermes_file_helpers_report_path_lock_read_remove_and_restore_failures() {
    let temp = tempfile::tempdir().unwrap();

    let error = PersistentPaths::for_config(PathBuf::from("/"))
        .unwrap_err()
        .to_string();
    assert!(error.contains("has no parent directory"), "{error}");
    let error = acquire_install_lock(Path::new("/"), Duration::ZERO).unwrap_err();
    assert!(error.contains("has no parent directory"), "{error}");

    let parent_file = temp.path().join("parent-file");
    std::fs::write(&parent_file, "file").unwrap();
    let error = acquire_allowlist_lock(&parent_file.join("allowlist"), Duration::ZERO).unwrap_err();
    assert!(error.contains("failed to create"), "{error}");
    let error =
        acquire_allowlist_lock(&parent_file.join("nested/allowlist"), Duration::ZERO).unwrap_err();
    assert!(error.contains("failed to create"), "{error}");

    let allowlist = temp.path().join("allowlist.json");
    let lock_dir = temp.path().join("allowlist.json.lock");
    std::fs::create_dir(&lock_dir).unwrap();
    let error = acquire_allowlist_lock(&allowlist, Duration::ZERO).unwrap_err();
    assert!(
        error.contains("failed to open Hermes install lock"),
        "{error}"
    );

    let held_config = temp.path().join("held/config.yaml");
    let _held = acquire_install_lock(&held_config, Duration::ZERO).unwrap();
    let error = acquire_install_lock(&held_config, Duration::from_millis(30)).unwrap_err();
    assert!(error.contains("timed out waiting"), "{error}");

    let directory = temp.path().join("directory");
    std::fs::create_dir(&directory).unwrap();
    let error = read_optional_utf8(&directory).unwrap_err().to_string();
    assert!(error.contains("failed to read"), "{error}");
    let error = match FileSnapshot::capture(&directory) {
        Ok(_) => panic!("directory snapshot unexpectedly succeeded"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("failed to snapshot"), "{error}");
    let error = remove_optional_file(&directory).unwrap_err();
    assert!(error.contains("failed to remove"), "{error}");
    remove_optional_file(&temp.path().join("missing")).unwrap();

    let restored = temp.path().join("restored");
    std::fs::write(&restored, "original").unwrap();
    let snapshot = FileSnapshot::capture(&restored).unwrap();
    std::fs::remove_file(&restored).unwrap();
    let error = snapshot.restore(&mut |_path, _bytes| Ok(())).unwrap_err();
    assert!(error.contains("failed to restore permissions"), "{error}");

    let absent = temp.path().join("absent");
    let snapshot = FileSnapshot::capture(&absent).unwrap();
    std::fs::write(&absent, "transient").unwrap();
    snapshot.restore(&mut atomic_write).unwrap();
    assert!(!absent.exists());
}

#[test]
fn hermes_rollback_reports_both_primary_and_snapshot_restore_errors() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("state");
    std::fs::write(&path, "original").unwrap();
    let snapshot = FileSnapshot::capture(&path).unwrap();
    let error = rollback_error::<(), _>(
        "install",
        "primary failure".into(),
        &[snapshot],
        &mut |_path, _bytes| Err("restore failure".into()),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("primary failure"), "{error}");
    assert!(
        error.contains("rollback also failed: restore failure"),
        "{error}"
    );
}

#[test]
fn hermes_uninstall_removes_an_allowlist_containing_only_managed_approvals() {
    let temp = tempfile::tempdir().unwrap();
    let paths = paths(&temp.path().join("hermes"));
    std::fs::create_dir_all(paths.allowlist.parent().unwrap()).unwrap();
    std::fs::write(
        &paths.allowlist,
        serde_json::to_vec(&json!({
            "approvals": [{
                "event": "on_session_start",
                "command": "nemo-relay hook-forward hermes"
            }]
        }))
        .unwrap(),
    )
    .unwrap();

    let affected = uninstall_persistent_with(paths.clone(), atomic_write).unwrap();

    assert_eq!(affected, vec![paths.allowlist.clone()]);
    assert!(!paths.allowlist.exists());
}
