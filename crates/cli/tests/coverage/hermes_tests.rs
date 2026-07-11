// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cell::Cell;
use std::path::Path;
use std::time::{Duration, UNIX_EPOCH};

use serde_json::{Value, json};

use super::*;

fn relay_binary(root: &Path) -> PathBuf {
    let path = root.join("NeMo Relay's bin").join("nemo-relay");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, b"relay").unwrap();
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
    assert_eq!(
        persistent_hook_command_for_platform(relay, false),
        "'/tmp/NeMo $Relay`test'\\''/bin/nemo-relay' plugin-shim hook hermes"
    );
    assert_eq!(
        persistent_hook_command_for_platform(
            Path::new(r"C:\Program Files\NeMo 100%\bin\nemo-relay.exe"),
            true,
        ),
        r#""C:\Program Files\NeMo 100%%\bin\nemo-relay.exe" plugin-shim hook hermes"#
    );
    for command in [
        "nemo-relay hook-forward hermes",
        "/old/path/nemo-relay plugin-shim hook hermes",
        "'/old/plugin-shim hook hermes/nemo-relay' plugin-shim hook hermes",
        "'/old install/nemo-relay' plugin-shim hook hermes --gateway-url http://127.0.0.1:47632",
        r#""C:\Program Files\NeMo 100%%\nemo-relay.exe" plugin-shim hook hermes"#,
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
    ] {
        assert!(!is_managed_hook_command(command), "overmatched: {command}");
    }
}

#[test]
fn forwarded_environment_is_minimal_and_includes_explicit_config_references() {
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
    assert!(!names.contains(&"NEMO_RELAY_WORKER_TOKEN".into()));
    assert!(!names.contains(&"UNRELATED_SECRET".into()));
    assert!(!names.contains(&"OTEL_EXPORTER_OTLP_ENDPOINT".into()));
}

#[test]
fn persistent_config_migrates_owned_state_and_preserves_unrelated_config() {
    let temp = tempfile::tempdir().unwrap();
    let relay = relay_binary(temp.path());
    let generation = temp.path().join(GENERATION_FILE_NAME);
    let command = persistent_hook_command(&relay);
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
        expected_mcp_server(&relay, &generation, &["AWS_REGION".into()])
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
    for event in HERMES_HOOK_EVENTS {
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
fn trusted_hooks_migrates_only_relay_approvals_and_records_every_event() {
    let temp = tempfile::tempdir().unwrap();
    let relay = relay_binary(temp.path());
    let command = persistent_hook_command(&relay);
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
    assert_eq!(approvals.len(), HERMES_HOOK_EVENTS.len() + 1);
    for event in HERMES_HOOK_EVENTS {
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
    let first_generation = std::fs::read_to_string(&paths.generation).unwrap();

    install_persistent_with(paths.clone(), &relay, &environment, None, now, atomic_write).unwrap();
    let second_generation = std::fs::read_to_string(&paths.generation).unwrap();
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
        config["mcp_servers"][MCP_SERVER_NAME]["env"]["OTEL_SERVICE_NAME"],
        json!("${OTEL_SERVICE_NAME}")
    );
    assert_eq!(
        json_file(&paths.allowlist)["approvals"]
            .as_array()
            .unwrap()
            .len(),
        HERMES_HOOK_EVENTS.len()
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
fn uninstall_noops_without_creating_a_hermes_home() {
    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("missing-hermes-home");
    let config = home.join("config.yaml");

    assert!(uninstall_persistent(&config).unwrap().is_empty());
    assert!(!home.exists());
}

#[test]
fn transparent_config_suppresses_only_the_managed_mcp_and_uses_one_relay_hook() {
    let temp = tempfile::tempdir().unwrap();
    let relay = relay_binary(temp.path());
    let command = persistent_hook_command(&relay);
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
    let patched: Value =
        serde_yaml::from_str(&transparent_config(&existing, &relay).unwrap()).unwrap();

    assert!(patched["mcp_servers"].get(MCP_SERVER_NAME).is_none());
    assert_eq!(
        patched["mcp_servers"]["filesystem"]["command"],
        json!("fs-mcp")
    );
    for event in HERMES_HOOK_EVENTS {
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
