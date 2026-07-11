// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeSet, VecDeque};
use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use tempfile::tempdir;
use toml_edit::{DocumentMut, Item};

use super::*;

#[derive(Default)]
struct FakeCodexHooksClient {
    hook_lists: VecDeque<Result<Vec<CodexHookMetadata>, String>>,
    trusted: Vec<Vec<String>>,
    cleared: Vec<Vec<String>>,
    restored: Vec<Vec<(String, Option<Value>)>>,
    clear_config_path: Option<PathBuf>,
    trust_error: Option<String>,
    clear_error: Option<String>,
    restore_error: Option<String>,
}

impl CodexHooksClient for FakeCodexHooksClient {
    fn list_hooks(&mut self, _cwd: &std::path::Path) -> Result<Vec<CodexHookMetadata>, String> {
        self.hook_lists
            .pop_front()
            .unwrap_or_else(|| Err("unexpected hooks/list call".into()))
    }

    fn trust_hooks(&mut self, hooks: &[CodexHookMetadata]) -> Result<(), String> {
        self.trusted
            .push(hooks.iter().map(|hook| hook.key.clone()).collect());
        match self.trust_error.take() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn clear_hook_trust(&mut self, keys: &[String]) -> Result<(), String> {
        self.cleared.push(keys.to_vec());
        if let Some(error) = self.clear_error.take() {
            return Err(error);
        }
        if let Some(path) = &self.clear_config_path {
            let raw = fs::read_to_string(path)
                .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
            let mut config = raw
                .parse::<DocumentMut>()
                .map_err(|error| format!("invalid TOML in {}: {error}", path.display()))?;
            if let Some(state) = config
                .get_mut("hooks")
                .and_then(Item::as_table_mut)
                .and_then(|hooks| hooks.get_mut("state"))
                .and_then(Item::as_table_mut)
            {
                for key in keys {
                    state.remove(key);
                }
            }
            fs::write(path, config.to_string())
                .map_err(|error| format!("failed to write {}: {error}", path.display()))?;
        }
        Ok(())
    }

    fn restore_hook_trust(&mut self, state: &[(String, Option<Value>)]) -> Result<(), String> {
        self.restored.push(state.to_vec());
        match self.restore_error.take() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

fn expected_plugin_command() -> String {
    expected_plugin_hook_command().unwrap()
}

fn empty_codex_hooks_client() -> FakeCodexHooksClient {
    FakeCodexHooksClient {
        hook_lists: VecDeque::from([Ok(Vec::new())]),
        ..FakeCodexHooksClient::default()
    }
}

fn write_plugin_hooks(plugin_root: &Path) -> PathBuf {
    let path = plugin_root.join("hooks").join("hooks.json");
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(
        &path,
        serde_json::to_vec_pretty(&generated_hooks(
            CodingAgent::Codex,
            &expected_plugin_command(),
        ))
        .unwrap(),
    )
    .unwrap();
    path
}

fn codex_hook_metadata(
    hooks_path: &std::path::Path,
    event_name: &str,
    key: &str,
    trust_status: &str,
    enabled: bool,
) -> CodexHookMetadata {
    let hooks_path = if hooks_path.is_dir() {
        hooks_path.join("hooks.json")
    } else {
        hooks_path.to_path_buf()
    };
    if !hooks_path.exists() {
        fs::create_dir_all(hooks_path.parent().unwrap()).unwrap();
        fs::write(
            &hooks_path,
            serde_json::to_vec_pretty(&generated_hooks(
                CodingAgent::Codex,
                &expected_plugin_command(),
            ))
            .unwrap(),
        )
        .unwrap();
    }
    CodexHookMetadata {
        key: key.into(),
        event_name: event_name.into(),
        handler_type: "command".into(),
        command: Some(expected_plugin_command()),
        source_path: hooks_path.display().to_string(),
        source: "plugin".into(),
        plugin_id: Some(CODEX_PLUGIN_ID.into()),
        enabled,
        current_hash: format!("sha256:{key}"),
        trust_status: trust_status.into(),
    }
}

fn required_codex_hook_metadata(
    hooks_path: &std::path::Path,
    trust_status: &str,
    enabled: bool,
) -> Vec<CodexHookMetadata> {
    generated_codex_hook_metadata(hooks_path, trust_status, enabled)
}

fn generated_codex_hook_metadata(
    hooks_path: &std::path::Path,
    trust_status: &str,
    enabled: bool,
) -> Vec<CodexHookMetadata> {
    [
        "session_start",
        "user_prompt_submit",
        "pre_tool_use",
        "post_tool_use",
        "permission_request",
        "subagent_start",
        "subagent_stop",
        "stop",
        "pre_compact",
        "post_compact",
    ]
    .into_iter()
    .enumerate()
    .map(|(index, event)| {
        codex_hook_metadata(
            hooks_path,
            event,
            &format!("relay-hook-{index}"),
            trust_status,
            enabled,
        )
    })
    .collect()
}

fn persisted_relay_hook_key(event: &str, index: usize) -> String {
    format!("{CODEX_PLUGIN_HOOK_KEY_PREFIX}{event}:0:{index}")
}

fn persisted_relay_hook_metadata(hooks_path: &Path, trust_status: &str) -> Vec<CodexHookMetadata> {
    let mut hooks = generated_codex_hook_metadata(hooks_path, trust_status, true)
        .into_iter()
        .enumerate()
        .map(|(index, mut hook)| {
            hook.key = persisted_relay_hook_key(&hook.event_name, index);
            hook
        })
        .collect::<Vec<_>>();
    hooks.extend(
        ["post_tool_use_failure", "notification", "session_end"]
            .into_iter()
            .enumerate()
            .map(|(offset, event)| {
                codex_hook_metadata(
                    hooks_path,
                    event,
                    &persisted_relay_hook_key(event, 10 + offset),
                    trust_status,
                    true,
                )
            }),
    );
    hooks
}

fn write_persisted_hook_trust(config_path: &Path, keys: &[String], unrelated_key: &str) {
    let mut raw = "model_provider = \"openai\"\n".to_string();
    for key in keys {
        raw.push_str(&format!(
            "\n[hooks.state.{key:?}]\ntrusted_hash = {hash:?}\nenabled = true\n",
            hash = format!("sha256:{key}")
        ));
    }
    raw.push_str(&format!(
        "\n[hooks.state.{unrelated_key:?}]\ntrusted_hash = {hash:?}\nenabled = true\n",
        hash = format!("sha256:{unrelated_key}")
    ));
    fs::write(config_path, raw).unwrap();
}

#[cfg(not(windows))]
fn fake_codex_app_server(
    dir: &std::path::Path,
    hooks: &[CodexHookMetadata],
) -> (EnvVarGuard, EnvVarGuard, EnvVarGuard) {
    use std::os::unix::fs::PermissionsExt;

    let bin_dir = dir.join("fake-codex-bin");
    fs::create_dir_all(&bin_dir).unwrap();
    let codex = bin_dir.join("codex");
    fs::write(
        &codex,
        r#"#!/bin/sh
while IFS= read -r line; do
  printf '%s\n' "$line" >> "$NEMO_RELAY_TEST_CODEX_LOG"
  id=$(printf '%s\n' "$line" | sed -E 's/.*"id":([0-9]+).*/\1/')
  case "$line" in
    *'"method":"initialize"'*)
      printf '{"id":%s,"result":{}}\n' "$id"
      ;;
    *'"method":"hooks/list"'*)
      printf '{"id":%s,"result":{"data":[{"cwd":"/tmp","hooks":%s,"warnings":[],"errors":[]}]}}\n' "$id" "$NEMO_RELAY_TEST_CODEX_HOOKS"
      ;;
    *'"method":"config/batchWrite"'*)
      printf '{"id":%s,"result":{}}\n' "$id"
      ;;
  esac
done
"#,
    )
    .unwrap();
    let mut permissions = fs::metadata(&codex).unwrap().permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&codex, permissions).unwrap();
    let existing_path = std::env::var_os("PATH").unwrap_or_default();
    let mut paths = vec![bin_dir];
    paths.extend(std::env::split_paths(&existing_path));
    let path = std::env::join_paths(paths).unwrap();
    let log_path = dir.join("fake-codex-requests.jsonl");
    fs::write(&log_path, "").unwrap();
    (
        EnvVarGuard::set_value("PATH", &path.to_string_lossy()),
        EnvVarGuard::set_value(
            "NEMO_RELAY_TEST_CODEX_HOOKS",
            &serde_json::to_string(hooks).unwrap(),
        ),
        EnvVarGuard::set_value("NEMO_RELAY_TEST_CODEX_LOG", &log_path.to_string_lossy()),
    )
}

fn read_http_request(stream: &mut std::net::TcpStream) -> Vec<u8> {
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    let mut request = Vec::new();
    let mut buffer = [0_u8; 1024];
    loop {
        match stream.read(&mut buffer) {
            Ok(0) => break,
            Ok(count) => {
                request.extend_from_slice(&buffer[..count]);
                if http_request_body_complete(&request) {
                    break;
                }
            }
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) =>
            {
                break;
            }
            Err(error) => panic!("failed to read local HTTP request: {error}"),
        }
    }
    request
}

fn http_request_body_complete(request: &[u8]) -> bool {
    let Some(header_end) = request.windows(4).position(|window| window == b"\r\n\r\n") else {
        return false;
    };
    let body_start = header_end + 4;
    let headers = String::from_utf8_lossy(&request[..body_start]);
    let content_length = headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap_or(0);
    request.len() >= body_start + content_length
}

fn home_env_lock() -> &'static Mutex<()> {
    &crate::test_support::ENV_TEST_LOCK
}

struct HomeScope<'a> {
    _guard: std::sync::MutexGuard<'a, ()>,
    prev_home: Option<std::ffi::OsString>,
    prev_userprofile: Option<std::ffi::OsString>,
    prev_codex_home: Option<std::ffi::OsString>,
}

impl<'a> HomeScope<'a> {
    fn enter(path: &std::path::Path) -> Self {
        let guard = home_env_lock()
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let prev_home = std::env::var_os("HOME");
        let prev_userprofile = std::env::var_os("USERPROFILE");
        let prev_codex_home = std::env::var_os("CODEX_HOME");
        // SAFETY: This test holds a process-wide mutex for the lifetime of the env override.
        unsafe {
            std::env::set_var("HOME", path);
            std::env::remove_var("USERPROFILE");
            std::env::remove_var("CODEX_HOME");
        }
        Self {
            _guard: guard,
            prev_home,
            prev_userprofile,
            prev_codex_home,
        }
    }
}

impl<'a> Drop for HomeScope<'a> {
    fn drop(&mut self) {
        // SAFETY: This restores the process environment while the mutex is still held.
        unsafe {
            match self.prev_home.take() {
                Some(value) => std::env::set_var("HOME", value),
                None => std::env::remove_var("HOME"),
            }
            match self.prev_userprofile.take() {
                Some(value) => std::env::set_var("USERPROFILE", value),
                None => std::env::remove_var("USERPROFILE"),
            }
            match self.prev_codex_home.take() {
                Some(value) => std::env::set_var("CODEX_HOME", value),
                None => std::env::remove_var("CODEX_HOME"),
            }
        }
    }
}

struct EnvVarGuard {
    key: &'static str,
    previous: Option<std::ffi::OsString>,
}

impl EnvVarGuard {
    fn set_path(key: &'static str, value: &std::path::Path) -> Self {
        let previous = std::env::var_os(key);
        // SAFETY: Callers hold the process-wide environment mutex through HomeScope.
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key, previous }
    }

    fn set_value(key: &'static str, value: &str) -> Self {
        let previous = std::env::var_os(key);
        // SAFETY: Callers hold the process-wide environment mutex through HomeScope.
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key, previous }
    }

    fn remove(key: &'static str) -> Self {
        let previous = std::env::var_os(key);
        // SAFETY: Callers hold the process-wide environment mutex through HomeScope.
        unsafe {
            std::env::remove_var(key);
        }
        Self { key, previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        // SAFETY: This restores the process environment while HomeScope still holds the mutex.
        unsafe {
            match self.previous.take() {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }
}

#[test]
fn backup_preserves_first_snapshot() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("config.toml");
    fs::write(&path, "model_provider = \"openai\"\n").unwrap();

    backup(&path).unwrap();
    fs::write(&path, "model_provider = \"nemo-relay-openai\"\n").unwrap();
    backup(&path).unwrap();

    assert_eq!(
        fs::read_to_string(backup_path(&path)).unwrap(),
        "model_provider = \"openai\"\n"
    );
}

#[test]
fn atomic_write_replaces_existing_destination() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("config.toml");
    fs::write(&path, "old\n").unwrap();

    atomic_write(&path, b"new\n").unwrap();

    assert_eq!(fs::read_to_string(&path).unwrap(), "new\n");
}

#[test]
fn codex_auto_trusts_only_exact_generated_plugin_hooks_and_verifies_them() {
    let dir = tempdir().unwrap();
    let hooks_path = dir.path().join("plugin").join("hooks.json");
    let config_path = dir.path().join(".codex").join("config.toml");
    fs::create_dir_all(config_path.parent().unwrap()).unwrap();
    fs::write(&config_path, "model = \"test\"\n").unwrap();
    let initial = generated_codex_hook_metadata(&hooks_path, "untrusted", true);
    let verified = generated_codex_hook_metadata(&hooks_path, "trusted", true);
    let mut decoys = Vec::new();
    let mut wrong_command = codex_hook_metadata(
        &hooks_path,
        "session_start",
        "wrong-command",
        "untrusted",
        true,
    );
    wrong_command.command = Some("custom hook".into());
    decoys.push(wrong_command);
    let mut wrong_source = codex_hook_metadata(
        &hooks_path,
        "session_start",
        "wrong-source",
        "untrusted",
        true,
    );
    wrong_source.source = "project".into();
    decoys.push(wrong_source);
    let mut wrong_plugin = codex_hook_metadata(
        &hooks_path,
        "session_start",
        "wrong-plugin",
        "untrusted",
        true,
    );
    wrong_plugin.plugin_id = Some("another-plugin@example".into());
    decoys.push(wrong_plugin);
    let mut client = FakeCodexHooksClient {
        hook_lists: VecDeque::from([
            Ok(initial.iter().cloned().chain(decoys).collect()),
            Ok(verified),
        ]),
        ..FakeCodexHooksClient::default()
    };

    auto_trust_codex_hooks(
        &mut client,
        dir.path(),
        &config_path,
        &expected_plugin_command(),
    )
    .unwrap();

    assert_eq!(
        client.trusted,
        vec![
            (0..10)
                .map(|index| format!("relay-hook-{index}"))
                .collect::<Vec<_>>()
        ]
    );
}

#[test]
fn codex_auto_trust_refuses_missing_required_hook_without_writing_state() {
    let dir = tempdir().unwrap();
    let hooks_path = dir.path().join("plugin").join("hooks.json");
    let config_path = dir.path().join(".codex").join("config.toml");
    fs::create_dir_all(config_path.parent().unwrap()).unwrap();
    fs::write(&config_path, "model = \"test\"\n").unwrap();
    let mut hooks = required_codex_hook_metadata(&hooks_path, "untrusted", true);
    hooks.retain(|hook| hook.event_name != "stop");
    let mut client = FakeCodexHooksClient {
        hook_lists: VecDeque::from([Ok(hooks)]),
        ..FakeCodexHooksClient::default()
    };

    let error = auto_trust_codex_hooks(
        &mut client,
        dir.path(),
        &config_path,
        &expected_plugin_command(),
    )
    .unwrap_err();

    assert!(error.contains("Stop"));
    assert!(client.trusted.is_empty());
}

#[test]
fn codex_auto_trust_refuses_duplicate_discovered_handler() {
    let dir = tempdir().unwrap();
    let config_path = dir.path().join("config.toml");
    fs::write(&config_path, "").unwrap();
    let mut hooks = generated_codex_hook_metadata(dir.path(), "untrusted", true);
    let mut duplicate = hooks.last().unwrap().clone();
    duplicate.key = "duplicate-post-compact".into();
    hooks.push(duplicate);
    let mut client = FakeCodexHooksClient {
        hook_lists: VecDeque::from([Ok(hooks)]),
        ..FakeCodexHooksClient::default()
    };

    let error = auto_trust_codex_hooks(
        &mut client,
        dir.path(),
        &config_path,
        &expected_plugin_command(),
    )
    .unwrap_err();

    assert!(error.contains("duplicate: PostCompact"), "{error}");
    assert!(client.trusted.is_empty());
}

#[test]
fn every_generated_codex_hook_is_required_exactly_once_and_trusted() {
    for (event, display) in [
        ("session_start", "SessionStart"),
        ("user_prompt_submit", "UserPromptSubmit"),
        ("pre_tool_use", "PreToolUse"),
        ("post_tool_use", "PostToolUse"),
        ("permission_request", "PermissionRequest"),
        ("subagent_start", "SubagentStart"),
        ("subagent_stop", "SubagentStop"),
        ("stop", "Stop"),
        ("pre_compact", "PreCompact"),
        ("post_compact", "PostCompact"),
    ] {
        for condition in ["missing", "duplicate", "disabled", "modified"] {
            let dir = tempdir().unwrap();
            let mut hooks = generated_codex_hook_metadata(dir.path(), "trusted", true);
            let target = hooks
                .iter()
                .position(|hook| hook.event_name == event)
                .unwrap();
            let target_key = hooks[target].key.clone();
            match condition {
                "missing" => {
                    hooks.remove(target);
                }
                "duplicate" => {
                    let mut duplicate = hooks[target].clone();
                    duplicate.key = format!("duplicate-{event}");
                    hooks.push(duplicate);
                }
                "disabled" => hooks[target].enabled = false,
                "modified" => hooks[target].trust_status = "modified".into(),
                _ => unreachable!(),
            }

            let report = codex_hook_trust_report_for(&hooks);
            let json = report.to_json();
            assert!(
                !report.ready(),
                "{event} unexpectedly ready while {condition}"
            );
            match condition {
                "missing" => assert!(
                    json["missing_required"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|value| value == display),
                    "{json}"
                ),
                "duplicate" => assert!(
                    json["duplicate_required"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|value| value == display),
                    "{json}"
                ),
                "disabled" => assert!(
                    json["disabled"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|value| value == &target_key),
                    "{json}"
                ),
                "modified" => assert!(
                    json["modified"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|value| value == &target_key),
                    "{json}"
                ),
                _ => unreachable!(),
            }
        }
    }
}

#[test]
fn codex_auto_trust_reverifies_every_generated_hook_after_writing() {
    for event in [
        "session_start",
        "user_prompt_submit",
        "pre_tool_use",
        "post_tool_use",
        "permission_request",
        "subagent_start",
        "subagent_stop",
        "stop",
        "pre_compact",
        "post_compact",
    ] {
        for condition in ["missing", "duplicate", "disabled", "modified"] {
            let dir = tempdir().unwrap();
            let config_path = dir.path().join("config.toml");
            fs::write(&config_path, "").unwrap();
            let initial = generated_codex_hook_metadata(dir.path(), "untrusted", true);
            let mut verified = generated_codex_hook_metadata(dir.path(), "trusted", true);
            let target = verified
                .iter()
                .position(|hook| hook.event_name == event)
                .unwrap();
            match condition {
                "missing" => {
                    verified.remove(target);
                }
                "duplicate" => {
                    let mut duplicate = verified[target].clone();
                    duplicate.key = format!("duplicate-{event}");
                    verified.push(duplicate);
                }
                "disabled" => verified[target].enabled = false,
                "modified" => verified[target].trust_status = "modified".into(),
                _ => unreachable!(),
            }
            let mut client = FakeCodexHooksClient {
                hook_lists: VecDeque::from([
                    Ok(initial.clone()),
                    Ok(verified),
                    Ok(initial.clone()),
                ]),
                ..FakeCodexHooksClient::default()
            };

            let error = auto_trust_codex_hooks(
                &mut client,
                dir.path(),
                &config_path,
                &expected_plugin_command(),
            )
            .unwrap_err();

            assert!(
                error.contains("did not enable and trust"),
                "{event} {condition}: {error}"
            );
            assert_eq!(client.restored.len(), 1, "{event} {condition}");
            assert_eq!(client.restored[0].len(), 10, "{event} {condition}");
        }
    }
}

#[test]
fn codex_auto_trust_rejects_targeted_hook_that_disappears_after_write() {
    let dir = tempdir().unwrap();
    let config_path = dir.path().join("config.toml");
    fs::write(&config_path, "").unwrap();
    let initial = generated_codex_hook_metadata(dir.path(), "untrusted", true);
    let mut verified = initial.clone();
    for hook in &mut verified {
        hook.trust_status = "trusted".into();
    }
    verified.retain(|hook| hook.event_name != "post_compact");
    let mut client = FakeCodexHooksClient {
        hook_lists: VecDeque::from([Ok(initial.clone()), Ok(verified), Ok(initial.clone())]),
        ..FakeCodexHooksClient::default()
    };

    let error = auto_trust_codex_hooks(
        &mut client,
        dir.path(),
        &config_path,
        &expected_plugin_command(),
    )
    .unwrap_err();

    assert!(
        error.contains("unverified targeted hooks=relay-hook-9"),
        "{error}"
    );
    assert_eq!(client.restored.len(), 1);
    assert_eq!(client.restored[0].len(), initial.len());
}

#[test]
fn codex_auto_trust_rejects_targeted_hook_that_changes_key_after_write() {
    let dir = tempdir().unwrap();
    let config_path = dir.path().join("config.toml");
    fs::write(&config_path, "").unwrap();
    let initial = generated_codex_hook_metadata(dir.path(), "untrusted", true);
    let mut verified = initial.clone();
    for hook in &mut verified {
        hook.trust_status = "trusted".into();
    }
    verified.last_mut().unwrap().key = "replacement-post-compact".into();
    let mut client = FakeCodexHooksClient {
        hook_lists: VecDeque::from([Ok(initial.clone()), Ok(verified), Ok(initial.clone())]),
        ..FakeCodexHooksClient::default()
    };

    let error = auto_trust_codex_hooks(
        &mut client,
        dir.path(),
        &config_path,
        &expected_plugin_command(),
    )
    .unwrap_err();

    assert!(
        error.contains("unverified targeted hooks=relay-hook-9"),
        "{error}"
    );
    assert_eq!(client.restored.len(), 1);
}

#[test]
fn codex_auto_trust_restores_exact_prior_state_after_verification_failure() {
    let dir = tempdir().unwrap();
    let config_path = dir.path().join("config.toml");
    fs::write(
        &config_path,
        r#"
[hooks.state."relay-hook-0"]
trusted_hash = "sha256:original"
enabled = false
custom = "preserve"
"#,
    )
    .unwrap();
    let initial = generated_codex_hook_metadata(dir.path(), "untrusted", true);
    let mut failed_verification = generated_codex_hook_metadata(dir.path(), "trusted", true);
    failed_verification[9].enabled = false;
    let mut client = FakeCodexHooksClient {
        hook_lists: VecDeque::from([Ok(initial.clone()), Ok(failed_verification), Ok(initial)]),
        ..FakeCodexHooksClient::default()
    };

    let error = auto_trust_codex_hooks(
        &mut client,
        dir.path(),
        &config_path,
        &expected_plugin_command(),
    )
    .unwrap_err();

    assert!(error.contains("did not enable and trust"), "{error}");
    assert_eq!(client.restored.len(), 1);
    assert_eq!(client.restored[0].len(), 10);
    assert_eq!(
        client.restored[0]
            .iter()
            .find(|(key, _)| key == "relay-hook-0")
            .unwrap()
            .1,
        Some(json!({
            "trusted_hash": "sha256:original",
            "enabled": false,
            "custom": "preserve"
        }))
    );
    assert!(
        client.restored[0]
            .iter()
            .filter(|(key, _)| key != "relay-hook-0")
            .all(|(_, value)| value.is_none())
    );
}

#[test]
fn codex_auto_trust_aggregates_original_and_rollback_errors() {
    let dir = tempdir().unwrap();
    let config_path = dir.path().join("config.toml");
    fs::write(&config_path, "").unwrap();
    let initial = required_codex_hook_metadata(dir.path(), "untrusted", true);
    let mut client = FakeCodexHooksClient {
        hook_lists: VecDeque::from([Ok(initial)]),
        trust_error: Some("trust write failed".into()),
        restore_error: Some("trust restore failed".into()),
        ..FakeCodexHooksClient::default()
    };

    let error = auto_trust_codex_hooks(
        &mut client,
        dir.path(),
        &config_path,
        &expected_plugin_command(),
    )
    .unwrap_err();

    assert!(error.contains("trust write failed"), "{error}");
    assert!(error.contains("trust restore failed"), "{error}");
}

#[test]
fn codex_auto_trust_does_not_depend_on_reported_plugin_source_path() {
    let dir = tempdir().unwrap();
    let reported_hooks_path = dir.path().join("codex-cache").join("hooks.json");
    let config_path = dir.path().join(".codex").join("config.toml");
    fs::create_dir_all(config_path.parent().unwrap()).unwrap();
    fs::write(&config_path, "").unwrap();
    let initial = required_codex_hook_metadata(&reported_hooks_path, "untrusted", true);
    let verified = required_codex_hook_metadata(&reported_hooks_path, "trusted", true);
    let mut client = FakeCodexHooksClient {
        hook_lists: VecDeque::from([Ok(initial), Ok(verified)]),
        ..FakeCodexHooksClient::default()
    };

    auto_trust_codex_hooks(
        &mut client,
        dir.path(),
        &config_path,
        &expected_plugin_command(),
    )
    .unwrap();

    assert_eq!(client.trusted[0].len(), 10);
}

#[test]
fn codex_auto_trust_rejects_modified_loaded_plugin_hook_file() {
    let dir = tempdir().unwrap();
    let reported_hooks_path = dir.path().join("codex-cache").join("hooks.json");
    fs::create_dir_all(reported_hooks_path.parent().unwrap()).unwrap();
    fs::write(
        &reported_hooks_path,
        serde_json::to_vec_pretty(&generated_hooks(CodingAgent::Codex, "malicious-command"))
            .unwrap(),
    )
    .unwrap();
    let config_path = dir.path().join("config.toml");
    fs::write(&config_path, "").unwrap();
    let hooks = required_codex_hook_metadata(&reported_hooks_path, "untrusted", true);
    let mut client = FakeCodexHooksClient {
        hook_lists: VecDeque::from([Ok(hooks)]),
        ..FakeCodexHooksClient::default()
    };

    let error = auto_trust_codex_hooks(
        &mut client,
        dir.path(),
        &config_path,
        &expected_plugin_command(),
    )
    .unwrap_err();

    assert!(error.contains("loaded modified Relay hooks"), "{error}");
    assert!(client.trusted.is_empty());
}

#[test]
fn codex_hook_trust_report_distinguishes_modified_disabled_and_missing_hooks() {
    let dir = tempdir().unwrap();
    let hooks_path = dir.path().join(".codex").join("hooks.json");
    let hooks = vec![
        codex_hook_metadata(
            &hooks_path,
            "session_start",
            "trusted-hook",
            "trusted",
            true,
        ),
        codex_hook_metadata(
            &hooks_path,
            "user_prompt_submit",
            "modified-hook",
            "modified",
            false,
        ),
    ];
    let mut client = FakeCodexHooksClient {
        hook_lists: VecDeque::from([Ok(hooks)]),
        ..FakeCodexHooksClient::default()
    };

    let report =
        codex_hook_trust_report_with_client(&mut client, dir.path(), &expected_plugin_command())
            .unwrap();
    let json = report.to_json();

    assert!(!report.ready());
    assert_eq!(json["trusted"], json!(["trusted-hook"]));
    assert_eq!(json["modified"], json!(["modified-hook"]));
    assert_eq!(json["disabled"], json!(["modified-hook"]));
    assert_eq!(
        json["missing_required"],
        json!([
            "PreToolUse",
            "PostToolUse",
            "PermissionRequest",
            "SubagentStart",
            "SubagentStop",
            "Stop",
            "PreCompact",
            "PostCompact"
        ])
    );
}

#[test]
fn codex_hook_state_key_path_quotes_arbitrary_hook_identity() {
    assert_eq!(
        hook_state_key_path("path:hook.\"quoted\""),
        r#"hooks.state."path:hook.\"quoted\"""#
    );
}

#[cfg(not(windows))]
#[test]
fn codex_app_server_client_handshakes_lists_trusts_and_clears_hooks() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let hooks_path = dir.path().join(".codex").join("hooks.json");
    let hooks = required_codex_hook_metadata(&hooks_path, "trusted", true);
    let (_path, _hooks, _log) = fake_codex_app_server(dir.path(), &hooks);
    let mut client = CodexAppServerClient::start().unwrap();

    let listed = client.list_hooks(dir.path()).unwrap();
    client.trust_hooks(&listed).unwrap();
    client
        .clear_hook_trust(&["relay-hook-0".to_string()])
        .unwrap();
    client
        .restore_hook_trust(&[
            (
                "relay-hook-0".to_string(),
                Some(json!({"trusted_hash": "sha256:old", "enabled": false})),
            ),
            ("relay-hook-1".to_string(), None),
        ])
        .unwrap();
    drop(client);

    let requests = fs::read_to_string(dir.path().join("fake-codex-requests.jsonl")).unwrap();
    assert!(requests.contains(r#""method":"initialize""#));
    assert!(requests.contains(r#""method":"hooks/list""#));
    assert!(requests.contains(r#""method":"config/batchWrite""#));
    assert!(requests.contains(r#""trusted_hash":"sha256:relay-hook-0""#));
    assert!(requests.contains(r#""keyPath":"hooks.state.\"relay-hook-0\"""#));
    assert!(requests.contains(r#""trusted_hash":"sha256:old""#));
    assert!(requests.contains(r#""keyPath":"hooks.state.\"relay-hook-1\"""#));
    assert!(requests.contains(r#""value":null"#));
}

#[cfg(not(windows))]
#[test]
fn codex_setup_snapshot_restores_exact_files_and_trust() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let codex_dir = dir.path().join(".codex");
    fs::create_dir_all(&codex_dir).unwrap();
    let config_path = codex_dir.join("config.toml");
    let hooks_path = codex_dir.join("hooks.json");
    let config_backup = backup_path(&config_path);
    let hooks_backup = backup_path(&hooks_path);
    fs::write(
        &config_path,
        "[hooks.state.\"relay-hook-0\"]\ntrusted_hash = \"sha256:relay-hook-0\"\nenabled = true\n",
    )
    .unwrap();
    fs::write(&hooks_path, "{\"custom\":true}\n").unwrap();
    fs::write(&config_backup, "original config backup\n").unwrap();
    fs::write(&hooks_backup, "original hooks backup\n").unwrap();
    let original = [
        fs::read(&config_path).unwrap(),
        fs::read(&config_backup).unwrap(),
        fs::read(&hooks_path).unwrap(),
        fs::read(&hooks_backup).unwrap(),
    ];
    let metadata = required_codex_hook_metadata(&hooks_path, "trusted", true);
    let (_path, _hooks, _log) = fake_codex_app_server(dir.path(), &metadata);
    let snapshot = snapshot_codex_setup().unwrap();

    fs::write(&config_path, "model = \"changed\"\n").unwrap();
    fs::write(&hooks_path, "{}\n").unwrap();
    fs::remove_file(&config_backup).unwrap();
    fs::remove_file(&hooks_backup).unwrap();
    restore_codex_setup(&snapshot).unwrap();

    assert_eq!(fs::read(&config_path).unwrap(), original[0]);
    assert_eq!(fs::read(&config_backup).unwrap(), original[1]);
    assert_eq!(fs::read(&hooks_path).unwrap(), original[2]);
    assert_eq!(fs::read(&hooks_backup).unwrap(), original[3]);
}

#[test]
fn codex_install_rolls_back_all_files_when_trust_activation_fails() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let codex_dir = dir.path().join(".codex");
    fs::create_dir_all(&codex_dir).unwrap();
    let config_path = codex_dir.join("config.toml");
    let hooks_path = codex_dir.join("hooks.json");
    let config_backup = backup_path(&config_path);
    let hooks_backup = backup_path(&hooks_path);
    fs::write(&config_path, "model_provider = \"openai\"\n").unwrap();
    fs::write(&hooks_path, "{}\n").unwrap();
    fs::write(&config_backup, "original config backup\n").unwrap();
    fs::write(&hooks_backup, "original hooks backup\n").unwrap();

    let error = install_codex_with_trust(
        DEFAULT_URL,
        &expected_plugin_command(),
        |_home, _config, _command| Err("Codex trust write rejected".into()),
    )
    .unwrap_err();

    assert!(error.contains("trust write rejected"));
    assert_eq!(
        fs::read_to_string(&config_path).unwrap(),
        "model_provider = \"openai\"\n"
    );
    assert_eq!(fs::read_to_string(&hooks_path).unwrap(), "{}\n");
    assert_eq!(
        fs::read_to_string(&config_backup).unwrap(),
        "original config backup\n"
    );
    assert_eq!(
        fs::read_to_string(&hooks_backup).unwrap(),
        "original hooks backup\n"
    );
}

#[test]
fn repeated_codex_install_does_not_overwrite_original_backup() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("config.toml");
    fs::write(&path, "model_provider = \"openai\"\n").unwrap();

    install_codex_config(&path, DEFAULT_URL).unwrap();
    install_codex_config(&path, DEFAULT_URL).unwrap();

    assert_eq!(
        fs::read_to_string(backup_path(&path)).unwrap(),
        "model_provider = \"openai\"\n"
    );
}

#[test]
fn codex_install_backs_up_when_relay_provider_table_is_not_active() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("config.toml");
    fs::write(
        &path,
        r#"
model_provider = "openai"

[model_providers.nemo-relay-openai]
name = "NeMo Relay"
base_url = "http://127.0.0.1:47632"
wire_api = "responses"
requires_openai_auth = true
supports_websockets = false
"#,
    )
    .unwrap();

    install_codex_config(&path, DEFAULT_URL).unwrap();

    assert!(
        fs::read_to_string(backup_path(&path))
            .unwrap()
            .contains("model_provider = \"openai\"")
    );
}

#[test]
fn codex_install_backs_up_when_hooks_flag_changes_even_with_managed_provider() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("config.toml");
    fs::write(
        &path,
        r#"
model_provider = "nemo-relay-openai"

[features]
hooks = false

[model_providers.nemo-relay-openai]
name = "NeMo Relay"
base_url = "http://127.0.0.1:47632"
wire_api = "responses"
requires_openai_auth = true
supports_websockets = false
"#,
    )
    .unwrap();

    install_codex_config(&path, DEFAULT_URL).unwrap();

    let backup = fs::read_to_string(backup_path(&path)).unwrap();
    assert!(backup.contains("hooks = false"));
    uninstall_codex_config(&path, DEFAULT_URL, false).unwrap();
    let updated = fs::read_to_string(&path).unwrap();
    assert!(updated.contains("hooks = false"));
}

#[test]
fn codex_provider_installed_requires_active_managed_provider() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let codex_dir = dir.path().join(".codex");
    fs::create_dir_all(&codex_dir).unwrap();
    let path = codex_dir.join("config.toml");
    fs::write(
        &path,
        r#"
model_provider = "openai"

[model_providers.nemo-relay-openai]
name = "NeMo Relay"
base_url = "http://127.0.0.1:47632"
wire_api = "responses"
requires_openai_auth = true
supports_websockets = false
"#,
    )
    .unwrap();

    assert!(!codex_provider_installed(DEFAULT_URL));
    install_codex_config(&path, DEFAULT_URL).unwrap();
    assert!(codex_provider_installed(DEFAULT_URL));
    assert!(!codex_provider_installed("http://127.0.0.1:47633"));
    fs::write(
        &path,
        r#"
model_provider = "nemo-relay-openai"

[features]
hooks = false

[model_providers.nemo-relay-openai]
name = "NeMo Relay"
base_url = "http://127.0.0.1:47632"
wire_api = "responses"
requires_openai_auth = true
supports_websockets = false
"#,
    )
    .unwrap();
    assert!(!codex_provider_installed(DEFAULT_URL));
}

#[test]
fn codex_hooks_installed_requires_generated_plugin_local_groups() {
    let dir = tempdir().unwrap();
    let plugin_root = dir.path().join("plugin");
    let path = plugin_root.join("hooks").join("hooks.json");
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(
        &path,
        serde_json::to_vec_pretty(&json!({
            "hooks": {
                "SessionStart": [
                    {
                        "hooks": [
                            {
                                "type": "command",
                                "command": "nemo-relay plugin-shim hook codex --gateway-url http://127.0.0.1:47632",
                                "timeout": 30
                            }
                        ]
                    }
                ]
            }
        }))
        .unwrap(),
    )
    .unwrap();

    assert!(!codex_hooks_installed(&path).unwrap());
    write_plugin_hooks(&plugin_root);
    assert!(codex_hooks_installed(&path).unwrap());
}

#[cfg(not(windows))]
#[test]
fn codex_doctor_requires_app_server_reported_trust_but_allows_stopped_sidecar() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let codex_dir = dir.path().join(".codex");
    fs::create_dir_all(&codex_dir).unwrap();
    install_codex_config(&codex_dir.join("config.toml"), DEFAULT_URL).unwrap();
    let plugin_root = dir.path().join("plugin");
    let hooks_path = write_plugin_hooks(&plugin_root);
    let trusted = required_codex_hook_metadata(&hooks_path, "trusted", true);
    let (_path, _hooks, _log) = fake_codex_app_server(dir.path(), &trusted);
    let _plugin_root = EnvVarGuard::set_path("PLUGIN_ROOT", &plugin_root);

    doctor_plugin(CodingAgent::Codex, DEFAULT_URL, &plugin_root).unwrap();
    let report = doctor_plugin_json(CodingAgent::Codex, DEFAULT_URL, &plugin_root).unwrap();
    assert_eq!(report["checks"]["codex_hooks_trusted"], json!(true));
    assert_eq!(
        report["codex_hook_trust"]["trusted"],
        json!(
            (0..10)
                .map(|index| format!("relay-hook-{index}"))
                .collect::<Vec<_>>()
        )
    );
}

#[test]
fn codex_provider_install_check_requires_enabled_hooks_feature() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let codex_dir = dir.path().join(".codex");
    fs::create_dir_all(&codex_dir).unwrap();
    fs::write(
        codex_dir.join("config.toml"),
        r#"
model_provider = "nemo-relay-openai"

[features]
hooks = false

[model_providers.nemo-relay-openai]
name = "NeMo Relay"
base_url = "http://127.0.0.1:47632"
wire_api = "responses"
requires_openai_auth = true
supports_websockets = false
"#,
    )
    .unwrap();
    let plugin_root = dir.path().join("plugin");
    let hooks_path = write_plugin_hooks(&plugin_root);

    assert!(!codex_provider_installed(DEFAULT_URL));
    assert!(codex_hooks_installed(&hooks_path).unwrap());
}

#[test]
fn plugin_host_doctor_rejects_unsupported_agents_and_reports_lazy_claude_status() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());

    assert!(
        doctor_plugin(CodingAgent::Hermes, DEFAULT_URL, dir.path())
            .unwrap_err()
            .contains("supports claude and codex")
    );
    assert!(
        doctor_plugin_json(CodingAgent::Hermes, DEFAULT_URL, dir.path())
            .unwrap_err()
            .contains("supports claude and codex")
    );

    let report = doctor_plugin_json(CodingAgent::ClaudeCode, DEFAULT_URL, dir.path()).unwrap();
    assert_eq!(report["ok"], json!(false));
    assert_eq!(report["sidecar_health"], json!("not_running_mcp_start"));
    assert_eq!(report["checks"]["claude_provider_routing"], json!(false));
}

#[test]
fn codex_setup_uses_plugin_hooks_without_writing_user_hooks() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let codex_dir = dir.path().join(".codex");
    fs::create_dir_all(&codex_dir).unwrap();

    install_codex_with_trust(
        DEFAULT_URL,
        &expected_plugin_command(),
        |_home, _config, command| {
            assert_eq!(command, expected_plugin_command());
            Ok(())
        },
    )
    .unwrap();

    let hooks_path = codex_dir.join("hooks.json");
    assert!(!hooks_path.exists());
    assert!(codex_provider_installed(DEFAULT_URL));

    let mut client = empty_codex_hooks_client();
    uninstall_codex_with_client(DEFAULT_URL, Some(&mut client)).unwrap();
    assert!(!hooks_path.exists());
}

#[test]
fn codex_setup_and_uninstall_honor_custom_codex_home() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(&dir.path().join("home"));
    let codex_home = dir.path().join("custom-codex-home");
    let _codex_home = EnvVarGuard::set_path("CODEX_HOME", &codex_home);

    install_codex_with_trust(
        DEFAULT_URL,
        &expected_plugin_command(),
        |_cwd, config_path, _command| {
            assert_eq!(config_path, codex_home.join("config.toml"));
            Ok(())
        },
    )
    .unwrap();

    assert!(codex_provider_installed(DEFAULT_URL));
    assert!(codex_home.join("config.toml").exists());
    assert!(!dir.path().join("home/.codex/config.toml").exists());

    let mut client = empty_codex_hooks_client();
    uninstall_codex_with_client(DEFAULT_URL, Some(&mut client)).unwrap();
    assert!(!codex_provider_installed(DEFAULT_URL));
}

#[test]
fn relay_binary_prefers_sidecar_binary_override() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let sidecar_override = dir.path().join("sidecar").join("nemo-relay");
    fs::create_dir_all(sidecar_override.parent().unwrap()).unwrap();
    fs::write(&sidecar_override, b"sidecar override").unwrap();
    let _binary_override = EnvVarGuard::set_path("NEMO_RELAY_PLUGIN_BINARY", &sidecar_override);

    assert_eq!(relay_binary().unwrap(), sidecar_override);
}

#[test]
fn codex_uninstall_without_backup_removes_managed_hooks_flag() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("config.toml");
    fs::write(
        &path,
        r#"
model_provider = "nemo-relay-openai"

[features]
hooks = true

[model_providers.nemo-relay-openai]
name = "NeMo Relay"
base_url = "http://127.0.0.1:47632"
wire_api = "responses"
requires_openai_auth = true
supports_websockets = false
"#,
    )
    .unwrap();

    uninstall_codex_config(&path, DEFAULT_URL, false).unwrap();
    let updated = fs::read_to_string(&path).unwrap();

    assert!(!updated.contains("model_provider"));
    assert!(!updated.contains("nemo-relay-openai"));
    assert!(!updated.contains("hooks = true"));
}

#[test]
fn codex_uninstall_clears_all_trust_for_the_exact_relay_plugin() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let codex_dir = dir.path().join(".codex");
    fs::create_dir_all(&codex_dir).unwrap();
    let config_path = codex_dir.join("config.toml");
    let hooks_path = codex_dir.join("hooks.json");
    fs::write(&config_path, "model_provider = \"openai\"\n").unwrap();
    fs::write(&hooks_path, "{}\n").unwrap();
    install_codex_hooks(&hooks_path, DEFAULT_URL).unwrap();
    install_codex_config(&config_path, DEFAULT_URL).unwrap();
    let mut hooks = generated_codex_hook_metadata(&hooks_path, "trusted", true);
    let mut unrelated = codex_hook_metadata(
        &hooks_path,
        "session_start",
        "unrelated-hook",
        "trusted",
        true,
    );
    unrelated.command = Some("custom hook".into());
    hooks.push(unrelated);
    let mut cleared = hooks.clone();
    for hook in &mut cleared {
        hook.trust_status = "untrusted".into();
    }
    let mut client = FakeCodexHooksClient {
        hook_lists: VecDeque::from([Ok(hooks), Ok(cleared)]),
        ..FakeCodexHooksClient::default()
    };

    uninstall_codex_with_client(DEFAULT_URL, Some(&mut client)).unwrap();

    assert_eq!(
        client.cleared,
        vec![
            (0..10)
                .map(|index| format!("relay-hook-{index}"))
                .chain(["unrelated-hook".into()])
                .collect::<Vec<_>>()
        ]
    );
    assert!(
        !serde_json::from_str::<Value>(&fs::read_to_string(&hooks_path).unwrap())
            .unwrap()
            .to_string()
            .contains("plugin-shim hook codex")
    );
}

#[test]
fn codex_uninstall_clears_persisted_optional_hooks_after_downgrade() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let codex_dir = dir.path().join(".codex");
    fs::create_dir_all(&codex_dir).unwrap();
    let config_path = codex_dir.join("config.toml");
    let hooks_path = codex_dir.join("hooks.json");
    let all_hooks = persisted_relay_hook_metadata(&hooks_path, "trusted");
    let all_keys = all_hooks
        .iter()
        .map(|hook| hook.key.clone())
        .collect::<Vec<_>>();
    let unrelated_key = "other-plugin@example:hooks/hooks.json:session_start:0:0";
    write_persisted_hook_trust(&config_path, &all_keys, unrelated_key);
    let visible = all_hooks
        .into_iter()
        .filter(|hook| {
            !matches!(
                hook.event_name.as_str(),
                "post_tool_use_failure" | "notification" | "session_end"
            )
        })
        .collect::<Vec<_>>();
    let mut cleared = visible.clone();
    for hook in &mut cleared {
        hook.trust_status = "untrusted".into();
    }
    let mut client = FakeCodexHooksClient {
        hook_lists: VecDeque::from([Ok(visible), Ok(cleared)]),
        clear_config_path: Some(config_path.clone()),
        ..FakeCodexHooksClient::default()
    };

    uninstall_codex_with_client(DEFAULT_URL, Some(&mut client)).unwrap();

    assert_eq!(
        client.cleared[0].iter().cloned().collect::<BTreeSet<_>>(),
        all_keys.into_iter().collect::<BTreeSet<_>>()
    );
    assert_eq!(
        configured_hook_trust_keys(&config_path).unwrap(),
        BTreeSet::from([unrelated_key.to_string()])
    );
}

#[test]
fn codex_uninstall_clears_persisted_relay_trust_when_discovery_is_empty() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let codex_dir = dir.path().join(".codex");
    fs::create_dir_all(&codex_dir).unwrap();
    let config_path = codex_dir.join("config.toml");
    let hooks_path = codex_dir.join("hooks.json");
    let relay_keys = persisted_relay_hook_metadata(&hooks_path, "trusted")
        .into_iter()
        .map(|hook| hook.key)
        .collect::<Vec<_>>();
    let unrelated_key = "other-plugin@example:hooks/hooks.json:session_start:0:0";
    write_persisted_hook_trust(&config_path, &relay_keys, unrelated_key);
    let mut client = FakeCodexHooksClient {
        hook_lists: VecDeque::from([Ok(Vec::new())]),
        clear_config_path: Some(config_path.clone()),
        ..FakeCodexHooksClient::default()
    };

    uninstall_codex_with_client(DEFAULT_URL, Some(&mut client)).unwrap();

    assert_eq!(client.cleared.len(), 1);
    assert_eq!(
        client.cleared[0].iter().cloned().collect::<BTreeSet<_>>(),
        relay_keys.into_iter().collect::<BTreeSet<_>>()
    );
    assert_eq!(
        configured_hook_trust_keys(&config_path).unwrap(),
        BTreeSet::from([unrelated_key.to_string()])
    );
}

#[test]
fn codex_uninstall_rolls_back_persisted_relay_trust_when_clear_is_not_applied() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let codex_dir = dir.path().join(".codex");
    fs::create_dir_all(&codex_dir).unwrap();
    let config_path = codex_dir.join("config.toml");
    let hooks_path = codex_dir.join("hooks.json");
    let relay_keys = persisted_relay_hook_metadata(&hooks_path, "trusted")
        .into_iter()
        .map(|hook| hook.key)
        .collect::<Vec<_>>();
    let unrelated_key = "other-plugin@example:hooks/hooks.json:session_start:0:0";
    write_persisted_hook_trust(&config_path, &relay_keys, unrelated_key);
    let original_config = fs::read(&config_path).unwrap();
    let mut client = FakeCodexHooksClient {
        hook_lists: VecDeque::from([Ok(Vec::new())]),
        ..FakeCodexHooksClient::default()
    };

    let error = uninstall_codex_with_client(DEFAULT_URL, Some(&mut client)).unwrap_err();

    assert!(error.contains("did not clear trust"), "{error}");
    assert_eq!(fs::read(&config_path).unwrap(), original_config);
    assert_eq!(client.restored.len(), 1);
    assert_eq!(
        client.restored[0]
            .iter()
            .map(|(key, _)| key.clone())
            .collect::<BTreeSet<_>>(),
        relay_keys.into_iter().collect::<BTreeSet<_>>()
    );
    assert!(client.restored[0].iter().all(|(_, value)| value.is_some()));
}

#[test]
fn codex_uninstall_restores_files_even_when_trust_cleanup_fails() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let codex_dir = dir.path().join(".codex");
    fs::create_dir_all(&codex_dir).unwrap();
    let config_path = codex_dir.join("config.toml");
    let hooks_path = codex_dir.join("hooks.json");
    fs::write(&config_path, "model_provider = \"openai\"\n").unwrap();
    fs::write(&hooks_path, "{}\n").unwrap();
    install_codex_hooks(&hooks_path, DEFAULT_URL).unwrap();
    install_codex_config(&config_path, DEFAULT_URL).unwrap();
    let original_config = fs::read(&config_path).unwrap();
    let original_config_backup = fs::read(backup_path(&config_path)).unwrap();
    let original_hooks = fs::read(&hooks_path).unwrap();
    let original_hooks_backup = fs::read(backup_path(&hooks_path)).unwrap();
    let original_metadata = required_codex_hook_metadata(&hooks_path, "trusted", true);
    let mut client = FakeCodexHooksClient {
        hook_lists: VecDeque::from([Ok(original_metadata.clone()), Ok(original_metadata)]),
        clear_error: Some("config is locked".into()),
        ..FakeCodexHooksClient::default()
    };

    let error = uninstall_codex_with_client(DEFAULT_URL, Some(&mut client)).unwrap_err();

    assert!(error.contains("config is locked"), "{error}");
    assert_eq!(fs::read(&config_path).unwrap(), original_config);
    assert_eq!(
        fs::read(backup_path(&config_path)).unwrap(),
        original_config_backup
    );
    assert_eq!(fs::read(&hooks_path).unwrap(), original_hooks);
    assert_eq!(
        fs::read(backup_path(&hooks_path)).unwrap(),
        original_hooks_backup
    );
    assert_eq!(client.restored.len(), 1);
}

#[test]
fn codex_uninstall_requires_trust_client_before_mutating_files() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let codex_dir = dir.path().join(".codex");
    fs::create_dir_all(&codex_dir).unwrap();
    let config_path = codex_dir.join("config.toml");
    let hooks_path = codex_dir.join("hooks.json");
    fs::write(&config_path, "model_provider = \"openai\"\n").unwrap();
    fs::write(&hooks_path, "{\"custom\":true}\n").unwrap();
    let original_config = fs::read(&config_path).unwrap();
    let original_hooks = fs::read(&hooks_path).unwrap();

    let error = uninstall_codex_with_client(DEFAULT_URL, None).unwrap_err();

    assert!(error.contains("app-server is required"), "{error}");
    assert_eq!(fs::read(&config_path).unwrap(), original_config);
    assert_eq!(fs::read(&hooks_path).unwrap(), original_hooks);
}

#[test]
fn codex_uninstall_rolls_back_when_trust_cleanup_cannot_be_verified() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let codex_dir = dir.path().join(".codex");
    fs::create_dir_all(&codex_dir).unwrap();
    let config_path = codex_dir.join("config.toml");
    let hooks_path = codex_dir.join("hooks.json");
    fs::write(&config_path, "model_provider = \"openai\"\n").unwrap();
    fs::write(&hooks_path, "{}\n").unwrap();
    install_codex_hooks(&hooks_path, DEFAULT_URL).unwrap();
    install_codex_config(&config_path, DEFAULT_URL).unwrap();
    let original_config = fs::read(&config_path).unwrap();
    let original_hooks = fs::read(&hooks_path).unwrap();
    let trusted = required_codex_hook_metadata(&hooks_path, "trusted", true);
    let mut client = FakeCodexHooksClient {
        hook_lists: VecDeque::from([Ok(trusted.clone()), Ok(trusted.clone()), Ok(trusted)]),
        ..FakeCodexHooksClient::default()
    };

    let error = uninstall_codex_with_client(DEFAULT_URL, Some(&mut client)).unwrap_err();

    assert!(error.contains("did not clear trust"), "{error}");
    assert_eq!(fs::read(&config_path).unwrap(), original_config);
    assert_eq!(fs::read(&hooks_path).unwrap(), original_hooks);
    assert_eq!(client.restored.len(), 1);
}

#[test]
fn codex_uninstall_with_backup_preserves_user_changed_model_provider() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("config.toml");
    fs::write(&path, "model_provider = \"openai\"\n").unwrap();
    install_codex_config(&path, DEFAULT_URL).unwrap();
    fs::write(
        &path,
        r#"
model_provider = "local"

[features]
hooks = true

[model_providers.nemo-relay-openai]
name = "NeMo Relay"
base_url = "http://127.0.0.1:47632"
wire_api = "responses"
requires_openai_auth = true
supports_websockets = false
"#,
    )
    .unwrap();

    uninstall_codex_config(&path, DEFAULT_URL, false).unwrap();
    let updated = fs::read_to_string(&path).unwrap();

    assert!(updated.contains("model_provider = \"local\""));
    assert!(!updated.contains("nemo-relay-openai"));
    assert!(!backup_path(&path).exists());
}

#[test]
fn codex_uninstall_with_backup_preserves_user_changed_provider_table() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("config.toml");
    fs::write(&path, "model_provider = \"openai\"\n").unwrap();
    install_codex_config(&path, DEFAULT_URL).unwrap();
    fs::write(
        &path,
        r#"
model_provider = "nemo-relay-openai"

[features]
hooks = true

[model_providers.nemo-relay-openai]
name = "Custom Relay"
base_url = "http://127.0.0.1:47632"
wire_api = "responses"
requires_openai_auth = true
supports_websockets = false
"#,
    )
    .unwrap();

    uninstall_codex_config(&path, DEFAULT_URL, false).unwrap();
    let updated = fs::read_to_string(&path).unwrap();

    assert!(updated.contains("model_provider = \"nemo-relay-openai\""));
    assert!(updated.contains("name = \"Custom Relay\""));
    assert!(updated.contains("nemo-relay-openai"));
    assert!(!backup_path(&path).exists());
}

#[test]
fn codex_uninstall_preserves_user_changed_provider_url() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("config.toml");
    fs::write(&path, "model_provider = \"openai\"\n").unwrap();
    install_codex_config(&path, DEFAULT_URL).unwrap();
    fs::write(
        &path,
        r#"
model_provider = "nemo-relay-openai"

[features]
hooks = true

[model_providers.nemo-relay-openai]
name = "NeMo Relay"
base_url = "http://127.0.0.1:49999"
wire_api = "responses"
requires_openai_auth = true
supports_websockets = false
"#,
    )
    .unwrap();

    uninstall_codex_config(&path, DEFAULT_URL, false).unwrap();
    let updated = fs::read_to_string(&path).unwrap();

    assert!(updated.contains("model_provider = \"nemo-relay-openai\""));
    assert!(updated.contains("base_url = \"http://127.0.0.1:49999\""));
    assert!(!backup_path(&path).exists());
}

#[test]
fn codex_uninstall_without_backup_preserves_user_changed_provider_url() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("config.toml");
    fs::write(
        &path,
        r#"
model_provider = "nemo-relay-openai"

[features]
hooks = true

[model_providers.nemo-relay-openai]
name = "NeMo Relay"
base_url = "http://127.0.0.1:49999"
wire_api = "responses"
requires_openai_auth = true
supports_websockets = false
"#,
    )
    .unwrap();

    uninstall_codex_config(&path, DEFAULT_URL, false).unwrap();
    let updated = fs::read_to_string(&path).unwrap();

    assert!(updated.contains("model_provider = \"nemo-relay-openai\""));
    assert!(updated.contains("base_url = \"http://127.0.0.1:49999\""));
}

#[test]
fn codex_uninstall_without_backup_preserves_user_hooks_when_provider_is_not_managed() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("config.toml");
    fs::write(
        &path,
        r#"
model_provider = "openai"

[features]
hooks = true
"#,
    )
    .unwrap();

    uninstall_codex_config(&path, DEFAULT_URL, false).unwrap();
    let updated = fs::read_to_string(&path).unwrap();

    assert!(updated.contains("hooks = true"));
}

#[test]
fn codex_uninstall_preserves_hooks_feature_when_user_hooks_remain() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let codex_dir = dir.path().join(".codex");
    fs::create_dir_all(&codex_dir).unwrap();
    fs::write(
        codex_dir.join("config.toml"),
        r#"
model_provider = "openai"

[features]
hooks = false
"#,
    )
    .unwrap();

    let hooks_path = codex_dir.join("hooks.json");
    fs::write(
        &hooks_path,
        serde_json::to_vec_pretty(&json!({
            "hooks": {"SessionStart": [{
            "hooks": [
                {
                    "type": "command",
                    "command": "custom-hook",
                    "timeout": 30
                }
            ]
        }]}}))
        .unwrap(),
    )
    .unwrap();
    install_codex_with_trust(
        DEFAULT_URL,
        &expected_plugin_command(),
        |_home, _config, _command| Ok(()),
    )
    .unwrap();

    let mut client = empty_codex_hooks_client();
    uninstall_codex_with_client(DEFAULT_URL, Some(&mut client)).unwrap();

    let updated_config = fs::read_to_string(codex_dir.join("config.toml")).unwrap();
    assert!(updated_config.contains("hooks = true"));
    let updated_hooks: Value =
        serde_json::from_str(&fs::read_to_string(&hooks_path).unwrap()).unwrap();
    assert!(event_contains_command(
        &updated_hooks,
        "SessionStart",
        "custom-hook"
    ));
    assert!(
        !serde_json::to_string(&updated_hooks)
            .unwrap()
            .contains("plugin-shim hook codex")
    );
}

#[test]
fn codex_reinstall_uses_fresh_backup_after_prior_uninstall() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("config.toml");
    fs::write(&path, "model_provider = \"openai\"\n").unwrap();

    install_codex_config(&path, DEFAULT_URL).unwrap();
    uninstall_codex_config(&path, DEFAULT_URL, false).unwrap();
    assert!(!backup_path(&path).exists());

    fs::write(&path, "model_provider = \"local\"\n").unwrap();
    install_codex_config(&path, DEFAULT_URL).unwrap();
    uninstall_codex_config(&path, DEFAULT_URL, false).unwrap();

    assert_eq!(
        fs::read_to_string(&path).unwrap(),
        "model_provider = \"local\"\n"
    );
    assert!(!backup_path(&path).exists());
}

#[test]
fn claude_restore_without_backup_preserves_matching_user_relay_url() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let settings = dir.path().join(".claude").join("settings.json");
    fs::create_dir_all(settings.parent().unwrap()).unwrap();
    fs::write(
        &settings,
        serde_json::to_vec_pretty(&json!({
            "env": {
                "ANTHROPIC_BASE_URL": DEFAULT_URL,
                "OTHER": "kept"
            }
        }))
        .unwrap(),
    )
    .unwrap();

    restore_claude_provider(DEFAULT_URL).unwrap();

    let updated: Value = serde_json::from_str(&fs::read_to_string(&settings).unwrap()).unwrap();
    assert_eq!(
        json_env_string(&updated, "ANTHROPIC_BASE_URL"),
        Some(DEFAULT_URL)
    );
    assert_eq!(json_env_string(&updated, "OTHER"), Some("kept"));
    assert!(!backup_path(&settings).exists());
}

#[test]
fn claude_enable_rolls_back_backup_when_settings_write_fails() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let settings = dir.path().join(".claude").join("settings.json");
    fs::create_dir_all(settings.parent().unwrap()).unwrap();
    fs::write(
        &settings,
        serde_json::to_vec_pretty(&json!({
            "env": {
                "ANTHROPIC_BASE_URL": "https://api.anthropic.com"
            }
        }))
        .unwrap(),
    )
    .unwrap();
    fs::create_dir(settings.with_extension("json.tmp")).unwrap();

    let error = enable_claude_provider(DEFAULT_URL).unwrap_err();

    assert!(error.contains("failed to write"));
    assert!(!backup_path(&settings).exists());
}

#[test]
fn claude_enable_does_not_back_up_when_env_shape_is_invalid() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let settings = dir.path().join(".claude").join("settings.json");
    fs::create_dir_all(settings.parent().unwrap()).unwrap();
    fs::write(
        &settings,
        serde_json::to_vec_pretty(&json!({
            "env": "invalid"
        }))
        .unwrap(),
    )
    .unwrap();

    let error = enable_claude_provider(DEFAULT_URL).unwrap_err();

    assert!(error.contains("non-object env field"));
    assert!(!backup_path(&settings).exists());
    let unchanged: Value = serde_json::from_str(&fs::read_to_string(&settings).unwrap()).unwrap();
    assert_eq!(unchanged["env"], json!("invalid"));
}

#[test]
fn claude_restore_with_backup_preserves_user_settings_added_after_install() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let settings = dir.path().join(".claude").join("settings.json");
    fs::create_dir_all(settings.parent().unwrap()).unwrap();
    fs::write(
        &settings,
        serde_json::to_vec_pretty(&json!({
            "env": {
                "ANTHROPIC_BASE_URL": "https://api.anthropic.com",
                "ORIGINAL": "kept"
            }
        }))
        .unwrap(),
    )
    .unwrap();
    enable_claude_provider(DEFAULT_URL).unwrap();
    fs::write(
        &settings,
        serde_json::to_vec_pretty(&json!({
            "env": {
                "ANTHROPIC_BASE_URL": DEFAULT_URL,
                "ORIGINAL": "updated",
                "ADDED": "kept"
            },
            "theme": "dark"
        }))
        .unwrap(),
    )
    .unwrap();

    restore_claude_provider(DEFAULT_URL).unwrap();

    let updated: Value = serde_json::from_str(&fs::read_to_string(&settings).unwrap()).unwrap();
    assert_eq!(
        json_env_string(&updated, "ANTHROPIC_BASE_URL"),
        Some("https://api.anthropic.com")
    );
    assert_eq!(json_env_string(&updated, "ORIGINAL"), Some("updated"));
    assert_eq!(json_env_string(&updated, "ADDED"), Some("kept"));
    assert_eq!(updated["theme"], json!("dark"));
    assert!(!backup_path(&settings).exists());
}

#[test]
fn claude_restore_with_backup_preserves_user_changed_provider_url() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let settings = dir.path().join(".claude").join("settings.json");
    fs::create_dir_all(settings.parent().unwrap()).unwrap();
    fs::write(
        &settings,
        serde_json::to_vec_pretty(&json!({
            "env": {
                "ANTHROPIC_BASE_URL": "https://api.anthropic.com"
            }
        }))
        .unwrap(),
    )
    .unwrap();
    enable_claude_provider(DEFAULT_URL).unwrap();
    fs::write(
        &settings,
        serde_json::to_vec_pretty(&json!({
            "env": {
                "ANTHROPIC_BASE_URL": "http://127.0.0.1:49999"
            }
        }))
        .unwrap(),
    )
    .unwrap();

    restore_claude_provider(DEFAULT_URL).unwrap();

    let updated: Value = serde_json::from_str(&fs::read_to_string(&settings).unwrap()).unwrap();
    assert_eq!(
        json_env_string(&updated, "ANTHROPIC_BASE_URL"),
        Some("http://127.0.0.1:49999")
    );
    assert!(backup_path(&settings).exists());
}

#[test]
fn claude_reinstall_refreshes_backup_after_user_owned_restore() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let settings = dir.path().join(".claude").join("settings.json");
    fs::create_dir_all(settings.parent().unwrap()).unwrap();
    fs::write(
        &settings,
        serde_json::to_vec_pretty(&json!({
            "env": {
                "ANTHROPIC_BASE_URL": "https://api.anthropic.com"
            }
        }))
        .unwrap(),
    )
    .unwrap();

    enable_claude_provider(DEFAULT_URL).unwrap();
    fs::write(
        &settings,
        serde_json::to_vec_pretty(&json!({
            "env": {
                "ANTHROPIC_BASE_URL": "https://custom.example"
            }
        }))
        .unwrap(),
    )
    .unwrap();
    restore_claude_provider(DEFAULT_URL).unwrap();
    assert!(backup_path(&settings).exists());

    enable_claude_provider(DEFAULT_URL).unwrap();
    let refreshed_backup: Value =
        serde_json::from_str(&fs::read_to_string(backup_path(&settings)).unwrap()).unwrap();
    assert_eq!(
        json_env_string(&refreshed_backup, "ANTHROPIC_BASE_URL"),
        Some("https://custom.example")
    );

    restore_claude_provider(DEFAULT_URL).unwrap();

    let updated: Value = serde_json::from_str(&fs::read_to_string(&settings).unwrap()).unwrap();
    assert_eq!(
        json_env_string(&updated, "ANTHROPIC_BASE_URL"),
        Some("https://custom.example")
    );
    assert!(!backup_path(&settings).exists());
}

#[test]
fn claude_reinstall_uses_fresh_backup_after_prior_restore() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let settings = dir.path().join(".claude").join("settings.json");
    fs::create_dir_all(settings.parent().unwrap()).unwrap();
    fs::write(
        &settings,
        serde_json::to_vec_pretty(&json!({
            "env": {
                "ANTHROPIC_BASE_URL": "https://api.anthropic.com"
            }
        }))
        .unwrap(),
    )
    .unwrap();

    enable_claude_provider(DEFAULT_URL).unwrap();
    restore_claude_provider(DEFAULT_URL).unwrap();
    assert!(!backup_path(&settings).exists());

    fs::write(
        &settings,
        serde_json::to_vec_pretty(&json!({
            "env": {
                "ANTHROPIC_BASE_URL": "https://custom.example"
            }
        }))
        .unwrap(),
    )
    .unwrap();

    enable_claude_provider(DEFAULT_URL).unwrap();
    restore_claude_provider(DEFAULT_URL).unwrap();

    let updated: Value = serde_json::from_str(&fs::read_to_string(&settings).unwrap()).unwrap();
    assert_eq!(
        json_env_string(&updated, "ANTHROPIC_BASE_URL"),
        Some("https://custom.example")
    );
    assert!(!backup_path(&settings).exists());
}

#[test]
fn advisory_sidecar_lock_blocks_until_the_owner_releases_it() {
    let dir = tempdir().unwrap();
    let url = "http://127.0.0.1:47632";
    let owner = lock_sidecar_endpoint(dir.path(), url).unwrap();
    let path = dir.path().to_path_buf();
    let (ready_sender, ready_receiver) = std::sync::mpsc::channel();
    let (acquired_sender, acquired_receiver) = std::sync::mpsc::channel();
    let contender = thread::spawn(move || {
        ready_sender.send(()).unwrap();
        let lock = lock_sidecar_endpoint(&path, url).unwrap();
        acquired_sender.send(()).unwrap();
        drop(lock);
    });

    ready_receiver.recv_timeout(Duration::from_secs(5)).unwrap();
    let error = lock_sidecar_endpoint_for(dir.path(), url, Duration::ZERO).unwrap_err();
    assert!(error.contains("timed out waiting for sidecar lock"));
    drop(owner);
    acquired_receiver
        .recv_timeout(Duration::from_secs(5))
        .unwrap();
    contender.join().unwrap();
}

#[test]
fn sidecar_lock_zero_wait_fails_when_the_endpoint_is_owned() {
    let dir = tempdir().unwrap();
    let url = "http://127.0.0.1:47632";
    let owner = lock_sidecar_endpoint(dir.path(), url).unwrap();

    let error = lock_sidecar_endpoint_for(dir.path(), url, Duration::ZERO).unwrap_err();

    assert!(error.contains("timed out waiting for sidecar lock"));
    drop(owner);
}

#[test]
fn sidecar_ownership_records_are_endpoint_specific_and_enumerated() {
    let dir = tempdir().unwrap();
    let first = sidecar_owner_path(dir.path(), CodingAgent::Codex, "http://127.0.0.1:47632");
    let second = sidecar_owner_path(dir.path(), CodingAgent::Codex, "http://127.0.0.1:47633");
    let legacy = dir.path().join("codex-sidecar.owner.json");
    let ignored = sidecar_owner_path(
        dir.path(),
        CodingAgent::ClaudeCode,
        "http://127.0.0.1:47634",
    );
    for path in [&first, &second, &legacy, &ignored] {
        fs::write(path, "{}").unwrap();
    }

    let paths = sidecar_owner_paths(dir.path(), CodingAgent::Codex).unwrap();

    assert_eq!(paths.len(), 3);
    assert!(paths.contains(&first));
    assert!(paths.contains(&second));
    assert!(paths.contains(&legacy));
    assert_ne!(first, second);
}

#[test]
fn managed_sidecar_publishes_valid_ownership_before_parent_validation() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let state = dir.path().join("bootstrap-state");
    let _state = EnvVarGuard::set_path(BOOTSTRAP_STATE_DIR_ENV, &state);
    let _agent = EnvVarGuard::set_value(BOOTSTRAP_AGENT_ENV, "codex");
    let _token =
        EnvVarGuard::set_value("NEMO_RELAY_BOOTSTRAP_SHUTDOWN_TOKEN", "test-shutdown-token");
    let _fingerprint =
        EnvVarGuard::set_value(crate::config::BOOTSTRAP_FINGERPRINT_ENV, "test-fingerprint");
    let address = "127.0.0.1:47632".parse().unwrap();

    publish_sidecar_owner_from_env(address).unwrap();

    let url = format!("http://{address}");
    let owner_path = sidecar_owner_path(&state, CodingAgent::Codex, &url);
    let pid_path = sidecar_pid_path(&state, CodingAgent::Codex, &url);
    validate_sidecar_owner(
        &owner_path,
        &pid_path,
        std::process::id(),
        &url,
        "test-shutdown-token",
        Some("test-fingerprint"),
    )
    .unwrap();
}

#[test]
fn managed_shutdown_rejects_a_listener_without_authenticated_health_proof() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let state = dir.path().join("bootstrap-state");
    fs::create_dir_all(&state).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let url = format!("http://{address}");
    let owner_path = sidecar_owner_path(&state, CodingAgent::Codex, &url);
    write_sidecar_owner(
        &owner_path,
        std::process::id(),
        &url,
        "must-not-be-sent",
        Some("hmac-sha256:fake-owner-fingerprint"),
    )
    .unwrap();
    let (request_sender, request_receiver) = std::sync::mpsc::channel();
    let listener_thread = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let request = read_http_request(&mut stream);
        request_sender.send(request).unwrap();
        let body = format!(
            r#"{{"status":"ok","service":"nemo-relay","version":"{}","bootstrap_protocol":{}}}"#,
            env!("CARGO_PKG_VERSION"),
            BOOTSTRAP_PROTOCOL_VERSION
        );
        stream
            .write_all(
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
                .as_bytes(),
            )
            .unwrap();
    });

    let error = stop_owned_sidecar_record(CodingAgent::Codex, &state, &owner_path).unwrap_err();

    assert!(error.contains("foreign listener"), "{error}");
    let request = request_receiver
        .recv_timeout(Duration::from_secs(2))
        .unwrap();
    let request = String::from_utf8(request).unwrap();
    assert!(request.starts_with("GET /healthz HTTP/1.1"), "{request}");
    assert!(!request.contains("must-not-be-sent"), "{request}");
    assert!(owner_path.exists());
    listener_thread.join().unwrap();
}

#[cfg(unix)]
#[test]
fn failed_sidecar_startup_terminates_the_detached_process_group() {
    let dir = tempdir().unwrap();
    let grandchild_pid_path = dir.path().join("grandchild.pid");
    let mut command = std::process::Command::new("sh");
    command
        .args(["-c", "sleep 30 & echo $! > \"$1\"; exit 1", "sh"])
        .arg(&grandchild_pid_path);
    configure_detached_sidecar(&mut command);
    let mut child = command.spawn().unwrap();
    let deadline = Instant::now() + Duration::from_secs(3);
    while !grandchild_pid_path.exists() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(10));
    }
    let grandchild_pid = fs::read_to_string(&grandchild_pid_path)
        .unwrap()
        .trim()
        .parse::<i32>()
        .unwrap();
    assert!(!child.wait().unwrap().success());

    terminate_sidecar_process_tree(&mut child);

    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        // SAFETY: Signal 0 performs an existence check and does not alter the target process.
        let result = unsafe { libc::kill(grandchild_pid, 0) };
        if result == -1 && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH) {
            break;
        }
        if Instant::now() >= deadline {
            // SAFETY: Best-effort cleanup of the test process if the process-group assertion fails.
            unsafe { libc::kill(grandchild_pid, libc::SIGKILL) };
            panic!("detached sidecar grandchild {grandchild_pid} survived startup termination");
        }
        thread::sleep(Duration::from_millis(20));
    }
}

#[cfg(windows)]
#[test]
fn windows_sidecar_job_terminates_an_assigned_process() {
    let job = SidecarJob::create().unwrap();
    assert!(job.name().starts_with("Local\\NeMoRelaySidecar-"));
    let mut command = std::process::Command::new("cmd");
    command.args(["/C", "ping -n 30 127.0.0.1 >NUL"]);
    configure_detached_sidecar(&mut command);
    let mut child = command.spawn().unwrap();
    match job.assign(&child) {
        Ok(()) => {
            job.terminate();
            assert!(!child.wait().unwrap().success());
        }
        Err(error) => {
            // Restrictive host Job Objects may forbid nested assignment; production reports the
            // error and refuses to serve without its process-tree cleanup guarantee.
            assert!(error.contains("Job Object"), "{error}");
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

#[test]
fn sidecar_reaper_removes_only_the_exited_process_records() {
    let dir = tempdir().unwrap();
    let url = "http://127.0.0.1:47632";
    let owner_path = sidecar_owner_path(dir.path(), CodingAgent::Codex, url);
    let pid_path = sidecar_pid_path(dir.path(), CodingAgent::Codex, url);
    let endpoint_lock = lock_sidecar_endpoint(dir.path(), url).unwrap();
    #[cfg(windows)]
    let mut command = {
        let mut command = std::process::Command::new("cmd");
        command.args(["/C", "ping -n 2 127.0.0.1 >NUL"]);
        command
    };
    #[cfg(not(windows))]
    let mut command = {
        let mut command = std::process::Command::new("sh");
        command.args(["-c", "sleep 0.1"]);
        command
    };
    let child = command.spawn().unwrap();
    let pid = child.id();
    write_sidecar_owner(
        &owner_path,
        pid,
        url,
        "test-shutdown-token",
        Some("test-fingerprint"),
    )
    .unwrap();
    fs::write(&pid_path, pid.to_string()).unwrap();

    handoff_sidecar_to_reaper(
        child,
        owner_path.clone(),
        pid_path.clone(),
        sidecar_lock_path(dir.path(), url),
    )
    .unwrap();
    drop(endpoint_lock);

    let deadline = Instant::now() + Duration::from_secs(3);
    while (owner_path.exists() || pid_path.exists()) && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(20));
    }
    assert!(!owner_path.exists());
    assert!(!pid_path.exists());
}

#[test]
fn sidecar_reaper_does_not_block_other_cleanup_on_a_contended_lock() {
    let dir = tempdir().unwrap();
    let blocked_url = "http://127.0.0.1:47632";
    let free_url = "http://127.0.0.1:47633";
    let blocked_lock = lock_sidecar_endpoint(dir.path(), blocked_url).unwrap();
    let mut records = Vec::new();
    for url in [blocked_url, free_url] {
        let child = short_lived_command().spawn().unwrap();
        let pid = child.id();
        let owner_path = sidecar_owner_path(dir.path(), CodingAgent::Codex, url);
        let pid_path = sidecar_pid_path(dir.path(), CodingAgent::Codex, url);
        write_sidecar_owner(
            &owner_path,
            pid,
            url,
            "test-shutdown-token",
            Some("test-fingerprint"),
        )
        .unwrap();
        fs::write(&pid_path, pid.to_string()).unwrap();
        handoff_sidecar_to_reaper(
            child,
            owner_path.clone(),
            pid_path.clone(),
            sidecar_lock_path(dir.path(), url),
        )
        .unwrap();
        records.push((owner_path, pid_path));
    }

    let deadline = Instant::now() + Duration::from_secs(3);
    while (records[1].0.exists() || records[1].1.exists()) && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(20));
    }
    assert!(!records[1].0.exists());
    assert!(!records[1].1.exists());
    assert!(records[0].0.exists());

    drop(blocked_lock);
    let deadline = Instant::now() + Duration::from_secs(3);
    while (records[0].0.exists() || records[0].1.exists()) && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(20));
    }
    assert!(!records[0].0.exists());
    assert!(!records[0].1.exists());
}

#[test]
fn sidecar_state_directory_does_not_follow_runtime_environment() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let _config_home = EnvVarGuard::remove("XDG_CONFIG_HOME");
    let first = {
        let _runtime = EnvVarGuard::set_path("XDG_RUNTIME_DIR", &dir.path().join("runtime-a"));
        sidecar_state_dir().unwrap()
    };
    let second = {
        let _runtime = EnvVarGuard::set_path("XDG_RUNTIME_DIR", &dir.path().join("runtime-b"));
        sidecar_state_dir().unwrap()
    };

    assert_eq!(first, second);
}

#[test]
fn sidecar_lock_name_uses_gateway_host_and_port() {
    assert_eq!(
        sidecar_lock_name("http://127.0.0.1:47632/hooks"),
        "127.0.0.1-47632"
    );
    assert_eq!(sidecar_lock_name("http://localhost"), "localhost-80");
    assert_eq!(
        sidecar_lock_name("not a url/with spaces"),
        "not_a_url_with_spaces"
    );
}

#[test]
fn runtime_dir_fallback_is_user_scoped() {
    let runtime = runtime_dir_for(
        None,
        None,
        None,
        std::path::PathBuf::from("/tmp"),
        Some("alice/example".into()),
        None,
    );

    assert_eq!(
        runtime,
        std::path::PathBuf::from("/tmp")
            .join("alice_example")
            .join("nemo-relay-plugin")
    );
}

#[test]
fn runtime_dir_prefers_explicit_runtime_base_without_user_segment() {
    let runtime = runtime_dir_for(
        Some("/run/user/1000".into()),
        None,
        None,
        std::path::PathBuf::from("/tmp"),
        Some("alice".into()),
        None,
    );

    assert_eq!(
        runtime,
        std::path::PathBuf::from("/run/user/1000").join("nemo-relay-plugin")
    );
}

#[test]
fn codex_hook_command_uses_cmd_quoting_for_windows_paths() {
    let relay = std::path::PathBuf::from(r"C:\Program Files\NeMo 100%\bin\nemo-relay.exe");
    let command = codex_hook_command_for_platform(&relay, DEFAULT_URL, true);

    assert_eq!(
        command,
        r#""C:\Program Files\NeMo 100%%\bin\nemo-relay.exe" hook-forward codex --gateway-url http://127.0.0.1:47632"#
    );
    assert_eq!(
        codex_plugin_hook_command_for_platform(&relay, true),
        r#""C:\Program Files\NeMo 100%%\bin\nemo-relay.exe" hook-forward codex --gateway-url http://127.0.0.1:47632"#
    );
    assert_eq!(
        shell_quote_arg_for_platform("foo&bar", true),
        r#""foo^&bar""#
    );
}

#[test]
fn windows_sidecar_flags_only_request_permitted_job_breakaway() {
    let base = WINDOWS_CREATE_NEW_PROCESS_GROUP | WINDOWS_CREATE_NO_WINDOW;
    assert_eq!(windows_sidecar_creation_flags(false, None), (base, false));
    assert_eq!(
        windows_sidecar_creation_flags(true, Some(WINDOWS_JOB_OBJECT_LIMIT_BREAKAWAY_OK)),
        (base | WINDOWS_CREATE_BREAKAWAY_FROM_JOB, false)
    );
    assert_eq!(
        windows_sidecar_creation_flags(true, Some(WINDOWS_JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK)),
        (base, false)
    );
    assert_eq!(windows_sidecar_creation_flags(true, Some(0)), (base, true));
    assert_eq!(windows_sidecar_creation_flags(true, None), (base, true));
    assert_eq!(WINDOWS_JOB_OBJECT_LIMIT_KILL_ON_CLOSE, 0x0000_2000);
}

#[test]
fn codex_hook_command_uses_posix_single_quote_escaping() {
    let relay = std::path::PathBuf::from("/tmp/NeMo $Relay`test'/bin/nemo-relay");
    let command = codex_hook_command_for_platform(&relay, DEFAULT_URL, false);

    assert_eq!(
        command,
        "'/tmp/NeMo $Relay`test'\\''/bin/nemo-relay' hook-forward codex --gateway-url http://127.0.0.1:47632"
    );
    assert_eq!(
        codex_plugin_hook_command_for_platform(&relay, false),
        "'/tmp/NeMo $Relay`test'\\''/bin/nemo-relay' hook-forward codex --gateway-url http://127.0.0.1:47632"
    );
    assert_eq!(shell_quote_arg_for_platform("", false), "''");
    assert_eq!(
        shell_quote_arg_for_platform(r"/tmp/path\with-backslash", false),
        r#"'/tmp/path\with-backslash'"#
    );
}

#[test]
fn unready_sidecar_child_is_terminated_and_pid_removed() {
    let dir = tempdir().unwrap();
    let pid_path = dir.path().join("codex-sidecar.pid");
    let mut command = long_lived_command();
    let child = command.spawn().unwrap();
    fs::write(&pid_path, child.id().to_string()).unwrap();

    let error = terminate_unready_sidecar(child, &pid_path, DEFAULT_URL).unwrap_err();

    assert!(error.contains("terminated startup process"));
    assert!(!pid_path.exists());
}

#[cfg(unix)]
#[test]
fn spawned_sidecar_uses_an_independent_process_group() {
    let mut command = long_lived_command();
    configure_detached_sidecar(&mut command);
    let mut child = command.spawn().unwrap();

    let output = std::process::Command::new("ps")
        .args(["-o", "pgid=", "-p", &child.id().to_string()])
        .output()
        .unwrap();
    let process_group = String::from_utf8(output.stdout)
        .unwrap()
        .trim()
        .parse::<u32>()
        .unwrap();

    assert_eq!(process_group, child.id());
    child.kill().unwrap();
    child.wait().unwrap();
}

#[test]
fn ensure_sidecar_releases_lock_when_startup_fails_fast() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let runtime = dir.path().join("runtime");
    let _runtime = EnvVarGuard::set_path("XDG_RUNTIME_DIR", &runtime);

    let error = loopback_bind("not a loopback url")
        .and_then(|bind| GatewaySpec::new(CodingAgent::Codex, bind).ensure())
        .unwrap_err();

    assert!(error.contains("loopback URL"));
    assert!(
        !runtime
            .join("nemo-relay-plugin")
            .join("not_a_loopback_url-sidecar.lock")
            .exists()
    );
}

#[test]
fn healthz_rejects_foreign_success_response() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let handle = thread::spawn(move || {
        for _ in 0..4 {
            let (mut stream, _) = listener.accept().unwrap();
            let _ = read_http_request(&mut stream);
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Length: 15\r\nConnection: close\r\n\r\n{\"status\":\"ok\"}",
                )
                .unwrap();
        }
    });

    let error = ensure_sidecar_bind(CodingAgent::Codex, address).unwrap_err();
    assert!(
        error.contains("not a compatible NeMo Relay gateway"),
        "{error}"
    );
    handle.join().unwrap();
}

#[test]
fn startup_reprobes_a_transient_foreign_health_result_after_locking() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let handle = thread::spawn(move || {
        let (mut starting, _) = listener.accept().unwrap();
        let _ = read_http_request(&mut starting);
        starting
            .write_all(b"HTTP/1.1 503 Starting\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            .unwrap();
        drop(starting);

        let (mut ready, _) = listener.accept().unwrap();
        let _ = read_http_request(&mut ready);
        let body = format!(
            r#"{{"status":"ok","service":"nemo-relay","version":"{}","bootstrap_protocol":{}}}"#,
            env!("CARGO_PKG_VERSION"),
            BOOTSTRAP_PROTOCOL_VERSION
        );
        ready
            .write_all(
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
                .as_bytes(),
            )
            .unwrap();
    });

    let endpoint = ensure_sidecar_bind(CodingAgent::Codex, address).unwrap();

    assert_eq!(endpoint.address, address);
    handle.join().unwrap();
}

#[test]
fn active_startup_lock_waits_for_relay_identity_instead_of_rejecting_listener() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let runtime_base = dir.path().join("runtime");
    let _runtime = EnvVarGuard::set_path("XDG_RUNTIME_DIR", &runtime_base);
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let state = sidecar_state_dir().unwrap();
    fs::create_dir_all(&state).unwrap();
    let url = format!("http://{address}");
    let lock_path = sidecar_lock_path(&state, &url);
    let owner_lock = lock_sidecar_endpoint(&state, &url).unwrap();
    let handle = thread::spawn(move || {
        let (mut starting, _) = listener.accept().unwrap();
        let _ = read_http_request(&mut starting);
        starting
            .write_all(b"HTTP/1.1 503 Starting\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            .unwrap();
        drop(starting);

        let (mut ready, _) = listener.accept().unwrap();
        let _ = read_http_request(&mut ready);
        let body = format!(
            r#"{{"status":"ok","service":"nemo-relay","version":"{}","bootstrap_protocol":{}}}"#,
            env!("CARGO_PKG_VERSION"),
            BOOTSTRAP_PROTOCOL_VERSION
        );
        ready
            .write_all(
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
                .as_bytes(),
            )
            .unwrap();
        drop(owner_lock);
    });

    let endpoint = ensure_sidecar_bind(CodingAgent::Codex, address).unwrap();

    assert_eq!(endpoint.address, address);
    assert_eq!(endpoint.url, format!("http://{address}"));
    handle.join().unwrap();
    assert!(lock_path.exists());
}

#[cfg(unix)]
#[test]
fn start_sidecar_reports_child_exit_before_healthz_ready() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let relay = dir.path().join("nemo-relay");
    fs::write(&relay, "#!/bin/sh\nexit 7\n").unwrap();
    let mut permissions = fs::metadata(&relay).unwrap().permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&relay, permissions).unwrap();
    let _binary = EnvVarGuard::set_path("NEMO_RELAY_PLUGIN_BINARY", &relay);

    let error = start_sidecar(CodingAgent::Codex, "http://127.0.0.1:0", dir.path()).unwrap_err();

    assert!(error.contains("exited before becoming ready"), "{error}");
    assert!(!dir.path().join("codex-sidecar.pid").exists());
}

#[test]
fn codex_uninstall_removes_only_exact_generated_hook_groups() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("hooks.json");
    let command = codex_hook_command("http://127.0.0.1:47633");
    let generated = generated_hooks(CodingAgent::Codex, &command);
    let user_command = "custom-user-codex-hook";
    let config = json!({
        "hooks": {
            "SessionStart": [
                generated["hooks"]["SessionStart"][0].clone(),
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": user_command,
                            "timeout": 30
                        }
                    ]
                }
            ]
        }
    });
    fs::write(&path, serde_json::to_vec_pretty(&config).unwrap()).unwrap();

    uninstall_codex_hooks(&path, "http://127.0.0.1:47633").unwrap();
    let updated: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();

    assert!(event_contains_command(
        &updated,
        "SessionStart",
        user_command
    ));
    assert!(!generated_event_contains_group(
        &updated,
        "SessionStart",
        &generated["hooks"]["SessionStart"][0]
    ));
}

#[test]
fn codex_install_hooks_removes_prior_non_default_generated_url() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("hooks.json");
    let old_command = codex_hook_command("http://127.0.0.1:47633");
    let new_command = codex_hook_command("http://127.0.0.1:47634");

    install_codex_hooks(&path, "http://127.0.0.1:47633").unwrap();
    install_codex_hooks(&path, "http://127.0.0.1:47634").unwrap();
    let updated: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();

    assert!(!event_contains_command(
        &updated,
        "SessionStart",
        &old_command
    ));
    assert!(event_contains_command(
        &updated,
        "SessionStart",
        &new_command
    ));
}

#[test]
fn codex_uninstall_hooks_removes_all_generated_url_variants_for_launcher() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("hooks.json");
    let old_command = codex_hook_command("http://127.0.0.1:47633");
    let new_command = codex_hook_command("http://127.0.0.1:47634");
    let mut old_generated = generated_hooks(CodingAgent::Codex, &old_command);
    let new_generated = generated_hooks(CodingAgent::Codex, &new_command);
    old_generated["hooks"]["SessionStart"]
        .as_array_mut()
        .unwrap()
        .push(new_generated["hooks"]["SessionStart"][0].clone());
    fs::write(&path, serde_json::to_vec_pretty(&old_generated).unwrap()).unwrap();

    uninstall_codex_hooks(&path, "http://127.0.0.1:47634").unwrap();
    let updated: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();

    assert!(!event_contains_command(
        &updated,
        "SessionStart",
        &old_command
    ));
    assert!(!event_contains_command(
        &updated,
        "SessionStart",
        &new_command
    ));
}

#[cfg(windows)]
fn long_lived_command() -> std::process::Command {
    let mut command = std::process::Command::new("cmd");
    command.args(["/C", "ping -n 60 127.0.0.1 >NUL"]);
    command
}

#[cfg(windows)]
fn short_lived_command() -> std::process::Command {
    let mut command = std::process::Command::new("cmd");
    command.args(["/C", "ping -n 2 127.0.0.1 >NUL"]);
    command
}

#[cfg(not(windows))]
fn long_lived_command() -> std::process::Command {
    let mut command = std::process::Command::new("sh");
    command.args(["-c", "sleep 60"]);
    command
}

#[cfg(not(windows))]
fn short_lived_command() -> std::process::Command {
    let mut command = std::process::Command::new("sh");
    command.args(["-c", "sleep 0.1"]);
    command
}

#[test]
fn codex_install_hooks_persist_custom_gateway_url() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("hooks.json");

    install_codex_hooks(&path, "http://127.0.0.1:47633").unwrap();
    let updated: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    let command = updated["hooks"]["SessionStart"][0]["hooks"][0]["command"]
        .as_str()
        .unwrap();

    assert!(command.contains("hook-forward codex"));
    assert!(command.contains("--gateway-url http://127.0.0.1:47633"));
}

#[test]
fn codex_install_migration_removes_legacy_relay_groups_and_preserves_unrelated_hooks() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("hooks.json");
    let relay = current_exe().unwrap();
    let legacy_command = legacy_codex_hook_command(&relay);
    let mut legacy = generated_hooks(CodingAgent::Codex, &legacy_command);
    legacy["hooks"]["SessionStart"]
        .as_array_mut()
        .unwrap()
        .push(json!({
            "hooks": [{
                "type": "command",
                "command": "custom-user-hook",
                "timeout": 30
            }]
        }));
    let original = serde_json::to_vec_pretty(&legacy).unwrap();
    fs::write(&path, &original).unwrap();

    remove_legacy_codex_hooks(&path).unwrap();
    let updated: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();

    assert!(!event_contains_command(
        &updated,
        "SessionStart",
        &legacy_command
    ));
    assert!(event_contains_command(
        &updated,
        "SessionStart",
        "custom-user-hook"
    ));
    assert_eq!(fs::read(backup_path(&path)).unwrap(), original);
}

#[test]
fn codex_migration_removes_modified_relay_handler_from_mixed_user_group() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("hooks.json");
    let legacy_command =
        "'/old install/nemo-relay' plugin-shim hook codex --gateway-url http://127.0.0.1:47632";
    write_json(
        &path,
        &json!({
            "hooks": {
                "SessionStart": [{
                    "hooks": [
                        {"type": "command", "command": legacy_command, "timeout": 60},
                        {"type": "command", "command": "custom-user-hook", "timeout": 45}
                    ]
                }]
            }
        }),
    )
    .unwrap();

    remove_legacy_codex_hooks(&path).unwrap();
    let updated = read_json_object(&path).unwrap();

    assert!(!event_contains_command(
        &updated,
        "SessionStart",
        legacy_command
    ));
    assert!(event_contains_command(
        &updated,
        "SessionStart",
        "custom-user-hook"
    ));
}

#[test]
fn codex_install_does_not_write_provider_config_when_hooks_are_invalid() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let codex_dir = dir.path().join(".codex");
    fs::create_dir_all(&codex_dir).unwrap();
    fs::write(
        codex_dir.join("config.toml"),
        "model_provider = \"openai\"\n",
    )
    .unwrap();
    let plugin_hooks = dir.path().join("plugin").join("hooks").join("hooks.json");
    fs::create_dir_all(plugin_hooks.parent().unwrap()).unwrap();
    fs::write(&plugin_hooks, "{ invalid json").unwrap();

    let error = install_codex(DEFAULT_URL, &plugin_hooks).unwrap_err();
    assert!(error.contains("invalid JSON"));

    assert_eq!(
        fs::read_to_string(codex_dir.join("config.toml")).unwrap(),
        "model_provider = \"openai\"\n"
    );
    assert!(!backup_path(&codex_dir.join("config.toml")).exists());
}

#[test]
fn codex_install_does_not_write_hooks_when_config_is_invalid() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let codex_dir = dir.path().join(".codex");
    fs::create_dir_all(&codex_dir).unwrap();
    fs::write(codex_dir.join("config.toml"), "model_provider = [").unwrap();
    let hooks_path = codex_dir.join("hooks.json");
    let original_hooks = serde_json::to_vec_pretty(&json!({
        "hooks": {
            "SessionStart": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": "custom-hook",
                            "timeout": 30
                        }
                    ]
                }
            ]
        }
    }))
    .unwrap();
    fs::write(&hooks_path, &original_hooks).unwrap();

    let error = install_codex_with_trust(
        DEFAULT_URL,
        &expected_plugin_command(),
        |_home, _config, _command| Err("expected exactly one Relay handler".into()),
    )
    .unwrap_err();
    assert!(error.contains("invalid TOML"));

    assert_eq!(fs::read(&hooks_path).unwrap(), original_hooks);
    assert!(!backup_path(&hooks_path).exists());
}

#[test]
fn codex_install_does_not_write_hooks_when_config_is_not_readable() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let codex_dir = dir.path().join(".codex");
    fs::create_dir_all(&codex_dir).unwrap();
    fs::create_dir(codex_dir.join("config.toml")).unwrap();
    let hooks_path = codex_dir.join("hooks.json");
    let original_hooks = serde_json::to_vec_pretty(&json!({
        "hooks": {
            "SessionStart": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": "custom-hook",
                            "timeout": 30
                        }
                    ]
                }
            ]
        }
    }))
    .unwrap();
    fs::write(&hooks_path, &original_hooks).unwrap();

    let error = install_codex_with_trust(
        DEFAULT_URL,
        &expected_plugin_command(),
        |_home, _config, _command| Err("expected exactly one Relay handler".into()),
    )
    .unwrap_err();
    assert!(error.contains("failed to read"));

    assert_eq!(fs::read(&hooks_path).unwrap(), original_hooks);
    assert!(!backup_path(&hooks_path).exists());
}

#[test]
fn codex_install_config_rolls_back_backup_when_write_fails() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("config.toml");
    fs::write(&path, "model_provider = \"openai\"\n").unwrap();
    fs::create_dir(path.with_extension("toml.tmp")).unwrap();

    let error = install_codex_config(&path, DEFAULT_URL).unwrap_err();

    assert!(error.contains("failed to write"));
    assert!(!backup_path(&path).exists());
}

#[test]
fn codex_install_preserves_invalid_user_hooks_when_trust_fails() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let codex_dir = dir.path().join(".codex");
    fs::create_dir_all(&codex_dir).unwrap();
    fs::write(
        codex_dir.join("config.toml"),
        "model_provider = \"openai\"\n",
    )
    .unwrap();
    let hooks_path = codex_dir.join("hooks.json");
    let original_hooks = serde_json::to_vec_pretty(&json!({
        "hooks": {
            "SessionStart": "invalid"
        }
    }))
    .unwrap();
    fs::write(&hooks_path, &original_hooks).unwrap();

    let error = install_codex_with_trust(
        DEFAULT_URL,
        &expected_plugin_command(),
        |_home, _config, _command| Err("expected exactly one Relay handler".into()),
    )
    .unwrap_err();

    assert!(error.contains("exactly one Relay handler"), "{error}");
    assert_eq!(fs::read(&hooks_path).unwrap(), original_hooks);
    assert!(!backup_path(&hooks_path).exists());
    assert_eq!(
        fs::read_to_string(codex_dir.join("config.toml")).unwrap(),
        "model_provider = \"openai\"\n"
    );
}

#[test]
fn codex_uninstall_rolls_back_hooks_when_provider_config_is_invalid() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let codex_dir = dir.path().join(".codex");
    fs::create_dir_all(&codex_dir).unwrap();
    fs::write(codex_dir.join("config.toml"), "model_provider = [").unwrap();
    let hooks_path = codex_dir.join("hooks.json");
    install_codex_hooks(&hooks_path, DEFAULT_URL).unwrap();
    let original_hooks = fs::read(&hooks_path).unwrap();

    let mut client = empty_codex_hooks_client();
    let error = uninstall_codex_with_client(DEFAULT_URL, Some(&mut client)).unwrap_err();

    assert!(error.contains("invalid TOML"));
    assert_eq!(fs::read(&hooks_path).unwrap(), original_hooks);
}

#[test]
fn codex_install_rolls_back_hooks_when_provider_config_write_fails() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let codex_dir = dir.path().join(".codex");
    fs::create_dir_all(&codex_dir).unwrap();
    fs::write(
        codex_dir.join("config.toml"),
        "model_provider = \"openai\"\n",
    )
    .unwrap();
    fs::create_dir(codex_dir.join("config.toml.tmp")).unwrap();
    let hooks_path = codex_dir.join("hooks.json");
    let original_hooks = serde_json::to_vec_pretty(&json!({
        "hooks": {
            "SessionStart": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": "custom-hook",
                            "timeout": 30
                        }
                    ]
                }
            ]
        }
    }))
    .unwrap();
    fs::write(&hooks_path, &original_hooks).unwrap();

    let plugin_hooks = write_plugin_hooks(&dir.path().join("plugin"));
    let error = install_codex(DEFAULT_URL, &plugin_hooks).unwrap_err();

    assert!(error.contains("failed to write"));
    assert_eq!(fs::read(&hooks_path).unwrap(), original_hooks);
    assert!(!backup_path(&hooks_path).exists());
    assert_eq!(
        fs::read_to_string(codex_dir.join("config.toml")).unwrap(),
        "model_provider = \"openai\"\n"
    );
}

#[test]
fn codex_uninstall_hooks_removes_legacy_generated_command() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("hooks.json");
    let relay = current_exe().unwrap();
    let legacy_command = legacy_codex_hook_command(&relay);
    let legacy = generated_hooks(CodingAgent::Codex, &legacy_command);
    fs::write(&path, serde_json::to_vec_pretty(&legacy).unwrap()).unwrap();

    uninstall_codex_hooks(&path, DEFAULT_URL).unwrap();
    let updated: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();

    assert!(!event_contains_command(
        &updated,
        "SessionStart",
        &legacy_command
    ));
}

#[test]
fn codex_provider_gateway_url_reads_managed_provider_url() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("config.toml");
    fs::write(
        &path,
        r#"
[model_providers.nemo-relay-openai]
base_url = "http://127.0.0.1:47633"
"#,
    )
    .unwrap();

    assert_eq!(
        codex_provider_gateway_url(&path).as_deref(),
        Some("http://127.0.0.1:47633")
    );
}

#[test]
fn healthz_times_out_for_bad_port_occupant() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let (accepted_sender, accepted_receiver) = std::sync::mpsc::channel();
    let (release_sender, release_receiver) = std::sync::mpsc::channel();
    let server = thread::spawn(move || {
        let Ok((mut stream, _)) = listener.accept() else {
            return;
        };
        accepted_sender.send(()).unwrap();
        release_receiver
            .recv_timeout(Duration::from_secs(5))
            .unwrap();
        let _ = stream.write_all(b"HTTP/1.1 200 OK\r\n\r\n");
    });
    let (result_sender, result_receiver) = std::sync::mpsc::channel();
    let health = thread::spawn(move || {
        let result = healthz(&format!("http://127.0.0.1:{port}"));
        result_sender.send(result).unwrap();
    });

    accepted_receiver
        .recv_timeout(Duration::from_secs(5))
        .unwrap();
    let result = result_receiver.recv_timeout(Duration::from_secs(5));
    release_sender.send(()).unwrap();
    server.join().unwrap();
    health.join().unwrap();
    assert!(!result.expect("health probe did not honor its read timeout"));
}

#[test]
fn shared_json_helpers_cover_missing_invalid_and_non_object_inputs() {
    let dir = tempdir().unwrap();
    let missing = dir.path().join("missing.json");
    assert_eq!(read_json_object(&missing).unwrap(), json!({}));

    let invalid = dir.path().join("invalid.json");
    fs::write(&invalid, "{not json").unwrap();
    assert!(
        read_json_object(&invalid)
            .unwrap_err()
            .contains("invalid JSON")
    );

    let array = dir.path().join("array.json");
    fs::write(&array, "[]").unwrap();
    assert!(
        read_json_object(&array)
            .unwrap_err()
            .contains("must contain a JSON object")
    );

    let nested = dir.path().join("nested").join("settings.json");
    write_json(&nested, &json!({"ok": true})).unwrap();
    assert_eq!(
        fs::read_to_string(&nested).unwrap(),
        "{\n  \"ok\": true\n}\n"
    );
}

#[test]
fn shared_filesystem_helpers_cover_tables_snapshots_and_lock_branches() {
    let dir = tempdir().unwrap();
    let mut doc = "agent = \"codex\"\n"
        .parse::<toml_edit::DocumentMut>()
        .unwrap();
    ensure_table(&mut doc, "agent").insert("enabled", toml_edit::value(true));
    assert!(doc["agent"].is_table());
    assert_eq!(doc["agent"]["enabled"].as_bool(), Some(true));

    let missing = dir.path().join("missing.txt");
    let snapshot = snapshot_optional_file(&missing).unwrap();
    fs::write(&missing, "created").unwrap();
    restore_file_snapshot(&snapshot).unwrap();
    assert!(!missing.exists());

    let existing = dir.path().join("existing.txt");
    fs::write(&existing, "before").unwrap();
    let snapshot = snapshot_optional_file(&existing).unwrap();
    fs::write(&existing, "after").unwrap();
    restore_file_snapshot(&snapshot).unwrap();
    assert_eq!(fs::read_to_string(&existing).unwrap(), "before");
}

#[test]
fn shared_defaults_cover_runtime_username_and_empty_segments() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let _plugin_url = EnvVarGuard::remove("NEMO_RELAY_PLUGIN_GATEWAY_URL");
    let _claude_url = EnvVarGuard::remove("NEMO_RELAY_GATEWAY_URL");
    let _timeout = EnvVarGuard::remove("NEMO_RELAY_PLUGIN_IDLE_TIMEOUT_SECS");
    let _fail_closed = EnvVarGuard::remove("NEMO_RELAY_FAIL_CLOSED");

    assert_eq!(plugin_idle_timeout().unwrap(), Duration::from_secs(300));
    assert_eq!(
        plugin_heartbeat_interval().unwrap(),
        Duration::from_secs(30)
    );
    assert_eq!(
        runtime_dir_for(
            None,
            None,
            Some("/tmp/temp-base".into()),
            dir.path().join("ignored"),
            None,
            Some("bob/name".into()),
        ),
        std::path::PathBuf::from("/tmp/temp-base")
            .join("bob_name")
            .join("nemo-relay-plugin")
    );
    assert_eq!(sidecar_lock_name(""), "unknown");
    assert_eq!(
        runtime_dir_for(None, None, None, dir.path().into(), None, None),
        dir.path().join("unknown-user").join("nemo-relay-plugin")
    );
}

#[test]
fn relay_binary_rejects_missing_override_and_uses_current_exe_fallback() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let missing = dir.path().join("missing-nemo-relay");
    let _binary_override = EnvVarGuard::set_path("NEMO_RELAY_PLUGIN_BINARY", &missing);
    assert!(
        relay_binary()
            .unwrap_err()
            .contains("NEMO_RELAY_PLUGIN_BINARY does not exist")
    );
    drop(_binary_override);
    let _binary_override = EnvVarGuard::remove("NEMO_RELAY_PLUGIN_BINARY");
    assert!(relay_binary().unwrap().exists());
}

#[test]
fn claude_provider_enable_status_and_restore_cover_managed_backup_paths() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let settings_path = dir.path().join(".claude").join("settings.json");
    fs::create_dir_all(settings_path.parent().unwrap()).unwrap();
    fs::write(
        &settings_path,
        serde_json::to_vec_pretty(&json!({
            "env": {
                "ANTHROPIC_BASE_URL": "https://api.anthropic.com",
                "OTHER": "kept"
            }
        }))
        .unwrap(),
    )
    .unwrap();

    assert_eq!(claude_settings_path().unwrap(), settings_path);
    assert_eq!(
        claude_settings_base_url().as_deref(),
        Some("https://api.anthropic.com")
    );
    enable_claude_provider(DEFAULT_URL).unwrap();
    assert_eq!(claude_settings_base_url().as_deref(), Some(DEFAULT_URL));
    assert_eq!(
        json_env_string(&read_json_object(&settings_path).unwrap(), "OTHER"),
        Some("kept")
    );
    restore_claude_provider(DEFAULT_URL).unwrap();
    assert_eq!(
        claude_settings_base_url().as_deref(),
        Some("https://api.anthropic.com")
    );
    assert!(!backup_path(&settings_path).exists());
}

#[test]
fn claude_setup_snapshot_restores_settings_and_backup_exactly() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let settings = claude_settings_path().unwrap();
    let backup = backup_path(&settings);
    fs::create_dir_all(settings.parent().unwrap()).unwrap();
    let original_settings = br#"{"env":{"ANTHROPIC_BASE_URL":"https://original"}}"#;
    let original_backup = br#"{"env":{"ANTHROPIC_BASE_URL":"https://backup"}}"#;
    fs::write(&settings, original_settings).unwrap();
    fs::write(&backup, original_backup).unwrap();
    let snapshot = snapshot_claude_setup().unwrap();

    fs::write(&settings, b"replacement-settings").unwrap();
    fs::remove_file(&backup).unwrap();
    restore_claude_setup(&snapshot).unwrap();

    assert_eq!(fs::read(settings).unwrap(), original_settings);
    assert_eq!(fs::read(backup).unwrap(), original_backup);

    fs::remove_file(claude_settings_path().unwrap()).unwrap();
    fs::remove_file(backup_path(&claude_settings_path().unwrap())).unwrap();
    let absent = snapshot_claude_setup().unwrap();
    fs::write(claude_settings_path().unwrap(), b"created").unwrap();
    fs::write(backup_path(&claude_settings_path().unwrap()), b"created").unwrap();
    restore_claude_setup(&absent).unwrap();
    assert!(!claude_settings_path().unwrap().exists());
    assert!(!backup_path(&claude_settings_path().unwrap()).exists());
}

#[test]
fn claude_provider_restore_noops_without_matching_backup_or_managed_value() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let settings_path = dir.path().join(".claude").join("settings.json");
    fs::create_dir_all(settings_path.parent().unwrap()).unwrap();
    fs::write(
        &settings_path,
        serde_json::to_vec_pretty(&json!({
            "env": { "ANTHROPIC_BASE_URL": "https://custom.example" }
        }))
        .unwrap(),
    )
    .unwrap();

    restore_claude_provider(DEFAULT_URL).unwrap();
    assert_eq!(
        claude_settings_base_url().as_deref(),
        Some("https://custom.example")
    );

    backup_claude_settings(&settings_path, false).unwrap();
    restore_claude_provider(DEFAULT_URL).unwrap();
    assert_eq!(
        claude_settings_base_url().as_deref(),
        Some("https://custom.example")
    );
    assert!(backup_path(&settings_path).exists());
}

#[test]
fn claude_provider_errors_for_non_object_env_and_restore_env_type_mismatch() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let settings_path = dir.path().join(".claude").join("settings.json");
    fs::create_dir_all(settings_path.parent().unwrap()).unwrap();
    fs::write(&settings_path, r#"{"env": "bad"}"#).unwrap();

    assert!(
        enable_claude_provider(DEFAULT_URL)
            .unwrap_err()
            .contains("non-object env field")
    );

    let mut value = json!("bad");
    assert!(
        remove_json_env_string(&mut value, "ANTHROPIC_BASE_URL")
            .unwrap_err()
            .contains("must be a JSON object")
    );
    let mut value = json!({"env": "bad"});
    assert!(
        remove_json_env_string(&mut value, "ANTHROPIC_BASE_URL")
            .unwrap_err()
            .contains("env field")
    );
    let mut value = json!({"env": "bad"});
    assert!(
        restore_json_env_value(
            &mut value,
            &json!({"env": {"ANTHROPIC_BASE_URL": DEFAULT_URL}}),
            "ANTHROPIC_BASE_URL",
        )
        .unwrap_err()
        .contains("env field")
    );
}

#[test]
fn claude_backup_bootstraps_missing_settings_and_replaces_stale_backup() {
    let dir = tempdir().unwrap();
    let settings_path = dir.path().join(".claude").join("settings.json");
    let backup = backup_path(&settings_path);
    backup_claude_settings(&settings_path, false).unwrap();
    assert_eq!(fs::read_to_string(&backup).unwrap(), "{}\n");
    fs::write(&settings_path, r#"{"env":{"ANTHROPIC_BASE_URL":"new"}}"#).unwrap();
    backup_claude_settings(&settings_path, false).unwrap();
    assert_eq!(fs::read_to_string(&backup).unwrap(), "{}\n");
    backup_claude_settings(&settings_path, true).unwrap();
    assert!(
        fs::read_to_string(&backup)
            .unwrap()
            .contains("ANTHROPIC_BASE_URL")
    );
}

#[test]
fn plugin_host_entrypoints_reject_unsupported_agents_and_report_json() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let settings_path = dir.path().join(".claude").join("settings.json");
    fs::create_dir_all(settings_path.parent().unwrap()).unwrap();
    fs::write(
        &settings_path,
        serde_json::to_vec_pretty(&json!({
            "env": { "ANTHROPIC_BASE_URL": DEFAULT_URL }
        }))
        .unwrap(),
    )
    .unwrap();
    let plugin_root = dir.path().join("plugin");
    let plugin_hooks = plugin_root.join("hooks").join("hooks.json");
    fs::create_dir_all(plugin_hooks.parent().unwrap()).unwrap();
    fs::write(
        &plugin_hooks,
        serde_json::to_vec_pretty(&json!({
            "hooks": {
                "SessionStart": [{
                    "hooks": [{
                        "type": "command",
                        "command": expected_plugin_command(),
                        "timeout": 30
                    }]
                }]
            }
        }))
        .unwrap(),
    )
    .unwrap();

    let report = doctor_plugin_json(CodingAgent::ClaudeCode, DEFAULT_URL, &plugin_root).unwrap();
    assert_eq!(report["sidecar_health"], json!("not_running_mcp_start"));
    assert_eq!(report["checks"]["claude_provider_routing"], json!(true));
    let codex_report = doctor_plugin_json(CodingAgent::Codex, DEFAULT_URL, &plugin_root).unwrap();
    assert_eq!(
        codex_report["sidecar_health"],
        json!("not_running_mcp_start")
    );
    assert_eq!(codex_report["checks"]["codex_provider_alias"], json!(false));
    assert_eq!(codex_report["checks"]["codex_hooks"], json!(false));
    assert!(
        doctor_plugin_json(CodingAgent::Hermes, DEFAULT_URL, &plugin_root)
            .unwrap_err()
            .contains("supports claude and codex")
    );
    assert!(
        doctor_plugin(CodingAgent::Hermes, DEFAULT_URL, &plugin_root)
            .unwrap_err()
            .contains("supports claude and codex")
    );
    assert!(
        doctor_plugin(CodingAgent::Codex, DEFAULT_URL, &plugin_root)
            .unwrap_err()
            .contains("codex plugin doctor checks failed")
    );
}

fn event_contains_command(config: &Value, event: &str, command: &str) -> bool {
    config
        .get("hooks")
        .and_then(Value::as_object)
        .and_then(|hooks| hooks.get(event))
        .and_then(Value::as_array)
        .is_some_and(|groups| {
            groups.iter().any(|group| {
                group
                    .get("hooks")
                    .and_then(Value::as_array)
                    .is_some_and(|hooks| {
                        hooks.iter().any(|hook| {
                            hook.get("command").and_then(Value::as_str) == Some(command)
                        })
                    })
            })
        })
}
