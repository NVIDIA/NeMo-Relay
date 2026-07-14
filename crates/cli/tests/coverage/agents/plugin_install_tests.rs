// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, VecDeque};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

use serde_json::json;
use tempfile::tempdir;

use super::host::{
    CommandOutput, HostRegistrationReport, format_command, host_registration_report,
    require_host_cli, require_relay, run_capture_command, run_command, run_path_command,
    validate_host_registration, validate_host_version, validate_relay_hook_forward,
    validate_relay_mcp,
};
use super::*;
use crate::agents::CodingAgent;
use crate::agents::strip_windows_verbatim_prefix;

const OPERATION_LOCK_HELPER_DIR_ENV: &str = "NEMO_RELAY_TEST_OPERATION_LOCK_DIR";
const OPERATION_LOCK_HELPER_GLOBAL_DIR_ENV: &str = "NEMO_RELAY_TEST_OPERATION_LOCK_GLOBAL_DIR";
const GENERATION_LOCK_HELPER_PATH_ENV: &str = "NEMO_RELAY_TEST_GENERATION_LOCK_PATH";
const LOCK_HELPER_READY_ENV: &str = "NEMO_RELAY_TEST_LOCK_READY";
const LOCK_HELPER_RELEASE_ENV: &str = "NEMO_RELAY_TEST_LOCK_RELEASE";
const TEST_GENERATION_TOKEN: &str = "test-generation";

fn force_snapshot_with_backups(
    backup_marketplace_root: PathBuf,
    backup_plugin_root: Option<PathBuf>,
) -> ForceInstallSnapshot {
    ForceInstallSnapshot {
        state_bytes: None,
        setup_snapshot: None,
        original_marketplace_root: PathBuf::from("original-marketplace"),
        original_plugin_root: PathBuf::from("separate-original-plugin"),
        original_generation_fence: PathBuf::from("original-generation"),
        plugin_registered: false,
        marketplace_registered: false,
        backup_marketplace_root,
        backup_plugin_root,
        marketplace_moved: true,
        plugin_moved: true,
        replacement_promoted: false,
        generation_retirement: None,
    }
}

fn plugin_install_env_lock() -> &'static Mutex<()> {
    &crate::test_support::ENV_TEST_LOCK
}

#[test]
fn windows_verbatim_relay_paths_are_normalized_for_mcp_config() {
    let normalize = |path: &str| {
        let encoded = path.encode_utf16().collect::<Vec<_>>();
        strip_windows_verbatim_prefix(&encoded)
            .map(|normalized| String::from_utf16(&normalized).unwrap())
    };

    assert_eq!(
        normalize(r"\\?\C:\Program Files\NVIDIA\nemo-relay.exe"),
        Some(r"C:\Program Files\NVIDIA\nemo-relay.exe".into())
    );
    assert_eq!(
        normalize(r"\\?\UNC\server\share\nemo-relay.exe"),
        Some(r"\\server\share\nemo-relay.exe".into())
    );
    assert_eq!(normalize(r"C:\nemo-relay.exe"), None);
}

#[test]
fn readiness_worker_returns_a_report_and_handles_channel_disconnects() {
    let dir = tempdir().unwrap();
    let readiness = collect_marketplace_readiness(
        CodingAgent::Codex,
        &options(dir.path()),
        &MockRunner::default(),
    );
    assert_eq!(readiness.host, "codex");
    assert!(!readiness.checks.is_empty());

    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    drop(sender);
    let readiness = crate::agents::receive_integration_readiness_for_test(
        CodingAgent::ClaudeCode,
        dir.path().join("claude-state.json"),
        receiver,
        dir.path(),
        Duration::from_secs(1),
    );
    assert!(!readiness.ok());
    assert!(
        readiness.checks[0]
            .details
            .contains("collector stopped unexpectedly")
    );

    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    drop(sender);
    let hermes = crate::agents::receive_integration_readiness_for_test(
        CodingAgent::Hermes,
        dir.path().join("config.yaml"),
        receiver,
        dir.path(),
        Duration::from_secs(1),
    );
    assert!(hermes.marketplace.is_none());
    assert!(hermes.plugin.is_none());
    assert!(!hermes.ok());
}

#[test]
fn committed_force_snapshot_removes_all_backup_trees_best_effort() {
    let dir = tempdir().unwrap();
    let marketplace = dir.path().join("marketplace-backup");
    let plugin = dir.path().join("plugin-backup");
    std::fs::create_dir_all(&marketplace).unwrap();
    std::fs::create_dir_all(&plugin).unwrap();

    force_snapshot_with_backups(marketplace.clone(), Some(plugin.clone()))
        .commit(&dir.path().join("replacement.lock"));

    assert!(!marketplace.exists());
    assert!(!plugin.exists());

    let missing_marketplace = dir.path().join("missing-marketplace");
    let missing_plugin = dir.path().join("missing-plugin");
    force_snapshot_with_backups(missing_marketplace, Some(missing_plugin))
        .commit(&dir.path().join("replacement.lock"));

    let marketplace_file = dir.path().join("marketplace-file");
    let plugin_file = dir.path().join("plugin-file");
    std::fs::write(&marketplace_file, "file").unwrap();
    std::fs::write(&plugin_file, "file").unwrap();
    force_snapshot_with_backups(marketplace_file.clone(), Some(plugin_file.clone()))
        .commit(&dir.path().join("replacement.lock"));
    assert!(marketplace_file.exists());
    assert!(plugin_file.exists());
}

#[test]
fn dry_run_cleanup_and_rollback_cover_absent_install_state() {
    let dir = tempdir().unwrap();
    let mut dry_run = options(dir.path());
    dry_run.dry_run = true;
    let layout = PluginLayout::new(CodingAgent::ClaudeCode, dir.path());
    let runner = MockRunner::default();
    let setup_runner = MockSetupRunner::default();

    force_cleanup_existing_install(
        CodingAgent::ClaudeCode,
        &layout,
        &dry_run,
        &runner,
        &setup_runner,
    )
    .unwrap();
    rollback_install(
        CodingAgent::ClaudeCode,
        &layout,
        HostRegistrationProgress::default(),
        false,
        &dry_run,
        &runner,
        &setup_runner,
    )
    .unwrap();
}

#[test]
fn staged_marketplace_promotion_reports_the_source_and_target() {
    let dir = tempdir().unwrap();
    let staged_parent = dir.path().join("stage");
    let target_parent = dir.path().join("target");
    let staged = StagedPluginMarketplace {
        layout: PluginLayout::new(CodingAgent::Codex, &staged_parent),
        parent: staged_parent,
        generation_lock_created: false,
    };
    let target = PluginLayout::new(CodingAgent::Codex, &target_parent);

    let error = staged.promote(&target).unwrap_err();

    assert!(error.contains("failed to promote staged marketplace"));
    assert!(
        error.contains(&staged.layout.marketplace_root.display().to_string()),
        "{error}"
    );
    assert!(
        error.contains(&target.marketplace_root.display().to_string()),
        "{error}"
    );
}

#[test]
fn replacement_generation_guard_removes_an_owned_lock_after_marker_removal() {
    let dir = tempdir().unwrap();
    let marker = dir.path().join("generation-marker");
    let lock = dir.path().join("generation.lock");
    crate::installation::generation::write_new_generation_with_token_at(&marker, &lock).unwrap();
    let guard =
        acquire_replacement_generation_lock(CodingAgent::Codex, &marker, &lock, true).unwrap();

    std::fs::remove_file(marker).unwrap();
    drop(guard);

    assert!(!lock.exists());
}

#[test]
fn replacement_generation_guard_retains_its_lock_when_marker_state_is_uncertain() {
    let dir = tempdir().unwrap();
    let marker = dir.path().join("generation-marker");
    let lock = dir.path().join("generation.lock");
    crate::installation::generation::write_new_generation_with_token_at(&marker, &lock).unwrap();
    let guard =
        acquire_replacement_generation_lock(CodingAgent::Codex, &marker, &lock, true).unwrap();

    std::fs::remove_file(&marker).unwrap();
    std::fs::create_dir(&marker).unwrap();
    drop(guard);

    assert!(lock.exists());
}

#[test]
fn codex_plugin_requires_version_with_complete_hook_support() {
    let dir = tempdir().unwrap();
    let normal = options(dir.path());
    let supported = MockRunner::default()
        .with_executable("codex", "/bin/codex")
        .with_capture_output("/bin/codex --version", "codex-cli 0.143.0\n");
    validate_host_version(CodingAgent::Codex, &normal, &supported).unwrap();

    let old = MockRunner::default()
        .with_executable("codex", "/bin/codex")
        .with_capture_output("/bin/codex --version", "codex-cli 0.142.9\n");
    assert!(
        validate_host_version(CodingAgent::Codex, &normal, &old)
            .unwrap_err()
            .contains("requires codex-cli 0.143.0")
    );

    let invalid = MockRunner::default()
        .with_executable("codex", "/bin/codex")
        .with_capture_output("/bin/codex --version", "codex nightly\n");
    assert!(
        validate_host_version(CodingAgent::Codex, &normal, &invalid)
            .unwrap_err()
            .contains("could not parse")
    );

    let prerelease = MockRunner::default()
        .with_executable("codex", "/bin/codex")
        .with_capture_output("/bin/codex --version", "codex-cli 0.143.0-alpha.1\n");
    assert!(
        validate_host_version(CodingAgent::Codex, &normal, &prerelease)
            .unwrap_err()
            .contains("codex-cli 0.143.0-alpha.1 is unsupported")
    );

    for malformed in [
        "codex-cli 0.143\n",
        "codex-cli v0.143.0\n",
        "warning 1.2.3\ncodex-cli 0.143.0\n",
    ] {
        let runner = MockRunner::default()
            .with_executable("codex", "/bin/codex")
            .with_capture_output("/bin/codex --version", malformed);
        assert!(
            validate_host_version(CodingAgent::Codex, &normal, &runner)
                .unwrap_err()
                .contains("could not parse"),
            "unexpectedly parsed {malformed:?}"
        );
    }
}

#[test]
fn claude_plugin_requires_version_with_always_load_support() {
    let dir = tempdir().unwrap();
    let normal = options(dir.path());
    let supported = MockRunner::default()
        .with_executable("claude", "/bin/claude")
        .with_capture_output("/bin/claude --version", "2.1.121 (Claude Code)\n");
    validate_host_version(CodingAgent::ClaudeCode, &normal, &supported).unwrap();

    for unsupported in ["2.1.120 (Claude Code)\n", "2.1.121-beta (Claude Code)\n"] {
        let runner = MockRunner::default()
            .with_executable("claude", "/bin/claude")
            .with_capture_output("/bin/claude --version", unsupported);
        assert!(
            validate_host_version(CodingAgent::ClaudeCode, &normal, &runner)
                .unwrap_err()
                .contains("requires Claude Code 2.1.121"),
            "unexpectedly accepted {unsupported:?}"
        );
    }

    for malformed in ["Claude Code 2.1.121\n", "2.1\n", "warning\n2.1.121\n"] {
        let runner = MockRunner::default()
            .with_executable("claude", "/bin/claude")
            .with_capture_output("/bin/claude --version", malformed);
        assert!(
            validate_host_version(CodingAgent::ClaudeCode, &normal, &runner)
                .unwrap_err()
                .contains("could not parse"),
            "unexpectedly parsed {malformed:?}"
        );
    }
}

struct HomeScope<'a> {
    _guard: std::sync::MutexGuard<'a, ()>,
    prev_home: Option<std::ffi::OsString>,
    prev_userprofile: Option<std::ffi::OsString>,
    prev_codex_home: Option<std::ffi::OsString>,
    prev_hermes_home: Option<std::ffi::OsString>,
}

impl<'a> HomeScope<'a> {
    fn enter(path: &Path) -> Self {
        let guard = plugin_install_env_lock()
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let prev_home = std::env::var_os("HOME");
        let prev_userprofile = std::env::var_os("USERPROFILE");
        let prev_codex_home = std::env::var_os("CODEX_HOME");
        let prev_hermes_home = std::env::var_os("HERMES_HOME");
        // SAFETY: This test holds a process-wide mutex for the lifetime of the env override.
        unsafe {
            std::env::set_var("HOME", path);
            std::env::remove_var("USERPROFILE");
            std::env::remove_var("CODEX_HOME");
            std::env::remove_var("HERMES_HOME");
        }
        Self {
            _guard: guard,
            prev_home,
            prev_userprofile,
            prev_codex_home,
            prev_hermes_home,
        }
    }

    fn without_home() -> Self {
        let guard = plugin_install_env_lock()
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let prev_home = std::env::var_os("HOME");
        let prev_userprofile = std::env::var_os("USERPROFILE");
        let prev_codex_home = std::env::var_os("CODEX_HOME");
        let prev_hermes_home = std::env::var_os("HERMES_HOME");
        // SAFETY: This test holds a process-wide mutex for the lifetime of the env override.
        unsafe {
            std::env::remove_var("HOME");
            std::env::remove_var("USERPROFILE");
            std::env::remove_var("CODEX_HOME");
            std::env::remove_var("HERMES_HOME");
        }
        Self {
            _guard: guard,
            prev_home,
            prev_userprofile,
            prev_codex_home,
            prev_hermes_home,
        }
    }
}

impl Drop for HomeScope<'_> {
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
            match self.prev_hermes_home.take() {
                Some(value) => std::env::set_var("HERMES_HOME", value),
                None => std::env::remove_var("HERMES_HOME"),
            }
        }
    }
}

#[test]
fn plugin_operation_lock_directory_requires_a_user_home() {
    let _home = HomeScope::without_home();

    let error = default_operation_lock_dir().unwrap_err();

    assert!(error.contains("set HOME or USERPROFILE"), "{error}");
}

struct PathScope<'a> {
    _guard: std::sync::MutexGuard<'a, ()>,
    previous: Option<OsString>,
    previous_home: Option<OsString>,
    previous_codex_home: Option<OsString>,
    previous_hermes_home: Option<OsString>,
}

impl<'a> PathScope<'a> {
    fn set_isolated(path: &Path, home: &Path) -> Self {
        let guard = plugin_install_env_lock()
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let previous = std::env::var_os("PATH");
        let previous_home = std::env::var_os("HOME");
        let previous_codex_home = std::env::var_os("CODEX_HOME");
        let previous_hermes_home = std::env::var_os("HERMES_HOME");
        // SAFETY: This test holds the process-wide environment mutex for the override lifetime.
        unsafe {
            std::env::set_var("PATH", path);
            std::env::set_var("HOME", home);
            std::env::remove_var("CODEX_HOME");
            std::env::remove_var("HERMES_HOME");
        }
        Self {
            _guard: guard,
            previous,
            previous_home,
            previous_codex_home,
            previous_hermes_home,
        }
    }
}

impl Drop for PathScope<'_> {
    fn drop(&mut self) {
        // SAFETY: This restores PATH while the process-wide environment mutex is still held.
        unsafe {
            match self.previous.take() {
                Some(value) => std::env::set_var("PATH", value),
                None => std::env::remove_var("PATH"),
            }
            match self.previous_home.take() {
                Some(value) => std::env::set_var("HOME", value),
                None => std::env::remove_var("HOME"),
            }
            match self.previous_codex_home.take() {
                Some(value) => std::env::set_var("CODEX_HOME", value),
                None => std::env::remove_var("CODEX_HOME"),
            }
            match self.previous_hermes_home.take() {
                Some(value) => std::env::set_var("HERMES_HOME", value),
                None => std::env::remove_var("HERMES_HOME"),
            }
        }
    }
}

#[derive(Default)]
struct MockRunner {
    current_executable: Option<PathBuf>,
    executables: HashMap<String, PathBuf>,
    commands: RefCell<Vec<String>>,
    quiet_commands: RefCell<Vec<String>>,
    capture_commands: RefCell<Vec<String>>,
    capture_outputs: HashMap<String, CommandOutput>,
    capture_output_sequences: RefCell<HashMap<String, VecDeque<CommandOutput>>>,
    failing_suffix: Option<String>,
    failing_suffixes: Vec<String>,
    failing_quiet_suffix: Option<String>,
}

impl MockRunner {
    fn with_current_executable(mut self, path: &str) -> Self {
        self.current_executable = Some(PathBuf::from(path));
        self
    }

    fn with_executable(mut self, name: &str, path: &str) -> Self {
        self.executables.insert(name.into(), PathBuf::from(path));
        self
    }

    fn with_capture_output(mut self, command: &str, stdout: impl Into<String>) -> Self {
        self.capture_outputs
            .insert(command.into(), CommandOutput::success(stdout.into()));
        self
    }

    fn with_capture_status(
        mut self,
        command: &str,
        status: i32,
        stdout: impl Into<String>,
        stderr: impl Into<String>,
    ) -> Self {
        self.capture_outputs.insert(
            command.into(),
            CommandOutput {
                status,
                stdout: stdout.into(),
                stderr: stderr.into(),
            },
        );
        self
    }

    fn with_codex_registration(mut self, plugin: bool, marketplace: bool) -> Self {
        let plugin_output = if plugin {
            "nemo-relay-plugin@nemo-relay-local installed, enabled\n"
        } else {
            ""
        };
        let marketplace_output = if marketplace {
            "nemo-relay-local /tmp/nemo-relay-local\n"
        } else {
            ""
        };
        self.capture_outputs.insert(
            "/bin/codex plugin list".into(),
            CommandOutput::success(plugin_output.into()),
        );
        self.capture_outputs.insert(
            "/bin/codex plugin marketplace list".into(),
            CommandOutput::success(marketplace_output.into()),
        );
        self
    }

    fn with_claude_registration(mut self, plugin: bool, marketplace: bool) -> Self {
        let plugins = if plugin {
            json!([{ "id": "nemo-relay-plugin@nemo-relay-local" }])
        } else {
            json!([])
        };
        let marketplaces = if marketplace {
            json!([{ "name": "nemo-relay-local" }])
        } else {
            json!([])
        };
        self.capture_outputs.insert(
            "/bin/claude plugin list --json".into(),
            CommandOutput::success(plugins.to_string()),
        );
        self.capture_outputs.insert(
            "/bin/claude plugin marketplace list --json".into(),
            CommandOutput::success(marketplaces.to_string()),
        );
        self
    }

    fn with_codex_registration_sequence(mut self, states: &[(bool, bool)]) -> Self {
        let plugin_outputs = states
            .iter()
            .map(|(plugin, _)| {
                CommandOutput::success(
                    plugin
                        .then_some("nemo-relay-plugin@nemo-relay-local installed, enabled\n")
                        .unwrap_or_default()
                        .into(),
                )
            })
            .collect();
        let marketplace_outputs = states
            .iter()
            .map(|(_, marketplace)| {
                CommandOutput::success(
                    marketplace
                        .then_some("nemo-relay-local /tmp/nemo-relay-local\n")
                        .unwrap_or_default()
                        .into(),
                )
            })
            .collect();
        self.capture_output_sequences
            .get_mut()
            .insert("/bin/codex plugin list".into(), plugin_outputs);
        self.capture_output_sequences.get_mut().insert(
            "/bin/codex plugin marketplace list".into(),
            marketplace_outputs,
        );
        self
    }

    fn commands(&self) -> Vec<String> {
        self.commands.borrow().clone()
    }

    fn quiet_commands(&self) -> Vec<String> {
        self.quiet_commands.borrow().clone()
    }

    fn capture_commands(&self) -> Vec<String> {
        self.capture_commands.borrow().clone()
    }
}

impl CommandRunner for MockRunner {
    fn current_executable(&self) -> Result<PathBuf, String> {
        self.current_executable
            .clone()
            .or_else(|| self.executables.get(RELAY_COMMAND).cloned())
            .ok_or_else(|| "failed to resolve current nemo-relay executable".into())
    }

    fn resolve_executable(&self, command: &str) -> Result<Option<PathBuf>, String> {
        Ok(self.executables.get(command).cloned())
    }

    fn run(&self, program: &Path, args: &[String]) -> Result<i32, String> {
        let rendered = format!(
            "{} {}",
            program.display(),
            args.iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
                .join(" ")
        );
        self.commands.borrow_mut().push(rendered.clone());
        Ok(
            if command_matches_suffix(&rendered, self.failing_suffix.as_deref())
                || self
                    .failing_suffixes
                    .iter()
                    .any(|suffix| rendered.ends_with(suffix))
            {
                1
            } else {
                0
            },
        )
    }

    fn run_quiet(&self, program: &Path, args: &[String]) -> Result<i32, String> {
        let rendered = format!(
            "{} {}",
            program.display(),
            args.iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
                .join(" ")
        );
        self.quiet_commands.borrow_mut().push(rendered.clone());
        Ok(
            if command_matches_suffix(&rendered, self.failing_quiet_suffix.as_deref()) {
                1
            } else {
                0
            },
        )
    }

    fn run_capture(&self, program: &Path, args: &[String]) -> Result<CommandOutput, String> {
        let rendered = format!(
            "{} {}",
            program.display(),
            args.iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
                .join(" ")
        );
        self.capture_commands.borrow_mut().push(rendered.clone());
        if let Some(output) = self
            .capture_output_sequences
            .borrow_mut()
            .get_mut(&rendered)
            .and_then(VecDeque::pop_front)
        {
            return Ok(output);
        }
        Ok(self
            .capture_outputs
            .get(&rendered)
            .cloned()
            .unwrap_or_else(|| {
                if rendered.ends_with("codex --version") {
                    CommandOutput::success("codex-cli 0.143.0\n".into())
                } else if rendered.ends_with("claude --version") {
                    CommandOutput::success("2.1.121 (Claude Code)\n".into())
                } else if rendered.ends_with("claude plugin list --json")
                    || rendered.ends_with("claude plugin marketplace list --json")
                {
                    CommandOutput::success("[]\n".into())
                } else {
                    CommandOutput::success(String::new())
                }
            }))
    }
}

fn command_matches_suffix(command: &str, suffix: Option<&str>) -> bool {
    suffix.is_some_and(|suffix| command.ends_with(suffix))
}

#[derive(Default)]
struct MockSetupRunner {
    calls: RefCell<Vec<String>>,
    doctor_roots: RefCell<Vec<PathBuf>>,
    failing_call: Option<String>,
}

struct BlockingRefreshFailure {
    entered: std::sync::mpsc::Sender<()>,
    continue_refresh: std::sync::mpsc::Receiver<()>,
}

struct FailStateWriteAfterRefresh {
    state_path: PathBuf,
    injected: Cell<bool>,
}

impl PluginSetupRunner for FailStateWriteAfterRefresh {
    fn snapshot(&self, _host_arg: &str) -> Result<Option<PluginSetupSnapshot>, String> {
        Ok(Some(PluginSetupSnapshot::Mock))
    }

    fn restore_snapshot(&self, _snapshot: &PluginSetupSnapshot) -> Result<(), String> {
        Ok(())
    }

    fn refresh_gateway(&self) -> Result<(), String> {
        if !self.injected.replace(true) {
            crate::filesystem::fail_next_atomic_write(&self.state_path);
        }
        Ok(())
    }

    fn setup(
        &self,
        _host_arg: &str,
        _gateway_url: &str,
        _plugin_root: &Path,
    ) -> Result<(), String> {
        Ok(())
    }

    fn uninstall(
        &self,
        _host_arg: &str,
        _gateway_url: &str,
        _plugin_root: &Path,
    ) -> Result<(), String> {
        Ok(())
    }

    fn doctor(
        &self,
        _host_arg: &str,
        _gateway_url: &str,
        _plugin_root: &Path,
    ) -> Result<(), String> {
        Ok(())
    }

    fn doctor_json(
        &self,
        _host_arg: &str,
        _gateway_url: &str,
        _plugin_root: &Path,
    ) -> Result<serde_json::Value, String> {
        Ok(json!({"ok": true, "checks": {}}))
    }
}

impl PluginSetupRunner for BlockingRefreshFailure {
    fn snapshot(&self, _host_arg: &str) -> Result<Option<PluginSetupSnapshot>, String> {
        Ok(Some(PluginSetupSnapshot::Mock))
    }

    fn restore_snapshot(&self, _snapshot: &PluginSetupSnapshot) -> Result<(), String> {
        Ok(())
    }

    fn refresh_gateway(&self) -> Result<(), String> {
        self.entered.send(()).unwrap();
        self.continue_refresh.recv().unwrap();
        Err("refresh gateway failed".into())
    }

    fn setup(
        &self,
        _host_arg: &str,
        _gateway_url: &str,
        _plugin_root: &Path,
    ) -> Result<(), String> {
        Ok(())
    }

    fn uninstall(
        &self,
        _host_arg: &str,
        _gateway_url: &str,
        _plugin_root: &Path,
    ) -> Result<(), String> {
        Ok(())
    }

    fn doctor(
        &self,
        _host_arg: &str,
        _gateway_url: &str,
        _plugin_root: &Path,
    ) -> Result<(), String> {
        Ok(())
    }

    fn doctor_json(
        &self,
        _host_arg: &str,
        _gateway_url: &str,
        _plugin_root: &Path,
    ) -> Result<serde_json::Value, String> {
        Ok(json!({"ok": true, "checks": {}}))
    }
}

impl MockSetupRunner {
    fn calls(&self) -> Vec<String> {
        self.calls.borrow().clone()
    }

    fn doctor_roots(&self) -> Vec<PathBuf> {
        self.doctor_roots.borrow().clone()
    }
}

impl PluginSetupRunner for MockSetupRunner {
    fn snapshot(&self, host_arg: &str) -> Result<Option<PluginSetupSnapshot>, String> {
        self.record(format!("snapshot {host_arg}"))?;
        Ok(Some(PluginSetupSnapshot::Mock))
    }

    fn restore_snapshot(&self, _snapshot: &PluginSetupSnapshot) -> Result<(), String> {
        self.record("restore snapshot".into())
    }

    fn refresh_gateway(&self) -> Result<(), String> {
        self.record("refresh gateway".into())
    }

    fn setup(&self, host_arg: &str, gateway_url: &str, _plugin_root: &Path) -> Result<(), String> {
        self.record(format!("setup {host_arg} {gateway_url}"))
    }

    fn uninstall(
        &self,
        host_arg: &str,
        gateway_url: &str,
        _plugin_root: &Path,
    ) -> Result<(), String> {
        self.record(format!("uninstall {host_arg} {gateway_url}"))
    }

    fn doctor(&self, host_arg: &str, gateway_url: &str, _plugin_root: &Path) -> Result<(), String> {
        self.record(format!("doctor {host_arg} {gateway_url}"))
    }

    fn doctor_json(
        &self,
        host_arg: &str,
        gateway_url: &str,
        plugin_root: &Path,
    ) -> Result<serde_json::Value, String> {
        self.doctor_roots
            .borrow_mut()
            .push(plugin_root.to_path_buf());
        self.record(format!("doctor-json {host_arg} {gateway_url}"))?;
        Ok(json!({
            "ok": true,
            "checks": {}
        }))
    }
}

impl MockSetupRunner {
    fn record(&self, call: String) -> Result<(), String> {
        self.calls.borrow_mut().push(call.clone());
        if self.failing_call.as_deref() == Some(call.as_str()) {
            Err(format!("{call} failed"))
        } else {
            Ok(())
        }
    }
}

fn options(dir: &Path) -> PluginInstallOptions {
    PluginInstallOptions {
        install_dir: dir.to_path_buf(),
        operation_lock_dir: dir.join("operation-locks"),
        force: false,
        dry_run: false,
        skip_doctor: true,
    }
}

fn relay_validation_command() -> String {
    "/bin/nemo-relay hook-forward --help".into()
}

fn relay_mcp_validation_command() -> String {
    "/bin/nemo-relay mcp --help".into()
}

fn write_installed_state(host: CodingAgent, dir: &Path) {
    let layout = PluginLayout::new(host, dir);
    write_plugin_marketplace(host, &layout, Path::new("/bin/nemo-relay"), &options(dir)).unwrap();
    write_state(&layout, &options(dir)).unwrap();
    mark_plugin_setup_installed(host, &layout, &options(dir)).unwrap();
}

#[cfg(windows)]
fn replace_generation_with_legacy_marker(layout: &PluginLayout) -> (String, PathBuf) {
    let token = {
        let generation = InstallGeneration::capture(layout.generation_fence.clone()).unwrap();
        generation.token().to_owned()
    };
    std::fs::remove_file(&layout.generation_lock).unwrap();
    crate::installation::generation::write_legacy_generation(&layout.generation_fence, &token)
        .unwrap();
    let mut lock_path = layout.generation_fence.as_os_str().to_os_string();
    lock_path.push(".lock");
    (token, PathBuf::from(lock_path))
}

fn write_relocated_codex_install(selected_dir: &Path, relocated_dir: &Path) -> PluginLayout {
    let relocated = PluginLayout::new(CodingAgent::Codex, relocated_dir);
    write_plugin_marketplace(
        CodingAgent::Codex,
        &relocated,
        Path::new("/bin/nemo-relay"),
        &options(selected_dir),
    )
    .unwrap();
    write_state_for_host(
        CodingAgent::Codex,
        &PluginState {
            marketplace_root: relocated.marketplace_root.clone(),
            plugin_root: relocated.plugin_root.clone(),
            host_plugin_removed: false,
            host_marketplace_removed: false,
            plugin_setup_installed: true,
        },
        selected_dir,
        &options(selected_dir),
    )
    .unwrap();
    relocated
}

fn assert_no_install_stage(dir: &Path) {
    assert!(std::fs::read_dir(dir).unwrap().all(|entry| {
        !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .contains("install-stage")
    }));
}

fn assert_no_force_replacement_residue(dir: &Path) {
    assert!(std::fs::read_dir(dir).unwrap().all(|entry| {
        let name = entry.unwrap().file_name();
        let name = name.to_string_lossy();
        !name.contains("install-stage")
            && !name.contains("marketplace-backup")
            && !name.contains("plugin-backup")
    }));
}

fn assert_actionable_generation_error(error: &str, cause: &str) {
    assert!(error.contains(cause), "{error}");
    assert!(error.contains("close all Codex clients"), "{error}");
    assert!(
        error.contains("codex plugin remove nemo-relay-plugin@nemo-relay-local"),
        "{error}"
    );
    assert!(
        error.contains("codex plugin marketplace remove nemo-relay-local"),
        "{error}"
    );
    assert!(error.contains("stale marketplace and state"), "{error}");
    assert!(
        error.contains("nemo-relay install codex --force"),
        "{error}"
    );
}

fn corrupt_generation_fence(path: &Path, corruption: &str) {
    match corruption {
        "empty" => std::fs::write(path, b"").unwrap(),
        "oversized" => std::fs::write(path, vec![b'x'; 129]).unwrap(),
        "unreadable" => {
            std::fs::remove_file(path).unwrap();
            std::fs::create_dir(path).unwrap();
        }
        _ => unreachable!(),
    }
}

struct CrossProcessLockHolder {
    child: Option<Child>,
    release: PathBuf,
}

impl CrossProcessLockHolder {
    fn spawn(
        env_name: &str,
        target: &Path,
        global_lock_dir: Option<&Path>,
        synchronization_dir: &Path,
    ) -> Self {
        let ready = synchronization_dir.join("ready");
        let release = synchronization_dir.join("release");
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                "installation::marketplace::tests::cross_process_lock_holder",
                "--nocapture",
            ])
            .env(env_name, target)
            .env(LOCK_HELPER_READY_ENV, &ready)
            .env(LOCK_HELPER_RELEASE_ENV, &release)
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        if let Some(global_lock_dir) = global_lock_dir {
            command.env(OPERATION_LOCK_HELPER_GLOBAL_DIR_ENV, global_lock_dir);
        }
        let mut child = command.spawn().unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if ready.exists() {
                break;
            }
            if let Some(status) = child.try_wait().unwrap() {
                panic!("cross-process lock holder exited before acquiring its lock: {status}");
            }
            assert!(
                Instant::now() < deadline,
                "cross-process lock holder did not become ready"
            );
            thread::sleep(Duration::from_millis(10));
        }
        Self {
            child: Some(child),
            release,
        }
    }

    fn release(mut self) {
        self.finish();
    }

    fn finish(&mut self) {
        let Some(mut child) = self.child.take() else {
            return;
        };
        std::fs::write(&self.release, b"release").unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if child.try_wait().unwrap().is_some() {
                return;
            }
            if Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                panic!("cross-process lock holder did not exit after release");
            }
            thread::sleep(Duration::from_millis(10));
        }
    }
}

impl Drop for CrossProcessLockHolder {
    fn drop(&mut self) {
        if self.child.is_some() {
            self.finish();
        }
    }
}

#[test]
fn cross_process_lock_holder() {
    let ready = match std::env::var_os(LOCK_HELPER_READY_ENV) {
        Some(path) => PathBuf::from(path),
        None => return,
    };
    let release = PathBuf::from(std::env::var_os(LOCK_HELPER_RELEASE_ENV).unwrap());
    let _operation_lock;
    let _generation_retirement;
    if let Some(path) = std::env::var_os(OPERATION_LOCK_HELPER_DIR_ENV) {
        let global_lock_dir =
            PathBuf::from(std::env::var_os(OPERATION_LOCK_HELPER_GLOBAL_DIR_ENV).unwrap());
        _operation_lock = Some(
            PluginOperationLock::acquire(
                CodingAgent::Codex.install_arg(),
                &global_lock_dir,
                Path::new(&path),
                Duration::from_secs(5),
            )
            .unwrap(),
        );
        _generation_retirement = None;
    } else if let Some(path) = std::env::var_os(GENERATION_LOCK_HELPER_PATH_ENV) {
        _operation_lock = None;
        _generation_retirement = GenerationRetirement::acquire(Path::new(&path)).unwrap();
        assert!(_generation_retirement.is_some());
    } else {
        return;
    }
    std::fs::write(ready, b"ready").unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while !release.exists() {
        assert!(Instant::now() < deadline, "lock holder release timed out");
        thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn concurrent_install_install_times_out_without_mutating() {
    let dir = tempdir().unwrap();
    let synchronization = tempdir().unwrap();
    let install_options = options(dir.path());
    let holder = CrossProcessLockHolder::spawn(
        OPERATION_LOCK_HELPER_DIR_ENV,
        dir.path(),
        Some(&install_options.operation_lock_dir),
        synchronization.path(),
    );
    let runner = MockRunner::default();
    let setup_runner = MockSetupRunner::default();

    let error = install_host_with_operation_timeout(
        CodingAgent::Codex,
        &install_options,
        &runner,
        &setup_runner,
        Duration::from_millis(75),
    )
    .expect_err("contended install unexpectedly succeeded");

    assert!(error.contains("another codex plugin install or uninstall"));
    assert!(runner.commands().is_empty());
    assert!(setup_runner.calls().is_empty());
    holder.release();
}

#[test]
fn concurrent_install_uninstall_times_out_without_mutating() {
    let dir = tempdir().unwrap();
    let synchronization = tempdir().unwrap();
    let install_options = options(dir.path());
    let holder = CrossProcessLockHolder::spawn(
        OPERATION_LOCK_HELPER_DIR_ENV,
        dir.path(),
        Some(&install_options.operation_lock_dir),
        synchronization.path(),
    );
    let runner = MockRunner::default();
    let setup_runner = MockSetupRunner::default();

    let error = uninstall_host_with_operation_timeout(
        CodingAgent::Codex,
        &install_options,
        &runner,
        &setup_runner,
        Duration::from_millis(75),
    )
    .expect_err("contended uninstall unexpectedly succeeded");

    assert!(error.contains("another codex plugin install or uninstall"));
    assert!(runner.commands().is_empty());
    assert!(setup_runner.calls().is_empty());
    holder.release();
}

#[test]
fn concurrent_different_install_roots_share_the_global_host_lock() {
    let root = tempdir().unwrap();
    let first_install_dir = root.path().join("first-install");
    let second_install_dir = root.path().join("second-install");
    let global_lock_dir = root.path().join("global-operation-locks");
    let synchronization = tempdir().unwrap();
    let holder = CrossProcessLockHolder::spawn(
        OPERATION_LOCK_HELPER_DIR_ENV,
        &first_install_dir,
        Some(&global_lock_dir),
        synchronization.path(),
    );
    let runner = MockRunner::default();
    let setup_runner = MockSetupRunner::default();
    let mut second_options = options(&second_install_dir);
    second_options.operation_lock_dir = global_lock_dir;

    let install_error = install_host_with_operation_timeout(
        CodingAgent::Codex,
        &second_options,
        &runner,
        &setup_runner,
        Duration::from_millis(75),
    )
    .unwrap_err();

    assert!(install_error.contains("global lock"), "{install_error}");
    assert!(runner.commands().is_empty());
    assert!(setup_runner.calls().is_empty());
    holder.release();
}

#[test]
fn plugin_operation_lock_acquires_an_aliased_global_and_install_root_once() {
    let root = tempdir().unwrap();

    let _lock = PluginOperationLock::acquire(
        "codex",
        root.path(),
        &root.path().join("."),
        Duration::from_millis(75),
    )
    .unwrap();

    assert!(
        crate::installation::operation_lock::operation_lock_path("codex", root.path()).exists()
    );
}

#[test]
fn generation_retirement_lock_contention_is_bounded_across_processes() {
    let dir = tempdir().unwrap();
    let synchronization = tempdir().unwrap();
    let generation = dir.path().join(GENERATION_FILE_NAME);
    crate::installation::generation::write_new_generation(&generation).unwrap();
    let holder = CrossProcessLockHolder::spawn(
        GENERATION_LOCK_HELPER_PATH_ENV,
        &generation,
        None,
        synchronization.path(),
    );

    let error = GenerationRetirement::acquire_with_timeout(&generation, Duration::from_millis(75))
        .err()
        .expect("contended generation retirement must time out");

    assert!(error.contains("timed out waiting for MCP install generation lock"));
    holder.release();
}

#[test]
fn default_install_dir_follows_platform_conventions() {
    assert_eq!(
        default_install_dir_for("macos", Some("/Users/example".into()), None, None, None),
        PathBuf::from("/Users/example/Library/Application Support/nemo-relay/plugins")
    );
    assert_eq!(
        default_install_dir_for("linux", Some("/home/example".into()), None, None, None),
        PathBuf::from("/home/example/.local/share/nemo-relay/plugins")
    );
    assert_eq!(
        default_install_dir_for(
            "linux",
            Some("/home/example".into()),
            None,
            None,
            Some("/data".into())
        ),
        PathBuf::from("/data/nemo-relay/plugins")
    );
    assert_eq!(
        default_install_dir_for(
            "windows",
            None,
            Some(r"C:\Users\example".into()),
            Some(r"C:\Users\example\AppData\Local".into()),
            None
        ),
        PathBuf::from(r"C:\Users\example\AppData\Local")
            .join("nemo-relay")
            .join("plugins")
    );
}

#[test]
fn plugin_manifests_and_hooks_use_path_based_relay_command() {
    assert_eq!(
        marketplace_manifest(CodingAgent::Codex)["name"],
        json!(MARKETPLACE_NAME)
    );
    assert_eq!(
        marketplace_manifest(CodingAgent::ClaudeCode)["plugins"][0]["source"],
        json!("./plugins/nemo-relay-plugin")
    );
    assert_eq!(
        plugin_manifest(CodingAgent::Codex)["name"],
        json!(PLUGIN_NAME)
    );
    assert_eq!(
        plugin_manifest(CodingAgent::Codex)["mcpServers"],
        json!("./.mcp.json")
    );
    let generation_fence = std::env::current_dir()
        .unwrap()
        .join("plugins/nemo-relay-plugin/.nemo-relay-generation");
    let mcp = plugin_mcp_config(
        CodingAgent::Codex,
        Path::new("/bin/nemo-relay"),
        &generation_fence,
        TEST_GENERATION_TOKEN,
    )
    .unwrap();
    let server = &mcp["nemo-relay"];
    assert_eq!(server["command"], json!("/bin/nemo-relay"));
    assert_eq!(server["args"], json!(["mcp"]));
    assert_eq!(
        server["env"],
        json!({
            "NEMO_RELAY_GATEWAY_BIND": "127.0.0.1:47632",
            "NEMO_RELAY_MCP_GENERATION_FILE": &generation_fence,
            "NEMO_RELAY_MCP_GENERATION": TEST_GENERATION_TOKEN
        })
    );
    assert_eq!(server["required"], json!(true));
    assert_eq!(server["startup_timeout_sec"], json!(20));
    assert!(
        server["env_vars"]
            .as_array()
            .unwrap()
            .contains(&json!("OPENAI_API_KEY"))
    );
    let claude_mcp = plugin_mcp_config(
        CodingAgent::ClaudeCode,
        Path::new("/bin/nemo-relay"),
        &generation_fence,
        TEST_GENERATION_TOKEN,
    );
    let claude_server = &claude_mcp.unwrap()["mcpServers"]["nemo-relay"];
    assert_eq!(claude_server["command"], json!("/bin/nemo-relay"));
    assert_eq!(claude_server["args"], json!(["mcp"]));
    assert_eq!(claude_server["alwaysLoad"], json!(true));
    assert_eq!(
        claude_server["env"]["NEMO_RELAY_MCP_GENERATION_FILE"],
        json!(&generation_fence)
    );
    assert_eq!(
        claude_server["env"]["NEMO_RELAY_MCP_GENERATION"],
        json!(TEST_GENERATION_TOKEN)
    );
    assert_eq!(
        plugin_hooks(
            CodingAgent::Codex,
            Path::new("/bin/nemo-relay"),
            &generation_fence,
            TEST_GENERATION_TOKEN,
        )
        .unwrap()["hooks"]["SessionStart"][0]["hooks"][0]["command"],
        json!(
            crate::hooks::persistent_hook_forward_command(
                Path::new("/bin/nemo-relay"),
                CodingAgent::Codex,
                &generation_fence,
                TEST_GENERATION_TOKEN,
            )
            .unwrap()
        )
    );
    assert_eq!(
        plugin_hooks(
            CodingAgent::ClaudeCode,
            Path::new("/bin/nemo-relay"),
            &generation_fence,
            TEST_GENERATION_TOKEN,
        )
        .unwrap()["hooks"]["SessionStart"][0]["hooks"][0]["command"],
        json!(
            crate::hooks::persistent_hook_forward_command(
                Path::new("/bin/nemo-relay"),
                CodingAgent::ClaudeCode,
                &generation_fence,
                TEST_GENERATION_TOKEN,
            )
            .unwrap()
        )
    );
}

#[test]
fn relay_identity_prefers_the_path_resolved_executable() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_current_executable("/opt/nemo-relay/current/nemo-relay")
        .with_executable("nemo-relay", "/opt/nemo-relay/stale/nemo-relay");

    let relay = require_relay(&options(dir.path()), &runner).unwrap();
    let generation = dir
        .path()
        .join("plugins/nemo-relay-plugin/.nemo-relay-generation");

    assert_eq!(relay, PathBuf::from("/opt/nemo-relay/stale/nemo-relay"));
    assert_eq!(
        plugin_hooks(
            CodingAgent::Codex,
            &relay,
            &generation,
            TEST_GENERATION_TOKEN,
        )
        .unwrap()["hooks"]["SessionStart"][0]["hooks"][0]["command"],
        json!(
            crate::hooks::persistent_hook_forward_command(
                &relay,
                CodingAgent::Codex,
                &generation,
                TEST_GENERATION_TOKEN,
            )
            .unwrap()
        )
    );
    assert_eq!(
        plugin_mcp_config(
            CodingAgent::Codex,
            &relay,
            &generation,
            TEST_GENERATION_TOKEN,
        )
        .unwrap()["nemo-relay"]["command"],
        json!(relay)
    );
}

#[test]
fn codex_mcp_env_vars_include_approved_dynamic_and_config_references_only() {
    let config = json!({
        "components": [{
            "kind": "observability",
            "config": {
                "atof": {
                    "storage": [
                        {"header_env": {"authorization": "CUSTOM_HTTP_TOKEN"}},
                        {"header_env": {
                            "blocked": "NEMO_RELAY_PLUGIN_BINARY",
                            "blocked_mixed_case": "NEMO_RELAY_Plugin_Binary",
                            "empty": ""
                        }},
                        {
                            "secret_access_key_var": "CUSTOM_AWS_SECRET",
                            "session_token_var": "CUSTOM_AWS_SESSION"
                        },
                        {
                            "secret_access_key_var": "NEMO_RELAY_GATEWAY_BIND",
                            "session_token_var": "NEMO_RELAY_BOOTSTRAP_SHUTDOWN_TOKEN"
                        }
                    ]
                }
            }
        }]
    });
    let names = crate::agents::codex_mcp_env_vars_from(
        [
            "NEMO_RELAY_CUSTOM_SETTING",
            "OTEL_CUSTOM_SETTING",
            "AWS_CUSTOM_SETTING",
            "NEMO_RELAY_WORKER_TOKEN",
            "NEMO_RELAY_PLUGIN_BINARY",
            "NEMO_RELAY_Plugin_Binary",
            "NEMO_RELAY_GATEWAY_BIND",
            "NEMO_RELAY_MCP_GENERATION",
            "NEMO_RELAY_MCP_GENERATION_FILE",
            "NEMO_RELAY_FAIL_CLOSED",
            "NEMO_RELAY_TRANSPARENT_RUN",
            "NEMO_RELAY_TEST_CODEX_LOG",
            "NEMO_RELAY_Test_CodeX_Log",
        ]
        .map(str::to_string),
        Some(&config),
    );

    assert!(names.is_sorted());
    for expected in [
        "ALL_PROXY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "NEMO_RELAY_GATEWAY_URL",
        "NEMO_RELAY_TRANSPARENT_RUN",
        "NEMO_RELAY_CUSTOM_SETTING",
        "OTEL_CUSTOM_SETTING",
        "AWS_CUSTOM_SETTING",
        "CUSTOM_HTTP_TOKEN",
        "CUSTOM_AWS_SECRET",
        "CUSTOM_AWS_SESSION",
    ] {
        assert!(
            names.iter().any(|name| name == expected),
            "missing {expected}"
        );
    }
    let all_proxy_names = names
        .iter()
        .filter(|name| name.eq_ignore_ascii_case("ALL_PROXY"))
        .collect::<Vec<_>>();
    assert_eq!(all_proxy_names.len(), if cfg!(windows) { 1 } else { 2 });
    assert_eq!(names.iter().any(|name| name == "all_proxy"), !cfg!(windows));
    for excluded in [
        "NEMO_RELAY_WORKER_TOKEN",
        "NEMO_RELAY_PLUGIN_BINARY",
        "NEMO_RELAY_Plugin_Binary",
        "NEMO_RELAY_GATEWAY_BIND",
        "NEMO_RELAY_MCP_GENERATION",
        "NEMO_RELAY_MCP_GENERATION_FILE",
        "NEMO_RELAY_BOOTSTRAP_SHUTDOWN_TOKEN",
        "NEMO_RELAY_FAIL_CLOSED",
        "NEMO_RELAY_TEST_CODEX_LOG",
        "NEMO_RELAY_Test_CodeX_Log",
        "",
    ] {
        assert!(
            !names.iter().any(|name| name == excluded),
            "included {excluded}"
        );
    }
}

#[test]
fn checked_in_codex_mcp_env_vars_match_the_generated_base_allowlist() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../integrations/coding-agents/codex/.mcp.json");
    let checked_in: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let checked_in = checked_in["nemo-relay"]["env_vars"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_str().unwrap().to_string())
        .collect::<Vec<_>>();

    assert_eq!(
        checked_in,
        crate::mcp_environment::forwarded_names_for_platform(std::iter::empty(), None, false),
        "{} drifted from generated MCP environment names",
        path.display()
    );
}

#[test]
fn codex_mcp_env_vars_match_and_deduplicate_names_using_platform_semantics() {
    let config = json!({
        "header_env": {
            "role": "AWS_ROLE_ARN",
            "api_key": "openai_api_key"
        }
    });
    let environment = ["Aws_Role_Arn", "Otel_Custom_Signal"].map(str::to_string);

    let windows = crate::mcp_environment::forwarded_names_for_platform(
        environment.clone(),
        Some(&config),
        true,
    );
    assert!(windows.iter().any(|name| name == "Otel_Custom_Signal"));
    assert_eq!(
        windows
            .iter()
            .filter(|name| name.eq_ignore_ascii_case("AWS_ROLE_ARN"))
            .count(),
        1
    );
    assert_eq!(
        windows
            .iter()
            .filter(|name| name.eq_ignore_ascii_case("OPENAI_API_KEY"))
            .count(),
        1
    );
    assert!(windows.iter().any(|name| name == "OPENAI_API_KEY"));
    for proxy in ["ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"] {
        assert_eq!(
            windows
                .iter()
                .filter(|name| name.eq_ignore_ascii_case(proxy))
                .count(),
            1,
            "Windows MCP environment contains duplicate {proxy} spellings"
        );
    }

    let unix =
        crate::mcp_environment::forwarded_names_for_platform(environment, Some(&config), false);
    assert!(!unix.iter().any(|name| name == "Otel_Custom_Signal"));
    assert!(!unix.iter().any(|name| name == "Aws_Role_Arn"));
    assert!(unix.iter().any(|name| name == "AWS_ROLE_ARN"));
    assert!(unix.iter().any(|name| name == "OPENAI_API_KEY"));
    assert!(unix.iter().any(|name| name == "openai_api_key"));
}

#[test]
fn plugin_setup_delegates_and_dry_run_skips_runner_calls() {
    let dir = tempdir().unwrap();
    let layout = PluginLayout::new(CodingAgent::Codex, dir.path());
    let setup_runner = MockSetupRunner::default();
    let dry_run = PluginInstallOptions {
        dry_run: true,
        ..options(dir.path())
    };

    run_plugin_setup(CodingAgent::Codex, &layout, &dry_run, &setup_runner).unwrap();
    run_plugin_uninstall(
        CodingAgent::ClaudeCode,
        &layout.plugin_root,
        &dry_run,
        &setup_runner,
    )
    .unwrap();
    run_plugin_doctor(
        CodingAgent::Codex,
        &layout.plugin_root,
        &dry_run,
        &setup_runner,
    )
    .unwrap();
    uninstall_host_locked(
        CodingAgent::Codex,
        &dry_run,
        &MockRunner::default(),
        &setup_runner,
    )
    .unwrap();
    assert!(setup_runner.calls().is_empty());

    let normal = options(dir.path());
    run_plugin_setup(CodingAgent::Codex, &layout, &normal, &setup_runner).unwrap();
    run_plugin_uninstall(
        CodingAgent::ClaudeCode,
        &layout.plugin_root,
        &normal,
        &setup_runner,
    )
    .unwrap();
    run_plugin_doctor(
        CodingAgent::Codex,
        &layout.plugin_root,
        &normal,
        &setup_runner,
    )
    .unwrap();
    let report =
        run_plugin_doctor_json(CodingAgent::ClaudeCode, &layout.plugin_root, &setup_runner)
            .unwrap();

    assert_eq!(
        setup_runner.calls(),
        vec![
            format!("setup codex {DEFAULT_GATEWAY_URL}"),
            format!("uninstall claude-code {DEFAULT_GATEWAY_URL}"),
            format!("doctor codex {DEFAULT_GATEWAY_URL}"),
            format!("doctor-json claude-code {DEFAULT_GATEWAY_URL}"),
        ]
    );
    assert_eq!(report["ok"], json!(true));
}

#[test]
fn real_plugin_setup_runner_uses_temp_home_for_claude_paths() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let runner = HostPluginSetupRunner::new(CodingAgent::ClaudeCode);
    let plugin_root = dir.path().join("plugin");

    runner
        .setup("claude-code", DEFAULT_GATEWAY_URL, &plugin_root)
        .unwrap();
    assert!(
        runner
            .doctor("claude-code", DEFAULT_GATEWAY_URL, &plugin_root)
            .is_ok()
    );
    let claude_report = runner
        .doctor_json("claude-code", DEFAULT_GATEWAY_URL, &plugin_root)
        .unwrap();
    assert_eq!(
        claude_report["checks"]["claude_provider_routing"],
        json!(true)
    );
    runner
        .uninstall("claude-code", DEFAULT_GATEWAY_URL, &plugin_root)
        .unwrap();
}

#[test]
fn setup_action_descriptions_cover_supported_hosts_and_actions() {
    assert_eq!(
        CodingAgent::Codex.setup_action_description("configure"),
        "configure Codex provider and trust plugin-owned hooks"
    );
    assert_eq!(
        CodingAgent::Codex.setup_action_description("restore"),
        "remove Codex provider and plugin hook trust"
    );
    assert_eq!(
        CodingAgent::Codex.setup_action_description("doctor"),
        "check Codex provider and plugin-owned hooks"
    );
    assert_eq!(
        CodingAgent::ClaudeCode.setup_action_description("configure"),
        "enable Claude Code provider routing through NeMo Relay"
    );
    assert_eq!(
        CodingAgent::ClaudeCode.setup_action_description("restore"),
        "restore Claude Code provider routing from NeMo Relay backup"
    );
    assert_eq!(
        CodingAgent::ClaudeCode.setup_action_description("doctor"),
        "check Claude Code provider routing"
    );
}

#[test]
fn host_command_helpers_cover_dry_run_missing_failure_and_reporting() {
    let dir = tempdir().unwrap();
    let dry_run = PluginInstallOptions {
        dry_run: true,
        ..options(dir.path())
    };
    let runner = MockRunner::default();

    assert_eq!(
        require_relay(&dry_run, &runner).unwrap(),
        PathBuf::from(RELAY_COMMAND)
    );
    require_host_cli(CodingAgent::Codex, &dry_run, &runner).unwrap();
    validate_host_version(CodingAgent::ClaudeCode, &dry_run, &runner).unwrap();
    validate_relay_hook_forward(Path::new("nemo-relay"), &dry_run, &runner).unwrap();
    validate_relay_mcp(Path::new("nemo-relay"), &dry_run, &runner).unwrap();
    run_command(
        "codex",
        &["plugin".into(), "add space".into()],
        &dry_run,
        &runner,
    )
    .unwrap();
    run_path_command(
        Path::new("/bin/codex"),
        &["arg with space".into()],
        &dry_run,
        &runner,
    )
    .unwrap();
    let capture = run_capture_command("codex", &["plugin".into()], &dry_run, &runner).unwrap();
    assert_eq!(capture.stdout, "null\n");
    let report = host_registration_report(CodingAgent::Codex, &dry_run, &runner).unwrap();
    assert!(report.ok());
    assert_eq!(report.to_json()["ok"], json!(true));
    assert_eq!(
        HostRegistrationReport {
            host_plugin_registered: false,
            host_marketplace_registered: true,
        }
        .to_json()["host_plugin_registered"],
        json!(false)
    );

    let normal = options(dir.path());
    assert!(
        require_relay(&normal, &runner)
            .unwrap_err()
            .contains("nemo-relay")
    );
    assert!(
        require_host_cli(CodingAgent::Codex, &normal, &runner)
            .unwrap_err()
            .contains("codex")
    );
    assert!(
        run_command("codex", &["plugin".into()], &normal, &runner)
            .unwrap_err()
            .contains("codex")
    );

    let mut runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex");
    runner.failing_quiet_suffix = Some("hook-forward --help".into());
    assert!(
        validate_relay_hook_forward(Path::new("/bin/nemo-relay"), &normal, &runner)
            .unwrap_err()
            .contains("hook-forward")
    );
    runner.failing_quiet_suffix = Some("mcp --help".into());
    assert!(
        validate_relay_mcp(Path::new("/bin/nemo-relay"), &normal, &runner)
            .unwrap_err()
            .contains("nemo-relay mcp")
    );
    runner.failing_suffix = Some("plugin add".into());
    assert!(
        run_path_command(
            Path::new("/bin/codex"),
            &["plugin".into(), "add".into()],
            &normal,
            &runner
        )
        .unwrap_err()
        .contains("exit code 1")
    );
    let quoted = format_command(
        "codex",
        &["plugin".into(), "arg with space".into(), "quote\"$".into()],
    );
    #[cfg(not(windows))]
    assert!(quoted.contains("'arg with space'"));
    #[cfg(not(windows))]
    assert!(quoted.contains("'quote\"$'"));
    #[cfg(windows)]
    assert!(quoted.contains("\"arg with space\""));
    #[cfg(windows)]
    assert!(quoted.contains("\"quote\"\"$\""));

    let runner = MockRunner::default()
        .with_executable("codex", "/bin/codex")
        .with_capture_status("/bin/codex plugin bad", 2, "", "")
        .with_capture_status("/bin/codex plugin noisy", 3, "", "boom");
    assert!(
        run_capture_command("codex", &["plugin".into(), "bad".into()], &normal, &runner)
            .unwrap_err()
            .contains("exit code 2")
    );
    assert!(
        run_capture_command(
            "codex",
            &["plugin".into(), "noisy".into()],
            &normal,
            &runner
        )
        .unwrap_err()
        .contains(": boom")
    );

    let runner = MockRunner::default()
        .with_executable("codex", "/bin/codex")
        .with_capture_output("/bin/codex plugin list", "PLUGIN  STATUS  VERSION  PATH\n")
        .with_capture_output("/bin/codex plugin marketplace list", "MARKETPLACE ROOT\n");
    let error = validate_host_registration(CodingAgent::Codex, &normal, &runner).unwrap_err();
    assert!(
        error.contains("host plugin") && error.contains("host marketplace"),
        "error was: {error}"
    );
}

#[test]
fn host_registration_report_accepts_claude_and_codex_shape_variants() {
    let dir = tempdir().unwrap();
    let normal = options(dir.path());
    let plugin_id = format!("{PLUGIN_NAME}@{MARKETPLACE_NAME}");

    for (plugin_entry, marketplace_entry) in [
        (
            json!({"id": plugin_id.clone()}),
            json!({"id": MARKETPLACE_NAME}),
        ),
        (
            json!({"pluginId": plugin_id.clone()}),
            json!({"name": MARKETPLACE_NAME}),
        ),
        (
            json!({"name": PLUGIN_NAME, "marketplaceName": MARKETPLACE_NAME}),
            json!({"id": MARKETPLACE_NAME}),
        ),
    ] {
        let runner = MockRunner::default()
            .with_executable("claude", "/bin/claude")
            .with_capture_output(
                "/bin/claude plugin list --json",
                json!([plugin_entry]).to_string(),
            )
            .with_capture_output(
                "/bin/claude plugin marketplace list --json",
                json!([marketplace_entry]).to_string(),
            );
        let report = host_registration_report(CodingAgent::ClaudeCode, &normal, &runner).unwrap();
        assert!(report.ok());
        assert!(report.host_plugin_registered);
        assert!(report.host_marketplace_registered);
    }

    let runner = MockRunner::default()
        .with_executable("codex", "/bin/codex")
        .with_capture_output(
            "/bin/codex plugin list",
            format!("{plugin_id}  installed, enabled  0.4.0  /tmp/nemo-relay-plugin\n"),
        )
        .with_capture_output(
            "/bin/codex plugin marketplace list",
            format!("{MARKETPLACE_NAME} /tmp/nemo-relay-local\n"),
        );
    let report = host_registration_report(CodingAgent::Codex, &normal, &runner).unwrap();
    assert!(report.ok());

    let runner = MockRunner::default()
        .with_executable("codex", "/bin/codex")
        .with_capture_output(
            "/bin/codex plugin list",
            format!("{plugin_id}  not installed\n"),
        )
        .with_capture_output(
            "/bin/codex plugin marketplace list",
            format!("{MARKETPLACE_NAME} /tmp/nemo-relay-local\n"),
        );
    let report = host_registration_report(CodingAgent::Codex, &normal, &runner).unwrap();
    assert!(!report.host_plugin_registered);
    assert!(report.host_marketplace_registered);

    let runner = MockRunner::default()
        .with_executable("codex", "/bin/codex")
        .with_capture_output(
            "/bin/codex plugin list",
            format!("{PLUGIN_NAME}@other  installed, enabled  0.4.0  /tmp/other\n"),
        )
        .with_capture_output("/bin/codex plugin marketplace list", "other /tmp/other\n");
    let report = host_registration_report(CodingAgent::Codex, &normal, &runner).unwrap();
    assert!(!report.ok());
    assert!(!report.host_plugin_registered);
    assert!(!report.host_marketplace_registered);
}

#[test]
fn host_registration_report_surfaces_capture_status_and_stderr_variants() {
    let dir = tempdir().unwrap();
    let normal = options(dir.path());

    let runner = MockRunner::default()
        .with_executable("claude", "/bin/claude")
        .with_capture_output("/bin/claude plugin list --json", "not json");
    assert!(
        host_registration_report(CodingAgent::ClaudeCode, &normal, &runner)
            .unwrap_err()
            .contains("failed to parse")
    );

    let runner = MockRunner::default()
        .with_executable("claude", "/bin/claude")
        .with_capture_status(
            "/bin/claude plugin list --json",
            4,
            "ignored stdout",
            "  noisy failure  \n",
        );
    let error = host_registration_report(CodingAgent::ClaudeCode, &normal, &runner).unwrap_err();
    assert!(error.contains("exit code 4: noisy failure"));

    let runner = MockRunner::default()
        .with_executable("claude", "/bin/claude")
        .with_capture_output(
            "/bin/claude plugin list --json",
            json!([{ "id": format!("{PLUGIN_NAME}@{MARKETPLACE_NAME}") }]).to_string(),
        )
        .with_capture_status(
            "/bin/claude plugin marketplace list --json",
            5,
            "ignored stdout",
            "",
        );
    let error = host_registration_report(CodingAgent::ClaudeCode, &normal, &runner).unwrap_err();
    assert!(error.contains("exit code 5"));
    assert!(!error.contains("exit code 5:"));
}

#[test]
fn top_level_install_uninstall_and_doctor_report_empty_host_selection() {
    let dir = tempdir().unwrap();
    let empty_path = dir.path().join("empty-path");
    std::fs::create_dir_all(&empty_path).unwrap();
    let _path = PathScope::set_isolated(&empty_path, &dir.path().join("home"));

    assert!(crate::agents::detected_install_integrations(&CodingAgent::ALL).is_empty());
    assert!(
        crate::agents::installed_integrations(
            &CodingAgent::ALL,
            Some(&dir.path().join("install")),
        )
        .is_empty()
    );

    assert_eq!(
        install(
            CodingAgent::Codex,
            crate::installation::InstallRequest {
                install_dir: Some(dir.path().join("dry-run-install")),
                force: false,
                dry_run: true,
                skip_doctor: true,
            }
        )
        .unwrap(),
        std::process::ExitCode::SUCCESS
    );

    let doctor_options = plugin_doctor_options(Some(dir.path().join("install")));
    let codex_doctor_error = crate::agents::doctor_integration(CodingAgent::Codex, &doctor_options)
        .unwrap_err()
        .to_string();
    assert!(
        codex_doctor_error.contains("nemo-relay install codex --force"),
        "error was: {codex_doctor_error}"
    );

    assert_eq!(CodingAgent::Codex.as_arg(), "codex");
    assert_eq!(CodingAgent::Codex.label(), "Codex");
    assert_eq!(CodingAgent::Codex.executable(), "codex");
}

#[test]
fn installed_selection_uses_persisted_integration_state() {
    let dir = tempdir().unwrap();
    let home = dir.path().join("home");
    std::fs::create_dir_all(home.join(".hermes")).unwrap();
    let _home = HomeScope::enter(&home);
    std::fs::write(
        state_path(CodingAgent::ClaudeCode, dir.path()),
        r#"{"marketplaceRoot":"/tmp/m","pluginRoot":"/tmp/p"}"#,
    )
    .unwrap();
    let selected = crate::agents::installed_integrations(&CodingAgent::ALL, Some(dir.path()));
    assert_eq!(selected, vec![CodingAgent::ClaudeCode]);

    let unrelated_hermes_config = b"# user-owned formatting\nmodel: custom\n";
    let hermes_config = home.join(".hermes/config.yaml");
    std::fs::write(&hermes_config, unrelated_hermes_config).unwrap();
    let selected = crate::agents::installed_integrations(&CodingAgent::ALL, Some(dir.path()));
    assert_eq!(selected, vec![CodingAgent::ClaudeCode]);
    assert_eq!(
        std::fs::read(&hermes_config).unwrap(),
        unrelated_hermes_config
    );
}

#[test]
fn hermes_doctor_probes_the_configured_relay_and_top_level_doctor_discovers_it() {
    let dir = tempdir().unwrap();
    let home = dir.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let _home = HomeScope::enter(&home);
    let config = crate::agents::hermes::install::config_path().unwrap();
    let relay = home
        .join("bin")
        .join(format!("nemo-relay{}", std::env::consts::EXE_SUFFIX));
    std::fs::create_dir_all(relay.parent().unwrap()).unwrap();
    std::fs::copy(std::env::current_exe().unwrap(), &relay).unwrap();
    crate::agents::hermes::install_persistent(&config, &relay).unwrap();
    let configured_relay = crate::agents::hermes::configured_relay_executable(&config).unwrap();
    let runner = MockRunner::default()
        .with_executable("hermes", "/bin/hermes")
        .with_capture_output("/bin/hermes --version", "Hermes Agent v0.18.2 (test)\n");

    let report =
        crate::agents::hermes::install::doctor_json_value(&options(dir.path()), &runner).unwrap();

    assert_eq!(report["ok"], json!(true));
    assert_eq!(
        runner.quiet_commands(),
        vec![
            format!("{} hook-forward --help", configured_relay.display()),
            format!("{} mcp --help", configured_relay.display()),
        ]
    );
    let checks = report["readiness_checks"].as_array().unwrap();
    for expected in [
        "Host CLI",
        "Hermes Agent version",
        "Configured Relay binary",
        "Relay hook support",
        "Relay MCP support",
        "Hermes MCP, hooks, and trust",
    ] {
        assert!(
            checks
                .iter()
                .any(|check| check["name"] == expected && check["ok"] == json!(true)),
            "missing successful {expected} check: {checks:?}"
        );
    }

    let readiness = crate::agents::collect_default_integration_readiness();
    let hermes = readiness
        .iter()
        .find(|readiness| readiness.host == "hermes")
        .expect("top-level doctor should discover install-only Hermes state");
    assert_eq!(hermes.state_path, config);
    assert!(hermes.marketplace.is_none());
    assert!(hermes.plugin.is_none());

    crate::agents::hermes::install::doctor(&options(dir.path()), &runner).unwrap();

    std::fs::remove_file(&configured_relay).unwrap();
    let error = crate::agents::hermes::install::doctor(&options(dir.path()), &runner).unwrap_err();
    assert!(error.contains("doctor checks failed"), "{error}");
    let report =
        crate::agents::hermes::install::doctor_json_value(&options(dir.path()), &runner).unwrap();
    let failed = report["readiness_checks"].as_array().unwrap();
    for expected in [
        "Configured Relay binary",
        "Relay hook support",
        "Relay MCP support",
    ] {
        assert!(
            failed
                .iter()
                .any(|check| check["name"] == expected && check["ok"] == json!(false)),
            "missing failed {expected} check: {failed:?}"
        );
    }
}

#[test]
fn hermes_install_and_uninstall_dry_runs_preserve_persistent_state() {
    let dir = tempdir().unwrap();
    let home = dir.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let _home = HomeScope::enter(&home);
    crate::agents::hermes::install::install(crate::installation::InstallRequest {
        install_dir: Some(dir.path().to_path_buf()),
        force: false,
        dry_run: true,
        skip_doctor: false,
    })
    .unwrap();
    let config = crate::agents::hermes::install::config_path().unwrap();
    assert!(!config.exists());

    let hermes_home = config.parent().unwrap();
    std::fs::create_dir_all(hermes_home).unwrap();
    let allowlist = hermes_home.join("shell-hooks-allowlist.json");
    let generation = hermes_home.join(GENERATION_FILE_NAME);
    let sentinels = [
        (&config, b"sentinel config\n".as_slice()),
        (&allowlist, b"sentinel allowlist\n".as_slice()),
        (&generation, b"sentinel generation\n".as_slice()),
    ];
    for (path, contents) in sentinels {
        std::fs::write(path, contents).unwrap();
    }

    crate::agents::hermes::install::uninstall(crate::installation::UninstallRequest {
        install_dir: Some(dir.path().to_path_buf()),
        dry_run: true,
    })
    .unwrap();

    for (path, contents) in sentinels {
        assert_eq!(std::fs::read(path).unwrap(), contents);
    }
}

#[test]
fn install_codex_generates_marketplace_and_runs_setup() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex");
    let setup_runner = MockSetupRunner::default();

    install_host(
        CodingAgent::Codex,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap();

    let layout = PluginLayout::new(CodingAgent::Codex, dir.path());
    let generation = InstallGeneration::capture(layout.generation_fence.clone()).unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(&std::fs::read_to_string(&layout.hooks_path).unwrap())
            .unwrap(),
        plugin_hooks(
            CodingAgent::Codex,
            Path::new("/bin/nemo-relay"),
            &layout.generation_fence,
            generation.token(),
        )
        .unwrap()
    );
    assert_eq!(
        runner.commands(),
        vec![
            format!(
                "/bin/codex plugin marketplace add {}",
                layout.marketplace_root.display()
            ),
            "/bin/codex plugin add nemo-relay-plugin@nemo-relay-local".into(),
        ]
    );
    assert_eq!(
        runner.quiet_commands(),
        vec![relay_validation_command(), relay_mcp_validation_command()]
    );
    assert_eq!(
        serde_json::from_str::<Value>(&std::fs::read_to_string(&layout.mcp_config).unwrap())
            .unwrap(),
        plugin_mcp_config(
            CodingAgent::Codex,
            Path::new("/bin/nemo-relay"),
            &layout.generation_fence,
            generation.token(),
        )
        .unwrap()
    );
    assert_eq!(
        setup_runner.calls(),
        vec![format!("setup codex {DEFAULT_GATEWAY_URL}")]
    );
}

#[test]
fn install_prunes_stale_managed_plugin_root() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("claude", "/bin/claude");
    let setup_runner = MockSetupRunner::default();
    let layout = PluginLayout::new(CodingAgent::ClaudeCode, dir.path());
    let stale = layout.plugin_root.join("bin").join("nemo-relay");
    std::fs::create_dir_all(stale.parent().unwrap()).unwrap();
    std::fs::write(&stale, "stale").unwrap();

    install_host(
        CodingAgent::ClaudeCode,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap();

    assert!(!stale.exists());
    assert!(layout.plugin_manifest.exists());
}

#[test]
fn ordinary_codex_reinstall_refuses_a_fenced_install_without_mutating_it() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex")
        .with_codex_registration(true, true);
    let setup_runner = MockSetupRunner::default();
    write_installed_state(CodingAgent::Codex, dir.path());
    let layout = PluginLayout::new(CodingAgent::Codex, dir.path());
    let sentinel = layout.plugin_root.join("existing-install");
    std::fs::write(&sentinel, b"preserve").unwrap();
    let state = std::fs::read(&layout.state_path).unwrap();
    let generation = std::fs::read(&layout.generation_fence).unwrap();

    let error = install_host(
        CodingAgent::Codex,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap_err();

    assert!(error.contains("existing fenced Codex plugin"), "{error}");
    assert!(
        error.contains("nemo-relay install codex --force"),
        "{error}"
    );
    assert_eq!(std::fs::read(&sentinel).unwrap(), b"preserve");
    assert_eq!(std::fs::read(&layout.state_path).unwrap(), state);
    assert_eq!(std::fs::read(&layout.generation_fence).unwrap(), generation);
    assert!(runner.commands().is_empty());
    assert!(setup_runner.calls().is_empty());
    assert_no_install_stage(dir.path());
}

#[test]
fn ordinary_codex_reinstall_refuses_a_legacy_install_before_staging() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex")
        .with_codex_registration(false, false);
    let setup_runner = MockSetupRunner::default();
    write_installed_state(CodingAgent::Codex, dir.path());
    let layout = PluginLayout::new(CodingAgent::Codex, dir.path());
    std::fs::remove_file(&layout.generation_fence).unwrap();
    let state = std::fs::read(&layout.state_path).unwrap();

    let error = install_host(
        CodingAgent::Codex,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap_err();

    assert_actionable_generation_error(&error, "MCP generation marker is missing");
    assert_eq!(std::fs::read(&layout.state_path).unwrap(), state);
    assert!(layout.marketplace_root.exists());
    assert!(runner.commands().is_empty());
    assert!(setup_runner.calls().is_empty());
    assert_no_install_stage(dir.path());
}

#[test]
fn ordinary_codex_reinstall_refuses_a_registration_without_local_state() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex")
        .with_codex_registration(true, true);
    let setup_runner = MockSetupRunner::default();

    let error = install_host(
        CodingAgent::Codex,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap_err();

    assert_actionable_generation_error(&error, "MCP generation marker is missing");
    assert!(runner.commands().is_empty());
    assert!(setup_runner.calls().is_empty());
    assert_no_install_stage(dir.path());
}

#[test]
fn ordinary_codex_reinstall_refuses_a_corrupt_install_before_staging() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex")
        .with_codex_registration(false, false);
    let setup_runner = MockSetupRunner::default();
    write_installed_state(CodingAgent::Codex, dir.path());
    let layout = PluginLayout::new(CodingAgent::Codex, dir.path());
    std::fs::write(&layout.generation_fence, b"").unwrap();
    let state = std::fs::read(&layout.state_path).unwrap();

    let error = install_host(
        CodingAgent::Codex,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap_err();

    assert_actionable_generation_error(&error, "is invalid or unreadable");
    assert!(error.contains("is empty"), "{error}");
    assert_eq!(std::fs::read(&layout.state_path).unwrap(), state);
    assert!(layout.marketplace_root.exists());
    assert!(runner.commands().is_empty());
    assert!(setup_runner.calls().is_empty());
    assert_no_install_stage(dir.path());
}

#[test]
fn force_install_unregisters_existing_host_before_reinstall() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex")
        .with_codex_registration(true, true);
    let setup_runner = MockSetupRunner::default();
    let options = PluginInstallOptions {
        force: true,
        ..options(dir.path())
    };
    write_installed_state(CodingAgent::Codex, dir.path());

    install_host(CodingAgent::Codex, &options, &runner, &setup_runner).unwrap();

    let commands = runner.commands();
    let remove_index = commands
        .iter()
        .position(|command| {
            command == "/bin/codex plugin remove nemo-relay-plugin@nemo-relay-local"
        })
        .unwrap();
    let add_index = commands
        .iter()
        .position(|command| command.ends_with("plugin add nemo-relay-plugin@nemo-relay-local"))
        .unwrap();
    assert!(remove_index < add_index);
    assert!(
        setup_runner
            .calls()
            .iter()
            .any(|call| call == "snapshot codex")
    );
    assert!(
        setup_runner
            .calls()
            .iter()
            .any(|call| call == &format!("uninstall codex {DEFAULT_GATEWAY_URL}"))
    );
    assert!(
        setup_runner
            .calls()
            .iter()
            .any(|call| call == "refresh gateway")
    );
    let setup_calls = setup_runner.calls();
    let refresh_index = setup_calls
        .iter()
        .position(|call| call == "refresh gateway")
        .unwrap();
    let snapshot_index = setup_calls
        .iter()
        .position(|call| call == "snapshot codex")
        .unwrap();
    let uninstall_index = setup_calls
        .iter()
        .position(|call| call == &format!("uninstall codex {DEFAULT_GATEWAY_URL}"))
        .unwrap();
    assert!(snapshot_index < uninstall_index);
    assert!(uninstall_index < refresh_index);
}

#[test]
fn force_install_retires_previous_mcp_generation() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex")
        .with_codex_registration(true, true);
    let setup_runner = MockSetupRunner::default();
    let options = PluginInstallOptions {
        force: true,
        ..options(dir.path())
    };
    write_installed_state(CodingAgent::Codex, dir.path());
    let layout = PluginLayout::new(CodingAgent::Codex, dir.path());
    let previous = InstallGeneration::capture(layout.generation_fence.clone()).unwrap();
    let previous_token = previous.token().to_string();
    let cached_mcp =
        serde_json::from_str::<Value>(&std::fs::read_to_string(&layout.mcp_config).unwrap())
            .unwrap();
    let cached_hooks = serde_json::from_str::<serde_json::Value>(
        &std::fs::read_to_string(&layout.hooks_path).unwrap(),
    )
    .unwrap();

    install_host(CodingAgent::Codex, &options, &runner, &setup_runner).unwrap();

    let error = previous.verify_current().unwrap_err();
    assert!(error.contains("has been retired"));
    let current = InstallGeneration::capture(layout.generation_fence.clone()).unwrap();
    assert_ne!(current.token(), previous_token);
    let mcp = serde_json::from_str::<Value>(&std::fs::read_to_string(&layout.mcp_config).unwrap())
        .unwrap();
    assert_eq!(
        mcp["nemo-relay"]["env"]["NEMO_RELAY_MCP_GENERATION_FILE"],
        json!(layout.generation_fence)
    );
    assert_eq!(
        mcp["nemo-relay"]["env"]["NEMO_RELAY_MCP_GENERATION"],
        json!(current.token())
    );
    assert_eq!(
        cached_mcp["nemo-relay"]["env"]["NEMO_RELAY_MCP_GENERATION"],
        json!(previous_token)
    );
    assert!(crate::hook_assertions::value_has_command_arguments(
        &cached_hooks,
        &["--generation-token", &previous_token]
    ));
    let current_hooks = serde_json::from_str::<serde_json::Value>(
        &std::fs::read_to_string(&layout.hooks_path).unwrap(),
    )
    .unwrap();
    assert!(crate::hook_assertions::value_has_command_arguments(
        &current_hooks,
        &["--generation-token", current.token()]
    ));
    assert!(layout.generation_lock.exists());
}

#[cfg(windows)]
#[test]
fn force_install_reuses_the_same_windows_lock_after_install_dir_canonicalization() {
    let root = tempdir().unwrap();
    let requested_install_dir = root.path().join("not-created-yet");
    let first_install_dir = requested_install_dir.clone().canonicalize_or_self();
    assert!(!first_install_dir.exists());
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex")
        .with_codex_registration(false, false);
    let setup_runner = MockSetupRunner::default();

    install_host(
        CodingAgent::Codex,
        &options(&first_install_dir),
        &runner,
        &setup_runner,
    )
    .unwrap();
    let first_layout = PluginLayout::new(CodingAgent::Codex, &first_install_dir);
    let previous = InstallGeneration::capture(first_layout.generation_fence).unwrap();

    let canonical_install_dir = requested_install_dir.canonicalize().unwrap();
    assert_ne!(first_install_dir, canonical_install_dir);
    install_host(
        CodingAgent::Codex,
        &PluginInstallOptions {
            force: true,
            ..options(&canonical_install_dir)
        },
        &runner,
        &setup_runner,
    )
    .unwrap();

    assert!(previous.verify_current().unwrap_err().contains("retired"));
    let current_layout = PluginLayout::new(CodingAgent::Codex, &canonical_install_dir);
    assert!(current_layout.generation_lock.exists());
    InstallGeneration::capture(current_layout.generation_fence).unwrap();
}

#[cfg(windows)]
#[test]
fn force_install_migrates_a_legacy_sibling_lock_before_moving_the_marketplace() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex")
        .with_codex_registration(true, true);
    let setup_runner = MockSetupRunner::default();
    write_installed_state(CodingAgent::Codex, dir.path());
    let layout = PluginLayout::new(CodingAgent::Codex, dir.path());
    let (previous_token, legacy_lock) = replace_generation_with_legacy_marker(&layout);

    install_host(
        CodingAgent::Codex,
        &PluginInstallOptions {
            force: true,
            ..options(dir.path())
        },
        &runner,
        &setup_runner,
    )
    .unwrap();

    let current = InstallGeneration::capture(layout.generation_fence).unwrap();
    assert_ne!(current.token(), previous_token);
    assert!(layout.generation_lock.exists());
    assert!(!legacy_lock.exists());
}

#[cfg(windows)]
#[test]
fn uninstall_releases_a_legacy_sibling_lock_before_removing_the_marketplace() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex");
    let setup_runner = MockSetupRunner::default();
    write_installed_state(CodingAgent::Codex, dir.path());
    let layout = PluginLayout::new(CodingAgent::Codex, dir.path());
    let (_, legacy_lock) = replace_generation_with_legacy_marker(&layout);

    uninstall_host(
        CodingAgent::Codex,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap();

    assert!(!layout.marketplace_root.exists());
    assert!(!layout.state_path.exists());
    assert!(!legacy_lock.exists());
}

#[cfg(windows)]
#[test]
fn legacy_force_install_rollback_restores_the_sibling_lock_without_external_residue() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex")
        .with_codex_registration(true, true);
    let setup_runner = MockSetupRunner {
        failing_call: Some(format!("doctor codex {DEFAULT_GATEWAY_URL}")),
        ..MockSetupRunner::default()
    };
    write_installed_state(CodingAgent::Codex, dir.path());
    let layout = PluginLayout::new(CodingAgent::Codex, dir.path());
    let (previous_token, legacy_lock) = replace_generation_with_legacy_marker(&layout);

    let error = install_host(
        CodingAgent::Codex,
        &PluginInstallOptions {
            force: true,
            skip_doctor: false,
            ..options(dir.path())
        },
        &runner,
        &setup_runner,
    )
    .unwrap_err();

    assert!(error.contains("doctor codex"), "{error}");
    let restored = InstallGeneration::capture(layout.generation_fence).unwrap();
    assert_eq!(restored.token(), previous_token);
    assert!(legacy_lock.exists());
    assert!(!layout.generation_lock.exists());
}

#[test]
fn claude_force_install_retires_and_replaces_its_mcp_generation() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("claude", "/bin/claude")
        .with_claude_registration(true, true);
    let setup_runner = MockSetupRunner::default();
    let options = PluginInstallOptions {
        force: true,
        ..options(dir.path())
    };
    write_installed_state(CodingAgent::ClaudeCode, dir.path());
    let layout = PluginLayout::new(CodingAgent::ClaudeCode, dir.path());
    let previous = InstallGeneration::capture(layout.generation_fence.clone()).unwrap();
    let previous_token = previous.token().to_string();
    let cached_mcp =
        serde_json::from_str::<Value>(&std::fs::read_to_string(&layout.mcp_config).unwrap())
            .unwrap();

    install_host(CodingAgent::ClaudeCode, &options, &runner, &setup_runner).unwrap();

    assert!(previous.verify_current().unwrap_err().contains("retired"));
    let current = InstallGeneration::capture(layout.generation_fence.clone()).unwrap();
    assert_ne!(current.token(), previous_token);
    let mcp = serde_json::from_str::<Value>(&std::fs::read_to_string(&layout.mcp_config).unwrap())
        .unwrap();
    assert_eq!(
        mcp["mcpServers"]["nemo-relay"]["env"]["NEMO_RELAY_MCP_GENERATION_FILE"],
        json!(layout.generation_fence)
    );
    assert_eq!(
        mcp["mcpServers"]["nemo-relay"]["env"]["NEMO_RELAY_MCP_GENERATION"],
        json!(current.token())
    );
    assert_eq!(
        cached_mcp["mcpServers"]["nemo-relay"]["env"]["NEMO_RELAY_MCP_GENERATION"],
        json!(previous_token)
    );
    assert!(
        setup_runner
            .calls()
            .iter()
            .any(|call| call == "refresh gateway")
    );
}

#[test]
fn claude_force_install_rollback_restores_generation_files_and_setup_snapshot() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("claude", "/bin/claude")
        .with_claude_registration(true, true);
    let setup_runner = MockSetupRunner {
        failing_call: Some(format!("doctor claude-code {DEFAULT_GATEWAY_URL}")),
        ..MockSetupRunner::default()
    };
    let options = PluginInstallOptions {
        force: true,
        skip_doctor: false,
        ..options(dir.path())
    };
    write_installed_state(CodingAgent::ClaudeCode, dir.path());
    let layout = PluginLayout::new(CodingAgent::ClaudeCode, dir.path());
    let sentinel = layout.plugin_root.join("previous-install");
    std::fs::write(&sentinel, "restore-exactly").unwrap();
    let original_state = std::fs::read(&layout.state_path).unwrap();
    let previous = InstallGeneration::capture(layout.generation_fence.clone()).unwrap();

    let error =
        install_host(CodingAgent::ClaudeCode, &options, &runner, &setup_runner).unwrap_err();

    assert!(error.contains("doctor claude-code"), "{error}");
    assert_eq!(
        std::fs::read_to_string(&sentinel).unwrap(),
        "restore-exactly"
    );
    assert_eq!(std::fs::read(&layout.state_path).unwrap(), original_state);
    previous.verify_current().unwrap();
    let setup_calls = setup_runner.calls();
    assert!(
        setup_calls
            .iter()
            .any(|call| call == "snapshot claude-code")
    );
    assert!(setup_calls.iter().any(|call| call == "restore snapshot"));
    assert_eq!(
        setup_calls
            .iter()
            .filter(|call| call.as_str() == "refresh gateway")
            .count(),
        2,
        "the previous and replacement gateway generations must both be retired: {setup_calls:?}"
    );
    assert_no_force_replacement_residue(dir.path());
}

#[test]
fn claude_force_install_migrates_a_legacy_hook_only_plugin() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("claude", "/bin/claude")
        .with_claude_registration(true, true);
    let setup_runner = MockSetupRunner::default();
    let options = PluginInstallOptions {
        force: true,
        ..options(dir.path())
    };
    let layout = PluginLayout::new(CodingAgent::ClaudeCode, dir.path());
    std::fs::create_dir_all(layout.plugin_manifest.parent().unwrap()).unwrap();
    write_json(
        &layout.marketplace_manifest,
        &marketplace_manifest(CodingAgent::ClaudeCode),
    )
    .unwrap();
    let mut legacy_manifest = plugin_manifest(CodingAgent::ClaudeCode);
    legacy_manifest
        .as_object_mut()
        .unwrap()
        .remove("mcpServers");
    write_json(&layout.plugin_manifest, &legacy_manifest).unwrap();
    write_state(&layout, &options).unwrap();
    mark_plugin_setup_installed(CodingAgent::ClaudeCode, &layout, &options).unwrap();

    install_host(CodingAgent::ClaudeCode, &options, &runner, &setup_runner).unwrap();

    InstallGeneration::capture(layout.generation_fence.clone()).unwrap();
    assert!(layout.mcp_config.is_file());
    let installed =
        serde_json::from_str::<Value>(&std::fs::read_to_string(&layout.plugin_manifest).unwrap())
            .unwrap();
    assert_eq!(installed["mcpServers"], json!("./.mcp.json"));
}

#[test]
fn ordinary_claude_reinstall_requires_force_for_a_fenced_plugin() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("claude", "/bin/claude")
        .with_claude_registration(true, true);
    let setup_runner = MockSetupRunner::default();
    write_installed_state(CodingAgent::ClaudeCode, dir.path());
    let layout = PluginLayout::new(CodingAgent::ClaudeCode, dir.path());
    let original_generation = std::fs::read(&layout.generation_fence).unwrap();

    let error = install_host(
        CodingAgent::ClaudeCode,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap_err();

    assert!(error.contains("nemo-relay install claude-code --force"));
    assert_eq!(
        std::fs::read(&layout.generation_fence).unwrap(),
        original_generation
    );
    assert!(runner.commands().is_empty());
    assert!(setup_runner.calls().is_empty());
}

#[test]
fn force_install_rejects_persisted_roots_outside_selected_layout() {
    let dir = tempdir().unwrap();
    let selected_dir = dir.path().join("selected");
    let relocated_dir = dir.path().join("relocated");
    let relocated = write_relocated_codex_install(&selected_dir, &relocated_dir);
    let sentinel = relocated.plugin_root.join("relocated-install");
    std::fs::write(&sentinel, "preserve-until-commit").unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex")
        .with_codex_registration(true, true);
    let setup_runner = MockSetupRunner::default();
    let install_options = PluginInstallOptions {
        force: true,
        ..options(&selected_dir)
    };

    let error =
        install_host(CodingAgent::Codex, &install_options, &runner, &setup_runner).unwrap_err();

    let current = PluginLayout::new(CodingAgent::Codex, &selected_dir);
    assert!(
        error.contains("outside the selected install layout"),
        "{error}"
    );
    assert!(!current.marketplace_root.exists());
    assert!(relocated.marketplace_root.exists());
    assert!(relocated.generation_lock.exists());
    assert_eq!(
        std::fs::read_to_string(&sentinel).unwrap(),
        "preserve-until-commit"
    );
    assert_no_force_replacement_residue(&selected_dir);
    assert_no_force_replacement_residue(&relocated_dir);
}

#[cfg(unix)]
#[test]
fn persisted_roots_accept_an_equivalent_symlinked_install_path() {
    use std::os::unix::fs::symlink;

    let dir = tempdir().unwrap();
    let canonical = dir.path().join("canonical");
    let selected = dir.path().join("selected");
    std::fs::create_dir_all(&canonical).unwrap();
    symlink(&canonical, &selected).unwrap();
    let selected_layout = PluginLayout::new(CodingAgent::Codex, &selected);
    std::fs::create_dir_all(&selected_layout.plugin_root).unwrap();
    let canonical_layout = PluginLayout::new(CodingAgent::Codex, &canonical);
    let state = PluginState {
        marketplace_root: canonical_layout.marketplace_root,
        plugin_root: canonical_layout.plugin_root,
        host_plugin_removed: false,
        host_marketplace_removed: false,
        plugin_setup_installed: true,
    };

    selected_layout.validate_persisted_state(&state).unwrap();
}

#[test]
fn uninstall_rejects_persisted_roots_outside_selected_layout() {
    let dir = tempdir().unwrap();
    let selected_dir = dir.path().join("selected");
    let relocated_dir = dir.path().join("relocated");
    let relocated = write_relocated_codex_install(&selected_dir, &relocated_dir);
    let sentinel = relocated.plugin_root.join("relocated-install");
    std::fs::write(&sentinel, "restore-exactly").unwrap();
    let original_state = std::fs::read(state_path(CodingAgent::Codex, &selected_dir)).unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex")
        .with_codex_registration(true, true);
    let setup_runner = MockSetupRunner::default();

    let error = uninstall_host(
        CodingAgent::Codex,
        &options(&selected_dir),
        &runner,
        &setup_runner,
    )
    .unwrap_err();

    assert!(
        error.contains("outside the selected install layout"),
        "{error}"
    );
    let current = PluginLayout::new(CodingAgent::Codex, &selected_dir);
    assert!(!current.marketplace_root.exists());
    assert!(!current.generation_lock.exists());
    assert!(relocated.marketplace_root.exists());
    assert!(relocated.generation_lock.exists());
    assert_eq!(
        std::fs::read_to_string(&sentinel).unwrap(),
        "restore-exactly"
    );
    assert_eq!(
        std::fs::read(state_path(CodingAgent::Codex, &selected_dir)).unwrap(),
        original_state
    );
    assert!(runner.commands().is_empty());
    assert!(setup_runner.calls().is_empty());
    assert_no_force_replacement_residue(&selected_dir);
    assert_no_force_replacement_residue(&relocated_dir);
}

#[test]
fn force_install_rejects_registered_legacy_plugin_without_generation_fence() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex")
        .with_capture_output(
            "/bin/codex plugin list",
            "nemo-relay-plugin@nemo-relay-local installed, enabled\n",
        )
        .with_capture_output(
            "/bin/codex plugin marketplace list",
            "nemo-relay-local /tmp/nemo-relay-local\n",
        );
    let setup_runner = MockSetupRunner::default();
    let options = PluginInstallOptions {
        force: true,
        ..options(dir.path())
    };

    let error = install_host(CodingAgent::Codex, &options, &runner, &setup_runner).unwrap_err();

    assert!(
        error.contains("MCP generation marker is missing"),
        "{error}"
    );
    assert!(error.contains("close all Codex clients"), "{error}");
    assert!(error.contains("codex plugin remove"), "{error}");
    assert!(runner.commands().is_empty());
    assert!(setup_runner.calls().is_empty());
    assert_no_install_stage(dir.path());
}

#[test]
fn force_install_rejects_unregistered_legacy_plugin_without_generation_fence() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex")
        .with_codex_registration(false, false);
    let setup_runner = MockSetupRunner::default();
    let options = PluginInstallOptions {
        force: true,
        ..options(dir.path())
    };
    write_installed_state(CodingAgent::Codex, dir.path());
    let layout = PluginLayout::new(CodingAgent::Codex, dir.path());
    std::fs::remove_file(&layout.generation_fence).unwrap();

    let error = install_host(CodingAgent::Codex, &options, &runner, &setup_runner).unwrap_err();

    assert!(
        error.contains("MCP generation marker is missing"),
        "{error}"
    );
    assert!(layout.marketplace_root.exists());
    assert!(layout.state_path.exists());
    assert!(runner.commands().is_empty());
    assert!(setup_runner.calls().is_empty());
    assert_no_install_stage(dir.path());
}

#[test]
fn force_install_rejects_corrupt_generation_marker_without_mutating() {
    for (corruption, cause) in [
        ("empty", "is empty"),
        ("oversized", "exceeds the 128-byte limit"),
        ("unreadable", "failed to"),
    ] {
        let dir = tempdir().unwrap();
        let runner = MockRunner::default()
            .with_executable("nemo-relay", "/bin/nemo-relay")
            .with_executable("codex", "/bin/codex")
            .with_codex_registration(false, false);
        let setup_runner = MockSetupRunner::default();
        let options = PluginInstallOptions {
            force: true,
            ..options(dir.path())
        };
        write_installed_state(CodingAgent::Codex, dir.path());
        let layout = PluginLayout::new(CodingAgent::Codex, dir.path());
        corrupt_generation_fence(&layout.generation_fence, corruption);

        let error = install_host(CodingAgent::Codex, &options, &runner, &setup_runner).unwrap_err();

        assert_actionable_generation_error(&error, "is invalid or unreadable");
        assert!(error.contains(cause), "{corruption}: {error}");
        assert!(layout.marketplace_root.exists());
        assert!(layout.state_path.exists());
        assert!(runner.commands().is_empty());
        assert!(setup_runner.calls().is_empty());
        assert_no_install_stage(dir.path());
    }
}

#[test]
fn force_install_allows_a_clean_first_install_without_generation_fence() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex")
        .with_codex_registration(false, false);
    let setup_runner = MockSetupRunner::default();
    let options = PluginInstallOptions {
        force: true,
        ..options(dir.path())
    };

    install_host(CodingAgent::Codex, &options, &runner, &setup_runner).unwrap();

    let layout = PluginLayout::new(CodingAgent::Codex, dir.path());
    assert!(layout.generation_fence.exists());
    assert!(layout.state_path.exists());
}

#[test]
fn force_install_uses_live_absent_registration_instead_of_stale_installed_state() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex")
        .with_codex_registration(false, false);
    let setup_runner = MockSetupRunner::default();
    let options = PluginInstallOptions {
        force: true,
        ..options(dir.path())
    };
    write_installed_state(CodingAgent::Codex, dir.path());

    install_host(CodingAgent::Codex, &options, &runner, &setup_runner).unwrap();

    let commands = runner.commands();
    assert!(
        commands
            .iter()
            .all(|command| !command.contains("plugin remove")
                && !command.contains("marketplace remove")),
        "unexpected removal commands: {commands:?}"
    );
    assert!(
        commands
            .iter()
            .any(|command| command.ends_with("plugin add nemo-relay-plugin@nemo-relay-local"))
    );
}

#[test]
fn force_install_uses_live_present_registration_instead_of_stale_removed_state() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex")
        .with_codex_registration(true, true);
    let setup_runner = MockSetupRunner::default();
    let options = PluginInstallOptions {
        force: true,
        ..options(dir.path())
    };
    write_installed_state(CodingAgent::Codex, dir.path());
    let layout = PluginLayout::new(CodingAgent::Codex, dir.path());
    write_state_for_host(
        CodingAgent::Codex,
        &PluginState {
            marketplace_root: layout.marketplace_root.clone(),
            plugin_root: layout.plugin_root.clone(),
            host_plugin_removed: true,
            host_marketplace_removed: true,
            plugin_setup_installed: true,
        },
        dir.path(),
        &options,
    )
    .unwrap();

    install_host(CodingAgent::Codex, &options, &runner, &setup_runner).unwrap();

    let commands = runner.commands();
    assert!(commands.iter().any(|command| {
        command == "/bin/codex plugin remove nemo-relay-plugin@nemo-relay-local"
    }));
    assert!(
        commands
            .iter()
            .any(|command| { command == "/bin/codex plugin marketplace remove nemo-relay-local" })
    );
}

#[test]
fn force_install_commit_does_not_fail_when_backup_cleanup_errors() {
    let dir = tempdir().unwrap();
    let backup = dir.path().join("codex-marketplace-backup");
    std::fs::write(&backup, "not a directory").unwrap();

    ForceInstallSnapshot {
        state_bytes: None,
        setup_snapshot: None,
        plugin_registered: false,
        marketplace_registered: false,
        original_marketplace_root: dir.path().join("original-marketplace"),
        original_plugin_root: dir
            .path()
            .join("original-marketplace/plugins/nemo-relay-plugin"),
        original_generation_fence: dir
            .path()
            .join("original-marketplace/plugins/nemo-relay-plugin/.nemo-relay-generation"),
        backup_marketplace_root: backup.clone(),
        backup_plugin_root: None,
        marketplace_moved: true,
        plugin_moved: false,
        replacement_promoted: true,
        generation_retirement: None,
    }
    .commit(&dir.path().join("replacement.lock"));

    assert!(backup.is_file());
}

#[test]
fn force_install_keeps_existing_registration_when_gateway_refresh_fails() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex")
        .with_codex_registration(true, true);
    let setup_runner = MockSetupRunner {
        failing_call: Some("refresh gateway".into()),
        ..MockSetupRunner::default()
    };
    let options = PluginInstallOptions {
        force: true,
        ..options(dir.path())
    };
    write_installed_state(CodingAgent::Codex, dir.path());
    let layout = PluginLayout::new(CodingAgent::Codex, dir.path());
    let previous = InstallGeneration::capture(layout.generation_fence.clone()).unwrap();

    let error = install_host(CodingAgent::Codex, &options, &runner, &setup_runner).unwrap_err();

    assert!(error.contains("refresh gateway failed"));
    assert!(layout.state_path.exists());
    assert!(layout.plugin_root.exists());
    previous.verify_current().unwrap();
    assert_eq!(
        runner.commands(),
        vec![
            "/bin/codex plugin remove nemo-relay-plugin@nemo-relay-local".to_string(),
            "/bin/codex plugin marketplace remove nemo-relay-local".to_string(),
        ]
    );
    assert_eq!(
        setup_runner.calls(),
        vec![
            "snapshot codex".to_string(),
            format!("uninstall codex {DEFAULT_GATEWAY_URL}"),
            "refresh gateway".to_string(),
            "restore snapshot".to_string(),
        ]
    );
}

#[test]
fn failed_force_refresh_hides_transient_generation_retirement_from_mcp() {
    let dir = tempdir().unwrap();
    write_installed_state(CodingAgent::Codex, dir.path());
    let layout = PluginLayout::new(CodingAgent::Codex, dir.path());
    let previous = InstallGeneration::capture(layout.generation_fence.clone()).unwrap();
    let install_dir = dir.path().to_path_buf();
    let (entered_tx, entered_rx) = std::sync::mpsc::channel();
    let (continue_tx, continue_rx) = std::sync::mpsc::channel();

    let install = std::thread::spawn(move || {
        let runner = MockRunner::default()
            .with_executable("nemo-relay", "/bin/nemo-relay")
            .with_executable("codex", "/bin/codex")
            .with_codex_registration(true, true);
        let setup_runner = BlockingRefreshFailure {
            entered: entered_tx,
            continue_refresh: continue_rx,
        };
        let options = PluginInstallOptions {
            force: true,
            ..options(&install_dir)
        };
        install_host(CodingAgent::Codex, &options, &runner, &setup_runner)
    });

    entered_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    let (verified_tx, verified_rx) = std::sync::mpsc::channel();
    let verifier = std::thread::spawn(move || verified_tx.send(previous.verify_current()).unwrap());
    assert!(
        verified_rx
            .recv_timeout(Duration::from_millis(100))
            .is_err(),
        "MCP observed the force-install retirement before refresh committed"
    );

    continue_tx.send(()).unwrap();
    let error = install.join().unwrap().unwrap_err();
    assert!(error.contains("refresh gateway failed"), "{error}");
    verified_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    verifier.join().unwrap();
    assert!(layout.plugin_root.exists());
}

#[test]
fn replacement_retirement_aggregates_refresh_and_restore_failures_without_rewriting_new_tree() {
    let dir = tempdir().unwrap();
    write_installed_state(CodingAgent::Codex, dir.path());
    let layout = PluginLayout::new(CodingAgent::Codex, dir.path());
    let backup = dir.path().join("replacement-backup");
    let install_dir = dir.path().to_path_buf();
    let retirement_layout = layout.clone();
    let (entered_tx, entered_rx) = std::sync::mpsc::channel();
    let (continue_tx, continue_rx) = std::sync::mpsc::channel();

    let retirement = std::thread::spawn(move || {
        let setup_runner = BlockingRefreshFailure {
            entered: entered_tx,
            continue_refresh: continue_rx,
        };
        retire_replacement_before_rollback(
            CodingAgent::Codex,
            &retirement_layout,
            &options(&install_dir),
            &setup_runner,
            None,
        )
    });

    entered_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    std::fs::rename(&layout.plugin_root, &backup).unwrap();
    let replacement_token = crate::installation::generation::write_staged_generation_with_token(
        &layout.generation_fence,
        &layout.generation_lock,
    )
    .unwrap();
    let replacement_marker = std::fs::read(&layout.generation_fence).unwrap();
    continue_tx.send(()).unwrap();

    let error = match retirement.join().unwrap() {
        Ok(_) => panic!("replacement retirement unexpectedly succeeded"),
        Err(error) => error,
    };
    assert!(error.contains("refresh gateway failed"), "{error}");
    assert!(error.contains("lock identity changed"), "{error}");
    assert_eq!(
        std::fs::read(&layout.generation_fence).unwrap(),
        replacement_marker
    );
    assert_eq!(
        InstallGeneration::capture(layout.generation_fence)
            .unwrap()
            .token(),
        replacement_token
    );
}

#[test]
fn uninstall_restores_mcp_generation_when_gateway_refresh_fails() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex");
    let setup_runner = MockSetupRunner {
        failing_call: Some("refresh gateway".into()),
        ..MockSetupRunner::default()
    };
    let options = options(dir.path());
    write_installed_state(CodingAgent::Codex, dir.path());
    let layout = PluginLayout::new(CodingAgent::Codex, dir.path());
    let previous = InstallGeneration::capture(layout.generation_fence.clone()).unwrap();

    let error = uninstall_host(CodingAgent::Codex, &options, &runner, &setup_runner).unwrap_err();

    assert!(error.contains("refresh gateway failed"));
    previous.verify_current().unwrap();
    assert!(layout.state_path.exists());
    assert!(layout.plugin_root.exists());
    assert_eq!(setup_runner.calls(), vec!["refresh gateway"]);
    assert!(runner.commands().is_empty());
}

#[test]
fn force_install_restores_previous_install_after_doctor_failure() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex")
        .with_codex_registration(true, true);
    let setup_runner = MockSetupRunner {
        failing_call: Some(format!("doctor codex {DEFAULT_GATEWAY_URL}")),
        ..MockSetupRunner::default()
    };
    let options = PluginInstallOptions {
        force: true,
        skip_doctor: false,
        ..options(dir.path())
    };
    write_installed_state(CodingAgent::Codex, dir.path());
    let layout = PluginLayout::new(CodingAgent::Codex, dir.path());
    let sentinel = layout.plugin_root.join("previous-install");
    std::fs::write(&sentinel, "preserve").unwrap();
    let original_state = std::fs::read(&layout.state_path).unwrap();

    let error = install_host(CodingAgent::Codex, &options, &runner, &setup_runner).unwrap_err();

    assert!(error.contains("doctor codex"), "{error}");
    assert_eq!(std::fs::read_to_string(sentinel).unwrap(), "preserve");
    assert_eq!(std::fs::read(&layout.state_path).unwrap(), original_state);
    assert!(
        setup_runner
            .calls()
            .iter()
            .any(|call| call == "restore snapshot")
    );
    let setup_calls = setup_runner.calls();
    let refreshes = setup_calls
        .iter()
        .enumerate()
        .filter(|(_, call)| call.as_str() == "refresh gateway")
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    assert_eq!(
        refreshes.len(),
        2,
        "the previous and replacement MCP generations must each be stopped: {setup_calls:?}"
    );
    let restore_index = setup_calls
        .iter()
        .position(|call| call == "restore snapshot")
        .unwrap();
    assert!(refreshes[1] < restore_index);
    assert!(std::fs::read_dir(dir.path()).unwrap().all(|entry| {
        let name = entry.unwrap().file_name();
        let name = name.to_string_lossy();
        !name.contains("install-stage") && !name.contains("marketplace-backup")
    }));
}

#[test]
fn force_install_restores_previous_install_after_state_write_failure() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex")
        .with_codex_registration(true, true);
    let install_options = PluginInstallOptions {
        force: true,
        ..options(dir.path())
    };
    write_installed_state(CodingAgent::Codex, dir.path());
    let layout = PluginLayout::new(CodingAgent::Codex, dir.path());
    let sentinel = layout.plugin_root.join("previous-install");
    std::fs::write(&sentinel, "preserve").unwrap();
    let original_state = std::fs::read(&layout.state_path).unwrap();
    let previous = InstallGeneration::capture(layout.generation_fence.clone()).unwrap();
    let setup_runner = FailStateWriteAfterRefresh {
        state_path: layout.state_path.clone(),
        injected: Cell::new(false),
    };

    let error =
        install_host(CodingAgent::Codex, &install_options, &runner, &setup_runner).unwrap_err();

    assert!(error.contains("injected test failure"), "{error}");
    assert_eq!(std::fs::read_to_string(sentinel).unwrap(), "preserve");
    assert_eq!(std::fs::read(&layout.state_path).unwrap(), original_state);
    previous.verify_current().unwrap();
    assert_no_force_replacement_residue(dir.path());
}

#[test]
fn first_install_cleans_generated_marketplace_after_state_write_failure() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex");
    let layout = PluginLayout::new(CodingAgent::Codex, dir.path());
    crate::filesystem::fail_next_atomic_write(&layout.state_path);

    let error = install_host(
        CodingAgent::Codex,
        &options(dir.path()),
        &runner,
        &MockSetupRunner::default(),
    )
    .unwrap_err();

    assert!(error.contains("injected test failure"), "{error}");
    assert!(!layout.marketplace_root.exists());
    assert!(!layout.state_path.exists());
    assert!(!layout.generation_lock.exists());
}

#[test]
fn force_replacement_restoration_aggregates_independent_cleanup_failures() {
    let dir = tempdir().unwrap();
    let install_file = dir.path().join("install-file");
    std::fs::write(&install_file, "not a directory").unwrap();
    let layout = PluginLayout::new(CodingAgent::Codex, &install_file);
    let original_marketplace_root = dir.path().join("original-marketplace");
    let original_plugin_root = dir.path().join("original-plugin");
    let mut snapshot = ForceInstallSnapshot {
        state_bytes: Some(b"original state".to_vec()),
        setup_snapshot: Some(PluginSetupSnapshot::Mock),
        original_marketplace_root: original_marketplace_root.clone(),
        original_plugin_root: original_plugin_root.clone(),
        original_generation_fence: original_plugin_root.join(GENERATION_FILE_NAME),
        plugin_registered: false,
        marketplace_registered: false,
        backup_marketplace_root: dir.path().join("missing-marketplace-backup"),
        backup_plugin_root: Some(dir.path().join("missing-plugin-backup")),
        marketplace_moved: true,
        plugin_moved: true,
        replacement_promoted: true,
        generation_retirement: None,
    };
    let mut runner = MockRunner::default()
        .with_executable("codex", "/bin/codex")
        .with_codex_registration(true, true);
    runner.failing_suffixes = vec![
        "plugin remove nemo-relay-plugin@nemo-relay-local".into(),
        "plugin marketplace remove nemo-relay-local".into(),
    ];
    let setup_runner = MockSetupRunner {
        failing_call: Some("restore snapshot".into()),
        ..MockSetupRunner::default()
    };

    let error = restore_force_replacement_after_error::<()>(
        CodingAgent::Codex,
        &layout,
        &mut snapshot,
        &options(&install_file),
        &runner,
        &setup_runner,
        "replacement failed".into(),
    )
    .unwrap_err();

    assert!(error.contains("replacement failed"), "{error}");
    assert!(
        error.contains("failed to restore previous install"),
        "{error}"
    );
    assert!(error.contains("plugin remove"), "{error}");
    assert!(error.contains("plugin marketplace remove"), "{error}");
    assert!(error.contains("failed to restore marketplace"), "{error}");
    assert!(error.contains("failed to restore plugin root"), "{error}");
    assert!(error.contains("restore snapshot failed"), "{error}");
    assert!(error.contains("failed to restore"), "{error}");
}

#[test]
fn force_replacement_restoration_reports_failed_host_reregistration() {
    let dir = tempdir().unwrap();
    let layout = PluginLayout::new(CodingAgent::Codex, dir.path());
    let original_marketplace_root = dir.path().join("original-marketplace");
    let mut snapshot = ForceInstallSnapshot {
        state_bytes: None,
        setup_snapshot: None,
        original_marketplace_root: original_marketplace_root.clone(),
        original_plugin_root: original_marketplace_root.join("plugins/nemo-relay-plugin"),
        original_generation_fence: original_marketplace_root.join(GENERATION_FILE_NAME),
        plugin_registered: true,
        marketplace_registered: true,
        backup_marketplace_root: dir.path().join("unused-marketplace-backup"),
        backup_plugin_root: None,
        marketplace_moved: false,
        plugin_moved: false,
        replacement_promoted: false,
        generation_retirement: None,
    };
    let mut runner = MockRunner::default()
        .with_executable("codex", "/bin/codex")
        .with_codex_registration(false, false);
    runner.failing_suffixes = vec![
        format!(
            "plugin marketplace add {}",
            original_marketplace_root.display()
        ),
        "plugin add nemo-relay-plugin@nemo-relay-local".into(),
    ];

    let error = restore_force_replacement(
        CodingAgent::Codex,
        &layout,
        &mut snapshot,
        &options(dir.path()),
        &runner,
        &MockSetupRunner::default(),
    )
    .unwrap_err();

    assert!(error.contains("plugin marketplace add"), "{error}");
    assert!(error.contains("plugin add"), "{error}");
}

#[test]
fn force_replacement_moves_and_restores_a_separate_plugin_tree() {
    let dir = tempdir().unwrap();
    let previous_marketplace_root = dir.path().join("previous-marketplace");
    let previous_plugin_root = dir.path().join("relocated-plugin");
    std::fs::create_dir_all(&previous_marketplace_root).unwrap();
    std::fs::create_dir_all(&previous_plugin_root).unwrap();
    std::fs::write(
        previous_marketplace_root.join("marketplace.json"),
        "marketplace",
    )
    .unwrap();
    std::fs::write(previous_plugin_root.join("plugin.json"), "plugin").unwrap();
    let target = PluginLayout::new(CodingAgent::Codex, &dir.path().join("target"));
    let preflight = PluginInstallPreflight {
        persisted: None,
        state_bytes: None,
        previous_marketplace_root: previous_marketplace_root.clone(),
        previous_plugin_root: previous_plugin_root.clone(),
        previous_generation_fence: previous_plugin_root.join(GENERATION_FILE_NAME),
        plugin_registered: false,
        marketplace_registered: false,
        previous_setup_installed: false,
        previous_install_exists: true,
        generation_retirement: None,
    };
    let setup_runner = MockSetupRunner::default();
    let runner = MockRunner::default().with_executable("codex", "/bin/codex");
    let mut snapshot = begin_force_replacement(
        CodingAgent::Codex,
        &target,
        preflight,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap();

    assert!(snapshot.marketplace_moved);
    assert!(snapshot.plugin_moved);
    assert!(!previous_marketplace_root.exists());
    assert!(!previous_plugin_root.exists());

    restore_force_replacement(
        CodingAgent::Codex,
        &target,
        &mut snapshot,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap();

    assert_eq!(
        std::fs::read_to_string(previous_marketplace_root.join("marketplace.json")).unwrap(),
        "marketplace"
    );
    assert_eq!(
        std::fs::read_to_string(previous_plugin_root.join("plugin.json")).unwrap(),
        "plugin"
    );
}

#[test]
fn force_install_cleans_only_the_previous_setup_after_replacement_setup_failure() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex")
        .with_codex_registration(true, true);
    let setup_runner = MockSetupRunner {
        failing_call: Some(format!("setup codex {DEFAULT_GATEWAY_URL}")),
        ..MockSetupRunner::default()
    };
    let options = PluginInstallOptions {
        force: true,
        ..options(dir.path())
    };
    write_installed_state(CodingAgent::Codex, dir.path());
    let layout = PluginLayout::new(CodingAgent::Codex, dir.path());
    let sentinel = layout.plugin_root.join("previous-install");
    std::fs::write(&sentinel, "preserve").unwrap();
    let original_state = std::fs::read(&layout.state_path).unwrap();

    let error = install_host(CodingAgent::Codex, &options, &runner, &setup_runner).unwrap_err();

    assert!(error.contains("setup codex"), "{error}");
    assert_eq!(std::fs::read_to_string(sentinel).unwrap(), "preserve");
    assert_eq!(std::fs::read(&layout.state_path).unwrap(), original_state);
    assert!(
        setup_runner
            .calls()
            .iter()
            .any(|call| call == "restore snapshot")
    );
    assert_eq!(
        setup_runner
            .calls()
            .iter()
            .filter(|call| call.as_str() == format!("uninstall codex {DEFAULT_GATEWAY_URL}"))
            .count(),
        1,
        "only the previous setup should be removed before replacement registration"
    );
}

#[test]
fn first_install_removes_a_partially_written_marketplace() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex");
    let setup_runner = MockSetupRunner::default();
    let layout = PluginLayout::new(CodingAgent::Codex, dir.path());
    crate::filesystem::fail_next_atomic_write(&layout.plugin_manifest);

    let error = install_host(
        CodingAgent::Codex,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap_err();

    assert!(error.contains("injected test failure"), "{error}");
    assert!(!layout.marketplace_root.exists());
    assert!(!layout.state_path.exists());
    assert!(!layout.generation_lock.exists());
    assert!(runner.commands().is_empty());
    assert!(setup_runner.calls().is_empty());
}

#[test]
fn failed_staging_removes_a_new_external_generation_lock() {
    let dir = tempdir().unwrap();
    let target = PluginLayout::new(CodingAgent::Codex, dir.path());
    let stage_parent = dir.path().join("deterministic-stage");
    let staged = PluginLayout::new(CodingAgent::Codex, &stage_parent);
    crate::filesystem::fail_next_atomic_write(&staged.mcp_config);

    let error = match stage_plugin_marketplace_at(
        CodingAgent::Codex,
        Path::new("/bin/nemo-relay"),
        &target,
        true,
        &options(dir.path()),
        stage_parent.clone(),
    ) {
        Ok(_) => panic!("staging unexpectedly succeeded"),
        Err(error) => error,
    };

    assert!(error.contains("injected test failure"), "{error}");
    assert!(!stage_parent.exists());
    assert!(!target.generation_lock.exists());
}

#[test]
fn failed_staging_preserves_a_preexisting_external_generation_lock() {
    let dir = tempdir().unwrap();
    let target = PluginLayout::new(CodingAgent::Codex, dir.path());
    let orphan_marker = dir.path().join("orphan-generation");
    crate::installation::generation::write_new_generation_with_token_at(
        &orphan_marker,
        &target.generation_lock,
    )
    .unwrap();
    std::fs::remove_file(orphan_marker).unwrap();
    let original_lock = std::fs::read(&target.generation_lock).unwrap();
    let stage_parent = dir.path().join("deterministic-existing-lock-stage");
    let staged = PluginLayout::new(CodingAgent::Codex, &stage_parent);
    crate::filesystem::fail_next_atomic_write(&staged.mcp_config);

    let error = match stage_plugin_marketplace_at(
        CodingAgent::Codex,
        Path::new("/bin/nemo-relay"),
        &target,
        true,
        &options(dir.path()),
        stage_parent.clone(),
    ) {
        Ok(_) => panic!("staging unexpectedly succeeded"),
        Err(error) => error,
    };

    assert!(error.contains("injected test failure"), "{error}");
    assert!(!stage_parent.exists());
    assert_eq!(
        std::fs::read(target.generation_lock).unwrap(),
        original_lock
    );
}

#[cfg(unix)]
#[test]
fn failed_staging_preserves_a_preexisting_dangling_generation_lock_symlink() {
    use std::os::unix::fs::symlink;

    let dir = tempdir().unwrap();
    let target = PluginLayout::new(CodingAgent::Codex, dir.path());
    let symlink_target = dir.path().join("generation-lock-target");
    symlink(&symlink_target, &target.generation_lock).unwrap();
    let stage_parent = dir.path().join("deterministic-symlink-stage");
    let staged = PluginLayout::new(CodingAgent::Codex, &stage_parent);
    crate::filesystem::fail_next_atomic_write(&staged.mcp_config);

    let error = match stage_plugin_marketplace_at(
        CodingAgent::Codex,
        Path::new("/bin/nemo-relay"),
        &target,
        true,
        &options(dir.path()),
        stage_parent.clone(),
    ) {
        Ok(_) => panic!("staging unexpectedly succeeded"),
        Err(error) => error,
    };

    assert!(error.contains("generation lock"), "{error}");
    assert!(!stage_parent.exists());
    assert!(
        std::fs::symlink_metadata(&target.generation_lock)
            .unwrap()
            .file_type()
            .is_symlink()
    );
    assert!(!symlink_target.exists());
}

#[test]
fn install_claude_enables_provider_routing() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("claude", "/bin/claude");
    let setup_runner = MockSetupRunner::default();

    install_host(
        CodingAgent::ClaudeCode,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap();

    let layout = PluginLayout::new(CodingAgent::ClaudeCode, dir.path());
    assert_eq!(
        runner.commands(),
        vec![
            format!(
                "/bin/claude plugin marketplace add {}",
                layout.marketplace_root.display()
            ),
            "/bin/claude plugin install nemo-relay-plugin@nemo-relay-local --scope user".into(),
        ]
    );
    assert_eq!(
        runner.quiet_commands(),
        vec![relay_validation_command(), relay_mcp_validation_command()]
    );
    assert_eq!(
        setup_runner.calls(),
        vec![format!("setup claude-code {DEFAULT_GATEWAY_URL}")]
    );
}

#[test]
fn install_claude_rejects_hosts_without_always_load_support_before_writing() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("claude", "/bin/claude")
        .with_capture_output("/bin/claude --version", "2.1.120 (Claude Code)\n");
    let setup_runner = MockSetupRunner::default();

    let error = install_host(
        CodingAgent::ClaudeCode,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap_err();

    assert!(error.contains("requires Claude Code 2.1.121"), "{error}");
    let layout = PluginLayout::new(CodingAgent::ClaudeCode, dir.path());
    assert!(!layout.marketplace_root.exists());
    assert!(!layout.state_path.exists());
    assert!(runner.commands().is_empty());
    assert!(setup_runner.calls().is_empty());
}

#[test]
fn missing_relay_path_fails_before_generating_plugin() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default().with_executable("codex", "/bin/codex");
    let setup_runner = MockSetupRunner::default();

    let error = install_host(
        CodingAgent::Codex,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap_err();

    assert!(error.contains("nemo-relay"));
    assert!(
        !PluginLayout::new(CodingAgent::Codex, dir.path())
            .marketplace_root
            .exists()
    );
}

#[test]
fn unsupported_relay_path_fails_before_generating_plugin() {
    let dir = tempdir().unwrap();
    let mut runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex");
    runner.failing_quiet_suffix = Some("hook-forward --help".into());
    let setup_runner = MockSetupRunner::default();

    let error = install_host(
        CodingAgent::Codex,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap_err();

    assert!(error.contains("hook-forward"));
    assert!(
        !PluginLayout::new(CodingAgent::Codex, dir.path())
            .marketplace_root
            .exists()
    );
}

#[test]
fn relay_without_native_mcp_fails_codex_install_before_generating_plugin() {
    let dir = tempdir().unwrap();
    let mut runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex");
    runner.failing_quiet_suffix = Some("mcp --help".into());
    let setup_runner = MockSetupRunner::default();

    let error = install_host(
        CodingAgent::Codex,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap_err();

    assert!(error.contains("native `nemo-relay mcp` support"));
    assert!(
        !PluginLayout::new(CodingAgent::Codex, dir.path())
            .marketplace_root
            .exists()
    );
}

#[test]
fn setup_failure_rolls_back_generated_files_and_registration() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("claude", "/bin/claude");
    let setup_runner = MockSetupRunner {
        failing_call: Some(format!("setup claude-code {DEFAULT_GATEWAY_URL}")),
        ..MockSetupRunner::default()
    };

    let error = install_host(
        CodingAgent::ClaudeCode,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap_err();

    assert!(error.contains("setup claude-code"));
    let layout = PluginLayout::new(CodingAgent::ClaudeCode, dir.path());
    assert!(!layout.marketplace_root.exists());
    assert!(!layout.generation_lock.exists());
    assert!(
        runner
            .commands()
            .iter()
            .any(|command| command == "/bin/claude plugin uninstall nemo-relay-plugin")
    );
    assert!(
        setup_runner
            .calls()
            .iter()
            .any(|call| call == &format!("uninstall claude-code {DEFAULT_GATEWAY_URL}"))
    );
}

#[test]
fn doctor_failure_fails_install_and_rolls_back() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("claude", "/bin/claude");
    let setup_runner = MockSetupRunner {
        failing_call: Some(format!("doctor claude-code {DEFAULT_GATEWAY_URL}")),
        ..MockSetupRunner::default()
    };
    let options = PluginInstallOptions {
        skip_doctor: false,
        ..options(dir.path())
    };

    let error =
        install_host(CodingAgent::ClaudeCode, &options, &runner, &setup_runner).unwrap_err();

    assert!(error.contains("doctor claude-code"));
    let layout = PluginLayout::new(CodingAgent::ClaudeCode, dir.path());
    assert!(!layout.marketplace_root.exists());
    assert!(!layout.generation_lock.exists());
}

#[test]
fn registration_failure_does_not_restore_plugin_setup_that_never_ran() {
    let dir = tempdir().unwrap();
    let mut runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("claude", "/bin/claude");
    runner.failing_suffix = Some("claude-code-marketplace".into());
    let setup_runner = MockSetupRunner::default();
    let install_dir = dir.path().join("failure");

    let error = install_host(
        CodingAgent::ClaudeCode,
        &options(&install_dir),
        &runner,
        &setup_runner,
    )
    .unwrap_err();

    assert!(error.contains("plugin marketplace add"));
    assert!(
        setup_runner.calls().is_empty(),
        "setup rollback should not run before setup was attempted"
    );
    assert!(
        !PluginLayout::new(CodingAgent::ClaudeCode, &install_dir)
            .marketplace_root
            .exists()
    );
}

#[test]
fn plugin_registration_failure_rolls_back_marketplace_without_plugin_removal() {
    let dir = tempdir().unwrap();
    let mut runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex");
    runner.failing_suffix = Some("plugin add nemo-relay-plugin@nemo-relay-local".into());
    let setup_runner = MockSetupRunner::default();
    let layout = PluginLayout::new(CodingAgent::Codex, dir.path());

    let error = install_host(
        CodingAgent::Codex,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap_err();

    assert!(error.contains("plugin add"), "{error}");
    assert!(
        error.contains("nemo-relay-plugin@nemo-relay-local"),
        "{error}"
    );
    assert!(!layout.marketplace_root.exists());
    assert!(!layout.state_path.exists());
    assert!(
        runner
            .commands()
            .iter()
            .any(|command| command.ends_with("plugin marketplace remove nemo-relay-local"))
    );
    assert!(
        runner
            .commands()
            .iter()
            .all(|command| !command.contains("plugin remove nemo-relay-plugin"))
    );
    assert!(setup_runner.calls().is_empty());
}

#[test]
fn failed_marketplace_registration_rolls_back_observed_host_side_effects() {
    let dir = tempdir().unwrap();
    let layout = PluginLayout::new(CodingAgent::Codex, dir.path());
    let mut runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex")
        .with_codex_registration_sequence(&[(false, false), (false, true)]);
    runner.failing_suffix = Some(layout.marketplace_root.display().to_string());
    let setup_runner = MockSetupRunner::default();

    let error = install_host(
        CodingAgent::Codex,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap_err();

    assert!(error.contains("plugin marketplace add"), "{error}");
    assert!(
        runner
            .commands()
            .iter()
            .any(|command| command.ends_with("plugin marketplace remove nemo-relay-local"))
    );
    assert!(!layout.marketplace_root.exists());
    assert!(!layout.state_path.exists());
    assert!(!layout.generation_lock.exists());
}

#[test]
fn failed_plugin_registration_rolls_back_observed_plugin_side_effects() {
    let dir = tempdir().unwrap();
    let mut runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex")
        .with_codex_registration_sequence(&[(false, false), (true, true)]);
    runner.failing_suffix = Some("plugin add nemo-relay-plugin@nemo-relay-local".into());
    let setup_runner = MockSetupRunner::default();
    let layout = PluginLayout::new(CodingAgent::Codex, dir.path());

    let error = install_host(
        CodingAgent::Codex,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap_err();

    assert!(error.contains("plugin add"), "{error}");
    assert!(
        runner
            .commands()
            .iter()
            .any(|command| command.ends_with("plugin remove nemo-relay-plugin@nemo-relay-local"))
    );
    assert!(!layout.marketplace_root.exists());
    assert!(!layout.state_path.exists());
    assert!(!layout.generation_lock.exists());
}

#[test]
fn unverifiable_registration_failure_preserves_the_install_tree() {
    let dir = tempdir().unwrap();
    let layout = PluginLayout::new(CodingAgent::Codex, dir.path());
    let mut runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex");
    runner.failing_suffix = Some(layout.marketplace_root.display().to_string());
    runner.capture_output_sequences.get_mut().insert(
        "/bin/codex plugin list".into(),
        VecDeque::from([
            CommandOutput::success(String::new()),
            CommandOutput {
                status: 1,
                stdout: String::new(),
                stderr: "registration report unavailable".into(),
            },
        ]),
    );
    runner.capture_output_sequences.get_mut().insert(
        "/bin/codex plugin marketplace list".into(),
        VecDeque::from([CommandOutput::success(String::new())]),
    );
    let setup_runner = MockSetupRunner::default();

    let error = install_host(
        CodingAgent::Codex,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap_err();

    assert!(error.contains("refusing destructive rollback"), "{error}");
    assert!(error.contains("registration report unavailable"), "{error}");
    assert!(layout.marketplace_root.exists());
    assert!(layout.state_path.exists());
    assert!(layout.generation_lock.exists());
    InstallGeneration::capture(layout.generation_fence).unwrap();
    assert!(
        runner
            .commands()
            .iter()
            .all(|command| !command.contains("plugin marketplace remove"))
    );
}

#[test]
fn invalid_existing_state_fails_before_generating_marketplace() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex");
    let setup_runner = MockSetupRunner::default();
    let layout = PluginLayout::new(CodingAgent::Codex, dir.path());
    std::fs::create_dir_all(&layout.state_path).unwrap();

    let error = install_host(
        CodingAgent::Codex,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap_err();

    assert!(error.contains("failed to snapshot"), "{error}");
    assert!(!layout.marketplace_root.exists());
    assert!(layout.state_path.exists());
    assert!(runner.commands().is_empty());
    assert!(setup_runner.calls().is_empty());
}

#[test]
fn retry_after_partial_registration_rollback_does_not_restore_uninstalled_setup() {
    let dir = tempdir().unwrap();
    let mut runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex");
    runner.failing_suffixes = vec![
        "plugin add nemo-relay-plugin@nemo-relay-local".into(),
        "plugin marketplace remove nemo-relay-local".into(),
    ];
    let setup_runner = MockSetupRunner::default();

    let error = install_host(
        CodingAgent::Codex,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap_err();

    assert!(error.contains("additionally failed to roll back install"));
    let state = read_state(CodingAgent::Codex, dir.path()).unwrap();
    assert!(state.host_plugin_removed);
    assert!(!state.host_marketplace_removed);
    assert!(!state.plugin_setup_installed);

    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex");
    uninstall_host(
        CodingAgent::Codex,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap();

    assert_eq!(
        setup_runner.calls(),
        vec!["refresh gateway"],
        "retry cleanup may stop the gateway but must not restore provider/hooks setup that install never reached"
    );
}

#[test]
fn retry_after_failed_codex_setup_does_not_uninstall_restored_setup() {
    let dir = tempdir().unwrap();
    let mut runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex");
    runner.failing_suffix = Some("plugin marketplace remove nemo-relay-local".into());
    let setup_runner = MockSetupRunner {
        failing_call: Some(format!("setup codex {DEFAULT_GATEWAY_URL}")),
        ..MockSetupRunner::default()
    };

    let error = install_host(
        CodingAgent::Codex,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap_err();

    assert!(error.contains("additionally failed to roll back install"));
    let state = read_state(CodingAgent::Codex, dir.path()).unwrap();
    assert!(state.host_plugin_removed);
    assert!(!state.host_marketplace_removed);
    assert!(!state.plugin_setup_installed);

    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex");
    uninstall_host(
        CodingAgent::Codex,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap();

    assert!(
        setup_runner
            .calls()
            .iter()
            .all(|call| call != &format!("uninstall codex {DEFAULT_GATEWAY_URL}"))
    );
}

#[test]
fn uninstall_uses_installed_state_and_removes_marketplace() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex");
    let setup_runner = MockSetupRunner::default();
    install_host(
        CodingAgent::Codex,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap();
    let layout = PluginLayout::new(CodingAgent::Codex, dir.path());
    assert!(layout.marketplace_root.exists());

    uninstall_host(
        CodingAgent::Codex,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap();

    assert!(!layout.marketplace_root.exists());
    assert!(!layout.state_path.exists());
    let setup_calls = setup_runner.calls();
    let refresh_index = setup_calls
        .iter()
        .position(|call| call == "refresh gateway")
        .unwrap();
    let uninstall_index = setup_calls
        .iter()
        .position(|call| call == &format!("uninstall codex {DEFAULT_GATEWAY_URL}"))
        .unwrap();
    assert!(refresh_index < uninstall_index);
}

#[test]
fn uninstall_rejects_registered_legacy_plugin_without_generation_fence() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex")
        .with_codex_registration(true, true);
    let setup_runner = MockSetupRunner::default();
    write_installed_state(CodingAgent::Codex, dir.path());
    let layout = PluginLayout::new(CodingAgent::Codex, dir.path());
    std::fs::remove_file(&layout.generation_fence).unwrap();

    let error = uninstall_host(
        CodingAgent::Codex,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap_err();

    assert!(
        error.contains("MCP generation marker is missing"),
        "{error}"
    );
    assert!(error.contains("close all Codex clients"), "{error}");
    assert!(layout.marketplace_root.exists());
    assert!(layout.state_path.exists());
    assert!(runner.commands().is_empty());
    assert!(setup_runner.calls().is_empty());
}

#[test]
fn uninstall_rejects_unregistered_legacy_plugin_without_generation_fence() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex")
        .with_codex_registration(false, false);
    let setup_runner = MockSetupRunner::default();
    write_installed_state(CodingAgent::Codex, dir.path());
    let layout = PluginLayout::new(CodingAgent::Codex, dir.path());
    std::fs::remove_file(&layout.generation_fence).unwrap();

    let error = uninstall_host(
        CodingAgent::Codex,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap_err();

    assert!(
        error.contains("MCP generation marker is missing"),
        "{error}"
    );
    assert!(layout.marketplace_root.exists());
    assert!(layout.state_path.exists());
    assert!(runner.commands().is_empty());
    assert!(setup_runner.calls().is_empty());
}

#[test]
fn uninstall_rejects_each_corrupt_generation_marker_actionably() {
    for (corruption, cause) in [
        ("empty", "is empty"),
        ("oversized", "exceeds the 128-byte limit"),
        ("unreadable", "failed to"),
    ] {
        let dir = tempdir().unwrap();
        let runner = MockRunner::default()
            .with_executable("nemo-relay", "/bin/nemo-relay")
            .with_executable("codex", "/bin/codex")
            .with_codex_registration(false, false);
        let setup_runner = MockSetupRunner::default();
        write_installed_state(CodingAgent::Codex, dir.path());
        let layout = PluginLayout::new(CodingAgent::Codex, dir.path());
        corrupt_generation_fence(&layout.generation_fence, corruption);

        let error = uninstall_host(
            CodingAgent::Codex,
            &options(dir.path()),
            &runner,
            &setup_runner,
        )
        .unwrap_err();

        assert_actionable_generation_error(&error, "is invalid or unreadable");
        assert!(error.contains(cause), "{corruption}: {error}");
        assert!(layout.marketplace_root.exists());
        assert!(layout.state_path.exists());
        assert!(runner.commands().is_empty());
        assert!(setup_runner.calls().is_empty());
    }
}

#[test]
fn uninstall_continues_when_relay_is_missing() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default().with_executable("codex", "/bin/codex");
    let setup_runner = MockSetupRunner::default();
    let layout = PluginLayout::new(CodingAgent::Codex, dir.path());
    write_installed_state(CodingAgent::Codex, dir.path());

    uninstall_host(
        CodingAgent::Codex,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap();

    assert!(!layout.marketplace_root.exists());
    assert!(!layout.state_path.exists());
    assert!(
        setup_runner
            .calls()
            .iter()
            .any(|call| call == &format!("uninstall codex {DEFAULT_GATEWAY_URL}"))
    );
}

#[test]
fn doctor_json_uses_quiet_plugin_report() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex")
        .with_capture_output(
            "/bin/codex plugin list",
            "PLUGIN                              STATUS              VERSION  PATH\n\
             nemo-relay-plugin@nemo-relay-local  installed, enabled  0.4.0    /tmp/nemo-relay-plugin\n",
        )
        .with_capture_output(
            "/bin/codex plugin marketplace list",
            "MARKETPLACE        ROOT\nnemo-relay-local  /tmp/nemo-relay-local\n",
        );
    let setup_runner = MockSetupRunner::default();
    let options = options(dir.path());
    write_installed_state(CodingAgent::Codex, dir.path());

    let report =
        doctor_host_json_value(CodingAgent::Codex, &options, &runner, &setup_runner).unwrap();

    assert_eq!(
        setup_runner.calls(),
        vec![format!("doctor-json codex {DEFAULT_GATEWAY_URL}")]
    );
    assert_eq!(report["host"], json!("codex"));
    assert_eq!(report["ok"], json!(true));
    assert_eq!(report["host_registration"]["ok"], json!(true));
    assert_eq!(
        runner.capture_commands(),
        vec![
            "/bin/codex --version",
            "/bin/codex plugin list",
            "/bin/codex plugin marketplace list"
        ]
    );
}

#[test]
fn doctor_uses_plugin_root_persisted_in_install_state() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex")
        .with_codex_registration(true, true);
    let setup_runner = MockSetupRunner::default();
    let install_options = options(dir.path());
    write_installed_state(CodingAgent::Codex, dir.path());
    let layout = PluginLayout::new(CodingAgent::Codex, dir.path());
    let relocated_root = dir.path().join("relocated-plugin-root");
    std::fs::rename(&layout.plugin_root, &relocated_root).unwrap();
    write_state_for_host(
        CodingAgent::Codex,
        &PluginState {
            marketplace_root: layout.marketplace_root.clone(),
            plugin_root: relocated_root.clone(),
            host_plugin_removed: false,
            host_marketplace_removed: false,
            plugin_setup_installed: true,
        },
        dir.path(),
        &install_options,
    )
    .unwrap();

    let _readiness =
        collect_host_plugin_readiness(CodingAgent::Codex, &install_options, &runner, &setup_runner);

    assert_eq!(setup_runner.doctor_roots(), vec![relocated_root]);
}

#[test]
fn codex_doctor_reports_upgrade_remediation_for_old_and_malformed_versions() {
    for (version_output, expected_detail) in [
        ("codex-cli 0.142.9\n", "requires codex-cli 0.143.0"),
        ("codex nightly\n", "could not parse"),
    ] {
        let dir = tempdir().unwrap();
        let runner = MockRunner::default()
            .with_executable("nemo-relay", "/bin/nemo-relay")
            .with_executable("codex", "/bin/codex")
            .with_codex_registration(true, true)
            .with_capture_output("/bin/codex --version", version_output);
        let setup_runner = MockSetupRunner::default();
        let options = options(dir.path());
        write_installed_state(CodingAgent::Codex, dir.path());

        let report =
            doctor_host_json_value(CodingAgent::Codex, &options, &runner, &setup_runner).unwrap();
        let version_check = report["readiness_checks"]
            .as_array()
            .unwrap()
            .iter()
            .find(|check| check["name"] == "Codex version")
            .unwrap();
        assert_eq!(report["ok"], json!(false));
        assert_eq!(version_check["ok"], json!(false));
        assert!(
            version_check["details"]
                .as_str()
                .unwrap()
                .contains(expected_detail)
        );
        assert_eq!(
            report["remediation"],
            json!(
                "upgrade to codex-cli 0.143.0 or newer, then run `nemo-relay install codex --force`"
            )
        );

        let text_error =
            doctor_host(CodingAgent::Codex, &options, &runner, &setup_runner).unwrap_err();
        assert!(text_error.contains("remediation: upgrade to codex-cli"));
        assert!(text_error.contains("codex-cli 0.143.0 or newer"));
    }
}

#[test]
fn claude_doctor_reports_upgrade_remediation_for_old_and_malformed_versions() {
    for (version_output, expected_detail) in [
        ("2.1.120 (Claude Code)\n", "requires Claude Code 2.1.121"),
        ("Claude Code nightly\n", "could not parse"),
    ] {
        let dir = tempdir().unwrap();
        let runner = MockRunner::default()
            .with_executable("nemo-relay", "/bin/nemo-relay")
            .with_executable("claude", "/bin/claude")
            .with_claude_registration(true, true)
            .with_capture_output("/bin/claude --version", version_output);
        let setup_runner = MockSetupRunner::default();
        let options = options(dir.path());
        write_installed_state(CodingAgent::ClaudeCode, dir.path());

        let report =
            doctor_host_json_value(CodingAgent::ClaudeCode, &options, &runner, &setup_runner)
                .unwrap();
        let version_check = report["readiness_checks"]
            .as_array()
            .unwrap()
            .iter()
            .find(|check| check["name"] == "Claude Code version")
            .unwrap();
        assert_eq!(report["ok"], json!(false));
        assert_eq!(version_check["ok"], json!(false));
        assert!(
            version_check["details"]
                .as_str()
                .unwrap()
                .contains(expected_detail)
        );
        assert_eq!(
            report["remediation"],
            json!(
                "upgrade to Claude Code 2.1.121 or newer, then run `nemo-relay install claude-code --force`"
            )
        );

        let text_error =
            doctor_host(CodingAgent::ClaudeCode, &options, &runner, &setup_runner).unwrap_err();
        assert!(text_error.contains("remediation: upgrade to Claude Code"));
        assert!(text_error.contains("upgrade to Claude Code 2.1.121 or newer"));
    }
}

#[test]
fn readiness_report_marks_missing_generated_plugin_files_as_failed() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex")
        .with_capture_output(
            "/bin/codex plugin list",
            "nemo-relay-plugin@nemo-relay-local installed, enabled\n",
        )
        .with_capture_output(
            "/bin/codex plugin marketplace list",
            "nemo-relay-local /tmp/nemo-relay-local\n",
        );
    let setup_runner = MockSetupRunner::default();
    let options = options(dir.path());
    write_installed_state(CodingAgent::Codex, dir.path());
    let layout = PluginLayout::new(CodingAgent::Codex, dir.path());
    std::fs::remove_file(layout.plugin_manifest).unwrap();

    let report =
        collect_host_plugin_readiness(CodingAgent::Codex, &options, &runner, &setup_runner);

    assert!(!report.ok());
    assert!(report.checks.iter().any(|check| {
        check.name == "Generated plugin" && !check.ok && check.details.contains("missing")
    }));
    assert_eq!(
        setup_runner.calls(),
        vec![format!("doctor-json codex {DEFAULT_GATEWAY_URL}")]
    );
}

#[test]
fn readiness_report_rejects_missing_generated_mcp_server() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex");
    let setup_runner = MockSetupRunner::default();
    let options = options(dir.path());
    write_installed_state(CodingAgent::Codex, dir.path());
    let layout = PluginLayout::new(CodingAgent::Codex, dir.path());
    std::fs::remove_file(layout.mcp_config).unwrap();

    let report =
        collect_host_plugin_readiness(CodingAgent::Codex, &options, &runner, &setup_runner);

    assert!(!report.ok());
    assert!(report.checks.iter().any(|check| {
        check.name == "Generated MCP server" && !check.ok && check.details.contains("missing")
    }));
}

#[test]
fn readiness_report_rejects_missing_mcp_generation_fence() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex");
    let setup_runner = MockSetupRunner::default();
    let options = options(dir.path());
    write_installed_state(CodingAgent::Codex, dir.path());
    let layout = PluginLayout::new(CodingAgent::Codex, dir.path());
    std::fs::remove_file(layout.generation_fence).unwrap();

    let report =
        collect_host_plugin_readiness(CodingAgent::Codex, &options, &runner, &setup_runner);

    assert!(!report.ok());
    assert!(report.checks.iter().any(|check| {
        check.name == "MCP generation fence"
            && !check.ok
            && check.details.contains("failed to open")
    }));
}

#[test]
fn claude_readiness_requires_its_mcp_server_and_generation_fence() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("claude", "/bin/claude")
        .with_claude_registration(true, true);
    let setup_runner = MockSetupRunner::default();
    let options = options(dir.path());
    write_installed_state(CodingAgent::ClaudeCode, dir.path());
    let layout = PluginLayout::new(CodingAgent::ClaudeCode, dir.path());
    std::fs::remove_file(layout.mcp_config).unwrap();
    std::fs::remove_file(layout.generation_fence).unwrap();

    let report =
        collect_host_plugin_readiness(CodingAgent::ClaudeCode, &options, &runner, &setup_runner);

    assert!(!report.ok());
    for name in ["Generated MCP server", "MCP generation fence"] {
        assert!(
            report
                .checks
                .iter()
                .any(|check| check.name == name && !check.ok),
            "missing failed readiness check for {name}"
        );
    }
}

#[test]
fn readiness_report_rejects_mcp_server_for_different_binary() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex");
    let setup_runner = MockSetupRunner::default();
    let options = options(dir.path());
    write_installed_state(CodingAgent::Codex, dir.path());
    let layout = PluginLayout::new(CodingAgent::Codex, dir.path());
    let generation = InstallGeneration::capture(layout.generation_fence.clone()).unwrap();
    write_json(
        &layout.mcp_config,
        &plugin_mcp_config(
            CodingAgent::Codex,
            Path::new("/tmp/other-relay"),
            &layout.generation_fence,
            generation.token(),
        )
        .unwrap(),
    )
    .unwrap();

    let report =
        collect_host_plugin_readiness(CodingAgent::Codex, &options, &runner, &setup_runner);

    assert!(!report.ok());
    assert!(report.checks.iter().any(|check| {
        check.name == "Generated MCP server" && !check.ok && check.details.contains("unexpected")
    }));
}

#[test]
fn readiness_report_rejects_a_stale_mcp_generation_identity() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex");
    let setup_runner = MockSetupRunner::default();
    let options = options(dir.path());
    write_installed_state(CodingAgent::Codex, dir.path());
    let layout = PluginLayout::new(CodingAgent::Codex, dir.path());
    let mut mcp =
        serde_json::from_str::<Value>(&std::fs::read_to_string(&layout.mcp_config).unwrap())
            .unwrap();
    mcp["nemo-relay"]["env"]["NEMO_RELAY_MCP_GENERATION"] = json!("stale-generation");
    write_json(&layout.mcp_config, &mcp).unwrap();

    let report =
        collect_host_plugin_readiness(CodingAgent::Codex, &options, &runner, &setup_runner);

    assert!(!report.ok());
    let check = report
        .checks
        .iter()
        .find(|check| check.name == "Generated MCP server")
        .unwrap();
    assert!(!check.ok);
    assert!(check.details.contains("unexpected MCP server manifest"));
    assert!(check.details.contains("nemo-relay install codex --force"));
}

#[test]
fn generated_codex_mcp_check_allows_previously_captured_environment_names() {
    let dir = tempdir().unwrap();
    let path = dir.path().join(".mcp.json");
    let expected = json!({
        "nemo-relay": {
            "command": "/bin/nemo-relay",
            "args": ["mcp"],
            "env_vars": ["OPENAI_API_KEY"]
        }
    });
    let mut installed = expected.clone();
    installed["nemo-relay"]["env_vars"]
        .as_array_mut()
        .unwrap()
        .push(json!("NEMO_RELAY_PREVIOUSLY_DEFINED"));
    write_json(&path, &installed).unwrap();

    let result = generated_mcp_config_check(CodingAgent::Codex, &path, &expected);

    assert_eq!(result.unwrap(), format!("valid at {}", path.display()));
}

#[test]
fn generated_codex_mcp_check_accepts_a_windows_allowlist_with_a_historical_name() {
    let dir = tempdir().unwrap();
    let path = dir.path().join(".mcp.json");
    let expected_vars =
        crate::mcp_environment::forwarded_names_for_platform(std::iter::empty(), None, true);
    let expected = json!({
        "nemo-relay": {
            "command": "C:\\Program Files\\NeMo Relay\\nemo-relay.exe",
            "args": ["mcp"],
            "env_vars": expected_vars
        }
    });
    let mut installed = expected.clone();
    installed["nemo-relay"]["env_vars"]
        .as_array_mut()
        .unwrap()
        .push(json!("NEMO_RELAY_PREVIOUSLY_DEFINED"));
    write_json(&path, &installed).unwrap();

    let result =
        generated_mcp_config_check_for_platform(CodingAgent::Codex, &path, &expected, true);

    assert_eq!(result.unwrap(), format!("valid at {}", path.display()));
}

#[test]
fn generated_codex_mcp_check_rejects_malformed_or_unapproved_environment_supersets() {
    let dir = tempdir().unwrap();
    let path = dir.path().join(".mcp.json");
    let expected = json!({
        "nemo-relay": {
            "command": "/bin/nemo-relay",
            "args": ["mcp"],
            "env_vars": ["OPENAI_API_KEY"]
        }
    });

    for invalid in [
        json!({"not": "a name"}),
        json!("NEMO_RELAY_WORKER_TOKEN"),
        json!("UNRELATED_SECRET"),
        json!("OPENAI_API_KEY"),
    ] {
        let mut installed = expected.clone();
        installed["nemo-relay"]["env_vars"]
            .as_array_mut()
            .unwrap()
            .push(invalid);
        write_json(&path, &installed).unwrap();

        let error = generated_mcp_config_check(CodingAgent::Codex, &path, &expected)
            .expect_err("invalid environment superset passed doctor validation");
        assert!(error.contains("unexpected MCP server manifest"), "{error}");
        assert!(
            error.contains("nemo-relay install codex --force"),
            "{error}"
        );
    }
}

#[test]
fn generated_mcp_check_rejects_host_shape_and_non_environment_drift() {
    let dir = tempdir().unwrap();
    let path = dir.path().join(".mcp.json");
    let expected = json!({
        "nemo-relay": {
            "command": "/bin/nemo-relay",
            "args": ["mcp"],
            "env_vars": ["OPENAI_API_KEY"]
        }
    });

    let mut wrong_command = expected.clone();
    wrong_command["nemo-relay"]["command"] = json!("/bin/foreign-relay");
    write_json(&path, &wrong_command).unwrap();
    let error = generated_mcp_config_check(CodingAgent::ClaudeCode, &path, &expected).unwrap_err();
    assert!(error.contains("install claude-code --force"), "{error}");

    let expected_without_vars = json!({
        "nemo-relay": {
            "command": "/bin/nemo-relay",
            "args": ["mcp"]
        }
    });
    write_json(&path, &wrong_command).unwrap();
    let error =
        generated_mcp_config_check(CodingAgent::Codex, &path, &expected_without_vars).unwrap_err();
    assert!(error.contains("unexpected MCP server manifest"), "{error}");

    let mut actual_without_vars = expected.clone();
    actual_without_vars["nemo-relay"]
        .as_object_mut()
        .unwrap()
        .remove("env_vars");
    write_json(&path, &actual_without_vars).unwrap();
    let error = generated_mcp_config_check(CodingAgent::Codex, &path, &expected).unwrap_err();
    assert!(error.contains("unexpected MCP server manifest"), "{error}");

    write_json(&path, &wrong_command).unwrap();
    let error = generated_mcp_config_check(CodingAgent::Codex, &path, &expected).unwrap_err();
    assert!(error.contains("unexpected MCP server manifest"), "{error}");
    assert!(error.contains("install codex --force"), "{error}");
}

#[test]
fn legacy_claude_manifest_inspection_distinguishes_absent_unreadable_and_malformed_files() {
    let dir = tempdir().unwrap();
    let plugin_root = dir.path().join("plugin");
    std::fs::create_dir_all(&plugin_root).unwrap();
    assert!(!legacy_plugin_without_mcp(CodingAgent::ClaudeCode, &plugin_root).unwrap());

    let manifest = plugin_manifest_path(CodingAgent::ClaudeCode, &plugin_root);
    std::fs::create_dir_all(&manifest).unwrap();
    let error = legacy_plugin_without_mcp(CodingAgent::ClaudeCode, &plugin_root).unwrap_err();
    assert!(
        error.contains("failed to inspect legacy plugin manifest"),
        "{error}"
    );

    std::fs::remove_dir(&manifest).unwrap();
    std::fs::write(&manifest, "{not-json").unwrap();
    let error = legacy_plugin_without_mcp(CodingAgent::ClaudeCode, &plugin_root).unwrap_err();
    assert!(
        error.contains("failed to inspect legacy plugin manifest"),
        "{error}"
    );

    std::fs::write(&manifest, r#"{"name":"legacy"}"#).unwrap();
    assert!(legacy_plugin_without_mcp(CodingAgent::ClaudeCode, &plugin_root).unwrap());
    std::fs::write(&manifest, r#"{"mcpServers":{}}"#).unwrap();
    assert!(!legacy_plugin_without_mcp(CodingAgent::ClaudeCode, &plugin_root).unwrap());
}

#[test]
fn readiness_report_names_newly_required_mcp_env_vars_and_force_remediation() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex");
    let setup_runner = MockSetupRunner::default();
    let options = options(dir.path());
    write_installed_state(CodingAgent::Codex, dir.path());
    let layout = PluginLayout::new(CodingAgent::Codex, dir.path());
    let mut mcp: Value =
        serde_json::from_str(&std::fs::read_to_string(&layout.mcp_config).unwrap()).unwrap();
    mcp["nemo-relay"]["env_vars"]
        .as_array_mut()
        .unwrap()
        .retain(|name| name != "OPENAI_API_KEY");
    write_json(&layout.mcp_config, &mcp).unwrap();

    let report =
        collect_host_plugin_readiness(CodingAgent::Codex, &options, &runner, &setup_runner);

    assert!(!report.ok());
    let check = report
        .checks
        .iter()
        .find(|check| check.name == "Generated MCP server")
        .unwrap();
    assert!(!check.ok);
    assert!(check.details.contains("OPENAI_API_KEY"));
    assert!(check.details.contains("nemo-relay install codex --force"));
}

#[test]
fn readiness_report_rejects_invalid_generated_manifest_contents() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex")
        .with_capture_output(
            "/bin/codex plugin list",
            "nemo-relay-plugin@nemo-relay-local installed, enabled\n",
        )
        .with_capture_output(
            "/bin/codex plugin marketplace list",
            "nemo-relay-local /tmp/nemo-relay-local\n",
        );
    let setup_runner = MockSetupRunner::default();
    let options = options(dir.path());
    write_installed_state(CodingAgent::Codex, dir.path());
    let layout = PluginLayout::new(CodingAgent::Codex, dir.path());
    std::fs::write(
        &layout.marketplace_manifest,
        r#"{"name":"wrong-marketplace"}"#,
    )
    .unwrap();

    let report =
        collect_host_plugin_readiness(CodingAgent::Codex, &options, &runner, &setup_runner);

    assert!(!report.ok());
    assert!(report.checks.iter().any(|check| {
        check.name == "Generated marketplace" && !check.ok && check.details.contains("unexpected")
    }));
}

#[test]
fn readiness_report_accepts_generated_plugin_manifest_from_an_older_version() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex")
        .with_capture_output(
            "/bin/codex plugin list",
            "nemo-relay-plugin@nemo-relay-local installed, enabled\n",
        )
        .with_capture_output(
            "/bin/codex plugin marketplace list",
            "nemo-relay-local /tmp/nemo-relay-local\n",
        );
    let setup_runner = MockSetupRunner::default();
    let options = options(dir.path());
    write_installed_state(CodingAgent::Codex, dir.path());
    let layout = PluginLayout::new(CodingAgent::Codex, dir.path());
    let mut manifest = plugin_manifest(CodingAgent::Codex);
    manifest["version"] = json!("0.0.0");
    std::fs::write(
        &layout.plugin_manifest,
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();

    let report =
        collect_host_plugin_readiness(CodingAgent::Codex, &options, &runner, &setup_runner);

    assert!(report.ok());
    assert!(
        report
            .checks
            .iter()
            .any(|check| check.name == "Generated plugin" && check.ok)
    );
}

#[test]
fn doctor_json_preserves_unknown_host_registration_state() {
    let dir = tempdir().unwrap();
    let setup_runner = MockSetupRunner::default();
    let options = options(dir.path());
    write_installed_state(CodingAgent::Codex, dir.path());

    let report = doctor_host_json_value(
        CodingAgent::Codex,
        &options,
        &MockRunner::default(),
        &setup_runner,
    )
    .unwrap();

    assert_eq!(report["host_registration"]["ok"], json!(false));
    assert!(report["host_registration"]["host_plugin_registered"].is_null());
    assert!(report["host_registration"]["host_marketplace_registered"].is_null());
}

#[test]
fn timed_out_host_plugin_readiness_is_actionable() {
    let state_path = PathBuf::from("/tmp/nemo-relay/codex.json");
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    let _sender = sender;

    let report = crate::agents::receive_integration_readiness_for_test(
        CodingAgent::Codex,
        state_path.clone(),
        receiver,
        Path::new("/tmp/nemo-relay"),
        Duration::ZERO,
    );

    assert!(!report.ok());
    assert_eq!(report.state_path, state_path);
    assert_eq!(report.remediation, "nemo-relay install codex --force");
    assert!(
        report
            .checks
            .iter()
            .any(|check| check.name == "Host readiness" && !check.ok)
    );
}

#[test]
fn stopped_lazy_sidecar_does_not_fail_host_readiness() {
    let mut readiness = HostPluginReadiness {
        host: "codex".into(),
        remediation: "nemo-relay install codex --force".into(),
        state_path: PathBuf::from("/tmp/codex.json"),
        marketplace: None,
        plugin: None,
        checks: vec![],
        relay: None,
        host_plugin_registered: None,
        host_marketplace_registered: None,
        plugin_setup: None,
    };

    append_plugin_setup_checks(
        &mut readiness,
        &json!({
            "sidecar_health": "not_running_mcp_start",
            "checks": {
                "plugin_binary": true,
                "sidecar_running": false,
                "codex_provider_alias": true,
                "codex_hooks": true
            }
        }),
    );

    assert!(readiness.ok());
    assert!(
        readiness
            .checks
            .iter()
            .any(|check| check.name == "Sidecar health")
    );
    assert!(
        !readiness
            .checks
            .iter()
            .any(|check| check.name == "sidecar running")
    );
}

#[test]
fn doctor_validates_claude_host_registration_before_setup_doctor() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("claude", "/bin/claude")
        .with_capture_output(
            "/bin/claude plugin list --json",
            json!([
                { "id": "nemo-relay-plugin@nemo-relay-local" }
            ])
            .to_string(),
        )
        .with_capture_output(
            "/bin/claude plugin marketplace list --json",
            json!([
                { "name": "nemo-relay-local" }
            ])
            .to_string(),
        );
    let setup_runner = MockSetupRunner::default();
    let options = options(dir.path());
    write_installed_state(CodingAgent::ClaudeCode, dir.path());

    doctor_host(CodingAgent::ClaudeCode, &options, &runner, &setup_runner).unwrap();

    assert_eq!(
        setup_runner.calls(),
        vec![format!("doctor-json claude-code {DEFAULT_GATEWAY_URL}")]
    );
    assert_eq!(
        runner.capture_commands(),
        vec![
            "/bin/claude --version",
            "/bin/claude plugin list --json",
            "/bin/claude plugin marketplace list --json"
        ]
    );
}

#[test]
fn doctor_fails_when_claude_host_plugin_is_missing() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("claude", "/bin/claude")
        .with_capture_output("/bin/claude plugin list --json", json!([]).to_string())
        .with_capture_output(
            "/bin/claude plugin marketplace list --json",
            json!([
                { "name": "nemo-relay-local" }
            ])
            .to_string(),
        );
    let setup_runner = MockSetupRunner::default();
    let options = options(dir.path());
    write_installed_state(CodingAgent::ClaudeCode, dir.path());

    let error = doctor_host(CodingAgent::ClaudeCode, &options, &runner, &setup_runner).unwrap_err();

    assert!(error.contains("nemo-relay install claude-code --force"));
    assert_eq!(
        setup_runner.calls(),
        vec![format!("doctor-json claude-code {DEFAULT_GATEWAY_URL}")]
    );
}

#[test]
fn doctor_fails_when_claude_host_marketplace_is_missing() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("claude", "/bin/claude")
        .with_capture_output(
            "/bin/claude plugin list --json",
            json!([
                { "id": "nemo-relay-plugin@nemo-relay-local" }
            ])
            .to_string(),
        )
        .with_capture_output(
            "/bin/claude plugin marketplace list --json",
            json!([]).to_string(),
        );
    let setup_runner = MockSetupRunner::default();
    let options = options(dir.path());
    write_installed_state(CodingAgent::ClaudeCode, dir.path());

    let error = doctor_host(CodingAgent::ClaudeCode, &options, &runner, &setup_runner).unwrap_err();

    assert!(error.contains("nemo-relay install claude-code --force"));
    assert_eq!(
        setup_runner.calls(),
        vec![format!("doctor-json claude-code {DEFAULT_GATEWAY_URL}")]
    );
}

#[test]
fn uninstall_cleans_up_plugin_setup_before_host_removal_failure() {
    let dir = tempdir().unwrap();
    let mut runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex");
    runner.failing_suffix = Some("plugin remove nemo-relay-plugin@nemo-relay-local".into());
    let setup_runner = MockSetupRunner::default();

    let error = uninstall_host(
        CodingAgent::Codex,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap_err();

    assert!(error.contains("plugin remove"));
    assert_eq!(
        setup_runner.calls(),
        vec![
            "refresh gateway".to_string(),
            format!("uninstall codex {DEFAULT_GATEWAY_URL}"),
        ]
    );
}

#[test]
fn force_install_recovers_from_a_generation_retired_by_partial_uninstall() {
    let dir = tempdir().unwrap();
    let mut failing_runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex")
        .with_codex_registration(true, true);
    failing_runner.failing_suffix = Some("plugin remove nemo-relay-plugin@nemo-relay-local".into());
    let setup_runner = MockSetupRunner::default();
    write_installed_state(CodingAgent::Codex, dir.path());
    let layout = PluginLayout::new(CodingAgent::Codex, dir.path());

    let error = uninstall_host(
        CodingAgent::Codex,
        &options(dir.path()),
        &failing_runner,
        &setup_runner,
    )
    .unwrap_err();
    assert!(error.contains("plugin remove"), "{error}");
    assert!(
        std::fs::read_to_string(&layout.generation_fence)
            .unwrap()
            .starts_with("retired:")
    );

    let retry_runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex")
        .with_codex_registration(true, true);
    install_host(
        CodingAgent::Codex,
        &PluginInstallOptions {
            force: true,
            ..options(dir.path())
        },
        &retry_runner,
        &setup_runner,
    )
    .unwrap();

    InstallGeneration::capture(layout.generation_fence).unwrap();
    assert!(layout.generation_lock.exists());
}

#[test]
fn uninstall_does_not_unregister_host_when_plugin_cleanup_fails() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex");
    let setup_runner = MockSetupRunner {
        failing_call: Some(format!("uninstall codex {DEFAULT_GATEWAY_URL}")),
        ..MockSetupRunner::default()
    };
    write_installed_state(CodingAgent::Codex, dir.path());

    let error = uninstall_host(
        CodingAgent::Codex,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap_err();

    assert!(error.contains("uninstall codex"));
    let state = read_state(CodingAgent::Codex, dir.path()).unwrap();
    assert!(!state.host_plugin_removed);
    assert!(!state.host_marketplace_removed);
    assert!(state.plugin_setup_installed);
}

#[test]
fn uninstall_retry_skips_host_removal_after_prior_success() {
    let dir = tempdir().unwrap();
    let mut runner = MockRunner::default().with_executable("nemo-relay", "/bin/nemo-relay");
    runner.failing_suffix = Some("plugin remove nemo-relay-plugin@nemo-relay-local".into());
    let setup_runner = MockSetupRunner::default();
    let layout = PluginLayout::new(CodingAgent::Codex, dir.path());
    write_state_for_host(
        CodingAgent::Codex,
        &PluginState {
            marketplace_root: layout.marketplace_root.clone(),
            plugin_root: layout.plugin_root.clone(),
            host_plugin_removed: true,
            host_marketplace_removed: true,
            plugin_setup_installed: true,
        },
        dir.path(),
        &options(dir.path()),
    )
    .unwrap();
    crate::installation::generation::write_new_generation(&layout.generation_fence).unwrap();

    uninstall_host(
        CodingAgent::Codex,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap();

    assert!(
        runner
            .commands()
            .iter()
            .all(|command| !command.contains("plugin remove nemo-relay-plugin"))
    );
    assert!(
        setup_runner
            .calls()
            .iter()
            .any(|call| call == &format!("uninstall codex {DEFAULT_GATEWAY_URL}"))
    );
    assert!(!layout.state_path.exists());
}

#[test]
fn uninstall_retry_skips_plugin_removal_after_marketplace_failure() {
    let dir = tempdir().unwrap();
    let mut runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex");
    runner.failing_suffix = Some("plugin marketplace remove nemo-relay-local".into());
    let setup_runner = MockSetupRunner::default();
    let layout = PluginLayout::new(CodingAgent::Codex, dir.path());
    write_state(&layout, &options(dir.path())).unwrap();
    crate::installation::generation::write_new_generation(&layout.generation_fence).unwrap();

    let error = uninstall_host(
        CodingAgent::Codex,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap_err();

    assert!(error.contains("plugin marketplace remove"));
    let state = read_state(CodingAgent::Codex, dir.path()).unwrap();
    assert!(state.host_plugin_removed);
    assert!(!state.host_marketplace_removed);

    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex");
    uninstall_host(
        CodingAgent::Codex,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap();

    assert!(
        runner
            .commands()
            .iter()
            .all(|command| !command.contains("plugin remove nemo-relay-plugin"))
    );
    assert!(!layout.state_path.exists());
}
