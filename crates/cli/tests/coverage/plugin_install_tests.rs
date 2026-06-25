// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde_json::json;
use tempfile::tempdir;

use super::host::{
    CommandOutput, HostRegistrationReport, format_command, host_registration_report,
    require_host_cli, require_relay, run_capture_command, run_command, run_path_command,
    validate_host_registration, validate_relay_plugin_shim,
};
use super::*;

fn plugin_install_env_lock() -> &'static Mutex<()> {
    &crate::test_support::ENV_TEST_LOCK
}

struct HomeScope<'a> {
    _guard: std::sync::MutexGuard<'a, ()>,
    prev_home: Option<std::ffi::OsString>,
    prev_userprofile: Option<std::ffi::OsString>,
}

impl<'a> HomeScope<'a> {
    fn enter(path: &Path) -> Self {
        let guard = plugin_install_env_lock()
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let prev_home = std::env::var_os("HOME");
        let prev_userprofile = std::env::var_os("USERPROFILE");
        // SAFETY: This test holds a process-wide mutex for the lifetime of the env override.
        unsafe {
            std::env::set_var("HOME", path);
            std::env::remove_var("USERPROFILE");
        }
        Self {
            _guard: guard,
            prev_home,
            prev_userprofile,
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
        }
    }
}

struct PathScope<'a> {
    _guard: std::sync::MutexGuard<'a, ()>,
    previous: Option<OsString>,
}

impl<'a> PathScope<'a> {
    fn set(path: &Path) -> Self {
        let guard = plugin_install_env_lock()
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let previous = std::env::var_os("PATH");
        // SAFETY: This test holds the process-wide environment mutex for the override lifetime.
        unsafe {
            std::env::set_var("PATH", path);
        }
        Self {
            _guard: guard,
            previous,
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
        }
    }
}

#[derive(Default)]
struct MockRunner {
    executables: HashMap<String, PathBuf>,
    commands: RefCell<Vec<String>>,
    quiet_commands: RefCell<Vec<String>>,
    capture_commands: RefCell<Vec<String>>,
    capture_outputs: HashMap<String, CommandOutput>,
    failing_suffix: Option<String>,
    failing_suffixes: Vec<String>,
    failing_quiet_suffix: Option<String>,
}

impl MockRunner {
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
        Ok(self
            .capture_outputs
            .get(&rendered)
            .cloned()
            .unwrap_or_else(|| CommandOutput::success(String::new())))
    }
}

fn command_matches_suffix(command: &str, suffix: Option<&str>) -> bool {
    suffix.is_some_and(|suffix| command.ends_with(suffix))
}

#[derive(Default)]
struct MockSetupRunner {
    calls: RefCell<Vec<String>>,
    failing_call: Option<String>,
}

impl MockSetupRunner {
    fn calls(&self) -> Vec<String> {
        self.calls.borrow().clone()
    }
}

impl PluginSetupRunner for MockSetupRunner {
    fn setup(&self, host: PluginHost, gateway_url: &str) -> Result<(), String> {
        self.record(format!("setup {} {gateway_url}", host_arg(host)))
    }

    fn uninstall(&self, host: PluginHost, gateway_url: &str) -> Result<(), String> {
        self.record(format!("uninstall {} {gateway_url}", host_arg(host)))
    }

    fn doctor(&self, host: PluginHost, gateway_url: &str) -> Result<(), String> {
        self.record(format!("doctor {} {gateway_url}", host_arg(host)))
    }

    fn doctor_json(
        &self,
        host: PluginHost,
        gateway_url: &str,
    ) -> Result<serde_json::Value, String> {
        self.record(format!("doctor-json {} {gateway_url}", host_arg(host)))?;
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
        force: false,
        dry_run: false,
        skip_doctor: true,
    }
}

fn relay_validation_command() -> String {
    "/bin/nemo-relay plugin-shim hook --help".into()
}

fn write_installed_state(host: PluginHost, dir: &Path) {
    let layout = PluginLayout::new(host, dir);
    write_state(&layout, &options(dir)).unwrap();
    mark_plugin_setup_installed(host, &layout, &options(dir)).unwrap();
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
        marketplace_manifest(PluginHost::Codex)["name"],
        json!(MARKETPLACE_NAME)
    );
    assert_eq!(
        marketplace_manifest(PluginHost::ClaudeCode)["plugins"][0]["source"],
        json!("./plugins/nemo-relay-plugin")
    );
    assert_eq!(
        plugin_manifest(PluginHost::Codex)["name"],
        json!(PLUGIN_NAME)
    );
    assert_eq!(
        plugin_hooks(PluginHost::Codex)["hooks"]["SessionStart"][0]["hooks"][0]["command"],
        json!("nemo-relay plugin-shim hook codex")
    );
    assert_eq!(
        plugin_hooks(PluginHost::ClaudeCode)["hooks"]["SessionStart"][0]["hooks"][0]["command"],
        json!("nemo-relay plugin-shim hook claude")
    );
}

#[test]
fn plugin_setup_delegates_and_dry_run_skips_runner_calls() {
    let dir = tempdir().unwrap();
    let setup_runner = MockSetupRunner::default();
    let dry_run = PluginInstallOptions {
        dry_run: true,
        ..options(dir.path())
    };

    run_plugin_setup(PluginHost::Codex, &dry_run, &setup_runner).unwrap();
    run_plugin_uninstall(PluginHost::ClaudeCode, &dry_run, &setup_runner).unwrap();
    run_plugin_doctor(PluginHost::Codex, &dry_run, &setup_runner).unwrap();
    assert!(setup_runner.calls().is_empty());

    let normal = options(dir.path());
    run_plugin_setup(PluginHost::Codex, &normal, &setup_runner).unwrap();
    run_plugin_uninstall(PluginHost::ClaudeCode, &normal, &setup_runner).unwrap();
    run_plugin_doctor(PluginHost::Codex, &normal, &setup_runner).unwrap();
    let report = run_plugin_doctor_json(PluginHost::ClaudeCode, &setup_runner).unwrap();

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
fn real_plugin_setup_runner_uses_temp_home_for_codex_and_claude_paths() {
    let dir = tempdir().unwrap();
    let _home = HomeScope::enter(dir.path());
    let runner = RealPluginSetupRunner;

    runner
        .setup(PluginHost::Codex, DEFAULT_GATEWAY_URL)
        .unwrap();
    assert!(
        runner
            .doctor(PluginHost::Codex, DEFAULT_GATEWAY_URL)
            .is_ok()
    );
    let codex_report = runner
        .doctor_json(PluginHost::Codex, DEFAULT_GATEWAY_URL)
        .unwrap();
    assert_eq!(codex_report["checks"]["codex_provider_alias"], json!(true));
    assert_eq!(codex_report["checks"]["codex_hooks"], json!(true));
    runner
        .uninstall(PluginHost::Codex, DEFAULT_GATEWAY_URL)
        .unwrap();

    runner
        .setup(PluginHost::ClaudeCode, DEFAULT_GATEWAY_URL)
        .unwrap();
    assert!(
        runner
            .doctor(PluginHost::ClaudeCode, DEFAULT_GATEWAY_URL)
            .is_ok()
    );
    let claude_report = runner
        .doctor_json(PluginHost::ClaudeCode, DEFAULT_GATEWAY_URL)
        .unwrap();
    assert_eq!(
        claude_report["checks"]["claude_provider_routing"],
        json!(true)
    );
    runner
        .uninstall(PluginHost::ClaudeCode, DEFAULT_GATEWAY_URL)
        .unwrap();
}

#[test]
fn setup_action_descriptions_cover_supported_hosts_and_actions() {
    assert_eq!(
        setup_action_description(PluginHost::Codex, "configure"),
        "configure Codex provider and hook-supervised lazy startup"
    );
    assert_eq!(
        setup_action_description(PluginHost::Codex, "restore"),
        "restore Codex provider and generated hook configuration"
    );
    assert_eq!(
        setup_action_description(PluginHost::Codex, "doctor"),
        "check Codex provider and generated hooks"
    );
    assert_eq!(
        setup_action_description(PluginHost::ClaudeCode, "configure"),
        "enable Claude Code provider routing through NeMo Relay"
    );
    assert_eq!(
        setup_action_description(PluginHost::ClaudeCode, "restore"),
        "restore Claude Code provider routing from NeMo Relay backup"
    );
    assert_eq!(
        setup_action_description(PluginHost::ClaudeCode, "doctor"),
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
    require_host_cli(PluginHost::Codex, &dry_run, &runner).unwrap();
    validate_relay_plugin_shim(Path::new("nemo-relay"), &dry_run, &runner).unwrap();
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
    let report = host_registration_report(PluginHost::Codex, &dry_run, &runner).unwrap();
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
        require_host_cli(PluginHost::Codex, &normal, &runner)
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
    runner.failing_quiet_suffix = Some("plugin-shim hook --help".into());
    assert!(
        validate_relay_plugin_shim(Path::new("/bin/nemo-relay"), &normal, &runner)
            .unwrap_err()
            .contains("plugin-shim hook")
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
    assert!(quoted.contains("\"arg with space\""));
    assert!(quoted.contains("\"quote\\\"\\$\""));

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
    let error = validate_host_registration(PluginHost::Codex, &normal, &runner).unwrap_err();
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
        let report = host_registration_report(PluginHost::ClaudeCode, &normal, &runner).unwrap();
        assert!(report.ok());
        assert!(report.host_plugin_registered);
        assert!(report.host_marketplace_registered);
    }

    // Codex `plugin list` is a text table; an installed row registers the plugin.
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
    let report = host_registration_report(PluginHost::Codex, &normal, &runner).unwrap();
    assert!(report.ok());

    // A row that is present but `not installed` must not count as registered.
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
    let report = host_registration_report(PluginHost::Codex, &normal, &runner).unwrap();
    assert!(!report.host_plugin_registered);
    assert!(report.host_marketplace_registered);

    // A plugin installed under a different marketplace id must not match.
    let runner = MockRunner::default()
        .with_executable("codex", "/bin/codex")
        .with_capture_output(
            "/bin/codex plugin list",
            format!("{PLUGIN_NAME}@other  installed, enabled  0.4.0  /tmp/other\n"),
        )
        .with_capture_output("/bin/codex plugin marketplace list", "other /tmp/other\n");
    let report = host_registration_report(PluginHost::Codex, &normal, &runner).unwrap();
    assert!(!report.ok());
    assert!(!report.host_plugin_registered);
    assert!(!report.host_marketplace_registered);
}

#[test]
fn host_registration_report_surfaces_capture_status_and_stderr_variants() {
    let dir = tempdir().unwrap();
    let normal = options(dir.path());

    // Non-JSON `claude plugin list` output surfaces a parse error. Codex no longer
    // has a JSON path, so this only applies to the host that still uses `--json`.
    let runner = MockRunner::default()
        .with_executable("claude", "/bin/claude")
        .with_capture_output("/bin/claude plugin list --json", "not json");
    assert!(
        host_registration_report(PluginHost::ClaudeCode, &normal, &runner)
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
    let error = host_registration_report(PluginHost::ClaudeCode, &normal, &runner).unwrap_err();
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
    let error = host_registration_report(PluginHost::ClaudeCode, &normal, &runner).unwrap_err();
    assert!(error.contains("exit code 5"));
    assert!(!error.contains("exit code 5:"));
}

#[test]
fn top_level_install_uninstall_and_doctor_report_empty_host_selection() {
    let dir = tempdir().unwrap();
    let empty_path = dir.path().join("empty-path");
    std::fs::create_dir_all(&empty_path).unwrap();
    let _path = PathScope::set(&empty_path);

    let install_error = install(crate::config::InstallCommand {
        host: PluginHost::All,
        install_dir: Some(dir.path().join("install")),
        force: false,
        dry_run: false,
        skip_doctor: true,
    })
    .unwrap_err()
    .to_string();
    assert!(
        install_error.contains("no supported Claude Code or Codex host CLI"),
        "error was: {install_error}"
    );

    let uninstall_error = uninstall(crate::config::UninstallCommand {
        host: PluginHost::All,
        install_dir: Some(dir.path().join("install")),
        dry_run: false,
    })
    .unwrap_err()
    .to_string();
    assert!(
        uninstall_error.contains("no installed Claude Code or Codex plugin state"),
        "error was: {uninstall_error}"
    );

    let doctor_error = doctor(PluginHost::All, Some(dir.path().join("install")), true)
        .unwrap_err()
        .to_string();
    assert!(
        doctor_error.contains("no installed Claude Code or Codex plugin state"),
        "error was: {doctor_error}"
    );
    let doctor_human_error = doctor(PluginHost::All, Some(dir.path().join("install")), false)
        .unwrap_err()
        .to_string();
    assert!(
        doctor_human_error.contains("no installed Claude Code or Codex plugin state"),
        "error was: {doctor_human_error}"
    );

    assert_eq!(
        install(crate::config::InstallCommand {
            host: PluginHost::Codex,
            install_dir: Some(dir.path().join("dry-run-install")),
            force: false,
            dry_run: true,
            skip_doctor: true,
        })
        .unwrap(),
        std::process::ExitCode::SUCCESS
    );

    let codex_doctor_error = doctor(PluginHost::Codex, Some(dir.path().join("install")), false)
        .unwrap_err()
        .to_string();
    assert!(
        codex_doctor_error.contains("required `nemo-relay` executable"),
        "error was: {codex_doctor_error}"
    );

    let codex_uninstall_error = uninstall(crate::config::UninstallCommand {
        host: PluginHost::Codex,
        install_dir: Some(dir.path().join("install")),
        dry_run: false,
    })
    .unwrap_err()
    .to_string();
    assert!(
        codex_uninstall_error.contains("required `codex` CLI"),
        "error was: {codex_uninstall_error}"
    );

    assert_eq!(host_arg(PluginHost::All), "all");
    assert_eq!(host_label(PluginHost::All), "all");
    print_json(&json!({"ok": true})).unwrap();
    assert_eq!(
        with_schema(json!({"ok": true})),
        json!({"ok": true, "schema_version": 1})
    );
    assert_eq!(with_schema(json!("not-an-object")), json!("not-an-object"));
    assert!(std::panic::catch_unwind(|| host_cli(PluginHost::All)).is_err());
}

#[test]
fn select_all_uses_operation_specific_inputs() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default().with_executable("codex", "/bin/codex");
    let selected = select_hosts(
        PluginHost::All,
        HostSelectionMode::Install,
        &options(dir.path()),
        &runner,
    )
    .unwrap();
    assert_eq!(selected, vec![PluginHost::Codex]);

    std::fs::write(
        state_path(PluginHost::ClaudeCode, dir.path()),
        r#"{"marketplaceRoot":"/tmp/m","pluginRoot":"/tmp/p"}"#,
    )
    .unwrap();
    let selected = select_hosts(
        PluginHost::All,
        HostSelectionMode::Install,
        &options(dir.path()),
        &runner,
    )
    .unwrap();
    assert_eq!(selected, vec![PluginHost::Codex]);

    let selected = select_hosts(
        PluginHost::All,
        HostSelectionMode::InstalledState,
        &options(dir.path()),
        &runner,
    )
    .unwrap();
    assert_eq!(selected, vec![PluginHost::ClaudeCode]);
}

#[test]
fn install_codex_generates_marketplace_and_runs_setup() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex");
    let setup_runner = MockSetupRunner::default();

    install_host(
        PluginHost::Codex,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap();

    let layout = PluginLayout::new(PluginHost::Codex, dir.path());
    assert!(
        !layout.hooks_path.exists(),
        "generated Codex marketplace must not also install plugin hook templates"
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
    assert_eq!(runner.quiet_commands(), vec![relay_validation_command()]);
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
    let layout = PluginLayout::new(PluginHost::ClaudeCode, dir.path());
    let stale = layout.plugin_root.join("bin").join("nemo-relay");
    std::fs::create_dir_all(stale.parent().unwrap()).unwrap();
    std::fs::write(&stale, "stale").unwrap();

    install_host(
        PluginHost::ClaudeCode,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap();

    assert!(!stale.exists());
    assert!(layout.plugin_manifest.exists());
}

#[test]
fn force_install_unregisters_existing_host_before_reinstall() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex");
    let setup_runner = MockSetupRunner::default();
    let options = PluginInstallOptions {
        force: true,
        ..options(dir.path())
    };
    write_installed_state(PluginHost::Codex, dir.path());

    install_host(PluginHost::Codex, &options, &runner, &setup_runner).unwrap();

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
            .any(|call| call == &format!("uninstall codex {DEFAULT_GATEWAY_URL}"))
    );
}

#[test]
fn force_install_without_state_unregisters_host_before_reinstall() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex");
    let setup_runner = MockSetupRunner::default();
    let options = PluginInstallOptions {
        force: true,
        ..options(dir.path())
    };

    install_host(PluginHost::Codex, &options, &runner, &setup_runner).unwrap();

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
}

#[test]
fn install_claude_enables_provider_routing() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("claude", "/bin/claude");
    let setup_runner = MockSetupRunner::default();

    install_host(
        PluginHost::ClaudeCode,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap();

    let layout = PluginLayout::new(PluginHost::ClaudeCode, dir.path());
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
    assert_eq!(runner.quiet_commands(), vec![relay_validation_command()]);
    assert_eq!(
        setup_runner.calls(),
        vec![format!("setup claude-code {DEFAULT_GATEWAY_URL}")]
    );
}

#[test]
fn missing_relay_path_fails_before_generating_plugin() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default().with_executable("codex", "/bin/codex");
    let setup_runner = MockSetupRunner::default();

    let error = install_host(
        PluginHost::Codex,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap_err();

    assert!(error.contains("nemo-relay"));
    assert!(
        !PluginLayout::new(PluginHost::Codex, dir.path())
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
    runner.failing_quiet_suffix = Some("plugin-shim hook --help".into());
    let setup_runner = MockSetupRunner::default();

    let error = install_host(
        PluginHost::Codex,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap_err();

    assert!(error.contains("plugin-shim hook"));
    assert!(
        !PluginLayout::new(PluginHost::Codex, dir.path())
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
        PluginHost::ClaudeCode,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap_err();

    assert!(error.contains("setup claude-code"));
    assert!(
        !PluginLayout::new(PluginHost::ClaudeCode, dir.path())
            .marketplace_root
            .exists()
    );
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

    let error = install_host(PluginHost::ClaudeCode, &options, &runner, &setup_runner).unwrap_err();

    assert!(error.contains("doctor claude-code"));
    assert!(
        !PluginLayout::new(PluginHost::ClaudeCode, dir.path())
            .marketplace_root
            .exists()
    );
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
        PluginHost::ClaudeCode,
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
        !PluginLayout::new(PluginHost::ClaudeCode, &install_dir)
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
    let layout = PluginLayout::new(PluginHost::Codex, dir.path());

    let error = install_host(
        PluginHost::Codex,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap_err();

    assert!(error.contains("plugin add nemo-relay-plugin"));
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
fn state_write_failure_removes_generated_marketplace() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex");
    let setup_runner = MockSetupRunner::default();
    let layout = PluginLayout::new(PluginHost::Codex, dir.path());
    std::fs::create_dir_all(&layout.state_path).unwrap();

    let error = install_host(
        PluginHost::Codex,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap_err();

    assert!(error.contains("failed to write"));
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
        PluginHost::Codex,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap_err();

    assert!(error.contains("additionally failed to roll back install"));
    let state = read_state(PluginHost::Codex, dir.path()).unwrap();
    assert!(state.host_plugin_removed);
    assert!(!state.host_marketplace_removed);
    assert!(!state.plugin_setup_installed);

    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex");
    uninstall_host(
        PluginHost::Codex,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap();

    assert!(
        setup_runner.calls().is_empty(),
        "retry cleanup must not restore provider/hooks setup that install never reached"
    );
}

#[test]
fn retry_after_setup_attempted_rollback_restores_setup() {
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
        PluginHost::Codex,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap_err();

    assert!(error.contains("additionally failed to roll back install"));
    let state = read_state(PluginHost::Codex, dir.path()).unwrap();
    assert!(state.host_plugin_removed);
    assert!(!state.host_marketplace_removed);
    assert!(state.plugin_setup_installed);

    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex");
    uninstall_host(
        PluginHost::Codex,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap();

    assert!(
        setup_runner
            .calls()
            .iter()
            .any(|call| call == &format!("uninstall codex {DEFAULT_GATEWAY_URL}"))
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
        PluginHost::Codex,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap();
    let layout = PluginLayout::new(PluginHost::Codex, dir.path());
    assert!(layout.marketplace_root.exists());

    uninstall_host(
        PluginHost::Codex,
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
fn uninstall_continues_when_relay_is_missing() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default().with_executable("codex", "/bin/codex");
    let setup_runner = MockSetupRunner::default();
    let layout = PluginLayout::new(PluginHost::Codex, dir.path());
    write_installed_state(PluginHost::Codex, dir.path());

    uninstall_host(
        PluginHost::Codex,
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
    write_installed_state(PluginHost::Codex, dir.path());

    let report =
        doctor_host_json_value(PluginHost::Codex, &options, &runner, &setup_runner).unwrap();

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
            "/bin/codex plugin list",
            "/bin/codex plugin marketplace list"
        ]
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
    write_installed_state(PluginHost::ClaudeCode, dir.path());

    doctor_host(PluginHost::ClaudeCode, &options, &runner, &setup_runner).unwrap();

    assert_eq!(
        setup_runner.calls(),
        vec![format!("doctor claude-code {DEFAULT_GATEWAY_URL}")]
    );
    assert_eq!(
        runner.capture_commands(),
        vec![
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
    write_installed_state(PluginHost::ClaudeCode, dir.path());

    let error = doctor_host(PluginHost::ClaudeCode, &options, &runner, &setup_runner).unwrap_err();

    assert!(error.contains("nemo-relay-plugin@nemo-relay-local"));
    assert!(setup_runner.calls().is_empty());
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
    write_installed_state(PluginHost::ClaudeCode, dir.path());

    let error = doctor_host(PluginHost::ClaudeCode, &options, &runner, &setup_runner).unwrap_err();

    assert!(error.contains("nemo-relay-local host marketplace"));
    assert!(setup_runner.calls().is_empty());
}

#[test]
fn uninstall_host_failure_does_not_restore_plugin_setup() {
    let dir = tempdir().unwrap();
    let mut runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex");
    runner.failing_suffix = Some("plugin remove nemo-relay-plugin@nemo-relay-local".into());
    let setup_runner = MockSetupRunner::default();

    let error = uninstall_host(
        PluginHost::Codex,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap_err();

    assert!(error.contains("plugin remove"));
    assert!(
        setup_runner.calls().is_empty(),
        "provider/hook setup should not be restored until host unregister succeeds"
    );
}

#[test]
fn uninstall_records_host_removal_phases_before_plugin_restore() {
    let dir = tempdir().unwrap();
    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex");
    let setup_runner = MockSetupRunner {
        failing_call: Some(format!("uninstall codex {DEFAULT_GATEWAY_URL}")),
        ..MockSetupRunner::default()
    };
    write_installed_state(PluginHost::Codex, dir.path());

    let error = uninstall_host(
        PluginHost::Codex,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap_err();

    assert!(error.contains("uninstall codex"));
    let state = read_state(PluginHost::Codex, dir.path()).unwrap();
    assert!(state.host_plugin_removed);
    assert!(state.host_marketplace_removed);
}

#[test]
fn uninstall_retry_skips_host_removal_after_prior_success() {
    let dir = tempdir().unwrap();
    let mut runner = MockRunner::default().with_executable("nemo-relay", "/bin/nemo-relay");
    runner.failing_suffix = Some("plugin remove nemo-relay-plugin@nemo-relay-local".into());
    let setup_runner = MockSetupRunner::default();
    let layout = PluginLayout::new(PluginHost::Codex, dir.path());
    write_state_for_host(
        PluginHost::Codex,
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

    uninstall_host(
        PluginHost::Codex,
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
    let layout = PluginLayout::new(PluginHost::Codex, dir.path());
    write_state(&layout, &options(dir.path())).unwrap();

    let error = uninstall_host(
        PluginHost::Codex,
        &options(dir.path()),
        &runner,
        &setup_runner,
    )
    .unwrap_err();

    assert!(error.contains("plugin marketplace remove"));
    let state = read_state(PluginHost::Codex, dir.path()).unwrap();
    assert!(state.host_plugin_removed);
    assert!(!state.host_marketplace_removed);

    let runner = MockRunner::default()
        .with_executable("nemo-relay", "/bin/nemo-relay")
        .with_executable("codex", "/bin/codex");
    uninstall_host(
        PluginHost::Codex,
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
