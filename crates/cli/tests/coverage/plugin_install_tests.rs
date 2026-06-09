// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde_json::json;
use tempfile::tempdir;

use super::*;

#[derive(Default)]
struct MockRunner {
    executables: HashMap<String, PathBuf>,
    commands: RefCell<Vec<String>>,
    quiet_commands: RefCell<Vec<String>>,
    failing_suffix: Option<String>,
    failing_suffixes: Vec<String>,
    failing_quiet_suffix: Option<String>,
}

impl MockRunner {
    fn with_executable(mut self, name: &str, path: &str) -> Self {
        self.executables.insert(name.into(), PathBuf::from(path));
        self
    }

    fn commands(&self) -> Vec<String> {
        self.commands.borrow().clone()
    }

    fn quiet_commands(&self) -> Vec<String> {
        self.quiet_commands.borrow().clone()
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
        json: false,
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
        .with_executable("codex", "/bin/codex");
    let setup_runner = MockSetupRunner::default();
    let options = PluginInstallOptions {
        json: true,
        ..options(dir.path())
    };
    write_installed_state(PluginHost::Codex, dir.path());

    doctor_host(PluginHost::Codex, &options, &runner, &setup_runner).unwrap();

    assert_eq!(
        setup_runner.calls(),
        vec![format!("doctor-json codex {DEFAULT_GATEWAY_URL}")]
    );
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
