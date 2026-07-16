// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use clap::Parser;
use std::ffi::OsString;

use super::completions::CompletionsCommand;
use super::serve::ServerArgs;
use super::*;
use crate::commands::model_pricing::{PricingSubcommand, PricingValidateCommand};
use crate::commands::plugins::{
    PluginsCommand, PluginsEditCommand, PluginsInspectCommand, PluginsListCommand,
    PluginsSubcommand, PluginsValidateCommand,
};

#[test]
fn bootstrap_shutdown_token_is_removed_before_runtime_startup() {
    let _environment = crate::test_support::EnvScope::set(&[(
        crate::bootstrap::state::BOOTSTRAP_SHUTDOWN_TOKEN_ENV,
        Some(std::ffi::OsStr::new("private-token")),
    )]);

    assert_eq!(
        crate::take_bootstrap_shutdown_token().as_deref(),
        Some("private-token")
    );
    assert!(std::env::var_os(crate::bootstrap::state::BOOTSTRAP_SHUTDOWN_TOKEN_ENV).is_none());
}

struct EnvScope {
    _cwd_guard: Option<crate::test_support::CwdTestScope>,
    _guard: std::sync::MutexGuard<'static, ()>,
    values: Vec<(&'static str, Option<OsString>)>,
}

impl EnvScope {
    fn hermetic(temp: &tempfile::TempDir) -> Self {
        let xdg = temp.path().join("xdg");
        std::fs::create_dir_all(&xdg).unwrap();
        Self::set_with_cwd_guard(
            &[
                ("HOME", Some(temp.path().as_os_str())),
                ("XDG_CONFIG_HOME", Some(xdg.as_os_str())),
            ],
            Some(crate::test_support::CwdTestScope::locked()),
        )
    }

    fn set_with_cwd_guard(
        values: &[(&'static str, Option<&std::ffi::OsStr>)],
        cwd_guard: Option<crate::test_support::CwdTestScope>,
    ) -> Self {
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
            _cwd_guard: cwd_guard,
            _guard: guard,
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

#[test]
fn completions_helper_reports_missing_shell_and_generates_requested_shell() {
    let error = generate_completions_to(None, &mut Vec::new())
        .unwrap_err()
        .to_string();
    assert!(error.contains("missing shell argument"));

    let mut output = Vec::new();
    generate_completions_to(Some(clap_complete::Shell::Bash), &mut output).unwrap();
    let script = String::from_utf8(output).unwrap();
    assert!(script.contains("_nemo-relay"));
}

#[test]
fn cli_parses_native_mcp_subcommand_and_bind_override() {
    let cli = Cli::try_parse_from(["nemo-relay", "mcp"]).unwrap();
    assert!(matches!(cli.command, Some(Command::Mcp)));
    assert!(cli.server.bind.is_none());

    let cli = Cli::try_parse_from(["nemo-relay", "--bind", "127.0.0.1:4041", "mcp"]).unwrap();
    assert!(matches!(cli.command, Some(Command::Mcp)));
    assert_eq!(cli.server.bind.unwrap().to_string(), "127.0.0.1:4041");

    assert!(Cli::try_parse_from(["nemo-relay", "mcp", "--agent", "codex"]).is_err());
}

#[test]
fn cli_logging_options_override_environment_source() {
    let _environment = crate::test_support::EnvScope::set(&[
        (
            "NEMO_RELAY_LOG",
            Some(std::ffi::OsStr::new("intentionally-invalid")),
        ),
        ("NEMO_RELAY_LOG_STDERR_FORMAT", None),
        ("NEMO_RELAY_LOG_CONFIG_PATH", None),
    ]);
    let cli = Cli::try_parse_from([
        "nemo-relay",
        "--log-level",
        "trace",
        "--log-stderr-format",
        "jsonl",
        "agents",
    ])
    .unwrap();

    let config = cli.logging.resolve(None, false).unwrap();

    assert_eq!(config.level, nemo_relay::logging::LogLevel::Trace);
    assert_eq!(config.stderr_format, nemo_relay::logging::LogFormat::Jsonl);
    assert!(config.sinks.is_empty());
}

#[test]
fn cli_logging_resolves_explicit_relay_config() {
    let _environment = crate::test_support::EnvScope::set(&[
        ("NEMO_RELAY_LOG", None),
        ("NEMO_RELAY_LOG_STDERR_FORMAT", None),
        ("NEMO_RELAY_LOG_CONFIG_PATH", None),
    ]);
    let temp = tempfile::tempdir().unwrap();
    let config_path = temp.path().join("config.toml");
    std::fs::write(
        &config_path,
        r#"
[logging]
level = "warn"
stderr_format = "jsonl"
"#,
    )
    .unwrap();
    let cli = Cli::try_parse_from(vec![
        OsString::from("nemo-relay"),
        OsString::from("--config"),
        config_path.into_os_string(),
        OsString::from("agents"),
    ])
    .unwrap();

    let config = cli
        .logging
        .resolve(cli.server.config.as_deref(), false)
        .unwrap();

    assert_eq!(config.level, nemo_relay::logging::LogLevel::Warn);
    assert_eq!(config.stderr_format, nemo_relay::logging::LogFormat::Jsonl);
    assert!(config.sinks.is_empty());
}

#[test]
fn cli_rejects_mixed_direct_and_file_logging_options() {
    let config_path = std::env::current_dir().unwrap().join("logging.toml");
    let error = Cli::try_parse_from(vec![
        OsString::from("nemo-relay"),
        OsString::from("--log-level"),
        OsString::from("info"),
        OsString::from("--log-config-path"),
        config_path.into_os_string(),
        OsString::from("agents"),
    ])
    .unwrap_err();

    assert_eq!(error.kind(), clap::error::ErrorKind::ArgumentConflict);
    assert!(error.to_string().contains("cannot be used with"));
}

#[test]
fn command_logging_policy_excludes_only_configuration_editors() {
    let config = Cli::try_parse_from(["nemo-relay", "config"]).unwrap();
    assert!(config.command.as_ref().unwrap().skips_logging());

    let plugins_edit = Cli::try_parse_from(["nemo-relay", "plugins", "edit", "--project"]).unwrap();
    assert!(plugins_edit.command.as_ref().unwrap().skips_logging());

    let plugins_list = Cli::try_parse_from(["nemo-relay", "plugins", "list"]).unwrap();
    assert!(!plugins_list.command.as_ref().unwrap().skips_logging());

    let agents = Cli::try_parse_from(["nemo-relay", "agents"]).unwrap();
    assert!(!agents.command.as_ref().unwrap().skips_logging());
}

#[test]
fn doctor_rejects_conflicting_agent_and_plugin_targets() {
    let error =
        Cli::try_parse_from(["nemo-relay", "doctor", "codex", "--plugin", "all"]).unwrap_err();
    assert!(error.to_string().contains("cannot be used with"));
}

#[test]
fn multi_agent_operations_attempt_every_target_before_reporting_errors() {
    let visited = std::cell::RefCell::new(Vec::new());
    let error = install::run_agent_operations(CodingAgent::ALL.to_vec(), "install", |agent| {
        visited.borrow_mut().push(agent);
        match agent {
            CodingAgent::Codex => Err(error::CliError::Install("codex failure".into())),
            CodingAgent::ClaudeCode => Ok(ExitCode::FAILURE),
            CodingAgent::Hermes => Ok(ExitCode::SUCCESS),
        }
    })
    .unwrap_err()
    .to_string();

    assert_eq!(*visited.borrow(), CodingAgent::ALL);
    assert!(error.contains("codex failure"), "{error}");
}

#[test]
fn safe_dispatch_helpers_cover_completions_and_plugins_paths() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let config_path = temp.path().join("config.toml");
    std::fs::write(&config_path, "").unwrap();
    let server = ServerArgs {
        config: Some(config_path),
        ..ServerArgs::default()
    };

    assert_eq!(
        run_completions(CompletionsCommand {
            shell: Some(clap_complete::Shell::Bash),
            install: false,
        })
        .unwrap(),
        ExitCode::SUCCESS
    );

    let error = run_plugins(
        PluginsCommand {
            command: PluginsSubcommand::Edit(PluginsEditCommand::default()),
        },
        &server,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("interactive terminal") || error.contains("TTY"));

    assert_eq!(
        run_plugins(
            PluginsCommand {
                command: PluginsSubcommand::List(PluginsListCommand::default()),
            },
            &server
        )
        .unwrap(),
        ExitCode::SUCCESS
    );

    assert_eq!(
        run_plugins(
            PluginsCommand {
                command: PluginsSubcommand::Inspect(PluginsInspectCommand {
                    id: "missing.plugin".into(),
                    json: false,
                }),
            },
            &server,
        )
        .unwrap(),
        ExitCode::from(2)
    );

    assert_eq!(
        run_plugins(
            PluginsCommand {
                command: PluginsSubcommand::Validate(PluginsValidateCommand {
                    target: "missing.plugin".into(),
                    json: false,
                }),
            },
            &server,
        )
        .unwrap(),
        ExitCode::from(2)
    );

    assert_eq!(
        run_plugins(
            PluginsCommand {
                command: PluginsSubcommand::List(PluginsListCommand {
                    all: false,
                    json: false,
                }),
            },
            &server
        )
        .unwrap(),
        ExitCode::SUCCESS
    );
}

#[test]
fn safe_dispatch_plugin_json_errors_return_exit_codes() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let config_path = temp.path().join("config.toml");
    std::fs::write(&config_path, "").unwrap();
    let server = ServerArgs {
        config: Some(config_path),
        ..ServerArgs::default()
    };

    assert_eq!(
        run_plugins(
            PluginsCommand {
                command: PluginsSubcommand::Inspect(PluginsInspectCommand {
                    id: "missing.plugin".into(),
                    json: true,
                }),
            },
            &server,
        )
        .unwrap(),
        ExitCode::from(2)
    );

    assert_eq!(
        run_plugins(
            PluginsCommand {
                command: PluginsSubcommand::Validate(PluginsValidateCommand {
                    target: "missing.plugin".into(),
                    json: true,
                }),
            },
            &server,
        )
        .unwrap(),
        ExitCode::from(2)
    );
}

#[tokio::test]
async fn run_command_dispatches_safe_plugin_and_install_paths() {
    let dir = tempfile::tempdir().unwrap();
    let install_dir = dir.path().join("plugin-install");
    let install_dir_arg = install_dir.to_string_lossy().to_string();
    let cli = Cli::try_parse_from([
        "nemo-relay",
        "install",
        "codex",
        "--install-dir",
        install_dir_arg.as_str(),
        "--dry-run",
        "--skip-doctor",
    ])
    .unwrap();
    assert_eq!(
        run_command(cli.command.unwrap(), &cli.server)
            .await
            .unwrap(),
        ExitCode::SUCCESS
    );

    let cli = Cli::try_parse_from([
        "nemo-relay",
        "uninstall",
        "codex",
        "--install-dir",
        install_dir_arg.as_str(),
        "--dry-run",
    ])
    .unwrap();
    assert_eq!(
        run_command(cli.command.unwrap(), &cli.server)
            .await
            .unwrap(),
        ExitCode::SUCCESS
    );
}

#[test]
fn pricing_validate_dispatch_covers_success_read_and_parse_errors() {
    let dir = tempfile::tempdir().unwrap();
    let valid = dir.path().join("pricing.json");
    std::fs::write(
        &valid,
        serde_json::json!({
            "version": 1,
            "entries": [{
                "provider": "test",
                "model_id": "model",
                "pricing_as_of": "2026-06-04",
                "pricing_source": "unit-test",
                "rates": {
                    "input_per_million": 1.0,
                    "output_per_million": 2.0
                },
                "prompt_cache": {
                    "read_accounting": "separate"
                }
            }]
        })
        .to_string(),
    )
    .unwrap();

    assert_eq!(
        run_pricing(PricingCommand {
            command: PricingSubcommand::Validate(PricingValidateCommand {
                path: valid.clone(),
            }),
        })
        .unwrap(),
        ExitCode::SUCCESS
    );

    let missing = run_pricing(PricingCommand {
        command: PricingSubcommand::Validate(PricingValidateCommand {
            path: dir.path().join("missing.json"),
        }),
    })
    .unwrap_err()
    .to_string();
    assert!(missing.contains("could not read model pricing catalog"));

    let invalid = dir.path().join("invalid.json");
    std::fs::write(&invalid, "{\"version\":2,\"entries\":[]}").unwrap();
    let invalid_error = run_pricing(PricingCommand {
        command: PricingSubcommand::Validate(PricingValidateCommand { path: invalid }),
    })
    .unwrap_err()
    .to_string();
    assert!(invalid_error.contains("invalid model pricing catalog"));
}
