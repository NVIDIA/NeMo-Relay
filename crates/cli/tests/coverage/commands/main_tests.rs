// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use clap::Parser;
use std::ffi::OsString;
#[cfg(unix)]
use std::io::{BufRead, BufReader};
use std::net::TcpListener;
use std::path::PathBuf;
#[cfg(unix)]
use std::process::{Command as ProcessCommand, Stdio};
#[cfg(unix)]
use std::time::{Duration, Instant};

use listeners::{Listener, Process, Protocol};

use super::completions::CompletionsCommand;
use super::serve::ServerArgs;
use super::*;
use crate::commands::configure::ConfigSubcommand;
use crate::commands::model_pricing::{PricingSubcommand, PricingValidateCommand};
use crate::commands::plugins::{
    PluginsCommand, PluginsEditCommand, PluginsInspectCommand, PluginsListCommand,
    PluginsScopeArgs, PluginsSubcommand, PluginsValidateCommand,
};
use crate::commands::root::AgentArg;

#[test]
fn plugins_edit_treats_explicit_target_as_the_user_layer() {
    let config = PathBuf::from("/managed/config.toml");
    let server = crate::server::GatewayOverrides {
        config: Some(config),
        ..crate::server::GatewayOverrides::default()
    };

    let default = PluginsEditCommand::default();
    let request = plugins::edit_request(default, &server);
    assert_eq!(
        request.explicit_path,
        Some(PathBuf::from("/managed/plugins.toml"))
    );

    let server = crate::server::GatewayOverrides {
        config: Some(PathBuf::from("/managed/config.toml")),
        plugin_config_path: Some(PathBuf::from("/override/plugins.toml")),
        ..crate::server::GatewayOverrides::default()
    };
    let request = plugins::edit_request(PluginsEditCommand::default(), &server);
    assert_eq!(
        request.explicit_path,
        Some(PathBuf::from("/override/plugins.toml"))
    );

    let user = PluginsEditCommand {
        scope: PluginsScopeArgs {
            user: true,
            ..PluginsScopeArgs::default()
        },
    };
    let request = plugins::edit_request(user, &server);
    assert_eq!(
        request.explicit_path,
        Some(PathBuf::from("/override/plugins.toml"))
    );

    let global = PluginsEditCommand {
        scope: PluginsScopeArgs {
            global: true,
            ..PluginsScopeArgs::default()
        },
    };
    let request = plugins::edit_request(global, &server);
    assert_eq!(request.explicit_path, None);
}

#[test]
fn easy_path_setup_inherits_explicit_plugin_target() {
    let plugin_config_path = PathBuf::from("/managed/plugins.toml");
    let inherited = crate::server::GatewayOverrides {
        plugin_config_path: Some(plugin_config_path.clone()),
        ..crate::server::GatewayOverrides::default()
    };

    assert_eq!(
        run::easy_path_plugin_config_path(&inherited),
        Some(plugin_config_path)
    );
}

#[test]
fn operational_command_names_cover_logging_exempt_commands() {
    for (args, expected) in [
        (vec!["nemo-relay", "codex"], "codex"),
        (vec!["nemo-relay", "config"], "config"),
        (vec!["nemo-relay", "gateway", "start"], "gateway"),
        (vec!["nemo-relay", "gateway", "stop"], "gateway"),
    ] {
        let cli = Cli::try_parse_from(args).unwrap();
        assert_eq!(cli.command.unwrap().log_name(), expected);
    }
}

#[test]
fn gateway_stop_uses_the_daemon_bind_and_parses_force() {
    let default = Cli::try_parse_from(["nemo-relay", "gateway", "stop"]).unwrap();
    assert_eq!(
        gateway::stop_bind(&default.server),
        "127.0.0.1:4040".parse().unwrap()
    );

    let explicit = Cli::try_parse_from([
        "nemo-relay",
        "--bind",
        "127.0.0.1:47632",
        "gateway",
        "stop",
        "--force",
    ])
    .unwrap();
    assert_eq!(
        gateway::stop_bind(&explicit.server),
        "127.0.0.1:47632".parse().unwrap()
    );

    let zero = gateway::validate_stop_bind("127.0.0.1:0".parse().unwrap()).unwrap_err();
    assert!(zero.to_string().contains("nonzero port"), "{zero}");
    let remote = gateway::validate_stop_bind("192.0.2.1:4040".parse().unwrap()).unwrap_err();
    assert!(remote.to_string().contains("loopback address"), "{remote}");
}

#[test]
fn integrations_refresh_parses_the_default_and_legacy_install_directory_forms() {
    let default = Cli::try_parse_from(["nemo-relay", "integrations", "refresh"]).unwrap();
    assert_eq!(default.command.unwrap().log_name(), "integrations");

    let explicit = Cli::try_parse_from([
        "nemo-relay",
        "integrations",
        "refresh",
        "--install-dir",
        "/managed/plugins",
        "--dry-run",
    ])
    .unwrap();
    assert_eq!(explicit.command.unwrap().log_name(), "integrations");
}

#[test]
fn gateway_stop_selects_only_the_exact_relay_tcp_listener() {
    let bind = "127.0.0.1:4040".parse().unwrap();
    let listeners = [
        test_listener("127.0.0.1:4041", 10, "nemo-relay", Protocol::TCP),
        test_listener("127.0.0.1:4040", 11, "nemo-relay", Protocol::UDP),
        test_listener("127.0.0.1:4040", 13, "nemo-relay", Protocol::TCP),
    ];

    let process = gateway::select_relay_listener(bind, listeners)
        .unwrap()
        .unwrap();
    assert_eq!(process.pid, 13);
}

#[test]
fn gateway_stop_refuses_foreign_or_ambiguous_listener_owners() {
    let bind = "127.0.0.1:4040".parse().unwrap();
    let foreign = gateway::select_relay_listener(
        bind,
        [test_listener(
            "127.0.0.1:4040",
            20,
            "postgres",
            Protocol::TCP,
        )],
    )
    .unwrap_err();
    assert!(
        foreign.to_string().contains("non-Relay process"),
        "{foreign}"
    );

    let ambiguous = gateway::select_relay_listener(
        bind,
        [
            test_listener("127.0.0.1:4040", 21, "nemo-relay", Protocol::TCP),
            test_listener("127.0.0.1:4040", 22, "nemo-relay", Protocol::TCP),
        ],
    )
    .unwrap_err();
    assert!(
        ambiguous.to_string().contains("multiple processes listen"),
        "{ambiguous}"
    );
}

#[cfg(unix)]
#[test]
fn gateway_force_stop_terminates_descendants() {
    let mut gateway = ProcessCommand::new("/bin/sh")
        .args(["-c", "sleep 30 & echo $!; wait"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut stdout = BufReader::new(gateway.stdout.take().unwrap());
    let mut child_pid = String::new();
    stdout.read_line(&mut child_pid).unwrap();
    let child_pid = child_pid.trim().parse::<u32>().unwrap();
    let native_child_pid = libc::pid_t::try_from(child_pid).unwrap();

    if let Err(error) = gateway::signal_process(gateway.id(), true) {
        let _ = gateway.kill();
        // SAFETY: best-effort cleanup for the test child if force-stop fails.
        unsafe { libc::kill(native_child_pid, libc::SIGKILL) };
        panic!("force stop failed: {error}");
    }
    gateway.wait().unwrap();

    let deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < deadline {
        // SAFETY: signal 0 only checks whether the test child still exists.
        if unsafe { libc::kill(native_child_pid, 0) } == -1
            && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH)
        {
            return;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    // SAFETY: best-effort cleanup for a failed process-tree assertion.
    unsafe { libc::kill(native_child_pid, libc::SIGKILL) };
    panic!("force stop left child PID {child_pid} running");
}

fn test_listener(address: &str, pid: u32, name: &str, protocol: Protocol) -> Listener {
    Listener {
        process: Process {
            pid,
            name: name.into(),
            path: format!("/test/{name}"),
        },
        socket: address.parse().unwrap(),
        protocol,
    }
}

#[tokio::test]
async fn gateway_start_uses_the_requested_bind() {
    let temp = tempfile::tempdir().unwrap();
    let _environment = EnvScope::hermetic(&temp);

    let occupied = TcpListener::bind("127.0.0.1:0").unwrap();
    let occupied_address = occupied.local_addr().unwrap();
    let start = Cli::try_parse_from([
        "nemo-relay",
        "--bind",
        &occupied_address.to_string(),
        "gateway",
        "start",
    ])
    .unwrap();
    assert!(
        run_command(start.command.unwrap(), &start.server, None, None)
            .await
            .is_err(),
        "gateway start should attempt to serve on the requested bind"
    );
}

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

    let config = cli.logging.resolve(None).unwrap();

    assert_eq!(config.level, nemo_relay::logging::LogLevel::Trace);
    assert_eq!(config.stderr_format, nemo_relay::logging::LogFormat::Jsonl);
    assert!(config.sinks.is_empty());
}

#[test]
fn cli_logging_resolves_explicit_relay_config() {
    let _cwd = crate::test_support::CwdTestScope::locked();
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

    let config = cli.logging.resolve(cli.server.config.as_deref()).unwrap();

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

    let plugins_edit = Cli::try_parse_from(["nemo-relay", "plugins", "edit", "--user"]).unwrap();
    assert!(plugins_edit.command.as_ref().unwrap().skips_logging());

    let plugins_list = Cli::try_parse_from(["nemo-relay", "plugins", "list"]).unwrap();
    assert!(!plugins_list.command.as_ref().unwrap().skips_logging());

    let agents = Cli::try_parse_from(["nemo-relay", "agents"]).unwrap();
    assert!(!agents.command.as_ref().unwrap().skips_logging());
}

#[test]
fn uninstall_logging_ignores_malformed_ambient_relay_config() {
    let _cwd = crate::test_support::CwdTestScope::locked();
    let temp = tempfile::tempdir().unwrap();
    let xdg = temp.path().join("xdg");
    let config_path = xdg.join("nemo-relay/config.toml");
    std::fs::create_dir_all(config_path.parent().unwrap()).unwrap();
    std::fs::write(&config_path, "[logging\n").unwrap();
    let _environment = crate::test_support::EnvScope::set(&[
        ("NEMO_RELAY_LOG", None),
        ("NEMO_RELAY_LOG_STDERR", None),
        ("NEMO_RELAY_LOG_STDERR_FORMAT", None),
        ("NEMO_RELAY_LOG_CONFIG_PATH", None),
        ("XDG_CONFIG_HOME", Some(xdg.as_os_str())),
        ("HOME", Some(temp.path().as_os_str())),
    ]);
    let cli = Cli::try_parse_from(["nemo-relay", "uninstall", "all"]).unwrap();

    let config = cli.logging.resolve_without_ambient_config().unwrap();

    assert_eq!(config, nemo_relay::logging::LoggingConfig::default());
}

#[test]
fn uninstall_logging_rejects_an_explicit_malformed_logging_config() {
    let _environment = crate::test_support::EnvScope::set(&[
        ("NEMO_RELAY_LOG", None),
        ("NEMO_RELAY_LOG_STDERR", None),
        ("NEMO_RELAY_LOG_STDERR_FORMAT", None),
        ("NEMO_RELAY_LOG_CONFIG_PATH", None),
    ]);
    let temp = tempfile::tempdir().unwrap();
    let config_path = temp.path().join("logging.toml");
    std::fs::write(&config_path, "[logging\n").unwrap();
    let cli = Cli::try_parse_from(vec![
        OsString::from("nemo-relay"),
        OsString::from("--log-config-path"),
        config_path.into_os_string(),
        OsString::from("uninstall"),
        OsString::from("all"),
    ])
    .unwrap();

    cli.logging.resolve_without_ambient_config().unwrap_err();
}

#[test]
fn cli_parses_force_uninstall() {
    let cli = Cli::try_parse_from(["nemo-relay", "uninstall", "all", "--force"]).unwrap();

    let Command::Uninstall(command) = cli.command.unwrap() else {
        panic!("expected uninstall command");
    };
    assert!(command.force);
}

#[test]
fn cli_parses_config_edit_scopes_and_rejects_conflicts() {
    let legacy = Cli::try_parse_from(["nemo-relay", "config", "codex"]).unwrap();
    let Command::Config(command) = legacy.command.unwrap() else {
        panic!("expected config command");
    };
    assert!(matches!(command.agent, Some(AgentArg::Codex)));

    let user = Cli::try_parse_from(["nemo-relay", "config", "edit"]).unwrap();
    let Command::Config(command) = user.command.unwrap() else {
        panic!("expected config command");
    };
    let Some(ConfigSubcommand::Edit(command)) = command.command else {
        panic!("expected config edit command");
    };
    assert!(!command.user);
    assert!(!command.global);

    let explicit = Cli::try_parse_from([
        "nemo-relay",
        "--config",
        "/managed/config.toml",
        "config",
        "edit",
    ])
    .unwrap();
    assert_eq!(
        explicit.server.config,
        Some(PathBuf::from("/managed/config.toml"))
    );

    assert!(Cli::try_parse_from(["nemo-relay", "config", "edit", "--project"]).is_err());
    assert!(Cli::try_parse_from(["nemo-relay", "plugins", "edit", "--project"]).is_err());
    assert!(Cli::try_parse_from(["nemo-relay", "model-pricing", "init", "--project"]).is_err());
    assert!(Cli::try_parse_from(["nemo-relay", "config", "--reset", "--scope", "user"]).is_err());

    let error =
        Cli::try_parse_from(["nemo-relay", "config", "edit", "--user", "--global"]).unwrap_err();
    assert_eq!(error.kind(), clap::error::ErrorKind::ArgumentConflict);

    for arguments in [
        ["nemo-relay", "config", "codex", "edit", "--reset"].as_slice(),
        ["nemo-relay", "config", "--reset", "edit"].as_slice(),
    ] {
        assert!(Cli::try_parse_from(arguments).is_err());
    }
}

#[test]
fn doctor_rejects_conflicting_agent_and_plugin_targets() {
    let error =
        Cli::try_parse_from(["nemo-relay", "doctor", "codex", "--plugin", "all"]).unwrap_err();
    assert!(error.to_string().contains("cannot be used with"));
}

#[test]
fn doctor_rejects_install_dir_without_plugin_target() {
    let error =
        Cli::try_parse_from(["nemo-relay", "doctor", "--install-dir", "/tmp/plugins"]).unwrap_err();
    assert_eq!(
        error.kind(),
        clap::error::ErrorKind::MissingRequiredArgument
    );
    assert!(error.to_string().contains("--plugin"));
}

#[test]
fn doctor_accepts_offline_flag() {
    let cli = Cli::try_parse_from(["nemo-relay", "doctor", "--offline"]).unwrap();
    match cli.command {
        Some(Command::Doctor(command)) => assert!(command.offline),
        other => panic!("expected doctor command, got {other:?}"),
    }
}

#[test]
fn agent_shortcut_parser_accepts_dry_run_before_forwarded_arguments() {
    for shortcut in ["claude", "codex"] {
        let cli = Cli::try_parse_from([
            "nemo-relay",
            shortcut,
            "--dry-run",
            "--",
            shortcut,
            "synthetic argument",
        ])
        .unwrap();
        let command = match cli.command {
            Some(Command::Claude(command)) | Some(Command::Codex(command)) => command,
            other => panic!("expected agent shortcut command, got {other:?}"),
        };
        assert!(command.dry_run);
        assert_eq!(command.command, [shortcut, "synthetic argument"]);
    }
}

#[test]
fn multi_agent_operations_attempt_every_target_before_reporting_errors() {
    let visited = std::cell::RefCell::new(Vec::new());
    let error = install::run_agent_operations(CodingAgent::ALL.to_vec(), "install", |agent| {
        visited.borrow_mut().push(agent);
        match agent {
            CodingAgent::Codex => Err(error::CliError::Install("codex failure".into())),
            CodingAgent::ClaudeCode => Ok(ExitCode::FAILURE),
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
        run_command(cli.command.unwrap(), &cli.server, None, None)
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
        run_command(cli.command.unwrap(), &cli.server, None, None)
            .await
            .unwrap(),
        ExitCode::SUCCESS
    );
}

#[tokio::test]
async fn run_command_install_requires_a_valid_bootstrap_key_except_dry_runs() {
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::hermetic(&temp);
    let install_dir = temp.path().join("plugin-install");
    let install_dir_arg = install_dir.to_string_lossy().to_string();
    let key_path = temp
        .path()
        .join("xdg/nemo-relay/bootstrap/fingerprint-hmac.key");
    std::fs::create_dir_all(key_path.parent().unwrap()).unwrap();
    std::fs::write(&key_path, [0_u8; 31]).unwrap();

    let cli = Cli::try_parse_from([
        "nemo-relay",
        "install",
        "codex",
        "--install-dir",
        install_dir_arg.as_str(),
        "--skip-doctor",
    ])
    .unwrap();
    let error = run_command(cli.command.unwrap(), &cli.server, None, None)
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("bootstrap HMAC key"));
    assert!(error.contains("invalid length 31"));
    assert!(!install_dir.exists());

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
        run_command(cli.command.unwrap(), &cli.server, None, None)
            .await
            .unwrap(),
        ExitCode::SUCCESS
    );
    assert_eq!(std::fs::read(key_path).unwrap(), [0_u8; 31]);
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
