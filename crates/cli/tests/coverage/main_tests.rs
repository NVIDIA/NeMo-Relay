// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use clap::Parser;

use super::*;
use crate::config::{
    CompletionsCommand, PluginsCommand, PluginsEditCommand, PluginsInspectCommand,
    PluginsListCommand, PluginsSubcommand, PluginsValidateCommand, PricingSubcommand,
    PricingValidateCommand, ServerArgs,
};

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
fn safe_dispatch_helpers_cover_completions_and_plugins_paths() {
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
        &ServerArgs::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("interactive terminal") || error.contains("TTY"));

    assert_eq!(
        run_plugins(
            PluginsCommand {
                command: PluginsSubcommand::List(PluginsListCommand::default()),
            },
            &ServerArgs::default()
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
            &ServerArgs::default(),
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
            &ServerArgs::default(),
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
            &ServerArgs::default()
        )
        .unwrap(),
        ExitCode::SUCCESS
    );
}

#[test]
fn safe_dispatch_plugin_json_errors_return_exit_codes() {
    assert_eq!(
        run_plugins(
            PluginsCommand {
                command: PluginsSubcommand::Inspect(PluginsInspectCommand {
                    id: "missing.plugin".into(),
                    json: true,
                }),
            },
            &ServerArgs::default(),
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
            &ServerArgs::default(),
        )
        .unwrap(),
        ExitCode::from(2)
    );
}

#[tokio::test]
async fn run_command_dispatches_safe_plugin_and_install_paths() {
    let cli = Cli::try_parse_from(["nemo-relay", "plugin-shim", "install", "hermes"]).unwrap();
    let error = run_command(cli.command.unwrap(), &cli.server)
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("plugin install supports codex"));

    let cli = Cli::try_parse_from(["nemo-relay", "plugin-shim", "uninstall", "cursor"]).unwrap();
    let error = run_command(cli.command.unwrap(), &cli.server)
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("plugin uninstall supports codex"));

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
    assert!(missing.contains("could not read pricing catalog"));

    let invalid = dir.path().join("invalid.json");
    std::fs::write(&invalid, "{\"version\":2,\"entries\":[]}").unwrap();
    let invalid_error = run_pricing(PricingCommand {
        command: PricingSubcommand::Validate(PricingValidateCommand { path: invalid }),
    })
    .unwrap_err()
    .to_string();
    assert!(invalid_error.contains("invalid pricing catalog"));
}
