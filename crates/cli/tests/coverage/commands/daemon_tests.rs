// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn daemon_address_checks_bracketed_ipv6() {
    assert!(parse_daemon_address("http://[::1]:47632").is_ok());
    assert!(
        parse_daemon_address("https://[::]:47632")
            .unwrap_err()
            .contains("bind address")
    );
    assert!(
        parse_daemon_address("http://[2001:db8::1]:47632")
            .unwrap_err()
            .contains("must use https")
    );
}

fn command(subcommand: Option<DaemonSubcommand>) -> DaemonCommand {
    DaemonCommand {
        bind: Ipv4Addr::LOCALHOST,
        port: 47632,
        advertise_address: None,
        tls_cert: None,
        tls_key: None,
        pass_through: false,
        require_worker: false,
        max_tokens_per_identity: None,
        command: subcommand,
    }
}

#[tokio::test]
async fn daemon_execute_rejects_unspecified_listeners_without_advertisement() {
    let server = crate::commands::serve::ServerArgs::default();
    let mut daemon = command(None);
    daemon.bind = Ipv4Addr::UNSPECIFIED;
    assert!(matches!(
        execute(daemon, &server).await,
        Err(CliError::Config(message)) if message.contains("requires --advertise-address")
    ));

    let worker = command(Some(DaemonSubcommand::Worker(DaemonWorkerCommand {
        daemon_address: "http://127.0.0.1:47632".into(),
        bind: Ipv4Addr::UNSPECIFIED,
        port: None,
        advertise_address: None,
    })));
    assert!(matches!(
        execute(worker, &server).await,
        Err(CliError::Config(message)) if message == "a worker bound to 0.0.0.0 requires --advertise-address"
    ));
}

#[tokio::test]
async fn managed_bundle_command_maps_every_agent_and_platform() {
    let server = crate::commands::serve::ServerArgs::default();
    for (index, platform) in [
        ManagedPlatformArg::Linux,
        ManagedPlatformArg::Macos,
        ManagedPlatformArg::Windows,
    ]
    .into_iter()
    .enumerate()
    {
        let directory = tempfile::tempdir().expect("bundle parent");
        let output = directory.path().join(format!("bundle-{index}"));
        let dispatcher = match platform {
            ManagedPlatformArg::Windows => r"C:\ProgramData\NVIDIA\nemo-relay.exe",
            _ => "/opt/nvidia/bin/nemo-relay",
        };
        let result = execute(
            command(Some(DaemonSubcommand::ManagedBundle(
                DaemonManagedBundleCommand {
                    output: output.clone(),
                    daemon_address: "https://relay.example.com:443".into(),
                    dispatcher_command: dispatcher.into(),
                    platform,
                    agents: vec![AgentArg::Codex, AgentArg::Claude, AgentArg::Pi],
                },
            ))),
            &server,
        )
        .await;
        assert_eq!(result.expect("managed bundle"), ExitCode::SUCCESS);
        assert!(output.is_dir());
        assert!(std::fs::read_dir(&output).unwrap().next().is_some());
    }
}

#[test]
fn daemon_value_parsers_cover_valid_and_invalid_address_shapes() {
    assert_eq!(
        parse_bind_address("127.0.0.1").unwrap(),
        Ipv4Addr::LOCALHOST
    );
    assert_eq!(
        parse_bind_address("0.0.0.0").unwrap(),
        Ipv4Addr::UNSPECIFIED
    );
    assert!(parse_bind_address("localhost").is_err());
    assert!(parse_bind_address("192.0.2.1").is_err());
    assert_eq!(parse_nonzero_port("1").unwrap(), 1);
    assert!(parse_nonzero_port("0").is_err());
    assert!(parse_nonzero_port("65536").is_err());

    assert_eq!(
        parse_daemon_address("http://localhost:47632/").unwrap(),
        "http://localhost:47632"
    );
    assert!(parse_daemon_address("ftp://relay.example:21").is_err());
    assert!(parse_daemon_address("https://relay.example").is_err());
    assert!(parse_daemon_address("https://user@relay.example:443").is_err());
    assert!(parse_daemon_address("https://relay.example:443/path").is_err());
    assert!(parse_daemon_address("https://0.0.0.0:443").is_err());
    assert!(parse_daemon_address("http://relay.example:80").is_err());
}

#[test]
fn token_ensure_output_names_status_and_path_only() {
    let path = std::path::Path::new("/home/user/.config/nemo-relay/.client-token");
    for (status, expected) in [
        (
            daemon::common::client_token::EnsureStatus::Created,
            "created",
        ),
        (
            daemon::common::client_token::EnsureStatus::Existing,
            "existing",
        ),
    ] {
        let json: serde_json::Value =
            serde_json::from_str(&render_token_ensure(status, path, true).unwrap()).unwrap();
        assert_eq!(
            json,
            serde_json::json!({"status": expected, "path": path.display().to_string()})
        );
        let human = render_token_ensure(status, path, false).unwrap();
        assert!(human.contains(&path.display().to_string()), "{human}");
    }
}

#[test]
fn max_tokens_per_identity_prefers_the_flag_then_the_environment() {
    assert_eq!(resolve_max_tokens_per_identity(None, None).unwrap(), 4);
    assert_eq!(
        resolve_max_tokens_per_identity(None, Some(" 8 ")).unwrap(),
        8
    );
    assert_eq!(
        resolve_max_tokens_per_identity(Some(2), Some("8")).unwrap(),
        2
    );
    for invalid in ["0", "65", "four", ""] {
        let error = resolve_max_tokens_per_identity(None, Some(invalid))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("NEMO_RELAY_DAEMON_MAX_TOKENS_PER_IDENTITY"),
            "{error}"
        );
        assert!(error.contains("1 through 64"), "{error}");
    }
    assert!(parse_max_tokens_per_identity("64").is_ok());
    assert!(parse_max_tokens_per_identity("65").is_err());
}
