// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use clap::Parser;

#[derive(Parser)]
struct Inspector {
    #[command(flatten)]
    command: InspectCommand,
}

#[test]
fn inspector_defaults_and_response_bounds_are_enforced_by_parser() {
    let command = Inspector::try_parse_from(["inspect"]).unwrap().command;
    assert_eq!(command.gateway_url, crate::bootstrap::DEFAULT_URL);
    assert!(command.max_response_bytes.is_none());
    for limit in ["4096", "33554432"] {
        assert!(Inspector::try_parse_from(["inspect", "--max-response-bytes", limit]).is_ok());
    }
    for limit in ["4095", "33554433", "-1"] {
        assert!(Inspector::try_parse_from(["inspect", "--max-response-bytes", limit]).is_err());
    }
}

#[tokio::test]
async fn inspector_rejects_large_queries_before_attempting_transport() {
    let query = "x".repeat(1025);
    let command = Inspector::try_parse_from(["inspect", "--query", &query])
        .unwrap()
        .command;
    assert!(matches!(inspect(command).await, Err(CliError::Config(_))));
}

#[tokio::test]
async fn inspector_reports_unavailable_for_gateway_and_daemon_worker_transport() {
    let worker = uuid::Uuid::now_v7().to_string();
    for args in [
        vec!["inspect", "--gateway-url", "invalid-url"],
        vec![
            "inspect",
            "--daemon-url",
            "invalid-url",
            "--worker",
            &worker,
        ],
    ] {
        let command = Inspector::try_parse_from(args).unwrap().command;
        let error = inspect(command).await.unwrap_err();
        assert!(
            error
                .to_string()
                .contains("diagnostic inspection unavailable")
        );
    }
}

#[test]
fn worker_inspection_routes_require_daemon_and_uuid_without_header_injection() {
    for (path, daemon) in [
        ("/_nemo_relay/v1/workers/not-a-uuid/diagnostics", true),
        (
            "/_nemo_relay/v1/workers/00000000-0000-0000-0000-000000000000/diagnostics",
            false,
        ),
        ("/_nemo_relay/v1/diagnostics\r\nInjected: true", true),
    ] {
        let error =
            crate::gateway::client::get_diagnostics("invalid-url", path, 8192, daemon).unwrap_err();
        assert!(error.to_string().contains("invalid diagnostic route"));
    }
}
