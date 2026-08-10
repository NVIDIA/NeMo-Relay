// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[tokio::test]
async fn grpc_probe_uses_tcp_connectivity() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let check = probe_tcp_named("OpenTelemetry endpoint", &endpoint).await;
    assert_eq!(check.status, Status::Pass);
    assert!(
        check
            .details
            .contains("live gRPC reachability probe connected to the TCP port")
    );
    assert!(check.details.contains("OTLP handshake not verified"));
}

#[tokio::test]
async fn grpc_probe_reports_invalid_hostless_and_refused_endpoints() {
    let invalid = probe_tcp_named("OpenTelemetry endpoint", "not a url").await;
    assert_eq!(invalid.status, Status::Fail);
    assert!(invalid.details.contains("invalid gRPC endpoint"));

    let hostless = probe_tcp_named("OpenTelemetry endpoint", "file:///tmp/collector").await;
    assert_eq!(hostless.status, Status::Fail);
    assert!(
        hostless
            .details
            .contains("gRPC endpoint must use http:// or https://")
    );

    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);
    let refused = probe_tcp_named("OpenTelemetry endpoint", &endpoint).await;
    assert_eq!(refused.status, Status::Fail);
    assert!(
        refused
            .details
            .contains("live gRPC reachability probe failed")
            || refused
                .details
                .contains("live gRPC reachability probe timed out"),
        "{}",
        refused.details
    );
}

#[test]
fn grpc_probe_uses_tls_and_otlp_default_ports() {
    assert_eq!(
        grpc_endpoint_port(&reqwest::Url::parse("https://collector.example.com").unwrap()),
        443
    );
    assert_eq!(
        grpc_endpoint_port(&reqwest::Url::parse("http://collector.example.com").unwrap()),
        4317
    );
    assert_eq!(
        grpc_endpoint_port(&reqwest::Url::parse("https://collector.example.com:8443").unwrap()),
        8443
    );
}
