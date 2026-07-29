// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded filesystem and network diagnostic probes.

use super::{Check, NETWORK_TIMEOUT, Status};
use std::fs::OpenOptions;
use std::path::Path;

pub(super) fn check_directory(name: &'static str, path: &Path) -> Check {
    match check_dir_writable(path) {
        Ok(()) => Check {
            name,
            status: Status::Pass,
            details: format!("{} (appears writable)", path.display()),
        },
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Check {
            name,
            status: Status::Warn,
            details: format!("{}: not present; runtime will create it", path.display()),
        },
        Err(error) => Check {
            name,
            status: Status::Fail,
            details: format!("{}: {error}", path.display()),
        },
    }
}

pub(super) fn check_dir_writable(directory: &Path) -> Result<(), std::io::Error> {
    let metadata = std::fs::metadata(directory)?;
    if !metadata.is_dir() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "path is not a directory",
        ));
    }
    let probe = directory.join(format!(".nemo-relay-write-probe-{}", uuid::Uuid::now_v7()));
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&probe)?;
    drop(file);
    std::fs::remove_file(probe)
}

pub(super) async fn probe_http_named(name: &'static str, url: &str) -> Check {
    let client = match reqwest::Client::builder().timeout(NETWORK_TIMEOUT).build() {
        Ok(client) => client,
        Err(error) => {
            return Check {
                name,
                status: Status::Fail,
                details: format!("could not build HTTP client: {error}"),
            };
        }
    };
    match client.get(url).send().await {
        Ok(response) => Check {
            name,
            status: if response.status().is_success() || response.status().is_redirection() {
                Status::Pass
            } else {
                Status::Warn
            },
            details: format!("{} (HTTP {})", url, response.status().as_u16()),
        },
        Err(error) => Check {
            name,
            status: Status::Fail,
            details: format!("{url}: {error}"),
        },
    }
}

pub(super) async fn probe_tcp_named(name: &'static str, endpoint: &str) -> Check {
    let parsed = match reqwest::Url::parse(endpoint) {
        Ok(parsed) => parsed,
        Err(error) => {
            return Check {
                name,
                status: Status::Fail,
                details: format!("{endpoint}: invalid gRPC endpoint: {error}"),
            };
        }
    };
    let Some(host) = parsed.host_str() else {
        return Check {
            name,
            status: Status::Fail,
            details: format!("{endpoint}: gRPC endpoint has no host"),
        };
    };
    let port = grpc_endpoint_port(&parsed);
    match tokio::time::timeout(
        NETWORK_TIMEOUT,
        tokio::net::TcpStream::connect((host, port)),
    )
    .await
    {
        Ok(Ok(_)) => Check {
            name,
            status: Status::Pass,
            details: format!("{endpoint} (gRPC TCP connection succeeded)"),
        },
        Ok(Err(error)) => Check {
            name,
            status: Status::Fail,
            details: format!("{endpoint}: gRPC TCP connection failed: {error}"),
        },
        Err(_) => Check {
            name,
            status: Status::Fail,
            details: format!("{endpoint}: gRPC TCP connection timed out"),
        },
    }
}

fn grpc_endpoint_port(endpoint: &reqwest::Url) -> u16 {
    endpoint.port().unwrap_or_else(|| {
        if endpoint.scheme() == "https" {
            443
        } else {
            4317
        }
    })
}

#[cfg(test)]
mod tcp_tests {
    use super::*;

    #[tokio::test]
    async fn grpc_probe_uses_tcp_connectivity() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let check = probe_tcp_named("OpenTelemetry endpoint", &endpoint).await;
        assert_eq!(check.status, Status::Pass);
        assert!(check.details.contains("gRPC TCP connection succeeded"));
    }

    #[tokio::test]
    async fn grpc_probe_reports_invalid_hostless_and_refused_endpoints() {
        let invalid = probe_tcp_named("OpenTelemetry endpoint", "not a url").await;
        assert_eq!(invalid.status, Status::Fail);
        assert!(invalid.details.contains("invalid gRPC endpoint"));

        let hostless = probe_tcp_named("OpenTelemetry endpoint", "file:///tmp/collector").await;
        assert_eq!(hostless.status, Status::Fail);
        assert!(hostless.details.contains("has no host"));

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        drop(listener);
        let refused = probe_tcp_named("OpenTelemetry endpoint", &endpoint).await;
        assert_eq!(refused.status, Status::Fail);
        assert!(
            refused.details.contains("connection failed")
                || refused.details.contains("connection timed out"),
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
}
