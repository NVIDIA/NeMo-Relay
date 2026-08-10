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

pub(super) async fn probe_otlp_http_named(name: &'static str, url: &str) -> Check {
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
            status: if response.status().is_success()
                || response.status().is_redirection()
                || response.status() == reqwest::StatusCode::METHOD_NOT_ALLOWED
            {
                Status::Pass
            } else {
                Status::Warn
            },
            details: format!(
                "{} (live HTTP reachability probe returned HTTP {})",
                url,
                response.status().as_u16()
            ),
        },
        Err(error) => Check {
            name,
            status: Status::Fail,
            details: format!("{url}: live HTTP reachability probe failed: {error}"),
        },
    }
}

pub(super) async fn probe_tcp_named(name: &'static str, endpoint: &str) -> Check {
    let (parsed, host) = match validate_grpc_endpoint(endpoint) {
        Ok((parsed, host)) => (parsed, host),
        Err(details) => {
            return Check {
                name,
                status: Status::Fail,
                details,
            };
        }
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
            details: format!(
                "{endpoint} (live gRPC reachability probe connected to the TCP port; OTLP handshake not verified)"
            ),
        },
        Ok(Err(error)) => Check {
            name,
            status: Status::Fail,
            details: format!("{endpoint}: live gRPC reachability probe failed: {error}"),
        },
        Err(_) => Check {
            name,
            status: Status::Fail,
            details: format!("{endpoint}: live gRPC reachability probe timed out"),
        },
    }
}

pub(super) fn validate_grpc_endpoint(endpoint: &str) -> Result<(reqwest::Url, String), String> {
    let parsed = reqwest::Url::parse(endpoint)
        .map_err(|error| format!("{endpoint}: invalid gRPC endpoint: {error}"))?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(format!(
            "{endpoint}: gRPC endpoint must use http:// or https://"
        ));
    }
    let host = parsed
        .host_str()
        .map(str::to_owned)
        .ok_or_else(|| format!("{endpoint}: gRPC endpoint has no host"))?;
    Ok((parsed, host))
}

pub(super) fn validate_otlp_http_endpoint(endpoint: &str) -> Result<(), String> {
    let parsed = reqwest::Url::parse(endpoint)
        .map_err(|error| format!("{endpoint}: invalid OTLP HTTP endpoint: {error}"))?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(format!(
            "{endpoint}: OTLP HTTP endpoint must use http:// or https://"
        ));
    }
    if parsed.host_str().is_none() {
        return Err(format!("{endpoint}: OTLP HTTP endpoint has no host"));
    }
    Ok(())
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
#[path = "../../tests/coverage/shared/probes_tests.rs"]
mod tcp_tests;
