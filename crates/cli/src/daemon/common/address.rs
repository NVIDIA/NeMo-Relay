// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use reqwest::Url;

use crate::error::CliError;

pub(crate) const DEFAULT_DAEMON_PORT: u16 = 47_632;
pub(crate) const DEFAULT_DAEMON_BIND: Ipv4Addr = Ipv4Addr::LOCALHOST;
pub(crate) const DEFAULT_WORKER_BIND: Ipv4Addr = Ipv4Addr::LOCALHOST;

pub(crate) fn validate_bind_ip(ip: Ipv4Addr, component: &str) -> Result<(), CliError> {
    if matches!(ip, Ipv4Addr::LOCALHOST | Ipv4Addr::UNSPECIFIED) {
        return Ok(());
    }
    Err(CliError::Config(format!(
        "{component} bind address must be 127.0.0.1 or 0.0.0.0, got {ip}"
    )))
}

pub(crate) fn daemon_url(raw: &str) -> Result<Url, CliError> {
    let explicit_port = raw
        .parse::<http::Uri>()
        .ok()
        .and_then(|uri| uri.authority().and_then(http::uri::Authority::port_u16));
    let url = Url::parse(raw)
        .map_err(|error| CliError::Config(format!("invalid daemon address {raw:?}: {error}")))?;
    if !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || !matches!(url.path(), "" | "/")
    {
        return Err(CliError::Config(
            "daemon address must be an origin URL without credentials, path, query, or fragment"
                .into(),
        ));
    }
    if explicit_port.is_none() {
        return Err(CliError::Config(
            "daemon address must include an explicit port".into(),
        ));
    }
    let host = url
        .host_str()
        .ok_or_else(|| CliError::Config("daemon address is missing a host".into()))?;
    if host == "0.0.0.0" {
        return Err(CliError::Config(
            "0.0.0.0 is a bind address and cannot be used as a daemon target".into(),
        ));
    }
    let loopback = host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<IpAddr>()
            .is_ok_and(|address| address.is_loopback());
    match url.scheme() {
        "https" => {}
        "http" if loopback => {}
        "http" => {
            return Err(CliError::Config(
                "non-loopback daemon addresses must use https".into(),
            ));
        }
        scheme => {
            return Err(CliError::Config(format!(
                "daemon address scheme must be http or https, got {scheme}"
            )));
        }
    }
    Ok(url)
}

pub(crate) fn worker_socket(bind: Ipv4Addr, port: Option<u16>) -> Result<SocketAddr, CliError> {
    validate_bind_ip(bind, "worker")?;
    if port == Some(0) {
        return Err(CliError::Config(
            "an explicitly supplied worker port must be between 1 and 65535; omit --port for automatic allocation"
                .into(),
        ));
    }
    Ok(SocketAddr::new(IpAddr::V4(bind), port.unwrap_or(0)))
}

pub(crate) fn worker_advertised_address(
    local: SocketAddr,
    configured: Option<&str>,
) -> Result<String, CliError> {
    if local.ip().is_unspecified() {
        let host = configured.ok_or_else(|| {
            CliError::Config(
                "--advertise-address is required when the worker binds to 0.0.0.0".into(),
            )
        })?;
        let host = host.trim();
        if host.is_empty() || host == "0.0.0.0" {
            return Err(CliError::Config(
                "worker advertised address must be a concrete host or IP, not 0.0.0.0".into(),
            ));
        }
        return Ok(format_host_port(host, local.port()));
    }
    if configured.is_some() {
        return Err(CliError::Config(
            "--advertise-address is only valid with --bind 0.0.0.0".into(),
        ));
    }
    Ok(local.to_string())
}

fn format_host_port(host: &str, port: u16) -> String {
    if host.starts_with('[') || !host.contains(':') {
        format!("{host}:{port}")
    } else {
        format!("[{host}]:{port}")
    }
}

#[cfg(test)]
#[path = "../../../tests/coverage/daemon/address_tests.rs"]
mod tests;
