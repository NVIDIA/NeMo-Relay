// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Authenticated health and shutdown transport for loopback sidecars.

use std::io::{Read, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpStream, ToSocketAddrs};
use std::thread;
use std::time::Duration;

use reqwest::Url;
use ring::rand::{SecureRandom, SystemRandom};
use serde_json::Value;

use crate::config::BootstrapChallengeKey;

use super::{BOOTSTRAP_PROTOCOL_VERSION, HEALTHZ_TIMEOUT};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RelayHealth {
    Compatible,
    Incompatible,
    Foreign,
    Unavailable,
}

pub(crate) fn healthz(url: &str) -> bool {
    probe(url, None) == RelayHealth::Compatible
}

#[cfg(test)]
pub(crate) fn healthz_compatible(url: &str, bootstrap_fingerprint: &str) -> bool {
    probe(url, Some(bootstrap_fingerprint)) == RelayHealth::Compatible
}

pub(super) fn probe_after_lock(
    url: &str,
    bootstrap_fingerprint: Option<&str>,
) -> (RelayHealth, Option<String>) {
    let mut result = probe_with_instance(url, bootstrap_fingerprint);
    for _ in 1..3 {
        if result.0 != RelayHealth::Foreign {
            break;
        }
        thread::sleep(Duration::from_millis(50));
        result = probe_with_instance(url, bootstrap_fingerprint);
    }
    result
}

pub(super) fn probe(url: &str, bootstrap_fingerprint: Option<&str>) -> RelayHealth {
    probe_with_instance(url, bootstrap_fingerprint).0
}

pub(super) fn compatible_instance_id(
    url: &str,
    bootstrap_fingerprint: Option<&str>,
) -> Option<String> {
    let (health, instance_id) = probe_with_instance(url, bootstrap_fingerprint);
    (health == RelayHealth::Compatible)
        .then_some(instance_id)
        .flatten()
}

pub(super) fn probe_with_instance(
    url: &str,
    bootstrap_fingerprint: Option<&str>,
) -> (RelayHealth, Option<String>) {
    let Ok((host, port)) = parse_loopback_url(url) else {
        return (RelayHealth::Unavailable, None);
    };
    let Ok(addrs) = (host.as_str(), port).to_socket_addrs() else {
        return (RelayHealth::Unavailable, None);
    };
    let mut stream = None;
    for addr in addrs {
        match TcpStream::connect_timeout(&addr, HEALTHZ_TIMEOUT) {
            Ok(candidate) => {
                stream = Some(candidate);
                break;
            }
            Err(_) => continue,
        }
    }
    let Some(mut stream) = stream else {
        return (RelayHealth::Unavailable, None);
    };
    if stream.set_read_timeout(Some(HEALTHZ_TIMEOUT)).is_err()
        || stream.set_write_timeout(Some(HEALTHZ_TIMEOUT)).is_err()
    {
        return (RelayHealth::Foreign, None);
    }
    let challenge = bootstrap_fingerprint.map(|fingerprint| {
        let key = BootstrapChallengeKey::load().map_err(|_| ())?;
        let mut nonce = [0_u8; 32];
        SystemRandom::new().fill(&mut nonce).map_err(|_| ())?;
        let nonce = nonce
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        Ok::<_, ()>((fingerprint, nonce, key))
    });
    let challenge = match challenge.transpose() {
        Ok(challenge) => challenge,
        Err(()) => return (RelayHealth::Foreign, None),
    };
    let fingerprint_headers = challenge
        .as_ref()
        .map(|(fingerprint, nonce, _)| {
            format!(
                "X-NeMo-Relay-Bootstrap-Fingerprint: {fingerprint}\r\nX-NeMo-Relay-Bootstrap-Nonce: {nonce}\r\n"
            )
        })
        .unwrap_or_default();
    let request = format!(
        "GET /healthz HTTP/1.1\r\nHost: {}\r\n{fingerprint_headers}Connection: close\r\n\r\n",
        loopback_authority(&host, port)
    );
    if stream.write_all(request.as_bytes()).is_err() {
        return (RelayHealth::Foreign, None);
    }
    let mut response = Vec::new();
    if stream.take(16 * 1024).read_to_end(&mut response).is_err() {
        return (RelayHealth::Foreign, None);
    }
    let Some((headers, body)) = split_http_response(&response) else {
        return (RelayHealth::Foreign, None);
    };
    let Ok(body) = serde_json::from_slice::<Value>(body) else {
        return (RelayHealth::Foreign, None);
    };
    if body.get("service").and_then(Value::as_str) != Some("nemo-relay")
        || body.get("bootstrap_protocol").and_then(Value::as_u64)
            != Some(BOOTSTRAP_PROTOCOL_VERSION)
    {
        return (RelayHealth::Foreign, None);
    }
    if body.get("version").and_then(Value::as_str) != Some(env!("CARGO_PKG_VERSION"))
        || headers.starts_with(b"HTTP/1.1 409")
        || headers.starts_with(b"HTTP/1.0 409")
    {
        return (RelayHealth::Incompatible, None);
    }
    if (headers.starts_with(b"HTTP/1.1 200") || headers.starts_with(b"HTTP/1.0 200"))
        && body.get("status").and_then(Value::as_str) == Some("ok")
    {
        if let Some((fingerprint, nonce, key)) = challenge {
            let Some(proof) = http_header(headers, "x-nemo-relay-bootstrap-proof") else {
                return (RelayHealth::Foreign, None);
            };
            if !key.verify(fingerprint, &nonce, proof) {
                return (RelayHealth::Foreign, None);
            }
        }
        let Some(instance_id) = body
            .get("instance_id")
            .and_then(Value::as_str)
            .filter(|instance_id| !instance_id.is_empty() && instance_id.len() <= 128)
        else {
            return (RelayHealth::Foreign, None);
        };
        return (RelayHealth::Compatible, Some(instance_id.to_owned()));
    }
    (RelayHealth::Foreign, None)
}

pub(super) fn request_shutdown(url: &str, token: &str) -> Result<(), String> {
    let (host, port) = parse_loopback_url(url)?;
    let address = (host.as_str(), port)
        .to_socket_addrs()
        .map_err(|error| format!("failed to resolve managed sidecar {url}: {error}"))?
        .next()
        .ok_or_else(|| format!("managed sidecar {url} has no socket address"))?;
    let mut stream = TcpStream::connect_timeout(&address, HEALTHZ_TIMEOUT)
        .map_err(|error| format!("failed to connect to managed sidecar {url}: {error}"))?;
    stream
        .set_read_timeout(Some(HEALTHZ_TIMEOUT))
        .map_err(|error| format!("failed to configure sidecar shutdown read timeout: {error}"))?;
    stream
        .set_write_timeout(Some(HEALTHZ_TIMEOUT))
        .map_err(|error| format!("failed to configure sidecar shutdown write timeout: {error}"))?;
    let request = format!(
        "POST /bootstrap/shutdown HTTP/1.1\r\nHost: {}\r\nX-NeMo-Relay-Bootstrap-Token: {token}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        loopback_authority(&host, port)
    );
    stream
        .write_all(request.as_bytes())
        .map_err(|error| format!("failed to request managed sidecar shutdown: {error}"))?;
    let mut response = Vec::new();
    stream
        .take(16 * 1024)
        .read_to_end(&mut response)
        .map_err(|error| format!("failed to read managed sidecar shutdown response: {error}"))?;
    let Some((headers, _)) = split_http_response(&response) else {
        return Err("managed sidecar returned a malformed shutdown response".into());
    };
    if headers.starts_with(b"HTTP/1.1 204") || headers.starts_with(b"HTTP/1.0 204") {
        Ok(())
    } else {
        Err(format!(
            "managed sidecar rejected shutdown: {}",
            String::from_utf8_lossy(headers)
                .lines()
                .next()
                .unwrap_or("unknown response")
        ))
    }
}

fn http_header<'a>(headers: &'a [u8], name: &str) -> Option<&'a str> {
    headers.split(|byte| *byte == b'\n').find_map(|line| {
        let line = std::str::from_utf8(line).ok()?.trim_end_matches('\r');
        let (candidate, value) = line.split_once(':')?;
        candidate.eq_ignore_ascii_case(name).then(|| value.trim())
    })
}

fn split_http_response(response: &[u8]) -> Option<(&[u8], &[u8])> {
    response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|index| (&response[..index], &response[index + 4..]))
}

pub(crate) fn parse_loopback_url(url: &str) -> Result<(String, u16), String> {
    let parsed = Url::parse(url)
        .map_err(|error| format!("invalid shared gateway loopback URL {url}: {error}"))?;
    if parsed.scheme() != "http" {
        return Err(format!(
            "shared gateway recovery only supports http loopback URLs: {url}"
        ));
    }
    let host = parsed
        .host_str()
        .ok_or_else(|| format!("missing host in gateway URL: {url}"))?
        .trim_start_matches('[')
        .trim_end_matches(']');
    let loopback = host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|address| address.is_loopback());
    if !loopback {
        return Err(format!(
            "shared gateway recovery only supports loopback gateway URLs: {url}"
        ));
    }
    let port = parsed
        .port()
        .ok_or_else(|| format!("missing port in gateway URL: {url}"))?;
    Ok((host.to_string(), port))
}

pub(crate) fn loopback_bind(url: &str) -> Result<SocketAddr, String> {
    let (host, port) = parse_loopback_url(url)?;
    let address = if host.eq_ignore_ascii_case("localhost") {
        std::net::IpAddr::V4(Ipv4Addr::LOCALHOST)
    } else {
        host.parse::<std::net::IpAddr>()
            .map_err(|error| format!("invalid loopback address in gateway URL {url}: {error}"))?
    };
    Ok(SocketAddr::new(address, port))
}

pub(crate) fn loopback_authority(host: &str, port: u16) -> String {
    if host.contains(':') {
        format!("[{host}]:{port}")
    } else {
        format!("{host}:{port}")
    }
}

#[cfg(test)]
#[path = "../../tests/coverage/sidecar_health_tests.rs"]
mod tests;
