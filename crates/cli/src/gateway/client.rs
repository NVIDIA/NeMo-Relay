// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Authenticated health and shutdown transport for loopback sidecars.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpStream, ToSocketAddrs};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use reqwest::Url;
use ring::rand::{SecureRandom, SystemRandom};
use serde_json::Value;

use crate::configuration::BootstrapChallengeKey;

use crate::bootstrap::{BOOTSTRAP_PROTOCOL_VERSION, HEALTHZ_TIMEOUT};

static CHALLENGE_KEY_CACHE: OnceLock<Mutex<HashMap<PathBuf, Arc<BootstrapChallengeKey>>>> =
    OnceLock::new();
static TLS_IDENTITY_CACHE: OnceLock<
    Mutex<HashMap<PathBuf, Arc<crate::gateway::tls::RelayTlsIdentity>>>,
> = OnceLock::new();

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RelayHealth {
    Compatible,
    Incompatible,
    Foreign,
    Unavailable,
}

#[derive(Debug)]
pub(crate) struct VerifiedHttpResponse {
    pub(crate) status: u16,
    pub(crate) body: Vec<u8>,
}

#[derive(Debug)]
pub(crate) struct VerifiedHttpError {
    message: String,
}

impl VerifiedHttpError {
    fn before_payload(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    fn after_payload(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub(crate) fn missing_fingerprint() -> Self {
        Self::after_payload("managed Relay gateway is missing its bootstrap fingerprint")
    }
}

impl std::fmt::Display for VerifiedHttpError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

/// Authenticates a Relay gateway and sends one HTTP request on that same TCP connection.
///
/// Keeping the challenge and payload on one established connection closes the port-replacement
/// gap between an authenticated health probe and hook delivery. A foreign listener can receive
/// the challenge, but the payload is not written until Relay proves possession of the per-user
/// bootstrap key on the connection that will receive it.
pub(crate) fn post_verified(
    url: &str,
    bootstrap_fingerprint: &str,
    path: &str,
    headers: &[(String, String)],
    body: &[u8],
    timeout: Duration,
    max_response_bytes: usize,
) -> Result<VerifiedHttpResponse, VerifiedHttpError> {
    let (host, port) = parse_loopback_url(url).map_err(VerifiedHttpError::after_payload)?;
    let addresses = (host.as_str(), port).to_socket_addrs().map_err(|error| {
        VerifiedHttpError::before_payload(format!(
            "failed to resolve verified gateway {url}: {error}"
        ))
    })?;
    let mut stream = connect_loopback(addresses, timeout).map_err(|error| {
        VerifiedHttpError::before_payload(format!(
            "failed to connect to verified gateway {url}: {error}"
        ))
    })?;
    stream.set_read_timeout(Some(timeout)).map_err(|error| {
        VerifiedHttpError::before_payload(format!(
            "failed to configure verified gateway read timeout: {error}"
        ))
    })?;
    stream.set_write_timeout(Some(timeout)).map_err(|error| {
        VerifiedHttpError::before_payload(format!(
            "failed to configure verified gateway write timeout: {error}"
        ))
    })?;

    let key = cached_bootstrap_challenge_key().map_err(|error| {
        VerifiedHttpError::before_payload(format!(
            "failed to load the Relay bootstrap challenge key: {error}"
        ))
    })?;
    let tls_identity = cached_tls_identity().map_err(|error| {
        VerifiedHttpError::before_payload(format!(
            "failed to load pinned Relay TLS identity: {error}"
        ))
    })?;
    let mut nonce = [0_u8; 32];
    SystemRandom::new().fill(&mut nonce).map_err(|_| {
        VerifiedHttpError::after_payload("failed to generate a Relay bootstrap challenge")
    })?;
    let nonce = nonce
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let authority = loopback_authority(&host, port);
    let tunnel = format!(
        "GET /bootstrap/tunnel HTTP/1.1\r\nHost: {authority}\r\nX-NeMo-Relay-Bootstrap-Fingerprint: {bootstrap_fingerprint}\r\nX-NeMo-Relay-Bootstrap-Nonce: {nonce}\r\nConnection: upgrade\r\nUpgrade: nemo-relay-tls\r\n\r\n"
    );
    stream.write_all(tunnel.as_bytes()).map_err(|error| {
        VerifiedHttpError::before_payload(format!(
            "failed to request the Relay TLS tunnel: {error}"
        ))
    })?;
    let (tunnel_headers, _) = read_http_message(&mut stream, 0).map_err(|error| {
        VerifiedHttpError::before_payload(format!(
            "failed to read the Relay TLS tunnel response: {error}"
        ))
    })?;
    let proof_valid = http_header(&tunnel_headers, "x-nemo-relay-bootstrap-proof")
        .is_some_and(|proof| key.verify(bootstrap_fingerprint, &nonce, proof));
    if http_status(&tunnel_headers) != Some(101)
        || !proof_valid
        || http_header(&tunnel_headers, "upgrade") != Some("nemo-relay-tls")
    {
        return Err(VerifiedHttpError::before_payload(
            "gateway did not establish an authenticated Relay TLS tunnel",
        ));
    }
    let client_config = tls_identity
        .client_config()
        .map_err(VerifiedHttpError::before_payload)?;
    let server_name = rustls::pki_types::ServerName::try_from("localhost").map_err(|error| {
        VerifiedHttpError::before_payload(format!("invalid Relay TLS server name: {error}"))
    })?;
    let connection =
        rustls::ClientConnection::new(client_config, server_name).map_err(|error| {
            VerifiedHttpError::before_payload(format!("failed to create Relay TLS client: {error}"))
        })?;
    let mut stream = rustls::StreamOwned::new(connection, stream);

    let challenge = format!(
        "GET /healthz HTTP/1.1\r\nHost: {authority}\r\nX-NeMo-Relay-Bootstrap-Fingerprint: {bootstrap_fingerprint}\r\nX-NeMo-Relay-Bootstrap-Nonce: {nonce}\r\nConnection: keep-alive\r\n\r\n"
    );
    stream.write_all(challenge.as_bytes()).map_err(|error| {
        VerifiedHttpError::before_payload(format!(
            "Relay TLS handshake or health request failed: {error}"
        ))
    })?;
    let (health_headers, health_body) =
        read_http_message(&mut stream, 16 * 1024).map_err(|error| {
            VerifiedHttpError::before_payload(format!(
                "failed to read health response through Relay TLS: {error}"
            ))
        })?;
    let (health, _) = classify_health_response(
        &health_headers,
        &health_body,
        Some((bootstrap_fingerprint, nonce.as_str(), key.as_ref())),
    );
    match health {
        RelayHealth::Compatible => {}
        RelayHealth::Incompatible => {
            return Err(VerifiedHttpError::after_payload(format!(
                "an incompatible NeMo Relay gateway is listening at {url}"
            )));
        }
        RelayHealth::Foreign | RelayHealth::Unavailable => {
            return Err(VerifiedHttpError::after_payload(format!(
                "a foreign process is listening at the shared Relay gateway URL {url}"
            )));
        }
    }
    if http_header(&health_headers, "connection")
        .is_some_and(|value| value.eq_ignore_ascii_case("close"))
    {
        return Err(VerifiedHttpError::before_payload(
            "verified Relay gateway closed the authenticated connection before request delivery",
        ));
    }

    let mut request = format!("POST {path} HTTP/1.1\r\nHost: {authority}\r\n");
    for (name, value) in headers {
        request.push_str(name);
        request.push_str(": ");
        request.push_str(value);
        request.push_str("\r\n");
    }
    request.push_str(&format!(
        "Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    ));
    stream.write_all(request.as_bytes()).map_err(|error| {
        VerifiedHttpError::before_payload(format!(
            "failed to send verified gateway request headers: {error}"
        ))
    })?;
    stream.write_all(body).map_err(|error| {
        VerifiedHttpError::after_payload(format!(
            "verified gateway payload delivery became indeterminate: {error}"
        ))
    })?;
    let (response_headers, response_body) = read_http_message(&mut stream, max_response_bytes)
        .map_err(|error| {
            VerifiedHttpError::after_payload(format!(
                "failed to read verified gateway response: {error}"
            ))
        })?;
    let status = http_status(&response_headers).ok_or_else(|| {
        VerifiedHttpError::after_payload("verified gateway response had an invalid HTTP status")
    })?;
    Ok(VerifiedHttpResponse {
        status,
        body: response_body,
    })
}

pub(crate) fn healthz(url: &str) -> bool {
    probe(url, None) == RelayHealth::Compatible
}

pub(crate) fn healthz_compatible(url: &str, bootstrap_fingerprint: &str) -> bool {
    probe(url, Some(bootstrap_fingerprint)) == RelayHealth::Compatible
}

pub(crate) fn authenticated_instance_id(url: &str, bootstrap_fingerprint: &str) -> Option<String> {
    compatible_instance_id(url, Some(bootstrap_fingerprint))
}

pub(crate) fn probe(url: &str, bootstrap_fingerprint: Option<&str>) -> RelayHealth {
    probe_with_instance(url, bootstrap_fingerprint).0
}

pub(crate) fn compatible_instance_id(
    url: &str,
    bootstrap_fingerprint: Option<&str>,
) -> Option<String> {
    let (health, instance_id) = probe_with_instance(url, bootstrap_fingerprint);
    (health == RelayHealth::Compatible)
        .then_some(instance_id)
        .flatten()
}

pub(crate) fn probe_with_instance(
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
    for addr in addrs.filter(|addr| addr.ip().is_loopback()) {
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
        let key = cached_bootstrap_challenge_key().map_err(|_| ())?;
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
    let Ok((headers, body)) = read_http_message(&mut stream, 16 * 1024) else {
        return (RelayHealth::Foreign, None);
    };
    classify_health_response(
        &headers,
        &body,
        challenge
            .as_ref()
            .map(|(fingerprint, nonce, key)| (*fingerprint, nonce.as_str(), key.as_ref())),
    )
}

pub(crate) fn request_shutdown(
    url: &str,
    bootstrap_fingerprint: &str,
    token: &str,
) -> Result<(), String> {
    let (host, port) = parse_loopback_url(url)?;
    let addresses = (host.as_str(), port)
        .to_socket_addrs()
        .map_err(|error| format!("failed to resolve managed sidecar {url}: {error}"))?;
    let mut stream = connect_loopback(addresses, HEALTHZ_TIMEOUT)
        .map_err(|error| format!("failed to connect to managed sidecar {url}: {error}"))?;
    stream
        .set_read_timeout(Some(HEALTHZ_TIMEOUT))
        .map_err(|error| format!("failed to configure sidecar shutdown read timeout: {error}"))?;
    stream
        .set_write_timeout(Some(HEALTHZ_TIMEOUT))
        .map_err(|error| format!("failed to configure sidecar shutdown write timeout: {error}"))?;
    let key = cached_bootstrap_challenge_key()
        .map_err(|error| format!("failed to load the Relay bootstrap challenge key: {error}"))?;
    let mut nonce = [0_u8; 32];
    SystemRandom::new()
        .fill(&mut nonce)
        .map_err(|_| "failed to generate a Relay bootstrap shutdown challenge".to_string())?;
    let nonce = nonce
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let authority = loopback_authority(&host, port);
    let challenge = format!(
        "GET /healthz HTTP/1.1\r\nHost: {authority}\r\nX-NeMo-Relay-Bootstrap-Fingerprint: {bootstrap_fingerprint}\r\nX-NeMo-Relay-Bootstrap-Nonce: {nonce}\r\nConnection: keep-alive\r\n\r\n"
    );
    stream
        .write_all(challenge.as_bytes())
        .map_err(|error| format!("failed to authenticate managed sidecar shutdown: {error}"))?;
    let (health_headers, health_body) = read_http_message(&mut stream, 16 * 1024)
        .map_err(|error| format!("failed to read managed sidecar shutdown proof: {error}"))?;
    if classify_health_response(
        &health_headers,
        &health_body,
        Some((bootstrap_fingerprint, nonce.as_str(), key.as_ref())),
    )
    .0 != RelayHealth::Compatible
    {
        return Err("managed sidecar did not authenticate the shutdown connection".into());
    }
    if http_header(&health_headers, "connection")
        .is_some_and(|value| value.eq_ignore_ascii_case("close"))
    {
        return Err("managed sidecar closed the authenticated shutdown connection".into());
    }
    let request = format!(
        "POST /bootstrap/shutdown HTTP/1.1\r\nHost: {}\r\nX-NeMo-Relay-Bootstrap-Token: {token}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        authority
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

fn cached_bootstrap_challenge_key() -> Result<Arc<BootstrapChallengeKey>, String> {
    let state = crate::bootstrap::state::state_dir()?;
    let cache = CHALLENGE_KEY_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| "Relay bootstrap challenge key cache is poisoned".to_string())?;
    if let Some(key) = cache.get(&state) {
        return Ok(Arc::clone(key));
    }
    let key = Arc::new(BootstrapChallengeKey::load().map_err(|error| error.to_string())?);
    cache.insert(state, Arc::clone(&key));
    Ok(key)
}

fn cached_tls_identity() -> Result<Arc<crate::gateway::tls::RelayTlsIdentity>, String> {
    let state = crate::bootstrap::state::state_dir()?;
    let cache = TLS_IDENTITY_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| "Relay TLS identity cache is poisoned".to_string())?;
    if let Some(identity) = cache.get(&state) {
        return Ok(Arc::clone(identity));
    }
    let identity = Arc::new(crate::gateway::tls::RelayTlsIdentity::load()?);
    cache.insert(state, Arc::clone(&identity));
    Ok(identity)
}

fn http_header<'a>(headers: &'a [u8], name: &str) -> Option<&'a str> {
    headers.split(|byte| *byte == b'\n').find_map(|line| {
        let line = std::str::from_utf8(line).ok()?.trim_end_matches('\r');
        let (candidate, value) = line.split_once(':')?;
        candidate.eq_ignore_ascii_case(name).then(|| value.trim())
    })
}

fn http_status(headers: &[u8]) -> Option<u16> {
    let line = headers.split(|byte| *byte == b'\n').next()?;
    let line = std::str::from_utf8(line).ok()?.trim_end_matches('\r');
    let mut fields = line.split_ascii_whitespace();
    matches!(fields.next(), Some("HTTP/1.1" | "HTTP/1.0"))
        .then(|| fields.next()?.parse().ok())
        .flatten()
}

fn classify_health_response(
    headers: &[u8],
    body: &[u8],
    challenge: Option<(&str, &str, &BootstrapChallengeKey)>,
) -> (RelayHealth, Option<String>) {
    let Ok(body) = serde_json::from_slice::<Value>(body) else {
        return (RelayHealth::Foreign, None);
    };
    if body.get("service").and_then(Value::as_str) != Some("nemo-relay")
        || body.get("bootstrap_protocol").and_then(Value::as_u64)
            != Some(BOOTSTRAP_PROTOCOL_VERSION)
    {
        return (RelayHealth::Foreign, None);
    }
    if http_status(headers) == Some(409) {
        return (RelayHealth::Incompatible, None);
    }
    if http_status(headers) != Some(200) || body.get("status").and_then(Value::as_str) != Some("ok")
    {
        return (RelayHealth::Foreign, None);
    }
    if let Some((fingerprint, nonce, key)) = challenge {
        let Some(proof) = http_header(headers, "x-nemo-relay-bootstrap-proof") else {
            return (RelayHealth::Foreign, None);
        };
        if !key.verify(fingerprint, nonce, proof) {
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
    (RelayHealth::Compatible, Some(instance_id.to_owned()))
}

fn connect_loopback(
    addresses: impl IntoIterator<Item = SocketAddr>,
    timeout: Duration,
) -> std::io::Result<TcpStream> {
    let mut last_error = None;
    for address in addresses
        .into_iter()
        .filter(|address| address.ip().is_loopback())
    {
        match TcpStream::connect_timeout(&address, timeout) {
            Ok(stream) => return Ok(stream),
            Err(error) => last_error = Some(error),
        }
    }
    Err(last_error.unwrap_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::AddrNotAvailable,
            "gateway URL resolved to no loopback socket addresses",
        )
    }))
}

fn read_http_message(
    stream: &mut impl Read,
    max_body_bytes: usize,
) -> std::io::Result<(Vec<u8>, Vec<u8>)> {
    const MAX_HEADER_BYTES: usize = 16 * 1024;

    let mut response = Vec::new();
    let header_end = loop {
        if let Some(index) = response.windows(4).position(|window| window == b"\r\n\r\n") {
            break index + 4;
        }
        if response.len() >= MAX_HEADER_BYTES {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "HTTP response headers exceed the Relay limit",
            ));
        }
        let mut chunk = [0_u8; 1024];
        let read = stream.read(&mut chunk)?;
        if read == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "HTTP response ended before its headers",
            ));
        }
        response.extend_from_slice(&chunk[..read]);
    };
    let headers = response[..header_end - 4].to_vec();
    if http_header(&headers, "transfer-encoding").is_some() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "chunked HTTP responses are not supported by the verified Relay transport",
        ));
    }
    if http_status(&headers) == Some(101) {
        return Ok((headers, Vec::new()));
    }
    let content_length = http_header(&headers, "content-length")
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "HTTP response omitted Content-Length",
            )
        })?
        .parse::<usize>()
        .map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "HTTP response had an invalid Content-Length",
            )
        })?;
    if content_length > max_body_bytes {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("HTTP response exceeds the {max_body_bytes}-byte Relay limit"),
        ));
    }
    let mut body = response[header_end..].to_vec();
    if body.len() > content_length {
        body.truncate(content_length);
    }
    while body.len() < content_length {
        let mut chunk = [0_u8; 4096];
        let needed = (content_length - body.len()).min(chunk.len());
        let read = stream.read(&mut chunk[..needed])?;
        if read == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "HTTP response ended before its declared body",
            ));
        }
        body.extend_from_slice(&chunk[..read]);
    }
    Ok((headers, body))
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
#[path = "../../tests/coverage/shared/gateway_client_tests.rs"]
mod tests;
