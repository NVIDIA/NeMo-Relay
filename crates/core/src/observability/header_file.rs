// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request-time resolution for file-backed exporter headers.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::net::{IpAddr, Ipv6Addr};
use std::sync::Arc;

use async_trait::async_trait;
use opentelemetry_http::{Bytes, HttpClient, HttpError, Request, Response};

/// File paths keyed by HTTP header name.
pub(crate) type HeaderFiles = HashMap<String, String>;

fn is_loopback_host(host: &str) -> bool {
    match parse_host(host) {
        Some(HostRef::Ip(ip)) => ip.is_loopback(),
        Some(HostRef::Name(name)) => name.eq_ignore_ascii_case("localhost"),
        None => false,
    }
}

enum HostRef<'a> {
    Ip(IpAddr),
    Name(&'a str),
}

/// Parse a host strictly enough that every HTTP stack resolves it to the same address.
fn parse_host(host: &str) -> Option<HostRef<'_>> {
    if let Some(bracketed) = host.strip_prefix('[') {
        let ip = bracketed.strip_suffix(']')?.parse::<Ipv6Addr>().ok()?;
        return Some(HostRef::Ip(ip.into()));
    }
    let host = host.strip_suffix('.').unwrap_or(host);
    if let Ok(ip) = host.parse::<IpAddr>() {
        return Some(HostRef::Ip(ip));
    }
    let labels_valid = host.split('.').all(|label| {
        !label.is_empty()
            && label
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
    });
    let last_label = host.rsplit('.').next().unwrap_or_default();
    let numeric = last_label.bytes().all(|byte| byte.is_ascii_digit())
        || last_label
            .get(..2)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("0x"));
    (labels_valid && !numeric).then_some(HostRef::Name(host))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TrustedHost {
    Ip(IpAddr),
    Name(String),
}

impl TrustedHost {
    fn matches(&self, host: &HostRef<'_>) -> bool {
        match (self, host) {
            (Self::Ip(trusted), HostRef::Ip(host)) => trusted == host,
            (Self::Name(trusted), HostRef::Name(host)) => trusted.eq_ignore_ascii_case(host),
            _ => false,
        }
    }
}

/// Destinations beyond loopback that may receive plaintext HTTP or WebSocket traffic.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) enum PlaintextPolicy {
    #[default]
    LoopbackOnly,
    Hosts(Vec<TrustedHost>),
    AnyHost,
}

impl PlaintextPolicy {
    pub(crate) fn from_allow_insecure(allow_insecure: bool) -> Self {
        if allow_insecure {
            Self::AnyHost
        } else {
            Self::LoopbackOnly
        }
    }

    /// Build a policy from `insecure_http_hosts`, rejecting entries that are not bare hosts.
    pub(crate) fn from_hosts(hosts: &[String]) -> Result<Self, String> {
        let mut trusted = Vec::with_capacity(hosts.len());
        for (index, host) in hosts.iter().enumerate() {
            let entry = match parse_host(host) {
                Some(HostRef::Ip(ip)) => TrustedHost::Ip(ip),
                Some(HostRef::Name(name)) => TrustedHost::Name(name.to_ascii_lowercase()),
                None => {
                    return Err(format!(
                        "insecure_http_hosts[{index}] must be a bare hostname or IP address without a scheme, port, path, user information, or wildcard"
                    ));
                }
            };
            trusted.push(entry);
        }
        Ok(if trusted.is_empty() {
            Self::LoopbackOnly
        } else {
            Self::Hosts(trusted)
        })
    }

    fn allows(&self, host: &str) -> bool {
        match self {
            Self::AnyHost => true,
            Self::LoopbackOnly => is_loopback_host(host),
            Self::Hosts(trusted) => {
                is_loopback_host(host)
                    || parse_host(host)
                        .is_some_and(|host| trusted.iter().any(|entry| entry.matches(&host)))
            }
        }
    }

    fn allowance(&self) -> &'static str {
        match self {
            Self::LoopbackOnly => {
                "localhost or loopback IP addresses unless the host is listed in insecure_http_hosts or allow_insecure is enabled"
            }
            Self::Hosts(_) => {
                "localhost, loopback IP addresses, or hosts listed in insecure_http_hosts"
            }
            Self::AnyHost => "any host",
        }
    }
}

/// Reject `insecure_http_hosts` entries that are not a bare hostname or IP address.
pub(crate) fn validate_insecure_http_hosts(hosts: &[String]) -> Result<(), String> {
    PlaintextPolicy::from_hosts(hosts).map(drop)
}

/// Return an endpoint's scheme, host, and port without userinfo, path, or query.
pub(crate) fn redacted_origin(endpoint: &str) -> Option<String> {
    let url = reqwest::Url::parse(endpoint).ok()?;
    let host = url.host_str()?;
    let port = url.port_or_known_default()?;
    Some(format!("{}://{host}:{port}", url.scheme()))
}

/// Explain why an endpoint may not carry protected traffic, or return `None` when it may.
fn plaintext_rejection(
    endpoint: &str,
    protected_scheme: &str,
    plaintext_scheme: &str,
    plaintext: &PlaintextPolicy,
) -> Option<String> {
    let invalid = || {
        Some(format!(
            "the endpoint must be a valid {protected_scheme}:// or {plaintext_scheme}:// URI"
        ))
    };
    let (Ok(url), Ok(uri)) = (reqwest::Url::parse(endpoint), endpoint.parse::<http::Uri>()) else {
        return invalid();
    };
    if url.scheme() == protected_scheme {
        return None;
    }
    if url.scheme() != plaintext_scheme {
        return invalid();
    }
    let allowed = url.host_str().is_some_and(|host| plaintext.allows(host))
        && uri.host().is_some_and(|host| plaintext.allows(host));
    if allowed {
        return None;
    }
    let origin = redacted_origin(endpoint).unwrap_or_else(|| "this endpoint".to_string());
    Some(format!(
        "{plaintext_scheme} to {origin} is allowed only for {}",
        plaintext.allowance()
    ))
}

/// Require HTTPS, or plain HTTP to a destination the policy allows, for OTLP trace export.
pub(crate) fn validate_trace_transport(
    endpoint: &str,
    plaintext: &PlaintextPolicy,
) -> Result<(), String> {
    match plaintext_rejection(endpoint, "https", "http", plaintext) {
        None => Ok(()),
        Some(reason) => Err(format!("OTLP trace endpoints require HTTPS; {reason}")),
    }
}

/// Return whether any configured header source adds request credentials or metadata.
pub(crate) fn has_configured_headers(
    headers: &HashMap<String, String>,
    header_env: &HashMap<String, String>,
    header_files: &HeaderFiles,
) -> bool {
    !headers.is_empty() || !header_env.is_empty() || !header_files.is_empty()
}

fn validate_header_destination(
    endpoint: &str,
    protected_scheme: &str,
    plaintext_scheme: &str,
    plaintext: &PlaintextPolicy,
) -> Result<(), String> {
    match plaintext_rejection(endpoint, protected_scheme, plaintext_scheme, plaintext) {
        None => Ok(()),
        Some(reason) => Err(format!(
            "configured headers require {protected_scheme} for remote endpoints; {reason}"
        )),
    }
}

/// Require protected HTTP transport before attaching configured headers.
pub(crate) fn validate_header_http_endpoint(
    endpoint: &str,
    plaintext: &PlaintextPolicy,
) -> Result<(), String> {
    validate_header_destination(endpoint, "https", "http", plaintext)
}

/// Require protected WebSocket transport before attaching configured headers.
#[cfg_attr(not(feature = "atof-streaming"), allow(dead_code))]
pub(crate) fn validate_header_websocket_endpoint(
    endpoint: &str,
    plaintext: &PlaintextPolicy,
) -> Result<(), String> {
    validate_header_destination(endpoint, "wss", "ws", plaintext)
}

/// Validate header source names and require configured files to exist.
pub(crate) fn validate_header_files(
    headers: &HashMap<String, String>,
    header_env: &HashMap<String, String>,
    header_files: &HeaderFiles,
) -> Result<(), String> {
    let mut names = HashSet::new();
    for key in headers
        .keys()
        .chain(header_env.keys())
        .chain(header_files.keys())
    {
        if !names.insert(key.to_ascii_lowercase()) {
            return Err(format!(
                "header {key:?} must be unique across headers and header_env and header_file"
            ));
        }
    }
    for (header, path) in header_files {
        reqwest::header::HeaderName::from_bytes(header.as_bytes())
            .map_err(|_| format!("header_file.{header} has an invalid header name"))?;
        if path.is_empty() {
            return Err(format!(
                "header_file.{header} must name a non-empty file path"
            ));
        }
        match fs::metadata(path) {
            Ok(metadata) if metadata.is_file() => {}
            Ok(_) => return Err(format!("header_file.{header} must name a regular file")),
            Err(_) => return Err(format!("header_file.{header} file is unavailable")),
        }
    }
    Ok(())
}

/// Read header values immediately before a request. Never include a value in errors.
pub(crate) fn resolve_header_files(
    header_files: &HeaderFiles,
) -> Result<HashMap<String, String>, String> {
    let mut resolved = HashMap::with_capacity(header_files.len());
    for (header, path) in header_files {
        let value = fs::read_to_string(path)
            .map_err(|_| format!("could not read header_file for header {header:?}"))?;
        let value = value.trim_end();
        if value.is_empty() {
            return Err(format!("header_file for header {header:?} is blank"));
        }
        reqwest::header::HeaderValue::from_str(value)
            .map_err(|_| format!("header_file for header {header:?} contains an invalid value"))?;
        resolved.insert(header.clone(), value.to_string());
    }
    Ok(resolved)
}

/// A request-time header resolver shared by OTLP HTTP and gRPC exporters.
#[derive(Clone, Debug)]
pub(crate) struct HeaderFileResolver(Arc<HeaderFiles>);

impl HeaderFileResolver {
    pub(crate) fn new(header_files: HeaderFiles) -> Self {
        Self(Arc::new(header_files))
    }

    pub(crate) fn resolve(&self) -> Result<HashMap<String, String>, String> {
        resolve_header_files(&self.0)
    }
}

/// Async OTLP HTTP client that applies current file-backed headers per request.
#[derive(Debug)]
pub(crate) struct HeaderFileHttpClient {
    inner: reqwest::Client,
    resolver: HeaderFileResolver,
    plaintext: PlaintextPolicy,
}

impl HeaderFileHttpClient {
    pub(crate) fn new(
        inner: reqwest::Client,
        resolver: HeaderFileResolver,
        plaintext: PlaintextPolicy,
    ) -> Self {
        Self {
            inner,
            resolver,
            plaintext,
        }
    }
}

#[async_trait]
impl HttpClient for HeaderFileHttpClient {
    async fn send_bytes(&self, request: Request<Bytes>) -> Result<Response<Bytes>, HttpError> {
        validate_header_http_endpoint(&request.uri().to_string(), &self.plaintext)
            .map_err(std::io::Error::other)?;
        let mut request = request;
        let resolver = self.resolver.clone();
        let headers = tokio::task::spawn_blocking(move || resolver.resolve())
            .await
            .map_err(|error| std::io::Error::other(error.to_string()))?
            .map_err(std::io::Error::other)?;
        for (header, value) in headers {
            let name = reqwest::header::HeaderName::from_bytes(header.as_bytes())
                .map_err(std::io::Error::other)?;
            let value =
                reqwest::header::HeaderValue::from_str(&value).map_err(std::io::Error::other)?;
            request.headers_mut().insert(name, value);
        }
        let request = request.try_into()?;
        let mut response = self.inner.execute(request).await?.error_for_status()?;
        let headers = std::mem::take(response.headers_mut());
        let mut http_response = Response::builder()
            .status(response.status())
            .body(response.bytes().await?)?;
        *http_response.headers_mut() = headers;
        Ok(http_response)
    }
}

/// Tonic request interceptor that applies current file-backed headers per RPC.
#[derive(Clone, Debug)]
pub(crate) struct HeaderFileInterceptor {
    resolver: HeaderFileResolver,
}

impl HeaderFileInterceptor {
    pub(crate) fn new(resolver: HeaderFileResolver) -> Self {
        Self { resolver }
    }
}

impl tonic::service::Interceptor for HeaderFileInterceptor {
    fn call(
        &mut self,
        mut request: tonic::Request<()>,
    ) -> Result<tonic::Request<()>, tonic::Status> {
        for (header, value) in self.resolver.resolve().map_err(tonic::Status::internal)? {
            let key = tonic::metadata::MetadataKey::from_bytes(header.as_bytes())
                .map_err(|_| tonic::Status::internal("header_file has an invalid header name"))?;
            let value = tonic::metadata::MetadataValue::try_from(value).map_err(|_| {
                tonic::Status::internal("header_file contains an invalid header value")
            })?;
            request.metadata_mut().insert(key, value);
        }
        Ok(request)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/observability/header_file_tests.rs"]
mod tests;
