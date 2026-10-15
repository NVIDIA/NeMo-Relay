// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request-time resolution for file-backed exporter headers.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::sync::Arc;

use async_trait::async_trait;
use opentelemetry_http::{Bytes, HttpClient, HttpError, Request, Response};

/// File paths keyed by HTTP header name.
pub(crate) type HeaderFiles = HashMap<String, String>;

fn is_loopback_host(host: &str) -> bool {
    host.eq_ignore_ascii_case("localhost")
        || host
            .trim_start_matches('[')
            .trim_end_matches(']')
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback())
}

fn canonical_host(host: &str) -> Option<String> {
    let bracketed = if host.contains(':') && !host.starts_with('[') {
        format!("[{host}]")
    } else {
        host.to_string()
    };
    let url = reqwest::Url::parse(&format!("http://{bracketed}/")).ok()?;
    let canonical = url
        .host_str()?
        .trim_start_matches('[')
        .trim_end_matches(']')
        .trim_end_matches('.')
        .to_ascii_lowercase();
    let bare = url.port().is_none()
        && url.path() == "/"
        && url.query().is_none()
        && url.fragment().is_none()
        && url.username().is_empty()
        && url.password().is_none();
    (bare && !canonical.is_empty()).then_some(canonical)
}

/// Reject `insecure_http_hosts` entries that are not a bare hostname or IP address.
pub(crate) fn validate_insecure_http_hosts(hosts: &[String]) -> Result<(), String> {
    for host in hosts {
        let malformed = host.contains("://")
            || host.contains('*')
            || host.chars().any(char::is_whitespace)
            || canonical_host(host).is_none();
        if malformed {
            return Err(format!(
                "insecure_http_hosts entry {host:?} must be a bare hostname or IP address without a scheme, port, path, or wildcard"
            ));
        }
    }
    Ok(())
}

fn plaintext_host_allowed(host: &str, insecure_http_hosts: &[String]) -> bool {
    is_loopback_host(host)
        || canonical_host(host).is_some_and(|host| {
            insecure_http_hosts
                .iter()
                .any(|trusted| canonical_host(trusted).as_deref() == Some(host.as_str()))
        })
}

/// Return whether a URL uses HTTPS or plain HTTP to loopback or an explicitly trusted host.
pub(crate) fn is_protected_http_endpoint(endpoint: &str, insecure_http_hosts: &[String]) -> bool {
    reqwest::Url::parse(endpoint).is_ok_and(|url| {
        url.scheme() == "https"
            || (url.scheme() == "http"
                && url
                    .host_str()
                    .is_some_and(|host| plaintext_host_allowed(host, insecure_http_hosts)))
    })
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
    insecure_http_hosts: &[String],
) -> Result<(), String> {
    let protected = reqwest::Url::parse(endpoint).is_ok_and(|url| {
        url.scheme() == protected_scheme
            || (url.scheme() == plaintext_scheme
                && url
                    .host_str()
                    .is_some_and(|host| plaintext_host_allowed(host, insecure_http_hosts)))
    });
    if protected {
        Ok(())
    } else if insecure_http_hosts.is_empty() {
        Err(format!(
            "configured headers require {protected_scheme} for remote endpoints; {plaintext_scheme} is allowed only for localhost or loopback IP addresses"
        ))
    } else {
        Err(format!(
            "configured headers require {protected_scheme} for remote endpoints; {plaintext_scheme} is allowed only for localhost, loopback IP addresses, or hosts listed in insecure_http_hosts"
        ))
    }
}

/// Require protected HTTP transport before attaching configured headers.
pub(crate) fn validate_header_http_endpoint(
    endpoint: &str,
    insecure_http_hosts: &[String],
) -> Result<(), String> {
    validate_header_destination(endpoint, "https", "http", insecure_http_hosts)
}

/// Require protected WebSocket transport before attaching configured headers.
#[cfg_attr(not(feature = "atof-streaming"), allow(dead_code))]
pub(crate) fn validate_header_websocket_endpoint(endpoint: &str) -> Result<(), String> {
    validate_header_destination(endpoint, "wss", "ws", &[])
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
    insecure_http_hosts: Vec<String>,
}

impl HeaderFileHttpClient {
    pub(crate) fn new(
        inner: reqwest::Client,
        resolver: HeaderFileResolver,
        insecure_http_hosts: Vec<String>,
    ) -> Self {
        Self {
            inner,
            resolver,
            insecure_http_hosts,
        }
    }
}

#[async_trait]
impl HttpClient for HeaderFileHttpClient {
    async fn send_bytes(&self, request: Request<Bytes>) -> Result<Response<Bytes>, HttpError> {
        validate_header_http_endpoint(&request.uri().to_string(), &self.insecure_http_hosts)
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
