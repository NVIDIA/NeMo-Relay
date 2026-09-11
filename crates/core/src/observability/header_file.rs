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

fn validate_header_file_destination(
    endpoint: &str,
    protected_scheme: &str,
    plaintext_scheme: &str,
) -> Result<(), String> {
    let protected = reqwest::Url::parse(endpoint).is_ok_and(|url| {
        url.scheme() == protected_scheme
            || (url.scheme() == plaintext_scheme && url.host_str().is_some_and(is_loopback_host))
    });
    if protected {
        Ok(())
    } else {
        Err(format!(
            "header_file requires {protected_scheme} for remote endpoints; {plaintext_scheme} is allowed only for localhost or loopback IP addresses"
        ))
    }
}

/// Require protected HTTP transport before attaching file-backed values.
pub(crate) fn validate_header_file_http_endpoint(endpoint: &str) -> Result<(), String> {
    validate_header_file_destination(endpoint, "https", "http")
}

/// Require protected WebSocket transport before attaching file-backed values.
#[cfg_attr(not(feature = "atof-streaming"), allow(dead_code))]
pub(crate) fn validate_header_file_websocket_endpoint(endpoint: &str) -> Result<(), String> {
    validate_header_file_destination(endpoint, "wss", "ws")
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

/// Blocking OTLP HTTP client that applies current file-backed headers per request.
#[derive(Debug)]
pub(crate) struct HeaderFileHttpClient {
    inner: reqwest_otel::blocking::Client,
    resolver: HeaderFileResolver,
}

impl HeaderFileHttpClient {
    pub(crate) fn new(inner: reqwest_otel::blocking::Client, resolver: HeaderFileResolver) -> Self {
        Self { inner, resolver }
    }
}

#[async_trait]
impl HttpClient for HeaderFileHttpClient {
    async fn send_bytes(&self, request: Request<Bytes>) -> Result<Response<Bytes>, HttpError> {
        let client = self.inner.clone();
        let resolver = self.resolver.clone();
        tokio::task::spawn_blocking(move || {
            validate_header_file_http_endpoint(&request.uri().to_string())
                .map_err(std::io::Error::other)?;
            let mut request = request;
            for (header, value) in resolver.resolve().map_err(std::io::Error::other)? {
                let name = reqwest::header::HeaderName::from_bytes(header.as_bytes())
                    .map_err(std::io::Error::other)?;
                let value = reqwest::header::HeaderValue::from_str(&value)
                    .map_err(std::io::Error::other)?;
                request.headers_mut().insert(name, value);
            }
            let request = request.try_into()?;
            let mut response = client.execute(request)?.error_for_status()?;
            let headers = std::mem::take(response.headers_mut());
            let mut http_response = Response::builder()
                .status(response.status())
                .body(response.bytes()?)?;
            *http_response.headers_mut() = headers;
            Ok(http_response)
        })
        .await
        .map_err(std::io::Error::other)?
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
