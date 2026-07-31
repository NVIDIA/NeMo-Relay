// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Invocation-scoped target and core HTTP transport for managed LLM continuations.

use std::collections::BTreeMap;
use std::fmt;
use std::future::Future;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use async_stream::stream;
use futures_util::StreamExt;
use reqwest::header::{self, HeaderMap, HeaderName, HeaderValue};
use reqwest::{Client, Method, StatusCode, Url};

use crate::api::llm::LlmRequest;
use crate::api::runtime::{LlmExecutionNextFn, LlmJsonStream, LlmStreamExecutionNextFn};
use crate::codec::streaming::SseEventDecoder;
use crate::error::{FlowError, Result, UpstreamFailure, UpstreamFailureClass};
use crate::json::Json;

const HTTP_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const HTTP_REQUEST_TIMEOUT: Duration = Duration::from_secs(300);
const HTTP_READ_TIMEOUT: Duration = Duration::from_secs(300);
const MAX_UPSTREAM_ERROR_BODY_BYTES: usize = 16 * 1024;
const MAX_UPSTREAM_ERROR_HEADER_VALUE_BYTES: usize = 1024;

tokio::task_local! {
    static TASK_LLM_DISPATCH_TARGET: LlmDispatchTargetContext;
}

/// Validated provider transport target bound to one LLM continuation invocation.
///
/// The target stays outside [`crate::api::llm::LlmRequest`] so credentials and
/// transport routing cannot leak into provider JSON or observability payloads.
#[doc(hidden)]
#[derive(Clone)]
pub struct LlmDispatchTargetContext {
    method: Method,
    url: Url,
    route: String,
    headers: HeaderMap,
}

impl LlmDispatchTargetContext {
    /// Validate and construct a target for one continuation invocation.
    pub(crate) fn try_new(
        method: String,
        url: String,
        route: String,
        headers: BTreeMap<String, String>,
    ) -> Result<Self> {
        let method = Method::from_bytes(method.as_bytes()).map_err(|_| {
            FlowError::InvalidArgument("LLM continuation method was invalid or prohibited".into())
        })?;
        if matches!(method, Method::CONNECT | Method::TRACE) {
            return Err(FlowError::InvalidArgument(
                "LLM continuation method was invalid or prohibited".into(),
            ));
        }
        let url = Url::parse(&url).map_err(|_| invalid_target_url())?;
        if !matches!(url.scheme(), "http" | "https")
            || !url.has_host()
            || !url.username().is_empty()
            || url.password().is_some()
        {
            return Err(invalid_target_url());
        }
        let mut validated_headers = HeaderMap::new();
        for (name, value) in headers {
            let name = HeaderName::from_bytes(name.as_bytes()).map_err(|_| {
                FlowError::InvalidArgument(
                    "LLM continuation contained an invalid target header name".into(),
                )
            })?;
            if prohibited_target_header(&name) {
                return Err(FlowError::InvalidArgument(format!(
                    "LLM continuation target header {name} is host-owned or prohibited"
                )));
            }
            let value = HeaderValue::from_str(&value).map_err(|_| {
                FlowError::InvalidArgument(format!(
                    "LLM continuation target header {name} had an invalid value"
                ))
            })?;
            validated_headers.insert(name, value);
        }
        validated_headers
            .entry(header::CONTENT_TYPE)
            .or_insert(HeaderValue::from_static("application/json"));
        Ok(Self {
            method,
            url,
            route,
            headers: validated_headers,
        })
    }

    /// HTTP method selected for this invocation.
    #[doc(hidden)]
    #[must_use]
    pub(crate) fn method(&self) -> &Method {
        &self.method
    }

    /// Absolute provider URL selected for this invocation.
    #[doc(hidden)]
    #[must_use]
    pub(crate) fn url(&self) -> &Url {
        &self.url
    }

    #[cfg(test)]
    pub(crate) fn route(&self) -> &str {
        &self.route
    }

    /// Explicit provider headers selected for this invocation.
    #[doc(hidden)]
    #[must_use]
    pub(crate) fn headers(&self) -> &HeaderMap {
        &self.headers
    }
}

impl fmt::Debug for LlmDispatchTargetContext {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut redacted_url = self.url.clone();
        redacted_url.set_query(None);
        redacted_url.set_fragment(None);
        formatter
            .debug_struct("LlmDispatchTargetContext")
            .field("method", &self.method)
            .field("url", &redacted_url)
            .field("route", &self.route)
            .field(
                "header_names",
                &self
                    .headers
                    .keys()
                    .map(HeaderName::as_str)
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

fn invalid_target_url() -> FlowError {
    FlowError::InvalidArgument(
        "LLM continuation target must be an absolute HTTP(S) URL without user info".into(),
    )
}

fn prohibited_target_header(name: &HeaderName) -> bool {
    let name = name.as_str();
    name.starts_with("x-nemo-relay-internal-")
        || matches!(
            name,
            "host"
                | "content-length"
                | "connection"
                | "transfer-encoding"
                | "upgrade"
                | "proxy-connection"
                | "keep-alive"
                | "trailer"
                | "te"
        )
}

pub(crate) fn current_llm_dispatch_target() -> Option<LlmDispatchTargetContext> {
    TASK_LLM_DISPATCH_TARGET.try_with(Clone::clone).ok()
}

/// Poll a future with one typed target bound to its continuation invocation.
pub(crate) async fn scope_llm_dispatch_target<F: Future>(
    target: LlmDispatchTargetContext,
    future: F,
) -> F::Output {
    TASK_LLM_DISPATCH_TARGET.scope(target, future).await
}

/// Wrap a host callback with core-owned targeted dispatch at the terminal step.
pub(crate) fn targeted_llm_execution(fallback: LlmExecutionNextFn) -> LlmExecutionNextFn {
    Arc::new(move |request| {
        let fallback = fallback.clone();
        Box::pin(async move {
            match current_llm_dispatch_target() {
                Some(target) => dispatch_buffered(&target, request).await,
                None => fallback(request).await,
            }
        })
    })
}

/// Wrap a streaming host callback with core-owned targeted dispatch at the terminal step.
pub(crate) fn targeted_llm_stream_execution(
    fallback: LlmStreamExecutionNextFn,
) -> LlmStreamExecutionNextFn {
    Arc::new(move |request| {
        let fallback = fallback.clone();
        Box::pin(async move {
            match current_llm_dispatch_target() {
                Some(target) => dispatch_stream(&target, request).await,
                None => fallback(request).await,
            }
        })
    })
}

async fn dispatch_buffered(target: &LlmDispatchTargetContext, request: LlmRequest) -> Result<Json> {
    let response = send(target, request).await?;
    let status = response.status();
    let headers = safe_failure_headers(response.headers());
    if !status.is_success() {
        let bytes = bounded_response_body(response).await?;
        return Err(http_error(status, headers, &bytes));
    }
    let bytes = response.bytes().await.map_err(transport_error)?;
    serde_json::from_slice(&bytes).map_err(|_| http_error(status, headers, &bytes))
}

async fn dispatch_stream(
    target: &LlmDispatchTargetContext,
    request: LlmRequest,
) -> Result<LlmJsonStream> {
    let response = send(target, request).await?;
    let status = response.status();
    if !status.is_success() {
        let headers = safe_failure_headers(response.headers());
        let body = bounded_response_body(response).await?;
        return Err(http_error(status, headers, &body));
    }

    let mut decoder = SseEventDecoder::new();
    let mut bytes = response.bytes_stream();
    Ok(LlmJsonStream::new(stream! {
        while let Some(chunk) = bytes.next().await {
            match chunk {
                Ok(buffer) => {
                    for result in decoder.push_bytes_results(&buffer) {
                        match result {
                            Ok(event) => yield Ok(event.data),
                            Err(error) => {
                                yield Err(error);
                                return;
                            }
                        }
                    }
                }
                Err(error) => {
                    yield Err(transport_error(error));
                    return;
                }
            }
        }
        match decoder.finish() {
            Ok(Some(event)) => yield Ok(event.data),
            Ok(None) => {}
            Err(error) => yield Err(error),
        }
    }))
}

async fn send(target: &LlmDispatchTargetContext, request: LlmRequest) -> Result<reqwest::Response> {
    let body = serde_json::to_vec(&request.content)
        .map_err(|error| FlowError::InvalidArgument(error.to_string()))?;
    let mut outbound = targeted_http_client()
        .request(target.method().clone(), target.url().clone())
        .body(body);
    for (name, value) in target.headers() {
        outbound = outbound.header(name, value);
    }
    outbound.send().await.map_err(transport_error)
}

fn targeted_http_client() -> &'static Client {
    static CLIENT: OnceLock<Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        Client::builder()
            .connect_timeout(HTTP_CONNECT_TIMEOUT)
            .timeout(HTTP_REQUEST_TIMEOUT)
            .read_timeout(HTTP_READ_TIMEOUT)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .expect("core targeted LLM HTTP client configuration is valid")
    })
}

async fn bounded_response_body(response: reqwest::Response) -> Result<Vec<u8>> {
    let mut body = Vec::new();
    let mut stream = response.bytes_stream();
    while body.len() < MAX_UPSTREAM_ERROR_BODY_BYTES {
        let Some(chunk) = stream.next().await else {
            break;
        };
        let chunk = chunk.map_err(transport_error)?;
        let remaining = MAX_UPSTREAM_ERROR_BODY_BYTES - body.len();
        body.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
    }
    Ok(body)
}

fn transport_error(error: reqwest::Error) -> FlowError {
    let timeout = error.is_timeout();
    FlowError::Upstream(UpstreamFailure {
        status: None,
        body: if timeout {
            "provider request timed out".into()
        } else {
            "provider transport failed".into()
        },
        headers: BTreeMap::new(),
        class: if timeout {
            UpstreamFailureClass::Timeout
        } else {
            UpstreamFailureClass::Connection
        },
    })
}

fn http_error(status: StatusCode, headers: BTreeMap<String, String>, body: &[u8]) -> FlowError {
    let body = String::from_utf8_lossy(&body[..body.len().min(MAX_UPSTREAM_ERROR_BODY_BYTES)]);
    FlowError::Upstream(UpstreamFailure {
        status: Some(status.as_u16()),
        body: body.into_owned(),
        headers,
        class: if matches!(status.as_u16(), 408 | 425 | 429 | 500 | 502 | 503 | 504) {
            UpstreamFailureClass::RetryableStatus
        } else {
            UpstreamFailureClass::Other
        },
    })
}

fn safe_failure_headers(headers: &HeaderMap) -> BTreeMap<String, String> {
    headers
        .iter()
        .filter_map(|(name, value)| {
            let name = name.as_str();
            matches!(
                name,
                "retry-after"
                    | "request-id"
                    | "traceparent"
                    | "x-request-id"
                    | "x-ratelimit-limit"
                    | "x-ratelimit-remaining"
                    | "x-ratelimit-reset"
                    | "ratelimit-limit"
                    | "ratelimit-remaining"
                    | "ratelimit-reset"
            )
            .then(|| {
                (
                    name.to_owned(),
                    bounded_utf8(
                        String::from_utf8_lossy(value.as_bytes()).into_owned(),
                        MAX_UPSTREAM_ERROR_HEADER_VALUE_BYTES,
                    ),
                )
            })
        })
        .collect()
}

fn bounded_utf8(value: String, max_bytes: usize) -> String {
    if value.len() <= max_bytes {
        return value;
    }
    let mut boundary = max_bytes;
    while !value.is_char_boundary(boundary) {
        boundary -= 1;
    }
    value[..boundary].to_owned()
}

#[cfg(test)]
#[path = "../../../tests/unit/llm_dispatch_context_tests.rs"]
mod tests;
