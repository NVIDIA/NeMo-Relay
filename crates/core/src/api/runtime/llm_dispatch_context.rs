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
use reqwest::{Client, StatusCode, Url};

use crate::api::llm::LlmRequest;
use crate::api::runtime::scope_stack::active_event_uuid;
use crate::api::runtime::{LlmExecutionNextFn, LlmJsonStream, LlmStreamExecutionNextFn};
use crate::codec::streaming::SseEventDecoder;
use crate::error::{
    FlowError, MAX_UPSTREAM_FAILURE_BODY_BYTES, Result, UpstreamFailure, UpstreamFailureClass,
    sanitize_upstream_failure_headers,
};
use crate::json::Json;

const HTTP_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const HTTP_REQUEST_TIMEOUT: Duration = Duration::from_secs(300);
const HTTP_READ_TIMEOUT: Duration = Duration::from_secs(300);
tokio::task_local! {
    static TASK_LLM_DISPATCH_TARGET: LlmDispatchTargetBinding;
}

#[derive(Clone)]
struct LlmDispatchTargetBinding {
    // The active LLM event identifies the exact managed continuation chain.
    // Nested managed calls install their own event UUID and cannot consume it.
    active_event_uuid: Option<uuid::Uuid>,
    target: LlmDispatchTargetContext,
}

/// Validated provider transport target bound to one LLM continuation invocation.
///
/// The target stays outside [`crate::api::llm::LlmRequest`] so credentials and
/// transport routing cannot leak into provider JSON or observability payloads.
#[doc(hidden)]
#[derive(Clone)]
pub struct LlmDispatchTargetContext {
    url: Url,
    headers: HeaderMap,
}

impl LlmDispatchTargetContext {
    /// Validate and construct a target for one continuation invocation.
    pub(crate) fn try_new(url: String, headers: BTreeMap<String, String>) -> Result<Self> {
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
            if validated_headers.contains_key(&name) {
                return Err(FlowError::InvalidArgument(format!(
                    "LLM continuation target header {name} was specified more than once"
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
            url,
            headers: validated_headers,
        })
    }

    /// Absolute provider URL selected for this invocation.
    #[doc(hidden)]
    #[must_use]
    pub(crate) fn url(&self) -> &Url {
        &self.url
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
            .field("url", &redacted_url)
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
    TASK_LLM_DISPATCH_TARGET
        .try_with(|binding| {
            (binding.active_event_uuid == active_event_uuid()).then(|| binding.target.clone())
        })
        .ok()
        .flatten()
}

/// Poll a future with one typed target bound to its continuation invocation.
pub(crate) async fn scope_llm_dispatch_target<F: Future>(
    event_uuid: Option<uuid::Uuid>,
    target: LlmDispatchTargetContext,
    future: F,
) -> F::Output {
    TASK_LLM_DISPATCH_TARGET
        .scope(
            LlmDispatchTargetBinding {
                active_event_uuid: event_uuid,
                target,
            },
            future,
        )
        .await
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
    let response = send(target, request, Some(HTTP_REQUEST_TIMEOUT)).await?;
    let status = response.status();
    if !status.is_success() {
        let headers = safe_failure_headers(response.headers());
        let bytes = bounded_response_body(target, response).await?;
        return Err(http_error(status, headers, &bytes));
    }
    let bytes = response
        .bytes()
        .await
        .map_err(|error| transport_error(target, error))?;
    serde_json::from_slice(&bytes).map_err(|_| {
        FlowError::Internal("targeted LLM provider returned malformed response JSON".into())
    })
}

async fn dispatch_stream(
    target: &LlmDispatchTargetContext,
    request: LlmRequest,
) -> Result<LlmJsonStream> {
    let response = send(target, request, None).await?;
    let status = response.status();
    if !status.is_success() {
        let headers = safe_failure_headers(response.headers());
        let body = bounded_response_body(target, response).await?;
        return Err(http_error(status, headers, &body));
    }

    let target = target.clone();
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
                    yield Err(transport_error(&target, error));
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

async fn send(
    target: &LlmDispatchTargetContext,
    request: LlmRequest,
    timeout: Option<Duration>,
) -> Result<reqwest::Response> {
    let body = serde_json::to_vec(&request.content)
        .map_err(|error| FlowError::InvalidArgument(error.to_string()))?;
    let mut outbound = targeted_http_client().post(target.url().clone()).body(body);
    for (name, value) in target.headers() {
        outbound = outbound.header(name, value);
    }
    if let Some(timeout) = timeout {
        outbound = outbound.timeout(timeout);
    }
    outbound
        .send()
        .await
        .map_err(|error| transport_error(target, error))
}

fn targeted_http_client() -> &'static Client {
    static CLIENT: OnceLock<Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        Client::builder()
            .connect_timeout(HTTP_CONNECT_TIMEOUT)
            .read_timeout(HTTP_READ_TIMEOUT)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .expect("core targeted LLM HTTP client configuration is valid")
    })
}

async fn bounded_response_body(
    target: &LlmDispatchTargetContext,
    response: reqwest::Response,
) -> Result<Vec<u8>> {
    let mut body = Vec::new();
    let mut stream = response.bytes_stream();
    while body.len() < MAX_UPSTREAM_FAILURE_BODY_BYTES {
        let Some(chunk) = stream.next().await else {
            break;
        };
        let chunk = chunk.map_err(|error| transport_error(target, error))?;
        let remaining = MAX_UPSTREAM_FAILURE_BODY_BYTES - body.len();
        body.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
    }
    Ok(body)
}

fn transport_error(target: &LlmDispatchTargetContext, error: reqwest::Error) -> FlowError {
    let timeout = error.is_timeout();
    let diagnostic = error.without_url();
    log::warn!(
        target: "nemo_relay.runtime",
        event = "targeted_llm_transport_failed",
        provider_host = target.url().host_str().unwrap_or("<unknown>"),
        failure_kind = if timeout { "timeout" } else { "transport" };
        "Targeted LLM provider request failed: {diagnostic}"
    );
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
    let body = String::from_utf8_lossy(&body[..body.len().min(MAX_UPSTREAM_FAILURE_BODY_BYTES)]);
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
    sanitize_upstream_failure_headers(headers.iter().map(|(name, value)| {
        (
            name.as_str().to_owned(),
            String::from_utf8_lossy(value.as_bytes()).into_owned(),
        )
    }))
}

#[cfg(test)]
#[path = "../../../tests/unit/llm_dispatch_context_tests.rs"]
mod tests;
