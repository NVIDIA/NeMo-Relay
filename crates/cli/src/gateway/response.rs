// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Observable header policy and downstream response construction.

use axum::body::Body;
use axum::http::{HeaderMap, HeaderName, Response, StatusCode};
use serde_json::{Map, Value, json};

use crate::configuration::BOOTSTRAP_CLIENT_TOKEN_HEADER;
use crate::error::CliError;

pub(super) fn observable_headers(headers: &HeaderMap) -> Map<String, Value> {
    let mut output = Map::new();
    for (name, value) in headers {
        if should_record_header(name)
            && let Ok(value) = value.to_str()
        {
            output.insert(name.as_str().to_string(), json!(value));
        }
    }
    output
}

// Copies upstream response headers except hop-by-hop transport headers that Axum/hyper must manage
// for the downstream connection. Multiple values are appended to preserve provider behavior.
// Content-Length is also dropped because the gateway re-encodes streaming responses and the
// upstream-reported length will not match the bytes the client sees.
pub(super) fn response_headers(headers: &HeaderMap) -> HeaderMap {
    let mut output = HeaderMap::new();
    for (name, value) in headers {
        if !is_hop_by_hop(name) && name != http::header::CONTENT_LENGTH {
            output.append(name.clone(), value.clone());
        }
    }
    output
}

// Reconstructs an Axum response from upstream status, filtered headers, and the selected body. All
// builder errors are converted into gateway HTTP errors rather than panics.
pub(super) fn build_response(
    status: StatusCode,
    headers: HeaderMap,
    body: Body,
) -> Result<Response<Body>, CliError> {
    let mut builder = Response::builder().status(status);
    for (name, value) in &headers {
        builder = builder.header(name, value);
    }
    Ok(builder.body(body)?)
}

// Allows provider request headers through unless they are transport-owned or must be recalculated
// for the forwarded body. Host and content length are intentionally excluded because reqwest sets
// them for the upstream connection.
pub(super) fn should_forward_request_header(name: &HeaderName) -> bool {
    !is_hop_by_hop(name)
        && name != http::header::HOST
        && name != http::header::CONTENT_LENGTH
        && name.as_str() != BOOTSTRAP_CLIENT_TOKEN_HEADER
        // Strip Accept-Encoding so upstreams return identity-encoded bodies; otherwise the
        // observability capture (`output.value` on LLM spans, ATIF trajectory bodies) records
        // gzip/br/zstd bytes that downstream consumers can't read. Bandwidth cost is paid only
        // on the gateway-upstream hop. The client never asked for the encoding it would have
        // received from upstream, so its decoders never trigger.
        && name != http::header::ACCEPT_ENCODING
}

// Allows headers into observability metadata only after removing credentials and provider API keys.
// The forwarding filter runs first so hop-by-hop transport headers are also excluded from recorded
// LLM request attributes. The credential blocklist covers the four canonical cases we see in
// practice: `Authorization` (most providers), `Cookie` (session credentials), `x-api-key` (OpenAI
// SDK and similar), `anthropic-api-key` (Anthropic), and the generic `api-key` alias used by some
// providers/proxies (e.g., Azure OpenAI). `HeaderName::as_str()` already returns the canonical
// lowercase form so string comparisons are case-insensitive by construction.
pub(super) fn should_record_header(name: &HeaderName) -> bool {
    should_forward_request_header(name)
        && name != http::header::AUTHORIZATION
        && name != http::header::COOKIE
        && name.as_str() != "x-api-key"
        && name.as_str() != "api-key"
        && name.as_str() != "anthropic-api-key"
}

// Identifies headers that describe a single transport hop and therefore must not be proxied across
// the client-gateway-upstream boundary.
pub(super) fn is_hop_by_hop(name: &HeaderName) -> bool {
    matches!(
        name.as_str(),
        "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
    )
}
