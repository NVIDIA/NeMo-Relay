// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! W3C trace context inherited from the process that launched the coding agent.
//!
//! Coding agents run hook commands in their own environment. A host that launches the agent with
//! `TRACEPARENT` (and optionally `TRACESTATE`) set therefore reaches every hook forwarder, which
//! sends the pair as standard W3C headers so the receiving session can join the host's trace.

use reqwest::header::HeaderValue;

/// Environment variable carrying the launcher's W3C `traceparent`.
pub(crate) const TRACEPARENT_ENV: &str = "TRACEPARENT";
/// Environment variable carrying the launcher's W3C `tracestate`.
pub(crate) const TRACESTATE_ENV: &str = "TRACESTATE";
/// Hook request header carrying the launcher's W3C `traceparent`.
pub(crate) const TRACEPARENT_HEADER: &str = "traceparent";
/// Hook request header carrying the launcher's W3C `tracestate`.
pub(crate) const TRACESTATE_HEADER: &str = "tracestate";

/// Returns the launcher trace headers found in this process's environment.
pub(crate) fn launcher_trace_headers() -> Vec<(&'static str, String)> {
    launcher_trace_headers_from(|name| std::env::var(name).ok())
}

/// Returns the launcher trace headers produced by `lookup`.
///
/// Only values that are legal HTTP header text are forwarded, and `tracestate` is forwarded only
/// with a `traceparent`. The receiver validates W3C syntax and ignores an invalid pair, so a
/// malformed environment never fails hook delivery.
pub(crate) fn launcher_trace_headers_from(
    lookup: impl Fn(&str) -> Option<String>,
) -> Vec<(&'static str, String)> {
    let header_text = |name: &str| {
        lookup(name)
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty() && HeaderValue::from_str(value).is_ok())
    };
    let Some(traceparent) = header_text(TRACEPARENT_ENV) else {
        return Vec::new();
    };
    let mut headers = vec![(TRACEPARENT_HEADER, traceparent)];
    if let Some(tracestate) = header_text(TRACESTATE_ENV) {
        headers.push((TRACESTATE_HEADER, tracestate));
    }
    headers
}
