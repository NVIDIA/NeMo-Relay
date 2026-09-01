// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::http::{HeaderMap, HeaderValue};

use super::*;

fn headers_naming(upstream: &str) -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(
        UPSTREAM_BASE_URL_HEADER,
        HeaderValue::from_str(upstream).expect("test upstream must be a legal header value"),
    );
    headers
}

#[test]
fn an_authenticated_request_names_its_own_upstream() {
    let headers = headers_naming("https://integrate.api.nvidia.com/v1");

    assert_eq!(
        client_named_upstream_base(&headers, true).as_deref(),
        Some("https://integrate.api.nvidia.com/v1"),
        "the base is returned as written, so an operator's path prefix survives"
    );
}

/// The whole security boundary, stated as a test.
///
/// Without the invocation credential this is an unauthenticated local caller telling a gateway
/// where to send traffic that carries provider keys. It has to read as absent, not as an error,
/// so the request still completes against the configured upstream.
#[test]
fn an_unauthenticated_request_cannot_name_an_upstream() {
    let headers = headers_naming("https://integrate.api.nvidia.com/v1");

    assert_eq!(client_named_upstream_base(&headers, false), None);
}

#[test]
fn a_request_without_the_header_names_nothing() {
    assert_eq!(client_named_upstream_base(&HeaderMap::new(), true), None);
}

#[test]
fn a_blank_header_names_nothing() {
    assert_eq!(
        client_named_upstream_base(&headers_naming("   "), true),
        None
    );
}

/// Each of these would reach somewhere the caller should not be able to send credentialed traffic.
#[test]
fn only_absolute_http_urls_with_a_bare_host_are_accepted() {
    for rejected in [
        // Resolves against the gateway itself rather than naming a destination.
        "/v1/chat/completions",
        "api.openai.com/v1",
        // Non-http schemes reqwest treats very differently.
        "file:///etc/passwd",
        "ftp://example.com",
        // Credentials in the authority would travel to whatever host follows them.
        "https://user:secret@example.com/v1",
        "https://user@example.com/v1",
        // Parses, but there is no host to forward to.
        "https://",
    ] {
        assert_eq!(
            client_named_upstream_base(&headers_naming(rejected), true),
            None,
            "{rejected} must not be accepted as an upstream"
        );
    }
}

#[test]
fn loopback_and_plain_http_upstreams_are_allowed() {
    // A local model server or an on-prem proxy is a legitimate destination, and the credential
    // already establishes that the caller is this invocation's own agent.
    for accepted in ["http://127.0.0.1:8000/v1", "http://localhost:11434/v1"] {
        assert_eq!(
            client_named_upstream_base(&headers_naming(accepted), true).as_deref(),
            Some(accepted)
        );
    }
}
