// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::http::{HeaderMap, HeaderValue};

use super::*;

/// The base a decision names, or `None` for anything that is not a usable destination.
fn named(decision: NamedUpstream) -> Option<String> {
    match decision {
        NamedUpstream::Named(base) => Some(base),
        _ => None,
    }
}

fn is_rejected(decision: NamedUpstream) -> bool {
    matches!(decision, NamedUpstream::Rejected(_))
}

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
        named(client_named_upstream_base(&headers, true)).as_deref(),
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

    assert!(matches!(
        client_named_upstream_base(&headers, false),
        NamedUpstream::Absent
    ));
}

#[test]
fn a_request_without_the_header_names_nothing() {
    assert!(matches!(
        client_named_upstream_base(&HeaderMap::new(), true),
        NamedUpstream::Absent
    ));
}

#[test]
fn a_blank_header_names_nothing() {
    assert!(is_rejected(client_named_upstream_base(
        &headers_naming("   "),
        true
    )));
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
        assert!(
            is_rejected(client_named_upstream_base(&headers_naming(rejected), true)),
            "{rejected} must be refused, not silently ignored"
        );
    }
}

/// Cleartext is allowed only where it cannot leave the machine.
///
/// A local model server is exactly the kind of provider this feature exists to reach, and its
/// traffic never touches a network, so requiring TLS there would cost the main use case
/// without protecting anything.
#[test]
fn loopback_upstreams_may_use_plain_http() {
    for accepted in [
        "http://127.0.0.1:8000/v1",
        "http://localhost:11434/v1",
        "http://[::1]:8000/v1",
        "http://127.2.3.4:8000/v1",
    ] {
        assert_eq!(
            named(client_named_upstream_base(&headers_naming(accepted), true)).as_deref(),
            Some(accepted),
            "{accepted} is unreachable from off the machine and must be allowed"
        );
    }
}

/// The named destination is forwarded the provider credential the request carried, so plain
/// `http` to anywhere reachable would put that key on the wire in the clear.
#[test]
fn a_remote_upstream_must_use_https() {
    for rejected in [
        "http://integrate.api.nvidia.com/v1",
        "http://192.168.1.10:8000/v1",
        "http://example.com/v1",
        // Reserved for loopback by RFC 6761, but it still resolves through the host's resolver.
        "http://ollama.localhost/v1",
    ] {
        assert!(
            is_rejected(client_named_upstream_base(&headers_naming(rejected), true)),
            "{rejected} would carry a provider credential in cleartext off the machine"
        );
    }

    // The same hosts over TLS are fine.
    assert_eq!(
        named(client_named_upstream_base(
            &headers_naming("https://integrate.api.nvidia.com/v1"),
            true
        ))
        .as_deref(),
        Some("https://integrate.api.nvidia.com/v1")
    );
}

/// A base is a prefix and the request path is appended to it, so neither of these survives.
///
/// A query would land before the path, and a fragment would swallow it — everything after `#`
/// is never sent to the server at all. Either one routes somewhere other than the endpoint the
/// caller named, which is the failure this whole path exists to avoid.
#[test]
fn a_base_carrying_a_query_or_fragment_is_refused() {
    for rejected in [
        "https://provider.example/api?version=1",
        "https://provider.example/api#section",
        "https://provider.example/api?version=1#section",
        // Empty but present still changes the composed URL.
        "https://provider.example/api?",
        "https://provider.example/api#",
    ] {
        assert!(
            is_rejected(client_named_upstream_base(&headers_naming(rejected), true)),
            "{rejected} cannot have a request path appended to it"
        );
    }
}
