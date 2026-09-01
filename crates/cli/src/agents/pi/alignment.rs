// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A client-named upstream, for pi models the gateway does not statically front.
//!
//! The gateway forwards to one configured upstream per API family, so redirecting pi's model
//! traffic is only correct when that upstream is already the endpoint the selected model would
//! otherwise call. pi resolves a base URL per model from a catalog of dozens of providers, so
//! for most of them it is not, and the extension refuses to redirect rather than break the
//! session. The cost is that those models produce no LLM spans and get no model-call
//! enforcement.
//!
//! This lets the extension name the endpoint instead, in a request header, so one gateway can
//! front a provider it was never configured for.
//!
//! **Why this is not simply reusing the internal dispatch header.**
//! `x-nemo-relay-internal-dispatch-url` already redirects a single request, and the gateway
//! strips it from every inbound request precisely so a client cannot steer it: that header is
//! for a request intercept, which is trusted plugin code running inside the gateway. Widening
//! it to inbound traffic would hand the same authority to anything that can reach the port.
//! A separate header keeps that boundary intact and readable -- internal dispatch stays
//! intercept-only, and this one is client-supplied and therefore has to earn its trust.
//!
//! **What earns it.** The request must carry this invocation's transparent proxy credential,
//! which the launcher generates per run and gives only to the process it starts. That is a
//! real bound rather than a nominal one: a gateway is being told where to send credentialed
//! traffic, so the question that matters is whether the caller is the agent this invocation
//! launched, and the credential is the only thing that answers it. A standalone
//! `nemo-relay --bind` daemon issues no credential and so never honors this header; its
//! upstreams stay static, which is the conservative outcome for the shared case.

use axum::http::HeaderMap;
use reqwest::Url;

/// Header naming the provider endpoint the gateway should forward this request to.
///
/// Not prefixed `x-nemo-relay-internal-`: those are stripped from inbound requests by design,
/// and this one is meant to arrive from a client.
pub(crate) const UPSTREAM_BASE_URL_HEADER: &str = "x-nemo-relay-upstream-base-url";

/// The upstream this request asks for, when the request is entitled to ask.
///
/// Returns the base URL as written rather than a reserialized one, so an operator's `/v1` or
/// path prefix survives; composing it with the request path stays in `ProviderRoute`, which is
/// where the OpenAI `/v1` normalization already lives.
pub(crate) fn client_named_upstream_base(
    headers: &HeaderMap,
    invocation_authenticated: bool,
) -> Option<String> {
    // Checked before the header is even read: an unauthenticated request has no say in where
    // this gateway sends traffic, and treating the header as absent keeps the fallback to
    // configured routing on exactly one path.
    if !invocation_authenticated {
        return None;
    }

    let raw = headers.get(UPSTREAM_BASE_URL_HEADER)?.to_str().ok()?.trim();
    if raw.is_empty() {
        return None;
    }

    let url = Url::parse(raw).ok()?;
    // An absolute http(s) URL with a host, and nothing else. A relative URL would resolve
    // against the gateway itself, a non-http scheme reaches schemes reqwest treats very
    // differently (`file:`), and credentials in the authority would be forwarded to whatever
    // host follows them.
    if !matches!(url.scheme(), "http" | "https") {
        return None;
    }
    url.host_str()?;
    if !url.username().is_empty() || url.password().is_some() {
        return None;
    }

    Some(raw.to_string())
}

#[cfg(test)]
#[path = "../../../tests/coverage/agents/pi_alignment_tests.rs"]
mod tests;
