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

/// What a request asked for, and whether it may have it.
///
/// The three states are distinct on purpose. Collapsing "asked for something we refuse" into
/// "asked for nothing" is not a safe default here: the caller registered this gateway as its
/// provider and is sending a prompt and a provider credential intended for the endpoint it
/// named. Quietly forwarding that to the configured OpenAI or Anthropic upstream instead
/// would deliver both to the wrong company. A refusal has to be visible.
pub(crate) enum NamedUpstream {
    /// No opinion. Configured routing applies, which is the ordinary path for every other agent.
    Absent,
    /// Use this base.
    Named(String),
    /// A destination was named that may not be used. Refuse the request rather than reroute it.
    Rejected(&'static str),
}

/// The upstream this request asks for, when the request is entitled to ask.
///
/// Returns the base URL as written rather than a reserialized one, so the endpoint the caller
/// named is the endpoint it gets.
pub(crate) fn client_named_upstream_base(
    headers: &HeaderMap,
    invocation_authenticated: bool,
) -> NamedUpstream {
    // Checked before the header is even read: an unauthenticated request has no say in where
    // this gateway sends traffic. `Absent` rather than `Rejected` deliberately -- a standalone
    // daemon serves clients that never had a credential to present, and its documented
    // behaviour is that the header is inert there, not that such requests fail.
    if !invocation_authenticated {
        return NamedUpstream::Absent;
    }

    let Some(raw) = headers.get(UPSTREAM_BASE_URL_HEADER) else {
        return NamedUpstream::Absent;
    };
    let Ok(raw) = raw.to_str() else {
        return NamedUpstream::Rejected("upstream base URL header is not valid text");
    };
    let raw = raw.trim();
    if raw.is_empty() {
        return NamedUpstream::Rejected("upstream base URL header is empty");
    }

    let Ok(url) = Url::parse(raw) else {
        return NamedUpstream::Rejected("upstream base URL header is not an absolute URL");
    };
    // An absolute http(s) URL with a host, and nothing else. A relative URL would resolve
    // against the gateway itself, a non-http scheme reaches schemes reqwest treats very
    // differently (`file:`), and credentials in the authority would be forwarded to whatever
    // host follows them.
    if !matches!(url.scheme(), "http" | "https") {
        return NamedUpstream::Rejected("upstream base URL must use http or https");
    }
    if url.host_str().is_none() {
        return NamedUpstream::Rejected("upstream base URL has no host");
    }
    if !url.username().is_empty() || url.password().is_some() {
        return NamedUpstream::Rejected("upstream base URL must not carry credentials");
    }
    // A base is a prefix, and the request path is appended to it as text. Neither of these can
    // survive that: a query would end up before the path (`...?version=1/chat/completions`), and
    // a fragment would swallow it entirely, since everything after `#` is never sent to the
    // server. Both would route somewhere other than the endpoint that was named.
    if url.query().is_some() {
        return NamedUpstream::Rejected("upstream base URL must not carry a query string");
    }
    if url.fragment().is_some() {
        return NamedUpstream::Rejected("upstream base URL must not carry a fragment");
    }
    // Cleartext only where it cannot leave the machine.
    //
    // A named destination is forwarded the provider credential the request carried, so plain
    // `http` to a remote host would put that key on the wire in the clear. Loopback is the
    // exception every secure-context rule makes, and it is the case that matters here: a local
    // model server -- Ollama, vLLM, LM Studio -- is exactly the kind of provider this feature
    // exists to reach, and its traffic never reaches a network.
    if url.scheme() == "http" && !is_loopback(&url) {
        return NamedUpstream::Rejected(
            "upstream base URL must use https unless it is a loopback address",
        );
    }

    NamedUpstream::Named(raw.to_string())
}

/// Whether this host is one that cannot be reached from off the machine.
///
/// Deliberately narrow: the loopback IP ranges and the literal `localhost`. A name merely
/// ending in `.localhost` is reserved for loopback by RFC 6761 but still resolves through
/// whatever the host's resolver says, so it is left to the `https` requirement rather than
/// trusted here.
fn is_loopback(url: &Url) -> bool {
    let Some(host) = url.host_str() else {
        return false;
    };
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    // `host_str` brackets an IPv6 literal, which `IpAddr` does not parse.
    host.trim_start_matches('[')
        .trim_end_matches(']')
        .parse::<std::net::IpAddr>()
        .is_ok_and(|address| address.is_loopback())
}

#[cfg(test)]
#[path = "../../../tests/coverage/agents/pi_alignment_tests.rs"]
mod tests;
