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
//!
//! **Where it may point.** The credential proves who is asking, not that the destination is
//! harmless. The gateway may sit on a network the agent cannot reach directly, so a named
//! upstream is confined to public addresses and loopback: the remote providers in pi's
//! catalog, and local model servers. Private, link-local, and other special-use ranges are
//! refused, both for IP literals in the header and for every address a hostname resolves to
//! at connect time ([`NamedUpstreamResolver`]), so a name cannot be rebound past the check.

use std::future::Future;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

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
    client_named_upstream_base_with_proxies(
        headers,
        invocation_authenticated,
        environment_proxy_hosts(),
    )
}

fn client_named_upstream_base_with_proxies(
    headers: &HeaderMap,
    invocation_authenticated: bool,
    proxy_hosts: &[String],
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
    // An IP literal never reaches a resolver, so the address policy has to be applied here.
    if ip_literal(&url).is_some_and(|address| !is_permitted_destination(address)) {
        return NamedUpstream::Rejected("upstream base URL must name a public or loopback address");
    }
    // The resolver exempts proxy hosts, because resolving the proxy is how a proxied request
    // reaches the network. It cannot tell that apart from a request dialing the same host
    // directly -- which reqwest does when `NO_PROXY` excludes it, or when the proxy serves only
    // the other scheme -- so a proxy is never a destination that can be named.
    if url.host_str().is_some_and(|host| {
        proxy_hosts
            .iter()
            .any(|proxy| proxy.eq_ignore_ascii_case(host))
    }) {
        return NamedUpstream::Rejected("upstream base URL must not name a configured proxy");
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
    host.eq_ignore_ascii_case("localhost")
        || ip_literal(url).is_some_and(|address| address.is_loopback())
}

/// The host as an address, when it is written as one. `Url` normalizes IPv4 spellings such as
/// `0x7f.1` to dotted decimal, so this sees the address that will actually be dialed.
fn ip_literal(url: &Url) -> Option<IpAddr> {
    // `host_str` brackets an IPv6 literal, which `IpAddr` does not parse.
    url.host_str()?
        .trim_start_matches('[')
        .trim_end_matches(']')
        .parse()
        .ok()
}

/// Whether a named upstream may connect to this address: loopback, or public unicast.
///
/// Everything else is a network the gateway can reach but the caller was never granted:
/// private and shared ranges, link-local (including cloud metadata endpoints), multicast, and
/// the reserved and documentation blocks. IPv6 forms that embed an IPv4 address are judged by
/// that address, so `::ffff:10.0.0.1` is as private as `10.0.0.1`.
pub(crate) fn is_permitted_destination(address: IpAddr) -> bool {
    match address {
        IpAddr::V4(address) => is_permitted_ipv4(address),
        IpAddr::V6(address) => {
            if let Some(embedded) = address.to_ipv4_mapped() {
                return is_permitted_ipv4(embedded);
            }
            let segments = address.segments();
            // NAT64 well-known prefix (RFC 6052): a translator forwards to the embedded address.
            if segments[..6] == [0x64, 0xff9b, 0, 0, 0, 0] {
                let embedded = (u32::from(segments[6]) << 16) | u32::from(segments[7]);
                return is_permitted_ipv4(Ipv4Addr::from(embedded));
            }
            address.is_loopback()
                || !(address.is_unspecified()
                    || address.is_multicast()
                    || address.is_unique_local()
                    || address.is_unicast_link_local()
                    // Deprecated site-local, documentation, and IPv4-compatible ranges.
                    || segments[0] & 0xffc0 == 0xfec0
                    || segments[..2] == [0x2001, 0x0db8]
                    || segments[..6] == [0; 6])
        }
    }
}

fn is_permitted_ipv4(address: Ipv4Addr) -> bool {
    let [first, second, third, _] = address.octets();
    address.is_loopback()
        || !(first == 0
            || address.is_private()
            || address.is_link_local()
            || address.is_broadcast()
            || address.is_multicast()
            || address.is_documentation()
            // Shared address space (100.64.0.0/10), IETF protocol assignments, benchmarking,
            // and the reserved class E block.
            || (first == 100 && second & 0xc0 == 64)
            || (first == 192 && second == 0 && third == 0)
            || (first == 198 && second & 0xfe == 18)
            || first >= 240)
}

/// Hosts of the proxies reqwest picks up from the environment, read once like reqwest does when
/// the gateway's clients are built.
fn environment_proxy_hosts() -> &'static [String] {
    static HOSTS: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    HOSTS.get_or_init(|| {
        [
            "HTTPS_PROXY",
            "https_proxy",
            "HTTP_PROXY",
            "http_proxy",
            "ALL_PROXY",
            "all_proxy",
        ]
        .into_iter()
        .filter_map(|name| std::env::var(name).ok())
        .filter_map(|value| Url::parse(value.trim()).ok())
        .filter_map(|url| url.host_str().map(str::to_ascii_lowercase))
        .collect()
    })
}

/// Resolves hosts for the clients that dispatch to named upstreams, keeping only addresses
/// [`is_permitted_destination`] allows.
///
/// Filtering at resolution is what makes the policy hold for hostnames: the address checked is
/// the address connected to, so a name that resolves somewhere private -- or is rebound there
/// after the header was validated -- has nothing left to connect to.
///
/// `localhost` is answered without DNS, as RFC 6761 recommends, so a resolver that maps it
/// elsewhere cannot turn the plain-`http` loopback allowance into cleartext off the machine.
#[derive(Clone, Default)]
pub(crate) struct NamedUpstreamResolver {
    /// Operator-configured proxy hosts, resolved without the policy.
    ///
    /// When reqwest routes through a proxy it resolves the proxy rather than the destination,
    /// and a corporate proxy commonly lives at a private address. The proxy then resolves and
    /// reaches the destination itself, so it, not this resolver, enforces where traffic goes.
    proxy_hosts: Arc<[String]>,
}

impl NamedUpstreamResolver {
    /// A resolver that exempts the proxies reqwest picks up from the environment.
    pub(crate) fn with_environment_proxies() -> Self {
        Self {
            proxy_hosts: environment_proxy_hosts().into(),
        }
    }

    async fn permitted_addresses(self, host: String) -> std::io::Result<Vec<SocketAddr>> {
        if host.eq_ignore_ascii_case("localhost") {
            return Ok(vec![
                SocketAddr::from((Ipv4Addr::LOCALHOST, 0)),
                SocketAddr::from((Ipv6Addr::LOCALHOST, 0)),
            ]);
        }
        let resolved = tokio::net::lookup_host((host.as_str(), 0)).await?;
        if self
            .proxy_hosts
            .iter()
            .any(|proxy| proxy.eq_ignore_ascii_case(&host))
        {
            return Ok(resolved.collect());
        }
        let permitted = resolved
            .filter(|address| is_permitted_destination(address.ip()))
            .collect::<Vec<_>>();
        if permitted.is_empty() {
            log::warn!(
                target: "nemo_relay.gateway",
                event = "named_upstream_rejected",
                reason = "resolves to no public or loopback address";
                "Rejected a client-named upstream at connect time"
            );
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                format!("named upstream {host} resolves to no public or loopback address"),
            ));
        }
        Ok(permitted)
    }
}

impl reqwest::dns::Resolve for NamedUpstreamResolver {
    fn resolve(&self, name: reqwest::dns::Name) -> reqwest::dns::Resolving {
        let resolution = self.clone().permitted_addresses(name.as_str().to_owned());
        Box::pin(async move { Ok(Box::new(resolution.await?.into_iter()) as reqwest::dns::Addrs) })
    }
}

impl tower_service::Service<hyper_util::client::legacy::connect::dns::Name>
    for NamedUpstreamResolver
{
    type Response = std::vec::IntoIter<SocketAddr>;
    type Error = std::io::Error;
    type Future = Pin<Box<dyn Future<Output = std::io::Result<Self::Response>> + Send>>;

    fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, name: hyper_util::client::legacy::connect::dns::Name) -> Self::Future {
        let resolution = self.clone().permitted_addresses(name.as_str().to_owned());
        Box::pin(async move { resolution.await.map(Vec::into_iter) })
    }
}

#[cfg(test)]
#[path = "../../../tests/coverage/agents/pi_alignment_tests.rs"]
mod tests;
