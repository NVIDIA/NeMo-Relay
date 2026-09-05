// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lossless, pull-driven HTTP transport used between daemon data-plane hops.
//!
//! The adapters in this module deliberately operate on [`Frame`] values rather than on decoded
//! payloads. Data and trailer frames therefore stay under Hyper's normal demand-driven
//! backpressure, and dropping the downstream body cancels the upstream body without a forwarding
//! task or intermediate queue.

use std::error::Error;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use bytes::Bytes;
use http::header::{CONNECTION, HOST, HeaderName, HeaderValue, TE, UPGRADE};
use http::{HeaderMap, Method, Request, Response, StatusCode, Uri, Version};
use http_body_util::BodyExt;
use http_body_util::combinators::UnsyncBoxBody;
use hyper::body::Body;
use hyper_rustls::{HttpsConnector, HttpsConnectorBuilder};
use hyper_util::client::legacy::Client;
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::rt::{TokioExecutor, TokioTimer};
use thiserror::Error;

/// Error type shared by transport bodies after their concrete body implementation is erased.
pub(crate) type BoxError = Box<dyn Error + Send + Sync>;

/// A pull-driven body that preserves both data and trailer frames.
pub(crate) type RelayBody = UnsyncBoxBody<Bytes, BoxError>;

/// A pooled client supporting cleartext HTTP and rustls-backed HTTPS with HTTP/1.1 and HTTP/2.
pub(crate) type PooledClient = Client<HttpsConnector<HttpConnector>, RelayBody>;

/// A pooled cleartext client. The normal builder uses HTTP/1.1; the h2c builder uses HTTP/2 prior
/// knowledge.
#[cfg(test)]
pub(crate) type PooledHttpClient = Client<HttpConnector, RelayBody>;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const MAX_IDLE_CONNECTIONS_PER_HOST: usize = 256;

#[derive(Debug, Error)]
pub(crate) enum TransportError {
    #[error("CONNECT and HTTP Upgrade are not supported by the daemon data plane")]
    UnsupportedTunnel,
    #[error("forward destination must contain an HTTP or HTTPS scheme and an authority")]
    InvalidDestination,
    #[error("invalid Connection header value")]
    InvalidConnectionHeader,
    #[error("forward destination authority is not a valid Host header")]
    InvalidHost(#[source] http::header::InvalidHeaderValue),
    #[error("failed to load native TLS trust roots")]
    NativeRoots(#[source] std::io::Error),
}

/// Erases a body's implementation and error while retaining its pull-based [`Body::poll_frame`]
/// behavior. This function does not spawn a forwarding task, decode frames, or queue bytes.
pub(crate) fn box_body<B>(body: B) -> RelayBody
where
    B: Body<Data = Bytes> + Send + 'static,
    B::Error: Into<BoxError>,
{
    body.map_err(Into::into).boxed_unsync()
}

/// Keeps request accounting or another lifetime guard alive until a body completes or is dropped.
/// The body is still polled directly; no forwarding task or queue is introduced.
pub(crate) fn hold_body<B, H>(body: B, hold: H) -> RelayBody
where
    B: Body<Data = Bytes> + Send + Unpin + 'static,
    B::Error: Into<BoxError>,
    H: Send + Unpin + 'static,
{
    box_body(HeldBody {
        body,
        hold: Some(hold),
    })
}

struct HeldBody<B, H> {
    body: B,
    hold: Option<H>,
}

impl<B, H> Body for HeldBody<B, H>
where
    B: Body<Data = Bytes> + Unpin,
    B::Error: Into<BoxError>,
    H: Unpin,
{
    type Data = Bytes;
    type Error = BoxError;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Option<Result<hyper::body::Frame<Self::Data>, Self::Error>>> {
        let frame = Pin::new(&mut self.body).poll_frame(context);
        if matches!(frame, Poll::Ready(None)) {
            self.hold.take();
        }
        frame.map(|frame| frame.map(|result| result.map_err(Into::into)))
    }

    fn is_end_stream(&self) -> bool {
        self.body.is_end_stream()
    }

    fn size_hint(&self) -> hyper::body::SizeHint {
        self.body.size_hint()
    }
}

/// Removes fields scoped to one HTTP connection.
///
/// `Trailer` is intentionally not in the fixed hop-by-hop list. It declares the fields carried by
/// a later trailer frame and remains valid across a framing-preserving intermediary. It is removed
/// only when an incoming `Connection` field explicitly nominates it.
pub(crate) fn strip_hop_by_hop_headers(headers: &mut HeaderMap) -> Result<(), TransportError> {
    let nominated = connection_nominated_headers(headers)?;

    for name in nominated {
        headers.remove(name);
    }
    for name in [
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "transfer-encoding",
        "upgrade",
    ] {
        headers.remove(name);
    }

    Ok(())
}

/// Rewrites a request head for one proxy hop while transferring ownership of its body unchanged.
///
/// `destination` is the complete URI selected by the router, including its path and query. Names
/// in `additional_strip` are routing or authentication fields consumed by the current hop.
pub(crate) fn prepare_forward_request<B>(
    mut request: Request<B>,
    destination: Uri,
    additional_strip: &[HeaderName],
) -> Result<Request<B>, TransportError> {
    if request.method() == Method::CONNECT || request.headers().contains_key(UPGRADE) {
        return Err(TransportError::UnsupportedTunnel);
    }

    let scheme = destination
        .scheme_str()
        .filter(|scheme| matches!(*scheme, "http" | "https"));
    let authority = destination.authority().cloned();
    if scheme.is_none() || authority.is_none() {
        return Err(TransportError::InvalidDestination);
    }
    let authority = authority.expect("authority was checked above");
    let host = HeaderValue::from_str(authority.as_str()).map_err(TransportError::InvalidHost)?;

    strip_hop_by_hop_headers(request.headers_mut())?;
    for name in additional_strip {
        request.headers_mut().remove(name);
    }
    // `TE` is scoped to one connection, but Relay accepts and relays trailer frames. Advertise
    // that capability independently on every upstream hop after consuming the caller's value.
    request
        .headers_mut()
        .insert(TE, HeaderValue::from_static("trailers"));
    request.headers_mut().insert(HOST, host);
    *request.uri_mut() = destination;
    // The protocol version belongs to the connection on which this request arrived. It is not a
    // requirement for the next proxy hop: leaving HTTP/2 here makes Hyper reject an H2 ingress
    // request when the selected upstream only speaks HTTP/1.1. HTTP/1.1 is the neutral request
    // value; ALPN or an H2-only client still selects HTTP/2 independently.
    *request.version_mut() = Version::HTTP_11;

    Ok(request)
}

/// Filters a response head for one proxy hop while transferring ownership of its body unchanged.
pub(crate) fn prepare_forward_response<B>(
    mut response: Response<B>,
    additional_strip: &[HeaderName],
) -> Result<Response<B>, TransportError> {
    if response.status() == StatusCode::SWITCHING_PROTOCOLS
        || response.headers().contains_key(UPGRADE)
    {
        return Err(TransportError::UnsupportedTunnel);
    }

    strip_hop_by_hop_headers(response.headers_mut())?;
    for name in additional_strip {
        response.headers_mut().remove(name);
    }
    Ok(response)
}

/// Builds a pooled HTTP(S) client. Callers should construct this once per process and clone its
/// lightweight handle rather than building one per request.
pub(crate) fn pooled_client() -> Result<PooledClient, TransportError> {
    // The workspace enables more than one rustls backend through unrelated integrations. Select
    // Relay's direct `ring` dependency before rustls tries to infer a process-wide provider.
    let _ = rustls::crypto::ring::default_provider().install_default();
    let mut http = HttpConnector::new();
    http.enforce_http(false);
    http.set_nodelay(true);
    http.set_connect_timeout(Some(CONNECT_TIMEOUT));

    let connector = HttpsConnectorBuilder::new()
        .with_native_roots()
        .map_err(TransportError::NativeRoots)?
        .https_or_http()
        .enable_http1()
        .enable_http2()
        .wrap_connector(http);

    let builder = pooled_builder();
    Ok(builder.build(connector))
}

/// Builds a pooled cleartext HTTP/1.1 client with persistent connections and `TCP_NODELAY`.
#[cfg(test)]
pub(crate) fn pooled_http_client() -> PooledHttpClient {
    let mut connector = HttpConnector::new();
    connector.enforce_http(true);
    connector.set_nodelay(true);
    connector.set_connect_timeout(Some(CONNECT_TIMEOUT));
    let builder = pooled_builder();
    builder.build(connector)
}

/// Builds a pooled cleartext HTTP/2 client using prior knowledge rather than an Upgrade exchange.
#[cfg(test)]
pub(crate) fn pooled_h2c_client() -> PooledHttpClient {
    let mut connector = HttpConnector::new();
    connector.enforce_http(true);
    connector.set_nodelay(true);
    connector.set_connect_timeout(Some(CONNECT_TIMEOUT));

    let mut builder = pooled_builder();
    builder.http2_only(true);
    builder.build(connector)
}

/// Builds the same HTTP(S)-capable client type used by daemon and worker state, while forcing
/// cleartext HTTP/2 prior knowledge for deterministic end-to-end transport tests.
pub(crate) fn pooled_worker_h2c_client() -> Result<PooledClient, TransportError> {
    let _ = rustls::crypto::ring::default_provider().install_default();
    let mut http = HttpConnector::new();
    http.enforce_http(false);
    http.set_nodelay(true);
    http.set_connect_timeout(Some(CONNECT_TIMEOUT));

    let connector = HttpsConnectorBuilder::new()
        .with_native_roots()
        .map_err(TransportError::NativeRoots)?
        .https_or_http()
        .enable_http1()
        .enable_http2()
        .wrap_connector(http);
    let mut builder = pooled_builder();
    builder.http2_only(true);
    Ok(builder.build(connector))
}

fn pooled_builder() -> hyper_util::client::legacy::Builder {
    let mut builder = Client::builder(TokioExecutor::new());
    builder.timer(TokioTimer::new());
    builder.pool_idle_timeout(Duration::from_secs(120));
    builder.pool_max_idle_per_host(MAX_IDLE_CONNECTIONS_PER_HOST);
    builder.http2_keep_alive_interval(Duration::from_secs(15));
    builder.http2_keep_alive_timeout(Duration::from_secs(5));
    builder.http2_keep_alive_while_idle(true);
    builder
}

fn connection_nominated_headers(headers: &HeaderMap) -> Result<Vec<HeaderName>, TransportError> {
    let mut nominated = Vec::new();
    for value in headers.get_all(CONNECTION) {
        let value = value
            .to_str()
            .map_err(|_| TransportError::InvalidConnectionHeader)?;
        for token in value
            .split(',')
            .map(str::trim)
            .filter(|token| !token.is_empty())
        {
            let name = HeaderName::from_bytes(token.as_bytes())
                .map_err(|_| TransportError::InvalidConnectionHeader)?;
            nominated.push(name);
        }
    }
    Ok(nominated)
}

#[cfg(test)]
#[path = "../../../tests/coverage/daemon/transport_tests.rs"]
mod tests;
