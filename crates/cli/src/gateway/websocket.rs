// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unmanaged RFC 6455 tunnels. Frames remain opaque, including close and control frames.

use axum::body::Body;
use axum::http::{
    HeaderMap, HeaderName, HeaderValue, Method, Request, Response, StatusCode, header,
};
use axum::response::IntoResponse;
use base64::Engine;
use hyper_util::rt::TokioIo;

use crate::configuration::GatewayConfig;
use crate::daemon::common::transport::{
    PooledClient, box_body, prepare_forward_request, prepare_forward_response,
    strip_hop_by_hop_headers,
};

pub(crate) fn is_websocket(request: &Request<Body>) -> bool {
    request.method() == Method::GET && token(request.headers(), header::UPGRADE, "websocket")
}

pub(crate) fn supported_path(path: &str) -> bool {
    matches!(path, "/live" | "/v1/live" | "/realtime" | "/v1/realtime") || live_sideband_path(path)
}

/// Codex Live attaches by call ID; public Live uses a session attach path.
pub(crate) fn live_sideband_path(path: &str) -> bool {
    let Some(suffix) = path
        .strip_prefix("/v1/live/")
        .or_else(|| path.strip_prefix("/live/"))
    else {
        return false;
    };
    let id = if let Some(suffix) = suffix.strip_prefix("sessions/") {
        let Some(id) = suffix.strip_suffix("/attach") else {
            return false;
        };
        id
    } else {
        suffix
    };
    !id.is_empty() && !id.contains('/') && !matches!(id, "." | ".." | "sessions")
}

fn token(headers: &HeaderMap, name: HeaderName, expected: &str) -> bool {
    headers.get_all(name).iter().any(|value| {
        value.to_str().is_ok_and(|value| {
            value
                .split(',')
                .any(|value| value.trim().eq_ignore_ascii_case(expected))
        })
    })
}

pub(super) fn strip_private_headers(headers: &mut HeaderMap, keep_named_upstream: bool) {
    let private = headers
        .keys()
        .filter(|name| {
            name.as_str().starts_with("x-nemo-relay-")
                && !(keep_named_upstream
                    && name.as_str() == crate::agents::pi::alignment::UPSTREAM_BASE_URL_HEADER)
        })
        .cloned()
        .collect::<Vec<_>>();
    for name in private {
        headers.remove(name);
    }
}

/// Establishes the upstream handshake before acknowledging the downstream upgrade. The guard
/// owns the selected worker/admission for the entire tunnel, rather than the empty 101 body.
pub(crate) async fn forward<H: Send + 'static>(
    client: &PooledClient,
    mut request: Request<Body>,
    destination: &str,
    authentication: Option<(HeaderName, String)>,
    hold: H,
    config: &GatewayConfig,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> Response<Body> {
    let key = request.headers().get(header::SEC_WEBSOCKET_KEY).cloned();
    if !is_websocket(&request)
        || !token(request.headers(), header::CONNECTION, "upgrade")
        || request
            .headers()
            .get(header::SEC_WEBSOCKET_VERSION)
            .is_none_or(|v| v != "13")
        || key.as_ref().is_none_or(|key| {
            base64::engine::general_purpose::STANDARD
                .decode(key.as_bytes())
                .map_or(true, |bytes| bytes.len() != 16)
        })
    {
        return StatusCode::BAD_REQUEST.into_response();
    }
    let key = key.expect("validated key");
    let protocols = request
        .headers()
        .get(header::SEC_WEBSOCKET_PROTOCOL)
        .cloned();
    let downstream = hyper::upgrade::on(&mut request);
    request.headers_mut().remove(header::UPGRADE);
    // Routing credentials are consumed at each hop; only the next hop's worker credential may
    // be reintroduced below. Never forward bootstrap, capability, or operational private fields.
    // A worker hop retains the caller's named destination until the worker resolves it.
    // The provider hop consumes it along with every other Relay-private header.
    strip_private_headers(request.headers_mut(), authentication.is_some());
    let Ok(destination) = destination.parse() else {
        return StatusCode::BAD_GATEWAY.into_response();
    };
    let mut request = match prepare_forward_request(request, destination, &[]) {
        Ok(request) => request,
        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
    };
    for name in [header::CONTENT_LENGTH, header::CONTENT_TYPE, header::TE] {
        request.headers_mut().remove(name);
    }
    request
        .headers_mut()
        .insert(header::CONNECTION, HeaderValue::from_static("Upgrade"));
    request
        .headers_mut()
        .insert(header::UPGRADE, HeaderValue::from_static("websocket"));
    if let Some((name, value)) = authentication {
        let Ok(value) = value.parse() else {
            return StatusCode::INTERNAL_SERVER_ERROR.into_response();
        };
        request.headers_mut().insert(name, value);
    }
    let (parts, _) = request.into_parts();
    let request = Request::from_parts(parts, box_body(Body::empty()));
    let mut shutdown = Box::pin(shutdown);
    let response = tokio::select! {
        _ = &mut shutdown => return StatusCode::SERVICE_UNAVAILABLE.into_response(),
        response = config.wait_for_response(client.request(request)) => response,
    };
    let mut response = match response {
        Ok(Ok(response)) => response,
        Ok(Err(_)) => return StatusCode::BAD_GATEWAY.into_response(),
        Err(_) => return StatusCode::GATEWAY_TIMEOUT.into_response(),
    };
    if response.status() != StatusCode::SWITCHING_PROTOCOLS {
        return match prepare_forward_response(response, &[]) {
            Ok(mut response) => {
                strip_private_headers(response.headers_mut(), false);
                response.map(Body::new)
            }
            Err(_) => StatusCode::BAD_GATEWAY.into_response(),
        };
    }
    let expected = tokio_tungstenite::tungstenite::handshake::derive_accept_key(key.as_bytes());
    if !token(response.headers(), header::UPGRADE, "websocket")
        || !token(response.headers(), header::CONNECTION, "upgrade")
        || response
            .headers()
            .get(header::SEC_WEBSOCKET_ACCEPT)
            .is_none_or(|v| v != expected.as_str())
        || response
            .headers()
            .get(header::SEC_WEBSOCKET_PROTOCOL)
            .is_some_and(|selected| {
                selected.to_str().map_or(true, |selected| {
                    protocols
                        .as_ref()
                        .and_then(|p| p.to_str().ok())
                        .is_none_or(|offered| !offered.split(',').any(|p| p.trim() == selected))
                })
            })
    {
        return StatusCode::BAD_GATEWAY.into_response();
    }
    let upstream = hyper::upgrade::on(&mut response);
    let (mut parts, _) = response.into_parts();
    if strip_hop_by_hop_headers(&mut parts.headers).is_err() {
        return StatusCode::BAD_GATEWAY.into_response();
    }
    strip_private_headers(&mut parts.headers, false);
    parts
        .headers
        .insert(header::CONNECTION, HeaderValue::from_static("Upgrade"));
    parts
        .headers
        .insert(header::UPGRADE, HeaderValue::from_static("websocket"));
    tokio::spawn(async move {
        let _hold = hold;
        // Bound upgrade completion too: a client that abandons a successful handshake must not
        // retain a worker guard indefinitely.
        let sockets = tokio::select! {
            _ = &mut shutdown => return,
            sockets = tokio::time::timeout(std::time::Duration::from_secs(10), async {
                tokio::try_join!(downstream, upstream)
            }) => sockets,
        };
        let Ok(Ok((downstream, upstream))) = sockets else {
            return;
        };
        let (mut down_read, mut down_write) = tokio::io::split(TokioIo::new(downstream));
        let (mut up_read, mut up_write) = tokio::io::split(TokioIo::new(upstream));
        // copy uses bounded buffers and awaits writes. EOF/error on either peer drops both
        // connections immediately; normal WebSocket close frames are forwarded before EOF.
        tokio::select! {
            _ = &mut shutdown => {},
            _ = tokio::io::copy(&mut down_read, &mut up_write) => {},
            _ = tokio::io::copy(&mut up_read, &mut down_write) => {},
        }
    });
    Response::from_parts(parts, Body::empty())
}

#[cfg(test)]
#[path = "../../tests/coverage/shared/websocket_tests.rs"]
pub(crate) mod tests;
