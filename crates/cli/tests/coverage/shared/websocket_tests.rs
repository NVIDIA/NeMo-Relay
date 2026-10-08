// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use axum::http::{HeaderMap, Uri};
use axum::response::IntoResponse;
use axum::{
    Router,
    extract::ws::{Message as AxumMessage, WebSocketUpgrade},
    routing::get,
};
use futures_util::{SinkExt, StreamExt};
use std::collections::HashSet;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Duration;
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::{Message, client::IntoClientRequest};

pub(crate) async fn serve(app: Router) -> (String, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (url, task)
}

pub(crate) async fn provider() -> (String, tokio::task::JoinHandle<()>, Arc<AtomicUsize>) {
    let active = Arc::new(AtomicUsize::new(0));
    let sessions = Arc::new(Mutex::new(HashSet::<String>::new()));
    let app = sideband_setup(Arc::clone(&sessions)).merge(unmanaged_http_routes());
    let websocket = get({
        let active = active.clone();
        move |ws: WebSocketUpgrade, headers: HeaderMap, uri: Uri| {
            let active = active.clone();
            let sessions = Arc::clone(&sessions);
            async move {
                assert_eq!(headers[header::AUTHORIZATION], "Bearer caller-voice-key");
                assert!(
                    headers
                        .keys()
                        .all(|h| !h.as_str().starts_with("x-nemo-relay-"))
                );
                if uri.query() == Some("fail=1") {
                    return (
                        StatusCode::UNAUTHORIZED,
                        [("retry-after", "7")],
                        "upstream denied",
                    )
                        .into_response();
                }
                if let Some(id) = uri.query().and_then(|q| q.strip_prefix("call_id=")) {
                    assert_eq!(uri.path(), "/v1/realtime");
                    if !sessions.lock().unwrap().contains(id) {
                        return StatusCode::NOT_FOUND.into_response();
                    }
                } else if let Some(id) = uri.path().strip_prefix("/v1/live/sessions/") {
                    assert_eq!(uri.query(), Some("graceful_close=true"));
                    if !sessions
                        .lock()
                        .unwrap()
                        .contains(id.strip_suffix("/attach").unwrap())
                    {
                        return StatusCode::NOT_FOUND.into_response();
                    }
                } else if let Some(id) = uri.path().strip_prefix("/v1/live/") {
                    assert_eq!(uri.query(), Some("intent=quicksilver&architecture=avas"));
                    if !sessions.lock().unwrap().contains(id) {
                        return StatusCode::NOT_FOUND.into_response();
                    }
                } else {
                    assert_eq!(uri.query(), Some("model=voice"));
                    assert!(matches!(uri.path(), "/v1/live" | "/v1/realtime"));
                }
                ws.protocols(["voice"])
                    .on_upgrade(move |mut socket| async move {
                        active.fetch_add(1, Ordering::SeqCst);
                        struct Active(Arc<AtomicUsize>);
                        impl Drop for Active {
                            fn drop(&mut self) {
                                self.0.fetch_sub(1, Ordering::SeqCst);
                            }
                        }
                        let _active = Active(active);
                        socket
                            .send(AxumMessage::Binary(vec![0, 1, 255].into()))
                            .await
                            .unwrap();
                        while let Some(Ok(message)) = socket.next().await {
                            match message {
                                AxumMessage::Close(_) => {
                                    let _ = socket.flush().await;
                                    break;
                                }
                                AxumMessage::Ping(_) | AxumMessage::Pong(_) => {
                                    let _ = socket.flush().await;
                                }
                                message => {
                                    if socket.send(message).await.is_err() {
                                        break;
                                    }
                                }
                            }
                        }
                    })
                    .into_response()
            }
        }
    });
    let app = app
        .route("/v1/live", websocket.clone())
        .route("/{*path}", websocket);
    let (url, task) = serve(app).await;
    (url, task, active)
}

fn unmanaged_http_routes() -> Router {
    async fn echo(request: Request<Body>) -> Response<Body> {
        assert_eq!(
            request.headers()[header::AUTHORIZATION],
            "Bearer caller-voice-key"
        );
        assert!(
            request
                .headers()
                .keys()
                .all(|h| !h.as_str().starts_with("x-nemo-relay-"))
        );
        assert_eq!(request.uri().query(), Some("trace=opaque%2Fquery"));
        let path = request.uri().path().to_owned();
        let content_type = request.headers()[header::CONTENT_TYPE].clone();
        let body = axum::body::to_bytes(request.into_body(), usize::MAX)
            .await
            .unwrap();
        Response::builder()
            .status(StatusCode::ACCEPTED)
            .header("x-upstream-path", path)
            .header(header::CONTENT_TYPE, content_type)
            .body(Body::from(body))
            .unwrap()
    }
    Router::new()
        .route("/v1/images/edits", axum::routing::post(echo))
        .route("/v1/memories/trace_summarize", axum::routing::post(echo))
        .route("/v1/alpha/search", axum::routing::post(echo))
}

pub(crate) async fn exercise_unmanaged_http(origin: &str, token: Option<&str>, prefixes: &[&str]) {
    for prefix in prefixes {
        for path in ["images/edits", "memories/trace_summarize", "alpha/search"] {
            for (content_type, body) in [
                ("application/json", br#"{ "model": "test", "input": [], "opaque": true }"#.as_slice()),
                ("multipart/form-data; boundary=relay", b"--relay\r\nContent-Disposition: form-data; name=\"image\"; filename=\"test.bin\"\r\n\r\n\x00\xff\r\n--relay--\r\n".as_slice()),
            ] {
                if path != "images/edits" && content_type != "application/json" { continue; }
                let mut request = reqwest::Client::new()
                    .post(format!("{origin}{prefix}/{path}?trace=opaque%2Fquery"))
                    .bearer_auth("caller-voice-key").header("content-type", content_type)
                    .header("x-nemo-relay-private", "must-not-leak").body(body.to_vec());
                if let Some(token) = token { request = request.header("x-nemo-relay-client-token", token); }
                let response = request.send().await.unwrap();
                assert_eq!(response.status(), StatusCode::ACCEPTED);
                assert_eq!(response.headers()["x-upstream-path"], format!("/v1/{path}"));
                assert_eq!(response.headers()["content-type"], content_type);
                assert_eq!(response.bytes().await.unwrap().as_ref(), body);
            }
        }
    }
}

fn sideband_setup(sessions: Arc<Mutex<HashSet<String>>>) -> Router {
    let route = axum::routing::post(move |headers: HeaderMap, uri: Uri, body: String| {
        let sessions = Arc::clone(&sessions);
        async move {
            assert_eq!(headers[header::AUTHORIZATION], "Bearer caller-voice-key");
            assert!(
                headers
                    .keys()
                    .all(|h| !h.as_str().starts_with("x-nemo-relay-"))
            );
            let live = uri.path() == "/v1/live/sessions";
            assert!(live || matches!(uri.path(), "/v1/realtime/calls" | "/v1/live"));
            assert!(body.contains("offer"));
            if uri.query() == Some("intent=quicksilver&architecture=avas") {
                assert_eq!(uri.path(), "/v1/realtime/calls");
                assert_eq!(headers[header::CONTENT_TYPE], "application/json");
                assert_eq!(body, r#"{"sdp":"offer","session":{"type":"realtime"}}"#);
            }
            let mut sessions = sessions.lock().unwrap();
            let id = format!("{}_{}", if live { "live" } else { "rtc" }, sessions.len());
            sessions.insert(id.clone());
            if live {
                (
                    StatusCode::CREATED,
                    axum::Json(serde_json::json!({
                        "session": {"id": id}, "transport": {"type": "webrtc", "sdp": "answer"}
                    })),
                )
                    .into_response()
            } else {
                (
                    StatusCode::CREATED,
                    [("location", format!("{}/{id}", uri.path()))],
                    "answer",
                )
                    .into_response()
            }
        }
    });
    Router::new()
        .route("/v1/live/sessions", route.clone())
        .route("/v1/realtime/calls", route.clone())
        .route("/v1/live", route)
}

pub(crate) async fn exercise_sideband(
    origin: &str,
    token: Option<&str>,
    active: &AtomicUsize,
    prefixes: &[&str],
) {
    for prefix in prefixes {
        for mode in [0, 1, 2, 3] {
            let backend_call = mode == 3;
            if backend_call != (*prefix == "/backend-api/codex") {
                continue;
            }
            let sideband_prefix = if backend_call { "" } else { prefix };
            let live = mode == 1;
            let path = match mode {
                1 => "live/sessions",
                2 => "live",
                _ => "realtime/calls",
            };
            let mut request = reqwest::Client::new()
                .post(format!(
                    "{origin}{prefix}/{path}{}",
                    if backend_call {
                        "?intent=quicksilver&architecture=avas"
                    } else {
                        ""
                    }
                ))
                .bearer_auth("caller-voice-key")
                .header(
                    "content-type",
                    if live || backend_call {
                        "application/json"
                    } else {
                        "application/sdp"
                    },
                )
                .body(if live {
                    r#"{"session":{"model":"voice"},"transport":{"type":"webrtc","sdp":"offer"}}"#
                } else if backend_call {
                    r#"{"sdp":"offer","session":{"type":"realtime"}}"#
                } else {
                    "offer"
                });
            if let Some(token) = token {
                request = request.header("x-nemo-relay-client-token", token);
            }
            let response = request.send().await.unwrap();
            assert_eq!(response.status(), StatusCode::CREATED);
            let sideband = sideband_path(response, sideband_prefix, mode).await;
            let mut request = format!("{}{sideband}", origin.replacen("http", "ws", 1))
                .into_client_request()
                .unwrap();
            request.headers_mut().insert(
                header::AUTHORIZATION,
                HeaderValue::from_static("Bearer caller-voice-key"),
            );
            request.headers_mut().insert(
                "x-nemo-relay-private",
                HeaderValue::from_static("must-not-leak"),
            );
            if let Some(token) = token {
                request
                    .headers_mut()
                    .insert("x-nemo-relay-client-token", token.parse().unwrap());
            }
            let (mut socket, _) = tokio_tungstenite::connect_async(request).await.unwrap();
            exchange_frames(&mut socket).await;
            drop(socket);
            wait_disconnected(active).await;
        }
    }
}

async fn sideband_path(response: reqwest::Response, prefix: &str, mode: u8) -> String {
    if mode == 1 {
        let body: serde_json::Value = response.json().await.unwrap();
        return format!(
            "{prefix}/live/sessions/{}/attach?graceful_close=true",
            body["session"]["id"].as_str().unwrap()
        );
    }
    let id = response.headers()["location"]
        .to_str()
        .unwrap()
        .rsplit('/')
        .next()
        .unwrap();
    if mode == 2 || mode == 3 {
        format!("{prefix}/live/{id}?intent=quicksilver&architecture=avas")
    } else {
        format!("{prefix}/realtime?call_id={id}")
    }
}

pub(crate) async fn exercise(origin: &str, token: Option<&str>, active: &AtomicUsize) {
    for path in ["/live", "/v1/live", "/realtime", "/v1/realtime"] {
        let url = format!("{}{path}?model=voice", origin.replacen("http", "ws", 1));
        let mut request = url.into_client_request().unwrap();
        request.headers_mut().insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer caller-voice-key"),
        );
        request.headers_mut().insert(
            header::SEC_WEBSOCKET_PROTOCOL,
            HeaderValue::from_static("voice"),
        );
        request.headers_mut().insert(
            "x-nemo-relay-private",
            HeaderValue::from_static("must-not-leak"),
        );
        if let Some(token) = token {
            request
                .headers_mut()
                .insert("x-nemo-relay-client-token", token.parse().unwrap());
        }
        let (mut socket, response) = tokio_tungstenite::connect_async(request).await.unwrap();
        assert_eq!(response.headers()[header::SEC_WEBSOCKET_PROTOCOL], "voice");
        exchange_frames(&mut socket).await;
        drop(socket);
        wait_disconnected(active).await;
    }
    exercise_unmanaged_http(origin, token, &["", "/v1"]).await;
    exercise_sideband(origin, token, active, &["", "/v1", "/backend-api/codex"]).await;
    sideband_rejection(origin, token).await;
    abrupt_disconnect(origin, token, active).await;
    handshake_rejection(origin, token).await;
}

async fn sideband_rejection(origin: &str, token: Option<&str>) {
    for path in [
        "/v1/realtime?call_id=rtc_missing",
        "/v1/live/rtc_missing?intent=quicksilver&architecture=avas",
        "/v1/live/sessions/live_missing/attach?graceful_close=true",
    ] {
        let mut request = format!("{}{path}", origin.replacen("http", "ws", 1))
            .into_client_request()
            .unwrap();
        request.headers_mut().insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer caller-voice-key"),
        );
        if let Some(token) = token {
            request
                .headers_mut()
                .insert("x-nemo-relay-client-token", token.parse().unwrap());
        }
        let error = tokio_tungstenite::connect_async(request).await.unwrap_err();
        let tokio_tungstenite::tungstenite::Error::Http(response) = error else {
            panic!("{error}");
        };
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }
}

async fn exchange_frames(
    socket: &mut tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
) {
    assert_eq!(
        socket.next().await.unwrap().unwrap(),
        Message::Binary(vec![0, 1, 255].into())
    );
    for message in [
        Message::Text("hello".into()),
        Message::Binary(vec![255; 128 * 1024].into()),
    ] {
        socket.send(message.clone()).await.unwrap();
        assert_eq!(socket.next().await.unwrap().unwrap(), message);
    }
    socket.send(Message::Ping(vec![7].into())).await.unwrap();
    assert_eq!(
        socket.next().await.unwrap().unwrap(),
        Message::Pong(vec![7].into())
    );
    let close = tokio_tungstenite::tungstenite::protocol::CloseFrame {
        code: tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode::Normal,
        reason: "done".into(),
    };
    socket.close(Some(close.clone())).await.unwrap();
    assert_eq!(
        socket.next().await.unwrap().unwrap(),
        Message::Close(Some(close))
    );
}

async fn wait_disconnected(active: &AtomicUsize) {
    tokio::time::timeout(Duration::from_secs(2), async {
        while active.load(Ordering::SeqCst) != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
}

async fn abrupt_disconnect(origin: &str, token: Option<&str>, active: &AtomicUsize) {
    let mut request = format!(
        "{}/v1/realtime?model=voice",
        origin.replacen("http", "ws", 1)
    )
    .into_client_request()
    .unwrap();
    request.headers_mut().insert(
        header::AUTHORIZATION,
        HeaderValue::from_static("Bearer caller-voice-key"),
    );
    if let Some(token) = token {
        request
            .headers_mut()
            .insert("x-nemo-relay-client-token", token.parse().unwrap());
    }
    let (mut socket, _) = tokio_tungstenite::connect_async(request).await.unwrap();
    let _ = socket.next().await.unwrap().unwrap();
    drop(socket);
    wait_disconnected(active).await;
}

async fn handshake_rejection(origin: &str, token: Option<&str>) {
    let mut request = format!("{}/v1/live?fail=1", origin.replacen("http", "ws", 1))
        .into_client_request()
        .unwrap();
    request.headers_mut().insert(
        header::AUTHORIZATION,
        HeaderValue::from_static("Bearer caller-voice-key"),
    );
    if let Some(token) = token {
        request
            .headers_mut()
            .insert("x-nemo-relay-client-token", token.parse().unwrap());
    }
    let error = tokio_tungstenite::connect_async(request).await.unwrap_err();
    let tokio_tungstenite::tungstenite::Error::Http(response) = error else {
        panic!("{error}");
    };
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(response.headers()["retry-after"], "7");
    assert_eq!(
        response.body().as_deref(),
        Some(b"upstream denied".as_slice())
    );
}

#[tokio::test]
async fn standalone_websocket_proxy_preserves_frames_and_handshake_failures() {
    let (upstream, provider_task, active) = provider().await;
    let config = GatewayConfig {
        openai_base_url: upstream,
        ..GatewayConfig::default()
    };
    let (origin, task) = serve(crate::server::router(config)).await;
    exercise(&origin, None, &active).await;
    task.abort();
    provider_task.abort();
}

#[tokio::test]
async fn websocket_proxy_rejects_invalid_upstream_handshake_before_upgrading() {
    for invalid in [
        "accept",
        "missing_accept",
        "upgrade",
        "connection",
        "unsolicited_protocol",
        "unoffered_protocol",
    ] {
        let app = Router::new().fallback(move |request: Request<Body>| async move {
            let key = request.headers()[header::SEC_WEBSOCKET_KEY].as_bytes();
            let mut headers = HeaderMap::new();
            headers.insert(header::CONNECTION, HeaderValue::from_static("Upgrade"));
            headers.insert(header::UPGRADE, HeaderValue::from_static("websocket"));
            headers.insert(
                header::SEC_WEBSOCKET_ACCEPT,
                tokio_tungstenite::tungstenite::handshake::derive_accept_key(key)
                    .parse()
                    .unwrap(),
            );
            headers.insert(
                "x-nemo-relay-private",
                HeaderValue::from_static("must-not-leak"),
            );
            match invalid {
                "accept" => {
                    headers.insert(
                        header::SEC_WEBSOCKET_ACCEPT,
                        HeaderValue::from_static("invalid"),
                    );
                }
                "missing_accept" => {
                    headers.remove(header::SEC_WEBSOCKET_ACCEPT);
                }
                "upgrade" => {
                    headers.insert(header::UPGRADE, HeaderValue::from_static("other"));
                }
                "connection" => {
                    headers.insert(header::CONNECTION, HeaderValue::from_static("keep-alive"));
                }
                _ => {
                    headers.insert(
                        header::SEC_WEBSOCKET_PROTOCOL,
                        HeaderValue::from_static("unoffered"),
                    );
                }
            }
            let mut response = Response::new(Body::empty());
            *response.status_mut() = StatusCode::SWITCHING_PROTOCOLS;
            *response.headers_mut() = headers;
            response
        });
        let (upstream, provider_task) = serve(app).await;
        let (origin, task) = serve(crate::server::router(GatewayConfig {
            openai_base_url: upstream,
            ..GatewayConfig::default()
        }))
        .await;
        let mut request = format!("{}/v1/live", origin.replacen("http", "ws", 1))
            .into_client_request()
            .unwrap();
        if invalid == "unoffered_protocol" {
            request.headers_mut().insert(
                header::SEC_WEBSOCKET_PROTOCOL,
                HeaderValue::from_static("voice"),
            );
        }
        let error = tokio_tungstenite::connect_async(request).await.unwrap_err();
        let tokio_tungstenite::tungstenite::Error::Http(response) = error else {
            panic!("{invalid}: {error}");
        };
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY, "{invalid}");
        assert!(!response.headers().contains_key("x-nemo-relay-private"));
        task.abort();
        provider_task.abort();
    }
}

#[tokio::test]
async fn websocket_proxy_rejects_invalid_client_handshakes_without_contacting_upstream() {
    let contacted = Arc::new(AtomicUsize::new(0));
    let app = Router::new().fallback({
        let contacted = Arc::clone(&contacted);
        move || {
            let contacted = Arc::clone(&contacted);
            async move {
                contacted.fetch_add(1, Ordering::SeqCst);
                StatusCode::INTERNAL_SERVER_ERROR
            }
        }
    });
    let (upstream, provider_task) = serve(app).await;
    let client = crate::daemon::common::transport::pooled_websocket_client().unwrap();
    for invalid in [
        "method",
        "upgrade",
        "connection",
        "version",
        "missing_version",
        "key",
        "short_key",
        "missing_key",
    ] {
        let mut request = Request::get("/v1/live")
            .header(header::CONNECTION, "keep-alive, Upgrade")
            .header(header::UPGRADE, "WebSocket")
            .header(header::SEC_WEBSOCKET_VERSION, "13")
            .header(header::SEC_WEBSOCKET_KEY, "dGhlIHNhbXBsZSBub25jZQ==")
            .body(Body::empty())
            .unwrap();
        match invalid {
            "method" => {
                *request.method_mut() = Method::POST;
            }
            "upgrade" => {
                request.headers_mut().remove(header::UPGRADE);
            }
            "connection" => {
                request.headers_mut().remove(header::CONNECTION);
            }
            "version" => {
                request.headers_mut().insert(
                    header::SEC_WEBSOCKET_VERSION,
                    HeaderValue::from_static("12"),
                );
            }
            "missing_version" => {
                request.headers_mut().remove(header::SEC_WEBSOCKET_VERSION);
            }
            "key" => {
                request.headers_mut().insert(
                    header::SEC_WEBSOCKET_KEY,
                    HeaderValue::from_static("invalid!"),
                );
            }
            "short_key" => {
                request
                    .headers_mut()
                    .insert(header::SEC_WEBSOCKET_KEY, HeaderValue::from_static("YQ=="));
            }
            _ => {
                request.headers_mut().remove(header::SEC_WEBSOCKET_KEY);
            }
        }
        let response = forward(
            &client,
            request,
            &format!("{upstream}/v1/live"),
            None,
            (),
            &GatewayConfig::default(),
            std::future::pending(),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{invalid}");
    }
    assert_eq!(contacted.load(Ordering::SeqCst), 0);
    provider_task.abort();
}

#[tokio::test]
async fn websocket_proxy_reports_connect_failures_and_handshake_timeouts() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let unavailable = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);
    let stalled = Router::new().fallback(|| async { std::future::pending::<StatusCode>().await });
    let (upstream, provider_task) = serve(stalled).await;
    for (upstream, expected) in [
        (unavailable, StatusCode::BAD_GATEWAY),
        (upstream, StatusCode::GATEWAY_TIMEOUT),
    ] {
        let (origin, task) = serve(crate::server::router(GatewayConfig {
            openai_base_url: upstream,
            response_timeout_secs: 1,
            ..GatewayConfig::default()
        }))
        .await;
        let error = tokio::time::timeout(
            Duration::from_secs(5),
            tokio_tungstenite::connect_async(format!(
                "{}/v1/live",
                origin.replacen("http", "ws", 1)
            )),
        )
        .await
        .unwrap()
        .unwrap_err();
        let tokio_tungstenite::tungstenite::Error::Http(response) = error else {
            panic!("{error}");
        };
        assert_eq!(response.status(), expected);
        task.abort();
    }
    provider_task.abort();
}

#[tokio::test]
async fn websocket_proxy_preserves_upstream_close_and_disconnects() {
    let app = Router::new().route(
        "/v1/live",
        get(|ws: WebSocketUpgrade, uri: Uri| async move {
            ws.on_upgrade(move |mut socket| async move {
                socket
                    .send(AxumMessage::Text("ready".into()))
                    .await
                    .unwrap();
                if uri.query() == Some("close=1") {
                    socket
                        .send(AxumMessage::Close(Some(axum::extract::ws::CloseFrame {
                            code: 1001,
                            reason: "upstream shutdown".into(),
                        })))
                        .await
                        .unwrap();
                    let _ = socket.next().await;
                }
            })
        }),
    );
    let (upstream, provider_task) = serve(app).await;
    let (origin, task) = serve(crate::server::router(GatewayConfig {
        openai_base_url: upstream,
        ..GatewayConfig::default()
    }))
    .await;
    for query in ["close=1", "abort=1"] {
        let (mut socket, _) = tokio_tungstenite::connect_async(format!(
            "{}/v1/live?{query}",
            origin.replacen("http", "ws", 1)
        ))
        .await
        .unwrap();
        assert_eq!(
            socket.next().await.unwrap().unwrap(),
            Message::Text("ready".into())
        );
        let message = tokio::time::timeout(Duration::from_secs(2), socket.next())
            .await
            .unwrap();
        if query == "close=1" {
            assert_eq!(
                message.unwrap().unwrap(),
                Message::Close(Some(tokio_tungstenite::tungstenite::protocol::CloseFrame {
                    code: tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode::Away,
                    reason: "upstream shutdown".into()
                }))
            );
            let _ = socket.flush().await;
        } else {
            assert!(message.is_none() || message.unwrap().is_err());
        }
    }
    task.abort();
    provider_task.abort();
}

#[tokio::test]
async fn websocket_accepts_protocol_from_second_header() {
    let app = Router::new().fallback(|request: Request<Body>| async move {
        let key = request.headers()[header::SEC_WEBSOCKET_KEY].as_bytes();
        Response::builder()
            .status(StatusCode::SWITCHING_PROTOCOLS)
            .header(header::CONNECTION, "Upgrade")
            .header(header::UPGRADE, "websocket")
            .header(
                header::SEC_WEBSOCKET_ACCEPT,
                tokio_tungstenite::tungstenite::handshake::derive_accept_key(key),
            )
            .header(header::SEC_WEBSOCKET_PROTOCOL, "voice")
            .body(Body::empty())
            .unwrap()
    });
    let (origin, task) = serve(app).await;
    let mut request = Request::builder()
        .method("GET")
        .uri("/v1/live")
        .header(header::CONNECTION, "Upgrade")
        .header(header::UPGRADE, "websocket")
        .header(header::SEC_WEBSOCKET_KEY, "dGhlIHNhbXBsZSBub25jZQ==")
        .header(header::SEC_WEBSOCKET_VERSION, "13")
        .body(Body::empty())
        .unwrap();
    request.headers_mut().append(
        header::SEC_WEBSOCKET_PROTOCOL,
        HeaderValue::from_static("other"),
    );
    request.headers_mut().append(
        header::SEC_WEBSOCKET_PROTOCOL,
        HeaderValue::from_static("voice"),
    );
    let client = crate::daemon::common::transport::pooled_websocket_client().unwrap();
    let response = crate::gateway::websocket::forward(
        &client,
        request,
        &format!("{origin}/v1/live"),
        None,
        (),
        &GatewayConfig::default(),
        std::future::pending(),
    )
    .await;
    task.abort();
    assert_eq!(response.status(), StatusCode::SWITCHING_PROTOCOLS);
}
