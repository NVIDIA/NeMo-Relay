// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::convert::Infallible;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;

use http::header::{HeaderValue, SET_COOKIE, TE, TRAILER};
use http_body_util::{BodyExt, Empty};
use hyper::body::{Frame, Incoming, SizeHint};
use hyper::server::conn::{http1, http2};
use hyper::service::service_fn;
use hyper_util::rt::{TokioExecutor, TokioIo};
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

use super::*;

struct CausalBody {
    phase: u8,
    release_second: oneshot::Receiver<()>,
    trailers: Option<HeaderMap>,
}

impl CausalBody {
    fn new(release_second: oneshot::Receiver<()>, trailers: HeaderMap) -> Self {
        Self {
            phase: 0,
            release_second,
            trailers: Some(trailers),
        }
    }
}

impl Body for CausalBody {
    type Data = Bytes;
    type Error = Infallible;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let this = self.get_mut();
        match this.phase {
            0 => {
                this.phase = 1;
                Poll::Ready(Some(Ok(Frame::data(Bytes::from_static(b"event-a\n\n")))))
            }
            1 => match Pin::new(&mut this.release_second).poll(cx) {
                Poll::Pending => Poll::Pending,
                Poll::Ready(_) => {
                    this.phase = 2;
                    Poll::Ready(Some(Ok(Frame::data(Bytes::from_static(b"event-b\n\n")))))
                }
            },
            2 => {
                this.phase = 3;
                Poll::Ready(
                    this.trailers
                        .take()
                        .map(|trailers| Ok(Frame::trailers(trailers))),
                )
            }
            _ => Poll::Ready(None),
        }
    }

    fn is_end_stream(&self) -> bool {
        self.phase >= 3 && self.trailers.is_none()
    }

    fn size_hint(&self) -> SizeHint {
        SizeHint::default()
    }
}

#[derive(Clone, Copy)]
enum TestProtocol {
    Http1,
    Http2,
}

fn client_for(protocol: TestProtocol) -> PooledHttpClient {
    match protocol {
        TestProtocol::Http1 => pooled_http_client(),
        TestProtocol::Http2 => pooled_h2c_client(),
    }
}

fn request_with_empty_body(uri: String, protocol: TestProtocol) -> Request<RelayBody> {
    let mut request = Request::get(uri)
        .body(box_body(Empty::<Bytes>::new()))
        .expect("valid request");
    if matches!(protocol, TestProtocol::Http1) {
        request
            .headers_mut()
            .insert(TE, HeaderValue::from_static("trailers"));
    }
    request
}

async fn serve_one_connection<S>(listener: TcpListener, protocol: TestProtocol, service: S)
where
    S: hyper::service::Service<
            Request<Incoming>,
            Response = Response<RelayBody>,
            Error = Infallible,
        > + Send
        + 'static,
    S::Future: Send,
{
    let (stream, _) = listener.accept().await.expect("accept test client");
    match protocol {
        TestProtocol::Http1 => http1::Builder::new()
            .serve_connection(TokioIo::new(stream), service)
            .await
            .expect("serve HTTP/1.1 test connection"),
        TestProtocol::Http2 => {
            let mut builder = http2::Builder::new(TokioExecutor::new());
            builder.max_concurrent_streams(256);
            builder.max_pending_accept_reset_streams(256);
            builder
                .serve_connection(TokioIo::new(stream), service)
                .await
                .expect("serve HTTP/2 test connection")
        }
    }
}

async fn spawn_causal_provider(
    protocol: TestProtocol,
    release_second: oneshot::Receiver<()>,
) -> (std::net::SocketAddr, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind causal provider");
    let address = listener.local_addr().expect("bound provider address");
    let release_second = Arc::new(Mutex::new(Some(release_second)));
    let service = service_fn(move |_request: Request<Incoming>| {
        let release_second = release_second
            .lock()
            .expect("release gate lock")
            .take()
            .expect("causal provider receives exactly one request");
        async move {
            let mut trailers = HeaderMap::new();
            trailers.append("x-checksum", HeaderValue::from_static("one"));
            trailers.append("x-checksum", HeaderValue::from_static("two"));
            let response = Response::builder()
                .status(StatusCode::CREATED)
                .header(TRAILER, "x-checksum")
                .body(box_body(CausalBody::new(release_second, trailers)))
                .expect("valid causal response");
            Ok::<_, Infallible>(response)
        }
    });
    let task = tokio::spawn(serve_one_connection(listener, protocol, service));
    (address, task)
}

async fn spawn_relay(
    protocol: TestProtocol,
    provider: std::net::SocketAddr,
) -> (std::net::SocketAddr, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind relay");
    let address = listener.local_addr().expect("bound relay address");
    let upstream = client_for(protocol);
    let service = service_fn(move |request: Request<Incoming>| {
        let upstream = upstream.clone();
        async move {
            let path = request
                .uri()
                .path_and_query()
                .map(|value| value.as_str())
                .unwrap_or("/");
            let destination = format!("http://{provider}{path}")
                .parse()
                .expect("valid provider URI");
            let request = request.map(box_body);
            let request = prepare_forward_request(request, destination, &[])
                .expect("relay request head is valid");
            let response = upstream.request(request).await.expect("provider responds");
            let response = prepare_forward_response(response, &[])
                .expect("relay response head is valid")
                .map(box_body);
            Ok::<_, Infallible>(response)
        }
    });
    let task = tokio::spawn(serve_one_connection(listener, protocol, service));
    (address, task)
}

#[allow(clippy::cognitive_complexity)]
async fn assert_causal_relay(protocol: TestProtocol, path: &str) {
    let (release_second, wait_for_release) = oneshot::channel();
    let (provider, provider_task) = spawn_causal_provider(protocol, wait_for_release).await;
    let (relay, relay_task) = spawn_relay(protocol, provider).await;

    let client = client_for(protocol);
    let request = request_with_empty_body(format!("http://{relay}{path}"), protocol);
    let response = client.request(request).await.expect("relay responds");
    assert_eq!(response.status(), StatusCode::CREATED);
    assert_eq!(response.headers()[TRAILER], "x-checksum");

    let mut body = box_body(response.into_body());
    let first = body
        .frame()
        .await
        .expect("first frame exists")
        .expect("first frame succeeds")
        .into_data()
        .expect("first frame is data");
    assert_eq!(first, "event-a\n\n");

    let second = body.frame();
    tokio::pin!(second);
    assert!(
        futures_util::poll!(second.as_mut()).is_pending(),
        "relay must expose event A without waiting for event B"
    );
    release_second.send(()).expect("release provider event B");

    assert_eq!(
        second
            .await
            .expect("second frame exists")
            .expect("second frame succeeds")
            .into_data()
            .expect("second frame is data"),
        "event-b\n\n"
    );
    let trailers = body
        .frame()
        .await
        .expect("trailer frame exists")
        .expect("trailer frame succeeds")
        .into_trailers()
        .expect("last frame contains trailers");
    assert_eq!(
        trailers
            .get_all("x-checksum")
            .iter()
            .map(|value| value.to_str().expect("ASCII trailer"))
            .collect::<Vec<_>>(),
        ["one", "two"]
    );
    assert!(body.frame().await.is_none());

    drop(client);
    match protocol {
        TestProtocol::Http1 => {
            relay_task.await.expect("relay task succeeds");
            provider_task.await.expect("provider task succeeds");
        }
        TestProtocol::Http2 => {
            relay_task.abort();
            provider_task.abort();
        }
    }
}

struct FramesBody {
    frames: VecDeque<Frame<Bytes>>,
}

impl FramesBody {
    fn new(frames: impl IntoIterator<Item = Frame<Bytes>>) -> Self {
        Self {
            frames: frames.into_iter().collect(),
        }
    }
}

impl Body for FramesBody {
    type Data = Bytes;
    type Error = Infallible;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        Poll::Ready(self.frames.pop_front().map(Ok))
    }

    fn is_end_stream(&self) -> bool {
        self.frames.is_empty()
    }
}

struct CountedBody {
    remaining: usize,
    frame: Bytes,
    polls: Arc<AtomicUsize>,
}

impl Body for CountedBody {
    type Data = Bytes;
    type Error = Infallible;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        self.polls.fetch_add(1, Ordering::SeqCst);
        if self.remaining == 0 {
            return Poll::Ready(None);
        }
        self.remaining -= 1;
        Poll::Ready(Some(Ok(Frame::data(self.frame.clone()))))
    }
}

struct CancellationBody {
    first_sent: bool,
    dropped: Option<oneshot::Sender<()>>,
}

impl Body for CancellationBody {
    type Data = Bytes;
    type Error = Infallible;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        if self.first_sent {
            Poll::Pending
        } else {
            self.first_sent = true;
            Poll::Ready(Some(Ok(Frame::data(Bytes::from_static(b"first\n\n")))))
        }
    }
}

impl Drop for CancellationBody {
    fn drop(&mut self) {
        if let Some(dropped) = self.dropped.take() {
            let _ = dropped.send(());
        }
    }
}

#[test]
fn strips_connection_scoped_headers_and_preserves_trailer_declaration() {
    let mut headers = HeaderMap::new();
    headers.append(
        CONNECTION,
        HeaderValue::from_static("keep-alive, x-private"),
    );
    headers.append(CONNECTION, HeaderValue::from_static("proxy-connection"));
    headers.insert("keep-alive", HeaderValue::from_static("timeout=5"));
    headers.insert("proxy-connection", HeaderValue::from_static("keep-alive"));
    headers.insert("x-private", HeaderValue::from_static("secret"));
    headers.append(TRAILER, HeaderValue::from_static("x-checksum"));
    headers.append(TRAILER, HeaderValue::from_static("x-signature"));
    headers.append("x-end-to-end", HeaderValue::from_static("one"));
    headers.append("x-end-to-end", HeaderValue::from_static("two"));

    strip_hop_by_hop_headers(&mut headers).expect("valid headers");

    assert!(!headers.contains_key(CONNECTION));
    assert!(!headers.contains_key("keep-alive"));
    assert!(!headers.contains_key("proxy-connection"));
    assert!(!headers.contains_key("x-private"));
    assert_eq!(
        headers
            .get_all(TRAILER)
            .iter()
            .map(|value| value.to_str().expect("ASCII trailer declaration"))
            .collect::<Vec<_>>(),
        ["x-checksum", "x-signature"]
    );
    assert_eq!(
        headers
            .get_all("x-end-to-end")
            .iter()
            .map(|value| value.to_str().expect("ASCII test header"))
            .collect::<Vec<_>>(),
        ["one", "two"]
    );
}

#[test]
fn removes_trailer_declaration_when_connection_nominates_it() {
    let mut headers = HeaderMap::new();
    headers.insert(CONNECTION, HeaderValue::from_static("trailer"));
    headers.insert(TRAILER, HeaderValue::from_static("x-checksum"));

    strip_hop_by_hop_headers(&mut headers).expect("valid headers");

    assert!(!headers.contains_key(TRAILER));
}

#[test]
fn rewrites_only_the_request_head() {
    let (release, receiver) = oneshot::channel();
    let body = CausalBody::new(receiver, HeaderMap::new());
    let mut request = Request::post("http://old.example/v1/responses")
        .header(HOST, "old.example")
        .header("x-route-token", "private")
        .body(body)
        .expect("valid request");
    *request.version_mut() = http::Version::HTTP_2;
    request
        .headers_mut()
        .append("x-preserved", HeaderValue::from_static("first"));
    request
        .headers_mut()
        .append("x-preserved", HeaderValue::from_static("second"));

    let destination = "https://worker.example:8443/v1/responses?stream=true"
        .parse()
        .expect("valid destination");
    let request = prepare_forward_request(
        request,
        destination,
        &[HeaderName::from_static("x-route-token")],
    )
    .expect("request can be forwarded");

    assert_eq!(
        request.uri(),
        &"https://worker.example:8443/v1/responses?stream=true"
            .parse::<Uri>()
            .expect("valid expected URI")
    );
    assert_eq!(request.headers()[HOST], "worker.example:8443");
    assert_eq!(request.version(), http::Version::HTTP_11);
    assert!(!request.headers().contains_key("x-route-token"));
    assert_eq!(request.headers().get_all("x-preserved").iter().count(), 2);

    drop(request);
    assert!(
        release.send(()).is_err(),
        "the unchanged body owns the receiver"
    );
}

#[tokio::test]
async fn forwards_first_frame_before_source_releases_second_and_preserves_trailers() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind test server");
    let address = listener.local_addr().expect("bound address");
    let (release_second, wait_for_release) = oneshot::channel();

    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.expect("accept test client");
        let wait_for_release = Arc::new(Mutex::new(Some(wait_for_release)));
        let service = service_fn(move |_request: Request<Incoming>| {
            let wait_for_release = wait_for_release
                .lock()
                .expect("release gate lock")
                .take()
                .expect("test server receives exactly one request");
            async move {
                let mut trailers = HeaderMap::new();
                trailers.append("x-checksum", HeaderValue::from_static("one"));
                trailers.append("x-checksum", HeaderValue::from_static("two"));
                let body = box_body(CausalBody::new(wait_for_release, trailers));
                let response = Response::builder()
                    .header(TRAILER, "x-checksum")
                    .body(body)
                    .expect("valid response");
                Ok::<_, Infallible>(response)
            }
        });

        let mut connection = http1::Builder::new();
        connection.keep_alive(false);
        connection
            .serve_connection(TokioIo::new(stream), service)
            .await
            .expect("serve causal response");
    });

    let client = pooled_http_client();
    let request = Request::get(format!("http://{address}/v1/responses"))
        .header(TE, "trailers")
        .body(box_body(Empty::<Bytes>::new()))
        .expect("valid request");
    let response = client.request(request).await.expect("request succeeds");
    assert_eq!(response.headers()[TRAILER], "x-checksum");

    let mut body = box_body(response.into_body());
    let first = body
        .frame()
        .await
        .expect("first frame exists")
        .expect("first frame succeeds")
        .into_data()
        .expect("first frame is data");
    assert_eq!(first, "event-a\n\n");

    let second = body.frame();
    tokio::pin!(second);
    assert!(futures_util::poll!(second.as_mut()).is_pending());
    release_second.send(()).expect("release source");

    let second = second
        .await
        .expect("second frame exists")
        .expect("second frame succeeds")
        .into_data()
        .expect("second frame is data");
    assert_eq!(second, "event-b\n\n");

    let trailers = body
        .frame()
        .await
        .expect("trailer frame exists")
        .expect("trailer frame succeeds")
        .into_trailers()
        .expect("last frame contains trailers");
    let values = trailers
        .get_all("x-checksum")
        .iter()
        .map(|value| value.to_str().expect("ASCII trailer"))
        .collect::<Vec<_>>();
    assert_eq!(values, ["one", "two"]);
    assert!(body.frame().await.is_none());

    drop(client);
    server.await.expect("server task succeeds");
}

#[tokio::test]
async fn h2_preserves_duplicate_trailer_multimap_values() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind test server");
    let address = listener.local_addr().expect("bound address");
    let (release_second, wait_for_release) = oneshot::channel();

    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.expect("accept test client");
        let release = Arc::new(Mutex::new(Some(wait_for_release)));
        let service = service_fn(move |_request: Request<Incoming>| {
            let wait_for_release = release
                .lock()
                .expect("release gate lock")
                .take()
                .expect("test server receives exactly one request");
            async move {
                let mut trailers = HeaderMap::new();
                trailers.append("x-checksum", HeaderValue::from_static("one"));
                trailers.append("x-checksum", HeaderValue::from_static("two"));
                Ok::<_, Infallible>(Response::new(box_body(CausalBody::new(
                    wait_for_release,
                    trailers,
                ))))
            }
        });

        http2::Builder::new(TokioExecutor::new())
            .serve_connection(TokioIo::new(stream), service)
            .await
            .expect("serve causal HTTP/2 response");
    });

    let client = pooled_h2c_client();
    let request = Request::get(format!("http://{address}/v1/messages"))
        .body(box_body(Empty::<Bytes>::new()))
        .expect("valid request");
    let response = client.request(request).await.expect("request succeeds");
    let mut body = box_body(response.into_body());
    assert_eq!(
        body.frame()
            .await
            .expect("first frame")
            .expect("first frame succeeds")
            .into_data()
            .expect("data frame"),
        "event-a\n\n"
    );
    release_second.send(()).expect("release source");
    assert_eq!(
        body.frame()
            .await
            .expect("second frame")
            .expect("second frame succeeds")
            .into_data()
            .expect("data frame"),
        "event-b\n\n"
    );
    let trailers = body
        .frame()
        .await
        .expect("trailer frame")
        .expect("trailer succeeds")
        .into_trailers()
        .expect("trailers");
    let values = trailers
        .get_all("x-checksum")
        .iter()
        .map(|value| value.to_str().expect("ASCII trailer"))
        .collect::<Vec<_>>();
    assert_eq!(values, ["one", "two"]);
    assert!(body.frame().await.is_none());

    drop(client);
    server.abort();
}

#[tokio::test]
async fn relay_is_causally_non_aggregating_for_openai_and_anthropic_over_http1() {
    assert_causal_relay(TestProtocol::Http1, "/v1/responses").await;
    assert_causal_relay(TestProtocol::Http1, "/v1/messages").await;
}

#[tokio::test]
async fn relay_is_causally_non_aggregating_for_openai_and_anthropic_over_http2() {
    assert_causal_relay(TestProtocol::Http2, "/v1/responses").await;
    assert_causal_relay(TestProtocol::Http2, "/v1/messages").await;
}

async fn assert_exact_relay_fidelity(protocol: TestProtocol) {
    let chunks = [
        Bytes::from_static(b": heartbeat\r\n\r\n"),
        Bytes::from_static(b"event: delta\r\nid: 17\r\nretry: 500\r\n"),
        Bytes::new(),
        Bytes::from_static(b"data: first\r\ndata: second\r\n\r\n"),
        Bytes::from_static(b"data: \xff\x00\xfe\r\n\r\n"),
        Bytes::from_static(b"data: [DONE]\r\n\r\n"),
    ];
    let expected = chunks
        .iter()
        .flat_map(|chunk| chunk.iter().copied())
        .collect::<Vec<_>>();

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind fidelity provider");
    let provider = listener.local_addr().expect("bound provider address");
    let service = service_fn(move |_request: Request<Incoming>| {
        let chunks = chunks.clone();
        async move {
            let mut trailers = HeaderMap::new();
            trailers.append("x-checksum", HeaderValue::from_static("first"));
            trailers.append("x-checksum", HeaderValue::from_static("second"));
            trailers.append("x-binary-safe", HeaderValue::from_static("yes"));
            let frames = chunks
                .into_iter()
                .map(Frame::data)
                .chain(std::iter::once(Frame::trailers(trailers)));
            let mut response = Response::builder()
                .status(StatusCode::PARTIAL_CONTENT)
                .header(TRAILER, "x-checksum, x-binary-safe")
                .body(box_body(FramesBody::new(frames)))
                .expect("valid fidelity response");
            response
                .headers_mut()
                .append(SET_COOKIE, HeaderValue::from_static("a=1"));
            response
                .headers_mut()
                .append(SET_COOKIE, HeaderValue::from_static("b=2"));
            Ok::<_, Infallible>(response)
        }
    });
    let provider_task = tokio::spawn(serve_one_connection(listener, protocol, service));
    let (relay, relay_task) = spawn_relay(protocol, provider).await;

    let client = client_for(protocol);
    let response = client
        .request(request_with_empty_body(
            format!("http://{relay}/v1/responses"),
            protocol,
        ))
        .await
        .expect("relay responds");
    assert_eq!(response.status(), StatusCode::PARTIAL_CONTENT);
    assert_eq!(
        response
            .headers()
            .get_all(SET_COOKIE)
            .iter()
            .map(|value| value.to_str().expect("ASCII cookie"))
            .collect::<Vec<_>>(),
        ["a=1", "b=2"]
    );

    let mut actual = Vec::new();
    let mut actual_trailers = None;
    let mut body = box_body(response.into_body());
    while let Some(frame) = body.frame().await {
        let frame = frame.expect("fidelity frame succeeds");
        match frame.into_data() {
            Ok(data) => actual.extend_from_slice(&data),
            Err(frame) => {
                actual_trailers = Some(frame.into_trailers().expect("only data or trailers"));
            }
        }
    }
    assert_eq!(actual, expected);
    let trailers = actual_trailers.expect("trailers preserved");
    assert_eq!(
        trailers
            .get_all("x-checksum")
            .iter()
            .map(|value| value.to_str().expect("ASCII checksum"))
            .collect::<Vec<_>>(),
        ["first", "second"]
    );
    assert_eq!(trailers["x-binary-safe"], "yes");

    drop(client);
    match protocol {
        TestProtocol::Http1 => {
            relay_task.await.expect("relay task succeeds");
            provider_task.await.expect("provider task succeeds");
        }
        TestProtocol::Http2 => {
            relay_task.abort();
            provider_task.abort();
        }
    }
}

#[tokio::test]
async fn relay_preserves_exact_bytes_duplicate_headers_and_trailers_over_http1() {
    assert_exact_relay_fidelity(TestProtocol::Http1).await;
}

#[tokio::test]
async fn relay_preserves_exact_bytes_duplicate_headers_and_trailers_over_http2() {
    assert_exact_relay_fidelity(TestProtocol::Http2).await;
}

#[tokio::test]
async fn slow_http2_reader_applies_bounded_backpressure_and_then_resumes() {
    const FRAME_COUNT: usize = 128;
    const FRAME_SIZE: usize = 64 * 1024;

    let polls = Arc::new(AtomicUsize::new(0));
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind backpressure provider");
    let provider = listener.local_addr().expect("bound provider address");
    let provider_polls = polls.clone();
    let service = service_fn(move |_request: Request<Incoming>| {
        let polls = provider_polls.clone();
        async move {
            Ok::<_, Infallible>(Response::new(box_body(CountedBody {
                remaining: FRAME_COUNT,
                frame: Bytes::from(vec![0x5a; FRAME_SIZE]),
                polls,
            })))
        }
    });
    let provider_task = tokio::spawn(serve_one_connection(listener, TestProtocol::Http2, service));
    let (relay, relay_task) = spawn_relay(TestProtocol::Http2, provider).await;

    let client = pooled_h2c_client();
    let response = client
        .request(request_with_empty_body(
            format!("http://{relay}/v1/responses"),
            TestProtocol::Http2,
        ))
        .await
        .expect("relay returns response head");
    assert!(
        polls.load(Ordering::SeqCst) < FRAME_COUNT,
        "an unread downstream must stop the provider before the full 8 MiB body is polled"
    );

    let mut received = 0;
    let mut body = box_body(response.into_body());
    while let Some(frame) = body.frame().await {
        received += frame
            .expect("backpressure frame succeeds")
            .into_data()
            .expect("provider emits only data")
            .len();
    }
    assert_eq!(received, FRAME_COUNT * FRAME_SIZE);
    assert_eq!(polls.load(Ordering::SeqCst), FRAME_COUNT + 1);

    drop(client);
    relay_task.abort();
    provider_task.abort();
}

#[tokio::test]
async fn dropping_http2_client_body_promptly_cancels_provider_body() {
    let (dropped, wait_for_drop) = oneshot::channel();
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind cancellation provider");
    let provider = listener.local_addr().expect("bound provider address");
    let dropped = Arc::new(Mutex::new(Some(dropped)));
    let service = service_fn(move |_request: Request<Incoming>| {
        let dropped = dropped
            .lock()
            .expect("cancellation signal lock")
            .take()
            .expect("provider receives exactly one request");
        async move {
            Ok::<_, Infallible>(Response::new(box_body(CancellationBody {
                first_sent: false,
                dropped: Some(dropped),
            })))
        }
    });
    let provider_task = tokio::spawn(serve_one_connection(listener, TestProtocol::Http2, service));
    let (relay, relay_task) = spawn_relay(TestProtocol::Http2, provider).await;

    let client = pooled_h2c_client();
    let response = client
        .request(request_with_empty_body(
            format!("http://{relay}/v1/responses"),
            TestProtocol::Http2,
        ))
        .await
        .expect("relay returns response head");
    let mut body = box_body(response.into_body());
    assert_eq!(
        body.frame()
            .await
            .expect("first frame exists")
            .expect("first frame succeeds")
            .into_data()
            .expect("first frame is data"),
        "first\n\n"
    );
    drop(body);
    drop(client);

    tokio::time::timeout(Duration::from_secs(2), wait_for_drop)
        .await
        .expect("provider body cancellation must be prompt")
        .expect("provider drop signal sent");
    relay_task.abort();
    provider_task.abort();
}

async fn spawn_multiplexed_provider(
    connections: Arc<AtomicUsize>,
) -> (std::net::SocketAddr, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind multiplexed provider");
    let address = listener.local_addr().expect("bound provider address");
    let task = tokio::spawn(async move {
        loop {
            let (stream, _) = listener.accept().await.expect("accept provider client");
            connections.fetch_add(1, Ordering::SeqCst);
            tokio::spawn(async move {
                let service = service_fn(|request: Request<Incoming>| async move {
                    let sequence = request.uri().path().trim_start_matches('/').to_owned();
                    let frames = (0..4).map(|part| Frame::data(multiplexed_chunk(&sequence, part)));
                    Ok::<_, Infallible>(Response::new(box_body(FramesBody::new(frames))))
                });
                let mut builder = http2::Builder::new(TokioExecutor::new());
                builder.max_concurrent_streams(256);
                builder.max_pending_accept_reset_streams(256);
                let _ = builder
                    .serve_connection(TokioIo::new(stream), service)
                    .await;
            });
        }
    });
    (address, task)
}

fn multiplexed_chunk(sequence: &str, part: usize) -> Bytes {
    let prefix = format!("stream={sequence};part={part};");
    let mut chunk = vec![b'x'; 512];
    chunk[..prefix.len()].copy_from_slice(prefix.as_bytes());
    chunk[511] = b'\n';
    Bytes::from(chunk)
}

async fn spawn_multiplexed_relay(
    provider: std::net::SocketAddr,
    connections: Arc<AtomicUsize>,
) -> (std::net::SocketAddr, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind multiplexed relay");
    let address = listener.local_addr().expect("bound relay address");
    let upstream = pooled_h2c_client();
    let task = tokio::spawn(async move {
        loop {
            let (stream, _) = listener.accept().await.expect("accept relay client");
            connections.fetch_add(1, Ordering::SeqCst);
            let upstream = upstream.clone();
            tokio::spawn(async move {
                let service = service_fn(move |request: Request<Incoming>| {
                    let upstream = upstream.clone();
                    async move {
                        let path = request
                            .uri()
                            .path_and_query()
                            .map(|value| value.as_str())
                            .unwrap_or("/");
                        let destination = format!("http://{provider}{path}")
                            .parse()
                            .expect("valid provider URI");
                        let request =
                            prepare_forward_request(request.map(box_body), destination, &[])
                                .expect("valid multiplexed request head");
                        let response = upstream.request(request).await.expect("provider responds");
                        Ok::<_, Infallible>(
                            prepare_forward_response(response, &[])
                                .expect("valid multiplexed response head")
                                .map(box_body),
                        )
                    }
                });
                let mut builder = http2::Builder::new(TokioExecutor::new());
                builder.max_concurrent_streams(256);
                builder.max_pending_accept_reset_streams(256);
                let _ = builder
                    .serve_connection(TokioIo::new(stream), service)
                    .await;
            });
        }
    });
    (address, task)
}

#[tokio::test]
async fn multiplexed_http2_keeps_128_concurrent_streams_isolated() {
    const STREAMS: usize = 128;

    let provider_connections = Arc::new(AtomicUsize::new(0));
    let relay_connections = Arc::new(AtomicUsize::new(0));
    let (provider, provider_task) = spawn_multiplexed_provider(provider_connections.clone()).await;
    let (relay, relay_task) = spawn_multiplexed_relay(provider, relay_connections.clone()).await;
    let client = pooled_h2c_client();

    // Warm both pools before introducing concurrency so all work multiplexes over established
    // HTTP/2 connections rather than racing connection establishment.
    let warm = client
        .request(request_with_empty_body(
            format!("http://{relay}/warm"),
            TestProtocol::Http2,
        ))
        .await
        .expect("warm request succeeds");
    let mut warm_body = box_body(warm.into_body());
    while warm_body.frame().await.is_some() {}

    let mut tasks = Vec::with_capacity(STREAMS);
    for sequence in 0..STREAMS {
        let client = client.clone();
        tasks.push(tokio::spawn(async move {
            let response = client
                .request(request_with_empty_body(
                    format!("http://{relay}/{sequence}"),
                    TestProtocol::Http2,
                ))
                .await
                .expect("concurrent request succeeds");
            let mut body = box_body(response.into_body());
            let mut actual = Vec::new();
            while let Some(frame) = body.frame().await {
                actual.extend_from_slice(
                    &frame
                        .expect("concurrent frame succeeds")
                        .into_data()
                        .expect("concurrent provider emits data"),
                );
            }
            let expected = (0..4)
                .flat_map(|part| multiplexed_chunk(&sequence.to_string(), part))
                .collect::<Vec<_>>();
            assert_eq!(actual, expected);
        }));
    }
    for task in tasks {
        task.await.expect("stream verification task succeeds");
    }

    assert_eq!(relay_connections.load(Ordering::SeqCst), 1);
    assert_eq!(provider_connections.load(Ordering::SeqCst), 1);

    drop(client);
    relay_task.abort();
    provider_task.abort();
}

async fn spawn_pooled_http1_provider() -> (std::net::SocketAddr, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind pooled HTTP/1.1 provider");
    let address = listener.local_addr().expect("bound provider address");
    let task = tokio::spawn(async move {
        loop {
            let (stream, _) = listener.accept().await.expect("accept provider client");
            tokio::spawn(async move {
                let service = service_fn(|request: Request<Incoming>| async move {
                    let sequence = request.uri().path().trim_start_matches('/').to_owned();
                    let frames = (0..4).map(|part| Frame::data(multiplexed_chunk(&sequence, part)));
                    Ok::<_, Infallible>(Response::new(box_body(FramesBody::new(frames))))
                });
                let _ = http1::Builder::new()
                    .serve_connection(TokioIo::new(stream), service)
                    .await;
            });
        }
    });
    (address, task)
}

async fn spawn_pooled_http1_relay(
    provider: std::net::SocketAddr,
) -> (std::net::SocketAddr, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind pooled HTTP/1.1 relay");
    let address = listener.local_addr().expect("bound relay address");
    let upstream = pooled_http_client();
    let task = tokio::spawn(async move {
        loop {
            let (stream, _) = listener.accept().await.expect("accept relay client");
            let upstream = upstream.clone();
            tokio::spawn(async move {
                let service = service_fn(move |request: Request<Incoming>| {
                    let upstream = upstream.clone();
                    async move {
                        let path = request
                            .uri()
                            .path_and_query()
                            .map(|value| value.as_str())
                            .unwrap_or("/");
                        let destination = format!("http://{provider}{path}")
                            .parse()
                            .expect("valid provider URI");
                        let request =
                            prepare_forward_request(request.map(box_body), destination, &[])
                                .expect("valid pooled request head");
                        let response = upstream.request(request).await.expect("provider responds");
                        Ok::<_, Infallible>(
                            prepare_forward_response(response, &[])
                                .expect("valid pooled response head")
                                .map(box_body),
                        )
                    }
                });
                let _ = http1::Builder::new()
                    .serve_connection(TokioIo::new(stream), service)
                    .await;
            });
        }
    });
    (address, task)
}

#[tokio::test]
async fn pooled_http1_keeps_128_concurrent_streams_isolated() {
    const STREAMS: usize = 128;

    let (provider, provider_task) = spawn_pooled_http1_provider().await;
    let (relay, relay_task) = spawn_pooled_http1_relay(provider).await;
    let client = pooled_http_client();
    let mut tasks = Vec::with_capacity(STREAMS);
    for sequence in 0..STREAMS {
        let client = client.clone();
        tasks.push(tokio::spawn(async move {
            let response = client
                .request(request_with_empty_body(
                    format!("http://{relay}/{sequence}"),
                    TestProtocol::Http1,
                ))
                .await
                .expect("concurrent request succeeds");
            let mut body = box_body(response.into_body());
            let mut actual = Vec::new();
            while let Some(frame) = body.frame().await {
                actual.extend_from_slice(
                    &frame
                        .expect("concurrent frame succeeds")
                        .into_data()
                        .expect("concurrent provider emits data"),
                );
            }
            let sequence = sequence.to_string();
            let expected = (0..4)
                .flat_map(|part| multiplexed_chunk(&sequence, part))
                .collect::<Vec<_>>();
            assert_eq!(actual, expected);
        }));
    }
    for task in tasks {
        task.await.expect("stream verification task succeeds");
    }

    drop(client);
    relay_task.abort();
    provider_task.abort();
}
