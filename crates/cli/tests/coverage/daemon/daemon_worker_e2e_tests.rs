// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::convert::Infallible;
use std::net::Ipv4Addr;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;

use http::header::{AUTHORIZATION, CONTENT_TYPE, HeaderValue, TE, TRAILER, UPGRADE};
use http_body_util::{BodyExt as _, Empty, Full};
use hyper::body::{Frame, Incoming, SizeHint};
use hyper::server::conn::{http1, http2};
use hyper::service::service_fn;
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::service::TowerToHyperService;
use sha2::{Digest, Sha256};
use tokio::net::TcpListener;
use tokio::sync::{Barrier, OwnedSemaphorePermit, Semaphore, oneshot};

use super::*;
use crate::daemon::common::transport::pooled_worker_h2c_client;
use crate::daemon::worker::{TestWorkerHandle, test_router as worker_test_router};

const EVENT_A: &[u8] = b": heartbeat\r\nevent: response.output_text.delta\r\nid: 7\r\nretry: 1000\r\ndata: first\r\ndata: second\r\n\r\n";
const EVENT_B: &[u8] = b"data: [DONE]\r\n\r\n\x80\xff";
const WORKER_TOKEN: &str = "test-daemon-to-worker-token";
const TEST_SEQUENCE_HEADER: &str = "x-test-stream-sequence";
const SEQUENCE_PARTS: usize = 4;

#[derive(Clone, Copy, Debug)]
enum TestProtocol {
    Http1,
    Http2,
}

#[derive(Clone, Copy, Debug)]
enum ProviderKind {
    OpenAi,
    Anthropic,
}

impl ProviderKind {
    const fn path(self) -> &'static str {
        match self {
            Self::OpenAi => "/v1/responses",
            Self::Anthropic => "/v1/messages",
        }
    }
}

#[derive(Clone, Copy)]
enum DuringStream {
    None,
    PauseBeyondFormerTotalTimeout,
    ControlLoss,
    Drain,
}

struct CausalProviderBody {
    phase: u8,
    release_second: oneshot::Receiver<()>,
    trailers: Option<HeaderMap>,
}

impl hyper::body::Body for CausalProviderBody {
    type Data = Bytes;
    type Error = Infallible;

    fn poll_frame(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let this = self.get_mut();
        match this.phase {
            0 => {
                this.phase = 1;
                Poll::Ready(Some(Ok(Frame::data(Bytes::from_static(EVENT_A)))))
            }
            1 => match Pin::new(&mut this.release_second).poll(context) {
                Poll::Pending => Poll::Pending,
                Poll::Ready(_) => {
                    this.phase = 2;
                    Poll::Ready(Some(Ok(Frame::data(Bytes::from_static(EVENT_B)))))
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

struct CountedProviderBody {
    remaining: usize,
    frame: Bytes,
    polls: Arc<AtomicUsize>,
}

impl hyper::body::Body for CountedProviderBody {
    type Data = Bytes;
    type Error = Infallible;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        _context: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        self.polls.fetch_add(1, Ordering::SeqCst);
        if self.remaining == 0 {
            return Poll::Ready(None);
        }
        self.remaining -= 1;
        Poll::Ready(Some(Ok(Frame::data(self.frame.clone()))))
    }
}

struct CancellationProviderBody {
    first_sent: bool,
    dropped: Option<oneshot::Sender<()>>,
}

impl hyper::body::Body for CancellationProviderBody {
    type Data = Bytes;
    type Error = Infallible;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        _context: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        if self.first_sent {
            Poll::Pending
        } else {
            self.first_sent = true;
            Poll::Ready(Some(Ok(Frame::data(Bytes::from_static(b"first\n\n")))))
        }
    }
}

impl Drop for CancellationProviderBody {
    fn drop(&mut self) {
        if let Some(dropped) = self.dropped.take() {
            let _ = dropped.send(());
        }
    }
}

struct SequencedProviderBody {
    sequence: usize,
    next_part: usize,
    _response_permit: OwnedSemaphorePermit,
}

impl hyper::body::Body for SequencedProviderBody {
    type Data = Bytes;
    type Error = Infallible;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        _context: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        if self.next_part == SEQUENCE_PARTS {
            return Poll::Ready(None);
        }
        let part = self.next_part;
        self.next_part += 1;
        Poll::Ready(Some(Ok(Frame::data(sequence_chunk(self.sequence, part)))))
    }
}

struct FidelityProviderBody {
    chunks: VecDeque<Bytes>,
    trailers: Option<HeaderMap>,
}

impl hyper::body::Body for FidelityProviderBody {
    type Data = Bytes;
    type Error = Infallible;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        _context: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        if let Some(chunk) = self.chunks.pop_front() {
            return Poll::Ready(Some(Ok(Frame::data(chunk))));
        }
        Poll::Ready(
            self.trailers
                .take()
                .map(|trailers| Ok(Frame::trailers(trailers))),
        )
    }
}

fn fidelity_corpus() -> Bytes {
    let mut bytes = Vec::with_capacity(300 * 1024);
    bytes.extend_from_slice(b": heartbeat\r\n\r\n");
    bytes.extend_from_slice(b"event: response.output_text.delta\r\nid: 17\r\nretry: 500\r\n");
    bytes.extend_from_slice(b"data: first\r\ndata: second\r\n\r\n");
    bytes.extend_from_slice(b"data: \xff\x00\xfe\r\n\r\n");
    bytes.extend_from_slice(b"event: large\r\ndata: ");
    bytes.extend(std::iter::repeat_n(b'L', 256 * 1024));
    bytes.extend_from_slice(b"\r\n\r\ndata: [DONE]\r\n\r\n");
    Bytes::from(bytes)
}

fn arbitrarily_split_fidelity_corpus(corpus: &Bytes) -> VecDeque<Bytes> {
    const WIDTHS: &[usize] = &[1, 2, 3, 7, 31, 257, 4_093, 16_384, 65_521];
    let mut chunks = VecDeque::new();
    let mut offset = 0;
    let mut split = 0;
    while offset < corpus.len() {
        let end = offset
            .saturating_add(WIDTHS[split % WIDTHS.len()])
            .min(corpus.len());
        chunks.push_back(corpus.slice(offset..end));
        offset = end;
        split += 1;
        if split == 4 {
            chunks.push_back(Bytes::new());
        }
    }
    chunks
}

fn sequence_chunk(sequence: usize, part: usize) -> Bytes {
    let prefix = format!("stream={sequence};part={part};");
    let mut chunk = vec![b'x'; 16 * 1024];
    chunk[..prefix.len()].copy_from_slice(prefix.as_bytes());
    *chunk.last_mut().expect("sequence chunk is non-empty") = b'\n';
    Bytes::from(chunk)
}

#[derive(Debug)]
struct ProviderObservation {
    path: String,
    authorization: Option<HeaderValue>,
    retained_client_token: bool,
    retained_worker_token: bool,
    body: Bytes,
}

fn client_for(protocol: TestProtocol) -> PooledClient {
    match protocol {
        TestProtocol::Http1 => pooled_client().expect("HTTP/1.1 pooled client"),
        TestProtocol::Http2 => {
            pooled_worker_h2c_client().expect("HTTP/2 prior-knowledge pooled client")
        }
    }
}

async fn spawn_causal_provider(
    protocol: TestProtocol,
) -> (
    std::net::SocketAddr,
    oneshot::Sender<()>,
    oneshot::Receiver<ProviderObservation>,
    tokio::task::JoinHandle<()>,
) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind causal provider");
    let address = listener.local_addr().expect("provider address");
    let (release_second, wait_for_release) = oneshot::channel();
    let (observed, observation) = oneshot::channel();
    let wait_for_release = Arc::new(Mutex::new(Some(wait_for_release)));
    let observed = Arc::new(Mutex::new(Some(observed)));
    let task = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.expect("accept worker connection");
        stream.set_nodelay(true).expect("provider TCP_NODELAY");
        let service = service_fn(move |request: Request<Incoming>| {
            let wait_for_release = wait_for_release
                .lock()
                .expect("release gate lock")
                .take()
                .expect("provider receives one request");
            let observed = observed
                .lock()
                .expect("observation lock")
                .take()
                .expect("provider observes one request");
            async move {
                let (parts, body) = request.into_parts();
                let body = body
                    .collect()
                    .await
                    .expect("provider request body")
                    .to_bytes();
                let _ = observed.send(ProviderObservation {
                    path: parts
                        .uri
                        .path_and_query()
                        .map_or("/", |value| value.as_str())
                        .to_owned(),
                    authorization: parts.headers.get(AUTHORIZATION).cloned(),
                    retained_client_token: parts.headers.contains_key(CLIENT_TOKEN_HEADER),
                    retained_worker_token: parts.headers.contains_key(WORKER_TOKEN_HEADER),
                    body,
                });

                let mut trailers = HeaderMap::new();
                trailers.append("x-stream-checksum", HeaderValue::from_static("one"));
                trailers.append("x-stream-checksum", HeaderValue::from_static("two"));
                let mut response = Response::new(box_body(CausalProviderBody {
                    phase: 0,
                    release_second: wait_for_release,
                    trailers: Some(trailers),
                }));
                *response.status_mut() = StatusCode::CREATED;
                response
                    .headers_mut()
                    .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
                response
                    .headers_mut()
                    .append("x-provider", HeaderValue::from_static("first"));
                response
                    .headers_mut()
                    .append("x-provider", HeaderValue::from_static("second"));
                response
                    .headers_mut()
                    .insert(TRAILER, HeaderValue::from_static("x-stream-checksum"));
                Ok::<_, Infallible>(response)
            }
        });

        match protocol {
            TestProtocol::Http1 => {
                let mut builder = http1::Builder::new();
                builder.keep_alive(false);
                builder
                    .serve_connection(TokioIo::new(stream), service)
                    .await
                    .expect("serve HTTP/1.1 provider")
            }
            TestProtocol::Http2 => http2::Builder::new(TokioExecutor::new())
                .serve_connection(TokioIo::new(stream), service)
                .await
                .expect("serve HTTP/2 provider"),
        }
    });
    (address, release_second, observation, task)
}

async fn spawn_counted_provider(
    protocol: TestProtocol,
    polls: Arc<AtomicUsize>,
    frame_count: usize,
    frame_size: usize,
) -> (std::net::SocketAddr, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind counted provider");
    let address = listener.local_addr().expect("counted provider address");
    let task = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.expect("accept worker connection");
        stream.set_nodelay(true).expect("provider TCP_NODELAY");
        let service = service_fn(move |_request: Request<Incoming>| {
            let polls = Arc::clone(&polls);
            async move {
                Ok::<_, Infallible>(Response::new(box_body(CountedProviderBody {
                    remaining: frame_count,
                    frame: Bytes::from(vec![0x5a; frame_size]),
                    polls,
                })))
            }
        });
        match protocol {
            TestProtocol::Http1 => {
                let mut builder = http1::Builder::new();
                builder.keep_alive(false);
                let _ = builder
                    .serve_connection(TokioIo::new(stream), service)
                    .await;
            }
            TestProtocol::Http2 => {
                let mut builder = http2::Builder::new(TokioExecutor::new());
                builder.max_concurrent_streams(256);
                builder.max_pending_accept_reset_streams(256);
                let _ = builder
                    .serve_connection(TokioIo::new(stream), service)
                    .await;
            }
        }
    });
    (address, task)
}

async fn spawn_cancellation_provider(
    protocol: TestProtocol,
) -> (
    std::net::SocketAddr,
    oneshot::Receiver<()>,
    tokio::task::JoinHandle<()>,
) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind cancellation provider");
    let address = listener
        .local_addr()
        .expect("cancellation provider address");
    let (dropped, wait_for_drop) = oneshot::channel();
    let dropped = Arc::new(Mutex::new(Some(dropped)));
    let task = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.expect("accept worker connection");
        stream.set_nodelay(true).expect("provider TCP_NODELAY");
        let service = service_fn(move |_request: Request<Incoming>| {
            let dropped = dropped
                .lock()
                .expect("cancellation signal lock")
                .take()
                .expect("provider receives exactly one request");
            async move {
                Ok::<_, Infallible>(Response::new(box_body(CancellationProviderBody {
                    first_sent: false,
                    dropped: Some(dropped),
                })))
            }
        });
        match protocol {
            TestProtocol::Http1 => {
                let mut builder = http1::Builder::new();
                builder.keep_alive(false);
                let _ = builder
                    .serve_connection(TokioIo::new(stream), service)
                    .await;
            }
            TestProtocol::Http2 => {
                let mut builder = http2::Builder::new(TokioExecutor::new());
                builder.max_concurrent_streams(256);
                builder.max_pending_accept_reset_streams(256);
                let _ = builder
                    .serve_connection(TokioIo::new(stream), service)
                    .await;
            }
        }
    });
    (address, wait_for_drop, task)
}

async fn spawn_fidelity_provider(
    protocol: TestProtocol,
) -> (std::net::SocketAddr, Bytes, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind fidelity provider");
    let address = listener.local_addr().expect("fidelity provider address");
    let corpus = fidelity_corpus();
    let chunks = Arc::new(Mutex::new(Some(arbitrarily_split_fidelity_corpus(&corpus))));
    let task = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.expect("accept worker connection");
        stream.set_nodelay(true).expect("provider TCP_NODELAY");
        let service = service_fn(move |_request: Request<Incoming>| {
            let chunks = chunks
                .lock()
                .expect("fidelity chunks lock")
                .take()
                .expect("fidelity provider receives exactly one request");
            async move {
                let mut trailers = HeaderMap::new();
                trailers.append("x-stream-checksum", HeaderValue::from_static("first"));
                trailers.append("x-stream-checksum", HeaderValue::from_static("second"));
                trailers.append("x-binary-safe", HeaderValue::from_static("yes"));
                let mut response = Response::new(box_body(FidelityProviderBody {
                    chunks,
                    trailers: Some(trailers),
                }));
                *response.status_mut() = StatusCode::PARTIAL_CONTENT;
                response
                    .headers_mut()
                    .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
                response
                    .headers_mut()
                    .append("x-provider", HeaderValue::from_static("first"));
                response
                    .headers_mut()
                    .append("x-provider", HeaderValue::from_static("second"));
                response.headers_mut().insert(
                    TRAILER,
                    HeaderValue::from_static("x-stream-checksum, x-binary-safe"),
                );
                Ok::<_, Infallible>(response)
            }
        });
        match protocol {
            TestProtocol::Http1 => {
                let mut builder = http1::Builder::new();
                builder.keep_alive(false);
                let _ = builder
                    .serve_connection(TokioIo::new(stream), service)
                    .await;
            }
            TestProtocol::Http2 => {
                let mut builder = http2::Builder::new(TokioExecutor::new());
                builder.max_concurrent_streams(256);
                builder.max_pending_accept_reset_streams(256);
                let _ = builder
                    .serve_connection(TokioIo::new(stream), service)
                    .await;
            }
        }
    });
    (address, corpus, task)
}

async fn spawn_sequenced_provider(
    protocol: TestProtocol,
    streams: usize,
    connections: Arc<AtomicUsize>,
) -> (std::net::SocketAddr, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind sequenced provider");
    let address = listener.local_addr().expect("sequenced provider address");
    let barrier = Arc::new(Barrier::new(streams));
    let response_permits = Arc::new(Semaphore::new(16));
    let task = tokio::spawn(async move {
        loop {
            let (stream, _) = listener.accept().await.expect("accept worker connection");
            stream.set_nodelay(true).expect("provider TCP_NODELAY");
            connections.fetch_add(1, Ordering::SeqCst);
            let barrier = Arc::clone(&barrier);
            let response_permits = Arc::clone(&response_permits);
            tokio::spawn(async move {
                let service = service_fn(move |request: Request<Incoming>| {
                    let barrier = Arc::clone(&barrier);
                    let response_permits = Arc::clone(&response_permits);
                    async move {
                        let sequence = request
                            .headers()
                            .get(TEST_SEQUENCE_HEADER)
                            .expect("sequence header reaches provider")
                            .to_str()
                            .expect("sequence header is ASCII")
                            .parse::<usize>()
                            .expect("sequence header is an integer");
                        request
                            .into_body()
                            .collect()
                            .await
                            .expect("provider receives the complete request body");
                        barrier.wait().await;
                        let response_permit = response_permits
                            .acquire_owned()
                            .await
                            .expect("response concurrency semaphore remains open");
                        Ok::<_, Infallible>(Response::new(box_body(SequencedProviderBody {
                            sequence,
                            next_part: 0,
                            _response_permit: response_permit,
                        })))
                    }
                });
                match protocol {
                    TestProtocol::Http1 => {
                        let _ = http1::Builder::new()
                            .serve_connection(TokioIo::new(stream), service)
                            .await;
                    }
                    TestProtocol::Http2 => {
                        let mut builder = http2::Builder::new(TokioExecutor::new());
                        builder.max_concurrent_streams(256);
                        builder.max_pending_accept_reset_streams(256);
                        let _ = builder
                            .serve_connection(TokioIo::new(stream), service)
                            .await;
                    }
                }
            });
        }
    });
    (address, task)
}

async fn spawn_router(
    protocol: TestProtocol,
    app: Router,
) -> (std::net::SocketAddr, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind test router");
    let address = listener.local_addr().expect("router address");
    let task = tokio::spawn(async move {
        let mut connections = tokio::task::JoinSet::new();
        loop {
            tokio::select! {
                accepted = listener.accept() => {
                    let (stream, _) = accepted.expect("accept test connection");
                    stream.set_nodelay(true).expect("router TCP_NODELAY");
                    let service = TowerToHyperService::new(app.clone());
                    connections.spawn(async move {
                        match protocol {
                            TestProtocol::Http1 => {
                                let mut builder = http1::Builder::new();
                                builder.keep_alive(true);
                                builder
                                    .serve_connection(TokioIo::new(stream), service)
                                    .await
                                    .expect("serve HTTP/1.1 router");
                            }
                            TestProtocol::Http2 => {
                                let mut builder = http2::Builder::new(TokioExecutor::new());
                                builder.max_concurrent_streams(256);
                                builder.max_pending_accept_reset_streams(256);
                                builder
                                    .serve_connection(TokioIo::new(stream), service)
                                    .await
                                    .expect("serve HTTP/2 router");
                            }
                        }
                    });
                }
                Some(completed) = connections.join_next(), if !connections.is_empty() => {
                    completed.expect("test router connection task");
                }
            }
        }
    });
    (address, task)
}

fn configured_worker_router(
    protocol: TestProtocol,
    provider: ProviderKind,
    provider_address: std::net::SocketAddr,
) -> (Router, TestWorkerHandle) {
    configured_worker_router_with_token(protocol, provider, provider_address, WORKER_TOKEN)
}

fn configured_worker_router_with_token(
    protocol: TestProtocol,
    provider: ProviderKind,
    provider_address: std::net::SocketAddr,
    worker_token: &str,
) -> (Router, TestWorkerHandle) {
    let mut config = GatewayConfig::default();
    match provider {
        ProviderKind::OpenAi => {
            config.openai_base_url = format!("http://{provider_address}/v1");
        }
        ProviderKind::Anthropic => {
            config.anthropic_base_url = format!("http://{provider_address}");
        }
    }
    worker_test_router(config, client_for(protocol), worker_token.as_bytes())
}

fn daemon_router_with_ready_worker(
    protocol: TestProtocol,
    route_token: &str,
    worker_address: std::net::SocketAddr,
) -> (Router, Arc<WorkerTarget>, tempfile::TempDir) {
    let machine_identity = MachineIdentity::generate()
        .expect("machine identity")
        .identity;
    let fingerprint = machine_identity.fingerprint();
    let credential = RouteCredential::parse(route_token.to_owned()).expect("route credential");
    let registry = Registry::new(false);
    let mcp_session = McpSessionId::new("test-mcp-session").expect("MCP session ID");
    let launch = WorkerLaunch {
        activation_id: "test-activation".into(),
        activation_token: SensitiveString::new("unused-test-activation-token")
            .expect("activation token"),
        deadline_unix_ms: now_unix_ms().saturating_add(ACTIVATION_LIFETIME_MS),
        bind_ip: Ipv4Addr::LOCALHOST,
        port: 0,
        advertise_address: None,
    };
    let directive = registry
        .register_mcp(
            McpRegistration {
                fingerprint,
                token_digest: credential.digest(),
                session_id: mcp_session,
                lease_expires_at_unix_ms: now_unix_ms().saturating_add(MCP_LEASE_MS),
            },
            launch,
        )
        .expect("register test MCP");
    assert!(matches!(directive, BrokerDirective::LaunchWorker { .. }));
    let target = Arc::new(
        WorkerTarget::with_client(
            "test-worker",
            format!("http://{worker_address}"),
            SensitiveString::new(WORKER_TOKEN).expect("worker token"),
            client_for(protocol),
        )
        .expect("worker target"),
    );
    registry
        .mark_worker_ready(fingerprint, "test-activation", Arc::clone(&target))
        .expect("publish test worker");

    let identity = MachineIdentity::generate()
        .expect("daemon identity")
        .identity;
    let generation_state = tempfile::tempdir().expect("generation state directory");
    let active_worker_generations = ActiveWorkerGenerations::load_for_test(
        generation_state
            .path()
            .join("active-worker-generations.json"),
    )
    .expect("active generation state");
    let state = Arc::new(DaemonState {
        registry,
        identity,
        descriptor: crate::daemon::common::control::descriptor(ComponentRole::Daemon),
        instance_id: "test-daemon".into(),
        public_origin: "http://127.0.0.1:1".into(),
        config: GatewayConfig::default(),
        upstream: pooled_client().expect("daemon pass-through client"),
        worker_clients: WorkerClientPool::new().expect("daemon worker clients"),
        allowed_route_tokens: HashSet::from([credential.digest()]),
        challenges: Mutex::new(HashMap::new()),
        activations: Mutex::new(HashMap::new()),
        mcp_sessions: Mutex::new(HashMap::new()),
        mcp_heartbeat_serialization: Mutex::new(()),
        worker_sessions: Mutex::new(HashMap::new()),
        pending_directives: Mutex::new(HashMap::new()),
        active_worker_generations,
        worker_generation_publication: Mutex::new(()),
    });
    (router(state), target, generation_state)
}

fn daemon_router_with_pass_through(
    protocol: TestProtocol,
    provider: ProviderKind,
    route_token: &str,
    provider_address: std::net::SocketAddr,
) -> (Router, tempfile::TempDir) {
    let credential = RouteCredential::parse(route_token.to_owned()).expect("route credential");
    let fingerprint = MachineIdentity::generate()
        .expect("machine identity")
        .identity
        .fingerprint();
    let registry = Registry::new(true);
    let directive = registry
        .register_mcp(
            McpRegistration {
                fingerprint,
                token_digest: credential.digest(),
                session_id: McpSessionId::new("test-pass-through-mcp").expect("MCP session ID"),
                lease_expires_at_unix_ms: now_unix_ms().saturating_add(MCP_LEASE_MS),
            },
            WorkerLaunch {
                activation_id: "unused-pass-through-activation".into(),
                activation_token: SensitiveString::new("unused-pass-through-token")
                    .expect("activation token"),
                deadline_unix_ms: now_unix_ms().saturating_add(ACTIVATION_LIFETIME_MS),
                bind_ip: Ipv4Addr::LOCALHOST,
                port: 0,
                advertise_address: None,
            },
        )
        .expect("register pass-through MCP");
    assert!(matches!(directive, BrokerDirective::UsePassThrough));

    let mut config = GatewayConfig::default();
    match provider {
        ProviderKind::OpenAi => {
            config.openai_base_url = format!("http://{provider_address}/v1");
        }
        ProviderKind::Anthropic => {
            config.anthropic_base_url = format!("http://{provider_address}");
        }
    }
    let identity = MachineIdentity::generate()
        .expect("daemon identity")
        .identity;
    let generation_state = tempfile::tempdir().expect("generation state directory");
    let active_worker_generations = ActiveWorkerGenerations::load_for_test(
        generation_state
            .path()
            .join("active-worker-generations.json"),
    )
    .expect("active generation state");
    let state = Arc::new(DaemonState {
        registry,
        identity,
        descriptor: crate::daemon::common::control::descriptor(ComponentRole::Daemon),
        instance_id: "test-pass-through-daemon".into(),
        public_origin: "http://127.0.0.1:1".into(),
        config,
        upstream: client_for(protocol),
        worker_clients: WorkerClientPool::new().expect("daemon worker clients"),
        allowed_route_tokens: HashSet::from([credential.digest()]),
        challenges: Mutex::new(HashMap::new()),
        activations: Mutex::new(HashMap::new()),
        mcp_sessions: Mutex::new(HashMap::new()),
        mcp_heartbeat_serialization: Mutex::new(()),
        worker_sessions: Mutex::new(HashMap::new()),
        pending_directives: Mutex::new(HashMap::new()),
        active_worker_generations,
        worker_generation_publication: Mutex::new(()),
    });
    (router(state), generation_state)
}

fn daemon_router_with_two_ready_workers(
    protocol: TestProtocol,
    route_tokens: [&str; 2],
    worker_addresses: [std::net::SocketAddr; 2],
    worker_tokens: [&str; 2],
) -> (Router, Vec<Arc<WorkerTarget>>, tempfile::TempDir) {
    let registry = Registry::new(false);
    let mut allowed_route_tokens = HashSet::new();
    let mut targets = Vec::with_capacity(2);

    for index in 0..2 {
        let credential =
            RouteCredential::parse(route_tokens[index].to_owned()).expect("route credential");
        let fingerprint = MachineIdentity::generate()
            .expect("machine identity")
            .identity
            .fingerprint();
        let activation_id = format!("test-activation-{index}");
        let directive = registry
            .register_mcp(
                McpRegistration {
                    fingerprint,
                    token_digest: credential.digest(),
                    session_id: McpSessionId::new(format!("test-mcp-session-{index}"))
                        .expect("MCP session ID"),
                    lease_expires_at_unix_ms: now_unix_ms().saturating_add(MCP_LEASE_MS),
                },
                WorkerLaunch {
                    activation_id: activation_id.clone(),
                    activation_token: SensitiveString::new(format!("unused-activation-{index}"))
                        .expect("activation token"),
                    deadline_unix_ms: now_unix_ms().saturating_add(ACTIVATION_LIFETIME_MS),
                    bind_ip: Ipv4Addr::LOCALHOST,
                    port: 0,
                    advertise_address: None,
                },
            )
            .expect("register test MCP");
        assert!(matches!(directive, BrokerDirective::LaunchWorker { .. }));
        let target = Arc::new(
            WorkerTarget::with_client(
                format!("test-worker-{index}"),
                format!("http://{}", worker_addresses[index]),
                SensitiveString::new(worker_tokens[index]).expect("worker token"),
                client_for(protocol),
            )
            .expect("worker target"),
        );
        registry
            .mark_worker_ready(fingerprint, &activation_id, Arc::clone(&target))
            .expect("publish test worker");
        allowed_route_tokens.insert(credential.digest());
        targets.push(target);
    }

    let identity = MachineIdentity::generate()
        .expect("daemon identity")
        .identity;
    let generation_state = tempfile::tempdir().expect("generation state directory");
    let active_worker_generations = ActiveWorkerGenerations::load_for_test(
        generation_state
            .path()
            .join("active-worker-generations.json"),
    )
    .expect("active generation state");
    let state = Arc::new(DaemonState {
        registry,
        identity,
        descriptor: crate::daemon::common::control::descriptor(ComponentRole::Daemon),
        instance_id: "test-two-route-daemon".into(),
        public_origin: "http://127.0.0.1:1".into(),
        config: GatewayConfig::default(),
        upstream: client_for(protocol),
        worker_clients: WorkerClientPool::new().expect("daemon worker clients"),
        allowed_route_tokens,
        challenges: Mutex::new(HashMap::new()),
        activations: Mutex::new(HashMap::new()),
        mcp_sessions: Mutex::new(HashMap::new()),
        mcp_heartbeat_serialization: Mutex::new(()),
        worker_sessions: Mutex::new(HashMap::new()),
        pending_directives: Mutex::new(HashMap::new()),
        active_worker_generations,
        worker_generation_publication: Mutex::new(()),
    });
    (router(state), targets, generation_state)
}

fn route_token() -> String {
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([0x5a; 32])
}

fn route_token_with(byte: u8) -> String {
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([byte; 32])
}

fn provider_request(
    daemon_address: std::net::SocketAddr,
    provider: ProviderKind,
    token: &str,
) -> Request<RelayBody> {
    Request::post(format!("http://{daemon_address}{}", provider.path()))
        .header(CLIENT_TOKEN_HEADER, token)
        .header(AUTHORIZATION, "Bearer caller-provider-token")
        .header(CONTENT_TYPE, "application/json")
        .header(TE, "trailers")
        .body(box_body(Full::new(Bytes::from_static(
            br#"{"model":"test","stream":true}"#,
        ))))
        .expect("provider request")
}

fn sequenced_provider_request(
    daemon_address: std::net::SocketAddr,
    provider: ProviderKind,
    token: &str,
    sequence: usize,
) -> Request<RelayBody> {
    Request::post(format!("http://{daemon_address}{}", provider.path()))
        .header(CLIENT_TOKEN_HEADER, token)
        .header(
            TEST_SEQUENCE_HEADER,
            HeaderValue::from_str(&sequence.to_string()).expect("valid sequence header"),
        )
        .body(box_body(Empty::<Bytes>::new()))
        .expect("sequenced provider request")
}

fn worker_provider_request(
    worker_address: std::net::SocketAddr,
    provider: ProviderKind,
    worker_token: &str,
) -> Request<RelayBody> {
    Request::post(format!("http://{worker_address}{}", provider.path()))
        .header(WORKER_TOKEN_HEADER, worker_token)
        .header(AUTHORIZATION, "Bearer caller-provider-token")
        .header(CONTENT_TYPE, "application/json")
        .header(TE, "trailers")
        .body(box_body(Full::new(Bytes::from_static(
            br#"{"model":"test","stream":true}"#,
        ))))
        .expect("worker provider request")
}

async fn assert_fidelity_response(response: Response<Incoming>, expected: &Bytes) {
    assert_eq!(response.status(), StatusCode::PARTIAL_CONTENT);
    assert_eq!(response.headers()[CONTENT_TYPE], "text/event-stream");
    assert_eq!(
        response
            .headers()
            .get_all("x-provider")
            .iter()
            .map(|value| value.to_str().expect("ASCII provider header"))
            .collect::<Vec<_>>(),
        ["first", "second"]
    );
    let mut actual = Vec::with_capacity(expected.len());
    let mut actual_trailers = None;
    let mut body = response.into_body();
    while let Some(frame) = body.frame().await {
        let frame = frame.expect("fidelity frame succeeds");
        match frame.into_data() {
            Ok(data) => actual.extend_from_slice(&data),
            Err(frame) => {
                actual_trailers = Some(frame.into_trailers().expect("only data or trailers"));
            }
        }
    }
    assert_eq!(Sha256::digest(&actual), Sha256::digest(expected));
    assert_eq!(actual.as_slice(), expected.as_ref());
    let trailers = actual_trailers.expect("fidelity trailers are preserved");
    assert_eq!(
        trailers
            .get_all("x-stream-checksum")
            .iter()
            .map(|value| value.to_str().expect("ASCII trailer"))
            .collect::<Vec<_>>(),
        ["first", "second"]
    );
    assert_eq!(trailers["x-binary-safe"], "yes");
}

async fn wait_for_poll_plateau(polls: &AtomicUsize, frame_count: usize) -> usize {
    tokio::time::timeout(Duration::from_secs(3), async {
        let mut previous = usize::MAX;
        let mut stable_rounds = 0;
        loop {
            tokio::task::yield_now().await;
            let current = polls.load(Ordering::SeqCst);
            assert!(
                current < frame_count,
                "an unread client must stop provider polling before the complete body"
            );
            if current > 0 && current == previous {
                stable_rounds += 1;
                if stable_rounds == 32 {
                    return current;
                }
            } else {
                previous = current;
                stable_rounds = 0;
            }
        }
    })
    .await
    .expect("provider polling reaches a bounded plateau")
}

async fn wait_for_in_flight_zero(worker_target: &WorkerTarget, worker_handle: &TestWorkerHandle) {
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if worker_target.in_flight() == 0 && worker_handle.in_flight() == 0 {
                return;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("daemon and worker in-flight counters return to zero");
}

async fn read_exact_data(body: &mut Incoming, expected: &[u8]) {
    let mut actual = Vec::new();
    while actual.len() < expected.len() {
        let frame = body
            .frame()
            .await
            .expect("data frame exists")
            .expect("data frame succeeds");
        let data = frame.into_data().expect("expected data before trailers");
        actual.extend_from_slice(&data);
    }
    assert_eq!(actual, expected);
}

async fn assert_new_request_rejected(
    client: &PooledClient,
    daemon_address: std::net::SocketAddr,
    provider: ProviderKind,
    token: &str,
) {
    let response = client
        .request(provider_request(daemon_address, provider, token))
        .await
        .expect("daemon returns worker admission failure");
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[allow(clippy::cognitive_complexity)]
async fn assert_full_hop(
    protocol: TestProtocol,
    provider: ProviderKind,
    during_stream: DuringStream,
) {
    let (provider_address, release_second, observed, provider_task) =
        spawn_causal_provider(protocol).await;
    let (worker_router, worker_handle) =
        configured_worker_router(protocol, provider, provider_address);
    let (worker_address, worker_task) = spawn_router(protocol, worker_router).await;
    let token = route_token();
    let (daemon_router, worker_target, _generation_state) =
        daemon_router_with_ready_worker(protocol, &token, worker_address);
    let (daemon_address, daemon_task) = spawn_router(protocol, daemon_router).await;
    let client = client_for(protocol);

    let response = client
        .request(provider_request(daemon_address, provider, &token))
        .await
        .expect("full-hop request succeeds");
    assert_eq!(response.status(), StatusCode::CREATED);
    assert_eq!(response.headers()[CONTENT_TYPE], "text/event-stream");
    assert_eq!(response.headers()[TRAILER], "x-stream-checksum");
    assert_eq!(
        response
            .headers()
            .get_all("x-provider")
            .iter()
            .map(|value| value.to_str().expect("ASCII provider header"))
            .collect::<Vec<_>>(),
        ["first", "second"]
    );
    let observed = observed.await.expect("provider observation");
    assert_eq!(observed.path, provider.path());
    assert_eq!(
        observed.authorization.as_ref().expect("provider auth"),
        "Bearer caller-provider-token"
    );
    assert!(!observed.retained_client_token);
    assert!(!observed.retained_worker_token);
    assert_eq!(observed.body.as_ref(), br#"{"model":"test","stream":true}"#);

    let mut body = response.into_body();
    read_exact_data(&mut body, EVENT_A).await;
    assert_eq!(worker_target.in_flight(), 1);
    assert_eq!(worker_handle.in_flight(), 1);

    match during_stream {
        DuringStream::None => {}
        DuringStream::PauseBeyondFormerTotalTimeout => {
            tokio::time::pause();
            tokio::time::advance(Duration::from_secs(301)).await;
        }
        DuringStream::ControlLoss => {
            worker_handle.control_lost();
            assert_new_request_rejected(&client, daemon_address, provider, &token).await;
        }
        DuringStream::Drain => {
            worker_handle.begin_drain(now_unix_ms().saturating_add(DRAIN_LIFETIME_MS));
            assert_new_request_rejected(&client, daemon_address, provider, &token).await;
        }
    }

    let next = body.frame();
    tokio::pin!(next);
    assert!(
        futures_util::poll!(next.as_mut()).is_pending(),
        "full-hop delivery must expose event A while the provider still withholds event B"
    );
    release_second.send(()).expect("release provider event B");
    let frame = next
        .await
        .expect("event B frame exists")
        .expect("event B frame succeeds");
    let mut event_b = frame
        .into_data()
        .expect("event B begins in a data frame")
        .to_vec();
    while event_b.len() < EVENT_B.len() {
        let frame = body
            .frame()
            .await
            .expect("remaining event B data")
            .expect("remaining event B frame succeeds");
        event_b.extend_from_slice(frame.data_ref().expect("event B completes before trailers"));
    }
    assert_eq!(event_b, EVENT_B);

    let trailers = body
        .frame()
        .await
        .expect("trailer frame exists")
        .expect("trailer frame succeeds")
        .into_trailers()
        .expect("last frame contains trailers");
    assert_eq!(
        trailers
            .get_all("x-stream-checksum")
            .iter()
            .map(|value| value.to_str().expect("ASCII trailer"))
            .collect::<Vec<_>>(),
        ["one", "two"]
    );
    assert!(body.frame().await.is_none());
    assert_eq!(worker_target.in_flight(), 0);
    assert_eq!(worker_handle.in_flight(), 0);

    daemon_task.abort();
    worker_task.abort();
    if matches!(protocol, TestProtocol::Http1) {
        provider_task.await.expect("HTTP/1.1 provider task");
    } else {
        provider_task.abort();
    }
}

#[tokio::test]
async fn full_hop_is_causally_non_aggregating_for_both_providers_over_http1() {
    assert_full_hop(
        TestProtocol::Http1,
        ProviderKind::OpenAi,
        DuringStream::None,
    )
    .await;
    assert_full_hop(
        TestProtocol::Http1,
        ProviderKind::Anthropic,
        DuringStream::None,
    )
    .await;
}

#[tokio::test]
async fn full_hop_is_causally_non_aggregating_for_both_providers_over_http2() {
    assert_full_hop(
        TestProtocol::Http2,
        ProviderKind::OpenAi,
        DuringStream::None,
    )
    .await;
    assert_full_hop(
        TestProtocol::Http2,
        ProviderKind::Anthropic,
        DuringStream::None,
    )
    .await;
}

#[tokio::test]
async fn full_hop_stream_has_no_former_three_hundred_second_total_timeout() {
    assert_full_hop(
        TestProtocol::Http1,
        ProviderKind::OpenAi,
        DuringStream::PauseBeyondFormerTotalTimeout,
    )
    .await;
}

#[tokio::test]
async fn admitted_full_hop_stream_survives_worker_control_loss() {
    assert_full_hop(
        TestProtocol::Http1,
        ProviderKind::OpenAi,
        DuringStream::ControlLoss,
    )
    .await;
}

#[tokio::test]
async fn admitted_full_hop_stream_survives_worker_drain() {
    assert_full_hop(
        TestProtocol::Http2,
        ProviderKind::Anthropic,
        DuringStream::Drain,
    )
    .await;
}

#[tokio::test]
async fn authenticated_codex_responses_route_preserves_pr994_method_compatibility() {
    let (provider_address, release_second, observed, provider_task) =
        spawn_causal_provider(TestProtocol::Http1).await;
    let (worker_router, _worker_handle) =
        configured_worker_router(TestProtocol::Http1, ProviderKind::OpenAi, provider_address);
    let (worker_address, worker_task) = spawn_router(TestProtocol::Http1, worker_router).await;
    let token = route_token();
    let (daemon_router, _worker_target, _generation_state) =
        daemon_router_with_ready_worker(TestProtocol::Http1, &token, worker_address);
    let (daemon_address, daemon_task) = spawn_router(TestProtocol::Http1, daemon_router).await;
    let client = client_for(TestProtocol::Http1);
    for path in [
        "/responses",
        "/v1/responses",
        "/backend-api/codex/responses",
    ] {
        let uri = format!("http://{daemon_address}{path}");
        let websocket_probe = Request::get(&uri)
            .header(CLIENT_TOKEN_HEADER, &token)
            .header(UPGRADE, "websocket")
            .body(box_body(Full::new(Bytes::new())))
            .expect("WebSocket probe request");
        let response = client
            .request(websocket_probe)
            .await
            .expect("daemon answers WebSocket probe");
        assert_eq!(response.status(), StatusCode::UPGRADE_REQUIRED, "{path}");
        response
            .into_body()
            .collect()
            .await
            .expect("WebSocket probe response body");

        let ordinary_get = Request::get(&uri)
            .header(CLIENT_TOKEN_HEADER, &token)
            .body(box_body(Full::new(Bytes::new())))
            .expect("ordinary GET request");
        let response = client
            .request(ordinary_get)
            .await
            .expect("daemon answers ordinary GET");
        assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED, "{path}");
        response
            .into_body()
            .collect()
            .await
            .expect("ordinary GET response body");
    }

    let uri = format!("http://{daemon_address}/backend-api/codex/responses");
    let mut post = provider_request(daemon_address, ProviderKind::OpenAi, &token);
    *post.uri_mut() = format!("{uri}?client=codex")
        .parse()
        .expect("canonical Codex response URI");
    let response = client
        .request(post)
        .await
        .expect("POST continues to forward");
    assert_eq!(response.status(), StatusCode::CREATED);
    let observed = observed.await.expect("provider observes forwarded POST");
    assert_eq!(observed.path, "/v1/responses?client=codex");
    release_second.send(()).expect("release provider response");
    response
        .into_body()
        .collect()
        .await
        .expect("forwarded POST response body");

    drop(client);
    daemon_task.abort();
    worker_task.abort();
    provider_task.await.expect("HTTP/1.1 provider task");
}

#[tokio::test]
async fn authenticated_pi_named_endpoint_crosses_the_daemon_and_worker() {
    let (provider_address, release_second, observed, provider_task) =
        spawn_causal_provider(TestProtocol::Http1).await;
    let worker_config = GatewayConfig {
        openai_base_url: "http://127.0.0.1:1/v1".into(),
        ..GatewayConfig::default()
    };
    let (worker_router, _worker_handle) = worker_test_router(
        worker_config,
        client_for(TestProtocol::Http1),
        WORKER_TOKEN.as_bytes(),
    );
    let (worker_address, worker_task) = spawn_router(TestProtocol::Http1, worker_router).await;
    let token = route_token();
    let (daemon_router, _worker_target, _generation_state) =
        daemon_router_with_ready_worker(TestProtocol::Http1, &token, worker_address);
    let (daemon_address, daemon_task) = spawn_router(TestProtocol::Http1, daemon_router).await;
    let client = client_for(TestProtocol::Http1);
    let mut request = provider_request(daemon_address, ProviderKind::OpenAi, &token);
    *request.uri_mut() = format!("http://{daemon_address}/responses?client=pi")
        .parse()
        .expect("Pi Responses URI");
    request.headers_mut().insert(
        crate::agents::pi::alignment::UPSTREAM_BASE_URL_HEADER,
        HeaderValue::from_str(&format!("http://{provider_address}/v1")).expect("named Pi endpoint"),
    );

    let response = client
        .request(request)
        .await
        .expect("named Pi request crosses daemon and worker");
    assert_eq!(response.status(), StatusCode::CREATED);
    let observed = observed.await.expect("named provider receives Pi request");
    assert_eq!(observed.path, "/v1/responses?client=pi");
    assert!(!observed.retained_client_token);
    assert!(!observed.retained_worker_token);

    release_second.send(()).expect("release provider response");
    response
        .into_body()
        .collect()
        .await
        .expect("named Pi response body");
    drop(client);
    daemon_task.abort();
    worker_task.abort();
    provider_task.await.expect("HTTP/1.1 provider task");
}

#[tokio::test]
async fn authenticated_pi_named_endpoint_is_honored_in_pass_through() {
    let (provider_address, release_second, observed, provider_task) =
        spawn_causal_provider(TestProtocol::Http1).await;
    let token = route_token();
    let (daemon_router, _generation_state) = daemon_router_with_pass_through(
        TestProtocol::Http1,
        ProviderKind::OpenAi,
        &token,
        provider_address,
    );
    let (daemon_address, daemon_task) = spawn_router(TestProtocol::Http1, daemon_router).await;
    let client = client_for(TestProtocol::Http1);
    let mut request = provider_request(daemon_address, ProviderKind::OpenAi, &token);
    *request.uri_mut() = format!("http://{daemon_address}/responses?client=pi")
        .parse()
        .expect("Pi Responses URI");
    request.headers_mut().insert(
        crate::agents::pi::alignment::UPSTREAM_BASE_URL_HEADER,
        HeaderValue::from_str(&format!("http://{provider_address}/custom/v1"))
            .expect("named Pi endpoint"),
    );

    let response = client
        .request(request)
        .await
        .expect("named Pi request crosses pass-through daemon");
    assert_eq!(response.status(), StatusCode::CREATED);
    let observed = observed.await.expect("named provider receives Pi request");
    assert_eq!(observed.path, "/custom/v1/responses?client=pi");
    assert!(!observed.retained_client_token);
    assert!(!observed.retained_worker_token);

    release_second.send(()).expect("release provider response");
    response
        .into_body()
        .collect()
        .await
        .expect("named Pi response body");
    drop(client);
    daemon_task.abort();
    provider_task.await.expect("HTTP/1.1 provider task");
}

#[tokio::test]
async fn dropping_full_hop_http2_client_cancels_provider_and_releases_accounting() {
    let (provider_address, provider_dropped, provider_task) =
        spawn_cancellation_provider(TestProtocol::Http2).await;
    let (worker_router, worker_handle) =
        configured_worker_router(TestProtocol::Http2, ProviderKind::OpenAi, provider_address);
    let (worker_address, worker_task) = spawn_router(TestProtocol::Http2, worker_router).await;
    let token = route_token();
    let (daemon_router, worker_target, _generation_state) =
        daemon_router_with_ready_worker(TestProtocol::Http2, &token, worker_address);
    let (daemon_address, daemon_task) = spawn_router(TestProtocol::Http2, daemon_router).await;
    let client = client_for(TestProtocol::Http2);

    let response = client
        .request(provider_request(
            daemon_address,
            ProviderKind::OpenAi,
            &token,
        ))
        .await
        .expect("full-hop cancellation response head");
    let mut body = response.into_body();
    read_exact_data(&mut body, b"first\n\n").await;
    assert_eq!(worker_target.in_flight(), 1);
    assert_eq!(worker_handle.in_flight(), 1);

    drop(body);
    tokio::time::timeout(Duration::from_secs(2), provider_dropped)
        .await
        .expect("provider body cancellation must be prompt")
        .expect("provider drop signal sent");
    wait_for_in_flight_zero(&worker_target, &worker_handle).await;

    drop(client);
    daemon_task.abort();
    worker_task.abort();
    provider_task.abort();
}

#[tokio::test]
async fn slow_full_hop_http2_reader_applies_bounded_backpressure_and_resumes() {
    const FRAME_COUNT: usize = 512;
    const FRAME_SIZE: usize = 64 * 1024;

    let polls = Arc::new(AtomicUsize::new(0));
    let (provider_address, provider_task) = spawn_counted_provider(
        TestProtocol::Http2,
        Arc::clone(&polls),
        FRAME_COUNT,
        FRAME_SIZE,
    )
    .await;
    let (worker_router, worker_handle) =
        configured_worker_router(TestProtocol::Http2, ProviderKind::OpenAi, provider_address);
    let (worker_address, worker_task) = spawn_router(TestProtocol::Http2, worker_router).await;
    let token = route_token();
    let (daemon_router, worker_target, _generation_state) =
        daemon_router_with_ready_worker(TestProtocol::Http2, &token, worker_address);
    let (daemon_address, daemon_task) = spawn_router(TestProtocol::Http2, daemon_router).await;
    let client = client_for(TestProtocol::Http2);

    let response = client
        .request(provider_request(
            daemon_address,
            ProviderKind::OpenAi,
            &token,
        ))
        .await
        .expect("full-hop backpressure response head");
    let polls_while_unread = wait_for_poll_plateau(&polls, FRAME_COUNT).await;
    assert!(polls_while_unread > 0, "provider body must begin streaming");

    let mut received = 0;
    let mut body = response.into_body();
    while let Some(frame) = body.frame().await {
        received += frame
            .expect("backpressure frame succeeds")
            .into_data()
            .expect("provider emits only data")
            .len();
    }
    assert_eq!(received, FRAME_COUNT * FRAME_SIZE);
    assert_eq!(polls.load(Ordering::SeqCst), FRAME_COUNT + 1);
    wait_for_in_flight_zero(&worker_target, &worker_handle).await;

    drop(client);
    daemon_task.abort();
    worker_task.abort();
    provider_task.abort();
}

#[allow(clippy::cognitive_complexity)]
async fn assert_pass_through_causal(protocol: TestProtocol, provider: ProviderKind) {
    let (provider_address, release_second, observed, provider_task) =
        spawn_causal_provider(protocol).await;
    let token = route_token();
    let (daemon_router, _generation_state) =
        daemon_router_with_pass_through(protocol, provider, &token, provider_address);
    let (daemon_address, daemon_task) = spawn_router(protocol, daemon_router).await;
    let client = client_for(protocol);
    let mut request = provider_request(daemon_address, provider, &token);
    let expected_authorization = if matches!(provider, ProviderKind::OpenAi) {
        *request.uri_mut() = format!("http://{daemon_address}/responses")
            .parse()
            .expect("generic OpenAI responses URI");
        request.headers_mut().insert(
            AUTHORIZATION,
            HeaderValue::from_static("Bearer at-caller-controlled-token"),
        );
        "Bearer at-caller-controlled-token"
    } else {
        "Bearer caller-provider-token"
    };

    let response = client
        .request(request)
        .await
        .expect("pass-through response head");
    assert_eq!(response.status(), StatusCode::CREATED);
    assert_eq!(response.headers()[CONTENT_TYPE], "text/event-stream");
    let observed = observed
        .await
        .expect("provider observes pass-through request");
    assert_eq!(observed.path, provider.path());
    assert_eq!(
        observed.authorization.as_ref().expect("provider auth"),
        expected_authorization
    );
    assert!(!observed.retained_client_token);
    assert!(!observed.retained_worker_token);

    let mut body = response.into_body();
    read_exact_data(&mut body, EVENT_A).await;
    let next = body.frame();
    tokio::pin!(next);
    assert!(
        futures_util::poll!(next.as_mut()).is_pending(),
        "pass-through must expose event A while the provider withholds event B"
    );
    release_second.send(()).expect("release provider event B");
    let mut remaining = Vec::new();
    let mut trailers = None;
    if let Some(frame) = next.await {
        let frame = frame.expect("remaining pass-through frame succeeds");
        match frame.into_data() {
            Ok(data) => remaining.extend_from_slice(&data),
            Err(frame) => trailers = Some(frame.into_trailers().expect("trailers frame")),
        }
    }
    while let Some(frame) = body.frame().await {
        let frame = frame.expect("remaining pass-through frame succeeds");
        match frame.into_data() {
            Ok(data) => remaining.extend_from_slice(&data),
            Err(frame) => trailers = Some(frame.into_trailers().expect("trailers frame")),
        }
    }
    assert_eq!(remaining, EVENT_B);
    assert_eq!(
        trailers
            .expect("pass-through trailers")
            .get_all("x-stream-checksum")
            .iter()
            .map(|value| value.to_str().expect("ASCII trailer"))
            .collect::<Vec<_>>(),
        ["one", "two"]
    );

    drop(client);
    daemon_task.abort();
    if matches!(protocol, TestProtocol::Http1) {
        provider_task.await.expect("HTTP/1.1 provider task");
    } else {
        provider_task.abort();
    }
}

async fn assert_pass_through_cancellation(protocol: TestProtocol) {
    let (provider_address, provider_dropped, provider_task) =
        spawn_cancellation_provider(protocol).await;
    let token = route_token();
    let (daemon_router, _generation_state) =
        daemon_router_with_pass_through(protocol, ProviderKind::OpenAi, &token, provider_address);
    let (daemon_address, daemon_task) = spawn_router(protocol, daemon_router).await;
    let client = client_for(protocol);

    let response = client
        .request(provider_request(
            daemon_address,
            ProviderKind::OpenAi,
            &token,
        ))
        .await
        .expect("pass-through cancellation response head");
    let mut body = response.into_body();
    read_exact_data(&mut body, b"first\n\n").await;
    drop(body);
    tokio::time::timeout(Duration::from_secs(2), provider_dropped)
        .await
        .expect("pass-through provider cancellation must be prompt")
        .expect("provider drop signal sent");

    drop(client);
    daemon_task.abort();
    provider_task.abort();
}

async fn assert_pass_through_backpressure(protocol: TestProtocol) {
    const FRAME_COUNT: usize = 512;
    const FRAME_SIZE: usize = 64 * 1024;

    let polls = Arc::new(AtomicUsize::new(0));
    let (provider_address, provider_task) =
        spawn_counted_provider(protocol, Arc::clone(&polls), FRAME_COUNT, FRAME_SIZE).await;
    let token = route_token();
    let (daemon_router, _generation_state) =
        daemon_router_with_pass_through(protocol, ProviderKind::OpenAi, &token, provider_address);
    let (daemon_address, daemon_task) = spawn_router(protocol, daemon_router).await;
    let client = client_for(protocol);
    let response = client
        .request(provider_request(
            daemon_address,
            ProviderKind::OpenAi,
            &token,
        ))
        .await
        .expect("pass-through backpressure response head");

    let plateau = wait_for_poll_plateau(&polls, FRAME_COUNT).await;
    assert!(
        plateau > 0,
        "provider body begins before it is backpressured"
    );
    let mut received = 0;
    let mut body = response.into_body();
    while let Some(frame) = body.frame().await {
        received += frame
            .expect("pass-through backpressure frame succeeds")
            .into_data()
            .expect("provider emits only data")
            .len();
    }
    assert_eq!(received, FRAME_COUNT * FRAME_SIZE);
    assert_eq!(polls.load(Ordering::SeqCst), FRAME_COUNT + 1);

    drop(client);
    daemon_task.abort();
    provider_task.abort();
}

async fn assert_pass_through_fidelity(protocol: TestProtocol) {
    let (provider_address, expected, provider_task) = spawn_fidelity_provider(protocol).await;
    let token = route_token();
    let (daemon_router, _generation_state) =
        daemon_router_with_pass_through(protocol, ProviderKind::OpenAi, &token, provider_address);
    let (daemon_address, daemon_task) = spawn_router(protocol, daemon_router).await;
    let client = client_for(protocol);
    let response = client
        .request(provider_request(
            daemon_address,
            ProviderKind::OpenAi,
            &token,
        ))
        .await
        .expect("pass-through fidelity response");
    assert_fidelity_response(response, &expected).await;

    drop(client);
    daemon_task.abort();
    provider_task.abort();
}

async fn assert_worker_only_fidelity(protocol: TestProtocol) {
    let (provider_address, expected, provider_task) = spawn_fidelity_provider(protocol).await;
    let (worker_router, worker_handle) =
        configured_worker_router(protocol, ProviderKind::OpenAi, provider_address);
    let (worker_address, worker_task) = spawn_router(protocol, worker_router).await;
    let client = client_for(protocol);
    let response = client
        .request(worker_provider_request(
            worker_address,
            ProviderKind::OpenAi,
            WORKER_TOKEN,
        ))
        .await
        .expect("worker-only fidelity response");
    assert_fidelity_response(response, &expected).await;
    tokio::time::timeout(Duration::from_secs(2), async {
        while worker_handle.in_flight() != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("worker-only in-flight count returns to zero");

    drop(client);
    worker_task.abort();
    provider_task.abort();
}

async fn assert_full_hop_fidelity(protocol: TestProtocol) {
    let (provider_address, expected, provider_task) = spawn_fidelity_provider(protocol).await;
    let (worker_router, worker_handle) =
        configured_worker_router(protocol, ProviderKind::OpenAi, provider_address);
    let (worker_address, worker_task) = spawn_router(protocol, worker_router).await;
    let token = route_token();
    let (daemon_router, worker_target, _generation_state) =
        daemon_router_with_ready_worker(protocol, &token, worker_address);
    let (daemon_address, daemon_task) = spawn_router(protocol, daemon_router).await;
    let client = client_for(protocol);
    let response = client
        .request(provider_request(
            daemon_address,
            ProviderKind::OpenAi,
            &token,
        ))
        .await
        .expect("full-hop fidelity response");
    assert_fidelity_response(response, &expected).await;
    wait_for_in_flight_zero(&worker_target, &worker_handle).await;

    drop(client);
    daemon_task.abort();
    worker_task.abort();
    provider_task.abort();
}

#[tokio::test]
async fn authenticated_pass_through_is_causal_for_both_providers_over_http1() {
    assert_pass_through_causal(TestProtocol::Http1, ProviderKind::OpenAi).await;
    assert_pass_through_causal(TestProtocol::Http1, ProviderKind::Anthropic).await;
}

#[tokio::test]
async fn authenticated_pass_through_is_causal_for_both_providers_over_http2() {
    assert_pass_through_causal(TestProtocol::Http2, ProviderKind::OpenAi).await;
    assert_pass_through_causal(TestProtocol::Http2, ProviderKind::Anthropic).await;
}

#[tokio::test]
async fn fidelity_corpus_crosses_pass_through_worker_and_full_hop_over_http1() {
    assert_pass_through_fidelity(TestProtocol::Http1).await;
    assert_worker_only_fidelity(TestProtocol::Http1).await;
    assert_full_hop_fidelity(TestProtocol::Http1).await;
}

#[tokio::test]
async fn fidelity_corpus_crosses_pass_through_worker_and_full_hop_over_http2() {
    assert_pass_through_fidelity(TestProtocol::Http2).await;
    assert_worker_only_fidelity(TestProtocol::Http2).await;
    assert_full_hop_fidelity(TestProtocol::Http2).await;
}

#[tokio::test]
async fn pass_through_cancellation_and_backpressure_hold_over_http1() {
    assert_pass_through_cancellation(TestProtocol::Http1).await;
    assert_pass_through_backpressure(TestProtocol::Http1).await;
}

#[tokio::test]
async fn pass_through_cancellation_and_backpressure_hold_over_http2() {
    assert_pass_through_cancellation(TestProtocol::Http2).await;
    assert_pass_through_backpressure(TestProtocol::Http2).await;
}

async fn assert_128_concurrent_full_hop_streams(protocol: TestProtocol) {
    const STREAMS: usize = 128;
    const ROUTE_STREAMS: usize = STREAMS / 2;

    let provider_connections = [Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0))];
    let (provider_address_a, provider_task_a) = spawn_sequenced_provider(
        protocol,
        ROUTE_STREAMS,
        Arc::clone(&provider_connections[0]),
    )
    .await;
    let (provider_address_b, provider_task_b) = spawn_sequenced_provider(
        protocol,
        ROUTE_STREAMS,
        Arc::clone(&provider_connections[1]),
    )
    .await;
    let worker_tokens = ["worker-route-a-token", "worker-route-b-token"];
    let (worker_router_a, worker_handle_a) = configured_worker_router_with_token(
        protocol,
        ProviderKind::OpenAi,
        provider_address_a,
        worker_tokens[0],
    );
    let (worker_router_b, worker_handle_b) = configured_worker_router_with_token(
        protocol,
        ProviderKind::OpenAi,
        provider_address_b,
        worker_tokens[1],
    );
    let (worker_address_a, worker_task_a) = spawn_router(protocol, worker_router_a).await;
    let (worker_address_b, worker_task_b) = spawn_router(protocol, worker_router_b).await;
    let tokens = [route_token_with(0x5a), route_token_with(0xa5)];
    let (daemon_router, worker_targets, _generation_state) = daemon_router_with_two_ready_workers(
        protocol,
        [&tokens[0], &tokens[1]],
        [worker_address_a, worker_address_b],
        worker_tokens,
    );
    let (daemon_address, daemon_task) = spawn_router(protocol, daemon_router).await;
    let client = client_for(protocol);

    let mut tasks = Vec::with_capacity(STREAMS);
    for sequence in 0..STREAMS {
        let client = client.clone();
        let token = tokens[sequence % tokens.len()].clone();
        tasks.push(tokio::spawn(async move {
            let response = client
                .request(sequenced_provider_request(
                    daemon_address,
                    ProviderKind::OpenAi,
                    &token,
                    sequence,
                ))
                .await
                .expect("concurrent full-hop request succeeds");
            assert_eq!(response.status(), StatusCode::OK);
            let mut actual = Vec::new();
            let mut body = response.into_body();
            while let Some(frame) = body.frame().await {
                actual.extend_from_slice(
                    &frame
                        .expect("concurrent full-hop frame succeeds")
                        .into_data()
                        .expect("sequenced provider emits only data"),
                );
            }
            let expected = (0..SEQUENCE_PARTS)
                .flat_map(|part| sequence_chunk(sequence, part))
                .collect::<Vec<_>>();
            assert_eq!(actual, expected);
        }));
    }
    tokio::time::timeout(Duration::from_secs(30), async {
        for task in tasks {
            task.await.expect("stream verification task succeeds");
        }
    })
    .await
    .expect("all 128 requests become concurrent and complete");

    wait_for_in_flight_zero(&worker_targets[0], &worker_handle_a).await;
    wait_for_in_flight_zero(&worker_targets[1], &worker_handle_b).await;
    for (route, connections) in provider_connections.iter().enumerate() {
        match protocol {
            TestProtocol::Http1 => assert_eq!(
                connections.load(Ordering::SeqCst),
                ROUTE_STREAMS,
                "each route's blocked HTTP/1.1 streams require independent pooled connections"
            ),
            TestProtocol::Http2 => assert_eq!(
                connections.load(Ordering::SeqCst),
                1,
                "route {route} must multiplex all streams over its own provider connection"
            ),
        }
    }

    drop(client);
    daemon_task.abort();
    worker_task_a.abort();
    worker_task_b.abort();
    provider_task_a.abort();
    provider_task_b.abort();
}

#[tokio::test]
async fn full_hop_keeps_128_concurrent_http2_streams_isolated() {
    assert_128_concurrent_full_hop_streams(TestProtocol::Http2).await;
}

#[tokio::test]
async fn full_hop_keeps_128_concurrent_http1_streams_isolated() {
    assert_128_concurrent_full_hop_streams(TestProtocol::Http1).await;
}

struct LifecycleDaemonHarness {
    state: Arc<DaemonState>,
    fingerprint: Fingerprint,
    mcp_session_id: String,
    mcp_secret: SensitiveString,
    worker_id: String,
    worker_control_secret: SensitiveString,
    generation_id: String,
    target: Arc<WorkerTarget>,
    _generation_state: tempfile::TempDir,
}

fn lifecycle_daemon_router(
    route_token: &str,
    worker_address: std::net::SocketAddr,
) -> (Router, LifecycleDaemonHarness) {
    let fingerprint = MachineIdentity::generate()
        .expect("machine identity")
        .identity
        .fingerprint();
    let credential = RouteCredential::parse(route_token.to_owned()).expect("route credential");
    let registry = Registry::new(false);
    let mcp_session_id = "lifecycle-mcp-session".to_owned();
    let activation_id = "lifecycle-activation";
    registry
        .register_mcp(
            McpRegistration {
                fingerprint,
                token_digest: credential.digest(),
                session_id: McpSessionId::new(mcp_session_id.clone()).expect("MCP session ID"),
                lease_expires_at_unix_ms: now_unix_ms().saturating_add(MCP_LEASE_MS),
            },
            WorkerLaunch {
                activation_id: activation_id.into(),
                activation_token: SensitiveString::new("unused-lifecycle-activation-token")
                    .expect("activation token"),
                deadline_unix_ms: now_unix_ms().saturating_add(ACTIVATION_LIFETIME_MS),
                bind_ip: Ipv4Addr::LOCALHOST,
                port: 0,
                advertise_address: None,
            },
        )
        .expect("register lifecycle MCP");

    let worker_id = "test-worker".to_owned();
    let endpoint = format!("http://{worker_address}");
    let target = Arc::new(
        WorkerTarget::with_client(
            worker_id.clone(),
            endpoint.clone(),
            SensitiveString::new(WORKER_TOKEN).expect("worker data token"),
            client_for(TestProtocol::Http1),
        )
        .expect("worker target"),
    );
    registry
        .mark_worker_ready(fingerprint, activation_id, Arc::clone(&target))
        .expect("publish lifecycle worker");

    let daemon_identity = MachineIdentity::generate()
        .expect("daemon identity")
        .identity;
    let generation_grant =
        WorkerGenerationGrant::issue(&worker_id, fingerprint, &endpoint, None, &daemon_identity)
            .expect("worker generation grant");
    let generation_id = generation_grant.generation_id.clone();
    let generation_state = tempfile::tempdir().expect("generation state directory");
    let active_worker_generations = ActiveWorkerGenerations::load_for_test(
        generation_state
            .path()
            .join("active-worker-generations.json"),
    )
    .expect("active generation state");
    active_worker_generations
        .publish(fingerprint, &generation_id)
        .expect("publish active generation");

    let mcp_secret = SensitiveString::new("lifecycle-mcp-control-token").expect("MCP token");
    let worker_control_secret =
        SensitiveString::new("unused-test-control-token").expect("worker control token");
    let state = Arc::new(DaemonState {
        registry,
        identity: daemon_identity,
        descriptor: crate::daemon::common::control::descriptor(ComponentRole::Daemon),
        instance_id: "lifecycle-daemon".into(),
        public_origin: "http://127.0.0.1:1".into(),
        config: GatewayConfig::default(),
        upstream: pooled_client().expect("daemon pass-through client"),
        worker_clients: WorkerClientPool::new().expect("daemon worker clients"),
        allowed_route_tokens: HashSet::from([credential.digest()]),
        challenges: Mutex::new(HashMap::new()),
        activations: Mutex::new(HashMap::new()),
        mcp_sessions: Mutex::new(HashMap::from([(
            mcp_session_id.clone(),
            McpControlSession {
                fingerprint,
                token_digest: credential.digest(),
                secret: mcp_secret.clone(),
                secret_digest: TokenDigest::from_token(mcp_secret.expose().as_bytes()),
                lease_expires_at_unix_ms: now_unix_ms().saturating_add(MCP_LEASE_MS),
                last_sequence: 0,
                last_request_id: String::new(),
                last_heartbeat: None,
                worker_network: WorkerNetworkHint {
                    advertised_host: Ipv4Addr::LOCALHOST.to_string(),
                    port: None,
                },
                released: false,
            },
        )])),
        mcp_heartbeat_serialization: Mutex::new(()),
        worker_sessions: Mutex::new(HashMap::from([(
            worker_id.clone(),
            WorkerControlSession {
                fingerprint,
                worker_id: worker_id.clone(),
                secret: worker_control_secret.clone(),
                secret_digest: TokenDigest::from_token(worker_control_secret.expose().as_bytes()),
                last_sequence: 0,
                last_request_id: String::new(),
                next_daemon_sequence: 0,
                lease_expires_at_unix_ms: now_unix_ms().saturating_add(WORKER_LEASE_MS),
                pending_target: Arc::clone(&target),
                publication: WorkerPublication::Activation {
                    activation_id: activation_id.into(),
                },
                published: true,
                generation_grant,
            },
        )])),
        pending_directives: Mutex::new(HashMap::new()),
        active_worker_generations,
        worker_generation_publication: Mutex::new(()),
    });
    (
        router(Arc::clone(&state)),
        LifecycleDaemonHarness {
            state,
            fingerprint,
            mcp_session_id,
            mcp_secret,
            worker_id,
            worker_control_secret,
            generation_id,
            target,
            _generation_state: generation_state,
        },
    )
}

async fn finish_causal_lifecycle_stream(body: &mut Incoming, release_second: oneshot::Sender<()>) {
    release_second.send(()).expect("release provider event B");
    let mut data = Vec::new();
    let mut trailers = None;
    while let Some(frame) = body.frame().await {
        let frame = frame.expect("remaining lifecycle stream frame succeeds");
        match frame.into_data() {
            Ok(bytes) => data.extend_from_slice(&bytes),
            Err(frame) => {
                trailers = Some(
                    frame
                        .into_trailers()
                        .expect("remaining lifecycle frame is trailers"),
                );
            }
        }
    }
    assert_eq!(data, EVENT_B);
    assert_eq!(
        trailers
            .expect("lifecycle stream trailers")
            .get_all("x-stream-checksum")
            .iter()
            .map(|value| value.to_str().expect("ASCII trailer"))
            .collect::<Vec<_>>(),
        ["one", "two"]
    );
}

async fn wait_for_worker_drain_control(
    client: &PooledClient,
    worker_address: std::net::SocketAddr,
) {
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let request = Request::get(format!("http://{worker_address}{WORKER_PROBE_PATH}"))
                .header(WORKER_TOKEN_HEADER, WORKER_TOKEN)
                .body(box_body(Empty::<Bytes>::new()))
                .expect("worker readiness request");
            let response = client
                .request(request)
                .await
                .expect("worker readiness response");
            let status = response.status();
            response
                .into_body()
                .collect()
                .await
                .expect("worker readiness body");
            if status == StatusCode::SERVICE_UNAVAILABLE {
                return;
            }
            assert_eq!(status, StatusCode::NO_CONTENT);
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("daemon drain control reaches worker");
}

#[tokio::test]
async fn broker_release_enters_draining_while_admitted_stream_finishes() {
    let (provider_address, release_second, _observed, provider_task) =
        spawn_causal_provider(TestProtocol::Http1).await;
    let (worker_router, worker_handle) =
        configured_worker_router(TestProtocol::Http1, ProviderKind::OpenAi, provider_address);
    let (worker_address, worker_task) = spawn_router(TestProtocol::Http1, worker_router).await;
    let token = route_token();
    let (daemon_router, harness) = lifecycle_daemon_router(&token, worker_address);
    let (daemon_address, daemon_task) = spawn_router(TestProtocol::Http1, daemon_router).await;
    let client = client_for(TestProtocol::Http1);

    let response = client
        .request(provider_request(
            daemon_address,
            ProviderKind::OpenAi,
            &token,
        ))
        .await
        .expect("admitted stream response head");
    assert_eq!(response.status(), StatusCode::CREATED);
    let mut body = response.into_body();
    read_exact_data(&mut body, EVENT_A).await;
    assert_eq!(harness.target.in_flight(), 1);
    assert_eq!(worker_handle.in_flight(), 1);

    let release = SessionRequest::new(
        harness.mcp_session_id.clone(),
        harness.mcp_secret.clone(),
        1,
        EmptyPayload::default(),
    )
    .expect("MCP release request");
    let request = Request::post(format!("http://{daemon_address}{MCP_RELEASE_PATH}"))
        .header(CONTENT_TYPE, "application/json")
        .body(box_body(Full::new(Bytes::from(
            serde_json::to_vec(&release).expect("serialize MCP release"),
        ))))
        .expect("MCP release HTTP request");
    let release_response = client.request(request).await.expect("MCP release response");
    assert_eq!(release_response.status(), StatusCode::NO_CONTENT);
    release_response
        .into_body()
        .collect()
        .await
        .expect("MCP release body");

    assert_eq!(
        harness
            .state
            .registry
            .snapshot(harness.fingerprint)
            .expect("draining route")
            .state,
        crate::daemon::broker::lifecycle::RouteStateKind::Draining
    );
    assert!(
        !harness
            .state
            .active_worker_generations
            .matches(harness.fingerprint, &harness.generation_id)
            .expect("generation revocation")
    );
    assert_new_request_rejected(&client, daemon_address, ProviderKind::OpenAi, &token).await;
    wait_for_worker_drain_control(&client, worker_address).await;

    finish_causal_lifecycle_stream(&mut body, release_second).await;
    wait_for_in_flight_zero(&harness.target, &worker_handle).await;
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if harness
                .state
                .registry
                .snapshot(harness.fingerprint)
                .expect("drain completion route")
                .state
                == crate::daemon::broker::lifecycle::RouteStateKind::Empty
            {
                return;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("draining route returns to empty after the accepted stream completes");

    drop(client);
    daemon_task.abort();
    worker_task.abort();
    provider_task.await.expect("HTTP/1.1 provider task");
}

#[tokio::test]
async fn broker_worker_heartbeat_expiry_rejects_new_work_but_preserves_admitted_stream() {
    let (provider_address, release_second, _observed, provider_task) =
        spawn_causal_provider(TestProtocol::Http1).await;
    let (worker_router, worker_handle) =
        configured_worker_router(TestProtocol::Http1, ProviderKind::OpenAi, provider_address);
    let (worker_address, worker_task) = spawn_router(TestProtocol::Http1, worker_router).await;
    let token = route_token();
    let (daemon_router, harness) = lifecycle_daemon_router(&token, worker_address);
    let (daemon_address, daemon_task) = spawn_router(TestProtocol::Http1, daemon_router).await;
    let client = client_for(TestProtocol::Http1);

    let heartbeat = SessionRequest::new(
        harness.worker_id.clone(),
        harness.worker_control_secret.clone(),
        1,
        WorkerHeartbeatPayload {
            worker_id: harness.worker_id.clone(),
        },
    )
    .expect("worker heartbeat request");
    let request = Request::post(format!("http://{daemon_address}{WORKER_HEARTBEAT_PATH}"))
        .header(CONTENT_TYPE, "application/json")
        .body(box_body(Full::new(Bytes::from(
            serde_json::to_vec(&heartbeat).expect("serialize worker heartbeat"),
        ))))
        .expect("worker heartbeat HTTP request");
    let heartbeat_response = client
        .request(request)
        .await
        .expect("worker heartbeat response");
    assert_eq!(heartbeat_response.status(), StatusCode::NO_CONTENT);
    heartbeat_response
        .into_body()
        .collect()
        .await
        .expect("worker heartbeat body");

    let response = client
        .request(provider_request(
            daemon_address,
            ProviderKind::OpenAi,
            &token,
        ))
        .await
        .expect("admitted stream response head");
    assert_eq!(response.status(), StatusCode::CREATED);
    let mut body = response.into_body();
    read_exact_data(&mut body, EVENT_A).await;
    assert_eq!(harness.target.in_flight(), 1);
    assert_eq!(worker_handle.in_flight(), 1);

    lock(&harness.state.worker_sessions)
        .get_mut(&harness.worker_id)
        .expect("live worker control session")
        .lease_expires_at_unix_ms = now_unix_ms().saturating_sub(1);
    spawn_maintenance(Arc::clone(&harness.state));
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let worker_expired =
                !lock(&harness.state.worker_sessions).contains_key(&harness.worker_id);
            let replacement_activating = harness
                .state
                .registry
                .snapshot(harness.fingerprint)
                .is_ok_and(|snapshot| {
                    snapshot.state == crate::daemon::broker::lifecycle::RouteStateKind::Activating
                });
            let relaunch_pending = matches!(
                lock(&harness.state.pending_directives).get(&harness.mcp_session_id),
                Some(BrokerDirective::LaunchWorker { .. })
            );
            if worker_expired && replacement_activating && relaunch_pending {
                return;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("expired worker heartbeat removes the control session");

    assert_eq!(
        harness
            .state
            .registry
            .snapshot(harness.fingerprint)
            .expect("recovering route")
            .state,
        crate::daemon::broker::lifecycle::RouteStateKind::Activating
    );
    assert!(matches!(
        lock(&harness.state.pending_directives).get(&harness.mcp_session_id),
        Some(BrokerDirective::LaunchWorker { .. })
    ));
    assert!(
        !harness
            .state
            .active_worker_generations
            .matches(harness.fingerprint, &harness.generation_id)
            .expect("expired generation revocation")
    );
    assert_new_request_rejected(&client, daemon_address, ProviderKind::OpenAi, &token).await;

    finish_causal_lifecycle_stream(&mut body, release_second).await;
    wait_for_in_flight_zero(&harness.target, &worker_handle).await;

    drop(client);
    daemon_task.abort();
    worker_task.abort();
    provider_task.await.expect("HTTP/1.1 provider task");
}
