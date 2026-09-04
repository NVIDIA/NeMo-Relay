// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, ensure};
use bytes::Bytes;
use http::header::{CONTENT_TYPE, TRAILER};
use http::{HeaderMap, HeaderValue, Method, Request, Response, StatusCode};
use http_body_util::{BodyExt, Full, StreamBody, combinators::UnsyncBoxBody};
use hyper::body::{Frame, Incoming};
use hyper::service::service_fn;
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder;
use serde::Serialize;
use sha2::{Digest, Sha256};
use tokio::net::TcpListener;
use tokio::sync::oneshot;

const RESPONSE_BYTES: &str = "x-benchmark-response-bytes";
const EVENT_COUNT: &str = "x-benchmark-event-count";
const EVENT_DELAY_MICROS: &str = "x-benchmark-event-delay-micros";
const STREAM_ID: &str = "x-benchmark-stream-id";
const BODY_SHA256: &str = "x-benchmark-body-sha256";

type ResponseBody = UnsyncBoxBody<Bytes, Infallible>;

#[derive(Default)]
struct ProviderStats {
    next_stream: AtomicU64,
    accepted: AtomicU64,
    completed: AtomicU64,
    cancelled: AtomicU64,
}

#[derive(Serialize)]
struct ProviderSnapshot {
    accepted: u64,
    completed: u64,
    cancelled: u64,
}

impl ProviderStats {
    fn snapshot(&self) -> ProviderSnapshot {
        ProviderSnapshot {
            accepted: self.accepted.load(Ordering::Relaxed),
            completed: self.completed.load(Ordering::Relaxed),
            cancelled: self.cancelled.load(Ordering::Relaxed),
        }
    }
}

struct CompletionGuard {
    stats: Arc<ProviderStats>,
    complete: bool,
}

impl Drop for CompletionGuard {
    fn drop(&mut self) {
        if !self.complete {
            self.stats.cancelled.fetch_add(1, Ordering::Relaxed);
        }
    }
}

pub async fn run(bind: SocketAddr, ready_file: Option<PathBuf>) -> Result<()> {
    let listener = TcpListener::bind(bind)
        .await
        .with_context(|| format!("failed to bind provider on {bind}"))?;
    let address = listener
        .local_addr()
        .context("failed to read provider address")?;
    let url = format!("http://{address}");
    if let Some(path) = ready_file {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        std::fs::write(&path, format!("{url}\n"))
            .with_context(|| format!("failed to write {}", path.display()))?;
    }
    println!("{{\"provider_url\":\"{url}\"}}");
    serve(listener, async {
        let _ = tokio::signal::ctrl_c().await;
    })
    .await
}

pub async fn spawn_ephemeral() -> Result<(
    String,
    oneshot::Sender<()>,
    tokio::task::JoinHandle<Result<()>>,
)> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .context("failed to bind smoke provider")?;
    let url = format!(
        "http://{}",
        listener
            .local_addr()
            .context("failed to read smoke provider address")?
    );
    let (stop, stopped) = oneshot::channel();
    let task = tokio::spawn(async move {
        serve(listener, async {
            let _ = stopped.await;
        })
        .await
    });
    Ok((url, stop, task))
}

async fn serve(listener: TcpListener, shutdown: impl Future<Output = ()>) -> Result<()> {
    let stats = Arc::new(ProviderStats::default());
    tokio::pin!(shutdown);
    loop {
        let (stream, _) = tokio::select! {
            accepted = listener.accept() => accepted.context("provider accept failed")?,
            _ = &mut shutdown => break,
        };
        stream
            .set_nodelay(true)
            .context("failed to set TCP_NODELAY")?;
        let connection_stats = Arc::clone(&stats);
        tokio::spawn(async move {
            let service = service_fn(move |request| handle(request, Arc::clone(&connection_stats)));
            let builder = Builder::new(TokioExecutor::new());
            if let Err(error) = builder
                .serve_connection(TokioIo::new(stream), service)
                .await
            {
                eprintln!("provider connection failed: {error}");
            }
        });
    }
    Ok(())
}

async fn handle(
    request: Request<Incoming>,
    stats: Arc<ProviderStats>,
) -> Result<Response<ResponseBody>, Infallible> {
    let response = match (request.method(), request.uri().path()) {
        (&Method::GET, "/healthz") => {
            full_response(StatusCode::OK, Bytes::from_static(b"ok"), "text/plain")
        }
        (&Method::GET, "/metrics") => {
            let body = serde_json::to_vec(&stats.snapshot()).expect("provider metrics serialize");
            full_response(StatusCode::OK, Bytes::from(body), "application/json")
        }
        (&Method::POST, "/v1/responses") => {
            stream_response(&request, "response.output_text.delta", stats)
        }
        (&Method::POST, "/v1/messages") => stream_response(&request, "content_block_delta", stats),
        _ => full_response(
            StatusCode::NOT_FOUND,
            Bytes::from_static(b"not found"),
            "text/plain",
        ),
    };
    Ok(response)
}

fn stream_response(
    request: &Request<Incoming>,
    event_type: &'static str,
    stats: Arc<ProviderStats>,
) -> Response<ResponseBody> {
    let parameters = parse_parameters(request.headers(), &stats);
    let Ok((response_bytes, event_count, delay, stream_id)) = parameters else {
        return full_response(
            StatusCode::BAD_REQUEST,
            Bytes::from(parameters.unwrap_err().to_string()),
            "text/plain",
        );
    };
    let base_size = (0..event_count)
        .map(|sequence| make_event(event_type, &stream_id, sequence, 0, 0).len())
        .sum::<usize>()
        + done_event().len();
    if response_bytes < base_size {
        return full_response(
            StatusCode::BAD_REQUEST,
            Bytes::from(format!(
                "response size {response_bytes} is smaller than minimum {base_size} for {event_count} events"
            )),
            "text/plain",
        );
    }

    stats.accepted.fetch_add(1, Ordering::Relaxed);
    let remaining = response_bytes - base_size;
    let body_stats = Arc::clone(&stats);
    let response_stream_id = stream_id.clone();
    let body = async_stream::stream! {
        let mut guard = CompletionGuard { stats: Arc::clone(&body_stats), complete: false };
        let mut hasher = Sha256::new();
        for sequence in 0..event_count {
            if sequence > 0 && !delay.is_zero() {
                tokio::time::sleep(delay).await;
            }
            let padding = remaining / event_count + usize::from(sequence < remaining % event_count);
            let emitted = unix_time_nanos();
            let event = make_event(event_type, &stream_id, sequence, emitted, padding);
            hasher.update(&event);
            yield Ok(Frame::data(event));
        }
        let done = done_event();
        hasher.update(&done);
        yield Ok(Frame::data(done));
        let mut trailers = HeaderMap::new();
        trailers.insert(BODY_SHA256, HeaderValue::from_str(&format!("{:x}", hasher.finalize())).expect("SHA-256 header"));
        trailers.insert(EVENT_COUNT, HeaderValue::from_str(&event_count.to_string()).expect("event count header"));
        yield Ok(Frame::trailers(trailers));
        guard.complete = true;
        body_stats.completed.fetch_add(1, Ordering::Relaxed);
    };

    Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, "text/event-stream")
        .header(TRAILER, format!("{BODY_SHA256}, {EVENT_COUNT}"))
        .header(STREAM_ID, response_stream_id)
        .header(EVENT_COUNT, event_count)
        .body(StreamBody::new(body).boxed_unsync())
        .expect("valid benchmark response")
}

fn parse_parameters(
    headers: &HeaderMap,
    stats: &ProviderStats,
) -> Result<(usize, usize, Duration, String)> {
    let response_bytes = header_number(headers, RESPONSE_BYTES, 16 * 1024)?;
    let event_count = header_number(headers, EVENT_COUNT, 128)?;
    ensure!(event_count >= 128, "event count must be at least 128");
    ensure!(event_count <= 1_000_000, "event count is too large");
    let delay_micros: u64 = header_number(headers, EVENT_DELAY_MICROS, 0)?;
    let stream_id = headers
        .get(STREAM_ID)
        .map(|value| value.to_str().context("stream ID is not ASCII"))
        .transpose()?
        .map(str::to_owned)
        .unwrap_or_else(|| format!("{:016x}", stats.next_stream.fetch_add(1, Ordering::Relaxed)));
    ensure!(
        stream_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-'),
        "stream ID contains unsupported characters"
    );
    Ok((
        response_bytes,
        event_count,
        Duration::from_micros(delay_micros),
        stream_id,
    ))
}

fn header_number<T>(headers: &HeaderMap, name: &'static str, default: T) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    headers
        .get(name)
        .map(|value| {
            value
                .to_str()
                .context("numeric benchmark header is not ASCII")?
                .parse()
                .context("invalid numeric benchmark header")
        })
        .transpose()
        .map(|value| value.unwrap_or(default))
}

fn make_event(
    event_type: &str,
    stream_id: &str,
    sequence: usize,
    emitted_unix_nanos: u128,
    padding: usize,
) -> Bytes {
    let metadata = if sequence == 0 {
        format!(": benchmark-heartbeat\nevent: {event_type}\nid: {stream_id}-0\nretry: 1000\n")
    } else {
        String::new()
    };
    Bytes::from(format!(
        "{metadata}data: {{\"type\":\"{event_type}\",\"i\":\"{stream_id}\",\"s\":{sequence},\"t\":\"{emitted_unix_nanos:020}\",\"d\":\"{}\"}}\n\n",
        "x".repeat(padding)
    ))
}

fn done_event() -> Bytes {
    Bytes::from_static(b"data: [DONE]\n\n")
}

fn unix_time_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

fn full_response(
    status: StatusCode,
    body: Bytes,
    content_type: &'static str,
) -> Response<ResponseBody> {
    Response::builder()
        .status(status)
        .header(CONTENT_TYPE, content_type)
        .body(Full::new(body).boxed_unsync())
        .expect("valid benchmark response")
}
