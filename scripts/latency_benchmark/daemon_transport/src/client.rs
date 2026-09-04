// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::task::{Context, Poll};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context as _, Result, anyhow, ensure};
use bytes::Bytes;
use hdrhistogram::Histogram;
use http::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE, TE};
use http::{Request, Uri};
use http_body_util::{BodyExt, Full};
use hyper_rustls::{HttpsConnector, HttpsConnectorBuilder};
use hyper_util::client::legacy::Client;
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::rt::{TokioExecutor, TokioTimer};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use tower_service::Service;

use crate::config::{LoadOptions, MatrixConfig, Protocol, Provider, Target, Topology};
use crate::metadata::{self, EnvironmentRecord};
use crate::provider;
use crate::resources::{ResourceRecord, ResourceSampler};

const RESPONSE_BYTES: &str = "x-benchmark-response-bytes";
const EVENT_COUNT: &str = "x-benchmark-event-count";
const EVENT_DELAY_MICROS: &str = "x-benchmark-event-delay-micros";
const STREAM_ID: &str = "x-benchmark-stream-id";
const BODY_SHA256: &str = "x-benchmark-body-sha256";
const MAX_HISTOGRAM_NANOS: u64 = 3_600_000_000_000;

type BaseConnector = HttpsConnector<HttpConnector>;

#[derive(Clone)]
struct CountingConnector {
    inner: BaseConnector,
    connections: Arc<AtomicU64>,
}

impl Service<Uri> for CountingConnector {
    type Response = <BaseConnector as Service<Uri>>::Response;
    type Error = <BaseConnector as Service<Uri>>::Error;
    type Future = <BaseConnector as Service<Uri>>::Future;

    fn poll_ready(&mut self, context: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(context)
    }

    fn call(&mut self, uri: Uri) -> Self::Future {
        self.connections.fetch_add(1, Ordering::Relaxed);
        self.inner.call(uri)
    }
}

#[derive(Clone)]
struct BenchmarkClient {
    client: Client<CountingConnector, Full<Bytes>>,
    connections: Arc<AtomicU64>,
}

impl BenchmarkClient {
    fn new(protocol: Protocol) -> Result<Self> {
        let mut http = HttpConnector::new();
        http.enforce_http(false);
        http.set_nodelay(true);
        let connector = match protocol {
            Protocol::Http1 => HttpsConnectorBuilder::new()
                .with_native_roots()
                .context("failed to load native TLS roots")?
                .https_or_http()
                .enable_http1()
                .wrap_connector(http),
            Protocol::Http2 => HttpsConnectorBuilder::new()
                .with_native_roots()
                .context("failed to load native TLS roots")?
                .https_or_http()
                .enable_http2()
                .wrap_connector(http),
        };
        let connections = Arc::new(AtomicU64::new(0));
        let connector = CountingConnector {
            inner: connector,
            connections: Arc::clone(&connections),
        };
        let mut builder = Client::builder(TokioExecutor::new());
        builder.timer(TokioTimer::new());
        builder.pool_idle_timeout(Duration::from_secs(120));
        builder.pool_max_idle_per_host(usize::MAX);
        if matches!(protocol, Protocol::Http2) {
            builder.http2_only(true);
            builder.http2_keep_alive_interval(Duration::from_secs(15));
            builder.http2_keep_alive_timeout(Duration::from_secs(5));
            builder.http2_keep_alive_while_idle(true);
        }
        Ok(Self {
            client: builder.build(connector),
            connections,
        })
    }

    fn connections(&self) -> u64 {
        self.connections.load(Ordering::Relaxed)
    }
}

struct Clients {
    http1: BenchmarkClient,
    http2: BenchmarkClient,
}

impl Clients {
    fn new() -> Result<Self> {
        Ok(Self {
            http1: BenchmarkClient::new(Protocol::Http1)?,
            http2: BenchmarkClient::new(Protocol::Http2)?,
        })
    }

    fn get(&self, protocol: Protocol) -> &BenchmarkClient {
        match protocol {
            Protocol::Http1 => &self.http1,
            Protocol::Http2 => &self.http2,
        }
    }
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u8,
    informational_only: bool,
    environment: EnvironmentRecord,
    parameters: MatrixConfig,
    targets: Vec<TargetRecord>,
    scenarios: Vec<ScenarioRecord>,
    validation_errors: Vec<String>,
}

#[derive(Debug, Serialize)]
struct TargetRecord {
    topology: Topology,
    url: String,
    configured_header_names: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ScenarioRecord {
    kind: &'static str,
    topology: Topology,
    protocol: Protocol,
    provider: Provider,
    response_bytes: usize,
    events: usize,
    concurrency: usize,
    configured_duration_seconds: Option<u64>,
    measured_duration_seconds: f64,
    requests_started: u64,
    requests_completed: u64,
    requests_cancelled: u64,
    requests_per_second: f64,
    goodput_mib_per_second: f64,
    response_head: HistogramRecord,
    first_content: HistogramRecord,
    per_event_forwarding_delay: HistogramRecord,
    total: HistogramRecord,
    integrity: IntegrityRecord,
    transport_errors: u64,
    connections_opened_during_measurement: u64,
    estimated_pool_reuses: u64,
    reconnect_count: u64,
    max_active_http2_streams: Option<usize>,
    queued_bytes_peak: Option<u64>,
    backpressure_stalls: Option<u64>,
    resources: BTreeMap<String, ResourceRecord>,
    unavailable_metrics: Vec<&'static str>,
}

#[derive(Debug, Serialize)]
struct HistogramRecord {
    samples: u64,
    p50_ms: Option<f64>,
    p95_ms: Option<f64>,
    p99_ms: Option<f64>,
    min_ms: Option<f64>,
    max_ms: Option<f64>,
}

impl HistogramRecord {
    fn from_histogram(histogram: &Histogram<u64>) -> Self {
        if histogram.is_empty() {
            return Self {
                samples: 0,
                p50_ms: None,
                p95_ms: None,
                p99_ms: None,
                min_ms: None,
                max_ms: None,
            };
        }
        Self {
            samples: histogram.len(),
            p50_ms: Some(nanos_to_millis(histogram.value_at_quantile(0.50))),
            p95_ms: Some(nanos_to_millis(histogram.value_at_quantile(0.95))),
            p99_ms: Some(nanos_to_millis(histogram.value_at_quantile(0.99))),
            min_ms: Some(nanos_to_millis(histogram.min())),
            max_ms: Some(nanos_to_millis(histogram.max())),
        }
    }
}

#[derive(Debug, Default, Serialize)]
struct IntegrityRecord {
    missing_events: u64,
    duplicate_events: u64,
    reordered_events: u64,
    corrupt_events: u64,
    cross_stream_events: u64,
    body_hash_mismatches: u64,
    trailer_mismatches: u64,
    status_errors: u64,
}

struct Accumulator {
    head: Histogram<u64>,
    first: Histogram<u64>,
    event_delay: Histogram<u64>,
    total: Histogram<u64>,
    started: u64,
    completed: u64,
    cancelled: u64,
    bytes: u64,
    integrity: IntegrityRecord,
    transport_errors: u64,
}

impl Accumulator {
    fn new() -> Self {
        Self {
            head: latency_histogram(),
            first: latency_histogram(),
            event_delay: latency_histogram(),
            total: latency_histogram(),
            started: 0,
            completed: 0,
            cancelled: 0,
            bytes: 0,
            integrity: IntegrityRecord::default(),
            transport_errors: 0,
        }
    }

    fn record(&mut self, result: Result<Observation>) {
        self.started += 1;
        match result {
            Ok(observation) => {
                record_duration(&mut self.head, observation.head);
                record_duration(&mut self.first, observation.first);
                if observation.cancelled {
                    self.cancelled += 1;
                    return;
                }
                record_duration(&mut self.total, observation.total);
                for delay in observation.event_delays {
                    let _ = self.event_delay.record(delay.clamp(1, MAX_HISTOGRAM_NANOS));
                }
                self.completed += 1;
                self.bytes += observation.bytes as u64;
                self.integrity.add(observation.integrity);
            }
            Err(_) => self.transport_errors += 1,
        }
    }

    fn add(&mut self, other: Self) -> Result<()> {
        self.head.add(&other.head)?;
        self.first.add(&other.first)?;
        self.event_delay.add(&other.event_delay)?;
        self.total.add(&other.total)?;
        self.started += other.started;
        self.completed += other.completed;
        self.cancelled += other.cancelled;
        self.bytes += other.bytes;
        self.integrity.add(other.integrity);
        self.transport_errors += other.transport_errors;
        Ok(())
    }
}

impl IntegrityRecord {
    fn add(&mut self, other: Self) {
        self.missing_events += other.missing_events;
        self.duplicate_events += other.duplicate_events;
        self.reordered_events += other.reordered_events;
        self.corrupt_events += other.corrupt_events;
        self.cross_stream_events += other.cross_stream_events;
        self.body_hash_mismatches += other.body_hash_mismatches;
        self.trailer_mismatches += other.trailer_mismatches;
        self.status_errors += other.status_errors;
    }

    fn error_count(&self) -> u64 {
        self.missing_events
            + self.duplicate_events
            + self.reordered_events
            + self.corrupt_events
            + self.cross_stream_events
            + self.body_hash_mismatches
            + self.trailer_mismatches
            + self.status_errors
    }
}

struct Observation {
    head: Duration,
    first: Duration,
    total: Duration,
    event_delays: Vec<u64>,
    bytes: usize,
    integrity: IntegrityRecord,
    cancelled: bool,
}

pub async fn run(options: LoadOptions) -> Result<()> {
    let environment = metadata::collect(&options.binaries)?;
    let clients = Clients::new()?;
    let request_sequence = Arc::new(AtomicU64::new(0));
    let mut scenarios = Vec::new();
    let mut validation_errors = Vec::new();

    for protocol in &options.matrix.protocols {
        for provider in &options.matrix.providers {
            for response_bytes in &options.matrix.response_bytes {
                for concurrency in &options.matrix.concurrency {
                    for target in &options.targets {
                        println!(
                            "benchmarking {} {} {} bytes={} concurrency={}",
                            target.topology, protocol, provider, response_bytes, concurrency
                        );
                        let scenario = run_sustained_scenario(
                            clients.get(*protocol),
                            target,
                            *protocol,
                            *provider,
                            *response_bytes,
                            *concurrency,
                            &options,
                            Arc::clone(&request_sequence),
                        )
                        .await?;
                        collect_validation_errors(&scenario, &mut validation_errors);
                        scenarios.push(scenario);
                    }
                }
            }
        }
    }

    if options.matrix.slow_streams > 0 {
        for protocol in &options.matrix.protocols {
            for target in &options.targets {
                println!(
                    "benchmarking slow capacity {} {} streams={}",
                    target.topology, protocol, options.matrix.slow_streams
                );
                let scenario = run_slow_scenario(
                    clients.get(*protocol),
                    target,
                    *protocol,
                    &options,
                    Arc::clone(&request_sequence),
                )
                .await?;
                collect_validation_errors(&scenario, &mut validation_errors);
                scenarios.push(scenario);
            }
        }
    }

    let report = Report {
        schema_version: 1,
        informational_only: true,
        environment,
        parameters: options.matrix,
        targets: options
            .targets
            .iter()
            .map(|target| TargetRecord {
                topology: target.topology,
                url: target.url.clone(),
                configured_header_names: target
                    .headers
                    .iter()
                    .map(|(name, _)| name.to_string())
                    .collect(),
            })
            .collect(),
        scenarios,
        validation_errors,
    };
    metadata::write_json(&options.output, &report)?;
    println!("daemon transport report: {}", options.output.display());
    ensure!(
        report.validation_errors.is_empty(),
        "transport correctness validation failed: {}",
        report.validation_errors.join("; ")
    );
    Ok(())
}

pub async fn run_smoke(output: std::path::PathBuf) -> Result<()> {
    let (url, stop, provider_task) = provider::spawn_ephemeral().await?;
    let result = run(LoadOptions::smoke(output, url)).await;
    let _ = stop.send(());
    provider_task
        .await
        .context("smoke provider task failed")??;
    result
}

#[allow(clippy::too_many_arguments)]
async fn run_sustained_scenario(
    client: &BenchmarkClient,
    target: &Target,
    protocol: Protocol,
    provider: Provider,
    response_bytes: usize,
    concurrency: usize,
    options: &LoadOptions,
    request_sequence: Arc<AtomicU64>,
) -> Result<ScenarioRecord> {
    if options.matrix.warmup_seconds > 0 {
        let _ = run_timed_phase(
            client,
            target,
            provider,
            response_bytes,
            options.matrix.events,
            concurrency,
            Duration::from_secs(options.matrix.warmup_seconds),
            options.matrix.event_delay_micros,
            0,
            Arc::clone(&request_sequence),
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
        )
        .await?;
    }

    let connections_before = client.connections();
    let active = Arc::new(AtomicUsize::new(0));
    let max_active = Arc::new(AtomicUsize::new(0));
    let sampler = ResourceSampler::start(options.processes.clone()).await;
    let started = Instant::now();
    let accumulator = run_timed_phase(
        client,
        target,
        provider,
        response_bytes,
        options.matrix.events,
        concurrency,
        Duration::from_secs(options.matrix.duration_seconds),
        options.matrix.event_delay_micros,
        options.matrix.cancel_every,
        request_sequence,
        Arc::clone(&active),
        Arc::clone(&max_active),
    )
    .await?;
    let elapsed = started.elapsed();
    let resources = sampler.finish(max_active.load(Ordering::Relaxed)).await;
    let connections = client.connections().saturating_sub(connections_before);
    Ok(make_scenario(
        "sustained",
        target.topology,
        protocol,
        provider,
        response_bytes,
        options.matrix.events,
        concurrency,
        Some(options.matrix.duration_seconds),
        elapsed,
        accumulator,
        connections,
        max_active.load(Ordering::Relaxed),
        resources,
    ))
}

async fn run_slow_scenario(
    client: &BenchmarkClient,
    target: &Target,
    protocol: Protocol,
    options: &LoadOptions,
    request_sequence: Arc<AtomicU64>,
) -> Result<ScenarioRecord> {
    let concurrency = options.matrix.slow_streams;
    let response_bytes = options.matrix.response_bytes[0];
    let connections_before = client.connections();
    let active = Arc::new(AtomicUsize::new(0));
    let max_active = Arc::new(AtomicUsize::new(0));
    let sampler = ResourceSampler::start(options.processes.clone()).await;
    let started = Instant::now();
    let mut tasks = Vec::with_capacity(concurrency);
    for _ in 0..concurrency {
        let client = client.clone();
        let target = target.clone();
        let request_sequence = Arc::clone(&request_sequence);
        let active = Arc::clone(&active);
        let max_active = Arc::clone(&max_active);
        let events = options.matrix.events;
        let delay = options.matrix.slow_event_delay_millis.saturating_mul(1_000);
        tasks.push(tokio::spawn(async move {
            let mut accumulator = Accumulator::new();
            accumulator.record(
                perform_request(
                    &client,
                    &target,
                    Provider::Openai,
                    response_bytes,
                    events,
                    delay,
                    false,
                    request_sequence,
                    active,
                    max_active,
                )
                .await,
            );
            accumulator
        }));
    }
    let mut accumulator = Accumulator::new();
    for task in tasks {
        accumulator.add(task.await.context("slow-stream load task failed")?)?;
    }
    let elapsed = started.elapsed();
    let resources = sampler.finish(max_active.load(Ordering::Relaxed)).await;
    let connections = client.connections().saturating_sub(connections_before);
    Ok(make_scenario(
        "slow-capacity",
        target.topology,
        protocol,
        Provider::Openai,
        response_bytes,
        options.matrix.events,
        concurrency,
        None,
        elapsed,
        accumulator,
        connections,
        max_active.load(Ordering::Relaxed),
        resources,
    ))
}

#[allow(clippy::too_many_arguments)]
async fn run_timed_phase(
    client: &BenchmarkClient,
    target: &Target,
    provider: Provider,
    response_bytes: usize,
    events: usize,
    concurrency: usize,
    duration: Duration,
    event_delay_micros: u64,
    cancel_every: usize,
    request_sequence: Arc<AtomicU64>,
    active: Arc<AtomicUsize>,
    max_active: Arc<AtomicUsize>,
) -> Result<Accumulator> {
    let barrier = Arc::new(tokio::sync::Barrier::new(concurrency));
    let mut tasks = Vec::with_capacity(concurrency);
    for _ in 0..concurrency {
        let client = client.clone();
        let target = target.clone();
        let barrier = Arc::clone(&barrier);
        let request_sequence = Arc::clone(&request_sequence);
        let active = Arc::clone(&active);
        let max_active = Arc::clone(&max_active);
        tasks.push(tokio::spawn(async move {
            barrier.wait().await;
            let deadline = Instant::now() + duration;
            let mut accumulator = Accumulator::new();
            while Instant::now() < deadline {
                let ordinal = request_sequence.fetch_add(1, Ordering::Relaxed);
                let cancel = cancel_every > 0 && ordinal.is_multiple_of(cancel_every as u64);
                accumulator.record(
                    perform_request(
                        &client,
                        &target,
                        provider,
                        response_bytes,
                        events,
                        event_delay_micros,
                        cancel,
                        Arc::clone(&request_sequence),
                        Arc::clone(&active),
                        Arc::clone(&max_active),
                    )
                    .await,
                );
            }
            accumulator
        }));
    }
    let mut accumulator = Accumulator::new();
    for task in tasks {
        accumulator.add(task.await.context("load task failed")?)?;
    }
    Ok(accumulator)
}

#[allow(clippy::too_many_arguments)]
async fn perform_request(
    client: &BenchmarkClient,
    target: &Target,
    provider: Provider,
    response_bytes: usize,
    events: usize,
    event_delay_micros: u64,
    cancel_after_first: bool,
    request_sequence: Arc<AtomicU64>,
    active: Arc<AtomicUsize>,
    max_active: Arc<AtomicUsize>,
) -> Result<Observation> {
    let stream_id = format!("{:016x}", request_sequence.fetch_add(1, Ordering::Relaxed));
    let uri: Uri = format!("{}{}", target.url, provider.path())
        .parse()
        .context("failed to construct target URI")?;
    let mut builder = Request::post(uri)
        .header(ACCEPT, "text/event-stream")
        .header(TE, "trailers")
        .header(CONTENT_TYPE, "application/json")
        .header(AUTHORIZATION, "Bearer benchmark-provider-token")
        .header(RESPONSE_BYTES, response_bytes)
        .header(EVENT_COUNT, events)
        .header(EVENT_DELAY_MICROS, event_delay_micros)
        .header(STREAM_ID, &stream_id);
    for (name, value) in &target.headers {
        builder = builder.header(name, value);
    }
    let body = match provider {
        Provider::Openai => Bytes::from_static(b"{\"model\":\"benchmark\",\"stream\":true,\"input\":\"ping\"}"),
        Provider::Anthropic => Bytes::from_static(
            b"{\"model\":\"benchmark\",\"stream\":true,\"max_tokens\":1024,\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}]}",
        ),
    };
    let request = builder
        .body(Full::new(body))
        .context("failed to build benchmark request")?;
    let started = Instant::now();
    let active_now = active.fetch_add(1, Ordering::Relaxed) + 1;
    max_active.fetch_max(active_now, Ordering::Relaxed);
    let _active_guard = ActiveGuard(active);
    let response = client
        .client
        .request(request)
        .await
        .context("request failed")?;
    let head = started.elapsed();
    let mut integrity = IntegrityRecord::default();
    if !response.status().is_success() {
        integrity.status_errors += 1;
    }
    let expected_events = response
        .headers()
        .get(EVENT_COUNT)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse().ok())
        .unwrap_or(events);
    let expected_stream = response
        .headers()
        .get(STREAM_ID)
        .and_then(|value| value.to_str().ok())
        .unwrap_or(&stream_id)
        .to_owned();
    let mut body = response.into_body();
    let mut tracker = EventTracker::new(expected_stream, expected_events);
    let mut hasher = Sha256::new();
    let mut bytes = 0;
    let mut first = None;
    let mut trailer_hash = None;
    let mut trailer_events = None;
    while let Some(frame) = body.frame().await {
        let frame = frame.context("response body failed")?;
        match frame.into_data() {
            Ok(data) => {
                bytes += data.len();
                hasher.update(&data);
                let before = tracker.unique_events();
                tracker.push(&data);
                if tracker.unique_events() > before && first.is_none() {
                    first = Some(started.elapsed());
                    if cancel_after_first {
                        return Ok(Observation {
                            head,
                            first: first.expect("first content recorded"),
                            total: started.elapsed(),
                            event_delays: Vec::new(),
                            bytes,
                            integrity: IntegrityRecord::default(),
                            cancelled: true,
                        });
                    }
                }
            }
            Err(frame) => {
                if let Ok(trailers) = frame.into_trailers() {
                    trailer_hash = trailers
                        .get(BODY_SHA256)
                        .and_then(|value| value.to_str().ok())
                        .map(str::to_owned);
                    trailer_events = trailers
                        .get(EVENT_COUNT)
                        .and_then(|value| value.to_str().ok())
                        .and_then(|value| value.parse::<usize>().ok());
                }
            }
        }
    }
    tracker.finish();
    integrity.add(tracker.integrity);
    let actual_hash = format!("{:x}", hasher.finalize());
    if trailer_hash.as_deref() != Some(&actual_hash) {
        integrity.body_hash_mismatches += 1;
    }
    if trailer_events != Some(expected_events) {
        integrity.trailer_mismatches += 1;
    }
    Ok(Observation {
        head,
        first: first.ok_or_else(|| anyhow!("response contained no content event"))?,
        total: started.elapsed(),
        event_delays: tracker.event_delays,
        bytes,
        integrity,
        cancelled: false,
    })
}

struct ActiveGuard(Arc<AtomicUsize>);

impl Drop for ActiveGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

struct EventTracker {
    buffer: Vec<u8>,
    expected_stream: String,
    expected_events: usize,
    seen: HashSet<usize>,
    last_sequence: Option<usize>,
    event_delays: Vec<u64>,
    integrity: IntegrityRecord,
}

impl EventTracker {
    fn new(expected_stream: String, expected_events: usize) -> Self {
        Self {
            buffer: Vec::new(),
            expected_stream,
            expected_events,
            seen: HashSet::with_capacity(expected_events),
            last_sequence: None,
            event_delays: Vec::with_capacity(expected_events),
            integrity: IntegrityRecord::default(),
        }
    }

    fn unique_events(&self) -> usize {
        self.seen.len()
    }

    fn push(&mut self, bytes: &[u8]) {
        self.buffer.extend_from_slice(bytes);
        while let Some(end) = self.buffer.windows(2).position(|window| window == b"\n\n") {
            let event = self.buffer.drain(..end + 2).collect::<Vec<_>>();
            self.parse_event(&event);
        }
    }

    fn parse_event(&mut self, event: &[u8]) {
        for line in event.split(|byte| *byte == b'\n') {
            let Some(data) = line.strip_prefix(b"data: ") else {
                continue;
            };
            if data == b"[DONE]" {
                continue;
            }
            let parsed = serde_json::from_slice::<Value>(data);
            let Ok(parsed) = parsed else {
                self.integrity.corrupt_events += 1;
                continue;
            };
            let Some(sequence) = parsed
                .get("s")
                .and_then(Value::as_u64)
                .map(|value| value as usize)
            else {
                self.integrity.corrupt_events += 1;
                continue;
            };
            if parsed.get("i").and_then(Value::as_str) != Some(&self.expected_stream) {
                self.integrity.cross_stream_events += 1;
            }
            if !self.seen.insert(sequence) {
                self.integrity.duplicate_events += 1;
            }
            if self.last_sequence.is_some_and(|last| sequence <= last) {
                self.integrity.reordered_events += 1;
            }
            self.last_sequence = Some(sequence);
            let emitted = parsed
                .get("t")
                .and_then(Value::as_str)
                .and_then(|value| value.parse::<u128>().ok());
            if let Some(emitted) = emitted {
                let delay = unix_time_nanos()
                    .saturating_sub(emitted)
                    .min(u64::MAX as u128) as u64;
                self.event_delays.push(delay.max(1));
            } else {
                self.integrity.corrupt_events += 1;
            }
        }
    }

    fn finish(&mut self) {
        if self.buffer.iter().any(|byte| !byte.is_ascii_whitespace()) {
            self.integrity.corrupt_events += 1;
        }
        self.integrity.missing_events +=
            self.expected_events.saturating_sub(self.seen.len()) as u64;
    }
}

#[allow(clippy::too_many_arguments)]
fn make_scenario(
    kind: &'static str,
    topology: Topology,
    protocol: Protocol,
    provider: Provider,
    response_bytes: usize,
    events: usize,
    concurrency: usize,
    configured_duration_seconds: Option<u64>,
    elapsed: Duration,
    accumulator: Accumulator,
    connections: u64,
    max_active: usize,
    resources: BTreeMap<String, ResourceRecord>,
) -> ScenarioRecord {
    let seconds = elapsed.as_secs_f64().max(f64::EPSILON);
    let pool_reuses = accumulator.started.saturating_sub(connections);
    ScenarioRecord {
        kind,
        topology,
        protocol,
        provider,
        response_bytes,
        events,
        concurrency,
        configured_duration_seconds,
        measured_duration_seconds: seconds,
        requests_started: accumulator.started,
        requests_completed: accumulator.completed,
        requests_cancelled: accumulator.cancelled,
        requests_per_second: accumulator.completed as f64 / seconds,
        goodput_mib_per_second: accumulator.bytes as f64 / (1024.0 * 1024.0) / seconds,
        response_head: HistogramRecord::from_histogram(&accumulator.head),
        first_content: HistogramRecord::from_histogram(&accumulator.first),
        per_event_forwarding_delay: HistogramRecord::from_histogram(&accumulator.event_delay),
        total: HistogramRecord::from_histogram(&accumulator.total),
        integrity: accumulator.integrity,
        transport_errors: accumulator.transport_errors,
        connections_opened_during_measurement: connections,
        estimated_pool_reuses: pool_reuses,
        reconnect_count: connections,
        max_active_http2_streams: matches!(protocol, Protocol::Http2).then_some(max_active),
        queued_bytes_peak: None,
        backpressure_stalls: None,
        resources,
        unavailable_metrics: vec![
            "queued byte depth requires daemon/worker instrumentation",
            "backpressure stall count requires daemon/worker instrumentation",
        ],
    }
}

fn collect_validation_errors(scenario: &ScenarioRecord, errors: &mut Vec<String>) {
    if scenario.integrity.error_count() > 0 || scenario.transport_errors > 0 {
        errors.push(format!(
            "{} {} {} bytes={} concurrency={} had {} integrity and {} transport errors",
            scenario.topology,
            scenario.protocol,
            scenario.provider,
            scenario.response_bytes,
            scenario.concurrency,
            scenario.integrity.error_count(),
            scenario.transport_errors
        ));
    }
}

fn latency_histogram() -> Histogram<u64> {
    Histogram::new_with_bounds(1, MAX_HISTOGRAM_NANOS, 3).expect("valid latency histogram")
}

fn record_duration(histogram: &mut Histogram<u64>, duration: Duration) {
    let value = duration.as_nanos().min(MAX_HISTOGRAM_NANOS as u128) as u64;
    let _ = histogram.record(value.max(1));
}

fn nanos_to_millis(value: u64) -> f64 {
    value as f64 / 1_000_000.0
}

fn unix_time_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}
