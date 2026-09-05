// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Benchmark-only orchestration for an authenticated, directly measured worker hop.
//!
//! Production workers deliberately expose neither their endpoint nor their daemon-to-worker
//! credential. This module starts an isolated real daemon and MCP, observes their loopback control
//! exchange without persisting either value, and adds the resulting worker target to the load
//! driver in memory. It never weakens the production worker's request authentication.

use std::convert::Infallible;
use std::net::{IpAddr, Ipv4Addr};
use std::path::Path;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail, ensure};
use base64::Engine as _;
use bytes::Bytes;
use http::header::{CONNECTION, CONTENT_LENGTH, HOST, TRANSFER_ENCODING};
use http::{HeaderName, HeaderValue, Method, Request, Response, StatusCode, Uri};
use http_body_util::{BodyExt as _, Full, Limited};
use hyper::body::Incoming;
use hyper::service::service_fn;
use hyper_util::client::legacy::Client;
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::rt::{TokioExecutor, TokioIo, TokioTimer};
use hyper_util::server::conn::auto::Builder as ServerBuilder;
use ring::rand::{SecureRandom as _, SystemRandom};
use serde_json::Value;
use tempfile::TempDir;
use tokio::net::{TcpListener, TcpStream};
use tokio::process::{Child, Command};
use tokio::sync::oneshot;

use crate::config::{LoadOptions, ProcessSpec, Target, Topology};

const WORKER_TOKEN_HEADER: &str = "x-nemo-relay-worker-token";
const CHALLENGE_PATH: &str = "/_nemo-relay/control/v1/challenge";
const MCP_REGISTER_PATH: &str = "/_nemo-relay/control/v1/mcp/register";
const MCP_HEARTBEAT_PATH: &str = "/_nemo-relay/control/v1/mcp/heartbeat";
const MCP_RELEASE_PATH: &str = "/_nemo-relay/control/v1/mcp/release";
const MCP_ACTIVATION_FAILED_PATH: &str = "/_nemo-relay/control/v1/mcp/activation-failed";
const WORKER_REGISTER_PATH: &str = "/_nemo-relay/control/v1/worker/register";
const WORKER_RECOVER_PATH: &str = "/_nemo-relay/control/v1/worker/recover";
const WORKER_READY_PATH: &str = "/_nemo-relay/control/v1/worker/ready";
const WORKER_HEARTBEAT_PATH: &str = "/_nemo-relay/control/v1/worker/heartbeat";
const MAX_CONTROL_BODY_BYTES: usize = 256 * 1024;
const STARTUP_TIMEOUT: Duration = Duration::from_secs(30);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(10);

type ProxyClient = Client<HttpConnector, Full<Bytes>>;

struct PendingWorker {
    worker_id: String,
    endpoint: String,
    data_token: String,
}

struct WorkerAccess {
    endpoint: String,
    data_token: String,
}

#[derive(Default)]
struct CaptureState {
    pending: Option<PendingWorker>,
    terminal: bool,
}

struct ProxyState {
    backend_origin: String,
    expected_worker_endpoint: String,
    client: ProxyClient,
    capture: Mutex<CaptureState>,
    ready: Mutex<Option<oneshot::Sender<Result<WorkerAccess, String>>>>,
}

/// Owns every process and temporary identity file created for one worker-only benchmark target.
pub struct WorkerHarness {
    daemon: Child,
    mcp: Child,
    proxy: tokio::task::JoinHandle<()>,
    state_dir: TempDir,
    endpoint: String,
    data_token: String,
    worker_port: u16,
    process_specs: Vec<ProcessSpec>,
}

impl WorkerHarness {
    /// Starts a real worker through the normal daemon/MCP activation path.
    pub async fn start(relay_binary: &Path, provider_url: &str) -> Result<Self> {
        ensure!(
            relay_binary.is_file(),
            "Relay binary does not exist: {}",
            relay_binary.display()
        );
        ensure_loopback_origin(provider_url, "provider")?;

        let daemon_port = reserve_loopback_port().await?;
        let worker_port = reserve_loopback_port().await?;
        ensure!(
            daemon_port != worker_port,
            "ephemeral port allocator returned a duplicate port"
        );
        let proxy_listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
            .await
            .context("failed to bind benchmark control proxy")?;
        let proxy_address = proxy_listener
            .local_addr()
            .context("failed to read benchmark control proxy address")?;
        let proxy_origin = format!("http://{proxy_address}");
        let backend_origin = format!("http://127.0.0.1:{daemon_port}");
        let expected_worker_endpoint = format!("http://127.0.0.1:{worker_port}");
        let state_dir = tempfile::Builder::new()
            .prefix("nemo-relay-daemon-benchmark-")
            .tempdir()
            .context("failed to create isolated benchmark state directory")?;
        let route_token = random_route_token()?;

        let (ready, wait_for_ready) = oneshot::channel();
        let proxy_state = Arc::new(ProxyState {
            backend_origin: backend_origin.clone(),
            expected_worker_endpoint: expected_worker_endpoint.clone(),
            client: proxy_client(),
            capture: Mutex::new(CaptureState::default()),
            ready: Mutex::new(Some(ready)),
        });
        let proxy = tokio::spawn(serve_control_proxy(
            proxy_listener,
            Arc::clone(&proxy_state),
        ));

        let mut daemon = relay_command(relay_binary, state_dir.path(), provider_url);
        daemon
            .arg("daemon")
            .arg("--bind")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(daemon_port.to_string())
            .arg("--advertise-address")
            .arg(&proxy_origin)
            .env("NEMO_RELAY_CLIENT_TOKEN", &route_token)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .kill_on_drop(true);
        let mut daemon = daemon
            .spawn()
            .with_context(|| format!("failed to launch {} daemon", relay_binary.display()))?;
        if let Err(error) = wait_for_listener(daemon_port, &mut daemon).await {
            proxy.abort();
            let _ = daemon.start_kill();
            return Err(error);
        }

        let mut mcp = relay_command(relay_binary, state_dir.path(), provider_url);
        mcp.arg("daemon")
            .arg("mcp")
            .arg("--daemon-address")
            .arg(&proxy_origin)
            .env("NEMO_RELAY_CLIENT_TOKEN", &route_token)
            .env("NEMO_RELAY_WORKER_PORT", worker_port.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .kill_on_drop(true);
        let mut mcp = match mcp.spawn() {
            Ok(child) => child,
            Err(error) => {
                proxy.abort();
                let _ = daemon.start_kill();
                return Err(error).context("failed to launch benchmark MCP");
            }
        };

        let access = match tokio::time::timeout(STARTUP_TIMEOUT, wait_for_ready).await {
            Ok(Ok(Ok(access))) => access,
            Ok(Ok(Err(error))) => {
                stop_child(&mut mcp).await;
                stop_child(&mut daemon).await;
                proxy.abort();
                bail!("benchmark control proxy rejected worker activation: {error}");
            }
            Ok(Err(_)) => {
                stop_child(&mut mcp).await;
                stop_child(&mut daemon).await;
                proxy.abort();
                bail!("benchmark control proxy stopped before worker readiness");
            }
            Err(_) => {
                let mcp_status = mcp.try_wait().context("failed to inspect benchmark MCP")?;
                stop_child(&mut mcp).await;
                stop_child(&mut daemon).await;
                proxy.abort();
                bail!(
                    "timed out waiting for authenticated worker readiness; MCP status: {mcp_status:?}"
                );
            }
        };
        ensure!(
            access.endpoint == expected_worker_endpoint,
            "worker registered an unexpected endpoint"
        );
        if let Err(error) = wait_for_worker_acceptance(&access).await {
            let mcp_cleanup = release_mcp(&mut mcp).await;
            let worker_cleanup = wait_for_worker_exit(worker_port).await;
            stop_child(&mut daemon).await;
            proxy.abort();
            if let Err(cleanup_error) = mcp_cleanup.and(worker_cleanup) {
                return Err(error.context(format!(
                    "benchmark harness cleanup also failed: {cleanup_error:#}"
                )));
            }
            return Err(error);
        }

        let mut process_specs = vec![
            ProcessSpec {
                name: "worker-harness-daemon".into(),
                pid: daemon.id().context("benchmark daemon has no process ID")?,
            },
            ProcessSpec {
                name: "worker-harness-mcp".into(),
                pid: mcp.id().context("benchmark MCP has no process ID")?,
            },
        ];
        if let Some(pid) = find_worker_pid(worker_port, &proxy_origin).await {
            process_specs.push(ProcessSpec {
                name: "worker".into(),
                pid,
            });
        }

        Ok(Self {
            daemon,
            mcp,
            proxy,
            state_dir,
            endpoint: access.endpoint,
            data_token: access.data_token,
            worker_port,
            process_specs,
        })
    }

    /// Adds the authenticated direct-worker target without exposing its credential to the caller.
    pub fn add_to(&self, options: &mut LoadOptions) -> Result<()> {
        ensure!(
            !options
                .targets
                .iter()
                .any(|target| target.topology == Topology::WorkerOnly),
            "worker-only target is already configured"
        );
        options.targets.push(Target {
            topology: Topology::WorkerOnly,
            url: self.endpoint.clone(),
            headers: vec![(
                HeaderName::from_static(WORKER_TOKEN_HEADER),
                HeaderValue::from_str(&self.data_token)
                    .context("daemon returned an invalid worker credential")?,
            )],
        });
        options.processes.extend(self.process_specs.clone());
        Ok(())
    }

    /// Gracefully releases the MCP reference, verifies the worker exits, and removes state.
    pub async fn shutdown(mut self) -> Result<()> {
        let _state_dir_lifetime = &self.state_dir;
        let mcp_result = release_mcp(&mut self.mcp).await;
        let worker_result = wait_for_worker_exit(self.worker_port).await;
        stop_child(&mut self.daemon).await;
        self.proxy.abort();
        mcp_result.and(worker_result)
    }
}

fn relay_command(binary: &Path, state_dir: &Path, provider_url: &str) -> Command {
    let mut command = Command::new(binary);
    command
        .arg("--openai-base-url")
        .arg(provider_url)
        .arg("--anthropic-base-url")
        .arg(provider_url)
        .env("XDG_CONFIG_HOME", state_dir)
        .env("XDG_CONFIG_DIRS", state_dir)
        .env("NEMO_RELAY_OPENAI_BASE_URL", provider_url)
        .env("NEMO_RELAY_ANTHROPIC_BASE_URL", provider_url);
    command
}

async fn reserve_loopback_port() -> Result<u16> {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
        .await
        .context("failed to reserve a loopback port")?;
    listener
        .local_addr()
        .map(|address| address.port())
        .context("failed to read reserved loopback port")
}

fn random_route_token() -> Result<String> {
    let mut bytes = [0_u8; 32];
    SystemRandom::new()
        .fill(&mut bytes)
        .map_err(|_| anyhow!("failed to generate benchmark route credential"))?;
    Ok(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes))
}

fn ensure_loopback_origin(origin: &str, label: &str) -> Result<()> {
    let uri = origin
        .parse::<Uri>()
        .with_context(|| format!("invalid {label} URL"))?;
    ensure!(
        uri.scheme_str() == Some("http"),
        "benchmark {label} must use loopback HTTP"
    );
    let authority = uri
        .authority()
        .with_context(|| format!("{label} URL has no authority"))?;
    let host = authority.host();
    let loopback = host
        .parse::<IpAddr>()
        .is_ok_and(|address| address.is_loopback());
    ensure!(loopback, "benchmark {label} must be loopback-only");
    ensure!(
        authority.port_u16().is_some(),
        "benchmark {label} needs an explicit port"
    );
    ensure!(
        uri.path_and_query()
            .is_none_or(|value| value.as_str() == "/"),
        "benchmark {label} URL cannot contain a path or query"
    );
    Ok(())
}

fn proxy_client() -> ProxyClient {
    let mut connector = HttpConnector::new();
    connector.enforce_http(true);
    connector.set_nodelay(true);
    let mut builder = Client::builder(TokioExecutor::new());
    builder.timer(TokioTimer::new());
    builder.pool_idle_timeout(Duration::from_secs(30));
    builder.build(connector)
}

async fn serve_control_proxy(listener: TcpListener, state: Arc<ProxyState>) {
    let mut connections = tokio::task::JoinSet::new();
    loop {
        tokio::select! {
            accepted = listener.accept() => {
                let Ok((stream, peer)) = accepted else {
                    fail_capture(&state, "control proxy listener failed".into());
                    return;
                };
                if !peer.ip().is_loopback() {
                    continue;
                }
                let service_state = Arc::clone(&state);
                connections.spawn(async move {
                    let service = service_fn(move |request| {
                        proxy_control_request(request, Arc::clone(&service_state))
                    });
                    let builder = ServerBuilder::new(TokioExecutor::new());
                    let _ = builder
                        .serve_connection(TokioIo::new(stream), service)
                        .await;
                });
            }
            Some(completed) = connections.join_next(), if !connections.is_empty() => {
                if completed.is_err() {
                    fail_capture(&state, "control proxy connection task failed".into());
                    return;
                }
            }
        }
    }
}

async fn proxy_control_request(
    request: Request<Incoming>,
    state: Arc<ProxyState>,
) -> Result<Response<Full<Bytes>>, Infallible> {
    let response = match proxy_control_request_inner(request, &state).await {
        Ok(response) => response,
        Err(error) => {
            fail_capture(&state, error.to_string());
            Response::builder()
                .status(StatusCode::BAD_GATEWAY)
                .body(Full::new(Bytes::from_static(
                    b"benchmark control proxy failure",
                )))
                .expect("static proxy error response")
        }
    };
    Ok(response)
}

async fn proxy_control_request_inner(
    request: Request<Incoming>,
    state: &ProxyState,
) -> Result<Response<Full<Bytes>>> {
    ensure!(
        request.method() == Method::POST,
        "control proxy accepts only POST"
    );
    let path = request.uri().path().to_owned();
    ensure!(
        allowed_control_path(&path),
        "control proxy rejected non-control path"
    );
    let path_and_query = request
        .uri()
        .path_and_query()
        .context("control request has no path")?
        .as_str()
        .to_owned();
    let (parts, body) = request.into_parts();
    let body = Limited::new(body, MAX_CONTROL_BODY_BYTES)
        .collect()
        .await
        .map_err(|error| {
            anyhow!("control request body failed or exceeded the benchmark proxy limit: {error}")
        })?
        .to_bytes();
    let request_json = if matches!(path.as_str(), WORKER_REGISTER_PATH | WORKER_READY_PATH) {
        Some(serde_json::from_slice::<Value>(&body).context("invalid worker control JSON")?)
    } else {
        None
    };

    let uri = format!("{}{path_and_query}", state.backend_origin)
        .parse::<Uri>()
        .context("failed to construct control backend URI")?;
    let mut forwarded = Request::builder()
        .method(parts.method)
        .version(parts.version)
        .uri(uri)
        .body(Full::new(body))
        .context("failed to build forwarded control request")?;
    copy_end_to_end_headers(&parts.headers, forwarded.headers_mut());
    let response = state
        .client
        .request(forwarded)
        .await
        .context("control backend request failed")?;
    let (response_parts, response_body) = response.into_parts();
    let response_body = Limited::new(response_body, MAX_CONTROL_BODY_BYTES)
        .collect()
        .await
        .map_err(|error| {
            anyhow!("control response body failed or exceeded the benchmark proxy limit: {error}")
        })?
        .to_bytes();

    if response_parts.status.is_success() {
        match path.as_str() {
            WORKER_REGISTER_PATH => capture_registration(
                state,
                request_json
                    .as_ref()
                    .context("missing registration request")?,
                &serde_json::from_slice(&response_body)
                    .context("invalid worker registration response")?,
            )?,
            WORKER_READY_PATH => capture_readiness(
                state,
                request_json.as_ref().context("missing readiness request")?,
            )?,
            _ => {}
        }
    }

    let mut rebuilt = Response::builder()
        .status(response_parts.status)
        .version(response_parts.version)
        .body(Full::new(response_body))
        .context("failed to rebuild control response")?;
    copy_end_to_end_headers(&response_parts.headers, rebuilt.headers_mut());
    Ok(rebuilt)
}

fn allowed_control_path(path: &str) -> bool {
    matches!(
        path,
        CHALLENGE_PATH
            | MCP_REGISTER_PATH
            | MCP_HEARTBEAT_PATH
            | MCP_RELEASE_PATH
            | MCP_ACTIVATION_FAILED_PATH
            | WORKER_REGISTER_PATH
            | WORKER_RECOVER_PATH
            | WORKER_READY_PATH
            | WORKER_HEARTBEAT_PATH
    )
}

fn copy_end_to_end_headers(source: &http::HeaderMap, destination: &mut http::HeaderMap) {
    for (name, value) in source {
        if !matches!(
            name,
            &CONNECTION | &CONTENT_LENGTH | &HOST | &TRANSFER_ENCODING
        ) {
            destination.append(name, value.clone());
        }
    }
}

fn capture_registration(state: &ProxyState, request: &Value, response: &Value) -> Result<()> {
    let worker_id = json_string(request, "/worker_id")?;
    let endpoint = json_string(request, "/endpoint")?;
    ensure!(
        endpoint == state.expected_worker_endpoint,
        "worker registration endpoint did not match the prescribed loopback port"
    );
    let data_token = json_string(response, "/data_token")?;
    let mut capture = lock(&state.capture);
    ensure!(
        !capture.terminal,
        "worker registered after terminal readiness"
    );
    if let Some(existing) = capture.pending.as_ref() {
        ensure!(
            existing.worker_id == worker_id
                && existing.endpoint == endpoint
                && existing.data_token == data_token,
            "worker registration retry changed authenticated values"
        );
    } else {
        capture.pending = Some(PendingWorker {
            worker_id,
            endpoint,
            data_token,
        });
    }
    Ok(())
}

fn capture_readiness(state: &ProxyState, request: &Value) -> Result<()> {
    let worker_id = json_string(request, "/payload/worker_id")?;
    let access = {
        let mut capture = lock(&state.capture);
        ensure!(!capture.terminal, "duplicate terminal worker readiness");
        let pending = capture
            .pending
            .take()
            .context("worker became ready before an authenticated registration response")?;
        ensure!(
            pending.worker_id == worker_id,
            "worker readiness ID did not match registration"
        );
        capture.terminal = true;
        WorkerAccess {
            endpoint: pending.endpoint,
            data_token: pending.data_token,
        }
    };
    let sender = lock(&state.ready)
        .take()
        .context("worker readiness was already reported")?;
    sender
        .send(Ok(access))
        .map_err(|_| anyhow!("worker readiness receiver was dropped"))
}

fn fail_capture(state: &ProxyState, error: String) {
    let should_send = {
        let mut capture = lock(&state.capture);
        if capture.terminal {
            false
        } else {
            capture.terminal = true;
            true
        }
    };
    if should_send && let Some(sender) = lock(&state.ready).take() {
        let _ = sender.send(Err(error));
    }
}

fn json_string(value: &Value, pointer: &str) -> Result<String> {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .with_context(|| format!("control JSON omitted {pointer}"))
}

async fn wait_for_listener(port: u16, child: &mut Child) -> Result<()> {
    let deadline = Instant::now() + STARTUP_TIMEOUT;
    loop {
        if let Some(status) = child
            .try_wait()
            .context("failed to inspect benchmark daemon")?
        {
            bail!("benchmark daemon exited before listening: {status}");
        }
        if TcpStream::connect((Ipv4Addr::LOCALHOST, port))
            .await
            .is_ok()
        {
            return Ok(());
        }
        ensure!(
            Instant::now() < deadline,
            "timed out waiting for benchmark daemon listener"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

async fn wait_for_worker_acceptance(access: &WorkerAccess) -> Result<()> {
    let client = proxy_client();
    let deadline = Instant::now() + STARTUP_TIMEOUT;
    loop {
        let uri = format!("{}/v1/responses", access.endpoint)
            .parse::<Uri>()
            .context("invalid captured worker endpoint")?;
        let request = Request::post(uri)
            .header(WORKER_TOKEN_HEADER, &access.data_token)
            .header("content-type", "application/json")
            .header("authorization", "Bearer benchmark-provider-token")
            .header("x-benchmark-response-bytes", 16 * 1024)
            .header("x-benchmark-event-count", 128)
            .body(Full::new(Bytes::from_static(
                b"{\"model\":\"benchmark\",\"stream\":true,\"input\":\"readiness\"}",
            )))
            .context("failed to build worker readiness request")?;
        match client.request(request).await {
            Ok(response) if response.status().is_success() => {
                response
                    .into_body()
                    .collect()
                    .await
                    .context("worker readiness response body failed")?;
                return Ok(());
            }
            Ok(response) if response.status() == StatusCode::SERVICE_UNAVAILABLE => {}
            Ok(response) => bail!(
                "worker readiness request returned HTTP {}",
                response.status()
            ),
            Err(_) => {}
        }
        ensure!(
            Instant::now() < deadline,
            "timed out waiting for worker request admission"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

async fn wait_for_worker_exit(port: u16) -> Result<()> {
    let deadline = Instant::now() + SHUTDOWN_TIMEOUT;
    loop {
        if TcpStream::connect((Ipv4Addr::LOCALHOST, port))
            .await
            .is_err()
        {
            return Ok(());
        }
        ensure!(
            Instant::now() < deadline,
            "benchmark worker did not exit after MCP release"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

async fn release_mcp(mcp: &mut Child) -> Result<()> {
    drop(mcp.stdin.take());
    match tokio::time::timeout(SHUTDOWN_TIMEOUT, mcp.wait()).await {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(error)) => Err(error).context("failed to wait for benchmark MCP"),
        Err(_) => {
            stop_child(mcp).await;
            bail!("benchmark MCP did not exit after its input was closed")
        }
    }
}

async fn stop_child(child: &mut Child) {
    if child.try_wait().ok().flatten().is_none() {
        let _ = child.start_kill();
    }
    let _ = tokio::time::timeout(Duration::from_secs(2), child.wait()).await;
}

#[cfg(unix)]
async fn find_worker_pid(port: u16, daemon_origin: &str) -> Option<u32> {
    let output = Command::new("ps")
        .args(["-axo", "pid=,command="])
        .output()
        .await
        .ok()?;
    let port = format!("--port {port}");
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .find_map(|line| {
            let trimmed = line.trim();
            let (pid, command) = trimmed.split_once(char::is_whitespace)?;
            (command.contains("daemon worker")
                && command.contains(&port)
                && command.contains(daemon_origin))
            .then(|| pid.parse().ok())
            .flatten()
        })
}

#[cfg(not(unix))]
async fn find_worker_pid(_port: u16, _daemon_origin: &str) -> Option<u32> {
    None
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|error| error.into_inner())
}
