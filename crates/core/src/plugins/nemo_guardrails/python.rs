// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::env;
use std::ffi::{OsStr, OsString};
use std::future::Future;
use std::io::{BufRead, BufReader, Write};
use std::pin::Pin;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc as std_mpsc};
use std::task::{Context, Poll};
use std::thread;
use std::time::Duration;

use serde::Deserialize;
use serde_json::json;
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

use crate::api::llm::LlmRequest;
use crate::api::runtime::{
    LlmExecutionFn, LlmJsonStream, LlmStreamExecutionFn, LlmStreamInner, ToolExecutionFn,
};
use crate::codec::request::{AnnotatedLlmRequest, Message, MessageContent};
use crate::codec::resolve::{ProviderSurface, request_codec, response_codec};
use crate::error::{FlowError, Result as FlowResult};
use crate::json::Json;
use crate::plugin::{PluginError, PluginRegistrationContext, Result as PluginResult};

use super::NeMoGuardrailsConfig;

#[cfg(not(windows))]
const DEFAULT_PYTHON_EXECUTABLE: &str = "python3";
#[cfg(windows)]
const DEFAULT_PYTHON_EXECUTABLE: &str = "python";
const PYTHON_EXECUTABLE_ENV: &str = "NEMO_RELAY_PYTHON";
const PYO3_PYTHON_ENV: &str = "PYO3_PYTHON";
const UV_PYTHON_ENV: &str = "UV_PYTHON";
const WORKER_INIT_TIMEOUT: Duration = Duration::from_secs(30);
const WORKER_RPC_TIMEOUT: Duration = Duration::from_secs(30);
const WORKER_SCRIPT: &str = include_str!("local_worker.py");

pub(super) fn register_local_backend(
    config: NeMoGuardrailsConfig,
    ctx: &mut PluginRegistrationContext,
) -> PluginResult<()> {
    let runtime = Arc::new(LocalGuardrailsRuntime::new(&config)?);

    if config.input || config.output {
        let llm_runtime = Arc::clone(&runtime);
        let enable_input = config.input;
        let enable_output = config.output;
        let llm_execution: LlmExecutionFn = Arc::new(move |_name, request, next| {
            let runtime = Arc::clone(&llm_runtime);
            Box::pin(async move {
                runtime
                    .execute_llm(request, next, enable_input, enable_output)
                    .await
            })
        });
        ctx.register_llm_execution_intercept(
            "nemo_guardrails_local",
            config.priority,
            llm_execution,
        )?;

        let stream_runtime = Arc::clone(&runtime);
        let enable_input = config.input;
        let enable_output = config.output;
        let llm_stream_execution: LlmStreamExecutionFn = Arc::new(move |_name, request, next| {
            let runtime = Arc::clone(&stream_runtime);
            Box::pin(async move {
                runtime
                    .execute_llm_stream(request, next, enable_input, enable_output)
                    .await
            })
        });
        ctx.register_llm_stream_execution_intercept(
            "nemo_guardrails_local_stream",
            config.priority,
            llm_stream_execution,
        )?;
    }

    if config.tool_input || config.tool_output {
        let tool_runtime = Arc::clone(&runtime);
        let enable_tool_input = config.tool_input;
        let enable_tool_output = config.tool_output;
        let tool_execution: ToolExecutionFn = Arc::new(move |tool_name, args, next| {
            let runtime = Arc::clone(&tool_runtime);
            let tool_name = tool_name.to_string();
            Box::pin(async move {
                let current_args = if enable_tool_input {
                    runtime.check_tool_input(&tool_name, &args).await?
                } else {
                    args
                };

                let tool_result = next(current_args.clone()).await?;
                let tool_result = if enable_tool_output {
                    runtime
                        .check_tool_output(&tool_name, &current_args, &tool_result)
                        .await?
                } else {
                    tool_result
                };
                Ok(tool_result.into())
            })
        });
        ctx.register_tool_execution_intercept(
            "nemo_guardrails_local",
            config.priority,
            tool_execution,
        )?;
    }

    Ok(())
}

struct LocalGuardrailsRuntime {
    bridge: LocalGuardrailsBridge,
    codec: Option<LocalGuardrailsCodec>,
}

impl LocalGuardrailsRuntime {
    fn new(config: &NeMoGuardrailsConfig) -> PluginResult<Self> {
        Ok(Self {
            bridge: LocalGuardrailsBridge::new(config)?,
            codec: resolve_codec(config)?,
        })
    }

    async fn execute_llm(
        &self,
        request: LlmRequest,
        next: crate::api::runtime::LlmExecutionNextFn,
        enable_input: bool,
        enable_output: bool,
    ) -> FlowResult<Json> {
        let (request, messages) = self.prepare_llm_request(request, enable_input).await?;
        let response = next(request).await?;

        if enable_output {
            let annotated_response = self.codec()?.decode_response(&response)?;
            if let Some(response_text) = annotated_response.response_text() {
                self.check_output_rails(&messages, response_text).await?;
            }
        }

        Ok(response)
    }

    async fn execute_llm_stream(
        &self,
        request: LlmRequest,
        next: crate::api::runtime::LlmStreamExecutionNextFn,
        enable_input: bool,
        enable_output: bool,
    ) -> FlowResult<LlmJsonStream> {
        let (request, messages) = self.prepare_llm_request(request, enable_input).await?;
        let provider_stream = next(request).await?;

        if !enable_output || !self.bridge.has_streaming_output_rails().await? {
            return Ok(provider_stream);
        }

        self.bridge.ensure_streaming_output_supported().await?;
        self.guard_provider_stream(messages, provider_stream).await
    }

    async fn prepare_llm_request(
        &self,
        request: LlmRequest,
        enable_input: bool,
    ) -> FlowResult<(LlmRequest, Vec<Json>)> {
        let codec = self.codec()?;
        let mut current_request = request;
        let mut annotated = codec.decode(&current_request)?;
        let mut messages = messages_from_annotated(&annotated)?;

        if enable_input {
            match self
                .bridge
                .check(messages.clone(), LocalRailKind::Input)
                .await?
            {
                LocalCheckOutcome::Passed => {}
                LocalCheckOutcome::Blocked { rail, .. } => {
                    return Err(blocked_error("input", rail.as_deref()));
                }
                LocalCheckOutcome::Modified { content, .. } => {
                    replace_last_role_content(&mut annotated, "user", content)?;
                    current_request = codec.encode(&annotated, &current_request)?;
                    messages = messages_from_annotated(&annotated)?;
                }
            }
        }

        Ok((current_request, messages))
    }

    async fn check_output_rails(&self, messages: &[Json], response_text: &str) -> FlowResult<()> {
        let mut output_messages = messages.to_vec();
        output_messages.push(json!({
            "role": "assistant",
            "content": response_text,
        }));

        match self
            .bridge
            .check(output_messages, LocalRailKind::Output)
            .await?
        {
            LocalCheckOutcome::Passed => Ok(()),
            LocalCheckOutcome::Blocked { rail, .. } => {
                Err(blocked_error("output", rail.as_deref()))
            }
            LocalCheckOutcome::Modified { .. } => Err(local_violation(
                "NeMo Guardrails output rail returned modified content, but the local backend \
                 does not rewrite provider responses yet.",
            )),
        }
    }

    async fn check_tool_input(&self, tool_name: &str, args: &Json) -> FlowResult<Json> {
        let messages = vec![json!({
            "role": "user",
            "content": tool_input_content(tool_name, args)?,
        })];

        match self.bridge.check(messages, LocalRailKind::Input).await? {
            LocalCheckOutcome::Passed => Ok(args.clone()),
            LocalCheckOutcome::Blocked { rail, .. } => {
                Err(blocked_error("tool_input", rail.as_deref()))
            }
            LocalCheckOutcome::Modified { content, .. } => {
                modified_tool_payload(&content, "arguments")
            }
        }
    }

    async fn check_tool_output(
        &self,
        tool_name: &str,
        args: &Json,
        result: &Json,
    ) -> FlowResult<Json> {
        let messages = vec![
            json!({
                "role": "user",
                "content": tool_input_content(tool_name, args)?,
            }),
            json!({
                "role": "assistant",
                "content": tool_output_content(tool_name, args, result)?,
            }),
        ];

        match self.bridge.check(messages, LocalRailKind::Output).await? {
            LocalCheckOutcome::Passed => Ok(result.clone()),
            LocalCheckOutcome::Blocked { rail, .. } => {
                Err(blocked_error("tool_output", rail.as_deref()))
            }
            LocalCheckOutcome::Modified { content, .. } => {
                modified_tool_payload(&content, "result")
            }
        }
    }

    async fn guard_provider_stream(
        &self,
        messages: Vec<Json>,
        provider_stream: LlmJsonStream,
    ) -> FlowResult<LlmJsonStream> {
        let (text_tx, text_rx) = mpsc::channel::<Option<String>>(32);
        let (chunk_tx, chunk_rx) = mpsc::channel::<FlowResult<Json>>(32);
        let blocked = Arc::new(Mutex::new(None));
        let monitor = self
            .bridge
            .spawn_stream_monitor(messages, text_rx, Arc::clone(&blocked))?;
        let codec = *self.codec()?;

        let (cancel, cancel_rx) = watch::channel(false);
        let (closed, closed_rx) = watch::channel(None);
        tokio::spawn(async move {
            forward_guarded_provider_stream(
                provider_stream,
                codec,
                text_tx,
                chunk_tx,
                monitor,
                blocked,
                cancel_rx,
                closed,
            )
            .await;
        });

        Ok(LlmJsonStream::from_closeable(GuardedProviderStream {
            receiver: ReceiverStream::new(chunk_rx),
            cancel,
            closed: closed_rx,
        }))
    }

    fn codec(&self) -> FlowResult<&LocalGuardrailsCodec> {
        self.codec.as_ref().ok_or_else(|| {
            FlowError::Internal(
                "local NeMo Guardrails backend requires a supported codec".to_string(),
            )
        })
    }
}

struct LocalGuardrailsBridge {
    worker: Arc<LocalGuardrailsWorker>,
}

impl LocalGuardrailsBridge {
    fn new(config: &NeMoGuardrailsConfig) -> PluginResult<Self> {
        Ok(Self {
            worker: LocalGuardrailsWorker::start(config)?,
        })
    }

    async fn check(
        &self,
        messages: Vec<Json>,
        kind: LocalRailKind,
    ) -> FlowResult<LocalCheckOutcome> {
        let result = self
            .worker
            .request(json!({
                "command": "check",
                "messages": messages,
                "rail_type": kind.as_str(),
            }))
            .await?;
        parse_check_result(result)
    }

    async fn has_streaming_output_rails(&self) -> FlowResult<bool> {
        let result = self
            .worker
            .request(json!({ "command": "has_streaming_output_rails" }))
            .await?;
        result
            .get("enabled")
            .and_then(Json::as_bool)
            .ok_or_else(|| FlowError::Internal("worker returned invalid streaming probe".into()))
    }

    async fn ensure_streaming_output_supported(&self) -> FlowResult<()> {
        self.worker
            .request(json!({ "command": "ensure_streaming_output_supported" }))
            .await
            .map(|_| ())
    }

    fn spawn_stream_monitor(
        &self,
        messages: Vec<Json>,
        text_rx: mpsc::Receiver<Option<String>>,
        blocked: Arc<Mutex<Option<String>>>,
    ) -> FlowResult<JoinHandle<FlowResult<()>>> {
        let (stream_id, event_rx) = self.worker.start_stream(messages)?;
        let worker = Arc::clone(&self.worker);
        Ok(tokio::spawn(async move {
            monitor_guardrails_stream(worker, stream_id, text_rx, event_rx, blocked).await
        }))
    }
}

struct LocalGuardrailsWorker {
    writer: Mutex<Option<WorkerCommandWriter>>,
    child: Mutex<Child>,
    waiters: Arc<Mutex<HashMap<String, std_mpsc::Sender<WorkerEnvelope>>>>,
    stream_events: Arc<Mutex<HashMap<String, mpsc::UnboundedSender<WorkerEnvelope>>>>,
    next_id: AtomicU64,
    shutdown_started: AtomicBool,
}

impl LocalGuardrailsWorker {
    fn start(config: &NeMoGuardrailsConfig) -> PluginResult<Arc<Self>> {
        log::info!(
            target: "nemo_relay.worker",
            event = "worker_starting",
            plugin_id = "nemo_guardrails";
            "NeMo Guardrails local worker is starting"
        );
        let python = python_executable(config);
        let mut command = Command::new(&python);
        command
            .arg("-u")
            .arg("-c")
            .arg(WORKER_SCRIPT)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        if let Some(python_path) = worker_python_path(config) {
            command.env("PYTHONPATH", python_path);
        }

        let mut child = command.spawn().map_err(|err| {
            PluginError::RegistrationFailed(format!(
                "failed to start NeMo Guardrails local Python worker with {python:?}: {err}"
            ))
        })?;
        let stdin = child.stdin.take().ok_or_else(|| {
            PluginError::RegistrationFailed(
                "failed to open stdin for NeMo Guardrails local Python worker".to_string(),
            )
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            PluginError::RegistrationFailed(
                "failed to open stdout for NeMo Guardrails local Python worker".to_string(),
            )
        })?;

        let worker = Arc::new(Self {
            writer: Mutex::new(Some(WorkerCommandWriter::spawn(stdin))),
            child: Mutex::new(child),
            waiters: Arc::new(Mutex::new(HashMap::new())),
            stream_events: Arc::new(Mutex::new(HashMap::new())),
            next_id: AtomicU64::new(1),
            shutdown_started: AtomicBool::new(false),
        });
        worker.spawn_reader(stdout);
        worker.initialize(config)?;
        log::info!(
            target: "nemo_relay.plugin",
            event = "plugin_resource_access_validated",
            plugin_kind = "nemo_guardrails",
            resource_kind = "python_worker",
            permission = "execute";
            "Plugin resource access validated"
        );
        log::info!(
            target: "nemo_relay.worker",
            event = "worker_connected",
            plugin_id = "nemo_guardrails";
            "NeMo Guardrails local worker connected"
        );
        Ok(worker)
    }

    fn spawn_reader(&self, stdout: ChildStdout) {
        let waiters = Arc::clone(&self.waiters);
        let stream_events = Arc::clone(&self.stream_events);
        thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                let line = match line {
                    Ok(line) => line,
                    Err(err) => {
                        notify_worker_closed(&waiters, &stream_events, err.to_string());
                        return;
                    }
                };
                if line.trim().is_empty() {
                    continue;
                }
                let envelope = match serde_json::from_str::<WorkerEnvelope>(&line) {
                    Ok(envelope) => envelope,
                    Err(err) => {
                        notify_worker_closed(
                            &waiters,
                            &stream_events,
                            format!("invalid worker response: {err}"),
                        );
                        return;
                    }
                };
                dispatch_worker_envelope(&waiters, &stream_events, envelope);
            }
            notify_worker_closed(&waiters, &stream_events, "worker exited".to_string());
        });
    }

    fn initialize(&self, config: &NeMoGuardrailsConfig) -> PluginResult<()> {
        let response = self
            .request_blocking(
                json!({
                    "command": "init",
                    "config": config,
                }),
                WORKER_INIT_TIMEOUT,
            )
            .map_err(|err| PluginError::RegistrationFailed(err.to_string()))?;
        if response.ok {
            Ok(())
        } else {
            Err(PluginError::RegistrationFailed(
                response
                    .error
                    .unwrap_or_else(|| "NeMo Guardrails local Python worker failed".to_string()),
            ))
        }
    }

    async fn request(&self, payload: Json) -> FlowResult<Json> {
        self.request_with_timeout(payload, WORKER_RPC_TIMEOUT).await
    }

    async fn request_with_timeout(&self, mut payload: Json, timeout: Duration) -> FlowResult<Json> {
        let receiver = self.send_request(&mut payload)?;
        let response_task = tokio::task::spawn_blocking(move || receiver.recv());
        let envelope = match tokio::time::timeout(timeout, response_task).await {
            Ok(result) => result
                .map_err(|err| FlowError::Internal(format!("worker response task failed: {err}")))?
                .map_err(|err| {
                    FlowError::Internal(format!("worker response channel closed: {err}"))
                })?,
            Err(_) => {
                log::error!(
                    target: "nemo_relay.worker",
                    event = "worker_failed",
                    plugin_id = "nemo_guardrails",
                    reason = "request_timeout";
                    "NeMo Guardrails local worker request timed out"
                );
                self.shutdown();
                return Err(FlowError::Internal(format!(
                    "worker request timed out after {} seconds",
                    timeout.as_secs()
                )));
            }
        };
        worker_result(envelope)
    }

    fn request_blocking(&self, mut payload: Json, timeout: Duration) -> FlowResult<WorkerEnvelope> {
        let receiver = self.send_request(&mut payload)?;
        receiver
            .recv_timeout(timeout)
            .map_err(|err| FlowError::Internal(format!("worker did not initialize: {err}")))
    }

    fn send_request(&self, payload: &mut Json) -> FlowResult<std_mpsc::Receiver<WorkerEnvelope>> {
        let id = self.next_request_id();
        set_request_id(payload, &id)?;
        let (tx, rx) = std_mpsc::channel();
        self.waiters
            .lock()
            .map_err(|err| FlowError::Internal(format!("worker waiter lock poisoned: {err}")))?
            .insert(id.clone(), tx);
        if let Err(err) = self.write_command(payload) {
            let _ = self.waiters.lock().map(|mut waiters| waiters.remove(&id));
            return Err(err);
        }
        Ok(rx)
    }

    fn start_stream(
        &self,
        messages: Vec<Json>,
    ) -> FlowResult<(String, mpsc::UnboundedReceiver<WorkerEnvelope>)> {
        let id = self.next_request_id();
        let (tx, rx) = mpsc::unbounded_channel();
        self.stream_events
            .lock()
            .map_err(|err| FlowError::Internal(format!("worker stream lock poisoned: {err}")))?
            .insert(id.clone(), tx);
        let payload = json!({
            "id": id,
            "command": "stream_start",
            "messages": messages,
        });
        if let Err(err) = self.write_command(&payload) {
            self.forget_stream(&id);
            return Err(err);
        }
        Ok((id, rx))
    }

    fn send_stream_text(&self, id: &str, text: String) -> FlowResult<()> {
        self.write_command(&json!({
            "id": id,
            "command": "stream_text",
            "text": text,
        }))
    }

    fn send_stream_end(&self, id: &str) -> FlowResult<()> {
        self.write_command(&json!({
            "id": id,
            "command": "stream_end",
        }))
    }

    fn forget_stream(&self, id: &str) {
        let _ = self
            .stream_events
            .lock()
            .map(|mut streams| streams.remove(id));
    }

    fn next_request_id(&self) -> String {
        self.next_id.fetch_add(1, Ordering::Relaxed).to_string()
    }

    fn write_command(&self, payload: &Json) -> FlowResult<()> {
        let line = serde_json::to_string(payload).map_err(|err| {
            FlowError::Internal(format!("failed to serialize worker command: {err}"))
        })?;
        let writer = self
            .writer
            .lock()
            .map_err(|err| FlowError::Internal(format!("worker writer lock poisoned: {err}")))?;
        writer
            .as_ref()
            .ok_or_else(|| FlowError::Internal("worker command writer is closed".to_string()))?
            .send(line)
    }

    fn shutdown(&self) {
        if self.shutdown_started.swap(true, Ordering::AcqRel) {
            return;
        }
        log::info!(
            target: "nemo_relay.worker",
            event = "worker_stopping",
            plugin_id = "nemo_guardrails";
            "NeMo Guardrails local worker is stopping"
        );
        let writer = self.writer.lock().ok().and_then(|mut writer| writer.take());
        let mut cleanup_succeeded = true;
        if let Ok(mut child) = self.child.lock() {
            if child.kill().is_err() {
                cleanup_succeeded = false;
                log::warn!(
                    target: "nemo_relay.worker",
                    event = "worker_cleanup_failed",
                    plugin_id = "nemo_guardrails",
                    operation = "kill";
                    "NeMo Guardrails local worker cleanup failed"
                );
            }
            if child.wait().is_err() {
                cleanup_succeeded = false;
                log::warn!(
                    target: "nemo_relay.worker",
                    event = "worker_cleanup_failed",
                    plugin_id = "nemo_guardrails",
                    operation = "wait";
                    "NeMo Guardrails local worker cleanup failed"
                );
            }
        }
        if let Some(writer) = writer {
            writer.join();
        }
        if cleanup_succeeded {
            log::info!(
                target: "nemo_relay.worker",
                event = "worker_stopped",
                plugin_id = "nemo_guardrails";
                "NeMo Guardrails local worker stopped"
            );
        }
    }
}

impl Drop for LocalGuardrailsWorker {
    fn drop(&mut self) {
        self.shutdown();
    }
}

struct WorkerCommandWriter {
    sender: std_mpsc::Sender<String>,
    error: Arc<Mutex<Option<String>>>,
    handle: Option<thread::JoinHandle<()>>,
}

impl WorkerCommandWriter {
    fn spawn(mut stdin: ChildStdin) -> Self {
        let (sender, receiver) = std_mpsc::channel::<String>();
        let error = Arc::new(Mutex::new(None));
        let writer_error = Arc::clone(&error);
        let handle = thread::spawn(move || {
            for line in receiver {
                if let Err(err) = writeln!(stdin, "{line}").and_then(|_| stdin.flush()) {
                    if let Ok(mut stored_error) = writer_error.lock() {
                        *stored_error = Some(err.to_string());
                    }
                    return;
                }
            }
            let _ = stdin.flush();
        });
        Self {
            sender,
            error,
            handle: Some(handle),
        }
    }

    fn send(&self, line: String) -> FlowResult<()> {
        if let Some(error) = self
            .error
            .lock()
            .map_err(|err| {
                FlowError::Internal(format!("worker writer error lock poisoned: {err}"))
            })?
            .clone()
        {
            return Err(FlowError::Internal(format!(
                "failed to write worker command: {error}"
            )));
        }
        self.sender.send(line).map_err(|err| {
            FlowError::Internal(format!("worker command writer channel closed: {err}"))
        })
    }

    fn join(mut self) {
        drop(self.sender);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct WorkerEnvelope {
    id: String,
    ok: bool,
    #[serde(default)]
    result: Option<Json>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    event: Option<String>,
    #[serde(default)]
    message: Option<String>,
}

#[derive(Deserialize)]
struct WorkerCheckResult {
    status: String,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    rail: Option<String>,
}

fn python_executable(config: &NeMoGuardrailsConfig) -> String {
    config
        .local
        .as_ref()
        .and_then(|local| local.python_executable.as_deref())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .or_else(|| env_executable(PYTHON_EXECUTABLE_ENV))
        .or_else(|| env_executable(PYO3_PYTHON_ENV))
        .or_else(|| env_executable(UV_PYTHON_ENV))
        .unwrap_or_else(|| DEFAULT_PYTHON_EXECUTABLE.to_string())
}

fn env_executable(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn python_path(config: &NeMoGuardrailsConfig) -> Option<String> {
    config
        .local
        .as_ref()
        .and_then(|local| local.python_path.as_deref())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn worker_python_path(config: &NeMoGuardrailsConfig) -> Option<OsString> {
    let configured = python_path(config)?;
    merge_python_path(
        OsStr::new(&configured),
        env::var_os("PYTHONPATH").as_deref(),
    )
}

fn merge_python_path(configured: &OsStr, inherited: Option<&OsStr>) -> Option<OsString> {
    let mut paths = env::split_paths(configured).collect::<Vec<_>>();
    if let Some(inherited) = inherited.filter(|value| !value.is_empty()) {
        paths.extend(env::split_paths(inherited));
    }
    env::join_paths(paths).ok()
}

fn set_request_id(payload: &mut Json, id: &str) -> FlowResult<()> {
    let object = payload.as_object_mut().ok_or_else(|| {
        FlowError::Internal("worker command payload must be a JSON object".to_string())
    })?;
    object.insert("id".to_string(), Json::String(id.to_string()));
    Ok(())
}

fn dispatch_worker_envelope(
    waiters: &Arc<Mutex<HashMap<String, std_mpsc::Sender<WorkerEnvelope>>>>,
    stream_events: &Arc<Mutex<HashMap<String, mpsc::UnboundedSender<WorkerEnvelope>>>>,
    envelope: WorkerEnvelope,
) {
    if envelope.event.is_some() {
        let sender = stream_events
            .lock()
            .ok()
            .and_then(|streams| streams.get(&envelope.id).cloned());
        if let Some(sender) = sender {
            let _ = sender.send(envelope);
        }
        return;
    }

    let sender = waiters
        .lock()
        .ok()
        .and_then(|mut waiters| waiters.remove(&envelope.id));
    if let Some(sender) = sender {
        let _ = sender.send(envelope);
    }
}

fn notify_worker_closed(
    waiters: &Arc<Mutex<HashMap<String, std_mpsc::Sender<WorkerEnvelope>>>>,
    stream_events: &Arc<Mutex<HashMap<String, mpsc::UnboundedSender<WorkerEnvelope>>>>,
    message: String,
) {
    if let Ok(mut waiters) = waiters.lock() {
        for (id, sender) in waiters.drain() {
            let _ = sender.send(WorkerEnvelope {
                id,
                ok: false,
                result: None,
                error: Some(message.clone()),
                event: None,
                message: None,
            });
        }
    }
    if let Ok(mut streams) = stream_events.lock() {
        for (id, sender) in streams.drain() {
            let _ = sender.send(WorkerEnvelope {
                id,
                ok: false,
                result: None,
                error: Some(message.clone()),
                event: Some("error".to_string()),
                message: None,
            });
        }
    }
}

fn worker_result(envelope: WorkerEnvelope) -> FlowResult<Json> {
    if envelope.ok {
        Ok(envelope.result.unwrap_or(Json::Null))
    } else {
        Err(FlowError::Internal(envelope.error.unwrap_or_else(|| {
            "NeMo Guardrails local Python worker failed".to_string()
        })))
    }
}

fn parse_check_result(result: Json) -> FlowResult<LocalCheckOutcome> {
    let result: WorkerCheckResult = serde_json::from_value(result).map_err(|err| {
        FlowError::Internal(format!("worker returned invalid check result: {err}"))
    })?;
    match result.status.as_str() {
        "blocked" => Ok(LocalCheckOutcome::Blocked { rail: result.rail }),
        "modified" => Ok(LocalCheckOutcome::Modified {
            content: result.content.unwrap_or_default(),
        }),
        "passed" => Ok(LocalCheckOutcome::Passed),
        unexpected => Err(FlowError::Internal(format!(
            "unexpected worker check status: {unexpected}"
        ))),
    }
}

#[derive(Clone, Copy)]
enum LocalGuardrailsCodec {
    OpenAIChat,
    OpenAIResponses,
    AnthropicMessages,
}

impl LocalGuardrailsCodec {
    fn provider_surface(self) -> ProviderSurface {
        match self {
            Self::OpenAIChat => ProviderSurface::OpenAIChat,
            Self::OpenAIResponses => ProviderSurface::OpenAIResponses,
            Self::AnthropicMessages => ProviderSurface::AnthropicMessages,
        }
    }

    fn from_provider_surface(surface: ProviderSurface) -> Self {
        match surface {
            ProviderSurface::OpenAIChat => Self::OpenAIChat,
            ProviderSurface::OpenAIResponses => Self::OpenAIResponses,
            ProviderSurface::AnthropicMessages => Self::AnthropicMessages,
        }
    }

    fn decode(&self, request: &LlmRequest) -> FlowResult<AnnotatedLlmRequest> {
        request_codec(self.provider_surface()).decode(request)
    }

    fn encode(
        &self,
        annotated: &AnnotatedLlmRequest,
        original: &LlmRequest,
    ) -> FlowResult<LlmRequest> {
        request_codec(self.provider_surface()).encode(annotated, original)
    }

    fn decode_response(
        &self,
        response: &Json,
    ) -> FlowResult<crate::codec::response::AnnotatedLlmResponse> {
        response_codec(self.provider_surface()).decode_response(response)
    }
}

fn resolve_codec(config: &NeMoGuardrailsConfig) -> PluginResult<Option<LocalGuardrailsCodec>> {
    if !(config.input || config.output) {
        return Ok(None);
    }

    match config.codec.as_deref() {
        Some(name) => match ProviderSurface::from_codec_name(name) {
            Some(surface) => Ok(Some(LocalGuardrailsCodec::from_provider_surface(surface))),
            None => Err(PluginError::InvalidConfig(format!(
                "unsupported local NeMo Guardrails codec '{name}'"
            ))),
        },
        None => Err(PluginError::InvalidConfig(
            "local NeMo Guardrails backend requires a supported codec".to_string(),
        )),
    }
}

enum LocalCheckOutcome {
    Passed,
    Blocked { rail: Option<String> },
    Modified { content: String },
}

#[derive(Clone, Copy)]
enum LocalRailKind {
    Input,
    Output,
}

impl LocalRailKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::Output => "output",
        }
    }
}

fn messages_from_annotated(annotated: &AnnotatedLlmRequest) -> FlowResult<Vec<Json>> {
    match serde_json::to_value(&annotated.messages)
        .map_err(|err| FlowError::Internal(format!("failed to serialize messages: {err}")))?
    {
        Json::Array(messages) => Ok(messages),
        _ => Err(FlowError::Internal(
            "serialized messages were not a JSON array".to_string(),
        )),
    }
}

fn replace_last_role_content(
    annotated: &mut AnnotatedLlmRequest,
    role: &str,
    content: String,
) -> FlowResult<()> {
    for message in annotated.messages.iter_mut().rev() {
        match (role, message) {
            (
                "user",
                Message::User {
                    content: target, ..
                },
            ) => {
                *target = MessageContent::Text(content);
                return Ok(());
            }
            (
                "assistant",
                Message::Assistant {
                    content: target, ..
                },
            ) => {
                *target = Some(MessageContent::Text(content));
                return Ok(());
            }
            _ => {}
        }
    }

    Err(local_violation(format!(
        "NeMo Guardrails returned modified {role} content but no {role} message was present."
    )))
}

fn tool_input_content(name: &str, args: &Json) -> FlowResult<String> {
    serde_json::to_string(&json!({
        "tool_name": name,
        "arguments": args,
    }))
    .map_err(|err| FlowError::Internal(format!("failed to serialize tool input: {err}")))
}

fn tool_output_content(name: &str, args: &Json, result: &Json) -> FlowResult<String> {
    serde_json::to_string(&json!({
        "tool_name": name,
        "arguments": args,
        "result": result,
    }))
    .map_err(|err| FlowError::Internal(format!("failed to serialize tool output: {err}")))
}

fn modified_tool_payload(content: &str, field: &str) -> FlowResult<Json> {
    let value: Json = serde_json::from_str(content).map_err(|_| {
        local_violation(format!(
            "NeMo Guardrails returned modified tool {field} content that is not valid JSON."
        ))
    })?;

    let Json::Object(object) = value else {
        return Err(local_violation(format!(
            "NeMo Guardrails returned modified tool {field} content without a '{field}' field."
        )));
    };
    object.get(field).cloned().ok_or_else(|| {
        local_violation(format!(
            "NeMo Guardrails returned modified tool {field} content without a '{field}' field."
        ))
    })
}

fn blocked_error(rail_type: &str, rail: Option<&str>) -> FlowError {
    let detail = rail
        .filter(|rail| !rail.is_empty())
        .map(|rail| format!(" by rail '{rail}'"))
        .unwrap_or_default();
    let subject = if matches!(rail_type, "input" | "output") {
        "LLM call"
    } else {
        "tool call"
    };
    local_violation(format!(
        "NeMo Guardrails {rail_type} rail blocked the {subject}{detail}."
    ))
}

fn local_violation(message: impl Into<String>) -> FlowError {
    FlowError::Internal(message.into())
}

struct GuardedProviderStream {
    receiver: ReceiverStream<FlowResult<Json>>,
    cancel: watch::Sender<bool>,
    closed: watch::Receiver<Option<std::result::Result<(), String>>>,
}

impl tokio_stream::Stream for GuardedProviderStream {
    type Item = FlowResult<Json>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.receiver).poll_next(cx)
    }
}

impl Drop for GuardedProviderStream {
    fn drop(&mut self) {
        self.cancel.send_replace(true);
    }
}

impl LlmStreamInner for GuardedProviderStream {
    fn close(self: Pin<&mut Self>) -> Pin<Box<dyn Future<Output = FlowResult<()>> + Send + '_>> {
        let this = self.get_mut();
        this.cancel.send_replace(true);
        let mut closed = this.closed.clone();
        Box::pin(async move {
            while closed.borrow().is_none() {
                closed.changed().await.map_err(|_| {
                    FlowError::Internal("guarded stream cleanup task ended early".into())
                })?;
            }
            closed
                .borrow()
                .clone()
                .expect("close state checked above")
                .map_err(FlowError::Internal)
        })
    }
}

#[allow(clippy::too_many_arguments)]
async fn forward_guarded_provider_stream(
    mut provider_stream: LlmJsonStream,
    codec: LocalGuardrailsCodec,
    text_tx: mpsc::Sender<Option<String>>,
    chunk_tx: mpsc::Sender<FlowResult<Json>>,
    monitor: JoinHandle<FlowResult<()>>,
    blocked: Arc<Mutex<Option<String>>>,
    mut cancel: watch::Receiver<bool>,
    closed: watch::Sender<Option<std::result::Result<(), String>>>,
) {
    let mut monitor = Some(monitor);
    loop {
        if *cancel.borrow() {
            break;
        }
        let item = tokio::select! {
            _ = cancel.changed() => break,
            item = provider_stream.next() => item,
        };
        let Some(item) = item else {
            break;
        };
        let chunk = match item {
            Ok(chunk) => chunk,
            Err(err) => {
                let _ = chunk_tx.send(Err(err)).await;
                let _ = text_tx.send(None).await;
                let _ = monitor.take().expect("monitor available").await;
                break;
            }
        };

        if let Some(message) = blocked_message(&blocked) {
            let _ = chunk_tx.send(Err(streaming_output_blocked(message))).await;
            let _ = text_tx.send(None).await;
            let _ = monitor.take().expect("monitor available").await;
            break;
        }
        if let Some(text) = extract_stream_text(codec, &chunk)
            && text_tx.send(Some(text)).await.is_err()
        {
            send_stream_monitor_error(
                monitor.take().expect("monitor available"),
                &chunk_tx,
                &blocked,
            )
            .await;
            break;
        }

        let sent = tokio::select! {
            _ = cancel.changed() => break,
            sent = chunk_tx.send(Ok(chunk)) => sent,
        };
        if sent.is_err() {
            let _ = text_tx.send(None).await;
            let _ = monitor.take().expect("monitor available").await;
            break;
        }
    }
    let _ = text_tx.send(None).await;
    if *cancel.borrow() {
        if let Some(monitor) = monitor.take() {
            monitor.abort();
        }
    } else if let Some(monitor) = monitor.take() {
        let _ = send_stream_monitor_error(monitor, &chunk_tx, &blocked).await;
    }
    closed.send_replace(Some(
        provider_stream
            .close()
            .await
            .map_err(|error| error.to_string()),
    ));
}

async fn send_stream_monitor_error(
    monitor: JoinHandle<FlowResult<()>>,
    chunk_tx: &mpsc::Sender<FlowResult<Json>>,
    blocked: &Arc<Mutex<Option<String>>>,
) -> bool {
    match monitor.await {
        Ok(Ok(())) => {}
        Ok(Err(err)) => {
            let _ = chunk_tx.send(Err(err)).await;
            return true;
        }
        Err(err) => {
            let _ = chunk_tx
                .send(Err(FlowError::Internal(format!(
                    "nemo_guardrails stream monitor task failed: {err}"
                ))))
                .await;
            return true;
        }
    }

    if let Some(message) = blocked_message(blocked) {
        let _ = chunk_tx.send(Err(streaming_output_blocked(message))).await;
        return true;
    }

    false
}

fn blocked_message(blocked: &Arc<Mutex<Option<String>>>) -> Option<String> {
    blocked.lock().ok().and_then(|guard| guard.clone())
}

fn streaming_output_blocked(message: String) -> FlowError {
    local_violation(format!(
        "NeMo Guardrails output rail blocked the LLM call: {message}"
    ))
}

fn extract_stream_text(codec: LocalGuardrailsCodec, chunk: &Json) -> Option<String> {
    let chunk = chunk.as_object()?;
    match codec {
        LocalGuardrailsCodec::OpenAIChat => {
            let choices = chunk.get("choices")?.as_array()?;
            let mut parts = vec![];
            for choice in choices {
                let content = choice
                    .get("delta")
                    .and_then(Json::as_object)
                    .and_then(|delta| delta.get("content"))
                    .and_then(Json::as_str);
                if let Some(content) = content
                    && !content.is_empty()
                {
                    parts.push(content);
                }
            }
            (!parts.is_empty()).then(|| parts.join(""))
        }
        LocalGuardrailsCodec::OpenAIResponses => {
            if chunk.get("type").and_then(Json::as_str) == Some("response.output_text.delta") {
                chunk
                    .get("delta")
                    .and_then(Json::as_str)
                    .filter(|delta| !delta.is_empty())
                    .map(str::to_string)
            } else {
                None
            }
        }
        LocalGuardrailsCodec::AnthropicMessages => {
            if chunk.get("type").and_then(Json::as_str) != Some("content_block_delta") {
                return None;
            }
            let delta = chunk.get("delta")?.as_object()?;
            if delta.get("type").and_then(Json::as_str) != Some("text_delta") {
                return None;
            }
            delta
                .get("text")
                .and_then(Json::as_str)
                .filter(|text| !text.is_empty())
                .map(str::to_string)
        }
    }
}

async fn monitor_guardrails_stream(
    worker: Arc<LocalGuardrailsWorker>,
    stream_id: String,
    mut text_rx: mpsc::Receiver<Option<String>>,
    mut event_rx: mpsc::UnboundedReceiver<WorkerEnvelope>,
    blocked: Arc<Mutex<Option<String>>>,
) -> FlowResult<()> {
    let mut input_closed = false;
    loop {
        tokio::select! {
            maybe_text = text_rx.recv(), if !input_closed => {
                match maybe_text {
                    Some(Some(text)) => worker.send_stream_text(&stream_id, text)?,
                    Some(None) | None => {
                        worker.send_stream_end(&stream_id)?;
                        input_closed = true;
                    }
                }
            }
            maybe_event = event_rx.recv() => {
                let Some(event) = maybe_event else {
                    worker.forget_stream(&stream_id);
                    return Err(FlowError::Internal(
                        "NeMo Guardrails local Python worker stream closed unexpectedly".to_string(),
                    ));
                };
                if !event.ok {
                    worker.forget_stream(&stream_id);
                    return Err(FlowError::Internal(event.error.unwrap_or_else(|| {
                        "NeMo Guardrails local Python worker stream failed".to_string()
                    })));
                }
                match event.event.as_deref() {
                    Some("blocked") => {
                        if let Some(message) = event.message {
                            let mut guard = blocked.lock().map_err(|err| {
                                FlowError::Internal(format!("stream block state lock poisoned: {err}"))
                            })?;
                            *guard = Some(message);
                        }
                        worker.forget_stream(&stream_id);
                        return Ok(());
                    }
                    Some("done") => {
                        worker.forget_stream(&stream_id);
                        return Ok(());
                    }
                    Some(other) => {
                        worker.forget_stream(&stream_id);
                        return Err(FlowError::Internal(format!(
                            "NeMo Guardrails local Python worker returned unknown stream event '{other}'"
                        )));
                    }
                    None => {}
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/plugins/nemo_guardrails/local_python_tests.rs"]
mod tests;
