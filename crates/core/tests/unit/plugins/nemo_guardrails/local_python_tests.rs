// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::await_holding_lock)] // Runtime isolation requires serial async plugin tests.

#[cfg(unix)]
use std::fs;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
#[cfg(unix)]
use std::path::{Path, PathBuf};
#[cfg(unix)]
use std::process::Command;
#[cfg(unix)]
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::json;

use super::*;
#[cfg(unix)]
use crate::api::llm::{LlmAttributes, LlmCallExecuteParams, LlmRequest, llm_call_execute};
#[cfg(unix)]
use crate::api::runtime::{
    LlmExecutionNextFn, NemoRelayContextState, ThreadScopeStackBinding, capture_thread_scope_stack,
    create_scope_stack, global_context, restore_thread_scope_stack, set_thread_scope_stack,
};
#[cfg(unix)]
use crate::api::tool::{ToolCallExecuteParams, tool_call_execute};
#[cfg(unix)]
use crate::codec::openai_chat::OpenAIChatCodec;
#[cfg(unix)]
use crate::codec::traits::LlmResponseCodec;
#[cfg(unix)]
use crate::plugin::{
    PluginComponentSpec, PluginConfig, clear_plugin_configuration, initialize_plugins,
};
use crate::plugins::nemo_guardrails::component::LocalBackendConfig;

#[cfg(unix)]
static NEXT_FIXTURE_ID: AtomicUsize = AtomicUsize::new(1);
static PYTHON_EXECUTABLE_ENV_MUTEX: Mutex<()> = Mutex::new(());

struct EnvVarGuard {
    name: &'static str,
    value: Option<std::ffi::OsString>,
}

impl EnvVarGuard {
    fn set(name: &'static str, value: &str) -> Self {
        let old_value = std::env::var_os(name);
        unsafe {
            std::env::set_var(name, value);
        }
        Self {
            name,
            value: old_value,
        }
    }

    fn remove(name: &'static str) -> Self {
        let old_value = std::env::var_os(name);
        unsafe {
            std::env::remove_var(name);
        }
        Self {
            name,
            value: old_value,
        }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        unsafe {
            match &self.value {
                Some(value) => std::env::set_var(self.name, value),
                None => std::env::remove_var(self.name),
            }
        }
    }
}

#[test]
fn python_executable_prefers_config_over_environment() {
    let _env_guard = PYTHON_EXECUTABLE_ENV_MUTEX.lock().unwrap();
    let _nemo_python = EnvVarGuard::set(PYTHON_EXECUTABLE_ENV, "env-python");
    let _pyo3_python = EnvVarGuard::set(PYO3_PYTHON_ENV, "pyo3-python");
    let _uv_python = EnvVarGuard::set(UV_PYTHON_ENV, "uv-python");

    let config = NeMoGuardrailsConfig {
        local: Some(LocalBackendConfig {
            python_executable: Some("configured-python".to_string()),
            ..LocalBackendConfig::default()
        }),
        ..NeMoGuardrailsConfig::default()
    };

    assert_eq!(python_executable(&config), "configured-python");
}

#[test]
fn python_executable_uses_python_environment_before_default() {
    let _env_guard = PYTHON_EXECUTABLE_ENV_MUTEX.lock().unwrap();
    let _nemo_python = EnvVarGuard::remove(PYTHON_EXECUTABLE_ENV);
    let _pyo3_python = EnvVarGuard::set(PYO3_PYTHON_ENV, "pyo3-python");
    let _uv_python = EnvVarGuard::set(UV_PYTHON_ENV, "uv-python");

    assert_eq!(
        python_executable(&NeMoGuardrailsConfig::default()),
        "pyo3-python"
    );
}

#[test]
fn worker_python_path_prepends_configured_path_to_inherited_pythonpath() {
    let configured = std::path::PathBuf::from("fake-guardrails");
    let stdlib = std::path::PathBuf::from("stdlib");
    let platstdlib = std::path::PathBuf::from("platstdlib");
    let configured_path = std::env::join_paths([configured.clone()]).unwrap();
    let inherited_path = std::env::join_paths([stdlib.clone(), platstdlib.clone()]).unwrap();

    let merged = merge_python_path(&configured_path, Some(&inherited_path)).unwrap();

    assert_eq!(
        std::env::split_paths(&merged).collect::<Vec<_>>(),
        vec![configured, stdlib, platstdlib]
    );
}

#[cfg(unix)]
struct FakeGuardrails {
    root: PathBuf,
    module_name: String,
    python: PathBuf,
}

#[cfg(unix)]
impl FakeGuardrails {
    fn new(version: &str) -> Self {
        let _ = spdlog::init_log_crate_proxy();
        log::set_max_level(log::LevelFilter::Info);
        let id = NEXT_FIXTURE_ID.fetch_add(1, Ordering::Relaxed);
        let module_name = format!("fake_guardrails_{id}");
        let root = std::env::temp_dir().join(format!(
            "nemo_relay_fake_guardrails_{}_{}",
            std::process::id(),
            id
        ));
        let package = root.join(&module_name);
        fs::create_dir_all(package.join("rails/llm")).unwrap();
        fs::write(package.join("rails/__init__.py"), "").unwrap();
        fs::write(package.join("rails/llm/__init__.py"), "").unwrap();
        fs::write(package.join("rails/llm/options.py"), fake_options_module()).unwrap();
        fs::write(package.join("__init__.py"), fake_root_module(version)).unwrap();

        let python = root.join("python-wrapper");
        fs::write(
            &python,
            format!(
                "#!/bin/sh\nPYTHONPATH='{}' exec python3 \"$@\"\n",
                shell_single_quote(&root)
            ),
        )
        .unwrap();
        let mut permissions = fs::metadata(&python).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&python, permissions).unwrap();

        Self {
            root,
            module_name,
            python,
        }
    }

    fn config(&self) -> NeMoGuardrailsConfig {
        NeMoGuardrailsConfig {
            mode: "local".to_string(),
            codec: Some("openai_chat".to_string()),
            config_yaml: Some("models: []".to_string()),
            colang_content: Some("define flow noop\n  pass".to_string()),
            local: Some(LocalBackendConfig {
                python_module: Some(self.module_name.clone()),
                python_executable: Some(self.python.to_string_lossy().into_owned()),
                python_path: None,
            }),
            ..NeMoGuardrailsConfig::default()
        }
    }
}

#[cfg(unix)]
impl Drop for FakeGuardrails {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[cfg(unix)]
fn python3_available() -> bool {
    Command::new("python3")
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

#[test]
#[cfg(unix)]
fn shutdown_handles_an_already_reaped_worker_process() {
    let _ = spdlog::init_log_crate_proxy();
    log::set_max_level(log::LevelFilter::Info);
    let mut child = Command::new("true").spawn().unwrap();
    child.wait().unwrap();
    let worker = LocalGuardrailsWorker {
        writer: Mutex::new(None),
        child: Mutex::new(child),
        waiters: Arc::new(Mutex::new(HashMap::new())),
        stream_events: Arc::new(Mutex::new(HashMap::new())),
        next_id: AtomicU64::new(0),
        shutdown_started: AtomicBool::new(false),
    };

    worker.shutdown();
}

#[tokio::test]
#[cfg(unix)]
async fn request_timeout_stops_an_unresponsive_worker() {
    let _ = spdlog::init_log_crate_proxy();
    log::set_max_level(log::LevelFilter::Info);
    let child = Command::new("sleep").arg("60").spawn().unwrap();
    let (sender, receiver) = std_mpsc::channel::<String>();
    let waiters = Arc::new(Mutex::new(HashMap::new()));
    let stream_events = Arc::new(Mutex::new(HashMap::new()));
    let closed_waiters = Arc::clone(&waiters);
    let closed_stream_events = Arc::clone(&stream_events);
    let handle = thread::spawn(move || {
        while receiver.recv().is_ok() {
            thread::sleep(Duration::from_millis(50));
            notify_worker_closed(
                &closed_waiters,
                &closed_stream_events,
                "test worker closed".into(),
            );
        }
    });
    let worker = LocalGuardrailsWorker {
        writer: Mutex::new(Some(WorkerCommandWriter {
            sender,
            error: Arc::new(Mutex::new(None)),
            handle: Some(handle),
        })),
        child: Mutex::new(child),
        waiters,
        stream_events,
        next_id: AtomicU64::new(0),
        shutdown_started: AtomicBool::new(false),
    };

    let error = worker
        .request_with_timeout(json!({"command": "never-reply"}), Duration::from_millis(10))
        .await;
    assert!(error.unwrap_err().to_string().contains("timed out"));
}

#[cfg(unix)]
fn shell_single_quote(path: &Path) -> String {
    path.to_string_lossy().replace('\'', "'\\''")
}

#[cfg(unix)]
fn fake_options_module() -> &'static str {
    r#"
class RailType:
    INPUT = "input"
    OUTPUT = "output"

class RailStatus:
    BLOCKED = "blocked"
    MODIFIED = "modified"
    PASSED = "passed"
"#
}

#[cfg(unix)]
fn fake_root_module(version: &str) -> String {
    format!(
        r#"
import json
import types
from .rails.llm.options import RailStatus

__version__ = {version:?}

class Result:
    def __init__(self, status, content=None, rail=None):
        self.status = status
        self.content = content
        self.rail = rail

class RailsConfig:
    @staticmethod
    def from_content(*, colang_content=None, yaml_content=None):
        stream_first = "stream_first_false" not in (yaml_content or "")
        flows = [] if "no_stream" in (yaml_content or "") else ["self check output"]
        return types.SimpleNamespace(
            yaml=yaml_content,
            colang=colang_content,
            rails=types.SimpleNamespace(
                output=types.SimpleNamespace(
                    flows=flows,
                    streaming=types.SimpleNamespace(enabled=True, stream_first=stream_first),
                )
            )
        )

    @staticmethod
    def from_path(path):
        return types.SimpleNamespace(
            path=path,
            rails=types.SimpleNamespace(
                output=types.SimpleNamespace(
                    flows=["self check output"],
                    streaming=types.SimpleNamespace(enabled=True, stream_first=True),
                )
            )
        )

class LLMRails:
    def __init__(self, config):
        self.config = config

    async def check_async(self, messages, rail_types=None):
        content = " ".join(str(message.get("content", "")) for message in messages)
        if "block" in content:
            return Result(RailStatus.BLOCKED, "", "policy")
        if "modify-tool" in content:
            return Result(RailStatus.MODIFIED, '{{"arguments":{{"safe":true}},"result":{{"ok":true}}}}')
        if "modify" in content:
            return Result(RailStatus.MODIFIED, "rewritten")
        return Result(RailStatus.PASSED, "")

    async def stream_async(self, *, messages=None, generator=None, include_metadata=False):
        async for text in generator:
            if "stream-block" in text:
                yield json.dumps({{"error": {{"type": "guardrails_violation", "message": "blocked stream"}}}})
                return
        yield json.dumps({{"ok": True}})
"#
    )
}

#[cfg(unix)]
#[tokio::test(flavor = "current_thread")]
async fn bridge_checks_pass_block_and_modify_outcomes() {
    if !python3_available() {
        return;
    }

    let fixture = FakeGuardrails::new("0.22.0");
    let bridge = LocalGuardrailsBridge::new(&fixture.config()).unwrap();

    assert!(matches!(
        bridge
            .check(
                vec![json!({"role": "user", "content": "hello"})],
                LocalRailKind::Input,
            )
            .await
            .unwrap(),
        LocalCheckOutcome::Passed
    ));

    match bridge
        .check(
            vec![json!({"role": "user", "content": "block this"})],
            LocalRailKind::Input,
        )
        .await
        .unwrap()
    {
        LocalCheckOutcome::Blocked { rail } => assert_eq!(rail.as_deref(), Some("policy")),
        _ => panic!("expected blocked outcome"),
    }

    match bridge
        .check(
            vec![json!({"role": "user", "content": "modify this"})],
            LocalRailKind::Input,
        )
        .await
        .unwrap()
    {
        LocalCheckOutcome::Modified { content } => assert_eq!(content, "rewritten"),
        _ => panic!("expected modified outcome"),
    }
}

#[cfg(unix)]
#[test]
fn bridge_rejects_unsupported_guardrails_version() {
    if !python3_available() {
        return;
    }

    let fixture = FakeGuardrails::new("0.21.0");
    let error = match LocalGuardrailsBridge::new(&fixture.config()) {
        Ok(_) => panic!("expected unsupported version error"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("nemoguardrails==0.22.0"));
}

#[cfg(unix)]
#[tokio::test(flavor = "current_thread")]
async fn streaming_support_rejects_stream_first_false() {
    if !python3_available() {
        return;
    }

    let fixture = FakeGuardrails::new("0.22.0");
    let mut config = fixture.config();
    config.config_yaml = Some("stream_first_false".to_string());
    let bridge = LocalGuardrailsBridge::new(&config).unwrap();

    assert!(bridge.has_streaming_output_rails().await.unwrap());
    let error = bridge
        .ensure_streaming_output_supported()
        .await
        .unwrap_err();
    assert!(error.to_string().contains("stream_first = true"));
}

#[cfg(unix)]
#[tokio::test(flavor = "current_thread")]
async fn stream_monitor_records_blocked_message() {
    if !python3_available() {
        return;
    }

    let fixture = FakeGuardrails::new("0.22.0");
    let bridge = LocalGuardrailsBridge::new(&fixture.config()).unwrap();
    let (text_tx, text_rx) = mpsc::channel(8);
    let blocked = Arc::new(Mutex::new(None));
    let monitor = bridge
        .spawn_stream_monitor(
            vec![json!({"role": "user", "content": "hello"})],
            text_rx,
            Arc::clone(&blocked),
        )
        .unwrap();

    text_tx
        .send(Some("stream-block".to_string()))
        .await
        .unwrap();
    text_tx.send(None).await.unwrap();
    monitor.await.unwrap().unwrap();

    assert_eq!(blocked.lock().unwrap().as_deref(), Some("blocked stream"));
}

#[tokio::test(flavor = "current_thread")]
async fn guarded_provider_stream_reports_block_after_forwarded_chunks() {
    let provider_stream: LlmJsonStream = Box::pin(tokio_stream::iter(vec![Ok(json!({
        "choices": [{"delta": {"content": "blocked"}}],
    }))]));
    let (text_tx, mut text_rx) = mpsc::channel::<Option<String>>(8);
    let (chunk_tx, mut chunk_rx) = mpsc::channel(8);
    let blocked = Arc::new(Mutex::new(None));
    let monitor_blocked = Arc::clone(&blocked);
    let monitor = tokio::spawn(async move {
        while let Some(item) = text_rx.recv().await {
            match item {
                Some(text) if text.contains("blocked") => {
                    *monitor_blocked.lock().unwrap() = Some("blocked stream".to_string());
                }
                Some(_) => {}
                None => break,
            }
        }
        Ok(())
    });

    forward_guarded_provider_stream(
        provider_stream,
        LocalGuardrailsCodec::OpenAIChat,
        text_tx,
        chunk_tx,
        monitor,
        blocked,
    )
    .await;

    let chunk = chunk_rx.recv().await.unwrap().unwrap();
    assert_eq!(
        chunk,
        json!({
            "choices": [{"delta": {"content": "blocked"}}],
        })
    );

    let error = chunk_rx.recv().await.unwrap().unwrap_err();
    assert!(
        error.to_string().contains("blocked stream"),
        "unexpected error: {error}"
    );
    assert!(chunk_rx.recv().await.is_none());
}

#[test]
fn parse_check_result_rejects_unknown_status() {
    assert!(matches!(
        parse_check_result(json!({"status": "passed"})).unwrap(),
        LocalCheckOutcome::Passed
    ));

    let error = match parse_check_result(json!({"status": "surprising"})) {
        Ok(_) => panic!("expected unknown status to fail"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("unexpected worker check status: surprising"),
        "unexpected error: {error}"
    );
}

#[test]
fn modified_tool_payload_rejects_malformed_content() {
    let error = modified_tool_payload("not-json", "arguments").unwrap_err();
    assert!(
        error
            .to_string()
            .contains("modified tool arguments content that is not valid JSON")
    );

    let error = modified_tool_payload(r#"{"tool_name":"demo"}"#, "result").unwrap_err();
    assert!(
        error
            .to_string()
            .contains("modified tool result content without a 'result' field")
    );
}

#[test]
fn stream_text_extraction_handles_supported_codecs() {
    assert_eq!(
        extract_stream_text(
            LocalGuardrailsCodec::OpenAIChat,
            &json!({"choices": [{"delta": {"content": "hel"}}, {"delta": {"content": "lo"}}]})
        ),
        Some("hello".to_string())
    );
    assert_eq!(
        extract_stream_text(
            LocalGuardrailsCodec::OpenAIResponses,
            &json!({"type": "response.output_text.delta", "delta": "hello"})
        ),
        Some("hello".to_string())
    );
    assert_eq!(
        extract_stream_text(
            LocalGuardrailsCodec::AnthropicMessages,
            &json!({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hello"}})
        ),
        Some("hello".to_string())
    );
}

#[cfg(unix)]
async fn install_local_plugin(config: &NeMoGuardrailsConfig) {
    let component_config = serde_json::to_value(config)
        .unwrap()
        .as_object()
        .unwrap()
        .clone();
    initialize_plugins(PluginConfig {
        version: 1,
        components: vec![PluginComponentSpec {
            kind: crate::plugins::nemo_guardrails::component::NEMO_GUARDRAILS_PLUGIN_KIND
                .to_string(),
            enabled: true,
            config: component_config,
        }],
        policy: Default::default(),
    })
    .await
    .unwrap();
}

#[cfg(unix)]
struct PluginRuntimeResetGuard {
    previous_scope_stack: ThreadScopeStackBinding,
}

#[cfg(unix)]
impl Drop for PluginRuntimeResetGuard {
    fn drop(&mut self) {
        let _ = clear_plugin_configuration();
        crate::shared_runtime::reset_runtime_owner_for_tests();
        *global_context().write().unwrap() = NemoRelayContextState::new();
        restore_thread_scope_stack(self.previous_scope_stack.clone());
    }
}

#[cfg(unix)]
fn reset_plugin_runtime() -> PluginRuntimeResetGuard {
    let previous_scope_stack = capture_thread_scope_stack();
    let _ = clear_plugin_configuration();
    crate::shared_runtime::reset_runtime_owner_for_tests();
    *global_context().write().unwrap() = NemoRelayContextState::new();
    set_thread_scope_stack(create_scope_stack());
    PluginRuntimeResetGuard {
        previous_scope_stack,
    }
}

#[cfg(unix)]
#[tokio::test(flavor = "current_thread")]
async fn registered_local_backend_rewrites_llm_requests_and_tool_payloads() {
    if !python3_available() {
        return;
    }

    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    let _runtime_guard = reset_plugin_runtime();

    let fixture = FakeGuardrails::new("0.22.0");
    let mut config = fixture.config();
    config.input = true;
    config.output = true;
    config.tool_input = true;
    config.tool_output = true;
    install_local_plugin(&config).await;

    let observed_request = Arc::new(Mutex::new(None));
    let observed_callback_request = Arc::clone(&observed_request);
    let callback: LlmExecutionNextFn = Arc::new(move |request| {
        *observed_callback_request.lock().unwrap() = Some(request);
        Box::pin(async move {
            Ok(json!({
                "choices": [{"message": {"role": "assistant", "content": "provider answer"}}]
            }))
        })
    });
    let request = LlmRequest {
        headers: Default::default(),
        content: json!({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "modify this request"}]
        }),
    };
    let response = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("openai")
            .request(request)
            .func(callback)
            .attributes(LlmAttributes::empty())
            .response_codec(Arc::new(OpenAIChatCodec) as Arc<dyn LlmResponseCodec>)
            .build(),
    )
    .await
    .unwrap();
    assert_eq!(
        observed_request.lock().unwrap().as_ref().unwrap().content["messages"][0]["content"],
        json!("rewritten")
    );
    assert_eq!(
        response["choices"][0]["message"]["content"],
        json!("provider answer")
    );

    let tool_result = tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("lookup")
            .args(json!({"modify-tool": true}))
            .func(Arc::new(|args| {
                Box::pin(async move {
                    assert_eq!(args, json!({"safe": true}));
                    Ok(json!({"original": true}))
                })
            }))
            .build(),
    )
    .await
    .unwrap();
    assert_eq!(tool_result, json!({"original": true}));
}

#[cfg(unix)]
#[tokio::test(flavor = "current_thread")]
async fn registered_local_backend_rejects_blocked_llm_and_tool_inputs() {
    if !python3_available() {
        return;
    }

    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    let _runtime_guard = reset_plugin_runtime();

    let fixture = FakeGuardrails::new("0.22.0");
    let mut config = fixture.config();
    config.input = true;
    config.tool_input = true;
    install_local_plugin(&config).await;

    let llm_callback_called = Arc::new(AtomicBool::new(false));
    let llm_callback_marker = Arc::clone(&llm_callback_called);
    let llm_error = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("openai")
            .request(LlmRequest {
                headers: Default::default(),
                content: json!({
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "block this request"}]
                }),
            })
            .func(Arc::new(move |_| {
                llm_callback_marker.store(true, Ordering::SeqCst);
                Box::pin(async { Ok(json!({})) })
            }))
            .attributes(LlmAttributes::empty())
            .response_codec(Arc::new(OpenAIChatCodec) as Arc<dyn LlmResponseCodec>)
            .build(),
    )
    .await
    .unwrap_err();
    assert!(
        llm_error
            .to_string()
            .contains("input rail blocked the LLM call")
    );
    assert!(!llm_callback_called.load(Ordering::SeqCst));

    let tool_callback_called = Arc::new(AtomicBool::new(false));
    let tool_callback_marker = Arc::clone(&tool_callback_called);
    let tool_error = tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("lookup")
            .args(json!({"block": true}))
            .func(Arc::new(move |_| {
                tool_callback_marker.store(true, Ordering::SeqCst);
                Box::pin(async { Ok(json!({})) })
            }))
            .build(),
    )
    .await
    .unwrap_err();
    assert!(
        tool_error
            .to_string()
            .contains("tool_input rail blocked the tool call")
    );
    assert!(!tool_callback_called.load(Ordering::SeqCst));
}
