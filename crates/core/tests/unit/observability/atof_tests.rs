// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the ATOF JSONL exporter.

use super::*;
use crate::api::event::{
    BaseEvent, CategoryProfile, DataSchema, Event, EventCategory, MarkEvent, ScopeCategory,
    ScopeEvent,
};
use crate::api::runtime::NemoRelayContextState;
use crate::api::runtime::global_context;
use crate::api::scope::{EmitMarkEventParams, PopScopeParams, PushScopeParams, ScopeType};
use crate::codec::request::{AnnotatedLlmRequest, Message, MessageContent};
#[cfg(feature = "atof-streaming")]
use futures_util::StreamExt;
use serde_json::{Map, json};
use std::fs;
#[cfg(feature = "atof-streaming")]
use std::io::{Read, Write};
#[cfg(feature = "atof-streaming")]
use std::net::TcpListener;
use std::sync::Arc;
#[cfg(feature = "atof-streaming")]
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

fn enable_operational_logs() {
    let _ = spdlog::init_log_crate_proxy();
    log::set_max_level(log::LevelFilter::Info);
}

fn temp_dir(prefix: &str) -> PathBuf {
    let id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("nemo-relay-{prefix}-{id}"));
    fs::create_dir_all(&path).unwrap();
    path
}

fn reset_global() {
    crate::shared_runtime::reset_runtime_owner_for_tests();
    let context = global_context();
    *context.write().unwrap() = NemoRelayContextState::new();
}

fn make_mark_event(name: &str) -> Event {
    Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .uuid(Uuid::now_v7())
            .name(name)
            .data(json!({"step": 1}))
            .build(),
        None,
        None,
    ))
}

fn make_scope_start_event(name: &str) -> Event {
    Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(Uuid::now_v7())
            .name(name)
            .data(json!({"input": true}))
            .build(),
        ScopeCategory::Start,
        Vec::new(),
        EventCategory::agent(),
        None,
    ))
}

fn make_annotated_llm_event(name: &str) -> Event {
    let request = AnnotatedLlmRequest {
        messages: vec![Message::User {
            content: MessageContent::Text("hello".into()),
            name: None,
        }],
        model: Some("demo-model".into()),
        params: None,
        tools: None,
        tool_choice: None,
        store: None,
        previous_response_id: None,
        truncation: None,
        reasoning: None,
        include: None,
        user: None,
        metadata: None,
        service_tier: None,
        parallel_tool_calls: None,
        max_output_tokens: None,
        max_tool_calls: None,
        top_logprobs: None,
        stream: None,
        extra: Map::new(),
    };

    Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(Uuid::now_v7())
            .name(name)
            .data(json!({"input": true}))
            .build(),
        ScopeCategory::Start,
        Vec::new(),
        EventCategory::llm(),
        Some(
            CategoryProfile::builder()
                .model_name("demo-model")
                .annotated_request(Arc::new(request))
                .build(),
        ),
    ))
}

fn wire_format_llm_event(
    uuid: Uuid,
    parent_uuid: Option<Uuid>,
    scope_category: ScopeCategory,
    name: &str,
    model_name: &str,
    gateway_path: &str,
    data: serde_json::Value,
) -> Event {
    Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(uuid)
            .parent_uuid_opt(parent_uuid)
            .name(name)
            .data(data)
            .data_schema(
                DataSchema::builder()
                    .name("llm.provider_payload")
                    .version("1")
                    .build(),
            )
            .metadata(json!({
                "source": "openclaw.public_plugin",
                "gateway_path": gateway_path,
                "provider_payload_exact": true
            }))
            .build(),
        scope_category,
        Vec::new(),
        EventCategory::llm(),
        Some(CategoryProfile::builder().model_name(model_name).build()),
    ))
}

fn openclaw_agent_scope_event(
    uuid: Uuid,
    parent_uuid: Option<Uuid>,
    scope_category: ScopeCategory,
    name: &str,
    session_id: &str,
    scope_role: Option<&str>,
) -> Event {
    let mut metadata = Map::new();
    metadata.insert("source".to_string(), json!("openclaw.session_start"));
    metadata.insert("hook_event_name".to_string(), json!("session_start"));
    metadata.insert("session_id".to_string(), json!(session_id));
    if let Some(scope_role) = scope_role {
        metadata.insert("nemo_relay_scope_role".to_string(), json!(scope_role));
    }

    Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(uuid)
            .parent_uuid_opt(parent_uuid)
            .name(name)
            .data(json!({"session_id": session_id}))
            .metadata(serde_json::Value::Object(metadata))
            .build(),
        scope_category,
        Vec::new(),
        EventCategory::agent(),
        None,
    ))
}

fn openclaw_replay_llm_event(
    uuid: Uuid,
    parent_uuid: Option<Uuid>,
    scope_category: ScopeCategory,
    data: serde_json::Value,
) -> Event {
    Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .uuid(uuid)
            .parent_uuid_opt(parent_uuid)
            .name("openclaw-model-call")
            .data(data)
            .metadata(json!({
                "source": "openclaw.llm_output",
                "hook_event_name": "llm_output"
            }))
            .build(),
        scope_category,
        Vec::new(),
        EventCategory::llm(),
        Some(
            CategoryProfile::builder()
                .model_name("claude-sonnet-4")
                .build(),
        ),
    ))
}

fn openclaw_timing_mark_event(
    uuid: Uuid,
    parent_uuid: Option<Uuid>,
    name: &str,
    data: serde_json::Value,
) -> Event {
    Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .uuid(uuid)
            .parent_uuid_opt(parent_uuid)
            .name(name)
            .data(data)
            .build(),
        None,
        None,
    ))
}

fn read_jsonl(path: &Path) -> Vec<serde_json::Value> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[cfg(feature = "atof-streaming")]
fn read_http_request(stream: &mut std::net::TcpStream) -> String {
    let mut data = Vec::new();
    let mut buf = [0_u8; 1];
    while !data.ends_with(b"\r\n\r\n") {
        stream.read_exact(&mut buf).unwrap();
        data.push(buf[0]);
    }
    let headers = String::from_utf8_lossy(&data).to_string();
    if let Some(length) = headers
        .lines()
        .find_map(|line| {
            line.split_once(':').and_then(|(name, value)| {
                name.eq_ignore_ascii_case("content-length")
                    .then_some(value.trim())
            })
        })
        .and_then(|value| value.trim().parse::<usize>().ok())
    {
        let mut body = vec![0_u8; length];
        stream.read_exact(&mut body).unwrap();
        return String::from_utf8(body).unwrap();
    }
    if headers
        .lines()
        .any(|line| line.eq_ignore_ascii_case("Transfer-Encoding: chunked"))
    {
        let mut body = Vec::new();
        loop {
            let mut size_line = Vec::new();
            loop {
                stream.read_exact(&mut buf).unwrap();
                size_line.push(buf[0]);
                if size_line.ends_with(b"\r\n") {
                    break;
                }
            }
            let size_text = String::from_utf8_lossy(&size_line);
            let size = usize::from_str_radix(size_text.trim(), 16).unwrap();
            if size == 0 {
                let mut trailer = [0_u8; 2];
                stream.read_exact(&mut trailer).unwrap();
                break;
            }
            let mut chunk = vec![0_u8; size];
            stream.read_exact(&mut chunk).unwrap();
            body.extend(chunk);
            let mut crlf = [0_u8; 2];
            stream.read_exact(&mut crlf).unwrap();
        }
        return String::from_utf8(body).unwrap();
    }
    String::new()
}

#[cfg(feature = "atof-streaming")]
fn start_http_capture_server(expected_requests: usize) -> (String, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let captures = Arc::new(Mutex::new(Vec::new()));
    let thread_captures = Arc::clone(&captures);
    std::thread::spawn(move || {
        for _ in 0..expected_requests {
            let (mut stream, _) = listener.accept().unwrap();
            let request_captures = Arc::clone(&thread_captures);
            std::thread::spawn(move || {
                let body = read_http_request(&mut stream);
                request_captures.lock().unwrap().push(body);
                stream
                    .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                    .unwrap();
            });
        }
    });
    (url, captures)
}

#[cfg(feature = "atof-streaming")]
fn wait_for_captures(captures: &Arc<Mutex<Vec<String>>>, expected: usize) -> Vec<String> {
    for _ in 0..100 {
        let snapshot = captures.lock().unwrap().clone();
        if snapshot.len() >= expected {
            return snapshot;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    captures.lock().unwrap().clone()
}

#[cfg(feature = "atof-streaming")]
fn start_websocket_capture_server(
    listener: TcpListener,
    captures: Arc<Mutex<Vec<String>>>,
    expected_messages: usize,
) {
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async move {
            let listener = tokio::net::TcpListener::from_std(listener).unwrap();
            let (stream, _) = listener.accept().await.unwrap();
            let mut websocket = tokio_tungstenite::accept_async(stream).await.unwrap();
            while let Some(message) = websocket.next().await {
                let message = message.unwrap();
                if message.is_text() {
                    captures
                        .lock()
                        .unwrap()
                        .push(message.into_text().unwrap().to_string());
                    if captures.lock().unwrap().len() >= expected_messages {
                        return;
                    }
                }
            }
        });
    });
}

#[test]
fn default_config_uses_cwd_append_and_timestamped_filename() {
    let config = AtofExporterConfig::default();
    let AtofSinkConfig::File(file) = config.sink else {
        panic!("default ATOF config must use a file sink");
    };

    assert_eq!(file.output_directory, std::env::current_dir().unwrap());
    assert_eq!(file.mode, AtofExporterMode::Append);
    assert_eq!(AtofExporterMode::Append.as_str(), "append");
    assert_eq!(AtofExporterMode::Overwrite.as_str(), "overwrite");
    assert!(file.filename.starts_with("nemo-relay-events-"));
    assert!(file.filename.ends_with(".jsonl"));
    assert_eq!(
        file.filename.len(),
        "nemo-relay-events-YYYY-MM-DD-HH.MM.SS.jsonl".len()
    );
}

#[test]
fn endpoint_and_exporter_config_builders_preserve_values() {
    let dir = temp_dir("atof-config-builders");
    let endpoint =
        AtofEndpointConfig::new("http://127.0.0.1:9/events", AtofEndpointTransport::HttpPost)
            .with_header("x-test", "enabled")
            .with_timeout_millis(42)
            .with_field_name_policy(AtofEndpointFieldNamePolicy::ReplaceDots);
    let config = AtofExporterConfig::new()
        .with_output_directory(&dir)
        .with_mode(AtofExporterMode::Overwrite)
        .with_filename("custom.jsonl")
        .with_stream_sink(endpoint.clone());

    assert_eq!(
        endpoint.headers.get("x-test").map(String::as_str),
        Some("enabled")
    );
    assert_eq!(endpoint.timeout_millis, 42);
    assert_eq!(
        endpoint.field_name_policy,
        AtofEndpointFieldNamePolicy::ReplaceDots
    );
    assert_eq!(AtofEndpointFieldNamePolicy::Preserve.as_str(), "preserve");
    assert_eq!(
        AtofEndpointFieldNamePolicy::parse("replace_dots"),
        Some(AtofEndpointFieldNamePolicy::ReplaceDots)
    );
    assert_eq!(config.path(), None);
    assert_eq!(config.sink, AtofSinkConfig::Stream(endpoint));
}

#[test]
#[cfg(feature = "atof-streaming")]
fn endpoint_field_name_policy_replaces_dots_recursively() {
    let config =
        AtofEndpointConfig::new("http://127.0.0.1:9/events", AtofEndpointTransport::HttpPost)
            .with_field_name_policy(AtofEndpointFieldNamePolicy::ReplaceDots);
    let transformed = endpoint_event_json(
        &config,
        json!({
            "kind": "scope",
            "metadata": {
                "otel_status_code": "existing",
                "otel.status_code": "OK",
                "nested": [{"a.b": true}]
            }
        })
        .to_string(),
    );
    let value: Json = serde_json::from_str(&transformed).unwrap();
    assert_eq!(value["metadata"]["otel_status_code"], json!("OK"));
    assert_eq!(value["metadata"]["otel_status_code_2"], json!("existing"));
    assert_eq!(value["metadata"]["nested"][0]["a_b"], json!(true));
}

#[test]
#[cfg(feature = "atof-streaming")]
fn endpoint_field_name_policy_preserves_raw_json_and_falls_back_for_invalid_json() {
    let preserve =
        AtofEndpointConfig::new("http://127.0.0.1:9/events", AtofEndpointTransport::HttpPost);
    let raw = "{\"metadata\":{\"otel.status_code\":\"OK\"}}";
    assert_eq!(endpoint_event_json(&preserve, raw.into()), raw);

    let replace =
        AtofEndpointConfig::new("http://127.0.0.1:9/events", AtofEndpointTransport::HttpPost)
            .with_field_name_policy(AtofEndpointFieldNamePolicy::ReplaceDots);
    assert_eq!(endpoint_event_json(&replace, "not-json".into()), "not-json");
}

#[test]
#[cfg(feature = "atof-streaming")]
fn endpoint_http_helper_edges_are_safe() {
    assert_eq!(
        AtofEndpointFieldNamePolicy::parse("unknown"),
        None,
        "unknown field name policies should be rejected"
    );
}

#[test]
fn append_mode_preserves_existing_lines() {
    enable_operational_logs();
    let dir = temp_dir("atof-append");
    let path = dir.join("events.jsonl");
    fs::write(&path, "{\"existing\":true}\n").unwrap();

    let exporter = AtofExporter::new(
        AtofExporterConfig::new()
            .with_output_directory(&dir)
            .with_filename("events.jsonl"),
    )
    .unwrap();
    (exporter.subscriber())(&make_mark_event("appended"));
    exporter.force_flush().unwrap();

    let lines = read_jsonl(&path);
    assert_eq!(lines[0], json!({"existing": true}));
    assert_eq!(lines[1]["kind"], "mark");
    assert_eq!(lines[1]["name"], "appended");
}

#[test]
fn overwrite_mode_truncates_existing_lines() {
    let dir = temp_dir("atof-overwrite");
    let path = dir.join("events.jsonl");
    fs::write(&path, "{\"existing\":true}\n").unwrap();

    let exporter = AtofExporter::new(
        AtofExporterConfig::new()
            .with_output_directory(&dir)
            .with_mode(AtofExporterMode::Overwrite)
            .with_filename("events.jsonl"),
    )
    .unwrap();
    (exporter.subscriber())(&make_mark_event("replacement"));
    exporter.shutdown().unwrap();

    let lines = read_jsonl(&path);
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0]["kind"], "mark");
    assert_eq!(lines[0]["name"], "replacement");
}

#[test]
fn subscriber_writes_scope_and_mark_events_as_raw_jsonl() {
    let dir = temp_dir("atof-shape");
    let exporter = AtofExporter::new(
        AtofExporterConfig::new()
            .with_output_directory(&dir)
            .with_filename("events.jsonl"),
    )
    .unwrap();
    let subscriber = exporter.subscriber();

    subscriber(&make_scope_start_event("agent-start"));
    subscriber(&make_mark_event("checkpoint"));
    exporter.force_flush().unwrap();

    let lines = read_jsonl(exporter.path().expect("file sink path"));
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[0]["kind"], "scope");
    assert_eq!(lines[0]["scope_category"], "start");
    assert_eq!(lines[0]["category"], "agent");
    assert_eq!(lines[1]["kind"], "mark");
    assert_eq!(lines[1]["data"], json!({"step": 1}));
}

#[test]
fn shutdown_is_idempotent_and_subscriber_noops_after_close() {
    let dir = temp_dir("atof-closed");
    let exporter = AtofExporter::new(
        AtofExporterConfig::new()
            .with_output_directory(&dir)
            .with_filename("events.jsonl"),
    )
    .unwrap();
    let subscriber = exporter.subscriber();

    subscriber(&make_mark_event("before-close"));
    exporter.shutdown().unwrap();
    subscriber(&make_mark_event("after-close"));
    exporter.shutdown().unwrap();

    let lines = read_jsonl(exporter.path().expect("file sink path"));
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0]["name"], "before-close");
}

#[test]
fn subscriber_writes_canonical_event_jsonl() {
    let dir = temp_dir("atof-canonical");
    let exporter = AtofExporter::new(
        AtofExporterConfig::new()
            .with_output_directory(&dir)
            .with_filename("events.jsonl"),
    )
    .unwrap();
    let event = make_annotated_llm_event("llm-start");

    (exporter.subscriber())(&event);
    exporter.force_flush().unwrap();

    let lines = read_jsonl(exporter.path().expect("file sink path"));
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0], event.try_to_json_value().unwrap());
    assert!(lines[0].get("annotated_request").is_none());
    assert_eq!(
        lines[0]["category_profile"]["annotated_request"]["model"],
        "demo-model"
    );
}

#[test]
#[cfg(feature = "atof-streaming")]
fn streaming_sink_receives_raw_atof_events() {
    enable_operational_logs();
    let dir = temp_dir("atof-streaming-http");
    let (url, captures) = start_http_capture_server(3);
    let exporter = AtofExporter::new(
        AtofExporterConfig::new()
            .with_output_directory(&dir)
            .with_filename("events.jsonl")
            .with_endpoint(AtofEndpointConfig::new(
                url.clone(),
                AtofEndpointTransport::HttpPost,
            )),
    )
    .unwrap();
    let subscriber = exporter.subscriber();

    subscriber(&make_mark_event("first"));
    subscriber(&make_mark_event("second"));
    exporter.force_flush().unwrap();
    subscriber(&make_mark_event("after-flush"));
    exporter.shutdown().unwrap();

    assert_eq!(exporter.path(), None);
    let bodies = wait_for_captures(&captures, 3);
    assert_eq!(bodies.len(), 3, "captured bodies: {bodies:?}");
    let all_streamed = bodies.join("");
    assert!(all_streamed.contains("\"name\":\"first\""));
    assert!(all_streamed.contains("\"name\":\"second\""));
    assert!(all_streamed.contains("\"name\":\"after-flush\""));
    assert_eq!(
        all_streamed
            .lines()
            .filter(|line| line.contains("\"kind\":\"mark\""))
            .count(),
        3,
        "three HTTP POST records plus three NDJSON records: {bodies:?}"
    );
}

#[test]
#[cfg(feature = "atof-streaming")]
fn websocket_endpoint_receives_fifo_json_text_events() {
    enable_operational_logs();
    let dir = temp_dir("atof-streaming-websocket");
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let url = format!("ws://{}", listener.local_addr().unwrap());
    let captures = Arc::new(Mutex::new(Vec::new()));
    start_websocket_capture_server(listener, Arc::clone(&captures), 2);

    let exporter = AtofExporter::new(
        AtofExporterConfig::new()
            .with_output_directory(&dir)
            .with_filename("events.jsonl")
            .with_endpoint(AtofEndpointConfig::new(
                url,
                AtofEndpointTransport::Websocket,
            )),
    )
    .unwrap();
    let subscriber = exporter.subscriber();

    subscriber(&make_mark_event("first"));
    subscriber(&make_mark_event("second"));
    exporter.force_flush().unwrap();

    let messages = wait_for_captures(&captures, 2);
    assert_eq!(messages.len(), 2);
    assert!(messages[0].contains("\"name\":\"first\""));
    assert!(messages[1].contains("\"name\":\"second\""));
}

#[test]
#[cfg(feature = "atof-streaming")]
fn websocket_flush_drains_events_queued_before_reconnect() {
    enable_operational_logs();
    let dir = temp_dir("atof-streaming-websocket-reconnect");
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let local_addr = listener.local_addr().unwrap();
    drop(listener);

    let exporter = AtofExporter::new(
        AtofExporterConfig::new()
            .with_output_directory(&dir)
            .with_filename("events.jsonl")
            .with_endpoint(
                AtofEndpointConfig::new(
                    format!("ws://{local_addr}"),
                    AtofEndpointTransport::Websocket,
                )
                .with_timeout_millis(200),
            ),
    )
    .unwrap();
    let subscriber = exporter.subscriber();

    subscriber(&make_mark_event("first"));
    subscriber(&make_mark_event("second"));
    std::thread::sleep(std::time::Duration::from_millis(300));

    let listener = TcpListener::bind(local_addr).unwrap();
    listener.set_nonblocking(true).unwrap();
    let captures = Arc::new(Mutex::new(Vec::new()));
    start_websocket_capture_server(listener, Arc::clone(&captures), 2);

    exporter.force_flush().unwrap();

    let messages = wait_for_captures(&captures, 2);
    assert_eq!(messages.len(), 2);
    assert!(messages[0].contains("\"name\":\"first\""));
    assert!(messages[1].contains("\"name\":\"second\""));
    exporter.shutdown().unwrap();
}

#[test]
fn subscriber_preserves_wire_format_llm_lifecycle_payloads_as_raw_jsonl() {
    let dir = temp_dir("atof-wire-formats");
    let exporter = AtofExporter::new(
        AtofExporterConfig::new()
            .with_output_directory(&dir)
            .with_filename("events.jsonl"),
    )
    .unwrap();
    let subscriber = exporter.subscriber();

    let anthropic_uuid = Uuid::now_v7();
    let responses_uuid = Uuid::now_v7();
    let chat_uuid = Uuid::now_v7();
    let parent_uuid = Uuid::now_v7();

    let events = [
        wire_format_llm_event(
            anthropic_uuid,
            Some(parent_uuid),
            ScopeCategory::Start,
            "anthropic.messages",
            "claude-sonnet-4",
            "/v1/messages",
            json!({
                "model": "claude-sonnet-4",
                "messages": [{"role": "user", "content": "Find the file."}],
                "tools": [{"name": "search", "input_schema": {"type": "object"}}]
            }),
        ),
        wire_format_llm_event(
            anthropic_uuid,
            Some(parent_uuid),
            ScopeCategory::End,
            "anthropic.messages",
            "claude-sonnet-4",
            "/v1/messages",
            json!({
                "id": "msg_01",
                "type": "message",
                "content": [
                    {"type": "text", "text": "I will search."},
                    {"type": "tool_use", "id": "toolu_01", "name": "search", "input": {"query": "file"}}
                ],
                "usage": {
                    "input_tokens": 11,
                    "output_tokens": 7,
                    "cache_read_input_tokens": 3,
                    "cache_creation_input_tokens": 5,
                    "cost": {"total": 0.0042}
                }
            }),
        ),
        wire_format_llm_event(
            responses_uuid,
            Some(parent_uuid),
            ScopeCategory::Start,
            "openai.responses",
            "gpt-4o",
            "/v1/responses",
            json!({
                "model": "gpt-4o",
                "input": "Find the weather.",
                "tools": [{"type": "function", "name": "get_weather"}]
            }),
        ),
        wire_format_llm_event(
            responses_uuid,
            Some(parent_uuid),
            ScopeCategory::End,
            "openai.responses",
            "gpt-4o",
            "/v1/responses",
            json!({
                "id": "resp_1",
                "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": "I will check."}]},
                    {"type": "function_call", "call_id": "call_weather_1", "name": "get_weather", "arguments": "{\"city\":\"SF\"}"}
                ],
                "usage": {
                    "input_tokens": 75,
                    "output_tokens": 20,
                    "total_tokens": 95,
                    "input_tokens_details": {"cached_tokens": 10},
                    "cost_usd": 0.005
                }
            }),
        ),
        wire_format_llm_event(
            chat_uuid,
            Some(parent_uuid),
            ScopeCategory::Start,
            "openai.chat_completions",
            "gpt-4o",
            "/v1/chat/completions",
            json!({
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Inspect the files."}],
                "tools": [{"type": "function", "function": {"name": "read"}}]
            }),
        ),
        wire_format_llm_event(
            chat_uuid,
            Some(parent_uuid),
            ScopeCategory::End,
            "openai.chat_completions",
            "gpt-4o",
            "/v1/chat/completions",
            json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "I will inspect.",
                        "tool_calls": [{"id": "call_read_1", "function": {"name": "read", "arguments": "{\"path\":\"api.py\"}"}}]
                    }
                }],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 4,
                    "total_tokens": 7,
                    "prompt_tokens_details": {"cached_tokens": 2},
                    "cost_usd": 0.001
                }
            }),
        ),
    ];

    for event in &events {
        subscriber(event);
    }
    exporter.force_flush().unwrap();

    let lines = read_jsonl(exporter.path().expect("file sink path"));
    assert_eq!(lines.len(), events.len());
    for (line, event) in lines.iter().zip(events.iter()) {
        assert_eq!(line, &event.try_to_json_value().unwrap());
        assert_eq!(line["kind"], "scope");
        assert_eq!(line["atof_version"], "0.1");
        assert_eq!(line["parent_uuid"], parent_uuid.to_string());
        assert_eq!(line["category"], "llm");
        assert_eq!(line["data_schema"]["name"], "llm.provider_payload");
        assert_eq!(line["data_schema"]["version"], "1");
        assert_eq!(line["metadata"]["source"], "openclaw.public_plugin");
        assert_eq!(line["metadata"]["provider_payload_exact"], true);
    }

    assert_eq!(lines[0]["name"], "anthropic.messages");
    assert_eq!(lines[0]["scope_category"], "start");
    assert_eq!(lines[0]["metadata"]["gateway_path"], "/v1/messages");
    assert_eq!(
        lines[0]["category_profile"]["model_name"],
        "claude-sonnet-4"
    );
    assert_eq!(lines[0]["data"]["messages"][0]["content"], "Find the file.");
    assert_eq!(lines[1]["scope_category"], "end");
    assert_eq!(lines[1]["data"]["content"][1]["type"], "tool_use");
    assert_eq!(lines[1]["data"]["usage"]["cache_creation_input_tokens"], 5);
    assert_eq!(lines[1]["data"]["usage"]["cost"]["total"], 0.0042);

    assert_eq!(lines[2]["metadata"]["gateway_path"], "/v1/responses");
    assert_eq!(lines[2]["data"]["input"], "Find the weather.");
    assert_eq!(lines[3]["data"]["output"][1]["type"], "function_call");
    assert_eq!(
        lines[3]["data"]["usage"]["input_tokens_details"]["cached_tokens"],
        10
    );
    assert_eq!(lines[3]["data"]["usage"]["cost_usd"], 0.005);

    assert_eq!(lines[4]["metadata"]["gateway_path"], "/v1/chat/completions");
    assert_eq!(
        lines[4]["data"]["messages"][0]["content"],
        "Inspect the files."
    );
    assert_eq!(
        lines[5]["data"]["choices"][0]["message"]["tool_calls"][0]["id"],
        "call_read_1"
    );
    assert_eq!(
        lines[5]["data"]["usage"]["prompt_tokens_details"]["cached_tokens"],
        2
    );
    assert_eq!(lines[5]["data"]["usage"]["cost_usd"], 0.001);
}

#[test]
fn openclaw_subagent_events_preserve_nested_and_fallback_parent_uuid() {
    let dir = temp_dir("atof-openclaw-subagent-parentage");
    let exporter = AtofExporter::new(
        AtofExporterConfig::new()
            .with_output_directory(&dir)
            .with_filename("events.jsonl"),
    )
    .unwrap();
    let subscriber = exporter.subscriber();
    let parent_uuid = Uuid::now_v7();
    let nested_child_uuid = Uuid::now_v7();
    let fallback_child_uuid = Uuid::now_v7();

    let events = [
        openclaw_agent_scope_event(
            parent_uuid,
            None,
            ScopeCategory::Start,
            "requester-agent",
            "parent-session",
            None,
        ),
        openclaw_agent_scope_event(
            nested_child_uuid,
            Some(parent_uuid),
            ScopeCategory::Start,
            "nested-worker",
            "nested-child-session",
            Some("subagent"),
        ),
        openclaw_agent_scope_event(
            nested_child_uuid,
            Some(parent_uuid),
            ScopeCategory::End,
            "nested-worker",
            "nested-child-session",
            Some("subagent"),
        ),
        openclaw_agent_scope_event(
            parent_uuid,
            None,
            ScopeCategory::End,
            "requester-agent",
            "parent-session",
            None,
        ),
        openclaw_agent_scope_event(
            fallback_child_uuid,
            None,
            ScopeCategory::Start,
            "fallback-worker",
            "fallback-child-session",
            Some("subagent"),
        ),
        openclaw_agent_scope_event(
            fallback_child_uuid,
            None,
            ScopeCategory::End,
            "fallback-worker",
            "fallback-child-session",
            Some("subagent"),
        ),
    ];

    for event in &events {
        subscriber(event);
    }
    exporter.force_flush().unwrap();

    let lines = read_jsonl(exporter.path().expect("file sink path"));
    let nested_start = lines
        .iter()
        .find(|line| {
            line["uuid"] == nested_child_uuid.to_string() && line["scope_category"] == "start"
        })
        .unwrap();
    assert_eq!(nested_start["parent_uuid"], parent_uuid.to_string());
    assert_eq!(
        nested_start["metadata"]["nemo_relay_scope_role"],
        json!("subagent")
    );

    let fallback_start = lines
        .iter()
        .find(|line| {
            line["uuid"] == fallback_child_uuid.to_string() && line["scope_category"] == "start"
        })
        .unwrap();
    assert!(
        !fallback_start
            .as_object()
            .unwrap()
            .contains_key("parent_uuid")
            || fallback_start["parent_uuid"].is_null()
    );
    assert_eq!(
        fallback_start["metadata"]["nemo_relay_scope_role"],
        json!("subagent")
    );
}

#[test]
fn subscriber_preserves_openclaw_placeholder_replay_payloads_as_raw_jsonl() {
    let dir = temp_dir("atof-openclaw-placeholder");
    let exporter = AtofExporter::new(
        AtofExporterConfig::new()
            .with_output_directory(&dir)
            .with_filename("events.jsonl"),
    )
    .unwrap();
    let subscriber = exporter.subscriber();

    let uuid = Uuid::now_v7();
    let parent_uuid = Uuid::now_v7();
    let events = [
        openclaw_replay_llm_event(
            uuid,
            Some(parent_uuid),
            ScopeCategory::Start,
            json!({
                "headers": {},
                "content": {
                    "provider": "nvidia-inference",
                    "model": "claude-sonnet-4",
                    "prompt": "",
                    "messages": [],
                    "imagesCount": 0,
                    "placeholderRequest": true,
                    "source": "openclaw.llm_output"
                }
            }),
        ),
        openclaw_replay_llm_event(
            uuid,
            Some(parent_uuid),
            ScopeCategory::End,
            json!({
                "role": "assistant",
                "content": "I will search.",
                "assistant_texts_count": 1,
                "openclaw": {
                    "assistant_tool_call_names": []
                }
            }),
        ),
    ];

    for event in &events {
        subscriber(event);
    }
    exporter.force_flush().unwrap();

    let lines = read_jsonl(exporter.path().expect("file sink path"));
    assert_eq!(lines.len(), events.len());
    for (line, event) in lines.iter().zip(events.iter()) {
        assert_eq!(line, &event.try_to_json_value().unwrap());
        assert_eq!(line["kind"], "scope");
        assert_eq!(line["atof_version"], "0.1");
        assert_eq!(line["parent_uuid"], parent_uuid.to_string());
        assert_eq!(line["category"], "llm");
        assert_eq!(line["metadata"]["source"], "openclaw.llm_output");
        assert_eq!(line["metadata"]["hook_event_name"], "llm_output");
    }

    assert_eq!(lines[0]["scope_category"], "start");
    assert_eq!(lines[0]["data"]["content"]["placeholderRequest"], true);
    assert_eq!(lines[0]["data"]["content"]["messages"], json!([]));
    assert_eq!(lines[1]["scope_category"], "end");
    assert_eq!(lines[1]["data"]["content"], "I will search.");
}

#[test]
fn subscriber_preserves_openclaw_model_timing_marks_as_raw_jsonl() {
    let dir = temp_dir("atof-openclaw-model-timing");
    let exporter = AtofExporter::new(
        AtofExporterConfig::new()
            .with_output_directory(&dir)
            .with_filename("events.jsonl"),
    )
    .unwrap();
    let subscriber = exporter.subscriber();

    let parent_uuid = Uuid::now_v7();
    let events = [
        openclaw_timing_mark_event(
            Uuid::now_v7(),
            Some(parent_uuid),
            "openclaw.model_call_timing_ambiguous",
            json!({
                "runId": "run-1",
                "sessionId": "session-1",
                "provider": "openai",
                "model": "gpt-4",
                "candidateCount": 2
            }),
        ),
        openclaw_timing_mark_event(
            Uuid::now_v7(),
            Some(parent_uuid),
            "openclaw.model_call_timing_unpaired",
            json!({
                "runId": "run-1",
                "callId": "call-1",
                "provider": "openai",
                "model": "gpt-4",
                "durationMs": 42,
                "outcome": "completed"
            }),
        ),
    ];

    for event in &events {
        subscriber(event);
    }
    exporter.force_flush().unwrap();

    let lines = read_jsonl(exporter.path().expect("file sink path"));
    assert_eq!(lines.len(), events.len());
    for (line, event) in lines.iter().zip(events.iter()) {
        assert_eq!(line, &event.try_to_json_value().unwrap());
        assert_eq!(line["kind"], "mark");
        assert_eq!(line["parent_uuid"], parent_uuid.to_string());
    }

    assert_eq!(lines[0]["name"], "openclaw.model_call_timing_ambiguous");
    assert_eq!(lines[0]["data"]["candidateCount"], 2);
    assert_eq!(lines[1]["name"], "openclaw.model_call_timing_unpaired");
    assert_eq!(lines[1]["data"]["durationMs"], 42);
}

#[test]
fn subscriber_preserves_openclaw_hook_only_fallback_payloads_as_raw_jsonl() {
    let dir = temp_dir("atof-openclaw-hook-fallbacks");
    let exporter = AtofExporter::new(
        AtofExporterConfig::new()
            .with_output_directory(&dir)
            .with_filename("events.jsonl"),
    )
    .unwrap();
    let subscriber = exporter.subscriber();

    let stripped_uuid = Uuid::now_v7();
    let partial_uuid = Uuid::now_v7();
    let parent_uuid = Uuid::now_v7();
    let events = [
        openclaw_replay_llm_event(
            stripped_uuid,
            Some(parent_uuid),
            ScopeCategory::Start,
            json!({
                "headers": {},
                "content": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "messages": [],
                    "imagesCount": 1,
                    "source": "openclaw.llm_output"
                }
            }),
        ),
        openclaw_replay_llm_event(
            stripped_uuid,
            Some(parent_uuid),
            ScopeCategory::End,
            json!({
                "role": "assistant",
                "assistant_texts_count": 1,
                "usage": {
                    "cost_usd": 0.001
                },
                "openclaw": {
                    "assistant_tool_call_names": []
                }
            }),
        ),
        openclaw_replay_llm_event(
            partial_uuid,
            Some(parent_uuid),
            ScopeCategory::Start,
            json!({
                "headers": {},
                "content": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "prompt": "visible prompt",
                    "messages": [{"role": "user", "content": "visible prompt"}],
                    "imagesCount": 0,
                    "source": "openclaw.llm_output"
                }
            }),
        ),
        openclaw_replay_llm_event(
            partial_uuid,
            Some(parent_uuid),
            ScopeCategory::End,
            json!({
                "role": "assistant",
                "content": "visible answer",
                "usage": {
                    "prompt_tokens": 42
                },
                "openclaw": {
                    "assistant_tool_call_names": []
                }
            }),
        ),
    ];

    for event in &events {
        subscriber(event);
    }
    exporter.force_flush().unwrap();

    let lines = read_jsonl(exporter.path().expect("file sink path"));
    assert_eq!(lines.len(), events.len());
    for (line, event) in lines.iter().zip(events.iter()) {
        assert_eq!(line, &event.try_to_json_value().unwrap());
        assert_eq!(line["kind"], "scope");
        assert_eq!(line["parent_uuid"], parent_uuid.to_string());
    }

    assert!(lines[0]["data"]["content"].get("prompt").is_none());
    assert_eq!(lines[0]["data"]["content"]["messages"], json!([]));
    assert!(lines[1]["data"].get("content").is_none());
    assert_eq!(lines[1]["data"]["usage"]["cost_usd"], 0.001);
    assert_eq!(lines[3]["data"]["usage"]["prompt_tokens"], 42);
    assert!(lines[3]["data"]["usage"].get("completion_tokens").is_none());
}

#[test]
fn register_deregister_flush_and_shutdown_work_with_runtime_events() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_global();

    let dir = temp_dir("atof-runtime");
    let exporter = AtofExporter::new(
        AtofExporterConfig::new()
            .with_output_directory(&dir)
            .with_filename("events.jsonl"),
    )
    .unwrap();
    let name = format!("atof_exporter_{}", Uuid::now_v7());

    exporter.register(&name).unwrap();
    let handle = crate::api::scope::push_scope(
        PushScopeParams::builder()
            .name("atof_scope")
            .scope_type(ScopeType::Agent)
            .input(json!({"scope": true}))
            .build(),
    )
    .unwrap();
    crate::api::scope::event(
        EmitMarkEventParams::builder()
            .name("atof_mark")
            .parent(&handle)
            .data(json!({"mark": true}))
            .build(),
    )
    .unwrap();
    crate::api::scope::pop_scope(
        PopScopeParams::builder()
            .handle_uuid(&handle.uuid)
            .output(json!({"done": true}))
            .build(),
    )
    .unwrap();

    assert!(exporter.deregister(&name).unwrap());
    assert!(!exporter.deregister(&name).unwrap());
    exporter.force_flush().unwrap();
    exporter.shutdown().unwrap();
    exporter.shutdown().unwrap();

    let lines = read_jsonl(exporter.path().expect("file sink path"));
    assert_eq!(lines.len(), 3);
    assert_eq!(lines[0]["name"], "atof_scope");
    assert_eq!(lines[1]["name"], "atof_mark");
    assert_eq!(lines[2]["scope_category"], "end");
}

#[test]
fn invalid_output_path_errors_cleanly() {
    let dir = temp_dir("atof-invalid");
    let file_as_dir = dir.join("not-a-directory");
    fs::write(&file_as_dir, "not a directory").unwrap();

    let error = match AtofExporter::new(
        AtofExporterConfig::new()
            .with_output_directory(&file_as_dir)
            .with_filename("events.jsonl"),
    ) {
        Ok(_) => panic!("expected invalid output path error"),
        Err(error) => error,
    };

    assert!(matches!(error, AtofExporterError::OpenFile { .. }));
}

#[test]
fn missing_output_directory_is_created() {
    let dir = temp_dir("atof-missing-output-dir");
    let output_dir = dir.join("nested/atof");

    let exporter = AtofExporter::new(
        AtofExporterConfig::new()
            .with_output_directory(&output_dir)
            .with_filename("events.jsonl"),
    )
    .unwrap();

    let output_path = output_dir.join("events.jsonl");
    assert_eq!(exporter.path(), Some(output_path.as_path()));
    assert!(output_dir.is_dir());
    assert!(output_path.exists());
}

#[test]
fn invalid_filename_errors_cleanly() {
    let dir = temp_dir("atof-invalid-filename");

    let error = match AtofExporter::new(
        AtofExporterConfig::new()
            .with_output_directory(&dir)
            .with_filename("missing-parent/events.jsonl"),
    ) {
        Ok(_) => panic!("expected invalid filename path error"),
        Err(error) => error,
    };

    assert!(matches!(error, AtofExporterError::OpenFile { .. }));
}

#[test]
#[cfg(feature = "atof-streaming")]
fn invalid_endpoint_config_errors_cleanly() {
    let dir = temp_dir("atof-invalid-endpoint");

    let error = match AtofExporter::new(
        AtofExporterConfig::new()
            .with_output_directory(&dir)
            .with_filename("events.jsonl")
            .with_endpoint(AtofEndpointConfig::new(
                "not a url",
                AtofEndpointTransport::HttpPost,
            )),
    ) {
        Ok(_) => panic!("expected invalid endpoint config error"),
        Err(error) => error,
    };

    match error {
        AtofExporterError::InvalidEndpoint(message) => {
            assert!(message.contains("endpoints[0]"));
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
#[cfg(feature = "atof-streaming")]
fn invalid_endpoint_scheme_errors_cleanly() {
    let dir = temp_dir("atof-invalid-endpoint-scheme");

    let cases = [
        (
            AtofEndpointTransport::HttpPost,
            "ws://localhost:8080/events",
            "http_post",
            "ws",
        ),
        (
            AtofEndpointTransport::Ndjson,
            "ws://localhost:8080/events",
            "ndjson",
            "ws",
        ),
        (
            AtofEndpointTransport::Websocket,
            "http://localhost:8080/events",
            "websocket",
            "http",
        ),
    ];

    for (transport, url, transport_name, scheme) in cases {
        let error = match AtofExporter::new(
            AtofExporterConfig::new()
                .with_output_directory(&dir)
                .with_filename(format!("{transport_name}.jsonl"))
                .with_endpoint(AtofEndpointConfig::new(url, transport)),
        ) {
            Ok(_) => panic!("expected invalid endpoint scheme error"),
            Err(error) => error,
        };

        match error {
            AtofExporterError::InvalidEndpoint(message) => {
                assert!(message.contains("endpoints[0]"));
                assert!(message.contains(transport_name));
                assert!(message.contains(scheme));
            }
            other => panic!("unexpected error: {other}"),
        }
    }
}

#[test]
#[cfg(feature = "atof-streaming")]
fn endpoint_validation_rejects_empty_timeout_and_invalid_headers() {
    let mut headers = std::collections::HashMap::new();
    headers.insert("x-test".to_string(), "ok".to_string());
    validate_endpoint_config(AtofEndpointConfig {
        url: "http://127.0.0.1:9/events".into(),
        transport: AtofEndpointTransport::HttpPost,
        headers: headers.clone(),
        header_env: std::collections::HashMap::new(),
        timeout_millis: 1,
        field_name_policy: AtofEndpointFieldNamePolicy::Preserve,
    })
    .unwrap();
    assert_eq!(
        resolved_header_map(&headers, &Default::default())
            .unwrap()
            .len(),
        1
    );
    let env_headers = std::collections::HashMap::from([("x-path".into(), "PATH".into())]);
    assert_eq!(
        resolved_header_map(&Default::default(), &env_headers)
            .unwrap()
            .len(),
        1
    );
    let missing_env = std::collections::HashMap::from([(
        "authorization".into(),
        "NEMO_RELAY_TEST_MISSING_ATOF_SECRET".into(),
    )]);
    assert!(resolved_header_map(&Default::default(), &missing_env).is_err());
    let conflict = std::collections::HashMap::from([("x-test".into(), "PATH".into())]);
    assert!(resolved_header_map(&headers, &conflict).is_err());
    let mixed_case_headers =
        std::collections::HashMap::from([("Authorization".into(), "Bearer literal".into())]);
    let mixed_case_conflict =
        std::collections::HashMap::from([("authorization".into(), "PATH".into())]);
    assert!(resolved_header_map(&mixed_case_headers, &mixed_case_conflict).is_err());

    let empty_url = AtofEndpointConfig {
        url: "  ".into(),
        transport: AtofEndpointTransport::HttpPost,
        headers: std::collections::HashMap::new(),
        header_env: std::collections::HashMap::new(),
        timeout_millis: 1,
        field_name_policy: AtofEndpointFieldNamePolicy::Preserve,
    };
    assert!(
        validate_endpoint_config(empty_url)
            .unwrap_err()
            .to_string()
            .contains("endpoint url must be non-empty")
    );

    let zero_timeout = AtofEndpointConfig {
        url: "http://127.0.0.1:9/events".into(),
        transport: AtofEndpointTransport::HttpPost,
        headers: std::collections::HashMap::new(),
        header_env: std::collections::HashMap::new(),
        timeout_millis: 0,
        field_name_policy: AtofEndpointFieldNamePolicy::Preserve,
    };
    assert!(
        validate_endpoint_config(zero_timeout)
            .unwrap_err()
            .to_string()
            .contains("timeout_millis")
    );

    let mut bad_header_name = std::collections::HashMap::new();
    bad_header_name.insert("bad header".to_string(), "ok".to_string());
    assert!(resolved_header_map(&bad_header_name, &Default::default()).is_err());

    let mut bad_header_value = std::collections::HashMap::new();
    bad_header_value.insert("x-test".to_string(), "bad\nvalue".to_string());
    assert!(resolved_header_map(&bad_header_value, &Default::default()).is_err());
    assert!(
        validate_endpoint_config(AtofEndpointConfig {
            url: "http://127.0.0.1:9/events".into(),
            transport: AtofEndpointTransport::Ndjson,
            headers: bad_header_value,
            header_env: std::collections::HashMap::new(),
            timeout_millis: 1,
            field_name_policy: AtofEndpointFieldNamePolicy::Preserve,
        })
        .is_err()
    );
}

#[test]
#[cfg(feature = "atof-streaming")]
fn endpoint_activation_snapshots_header_env() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    let variable = format!("NEMO_RELAY_TEST_ATOF_HEADER_ENV_{}", std::process::id());
    // SAFETY: The test mutex serializes environment access for this process.
    unsafe { std::env::set_var(&variable, "Bearer relay-499") };

    let endpoint = validate_endpoint_config(AtofEndpointConfig {
        url: "http://127.0.0.1:9/events".into(),
        transport: AtofEndpointTransport::HttpPost,
        headers: std::collections::HashMap::new(),
        header_env: std::collections::HashMap::from([("authorization".into(), variable.clone())]),
        timeout_millis: 1,
        field_name_policy: AtofEndpointFieldNamePolicy::Preserve,
    })
    .unwrap();

    // SAFETY: The activated endpoint must not read this environment variable again.
    unsafe { std::env::remove_var(&variable) };
    assert_eq!(
        endpoint.headers.get("authorization").unwrap(),
        "Bearer relay-499"
    );
}

#[test]
#[cfg(feature = "atof-streaming")]
fn endpoint_worker_helpers_acknowledge_flush_and_close_error_paths() {
    enable_operational_logs();
    let (body_tx, body) = ndjson_body_channel();
    drop(body);
    send_ndjson_event(0, &body_tx, "{}".into());
    let (flush_tx, flush_rx) = std::sync::mpsc::channel();
    send_ndjson_flush(0, &body_tx, flush_tx);
    flush_rx
        .recv_timeout(std::time::Duration::from_secs(1))
        .unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    tx.send(EndpointMessage::Event("{}".into())).unwrap();
    let (flush_tx, flush_rx) = std::sync::mpsc::channel();
    tx.send(EndpointMessage::Flush(flush_tx)).unwrap();
    let (close_tx, close_rx) = std::sync::mpsc::channel();
    tx.send(EndpointMessage::Close(close_tx)).unwrap();
    drop(tx);

    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(async { drain_closed(rx).await });
    flush_rx
        .recv_timeout(std::time::Duration::from_secs(1))
        .unwrap();
    close_rx
        .recv_timeout(std::time::Duration::from_secs(1))
        .unwrap();
}

#[test]
#[cfg(feature = "atof-streaming")]
fn endpoint_worker_reports_flush_and_close_timeouts() {
    enable_operational_logs();
    let (sender, _receiver) = tokio::sync::mpsc::unbounded_channel();
    let worker = AtofEndpointWorker {
        sender,
        timeout: std::time::Duration::from_millis(1),
        index: 7,
        transport: AtofEndpointTransport::HttpPost,
    };

    worker.flush();
    worker.close();
}

#[test]
#[cfg(feature = "atof-streaming")]
fn http_endpoint_worker_acknowledges_flush_close_and_logs_http_errors() {
    enable_operational_logs();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(5)))
            .unwrap();
        let _ = read_http_request(&mut stream);
        stream
            .write_all(
                b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            )
            .unwrap();
    });

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let worker = std::thread::spawn(move || {
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            run_http_post_endpoint(
                0,
                validate_endpoint_config(
                    AtofEndpointConfig::new(url, AtofEndpointTransport::HttpPost)
                        .with_timeout_millis(5_000),
                )
                .unwrap(),
                rx,
            )
            .await;
        });
    });

    tx.send(EndpointMessage::Event("{\"kind\":\"mark\"}".into()))
        .unwrap();
    let (flush_tx, flush_rx) = std::sync::mpsc::channel();
    tx.send(EndpointMessage::Flush(flush_tx)).unwrap();
    flush_rx
        .recv_timeout(std::time::Duration::from_secs(10))
        .unwrap();
    let (close_tx, close_rx) = std::sync::mpsc::channel();
    tx.send(EndpointMessage::Close(close_tx)).unwrap();
    close_rx
        .recv_timeout(std::time::Duration::from_secs(10))
        .unwrap();
    worker.join().unwrap();
    server.join().unwrap();
}

#[test]
#[cfg(feature = "atof-streaming")]
fn http_endpoint_worker_reports_request_transport_failure() {
    enable_operational_logs();
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let worker = std::thread::spawn(move || {
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            run_http_post_endpoint(
                0,
                validate_endpoint_config(
                    AtofEndpointConfig::new(
                        "http://127.0.0.1:0/events",
                        AtofEndpointTransport::HttpPost,
                    )
                    .with_timeout_millis(100),
                )
                .unwrap(),
                rx,
            )
            .await;
        });
    });

    tx.send(EndpointMessage::Event("{\"kind\":\"mark\"}".into()))
        .unwrap();
    let (close_tx, close_rx) = std::sync::mpsc::channel();
    tx.send(EndpointMessage::Close(close_tx)).unwrap();
    close_rx
        .recv_timeout(std::time::Duration::from_secs(1))
        .unwrap();
    worker.join().unwrap();
}

#[test]
#[cfg(feature = "atof-streaming")]
fn websocket_helpers_cover_invalid_headers_and_timeout_reconnect_path() {
    enable_operational_logs();
    let endpoint = validate_endpoint_config(AtofEndpointConfig {
        url: "ws://127.0.0.1:9/events".into(),
        transport: AtofEndpointTransport::Websocket,
        headers: std::collections::HashMap::new(),
        header_env: std::collections::HashMap::new(),
        timeout_millis: 1,
        field_name_policy: AtofEndpointFieldNamePolicy::Preserve,
    })
    .unwrap();
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        assert!(connect_websocket(&endpoint).await.is_err());

        let mut socket = None;
        let mut pending = std::collections::VecDeque::from(["{\"kind\":\"mark\"}".to_string()]);
        let mut retry = WebSocketRetryState::default();
        assert!(
            !drain_websocket_pending(0, &endpoint, &mut socket, &mut pending, &mut retry).await
        );
        assert_eq!(pending.len(), 1);
        assert_eq!(retry.attempts, 1);
    });
}

#[test]
#[cfg(feature = "atof-streaming")]
fn ndjson_upload_close_timeout_acknowledges_close() {
    enable_operational_logs();
    let request = tokio::runtime::Runtime::new().unwrap().block_on(async {
        let request: tokio::task::JoinHandle<reqwest::Result<reqwest::Response>> =
            tokio::spawn(async {
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                unreachable!("timeout should finish before this task completes")
            });
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        finish_ndjson_upload(0, request, std::time::Duration::from_millis(1), done_tx).await;
        done_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .unwrap();
        true
    });
    assert!(request);
}

#[test]
#[cfg(feature = "atof-streaming")]
fn ndjson_upload_completion_reports_success_http_upload_and_task_failures() {
    enable_operational_logs();
    let runtime = tokio::runtime::Runtime::new().unwrap();

    for response in [
        "HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
    ] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let _ = read_http_request(&mut stream);
            stream.write_all(response.as_bytes()).unwrap();
        });
        runtime.block_on(async {
            let request = tokio::spawn(async move { reqwest::get(url).await });
            let (done_tx, done_rx) = std::sync::mpsc::channel();
            finish_ndjson_upload(2, request, std::time::Duration::from_secs(5), done_tx).await;
            done_rx
                .recv_timeout(std::time::Duration::from_secs(1))
                .unwrap();
        });
        server.join().unwrap();
    }

    runtime.block_on(async {
        let upload_error = tokio::spawn(async { reqwest::get("http://127.0.0.1:0").await });
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        finish_ndjson_upload(3, upload_error, std::time::Duration::from_secs(5), done_tx).await;
        done_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .unwrap();

        let task_error = tokio::spawn(async {
            std::future::pending::<reqwest::Result<reqwest::Response>>().await
        });
        task_error.abort();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        finish_ndjson_upload(4, task_error, std::time::Duration::from_secs(5), done_tx).await;
        done_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .unwrap();
    });
}

#[test]
fn force_flush_reports_stored_subscriber_failure() {
    enable_operational_logs();
    let dir = temp_dir("atof-stored-failure");
    let exporter = AtofExporter::new(
        AtofExporterConfig::new()
            .with_output_directory(&dir)
            .with_filename("events.jsonl"),
    )
    .unwrap();

    exporter.state.lock().unwrap().last_error = Some("write failed".to_string());
    let error = exporter.force_flush().unwrap_err();

    match error {
        AtofExporterError::StoredFailure { path, message } => {
            assert_eq!(path, dir.join("events.jsonl"));
            assert_eq!(message, "write failed");
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn force_flush_keeps_exporter_open_and_shutdown_is_terminal() {
    let dir = temp_dir("atof-flush-not-terminal");
    let exporter = AtofExporter::new(
        AtofExporterConfig::new()
            .with_output_directory(&dir)
            .with_filename("events.jsonl"),
    )
    .unwrap();
    let subscriber = exporter.subscriber();

    subscriber(&make_mark_event("before_flush"));
    exporter.force_flush().unwrap();
    subscriber(&make_mark_event("after_flush"));
    exporter.shutdown().unwrap();
    subscriber(&make_mark_event("after_shutdown"));
    exporter.shutdown().unwrap();

    let lines = read_jsonl(exporter.path().expect("file sink path"));
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[0]["name"], "before_flush");
    assert_eq!(lines[1]["name"], "after_flush");
}
