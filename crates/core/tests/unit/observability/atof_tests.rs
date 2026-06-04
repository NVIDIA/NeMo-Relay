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
use serde_json::{Map, json};
use std::fs;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

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

fn read_jsonl(path: &Path) -> Vec<serde_json::Value> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[test]
fn default_config_uses_cwd_append_and_timestamped_filename() {
    let config = AtofExporterConfig::default();

    assert_eq!(config.output_directory, std::env::current_dir().unwrap());
    assert_eq!(config.mode, AtofExporterMode::Append);
    assert_eq!(AtofExporterMode::Append.as_str(), "append");
    assert_eq!(AtofExporterMode::Overwrite.as_str(), "overwrite");
    assert!(config.filename.starts_with("nemo-relay-events-"));
    assert!(config.filename.ends_with(".jsonl"));
    assert_eq!(
        config.filename.len(),
        "nemo-relay-events-YYYY-MM-DD-HH.MM.SS.jsonl".len()
    );
}

#[test]
fn append_mode_preserves_existing_lines() {
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

    let lines = read_jsonl(exporter.path());
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[0]["kind"], "scope");
    assert_eq!(lines[0]["scope_category"], "start");
    assert_eq!(lines[0]["category"], "agent");
    assert_eq!(lines[1]["kind"], "mark");
    assert_eq!(lines[1]["data"], json!({"step": 1}));
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

    let lines = read_jsonl(exporter.path());
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0], event.try_to_json_value().unwrap());
    assert!(lines[0].get("annotated_request").is_none());
    assert_eq!(
        lines[0]["category_profile"]["annotated_request"]["model"],
        "demo-model"
    );
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

    let lines = read_jsonl(exporter.path());
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

    let lines = read_jsonl(exporter.path());
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
fn force_flush_reports_stored_subscriber_failure() {
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
