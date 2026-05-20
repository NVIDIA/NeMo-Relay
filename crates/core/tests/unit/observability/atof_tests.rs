// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the ATOF JSONL exporter.

use super::*;
use crate::api::event::{
    BaseEvent, CategoryProfile, Event, EventCategory, MarkEvent, ScopeCategory, ScopeEvent,
};
use crate::api::runtime::NemoFlowContextState;
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
    let path = std::env::temp_dir().join(format!("nemo-flow-{prefix}-{id}"));
    fs::create_dir_all(&path).unwrap();
    path
}

fn reset_global() {
    crate::shared_runtime::reset_runtime_owner_for_tests();
    let context = global_context();
    *context.write().unwrap() = NemoFlowContextState::new();
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
    assert!(config.filename.starts_with("nemo-flow-events-"));
    assert!(config.filename.ends_with(".jsonl"));
    assert_eq!(
        config.filename.len(),
        "nemo-flow-events-YYYY-MM-DD-HH.MM.SS.jsonl".len()
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
