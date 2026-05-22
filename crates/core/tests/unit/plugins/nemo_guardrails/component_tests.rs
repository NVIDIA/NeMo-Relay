// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the planned NeMo Guardrails plugin component contract.
#![allow(clippy::await_holding_lock)]

use super::*;
use crate::api::runtime::NemoRelayContextState;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::Duration;

use crate::api::event::Event;
use crate::api::llm::{
    LlmAttributes, LlmCallExecuteParams, LlmRequest, LlmStreamCallExecuteParams, llm_call_execute,
    llm_stream_call_execute,
};
use crate::api::runtime::global_context;
use crate::api::runtime::{
    LlmExecutionNextFn, LlmJsonStream, LlmStreamExecutionNextFn, create_scope_stack,
    set_thread_scope_stack,
};
use crate::api::subscriber::{deregister_subscriber, register_subscriber};
use crate::api::tool::{ToolCallExecuteParams, tool_call_execute};
use crate::codec::openai_chat::{OpenAIChatCodec, OpenAIChatStreamingCodec};
use crate::codec::streaming::StreamingCodec;
use crate::codec::traits::LlmResponseCodec;
use crate::config_editor::{EditorConfig, EditorFieldKind};
#[cfg(feature = "schema")]
use crate::plugin::plugin_config_schema;
use crate::plugin::{
    PluginComponentSpec, PluginConfig, clear_plugin_configuration, initialize_plugins,
    list_plugin_kinds, lookup_plugin, validate_plugin_config,
};
use futures::StreamExt;
use serde_json::json;

const TEST_TIMEOUT: Duration = Duration::from_secs(5);

fn reset_runtime() {
    let _ = clear_plugin_configuration();
    crate::shared_runtime::reset_runtime_owner_for_tests();
    let context = global_context();
    *context.write().unwrap() = NemoRelayContextState::new();
}

fn setup_isolated_thread() {
    let stack = create_scope_stack();
    set_thread_scope_stack(stack);
}

fn component(config: Json) -> PluginComponentSpec {
    let Json::Object(config) = config else {
        panic!("component config must be an object");
    };
    PluginComponentSpec {
        kind: NEMO_GUARDRAILS_PLUGIN_KIND.to_string(),
        enabled: true,
        config,
    }
}

fn disabled_component(config: Json) -> PluginComponentSpec {
    let Json::Object(config) = config else {
        panic!("component config must be an object");
    };
    PluginComponentSpec {
        kind: NEMO_GUARDRAILS_PLUGIN_KIND.to_string(),
        enabled: false,
        config,
    }
}

fn plugin_config(config: Json) -> PluginConfig {
    PluginConfig {
        version: 1,
        components: vec![component(config)],
        policy: Default::default(),
    }
}

fn remote_valid_config() -> Json {
    json!({
        "mode": "remote",
        "codec": "openai_chat",
        "remote": {
            "endpoint": "http://localhost:8000",
            "config_id": "safety-default"
        }
    })
}

#[derive(Debug)]
struct CapturedHttpRequest {
    path: String,
    content_type: String,
    body: Vec<u8>,
}

fn spawn_http_responder(
    listener: TcpListener,
    response: Vec<u8>,
    request_tx: mpsc::Sender<CapturedHttpRequest>,
) {
    thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let request = read_http_request(&mut stream);
        stream.write_all(&response).unwrap();
        request_tx.send(request).unwrap();
    });
}

fn spawn_http_responder_sequence(
    listener: TcpListener,
    responses: Vec<Vec<u8>>,
    request_tx: mpsc::Sender<CapturedHttpRequest>,
) {
    thread::spawn(move || {
        for response in responses {
            let (mut stream, _) = listener.accept().unwrap();
            let request = read_http_request(&mut stream);
            stream.write_all(&response).unwrap();
            request_tx.send(request).unwrap();
        }
    });
}

fn read_http_request(stream: &mut impl Read) -> CapturedHttpRequest {
    let mut bytes = Vec::new();
    let mut buf = [0_u8; 4096];
    let (header_end, content_length) = read_http_headers(stream, &mut bytes, &mut buf);
    read_http_body(stream, &mut bytes, &mut buf, header_end + content_length);

    let headers_text = String::from_utf8_lossy(&bytes[..header_end]);
    let request_line = headers_text.lines().next().unwrap();
    CapturedHttpRequest {
        path: request_line.split_whitespace().nth(1).unwrap().to_string(),
        content_type: header_value(&headers_text, "content-type")
            .unwrap_or_default()
            .to_string(),
        body: bytes[header_end..header_end + content_length].to_vec(),
    }
}

fn read_http_headers(
    stream: &mut impl Read,
    bytes: &mut Vec<u8>,
    buf: &mut [u8; 4096],
) -> (usize, usize) {
    loop {
        let read = stream.read(buf).unwrap();
        if read == 0 {
            panic!("remote responder closed before receiving request");
        }
        bytes.extend_from_slice(&buf[..read]);

        if let Some(header_end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
            let header_end = header_end + 4;
            let headers_text = String::from_utf8_lossy(&bytes[..header_end]);
            let content_length = header_value(&headers_text, "content-length")
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(0);
            return (header_end, content_length);
        }
    }
}

fn read_http_body(
    stream: &mut impl Read,
    bytes: &mut Vec<u8>,
    buf: &mut [u8; 4096],
    expected_total: usize,
) {
    while bytes.len() < expected_total {
        let read = stream.read(buf).unwrap();
        if read == 0 {
            panic!("remote responder closed before full request body");
        }
        bytes.extend_from_slice(&buf[..read]);
    }
}

fn header_value<'a>(headers_text: &'a str, header_name: &str) -> Option<&'a str> {
    headers_text.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        if name.eq_ignore_ascii_case(header_name) {
            Some(value.trim())
        } else {
            None
        }
    })
}

fn recv_captured_request(request_rx: &mpsc::Receiver<CapturedHttpRequest>) -> CapturedHttpRequest {
    request_rx
        .recv_timeout(TEST_TIMEOUT)
        .expect("timed out waiting for captured HTTP request")
}

fn make_chat_request(stream: bool) -> LlmRequest {
    LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.2,
            "stream": stream
        }),
    }
}

fn capture_events(name: &str) -> Arc<Mutex<Vec<Event>>> {
    let events = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&events);
    register_subscriber(
        name,
        Arc::new(move |event| sink.lock().unwrap().push(event.clone())),
    )
    .unwrap();
    events
}

fn unused_local_endpoint() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);
    format!("http://{address}")
}

#[test]
fn editor_schema_tracks_nemo_guardrails_config_types() {
    let schema = NeMoGuardrailsConfig::editor_schema();
    let mode = schema.field("mode").expect("mode field");
    assert_eq!(mode.kind, EditorFieldKind::Enum);
    assert_eq!(mode.enum_values, &["remote", "local"]);

    let remote = schema.field("remote").expect("remote section");
    assert_eq!(remote.kind, EditorFieldKind::Section);
    assert!(remote.optional);

    let remote_schema = remote.schema().expect("remote editor schema");
    let headers = remote_schema.field("headers").expect("headers field");
    assert_eq!(headers.kind, EditorFieldKind::StringMap);

    let request_defaults = schema
        .field("request_defaults")
        .expect("request_defaults section");
    assert_eq!(request_defaults.kind, EditorFieldKind::Section);
    assert!(request_defaults.optional);

    let request_defaults_schema = request_defaults
        .schema()
        .expect("request_defaults editor schema");
    let rails = request_defaults_schema.field("rails").expect("rails field");
    assert_eq!(rails.kind, EditorFieldKind::Section);

    let rails_schema = rails.schema().expect("request rails editor schema");
    let retrieval = rails_schema.field("retrieval").expect("retrieval field");
    assert_eq!(retrieval.kind, EditorFieldKind::Json);
}

#[test]
fn default_config_and_component_conversion_cover_public_shape() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();

    let defaults = NeMoGuardrailsConfig::default();
    assert_eq!(defaults.version, 1);
    assert_eq!(defaults.mode, "remote");
    assert!(defaults.input);
    assert!(defaults.output);
    assert!(!defaults.tool_input);
    assert!(!defaults.tool_output);
    assert_eq!(defaults.priority, 100);
    assert!(defaults.remote.is_none());
    assert!(defaults.local.is_none());
    assert!(defaults.request_defaults.is_none());

    let remote = RemoteBackendConfig::default();
    assert_eq!(remote.timeout_millis, 3_000);
    assert!(remote.headers.is_empty());
    assert!(remote.config_ids.is_empty());

    let generic: PluginComponentSpec = ComponentSpec::new(NeMoGuardrailsConfig {
        remote: Some(RemoteBackendConfig {
            endpoint: Some("http://localhost:8000".into()),
            config_id: Some("default".into()),
            ..RemoteBackendConfig::default()
        }),
        ..NeMoGuardrailsConfig::default()
    })
    .into();
    assert_eq!(generic.kind, NEMO_GUARDRAILS_PLUGIN_KIND);
    assert!(generic.enabled);
    assert_eq!(generic.config["mode"], json!("remote"));
    assert_eq!(generic.config["remote"]["config_id"], json!("default"));
}

#[cfg(feature = "schema")]
fn schema_has_property(schema: &Json, name: &str) -> bool {
    schema_property(schema, name).is_some()
}

#[cfg(feature = "schema")]
fn schema_property_has_enum(schema: &Json, name: &str, expected: &[&str]) -> bool {
    schema_property(schema, name)
        .and_then(|property| property.get("enum"))
        .and_then(Json::as_array)
        .is_some_and(|values| {
            expected
                .iter()
                .all(|expected| values.iter().any(|value| value == *expected))
        })
}

#[cfg(feature = "schema")]
fn schema_property_has_default(schema: &Json, name: &str, expected: Json) -> bool {
    schema_property(schema, name)
        .and_then(|property| property.get("default"))
        .is_some_and(|default| default == &expected)
}

#[cfg(feature = "schema")]
fn schema_property<'a>(schema: &'a Json, name: &str) -> Option<&'a Json> {
    match schema {
        Json::Object(object) => {
            if let Some(property) = object
                .get("properties")
                .and_then(Json::as_object)
                .and_then(|properties| properties.get(name))
            {
                return Some(property);
            }
            object
                .values()
                .find_map(|value| schema_property(value, name))
        }
        Json::Array(values) => values.iter().find_map(|value| schema_property(value, name)),
        _ => None,
    }
}

#[cfg(feature = "schema")]
#[test]
fn schema_contains_every_supported_nemo_guardrails_option() {
    let schema = nemo_guardrails_config_schema();
    for field in [
        "version",
        "mode",
        "config_path",
        "config_yaml",
        "colang_content",
        "codec",
        "input",
        "output",
        "tool_input",
        "tool_output",
        "priority",
        "remote",
        "local",
        "request_defaults",
        "policy",
        "endpoint",
        "config_id",
        "config_ids",
        "headers",
        "timeout_millis",
        "python_module",
        "context",
        "thread_id",
        "state",
        "rails",
        "llm_params",
        "llm_output",
        "output_vars",
        "log",
        "retrieval",
        "dialog",
        "unknown_component",
        "unknown_field",
        "unsupported_value",
    ] {
        assert!(
            schema_has_property(&schema, field),
            "schema missing property `{field}`:\n{}",
            serde_json::to_string_pretty(&schema).unwrap()
        );
    }
    assert!(schema_property_has_enum(
        &schema,
        "mode",
        &["remote", "local"]
    ));
    assert!(schema_property_has_enum(
        &schema,
        "codec",
        &["openai_chat", "openai_responses", "anthropic_messages"]
    ));
    assert!(schema_property_has_default(
        &schema,
        "mode",
        json!("remote")
    ));
}

#[cfg(feature = "schema")]
#[test]
fn plugin_schema_contains_generic_plugin_surface() {
    let schema = plugin_config_schema();
    for field in [
        "version",
        "components",
        "policy",
        "kind",
        "enabled",
        "config",
    ] {
        assert!(
            schema_has_property(&schema, field),
            "plugin schema missing property `{field}`"
        );
    }
}

#[test]
fn builtin_registration_is_automatic() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();

    assert!(list_plugin_kinds().contains(&NEMO_GUARDRAILS_PLUGIN_KIND.to_string()));
    assert!(lookup_plugin(NEMO_GUARDRAILS_PLUGIN_KIND).is_some());
}

#[test]
fn disabled_component_validates_and_initializes_without_runtime_work() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();

    let config = PluginConfig {
        version: 1,
        components: vec![disabled_component(remote_valid_config())],
        policy: Default::default(),
    };
    assert!(!validate_plugin_config(&config).has_errors());
    futures::executor::block_on(initialize_plugins(config)).unwrap();
}

#[test]
fn duplicate_component_is_rejected_as_singleton() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();

    let config = PluginConfig {
        version: 1,
        components: vec![
            component(remote_valid_config()),
            component(remote_valid_config()),
        ],
        policy: Default::default(),
    };
    let report = validate_plugin_config(&config);
    assert!(report.has_errors());
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.code == "plugin.duplicate_component")
    );
}

#[test]
fn invalid_shapes_and_values_are_reported() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();

    let invalid_shape = validate_plugin_config(&plugin_config(json!({
        "version": "one",
    })));
    assert!(invalid_shape.has_errors());
    assert!(
        invalid_shape
            .diagnostics
            .iter()
            .any(|diag| diag.code == "nemo_guardrails.invalid_plugin_config")
    );

    let local_missing_source = validate_plugin_config(&plugin_config(json!({
        "mode": "local",
        "codec": "openai_chat",
    })));
    assert!(local_missing_source.has_errors());
    assert!(local_missing_source.diagnostics.iter().any(|diag| {
        diag.message
            .contains("exactly one of config_path or config_yaml is required in local mode")
    }));

    let local_bad_colang = validate_plugin_config(&plugin_config(json!({
        "mode": "local",
        "config_path": "./rails",
        "colang_content": "define flow x",
        "codec": "openai_chat",
    })));
    assert!(local_bad_colang.has_errors());
    assert!(
        local_bad_colang
            .diagnostics
            .iter()
            .any(|diag| diag.message.contains("colang_content can only be used"))
    );

    let remote_missing_identity = validate_plugin_config(&plugin_config(json!({
        "mode": "remote",
        "codec": "openai_chat",
        "remote": {"endpoint": "http://localhost:8000"},
    })));
    assert!(remote_missing_identity.has_errors());
    assert!(remote_missing_identity.diagnostics.iter().any(|diag| {
        diag.message
            .contains("remote mode requires remote.config_id or remote.config_ids")
    }));

    let remote_conflicting_ids = validate_plugin_config(&plugin_config(json!({
        "mode": "remote",
        "codec": "openai_chat",
        "remote": {
            "endpoint": "http://localhost:8000",
            "config_id": "one",
            "config_ids": ["two"]
        },
    })));
    assert!(remote_conflicting_ids.has_errors());
    assert!(remote_conflicting_ids.diagnostics.iter().any(|diag| {
        diag.message
            .contains("remote.config_id and remote.config_ids cannot be used together")
    }));

    let missing_codec = validate_plugin_config(&plugin_config(json!({
        "mode": "remote",
        "remote": {
            "endpoint": "http://localhost:8000",
            "config_id": "default"
        }
    })));
    assert!(missing_codec.has_errors());
    assert!(
        missing_codec
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("codec"))
    );

    let bad_codec = validate_plugin_config(&plugin_config(json!({
        "mode": "remote",
        "codec": "openai_agents",
        "remote": {
            "endpoint": "http://localhost:8000",
            "config_id": "default"
        }
    })));
    assert!(bad_codec.has_errors());
    assert!(bad_codec.diagnostics.iter().any(|diag| {
        diag.message
            .contains("codec must be 'openai_chat', 'openai_responses', or 'anthropic_messages'")
    }));

    let unsupported_remote_codec = validate_plugin_config(&plugin_config(json!({
        "mode": "remote",
        "codec": "openai_responses",
        "remote": {
            "endpoint": "http://localhost:8000",
            "config_id": "default"
        }
    })));
    assert!(unsupported_remote_codec.has_errors());
    assert!(unsupported_remote_codec.diagnostics.iter().any(|diag| {
        diag.message
            .contains("remote mode currently supports only codec = 'openai_chat'")
    }));

    let unsupported_remote_anthropic_codec = validate_plugin_config(&plugin_config(json!({
        "mode": "remote",
        "codec": "anthropic_messages",
        "remote": {
            "endpoint": "http://localhost:8000",
            "config_id": "default"
        }
    })));
    assert!(unsupported_remote_anthropic_codec.has_errors());
    assert!(
        unsupported_remote_anthropic_codec
            .diagnostics
            .iter()
            .any(|diag| {
                diag.message
                    .contains("remote mode currently supports only codec = 'openai_chat'")
            })
    );

    let supported_remote_tool_surface = validate_plugin_config(&plugin_config(json!({
        "mode": "remote",
        "codec": "openai_chat",
        "tool_input": true,
        "remote": {
            "endpoint": "http://localhost:8000",
            "config_id": "default"
        }
    })));
    assert!(!supported_remote_tool_surface.has_errors());

    let remote_empty_fields = validate_plugin_config(&plugin_config(json!({
        "mode": "remote",
        "codec": "openai_chat",
        "remote": {
            "endpoint": "",
            "config_id": "",
            "config_ids": ["default", ""]
        }
    })));
    assert!(remote_empty_fields.has_errors());
    assert!(
        remote_empty_fields
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("remote.endpoint"))
    );
    assert!(
        remote_empty_fields
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("remote.config_id"))
    );
    assert!(
        remote_empty_fields
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("remote.config_ids[1]"))
    );

    let remote_local_mix = validate_plugin_config(&plugin_config(json!({
        "mode": "remote",
        "config_path": "./rails",
        "codec": "openai_chat",
        "remote": {
            "endpoint": "http://localhost:8000",
            "config_id": "default"
        },
        "local": {"python_module": "nemoguardrails"}
    })));
    assert!(remote_local_mix.has_errors());
    assert!(
        remote_local_mix
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("local"))
    );
    assert!(remote_local_mix.diagnostics.iter().any(|diag| {
        diag.message
            .contains("remote mode uses remote config identity")
    }));

    let no_surfaces = validate_plugin_config(&plugin_config(json!({
        "mode": "local",
        "config_path": "./rails",
        "input": false,
        "output": false,
        "tool_input": false,
        "tool_output": false
    })));
    assert!(no_surfaces.has_errors());
    assert!(
        no_surfaces
            .diagnostics
            .iter()
            .any(|diag| diag.message.contains("at least one Guardrails surface"))
    );

    let local_empty_fields = validate_plugin_config(&plugin_config(json!({
        "mode": "local",
        "config_yaml": "",
        "colang_content": "",
        "codec": "openai_chat",
        "local": {"python_module": ""}
    })));
    assert!(local_empty_fields.has_errors());
    assert!(
        local_empty_fields
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("config_yaml"))
    );
    assert!(
        local_empty_fields
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("colang_content"))
    );
    assert!(
        local_empty_fields
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("local.python_module"))
    );

    let invalid_request_defaults = validate_plugin_config(&plugin_config(json!({
        "mode": "remote",
        "codec": "openai_chat",
        "remote": {
            "endpoint": "http://localhost:8000",
            "config_id": "default"
        },
        "request_defaults": {
            "context": true,
            "thread_id": "short",
            "state": {"foo": "bar"},
            "llm_params": [],
            "log": "verbose",
            "output_vars": 7,
            "rails": {
                "retrieval": [""]
            }
        }
    })));
    assert!(invalid_request_defaults.has_errors());
    assert!(
        invalid_request_defaults
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("request_defaults.context"))
    );
    assert!(
        invalid_request_defaults
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("request_defaults.thread_id"))
    );
    assert!(invalid_request_defaults.diagnostics.iter().any(|diag| {
        diag.message
            .contains("request_defaults.thread_id must be at least 16 characters long")
    }));
    assert!(
        invalid_request_defaults
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("request_defaults.state"))
    );
    assert!(invalid_request_defaults.diagnostics.iter().any(|diag| {
        diag.message
            .contains("request_defaults.state must be empty or contain only 'events' or 'state'")
    }));
    assert!(
        invalid_request_defaults
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("request_defaults.llm_params"))
    );
    assert!(
        invalid_request_defaults
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("request_defaults.log"))
    );
    assert!(
        invalid_request_defaults
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("request_defaults.output_vars"))
    );
    assert!(
        invalid_request_defaults
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("request_defaults.rails.retrieval[0]"))
    );
}

#[test]
fn unknown_fields_follow_policy() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();

    let warn_report = validate_plugin_config(&plugin_config(json!({
        "mode": "remote",
        "codec": "openai_chat",
        "remote": {"endpoint": "http://localhost:8000", "config_id": "default"},
        "bogus": true
    })));
    assert!(
        warn_report
            .diagnostics
            .iter()
            .any(|diag| diag.code == "nemo_guardrails.unknown_field")
    );

    let nested_warn_report = validate_plugin_config(&plugin_config(json!({
        "mode": "remote",
        "codec": "openai_chat",
        "remote": {"endpoint": "http://localhost:8000", "config_id": "default"},
        "request_defaults": {
            "rails": {
                "bogus": true
            }
        }
    })));
    assert!(
        nested_warn_report
            .diagnostics
            .iter()
            .any(|diag| diag.component.as_deref() == Some("request_defaults.rails"))
    );

    let ignored = validate_plugin_config(&plugin_config(json!({
        "policy": {"unknown_field": "ignore", "unsupported_value": "ignore"},
        "mode": "remote",
        "codec": "openai_chat",
        "remote": {"endpoint": "http://localhost:8000", "config_id": "default"},
        "bogus": true
    })));
    assert!(!ignored.has_errors());
    assert!(ignored.diagnostics.is_empty());
}

#[test]
fn enabled_local_initialization_fails_fast_until_backend_exists() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();

    let error = futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "mode": "local",
        "codec": "openai_chat",
        "config_path": "./rails"
    }))))
    .unwrap_err();

    match error {
        crate::plugin::PluginError::RegistrationFailed(message) => {
            assert!(message.contains("local backend"));
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[tokio::test]
async fn remote_initialization_installs_non_streaming_execution_intercept() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    setup_isolated_thread();
    let events = capture_events("nemo-guardrails-remote-execution-events");

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (request_tx, request_rx) = mpsc::channel();
    let response_body = json!({
        "id": "chatcmpl-remote",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "guarded"},
            "finish_reason": "stop"
        }],
        "guardrails": {
            "config_id": "safety-default",
            "state": {"state": {"conversation": "server-state"}},
            "output_data": {"decision": "allow"}
        }
    })
    .to_string();
    let http_response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        response_body.len(),
        response_body
    )
    .into_bytes();
    spawn_http_responder(listener, http_response, request_tx);

    initialize_plugins(plugin_config(json!({
        "mode": "remote",
        "codec": "openai_chat",
        "remote": {
            "endpoint": format!("http://{address}"),
            "config_id": "safety-default",
            "headers": {"x-guardrails-auth": "token"},
            "timeout_millis": 5_000
        },
        "request_defaults": {
            "context": {"tenant": "acme"},
            "thread_id": "thread-1234567890",
            "state": {"state": {"conversation": "client-state"}},
            "rails": {"input": true, "retrieval": ["kb"]},
            "llm_params": {"temperature": 0.1},
            "llm_output": true,
            "output_vars": ["answer"],
            "log": {"activated_rails": true}
        }
    })))
    .await
    .unwrap();

    let original_called = Arc::new(AtomicBool::new(false));
    let called = Arc::clone(&original_called);
    let func: LlmExecutionNextFn = Arc::new(move |_req| {
        called.store(true, Ordering::SeqCst);
        Box::pin(async move { Ok(json!({"response": "original"})) })
    });

    let response = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("openai")
            .request(make_chat_request(false))
            .func(func)
            .attributes(LlmAttributes::empty())
            .response_codec(Arc::new(OpenAIChatCodec) as Arc<dyn LlmResponseCodec>)
            .build(),
    )
    .await
    .unwrap();

    assert!(!original_called.load(Ordering::SeqCst));
    assert_eq!(response["id"], json!("chatcmpl-remote"));
    assert_eq!(response["object"], json!("chat.completion"));
    assert_eq!(response["model"], json!("gpt-4o-mini"));
    assert_eq!(
        response["choices"][0]["message"]["content"],
        json!("guarded")
    );
    assert_eq!(
        response["guardrails"]["output_data"]["decision"],
        json!("allow")
    );
    assert_eq!(
        response["guardrails"]["state"]["state"]["conversation"],
        json!("server-state")
    );

    let captured = recv_captured_request(&request_rx);
    assert_eq!(captured.path, "/v1/chat/completions");
    assert!(captured.content_type.starts_with("application/json"));

    let request_json: Json = serde_json::from_slice(&captured.body).unwrap();
    assert_eq!(request_json["messages"][0]["content"], json!("hello"));
    assert_eq!(request_json["stream"], json!(false));
    assert_eq!(
        request_json["guardrails"]["config_id"],
        json!("safety-default")
    );
    assert_eq!(
        request_json["guardrails"]["context"]["tenant"],
        json!("acme")
    );
    assert_eq!(
        request_json["guardrails"]["thread_id"],
        json!("thread-1234567890")
    );
    assert_eq!(
        request_json["guardrails"]["state"]["state"]["conversation"],
        json!("client-state")
    );
    assert_eq!(
        request_json["guardrails"]["options"]["rails"]["retrieval"],
        json!(["kb"])
    );
    assert_eq!(
        request_json["guardrails"]["options"]["llm_output"],
        json!(true)
    );

    let captured_events = events.lock().unwrap().clone();
    let mark_names: Vec<_> = captured_events
        .iter()
        .filter(|event| event.kind() == "mark")
        .map(|event| event.name().to_string())
        .collect();
    assert!(mark_names.contains(&"nemo_guardrails.remote.start".to_string()));
    assert!(mark_names.contains(&"nemo_guardrails.remote.end".to_string()));

    let start_mark = captured_events
        .iter()
        .find(|event| event.name() == "nemo_guardrails.remote.start")
        .unwrap();
    assert_eq!(
        start_mark.data().unwrap()["config_id"],
        json!("safety-default")
    );
    assert_eq!(start_mark.data().unwrap()["stream"], json!(false));

    let end_mark = captured_events
        .iter()
        .find(|event| event.name() == "nemo_guardrails.remote.end")
        .unwrap();
    assert_eq!(end_mark.data().unwrap()["http_status"], json!(200));
    assert_eq!(end_mark.data().unwrap()["stream"], json!(false));

    deregister_subscriber("nemo-guardrails-remote-execution-events").unwrap();
}

#[tokio::test]
async fn remote_request_uses_config_ids_when_config_id_is_not_set() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    setup_isolated_thread();

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (request_tx, request_rx) = mpsc::channel();
    let response_body = json!({
        "id": "chatcmpl-remote",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "guarded"},
            "finish_reason": "stop"
        }]
    })
    .to_string();
    let http_response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        response_body.len(),
        response_body
    )
    .into_bytes();
    spawn_http_responder(listener, http_response, request_tx);

    initialize_plugins(plugin_config(json!({
        "mode": "remote",
        "codec": "openai_chat",
        "remote": {
            "endpoint": format!("http://{address}"),
            "config_ids": ["safety-a", "safety-b"]
        }
    })))
    .await
    .unwrap();

    let func: LlmExecutionNextFn =
        Arc::new(move |_req| Box::pin(async move { Ok(json!({"response": "original"})) }));

    let _ = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("openai")
            .request(make_chat_request(false))
            .func(func)
            .attributes(LlmAttributes::empty())
            .response_codec(Arc::new(OpenAIChatCodec) as Arc<dyn LlmResponseCodec>)
            .build(),
    )
    .await
    .unwrap();

    let captured = recv_captured_request(&request_rx);
    let request_json: Json = serde_json::from_slice(&captured.body).unwrap();
    assert_eq!(
        request_json["guardrails"]["config_ids"],
        json!(["safety-a", "safety-b"])
    );
    assert!(request_json["guardrails"].get("config_id").is_none());
}

#[tokio::test]
async fn remote_initialization_installs_stream_execution_intercept() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    setup_isolated_thread();
    let events = capture_events("nemo-guardrails-remote-stream-events");

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (request_tx, request_rx) = mpsc::channel();
    let sse_body = concat!(
        "data: {\"id\":\"chatcmpl-remote\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"guard\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-remote\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ed\"},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n"
    );
    let http_response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n{}",
        sse_body.len(),
        sse_body
    )
    .into_bytes();
    spawn_http_responder(listener, http_response, request_tx);

    initialize_plugins(plugin_config(json!({
        "mode": "remote",
        "codec": "openai_chat",
        "remote": {
            "endpoint": format!("http://{address}"),
            "config_id": "safety-default"
        }
    })))
    .await
    .unwrap();

    let original_called = Arc::new(AtomicBool::new(false));
    let called = Arc::clone(&original_called);
    let func: LlmStreamExecutionNextFn = Arc::new(move |_req| {
        called.store(true, Ordering::SeqCst);
        Box::pin(async move {
            let stream = tokio_stream::iter(vec![Ok(json!({"chunk": "original"}))]);
            Ok(Box::pin(stream) as LlmJsonStream)
        })
    });

    let streaming_codec = OpenAIChatStreamingCodec::new();
    let collector = streaming_codec.collector();
    let finalizer = streaming_codec.finalizer();
    let response_codec: Arc<dyn LlmResponseCodec> = Arc::new(OpenAIChatCodec);

    let mut stream = llm_stream_call_execute(
        LlmStreamCallExecuteParams::builder()
            .name("openai")
            .request(make_chat_request(true))
            .func(func)
            .collector(collector)
            .finalizer(finalizer)
            .attributes(LlmAttributes::STREAMING)
            .response_codec(response_codec)
            .build(),
    )
    .await
    .unwrap();

    let mut chunks = Vec::new();
    while let Some(chunk) = tokio::time::timeout(TEST_TIMEOUT, stream.next())
        .await
        .expect("timed out waiting for remote stream chunk")
    {
        chunks.push(chunk.unwrap());
    }

    assert!(!original_called.load(Ordering::SeqCst));
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0]["choices"][0]["delta"]["content"], json!("guard"));
    assert_eq!(chunks[1]["choices"][0]["delta"]["content"], json!("ed"));

    let captured = recv_captured_request(&request_rx);
    let request_json: Json = serde_json::from_slice(&captured.body).unwrap();
    assert_eq!(request_json["stream"], json!(true));
    assert_eq!(
        request_json["guardrails"]["config_id"],
        json!("safety-default")
    );

    let captured_events = events.lock().unwrap().clone();
    let start_mark = captured_events
        .iter()
        .find(|event| event.name() == "nemo_guardrails.remote.start")
        .unwrap();
    assert_eq!(start_mark.data().unwrap()["stream"], json!(true));

    let end_mark = captured_events
        .iter()
        .find(|event| event.name() == "nemo_guardrails.remote.end")
        .unwrap();
    assert_eq!(end_mark.data().unwrap()["http_status"], json!(200));
    assert_eq!(end_mark.data().unwrap()["stream"], json!(true));

    deregister_subscriber("nemo-guardrails-remote-stream-events").unwrap();
}

#[tokio::test]
async fn remote_non_streaming_http_errors_are_reported_and_marked() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    setup_isolated_thread();
    let events = capture_events("nemo-guardrails-remote-error-events");

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (request_tx, _request_rx) = mpsc::channel();
    let response_body = r#"{"error":"backend unavailable"}"#;
    let http_response = format!(
        "HTTP/1.1 502 Bad Gateway\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        response_body.len(),
        response_body
    )
    .into_bytes();
    spawn_http_responder(listener, http_response, request_tx);

    initialize_plugins(plugin_config(json!({
        "mode": "remote",
        "codec": "openai_chat",
        "remote": {
            "endpoint": format!("http://{address}"),
            "config_id": "safety-default"
        }
    })))
    .await
    .unwrap();

    let original_called = Arc::new(AtomicBool::new(false));
    let called = Arc::clone(&original_called);
    let func: LlmExecutionNextFn = Arc::new(move |_req| {
        called.store(true, Ordering::SeqCst);
        Box::pin(async move { Ok(json!({"response": "original"})) })
    });

    let error = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("openai")
            .request(make_chat_request(false))
            .func(func)
            .attributes(LlmAttributes::empty())
            .response_codec(Arc::new(OpenAIChatCodec) as Arc<dyn LlmResponseCodec>)
            .build(),
    )
    .await
    .unwrap_err();

    assert!(!original_called.load(Ordering::SeqCst));
    match error {
        crate::error::FlowError::Internal(message) => {
            assert!(message.contains("status 502"));
            assert!(message.contains("backend unavailable"));
        }
        other => panic!("unexpected error: {other}"),
    }

    let captured_events = events.lock().unwrap().clone();
    assert!(
        captured_events
            .iter()
            .any(|event| event.name() == "nemo_guardrails.remote.start")
    );
    let error_mark = captured_events
        .iter()
        .find(|event| event.name() == "nemo_guardrails.remote.error")
        .unwrap();
    assert_eq!(error_mark.data().unwrap()["http_status"], json!(502));
    assert_eq!(error_mark.data().unwrap()["stream"], json!(false));
    assert!(
        error_mark.data().unwrap()["error"]
            .as_str()
            .unwrap()
            .contains("error body omitted from marks")
    );

    deregister_subscriber("nemo-guardrails-remote-error-events").unwrap();
}

#[tokio::test]
async fn remote_streaming_http_errors_are_reported_and_marked() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    setup_isolated_thread();
    let events = capture_events("nemo-guardrails-remote-stream-error-events");

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (request_tx, _request_rx) = mpsc::channel();
    let response_body = r#"{"error":"stream backend unavailable"}"#;
    let http_response = format!(
        "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        response_body.len(),
        response_body
    )
    .into_bytes();
    spawn_http_responder(listener, http_response, request_tx);

    initialize_plugins(plugin_config(json!({
        "mode": "remote",
        "codec": "openai_chat",
        "remote": {
            "endpoint": format!("http://{address}"),
            "config_id": "safety-default"
        }
    })))
    .await
    .unwrap();

    let original_called = Arc::new(AtomicBool::new(false));
    let called = Arc::clone(&original_called);
    let func: LlmStreamExecutionNextFn = Arc::new(move |_req| {
        called.store(true, Ordering::SeqCst);
        Box::pin(async move {
            let stream = tokio_stream::iter(vec![Ok(json!({"chunk": "original"}))]);
            Ok(Box::pin(stream) as LlmJsonStream)
        })
    });

    let streaming_codec = OpenAIChatStreamingCodec::new();
    let collector = streaming_codec.collector();
    let finalizer = streaming_codec.finalizer();
    let response_codec: Arc<dyn LlmResponseCodec> = Arc::new(OpenAIChatCodec);

    let error = match llm_stream_call_execute(
        LlmStreamCallExecuteParams::builder()
            .name("openai")
            .request(make_chat_request(true))
            .func(func)
            .collector(collector)
            .finalizer(finalizer)
            .attributes(LlmAttributes::STREAMING)
            .response_codec(response_codec)
            .build(),
    )
    .await
    {
        Ok(_) => panic!("expected remote streaming request to fail"),
        Err(error) => error,
    };

    assert!(!original_called.load(Ordering::SeqCst));
    match error {
        crate::error::FlowError::Internal(message) => {
            assert!(message.contains("status 503"));
            assert!(message.contains("stream backend unavailable"));
        }
        other => panic!("unexpected error: {other}"),
    }

    let captured_events = events.lock().unwrap().clone();
    assert!(
        captured_events
            .iter()
            .any(|event| event.name() == "nemo_guardrails.remote.start")
    );
    let error_mark = captured_events
        .iter()
        .find(|event| event.name() == "nemo_guardrails.remote.error")
        .unwrap();
    assert_eq!(error_mark.data().unwrap()["http_status"], json!(503));
    assert_eq!(error_mark.data().unwrap()["stream"], json!(true));
    assert!(
        error_mark.data().unwrap()["error"]
            .as_str()
            .unwrap()
            .contains("error body omitted from marks")
    );

    deregister_subscriber("nemo-guardrails-remote-stream-error-events").unwrap();
}

#[tokio::test]
async fn remote_non_streaming_invalid_json_is_reported_and_marked() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    setup_isolated_thread();
    let events = capture_events("nemo-guardrails-remote-invalid-json-events");

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (request_tx, _request_rx) = mpsc::channel();
    let response_body = "{not-json}";
    let http_response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        response_body.len(),
        response_body
    )
    .into_bytes();
    spawn_http_responder(listener, http_response, request_tx);

    initialize_plugins(plugin_config(json!({
        "mode": "remote",
        "codec": "openai_chat",
        "remote": {
            "endpoint": format!("http://{address}"),
            "config_id": "safety-default"
        }
    })))
    .await
    .unwrap();

    let func: LlmExecutionNextFn =
        Arc::new(move |_req| Box::pin(async move { Ok(json!({"response": "original"})) }));

    let error = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("openai")
            .request(make_chat_request(false))
            .func(func)
            .attributes(LlmAttributes::empty())
            .response_codec(Arc::new(OpenAIChatCodec) as Arc<dyn LlmResponseCodec>)
            .build(),
    )
    .await
    .unwrap_err();

    match error {
        crate::error::FlowError::Internal(message) => {
            assert!(message.contains("failed to parse remote response JSON"));
        }
        other => panic!("unexpected error: {other}"),
    }

    let captured_events = events.lock().unwrap().clone();
    let error_mark = captured_events
        .iter()
        .find(|event| event.name() == "nemo_guardrails.remote.error")
        .unwrap();
    assert_eq!(error_mark.data().unwrap()["http_status"], json!(200));
    assert_eq!(error_mark.data().unwrap()["stream"], json!(false));

    deregister_subscriber("nemo-guardrails-remote-invalid-json-events").unwrap();
}

#[tokio::test]
async fn remote_streaming_malformed_chunk_is_reported_and_marked() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    setup_isolated_thread();
    let events = capture_events("nemo-guardrails-remote-malformed-stream-events");

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (request_tx, _request_rx) = mpsc::channel();
    let sse_body = "data: {not-json}\n\n";
    let http_response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n{}",
        sse_body.len(),
        sse_body
    )
    .into_bytes();
    spawn_http_responder(listener, http_response, request_tx);

    initialize_plugins(plugin_config(json!({
        "mode": "remote",
        "codec": "openai_chat",
        "remote": {
            "endpoint": format!("http://{address}"),
            "config_id": "safety-default"
        }
    })))
    .await
    .unwrap();

    let func: LlmStreamExecutionNextFn = Arc::new(move |_req| {
        Box::pin(async move {
            let stream = tokio_stream::iter(vec![Ok(json!({"chunk": "original"}))]);
            Ok(Box::pin(stream) as LlmJsonStream)
        })
    });

    let streaming_codec = OpenAIChatStreamingCodec::new();
    let collector = streaming_codec.collector();
    let finalizer = streaming_codec.finalizer();
    let response_codec: Arc<dyn LlmResponseCodec> = Arc::new(OpenAIChatCodec);

    let mut stream = llm_stream_call_execute(
        LlmStreamCallExecuteParams::builder()
            .name("openai")
            .request(make_chat_request(true))
            .func(func)
            .collector(collector)
            .finalizer(finalizer)
            .attributes(LlmAttributes::STREAMING)
            .response_codec(response_codec)
            .build(),
    )
    .await
    .unwrap();

    let error = tokio::time::timeout(TEST_TIMEOUT, stream.next())
        .await
        .expect("timed out waiting for remote stream error")
        .unwrap()
        .unwrap_err();
    match error {
        crate::error::FlowError::Internal(message) => {
            assert!(!message.is_empty());
        }
        other => panic!("unexpected error: {other}"),
    }

    let captured_events = events.lock().unwrap().clone();
    let error_mark = captured_events
        .iter()
        .find(|event| event.name() == "nemo_guardrails.remote.error")
        .unwrap();
    assert_eq!(error_mark.data().unwrap()["http_status"], json!(200));
    assert_eq!(error_mark.data().unwrap()["stream"], json!(true));

    deregister_subscriber("nemo-guardrails-remote-malformed-stream-events").unwrap();
}

#[tokio::test]
async fn remote_preflight_tool_choice_failure_is_reported_and_marked() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    setup_isolated_thread();
    let events = capture_events("nemo-guardrails-remote-preflight-error-events");

    initialize_plugins(plugin_config(json!({
        "mode": "remote",
        "codec": "openai_chat",
        "remote": {
            "endpoint": unused_local_endpoint(),
            "config_id": "safety-default"
        }
    })))
    .await
    .unwrap();

    let func: LlmExecutionNextFn =
        Arc::new(move |_req| Box::pin(async move { Ok(json!({"response": "original"})) }));
    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Lookup data",
                    "parameters": {"type": "object"}
                }
            }]
        }),
    };

    let error = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("openai")
            .request(request)
            .func(func)
            .attributes(LlmAttributes::empty())
            .response_codec(Arc::new(OpenAIChatCodec) as Arc<dyn LlmResponseCodec>)
            .build(),
    )
    .await
    .unwrap_err();

    match error {
        crate::error::FlowError::Internal(message) => {
            assert!(message.contains("does not support OpenAI tool definitions or tool_choice"));
        }
        other => panic!("unexpected error: {other}"),
    }

    let captured_events = events.lock().unwrap().clone();
    assert!(
        captured_events
            .iter()
            .any(|event| event.name() == "nemo_guardrails.remote.start")
    );
    let error_mark = captured_events
        .iter()
        .find(|event| event.name() == "nemo_guardrails.remote.error")
        .unwrap();
    assert_eq!(error_mark.data().unwrap()["stream"], json!(false));
    assert!(
        error_mark.data().unwrap()["error"]
            .as_str()
            .unwrap()
            .contains("does not support OpenAI tool definitions or tool_choice")
    );

    deregister_subscriber("nemo-guardrails-remote-preflight-error-events").unwrap();
}

#[tokio::test]
async fn remote_transport_failure_is_reported_and_marked() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    setup_isolated_thread();
    let events = capture_events("nemo-guardrails-remote-transport-error-events");

    initialize_plugins(plugin_config(json!({
        "mode": "remote",
        "codec": "openai_chat",
        "remote": {
            "endpoint": unused_local_endpoint(),
            "config_id": "safety-default",
            "timeout_millis": 50
        }
    })))
    .await
    .unwrap();

    let func: LlmExecutionNextFn =
        Arc::new(move |_req| Box::pin(async move { Ok(json!({"response": "original"})) }));

    let error = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("openai")
            .request(make_chat_request(false))
            .func(func)
            .attributes(LlmAttributes::empty())
            .response_codec(Arc::new(OpenAIChatCodec) as Arc<dyn LlmResponseCodec>)
            .build(),
    )
    .await
    .unwrap_err();

    match error {
        crate::error::FlowError::Internal(message) => {
            assert!(message.contains("remote request failed"));
        }
        other => panic!("unexpected error: {other}"),
    }

    let captured_events = events.lock().unwrap().clone();
    let error_mark = captured_events
        .iter()
        .find(|event| event.name() == "nemo_guardrails.remote.error")
        .unwrap();
    assert_eq!(error_mark.data().unwrap()["stream"], json!(false));
    assert!(error_mark.data().unwrap().get("http_status").is_none());

    deregister_subscriber("nemo-guardrails-remote-transport-error-events").unwrap();
}

#[tokio::test]
async fn remote_success_without_guardrails_payload_is_allowed() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    setup_isolated_thread();

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (request_tx, _request_rx) = mpsc::channel();
    let response_body = json!({
        "id": "chatcmpl-remote",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "guarded"},
            "finish_reason": "stop"
        }]
    })
    .to_string();
    let http_response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        response_body.len(),
        response_body
    )
    .into_bytes();
    spawn_http_responder(listener, http_response, request_tx);

    initialize_plugins(plugin_config(json!({
        "mode": "remote",
        "codec": "openai_chat",
        "remote": {
            "endpoint": format!("http://{address}"),
            "config_id": "safety-default"
        }
    })))
    .await
    .unwrap();

    let func: LlmExecutionNextFn =
        Arc::new(move |_req| Box::pin(async move { Ok(json!({"response": "original"})) }));

    let response = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("openai")
            .request(make_chat_request(false))
            .func(func)
            .attributes(LlmAttributes::empty())
            .response_codec(Arc::new(OpenAIChatCodec) as Arc<dyn LlmResponseCodec>)
            .build(),
    )
    .await
    .unwrap();

    assert_eq!(response["id"], json!("chatcmpl-remote"));
    assert!(response.get("guardrails").is_none());
}

#[tokio::test]
async fn remote_tool_input_block_rejects_before_tool_execution() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    setup_isolated_thread();
    let events = capture_events("nemo-guardrails-remote-tool-input-events");

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (request_tx, request_rx) = mpsc::channel();
    let response_body = json!({
        "id": "chatcmpl-tool-input-blocked",
        "object": "chat.completion",
        "created": 1,
        "model": "",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "blocked"},
            "finish_reason": "stop"
        }],
        "guardrails": {
            "config_id": "safety-default",
            "log": {
                "activated_rails": [{
                    "name": "tool_input_block",
                    "stop": true
                }]
            }
        }
    })
    .to_string();
    let http_response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        response_body.len(),
        response_body
    )
    .into_bytes();
    spawn_http_responder(listener, http_response, request_tx);

    initialize_plugins(plugin_config(json!({
        "mode": "remote",
        "input": false,
        "output": false,
        "tool_input": true,
        "remote": {
            "endpoint": format!("http://{address}"),
            "config_id": "safety-default"
        }
    })))
    .await
    .unwrap();

    let original_called = Arc::new(AtomicBool::new(false));
    let called = Arc::clone(&original_called);
    let error = tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("weather_lookup")
            .args(json!({"city": "Phoenix"}))
            .func(Arc::new(move |_args| {
                called.store(true, Ordering::SeqCst);
                Box::pin(async move { Ok(json!({"forecast": "sunny"})) })
            }))
            .build(),
    )
    .await
    .unwrap_err();

    assert!(!original_called.load(Ordering::SeqCst));
    match error {
        crate::error::FlowError::GuardrailRejected(message) => {
            assert!(message.contains("tool_input"));
        }
        other => panic!("unexpected error: {other}"),
    }

    let captured = recv_captured_request(&request_rx);
    let request_json: Json = serde_json::from_slice(&captured.body).unwrap();
    assert_eq!(
        request_json["guardrails"]["options"]["rails"]["tool_input"],
        json!(true)
    );
    assert_eq!(
        request_json["guardrails"]["options"]["rails"]["tool_output"],
        json!(false)
    );

    let captured_events = events.lock().unwrap().clone();
    let start_mark = captured_events
        .iter()
        .find(|event| event.name() == "nemo_guardrails.remote.start")
        .unwrap();
    assert_eq!(start_mark.data().unwrap()["surface"], json!("tool_input"));
    assert_eq!(
        start_mark.data().unwrap()["tool_name"],
        json!("weather_lookup")
    );
    let end_mark = captured_events
        .iter()
        .find(|event| event.name() == "nemo_guardrails.remote.end")
        .unwrap();
    assert_eq!(end_mark.data().unwrap()["surface"], json!("tool_input"));

    deregister_subscriber("nemo-guardrails-remote-tool-input-events").unwrap();
}

#[tokio::test]
async fn remote_tool_input_can_rewrite_tool_arguments() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    setup_isolated_thread();

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (request_tx, request_rx) = mpsc::channel();
    let response_body = json!({
        "id": "chatcmpl-tool-input-modified",
        "object": "chat.completion",
        "created": 1,
        "model": "",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "{\"tool_name\":\"weather_lookup\",\"arguments\":{\"city\":\"Boston\"}}"
            },
            "finish_reason": "stop"
        }],
        "guardrails": {
            "config_id": "safety-default",
            "log": {
                "activated_rails": []
            }
        }
    })
    .to_string();
    let http_response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        response_body.len(),
        response_body
    )
    .into_bytes();
    spawn_http_responder(listener, http_response, request_tx);

    initialize_plugins(plugin_config(json!({
        "mode": "remote",
        "input": false,
        "output": false,
        "tool_input": true,
        "remote": {
            "endpoint": format!("http://{address}"),
            "config_id": "safety-default"
        }
    })))
    .await
    .unwrap();

    let seen_args = Arc::new(Mutex::new(None::<Json>));
    let seen_args_for_call = Arc::clone(&seen_args);
    let result = tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("weather_lookup")
            .args(json!({"city": "Phoenix"}))
            .func(Arc::new(move |args| {
                *seen_args_for_call.lock().unwrap() = Some(args.clone());
                Box::pin(async move { Ok(json!({"forecast": "sunny"})) })
            }))
            .build(),
    )
    .await
    .unwrap();

    assert_eq!(result, json!({"forecast": "sunny"}));
    assert_eq!(*seen_args.lock().unwrap(), Some(json!({"city": "Boston"})));

    let captured = recv_captured_request(&request_rx);
    let request_json: Json = serde_json::from_slice(&captured.body).unwrap();
    assert_eq!(request_json["messages"][0]["role"], json!("user"));
}

#[tokio::test]
async fn remote_tool_output_can_rewrite_tool_result() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    setup_isolated_thread();

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (request_tx, request_rx) = mpsc::channel();
    let response_body = json!({
        "id": "chatcmpl-tool-output-modified",
        "object": "chat.completion",
        "created": 1,
        "model": "",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "{\"tool_name\":\"weather_lookup\",\"arguments\":{\"city\":\"Phoenix\"},\"result\":{\"forecast\":\"cloudy\"}}"
            },
            "finish_reason": "stop"
        }],
        "guardrails": {
            "config_id": "safety-default",
            "log": {
                "activated_rails": []
            }
        }
    })
    .to_string();
    let http_response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        response_body.len(),
        response_body
    )
    .into_bytes();
    spawn_http_responder(listener, http_response, request_tx);

    initialize_plugins(plugin_config(json!({
        "mode": "remote",
        "input": false,
        "output": false,
        "tool_output": true,
        "remote": {
            "endpoint": format!("http://{address}"),
            "config_id": "safety-default"
        }
    })))
    .await
    .unwrap();

    let result = tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("weather_lookup")
            .args(json!({"city": "Phoenix"}))
            .func(Arc::new(move |_args| {
                Box::pin(async move { Ok(json!({"forecast": "sunny"})) })
            }))
            .build(),
    )
    .await
    .unwrap();

    assert_eq!(result, json!({"forecast": "cloudy"}));

    let captured = recv_captured_request(&request_rx);
    let request_json: Json = serde_json::from_slice(&captured.body).unwrap();
    assert_eq!(
        request_json["guardrails"]["options"]["rails"]["tool_input"],
        json!(false)
    );
    assert_eq!(
        request_json["guardrails"]["options"]["rails"]["tool_output"],
        json!(true)
    );
}

#[tokio::test]
async fn remote_tool_input_invalid_modified_arguments_are_reported() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    setup_isolated_thread();

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (request_tx, _request_rx) = mpsc::channel();
    let response_body = json!({
        "id": "chatcmpl-tool-input-invalid",
        "object": "chat.completion",
        "created": 1,
        "model": "",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "{not-json}"
            },
            "finish_reason": "stop"
        }],
        "guardrails": {
            "config_id": "safety-default",
            "log": {
                "activated_rails": []
            }
        }
    })
    .to_string();
    let http_response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        response_body.len(),
        response_body
    )
    .into_bytes();
    spawn_http_responder(listener, http_response, request_tx);

    initialize_plugins(plugin_config(json!({
        "mode": "remote",
        "input": false,
        "output": false,
        "tool_input": true,
        "remote": {
            "endpoint": format!("http://{address}"),
            "config_id": "safety-default"
        }
    })))
    .await
    .unwrap();

    let error = tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("weather_lookup")
            .args(json!({"city": "Phoenix"}))
            .func(Arc::new(move |_args| {
                Box::pin(async move { Ok(json!({"forecast": "sunny"})) })
            }))
            .build(),
    )
    .await
    .unwrap_err();

    match error {
        crate::error::FlowError::Internal(message) => {
            assert!(message.contains("modified tool arguments content that is not valid JSON"));
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[tokio::test]
async fn remote_tool_output_missing_result_field_is_reported() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    setup_isolated_thread();

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (request_tx, _request_rx) = mpsc::channel();
    let response_body = json!({
        "id": "chatcmpl-tool-output-missing-result",
        "object": "chat.completion",
        "created": 1,
        "model": "",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "{\"tool_name\":\"weather_lookup\",\"arguments\":{\"city\":\"Phoenix\"}}"
            },
            "finish_reason": "stop"
        }],
        "guardrails": {
            "config_id": "safety-default",
            "log": {
                "activated_rails": []
            }
        }
    })
    .to_string();
    let http_response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        response_body.len(),
        response_body
    )
    .into_bytes();
    spawn_http_responder(listener, http_response, request_tx);

    initialize_plugins(plugin_config(json!({
        "mode": "remote",
        "input": false,
        "output": false,
        "tool_output": true,
        "remote": {
            "endpoint": format!("http://{address}"),
            "config_id": "safety-default"
        }
    })))
    .await
    .unwrap();

    let error = tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("weather_lookup")
            .args(json!({"city": "Phoenix"}))
            .func(Arc::new(move |_args| {
                Box::pin(async move { Ok(json!({"forecast": "sunny"})) })
            }))
            .build(),
    )
    .await
    .unwrap_err();

    match error {
        crate::error::FlowError::Internal(message) => {
            assert!(message.contains("without a 'result' field"));
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[tokio::test]
async fn remote_tool_output_does_not_run_when_tool_callback_errors() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    setup_isolated_thread();

    initialize_plugins(plugin_config(json!({
        "mode": "remote",
        "input": false,
        "output": false,
        "tool_output": true,
        "remote": {
            "endpoint": unused_local_endpoint(),
            "config_id": "safety-default"
        }
    })))
    .await
    .unwrap();

    let error = tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("weather_lookup")
            .args(json!({"city": "Phoenix"}))
            .func(Arc::new(move |_args| {
                Box::pin(async move {
                    Err(crate::error::FlowError::Internal(
                        "tool callback failed".to_string(),
                    ))
                })
            }))
            .build(),
    )
    .await
    .unwrap_err();

    match error {
        crate::error::FlowError::Internal(message) => {
            assert_eq!(message, "tool callback failed");
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[tokio::test]
async fn remote_tool_input_rewrite_with_mismatched_tool_name_is_rejected() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    setup_isolated_thread();

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (request_tx, _request_rx) = mpsc::channel();
    let response_body = json!({
        "id": "chatcmpl-tool-input-mismatch",
        "object": "chat.completion",
        "created": 1,
        "model": "",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "{\"tool_name\":\"different_lookup\",\"arguments\":{\"city\":\"Boston\"}}"
            },
            "finish_reason": "stop"
        }],
        "guardrails": {
            "config_id": "safety-default",
            "log": {
                "activated_rails": []
            }
        }
    })
    .to_string();
    let http_response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        response_body.len(),
        response_body
    )
    .into_bytes();
    spawn_http_responder(listener, http_response, request_tx);

    initialize_plugins(plugin_config(json!({
        "mode": "remote",
        "input": false,
        "output": false,
        "tool_input": true,
        "remote": {
            "endpoint": format!("http://{address}"),
            "config_id": "safety-default"
        }
    })))
    .await
    .unwrap();

    let error = tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("weather_lookup")
            .args(json!({"city": "Phoenix"}))
            .func(Arc::new(move |_args| {
                Box::pin(async move { Ok(json!({"forecast": "sunny"})) })
            }))
            .build(),
    )
    .await
    .unwrap_err();

    match error {
        crate::error::FlowError::Internal(message) => {
            assert!(message.contains("unexpected tool 'different_lookup'"));
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[tokio::test]
async fn remote_tool_input_and_output_run_in_order() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    setup_isolated_thread();

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (request_tx, request_rx) = mpsc::channel();
    let input_response_body = json!({
        "id": "chatcmpl-tool-input-modified",
        "object": "chat.completion",
        "created": 1,
        "model": "",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "{\"tool_name\":\"weather_lookup\",\"arguments\":{\"city\":\"Boston\"}}"
            },
            "finish_reason": "stop"
        }],
        "guardrails": {
            "config_id": "safety-default",
            "log": {
                "activated_rails": []
            }
        }
    })
    .to_string();
    let output_response_body = json!({
        "id": "chatcmpl-tool-output-modified",
        "object": "chat.completion",
        "created": 1,
        "model": "",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "{\"tool_name\":\"weather_lookup\",\"arguments\":{\"city\":\"Boston\"},\"result\":{\"forecast\":\"cloudy\"}}"
            },
            "finish_reason": "stop"
        }],
        "guardrails": {
            "config_id": "safety-default",
            "log": {
                "activated_rails": []
            }
        }
    })
    .to_string();
    let input_response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        input_response_body.len(),
        input_response_body
    )
    .into_bytes();
    let output_response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        output_response_body.len(),
        output_response_body
    )
    .into_bytes();
    spawn_http_responder_sequence(listener, vec![input_response, output_response], request_tx);

    initialize_plugins(plugin_config(json!({
        "mode": "remote",
        "input": false,
        "output": false,
        "tool_input": true,
        "tool_output": true,
        "remote": {
            "endpoint": format!("http://{address}"),
            "config_id": "safety-default"
        }
    })))
    .await
    .unwrap();

    let seen_args = Arc::new(Mutex::new(None::<Json>));
    let seen_args_for_call = Arc::clone(&seen_args);
    let result = tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("weather_lookup")
            .args(json!({"city": "Phoenix"}))
            .func(Arc::new(move |args| {
                *seen_args_for_call.lock().unwrap() = Some(args.clone());
                Box::pin(async move { Ok(json!({"forecast": "sunny"})) })
            }))
            .build(),
    )
    .await
    .unwrap();

    assert_eq!(*seen_args.lock().unwrap(), Some(json!({"city": "Boston"})));
    assert_eq!(result, json!({"forecast": "cloudy"}));

    let first_request = recv_captured_request(&request_rx);
    let first_request_json: Json = serde_json::from_slice(&first_request.body).unwrap();
    assert_eq!(first_request_json["messages"][0]["role"], json!("user"));
    assert_eq!(
        first_request_json["guardrails"]["options"]["rails"]["tool_input"],
        json!(true)
    );
    assert_eq!(
        first_request_json["guardrails"]["options"]["rails"]["tool_output"],
        json!(false)
    );

    let second_request = recv_captured_request(&request_rx);
    let second_request_json: Json = serde_json::from_slice(&second_request.body).unwrap();
    assert_eq!(second_request_json["messages"][0]["role"], json!("user"));
    assert_eq!(
        second_request_json["messages"][1]["role"],
        json!("assistant")
    );
    assert_eq!(
        second_request_json["guardrails"]["options"]["rails"]["tool_input"],
        json!(false)
    );
    assert_eq!(
        second_request_json["guardrails"]["options"]["rails"]["tool_output"],
        json!(true)
    );
}

#[tokio::test]
async fn remote_tool_checks_forward_context_state_and_thread_id() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    setup_isolated_thread();

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (request_tx, request_rx) = mpsc::channel();
    let response_body = json!({
        "id": "chatcmpl-tool-input-context",
        "object": "chat.completion",
        "created": 1,
        "model": "",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "{\"tool_name\":\"weather_lookup\",\"arguments\":{\"city\":\"Phoenix\"}}"
            },
            "finish_reason": "stop"
        }],
        "guardrails": {
            "config_id": "safety-default",
            "log": {
                "activated_rails": []
            }
        }
    })
    .to_string();
    let http_response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        response_body.len(),
        response_body
    )
    .into_bytes();
    spawn_http_responder(listener, http_response, request_tx);

    initialize_plugins(plugin_config(json!({
        "mode": "remote",
        "input": false,
        "output": false,
        "tool_input": true,
        "remote": {
            "endpoint": format!("http://{address}"),
            "config_id": "safety-default"
        },
        "request_defaults": {
            "context": {"tenant": "smoke"},
            "thread_id": "1234567890abcdef",
            "state": {"events": []}
        }
    })))
    .await
    .unwrap();

    let result = tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("weather_lookup")
            .args(json!({"city": "Phoenix"}))
            .func(Arc::new(move |_args| {
                Box::pin(async move { Ok(json!({"forecast": "sunny"})) })
            }))
            .build(),
    )
    .await
    .unwrap();

    assert_eq!(result, json!({"forecast": "sunny"}));

    let captured = recv_captured_request(&request_rx);
    let request_json: Json = serde_json::from_slice(&captured.body).unwrap();
    assert_eq!(
        request_json["guardrails"]["context"],
        json!({"tenant": "smoke"})
    );
    assert_eq!(
        request_json["guardrails"]["thread_id"],
        json!("1234567890abcdef")
    );
    assert_eq!(request_json["guardrails"]["state"], json!({"events": []}));
}

#[tokio::test]
async fn remote_tool_only_configuration_does_not_intercept_llm_calls() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();
    setup_isolated_thread();

    initialize_plugins(plugin_config(json!({
        "mode": "remote",
        "input": false,
        "output": false,
        "tool_input": true,
        "remote": {
            "endpoint": unused_local_endpoint(),
            "config_id": "safety-default"
        }
    })))
    .await
    .unwrap();

    let expected = json!({"response": "original"});
    let func: LlmExecutionNextFn = Arc::new(move |_req| {
        let expected = expected.clone();
        Box::pin(async move { Ok(expected) })
    });

    let response = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("openai")
            .request(make_chat_request(false))
            .func(func)
            .attributes(LlmAttributes::empty())
            .response_codec(Arc::new(OpenAIChatCodec) as Arc<dyn LlmResponseCodec>)
            .build(),
    )
    .await
    .unwrap();

    assert_eq!(response, json!({"response": "original"}));
}
