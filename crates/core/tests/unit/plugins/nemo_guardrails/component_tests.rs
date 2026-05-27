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
fn explicit_registration_helpers_are_idempotent_and_reversible() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();

    assert!(register_nemo_guardrails_component().is_ok());
    assert!(register_nemo_guardrails_component().is_ok());
    assert!(deregister_nemo_guardrails_component());
    assert!(!deregister_nemo_guardrails_component());
    register_nemo_guardrails_component().unwrap();
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

    let unsupported_version_and_mode = validate_plugin_config(&plugin_config(json!({
        "version": 2,
        "mode": "hybrid",
        "codec": "openai_chat",
        "remote": {"endpoint": "http://localhost:8000", "config_id": "default"}
    })));
    assert!(unsupported_version_and_mode.has_errors());
    assert!(
        unsupported_version_and_mode
            .diagnostics
            .iter()
            .any(
                |diag| diag.code == "nemo_guardrails.unsupported_config_version"
                    && diag.field.as_deref() == Some("version")
            )
    );
    assert!(
        unsupported_version_and_mode
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("mode")
                && diag.message.contains("mode must be 'remote' or 'local'"))
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

    let local_rejects_remote_section = validate_plugin_config(&plugin_config(json!({
        "mode": "local",
        "config_yaml": "rails:\n  input: []\n",
        "codec": "openai_chat",
        "remote": {
            "endpoint": "http://localhost:8000",
            "config_id": "default"
        }
    })));
    assert!(local_rejects_remote_section.has_errors());
    assert!(
        local_rejects_remote_section
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("remote")
                && diag.message.contains("cannot be used when mode is 'local'"))
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

    let unsupported_remote_tool_input = validate_plugin_config(&plugin_config(json!({
        "mode": "remote",
        "codec": "openai_chat",
        "tool_input": true,
        "remote": {
            "endpoint": "http://localhost:8000",
            "config_id": "default"
        }
    })));
    assert!(unsupported_remote_tool_input.has_errors());
    assert!(
        unsupported_remote_tool_input
            .diagnostics
            .iter()
            .any(|diag| {
                diag.field.as_deref() == Some("tool_input")
                    && diag
                        .message
                        .contains("does not currently support managed tool_input")
            })
    );

    let supported_remote_tool_output = validate_plugin_config(&plugin_config(json!({
        "mode": "remote",
        "codec": "openai_chat",
        "tool_output": true,
        "remote": {
            "endpoint": "http://localhost:8000",
            "config_id": "default"
        }
    })));
    assert!(!supported_remote_tool_output.has_errors());

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
        "config_path": "",
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
            .any(|diag| diag.field.as_deref() == Some("config_path"))
    );
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
            "thread_id": "   ",
            "state": {"foo": "bar"},
            "llm_params": [],
            "log": "verbose",
            "output_vars": ["answer", "", 7],
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
            .contains("request_defaults.thread_id must not be empty")
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
            .any(|diag| diag.field.as_deref() == Some("request_defaults.output_vars[1]"))
    );
    assert!(
        invalid_request_defaults
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("request_defaults.output_vars[2]"))
    );
    assert!(
        invalid_request_defaults
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("request_defaults.rails.retrieval[0]"))
    );

    let invalid_request_output_vars_shape = validate_plugin_config(&plugin_config(json!({
        "mode": "remote",
        "codec": "openai_chat",
        "remote": {
            "endpoint": "http://localhost:8000",
            "config_id": "default"
        },
        "request_defaults": {
            "thread_id": "short",
            "output_vars": 7
        }
    })));
    assert!(invalid_request_output_vars_shape.has_errors());
    assert!(
        invalid_request_output_vars_shape
            .diagnostics
            .iter()
            .any(
                |diag| diag.field.as_deref() == Some("request_defaults.thread_id")
                    && diag
                        .message
                        .contains("request_defaults.thread_id must be at least 16 characters long")
            )
    );
    assert!(
        invalid_request_output_vars_shape
            .diagnostics
            .iter()
            .any(|diag| diag.field.as_deref() == Some("request_defaults.output_vars"))
    );

    let valid_bool_output_vars = validate_plugin_config(&plugin_config(json!({
        "mode": "remote",
        "codec": "openai_chat",
        "remote": {
            "endpoint": "http://localhost:8000",
            "config_id": "default"
        },
        "request_defaults": {
            "output_vars": true
        }
    })));
    assert!(!valid_bool_output_vars.has_errors());
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

#[test]
fn enabled_unknown_mode_initialization_fails_fast_when_policy_ignores_validation() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    reset_runtime();

    let error = futures::executor::block_on(initialize_plugins(plugin_config(json!({
        "policy": {"unsupported_value": "ignore"},
        "mode": "hybrid",
        "codec": "openai_chat",
        "remote": {
            "endpoint": "http://localhost:8000",
            "config_id": "default"
        }
    }))))
    .unwrap_err();

    match error {
        crate::plugin::PluginError::InvalidConfig(message) => {
            assert!(message.contains("unsupported NeMo Guardrails mode 'hybrid'"));
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[path = "remote_tests.rs"]
mod remote_tests;
