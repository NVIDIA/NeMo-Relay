// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for stable worker protocol helpers, structural tool results, and enum values.

use nemo_relay_worker_proto::v1::{
    ConditionalMiddlewareGuardrailRegistration, ConditionalMiddlewareInvocation, EmitMarkRequest,
    GetRuntimeDiagnosticsRequest, GetRuntimeDiagnosticsResponse, HandshakeRequest, HealthRequest,
    InvokeRequest, JsonEnvelope, JsonValue, RegisterConditionalMiddlewareGuardrailRequest,
    RegistrationSurface, RuntimeDiagnostic, ScopeType,
    ToolExecutionResult as ProtoToolExecutionResult, invoke_request,
};
use nemo_relay_worker_proto::{
    WORKER_PROTOCOL_GRPC_V1, decode_json_envelope, decode_json_value, json_envelope, json_value,
};
use prost::Message;
use serde_json::json;

#[test]
fn worker_protocol_identifier_is_stable() {
    assert_eq!(WORKER_PROTOCOL_GRPC_V1, "grpc-v1");
}

#[test]
fn registration_surface_values_are_stable() {
    assert_eq!(RegistrationSurface::Subscriber as i32, 1);
    assert_eq!(RegistrationSurface::EventMetadataInjector as i32, 2);
    assert_eq!(RegistrationSurface::ToolSanitizeRequestGuardrail as i32, 10);
    assert_eq!(
        RegistrationSurface::ToolSanitizeResponseGuardrail as i32,
        11
    );
    assert_eq!(
        RegistrationSurface::ToolConditionalExecutionGuardrail as i32,
        12
    );
    assert_eq!(RegistrationSurface::ToolRequestIntercept as i32, 13);
    assert_eq!(RegistrationSurface::ToolExecutionIntercept as i32, 14);
    assert_eq!(RegistrationSurface::LlmSanitizeRequestGuardrail as i32, 20);
    assert_eq!(RegistrationSurface::LlmSanitizeResponseGuardrail as i32, 21);
    assert_eq!(
        RegistrationSurface::LlmConditionalExecutionGuardrail as i32,
        22
    );
    assert_eq!(RegistrationSurface::LlmRequestIntercept as i32, 23);
    assert_eq!(RegistrationSurface::LlmExecutionIntercept as i32, 24);
    assert_eq!(RegistrationSurface::LlmStreamExecutionIntercept as i32, 25);
    assert_eq!(RegistrationSurface::MarkSanitizeGuardrail as i32, 30);
    assert_eq!(RegistrationSurface::ScopeSanitizeStartGuardrail as i32, 31);
    assert_eq!(RegistrationSurface::ScopeSanitizeEndGuardrail as i32, 32);
    assert_eq!(
        RegistrationSurface::ConditionalMiddlewareGuardrail as i32,
        40
    );
}

#[test]
fn scope_type_values_are_stable() {
    assert_eq!(ScopeType::Agent as i32, 1);
    assert_eq!(ScopeType::Function as i32, 2);
    assert_eq!(ScopeType::Tool as i32, 3);
    assert_eq!(ScopeType::Llm as i32, 4);
    assert_eq!(ScopeType::Retriever as i32, 5);
    assert_eq!(ScopeType::Embedder as i32, 6);
    assert_eq!(ScopeType::Reranker as i32, 7);
    assert_eq!(ScopeType::Guardrail as i32, 8);
    assert_eq!(ScopeType::Evaluator as i32, 9);
    assert_eq!(ScopeType::Custom as i32, 10);
    assert_eq!(ScopeType::Unknown as i32, 11);
}

#[test]
fn request_field_numbers_are_stable() {
    let handshake = HandshakeRequest {
        activation_id: "act".into(),
        plugin_id: "plugin".into(),
        relay_version: "0.8.0".into(),
        worker_protocol: WORKER_PROTOCOL_GRPC_V1.into(),
        auth_token: "token".into(),
        host_endpoint: "unix:///tmp/host.sock".into(),
    };
    let encoded = handshake.encode_to_vec();
    assert_eq!(
        encoded,
        b"\x0a\x03act\x12\x06plugin\x1a\x050.8.0\x22\x07grpc-v1\x2a\x05token\x32\x15unix:///tmp/host.sock"
            .to_vec()
    );
    assert_eq!(
        HandshakeRequest::decode(encoded.as_slice()).expect("decode handshake"),
        handshake
    );

    let health = HealthRequest {
        activation_id: "act".into(),
        auth_token: "token".into(),
    };
    let encoded = health.encode_to_vec();
    assert_eq!(encoded, b"\x0a\x03act\x12\x05token".to_vec());
    assert_eq!(
        HealthRequest::decode(encoded.as_slice()).expect("decode health"),
        health
    );

    let invoke = InvokeRequest {
        activation_id: "act".into(),
        invocation_id: "invoke".into(),
        registration_name: "tool".into(),
        surface: RegistrationSurface::ToolRequestIntercept as i32,
        continuation_id: "next".into(),
        scope: None,
        auth_token: "token".into(),
        payload: None,
    };
    let encoded = invoke.encode_to_vec();
    assert_eq!(
        encoded,
        b"\x0a\x03act\x12\x06invoke\x1a\x04tool\x20\x0d\x2a\x04next\x3a\x05token".to_vec()
    );
    assert_eq!(
        InvokeRequest::decode(encoded.as_slice()).expect("decode invoke"),
        invoke
    );

    let activation_gate = ConditionalMiddlewareGuardrailRegistration {
        callback: true,
        ..Default::default()
    };
    assert_eq!(activation_gate.encode_to_vec(), b"\x28\x01".to_vec());

    let runtime_gate = RegisterConditionalMiddlewareGuardrailRequest {
        callback: true,
        ..Default::default()
    };
    assert_eq!(runtime_gate.encode_to_vec(), b"\x38\x01".to_vec());

    let conditional_invoke = InvokeRequest {
        payload: Some(invoke_request::Payload::ConditionalMiddleware(
            ConditionalMiddlewareInvocation {
                kinds: vec![RegistrationSurface::ConditionalMiddlewareGuardrail as i32],
                registration_name: "target".into(),
            },
        )),
        ..Default::default()
    };
    assert_eq!(
        conditional_invoke.encode_to_vec(),
        b"\x6a\x0b\x0a\x01\x28\x12\x06target".to_vec()
    );
}

#[test]
fn runtime_diagnostics_messages_are_stable() {
    let request = GetRuntimeDiagnosticsRequest {
        activation_id: "act".into(),
        auth_token: "token".into(),
    };
    assert_eq!(
        request.encode_to_vec(),
        b"\x0a\x03act\x12\x05token".to_vec()
    );

    let response = GetRuntimeDiagnosticsResponse {
        entries: vec![RuntimeDiagnostic {
            code: "otel.metric_mark_invalid".into(),
            message: "unsupported metric schema version".into(),
            count: 3,
        }],
    };
    assert_eq!(
        response.encode_to_vec(),
        b"\x0a\x3f\x0a\x18otel.metric_mark_invalid\x12\x21unsupported metric schema version\x18\x03"
            .to_vec()
    );
    assert_eq!(response.entries[0].count, 3);
    assert_eq!(response.entries[0].code, "otel.metric_mark_invalid");
}

#[test]
fn json_envelope_round_trips_payload() {
    let payload = json!({"answer": 42});
    let envelope = json_envelope("nemo.relay.Json@1", &payload).unwrap();

    assert_eq!(envelope.schema, "nemo.relay.Json@1");
    assert_eq!(
        decode_json_envelope::<serde_json::Value>(&envelope).unwrap(),
        payload
    );
}

#[test]
fn invalid_json_envelope_reports_decode_error() {
    let envelope = JsonEnvelope {
        schema: "nemo.relay.Json@1".into(),
        json: b"{".to_vec(),
    };

    assert!(decode_json_envelope::<serde_json::Value>(&envelope).is_err());
}

#[test]
fn tool_execution_result_has_structural_wire_fields() {
    let value = ProtoToolExecutionResult {
        result: Some(JsonValue {
            json: b"1".to_vec(),
        }),
        annotation: Some(JsonValue {
            json: b"2".to_vec(),
        }),
    };

    assert_eq!(
        value.encode_to_vec(),
        vec![0x0a, 0x03, 0x0a, 0x01, b'1', 0x12, 0x03, 0x0a, 0x01, b'2']
    );
}

#[test]
fn json_value_round_trips_lossless_json() {
    let value = json!({"large_integer": 9_007_199_254_740_993_u64});
    let encoded = json_value(&value).unwrap();
    assert_eq!(
        decode_json_value::<serde_json::Value>(&encoded).unwrap(),
        value
    );
}

#[test]
fn tool_execution_result_tolerates_unknown_protobuf_fields() {
    let mut bytes = ProtoToolExecutionResult {
        result: Some(JsonValue {
            json: br#"{"ok":true}"#.to_vec(),
        }),
        annotation: None,
    }
    .encode_to_vec();
    // Unknown field 31, varint wire type, value 7.
    bytes.extend_from_slice(&[0xf8, 0x01, 0x07]);

    let decoded_proto = ProtoToolExecutionResult::decode(bytes.as_slice()).unwrap();
    assert_eq!(
        decode_json_value::<serde_json::Value>(decoded_proto.result.as_ref().unwrap()).unwrap(),
        json!({"ok": true})
    );
}

#[test]
fn emit_mark_additive_fields_preserve_legacy_wire_compatibility() {
    let legacy = EmitMarkRequest::decode(b"\x22\x04mark".as_slice()).expect("decode legacy mark");
    assert_eq!(legacy.name, "mark");
    assert!(legacy.data_schema.is_none());
    assert!(legacy.severity.is_empty());
    assert!(legacy.category.is_empty());

    let request = EmitMarkRequest {
        name: "mark".into(),
        data_schema: Some(
            json_envelope(
                "nemo.relay.DataSchema@1",
                &json!({"name": "nemo.relay.metric_measurements", "version": "1"}),
            )
            .unwrap(),
        ),
        severity: "warn".into(),
        category: "custom".into(),
        ..EmitMarkRequest::default()
    };
    let encoded = request.encode_to_vec();
    assert_eq!(
        encoded,
        b"\x22\x04mark\x3a\x52\x0a\x17nemo.relay.DataSchema@1\x12\x37{\"name\":\"nemo.relay.metric_measurements\",\"version\":\"1\"}\x42\x04warn\x4a\x06custom"
            .to_vec()
    );
    let round_trip = EmitMarkRequest::decode(encoded.as_slice()).unwrap();
    assert_eq!(round_trip, request);
}
