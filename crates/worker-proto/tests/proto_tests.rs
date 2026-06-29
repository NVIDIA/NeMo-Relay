// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for stable worker protocol helpers and enum values.

use nemo_relay_worker_proto::v1::{
    HandshakeRequest, HealthRequest, InvokeRequest, JsonEnvelope, RegistrationSurface, ScopeType,
};
use nemo_relay_worker_proto::{WORKER_PROTOCOL_GRPC_V1, decode_json_envelope, json_envelope};
use prost::Message;
use serde_json::json;

#[test]
fn worker_protocol_identifier_is_stable() {
    assert_eq!(WORKER_PROTOCOL_GRPC_V1, "grpc-v1");
}

#[test]
fn registration_surface_values_are_stable() {
    assert_eq!(RegistrationSurface::Subscriber as i32, 1);
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
        relay_version: "0.5.0".into(),
        worker_protocol: WORKER_PROTOCOL_GRPC_V1.into(),
        auth_token: "token".into(),
        host_endpoint: "unix:///tmp/host.sock".into(),
    };
    let encoded = handshake.encode_to_vec();
    assert_eq!(
        encoded,
        b"\x0a\x03act\x12\x06plugin\x1a\x050.5.0\x22\x07grpc-v1\x2a\x05token\x32\x15unix:///tmp/host.sock"
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
