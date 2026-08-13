// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for stable worker protocol helpers, structural tool results, and enum values.

use nemo_relay_types::api::event::{CategoryProfile, EventCategory, PendingMarkSpec};
use nemo_relay_types::api::tool::{
    ToolExecutionInterceptOutcome as SharedToolExecutionInterceptOutcome,
    ToolExecutionResult as SharedToolExecutionResult,
};
use nemo_relay_worker_proto::v1::{
    HandshakeRequest, HealthRequest, InvokeRequest, JsonEnvelope, JsonValue, RegistrationSurface,
    ScopeType, ToolExecutionResult as ProtoToolExecutionResult,
};
use nemo_relay_worker_proto::{
    WORKER_PROTOCOL_GRPC_V1, decode_json_envelope, decode_tool_execution_intercept_outcome,
    decode_tool_execution_result, encode_tool_execution_intercept_outcome,
    encode_tool_execution_result, json_envelope,
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
fn tool_execution_contract_round_trips_lossless_json_and_pending_marks() {
    let mut profile = CategoryProfile {
        subtype: Some("checkpoint".into()),
        ..CategoryProfile::default()
    };
    profile.extra.insert("score".into(), json!(0.75));
    let value = SharedToolExecutionInterceptOutcome::annotated(
        json!({"large_integer": 9_007_199_254_740_993_u64}),
        json!({"source": "worker"}),
    )
    .with_pending_mark(PendingMarkSpec {
        name: "worker-checkpoint".into(),
        category: Some(EventCategory::new("vendor.category")),
        category_profile: Some(profile),
        data: Some(json!({"ok": true})),
        metadata: Some(json!({"sequence": 1})),
    });

    let encoded = encode_tool_execution_intercept_outcome(&value).unwrap();
    assert_eq!(
        encoded.pending_marks[0].category.as_deref(),
        Some("vendor.category")
    );
    assert_eq!(
        decode_tool_execution_intercept_outcome(encoded).unwrap(),
        value
    );
}

#[test]
fn tool_execution_result_normalizes_null_annotation_to_absence() {
    let shared = SharedToolExecutionResult {
        result: json!("ok"),
        annotation: Some(serde_json::Value::Null),
    };
    let encoded = encode_tool_execution_result(&shared).unwrap();
    assert!(encoded.annotation.is_none());

    let decoded = decode_tool_execution_result(ProtoToolExecutionResult {
        result: Some(JsonValue {
            json: br#""ok""#.to_vec(),
        }),
        annotation: Some(JsonValue {
            json: b"null".to_vec(),
        }),
    })
    .unwrap();
    assert_eq!(decoded.annotation, None);
}

#[test]
fn tool_execution_result_rejects_missing_result_and_contextualizes_invalid_json() {
    let missing = decode_tool_execution_result(ProtoToolExecutionResult {
        result: None,
        annotation: None,
    })
    .expect_err("result is semantically required");
    assert_eq!(
        missing.to_string(),
        "tool execution result.result is missing"
    );

    let invalid = decode_tool_execution_result(ProtoToolExecutionResult {
        result: Some(JsonValue { json: vec![0xff] }),
        annotation: None,
    })
    .expect_err("invalid UTF-8/JSON must fail");
    assert!(
        invalid
            .to_string()
            .contains("invalid JSON in tool execution result.result")
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
    let decoded = decode_tool_execution_result(decoded_proto).unwrap();
    assert_eq!(decoded.result, json!({"ok": true}));
}

#[test]
fn tool_execution_outcome_rejects_non_object_pending_mark_category_profile() {
    let outcome = nemo_relay_worker_proto::v1::ToolExecutionInterceptOutcome {
        result: Some(JsonValue {
            json: br#"{"ok":true}"#.to_vec(),
        }),
        annotation: None,
        pending_marks: vec![nemo_relay_worker_proto::v1::PendingMarkSpec {
            name: "worker.mark".into(),
            category: None,
            category_profile: Some(JsonValue {
                json: br#"["not","an","object"]"#.to_vec(),
            }),
            data: None,
            metadata: None,
        }],
    };

    let error = decode_tool_execution_intercept_outcome(outcome)
        .expect_err("category_profile must decode as a CategoryProfile object");
    assert!(error.to_string().contains("pending_marks.category_profile"));
}
