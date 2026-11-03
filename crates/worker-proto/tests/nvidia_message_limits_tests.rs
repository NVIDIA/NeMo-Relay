// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Current worker message limits and structural protobuf compatibility.

use nemo_relay_worker_proto::{
    conversions::decode,
    nvidia_worker_v1::{JsonValue, ToolExecutionResult as ProtoToolExecutionResult},
};
use prost::Message;
use serde_json::json;

#[test]
fn worker_callback_limits_are_independent() {
    let mut limits = nemo_relay_worker_proto::WorkerCallbackMessageLimits::default();
    limits
        .by_name
        .insert("large.guardrail-α".into(), 16 * 1024 * 1024);
    assert_eq!(limits.for_callback("large.guardrail-α"), 16 * 1024 * 1024);
    for name in [
        "LARGE.guardrail-α",
        "other_guardrail",
        "Invoke",
        "ToolNext",
        "",
    ] {
        assert_eq!(limits.for_callback(name), 4 * 1024 * 1024);
    }
    limits.default_bytes = 16 * 1024 * 1024;
    limits
        .by_name
        .insert("small_guardrail".into(), 4 * 1024 * 1024);
    assert_eq!(limits.for_callback("small_guardrail"), 4 * 1024 * 1024);
}

#[test]
fn callback_metadata_preserves_exact_names_and_rejects_mismatches() {
    use nemo_relay_worker_proto::{
        WorkerCallbackMessageLimits, callback_name, callback_request, validate_callback_request,
    };
    let name = "large.guardrail-α";
    let mut limits = WorkerCallbackMessageLimits::default();
    limits.by_name.insert(name.into(), 16 * 1024 * 1024);
    let request = callback_request((), name);
    assert_eq!(callback_name(request.metadata()).as_deref(), Some(name));
    assert!(validate_callback_request(&request, name, &limits).is_ok());
    assert!(validate_callback_request(&request, "other_guardrail", &limits).is_err());
    let unannotated = tonic::Request::new(());
    assert!(validate_callback_request(&unannotated, name, &limits).is_err());
    assert!(
        validate_callback_request(&unannotated, name, &WorkerCallbackMessageLimits::default())
            .is_ok()
    );
}

#[test]
/// Keeps Rust parsing aligned with the finite cross-language byte-limit contract.
fn worker_message_limit_accepts_only_bounded_positive_decimal_bytes() {
    use nemo_relay_worker_proto::parse_worker_max_message_bytes;
    for (value, expected) in [
        ("1", 1),
        ("4194304", 4_194_304),
        ("16777216", 16_777_216),
        ("2147483647", 2_147_483_647),
    ] {
        assert_eq!(parse_worker_max_message_bytes(value), Ok(expected));
    }
    for value in [
        "",
        "0",
        "-1",
        "+1",
        " 10",
        "10 ",
        "1.0",
        "16MiB",
        "2147483648",
        "999999999999999999999999",
        "１２",
    ] {
        assert!(parse_worker_max_message_bytes(value).is_err(), "{value:?}");
    }
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
        decode::<serde_json::Value, _>(decoded_proto.result.unwrap()).unwrap(),
        json!({"ok": true})
    );
}
