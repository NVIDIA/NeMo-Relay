// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Current generated-message round trips and service descriptors.
//! Rust/Python interoperability is exercised by the installed-wheel host tests.

use nemo_relay_worker_proto::{FILE_DESCRIPTOR_SET, nvidia_worker_v1 as pb};
use prost::Message;
use prost_types::FileDescriptorSet;
use serde_json::json;

fn round_trip<T: Message + Default + PartialEq + std::fmt::Debug>(value: T) {
    let bytes = value.encode_to_vec();
    assert_eq!(T::decode(bytes.as_slice()).unwrap(), value);
}

#[test]
fn open_json_payloads_preserve_null_and_exact_integers() {
    for value in [
        json!(null),
        json!(9_007_199_254_740_993_u64),
        json!({"vendor": {"n": u64::MAX, "enabled": false}}),
    ] {
        let message = pb::JsonValue {
            json: serde_json::to_vec(&value).unwrap(),
        };
        let decoded = pb::JsonValue::decode(message.encode_to_vec().as_slice()).unwrap();
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&decoded.json).unwrap(),
            value
        );
    }
}

#[test]
fn explicit_defaults_and_empty_collections_keep_presence() {
    round_trip(pb::AnnotatedLlmRequest {
        tools: Some(pb::ToolDefinitionList::default()),
        store: Some(false),
        max_output_tokens: Some(0),
        ..Default::default()
    });
    round_trip(pb::GenerationParams {
        stop: Some(pb::StringList::default()),
        ..Default::default()
    });
    round_trip(pb::DiagnosticQuery {
        after_sequence: Some(0),
        max_events: Some(0),
        ..Default::default()
    });
    round_trip(pb::MessageContent {
        content: Some(pb::message_content::Content::Text(String::new())),
    });
    round_trip(pb::MessageContent {
        content: Some(pb::message_content::Content::Parts(
            pb::ContentPartList::default(),
        )),
    });
}

#[test]
fn outcomes_preserve_open_variants_and_error_provenance() {
    round_trip(pb::FinishReason {
        reason: Some(pb::finish_reason::Reason::Unknown("vendor_reason".into())),
    });
    round_trip(pb::GuardrailDecision {
        decision: Some(pb::guardrail_decision::Decision::Allow(())),
    });
    round_trip(pb::ToolExecutionResult {
        result: Some(pb::JsonValue {
            json: b"null".to_vec(),
        }),
        annotation: None,
    });
    round_trip(pb::ToolExecutionInterceptOutcome {
        value: Some(pb::ToolExecutionResult {
            result: Some(pb::JsonValue {
                json: b"null".to_vec(),
            }),
            annotation: None,
        }),
        pending_marks: vec![pb::PendingMarkSpec {
            name: "mark".into(),
            category: Some("vendor.custom".into()),
            data: Some(pb::JsonValue {
                json: b"9007199254740993".to_vec(),
            }),
            ..Default::default()
        }],
    });
    round_trip(pb::InvokeResponse {
        result: Some(pb::invoke_response::Result::Error(pb::WorkerError {
            code: "upstream.error".into(),
            message: "unavailable".into(),
            downstream_continuation_id: Some("next".into()),
            http_status: Some(503),
            ..Default::default()
        })),
    });
}

#[test]
fn diagnostic_values_preserve_numeric_kinds_and_slot_positions() {
    round_trip(pb::DiagnosticCount {
        value: u64::MAX,
        saturated: false,
    });
    round_trip(pb::ResourceMemoryNumber {
        kind: Some(pb::resource_memory_number::Kind::Integer(u64::MAX)),
    });
    round_trip(pb::ResourceMemoryNumber {
        kind: Some(pb::resource_memory_number::Kind::NonFinite(())),
    });
    round_trip(pb::ReadDiagnosticPageResponse {
        result: Some(pb::read_diagnostic_page_response::Result::ReadReason(
            pb::DiagnosticReadReason::Busy as i32,
        )),
    });
    round_trip(pb::ResourceMemoryMetrics {
        limit_events: (0..8)
            .map(|_| pb::MemoryLimitEventSlot {
                state: Some(pb::memory_limit_event_slot::State::Absent(())),
            })
            .collect(),
        ..Default::default()
    });
}

#[test]
fn compiled_descriptors_expose_only_the_current_typed_services() {
    let descriptors = FileDescriptorSet::decode(FILE_DESCRIPTOR_SET).unwrap();
    assert!(
        descriptors
            .file
            .iter()
            .all(|file| file.package() != "nemo.relay.worker.v1")
    );
    let services: Vec<_> = descriptors
        .file
        .iter()
        .filter(|file| file.package() == "nvidia.nemo.relay.worker.v1")
        .flat_map(|file| &file.service)
        .collect();
    assert_eq!(services.len(), 3);
    assert_eq!(
        services
            .iter()
            .map(|service| service.method.len())
            .sum::<usize>(),
        31
    );
    for service in services {
        assert!(service.name().ends_with("Service"));
        for method in &service.method {
            assert_eq!(
                method.input_type(),
                format!(".nvidia.nemo.relay.worker.v1.{}Request", method.name())
            );
            assert_eq!(
                method.output_type(),
                format!(".nvidia.nemo.relay.worker.v1.{}Response", method.name())
            );
        }
    }
    assert_eq!(pb::WORKER_PROTOCOL, "nvidia.nemo.relay.worker.v1");
    assert_eq!(pb::CONTRACT_VERSION, "1.0.0");
}
