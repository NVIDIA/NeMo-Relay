// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lossless domain conversions through actual protobuf encoding.

use nemo_relay_types::{Json, codec::request::AnnotatedLlmRequest};
use nemo_relay_worker_proto::{
    conversions::{WireValue, decode, encode},
    nvidia_worker_v1 as pb,
};
use prost::Message;
use serde::{Serialize, de::DeserializeOwned};
use serde_json::json;

fn roundtrip<D, W>(domain: D)
where
    D: WireValue<W> + Serialize + DeserializeOwned,
    W: Message + Default,
{
    let expected = serde_json::to_value(&domain).unwrap();
    let wire: W = encode(domain).unwrap();
    let wire = W::decode(wire.encode_to_vec().as_slice()).unwrap();
    let returned: D = decode(wire).unwrap();
    assert_eq!(serde_json::to_value(returned).unwrap(), expected);
}

#[test]
fn request_preserves_presence_and_all_message_variants() {
    let value: AnnotatedLlmRequest = serde_json::from_value(json!({
        "messages": [
            {"role":"system","content":""},
            {"role":"user","content":[{"type":"text","text":"question"}]},
            {"role":"assistant","content":null,"tool_calls":[]},
            {"role":"tool","tool_call_id":"call","content":""},
            {"role":"function","name":"legacy","content":null}
        ],
        "tools": [], "store": false, "max_output_tokens": 0,
        "reasoning": null, "vendor": {"integer":18446744073709551615_u64},
        "params": {"stop": []}
    }))
    .unwrap();
    roundtrip::<_, pb::AnnotatedLlmRequest>(value);
}

#[test]
fn json_null_and_large_integers_are_lossless() {
    roundtrip::<Json, pb::JsonValue>(json!(null));
    roundtrip::<Json, pb::JsonValue>(json!({"large":u64::MAX, "null":null}));
}

#[test]
fn diagnostic_domain_fixtures_survive_typed_transport() {
    use nemo_relay_types::diagnostics::{DiagnosticMemoryPage, DiagnosticSnapshot};
    let snapshot: DiagnosticSnapshot = serde_json::from_str(include_str!(
        "../../types/tests/fixtures/diagnostics/diagnostic-snapshot-schema-1.json"
    ))
    .unwrap();
    roundtrip::<_, pb::DiagnosticSnapshot>(snapshot);
    let memory: DiagnosticMemoryPage = serde_json::from_str(include_str!(
        "../../types/tests/fixtures/diagnostics/diagnostic-memory-schema-1.json"
    ))
    .unwrap();
    roundtrip::<_, pb::DiagnosticMemoryPage>(memory);
}

#[test]
fn optimization_fixture_survives_typed_transport() {
    use nemo_relay_types::codec::optimization::LlmOptimizationContribution;
    let value: LlmOptimizationContribution = serde_json::from_str(include_str!(
        "../../types/tests/fixtures/llm_optimization_contribution_v1.json"
    ))
    .unwrap();
    roundtrip::<_, pb::LlmOptimizationContribution>(value);
}

#[test]
fn provider_data_uses_native_json_for_every_codec_flavor() {
    use nemo_relay_types::codec::{request::ApiSpecificRequest, response::ApiSpecificResponse};
    for value in [
        json!({"api": "openai_chat", "seed": i64::MIN, "n": u64::MAX}),
        json!({"api": "custom", "api_name": "runtime.example", "data": {"large": u64::MAX, "nullable": null}}),
        json!({"api": "custom", "api_name": "dynamic.example/v7", "data": [false, null, {"future": "field"}]}),
    ] {
        let domain: ApiSpecificRequest = serde_json::from_value(value).unwrap();
        let native_json = serde_json::to_vec(&domain).unwrap();
        let wire: pb::JsonValue = encode(domain.clone()).unwrap();
        assert_eq!(wire.json, native_json);
        roundtrip::<_, pb::JsonValue>(domain);
    }
    let response: ApiSpecificResponse = serde_json::from_value(json!({
        "api": "custom", "api_name": "dynamic.example/v7", "data": {"output": [null, u64::MAX]}
    }))
    .unwrap();
    roundtrip::<_, pb::JsonValue>(response);
}

#[test]
fn codec_references_are_metadata_rather_than_a_codec_catalogue() {
    use nemo_relay_types::codec::identity::{BuiltinLlmCodec, LlmCodecIdentity};
    for identity in [
        LlmCodecIdentity::None,
        LlmCodecIdentity::Opaque,
        LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OpenAiChat),
        LlmCodecIdentity::Runtime("runtime.example/v7".into()),
    ] {
        roundtrip::<_, pb::LlmCodecReference>(identity);
    }
    let reference = pb::LlmCodecReference {
        kind: "dynamic.example".into(),
        id: Some("future.codec/v7".into()),
    };
    nemo_relay_worker_proto::validation::validate(&reference).unwrap();
    assert_eq!(
        decode::<LlmCodecIdentity, _>(reference).unwrap(),
        LlmCodecIdentity::Runtime("future.codec/v7".into())
    );
}

#[test]
fn absent_and_null_optional_codec_data_follow_native_serde() {
    let request: AnnotatedLlmRequest =
        serde_json::from_value(json!({"messages": [], "api_specific": null})).unwrap();
    let mut wire: pb::AnnotatedLlmRequest = encode(request).unwrap();
    assert!(wire.api_specific.is_none());
    wire.api_specific = Some(pb::JsonValue {
        json: b"null".to_vec(),
    });
    assert!(
        decode::<AnnotatedLlmRequest, _>(wire)
            .unwrap()
            .api_specific
            .is_none()
    );
}

#[test]
fn flattened_dto_extensions_cannot_shadow_modeled_native_fields() {
    macro_rules! check {
        ($wire:ident, $domain:ty, $field:literal) => {{
            let mut wire = pb::$wire::default();
            let mut domain: $domain = decode(wire.clone()).unwrap();
            domain.extra.insert($field.into(), json!(42));
            assert!(
                encode::<_, pb::$wire>(domain)
                    .unwrap_err()
                    .to_string()
                    .contains("shadows")
            );
            wire.extra.insert(
                $field.into(),
                pb::JsonValue {
                    json: b"42".to_vec(),
                },
            );
            assert!(
                decode::<$domain, _>(wire)
                    .unwrap_err()
                    .to_string()
                    .contains("shadows")
            );
        }};
    }
    check!(
        AnnotatedLlmRequest,
        nemo_relay_types::codec::request::AnnotatedLlmRequest,
        "messages"
    );
    check!(
        AnnotatedLlmResponse,
        nemo_relay_types::codec::response::AnnotatedLlmResponse,
        "message"
    );
    check!(
        FunctionDefinition,
        nemo_relay_types::codec::request::FunctionDefinition,
        "parameters"
    );
    check!(
        LlmOptimizationContribution,
        nemo_relay_types::codec::optimization::LlmOptimizationContribution,
        "producer"
    );
    check!(
        CategoryProfile,
        nemo_relay_types::api::event::CategoryProfile,
        "time_to_first_chunk"
    );
}

#[test]
fn flattened_variant_extensions_reject_native_fields_and_discriminators() {
    use nemo_relay_types::codec::request::{ContentPart, Message, ToolDefinition};
    for key in ["refusal", "type"] {
        let native = ContentPart::Refusal {
            refusal: "body".into(),
            extra: [(key.into(), json!("shadow"))].into_iter().collect(),
        };
        assert!(
            encode::<_, pb::ContentPart>(native)
                .unwrap_err()
                .to_string()
                .contains("shadows")
        );
        let wire = pb::ContentPart {
            part: Some(pb::content_part::Part::Refusal(pb::TextContentPart {
                text: "body".into(),
                extra: [(
                    key.into(),
                    pb::JsonValue {
                        json: b"42".to_vec(),
                    },
                )]
                .into(),
            })),
        };
        assert!(
            decode::<ContentPart, _>(wire)
                .unwrap_err()
                .to_string()
                .contains("shadows")
        );
    }
    for key in ["image", "type"] {
        let native = ContentPart::Image {
            image: json!({"source":"body"}),
            extra: [(key.into(), json!("shadow"))].into_iter().collect(),
        };
        assert!(
            encode::<_, pb::ContentPart>(native)
                .unwrap_err()
                .to_string()
                .contains("shadows")
        );
        let wire = pb::ContentPart {
            part: Some(pb::content_part::Part::Image(pb::JsonContentPart {
                value: Some(pb::JsonValue {
                    json: b"null".to_vec(),
                }),
                extra: [(
                    key.into(),
                    pb::JsonValue {
                        json: b"42".to_vec(),
                    },
                )]
                .into(),
            })),
        };
        assert!(
            decode::<ContentPart, _>(wire)
                .unwrap_err()
                .to_string()
                .contains("shadows")
        );
    }
    let wire = pb::LlmMessage {
        role: Some(pb::llm_message::Role::ToolResult(pb::ToolResultItem {
            id: None,
            call_id: "call".into(),
            output: Some(pb::JsonValue {
                json: b"null".to_vec(),
            }),
            extra: [(
                "role".into(),
                pb::JsonValue {
                    json: b"42".to_vec(),
                },
            )]
            .into(),
        })),
    };
    assert!(
        decode::<Message, _>(wire)
            .unwrap_err()
            .to_string()
            .contains("shadows")
    );
    let wire = pb::ToolDefinition {
        definition: Some(pb::tool_definition::Definition::Function(
            pb::FunctionToolDefinition {
                function: Some(pb::FunctionDefinition::default()),
                extra: [(
                    "type".into(),
                    pb::JsonValue {
                        json: b"42".to_vec(),
                    },
                )]
                .into(),
            },
        )),
    };
    assert!(
        decode::<ToolDefinition, _>(wire)
            .unwrap_err()
            .to_string()
            .contains("shadows")
    );
}

#[test]
fn transport_names_and_flattened_extra_are_valid_native_extensions() {
    use nemo_relay_types::{api::event::CategoryProfile, codec::request::ContentPart};
    let mut request = AnnotatedLlmRequest::default();
    request
        .extra
        .insert("extra".into(), json!({"vendor": true}));
    roundtrip::<_, pb::AnnotatedLlmRequest>(request);
    let profile = CategoryProfile {
        time_to_first_chunk: Some(0.5),
        extra: [(
            "time_to_first_chunk_seconds".into(),
            json!("provider extension"),
        )]
        .into(),
        ..Default::default()
    };
    roundtrip::<_, pb::CategoryProfile>(profile);
    roundtrip::<_, pb::ContentPart>(ContentPart::Refusal {
        refusal: "body".into(),
        extra: [("text".into(), json!("provider extension"))]
            .into_iter()
            .collect(),
    });
    roundtrip::<_, pb::ContentPart>(ContentPart::Image {
        image: json!({"source":"body"}),
        extra: [
            ("value".into(), json!(1)),
            ("audio".into(), json!(2)),
            ("refusal".into(), json!(3)),
        ]
        .into_iter()
        .collect(),
    });
}
