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

#[test]
#[allow(deprecated)] // Exercise the aggregate fields retained for older worker readers.
fn cache_write_usage_preserves_canonical_entries_and_projects_legacy_counts() {
    use nemo_relay_types::codec::response::Usage;

    for (value, total, one_hour) in [
        (json!({}), None, None),
        (json!({"cache_write_tokens": []}), Some(0), None),
        (json!({"cache_write_tokens": [{"count": 0}]}), Some(0), None),
        (
            json!({"cache_write_tokens": [{"count": 0, "duration": 3600}]}),
            Some(0),
            Some(0),
        ),
        (
            json!({
                "prompt_tokens": 0, "completion_tokens": 2, "total_tokens": 2,
                "cache_read_tokens": 0, "uncached_input_tokens": 0,
                "cost": {"total": 0.25, "currency": "USD", "source": "provider_reported"},
                "cache_write_tokens": [
                    {"count": u64::MAX},
                    {"count": 1, "duration": 3600},
                    {"count": 2, "duration": 0},
                    {"count": 0, "duration": 999},
                    {"count": 4, "duration": 3600},
                    {"count": 3, "duration": u64::MAX}
                ]
            }),
            Some(u64::MAX),
            Some(5),
        ),
        (
            json!({"cache_write_tokens": [
                {"count": u64::MAX, "duration": 3600},
                {"count": 1, "duration": 3600}
            ]}),
            Some(u64::MAX),
            Some(u64::MAX),
        ),
    ] {
        let usage: Usage = serde_json::from_value(value).unwrap();
        let expected = serde_json::to_value(&usage).unwrap();
        let wire: pb::Usage = encode(usage).unwrap();
        assert_eq!(wire.cache_write_tokens, total);
        assert_eq!(wire.cache_write_1h_tokens, one_hour);
        let wire = pb::Usage::decode(wire.encode_to_vec().as_slice()).unwrap();
        let returned: Usage = decode(wire).unwrap();
        assert_eq!(serde_json::to_value(returned).unwrap(), expected);
    }
}

#[test]
#[allow(deprecated)] // Conflicting old-reader projections must not override the canonical list.
fn canonical_cache_write_usage_overrides_contradictory_legacy_counts() {
    use nemo_relay_types::codec::response::Usage;

    for entries in [
        json!([]),
        json!([{"count": 0, "duration": 0}]),
        json!([{"count": 7}, {"count": 2, "duration": 3600}]),
    ] {
        let expected = json!({"cache_write_tokens": entries});
        let usage: Usage = serde_json::from_value(expected.clone()).unwrap();
        let mut wire: pb::Usage = encode(usage).unwrap();
        wire.cache_write_tokens = Some(1);
        wire.cache_write_1h_tokens = Some(9);
        let wire = pb::Usage::decode(wire.encode_to_vec().as_slice()).unwrap();
        let returned: Usage = decode(wire).unwrap();
        assert_eq!(serde_json::to_value(returned).unwrap(), expected);
    }
}

#[test]
#[allow(deprecated)] // Decode messages containing only the original scalar fields.
fn legacy_cache_write_usage_uses_authoritative_total_normalization() {
    use nemo_relay_types::codec::response::Usage;

    for (total, one_hour, expected) in [
        (None, None, json!({})),
        (None, Some(0), json!({})),
        (None, Some(7), json!({})),
        (Some(0), None, json!({"cache_write_tokens": [{"count": 0}]})),
        (
            Some(0),
            Some(9),
            json!({"cache_write_tokens": [{"count": 0}]}),
        ),
        (
            Some(10),
            None,
            json!({"cache_write_tokens": [{"count": 10}]}),
        ),
        (
            Some(10),
            Some(0),
            json!({"cache_write_tokens": [{"count": 10}]}),
        ),
        (
            Some(10),
            Some(4),
            json!({"cache_write_tokens": [{"count": 6}, {"count": 4, "duration": 3600}]}),
        ),
        (
            Some(10),
            Some(12),
            json!({"cache_write_tokens": [{"count": 10, "duration": 3600}]}),
        ),
    ] {
        let wire = pb::Usage {
            cache_write_tokens: total,
            cache_write_1h_tokens: one_hour,
            ..Default::default()
        };
        let wire = pb::Usage::decode(wire.encode_to_vec().as_slice()).unwrap();
        let returned: Usage = decode(wire).unwrap();
        assert_eq!(serde_json::to_value(returned).unwrap(), expected);
    }
}

#[test]
fn cache_write_usage_rejects_an_entry_without_a_count() {
    use nemo_relay_types::codec::response::Usage;

    // Usage tag 9 contains list.values[0] with duration = 1 but no count.
    let wire = pb::Usage::decode([0x4a, 4, 0x0a, 2, 0x10, 1].as_slice()).unwrap();
    assert!(decode::<Usage, _>(wire).is_err());
}

#[test]
fn annotated_response_preserves_distinct_nested_cache_write_usage() {
    use nemo_relay_types::codec::response::AnnotatedLlmResponse;

    let value: AnnotatedLlmResponse = serde_json::from_value(json!({
        "usage": {"cache_write_tokens": [{"count": 0, "duration": 0}, {"count": 7}]},
        "optimization_summary": {
            "schema_version": "1.0.0", "calculation_version": "1.0.0", "status": "partial",
            "effective_usage": {"cache_write_tokens": []},
            "baseline_usage": {"cache_write_tokens": [
                {"count": u64::MAX, "duration": 999}, {"count": 2, "duration": 3600}
            ]},
            "tokens_saved": {}, "contributions": []
        }
    }))
    .unwrap();
    roundtrip::<_, pb::AnnotatedLlmResponse>(value);
}

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

/// Every provider-neutral and provider-native variant survives actual wire encoding.
#[test]
fn multimodal_and_tool_variants_preserve_payloads_and_extensions() {
    use nemo_relay_types::codec::request::{ContentPart, Message, ToolChoice, ToolDefinition};
    use nemo_relay_types::codec::response::{FinishReason, Usage};
    let extension = json!({"nested": [null, u64::MAX, "雪\u{0}"]});
    for value in [
        json!({"type":"text", "text":"雪\u{0}", "vendor":extension}),
        json!({"type":"image_url", "image_url":{"url":"https://example.test/image", "detail":"auto"}, "vendor":extension}),
        json!({"type":"image", "image":{"data":"encoded"}, "vendor":extension}),
        json!({"type":"audio", "audio":{"format":"wav"}, "vendor":extension}),
        json!({"type":"file", "file":{"id":"file"}, "vendor":extension}),
        json!({"type":"refusal", "refusal":"declined", "vendor":extension}),
        json!({"type":"tool_use", "id":"call", "name":"tool", "input":extension, "vendor":extension}),
        json!({"type":"tool_result", "tool_use_id":"call", "content":extension, "is_error":true, "vendor":extension}),
        json!({"type":"provider_native", "provider":"custom", "kind":"opaque", "value":extension}),
    ] {
        roundtrip::<ContentPart, pb::ContentPart>(serde_json::from_value(value).unwrap());
    }
    for value in [
        json!({"role":"developer", "content":"instruction", "name":"author"}),
        json!({"role":"tool_call", "id":"item", "call_id":"call", "name":"tool", "arguments":extension, "vendor":extension}),
        json!({"role":"tool_result", "id":"result", "call_id":"call", "output":extension, "vendor":extension}),
        json!({"role":"provider_native", "provider":"custom", "kind":"opaque", "value":extension}),
    ] {
        roundtrip::<Message, pb::LlmMessage>(serde_json::from_value(value).unwrap());
    }
    for value in [
        json!({"type":"function", "function":{"name":"tool", "parameters":extension}, "vendor":extension}),
        json!({"type":"provider_native", "provider":"custom", "kind":"opaque", "value":extension}),
    ] {
        roundtrip::<ToolDefinition, pb::ToolDefinition>(serde_json::from_value(value).unwrap());
    }
    for value in [
        json!("auto"),
        json!("none"),
        json!("required"),
        json!({"type":"function", "function":{"name":"tool"}}),
        json!({"provider":"custom", "kind":"opaque", "value":extension}),
    ] {
        roundtrip::<ToolChoice, pb::ToolChoice>(serde_json::from_value(value).unwrap());
    }
    for reason in [
        FinishReason::Complete,
        FinishReason::Length,
        FinishReason::ToolUse,
        FinishReason::ContentFilter,
        FinishReason::Unknown("vendor\u{0}".into()),
    ] {
        roundtrip::<_, pb::FinishReason>(reason);
    }
    let usage: Usage =
        serde_json::from_value(json!({"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}))
            .unwrap();
    let wire: Box<pb::Usage> = encode(usage).unwrap();
    let decoded: Usage = decode(Box::new(
        pb::Usage::decode(wire.encode_to_vec().as_slice()).unwrap(),
    ))
    .unwrap();
    assert_eq!(decoded.total_tokens, Some(3));
}

/// Collection envelopes must preserve their JSON container type and reject mismatches.
#[test]
fn json_collection_envelopes_preserve_shape_and_reject_wrong_container_types() {
    for value in [
        json!([]),
        json!([null, true, i64::MIN, u64::MAX, "α\u{0000}", {"nested":[1]}]),
    ] {
        let wire: pb::JsonArray = encode(value.clone()).unwrap();
        let wire = pb::JsonArray::decode(wire.encode_to_vec().as_slice()).unwrap();
        assert_eq!(decode::<Json, _>(wire).unwrap(), value);
    }
    for value in [Json::Null, json!({}), json!("array"), json!(0)] {
        assert!(encode::<_, pb::JsonArray>(value).is_err());
    }
    assert!(encode::<_, pb::JsonObject>(json!([])).is_err());
    assert!(
        decode::<Json, _>(pb::JsonArray {
            values: vec![pb::JsonValue {
                json: b"invalid JSON".to_vec()
            }]
        })
        .is_err()
    );
    let fields =
        std::collections::BTreeMap::from([("extension".to_owned(), json!([u64::MAX, "α"]))]);
    let wire: pb::JsonObject = encode(fields.clone()).unwrap();
    assert_eq!(
        decode::<std::collections::BTreeMap<String, Json>, _>(wire).unwrap(),
        fields
    );
}

/// Diagnostic transport must retain opaque execution and phase identifiers unchanged.
#[test]
fn diagnostic_transport_preserves_opaque_execution_and_phase_names() {
    use nemo_relay_types::diagnostics::{PluginExecutionName, PluginPhaseName};
    for name in ["native", "future.extension-α", "opaque\u{0000}value"] {
        let execution: PluginExecutionName = decode(name.to_owned()).unwrap();
        assert_eq!(encode::<_, String>(execution).unwrap(), name);
        let phase: PluginPhaseName = decode(name.to_owned()).unwrap();
        assert_eq!(encode::<_, String>(phase).unwrap(), name);
    }
}

#[test]
fn content_inspection_projection_round_trips_all_typed_payloads_and_locations() {
    use nemo_relay_types::codec::content::{
        ContentField, ContentLocation, ContentRoot, ContentSelection, ContentSnapshot,
    };
    use nemo_relay_types::codec::request::Message;
    let messages = json!([
        {"role":"assistant","content":[
            {"type":"text","text":""},{"type":"refusal","refusal":"no"},
            {"type":"tool_use","id":"c","name":"f","input":{"n":18446744073709551615u64}},
            {"type":"tool_result","tool_use_id":"c","content":"hidden","is_error":false},
            {"type":"image_url","image_url":{"url":"image"}}, {"type":"image","image":{}},
            {"type":"audio","audio":{}},{"type":"file","file":{}},
            {"type":"provider_native","provider":"custom","kind":"extension","value":{"text":"fallback"}}
        ],"tool_calls":[{"id":"c","type":"function","function":{"name":"f","arguments":"{invalid}"}}]},
        {"role":"tool","tool_call_id":"c","content":"hidden"},
        {"role":"function","name":"f","content":null},
        {"role":"function","name":"f","content":"hidden"},
        {"role":"tool_call","id":"i","call_id":"c","name":"f","arguments":"{invalid}"},
        {"role":"tool_result","id":"i","call_id":"c","output":null}
    ]);
    for value in messages.as_array().unwrap() {
        let message: Message = serde_json::from_value(value.clone()).unwrap();
        let original = message.content_view().snapshot();
        let wire: pb::LlmContentSnapshot = encode(original.clone()).unwrap();
        nemo_relay_worker_proto::validation::validate(&wire).unwrap();
        let wire = pb::LlmContentSnapshot::decode(wire.encode_to_vec().as_slice()).unwrap();
        assert_eq!(decode::<ContentSnapshot, _>(wire).unwrap(), original);
    }
    for selection in [
        ContentSelection::LastMessage,
        ContentSelection::Message(-1),
        ContentSelection::Instructions,
        ContentSelection::SystemPrompt,
        ContentSelection::LastUserMessage,
        ContentSelection::Output,
        ContentSelection::All,
    ] {
        let wire: pb::LlmContentSelection = encode(selection).unwrap();
        assert_eq!(decode::<ContentSelection, _>(wire).unwrap(), selection);
    }
    for root in [
        ContentRoot::RequestMessage(7),
        ContentRoot::Message,
        ContentRoot::Response,
        ContentRoot::Content,
    ] {
        for field in [
            ContentField::Content,
            ContentField::Message,
            ContentField::ToolCalls,
            ContentField::ToolCallItem,
            ContentField::Output,
            ContentField::Value,
            ContentField::NativeContent,
        ] {
            let location = ContentLocation {
                root,
                field,
                index: Some(0),
            };
            let wire: pb::LlmContentLocation = encode(location).unwrap();
            nemo_relay_worker_proto::validation::validate(&wire).unwrap();
            assert_eq!(decode::<ContentLocation, _>(wire).unwrap(), location);
        }
    }
}
