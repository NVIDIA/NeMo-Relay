// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::codec::response::{
    PricingCatalog, PricingResolver, reset_active_pricing_resolver, set_active_pricing_resolver,
};
use serde_json::json;

fn request(content: Json) -> LlmRequest {
    LlmRequest {
        headers: Map::new(),
        content,
    }
}

fn fixture() -> Json {
    json!({
        "model": "jev-latest",
        "state": {"candidate": "The answer is 42."},
        "questions": {
            "correct": {
                "type": "noul",
                "instructions": "Is the candidate correct?",
                "criteria": {
                    "true": "correct",
                    "false": "incorrect",
                    "provider_label": "preserve-nested"
                },
                "provider_extension": "preserve-me"
            },
            "quality": {
                "type": "choice",
                "instructions": ["Choose a quality band"],
                "criteria": {"high": {"meaning": "good"}, "low": null}
            },
            "score": {
                "type": "score",
                "instructions": {"rubric": "Score the answer"},
                "criteria": ["wrong", {"meaning": "partly right"}, null]
            }
        },
        "request_id": "req-123"
    })
}

#[test]
fn decodes_into_provider_neutral_evaluation_without_chat_semantics() {
    let annotated = TypeSafeSystemOneCodec.decode(&request(fixture())).unwrap();

    assert_eq!(annotated.model.as_deref(), Some("jev-latest"));
    assert!(annotated.messages.is_empty());
    assert!(annotated.tools.is_none());
    assert_eq!(annotated.extra.get("request_id"), Some(&json!("req-123")));

    let evaluation = evaluation_request(&annotated).unwrap();
    assert_eq!(evaluation.questions.len(), 3);
    assert!(matches!(
        evaluation.questions.get("correct"),
        Some(EvaluationQuestion::Boolean { .. })
    ));
    assert!(matches!(
        evaluation.questions.get("quality"),
        Some(EvaluationQuestion::Choice { .. })
    ));
    assert!(matches!(
        evaluation.questions.get("score"),
        Some(EvaluationQuestion::Score { .. })
    ));
}

#[test]
fn unchanged_round_trip_is_lossless() {
    let original = request(fixture());
    let annotated = TypeSafeSystemOneCodec.decode(&original).unwrap();
    let encoded = TypeSafeSystemOneCodec
        .encode(&annotated, &original)
        .unwrap();
    assert_eq!(encoded, original);
}

#[test]
fn edits_model_and_evaluation_while_preserving_unknown_fields() {
    let original = request(fixture());
    let mut annotated = TypeSafeSystemOneCodec.decode(&original).unwrap();
    annotated.model = Some("jev-1.13.0".into());
    let ApiSpecificRequest::Custom { data, .. } = annotated.api_specific.as_mut().unwrap() else {
        panic!("expected custom evaluation data");
    };
    let mut envelope: EvaluationEnvelope<EvaluationRequest> =
        serde_json::from_value(data.clone()).unwrap();
    envelope.value.model = "jev-1.13.0".into();
    envelope.value.state = json!(["new", "state"]);
    if let EvaluationQuestion::Boolean { instructions, .. } = envelope
        .value
        .questions
        .get_mut("correct")
        .expect("boolean question")
    {
        *instructions = json!("Updated instruction");
    }
    *data = serde_json::to_value(envelope).unwrap();

    let encoded = TypeSafeSystemOneCodec
        .encode(&annotated, &original)
        .unwrap();
    assert_eq!(encoded.content["model"], "jev-1.13.0");
    assert_eq!(encoded.content["state"], json!(["new", "state"]));
    assert_eq!(encoded.content["request_id"], "req-123");
    assert_eq!(
        encoded.content["questions"]["correct"]["provider_extension"],
        "preserve-me"
    );
    assert_eq!(
        encoded.content["questions"]["correct"]["criteria"]["provider_label"],
        "preserve-nested"
    );
}

#[test]
fn choice_edits_can_remove_options_without_restoring_them_as_extensions() {
    let original = request(fixture());
    let mut annotated = TypeSafeSystemOneCodec.decode(&original).unwrap();
    let ApiSpecificRequest::Custom { data, .. } = annotated.api_specific.as_mut().unwrap() else {
        panic!("expected custom evaluation data");
    };
    let mut envelope: EvaluationEnvelope<EvaluationRequest> =
        serde_json::from_value(data.clone()).unwrap();
    let EvaluationQuestion::Choice { criteria, .. } = envelope
        .value
        .questions
        .get_mut("quality")
        .expect("choice question")
    else {
        panic!("expected choice question");
    };
    criteria.remove("low");
    *data = serde_json::to_value(envelope).unwrap();

    let encoded = TypeSafeSystemOneCodec
        .encode(&annotated, &original)
        .unwrap();
    assert_eq!(
        encoded.content["questions"]["quality"]["criteria"],
        json!({"high": {"meaning": "good"}})
    );
}

#[test]
fn rejects_modeled_fields_in_the_extra_map() {
    let original = request(fixture());
    let mut annotated = TypeSafeSystemOneCodec.decode(&original).unwrap();
    annotated.extra.insert("model".into(), json!("shadowed"));
    let error = TypeSafeSystemOneCodec
        .encode(&annotated, &original)
        .unwrap_err();
    assert!(error.to_string().contains("cannot be edited through extra"));
}

#[test]
fn explicitly_rejects_every_streaming_declaration() {
    for stream in [json!(true), json!(false), json!(null)] {
        let mut body = fixture();
        body["stream"] = stream;
        let error = TypeSafeSystemOneCodec.decode(&request(body)).unwrap_err();
        assert!(error.to_string().contains("does not support streaming"));
    }
}

#[test]
fn rejects_malformed_requests() {
    for body in [
        json!({"model": "jev", "state": 7, "questions": {"q": {"type": "noul", "instructions": "x"}}}),
        json!({"model": "jev", "state": "x", "questions": {}}),
        json!({"model": "jev", "state": "x", "questions": {"q": {"type": "score", "instructions": "x", "criteria": []}}}),
    ] {
        assert!(TypeSafeSystemOneCodec.decode(&request(body)).is_err());
    }
}

#[test]
fn accepts_python_sdk_single_score_and_choice_shapes() {
    for question in [
        json!({"type": "score", "criteria": ["only"]}),
        json!({"type": "choice", "criteria": {}}),
    ] {
        let body = json!({
            "model": "jev-latest",
            "state": "candidate",
            "questions": {"q": question}
        });
        assert!(TypeSafeSystemOneCodec.decode(&request(body)).is_ok());
    }
}

#[test]
fn accepts_null_state_and_omitted_optional_instructions() {
    let body = json!({
        "model": "jev-latest",
        "state": null,
        "questions": {"q": {"type": "noul"}}
    });
    let annotated = TypeSafeSystemOneCodec
        .decode(&request(body.clone()))
        .unwrap();
    assert_eq!(evaluation_request(&annotated).unwrap().state, Json::Null);
    assert_eq!(
        super::super::resolve::detect_request_surface(&body),
        Some(ProviderSurface::TypeSafeSystemOne)
    );
}

fn response_fixture(model: &str) -> Json {
    json!({
        "model": model,
        "answers": {
            "correct": {"type": "noul", "noul": 0.93},
            "quality": {
                "type": "choice",
                "choice": "high",
                "probabilities": {"high": 0.8, "low": 0.2},
                "confidence": 0.8
            },
            "score": {
                "type": "score",
                "score": 1.8,
                "legend": {"0": "wrong", "1": "partial", "2": "right"},
                "probabilities": {"0": 0.05, "1": 0.1, "2": 0.85},
                "confidence": 0.85
            }
        },
        "usage": {"billing_units": 42, "input_tokens": 1_000_000, "output_tokens": 0},
        "trace_id": "typesafe-trace"
    })
}

#[test]
fn decodes_structured_answers_and_usage_without_text_output() {
    let response = TypeSafeSystemOneCodec
        .decode_response(&response_fixture("jev-1.13.0"))
        .unwrap();

    assert_eq!(response.model.as_deref(), Some("jev-1.13.0"));
    assert!(response.message.is_none());
    assert!(response.finish_reason.is_none());
    let usage = response.usage.as_ref().unwrap();
    assert_eq!(usage.prompt_tokens, Some(1_000_000));
    assert_eq!(usage.completion_tokens, Some(0));
    assert_eq!(usage.total_tokens, Some(1_000_000));
    assert_eq!(
        response.extra.get("trace_id"),
        Some(&json!("typesafe-trace"))
    );

    let evaluation = evaluation_response(&response).unwrap();
    assert_eq!(evaluation.answers.len(), 3);
    assert_eq!(evaluation.usage.unwrap().billing_units, Some(42));
}

#[test]
fn rejects_invalid_answer_probabilities() {
    let mut response = response_fixture("jev-1.13.0");
    response["answers"]["correct"]["noul"] = json!(1.01);
    assert!(TypeSafeSystemOneCodec.decode_response(&response).is_err());
}

#[test]
fn uses_provider_scoped_model_alias_pricing() {
    struct ResetPricing;
    impl Drop for ResetPricing {
        fn drop(&mut self) {
            let _ = reset_active_pricing_resolver();
        }
    }
    let _reset = ResetPricing;
    let catalog = PricingCatalog::from_json_str(
        &json!({
            "version": 1,
            "entries": [{
                "provider": "typesafe",
                "model_id": "jev-1.13.0",
                "aliases": ["jev-latest"],
                "pricing_as_of": "2026-09-17",
                "pricing_source": "https://www.typesafe.ai/",
                "rates": {"input_per_million": 0.042, "output_per_million": 0.0},
                "prompt_cache": {"read_accounting": "included_in_prompt_tokens"}
            }]
        })
        .to_string(),
    )
    .unwrap();
    set_active_pricing_resolver(PricingResolver::from_catalogs(vec![catalog])).unwrap();

    let response = TypeSafeSystemOneCodec
        .decode_response(&response_fixture("jev-latest"))
        .unwrap();
    let cost = response.usage.unwrap().cost.unwrap();
    assert_eq!(cost.total, Some(0.042));
    assert_eq!(cost.pricing_provider.as_deref(), Some("typesafe"));
    assert_eq!(cost.pricing_model.as_deref(), Some("jev-1.13.0"));
}

#[test]
fn exposes_builtin_identity_and_surface_detection() {
    assert_eq!(
        LlmCodec::codec_identity(&TypeSafeSystemOneCodec),
        LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::TypeSafeSystemOne)
    );
    assert_eq!(
        super::super::resolve::detect_request_surface(&fixture()),
        Some(ProviderSurface::TypeSafeSystemOne)
    );
    assert_eq!(
        super::super::resolve::detect_response_surface(&response_fixture("jev-latest")),
        Some(ProviderSurface::TypeSafeSystemOne)
    );
}
