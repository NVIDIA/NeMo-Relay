// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Built-in codec for TypeSafe AI's System One evaluation API.
//!
//! System One is an evaluation surface, not a chat or text-generation API.
//! The codec therefore leaves messages, tools, text output, and finish reasons
//! empty and carries provider-neutral evaluation data in the annotated
//! request/response `custom` envelopes.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::api::llm::LlmRequest;
use crate::api::runtime::{BuiltinLlmCodec, LlmCodecIdentity};
use crate::error::{FlowError, Result};
use crate::json::Json;
use nemo_relay_types::evaluation::{
    BooleanCriteria, EvaluationAnswer, EvaluationQuestion, EvaluationRequest, EvaluationResponse,
    EvaluationUsage,
};

use super::request::{AnnotatedLlmRequest, ApiSpecificRequest};
use super::resolve::{ProviderSurface, ProviderSurfaceDescriptor};
use super::response::{
    AnnotatedLlmResponse, ApiSpecificResponse, Usage, estimate_cost_for_provider,
};
use super::traits::{LlmCodec, LlmResponseCodec};

const API_NAME: &str = "typesafe.system_one";
const PROVIDER: &str = "typesafe";
const OPERATION: &str = "system_one";
const MODELED_REQUEST_KEYS: &[&str] = &["model", "state", "questions"];
const MODELED_RESPONSE_KEYS: &[&str] = &["model", "answers", "usage"];

/// Built-in codec for `POST /v1/systemone`.
pub struct TypeSafeSystemOneCodec;

pub(crate) const PROVIDER_SURFACE: ProviderSurfaceDescriptor = ProviderSurfaceDescriptor {
    surface: ProviderSurface::TypeSafeSystemOne,
    detect_request: |obj, _hint| {
        obj.get("state").is_some_and(is_entry_value)
            && obj.get("questions").is_some_and(Value::is_object)
    },
    detect_response: |obj| {
        obj.get("answers").is_some_and(Value::is_object)
            && obj.get("model").is_some_and(Value::is_string)
    },
    decode_request: |request| TypeSafeSystemOneCodec.decode(request),
    decode_response: |raw| TypeSafeSystemOneCodec.decode_response(raw),
    codec_name: "typesafe_system_one",
    request_codec: || std::sync::Arc::new(TypeSafeSystemOneCodec),
    response_codec: || std::sync::Arc::new(TypeSafeSystemOneCodec),
    streaming_codec: || Box::new(UnsupportedSystemOneStreamingCodec),
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct EvaluationEnvelope<T> {
    provider: String,
    operation: String,
    value: T,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
enum WireQuestion {
    #[serde(rename = "noul")]
    Noul {
        #[serde(default)]
        instructions: Json,
        #[serde(skip_serializing_if = "Option::is_none")]
        criteria: Option<BooleanCriteria>,
    },
    #[serde(rename = "choice")]
    Choice {
        #[serde(default)]
        instructions: Json,
        criteria: BTreeMap<String, Json>,
    },
    #[serde(rename = "score")]
    Score {
        #[serde(default)]
        instructions: Json,
        criteria: Vec<Json>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
enum WireAnswer {
    #[serde(rename = "noul")]
    Noul { noul: f64 },
    #[serde(rename = "choice")]
    Choice {
        choice: String,
        probabilities: BTreeMap<String, f64>,
        confidence: f64,
    },
    #[serde(rename = "score")]
    Score {
        score: f64,
        legend: BTreeMap<String, Json>,
        probabilities: BTreeMap<String, f64>,
        confidence: f64,
    },
}

#[derive(Debug, Deserialize)]
struct WireResponse {
    model: String,
    answers: BTreeMap<String, WireAnswer>,
    usage: WireUsage,
    #[serde(flatten)]
    extra: Map<String, Json>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
struct WireUsage {
    #[serde(default)]
    billing_units: Option<u64>,
    #[serde(default)]
    input_tokens: Option<u64>,
    #[serde(default)]
    output_tokens: Option<u64>,
}

impl From<WireQuestion> for EvaluationQuestion {
    fn from(value: WireQuestion) -> Self {
        match value {
            WireQuestion::Noul {
                instructions,
                criteria,
            } => Self::Boolean {
                instructions,
                criteria,
            },
            WireQuestion::Choice {
                instructions,
                criteria,
            } => Self::Choice {
                instructions,
                criteria,
            },
            WireQuestion::Score {
                instructions,
                criteria,
            } => Self::Score {
                instructions,
                criteria,
            },
        }
    }
}

impl From<&EvaluationQuestion> for WireQuestion {
    fn from(value: &EvaluationQuestion) -> Self {
        match value {
            EvaluationQuestion::Boolean {
                instructions,
                criteria,
            } => Self::Noul {
                instructions: instructions.clone(),
                criteria: criteria.clone(),
            },
            EvaluationQuestion::Choice {
                instructions,
                criteria,
            } => Self::Choice {
                instructions: instructions.clone(),
                criteria: criteria.clone(),
            },
            EvaluationQuestion::Score {
                instructions,
                criteria,
            } => Self::Score {
                instructions: instructions.clone(),
                criteria: criteria.clone(),
            },
        }
    }
}

impl From<WireAnswer> for EvaluationAnswer {
    fn from(value: WireAnswer) -> Self {
        match value {
            WireAnswer::Noul { noul } => Self::Boolean { probability: noul },
            WireAnswer::Choice {
                choice,
                probabilities,
                confidence,
            } => Self::Choice {
                choice,
                probabilities,
                confidence: Some(confidence),
            },
            WireAnswer::Score {
                score,
                legend,
                probabilities,
                confidence,
            } => Self::Score {
                score,
                legend,
                probabilities,
                confidence: Some(confidence),
            },
        }
    }
}

fn decode_evaluation_request(obj: &Map<String, Json>) -> Result<EvaluationRequest> {
    if obj.contains_key("stream") {
        return Err(FlowError::InvalidArgument(
            "TypeSafe System One does not support streaming; remove the stream field".into(),
        ));
    }
    let model = required_non_empty_string(obj, "model", "TypeSafe System One request")?;
    let state = obj
        .get("state")
        .filter(|value| is_entry_value(value))
        .cloned()
        .ok_or_else(|| {
            FlowError::InvalidArgument(
                "TypeSafe System One request state must be a string, object, array, or null".into(),
            )
        })?;
    let questions_value = obj
        .get("questions")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            FlowError::InvalidArgument(
                "TypeSafe System One request questions must be an object".into(),
            )
        })?;
    if questions_value.is_empty() {
        return Err(FlowError::InvalidArgument(
            "TypeSafe System One request questions must not be empty".into(),
        ));
    }
    let mut questions = BTreeMap::new();
    for (id, value) in questions_value {
        if id.trim().is_empty() {
            return Err(FlowError::InvalidArgument(
                "TypeSafe System One question IDs must not be empty".into(),
            ));
        }
        let wire: WireQuestion = serde_json::from_value(value.clone()).map_err(|error| {
            FlowError::InvalidArgument(format!(
                "TypeSafe System One question '{id}' is invalid: {error}"
            ))
        })?;
        let question: EvaluationQuestion = wire.into();
        validate_question(id, &question)?;
        questions.insert(id.clone(), question);
    }
    Ok(EvaluationRequest {
        model,
        state,
        questions,
    })
}

fn validate_question(id: &str, question: &EvaluationQuestion) -> Result<()> {
    let instructions = match question {
        EvaluationQuestion::Boolean { instructions, .. }
        | EvaluationQuestion::Choice { instructions, .. }
        | EvaluationQuestion::Score { instructions, .. } => instructions,
    };
    if !is_entry_value(instructions) {
        return Err(FlowError::InvalidArgument(format!(
            "TypeSafe System One question '{id}' instructions must be a string, object, array, or null"
        )));
    }
    match question {
        EvaluationQuestion::Boolean {
            criteria: Some(criteria),
            ..
        } if criteria
            .true_description
            .iter()
            .chain(criteria.false_description.iter())
            .any(|value| !is_entry_value(value)) =>
        {
            Err(FlowError::InvalidArgument(format!(
                "TypeSafe System One boolean question '{id}' criteria must contain JSON entry values"
            )))
        }
        EvaluationQuestion::Choice { criteria, .. }
            if criteria.values().any(|value| !is_entry_value(value)) =>
        {
            Err(FlowError::InvalidArgument(format!(
                "TypeSafe System One choice question '{id}' criteria must contain JSON entry values"
            )))
        }
        EvaluationQuestion::Score { criteria, .. } if criteria.is_empty() => {
            Err(FlowError::InvalidArgument(format!(
                "TypeSafe System One score question '{id}' needs at least one criterion"
            )))
        }
        EvaluationQuestion::Score { criteria, .. }
            if criteria.iter().any(|value| !is_entry_value(value)) =>
        {
            Err(FlowError::InvalidArgument(format!(
                "TypeSafe System One score question '{id}' criteria must contain JSON entry values"
            )))
        }
        _ => Ok(()),
    }
}

fn is_entry_value(value: &Json) -> bool {
    value.is_string() || value.is_object() || value.is_array() || value.is_null()
}

fn required_non_empty_string(obj: &Map<String, Json>, key: &str, surface: &str) -> Result<String> {
    obj.get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| {
            FlowError::InvalidArgument(format!("{surface} {key} must be a non-empty string"))
        })
}

fn request_envelope(request: &EvaluationRequest) -> Result<Json> {
    serde_json::to_value(EvaluationEnvelope {
        provider: PROVIDER.to_string(),
        operation: OPERATION.to_string(),
        value: request.clone(),
    })
    .map_err(|error| FlowError::Internal(format!("evaluation request encode: {error}")))
}

/// Extract the provider-neutral evaluation value from a decoded System One request.
pub fn evaluation_request(annotated: &AnnotatedLlmRequest) -> Result<EvaluationRequest> {
    let Some(ApiSpecificRequest::Custom { api_name, data }) = annotated.api_specific.as_ref()
    else {
        return Err(FlowError::InvalidArgument(
            "TypeSafe System One annotations require api_specific custom evaluation data".into(),
        ));
    };
    if api_name != API_NAME {
        return Err(FlowError::InvalidArgument(format!(
            "TypeSafe System One api_specific provider mismatch: expected {API_NAME}, got {api_name}"
        )));
    }
    let envelope: EvaluationEnvelope<EvaluationRequest> = serde_json::from_value(data.clone())
        .map_err(|error| {
            FlowError::InvalidArgument(format!(
                "TypeSafe System One evaluation annotation is invalid: {error}"
            ))
        })?;
    if envelope.provider != PROVIDER || envelope.operation != OPERATION {
        return Err(FlowError::InvalidArgument(
            "TypeSafe System One evaluation annotation has the wrong provider or operation".into(),
        ));
    }
    Ok(envelope.value)
}

/// Extract the provider-neutral evaluation value from a decoded System One response.
pub fn evaluation_response(annotated: &AnnotatedLlmResponse) -> Result<EvaluationResponse> {
    let Some(ApiSpecificResponse::Custom { api_name, data }) = annotated.api_specific.as_ref()
    else {
        return Err(FlowError::InvalidArgument(
            "TypeSafe System One annotations require api_specific custom evaluation data".into(),
        ));
    };
    if api_name != API_NAME {
        return Err(FlowError::InvalidArgument(format!(
            "TypeSafe System One api_specific provider mismatch: expected {API_NAME}, got {api_name}"
        )));
    }
    let envelope: EvaluationEnvelope<EvaluationResponse> = serde_json::from_value(data.clone())
        .map_err(|error| {
            FlowError::InvalidArgument(format!(
                "TypeSafe System One evaluation annotation is invalid: {error}"
            ))
        })?;
    if envelope.provider != PROVIDER || envelope.operation != OPERATION {
        return Err(FlowError::InvalidArgument(
            "TypeSafe System One evaluation annotation has the wrong provider or operation".into(),
        ));
    }
    Ok(envelope.value)
}

fn encode_question(question: &EvaluationQuestion) -> Result<Json> {
    serde_json::to_value(WireQuestion::from(question))
        .map_err(|error| FlowError::Internal(format!("System One question encode: {error}")))
}

fn patch_questions(
    obj: &mut Map<String, Json>,
    edited: &BTreeMap<String, EvaluationQuestion>,
    baseline: &BTreeMap<String, EvaluationQuestion>,
) -> Result<()> {
    let original = obj
        .get("questions")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let mut patched = Map::new();
    for (id, question) in edited {
        validate_question(id, question)?;
        if baseline.get(id) == Some(question)
            && let Some(original_question) = original.get(id)
        {
            patched.insert(id.clone(), original_question.clone());
            continue;
        }
        let mut encoded = encode_question(question)?;
        if let (Some(encoded), Some(original)) = (
            encoded.as_object_mut(),
            original.get(id).and_then(Value::as_object),
        ) {
            // Boolean criteria have two modeled keys, so other keys are provider extensions.
            // Choice criteria are themselves the option map: removed options must stay removed.
            if matches!(question, EvaluationQuestion::Boolean { .. })
                && let (Some(encoded_criteria), Some(original_criteria)) = (
                    encoded.get_mut("criteria").and_then(Value::as_object_mut),
                    original.get("criteria").and_then(Value::as_object),
                )
            {
                for (key, value) in original_criteria {
                    if !matches!(key.as_str(), "true" | "false") {
                        encoded_criteria
                            .entry(key.clone())
                            .or_insert_with(|| value.clone());
                    }
                }
            }
            for (key, value) in original {
                if !matches!(key.as_str(), "type" | "instructions" | "criteria") {
                    encoded.entry(key.clone()).or_insert_with(|| value.clone());
                }
            }
        }
        patched.insert(id.clone(), encoded);
    }
    obj.insert("questions".into(), Json::Object(patched));
    Ok(())
}

fn patch_extra_fields(
    obj: &mut Map<String, Json>,
    baseline: &Map<String, Json>,
    edited: &Map<String, Json>,
) {
    for key in baseline.keys() {
        if !edited.contains_key(key) {
            obj.remove(key);
        }
    }
    for (key, value) in edited {
        if baseline.get(key) != Some(value) {
            obj.insert(key.clone(), value.clone());
        }
    }
}

fn validate_probability(value: f64, field: &str) -> Result<()> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(FlowError::InvalidArgument(format!(
            "TypeSafe System One response {field} must be between 0 and 1"
        )))
    }
}

fn validate_answer(id: &str, answer: &EvaluationAnswer) -> Result<()> {
    match answer {
        EvaluationAnswer::Boolean { probability } => {
            validate_probability(*probability, &format!("answer '{id}' probability"))
        }
        EvaluationAnswer::Choice {
            choice,
            probabilities,
            confidence,
        } => {
            if !probabilities.contains_key(choice) {
                return Err(FlowError::InvalidArgument(format!(
                    "TypeSafe System One choice answer '{id}' selected an option missing from probabilities"
                )));
            }
            for (option, probability) in probabilities {
                validate_probability(
                    *probability,
                    &format!("answer '{id}' probability for '{option}'"),
                )?;
            }
            if let Some(confidence) = confidence {
                validate_probability(*confidence, &format!("answer '{id}' confidence"))?;
            }
            Ok(())
        }
        EvaluationAnswer::Score {
            probabilities,
            confidence,
            ..
        } => {
            for (level, probability) in probabilities {
                validate_probability(
                    *probability,
                    &format!("answer '{id}' probability for level '{level}'"),
                )?;
            }
            if let Some(confidence) = confidence {
                validate_probability(*confidence, &format!("answer '{id}' confidence"))?;
            }
            Ok(())
        }
    }
}

impl LlmCodec for TypeSafeSystemOneCodec {
    fn codec_identity(&self) -> LlmCodecIdentity {
        LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::TypeSafeSystemOne)
    }

    fn decode(&self, request: &LlmRequest) -> Result<AnnotatedLlmRequest> {
        let obj = request.content.as_object().ok_or_else(|| {
            FlowError::InvalidArgument("TypeSafe System One request must be an object".into())
        })?;
        let evaluation = decode_evaluation_request(obj)?;
        let extra = obj
            .iter()
            .filter(|(key, _)| !MODELED_REQUEST_KEYS.contains(&key.as_str()))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        Ok(AnnotatedLlmRequest {
            model: Some(evaluation.model.clone()),
            api_specific: Some(ApiSpecificRequest::Custom {
                api_name: API_NAME.into(),
                data: request_envelope(&evaluation)?,
            }),
            extra,
            ..AnnotatedLlmRequest::default()
        })
    }

    fn encode(&self, annotated: &AnnotatedLlmRequest, original: &LlmRequest) -> Result<LlmRequest> {
        let baseline = self.decode(original)?;
        validate_non_evaluation_fields(annotated, &baseline)?;
        let mut edited = evaluation_request(annotated)?;
        let baseline_evaluation = evaluation_request(&baseline)?;

        let annotation_model_changed = edited.model != baseline_evaluation.model;
        let top_level_model_changed = annotated.model != baseline.model;
        if annotation_model_changed
            && top_level_model_changed
            && annotated.model.as_ref() != Some(&edited.model)
        {
            return Err(FlowError::InvalidArgument(
                "TypeSafe System One model was changed inconsistently in model and api_specific"
                    .into(),
            ));
        }
        if top_level_model_changed {
            edited.model = annotated.model.clone().ok_or_else(|| {
                FlowError::InvalidArgument("TypeSafe System One model cannot be removed".into())
            })?;
        }
        if edited.model.trim().is_empty() {
            return Err(FlowError::InvalidArgument(
                "TypeSafe System One model must not be empty".into(),
            ));
        }

        let mut content = original.content.clone();
        let obj = content.as_object_mut().ok_or_else(|| {
            FlowError::InvalidArgument("TypeSafe System One request must be an object".into())
        })?;
        if edited.model != baseline_evaluation.model {
            obj.insert("model".into(), Json::String(edited.model.clone()));
        }
        if edited.state != baseline_evaluation.state {
            if !is_entry_value(&edited.state) {
                return Err(FlowError::InvalidArgument(
                    "TypeSafe System One state must be a string, object, array, or null".into(),
                ));
            }
            obj.insert("state".into(), edited.state.clone());
        }
        if edited.questions != baseline_evaluation.questions {
            if edited.questions.is_empty() {
                return Err(FlowError::InvalidArgument(
                    "TypeSafe System One questions must not be empty".into(),
                ));
            }
            patch_questions(obj, &edited.questions, &baseline_evaluation.questions)?;
        }
        patch_extra_fields(obj, &baseline.extra, &annotated.extra);
        Ok(LlmRequest {
            headers: original.headers.clone(),
            content,
        })
    }
}

fn validate_non_evaluation_fields(
    annotated: &AnnotatedLlmRequest,
    baseline: &AnnotatedLlmRequest,
) -> Result<()> {
    if let Some(key) = annotated
        .extra
        .keys()
        .find(|key| MODELED_REQUEST_KEYS.contains(&key.as_str()))
    {
        return Err(FlowError::InvalidArgument(format!(
            "TypeSafe System One modeled field '{key}' cannot be edited through extra"
        )));
    }
    macro_rules! reject_if_changed {
        ($field:ident) => {
            if annotated.$field != baseline.$field {
                return Err(FlowError::InvalidArgument(format!(
                    "TypeSafe System One does not support annotated field '{}'",
                    stringify!($field)
                )));
            }
        };
    }
    reject_if_changed!(messages);
    reject_if_changed!(instructions);
    reject_if_changed!(params);
    reject_if_changed!(tools);
    reject_if_changed!(tool_choice);
    reject_if_changed!(store);
    reject_if_changed!(previous_response_id);
    reject_if_changed!(truncation);
    reject_if_changed!(reasoning);
    reject_if_changed!(include);
    reject_if_changed!(user);
    reject_if_changed!(metadata);
    reject_if_changed!(service_tier);
    reject_if_changed!(parallel_tool_calls);
    reject_if_changed!(max_output_tokens);
    reject_if_changed!(max_tool_calls);
    reject_if_changed!(top_logprobs);
    reject_if_changed!(stream);
    Ok(())
}

impl LlmResponseCodec for TypeSafeSystemOneCodec {
    fn codec_identity(&self) -> LlmCodecIdentity {
        LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::TypeSafeSystemOne)
    }

    fn decode_response(&self, response: &Json) -> Result<AnnotatedLlmResponse> {
        let raw: WireResponse = serde_json::from_value(response.clone()).map_err(|error| {
            FlowError::InvalidArgument(format!("TypeSafe System One response is invalid: {error}"))
        })?;
        if raw.model.trim().is_empty() {
            return Err(FlowError::InvalidArgument(
                "TypeSafe System One response model must not be empty".into(),
            ));
        }
        let answers = raw
            .answers
            .into_iter()
            .map(|(id, answer)| {
                let answer: EvaluationAnswer = answer.into();
                validate_answer(&id, &answer)?;
                Ok((id, answer))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        let evaluation = EvaluationResponse {
            model: raw.model.clone(),
            answers,
            usage: Some(EvaluationUsage {
                billing_units: raw.usage.billing_units,
                input_tokens: raw.usage.input_tokens,
                output_tokens: raw.usage.output_tokens,
            }),
        };
        let mut usage = Usage {
            prompt_tokens: raw.usage.input_tokens,
            completion_tokens: raw.usage.output_tokens,
            total_tokens: raw
                .usage
                .input_tokens
                .zip(raw.usage.output_tokens)
                .and_then(|(input, output)| input.checked_add(output)),
            ..Usage::default()
        };
        usage.cost = estimate_cost_for_provider(Some(PROVIDER), &raw.model, &usage);
        let data = serde_json::to_value(EvaluationEnvelope {
            provider: PROVIDER.to_string(),
            operation: OPERATION.to_string(),
            value: evaluation,
        })
        .map_err(|error| FlowError::Internal(format!("evaluation response encode: {error}")))?;
        Ok(AnnotatedLlmResponse {
            model: Some(raw.model),
            usage: Some(usage),
            api_specific: Some(ApiSpecificResponse::Custom {
                api_name: API_NAME.into(),
                data,
            }),
            extra: raw
                .extra
                .into_iter()
                .filter(|(key, _)| !MODELED_RESPONSE_KEYS.contains(&key.as_str()))
                .collect(),
            ..AnnotatedLlmResponse::default()
        })
    }
}

/// Defensive streaming placeholder. Requests declaring streaming are rejected
/// by `decode` before a provider continuation can be opened.
struct UnsupportedSystemOneStreamingCodec;

impl super::streaming::StreamingCodec for UnsupportedSystemOneStreamingCodec {
    fn collector(&self) -> crate::api::runtime::LlmCollectorFn {
        Box::new(|_| {
            Err(FlowError::InvalidArgument(
                "TypeSafe System One does not support streaming".into(),
            ))
        })
    }

    fn finalizer(&self) -> crate::api::runtime::LlmFinalizerFn {
        Box::new(|| Json::Null)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/codec/typesafe_system_one_tests.rs"]
mod tests;
