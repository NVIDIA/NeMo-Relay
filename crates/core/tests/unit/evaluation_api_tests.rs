// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::await_holding_lock)] // Serializes access to process-wide runtime state.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use serde_json::json;

use super::{EvaluationExecuteParams, EvaluationExecutionNextFn, evaluation_execute};
use crate::api::event::{Event, ScopeCategory};
use crate::api::runtime::{NemoRelayContextState, global_context};
use crate::api::scope::ScopeType;
use crate::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use crate::evaluation::{
    EvaluationAnswer, EvaluationQuestion, EvaluationRequest, EvaluationResponse, EvaluationUsage,
};

fn reset_global() {
    crate::shared_runtime::reset_runtime_owner_for_tests();
    *global_context().write().unwrap() = NemoRelayContextState::new();
}

#[tokio::test]
async fn managed_evaluation_emits_evaluator_input_and_output() {
    let _guard = crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    reset_global();
    let captured = Arc::new(Mutex::new(Vec::<Event>::new()));
    let subscriber_events = captured.clone();
    register_subscriber(
        "evaluation-api-observer",
        Arc::new(move |event| subscriber_events.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let request = EvaluationRequest {
        model: "jev-latest".into(),
        state: json!({"candidate": "42"}),
        questions: BTreeMap::from([(
            "correct".into(),
            EvaluationQuestion::Boolean {
                instructions: json!("Is this correct?"),
                criteria: None,
            },
        )]),
    };
    let callback: EvaluationExecutionNextFn = Arc::new(|request| {
        Box::pin(async move {
            Ok(EvaluationResponse {
                model: request.model,
                answers: BTreeMap::from([(
                    "correct".into(),
                    EvaluationAnswer::Boolean { probability: 0.9 },
                )]),
                usage: Some(EvaluationUsage {
                    billing_units: None,
                    input_tokens: Some(7),
                    output_tokens: Some(0),
                }),
            })
        })
    });
    let response = evaluation_execute(
        EvaluationExecuteParams::builder()
            .name("typesafe.system_one")
            .request(request)
            .func(callback)
            .build(),
    )
    .await
    .unwrap();
    assert_eq!(response.model, "jev-latest");

    flush_subscribers().unwrap();
    assert!(deregister_subscriber("evaluation-api-observer").unwrap());
    let events = captured.lock().unwrap();
    let start = events
        .iter()
        .find(|event| event.scope_category() == Some(ScopeCategory::Start))
        .unwrap();
    let end = events
        .iter()
        .find(|event| event.scope_category() == Some(ScopeCategory::End))
        .unwrap();
    assert_eq!(start.scope_type(), Some(ScopeType::Evaluator));
    assert_eq!(start.input().unwrap()["model"], "jev-latest");
    assert_eq!(
        end.output().unwrap()["answers"]["correct"]["probability"],
        0.9
    );
    assert_eq!(end.metadata().unwrap()["otel.status_code"], "OK");
}
