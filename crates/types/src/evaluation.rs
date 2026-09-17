// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider-neutral decision and evaluation data types.
//!
//! Evaluation operations inspect one state value and answer a declared set of
//! typed questions. They are intentionally separate from chat messages and
//! generated text so providers such as TypeSafe System One can be represented
//! without inventing conversational semantics.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::Json;

/// One provider-neutral evaluation request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationRequest {
    /// Model identifier requested by the caller.
    pub model: String,
    /// String, object, array, or null describing the state to evaluate.
    pub state: Json,
    /// Named questions evaluated over the shared state.
    pub questions: BTreeMap<String, EvaluationQuestion>,
}

/// A typed question whose answer space is declared before inference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EvaluationQuestion {
    /// A binary judgment represented as the probability that the proposition is true.
    Boolean {
        /// Question or rubric supplied to the evaluator.
        instructions: Json,
        /// Optional descriptions of the true and false outcomes.
        #[serde(skip_serializing_if = "Option::is_none")]
        criteria: Option<BooleanCriteria>,
    },
    /// Selection from a caller-declared set of options.
    Choice {
        /// Question or rubric supplied to the evaluator.
        instructions: Json,
        /// Option names and optional descriptions.
        criteria: BTreeMap<String, Json>,
    },
    /// Rating against an ordered caller-declared rubric.
    Score {
        /// Question or rubric supplied to the evaluator.
        instructions: Json,
        /// Ordered descriptions from the lowest to highest score.
        criteria: Vec<Json>,
    },
}

/// Optional semantic labels for a binary evaluation question.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BooleanCriteria {
    /// Meaning of a result approaching one.
    #[serde(rename = "true", skip_serializing_if = "Option::is_none")]
    pub true_description: Option<Json>,
    /// Meaning of a result approaching zero.
    #[serde(rename = "false", skip_serializing_if = "Option::is_none")]
    pub false_description: Option<Json>,
}

/// One provider-neutral evaluation response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationResponse {
    /// Model version that performed the evaluation.
    pub model: String,
    /// Answers keyed by the question IDs from the request.
    pub answers: BTreeMap<String, EvaluationAnswer>,
    /// Provider token usage, when reported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<EvaluationUsage>,
}

/// A typed answer produced by an evaluation model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EvaluationAnswer {
    /// Probability that a binary proposition is true.
    Boolean {
        /// Probability in the inclusive range zero to one.
        probability: f64,
    },
    /// Selected option plus the complete option distribution.
    Choice {
        /// Highest-probability option.
        choice: String,
        /// Probability for every declared option.
        probabilities: BTreeMap<String, f64>,
        /// Provider-calculated confidence, when available.
        #[serde(skip_serializing_if = "Option::is_none")]
        confidence: Option<f64>,
    },
    /// Probability-weighted position on an ordered rubric.
    Score {
        /// Weighted score; values may fall between rubric indices.
        score: f64,
        /// Rubric index to description mapping.
        legend: BTreeMap<String, Json>,
        /// Probability for every rubric index.
        probabilities: BTreeMap<String, f64>,
        /// Provider-calculated confidence, when available.
        #[serde(skip_serializing_if = "Option::is_none")]
        confidence: Option<f64>,
    },
}

/// Token accounting returned by an evaluation provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvaluationUsage {
    /// Provider billing units charged for this request, when reported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub billing_units: Option<u64>,
    /// Tokens consumed by the input state, questions, and rubrics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    /// Tokens or equivalent units consumed by answer production.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
}
