// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LLM codec types, traits, and built-in implementations.
//!
//! This module provides the type system and traits for bidirectional
//! request codecs ([`traits::LlmCodec`] / [`request::AnnotatedLlmRequest`]),
//! the decode-only response codec
//! ([`traits::LlmResponseCodec`] / [`response::AnnotatedLlmResponse`]), and
//! the streaming response codec
//! ([`streaming::StreamingCodec`]) used with the managed
//! streaming LLM execution pipeline.
//!
//! [`resolve`] is the detect-then-decode entry point for selecting a built-in
//! provider codec from a raw payload when no codec annotation is present.

pub mod anthropic;
pub mod model_pricing;
pub mod openai_chat;
pub mod openai_responses;
pub mod optimization;
pub mod request;
pub mod resolve;
pub mod response;
pub mod streaming;
pub mod traits;

use nemo_relay_types::Json;

use crate::error::Result;

fn encode_changed_items<T, F>(
    edited: &[T],
    baseline: &[T],
    original: Option<&[Json]>,
    mut encode: F,
) -> Result<Vec<Json>>
where
    T: PartialEq,
    F: FnMut(&T) -> Result<Json>,
{
    let structurally_aligned = edited.len() == baseline.len();
    edited
        .iter()
        .enumerate()
        .map(|(index, item)| {
            let original_item = original.and_then(|items| items.get(index));
            if let Some(baseline_item) = baseline.get(index) {
                if baseline_item == item
                    && let Some(original_item) = original_item
                {
                    return Ok(original_item.clone());
                }
                if structurally_aligned && let Some(original_item) = original_item {
                    let baseline_value = encode(baseline_item)?;
                    let edited_value = encode(item)?;
                    return Ok(patch_changed_json(
                        original_item,
                        &baseline_value,
                        &edited_value,
                    ));
                }
            }
            encode(item)
        })
        .collect()
}

fn patch_changed_json(original: &Json, baseline: &Json, edited: &Json) -> Json {
    if baseline == edited {
        return original.clone();
    }

    match (original, baseline, edited) {
        (Json::Object(original), Json::Object(baseline), Json::Object(edited)) => {
            if ["type", "role"]
                .into_iter()
                .any(|key| baseline.get(key) != edited.get(key))
            {
                return Json::Object(edited.clone());
            }
            let mut patched = original.clone();
            for key in baseline.keys().filter(|key| !edited.contains_key(*key)) {
                patched.remove(key);
            }
            for (key, edited_value) in edited {
                if baseline.get(key) == Some(edited_value) {
                    continue;
                }
                let value = match (original.get(key), baseline.get(key)) {
                    (Some(original_value), Some(baseline_value)) => {
                        patch_changed_json(original_value, baseline_value, edited_value)
                    }
                    _ => edited_value.clone(),
                };
                patched.insert(key.clone(), value);
            }
            Json::Object(patched)
        }
        (Json::Array(original), Json::Array(baseline), Json::Array(edited)) => Json::Array(
            edited
                .iter()
                .enumerate()
                .map(
                    |(index, edited_value)| match (original.get(index), baseline.get(index)) {
                        (Some(original_value), Some(baseline_value)) => {
                            patch_changed_json(original_value, baseline_value, edited_value)
                        }
                        _ => edited_value.clone(),
                    },
                )
                .collect(),
        ),
        _ => edited.clone(),
    }
}

#[cfg(test)]
#[path = "../../tests/unit/codec/parity_tests.rs"]
mod parity_tests;
