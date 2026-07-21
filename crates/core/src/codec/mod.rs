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

use crate::error::{FlowError, Result};

fn optional_bool(
    obj: &serde_json::Map<String, Json>,
    key: &str,
    surface: &str,
) -> Result<Option<bool>> {
    match obj.get(key) {
        Some(Json::Null) | None => Ok(None),
        Some(Json::Bool(value)) => Ok(Some(*value)),
        Some(_) => Err(FlowError::InvalidArgument(format!(
            "{surface} {key} must be a boolean or null"
        ))),
    }
}

fn optional_u64(
    obj: &serde_json::Map<String, Json>,
    key: &str,
    surface: &str,
) -> Result<Option<u64>> {
    match obj.get(key) {
        Some(Json::Null) | None => Ok(None),
        Some(value) if value.as_u64().is_some() => Ok(value.as_u64()),
        Some(_) => Err(FlowError::InvalidArgument(format!(
            "{surface} {key} must be a non-negative integer or null"
        ))),
    }
}

fn optional_i64(
    obj: &serde_json::Map<String, Json>,
    key: &str,
    surface: &str,
) -> Result<Option<i64>> {
    match obj.get(key) {
        Some(Json::Null) | None => Ok(None),
        Some(value) if value.as_i64().is_some() => Ok(value.as_i64()),
        Some(_) => Err(FlowError::InvalidArgument(format!(
            "{surface} {key} must be an integer or null"
        ))),
    }
}

fn optional_f64(
    obj: &serde_json::Map<String, Json>,
    key: &str,
    surface: &str,
) -> Result<Option<f64>> {
    match obj.get(key) {
        Some(Json::Null) | None => Ok(None),
        Some(value) if value.as_f64().is_some() => Ok(value.as_f64()),
        Some(_) => Err(FlowError::InvalidArgument(format!(
            "{surface} {key} must be a number or null"
        ))),
    }
}

fn optional_string(
    obj: &serde_json::Map<String, Json>,
    key: &str,
    surface: &str,
) -> Result<Option<String>> {
    match obj.get(key) {
        Some(Json::Null) | None => Ok(None),
        Some(Json::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(FlowError::InvalidArgument(format!(
            "{surface} {key} must be a string or null"
        ))),
    }
}

fn encode_changed_items<T, F>(
    edited: &[T],
    baseline: &[T],
    original: Option<&[Json]>,
    encode: F,
) -> Result<Vec<Json>>
where
    T: PartialEq,
    F: FnMut(&T) -> Result<Json>,
{
    encode_changed_items_with_patch(
        edited,
        baseline,
        original,
        encode,
        |original, _, _, baseline_value, edited_value| {
            patch_changed_json(original, baseline_value, edited_value)
        },
    )
}

fn encode_changed_items_with_patch<T, F, P>(
    edited: &[T],
    baseline: &[T],
    original: Option<&[Json]>,
    mut encode: F,
    mut patch: P,
) -> Result<Vec<Json>>
where
    T: PartialEq,
    F: FnMut(&T) -> Result<Json>,
    P: FnMut(&Json, &T, &T, &Json, &Json) -> Result<Json>,
{
    let alignment = if original.is_some() {
        align_changed_items(edited, baseline)?
    } else {
        vec![None; edited.len()]
    };
    edited
        .iter()
        .enumerate()
        .map(|(index, item)| {
            if let Some(baseline_index) = alignment[index] {
                let baseline_item = &baseline[baseline_index];
                if let Some(original_item) = original.and_then(|items| items.get(baseline_index)) {
                    if baseline_item == item {
                        return Ok(original_item.clone());
                    }
                    let baseline_value = encode(baseline_item)?;
                    let edited_value = encode(item)?;
                    return patch(
                        original_item,
                        baseline_item,
                        item,
                        &baseline_value,
                        &edited_value,
                    );
                }
            }
            encode(item)
        })
        .collect()
}

fn align_changed_items<T: PartialEq>(edited: &[T], baseline: &[T]) -> Result<Vec<Option<usize>>> {
    let mut alignment = vec![None; edited.len()];
    let mut used = vec![false; baseline.len()];

    for index in 0..edited.len().min(baseline.len()) {
        if edited[index] == baseline[index] {
            alignment[index] = Some(index);
            used[index] = true;
        }
    }

    let mut structurally_changed = edited.len() != baseline.len();
    for (edited_index, item) in edited.iter().enumerate() {
        if alignment[edited_index].is_some() {
            continue;
        }
        if let Some(baseline_index) = baseline
            .iter()
            .enumerate()
            .position(|(baseline_index, candidate)| !used[baseline_index] && candidate == item)
        {
            alignment[edited_index] = Some(baseline_index);
            used[baseline_index] = true;
            structurally_changed |= baseline_index != edited_index;
        }
    }

    let unmatched_edited = alignment
        .iter()
        .enumerate()
        .filter_map(|(index, item)| item.is_none().then_some(index))
        .collect::<Vec<_>>();
    let unmatched_baseline = used
        .iter()
        .enumerate()
        .filter_map(|(index, item)| (!item).then_some(index))
        .collect::<Vec<_>>();

    if unmatched_edited.len() > 1 && unmatched_baseline.len() > 1 {
        return Err(FlowError::InvalidArgument(
            "cannot safely preserve provider fields for multiple edited array items without stable identities"
                .into(),
        ));
    }

    if !structurally_changed && edited.len() == baseline.len() {
        for index in unmatched_edited {
            alignment[index] = Some(index);
        }
    }

    Ok(alignment)
}

fn patch_changed_array(original: &[Json], baseline: &[Json], edited: &[Json]) -> Result<Vec<Json>> {
    let alignment = align_changed_items(edited, baseline)?;
    edited
        .iter()
        .enumerate()
        .map(|(edited_index, edited_value)| {
            if let Some(baseline_index) = alignment[edited_index]
                && let (Some(original_value), Some(baseline_value)) =
                    (original.get(baseline_index), baseline.get(baseline_index))
            {
                return patch_changed_json(original_value, baseline_value, edited_value);
            }
            Ok(edited_value.clone())
        })
        .collect()
}

fn patch_changed_json(original: &Json, baseline: &Json, edited: &Json) -> Result<Json> {
    if baseline == edited {
        return Ok(original.clone());
    }

    match (original, baseline, edited) {
        (Json::Object(original), Json::Object(baseline), Json::Object(edited)) => {
            if ["type", "role"]
                .into_iter()
                .any(|key| baseline.get(key) != edited.get(key))
            {
                return Ok(Json::Object(edited.clone()));
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
                        patch_changed_json(original_value, baseline_value, edited_value)?
                    }
                    _ => edited_value.clone(),
                };
                patched.insert(key.clone(), value);
            }
            Ok(Json::Object(patched))
        }
        (Json::Array(original), Json::Array(baseline), Json::Array(edited)) => Ok(Json::Array(
            patch_changed_array(original, baseline, edited)?,
        )),
        _ => Ok(edited.clone()),
    }
}

#[cfg(test)]
#[path = "../../tests/unit/codec/parity_tests.rs"]
mod parity_tests;
