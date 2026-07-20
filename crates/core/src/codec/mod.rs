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
    edited
        .iter()
        .enumerate()
        .map(|(index, item)| {
            if baseline.get(index) == Some(item)
                && let Some(original) = original.and_then(|items| items.get(index))
            {
                return Ok(original.clone());
            }
            encode(item)
        })
        .collect()
}

#[cfg(test)]
#[path = "../../tests/unit/codec/parity_tests.rs"]
mod parity_tests;
