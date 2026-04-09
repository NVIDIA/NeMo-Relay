// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LLM codec types, traits, and built-in implementations.
//!
//! This module provides the type system and traits for bidirectional
//! request codec ([`LlmCodec`] / [`AnnotatedLLMRequest`]) and will host
//! the decode-only response codec ([`LlmResponseCodec`] / `AnnotatedLLMResponse`).

mod request;
mod response;
mod traits;

mod anthropic;
mod openai_chat;
mod openai_responses;

pub use request::*;
pub use response::*;
pub use traits::*;

pub use anthropic::*;
pub use openai_chat::*;
pub use openai_responses::*;
