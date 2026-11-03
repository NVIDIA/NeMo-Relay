<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Relay Codecs

Provider request, response, and streaming implementations for OpenAI Chat,
OpenAI Responses, Anthropic Messages, Gemini generateContent, and OCI GenAI.

Use `nemo-relay-types` for codec traits, normalized request and response models,
and streaming contracts. `resolve::BuiltinLlmNormalizer` implements the shared
`LlmNormalizer` contract without runtime state or configured pricing.

Codecs preserve provider-reported costs. To include configured pricing, use
`nemo_relay_runtime::codec::RuntimeLlmNormalizer` or the runtime's
`codec::response_codec` factory after installing runtime composition. Runtime owns
codec registration and execution; this crate has no runtime or SDK dependency.

This crate participates in the unified NeMo Relay release.
