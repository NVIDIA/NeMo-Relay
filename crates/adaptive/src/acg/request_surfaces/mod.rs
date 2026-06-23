// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request surface appliers for semantic ACG hint plans.
//!
//! Each applier consumes a semantic `hint plan` and mutates one concrete
//! request surface such as Anthropic Messages, OpenAI Chat, or OpenAI
//! Responses. Provider selection stays semantic (`AdaptiveConfig.acg.provider`),
//! request-surface resolution stays internal, and response codecs remain
//! observability-only helpers on the post-execution path.

pub(crate) mod anthropic_messages;
pub(crate) mod openai_chat;
pub(crate) mod openai_responses;

use std::collections::HashSet;

use nemo_relay::api::llm::LlmRequest;
use nemo_relay::codec::resolve::{ProviderSurface, detect_request_surface};
use serde_json::Value;

use crate::acg::prompt_ir::PromptIR;
use crate::acg::translation::{HintPlan, HintTarget};

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RequestSurface {
    AnthropicMessages,
    OpenAIChat,
    OpenAIResponses,
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) trait RequestSurfaceApplier: Send + Sync {
    fn apply(
        &self,
        request: &LlmRequest,
        prompt_ir: &PromptIR,
        plan: &HintPlan,
    ) -> crate::acg::Result<LlmRequest>;
}

impl RequestSurface {
    fn from_provider_surface(surface: ProviderSurface) -> Self {
        match surface {
            ProviderSurface::OpenAiChat => Self::OpenAIChat,
            ProviderSurface::OpenAiResponses => Self::OpenAIResponses,
            ProviderSurface::AnthropicMessages => Self::AnthropicMessages,
        }
    }

    pub(crate) fn supports_provider(self, provider: &str) -> bool {
        match provider {
            "anthropic" => matches!(self, Self::AnthropicMessages),
            "openai" => matches!(self, Self::OpenAIChat | Self::OpenAIResponses),
            "passthrough" => true,
            _ => false,
        }
    }

    pub(crate) fn apply(
        self,
        request: &LlmRequest,
        prompt_ir: &PromptIR,
        plan: &HintPlan,
    ) -> crate::acg::Result<LlmRequest> {
        match self {
            Self::AnthropicMessages => {
                anthropic_messages::AnthropicMessages.apply(request, prompt_ir, plan)
            }
            Self::OpenAIChat => openai_chat::OpenAIChat.apply(request, prompt_ir, plan),
            Self::OpenAIResponses => {
                openai_responses::OpenAIResponses.apply(request, prompt_ir, plan)
            }
        }
    }
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn resolve_request_surface_from_request(
    request: &LlmRequest,
) -> crate::acg::Result<RequestSurface> {
    detect_request_surface(&request.content)
        .map(RequestSurface::from_provider_surface)
        .ok_or_else(|| {
            crate::acg::AcgError::Internal(
                "unable to resolve request surface from request shape".to_string(),
            )
        })
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn resolve_request_surface(
    provider: &str,
    request: &LlmRequest,
) -> crate::acg::Result<RequestSurface> {
    let surface = resolve_request_surface_from_request(request)?;
    if surface.supports_provider(provider) {
        Ok(surface)
    } else {
        Err(crate::acg::AcgError::Internal(format!(
            "provider '{provider}' does not support resolved request surface {surface:?}"
        )))
    }
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn apply_request_surface(
    provider: &str,
    request: &LlmRequest,
    prompt_ir: &PromptIR,
    plan: &HintPlan,
) -> crate::acg::Result<LlmRequest> {
    resolve_request_surface(provider, request)?.apply(request, prompt_ir, plan)
}

pub(crate) fn canonicalize_tools(content: &mut Value) -> std::result::Result<u32, String> {
    let tools = match content.get_mut("tools").and_then(Value::as_array_mut) {
        Some(arr) => arr,
        None => return Ok(0),
    };

    let mut count = 0u32;
    let mut last_error: Option<String> = None;

    for tool in tools.iter_mut() {
        match canonicalize_tool(tool) {
            Ok(()) => count += 1,
            Err(error) => last_error = Some(error),
        }
    }

    if let Some(error) = last_error
        && count == 0
    {
        return Err(error);
    }

    Ok(count)
}

fn canonicalize_tool(tool: &mut Value) -> std::result::Result<(), String> {
    let canonical = crate::acg::canonicalize::canonicalize_value(tool)
        .map_err(|error| format!("canonicalization failed: {error}"))?;
    *tool = serde_json::from_str(&canonical)
        .map_err(|error| format!("failed to re-parse canonical JSON: {error}"))?;
    Ok(())
}

pub(crate) fn resolve_target_block_index(
    prompt_ir: &PromptIR,
    target: &HintTarget,
) -> Option<usize> {
    if let Some(span_id) = target.last_span_id() {
        prompt_ir
            .blocks
            .iter()
            .position(|block| &block.span_id == span_id)
    } else {
        target
            .end_exclusive()
            .and_then(|end_exclusive| end_exclusive.checked_sub(1))
            .filter(|index| *index < prompt_ir.blocks.len())
    }
}

pub(crate) fn target_block_indices(prompt_ir: &PromptIR, target: &HintTarget) -> Vec<usize> {
    match target {
        HintTarget::Span { .. } => resolve_target_block_index(prompt_ir, target)
            .into_iter()
            .collect(),
        HintTarget::StablePrefix { end_exclusive, .. } => {
            let resolved_end = resolve_target_block_index(prompt_ir, target)
                .map(|index| index + 1)
                .unwrap_or(*end_exclusive)
                .min(prompt_ir.blocks.len());
            (0..resolved_end).collect()
        }
    }
}

pub(crate) fn prompt_ir_tool_index(prompt_ir: &PromptIR, target_block_index: usize) -> usize {
    prompt_ir
        .blocks
        .iter()
        .take(target_block_index + 1)
        .filter(|block| block.content_type == crate::acg::prompt_ir::BlockContentType::ToolSchema)
        .count()
        .saturating_sub(1)
}

pub(crate) fn prompt_ir_message_index(
    prompt_ir: &PromptIR,
    target_block_index: usize,
    include_system_messages: bool,
) -> usize {
    prompt_ir
        .blocks
        .iter()
        .take(target_block_index + 1)
        .filter(|block| {
            block.content_type != crate::acg::prompt_ir::BlockContentType::ToolSchema
                && (include_system_messages
                    || block.role != crate::acg::prompt_ir::PromptRole::System)
        })
        .count()
        .saturating_sub(1)
}

pub(crate) fn canonicalize_message_content_blocks(message: &mut Value) -> bool {
    let Some(msg_content) = message.get_mut("content") else {
        return false;
    };
    canonicalize_content_blocks(msg_content)
}

pub(crate) fn canonicalize_content_blocks(content: &mut Value) -> bool {
    let Some(blocks) = content.as_array_mut() else {
        return false;
    };

    let mut changed = false;
    for block in blocks.iter_mut() {
        let block_type = block.get("type").and_then(Value::as_str).unwrap_or("text");
        if block_type != "text"
            && let Ok(canonical) = crate::acg::canonicalize::canonicalize_value(block)
            && let Ok(parsed) = serde_json::from_str::<Value>(&canonical)
        {
            *block = parsed;
            changed = true;
        }
    }

    changed
}

pub(crate) fn canonicalize_target_messages(
    content: &mut Value,
    prompt_ir: &PromptIR,
    target: &HintTarget,
    include_system_messages: bool,
    message_field: &str,
) {
    let Some(messages) = content.get_mut(message_field).and_then(Value::as_array_mut) else {
        return;
    };

    let mut seen = HashSet::new();
    for block_index in target_block_indices(prompt_ir, target) {
        let block = &prompt_ir.blocks[block_index];
        if block.content_type == crate::acg::prompt_ir::BlockContentType::ToolSchema {
            continue;
        }
        let message_index =
            prompt_ir_message_index(prompt_ir, block_index, include_system_messages);
        if !seen.insert(message_index) {
            continue;
        }
        if let Some(message) = messages.get_mut(message_index) {
            let _ = canonicalize_message_content_blocks(message);
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/acg/request_surface_tests.rs"]
mod tests;
