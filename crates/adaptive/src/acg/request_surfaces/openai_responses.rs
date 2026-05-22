// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenAI Responses request-surface applier.

use nemo_relay::api::llm::LlmRequest;
use serde_json::Value;

use crate::acg::prompt_ir::PromptIR;
use crate::acg::request_surfaces::RequestSurfaceApplier;
use crate::acg::translation::{HintDirective, HintPlan, OpenAIHintDirective};

pub(crate) struct OpenAIResponses;

impl RequestSurfaceApplier for OpenAIResponses {
    fn apply(
        &self,
        request: &LlmRequest,
        prompt_ir: &PromptIR,
        plan: &HintPlan,
    ) -> crate::acg::Result<LlmRequest> {
        let mut translated = request.clone();
        let content = &mut translated.content;

        for directive in &plan.directives {
            match directive {
                HintDirective::OpenAI(OpenAIHintDirective::CanonicalizeToolSchemas) => {
                    let _ = crate::acg::request_surfaces::canonicalize_tools(content);
                }
                HintDirective::OpenAI(OpenAIHintDirective::CanonicalizeStablePrefix { target }) => {
                    canonicalize_responses_stable_prefix(content, prompt_ir, target);
                }
                HintDirective::Anthropic(_) => {}
            }
        }

        Ok(translated)
    }
}

fn canonicalize_responses_stable_prefix(
    content: &mut Value,
    prompt_ir: &PromptIR,
    target: &crate::acg::translation::HintTarget,
) {
    let Some(input_items) = content.get_mut("input").and_then(Value::as_array_mut) else {
        return;
    };

    let mut seen = std::collections::HashSet::new();
    for block_index in crate::acg::request_surfaces::target_block_indices(prompt_ir, target) {
        let block = &prompt_ir.blocks[block_index];
        if block.content_type == crate::acg::prompt_ir::BlockContentType::ToolSchema
            || block.role == crate::acg::prompt_ir::PromptRole::System
        {
            continue;
        }

        let input_index =
            crate::acg::request_surfaces::prompt_ir_message_index(prompt_ir, block_index, false);
        if !seen.insert(input_index) {
            continue;
        }
        if let Some(item) = input_items.get_mut(input_index) {
            let _ = crate::acg::request_surfaces::canonicalize_message_content_blocks(item);
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/acg/openai_responses_surface_tests.rs"]
mod tests;
