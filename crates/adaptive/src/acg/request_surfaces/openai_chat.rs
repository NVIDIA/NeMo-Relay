// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenAI Chat request-surface applier.

use nemo_relay::api::llm::LlmRequest;

use crate::acg::prompt_ir::PromptIR;
use crate::acg::request_surfaces::RequestSurfaceApplier;
use crate::acg::translation::{HintDirective, HintPlan, OpenAIHintDirective};

pub(crate) struct OpenAIChat;

impl RequestSurfaceApplier for OpenAIChat {
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
                    crate::acg::request_surfaces::canonicalize_target_messages(
                        content, prompt_ir, target, true, "messages",
                    );
                }
                HintDirective::Anthropic(_) => {}
            }
        }

        Ok(translated)
    }
}
