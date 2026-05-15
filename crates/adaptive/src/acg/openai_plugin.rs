// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenAI cache plugin for the Adaptive Cache Governor (ACG) system.
//!
//! Maximizes automatic prefix cache hits through deterministic JSON
//! serialization. OpenAI uses automatic prefix caching at 1024+ tokens
//! with exact prefix matching -- no explicit annotations are needed.
//! The plugin's job is to ensure that semantically identical prefixes
//! produce byte-identical JSON so the cache hits rather than misses.
//!
//! Implements the [`ProviderPlugin`] trait with:
//!
//! - **Tool schema canonicalization**: RFC 8785 via [`canonicalize_value`]
//!   for deterministic key ordering in function parameter schemas.
//! - **Stable message content canonicalization**: Structured JSON content
//!   blocks in the stable prefix are canonicalized for byte-identical output.
//! - **No annotations injected**: OpenAI handles caching automatically.
//!
//! # Threat mitigations
//!
//! - T-08-06: RFC 8785 is a semantic-preserving transform (only reorders keys,
//!   normalizes numbers). The plugin canonicalizes tool schemas (structured JSON)
//!   but does NOT modify text content in messages.
//! - T-08-09: If canonicalization fails for one tool, the plugin reports Degraded
//!   (not Applied) and continues with remaining tools.

use crate::acg::capability::{BackendCapabilities, CapabilityRegistry};
use crate::acg::plugin::{
    HintPlanApplier, PluginInput, PluginOutput, ProviderPlugin, translate_with_hint_plan,
};
use crate::acg::prompt_ir::PromptIR;
use crate::acg::translation::openai::OpenAIHintTranslator;
use crate::acg::translation::{HintPlan, HintTranslation, HintTranslator};

// ===================================================================
// OpenAICachePlugin
// ===================================================================

/// OpenAI-specific provider plugin for deterministic JSON serialization.
///
/// Ensures that semantically identical request prefixes produce
/// byte-identical JSON output, maximizing OpenAI's automatic prefix
/// cache hit rate. Stateless -- no constructor arguments needed.
///
/// # Usage
///
/// ```rust,ignore
/// let plugin = OpenAICachePlugin;
/// let output = plugin.translate(&input)?;
/// ```
pub struct OpenAICachePlugin;

impl OpenAICachePlugin {
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn build_hint_translation(
        &self,
        input: &PluginInput<'_>,
    ) -> crate::acg::error::Result<HintTranslation> {
        let translator = OpenAIHintTranslator;
        translator.translate(input)
    }
}

impl ProviderPlugin for OpenAICachePlugin {
    fn plugin_id(&self) -> &str {
        "openai"
    }

    fn plugin_name(&self) -> &str {
        "OpenAI Cache Plugin"
    }

    fn translate(&self, input: &PluginInput<'_>) -> crate::acg::error::Result<PluginOutput> {
        let translator = OpenAIHintTranslator;
        translate_with_hint_plan(&translator, self, input)
    }

    fn capabilities(&self) -> BackendCapabilities {
        CapabilityRegistry::with_defaults()
            .get_backend("openai")
            .cloned()
            .unwrap_or_else(|| BackendCapabilities::none("openai"))
    }
}

impl HintPlanApplier for OpenAICachePlugin {
    fn apply_hint_plan(
        &self,
        request: &nemo_flow::api::llm::LlmRequest,
        prompt_ir: &PromptIR,
        hint_plan: &HintPlan,
    ) -> crate::acg::error::Result<nemo_flow::api::llm::LlmRequest> {
        crate::acg::request_surfaces::apply_request_surface(
            self.plugin_id(),
            request,
            prompt_ir,
            hint_plan,
        )
    }
}

#[cfg(test)]
#[path = "../../tests/unit/acg/openai_plugin_tests.rs"]
mod tests;
