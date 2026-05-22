// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Anthropic cache plugin for the Adaptive Cache Governor (ACG) system.
//!
//! Translates ACG stability classifications into Anthropic-specific
//! `cache_control` breakpoints on content blocks. Implements the
//! [`ProviderPlugin`] trait with:
//!
//! - **Breakpoint budget allocation**: Up to 4 breakpoints per request.
//! - **Token minimum enforcement**: Per-model from [`CapabilityRegistry`].
//! - **TTL mapping**: `RetentionTier` to Anthropic TTL (5m default or 1h extended).
//! - **Tool schema canonicalization**: RFC 8785 via [`canonicalize_value`].
//! - **System string-to-array conversion**: Handles both string and array-of-blocks format.
//!
//! # Threat mitigations
//!
//! - T-08-01: Only injects `cache_control` annotations with fixed structure `{"type": "ephemeral"}`.
//! - T-08-02: `scope_label` is logged in detail but does NOT change cache visibility.
//! - T-08-03: All JSON access uses Option-returning methods; errors propagate via `Result`.
//! - T-08-05: TTL is hardcoded to exactly 2 behaviors (omit or `"1h"`).

use crate::acg::capability::{BackendCapabilities, CapabilityRegistry};
use crate::acg::plugin::{
    HintPlanApplier, PluginInput, PluginOutput, ProviderPlugin, translate_with_hint_plan,
};
use crate::acg::prompt_ir::PromptIR;
use crate::acg::translation::anthropic::AnthropicHintTranslator;
use crate::acg::translation::{HintPlan, HintTranslation, HintTranslator};

// ===================================================================
// AnthropicCachePlugin
// ===================================================================

/// Anthropic-specific provider plugin for cache_control breakpoint injection.
///
/// Translates `CacheStability` and `Retention` intents into Anthropic's
/// explicit `cache_control` annotations. Other intent types are marked
/// `Ignored` / `NotRelevant`.
///
/// # Construction
///
/// ```rust,ignore
/// let registry = CapabilityRegistry::with_defaults();
/// let plugin = AnthropicCachePlugin::new(&registry);
/// ```
pub struct AnthropicCachePlugin {
    translator: AnthropicHintTranslator,
    capabilities: BackendCapabilities,
}

impl AnthropicCachePlugin {
    /// Create a new Anthropic cache plugin backed by the given capability registry.
    ///
    /// The registry is cloned into an `Arc` for shared ownership.
    pub fn new(registry: &CapabilityRegistry) -> Self {
        Self {
            translator: AnthropicHintTranslator::new(registry),
            capabilities: registry
                .get_backend("anthropic")
                .cloned()
                .unwrap_or_else(|| BackendCapabilities::none("anthropic")),
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn build_hint_translation(
        &self,
        input: &PluginInput<'_>,
    ) -> crate::acg::error::Result<HintTranslation> {
        self.translator.translate(input)
    }
}

impl ProviderPlugin for AnthropicCachePlugin {
    fn plugin_id(&self) -> &str {
        "anthropic"
    }

    fn plugin_name(&self) -> &str {
        "Anthropic Cache Plugin"
    }

    fn translate(&self, input: &PluginInput<'_>) -> crate::acg::error::Result<PluginOutput> {
        translate_with_hint_plan(&self.translator, self, input)
    }

    fn capabilities(&self) -> BackendCapabilities {
        self.capabilities.clone()
    }
}

impl HintPlanApplier for AnthropicCachePlugin {
    fn apply_hint_plan(
        &self,
        request: &nemo_relay::api::llm::LlmRequest,
        prompt_ir: &PromptIR,
        hint_plan: &HintPlan,
    ) -> crate::acg::error::Result<nemo_relay::api::llm::LlmRequest> {
        crate::acg::request_surfaces::apply_request_surface(
            self.plugin_id(),
            request,
            prompt_ir,
            hint_plan,
        )
    }
}

#[cfg(test)]
#[path = "../../tests/unit/acg/anthropic_plugin_tests.rs"]
mod tests;
