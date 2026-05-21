// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider plugin trait and input/output types for the Adaptive Cache Governor
//! (ACG) system.
//!
//! The [`ProviderPlugin`] trait defines the contract between ACG's
//! provider-agnostic optimization pipeline and backend-specific
//! translation logic. Plugins receive a [`PluginInput`] containing
//! the original request, Prompt IR, intent bundle, and agent identity,
//! and produce a [`PluginOutput`] with the translated request and a
//! [`TranslationReport`].
//!
//! # Design
//!
//! - **Synchronous**: `translate` is a pure data transform (JSON
//!   restructuring), not an I/O operation. This matches the `LlmCodec`
//!   pattern in `crates/core/src/codec/traits.rs`.
//! - **Compatibility facade**: provider plugins keep their existing
//!   synchronous trait surface, but can internally split translation into
//!   semantic hint planning plus request-surface application.
//! - **`Send + Sync`**: Required for storage as `Arc<dyn ProviderPlugin>`
//!   in concurrent contexts.
//! - **Object-safe**: The trait is designed to be used as a trait object.

use nemo_relay::api::llm::LlmRequest;

use crate::acg::capability::BackendCapabilities;
use crate::acg::prompt_ir::PromptIR;
use crate::acg::request_surfaces::RequestSurfaceApplier;
use crate::acg::translation::{HintPlan, HintTranslation, HintTranslator};
use crate::acg::types::{AgentIdentity, OptimizationIntentBundle, TranslationReport};

// ===================================================================
// Plugin input / output types
// ===================================================================

/// Input data provided to a plugin for translation.
///
/// All fields are borrowed references to avoid unnecessary cloning.
/// The lifetime `'a` ties the input to the caller's data.
#[derive(Debug)]
pub struct PluginInput<'a> {
    /// The original (pre-rewrite) request.
    pub original_request: &'a LlmRequest,
    /// The rewritten request (may be identical to original in early phases).
    pub rewritten_request: &'a LlmRequest,
    /// The Prompt IR decomposition of the request.
    pub prompt_ir: &'a PromptIR,
    /// The optimization intent bundle from the policy engine.
    pub intent_bundle: &'a OptimizationIntentBundle,
    /// The agent identity for context.
    pub agent_identity: &'a AgentIdentity,
}

/// Output produced by a plugin after translation.
///
/// Contains the translated request in the backend's native API format
/// and a [`TranslationReport`] describing what happened to each intent.
#[derive(Debug)]
pub struct PluginOutput {
    /// The final request in the backend's native API format.
    pub translated_request: LlmRequest,
    /// Report describing what happened to each intent.
    pub translation_report: TranslationReport,
}

/// Internal adapter boundary used while the provider-plugin facade remains stable.
///
/// Translators produce surface-agnostic semantic plans. Appliers remain
/// responsible for delegating raw request mutation to dedicated request-surface
/// modules while the public provider-plugin facade remains stable.
pub(crate) trait HintPlanApplier {
    fn apply_hint_plan(
        &self,
        request: &LlmRequest,
        prompt_ir: &PromptIR,
        hint_plan: &HintPlan,
    ) -> crate::acg::error::Result<LlmRequest>;
}

impl<T> RequestSurfaceApplier for T
where
    T: HintPlanApplier + Send + Sync + ?Sized,
{
    fn apply(
        &self,
        request: &LlmRequest,
        prompt_ir: &PromptIR,
        plan: &HintPlan,
    ) -> crate::acg::Result<LlmRequest> {
        self.apply_hint_plan(request, prompt_ir, plan)
    }
}

/// Run the internal translator -> applier split behind the stable plugin facade.
pub(crate) fn translate_with_hint_plan<T, A>(
    translator: &T,
    applier: &A,
    input: &PluginInput<'_>,
) -> crate::acg::error::Result<PluginOutput>
where
    T: HintTranslator,
    A: HintPlanApplier + Send + Sync,
{
    let HintTranslation {
        hint_plan,
        translation_report,
    } = translator.translate(input)?;

    let translated_request = RequestSurfaceApplier::apply(
        applier,
        input.rewritten_request,
        input.prompt_ir,
        &hint_plan,
    )?;

    Ok(PluginOutput {
        translated_request,
        translation_report,
    })
}

// ===================================================================
// ProviderPlugin trait
// ===================================================================

/// Provider plugin trait for backend-specific translation.
///
/// Plugins are stateless on the forward path. They translate
/// provider-agnostic intents into backend-native API parameters.
///
/// # Object Safety
///
/// This trait is object-safe and can be stored as `Arc<dyn ProviderPlugin>`.
///
/// # Thread Safety
///
/// The `Send + Sync` bounds allow plugins to be shared across async
/// tasks and threads.
pub trait ProviderPlugin: Send + Sync {
    /// Unique identifier for this plugin (e.g., "anthropic", "openai", "passthrough").
    fn plugin_id(&self) -> &str;

    /// Human-readable name for this plugin.
    fn plugin_name(&self) -> &str;

    /// Translate intents into backend-native API parameters.
    fn translate(&self, input: &PluginInput<'_>) -> crate::acg::error::Result<PluginOutput>;

    /// Report the capabilities of the backend this plugin targets.
    fn capabilities(&self) -> BackendCapabilities;
}

#[cfg(test)]
#[path = "../../tests/unit/acg/plugin_tests.rs"]
mod tests;
