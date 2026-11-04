// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Passthrough plugin that applies no transformations.
//!
//! Used as a baseline for A/B testing, as the default when no
//! provider-specific plugin is configured, and for backends with no
//! explicit cache control APIs.
//!
//! [`PassthroughPlugin`] implements [`ProviderPlugin`] by cloning the
//! rewritten request unchanged and generating a [`TranslationReport`]
//! where every intent is marked [`TranslationStatus::Ignored`](crate::acg::types::TranslationStatus::Ignored) with
//! [`ReasonCode::NotRelevant`].

use crate::acg::capability::BackendCapabilities;
use crate::acg::plugin::{PluginInput, PluginOutput, ProviderPlugin};
use crate::acg::types::{ReasonCode, TranslationReport};

/// A no-op provider plugin that passes requests through unchanged.
///
/// Returns the `rewritten_request` as-is (cloned) and generates a
/// [`TranslationReport`] where every intent is marked
/// [`TranslationStatus::Ignored`](crate::acg::types::TranslationStatus::Ignored) with [`ReasonCode::NotRelevant`].
///
/// # Usage
///
/// - Default plugin when no provider-specific plugin is configured
/// - Baseline for A/B testing (compare against optimized plugins)
/// - Backends with no explicit cache control APIs
pub struct PassthroughPlugin;

impl ProviderPlugin for PassthroughPlugin {
    fn plugin_id(&self) -> &str {
        "passthrough"
    }

    fn plugin_name(&self) -> &str {
        "Passthrough (No-Op)"
    }

    fn translate(&self, input: &PluginInput<'_>) -> crate::acg::error::Result<PluginOutput> {
        let translated_request = input.rewritten_request.clone();
        let translation_report = TranslationReport::all_ignored(
            input.intent_bundle,
            "passthrough",
            ReasonCode::NotRelevant,
            Some("passthrough plugin applies no transformations".to_string()),
        );
        Ok(PluginOutput {
            translated_request,
            translation_report,
        })
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::none("passthrough")
    }
}

#[cfg(test)]
#[path = "../../tests/unit/acg/passthrough_tests.rs"]
mod tests;
