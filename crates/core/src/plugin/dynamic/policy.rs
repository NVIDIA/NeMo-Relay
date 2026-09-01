// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamic-plugin host trust policy.
//!
//! Policy is deliberately file-shaped: embedding APIs provide plugin
//! configuration, while authorization decisions remain auditable in
//! `plugins.toml` layers.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::{
    DynamicPluginAttestationMode, DynamicPluginCheckState, DynamicPluginFailure,
    DynamicPluginFailurePhase, DynamicPluginKind, DynamicPluginManifest, DynamicPluginStartupClass,
};

/// Effective host policy for dynamic plugins.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DynamicPluginHostPolicy {
    /// Rules applied to every plugin before matching rules and ID overrides.
    pub defaults: DynamicPluginHostPolicyEffect,
    /// Ordered matching rules.
    pub rules: Vec<DynamicPluginHostPolicyRule>,
    /// Per-plugin overrides.
    pub overrides: BTreeMap<String, DynamicPluginHostPolicyEffect>,
}

impl DynamicPluginHostPolicy {
    /// Overlays a higher-precedence policy onto this policy.
    pub fn merge_from(&mut self, other: Self) {
        self.defaults.merge_from(other.defaults);
        self.rules.extend(other.rules);
        for (plugin_id, effect) in other.overrides {
            self.overrides
                .entry(plugin_id)
                .or_default()
                .merge_from(effect);
        }
    }

    /// Applies Relay's fail-closed production defaults when fields were omitted.
    pub fn apply_secure_defaults(&mut self) {
        self.defaults
            .startup
            .get_or_insert(DynamicPluginStartupClass::Required);
        self.defaults
            .attestation
            .get_or_insert(DynamicPluginAttestationMode::SignatureRequired);
    }
}

/// A partial policy effect that may overlay another effect.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DynamicPluginHostPolicyEffect {
    /// Whether matching plugins are permitted to load.
    pub allowed: Option<bool>,
    /// Whether failure blocks startup.
    pub startup: Option<DynamicPluginStartupClass>,
    /// Required artifact attestation mode.
    pub attestation: Option<DynamicPluginAttestationMode>,
    /// Trusted Ed25519 public keys using the `ed25519:<base64>` encoding.
    pub trusted_public_keys: Option<Vec<String>>,
}

impl DynamicPluginHostPolicyEffect {
    fn merge_from(&mut self, other: Self) {
        if other.allowed.is_some() {
            self.allowed = other.allowed;
        }
        if other.startup.is_some() {
            self.startup = other.startup;
        }
        if other.attestation.is_some() {
            self.attestation = other.attestation;
        }
        if other.trusted_public_keys.is_some() {
            self.trusted_public_keys = other.trusted_public_keys;
        }
    }
}

/// One ordered policy rule.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DynamicPluginHostPolicyRule {
    /// Optional dynamic plugin kind selector.
    pub match_kind: Option<DynamicPluginKind>,
    /// Optional exact plugin ID selector.
    pub match_plugin_id: Option<String>,
    /// Effect applied when every selector matches.
    pub effect: DynamicPluginHostPolicyEffect,
}

/// The TOML representation accepted below `[plugins.policy]`.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FileDynamicPluginHostPolicy {
    /// Default effect.
    #[serde(default)]
    pub defaults: FileDynamicPluginHostPolicyEffect,
    /// Ordered rules.
    #[serde(default)]
    pub rules: Vec<FileDynamicPluginHostPolicyRule>,
    /// ID-specific effects.
    #[serde(default)]
    pub overrides: BTreeMap<String, FileDynamicPluginHostPolicyEffect>,
}

/// File-level partial policy effect.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FileDynamicPluginHostPolicyEffect {
    allowed: Option<bool>,
    startup: Option<DynamicPluginStartupClass>,
    attestation: Option<DynamicPluginAttestationMode>,
    trusted_public_keys: Option<Vec<String>>,
}

/// File-level ordered policy rule.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FileDynamicPluginHostPolicyRule {
    match_kind: Option<DynamicPluginKind>,
    match_plugin_id: Option<String>,
    allowed: Option<bool>,
    startup: Option<DynamicPluginStartupClass>,
    attestation: Option<DynamicPluginAttestationMode>,
    trusted_public_keys: Option<Vec<String>>,
}

impl From<FileDynamicPluginHostPolicy> for DynamicPluginHostPolicy {
    fn from(value: FileDynamicPluginHostPolicy) -> Self {
        Self {
            defaults: value.defaults.into(),
            rules: value.rules.into_iter().map(Into::into).collect(),
            overrides: value
                .overrides
                .into_iter()
                .map(|(id, effect)| (id.trim().to_owned(), effect.into()))
                .collect(),
        }
    }
}

impl From<FileDynamicPluginHostPolicyEffect> for DynamicPluginHostPolicyEffect {
    fn from(value: FileDynamicPluginHostPolicyEffect) -> Self {
        Self {
            allowed: value.allowed,
            startup: value.startup,
            attestation: value.attestation,
            trusted_public_keys: value
                .trusted_public_keys
                .map(|keys| keys.into_iter().map(|key| key.trim().to_owned()).collect()),
        }
    }
}

impl From<FileDynamicPluginHostPolicyRule> for DynamicPluginHostPolicyRule {
    fn from(value: FileDynamicPluginHostPolicyRule) -> Self {
        Self {
            match_kind: value.match_kind,
            match_plugin_id: value.match_plugin_id.map(|id| id.trim().to_owned()),
            effect: DynamicPluginHostPolicyEffect {
                allowed: value.allowed,
                startup: value.startup,
                attestation: value.attestation,
                trusted_public_keys: value
                    .trusted_public_keys
                    .map(|keys| keys.into_iter().map(|key| key.trim().to_owned()).collect()),
            },
        }
    }
}

/// The result of applying the effective host policy to one manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvaluatedDynamicPluginHostPolicy {
    /// Whether policy allows this plugin.
    pub policy_satisfied: bool,
    /// Effective startup class.
    pub startup_class: DynamicPluginStartupClass,
    /// Effective artifact-attestation mode.
    pub attestation_mode: DynamicPluginAttestationMode,
    /// Effective trusted signing keys.
    pub trusted_public_keys: Vec<String>,
    /// Policy failure, if denied.
    pub failure: Option<DynamicPluginHostPolicyFailure>,
}

/// A policy refusal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DynamicPluginHostPolicyFailure {
    /// A matching policy explicitly denied the plugin.
    Blocked,
}

impl DynamicPluginHostPolicyFailure {
    /// Returns a stable human-readable refusal description.
    pub fn display(self, plugin_id: &str) -> String {
        match self {
            Self::Blocked => format!("dynamic plugin '{plugin_id}' is blocked by host policy"),
        }
    }
}

impl EvaluatedDynamicPluginHostPolicy {
    /// Converts policy evaluation into the persisted check state.
    pub fn check_state(&self) -> DynamicPluginCheckState {
        if self.policy_satisfied {
            DynamicPluginCheckState::Valid
        } else {
            DynamicPluginCheckState::Invalid
        }
    }

    /// Returns the corresponding structured failure when denied.
    pub fn last_error(&self, plugin_id: &str) -> Option<DynamicPluginFailure> {
        self.failure.map(|failure| DynamicPluginFailure {
            phase: DynamicPluginFailurePhase::Policy,
            code: "policy_blocked".into(),
            message: failure.display(plugin_id),
        })
    }

    /// Returns the policy refusal, if any.
    pub fn failure(&self) -> Option<&DynamicPluginHostPolicyFailure> {
        self.failure.as_ref()
    }
}

/// Evaluates a dynamic plugin manifest against a resolved host policy.
pub fn evaluate_dynamic_plugin_host_policy(
    policy: &DynamicPluginHostPolicy,
    manifest: &DynamicPluginManifest,
) -> EvaluatedDynamicPluginHostPolicy {
    let mut effect = DynamicPluginHostPolicyEffect {
        allowed: Some(true),
        startup: Some(DynamicPluginStartupClass::Optional),
        attestation: Some(DynamicPluginAttestationMode::IntegrityOnly),
        trusted_public_keys: None,
    };
    effect.merge_from(policy.defaults.clone());
    for rule in &policy.rules {
        if rule
            .match_kind
            .is_some_and(|kind| kind != manifest.plugin.kind)
            || rule
                .match_plugin_id
                .as_deref()
                .is_some_and(|id| id != manifest.plugin.id.trim())
        {
            continue;
        }
        effect.merge_from(rule.effect.clone());
    }
    if let Some(effect_override) = policy.overrides.get(manifest.plugin.id.trim()) {
        effect.merge_from(effect_override.clone());
    }

    let failure =
        (effect.allowed == Some(false)).then_some(DynamicPluginHostPolicyFailure::Blocked);
    EvaluatedDynamicPluginHostPolicy {
        policy_satisfied: failure.is_none(),
        startup_class: effect
            .startup
            .unwrap_or(DynamicPluginStartupClass::Optional),
        attestation_mode: effect
            .attestation
            .unwrap_or(DynamicPluginAttestationMode::IntegrityOnly),
        trusted_public_keys: effect.trusted_public_keys.unwrap_or_default(),
        failure,
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/plugin_dynamic_policy_tests.rs"]
mod tests;
