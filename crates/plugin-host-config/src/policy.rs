// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::fmt;

use nemo_relay::plugin::dynamic::{
    DynamicPluginAttestationMode, DynamicPluginCheckState, DynamicPluginFailure,
    DynamicPluginFailurePhase, DynamicPluginKind, DynamicPluginManifest, DynamicPluginStartupClass,
};
use serde::Deserialize;

/// Layered host policy applied to discovered dynamic plugins.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DynamicPluginHostPolicy {
    /// Default effects inherited by every dynamic plugin.
    pub defaults: DynamicPluginHostPolicyEffect,
    /// Ordered matching rules.
    pub rules: Vec<DynamicPluginHostPolicyRule>,
    /// Plugin-ID-specific effects.
    pub overrides: BTreeMap<String, DynamicPluginHostPolicyEffect>,
}

impl DynamicPluginHostPolicy {
    /// Layers a higher-precedence policy onto this policy.
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
}

/// One resolved dynamic-plugin host-policy effect.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DynamicPluginHostPolicyEffect {
    /// Whether matching plugins may activate.
    pub allowed: Option<bool>,
    /// Required or optional startup classification.
    pub startup: Option<DynamicPluginStartupClass>,
    /// Required artifact attestation mode.
    pub attestation: Option<DynamicPluginAttestationMode>,
    /// Trusted Ed25519 public keys.
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

/// One ordered dynamic-plugin host-policy rule.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DynamicPluginHostPolicyRule {
    /// Optional dynamic execution-lane selector.
    pub match_kind: Option<DynamicPluginKind>,
    /// Optional canonical plugin-ID selector.
    pub match_plugin_id: Option<String>,
    /// Effect applied when selectors match.
    pub effect: DynamicPluginHostPolicyEffect,
}

/// Reason a dynamic plugin was rejected by host policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DynamicPluginHostPolicyFailure {
    /// Host policy explicitly blocks the plugin.
    Blocked,
}

impl DynamicPluginHostPolicyFailure {
    /// Formats this failure for one canonical plugin ID.
    pub fn display<'a>(&'a self, plugin_id: &'a str) -> DynamicPluginHostPolicyFailureDisplay<'a> {
        DynamicPluginHostPolicyFailureDisplay {
            failure: self,
            plugin_id,
        }
    }
}

/// Display adapter for a policy failure associated with a plugin ID.
pub struct DynamicPluginHostPolicyFailureDisplay<'a> {
    failure: &'a DynamicPluginHostPolicyFailure,
    plugin_id: &'a str,
}

impl fmt::Display for DynamicPluginHostPolicyFailureDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.failure {
            DynamicPluginHostPolicyFailure::Blocked => write!(
                f,
                "dynamic plugin '{}' is blocked by host policy",
                self.plugin_id
            ),
        }
    }
}

/// Effective policy decision for one validated dynamic-plugin manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvaluatedDynamicPluginHostPolicy {
    /// Whether policy permits activation.
    pub policy_satisfied: bool,
    /// Effective startup class.
    pub startup_class: DynamicPluginStartupClass,
    /// Effective attestation mode.
    pub attestation_mode: DynamicPluginAttestationMode,
    /// Effective trusted Ed25519 public keys.
    pub trusted_public_keys: Vec<String>,
    /// Policy rejection, if any.
    pub failure: Option<DynamicPluginHostPolicyFailure>,
}

impl EvaluatedDynamicPluginHostPolicy {
    /// Converts the decision to durable lifecycle validation state.
    pub fn check_state(&self) -> DynamicPluginCheckState {
        if self.policy_satisfied {
            DynamicPluginCheckState::Valid
        } else {
            DynamicPluginCheckState::Invalid
        }
    }

    /// Returns the durable lifecycle failure for this decision.
    pub fn last_error(&self, plugin_id: &str) -> Option<DynamicPluginFailure> {
        self.failure.as_ref().map(|failure| DynamicPluginFailure {
            phase: DynamicPluginFailurePhase::Policy,
            code: "policy_blocked".into(),
            message: failure.display(plugin_id).to_string(),
        })
    }

    /// Returns the policy rejection, if any.
    pub fn failure(&self) -> Option<&DynamicPluginHostPolicyFailure> {
        self.failure.as_ref()
    }
}

/// Evaluates one manifest against a fully layered host policy.
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
        if policy_rule_matches(rule, manifest) {
            effect.merge_from(rule.effect.clone());
        }
    }
    if let Some(override_effect) = policy.overrides.get(manifest.plugin.id.trim()) {
        effect.merge_from(override_effect.clone());
    }

    let startup_class = effect
        .startup
        .unwrap_or(DynamicPluginStartupClass::Optional);
    let attestation_mode = effect
        .attestation
        .unwrap_or(DynamicPluginAttestationMode::IntegrityOnly);
    let trusted_public_keys = effect.trusted_public_keys.unwrap_or_default();
    let failure =
        (effect.allowed == Some(false)).then_some(DynamicPluginHostPolicyFailure::Blocked);
    EvaluatedDynamicPluginHostPolicy {
        policy_satisfied: failure.is_none(),
        startup_class,
        attestation_mode,
        trusted_public_keys,
        failure,
    }
}

fn policy_rule_matches(
    rule: &DynamicPluginHostPolicyRule,
    manifest: &DynamicPluginManifest,
) -> bool {
    if let Some(match_kind) = rule.match_kind
        && manifest.plugin.kind != match_kind
    {
        return false;
    }
    if let Some(match_plugin_id) = &rule.match_plugin_id
        && manifest.plugin.id.trim() != match_plugin_id
    {
        return false;
    }
    true
}

/// TOML representation of dynamic-plugin host policy.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileDynamicPluginHostPolicy {
    /// Default policy effects.
    #[serde(default)]
    pub defaults: FileDynamicPluginHostPolicyEffect,
    /// Ordered policy rules.
    #[serde(default)]
    pub rules: Vec<FileDynamicPluginHostPolicyRule>,
    /// Plugin-ID-specific effects.
    #[serde(default)]
    pub overrides: BTreeMap<String, FileDynamicPluginHostPolicyEffect>,
}

impl From<FileDynamicPluginHostPolicy> for DynamicPluginHostPolicy {
    fn from(value: FileDynamicPluginHostPolicy) -> Self {
        Self {
            defaults: value.defaults.into(),
            rules: value.rules.into_iter().map(Into::into).collect(),
            overrides: value
                .overrides
                .into_iter()
                .map(|(plugin_id, effect)| (plugin_id.trim().to_owned(), effect.into()))
                .collect(),
        }
    }
}

/// TOML representation of a dynamic-plugin policy effect.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileDynamicPluginHostPolicyEffect {
    allowed: Option<bool>,
    startup: Option<DynamicPluginStartupClass>,
    attestation: Option<DynamicPluginAttestationMode>,
    trusted_public_keys: Option<Vec<String>>,
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

/// TOML representation of one dynamic-plugin policy rule.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileDynamicPluginHostPolicyRule {
    match_kind: Option<DynamicPluginKind>,
    match_plugin_id: Option<String>,
    allowed: Option<bool>,
    startup: Option<DynamicPluginStartupClass>,
    attestation: Option<DynamicPluginAttestationMode>,
    trusted_public_keys: Option<Vec<String>>,
}

impl From<FileDynamicPluginHostPolicyRule> for DynamicPluginHostPolicyRule {
    fn from(value: FileDynamicPluginHostPolicyRule) -> Self {
        Self {
            match_kind: value.match_kind,
            match_plugin_id: value.match_plugin_id.map(|value| value.trim().to_owned()),
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
