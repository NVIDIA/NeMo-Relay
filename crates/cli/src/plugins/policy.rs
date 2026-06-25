// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::fmt;

use nemo_relay::plugin::dynamic::{
    DynamicPluginAttestationMode, DynamicPluginCheckState, DynamicPluginFailure,
    DynamicPluginFailurePhase, DynamicPluginKind, DynamicPluginManifest, DynamicPluginStartupClass,
};
use serde::Deserialize;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct DynamicPluginHostPolicy {
    pub(crate) defaults: DynamicPluginHostPolicyEffect,
    pub(crate) rules: Vec<DynamicPluginHostPolicyRule>,
    pub(crate) overrides: BTreeMap<String, DynamicPluginHostPolicyEffect>,
}

impl DynamicPluginHostPolicy {
    pub(crate) fn merge_from(&mut self, other: Self) {
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

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct DynamicPluginHostPolicyEffect {
    pub(crate) allowed: Option<bool>,
    pub(crate) startup: Option<DynamicPluginStartupClass>,
    pub(crate) attestation: Option<DynamicPluginAttestationMode>,
    pub(crate) trusted_public_keys: Option<Vec<String>>,
}

impl DynamicPluginHostPolicyEffect {
    fn merge_from(&mut self, other: Self) {
        if let Some(value) = other.allowed {
            self.allowed = Some(value);
        }
        if let Some(value) = other.startup {
            self.startup = Some(value);
        }
        if let Some(value) = other.attestation {
            self.attestation = Some(value);
        }
        if let Some(value) = other.trusted_public_keys {
            self.trusted_public_keys = Some(value);
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct DynamicPluginHostPolicyRule {
    pub(crate) match_kind: Option<DynamicPluginKind>,
    pub(crate) match_plugin_id: Option<String>,
    pub(crate) effect: DynamicPluginHostPolicyEffect,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DynamicPluginHostPolicyFailure {
    Blocked,
}

impl DynamicPluginHostPolicyFailure {
    pub(crate) fn display<'a>(
        &'a self,
        plugin_id: &'a str,
    ) -> DynamicPluginHostPolicyFailureDisplay<'a> {
        DynamicPluginHostPolicyFailureDisplay {
            failure: self,
            plugin_id,
        }
    }
}

pub(crate) struct DynamicPluginHostPolicyFailureDisplay<'a> {
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EvaluatedDynamicPluginHostPolicy {
    pub(crate) policy_satisfied: bool,
    pub(crate) startup_class: DynamicPluginStartupClass,
    pub(crate) attestation_mode: DynamicPluginAttestationMode,
    pub(crate) trusted_public_keys: Vec<String>,
    pub(crate) failure: Option<DynamicPluginHostPolicyFailure>,
}

impl EvaluatedDynamicPluginHostPolicy {
    pub(crate) fn check_state(&self) -> DynamicPluginCheckState {
        if self.policy_satisfied {
            DynamicPluginCheckState::Valid
        } else {
            DynamicPluginCheckState::Invalid
        }
    }

    pub(crate) fn last_error(&self, plugin_id: &str) -> Option<DynamicPluginFailure> {
        self.failure.as_ref().map(|failure| DynamicPluginFailure {
            phase: DynamicPluginFailurePhase::Policy,
            code: "policy_blocked".into(),
            message: failure.display(plugin_id).to_string(),
        })
    }

    pub(crate) fn failure(&self) -> Option<&DynamicPluginHostPolicyFailure> {
        self.failure.as_ref()
    }
}

pub(crate) fn evaluate_dynamic_plugin_host_policy(
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
        if !policy_rule_matches(rule, manifest) {
            continue;
        }
        effect.merge_from(rule.effect.clone());
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

    if effect.allowed == Some(false) {
        return EvaluatedDynamicPluginHostPolicy {
            policy_satisfied: false,
            startup_class,
            attestation_mode,
            trusted_public_keys,
            failure: Some(DynamicPluginHostPolicyFailure::Blocked),
        };
    }

    EvaluatedDynamicPluginHostPolicy {
        policy_satisfied: true,
        startup_class,
        attestation_mode,
        trusted_public_keys,
        failure: None,
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

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FileDynamicPluginHostPolicy {
    #[serde(default)]
    pub(crate) defaults: FileDynamicPluginHostPolicyEffect,
    #[serde(default)]
    pub(crate) rules: Vec<FileDynamicPluginHostPolicyRule>,
    #[serde(default)]
    pub(crate) overrides: BTreeMap<String, FileDynamicPluginHostPolicyEffect>,
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

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FileDynamicPluginHostPolicyEffect {
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

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FileDynamicPluginHostPolicyRule {
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
