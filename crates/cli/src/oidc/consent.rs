// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Consent restrictions an administrator pins in `[upstream.<provider>_oidc]`.
//!
//! OpenRouter's consent page reads them from the OAuth `state`: a random value, a `.`, and an
//! unpadded base64url JSON object with the members `organizationId`, `workspaceId`,
//! `creditLimit`, `creditLimitReset` and `hiddenFields`, in that order. The encoding matches the
//! Ori CLI byte for byte, so the one consent service enforces both clients the same way.

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use serde::{Deserialize, Serialize};

use super::OidcSource;
use crate::error::CliError;

/// The consent service refuses a longer restriction segment.
pub(crate) const MAX_SEGMENT_LEN: usize = 1024;
/// Largest integer a JSON number keeps exactly in JavaScript.
const MAX_SAFE_INTEGER: f64 = 9_007_199_254_740_991.0;

/// How often a pinned credit limit resets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum CreditLimitReset {
    Daily,
    Weekly,
    Monthly,
}

/// A consent page control the restriction hides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) enum ConsentField {
    KeyLabel,
    Expiration,
    CreditLimit,
    ExistingKey,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RestrictionClaim<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    organization_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    workspace_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    credit_limit: Option<serde_json::Number>,
    #[serde(skip_serializing_if = "Option::is_none")]
    credit_limit_reset: Option<CreditLimitReset>,
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    hidden_fields: &'a [ConsentField],
}

impl OidcSource {
    pub(crate) fn has_consent_restrictions(&self) -> bool {
        self.organization_id.is_some()
            || self.workspace_id.is_some()
            || self.credit_limit.is_some()
            || self.credit_limit_reset.is_some()
            || !self.hidden_fields.is_empty()
    }

    /// The encoded restriction segment, or `None` when the section pins nothing.
    pub(crate) fn restriction_segment(&self) -> Option<String> {
        if !self.has_consent_restrictions() {
            return None;
        }
        let claim = RestrictionClaim {
            organization_id: self.organization_id.as_deref(),
            workspace_id: self.workspace_id.as_deref(),
            credit_limit: self.credit_limit.as_ref().map(javascript_number),
            credit_limit_reset: self.credit_limit_reset,
            hidden_fields: &self.hidden_fields,
        };
        let json = serde_json::to_string(&claim).unwrap_or_default();
        Some(URL_SAFE_NO_PAD.encode(json))
    }

    /// The OAuth `state`: the random value alone, or followed by `.` and the restriction.
    pub(crate) fn authorization_state(&self, random: &str) -> String {
        match self.restriction_segment() {
            Some(segment) => format!("{random}.{segment}"),
            None => random.to_owned(),
        }
    }

    /// Refuses a restriction the consent service would refuse, when the configuration loads.
    pub(crate) fn validate_consent_restrictions(&self, name: &str) -> Result<(), CliError> {
        if !self.has_consent_restrictions() {
            return Ok(());
        }
        let invalid = |detail: &str| Err(CliError::Config(format!("{name}: {detail}")));
        let blank =
            |value: &Option<String>| value.as_deref().is_some_and(|id| id.trim().is_empty());
        if blank(&self.organization_id) || blank(&self.workspace_id) {
            return invalid("organization_id and workspace_id must not be empty");
        }
        if let Some(limit) = &self.credit_limit
            && !limit
                .as_f64()
                .is_some_and(|value| value.is_finite() && value >= 0.0)
        {
            return invalid("credit_limit must be a finite amount of zero or more");
        }
        if self.credit_limit_reset.is_some() && self.credit_limit.is_none() {
            return invalid("credit_limit_reset needs credit_limit");
        }
        if self.organization_id.is_none()
            && self.workspace_id.is_none()
            && self.credit_limit.is_none()
        {
            return invalid("hidden_fields needs organization_id, workspace_id or credit_limit");
        }
        let repeated = self
            .hidden_fields
            .iter()
            .enumerate()
            .any(|(index, field)| self.hidden_fields[..index].contains(field));
        if repeated {
            return invalid("hidden_fields lists a field twice");
        }
        if self
            .restriction_segment()
            .map_or(0, |segment| segment.len())
            > MAX_SEGMENT_LEN
        {
            return invalid(
                "the consent restriction does not fit the OAuth state (1024 characters)",
            );
        }
        Ok(())
    }
}

/// `JSON.stringify` writes a whole amount without a fraction, so `25.0` encodes as `25`.
fn javascript_number(number: &serde_json::Number) -> serde_json::Number {
    match number.as_f64() {
        Some(value)
            if number.is_f64()
                && value.fract() == 0.0
                && (0.0..=MAX_SAFE_INTEGER).contains(&value) =>
        {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            serde_json::Number::from(value as u64)
        }
        _ => number.clone(),
    }
}

#[cfg(test)]
#[path = "../../tests/coverage/shared/oidc_consent_tests.rs"]
mod tests;
