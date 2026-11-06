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
const STATE_SEPARATOR: char = '.';
/// Largest whole number JavaScript keeps exact.
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

/// A restriction the consent service would refuse.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub(crate) enum RestrictionError {
    #[error("organization_id and workspace_id must not be empty")]
    EmptyIdentifier,
    #[error("credit_limit must be a finite amount of zero or more")]
    InvalidCreditLimit,
    #[error("credit_limit_reset needs credit_limit")]
    ResetWithoutLimit,
    #[error("hidden_fields needs organization_id, workspace_id or credit_limit")]
    HiddenFieldsWithoutPin,
    #[error("hidden_fields lists a field twice")]
    RepeatedHiddenField,
    #[error("the consent restriction does not fit the OAuth state ({MAX_SEGMENT_LEN} characters)")]
    TooLong,
}

/// The JSON the consent service decodes; serde writes the members in declaration order.
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

impl<'a> From<&'a OidcSource> for RestrictionClaim<'a> {
    fn from(source: &'a OidcSource) -> Self {
        Self {
            organization_id: source.organization_id.as_deref(),
            workspace_id: source.workspace_id.as_deref(),
            credit_limit: source.credit_limit.as_ref().map(javascript_number),
            credit_limit_reset: source.credit_limit_reset,
            hidden_fields: &source.hidden_fields,
        }
    }
}

impl OidcSource {
    pub(crate) fn has_consent_restrictions(&self) -> bool {
        self.has_pin() || self.credit_limit_reset.is_some() || !self.hidden_fields.is_empty()
    }

    /// An organization, a workspace or a credit limit: what makes a restriction claim.
    fn has_pin(&self) -> bool {
        self.organization_id.is_some() || self.workspace_id.is_some() || self.credit_limit.is_some()
    }

    /// The encoded restriction segment, or `None` when the section pins nothing.
    pub(crate) fn restriction_segment(&self) -> Option<String> {
        if !self.has_consent_restrictions() {
            return None;
        }
        let json = serde_json::to_string(&RestrictionClaim::from(self)).ok()?;
        Some(URL_SAFE_NO_PAD.encode(json))
    }

    /// The OAuth `state`: the random value alone, or followed by `.` and the restriction.
    pub(crate) fn authorization_state(&self, random: &str) -> String {
        match self.restriction_segment() {
            Some(segment) => format!("{random}{STATE_SEPARATOR}{segment}"),
            None => random.to_owned(),
        }
    }

    /// Refuses, when the configuration loads, a restriction the consent service would refuse.
    pub(crate) fn validate_consent_restrictions(&self, name: &str) -> Result<(), CliError> {
        self.restriction_error().map_or(Ok(()), |error| {
            Err(CliError::Config(format!("{name}: {error}")))
        })
    }

    fn restriction_error(&self) -> Option<RestrictionError> {
        if !self.has_consent_restrictions() {
            return None;
        }
        let checks = [
            Self::check_identifiers,
            Self::check_credit_limit,
            Self::check_reset,
            Self::check_pin,
            Self::check_hidden_fields,
            Self::check_length,
        ];
        checks.iter().find_map(|check| check(self).err())
    }

    fn check_identifiers(&self) -> Result<(), RestrictionError> {
        let blank = |id: &Option<String>| id.as_deref().is_some_and(|id| id.trim().is_empty());
        if blank(&self.organization_id) || blank(&self.workspace_id) {
            return Err(RestrictionError::EmptyIdentifier);
        }
        Ok(())
    }

    fn check_credit_limit(&self) -> Result<(), RestrictionError> {
        let valid = |limit: &serde_json::Number| {
            limit
                .as_f64()
                .is_some_and(|value| value.is_finite() && value >= 0.0)
        };
        match &self.credit_limit {
            Some(limit) if !valid(limit) => Err(RestrictionError::InvalidCreditLimit),
            _ => Ok(()),
        }
    }

    fn check_reset(&self) -> Result<(), RestrictionError> {
        if self.credit_limit_reset.is_some() && self.credit_limit.is_none() {
            return Err(RestrictionError::ResetWithoutLimit);
        }
        Ok(())
    }

    fn check_pin(&self) -> Result<(), RestrictionError> {
        if self.has_pin() {
            Ok(())
        } else {
            Err(RestrictionError::HiddenFieldsWithoutPin)
        }
    }

    fn check_hidden_fields(&self) -> Result<(), RestrictionError> {
        let fields = &self.hidden_fields;
        let repeated = (1..fields.len()).any(|index| fields[..index].contains(&fields[index]));
        if repeated {
            return Err(RestrictionError::RepeatedHiddenField);
        }
        Ok(())
    }

    fn check_length(&self) -> Result<(), RestrictionError> {
        let length = self
            .restriction_segment()
            .map_or(0, |segment| segment.len());
        if length > MAX_SEGMENT_LEN {
            return Err(RestrictionError::TooLong);
        }
        Ok(())
    }
}

/// `JSON.stringify` writes a whole amount without a fraction, so `25.0` encodes as `25`.
fn javascript_number(number: &serde_json::Number) -> serde_json::Number {
    let whole = number
        .as_f64()
        .filter(|value| number.is_f64() && value.fract() == 0.0)
        .filter(|value| (0.0..=MAX_SAFE_INTEGER).contains(value));
    match whole {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        Some(value) => serde_json::Number::from(value as u64),
        None => number.clone(),
    }
}

#[cfg(test)]
#[path = "../../tests/coverage/shared/oidc_consent_tests.rs"]
mod tests;
