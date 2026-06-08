// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay::plugin::PluginError;

use super::builtin::mask_text;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum BuiltinDetector {
    Email,
    Phone,
    ApiKey,
    IpAddress,
    Ipv6,
    Url,
    Uuid,
    BearerToken,
    Jwt,
    CreditCard,
    AwsAccessKeyId,
    AwsSecretAccessKey,
    GcpApiKey,
    AzureStorageAccountKey,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BuiltinDetectorCategory {
    CommonPii,
    StructuredSecret,
    CloudCredential,
}

#[derive(Clone, Copy)]
struct BuiltinDetectorSpec {
    detector: BuiltinDetector,
    name: &'static str,
    category: BuiltinDetectorCategory,
    regex_pattern: &'static str,
}

const BUILTIN_DETECTOR_SPECS: &[BuiltinDetectorSpec] = &[
    BuiltinDetectorSpec {
        detector: BuiltinDetector::Email,
        name: "email",
        category: BuiltinDetectorCategory::CommonPii,
        regex_pattern: r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    },
    BuiltinDetectorSpec {
        detector: BuiltinDetector::Phone,
        name: "phone",
        category: BuiltinDetectorCategory::CommonPii,
        regex_pattern: r"\+?[0-9][0-9()\-\s]{6,}[0-9]",
    },
    BuiltinDetectorSpec {
        detector: BuiltinDetector::ApiKey,
        name: "api_key",
        category: BuiltinDetectorCategory::StructuredSecret,
        regex_pattern: r"(?:sk|rk|pk|ak)-[A-Za-z0-9_-]{8,}",
    },
    BuiltinDetectorSpec {
        detector: BuiltinDetector::IpAddress,
        name: "ip_address",
        category: BuiltinDetectorCategory::CommonPii,
        regex_pattern: r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    },
    BuiltinDetectorSpec {
        detector: BuiltinDetector::Ipv6,
        name: "ipv6",
        category: BuiltinDetectorCategory::CommonPii,
        regex_pattern: r"\b(?:[A-Fa-f0-9]{1,4}:){2,7}[A-Fa-f0-9]{1,4}\b",
    },
    BuiltinDetectorSpec {
        detector: BuiltinDetector::Url,
        name: "url",
        category: BuiltinDetectorCategory::CommonPii,
        regex_pattern: r"https?://[^\s]+",
    },
    BuiltinDetectorSpec {
        detector: BuiltinDetector::Uuid,
        name: "uuid",
        category: BuiltinDetectorCategory::StructuredSecret,
        regex_pattern: r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-8][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b",
    },
    BuiltinDetectorSpec {
        detector: BuiltinDetector::BearerToken,
        name: "bearer_token",
        category: BuiltinDetectorCategory::StructuredSecret,
        regex_pattern: r"(?i)\bBearer\s+[A-Za-z0-9._~+/\-]+=*\b",
    },
    BuiltinDetectorSpec {
        detector: BuiltinDetector::Jwt,
        name: "jwt",
        category: BuiltinDetectorCategory::StructuredSecret,
        regex_pattern: r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b",
    },
    BuiltinDetectorSpec {
        detector: BuiltinDetector::CreditCard,
        name: "credit_card",
        category: BuiltinDetectorCategory::StructuredSecret,
        regex_pattern: r"\b(?:\d[ -]?){13,19}\b",
    },
    BuiltinDetectorSpec {
        detector: BuiltinDetector::AwsAccessKeyId,
        name: "aws_access_key_id",
        category: BuiltinDetectorCategory::CloudCredential,
        regex_pattern: r"\b(?:A3T[A-Z0-9]|AKIA|ASIA|ABIA|ACCA|AGPA|AIDA|AIPA|ANPA|ANVA|APKA|AROA|AUSA)[A-Z0-9]{16}\b",
    },
    BuiltinDetectorSpec {
        detector: BuiltinDetector::AwsSecretAccessKey,
        name: "aws_secret_access_key",
        category: BuiltinDetectorCategory::CloudCredential,
        regex_pattern: r"\b[A-Za-z0-9/+=]{40}\b",
    },
    BuiltinDetectorSpec {
        detector: BuiltinDetector::GcpApiKey,
        name: "gcp_api_key",
        category: BuiltinDetectorCategory::CloudCredential,
        regex_pattern: r"\bAIza[0-9A-Za-z\-_]{35}\b",
    },
    BuiltinDetectorSpec {
        detector: BuiltinDetector::AzureStorageAccountKey,
        name: "azure_storage_account_key",
        category: BuiltinDetectorCategory::CloudCredential,
        regex_pattern: r"\b[A-Za-z0-9+/]{86}==\b",
    },
];

impl BuiltinDetector {
    pub(super) fn parse(value: &str) -> Result<Self, PluginError> {
        BUILTIN_DETECTOR_SPECS
            .iter()
            .find(|spec| spec.name == value)
            .map(|spec| spec.detector)
            .ok_or_else(|| {
                PluginError::InvalidConfig(format!("unsupported builtin.detector '{value}'"))
            })
    }

    fn spec(self) -> &'static BuiltinDetectorSpec {
        BUILTIN_DETECTOR_SPECS
            .iter()
            .find(|spec| spec.detector == self)
            .expect("every builtin detector must have a metadata spec")
    }

    pub(super) fn regex_pattern(self) -> &'static str {
        self.spec().regex_pattern
    }

    pub(super) fn default_mask(self, text: &str, mask_char: &str) -> String {
        match self {
            Self::Email => mask_email(text, mask_char),
            Self::Phone => mask_phone(text, mask_char),
            Self::ApiKey => mask_api_key(text, mask_char),
            Self::IpAddress => mask_ip_address(text, mask_char),
            Self::Ipv6 => mask_ipv6(text, mask_char),
            Self::Url => mask_url(text, mask_char),
            Self::Uuid => mask_text(text, mask_char, 0, 4),
            Self::BearerToken => mask_bearer_token(text, mask_char),
            Self::Jwt => mask_jwt(text, mask_char),
            Self::CreditCard => mask_credit_card(text, mask_char),
            Self::AwsAccessKeyId => mask_text(text, mask_char, 4, 4),
            Self::AwsSecretAccessKey => mask_text(text, mask_char, 0, 4),
            Self::GcpApiKey => mask_text(text, mask_char, 6, 4),
            Self::AzureStorageAccountKey => mask_text(text, mask_char, 0, 4),
        }
    }
}

pub(super) fn detector_regex_pattern(detector: &str) -> Option<&'static str> {
    BuiltinDetector::parse(detector)
        .ok()
        .map(BuiltinDetector::regex_pattern)
}

fn supported_detector_names_for_category(
    category: BuiltinDetectorCategory,
) -> impl Iterator<Item = &'static str> {
    BUILTIN_DETECTOR_SPECS
        .iter()
        .filter(move |spec| spec.category == category)
        .map(|spec| spec.name)
}

pub(super) fn supported_detector_summary() -> String {
    let common = supported_detector_names_for_category(BuiltinDetectorCategory::CommonPii)
        .collect::<Vec<_>>()
        .join(", ");
    let structured =
        supported_detector_names_for_category(BuiltinDetectorCategory::StructuredSecret)
            .collect::<Vec<_>>()
            .join(", ");
    let cloud = supported_detector_names_for_category(BuiltinDetectorCategory::CloudCredential)
        .collect::<Vec<_>>()
        .join(", ");
    format!("common PII: {common}; structured secrets: {structured}; cloud credentials: {cloud}")
}

fn mask_email(text: &str, mask_char: &str) -> String {
    let Some((local, domain)) = text.split_once('@') else {
        return mask_text(text, mask_char, 0, 0);
    };

    let local_chars: Vec<char> = local.chars().collect();
    if local_chars.len() <= 1 {
        return text.to_string();
    }

    let mut output = String::new();
    output.push(local_chars[0]);
    for _ in 1..local_chars.len() {
        output.push_str(mask_char);
    }
    output.push('@');
    output.push_str(domain);
    output
}

fn mask_phone(text: &str, mask_char: &str) -> String {
    let total_digits = text.chars().filter(|ch| ch.is_ascii_digit()).count();
    if total_digits <= 4 {
        return text.to_string();
    }

    let mut masked_digits_remaining = total_digits - 4;
    let mut output = String::with_capacity(text.len());
    for ch in text.chars() {
        if ch.is_ascii_digit() {
            if masked_digits_remaining > 0 {
                output.push_str(mask_char);
                masked_digits_remaining -= 1;
            } else {
                output.push(ch);
            }
        } else {
            output.push(ch);
        }
    }
    output
}

fn mask_api_key(text: &str, mask_char: &str) -> String {
    let prefix = text.find('-').map_or(0, |idx| idx + 1);
    mask_text(text, mask_char, prefix, 4)
}

fn mask_ip_address(text: &str, mask_char: &str) -> String {
    let mut octets = text.split('.').collect::<Vec<_>>();
    if octets.len() != 4 {
        return mask_text(text, mask_char, 0, 0);
    }

    for octet in octets.iter_mut().take(3) {
        *octet = "***";
    }
    octets.join(".")
}

fn mask_ipv6(text: &str, mask_char: &str) -> String {
    let mut segments = text.split(':').collect::<Vec<_>>();
    if segments.len() < 3 {
        return mask_text(text, mask_char, 0, 0);
    }

    let visible_tail_start = segments.len().saturating_sub(1);
    for segment in segments.iter_mut().take(visible_tail_start) {
        if !segment.is_empty() {
            *segment = "****";
        }
    }
    segments.join(":")
}

fn mask_url(text: &str, mask_char: &str) -> String {
    let Some(scheme_idx) = text.find("://") else {
        return mask_text(text, mask_char, 0, 0);
    };
    let prefix_end = scheme_idx + 3;
    let remainder = &text[prefix_end..];
    let Some(path_idx) = remainder.find('/') else {
        return text.to_string();
    };

    let mut output = String::with_capacity(text.len());
    output.push_str(&text[..prefix_end + path_idx + 1]);
    output.push_str(mask_char);
    output
}

fn mask_bearer_token(text: &str, mask_char: &str) -> String {
    let Some((scheme, token)) = text.split_once(char::is_whitespace) else {
        return mask_text(text, mask_char, 0, 4);
    };
    let trimmed = token.trim_start();
    if trimmed.is_empty() {
        return text.to_string();
    }

    let mut output = String::new();
    output.push_str(scheme);
    output.push(' ');
    output.push_str(&mask_text(trimmed, mask_char, 0, 4));
    output
}

fn mask_jwt(text: &str, mask_char: &str) -> String {
    let parts = text.split('.').collect::<Vec<_>>();
    if parts.len() != 3 {
        return mask_text(text, mask_char, 0, 6);
    }

    format!(
        "{}.{}.{}",
        parts[0],
        mask_text(parts[1], mask_char, 0, 0),
        mask_text(parts[2], mask_char, 0, 6)
    )
}

fn mask_credit_card(text: &str, mask_char: &str) -> String {
    let total_digits = text.chars().filter(|ch| ch.is_ascii_digit()).count();
    if total_digits <= 4 {
        return text.to_string();
    }

    let mut masked_digits_remaining = total_digits - 4;
    let mut output = String::with_capacity(text.len());
    for ch in text.chars() {
        if ch.is_ascii_digit() {
            if masked_digits_remaining > 0 {
                output.push_str(mask_char);
                masked_digits_remaining -= 1;
            } else {
                output.push(ch);
            }
        } else {
            output.push(ch);
        }
    }
    output
}
