// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Structured prefilter required by the pinned Rampart model's input contract.

use std::cmp::Reverse;
use std::net::{Ipv4Addr, Ipv6Addr};
use std::str::FromStr;
use std::sync::LazyLock;

use regex::Regex;

static EMAIL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
        .expect("Rampart email pattern must compile")
});
static SCHEME_URL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"\bhttps?://[^\s<>"'\])}]+"#).expect("Rampart URL pattern must compile")
});
static WWW_URL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"\bwww\.[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:/[^\s<>"'\])}]*)?"#)
        .expect("Rampart www URL pattern must compile")
});
static IPV4_CANDIDATE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b").expect("Rampart IPv4 pattern must compile")
});
static MAC_ADDRESS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b(?:(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}|(?:[0-9A-Fa-f]{2}-){5}[0-9A-Fa-f]{2})\b")
        .expect("Rampart MAC address pattern must compile")
});

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct StructuredSpan {
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) label: &'static str,
}

pub(super) struct PreparedText {
    masked: String,
    raw_starts: Vec<usize>,
    raw_ends: Vec<usize>,
    spans: Vec<StructuredSpan>,
}

impl PreparedText {
    pub(super) fn new(raw: &str) -> Self {
        let spans = merge_spans(detect_structured(raw));
        let mut masked = String::with_capacity(raw.len());
        let mut raw_starts = Vec::with_capacity(raw.len());
        let mut raw_ends = Vec::with_capacity(raw.len());
        let mut cursor = 0;

        for span in &spans {
            copy_verbatim(
                raw,
                cursor,
                span.start,
                &mut masked,
                &mut raw_starts,
                &mut raw_ends,
            );
            let sentinel = format!("[{}]", span.label);
            masked.push_str(&sentinel);
            raw_starts.extend(std::iter::repeat_n(span.start, sentinel.len()));
            raw_ends.extend(std::iter::repeat_n(span.end, sentinel.len()));
            cursor = span.end;
        }
        copy_verbatim(
            raw,
            cursor,
            raw.len(),
            &mut masked,
            &mut raw_starts,
            &mut raw_ends,
        );

        debug_assert_eq!(masked.len(), raw_starts.len());
        debug_assert_eq!(masked.len(), raw_ends.len());
        Self {
            masked,
            raw_starts,
            raw_ends,
            spans,
        }
    }

    pub(super) fn masked(&self) -> &str {
        &self.masked
    }

    pub(super) fn spans(&self) -> &[StructuredSpan] {
        &self.spans
    }

    pub(super) fn project(&self, start: usize, end: usize) -> Option<(usize, usize)> {
        if start >= end || end > self.masked.len() {
            return None;
        }
        let raw_start = *self.raw_starts.get(start)?;
        let raw_end = *self.raw_ends.get(end - 1)?;
        (raw_start < raw_end).then_some((raw_start, raw_end))
    }
}

fn copy_verbatim(
    raw: &str,
    start: usize,
    end: usize,
    masked: &mut String,
    raw_starts: &mut Vec<usize>,
    raw_ends: &mut Vec<usize>,
) {
    masked.push_str(&raw[start..end]);
    raw_starts.extend(start..end);
    raw_ends.extend((start + 1)..=end);
}

fn detect_structured(raw: &str) -> Vec<StructuredSpan> {
    let mut spans = detect_digit_entities(raw);
    detect_regex_entities(raw, "EMAIL", &EMAIL, &mut spans);
    detect_regex_entities(raw, "URL", &SCHEME_URL, &mut spans);
    detect_regex_entities(raw, "URL", &WWW_URL, &mut spans);
    for candidate in IPV4_CANDIDATE.find_iter(raw) {
        if Ipv4Addr::from_str(candidate.as_str()).is_ok() {
            spans.push(StructuredSpan {
                start: candidate.start(),
                end: candidate.end(),
                label: "IP_ADDRESS",
            });
        }
    }
    detect_ipv6(raw, &mut spans);
    detect_regex_entities(raw, "IP_ADDRESS", &MAC_ADDRESS, &mut spans);
    spans
}

fn detect_digit_entities(raw: &str) -> Vec<StructuredSpan> {
    let bytes = raw.as_bytes();
    let mut spans = Vec::new();
    let mut cursor = 0;
    while cursor < bytes.len() {
        if !bytes[cursor].is_ascii_digit() {
            cursor += 1;
            continue;
        }

        let start = cursor;
        let mut digits = Vec::new();
        digits.push(bytes[cursor]);
        cursor += 1;
        while cursor < bytes.len() {
            if bytes[cursor].is_ascii_digit() {
                digits.push(bytes[cursor]);
                cursor += 1;
                continue;
            }
            if matches!(bytes[cursor], b' ' | b'.' | b'-')
                && bytes.get(cursor + 1).is_some_and(u8::is_ascii_digit)
            {
                cursor += 1;
                digits.push(bytes[cursor]);
                cursor += 1;
                continue;
            }
            break;
        }

        let label = if matches!(digits.len(), 14..=16) && is_luhn_valid(&digits) {
            Some("CREDIT_CARD")
        } else if is_valid_ssn(&digits) {
            Some("SSN")
        } else {
            None
        };
        if let Some(label) = label {
            spans.push(StructuredSpan {
                start,
                end: cursor,
                label,
            });
        }
    }
    spans
}

fn is_luhn_valid(digits: &[u8]) -> bool {
    let mut sum = 0_u32;
    let mut double = false;
    for digit in digits.iter().rev() {
        let mut value = u32::from(*digit - b'0');
        if double {
            value *= 2;
            if value > 9 {
                value -= 9;
            }
        }
        sum += value;
        double = !double;
    }
    sum.rem_euclid(10) == 0
}

fn is_valid_ssn(digits: &[u8]) -> bool {
    if digits.len() != 9 {
        return false;
    }
    let area = u16::from(digits[0] - b'0') * 100
        + u16::from(digits[1] - b'0') * 10
        + u16::from(digits[2] - b'0');
    area != 0 && area != 666 && area < 900 && digits[3..5] != *b"00" && digits[5..] != *b"0000"
}

fn detect_regex_entities(
    raw: &str,
    label: &'static str,
    pattern: &Regex,
    spans: &mut Vec<StructuredSpan>,
) {
    spans.extend(pattern.find_iter(raw).map(|found| StructuredSpan {
        start: found.start(),
        end: found.end(),
        label,
    }));
}

fn detect_ipv6(raw: &str, spans: &mut Vec<StructuredSpan>) {
    let mut start = None;
    for (index, value) in raw.char_indices().chain(std::iter::once((raw.len(), '\0'))) {
        if value.is_ascii_hexdigit() || matches!(value, ':' | '.') {
            start.get_or_insert(index);
            continue;
        }
        let Some(candidate_start) = start.take() else {
            continue;
        };
        let candidate = &raw[candidate_start..index];
        if candidate.contains(':')
            && Ipv6Addr::from_str(candidate).is_ok()
            && has_network_boundaries(raw, candidate_start, index)
        {
            spans.push(StructuredSpan {
                start: candidate_start,
                end: index,
                label: "IP_ADDRESS",
            });
        }
    }
}

fn has_network_boundaries(raw: &str, start: usize, end: usize) -> bool {
    let invalid_boundary = |value: char| value.is_alphanumeric() || matches!(value, ':' | '.');
    !raw[..start]
        .chars()
        .next_back()
        .is_some_and(invalid_boundary)
        && !raw[end..].chars().next().is_some_and(invalid_boundary)
}

fn merge_spans(mut spans: Vec<StructuredSpan>) -> Vec<StructuredSpan> {
    spans.sort_by_key(|span| (span.start, Reverse(span.end), span.label));
    let mut merged: Vec<StructuredSpan> = Vec::new();
    for span in spans {
        let Some(previous) = merged.last_mut() else {
            merged.push(span);
            continue;
        };
        if span.start >= previous.end {
            merged.push(span);
            continue;
        }

        let previous_length = previous.end - previous.start;
        let span_length = span.end - span.start;
        if span_length > previous_length
            || (span_length == previous_length && span.label < previous.label)
        {
            previous.label = span.label;
        }
        previous.start = previous.start.min(span.start);
        previous.end = previous.end.max(span.end);
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels_and_text(raw: &str) -> Vec<(&'static str, &str)> {
        merge_spans(detect_structured(raw))
            .into_iter()
            .map(|span| (span.label, &raw[span.start..span.end]))
            .collect()
    }

    #[test]
    fn detects_validator_backed_identifiers_and_rejects_lookalikes() {
        for value in ["472-81-0094", "472 81 0094", "472.81.0094", "472810094"] {
            assert_eq!(labels_and_text(value), [("SSN", value)]);
        }
        for value in [
            "000-81-0094",
            "666-81-0094",
            "900-81-0094",
            "472-00-0094",
            "472-81-0000",
        ] {
            assert!(labels_and_text(value).is_empty(), "{value}");
        }

        assert_eq!(
            labels_and_text("card 4111 1111 1111 1111"),
            [("CREDIT_CARD", "4111 1111 1111 1111")]
        );
        assert_eq!(
            labels_and_text("card 378282246310005"),
            [("CREDIT_CARD", "378282246310005")]
        );
        assert_eq!(
            labels_and_text("card 30569309025904"),
            [("CREDIT_CARD", "30569309025904")]
        );
        assert!(labels_and_text("card 1234 5678 1234 5678").is_empty());
    }

    #[test]
    fn detects_text_and_network_identifiers_with_valid_boundaries() {
        let raw = "mail alex+home@sub.example.org, visit https://example.org/private, \
                   www.example.net/account, then use 10.0.0.1, 2001:db8::1, \
                   fe80::1ff:fe23:4567:890a, ::1, \
                   2001:0db8:85a3:0000:0000:8a2e:0370:7334, \
                   00:1B:44:11:3A:B7, or 00-1B-44-11-3A-B7";
        assert_eq!(
            labels_and_text(raw),
            [
                ("EMAIL", "alex+home@sub.example.org"),
                ("URL", "https://example.org/private,"),
                ("URL", "www.example.net/account,"),
                ("IP_ADDRESS", "10.0.0.1"),
                ("IP_ADDRESS", "2001:db8::1"),
                ("IP_ADDRESS", "fe80::1ff:fe23:4567:890a"),
                ("IP_ADDRESS", "::1"),
                ("IP_ADDRESS", "2001:0db8:85a3:0000:0000:8a2e:0370:7334"),
                ("IP_ADDRESS", "00:1B:44:11:3A:B7"),
                ("IP_ADDRESS", "00-1B-44-11-3A-B7"),
            ]
        );
        assert!(
            labels_and_text(
                "invalid 999.0.0.1, time 12:34:56, opcode ff:00, \
                 phone 415-555-2671, and address 31 Birchwood Avenue"
            )
            .is_empty()
        );
    }

    #[test]
    fn premask_preserves_utf8_projection_and_structured_offsets() {
        let raw = "José 472-81-0094 met Ana";
        let prepared = PreparedText::new(raw);
        assert_eq!(prepared.masked(), "José [SSN] met Ana");
        assert_eq!(
            prepared.spans(),
            &[StructuredSpan {
                start: 6,
                end: 17,
                label: "SSN",
            }]
        );

        let sentinel_start = prepared.masked().find("[SSN]").unwrap();
        assert_eq!(
            prepared.project(sentinel_start + 1, sentinel_start + 4),
            Some((6, 17))
        );
        let ana_start = prepared.masked().find("Ana").unwrap();
        assert_eq!(
            prepared.project(ana_start, ana_start + "Ana".len()),
            Some((22, 25))
        );
    }

    #[test]
    fn overlapping_structured_matches_produce_one_typed_sentinel() {
        let raw = "open https://10.0.0.1/private";
        let prepared = PreparedText::new(raw);
        assert_eq!(prepared.masked(), "open [URL]");
        assert_eq!(
            prepared.spans(),
            &[StructuredSpan {
                start: 5,
                end: raw.len(),
                label: "URL",
            }]
        );
    }
}
