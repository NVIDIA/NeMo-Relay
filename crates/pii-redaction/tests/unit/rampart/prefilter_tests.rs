// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for Rampart structured PII prefiltering.

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
    assert_eq!(
        labels_and_text(
            "invalid 999.0.0.1, time 12:34:56, opcode ff:00, \
                 phone 415-555-2671, and address 31 Birchwood Avenue"
        ),
        [("PHONE", "415-555-2671")]
    );
}

#[test]
fn detects_formatted_phone_numbers_without_matching_plain_identifiers() {
    for value in [
        "+1 415-555-2671",
        "415-555-2671",
        "(415) 555-2671",
        "+14155552671",
    ] {
        assert_eq!(labels_and_text(value), [("PHONE", value)], "{value}");
    }

    for value in [
        "Order 12345 is ready",
        "build 1234567890",
        "release 2026-08-03",
        "abc415-555-2671xyz",
    ] {
        assert!(labels_and_text(value).is_empty(), "{value}");
    }
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

#[test]
fn projection_rejects_empty_reversed_and_out_of_bounds_ranges() {
    let prepared = PreparedText::new("José 472-81-0094");

    assert_eq!(prepared.project(0, 0), None);
    assert_eq!(prepared.project(2, 1), None);
    assert_eq!(prepared.project(0, prepared.masked().len() + 1), None);
}
