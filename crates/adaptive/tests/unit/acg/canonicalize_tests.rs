// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for canonicalize in the NeMo Relay adaptive crate.

use super::*;

// -------------------------------------------------------------------
// RFC 8785 canonicalization
// -------------------------------------------------------------------

#[test]
fn test_rfc8785_key_ordering() {
    let a = canonicalize_json(r#"{"b":2,"a":1}"#).unwrap();
    let b = canonicalize_json(r#"{"a":1,"b":2}"#).unwrap();
    assert_eq!(a, b);
    // Keys should be sorted: "a" before "b"
    assert_eq!(a, r#"{"a":1,"b":2}"#);
}

#[test]
fn test_rfc8785_number_normalization() {
    // 1.0 should serialize without trailing zero per RFC 8785
    let result = canonicalize_json("1.0").unwrap();
    assert_eq!(result, "1");
    let result = canonicalize_json("1.10").unwrap();
    assert_eq!(result, "1.1");
}

#[test]
fn test_rfc8785_nested_objects() {
    let a = canonicalize_json(r#"{"z":{"b":2,"a":1},"y":3}"#).unwrap();
    let b = canonicalize_json(r#"{"y":3,"z":{"a":1,"b":2}}"#).unwrap();
    assert_eq!(a, b);
}

// -------------------------------------------------------------------
// Whitespace normalization
// -------------------------------------------------------------------

#[test]
fn test_normalize_whitespace_trims() {
    assert_eq!(normalize_whitespace("  hello  "), "hello");
    assert_eq!(normalize_whitespace("\thello\t"), "hello");
    assert_eq!(normalize_whitespace("\n hello \n"), "hello");
}

#[test]
fn test_normalize_whitespace_collapses_internal() {
    assert_eq!(normalize_whitespace("a   b"), "a b");
    assert_eq!(normalize_whitespace("a\t\tb"), "a b");
    assert_eq!(normalize_whitespace("a  \t  b"), "a b");
}

#[test]
fn test_normalize_whitespace_preserves_single_newlines() {
    assert_eq!(normalize_whitespace("a\nb"), "a\nb");
    assert_eq!(normalize_whitespace("a \t\nb"), "a\nb");
    assert_eq!(
        normalize_whitespace("line1\nline2\nline3"),
        "line1\nline2\nline3"
    );
}

// -------------------------------------------------------------------
// SHA-256 hashing
// -------------------------------------------------------------------

#[test]
fn test_sha256_hex_deterministic() {
    let h1 = sha256_hex("hello world");
    let h2 = sha256_hex("hello world");
    assert_eq!(h1, h2);
}

#[test]
fn test_sha256_hex_different_inputs() {
    let h1 = sha256_hex("hello");
    let h2 = sha256_hex("world");
    assert_ne!(h1, h2);
}

#[test]
fn test_sha256_hex_format() {
    let h = sha256_hex("test");
    assert!(
        h.starts_with("sha256:"),
        "expected sha256: prefix, got: {h}"
    );
    // After prefix, should be hex characters
    let hex_part = &h["sha256:".len()..];
    assert!(
        hex_part.chars().all(|c| c.is_ascii_hexdigit()),
        "non-hex chars in: {hex_part}"
    );
    // SHA-256 produces 64 hex characters
    assert_eq!(
        hex_part.len(),
        64,
        "expected 64 hex chars, got {}",
        hex_part.len()
    );
}

// -------------------------------------------------------------------
// canonicalize_json edge cases
// -------------------------------------------------------------------

#[test]
fn test_canonicalize_json_valid() {
    let result = canonicalize_json(r#"{"key": "value"}"#).unwrap();
    assert_eq!(result, r#"{"key":"value"}"#);
}

#[test]
fn test_canonicalize_json_invalid() {
    let result = canonicalize_json("not valid json");
    assert!(result.is_err());
}

// -------------------------------------------------------------------
// Integration tests (Task 2)
// -------------------------------------------------------------------

#[test]
fn test_canonicalize_tool_schema_deterministic() {
    // Two tool schemas with identical logical content but different key orders
    let schema_a = r#"{"type":"function","function":{"name":"search","description":"Search","parameters":{"type":"object","properties":{"q":{"type":"string"}}}}}"#;
    let schema_b = r#"{"function":{"parameters":{"properties":{"q":{"type":"string"}},"type":"object"},"description":"Search","name":"search"},"type":"function"}"#;
    let canon_a = canonicalize_json(schema_a).unwrap();
    let canon_b = canonicalize_json(schema_b).unwrap();
    assert_eq!(
        canon_a, canon_b,
        "Canonical form should be identical for same logical schema"
    );
    // Hashes should also match
    assert_eq!(sha256_hex(&canon_a), sha256_hex(&canon_b));
}

#[test]
fn test_normalize_whitespace_unicode() {
    // Non-breaking space (U+00A0) and em space (U+2003) should be collapsed
    let input = "hello\u{00A0}\u{00A0}world";
    assert_eq!(normalize_whitespace(input), "hello world");

    let input2 = "foo\u{2003}bar";
    assert_eq!(normalize_whitespace(input2), "foo bar");

    // Mix of regular and unicode whitespace
    let input3 = "a \u{00A0} \t b";
    assert_eq!(normalize_whitespace(input3), "a b");
}

#[test]
fn test_normalize_whitespace_handles_empty_and_whitespace_only_inputs() {
    assert_eq!(normalize_whitespace(""), "");
    assert_eq!(normalize_whitespace("   \t  "), "");
}
