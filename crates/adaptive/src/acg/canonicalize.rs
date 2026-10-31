// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RFC 8785 JSON Canonicalization and whitespace normalization for cache stability.
//!
//! This module provides deterministic JSON serialization (RFC 8785 / JCS) and
//! whitespace normalization so that semantically equivalent content always
//! produces byte-identical output. This is critical for cache key stability:
//! without canonicalization, tool schemas or JSON content with different key
//! orders would produce different hashes and miss the cache.
//!
//! # Public functions
//!
//! - [`canonicalize_json`] -- Canonicalize a JSON string per RFC 8785.
//! - [`canonicalize_value`] -- Canonicalize a `serde_json::Value` per RFC 8785.
//! - [`normalize_whitespace`] -- Trim and collapse whitespace in text content.
//! - [`sha256_hex`] -- Compute SHA-256 hex digest with `"sha256:"` prefix.

use sha2::{Digest, Sha256};

use super::error::Result;

/// Canonicalize a JSON string per RFC 8785 (JCS).
///
/// Parses the input as JSON, then re-serializes using deterministic key
/// ordering and number formatting per the JSON Canonicalization Scheme.
///
/// # Errors
///
/// Returns [`AcgError::Serialization`](super::error::AcgError::Serialization) if the input is not valid JSON, or
/// [`AcgError::Internal`](super::error::AcgError::Internal) if canonicalization fails.
pub fn canonicalize_json(json_str: &str) -> Result<String> {
    let value: serde_json::Value = serde_json::from_str(json_str)?;
    canonicalize_value(&value)
}

/// Canonicalize a `serde_json::Value` per RFC 8785.
///
/// # Errors
///
/// Returns [`AcgError::Internal`](super::error::AcgError::Internal) if canonicalization fails.
pub fn canonicalize_value(value: &serde_json::Value) -> Result<String> {
    let canonical = serde_json_canonicalizer::to_string(value)
        .map_err(|e| super::error::AcgError::Internal(format!("canonicalization failed: {e}")))?;
    Ok(canonical)
}

/// Normalize whitespace in text content.
///
/// 1. Trims leading and trailing whitespace.
/// 2. Collapses runs of horizontal whitespace (spaces, tabs, and other
///    Unicode whitespace characters except `\n`) into a single ASCII space.
/// 3. Preserves single newline characters (they separate paragraphs in prompts).
pub fn normalize_whitespace(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    let mut result = String::with_capacity(trimmed.len());
    let mut in_whitespace_run = false;

    for ch in trimmed.chars() {
        if ch == '\n' {
            // Newlines are preserved; end any whitespace run first
            if in_whitespace_run {
                // Drop trailing horizontal whitespace before newline
                in_whitespace_run = false;
            }
            result.push('\n');
        } else if ch.is_whitespace() {
            // Any whitespace character (space, tab, NBSP, em space, etc.)
            in_whitespace_run = true;
        } else {
            if in_whitespace_run {
                result.push(' ');
                in_whitespace_run = false;
            }
            result.push(ch);
        }
    }
    result
}

/// Compute SHA-256 hex digest with `"sha256:"` prefix.
///
/// The output format is `sha256:<hex>` where `<hex>` is the lowercase
/// hexadecimal encoding of the 256-bit digest.
pub fn sha256_hex(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    let result = hasher.finalize();
    let hex: String = result.iter().map(|b| format!("{b:02x}")).collect();
    format!("sha256:{hex}")
}

#[cfg(test)]
#[path = "../../tests/unit/acg/canonicalize_tests.rs"]
mod tests;
