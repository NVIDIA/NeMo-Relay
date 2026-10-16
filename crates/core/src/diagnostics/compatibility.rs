// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Normalize existing producer classifications without exposing arbitrary metadata keys.

use sha2::{Digest, Sha256};
use std::fmt::Write;

use super::{DiagnosticCode, valid_component_key};

/// Existing reports may append arbitrary application metadata keys to a code.
/// Preserve those public reports while using a stable bounded key in shared state.
pub(crate) fn legacy_diagnostic_code(code: &str) -> String {
    if DiagnosticCode::new(code).is_ok() {
        code.to_owned()
    } else {
        format!("legacy.code.{}", digest(code))
    }
}

/// Preserve the producer class and a stable opaque discriminator for configuration fields.
pub(crate) fn legacy_component_key(prefix: &str, field: Option<&str>) -> String {
    let discriminator = field.map_or_else(|| "default".to_owned(), digest);
    let candidate = format!("{prefix}.{discriminator}");
    if valid_component_key(&candidate) {
        candidate
    } else {
        format!("legacy.component.{}", digest(&candidate))
    }
}

fn digest(value: &str) -> String {
    let mut result = String::with_capacity(64);
    for byte in Sha256::digest(value.as_bytes()) {
        let _ = write!(result, "{byte:02x}");
    }
    result
}
