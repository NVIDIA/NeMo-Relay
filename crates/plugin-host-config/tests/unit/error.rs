// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::sanitize_parser_reason;

#[test]
fn parser_reason_preserves_schema_identifiers_but_redacts_values() {
    let reason =
        "unknown field `enabled`, expected `manifest` or `config`; invalid value \"secret\"";

    let sanitized = sanitize_parser_reason(reason);

    assert!(sanitized.contains("`enabled`"));
    assert!(sanitized.contains("`manifest`"));
    assert!(sanitized.contains("`config`"));
    assert!(!sanitized.contains("secret"));
    assert!(sanitized.contains("\"<redacted>\""));
}
