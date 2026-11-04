// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Preconditions for SDK tests that resolve plugin configuration.

pub fn require_isolated_plugin_config() {
    assert_eq!(
        std::env::var("NEMO_RELAY_TEST_SKIP_IMPLICIT_CONFIG")
            .ok()
            .as_deref(),
        Some("1"),
        "Run with NEMO_RELAY_TEST_SKIP_IMPLICIT_CONFIG=1 and --features __skip-implicit-config; SDK tests must not read ambient plugin configuration",
    );
}
