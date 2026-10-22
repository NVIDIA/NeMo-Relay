// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Emit public schema documents from the authoritative Rust types.
//! Run: cargo run -p nemo-relay-types --features schema --example diagnostics_schema

use nemo_relay_types::diagnostics::{DiagnosticDiscovery, DiagnosticQuery, DiagnosticSnapshot};

fn main() {
    let schemas = serde_json::json!({
        "snapshot": schemars::schema_for!(DiagnosticSnapshot),
        "discovery": schemars::schema_for!(DiagnosticDiscovery),
        "query": schemars::schema_for!(DiagnosticQuery),
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&schemas).expect("schemas serialize")
    );
}
