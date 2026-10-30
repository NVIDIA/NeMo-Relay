// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay::{
    diagnostics::{DiagnosticQuery, DiagnosticSnapshot, DiagnosticsSettings, embedded},
    plugin::dynamic::{PluginHostConfig, initialize, validate},
};

let config = PluginHostConfig {
    diagnostics: Some(DiagnosticsSettings {
        enabled: true,
        ..Default::default()
    }),
    ..Default::default()
};
validate(config.clone(), None)?;
let mut activation = initialize(config, None).await?;
let reader = embedded::reader()?;
let page = reader.snapshot(&DiagnosticQuery::default())?;
let snapshot: DiagnosticSnapshot = serde_json::from_str(page.as_str())?;
let failures: u64 = snapshot.lifetime.totals.failures_total.value.get();
drop(page); // Return this page's adapter permit before another capture.
let info = reader.info()?;
let reserved = info.binding_reserved_bytes;
let logging_coverage = info.logging_coverage;
activation.close()?;
let final_page = reader.snapshot_json("{}")?; // Same process-owned service.
