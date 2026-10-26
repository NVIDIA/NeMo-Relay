// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Run with `cargo run -p nemo-relay --example embedded_diagnostics`.
use nemo_relay::{
    diagnostics::{DiagnosticQuery, DiagnosticSnapshot, DiagnosticsSettings, embedded},
    plugin::dynamic::{PluginHostConfig, initialize, validate},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = PluginHostConfig {
        diagnostics: Some(DiagnosticsSettings {
            enabled: true,
            ..Default::default()
        }),
        ..Default::default()
    };
    validate(config.clone(), None)?; // Does not start the diagnostics service.
    let mut activation = initialize(config, None).await?;
    let reader = embedded::reader()?;
    let page = reader.snapshot(&DiagnosticQuery::default())?;
    let first: DiagnosticSnapshot = serde_json::from_str(page.as_str())?;
    assert_eq!(reader.info()?.binding_copy_concurrency, 2);
    drop(page); // Return this page's independent adapter permit.

    let mut query = DiagnosticQuery {
        max_components: Some(1),
        max_events: Some(1),
        ..Default::default()
    };
    let mut complete = false;
    for _ in 0..32 {
        let page = reader.snapshot(&query)?;
        let snapshot: DiagnosticSnapshot = serde_json::from_str(page.as_str())?;
        // Consume the selected component/event rows within this trust boundary.
        query.continuation = snapshot.continuation;
        if query.continuation.is_none() {
            complete = true;
            break;
        }
    }
    if !complete {
        return Err("The caller's page budget was exhausted".into());
    }
    assert_eq!(
        reader
            .snapshot_json(r#"{"unknown":true}"#)
            .err()
            .unwrap()
            .reason(),
        "invalid_query"
    );
    activation.close()?;
    let page = reader.snapshot_json("{}")?;
    let final_snapshot: DiagnosticSnapshot = serde_json::from_str(page.as_str())?;
    assert_eq!(final_snapshot.source, first.source);
    Ok(())
}
