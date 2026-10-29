// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Plugin report mirroring by the shared exporter runtime diagnostics recorder.

use super::*;
use crate::api::runtime::{NemoRelayContextState, global_context};
use crate::plugin::{
    PluginConfig, active_runtime_diagnostics_snapshot, test_close_plugin_host,
    test_initialize_plugin_host_exact,
};

fn reset_runtime() {
    let _ = test_close_plugin_host();
    crate::shared_runtime::reset_runtime_owner_for_tests();
    let context = global_context();
    *context.write().unwrap() = NemoRelayContextState::new();
}

#[test]
fn recorder_mirrors_into_the_active_plugin_report_only_with_a_plugin_field() {
    let _guard = crate::observability::test_mutex().lock().unwrap();
    reset_runtime();
    futures::executor::block_on(test_initialize_plugin_host_exact(PluginConfig::default()))
        .unwrap();

    let local = ExporterRuntimeDiagnosticsRecorder::new(None);
    assert_eq!(
        local.record("atof.stream_delivery_failed", "local".into(), 1),
        1
    );
    assert_eq!(
        local
            .snapshot()
            .get("atof.stream_delivery_failed")
            .unwrap()
            .count,
        1
    );
    assert!(active_runtime_diagnostics_snapshot().is_empty());

    let managed = ExporterRuntimeDiagnosticsRecorder::new(Some("atof.sinks[0].url".into()));
    assert_eq!(
        managed.record("atof.stream_delivery_failed", "managed".into(), 2),
        2
    );
    let mirrored = active_runtime_diagnostics_snapshot();
    assert_eq!(
        mirrored
            .iter()
            .map(|entry| (entry.code.as_str(), entry.message.as_str(), entry.count))
            .collect::<Vec<_>>(),
        vec![("atof.stream_delivery_failed", "managed", 2)]
    );

    test_close_plugin_host().unwrap();
}
