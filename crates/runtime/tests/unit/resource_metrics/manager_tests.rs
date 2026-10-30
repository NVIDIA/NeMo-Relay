// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::api::resource_metrics::{CpuMetrics, DataUnit, DurationUnit};
use nemo_relay_diagnostics::{DiagnosticMemoryPage, DiagnosticQuery, Diagnostics};

#[cfg(any(target_os = "linux", target_os = "macos", windows))]
fn isolated_resource_manager_test() -> bool {
    // Isolate process-global ownership markers and bound regressions that could deadlock.
    const CHILD_ENV: &str = "NEMO_RELAY_TEST_RESOURCE_MANAGER_OWNER_CHILD";
    if std::env::var_os(CHILD_ENV).is_none() {
        let current_thread = std::thread::current();
        let test_name = current_thread.name().expect("named test thread");
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", test_name, "--nocapture"])
            .env(CHILD_ENV, "1")
            .spawn()
            .expect("resource manager test process should start");
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            if let Some(status) = child.try_wait().unwrap() {
                assert!(status.success(), "resource manager test failed: {status}");
                return false;
            }
            if Instant::now() >= deadline {
                child.kill().unwrap();
                child.wait().unwrap();
                panic!("resource manager test did not finish within 20 seconds");
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    true
}

#[test]
#[cfg(any(target_os = "linux", target_os = "macos", windows))]
fn manager_installation_reconciles_inherited_inspection_ownership() {
    use nemo_relay_types::diagnostics::ResourceMemoryAvailability;

    if !isolated_resource_manager_test() {
        return;
    }

    let pid = std::process::id();
    let inherited_pid = pid.wrapping_add(1).max(1);
    assert_eq!(
        latest_memory_view(Some(pid)).availability,
        ResourceMemoryAvailability::Disabled
    );
    assert!(GLOBAL_MANAGER.lock().unwrap().is_none());
    assert_eq!(MANAGER_OWNER_PID.load(Ordering::Acquire), pid);

    // Simulate a parent that inspected before activation, then forked while idle.
    MANAGER_OWNER_PID.store(inherited_pid, Ordering::Release);
    assert_eq!(
        latest_memory_view(Some(pid)).availability,
        ResourceMemoryAvailability::Invalidated
    );
    let manager = global_manager().unwrap();
    let target = current_process_target(ResourceMeasurementScope::ApplicationProcess).unwrap();
    let config = ResourceMetricsConfig {
        measurement_scope: ResourceMetricsMeasurementScope::ApplicationProcess,
        ..Default::default()
    };
    let generation = manager.activate(target, config).unwrap();
    assert!(
        manager
            .collect_blocking(SamplingSeries::OnDemand)
            .unwrap()
            .memory
            .is_some()
    );
    assert_eq!(
        latest_memory_view(Some(pid)).availability,
        ResourceMemoryAvailability::Available
    );
    assert_eq!(MANAGER_OWNER_PID.load(Ordering::Acquire), pid);
    assert!(Arc::ptr_eq(&manager, &global_manager().unwrap()));
    manager.deactivate(generation);

    // Neither inspection nor initialization may wait on an inherited locked mutex.
    *GLOBAL_MANAGER.lock().unwrap() = None;
    MANAGER_OWNER_PID.store(inherited_pid, Ordering::Release);
    let _locked = GLOBAL_MANAGER.lock().unwrap();
    ACTIVE_OWNER_PID.store(inherited_pid, Ordering::Release);
    assert!(global_manager().is_err());
    assert_eq!(
        latest_memory_view(Some(pid)).availability,
        ResourceMemoryAvailability::Invalidated
    );
    ACTIVE_OWNER_PID.store(0, Ordering::Release);
    assert!(global_manager().is_err());
    assert_eq!(MANAGER_OWNER_PID.load(Ordering::Acquire), inherited_pid);
    assert_eq!(
        latest_memory_view(Some(pid)).availability,
        ResourceMemoryAvailability::Invalidated
    );
}

#[test]
#[cfg(any(target_os = "linux", target_os = "macos", windows))]
fn resource_memory_pages_preserve_budgets_and_samples() {
    use nemo_relay_diagnostics::*;
    if !isolated_resource_manager_test() {
        return;
    }
    let diagnostics = Diagnostics::from_settings(
        Diagnostics::default().source().clone(),
        DiagnosticsSettings {
            enabled: true,
            limits: DiagnosticLimits {
                max_queries: 1,
                ..Default::default()
            },
        },
    )
    .unwrap();
    let code = BuiltinDiagnosticCode::Success(SuccessCode::SinkInitialized);
    let recorder = diagnostics
        .register_component(ComponentRegistration::new("memory.policy", &[code]))
        .unwrap();
    recorder
        .record_without_projection_lazy(|| {
            Observation::new(
                Classification::success(SuccessCode::SinkInitialized),
                "retained",
                std::num::NonZeroU64::MIN,
            )
        })
        .unwrap();
    let query = DiagnosticQuery {
        max_components: Some(0),
        max_events: Some(0),
        ..Default::default()
    };
    let baseline_bytes = diagnostics
        .reader()
        .inspect_memory(&query)
        .unwrap()
        .as_bytes()
        .len();
    assert!(GLOBAL_MANAGER.lock().unwrap().is_none());
    let before = diagnostics
        .snapshot(&DiagnosticQuery::default())
        .unwrap()
        .to_snapshot()
        .unwrap();
    let manager = global_manager().unwrap();
    let target = current_process_target(ResourceMeasurementScope::ApplicationProcess).unwrap();
    let config = ResourceMetricsConfig {
        measurement_scope: ResourceMetricsMeasurementScope::ApplicationProcess,
        ..Default::default()
    };
    let generation = manager.activate(target.clone(), config.clone()).unwrap();
    assert_eq!(
        diagnostics
            .reader()
            .inspect_memory(&query)
            .unwrap()
            .resource()
            .availability,
        ResourceMemoryAvailability::NotYetSampled
    );
    manager.collect_blocking(SamplingSeries::OnDemand).unwrap();
    let full = diagnostics.reader().inspect_memory(&query).unwrap();
    assert_eq!(
        full.resource().availability,
        ResourceMemoryAvailability::Available
    );
    let sample = full.resource().sample.unwrap();
    let full_page = full.to_memory_page().unwrap();
    let budget = baseline_bytes + 128;
    assert!(full.as_bytes().len() > budget);
    drop(full);
    assert_bounded_memory_page(&diagnostics, &query, budget, &full_page);
    assert_eq!(diagnostics.memory_usage().query_workspaces, 0);
    assert_eq!(
        latest_memory_view(Some(std::process::id())).sample,
        Some(sample)
    );
    assert_eq!(
        diagnostics
            .reader()
            .inspect_memory(&DiagnosticQuery {
                max_response_bytes: Some(1),
                ..query.clone()
            })
            .unwrap_err(),
        DiagnosticError::SnapshotCapacity
    );
    assert_eq!(diagnostics.memory_usage().query_workspaces, 0);
    assert_eq!(
        diagnostics
            .reader()
            .inspect_memory(&query)
            .unwrap()
            .resource()
            .sample,
        Some(sample)
    );
    let after = diagnostics
        .snapshot(&DiagnosticQuery::default())
        .unwrap()
        .to_snapshot()
        .unwrap();
    assert_eq!(before.sequence, after.sequence);
    assert_eq!(before.events, after.events);
    assert_eq!(before.lifetime.totals, after.lifetime.totals);
    manager.deactivate(generation);
    assert_eq!(
        latest_memory_view(Some(std::process::id())).availability,
        ResourceMemoryAvailability::Disabled
    );
}

#[cfg(any(target_os = "linux", target_os = "macos", windows))]
fn assert_bounded_memory_page(
    diagnostics: &Diagnostics,
    query: &DiagnosticQuery,
    budget: usize,
    full_page: &DiagnosticMemoryPage,
) {
    use nemo_relay_diagnostics::ResourceMemoryAvailability;
    let limited = diagnostics
        .reader()
        .inspect_memory(&DiagnosticQuery {
            max_response_bytes: Some(budget as u32),
            ..query.clone()
        })
        .unwrap();
    assert!(limited.as_bytes().len() <= budget);
    let page = limited.to_memory_page().unwrap();
    assert_eq!(
        page.resource.availability,
        ResourceMemoryAvailability::CapacityUnavailable
    );
    assert!(page.resource.sample.is_none());
    assert_eq!(
        page.aggregate.reserved_bytes,
        full_page.aggregate.reserved_bytes
    );
    assert_eq!(page.aggregate.events, 1);
    assert_eq!(
        page.component_occupied_bytes,
        full_page.component_occupied_bytes
    );
    assert_eq!(
        page.component_occupied_bytes.get() + page.shared_or_unassigned_reserved_bytes.get(),
        page.aggregate.reserved_bytes as u64
    );
    drop(limited);
}

#[test]
#[cfg(any(target_os = "linux", target_os = "macos", windows))]
fn resource_memory_pages_respect_inspection_policy_and_pending_targets() {
    use nemo_relay_diagnostics::*;
    if !isolated_resource_manager_test() {
        return;
    }
    let diagnostics = Diagnostics::from_settings(
        Diagnostics::default().source().clone(),
        DiagnosticsSettings {
            enabled: true,
            ..Default::default()
        },
    )
    .unwrap();
    let query = DiagnosticQuery::default();
    let manager = global_manager().unwrap();
    let target = current_process_target(ResourceMeasurementScope::ApplicationProcess).unwrap();
    let mut config = ResourceMetricsConfig {
        measurement_scope: ResourceMetricsMeasurementScope::ApplicationProcess,
        ..Default::default()
    };
    for disable_collection in [false, true] {
        config.memory_inspection.enabled = disable_collection;
        config.memory.enabled = !disable_collection;
        let generation = manager.activate(target.clone(), config.clone()).unwrap();
        assert_eq!(
            diagnostics
                .reader()
                .inspect_memory(&query)
                .unwrap()
                .resource()
                .availability,
            ResourceMemoryAvailability::Disabled
        );
        assert!(
            latest_memory_view(Some(std::process::id()))
                .sample
                .is_none()
        );
        manager.deactivate(generation);
    }
    config.memory.enabled = true;
    config.memory_inspection.enabled = true;
    config.measurement_scope = ResourceMetricsMeasurementScope::ProcessTree;
    let launch = manager.prepare_cli_owned_target().unwrap();
    let generation = manager
        .activate(
            current_process_target(ResourceMeasurementScope::ProcessTree).unwrap(),
            config,
        )
        .unwrap();
    assert_eq!(
        diagnostics
            .reader()
            .inspect_memory(&query)
            .unwrap()
            .resource()
            .availability,
        ResourceMemoryAvailability::AwaitingTarget
    );
    manager.deactivate(generation);
    drop(launch);
}
