// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use chrono::{DateTime, Utc};
use nemo_relay_types::api::resource_metrics::{
    ResourceMeasurement, ResourceMeasurementUnit, ResourceMetricsSnapshot,
};

use crate::error::{FlowError, Result};

#[cfg(windows)]
use super::collector::owned_process_tree_target_with_job_handle;
use super::collector::{
    self, CollectedSnapshot, CollectionTarget, current_process_target, owned_process_tree_target,
};
use super::config::ResourceMetricsConfig;
use super::file::ResourceMetricsFile;

static GLOBAL_MANAGER: Mutex<Option<Arc<ResourceMetricsManager>>> = Mutex::new(None);

#[derive(Debug, Clone, Copy)]
pub(crate) enum SamplingSeries {
    OnDemand,
    Polling,
    AgentEvent,
}

#[derive(Debug, Clone)]
struct SamplingBaseline {
    completed_at: DateTime<Utc>,
    total_cpu_time: u64,
    target_generation: u64,
}

#[derive(Debug)]
struct ManagerState {
    application_target: CollectionTarget,
    target: CollectionTarget,
    target_generation: u64,
    active_runtime_generation: Option<u64>,
    on_demand_baseline: Option<SamplingBaseline>,
    polling_baseline: Option<SamplingBaseline>,
    agent_event_baseline: Option<SamplingBaseline>,
    latest: Option<ResourceMetricsSnapshot>,
    history: VecDeque<ResourceMetricsSnapshot>,
    retained_snapshots: usize,
}

pub(crate) struct ResourceMetricsManager {
    state: Mutex<ManagerState>,
}

impl ResourceMetricsManager {
    fn new() -> Result<Self> {
        let application_target = current_process_target()?;
        Ok(Self {
            state: Mutex::new(ManagerState {
                application_target: application_target.clone(),
                target: application_target,
                target_generation: 1,
                active_runtime_generation: None,
                on_demand_baseline: None,
                polling_baseline: None,
                agent_event_baseline: None,
                latest: None,
                history: VecDeque::new(),
                retained_snapshots: 1,
            }),
        })
    }

    fn activate(&self, target: CollectionTarget, config: &ResourceMetricsConfig) -> Result<u64> {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        if state.active_runtime_generation.is_some() {
            return Err(FlowError::AlreadyExists(
                "resource metrics runtime is already configured".into(),
            ));
        }
        state.target_generation = state.target_generation.saturating_add(1);
        let runtime_generation = state.target_generation;
        state.target = target;
        state.active_runtime_generation = Some(runtime_generation);
        state.on_demand_baseline = None;
        state.polling_baseline = None;
        state.agent_event_baseline = None;
        state.latest = None;
        state.history.clear();
        state.retained_snapshots = config.polling.retained_snapshots.max(1);
        Ok(runtime_generation)
    }

    fn deactivate(&self, runtime_generation: u64) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        if state.active_runtime_generation == Some(runtime_generation) {
            state.target_generation = state.target_generation.saturating_add(1);
            state.target = state.application_target.clone();
            state.active_runtime_generation = None;
            state.on_demand_baseline = None;
            state.polling_baseline = None;
            state.agent_event_baseline = None;
        }
    }

    pub(crate) fn collect(&self, series: SamplingSeries) -> ResourceMetricsSnapshot {
        let (target, target_generation) = {
            let state = self.state.lock().unwrap_or_else(|error| error.into_inner());
            (state.target.clone(), state.target_generation)
        };
        let collected = collector::collect(&target);
        self.finish_collection(series, target_generation, collected)
    }

    fn finish_collection(
        &self,
        series: SamplingSeries,
        target_generation: u64,
        mut collected: CollectedSnapshot,
    ) -> ResourceMetricsSnapshot {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        if state.target_generation != target_generation {
            return invalidated_snapshot(collected.snapshot);
        }
        if collected.successful {
            derive_cpu_rate(
                &mut collected.snapshot,
                collected.completed_at,
                baseline_for_series(&mut state, series),
                target_generation,
            );
        }
        collected.snapshot
    }

    fn record_poll(&self, runtime_generation: u64, snapshot: ResourceMetricsSnapshot) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        if state.active_runtime_generation != Some(runtime_generation)
            || snapshot.active_process_count.value().is_none()
        {
            return;
        }
        state.latest = Some(snapshot.clone());
        state.history.push_back(snapshot);
        while state.history.len() > state.retained_snapshots {
            state.history.pop_front();
        }
    }

    pub(crate) fn latest(&self) -> Option<ResourceMetricsSnapshot> {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .latest
            .clone()
    }

    pub(crate) fn history(&self) -> Vec<ResourceMetricsSnapshot> {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .history
            .iter()
            .cloned()
            .collect()
    }
}

fn invalidated_snapshot(snapshot: ResourceMetricsSnapshot) -> ResourceMetricsSnapshot {
    ResourceMetricsSnapshot {
        operating_system: snapshot.operating_system,
        cpu_user_time: unavailable_measurement(),
        cpu_system_time: unavailable_measurement(),
        cpu_total_time: unavailable_measurement(),
        cpu_consumption_rate: unavailable_measurement(),
        cpu_throttled_time: unavailable_measurement(),
        effective_cpu_limit: unavailable_measurement(),
        cpu_some_pressure_stall_time: unavailable_measurement(),
        cpu_full_pressure_stall_time: unavailable_measurement(),
        resident_memory: unavailable_measurement(),
        private_memory: unavailable_measurement(),
        physical_footprint: unavailable_measurement(),
        virtual_memory: unavailable_measurement(),
        peak_resident_memory: unavailable_measurement(),
        memory_limit: unavailable_measurement(),
        environment_accounted_memory: unavailable_measurement(),
        memory_some_pressure_stall_time: unavailable_measurement(),
        memory_full_pressure_stall_time: unavailable_measurement(),
        out_of_memory_event_count: unavailable_measurement(),
        active_process_count: unavailable_measurement(),
        descendant_process_count: unavailable_measurement(),
        thread_count: unavailable_measurement(),
        lifetime_process_creation_count: unavailable_measurement(),
        open_file_descriptor_count: unavailable_measurement(),
        windows_handle_count: unavailable_measurement(),
        resource_limit_events: Vec::new(),
        accelerator_devices: Vec::new(),
        accelerator_processes: Vec::new(),
    }
}

fn unavailable_measurement<T>() -> ResourceMeasurement<T> {
    ResourceMeasurement::unavailable(Utc::now())
}

fn baseline_for_series(
    state: &mut ManagerState,
    series: SamplingSeries,
) -> &mut Option<SamplingBaseline> {
    match series {
        SamplingSeries::OnDemand => &mut state.on_demand_baseline,
        SamplingSeries::Polling => &mut state.polling_baseline,
        SamplingSeries::AgentEvent => &mut state.agent_event_baseline,
    }
}

fn derive_cpu_rate(
    snapshot: &mut ResourceMetricsSnapshot,
    completed_at: DateTime<Utc>,
    baseline: &mut Option<SamplingBaseline>,
    target_generation: u64,
) {
    let Some(total_cpu_time) = snapshot.cpu_total_time.value().copied() else {
        *baseline = None;
        return;
    };
    let next = SamplingBaseline {
        completed_at,
        total_cpu_time,
        target_generation,
    };
    let Some(previous) = baseline.replace(next) else {
        return;
    };
    let Some(interval_microseconds) = completed_at
        .signed_duration_since(previous.completed_at)
        .num_microseconds()
    else {
        return;
    };
    if previous.target_generation != target_generation
        || interval_microseconds <= 0
        || total_cpu_time < previous.total_cpu_time
    {
        return;
    }
    let interval_seconds = interval_microseconds as f64 / 1_000_000.0;
    let rate = (total_cpu_time - previous.total_cpu_time) as f64 / interval_seconds;
    let timestamp = Utc::now();
    let unit = match snapshot.cpu_total_time.unit {
        Some(ResourceMeasurementUnit::Nanoseconds) => ResourceMeasurementUnit::NanosecondsPerSecond,
        Some(ResourceMeasurementUnit::ClockTicks) => ResourceMeasurementUnit::ClockTicksPerSecond,
        Some(ResourceMeasurementUnit::HundredNanosecondIntervals) => {
            ResourceMeasurementUnit::HundredNanosecondIntervalsPerSecond
        }
        _ => return,
    };
    snapshot.cpu_consumption_rate = ResourceMeasurement::available(timestamp, rate, unit);
}

pub(crate) fn global_manager() -> Result<Arc<ResourceMetricsManager>> {
    let mut manager = GLOBAL_MANAGER
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if let Some(manager) = manager.as_ref() {
        return Ok(Arc::clone(manager));
    }
    let created = Arc::new(ResourceMetricsManager::new()?);
    *manager = Some(Arc::clone(&created));
    Ok(created)
}

/// Owns the optional resource metrics polling thread and its file output.
pub struct ResourceMetricsRuntime {
    manager: Arc<ResourceMetricsManager>,
    runtime_generation: u64,
    stop: Option<Sender<()>>,
    polling_thread: Option<JoinHandle<()>>,
}

impl ResourceMetricsRuntime {
    pub(crate) fn configure_current_process(config: ResourceMetricsConfig) -> Result<Self> {
        Self::configure_target(config, current_process_target()?)
    }

    pub(crate) fn configure_owned_target(
        config: ResourceMetricsConfig,
        process_id: u32,
    ) -> Result<Self> {
        Self::configure_target(config, owned_process_tree_target(process_id)?)
    }

    #[cfg(windows)]
    pub(crate) fn configure_owned_target_with_job_handle(
        config: ResourceMetricsConfig,
        process_id: u32,
        job_handle: isize,
    ) -> Result<Self> {
        Self::configure_target(
            config,
            owned_process_tree_target_with_job_handle(process_id, job_handle)?,
        )
    }

    fn configure_target(config: ResourceMetricsConfig, target: CollectionTarget) -> Result<Self> {
        config.validate()?;
        let file = config
            .file
            .enabled
            .then(|| ResourceMetricsFile::open(&config.file))
            .transpose()?;
        let manager = global_manager()?;
        let runtime_generation = manager.activate(target, &config)?;
        let (stop, polling_thread) = if config.polling.enabled {
            let (stop, receiver) = mpsc::channel();
            let manager_for_thread = Arc::clone(&manager);
            let interval = Duration::from_millis(config.polling.interval_millis);
            let polling_thread = thread::Builder::new()
                .name("nemo-relay-resource-metrics".into())
                .spawn(move || {
                    polling_loop(
                        manager_for_thread,
                        runtime_generation,
                        receiver,
                        interval,
                        file,
                    );
                })
                .map_err(|error| {
                    manager.deactivate(runtime_generation);
                    FlowError::Internal(format!(
                        "failed to start resource metrics polling thread: {error}"
                    ))
                })?;
            (Some(stop), Some(polling_thread))
        } else {
            drop(file);
            (None, None)
        };
        Ok(Self {
            manager,
            runtime_generation,
            stop,
            polling_thread,
        })
    }

    /// Stop polling, flush file output, and release process-global configuration ownership.
    pub fn shutdown(self) {
        drop(self);
    }
}

impl Drop for ResourceMetricsRuntime {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(polling_thread) = self.polling_thread.take()
            && polling_thread.join().is_err()
        {
            log::warn!(
                target: "nemo_relay.resource_metrics",
                event = "resource_metrics_polling_join_failed",
                error_kind = "thread_panic";
                "Resource metrics polling thread panicked during shutdown"
            );
        }
        self.manager.deactivate(self.runtime_generation);
    }
}

fn polling_loop(
    manager: Arc<ResourceMetricsManager>,
    runtime_generation: u64,
    stop: mpsc::Receiver<()>,
    interval: Duration,
    mut file: Option<ResourceMetricsFile>,
) {
    loop {
        let snapshot = manager.collect(SamplingSeries::Polling);
        if snapshot.active_process_count.value().is_some() {
            manager.record_poll(runtime_generation, snapshot.clone());
            if let Some(file) = file.as_mut()
                && let Err(error) = file.append(&snapshot)
            {
                log::warn!(
                    target: "nemo_relay.resource_metrics",
                    event = "resource_metrics_file_write_failed",
                    error_kind = "io";
                    "Resource metrics file write failed: {error}"
                );
            }
        }
        match stop.recv_timeout(interval) {
            Ok(()) | Err(RecvTimeoutError::Disconnected) => break,
            Err(RecvTimeoutError::Timeout) => {}
        }
    }
}
