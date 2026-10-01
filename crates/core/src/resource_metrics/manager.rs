// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex, RwLock, Weak};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use chrono::Utc;
use nemo_relay_types::api::resource_metrics::{
    BandwidthUnit, CpuUnit, ResourceMeasurement, ResourceMeasurementScope, ResourceMetricsSnapshot,
};

use super::network::NetworkSampler;
use super::units::convert_snapshot;
use crate::error::{FlowError, Result};

#[cfg(not(windows))]
use super::collector::owned_process_tree_target;
#[cfg(windows)]
use super::collector::owned_process_tree_target_with_job_handle;
use super::collector::{
    self, CollectionTarget, GlobalCpuSampler, ProcessCpuSample, ProcessIoSample,
    current_process_target,
};
use crate::api::runtime::scope_stack::ScopeStack;
use crate::plugins::resource_metrics::config::{
    ResourceMetricsConfig, ResourceMetricsMeasurementScope,
};

static GLOBAL_MANAGER: Mutex<Option<Arc<ResourceMetricsManager>>> = Mutex::new(None);
static ACTIVE_OWNER_PID: AtomicU32 = AtomicU32::new(0);

pub(crate) fn runtime_inherited_across_fork() -> bool {
    let owner = ACTIVE_OWNER_PID.load(Ordering::Acquire);
    owner != 0 && owner != std::process::id()
}

fn ensure_process_owner() -> Result<()> {
    if runtime_inherited_across_fork() {
        return Err(FlowError::InvalidArgument(
            "resource_metrics runtime was inherited across fork; exec and activate it in the child before collecting"
                .into(),
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum SamplingSeries {
    OnDemand,
    Polling,
}

#[derive(Debug, Clone)]
struct SamplingBaseline {
    sampled_at: Instant,
    process_cpu_times: HashMap<(u32, u64), u64>,
    target_generation: u64,
}

struct DiskBaseline {
    sampled_at: Instant,
    counters: HashMap<(u32, u64), (Option<u64>, Option<u64>)>,
    target_generation: u64,
}

struct ManagerState {
    application_target: CollectionTarget,
    target: CollectionTarget,
    target_generation: u64,
    active_runtime_generation: Option<u64>,
    config: Option<ResourceMetricsConfig>,
    on_demand_baseline: Option<SamplingBaseline>,
    polling_baseline: Option<SamplingBaseline>,
    on_demand_disk_baseline: Option<DiskBaseline>,
    polling_disk_baseline: Option<DiskBaseline>,
    on_demand_network_sampler: Option<NetworkSampler>,
    polling_network_sampler: Option<NetworkSampler>,
    on_demand_global_cpu_sampler: Option<GlobalCpuSampler>,
    polling_global_cpu_sampler: Option<GlobalCpuSampler>,
    on_demand_sampling_state: Arc<Mutex<collector::SamplingState>>,
    polling_sampling_state: Arc<Mutex<collector::SamplingState>>,
    cli_launch_prepared: bool,
    awaiting_owned_target: bool,
}

pub(crate) struct ResourceMetricsManager {
    owner_pid: u32,
    state: Mutex<ManagerState>,
}

impl ResourceMetricsManager {
    fn new() -> Result<Self> {
        let application_target = current_process_target(ResourceMeasurementScope::ProcessTree)?;
        Ok(Self {
            owner_pid: std::process::id(),
            state: Mutex::new(ManagerState {
                application_target: application_target.clone(),
                target: application_target,
                target_generation: 1,
                active_runtime_generation: None,
                config: None,
                on_demand_baseline: None,
                polling_baseline: None,
                on_demand_disk_baseline: None,
                polling_disk_baseline: None,
                on_demand_network_sampler: None,
                polling_network_sampler: None,
                on_demand_global_cpu_sampler: None,
                polling_global_cpu_sampler: None,
                on_demand_sampling_state: Arc::new(Mutex::new(collector::SamplingState::default())),
                polling_sampling_state: Arc::new(Mutex::new(collector::SamplingState::default())),
                cli_launch_prepared: false,
                awaiting_owned_target: false,
            }),
        })
    }

    fn activate(&self, target: CollectionTarget, config: ResourceMetricsConfig) -> Result<u64> {
        ensure_process_owner()?;
        if self.owner_pid != std::process::id() {
            return Err(FlowError::InvalidArgument(
                "resource_metrics manager belongs to a different process".into(),
            ));
        }
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        if state.active_runtime_generation.is_some() {
            return Err(FlowError::AlreadyExists(
                "resource_metrics plugin is already active".into(),
            ));
        }
        state.target_generation = state.target_generation.saturating_add(1);
        let runtime_generation = state.target_generation;
        state.target = target;
        state.awaiting_owned_target = state.cli_launch_prepared
            && config.measurement_scope == ResourceMetricsMeasurementScope::ProcessTree;
        state.config = Some(config);
        state.active_runtime_generation = Some(runtime_generation);
        state.on_demand_network_sampler = None;
        state.polling_network_sampler = None;
        state.on_demand_baseline = None;
        state.polling_baseline = None;
        state.on_demand_disk_baseline = None;
        state.polling_disk_baseline = None;
        state.on_demand_global_cpu_sampler = None;
        state.polling_global_cpu_sampler = None;
        state.on_demand_sampling_state = Arc::new(Mutex::new(collector::SamplingState::default()));
        state.polling_sampling_state = Arc::new(Mutex::new(collector::SamplingState::default()));
        ACTIVE_OWNER_PID.store(self.owner_pid, Ordering::Release);
        Ok(runtime_generation)
    }

    fn deactivate(&self, runtime_generation: u64) {
        if self.owner_pid != std::process::id() {
            return;
        }
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        if state.active_runtime_generation == Some(runtime_generation) {
            state.target_generation = state.target_generation.saturating_add(1);
            state.target = state.application_target.clone();
            state.active_runtime_generation = None;
            state.config = None;
            state.on_demand_network_sampler = None;
            state.polling_network_sampler = None;
            state.on_demand_baseline = None;
            state.polling_baseline = None;
            state.on_demand_disk_baseline = None;
            state.polling_disk_baseline = None;
            state.on_demand_global_cpu_sampler = None;
            state.polling_global_cpu_sampler = None;
            state.on_demand_sampling_state =
                Arc::new(Mutex::new(collector::SamplingState::default()));
            state.polling_sampling_state =
                Arc::new(Mutex::new(collector::SamplingState::default()));
            state.awaiting_owned_target = false;
            ACTIVE_OWNER_PID.store(0, Ordering::Release);
        }
    }

    fn target_owned_process_tree(&self, target: CollectionTarget) -> Option<u64> {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        state.active_runtime_generation?;
        if state.config.as_ref().is_none_or(|config| {
            config.measurement_scope != ResourceMetricsMeasurementScope::ProcessTree
        }) {
            return None;
        }
        state.target_generation = state.target_generation.saturating_add(1);
        state.target = target;
        state.awaiting_owned_target = false;
        state.on_demand_baseline = None;
        state.polling_baseline = None;
        state.on_demand_disk_baseline = None;
        state.polling_disk_baseline = None;
        state.on_demand_global_cpu_sampler = None;
        state.polling_global_cpu_sampler = None;
        state.on_demand_sampling_state = Arc::new(Mutex::new(collector::SamplingState::default()));
        state.polling_sampling_state = Arc::new(Mutex::new(collector::SamplingState::default()));
        Some(state.target_generation)
    }

    fn restore_application_target(&self, generation: u64) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        if state.target_generation == generation {
            state.target_generation = state.target_generation.saturating_add(1);
            state.target = state.application_target.clone();
            state.awaiting_owned_target = state.cli_launch_prepared;
            state.on_demand_baseline = None;
            state.polling_baseline = None;
            state.on_demand_disk_baseline = None;
            state.polling_disk_baseline = None;
            state.on_demand_global_cpu_sampler = None;
            state.polling_global_cpu_sampler = None;
            state.on_demand_sampling_state =
                Arc::new(Mutex::new(collector::SamplingState::default()));
            state.polling_sampling_state =
                Arc::new(Mutex::new(collector::SamplingState::default()));
        }
    }

    pub(crate) fn collect_blocking(
        &self,
        series: SamplingSeries,
    ) -> Result<ResourceMetricsSnapshot> {
        ensure_process_owner()?;
        if self.owner_pid != std::process::id() {
            return Err(FlowError::InvalidArgument(
                "resource_metrics manager belongs to a different process".into(),
            ));
        }
        let (target, config, target_generation, sampling_state) = {
            let state = self.state.lock().unwrap_or_else(|error| error.into_inner());
            let config = state.config.clone().ok_or_else(|| {
                FlowError::InvalidArgument(
                    "resource_metrics.collect() requires an active resource_metrics component"
                        .into(),
                )
            })?;
            if state.awaiting_owned_target {
                return Err(FlowError::InvalidArgument(
                    "resource_metrics is waiting for the CLI-owned process tree target".into(),
                ));
            }
            let sampling_state = match series {
                SamplingSeries::OnDemand => Arc::clone(&state.on_demand_sampling_state),
                SamplingSeries::Polling => Arc::clone(&state.polling_sampling_state),
            };
            (
                state.target.clone(),
                config,
                state.target_generation,
                sampling_state,
            )
        };
        let collected = collector::collect_with_state(
            &target,
            &config,
            &mut sampling_state
                .lock()
                .unwrap_or_else(|error| error.into_inner()),
        );
        self.finish_collection(collected, &config, target_generation, series)
    }

    fn finish_collection(
        &self,
        mut collected: collector::CollectedSnapshot,
        config: &ResourceMetricsConfig,
        target_generation: u64,
        series: SamplingSeries,
    ) -> Result<ResourceMetricsSnapshot> {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        if state.target_generation != target_generation {
            let mut snapshot = invalidated_snapshot(collected.snapshot);
            if state.config.as_ref() == Some(config) && config.network.enabled {
                let sampler = match series {
                    SamplingSeries::OnDemand => &mut state.on_demand_network_sampler,
                    SamplingSeries::Polling => &mut state.polling_network_sampler,
                };
                snapshot.network = Some(
                    sampler
                        .get_or_insert_with(NetworkSampler::default)
                        .sample(&config.network.interfaces),
                );
            }
            convert_snapshot(&mut snapshot, &config.units);
            snapshot.timestamp = Utc::now();
            return Ok(snapshot);
        }
        if collected.successful {
            if collected.snapshot.measurement_scope == ResourceMeasurementScope::Global
                && let Some(cpu) = collected.snapshot.cpu.as_mut()
            {
                let sampler = match series {
                    SamplingSeries::OnDemand => &mut state.on_demand_global_cpu_sampler,
                    SamplingSeries::Polling => &mut state.polling_global_cpu_sampler,
                };
                cpu.consumption_rate = GlobalCpuSampler::sample(sampler);
            }
            let baseline = match series {
                SamplingSeries::OnDemand => &mut state.on_demand_baseline,
                SamplingSeries::Polling => &mut state.polling_baseline,
            };
            derive_cpu_rate(
                &mut collected.snapshot,
                collected.process_sampled_instant,
                &collected.process_cpu_samples,
                baseline,
                target_generation,
            );
            let disk_baseline = match series {
                SamplingSeries::OnDemand => &mut state.on_demand_disk_baseline,
                SamplingSeries::Polling => &mut state.polling_disk_baseline,
            };
            derive_disk_rates(
                &mut collected.snapshot,
                collected.process_sampled_instant,
                &collected.process_io_samples,
                disk_baseline,
                target_generation,
            );
        }
        if config.network.enabled {
            let sampler = match series {
                SamplingSeries::OnDemand => &mut state.on_demand_network_sampler,
                SamplingSeries::Polling => &mut state.polling_network_sampler,
            };
            collected.snapshot.network = Some(
                sampler
                    .get_or_insert_with(NetworkSampler::default)
                    .sample(&config.network.interfaces),
            );
        }
        convert_snapshot(&mut collected.snapshot, &config.units);
        collected.snapshot.timestamp = Utc::now();
        Ok(collected.snapshot)
    }

    fn polling_target_pending(&self) -> bool {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .awaiting_owned_target
    }

    fn prepare_cli_owned_target(self: &Arc<Self>) -> Result<CliResourceMetricsLaunchGuard> {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        if state.cli_launch_prepared {
            return Err(FlowError::AlreadyExists(
                "a CLI resource metrics process-tree launch is already prepared".into(),
            ));
        }
        state.cli_launch_prepared = true;
        Ok(CliResourceMetricsLaunchGuard {
            manager: Arc::clone(self),
        })
    }
}

fn invalidated_snapshot(mut snapshot: ResourceMetricsSnapshot) -> ResourceMetricsSnapshot {
    snapshot.timestamp = Utc::now();
    snapshot.process_sampling = None;
    if let Some(cpu) = &mut snapshot.cpu {
        cpu.user_time = None;
        cpu.system_time = None;
        cpu.total_time = None;
        cpu.consumption_rate = None;
        cpu.throttled_time = None;
        cpu.effective_limit = None;
        cpu.some_pressure_stall_time = None;
        cpu.full_pressure_stall_time = None;
        cpu.limit_events.clear();
    }
    if let Some(memory) = &mut snapshot.memory {
        memory.system_used = None;
        memory.system_total = None;
        memory.system_available = None;
        memory.resident = None;
        memory.private = None;
        memory.physical_footprint = None;
        memory.virtual_memory = None;
        memory.peak_resident = None;
        memory.limit = None;
        memory.environment_accounted = None;
        memory.some_pressure_stall_time = None;
        memory.full_pressure_stall_time = None;
        memory.out_of_memory_event_count = None;
        memory.limit_events.clear();
    }
    if let Some(process) = &mut snapshot.process {
        process.active_count = None;
        process.descendant_count = None;
        process.thread_count = None;
        process.lifetime_creation_count = None;
        process.open_file_descriptor_count = None;
        process.windows_handle_count = None;
        process.limit_events.clear();
    }
    if let Some(disk) = &mut snapshot.disk {
        disk.read_data = None;
        disk.write_data = None;
        disk.read_throughput = None;
        disk.write_throughput = None;
        disk.read_operations = None;
        disk.write_operations = None;
        for filesystem in &mut disk.filesystems {
            filesystem.total_capacity = None;
            filesystem.available_capacity = None;
            filesystem.free_capacity = None;
        }
    }
    if let Some(gpu) = &mut snapshot.gpu {
        gpu.device_metrics = None;
        gpu.process_metrics = None;
    }
    snapshot
}

fn derive_cpu_rate(
    snapshot: &mut ResourceMetricsSnapshot,
    sampled_at: Instant,
    process_cpu_samples: &[ProcessCpuSample],
    baseline: &mut Option<SamplingBaseline>,
    target_generation: u64,
) {
    let Some(cpu) = snapshot.cpu.as_mut() else {
        *baseline = None;
        return;
    };
    if snapshot.measurement_scope == ResourceMeasurementScope::Global {
        *baseline = None;
        return;
    }
    if cpu.total_time.is_none() || process_cpu_samples.is_empty() {
        *baseline = None;
        return;
    }
    let process_cpu_times = process_cpu_samples
        .iter()
        .map(|sample| {
            (
                (sample.process_id, sample.start_identity),
                sample.total_cpu_time_millis,
            )
        })
        .collect::<HashMap<_, _>>();
    let next = SamplingBaseline {
        sampled_at,
        process_cpu_times,
        target_generation,
    };
    let has_direct_rate = cpu.consumption_rate.is_some();
    if baseline.as_ref().is_some_and(|previous| {
        previous.target_generation == target_generation && sampled_at <= previous.sampled_at
    }) {
        return;
    }
    let Some(previous) = baseline.replace(next) else {
        return;
    };
    if has_direct_rate {
        return;
    }
    let Some(elapsed) = sampled_at.checked_duration_since(previous.sampled_at) else {
        return;
    };
    let interval_millis = elapsed.as_secs_f64() * 1_000.0;
    if previous.target_generation != target_generation || interval_millis <= 0.0 {
        return;
    }
    let mut delta = 0_u64;
    for sample in process_cpu_samples {
        let prior = previous
            .process_cpu_times
            .get(&(sample.process_id, sample.start_identity))
            .copied()
            .unwrap_or(0);
        let Some(increment) = sample.total_cpu_time_millis.checked_sub(prior) else {
            return;
        };
        let Some(total) = delta.checked_add(increment) else {
            return;
        };
        delta = total;
    }
    let value = delta as f64 / interval_millis;
    cpu.consumption_rate = Some(ResourceMeasurement::new(value, CpuUnit::LogicalProcessors));
}

fn calculate_disk_rate(
    samples: &[ProcessIoSample],
    previous: &DiskBaseline,
    read: bool,
    seconds: f64,
    allow_partial: bool,
) -> (Option<ResourceMeasurement<f64, BandwidthUnit>>, u64) {
    let mut delta = Some(0_u64);
    let mut supplied = 0_u64;
    let mut missing = false;
    let mut reset = false;
    for sample in samples {
        let current = if read {
            sample.read_bytes
        } else {
            sample.write_bytes
        };
        let prior = previous
            .counters
            .get(&(sample.process_id, sample.start_identity))
            .map(|(prior_read, prior_write)| if read { *prior_read } else { *prior_write })
            .unwrap_or(Some(0));
        let (Some(current), Some(prior)) = (current, prior) else {
            missing = true;
            continue;
        };
        let Some(change) = current.checked_sub(prior) else {
            reset = true;
            continue;
        };
        supplied += 1;
        delta = delta.and_then(|delta| delta.checked_add(change));
    }
    let rate = if supplied == 0 || reset || (missing && !allow_partial) {
        None
    } else {
        delta.and_then(|delta| {
            let value = delta as f64 / seconds;
            value
                .is_finite()
                .then(|| ResourceMeasurement::new(value, BandwidthUnit::BytesPerSecond))
        })
    };
    (rate, supplied)
}

fn derive_disk_rates(
    snapshot: &mut ResourceMetricsSnapshot,
    sampled_at: Instant,
    samples: &[ProcessIoSample],
    baseline: &mut Option<DiskBaseline>,
    target_generation: u64,
) {
    let allow_partial = snapshot.measurement_scope == ResourceMeasurementScope::Global;
    if let Some(coverage) = snapshot.process_sampling.as_mut() {
        for (counter, rate) in [
            ("disk.read_data", "disk.read_throughput"),
            ("disk.write_data", "disk.write_throughput"),
        ] {
            if coverage.field_sampled_processes.contains_key(counter) {
                coverage.field_sampled_processes.insert(rate.into(), 0);
            }
        }
    }
    let Some(disk) = snapshot.disk.as_mut() else {
        *baseline = None;
        return;
    };
    if samples.is_empty() || (disk.read_data.is_none() && disk.write_data.is_none()) {
        *baseline = None;
        return;
    }
    let counters = samples
        .iter()
        .map(|sample| {
            (
                (sample.process_id, sample.start_identity),
                (sample.read_bytes, sample.write_bytes),
            )
        })
        .collect::<HashMap<_, _>>();
    if baseline.as_ref().is_some_and(|previous| {
        previous.target_generation == target_generation && sampled_at <= previous.sampled_at
    }) {
        return;
    }
    let previous = baseline.replace(DiskBaseline {
        sampled_at,
        counters,
        target_generation,
    });
    let Some(previous) = previous else { return };
    if previous.target_generation != target_generation {
        return;
    }
    let Some(elapsed) = sampled_at.checked_duration_since(previous.sampled_at) else {
        return;
    };
    let seconds = elapsed.as_secs_f64();
    if seconds <= 0.0 {
        return;
    }
    for (read, field) in [
        (true, "disk.read_throughput"),
        (false, "disk.write_throughput"),
    ] {
        let available = if read {
            disk.read_data.is_some()
        } else {
            disk.write_data.is_some()
        };
        if !available {
            continue;
        }
        let (rate, supplied) =
            calculate_disk_rate(samples, &previous, read, seconds, allow_partial);
        if read {
            disk.read_throughput = rate;
        } else {
            disk.write_throughput = rate;
        }
        if let Some(coverage) = snapshot.process_sampling.as_mut() {
            coverage
                .field_sampled_processes
                .insert(field.into(), supplied);
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/resource_metrics/manager_tests.rs"]
mod cpu_rate_tests;

pub(crate) async fn collect_on_demand() -> Result<ResourceMetricsSnapshot> {
    // Reject an inherited runtime before attempting to schedule work on it.
    ensure_process_owner()?;
    tokio::task::spawn_blocking(|| global_manager()?.collect_blocking(SamplingSeries::OnDemand))
        .await
        .map_err(|error| {
            FlowError::Internal(format!("resource metrics collection task failed: {error}"))
        })?
}

pub(crate) fn global_manager() -> Result<Arc<ResourceMetricsManager>> {
    ensure_process_owner()?;
    let mut manager = GLOBAL_MANAGER
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if let Some(manager) = manager
        .as_ref()
        .filter(|manager| manager.owner_pid == std::process::id())
    {
        return Ok(Arc::clone(manager));
    }
    let created = Arc::new(ResourceMetricsManager::new()?);
    *manager = Some(Arc::clone(&created));
    Ok(created)
}

/// Owns the optional resource metrics polling thread for a built-in plugin activation.
pub(crate) struct ResourceMetricsRuntime {
    manager: Arc<ResourceMetricsManager>,
    runtime_generation: u64,
    stop: Option<Sender<()>>,
    polling_thread: Option<JoinHandle<()>>,
}

/// Temporarily directs collection to a CLI-owned process tree.
#[doc(hidden)]
pub struct ResourceMetricsTargetGuard {
    manager: Arc<ResourceMetricsManager>,
    generation: u64,
}

/// Keeps a CLI launch's process-tree target pending until the launched child is selected.
#[doc(hidden)]
pub struct CliResourceMetricsLaunchGuard {
    manager: Arc<ResourceMetricsManager>,
}

impl Drop for CliResourceMetricsLaunchGuard {
    fn drop(&mut self) {
        if self.manager.owner_pid != std::process::id() {
            return;
        }
        let mut state = self
            .manager
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        state.cli_launch_prepared = false;
    }
}

pub(crate) fn prepare_cli_owned_process_tree() -> Result<CliResourceMetricsLaunchGuard> {
    global_manager()?.prepare_cli_owned_target()
}

impl Drop for ResourceMetricsTargetGuard {
    fn drop(&mut self) {
        if self.manager.owner_pid != std::process::id() {
            return;
        }
        self.manager.restore_application_target(self.generation);
    }
}

#[cfg(not(windows))]
pub(crate) fn target_owned_process_tree(
    process_id: u32,
) -> Result<Option<ResourceMetricsTargetGuard>> {
    let manager = global_manager()?;
    let target = owned_process_tree_target(process_id)?;
    Ok(manager
        .target_owned_process_tree(target)
        .map(|generation| ResourceMetricsTargetGuard {
            manager,
            generation,
        }))
}

#[cfg(windows)]
pub(crate) fn target_owned_process_tree_with_job_handle(
    process_id: u32,
    job_handle: isize,
) -> Result<Option<ResourceMetricsTargetGuard>> {
    let manager = global_manager()?;
    let target = owned_process_tree_target_with_job_handle(process_id, job_handle)?;
    Ok(manager
        .target_owned_process_tree(target)
        .map(|generation| ResourceMetricsTargetGuard {
            manager,
            generation,
        }))
}

impl ResourceMetricsRuntime {
    pub(crate) fn configure_current_process(config: ResourceMetricsConfig) -> Result<Self> {
        let mut config = config;
        let scope = match config.measurement_scope {
            ResourceMetricsMeasurementScope::RuntimeDefault
            | ResourceMetricsMeasurementScope::ProcessTree => ResourceMeasurementScope::ProcessTree,
            ResourceMetricsMeasurementScope::Global => ResourceMeasurementScope::Global,
            ResourceMetricsMeasurementScope::ApplicationProcess => {
                ResourceMeasurementScope::ApplicationProcess
            }
        };
        config.measurement_scope = match scope {
            ResourceMeasurementScope::Global => ResourceMetricsMeasurementScope::Global,
            ResourceMeasurementScope::ApplicationProcess => {
                ResourceMetricsMeasurementScope::ApplicationProcess
            }
            ResourceMeasurementScope::ProcessTree => ResourceMetricsMeasurementScope::ProcessTree,
        };
        Self::configure_target(config, current_process_target(scope)?)
    }

    fn configure_target(config: ResourceMetricsConfig, target: CollectionTarget) -> Result<Self> {
        config.validate()?;
        let manager = global_manager()?;
        let active_scope_stack = crate::api::runtime::current_scope_stack();
        let fallback_root_stack =
            crate::api::runtime::scope_stack::root_scope_stack_snapshot(&active_scope_stack)?;
        let fallback_scope_stack = Arc::downgrade(&active_scope_stack);
        let runtime_generation = manager.activate(target, config.clone())?;
        let (stop, polling_thread) = if config.polling.enabled {
            let (stop, receiver) = mpsc::channel();
            let manager_for_thread = Arc::clone(&manager);
            let interval = Duration::from_millis(config.polling.interval_millis);
            let polling_thread = match thread::Builder::new()
                .name("nemo-relay-resource-metrics".into())
                .spawn(move || {
                    polling_loop(
                        manager_for_thread,
                        receiver,
                        interval,
                        fallback_scope_stack,
                        fallback_root_stack,
                    )
                }) {
                Ok(thread) => thread,
                Err(error) => {
                    manager.deactivate(runtime_generation);
                    return Err(FlowError::Internal(format!(
                        "failed to start resource metrics polling thread: {error}"
                    )));
                }
            };
            (Some(stop), Some(polling_thread))
        } else {
            (None, None)
        };
        Ok(Self {
            manager,
            runtime_generation,
            stop,
            polling_thread,
        })
    }
}

impl Drop for ResourceMetricsRuntime {
    fn drop(&mut self) {
        if self.manager.owner_pid != std::process::id() {
            return;
        }
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(polling_thread) = self.polling_thread.take()
            && polling_thread.join().is_err()
        {
            log::warn!(target: "nemo_relay.plugin", event = "resource_metrics_polling_join_failed", error_kind = "thread_panic"; "Resource metrics polling thread panicked during shutdown");
        }
        self.manager.deactivate(self.runtime_generation);
    }
}

fn polling_loop(
    manager: Arc<ResourceMetricsManager>,
    stop: mpsc::Receiver<()>,
    interval: Duration,
    fallback_scope_stack: Weak<RwLock<ScopeStack>>,
    fallback_root_stack: crate::api::runtime::ScopeStackHandle,
) {
    loop {
        if !manager.polling_target_pending() {
            match manager.collect_blocking(SamplingSeries::Polling) {
                Ok(snapshot) => {
                    let fallback = fallback_scope_stack
                        .upgrade()
                        .unwrap_or_else(|| Arc::clone(&fallback_root_stack));
                    crate::api::resource_metrics::emit_resource_metrics_snapshot(
                        &snapshot, &fallback,
                    )
                }
                Err(error) => {
                    log::warn!(target: "nemo_relay.plugin", event = "resource_metrics_collection_failed", error_kind = "inactive"; "Resource metrics poll was skipped: {error}")
                }
            }
        }
        match stop.recv_timeout(interval) {
            Ok(()) | Err(RecvTimeoutError::Disconnected) => break,
            Err(RecvTimeoutError::Timeout) => {}
        }
    }
}
