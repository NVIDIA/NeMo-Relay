// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One fixed memory sample, replaced at collection completion. Never used by exporters.
use super::collector::CollectionTarget;
use super::manager::SamplingSeries;
use chrono::{DateTime, Utc};
use nemo_relay_types::{api::resource_metrics::*, diagnostics::*};
use std::time::Instant;
use uuid::Uuid;

#[derive(Clone, Copy)]
pub(super) struct Ticket {
    pub collector: Uuid,
    pub runtime: u64,
    pub target: u64,
    pub ordinal: u64,
    pub started_at: DateTime<Utc>,
    pub started: Instant,
    pub stream: SamplingSeries,
}
pub(super) struct LatestMemory {
    pub sample: Option<ResourceMemorySample>,
    pub received: Option<Instant>,
    pub ordinal: u64,
    pub availability: ResourceMemoryAvailability,
}
impl Default for LatestMemory {
    fn default() -> Self {
        Self {
            sample: None,
            received: None,
            ordinal: 0,
            availability: ResourceMemoryAvailability::NotYetSampled,
        }
    }
}
impl LatestMemory {
    pub fn publish(
        &mut self,
        snapshot: &ResourceMetricsSnapshot,
        target: &CollectionTarget,
        root_verified: bool,
        successful: bool,
        ticket: Ticket,
    ) {
        if ticket.ordinal <= self.ordinal {
            return;
        }
        self.ordinal = ticket.ordinal;
        if !cfg!(any(target_os = "linux", target_os = "macos", windows)) {
            self.availability = ResourceMemoryAvailability::Unsupported;
            return;
        }
        if !successful {
            self.availability = ResourceMemoryAvailability::Failed;
            return;
        }
        let Some(memory) = project(snapshot) else {
            self.availability = ResourceMemoryAvailability::CapacityUnavailable;
            return;
        };
        self.sample = Some(ResourceMemorySample {
            collector_instance_id: ticket.collector,
            runtime_generation: ticket.runtime.into(),
            target_generation: ticket.target.into(),
            ordinal: ticket.ordinal.into(),
            stream: match ticket.stream {
                SamplingSeries::OnDemand => ResourceSampleStream::OnDemand,
                SamplingSeries::Polling => ResourceSampleStream::Polling,
            },
            started_at: ticket.started_at,
            completed_at: snapshot.timestamp,
            collection_millis: (u64::try_from(ticket.started.elapsed().as_millis())
                .unwrap_or(u64::MAX))
            .into(),
            operating_system: snapshot.operating_system,
            measurement_scope: snapshot.measurement_scope,
            root: root_verified.then_some(ResourceProcessIdentity {
                process_id: target.process_id,
                start_identity: target.start_identity.into(),
            }),
            visible_processes: snapshot
                .process_sampling
                .as_ref()
                .map(|c| c.visible_processes.into()),
            sampled_processes: snapshot
                .process_sampling
                .as_ref()
                .map(|c| c.sampled_processes.into()),
            memory,
        });
        self.received = Some(Instant::now());
        self.availability = ResourceMemoryAvailability::Available;
    }
    pub fn view(
        &self,
        source_pid: Option<u32>,
        application: &CollectionTarget,
        stale_after: u64,
    ) -> ResourceMemoryView {
        let age = self
            .received
            .map(|t| u64::try_from(t.elapsed().as_millis()).unwrap_or(u64::MAX));
        let association = self
            .sample
            .as_ref()
            .map_or(ResourceAssociation::Unverified, |s| {
                if s.measurement_scope == ResourceMeasurementScope::Global {
                    return ResourceAssociation::System;
                }
                match s.root {
                    Some(root)
                        if source_pid == Some(application.process_id)
                            && root.process_id == application.process_id
                            && root.start_identity.get() == application.start_identity =>
                    {
                        if s.measurement_scope == ResourceMeasurementScope::ApplicationProcess {
                            ResourceAssociation::HostProcess
                        } else {
                            ResourceAssociation::HostProcessTree
                        }
                    }
                    Some(_) if s.measurement_scope == ResourceMeasurementScope::ProcessTree => {
                        ResourceAssociation::OtherProcessTree
                    }
                    _ => ResourceAssociation::Unverified,
                }
            });
        ResourceMemoryView {
            availability: self.availability,
            sample: self.sample,
            association,
            age_millis: age.map(Into::into),
            stale: age.is_some_and(|n| n > stale_after),
            projection_slot_bytes: (size_of::<Self>() as u64).into(),
        }
    }
}
fn measurement<U: ResourceUnit + Copy>(
    v: &Option<ResourceMeasurement<ResourceMetricValue, U>>,
    unit: impl FnOnce(U) -> ResourceMemoryUnit,
    scope: MemoryFieldScope,
    coverage: Option<DiagnosticNumber>,
) -> Option<ResourceMemoryMeasurement> {
    v.as_ref().map(|v| ResourceMemoryMeasurement {
        value: match v.value {
            ResourceMetricValue::Integer(n) => ResourceMemoryNumber::Integer(n.into()),
            ResourceMetricValue::Decimal(n) if n.is_finite() => ResourceMemoryNumber::Decimal(n),
            ResourceMetricValue::Decimal(_) => ResourceMemoryNumber::NonFinite,
        },
        unit: unit(v.unit),
        measurement_scope: scope,
        sampled_processes: coverage,
    })
}
fn project(snapshot: &ResourceMetricsSnapshot) -> Option<ResourceMemoryMetrics> {
    let m = snapshot.memory.as_ref()?;
    let pressure_scope = match snapshot.measurement_scope {
        ResourceMeasurementScope::Global => MemoryFieldScope::System,
        ResourceMeasurementScope::ApplicationProcess | ResourceMeasurementScope::ProcessTree => {
            MemoryFieldScope::Environment
        }
    };
    let coverage = |key: &str| {
        snapshot
            .process_sampling
            .as_ref()
            .and_then(|p| p.field_sampled_processes.get(key))
            .copied()
            .map(Into::into)
    };
    let mut limits = [None; 8];
    if m.limit_events.len() > limits.len() {
        return None;
    }
    // Only finite known memory coverage is retained; reject added memory keys until supported.
    if snapshot.process_sampling.as_ref().is_some_and(|p| {
        p.field_sampled_processes.keys().any(|k| {
            k.starts_with("memory.")
                && !matches!(
                    k.as_str(),
                    "memory.resident"
                        | "memory.private"
                        | "memory.physical_footprint"
                        | "memory.virtual_memory"
                        | "memory.peak_resident"
                )
        })
    }) {
        return None;
    }
    for (slot, e) in limits.iter_mut().zip(&m.limit_events) {
        *slot = Some(MemoryLimitEvent {
            resource: e.resource,
            event: e.event,
            count: measurement(
                &e.count,
                ResourceMemoryUnit::Count,
                MemoryFieldScope::Environment,
                None,
            ),
        });
    }
    Some(ResourceMemoryMetrics {
        system_used: measurement(
            &m.system_used,
            ResourceMemoryUnit::Capacity,
            MemoryFieldScope::System,
            coverage("memory.system_used"),
        ),
        system_total: measurement(
            &m.system_total,
            ResourceMemoryUnit::Capacity,
            MemoryFieldScope::System,
            coverage("memory.system_total"),
        ),
        system_available: measurement(
            &m.system_available,
            ResourceMemoryUnit::Capacity,
            MemoryFieldScope::System,
            coverage("memory.system_available"),
        ),
        resident: measurement(
            &m.resident,
            ResourceMemoryUnit::Capacity,
            MemoryFieldScope::SelectedProcesses,
            coverage("memory.resident"),
        ),
        private: measurement(
            &m.private,
            ResourceMemoryUnit::Capacity,
            MemoryFieldScope::SelectedProcesses,
            coverage("memory.private"),
        ),
        physical_footprint: measurement(
            &m.physical_footprint,
            ResourceMemoryUnit::Capacity,
            MemoryFieldScope::SelectedProcesses,
            coverage("memory.physical_footprint"),
        ),
        virtual_memory: measurement(
            &m.virtual_memory,
            ResourceMemoryUnit::Capacity,
            MemoryFieldScope::SelectedProcesses,
            coverage("memory.virtual_memory"),
        ),
        peak_resident: measurement(
            &m.peak_resident,
            ResourceMemoryUnit::Capacity,
            MemoryFieldScope::SelectedProcesses,
            coverage("memory.peak_resident"),
        ),
        limit: measurement(
            &m.limit,
            ResourceMemoryUnit::Capacity,
            MemoryFieldScope::Environment,
            coverage("memory.limit"),
        ),
        environment_accounted: measurement(
            &m.environment_accounted,
            ResourceMemoryUnit::Capacity,
            MemoryFieldScope::Environment,
            coverage("memory.environment_accounted"),
        ),
        some_pressure_stall_time: measurement(
            &m.some_pressure_stall_time,
            ResourceMemoryUnit::Duration,
            pressure_scope,
            coverage("memory.some_pressure_stall_time"),
        ),
        full_pressure_stall_time: measurement(
            &m.full_pressure_stall_time,
            ResourceMemoryUnit::Duration,
            pressure_scope,
            coverage("memory.full_pressure_stall_time"),
        ),
        out_of_memory_event_count: measurement(
            &m.out_of_memory_event_count,
            ResourceMemoryUnit::Count,
            MemoryFieldScope::Environment,
            coverage("memory.out_of_memory_event_count"),
        ),
        limit_events: limits,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pressure_projection_preserves_system_and_environment_attribution() {
        let mut snapshot = super::super::snapshot_fixture::full_snapshot();
        for (scope, expected) in [
            (ResourceMeasurementScope::Global, MemoryFieldScope::System),
            (
                ResourceMeasurementScope::ApplicationProcess,
                MemoryFieldScope::Environment,
            ),
            (
                ResourceMeasurementScope::ProcessTree,
                MemoryFieldScope::Environment,
            ),
        ] {
            snapshot.measurement_scope = scope;
            let memory = project(&snapshot).unwrap();
            for pressure in [
                memory.some_pressure_stall_time,
                memory.full_pressure_stall_time,
            ] {
                assert_eq!(pressure.unwrap().measurement_scope, expected, "{scope:?}");
            }
        }
    }

    #[test]
    fn association_requires_verified_root_and_matching_process_incarnation() {
        let application =
            super::super::collector::current_process_target(ResourceMeasurementScope::ProcessTree)
                .unwrap();
        let pid = application.process_id;
        for (scope, source, other_pid, other_start, verified, expected) in [
            (
                ResourceMeasurementScope::ApplicationProcess,
                Some(pid),
                false,
                false,
                true,
                ResourceAssociation::HostProcess,
            ),
            (
                ResourceMeasurementScope::ProcessTree,
                Some(pid),
                false,
                false,
                true,
                ResourceAssociation::HostProcessTree,
            ),
            (
                ResourceMeasurementScope::ProcessTree,
                Some(pid),
                true,
                false,
                true,
                ResourceAssociation::OtherProcessTree,
            ),
            (
                ResourceMeasurementScope::ProcessTree,
                Some(pid),
                false,
                true,
                true,
                ResourceAssociation::OtherProcessTree,
            ),
            (
                ResourceMeasurementScope::ApplicationProcess,
                Some(pid),
                false,
                true,
                true,
                ResourceAssociation::Unverified,
            ),
            (
                ResourceMeasurementScope::ProcessTree,
                Some(pid),
                false,
                false,
                false,
                ResourceAssociation::Unverified,
            ),
            (
                ResourceMeasurementScope::ApplicationProcess,
                None,
                false,
                false,
                true,
                ResourceAssociation::Unverified,
            ),
        ] {
            let mut target = application.clone();
            if other_pid {
                target.process_id = pid.wrapping_add(1);
            }
            if other_start {
                target.start_identity = target.start_identity.wrapping_add(1);
            }
            let mut snapshot = super::super::snapshot_fixture::full_snapshot();
            snapshot.measurement_scope = scope;
            let mut cache = LatestMemory::default();
            cache.publish(&snapshot, &target, verified, true, ticket(1));
            let view = cache.view(source, &application, 30_000);
            assert_eq!(
                view.association, expected,
                "{scope:?}, source={source:?}, pid_changed={other_pid}, start_changed={other_start}, verified={verified}"
            );
            assert_eq!(view.sample.unwrap().root.is_some(), verified);
        }
    }

    fn ticket(ordinal: u64) -> Ticket {
        Ticket {
            collector: Uuid::nil(),
            runtime: 3,
            target: 5,
            ordinal,
            started_at: Utc::now(),
            started: Instant::now(),
            stream: SamplingSeries::OnDemand,
        }
    }
    #[test]
    fn latest_projection_preserves_exact_values_scopes_units_and_start_order() {
        let mut cache = LatestMemory::default();
        let target = super::super::collector::current_process_target(
            ResourceMeasurementScope::ApplicationProcess,
        )
        .unwrap();
        let mut snap = super::super::snapshot_fixture::full_snapshot();
        snap.measurement_scope = ResourceMeasurementScope::ApplicationProcess;
        snap.memory.as_mut().unwrap().resident =
            Some(ResourceMeasurement::new(u64::MAX, CapacityUnit::Bytes));
        snap.memory.as_mut().unwrap().private =
            Some(ResourceMeasurement::new(1.25, CapacityUnit::Kibibytes));
        cache.publish(&snap, &target, true, true, ticket(2));
        let view = cache.view(Some(target.process_id), &target, 30000);
        assert_eq!(view.association, ResourceAssociation::HostProcess);
        let sample = view.sample.unwrap();
        assert_eq!(
            sample.memory.resident.unwrap().value,
            ResourceMemoryNumber::Integer(u64::MAX.into())
        );
        assert_eq!(
            sample.memory.private.unwrap().value,
            ResourceMemoryNumber::Decimal(1.25)
        );
        assert_eq!(
            sample.memory.system_used.unwrap().measurement_scope,
            MemoryFieldScope::System
        );
        let json = serde_json::to_string(&view).unwrap();
        assert!(json.contains("18446744073709551615"));
        let decoded: ResourceMemoryView = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, view);
        cache.publish(&snap, &target, true, true, ticket(1));
        assert_eq!(cache.sample.unwrap().ordinal.get(), 2);
        cache.publish(&snap, &target, false, false, ticket(3));
        assert_eq!(cache.availability, ResourceMemoryAvailability::Failed);
        assert_eq!(cache.sample.unwrap().ordinal.get(), 2);
        snap.memory.as_mut().unwrap().limit_events = vec![
            ResourceLimitEventCount {
                resource: ResourceLimitResource::Memory,
                event: ResourceLimitEventKind::High,
                count: None
            };
            9
        ];
        cache.publish(&snap, &target, true, true, ticket(4));
        assert_eq!(
            cache.availability,
            ResourceMemoryAvailability::CapacityUnavailable
        );
        assert_eq!(cache.sample.unwrap().ordinal.get(), 2);
    }
    #[test]
    fn unverified_global_stale_and_nonfinite_values_remain_explicit() {
        let mut cache = LatestMemory::default();
        let target =
            super::super::collector::current_process_target(ResourceMeasurementScope::ProcessTree)
                .unwrap();
        let mut snap = super::super::snapshot_fixture::full_snapshot();
        snap.measurement_scope = ResourceMeasurementScope::ProcessTree;
        snap.memory.as_mut().unwrap().resident =
            Some(ResourceMeasurement::new(f64::INFINITY, CapacityUnit::Bytes));
        cache.publish(&snap, &target, false, true, ticket(1));
        cache.received = Some(Instant::now() - std::time::Duration::from_secs(60));
        let view = cache.view(Some(target.process_id), &target, 30000);
        assert!(view.stale);
        assert_eq!(view.association, ResourceAssociation::Unverified);
        assert_eq!(
            view.sample.unwrap().memory.resident.unwrap().value,
            ResourceMemoryNumber::NonFinite
        );
        snap.measurement_scope = ResourceMeasurementScope::Global;
        cache.publish(&snap, &target, false, true, ticket(2));
        assert_eq!(
            cache
                .view(Some(target.process_id), &target, 30000)
                .association,
            ResourceAssociation::System
        );
    }
}
