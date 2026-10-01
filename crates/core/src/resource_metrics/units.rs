// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::plugins::resource_metrics::config::ResourceMetricsUnits;
use nemo_relay_types::api::resource_metrics::{
    BandwidthUnit, CapacityUnit, CpuUnit, DataUnit, DurationUnit, ResourceMeasurement,
    ResourceMetricValue, ResourceMetricsSnapshot, ResourceUnit, UtilizationUnit,
};

trait IntegerUnit: ResourceUnit + PartialEq {
    fn factor(self) -> u128;
}

impl IntegerUnit for DurationUnit {
    fn factor(self) -> u128 {
        match self {
            Self::Microseconds => 1,
            Self::Milliseconds => 1_000,
            Self::Seconds => 1_000_000,
            Self::Minutes => 60_000_000,
        }
    }
}

impl IntegerUnit for CapacityUnit {
    fn factor(self) -> u128 {
        match self {
            Self::Bytes => 1,
            Self::Kilobytes => 1_000,
            Self::Megabytes => 1_000_000,
            Self::Gigabytes => 1_000_000_000,
            Self::Terabytes => 1_000_000_000_000,
            Self::Kibibytes => 1_024,
            Self::Mebibytes => 1_048_576,
            Self::Gibibytes => 1_073_741_824,
            Self::Tebibytes => 1_099_511_627_776,
        }
    }
}

impl IntegerUnit for DataUnit {
    fn factor(self) -> u128 {
        match self {
            Self::Bytes => 1,
            Self::Kilobytes => 1_000,
            Self::Megabytes => 1_000_000,
            Self::Gigabytes => 1_000_000_000,
            Self::Terabytes => 1_000_000_000_000,
            Self::Kibibytes => 1_024,
            Self::Mebibytes => 1_048_576,
            Self::Gibibytes => 1_073_741_824,
            Self::Tebibytes => 1_099_511_627_776,
        }
    }
}

trait FloatUnit: ResourceUnit + PartialEq {
    fn factor(self) -> f64;
}

impl FloatUnit for BandwidthUnit {
    fn factor(self) -> f64 {
        match self {
            Self::BytesPerSecond => 1.0,
            Self::KibibytesPerSecond => 1024.0,
            Self::MebibytesPerSecond => 1048576.0,
            Self::GibibytesPerSecond => 1073741824.0,
            Self::BitsPerSecond => 0.125,
            Self::MegabitsPerSecond => 125000.0,
            Self::GigabitsPerSecond => 125000000.0,
        }
    }
}

impl FloatUnit for CpuUnit {
    fn factor(self) -> f64 {
        match self {
            Self::LogicalProcessors => 1.0,
            Self::Millicores => 0.001,
        }
    }
}

impl FloatUnit for UtilizationUnit {
    fn factor(self) -> f64 {
        match self {
            Self::Percentage => 0.01,
            Self::Fraction => 1.0,
        }
    }
}

fn convert_integer<U: IntegerUnit>(
    measurement: &mut Option<ResourceMeasurement<ResourceMetricValue, U>>,
    target: U,
) {
    let Some(current) = measurement.as_mut() else {
        return;
    };
    if current.unit == target {
        return;
    }
    let source_factor = current.unit.factor();
    let target_factor = target.factor();
    let ResourceMetricValue::Integer(value) = current.value else {
        *measurement = None;
        return;
    };
    let Some(numerator) = u128::from(value).checked_mul(source_factor) else {
        *measurement = None;
        return;
    };
    let converted =
        if numerator % target_factor == 0 && numerator / target_factor <= u128::from(u64::MAX) {
            ResourceMetricValue::Integer((numerator / target_factor) as u64)
        } else {
            let decimal = numerator as f64 / target_factor as f64;
            if !decimal.is_finite() {
                *measurement = None;
                return;
            }
            ResourceMetricValue::Decimal(decimal)
        };
    current.value = converted;
    current.unit = target;
}

fn convert_float<U: FloatUnit>(measurement: &mut Option<ResourceMeasurement<f64, U>>, target: U) {
    let Some(current) = measurement.as_mut() else {
        return;
    };
    if current.unit == target {
        return;
    }
    let value = current.value * (current.unit.factor() / target.factor());
    if !value.is_finite() {
        *measurement = None;
        return;
    }
    current.value = value;
    current.unit = target;
}

pub(crate) fn convert_snapshot(
    snapshot: &mut ResourceMetricsSnapshot,
    units: &ResourceMetricsUnits,
) {
    if let Some(cpu) = snapshot.cpu.as_mut() {
        convert_integer(&mut cpu.user_time, units.cpu.user_time);
        convert_integer(&mut cpu.system_time, units.cpu.system_time);
        convert_integer(&mut cpu.total_time, units.cpu.total_time);
        convert_integer(&mut cpu.throttled_time, units.cpu.throttled_time);
        convert_integer(
            &mut cpu.some_pressure_stall_time,
            units.cpu.some_pressure_stall_time,
        );
        convert_integer(
            &mut cpu.full_pressure_stall_time,
            units.cpu.full_pressure_stall_time,
        );
        convert_float(&mut cpu.consumption_rate, units.cpu.consumption_rate);
        convert_float(&mut cpu.effective_limit, units.cpu.effective_limit);
    }
    if let Some(memory) = snapshot.memory.as_mut() {
        macro_rules! data { ($($field:ident),*) => { $(convert_integer(&mut memory.$field, units.memory.$field);)* } }
        data!(
            system_used,
            system_total,
            system_available,
            resident,
            private,
            physical_footprint,
            virtual_memory,
            peak_resident,
            limit,
            environment_accounted
        );
        convert_integer(
            &mut memory.some_pressure_stall_time,
            units.memory.some_pressure_stall_time,
        );
        convert_integer(
            &mut memory.full_pressure_stall_time,
            units.memory.full_pressure_stall_time,
        );
    }
    if let Some(disk) = snapshot.disk.as_mut() {
        convert_integer(&mut disk.read_data, units.disk.read_data);
        convert_integer(&mut disk.write_data, units.disk.write_data);
        convert_float(&mut disk.read_throughput, units.disk.read_throughput);
        convert_float(&mut disk.write_throughput, units.disk.write_throughput);
        for filesystem in &mut disk.filesystems {
            convert_integer(
                &mut filesystem.total_capacity,
                units.disk.filesystem_total_capacity,
            );
            convert_integer(
                &mut filesystem.available_capacity,
                units.disk.filesystem_available_capacity,
            );
            convert_integer(
                &mut filesystem.free_capacity,
                units.disk.filesystem_free_capacity,
            );
        }
    }
    if let Some(gpu) = snapshot.gpu.as_mut() {
        if let Some(devices) = gpu.device_metrics.as_mut() {
            for device in devices {
                convert_integer(&mut device.memory_used, units.gpu.device_memory_used);
                convert_float(
                    &mut device.compute_utilization,
                    units.gpu.device_compute_utilization,
                );
            }
        }
        if let Some(processes) = gpu.process_metrics.as_mut() {
            for process in processes {
                convert_integer(&mut process.memory_used, units.gpu.process_memory_used);
                convert_float(
                    &mut process.compute_utilization,
                    units.gpu.process_compute_utilization,
                );
            }
        }
    }
    if let Some(network) = snapshot.network.as_mut() {
        let system = &units.network.system;
        convert_integer(&mut network.system.received_data, system.received_data);
        convert_integer(
            &mut network.system.transmitted_data,
            system.transmitted_data,
        );
        convert_float(
            &mut network.system.receive_throughput,
            system.receive_throughput,
        );
        convert_float(
            &mut network.system.transmit_throughput,
            system.transmit_throughput,
        );
        for interface in &mut network.interfaces {
            let selected = &units.network.interface;
            convert_integer(&mut interface.traffic.received_data, selected.received_data);
            convert_integer(
                &mut interface.traffic.transmitted_data,
                selected.transmitted_data,
            );
            convert_float(
                &mut interface.traffic.receive_throughput,
                selected.receive_throughput,
            );
            convert_float(
                &mut interface.traffic.transmit_throughput,
                selected.transmit_throughput,
            );
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/resource_metrics/units_tests.rs"]
mod tests;
