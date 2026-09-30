// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::plugins::resource_metrics::config::ResourceMetricsUnits;
use nemo_relay_types::api::resource_metrics::{
    ResourceMeasurement, ResourceMeasurementUnit as Unit, ResourceMetricValue,
    ResourceMetricsSnapshot,
};

fn time_factor(unit: Unit) -> Option<u128> {
    Some(match unit {
        Unit::Microseconds => 1,
        Unit::Milliseconds => 1_000,
        Unit::Seconds => 1_000_000,
        Unit::Minutes => 60_000_000,
        _ => return None,
    })
}

fn data_factor(unit: Unit) -> Option<u128> {
    Some(match unit {
        Unit::Bytes => 1,
        Unit::Kilobytes => 1_000,
        Unit::Megabytes => 1_000_000,
        Unit::Gigabytes => 1_000_000_000,
        Unit::Terabytes => 1_000_000_000_000,
        Unit::Kibibytes => 1_024,
        Unit::Mebibytes => 1_048_576,
        Unit::Gibibytes => 1_073_741_824,
        Unit::Tebibytes => 1_099_511_627_776,
        _ => return None,
    })
}

fn rate_factor(unit: Unit) -> Option<f64> {
    Some(match unit {
        Unit::BytesPerSecond => 1.0,
        Unit::KibibytesPerSecond => 1_024.0,
        Unit::MebibytesPerSecond => 1_048_576.0,
        Unit::GibibytesPerSecond => 1_073_741_824.0,
        Unit::BitsPerSecond => 0.125,
        Unit::MegabitsPerSecond => 125_000.0,
        Unit::GigabitsPerSecond => 125_000_000.0,
        _ => return None,
    })
}

fn convert_integer(
    measurement: &mut Option<ResourceMeasurement<ResourceMetricValue>>,
    target: Unit,
) {
    let Some(current) = measurement.as_mut() else {
        return;
    };
    if current.unit == target {
        return;
    }
    let factors = time_factor(current.unit)
        .zip(time_factor(target))
        .or_else(|| data_factor(current.unit).zip(data_factor(target)));
    let Some((source_factor, target_factor)) = factors else {
        *measurement = None;
        return;
    };
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

fn convert_float(measurement: &mut Option<ResourceMeasurement<f64>>, target: Unit) {
    let Some(current) = measurement.as_mut() else {
        return;
    };
    if current.unit == target {
        return;
    }
    let value = match (current.unit, target) {
        (Unit::LogicalProcessors, Unit::Millicores) => current.value * 1_000.0,
        (Unit::Percentage, Unit::Fraction) => current.value / 100.0,
        (Unit::BytesPerSecond, _) => match rate_factor(target) {
            Some(factor) => current.value / factor,
            None => {
                *measurement = None;
                return;
            }
        },
        _ => {
            *measurement = None;
            return;
        }
    };
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
        convert_integer(&mut cpu.user_time, units.cpu.user_time.into());
        convert_integer(&mut cpu.system_time, units.cpu.system_time.into());
        convert_integer(&mut cpu.total_time, units.cpu.total_time.into());
        convert_integer(&mut cpu.throttled_time, units.cpu.throttled_time.into());
        convert_integer(
            &mut cpu.some_pressure_stall_time,
            units.cpu.some_pressure_stall_time.into(),
        );
        convert_integer(
            &mut cpu.full_pressure_stall_time,
            units.cpu.full_pressure_stall_time.into(),
        );
        convert_float(&mut cpu.consumption_rate, units.cpu.consumption_rate.into());
        convert_float(&mut cpu.effective_limit, units.cpu.effective_limit.into());
    }
    if let Some(memory) = snapshot.memory.as_mut() {
        macro_rules! data { ($($field:ident),*) => { $(convert_integer(&mut memory.$field, units.memory.$field.into());)* } }
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
            units.memory.some_pressure_stall_time.into(),
        );
        convert_integer(
            &mut memory.full_pressure_stall_time,
            units.memory.full_pressure_stall_time.into(),
        );
    }
    if let Some(disk) = snapshot.disk.as_mut() {
        convert_integer(&mut disk.read_data, units.disk.read_data.into());
        convert_integer(&mut disk.write_data, units.disk.write_data.into());
        convert_float(&mut disk.read_throughput, units.disk.read_throughput.into());
        convert_float(
            &mut disk.write_throughput,
            units.disk.write_throughput.into(),
        );
        for filesystem in &mut disk.filesystems {
            convert_integer(
                &mut filesystem.total_capacity,
                units.disk.filesystem_total_capacity.into(),
            );
            convert_integer(
                &mut filesystem.available_capacity,
                units.disk.filesystem_available_capacity.into(),
            );
            convert_integer(
                &mut filesystem.free_capacity,
                units.disk.filesystem_free_capacity.into(),
            );
        }
    }
    if let Some(gpu) = snapshot.gpu.as_mut() {
        if let Some(devices) = gpu.device_metrics.as_mut() {
            for device in devices {
                convert_integer(&mut device.memory_used, units.gpu.device_memory_used.into());
                convert_float(
                    &mut device.compute_utilization,
                    units.gpu.device_compute_utilization.into(),
                );
            }
        }
        if let Some(processes) = gpu.process_metrics.as_mut() {
            for process in processes {
                convert_integer(
                    &mut process.memory_used,
                    units.gpu.process_memory_used.into(),
                );
                convert_float(
                    &mut process.compute_utilization,
                    units.gpu.process_compute_utilization.into(),
                );
            }
        }
    }
    if let Some(network) = snapshot.network.as_mut() {
        let system = &units.network.system;
        convert_integer(
            &mut network.system.received_data,
            system.received_data.into(),
        );
        convert_integer(
            &mut network.system.transmitted_data,
            system.transmitted_data.into(),
        );
        convert_float(
            &mut network.system.receive_throughput,
            system.receive_throughput.into(),
        );
        convert_float(
            &mut network.system.transmit_throughput,
            system.transmit_throughput.into(),
        );
        for interface in &mut network.interfaces {
            let selected = &units.network.interface;
            convert_integer(
                &mut interface.traffic.received_data,
                selected.received_data.into(),
            );
            convert_integer(
                &mut interface.traffic.transmitted_data,
                selected.transmitted_data.into(),
            );
            convert_float(
                &mut interface.traffic.receive_throughput,
                selected.receive_throughput.into(),
            );
            convert_float(
                &mut interface.traffic.transmit_throughput,
                selected.transmit_throughput.into(),
            );
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/resource_metrics/units_tests.rs"]
mod tests;
