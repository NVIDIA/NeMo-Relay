// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use nemo_relay_types::api::resource_metrics::{CountUnit, ResourceMetricValue};

#[test]
fn storage_conversion_keeps_exact_integers_and_fractional_values() {
    let mut exact = Some(ResourceMeasurement::new(1024_u64, CapacityUnit::Kibibytes));
    convert_integer(&mut exact, CapacityUnit::Mebibytes);
    assert_eq!(exact.unwrap().value, ResourceMetricValue::Integer(1));

    let mut fractional = Some(ResourceMeasurement::new(1_u64, CapacityUnit::Kibibytes));
    convert_integer(&mut fractional, CapacityUnit::Mebibytes);
    let converted = fractional.unwrap();
    assert_eq!(converted.unit, CapacityUnit::Mebibytes);
    assert_eq!(converted.value, ResourceMetricValue::Decimal(1.0 / 1024.0));
}

#[test]
fn time_and_decimal_storage_factors_are_distinct() {
    let mut time = Some(ResourceMeasurement::new(
        1500_u64,
        DurationUnit::Milliseconds,
    ));
    convert_integer(&mut time, DurationUnit::Seconds);
    assert_eq!(time.unwrap().value, ResourceMetricValue::Decimal(1.5));
    let mut data = Some(ResourceMeasurement::new(1000_u64, DataUnit::Bytes));
    convert_integer(&mut data, DataUnit::Kilobytes);
    assert_eq!(data.unwrap().value, ResourceMetricValue::Integer(1));
}

#[test]
fn rate_cpu_and_utilization_conversions_use_their_own_dimensions() {
    let mut rate = Some(ResourceMeasurement::new(
        125_000.0,
        BandwidthUnit::BytesPerSecond,
    ));
    convert_float(&mut rate, BandwidthUnit::MegabitsPerSecond);
    assert_eq!(rate.unwrap().value, 1.0);
    let mut cpu = Some(ResourceMeasurement::new(1.5, CpuUnit::LogicalProcessors));
    convert_float(&mut cpu, CpuUnit::Millicores);
    assert_eq!(cpu.unwrap().value, 1500.0);
    let mut gpu = Some(ResourceMeasurement::new(25.0, UtilizationUnit::Percentage));
    convert_float(&mut gpu, UtilizationUnit::Fraction);
    assert_eq!(gpu.unwrap().value, 0.25);
}
use crate::resource_metrics::snapshot_fixture;

#[test]
fn every_category_uses_its_selected_units_and_keeps_fixed_counts() {
    let mut snapshot = snapshot_fixture::full_snapshot();
    let mut units = ResourceMetricsUnits::default();
    units.gpu.device_memory_used = crate::api::resource_metrics::CapacityUnit::Mebibytes;
    units.gpu.process_memory_used = crate::api::resource_metrics::CapacityUnit::Bytes;
    units.gpu.device_compute_utilization = crate::api::resource_metrics::UtilizationUnit::Fraction;
    units.gpu.process_compute_utilization = crate::api::resource_metrics::UtilizationUnit::Fraction;
    units.disk.filesystem_total_capacity = crate::api::resource_metrics::CapacityUnit::Kibibytes;
    units.network.interface.received_data = crate::api::resource_metrics::DataUnit::Kibibytes;
    units.network.interface.receive_throughput =
        crate::api::resource_metrics::BandwidthUnit::BitsPerSecond;
    convert_snapshot(&mut snapshot, &units);
    let gpu = snapshot.gpu.unwrap();
    let device = &gpu.device_metrics.unwrap()[0];
    assert_eq!(
        device.memory_used.as_ref().unwrap().value,
        ResourceMetricValue::Integer(2)
    );
    assert_eq!(device.compute_utilization.as_ref().unwrap().value, 0.25);
    let process = &gpu.process_metrics.unwrap()[0];
    assert_eq!(
        process.memory_used.as_ref().unwrap().value,
        ResourceMetricValue::Integer(2_097_152)
    );
    assert_eq!(process.compute_utilization.as_ref().unwrap().value, 0.5);
    let disk = snapshot.disk.unwrap();
    assert_eq!(
        disk.filesystems[0].total_capacity.as_ref().unwrap().value,
        ResourceMetricValue::Integer(2)
    );
    assert_eq!(
        disk.read_operations.as_ref().unwrap().unit,
        CountUnit::Operations
    );
    let interface = &snapshot.network.unwrap().interfaces[0];
    assert_eq!(
        interface.traffic.received_data.as_ref().unwrap().value,
        ResourceMetricValue::Integer(2)
    );
    assert_eq!(
        interface.traffic.receive_throughput.as_ref().unwrap().value,
        16_384.0
    );
    assert_eq!(
        interface.traffic.received_packets.as_ref().unwrap().unit,
        CountUnit::Packets
    );
}

#[test]
fn conversions_reject_noncanonical_values_and_nonfinite_measurements() {
    let mut measurement = Some(ResourceMeasurement::new(
        ResourceMetricValue::Decimal(1.5),
        DataUnit::Bytes,
    ));
    convert_integer(&mut measurement, DataUnit::Kilobytes);
    assert!(measurement.is_none());
    let mut measurement = Some(ResourceMeasurement::new(
        f64::INFINITY,
        UtilizationUnit::Percentage,
    ));
    convert_float(&mut measurement, UtilizationUnit::Fraction);
    assert!(measurement.is_none());
}
#[test]
fn all_configured_storage_time_and_throughput_units_match_canonical_units() {
    use crate::api::resource_metrics::{BandwidthUnit, CapacityUnit, DataUnit, DurationUnit};
    for (name, factor) in [
        ("bytes", 1_u64),
        ("kilobytes", 1000),
        ("megabytes", 1_000_000),
        ("gigabytes", 1_000_000_000),
        ("terabytes", 1_000_000_000_000),
        ("kibibytes", 1024),
        ("mebibytes", 1_048_576),
        ("gibibytes", 1_073_741_824),
        ("tebibytes", 1_099_511_627_776),
    ] {
        let selected: DataUnit = serde_json::from_value(serde_json::json!(name)).unwrap();
        let memory: CapacityUnit = serde_json::from_value(serde_json::json!(name)).unwrap();
        let unit = selected;
        assert_eq!(unit.as_str(), memory.as_str());
        assert_eq!(unit.as_str(), name);
        let mut measurement = Some(ResourceMeasurement::new(factor, DataUnit::Bytes));
        convert_integer(&mut measurement, unit);
        assert_eq!(measurement.unwrap().value, ResourceMetricValue::Integer(1));
        let mut capacity = Some(ResourceMeasurement::new(factor, CapacityUnit::Bytes));
        convert_integer(&mut capacity, memory);
        assert_eq!(capacity.unwrap().value, ResourceMetricValue::Integer(1));
    }
    for (name, micros) in [
        ("microseconds", 1_u64),
        ("milliseconds", 1000),
        ("seconds", 1_000_000),
        ("minutes", 60_000_000),
    ] {
        let selected: DurationUnit = serde_json::from_value(serde_json::json!(name)).unwrap();
        let unit = selected;
        assert_eq!(unit.as_str(), name);
        let mut measurement = Some(ResourceMeasurement::new(micros, DurationUnit::Microseconds));
        convert_integer(&mut measurement, unit);
        assert_eq!(measurement.unwrap().value, ResourceMetricValue::Integer(1));
    }
    for (name, bytes_per_second) in [
        ("bytes_per_second", 1.0),
        ("kibibytes_per_second", 1024.0),
        ("mebibytes_per_second", 1_048_576.0),
        ("gibibytes_per_second", 1_073_741_824.0),
        ("bits_per_second", 0.125),
        ("megabits_per_second", 125_000.0),
        ("gigabits_per_second", 125_000_000.0),
    ] {
        let selected: BandwidthUnit = serde_json::from_value(serde_json::json!(name)).unwrap();
        let unit = selected;
        assert_eq!(unit.as_str(), name);
        let mut measurement = Some(ResourceMeasurement::new(
            bytes_per_second,
            BandwidthUnit::BytesPerSecond,
        ));
        convert_float(&mut measurement, unit);
        assert_eq!(measurement.unwrap().value, 1.0);
    }
}
