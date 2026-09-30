// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use nemo_relay_types::api::resource_metrics::ResourceMetricValue;

#[test]
fn storage_conversion_keeps_exact_integers_and_fractional_values() {
    let mut exact = Some(ResourceMeasurement::new(1024_u64, Unit::Kibibytes));
    convert_integer(&mut exact, Unit::Mebibytes);
    assert_eq!(exact.unwrap().value, ResourceMetricValue::Integer(1));

    let mut fractional = Some(ResourceMeasurement::new(1_u64, Unit::Kibibytes));
    convert_integer(&mut fractional, Unit::Mebibytes);
    let converted = fractional.unwrap();
    assert_eq!(converted.unit, Unit::Mebibytes);
    assert_eq!(converted.value, ResourceMetricValue::Decimal(1.0 / 1024.0));
}

#[test]
fn time_and_decimal_storage_factors_are_distinct() {
    let mut time = Some(ResourceMeasurement::new(1500_u64, Unit::Milliseconds));
    convert_integer(&mut time, Unit::Seconds);
    assert_eq!(time.unwrap().value, ResourceMetricValue::Decimal(1.5));
    let mut data = Some(ResourceMeasurement::new(1000_u64, Unit::Bytes));
    convert_integer(&mut data, Unit::Kilobytes);
    assert_eq!(data.unwrap().value, ResourceMetricValue::Integer(1));
}

#[test]
fn rate_cpu_and_utilization_conversions_use_their_own_dimensions() {
    let mut rate = Some(ResourceMeasurement::new(125_000.0, Unit::BytesPerSecond));
    convert_float(&mut rate, Unit::MegabitsPerSecond);
    assert_eq!(rate.unwrap().value, 1.0);
    let mut cpu = Some(ResourceMeasurement::new(1.5, Unit::LogicalProcessors));
    convert_float(&mut cpu, Unit::Millicores);
    assert_eq!(cpu.unwrap().value, 1500.0);
    let mut gpu = Some(ResourceMeasurement::new(25.0, Unit::Percentage));
    convert_float(&mut gpu, Unit::Fraction);
    assert_eq!(gpu.unwrap().value, 0.25);
}
