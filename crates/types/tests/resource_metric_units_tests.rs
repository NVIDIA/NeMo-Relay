// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Serialization and category boundaries for resource metric units.

use nemo_relay_types::api::resource_metrics::{
    BandwidthUnit, CapacityUnit, CountUnit, CpuUnit, DataUnit, DurationUnit, ResourceMeasurement,
    ResourceUnit, UtilizationUnit,
};
use serde::Serialize;
use serde::de::DeserializeOwned;

fn check_units<U: ResourceUnit + Serialize + DeserializeOwned + PartialEq + std::fmt::Debug>(
    units: &[U],
) {
    for unit in units {
        let measurement = ResourceMeasurement::new(1_u64, *unit);
        let wire = serde_json::json!({"value": 1, "unit": unit.as_str()});
        assert_eq!(serde_json::to_value(&measurement).unwrap(), wire);
        let decoded: ResourceMeasurement<u64, U> = serde_json::from_value(wire).unwrap();
        assert_eq!(decoded, measurement);
        assert!(
            serde_json::from_value::<ResourceMeasurement<u64, U>>(
                serde_json::json!({"value": 1, "unit": "invalid"})
            )
            .is_err()
        );
    }
}

#[test]
fn semantic_unit_categories_keep_the_flat_measurement_wire_shape() {
    check_units(&[
        DurationUnit::Microseconds,
        DurationUnit::Milliseconds,
        DurationUnit::Seconds,
        DurationUnit::Minutes,
    ]);
    check_units(&[
        CapacityUnit::Bytes,
        CapacityUnit::Kilobytes,
        CapacityUnit::Megabytes,
        CapacityUnit::Gigabytes,
        CapacityUnit::Terabytes,
        CapacityUnit::Kibibytes,
        CapacityUnit::Mebibytes,
        CapacityUnit::Gibibytes,
        CapacityUnit::Tebibytes,
    ]);
    check_units(&[
        DataUnit::Bytes,
        DataUnit::Kilobytes,
        DataUnit::Megabytes,
        DataUnit::Gigabytes,
        DataUnit::Terabytes,
        DataUnit::Kibibytes,
        DataUnit::Mebibytes,
        DataUnit::Gibibytes,
        DataUnit::Tebibytes,
    ]);
    check_units(&[
        BandwidthUnit::BytesPerSecond,
        BandwidthUnit::KibibytesPerSecond,
        BandwidthUnit::MebibytesPerSecond,
        BandwidthUnit::GibibytesPerSecond,
        BandwidthUnit::BitsPerSecond,
        BandwidthUnit::MegabitsPerSecond,
        BandwidthUnit::GigabitsPerSecond,
    ]);
    check_units(&[CpuUnit::LogicalProcessors, CpuUnit::Millicores]);
    check_units(&[UtilizationUnit::Percentage, UtilizationUnit::Fraction]);
    check_units(&[
        CountUnit::Processes,
        CountUnit::Threads,
        CountUnit::FileDescriptors,
        CountUnit::Handles,
        CountUnit::Events,
        CountUnit::Operations,
        CountUnit::Packets,
        CountUnit::Errors,
    ]);
}

#[test]
fn duration_measurements_reject_capacity_and_bandwidth_units() {
    for unit in [
        "bytes",
        "megabits_per_second",
        "logical_processors",
        "fraction",
        "events",
    ] {
        assert!(
            serde_json::from_value::<ResourceMeasurement<u64, DurationUnit>>(
                serde_json::json!({"value": 1, "unit": unit})
            )
            .is_err()
        );
    }
    assert!(
        serde_json::from_value::<ResourceMeasurement<u64, CapacityUnit>>(
            serde_json::json!({"value": 1, "unit": "seconds"})
        )
        .is_err()
    );
    assert!(
        serde_json::from_value::<ResourceMeasurement<f64, BandwidthUnit>>(
            serde_json::json!({"value": 1, "unit": "bytes"})
        )
        .is_err()
    );
}
