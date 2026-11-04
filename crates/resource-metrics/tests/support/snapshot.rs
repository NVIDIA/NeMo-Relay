// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay_types::api::resource_metrics::ResourceMetricsSnapshot;
use serde_json::json;

pub fn full_snapshot() -> ResourceMetricsSnapshot {
    let measurement = |unit| json!({"value": 2048, "unit": unit});
    let mut value = json!({
        "timestamp": "2026-09-30T00:00:00Z", "operating_system": "linux",
        "measurement_scope": "process_tree",
        "process_sampling": {"visible_processes": 2, "sampled_processes": 2,
            "field_sampled_processes": {"cpu.total_time": 2}},
        "cpu": {"limit_events": [{"resource": "cpu", "event": "throttled", "count": measurement("events")}]},
        "memory": {"limit_events": [{"resource": "memory", "event": "out_of_memory", "count": measurement("events")}]},
        "process": {"limit_events": [{"resource": "processes", "event": "maximum", "count": measurement("events")}]},
        "disk": {"filesystems": [{"path": "/fixture", "total_capacity": measurement("bytes"),
            "available_capacity": measurement("bytes"), "free_capacity": measurement("bytes")}]},
        "gpu": {
            "device_metrics": [{"vendor": "nvidia", "device_identifier": "GPU-fixture", "device_index": 0,
                "memory_used": measurement("kibibytes"), "compute_utilization": {"value": 25.0, "unit": "percentage"}}],
            "process_metrics": [{"vendor": "nvidia", "device_identifier": "GPU-fixture", "device_index": 0,
                "process_id": 42, "memory_used": measurement("kibibytes"),
                "compute_utilization": {"value": 50.0, "unit": "percentage"}}]
        },
        "network": {"measurement_scope": "global", "system": {}, "interfaces": []}
    });
    for field in [
        "user_time",
        "system_time",
        "total_time",
        "throttled_time",
        "some_pressure_stall_time",
        "full_pressure_stall_time",
    ] {
        value["cpu"][field] = measurement("milliseconds");
    }
    for field in ["consumption_rate", "effective_limit"] {
        value["cpu"][field] = json!({"value": 1.5, "unit": "logical_processors"});
    }
    for field in [
        "system_used",
        "system_total",
        "system_available",
        "resident",
        "private",
        "physical_footprint",
        "virtual_memory",
        "peak_resident",
        "limit",
        "environment_accounted",
    ] {
        value["memory"][field] = measurement("kibibytes");
    }
    for field in ["some_pressure_stall_time", "full_pressure_stall_time"] {
        value["memory"][field] = measurement("milliseconds");
    }
    value["memory"]["out_of_memory_event_count"] = measurement("events");
    for (field, unit) in [
        ("active_count", "processes"),
        ("descendant_count", "processes"),
        ("thread_count", "threads"),
        ("lifetime_creation_count", "processes"),
        ("open_file_descriptor_count", "file_descriptors"),
        ("windows_handle_count", "handles"),
    ] {
        value["process"][field] = measurement(unit);
    }
    for (field, unit) in [
        ("read_data", "bytes"),
        ("write_data", "bytes"),
        ("read_throughput", "bytes_per_second"),
        ("write_throughput", "bytes_per_second"),
        ("read_operations", "operations"),
        ("write_operations", "operations"),
    ] {
        value["disk"][field] = measurement(unit);
    }
    for (field, unit) in [
        ("received_data", "bytes"),
        ("transmitted_data", "bytes"),
        ("receive_throughput", "bytes_per_second"),
        ("transmit_throughput", "bytes_per_second"),
        ("received_packets", "packets"),
        ("transmitted_packets", "packets"),
        ("receive_errors", "errors"),
        ("transmit_errors", "errors"),
    ] {
        value["network"]["system"][field] = measurement(unit);
    }
    value["network"]["interfaces"] =
        json!([{"name": "fixture0", "traffic": value["network"]["system"]}]);
    serde_json::from_value(value).unwrap()
}
