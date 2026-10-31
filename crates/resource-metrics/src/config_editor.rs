// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::config::*;

nemo_relay_types::editor_config! {
    impl ResourceMetricsCpuUnits {
        user_time => { label: "user_time", kind: Enum, values: ["microseconds", "milliseconds", "seconds", "minutes"] },
        system_time => { label: "system_time", kind: Enum, values: ["microseconds", "milliseconds", "seconds", "minutes"] },
        total_time => { label: "total_time", kind: Enum, values: ["microseconds", "milliseconds", "seconds", "minutes"] },
        throttled_time => { label: "throttled_time", kind: Enum, values: ["microseconds", "milliseconds", "seconds", "minutes"] },
        some_pressure_stall_time => { label: "some_pressure_stall_time", kind: Enum, values: ["microseconds", "milliseconds", "seconds", "minutes"] },
        full_pressure_stall_time => { label: "full_pressure_stall_time", kind: Enum, values: ["microseconds", "milliseconds", "seconds", "minutes"] },
        consumption_rate => { label: "consumption_rate", kind: Enum, values: ["logical_processors", "millicores"] },
        effective_limit => { label: "effective_limit", kind: Enum, values: ["logical_processors", "millicores"] },
    }
}

nemo_relay_types::editor_config! {
    impl ResourceMetricsMemoryUnits {
        system_used => { label: "system_used", kind: Enum, values: ["bytes", "kilobytes", "megabytes", "gigabytes", "terabytes", "kibibytes", "mebibytes", "gibibytes", "tebibytes"] },
        system_total => { label: "system_total", kind: Enum, values: ["bytes", "kilobytes", "megabytes", "gigabytes", "terabytes", "kibibytes", "mebibytes", "gibibytes", "tebibytes"] },
        system_available => { label: "system_available", kind: Enum, values: ["bytes", "kilobytes", "megabytes", "gigabytes", "terabytes", "kibibytes", "mebibytes", "gibibytes", "tebibytes"] },
        resident => { label: "resident", kind: Enum, values: ["bytes", "kilobytes", "megabytes", "gigabytes", "terabytes", "kibibytes", "mebibytes", "gibibytes", "tebibytes"] },
        private => { label: "private", kind: Enum, values: ["bytes", "kilobytes", "megabytes", "gigabytes", "terabytes", "kibibytes", "mebibytes", "gibibytes", "tebibytes"] },
        physical_footprint => { label: "physical_footprint", kind: Enum, values: ["bytes", "kilobytes", "megabytes", "gigabytes", "terabytes", "kibibytes", "mebibytes", "gibibytes", "tebibytes"] },
        virtual_memory => { label: "virtual_memory", kind: Enum, values: ["bytes", "kilobytes", "megabytes", "gigabytes", "terabytes", "kibibytes", "mebibytes", "gibibytes", "tebibytes"] },
        peak_resident => { label: "peak_resident", kind: Enum, values: ["bytes", "kilobytes", "megabytes", "gigabytes", "terabytes", "kibibytes", "mebibytes", "gibibytes", "tebibytes"] },
        limit => { label: "limit", kind: Enum, values: ["bytes", "kilobytes", "megabytes", "gigabytes", "terabytes", "kibibytes", "mebibytes", "gibibytes", "tebibytes"] },
        environment_accounted => { label: "environment_accounted", kind: Enum, values: ["bytes", "kilobytes", "megabytes", "gigabytes", "terabytes", "kibibytes", "mebibytes", "gibibytes", "tebibytes"] },
        some_pressure_stall_time => { label: "some_pressure_stall_time", kind: Enum, values: ["microseconds", "milliseconds", "seconds", "minutes"] },
        full_pressure_stall_time => { label: "full_pressure_stall_time", kind: Enum, values: ["microseconds", "milliseconds", "seconds", "minutes"] },
    }
}

nemo_relay_types::editor_config! {
    impl ResourceMetricsDiskUnits {
        read_data => { label: "read_data", kind: Enum, values: ["bytes", "kilobytes", "megabytes", "gigabytes", "terabytes", "kibibytes", "mebibytes", "gibibytes", "tebibytes"] },
        write_data => { label: "write_data", kind: Enum, values: ["bytes", "kilobytes", "megabytes", "gigabytes", "terabytes", "kibibytes", "mebibytes", "gibibytes", "tebibytes"] },
        filesystem_total_capacity => { label: "filesystem_total_capacity", kind: Enum, values: ["bytes", "kilobytes", "megabytes", "gigabytes", "terabytes", "kibibytes", "mebibytes", "gibibytes", "tebibytes"] },
        filesystem_available_capacity => { label: "filesystem_available_capacity", kind: Enum, values: ["bytes", "kilobytes", "megabytes", "gigabytes", "terabytes", "kibibytes", "mebibytes", "gibibytes", "tebibytes"] },
        filesystem_free_capacity => { label: "filesystem_free_capacity", kind: Enum, values: ["bytes", "kilobytes", "megabytes", "gigabytes", "terabytes", "kibibytes", "mebibytes", "gibibytes", "tebibytes"] },
        read_throughput => { label: "read_throughput", kind: Enum, values: ["bytes_per_second", "kibibytes_per_second", "mebibytes_per_second", "gibibytes_per_second", "bits_per_second", "megabits_per_second", "gigabits_per_second"] },
        write_throughput => { label: "write_throughput", kind: Enum, values: ["bytes_per_second", "kibibytes_per_second", "mebibytes_per_second", "gibibytes_per_second", "bits_per_second", "megabits_per_second", "gigabits_per_second"] },
    }
}

nemo_relay_types::editor_config! {
    impl ResourceMetricsGpuUnits {
        device_memory_used => { label: "device_memory_used", kind: Enum, values: ["bytes", "kilobytes", "megabytes", "gigabytes", "terabytes", "kibibytes", "mebibytes", "gibibytes", "tebibytes"] },
        process_memory_used => { label: "process_memory_used", kind: Enum, values: ["bytes", "kilobytes", "megabytes", "gigabytes", "terabytes", "kibibytes", "mebibytes", "gibibytes", "tebibytes"] },
        device_compute_utilization => { label: "device_compute_utilization", kind: Enum, values: ["percentage", "fraction"] },
        process_compute_utilization => { label: "process_compute_utilization", kind: Enum, values: ["percentage", "fraction"] },
    }
}

nemo_relay_types::editor_config! {
    impl ResourceMetricsNetworkTrafficUnits {
        received_data => { label: "received_data", kind: Enum, values: ["bytes", "kilobytes", "megabytes", "gigabytes", "terabytes", "kibibytes", "mebibytes", "gibibytes", "tebibytes"] },
        transmitted_data => { label: "transmitted_data", kind: Enum, values: ["bytes", "kilobytes", "megabytes", "gigabytes", "terabytes", "kibibytes", "mebibytes", "gibibytes", "tebibytes"] },
        receive_throughput => { label: "receive_throughput", kind: Enum, values: ["bytes_per_second", "kibibytes_per_second", "mebibytes_per_second", "gibibytes_per_second", "bits_per_second", "megabits_per_second", "gigabits_per_second"] },
        transmit_throughput => { label: "transmit_throughput", kind: Enum, values: ["bytes_per_second", "kibibytes_per_second", "mebibytes_per_second", "gibibytes_per_second", "bits_per_second", "megabits_per_second", "gigabits_per_second"] },
    }
}

nemo_relay_types::editor_config! {
    impl ResourceMetricsNetworkUnits {
        system => { label: "System", kind: Section, nested: ResourceMetricsNetworkTrafficUnits, default: ResourceMetricsNetworkTrafficUnits },
        interface => { label: "Interface", kind: Section, nested: ResourceMetricsNetworkTrafficUnits, default: ResourceMetricsNetworkTrafficUnits },
    }
}

nemo_relay_types::editor_config! {
    impl ResourceMetricsUnits {
        cpu => { label: "CPU", kind: Section, nested: ResourceMetricsCpuUnits, default: ResourceMetricsCpuUnits },
        memory => { label: "Memory", kind: Section, nested: ResourceMetricsMemoryUnits, default: ResourceMetricsMemoryUnits },
        disk => { label: "Disk", kind: Section, nested: ResourceMetricsDiskUnits, default: ResourceMetricsDiskUnits },
        gpu => { label: "GPU", kind: Section, nested: ResourceMetricsGpuUnits, default: ResourceMetricsGpuUnits },
        network => { label: "Network", kind: Section, nested: ResourceMetricsNetworkUnits, default: ResourceMetricsNetworkUnits },
    }
}

nemo_relay_types::editor_config! {
    impl ResourceMetricsNetworkConfig {
        enabled => { label: "enabled", kind: Boolean },
        interfaces => { label: "interfaces", kind: List, list: &nemo_relay_types::config_editor::STRING_LIST_ITEM },
    }
}

nemo_relay_types::editor_config! {
    impl ResourceMetricsPollingConfig {
        enabled => { label: "enabled", kind: Boolean },
        interval_millis => { label: "interval_millis", kind: Integer },
    }
}

nemo_relay_types::editor_config! {
    impl ResourceMetricsCpuConfig {
        enabled => { label: "enabled", kind: Boolean },
    }
}

nemo_relay_types::editor_config! {
    impl ResourceMetricsMemoryConfig {
        enabled => { label: "enabled", kind: Boolean },
    }
}

nemo_relay_types::editor_config! {
    impl ResourceMetricsProcessConfig {
        enabled => { label: "enabled", kind: Boolean },
    }
}

nemo_relay_types::editor_config! {
    impl ResourceMetricsDiskConfig {
        enabled => { label: "enabled", kind: Boolean },
        process_io => { label: "process_io", kind: Boolean },
        filesystem_paths => {
            label: "filesystem_paths",
            kind: List,
            list: &nemo_relay_types::config_editor::STRING_LIST_ITEM,
        },
    }
}

nemo_relay_types::editor_config! {
    impl ResourceMetricsGpuConfig {
        enabled => { label: "enabled", kind: Boolean },
        devices => {
            label: "devices",
            kind: List,
            list: &nemo_relay_types::config_editor::STRING_LIST_ITEM,
        },
        device_metrics => { label: "device_metrics", kind: Boolean },
        process_metrics => { label: "process_metrics", kind: Boolean },
    }
}

nemo_relay_types::editor_config! {
    impl ResourceMetricsConfig {
        memory_inspection => { label: "Memory inspection", kind: Section, nested: ResourceMemoryInspectionConfig, default: ResourceMemoryInspectionConfig },
        measurement_scope => {
            label: "measurement_scope",
            kind: Enum,
            values: ["runtime_default", "global", "application_process", "process_tree"],
        },
        polling => {
            label: "polling",
            kind: Section,
            nested: ResourceMetricsPollingConfig,
            default: ResourceMetricsPollingConfig,
        },
        cpu => {
            label: "CPU",
            kind: Section,
            nested: ResourceMetricsCpuConfig,
            default: ResourceMetricsCpuConfig,
        },
        memory => {
            label: "Memory",
            kind: Section,
            nested: ResourceMetricsMemoryConfig,
            default: ResourceMetricsMemoryConfig,
        },
        process => {
            label: "Process",
            kind: Section,
            nested: ResourceMetricsProcessConfig,
            default: ResourceMetricsProcessConfig,
        },
        disk => {
            label: "Disk",
            kind: Section,
            nested: ResourceMetricsDiskConfig,
            default: ResourceMetricsDiskConfig,
        },
        gpu => {
            label: "GPU",
            kind: Section,
            nested: ResourceMetricsGpuConfig,
            default: ResourceMetricsGpuConfig,
        },
        network => { label: "Network", kind: Section, nested: ResourceMetricsNetworkConfig, default: ResourceMetricsNetworkConfig },
        units => { label: "Units", kind: Section, nested: ResourceMetricsUnits, default: ResourceMetricsUnits },
    }
}

nemo_relay_types::editor_config! {
    impl ResourceMemoryInspectionConfig {
        enabled => { label: "enabled", kind: Boolean },
        stale_after_millis => { label: "stale_after_millis", kind: Integer },
    }
}
