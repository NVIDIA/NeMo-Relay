// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Semantic resource metric unit enums exposed to JavaScript and TypeScript.

use napi_derive::napi;

/// Units for duration measurements.
#[napi(string_enum = "snake_case")]
pub enum DurationUnit {
    /// Microseconds.
    Microseconds,
    /// Milliseconds.
    Milliseconds,
    /// Seconds.
    Seconds,
    /// Minutes.
    Minutes,
}

/// Units for capacity measurements.
#[napi(string_enum = "snake_case")]
pub enum CapacityUnit {
    /// Bytes.
    Bytes,
    /// Kilobytes.
    Kilobytes,
    /// Megabytes.
    Megabytes,
    /// Gigabytes.
    Gigabytes,
    /// Terabytes.
    Terabytes,
    /// Kibibytes.
    Kibibytes,
    /// Mebibytes.
    Mebibytes,
    /// Gibibytes.
    Gibibytes,
    /// Tebibytes.
    Tebibytes,
}

/// Units for data measurements.
#[napi(string_enum = "snake_case")]
pub enum DataUnit {
    /// Bytes.
    Bytes,
    /// Kilobytes.
    Kilobytes,
    /// Megabytes.
    Megabytes,
    /// Gigabytes.
    Gigabytes,
    /// Terabytes.
    Terabytes,
    /// Kibibytes.
    Kibibytes,
    /// Mebibytes.
    Mebibytes,
    /// Gibibytes.
    Gibibytes,
    /// Tebibytes.
    Tebibytes,
}

/// Units for bandwidth measurements.
#[napi(string_enum = "snake_case")]
#[allow(clippy::enum_variant_names)] // Preserve canonical unit names, including per-second denominators.
pub enum BandwidthUnit {
    /// Bytes per second.
    BytesPerSecond,
    /// Kibibytes per second.
    KibibytesPerSecond,
    /// Mebibytes per second.
    MebibytesPerSecond,
    /// Gibibytes per second.
    GibibytesPerSecond,
    /// Bits per second.
    BitsPerSecond,
    /// Megabits per second.
    MegabitsPerSecond,
    /// Gigabits per second.
    GigabitsPerSecond,
}

/// Units for CPU measurements.
#[napi(string_enum = "snake_case")]
pub enum CpuUnit {
    /// Logical processors.
    LogicalProcessors,
    /// Millicores.
    Millicores,
}

/// Units for utilization measurements.
#[napi(string_enum = "snake_case")]
pub enum UtilizationUnit {
    /// Percentage.
    Percentage,
    /// Fraction.
    Fraction,
}

/// Units for count measurements.
#[napi(string_enum = "snake_case")]
pub enum CountUnit {
    /// Processes.
    Processes,
    /// Threads.
    Threads,
    /// File descriptors.
    FileDescriptors,
    /// Handles.
    Handles,
    /// Events.
    Events,
    /// Operations.
    Operations,
    /// Packets.
    Packets,
    /// Errors.
    Errors,
}
