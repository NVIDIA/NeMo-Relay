// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Operational logging helpers for native plugin authors.

/// Operational log severity routed through the component runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PluginLogLevel {
    /// Error-level record.
    Error = 1,
    /// Warn-level record.
    Warn = 2,
    /// Info-level record.
    Info = 3,
    /// Debug-level record.
    Debug = 4,
    /// Trace-level record.
    Trace = 5,
}

/// Emit a formatted info-level native-plugin record through the Relay host.
#[macro_export]
macro_rules! relay_info {
    ($runtime:expr, target: $target:expr, fields: $fields:expr, $($arg:tt)+) => {{
        $runtime.log($crate::PluginLogLevel::Info, $target, &format!($($arg)+), Some(&$fields))
    }};
    ($runtime:expr, target: $target:expr, $($arg:tt)+) => {{
        $runtime.log($crate::PluginLogLevel::Info, $target, &format!($($arg)+), None)
    }};
}

/// Emit a formatted warn-level native-plugin record through the Relay host.
#[macro_export]
macro_rules! relay_warn {
    ($runtime:expr, target: $target:expr, fields: $fields:expr, $($arg:tt)+) => {{
        $runtime.log($crate::PluginLogLevel::Warn, $target, &format!($($arg)+), Some(&$fields))
    }};
    ($runtime:expr, target: $target:expr, $($arg:tt)+) => {{
        $runtime.log($crate::PluginLogLevel::Warn, $target, &format!($($arg)+), None)
    }};
}

/// Emit a formatted error-level native-plugin record through the Relay host.
#[macro_export]
macro_rules! relay_error {
    ($runtime:expr, target: $target:expr, fields: $fields:expr, $($arg:tt)+) => {{
        $runtime.log($crate::PluginLogLevel::Error, $target, &format!($($arg)+), Some(&$fields))
    }};
    ($runtime:expr, target: $target:expr, $($arg:tt)+) => {{
        $runtime.log($crate::PluginLogLevel::Error, $target, &format!($($arg)+), None)
    }};
}

/// Emit a formatted debug-level native-plugin record through the Relay host.
#[macro_export]
macro_rules! relay_debug {
    ($runtime:expr, target: $target:expr, fields: $fields:expr, $($arg:tt)+) => {{
        $runtime.log($crate::PluginLogLevel::Debug, $target, &format!($($arg)+), Some(&$fields))
    }};
    ($runtime:expr, target: $target:expr, $($arg:tt)+) => {{
        $runtime.log($crate::PluginLogLevel::Debug, $target, &format!($($arg)+), None)
    }};
}

/// Emit a formatted trace-level native-plugin record through the Relay host.
#[macro_export]
macro_rules! relay_trace {
    ($runtime:expr, target: $target:expr, fields: $fields:expr, $($arg:tt)+) => {{
        $runtime.log($crate::PluginLogLevel::Trace, $target, &format!($($arg)+), Some(&$fields))
    }};
    ($runtime:expr, target: $target:expr, $($arg:tt)+) => {{
        $runtime.log($crate::PluginLogLevel::Trace, $target, &format!($($arg)+), None)
    }};
}
