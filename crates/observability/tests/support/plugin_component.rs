// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#![allow(unused_imports)]
pub(crate) use crate::header_file::PlaintextPolicy;
pub(crate) use crate::host::EventSubscriberFn;
pub(crate) use crate::otel::{
    ExporterInitializationError, OpenTelemetryConfig as CoreOpenTelemetryConfig,
    OpenTelemetryError, OpenTelemetryFileSinkConfig as CoreOpenTelemetryFileSinkConfig,
    OpenTelemetrySubscriber, OtlpFileSinkMode, OtlpFileSinkSettings, OtlpTransport,
};
pub(crate) use crate::otel_file::OtlpFileFormat;
pub(crate) use crate::otel_logs::{
    OpenTelemetryLogConfig as CoreOpenTelemetryLogConfig, OpenTelemetryLogSubscriber,
};
pub(crate) use crate::otel_metrics::{
    MetricTemporality, OpenTelemetryMetricConfig as CoreOpenTelemetryMetricConfig,
    OpenTelemetryMetricSubscriber,
};
pub(crate) use crate::plugin::*;
pub(crate) use crate::{
    MarkProjection, OpenTelemetryTlsConfig, OpenTelemetryType, default_mark_exclude_names,
};
pub(crate) use nemo_relay_types::config_editor::*;
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::AtomicU8;
use std::time::Duration;
use uuid::Uuid;
#[path = "../unit/plugin_component_independent_tests.rs"]
mod independent_tests;
#[path = "../unit/tls_independent_tests.rs"]
mod tls_tests;
