// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(unused_imports)]

//! Core integration test views for the extracted observability implementation.

use nemo_relay_observability::otel_signal as implementation;

pub(crate) use implementation::*;

use std::borrow::Cow;

use std::collections::{BTreeMap, HashMap, HashSet};

use std::sync::mpsc;

use std::sync::{Arc, OnceLock};

use std::thread;

use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};

use opentelemetry::KeyValue;

use opentelemetry_sdk::{
    Resource,
    error::{OTelSdkError, OTelSdkResult},
};

use tonic::metadata::{MetadataKey, MetadataMap, MetadataValue};

use uuid::Uuid;

use super::otel::{
    ExporterInitializationError, ExporterInitializationResult, OpenTelemetryError, Result,
};

use super::runtime_diagnostics::{ExporterRuntimeDiagnostics, ExporterRuntimeDiagnosticsRecorder};

use super::{MetadataPromotionIssue, promote_event_metadata_attributes};

use crate::api::event::{
    Event, METRIC_DATA_SCHEMA_NAME, METRIC_DATA_SCHEMA_VERSION, MetricEnvelope, ScopeCategory,
    ValidatedMetricMeasurement,
};
