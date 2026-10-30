// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Resource manager integration tests with core diagnostics.

#![allow(unused_imports)]

use std::collections::HashMap;

use std::sync::atomic::{AtomicU32, Ordering};

use std::sync::mpsc::{self, RecvTimeoutError, Sender};

use std::sync::{Arc, Mutex, RwLock, TryLockError, Weak};

use std::thread::{self, JoinHandle};

use std::time::{Duration, Instant};

use chrono::Utc;

use nemo_relay_types::api::resource_metrics::{
    BandwidthUnit, CpuUnit, ResourceMeasurement, ResourceMeasurementScope, ResourceMetricsSnapshot,
};

use super::network::NetworkSampler;

use super::units::convert_snapshot;

#[cfg(not(windows))]
use super::collector::owned_process_tree_target;
#[cfg(windows)]
use super::collector::owned_process_tree_target_with_job_handle;
use crate::error::{FlowError, Result};

use super::collector::{
    self, CollectionTarget, GlobalCpuSampler, ProcessCpuSample, ProcessIoSample,
    current_process_target,
};

use nemo_relay_resource_metrics::config::{ResourceMetricsConfig, ResourceMetricsMeasurementScope};
pub(crate) use nemo_relay_resource_metrics::manager::*;

#[cfg(test)]
#[path = "../../unit/resource_metrics/manager_tests.rs"]
mod cpu_rate_tests;
