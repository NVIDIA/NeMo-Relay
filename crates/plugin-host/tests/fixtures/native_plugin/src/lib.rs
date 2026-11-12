// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native checked SDK conformance fixture.
mod checked;
use futures::StreamExt;
use nemo_relay_plugin::{
    CategoryProfile, ConfigDiagnostic, DiagnosticLevel, Event, EventCategory, EventSanitizeFields,
    Json, LlmRequest, LlmRequestInterceptOutcome, MetricKind, MetricMeasurement, MetricValueType,
    NativeExecutorConfig, PendingMarkSpec, RuntimeRegistrationKind, ScopeCategory, ScopeType,
    ToolExecutionInterceptOutcome,
};
use serde_json::{Map, json};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

static ASYNC_PENDING_ENTERED: AtomicBool = AtomicBool::new(false);

#[unsafe(no_mangle)]
pub extern "C" fn nemo_relay_fixture_async_pending_entered() -> bool {
    ASYNC_PENDING_ENTERED.swap(false, Ordering::AcqRel)
}
