// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(unused_imports)]

//! Core integration test views for the extracted observability implementation.

use nemo_relay_observability::runtime_diagnostics as implementation;

pub(crate) use implementation::*;

use std::collections::{BTreeMap, BTreeSet};

use std::sync::{Arc, Mutex};

use nemo_relay_plugin_host::{RuntimeDiagnostic, record_active_plugin_runtime_diagnostic};

#[cfg(test)]
#[path = "../../integration/observability/runtime_diagnostics_tests.rs"]
mod tests;
