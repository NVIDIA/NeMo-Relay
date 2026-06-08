// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! PII redaction plugin integrations for NeMo Relay Core.

#[cfg(test)]
use std::sync::Mutex;

#[cfg(test)]
pub(crate) fn test_mutex() -> &'static Mutex<()> {
    crate::shared_runtime::runtime_owner_test_mutex()
}

pub(crate) mod builtin;
pub mod component;
pub(crate) mod detectors;
pub(crate) mod local;
pub(crate) mod overlay;
