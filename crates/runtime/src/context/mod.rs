// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime context state, scope propagation, and middleware composition.
//!
//! This module exposes the shared runtime building blocks behind the public
//! scope, tool, and LLM APIs. Most callers interact with higher-level helpers
//! in [`crate::api`], but bindings and advanced integrations can use this
//! module directly when they need explicit control over global middleware
//! state, scope-local registrations, or scope-stack propagation across async
//! tasks and native threads.

/// Helpers for combining global and scope-local middleware registrations.
///
/// These helpers resolve the effective middleware order seen by a tool or LLM
/// execution after global and scope-owned entries have been merged together.
pub mod registries;

#[cfg(test)]
#[path = "../../tests/unit/context_tests.rs"]
mod tests;
