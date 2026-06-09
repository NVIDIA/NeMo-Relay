// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public API for the NeMo Relay runtime.

/// Lifecycle event types and builder-backed event constructors.
pub mod event;
/// LLM lifecycle helpers and managed execution entry points.
pub mod llm;
/// Global and scope-local middleware registration helpers.
pub mod registry;
/// Advanced runtime state, callbacks, and scope-stack helpers.
pub mod runtime;
/// Scope stack lifecycle and mark-event entry points.
pub mod scope;
/// Global and scope-local event subscriber registration helpers.
pub mod subscriber;
/// Tool lifecycle helpers and managed execution entry points.
pub mod tool;

pub(crate) mod shared;
