// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared middleware callbacks and invocation context.

/// Middleware callback and stream contracts.
pub mod callbacks;
/// Invocation-scoped codec context.
pub mod llm_execution_context;

pub use callbacks::*;
pub use llm_execution_context::LlmExecutionContext;
