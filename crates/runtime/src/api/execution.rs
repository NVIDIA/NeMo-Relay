// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stage options for managed tool and LLM calls.

/// Choose which middleware stages run for one managed call.
///
/// All stages run by default and keep their usual order. Nested calls use their
/// own options. These options do not change middleware registrations.
/// Relay still sanitizes event data, uses codecs, passes trace context,
/// and handles call completion. Turning off conditional execution skips its checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionOptions {
    /// Evaluate conditional-execution guardrails on the original input.
    pub conditional_execution: bool,
    /// Apply registered request intercepts before execution.
    pub request_intercepts: bool,
    /// Wrap the callback in registered execution intercepts.
    /// Streaming LLM calls use streaming execution intercepts.
    pub execution_intercepts: bool,
}

impl Default for ExecutionOptions {
    fn default() -> Self {
        Self {
            conditional_execution: true,
            request_intercepts: true,
            execution_intercepts: true,
        }
    }
}
