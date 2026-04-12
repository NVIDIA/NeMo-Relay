// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Intercept factories for the nemo-flow-adaptive crate.
//!
//! Provides [`create_tool_execution_intercept`], which builds a middleware-chain
//! intercept that reads the hot cache for [`ParallelHint`](crate::types::ParallelHint)
//! annotations.
//!
//! AgentHints injection is handled by [`AdaptiveHintsIntercept`](crate::adaptive_hints_intercept::AdaptiveHintsIntercept),
//! an opt-in LLM execution intercept. The [`AGENT_HINTS_HEADER_KEY`] constant
//! remains here for backward compatibility and is used by `AdaptiveHintsIntercept`.
//!
//! The tool execution intercept is designed for the hot path: reads use a
//! [`std::sync::RwLock`] read lock (non-blocking when no writer holds the
//! lock), and any failure (empty cache, poisoned lock) causes a graceful
//! pass-through with no panic.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, RwLock};

use nemo_flow::context::callbacks::{ToolExecutionFn, ToolExecutionNextFn};
use nemo_flow::error::Result as FlowResult;
use nemo_flow::json::Json;

use crate::types::cache::HotCache;

/// Header key used to inject agent hints into LLM requests.
pub const AGENT_HINTS_HEADER_KEY: &str = "x-nemo-flow-adaptive-agent-hints";

/// Creates a tool execution intercept that reads the hot cache for
/// [`ParallelHint`](crate::types::ParallelHint) annotations.
///
/// In v1, this intercept verifies that hints are accessible without panicking
/// and always continues the middleware chain by calling `next(args).await`.
/// In v2, this is where parallel dispatch scheduling will be added.
///
/// If the hot cache is empty (`None`) or the lock is poisoned, the intercept
/// passes through to `next` without error -- the intercept never panics.
///
/// # Arguments
///
/// * `hot_cache` - Shared reference to the hot cache holding the current
///   [`ExecutionPlan`]. The intercept only acquires a read lock.
///
/// # Returns
///
/// An [`Arc`]-wrapped closure matching [`ToolExecutionFn`] that can be registered
/// with the NeMo Flow runtime via
/// [`register_tool_execution_intercept`](nemo_flow::register_tool_execution_intercept).
pub(crate) fn create_tool_execution_intercept(hot_cache: Arc<RwLock<HotCache>>) -> ToolExecutionFn {
    Arc::new(move |_name: &str, args: Json, next: ToolExecutionNextFn| {
        let cache = hot_cache.clone();
        Box::pin(async move {
            // Read hot cache for parallel hints (non-blocking read lock).
            // Gracefully degrade if lock poisoned -- just skip hint checking.
            if let Ok(guard) = cache.read()
                && let Some(ref plan) = guard.plan
            {
                // v1: verify hints are accessible without panic.
                // v2: actual parallel scheduling logic goes here.
                for _hint in &plan.metadata_template.parallel_hints {
                    // ParallelHint { tool_name, group_id, explicit } accessible
                }
                for _group in &plan.parallel_groups {
                    // ParallelGroup { group_id, tool_names } accessible
                }
            }
            // Always continue the middleware chain in v1
            next(args).await
        }) as Pin<Box<dyn Future<Output = FlowResult<Json>> + Send>>
    })
}

#[cfg(test)]
#[path = "../tests/unit/intercepts_tests.rs"]
mod tests;
