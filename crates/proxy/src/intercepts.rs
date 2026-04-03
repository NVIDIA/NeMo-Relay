// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Intercept factories for the nexus-proxy crate.
//!
//! Provides [`create_tool_execution_intercept`], which builds a middleware-chain
//! intercept that reads the hot cache for [`ParallelHint`](crate::types::ParallelHint)
//! annotations.
//!
//! AgentHints injection is handled by [`DynamoIntercept`](crate::dynamo_intercept::DynamoIntercept),
//! an opt-in LLM execution intercept. The [`AGENT_HINTS_HEADER_KEY`] constant
//! remains here for backward compatibility and is used by `DynamoIntercept`.
//!
//! The tool execution intercept is designed for the hot path: reads use a
//! [`std::sync::RwLock`] read lock (non-blocking when no writer holds the
//! lock), and any failure (empty cache, poisoned lock) causes a graceful
//! pass-through with no panic.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, RwLock};

use nvidia_nat_nexus_core::{Json, ToolExecutionFn, ToolExecutionNextFn};

use crate::types::HotCache;

/// Header key used to inject agent hints into LLM requests.
pub const AGENT_HINTS_HEADER_KEY: &str = "x-nexus-proxy-agent-hints";

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
/// with the Nexus runtime via
/// [`nat_nexus_register_tool_execution_intercept`](nvidia_nat_nexus_core::nat_nexus_register_tool_execution_intercept).
pub(crate) fn create_tool_execution_intercept(hot_cache: Arc<RwLock<HotCache>>) -> ToolExecutionFn {
    Arc::new(move |_name: &str, args: Json, next: ToolExecutionNextFn| {
        let cache = hot_cache.clone();
        Box::pin(async move {
            // Read hot cache for parallel hints (non-blocking read lock).
            // Gracefully degrade if lock poisoned -- just skip hint checking.
            if let Ok(guard) = cache.read() {
                if let Some(ref plan) = guard.plan {
                    // v1: verify hints are accessible without panic.
                    // v2: actual parallel scheduling logic goes here.
                    for _hint in &plan.metadata_template.parallel_hints {
                        // ParallelHint { tool_name, group_id, explicit } accessible
                    }
                    for _group in &plan.parallel_groups {
                        // ParallelGroup { group_id, tool_names } accessible
                    }
                }
            }
            // Always continue the middleware chain in v1
            next(args).await
        }) as Pin<Box<dyn Future<Output = nvidia_nat_nexus_core::Result<Json>> + Send>>
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ExecutionPlan, HotCache, MetadataEnvelope, ParallelGroup, ParallelHint};
    use serde_json::json;
    use uuid::Uuid;

    /// Builds a test [`ExecutionPlan`] with one parallel hint.
    fn make_test_plan(agent_id: &str) -> ExecutionPlan {
        ExecutionPlan {
            agent_id: agent_id.to_string(),
            parallel_groups: vec![ParallelGroup {
                group_id: "pg-1".to_string(),
                tool_names: vec!["search".to_string(), "fetch".to_string()],
            }],
            metadata_template: MetadataEnvelope {
                run_id: Uuid::nil(),
                agent_id: agent_id.to_string(),
                parallel_hints: vec![ParallelHint {
                    tool_name: "search".to_string(),
                    group_id: "pg-1".to_string(),
                    explicit: true,
                }],
                extensions: json!({"version": 1}),
            },
        }
    }

    // ---- Tool execution intercept tests ----

    #[tokio::test]
    async fn test_tool_intercept_calls_next() {
        let hot_cache = Arc::new(RwLock::new(HotCache {
            plan: None,
            trie: None,
            agent_hints_default: None,
        }));
        let intercept = create_tool_execution_intercept(hot_cache);

        let next: ToolExecutionNextFn =
            Arc::new(|_args| Box::pin(async move { Ok(json!({"result": "ok"})) }));

        let result = intercept("test", json!({"input": 1}), next).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), json!({"result": "ok"}));
    }

    #[tokio::test]
    async fn test_tool_intercept_with_populated_cache() {
        let plan = make_test_plan("test-agent");
        let hot_cache = Arc::new(RwLock::new(HotCache {
            plan: Some(plan),
            trie: None,
            agent_hints_default: None,
        }));
        let intercept = create_tool_execution_intercept(hot_cache);

        let next: ToolExecutionNextFn =
            Arc::new(|_args| Box::pin(async move { Ok(json!({"from_next": true})) }));

        // Should not panic and should return next's result
        let result = intercept("test", json!({"tool_input": "data"}), next).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), json!({"from_next": true}));
    }

    #[tokio::test]
    async fn test_tool_intercept_passes_args_to_next() {
        let hot_cache = Arc::new(RwLock::new(HotCache {
            plan: None,
            trie: None,
            agent_hints_default: None,
        }));
        let intercept = create_tool_execution_intercept(hot_cache);

        // next captures and returns the args it received, proving pass-through
        let next: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args) }));

        let input = json!({"tool_arg": "value", "count": 42});
        let result = intercept("test", input.clone(), next).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), input);
    }
}
