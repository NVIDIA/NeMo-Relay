// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::types::cache::HotCache;
use crate::types::metadata::{MetadataEnvelope, ParallelHint};
use crate::types::plan::{ExecutionPlan, ParallelGroup};
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
