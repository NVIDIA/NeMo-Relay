// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Intercept factories for the `nemo-flow-adaptive` crate, including Adaptive
//! Cache Governor (ACG) intercepts.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use nemo_flow::api::runtime::{ToolExecutionFn, ToolExecutionNextFn};
use nemo_flow::error::Result as FlowResult;
use nemo_flow::json::Json;
use tokio::sync::{Mutex, Notify};
use uuid::Uuid;

use crate::acg::MIN_ACG_OBSERVATIONS;
use crate::context_helpers::resolve_shared_parent_scope_identity;
use crate::types::cache::HotCache;

/// Header key used to propagate serialized adaptive agent hints.
pub const AGENT_HINTS_HEADER_KEY: &str = "x-nemo-flow-adaptive-agent-hints";
pub(crate) const WARM_FIRST_MAX_WAIT_MS: u64 = 150;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CohortKey {
    root_uuid: Uuid,
    shared_parent_uuid: Uuid,
    group_id: String,
}

#[derive(Debug)]
struct CohortGate {
    primer_active: AtomicBool,
    released: AtomicBool,
    release_notify: Notify,
}

impl CohortGate {
    fn new() -> Self {
        Self {
            primer_active: AtomicBool::new(true),
            released: AtomicBool::new(false),
            release_notify: Notify::new(),
        }
    }

    fn is_released(&self) -> bool {
        self.released.load(Ordering::Acquire)
    }

    fn primer_active(&self) -> bool {
        self.primer_active.load(Ordering::Acquire)
    }

    fn release(&self) {
        self.primer_active.store(false, Ordering::Release);
        self.released.store(true, Ordering::Release);
        self.release_notify.notify_waiters();
    }

    async fn wait_for_release(&self) {
        if self.is_released() {
            return;
        }
        self.release_notify.notified().await;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WarmFirstEligibility {
    follower_count: u32,
    confidence_units: u32,
    expected_benefit_units: u32,
    coordination_cost_units: u32,
}

impl WarmFirstEligibility {
    fn evaluate(
        group_width: usize,
        stable_prefix_length: usize,
        acg_observation_count: u32,
    ) -> Self {
        let follower_count = group_width.saturating_sub(1) as u32;
        let confidence_units = acg_observation_count.min(4);
        let expected_benefit_units = (stable_prefix_length as u32)
            .saturating_mul(follower_count)
            .saturating_mul(confidence_units);
        let coordination_cost_units = 12 + follower_count.saturating_pow(2).saturating_mul(4);
        Self {
            follower_count,
            confidence_units,
            expected_benefit_units,
            coordination_cost_units,
        }
    }

    fn is_eligible(&self) -> bool {
        self.expected_benefit_units > self.coordination_cost_units
    }
}

#[derive(Debug, Clone)]
enum WarmFirstRole {
    Primer(Arc<CohortGate>),
    Follower(Arc<CohortGate>),
}

#[cfg(test)]
pub(crate) fn create_tool_execution_intercept(hot_cache: Arc<RwLock<HotCache>>) -> ToolExecutionFn {
    create_tool_execution_intercept_with_mode(hot_cache, "observe_only".to_string())
}

pub(crate) fn create_tool_execution_intercept_with_mode(
    hot_cache: Arc<RwLock<HotCache>>,
    mode: String,
) -> ToolExecutionFn {
    let cohort_registry: Arc<Mutex<HashMap<CohortKey, Arc<CohortGate>>>> =
        Arc::new(Mutex::new(HashMap::new()));

    Arc::new(move |name: &str, args: Json, next: ToolExecutionNextFn| {
        let cache = hot_cache.clone();
        let registry = cohort_registry.clone();
        let mode = mode.clone();
        let name = name.to_string();
        Box::pin(async move {
            let Some(cohort_key) = resolve_warm_first_cohort_key(&name, &mode, &cache) else {
                return next(args).await;
            };

            match resolve_warm_first_role(&registry, cohort_key.clone()).await {
                WarmFirstRole::Primer(gate) => {
                    let result = next(args).await;
                    gate.release();
                    cleanup_cohort_gate(&registry, &cohort_key, &gate).await;
                    result
                }
                WarmFirstRole::Follower(gate) => {
                    let _ = tokio::time::timeout(
                        Duration::from_millis(WARM_FIRST_MAX_WAIT_MS),
                        gate.wait_for_release(),
                    )
                    .await;
                    next(args).await
                }
            }
        }) as Pin<Box<dyn Future<Output = FlowResult<Json>> + Send>>
    })
}

fn resolve_warm_first_cohort_key(
    tool_name: &str,
    mode: &str,
    hot_cache: &Arc<RwLock<HotCache>>,
) -> Option<CohortKey> {
    if mode != "schedule" {
        return None;
    }

    let scope_identity = resolve_shared_parent_scope_identity()?;
    let guard = hot_cache.read().ok()?;
    let plan = guard.plan.as_ref()?;
    let hint = plan
        .metadata_template
        .parallel_hints
        .iter()
        .find(|hint| hint.tool_name == tool_name)?;
    let group = plan
        .parallel_groups
        .iter()
        .find(|group| group.group_id == hint.group_id)?;
    if group.tool_names.len() <= 1 {
        return None;
    }

    let stable_prefix_length = guard.acg_stability.as_ref()?.stable_prefix_length;
    if stable_prefix_length == 0 || guard.acg_observation_count < MIN_ACG_OBSERVATIONS {
        return None;
    }

    let eligibility = WarmFirstEligibility::evaluate(
        group.tool_names.len(),
        stable_prefix_length,
        guard.acg_observation_count,
    );
    if !eligibility.is_eligible() {
        return None;
    }

    Some(CohortKey {
        root_uuid: scope_identity.root_uuid,
        shared_parent_uuid: scope_identity.shared_parent_uuid,
        group_id: hint.group_id.clone(),
    })
}

async fn resolve_warm_first_role(
    registry: &Arc<Mutex<HashMap<CohortKey, Arc<CohortGate>>>>,
    cohort_key: CohortKey,
) -> WarmFirstRole {
    let mut guard = registry.lock().await;

    if let Some(existing_gate) = guard.get(&cohort_key).cloned() {
        if existing_gate.primer_active() && !existing_gate.is_released() {
            return WarmFirstRole::Follower(existing_gate);
        }
        guard.remove(&cohort_key);
    }

    let gate = Arc::new(CohortGate::new());
    guard.insert(cohort_key, gate.clone());
    WarmFirstRole::Primer(gate)
}

async fn cleanup_cohort_gate(
    registry: &Arc<Mutex<HashMap<CohortKey, Arc<CohortGate>>>>,
    cohort_key: &CohortKey,
    gate: &Arc<CohortGate>,
) {
    let mut guard = registry.lock().await;
    let should_remove = guard
        .get(cohort_key)
        .is_some_and(|registered_gate| Arc::ptr_eq(registered_gate, gate));
    if should_remove {
        guard.remove(cohort_key);
    }
}

#[cfg(test)]
#[path = "../tests/unit/intercepts_tests.rs"]
mod tests;
