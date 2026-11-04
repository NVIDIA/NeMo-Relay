// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Price estimation supplied independently from runtime execution.

use super::response::{CostEstimate, Usage};

/// Immutable price-estimation capability for a provider/model and token usage.
pub trait CostEstimator: Send + Sync {
    /// Estimate cost using one consistent resolver snapshot.
    fn estimate_cost_for_provider(
        &self,
        provider: Option<&str>,
        model: &str,
        usage: &Usage,
    ) -> Option<CostEstimate>;
}
