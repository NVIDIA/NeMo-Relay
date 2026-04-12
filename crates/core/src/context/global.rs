// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, RwLock};

use crate::context::state::NemoFlowContextState;

static GLOBAL_CONTEXT: std::sync::OnceLock<Arc<RwLock<NemoFlowContextState>>> =
    std::sync::OnceLock::new();

pub fn global_context() -> Arc<RwLock<NemoFlowContextState>> {
    GLOBAL_CONTEXT
        .get_or_init(|| Arc::new(RwLock::new(NemoFlowContextState::new())))
        .clone()
}
