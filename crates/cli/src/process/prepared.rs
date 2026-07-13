// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

/// Fully resolved child-process launch plan produced by one agent integration.
pub(crate) struct PreparedAgentLaunch {
    pub(crate) argv: Vec<String>,
    pub(crate) host_index: usize,
    pub(crate) env: Vec<(String, String)>,
    pub(crate) temp_dirs: Vec<PathBuf>,
    pub(crate) notes: Vec<String>,
}
