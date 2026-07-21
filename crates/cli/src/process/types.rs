// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

use crate::agents::CodingAgent;

#[derive(Debug, Clone)]
pub(crate) struct RunOverrides {
    pub(crate) agent: Option<CodingAgent>,
    pub(crate) config: Option<PathBuf>,
    pub(crate) openai_base_url: Option<String>,
    pub(crate) anthropic_base_url: Option<String>,
    pub(crate) session_metadata: Option<String>,
    pub(crate) plugin_config_path: Option<PathBuf>,
    pub(crate) dry_run: bool,
    pub(crate) print: bool,
    pub(crate) command: Vec<String>,
}
