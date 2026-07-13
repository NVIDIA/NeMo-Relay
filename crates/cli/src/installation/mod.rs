// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transactional installation primitives.

use std::path::PathBuf;

pub(crate) mod generation;
pub(crate) mod operation_lock;

#[derive(Debug, Clone)]
pub(crate) struct InstallRequest {
    pub(crate) install_dir: Option<PathBuf>,
    pub(crate) force: bool,
    pub(crate) dry_run: bool,
    pub(crate) skip_doctor: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct UninstallRequest {
    pub(crate) install_dir: Option<PathBuf>,
    pub(crate) dry_run: bool,
}
