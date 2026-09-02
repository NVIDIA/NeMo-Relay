// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transactional installation primitives.

use std::path::PathBuf;

pub(crate) mod generation;
pub(crate) mod marketplace;
pub(crate) mod operation_lock;

#[derive(Debug, Clone)]
pub(crate) struct InstallRequest {
    pub(crate) install_dir: Option<PathBuf>,
    pub(crate) force: bool,
    pub(crate) dry_run: bool,
    pub(crate) skip_doctor: bool,
    /// Experimental: rewrite pre-install Codex thread history onto the Relay provider.
    pub(crate) migrate_history: bool,
    /// Codex thread database to migrate, when it is not the default generation.
    pub(crate) history_database: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub(crate) struct UninstallRequest {
    pub(crate) install_dir: Option<PathBuf>,
    pub(crate) force: bool,
    pub(crate) dry_run: bool,
    /// Skip the reversal that a recorded history migration would otherwise infer.
    pub(crate) skip_history_migration: bool,
    /// Codex thread database to restore, overriding the one the journal recorded.
    pub(crate) history_database: Option<PathBuf>,
}
