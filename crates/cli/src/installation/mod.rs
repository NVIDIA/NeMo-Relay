// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transactional installation primitives.

use std::path::PathBuf;

use crate::agents::CodingAgent;

pub(crate) mod generation;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum IntegrationHost {
    Codex,
    ClaudeCode,
    Hermes,
    All,
}

impl IntegrationHost {
    pub(crate) const fn agent(self) -> Option<CodingAgent> {
        match self {
            Self::Codex => Some(CodingAgent::Codex),
            Self::ClaudeCode => Some(CodingAgent::ClaudeCode),
            Self::Hermes => Some(CodingAgent::Hermes),
            Self::All => None,
        }
    }

    pub(crate) const fn as_arg(self) -> &'static str {
        match self.agent() {
            Some(agent) => agent.install_arg(),
            None => "all",
        }
    }

    pub(crate) const fn label(self) -> &'static str {
        match self.agent() {
            Some(agent) => agent.label(),
            None => "all",
        }
    }

    pub(crate) const fn executable(self) -> Option<&'static str> {
        match self.agent() {
            Some(agent) => Some(agent.executable()),
            None => None,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct InstallRequest {
    pub(crate) host: IntegrationHost,
    pub(crate) install_dir: Option<PathBuf>,
    pub(crate) force: bool,
    pub(crate) dry_run: bool,
    pub(crate) skip_doctor: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct UninstallRequest {
    pub(crate) host: IntegrationHost,
    pub(crate) install_dir: Option<PathBuf>,
    pub(crate) dry_run: bool,
}
