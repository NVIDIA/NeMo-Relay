// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

pub(crate) fn hook_status(hooks_path: Option<&Path>) -> Result<String, String> {
    match hooks_path {
        Some(path) => super::diagnose_persistent(path).map_err(|error| {
            format!("persistent MCP/hooks: {error}; run `nemo-relay install hermes --force`")
        }),
        None => Ok("hooks: injected through an isolated HERMES_HOME during run".into()),
    }
}
