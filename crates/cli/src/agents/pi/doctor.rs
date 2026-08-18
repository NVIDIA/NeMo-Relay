// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Diagnostics for the pi integration.
//!
//! Codex and Claude Code can be checked by inspecting files NeMo Relay wrote
//! (generated hook config, a settings base URL). pi's hooks live inside an
//! extension the user loads, so the only thing checkable from here is whether
//! that extension is discoverable.

use std::path::PathBuf;

use super::launch::PI_EXTENSION_PATH_ENV;

/// Human-readable hook status for `nemo-relay doctor`.
pub(crate) fn hook_status() -> Result<String, String> {
    match extension_location() {
        Some(path) => Ok(format!(
            "pi extension resolved at {} (hooks are emitted by the extension, not by pi itself)",
            path.display()
        )),
        None => Ok(format!(
            "pi extension not located; set {PI_EXTENSION_PATH_ENV}, or install the extension with \
             `pi install <source>` or into an auto-discovered directory \
             (`~/.pi/agent/extensions/`, `.pi/extensions/`)"
        )),
    }
}

/// Whether the extension entry point can be found.
pub(crate) fn extension_configured() -> bool {
    extension_location().is_some()
}

fn extension_location() -> Option<PathBuf> {
    std::env::var_os(PI_EXTENSION_PATH_ENV)
        .map(PathBuf::from)
        .filter(|path| path.exists())
}
