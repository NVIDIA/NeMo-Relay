// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay_plugin::{PluginRuntime, ScopeType};
use serde_json::json;

use crate::config::RuntimeConfig;

pub(crate) fn emit_configured_runtime_events(
    runtime: &PluginRuntime,
    tag: &str,
    config: &RuntimeConfig,
) -> nemo_relay_plugin::Result<()> {
    let _current_scope = runtime.current_scope()?;
    if config.emit_marks {
        runtime.emit_mark(
            "example.native.request.seen",
            Some(&json!({ "tag": tag })),
            None,
        )?;
    }

    let mut scope = runtime.scope(
        "example.native.request",
        ScopeType::Custom,
        Some(&json!({ "tag": tag })),
        None,
        None,
    )?;
    scope.close(Some(&json!({ "done": true })), None)?;

    if config.emit_isolated_scope {
        let isolated = runtime.create_scope_stack()?;
        isolated.with_current(|| {
            if config.emit_marks {
                runtime.emit_mark(
                    "example.native.isolated.mark",
                    Some(&json!({ "tag": tag })),
                    None,
                )?;
            }
            let mut scope = runtime.scope(
                "example.native.isolated.scope",
                ScopeType::Custom,
                None,
                Some(&json!({ "visibility": "isolated" })),
                None,
            )?;
            scope.close(Some(&json!({ "done": true })), None)
        })?;
    }

    Ok(())
}
