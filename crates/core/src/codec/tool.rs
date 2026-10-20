// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared content and failure fields for tool results from any framework.

use crate::json::Json;

/// Schema name for version one of the tool-result JSON format.
pub const TOOL_RESULT_SCHEMA: &str = "nemo.relay.tool_result@1";

/// Read content and failure status from a valid tool-result wrapper.
pub(crate) fn projected_tool_result(value: &Json) -> Option<(&Json, bool)> {
    if value.get("schema")?.as_str()? != TOOL_RESULT_SCHEMA {
        return None;
    }
    let content = value.get("content")?;
    if !(content.is_null() || content.is_string() || content.is_array()) {
        return None;
    }
    let is_error = value.get("is_error")?.as_bool()?;
    value.get("result")?;
    Some((content, is_error))
}
