// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lightweight env-gated debug output for ACG planning and translation.

use std::sync::OnceLock;

use serde_json::{Map, Value};

const ACG_DEBUG_ENV: &str = "NEMO_RELAY_ACG_DEBUG";

fn env_flag_enabled(value: &str) -> bool {
    !matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "" | "0" | "false" | "off" | "no"
    )
}

pub(crate) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var(ACG_DEBUG_ENV)
            .ok()
            .is_some_and(|value| env_flag_enabled(&value))
    })
}

pub(crate) fn emit(event: &str, payload: Value) {
    if !enabled() {
        return;
    }

    let mut body = Map::new();
    body.insert("event".to_string(), Value::String(event.to_string()));

    match payload {
        Value::Object(map) => body.extend(map),
        other => {
            body.insert("payload".to_string(), other);
        }
    }

    eprintln!("nemo-relay-adaptive acg-debug {}", Value::Object(body));
}

#[cfg(test)]
#[path = "../../tests/unit/acg/debug_tests.rs"]
mod tests;
