// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Read-only inspection of normalized LLM content.

use libc::c_char;
use nemo_relay_types::codec::content::{ContentSelection, ContentSnapshot, ContentSource};
use serde::de::DeserializeOwned;

use crate::convert::{c_str_to_string, str_to_c_string};
use crate::error::{NemoRelayStatus, clear_last_error, set_last_error};

/// Normalized LLM request.
pub const NEMO_RELAY_CONTENT_SOURCE_REQUEST: u32 = 1;
/// Normalized LLM response.
pub const NEMO_RELAY_CONTENT_SOURCE_RESPONSE: u32 = 2;
/// Standalone normalized message.
pub const NEMO_RELAY_CONTENT_SOURCE_MESSAGE: u32 = 3;
/// Standalone message content (string or parts array).
pub const NEMO_RELAY_CONTENT_SOURCE_CONTENT: u32 = 4;

/// Final supplied request message.
pub const NEMO_RELAY_CONTENT_SELECT_LAST_MESSAGE: u32 = 1;
/// Request message at the signed index. Negative indices count from the end.
pub const NEMO_RELAY_CONTENT_SELECT_MESSAGE: u32 = 2;
/// Top-level request instructions.
pub const NEMO_RELAY_CONTENT_SELECT_INSTRUCTIONS: u32 = 3;
/// Request instructions or the first system/developer message.
pub const NEMO_RELAY_CONTENT_SELECT_SYSTEM_PROMPT: u32 = 4;
/// Last user message in the request.
pub const NEMO_RELAY_CONTENT_SELECT_LAST_USER_MESSAGE: u32 = 5;
/// Current normalized response output.
pub const NEMO_RELAY_CONTENT_SELECT_OUTPUT: u32 = 6;
/// All parts of a standalone message or content value.
pub const NEMO_RELAY_CONTENT_SELECT_ALL: u32 = 7;

fn parse<T: DeserializeOwned>(input: &str) -> Result<T, NemoRelayStatus> {
    serde_json::from_str(input).map_err(|_| {
        set_last_error("invalid normalized content JSON for the supplied source kind");
        NemoRelayStatus::InvalidJson
    })
}

fn inspect(
    source: u32,
    input: &str,
    selection: u32,
    index: i64,
) -> Result<Option<ContentSnapshot>, NemoRelayStatus> {
    let selection = match selection {
        NEMO_RELAY_CONTENT_SELECT_LAST_MESSAGE => ContentSelection::LastMessage,
        NEMO_RELAY_CONTENT_SELECT_MESSAGE => ContentSelection::Message(index),
        NEMO_RELAY_CONTENT_SELECT_INSTRUCTIONS => ContentSelection::Instructions,
        NEMO_RELAY_CONTENT_SELECT_SYSTEM_PROMPT => ContentSelection::SystemPrompt,
        NEMO_RELAY_CONTENT_SELECT_LAST_USER_MESSAGE => ContentSelection::LastUserMessage,
        NEMO_RELAY_CONTENT_SELECT_OUTPUT => ContentSelection::Output,
        NEMO_RELAY_CONTENT_SELECT_ALL => ContentSelection::All,
        _ => {
            set_last_error("unknown content selection");
            return Err(NemoRelayStatus::InvalidArg);
        }
    };
    let project = |source: ContentSource<'_>| {
        selection
            .select(source)
            .map(|view| view.map(|v| v.snapshot()))
            .map_err(|error| {
                set_last_error(&error.to_string());
                NemoRelayStatus::InvalidArg
            })
    };
    match source {
        NEMO_RELAY_CONTENT_SOURCE_REQUEST
        | NEMO_RELAY_CONTENT_SOURCE_RESPONSE
        | NEMO_RELAY_CONTENT_SOURCE_MESSAGE
            if !input.trim_start().starts_with('{') =>
        {
            set_last_error("normalized request, response, or message must be a JSON object");
            Err(NemoRelayStatus::InvalidJson)
        }
        NEMO_RELAY_CONTENT_SOURCE_REQUEST => project(ContentSource::Request(&parse(input)?)),
        NEMO_RELAY_CONTENT_SOURCE_RESPONSE => project(ContentSource::Response(&parse(input)?)),
        NEMO_RELAY_CONTENT_SOURCE_MESSAGE => project(ContentSource::Message(&parse(input)?)),
        NEMO_RELAY_CONTENT_SOURCE_CONTENT => project(ContentSource::Content(&parse(input)?)),
        _ => {
            set_last_error("unknown content source");
            Err(NemoRelayStatus::InvalidArg)
        }
    }
}

/// Inspect a normalized annotation, using NEMO_RELAY_CONTENT_SOURCE_* and
/// NEMO_RELAY_CONTENT_SELECT_* constants. Only SELECT_MESSAGE reads index.
/// Success returns an owned JSON object with ordered parts and text_parts;
/// an absent selection succeeds with a null output. An empty selection of
/// existing content still returns a snapshot. Free a snapshot with nemo_relay_string_free.
/// An invalid selection/index returns InvalidArg. On failure, *out is null
/// and nemo_relay_last_error describes the error on the calling thread.
///
/// # Safety
/// input must be readable through its NUL terminator for the call. out must
/// point to writable pointer storage distinct from input.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_inspect_llm_content_json(
    source: u32,
    input: *const c_char,
    selection: u32,
    index: i64,
    out: *mut *mut c_char,
) -> NemoRelayStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("null content snapshot output pointer");
        return NemoRelayStatus::NullPointer;
    }
    unsafe { *out = std::ptr::null_mut() };
    let snapshot =
        match c_str_to_string(input).and_then(|json| inspect(source, &json, selection, index)) {
            Ok(Some(snapshot)) => snapshot,
            Ok(None) => return NemoRelayStatus::Ok,
            Err(status) => return status,
        };
    match serde_json::to_string(&snapshot) {
        Ok(json) => {
            unsafe { *out = str_to_c_string(&json) };
            NemoRelayStatus::Ok
        }
        Err(error) => {
            set_last_error(&error.to_string());
            NemoRelayStatus::Internal
        }
    }
}

#[cfg(test)]
#[path = "../../tests/coverage/content_tests.rs"]
mod tests;
