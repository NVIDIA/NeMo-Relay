// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::convert::nemo_relay_string_free;
use crate::error::{NemoRelayStatus::*, last_error_message};
use nemo_relay_types::Json;
use serde_json::json;
use std::ffi::{CStr, CString};
use std::ptr;

fn snapshot(source: u32, input: &Json, selection: u32, index: i64) -> Option<Json> {
    let input = CString::new(input.to_string()).unwrap();
    let mut out = ptr::null_mut();
    let status = unsafe {
        nemo_relay_inspect_llm_content_json(source, input.as_ptr(), selection, index, &mut out)
    };
    drop(input);
    assert_eq!(status, NemoRelayStatus::Ok, "{:?}", last_error_message());
    if out.is_null() {
        return None;
    }
    let result = serde_json::from_slice(unsafe { CStr::from_ptr(out) }.to_bytes()).unwrap();
    unsafe { nemo_relay_string_free(out) };
    Some(result)
}

#[test]
fn request_selections_distinguish_absence_empty_and_textless() {
    let request = json!({"messages":[
        {"role":"developer","content":"system"}, {"role":"user","content":""},
        {"role":"assistant","content":[{"type":"file","file":{"id":"f"}}]}
    ]});
    for (selection, index, text) in [
        (
            NEMO_RELAY_CONTENT_SELECT_SYSTEM_PROMPT,
            0,
            json!(["system"]),
        ),
        (NEMO_RELAY_CONTENT_SELECT_LAST_USER_MESSAGE, 0, json!([""])),
        (NEMO_RELAY_CONTENT_SELECT_MESSAGE, -2, json!([""])),
        (NEMO_RELAY_CONTENT_SELECT_LAST_MESSAGE, 0, json!([])),
    ] {
        let view = snapshot(
            NEMO_RELAY_CONTENT_SOURCE_REQUEST,
            &request,
            selection,
            index,
        )
        .unwrap();
        assert_eq!(view["text_parts"], text);
        assert_eq!(view["parts"].as_array().unwrap().len(), 1);
    }
    assert!(
        snapshot(
            NEMO_RELAY_CONTENT_SOURCE_REQUEST,
            &request,
            NEMO_RELAY_CONTENT_SELECT_INSTRUCTIONS,
            0
        )
        .is_none()
    );
    assert!(
        snapshot(
            NEMO_RELAY_CONTENT_SOURCE_REQUEST,
            &json!({}),
            NEMO_RELAY_CONTENT_SELECT_LAST_MESSAGE,
            0
        )
        .is_none()
    );
    let request =
        json!({"instructions":"override","messages":[{"role":"system","content":"system"}]});
    for selection in [
        NEMO_RELAY_CONTENT_SELECT_INSTRUCTIONS,
        NEMO_RELAY_CONTENT_SELECT_SYSTEM_PROMPT,
    ] {
        assert_eq!(
            snapshot(NEMO_RELAY_CONTENT_SOURCE_REQUEST, &request, selection, 0).unwrap()["text_parts"],
            json!(["override"])
        );
    }
}

#[test]
fn snapshot_has_owned_ordered_typed_entries_and_exact_payloads() {
    let output = json!({"message":[{"type":"text","text":"a\0b"},{"type":"refusal","refusal":""}],
        "tool_calls":[{"id":"c","name":"f","arguments":{"n":u64::MAX}}]});
    let view = snapshot(
        NEMO_RELAY_CONTENT_SOURCE_RESPONSE,
        &output,
        NEMO_RELAY_CONTENT_SELECT_OUTPUT,
        0,
    )
    .unwrap();
    assert_eq!(view["text_parts"], json!(["a\0b", ""]));
    assert_eq!(view["parts"][0]["kind"], "text");
    assert_eq!(view["parts"][1]["kind"], "refusal");
    assert_eq!(view["parts"][2]["kind"], "tool_call");
    assert_eq!(
        view["parts"][2]["location"],
        json!({"root":"response","field":"tool_calls","index":0})
    );
    assert_eq!(
        view["parts"][2]["value"]["arguments"],
        json!({"kind":"json","value":{"n":u64::MAX}})
    );
    let native =
        json!({"role":"provider_native","provider":"custom","kind":"opaque","value":{"x":1}});
    let view = snapshot(
        NEMO_RELAY_CONTENT_SOURCE_MESSAGE,
        &native,
        NEMO_RELAY_CONTENT_SELECT_ALL,
        0,
    )
    .unwrap();
    assert_eq!(view["parts"][0]["kind"], "provider_native");
    assert_eq!(view["parts"][0]["value"]["value"], json!({"x":1}));
    assert_eq!(
        snapshot(
            NEMO_RELAY_CONTENT_SOURCE_CONTENT,
            &json!([]),
            NEMO_RELAY_CONTENT_SELECT_ALL,
            0
        )
        .unwrap(),
        json!({"parts":[],"text_parts":[]})
    );
}

#[test]
fn invalid_inputs_clear_the_output_and_report_the_error() {
    let source = NEMO_RELAY_CONTENT_SOURCE_REQUEST;
    let select = NEMO_RELAY_CONTENT_SELECT_MESSAGE;
    for (kind, input, selection, index, status) in [
        (source, ptr::null(), select, 0, NullPointer),
        (source, c"{".as_ptr(), select, 0, InvalidJson),
        (source, c"[]".as_ptr(), select, 0, InvalidJson),
        (source, c"{}".as_ptr(), select, i64::MIN, InvalidArg),
        (
            source,
            c"{}".as_ptr(),
            NEMO_RELAY_CONTENT_SELECT_OUTPUT,
            0,
            InvalidArg,
        ),
        (source, c"{}".as_ptr(), u32::MAX, 0, InvalidArg),
        (u32::MAX, c"{}".as_ptr(), select, 0, InvalidArg),
        (source, c"\xff".as_ptr(), select, 0, InvalidUtf8),
    ] {
        let mut out = ptr::dangling_mut();
        assert_eq!(
            unsafe { nemo_relay_inspect_llm_content_json(kind, input, selection, index, &mut out) },
            status
        );
        assert!(out.is_null());
        assert!(last_error_message().is_some());
    }
    assert_eq!(
        unsafe {
            nemo_relay_inspect_llm_content_json(source, c"{}".as_ptr(), select, 0, ptr::null_mut())
        },
        NullPointer
    );
}
