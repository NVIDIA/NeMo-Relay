// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};

static FREES: AtomicUsize = AtomicUsize::new(0);

unsafe extern "C" fn free_state(_: *mut libc::c_void) {
    FREES.fetch_add(1, Ordering::SeqCst);
}

unsafe extern "C" fn decode(_: *mut libc::c_void, _: *const FfiLLMRequest) -> *mut c_char {
    CString::new(r#"{"model":"custom","messages":[]}"#)
        .unwrap()
        .into_raw()
}

unsafe extern "C" fn encode(
    _: *mut libc::c_void,
    _: *const c_char,
    original: *const FfiLLMRequest,
) -> *mut c_char {
    json_to_c_string(&unsafe { &*original }.0.content)
}

unsafe extern "C" fn decode_response(_: *mut libc::c_void, _: *const c_char) -> *mut c_char {
    CString::new(r#"{"model":"custom-response"}"#)
        .unwrap()
        .into_raw()
}

unsafe fn value(ptr: *mut c_char) -> Json {
    assert!(
        !ptr.is_null(),
        "{}",
        unsafe { CStr::from_ptr(nemo_relay_last_error()) }.to_string_lossy()
    );
    let value = serde_json::from_str(&unsafe { CStr::from_ptr(ptr) }.to_string_lossy()).unwrap();
    unsafe { nemo_relay_string_free(ptr) };
    value
}

#[test]
fn custom_codec_shares_callback_state_until_both_directions_are_released() {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    FREES.store(0, Ordering::SeqCst);
    unsafe {
        let callbacks = NemoRelayCodecCallbacks {
            decode: Some(decode),
            encode: Some(encode),
            decode_response: Some(decode_response),
            user_data: ptr::null_mut(),
            free_fn: Some(free_state),
        };
        let codec = nemo_relay_codec_new(NEMO_RELAY_CODEC_CUSTOM, &callbacks);
        assert!(!codec.is_null());
        let original = FfiLLMRequest(LlmRequest {
            headers: serde_json::from_value(json!({"x-test":"kept"})).unwrap(),
            content: json!({"opaque":true}),
        });
        assert_eq!(
            value(nemo_relay_codec_decode(codec, &original))["model"],
            "custom"
        );
        let encoded = value(nemo_relay_codec_encode(
            codec,
            c"{\"model\":\"new\"}".as_ptr(),
            &original,
        ));
        assert_eq!(encoded["headers"]["x-test"], "kept");
        assert_eq!(encoded["content"], json!({"opaque":true}));
        assert_eq!(
            value(nemo_relay_codec_decode_response(codec, c"{}".as_ptr()))["model"],
            "custom-response"
        );
        let request = (&*codec).codec.clone();
        let response = (&*codec).response_codec.clone();
        nemo_relay_codec_free(codec);
        assert_eq!(FREES.load(Ordering::SeqCst), 0);
        drop(request);
        assert_eq!(FREES.load(Ordering::SeqCst), 0);
        drop(response);
        assert_eq!(FREES.load(Ordering::SeqCst), 1);
        let invalid = NemoRelayCodecCallbacks {
            encode: None,
            ..callbacks
        };
        assert!(nemo_relay_codec_new(NEMO_RELAY_CODEC_CUSTOM, &invalid).is_null());
        assert_eq!(FREES.load(Ordering::SeqCst), 2);
        assert!(nemo_relay_codec_new(NEMO_RELAY_CODEC_OPENAI_CHAT, &callbacks).is_null());
        assert_eq!(FREES.load(Ordering::SeqCst), 3);
    }
}

#[test]
fn owned_codec_operations_validate_direction_and_json_without_losing_headers() {
    unsafe {
        let codec = nemo_relay_codec_new(NEMO_RELAY_CODEC_OPENAI_CHAT, ptr::null());
        let original = FfiLLMRequest(LlmRequest {
            headers: serde_json::from_value(json!({"authorization":"kept"})).unwrap(),
            content: json!({"model":"before","messages":[],"vendor_hint":0.5}),
        });
        let mut annotated = value(nemo_relay_codec_decode(codec, &original));
        assert_eq!(annotated["model"], "before");
        annotated["model"] = json!("after");
        let changed = CString::new(annotated.to_string()).unwrap();
        let encoded = value(nemo_relay_codec_encode(codec, changed.as_ptr(), &original));
        assert_eq!(encoded["content"]["model"], "after");
        assert_eq!(encoded["content"]["vendor_hint"], 0.5);
        assert_eq!(encoded["headers"]["authorization"], "kept");
        assert!(nemo_relay_codec_encode(codec, c"{".as_ptr(), &original).is_null());
        assert!(nemo_relay_codec_decode_response(codec, c"{".as_ptr()).is_null());
        nemo_relay_codec_free(codec);
        let callbacks = NemoRelayCodecCallbacks {
            decode: None,
            encode: None,
            decode_response: Some(decode_response),
            user_data: ptr::null_mut(),
            free_fn: None,
        };
        let response_only = nemo_relay_codec_new(NEMO_RELAY_CODEC_CUSTOM, &callbacks);
        assert!(nemo_relay_codec_decode(response_only, &original).is_null());
        nemo_relay_codec_free(response_only);
        let oci = nemo_relay_codec_new(NEMO_RELAY_CODEC_OCI_GENAI, ptr::null());
        assert!(!oci.is_null());
        nemo_relay_codec_free(oci);
    }
}

#[test]
fn manual_llm_recorder_retains_independent_ownership() {
    unsafe {
        let handle = Box::into_raw(Box::new(FfiLLMHandle(
            LlmHandle::builder().name("manual-recorder").build(),
        )));
        let mut recorder = ptr::null_mut();
        assert_eq!(
            nemo_relay_llm_handle_retain_optimization_recorder(handle, &mut recorder),
            NemoRelayStatus::Ok
        );
        assert!(!recorder.is_null());
        let retained = nemo_relay_llm_optimization_recorder_retain(recorder);
        assert!(!retained.is_null());
        nemo_relay_llm_handle_free(handle);
        nemo_relay_llm_optimization_recorder_free(recorder);
        let mut accepted = false;
        assert_eq!(
            nemo_relay_llm_optimization_record(
                retained,
                c"{\"producer\":\"manual\",\"kind\":\"custom\",\"applied\":true}".as_ptr(),
                &mut accepted
            ),
            NemoRelayStatus::Ok
        );
        assert!(accepted);
        nemo_relay_llm_optimization_recorder_free(retained);
    }
}

#[test]
fn manual_llm_recorder_validates_handle_and_output() {
    unsafe {
        assert_eq!(
            nemo_relay_llm_handle_retain_optimization_recorder(ptr::null(), ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        let mut recorder = ptr::dangling_mut();
        assert_eq!(
            nemo_relay_llm_handle_retain_optimization_recorder(ptr::null(), &mut recorder),
            NemoRelayStatus::NullPointer
        );
        assert!(recorder.is_null());
        assert!(!nemo_relay_last_error().is_null());
    }
}
