// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Admission, callback lifetime, and recovery through the public C entry points.

use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

#[derive(Default)]
struct Probe {
    calls: AtomicUsize,
    frees: AtomicUsize,
    contexts: Mutex<Vec<Option<Json>>>,
}

impl Probe {
    fn pointer(&self) -> *mut libc::c_void {
        std::ptr::from_ref(self).cast_mut().cast()
    }
}

unsafe extern "C" fn release_probe(data: *mut libc::c_void) {
    let probe = unsafe { Arc::from_raw(data.cast::<Probe>()) };
    probe.frees.fetch_add(1, Ordering::SeqCst);
}

unsafe extern "C" fn context_provider(
    data: *mut libc::c_void,
    _request: *const c_char,
) -> *mut c_char {
    unsafe { &*data.cast::<Probe>() }
        .calls
        .fetch_add(1, Ordering::SeqCst);
    cstring(r#"{"provider":true}"#).into_raw()
}

unsafe extern "C" fn capture_context(
    data: *mut libc::c_void,
    _name: *const c_char,
    request: *const c_char,
    context: NemoRelayLlmExecutionContext,
    next: NemoRelayLlmExecNextFn,
    next_data: *mut libc::c_void,
) -> *mut c_char {
    let probe = unsafe { &*data.cast::<Probe>() };
    probe.calls.fetch_add(1, Ordering::SeqCst);
    let value = if context.execution_context_json.is_null() {
        None
    } else {
        // Copy while the borrowed callback pointer is valid; inspect after return.
        // Preserve presence even on a decoding failure, so omission assertions
        // cannot accidentally accept a malformed non-null callback value.
        Some(
            serde_json::from_slice(
                unsafe { CStr::from_ptr(context.execution_context_json) }.to_bytes(),
            )
            .unwrap_or(Json::Null),
        )
    };
    probe.contexts.lock().unwrap().push(value);
    unsafe { next(request, next_data) }
}

#[derive(Default)]
struct Resources {
    provider: Arc<Probe>,
    codec: Arc<Probe>,
    stream: Arc<Probe>,
}

fn owned_probe(probe: &Arc<Probe>) -> *mut libc::c_void {
    Arc::into_raw(Arc::clone(probe)).cast_mut().cast()
}

/// Supply owned callbacks on both paths, and drain/close successful streams.
unsafe fn execute(
    streaming: bool,
    context: Option<&CStr>,
    skip_stages: u32,
    resources: &Resources,
) -> NemoRelayStatus {
    let name = cstring("ffi-context-coverage");
    let request = cstring(r#"{"headers":{},"content":{}}"#);
    let context = context.map_or(ptr::null(), CStr::as_ptr);
    let status = if streaming {
        let mut stream = ptr::null_mut();
        let status = unsafe {
            api::nemo_relay_llm_stream_call_execute(
                name.as_ptr(),
                request.as_ptr(),
                Some(context_provider),
                None,
                owned_probe(&resources.provider),
                Some(release_probe),
                Some(collector_cb),
                Some(finalizer_cb),
                owned_probe(&resources.stream),
                Some(release_probe),
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                Some(codec_decode_cb),
                Some(codec_encode_cb),
                owned_probe(&resources.codec),
                Some(release_probe),
                ptr::null(),
                ptr::null(),
                skip_stages,
                context,
                ptr::null(),
                &mut stream,
            )
        };
        if status == NemoRelayStatus::Ok {
            let mut chunk = ptr::null_mut();
            assert_eq!(
                unsafe { api::nemo_relay_stream_next(stream, &mut chunk) },
                1
            );
            assert_eq!(unsafe { returned_json(chunk) }, json!({"provider": true}));
            assert_eq!(
                unsafe { api::nemo_relay_stream_next(stream, &mut chunk) },
                0
            );
            assert_status!(
                unsafe { api::nemo_relay_stream_close(stream) },
                NemoRelayStatus::Ok
            );
            unsafe { api::nemo_relay_stream_free(stream) };
        } else {
            assert!(stream.is_null());
        }
        status
    } else {
        let mut output = ptr::null_mut();
        let status = unsafe {
            api::nemo_relay_llm_call_execute(
                name.as_ptr(),
                request.as_ptr(),
                context_provider,
                owned_probe(&resources.provider),
                Some(release_probe),
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                Some(codec_decode_cb),
                Some(codec_encode_cb),
                owned_probe(&resources.codec),
                Some(release_probe),
                ptr::null(),
                ptr::null(),
                skip_stages,
                context,
                ptr::null(),
                &mut output,
            )
        };
        if status == NemoRelayStatus::Ok {
            assert_eq!(unsafe { returned_json(output) }, json!({"provider": true}));
        } else {
            assert!(output.is_null());
        }
        status
    };
    // Runtime tasks may release their captures after the C call returns.
    // Each callback owns an Arc so even timeout failures remain safe.
    let deadline = Instant::now() + Duration::from_secs(5);
    while resources.provider.frees.load(Ordering::SeqCst) == 0
        || resources.codec.frees.load(Ordering::SeqCst) == 0
        || (streaming && resources.stream.frees.load(Ordering::SeqCst) == 0)
    {
        assert!(Instant::now() < deadline, "callbacks were not released");
        std::thread::sleep(Duration::from_millis(1));
    }
    status
}

struct CaptureRegistration<'a> {
    streaming: bool,
    name: CString,
    _probe: &'a Probe,
}

impl Drop for CaptureRegistration<'_> {
    fn drop(&mut self) {
        // Cleanup must not panic while a failed assertion unwinds.
        unsafe {
            if self.streaming {
                let _ =
                    api::nemo_relay_deregister_llm_stream_execution_intercept(self.name.as_ptr());
            } else {
                let _ = api::nemo_relay_deregister_llm_execution_intercept(self.name.as_ptr());
            }
        }
    }
}

unsafe fn register_capture<'a>(
    streaming: bool,
    name: &CStr,
    probe: &'a Probe,
) -> CaptureRegistration<'a> {
    let status = if streaming {
        unsafe {
            api::nemo_relay_register_llm_stream_execution_intercept(
                name.as_ptr(),
                1,
                Some(capture_context),
                None,
                probe.pointer(),
                None,
            )
        }
    } else {
        unsafe {
            api::nemo_relay_register_llm_execution_intercept(
                name.as_ptr(),
                1,
                capture_context,
                probe.pointer(),
                None,
            )
        }
    };
    assert_status!(status, NemoRelayStatus::Ok);
    CaptureRegistration {
        streaming,
        name: name.to_owned(),
        _probe: probe,
    }
}

#[test]
fn ffi_llm_context_rejection_releases_resources_and_recovers() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();
    let mut invalid = vec![
        (vec![0xff], NemoRelayStatus::InvalidJson, "UTF-8"),
        (b"{".to_vec(), NemoRelayStatus::InvalidJson, "JSON"),
        (
            br#"{"data":{"value":"\"#.to_vec(),
            NemoRelayStatus::InvalidJson,
            "JSON",
        ),
        (
            br#"{"data":{"value":"\uZZZZ"}}"#.to_vec(),
            NemoRelayStatus::InvalidJson,
            "JSON",
        ),
        (
            br#"{"data":{"value":1e999}}"#.to_vec(),
            NemoRelayStatus::InvalidJson,
            "JSON",
        ),
    ];
    for input in ["null", "[]", r#"{"data":null}"#, r#"{"secret-sentinel":1}"#] {
        invalid.push((
            input.as_bytes().to_vec(),
            NemoRelayStatus::InvalidArg,
            "JSON Schema",
        ));
    }
    for text in [
        "x".repeat(16_384),
        "\\u0000".repeat(3000),
        "\\u20ac".repeat(5500),
    ] {
        invalid.push((
            format!(r#"{{"data":{{"secret-sentinel":"{text}"}}}}"#).into_bytes(),
            NemoRelayStatus::InvalidArg,
            "16384",
        ));
    }
    invalid.push((
        format!(
            r#"{{"data":{{"value":{}null{}}}}}"#,
            "[".repeat(63),
            "]".repeat(63)
        )
        .into_bytes(),
        NemoRelayStatus::InvalidArg,
        "64 container levels",
    ));
    unsafe {
        let stack = fresh_scope_stack();
        for streaming in [false, true] {
            let intercept = Probe::default();
            let name = cstring("context-admission");
            let _registration = register_capture(streaming, &name, &intercept);
            for (input, expected_status, message) in &invalid {
                for skip in [0, api::NEMO_RELAY_SKIP_EXECUTION_INTERCEPTS] {
                    let resources = Resources::default();
                    let before = intercept.calls.load(Ordering::SeqCst);
                    let input = CString::new(input.clone()).unwrap();
                    assert_status!(
                        execute(streaming, Some(&input), skip, &resources),
                        *expected_status
                    );
                    let error = read_last_error().unwrap();
                    assert!(error.contains(message), "{error}");
                    assert!(!error.contains("secret-sentinel"));
                    assert_eq!(intercept.calls.load(Ordering::SeqCst), before);
                    assert_eq!(resources.provider.calls.load(Ordering::SeqCst), 0);
                    assert_eq!(resources.provider.frees.load(Ordering::SeqCst), 1);
                    assert_eq!(resources.codec.frees.load(Ordering::SeqCst), 1);
                    assert_eq!(
                        resources.stream.frees.load(Ordering::SeqCst),
                        usize::from(streaming)
                    );
                    let recovery = cstring(r#"{"data":{"recovered":true}}"#);
                    let resources = Resources::default();
                    assert_status!(
                        execute(streaming, Some(&recovery), 0, &resources),
                        NemoRelayStatus::Ok
                    );
                    assert_eq!(resources.provider.calls.load(Ordering::SeqCst), 1);
                    assert_eq!(
                        intercept.contexts.lock().unwrap().last(),
                        Some(&Some(json!({"data":{"recovered":true}})))
                    );
                }
            }
        }
        nemo_relay_scope_stack_free(stack);
    }
}

#[test]
fn ffi_llm_context_preserves_shapes_unicode_and_exact_boundaries() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();
    let at_limit = format!(
        r#"{{"data":{{"value":"{}"}}}}"#,
        "x".repeat(16_384 - r#"{"data":{"value":""}}"#.len())
    );
    let at_depth = format!(
        r#"{{"data":{{"value":{}null{}}}}}"#,
        "[".repeat(62),
        "]".repeat(62)
    );
    let padded = format!(" \n\t\r{} ", " ".repeat(1 << 20)) + "{}";
    let unicode = r#"{"data":{"value":"\/\b\f\n\r\t\"\\\u0000\u20ac\ud83d\ude00[]{}","number":-1.25e+2,"flag":false}}"#;
    let inputs = [
        None,
        Some("{}"),
        Some(r#"{"data":{}}"#),
        Some(unicode),
        Some(at_limit.as_str()),
        Some(at_depth.as_str()),
        Some(padded.as_str()),
    ];
    unsafe {
        let stack = fresh_scope_stack();
        for streaming in [false, true] {
            let intercept = Probe::default();
            let name = cstring("context-roundtrip");
            let _registration = register_capture(streaming, &name, &intercept);
            for input in inputs {
                let expected = input.map(|text| serde_json::from_str::<Json>(text).unwrap());
                let input = input.map(cstring);
                let resources = Resources::default();
                assert_status!(
                    execute(streaming, input.as_deref(), 0, &resources),
                    NemoRelayStatus::Ok
                );
                assert_eq!(resources.provider.calls.load(Ordering::SeqCst), 1);
                assert_eq!(intercept.contexts.lock().unwrap().last(), Some(&expected));
                assert_eq!(resources.provider.frees.load(Ordering::SeqCst), 1);
                assert_eq!(resources.codec.frees.load(Ordering::SeqCst), 1);
                assert_eq!(
                    resources.stream.frees.load(Ordering::SeqCst),
                    usize::from(streaming)
                );
            }
        }
        nemo_relay_scope_stack_free(stack);
    }
}

#[test]
fn ffi_llm_context_registration_cleans_up_on_unwind() {
    let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    reset_globals();
    unsafe {
        let stack = fresh_scope_stack();
        for streaming in [false, true] {
            let intercept = Probe::default();
            let name = cstring("context-unwind");
            let result = std::panic::catch_unwind(|| {
                let _registration = register_capture(streaming, &name, &intercept);
                panic!("simulate a failed assertion");
            });
            assert!(result.is_err());
            let resources = Resources::default();
            assert_status!(execute(streaming, None, 0, &resources), NemoRelayStatus::Ok);
            assert_eq!(intercept.calls.load(Ordering::SeqCst), 0);
            assert_eq!(resources.provider.calls.load(Ordering::SeqCst), 1);
        }
        nemo_relay_scope_stack_free(stack);
    }
}
