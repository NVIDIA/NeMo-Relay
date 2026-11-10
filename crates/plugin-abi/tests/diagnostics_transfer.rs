// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded diagnostic calls must not allocate in the transport.
use nemo_relay_plugin_abi::{
    borrow::{SliceMut, Str},
    diagnostics::*,
    transfer::Value,
    xabi::XabiType,
};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};

struct CountingAllocator;
thread_local! { static ALLOCATIONS: Cell<Option<usize>> = const { Cell::new(None) }; }
fn allocated() {
    let _ = ALLOCATIONS.try_with(|count| {
        if let Some(value) = count.get() {
            count.set(Some(value + 1));
        }
    });
}
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        allocated();
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        allocated();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        allocated();
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

struct Diagnostics;
impl nemo_relay_plugin_abi::diagnostics::Diagnostics for Diagnostics {
    fn capabilities(&self) -> Result<DiagnosticCapabilities, u32> {
        Ok(DiagnosticCapabilities {
            contract_version: 1,
            max_calls: 42,
            ..Default::default()
        })
    }
    fn register(&self, registration: DiagnosticRegistration<'_>) -> Result<u64, u32> {
        assert_eq!(registration.name.as_str(), "component\0日本語");
        assert_eq!(registration.codes[0].name.as_str(), "code");
        Ok(42)
    }
    fn record(
        &self,
        token: u64,
        observation: DiagnosticObservation<'_>,
    ) -> Result<DiagnosticReceipt, u32> {
        assert_eq!(token, 42);
        assert_eq!(observation.detail.as_str(), "detail\0日本語");
        Ok(DiagnosticReceipt {
            sequence: 9,
            total: observation.count,
            total_present: 1,
            ..Default::default()
        })
    }
    fn close(&self, token: u64) -> Result<(), u32> {
        if token == 42 { Ok(()) } else { Err(7) }
    }
    fn read(&self, _: u32, query: Str<'_>, mut output: SliceMut<'_, u8>) -> Result<usize, u32> {
        assert_eq!(query.as_str(), "{}");
        output[..2].copy_from_slice(b"{}");
        Ok(2)
    }
}

#[test]
fn synchronous_diagnostics_use_borrowed_inputs_and_caller_owned_replies() {
    let payload = Value(DiagnosticsBox::new(Diagnostics)).into_payload();
    let remote = unsafe { Value::<DiagnosticsBox>::from_payload(payload) }
        .unwrap()
        .0;
    let codes = [DiagnosticCode {
        kind: 0,
        outcome: 0,
        name: "code".into(),
        description: "description".into(),
        classes: (&[][..]).into(),
    }];
    let observation = DiagnosticObservation {
        code: 1,
        outcome: 0,
        severity: 0,
        count: u64::MAX,
        failure_class: "".into(),
        detail: "detail\0日本語".into(),
        request_id: "".into(),
        session_id: "".into(),
    };
    let mut output = [0u8; 8];
    ALLOCATIONS.with(|count| count.set(Some(0)));
    for _ in 0..100 {
        assert_eq!(remote.capabilities().unwrap().max_calls, 42);
        let token = remote
            .register(DiagnosticRegistration {
                name: "component\0日本語".into(),
                codes: codes.as_slice().into(),
            })
            .unwrap();
        let receipt = remote.record(token, observation).unwrap();
        assert_eq!(receipt.sequence, 9);
        assert_eq!(receipt.total, u64::MAX);
        assert_eq!(
            remote.read(0, "{}".into(), SliceMut::from_mut_slice(&mut output)),
            Ok(2)
        );
        assert_eq!(&output[..2], b"{}");
        assert_eq!(remote.close(token), Ok(()));
        assert_eq!(remote.close(0), Err(7));
    }
    let allocations = ALLOCATIONS.with(|count| count.replace(None)).unwrap();
    assert_eq!(allocations, 0, "diagnostic transport allocated per call");
}

/// Reject owned payloads without reading padding or retaining borrowed storage.
fn assert_call_only<T: XabiType>(value: T) {
    let error = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| value.into_payload()))
        .err()
        .expect("borrowed values must reject owned payload encoding");
    assert_eq!(
        error.downcast_ref::<&str>().copied(),
        Some("borrowed ABI values support synchronous argument transfer only")
    );

    use std::sync::atomic::{AtomicUsize, Ordering};
    static RELEASED: AtomicUsize = AtomicUsize::new(0);
    unsafe extern "C" fn release(_: *mut u8, _: usize) {
        RELEASED.fetch_add(1, Ordering::SeqCst);
    }
    let before = RELEASED.load(Ordering::SeqCst);
    let payload = nemo_relay_plugin_abi::xabi::XabiOwnedBytes {
        ptr: std::ptr::null_mut(),
        len: 0,
        free: release,
    };
    let error = unsafe { T::from_payload(payload) }
        .err()
        .expect("borrowed values must reject owned payload decoding");
    assert!(
        error
            .to_string()
            .contains("synchronous argument transfer only")
    );
    assert_eq!(RELEASED.load(Ordering::SeqCst), before + 1);
}

#[test]
fn borrowed_views_reject_owned_payload_encoding_and_decoding() {
    use nemo_relay_plugin_abi::borrow::{Output, Reply};
    let text = String::from("local diagnostic text");
    let classes = [Str::from_str(&text)];
    let codes = [DiagnosticCode {
        kind: 0,
        outcome: 0,
        name: text.as_str().into(),
        description: "".into(),
        classes: classes.as_slice().into(),
    }];
    assert_call_only(Str::from_str(&text));
    assert_call_only(codes[0]);
    assert_call_only(DiagnosticRegistration {
        name: text.as_str().into(),
        codes: codes.as_slice().into(),
    });
    assert_call_only(DiagnosticObservation {
        code: 1,
        outcome: 0,
        severity: 0,
        count: 1,
        failure_class: "".into(),
        detail: text.as_str().into(),
        request_id: "".into(),
        session_id: "".into(),
    });
    let mut bytes = [0u8; 8];
    assert_call_only(SliceMut::from_mut_slice(&mut bytes));
    let mut reply = Reply::<DiagnosticCapabilities>::default();
    assert_call_only(Output::new(&mut reply));
}
