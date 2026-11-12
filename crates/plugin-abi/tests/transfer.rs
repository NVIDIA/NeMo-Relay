// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ownership and lossless values across the actual xabi transfer path.
#![allow(missing_docs)] // xabi generates public test-only plumbing.
use nemo_relay_plugin_abi::{
    Field, Json,
    future::{FfiFuture, FutureExt as _},
    transfer::Value,
    v1_0::{self as a, Component, ComponentBox, RegisteredComponent},
    xabi::{self, XabiType},
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct Owner(Arc<AtomicUsize>);
impl Drop for Owner {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
impl Component for Owner {
    fn shutdown(&self) -> FfiFuture<a::Result<()>> {
        async { Ok(()) }.into_ffi()
    }
}

#[test]
fn unclaimed_nested_owners_are_released() {
    let drops = Arc::new(AtomicUsize::new(0));
    let record = RegisteredComponent {
        error: None,
        registrations: Vec::new(),
        component: ComponentBox::new(Owner(drops.clone())),
    };
    drop(xabi::__private::XabiWire::new(Value(record)));
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    drop(xabi::__private::XabiWire::new(Value(Some(
        ComponentBox::new(Owner(drops.clone())),
    ))));
    assert_eq!(drops.load(Ordering::SeqCst), 2);
    drop(xabi::__private::XabiWire::new(Value(vec![
        ComponentBox::new(Owner(drops.clone())),
        ComponentBox::new(Owner(drops.clone())),
    ])));
    assert_eq!(drops.load(Ordering::SeqCst), 4);
}

#[test]
fn claimed_owner_survives_wire_release_and_drops_once() {
    let drops = Arc::new(AtomicUsize::new(0));
    let mut wire = Value(Some(ComponentBox::new(Owner(drops.clone())))).into_wire();
    let value = unsafe { Value::<Option<ComponentBox>>::xabi_take_from_wire(&mut wire) }.unwrap();
    unsafe { Value::<Option<ComponentBox>>::xabi_drop_wire(&mut wire) };
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(value);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn payload_transfer_preserves_nested_json_and_integer_ranges() {
    let value = Json::Object(vec![Field {
        key: "日本語\0key".into(),
        value: Json::Array(vec![
            Json::Null,
            Json::Bool(true),
            Json::Signed(i64::MIN),
            Json::Unsigned(u64::MAX),
            Json::Float(-0.25),
            Json::String("hello\0世界".into()),
        ]),
    }]);
    let payload = Value(value.clone()).into_payload();
    let result = unsafe { Value::<Json>::from_payload(payload) }.unwrap().0;
    assert_eq!(result, value);
}

/// Non-finite numbers and duplicate keys must fail instead of crossing as valid JSON.
#[test]
fn malformed_json_is_rejected_during_transfer() {
    for value in [
        Json::Float(f64::INFINITY),
        Json::Float(f64::NAN),
        Json::Object(vec![
            Field {
                key: "key".into(),
                value: Json::Null,
            },
            Field {
                key: "key".into(),
                value: Json::Null,
            },
        ]),
    ] {
        let payload = Value(value).into_payload();
        assert!(unsafe { Value::<Json>::from_payload(payload) }.is_err());
    }
}

#[xabi::xabi(id = "relay.tests.decode-rollback", version = 1)]
trait DecodeRollback {
    fn accept(&self, value: Value<Json>, owner: Value<RegisteredComponent>) -> xabi::Result<()>;
}
struct Decoder;
impl DecodeRollback for Decoder {
    fn accept(&self, _: Value<Json>, _: Value<RegisteredComponent>) -> xabi::Result<()> {
        panic!("invalid payload must be rejected before invocation")
    }
}
#[test]
fn earlier_argument_failure_releases_later_nested_callback_owner() {
    let drops = Arc::new(AtomicUsize::new(0));
    let decoder = XabiV1OwnedTraitDecodeRollback::new(Decoder);
    let owner = RegisteredComponent {
        error: None,
        registrations: Vec::new(),
        component: ComponentBox::new(Owner(drops.clone())),
    };
    assert!(
        decoder
            .xabi_borrow()
            .accept(Value(Json::Float(f64::NAN)), Value(owner))
            .is_err()
    );
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn dropping_unpolled_foreign_future_releases_captured_admission() {
    let drops = Arc::new(AtomicUsize::new(0));
    let owner = Owner(drops.clone());
    let future = async move {
        let _owner = owner;
        std::future::pending::<a::Result<Json>>().await
    }
    .into_ffi();
    let payload = Value(future).into_payload();
    let future = unsafe { Value::<FfiFuture<a::Result<Json>>>::from_payload(payload) }
        .unwrap()
        .0;
    drop(future);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

static FACTORY_DROPS: AtomicUsize = AtomicUsize::new(0);
struct IncompleteFactory;
impl a::Factory for IncompleteFactory {
    fn create(&self) -> a::Result<a::PluginBox> {
        panic!("incomplete factories must be rejected before plugin construction")
    }
}
impl Drop for IncompleteFactory {
    fn drop(&mut self) {
        FACTORY_DROPS.fetch_add(1, Ordering::SeqCst);
    }
}
unsafe extern "C" fn incomplete_factory() -> *mut std::ffi::c_void {
    use nemo_relay_plugin_abi::{transfer::Transfer, wire::XabiV1VtableTraitWireFactory};
    let wire = a::FactoryBox::new(IncompleteFactory).lower();
    unsafe { (*wire.vtable).size = XabiV1VtableTraitWireFactory::MIN_SIZE };
    wire.vtable.cast()
}
#[test]
fn incomplete_factory_is_rejected_and_releases_its_owner() {
    let export = xabi::XabiExport::new(
        xabi::XabiStr::from_static(nemo_relay_plugin_abi::wire::XabiV1AbiTraitWireFactory::ID),
        xabi::XabiStr::from_static(a::FACTORY_MODULE),
        1,
        1,
        0,
        incomplete_factory,
    );
    let error = unsafe { nemo_relay_plugin_abi::handles::load_factory(&export) }
        .err()
        .unwrap();
    assert!(
        error
            .to_string()
            .contains("incomplete native factory module")
    );
    assert_eq!(FACTORY_DROPS.load(Ordering::SeqCst), 1);
}

#[test]
fn payload_transfer_preserves_negative_zero() {
    let payload = Value(Json::Float(-0.0)).into_payload();
    let Json::Float(value) = unsafe { Value::<Json>::from_payload(payload) }.unwrap().0 else {
        panic!("floating-point JSON variant changed")
    };
    assert_eq!(value.to_bits(), (-0.0_f64).to_bits());
}

struct DeferredComponent {
    owner: Arc<Owner>,
    waker: Arc<std::sync::Mutex<Option<std::task::Waker>>>,
}
impl Component for DeferredComponent {
    fn shutdown(&self) -> FfiFuture<a::Result<()>> {
        let owner = self.owner.clone();
        let waker = self.waker.clone();
        let mut pending = true;
        std::future::poll_fn(move |cx| {
            let _keep_alive = &owner;
            if std::mem::take(&mut pending) {
                *waker.lock().unwrap() = Some(cx.waker().clone());
                std::task::Poll::Pending
            } else {
                std::task::Poll::Ready(Ok(()))
            }
        })
        .into_ffi()
    }
}
struct WakeCount(AtomicUsize);
impl std::task::Wake for WakeCount {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

/// Dropping a capability must not invalidate its pending future or cloned waker.
#[test]
fn foreign_future_and_cloned_waker_outlive_their_capability() {
    use std::{future::Future, pin::Pin, task::Context};
    for complete in [false, true] {
        let drops = Arc::new(AtomicUsize::new(0));
        let captured = Arc::new(std::sync::Mutex::new(None));
        let component = ComponentBox::new(DeferredComponent {
            owner: Arc::new(Owner(drops.clone())),
            waker: captured.clone(),
        });
        let payload = Value(component).into_payload();
        let remote = unsafe { Value::<ComponentBox>::from_payload(payload) }
            .unwrap()
            .0;
        let mut future = remote.shutdown();
        drop(remote);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        let wakes = Arc::new(WakeCount(AtomicUsize::new(0)));
        let waker = std::task::Waker::from(wakes.clone());
        let mut context = Context::from_waker(&waker);
        assert!(Pin::new(&mut future).poll(&mut context).is_pending());
        let retained = captured.lock().unwrap().take().unwrap();
        // Exercise the owned and borrowed foreign waker callbacks separately.
        #[allow(clippy::waker_clone_wake)]
        retained.clone().wake();
        retained.wake_by_ref();
        assert_eq!(wakes.0.load(Ordering::SeqCst), 2);
        if complete {
            assert!(matches!(
                Pin::new(&mut future).poll(&mut context),
                std::task::Poll::Ready(Ok(()))
            ));
            assert_eq!(drops.load(Ordering::SeqCst), 1);
        }
        drop(future);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        retained.wake();
        assert_eq!(wakes.0.load(Ordering::SeqCst), 3);
    }
}

#[test]
fn foreign_poll_panic_is_an_error_and_releases_captured_work() {
    use std::{future::Future, pin::Pin, task::Context};
    let drops = Arc::new(AtomicUsize::new(0));
    let owner = Owner(drops.clone());
    let future = std::future::poll_fn(move |_| -> std::task::Poll<a::Result<Json>> {
        let _keep_alive = &owner;
        panic!("source future panic");
    })
    .into_ffi();
    let payload = Value(future).into_payload();
    let mut future = unsafe { Value::<FfiFuture<a::Result<Json>>>::from_payload(payload) }
        .unwrap()
        .0;
    let mut context = Context::from_waker(std::task::Waker::noop());
    assert!(matches!(
        Pin::new(&mut future).poll(&mut context),
        std::task::Poll::Ready(Err(_))
    ));
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    drop(future);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn incompatible_future_generation_retains_rollback_ownership() {
    use nemo_relay_plugin_abi::transfer::Transfer;
    let drops = Arc::new(AtomicUsize::new(0));
    let owner = Owner(drops.clone());
    let future = async move {
        let _owner = owner;
        std::future::pending::<a::Result<Json>>().await
    }
    .into_ffi();
    let mut wire = future.lower();
    wire.abi_version += 1;
    assert!(unsafe { FfiFuture::<a::Result<Json>>::take(&mut wire) }.is_err());
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    unsafe { FfiFuture::<a::Result<Json>>::release(&mut wire) };
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}
