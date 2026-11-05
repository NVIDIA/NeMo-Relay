// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Validate diagnostic authoring against checked peers with invalid behavior.
use nemo_relay_plugin::{
    PluginRuntime,
    abi::{
        self,
        abi_stable::{
            sabi_trait::prelude::TD_Opaque,
            std_types::{RResult, RSliceMut, RStr, RString},
        },
        async_ffi::{FfiFuture, FutureExt as _},
        diagnostics as d, v1_0 as a,
    },
    diagnostics::{
        self as sdk, BuiltinDiagnosticCode, CommonPluginFailureCode, ComponentDeclaration,
        DiagnosticAvailability, DiagnosticCode, DiagnosticError, DiagnosticQuery, FailureClass,
        FailureCode, PluginFailureCode,
    },
};
use std::{
    num::NonZeroU64,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};
#[derive(Clone, Copy)]
enum Behavior {
    Disabled,
    Unsupported,
    InvalidCapabilities,
    UnknownStatus,
    InvalidReceipt,
    OversizedRead,
}
struct Peer {
    behavior: Behavior,
    registrations: AtomicUsize,
}
struct Runtime(Arc<Peer>);
struct DiagnosticPeer(Arc<Peer>);
fn unavailable<T>() -> a::Result<T> {
    a::Result::RErr(a::Error::new("test", "unavailable"))
}
impl a::Runtime for Runtime {
    fn call_sync(&self, _: RString, _: abi::Json) -> a::Result<abi::Json> {
        unavailable()
    }
    fn call(&self, _: RString, _: abi::Json) -> FfiFuture<a::Result<abi::Json>> {
        async { unavailable() }.into_ffi()
    }
    fn diagnostics(&self) -> d::DiagnosticsBox {
        d::Diagnostics_TO::from_value(DiagnosticPeer(self.0.clone()), TD_Opaque)
    }
    fn capture_scope(&self) -> a::Result<a::ScopeBox> {
        unavailable()
    }
    fn bind_component(&self, _: a::ComponentBox) -> a::Result<()> {
        unavailable()
    }
    fn register_callback(&self, _: a::Registration) -> a::Result<()> {
        unavailable()
    }
    fn register_gate(&self, _: a::Registration) -> a::Result<RString> {
        unavailable()
    }
}
impl d::Diagnostics for DiagnosticPeer {
    fn capabilities(&self) -> RResult<d::DiagnosticCapabilities, u32> {
        if matches!(self.0.behavior, Behavior::Unsupported) {
            return RResult::RErr(2);
        }
        RResult::ROk(d::DiagnosticCapabilities {
            contract_version: 1,
            availability: u32::from(!matches!(self.0.behavior, Behavior::Disabled)),
            can_read_context: 1,
            max_codes: if matches!(self.0.behavior, Behavior::InvalidCapabilities) {
                33
            } else {
                32
            },
            max_request_bytes: 16384,
            max_query_bytes: 1024,
            max_response_bytes: 16,
            ..Default::default()
        })
    }
    fn register(&self, _: d::DiagnosticRegistration<'_>) -> RResult<u64, u32> {
        self.0.registrations.fetch_add(1, Ordering::Relaxed);
        RResult::ROk(42)
    }
    fn record(
        &self,
        _: u64,
        _: d::DiagnosticObservation<'_>,
    ) -> RResult<d::DiagnosticReceipt, u32> {
        if matches!(self.0.behavior, Behavior::UnknownStatus) {
            return RResult::RErr(u32::MAX);
        }
        RResult::ROk(d::DiagnosticReceipt {
            sequence: 1,
            total_present: 2,
            ..Default::default()
        })
    }
    fn close(&self, _: u64) -> RResult<(), u32> {
        RResult::ROk(())
    }
    fn read(&self, _: u32, _: RStr<'_>, output: RSliceMut<'_, u8>) -> RResult<usize, u32> {
        RResult::ROk(output.len() + 1)
    }
}
fn facade(behavior: Behavior) -> (Arc<Peer>, sdk::Diagnostics) {
    let peer = Arc::new(Peer {
        behavior,
        registrations: AtomicUsize::new(0),
    });
    let runtime =
        PluginRuntime::from_abi(a::Runtime_TO::from_value(Runtime(peer.clone()), TD_Opaque));
    (peer, runtime.diagnostics())
}
const FAILURE: BuiltinDiagnosticCode = BuiltinDiagnosticCode::Failure(FailureCode::Plugin(
    PluginFailureCode::Common(CommonPluginFailureCode::Internal),
));
#[test]
fn inert_and_incompatible_peers_skip_author_factories() {
    for (behavior, expected) in [
        (Behavior::Disabled, DiagnosticAvailability::Disabled),
        (Behavior::Unsupported, DiagnosticAvailability::Unsupported),
        (
            Behavior::InvalidCapabilities,
            DiagnosticAvailability::Unsupported,
        ),
    ] {
        let (peer, diagnostics) = facade(behavior);
        assert_eq!(diagnostics.availability(), expected);
        let component = diagnostics
            .component_lazy(|| panic!("inert declaration factory"))
            .unwrap();
        assert!(matches!(
            component
                .try_record_lazy(|| panic!("inert observation factory"))
                .unwrap(),
            sdk::RecordOutcome::Disabled
        ));
        assert_eq!(peer.registrations.load(Ordering::Relaxed), 0);
    }
}
#[test]
fn invalid_receipts_and_unknown_errors_fail_closed_without_leaking_admission() {
    for behavior in [Behavior::UnknownStatus, Behavior::InvalidReceipt] {
        let (_, diagnostics) = facade(behavior);
        let codes = [DiagnosticCode::Builtin(FAILURE)];
        let component = diagnostics
            .component_lazy(|| ComponentDeclaration {
                name: "operation",
                codes: &codes,
            })
            .unwrap();
        for _ in 0..2 {
            assert_eq!(
                component
                    .try_record_lazy(|| sdk::Observation::failure(
                        1,
                        FailureClass::Internal,
                        "restricted",
                        NonZeroU64::MIN
                    ))
                    .unwrap_err(),
                DiagnosticError::Internal
            );
        }
        assert_eq!(diagnostics.delivery_stats().uncertain, 2);
    }
}
#[test]
fn checked_peer_cannot_claim_a_read_beyond_caller_owned_storage() {
    let (_, diagnostics) = facade(Behavior::OversizedRead);
    let reader = diagnostics.reader().unwrap();
    for _ in 0..2 {
        assert_eq!(
            reader
                .snapshot_json(&DiagnosticQuery::default())
                .unwrap_err(),
            DiagnosticError::Internal
        );
    }
}
