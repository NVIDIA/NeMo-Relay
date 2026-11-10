// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use nemo_relay_plugin::abi::{self as checked, v1_0 as a};
use nemo_relay_plugin::diagnostics::{self as sdk, ComponentDeclaration, DiagnosticCode};
use std::sync::Arc;

static SERIAL: Mutex<()> = Mutex::new(());
const FAILURE: BuiltinDiagnosticCode = BuiltinDiagnosticCode::Failure(FailureCode::Plugin(
    PluginFailureCode::Common(CommonPluginFailureCode::Internal),
));
fn new_context(enabled: bool, events: usize) -> Diagnostics {
    Diagnostics::from_settings(
        SourceIdentity {
            instance_id: Uuid::now_v7(),
            process_id: None,
            process_role: "native-test".into(),
            relay_version: "test".into(),
        },
        DiagnosticsSettings {
            enabled,
            limits: DiagnosticLimits {
                max_events: events,
                ..Default::default()
            },
        },
    )
    .unwrap()
}
struct DiagnosticRuntime(Arc<NativeDiagnostics>);
impl a::Runtime for DiagnosticRuntime {
    fn call_sync(&self, _: String, _: checked::Json) -> a::Result<checked::Json> {
        a::Result::Err(a::Error::new("test", "unavailable"))
    }
    fn call(
        &self,
        _: String,
        _: checked::Json,
    ) -> checked::future::FfiFuture<a::Result<checked::Json>> {
        use checked::future::FutureExt as _;
        async { a::Result::Err(a::Error::new("test", "unavailable")) }.into_ffi()
    }
    fn diagnostics(&self) -> checked::diagnostics::DiagnosticsBox {
        checked::diagnostics::DiagnosticsBox::new(CheckedDiagnostics {
            runtime: self.0.clone(),
        })
    }
    fn capture_scope(&self) -> a::Result<a::ScopeBox> {
        a::Result::Err(a::Error::new("test", "unavailable"))
    }
    fn bind_component(&self, _: a::ComponentBox) -> a::Result<()> {
        a::Result::Ok(())
    }
    fn register_callback(&self, _: a::Registration) -> a::Result<()> {
        a::Result::Ok(())
    }
    fn register_gate(&self, _: a::Registration) -> a::Result<String> {
        a::Result::Err(a::Error::new("test", "unavailable"))
    }
}
fn new_runtime(
    context: &Diagnostics,
    read: bool,
) -> (Arc<NativeDiagnostics>, nemo_relay_plugin::PluginRuntime) {
    let runtime = Arc::new(NativeDiagnostics::new(context.clone(), read, "fixture"));
    let sdk = nemo_relay_plugin::PluginRuntime::from_abi(a::RuntimeBox::new(DiagnosticRuntime(
        runtime.clone(),
    )));
    (runtime, sdk)
}
fn component(d: &sdk::Diagnostics, name: &str) -> sdk::DiagnosticComponent {
    let codes = [DiagnosticCode::Builtin(FAILURE)];
    d.component_lazy(|| ComponentDeclaration {
        name,
        codes: &codes,
    })
    .unwrap()
}
fn record(c: &sdk::DiagnosticComponent, count: u64) -> RecordReceipt {
    match c
        .try_record_lazy(|| {
            sdk::Observation::failure(
                1,
                FailureClass::Internal,
                "restricted-sentinel",
                NonZeroU64::new(count).unwrap(),
            )
        })
        .unwrap()
    {
        RecordOutcome::Recorded(receipt) => receipt,
        _ => panic!("recording must be enabled"),
    }
}
#[test]
fn native_authoring_uses_selected_context_and_preserves_counts_after_eviction() {
    let _lock = SERIAL.lock().unwrap();
    let context = new_context(true, 1);
    let other = new_context(true, 1);
    let (owner, runtime) = new_runtime(&context, false);
    let projected = Arc::new(AtomicUsize::new(0));
    let seen = projected.clone();
    let _projection = context
        .subscribe_projection(move |_| {
            seen.fetch_add(1, Ordering::Relaxed);
        })
        .unwrap();
    let d = runtime.diagnostics();
    let c = component(&d, "lookup");
    assert_eq!(record(&c, 2).code_total.unwrap().value.get(), 2);
    owner.registered();
    assert_eq!(record(&c, 3).code_total.unwrap().value.get(), 5);
    let snapshot = context
        .snapshot(&DiagnosticQuery::default())
        .unwrap()
        .to_snapshot()
        .unwrap();
    assert_eq!(
        snapshot.lifetime.totals.recorded_events_total.value.get(),
        2
    );
    assert_eq!(snapshot.lifetime.totals.occurrences_total.value.get(), 5);
    assert_eq!(
        snapshot.components[0]
            .lifetime
            .totals
            .history_evictions_total
            .value
            .get(),
        1
    );
    assert_eq!(snapshot.events.len(), 1);
    assert_eq!(
        snapshot.events[0]
            .context
            .execution
            .as_ref()
            .unwrap()
            .as_str(),
        "native_abi"
    );
    assert_eq!(
        snapshot.events[0].context.phase.as_ref().unwrap().as_str(),
        "runtime"
    );
    assert!(snapshot.events[0].remote.is_none());
    assert_eq!(projected.load(Ordering::Relaxed), 0);
    assert_eq!(
        other
            .snapshot(&DiagnosticQuery::default())
            .unwrap()
            .to_snapshot()
            .unwrap()
            .lifetime
            .totals
            .occurrences_total
            .value
            .get(),
        0
    );
    assert!(matches!(d.reader(), Err(Error::PermissionDenied)));
}
#[test]
fn native_disabled_skips_both_factories_without_backend_reservations() {
    let _lock = SERIAL.lock().unwrap();
    let context = new_context(false, 0);
    let (_owner, runtime) = new_runtime(&context, true);
    let d = runtime.diagnostics();
    assert_eq!(d.availability(), DiagnosticAvailability::Disabled);
    let c = d
        .component_lazy(|| panic!("disabled declaration factory"))
        .unwrap();
    assert_eq!(
        c.try_record_lazy(|| panic!("disabled observation factory"))
            .unwrap(),
        RecordOutcome::Disabled
    );
    assert_eq!(context.memory_usage().reserved_bytes, 0);
    assert_eq!(
        d.reader()
            .unwrap()
            .snapshot(&DiagnosticQuery::default())
            .unwrap_err(),
        Error::Read(DiagnosticReadReason::Disabled)
    );
    assert_eq!(d.delivery_stats(), sdk::DeliveryStats::default());
}
#[test]
fn native_extensions_reads_and_memory_share_canonical_contracts() {
    let _lock = SERIAL.lock().unwrap();
    let context = new_context(true, 8);
    let (_owner, runtime) = new_runtime(&context, true);
    let codes = [
        DiagnosticCode::Extension {
            name: "lookup_failed",
            description: "Internal lookup",
            outcome: DiagnosticOutcome::Failure,
            allowed_classes: &[FailureClass::Unavailable],
        },
        DiagnosticCode::Extension {
            name: "fallback_used",
            description: "Cached fallback",
            outcome: DiagnosticOutcome::Success,
            allowed_classes: &[],
        },
    ];
    let d = runtime.diagnostics();
    let c = d
        .component_lazy(|| ComponentDeclaration {
            name: "backend",
            codes: &codes,
        })
        .unwrap();
    let bad = c.try_record_lazy(|| {
        sdk::Observation::failure(1, FailureClass::Timeout, "invalid cause", NonZeroU64::MIN)
    });
    assert_eq!(bad.unwrap_err(), Error::Invalid);
    c.try_record_lazy(|| {
        sdk::Observation::failure(
            1,
            FailureClass::Unavailable,
            "restricted-sentinel",
            NonZeroU64::new(3).unwrap(),
        )
    })
    .unwrap();
    let reader = d.reader().unwrap();
    c.try_record_lazy(|| {
        assert!(matches!(
            reader.snapshot(&DiagnosticQuery::default()),
            Err(Error::Busy)
        ));
        sdk::Observation::success(2, "fallback used", NonZeroU64::MIN)
    })
    .unwrap();
    assert_eq!(d.delivery_stats().dropped, 0); // read contention is not authoring loss
    let page = reader.snapshot(&DiagnosticQuery::default()).unwrap();
    let direct = context
        .snapshot(&DiagnosticQuery::default())
        .unwrap()
        .to_snapshot()
        .unwrap();
    assert_eq!(page.source, direct.source);
    assert_eq!(page.events, direct.events);
    assert_eq!(page.sequence, direct.sequence);
    assert_eq!(page.lifetime.totals, direct.lifetime.totals);
    assert_eq!(reader.discovery().unwrap().source, direct.source);
    let memory = reader.inspect_memory(&DiagnosticQuery::default()).unwrap();
    assert_eq!(memory.source, direct.source);
    assert!(
        reader
            .snapshot_json(&DiagnosticQuery::default())
            .unwrap()
            .contains("restricted-sentinel")
    );
    assert!(
        reader
            .inspect_memory_json(&DiagnosticQuery::default())
            .unwrap()
            .contains("components")
    );
}
#[test]
fn native_revocation_and_component_close_fence_clones_and_foreign_tokens() {
    let _lock = SERIAL.lock().unwrap();
    let context = new_context(true, 8);
    let (owner, runtime) = new_runtime(&context, true);
    let (_other_owner, other_runtime) = new_runtime(&context, false);
    let d = runtime.diagnostics();
    let c = component(&d, "one");
    let clone = c.clone();
    record(&c, 1);
    let token = owner
        .with(|a| {
            Ok(a.session.as_ref().unwrap()[0].components[0]
                .as_ref()
                .unwrap()
                .token)
        })
        .unwrap();
    let foreign = _other_owner.close(token).unwrap_err();
    assert_eq!(foreign, Error::Closed);
    drop(other_runtime);
    c.close().unwrap();
    assert_eq!(
        clone
            .try_record_lazy(|| panic!("closed factory"))
            .unwrap_err(),
        Error::Closed
    );
    let next = component(&d, "two");
    record(&next, 2);
    assert_eq!(owner.close(token), Err(Error::Closed));
    owner.revoke();
    assert!(owner.state.lock().unwrap().is_none());
    assert_eq!(
        next.try_record_lazy(|| sdk::Observation::failure(
            1,
            FailureClass::Unknown,
            "late",
            NonZeroU64::MIN
        ))
        .unwrap_err(),
        Error::Closed
    );
    assert_eq!(d.reader().err().unwrap(), Error::Closed);
    let page = context
        .snapshot(&DiagnosticQuery::default())
        .unwrap()
        .to_snapshot()
        .unwrap();
    assert_eq!(page.lifetime.totals.occurrences_total.value.get(), 3);
    assert!(page.components.iter().all(|c| c.closed_at.is_some()));
}
#[test]
fn native_bounds_reject_new_registration_and_keep_existing_producer_writable() {
    let _lock = SERIAL.lock().unwrap();
    let context = new_context(true, 0);
    let (owner, runtime) = new_runtime(&context, false);
    let d = runtime.diagnostics();
    let mut components = Vec::new();
    for i in 0..bounds::MAX_COMPONENTS {
        components.push(component(&d, &format!("component{i}")));
    }
    let codes = [DiagnosticCode::Builtin(FAILURE)];
    assert!(matches!(
        d.component_lazy(|| ComponentDeclaration {
            name: "overflow",
            codes: &codes
        }),
        Err(Error::Capacity)
    ));
    assert!(matches!(
        d.component_lazy(|| ComponentDeclaration {
            name: "component0",
            codes: &codes
        }),
        Err(Error::Conflict)
    ));
    let receipt = record(&components[0], u64::MAX);
    assert_eq!(receipt.evidence, EvidenceDisposition::Dropped);
    assert!(record(&components[0], 1).code_total.unwrap().saturated);
    let oversized = "x".repeat(bounds::MAX_REQUEST_BYTES + 1);
    assert_eq!(
        owner.register(abi::DiagnosticRegistration {
            name: Str::from_str(&oversized),
            codes: [].as_slice().into(),
        }),
        Err(Error::Capacity)
    );
    let guard = Permit::acquire(&CALLS, bounds::MAX_CALLS).unwrap();
    let guards = (1..bounds::MAX_CALLS)
        .map(|_| Permit::acquire(&CALLS, bounds::MAX_CALLS).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        components[0]
            .try_record_lazy(|| sdk::Observation::failure(
                1,
                FailureClass::Unknown,
                "busy",
                NonZeroU64::MIN
            ))
            .unwrap_err(),
        Error::Busy
    );
    components[0].close().unwrap(); // release path needs no data permit
    drop(guards);
    drop(guard);
}
#[test]
fn native_read_checks_permission_before_queries_and_never_publishes_partial_output() {
    let _lock = SERIAL.lock().unwrap();
    let context = new_context(true, 8);
    let (denied, _runtime) = new_runtime(&context, false);
    let (allowed, _runtime2) = new_runtime(&context, true);
    let mut output = [0u8; 4];
    assert_eq!(
        denied.read(1, Str::from_str("invalid"), &mut output),
        Err(Error::PermissionDenied)
    );
    assert_eq!(
        allowed.read(0, Str::from_str("{}"), &mut output),
        Err(Error::BufferTooSmall)
    );
    assert_eq!(
        allowed.read(1, Str::from_str("{}"), &mut output),
        Err(Error::BufferTooSmall)
    );
    assert_eq!(
        allowed.read(1, Str::from_str("{\"unknown\":true}"), &mut output),
        Err(Error::Read(DiagnosticReadReason::InvalidQuery))
    );
    assert_eq!(context.memory_usage().query_workspaces, 0);
}

#[test]
fn native_session_admission_is_bounded_and_released_on_revocation() {
    let _lock = SERIAL.lock().unwrap();
    let context = new_context(true, 0);
    let baseline = SESSIONS.load(Ordering::Relaxed);
    let mut active = Vec::new();
    for _ in baseline..bounds::MAX_SESSIONS {
        let (owner, runtime) = new_runtime(&context, false);
        let d = runtime.diagnostics();
        assert_eq!(d.capabilities().unwrap().adapter_session_bytes, 0);
        let c = component(&d, "internal");
        assert_eq!(
            d.capabilities().unwrap().adapter_session_bytes,
            std::mem::size_of::<Session>()
        );
        active.push((owner, runtime, c));
    }
    let (owner, runtime) = new_runtime(&context, false);
    let d = runtime.diagnostics();
    let codes = [DiagnosticCode::Builtin(FAILURE)];
    assert!(matches!(
        d.component_lazy(|| ComponentDeclaration {
            name: "internal",
            codes: &codes
        }),
        Err(Error::Unavailable)
    ));
    active.pop().unwrap().0.revoke();
    let c = component(&d, "internal");
    record(&c, 1);
    owner.revoke();
    for (owner, _, _) in active {
        owner.revoke();
    }
    assert_eq!(SESSIONS.load(Ordering::Relaxed), baseline);
}

#[test]
fn native_rejected_declarations_roll_back_and_retention_reports_utf8_truncation() {
    let _lock = SERIAL.lock().unwrap();
    let context = new_context(true, 2);
    let (_owner, runtime) = new_runtime(&context, false);
    let d = runtime.diagnostics();
    // The second extension fails core name validation after the first was reserved.
    let codes = [
        DiagnosticCode::Extension {
            name: "valid",
            description: "valid",
            outcome: DiagnosticOutcome::Success,
            allowed_classes: &[],
        },
        DiagnosticCode::Extension {
            name: "invalid?",
            description: "invalid",
            outcome: DiagnosticOutcome::Success,
            allowed_classes: &[],
        },
    ];
    for _ in 0..64 {
        assert!(matches!(
            d.component_lazy(|| ComponentDeclaration {
                name: "rollback",
                codes: &codes
            }),
            Err(Error::Invalid)
        ));
    }
    let c = component(&d, "rollback");
    let text = "λ\"".repeat(2000);
    let receipt = c
        .try_record_lazy(|| {
            sdk::Observation::failure(1, FailureClass::Internal, &text, NonZeroU64::MIN)
        })
        .unwrap();
    assert!(matches!(
        receipt,
        RecordOutcome::Recorded(RecordReceipt {
            evidence: EvidenceDisposition::Truncated,
            ..
        })
    ));
    let page = context
        .snapshot(&DiagnosticQuery::default())
        .unwrap()
        .to_snapshot()
        .unwrap();
    assert_eq!(page.components.len(), 1);
    assert_eq!(page.lifetime.totals.occurrences_total.value.get(), 1);
    assert!(page.events[0].message().unwrap().len() <= context.limits().max_detail_bytes);
    let oversized = "x".repeat(bounds::MAX_REQUEST_BYTES + 1);
    assert_eq!(
        c.try_record_lazy(|| sdk::Observation::failure(
            1,
            FailureClass::Internal,
            &oversized,
            NonZeroU64::MIN
        )),
        Err(Error::Capacity)
    );
    assert_eq!(d.delivery_stats().dropped, 1);
}

#[test]
fn native_close_serializes_with_recording_without_reusing_authority() {
    let _lock = SERIAL.lock().unwrap();
    let context = new_context(true, 1);
    let (owner, runtime) = new_runtime(&context, false);
    let c = component(&runtime.diagnostics(), "concurrent");
    let start = Arc::new(std::sync::Barrier::new(2));
    let seen = start.clone();
    let writer = c.clone();
    let thread = std::thread::spawn(move || {
        seen.wait();
        let mut accepted = 0;
        for _ in 0..1000 {
            match writer.try_record_lazy(|| {
                sdk::Observation::failure(1, FailureClass::Internal, "concurrent", NonZeroU64::MIN)
            }) {
                Ok(RecordOutcome::Recorded(_)) => accepted += 1,
                Err(Error::Closed) => break,
                other => panic!("unexpected concurrent result: {other:?}"),
            }
        }
        accepted
    });
    start.wait();
    owner.revoke();
    let accepted = thread.join().unwrap();
    assert!(matches!(
        c.try_record_lazy(|| sdk::Observation::failure(
            1,
            FailureClass::Internal,
            "late",
            NonZeroU64::MIN
        )),
        Err(Error::Closed)
    ));
    let page = context
        .snapshot(&DiagnosticQuery::default())
        .unwrap()
        .to_snapshot()
        .unwrap();
    assert_eq!(
        page.lifetime.totals.recorded_events_total.value.get(),
        accepted
    );
    assert!(page.components[0].closed_at.is_some());
}

#[test]
fn native_sdk_admits_concurrent_clones_and_bounds_factories() {
    let _lock = SERIAL.lock().unwrap();
    let context = new_context(true, 8);
    let (_owner, runtime) = new_runtime(&context, true);
    let d = runtime.diagnostics();
    let c = component(&d, "parallel");
    let reader = d.reader().unwrap();
    std::thread::scope(|scope| {
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let mut releases = Vec::new();
        let mut writers = Vec::new();
        for _ in 0..bounds::MAX_CALLS {
            let c = c.clone();
            let ready = ready_tx.clone();
            let (release_tx, release_rx) = std::sync::mpsc::channel();
            releases.push(release_tx);
            writers.push(scope.spawn(move || {
                c.try_record_lazy(|| {
                    ready.send(()).unwrap();
                    release_rx.recv().unwrap();
                    sdk::Observation::failure(
                        1,
                        FailureClass::Internal,
                        "parallel",
                        NonZeroU64::MIN,
                    )
                })
            }));
        }
        drop(ready_tx);
        let all_entered = (0..bounds::MAX_CALLS).all(|_| {
            ready_rx
                .recv_timeout(std::time::Duration::from_secs(5))
                .is_ok()
        });
        let overflow = c.try_record_lazy(|| panic!("full admission must skip the factory"));
        let registration = d.component_lazy(|| panic!("registration shares the admission budget"));
        let read = reader.snapshot(&DiagnosticQuery::default());
        // Release blocked factories before asserting so a regression cannot strand a thread.
        for release in releases {
            let _ = release.send(());
        }
        let results = writers
            .into_iter()
            .map(|writer| writer.join().unwrap())
            .collect::<Vec<_>>();
        assert!(all_entered, "distinct threads must enter concurrently");
        assert_eq!(overflow, Err(Error::Busy));
        assert!(matches!(registration, Err(Error::Busy)));
        assert!(matches!(read, Err(Error::Busy)));
        for result in results {
            assert!(matches!(result, Ok(RecordOutcome::Recorded(_))));
        }
    });
    assert_eq!(d.delivery_stats().dropped, 2);
    assert_eq!(
        record(&c, 1).code_total.unwrap().value.get(),
        bounds::MAX_CALLS as u64 + 1
    );
}

#[test]
fn native_sdk_reentrancy_is_scoped_to_activation_and_unwinds_cleanly() {
    let _lock = SERIAL.lock().unwrap();
    let context = new_context(true, 8);
    let (_owner_a, runtime_a) = new_runtime(&context, true);
    let (_owner_b, runtime_b) = new_runtime(&context, false);
    let a = component(&runtime_a.diagnostics(), "first");
    let b = component(&runtime_b.diagnostics(), "second");
    let clone = a.clone();
    let reader = runtime_a.diagnostics().reader().unwrap();
    a.try_record_lazy(|| {
        b.try_record_lazy(|| {
            assert_eq!(
                clone.try_record_lazy(|| panic!("recursive factory")),
                Err(Error::Busy)
            );
            assert!(matches!(
                reader.snapshot(&DiagnosticQuery::default()),
                Err(Error::Busy)
            ));
            sdk::Observation::failure(1, FailureClass::Internal, "nested", NonZeroU64::MIN)
        })
        .unwrap();
        sdk::Observation::failure(1, FailureClass::Internal, "outer", NonZeroU64::MIN)
    })
    .unwrap();
    for _ in 0..=bounds::MAX_CALLS {
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            a.try_record_lazy(|| panic!("producer panic"))
        }));
        assert!(panic.is_err());
    }
    assert_eq!(record(&a, 1).code_total.unwrap().value.get(), 2);
    let d = runtime_a.diagnostics();
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            d.component_lazy(|| panic!("declaration panic"))
        }))
        .is_err()
    );
    assert!(matches!(
        d.component_lazy(|| ComponentDeclaration {
            name: "invalid",
            codes: &[]
        }),
        Err(Error::Invalid)
    ));
    let after_error = component(&d, "after-error");
    record(&after_error, 1);
    reader.snapshot(&DiagnosticQuery::default()).unwrap();
}
