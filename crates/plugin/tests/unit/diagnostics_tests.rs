// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use nemo_relay_plugin::diagnostics::{self as sdk, abi};
use std::num::NonZeroU64;
static ENABLED: AtomicBool = AtomicBool::new(false);
static ACCEPTED: AtomicUsize = AtomicUsize::new(0);
unsafe extern "C" fn capabilities(
    _: *const NemoRelayNativePluginRuntime,
    out: *mut abi::Capabilities,
) -> u32 {
    unsafe {
        out.write(abi::Capabilities {
            contract_version: 1,
            availability: u32::from(ENABLED.load(Ordering::Relaxed)),
            max_codes: 32,
            max_request_bytes: 16384,
            ..Default::default()
        });
    }
    0
}
unsafe extern "C" fn register(
    _: *const NemoRelayNativePluginRuntime,
    _: *const abi::Registration,
    out: *mut u64,
) -> u32 {
    unsafe {
        out.write(42);
    }
    0
}
unsafe extern "C" fn record(
    _: *const NemoRelayNativePluginRuntime,
    token: u64,
    input: *const abi::Observation,
    out: *mut abi::Receipt,
) -> u32 {
    if token != 42 {
        return 8;
    }
    let count = unsafe { (*input).count } as usize;
    let total = ACCEPTED.fetch_add(count, Ordering::SeqCst) + count;
    unsafe {
        out.write(abi::Receipt {
            sequence: total as u64,
            total: total as u64,
            total_present: 1,
            ..Default::default()
        });
    }
    0
}
unsafe extern "C" fn close(_: *const NemoRelayNativePluginRuntime, _: u64) -> u32 {
    0
}
unsafe extern "C" fn read(
    _: *const NemoRelayNativePluginRuntime,
    _: u32,
    _: abi::Bytes,
    _: *mut u8,
    _: usize,
    written: *mut usize,
) -> u32 {
    unsafe {
        written.write(0);
    }
    5
}
static API: abi::NativeDiagnosticsApiV1 = abi::NativeDiagnosticsApiV1 {
    struct_size: size_of::<abi::NativeDiagnosticsApiV1>(),
    contract_version: 1,
    capabilities,
    register,
    record,
    close,
    read,
};
fn extended() -> abi::NativeHostApiV8WithDiagnostics {
    let mut v7 = test_host_v7();
    v7.v6.v5.v4.v3.v1.abi_version = 8;
    v7.v6.v5.v4.v3.v1.struct_size = size_of::<abi::NativeHostApiV8WithDiagnostics>();
    v7.v6.v5.v4.plugin_context_runtime = capture_plugin_context_runtime;
    v7.v6.v5.v4.plugin_runtime_retain = capture_plugin_runtime_retain;
    v7.v6.v5.v4.plugin_runtime_release = capture_plugin_runtime_release;
    abi::NativeHostApiV8WithDiagnostics {
        v8: nemo_relay_plugin::NemoRelayNativeHostApiV8 {
            v7,
            current_scope_stack_snapshot: snapshot_response,
        },
        diagnostics_api: &API,
    }
}
#[test]
fn native_diagnostics_extension_preserves_offsets_and_skips_missing_or_unknown_tables() {
    let _lock = begin_test();
    assert_eq!(offset_of!(abi::NativeHostApiV8WithDiagnostics, v8), 0);
    assert_eq!(
        offset_of!(abi::NativeHostApiV8WithDiagnostics, diagnostics_api),
        size_of::<nemo_relay_plugin::NemoRelayNativeHostApiV8>()
    );
    let mut host = extended();
    host.v8.v7.v6.v5.v4.v3.v1.struct_size =
        size_of::<nemo_relay_plugin::NemoRelayNativeHostApiV8>();
    host.diagnostics_api = ptr::dangling(); // must not inspect an absent tail
    let d = PluginRuntime::new(&host.v8.v7.v6.v5.v4.v3.v1).diagnostics();
    assert_eq!(d.availability(), sdk::DiagnosticAvailability::Unsupported);
    let c = d.component_lazy(|| panic!("unsupported factory")).unwrap();
    assert_eq!(
        c.try_record_lazy(|| panic!("unsupported observation"))
            .unwrap(),
        sdk::RecordOutcome::Disabled
    );
    let mut api = API;
    api.contract_version = 900;
    let mut host = extended();
    host.diagnostics_api = &api;
    assert_eq!(
        PluginRuntime::new(&host.v8.v7.v6.v5.v4.v3.v1)
            .diagnostics()
            .availability(),
        sdk::DiagnosticAvailability::Unsupported
    );
    api.contract_version = 1;
    api.struct_size = size_of::<usize>();
    host.diagnostics_api = &api;
    assert_eq!(
        PluginRuntime::new(&host.v8.v7.v6.v5.v4.v3.v1)
            .diagnostics()
            .availability(),
        sdk::DiagnosticAvailability::Unsupported
    );
    let host = test_host_v7();
    assert_eq!(
        PluginRuntime::new(&host.v6.v5.v4.v3.v1)
            .diagnostics()
            .availability(),
        sdk::DiagnosticAvailability::Unsupported
    );
}
struct NativeDiagnosticsPlugin;
impl NativePlugin for NativeDiagnosticsPlugin {
    fn plugin_kind(&self) -> &str {
        "fixture.native_diagnostics"
    }
    fn register(
        &mut self,
        _: &Map<String, Json>,
        ctx: &mut PluginContext<'_>,
    ) -> nemo_relay_plugin::Result<()> {
        let d = ctx.runtime().clone().diagnostics();
        if !ENABLED.load(Ordering::Relaxed) {
            let c = d.component_lazy(|| panic!("disabled factory")).unwrap();
            c.try_record_lazy(|| panic!("disabled observation"))
                .unwrap();
            return Ok(());
        }
        let codes = [sdk::DiagnosticCode::Extension {
            name: "lookup",
            description: "lookup attempt",
            outcome: sdk::DiagnosticOutcome::Failure,
            allowed_classes: &[sdk::FailureClass::Unavailable],
        }];
        assert!(matches!(
            d.component_lazy(|| sdk::ComponentDeclaration {
                name: "empty",
                codes: &[],
            }),
            Err(sdk::DiagnosticError::Invalid)
        ));
        let c = d
            .component_lazy(|| sdk::ComponentDeclaration {
                name: "backend",
                codes: &codes,
            })
            .unwrap();
        for delta in [2, 3] {
            c.try_record_lazy(|| {
                if delta == 2 {
                    assert_eq!(
                        c.try_record_lazy(|| panic!("busy factory")),
                        Err(sdk::DiagnosticError::Busy)
                    );
                }
                sdk::Observation::failure(
                    1,
                    sdk::FailureClass::Unavailable,
                    "restricted",
                    NonZeroU64::new(delta).unwrap(),
                )
            })
            .unwrap();
        }
        assert_eq!(ACCEPTED.load(Ordering::SeqCst), 5);
        let cloned = c.clone();
        c.close().unwrap();
        assert_eq!(
            cloned
                .try_record_lazy(|| panic!("closed factory"))
                .unwrap_err(),
            sdk::DiagnosticError::Closed
        );
        assert!(matches!(
            d.reader(),
            Err(sdk::DiagnosticError::PermissionDenied)
        ));
        assert_eq!(d.delivery_stats().dropped, 3); // empty declaration, busy call, closed handle
        Ok(())
    }
}
#[test]
fn native_diagnostics_extension_survives_exported_plugins_owned_host_copy() {
    let _lock = begin_test();
    for enabled in [false, true] {
        ENABLED.store(enabled, Ordering::Relaxed);
        ACCEPTED.store(0, Ordering::SeqCst);
        let host = extended();
        let prefix = &host.v8.v7.v6.v5.v4.v3.v1;
        let mut plugin = NemoRelayNativePluginV1::default();
        assert_eq!(
            unsafe {
                nemo_relay_plugin::export_plugin(prefix, &mut plugin, NativeDiagnosticsPlugin)
            },
            NemoRelayStatus::Ok
        );
        let config = host_string(prefix, "{}");
        assert_eq!(
            unsafe {
                plugin.register.unwrap()(
                    plugin.user_data,
                    config,
                    NonNull::<NemoRelayNativePluginContext>::dangling().as_ptr(),
                )
            },
            NemoRelayStatus::Ok
        );
        unsafe {
            (prefix.string_free)(config);
            (plugin.drop.unwrap())(plugin.user_data);
            (prefix.string_free)(plugin.plugin_kind);
        }
    }
    assert_eq!(STRING_LIVE_COUNT.load(Ordering::SeqCst), 0);
}
#[test]
fn native_diagnostics_unknown_status_is_not_success() {
    assert_eq!(
        sdk::DiagnosticError::from_status(9999),
        sdk::DiagnosticError::Internal
    );
    assert_eq!(
        sdk::DiagnosticError::from_status(
            sdk::DiagnosticError::Read(sdk::DiagnosticReadReason::InvalidCursor).status()
        ),
        sdk::DiagnosticError::Read(sdk::DiagnosticReadReason::InvalidCursor)
    );
}
