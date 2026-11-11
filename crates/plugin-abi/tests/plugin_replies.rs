// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Foreign plugin replies fail closed without unwinding through the host.
use nemo_relay_plugin_abi::{
    Json,
    future::FfiFuture,
    transfer::{OwnedWire, Transfer, Value},
    v1_0::{self as a, ConfigDiagnostic, PluginBox},
    wire::{WirePlugin, XabiV1OwnedTraitWirePlugin},
    xabi::{self, XabiType},
};

struct PluginReplies(bool);
impl WirePlugin for PluginReplies {
    fn allows_multiple_components(&self) -> Result<Value<bool>, Value<a::Error>> {
        if self.0 {
            Err(Value(a::Error::new("fixture", "capability failed")))
        } else {
            Ok(Value(true))
        }
    }

    fn validate(&self, _: Value<Json>) -> Result<Value<Vec<ConfigDiagnostic>>, Value<a::Error>> {
        if self.0 {
            Err(Value(a::Error::new("fixture", "validation failed")))
        } else {
            Ok(Value(vec![]))
        }
    }

    fn register(
        &self,
        _: Value<Json>,
        _: Value<a::RuntimeBox>,
    ) -> Result<Value<FfiFuture<a::Result<a::RegisteredComponent>>>, Value<a::Error>> {
        panic!("reply tests must not register components")
    }
}

/// Return an invalid boolean byte while retaining a valid source-owned allocation.
unsafe extern "C" fn malformed_capability(
    _: *mut std::ffi::c_void,
    output: *mut xabi::XabiOwnedBytes,
) -> i32 {
    let payload = Value(false).into_payload();
    unsafe {
        *(*payload.ptr.cast::<OwnedWire>()).data.cast::<u8>() = 2;
        output.write(payload);
    }
    xabi::OK
}

/// Return an invalid payload length; the receiver must still release its allocation.
unsafe extern "C" fn malformed_validation(
    _: *mut std::ffi::c_void,
    _: *const OwnedWire,
    output: *mut xabi::XabiOwnedBytes,
) -> i32 {
    // The caller retains rollback ownership of the unused argument.
    let mut payload = Value(Vec::<ConfigDiagnostic>::new()).into_payload();
    payload.len = 0;
    unsafe { output.write(payload) };
    xabi::OK
}

/// Preserve successful replies and report exported errors as failed validation.
#[test]
fn plugin_reply_errors_fail_closed() {
    for fails in [false, true] {
        let mut wire = XabiV1OwnedTraitWirePlugin::new(PluginReplies(fails)).into_wire();
        let plugin = unsafe { PluginBox::take(&mut wire) }.unwrap();
        unsafe { PluginBox::release(&mut wire) };
        assert_eq!(plugin.allows_multiple_components(), !fails);
        let diagnostics = plugin.validate(Json::Null);
        if fails {
            assert_eq!(diagnostics.len(), 1);
            assert!(diagnostics[0].is_error);
            assert_eq!(diagnostics[0].code, "fixture");
            assert_eq!(diagnostics[0].message, "validation failed");
            assert!(diagnostics[0].component.is_none());
            assert!(diagnostics[0].field.is_none());
        } else {
            assert!(diagnostics.is_empty());
        }
    }
}

/// Malformed foreign values become a conservative capability and an ABI diagnostic.
#[test]
fn malformed_plugin_replies_do_not_panic() {
    let mut wire = XabiV1OwnedTraitWirePlugin::new(PluginReplies(false)).into_wire();
    unsafe {
        (*wire.vtable).allows_multiple_components = malformed_capability;
        (*wire.vtable).validate = malformed_validation;
    }
    let plugin = unsafe { PluginBox::take(&mut wire) }.unwrap();
    unsafe { PluginBox::release(&mut wire) };
    assert!(!plugin.allows_multiple_components());
    let diagnostics = plugin.validate(Json::String("owned argument".into()));
    assert_eq!(diagnostics.len(), 1);
    assert!(diagnostics[0].is_error);
    assert_eq!(diagnostics[0].code, "abi");
    assert!(diagnostics[0].message.contains("invalid typed payload"));
}
