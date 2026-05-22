// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for error in the NeMo Relay adaptive crate.

use super::*;

#[test]
fn test_invalid_intent_display() {
    let e = AcgError::InvalidIntent("bad stability score".into());
    assert_eq!(format!("{e}"), "invalid intent: bad stability score");
}

#[test]
fn test_serialization_from_serde_json() {
    let serde_err = serde_json::from_str::<String>("bad").unwrap_err();
    let e = AcgError::from(serde_err);
    let msg = format!("{e}");
    assert!(msg.starts_with("serialization error:"), "got: {msg}");
}

#[test]
fn test_internal_display() {
    let e = AcgError::Internal("lock poisoned".into());
    assert_eq!(format!("{e}"), "internal error: lock poisoned");
}

#[test]
fn test_error_is_std_error() {
    let e: Box<dyn std::error::Error> = Box::new(AcgError::Internal("test".into()));
    assert!(e.to_string().contains("internal error"));
}

#[test]
fn test_error_debug() {
    let e = AcgError::InvalidIntent("x".into());
    let debug = format!("{e:?}");
    assert!(debug.contains("InvalidIntent"));
}

#[test]
fn test_plugin_already_registered_display() {
    let e = AcgError::PluginAlreadyRegistered("anthropic".into());
    assert_eq!(format!("{e}"), "plugin already registered: anthropic");
}

#[test]
fn test_plugin_not_found_display() {
    let e = AcgError::PluginNotFound("openai".into());
    assert_eq!(format!("{e}"), "plugin not found: openai");
}

#[test]
fn test_translation_failed_display() {
    let e = AcgError::TranslationFailed("codec mismatch".into());
    assert_eq!(format!("{e}"), "plugin translation error: codec mismatch");
}

#[test]
fn test_ir_construction_error_display() {
    let e = AcgError::IrConstructionError("missing messages".into());
    assert_eq!(format!("{e}"), "IR construction error: missing messages");
}
