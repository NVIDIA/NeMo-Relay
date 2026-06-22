// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Coverage tests for error in the NeMo Relay adaptive crate.

use super::*;
use nemo_relay::plugin::PluginError;

#[test]
fn test_not_found_display() {
    let e = AdaptiveError::NotFound("x".into());
    assert_eq!(format!("{e}"), "not found: x");
}

#[test]
fn test_invalid_config_display() {
    let e = AdaptiveError::InvalidConfig("bad".into());
    assert_eq!(format!("{e}"), "invalid config: bad");
}

#[test]
fn test_storage_display() {
    let e = AdaptiveError::Storage("y".into());
    assert_eq!(format!("{e}"), "storage error: y");
}

#[test]
fn test_internal_display() {
    let e = AdaptiveError::Internal("z".into());
    assert_eq!(format!("{e}"), "internal error: z");
}

#[test]
fn test_serialization_from_serde_json() {
    let serde_err = serde_json::from_str::<String>("bad").unwrap_err();
    let e = AdaptiveError::from(serde_err);
    let msg = format!("{e}");
    assert!(msg.starts_with("serialization error:"), "got: {msg}");
}

#[test]
fn test_registration_failed_display() {
    let e = AdaptiveError::RegistrationFailed("subscriber".into());
    assert_eq!(format!("{e}"), "registration failed: subscriber");
}

#[test]
fn test_channel_closed_display() {
    let e = AdaptiveError::ChannelClosed("receiver dropped".into());
    assert_eq!(format!("{e}"), "channel closed: receiver dropped");
}

#[test]
fn test_error_is_std_error() {
    let e: Box<dyn std::error::Error> = Box::new(AdaptiveError::Internal("test".into()));
    assert!(e.to_string().contains("internal error"));
}

#[test]
fn test_plugin_error_conversion_maps_all_variants() {
    let invalid = AdaptiveError::from(PluginError::InvalidConfig("bad config".into()));
    assert_eq!(format!("{invalid}"), "invalid config: bad config");

    let conflict = AdaptiveError::from(PluginError::Conflict("duplicate plugin".into()));
    assert_eq!(
        format!("{conflict}"),
        "registration failed: duplicate plugin"
    );

    let missing = AdaptiveError::from(PluginError::NotFound("missing plugin".into()));
    assert_eq!(format!("{missing}"), "not found: missing plugin");

    let serde_error = serde_json::from_str::<serde_json::Value>("{]").unwrap_err();
    let serialization = AdaptiveError::from(PluginError::Serialization(serde_error));
    assert!(format!("{serialization}").starts_with("serialization error:"));

    let internal = AdaptiveError::from(PluginError::Internal("poisoned".into()));
    assert_eq!(format!("{internal}"), "internal error: poisoned");

    let registration = AdaptiveError::from(PluginError::RegistrationFailed("subscriber".into()));
    assert_eq!(format!("{registration}"), "registration failed: subscriber");
}

#[cfg(feature = "redis-backend")]
#[test]
fn test_redis_error_variant_exists() {
    // Verify that the Redis variant exists and displays correctly.
    // We construct a redis error via an invalid URL to get a RedisError.
    let redis_err = redis::Client::open("invalid://url").unwrap_err();
    let e = AdaptiveError::Redis(redis_err);
    let msg = format!("{e}");
    assert!(msg.starts_with("redis error:"), "got: {msg}");
}
