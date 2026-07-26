// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use serde_json::json;

use crate::plugin::{InferenceProviderDescriptor, InferenceProviderRegistry};

#[test]
fn provider_round_trips_json_and_receives_deadline() {
    let registry = InferenceProviderRegistry::default();
    let _registration = registry
        .register(
            InferenceProviderDescriptor::new("test-provider", "test.echo.v1").unwrap(),
            Arc::new(|request, timeout| {
                assert_eq!(timeout, Duration::from_millis(25));
                Ok(json!({"request": request}))
            }),
        )
        .unwrap();

    let provider = registry.resolve("test-provider", "test.echo.v1").unwrap();
    assert_eq!(
        provider
            .invoke(json!({"text": "hello"}), Duration::from_millis(25))
            .unwrap(),
        json!({"request": {"text": "hello"}})
    );
}

#[test]
fn registration_owns_provider_lifetime() {
    let registry = InferenceProviderRegistry::default();
    let registration = registry
        .register(
            InferenceProviderDescriptor::new("owned-provider", "test.echo.v1").unwrap(),
            Arc::new(|request, _| Ok(request)),
        )
        .unwrap();

    assert!(registry.resolve("owned-provider", "test.echo.v1").is_ok());
    drop(registration);
    assert!(registry.resolve("owned-provider", "test.echo.v1").is_err());
}

#[test]
fn duplicate_provider_names_are_rejected() {
    let registry = InferenceProviderRegistry::default();
    let _registration = registry
        .register(
            InferenceProviderDescriptor::new("duplicate-provider", "test.echo.v1").unwrap(),
            Arc::new(|request, _| Ok(request)),
        )
        .unwrap();
    let duplicate = registry
        .register(
            InferenceProviderDescriptor::new("duplicate-provider", "test.other.v1").unwrap(),
            Arc::new(|request, _| Ok(request)),
        )
        .err()
        .expect("duplicate provider names must fail");

    assert!(duplicate.to_string().contains("already registered"));
}

#[test]
fn provider_names_are_normalized_consistently() {
    let registry = InferenceProviderRegistry::default();
    let _registration = registry
        .register(
            InferenceProviderDescriptor::new("  normalized-provider  ", "  test.echo.v1  ")
                .unwrap(),
            Arc::new(|request, _| Ok(request)),
        )
        .unwrap();

    assert!(
        registry
            .resolve(" normalized-provider ", " test.echo.v1 ")
            .is_ok()
    );
}

#[test]
fn provider_contract_mismatch_is_rejected_before_invocation() {
    let registry = InferenceProviderRegistry::default();
    let _registration = registry
        .register(
            InferenceProviderDescriptor::new("detector", "test.detector.v1").unwrap(),
            Arc::new(|request, _| Ok(request)),
        )
        .unwrap();

    let error = registry
        .resolve("detector", "test.embedding.v1")
        .err()
        .expect("mismatched contracts must fail");
    assert!(error.to_string().contains("test.detector.v1"));
    assert!(error.to_string().contains("test.embedding.v1"));
}

#[test]
fn registries_isolate_provider_names_between_hosts() {
    let first = InferenceProviderRegistry::default();
    let second = InferenceProviderRegistry::default();
    let _first_registration = first
        .register(
            InferenceProviderDescriptor::new("shared-name", "test.echo.v1").unwrap(),
            Arc::new(|request, _| Ok(request)),
        )
        .unwrap();

    assert!(first.resolve("shared-name", "test.echo.v1").is_ok());
    assert!(second.resolve("shared-name", "test.echo.v1").is_err());
}
