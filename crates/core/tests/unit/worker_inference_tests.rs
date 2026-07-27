// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use serde_json::json;

use crate::plugin::{WorkerInferenceDescriptor, WorkerInferenceRegistry};

#[test]
fn inference_round_trips_json_and_receives_deadline() {
    let registry = WorkerInferenceRegistry::default();
    let _registration = registry
        .register(
            WorkerInferenceDescriptor::new("test-inference", "test.echo.v1").unwrap(),
            Arc::new(|request, timeout| {
                assert_eq!(timeout, Duration::from_millis(25));
                Ok(json!({"request": request}))
            }),
        )
        .unwrap();

    let inference = registry.resolve("test-inference", "test.echo.v1").unwrap();
    assert_eq!(
        inference
            .invoke(json!({"text": "hello"}), Duration::from_millis(25))
            .unwrap(),
        json!({"request": {"text": "hello"}})
    );
}

#[test]
fn registration_owns_inference_lifetime() {
    let registry = WorkerInferenceRegistry::default();
    let registration = registry
        .register(
            WorkerInferenceDescriptor::new("owned-inference", "test.echo.v1").unwrap(),
            Arc::new(|request, _| Ok(request)),
        )
        .unwrap();

    assert!(registry.resolve("owned-inference", "test.echo.v1").is_ok());
    drop(registration);
    assert!(registry.resolve("owned-inference", "test.echo.v1").is_err());
}

#[test]
fn duplicate_inference_names_are_rejected() {
    let registry = WorkerInferenceRegistry::default();
    let _registration = registry
        .register(
            WorkerInferenceDescriptor::new("duplicate-inference", "test.echo.v1").unwrap(),
            Arc::new(|request, _| Ok(request)),
        )
        .unwrap();
    let duplicate = registry
        .register(
            WorkerInferenceDescriptor::new("duplicate-inference", "test.other.v1").unwrap(),
            Arc::new(|request, _| Ok(request)),
        )
        .err()
        .expect("duplicate inference names must fail");

    assert!(duplicate.to_string().contains("already registered"));
}

#[test]
fn inference_names_are_normalized_consistently() {
    let registry = WorkerInferenceRegistry::default();
    let _registration = registry
        .register(
            WorkerInferenceDescriptor::new("  normalized-inference  ", "  test.echo.v1  ").unwrap(),
            Arc::new(|request, _| Ok(request)),
        )
        .unwrap();

    assert!(
        registry
            .resolve(" normalized-inference ", " test.echo.v1 ")
            .is_ok()
    );
}

#[test]
fn inference_contract_mismatch_is_rejected_before_invocation() {
    let registry = WorkerInferenceRegistry::default();
    let _registration = registry
        .register(
            WorkerInferenceDescriptor::new("detector", "test.detector.v1").unwrap(),
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
fn registries_isolate_inference_names_between_hosts() {
    let first = WorkerInferenceRegistry::default();
    let second = WorkerInferenceRegistry::default();
    let _first_registration = first
        .register(
            WorkerInferenceDescriptor::new("shared-name", "test.echo.v1").unwrap(),
            Arc::new(|request, _| Ok(request)),
        )
        .unwrap();

    assert!(first.resolve("shared-name", "test.echo.v1").is_ok());
    assert!(second.resolve("shared-name", "test.echo.v1").is_err());
}
