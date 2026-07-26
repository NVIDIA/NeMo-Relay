// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use serde_json::json;

use crate::plugin::{
    deregister_local_model_provider, local_model_provider, register_local_model_provider_tracked,
};

#[test]
fn provider_round_trips_json_and_receives_deadline() {
    let registration_id = register_local_model_provider_tracked(
        "test-provider",
        Arc::new(|request, timeout| {
            assert_eq!(timeout, Duration::from_millis(25));
            Ok(json!({"request": request}))
        }),
    )
    .unwrap();

    let provider = local_model_provider("test-provider").unwrap();
    assert_eq!(
        provider(json!({"text": "hello"}), Duration::from_millis(25)).unwrap(),
        json!({"request": {"text": "hello"}})
    );
    assert!(deregister_local_model_provider("test-provider", registration_id).unwrap());
}

#[test]
fn ownership_token_does_not_remove_another_registration() {
    let registration_id =
        register_local_model_provider_tracked("owned-provider", Arc::new(|request, _| Ok(request)))
            .unwrap();

    assert!(!deregister_local_model_provider("owned-provider", registration_id + 1).unwrap());
    assert!(local_model_provider("owned-provider").is_ok());
    assert!(deregister_local_model_provider("owned-provider", registration_id).unwrap());
}

#[test]
fn duplicate_provider_names_are_rejected() {
    let registration_id = register_local_model_provider_tracked(
        "duplicate-provider",
        Arc::new(|request, _| Ok(request)),
    )
    .unwrap();
    let duplicate = register_local_model_provider_tracked(
        "duplicate-provider",
        Arc::new(|request, _| Ok(request)),
    )
    .unwrap_err();

    assert!(duplicate.to_string().contains("already registered"));
    assert!(deregister_local_model_provider("duplicate-provider", registration_id).unwrap());
}

#[test]
fn provider_names_are_normalized_consistently() {
    let registration_id = register_local_model_provider_tracked(
        "  normalized-provider  ",
        Arc::new(|request, _| Ok(request)),
    )
    .unwrap();

    assert!(local_model_provider(" normalized-provider ").is_ok());
    assert!(deregister_local_model_provider(" normalized-provider ", registration_id).unwrap());
}
