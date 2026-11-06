// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Validate component executor configuration at the Rust authoring boundary.

use nemo_relay_plugin::NativeExecutorConfig;
use serde_json::Map;

#[test]
fn native_executor_config_accepts_only_positive_integer_overrides() {
    assert_eq!(NativeExecutorConfig::default().worker_threads, 2);
    assert_eq!(
        NativeExecutorConfig::default()
            .with_component_config(&Map::new())
            .unwrap()
            .worker_threads,
        2
    );
    assert_eq!(
        NativeExecutorConfig::default()
            .with_component_config(&Map::from_iter([(
                "executor".into(),
                serde_json::json!({"worker_threads": 3}),
            )]))
            .unwrap()
            .worker_threads,
        3
    );
    for config in [
        serde_json::json!(true),
        serde_json::json!({"worker_threads": 0}),
        serde_json::json!({"worker_threads": -1}),
        serde_json::json!({"worker_threads": true}),
        serde_json::json!({"worker_threads": 1.5}),
        serde_json::json!({"worker_threads": "two"}),
    ] {
        assert!(
            NativeExecutorConfig::default()
                .with_component_config(&Map::from_iter([("executor".into(), config)]))
                .is_err()
        );
    }
}
