// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compile and exercise the Adaptive examples displayed in the API reference.

static DOCUMENTATION_REGISTRATION_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn reset_runtime_state() {
    use nemo_relay_runtime::api::runtime::{NemoRelayContextState, global_context};

    let context = global_context();
    let mut state = context.write().unwrap();
    *state = NemoRelayContextState::new();
}

mod documentation_examples_adaptive_configuration {
    include!("../../../docs/reference/api/_code/rust/adaptive_configuration.rs");

    #[test]
    fn registers_and_shuts_down() {
        let _guard = super::DOCUMENTATION_REGISTRATION_LOCK.blocking_lock();
        super::reset_runtime_state();
        let before = nemo_relay_runtime::api::registry::list_runtime_registrations(None).unwrap();
        main().unwrap();
        assert_eq!(
            nemo_relay_runtime::api::registry::list_runtime_registrations(None).unwrap(),
            before
        );
    }
}

mod documentation_examples_acg {
    include!("../../../docs/reference/api/_code/rust/acg.rs");

    #[test]
    fn registers_and_shuts_down() {
        let _guard = super::DOCUMENTATION_REGISTRATION_LOCK.blocking_lock();
        super::reset_runtime_state();
        let before = nemo_relay_runtime::api::registry::list_runtime_registrations(None).unwrap();
        main().unwrap();
        assert_eq!(
            nemo_relay_runtime::api::registry::list_runtime_registrations(None).unwrap(),
            before
        );
    }
}

mod documentation_examples_adaptive_hints {
    include!("../../../docs/reference/api/_code/rust/adaptive_hints.rs");

    #[test]
    fn registers_and_shuts_down() {
        let _guard = super::DOCUMENTATION_REGISTRATION_LOCK.blocking_lock();
        super::reset_runtime_state();
        let before = nemo_relay_runtime::api::registry::list_runtime_registrations(None).unwrap();
        main().unwrap();
        assert_eq!(
            nemo_relay_runtime::api::registry::list_runtime_registrations(None).unwrap(),
            before
        );
    }
}

mod documentation_examples_response_cache {
    include!("../../../docs/reference/api/_code/rust/response_cache.rs");

    #[test]
    fn registers_and_shuts_down() {
        let _guard = super::DOCUMENTATION_REGISTRATION_LOCK.blocking_lock();
        super::reset_runtime_state();
        let before = nemo_relay_runtime::api::registry::list_runtime_registrations(None).unwrap();
        main().unwrap();
        assert_eq!(
            nemo_relay_runtime::api::registry::list_runtime_registrations(None).unwrap(),
            before
        );
    }
}
