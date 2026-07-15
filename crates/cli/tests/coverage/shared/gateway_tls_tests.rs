// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

struct Environment {
    _guard: std::sync::MutexGuard<'static, ()>,
    previous: Vec<(&'static str, Option<std::ffi::OsString>)>,
}

impl Environment {
    fn set(values: &[(&'static str, &std::ffi::OsStr)]) -> Self {
        let guard = crate::test_support::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let previous = values
            .iter()
            .map(|(name, _)| (*name, std::env::var_os(name)))
            .collect();
        for (name, value) in values {
            unsafe { std::env::set_var(name, value) };
        }
        Self {
            _guard: guard,
            previous,
        }
    }
}

impl Drop for Environment {
    fn drop(&mut self) {
        for (name, value) in self.previous.drain(..) {
            unsafe {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
        }
    }
}

#[test]
fn per_user_tls_identity_round_trips_as_server_and_pinned_client_configs() {
    let temp = tempfile::tempdir().unwrap();
    let _environment = Environment::set(&[
        ("XDG_CONFIG_HOME", temp.path().as_os_str()),
        ("HOME", temp.path().as_os_str()),
    ]);

    let identity = RelayTlsIdentity::load_or_create().unwrap();
    identity.server_config().unwrap();
    identity.client_config().unwrap();
    let reloaded = RelayTlsIdentity::load().unwrap();
    reloaded.server_config().unwrap();
    reloaded.client_config().unwrap();
}
