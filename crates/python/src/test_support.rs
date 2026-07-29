// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::OsString;
use std::sync::{Mutex, MutexGuard, OnceLock};

use pyo3::Python;

const BINDING_KIND_ENV: &str = "NEMO_RELAY_BINDING_KIND";
const RUNTIME_OWNER_ENV: &str = "NEMO_RELAY_RUNTIME_OWNER";
const XDG_CONFIG_HOME_ENV: &str = "XDG_CONFIG_HOME";

fn python_test_lock() -> &'static Mutex<()> {
    static PYTHON_TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    PYTHON_TEST_LOCK.get_or_init(|| Mutex::new(()))
}

pub(crate) fn lock_python_test() -> MutexGuard<'static, ()> {
    python_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn clear_runtime_owner_env() {
    unsafe {
        std::env::remove_var(RUNTIME_OWNER_ENV);
        std::env::remove_var(BINDING_KIND_ENV);
    }
}

pub(crate) struct PythonTestGuard {
    _lock: MutexGuard<'static, ()>,
    binding_kind: Option<OsString>,
    runtime_owner: Option<OsString>,
    xdg_config_home: Option<OsString>,
}

impl Drop for PythonTestGuard {
    fn drop(&mut self) {
        unsafe {
            match &self.runtime_owner {
                Some(value) => std::env::set_var(RUNTIME_OWNER_ENV, value),
                None => std::env::remove_var(RUNTIME_OWNER_ENV),
            };
            match &self.binding_kind {
                Some(value) => std::env::set_var(BINDING_KIND_ENV, value),
                None => std::env::remove_var(BINDING_KIND_ENV),
            };
            match &self.xdg_config_home {
                Some(value) => std::env::set_var(XDG_CONFIG_HOME_ENV, value),
                None => std::env::remove_var(XDG_CONFIG_HOME_ENV),
            };
        }
    }
}

pub(crate) fn init_python_test_locked(lock: MutexGuard<'static, ()>) -> PythonTestGuard {
    let binding_kind = std::env::var_os(BINDING_KIND_ENV);
    let runtime_owner = std::env::var_os(RUNTIME_OWNER_ENV);
    let xdg_config_home = std::env::var_os(XDG_CONFIG_HOME_ENV);
    clear_runtime_owner_env();
    let isolated_config_home =
        std::env::temp_dir().join(format!("nemo-relay-python-tests-{}", std::process::id()));
    unsafe {
        std::env::set_var(XDG_CONFIG_HOME_ENV, isolated_config_home);
    }
    Python::initialize();
    PythonTestGuard {
        _lock: lock,
        binding_kind,
        runtime_owner,
        xdg_config_home,
    }
}

pub(crate) fn init_python_test() -> PythonTestGuard {
    init_python_test_locked(lock_python_test())
}
