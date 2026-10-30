// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![deny(rustdoc::broken_intra_doc_links, rustdoc::private_intra_doc_links)]

//! First-party PII redaction plugin crate for NeMo Relay.

#[cfg(test)]
use std::sync::Mutex;

pub(crate) mod builtin;
pub mod component;
pub(crate) mod detectors;
pub(crate) mod local;
pub(crate) mod overlay;
pub(crate) mod trajectory;
pub(crate) mod trajectory_projection;

#[cfg(test)]
pub(crate) fn test_mutex() -> &'static Mutex<()> {
    static TEST_MUTEX: Mutex<()> = Mutex::new(());
    &TEST_MUTEX
}

#[cfg(test)]
#[allow(missing_docs)]
pub mod api {
    pub use nemo_relay_runtime::api::*;
}

#[cfg(test)]
#[allow(missing_docs)]
pub mod plugin {
    pub(crate) use crate::test_support::plugin_host::*;
    pub use nemo_relay_plugin_host::*;

    pub fn ensure_builtin_plugins_registered() -> Result<()> {
        nemo_relay_plugin_host::ensure_builtin_plugins_registered()?;
        crate::component::register_pii_redaction_component()
    }
}

#[cfg(test)]
#[allow(missing_docs)]
pub mod plugins {
    pub mod pii_redaction {
        pub use crate::component;

        #[cfg(test)]
        pub fn test_mutex() -> &'static std::sync::Mutex<()> {
            crate::test_mutex()
        }
    }
}

#[cfg(test)]
#[allow(missing_docs)]
pub mod shared_runtime {
    pub fn reset_runtime_owner_for_tests() {}
}

#[cfg(test)]
#[path = "../tests/support/mod.rs"]
pub(crate) mod test_support;
