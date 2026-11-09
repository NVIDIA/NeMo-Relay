// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The pi extension, embedded in the binary so installing it needs no checkout.
//!
//! `crates/cli/assets/pi-extension` is the npm workspace package and the one copy of
//! the source. It sits under this crate because `nemo-relay-cli` is published to
//! crates.io, and Cargo packages only files under the crate root.
//!
//! Only what pi loads is embedded: manifest, entry point, shared modules and the
//! standalone transport. The managed bundle in `crates/cli/src/daemon/managed` embeds
//! the same shared modules behind its own entry point.

/// One file of the vendored extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ExtensionFile {
    /// Path relative to the extension root, always `/`-separated.
    pub(crate) path: &'static str,
    /// The file's contents, embedded from `assets/pi-extension` at compile time.
    pub(crate) contents: &'static str,
}

macro_rules! extension_file {
    ($path:literal) => {
        ExtensionFile {
            path: $path,
            contents: include_str!(concat!("../../../assets/pi-extension/", $path)),
        }
    };
}

/// Every file a Relay-managed install writes.
pub(crate) const EXTENSION_FILES: &[ExtensionFile] = &[
    extension_file!("package.json"),
    extension_file!("index.ts"),
    extension_file!("src/argument-transform.ts"),
    extension_file!("src/extension.ts"),
    extension_file!("src/gateway.ts"),
    extension_file!("src/provider-redirect.ts"),
    extension_file!("src/tool-result.ts"),
    extension_file!("src/user-bash.ts"),
    extension_file!("src/standalone/config.ts"),
    extension_file!("src/standalone/transport.ts"),
];

/// The version an install records for the extension it wrote.
///
/// This crate's own version rather than a parse of the embedded `package.json`: the two are
/// tied by a test, so there is one value here and no fallible path at install time.
pub(crate) const EXTENSION_VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
#[path = "../../../tests/coverage/agents/pi_assets_tests.rs"]
mod tests;
