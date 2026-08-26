// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The pi extension, vendored into the binary so installing it needs no checkout.
//!
//! `integrations/pi` stays the source of truth; this is a copy of it, and the copy exists
//! for a packaging reason rather than a design one. `nemo-relay-cli` is published to
//! crates.io, and Cargo packages only files under the crate root -- so
//! `include_str!("../../../../../integrations/pi/index.ts")` compiles in a workspace
//! checkout and then fails to build from the published tarball, where that path does not
//! exist. A build script that copies into `OUT_DIR` hits the same wall for the same
//! reason: the source it would copy is not in the tarball either.
//!
//! Regenerate with `just sync-pi-extension`. The drift test fails when the two diverge,
//! and skips itself when `integrations/pi` is absent -- which is exactly the
//! published-crate case this vendoring exists to serve.
//!
//! Only what pi loads is vendored: the manifest, the entry point, and `src/`. `test/` and
//! `tsconfig.json` have no runtime role, and vendoring the README would mean a re-sync on
//! every wording change to a file nothing reads at run time.

/// One file of the vendored extension.
pub(crate) struct ExtensionFile {
    /// Path relative to the extension root, always `/`-separated.
    pub(crate) path: &'static str,
    /// The file's contents, byte-identical to its counterpart in `integrations/pi`.
    pub(crate) contents: &'static str,
}

/// Every file a Relay-managed install writes.
///
/// Hand-maintained rather than generated: seven entries are cheaper to read than a
/// generator, and a source file nobody added here fails the drift test by name.
pub(crate) const EXTENSION_FILES: &[ExtensionFile] = &[
    ExtensionFile {
        path: "package.json",
        contents: include_str!("../../../assets/pi-extension/package.json"),
    },
    ExtensionFile {
        path: "index.ts",
        contents: include_str!("../../../assets/pi-extension/index.ts"),
    },
    ExtensionFile {
        path: "src/argument-transform.ts",
        contents: include_str!("../../../assets/pi-extension/src/argument-transform.ts"),
    },
    ExtensionFile {
        path: "src/gateway-client.ts",
        contents: include_str!("../../../assets/pi-extension/src/gateway-client.ts"),
    },
    ExtensionFile {
        path: "src/pi-hook-types.ts",
        contents: include_str!("../../../assets/pi-extension/src/pi-hook-types.ts"),
    },
    ExtensionFile {
        path: "src/provider-redirect.ts",
        contents: include_str!("../../../assets/pi-extension/src/provider-redirect.ts"),
    },
    ExtensionFile {
        path: "src/user-bash.ts",
        contents: include_str!("../../../assets/pi-extension/src/user-bash.ts"),
    },
];

/// The version an install records for the extension it wrote.
///
/// This crate's own version rather than a parse of the vendored `package.json`: the two are
/// tied by a test, so there is one value here and no fallible path at install time. They are
/// meant to move together -- the repository's version recipe already bumps `integrations/pi`
/// alongside the crates -- and asserting it is not theoretical, because a merge once left pi
/// a release behind with no conflict to show for it.
pub(crate) const EXTENSION_VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
#[path = "../../../tests/coverage/agents/pi_assets_tests.rs"]
mod tests;
