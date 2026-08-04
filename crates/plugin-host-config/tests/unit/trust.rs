// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fs;

use tempfile::tempdir;

use super::*;

#[test]
fn configured_public_key_values_are_redacted_from_trust_errors() {
    let temp = tempdir().unwrap();
    let artifact = temp.path().join("artifact.bin");
    let signature = temp.path().join("signature.txt");
    let manifest = temp.path().join("relay-plugin.toml");
    fs::write(&artifact, b"artifact").unwrap();
    fs::write(&signature, "AA==").unwrap();
    let secret = "do-not-leak-trusted-key";

    let failure = verify_signature(
        manifest.to_string_lossy().as_ref(),
        &artifact,
        "signature.txt",
        &[secret.to_owned()],
    )
    .unwrap_err();
    let rendered = failure.display("fixture").to_string();
    assert!(!rendered.contains(secret), "{rendered}");
}
