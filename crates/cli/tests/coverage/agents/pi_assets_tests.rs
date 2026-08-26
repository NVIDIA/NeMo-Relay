// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};

use super::*;

/// `integrations/pi`, or `None` when there is no repository to compare against.
///
/// Absent is not a failure. The published crate is built from a tarball that contains the
/// crate root and nothing above it, which is the whole reason the extension is vendored --
/// so a test that failed on a missing source would fail exactly where the vendoring is
/// doing its job. It is still a sound discriminator: no registry checkout has an
/// `integrations/pi` two levels above the manifest.
fn source_root() -> Option<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()?
        .parent()?
        .join("integrations")
        .join("pi");
    root.is_dir().then_some(root)
}

/// The files a sync copies, discovered rather than listed, so a new one shows up here.
fn source_files(root: &Path) -> Vec<String> {
    let mut found = vec!["package.json".to_string(), "index.ts".to_string()];
    let mut sources: Vec<String> = std::fs::read_dir(root.join("src"))
        .expect("integrations/pi/src should be readable")
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.ends_with(".ts"))
        .map(|name| format!("src/{name}"))
        .collect();
    sources.sort();
    found.extend(sources);
    found
}

#[test]
fn vendored_extension_covers_every_file_pi_loads() {
    let Some(root) = source_root() else {
        return;
    };

    let embedded: Vec<&str> = EXTENSION_FILES.iter().map(|file| file.path).collect();
    let expected = source_files(&root);

    assert_eq!(
        embedded, expected,
        "the vendored pi extension no longer covers `integrations/pi`. Run \
         `just sync-pi-extension`, then add or remove the matching entries in \
         EXTENSION_FILES in crates/cli/src/agents/pi/assets.rs"
    );
}

#[test]
fn vendored_extension_matches_its_source_byte_for_byte() {
    let Some(root) = source_root() else {
        return;
    };

    for file in EXTENSION_FILES {
        let source = root.join(file.path);
        let actual = std::fs::read_to_string(&source)
            .unwrap_or_else(|error| panic!("{} should be readable: {error}", source.display()));
        assert_eq!(
            file.contents, actual,
            "the vendored copy of {} is stale. Run `just sync-pi-extension`",
            file.path
        );
    }
}

/// The version an install records has to be the version it installs.
///
/// Not theoretical: a merge on this branch once left `integrations/pi` a release behind
/// every other package, with no conflict to show for it, because the version recipe could
/// not touch a file that did not exist on the other side. This runs without the repository
/// too -- it reads the vendored manifest, so it holds for the published crate as well.
#[test]
fn vendored_manifest_declares_the_version_an_install_records() {
    let manifest = EXTENSION_FILES
        .iter()
        .find(|file| file.path == "package.json")
        .expect("the vendored extension must carry its manifest");
    let parsed: serde_json::Value =
        serde_json::from_str(manifest.contents).expect("the vendored manifest must be valid JSON");

    assert_eq!(
        parsed.get("version").and_then(serde_json::Value::as_str),
        Some(EXTENSION_VERSION),
        "integrations/pi/package.json and the CLI crate have drifted apart; `just \
         set-version` bumps both, then run `just sync-pi-extension`"
    );
}

/// pi finds the extension by its manifest name, so the vendored copy has to keep it.
#[test]
fn vendored_manifest_keeps_the_name_discovery_matches_on() {
    let manifest = EXTENSION_FILES
        .iter()
        .find(|file| file.path == "package.json")
        .expect("the vendored extension must carry its manifest");
    let parsed: serde_json::Value = serde_json::from_str(manifest.contents).unwrap();

    assert_eq!(
        parsed.get("name").and_then(serde_json::Value::as_str),
        Some("nemo-relay-pi")
    );
    assert_eq!(
        parsed
            .get("pi")
            .and_then(|pi| pi.get("extensions"))
            .and_then(serde_json::Value::as_array)
            .map(Vec::as_slice),
        Some(["./index.ts".into()].as_slice()),
        "pi resolves the entry point through this manifest key"
    );
}
