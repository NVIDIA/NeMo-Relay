// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};

use super::*;

/// The extension directory inside this crate: the one real copy of every file pi loads.
///
/// Resolvable from the published tarball as well as from a workspace checkout, because it
/// lives under the crate root -- which is the whole reason the files are here.
fn assets_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("assets")
        .join("pi-extension")
}

/// `integrations/pi`, or `None` when there is no workspace to look at.
///
/// Absent is not a failure. The published crate is built from a tarball containing the
/// crate root and nothing above it, so there is no `integrations/pi` to resolve. It is
/// still a sound discriminator: no registry checkout has one two levels above the manifest.
fn workspace_view() -> Option<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()?
        .parent()?
        .join("integrations")
        .join("pi");
    root.is_dir().then_some(root)
}

/// The files pi loads, discovered rather than listed, so a new one shows up here.
fn extension_files_on_disk(root: &Path) -> Vec<String> {
    let mut found = vec!["package.json".to_string(), "index.ts".to_string()];
    let mut sources: Vec<String> = std::fs::read_dir(root.join("src"))
        .unwrap_or_else(|error| panic!("{}/src should be readable: {error}", root.display()))
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.ends_with(".ts"))
        .map(|name| format!("src/{name}"))
        .collect();
    sources.sort();
    found.extend(sources);
    found
}

/// A source file nobody added to `EXTENSION_FILES` is a file `install pi` silently omits.
///
/// Runs without the workspace too, since it reads the crate's own assets -- so it holds for
/// the published crate, where an omission would be undiagnosable.
#[test]
fn every_file_pi_loads_is_embedded() {
    let embedded: Vec<&str> = EXTENSION_FILES.iter().map(|file| file.path).collect();

    assert_eq!(
        embedded,
        extension_files_on_disk(&assets_root()),
        "crates/cli/assets/pi-extension no longer matches EXTENSION_FILES in \
         crates/cli/src/agents/pi/assets.rs; add or remove the matching entries"
    );
}

/// `integrations/pi` is a symlinked view of this crate's assets, and this catches a
/// checkout where that did not survive.
///
/// The failure mode is specific and worth naming: git on Windows without `core.symlinks`
/// writes a symlink as a *text file containing the target path*. The crate is unaffected --
/// `include_str!` reads the real file either way -- so nothing in the Rust build notices,
/// and the damage surfaces much later as `just test-pi` importing a path string. Failing
/// here says what actually went wrong.
#[test]
fn the_workspace_view_resolves_to_the_embedded_files() {
    let Some(root) = workspace_view() else {
        return;
    };

    for file in EXTENSION_FILES {
        let view = root.join(file.path);
        let actual = std::fs::read_to_string(&view)
            .unwrap_or_else(|error| panic!("{} should be readable: {error}", view.display()));
        assert_eq!(
            file.contents, actual,
            "integrations/pi/{} does not resolve to crates/cli/assets/pi-extension/{}. It \
             should be a symlink to it; a checkout that does not support symlinks writes \
             the target path as file contents instead",
            file.path, file.path
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
        "the pi extension manifest and the CLI crate have drifted apart; `just set-version` \
         bumps both"
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
