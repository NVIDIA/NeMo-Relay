// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};

use super::super::DESCRIPTOR;
use super::super::alignment::UPSTREAM_BASE_URL_HEADER;
use super::super::launch::{PI_ANTHROPIC_UPSTREAM_ENV, PI_GATEWAY_URL_ENV, PI_OPENAI_UPSTREAM_ENV};
use super::*;
use crate::provider_auth::{TRANSPARENT_PROXY_CREDENTIAL_ENV, TRANSPARENT_PROXY_CREDENTIAL_HEADER};

fn assets_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("assets")
        .join("pi-extension")
}

fn typescript_files(root: &Path, directory: &str) -> Vec<String> {
    let mut files: Vec<String> = std::fs::read_dir(root.join(directory))
        .unwrap_or_else(|error| {
            panic!("{}/{directory} should be readable: {error}", root.display())
        })
        .flatten()
        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_file()))
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.ends_with(".ts"))
        .map(|name| format!("{directory}/{name}"))
        .collect();
    files.sort();
    files
}

fn embedded_text() -> String {
    EXTENSION_FILES
        .iter()
        .map(|file| file.contents)
        .collect::<String>()
}

#[test]
fn every_file_pi_loads_is_embedded() {
    let root = assets_root();
    let mut expected = vec!["index.ts".to_string(), "package.json".to_string()];
    expected.extend(typescript_files(&root, "src"));
    expected.extend(typescript_files(&root, "src/standalone"));
    expected.sort();

    let mut embedded: Vec<String> = EXTENSION_FILES
        .iter()
        .map(|file| file.path.to_string())
        .collect();
    embedded.sort();

    assert_eq!(
        embedded, expected,
        "crates/cli/assets/pi-extension no longer matches EXTENSION_FILES in \
         crates/cli/src/agents/pi/assets.rs; add or remove the matching entries"
    );
}

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

#[test]
fn vendored_manifest_pins_the_pi_release_the_descriptor_verifies() {
    let manifest = EXTENSION_FILES
        .iter()
        .find(|file| file.path == "package.json")
        .expect("the vendored extension must carry its manifest");
    let parsed: serde_json::Value = serde_json::from_str(manifest.contents).unwrap();
    let pinned = parsed["devDependencies"]["@earendil-works/pi-coding-agent"]
        .as_str()
        .expect("the extension types come from a pinned pi release");
    let (major, minor) = DESCRIPTOR
        .verified_through
        .expect("pi declares the release it was verified against");

    assert!(
        pinned.starts_with(&format!("{major}.{minor}.")),
        "the extension compiles against pi {pinned} but the descriptor verifies through \
         {major}.{minor}; move both together"
    );
}

#[test]
fn embedded_sources_speak_the_launcher_and_gateway_contract() {
    let text = embedded_text();
    for name in [
        PI_GATEWAY_URL_ENV,
        PI_OPENAI_UPSTREAM_ENV,
        PI_ANTHROPIC_UPSTREAM_ENV,
        TRANSPARENT_PROXY_CREDENTIAL_ENV,
        TRANSPARENT_PROXY_CREDENTIAL_HEADER,
        UPSTREAM_BASE_URL_HEADER,
    ] {
        assert!(
            text.contains(name),
            "the embedded extension never names {name}"
        );
    }
    for event in DESCRIPTOR.hook_events {
        assert!(
            text.contains(&format!("'{event}'")),
            "the embedded extension never posts {event}, but the descriptor lists it"
        );
    }
}
