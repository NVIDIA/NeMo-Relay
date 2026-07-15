// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

use super::PluginTarget;
use tempfile::tempdir;

#[test]
fn parse_treats_canonical_plugin_ids_as_ids() {
    assert_eq!(
        PluginTarget::parse("acme.worker"),
        PluginTarget::Id("acme.worker".into())
    );
    assert_eq!(
        PluginTarget::parse("acme.worker.v2"),
        PluginTarget::Id("acme.worker.v2".into())
    );
    assert_eq!(
        PluginTarget::parse("relay-plugin"),
        PluginTarget::Id("relay-plugin".into())
    );
}

#[test]
fn parse_treats_manifest_filenames_as_paths() {
    assert_eq!(
        PluginTarget::parse("relay-plugin.toml"),
        PluginTarget::Path(PathBuf::from("relay-plugin.toml"))
    );
}

#[test]
fn parse_treats_relative_path_syntax_as_paths() {
    assert_eq!(
        PluginTarget::parse("./plugins/acme/relay-plugin.toml"),
        PluginTarget::Path(PathBuf::from("./plugins/acme/relay-plugin.toml"))
    );
    assert_eq!(
        PluginTarget::parse("."),
        PluginTarget::Path(PathBuf::from("."))
    );
    assert_eq!(
        PluginTarget::parse(".."),
        PluginTarget::Path(PathBuf::from(".."))
    );
    assert_eq!(
        PluginTarget::parse(r"plugins\acme\relay-plugin.toml"),
        PluginTarget::Path(PathBuf::from(r"plugins\acme\relay-plugin.toml"))
    );
}

#[test]
fn parse_treats_absolute_paths_as_paths_even_when_missing() {
    let temp = tempdir().unwrap();
    let missing = temp.path().join("missing").join("relay-plugin.toml");
    assert_eq!(
        PluginTarget::parse(missing.to_str().unwrap()),
        PluginTarget::Path(missing)
    );
}

#[test]
fn parse_treats_existing_filesystem_entries_with_explicit_path_syntax_as_paths() {
    let temp = tempdir().unwrap();
    let existing = temp.path().join("plugins").join("acme.worker");
    std::fs::create_dir_all(&existing).unwrap();
    assert_eq!(
        PluginTarget::parse(
            existing
                .strip_prefix(temp.path())
                .unwrap()
                .to_str()
                .unwrap()
        ),
        PluginTarget::Path(PathBuf::from("plugins/acme.worker"))
    );
}
