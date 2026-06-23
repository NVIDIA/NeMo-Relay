// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum PluginTarget {
    Path(PathBuf),
    Id(String),
}

impl PluginTarget {
    pub(super) fn parse(target: &str) -> Self {
        match classify_target_syntax(target) {
            TargetSyntax::PathLike => Self::Path(PathBuf::from(target)),
            TargetSyntax::PluginId => Self::Id(target.to_owned()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TargetSyntax {
    PathLike,
    PluginId,
}

fn classify_target_syntax(target: &str) -> TargetSyntax {
    if should_treat_target_as_path(target) {
        TargetSyntax::PathLike
    } else {
        TargetSyntax::PluginId
    }
}

// CLI target parsing intentionally uses a conservative "path-like" heuristic rather than trying
// to validate every possible plugin ID. The goal is to treat explicit filesystem syntax as a path
// while keeping ordinary canonical IDs like `acme.worker` on the ID branch.
fn should_treat_target_as_path(target: &str) -> bool {
    let path = Path::new(target);
    if path.exists() || path.is_absolute() {
        return true;
    }

    target == "."
        || target == ".."
        || target.starts_with("./")
        || target.starts_with("../")
        || target.ends_with(".toml")
        || target.contains(std::path::MAIN_SEPARATOR)
        || target.contains('/')
        || target.contains('\\')
}

#[cfg(test)]
mod tests {
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
    fn parse_treats_existing_filesystem_entries_as_paths() {
        let temp = tempdir().unwrap();
        let existing = temp.path().join("acme.worker");
        std::fs::create_dir_all(&existing).unwrap();
        assert_eq!(
            PluginTarget::parse(existing.to_str().unwrap()),
            PluginTarget::Path(existing)
        );
    }
}
