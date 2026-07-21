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
    if path.is_absolute() {
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
#[path = "../../../tests/coverage/shared/plugins_lifecycle_target_tests.rs"]
mod tests;
