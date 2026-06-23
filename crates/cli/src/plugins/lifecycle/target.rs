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
        if looks_like_path(target) {
            Self::Path(PathBuf::from(target))
        } else {
            Self::Id(target.to_owned())
        }
    }
}

fn looks_like_path(target: &str) -> bool {
    let path = Path::new(target);
    path.exists()
        || target.ends_with(".toml")
        || target.contains(std::path::MAIN_SEPARATOR)
        || target.contains('/')
        || target.contains('\\')
}
