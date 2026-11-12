// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Preserve directory traversal before collapsing loader path components.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Component, Path, PathBuf},
};

#[derive(Default)]
pub(super) struct DirectoryLayout {
    pub(super) directories: BTreeSet<PathBuf>,
    pub(super) links: BTreeMap<PathBuf, PathBuf>,
}

impl DirectoryLayout {
    /// Resolve the parent one directory at a time. A symlink followed by `..`
    /// refers to the target's parent, not the symlink's lexical parent.
    pub(super) fn file_path(&mut self, root: &Path, path: &Path) -> Result<PathBuf, String> {
        let parent = path.parent().ok_or("native dependency has no parent")?;
        let components = parent
            .strip_prefix(root)
            .map_err(|_| "native dependency escapes its package")?;
        let mut current = root.to_owned();
        for component in components.components() {
            match component {
                Component::Normal(name) => {
                    let next = current.join(name);
                    let physical = next.canonicalize().map_err(|error| error.to_string())?;
                    if !physical.is_dir() {
                        return Err("native dependency parent must be a directory".into());
                    }
                    let target = physical
                        .strip_prefix(root)
                        .map_err(|_| "native dependency directory escapes its package")?
                        .to_owned();
                    let alias = next.strip_prefix(root).expect("checked parent").to_owned();
                    if alias != target {
                        self.links.insert(alias, target.clone());
                    }
                    if !target.as_os_str().is_empty() {
                        self.directories.insert(target);
                    }
                    current = physical;
                }
                Component::ParentDir => {
                    if current == root || !current.pop() {
                        return Err("native dependency directory escapes its package".into());
                    }
                }
                Component::CurDir => {}
                _ => return Err("invalid native dependency directory".into()),
            }
            if self.directories.len() + self.links.len() > 4096 {
                return Err("native dependency closure exceeds directory budget".into());
            }
        }
        Ok(current.join(
            path.file_name()
                .ok_or("native dependency has no filename")?,
        ))
    }
}
