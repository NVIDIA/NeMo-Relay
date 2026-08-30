// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::confined_fs::ConfinedDir;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Component, Path, PathBuf};

pub(super) fn create_private_dir_all(path: &Path) -> io::Result<()> {
    open_or_create_private_dir(path).map(drop)
}

fn open_or_create_private_dir(path: &Path) -> io::Result<ConfinedDir> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut anchor = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::Prefix(_) if anchor.as_os_str().is_empty() => {
                anchor.push(component.as_os_str());
            }
            // Windows absolute paths are represented as a drive prefix followed by a root
            // component (`C:` then `\\`). Both components make up the trusted filesystem anchor.
            Component::RootDir
                if anchor.as_os_str().is_empty()
                    || matches!(anchor.components().next(), Some(Component::Prefix(_)))
                        && anchor.components().nth(1).is_none() =>
            {
                anchor.push(component.as_os_str());
            }
            Component::Normal(name) => anchor.push(name),
            Component::CurDir => {}
            Component::ParentDir | Component::Prefix(_) | Component::RootDir => {
                return Err(io::Error::other(format!(
                    "observability directory '{}' contains unsafe traversal",
                    path.display()
                )));
            }
        }
    }

    if anchor.as_os_str().is_empty() {
        return Err(io::Error::other(format!(
            "observability directory '{}' has no filesystem anchor",
            path.display()
        )));
    }

    let mut existing = anchor.as_path();
    let mut missing = Vec::new();
    while !fs::metadata(existing).is_ok_and(|metadata| metadata.is_dir()) {
        let name = existing.file_name().ok_or_else(|| {
            io::Error::other(format!(
                "observability directory '{}' has no existing filesystem anchor",
                path.display()
            ))
        })?;
        missing.push(name.to_owned());
        existing = existing.parent().ok_or_else(|| {
            io::Error::other(format!(
                "observability directory '{}' has no existing filesystem anchor",
                path.display()
            ))
        })?;
    }

    // The configured root is trusted and may include platform filesystem aliases such as
    // macOS `/var`. Descendants are still opened relative to this stable directory handle
    // without following symlinks.
    let mut current = ConfinedDir::open_anchor(existing)?;
    for name in missing.into_iter().rev() {
        current = current.open_or_create_child(&name)?;
    }
    Ok(current)
}

#[cfg(all(test, windows))]
mod tests {
    use super::{atomic_private_write, create_private_dir_all};

    #[test]
    fn absolute_temp_directory_is_accepted() {
        let temporary = tempfile::tempdir().expect("temporary directory should be created");
        let output = temporary.path().join("atof");

        create_private_dir_all(&output).expect("absolute Windows output directory should open");
        atomic_private_write(&output, &output.join("trajectory.json"), b"{}")
            .expect("absolute Windows output file should write atomically");

        assert_eq!(
            std::fs::read(output.join("trajectory.json")).unwrap(),
            b"{}"
        );
    }
}

pub(super) fn open_private(root: &Path, path: &Path, append: bool) -> io::Result<File> {
    let (parent, filename) = prepare_confined_parent(root, path)?;
    parent.open_private_file(&filename, append)
}

pub(super) fn atomic_private_write(root: &Path, path: &Path, payload: &[u8]) -> io::Result<()> {
    let (parent, filename) = prepare_confined_parent(root, path)?;
    parent.reject_unsafe_target(&filename)?;
    let filename_text = filename
        .to_str()
        .ok_or_else(|| io::Error::other("observability output filename is not valid text"))?;
    let mut last_collision = None;
    for _ in 0..16 {
        let temporary = format!(".{filename_text}.{}.tmp", uuid::Uuid::now_v7());
        match parent.create_private_new(std::ffi::OsStr::new(&temporary)) {
            Ok(mut file) => {
                let result = (|| {
                    file.write_all(payload)?;
                    file.sync_all()?;
                    parent.reject_unsafe_target(&filename)?;
                    parent.rename_file(&file, std::ffi::OsStr::new(&temporary), &filename)
                })();
                if result.is_err() {
                    let _ = parent.remove_file(std::ffi::OsStr::new(&temporary));
                }
                return result;
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                last_collision = Some(error);
            }
            Err(error) => return Err(error),
        }
    }
    Err(last_collision.unwrap_or_else(|| {
        io::Error::other("failed to allocate a private observability temporary file")
    }))
}

fn prepare_confined_parent(root: &Path, path: &Path) -> io::Result<(ConfinedDir, OsString)> {
    let mut current = open_or_create_private_dir(root)?;
    let relative = path.strip_prefix(root).map_err(|_| {
        io::Error::other(format!(
            "observability output '{}' is outside configured directory '{}'",
            path.display(),
            root.display()
        ))
    })?;
    if relative.as_os_str().is_empty()
        || relative.components().any(|component| {
            matches!(
                component,
                Component::ParentDir
                    | Component::CurDir
                    | Component::RootDir
                    | Component::Prefix(_)
            )
        })
    {
        return Err(io::Error::other(
            "observability output must be a confined relative file path",
        ));
    }
    let filename = relative
        .file_name()
        .ok_or_else(|| io::Error::other("observability output path has no filename"))?
        .to_owned();
    let relative_parent = relative.parent().unwrap_or_else(|| Path::new(""));
    for component in relative_parent.components() {
        let component = component.as_os_str();
        current = current.open_or_create_child(component)?;
    }
    Ok((current, filename))
}
