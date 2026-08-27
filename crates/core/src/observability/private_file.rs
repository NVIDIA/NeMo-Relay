// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt, ambient_authority};
use cap_std::fs::{Dir, DirBuilder, OpenOptions};
#[cfg(unix)]
use cap_std::fs::{DirBuilderExt, OpenOptionsExt};
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Component, Path, PathBuf};

pub(super) fn create_private_dir_all(path: &Path) -> io::Result<()> {
    open_or_create_private_dir(path).map(drop)
}

fn open_or_create_private_dir(path: &Path) -> io::Result<Dir> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut anchor = PathBuf::new();
    let mut current = None;
    for component in absolute.components() {
        match component {
            Component::Prefix(_) | Component::RootDir if current.is_none() => {
                anchor.push(component.as_os_str());
            }
            Component::Normal(name) => {
                let parent = match current.take() {
                    Some(parent) => parent,
                    None if !anchor.as_os_str().is_empty() => {
                        Dir::open_ambient_dir(&anchor, ambient_authority())?
                    }
                    None => {
                        return Err(io::Error::other(format!(
                            "observability directory '{}' has no filesystem anchor",
                            path.display()
                        )));
                    }
                };
                current = Some(open_or_create_private_child(&parent, name)?);
            }
            Component::CurDir => {}
            Component::ParentDir | Component::Prefix(_) | Component::RootDir => {
                return Err(io::Error::other(format!(
                    "observability directory '{}' contains unsafe traversal",
                    path.display()
                )));
            }
        }
    }
    match current {
        Some(directory) => Ok(directory),
        None if !anchor.as_os_str().is_empty() => {
            Dir::open_ambient_dir(&anchor, ambient_authority())
        }
        None => Err(io::Error::other(format!(
            "observability directory '{}' has no filesystem anchor",
            path.display()
        ))),
    }
}

fn open_or_create_private_child(parent: &Dir, name: &std::ffi::OsStr) -> io::Result<Dir> {
    match parent.open_dir_nofollow(name) {
        Ok(directory) => Ok(directory),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            match parent.create_dir_with(name, &private_dir_builder()) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(error),
            }
            parent.open_dir_nofollow(name)
        }
        Err(error) => Err(io::Error::new(
            error.kind(),
            format!(
                "refusing unsafe observability directory component '{}': {error}",
                name.to_string_lossy()
            ),
        )),
    }
}

fn private_dir_builder() -> DirBuilder {
    #[cfg(unix)]
    {
        let mut builder = DirBuilder::new();
        builder.mode(0o700);
        builder
    }
    #[cfg(not(unix))]
    {
        DirBuilder::new()
    }
}

pub(super) fn open_private(root: &Path, path: &Path, append: bool) -> io::Result<File> {
    let (parent, filename) = prepare_confined_parent(root, path)?;
    let mut options = OpenOptions::new();
    options.create(true).write(true).follow(FollowSymlinks::No);
    if append {
        options.append(true);
    } else {
        options.truncate(true);
    }
    #[cfg(unix)]
    {
        options.mode(0o600);
    }
    let file = parent.open_with(&filename, &options)?.into_std();
    restrict_file(&file)?;
    Ok(file)
}

pub(super) fn atomic_private_write(root: &Path, path: &Path, payload: &[u8]) -> io::Result<()> {
    let (parent, filename) = prepare_confined_parent(root, path)?;
    reject_unsafe_target(&parent, &filename)?;
    let filename_text = filename
        .to_str()
        .ok_or_else(|| io::Error::other("observability output filename is not valid text"))?;
    let mut last_collision = None;
    for _ in 0..16 {
        let temporary = format!(".{filename_text}.{}.tmp", uuid::Uuid::now_v7());
        match create_private_new(&parent, &temporary) {
            Ok(mut file) => {
                let result = (|| {
                    file.write_all(payload)?;
                    file.sync_all()?;
                    drop(file);
                    reject_unsafe_target(&parent, &filename)?;
                    parent.rename(&temporary, &parent, &filename)
                })();
                if result.is_err() {
                    let _ = parent.remove_file(&temporary);
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

fn prepare_confined_parent(root: &Path, path: &Path) -> io::Result<(Dir, OsString)> {
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
        current = open_or_create_private_child(&current, component)?;
    }
    Ok((current, filename))
}

fn create_private_new(parent: &Dir, filename: &str) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options
        .create_new(true)
        .write(true)
        .follow(FollowSymlinks::No);
    #[cfg(unix)]
    {
        options.mode(0o600);
    }
    let file = parent.open_with(filename, &options)?.into_std();
    restrict_file(&file)?;
    Ok(file)
}

fn reject_unsafe_target(parent: &Dir, filename: &OsString) -> io::Result<()> {
    match parent.symlink_metadata(filename) {
        Ok(metadata) if metadata.file_type().is_symlink() => Err(io::Error::other(format!(
            "refusing symlinked observability file '{}'",
            filename.to_string_lossy()
        ))),
        Ok(metadata) if !metadata.is_file() => Err(io::Error::other(format!(
            "observability output '{}' is not a regular file",
            filename.to_string_lossy()
        ))),
        Ok(_) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

fn restrict_file(_file: &File) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        _file.set_permissions(fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}
