// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Component, Path};

pub(super) fn create_private_dir_all(path: &Path) -> io::Result<()> {
    let mut created = false;
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            return Err(io::Error::other(format!(
                "refusing symlinked observability directory '{}'",
                path.display()
            )));
        }
        Ok(metadata) if !metadata.is_dir() => {
            return Err(io::Error::other(format!(
                "observability output directory '{}' is not a directory",
                path.display()
            )));
        }
        Ok(_) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            fs::create_dir_all(path)?;
            created = true;
        }
        Err(error) => return Err(error),
    }
    if created {
        restrict_directory(path)?;
    }
    Ok(())
}

fn create_private_dir(path: &Path) -> io::Result<()> {
    let mut builder = fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    match builder.create(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
            let metadata = fs::symlink_metadata(path)?;
            if metadata.is_dir() && !metadata.file_type().is_symlink() {
                Ok(())
            } else {
                Err(io::Error::other(format!(
                    "observability directory '{}' changed during creation",
                    path.display()
                )))
            }
        }
        Err(error) => Err(error),
    }
}

pub(super) fn open_private(root: &Path, path: &Path, append: bool) -> io::Result<File> {
    prepare_confined_parent(root, path)?;
    reject_unsafe_target(path)?;
    let mut options = OpenOptions::new();
    options.create(true).write(true);
    if append {
        options.append(true);
    } else {
        options.truncate(true);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let file = options.open(path)?;
    restrict_file(&file)?;
    Ok(file)
}

pub(super) fn atomic_private_write(root: &Path, path: &Path, payload: &[u8]) -> io::Result<()> {
    prepare_confined_parent(root, path)?;
    let parent = path
        .parent()
        .ok_or_else(|| io::Error::other("observability output path has no parent"))?;
    reject_unsafe_target(path)?;
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| io::Error::other("observability output filename is not valid text"))?;
    let mut last_collision = None;
    for _ in 0..16 {
        let temporary = parent.join(format!(".{filename}.{}.tmp", uuid::Uuid::now_v7()));
        match create_private_new(&temporary) {
            Ok(mut file) => {
                let result = (|| {
                    file.write_all(payload)?;
                    file.sync_all()?;
                    drop(file);
                    reject_unsafe_target(path)?;
                    fs::rename(&temporary, path)
                })();
                if result.is_err() {
                    let _ = fs::remove_file(&temporary);
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

fn prepare_confined_parent(root: &Path, path: &Path) -> io::Result<()> {
    create_private_dir_all(root)?;
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
    let relative_parent = relative.parent().unwrap_or_else(|| Path::new(""));
    let mut current = root.to_path_buf();
    for component in relative_parent.components() {
        current.push(component);
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(io::Error::other(format!(
                    "refusing symlinked observability directory '{}'",
                    current.display()
                )));
            }
            Ok(metadata) if !metadata.is_dir() => {
                return Err(io::Error::other(format!(
                    "observability directory component '{}' is not a directory",
                    current.display()
                )));
            }
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                create_private_dir(&current)?;
                restrict_directory(&current)?;
            }
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

fn create_private_new(path: &Path) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.create_new(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let file = options.open(path)?;
    restrict_file(&file)?;
    Ok(file)
}

fn reject_unsafe_target(path: &Path) -> io::Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => Err(io::Error::other(format!(
            "refusing symlinked observability file '{}'",
            path.display()
        ))),
        Ok(metadata) if !metadata.is_file() => Err(io::Error::other(format!(
            "observability output '{}' is not a regular file",
            path.display()
        ))),
        Ok(_) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

fn restrict_file(file: &File) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

fn restrict_directory(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}
