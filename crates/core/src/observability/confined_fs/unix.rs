// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use rustix::fd::OwnedFd;
use rustix::fs::{
    AtFlags, FileType, Mode, OFlags, fchmod, fcntl_getfl, fcntl_setfl, fstat, mkdirat, openat,
    renameat, statat, unlinkat,
};
use std::ffi::OsStr;
use std::fs::File;
use std::io;
use std::path::Path;

pub(in crate::observability) struct ConfinedDir(OwnedFd);

impl ConfinedDir {
    pub(in crate::observability) fn open_anchor(path: &Path) -> io::Result<Self> {
        Ok(Self(openat(
            rustix::fs::CWD,
            path,
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC,
            Mode::empty(),
        )?))
    }

    pub(in crate::observability) fn open_or_create_child(&self, name: &OsStr) -> io::Result<Self> {
        match self.open_child(name) {
            Ok(directory) => Ok(directory),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                match mkdirat(&self.0, name, Mode::RUSR | Mode::WUSR | Mode::XUSR) {
                    Ok(()) => {}
                    Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
                    Err(error) => return Err(error.into()),
                }
                self.open_child(name)
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

    fn open_child(&self, name: &OsStr) -> io::Result<Self> {
        Ok(Self(openat(
            &self.0,
            name,
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC | OFlags::NOFOLLOW,
            Mode::empty(),
        )?))
    }

    pub(in crate::observability) fn open_private_file(
        &self,
        name: &OsStr,
        append: bool,
    ) -> io::Result<File> {
        self.reject_unsafe_target(name)?;
        let mut flags =
            OFlags::WRONLY | OFlags::CREATE | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK;
        flags |= if append {
            OFlags::APPEND
        } else {
            OFlags::TRUNC
        };
        let descriptor = openat(&self.0, name, flags, Mode::RUSR | Mode::WUSR)?;
        let metadata = fstat(&descriptor)?;
        if !FileType::from_raw_mode(metadata.st_mode).is_file() {
            return Err(io::Error::other(format!(
                "observability output '{}' is not a regular file",
                name.to_string_lossy()
            )));
        }
        let mut status_flags = fcntl_getfl(&descriptor)?;
        status_flags.remove(OFlags::NONBLOCK);
        fcntl_setfl(&descriptor, status_flags)?;
        fchmod(&descriptor, Mode::RUSR | Mode::WUSR)?;
        Ok(owned_fd_into_file(descriptor))
    }

    pub(in crate::observability) fn create_private_new(&self, name: &OsStr) -> io::Result<File> {
        let descriptor = openat(
            &self.0,
            name,
            OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::CLOEXEC | OFlags::NOFOLLOW,
            Mode::RUSR | Mode::WUSR,
        )?;
        fchmod(&descriptor, Mode::RUSR | Mode::WUSR)?;
        Ok(owned_fd_into_file(descriptor))
    }

    pub(in crate::observability) fn reject_unsafe_target(&self, name: &OsStr) -> io::Result<()> {
        match statat(&self.0, name, AtFlags::SYMLINK_NOFOLLOW) {
            Ok(metadata) if FileType::from_raw_mode(metadata.st_mode).is_symlink() => {
                Err(io::Error::other(format!(
                    "refusing symlinked observability file '{}'",
                    name.to_string_lossy()
                )))
            }
            Ok(metadata) if !FileType::from_raw_mode(metadata.st_mode).is_file() => {
                Err(io::Error::other(format!(
                    "observability output '{}' is not a regular file",
                    name.to_string_lossy()
                )))
            }
            Ok(_) => Ok(()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error.into()),
        }
    }

    pub(in crate::observability) fn rename_file(
        &self,
        _file: &File,
        source: &OsStr,
        target: &OsStr,
    ) -> io::Result<()> {
        Ok(renameat(&self.0, source, &self.0, target)?)
    }

    pub(in crate::observability) fn remove_file(&self, name: &OsStr) -> io::Result<()> {
        Ok(unlinkat(&self.0, name, AtFlags::empty())?)
    }
}

fn owned_fd_into_file(descriptor: OwnedFd) -> File {
    File::from(descriptor)
}
