// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::OsStr;
use std::fs::File;
use std::io;
use std::os::windows::ffi::OsStrExt;
use std::os::windows::io::{AsRawHandle, FromRawHandle};
use std::path::Path;
use windows_sys::Wdk::Foundation::OBJECT_ATTRIBUTES;
use windows_sys::Wdk::Storage::FileSystem::{
    FILE_CREATE, FILE_DIRECTORY_FILE, FILE_NON_DIRECTORY_FILE, FILE_OPEN, FILE_OPEN_IF,
    FILE_OPEN_REPARSE_POINT, FILE_OVERWRITE_IF, FILE_SYNCHRONOUS_IO_NONALERT, NtCreateFile,
};
use windows_sys::Win32::Foundation::{
    HANDLE, INVALID_HANDLE_VALUE, OBJ_CASE_INSENSITIVE, RtlNtStatusToDosError, UNICODE_STRING,
};
use windows_sys::Win32::Storage::FileSystem::{
    BY_HANDLE_FILE_INFORMATION, CreateFileW, DELETE, FILE_APPEND_DATA, FILE_ATTRIBUTE_DIRECTORY,
    FILE_ATTRIBUTE_NORMAL, FILE_ATTRIBUTE_REPARSE_POINT, FILE_DISPOSITION_INFO,
    FILE_FLAG_BACKUP_SEMANTICS, FILE_FLAG_OPEN_REPARSE_POINT, FILE_GENERIC_READ,
    FILE_GENERIC_WRITE, FILE_READ_ATTRIBUTES, FILE_RENAME_INFO, FILE_SHARE_DELETE, FILE_SHARE_READ,
    FILE_SHARE_WRITE, FileDispositionInfo, FileRenameInfo, GetFileInformationByHandle,
    OPEN_EXISTING, SetFileInformationByHandle,
};
use windows_sys::Win32::System::IO::IO_STATUS_BLOCK;

const SHARE_ALL: u32 = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;

pub(in crate::observability) struct ConfinedDir(File);

impl ConfinedDir {
    pub(in crate::observability) fn open_anchor(path: &Path) -> io::Result<Self> {
        let path = wide_null(path.as_os_str());
        // SAFETY: `path` is NUL-terminated, and a successful owned handle is transferred to File.
        let handle = unsafe {
            CreateFileW(
                path.as_ptr(),
                FILE_GENERIC_READ,
                SHARE_ALL,
                std::ptr::null(),
                OPEN_EXISTING,
                FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT,
                std::ptr::null_mut(),
            )
        };
        if handle == INVALID_HANDLE_VALUE {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: `handle` is newly owned and valid.
        let file = unsafe { File::from_raw_handle(handle) };
        validate_handle(&file, true)?;
        Ok(Self(file))
    }

    pub(in crate::observability) fn open_or_create_child(&self, name: &OsStr) -> io::Result<Self> {
        let file = open_relative(
            &self.0,
            name,
            FILE_GENERIC_READ,
            FILE_OPEN_IF,
            FILE_DIRECTORY_FILE,
            FILE_ATTRIBUTE_DIRECTORY,
        )?;
        validate_handle(&file, true)?;
        Ok(Self(file))
    }

    pub(in crate::observability) fn open_private_file(
        &self,
        name: &OsStr,
        append: bool,
    ) -> io::Result<File> {
        let file = open_relative(
            &self.0,
            name,
            if append {
                FILE_APPEND_DATA | FILE_READ_ATTRIBUTES
            } else {
                FILE_GENERIC_WRITE | FILE_READ_ATTRIBUTES
            },
            if append {
                FILE_OPEN_IF
            } else {
                FILE_OVERWRITE_IF
            },
            FILE_NON_DIRECTORY_FILE,
            FILE_ATTRIBUTE_NORMAL,
        )?;
        validate_handle(&file, false)?;
        Ok(file)
    }

    pub(in crate::observability) fn create_private_new(&self, name: &OsStr) -> io::Result<File> {
        let file = open_relative(
            &self.0,
            name,
            FILE_GENERIC_WRITE | FILE_READ_ATTRIBUTES | DELETE,
            FILE_CREATE,
            FILE_NON_DIRECTORY_FILE,
            FILE_ATTRIBUTE_NORMAL,
        )?;
        validate_handle(&file, false)?;
        Ok(file)
    }

    pub(in crate::observability) fn reject_unsafe_target(&self, name: &OsStr) -> io::Result<()> {
        match open_relative(
            &self.0,
            name,
            FILE_READ_ATTRIBUTES,
            FILE_OPEN,
            FILE_NON_DIRECTORY_FILE,
            FILE_ATTRIBUTE_NORMAL,
        ) {
            Ok(file) => validate_handle(&file, false),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error),
        }
    }

    pub(in crate::observability) fn rename_file(
        &self,
        file: &File,
        _source: &OsStr,
        target: &OsStr,
    ) -> io::Result<()> {
        let target = wide(target);
        let header_size = std::mem::size_of::<FILE_RENAME_INFO>() - std::mem::size_of::<u16>();
        let byte_len = header_size + target.len() * std::mem::size_of::<u16>();
        let mut storage = vec![0usize; byte_len.div_ceil(std::mem::size_of::<usize>())];
        let info = storage.as_mut_ptr().cast::<FILE_RENAME_INFO>();
        // SAFETY: `storage` is aligned and sized for the header plus the complete UTF-16 name.
        unsafe {
            (*info).Anonymous.ReplaceIfExists = true;
            (*info).RootDirectory = self.0.as_raw_handle();
            (*info).FileNameLength = (target.len() * std::mem::size_of::<u16>()) as u32;
            std::ptr::copy_nonoverlapping(
                target.as_ptr(),
                (*info).FileName.as_mut_ptr(),
                target.len(),
            );
        }
        // SAFETY: `info` points to the initialized buffer described above for the duration of the call.
        if unsafe {
            SetFileInformationByHandle(
                file.as_raw_handle(),
                FileRenameInfo,
                info.cast(),
                byte_len as u32,
            )
        } == 0
        {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    pub(in crate::observability) fn remove_file(&self, name: &OsStr) -> io::Result<()> {
        let file = open_relative(
            &self.0,
            name,
            DELETE,
            FILE_OPEN,
            FILE_NON_DIRECTORY_FILE,
            FILE_ATTRIBUTE_NORMAL,
        )?;
        let delete = FILE_DISPOSITION_INFO { DeleteFile: true };
        // SAFETY: `delete` has the required layout and remains valid for the call.
        if unsafe {
            SetFileInformationByHandle(
                file.as_raw_handle(),
                FileDispositionInfo,
                (&raw const delete).cast(),
                std::mem::size_of_val(&delete) as u32,
            )
        } == 0
        {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
}

fn open_relative(
    parent: &File,
    name: &OsStr,
    desired_access: u32,
    disposition: u32,
    type_option: u32,
    attributes: u32,
) -> io::Result<File> {
    let mut name = wide(name);
    let byte_len = name
        .len()
        .checked_mul(std::mem::size_of::<u16>())
        .and_then(|length| u16::try_from(length).ok())
        .ok_or_else(|| io::Error::other("observability path component is too long"))?;
    let unicode = UNICODE_STRING {
        Length: byte_len,
        MaximumLength: byte_len,
        Buffer: name.as_mut_ptr(),
    };
    let object = OBJECT_ATTRIBUTES {
        Length: std::mem::size_of::<OBJECT_ATTRIBUTES>() as u32,
        RootDirectory: parent.as_raw_handle(),
        ObjectName: &raw const unicode,
        Attributes: OBJ_CASE_INSENSITIVE,
        SecurityDescriptor: std::ptr::null(),
        SecurityQualityOfService: std::ptr::null(),
    };
    let mut status = IO_STATUS_BLOCK::default();
    let mut handle: HANDLE = std::ptr::null_mut();
    // SAFETY: all pointers reference initialized values for the duration of the synchronous call.
    let result = unsafe {
        NtCreateFile(
            &mut handle,
            desired_access,
            &object,
            &mut status,
            std::ptr::null(),
            attributes,
            SHARE_ALL,
            disposition,
            type_option | FILE_OPEN_REPARSE_POINT | FILE_SYNCHRONOUS_IO_NONALERT,
            std::ptr::null(),
            0,
        )
    };
    if result < 0 {
        // SAFETY: status conversion has no preconditions.
        let code = unsafe { RtlNtStatusToDosError(result) };
        return Err(io::Error::from_raw_os_error(code as i32));
    }
    // SAFETY: a successful NtCreateFile returns a newly owned handle.
    Ok(unsafe { File::from_raw_handle(handle) })
}

fn validate_handle(file: &File, expect_directory: bool) -> io::Result<()> {
    let mut information = BY_HANDLE_FILE_INFORMATION::default();
    // SAFETY: `information` is a writable output buffer with the required layout.
    if unsafe { GetFileInformationByHandle(file.as_raw_handle(), &mut information) } == 0 {
        return Err(io::Error::last_os_error());
    }
    if information.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
        return Err(io::Error::other(
            "refusing reparse-point observability path component",
        ));
    }
    let is_directory = information.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY != 0;
    if is_directory != expect_directory {
        return Err(io::Error::other(if expect_directory {
            "observability path component is not a directory"
        } else {
            "observability output is not a regular file"
        }));
    }
    Ok(())
}

fn wide(value: &OsStr) -> Vec<u16> {
    value.encode_wide().collect()
}

fn wide_null(value: &OsStr) -> Vec<u16> {
    value.encode_wide().chain(Some(0)).collect()
}
