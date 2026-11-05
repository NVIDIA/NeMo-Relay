// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Mach-O architecture selection before choosing a runtime search candidate.

use object::read::macho::FatArch as _;
use std::{io::Read, path::Path};

pub(super) fn host_architecture(architecture: object::Architecture) -> bool {
    match architecture {
        object::Architecture::Aarch64 => cfg!(target_arch = "aarch64"),
        object::Architecture::X86_64 => cfg!(target_arch = "x86_64"),
        object::Architecture::I386 => cfg!(target_arch = "x86"),
        _ => false,
    }
}

pub(super) fn compatible(bytes: &[u8]) -> Result<bool, String> {
    let read = || -> Result<bool, object::Error> {
        use object::FileKind;
        Ok(match FileKind::parse(bytes)? {
            FileKind::MachOFat32 => object::read::macho::MachOFatFile32::parse(bytes)?
                .arches()
                .iter()
                .any(|arch| host_architecture(arch.architecture())),
            FileKind::MachOFat64 => object::read::macho::MachOFatFile64::parse(bytes)?
                .arches()
                .iter()
                .any(|arch| host_architecture(arch.architecture())),
            FileKind::MachO32 | FileKind::MachO64 => {
                let word = bytes[4..8].try_into().expect("Mach-O header has CPU type");
                let cpu = if matches!(bytes[0], 0xce | 0xcf) {
                    u32::from_le_bytes(word)
                } else {
                    u32::from_be_bytes(word)
                };
                match cpu {
                    object::macho::CPU_TYPE_ARM64 => cfg!(target_arch = "aarch64"),
                    object::macho::CPU_TYPE_X86_64 => cfg!(target_arch = "x86_64"),
                    object::macho::CPU_TYPE_X86 => cfg!(target_arch = "x86"),
                    _ => false,
                }
            }
            _ => false,
        })
    };
    read().map_err(|error| format!("cannot read native Mach-O target: {error}"))
}

pub(super) fn compatible_file(path: &Path) -> Result<bool, String> {
    #[cfg(unix)]
    let mut file = {
        use rustix::fs::{Mode, OFlags, openat};
        std::fs::File::from(
            openat(
                rustix::fs::CWD,
                path,
                OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
                Mode::empty(),
            )
            .map_err(|error| error.to_string())?,
        )
    };
    #[cfg(not(unix))]
    let mut file = std::fs::File::open(path).map_err(|error| error.to_string())?;
    if !file
        .metadata()
        .map_err(|error| error.to_string())?
        .is_file()
    {
        return Err(format!(
            "native dependency '{}' must be a regular file",
            path.display()
        ));
    }
    let mut header = Vec::new();
    (&mut file)
        .take(32)
        .read_to_end(&mut header)
        .map_err(|error| error.to_string())?;
    if matches!(
        object::FileKind::parse(header.as_slice()),
        Ok(object::FileKind::MachOFat32 | object::FileKind::MachOFat64)
    ) {
        let count =
            u32::from_be_bytes(header[4..8].try_into().expect("fat header has count")) as usize;
        if count > 4096 {
            return Err("native universal library exceeds architecture budget".into());
        }
        let size = if header[..4] == object::macho::FAT_MAGIC.to_be_bytes() {
            20
        } else {
            32
        };
        let remaining = (8 + count * size).saturating_sub(header.len());
        file.take(remaining as u64)
            .read_to_end(&mut header)
            .map_err(|error| error.to_string())?;
    }
    compatible(&header)
}
