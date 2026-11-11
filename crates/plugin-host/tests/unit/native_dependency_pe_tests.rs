// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

// Small PE data fixtures keep import discovery testable on every host platform.
fn pe_library(is_64: bool, imported_ordinals: Option<bool>) -> Vec<u8> {
    fn word(bytes: &mut [u8], at: usize, value: u16) {
        bytes[at..at + 2].copy_from_slice(&value.to_le_bytes());
    }
    fn dword(bytes: &mut [u8], at: usize, value: u32) {
        bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
    }
    let mut bytes = vec![0; 0x400];
    bytes[..2].copy_from_slice(b"MZ");
    dword(&mut bytes, 0x3c, 0x80);
    bytes[0x80..0x84].copy_from_slice(b"PE\0\0");
    word(&mut bytes, 0x84, if is_64 { 0x8664 } else { 0x14c });
    word(&mut bytes, 0x86, 1);
    let optional_size = if is_64 { 240 } else { 224 };
    word(&mut bytes, 0x94, optional_size);
    word(&mut bytes, 0x96, 0x2022);
    let optional = 0x98;
    word(&mut bytes, optional, if is_64 { 0x20b } else { 0x10b });
    dword(&mut bytes, optional + 32, 0x1000);
    dword(&mut bytes, optional + 36, 0x200);
    dword(&mut bytes, optional + 56, 0x2000);
    dword(&mut bytes, optional + 60, 0x200);
    let directories = optional + if is_64 { 112 } else { 96 };
    dword(&mut bytes, directories - 4, 16);
    let section = optional + usize::from(optional_size);
    bytes[section..section + 8].copy_from_slice(b".rdata\0\0");
    dword(&mut bytes, section + 8, 0x200);
    dword(&mut bytes, section + 12, 0x1000);
    dword(&mut bytes, section + 16, 0x200);
    dword(&mut bytes, section + 20, 0x200);
    dword(&mut bytes, section + 36, 0x40000040);
    if let Some(ordinals) = imported_ordinals {
        dword(&mut bytes, directories + 8, 0x1000);
        dword(&mut bytes, directories + 12, 40);
        dword(&mut bytes, 0x200, 0x1040);
        dword(&mut bytes, 0x20c, 0x1080);
        dword(&mut bytes, 0x210, 0x1060);
        let thunk = if ordinals {
            (if is_64 { 1_u64 << 63 } else { 1_u64 << 31 }) | 1
        } else {
            0x10a0
        };
        for at in [0x240, 0x260] {
            if is_64 {
                bytes[at..at + 8].copy_from_slice(&thunk.to_le_bytes());
            } else {
                dword(&mut bytes, at, thunk as u32);
            }
        }
        bytes[0x280..0x28c].copy_from_slice(b"sidecar.dll\0");
        bytes[0x2a2..0x2ab].copy_from_slice(b"function\0");
    }
    bytes
}

/// Capture imported DLLs whether their symbols are selected by ordinal or name.
#[test]
fn pe_import_descriptors_capture_ordinal_only_and_named_dlls() {
    for is_64 in [false, true] {
        for ordinals in [false, true] {
            let directory = tempfile::tempdir().unwrap();
            let root = directory.path().canonicalize().unwrap();
            let main = pe_library(is_64, Some(ordinals));
            let sidecar = pe_library(is_64, None);
            std::fs::write(root.join("sidecar.dll"), &sidecar).unwrap();
            let actual = capture(&root.join("consumer.dll"), &root, main.clone()).unwrap();
            let expected = NativeLibraryArtifact::new(
                PathBuf::from("consumer.dll"),
                [
                    (PathBuf::from("consumer.dll"), main),
                    (PathBuf::from("sidecar.dll"), sidecar),
                ],
            )
            .unwrap();
            assert_eq!(
                actual.identity(),
                expected.identity(),
                "64-bit={is_64}, ordinal-only={ordinals}"
            );
        }
    }
}

/// Delayed PE imports must be included in the verified package dependency closure.
#[test]
fn pe_delay_imports_capture_package_sidecars() {
    for is_64 in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().canonicalize().unwrap();
        let mut main = pe_library(is_64, Some(false));
        let directories = 0x98 + if is_64 { 112 } else { 96 };
        main[directories + 8..directories + 16].fill(0);
        main[directories + 13 * 8..directories + 13 * 8 + 4]
            .copy_from_slice(&0x1000u32.to_le_bytes());
        main[directories + 13 * 8 + 4..directories + 14 * 8].copy_from_slice(&64u32.to_le_bytes());
        main[0x200..0x240].fill(0);
        for (at, value) in [
            (0x200, 1u32),
            (0x204, 0x1080),
            (0x208, 0x10c0),
            (0x20c, 0x1060),
            (0x210, 0x1040),
        ] {
            main[at..at + 4].copy_from_slice(&value.to_le_bytes());
        }
        let table = object::read::pe::DelayLoadImportTable::new(&main[0x200..], 0x1000, 0x1000);
        let mut descriptors = table.descriptors().unwrap();
        let descriptor = descriptors.next().unwrap().unwrap();
        assert_eq!(
            table
                .name(descriptor.dll_name_rva.get(object::LittleEndian))
                .unwrap(),
            b"sidecar.dll"
        );
        assert!(descriptors.next().unwrap().is_none());
        let sidecar = pe_library(is_64, None);
        std::fs::write(root.join("sidecar.dll"), &sidecar).unwrap();
        let actual = capture(&root.join("consumer.dll"), &root, main.clone()).unwrap();
        let expected = NativeLibraryArtifact::new(
            PathBuf::from("consumer.dll"),
            [
                (PathBuf::from("consumer.dll"), main),
                (PathBuf::from("sidecar.dll"), sidecar),
            ],
        )
        .unwrap();
        assert_eq!(actual.identity(), expected.identity(), "64-bit={is_64}");
    }
}
