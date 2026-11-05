// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn word(bytes: &mut [u8], at: usize, value: u16) {
    bytes[at..at + 2].copy_from_slice(&value.to_le_bytes());
}
fn dword(bytes: &mut [u8], at: usize, value: u32) {
    bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
}
fn qword(bytes: &mut [u8], at: usize, value: u64) {
    bytes[at..at + 8].copy_from_slice(&value.to_le_bytes());
}

// A minimal ELF image with matching PT_DYNAMIC and SHT_DYNAMIC metadata. Its
// string table has a nonzero virtual base to exercise address-to-file mapping.
fn elf_library(needed: Option<&str>, search: Option<&str>, section_headers: bool) -> Vec<u8> {
    elf_image(
        &needed.into_iter().collect::<Vec<_>>(),
        search.map(|path| (29, path)),
        section_headers,
        if needed.is_some() {
            "consumer.so"
        } else {
            "libdep.so"
        },
    )
}

fn elf_image(
    needed: &[&str],
    search: Option<(u64, &str)>,
    section_headers: bool,
    identity: &str,
) -> Vec<u8> {
    let mut bytes = vec![0; 1024];
    bytes[..7].copy_from_slice(b"\x7fELF\x02\x01\x01");
    word(&mut bytes, 16, 3);
    word(&mut bytes, 18, 62);
    dword(&mut bytes, 20, 1);
    qword(&mut bytes, 32, 64);
    for (at, value) in [(52, 64), (54, 56), (56, 2), (58, 64)] {
        word(&mut bytes, at, value);
    }
    let base = 0x4000;
    let mut strings = vec![0];
    let mut entries = Vec::new();
    for (tag, value) in needed
        .iter()
        .map(|value| (1, *value))
        .chain(search)
        .chain([(14, identity)])
    {
        entries.push((tag, strings.len() as u64));
        strings.extend_from_slice(value.as_bytes());
        strings.push(0);
    }
    entries.extend([(5, base + 256), (10, strings.len() as u64), (0, 0)]);
    bytes[256..256 + strings.len()].copy_from_slice(&strings);
    let dynamic_size = entries.len() as u64 * 16;
    for (i, (tag, value)) in entries.into_iter().enumerate() {
        qword(&mut bytes, 512 + i * 16, tag);
        qword(&mut bytes, 520 + i * 16, value);
    }
    for (at, kind, offset, size) in [(64, 1, 0, 1024), (120, 2, 512, dynamic_size)] {
        dword(&mut bytes, at, kind);
        dword(&mut bytes, at + 4, 6);
        qword(&mut bytes, at + 8, offset);
        qword(&mut bytes, at + 16, base + offset);
        qword(&mut bytes, at + 32, size);
        qword(&mut bytes, at + 40, size);
        qword(&mut bytes, at + 48, 8);
    }
    if section_headers {
        qword(&mut bytes, 40, 768);
        word(&mut bytes, 60, 3);
        word(&mut bytes, 62, 2);
        for (at, kind, offset, size, link, align, entry) in [
            (832, 6, 512, dynamic_size, 2, 8, 16),
            (896, 3, 256, strings.len() as u64, 0, 1, 0),
        ] {
            dword(&mut bytes, at + 4, kind);
            qword(&mut bytes, at + 8, 3);
            qword(&mut bytes, at + 16, base + offset);
            qword(&mut bytes, at + 24, offset);
            qword(&mut bytes, at + 32, size);
            dword(&mut bytes, at + 40, link);
            qword(&mut bytes, at + 48, align);
            qword(&mut bytes, at + 56, entry);
        }
    }
    bytes
}

#[test]
fn shared_elf_dependencies_follow_breadth_first_search_inheritance() {
    for deep in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().canonicalize().unwrap();
        let main = elf_image(
            &["libA.so", "libB.so"],
            Some((15, "$ORIGIN")),
            false,
            "main.so",
        );
        let first = elf_image(
            &[if deep { "libX.so" } else { "libC.so" }],
            Some((15, "$ORIGIN/a")),
            false,
            "libA.so",
        );
        let second = elf_image(&["libC.so"], Some((15, "$ORIGIN/b")), false, "libB.so");
        let shared = elf_image(&["libD.so"], None, false, "libC.so");
        let extra = elf_image(&["libC.so"], None, false, "libX.so");
        let mut leaf = elf_image(&[], None, false, "libD.so");
        let mut alternate = leaf.clone();
        leaf[1023] = 1;
        alternate[1023] = 2;
        std::fs::create_dir(root.join("a")).unwrap();
        std::fs::create_dir(root.join("b")).unwrap();
        let mut files = vec![
            (PathBuf::from("main.so"), main.clone()),
            (PathBuf::from("libA.so"), first),
            (PathBuf::from("libB.so"), second),
            (PathBuf::from("libC.so"), shared),
        ];
        if deep {
            files.push((PathBuf::from("libX.so"), extra));
        }
        for (path, bytes) in &files {
            std::fs::write(root.join(path), bytes).unwrap();
        }
        std::fs::write(root.join("a/libD.so"), &leaf).unwrap();
        std::fs::write(root.join("b/libD.so"), &alternate).unwrap();
        files.push(if deep {
            (PathBuf::from("b").join("libD.so"), alternate)
        } else {
            (PathBuf::from("a").join("libD.so"), leaf)
        });
        let expected = NativeLibraryArtifact::new(PathBuf::from("main.so"), files).unwrap();
        let actual = capture(&root.join("main.so"), &root, main).unwrap();
        assert_eq!(actual.identity(), expected.identity(), "deep={deep}");
    }
}

#[test]
fn elf_program_headers_preserve_imports_search_paths_and_identity_without_sections() {
    for sections in [true, false] {
        let bytes = elf_library(Some("libdep.so"), Some("$ORIGIN/deps:$ORIGIN"), sections);
        let actual = imports(&bytes).unwrap();
        assert_eq!(actual.libraries, ["libdep.so"]);
        assert_eq!(actual.search, ["$ORIGIN/deps", "$ORIGIN"]);
        assert_eq!(actual.identity.as_deref(), Some("consumer.so"));
        assert!(!actual.inherit_search);
    }
}

#[test]
fn elf_origin_imports_capture_sectionless_sidecars_in_both_token_forms() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().canonicalize().unwrap();
    let sidecar = elf_library(None, None, false);
    std::fs::write(root.join("libdep.so"), &sidecar).unwrap();
    for token in ["$ORIGIN", "${ORIGIN}"] {
        let main = elf_library(Some(&format!("{token}/libdep.so")), None, false);
        let actual = capture(&root.join("consumer.so"), &root, main.clone()).unwrap();
        let expected = NativeLibraryArtifact::new(
            PathBuf::from("consumer.so"),
            [
                (PathBuf::from("consumer.so"), main),
                (PathBuf::from("libdep.so"), sidecar.clone()),
            ],
        )
        .unwrap();
        assert_eq!(actual.identity(), expected.identity(), "token={token}");
    }
}

#[test]
fn elf_dynamic_strings_must_be_within_the_file_backed_load_segment() {
    let mut bytes = elf_library(Some("libdep.so"), None, false);
    // DT_STRTAB follows DT_NEEDED and DT_SONAME in this fixture.
    qword(&mut bytes, 552, 0xffff_ffff);
    let error = imports(&bytes).err().unwrap();
    assert!(error.contains("outside loadable segments"), "{error}");
}

#[test]
fn elf_runpath_preserves_ancestor_rpath_for_deeper_dependencies() {
    for direct_dependency_in_runpath in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().canonicalize().unwrap();
        std::fs::create_dir(root.join("ancestor")).unwrap();
        std::fs::create_dir(root.join("child")).unwrap();
        let main = elf_image(
            &["libA.so"],
            Some((15, "$ORIGIN:$ORIGIN/ancestor")),
            false,
            "main.so",
        );
        let intermediate = elf_image(&["libB.so"], Some((29, "$ORIGIN/child")), false, "libA.so");
        let child = elf_image(&["libD.so"], None, false, "libB.so");
        let leaf = elf_image(&[], None, false, "libD.so");
        let mut files = vec![
            (PathBuf::from("main.so"), main.clone()),
            (PathBuf::from("libA.so"), intermediate),
        ];
        for (path, bytes) in &files {
            std::fs::write(root.join(path), bytes).unwrap();
        }
        // A's RUNPATH must override, rather than inherit, main's RPATH for B.
        std::fs::write(root.join("ancestor/libB.so"), &child).unwrap();
        std::fs::write(root.join("ancestor/libD.so"), &leaf).unwrap();
        // A's RUNPATH must not become B's inherited search path for D.
        let mut alternate = leaf.clone();
        alternate[1023] = 1;
        std::fs::write(root.join("child/libD.so"), alternate).unwrap();
        if direct_dependency_in_runpath {
            std::fs::write(root.join("child/libB.so"), &child).unwrap();
            files.push((PathBuf::from("child").join("libB.so"), child));
            files.push((PathBuf::from("ancestor").join("libD.so"), leaf));
        }
        let expected = NativeLibraryArtifact::new(PathBuf::from("main.so"), files).unwrap();
        let actual = capture(&root.join("main.so"), &root, main).unwrap();
        assert_eq!(
            actual.identity(),
            expected.identity(),
            "direct_dependency_in_runpath={direct_dependency_in_runpath}",
        );
    }
}

#[test]
fn elf_search_skips_incompatible_class_and_machine_but_rejects_malformed_files() {
    for mode in ["class", "machine", "malformed"] {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().canonicalize().unwrap();
        for path in ["first", "second"] {
            std::fs::create_dir(root.join(path)).unwrap();
        }
        let main = elf_image(
            &["libdep.so"],
            Some((29, "$ORIGIN/first:$ORIGIN/second")),
            false,
            "main.so",
        );
        let dependency = elf_library(None, None, false);
        let mut incompatible = dependency.clone();
        match mode {
            "class" => incompatible[4] = 1,
            "machine" => word(&mut incompatible, 18, 183),
            _ => incompatible = b"malformed image".to_vec(),
        }
        std::fs::write(root.join("first/libdep.so"), incompatible).unwrap();
        std::fs::write(root.join("second/libdep.so"), &dependency).unwrap();
        let result = capture(&root.join("main.so"), &root, main.clone());
        if mode == "malformed" {
            assert!(result.err().unwrap().contains("cannot read native imports"));
            continue;
        }
        let expected = NativeLibraryArtifact::new(
            PathBuf::from("main.so"),
            [
                (PathBuf::from("main.so"), main.clone()),
                (PathBuf::from("second/libdep.so"), dependency),
            ],
        )
        .unwrap();
        assert_eq!(result.unwrap().identity(), expected.identity(), "{mode}");
        std::fs::remove_file(root.join("second/libdep.so")).unwrap();
        let expected = NativeLibraryArtifact::new(
            PathBuf::from("main.so"),
            [(PathBuf::from("main.so"), main.clone())],
        )
        .unwrap();
        assert_eq!(
            capture(&root.join("main.so"), &root, main)
                .unwrap()
                .identity(),
            expected.identity()
        );
    }
}

#[test]
fn elf_loader_tokens_capture_inherited_paths_without_reexpanding_origin() {
    let tokens = ElfTokens {
        lib: "lib/multiarch".into(),
        platform: Some("cpu".into()),
    };
    for spelling in ["$LIB/$PLATFORM", "${LIB}/${PLATFORM}"] {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("literal-$LIB");
        std::fs::create_dir_all(root.join("lib/multiarch/cpu")).unwrap();
        let root = root.canonicalize().unwrap();
        let main = elf_image(
            &["libchild.so"],
            Some((15, &format!("$ORIGIN:$ORIGIN/{spelling}"))),
            false,
            "main.so",
        );
        let child = elf_image(&["libdep.so"], None, false, "libchild.so");
        let dependency = elf_library(None, None, false);
        std::fs::write(root.join("libchild.so"), &child).unwrap();
        std::fs::write(root.join("lib/multiarch/cpu/libdep.so"), &dependency).unwrap();
        let expected = NativeLibraryArtifact::new(
            PathBuf::from("main.so"),
            [
                (PathBuf::from("main.so"), main.clone()),
                (PathBuf::from("libchild.so"), child),
                (PathBuf::from("lib/multiarch/cpu/libdep.so"), dependency),
            ],
        )
        .unwrap();
        assert_eq!(
            capture_with_tokens(&root.join("main.so"), &root, main, Some(&tokens))
                .unwrap()
                .identity(),
            expected.identity()
        );
    }
}

#[cfg(unix)]
#[test]
fn distinct_symlink_imports_keep_both_loader_paths() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().canonicalize().unwrap();
    let dependency = elf_library(None, None, false);
    std::fs::write(root.join("shared.so"), &dependency).unwrap();
    for name in ["first.so", "second.so"] {
        std::os::unix::fs::symlink("shared.so", root.join(name)).unwrap();
    }
    let main = elf_image(
        &["first.so", "second.so"],
        Some((29, "$ORIGIN")),
        false,
        "main.so",
    );
    let expected = NativeLibraryArtifact::new(
        PathBuf::from("main.so"),
        [
            (PathBuf::from("main.so"), main.clone()),
            (PathBuf::from("first.so"), dependency.clone()),
            (PathBuf::from("second.so"), dependency),
        ],
    )
    .unwrap();
    assert_eq!(
        capture(&root.join("main.so"), &root, main)
            .unwrap()
            .identity(),
        expected.identity()
    );
}

#[test]
fn elf_working_directory_imports_and_search_paths_are_rejected() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().canonicalize().unwrap();
    for import in [
        "deps/libdep.so",
        "./libdep.so",
        "../libdep.so",
        "$LIB/libdep.so",
    ] {
        let main = elf_image(&[import], Some((29, "$ORIGIN")), false, "main.so");
        let error = capture(&root.join("main.so"), &root, main).err().unwrap();
        assert!(error.contains("working directory"), "{import}: {error}");
    }
    for search in ["deps", "$ORIGIN:", ":$ORIGIN", ".:$ORIGIN"] {
        let main = elf_image(&["libdep.so"], Some((29, search)), false, "main.so");
        let error = capture(&root.join("main.so"), &root, main).err().unwrap();
        assert!(error.contains("working directory"), "{search}: {error}");
    }
}

#[test]
fn wholly_empty_elf_search_paths_disable_lookup_without_selecting_cwd() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().canonicalize().unwrap();
    std::fs::write(root.join("libdep.so"), elf_library(None, None, false)).unwrap();
    for tag in [15, 29] {
        let main = elf_image(&["libdep.so"], Some((tag, "")), false, "main.so");
        let parsed = imports(&main).unwrap();
        assert!(parsed.search.is_empty());
        assert_eq!(parsed.inherit_search, tag == 15);
        let expected = NativeLibraryArtifact::new(
            PathBuf::from("main.so"),
            [(PathBuf::from("main.so"), main.clone())],
        )
        .unwrap();
        assert_eq!(
            capture(&root.join("main.so"), &root, main)
                .unwrap()
                .identity(),
            expected.identity()
        );
    }
}
