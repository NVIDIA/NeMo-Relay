// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Verify relocated binary dependencies, including PE sidecars and versioned ELF imports.
use super::*;

#[test]
fn native_dependency_capture_ignores_unrelated_files_and_retains_linked_sidecars() {
    assert_dependency_capture(false);
}

#[cfg(not(windows))]
#[test]
fn native_dependency_capture_preserves_package_local_runtime_search_paths() {
    assert_dependency_capture(true);
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
#[test]
fn shared_dependencies_preserve_loader_search_inheritance() {
    let (fixtures, _, _) = dependency_fixture();
    for graph in ["direct", "deep"] {
        let source = tempfile::tempdir().unwrap();
        let root = source.path().canonicalize().unwrap();
        let fixtures = fixtures.join(format!("shared-search-{graph}"));
        for relative in ["", "a", "b"] {
            std::fs::create_dir_all(root.join(relative)).unwrap();
            for entry in std::fs::read_dir(fixtures.join(relative)).unwrap() {
                let entry = entry.unwrap();
                if entry.file_type().unwrap().is_file() {
                    std::fs::copy(entry.path(), root.join(relative).join(entry.file_name()))
                        .unwrap();
                }
            }
        }
        let filename = format!(
            "{}relay_search_main{}",
            std::env::consts::DLL_PREFIX,
            std::env::consts::DLL_SUFFIX
        );
        let main = root.join(filename);
        let artifact = crate::dynamic::native_dependencies::capture(
            &main,
            &root,
            std::fs::read(&main).unwrap(),
        )
        .unwrap();
        let leaf = if graph == "deep" && cfg!(target_os = "linux") {
            "b"
        } else {
            "a"
        };
        assert!(
            artifact.files.keys().any(|path| path.starts_with(leaf)),
            "graph={graph}"
        );
        assert!(
            !artifact
                .files
                .keys()
                .any(|path| path.starts_with(if leaf == "a" { "b" } else { "a" })),
            "graph={graph}"
        );
        let (path, _staging) = artifact.stage().unwrap();
        // Only the captured closure remains available to the OS loader.
        source.close().unwrap();
        let library = load_platform_library(&path).unwrap();
        // SAFETY: this fixture exports the matching zero-argument C function.
        let value = unsafe {
            library
                .get::<unsafe extern "C" fn() -> u32>(b"relay_search_value\0")
                .unwrap()()
        };
        assert_eq!(value, if leaf == "a" { 11 } else { 22 }, "graph={graph}");
    }
}

fn dependency_fixture() -> (PathBuf, String, String) {
    let fixtures = std::env::var_os("NEMO_RELAY_TEST_NATIVE_ABI_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/native-abi-fixtures/debug")
        })
        .join("dependencies");
    let consumer = format!(
        "{}relay_native_consumer{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    );
    let helper = if cfg!(target_os = "linux") {
        "librelay_native_dependency.so.1".to_owned()
    } else {
        format!(
            "{}relay_native_dependency{}",
            std::env::consts::DLL_PREFIX,
            std::env::consts::DLL_SUFFIX
        )
    };
    (fixtures, consumer, helper)
}

fn assert_dependency_capture(nested: bool) {
    let (fixtures, consumer, helper) = dependency_fixture();
    let source = tempfile::tempdir().unwrap();
    let package = source.path().canonicalize().unwrap();
    let binaries = package.join("runtime");
    std::fs::create_dir(&binaries).unwrap();
    let dependencies = if nested {
        binaries.join("deps")
    } else {
        binaries.clone()
    };
    std::fs::create_dir_all(&dependencies).unwrap();
    std::fs::copy(fixtures.join(&helper), dependencies.join(&helper))
        .expect("prepare the native ABI fixtures");
    if nested {
        // The binary's first runtime search path takes priority over this unrelated file.
        std::fs::write(binaries.join(&helper), b"unrelated native build output").unwrap();
    }
    let bytes = std::fs::read(fixtures.join(&consumer)).expect("prepare the native ABI fixtures");
    // Neither a large resource nor build caches belong to the binary import closure.
    std::fs::File::create(package.join("unrelated-data.txt"))
        .unwrap()
        .set_len(512 * 1024 * 1024 + 1)
        .unwrap();
    let artifact =
        crate::dynamic::native_dependencies::capture(&binaries.join(&consumer), &package, bytes)
            .unwrap();
    assert_eq!(artifact.files.len(), 2);
    assert!(
        artifact
            .files
            .contains_key(&dependencies.strip_prefix(&package).unwrap().join(&helper))
    );
    std::fs::remove_file(dependencies.join(&helper)).unwrap();
    let (path, _staging) = artifact.stage().unwrap();
    let library = load_platform_library(&path).unwrap();
    // SAFETY: this fixture exports the matching zero-argument C function.
    let value = unsafe {
        library
            .get::<unsafe extern "C" fn() -> u32>(b"relay_consumer_value\0")
            .unwrap()()
    };
    assert_eq!(value, 8);
}

#[test]
fn resident_dependencies_reject_conflicting_bytes_before_loading() {
    let (fixtures, consumer, helper) = dependency_fixture();
    let main = std::fs::read(fixtures.join(&consumer)).unwrap();
    let dependency = std::fs::read(fixtures.join(&helper)).unwrap();
    let alternate = std::fs::read(fixtures.join("alternate").join(&helper)).unwrap();
    assert_ne!(dependency, alternate);
    let artifact = |sidecar| {
        NativeLibraryArtifact::new(
            PathBuf::from(&consumer),
            [
                (PathBuf::from(&consumer), main.clone()),
                (PathBuf::from(&helper), sidecar),
            ],
        )
        .unwrap()
    };
    let first = artifact(dependency.clone());
    let identity = first.identity();
    // No bootstrap is exported, but the library and sidecar are already resident.
    let error = CheckedNativeLibrary::load(first, "v1.0", "dependency-test")
        .err()
        .unwrap();
    assert!(error.contains("bootstrap"), "{error}");
    assert!(resident_library_path(identity).is_some());
    #[cfg(not(windows))]
    {
        let renamed = NativeLibraryArtifact::new(
            PathBuf::from(&consumer),
            [
                (PathBuf::from(&consumer), main.clone()),
                (
                    PathBuf::from(format!("renamed-{helper}")),
                    alternate.clone(),
                ),
            ],
        )
        .unwrap();
        let error = CheckedNativeLibrary::load(renamed, "v1.0", "dependency-test")
            .err()
            .unwrap();
        assert!(
            error.contains("conflicts with a resident library"),
            "the embedded library identity must also be checked: {error}"
        );
    }
    let equivalent = NativeLibraryArtifact::new(
        PathBuf::from(&consumer),
        [
            (PathBuf::from(&consumer), main.clone()),
            (PathBuf::from(&helper), dependency.clone()),
            (
                PathBuf::from("closure-marker"),
                b"different artifact, identical sidecar".to_vec(),
            ),
        ],
    )
    .unwrap();
    let error = CheckedNativeLibrary::load(equivalent, "v1.0", "dependency-test")
        .err()
        .unwrap();
    assert!(
        error.contains("bootstrap"),
        "identical sidecars can be shared across artifacts: {error}"
    );
    let conflicting = artifact(alternate);
    let conflicting_identity = conflicting.identity();
    let error = CheckedNativeLibrary::load(conflicting, "v1.0", "dependency-test")
        .err()
        .unwrap();
    assert!(
        error.contains("conflicts with a resident library"),
        "{error}"
    );
    assert!(resident_library_path(conflicting_identity).is_none());
    let error = CheckedNativeLibrary::load(artifact(dependency), "v1.0", "dependency-test")
        .err()
        .unwrap();
    assert!(
        error.contains("bootstrap"),
        "identical dependencies must remain usable: {error}"
    );
}

#[cfg(not(windows))]
#[test]
fn package_absolute_imports_and_search_paths_fail_before_loading() {
    let (fixtures, consumer, _) = dependency_fixture();
    let root = fixtures.canonicalize().unwrap();
    for path in [
        root.join("absolute").join(&consumer),
        root.join(format!("absolute-search-{consumer}")),
    ] {
        let error = crate::dynamic::native_dependencies::capture(
            &path,
            &root,
            std::fs::read(&path).unwrap(),
        )
        .err()
        .expect("absolute package dependency must be rejected");
        assert!(error.contains("absolute within the package"), "{error}");
    }
}

#[test]
fn resident_main_image_cannot_override_a_captured_sidecar() {
    let (directory, consumer, helper) = dependency_fixture();
    let directory = directory.join("main-image");
    let helper = helper.replace("relay_native_dependency", "relay_resident_image");
    let consumer = consumer.replace("relay_native_consumer", "relay_image_consumer");
    let main = NativeLibraryArtifact::new(
        PathBuf::from(&helper),
        [(
            PathBuf::from(&helper),
            std::fs::read(directory.join(&helper)).unwrap(),
        )],
    )
    .unwrap();
    let identity = main.identity();
    // Rejected bootstrap checks retain the image, including its loader aliases.
    let error = CheckedNativeLibrary::load(main, "v1.0", "resident-main")
        .err()
        .unwrap();
    assert!(error.contains("bootstrap"), "{error}");
    assert!(resident_library_path(identity).is_some());
    let next = NativeLibraryArtifact::new(
        PathBuf::from(&consumer),
        [
            (
                PathBuf::from(&consumer),
                std::fs::read(directory.join(&consumer)).unwrap(),
            ),
            (
                PathBuf::from(&helper),
                std::fs::read(directory.join("alternate").join(&helper)).unwrap(),
            ),
        ],
    )
    .unwrap();
    let identity = next.identity();
    let error = CheckedNativeLibrary::load(next, "v1.0", "resident-main")
        .err()
        .unwrap();
    assert!(
        error.contains("conflicts with a resident library"),
        "{error}"
    );
    assert!(resident_library_path(identity).is_none());
}

#[cfg(windows)]
#[test]
fn delay_loaded_sidecar_resolves_from_resident_verified_staging() {
    let (fixtures, _, _) = dependency_fixture();
    let fixtures = fixtures.join("delay-loaded");
    let source = tempfile::tempdir().unwrap();
    let package = source.path().canonicalize().unwrap();
    let consumer = "relay_delayed_consumer.dll";
    let helper = "relay_delayed_dependency.dll";
    std::fs::copy(fixtures.join(helper), package.join(helper)).unwrap();
    let artifact = crate::dynamic::native_dependencies::capture(
        &package.join(consumer),
        &package,
        std::fs::read(fixtures.join(consumer)).unwrap(),
    )
    .unwrap();
    assert_eq!(artifact.files.len(), 2);
    let identity = artifact.identity();
    std::fs::remove_file(package.join(helper)).unwrap();
    let error = CheckedNativeLibrary::load(artifact, "v1.0", "delayed-dependency-test")
        .err()
        .unwrap();
    assert!(error.contains("bootstrap"), "{error}");
    let resident = *cache().lock().unwrap().get(&identity).unwrap();
    // SAFETY: the fixture exports this C function, which calls its delayed DLL.
    let value = unsafe {
        resident
            .raw
            .get::<unsafe extern "C" fn() -> u32>(b"relay_consumer_value\0")
            .unwrap()()
    };
    assert_eq!(value, 8);
}

#[test]
fn unchecked_bootstrap_and_factory_headers_fail_before_construction() {
    use nemo_relay_plugin::abi::abi_stable::{
        library::{CheckTypeLayout, RootModuleResult},
        prefix_type::PrefixTypeTrait,
        sabi_types::RRef,
        std_types::{ROption, RStr},
    };
    use std::sync::atomic::{AtomicUsize, Ordering};
    static CONSTRUCTIONS: AtomicUsize = AtomicUsize::new(0);
    extern "C" fn descriptor() -> PluginDescriptor {
        PluginDescriptor {
            native_api: NATIVE_API_VERSION,
            plugin_kind: "unchecked".into(),
        }
    }
    extern "C" fn query(_: RStr<'_>) -> ROption<RRef<'static, LibHeader>> {
        ROption::RNone
    }
    extern "C" fn create() -> v1_0::Result<PluginBox> {
        v1_0::Result::RErr(v1_0::Error::new("test", "factory must not run"))
    }
    extern "C" fn bootstrap() -> RootModuleResult {
        CONSTRUCTIONS.fetch_add(1, Ordering::SeqCst);
        nemo_relay_plugin::abi::abi_stable::library::__call_root_module_loader(|| {
            nemo_relay_plugin::abi::Bootstrap {
                descriptor,
                query_module: query,
            }
            .leak_into_prefix()
        })
    }
    extern "C" fn factory() -> RootModuleResult {
        CONSTRUCTIONS.fetch_add(1, Ordering::SeqCst);
        nemo_relay_plugin::abi::abi_stable::library::__call_root_module_loader(|| {
            v1_0::Factory { create }.leak_into_prefix()
        })
    }
    // SAFETY: both constructors return exactly their declared module type.
    let bootstrap =
        unsafe { LibHeader::from_constructor::<Bootstrap_Ref>(bootstrap, CheckTypeLayout::No) };
    // SAFETY: as above; the missing layout metadata is the rejected condition.
    let factory =
        unsafe { LibHeader::from_constructor::<v1_0::Factory_Ref>(factory, CheckTypeLayout::No) };
    let bootstrap_error = checked_module::<Bootstrap_Ref>(&bootstrap).err().unwrap();
    let factory_error = checked_module::<v1_0::Factory_Ref>(&factory).err().unwrap();
    for error in [bootstrap_error, factory_error] {
        assert!(error.contains("type-layout metadata"), "{error}");
    }
    assert_eq!(CONSTRUCTIONS.load(Ordering::SeqCst), 0);
}

#[cfg(target_os = "linux")]
#[test]
fn sectionless_shared_objects_load_sidecars_with_search_paths_and_braced_origins() {
    fn without_sections(mut bytes: Vec<u8>) -> Vec<u8> {
        assert_eq!(&bytes[..4], b"\x7fELF");
        match bytes[4] {
            1 => {
                bytes[32..36].fill(0);
                bytes[48..52].fill(0);
            }
            2 => {
                bytes[40..48].fill(0);
                bytes[60..64].fill(0);
            }
            _ => panic!("unknown ELF class"),
        }
        bytes
    }
    let (fixtures, consumer, helper) = dependency_fixture();
    for fixtures in [fixtures.clone(), fixtures.join("braced-origin")] {
        let source = tempfile::tempdir().unwrap();
        let package = source.path().canonicalize().unwrap();
        let main = without_sections(std::fs::read(fixtures.join(&consumer)).unwrap());
        let dependency = without_sections(std::fs::read(fixtures.join(&helper)).unwrap());
        std::fs::write(package.join(&helper), &dependency).unwrap();
        let artifact =
            crate::dynamic::native_dependencies::capture(&package.join(&consumer), &package, main)
                .unwrap();
        assert_eq!(artifact.files.len(), 2);
        std::fs::remove_file(package.join(&helper)).unwrap();
        let (path, _staging) = artifact.stage().unwrap();
        let library = load_platform_library(&path).unwrap();
        // SAFETY: the fixture exports the matching zero-argument C function.
        let value = unsafe {
            library
                .get::<unsafe extern "C" fn() -> u32>(b"relay_consumer_value\0")
                .unwrap()()
        };
        assert_eq!(value, 8);
    }
}

#[test]
fn native_aliases_preserve_identity_and_stage_all_import_spellings() {
    let main = PathBuf::from("main.bin");
    let original = PathBuf::from("libshared.bin");
    let alias = PathBuf::from("LIBSHARED.bin");
    let files = [(main.clone(), vec![1]), (original.clone(), vec![2])];
    let artifact = NativeLibraryArtifact::new_with_aliases(
        main.clone(),
        files.clone(),
        [(alias.clone(), original.clone())],
    )
    .unwrap();
    let expected = NativeLibraryArtifact::new(
        main.clone(),
        files.clone().into_iter().chain([(alias.clone(), vec![2])]),
    )
    .unwrap();
    assert_eq!(artifact.identity(), expected.identity());
    let (_, staging) = artifact.stage().unwrap();
    assert_eq!(std::fs::read(staging.path().join(&alias)).unwrap(), vec![2]);
    assert_eq!(
        std::fs::read(staging.path().join(&original)).unwrap(),
        vec![2]
    );
    assert!(
        NativeLibraryArtifact::new_with_aliases(
            main.clone(),
            files.clone(),
            [(alias.clone(), PathBuf::from("missing"))]
        )
        .is_err()
    );
    assert!(
        NativeLibraryArtifact::new_with_aliases(
            main.clone(),
            files.clone(),
            [(alias.clone(), PathBuf::from("../escape"))]
        )
        .is_err()
    );
    assert!(NativeLibraryArtifact::new_with_aliases(main, files, [(original, alias)]).is_err());
}

#[cfg(target_os = "macos")]
#[test]
fn case_equivalent_macho_imports_load_after_source_removal() {
    let (fixtures, _, _) = dependency_fixture();
    for symbolic in [false, true] {
        let source = tempfile::tempdir().unwrap();
        let root = source.path().canonicalize().unwrap();
        for entry in std::fs::read_dir(fixtures.join("case-aliases")).unwrap() {
            let entry = entry.unwrap();
            std::fs::copy(entry.path(), root.join(entry.file_name())).unwrap();
        }
        let lower = root.join("librelay_search_c.dylib");
        let upper = root.join("LIBRELAY_SEARCH_C.dylib");
        if symbolic {
            std::fs::create_dir(root.join("actual")).unwrap();
            std::fs::rename(&lower, root.join("actual/librelay_search_c.dylib")).unwrap();
            std::os::unix::fs::symlink("actual/librelay_search_c.dylib", &lower).unwrap();
        }
        // Case-sensitive source volumes need both directory entries as well.
        if !upper.exists() {
            std::fs::hard_link(&lower, &upper).unwrap();
        }
        let main = root.join("librelay_search_main.dylib");
        let artifact = crate::dynamic::native_dependencies::capture(
            &main,
            &root,
            std::fs::read(&main).unwrap(),
        )
        .unwrap();
        let (path, _staging) = artifact.stage().unwrap();
        source.close().unwrap();
        let library = load_platform_library(&path).unwrap();
        // SAFETY: this fixture exports the matching zero-argument C function.
        let value = unsafe {
            library
                .get::<unsafe extern "C" fn() -> u32>(b"relay_search_value\0")
                .unwrap()()
        };
        assert_eq!(value, 11);
    }
}

#[cfg(target_os = "macos")]
fn macho_loader_source() -> tempfile::TempDir {
    let (fixtures, _, _) = dependency_fixture();
    let source = tempfile::tempdir().unwrap();
    let fixtures = fixtures.join("loader-paths");
    for directory in ["", "actual", "first", "second"] {
        std::fs::create_dir_all(source.path().join(directory)).unwrap();
        for entry in std::fs::read_dir(fixtures.join(directory)).unwrap() {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_file() {
                std::fs::copy(
                    entry.path(),
                    source.path().join(directory).join(entry.file_name()),
                )
                .unwrap();
            }
        }
    }
    source
}

#[cfg(target_os = "macos")]
fn assert_macho_loader_value(path: &Path) {
    let library = load_platform_library(path).unwrap();
    // SAFETY: the fixture exports this zero-argument C function.
    let value = unsafe {
        library
            .get::<unsafe extern "C" fn() -> i32>(b"relay_path_value\0")
            .unwrap()()
    };
    assert_eq!(value, 42);
}

#[cfg(target_os = "macos")]
#[test]
fn macho_search_skips_thin_and_universal_candidates_without_host_architecture() {
    for fat in [false, true] {
        let source = macho_loader_source();
        let root = source.path().canonicalize().unwrap();
        if fat {
            std::fs::copy(
                root.join("first/fat-child.dylib"),
                root.join("first/librelay_path_child.dylib"),
            )
            .unwrap();
        }
        let main = root.join("librelay_path_arch.dylib");
        let artifact = crate::dynamic::native_dependencies::capture(
            &main,
            &root,
            std::fs::read(&main).unwrap(),
        )
        .unwrap();
        assert!(
            artifact
                .files
                .contains_key(Path::new("second/librelay_path_child.dylib"))
        );
        assert!(
            artifact
                .files
                .contains_key(Path::new("second/librelay_path_leaf.dylib"))
        );
        assert!(!artifact.files.keys().any(|path| path.starts_with("first")));
        let (path, _staging) = artifact.stage().unwrap();
        source.close().unwrap();
        assert_macho_loader_value(&path);
    }
}

#[cfg(target_os = "macos")]
#[test]
fn macho_file_directory_and_main_symlinks_keep_physical_loader_origins() {
    use std::os::unix::fs::symlink;
    for mode in ["file", "directory", "main"] {
        let source = macho_loader_source();
        let root = source.path().canonicalize().unwrap();
        let main = root.join("librelay_path_main.dylib");
        if mode == "file" {
            std::fs::create_dir(root.join("alias")).unwrap();
            symlink(
                "../actual/librelay_path_child.dylib",
                root.join("alias/librelay_path_child.dylib"),
            )
            .unwrap();
        } else if mode == "directory" {
            symlink("actual", root.join("alias")).unwrap();
        } else {
            std::fs::rename(&main, root.join("actual/librelay_path_main.dylib")).unwrap();
            symlink(".", root.join("actual/alias")).unwrap();
            symlink("actual/librelay_path_main.dylib", &main).unwrap();
        }
        let artifact = crate::dynamic::native_dependencies::capture(
            &main,
            &root,
            std::fs::read(&main).unwrap(),
        )
        .unwrap();
        assert!(
            artifact
                .files
                .contains_key(Path::new("actual/librelay_path_child.dylib")),
            "{mode}"
        );
        assert!(
            artifact
                .files
                .contains_key(Path::new("actual/librelay_path_leaf.dylib")),
            "{mode}"
        );
        let (path, _staging) = artifact.stage().unwrap();
        source.close().unwrap();
        assert_macho_loader_value(&path);
    }
}

#[cfg(target_os = "macos")]
#[test]
fn macho_working_directory_imports_fail_before_staging() {
    let source = macho_loader_source();
    let root = source.path().canonicalize().unwrap();
    for name in [
        "librelay_path_relative.dylib",
        "librelay_path_bare.dylib",
        "librelay_path_relative_search.dylib",
    ] {
        let main = root.join(name);
        let error = crate::dynamic::native_dependencies::capture(
            &main,
            &root,
            std::fs::read(&main).unwrap(),
        )
        .err()
        .unwrap();
        assert!(error.contains("working directory"), "{error}");
    }
}

#[cfg(unix)]
#[test]
fn symbolic_artifact_links_are_relative_validated_and_part_of_identity() {
    let main = PathBuf::from("alias/main.bin");
    let target = PathBuf::from("actual/main.bin");
    let files = [(target.clone(), vec![1, 2])];
    let links = [(main.clone(), target.clone())];
    let linked =
        NativeLibraryArtifact::new_with_links(main.clone(), files.clone(), [], links.clone())
            .unwrap();
    let copied = NativeLibraryArtifact::new(
        main.clone(),
        files
            .clone()
            .into_iter()
            .chain([(main.clone(), vec![1, 2])]),
    )
    .unwrap();
    assert_ne!(linked.identity(), copied.identity());
    let (path, staging) = linked.stage().unwrap();
    assert_eq!(
        std::fs::read_link(&path).unwrap(),
        Path::new("../actual/main.bin")
    );
    assert_eq!(
        path.canonicalize().unwrap(),
        staging.path().join(&target).canonicalize().unwrap()
    );
    assert_eq!(std::fs::read(path).unwrap(), [1, 2]);
    for invalid in [PathBuf::from("../escape"), PathBuf::from("missing")] {
        assert!(
            NativeLibraryArtifact::new_with_links(
                main.clone(),
                files.clone(),
                [],
                [(main.clone(), invalid)]
            )
            .is_err()
        );
    }
    assert!(
        NativeLibraryArtifact::new_with_links(
            main.clone(),
            files.clone(),
            [],
            [(target.clone(), target.clone())]
        )
        .is_err()
    );
    assert!(NativeLibraryArtifact::new_with_links(main, files, links.clone(), links).is_err());
}

#[cfg(target_os = "macos")]
#[test]
fn macho_parent_components_follow_directories_and_symlinks_before_staging() {
    use std::os::unix::fs::symlink;
    for symbolic in [false, true] {
        for filename in [
            "librelay_path_parent.dylib",
            "librelay_path_parent_search.dylib",
        ] {
            let source = macho_loader_source();
            let root = source.path().canonicalize().unwrap();
            if symbolic {
                std::fs::create_dir(root.join("actual/subdir")).unwrap();
                symlink("actual/subdir", root.join("alias")).unwrap();
            } else {
                std::fs::create_dir(root.join("alias")).unwrap();
                for name in ["librelay_path_child.dylib", "librelay_path_leaf.dylib"] {
                    std::fs::copy(root.join("actual").join(name), root.join(name)).unwrap();
                }
            }
            let main = root.join(filename);
            let artifact = crate::dynamic::native_dependencies::capture(
                &main,
                &root,
                std::fs::read(&main).unwrap(),
            )
            .unwrap();
            if symbolic {
                assert!(
                    artifact
                        .files
                        .contains_key(Path::new("actual/librelay_path_child.dylib"))
                );
                assert!(
                    !artifact
                        .files
                        .contains_key(Path::new("librelay_path_child.dylib")),
                    "must not capture the lexical decoy"
                );
            }
            let (path, _staging) = artifact.stage().unwrap();
            source.close().unwrap();
            assert_macho_loader_value(&path);
        }
    }
}

#[cfg(target_os = "macos")]
#[test]
fn macho_parent_component_cannot_hide_a_directory_escape() {
    use std::os::unix::fs::symlink;
    let source = macho_loader_source();
    let root = source.path().canonicalize().unwrap();
    let outside = tempfile::tempdir().unwrap();
    std::fs::create_dir(outside.path().join("subdir")).unwrap();
    std::fs::copy(
        root.join("actual/librelay_path_child.dylib"),
        outside.path().join("librelay_path_child.dylib"),
    )
    .unwrap();
    symlink(outside.path().join("subdir"), root.join("alias")).unwrap();
    let main = root.join("librelay_path_parent.dylib");
    let error =
        crate::dynamic::native_dependencies::capture(&main, &root, std::fs::read(&main).unwrap())
            .err()
            .unwrap();
    assert!(error.contains("escapes its package"), "{error}");
}

#[cfg(unix)]
#[test]
fn directory_layout_is_validated_and_part_of_artifact_identity() {
    let files = [
        (PathBuf::from("main"), vec![1]),
        (PathBuf::from("actual/value"), vec![2]),
    ];
    let linked = NativeLibraryArtifact::new_with_layout(
        PathBuf::from("main"),
        files.clone(),
        [],
        [],
        [PathBuf::from("actual/subdir")],
        [(PathBuf::from("alias"), PathBuf::from("actual/subdir"))],
    )
    .unwrap();
    let plain = NativeLibraryArtifact::new(PathBuf::from("main"), files.clone()).unwrap();
    assert_ne!(linked.identity(), plain.identity());
    let (_, staging) = linked.stage().unwrap();
    assert_eq!(
        std::fs::read(staging.path().join("alias/../value")).unwrap(),
        [2]
    );
    let root_link = NativeLibraryArtifact::new_with_layout(
        PathBuf::from("main"),
        files.clone(),
        [],
        [],
        [],
        [(PathBuf::from("alias"), PathBuf::new())],
    )
    .unwrap();
    let (_, staging) = root_link.stage().unwrap();
    assert_eq!(
        std::fs::read(staging.path().join("alias/main")).unwrap(),
        [1]
    );
    for (alias, target) in [
        ("../escape", "actual/subdir"),
        ("alias", "../escape"),
        ("alias", "missing"),
        ("actual/subdir", "actual/subdir"),
        ("main", "actual/subdir"),
    ] {
        assert!(
            NativeLibraryArtifact::new_with_layout(
                PathBuf::from("main"),
                files.clone(),
                [],
                [],
                [PathBuf::from("actual/subdir")],
                [(PathBuf::from(alias), PathBuf::from(target))]
            )
            .is_err()
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn path_qualified_sidecars_with_equal_basenames_load_together() {
    let (fixtures, _, _) = dependency_fixture();
    let source = tempfile::tempdir().unwrap();
    let root = source.path().canonicalize().unwrap();
    for relative in [
        "a/librelay_distinct.dylib",
        "b/librelay_distinct.dylib",
        "librelay_distinct_main.dylib",
    ] {
        let destination = root.join(relative);
        std::fs::create_dir_all(destination.parent().unwrap()).unwrap();
        std::fs::copy(
            fixtures.join("distinct-identities").join(relative),
            destination,
        )
        .unwrap();
    }
    let main = root.join("librelay_distinct_main.dylib");
    let artifact =
        crate::dynamic::native_dependencies::capture(&main, &root, std::fs::read(&main).unwrap())
            .unwrap();
    assert_eq!(artifact.files.len(), 3);
    let identities = artifact.image_identities().unwrap();
    assert!(
        identities
            .dependencies
            .contains_key("@rpath/a/librelay_distinct.dylib")
    );
    assert!(
        identities
            .dependencies
            .contains_key("@rpath/b/librelay_distinct.dylib")
    );
    assert!(!identities.images.contains_key("librelay_distinct.dylib"));
    let identity = artifact.identity();
    let error = CheckedNativeLibrary::load(artifact, "v1.0", "distinct-test")
        .err()
        .unwrap();
    assert!(error.contains("bootstrap"), "{error}");
    source.close().unwrap();
    assert_macho_loader_value(&resident_library_path(identity).unwrap());
}

/// Unnamed Unix path imports coexist, while bare imports still reject conflicting bytes.
#[cfg(unix)]
#[test]
fn unnamed_unix_images_use_staged_paths_without_claiming_shared_basenames() {
    // A minimal ELF image without program headers has no DT_SONAME.
    let mut first = vec![0; 128];
    first[..7].copy_from_slice(b"\x7fELF\x02\x01\x01");
    for (at, word) in [(16, 3_u16), (18, 62), (52, 64), (54, 56), (58, 64)] {
        first[at..at + 2].copy_from_slice(&word.to_le_bytes());
    }
    first[20..24].copy_from_slice(&1_u32.to_le_bytes());
    let mut second = first.clone();
    first[127] = 11;
    second[127] = 22;
    let mut artifact = NativeLibraryArtifact::new(
        PathBuf::from("main"),
        [
            (PathBuf::from("main"), vec![]),
            (PathBuf::from("a/libsame.so"), first),
            (PathBuf::from("b/libsame.so"), second),
        ],
    )
    .unwrap();
    artifact
        .path_qualified_imports
        .extend([PathBuf::from("a/libsame.so"), PathBuf::from("b/libsame.so")]);
    assert!(artifact.image_identities().unwrap().images.is_empty());
    artifact
        .path_qualified_imports
        .remove(Path::new("b/libsame.so"));
    assert!(
        artifact
            .image_identities()
            .unwrap()
            .images
            .contains_key("libsame.so")
    );
    artifact.path_qualified_imports.clear();
    let error = artifact.image_identities().err().unwrap();
    assert!(
        error.contains("conflicting native image 'libsame.so'"),
        "{error}"
    );
}
