// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Checked native loading and process residency for verified artifacts.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Component, Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use nemo_relay_plugin::abi::{
    Bootstrap_Ref, NATIVE_API_VERSION, NativeApiVersion, PluginDescriptor,
    abi_stable::library::{AbiHeaderRef, LibHeader, ROOT_MODULE_LOADER_NAME_WITH_NUL, RootModule},
    v1_0::{self, PluginBox},
    validate_descriptor,
};
use sha2::{Digest, Sha256};

/// Exact artifact bytes and dependency closure supplied after host trust checks.
///
/// Construction snapshots bytes only; it does not establish trust. Callers must
/// enforce Relay compatibility, signatures, integrity, and policy before load.
/// Paths describe the protected staging tree, never source filesystem paths.
pub struct NativeLibraryArtifact {
    library: PathBuf,
    files: BTreeMap<PathBuf, Vec<u8>>,
    aliases: BTreeMap<PathBuf, PathBuf>,
    symlinks: BTreeMap<PathBuf, PathBuf>,
    directories: BTreeSet<PathBuf>,
    directory_links: BTreeMap<PathBuf, PathBuf>,
    identity: [u8; 32],
}

impl NativeLibraryArtifact {
    /// Capture a verified main library and every file in its dependency closure.
    /// Duplicate or escaping paths fail before any library is opened.
    pub fn new(
        library: PathBuf,
        files: impl IntoIterator<Item = (PathBuf, Vec<u8>)>,
    ) -> Result<Self, String> {
        Self::new_with_aliases(library, files, [])
    }

    pub(super) fn new_with_aliases(
        library: PathBuf,
        files: impl IntoIterator<Item = (PathBuf, Vec<u8>)>,
        aliases: impl IntoIterator<Item = (PathBuf, PathBuf)>,
    ) -> Result<Self, String> {
        Self::new_with_links(library, files, aliases, [])
    }

    pub(super) fn new_with_links(
        library: PathBuf,
        files: impl IntoIterator<Item = (PathBuf, Vec<u8>)>,
        aliases: impl IntoIterator<Item = (PathBuf, PathBuf)>,
        symlinks: impl IntoIterator<Item = (PathBuf, PathBuf)>,
    ) -> Result<Self, String> {
        Self::new_with_layout(library, files, aliases, symlinks, [], [])
    }

    /// Validate captured files and links, then compute an identity for the complete staged artifact layout.
    pub(super) fn new_with_layout(
        library: PathBuf,
        files: impl IntoIterator<Item = (PathBuf, Vec<u8>)>,
        aliases: impl IntoIterator<Item = (PathBuf, PathBuf)>,
        symlinks: impl IntoIterator<Item = (PathBuf, PathBuf)>,
        directories: impl IntoIterator<Item = PathBuf>,
        directory_links: impl IntoIterator<Item = (PathBuf, PathBuf)>,
    ) -> Result<Self, String> {
        validate_relative_file(&library)?;
        let mut captured = BTreeMap::new();
        for (path, bytes) in files {
            validate_relative_file(&path)?;
            if captured.insert(path.clone(), bytes).is_some() {
                return Err(format!(
                    "duplicate native artifact path '{}'",
                    path.display()
                ));
            }
        }
        let mut captured_aliases = BTreeMap::new();
        for (alias, target) in aliases {
            validate_relative_file(&alias)?;
            validate_relative_file(&target)?;
            if !captured.contains_key(&target) {
                return Err("native artifact alias must refer to a captured file".into());
            }
            if captured.contains_key(&alias) || captured_aliases.insert(alias, target).is_some() {
                return Err("duplicate native artifact alias".into());
            }
        }
        let mut captured_symlinks = BTreeMap::new();
        for (alias, target) in symlinks {
            validate_relative_file(&alias)?;
            validate_relative_file(&target)?;
            if !captured.contains_key(&target) {
                return Err("native symbolic link must refer to a captured file".into());
            }
            if captured.contains_key(&alias)
                || captured_aliases.contains_key(&alias)
                || captured_symlinks.insert(alias, target).is_some()
            {
                return Err("duplicate native symbolic link".into());
            }
        }
        if !captured.contains_key(&library)
            && !captured_symlinks.contains_key(&library)
            && !captured_aliases.contains_key(&library)
        {
            return Err("native artifact closure does not contain load.library".into());
        }
        let mut directories = directories.into_iter().collect::<BTreeSet<_>>();
        for path in &directories {
            validate_relative_file(path)?;
        }
        let mut captured_directory_links = BTreeMap::new();
        for (alias, target) in directory_links {
            validate_relative_file(&alias)?;
            if !target.as_os_str().is_empty() {
                validate_relative_file(&target)?;
            }
            if (!target.as_os_str().is_empty() && !directories.contains(&target))
                || alias == target
                || directories.contains(&alias)
                || captured.contains_key(&alias)
                || captured_aliases.contains_key(&alias)
                || captured_symlinks.contains_key(&alias)
                || captured_directory_links.insert(alias, target).is_some()
            {
                return Err("invalid or duplicate native directory link".into());
            }
        }
        // File parents are already created by staging. Only additional empty
        // directories affect the artifact layout and its content identity.
        directories.retain(|directory| {
            !captured
                .keys()
                .chain(captured_aliases.keys())
                .chain(captured_symlinks.keys())
                .chain(captured_directory_links.keys())
                .any(|path| {
                    path.parent()
                        .is_some_and(|parent| parent.starts_with(directory))
                })
        });
        let mut artifact = Self {
            library,
            files: captured,
            aliases: captured_aliases,
            symlinks: captured_symlinks,
            directories,
            directory_links: captured_directory_links,
            identity: [0; 32],
        };
        let mut hash = Sha256::new();
        hash.update(b"nemo.relay.native.artifact.v1\0");
        hash_path(&mut hash, &artifact.library);
        let entries = artifact.entries().collect::<BTreeMap<_, _>>();
        for (path, (bytes, _)) in entries {
            hash_path(&mut hash, path);
            hash_field(&mut hash, bytes);
        }
        // Link topology changes @loader_path semantics even when bytes match.
        if !artifact.symlinks.is_empty() {
            hash.update(b"nemo.relay.native.symlinks.v1\0");
            for (alias, target) in &artifact.symlinks {
                hash_path(&mut hash, alias);
                hash_path(&mut hash, target);
            }
        }
        if !artifact.directories.is_empty() || !artifact.directory_links.is_empty() {
            hash.update(b"nemo.relay.native.directories.v1\0");
            for directory in &artifact.directories {
                hash_path(&mut hash, directory);
            }
            hash.update(b"nemo.relay.native.directory.links.v1\0");
            for (alias, target) in &artifact.directory_links {
                hash_path(&mut hash, alias);
                hash_path(&mut hash, target);
            }
        }
        artifact.identity = hash.finalize().into();
        Ok(artifact)
    }

    fn entries(&self) -> impl Iterator<Item = (&Path, (&[u8], bool))> {
        let main = self
            .symlinks
            .get(&self.library)
            .or_else(|| self.aliases.get(&self.library))
            .unwrap_or(&self.library);
        self.files
            .iter()
            .map(move |(path, bytes)| (path.as_path(), (bytes.as_slice(), path != main)))
            .chain(
                self.aliases
                    .iter()
                    .chain(&self.symlinks)
                    .map(move |(alias, target)| {
                        (
                            alias.as_path(),
                            (self.files[target].as_slice(), target != main),
                        )
                    }),
            )
    }

    /// Content identity includes the main path and the entire dependency closure.
    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }

    /// Collect loader aliases and content digests to detect conflicting resident image identities.
    fn image_identities(&self) -> Result<LibraryIdentities, String> {
        let mut identities = LibraryIdentities::default();
        for (path, (bytes, dependency)) in self.entries() {
            let digest: [u8; 32] = Sha256::digest(bytes).into();
            for alias in super::native_dependencies::library_aliases(path, bytes)? {
                if identities
                    .images
                    .insert(alias.clone(), digest)
                    .is_some_and(|prior| prior != digest)
                {
                    return Err(format!(
                        "conflicting native image '{alias}' within artifact"
                    ));
                }
                if dependency {
                    identities.dependencies.insert(alias, digest);
                }
            }
        }
        Ok(identities)
    }

    /// Write the captured artifact closure into a private staging directory and recreate its link layout.
    fn stage(self) -> Result<(PathBuf, tempfile::TempDir), String> {
        let mut builder = tempfile::Builder::new();
        builder.prefix("nemo-relay-native-checked-");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            builder.permissions(std::fs::Permissions::from_mode(0o700));
        }
        let directory = builder
            .tempdir()
            .map_err(|e| format!("cannot stage native artifact: {e}"))?;
        for path in &self.directories {
            std::fs::create_dir_all(directory.path().join(path))
                .map_err(|error| format!("cannot stage native directory: {error}"))?;
        }
        for (path, bytes) in self.files {
            let target = directory.path().join(path);
            std::fs::create_dir_all(target.parent().expect("staged file has parent"))
                .map_err(|e| format!("cannot stage native dependency directory: {e}"))?;
            // Reject colliding primary files, including case-equivalent paths.
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&target)
                .map_err(|e| format!("cannot stage native artifact '{}': {e}", target.display()))?;
            file.write_all(&bytes)
                .map_err(|e| format!("cannot write native artifact: {e}"))?;
        }
        for (alias, original) in self.directory_links {
            let target = directory.path().join(&alias);
            std::fs::create_dir_all(target.parent().expect("link has parent"))
                .map_err(|error| format!("cannot stage native directory: {error}"))?;
            #[cfg(unix)]
            if let Err(error) =
                std::os::unix::fs::symlink(relative_link(&alias, &original), &target)
            {
                let same = error.kind() == std::io::ErrorKind::AlreadyExists
                    && target.is_dir()
                    && target
                        .canonicalize()
                        .ok()
                        .zip(directory.path().join(&original).canonicalize().ok())
                        .is_some_and(|(a, b)| a == b);
                if !same {
                    return Err(format!(
                        "cannot stage native directory link '{}': {error}",
                        target.display()
                    ));
                }
            }
            #[cfg(not(unix))]
            {
                let _ = original;
                return Err("native directory links require a Unix host".into());
            }
        }
        // Hard links retain import spellings on case-sensitive staging volumes.
        // On a case-insensitive volume, both spellings already name this file.
        for (alias, original) in self.aliases {
            let alias = directory.path().join(alias);
            let original = directory.path().join(original);
            std::fs::create_dir_all(alias.parent().expect("alias has parent"))
                .map_err(|error| format!("cannot stage native alias directory: {error}"))?;
            if let Err(error) = std::fs::hard_link(&original, &alias) {
                let same = error.kind() == std::io::ErrorKind::AlreadyExists
                    && alias
                        .canonicalize()
                        .ok()
                        .zip(original.canonicalize().ok())
                        .is_some_and(|(alias, original)| alias == original);
                if !same {
                    return Err(format!(
                        "cannot stage native alias '{}': {error}",
                        alias.display()
                    ));
                }
            }
        }
        for (alias, original) in self.symlinks {
            let target = directory.path().join(&alias);
            std::fs::create_dir_all(target.parent().expect("alias has parent"))
                .map_err(|error| format!("cannot stage native link directory: {error}"))?;
            #[cfg(unix)]
            if let Err(error) =
                std::os::unix::fs::symlink(relative_link(&alias, &original), &target)
            {
                let same = error.kind() == std::io::ErrorKind::AlreadyExists
                    && std::fs::symlink_metadata(&target).is_ok_and(|m| m.file_type().is_symlink())
                    && target
                        .canonicalize()
                        .ok()
                        .zip(directory.path().join(&original).canonicalize().ok())
                        .is_some_and(|(alias, original)| alias == original);
                if !same {
                    return Err(format!(
                        "cannot stage native symbolic link '{}': {error}",
                        target.display()
                    ));
                }
            }
            #[cfg(not(unix))]
            {
                let _ = original;
                return Err("native symbolic links require a Unix host".into());
            }
        }
        Ok((directory.path().join(self.library), directory))
    }
}

#[cfg(unix)]
/// Build a relative link target between two paths within the staged artifact.
fn relative_link(alias: &Path, target: &Path) -> PathBuf {
    let parent = alias
        .parent()
        .expect("relative link has parent")
        .components()
        .collect::<Vec<_>>();
    let target = target.components().collect::<Vec<_>>();
    let common = parent
        .iter()
        .zip(&target)
        .take_while(|(a, b)| a == b)
        .count();
    let mut relative = PathBuf::new();
    for _ in common..parent.len() {
        relative.push("..");
    }
    for component in &target[common..] {
        relative.push(component.as_os_str());
    }
    if relative.as_os_str().is_empty() {
        relative.push(".");
    }
    relative
}

/// Hash path components so equivalent separator spellings share an identity.
fn hash_path(hash: &mut Sha256, path: &Path) {
    let normalized: PathBuf = path.components().collect();
    hash_field(hash, normalized.as_os_str().as_encoded_bytes());
}

fn hash_field(hash: &mut Sha256, bytes: &[u8]) {
    hash.update((bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}

/// Require a nonempty relative artifact path made only of normal path components.
fn validate_relative_file(path: &Path) -> Result<(), String> {
    if path.as_os_str().is_empty()
        || !path
            .components()
            .all(|part| matches!(part, Component::Normal(_)))
    {
        return Err(format!(
            "native artifact path '{}' must be a relative file without traversal",
            path.display()
        ));
    }
    Ok(())
}

#[derive(Default)]
struct LibraryIdentities {
    images: BTreeMap<String, [u8; 32]>,
    dependencies: BTreeMap<String, [u8; 32]>,
}

struct ResidentLibrary {
    raw: libloading::Library,
    path: PathBuf,
    // The library and its complete staging closure are deliberately process-owned.
    _directory: tempfile::TempDir,
    bootstrap: OnceLock<Result<Bootstrap_Ref, String>>,
    factory: OnceLock<Result<v1_0::Factory_Ref, String>>,
    identities: LibraryIdentities,
    #[cfg(windows)]
    sidecars: Vec<PathBuf>,
    #[cfg(windows)]
    sidecars_loaded: OnceLock<Result<(), String>>,
}

type ResidentCache = Mutex<HashMap<[u8; 32], &'static ResidentLibrary>>;
fn cache() -> &'static ResidentCache {
    static CACHE: OnceLock<ResidentCache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Inspect process residency independently of checked compatibility or activation.
/// A resident path can exist for a binary rejected after the OS opened it.
pub fn resident_library_path(identity: [u8; 32]) -> Option<PathBuf> {
    cache()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&identity)
        .map(|library| library.path.clone())
}

/// Checked library interfaces; holds no registrations or component executors.
///
/// Each call to `create_plugin` constructs fresh adapter state. Dropping this
/// handle leaves only resident code and staging, never an active component.
pub struct CheckedNativeLibrary {
    resident: &'static ResidentLibrary,
    factory: v1_0::Factory_Ref,
    descriptor: PluginDescriptor,
    identity: [u8; 32],
}

impl CheckedNativeLibrary {
    /// Check the frozen bootstrap, descriptor, and exact SDK contract's modules.
    ///
    /// The artifact must already have passed all host trust and compatibility
    /// checks. No plugin factory runs during this method. Unsupported APIs fail
    /// before staging/loading; manifest disagreement fails before factory lookup.
    pub fn load(
        artifact: NativeLibraryArtifact,
        manifest_api: &str,
        plugin_kind: &str,
    ) -> Result<Self, String> {
        let api: NativeApiVersion = manifest_api.parse().map_err(|e| format!("{e}"))?;
        if !NATIVE_API_VERSION.supports(api) {
            return Err(format!(
                "plugin requires native API {api}; host supports {NATIVE_API_VERSION}"
            ));
        }
        let identity = artifact.identity;
        let resident = {
            let mut cache = cache().lock().unwrap_or_else(|e| e.into_inner());
            if let Some(library) = cache.get(&identity) {
                *library
            } else {
                let identities = artifact.image_identities()?;
                for resident in cache.values() {
                    for (alias, digest) in &identities.dependencies {
                        if resident
                            .identities
                            .images
                            .get(alias)
                            .is_some_and(|prior| prior != digest)
                        {
                            return Err(format!(
                                "native dependency '{alias}' conflicts with a resident library; use a unique dependency identity"
                            ));
                        }
                    }
                }
                #[cfg(windows)]
                let sidecars = artifact
                    .files
                    .iter()
                    .filter(|(path, bytes)| {
                        path.as_path() != artifact.library.as_path()
                            && object::FileKind::parse(bytes.as_slice()).is_ok()
                    })
                    .map(|(path, _)| path.clone())
                    .collect();
                let (path, directory) = artifact.stage()?;
                let raw = load_platform_library(&path)
                    .map_err(|e| format!("cannot load native library '{}': {e}", path.display()))?;
                // Retain ownership as soon as dlopen succeeds, including when
                // subsequent layout/descriptor/module checks reject the binary.
                let library = &*Box::leak(Box::new(ResidentLibrary {
                    raw,
                    path,
                    _directory: directory,
                    bootstrap: OnceLock::new(),
                    factory: OnceLock::new(),
                    identities,
                    #[cfg(windows)]
                    sidecars,
                    #[cfg(windows)]
                    sidecars_loaded: OnceLock::new(),
                }));
                cache.insert(identity, library);
                library
            }
        };
        #[cfg(windows)]
        resident
            .sidecars_loaded
            .get_or_init(|| {
                // MSVC's delay loader uses LoadLibrary by basename after the main
                // LoadLibraryEx search flags expire. Preload verified sidecars so
                // that lookup resolves to the resident staged image.
                for sidecar in &resident.sidecars {
                    let path = resident._directory.path().join(sidecar);
                    let library = load_platform_library(&path).map_err(|e| {
                        format!("cannot load native dependency '{}': {e}", path.display())
                    })?;
                    std::mem::forget(library);
                }
                Ok(())
            })
            .clone()?;
        let bootstrap = resident
            .bootstrap
            .get_or_init(|| {
                // SAFETY: the library is resident for the process lifetime. This
                // checks abi_stable's header before interpreting its root layout.
                // Never use RootModule::load_from_file: its per-root-type cache
                // would return the first plugin library for every artifact.
                let reference = unsafe {
                    resident
                        .raw
                        .get::<AbiHeaderRef>(ROOT_MODULE_LOADER_NAME_WITH_NUL.as_bytes())
                }
                .map(|symbol| *symbol)
                .map_err(|e| format!("invalid native bootstrap header: {e}"))?;
                let header = reference
                    .upgrade()
                    .map_err(|e| format!("invalid native bootstrap header: {e}"))?;
                checked_module::<Bootstrap_Ref>(header)
                    .map_err(|e| format!("incompatible native bootstrap: {e}"))
            })
            .clone()?;
        let descriptor = (bootstrap.descriptor())();
        validate_descriptor(&descriptor, api, plugin_kind, NATIVE_API_VERSION)?;
        // v1.0 is the only implemented contract. A future host must add a
        // separate branch/module, retaining these exact types and semantics.
        let factory = resident
            .factory
            .get_or_init(|| {
                let header = (bootstrap.query_module())(v1_0::FACTORY_MODULE.into())
                    .into_option()
                    .ok_or_else(|| {
                        format!(
                            "native plugin is missing required module '{}'",
                            v1_0::FACTORY_MODULE
                        )
                    })?;
                checked_module::<v1_0::Factory_Ref>(header.get()).map_err(|e| {
                    format!("incompatible native module '{}': {e}", v1_0::FACTORY_MODULE)
                })
            })
            .clone()?;
        Ok(Self {
            resident,
            factory,
            descriptor,
            identity,
        })
    }

    /// Construct fresh plugin adapter state after all binary checks have passed.
    pub fn create_plugin(&self) -> Result<PluginBox, String> {
        (self.factory.create())()
            .into_result()
            .map_err(|e| format!("{}: {}", e.code, e.message))
    }

    /// Binary declaration checked against this activation's manifest.
    pub fn descriptor(&self) -> &PluginDescriptor {
        &self.descriptor
    }
    /// Resident artifact identity, independent of kind and component identity.
    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }
    /// Protected main-library path retained with its dependency closure.
    pub fn resident_path(&self) -> &Path {
        &self.resident.path
    }
}

/// Require ABI layout metadata and check the module type before running its constructor.
fn checked_module<M: RootModule>(header: &LibHeader) -> Result<M, String> {
    // abi_stable permits libraries to opt out of its layout check. A checked
    // native contract requires metadata before any module constructor runs.
    if header.layout().is_none() {
        return Err("native module omits required type-layout metadata".into());
    }
    header.init_root_module::<M>().map_err(|e| e.to_string())
}

fn load_platform_library(path: &Path) -> Result<libloading::Library, libloading::Error> {
    #[cfg(windows)]
    {
        use libloading::os::windows::{
            LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, Library,
        };
        // Search this protected library directory for sidecars without changing
        // PATH or the process-wide DLL search configuration.
        unsafe {
            Library::load_with_flags(
                path,
                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
            )
            .map(Into::into)
        }
    }
    #[cfg(not(windows))]
    {
        // SAFETY: caller has checked trust and retains the entire library closure.
        unsafe { libloading::Library::new(path) }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/native_library_tests.rs"]
mod tests;
