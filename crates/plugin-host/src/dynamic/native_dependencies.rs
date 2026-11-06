// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Capture package-local binary imports without scanning source or build trees.

use super::native_library::NativeLibraryArtifact;
#[path = "native_elf.rs"]
mod elf;
#[path = "native_macho.rs"]
mod macho;
#[path = "native_paths.rs"]
mod paths;
use elf::{Target as ElfTarget, Tokens as ElfTokens};
use object::read::elf::{Dyn as _, FileHeader, ProgramHeader as _};
use object::read::macho::FatArch as _;
use paths::DirectoryLayout;
use std::{
    collections::{BTreeMap, VecDeque},
    path::{Component, Path, PathBuf},
};

#[derive(Default)]
struct Imports {
    libraries: Vec<String>,
    search: Vec<String>,
    inherit_search: bool,
    depth_first: bool,
    identity: Option<String>,
}
fn pe_imports<Pe: object::read::pe::ImageNtHeaders>(
    file: &object::read::pe::PeFile<'_, Pe>,
) -> Result<Imports, Box<dyn std::error::Error>> {
    let mut result = Imports::default();
    if let Some(table) = file.import_table()? {
        let mut descriptors = table.descriptors()?;
        while let Some(descriptor) = descriptors.next()? {
            result.libraries.push(
                std::str::from_utf8(table.name(descriptor.name.get(object::LittleEndian))?)?
                    .to_owned(),
            );
        }
    }
    if let Some(table) = file
        .data_directories()
        .delay_load_import_table(file.data(), &file.section_table())?
    {
        let mut descriptors = table.descriptors()?;
        while let Some(descriptor) = descriptors.next()? {
            result.libraries.push(
                std::str::from_utf8(
                    table.name(descriptor.dll_name_rva.get(object::LittleEndian))?,
                )?
                .to_owned(),
            );
        }
    }
    Ok(result)
}

fn elf_strings<'data, Elf: FileHeader>(
    file: &object::read::elf::ElfFile<'data, Elf>,
    entries: &[object::read::elf::Dynamic],
) -> Result<&'data [u8], Box<dyn std::error::Error>> {
    let value = |tag| {
        entries
            .iter()
            .find(|entry| entry.tag == tag)
            .map(|entry| entry.val)
    };
    let address = value(object::elf::DT_STRTAB).ok_or("missing ELF DT_STRTAB")?;
    let size = value(object::elf::DT_STRSZ).ok_or("missing ELF DT_STRSZ")?;
    for segment in file.elf_program_headers() {
        if segment.p_type(file.endian()) == object::elf::PT_LOAD
            && let Some(bytes) = segment
                .data_range(file.endian(), file.data(), address, size)
                .map_err(|_| "invalid ELF dynamic string table segment")?
        {
            return Ok(bytes);
        }
    }
    Err("ELF dynamic string table is outside loadable segments".into())
}

fn elf_imports<Elf: FileHeader>(
    file: &object::read::elf::ElfFile<'_, Elf>,
) -> Result<Imports, Box<dyn std::error::Error>> {
    let mut result = Imports::default();
    let Some(segment) = file
        .elf_program_headers()
        .iter()
        .find(|segment| segment.p_type(file.endian()) == object::elf::PT_DYNAMIC)
    else {
        return Ok(result);
    };
    // The OS loader uses program headers. Section headers can be stripped from
    // a shared library without changing its imports or runtime search paths.
    let entries = segment
        .dynamic(file.endian(), file.data())?
        .unwrap_or_default()
        .iter()
        .take_while(|entry| entry.tag(file.endian()) != object::elf::DT_NULL)
        .map(|entry| object::read::elf::Dynamic {
            tag: entry.tag(file.endian()),
            val: entry.val(file.endian()),
        })
        .collect::<Vec<_>>();
    let strings = if entries.iter().any(|entry| entry.is_string()) {
        elf_strings(file, &entries)?
    } else {
        &[]
    };
    let string = |entry: object::read::elf::Dynamic| -> Result<&str, Box<dyn std::error::Error>> {
        let bytes = strings
            .get(usize::try_from(entry.val)?..)
            .ok_or("invalid ELF string offset")?;
        let end = bytes
            .iter()
            .position(|byte| *byte == 0)
            .ok_or("unterminated ELF string")?;
        Ok(std::str::from_utf8(&bytes[..end])?)
    };
    let mut rpath = Vec::new();
    let mut runpath = None;
    // A wholly empty ELF search path disables lookup; empty entries inside a
    // nonempty list instead select the working directory.
    let search = |path: &str| {
        path.split(':')
            .filter(|_| !path.is_empty())
            .map(str::to_owned)
            .collect::<Vec<_>>()
    };
    for entry in entries {
        match entry.tag {
            object::elf::DT_SONAME => result.identity = Some(string(entry)?.to_owned()),
            object::elf::DT_NEEDED => result.libraries.push(string(entry)?.to_owned()),
            object::elf::DT_RPATH => rpath.extend(search(string(entry)?)),
            object::elf::DT_RUNPATH => runpath
                .get_or_insert_with(Vec::new)
                .extend(search(string(entry)?)),
            _ => {}
        }
    }
    result.inherit_search = runpath.is_none();
    result.search = runpath.unwrap_or(rpath);
    Ok(result)
}

fn imports(bytes: &[u8]) -> Result<Imports, String> {
    fn read(bytes: &[u8]) -> Result<Imports, Box<dyn std::error::Error>> {
        macro_rules! fat {
            ($file:expr) => {{
                let file = $file;
                let arch = file
                    .arches()
                    .iter()
                    .find(|arch| macho::host_architecture(arch.architecture()))
                    .ok_or("native universal library has no host architecture")?;
                return read(arch.data(bytes)?);
            }};
        }
        match object::FileKind::parse(bytes)? {
            object::FileKind::MachOFat32 => {
                fat!(object::read::macho::MachOFatFile32::parse(bytes)?)
            }
            object::FileKind::MachOFat64 => {
                fat!(object::read::macho::MachOFatFile64::parse(bytes)?)
            }
            _ => {}
        }
        let mut result = Imports::default();
        macro_rules! macho {
            ($file:expr) => {{
                let file = $file;
                result.inherit_search = true;
                result.depth_first = true;
                let mut commands = file.macho_load_commands()?;
                while let Some(command) = commands.next()? {
                    if let object::read::macho::LoadCommandVariant::IdDylib(library) =
                        command.variant()?
                    {
                        result.identity = Some(
                            std::str::from_utf8(
                                command.string(file.endian(), library.dylib.name)?,
                            )?
                            .to_owned(),
                        );
                    } else if let Some(library) = command.dylib()? {
                        result.libraries.push(
                            std::str::from_utf8(
                                command.string(file.endian(), library.dylib.name)?,
                            )?
                            .to_owned(),
                        );
                    } else if let object::read::macho::LoadCommandVariant::Rpath(path) =
                        command.variant()?
                    {
                        result.search.push(
                            std::str::from_utf8(command.string(file.endian(), path.path)?)?
                                .to_owned(),
                        );
                    }
                }
            }};
        }
        let file = object::File::parse(bytes)?;
        match &file {
            object::File::Elf32(file) => return elf_imports(file),
            object::File::Elf64(file) => return elf_imports(file),
            object::File::MachO32(file) => macho!(file),
            object::File::MachO64(file) => macho!(file),
            object::File::Pe32(file) => return pe_imports(file),
            object::File::Pe64(file) => return pe_imports(file),
            _ => return Err("unsupported native library format".into()),
        }
        Ok(result)
    }
    read(bytes).map_err(|error| format!("cannot read native imports: {error}"))
}

fn normalized(path: &Path) -> Option<PathBuf> {
    let mut result = PathBuf::new();
    for component in path.components() {
        match component {
            Component::ParentDir => {
                if !result.pop() {
                    return None;
                }
            }
            Component::CurDir => {}
            component => result.push(component.as_os_str()),
        }
    }
    Some(result)
}
fn expand(
    path: &str,
    origin: &Path,
    target: Option<ElfTarget>,
    tokens: Option<&ElfTokens>,
) -> Result<Option<PathBuf>, String> {
    let Some(origin) = origin.to_str() else {
        return Ok(None);
    };
    let expanded = if target.is_some() {
        let Some(expanded) = elf::expand(path, origin, tokens)? else {
            return Ok(None);
        };
        expanded
    } else if let Some(tail) = path
        .strip_prefix("@loader_path")
        .filter(|tail| tail.is_empty() || tail.starts_with('/'))
    {
        format!("{origin}{tail}")
    } else if path.starts_with('@') {
        return Ok(None);
    } else {
        path.to_owned()
    };
    // Canonical Windows paths use the verbatim prefix, where `/` is not a
    // separator. Loader tokens still use `/`, so adapt them before filesystem access.
    #[cfg(windows)]
    let expanded = expanded.replace('/', "\\");
    Ok(Some(PathBuf::from(expanded)))
}
fn runtime_search_paths(
    paths: &[String],
    root: &Path,
    origin: &Path,
    target: Option<ElfTarget>,
    tokens: Option<&ElfTokens>,
) -> Result<Vec<String>, String> {
    let mut result = Vec::new();
    for path in paths {
        if package_absolute(path, root) {
            return Err(format!(
                "native runtime search path '{path}' is absolute within the package; use loader-relative paths"
            ));
        }
        if let Some(expanded) = expand(path, origin, target, tokens)? {
            if !expanded.is_absolute() {
                return Err(format!(
                    "native runtime search path '{path}' depends on the working directory; use loader-relative paths"
                ));
            }
            result.push(
                expanded
                    .to_str()
                    .ok_or("native search path is not UTF-8")?
                    .to_owned(),
            );
        }
    }
    Ok(result)
}

/// Resolve package-local imports while preserving Unix loader traversal through symlinks.
fn resolve(
    root: &Path,
    origin: &Path,
    name: &str,
    search: &[String],
    target: Option<ElfTarget>,
    tokens: Option<&ElfTokens>,
    is_macho: bool,
) -> Result<Option<PathBuf>, String> {
    let mut candidates = Vec::new();
    if let Some(suffix) = name.strip_prefix("@rpath/") {
        candidates.extend(
            search
                .iter()
                .map(|directory| Path::new(directory).join(suffix)),
        );
    } else if name.starts_with("@loader_path/")
        || name.starts_with("$ORIGIN/")
        || name.starts_with("${ORIGIN}/")
        || Path::new(name).is_absolute()
    {
        candidates.extend(expand(name, origin, target, tokens)?);
    } else {
        if (is_macho && !name.starts_with('@')) || (target.is_some() && name.contains('/')) {
            return Err(format!(
                "native dependency '{name}' depends on the working directory; use loader-relative imports"
            ));
        }
        candidates.extend(
            search
                .iter()
                .map(|directory| Path::new(directory).join(name)),
        );
        if name.to_ascii_lowercase().ends_with(".dll") {
            candidates.push(origin.join(name));
        }
    }
    for candidate in candidates {
        let candidate = if !is_macho && (target.is_none() || cfg!(windows)) {
            let Some(candidate) = normalized(&candidate) else {
                continue;
            };
            candidate
        } else {
            candidate
        };
        if !candidate.exists() {
            continue;
        }
        let physical = candidate
            .canonicalize()
            .map_err(|error| error.to_string())?;
        if !physical.starts_with(root) {
            if !candidate.starts_with(root) {
                continue;
            }
            return Err(format!(
                "native dependency '{}' escapes its package",
                candidate.display()
            ));
        }
        if let Some(target) = target
            && !target.accepts_file(&physical)?
        {
            continue;
        }
        if is_macho && !macho::compatible_file(&physical)? {
            continue;
        }
        return Ok(Some(if candidate.starts_with(root) {
            candidate
        } else {
            physical
        }));
    }
    Ok(None)
}

/// Names by which an OS loader can reuse a captured shared library.
pub(super) fn library_aliases(path: &Path, bytes: &[u8]) -> Result<Vec<String>, String> {
    // Artifacts may also carry non-binary files owned by the staging closure.
    if object::FileKind::parse(bytes).is_err() {
        return Ok(Vec::new());
    }
    let identity = imports(bytes)?.identity;
    // Unix loaders identify named images by SONAME/install name. A basename
    // alone must not conflate distinct path-qualified identities. PE lookup is
    // by DLL basename, even when the files occupy different directories.
    let mut aliases = if cfg!(windows) || identity.is_none() {
        vec![
            path.file_name()
                .ok_or("native dependency has no filename")?
                .to_string_lossy()
                .into_owned(),
        ]
    } else {
        Vec::new()
    };
    if let Some(identity) = identity {
        aliases.push(identity);
    }
    #[cfg(windows)]
    for alias in &mut aliases {
        *alias = alias.to_ascii_lowercase();
    }
    Ok(aliases)
}

fn package_absolute(value: &str, root: &Path) -> bool {
    let path = Path::new(value);
    path.is_absolute()
        && (normalized(path).is_some_and(|p| p.starts_with(root))
            || path.canonicalize().is_ok_and(|p| p.starts_with(root)))
}

// Compare directory entries without following symlinks. Equal canonical targets
// alone would also merge distinct symlink aliases that must remain in staging.
fn same_import_path(root: &Path, first: &Path, second: &Path) -> Result<bool, String> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if first.components().count() != second.components().count() {
            return Ok(false);
        }
        let mut left = root.to_owned();
        let mut right = root.to_owned();
        for (a, b) in first.components().zip(second.components()) {
            left.push(a);
            right.push(b);
            if left != right {
                let a = std::fs::symlink_metadata(&left).map_err(|error| error.to_string())?;
                let b = std::fs::symlink_metadata(&right).map_err(|error| error.to_string())?;
                if (a.dev(), a.ino()) != (b.dev(), b.ino()) {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }
    #[cfg(not(unix))]
    {
        Ok(root
            .join(first)
            .canonicalize()
            .map_err(|error| error.to_string())?
            == root
                .join(second)
                .canonicalize()
                .map_err(|error| error.to_string())?)
    }
}

// Mach-O resolves @loader_path from the physical image, including when a
// directory or file in its import path is a symbolic link.
fn macho_staged_path(
    root: &Path,
    imported: &Path,
    physical: &Path,
    aliases: &mut BTreeMap<PathBuf, PathBuf>,
    symlinks: &mut BTreeMap<PathBuf, PathBuf>,
) -> Result<PathBuf, String> {
    let canonical = physical
        .strip_prefix(root)
        .map_err(|_| "native library escapes its package")?;
    if imported != canonical {
        let links = if same_import_path(root, imported, canonical)? {
            aliases
        } else {
            symlinks
        };
        links.insert(imported.to_owned(), canonical.to_owned());
    }
    Ok(canonical.to_owned())
}

pub(super) fn capture(
    library: &Path,
    manifest_root: &Path,
    bytes: Vec<u8>,
) -> Result<NativeLibraryArtifact, String> {
    capture_with_tokens(library, manifest_root, bytes, None)
}

fn capture_with_tokens(
    library: &Path,
    manifest_root: &Path,
    bytes: Vec<u8>,
    tokens: Option<&ElfTokens>,
) -> Result<NativeLibraryArtifact, String> {
    let target = ElfTarget::from_header(&bytes);
    let is_macho = matches!(
        object::FileKind::parse(bytes.as_slice()),
        Ok(object::FileKind::MachO32
            | object::FileKind::MachO64
            | object::FileKind::MachOFat32
            | object::FileKind::MachOFat64)
    );
    let parent = library
        .parent()
        .ok_or("native library has no directory")?
        .canonicalize()
        .map_err(|error| error.to_string())?;
    let library = parent.join(
        library
            .file_name()
            .ok_or("native library must name a file")?,
    );
    let manifest_root = manifest_root
        .canonicalize()
        .map_err(|error| error.to_string())?;
    let root = if library.starts_with(&manifest_root) {
        manifest_root
    } else {
        parent
    };
    let relative = library
        .strip_prefix(&root)
        .map_err(|_| "native library must be within its package")?
        .to_owned();
    let physical = library.canonicalize().unwrap_or_else(|_| library.clone());
    let mut aliases = BTreeMap::new();
    let mut symlinks = BTreeMap::new();
    let mut layout = DirectoryLayout::default();
    let main = if is_macho {
        macho_staged_path(&root, &relative, &physical, &mut aliases, &mut symlinks)?
    } else {
        relative.clone()
    };
    let mut physical_files = BTreeMap::from([(physical, vec![main.clone()])]);
    let mut budget = bytes.len();
    let mut files = BTreeMap::from([(main.clone(), bytes)]);
    let mut pending = VecDeque::from([(main, Vec::<String>::new())]);
    while let Some((path, inherited)) = pending.pop_front() {
        let parsed = imports(&files[&path])?;
        let origin = root
            .join(&path)
            .parent()
            .expect("library has parent")
            .to_owned();
        let mut search = runtime_search_paths(&parsed.search, &root, &origin, target, tokens)?;
        let child_search = if parsed.inherit_search {
            search.extend(inherited);
            search.clone()
        } else {
            // RUNPATH applies only to this image's direct imports. Keep ancestor
            // RPATH entries for descendants that do not have their own RUNPATH.
            inherited
        };
        let mut discovered = Vec::new();
        for name in parsed.libraries {
            if package_absolute(&name, &root) {
                return Err(format!(
                    "native dependency '{name}' is absolute within the package; use loader-relative imports"
                ));
            }
            let Some(dependency) =
                resolve(&root, &origin, &name, &search, target, tokens, is_macho)?
            else {
                continue;
            };
            let physical = dependency
                .canonicalize()
                .map_err(|error| error.to_string())?;
            // Use Windows' canonical spelling to deduplicate case-insensitive imports.
            let staged = if cfg!(windows) {
                physical.clone()
            } else {
                layout.file_path(&root, &dependency)?
            };
            let imported = staged
                .strip_prefix(&root)
                .expect("checked package path")
                .to_owned();
            let relative = if is_macho {
                macho_staged_path(&root, &imported, &physical, &mut aliases, &mut symlinks)?
            } else {
                imported
            };
            if files.len()
                + aliases.len()
                + symlinks.len()
                + layout.directories.len()
                + layout.links.len()
                > 4096
            {
                return Err("native dependency closure exceeds file budget".into());
            }
            if files.contains_key(&relative) {
                continue;
            }
            if aliases.contains_key(&relative) {
                continue;
            }
            if files.len()
                + aliases.len()
                + symlinks.len()
                + layout.directories.len()
                + layout.links.len()
                >= 4096
            {
                return Err("native dependency closure exceeds file budget".into());
            }
            let mut alias_target = None;
            if let Some(previous) = physical_files.get(&physical) {
                for previous in previous {
                    if same_import_path(&root, previous, &relative)? {
                        alias_target = Some(previous.clone());
                        break;
                    }
                }
            }
            if let Some(target) = alias_target {
                aliases.insert(relative, target);
                continue;
            }
            physical_files
                .entry(physical.clone())
                .or_default()
                .push(relative.clone());
            let bytes = super::bounded::read_regular_file_with_limit(
                &physical,
                "native dependency",
                (512 * 1024 * 1024usize).saturating_sub(budget) as u64,
            )?;
            if let Some(target) = target
                && !target.accepts(&bytes)
            {
                return Err("native dependency changed ELF target while being captured".into());
            }
            if is_macho && !macho::compatible(&bytes)? {
                return Err("native dependency changed Mach-O target while being captured".into());
            }
            budget += bytes.len();
            files.insert(relative.clone(), bytes);
            discovered.push((relative, child_search.clone()));
        }
        // A shared image inherits paths from the importer that first discovers
        // it. Mach-O descends in import order; ELF maps siblings before children.
        if parsed.depth_first {
            for dependency in discovered.into_iter().rev() {
                pending.push_front(dependency);
            }
        } else {
            pending.extend(discovered);
        }
    }
    NativeLibraryArtifact::new_with_layout(
        relative,
        files,
        aliases,
        symlinks,
        layout.directories,
        layout.links,
    )
}

#[cfg(test)]
#[path = "../../tests/unit/native_dependency_pe_tests.rs"]
mod pe_tests;

#[cfg(test)]
#[path = "../../tests/unit/native_dependency_elf_tests.rs"]
mod elf_tests;
