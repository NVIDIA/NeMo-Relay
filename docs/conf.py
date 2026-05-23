# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sphinx configuration plus docs-build orchestration for generated API content."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import NamedTuple

import sphinx_js
from packaging.version import InvalidVersion, Version


class _RuntimeDocsProfile(NamedTuple):
    project_title: str
    github_url: str
    python_module: str
    rust_core_crate: str
    rust_adaptive_crate: str


_RELAY_RUNTIME_DOCS_PROFILE = _RuntimeDocsProfile(
    project_title="NVIDIA NeMo Relay",
    github_url="https://github.com/NVIDIA/NeMo-Relay",
    python_module="nemo_relay",
    rust_core_crate="nemo-relay",
    rust_adaptive_crate="nemo-relay-adaptive",
)
_FLOW_RUNTIME_DOCS_PROFILE = _RuntimeDocsProfile(
    project_title="NVIDIA NeMo Flow",
    github_url="https://github.com/NVIDIA/NeMo-Flow",
    python_module="nemo_flow",
    rust_core_crate="nemo-flow",
    rust_adaptive_crate="nemo-flow-adaptive",
)
_DEFAULT_RUNTIME_DOCS_PROFILE = _RELAY_RUNTIME_DOCS_PROFILE


project = _DEFAULT_RUNTIME_DOCS_PROFILE.project_title
copyright = "Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved."
author = "NVIDIA CORPORATION & AFFILIATES"

CONFIG_DOCS_DIR = Path(__file__).resolve().parent
CONFIG_REPO_ROOT = CONFIG_DOCS_DIR.parent
DOCS_DIR = CONFIG_DOCS_DIR
REPO_ROOT = CONFIG_REPO_ROOT
NODE_PACKAGE_DIR = REPO_ROOT / "crates" / "node"
RUST_CRATE_NAMES = ("core", "adaptive")

NODE_API_GENERATED_DIR = DOCS_DIR / "reference" / "api" / "nodejs" / "_generated"
NODE_API_SOURCE_DIR = NODE_API_GENERATED_DIR / "source"
PYTHON_API_GENERATED_DIR = DOCS_DIR / "reference" / "api" / "python" / "_generated"
RUST_API_GENERATED_DIR = DOCS_DIR / "reference" / "api" / "rust" / "_generated"
RUST_API_SOURCE_DIR = DOCS_DIR / "reference" / "api" / "rust" / "_source"

SPHINX_JS_PACKAGE_DIR = Path(sphinx_js.__file__).resolve().parent
SPHINX_JS_SOURCE_DIR = SPHINX_JS_PACKAGE_DIR / "js"
# We patch a local copy of sphinx-js under `_build` so docs generation does
# not mutate installed site-packages.
SPHINX_JS_WORK_DIR = DOCS_DIR / "_build" / ".tooling" / "sphinx-js"

os.environ.setdefault("TYPEDOC_NODE_MODULES", str(CONFIG_REPO_ROOT / "crates" / "node" / "node_modules"))

extensions = [
    "myst_parser",
    "sphinx_design",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
    "sphinx_multiversion",
    "sphinxcontrib_rust",
    "sphinxcontrib.mermaid",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "reference/api/rust/_source/*/README.md",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
root_doc = "index"

myst_enable_extensions = [
    "attrs_block",
    "colon_fence",
    "deflist",
    "html_admonition",
    "replacements",
    "smartquotes",
    "strikethrough",
    "tasklist",
]
myst_fence_as_directive = ["mermaid"]
myst_heading_anchors = 3

autoapi_dirs = ["../python/nemo_relay"]
autoapi_file_patterns = ["*.py", "*.pyi"]
autoapi_root = "reference/api/python/_generated"
autoapi_add_toctree_entry = False
autoapi_member_order = "bysource"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
suppress_warnings = [
    "autoapi.python_import_resolution",
    "ref.python",
    "myst.xref_missing",
    "myst.xref_ambiguous",
    "misc.highlighting_failure",
    "toc.not_included",
]
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_attr_annotations = True

rust_crates = {
    "nemo-relay": str(RUST_API_SOURCE_DIR / "core"),
    "nemo-relay-adaptive": str(RUST_API_SOURCE_DIR / "adaptive"),
}
rust_doc_dir = str(RUST_API_GENERATED_DIR)
rust_rustdoc_fmt = "md"
rust_strip_src = False
rust_visibility = "pub"
rust_generate_mode = "always"

html_theme = "nvidia_sphinx_theme"
html_title = _DEFAULT_RUNTIME_DOCS_PROFILE.project_title
html_static_path = ["_static"]
html_css_files = ["extra.css"]
html_js_files = ["version-switcher.js"]
html_theme_options = {
    "navbar_start": ["navbar-logo"],
    "navbar_center": [],
    "navbar_persistent": ["search-button-field"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "header_links_before_dropdown": 7,
    "navigation_depth": 4,
    "show_nav_level": 2,
    "secondary_sidebar_items": ["page-toc"],
    "navigation_with_keys": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": _DEFAULT_RUNTIME_DOCS_PROFILE.github_url,
            "icon": "fa-brands fa-github",
        }
    ],
}

html_context = {
    "default_docs_version": "main",
}

html_sidebars = {
    "**": ["version-switcher.html", "sidebar-nav-bs.html"],
}

_RELEASE_TAG_PATTERN = re.compile(r"^\d+\.\d+\.\d+(?:-(?:alpha|beta|rc)\.\d+)?$")
_MAX_RELEASE_VERSIONS = 5
_CARGO_SECTION_HEADERS = ("\n[features]\n", "\n[dependencies]\n")
_REMOVED_RUST_LINES = {"mod tests;", "mod private_tests;"}
_TEST_MODULE_DECL = re.compile(r"^mod [A-Za-z_][A-Za-z0-9_]*tests\s*;$")
_PRIVATE_MODULE_DECL = re.compile(r"^(?P<indent>\s*)mod (?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*;$")
_PUBLIC_GLOB_REEXPORT = re.compile(r"^\s*pub use (?P<name>[A-Za-z_][A-Za-z0-9_]*)::\*\s*;$")
_EXCLUDED_RUST_DOC_FEATURES = {"redis-backend"}


def _parse_release_name(name: str) -> Version | None:
    try:
        return Version(name)
    except InvalidVersion:
        return None


def _selected_release_tag_names() -> list[str]:
    try:
        output = subprocess.check_output(
            ("git", "tag", "--list"),
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return []

    selected: dict[tuple[int, int], tuple[Version, str]] = {}
    latest_stable: tuple[Version, str] | None = None
    for raw_name in output.splitlines():
        tag_name = raw_name.strip()
        if not tag_name or _RELEASE_TAG_PATTERN.fullmatch(tag_name) is None:
            continue

        parsed = _parse_release_name(tag_name)
        if parsed is None or parsed.is_devrelease:
            continue

        key = (parsed.major, parsed.minor)
        current = selected.get(key)
        if current is None or parsed > current[0]:
            selected[key] = (parsed, tag_name)

        if not parsed.is_prerelease and (latest_stable is None or parsed > latest_stable[0]):
            latest_stable = (parsed, tag_name)

    ordered = sorted(selected.values(), key=lambda item: item[0], reverse=True)
    chosen: list[str] = []
    seen: set[str] = set()

    if latest_stable is not None:
        seen.add(latest_stable[1])
        chosen.append(latest_stable[1])

    for _, tag_name in ordered:
        if tag_name in seen:
            continue
        chosen.append(tag_name)
        seen.add(tag_name)
        if len(chosen) >= _MAX_RELEASE_VERSIONS:
            break

    return chosen


def _selected_release_tag_whitelist() -> str:
    tag_names = _selected_release_tag_names()
    if not tag_names:
        return r"$^"
    return r"^(?:" + "|".join(re.escape(name) for name in tag_names) + r")$"


# sphinx-multiversion defaults to publishing all branches. Publish tags only.
# `main` still runs regular docs builds in CI, but it should not become a
# published version.
smv_branch_whitelist = r"$^"
smv_tag_whitelist = _selected_release_tag_whitelist()
smv_remote_whitelist = None
smv_released_pattern = r"^refs/tags/\d+\.\d+\.\d+(?:-(?:alpha|beta|rc)\.\d+)?$"
smv_outputdir_format = "{ref.name}"

mermaid_version = "11.6.0"


def _stable_release_order(items):
    # The version switcher mixes semver tags, prereleases, and branch-like
    # names such as `main`. Split them first so Python never has to compare
    # heterogeneous sort keys.
    stable = []
    prerelease = []
    other = []

    for item in items:
        parsed = _parse_release_name(item.name)
        if parsed is None:
            other.append(item)
        elif parsed.is_prerelease or parsed.is_devrelease:
            prerelease.append((parsed, item))
        else:
            stable.append((parsed, item))

    stable.sort(key=lambda entry: entry[0], reverse=True)
    prerelease.sort(key=lambda entry: entry[0], reverse=True)
    other.sort(key=lambda item: item.name.lower())

    ordered = [item for _, item in stable]
    ordered.extend(item for _, item in prerelease)
    ordered.extend(other)
    return ordered


def _find_version_by_name(items, name):
    return next((item for item in items if item.name == name), None)


def _find_latest_stable_release(items):
    for item in _stable_release_order(items):
        if not _is_prerelease_name(item.name):
            return item
    return None


def _display_release_label(name: str) -> str:
    parsed = _parse_release_name(name)
    if parsed is None:
        return name

    if parsed.is_prerelease:
        prerelease_kind, prerelease_number = parsed.pre or ("pre", 0)
        prerelease_labels = {
            "a": "alpha",
            "b": "beta",
            "rc": "rc",
        }
        prerelease_label = prerelease_labels.get(prerelease_kind, prerelease_kind)
        return f"v{parsed.major}.{parsed.minor} {prerelease_label}.{prerelease_number}"

    return f"v{parsed.major}.{parsed.minor}"


def _display_release_url(url: str, name: str) -> str:
    parsed = _parse_release_name(name)
    if parsed is None or parsed.is_prerelease:
        return url
    return url.replace(name, _display_release_label(name), 1)


def _is_prerelease_name(name: str) -> bool:
    parsed = _parse_release_name(name)
    return parsed is not None and (parsed.is_prerelease or parsed.is_devrelease)


def _register_version_template_helpers(app) -> None:
    templates = getattr(app.builder, "templates", None)
    if templates is None:
        return

    environment = templates.environment
    environment.filters["stable_release_order"] = _stable_release_order
    environment.globals["find_version_by_name"] = _find_version_by_name
    environment.globals["find_latest_stable_release"] = _find_latest_stable_release
    environment.globals["display_release_label"] = _display_release_label
    environment.globals["display_release_url"] = _display_release_url
    environment.globals["is_prerelease_name"] = _is_prerelease_name


def _reset_directory(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _replace_once(contents: str, original: str, replacement: str, *, label: str) -> str:
    # Fail fast if upstream plugin code changed; silent partial rewrites would
    # make docs generation much harder to debug.
    if replacement in contents:
        return contents
    if original not in contents:
        raise RuntimeError(f"Unable to apply docs patch for {label}: upstream source changed")
    return contents.replace(original, replacement, 1)


def _require_node_modules() -> None:
    node_modules_dir = NODE_PACKAGE_DIR / "node_modules"
    if not node_modules_dir.is_dir():
        raise RuntimeError(
            "Node.js docs dependencies are missing. "
            "Run `npm install --ignore-scripts` from the repository root before building docs."
        )


def _resolve_runtime_paths(source_docs_dir: Path) -> None:
    global DOCS_DIR
    global REPO_ROOT
    global NODE_PACKAGE_DIR
    global NODE_API_GENERATED_DIR
    global NODE_API_SOURCE_DIR
    global PYTHON_API_GENERATED_DIR
    global RUST_API_GENERATED_DIR
    global RUST_API_SOURCE_DIR
    global SPHINX_JS_WORK_DIR

    DOCS_DIR = source_docs_dir.resolve()
    REPO_ROOT = DOCS_DIR.parent
    NODE_PACKAGE_DIR = REPO_ROOT / "crates" / "node"
    NODE_API_GENERATED_DIR = DOCS_DIR / "reference" / "api" / "nodejs" / "_generated"
    NODE_API_SOURCE_DIR = NODE_API_GENERATED_DIR / "source"
    PYTHON_API_GENERATED_DIR = DOCS_DIR / "reference" / "api" / "python" / "_generated"
    RUST_API_GENERATED_DIR = DOCS_DIR / "reference" / "api" / "rust" / "_generated"
    RUST_API_SOURCE_DIR = DOCS_DIR / "reference" / "api" / "rust" / "_source"
    SPHINX_JS_WORK_DIR = DOCS_DIR / "_build" / ".tooling" / "sphinx-js"


def _detect_runtime_docs_profile(repo_root: Path) -> _RuntimeDocsProfile:
    # sphinx-multiversion runs the current conf.py against historical source
    # trees, so API package names must come from the source tree being built.
    if (repo_root / "python" / _RELAY_RUNTIME_DOCS_PROFILE.python_module).is_dir():
        return _RELAY_RUNTIME_DOCS_PROFILE
    if (repo_root / "python" / _FLOW_RUNTIME_DOCS_PROFILE.python_module).is_dir():
        return _FLOW_RUNTIME_DOCS_PROFILE

    raise RuntimeError(
        "Unable to locate a Python package for docs generation. "
        f"Expected {repo_root / 'python' / _RELAY_RUNTIME_DOCS_PROFILE.python_module} "
        f"or {repo_root / 'python' / _FLOW_RUNTIME_DOCS_PROFILE.python_module}."
    )


def _with_runtime_github_url(theme_options, github_url: str):
    updated_options = dict(theme_options)
    icon_links = [dict(link) for link in updated_options.get("icon_links", [])]
    for link in icon_links:
        if link.get("name") == "GitHub":
            link["url"] = github_url
            break
    updated_options["icon_links"] = icon_links
    return updated_options


def _wire_runtime_config(config) -> None:
    profile = _detect_runtime_docs_profile(REPO_ROOT)
    config.project = profile.project_title
    config.html_title = profile.project_title
    config.html_theme_options = _with_runtime_github_url(config.html_theme_options, profile.github_url)
    config.autoapi_dirs = [str(REPO_ROOT / "python" / profile.python_module)]
    config.rust_crates = {
        profile.rust_core_crate: str(RUST_API_SOURCE_DIR / "core"),
        profile.rust_adaptive_crate: str(RUST_API_SOURCE_DIR / "adaptive"),
    }
    config.rust_doc_dir = str(RUST_API_GENERATED_DIR)


def _ensure_runtime_node_modules() -> None:
    runtime_node_modules = NODE_PACKAGE_DIR / "node_modules"
    if runtime_node_modules.exists():
        return

    shared_node_modules = CONFIG_REPO_ROOT / "crates" / "node" / "node_modules"
    if not shared_node_modules.is_dir():
        return

    runtime_node_modules.parent.mkdir(parents=True, exist_ok=True)
    runtime_node_modules.symlink_to(shared_node_modules)


def _prepare_output_directories() -> None:
    for generated_dir in (NODE_API_GENERATED_DIR, PYTHON_API_GENERATED_DIR, RUST_API_GENERATED_DIR):
        _reset_directory(generated_dir)

    NODE_API_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    RUST_API_SOURCE_DIR.mkdir(parents=True, exist_ok=True)


def _prepare_patched_sphinx_js_runtime() -> None:
    # These patches are compatibility shims for declaration-heavy TypeDoc input
    # until the upstream sphinx-js behavior is robust enough for this repo.
    _reset_directory(SPHINX_JS_WORK_DIR)
    shutil.copytree(SPHINX_JS_SOURCE_DIR, SPHINX_JS_WORK_DIR, dirs_exist_ok=True)

    redirect_aliases = SPHINX_JS_WORK_DIR / "redirectPrivateAliases.ts"
    _write_text(
        redirect_aliases,
        _replace_once(
            _read_text(redirect_aliases),
            "        const decl = name.declarations![0];",
            "        const decl = name.declarations?.[0];\n        if (!decl) {\n          continue;\n        }",
            label="sphinx-js redirectPrivateAliases.ts",
        ),
    )

    convert_top_level = SPHINX_JS_WORK_DIR / "convertTopLevel.ts"
    _write_text(
        convert_top_level,
        _replace_once(
            _read_text(convert_top_level),
            "    const first_sig = func.signatures![0]; // Should always have at least one\n",
            (
                "    const first_sig = func.signatures?.[0];\n"
                "    if (!first_sig) {\n"
                "      return {\n"
                "        ...this.memberProps(func),\n"
                "        ...this.topLevelProperties(func),\n"
                "        is_async: false,\n"
                "        params: [],\n"
                "        returns: [],\n"
                "        type_params: this.typeParamsToIR(func.typeParameters),\n"
                '        kind: "function",\n'
                "        exceptions: [],\n"
                "      };\n"
                "    }\n"
            ),
            label="sphinx-js convertTopLevel.ts",
        ),
    )


def _configure_sphinx_js_environment() -> None:
    # The Node artifact builder launches sphinx-js through `tsx`, so it reads
    # these paths from the environment instead of importing Python directly.
    os.environ["NEMO_RELAY_SPHINX_JS_MAIN_TS"] = str(SPHINX_JS_WORK_DIR / "main.ts")
    os.environ["NEMO_RELAY_SPHINX_JS_IMPORT_HOOK"] = str(SPHINX_JS_WORK_DIR / "registerImportHook.mjs")
    os.environ["NEMO_RELAY_SPHINX_JS_TSX_TSCONFIG"] = str(SPHINX_JS_WORK_DIR / "tsconfig.json")


def _patch_autoapi_summary_signature_normalization() -> None:
    # AutoAPI emits Unicode arrows in summary signatures, but Sphinx's
    # autosummary mangler only understands ASCII `->` return annotations.
    try:
        import autoapi.directives as autoapi_directives
    except ImportError:
        return

    original_mangle_signature = autoapi_directives.mangle_signature
    if getattr(original_mangle_signature, "_nemo_relay_normalizes_unicode_arrow", False):
        return

    def _nemo_relay_mangle_signature(sig: str, *args, **kwargs) -> str:
        normalized = sig.replace(" \u2192 ", " -> ")
        return original_mangle_signature(normalized, *args, **kwargs)

    _nemo_relay_mangle_signature._nemo_relay_normalizes_unicode_arrow = True
    autoapi_directives.mangle_signature = _nemo_relay_mangle_signature


def _skip_imported_type_aliases(_app, what, _name, obj, skip, _options):
    if what == "module" and getattr(obj, "id", None) in {"nemo_relay._native", "nemo_flow._native"}:
        return False

    if skip:
        return skip

    is_type_alias = getattr(obj, "is_type_alias", lambda: False)
    if what == "data" and getattr(obj, "imported", False) and is_type_alias():
        return True

    return None


def _run_node_docs_artifact_builder() -> None:
    subprocess.run(
        ("node", str(CONFIG_REPO_ROOT / "scripts" / "docs" / "build_node_docs_artifacts.mjs")),
        check=True,
        cwd=CONFIG_REPO_ROOT,
        env={
            **os.environ,
            "NEMO_RELAY_DOCS_REPO_ROOT": str(REPO_ROOT),
            "NEMO_RELAY_DOCS_DIR": str(DOCS_DIR),
        },
    )


def _ensure_rustdoc_mod_entry(crate_dir: Path) -> None:
    # `sphinxcontrib-rust` is happier when the crate entrypoint looks like a
    # module tree rooted at `src/mod.rs` instead of `src/lib.rs`.
    lib_rs = crate_dir / "src" / "lib.rs"
    if not lib_rs.exists():
        return

    mod_rs = crate_dir / "src" / "mod.rs"
    _write_text(mod_rs, _read_text(lib_rs))
    lib_rs.unlink()

    cargo_toml = crate_dir / "Cargo.toml"
    cargo_text = _read_text(cargo_toml)
    if "\n[lib]\n" not in cargo_text:
        insert_at = -1
        for header in _CARGO_SECTION_HEADERS:
            insert_at = cargo_text.find(header)
            if insert_at != -1:
                break
        if insert_at == -1:
            cargo_text += '\n[lib]\npath = "src/mod.rs"\n'
        else:
            cargo_text = cargo_text[:insert_at] + '\n[lib]\npath = "src/mod.rs"\n' + cargo_text[insert_at:]
    elif 'path = "src/mod.rs"' not in cargo_text:
        cargo_text = cargo_text.replace("\n[lib]\n", '\n[lib]\npath = "src/mod.rs"\n', 1)
    _write_text(cargo_toml, cargo_text)


def _rust_attr_excludes_default_docs(stripped_line: str) -> bool:
    if stripped_line == "#[doc(hidden)]":
        return True
    if stripped_line.startswith(("#[cfg(test", "#[cfg(all(test", '#[cfg(target_arch = "wasm32"')):
        return True

    for feature in _EXCLUDED_RUST_DOC_FEATURES:
        if f'feature = "{feature}"' in stripped_line and not stripped_line.startswith("#[cfg(not("):
            return True

    return False


def _rust_skip_item_continues(
    line: str,
    saw_brace: bool,
    brace_depth: int,
    paren_depth: int,
) -> tuple[bool, int, int, bool]:
    stripped = line.strip()
    if not stripped:
        return True, brace_depth, paren_depth, saw_brace
    if stripped.startswith(("#[", "///", "//!")):
        return True, brace_depth, paren_depth, saw_brace

    open_count = line.count("{")
    close_count = line.count("}")
    if open_count:
        saw_brace = True
    brace_depth += open_count - close_count
    paren_depth += line.count("(") - line.count(")")

    if saw_brace:
        return brace_depth > 0, brace_depth, paren_depth, saw_brace

    item_ended = stripped.endswith(";") or (stripped.endswith(",") and paren_depth <= 0)
    return not item_ended, brace_depth, paren_depth, saw_brace


def _drop_trailing_rust_doc_comments(lines: list[str]) -> None:
    while lines and lines[-1].lstrip().startswith("///"):
        lines.pop()


def _strip_rustdoc_hostile_lines(crate_dir: Path) -> None:
    # Strip attributes and local test modules from the staged copy only. This
    # avoids rustdoc/plugin parse issues without touching real source files.
    # Items hidden from rustdoc or gated out of the documented default target are
    # removed with their following item so generated reference pages do not
    # expose unavailable public APIs.
    for rust_file in crate_dir.rglob("*.rs"):
        stripped = []
        skip_item = False
        skip_saw_brace = False
        skip_brace_depth = 0
        skip_paren_depth = 0
        skip_attr = False
        skip_attr_bracket_depth = 0

        for line in _read_text(rust_file).splitlines():
            line_stripped = line.strip()

            if skip_attr:
                skip_attr_bracket_depth += line.count("[") - line.count("]")
                if skip_attr_bracket_depth <= 0:
                    skip_attr = False
                continue

            if skip_item:
                skip_item, skip_brace_depth, skip_paren_depth, skip_saw_brace = _rust_skip_item_continues(
                    line,
                    skip_saw_brace,
                    skip_brace_depth,
                    skip_paren_depth,
                )
                continue

            if _rust_attr_excludes_default_docs(line_stripped):
                _drop_trailing_rust_doc_comments(stripped)
                skip_item = True
                skip_saw_brace = False
                skip_brace_depth = 0
                skip_paren_depth = 0
                continue

            if line_stripped in _REMOVED_RUST_LINES or _TEST_MODULE_DECL.fullmatch(line_stripped) is not None:
                continue

            if line.lstrip().startswith("#["):
                skip_attr_bracket_depth = line.count("[") - line.count("]")
                skip_attr = skip_attr_bracket_depth > 0
                continue

            stripped.append(line)
        _write_text(rust_file, "\n".join(stripped) + "\n")


def _promote_reexported_modules_for_docs(crate_dir: Path) -> None:
    # `sphinxcontrib-rust` only generates child module pages for public modules.
    # Some FFI surfaces flatten submodules with `pub use foo::*;` while keeping
    # the modules themselves private. In the staged docs copy, promote only
    # those re-exported modules so generated docs reflect the public surface.
    for rust_file in crate_dir.rglob("*.rs"):
        lines = _read_text(rust_file).splitlines()
        reexported = {match.group("name") for line in lines if (match := _PUBLIC_GLOB_REEXPORT.match(line)) is not None}
        if not reexported:
            continue

        updated = []
        changed = False
        for line in lines:
            match = _PRIVATE_MODULE_DECL.match(line)
            if match is not None and match.group("name") in reexported:
                updated.append(f"{match.group('indent')}pub mod {match.group('name')};")
                changed = True
            else:
                updated.append(line)

        if changed:
            _write_text(rust_file, "\n".join(updated) + "\n")


def _stage_rust_crate(crate_name: str) -> None:
    source_dir = REPO_ROOT / "crates" / crate_name
    dest_dir = RUST_API_SOURCE_DIR / crate_name
    shutil.rmtree(dest_dir, ignore_errors=True)
    shutil.copytree(source_dir, dest_dir, ignore=shutil.ignore_patterns("tests", "target"))
    _ensure_rustdoc_mod_entry(dest_dir)
    _strip_rustdoc_hostile_lines(dest_dir)
    _promote_reexported_modules_for_docs(dest_dir)


def _stage_rust_sources() -> None:
    for crate_name in RUST_CRATE_NAMES:
        _stage_rust_crate(crate_name)


def _rewrite_generated_rust_toctrees() -> None:
    for doc_path in RUST_API_GENERATED_DIR.rglob("*.md"):
        lines = _read_text(doc_path).splitlines()
        updated: list[str] = []
        in_toctree = False
        changed = False

        for line in lines:
            stripped = line.strip()
            if stripped == ":::{toctree}":
                in_toctree = True
                updated.append(line)
                continue

            if in_toctree and stripped == ":::":  # end of MyST directive block
                in_toctree = False
                updated.append(line)
                continue

            if not in_toctree or not stripped or stripped.startswith(":"):
                updated.append(line)
                continue

            direct_target = doc_path.parent / f"{stripped}.md"
            src_target = doc_path.parent / "src" / f"{stripped}.md"
            if not direct_target.exists() and src_target.exists():
                indent = line[: len(line) - len(line.lstrip())]
                updated.append(f"{indent}src/{stripped}")
                changed = True
                continue

            updated.append(line)

        if changed:
            _write_text(doc_path, "\n".join(updated) + "\n")


def _postprocess_generated_rust_docs(_app) -> None:
    _rewrite_generated_rust_toctrees()


def _prepare_api_sources(_app, _config) -> None:
    _resolve_runtime_paths(Path(_app.srcdir))
    _wire_runtime_config(_config)
    _ensure_runtime_node_modules()
    _prepare_output_directories()
    _require_node_modules()
    _prepare_patched_sphinx_js_runtime()
    _configure_sphinx_js_environment()
    _run_node_docs_artifact_builder()
    _stage_rust_sources()


def setup(app):
    _patch_autoapi_summary_signature_normalization()
    app.connect("autoapi-skip-member", _skip_imported_type_aliases)
    # Generated sources must exist before Sphinx resolves toctrees, so this
    # runs on `config-inited` rather than `builder-inited`.
    app.connect("config-inited", _prepare_api_sources)
    app.connect("builder-inited", _postprocess_generated_rust_docs)
    app.connect("builder-inited", _register_version_template_helpers)
