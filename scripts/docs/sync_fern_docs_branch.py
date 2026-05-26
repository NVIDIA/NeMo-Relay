# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the CI-managed Fern docs branch layout.

The source branch keeps author-facing content in ``docs/`` and local Fern
configuration in ``fern/``. The ``docs-website`` branch stores a Fern-native
layout with versioned navigation under ``fern/versions/`` and page content
under ``fern/pages-*``.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Any

import yaml

VERSION_RE = re.compile(r"^(?P<base>[0-9]+\.[0-9]+\.[0-9]+)(?:-(?P<label>alpha|beta|rc)\.(?P<number>[0-9]+))?$")
COPY_EXCLUDES = {
    ".DS_Store",
    "__pycache__",
    "_build",
    "_generated",
    "_source",
    "index.yml",
}
FERN_SYNC_ITEMS = (
    ".gitignore",
    "assets",
    "components",
    "fern.config.json",
    "main.css",
    "README.md",
)
YAML_HEADER = (
    "# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
    "# SPDX-License-Identifier: Apache-2.0\n\n"
)


def read_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(YAML_HEADER)
        yaml.safe_dump(value, handle, sort_keys=False, allow_unicode=False)


def copy_path(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def docs_ignore(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name in COPY_EXCLUDES}


def prefixed_doc_path(value: str, pages_directory: str) -> str:
    if value.startswith(("http://", "https://", "/", "../")):
        return value
    normalized = value[2:] if value.startswith("./") else value
    return f"../{pages_directory}/{normalized}"


def rewrite_doc_references(value: Any, pages_directory: str) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                prefixed_doc_path(item, pages_directory)
                if key in {"folder", "path"} and isinstance(item, str)
                else rewrite_doc_references(item, pages_directory)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [rewrite_doc_references(item, pages_directory) for item in value]
    return value


def rewrite_pages_directory(value: Any, source_directory: str, target_directory: str) -> Any:
    source_prefix = f"../{source_directory}/"
    target_prefix = f"../{target_directory}/"
    if isinstance(value, dict):
        return {
            key: (
                item.replace(source_prefix, target_prefix, 1)
                if key in {"folder", "path"} and isinstance(item, str) and item.startswith(source_prefix)
                else rewrite_pages_directory(item, source_directory, target_directory)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [rewrite_pages_directory(item, source_directory, target_directory) for item in value]
    return value


def parse_release_tag(tag: str) -> tuple[str, str, bool]:
    match = VERSION_RE.fullmatch(tag)
    if match is None:
        raise ValueError(f"docs tags must use raw SemVer, got: {tag}")
    base_version = match.group("base")
    label = match.group("label")
    if label == "alpha":
        raise ValueError(f"alpha docs tags are not published: {tag}")
    is_stable = label is None
    return f"v{base_version}", "stable" if is_stable else "beta", is_stable


def docs_website_product(source_product: dict[str, Any]) -> dict[str, Any]:
    product = dict(source_product)
    product["path"] = "./versions/dev.yml"
    product["versions"] = [
        {
            "display-name": "dev",
            "path": "./versions/dev.yml",
            "slug": "dev",
            "availability": "beta",
        }
    ]
    return product


def load_preserved_product(target_docs_yml: Path) -> dict[str, Any] | None:
    if not target_docs_yml.exists():
        return None
    docs_yml = read_yaml(target_docs_yml)
    products = docs_yml.get("products") if isinstance(docs_yml, dict) else None
    if not products:
        return None
    return products[0]


def write_docs_yml(source_docs_yml: Path, target_docs_yml: Path) -> None:
    preserved_product = load_preserved_product(target_docs_yml)
    docs_yml = read_yaml(source_docs_yml)
    products = docs_yml.get("products")
    if not isinstance(products, list) or not products:
        raise ValueError(f"{source_docs_yml} must define products[0]")

    docs_yml["products"][0] = preserved_product or docs_website_product(products[0])
    write_yaml(target_docs_yml, docs_yml)


def sync_dev(source_root: Path, target_root: Path) -> None:
    source_docs = source_root / "docs"
    source_fern = source_root / "fern"
    target_fern = target_root / "fern"

    if not source_docs.is_dir():
        raise FileNotFoundError(f"source docs directory not found: {source_docs}")
    if not source_fern.is_dir():
        raise FileNotFoundError(f"source Fern directory not found: {source_fern}")

    target_fern.mkdir(parents=True, exist_ok=True)
    versions_dir = target_fern / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)

    pages_dev = target_fern / "pages-dev"
    if pages_dev.exists():
        shutil.rmtree(pages_dev)
    shutil.copytree(source_docs, pages_dev, ignore=docs_ignore)

    navigation = read_yaml(source_docs / "index.yml")
    write_yaml(versions_dir / "dev.yml", rewrite_doc_references(navigation, "pages-dev"))

    for item in FERN_SYNC_ITEMS:
        source = source_fern / item
        if source.exists():
            copy_path(source, target_fern / item)

    write_docs_yml(source_fern / "docs.yml", target_fern / "docs.yml")


def update_github_links(pages_dir: Path, tag: str) -> None:
    replacements = (
        (
            "github.com/NVIDIA/NeMo-Relay/blob/main",
            f"github.com/NVIDIA/NeMo-Relay/blob/{tag}",
        ),
        (
            "github.com/NVIDIA/NeMo-Relay/tree/main",
            f"github.com/NVIDIA/NeMo-Relay/tree/{tag}",
        ),
    )
    for path in pages_dir.rglob("*"):
        if path.suffix not in {".md", ".mdx"} or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        updated = text
        for old, new in replacements:
            updated = updated.replace(old, new)
        if updated != text:
            path.write_text(updated, encoding="utf-8")


def release_version(target_root: Path, tag: str) -> None:
    display_tag, availability, is_stable = parse_release_tag(tag)
    target_fern = target_root / "fern"
    pages_dev = target_fern / "pages-dev"
    pages_version = target_fern / f"pages-{display_tag}"
    versions_dir = target_fern / "versions"
    dev_yml = versions_dir / "dev.yml"
    version_yml = versions_dir / f"{display_tag}.yml"
    docs_yml_path = target_fern / "docs.yml"

    if not pages_dev.is_dir():
        raise FileNotFoundError(f"dev pages directory not found: {pages_dev}")
    if not dev_yml.is_file():
        raise FileNotFoundError(f"dev navigation file not found: {dev_yml}")
    if pages_version.exists():
        shutil.rmtree(pages_version)
    if version_yml.exists():
        version_yml.unlink()

    shutil.copytree(pages_dev, pages_version, ignore=docs_ignore)
    update_github_links(pages_version, tag)

    version_navigation = rewrite_pages_directory(
        read_yaml(dev_yml),
        "pages-dev",
        f"pages-{display_tag}",
    )
    write_yaml(version_yml, version_navigation)

    docs_yml = read_yaml(docs_yml_path)
    product = docs_yml["products"][0]
    versions = product.get("versions", [])
    dev_entry = next(
        (entry for entry in versions if entry.get("display-name") == "dev" or entry.get("slug") == "dev"),
        {
            "display-name": "dev",
            "path": "./versions/dev.yml",
            "slug": "dev",
            "availability": "beta",
        },
    )
    dev_entry = {**dev_entry, "path": "./versions/dev.yml", "slug": "dev", "availability": "beta"}

    latest_entry = {
        "display-name": f"Latest ({display_tag})",
        "path": f"./versions/{display_tag}.yml",
        "slug": "latest",
        "availability": "stable",
    }
    version_entry = {
        "display-name": display_tag,
        "path": f"./versions/{display_tag}.yml",
        "slug": display_tag,
        "availability": availability,
    }
    existing_latest_entries = [
        entry
        for entry in versions
        if entry.get("slug") == "latest"
        or entry.get("display-name") == "Latest"
        or str(entry.get("display-name", "")).startswith("Latest (")
    ]
    remaining_versions = [
        entry
        for entry in versions
        if entry.get("slug") not in {"latest", "dev", display_tag}
        and entry.get("display-name") != "Latest"
        and entry.get("display-name") != display_tag
        and not str(entry.get("display-name", "")).startswith("Latest (")
    ]

    if is_stable:
        product["path"] = f"./versions/{display_tag}.yml"
        product["versions"] = [latest_entry, dev_entry, version_entry, *remaining_versions]
    else:
        product["versions"] = [*existing_latest_entries[:1], dev_entry, version_entry, *remaining_versions]
    write_yaml(docs_yml_path, docs_yml)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    sync_parser = subparsers.add_parser("sync-dev", help="sync source docs into docs-website layout")
    sync_parser.add_argument("--source-root", type=Path, required=True)
    sync_parser.add_argument("--target-root", type=Path, required=True)

    release_parser = subparsers.add_parser("release-version", help="snapshot dev docs as a version")
    release_parser.add_argument("--target-root", type=Path, required=True)
    release_parser.add_argument("--tag", required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "sync-dev":
        sync_dev(args.source_root.resolve(), args.target_root.resolve())
    elif args.command == "release-version":
        release_version(args.target_root.resolve(), args.tag)
    else:
        raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
