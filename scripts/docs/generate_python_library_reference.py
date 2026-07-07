# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate Fern-compatible Python library reference pages from local source."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
from pathlib import Path

from reference_common import display_path, escape_mdx_text, frontmatter, reset_output_dir

PRIVATE_MODULE_ALLOWLIST = {"__init__"}
BASE_URL = "/reference/api/python-library-reference"


@dataclass(frozen=True, slots=True)
class Package:
    distribution_name: str
    module_name: str
    source_root: Path
    description: str
    public_api_source: Path | None = None


PACKAGES = (
    Package(
        distribution_name="nemo-relay",
        module_name="nemo_relay",
        source_root=Path("python/nemo_relay"),
        description="Python runtime SDK for NeMo Relay.",
    ),
    Package(
        distribution_name="nemo-relay-plugin",
        module_name="nemo_relay_plugin",
        source_root=Path("python/plugin/src/nemo_relay_plugin"),
        description="Python SDK for NeMo Relay dynamic worker plugins.",
        public_api_source=Path("_api.py"),
    ),
)


@dataclass(slots=True)
class FunctionDoc:
    name: str
    signature: str
    summary: str
    is_async: bool = False


@dataclass(slots=True)
class ClassDoc:
    name: str
    bases: list[str]
    summary: str
    methods: list[FunctionDoc] = field(default_factory=list)


@dataclass(slots=True)
class ModuleDoc:
    package: Package
    name: str
    title: str
    source: Path
    summary: str
    classes: list[ClassDoc] = field(default_factory=list)
    functions: list[FunctionDoc] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    reexports: list[str] = field(default_factory=list)


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _is_public(name: str) -> bool:
    return not name.startswith("_")


def _module_name(package: Package, package_root: Path, path: Path) -> str:
    rel = path.relative_to(package_root)
    parts = list(rel.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join([package.module_name, *parts])


def _is_public_module(package_root: Path, path: Path) -> bool:
    rel = path.relative_to(package_root)
    return all(part in PRIVATE_MODULE_ALLOWLIST or not part.startswith("_") for part in rel.with_suffix("").parts)


def _escape_text(value: str) -> str:
    return escape_mdx_text(value.replace("``", "`"))


def _summary(docstring: str | None) -> str:
    if not docstring:
        return ""
    lines = [line.strip() for line in docstring.strip().splitlines()]
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        current.append(line)
    if current:
        paragraphs.append(" ".join(current))
    return _escape_text(paragraphs[0]) if paragraphs else ""


def _annotation(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    return ast.unparse(node)


def _format_arg(arg: ast.arg, default: ast.AST | None = None) -> str:
    rendered = arg.arg
    annotation = _annotation(arg.annotation)
    if annotation:
        rendered += f": {annotation}"
    if default is not None:
        rendered += f" = {ast.unparse(default)}"
    return rendered


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef, *, drop_self: bool = False) -> str:
    args = node.args
    positional = list(args.posonlyargs) + list(args.args)
    if drop_self and positional and positional[0].arg in {"self", "cls"}:
        positional = positional[1:]

    defaults: list[ast.AST | None] = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    rendered = [_format_arg(arg, default) for arg, default in zip(positional, defaults, strict=True)]

    if args.vararg is not None:
        rendered.append("*" + _format_arg(args.vararg))
    elif args.kwonlyargs:
        rendered.append("*")

    for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        rendered.append(_format_arg(arg, default))

    if args.kwarg is not None:
        rendered.append("**" + _format_arg(args.kwarg))

    return_annotation = _annotation(node.returns)
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    suffix = f" -> {return_annotation}" if return_annotation else ""
    return f"{prefix} {node.name}({', '.join(rendered)}){suffix}"


def _class_bases(node: ast.ClassDef) -> list[str]:
    return [ast.unparse(base) for base in node.bases]


def _class_init_signature(node: ast.ClassDef) -> str | None:
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "__init__":
            return _signature(item, drop_self=True).replace("__init__", node.name, 1)
    return None


def _class_doc(node: ast.ClassDef, impl_node: ast.ClassDef | None = None) -> ClassDoc:
    impl_methods = {
        item.name: item
        for item in (impl_node.body if impl_node is not None else [])
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    methods = [
        _function_doc(item, drop_self=True, impl_node=impl_methods.get(item.name))
        for item in node.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public(item.name)
    ]
    init_signature = _class_init_signature(node)
    if init_signature is not None:
        methods.insert(
            0,
            FunctionDoc(name=node.name, signature=init_signature, summary="Create an instance."),
        )
    return ClassDoc(
        name=node.name,
        bases=_class_bases(node),
        summary=_summary(ast.get_docstring(impl_node) if impl_node is not None else ast.get_docstring(node)),
        methods=methods,
    )


def _assignment_names(node: ast.Assign | ast.AnnAssign) -> list[str]:
    targets: list[ast.expr] = node.targets if isinstance(node, ast.Assign) else [node.target]
    names: list[str] = []
    for target in targets:
        if isinstance(target, ast.Name) and _is_public(target.id):
            names.append(target.id)
    return names


def _reexport_names(node: ast.ImportFrom) -> list[str]:
    if node.module == "__future__":
        return []
    if node.level == 0 and not (node.module or "").startswith("nemo_relay"):
        return []
    names: list[str] = []
    for alias in node.names:
        name = alias.asname or alias.name
        if _is_public(name):
            names.append(name)
    return names


def _impl_lookup(impl_tree: ast.Module | None) -> dict[str, ast.AST]:
    if impl_tree is None:
        return {}
    return {
        node.name: node
        for node in impl_tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _function_doc(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    drop_self: bool = False,
    impl_node: ast.FunctionDef | ast.AsyncFunctionDef | None = None,
) -> FunctionDoc:
    return FunctionDoc(
        name=node.name,
        signature=_signature(node, drop_self=drop_self),
        summary=_summary(ast.get_docstring(impl_node) if impl_node is not None else ast.get_docstring(node)),
        is_async=isinstance(node, ast.AsyncFunctionDef),
    )


def _collect_module(
    package: Package,
    package_root: Path,
    module_path: Path,
    source_path: Path,
    api_tree: ast.Module,
    impl_tree: ast.Module | None,
) -> ModuleDoc:
    impl_nodes = _impl_lookup(impl_tree)
    module = ModuleDoc(
        package=package,
        name=_module_name(package, package_root, module_path),
        title=_module_name(package, package_root, module_path),
        source=source_path,
        summary=_summary((ast.get_docstring(impl_tree) if impl_tree else None) or ast.get_docstring(api_tree)),
    )

    for node in api_tree.body:
        if isinstance(node, ast.ClassDef) and _is_public(node.name):
            impl_node = impl_nodes.get(node.name)
            module.classes.append(_class_doc(node, impl_node if isinstance(impl_node, ast.ClassDef) else None))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public(node.name):
            impl_node = impl_nodes.get(node.name)
            module.functions.append(
                _function_doc(
                    node,
                    impl_node=impl_node if isinstance(impl_node, (ast.FunctionDef, ast.AsyncFunctionDef)) else None,
                )
            )
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            module.aliases.extend(_assignment_names(node))
        elif module_path.stem == "__init__" and isinstance(node, ast.ImportFrom):
            module.reexports.extend(_reexport_names(node))

    module.aliases = sorted(set(module.aliases))
    module.reexports = sorted(set(module.reexports) - set(module.aliases))
    return module


def _discover_modules(package: Package, package_root: Path) -> list[ModuleDoc]:
    py_files = {path.with_suffix("").relative_to(package_root): path for path in package_root.rglob("*.py")}
    pyi_files = {path.with_suffix("").relative_to(package_root): path for path in package_root.rglob("*.pyi")}
    keys = sorted(set(py_files) | set(pyi_files), key=lambda key: (len(key.parts), key.parts))
    modules: list[ModuleDoc] = []

    for key in keys:
        api_path = pyi_files.get(key) or py_files.get(key)
        impl_path = py_files.get(key)
        if api_path is None or not _is_public_module(package_root, api_path):
            continue
        if api_path.name == "_native.pyi":
            continue
        source_path = (
            package_root / package.public_api_source
            if key == Path("__init__") and package.public_api_source is not None
            else api_path
        )
        api_tree = _parse(source_path)
        impl_tree = _parse(impl_path) if impl_path and impl_path != source_path else None
        modules.append(_collect_module(package, package_root, api_path, source_path, api_tree, impl_tree))

    return modules


def _slug_part(value: str) -> str:
    return value.replace("_", "-")


def _module_parts(module: ModuleDoc) -> list[str]:
    return module.name.split(".")


def _nav_parts(module: ModuleDoc) -> list[str]:
    parts = _module_parts(module)
    if len(parts) == 1:
        return [_slug_part(parts[0])]
    return [_slug_part(part) for part in parts[1:]]


def _leaf_title(module: ModuleDoc) -> str:
    return _module_parts(module)[-1]


def _package_slug(package: Package) -> str:
    return package.distribution_name


def _module_has_children(module: ModuleDoc, modules: list[ModuleDoc]) -> bool:
    parts = _module_parts(module)
    return any(
        other_parts[: len(parts)] == parts and len(other_parts) > len(parts)
        for other_parts in (_module_parts(other) for other in modules)
    )


def _module_url(module: ModuleDoc) -> str:
    return f"{BASE_URL}/{'/'.join((_package_slug(module.package), *_nav_parts(module)))}"


def _module_output_path(output_dir: Path, module: ModuleDoc, modules: list[ModuleDoc]) -> Path:
    nav_parts = _nav_parts(module)
    if len(_module_parts(module)) == 1:
        return output_dir / _package_slug(module.package) / "index.mdx"
    if _module_has_children(module, modules):
        return output_dir.joinpath(_package_slug(module.package), *nav_parts, "index.mdx")
    return output_dir.joinpath(_package_slug(module.package), *nav_parts[:-1], f"{nav_parts[-1]}.mdx")


def _sibling_sort_key(module: ModuleDoc) -> tuple[int, str]:
    return (0 if len(_module_parts(module)) == 1 else 1, _leaf_title(module))


def _sibling_positions(modules: list[ModuleDoc]) -> dict[str, int]:
    by_parent: dict[tuple[str, ...], list[ModuleDoc]] = {}
    for module in modules:
        nav_parts = _nav_parts(module)
        parent = (
            (_package_slug(module.package), *nav_parts[:-1])
            if len(_module_parts(module)) > 1
            else (_package_slug(module.package),)
        )
        by_parent.setdefault(parent, []).append(module)

    positions: dict[str, int] = {}
    for siblings in by_parent.values():
        siblings.sort(key=_sibling_sort_key)
        for position, module in enumerate(siblings, start=1):
            positions[module.name] = position
    return positions


def _write_index(output_dir: Path) -> None:
    lines = [
        frontmatter("Python Library Reference", "Generated Python API reference for NeMo Relay packages.", 1),
        "## Packages\n\n",
    ]
    for package in PACKAGES:
        lines.append(f"- [`{package.distribution_name}`]({BASE_URL}/{_package_slug(package)}): {package.description}\n")
    (output_dir / "index.mdx").write_text("".join(lines), encoding="utf-8")


def _write_module(output_dir: Path, module: ModuleDoc, position: int, modules: list[ModuleDoc]) -> None:
    sidebar_title = module.package.distribution_name if len(_module_parts(module)) == 1 else _leaf_title(module)
    lines = [
        frontmatter(module.title, module.summary, position, sidebar_title=sidebar_title),
        f"Generated from `{display_path(module.source)}`.\n\n",
        f"Module `{module.name}`.\n\n",
    ]
    if module.summary:
        lines.append(f"{module.summary}\n\n")

    if module.classes:
        lines.append("## Classes\n\n")
        for class_doc in module.classes:
            bases = f"({', '.join(class_doc.bases)})" if class_doc.bases else ""
            lines.append(f"### `{class_doc.name}{bases}`\n\n")
            if class_doc.summary:
                lines.append(f"{class_doc.summary}\n\n")
            if class_doc.methods:
                lines.append("#### Methods\n\n")
                for method in class_doc.methods:
                    lines.append(f"##### `{method.name}`\n\n")
                    lines.append(f"```python\n{method.signature}\n```\n\n")
                    if method.summary:
                        lines.append(f"{method.summary}\n\n")

    if module.functions:
        lines.append("## Functions\n\n")
        for function in module.functions:
            lines.append(f"### `{function.name}`\n\n")
            lines.append(f"```python\n{function.signature}\n```\n\n")
            if function.summary:
                lines.append(f"{function.summary}\n\n")

    if module.aliases:
        lines.append("## Type Aliases And Constants\n\n")
        for alias in module.aliases:
            lines.append(f"- `{alias}`\n")
        lines.append("\n")

    if module.reexports:
        lines.append("## Re-exports\n\n")
        for name in module.reexports:
            lines.append(f"- `{name}`\n")
        lines.append("\n")

    output_path = _module_output_path(output_dir, module, modules)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(lines), encoding="utf-8")


def generate(repo_root: Path, output_dir: Path) -> int:
    modules: list[ModuleDoc] = []
    for package in PACKAGES:
        package_root = repo_root / package.source_root
        if not package_root.is_dir():
            raise SystemExit(f"package root not found: {package_root}")
        modules.extend(_discover_modules(package, package_root))
    reset_output_dir(output_dir)
    positions = _sibling_positions(modules)
    _write_index(output_dir)
    for module in modules:
        _write_module(output_dir, module, positions[module.name], modules)
    return len(modules)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/reference/api/python-library-reference"),
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    count = generate(Path.cwd(), output_dir)
    print(f"Generated Python library reference for {count} module(s) in {output_dir}")


if __name__ == "__main__":
    main()
