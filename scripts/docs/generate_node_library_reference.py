# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate Fern-compatible Node.js library reference pages from declaration files."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from reference_common import display_path, escape_mdx_text, frontmatter, reset_output_dir

EXPORT_RE = re.compile(
    r"^export\s+"
    r"(?:(?P<declare>declare)\s+)?"
    r"(?:(?P<const_enum>const)\s+)?"
    r"(?P<kind>enum|interface|class|function|type|const)\s+"
    r"(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)"
)
REEXPORT_RE = re.compile(r"^export\s+\{(?P<names>.*?)\};?$")
INTERNAL_NAME_PREFIXES = ("__test",)
SECTION_ORDER = ("interface", "class", "enum", "function", "type", "const")
SECTION_TITLES = {
    "interface": "Interfaces",
    "class": "Classes",
    "enum": "Enums",
    "function": "Functions",
    "type": "Type Aliases",
    "const": "Constants",
}
KIND_LABELS = {
    "interface": "interface",
    "class": "class",
    "enum": "enum",
    "function": "function",
    "type": "type alias",
    "const": "constant",
}
MODULE_TITLES = {
    "nemo-flow-node": "Runtime",
    "nemo-flow-node/typed": "Typed Helpers",
    "nemo-flow-node/plugin": "Plugin Helpers",
    "nemo-flow-node/adaptive": "Adaptive Helpers",
    "nemo-flow-node/observability": "Observability Helpers",
}
MODULE_DESCRIPTIONS = {
    "nemo-flow-node": "Main runtime lifecycle, scope, middleware, subscriber, and exporter APIs.",
    "nemo-flow-node/typed": "Typed wrappers and codec-aware execution helpers.",
    "nemo-flow-node/plugin": "Plugin configuration, validation, activation, and registration helpers.",
    "nemo-flow-node/adaptive": "Adaptive plugin configuration helpers.",
    "nemo-flow-node/observability": "Observability plugin configuration helpers.",
}
BASE_URL = "/reference/api/nodejs-library-reference"


@dataclass(slots=True)
class ParamDoc:
    name: str
    text: str


@dataclass(slots=True)
class JsDoc:
    body: list[str] = field(default_factory=list)
    params: list[ParamDoc] = field(default_factory=list)
    returns: str = ""
    remarks: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        return self.body[0] if self.body else ""


@dataclass(slots=True)
class ApiItem:
    name: str
    kind: str
    declaration: str
    doc: JsDoc


@dataclass(slots=True)
class ModuleDoc:
    import_path: str
    title: str
    source: Path
    description: str
    items: list[ApiItem] = field(default_factory=list)
    reexports: list[str] = field(default_factory=list)


def _escape_text(value: str) -> str:
    value = value.replace("``", "`")
    value = value.replace("→", "->")
    return escape_mdx_text(value, preserve_ascii_arrows=True)


def _collapse_words(value: str) -> str:
    return " ".join(value.split())


def _paragraphs(lines: list[str]) -> list[str]:
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if not line.strip():
            if current:
                paragraphs.append(_escape_text(_collapse_words(" ".join(current))))
                current = []
            continue
        current.append(line.strip())
    if current:
        paragraphs.append(_escape_text(_collapse_words(" ".join(current))))
    return paragraphs


def _clean_jsdoc_line(line: str) -> str:
    stripped = line.strip()
    if stripped.startswith("/**"):
        stripped = stripped[3:]
    if stripped.endswith("*/"):
        stripped = stripped[:-2]
    stripped = stripped.strip()
    if stripped.startswith("*"):
        stripped = stripped[1:].lstrip()
    if stripped.startswith('r"'):
        stripped = stripped[2:].lstrip()
    return stripped.rstrip()


def _parse_jsdoc(raw_lines: list[str]) -> JsDoc:
    body_lines: list[str] = []
    params: list[ParamDoc] = []
    returns_lines: list[str] = []
    remarks_lines: list[str] = []
    current_tag: tuple[str, int | None] | None = None

    for raw_line in raw_lines:
        line = _clean_jsdoc_line(raw_line)
        if not line:
            if current_tag is None:
                body_lines.append("")
            continue

        param_match = re.match(r"@param\s+(?P<name>[^\s-]+)\s*-?\s*(?P<text>.*)$", line)
        if param_match is not None:
            params.append(
                ParamDoc(
                    name=param_match.group("name"),
                    text=_escape_text(_collapse_words(param_match.group("text"))),
                )
            )
            current_tag = ("param", len(params) - 1)
            continue

        returns_match = re.match(r"@returns?\s*-?\s*(?P<text>.*)$", line)
        if returns_match is not None:
            returns_lines.append(returns_match.group("text"))
            current_tag = ("returns", None)
            continue

        remarks_match = re.match(r"@remarks\s*-?\s*(?P<text>.*)$", line)
        if remarks_match is not None:
            remarks_lines.append(remarks_match.group("text"))
            current_tag = ("remarks", None)
            continue

        if line.startswith("@example"):
            current_tag = None
            continue

        if current_tag is None:
            body_lines.append(line)
        elif current_tag[0] == "param" and current_tag[1] is not None:
            param = params[current_tag[1]]
            param.text = _escape_text(_collapse_words(f"{param.text} {line}"))
        elif current_tag[0] == "returns":
            returns_lines.append(line)
        elif current_tag[0] == "remarks":
            remarks_lines.append(line)

    return JsDoc(
        body=_paragraphs(body_lines),
        params=params,
        returns=_escape_text(_collapse_words(" ".join(returns_lines))) if returns_lines else "",
        remarks=_paragraphs(remarks_lines),
    )


def _collect_jsdoc(lines: list[str], start: int) -> tuple[JsDoc, int]:
    raw_lines = [lines[start]]
    if "*/" in lines[start]:
        return _parse_jsdoc(raw_lines), start + 1

    index = start + 1
    while index < len(lines):
        raw_lines.append(lines[index])
        if "*/" in lines[index]:
            break
        index += 1
    return _parse_jsdoc(raw_lines), index + 1


def _brace_delta(line: str) -> int:
    return line.count("{") - line.count("}")


def _paren_delta(line: str) -> int:
    return line.count("(") - line.count(")")


def _bracket_delta(line: str) -> int:
    return line.count("[") - line.count("]")


def _collect_braced_declaration(lines: list[str], start: int) -> tuple[str, int]:
    collected: list[str] = []
    depth = 0
    seen_open = False
    index = start
    while index < len(lines):
        line = lines[index]
        collected.append(line.rstrip())
        if "{" in line:
            seen_open = True
        depth += _brace_delta(line)
        index += 1
        if seen_open and depth <= 0:
            break
    return "\n".join(collected).rstrip(), index


def _collect_statement_declaration(lines: list[str], start: int, kind: str) -> tuple[str, int]:
    collected: list[str] = []
    paren_depth = 0
    brace_depth = 0
    bracket_depth = 0
    index = start
    while index < len(lines):
        line = lines[index]
        collected.append(line.rstrip())
        paren_depth += _paren_delta(line)
        brace_depth += _brace_delta(line)
        bracket_depth += _bracket_delta(line)
        stripped = line.strip()
        index += 1

        if stripped.endswith(";"):
            break
        if kind == "function" and paren_depth <= 0 and brace_depth <= 0 and bracket_depth <= 0:
            break

    return "\n".join(collected).rstrip(), index


def _collect_declaration(lines: list[str], start: int, kind: str) -> tuple[str, int]:
    if kind in {"interface", "class", "enum"}:
        return _collect_braced_declaration(lines, start)
    return _collect_statement_declaration(lines, start, kind)


def _is_public_name(name: str) -> bool:
    return not name.startswith("_") and not any(name.startswith(prefix) for prefix in INTERNAL_NAME_PREFIXES)


def _parse_reexport_names(declaration: str) -> list[str]:
    match = REEXPORT_RE.match(" ".join(declaration.split()))
    if match is None:
        return []

    names: list[str] = []
    for item in match.group("names").split(","):
        name = item.strip()
        if not name:
            continue
        name = re.sub(r"^type\s+", "", name)
        name = name.split(" as ", 1)[-1].strip()
        if _is_public_name(name):
            names.append(name)
    return names


def _parse_declaration_file(path: Path, import_path: str) -> ModuleDoc:
    lines = path.read_text(encoding="utf-8").splitlines()
    title = MODULE_TITLES.get(import_path, import_path)
    description = MODULE_DESCRIPTIONS.get(import_path, f"Declarations exported by `{import_path}`.")
    module = ModuleDoc(import_path=import_path, title=title, source=path, description=description)
    pending_doc = JsDoc()
    module_doc: JsDoc | None = None
    index = 0

    while index < len(lines):
        stripped = lines[index].strip()

        if stripped.startswith("/**"):
            pending_doc, index = _collect_jsdoc(lines, index)
            continue

        if not stripped or stripped.startswith("//") or stripped.startswith("import "):
            if pending_doc.summary and module_doc is None and not module.items:
                module_doc = pending_doc
            pending_doc = JsDoc()
            index += 1
            continue

        if stripped.startswith("export {"):
            declaration, index = _collect_statement_declaration(lines, index, "reexport")
            module.reexports.extend(_parse_reexport_names(declaration))
            pending_doc = JsDoc()
            continue

        match = EXPORT_RE.match(stripped)
        if match is None:
            pending_doc = JsDoc()
            index += 1
            continue

        kind = match.group("kind")
        if kind == "enum" and match.group("const_enum"):
            kind = "enum"
        name = match.group("name")
        declaration, index = _collect_declaration(lines, index, kind)
        if _is_public_name(name):
            module.items.append(ApiItem(name=name, kind=kind, declaration=declaration, doc=pending_doc))
        pending_doc = JsDoc()

    if module_doc is not None and module_doc.summary:
        module.description = module_doc.summary
    module.reexports = sorted(set(module.reexports))
    return module


def _package_exports(package_json: Path) -> list[tuple[str, Path]]:
    data = json.loads(package_json.read_text(encoding="utf-8"))
    package_name = data["name"]
    exports = data["exports"]
    modules: list[tuple[str, Path]] = []

    for export_path, export_config in exports.items():
        if not isinstance(export_config, dict) or "types" not in export_config:
            continue
        import_path = package_name if export_path == "." else f"{package_name}/{export_path.removeprefix('./')}"
        modules.append((import_path, package_json.parent / export_config["types"]))

    return modules


def _slug(import_path: str) -> str:
    return import_path.replace("@", "").replace("/", "-").replace("_", "-")


def _write_index(output_dir: Path, modules: list[ModuleDoc]) -> None:
    lines = [
        frontmatter(
            "Node.js Library Reference",
            "Generated Node.js API reference for the nemo-flow-node package.",
            2,
        ),
        "Generated from the local `crates/node` package exports and TypeScript declaration files.\n\n",
        "## Entry Points\n\n",
    ]

    for module in modules:
        lines.append(f"- [{module.import_path}]({BASE_URL}/{_slug(module.import_path)})")
        if module.description:
            lines.append(f": {_escape_text(module.description)}")
        lines.append("\n")

    (output_dir / "index.mdx").write_text("".join(lines), encoding="utf-8")


def _write_doc(lines: list[str], doc: JsDoc) -> None:
    for paragraph in doc.body:
        lines.append(f"{paragraph}\n\n")
    if doc.params:
        lines.append("**Parameters**\n\n")
        for param in doc.params:
            lines.append(f"- `{param.name}`: {param.text}\n")
        lines.append("\n")
    if doc.returns:
        lines.append("**Returns**\n\n")
        lines.append(f"{doc.returns}\n\n")
    if doc.remarks:
        lines.append("**Remarks**\n\n")
        for remark in doc.remarks:
            lines.append(f"{remark}\n\n")


def _write_item(lines: list[str], item: ApiItem, *, disambiguate: bool = False) -> None:
    suffix = f" {KIND_LABELS[item.kind]}" if disambiguate else ""
    lines.append(f"### `{item.name}`{suffix}\n\n")
    _write_doc(lines, item.doc)
    lines.append(f"```ts\n{item.declaration}\n```\n\n")


def _write_module(output_dir: Path, module: ModuleDoc, position: int) -> None:
    lines = [
        frontmatter(module.title, module.description, position),
        f"Generated from `{display_path(module.source, resolve=True)}`.\n\n",
        f"Import from `{module.import_path}`.\n\n",
    ]
    if module.description:
        lines.append(f"{_escape_text(module.description)}\n\n")

    name_counts: dict[str, int] = {}
    for item in module.items:
        name_counts[item.name] = name_counts.get(item.name, 0) + 1

    for kind in SECTION_ORDER:
        items = [item for item in module.items if item.kind == kind]
        if not items:
            continue
        lines.append(f"## {SECTION_TITLES[kind]}\n\n")
        for item in items:
            _write_item(lines, item, disambiguate=name_counts[item.name] > 1)

    if module.reexports:
        lines.append("## Re-exports\n\n")
        for name in module.reexports:
            lines.append(f"- `{name}`\n")
        lines.append("\n")

    (output_dir / f"{_slug(module.import_path)}.mdx").write_text("".join(lines), encoding="utf-8")


def generate(package_root: Path, output_dir: Path) -> int:
    package_json = package_root / "package.json"
    if not package_json.is_file():
        raise SystemExit(f"package.json not found: {package_json}")

    modules = [
        _parse_declaration_file(types_path, import_path) for import_path, types_path in _package_exports(package_json)
    ]

    reset_output_dir(output_dir)
    _write_index(output_dir, modules)
    for position, module in enumerate(modules, start=2):
        _write_module(output_dir, module, position)
    return len(modules)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=Path("crates/node"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/reference/api/nodejs-library-reference"),
    )
    args = parser.parse_args()

    package_root = args.package_root.resolve()
    output_dir = args.output_dir.resolve()
    if not package_root.is_dir():
        raise SystemExit(f"package root not found: {package_root}")

    count = generate(package_root, output_dir)
    print(f"Generated Node.js library reference for {count} entry point(s) in {output_dir}")


if __name__ == "__main__":
    main()
