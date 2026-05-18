# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate Fern-compatible Rust library reference pages from rustdoc HTML."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urldefrag

from bs4 import BeautifulSoup
from bs4.element import NavigableString, PageElement, Tag
from reference_common import escape_mdx_text, frontmatter, reset_output_dir

CRATES = (
    ("nemo-flow", "nemo_flow", "Core Rust runtime APIs for NeMo Flow."),
    ("nemo-flow-adaptive", "nemo_flow_adaptive", "Adaptive runtime primitives and plugin components."),
    ("nemo-flow-ffi", "nemo_flow_ffi", "C-compatible FFI surface for NeMo Flow."),
)
BASE_URL = "/reference/api/rust-library-reference"
GENERATED_BY = "Generated from `cargo doc --no-deps -p nemo-flow -p nemo-flow-adaptive -p nemo-flow-ffi`."
TRANSLATION_TABLE = str.maketrans(
    {
        "\xa0": " ",
        "\u200b": "",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
        "\u2192": "->",
        "\u24d8": "",
        "\u26a0": "",
        "\u2139": "",
        "\ufe0f": "",
        "\u00a7": "",
        "\U0001f52c": "",
    }
)
FERN_COMPONENT_TAGS = (
    "Tabs",
    "Tab",
    "Cards",
    "Card",
    "Accordion",
    "Note",
    "Warning",
    "Tip",
    "Info",
    "Error",
)
FERN_COMPONENT_TAG_PATTERN = re.compile(
    rf"<(/?)({'|'.join(re.escape(tag) for tag in FERN_COMPONENT_TAGS)})(?=[\s>/]|$)"
)


@dataclass(frozen=True, slots=True)
class Page:
    html_path: Path
    output_path: Path
    url: str
    crate_name: str
    crate_dir_name: str


def _ascii(value: str) -> str:
    value = value.translate(TRANSLATION_TABLE)
    return value.encode("ascii", "xmlcharrefreplace").decode("ascii")


def _escape_text(value: str) -> str:
    return escape_mdx_text(value, normalize=_ascii)


def _clean_text(value: str) -> str:
    return re.sub(r"[ \t\r\f\v]+", " ", _ascii(value)).replace("\n ", "\n").strip()


def _clean_code(value: str) -> str:
    lines = _ascii(value).splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(line.rstrip() for line in lines)


def _mdx_safe_code(value: str) -> str:
    """Avoid accidental Fern component tags inside generated Rust signatures."""
    return FERN_COMPONENT_TAG_PATTERN.sub(
        lambda match: f"< {'/' if match.group(1) else ''}{match.group(2)}",
        value,
    )


def _inline_code(value: str) -> str:
    code = _mdx_safe_code(value).replace("`", "\\`")
    return f"`{code}`"


def _rust_fence(value: str) -> str:
    code = _mdx_safe_code(value)
    return f"```rust\n{code}\n```\n\n" if code else ""


def _slug_part(value: str) -> str:
    value = value.replace("!", "-bang")
    value = re.sub(r"[^A-Za-z0-9_-]+", "-", value)
    value = value.replace("_", "-").strip("-").lower()
    return value or "item"


def _crate_slug(crate_name: str) -> str:
    return crate_name.replace("_", "-")


def _run_cargo_doc(repo_root: Path) -> None:
    env = os.environ.copy()
    existing = env.get("RUSTDOCFLAGS", "")
    cap_lints = "--cap-lints allow"
    env["RUSTDOCFLAGS"] = f"{existing} {cap_lints}".strip()
    subprocess.run(
        [
            "cargo",
            "doc",
            "--no-deps",
            "-p",
            "nemo-flow",
            "-p",
            "nemo-flow-adaptive",
            "-p",
            "nemo-flow-ffi",
        ],
        cwd=repo_root,
        env=env,
        check=True,
    )


def _output_relative(crate_name: str, crate_dir: Path, html_path: Path) -> Path:
    crate_slug = _crate_slug(crate_name)
    rel = html_path.relative_to(crate_dir)
    if rel.name == "index.html":
        return Path(crate_slug, *rel.parent.parts, "index.mdx")
    stem = rel.name.removesuffix(".html")
    return Path(crate_slug, *rel.parent.parts, f"{_slug_part(stem)}.mdx")


def _page_url(output_rel: Path) -> str:
    without_suffix = output_rel.with_suffix("")
    parts = list(without_suffix.parts)
    if parts[-1] == "index":
        parts.pop()
    return f"{BASE_URL}/{'/'.join(parts)}"


def _needs_explicit_slug(page: Page) -> bool:
    """Fern drops underscores from auto-discovered folder URLs."""
    return "_" in page.url


def _discover_pages(doc_root: Path, output_dir: Path) -> dict[Path, Page]:
    pages: dict[Path, Page] = {}
    for crate_name, crate_dir_name, _description in CRATES:
        crate_dir = doc_root / crate_dir_name
        if not crate_dir.is_dir():
            raise SystemExit(f"rustdoc crate output not found: {crate_dir}")
        for html_path in sorted(crate_dir.rglob("*.html")):
            if html_path.name == "all.html":
                continue
            if 'id="main-content"' not in html_path.read_text(encoding="utf-8", errors="ignore"):
                continue
            output_rel = _output_relative(crate_name, crate_dir, html_path)
            pages[html_path.resolve()] = Page(
                html_path=html_path.resolve(),
                output_path=output_dir / output_rel,
                url=_page_url(output_rel),
                crate_name=crate_name,
                crate_dir_name=crate_dir_name,
            )
    return pages


def _resolve_href(page: Page, href: str, pages_by_html: dict[Path, Page]) -> str | None:
    if not href or href.startswith("#"):
        return None
    if href.startswith(("http://", "https://")):
        return href
    href_no_fragment, _fragment = urldefrag(href)
    if href_no_fragment.startswith("../src/") or "/src/" in href_no_fragment:
        return None
    target = (page.html_path.parent / href_no_fragment).resolve()
    target_page = pages_by_html.get(target)
    if target_page is None:
        return None
    return target_page.url


def _tag_classes(node: Tag) -> set[str]:
    classes = node.get("class")
    if classes is None:
        return set()
    if isinstance(classes, str):
        return set(classes.split())
    return {str(class_name) for class_name in classes}


def _tag_href(node: Tag) -> str:
    href = node.get("href")
    return href if isinstance(href, str) else ""


def _inline_markdown(node: PageElement, page: Page, pages_by_html: dict[Path, Page]) -> str:
    if isinstance(node, NavigableString):
        return _escape_text(str(node))
    if not isinstance(node, Tag):
        return ""

    classes = _tag_classes(node)
    if classes & {"anchor", "doc-anchor", "src"} or node.name in {"script", "style", "wbr", "button"}:
        return ""
    if node.name == "br":
        return "\n"
    if node.name == "code":
        return _inline_code(_clean_text(node.get_text("", strip=False)))
    if node.name == "a":
        label = "".join(_inline_markdown(child, page, pages_by_html) for child in node.children).strip()
        if not label or label == "Source":
            return label
        target = _resolve_href(page, _tag_href(node), pages_by_html)
        if target is None:
            return label
        return f"[{label}]({target})"
    if node.name == "strong":
        content = "".join(_inline_markdown(child, page, pages_by_html) for child in node.children).strip()
        return f"**{content}**" if content else ""
    if node.name == "em":
        content = "".join(_inline_markdown(child, page, pages_by_html) for child in node.children).strip()
        return f"*{content}*" if content else ""
    return "".join(_inline_markdown(child, page, pages_by_html) for child in node.children)


def _plain_code(node: Tag) -> str:
    return _clean_code(node.get_text("", strip=False))


def _heading_text(node: Tag) -> str:
    for removable in node.select(".anchor, .doc-anchor, .src, button"):
        removable.decompose()
    text = _clean_text(node.get_text(" ", strip=True))
    return re.sub(r"_\s+", "_", text)


def _block_markdown(node: PageElement, page: Page, pages_by_html: dict[Path, Page], depth: int = 0) -> str:
    if isinstance(node, NavigableString):
        return ""
    if not isinstance(node, Tag):
        return ""

    classes = _tag_classes(node)
    if node.name in {"script", "style", "rustdoc-toolbar"} or classes & {"anchor", "doc-anchor", "src"}:
        return ""
    if node.name == "div" and "main-heading" in classes:
        return ""
    if node.name == "span" and "item-info" in classes:
        return ""
    if node.get("id") in {"synthetic-implementations-list", "blanket-implementations-list"}:
        return ""

    if node.name in {"h2", "h3", "h4", "h5"} and "code-header" in classes:
        return _rust_fence(_plain_code(node))

    if node.name in {"h2", "h3", "h4", "h5"}:
        level = {"h2": "##", "h3": "###", "h4": "####", "h5": "#####"}[node.name]
        text = _heading_text(node)
        if text in {"Auto Trait Implementations", "Blanket Implementations"}:
            return ""
        return f"{level} {text}\n\n" if text else ""

    if node.name == "span" and "section-header" in classes:
        text = _plain_code(node)
        return f"### {_inline_code(text)}\n\n" if text else ""

    if node.name == "p":
        text = re.sub(r"\s+", " ", _inline_markdown(node, page, pages_by_html)).strip()
        return f"{text}\n\n" if text else ""

    if node.name == "pre":
        return _rust_fence(_plain_code(node))

    if node.name == "ul":
        lines: list[str] = []
        for li in node.find_all("li", recursive=False):
            text = re.sub(r"\s+", " ", _inline_markdown(li, page, pages_by_html)).strip()
            if text:
                lines.append(f"{'  ' * depth}- {text}")
        return "\n".join(lines) + ("\n\n" if lines else "")

    if node.name == "ol":
        lines = []
        for index, li in enumerate(node.find_all("li", recursive=False), start=1):
            text = re.sub(r"\s+", " ", _inline_markdown(li, page, pages_by_html)).strip()
            if text:
                lines.append(f"{'  ' * depth}{index}. {text}")
        return "\n".join(lines) + ("\n\n" if lines else "")

    if node.name == "dl":
        parts: list[str] = []
        children = [child for child in node.children if isinstance(child, Tag)]
        index = 0
        while index < len(children):
            if children[index].name != "dt":
                index += 1
                continue
            term = re.sub(r"\s+", " ", _inline_markdown(children[index], page, pages_by_html)).strip()
            description = ""
            if index + 1 < len(children) and children[index + 1].name == "dd":
                description = re.sub(
                    r"\s+",
                    " ",
                    _inline_markdown(children[index + 1], page, pages_by_html),
                ).strip()
                index += 1
            if term:
                line = f"- {term}"
                if description:
                    line += f": {description}"
                parts.append(line)
            index += 1
        return "\n".join(parts) + ("\n\n" if parts else "")

    if node.name == "table":
        rows: list[str] = []
        for row in node.find_all("tr"):
            cells = [
                re.sub(r"\s+", " ", _inline_markdown(cell, page, pages_by_html)).strip()
                for cell in row.find_all(["th", "td"], recursive=False)
            ]
            if cells:
                rows.append("- " + " | ".join(cell for cell in cells if cell))
        return "\n".join(rows) + ("\n\n" if rows else "")

    if node.name == "details":
        pieces: list[str] = []
        summary = node.find("summary", recursive=False)
        if isinstance(summary, Tag) and "hideme" not in _tag_classes(summary):
            pieces.append(_block_markdown(summary, page, pages_by_html, depth))
        for child in node.children:
            if child is summary:
                continue
            pieces.append(_block_markdown(child, page, pages_by_html, depth))
        return "".join(pieces)

    pieces = [_block_markdown(child, page, pages_by_html, depth) for child in node.children]
    return "".join(pieces)


def _remove_noisy_sections(content: Tag) -> None:
    for selector in [
        "script",
        "style",
        "rustdoc-toolbar",
        ".src",
        "button",
        ".item-info",
        "#notable-traits-data",
    ]:
        for element in content.select(selector):
            element.decompose()

    for section_id in ("synthetic-implementations", "blanket-implementations"):
        heading = content.find(id=section_id)
        if heading is None:
            continue
        for sibling in list(heading.find_next_siblings()):
            if isinstance(sibling, Tag) and sibling.name == "h2":
                break
            sibling.decompose()
        heading.decompose()


def _page_title(soup: BeautifulSoup, page: Page) -> str:
    h1 = soup.select_one("#main-content .main-heading h1")
    if h1 is not None:
        title = _heading_text(h1)
        title = re.sub(
            r"^(Crate|Module|Function|Struct|Enum|Trait|Type Alias|Constant|Macro)(?=\S)",
            r"\1 ",
            title,
        )
        title = re.sub(r"\s+", " ", title)
        if title.startswith("Crate "):
            return page.crate_name
        return title
    return page.crate_name


def _remove_duplicate_top_level_sections(markdown: str) -> str:
    lines = markdown.splitlines()
    seen: set[str] = set()
    kept: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("## ") and not line.startswith("### "):
            if line in seen:
                index += 1
                while index < len(lines) and not (
                    lines[index].startswith("## ") and not lines[index].startswith("### ")
                ):
                    index += 1
                continue
            seen.add(line)
        kept.append(line)
        index += 1
    return "\n".join(kept).strip()


def _sidebar_title(soup: BeautifulSoup, page: Page) -> str:
    topbar = soup.select_one("rustdoc-topbar h2")
    if topbar is not None:
        text = _clean_text(topbar.get_text("", strip=True))
        if text:
            if text.startswith("Crate "):
                return page.crate_name
            if text.startswith("Module "):
                return text.removeprefix("Module ")
            return text
    if page.output_path.name == "index.mdx":
        return page.output_path.parent.name
    return page.output_path.stem


def _description(soup: BeautifulSoup) -> str:
    meta = soup.find("meta", attrs={"name": "description"})
    if isinstance(meta, Tag):
        value = meta.get("content", "")
        if isinstance(value, str):
            return _clean_text(value)
    first_paragraph = soup.select_one("#main-content .top-doc .docblock p")
    if first_paragraph is not None:
        return _clean_text(first_paragraph.get_text(" ", strip=True))
    return ""


def _render_page(page: Page, pages_by_html: dict[Path, Page], position: int) -> str:
    soup = BeautifulSoup(page.html_path.read_text(encoding="utf-8"), "html.parser")
    content = soup.select_one("#main-content")
    if content is None:
        raise SystemExit(f"rustdoc main content not found: {page.html_path}")
    _remove_noisy_sections(content)

    body = _remove_duplicate_top_level_sections(_block_markdown(content, page, pages_by_html).strip())
    description = _description(soup)
    lines = [
        frontmatter(
            _page_title(soup, page),
            description,
            position,
            sidebar_title=_sidebar_title(soup, page),
            slug=page.url if _needs_explicit_slug(page) else None,
            normalize=_ascii,
        ),
        f"{GENERATED_BY}\n\n",
    ]
    if body:
        lines.append(body + "\n")
    return "".join(lines)


def _item_order(index_html: Path, pages_by_html: dict[Path, Page]) -> dict[Path, int]:
    soup = BeautifulSoup(index_html.read_text(encoding="utf-8"), "html.parser")
    positions: dict[Path, int] = {}
    position = 1
    for link in soup.select("#main-content dl.item-table dt a[href]"):
        href = link.get("href", "")
        if not isinstance(href, str):
            continue
        target_path = (index_html.resolve().parent / urldefrag(href)[0]).resolve()
        if target_path not in pages_by_html:
            continue
        positions.setdefault(target_path, position)
        position += 1
    return positions


def _positions(pages_by_html: dict[Path, Page], doc_root: Path) -> dict[Path, int]:
    positions: dict[Path, int] = {}
    for position, (_crate_name, crate_dir_name, _description) in enumerate(CRATES, start=1):
        crate_index = (doc_root / crate_dir_name / "index.html").resolve()
        positions[crate_index] = position

    for html_path, page in pages_by_html.items():
        if html_path.name != "index.html":
            continue
        for target, position in _item_order(html_path, pages_by_html).items():
            positions[target] = position

    return positions


def _write_index(output_dir: Path) -> None:
    lines = [
        frontmatter(
            "Rust Library Reference",
            "Generated Rust API reference for NeMo Flow crates.",
            3,
            normalize=_ascii,
        ),
        f"{GENERATED_BY}\n\n",
        "## Crates\n\n",
    ]
    for crate_name, _crate_dir_name, description in CRATES:
        lines.append(f"- [{crate_name}]({BASE_URL}/{_crate_slug(crate_name)}): {_escape_text(description)}\n")
    (output_dir / "index.mdx").write_text("".join(lines), encoding="utf-8")


def generate(repo_root: Path, doc_root: Path, output_dir: Path, *, run_cargo_doc: bool) -> int:
    if run_cargo_doc:
        _run_cargo_doc(repo_root)
    pages_by_html = _discover_pages(doc_root, output_dir)
    positions = _positions(pages_by_html, doc_root)

    reset_output_dir(output_dir)
    _write_index(output_dir)

    for index, page in enumerate(sorted(pages_by_html.values(), key=lambda page: page.output_path.as_posix()), start=1):
        page.output_path.parent.mkdir(parents=True, exist_ok=True)
        position = positions.get(page.html_path, index)
        page.output_path.write_text(_render_page(page, pages_by_html, position), encoding="utf-8")

    return len(pages_by_html)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-root", type=Path, default=Path("target/doc"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/reference/api/rust-library-reference"),
    )
    parser.add_argument("--skip-cargo-doc", action="store_true")
    args = parser.parse_args()

    repo_root = Path.cwd()
    doc_root = args.doc_root.resolve()
    output_dir = args.output_dir.resolve()
    count = generate(repo_root, doc_root, output_dir, run_cargo_doc=not args.skip_cargo_doc)
    print(f"Generated Rust library reference for {count} rustdoc page(s) in {output_dir}")


if __name__ == "__main__":
    main()
