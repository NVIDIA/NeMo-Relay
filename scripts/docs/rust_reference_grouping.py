# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Group rendered rustdoc items without changing their API coverage."""

from __future__ import annotations

import posixpath
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
from html import unescape
from pathlib import Path
from typing import Protocol
from urllib.parse import unquote, urldefrag

import yaml
from reference_common import escape_mdx_text, frontmatter

MAX_PAGE_CHARACTERS = 50_000
ANCHOR = re.compile(r'<a id="([^"]+)"></a>')
INTERNAL_LINK = re.compile(r"/reference/api/rust-library-reference/[a-z0-9_/-]+(?:#[^\s\"<>\\)]+)?")


class SourcePage(Protocol):
    @property
    def html_path(self) -> Path: ...

    @property
    def output_path(self) -> Path: ...

    @property
    def url(self) -> str: ...


@dataclass
class Item:
    source: SourcePage
    title: str
    name: str
    description: str
    position: int
    body: str

    @property
    def anchor(self) -> str:
        return self.source.url.rsplit("/", 1)[-1] if self.source.html_path.name != "index.html" else "index"


@dataclass
class Section:
    item: Item
    body: str
    part: int = 1


@dataclass
class Group:
    path: Path
    url: str
    title: str
    description: str
    position: int
    sections: list[Section]
    children: list[Group]


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _demote_headings(body: str) -> str:
    lines = []
    fence = ""
    for line in body.splitlines(keepends=True):
        marker = re.match(r"^(`{3,}|~{3,})", line)
        if marker:
            if not fence:
                fence = marker[1]
            elif line.strip() == fence:
                fence = ""
        elif not fence and re.match(r"^#{2,5} ", line):
            line = "#" + line
        lines.append(line)
    return "".join(lines)


def _boundaries(body: str, level: int) -> list[int]:
    """Find safe boundaries; a signature or example is always kept intact."""
    offsets = []
    offset = 0
    anchor_offset = None
    fence = ""
    for line in body.splitlines(keepends=True):
        marker = re.match(r"^(`{3,}|~{3,})", line)
        if marker:
            if not fence:
                fence = marker[1]
            elif line.strip() == fence:
                fence = ""
        elif not fence:
            if ANCHOR.fullmatch(line.strip()):
                if anchor_offset is None:
                    anchor_offset = offset
            elif line.strip():
                if (
                    (level <= 6 and line.startswith("#" * level + " "))
                    or (level == 7 and (offset == 0 or body[:offset].endswith("\n\n")))
                    or (level == 8 and re.match(r"^(?:- |\d+\. )", line))
                ):
                    offsets.append(anchor_offset if anchor_offset is not None else offset)
                anchor_offset = None
        offset += len(line)
    return sorted({0, *offsets, len(body)})


def _split_body(
    body: str, budget: int, level: int = 2, measure: Callable[[str], int] = len, *, coherent: bool = False
) -> list[str]:
    if measure(body) <= budget:
        return [body]
    if level > 8:
        raise ValueError("Rust reference contains an indivisible block above the page character ceiling")
    offsets = _boundaries(body, level)
    pieces = []
    for start, end in zip(offsets, offsets[1:]):
        piece = body[start:end].strip()
        if piece:
            pieces.extend(_split_body(piece, budget, level + 1, measure, coherent=coherent))
    if coherent and level <= 6 and len(offsets) > 2:
        # A large item becomes named sections, rather than consecutive windows
        # that combine whichever unrelated methods happen to fit the budget.
        return pieces
    # Prefer top-level sections, only descending to members/paragraphs when a
    # section itself is too large. This avoids cutting a method from its docs.
    packed: list[str] = []
    for piece in pieces:
        if packed and measure(packed[-1] + "\n\n" + piece) <= budget:
            packed[-1] += "\n\n" + piece
        else:
            packed.append(piece)
    return packed


def _section_markdown(section: Section) -> str:
    item = section.item
    suffix = f"-continued-{section.part}" if section.part > 1 else ""
    label = item.title + (f": {_section_label(section)}" if section.part > 1 else "")
    return f'<a id="{item.anchor}{suffix}"></a>\n\n## {escape_mdx_text(label)}\n\n{section.body.strip()}\n\n'


def _measure(item: Item, body: str) -> int:
    """Estimate signature links in their published, relative form when packing.

    Rendering uses absolute source URLs until destinations are known. Counting
    those temporary URLs wastes substantial space in signature-heavy modules.
    The final rendered page is still checked against the exact ceiling.
    """

    def relative(match: re.Match[str]) -> str:
        quote = match[1]
        base, fragment = urldefrag(match[2])
        name = base.rsplit("/", 1)[-1]
        anchor = name + (f"--{fragment}" if fragment else "")
        if base == item.source.url:
            return f"href={quote}#{anchor}{quote}"
        target = posixpath.dirname(base) + "/reference-types-99#" + anchor
        return f"href={quote}" + posixpath.relpath(target, posixpath.dirname(item.source.url)) + quote

    return len(re.sub(r'href=(\\?")(/reference/api/rust-library-reference/[^"\\]+)\1', relative, body))


def _section_size(section: Section) -> int:
    return _measure(section.item, _section_markdown(section))


def _sections(item: Item, budget: int) -> list[Section]:
    overhead = len(_section_markdown(Section(item, "", 999)))
    try:
        bodies = _split_body(
            item.body,
            budget - overhead,
            measure=lambda body: _measure(item, body),
            coherent=item.source.html_path.name != "index.html",
        )
    except ValueError as error:
        raise ValueError(f"{item.source.html_path}: {error} (content budget: {budget - overhead})") from error
    return [Section(item, body, index) for index, body in enumerate(bodies, 1)]


def _section_label(section: Section) -> str:
    for line in section.body.splitlines():
        if match := re.match(r"^#{3,6} (.+)", line):
            return unescape(match[1].replace("`", ""))
        if line.strip() and not ANCHOR.fullmatch(line.strip()):
            break
    entries = re.findall(r"^- (?:\[([^]]+)\]\([^)]+\)|([^:\n]+))", section.body, flags=re.MULTILINE)
    labels = [(linked or plain).strip() for linked, plain in entries]
    if labels:
        return labels[0] if len(labels) == 1 else f"{labels[0]} to {labels[-1]}"
    return "Overview" if section.part == 1 else "Reference details"


def _name_words(item: Item) -> list[str]:
    name = item.source.html_path.stem.partition(".")[2]
    words = re.findall(r"[A-Z]+[0-9]*(?=[A-Z][a-z]|_|$)|[A-Z]?[a-z]+[0-9]*|[0-9]+", name)
    return words[2:] if words[:2] == ["Nemo", "Relay"] and len(words) > 2 else words


def _family_title(words: list[str]) -> str:
    acronyms = {
        "llm": "LLM",
        "ffi": "FFI",
        "abi": "ABI",
        "sdk": "SDK",
        "grpc": "gRPC",
        "api": "API",
        "atif": "ATIF",
        "atof": "ATOF",
        "http": "HTTP",
        "pii": "PII",
        "s3": "S3",
        "cli": "CLI",
    }
    return " ".join(
        acronyms.get(word.lower(), word.capitalize() if index == 0 else word.lower())
        for index, word in enumerate(words)
    )


def _semantic_batches(
    sections: list[Section], budget: int, title: str, depth: int = 1, *, category: str | None = None
) -> list[tuple[str, list[Section]]]:
    category = category or title
    if sum(_section_size(section) for section in sections) <= budget:
        return [(title, sections)]
    items = {section.item.source.url for section in sections}
    if len(items) == 1:
        # Each section is already a complete heading/member unit of this API.
        # The item's name keeps even a short label such as "Fields" meaningful.
        return [
            (
                f"{'Module overview' if section.item.source.html_path.name == 'index.html' else section.item.name}: "
                f"{_section_label(section)}",
                [section],
            )
            for section in sections
        ]

    families: dict[str, list[Section]] = {}
    for section in sections:
        words = _name_words(section.item)
        family = _family_title(words[:depth]) if depth <= len(words) else section.item.title
        families.setdefault(family, []).append(section)
    if len(families) == 1:
        return _semantic_batches(sections, budget, title, depth + 1, category=category)
    suffix = {
        "Types": "types",
        "Traits": "traits",
        "Functions": "functions",
        "Constants": "constants",
        "Type aliases": "aliases",
        "Request types": "requests",
        "Response types": "responses",
    }.get(category, category.lower())
    return [
        batch
        for family, members in families.items()
        for batch in _semantic_batches(members, budget, f"{family} {suffix}", depth + 1, category=category)
    ]


def _topic(item: Item) -> str:
    kind, _, name = item.source.html_path.stem.partition(".")
    name = name.lower()
    if "stream" in name:
        return "Streaming"
    if "request" in name:
        return "Request processing" if kind == "fn" else "Request types"
    if "response" in name:
        return "Response processing" if kind == "fn" else "Response types"
    if "llmattributes" in name:
        return "LLM attributes and flags"
    if item.source.html_path.parent.name == "llm":
        return "LLM execution"
    return {
        "struct": "Types",
        "enum": "Types",
        "union": "Types",
        "type": "Type aliases",
        "trait": "Traits",
        "fn": "Functions",
        "constant": "Constants",
        "static": "Constants",
        "macro": "Macros",
    }.get(kind, "Other items")


def _plan(
    items: list[Item], budgets: dict[Path, int], overview_budgets: dict[Path, int], default_budget: int
) -> list[Group]:
    modules: dict[Path, list[Item]] = {}
    for item in items:
        modules.setdefault(item.source.output_path.parent, []).append(item)
    groups = []
    for directory, members in modules.items():
        budget = budgets.get(directory, default_budget)
        overview = next(item for item in members if item.source.html_path.name == "index.html")
        members = sorted(
            (item for item in members if item is not overview), key=lambda item: (item.position, item.name)
        )
        if _measure(overview, overview.body) > overview_budgets.get(directory, budget):
            member_urls = {item.source.url for item in members}

            def compact_inventory(match: re.Match[str]) -> str:
                # Keep the complete module inventory in one place. Summaries
                # already appear in each item's full documentation; repeating
                # them here can turn the index itself into a paginated list.
                return match[1] if urldefrag(match[2])[0] in member_urls else match[0]

            body = re.sub(r"^(- \[[^\n]+?\]\(([^)]+)\)): [^\n]*$", compact_inventory, overview.body, flags=re.MULTILINE)
            overview = replace(overview, body=body)
        overview_sections = _sections(overview, overview_budgets.get(directory, budget))
        sections = [section for item in members for section in _sections(item, budget)]
        index = Group(
            directory / "index.mdx",
            overview.source.url,
            overview.title,
            overview.description,
            overview.position,
            overview_sections[:1],
            [],
        )
        groups.append(index)
        if sum(_section_size(section) for section in [*overview_sections, *sections]) <= budget:
            index.sections = [*overview_sections, *sections]
            continue

        topics: dict[str, list[Section]] = {}
        if len(overview_sections) > 1:
            topics["Module overview"] = overview_sections[1:]
        for section in sections:
            topics.setdefault(_topic(section.item), []).append(section)
        for topic, topic_sections in topics.items():
            topic_size = sum(_section_size(section) for section in topic_sections)
            index_size = sum(_section_size(section) for section in index.sections)
            if topic != "Module overview" and topic_size <= budget // 6 and index_size + topic_size <= budget // 2:
                # Tiny categories belong in the module reference itself; they
                # do not warrant a near-empty page in navigation or llms.txt.
                index.sections.extend(topic_sections)
                continue
            batches = _semantic_batches(topic_sections, budget, topic)
            for title, batch in batches:
                batch_size = sum(_section_size(section) for section in batch)
                index_size = sum(_section_size(section) for section in index.sections)
                index_budget = overview_budgets.get(directory, budget)
                if (
                    topic != "Module overview"
                    and all(section.part == 1 for section in batch)
                    and batch_size <= budget // 6
                    and index_size + batch_size <= index_budget - 8_000
                ):
                    index.sections.extend(batch)
                    continue
                names = list(dict.fromkeys(section.item.name for section in batch))
                summary = names[0] if len(names) == 1 else f"{names[0]} to {names[-1]} ({len(names)} items)"
                description = f"{overview.title}: {summary}."
                slug = f"reference-{_slug(title)}"
                if any(child.path.stem == slug for child in index.children):
                    # Same-named Rust sections (for example several impl blocks)
                    # retain their rustdoc identity rather than a page counter.
                    identity = ANCHOR.search(batch[0].body)
                    slug += "-" + _slug(identity[1] if identity else batch[0].item.anchor)
                if any(child.path.stem == slug for child in index.children):
                    slug += "-" + sha256(batch[0].body.encode()).hexdigest()[:8]
                child = Group(
                    directory / f"{slug}.mdx",
                    f"{index.url}/{slug}",
                    title,
                    description,
                    len(index.children) + 1,
                    batch,
                    [],
                )
                index.children.append(child)
                groups.append(child)
    return groups


def _targets(groups: list[Group]) -> dict[str, str]:
    targets = {}
    for group in groups:
        for section in group.sections:
            item = section.item
            targets.setdefault(
                item.source.url, group.url + (f"#{item.anchor}" if item.source.html_path.name != "index.html" else "")
            )
            for anchor in ANCHOR.findall(section.body):
                original_id = unescape(anchor).removeprefix(item.anchor + "--")
                targets[f"{item.source.url}#{original_id}"] = f"{group.url}#{anchor}"
    return targets


def _render(group: Group, generated_by: str, targets: dict[str, str]) -> str:
    def rewrite(match: re.Match[str]) -> str:
        url = match[0]
        base, fragment = urldefrag(url)
        # rustdoc links can percent-encode member IDs. Removed rustdoc chrome
        # has no rendered anchor, so those links fall back to the item itself.
        return targets.get(f"{base}#{unquote(fragment)}", targets.get(base, url))

    sidebar_title = group.sections[0].item.name if group.path.name == "index.mdx" else group.title
    header = frontmatter(group.title, group.description, group.position, sidebar_title=sidebar_title, slug=group.url)
    navigation = ""
    if group.children:

        def link_label(title: str) -> str:
            return escape_mdx_text(title).replace("[", r"\[").replace("]", r"\]")

        navigation = (
            "## Reference pages\n\n"
            + "".join(f"- [{link_label(child.title)}]({child.url}): {child.description}\n" for child in group.children)
            + "\n"
        )
    body = "".join(_section_markdown(section) for section in group.sections)
    body = INTERNAL_LINK.sub(rewrite, body)

    # Fern does not resolve signature HTML links like Markdown links.
    # Relative hrefs retain the deployed version and
    # product prefix when the same generated tree is published as a snapshot.
    def signature_href(match: re.Match[str]) -> str:
        quote = match[1]
        base, fragment = urldefrag(match[2])
        target = (
            f"#{fragment}"
            if base == group.url and fragment
            else posixpath.relpath(match[2], posixpath.dirname(group.url))
        )
        return f"href={quote}{target}{quote}"

    body = re.sub(
        r'href=(\\?")(/reference/api/rust-library-reference/[^"\\]+)\1',
        signature_href,
        body,
    )
    return header + generated_by + "\n\n" + navigation + body


def group_reference(rendered: Sequence[tuple[SourcePage, str]], *, generated_by: str) -> dict[Path, str]:
    items = []
    for source, markdown in rendered:
        _, metadata, body = markdown.split("---", 2)
        fields = yaml.safe_load(metadata)
        body = body.split(generated_by, 1)[1].strip()
        items.append(
            Item(
                source,
                fields["title"],
                fields["sidebar-title"],
                fields["description"],
                fields["position"],
                _demote_headings(body),
            )
        )

    # Links can grow when an item moves to a page with a longer URL. Replan
    # against the measured output, including frontmatter, provenance and nav.
    # Fern adds version guidance, descriptions, and Markdown formatting when
    # serving .md. Measured module indexes grew by about 6%; retain 8% headroom
    # so the delivery target is not consumed entirely by generated MDX.
    source_ceiling = MAX_PAGE_CHARACTERS * 92 // 100
    default_budget = source_ceiling - len(generated_by) - 1_000
    budgets: dict[Path, int] = {}
    overview_budgets: dict[Path, int] = {}
    for _attempt in range(20):
        groups = _plan(items, budgets, overview_budgets, default_budget)
        targets = _targets(groups)
        result = {group.path: _render(group, generated_by, targets) for group in groups}
        if len(result) != len(groups):
            raise ValueError("Rust reference semantic groups have conflicting output paths")
        overflows: dict[Path, int] = {}
        overview_overflows: dict[Path, int] = {}
        groups_by_path = {group.path: group for group in groups}
        for path, body in result.items():
            if len(body) > source_ceiling:
                overflowing = overview_overflows if groups_by_path[path].children else overflows
                overflowing[path.parent] = max(overflowing.get(path.parent, 0), len(body) - source_ceiling)
        if not overflows and not overview_overflows:
            return result
        for directory, overflow in overflows.items():
            budgets[directory] = budgets.get(directory, default_budget) - overflow - 500
        for directory, overflow in overview_overflows.items():
            overview_budgets[directory] = (
                overview_budgets.get(directory, budgets.get(directory, default_budget)) - overflow - 500
            )
    raise ValueError("Rust reference navigation or an indivisible block exceeds the page character ceiling")
