# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise documentation retention against complete temporary Fern layouts."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "sync_fern_docs_branch", Path(__file__).resolve().parents[1] / "docs" / "sync_fern_docs_branch.py"
)
assert SPEC is not None and SPEC.loader is not None
SYNC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SYNC)


class DocsRetentionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.source = root / "source"
        self.target = root / "target"
        (self.source / "docs").mkdir(parents=True)
        (self.source / "fern").mkdir()
        SYNC.write_yaml(self.source / "docs" / "index.yml", {"navigation": [{"page": "Guide", "path": "guide.mdx"}]})
        (self.source / "docs" / "guide.mdx").write_text("See github.com/NVIDIA/NeMo-Relay/blob/main/README.md\n")
        SYNC.write_yaml(self.source / "fern" / "docs.yml", {"products": [{"display-name": "Relay", "slug": "/"}]})
        SYNC.sync_dev(self.source, self.target)

    def publish(self, *tags: str) -> None:
        for tag in tags:
            SYNC.release_version(self.target, tag, self.source)

    def assert_layout(self, expected: list[str], latest: str | None) -> None:
        fern = self.target / "fern"
        product = SYNC.read_yaml(fern / "docs.yml")["products"][0]
        entries = product["versions"]
        self.assertEqual([entry["slug"] for entry in entries], (["latest"] if latest else []) + ["dev", *expected])
        self.assertEqual(product["path"], f"./versions/{latest or 'dev'}.yml")
        if latest:
            self.assertEqual(entries[0]["path"], product["path"])
            self.assertEqual(entries[0]["display-name"], f"Latest ({latest})")
        self.assertEqual({path.name for path in fern.glob("pages-v*")}, {f"pages-{tag}" for tag in expected})
        self.assertEqual({path.stem for path in (fern / "versions").glob("v*.yml")}, set(expected))
        for entry in entries:
            self.assertTrue((fern / entry["path"]).is_file())
        self.assertTrue((fern / "pages-dev" / "guide.mdx").is_file())

    def test_replaces_patches_and_keeps_five_numeric_versions(self) -> None:
        self.publish("0.6.9", "0.7.4", "0.8.4", "0.9.3", "0.10.3", "0.10.4")
        self.assert_layout(["v0.10.4", "v0.9.3", "v0.8.4", "v0.7.4", "v0.6.9"], "v0.10.4")
        self.publish("0.11.0")
        self.assert_layout(["v0.11.0", "v0.10.4", "v0.9.3", "v0.8.4", "v0.7.4"], "v0.11.0")
        self.publish("0.12.0")
        self.assert_layout(["v0.12.0", "v0.11.0", "v0.10.4", "v0.9.3", "v0.8.4"], "v0.12.0")

    def test_republishing_old_tags_does_not_restore_pruned_docs_or_move_latest(self) -> None:
        self.publish("1.0.0", "0.10.4", "0.10.3", "0.9.0", "0.8.0", "0.7.0", "0.6.0", "1.0.0")
        self.assert_layout(["v1.0.0", "v0.10.4", "v0.9.0", "v0.8.0", "v0.7.0"], "v1.0.0")

    def test_prereleases_share_base_snapshot_and_preserve_stable_latest(self) -> None:
        self.publish("0.10.4", "0.11.0-beta.1", "0.11.0-rc.1")
        self.assert_layout(["v0.11.0", "v0.10.4"], "v0.10.4")
        self.publish("0.11.0")
        self.assert_layout(["v0.11.0", "v0.10.4"], "v0.11.0")

    def test_prerelease_only_layout_has_no_dangling_latest(self) -> None:
        self.publish("0.10.0", "0.10.1-rc.1")
        self.assert_layout(["v0.10.1"], None)

    def test_repository_redirects_survive_release_retention_and_dev_sync(self) -> None:
        """Keep legacy redirects usable after their original snapshots expire."""
        # Define targets independently of redirects so destination typos fail.
        routes = [
            "guide",
            "reference/migration-guides",
            "build-plugins/about",
            "build-plugins/native/about",
            "build-plugins/native/build-and-package",
            "build-plugins/workers/about",
            "build-plugins/workers/python",
            "build-plugins/workers/rust",
            "build-plugins/workers/grpc-v1-protocol",
            *(
                f"reference/api/python-library-reference/nemo-relay/{module}"
                for module in (
                    "pii-redaction",
                    "adaptive",
                    "codecs",
                    "guardrails",
                    "integrations",
                    "integrations/langchain/callbacks",
                    "intercepts",
                    "llm",
                    "model-pricing",
                    "observability",
                    "plugin",
                    "scope-local",
                    "scope",
                    "subscribers",
                    "tools",
                    "typed",
                    "utils",
                )
            ),
            "reference/api/rust-library-reference/nemo-relay/codec/model-pricing",
            "reference/api/rust-library-reference/nemo-relay/codec/model-pricing/enum-cachereadaccounting",
            "reference/api/rust-library-reference/nemo-relay/plugins/model-pricing",
            "reference/api/rust-library-reference/nemo-relay/plugins/model-pricing/constant-pricing-plugin-kind",
        ]
        navigation = []
        for route in routes:
            page = self.source / "docs" / f"{route}.mdx"
            page.parent.mkdir(parents=True, exist_ok=True)
            page.write_text(f"# {route}\n")
            navigation.append({"page": route, "slug": route, "path": f"./{route}.mdx"})
        SYNC.write_yaml(self.source / "docs" / "index.yml", {"navigation": navigation})
        config = SYNC.read_yaml(Path(__file__).resolve().parents[2] / "fern" / "docs.yml")
        config["redirects"].extend(
            [
                {"source": "/legacy/latest", "destination": "/nemo/relay/latest/guide"},
                {"source": "/legacy/stable", "destination": "/nemo/relay/v0.12.0/guide"},
            ]
        )
        SYNC.write_yaml(self.source / "fern" / "docs.yml", config)
        SYNC.sync_dev(self.source, self.target)
        self.publish("0.5.0", "0.6.0", "0.7.3", "0.8.0", "0.9.0", "0.10.0", "0.11.0", "0.12.0")
        for sync_dev in (False, True):
            if sync_dev:
                SYNC.sync_dev(self.source, self.target)
            self.assert_layout(["v0.12.0", "v0.11.0", "v0.10.0", "v0.9.0", "v0.8.0"], "v0.12.0")
            docs = SYNC.read_yaml(self.target / "fern" / "docs.yml")
            self.assertEqual(docs["redirects"], config["redirects"])
            entries = {entry["slug"]: entry for entry in docs["products"][0]["versions"]}
            for redirect in docs["redirects"]:
                destination = redirect["destination"].removeprefix("/nemo/relay/")
                version, _, route = destination.partition("/")
                if SYNC.SNAPSHOT_RE.fullmatch(version) or version in {"dev", "latest"}:
                    with self.subTest(sync_dev=sync_dev, source=redirect["source"]):
                        self.assertIn(version, entries, redirect["destination"])
                        if ":slug*" in route:
                            example = (
                                "langchain/callbacks"
                                if "python-library-reference" in route
                                else "enum-cachereadaccounting"
                                if "/codec/" in route
                                else "constant-pricing-plugin-kind"
                            )
                            route = route.replace(":slug*", example)
                        nav_path = self.target / "fern" / entries[version]["path"]
                        pages = {page["slug"]: page["path"] for page in SYNC.read_yaml(nav_path)["navigation"]}
                        self.assertIn(route, pages, redirect["destination"])
                        page_path = (nav_path.parent / pages[route]).resolve()
                        snapshot = nav_path.stem
                        self.assertTrue(
                            page_path.is_relative_to((self.target / "fern" / f"pages-{snapshot}").resolve())
                        )
                        self.assertTrue(page_path.is_file(), redirect["destination"])

    def test_dev_sync_prunes_existing_layout_and_orphaned_snapshots(self) -> None:
        # Assemble the legacy layout, which retained every patch and minor.
        fern = self.target / "fern"
        docs = SYNC.read_yaml(fern / "docs.yml")
        entries = docs["products"][0]["versions"]
        for tag in ["v0.6.0", "v0.7.0", "v0.8.0", "v0.9.0", "v0.10.3", "v0.10.4", "v0.11.0"]:
            (fern / f"pages-{tag}").mkdir()
            SYNC.write_yaml(fern / "versions" / f"{tag}.yml", {"navigation": []})
            entries.append({"slug": tag, "path": f"./versions/{tag}.yml", "availability": "stable"})
        (fern / "pages-v0.1.0").mkdir()
        SYNC.write_yaml(fern / "versions" / "v0.2.0.yml", {"navigation": []})
        SYNC.write_yaml(fern / "docs.yml", docs)
        SYNC.sync_dev(self.source, self.target)
        self.assert_layout(["v0.11.0", "v0.10.4", "v0.9.0", "v0.8.0", "v0.7.0"], "v0.11.0")
        before = (fern / "docs.yml").read_bytes()
        SYNC.sync_dev(self.source, self.target)
        self.assertEqual((fern / "docs.yml").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
