# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for coverage object discovery, grouping, and report merging."""

import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

with patch.object(sys, "path", [str(Path(__file__).resolve().parents[1] / "coverage"), *sys.path]):
    EXPORT = importlib.import_module("export_llvm_cov_lcov")


def nextest_document(target: Path, suites: dict[str, tuple[str, Path]]) -> dict[str, object]:
    return {
        "rust-build-meta": {"target-directory": str(target), "base-output-directories": ["debug"]},
        "rust-suites": {
            suite_id: {"kind": kind, "binary-path": str(binary)} for suite_id, (kind, binary) in suites.items()
        },
    }


def executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x7fELF")
    path.chmod(0o755)
    return path


class NextestListTests(unittest.TestCase):
    def test_unit_test_binaries_come_from_non_integration_kinds(self) -> None:
        target = Path("/tmp/target")
        build = EXPORT.parse_nextest_list(
            nextest_document(
                target,
                {
                    "ffi": ("lib", target / "debug/deps/nemo_relay_ffi-aa"),
                    "ffi::integration": ("test", target / "debug/deps/integration-bb"),
                    "cli": ("bin", target / "debug/deps/nemo_relay-cc"),
                },
            )
        )
        self.assertEqual(build.build_directories, (target.resolve() / "debug",))
        expected = {target / "debug/deps/nemo_relay_ffi-aa", target / "debug/deps/nemo_relay-cc"}
        self.assertEqual(build.unit_test_binaries, frozenset(path.resolve() for path in expected))

    def test_separate_build_directory_adds_the_uplift_root(self) -> None:
        target = Path("/tmp/target")
        document = nextest_document(target, {})
        meta = document["rust-build-meta"]
        assert isinstance(meta, dict)
        meta["build-directory"] = "/tmp/build"
        build = EXPORT.parse_nextest_list(document)
        self.assertEqual(build.build_directories, (Path("/tmp/build").resolve() / "debug", target.resolve() / "debug"))

    def test_unknown_suite_kind_is_rejected(self) -> None:
        document = nextest_document(Path("/tmp/target"), {"odd": ("custom-build", Path("/tmp/target/debug/x"))})
        with self.assertRaises(EXPORT.CoverageExportError):
            EXPORT.parse_nextest_list(document)


class ObjectOrderingTests(unittest.TestCase):
    @unittest.skipIf(sys.platform == "win32", "executable bits are POSIX only")
    def test_objects_match_cargo_artifacts_and_unit_tests_load_last(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory, "debug")
            integration = executable(build / "deps/integration-0123456789abcdef")
            unit = executable(build / "deps/nemo_relay_ffi-0123456789abcdef")
            cdylib = executable(build / "deps/libnemo_relay_ffi.dylib")
            cli = executable(build / "nemo-relay")
            executable(build / "deps/unrelated-0123456789abcdef")
            executable(build / "incremental/nemo_relay_ffi-0123456789abcdef")
            executable(build / "build/nemo-relay-ffi-0123456789abcdef/build-script-build")
            (build / "deps/nemo_relay_ffi-0123456789abcdef.d").write_text("")
            plain = build / "deps/nemo_relay_ffi-fedcba9876543210"
            plain.write_bytes(b"")
            plain.chmod(0o644)

            names = frozenset({"nemo-relay-ffi", "integration", "nemo-relay", "nemo_relay_ffi"})
            pattern = EXPORT.artifact_pattern(names)
            objects = EXPORT.collect_objects([build], pattern)
            self.assertEqual(objects, sorted(p.resolve() for p in (integration, unit, cdylib, cli)))

            shared, unit_tests = EXPORT.partition_objects(objects, frozenset({unit.resolve()}))
            self.assertEqual(unit_tests, [unit.resolve()])
            self.assertEqual(shared, sorted(p.resolve() for p in (integration, cdylib, cli)))

    def test_missing_unit_test_binary_is_an_error(self) -> None:
        with self.assertRaises(EXPORT.CoverageExportError):
            EXPORT.partition_objects([Path("/a/deps/integration-aa")], frozenset({Path("/a/deps/nemo_relay_ffi-bb")}))


class ReportMergeTests(unittest.TestCase):
    def test_source_records_become_workspace_relative(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "crates" / "ffi" / "src" / "lib.rs"
            lcov = f"SF:{source}\nDA:1,1\nDA:2,0\nend_of_record\n"
            self.assertEqual(EXPORT.line_counts(lcov, root), {"crates/ffi/src/lib.rs": {1: 1, 2: 0}})

    def test_sources_outside_the_workspace_are_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with self.assertRaises(EXPORT.CoverageExportError):
                EXPORT.line_counts(f"SF:{Path(os.sep) / 'elsewhere' / 'lib.rs'}\nend_of_record\n", root)

    def test_empty_reports_are_an_error(self) -> None:
        with self.assertRaises(EXPORT.CoverageExportError):
            EXPORT.line_counts("", Path("/tmp"))

    def test_union_keeps_a_line_covered_when_either_export_covers_it(self) -> None:
        shared = {"crates/ffi/src/a.rs": {1: 0, 2: 5}, "crates/ffi/src/b.rs": {7: 1}}
        unit_tests = {"crates/ffi/src/a.rs": {1: 3, 2: 0, 3: 1}}
        merged = EXPORT.merge_line_counts(shared, unit_tests)
        self.assertEqual(merged, {"crates/ffi/src/a.rs": {1: 3, 2: 5, 3: 1}, "crates/ffi/src/b.rs": {7: 1}})
        self.assertEqual(
            EXPORT.render_lcov(merged),
            "SF:crates/ffi/src/a.rs\nDA:1,3\nDA:2,5\nDA:3,1\nLF:3\nLH:3\nend_of_record\n"
            "SF:crates/ffi/src/b.rs\nDA:7,1\nLF:1\nLH:1\nend_of_record\n",
        )


class IgnoreRegexTests(unittest.TestCase):
    def test_regex_names_workspace_tests_and_build_outputs(self) -> None:
        root = Path("/work/repo")
        regex = EXPORT.ignore_filename_regex(root, [root / "target"])
        self.assertIn(EXPORT.ere_escape(str(root / "target")), regex)
        self.assertIn("(tests|examples|benches)", regex)
        self.assertEqual(EXPORT.ere_escape("a.b+c(d)"), "a\\.b\\+c\\(d\\)")


if __name__ == "__main__":
    unittest.main()
