# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check fixture path selection before cross-toolchain builds can overwrite binaries."""

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import check_native_abi_toolchains as matrix


class NativeAbiToolchainPathTests(unittest.TestCase):
    """Ensure both compilers' fixtures remain distinct and reachable by the matrix."""

    def setUp(self):
        """Isolate fixture paths and record commands without running compilers."""
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(directory).resolve()
        self.enterContext(patch.object(matrix, "__file__", str(self.root / "scripts/check_native_abi_toolchains.py")))
        self.enterContext(patch("sys.argv", ["check_native_abi_toolchains.py", "1.94.0"]))
        self.run_command = self.enterContext(patch.object(matrix, "run"))
        self.comparison = self.root / "target/native-abi-toolchains/1.94.0"

    def test_custom_fixture_directory_is_used_by_builds_and_matrix(self):
        """Relative and absolute overrides select current fixtures without changing comparison targets."""
        for override in ("target/custom-fixtures", str(self.root / "absolute-fixtures")):
            with (
                self.subTest(override=override),
                patch.dict(os.environ, {"NEMO_RELAY_NATIVE_ABI_FIXTURE_TARGET_DIR": override}, clear=True),
            ):
                self.run_command.reset_mock()
                matrix.main()
                calls = self.run_command.call_args_list
                current = (self.root / override).resolve()
                self.assertEqual(calls[0].args[2]["NEMO_RELAY_NATIVE_ABI_FIXTURE_TARGET_DIR"], str(current))
                self.assertEqual(calls[1].args[2]["NEMO_RELAY_NATIVE_ABI_FIXTURE_TARGET_DIR"], str(self.comparison))
                self.assertEqual(
                    calls[2].args[2]["NEMO_RELAY_TEST_NATIVE_ABI_FIXTURES"], str(self.comparison / "debug")
                )
                host_directories = set()
                for call in calls[3:]:
                    host, plugin = map(Path, call.args[0])
                    host_directories.add(host.parent)
                    self.assertIn(plugin.parent, {current / "debug", self.comparison / "debug"})
                self.assertEqual(host_directories, {current / "debug", self.comparison / "debug"})

    def test_toolchain_paths_cannot_become_build_targets(self):
        """Keep path-based toolchains inside the fixture root while passing their paths to rustup."""
        targets = set()
        for toolchain in (
            str(self.root / "toolchains/first/custom"),
            str(self.root / "toolchains/second/custom"),
            "../outside",
            ".",
            r"C:\Toolchains\custom",
        ):
            with (
                self.subTest(toolchain=toolchain),
                patch("sys.argv", ["check_native_abi_toolchains.py", toolchain]),
                patch.dict(os.environ, {}, clear=True),
            ):
                self.run_command.reset_mock()
                matrix.main()
                calls = self.run_command.call_args_list
                comparison_build = calls[1].args
                self.assertEqual(comparison_build[0][2], toolchain)
                target = Path(comparison_build[2]["NEMO_RELAY_NATIVE_ABI_FIXTURE_TARGET_DIR"])
                self.assertEqual(target.parent, self.comparison.parent)
                targets.add(target)
                for call in calls[3:]:
                    for executable in call.args[0]:
                        directory = Path(executable).parent
                        self.assertIn(directory, {self.root / "target/native-abi-fixtures/debug", target / "debug"})
        self.assertEqual(len(targets), 5)

    def test_overlapping_targets_fail_before_any_build(self):
        """Equal, ancestor, and descendant targets cannot mix compiler outputs."""
        for target in (self.comparison, self.comparison.parent, self.comparison / "nested/../nested"):
            with (
                self.subTest(target=target),
                patch.dict(os.environ, {"NEMO_RELAY_NATIVE_ABI_FIXTURE_TARGET_DIR": str(target)}, clear=True),
                contextlib.redirect_stderr(io.StringIO()) as error,
            ):
                with self.assertRaises(SystemExit) as failure:
                    matrix.main()
                self.assertEqual(failure.exception.code, 2)
                self.assertIn("must not overlap", error.getvalue())
                self.run_command.assert_not_called()

    def test_symlink_alias_cannot_hide_overlapping_targets(self):
        """Resolve aliases before deciding whether the two compilers share a target."""
        self.comparison.mkdir(parents=True)
        alias = self.root / "alias"
        try:
            alias.symlink_to(self.comparison, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"directory symlinks unavailable: {error}")
        with patch.dict(os.environ, {"NEMO_RELAY_NATIVE_ABI_FIXTURE_TARGET_DIR": str(alias)}, clear=True):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                matrix.main()
        self.run_command.assert_not_called()
