# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from scripts.package_python_musllinux import semver_to_pep440


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("0.7.0", "0.7.0"),
        ("0.7.0-rc.2", "0.7.0rc2"),
        ("0.7.0+SHA-123", "0.7.0+sha.123"),
    ],
)
def test_semver_to_pep440(version, expected):
    assert semver_to_pep440(version) == expected


def test_semver_to_pep440_rejects_invalid_version():
    with pytest.raises(ValueError, match="Unsupported Python package version format"):
        semver_to_pep440("0.7")
