#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Version helpers for the released NeMo Relay wheel used by Phase 2."""

from __future__ import annotations

import re
from pathlib import Path
from zipfile import ZipFile

RELAY_REQUIREMENT = "nemo-relay>=0.7.0"
RELAY_MIN_VERSION = "0.7.0"


def version_tuple(value: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", value)
    if match is None:
        raise ValueError(f"nemo-relay version must be a stable semantic version: {value!r}")
    return tuple(int(component) for component in match.groups())


def require_supported_version(value: str) -> str:
    if version_tuple(value) < version_tuple(RELAY_MIN_VERSION):
        raise ValueError(f"nemo-relay must satisfy {RELAY_REQUIREMENT}; found {value}")
    return value


def wheel_version(path: Path) -> str:
    with ZipFile(path) as wheel:
        metadata_names = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise ValueError("Relay wheel has an ambiguous METADATA payload")
        metadata = wheel.read(metadata_names[0]).decode("utf-8", errors="strict")
    name = re.search(r"^Name: (.+)$", metadata, re.MULTILINE)
    version = re.search(r"^Version: (.+)$", metadata, re.MULTILINE)
    if name is None or name.group(1) != "nemo-relay" or version is None:
        raise ValueError("Relay wheel metadata does not identify nemo-relay")
    return require_supported_version(version.group(1))
