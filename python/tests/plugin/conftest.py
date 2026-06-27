# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collection configuration for Python worker plugin SDK tests."""

from __future__ import annotations

import platform
import sys

_is_windows_arm64 = sys.platform == "win32" and platform.machine().lower() in {"arm64", "aarch64"}

collect_ignore_glob = ["test_*.py"] if _is_windows_arm64 else []
