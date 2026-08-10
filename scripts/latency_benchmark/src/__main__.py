# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the configurable latency benchmark."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("benchmark interrupted", file=sys.stderr)
        raise SystemExit(130) from None
