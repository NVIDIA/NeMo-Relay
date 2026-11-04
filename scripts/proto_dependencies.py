# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the locked protobuf dependency through Buf and print its include path."""

import runpy
from pathlib import Path


def main() -> None:
    backend = Path(__file__).resolve().parents[1] / "python/plugin/build_backend.py"
    resolve = runpy.run_path(str(backend))["resolve_proto_dependencies"]
    print(resolve())


if __name__ == "__main__":
    main()
