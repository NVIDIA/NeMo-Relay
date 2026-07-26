# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prefetch the pinned Rampart model before enabling the worker."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .detector import prefetch_verified_model


def main(argv: Sequence[str] | None = None) -> None:
    """Download and verify the model in the configured Hugging Face cache."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        help="Optional Hugging Face cache directory shared with the worker.",
    )
    args = parser.parse_args(argv)
    model_root = prefetch_verified_model(args.cache_dir)
    print(f"Verified Rampart model at {model_root}")


if __name__ == "__main__":
    main()
