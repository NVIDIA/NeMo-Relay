#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exec a command as the leader of a new POSIX process group."""

from __future__ import annotations

import os
import sys
from errno import EPERM


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: exec_process_group.py COMMAND [ARG ...]")
    try:
        os.setsid()
    except PermissionError as error:
        if error.errno != EPERM:
            raise
        # Some launchers already make the child a process-group leader. In
        # that case setsid(2) is forbidden, but the existing isolated group is
        # exactly what the supervisor needs to signal as a unit.
        if os.getpgrp() != os.getpid():
            raise RuntimeError("could not isolate the coordinator process group") from error
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
