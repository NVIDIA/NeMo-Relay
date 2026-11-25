# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor

import nemo_relay


def run_example() -> None:
    """Use an explicit scope stack across execution boundaries."""
    with nemo_relay.scope.scope("request", nemo_relay.ScopeType.Agent):
        nemo_relay.scope.event("started", data={"ok": True})
        shared = nemo_relay.propagate_scope_to_thread()

        def worker() -> None:
            nemo_relay.set_thread_scope_stack(shared)
            nemo_relay.scope.event("worker-ran")

        with ThreadPoolExecutor() as pool:
            pool.submit(worker).result()
