# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import nemo_relay


def main() -> None:
    """Emit two checkpoint marks within an agent scope."""

    def log_event(event: nemo_relay.Event) -> None:
        print(f"{event.kind} {event.name}")

    nemo_relay.subscribers.register("scope-check", log_event)

    try:
        with nemo_relay.scope.scope(
            "agent-run",
            nemo_relay.ScopeType.Agent,
            input={"request_id": "req-123"},
        ) as handle:
            nemo_relay.scope.event("planning-started", handle=handle, data={"step": 1})
            nemo_relay.scope.event("planning-finished", handle=handle, data={"step": 2})
    finally:
        nemo_relay.subscribers.flush()
        nemo_relay.subscribers.deregister("scope-check")


if __name__ == "__main__":
    main()
