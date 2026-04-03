# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the declarative proxy API via nat_nexus.proxy submodule.

Covers:
- Setter callability: set_use_proxy, set_proxy_backend, set_proxy_sensitivity, set_dynamo_intercept
- State query: proxy_active() initial state
- Lifecycle: enable -> ensure -> active -> teardown -> inactive
- Idempotency: double ensure_proxy() is safe
- Backwards compatibility: NexusProxy builder still works (D-13)
"""

import nat_nexus
import pytest


@pytest.fixture(autouse=True)
def _cleanup_proxy():
    """Ensure proxy is torn down after each test."""
    yield
    if nat_nexus.proxy.proxy_active():
        nat_nexus.proxy.teardown_proxy()


class TestDeclarativeProxySetters:
    """Test that setter functions are callable without error."""

    def test_proxy_active_initially_false(self):
        """proxy_active() returns False when no proxy has been configured."""
        assert nat_nexus.proxy.proxy_active() is False

    def test_set_use_proxy_callable(self):
        """set_use_proxy(True) does not raise."""
        nat_nexus.proxy.set_use_proxy(True)

    def test_set_proxy_backend_callable(self):
        """set_proxy_backend(InMemoryBackend()) does not raise."""
        nat_nexus.proxy.set_proxy_backend(nat_nexus.proxy.InMemoryBackend())

    def test_set_proxy_sensitivity_callable(self):
        """set_proxy_sensitivity(SensitivityConfig(...)) does not raise."""
        nat_nexus.proxy.set_proxy_sensitivity(
            nat_nexus.proxy.SensitivityConfig(
                w_critical=0.35,
                w_fanout=0.15,
                w_position=0.5,
                w_parallel=0.5,
            )
        )

    def test_set_dynamo_intercept_callable(self):
        """set_dynamo_intercept(True) does not raise."""
        nat_nexus.proxy.set_dynamo_intercept(True)


class TestDeclarativeProxyLifecycle:
    """Test the full declarative proxy lifecycle."""

    async def test_declarative_lifecycle(self):
        """Full lifecycle: enable -> ensure -> active -> teardown -> inactive."""
        nat_nexus.proxy.set_use_proxy(True)
        assert nat_nexus.proxy.proxy_active() is False  # not created yet

        nat_nexus.get_scope_stack()  # ensure scope stack exists for registration
        await nat_nexus.proxy.ensure_proxy()
        assert nat_nexus.proxy.proxy_active() is True

        nat_nexus.proxy.teardown_proxy()
        assert nat_nexus.proxy.proxy_active() is False

    async def test_ensure_proxy_idempotent(self):
        """Calling ensure_proxy() twice does not raise."""
        nat_nexus.proxy.set_use_proxy(True)
        nat_nexus.get_scope_stack()
        await nat_nexus.proxy.ensure_proxy()
        await nat_nexus.proxy.ensure_proxy()  # second call is no-op
        assert nat_nexus.proxy.proxy_active() is True
        nat_nexus.proxy.teardown_proxy()


class TestDeclarativeProxyBackwardsCompat:
    """Ensure the existing NexusProxy builder still works (D-13)."""

    async def test_existing_builder_still_works(self):
        """Existing explicit NexusProxy builder continues to work."""
        nat_nexus.get_scope_stack()
        proxy = nat_nexus.proxy.NexusProxy(
            agent_id="test-builder",
            backend=nat_nexus.proxy.InMemoryBackend(),
        )
        await proxy.register()
        proxy.deregister()
