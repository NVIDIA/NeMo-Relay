# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NeMo Relay observability over OCI Generative AI (langchain-oci) payloads."""

from __future__ import annotations

import types
import typing
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

if typing.TYPE_CHECKING:
    from nemo_relay.integrations.langchain.callbacks import NemoRelayCallbackHandler


OCI_CHAT_MODEL_ID = ["langchain_oci", "chat_models", "oci_generative_ai", "ChatOCIGenAI"]


def _make_mock_nemo_relay() -> MagicMock:
    """Build a minimal mock of the ``nemo_relay`` module."""
    mock_nemo_relay = MagicMock(name="nemo_relay")
    mock_nemo_relay.ScopeType = types.SimpleNamespace(Agent="Agent")

    scope = types.SimpleNamespace()
    scope.push = MagicMock(
        side_effect=lambda name, scope_type, **kwargs: types.SimpleNamespace(
            uuid=str(uuid4()),
            name=name,
            scope_type=scope_type,
            kwargs=kwargs,
        )
    )
    scope.pop = MagicMock()
    mock_nemo_relay.scope = scope
    return mock_nemo_relay


@pytest.fixture(name="callbacks_module", scope="session")
def callbacks_module_fixture() -> types.ModuleType:
    """Fixture to provide the callbacks module."""
    import nemo_relay.integrations.langchain.callbacks as callbacks_module

    return callbacks_module


@pytest.fixture()
def mock_nemo_relay(monkeypatch: pytest.MonkeyPatch, callbacks_module: types.ModuleType) -> MagicMock:
    mock_nemo_relay = _make_mock_nemo_relay()
    monkeypatch.setattr(callbacks_module, "nemo_relay", mock_nemo_relay)
    return mock_nemo_relay


@pytest.fixture()
def handler(mock_nemo_relay: MagicMock) -> NemoRelayCallbackHandler:
    from nemo_relay.integrations.langchain.callbacks import NemoRelayCallbackHandler

    return NemoRelayCallbackHandler()


class TestOciGenAiScopeNaming:
    """Verify OCI Generative AI chain runs resolve to a usable scope name."""

    def test_scope_named_from_langchain_oci_id_list(
        self, handler: NemoRelayCallbackHandler, mock_nemo_relay: MagicMock
    ):
        """A langchain-oci serialized payload without a name falls back to the class name."""
        run_id = uuid4()

        handler.on_chain_start(
            {"id": OCI_CHAT_MODEL_ID},
            {"input": "What is the weather in San Francisco?"},
            run_id=run_id,
        )

        mock_nemo_relay.scope.push.assert_called_once()
        args, _ = mock_nemo_relay.scope.push.call_args
        assert args == ("ChatOCIGenAI", mock_nemo_relay.ScopeType.Agent)

    def test_oci_model_metadata_passes_through(
        self, handler: NemoRelayCallbackHandler, mock_nemo_relay: MagicMock
    ):
        """OCI-specific metadata (model id, serving mode) survives into the scope."""
        run_id = uuid4()
        metadata = {
            "ls_provider": "oci_generative_ai",
            "ls_model_name": "meta.llama-3.3-70b-instruct",
            "serving_mode": "ON_DEMAND",
        }

        handler.on_chain_start(
            {"id": OCI_CHAT_MODEL_ID},
            {"input": "test"},
            run_id=run_id,
            metadata=metadata,
        )

        _, kwargs = mock_nemo_relay.scope.push.call_args
        assert kwargs["metadata"]["ls_provider"] == "oci_generative_ai"
        assert kwargs["metadata"]["ls_model_name"] == "meta.llama-3.3-70b-instruct"
        assert kwargs["metadata"]["serving_mode"] == "ON_DEMAND"
        assert kwargs["metadata"]["langchain_run_id"] == str(run_id)

    def test_dedicated_endpoint_model_id_in_metadata(
        self, handler: NemoRelayCallbackHandler, mock_nemo_relay: MagicMock
    ):
        """Dedicated AI cluster endpoints (endpoint OCID model ids) are preserved."""
        run_id = uuid4()
        endpoint_ocid = "ocid1.generativeaiendpoint.oc1.us-chicago-1.example"

        handler.on_chain_start(
            {"id": OCI_CHAT_MODEL_ID},
            {"input": "test"},
            run_id=run_id,
            metadata={"ls_model_name": endpoint_ocid, "serving_mode": "DEDICATED"},
        )

        _, kwargs = mock_nemo_relay.scope.push.call_args
        assert kwargs["metadata"]["ls_model_name"] == endpoint_ocid
        assert kwargs["metadata"]["serving_mode"] == "DEDICATED"


class TestOciGenAiScopeLifecycle:
    """Verify start/end/error pair correctly for OCI-backed chain runs."""

    def test_end_pops_scope(self, handler: NemoRelayCallbackHandler, mock_nemo_relay: MagicMock):
        run_id = uuid4()
        handler.on_chain_start({"id": OCI_CHAT_MODEL_ID}, {"input": "test"}, run_id=run_id)
        handler.on_chain_end({"output": "72 degrees"}, run_id=run_id)

        mock_nemo_relay.scope.pop.assert_called_once()

    def test_error_pops_scope_with_error_status(
        self, handler: NemoRelayCallbackHandler, mock_nemo_relay: MagicMock
    ):
        run_id = uuid4()
        handler.on_chain_start({"id": OCI_CHAT_MODEL_ID}, {"input": "test"}, run_id=run_id)
        handler.on_chain_error(RuntimeError("OCI service error: 429"), run_id=run_id)

        mock_nemo_relay.scope.pop.assert_called_once()
        _, kwargs = mock_nemo_relay.scope.pop.call_args
        metadata = kwargs.get("metadata") or {}
        assert metadata.get("otel.status_code") == "ERROR"
