# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live integration tests for the OCI Generative AI codecs.

These tests execute a real chat call through ``nemo_relay.llm.execute`` with
``OCIGenAIChatCodec`` and ``OCIGenAIResponseCodec`` against an OCI Generative
AI endpoint. They are skipped unless the environment opts in:

- ``NEMO_RELAY_OCI_LIVE=1``
- ``OCI_GENAI_COMPARTMENT_ID``: compartment OCID for the request.
- ``OCI_GENAI_ENDPOINT_ID`` (dedicated AI cluster endpoint OCID) or
  ``OCI_GENAI_MODEL_ID`` (on-demand model id such as
  ``meta.llama-3.3-70b-instruct``).
- ``OCI_CLI_PROFILE`` (default ``DEFAULT``) and optional
  ``OCI_GENAI_REGION`` (default ``us-chicago-1``).

Example::

    NEMO_RELAY_OCI_LIVE=1 \
    OCI_GENAI_COMPARTMENT_ID=ocid1.compartment.oc1..example \
    OCI_GENAI_ENDPOINT_ID=ocid1.generativeaiendpoint.oc1.us-chicago-1.example \
    OCI_CLI_PROFILE=DEFAULT \
    uv run --with oci pytest python/tests/providers/test_oci_genai_live.py
"""

from __future__ import annotations

import os
import typing

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("NEMO_RELAY_OCI_LIVE") != "1",
    reason="live OCI Generative AI tests require NEMO_RELAY_OCI_LIVE=1 and OCI credentials",
)


def _signer_and_endpoint():
    """Build a request signer and the regional chat endpoint URL.

    Signing raw HTTP (instead of using SDK model classes) sends the codec's
    encoded payload to the service verbatim, so the test also validates that
    ``OCIGenAIChatCodec.encode()`` produces the exact OCI wire format.
    """
    oci = pytest.importorskip("oci")

    profile = os.environ.get("OCI_CLI_PROFILE", "DEFAULT")
    region = os.environ.get("OCI_GENAI_REGION", "us-chicago-1")
    config = oci.config.from_file(profile_name=profile)

    token_file = config.get("security_token_file")
    if token_file:
        with open(os.path.expanduser(token_file), encoding="utf-8") as handle:
            token = handle.read().strip()
        private_key = oci.signer.load_private_key_from_file(config["key_file"])
        signer = oci.auth.signers.SecurityTokenSigner(token, private_key)
    else:
        signer = oci.signer.Signer(
            tenancy=config["tenancy"],
            user=config["user"],
            fingerprint=config["fingerprint"],
            private_key_file_location=config["key_file"],
            pass_phrase=config.get("pass_phrase"),
        )

    endpoint = f"https://inference.generativeai.{region}.oci.oraclecloud.com/20231130/actions/chat"
    return signer, endpoint


def _serving_mode() -> dict[str, str]:
    endpoint_id = os.environ.get("OCI_GENAI_ENDPOINT_ID")
    if endpoint_id:
        return {"servingType": "DEDICATED", "endpointId": endpoint_id}
    model_id = os.environ.get("OCI_GENAI_MODEL_ID")
    if not model_id:
        pytest.skip("set OCI_GENAI_ENDPOINT_ID or OCI_GENAI_MODEL_ID")
    return {"servingType": "ON_DEMAND", "modelId": model_id}


async def test_generic_chat_through_relay_with_codecs():
    import nemo_relay
    from nemo_relay.providers.oci_genai import OCIGenAIChatCodec, OCIGenAIResponseCodec

    requests = pytest.importorskip("requests")
    signer, endpoint = _signer_and_endpoint()
    compartment_id = os.environ["OCI_GENAI_COMPARTMENT_ID"]

    chat_details: dict[str, typing.Any] = {
        "compartmentId": compartment_id,
        "servingMode": _serving_mode(),
        "chatRequest": {
            "apiFormat": "GENERIC",
            "messages": [
                {
                    "role": "USER",
                    "content": [{"type": "TEXT", "text": "/no_think Reply with exactly: RELAY_OCI_OK"}],
                }
            ],
            "maxTokens": 600,
            "temperature": 0.0,
        },
    }

    async def call_oci(request: nemo_relay.LLMRequest):
        response = requests.post(endpoint, json=request.content, auth=signer, timeout=120)
        response.raise_for_status()
        return response.json()

    result = await nemo_relay.llm.execute(
        "oci-genai",
        nemo_relay.LLMRequest({}, chat_details),
        call_oci,
        codec=OCIGenAIChatCodec(),
        response_codec=OCIGenAIResponseCodec(),
    )

    text = result["chatResponse"]["choices"][0]["message"]["content"][0]["text"]
    assert "RELAY_OCI_OK" in text

    # The response codec must decode the same payload without error.
    annotated = OCIGenAIResponseCodec().decode_response(result)
    assert annotated.message is not None
    assert "RELAY_OCI_OK" in str(annotated.message)
