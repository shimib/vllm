# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The render fingerprint check on `/inference/v1/generate`: a worker must not
run token IDs that a differently configured renderer produced."""

from argparse import Namespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.scale_out.render.fingerprint import RenderFingerprint
from vllm.entrypoints.scale_out.token_in_token_out.api_router import router

OWN = "rf1:aa:vllm-py/1.0;renderer=hf;tokenizer=tokenizers/0.22.0"
OTHER_BUILD = "rf1:aa:vllm-rs/0.1.0;renderer=hf;tokenizer=fastokens/0.3.5"
HEADER = "X-Render-Fingerprint"
BODY = {"token_ids": [1, 2, 3], "sampling_params": {}}


async def _stream():
    yield b"data: [DONE]\n\n"


def _client(**flags) -> tuple[TestClient, AsyncMock]:
    app = FastAPI()
    app.include_router(router)
    app.state.args = Namespace(
        **{
            "fingerprint_compare": "full",
            "on_fingerprint_mismatch": "reject",
            "require_render_fingerprint": False,
            **flags,
        }
    )
    app.state.render_fingerprint = RenderFingerprint(OWN, "local", {})
    serve_tokens = AsyncMock(side_effect=lambda *_: _stream())
    app.state.serving_tokens = AsyncMock(serve_tokens=serve_tokens)
    return TestClient(app), serve_tokens


def test_mismatch_is_rejected_before_generation_and_names_both_sides():
    client, serve_tokens = _client()

    response = client.post(
        "/inference/v1/generate", json=BODY, headers={HEADER: OTHER_BUILD}
    )

    assert response.status_code == 409
    error = response.json()["error"]
    assert (error["received"], error["expected"]) == (OTHER_BUILD, OWN)
    serve_tokens.assert_not_called()


def test_match_runs_and_returns_the_workers_fingerprint():
    client, serve_tokens = _client()

    response = client.post("/inference/v1/generate", json=BODY, headers={HEADER: OWN})

    assert response.status_code == 200
    assert response.headers[HEADER] == OWN
    serve_tokens.assert_called_once()


@pytest.mark.parametrize(
    "flags",
    [{"fingerprint_compare": "inputs"}, {"on_fingerprint_mismatch": "warn"}],
)
def test_other_build_is_accepted_only_when_configured(flags):
    client, _ = _client(**flags)

    response = client.post(
        "/inference/v1/generate", json=BODY, headers={HEADER: OTHER_BUILD}
    )

    assert response.status_code == 200


def test_rerender_rejects_because_the_route_carries_no_messages():
    client, _ = _client(on_fingerprint_mismatch="rerender")

    response = client.post(
        "/inference/v1/generate", json=BODY, headers={HEADER: OTHER_BUILD}
    )

    assert response.status_code == 409


@pytest.mark.parametrize(("required", "status"), [(False, 200), (True, 400)])
def test_missing_fingerprint_is_rejected_only_when_required(required, status):
    client, _ = _client(require_render_fingerprint=required)

    response = client.post("/inference/v1/generate", json=BODY)

    assert response.status_code == status
