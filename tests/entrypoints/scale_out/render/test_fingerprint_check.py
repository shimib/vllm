# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Policy of the render fingerprint check: a worker must not run token IDs
that a differently configured renderer produced. The wiring into
`/inference/v1/generate` is covered in `test_serving_tokens.py`."""

from argparse import Namespace

import pytest
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient

from vllm.entrypoints.scale_out.render import fingerprint
from vllm.entrypoints.scale_out.render.fingerprint import (
    RENDER_FINGERPRINT_HEADER,
    RenderFingerprint,
    check_render_fingerprint,
)

DIGEST = "a" * 64
OWN = f"rf1:{DIGEST}:vllm-py/1.0;renderer=hf;tokenizer=tokenizers/0.22.0"
OTHER_BUILD = f"rf1:{DIGEST}:vllm-rs/0.1.0;renderer=hf;tokenizer=fastokens/0.3.5"


def _status(headers: dict[str, str], **flags) -> tuple[int, dict]:
    app = FastAPI()
    app.state.args = Namespace(
        **{
            "fingerprint_compare": "full",
            "on_fingerprint_mismatch": "reject",
            "require_render_fingerprint": False,
            **flags,
        }
    )
    app.state.render_fingerprint = RenderFingerprint(OWN, "local", {})

    @app.post("/check")
    async def check(request: Request):
        return check_render_fingerprint(request) or Response(status_code=200)

    response = TestClient(app).post("/check", headers=headers)
    return response.status_code, response.json() if response.content else {}


def test_mismatch_is_rejected_and_names_both_sides():
    status, body = _status({RENDER_FINGERPRINT_HEADER: OTHER_BUILD})

    assert status == 409
    assert (body["error"]["received"], body["error"]["expected"]) == (OTHER_BUILD, OWN)


def test_match_passes():
    assert _status({RENDER_FINGERPRINT_HEADER: OWN})[0] == 200


@pytest.mark.parametrize(
    "flags",
    [{"fingerprint_compare": "inputs"}, {"on_fingerprint_mismatch": "warn"}],
)
def test_other_build_is_accepted_only_when_configured(flags):
    assert _status({RENDER_FINGERPRINT_HEADER: OTHER_BUILD}, **flags)[0] == 200


@pytest.mark.parametrize(
    "received",
    [
        "not-a-fingerprint",
        f"rf1:{DIGEST[:-1]}:vllm-py/1.0",
        f"rf1:{DIGEST}:tag with space",
        f"rf1:{DIGEST}:{'x' * 300}",
    ],
)
def test_malformed_fingerprint_is_rejected_and_not_echoed(received):
    """The header is client input, so `warn` does not let it through."""
    status, body = _status(
        {RENDER_FINGERPRINT_HEADER: received}, on_fingerprint_mismatch="warn"
    )

    assert status == 400
    assert "received" not in body["error"]


@pytest.mark.parametrize(
    "flags",
    [{"fingerprint_compare": "inputs"}, {"on_fingerprint_mismatch": "warn"}],
)
def test_accepted_pairs_log_at_a_bounded_rate(flags, caplog_vllm, monkeypatch):
    """A caller that sends a new value per request cannot fill the log."""
    monkeypatch.setattr(fingerprint, "_last_logged", {})
    for build in ("0.1.0", "0.1.1", "0.1.2"):
        received = OTHER_BUILD.replace("0.1.0", build)
        assert _status({RENDER_FINGERPRINT_HEADER: received}, **flags)[0] == 200

    logged = [r for r in caplog_vllm.records if "accepted: received" in r.message]
    assert len(logged) == 1


@pytest.mark.parametrize(("required", "expected"), [(False, 200), (True, 400)])
def test_missing_fingerprint_is_rejected_only_when_required(required, expected):
    assert _status({}, require_render_fingerprint=required)[0] == expected
