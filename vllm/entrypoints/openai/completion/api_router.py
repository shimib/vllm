# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionStreamResponse,
)
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.scale_out.render.fingerprint import (
    check_render_fingerprint,
    render_fingerprint_headers,
)
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.utils.api_utils import (
    json_or_sse_responses,
    load_aware_call,
    validate_json_request,
    with_cancellation,
)
from vllm.entrypoints.serve.utils.orca_metrics import metrics_header
from vllm.entrypoints.serve.utils.sse_keep_alive import with_sse_keep_alive
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()
ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL = "endpoint-load-metrics-format"


def completion(request: Request) -> OpenAIServingCompletion | None:
    return request.app.state.openai_serving_completion


@router.post(
    "/v1/completions",
    dependencies=[Depends(validate_json_request)],
    **json_or_sse_responses(
        CompletionResponse,
        CompletionStreamResponse,
        {
            HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
            HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
            HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
        },
    ),
)
@with_cancellation
@load_aware_call
async def create_completion(request: CompletionRequest, raw_request: Request):
    metrics_header_format = raw_request.headers.get(
        ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL, ""
    )
    handler = completion(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support Completions API")

    # A prompt given as token IDs was rendered elsewhere.
    prompt = request.prompt
    is_token_prompt = (
        isinstance(prompt, list) and bool(prompt) and not isinstance(prompt[0], str)
    )
    fingerprint_headers: dict[str, str] = {}
    if is_token_prompt:
        if (rejection := check_render_fingerprint(raw_request)) is not None:
            return rejection
        fingerprint_headers = render_fingerprint_headers(raw_request)

    generator = await handler.create_completion(request, raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )
    elif isinstance(generator, CompletionResponse):
        return JSONResponse(
            content=generator.model_dump(),
            headers={
                **(metrics_header(metrics_header_format) or {}),
                **fingerprint_headers,
            },
        )

    args = getattr(raw_request.app.state, "args", None)
    keep_alive_interval = getattr(args, "sse_keep_alive_interval", 0)
    return StreamingResponse(
        content=with_sse_keep_alive(generator, float(keep_alive_interval)),
        media_type="text/event-stream",
        headers=fingerprint_headers,
    )


def attach_router(app: FastAPI):
    app.include_router(router)
