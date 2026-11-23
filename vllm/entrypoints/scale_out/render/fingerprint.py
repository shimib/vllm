# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compute the server's `render_fingerprint` (layout `rf1`) at startup."""

from argparse import Namespace
from dataclasses import dataclass
from functools import lru_cache
from http import HTTPStatus
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Literal

import huggingface_hub
from fastapi import Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter

from vllm.config import ModelConfig
from vllm.entrypoints.serve.engine.protocol import ErrorInfo
from vllm.logger import init_logger
from vllm.renderers import BaseRenderer
from vllm.renderers.chat_utils import load_chat_template
from vllm.renderers.fingerprint import (
    CONFIG_FILES,
    PROCESSOR_FILES,
    TOKENIZER_FILES,
    build_manifest,
    collect_files,
    fingerprints_match,
    render_fingerprint,
    renderer_tag,
)
from vllm.transformers_utils.chat_templates import (
    get_chat_template_fallback_path,
    get_chat_template_override_path,
)
from vllm.utils.mistral import is_mistral_tokenizer
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)

Presence = Literal["hub", "local"]

RENDER_FINGERPRINT_HEADER = "X-Render-Fingerprint"

_TOKENIZER_PATTERNS = [
    *TOKENIZER_FILES,
    "tokenizer.model.v*",
    "*.tiktoken",
    "*.jinja",
]


@dataclass(frozen=True)
class RenderFingerprint:
    render_fingerprint: str
    presence: Presence
    manifest: dict[str, Any]


def _resolve_snapshot(
    source: str, revision: str | None, patterns: list[str]
) -> tuple[Path, Presence]:
    """Fetch the listed files the Hub names, then hash what the snapshot holds."""
    if Path(source).exists():
        return Path(source), "local"

    offline = huggingface_hub.constants.HF_HUB_OFFLINE
    path = huggingface_hub.snapshot_download(
        source,
        revision=revision,
        allow_patterns=patterns,
        local_files_only=offline,
    )
    return Path(path), "local" if offline else "hub"


def _renderer_name(model_config: ModelConfig, renderer: BaseRenderer) -> str:
    if model_config.hf_config.model_type == "gpt_oss":
        return "harmony"
    renderer_cls = type(renderer)
    if renderer_cls.__name__ == "DeepseekV41Renderer":
        return "deepseek_v41"
    return renderer_cls.__module__.rsplit(".", 1)[-1]


def _tokenizer_library(renderer: BaseRenderer) -> str:
    tokenizer = renderer.get_tokenizer()
    if is_mistral_tokenizer(tokenizer):
        return f"mistral-common/{package_version('mistral_common')}"
    if getattr(tokenizer, "is_fast", False):
        return f"tokenizers/{package_version('tokenizers')}"
    return f"transformers/{package_version('transformers')}"


def _builtin_chat_template(
    model_config: ModelConfig, renderer: BaseRenderer
) -> str | None:
    """Text of the built-in template the HF renderer uses, if it uses one."""
    from vllm.renderers.hf import HfRenderer, resolve_chat_template

    if not isinstance(renderer, HfRenderer):
        return None

    tokenizer = renderer.get_tokenizer()
    model_type = model_config.hf_config.model_type
    override = get_chat_template_override_path(model_type, tokenizer.name_or_path)
    if override is not None:
        return load_chat_template(override)

    fallback = get_chat_template_fallback_path(model_type, tokenizer.name_or_path)
    if fallback is None:
        return None
    text = load_chat_template(fallback)
    resolved = resolve_chat_template(tokenizer, None, None, model_config=model_config)
    return text if resolved == text else None


def _flags(
    model_config: ModelConfig,
    args: Namespace,
    renderer: BaseRenderer,
    chat_template: str | None,
) -> dict[str, Any]:
    mm_config = model_config.multimodal_config
    language_model_only = bool(mm_config and mm_config.language_model_only)
    multimodal_active = mm_config is not None and not language_model_only
    pruning_rate = (mm_config.video_pruning_rate if mm_config else None) or None
    # `--cohere-format` defaults to a value on every server, but only the
    # Cohere renderer reads it.
    template_kwargs = dict(args.default_chat_template_kwargs or {})
    if _renderer_name(model_config, renderer) == "cohere":
        template_kwargs.setdefault("cohere_format", args.cohere_format)

    return {
        "tokenizer_mode": args.tokenizer_mode,
        "config_format": args.config_format,
        "trust_remote_code": args.trust_remote_code,
        "chat_template": chat_template,
        "chat_template_fallback": (
            _builtin_chat_template(model_config, renderer)
            if chat_template is None
            else None
        ),
        "chat_template_content_format": args.chat_template_content_format,
        "default_chat_template_kwargs": template_kwargs or None,
        "reasoning_parser": args.reasoning_parser or None,
        "reasoning_effort_map": args.reasoning_effort_map or None,
        "exclude_tools_when_tool_choice_none": args.exclude_tools_when_tool_choice_none,
        "hf_overrides": args.hf_overrides or None,
        "mm_processor_kwargs": args.mm_processor_kwargs or None,
        "media_io_kwargs": args.media_io_kwargs or None,
        "language_model_only": language_model_only,
        "interleave_mm_strings": bool(mm_config and mm_config.interleave_mm_strings),
        "video_pruning_rate": pruning_rate,
        "video_pruning_method": (
            mm_config.video_pruning_method if mm_config and pruning_rate else None
        ),
        "max_model_len": model_config.max_model_len if multimodal_active else None,
        "trust_request_chat_template": args.trust_request_chat_template,
    }


def compute_render_fingerprint(
    model_config: ModelConfig,
    args: Namespace,
    renderer: BaseRenderer,
    *,
    chat_template: str | None,
) -> RenderFingerprint:
    """Build the manifest from the resolved sources and the render flags.

    Args:
        model_config: The server's model config.
        args: Parsed CLI arguments; flags are recorded as configured.
        renderer: The renderer that produces prompts.
        chat_template: Text of `--chat-template`, if given.

    """
    tokenizer_dir, tokenizer_presence = _resolve_snapshot(
        model_config.tokenizer,
        model_config.tokenizer_revision,
        [*_TOKENIZER_PATTERNS, "additional_chat_templates/*.jinja"],
    )
    config_dir, config_presence = _resolve_snapshot(
        model_config.hf_config_path or model_config.model,
        model_config.revision,
        list(CONFIG_FILES),
    )
    model_dir, model_presence = _resolve_snapshot(
        model_config.model, model_config.revision, list(PROCESSOR_FILES)
    )

    manifest = build_manifest(
        collect_files(tokenizer_dir, config_dir, model_dir),
        _flags(model_config, args, renderer, chat_template),
    )
    tag = renderer_tag(
        version=VLLM_VERSION,
        renderer=_renderer_name(model_config, renderer),
        tokenizer=_tokenizer_library(renderer),
    )
    hub = "hub" in (tokenizer_presence, config_presence, model_presence)
    return RenderFingerprint(
        render_fingerprint=render_fingerprint(manifest, tag),
        presence="hub" if hub else "local",
        manifest=manifest,
    )


@lru_cache(maxsize=1)
def _checks_counter() -> Counter:
    return Counter(
        name="vllm:render_fingerprint_checks",
        documentation=(
            "Requests carrying prompt token IDs, by the outcome of the "
            "render fingerprint check."
        ),
        labelnames=["result"],
    )


def _error(status: HTTPStatus, err_type: str, message: str, **extra: str):
    error = ErrorInfo(message=message, type=err_type, code=status.value)
    content = {"error": {**error.model_dump(), **extra}}
    return JSONResponse(content=content, status_code=status.value)


def check_render_fingerprint(raw_request: Request) -> JSONResponse | None:
    """Compare the caller's `X-Render-Fingerprint` with this server's.

    For routes that carry token IDs only, so `rerender` rejects like `reject`.

    Returns:
        An error response when the request must be rejected, else None.

    """
    state = raw_request.app.state
    own: RenderFingerprint | None = getattr(state, "render_fingerprint", None)
    if own is None:
        return None

    args = state.args
    expected = own.render_fingerprint
    received = raw_request.headers.get(RENDER_FINGERPRINT_HEADER)
    if received is None:
        _checks_counter().labels("absent").inc()
        if not args.require_render_fingerprint:
            return None
        return _error(
            HTTPStatus.BAD_REQUEST,
            "RenderFingerprintRequired",
            f"This server requires the {RENDER_FINGERPRINT_HEADER} header on "
            "requests that carry prompt token IDs.",
            expected=expected,
        )

    if fingerprints_match(received, expected, args.fingerprint_compare):
        _checks_counter().labels("match").inc()
        return None

    _checks_counter().labels("mismatch").inc()
    if args.on_fingerprint_mismatch == "warn":
        logger.warning_once(
            "Render fingerprint mismatch accepted (--on-fingerprint-mismatch=warn): "
            "received %s, expected %s",
            received,
            expected,
        )
        return None
    return _error(
        HTTPStatus.CONFLICT,
        "RenderFingerprintMismatch",
        "The caller's render fingerprint does not match this server's. Compare "
        "GET /render_fingerprint on both sides to see which input differs.",
        received=received,
        expected=expected,
    )


def render_fingerprint_headers(raw_request: Request) -> dict[str, str]:
    own = getattr(raw_request.app.state, "render_fingerprint", None)
    return {RENDER_FINGERPRINT_HEADER: own.render_fingerprint} if own else {}
