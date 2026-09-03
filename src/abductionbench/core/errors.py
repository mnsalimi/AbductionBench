"""Error taxonomy for the core engine.

Classifying failures is what makes the retry policy meaningful: a rate limit
should be waited out, a dropped Cloudflare tunnel should trigger endpoint
re-discovery, and a request that is structurally invalid (e.g. one conversation
in a batch exceeds the model's context window) must *not* be retried as-is --
it has to be isolated by splitting the batch, because vLLM fails the whole
batch call when any single conversation in it is bad.
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "ErrorClass",
    "AbenchError",
    "ConfigError",
    "AdapterError",
    "TemplateError",
    "EndpointError",
    "TransientEndpointError",
    "RateLimitError",
    "AuthError",
    "InvalidRequestError",
    "ContextLengthError",
    "BatchProtocolError",
    "classify_status_code",
]


class ErrorClass(str, Enum):
    """How the engine should react to a failure."""

    #: Network hiccup, 5xx, timeout, tunnel drop -> retry with backoff.
    TRANSIENT = "transient"
    #: 429 / explicit rate limit -> retry after a longer, dedicated backoff.
    RATE_LIMIT = "rate_limit"
    #: Bad credentials -> abort the task, retrying cannot help.
    AUTH = "auth"
    #: Request is invalid for reasons that may be sample-specific -> bisect the
    #: batch to find the offending sample, then mark just that one failed.
    INVALID_REQUEST = "invalid_request"
    #: A conversation does not fit the model context -> bisect, then skip.
    CONTEXT_LENGTH = "context_length"
    #: Response did not match the batch protocol (wrong choice count etc.).
    PROTOCOL = "protocol"
    #: Anything unrecognized -> retried conservatively, then failed.
    UNKNOWN = "unknown"


class AbenchError(Exception):
    """Base class for all AbductionBench errors."""

    error_class: ErrorClass = ErrorClass.UNKNOWN


class ConfigError(AbenchError):
    """Malformed or inconsistent configuration."""


class AdapterError(AbenchError):
    """A child adapter failed (data missing, unreadable, etc.)."""


class TemplateError(AbenchError):
    """A prompt template is missing, malformed, or missing required variables."""


class EndpointError(AbenchError):
    """Base class for inference endpoint failures."""

    def __init__(self, message: str, *, status_code: int | None = None, body: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class TransientEndpointError(EndpointError):
    error_class = ErrorClass.TRANSIENT


class RateLimitError(EndpointError):
    error_class = ErrorClass.RATE_LIMIT


class AuthError(EndpointError):
    error_class = ErrorClass.AUTH


class InvalidRequestError(EndpointError):
    error_class = ErrorClass.INVALID_REQUEST


class ContextLengthError(InvalidRequestError):
    error_class = ErrorClass.CONTEXT_LENGTH


class BatchProtocolError(EndpointError):
    error_class = ErrorClass.PROTOCOL


#: Substrings that identify a context-length rejection across vLLM/OpenAI-style
#: servers.  Kept here (not in the client) so it is easy to extend.
_CONTEXT_LENGTH_MARKERS = (
    "maximum context length",
    "context_length_exceeded",
    "longer than the maximum",
    "reduce the length",
    "please reduce",
    "max_model_len",
    "token limit",
)

#: Cloudflare quick-tunnel and proxy failure markers -- these look like HTTP
#: errors but mean "the tunnel went away", i.e. transient + worth re-discovering.
_TUNNEL_MARKERS = (
    "error code: 1033",
    "error code: 502",
    "error code: 504",
    "argo tunnel",
    "cloudflared",
    "tunnel error",
    "bad gateway",
    "gateway time-out",
    "gateway timeout",
    "no tunnel",
    "web server is down",
)


def classify_status_code(status_code: int, body: str) -> EndpointError:
    """Map an HTTP failure onto the taxonomy above.

    Parameters
    ----------
    status_code:
        HTTP status of the failed response.
    body:
        Response body (truncated by the caller); used to distinguish a
        context-length rejection from other 400s and a tunnel outage from a
        genuine server error.
    """
    lowered = (body or "").lower()
    snippet = (body or "")[:2000]

    if status_code == 429:
        return RateLimitError(f"rate limited (HTTP 429): {snippet}", status_code=429, body=snippet)
    if status_code in (401, 403):
        return AuthError(
            f"authentication/authorization failed (HTTP {status_code}): {snippet}",
            status_code=status_code,
            body=snippet,
        )
    if status_code == 404:
        # A 404 on the batch route usually means the pass-through entry for this
        # model is not enabled on the gateway -- a configuration problem, not a
        # transient one.
        return InvalidRequestError(
            f"endpoint not found (HTTP 404): {snippet}", status_code=404, body=snippet
        )
    if any(marker in lowered for marker in _CONTEXT_LENGTH_MARKERS):
        return ContextLengthError(
            f"context length exceeded (HTTP {status_code}): {snippet}",
            status_code=status_code,
            body=snippet,
        )
    if status_code == 400 or status_code == 422:
        return InvalidRequestError(
            f"invalid request (HTTP {status_code}): {snippet}", status_code=status_code, body=snippet
        )
    if status_code in (408, 409, 425, 500, 502, 503, 504, 520, 521, 522, 523, 524, 530):
        return TransientEndpointError(
            f"transient endpoint failure (HTTP {status_code}): {snippet}",
            status_code=status_code,
            body=snippet,
        )
    if any(marker in lowered for marker in _TUNNEL_MARKERS):
        return TransientEndpointError(
            f"tunnel failure (HTTP {status_code}): {snippet}", status_code=status_code, body=snippet
        )
    return EndpointError(
        f"unexpected endpoint failure (HTTP {status_code}): {snippet}",
        status_code=status_code,
        body=snippet,
    )
