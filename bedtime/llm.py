"""Thin wrapper around the Anthropic Messages API.

The pipeline depends only on the two methods of `LLM`, so tests swap in a fake
and never touch the network.
"""
from typing import Protocol, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

MODEL = "claude-haiku-5-5"  # fixed by the assignment
MAX_TOKENS = 16000  # includes thinking, so leave real headroom

T = TypeVar("T", bound=BaseModel)


class StoryError(Exception):
    """Base class for everything the pipeline can fail with."""


class InvalidRequest(StoryError, ValueError):
    """The caller's input is out of bounds. Not retried; the message is shown."""


class ModelDeclined(StoryError):
    """The model refused, or its reply was unusable. Retried once where safe."""


class ModelUnavailable(StoryError):
    """The provider could not be reached or rejected the call. Not retried here
    (the SDK already retried transient errors); the run stops cleanly."""


TIMEOUT_SECONDS = 120  # per request; the SDK retries timeouts and 429/5xx twice


class LLM(Protocol):
    def text(self, system: str, prompt: str, effort: str = "medium") -> str: ...

    def structured(
        self, system: str, prompt: str, schema: type[T], effort: str = "medium"
    ) -> T: ...


class ClaudeLLM:
    def __init__(self, client: anthropic.Anthropic | None = None):
        # Reads ANTHROPIC_API_KEY from the environment; no key lives in the code.
        self.client = client or anthropic.Anthropic(
            timeout=TIMEOUT_SECONDS, max_retries=2
        )

    def text(self, system: str, prompt: str, effort: str = "medium") -> str:
        # No temperature: Haiku 5.5 rejects non-default sampling parameters, so
        # variety and restraint are steered from the prompts instead.
        try:
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": effort},
            )
        except anthropic.APIError as err:
            raise _unavailable(err) from err
        _check(response)
        # Thinking is on by default, so pick text blocks by type, not position.
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        if not text:
            raise ModelDeclined("empty reply")
        return text

    def structured(
        self, system: str, prompt: str, schema: type[T], effort: str = "medium"
    ) -> T:
        try:
            response = self.client.messages.parse(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": effort},
                output_format=schema,
            )
        except ValidationError as err:
            raise ModelDeclined("reply did not match the expected schema") from err
        except anthropic.APIError as err:
            raise _unavailable(err) from err
        _check(response)
        if response.parsed_output is None:
            raise ModelDeclined("reply did not match the expected schema")
        return response.parsed_output


def _check(response) -> None:
    if response.stop_reason == "refusal":
        raise ModelDeclined("request declined by the model's safety layer")
    if response.stop_reason == "max_tokens":
        raise ModelDeclined("reply was cut off before it finished")


def _unavailable(err: anthropic.APIError) -> ModelUnavailable:
    if isinstance(err, anthropic.AuthenticationError):
        return ModelUnavailable("The Anthropic API rejected ANTHROPIC_API_KEY.")
    if isinstance(err, anthropic.APIConnectionError):  # includes timeouts
        return ModelUnavailable("Couldn't reach the Anthropic API.")
    status = getattr(err, "status_code", "unknown")
    return ModelUnavailable(f"The Anthropic API returned an error ({status}).")
