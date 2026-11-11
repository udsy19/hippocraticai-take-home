"""Alternative backend: the same `LLM` interface on the Claude Agent SDK.

The Agent SDK drives the local Claude Code runtime and uses its login, so the
pipeline can run without ANTHROPIC_API_KEY. Tools are switched off: each call
is a single plain completion, exactly like the Messages API backend.
"""
import asyncio
from typing import TypeVar

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKError, ResultMessage, query
from pydantic import BaseModel, ValidationError

from .llm import MODEL, TIMEOUT_SECONDS, ModelDeclined, ModelUnavailable

T = TypeVar("T", bound=BaseModel)


class AgentSDKLLM:
    def text(self, system: str, prompt: str, effort: str = "medium") -> str:
        result = self._call(system, prompt, effort, schema=None)
        if not result.result:
            raise ModelDeclined("empty reply")
        return result.result.strip()

    def structured(
        self, system: str, prompt: str, schema: type[T], effort: str = "medium"
    ) -> T:
        result = self._call(system, prompt, effort, schema)
        try:
            return schema.model_validate(result.structured_output)
        except ValidationError as err:
            raise ModelDeclined("reply did not match the expected schema") from err

    def _call(self, system, prompt, effort, schema) -> ResultMessage:
        try:
            return asyncio.run(
                asyncio.wait_for(
                    self._run(system, prompt, effort, schema), TIMEOUT_SECONDS * 2
                )
            )
        except (ClaudeSDKError, asyncio.TimeoutError, OSError) as err:
            raise ModelUnavailable(f"The Claude Agent SDK call failed: {err}") from err

    async def _run(
        self, system: str, prompt: str, effort: str, schema: type[BaseModel] | None
    ) -> ResultMessage:
        options = ClaudeAgentOptions(
            model=MODEL,
            system_prompt=system,
            effort=effort,
            tools=[],  # a pure completion: no file, shell or web access
            setting_sources=[],  # ignore local project and user settings
            # Structured output is delivered through an internal tool call,
            # which takes an extra turn.
            max_turns=4 if schema else 1,
            output_format=(
                {"type": "json_schema", "schema": schema.model_json_schema()}
                if schema else None
            ),
        )
        result = None
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, ResultMessage):
                result = message
        if result is None or result.is_error:
            raise ModelDeclined(getattr(result, "subtype", "no result"))
        return result
