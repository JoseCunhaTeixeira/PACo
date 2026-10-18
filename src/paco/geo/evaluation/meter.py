"""A play's model calls, metered (docs/geo_plan.md §12, "Cost"): the agent's client wrapped, each
call and each form it fills counted with its tokens, those the API read from its cache apart. The
same for both agents, whose own records keep no cached tokens (PACo's transcript, the study's usage
log); on a frontier API they bill at a tenth of the others, so the money says them apart."""

from typing import Any

from openai.types.chat import ChatCompletionFunctionToolParam, ChatCompletionMessageParam

from paco.agent.model import ChatModel, Filled, Reply


class Meter:
    """`model`, its calls counted: PACo's ChatModel, its other attributes (name, vision,
    thinking) and keyword options passed through."""

    def __init__(self, model: ChatModel) -> None:
        self._model = model
        self.calls = 0  # the model's calls and the forms it filled
        self.prompt_tokens = 0  # every prompt token read, the cached ones included
        self.cached_tokens = 0
        self.completion_tokens = 0

    async def __call__(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionFunctionToolParam],
        **options: Any,  # noqa: ANN401
    ) -> Reply:
        reply = await self._model(messages, tools, **options)
        self._count(reply)
        return reply

    async def fill(
        self, messages: list[ChatCompletionMessageParam], schema: dict[str, Any]
    ) -> Filled:
        filled = await self._model.fill(messages, schema)
        self._count(filled)
        return filled

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        return getattr(self._model, name)

    def _count(self, used: Reply | Filled) -> None:
        self.calls += 1
        self.prompt_tokens += used.prompt_tokens or 0
        self.completion_tokens += used.completion_tokens or 0
        self.cached_tokens += getattr(used, "cached_tokens", None) or 0
