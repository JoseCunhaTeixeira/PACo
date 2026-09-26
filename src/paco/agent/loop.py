"""The agent loop: the model asks for tools, the host calls PACo's server, until the model
answers the user."""

import json
import textwrap
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from mcp import Client
from openai.types.chat import ChatCompletionFunctionToolParam, ChatCompletionMessageParam

from paco.agent import host
from paco.agent.conversion import result_for_model, tools_for_model
from paco.agent.model import ChatModel, Reply, ToolCall
from paco.agent.record import ModelStep, Step, ToolStep, Transcript

ROLE = (
    "You are PACo's assistant. You help a geophysicist turn MASW seismic profiles into "
    "dispersion curves and velocity models, with the tools you have. Work in a loop: plan the "
    "stages the request needs, act by calling a tool, observe its summary, adapt (go on, or go "
    "back with redo when a gate asks a change of an earlier stage), until the request is done. "
    "Do what the user asks, all of it and nothing more; the user cannot call the tools. Decide "
    "from the summaries. The window length is yours to choose when the user gave none: "
    "run_processing proposes one and lists the lengths it tried, for the line's length and the "
    "depth or detail the request needs. In your answer, report every item of the results' "
    "changed lists (the settings the gates changed, the user's among them) and the windows left "
    "without a result. "
    "Ask the user only when the request cannot be finished: no image or no curve left, the "
    "run's retry budget spent before the request is done, or a request the data do not allow; "
    "then ask one short question with 2 or 3 concrete options, your choice first, and wait. "
    "Rejected windows are gaps to report, not a reason to ask. Otherwise end with the answer: "
    "no offer, no question. Report only what the tools return: never invent a result."
)

# What the loop does, for the user to follow: tool calls, progress, failures.
type OnEvent = Callable[[str], None]


class Agent:
    """One conversation between the user, the model and PACo's server."""

    def __init__(
        self,
        client: Client,
        model: ChatModel,
        tools: list[ChatCompletionFunctionToolParam],
        instructions: str | None,
        max_tool_calls: int = 15,
        on_event: OnEvent = print,
    ) -> None:
        self._client = client
        self._model = model
        self._tools = tools
        self._max_tool_calls = max_tool_calls
        self._on_event = on_event
        system = f"{ROLE}\n\n{instructions}" if instructions else ROLE
        self.messages: list[ChatCompletionMessageParam] = [{"role": "system", "content": system}]
        self.steps: list[Step] = []
        self.started_at = datetime.now(UTC)

    @classmethod
    async def start(
        cls, client: Client, model: ChatModel, max_tool_calls: int = 15, on_event: OnEvent = print
    ) -> Agent:
        """An agent with the tools and instructions of the server `client` is connected to."""
        tools = tools_for_model((await client.list_tools()).tools)
        return cls(client, model, tools, client.instructions, max_tool_calls, on_event)

    async def answer(self, question: str) -> str:
        """The model's answer to `question`, after every tool call it asked for, with what the
        host guarantees (paco.agent.host): the settings the gates changed listed after it, an
        answer that asks or offers asked again once, no inversion the user did not ask for,
        seismic or petrophysical."""
        self.messages.append({"role": "user", "content": question})
        calls = 0
        failed: dict[tuple[str, str], str] = {}  # calls that failed in this answer: their error
        changes: list[str] = []
        stuck = asked_again = worked = False
        while True:
            start = time.perf_counter()
            reply = await self._model(self.messages, self._tools)
            self.steps.append(
                ModelStep(
                    duration_s=round(time.perf_counter() - start, 3),
                    prompt_tokens=reply.prompt_tokens,
                    completion_tokens=reply.completion_tokens,
                    tool_calls=len(reply.tool_calls),
                )
            )
            self.messages.append(_assistant_message(reply))
            if not reply.tool_calls:
                if worked and not (stuck or asked_again) and host.asks_or_offers(reply.content):
                    asked_again = True
                    self._on_event("   (asked to answer again, without a question or an offer)")
                    self.messages.append({"role": "user", "content": host.ANSWER_AGAIN})
                    continue
                answer = host.with_changes(reply.content, changes)
                self.messages[-1] = {"role": "assistant", "content": answer}
                return answer
            for call in reply.tool_calls:
                calls += 1
                key = (call.name, _canonical(call.arguments))
                step = await self._call(
                    call,
                    over_budget=calls > self._max_tool_calls,
                    failed_before=failed.get(key),
                    unasked=host.unasked(call.name, call.arguments, question),
                )
                for item in host.changed_items(step.result):
                    if item not in changes:
                        changes.append(item)
                stuck = stuck or host.STUCK in step.result
                worked = worked or (call.name in host.STAGE_TOOLS and not step.is_error)
                if step.is_error:
                    failed[key] = step.result
                else:
                    failed.pop(key, None)
                self.steps.append(step)
                self.messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": step.result}
                )

    def transcript(self, model: str) -> Transcript:
        """The conversation so far, and every step with its cost."""
        return Transcript(
            started_at=self.started_at,
            model=model,
            messages=[dict(message) for message in self.messages],
            steps=list(self.steps),
        )

    async def _call(
        self,
        call: ToolCall,
        over_budget: bool,
        failed_before: str | None = None,
        unasked: str | None = None,
    ) -> ToolStep:
        start = time.perf_counter()

        def refused(result: str) -> ToolStep:
            return ToolStep(
                name=call.name,
                arguments=call.arguments,
                called=False,
                is_error=True,
                duration_s=0.0,
                result=result,
            )

        if unasked is not None:
            # The model may start an inversion the request did not ask for.
            self._on_event(f"-> {call.name}({call.arguments}) refused: not asked for")
            return refused(unasked)
        if over_budget:
            return refused(
                f"Not called: this answer already made {self._max_tool_calls} tool calls. "
                "Answer the user with what you have, and say what is left to do."
            )
        if failed_before is not None:
            # The model may send a failed call again and again, then invent a result.
            self._on_event(f"-> {call.name}({call.arguments}) refused: the same call just failed")
            return refused(
                "Not called: this exact call just failed, and would fail again: "
                f"{failed_before[:300]} Change the arguments, call another tool, or answer the "
                "user with what you have."
            )
        try:
            parsed: Any = json.loads(call.arguments or "{}")
        except json.JSONDecodeError as error:
            return refused(
                f"Not called: the arguments of {call.name} are not valid JSON ({error})."
            )
        if not isinstance(parsed, dict):
            return refused(f"Not called: the arguments of {call.name} must be a JSON object.")

        async def on_progress(progress: float, total: float | None, message: str | None) -> None:
            self._on_event(f"   {call.name}: {message or f'{progress:g} of {total:g}'}")

        self._on_event(f"-> {call.name}({call.arguments})")
        result = await self._client.call_tool(call.name, parsed, progress_callback=on_progress)
        text = result_for_model(result)
        if result.is_error:
            # The SDK's prefix repeats the call shown just above.
            message = text.removeprefix(f"Error executing tool {call.name}: ")
            self._on_event(textwrap.indent(f"failed: {message}", "   "))
        return ToolStep(
            name=call.name,
            arguments=call.arguments,
            called=True,
            is_error=bool(result.is_error),
            duration_s=round(time.perf_counter() - start, 3),
            result=text,
        )


def _assistant_message(reply: Reply) -> ChatCompletionMessageParam:
    if not reply.tool_calls:
        return {"role": "assistant", "content": reply.content}
    return {
        "role": "assistant",
        "content": reply.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in reply.tool_calls
        ],
    }


def _canonical(arguments: str) -> str:
    """Tool arguments as a key: the same JSON object written with other spacing or key order is
    the same call."""
    try:
        return json.dumps(json.loads(arguments or "{}"), sort_keys=True)
    except json.JSONDecodeError:
        return arguments
