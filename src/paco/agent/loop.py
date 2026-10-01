"""The agent loop: the model asks for tools, the host calls PACo's server, until the model
answers the user."""

import json
import textwrap
import time
import uuid
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Any

import anyio
from mcp import Client
from mcp.types import RequestParamsMeta
from openai.types.chat import ChatCompletionFunctionToolParam, ChatCompletionMessageParam

from paco import prompts
from paco.agent import host
from paco.agent.conversion import result_for_model, tools_for_model
from paco.agent.model import ChatModel, Reply, ToolCall
from paco.agent.record import ModelStep, ScopeStep, Step, ToolStep, Transcript
from paco.agent.scope import (
    Context,
    Offer,
    Read,
    ScopeError,
    chosen,
    offers_in,
    read_scope,
    refusal,
)

# The answer a stopped question leaves in the conversation (see Agent.answer).
STOPPED = "(Stopped on the user's request before the answer was complete.)"

# The role the model plays: prompts/role.md.
ROLE = prompts.prompt("role")
# The answer when the model filled no valid scope form, twice.
UNREAD = "I could not read what your message asks ({error}). Please say it again in other words."

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
        # Sent with every call: the conversation, its turn (the user's messages so far) and the
        # scope of the turn's message. The server goes on with what it did in this conversation
        # without asking, and applies the user's rules with the scope (paco.choices).
        self.meta: RequestParamsMeta = {"conversation": uuid.uuid4().hex, "turn": 0}
        # What the host knows for the next message: the options a tool offered last, and the
        # profile and run the conversation is on.
        self._offers: tuple[Offer, ...] = ()
        self._profile: str | None = None
        self._run_id: str | None = None
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
        """The model's answer to `question`, after every tool call it asked for within the
        question's scope (paco.agent.scope), with what the host guarantees (paco.agent.host):
        the scope first, the parameters used and the settings the gates changed listed after
        it, and an inversion it started followed to its end.

        An answer that does not end (the host stopped it, or it failed) leaves in the
        conversation the question and one line saying so, none of the calls it had made: the
        next question starts from a history the model can read, no call left unanswered."""
        start = len(self.messages)
        self.meta["turn"] += 1
        try:
            return await self._answer(question)
        except BaseException as error:
            said = (
                STOPPED
                if isinstance(error, anyio.get_cancelled_exc_class())
                else f"(No answer: {type(error).__name__}: {error})"
            )
            del self.messages[start:]
            self.messages.append({"role": "user", "content": question})
            self.messages.append({"role": "assistant", "content": said})
            raise

    async def _answer(self, question: str) -> str:
        context = Context(self._offers, self._profile, self._run_id)
        self._offers = ()  # answered by this message, or left
        start = time.perf_counter()
        try:
            read = await read_scope(self._model, question, context)
        except ScopeError as error:
            self.steps.append(_scope_step(None, time.perf_counter() - start, str(error)))
            answer = UNREAD.format(error=error)
            self.messages.append({"role": "user", "content": question})
            self.messages.append({"role": "assistant", "content": answer})
            return answer
        self.steps.append(_scope_step(read, time.perf_counter() - start))
        scope = read.scope
        offer = chosen(scope, context)
        self.meta["scope"] = {**scope.for_server(), "chosen": offer.call if offer else None}
        self.messages.append({"role": "user", "content": f"{question}\n\n{scope.for_model(offer)}"})
        calls = 0
        failed: dict[tuple[str, str], str] = {}  # calls that failed in this answer: their error
        changes: list[str] = []
        used: list[str] = []
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
                answer = f"{scope.line()}\n\n{host.with_changes(reply.content, changes, used)}"
                self.messages[-1] = {"role": "assistant", "content": answer}
                return answer
            for call in reply.tool_calls:
                calls += 1
                key = (call.name, _canonical(call.arguments))
                step = await self._call(
                    call,
                    over_budget=calls > self._max_tool_calls,
                    failed_before=failed.get(key),
                    outside=refusal(scope, call.name, call.arguments),
                )
                self.steps.append(step)
                result = step.result
                job_id = (
                    None if step.is_error else host.started_job(call.name, call.arguments, result)
                )
                if job_id is not None:
                    # The model reads the job's end, not its start: it answers once it is done.
                    followed = await self._follow(job_id)
                    self.steps.append(followed)
                    result = followed.result
                for listed, items in (
                    (changes, host.changed_items(result)),
                    (used, host.used_items(result)),
                ):
                    listed.extend(item for item in items if item not in listed)
                if step.is_error:
                    failed[key] = step.result
                else:
                    failed.pop(key, None)
                    self._offers = offers_in(result) or self._offers
                    self._keep_track(call, result)
                self.messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    @property
    def offers(self) -> tuple[Offer, ...]:
        """The options a tool offered in the last answer, among which the user's next message
        may choose: that answer asks the user to choose."""
        return self._offers

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
        outside: str | None = None,
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

        if outside is not None:
            self._on_event(f"-> {call.name}({call.arguments}) refused: outside the scope")
            return refused(outside)
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
        result = await self._client.call_tool(
            call.name, parsed, progress_callback=on_progress, meta=self.meta
        )
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

    def _keep_track(self, call: ToolCall, result: str) -> None:
        """The profile and run the conversation is on, from a call and its result."""
        for text in (call.arguments, result):
            try:
                parsed: Any = json.loads(text or "{}")
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue
            if isinstance(profile := parsed.get("profile"), str):
                self._profile = profile
            if isinstance(run_id := parsed.get("run_id"), str) and run_id:
                self._run_id = run_id

    async def _follow(self, job_id: str) -> ToolStep:
        """Job `job_id` followed with job_status until it ends, its progress sent as events:
        the step holds its final status."""
        arguments = json.dumps({"job_id": job_id})

        async def on_progress(progress: float, total: float | None, message: str | None) -> None:
            self._on_event(f"   job_status: {message or f'{progress:g} of {total:g}'}")

        self._on_event(f"-> job_status({arguments})")
        start = time.perf_counter()
        while True:
            result = await self._client.call_tool(
                "job_status", {"job_id": job_id}, progress_callback=on_progress, meta=self.meta
            )
            text = result_for_model(result)
            if result.is_error or not host.job_running(text):
                break
        if result.is_error:
            message = text.removeprefix("Error executing tool job_status: ")
            self._on_event(textwrap.indent(f"failed: {message}", "   "))
        return ToolStep(
            name="job_status",
            arguments=arguments,
            called=True,
            is_error=bool(result.is_error),
            duration_s=round(time.perf_counter() - start, 3),
            result=text,
            by_host=True,
        )


def _scope_step(read: Read | None, duration_s: float, error: str | None = None) -> ScopeStep:
    fills = read.fills if read is not None else ()
    return ScopeStep(
        prompt_version=prompts.version(),
        scope=read.scope.model_dump() if read is not None else None,
        error=error,
        tries=len(fills) if read is not None else 2,
        duration_s=round(duration_s, 3),
        prompt_tokens=_total(fill.prompt_tokens for fill in fills),
        completion_tokens=_total(fill.completion_tokens for fill in fills),
    )


def _total(counts: Iterable[int | None]) -> int | None:
    known = [count for count in counts if count is not None]
    return sum(known) if known else None


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
