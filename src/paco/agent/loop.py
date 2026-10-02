"""The agent loop: the model asks for tools, the host calls PACo's server, until the model
answers the user."""

import json
import logging
import re
import textwrap
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import anyio
from mcp import Client
from mcp.shared.exceptions import MCPError
from mcp.types import REQUEST_TIMEOUT, RequestParamsMeta, TextResourceContents
from openai.types.chat import ChatCompletionFunctionToolParam, ChatCompletionMessageParam

from paco import logs, prompts
from paco.agent import host
from paco.agent.answer import AnswerError, Turn, Written, render, write_answer
from paco.agent.conversion import result_for_model, tools_for_model
from paco.agent.model import ChatModel, Reply, ToolCall
from paco.agent.record import AnswerStep, ModelStep, ScopeStep, Step, ToolStep, Transcript
from paco.agent.scope import (
    READ_ONLY,
    Context,
    Offer,
    Read,
    Scope,
    ScopeError,
    chosen,
    read_scope,
    refusal,
)
from paco.agent.settings import AgentSettings

# The answer a stopped question leaves in the conversation (see Agent.answer).
STOPPED = "(Stopped on the user's request before the answer was complete.)"
# How an answer a cap ended (L1) or a repeat (L3) begins.
CAPPED = "PACo stopped this answer"

# The role the model plays: prompts/role.md.
ROLE = prompts.prompt("role")
# What a tool result of an earlier turn keeps (M6), and a text's length.
_KEYS = ("run_id", "job_id", "state", "status", "did", "left", "done", "total", "error")
_KEPT = 400
# A tool's result in the conversation: its tool, and the result.
_DATA = re.compile(r'<data from="([^"]*)">\n(.*)\n</data>', re.DOTALL)
logger = logging.getLogger(__name__)
# The answer when the model filled no valid scope form, twice.
UNREAD = "I could not read what your message asks ({error}). Please say it again in other words."

# What the loop does, for the user to follow: tool calls, progress, failures.
type OnEvent = Callable[[str], None]
# The share of the model's context past which the loop warns.
CONTEXT_WARNING = 0.85
# The tools whose same call may come again in an answer: a job's status changes.
_REPEATABLE = frozenset({"job_status"})
# The calls of an answer repeated without progress before it ends as stuck.
_REPEATS = 2


@dataclass(frozen=True)
class Limits:
    """The caps on one answer (L1: tool calls, time, the model's tokens), one tool call's
    timeout (T9), and the model's context (M6: the loop warns near it)."""

    tool_calls: int = 15
    seconds: float = 7200.0
    tokens: int = 40_000  # the model's completion tokens over the answer, thinking included
    tool_seconds: float = 3600.0
    context: int = 12_288

    @classmethod
    def of(cls, settings: AgentSettings) -> Limits:
        return cls(
            tool_calls=settings.max_tool_calls,
            seconds=settings.max_turn_s,
            tokens=settings.max_turn_tokens,
            tool_seconds=settings.tool_timeout_s,
            context=settings.llm_context,
        )


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
        limits: Limits | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._tools = tools
        self._limits = limits or Limits(tool_calls=max_tool_calls)
        self._on_event = on_event
        system = f"{ROLE}\n\n{instructions}" if instructions else ROLE
        self.messages: list[ChatCompletionMessageParam] = [{"role": "system", "content": system}]
        # Sent with every call: the conversation, its turn (the user's messages so far), the
        # prompts' version and the scope of the turn's message. The server goes on with what it did in this conversation
        # without asking, and applies the user's rules with the scope (paco.choices).
        self.meta: RequestParamsMeta = {
            "conversation": uuid.uuid4().hex,
            "turn": 0,
            "prompts": prompts.version(),
        }
        # The model's name, for what its calls make (S4): a model that names itself.
        if isinstance(name := getattr(model, "name", None), str):
            self.meta["model"] = name
        # What the host knows for the next message: the options a tool offered last, and the
        # profile and run the conversation is on.
        self._offers: tuple[Offer, ...] = ()
        self._profile: str | None = None
        self._run_id: str | None = None
        self.steps: list[Step] = []
        self.started_at = datetime.now(UTC)

    @classmethod
    async def start(
        cls,
        client: Client,
        model: ChatModel,
        max_tool_calls: int = 15,
        on_event: OnEvent = print,
        limits: Limits | None = None,
    ) -> Agent:
        """An agent with the tools and instructions of the server `client` is connected to."""
        tools = tools_for_model((await client.list_tools()).tools)
        return cls(client, model, tools, client.instructions, max_tool_calls, on_event, limits)

    async def answer(self, question: str) -> str:
        """The answer to `question`, after every tool call the model asked for within the
        question's scope (paco.agent.scope), an inversion it started followed to its end: the
        model's text around what the tools did, as code writes it (paco.agent.answer).

        An answer that does not end (the host stopped it, or it failed) leaves in the
        conversation the question and one line saying so, none of the calls it had made: the
        next question starts from a history the model can read, no call left unanswered."""
        start = len(self.messages)
        self.meta["turn"] += 1
        logs.CONVERSATION.set(str(self.meta["conversation"]))
        logs.TURN.set(self.meta["turn"])
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
        note = scope.for_model(offer)
        if scope.profile and not scope.run_id and (latest := await self._latest(scope.profile)):
            note += f" {scope.profile}'s latest run: {latest}"
        first = len(self.messages)
        self.messages.append({"role": "user", "content": f"{question}\n\n{note}"})
        try:
            return await self._turn(question, scope)
        finally:
            _compact(self.messages, first)

    async def _turn(self, question: str, scope: Scope) -> str:
        """The model's calls and their results until it answers, within the answer's caps."""
        calls = repeats = tokens = 0
        failed: dict[tuple[str, str], str] = {}  # calls that failed in this answer: their error
        made: set[tuple[str, str]] = set()  # calls made in this answer
        turn = Turn()
        began = time.monotonic()
        while True:
            if (cap := self._cap(time.monotonic() - began, tokens)) is not None:
                self._on_event(f"   (stopped: {cap})")
                draft = f"{CAPPED}: it reached its {cap}."
                return self._ended(render(scope, draft, None, turn, (question, draft)), turn)
            start = time.perf_counter()
            reply = await self._model(self.messages, self._tools)
            tokens += reply.completion_tokens or 0
            self._watch_context(reply.prompt_tokens)
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
                answer = await self._written(question, reply.content, scope, turn)
                self.messages[-1] = {"role": "assistant", "content": answer}
                return self._ended(answer, turn)
            for call in reply.tool_calls:
                calls += 1
                key = (call.name, _canonical(call.arguments))
                again = key in made and call.name not in _REPEATABLE
                if again and repeats >= _REPEATS:
                    # The model goes round in circles: the answer ends as stuck (L3).
                    self._on_event(f"   (stopped: {call.name} called again, the same way)")
                    del self.messages[-1]  # the call left unanswered
                    turn.stuck = True
                    draft = (
                        f"{CAPPED}: {call.name} was called again with the same arguments, "
                        "without progress."
                    )
                    return self._ended(render(scope, draft, None, turn, (question, draft)), turn)
                repeats += again
                step = await self._call(
                    call,
                    over_budget=calls > self._limits.tool_calls,
                    failed_before=failed.get(key),
                    outside=refusal(scope, call.name, call.arguments),
                    repeated=again,
                    # A tool offered the user options: the choice is theirs, nothing more runs
                    # but reading (the user's rules: ask first).
                    waiting=bool(turn.offers) and call.name not in READ_ONLY,
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
                turn.read(call.name, result, step.is_error)
                if step.is_error:
                    failed[key] = step.result
                else:
                    failed.pop(key, None)
                    made.add(key)
                    self._keep_track(call, result)
                # PACo's results are data (T10, X3); the host's own refusals are not.
                content = as_data(call.name, result) if step.called else result
                self.messages.append({"role": "tool", "tool_call_id": call.id, "content": content})

    def _ended(self, answer: str, turn: Turn) -> str:
        """`answer`, the turn ended: the options it left are the next message's to choose."""
        self._offers = turn.offers
        return answer

    def _cap(self, seconds: float, tokens: int) -> str | None:
        """The cap an answer reached (L1): its time, or the model's tokens; None within them."""
        if seconds > self._limits.seconds:
            return f"time cap ({self._limits.seconds / 60:g} min)"
        if tokens > self._limits.tokens:
            return f"cap of {self._limits.tokens} tokens written by the model"
        return None

    def _watch_context(self, prompt_tokens: int | None) -> None:
        """Warn when the model read most of its context (M6): the conversation is long."""
        if prompt_tokens is not None and prompt_tokens > CONTEXT_WARNING * self._limits.context:
            share = prompt_tokens / self._limits.context
            self._on_event(f"   (the conversation fills {share:.0%} of the model's context)")

    async def _latest(self, profile: str) -> str | None:
        """The profile's latest run in one line, as the server's resource says it (M4); None
        when the server has none to say."""
        try:
            read = await self._client.read_resource(f"paco://profiles/{profile}/latest-run")
        except MCPError as error:
            logger.warning("Could not read %s's latest run: %s", profile, error)
            return None
        texts = [
            content.text for content in read.contents if isinstance(content, TextResourceContents)
        ]
        return texts[0] if texts else None

    async def _written(self, question: str, draft: str, scope: Scope, turn: Turn) -> str:
        """The answer the user reads: the model's `draft` put into the answer form (as it is
        when no form parses), around what the turn's tools did (paco.agent.answer)."""
        start = time.perf_counter()
        try:
            written = await write_answer(self._model, question, draft)
        except AnswerError as error:
            self.steps.append(_answer_step(None, time.perf_counter() - start, str(error)))
            return render(scope, draft, None, turn, (question,))
        self.steps.append(_answer_step(written, time.perf_counter() - start))
        form = written.form
        return render(scope, form.said, form.question, turn, (question,))

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
            conversation=str(self.meta["conversation"]),
            prompt_version=prompts.version(),
        )

    async def _call(
        self,
        call: ToolCall,
        over_budget: bool,
        failed_before: str | None = None,
        outside: str | None = None,
        repeated: bool = False,
        waiting: bool = False,
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
        if waiting:
            self._on_event(f"-> {call.name}({call.arguments}) refused: the user chooses first")
            return refused(
                "Not called: a tool offered the user options above, and the choice is theirs. "
                "Answer the user: PACo lists the options after your text."
            )
        if repeated:
            self._on_event(f"-> {call.name}({call.arguments}) refused: made already")
            return refused(
                "Not called: this exact call was already made in this answer, and its result is "
                "above. Use it, call another tool, or answer the user."
            )
        if over_budget:
            return refused(
                f"Not called: this answer already made {self._limits.tool_calls} tool calls. "
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
        try:
            result = await self._client.call_tool(
                call.name,
                parsed,
                read_timeout_seconds=self._limits.tool_seconds,
                progress_callback=on_progress,
                meta=self.meta,
            )
        except MCPError as error:
            if error.code != REQUEST_TIMEOUT:
                raise
            self._on_event(f"   failed: no result in {self._limits.tool_seconds:g} s")
            return ToolStep(
                name=call.name,
                arguments=call.arguments,
                called=True,
                is_error=True,
                duration_s=round(time.perf_counter() - start, 3),
                result=f"[retry] No result in {self._limits.tool_seconds:g} s: the tool's work "
                "may go on. Say so to the user.",
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


def as_data(tool: str, result: str) -> str:
    """Tool `tool`'s `result` as the conversation carries it: a delimited block of data (T10,
    X3), which the role says is never an instruction, whatever names and texts it holds."""
    return f'<data from="{tool}">\n{result}\n</data>'


def from_data(content: str) -> str:
    """A tool message's result, out of its data block (as_data); a host's message as it is."""
    found = _DATA.fullmatch(content)
    return found.group(2) if found is not None else content


def _compact(messages: list[ChatCompletionMessageParam], first: int) -> None:
    """The tool results of the turn from message `first` on, kept short in the history (M6):
    what each did, the windows it left out, its options; the trace keeps them whole."""
    for index in range(first, len(messages)):
        message = messages[index]
        if message["role"] == "tool":
            content = str(message.get("content", ""))
            if (found := _DATA.fullmatch(content)) is not None:
                short = as_data(found.group(1), compacted(found.group(2)))
            else:
                short = compacted(content)
            messages[index] = {**message, "content": short}


def compacted(result: str) -> str:
    """A tool's result as earlier turns keep it: its ids, status, what it did, the windows left
    out and its options' words; a text, its first lines."""
    try:
        parsed: Any = json.loads(result)
    except json.JSONDecodeError:
        return result if len(result) <= _KEPT else result[:_KEPT] + " (shortened)"
    if not isinstance(parsed, dict):
        return result
    kept: dict[str, Any] = {
        key: parsed[key] for key in _KEYS if parsed.get(key) not in (None, "", [], ())
    }
    if options := parsed.get("options"):
        kept["options"] = [one.get("label") for one in options if isinstance(one, dict)]
    return json.dumps(kept, separators=(",", ":"))


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


def _answer_step(
    written: Written | None, duration_s: float, error: str | None = None
) -> AnswerStep:
    fills = written.fills if written is not None else ()
    return AnswerStep(
        form=written.form.model_dump() if written is not None else None,
        error=error,
        tries=len(fills) if written is not None else 2,
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
