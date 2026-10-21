"""The host: the loop between the user, the model and the autonomous agent's server.

A conversation works on one study of a profile. The user's first message is read into the
objective form (P0): the profile, the objective, the settings the user gives, which stay as they
gave them. Then the model works through the phases, each offering its tools and its manual: the
survey (P1), the curves (P2), the freeze (P3), the inversion (P4), until the report (P5). The phase
is read from the study's records before every call, never kept in memory, so that a run stopped
anywhere resumes from them.

Before each of the model's calls the host rebuilds what it reads: the role and the phase's manual,
the conversation, and the state card last, written afresh, so that the prompt's beginning
stays the same from call to call. A tool result, once read, shrinks to its journal line or its first
line, and past 60 % of the model's context the oldest calls drop out. At the freeze the
conversation is rebuilt: the request, the objective, the frozen curves and a short summary of the
processing. The host refuses, saying why: a tool of another phase, a call past a budget,
a change of a setting the user gave. A reply without a call ends the turn: the model's
question, or once the curves are frozen, the report; a reply asking nothing before the freeze, or
one leaving a model undecided after it, sends the model back once. Every call and its cost go to the
study's usage log; the conversation's transcript is saved in the study as PACo saves its own."""

import base64
import json
import re
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache
from typing import Any, cast

from mcp import Client
from mcp.types import CallToolResult, RequestParamsMeta, TextContent, Tool
from openai.types.chat import ChatCompletionContentPartParam, ChatCompletionMessageParam
from sigpipe.masw.profiles import ProfileError, load_profile
from sigpipe.masw.runs import find_run
from sigpipe.masw.runs.writing import write_atomic

from paco import inspection
from paco.agent.answer import AnswerError
from paco.agent.conversion import tools_for_model
from paco.agent.loop import as_data, from_data
from paco.agent.model import ChatModel, Filled, Reply, ToolCall
from paco.agent.record import AnswerStep, ModelStep, ScopeStep, Step, ToolStep, Transcript
from paco.agent.scope import Context, ScopeError
from paco.autonomous import prompts
from paco.autonomous.conversion import tool_messages
from paco.autonomous.host import answer as answering
from paco.autonomous.host import budgets, judging
from paco.autonomous.host.card import card, effort_of, objective_of, state
from paco.autonomous.host.phases import (
    MANUALS,
    NAMES,
    Phase,
    offered,
    phase_of,
    refusal,
    reopen,
    study_for,
    wanted,
)
from paco.autonomous.host.scope import GeoScope, read_objective
from paco.autonomous.tools import given, journal, models
from paco.autonomous.tools import search as searching
from paco.autonomous.tools.commit import line_of
from paco.autonomous.tools.freeze import describe_frozen, frozen
from paco.autonomous.tools.invert import inversions_of
from paco.autonomous.tools.tidy import tidy
from paco.autonomous.tools.trials import Study
from paco.llm.settings import FRONTIER_CONTEXT, ModelSettings
from paco.qc.coherence import receiver_spacing
from paco.qc.report import stretches, xmid_of
from paco.settings import Settings

# What the user reads while the run goes: one line per experiment, commit and phase.
type OnEvent = Callable[[str], None]
# The share of the model's context past which the oldest calls drop out.
COMPACT_AT = 0.6
# The smallest context the autonomous agent runs in, in tokens: its prompts and tools take about
# 12k, a survey's and a trial's results as much again.
CONTEXT_FLOOR = 32_768
# The calls kept whole however full the context, the latest.
KEPT_CALLS = 3
# A tool result's first lines, as earlier calls keep it.
KEPT_LINES = 2
TRANSCRIPTS = "transcripts"  # in the study's folder
CARD = "[PACo] The state card, as it is now:"
# The phases a model with vision sees PACo's atlas in, with the request: until the freeze.
ATLAS_PHASES = ("P1", "P2")
DROPPED = (
    "[PACo] Older calls were dropped from the conversation to keep it short: the journal (journal "
    "read) and the state card hold what they found."
)
# What the host says to a message naming no profile, or one that is not there: the profiles.
NO_PROFILE = "Which profile should I study? The profiles:\n{profiles}"
NUDGE = (
    "[PACo] The run is not done: the curves are not frozen. Go on with the phase's tools, or ask "
    "the user what you need from them."
)
PENDING = (
    "[PACo] The run is not done: {count} model(s) not decided, at {where}. Decide each (accept "
    "it, or reject it with its reason), then answer; or ask the user what you need from them."
)
SPENT = "The run's budget of model calls is spent ({used} of {budget}): it stops here."
SPENT_TOKENS = "The run's budget of tokens is spent ({used} of {budget}): it stops here."
# The judge's answers the server may refuse in a row before the judging stops.
JUDGE_TRIES = 3
# The experiment a try_settings result names first.
_EXPERIMENT = re.compile(r"\b(E\d+):")
# An inversion job's id, as invert gives it.
_JOB = re.compile(r"inv-\d{8}-\d{6}-[0-9a-f]{4}")


@dataclass(frozen=True)
class Limits:
    """The model's context, in tokens (PACO_LLM_CONTEXT), and a tool call's timeout."""

    context: int = FRONTIER_CONTEXT
    tool_seconds: float = 3600.0

    @classmethod
    def of(cls, settings: ModelSettings) -> Limits:
        """The limits of the model `settings` name: its context, set or assumed."""
        return cls(context=settings.context())


def short_context(context: int) -> str | None:
    """Why a model of `context` tokens cannot run the autonomous mode, whose prompts and tools alone
    take about 12k; None when it can (128k advised: the work of a line fills it)."""
    if context >= CONTEXT_FLOOR:
        return None
    return (
        f"The autonomous mode needs a model with a long context: 128k tokens advised, "
        f"{CONTEXT_FLOOR // 1024}k at least, and this one has {context:,} (PACO_LLM_CONTEXT). "
        "Choose a frontier model, or the workflow mode (PACO_AGENT_MODE=workflow)."
    )


class Host:
    """One conversation between the user, the model and the autonomous agent's server."""

    def __init__(
        self,
        client: Client,
        model: ChatModel,
        tools: Sequence[Tool],
        settings: Settings,
        *,
        limits: Limits | None = None,
        vision: bool = False,
        on_event: OnEvent = print,
    ) -> None:
        self._client = client
        self._model = model
        self._tools = list(tools)
        self._settings = settings
        self._limits = limits or Limits()
        self._vision = vision
        self._on_event = on_event
        self.study: Study | None = None
        self.scope: GeoScope | None = None
        self.request = ""
        # The conversation since the last reset: no system message, no state card.
        self.history: list[ChatCompletionMessageParam] = []
        # Each tool message's short form, once the model has read it.
        self._short: dict[str, str] = {}
        self.steps: list[Step] = []
        self.started_at = datetime.now(UTC)
        self.meta: RequestParamsMeta = {
            "conversation": uuid.uuid4().hex,
            "turn": 0,
            "prompts": prompts.version(),
            "vision": vision,
        }
        if isinstance(name := getattr(model, "name", None), str):
            self.meta["model"] = name
        # The size of every call's messages, as the host estimated it.
        self.sizes: list[int] = []
        # The state card the model read last: stored state its answer's numbers may come from.
        self._card = ""

    @classmethod
    async def start(
        cls,
        client: Client,
        model: ChatModel,
        settings: Settings,
        *,
        limits: Limits | None = None,
        vision: bool = False,
        on_event: OnEvent = print,
    ) -> Host:
        """A host with the tools of the server `client` is connected to."""
        tools = (await client.list_tools()).tools
        return cls(
            client,
            model,
            tools,
            settings,
            limits=limits,
            vision=vision,
            on_event=on_event,
        )

    async def answer(self, message: str) -> str:
        """The answer to the user's `message`: the run goes on from where its study stands, until
        the model asks the user something, the report is given, or a budget stops it."""
        self.meta["turn"] += 1
        start = len(self.history)
        try:
            if self.study is None:
                asked = await self._objective(message)
                if asked is not None:
                    return asked
                start = 1
            elif phase_of(self.study) == "P5":
                asked = await self._follow_up(message)
                if asked is not None:
                    return asked
                start = 1
            else:
                self.history.append({"role": "user", "content": message})
                start += 1
            text = await self._run()
            # The answer the user reads ends the turn, in place of the model's draft, as PACo's
            # agent keeps it: the transcript's answer is theirs (the evaluation reads it).
            last = self.history[-1]
            if last["role"] == "assistant" and not last.get("tool_calls"):
                del self.history[-1]
            self.history.append({"role": "assistant", "content": text})
            return text
        except BaseException as error:
            # The turn's calls taken out, none left unanswered: the next message goes on from the
            # study's records, which keep what the calls did.
            del self.history[start:]
            self.history.append({"role": "assistant", "content": f"(No answer: {error!r})"})
            raise
        finally:
            self._save()

    def _profiles(self) -> str:
        """The question to a message naming no profile there is: the profiles, to choose one."""
        return NO_PROFILE.format(profiles=inspection.profiles_text(self._settings))

    async def _objective(self, message: str) -> str | None:
        """P0: the objective read from the user's message, and the study it works on, resumed
        when the profile has one not reported yet. A question when the message names no
        profile."""
        started = time.perf_counter()
        try:
            read = await read_objective(self._model, message, Context())
        except ScopeError as error:
            self.steps.append(_scope_step(None, time.perf_counter() - started, str(error)))
            return f"I could not read what your message asks ({error}). Say it again, please."
        scope = cast(GeoScope, read.scope)
        self.steps.append(_scope_step(scope, time.perf_counter() - started))
        if scope.profile is None:
            return self._profiles()
        try:
            spacing = receiver_spacing(load_profile(scope.profile, self._settings))
        except ProfileError:
            return f"There is no profile '{scope.profile}'. {self._profiles()}"
        study, lost = study_for(scope.profile, self._settings)
        if lost is not None:
            self._on_event(lost)
        self.study, self.scope, self.request = study, scope, message
        self.meta["study"] = study.id
        # Each window's inversion retries, the effort's: the server refuses past them.
        self.meta["inversion_retries"] = budgets.EFFORTS[effort_of(study)].inversion_retries
        goal = objective_of(study)
        # The settings the message types, kept over those given before.
        changed = given.give(study, scope.given(spacing))
        if goal is None:
            fields = scope.objective_fields(spacing)
            goal = journal.objective(study, fields, scope.objective())
            self._on_event(
                f"Objective: {goal.text}. Study {study.id} of {study.profile}; first, the survey."
            )
            self.history = [{"role": "user", "content": message}]
        else:
            phase = phase_of(study)
            self._on_event(f"Resuming study {study.id} at {phase}, the {NAMES[phase]}.")
            if changed:
                said = journal.settings(study, changed)
                self._on_event(f"{said.id}: {said.text}")
            self.history = [
                {"role": "user", "content": f"{message}\n\n{self._resumed(goal)}"},
            ]
        for record in read.fills:
            budgets.record(
                study, _use("form", "P0", 0.0, record.prompt_tokens, record.completion_tokens)
            )
        return None

    async def _follow_up(self, message: str) -> str | None:
        """A message after the report, read as the first one is, the study's profile and run its
        context. Asking for the curves again (processing, picking, their settings) or for the
        inversion (models, soils, its settings), the study opens again there, its journal and
        budgets kept; about another profile, that profile's study; any other, the report."""
        study = self._study
        line = line_of(study)
        context = Context(profile=study.profile, run_id=line.run_id if line is not None else None)
        started = time.perf_counter()
        try:
            read = await read_objective(self._model, message, context)
        except ScopeError as error:
            self.steps.append(_scope_step(None, time.perf_counter() - started, str(error)))
            return f"I could not read what your message asks ({error}). Say it again, please."
        scope = cast(GeoScope, read.scope)
        self.steps.append(_scope_step(scope, time.perf_counter() - started))
        for record in read.fills:
            budgets.record(
                study, _use("form", "P5", 0.0, record.prompt_tokens, record.completion_tokens)
            )
        if scope.profile is not None and scope.profile != study.profile:
            self.study = None
            return await self._objective(message)
        typed = scope.given(receiver_spacing(load_profile(study.profile, self._settings)))
        back = wanted(scope, typed)
        if back is None:
            report = self._report_given()
            self.history.append({"role": "user", "content": message})
            self.history.append({"role": "assistant", "content": report})
            return report
        if changed := given.give(study, typed):
            said = journal.settings(study, changed)
            self._on_event(f"{said.id}: {said.text}")
        entry = reopen(study, back, message)
        self.scope, self.request = scope, message
        self._on_event(f"{entry.id}: {entry.text}")
        self.history = [{"role": "user", "content": f"{message}\n\n{self._reopened(entry)}"}]
        return None

    async def _run(self) -> str:
        """The model's calls until the turn ends: a reply without a call, the report, or the
        model calls' budget spent."""
        study = self._study
        nudged = False
        while True:
            phase = phase_of(study)
            if phase == "P5":
                return self._report_given()
            spent = budgets.spent(study, effort_of(study))
            if (said := _stop(spent)) is not None:
                self._on_event(f"   ({said})")
                return self._stands(said)
            tools = tools_for_model(offered(phase, self._tools))
            messages = self._messages(phase, _tokens([json.dumps(tools)]))
            self.sizes.append(_size(messages, tools))
            started = time.perf_counter()
            reply = await self._model(messages, tools)
            seconds = time.perf_counter() - started
            self.steps.append(_model_step(reply, seconds))
            budgets.record(
                study,
                _use("model", phase, seconds, reply.prompt_tokens, reply.completion_tokens),
            )
            self._compact()
            self.history.append(_assistant_message(reply))
            if not reply.tool_calls:
                form = await self._form(reply.content, phase)
                if phase in ("P1", "P2") and not form.question and not nudged:
                    # Before the freeze, a reply asking nothing: the run goes on, once.
                    nudged = True
                    self.history.append({"role": "user", "content": NUDGE})
                    continue
                pending = self._pending() if phase == "P4" else []
                if pending and not nudged:
                    # A report, or a question, with models not decided: the run goes on, once.
                    nudged = True
                    said = PENDING.format(count=len(pending), where=stretches(pending))
                    self.history.append({"role": "user", "content": said})
                    continue
                return self._answered(form, phase)
            for call in reply.tool_calls:
                self.history.extend(await self._call(call, phase))
                if phase_of(study) == "P4" and phase == "P2":
                    self._reset()
                    break

    def _messages(self, phase: Phase, tools: int) -> list[ChatCompletionMessageParam]:
        """What the model reads: the role and the phase's manual, the conversation, then the
        state card as it is now. A model with vision sees PACo's atlas until the freeze,
        after the request: kept out of the conversation, which drops a figure once seen."""
        manual = MANUALS.get(phase)
        system = prompts.prompt("role") + (f"\n\n{prompts.prompt(manual)}" if manual else "")
        state(self._study, self._settings)
        self._card = now = card(self._study, self._settings)
        self._dropped(system, now, tools)
        conversation = self.history
        if self._vision and phase in ATLAS_PHASES and conversation:
            conversation = [_with_atlas(conversation[0]), *conversation[1:]]
        return [
            {"role": "system", "content": system},
            *conversation,
            {"role": "user", "content": f"{CARD}\n{now}"},
        ]

    async def _call(self, call: ToolCall, phase: Phase) -> list[ChatCompletionMessageParam]:
        """A tool call the model asked for: refused, saying why, or made on the server, its result
        as the conversation carries it."""
        study = self._study
        spent = budgets.spent(study, effort_of(study))
        try:
            arguments: Any = json.loads(call.arguments or "{}")
        except json.JSONDecodeError as error:
            arguments, said = None, f"Not called: the arguments are not valid JSON ({error})."
        else:
            said = (
                refusal(phase, call.name)
                or budgets.refusal(spent, call.name)
                or (None if isinstance(arguments, dict) else "Not called: arguments are an object.")
            )
        if said is not None:
            self._on_event(f"-> {call.name}({call.arguments}) refused: {said}")
            budgets.record(study, _use("tool", phase, 0.0, name=call.name, refused=True))
            self.steps.append(_tool_step(call, said, called=False, error=True, seconds=0.0))
            return [{"role": "tool", "tool_call_id": call.id, "content": said}]
        self._on_event(f"-> {call.name}({call.arguments})")
        started = time.perf_counter()
        try:
            result = await self._client.call_tool(
                call.name,
                cast(dict[str, Any], arguments),
                read_timeout_seconds=self._limits.tool_seconds,
                meta=self.meta,
            )
        except Exception as error:
            result = CallToolResult(
                content=[TextContent(type="text", text=f"The call failed: {error}")],
                is_error=True,
            )
        seconds = time.perf_counter() - started
        failed = bool(result.is_error)
        budgets.record(study, _use("tool", phase, seconds, name=call.name, failed=failed))
        knockout = searching.knockout_of(study) if call.name == "search" else None
        if not failed and knockout is not None and not knockout.finalists:
            # The model reads the finalists, not the sheets: the host has them judged.
            result = await self._judged(result, phase)
            failed = bool(result.is_error)
        carried: list[ChatCompletionMessageParam] = tool_messages(
            call.id, call.name, result, self._vision
        )
        text = str(carried[0].get("content") or "")
        self.steps.append(_tool_step(call, text, called=True, error=failed, seconds=seconds))
        if call.name == "invert" and not failed and (job := _JOB.search(text)) is not None:
            # The model reads the job's end, not its start: the host follows it (as PACo's).
            ended = await self._follow(job.group(0), phase)
            text = as_data(call.name, f"{from_data(text)}\n{ended}")
            followed: ChatCompletionMessageParam = {
                "role": "tool",
                "tool_call_id": call.id,
                "content": text,
            }
            carried = [followed]
        self._short[call.id] = self._shortened(call.name, text, failed)
        if not failed:
            self._said(call.name)
        return carried

    async def _follow(self, job_id: str, phase: Phase) -> str:
        """Job `job_id` followed with job_status until it ends, a line for the user each time it
        is asked: its status once ended."""
        while True:
            started = time.perf_counter()
            result = await self._client.call_tool(
                "job_status",
                {"job_id": job_id},
                read_timeout_seconds=self._limits.tool_seconds,
                meta=self.meta,
            )
            seconds = time.perf_counter() - started
            text = "\n".join(
                block.text for block in result.content if isinstance(block, TextContent)
            )
            budgets.record(
                self._study,
                _use("tool", phase, seconds, name="job_status", failed=bool(result.is_error)),
            )
            self.steps.append(
                ToolStep(
                    name="job_status",
                    arguments=json.dumps({"job_id": job_id}),
                    called=True,
                    is_error=bool(result.is_error),
                    duration_s=round(seconds, 3),
                    result=text,
                    by_host=True,
                )
            )
            head = text.splitlines()[0] if text else job_id
            self._on_event(f"   {head}")
            if result.is_error or "Call job_status again" not in text:
                return text

    async def _judged(self, searched: CallToolResult, phase: Phase) -> CallToolResult:
        """The search's sheets judged by eye in short calls of their own (judging), round after
        round, the server's judge recording each answer and giving the next sheets, a line for
        the user each time: the search's words, then the finalists tried side by side, as the
        model reads them; or, refused JUDGE_TRIES times in a row, why the judging stopped."""
        study = self._study
        goal = objective_of(study)
        mode = searching.mode_of(study, load_profile(study.profile, self._settings))
        line = judging.facts(study.profile, mode, goal.text if goal is not None else "none")
        head, *rest = searched.content
        said = head.text if isinstance(head, TextContent) else ""
        sheets = CallToolResult(content=rest)
        refused: str | None = None
        tries = calls = 0
        while True:
            if (stop := _stop(budgets.spent(study, effort_of(study)))) is not None:
                # The judging stops with the run: the model's next call is not made.
                stopped = f"{said}\n\nThe sheets could not all be judged: {stop}"
                return CallToolResult(
                    content=[TextContent(type="text", text=stopped)], is_error=True
                )
            started = time.perf_counter()
            calls += 1
            before = searching.knockout_of(study)
            try:
                judged = await judging.judge(
                    self._model, line, sheets, vision=self._vision, refused=refused
                )
            except judging.JudgeError as error:
                self._judge_steps(error.fills, phase, time.perf_counter() - started)
                refused, result = f"the form did not parse ({error})", None
            else:
                self._judge_steps(judged.fills, phase, time.perf_counter() - started)
                result = await self._judge(judging.as_arguments(study.profile, judged), phase)
                refused = judging.text_of(result) if result.is_error else None
                if not result.is_error and before is not None:
                    self._on_event(f"   Round {before.round}, judged by eye: {judged.said()}.")
            if refused is not None:
                tries += 1
                if tries == JUDGE_TRIES:
                    stopped = (
                        f"{said}\n\nThe sheets could not be judged ({refused}): search again "
                        "judges those left."
                    )
                    return CallToolResult(
                        content=[TextContent(type="text", text=stopped)], is_error=True
                    )
                continue
            assert result is not None
            tries = 0
            knockout = searching.knockout_of(study)
            if knockout is not None and knockout.finalists:
                text = f"{said}\n\nThe sheets judged by eye in {calls} short calls. "
                text += judging.text_of(result)
                figures = [block for block in result.content if not isinstance(block, TextContent)]
                return CallToolResult(content=[TextContent(type="text", text=text), *figures])
            sheets = result

    async def _judge(self, arguments: dict[str, Any], phase: Phase) -> CallToolResult:
        """The server's judge called with `arguments`, for the model: its result."""
        started = time.perf_counter()
        try:
            result = await self._client.call_tool(
                "judge", arguments, read_timeout_seconds=self._limits.tool_seconds, meta=self.meta
            )
        except Exception as error:
            result = CallToolResult(
                content=[TextContent(type="text", text=f"The call failed: {error}")],
                is_error=True,
            )
        seconds = time.perf_counter() - started
        failed = bool(result.is_error)
        budgets.record(self._study, _use("tool", phase, seconds, name="judge", failed=failed))
        self.steps.append(
            ToolStep(
                name="judge",
                arguments=json.dumps(arguments),
                called=True,
                is_error=failed,
                duration_s=round(seconds, 3),
                result=judging.text_of(result),
                by_host=True,
            )
        )
        return result

    def _judge_steps(self, fills: Sequence[Filled], phase: Phase, seconds: float) -> None:
        """The judge's forms recorded: in the usage log, out of the model calls' budget; in the
        transcript as the model's calls, their tokens counted."""
        each = seconds / len(fills)
        for filled in fills:
            budgets.record(
                self._study,
                _use(
                    "form",
                    phase,
                    each,
                    filled.prompt_tokens,
                    filled.completion_tokens,
                    name="judge",
                ),
            )
            self.steps.append(
                ModelStep(
                    duration_s=round(each, 3),
                    prompt_tokens=filled.prompt_tokens,
                    completion_tokens=filled.completion_tokens,
                    tool_calls=0,
                    cached_tokens=filled.cached_tokens,
                )
            )

    def _said(self, name: str) -> None:
        """The progress line after a tool that changed the study: its journal entry."""
        kinds = {
            "try_settings": "experiment",
            "commit": "commit",
            "freeze": "freeze",
            "invert": "inversion",
        }
        if name not in kinds:
            return
        found = journal.entries(self._study)
        entry = next((one for one in reversed(found) if one.kind == kinds[name]), None)
        if entry is not None:
            self._on_event(f"   {journal.line(entry)}")

    def _shortened(self, name: str, text: str, failed: bool) -> str:
        """A tool message as the conversation keeps it once read: an experiment's journal line
        with each variant's figures (the evidence of what the model concludes from it), else its
        first lines (an error whole)."""
        if failed:
            return text
        if name == "try_settings" and (made := _EXPERIMENT.search(text)) is not None:
            entry = next(
                (one for one in journal.entries(self._study) if one.id == made.group(1)), None
            )
            if entry is not None:
                shown = [
                    f"- {one['label']}: {one['figures']}"
                    for one in entry.data.get("variants", [])
                    if one.get("figures")
                ]
                said = "\nTheir figures:\n" + "\n".join(shown) if shown else ""
                return f'<data from="{name}">\n{journal.line(entry)}{said}\n</data>'
        if name == "invert":
            found = [one for one in journal.entries(self._study) if one.kind == "inversion"]
            if found:
                return f'<data from="{name}">\n{journal.line(found[-1])}\n</data>'
        lines = text.splitlines()
        body = lines[1:-1] if lines and lines[0].startswith("<data") else lines
        if len(body) <= KEPT_LINES:
            return text
        kept = "\n".join(body[:KEPT_LINES])
        return f'<data from="{name}">\n{kept}\n(shortened: call again for the rest)\n</data>'

    def _compact(self) -> None:
        """The tool results the model has read, shrunk to their short form, the figures it has
        seen left out."""
        for index, message in enumerate(self.history):
            if message["role"] == "tool":
                short = self._short.get(str(message.get("tool_call_id")))
                if short is not None and message.get("content") != short:
                    self.history[index] = {**message, "content": short}
            elif message["role"] == "user" and isinstance(message.get("content"), list):
                self.history[index] = {"role": "user", "content": "(A figure, seen.)"}

    def _dropped(self, system: str, now: str, tools: int) -> None:
        """Past COMPACT_AT of the model's context, the oldest calls dropped, each with its
        results, the latest KEPT_CALLS kept, and the user's own messages."""
        budget = COMPACT_AT * self._limits.context
        while tools + _tokens([system, now, json.dumps(self.history)]) > budget:
            calls = [i for i, one in enumerate(self.history) if one["role"] == "assistant"]
            if len(calls) <= KEPT_CALLS:
                return
            start = end = calls[0]
            end += 1
            while end < len(self.history) and self.history[end]["role"] == "tool":
                end += 1
            del self.history[start:end]
            if not any(one.get("content") == DROPPED for one in self.history):
                self.history.insert(start, {"role": "user", "content": DROPPED})

    def _reset(self) -> None:
        """The conversation rebuilt at the freeze: the user's request and objective, the
        frozen curves and a short summary of the processing."""
        study = self._study
        made = frozen(study)
        goal = objective_of(study)
        summary = "; ".join(
            journal.line(entry)
            for entry in journal.entries(study)
            if entry.kind in ("commit", "decision", "conclusion")
        )
        self.history = [
            {
                "role": "user",
                "content": (
                    f"{self.request}\n\n[PACo] The objective: "
                    f"{goal.text if goal is not None else 'none'}.\n[PACo] The curves are frozen: "
                    f"{describe_frozen(made) if made is not None else 'none'}\n[PACo] The "
                    f"processing, in short: {summary or 'nothing to say'}."
                ),
            }
        ]
        self._short = {}
        self._on_event(
            "Phase: P4, the inversion; the conversation starts afresh from the frozen curves."
        )

    async def _form(self, draft: str, phase: Phase) -> answering.GeoAnswerForm:
        """The model's reply without a call, in PACo's answer form: what it says, its question,
        its interpretation; the draft as it is when no form parses."""
        started = time.perf_counter()
        # After the freeze, the answer may be the report: the experiments, for it to say what was
        # tried.
        experiments = answering.tried(self._study) if phase == "P4" else []
        try:
            written = await answering.write_answer(self._model, self.request, draft, experiments)
        except AnswerError as error:
            self.steps.append(_answer_step(None, time.perf_counter() - started, 2, str(error)))
            return answering.GeoAnswerForm(
                said=draft.replace("?", "."), question=None, interpretation=None, tried=None
            )
        form = written.form
        self.steps.append(
            _answer_step(form.model_dump(), time.perf_counter() - started, len(written.fills))
        )
        for one in written.fills:
            budgets.record(
                self._study, _use("form", phase, 0.0, one.prompt_tokens, one.completion_tokens)
            )
        return form

    def _answered(self, form: answering.GeoAnswerForm, phase: Phase) -> str:
        """The turn's answer: the model's question in the middle of the run; once the curves are
        frozen and nothing is asked, the report (P5)."""
        sources = self._sources()
        if phase == "P4" and not form.question:
            text = answering.report(self._study, self._settings, self._line(), form, sources)
            journal.report(self._study, text)
            self._on_event("Phase: P5, the report.")
            return text
        return answering.progress(self._line(), form, self._where(), sources)

    def _pending(self) -> list[float]:
        """The windows inverted whose model is not decided, by xmid."""
        made = frozen(self._study)
        if made is None:
            return []
        folder = find_run(made.run_id, self._settings)
        done = models.statuses(self._study, folder, made.run_id)
        inverted = {one.unit for one in inversions_of(self._study, made.run_id)}
        return sorted(x for unit in inverted - set(done) if (x := xmid_of(unit)) is not None)

    def _report_given(self) -> str:
        """The study's report, given before: the journal's."""
        entry = next(e for e in reversed(journal.entries(self._study)) if e.kind == "report")
        return str(entry.data.get("answer", ""))

    def _stands(self, said: str) -> str:
        """An answer when the run stops: why, and where the work stands."""
        form = answering.GeoAnswerForm(said=said, question=None, interpretation=None, tried=None)
        return answering.progress(self._line(), form, self._where())

    def _where(self) -> str:
        """Where the work stands, from the state card's first line and its decisions."""
        lines = card(self._study, self._settings).splitlines()
        windows = next((line for line in lines if line.startswith("Windows:")), "")
        return " ".join(part for part in (lines[0], windows) if part)

    def _line(self) -> str:
        return self.scope.line() if self.scope is not None else ""

    def _reopened(self, entry: journal.Entry) -> str:
        """What the model reads when the study opens again after its report: why, the objective
        and the journal's last entries; the state card says the rest, its budgets as they were."""
        goal = objective_of(self._study)
        last = journal.read(self._study, last=8)
        return (
            f"[PACo] Study {self._study.id} opens again after its report ({entry.id}: "
            f"{entry.text}), its budgets as they were. The objective: "
            f"{goal.text if goal is not None else 'not read'}. The journal's last entries:\n{last}"
        )

    def _resumed(self, goal: journal.Entry) -> str:
        """What the model reads on resuming a study: its objective and its journal's last
        entries; the state card says the rest."""
        last = journal.read(self._study, last=8)
        return (
            f"[PACo] This conversation resumes study {self._study.id}: the objective, "
            f"{goal.text}. The journal's last entries:\n{last}"
        )

    def _sources(self) -> list[str]:
        """What the answer's numbers may come from: the tools' results of the conversation, and
        the state card the model read last (stored state)."""
        return (
            [
                str(message.get("content") or "")
                for message in self.history
                if message["role"] in ("user", "tool")
            ]
            + [step.result for step in self.steps if isinstance(step, ToolStep)]
            + [self._card]
        )

    def transcript(self) -> Transcript:
        """The conversation so far, and every step with its cost."""
        return Transcript(
            started_at=self.started_at,
            model=str(self.meta.get("model", "")),
            messages=[dict(message) for message in self.history],
            steps=list(self.steps),
            conversation=str(self.meta["conversation"]),
            prompt_version=prompts.version(),
        )

    def close(self) -> int:
        """The session's end: its study's working files removed (tidy), its results kept; the
        bytes freed."""
        return tidy(self.study) if self.study is not None else 0

    def _save(self) -> None:
        """The transcript saved in the study, as PACo saves its own, rewritten after each turn."""
        if self.study is None:
            return
        folder = self.study.folder / TRANSCRIPTS
        folder.mkdir(exist_ok=True)
        path = folder / f"{self.started_at:%Y%m%d-%H%M%S}-{self.meta['conversation']}.json"
        write_atomic(path, self.transcript().model_dump_json(indent=2))

    @property
    def _study(self) -> Study:
        if self.study is None:
            raise RuntimeError("No study yet: the objective is read first.")
        return self.study


def _stop(spent: budgets.Spent) -> str | None:
    """Why the run stops now: its model calls' budget or its tokens' spent; None within them."""
    if spent.calls_spent:
        return SPENT.format(used=spent.model_calls, budget=spent.budgets.model_calls)
    if spent.tokens_spent:
        used, budget = (f"{one / 1e6:.1f}M" for one in (spent.tokens, spent.budgets.tokens))
        return SPENT_TOKENS.format(used=used, budget=budget)
    return None


def _size(messages: Sequence[ChatCompletionMessageParam], tools: Sequence[object]) -> int:
    """A call's size in tokens, estimated: four characters a token."""
    return _tokens([json.dumps(messages), json.dumps(tools)])


def _tokens(texts: Sequence[str]) -> int:
    return sum(len(text) for text in texts) // 4


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


def _use(
    kind: str,
    phase: str,
    seconds: float,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    *,
    name: str | None = None,
    refused: bool = False,
    failed: bool = False,
) -> budgets.Use:
    return budgets.Use.model_validate(
        {
            "kind": kind,
            "phase": phase,
            "at": datetime.now(UTC),
            "seconds": round(seconds, 3),
            "name": name,
            "refused": refused,
            "failed": failed,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
    )


def _model_step(reply: Reply, seconds: float) -> ModelStep:
    return ModelStep(
        duration_s=round(seconds, 3),
        prompt_tokens=reply.prompt_tokens,
        completion_tokens=reply.completion_tokens,
        tool_calls=len(reply.tool_calls),
        cached_tokens=reply.cached_tokens,
    )


def _tool_step(
    call: ToolCall, result: str, *, called: bool, error: bool, seconds: float
) -> ToolStep:
    return ToolStep(
        name=call.name,
        arguments=call.arguments,
        called=called,
        is_error=error,
        duration_s=round(seconds, 3),
        result=result,
    )


def _scope_step(scope: GeoScope | None, seconds: float, error: str | None = None) -> ScopeStep:
    return ScopeStep(
        prompt_version=prompts.version(),
        scope=scope.model_dump() if scope is not None else None,
        error=error,
        tries=1 if error is None else 2,
        duration_s=round(seconds, 3),
        prompt_tokens=None,
        completion_tokens=None,
    )


def _answer_step(
    form: dict[str, Any] | None, seconds: float, tries: int, error: str | None = None
) -> AnswerStep:
    return AnswerStep(
        form=form,
        error=error,
        tries=tries,
        duration_s=round(seconds, 3),
        prompt_tokens=None,
        completion_tokens=None,
    )


def _with_atlas(first: ChatCompletionMessageParam) -> ChatCompletionMessageParam:
    """The conversation's first message, the user's, with PACo's atlas after it."""
    if first["role"] != "user":
        return first
    content = first["content"]
    parts: list[ChatCompletionContentPartParam] = (
        [{"type": "text", "text": content}] if isinstance(content, str) else list(content)
    )
    return {"role": "user", "content": [*parts, *_atlas()]}


@cache
def _atlas() -> tuple[ChatCompletionContentPartParam, ...]:
    """PACo's atlas as the model reads it: each sheet's caption as data, then its figure."""
    parts: list[ChatCompletionContentPartParam] = []
    for caption, figure in prompts.atlas():
        url = f"data:image/png;base64,{base64.b64encode(figure).decode()}"
        parts.append({"type": "text", "text": as_data("atlas", caption)})
        parts.append({"type": "image_url", "image_url": {"url": url}})
    return tuple(parts)
