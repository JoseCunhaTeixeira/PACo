"""Playing the scenarios: the agent against PACo's server, in this process, one scenario at a
time, each with its own output directory."""

import json
import os
import re
import secrets
import shutil
import time
from collections import Counter
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import anyio
import openai
from mcp import Client

from paco import prompts, server
from paco.agent import Agent, ChatModel
from paco.agent.loop import CAPPED
from paco.agent.scope import SCOPE_REFUSAL
from paco.evaluation.checks import Trial
from paco.evaluation.defects import build_inputs
from paco.evaluation.judging import judge
from paco.evaluation.models import CheckResult, EvaluationReport, ScenarioResult
from paco.evaluation.scenarios import Scenario
from paco.inversion import InversionRecord
from paco.settings import CACHE_FOLDER, get_settings

_JOBS_TIMEOUT_S = 1_800  # an inversion left running when the conversation ends
# Worker processes of the server during an evaluation: the zero-settings scenario inverts the
# whole demo line at PAC's effort, about 6 minutes on 8.
EVALUATION_WORKERS = 8
# A gate's name, as a retry's trigger starts with it.
_GATE = re.compile(r"G\d+")

type OnEvent = Callable[[str], None]


async def run_evaluation(
    scenarios: Sequence[Scenario],
    model: ChatModel,
    model_name: str,
    root: Path,
    judge_model: ChatModel | None = None,
    judge_name: str | None = None,
    repeat: int = 1,
    on_event: OnEvent = print,
) -> EvaluationReport:
    """Play every scenario `repeat` times, then write report.json in a new folder of `root`.

    Each play has its own folder: the scenario's name, or with repeats, one numbered folder per
    play inside it (pick_active/1, pick_active/2, ...).
    """
    started_at = datetime.now(UTC)
    eval_id = f"eval-{started_at:%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"
    folder = root / eval_id
    inputs = build_inputs(get_settings().input_dir, folder / "inputs")
    # One images' cache for every play: the same profiles, imaged once (S8); not a result.
    cache = folder / CACHE_FOLDER
    results: list[ScenarioResult] = []
    for scenario in scenarios:
        for attempt in range(1, repeat + 1):
            if repeat == 1:
                on_event(f"== {scenario.name}")
                where = folder / scenario.name
            else:
                on_event(f"== {scenario.name} #{attempt}")
                where = folder / scenario.name / str(attempt)
            play = run_scenario(
                scenario, model, model_name, where, judge_model, on_event, attempt, inputs, cache
            )
            results.append(await play)
    shutil.rmtree(cache, ignore_errors=True)
    report = EvaluationReport(
        eval_id=eval_id,
        model=model_name,
        judge_model=judge_name if judge_model is not None else None,
        prompt_version=prompts.version(),
        started_at=started_at,
        repeat=repeat,
        results=tuple(results),
        thresholds={scenario.name: scenario.threshold for scenario in scenarios},
    )
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "report.json").write_text(report.model_dump_json(indent=2))
    return report


async def run_scenario(
    scenario: Scenario,
    model: ChatModel,
    model_name: str,
    folder: Path,
    judge_model: ChatModel | None = None,
    on_event: OnEvent = print,
    attempt: int = 1,
    inputs: Path | None = None,
    cache: Path | None = None,
) -> ScenarioResult:
    """Play `scenario` with the server reading `inputs` (the settings' input_dir by default) and
    writing into `folder`/outputs, with EVALUATION_WORKERS, and score it; the conversation is
    saved as `folder`/transcript.json. `attempt` numbers the play; `cache`, the images' cache
    the plays share (the outputs' own by default)."""
    outputs = folder / "outputs"
    start = time.perf_counter()
    with _server_settings(outputs, inputs, cache):
        if scenario.setup is not None:
            scenario.setup(get_settings())
        error: str | None = None
        refused: str | None = None
        async with Client(server.server) as client:
            agent = await Agent.start(client, model, on_event=on_event)
            try:
                for question in scenario.questions:
                    await agent.answer(question)
            except (openai.APIConnectionError, openai.APIStatusError) as failed:
                if (lost := _lost(failed)) is not None:
                    # Not the model's doing: the play is lost, the evaluation goes on.
                    error = lost
                    on_event(f"   (play lost: {error})")
                else:
                    # A request the server refused (a schema it does not take, the context
                    # exceeded, a model it does not serve): this setup cannot play it, a
                    # failure, said.
                    refused = f"{type(failed).__name__}: {failed.message}"
                    on_event(f"   (refused by the model's server: {refused})")
        await _wait_for_jobs(outputs)
    duration_s = round(time.perf_counter() - start, 1)

    transcript = agent.transcript(model_name)
    # A scenario that never processed a profile left no folder behind.
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "transcript.json").write_text(transcript.model_dump_json(indent=2))
    trial = Trial(transcript, outputs)
    tokens = [step.prompt_tokens for step in transcript.steps if step.kind == "model"]
    known_tokens = [count for count in tokens if count is not None]
    return ScenarioResult(
        name=scenario.name,
        kind=scenario.kind,
        attempt=attempt,
        checks=_scored(scenario, trial, error, refused),
        judge=(
            await judge(judge_model, scenario.rubric, transcript)
            if judge_model and error is None
            else None
        ),
        tool_calls=len(transcript.tool_steps),
        failed_calls=sum(step.is_error for step in transcript.tool_steps),
        max_prompt_tokens=max(known_tokens) if known_tokens else None,
        duration_s=duration_s,
        answer=transcript.answer,
        scope_blocks=sum(
            not step.called and step.result.startswith(SCOPE_REFUSAL)
            for step in transcript.tool_steps
        ),
        caps_reached=sum(
            message.get("role") == "assistant" and CAPPED in str(message.get("content") or "")
            for message in transcript.messages
        ),
        retries=gate_retries(outputs),
        error=error,
    )


def _lost(failed: openai.APIError) -> str | None:
    """Why `failed` lost the play, when no model's server judged the request: the server out
    of reach or failing (a timeout too), or a 404 that is not its JSON, a page from whatever
    answers at its address (a proxy's while the server behind it is down). None for a refusal,
    the server's own 404 among them (a model it does not serve, a path it does not have)."""
    name = type(failed).__name__
    if isinstance(failed, openai.NotFoundError) and not isinstance(failed.body, dict):
        return f"{name}: a 404 page at {failed.request.url}, no model's server behind it"
    if isinstance(failed, (openai.APIConnectionError, openai.InternalServerError)):
        return f"{name}: {failed}"
    return None


def _scored(
    scenario: Scenario, trial: Trial, error: str | None, refused: str | None
) -> tuple[CheckResult, ...]:
    """The play's checks: none for a play lost; for a request the server refused, that alone."""
    if error is not None:
        return ()
    if refused is not None:
        return (
            CheckResult(name="the model's server took every request", passed=False, detail=refused),
        )
    return tuple(check(trial) for check in scenario.checks)


def gate_retries(outputs: Path) -> dict[str, int]:
    """The retries the gates asked in the runs under `outputs`, by gate: the attempts of the QC
    logs each triggered by a gate's flag ("G2:ridge_at_vmax"), those a reset left behind too."""
    counts: Counter[str] = Counter()
    for log in outputs.glob("*/*/qc_log.jsonl"):
        for line in log.read_text().splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(entry, dict) or entry.get("event", "stage") != "stage":
                continue
            gate, flagged, _ = str(entry.get("triggered_by") or "").partition(":")
            if flagged and _GATE.fullmatch(gate):
                counts[gate] += 1
    return dict(sorted(counts.items()))


@contextmanager
def _server_settings(
    outputs: Path, inputs: Path | None, cache: Path | None = None
) -> Generator[None]:
    """The in-process server reads its settings from the environment: point its outputs (and
    inputs, and images' cache) here, with the evaluation's workers."""
    values = {"PACO_OUTPUT_DIR": str(outputs), "PACO_WORKERS": str(EVALUATION_WORKERS)}
    if inputs is not None:
        values["PACO_INPUT_DIR"] = str(inputs)
    if cache is not None:
        values["PACO_CACHE_DIR"] = str(cache)
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    get_settings.cache_clear()
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                del os.environ[name]
            else:
                os.environ[name] = value
        get_settings.cache_clear()


async def _wait_for_jobs(outputs: Path) -> None:
    """Let the inversions the scenario started finish, so that scenarios never overlap."""
    deadline = time.monotonic() + _JOBS_TIMEOUT_S
    for path in outputs.glob("*/*/inversion.json"):
        job = InversionRecord.model_validate_json(path.read_text()).job_id
        while server.JOBS.is_live(job) and time.monotonic() < deadline:
            await anyio.sleep(0.5)
