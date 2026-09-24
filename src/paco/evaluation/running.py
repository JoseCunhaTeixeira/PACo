"""Playing the scenarios: the agent against PACo's server, in this process, one scenario at a
time, each with its own output directory."""

import os
import secrets
import time
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import anyio
from mcp import Client

from paco import server
from paco.agent import Agent, ChatModel
from paco.evaluation.checks import Trial
from paco.evaluation.defects import build_inputs
from paco.evaluation.judging import judge
from paco.evaluation.models import EvaluationReport, ScenarioResult
from paco.evaluation.scenarios import Scenario
from paco.inversion import InversionRecord
from paco.settings import get_settings

_JOBS_TIMEOUT_S = 1_800  # an inversion left running when the conversation ends
# Worker processes of the server during an evaluation (the user's choice of milestone 14): the
# zero-settings scenario inverts the whole demo line at PAC's effort, about 6 minutes on 8.
EVALUATION_WORKERS = 8

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
    play inside it (judge_active/1, judge_active/2, ...).
    """
    started_at = datetime.now(UTC)
    eval_id = f"eval-{started_at:%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"
    folder = root / eval_id
    inputs = build_inputs(get_settings().input_dir, folder / "inputs")
    results: list[ScenarioResult] = []
    for scenario in scenarios:
        for attempt in range(1, repeat + 1):
            if repeat == 1:
                on_event(f"== {scenario.name}")
                where = folder / scenario.name
            else:
                on_event(f"== {scenario.name} #{attempt}")
                where = folder / scenario.name / str(attempt)
            results.append(
                await run_scenario(
                    scenario, model, model_name, where, judge_model, on_event, attempt, inputs
                )
            )
    report = EvaluationReport(
        eval_id=eval_id,
        model=model_name,
        judge_model=judge_name if judge_model is not None else None,
        started_at=started_at,
        repeat=repeat,
        results=tuple(results),
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
) -> ScenarioResult:
    """Play `scenario` with the server reading `inputs` (the settings' input_dir by default) and
    writing into `folder`/outputs, with EVALUATION_WORKERS, and score it; the conversation is
    saved as `folder`/transcript.json. `attempt` numbers the play."""
    outputs = folder / "outputs"
    start = time.perf_counter()
    with _server_settings(outputs, inputs):
        async with Client(server.server) as client:
            agent = await Agent.start(client, model, on_event=on_event)
            for question in scenario.questions:
                await agent.answer(question)
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
        checks=tuple(check(trial) for check in scenario.checks),
        judge=await judge(judge_model, scenario.rubric, transcript) if judge_model else None,
        tool_calls=len(transcript.tool_steps),
        failed_calls=sum(step.is_error for step in transcript.tool_steps),
        max_prompt_tokens=max(known_tokens) if known_tokens else None,
        duration_s=duration_s,
        answer=transcript.answer,
    )


@contextmanager
def _server_settings(outputs: Path, inputs: Path | None) -> Generator[None]:
    """The in-process server reads its settings from the environment: point its outputs (and
    inputs) here, with the evaluation's workers."""
    values = {"PACO_OUTPUT_DIR": str(outputs), "PACO_WORKERS": str(EVALUATION_WORKERS)}
    if inputs is not None:
        values["PACO_INPUT_DIR"] = str(inputs)
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
