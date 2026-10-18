"""The A/B runner (docs/geo_plan.md §12): PACo's agent and the geophysicist agent play the same
scenarios, one play at a time, each in its own folder; the evaluation's report is written as
report.json in its folder.

PACo's agent is played by PACo's own runner (paco.evaluation.running.run_scenario: its agent
imported unchanged, against PACo's server in this process, scored by its scenario's checks); the
geophysicist agent the same way against its own server, through its host. Both read the same
lines (paco.geo.evaluation.inputs), share one images' cache, run on the evaluation's workers, and
have their inversions judged by the same QC configuration: the quick inversions' (PACo's
QUICK_MODEL) unless the scenario judges the models at PAC's effort. Each play's CPU and wall-clock
time are measured around it, its results read from what it left (paco.geo.evaluation.measures);
a known-truth scenario's line is prepared once, copied into each play, and its kept models
scored against the truth (paco.geo.evaluation.known_truth)."""

import secrets
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import anyio
import openai
from mcp import Client
from pydantic import BaseModel, ConfigDict

from paco import prompts as paco_prompts
from paco.agent.model import ChatModel
from paco.agent.record import Transcript
from paco.evaluation.checks import Check, Trial
from paco.evaluation.models import CheckResult, JudgeScore
from paco.evaluation.running import run_scenario
from paco.geo import prompts
from paco.geo import server as geo_server
from paco.geo.evaluation import known_truth
from paco.geo.evaluation.checks import study_of
from paco.geo.evaluation.clock import Clock
from paco.geo.evaluation.inputs import build_inputs
from paco.geo.evaluation.judging import judge_answer, judge_journal
from paco.geo.evaluation.measures import Agent, Measures, latest_run, measure
from paco.geo.evaluation.meter import Meter
from paco.geo.evaluation.scenarios import Pair
from paco.geo.host.budgets import Effort
from paco.geo.host.loop import Host, Limits
from paco.geo.tools.trials import Study
from paco.geo.upstream.running import lost, quick_config, server_settings
from paco.inversion import InversionRecord
from paco.settings import CACHE_FOLDER, Settings, get_settings

type OnEvent = Callable[[str], None]
# Which effort the plays' inversions are judged at: each scenario's own (quick, but where it
# judges the models), or every play quick, or every play at PAC's effort.
type Inversions = Literal["scenario", "quick", "full"]
_JOBS_TIMEOUT_S = 1_800  # PACo's: an inversion left running when the conversation ends


@dataclass(frozen=True)
class Variant:
    """How the geophysicist agent plays (§12): as it is, or without some tools, with vision, or at
    an effort whatever the message asks."""

    key: str = ""  # its name on the command line; "" for the agent as it is
    without: frozenset[str] = frozenset()
    vision: bool = False
    effort: Effort | None = None

    @property
    def label(self) -> str:
        return f"geo {self.key}" if self.key else "geo"


VARIANTS = {
    "no-advise": Variant("no-advise", without=frozenset({"advise"})),
    "vision": Variant("vision", vision=True),
    "quick": Variant("quick", effort="quick"),
    "normal": Variant("normal", effort="normal"),
    "thorough": Variant("thorough", effort="thorough"),
}


@dataclass(frozen=True)
class Models:
    """The models of an evaluation: the geophysicist agent's client, PACo's guided agent's (None:
    it is not played), each its own client of the same model, and the judge's (None: rule checks
    only)."""

    geo: ChatModel
    geo_name: str
    paco: ChatModel | None = None
    paco_name: str | None = None
    judge: ChatModel | None = None
    judge_name: str | None = None
    limits: Limits = field(default_factory=Limits)


class Play(BaseModel):
    """One play of a scenario by one agent: its checks, the judge's grades, its measures."""

    model_config = ConfigDict(frozen=True)

    scenario: str
    agent: Agent
    label: str  # "paco", "geo", or a variant's: "geo no-advise"
    attempt: int
    checks: tuple[CheckResult, ...]
    judge: JudgeScore | None = None  # the answer, against the scenario's rubric
    reasoning: JudgeScore | None = None  # the journal's (the geophysicist agent's)
    measures: Measures
    truth: known_truth.TruthResult | None = None  # a known-truth scenario's
    answer: str = ""
    error: str | None = None  # why the model's server cut the play short: out of the counts
    folder: str  # in the evaluation's folder

    @property
    def passed(self) -> bool:
        return self.error is None and all(check.passed for check in self.checks)


class ABReport(BaseModel):
    """An A/B evaluation: written as report.json, beside each play's folder."""

    model_config = ConfigDict(frozen=True)

    eval_id: str
    started_at: datetime
    paco_model: str | None  # None: PACo's agent was not played
    geo_model: str
    judge_model: str | None
    paco_prompts: str
    geo_prompts: str
    repeat: int
    inversions: Inversions
    questions: dict[str, tuple[str, ...]]  # each scenario's messages
    profiles: dict[str, str]  # each scenario's line
    plays: tuple[Play, ...]


async def run_ab(
    pairs: Sequence[Pair],
    models: Models,
    root: Path,
    source: Path,
    repeat: int = 1,
    variants: Sequence[Variant] = (Variant(),),
    inversions: Inversions = "scenario",
    on_event: OnEvent = print,
) -> ABReport:
    """Play every pair `repeat` times by both agents, PACo's then each variant of the geophysicist
    agent, on the lines of `source` (PACo's input folder); then write report.json in a new folder
    of `root`, where each play has its own: <scenario>/<agent>/<play>."""
    started_at = datetime.now(UTC)
    eval_id = f"ab-{started_at:%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"
    folder = root / eval_id
    inputs = build_inputs(source, folder / "inputs")
    # One images' cache for every play: the same lines, imaged once; not a result.
    cache = folder / CACHE_FOLDER
    plays: list[Play] = []
    for pair in pairs:
        quick = inversions == "quick" or (inversions == "scenario" and not pair.paco.full_inversion)
        prepared: Path | None = None
        if pair.truth is not None:
            # The line, made with the configuration its plays are judged by, which it keeps.
            prepared = folder / pair.name / "prepared"
            config = quick_config(folder / pair.name) if quick else None
            on_event(f"== {pair.name}: the line, its curves {pair.truth.name}'s")
            await known_truth.prepare(pair.truth, prepared, inputs, cache, config)
        for attempt in range(1, repeat + 1):
            if models.paco is not None:
                on_event(f"== {pair.name}, PACo's agent, play {attempt}")
                where = folder / pair.name / "paco" / str(attempt)
                paco = await play_paco(
                    pair, models, where, attempt, inputs, cache, quick, prepared, on_event
                )
                plays.append(_located(paco, folder))
            for variant in variants:
                on_event(f"== {pair.name}, {variant.label}, play {attempt}")
                where = folder / pair.name / variant.label.replace(" ", "-") / str(attempt)
                geo = await play_geo(
                    pair, models, variant, where, attempt, inputs, cache, quick, prepared, on_event
                )
                plays.append(_located(geo, folder))
    shutil.rmtree(cache, ignore_errors=True)
    report = ABReport(
        eval_id=eval_id,
        started_at=started_at,
        paco_model=models.paco_name if models.paco is not None else None,
        geo_model=models.geo_name,
        judge_model=models.judge_name if models.judge is not None else None,
        paco_prompts=paco_prompts.version(),
        geo_prompts=prompts.version(),
        repeat=repeat,
        inversions=inversions,
        questions={pair.name: pair.questions for pair in pairs},
        profiles={pair.name: pair.profile for pair in pairs},
        plays=tuple(plays),
    )
    (folder / "report.json").write_text(report.model_dump_json(indent=2))
    return report


async def play_paco(
    pair: Pair,
    models: Models,
    folder: Path,
    attempt: int,
    inputs: Path,
    cache: Path | None,
    quick: bool,
    prepared: Path | None = None,
    on_event: OnEvent = print,
) -> Play:
    """A play of PACo's agent, by PACo's runner, in `folder`: its scenario's checks and judge, and
    the measures; a known-truth scenario's line copied in first from `prepared`."""
    if models.paco is None:
        raise ValueError("PACo's agent has no model.")
    scenario = pair.paco
    if prepared is not None:
        scenario = replace(scenario, setup=_copier(prepared))
    meter = Meter(models.paco)
    clock = Clock()
    clock.start()
    result = await run_scenario(
        scenario,
        meter,
        models.paco_name or "",
        folder,
        models.judge,
        on_event,
        attempt,
        inputs,
        cache,
        quick_inversion=quick,
    )
    clock.stop()
    outputs = folder / "outputs"
    transcript = Transcript.model_validate_json((folder / "transcript.json").read_text())
    return Play(
        scenario=pair.name,
        agent="paco",
        label="paco",
        attempt=attempt,
        checks=result.checks,
        judge=result.judge,
        measures=measure("paco", outputs, transcript, clock, meter=meter),
        truth=_truth(pair, outputs, None),
        answer=result.answer,
        error=result.error,
        folder=str(folder),
    )


async def play_geo(
    pair: Pair,
    models: Models,
    variant: Variant,
    folder: Path,
    attempt: int,
    inputs: Path,
    cache: Path | None,
    quick: bool,
    prepared: Path | None = None,
    on_event: OnEvent = print,
) -> Play:
    """A play of the geophysicist agent, as PACo's runner plays PACo's, in `folder`: its server's
    outputs in `folder`/outputs, the conversation saved as `folder`/transcript.json; its checks,
    the judge's grades of its answer and its journal, and the measures."""
    outputs = folder / "outputs"
    config = quick_config(folder) if quick else None
    if prepared is not None:
        known_truth.copy_prepared(prepared, outputs)
    error: str | None = None
    refused: str | None = None
    meter = Meter(models.geo)
    clock = Clock()
    clock.start()
    with server_settings(outputs, inputs, cache, config):
        async with Client(geo_server.server) as client:
            host = await Host.start(
                client,
                meter,
                get_settings(),
                limits=models.limits,
                vision=variant.vision,
                without=variant.without,
                effort=variant.effort,
                on_event=on_event,
            )
            try:
                for question in pair.questions:
                    await host.answer(question)
            except (openai.APIConnectionError, openai.APIStatusError) as failed:
                if (why := lost(failed)) is not None:
                    # Not the model's doing: the play is lost, the evaluation goes on.
                    error = why
                    on_event(f"   (play lost: {error})")
                else:
                    refused = f"{type(failed).__name__}: {failed.message}"
                    on_event(f"   (refused by the model's server: {refused})")
        await _wait_for_jobs(outputs)
    clock.stop()
    transcript = host.transcript()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "transcript.json").write_text(transcript.model_dump_json(indent=2))
    trial = Trial(transcript, outputs, config)
    study = study_of(trial)
    judge: JudgeScore | None = None
    reasoning: JudgeScore | None = None
    if models.judge is not None and error is None:
        judge = await judge_answer(models.judge, pair.rubric, pair.questions, transcript)
        if study is not None:
            reasoning = await judge_journal(models.judge, study)
    return Play(
        scenario=pair.name,
        agent="geo",
        label=variant.label,
        attempt=attempt,
        checks=_scored(pair.checks, trial, error, refused),
        judge=judge,
        reasoning=reasoning,
        measures=measure("geo", outputs, transcript, clock, study, transcript.started_at, meter),
        truth=_truth(pair, outputs, study) if study is not None else None,
        answer=transcript.answer,
        error=error,
        folder=str(folder),
    )


def _scored(
    checks: Sequence[Check], trial: Trial, error: str | None, refused: str | None
) -> tuple[CheckResult, ...]:
    """The play's checks, as PACo's runner scores its own: none for a play lost; for a request
    the model's server refused, that alone."""
    if error is not None:
        return ()
    if refused is not None:
        return (
            CheckResult(name="the model's server took every request", passed=False, detail=refused),
        )
    return tuple(check(trial) for check in checks)


def _truth(pair: Pair, outputs: Path, study: Study | None) -> known_truth.TruthResult | None:
    """A known-truth scenario's score of the models the play kept."""
    run_folder = latest_run(outputs)
    if pair.truth is None or run_folder is None:
        return None
    return known_truth.score(pair.truth, run_folder, known_truth.kept(run_folder, study))


def _copier(prepared: Path) -> Callable[[Settings], None]:
    """A PACo scenario's setup copying the prepared line into the play's outputs."""

    def setup(settings: Settings) -> None:
        known_truth.copy_prepared(prepared, settings.output_dir)

    return setup


def _located(play: Play, root: Path) -> Play:
    """`play` with its folder relative to the evaluation's."""
    return play.model_copy(update={"folder": str(Path(play.folder).relative_to(root))})


async def _wait_for_jobs(outputs: Path) -> None:
    """Let the inversion the play left running finish, so that plays never overlap: the host
    follows each job to its end, but a play cut short leaves one."""
    deadline = time.monotonic() + _JOBS_TIMEOUT_S
    for path in outputs.glob("*/*/inversion.json"):
        job = InversionRecord.model_validate_json(path.read_text()).job_id
        while geo_server.JOBS.is_live(job) and time.monotonic() < deadline:
            await anyio.sleep(0.5)
