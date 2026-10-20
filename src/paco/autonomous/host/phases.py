"""The run's phases (docs/autonomous_plan.md §4): P0 the objective, P1 the survey, P2 the curves, P3 the
freeze, P4 the inversion, P5 the report. A study's phase is read from its records, never kept in
memory (§14): the journal's objective and report, the survey, the frozen curves. P3 is the freeze
itself, the one way from P2 to P4. Each phase offers the model its tools (§6) and its manual."""

from collections.abc import Sequence
from typing import Literal

from mcp.types import Tool
from sigpipe.masw.runs import RunError, find_run

from paco.autonomous.host.scope import GeoScope
from paco.autonomous.tools import journal
from paco.autonomous.tools.commit import line_of
from paco.autonomous.tools.freeze import frozen
from paco.autonomous.tools.given import Given
from paco.autonomous.tools.survey import SAVED
from paco.autonomous.tools.trials import CURVES, STUDIES, STUDY_ID, Study, new_study
from paco.settings import Settings

type Phase = Literal["P0", "P1", "P2", "P3", "P4", "P5"]
NAMES: dict[Phase, str] = {
    "P0": "objective",
    "P1": "survey",
    "P2": "curves",
    "P3": "freeze",
    "P4": "inversion",
    "P5": "report",
}
# PACo's read-only tools and the journal, in every phase (§6).
ALWAYS = ("inspect", "preset_settings", "inversion_settings", "journal")
# The tools each phase the model works in offers.
TOOLS: dict[Phase, frozenset[str]] = {
    "P1": frozenset({*ALWAYS, "survey", "look"}),
    "P2": frozenset(
        {
            *ALWAYS,
            "survey",
            "observe",
            "look",
            "advise",
            "search",
            "try_settings",
            "commit",
            "pick",
            "decide",
            "freeze",
        }
    ),
    "P4": frozenset(
        {
            *ALWAYS,
            "observe",
            "look",
            "advise",
            "invert",
            "job_status",
            "decide",
            "petro_models",
            "invert_petro",
        }
    ),
}
# The journal's entries made before the survey is said: the objective and the settings the user
# typed. Any other closes the survey once it is made: the model's first word on it, or what came
# after (an experiment, a commit, a reopened study).
UNSURVEYED = frozenset({"objective", "settings"})
# Each phase's manual (paco.autonomous/prompts), which loads only in it.
MANUALS: dict[Phase, str] = {
    "P1": "manual_survey",
    "P2": "manual_curves",
    "P4": "manual_inversion",
}


def phase_of(study: Study) -> Phase:
    """The study's phase, from its records: P0 before the objective, P1 until the survey is made
    and said in the journal, P2 before the freeze, P4 after it, P5 once the report is given,
    until the study opens again."""
    kinds = [entry.kind for entry in journal.entries(study)]
    if _last(kinds, "report") > _last(kinds, "reopen"):
        return "P5"
    if "objective" not in kinds:
        return "P0"
    if (study.folder / CURVES).exists():
        return "P4"
    if not (study.folder / SAVED).exists() or set(kinds) <= UNSURVEYED:
        return "P1"
    return "P2"


def _last(kinds: Sequence[str], kind: str) -> int:
    """Where `kind` comes last in `kinds`; -1 when it never does."""
    return max((index for index, one in enumerate(kinds) if one == kind), default=-1)


def wanted(scope: GeoScope, typed: Given) -> Literal["P2", "P4"] | None:
    """Where a message after the report takes the study: back to the curves when it asks to
    process or pick again, or gives their settings; to the inversion when it asks for models or
    soils, or gives the inversion's settings; nowhere for any other (the report answers it)."""
    if scope.process or scope.pick or scope.redo or typed.mode or typed.line:
        return "P2"
    if scope.invert or scope.soils or typed.inversion:
        return "P4"
    return None


def reopen(study: Study, back: Literal["P2", "P4"], request: str) -> journal.Entry:
    """`study` opened again after its report for the user's `request`, its journal and budgets
    kept: back to the curves, the freeze undone (its frozen curves set aside, as
    curves-B<n>.json), or to the inversion of the frozen curves."""
    made = study.folder / CURVES
    if back == "P2" and made.exists():
        made.replace(study.folder / f"curves-{journal.next_id(study, 'reopen')}.json")
    return journal.reopen(study, NAMES[back], request)


def study_for(profile: str, settings: Settings) -> tuple[Study, str | None]:
    """The study a conversation about `profile` works on: its newest study not reported yet,
    which it resumes (§14), else a new one. A new one too when that study's run is gone (deleted
    outside PACo: nothing to go on from), and why, said while that study is the newest."""
    folder = settings.output_dir / profile / STUDIES
    found = sorted(path for path in folder.glob("*") if STUDY_ID.fullmatch(path.name))
    for path in reversed(found):
        study = Study(profile, path)
        if phase_of(study) == "P5":
            continue
        if (gone := lost_run(study, settings)) is None:
            return study, None
        said = f"Study {study.id} cannot go on: its run {gone} is gone. A new study."
        return new_study(profile, settings), said if path == found[-1] else None
    return new_study(profile, settings), None


def lost_run(study: Study, settings: Settings) -> str | None:
    """The run `study`'s line or frozen curves are on, when it is no longer there."""
    line, made = line_of(study), frozen(study)
    for run_id in dict.fromkeys(one.run_id for one in (line, made) if one is not None):
        try:
            find_run(run_id, settings)
        except RunError:
            return run_id
    return None


def offered(phase: Phase, tools: Sequence[Tool]) -> list[Tool]:
    """The tools of `tools` that `phase` offers, in their order."""
    names = TOOLS.get(phase, frozenset())
    return [tool for tool in tools if tool.name in names]


def refusal(phase: Phase, tool: str) -> str | None:
    """Why `tool` is not called in `phase`: a tool of another phase; none when it is offered."""
    if tool in TOOLS.get(phase, frozenset()):
        return None
    where: list[Phase] = [name for name, tools in TOOLS.items() if tool in tools]
    later = (
        f" It belongs to {', '.join(f'{one} ({NAMES[one]})' for one in where)}." if where else ""
    )
    return f"Not called: {tool} is not offered in {phase} ({NAMES[phase]}).{later}"
