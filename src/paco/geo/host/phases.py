"""The run's phases (docs/geo_plan.md §4): P0 the objective, P1 the survey, P2 the curves, P3 the
freeze, P4 the inversion, P5 the report. A study's phase is read from its records, never kept in
memory (§14): the journal's objective and report, the survey, the frozen curves. P3 is the freeze
itself, the one way from P2 to P4. Each phase offers the model its tools (§6) and its manual."""

from collections.abc import Sequence
from typing import Literal

from mcp.types import Tool

from paco.geo.tools import journal
from paco.geo.tools.survey import SAVED
from paco.geo.tools.trials import CURVES, STUDIES, STUDY_ID, Study, new_study
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
            "try_settings",
            "commit",
            "pick",
            "decide",
            "freeze",
        }
    ),
    "P4": frozenset({*ALWAYS, "observe", "look", "advise", "invert", "job_status", "decide"}),
}
# Each phase's manual (paco.geo/prompts), which loads only in it.
MANUALS: dict[Phase, str] = {
    "P1": "manual_survey",
    "P2": "manual_curves",
    "P4": "manual_inversion",
}


def phase_of(study: Study) -> Phase:
    """The study's phase, from its records: P0 before the objective, P1 before the survey, P2
    before the freeze, P4 after it, P5 once the report is given."""
    kinds = {entry.kind for entry in journal.entries(study)}
    if "report" in kinds:
        return "P5"
    if "objective" not in kinds:
        return "P0"
    if (study.folder / CURVES).exists():
        return "P4"
    if not (study.folder / SAVED).exists():
        return "P1"
    return "P2"


def study_for(profile: str, settings: Settings) -> Study:
    """The study a conversation about `profile` works on: its newest study not reported yet,
    which it resumes (§14), else a new one."""
    folder = settings.output_dir / profile / STUDIES
    found = sorted(path for path in folder.glob("*") if STUDY_ID.fullmatch(path.name))
    for path in reversed(found):
        study = Study(profile, path)
        if phase_of(study) != "P5":
            return study
    return new_study(profile, settings)


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
