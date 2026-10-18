"""A labelled set of messages for the scope's form (paco.agent.scope): what each asks, in
English and French, with typos, negations, synonyms, work made by hand, positions, a run id
PACo never gives, and answers to the options a tool offered. The model's reading of each is
scored field by field against its label, once: the form is filled at temperature 0."""

import secrets
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from paco import prompts
from paco.agent.model import ChatModel
from paco.agent.scope import Context, Offer, ScopeError, read_scope

# The run the options and the conversation are on.
RUN = "20260930-161253-89f5"
# A message asking nothing: each case's label says what differs from it.
NOTHING: dict[str, Any] = {
    "process": False,
    "pick": False,
    "invert": False,
    "soils": False,
    "profile": None,
    "run_id": None,
    "positions_m": [],
    "length_receivers": None,
    "length_m": None,
    "step_receivers": None,
    "step_m": None,
    "compare_lengths_receivers": [],
    "compare_lengths_m": [],
    "redo": False,
    "replace_hand_work": False,
    "option": None,
    "workers": None,
    "mode": None,
}
# What an answer to an offer is scored on: the profile and run come from the conversation.
ANSWER = ("process", "pick", "invert", "soils", "redo", "replace_hand_work", "option")
# What a correction is scored on: the stage again; the profile and run, the conversation's.
CORRECTION = ("process", "pick", "invert", "soils", "redo")

ON_ACTIVE = Context(profile="active_test_profile", run_id=RUN)
# pick, on a run whose curves are there.
CURVES_THERE = Context(
    offers=(
        Offer("complete the 2 windows without a curve", f'pick(run_id="{RUN}", windows="missing")'),
        Offer("pick every window again", f'pick(run_id="{RUN}", windows="all")'),
        Offer("invert the curves as they are", f'invert(run_id="{RUN}")'),
        Offer("some windows only", f'pick(run_id="{RUN}", positions=["<m>"])'),
    ),
    profile="active_test_profile",
    run_id=RUN,
)
# pick every window again, over a curve picked by hand.
HAND_THERE = Context(
    offers=(
        Offer("keep it as it is", f'pick(run_id="{RUN}", windows="all", hand="keep")'),
        Offer(
            "replace it, theirs set aside in the window's by_hand folder",
            f'pick(run_id="{RUN}", windows="all", hand="replace")',
        ),
    ),
    profile="active_test_profile",
    run_id=RUN,
)
# run_processing, on a profile whose run has images: a new run, or that run to work on.
IMAGES_THERE = Context(
    offers=(
        Offer("a new run", 'run_processing(profile="active_test_profile", again=true)'),
        Offer(
            f"work on run {RUN} (active, windows of 24 receivers, images)",
            f'pick(run_id="{RUN}")',
        ),
    ),
    profile="active_test_profile",
    run_id=RUN,
)


@dataclass(frozen=True)
class Case:
    message: str
    expected: Mapping[str, Any]  # the fields that differ from a message asking nothing
    context: Context = field(default_factory=Context)
    scored: tuple[str, ...] | None = None  # the fields scored (None: every one)


CASES: tuple[Case, ...] = (
    # Looking only.
    Case("Which seismic profiles can I process?", {}),
    Case(f"What does run {RUN} hold?", {"run_id": RUN}),
    Case(f"Just look at what run {RUN} holds, don't change anything.", {"run_id": RUN}),
    # The stages asked, in words and in French.
    Case("Process active_test_profile.", {"process": True, "profile": "active_test_profile"}),
    Case(
        "Process active_test_profile and pick its curves.",
        {"process": True, "pick": True, "profile": "active_test_profile"},
    ),
    Case(
        "Pick and invert active_test_profile.",
        {"pick": True, "invert": True, "profile": "active_test_profile"},
    ),
    Case("Invert active_test_profile.", {"invert": True, "profile": "active_test_profile"}),
    Case("invret active_test_profile", {"invert": True, "profile": "active_test_profile"}),
    Case(
        "Give me the shear-wave velocity profile of active_test_profile.",
        {"invert": True, "profile": "active_test_profile"},
    ),
    Case(
        "Donne-moi le profil de vitesse des ondes S de active_test_profile.",
        {"invert": True, "profile": "active_test_profile"},
    ),
    Case(
        "Process active_test_profile with windows of 24 receivers, then pick and invert.",
        {
            "process": True,
            "pick": True,
            "invert": True,
            "profile": "active_test_profile",
            "length_receivers": 24,
        },
    ),
    # The lengths a comparison names, in the unit the message gives.
    Case(
        "On active_test_profile, compare windows of 3 m and of 6 m: which reaches deeper?",
        {"process": True, "profile": "active_test_profile", "compare_lengths_m": [3, 6]},
    ),
    Case(
        "Compare des fenêtres de 12, 24 et 48 capteurs sur active_test_profile.",
        {
            "process": True,
            "profile": "active_test_profile",
            "compare_lengths_receivers": [12, 24, 48],
        },
    ),
    # The windows, in the unit the message gives.
    Case(
        "Process active_test_profile with 6 m windows every 3 m.",
        {"process": True, "profile": "active_test_profile", "length_m": 6, "step_m": 3},
    ),
    Case(
        "Traite active_test_profile avec des fenêtres de 12 capteurs, tous les 12 capteurs.",
        {
            "process": True,
            "profile": "active_test_profile",
            "length_receivers": 12,
            "step_receivers": 12,
        },
    ),
    # The workers the work may use.
    Case(
        "Process active_test_profile and invert it, use 10 workers.",
        {"process": True, "invert": True, "profile": "active_test_profile", "workers": 10},
    ),
    Case(
        "Inverse active_test_profile sur 4 cœurs.",
        {"invert": True, "profile": "active_test_profile", "workers": 4},
    ),
    # The processing mode, in its words.
    Case(
        "Process active_test_profile using the passive-active method and invert it.",
        {
            "process": True,
            "invert": True,
            "profile": "active_test_profile",
            "mode": "passive-active",
        },
    ),
    Case(
        "Traite active_test_profile en mode actif.",
        {"process": True, "profile": "active_test_profile", "mode": "active"},
    ),
    Case(
        "Process active_test_profile using active-passive method but phase weighted stack at power 1.",
        {"process": True, "profile": "active_test_profile", "mode": "passive-active"},
    ),
    Case(
        "Process passive_test_profile and give me the soils.",
        {"process": True, "soils": True, "profile": "passive_test_profile"},
    ),
    Case(
        "What is the water table under active_test_profile?",
        {"soils": True, "profile": "active_test_profile"},
    ),
    Case(
        "Quelle est la profondeur de la nappe sous active_test_profile ?",
        {"soils": True, "profile": "active_test_profile"},
    ),
    Case(
        "Give me the Vs section and the soil types along active_test_profile.",
        {"invert": True, "soils": True, "profile": "active_test_profile"},
    ),
    Case(
        "Pick M0 and M1 of active_test_profile.", {"pick": True, "profile": "active_test_profile"}
    ),
    Case(
        "Pick active_test_profile with a threshold of 0.4.",
        {"pick": True, "profile": "active_test_profile"},
    ),
    Case(
        "Pick active_test_profile between 12 and 40 Hz.",
        {"pick": True, "profile": "active_test_profile"},
    ),
    Case(
        "Pick active_test_profile down to 12 Hz.", {"pick": True, "profile": "active_test_profile"}
    ),
    Case(
        "Invert active_test_profile with 4 layers and Vs up to 800 m/s.",
        {"invert": True, "profile": "active_test_profile"},
    ),
    # A request no stage makes asks none: the host then runs nothing.
    Case(
        "Make me a 3D shear-wave model of active_test_profile.", {"profile": "active_test_profile"}
    ),
    # Negations.
    Case(
        "Pick the curves of active_test_profile but don't invert them.",
        {"pick": True, "profile": "active_test_profile"},
    ),
    Case(
        "Pick les courbes de active_test_profile mais ne les inverse pas.",
        {"pick": True, "profile": "active_test_profile"},
    ),
    Case(
        "Process active_test_profile but don't pick anything.",
        {"process": True, "profile": "active_test_profile"},
    ),
    # Work already there, and work made by hand.
    Case(
        "Process active_test_profile again from scratch.",
        {"process": True, "profile": "active_test_profile", "redo": True},
    ),
    Case(
        "Process active_test_profile again from scratch and give me the water table.",
        {"process": True, "soils": True, "profile": "active_test_profile", "redo": True},
    ),
    Case(
        "Pick every window of active_test_profile again.",
        {"pick": True, "profile": "active_test_profile", "redo": True},
    ),
    Case(
        "Refais le picking de active_test_profile depuis le début.",
        {"pick": True, "profile": "active_test_profile", "redo": True},
    ),
    Case(
        "Redo the inversion of active_test_profile.",
        {"invert": True, "profile": "active_test_profile", "redo": True},
    ),
    Case(
        "Complete the windows of active_test_profile that have no curve.",
        {"pick": True, "profile": "active_test_profile"},
    ),
    Case(
        "Pick every window of active_test_profile again, including the curve I picked by hand.",
        {"pick": True, "profile": "active_test_profile", "redo": True, "replace_hand_work": True},
    ),
    Case(
        "Pick every window of active_test_profile again but keep the curves I picked by hand.",
        {"pick": True, "profile": "active_test_profile", "redo": True},
    ),
    # Positions, and a run id PACo never gives.
    Case(
        "Invert only the window at 9 m of my latest active_test_profile run.",
        {"invert": True, "profile": "active_test_profile", "positions_m": [9.0]},
    ),
    Case(
        "Inverse la fenêtre à 12,5 m de active_test_profile.",
        {"invert": True, "profile": "active_test_profile", "positions_m": [12.5]},
    ),
    Case(
        "Pick active_test_profile at 3 m and 15 m.",
        {"pick": True, "profile": "active_test_profile", "positions_m": [3.0, 15.0]},
    ),
    Case(f"Invert run {RUN}.", {"invert": True, "run_id": RUN}),
    Case(
        "Invert run 2026-09-30 of active_test_profile.",
        {"invert": True, "profile": "active_test_profile"},
    ),
    # The conversation's profile and run, and answers to the options offered.
    Case(
        "Invert them.", {"invert": True, "profile": "active_test_profile", "run_id": RUN}, ON_ACTIVE
    ),
    # A correction of what was done: the stage its setting is for, again.
    Case(
        "no i asked for phase-weighted stack power 1",
        {"process": True, "redo": True},
        ON_ACTIVE,
        CORRECTION,
    ),
    Case(
        "No, I wanted the curves picked with a threshold of 0.4.",
        {"pick": True, "redo": True},
        ON_ACTIVE,
        CORRECTION,
    ),
    Case("The second one.", {"pick": True, "option": 2}, CURVES_THERE, ANSWER),
    Case("Oui, la première.", {"pick": True, "option": 1}, CURVES_THERE, ANSWER),
    Case("Keep it.", {"pick": True, "option": 1}, HAND_THERE, ANSWER),
    Case(
        "Work on that run, and invert it.",
        {"pick": True, "invert": True, "option": 2},
        IMAGES_THERE,
        ANSWER,
    ),
    Case(
        "A new run.",
        {"process": True, "option": 1},
        IMAGES_THERE,
        ("process", "pick", "invert", "soils", "option"),
    ),
    # A typo is no answer to the options: it chooses none.
    Case("%", {}, CURVES_THERE, ANSWER),
)


class CaseResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    message: str
    # The fields read wrong: their label, then what the model read.
    wrong: dict[str, tuple[Any, Any]]
    error: str | None = None  # no form parsed, twice
    duration_s: float

    @property
    def passed(self) -> bool:
        return not self.wrong and self.error is None


class ScopeReport(BaseModel):
    """The scope set read once by a model: written as scopes.json."""

    model_config = ConfigDict(frozen=True)

    eval_id: str
    model: str
    prompt_version: str
    started_at: datetime
    results: tuple[CaseResult, ...]


async def read_scopes(
    model: ChatModel,
    model_name: str,
    folder: Path,
    cases: Sequence[Case] = CASES,
    on_event: Callable[[str], None] = print,
) -> ScopeReport:
    """Every case read by `model` and scored; the report written in `folder`/<eval_id>/."""
    started_at = datetime.now(UTC)
    eval_id = f"scopes-{started_at:%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"
    results: list[CaseResult] = []
    for case in cases:
        start = time.perf_counter()
        try:
            read = await read_scope(model, case.message, case.context)
        except ScopeError as error:
            result = CaseResult(
                message=case.message,
                wrong={},
                error=str(error),
                duration_s=round(time.perf_counter() - start, 3),
            )
        else:
            result = CaseResult(
                message=case.message,
                wrong=_wrong(case, read.scope.model_dump()),
                duration_s=round(time.perf_counter() - start, 3),
            )
        on_event(f"{'ok ' if result.passed else 'NO '} {case.message}")
        results.append(result)
    report = ScopeReport(
        eval_id=eval_id,
        model=model_name,
        prompt_version=prompts.version(),
        started_at=started_at,
        results=tuple(results),
    )
    (folder / eval_id).mkdir(parents=True, exist_ok=True)
    (folder / eval_id / "scopes.json").write_text(report.model_dump_json(indent=2))
    return report


def format_scope_report(report: ScopeReport) -> str:
    passed = sum(result.passed for result in report.results)
    lines = [
        f"Scopes {report.eval_id} of {report.model} on {report.prompt_version}: "
        f"{passed} of {len(report.results)} read right.",
    ]
    for result in report.results:
        if result.error is not None:
            lines.append(f"  {result.message}\n    no form: {result.error}")
        elif result.wrong:
            said = "; ".join(
                f"{name} {expected!r} -> {read!r}"
                for name, (expected, read) in result.wrong.items()
            )
            lines.append(f"  {result.message}\n    {said}")
    return "\n".join(lines)


def _wrong(case: Case, read: Mapping[str, Any]) -> dict[str, tuple[Any, Any]]:
    expected = {**NOTHING, **case.expected}
    names = case.scored if case.scored is not None else tuple(expected)
    return {
        name: (expected[name], read[name])
        for name in names
        if not _same(expected[name], read[name])
    }


def _same(expected: Any, read: Any) -> bool:  # noqa: ANN401
    if isinstance(expected, list) and isinstance(read, list):
        return len(expected) == len(read) and all(
            abs(float(a) - float(b)) < 1e-6 for a, b in zip(expected, read, strict=True)
        )
    return expected == read
