"""The answer the user reads (docs/geo_plan.md §4.9, §10, §11), written by code around the model's
text, as PACo's is: the model's draft goes into PACo's answer form, with the interpretation it
adds; code writes the rest from the study's records. In the middle of a run (a question for the
user, a budget spent): the scope, the model's text, its question, and where the work stands. The
report (P5) adds what was done, the windows left out, what was tried, each window's outcome, the
model's interpretation, the parameters used, each with its source (the model's reason, given by
the user, the plan's rule, or the preset's default), and the settings kept as the user gave them.
Numbers in the model's text that none of the results hold are flagged (PACo's R1)."""

import json
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, cast

from openai.types.chat import ChatCompletionMessageParam
from pydantic import Field, ValidationError
from sigpipe.masw.runs import RunManifest, find_run

from paco import prompts as paco_prompts
from paco.agent.answer import AnswerError, AnswerForm, unfound_numbers
from paco.agent.model import ChatModel, Filled
from paco.geo import prompts
from paco.geo.tools import decide, journal, models, pick
from paco.geo.tools.commit import Line, line_of
from paco.geo.tools.freeze import frozen
from paco.geo.tools.given import asks, given_of
from paco.geo.tools.invert import inversions_of
from paco.geo.tools.trials import MODE, NEAR_FIELD, PICKING, Study
from paco.qc.report import stretches, xmid_of
from paco.settings import Settings

DEFAULT = "the preset's default"
GIVEN = "given, in your request"
# The modes, as the report says them.
MODES_SAID = {
    "active": "each window stacks its shots' images",
    "passive-active": "each window correlates its farther shots and stacks them",
    "passive": "each window correlates the ambient noise's segments and stacks them",
}
STEP = "the plan's rule: one set of settings for the line, a window every receiver"
INTERPRETED = "Interpretation, by the model (not a measurement)"


class GeoAnswerForm(AnswerForm):
    """PACo's answer form, with the model's reading of the ground: an interpretation, not a
    measurement."""

    interpretation: str | None = Field(max_length=600)


# The form's JSON schema, as the model's output is constrained to it.
SCHEMA: dict[str, Any] = GeoAnswerForm.model_json_schema()


@dataclass(frozen=True)
class Written:
    """The answer form read, with the model's fills (one, or two when the first did not parse)."""

    form: GeoAnswerForm
    fills: tuple[Filled, ...]


async def write_answer(model: ChatModel, message: str, draft: str) -> Written:
    """The answer form for the model's `draft` to the user's `message`, strictly parsed; a form
    that does not parse is sent back once with its error. Raises AnswerError when the second
    does not parse either."""
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": f"{paco_prompts.prompt('answer')}\n{prompts.prompt('answer')}",
        },
    ]
    for example_message, example_draft, form in paco_prompts.answer_examples():
        messages.append(
            {"role": "user", "content": f"Message: {example_message}\nAnswer: {example_draft}"}
        )
        messages.append(
            {"role": "assistant", "content": json.dumps({**form, "interpretation": None})}
        )
    messages.append({"role": "user", "content": f"Message: {message}\nAnswer: {draft}"})
    fills: list[Filled] = []
    said = ""
    for _ in range(2):
        filled = await model.fill(messages, SCHEMA)
        fills.append(filled)
        try:
            return Written(GeoAnswerForm.model_validate_json(filled.content), tuple(fills))
        except ValidationError as error:
            said = "; ".join(item["msg"] for item in error.errors())
            messages.append({"role": "assistant", "content": filled.content})
            messages.append({"role": "user", "content": f"Not valid: {said}. Fill it again."})
    raise AnswerError(said)


def progress(scope_line: str, form: GeoAnswerForm, stands: str, sources: Sequence[str] = ()) -> str:
    """An answer in the middle of the run: the scope, the model's text and its numbers no result
    holds, its question, and where the work stands."""
    blocks = [scope_line, form.said.strip()]
    if unfound := unfound_numbers(form.said, [*sources, scope_line, stands]):
        blocks.append(f"Numbers not found in the tools' results: {', '.join(unfound)}.")
    if form.question:
        blocks.append(form.question)
    blocks.append(f"Where the work stands: {stands}")
    return "\n\n".join(block for block in blocks if block)


def report(
    study: Study,
    settings: Settings,
    scope_line: str,
    form: GeoAnswerForm,
    sources: Sequence[str] = (),
) -> str:
    """The report (P5): the scope, the model's text, what was done and left out, what was tried,
    each window's outcome, the model's interpretation, the parameters used and the settings kept
    as given."""
    said = f"{form.said} {form.interpretation or ''}"
    held = [*sources, scope_line, *(entry.text for entry in journal.entries(study))]
    blocks = [scope_line, form.said.strip()]
    if unfound := unfound_numbers(said, held):
        blocks.append(f"Numbers not found in the tools' results: {', '.join(unfound)}.")
    for title, items in (
        ("Done", done(study)),
        ("Left out", left_out(study, settings)),
        ("What I tried", tried(study)),
        ("Each window's outcome", outcomes(study, settings)),
    ):
        if items:
            blocks.append(f"{title}:\n" + "\n".join(f"- {item}" for item in items))
    if form.interpretation:
        blocks.append(f"{INTERPRETED}: {form.interpretation.strip()}")
    if form.question:
        blocks.append(form.question)
    for title, items in (
        ("Parameters used", parameters_used(study, settings)),
        ("Settings kept as you gave them", kept(study)),
        ("Held back by your settings", held_back(study, settings)),
    ):
        if items:
            blocks.append(f"{title}:\n" + "\n".join(f"- {item}" for item in items))
    return "\n\n".join(block for block in blocks if block)


def done(study: Study) -> list[str]:
    """What the run did, from the journal and the study's records: the survey, each commit, the
    model's picks, the decisions, the freeze."""
    found = journal.entries(study)
    lines: list[str] = []
    if (study.folder / "survey.json").exists():
        lines.append(f"the survey of {study.profile}: its geometry, limits and raw records")
    lines += [entry.text for entry in found if entry.kind == "commit"]
    line = line_of(study)
    if line is not None:
        own = [one for one in pick.picks_of(study, line.run_id) if one.recipe.how != "line"]
        if own:
            counts = Counter(one.recipe.how for one in own)
            said = ", ".join(f"{how} {count}" for how, count in counts.most_common())
            picks = (
                "a pick of a single window"
                if len(own) == 1
                else f"{len(own)} picks of single windows"
            )
            lines.append(f"{picks} ({said})")
    lines += [entry.text for entry in found if entry.kind == "freeze"]
    lines += [f"{entry.id} {entry.text}" for entry in found if entry.kind == "inversion"]
    return lines


def left_out(study: Study, settings: Settings) -> list[str]:
    """The windows rejected, curves then models, with why, by reason."""
    made = frozen(study)
    if made is None:
        return []
    by_reason: dict[str, list[float]] = {}
    for one in made.rejected:
        by_reason.setdefault(one.reason, []).append(one.xmid)
    lines = [f"{stretches(xmids)}: {reason}" for reason, xmids in by_reason.items()]
    by_model: dict[str, list[float]] = {}
    folder = find_run(made.run_id, settings)
    for unit, decision in models.current(study, folder, made.run_id).items():
        if decision.status == "rejected" and (xmid := xmid_of(unit)) is not None:
            by_model.setdefault(decision.reason or "", []).append(xmid)
    lines += [f"the models at {stretches(xmids)}: {reason}" for reason, xmids in by_model.items()]
    return lines


def tried(study: Study) -> list[str]:
    """The experiments that changed a decision: those a commit came from, or a conclusion or a
    decision rests on, one line each."""
    found = journal.entries(study)
    cited = {
        one
        for entry in found
        if entry.kind in ("commit", "conclusion", "decision")
        for one in entry.of
    }
    return [
        journal.line(entry) for entry in found if entry.kind == "experiment" and entry.id in cited
    ]


def outcomes(study: Study, settings: Settings) -> list[str]:
    """Each window's outcome: accepted or rejected (with the reason), as frozen; before the
    freeze, the decisions as they stand."""
    made = frozen(study)
    if made is not None:
        lines = [f"curves accepted: {stretches([curve.xmid for curve in made.curves])}"]
        if made.rejected:
            lines.append(f"curves rejected: {stretches([one.xmid for one in made.rejected])}")
        if inversions_of(study, made.run_id):
            folder = find_run(made.run_id, settings)
            decided = models.statuses(study, folder, made.run_id)
            for status in ("accepted", "rejected", "pending"):
                xmids = [
                    curve.xmid
                    for curve in made.curves
                    if decided.get(curve.unit, "pending") == status
                ]
                if xmids:
                    lines.append(f"models {status}: {stretches(xmids)}")
        return lines
    line = line_of(study)
    if line is None:
        return []
    folder = find_run(line.run_id, settings)
    manifest = RunManifest.model_validate_json((folder / "run.json").read_text())
    statuses = decide.statuses(study, folder, line.run_id)
    lines: list[str] = []
    for status in ("accepted", "rejected", "pending"):
        xmids = [
            window.xmid
            for window in manifest.windows
            if statuses.get(window.folder, "pending") == status
        ]
        if xmids:
            lines.append(f"{status}: {stretches(xmids)}")
    return lines


def parameters_used(study: Study, settings: Settings) -> list[str]:
    """The committed line's settings, each with its source: the model's reason (with its commit
    and experiment), given by the user, the plan's rule, or the preset's default."""
    line = line_of(study)
    if line is None:
        return []
    folder = find_run(line.run_id, settings)
    manifest = RunManifest.model_validate_json((folder / "run.json").read_text())
    preset = manifest.preset
    spacing = manifest.profile.receiver_spacing_m
    source = _sources(study, line)
    xmids = [window.xmid for window in manifest.windows]
    length = preset.masw.length
    mode = str(preset.mode)
    used = [
        f"{mode} mode ({MODES_SAID.get(mode, mode)}): {source(MODE)}",
        f"MASW windows of {length} receivers ({(length - 1) * spacing:g} m), {len(xmids)} "
        f"windows, xmid {min(xmids):.2f} to {max(xmids):.2f} m: {source('masw length')}",
        f"windows every receiver ({spacing:g} m): {STEP}",
    ]
    if mode != "passive":
        used.append(
            f"shots {_distances(preset.masw.distance_min, preset.masw.distance_max)} from a "
            f"window's middle: {source('masw distance_max', 'masw distance_min')}"
        )
    near = line.settings.get(NEAR_FIELD)
    if isinstance(near, int | float):
        used.append(
            f"no shot nearer than {near:g} m to a window's nearest receiver, where it has a "
            f"farther one: {source(NEAR_FIELD)}"
        )
    dispersion = getattr(preset, "dispersion", None)
    if dispersion is not None:
        used.append(
            f"phase shift {dispersion.fmin:g} to {dispersion.fmax:g} Hz: "
            f"{source('dispersion fmin', 'dispersion fmax')}"
        )
        used.append(
            f"velocities {dispersion.vmin:g} to {dispersion.vmax:g} m/s ({dispersion.nv} steps): "
            f"{source('dispersion vmin', 'dispersion vmax', 'dispersion nv')}"
        )
    stages = preset.model_dump(mode="json")
    for stage in (
        *("trigger", "muting", "slicing", "selection", "whitening", "normalization"),
        *("filtering", "stacking", "image_stacking"),
    ):
        dumped = stages.get(stage)
        if not isinstance(dumped, dict):
            continue
        dumped = cast(dict[str, Any], dumped)
        shown = ", ".join(f"{key} {value}" for key, value in dumped.items() if value is not None)
        keys = [f"{stage} {key}" for key in dumped]
        used.append(f"{stage}: {shown or 'none'}: {source(*keys)}")
    picking = line.settings.get(PICKING) or {}
    changed = json.dumps(picking, separators=(",", ":")) if picking else "none"
    keys = [f"{PICKING} {key}" for key in picking] if isinstance(picking, dict) else []
    recipes = pick.recipes(study, line.run_id)
    used.append(
        f"picking: PACo's values, the line's changes {changed}: "
        f"{source(*keys) if keys else DEFAULT}"
        + (
            f"; {len(recipes)} window{'s' if len(recipes) != 1 else ''} picked their own way, "
            "the model's (each in the study's picks.jsonl)"
            if recipes
            else ""
        )
    )
    return used + inversion_used(study)


def inversion_used(study: Study) -> list[str]:
    """The windows' inversions as they last ran, each with its source: the layering, the Vs
    bounds and the depth from each window's own curve through the checks before S4, with the
    values the model gave in its groups; its effort; its seed."""
    made = frozen(study)
    done = inversions_of(study, made.run_id) if made is not None else []
    last = {one.unit: one.parameters for one in done if one.parameters}
    if not last:
        return []
    ran = list(last.values())
    free = [one["free"] for one in ran if one.get("layering") == "free" and "free" in one]
    given: dict[str, Any] = {}
    for entry in journal.entries(study):
        if entry.kind == "inversion":
            for group in entry.data.get("groups", []):
                given |= group.get("parameters", {})
    return [
        f"inversion of {len(ran)} windows, "
        + (
            f"the layers chosen by the data (up to {_span(one['max_layers'] for one in free)}), "
            f"Vs {_span(one['vs_min'] for one in free)} to {_span(one['vs_max'] for one in free)} "
            f"m/s, interfaces {_span(one['depth_min'] for one in free)} to "
            f"{_span(one['depth_max'] for one in free)} m deep"
            if free
            else "the layers given"
        )
        + ": from each window's own curve, sigpipe's checks before S4"
        + (
            f", with the model's values {json.dumps(given, separators=(',', ':'))} where they "
            "passed"
            if given
            else ""
        ),
        f"{_span(one['n_iterations'] for one in ran)} iterations "
        f"({_span(one['n_burnin_iterations'] for one in ran)} burn-in), "
        f"{_span(one['n_chains'] for one in ran)} chains: the model's effort",
        "each window's inversion seeded: its seed in its attempt's notes and the study's "
        "inversions.jsonl, for the inversion to be made again exactly",
    ]


def _span(values: Iterable[Any]) -> str:
    """Values' range as said: "4", or "3-5"."""
    ordered = sorted({round(float(value), 2) for value in values})
    if not ordered:
        return "n/a"
    low, high = (f"{value:g}" for value in (ordered[0], ordered[-1]))
    return low if low == high else f"{low}-{high}"


def kept(study: Study) -> list[str]:
    """The settings the user gave, kept as they gave them."""
    return given_of(study).said()


def held_back(study: Study, settings: Settings) -> list[str]:
    """What the user's settings held back on the line: the windows the gates rejected for a
    change of one, the changes G5 asks of the inversion's, the bounds the checks kept."""
    made = frozen(study)
    run_id = (
        made.run_id if made is not None else (line.run_id if (line := line_of(study)) else None)
    )
    return asks(find_run(run_id, settings), given_of(study)) if run_id is not None else []


def _sources(study: Study, line: Line) -> Callable[..., str]:
    """Each setting's source, as said: the model's reason from its latest commit, given, or the
    preset's default."""
    reasons: dict[str, str] = {}
    for one in line.commits:
        where = f"L{one.number}" + (f", {one.experiment}" if one.experiment else "")
        for key, reason in one.reasons.items():
            reasons[key] = f"model: {reason} ({where})"
    given_keys = {
        f"{stage} {key}" for stage, values in given_of(study).line.items() for key in values
    }

    def source(*keys: str) -> str:
        for key in keys:
            if key in reasons:
                return reasons[key]
        if any(key in given_keys for key in keys):
            return GIVEN
        return DEFAULT

    return source


def _distances(near: float | None, far: float | None) -> str:
    if near is None and far is None:
        return "at any distance"
    if far is None:
        return f"beyond {near:g} m"
    return f"within {far:g} m" if near is None else f"{near:g} to {far:g} m"
