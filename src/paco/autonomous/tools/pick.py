"""pick (docs/autonomous_plan.md §4.6, §6): PAC's Dispersion picking page as a tool, each window's M0
picked as the model says. The tracker with the window's own parameters (auto); its corridor centred
on a guide, points or the neighbours' median curve (guide); on the maxima inside a polygon drawn on
the image (lasso); or the curve deleted. Each pick is PACo's (paco.qc.judging.pick_windows): saved
in PAC's layout, judged by G3 against G2's coherent band, and logged as the window's picking
attempt, `backtrack` (a window the model redoes, §2.3). Then G4 judges the line.

A lasso's maxima (sigpipe's pick_lasso) guide the tracker inside the polygon's band, so that G3
judges the pick as any other and paco-replay makes it again: a curve drawn from the polygon alone
would be neither.

The study keeps every pick (picks.jsonl): the window's recipe, so that a later commit of the line
picks it the same way, and what G3 said of it, so that decide reads what was tried (§4.6)."""

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sigpipe.algorithms.picking.dispersion.lasso import pick_lasso
from sigpipe.algorithms.picking.dispersion.tracking import PickingParameters
from sigpipe.base.dispersion_curve import Mode
from sigpipe.base.dispersion_image import DispersionImage
from sigpipe.masw.picks import CURVES_FILE, save_picks_figures
from sigpipe.masw.quality.line import Series
from sigpipe.masw.runs import RunManifest, load_image, window_folders

from paco.autonomous.sensors.flags import FlagClass, flag_class
from paco.autonomous.tools.trials import Study, check_open
from paco.qc.attempts import invalidate
from paco.qc.config import QCConfig
from paco.qc.curves import imaged_windows
from paco.qc.g4_profile import LINE, ProfileThresholds, median_guide
from paco.qc.judging import judge_line, pick_windows, saved_m0
from paco.qc.log import afresh, append_attempt, attempts_of, latest, read_attempts
from paco.qc.models import Attempt, Flag, GateResult, Override
from paco.qc.report import line_step, stretches, xmid_of

PICKS = "picks.jsonl"  # in the study's folder
# The trigger of a window the model redoes, as PAC knows it (§2.3).
BACKTRACK = "backtrack"
NEIGHBOURS = "neighbours"
M0 = Mode("M", 0)
# The most rows an answer lists; past it, the windows that do not pass.
ROWS = 30
type How = Literal["line", "auto", "guide", "lasso", "delete"]

logger = logging.getLogger(__name__)


class Recipe(BaseModel):
    """How a window is picked: with the line's picking (line), its parameters changed (auto), its
    corridor on a guide, points or the neighbours' median curve (guide), on a lasso's maxima
    (lasso), or its curve deleted (delete); `picking` changes the line's parameters for each."""

    model_config = ConfigDict(frozen=True)

    how: How
    picking: dict[str, Any] = Field(default_factory=dict[str, Any])
    guide: tuple[tuple[float, float], ...] | Literal["neighbours"] | None = None
    polygon: tuple[tuple[float, float], ...] | None = None


class Picked(BaseModel):
    """A window's pick as the study keeps it: the run, the window, its recipe, the parameters the
    tracker ran with, G3's verdict and flags (each with its class), or why it was not picked."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    unit: str
    recipe: Recipe
    parameters: dict[str, Any] | None = None  # none: deleted, or not picked
    verdict: str | None = None  # G3's; none: no pick
    flags: dict[str, FlagClass] = Field(default_factory=dict[str, FlagClass])
    error: str | None = None
    reapplied: bool = False  # its recipe applied again by a commit of the line
    at: datetime


@dataclass(frozen=True)
class Picks:
    """A pick call: its recipe, each window's pick and G3's result, then G4's on the line."""

    recipe: Recipe
    picked: dict[str, Picked]
    g3: dict[str, GateResult]
    g4: tuple[GateResult, ...]


def picks_of(study: Study, run_id: str | None = None) -> list[Picked]:
    """The study's picks in their order; with `run_id`, that run's alone."""
    path = study.folder / PICKS
    if not path.exists():
        return []
    found = [Picked.model_validate_json(line) for line in path.read_text().splitlines() if line]
    return [one for one in found if run_id is None or one.run_id == run_id]


def recipes(study: Study, run_id: str) -> dict[str, Recipe]:
    """Each window's recipe on run `run_id`: its latest pick's, unless the line picked it last."""
    found: dict[str, Recipe] = {}
    for one in picks_of(study, run_id):
        if one.recipe.how == "line":
            found.pop(one.unit, None)
        elif one.error is None:
            found[one.unit] = one.recipe
    return found


def check_recipe(recipe: Recipe, base: PickingParameters) -> None:
    """Refused, saying what to change, when `recipe` cannot be picked: parameters the tracker
    does not take, a guide or a polygon missing, or given to another way of picking."""
    if recipe.how == "line":
        raise ValueError("how: auto, guide, lasso or delete.")
    if recipe.guide is not None and recipe.how != "guide":
        raise ValueError("guide is for how=guide.")
    if recipe.polygon is not None and recipe.how != "lasso":
        raise ValueError("polygon is for how=lasso.")
    if recipe.how == "guide" and recipe.guide is None:
        raise ValueError(f"how=guide: give guide, points [f Hz, v m/s] or '{NEIGHBOURS}'.")
    if recipe.how == "guide" and recipe.guide != NEIGHBOURS and len(recipe.guide or ()) < 2:
        raise ValueError("guide: 2 points [f Hz, v m/s] at least.")
    if recipe.how == "lasso" and len(recipe.polygon or ()) < 3:
        raise ValueError("how=lasso: give polygon, 3 vertices [f Hz, v m/s] at least.")
    _validated({**base.model_dump(), **recipe.picking})


def pick(
    study: Study,
    run_folder: Path,
    units: Sequence[str],
    recipe: Recipe,
    base: PickingParameters,
    config: QCConfig,
    workers: int,
) -> Picks:
    """`units` of the committed line in `run_folder` picked as `recipe` says, from the line's
    picking `base`, then G4 over the line."""
    check_open(study)
    check_recipe(recipe, base)
    manifest = manifest_of(run_folder)
    ready = imaged_windows(run_folder, manifest)
    if unready := [unit for unit in units if unit not in ready]:
        raise ValueError(
            f"Nothing to pick at {_where(unready)}: no image, or one G2 rejected. Reject "
            "them (decide) with G2's reasons."
        )
    picked = apply(study, run_folder, manifest, dict.fromkeys(units, recipe), base, config, workers)
    attempts = read_attempts(run_folder)
    g3: dict[str, GateResult] = {}
    for unit in units:
        attempt = latest(attempts, unit, "picking")
        if picked[unit].verdict is not None and attempt is not None and "G3" in attempt.results:
            g3[unit] = attempt.results["G3"]
    return Picks(recipe, picked, g3, finish(run_folder, manifest, config))


def apply(
    study: Study,
    run_folder: Path,
    manifest: RunManifest,
    recipes: Mapping[str, Recipe],
    base: PickingParameters,
    config: QCConfig,
    workers: int,
    reapplied: bool = False,
) -> dict[str, Picked]:
    """Each window of `recipes` picked as its recipe says, logged as the model's (`backtrack`)
    and kept in the study: those guided by their neighbours once the others are picked, the
    deletions last. `reapplied`: by a commit of the line."""
    attempts = read_attempts(run_folder)
    bands = {unit: _band(attempts, unit) for unit in recipes}
    jobs: dict[str, tuple[PickingParameters, tuple[float, float] | None]] = {}
    errors: dict[str, str] = {}
    for unit, recipe in recipes.items():
        if recipe.how == "delete" or recipe.guide == NEIGHBOURS:
            continue
        try:
            jobs[unit] = (_parameters(recipe, base, run_folder / unit), bands[unit])
        except ValueError as error:
            errors[unit] = str(error)
    results = pick_windows(run_folder, jobs, config, BACKTRACK, workers) if jobs else {}
    if guided := [unit for unit, recipe in recipes.items() if recipe.guide == NEIGHBOURS]:
        passing = passing_curves(run_folder, manifest, read_attempts(run_folder))
        along: dict[str, tuple[PickingParameters, tuple[float, float] | None]] = {}
        for unit in guided:
            left, right = neighbours(unit, manifest, passing, config.profile)
            guide = median_guide([*left, *right])
            if not guide:
                errors[unit] = "no neighbour's curve passes G3 within G4's reach: nothing to follow"
                continue
            values = {**base.model_dump(), **recipes[unit].picking, "guide": guide}
            along[unit] = (_validated(values), bands[unit])
        if along:
            results |= pick_windows(run_folder, along, config, BACKTRACK, workers)
            jobs |= along
    for unit, recipe in recipes.items():
        if recipe.how == "delete":
            _delete(run_folder, unit)
    now = datetime.now(UTC)
    picked = {
        unit: Picked(
            run_id=manifest.run_id,
            unit=unit,
            recipe=recipe,
            parameters=jobs[unit][0].model_dump(mode="json") if unit in jobs else None,
            verdict=results[unit].verdict if unit in results else None,
            flags=_classes(results[unit].flags) if unit in results else {},
            error=errors.get(unit),
            reapplied=reapplied,
            at=now,
        )
        for unit, recipe in recipes.items()
    }
    keep(study, picked.values())
    return picked


def keep(study: Study, picked: Iterable[Picked]) -> None:
    """Picks appended to the study's record, one JSON line each."""
    with (study.folder / PICKS).open("a") as file:
        for one in picked:
            file.write(one.model_dump_json() + "\n")


def line_picks(
    run_id: str, results: Mapping[str, GateResult], parameters: PickingParameters
) -> list[Picked]:
    """The line's own picks, as a commit makes them: the line's picking, G3's results."""
    now = datetime.now(UTC)
    return [
        Picked(
            run_id=run_id,
            unit=unit,
            recipe=Recipe(how="line"),
            parameters=parameters.model_dump(mode="json"),
            verdict=result.verdict,
            flags=_classes(result.flags),
            at=now,
        )
        for unit, result in results.items()
    ]


def finish(run_folder: Path, manifest: RunManifest, config: QCConfig) -> tuple[GateResult, ...]:
    """G4 over the line once its windows are picked, then the picks' figures, as PAC shows them
    (best effort, as PACo's). G4's results."""
    results = judge_line(run_folder, manifest, config)
    try:
        save_picks_figures(run_folder, window_folders(run_folder))
    except Exception:
        logger.exception("Could not draw the picks' figures of %s", run_folder)
    return results


def passing_curves(
    run_folder: Path, manifest: RunManifest, attempts: tuple[Attempt, ...]
) -> list[Series]:
    """The windows' curves whose latest pick passed G3, along the line, as G4 reads them."""
    found: list[Series] = []
    for window in manifest.windows:
        picked = latest(attempts, window.folder, "picking")
        g3 = picked.results.get("G3") if picked is not None else None
        if g3 is None or g3.verdict != "pass":
            continue
        curve = saved_m0(run_folder / window.folder / CURVES_FILE)
        if curve is not None and len(curve.fs):
            found.append(Series.from_curve(window.folder, window.xmid, curve))
    return found


def neighbours(
    unit: str, manifest: RunManifest, passing: Sequence[Series], thresholds: ProfileThresholds
) -> tuple[list[Series], list[Series]]:
    """`unit`'s neighbours among the `passing` curves, as G4 takes them: the nearest
    `neighbours / 2` on each side, within `max_gap_steps` of the line's steps."""
    xmids = {window.folder: window.xmid for window in manifest.windows}
    step = line_step(xmids.values())
    reach = thresholds.max_gap_steps * step if step > 0 else float("inf")
    per_side = max(1, thresholds.neighbours // 2)
    here = xmids[unit]
    others = [one for one in passing if one.unit != unit]
    left = sorted(
        (one for one in others if here - reach <= one.xmid < here), key=lambda one: -one.xmid
    )
    right = sorted(
        (one for one in others if here < one.xmid <= here + reach), key=lambda one: one.xmid
    )
    return left[:per_side], right[:per_side]


def lasso_guide(
    image: DispersionImage, polygon: Sequence[tuple[float, float]]
) -> tuple[tuple[float, float], ...]:
    """The maxima inside `polygon` on `image` (sigpipe's pick_lasso: cleaned, resampled over
    wavelength) as the tracker's guide, thinned as G4 thins one."""
    lassoed = pick_lasso(image, polygon, M0)
    curves = lassoed.dispersion_curves.dispersion_curves if lassoed.dispersion_curves else ()
    curve = next(one for one in curves if one.mode == M0)
    return median_guide([Series.from_curve("lasso", 0.0, curve)])


def manifest_of(run_folder: Path) -> RunManifest:
    return RunManifest.model_validate_json((run_folder / "run.json").read_text())


def describe_picks(picks: Picks) -> str:
    """A pick call as the model reads it: each window's G3 verdict, its flags with their class
    and hint, what its curve keeps; then G4 on the line."""
    picked = picks.picked
    head = f"Picked {len(picked)} window{'s' if len(picked) > 1 else ''}, {_recipe(picks.recipe)}:"
    shown = list(picked)
    if len(shown) > ROWS:
        shown = [unit for unit in shown if picked[unit].verdict != "pass"][:ROWS]
        passed = len(picked) - sum(1 for one in picked.values() if one.verdict != "pass")
        head += f" {passed} pass G3 (observe curves for their rows); those that do not:"
    lines = [head, *(_row(picked[unit], picks.g3.get(unit)) for unit in shown)]
    windows = [result for result in picks.g4 if result.unit != LINE]
    off = [result.unit for result in windows if result.verdict != "pass"]
    passing = len(windows) - len(off)
    said = f"G4 on the line now: {passing} of {len(windows)} curves pass"
    if off:
        said += f"; off their neighbours: {_where(off)}"
    lines.append(said + ".")
    return "\n".join(lines)


def flag_said(flag: Flag) -> str:
    """A flag with its class, and the change it suggests when it is fixable."""
    kind = flag_class(flag)
    action = flag.action
    if kind == "fixable" and isinstance(action, Override):
        return f"{flag.name} (fixable: {flag.stage} {json.dumps(action.overrides)})"
    return f"{flag.name} ({kind})"


def _row(one: Picked, g3: GateResult | None) -> str:
    xmid = one.unit.removeprefix("xmid_")
    if one.error is not None:
        return f"- {xmid}: not picked: {one.error}"
    if one.recipe.how == "delete":
        return f"- {xmid}: its curve deleted"
    if g3 is None:
        return f"- {xmid}: no result"
    flags = ", ".join(flag_said(flag) for flag in g3.flags) or "no flag"
    kept = []
    if g3.kept.wavelength_m is not None:
        kept.append(f"λ {g3.kept.wavelength_m[0]:.3g}-{g3.kept.wavelength_m[1]:.3g} m")
    if g3.kept.n_points is not None:
        kept.append(f"{g3.kept.n_points} point{'s' if g3.kept.n_points != 1 else ''}")
    return f"- {xmid}: {g3.verdict}; {flags}" + "".join(f"; {part}" for part in kept)


def _recipe(recipe: Recipe) -> str:
    changed = f" with picking {json.dumps(recipe.picking)}" if recipe.picking else ""
    match recipe.how:
        case "guide" if recipe.guide == NEIGHBOURS:
            return "guided by the neighbours' median curve" + changed
        case "guide":
            return f"guided by {len(recipe.guide or ())} points" + changed
        case "lasso":
            return f"on a lasso of {len(recipe.polygon or ())} vertices" + changed
        case "delete":
            return "their curves deleted"
        case _:
            return "the tracker" + (changed or " with the line's picking")


def _parameters(recipe: Recipe, base: PickingParameters, folder: Path) -> PickingParameters:
    """The tracker's parameters for `recipe` on the window in `folder`: the line's, changed by
    the recipe's; a guide's points, or a lasso's maxima within its polygon's band."""
    values = {**base.model_dump(), **recipe.picking}
    if recipe.how == "guide" and isinstance(recipe.guide, tuple):
        values["guide"] = recipe.guide
    elif recipe.how == "lasso" and recipe.polygon is not None:
        values["guide"] = lasso_guide(load_image(folder), recipe.polygon)
        frequencies = [f for f, _ in recipe.polygon]
        if "fmin" not in recipe.picking:
            values["fmin"] = min(frequencies)
        if "fmax" not in recipe.picking:
            values["fmax"] = max(frequencies)
    return _validated(values)


def _validated(values: Mapping[str, Any]) -> PickingParameters:
    """The tracker's parameters, or what is wrong with them, said as a bad argument."""
    try:
        return PickingParameters.model_validate(values)
    except ValidationError as error:
        said = "; ".join(
            f"{'.'.join(str(part) for part in problem['loc'])}: {problem['msg']}"
            for problem in error.errors()
        )
        raise ValueError(f"picking: {said}.") from None


def _delete(run_folder: Path, unit: str) -> None:
    """The window's curve set aside, and its deletion logged as the model's picking attempt with
    no result: the window has no curve."""
    attempts = read_attempts(run_folder)
    previous = len(attempts_of(attempts, unit, "picking"))
    if previous:
        invalidate(run_folder / unit, "picking", previous)
    now = datetime.now(UTC)
    attempt = Attempt(
        unit=unit,
        stage="picking",
        attempt=previous + 1,
        parameters={},
        triggered_by=BACKTRACK,
        started_at=now,
        finished_at=now,
        status="succeeded",
        notes=("The curve deleted by the model: the window has none.",),
    )
    append_attempt(run_folder, afresh(run_folder, attempt))


def _band(attempts: tuple[Attempt, ...], unit: str) -> tuple[float, float] | None:
    """The band G2 found coherent on the window's latest image, which G3 judges its pick in."""
    imaged = latest(attempts, unit, "phase_shift")
    g2 = imaged.results.get("G2") if imaged is not None else None
    return g2.kept.band_hz if g2 is not None else None


def _classes(flags: Iterable[Flag]) -> dict[str, FlagClass]:
    return {flag.name: flag_class(flag) for flag in flags}


def _where(units: Sequence[str]) -> str:
    """Windows as stretches of xmids."""
    xmids = [x for unit in units if (x := xmid_of(unit)) is not None]
    return stretches(xmids)
