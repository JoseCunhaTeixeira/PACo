"""The petrophysical inversion of a judged run, the QC way (the user's decisions of 2026-09-26:
the agent runs it when the user asks for it; range, fit, line). The curves G4 passed that the
chosen Silex model's trained range covers (the others left out, with how they fall outside),
each inverted in worker processes; G7 on each, G8 along the line over those G7 passed, and PAC's
sections over the windows both passed. Every window's attempt goes to the QC log.

The model runs only with sigpipe's silex and santiludo extras (PACo's petro extra, or PAC's own
install); choosing one needs neither."""

import importlib.util
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict
from sigpipe.algorithms.inversion.rayleigh.petro.silex_catalog import (
    RangeGap,
    SilexCard,
    bundled_silex_model_dir,
    list_bundled_silex_models,
    load_silex_card,
    range_gaps,
)
from sigpipe.base import DispersionCurve
from sigpipe.masw.picks import CURVES_FILE
from sigpipe.masw.quality.line import Series
from sigpipe.masw.runs import RunError, RunManifest, find_run, load_manifest

from paco.qc.attempts import invalidate
from paco.qc.config import QCConfig, read_qc_config
from paco.qc.g7_petro import judge_petro
from paco.qc.g8_petro_line import LINE, judge_petro_line
from paco.qc.inverting import invertible, investigation_depth, line_depths
from paco.qc.judging import saved_m0
from paco.qc.log import append_attempt, attempts_of, latest, read_attempts, record_result
from paco.qc.models import Attempt, GateResult
from paco.qc.report import QCReport, build_report, write_report
from paco.settings import Settings

if TYPE_CHECKING:
    from sigpipe.masw.petro.measuring import PetroMeasures

MEASURES_FILE = "PetroInversion_Measures_0000.json"  # what G7 judged, and G8 compares
# In words, for the agent: how a curve falls outside a model's range.
GAPS: dict[RangeGap, str] = {
    "starts_late": "start above {start:g} Hz",
    "ends_early": "end below {end:g} Hz",
    "too_slow": "are slower than {slowest:.0f} m/s",
    "too_fast": "are faster than {fastest:.0f} m/s",
}

# Called with (windows done, windows to invert).
type ProgressCallback = Callable[[int, int], None]


class PetroModelCard(BaseModel):
    """A Silex model, what it was trained on, and how many of a run's curves it covers: what the
    agent chooses a model by. In words: Qwen3-8B took the soils and water tables a model was
    trained on for its results, and "covers": 1, "curves": 6 for all six (2026-09-26)."""

    model_config = ConfigDict(frozen=True)

    name: str
    trained_on: str  # the band, velocities, soils, layers and water tables of its training
    covers: str  # "1 of the 6 curves G4 passed; 5 end below 43 Hz"
    n_covered: int


class PetroChoice(BaseModel):
    """What petro_models returns: the models, and what to do next."""

    model_config = ConfigDict(frozen=True)

    models: tuple[PetroModelCard, ...]
    next: str


def petro_models(run_id: str, settings: Settings) -> PetroChoice:
    """Every bundled Silex model, against run `run_id`'s curves that G4 passed."""
    run_folder = find_run(run_id, settings)
    ready = invertible(run_folder, load_manifest(run_id, settings))
    cards: list[PetroModelCard] = []
    for name in list_bundled_silex_models():
        card = load_silex_card(bundled_silex_model_dir(name))
        gaps = {unit: range_gaps(card, curve) for unit, curve in ready.items()}
        n_covered = sum(not found for found in gaps.values())
        left = _left_out(card, gaps)
        cards.append(
            PetroModelCard(
                name=name,
                trained_on=f"{card.min_freq:g}-{card.max_freq:g} Hz, "
                f"{card.min_vel:.0f}-{card.max_vel:.0f} m/s; soils {', '.join(card.soils)}; up "
                f"to {card.max_layers} layers down to {card.max_depth:g} m; water tables "
                f"{card.water_table[0]:g}-{card.water_table[1]:g} m",
                covers=f"{n_covered} of the {len(ready)} curves G4 passed"
                + (f"; {left}" if left else ""),
                n_covered=n_covered,
            )
        )
    best = max(cards, key=lambda one: one.n_covered, default=None)
    if best is None or best.n_covered == 0:
        step = (
            "No model covers the run's curves: tell the user why (covers), and that a model "
            "trained on their band would be needed."
        )
    else:
        step = (
            f"Call invert_petro with model {best.name}, which covers the most: it gives the "
            "soils, N values and water table. trained_on is what the models learned, not a "
            "result."
        )
    return PetroChoice(models=tuple(cards), next=step)


def invert_petro_line(
    run_id: str,
    model_name: str,
    settings: Settings,
    on_progress: ProgressCallback | None = None,
) -> tuple[QCReport, str]:
    """Invert run `run_id`'s curves G4 passed that Silex model `model_name` covers, judge them
    (G7, G8), and write PAC's sections over the windows both passed. Returns the report and what
    the run's petrophysical models say, in words. A new inversion replaces the run's last one:
    its window files are archived, its sections removed."""
    if not all(importlib.util.find_spec(name) for name in ("santiludo", "keras", "keras_nlp")):
        raise RunError(
            "The petrophysical inversion needs sigpipe's silex and santiludo extras, not "
            "installed with this PACo: install PACo with its petro extra (uv sync --extra petro), "
            "or use PAC's assistant, which has them."
        )
    from sigpipe.masw.petro import invert_line_petro, save_line_sections
    from sigpipe.masw.petro.measuring import measure_petro

    run_folder = find_run(run_id, settings)
    manifest = load_manifest(run_id, settings)
    config = read_qc_config(run_folder)
    ready = invertible(run_folder, manifest)
    card = load_silex_card(bundled_silex_model_dir(model_name))
    gaps = {unit: range_gaps(card, curve) for unit, curve in ready.items()}
    covered = [unit for unit, found in gaps.items() if not found]
    if not covered:
        raise RunError(
            f"Silex model {model_name} covers none of the {len(ready)} curves G4 passed: "
            f"{_left_out(card, gaps)}. Choose another model with petro_models, or tell the user."
        )
    _archive(run_folder, manifest)

    started_at = datetime.now(UTC)
    outcomes = invert_line_petro(
        run_folder,
        covered,
        model_name,
        settings.workers,
        None if on_progress is None else lambda done, total, _: on_progress(done, total),
    )
    depths = line_depths(ready)
    others = _covering(ready, model_name)  # for G7's flags
    attempts = read_attempts(run_folder)
    measured: dict[str, PetroMeasures] = {}
    for outcome in outcomes:
        attempt = Attempt(
            unit=outcome.unit,
            stage="petro_inversion",
            attempt=len(attempts_of(attempts, outcome.unit, "petro_inversion")) + 1,
            parameters={"model": model_name},
            triggered_by="initial",
            started_at=started_at,
            finished_at=datetime.now(UTC),
            status="succeeded" if outcome.model is not None else "failed",
            error=None if outcome.model is not None else f"{outcome.error_type}: {outcome.message}",
        )
        if outcome.model is not None:
            folder = run_folder / outcome.unit
            measures = measure_petro(folder, model_name, depths, config.petro.n_bands)
            (folder / MEASURES_FILE).write_text(measures.model_dump_json(indent=2))
            measured[outcome.unit] = measures
            result = judge_petro(outcome.unit, measures, config.petro, others[outcome.unit])
            attempt = attempt.model_copy(update={"results": {result.gate: result}})
        append_attempt(run_folder, attempt)

    passed = _judge_line(run_folder, manifest, config, measured)
    saved = save_line_sections(run_folder, passed)
    report = build_report(run_id, run_folder, config.budgets, len(manifest.windows))
    write_report(report, run_folder)
    return report, _described(card, gaps, [measured[unit] for unit in passed], saved)


def _judge_line(
    run_folder: Path,
    manifest: RunManifest,
    config: QCConfig,
    measured: Mapping[str, PetroMeasures],
) -> list[str]:
    """G8 over the windows G7 passed; returns those G8 passed too, in line order."""
    attempts = read_attempts(run_folder)
    profiles: list[Series] = []
    water_tables: dict[str, float] = {}
    curves: dict[str, GateResult] = {}
    without: list[float] = []
    for window in manifest.windows:
        inverted = latest(attempts, window.folder, "petro_inversion")
        g7 = inverted.results.get("G7") if inverted is not None else None
        measures = measured.get(window.folder)
        if g7 is None or g7.verdict != "pass" or measures is None:
            without.append(window.xmid)
            continue
        curve = saved_m0(run_folder / window.folder / CURVES_FILE)
        limit = investigation_depth(curve, config) if curve is not None else np.inf
        depths = [(depth, vs) for depth, vs in measures.vs_at_depths if depth <= limit]
        if not depths:
            without.append(window.xmid)
            continue
        profiles.append(
            Series(
                window.folder,
                window.xmid,
                np.array([depth for depth, _ in depths]),
                np.array([vs for _, vs in depths]),
            )
        )
        water_tables[window.folder] = measures.water_table_m
        picked = latest(attempts, window.folder, "picking")
        if picked is not None and "G4" in picked.results:
            curves[window.folder] = picked.results["G4"]
    started_at = datetime.now(UTC)
    results = judge_petro_line(profiles, water_tables, curves, config.petro_line, without)
    passed: list[str] = []
    for result in results:
        if result.unit != LINE:
            inverted = latest(attempts, result.unit, "petro_inversion")
            if inverted is not None:
                record_result(run_folder, result.unit, "petro_inversion", inverted.attempt, result)
            if result.verdict == "pass":
                passed.append(result.unit)
            continue
        append_attempt(
            run_folder,
            Attempt(
                unit=LINE,
                stage="petro_inversion",
                attempt=len(attempts_of(attempts, LINE, "petro_inversion")) + 1,
                parameters={},
                triggered_by="initial",
                started_at=started_at,
                finished_at=datetime.now(UTC),
                status="succeeded",
                results={result.gate: result},
            ),
        )
    return passed


def _archive(run_folder: Path, manifest: RunManifest) -> None:
    """The run's last petrophysical inversion out of the way: each window's files into its
    attempts/ folder, the line's sections removed (they are written again)."""
    attempts = read_attempts(run_folder)
    for window in manifest.windows:
        if done := attempts_of(attempts, window.folder, "petro_inversion"):
            invalidate(run_folder / window.folder, "petro_inversion", len(done))
    for path in run_folder.glob("PetroInversion_*"):
        path.unlink()


def _covering(ready: Mapping[str, DispersionCurve], chosen: str) -> dict[str, list[str]]:
    """The bundled models covering each window's curve, the chosen one left out."""
    covering: dict[str, list[str]] = {unit: [] for unit in ready}
    for name in list_bundled_silex_models():
        if name == chosen:
            continue
        card = load_silex_card(bundled_silex_model_dir(name))
        for unit, curve in ready.items():
            if not range_gaps(card, curve):
                covering[unit].append(name)
    return covering


def _left_out(card: SilexCard, gaps: Mapping[str, Sequence[RangeGap]]) -> str | None:
    """How the curves `card`'s model leaves out fall outside its range, in words."""
    counts: Counter[RangeGap] = Counter(gap for found in gaps.values() for gap in found)
    if not counts:
        return None
    start, end = card.band_needed
    slowest, fastest = card.velocities_allowed
    return "; ".join(
        f"{count} {GAPS[gap].format(start=start, end=end, slowest=slowest, fastest=fastest)}"
        for gap, count in counts.most_common()
    )


def _described(
    card: SilexCard,
    gaps: Mapping[str, Sequence[RangeGap]],
    passed: Sequence[PetroMeasures],
    saved: Sequence[Path],
) -> str:
    """What the agent reports: coverage, then the soils and water table of the models G7 and G8
    passed, and the sections written."""
    covered = sum(not found for found in gaps.values())
    lines = [f"Silex model {card.name} covers {covered} of the {len(gaps)} curves G4 passed."]
    if left := _left_out(card, gaps):
        lines.append(f"Left out, outside its range: {left}.")
    if passed:
        thickness: defaultdict[str, float] = defaultdict(float)
        for measures in passed:
            for soil, layer in zip(measures.soils, measures.thicknesses_m, strict=True):
                thickness[soil] += layer
        total = sum(thickness.values())
        soils = ", ".join(
            f"{soil} {share / total:.0%}"
            for soil, share in sorted(thickness.items(), key=lambda item: -item[1])
        )
        tables = [measures.water_table_m for measures in passed]
        depth = (
            f"{min(tables):g} m"
            if min(tables) == max(tables)
            else f"{min(tables):g} to {max(tables):g} m"
        )
        models = "The model" if len(passed) == 1 else f"The {len(passed)} models"
        lines.append(
            f"{models} G7 and G8 passed: soils by thickness {soils}; water table {depth} deep."
        )
    if saved:
        lines.append("Written: " + ", ".join(path.name for path in saved) + ".")
    else:
        lines.append("No section: fewer than two models passed.")
    return "\n".join(lines)
