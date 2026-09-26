"""S4 on a judged run, the QC way (docs/qc_workflow.md, option B): the windows G4 passed, each
inverted with bounds derived from its own curve (the checks before S4) and the values given, in
worker processes; G5 on each model and its retries (sampling longer, a bound widened, a layer
more or fewer), then G6 over the line and its own (a non-unique model inverted again), within
the budgets; every attempt in the QC log, and the report written. What invert runs as a
background job, whose record (inversion.json) job_status reads."""

import logging
import traceback
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError
from sigpipe.base import DispersionCurve
from sigpipe.masw.inversion import InversionError, InversionParameters, invert_window
from sigpipe.masw.inversion.measuring import InversionMeasures, measure_inversion, report_depths
from sigpipe.masw.inversion.priors import Derived, broadcast_layers, checkable, derive_inversion
from sigpipe.masw.inversion.section import save_comparison, save_section, save_sections_file
from sigpipe.masw.picks import CURVES_FILE
from sigpipe.masw.quality.line import Series
from sigpipe.masw.runs import RunError, RunManifest, find_run, load_manifest, start_worker

from paco.inversion import (
    InversionRecord,
    WindowInversion,
    new_job_id,
    read_record,
    window_result,
    write_record,
)
from paco.qc.attempts import invalidate
from paco.qc.budgets import budget_spent
from paco.qc.config import QCConfig, read_qc_config
from paco.qc.g5_model import ModelThresholds, judge_model
from paco.qc.g6_models import LINE, judge_model_profile
from paco.qc.judging import saved_m0
from paco.qc.log import append_attempt, attempts_of, latest, read_attempts, record_result
from paco.qc.loops import RetryBudget, deep_merge, stage_changes
from paco.qc.models import Attempt, GateResult
from paco.qc.report import (
    build_report,
    changed_settings,
    read_report,
    summarize_report,
    write_report,
    xmid_of,
)
from paco.settings import Settings

MEASURES_FILE = "SeismicInversion_Measures_0000.json"  # what G5 judged, and G6 compares
ERROR_FILE = "inversion_error.log"

# Called with (windows done, windows to invert).
type ProgressCallback = Callable[[int, int], None]
# Called as each window's inversion ends, with what the job reports of it.
type OnWindow = Callable[[WindowInversion], None]

logger = logging.getLogger(__name__)


def submit_inversion(
    run_id: str, given: Mapping[str, Any] | None, settings: Settings
) -> InversionRecord:
    """Record a new inversion job of run `run_id`, queued, and return it: the windows G4 passed,
    with `given` (the inversion's parameters the user typed). Refuses a run G4 has not judged
    or rejected, values that cannot hold, and a run already being inverted."""
    run_folder = find_run(run_id, settings)
    ready = invertible(run_folder, load_manifest(run_id, settings))
    layers = read_qc_config(run_folder).priors.n_layers
    given = broadcast_layers(given or {}, layers)
    try:
        InversionParameters.model_validate(checkable(given, layers))
    except ValidationError as error:
        problems = "; ".join(str(problem["msg"]) for problem in error.errors())
        raise InversionError(f"The inversion's parameters do not hold: {problems}") from error
    previous = read_record(run_folder)
    if previous is not None and previous.state in ("queued", "running"):
        raise InversionError(
            f"Run '{run_id}' is already being inverted: job '{previous.job_id}' is "
            f"{previous.state}. Follow it with job_status."
        )
    record = InversionRecord(
        job_id=new_job_id(),
        run_id=run_id,
        given=given,
        state="queued",
        submitted_at=datetime.now(UTC),
        total=len(ready),
    )
    write_record(run_folder, record)
    return record


def run_inversion_job(
    record: InversionRecord,
    settings: Settings,
    on_progress: ProgressCallback | None = None,
    units: Sequence[str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> InversionRecord:
    """Run queued job `record` to its end: the first inversions (or, with `units`, those
    windows again with `overrides`: a backtrack), the gates' retries, each window recorded as it
    ends, then the gates' summary. The job fails when every window failed, or on any other
    failure."""
    run_folder = find_run(record.run_id, settings)
    record = record.model_copy(update={"state": "running", "started_at": datetime.now(UTC)})
    write_record(run_folder, record)
    windows: dict[str, WindowInversion] = {}
    try:
        ready = invertible(run_folder, load_manifest(record.run_id, settings))
        record = record.model_copy(update={"depths_m": line_depths(ready)})
        write_record(run_folder, record)
    except RunError:
        pass  # said again, as the job's error, below

    def on_window(window: WindowInversion) -> None:
        nonlocal record
        windows[window.folder] = window
        ordered = tuple(sorted(windows.values(), key=lambda one: one.xmid))
        record = record.model_copy(update={"windows": ordered})
        write_record(run_folder, record)

    try:
        if units is None:
            judge_inversions(record.run_id, settings, record.given, on_progress, on_window)
        else:
            rerun_inversion(
                record.run_id, units, overrides or {}, settings, "backtrack", on_progress, on_window
            )
        report = read_report(run_folder)
        summary = summarize_report(report, gates=("G5", "G6"))
        changed = changed_settings(report, ("inversion",))
        # The verdicts as they ended: a retry refused for want of budget is a reject.
        final = {unit.unit: unit.verdicts.get("G5") for unit in report.units}
        # PAC's section of the line, over the models G5 passed: best effort, never the job's
        # failure.
        passed = [unit for unit, verdict in final.items() if verdict == "pass"]
        try:
            if (section := save_section(run_folder, passed)) is not None:
                summary += f"\nSection of the {len(passed)} models G5 passed: {section.name}."
            save_sections_file(run_folder, passed)
            save_comparison(run_folder, passed)
        except Exception:
            logger.exception("Could not save the velocity section of %s", run_folder)
        record = record.model_copy(
            update={
                "windows": tuple(
                    window.model_copy(update={"verdict": final.get(window.folder, window.verdict)})
                    for window in record.windows
                )
            }
        )
        if record.windows and all(window.status == "failed" for window in record.windows):
            update: dict[str, Any] = {
                "state": "failed",
                "error": "Every window failed: see errors.",
            }
        else:
            update = {"state": "succeeded"}
        record = record.model_copy(update={**update, "summary": summary, "changed": changed})
    except Exception as exc:
        record = record.model_copy(
            update={"state": "failed", "error": f"{type(exc).__name__}: {exc}"}
        )
    record = record.model_copy(update={"finished_at": datetime.now(UTC)})
    write_record(run_folder, record)
    return record


def judge_inversions(
    run_id: str,
    settings: Settings,
    given: Mapping[str, Any] | None = None,
    on_progress: ProgressCallback | None = None,
    on_window: OnWindow | None = None,
) -> tuple[GateResult, ...]:
    """S4 on every window of run `run_id` that G4 passed and that has no inversion yet: bounds
    derived from each curve, with `given` (the inversion's parameters the user typed) kept where
    they pass the checks; then G5 and G6 with their retries. Returns G5's first results, by
    xmid."""
    run_folder = find_run(run_id, settings)
    manifest = load_manifest(run_id, settings)
    ready = invertible(run_folder, manifest)
    config = read_qc_config(run_folder)
    pending = {
        unit: curve
        for unit, curve in ready.items()
        if not (run_folder / unit / MEASURES_FILE).exists()
    }
    jobs = {unit: derive_inversion(curve, config.priors, given) for unit, curve in pending.items()}
    depths = line_depths(ready)
    results = _invert(
        run_folder, jobs, depths, config, "initial", settings.workers, on_progress, on_window
    )
    settle_models(run_folder, manifest, config, settings, ready, depths, on_window)
    write_report(
        build_report(run_id, run_folder, config.budgets, len(manifest.windows)), run_folder
    )
    return results


def rerun_inversion(
    run_id: str,
    units: Sequence[str],
    overrides: Mapping[str, Any],
    settings: Settings,
    triggered_by: str = "backtrack",
    on_progress: ProgressCallback | None = None,
    on_window: OnWindow | None = None,
) -> tuple[GateResult, ...]:
    """The inversion again for the windows `units` of run `run_id`, from each one's latest
    parameters with `overrides` on them, through the checks before S4 again; then G5 and G6
    with their retries."""
    run_folder = find_run(run_id, settings)
    manifest = load_manifest(run_id, settings)
    ready = invertible(run_folder, manifest)
    config = read_qc_config(run_folder)
    if unknown := [unit for unit in units if unit not in ready]:
        raise RunError(
            f"Run '{run_id}' has no window {', '.join(unknown)} that G4 passed. Those it passed: "
            f"{', '.join(ready)}."
        )
    if unexpected := sorted(set(overrides) - set(InversionParameters.model_fields)):
        raise RunError(
            f"Unknown inversion parameter(s) {', '.join(unexpected)}. The inversion's are "
            f"{', '.join(InversionParameters.model_fields)}."
        )
    attempts = read_attempts(run_folder)
    jobs: dict[str, Derived] = {}
    for unit in units:
        previous = latest(attempts, unit, "inversion")
        base = dict(previous.parameters) if previous is not None else {}
        jobs[unit] = derive_inversion(ready[unit], config.priors, given_again(base, overrides))
    depths = line_depths(ready)
    results = _invert(
        run_folder, jobs, depths, config, triggered_by, settings.workers, on_progress, on_window
    )
    settle_models(run_folder, manifest, config, settings, ready, depths, on_window)
    write_report(
        build_report(run_id, run_folder, config.budgets, len(manifest.windows)), run_folder
    )
    return results


def given_again(previous: Mapping[str, Any], changes: Mapping[str, Any]) -> dict[str, Any]:
    """The values an inversion done again starts from: its previous parameters with `changes`
    on them. Another number of layers derives every layer's bounds again; other iterations
    without a burn-in get a tenth of them."""
    base = dict(previous)
    if changes.get("n_layers", base.get("n_layers")) != base.get("n_layers"):
        base = {k: v for k, v in base.items() if k not in ("vs_layers", "thickness_layers")}
    if "n_iterations" in changes and "n_burnin_iterations" not in changes:
        base.pop("n_burnin_iterations", None)
    return deep_merge(base, changes)


def settle_models(
    run_folder: Path,
    manifest: RunManifest,
    config: QCConfig,
    settings: Settings,
    ready: Mapping[str, DispersionCurve],
    depths: tuple[float, ...],
    on_window: OnWindow | None = None,
) -> None:
    """G5's retries until none is asked or the budgets are spent, then G6 over the line and its
    own retries (then G5 on those models), until G6 asks for none."""
    n_units = max(1, len(manifest.windows))
    retry_failed(run_folder, config, settings, ready, depths, n_units, on_window)
    while True:
        while True:
            attempts = read_attempts(run_folder)
            results = [
                attempt.results["G5"]
                for unit in ready
                if (attempt := latest(attempts, unit, "inversion")) is not None
                and "G5" in attempt.results
            ]
            if not _retry_models(
                results, run_folder, config, settings, ready, depths, n_units, on_window
            ):
                break
        line = judge_model_line(run_folder, manifest, config)
        g6 = [result for result in line if result.unit != LINE]
        if not _retry_models(g6, run_folder, config, settings, ready, depths, n_units, on_window):
            return


def retry_failed(
    run_folder: Path,
    config: QCConfig,
    settings: Settings,
    ready: Mapping[str, DispersionCurve],
    depths: tuple[float, ...],
    n_units: int,
    on_window: OnWindow | None,
) -> None:
    """Invert once more, with the same parameters, the windows whose inversion failed: sigpipe's
    sampler is not seeded, and its failure when a chain keeps no predicted curve for some models
    (1 window of 66 on the demo line) does not come back every time."""
    attempts = read_attempts(run_folder)
    budget = RetryBudget(attempts, config.budgets, n_units)
    jobs: dict[str, Derived] = {}
    for unit in ready:
        attempt = latest(attempts, unit, "inversion")
        if attempt is None or attempt.status != "failed" or not budget.grant(unit, "S4"):
            continue
        parameters = InversionParameters.model_validate(attempt.parameters)
        derived = derive_inversion(ready[unit], config.priors)
        jobs[unit] = Derived(parameters, attempt.notes, derived.reach_m, derived.max_layers)
    if jobs:
        _invert(run_folder, jobs, depths, config, "S4:failed", settings.workers, None, on_window)


def _retry_models(
    results: Sequence[GateResult],
    run_folder: Path,
    config: QCConfig,
    settings: Settings,
    ready: Mapping[str, DispersionCurve],
    depths: tuple[float, ...],
    n_units: int,
    on_window: OnWindow | None,
) -> bool:
    """Invert again the windows whose `results` ask a change of the inversion and have budget
    left (the others that ask are rejected, "budget spent"); whether any was."""
    attempts = read_attempts(run_folder)
    budget = RetryBudget(attempts, config.budgets, n_units)
    by_trigger: dict[str, dict[str, Derived]] = {}
    for result in results:
        attempt = latest(attempts, result.unit, "inversion")
        wanted = stage_changes(result, "inversion")
        if attempt is None or result.verdict != "retry" or wanted is None:
            continue
        if not budget.grant(result.unit, result.gate):
            spent = budget_spent(result)
            record_result(run_folder, result.unit, "inversion", attempt.attempt, spent)
            continue
        changes, flag = wanted
        try:
            derived = derive_inversion(
                ready[result.unit], config.priors, given_again(attempt.parameters, changes)
            )
        except InversionError:
            spent = budget_spent(result)
            record_result(run_folder, result.unit, "inversion", attempt.attempt, spent)
            continue
        by_trigger.setdefault(f"{result.gate}:{flag}", {})[result.unit] = derived
    for trigger, jobs in by_trigger.items():
        _invert(run_folder, jobs, depths, config, trigger, settings.workers, None, on_window)
    return bool(by_trigger)


def invertible(run_folder: Path, manifest: RunManifest) -> dict[str, DispersionCurve]:
    """The windows G4 passed, with the curve each inverts: the gate decision before S4. Refuses
    a run G4 has not judged, and a line it rejected."""
    attempts = read_attempts(run_folder)
    line = latest(attempts, LINE, "picking")
    g4 = line.results.get("G4") if line is not None else None
    if g4 is None:
        raise RunError(
            f"Run '{manifest.run_id}' has not been judged up to G4: judge it before inverting."
        )
    if g4.verdict == "reject":
        raise RunError(
            f"G4 rejected the line of run '{manifest.run_id}': no curve passed, nothing to invert."
        )
    ready: dict[str, DispersionCurve] = {}
    for window in manifest.windows:
        picked = latest(attempts, window.folder, "picking")
        if picked is None or any(
            gate not in picked.results or picked.results[gate].verdict != "pass"
            for gate in ("G3", "G4")
        ):
            continue
        if (curve := saved_m0(run_folder / window.folder / CURVES_FILE)) is not None:
            ready[window.folder] = curve
    return ready


def judge_model_line(
    run_folder: Path, manifest: RunManifest, config: QCConfig
) -> tuple[GateResult, ...]:
    """G6 over the line: the smooth median of every window whose latest inversion passed G5,
    down to its depth of investigation (half its curve's longest wavelength), against its
    neighbours. A window's verdict goes to its latest inversion attempt, the line's own to an
    attempt of the unit "line"."""
    attempts = read_attempts(run_folder)
    models: list[Series] = []
    curves: dict[str, GateResult] = {}
    parameters: dict[str, InversionParameters] = {}
    useful: dict[str, float | None] = {}
    without: list[float] = []
    for window in manifest.windows:
        inverted = latest(attempts, window.folder, "inversion")
        g5 = inverted.results.get("G5") if inverted is not None else None
        path = run_folder / window.folder / MEASURES_FILE
        if inverted is None or g5 is None or g5.verdict != "pass" or not path.exists():
            without.append(window.xmid)
            continue
        measures = InversionMeasures.model_validate_json(path.read_text())
        # Down to the curve's depth of investigation, not the posterior's: with 3 layers or more
        # each layer's Vs spans most of its prior (PAC's uncertainties, 10 to 15 % of the
        # velocity), so the depth where the posterior's spread reaches half the prior's was 0 m
        # on 62 of the layer study's 72 models, and G6 had nothing to compare (2026-09-26).
        curve = saved_m0(run_folder / window.folder / CURVES_FILE)
        limit = investigation_depth(curve, config) if curve is not None else np.inf
        depths = [(depth, vs) for depth, vs in measures.vs_at_depths if depth <= limit]
        if not depths:
            without.append(window.xmid)
            continue
        models.append(
            Series(
                window.folder,
                window.xmid,
                np.array([depth for depth, _ in depths]),
                np.array([vs for _, vs in depths]),
            )
        )
        picked = latest(attempts, window.folder, "picking")
        if picked is not None and "G4" in picked.results:
            curves[window.folder] = picked.results["G4"]
        parameters[window.folder] = InversionParameters.model_validate(inverted.parameters)
        useful[window.folder] = None if np.isinf(limit) else limit
    started_at = datetime.now(UTC)
    results = judge_model_profile(models, curves, parameters, useful, config.models, without)
    for result in results:
        if result.unit != LINE:
            inverted = latest(attempts, result.unit, "inversion")
            if inverted is not None:
                record_result(run_folder, result.unit, "inversion", inverted.attempt, result)
            continue
        append_attempt(
            run_folder,
            Attempt(
                unit=LINE,
                stage="inversion",
                attempt=len(attempts_of(attempts, LINE, "inversion")) + 1,
                parameters={},
                triggered_by="initial",
                started_at=started_at,
                finished_at=datetime.now(UTC),
                status="succeeded",
                results={result.gate: result},
            ),
        )
    return results


def investigation_depth(curve: DispersionCurve, config: QCConfig) -> float:
    """The depth `curve` informs a model down to: its longest wavelength times the priors'
    `max_depth` (half of it), MASW's depth of investigation and the deepest the half-space's top
    may be (the checks before S4)."""
    wavelengths = np.asarray(curve.vs, dtype=float) / np.asarray(curve.fs, dtype=float)
    return round(config.priors.max_depth * float(wavelengths.max()), 2)


def line_depths(ready: Mapping[str, DispersionCurve]) -> tuple[float, ...]:
    """Where the line's models are reported and compared: round depths down to half the median
    longest wavelength."""
    return report_depths(
        [
            float(np.max(np.asarray(curve.vs, dtype=float) / np.asarray(curve.fs, dtype=float)))
            for curve in ready.values()
        ]
    )


def _invert(
    run_folder: Path,
    jobs: Mapping[str, Derived],
    depths: tuple[float, ...],
    config: QCConfig,
    triggered_by: str,
    workers: int,
    on_progress: ProgressCallback | None = None,
    on_window: OnWindow | None = None,
) -> tuple[GateResult, ...]:
    """Each window of `jobs` inverted with its parameters in a worker, the previous attempt's
    files archived first; G5 on each, logged as its inversion attempt."""
    attempts = read_attempts(run_folder)
    numbers: dict[str, int] = {}
    for unit in jobs:
        previous = attempts_of(attempts, unit, "inversion")
        if previous:
            invalidate(run_folder / unit, "inversion", len(previous))
        numbers[unit] = len(previous) + 1
    results: list[GateResult] = []
    started_at = datetime.now(UTC)
    with ProcessPoolExecutor(
        max_workers=workers, initializer=start_worker, initargs=(run_folder,)
    ) as executor:
        futures: dict[Future[InversionMeasures], str] = {
            executor.submit(
                _invert_and_measure, run_folder / unit, derived.parameters, depths, config.model
            ): unit
            for unit, derived in jobs.items()
        }
        if on_progress is not None:
            on_progress(0, len(futures))
        for done, future in enumerate(as_completed(futures), start=1):
            unit = futures[future]
            derived = jobs[unit]
            attempt = Attempt(
                unit=unit,
                stage="inversion",
                attempt=numbers[unit],
                parameters=derived.parameters.model_dump(mode="json"),
                triggered_by=triggered_by,
                started_at=started_at,
                finished_at=datetime.now(UTC),
                status="succeeded",
                notes=derived.notes,
            )
            xmid = xmid_of(unit) or 0.0
            try:
                measures = future.result()
                g5 = judge_model(
                    unit,
                    measures,
                    derived.parameters,
                    config.model,
                    derived.reach_m,
                    derived.max_layers,
                )
                attempt = attempt.model_copy(update={"results": {g5.gate: g5}})
                results.append(g5)
                window = window_result(xmid, unit, measures, g5.verdict, derived.reach_m)
            except Exception as exc:
                (run_folder / unit / ERROR_FILE).write_text(
                    "".join(traceback.format_exception(exc))
                )
                error = f"{type(exc).__name__}: {exc}"
                attempt = attempt.model_copy(update={"status": "failed", "error": error})
                window = WindowInversion(xmid=xmid, folder=unit, status="failed", error=error)
            append_attempt(run_folder, attempt)
            if on_window is not None:
                on_window(window)
            if on_progress is not None:
                on_progress(done, len(futures))
    return tuple(sorted(results, key=lambda result: float(result.unit.removeprefix("xmid_"))))


def _invert_and_measure(
    folder: Path,
    parameters: InversionParameters,
    depths: tuple[float, ...],
    thresholds: ModelThresholds,
) -> InversionMeasures:
    """Runs in a worker: the window's inversion, then what G5 judges, saved next to it."""
    invert_window(folder, parameters)
    measures = measure_inversion(
        folder,
        parameters,
        depths,
        thresholds.n_bands,
        thresholds.bound_edge,
        thresholds.useful_std_ratio,
    )
    (folder / MEASURES_FILE).write_text(measures.model_dump_json(indent=2))
    return measures
