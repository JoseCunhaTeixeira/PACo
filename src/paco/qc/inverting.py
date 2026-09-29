"""S4 on a judged run, the QC way (docs/qc_workflow.md, option B): the windows G4 passed, each
inverted with bounds derived from its own curve (the checks before S4) and the values given, in
worker processes; G5 on each model and its retries (sampling longer, a bound widened, a layer
more or fewer), then G6 over the line and its own (a non-unique model inverted again), within
the budgets; every attempt in the QC log, and the report written. What invert runs as a
background job, whose record (inversion.json) job_status reads."""

import logging
import math
import traceback
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import numpy as np
from pydantic import ValidationError
from sigpipe.base import DispersionCurve
from sigpipe.base.dispersion_curve import Mode
from sigpipe.masw.inversion import InversionError, InversionParameters, invert_window
from sigpipe.masw.inversion.measuring import InversionMeasures, measure_inversion, report_depths
from sigpipe.masw.inversion.priors import (
    FIXED_KEYS,
    MIN_LAYERS,
    THICKNESS_STEP_SHARE,
    Derived,
    broadcast_layers,
    checkable,
    derive_inversion,
)
from sigpipe.masw.inversion.section import save_comparison, save_section, save_sections_file
from sigpipe.masw.inversion.summary import save_line_summary
from sigpipe.masw.inversion.window import M0
from sigpipe.masw.picks import CURVES_FILE, load_curves
from sigpipe.masw.quality.line import Series
from sigpipe.masw.runs import (
    RunError,
    RunManifest,
    find_run,
    load_manifest,
    start_worker,
    window_length,
)
from sigpipe.masw.runs.stopping import Stopped, commit, finished, staging, undo
from sigpipe.workers import one_thread_each

from paco import stopping
from paco.inversion import (
    InversionRecord,
    JobProgress,
    WindowInversion,
    new_job_id,
    read_record,
    window_result,
    write_record,
)
from paco.qc.attempts import invalidate, restore
from paco.qc.budgets import budget_spent
from paco.qc.config import QCConfig, read_qc_config
from paco.qc.g5_model import ModelThresholds, judge_model, significant
from paco.qc.g6_models import LINE, judge_model_profile
from paco.qc.judging import saved_m0
from paco.qc.log import (
    afresh,
    append_attempt,
    attempts_of,
    latest,
    read_attempts,
    record_result,
    starts_afresh,
)
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

# Called with (windows done, windows in the batch, what the batch does: "inverted", or the
# retries of a gate's flag).
type ProgressCallback = Callable[[int, int, str], None]
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

    def progressed(done: int, total: int, doing: str) -> None:
        nonlocal record
        progress = JobProgress(done=done, total=total, doing=doing)
        record = record.model_copy(update={"progress": progress})
        write_record(run_folder, record)
        if on_progress is not None:
            on_progress(done, total, doing)

    try:
        if units is None:
            judge_inversions(record.run_id, settings, record.given, progressed, on_window)
        else:
            rerun_inversion(
                record.run_id, units, overrides or {}, settings, "backtrack", progressed, on_window
            )
        report = read_report(run_folder)
        summary = summarize_report(report, gates=("G5", "G6"))
        changed = changed_settings(report, ("inversion",))
        # The verdicts as they ended: a retry refused for want of budget is a reject.
        final = {unit.unit: unit.verdicts.get("G5") for unit in report.units}
        # PAC's section of the line, over the models G5 passed, as its Visualization shows it
        # (smoothed too, over the run's window length): best effort, never the job's failure.
        passed = [unit for unit, verdict in final.items() if verdict == "pass"]
        try:
            window_m = window_length(run_folder)
            if (section := save_section(run_folder, passed, window_m=window_m)) is not None:
                summary += f"\nSection of the {len(passed)} models G5 passed: {section.name}."
            save_sections_file(run_folder, passed)
            save_comparison(run_folder, passed)
            model = read_qc_config(run_folder).model
            save_line_summary(run_folder, passed, model.max_misfit, model.max_rhat)
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
        record = record.model_copy(
            update={**update, "summary": summary, "changed": changed, "progress": None}
        )
    except Stopped:
        # On request: the windows that finished kept, the others as they were.
        record = record.model_copy(update={"state": "stopped", "progress": None})
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
    jobs: dict[str, Derived] = {}
    refused: dict[str, InversionError] = {}
    for unit, curve in pending.items():
        try:
            jobs[unit] = derive_inversion(curve, config.priors, given)
        except InversionError as error:
            refused[unit] = error
    if refused and not jobs:
        # Every window refused: the parameters given, most likely, which the job fails with.
        raise next(iter(refused.values()))
    for unit, error in refused.items():
        # A curve too short in wavelength for a layered model: left out with its reason, the
        # line's other windows inverted all the same.
        _not_invertible(run_folder, unit, error, on_window)
    depths = line_depths(ready)
    results = _invert(
        run_folder, jobs, depths, config, "initial", settings.workers, on_progress, on_window
    )
    settle_models(run_folder, manifest, config, settings, ready, depths, on_window, on_progress)
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
    settle_models(run_folder, manifest, config, settings, ready, depths, on_window, on_progress)
    write_report(
        build_report(run_id, run_folder, config.budgets, len(manifest.windows)), run_folder
    )
    return results


def given_again(previous: Mapping[str, Any], changes: Mapping[str, Any]) -> dict[str, Any]:
    """The values an inversion done again starts from: its previous parameters with `changes`
    on them. Another number of layers spreads the ranges over the new count, down to the same
    depth (`relayered`); other iterations without a burn-in get a quarter of them. Layers asked
    of a window whose data chose them are the layers given, from the curve: the previous run's
    table was the defaults', unused."""
    base = dict(previous)
    if (
        base.get("layering") == "free"
        and "layering" not in changes
        and any(key in changes for key in FIXED_KEYS)
    ):
        base = {
            key: value
            for key, value in base.items()
            if key not in (*FIXED_KEYS, "free", "layering")
        }
    if "n_iterations" in changes and "n_burnin_iterations" not in changes:
        base.pop("n_burnin_iterations", None)
    merged = deep_merge(base, {key: value for key, value in changes.items() if key != "n_layers"})
    count = changes.get("n_layers")
    if isinstance(count, int) and count != merged.get("n_layers"):
        merged = relayered(merged, count)
    return merged


def relayered(values: Mapping[str, Any], n_layers: int) -> dict[str, Any]:
    """`values` for `n_layers` layers: every layer above the half-space takes the Vs range that
    holds all of them before, the half-space keeps its own, and the layers share the same depth
    evenly, each step in proportion to its range. A list of ranges that is not one per layer
    (given anew) is left as it is; one that lacks a bound is derived again."""
    before = values.get("n_layers")
    result = {**values, "n_layers": n_layers}
    if not isinstance(before, int):
        return result
    vs_layers = values.get("vs_layers")
    if isinstance(vs_layers, list | tuple) and len(cast(Sequence[Any], vs_layers)) == before:
        ranges = [dict(layer) for layer in cast(Sequence[Mapping[str, Any]], vs_layers)]
        if all({"vs_min", "vs_max", "vs_perturb_std"} <= set(layer) for layer in ranges):
            *above, half_space = ranges
            envelope = {
                "vs_min": min(layer["vs_min"] for layer in above),
                "vs_max": max(layer["vs_max"] for layer in above),
                "vs_perturb_std": max(layer["vs_perturb_std"] for layer in above),
            }
            result["vs_layers"] = [envelope] * (n_layers - 1) + [half_space]
        else:
            del result["vs_layers"]
    thickness_layers = values.get("thickness_layers")
    if (
        isinstance(thickness_layers, list | tuple)
        and len(cast(Sequence[Any], thickness_layers)) == before - 1
    ):
        ranges = [dict(layer) for layer in cast(Sequence[Mapping[str, Any]], thickness_layers)]
        keys = {"thickness_min", "thickness_max", "thickness_perturb_std"}
        if all(keys <= set(layer) for layer in ranges):
            depth = sum(float(layer["thickness_max"]) for layer in ranges)
            thinnest = max(float(layer["thickness_min"]) for layer in ranges)
            shares = [
                layer["thickness_perturb_std"] / (layer["thickness_max"] - layer["thickness_min"])
                for layer in ranges
                if layer["thickness_max"] > layer["thickness_min"]
            ]
            # Rounded down: the layers never reach deeper than the depth they share.
            thickest = math.floor(depth / (n_layers - 1) * 100) / 100
            step = max(shares, default=THICKNESS_STEP_SHARE) * (thickest - thinnest)
            layer = {
                "thickness_min": thinnest,
                "thickness_max": thickest,
                "thickness_perturb_std": max(significant(step), 0.01),
            }
            result["thickness_layers"] = [layer] * (n_layers - 1)
        else:
            del result["thickness_layers"]
    return result


# The triggers of an inversion sampled longer, its chains disagreeing.
LONGER = ("G5:not_converged", "G6:non_unique")


def longer_runs(
    attempts: Sequence[Attempt], unit: str, current: Mapping[str, Any] | None = None
) -> int:
    """How many times `unit`'s inversion was sampled longer: its attempts logged, and the one
    being judged (its parameters `current`), each with more iterations than the one before. A
    retry for chains that disagreed that narrowed the ranges instead is not one."""
    runs = _runs(attempts, unit, current)
    return sum(
        int(after.get("n_iterations", 0)) > int(before.get("n_iterations", 0))
        for before, after in pairwise(runs)
    )


def narrowed(
    attempts: Sequence[Attempt],
    unit: str,
    triggered_by: str = "",
    current: Mapping[str, Any] | None = None,
) -> bool:
    """Whether a retry for chains that disagreed narrowed `unit`'s ranges already: one
    triggered by G5's not_converged that kept the iterations of the attempt before, the one
    being judged (`triggered_by`, `current`) among them."""
    logged = attempts_of(attempts, unit, "inversion")
    triggers = [attempt.triggered_by for attempt in logged] + ([triggered_by] if current else [])
    runs = _runs(attempts, unit, current)
    return any(
        trigger == "G5:not_converged"
        and int(after.get("n_iterations", 0)) == int(before.get("n_iterations", 0))
        for trigger, (before, after) in zip(triggers[1:], pairwise(runs), strict=False)
    )


def _runs(
    attempts: Sequence[Attempt], unit: str, current: Mapping[str, Any] | None
) -> list[Mapping[str, Any]]:
    """The parameters of `unit`'s inversions, logged then `current`."""
    runs: list[Mapping[str, Any]] = [
        attempt.parameters for attempt in attempts_of(attempts, unit, "inversion")
    ]
    return runs + ([current] if current is not None else [])


def fewest_layers(attempts: Sequence[Attempt], unit: str) -> int:
    """The fewest layers `unit`'s inversion goes down to: one more than any count G5 found to
    misfit, so that the loop does not go back and forth between two counts."""
    misfit = [
        int(attempt.parameters.get("n_layers", 0))
        for attempt in attempts_of(attempts, unit, "inversion")
        if (g5 := attempt.results.get("G5")) is not None
        and any(flag.name == "underfit" for flag in g5.flags)
    ]
    return max([MIN_LAYERS, *(count + 1 for count in misfit)])


def settle_models(
    run_folder: Path,
    manifest: RunManifest,
    config: QCConfig,
    settings: Settings,
    ready: Mapping[str, DispersionCurve],
    depths: tuple[float, ...],
    on_window: OnWindow | None = None,
    on_progress: ProgressCallback | None = None,
) -> None:
    """G5's retries until none is asked or the budgets are spent, then G6 over the line and its
    own retries (then G5 on those models), until G6 asks for none."""
    n_units = max(1, len(manifest.windows))
    batch = RetryBatch(run_folder, config, settings, ready, depths, n_units, on_window, on_progress)
    retry_failed(batch)
    while True:
        while True:
            attempts = read_attempts(run_folder)
            results = [
                attempt.results["G5"]
                for unit in ready
                if (attempt := latest(attempts, unit, "inversion")) is not None
                and "G5" in attempt.results
            ]
            if not _retry_models(results, batch):
                break
        line = judge_model_line(run_folder, manifest, config)
        g6 = [result for result in line if result.unit != LINE]
        if not _retry_models(g6, batch):
            return


@dataclass(frozen=True)
class RetryBatch:
    """What every batch of retries of a job shares."""

    run_folder: Path
    config: QCConfig
    settings: Settings
    ready: Mapping[str, DispersionCurve]
    depths: tuple[float, ...]
    n_units: int
    on_window: OnWindow | None
    on_progress: ProgressCallback | None

    def invert(self, jobs: Mapping[str, Derived], trigger: str) -> None:
        _invert(
            self.run_folder,
            jobs,
            self.depths,
            self.config,
            trigger,
            self.settings.workers,
            self.on_progress,
            self.on_window,
        )


def retry_failed(batch: RetryBatch) -> None:
    """Invert once more, with the same parameters, the windows whose inversion failed: sigpipe's
    sampler is not seeded, so a failure may not come back."""
    attempts = read_attempts(batch.run_folder)
    budget = RetryBudget(attempts, batch.config.budgets, batch.n_units)
    jobs: dict[str, Derived] = {}
    for unit in batch.ready:
        attempt = latest(attempts, unit, "inversion")
        if attempt is None or attempt.status != "failed" or not budget.grant(unit, "S4"):
            continue
        try:
            derived = derive_inversion(batch.ready[unit], batch.config.priors)
        except InversionError:
            continue  # the curve gives no parameters (_not_invertible): no retry changes that
        parameters = InversionParameters.model_validate(attempt.parameters)
        jobs[unit] = Derived(parameters, attempt.notes, derived.reach_m, derived.max_layers)
    if jobs:
        batch.invert(jobs, "S4:failed")


def _retry_models(results: Sequence[GateResult], batch: RetryBatch) -> bool:
    """Invert again the windows whose `results` ask a change of the inversion and have budget
    left (the others that ask are rejected, "budget spent"); whether any was."""
    run_folder, config, ready = batch.run_folder, batch.config, batch.ready
    attempts = read_attempts(run_folder)
    budget = RetryBudget(attempts, config.budgets, batch.n_units)
    by_trigger: dict[str, dict[str, Derived]] = {}
    for result in results:
        attempt = latest(attempts, result.unit, "inversion")
        wanted = stage_changes(result, "inversion")
        if attempt is None or result.verdict != "retry" or wanted is None:
            continue
        changes, flag = wanted
        given = given_again(attempt.parameters, changes)
        # Its parameters already the attempt's: it would give its result again.
        unchanged = given == dict(attempt.parameters)
        if unchanged or not budget.grant(result.unit, result.gate):
            spent = budget_spent(result, "unchanged" if unchanged else "budget")
            record_result(run_folder, result.unit, "inversion", attempt.attempt, spent)
            continue
        try:
            derived = derive_inversion(ready[result.unit], config.priors, given)
        except InversionError:
            spent = budget_spent(result)
            record_result(run_folder, result.unit, "inversion", attempt.attempt, spent)
            continue
        by_trigger.setdefault(f"{result.gate}:{flag}", {})[result.unit] = derived
    for trigger, jobs in by_trigger.items():
        batch.invert(jobs, trigger)
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
    """G6 over the line: the ensemble of every window whose latest inversion passed G5,
    down to its depth of investigation (half its curve's longest wavelength), against its
    neighbours. A window's verdict goes to its latest inversion attempt, the line's own to an
    attempt of the unit "line"."""
    attempts = read_attempts(run_folder)
    models: list[Series] = []
    curves: dict[str, GateResult] = {}
    informed: dict[str, float | None] = {}
    without: list[float] = []
    for window in manifest.windows:
        inverted = latest(attempts, window.folder, "inversion")
        g5 = inverted.results.get("G5") if inverted is not None else None
        path = run_folder / window.folder / MEASURES_FILE
        if inverted is None or g5 is None or g5.verdict != "pass" or not path.exists():
            without.append(window.xmid)
            continue
        measures = InversionMeasures.model_validate_json(path.read_text())
        # Down to the depth the data inform (G5's, where the models' Vs spread U stays under its
        # limit: one depth informed), within the curve's depth of investigation: below it, the
        # priors speak, and models differ by them alone.
        curve = saved_m0(run_folder / window.folder / CURVES_FILE)
        reach = investigation_depth(curve, config) if curve is not None else np.inf
        own = measures.useful_depth_m
        limit = min(reach, own) if own is not None else reach
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
        informed[window.folder] = None if np.isinf(limit) else limit
    started_at = datetime.now(UTC)
    results = judge_model_profile(models, curves, informed, config.models, without)
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
    files archived first; G5 on each, logged as its inversion attempt. A window's inversion
    writes into a staging folder, moved into place once it finished: stopped (see
    paco.stopping), the windows not finished get their previous attempt back.

    A retry a gate asked for goes on from the window's attempts; any other inversion (the first,
    or one the agent was asked to do again) starts them afresh: once its result is in, the
    window's earlier attempts are forgotten, their lines and archived results."""
    attempts = read_attempts(run_folder)
    fresh = starts_afresh(triggered_by)
    # What the gates weigh this inversion against: its window's attempts, none when afresh.
    history = tuple(
        a for a in attempts if not (fresh and a.stage == "inversion" and a.unit in jobs)
    )
    numbers: dict[str, int] = {}
    archived: dict[str, int] = {}  # the attempt each window's files were archived as, if any
    for unit in jobs:
        previous = attempts_of(attempts, unit, "inversion")
        if previous:
            invalidate(run_folder / unit, "inversion", len(previous))
        archived[unit] = len(previous)
        numbers[unit] = 1 if fresh else len(previous) + 1
    results: list[GateResult] = []
    started_at = datetime.now(UTC)
    one_thread_each()  # the workers are the cores the inversions take
    with ProcessPoolExecutor(
        max_workers=max(1, min(workers, len(jobs))),
        initializer=start_worker,
        initargs=(run_folder,),
    ) as executor:
        futures: dict[Future[InversionMeasures], str] = {
            executor.submit(
                _invert_and_measure,
                run_folder / unit,
                derived.parameters,
                depths,
                config.model,
                chain_jobs(workers, len(jobs), derived.parameters.n_chains),
                staging(run_folder / unit),
            ): unit
            for unit, derived in jobs.items()
        }
        waiting = dict(futures)  # the windows not finished yet
        doing = {"initial": "inverted", "backtrack": "inverted again"}.get(
            triggered_by, f"{triggered_by} retries"
        )
        if on_progress is not None:
            on_progress(0, len(futures), doing)
        try:
            for done, future in enumerate(finished(executor, futures, stopping.current()), start=1):
                unit = waiting.pop(future)
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
                    commit(run_folder / unit)
                    ran = derived.parameters.model_dump(mode="json")
                    g5 = judge_model(
                        unit,
                        measures,
                        derived.parameters,
                        config.model,
                        derived.reach_m,
                        derived.max_layers,
                        fewest_layers(history, unit),
                        longer_runs(history, unit, ran),
                        narrowed(history, unit, triggered_by, ran),
                    )
                    attempt = attempt.model_copy(update={"results": {g5.gate: g5}})
                    results.append(g5)
                    window = window_result(xmid, unit, measures, g5.verdict, derived.reach_m)
                except Exception as exc:
                    undo(run_folder / unit, created=False)  # a failed inversion's files not kept
                    (run_folder / unit / ERROR_FILE).write_text(
                        "".join(traceback.format_exception(exc))
                    )
                    error = f"{type(exc).__name__}: {exc}"
                    attempt = attempt.model_copy(update={"status": "failed", "error": error})
                    window = WindowInversion(xmid=xmid, folder=unit, status="failed", error=error)
                append_attempt(run_folder, afresh(run_folder, attempt) if fresh else attempt)
                if on_window is not None:
                    on_window(window)
                if on_progress is not None:
                    on_progress(done, len(futures), doing)
        except Stopped:
            for unit in waiting.values():
                undo(run_folder / unit, created=False)
                if archived[unit]:
                    restore(run_folder / unit, "inversion", archived[unit])
            raise
    return tuple(sorted(results, key=lambda result: float(result.unit.removeprefix("xmid_"))))


def _not_invertible(
    run_folder: Path, unit: str, error: InversionError, on_window: OnWindow | None
) -> None:
    """Window `unit`, whose curve gives no inversion parameters, logged as a failed attempt and
    reported with why."""
    said = f"{type(error).__name__}: {error}"
    now = datetime.now(UTC)
    # A first inversion: whatever the window had of an earlier one, forgotten.
    append_attempt(
        run_folder,
        afresh(
            run_folder,
            Attempt(
                unit=unit,
                stage="inversion",
                attempt=1,
                parameters={},
                triggered_by="initial",
                started_at=now,
                finished_at=now,
                status="failed",
                error=said,
            ),
        ),
    )
    if on_window is not None:
        on_window(
            WindowInversion(xmid=xmid_of(unit) or 0.0, folder=unit, status="failed", error=said)
        )


def chain_jobs(workers: int, windows: int, chains: int) -> int:
    """The processes each window's chains run in: the cores the windows leave idle, shared
    between them, never more than its chains. One window of 5 chains with 6 workers: 5; with
    4 windows or more: 1, each window in its worker."""
    return max(1, min(chains, workers // max(1, windows)))


def _invert_and_measure(
    folder: Path,
    parameters: InversionParameters,
    depths: tuple[float, ...],
    thresholds: ModelThresholds,
    chain_jobs: int = 1,
    output: Path | None = None,
) -> InversionMeasures:
    """Runs in a worker: the window's inversion, of every mode picked in it, its chains in
    `chain_jobs` processes, then what G5 judges (M0's fit), saved next to it, or in `output` (a
    staging folder, moved into place once this ended)."""
    invert_window(
        folder, parameters, window_modes(folder), chain_jobs=chain_jobs, output_folder=output
    )
    measures = measure_inversion(
        folder,
        parameters,
        depths,
        thresholds.n_bands,
        thresholds.bound_edge,
        thresholds.useful_uncertainty,
        output_folder=output,
    )
    ((output or folder) / MEASURES_FILE).write_text(measures.model_dump_json(indent=2))
    return measures


def window_modes(folder: Path) -> tuple[Mode, ...]:
    """The modes a window's inversion inverts: every one picked in it, as PAC inverts them by
    default, M0 always. PACo picks M0 alone: a higher mode is too hazardous to pick without a
    person's eye (a ridge taken for the wrong mode misleads the inversion); a person picks it in
    PAC, and the inversion uses it."""
    return tuple(sorted({curve.mode for curve in load_curves(folder) or ()} | {M0}))
