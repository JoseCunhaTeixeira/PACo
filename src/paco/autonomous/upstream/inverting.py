"""From PACo, src/paco/qc/inverting.py, `_invert` and `_invert_and_measure`, at commit e5f1a24.

Why: private, and they invert without a seed, through sigpipe's `invert_window`. The autonomous agent
agent seeds each window's inversion (docs/autonomous_plan.md §10): the seed goes to the copy of
`invert_window` that passes it (paco.autonomous.upstream.inversion), and into the attempt's notes. Each
window's result comes back too, for the study's own record. Otherwise as they are: each window
in a worker, its previous files archived, staged until it finished, G5 on its model, logged as
its inversion attempt, afresh for a first inversion or one the model redoes."""

import traceback
from collections.abc import Mapping
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from sigpipe.masw.inversion import InversionParameters
from sigpipe.masw.inversion.measuring import InversionMeasures, measure_inversion
from sigpipe.masw.inversion.priors import Derived
from sigpipe.masw.runs import start_worker
from sigpipe.masw.runs.stopping import Stopped, commit, finished, staging, undo
from sigpipe.masw.runs.writing import write_atomic
from sigpipe.workers import one_thread_each

from paco import stopping
from paco.autonomous.upstream.inversion import invert_window
from paco.inversion import WindowInversion, window_result
from paco.qc.attempts import invalidate, restore
from paco.qc.config import QCConfig
from paco.qc.g5_model import ModelThresholds, judge_model
from paco.qc.inverting import (
    ERROR_FILE,
    MEASURES_FILE,
    OnWindow,
    ProgressCallback,
    chain_jobs,
    fewest_layers,
    longer_runs,
    narrowed,
    window_modes,
)
from paco.qc.log import afresh, append_attempt, attempts_of, read_attempts, starts_afresh
from paco.qc.models import Attempt
from paco.qc.report import xmid_of


def invert(
    run_folder: Path,
    jobs: Mapping[str, Derived],
    seeds: Mapping[str, int],
    depths: tuple[float, ...],
    config: QCConfig,
    triggered_by: str,
    workers: int,
    on_progress: ProgressCallback | None = None,
    on_window: OnWindow | None = None,
) -> dict[str, WindowInversion]:
    """PACo's `_invert`: each window of `jobs` inverted with its parameters and its seed (`seeds`)
    in a worker, the previous attempt's files archived first; G5 on each, logged as its
    inversion attempt, the seed in its notes. A window's inversion writes into a staging folder,
    moved into place once it finished: stopped (see paco.stopping), the windows not finished get
    their previous attempt back. Each window's result, by its folder.

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
    results: dict[str, WindowInversion] = {}
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
                seeds[unit],
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
                    notes=(*derived.notes, f"seed {seeds[unit]}"),
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
                results[unit] = window
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
    return results


def _invert_and_measure(
    folder: Path,
    parameters: InversionParameters,
    depths: tuple[float, ...],
    thresholds: ModelThresholds,
    chain_jobs: int = 1,
    output: Path | None = None,
    seed: int | None = None,
) -> InversionMeasures:
    """Runs in a worker: the window's inversion, of every mode picked in it, its chains in
    `chain_jobs` processes seeded with `seed`, then what G5 judges (M0's fit), saved next to it,
    or in `output` (a staging folder, moved into place once this ended)."""
    invert_window(
        folder,
        parameters,
        window_modes(folder),
        chain_jobs=chain_jobs,
        output_folder=output,
        seed=seed,
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
    write_atomic((output or folder) / MEASURES_FILE, measures.model_dump_json(indent=2))
    return measures
