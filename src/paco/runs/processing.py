"""Processing a profile with a preset: one sigpipe pipeline per MASW window, in worker processes."""

import json
import os
import secrets
import time
import traceback
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from importlib.metadata import distribution, version
from pathlib import Path

import matplotlib

from paco.pipelines import build_pipeline
from paco.presets import ActivePreset, PassivePreset, make_preset, resolve_preset
from paco.profiles import load_profile, summarize
from paco.runs.models import RunError, RunManifest, RunSummary, WindowOutcome
from paco.runs.summary import summarize_run
from paco.settings import Settings
from paco.windows import MASWWindow, build_windows

# Called with (windows done, windows in the run).
type ProgressCallback = Callable[[int, int], None]


def run_processing(
    profile: str,
    preset: str,
    overrides: Mapping[str, object] | None,
    settings: Settings,
    on_progress: ProgressCallback | None = None,
) -> RunSummary:
    """Process `profile` with `preset` and `overrides`, and write the run to disk.

    Unknown names and invalid overrides raise before anything is written. A window that fails
    does not stop the run: its error goes to run.json and to its folder's error.log.

    Windows run in worker processes, which Python 3.14 starts with forkserver: a script calling
    this function needs an `if __name__ == "__main__":` guard.
    """
    loaded = load_profile(profile, settings)
    resolved = resolve_preset(make_preset(preset, overrides), loaded)
    windows = build_windows(loaded, resolved.masw)
    masw = resolved.masw
    n_positions = len(range(0, len(loaded.receivers) - masw.length + 1, masw.step))
    if not windows:
        raise RunError(
            f"No window of profile '{profile}' has a valid shot: all {n_positions} positions "
            "were skipped. Widen masw.distance_min and masw.distance_max, or change masw.length."
        )

    run_id, run_folder = _new_run_folder(settings.output_dir / profile)
    started_at = datetime.now(UTC)
    outcomes = _process_windows(resolved, windows, run_folder, settings.workers, on_progress)

    manifest = RunManifest(
        run_id=run_id,
        profile=summarize(loaded),
        preset=resolved,
        versions=_versions(),
        started_at=started_at,
        finished_at=datetime.now(UTC),
        n_positions=n_positions,
        windows=outcomes,
    )
    (run_folder / "run.json").write_text(manifest.model_dump_json(indent=2))
    return summarize_run(manifest)


def _new_run_folder(profile_folder: Path) -> tuple[str, Path]:
    """A new run ID, e.g. 20260923-142501-a3f9 (UTC time and a random suffix), and its folder."""
    while True:
        run_id = f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"
        folder = profile_folder / run_id
        try:
            folder.mkdir(parents=True)
        except FileExistsError:
            continue  # same second and same suffix: draw again
        return run_id, folder


def _process_windows(
    preset: ActivePreset | PassivePreset,
    windows: list[MASWWindow],
    run_folder: Path,
    workers: int,
    on_progress: ProgressCallback | None,
) -> tuple[WindowOutcome, ...]:
    outcomes: list[WindowOutcome] = []
    with ProcessPoolExecutor(
        max_workers=workers, initializer=start_worker, initargs=(run_folder,)
    ) as executor:
        futures: dict[Future[float], tuple[float, Path]] = {}
        for window in windows:
            folder = run_folder / f"xmid_{window.xmid:.2f}"  # PAC's window folder name
            folder.mkdir()
            (folder / "window.json").write_text(window.model_dump_json(indent=2))
            futures[executor.submit(_process_window, preset, window, folder)] = (
                window.xmid,
                folder,
            )

        if on_progress is not None:
            on_progress(0, len(futures))
        for done, future in enumerate(as_completed(futures), start=1):
            xmid, folder = futures[future]
            try:
                outcome = WindowOutcome(
                    xmid=xmid, folder=folder.name, status="succeeded", duration_s=future.result()
                )
            except Exception as exc:
                (folder / "error.log").write_text("".join(traceback.format_exception(exc)))
                outcome = WindowOutcome(
                    xmid=xmid,
                    folder=folder.name,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            outcomes.append(outcome)
            if on_progress is not None:
                on_progress(done, len(futures))

    return tuple(sorted(outcomes, key=lambda outcome: outcome.xmid))


def start_worker(run_folder: Path) -> None:
    """Set up a worker process of a run: in the run folder, with no GUI backend."""
    # sigpipe's Pipeline.run creates a logs/ folder in the working directory and resets a global
    # logger: in a worker whose working directory is the run folder, both stay inside the run.
    os.chdir(run_folder)
    # Figures are only written to files: never start a GUI backend in a worker.
    matplotlib.use("Agg")


def _process_window(
    preset: ActivePreset | PassivePreset, window: MASWWindow, folder: Path
) -> float:
    """Runs in a worker: the window's pipeline, returning its duration in seconds."""
    start = time.perf_counter()
    build_pipeline(preset, window, folder).run(show_log=False)
    return time.perf_counter() - start


def _versions() -> dict[str, str]:
    versions = {name: version(name) for name in ("paco", "sigpipe")}
    # sigpipe is installed from git, so its commit says more than its version number.
    direct_url = json.loads(distribution("sigpipe").read_text("direct_url.json") or "{}")
    if commit := direct_url.get("vcs_info", {}).get("commit_id"):
        versions["sigpipe"] += f" ({commit[:7]})"
    return versions
