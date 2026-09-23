"""The quality of every window of a processed run: the work behind the dispersion_quality tool."""

from collections import Counter

from paco.picking import PickingParameters, pick_modes
from paco.quality.measuring import measure_quality
from paco.quality.models import (
    Flag,
    QualityParameters,
    QualitySummary,
    RunQuality,
    WindowQuality,
)
from paco.runs import RunError, find_run, load_image, load_manifest
from paco.settings import Settings

QUALITY_FILE = "quality.json"

# What to change when a flag is frequent: domain advice for the agent, to review with the user.
ADVICE: dict[Flag, str] = {
    "no_ridge": "no ridge above the noise floor; try longer windows (masw.length).",
    "sharpness": "peaks narrower than the window can resolve, so not a propagating wave; "
    "try longer windows (masw.length).",
    "prominence": "ridge barely above the rest of the image; try filtering, or for passive data "
    "whitening and normalization.",
    "on_data": "the pick often leaves the brightest value, a sign of competing ridges; narrow "
    "the dispersion band (dispersion.fmin, dispersion.fmax).",
    "constant_wavelength": "the pick follows the edge of what the window resolves, not a "
    "dispersion curve; try longer windows (masw.length).",
}
_MAX_ADVICE = 3


def dispersion_quality(
    run_id: str,
    settings: Settings,
    picking: PickingParameters | None = None,
    thresholds: QualityParameters | None = None,
) -> QualitySummary:
    """Pick M0 and measure the quality of every window of run `run_id` that has an image.

    Writes quality.json in each window folder, and one for the whole run with the parameters used.
    """
    picking = picking or PickingParameters()
    thresholds = thresholds or QualityParameters()
    run_folder = find_run(run_id, settings)
    manifest = load_manifest(run_id, settings)

    windows: list[WindowQuality] = []
    for outcome in manifest.windows:
        if outcome.status != "succeeded":
            continue
        folder = run_folder / outcome.folder
        image = load_image(folder)
        modes = pick_modes(image, picking)
        quality = measure_quality(image, modes[0] if modes else None, thresholds)
        (folder / QUALITY_FILE).write_text(quality.model_dump_json(indent=2))
        windows.append(WindowQuality(xmid=outcome.xmid, folder=outcome.folder, quality=quality))

    if not windows:
        raise RunError(
            f"Run '{run_id}' has no dispersion image to assess: every window failed during "
            "processing. See the errors of its run_processing summary."
        )

    record = RunQuality(
        run_id=run_id, picking=picking, thresholds=thresholds, windows=tuple(windows)
    )
    (run_folder / QUALITY_FILE).write_text(record.model_dump_json(indent=2))
    return summarize_quality(record, manifest.profile.name)


def load_quality(run_id: str, settings: Settings) -> RunQuality:
    """The quality of run `run_id`, as dispersion_quality last recorded it."""
    path = find_run(run_id, settings) / QUALITY_FILE
    if not path.exists():
        raise RunError(f"Run '{run_id}' has no quality assessment yet: call dispersion_quality.")
    return RunQuality.model_validate_json(path.read_text())


def summarize_quality(record: RunQuality, profile: str) -> QualitySummary:
    verdicts = Counter(window.quality.verdict for window in record.windows)
    flags: Counter[Flag] = Counter(
        flag for window in record.windows for flag in window.quality.flags
    )
    return QualitySummary(
        run_id=record.run_id,
        profile=profile,
        n_windows=len(record.windows),
        good=verdicts["good"],
        doubtful=verdicts["doubtful"],
        bad=verdicts["bad"],
        good_xmids=good_stretches(record.windows),
        flags=dict(flags.most_common()),
        advice=tuple(
            f"{flag} ({count} windows): {ADVICE[flag]}"
            for flag, count in flags.most_common(_MAX_ADVICE)
        ),
    )


def good_stretches(windows: tuple[WindowQuality, ...]) -> tuple[str, ...]:
    """Runs of consecutive good windows, e.g. "2.88-20.88 m (7)"."""
    runs: list[list[float]] = []
    previous_good = False
    for window in windows:
        good = window.quality.verdict == "good"
        if good and previous_good:
            runs[-1].append(window.xmid)
        elif good:
            runs.append([window.xmid])
        previous_good = good
    return tuple(
        f"{run[0]:.2f} m (1)" if len(run) == 1 else f"{run[0]:.2f}-{run[-1]:.2f} m ({len(run)})"
        for run in runs
    )
