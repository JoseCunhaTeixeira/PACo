"""Saving the M0 curve of a run's good windows: the work behind the pick tool."""

from pathlib import Path

import matplotlib.pyplot as plt

from paco.picking import pick_modes
from paco.picks.models import PickedWindow, PickSummary, RunPicks
from paco.quality import RunQuality, good_stretches, load_quality
from paco.runs import RunError, find_run, load_image, load_manifest
from paco.settings import Settings
from sigpipe.algorithms import min_resolvable_wavelength
from sigpipe.base import DispersionCurve, DispersionCurvesImage, DispersionImage
from sigpipe.dataio.dispersion.loading import load_dispersion_curves
from sigpipe.dataio.dispersion.plotting import plot_dispersion_image
from sigpipe.dataio.dispersion.saving import save_dispersion_curves
from sigpipe.transformers import Plot

CURVES_FILE = "DispersionCurves_0000.csv"
FIGURE_FILE = "DispersionImage_0000.png"
PICKS_FILE = "pick.json"


def pick(run_id: str, settings: Settings) -> PickSummary:
    """Save the M0 curve of every good window of run `run_id`, in PAC's layout.

    The curves are the picks dispersion_quality judged: the same images and picking parameters,
    read from the run's quality.json. As PAC's own picking does, each curve replaces its window's
    M0 curve and keeps the other labels, and the window's figure is redrawn with them. Writes
    pick.json with the windows picked.
    """
    run_folder = find_run(run_id, settings)
    quality = load_quality(run_id, settings)
    good = [window for window in quality.windows if window.quality.verdict == "good"]
    if not good:
        raise RunError(
            f"Run '{run_id}' has no good window to pick: see the advice of its "
            "dispersion_quality summary."
        )

    windows: list[PickedWindow] = []
    for window in good:
        folder = run_folder / window.folder
        image = load_image(folder)
        modes = pick_modes(image, quality.picking)
        m0 = modes[0].curve if modes else None
        if m0 is None:
            raise RunError(
                f"Run '{run_id}', {window.folder}: the pick no longer matches the run's quality "
                "assessment. Call dispersion_quality again."
            )
        curves, replaced = _replace_m0(folder / CURVES_FILE, m0)
        save_dispersion_curves(curves, path=folder / CURVES_FILE)
        _draw(folder / FIGURE_FILE, image, curves)
        windows.append(
            PickedWindow(
                xmid=window.xmid,
                folder=window.folder,
                n_points=int(m0.fs.size),
                band_hz=(float(m0.fs.min()), float(m0.fs.max())),
                replaced=replaced,
            )
        )

    record = RunPicks(run_id=run_id, picking=quality.picking, windows=tuple(windows))
    (run_folder / PICKS_FILE).write_text(record.model_dump_json(indent=2))
    return summarize_picks(record, quality, load_manifest(run_id, settings).profile.name)


def summarize_picks(record: RunPicks, quality: RunQuality, profile: str) -> PickSummary:
    return PickSummary(
        run_id=record.run_id,
        profile=profile,
        n_picked=len(record.windows),
        # The picked windows are the good ones.
        picked_xmids=good_stretches(quality.windows),
        n_skipped=len(quality.windows) - len(record.windows),
        n_replaced=sum(window.replaced for window in record.windows),
    )


def _replace_m0(path: Path, m0: DispersionCurve) -> tuple[DispersionCurvesImage, bool]:
    """The curves saved in `path` with `m0` in place of their M0 curve, and whether they had one."""
    saved = list(load_dispersion_curves([path])[0]) if path.exists() else []
    others = [curve for curve in saved if curve.mode != m0.mode]
    return DispersionCurvesImage(dispersion_curves=(*others, m0)), len(others) < len(saved)


def _draw(path: Path, image: DispersionImage, curves: DispersionCurvesImage) -> None:
    """The window's dispersion image with its curves, drawn as PAC does after a pick."""
    figure = plot_dispersion_image(
        image,
        picked_curves=curves,
        lbmin=min_resolvable_wavelength(image.acquisition),
        normalize=True,
        show_errorbars=True,
    )
    Plot.savefig(path=path, figure=figure)
    plt.close(figure)
