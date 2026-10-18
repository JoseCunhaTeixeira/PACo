"""look (docs/autonomous_plan.md §6): figures of the committed line, for a model with vision. A record's
gather or spectra, as its preprocessing and G1 drew them; a window's dispersion image with its pick
and the wavelengths that bound what it resolves (twice the receiver spacing, under which a ridge may
be an alias; three window lengths, its reach); a sheet of up to 12 windows' images; a window's
model, as PAC's inversion card draws it: its Vs profile with the curves' density, its fit on the
image, its marginals, its chains. A passive or passive-active window's stacked correlations (its
virtual shot gather), and a passive window's FK selection of its segments. Without vision, a
window's ridge table stands in its image's stead, G1's measures in a record's, G2's in its
correlations', the selection's counts in its figure's, and the model's measures (G5's) in its
figures'."""

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sigpipe.algorithms.picking.dispersion.curve import (
    longest_reached_wavelength,
    min_resolvable_wavelength,
)
from sigpipe.base.dispersion_image import DispersionImage
from sigpipe.dataio.selection_plotting import load_selection
from sigpipe.masw.inversion.measuring import BandFit, InversionMeasures
from sigpipe.masw.inversion.window import (
    CHAINS_FIGURE,
    IMAGE_FIGURE,
    MARGINALS_FIGURE,
    WINDOW_FIGURE,
)
from sigpipe.masw.picks import CURVES_FILE
from sigpipe.masw.runs import RunManifest, load_image

from paco.autonomous.sensors.ridges import format_ridge_table, ridge_table
from paco.autonomous.tools.observe import describe_result
from paco.qc.coherence import trial_indices
from paco.qc.inverting import MEASURES_FILE
from paco.qc.judging import saved_m0
from paco.qc.log import latest, read_attempts

type What = Literal[
    "gather",
    "spectra",
    "image",
    "correlations",
    "selection",
    "sheet",
    "profile",
    "fit",
    "marginals",
    "chains",
]
# The most windows a sheet shows, and the figures' resolution.
SHEET = 12
DPI = 90
# A record's figures, as its preprocessing (the gather) and G1 (its spectra) drew them.
RECORD_FIGURES = {"gather": "Stream_0000.png", "spectra": "Spectrum_0000.png"}
# A passive or passive-active window's figures, as its processing drew them: its stacked
# correlations, and a passive one's FK selection of its segments.
WINDOW_FIGURES = {"correlations": "Stream_0000.png", "selection": "Selection_0000.png"}
# A window's model, as its inversion drew it (sigpipe's draw_figures).
MODEL_FIGURES = {
    "profile": WINDOW_FIGURE,
    "fit": IMAGE_FIGURE,
    "marginals": MARGINALS_FIGURE,
    "chains": CHAINS_FIGURE,
}


@dataclass(frozen=True)
class Seen:
    """What look gives: its text, and its figures as PNG files."""

    text: str
    figures: tuple[bytes, ...] = ()


def look(run_folder: Path, what: What, where: str, vision: bool) -> Seen:
    """`what` of the run in `run_folder`: a record's (`where`, its file's name) gather or
    spectra; a window's image (`where`, its xmid, m); a sheet of the windows `where` names
    (`a-b`, m). Without `vision`, text in the figures' stead."""
    manifest = RunManifest.model_validate_json((run_folder / "run.json").read_text())
    if what in RECORD_FIGURES:
        return _record(run_folder, manifest, what, where, vision)
    folders = [window.folder for window in manifest.windows if window.status == "succeeded"]
    if not folders:
        raise ValueError("The run has no window with an image.")
    if what in MODEL_FIGURES:
        return _model(run_folder, _nearest(folders, _position(where)), what, vision)
    if what in WINDOW_FIGURES:
        return _window(run_folder, manifest, _nearest(folders, _position(where)), what, vision)
    if what == "image":
        folder = _nearest(folders, _position(where))
        image = load_image(run_folder / folder)
        if not vision:
            # Over the band G2 found coherent, when it found one: below and above, noise.
            attempt = latest(read_attempts(run_folder), folder, "phase_shift")
            g2 = attempt.results.get("G2") if attempt is not None else None
            band = g2.kept.band_hz if g2 is not None else None
            table = format_ridge_table(ridge_table(image, band=band))
            where_said = f" over G2's coherent band {band[0]:g}-{band[1]:g} Hz" if band else ""
            return Seen(
                f"{folder}, its image as a ridge table{where_said} (vision is off):\n{table}"
            )
        figure = Figure(figsize=(5, 4), dpi=DPI)
        _draw(figure.add_subplot(), image, run_folder / folder, folder)
        return Seen(
            f"{folder}: its dispersion image, its pick, and λ = 2·dx and 3·L.", (_png(figure),)
        )
    if not vision:
        raise ValueError(
            "A sheet is figures only: with vision off, look at one window's image (its ridge "
            "table), or observe the curves."
        )
    low, high = _span(where)
    within = [folder for folder in folders if low <= float(folder.removeprefix("xmid_")) <= high]
    if not within:
        raise ValueError(f"No window between {low:g} and {high:g} m.")
    shown = [within[index] for index in trial_indices(len(within), SHEET)]
    rows = (len(shown) + 3) // 4
    figure = Figure(figsize=(12, 2.6 * rows), dpi=DPI, layout="constrained")
    for index, folder in enumerate(shown):
        _draw(
            figure.add_subplot(rows, 4, index + 1),
            load_image(run_folder / folder),
            run_folder / folder,
            folder,
        )
    return Seen(
        f"{len(shown)} of the {len(within)} windows from {low:g} to {high:g} m.", (_png(figure),)
    )


def _window(run_folder: Path, manifest: RunManifest, folder: str, what: What, vision: bool) -> Seen:
    """A passive or passive-active window's stacked correlations, or a passive one's FK
    selection; G2's measures, or the selection's counts, without vision."""
    mode = str(manifest.preset.mode)
    path = run_folder / folder / WINDOW_FIGURES[what]
    if not path.exists():
        whose = "a passive line's" if what == "selection" else "a passive or passive-active line's"
        raise ValueError(f"No {what} for {folder}: they are {whose}, this line is {mode}.")
    if vision:
        said = (
            "its stacked correlations, the virtual shot gather its image is made of"
            if what == "correlations"
            else "its FK selection: each segment's ratio against the threshold"
        )
        return Seen(f"{folder}: {said}.", (path.read_bytes(),))
    if what == "correlations":
        attempt = latest(read_attempts(run_folder), folder, "phase_shift")
        g2 = attempt.results.get("G2") if attempt is not None else None
        said = describe_result(folder, g2) if g2 is not None else "G2 has not judged it"
        return Seen(f"{folder}, G2's measures in its correlations' stead (vision is off): {said}")
    selection = load_selection(run_folder / folder)
    if selection is None:
        raise ValueError(f"{folder}'s FK selection was not saved.")
    return Seen(
        f"{folder}, its FK selection (vision is off): {selection.kept_count} of "
        f"{selection.segments} segments kept ({selection.kept_share:.0%}) at a threshold of "
        f"{selection.threshold:g}, {selection.flipped_count} flipped."
    )


def _record(run_folder: Path, manifest: RunManifest, what: What, where: str, vision: bool) -> Seen:
    """A record's gather or spectra; G1's measures without vision."""
    record = next((one for one in manifest.records if one.name == where), None)
    if record is None:
        names = ", ".join(one.name for one in manifest.records)
        raise ValueError(f"No record '{where}' on this run: {names}.")
    if not vision:
        attempt = latest(read_attempts(run_folder), record.name, "preprocessing")
        result = attempt.results.get("G1") if attempt is not None else None
        said = describe_result(record.name, result) if result is not None else "not judged yet"
        return Seen(f"{record.name}, G1's measures in its {what}'s stead (vision is off): {said}")
    path = run_folder / record.folder / RECORD_FIGURES[what]
    if not path.exists():
        raise ValueError(f"{record.name} has no {what} drawn.")
    return Seen(f"{record.name}: its {what}.", (path.read_bytes(),))


def _model(run_folder: Path, folder: str, what: What, vision: bool) -> Seen:
    """A window's model: its figure; G5's measures without vision."""
    measures = run_folder / folder / MEASURES_FILE
    if not measures.exists():
        raise ValueError(f"{folder} has no model: invert it first.")
    if not vision:
        found = InversionMeasures.model_validate_json(measures.read_text())
        return Seen(
            f"{folder}, its model's measures in its {what}'s stead (vision is off): "
            + describe_measures(found)
        )
    path = run_folder / folder / MODEL_FIGURES[what]
    if not path.exists():
        raise ValueError(f"{folder}'s inversion drew no {what} figure.")
    return Seen(f"{folder}: its model's {what}.", (path.read_bytes(),))


def describe_measures(measures: InversionMeasures) -> str:
    """A model's measures in words: its fits by band, the chains' agreement, its samples at the
    priors' bounds, the depth the data inform, the layered median."""
    fits = "; ".join(
        f"{fit.model} misfit {_misfit(fit.misfit)} ({', '.join(_band(band) for band in fit.bands)})"
        for fit in measures.fits
    )
    rhat = [value for value in measures.rhat.values() if value is not None]
    piled = ", ".join(
        f"{share.parameter} {share.share:.0%} at its {share.bound}"
        for share in measures.at_bounds[:3]
    )
    layers = ", ".join(f"{vs:.0f}" for vs in measures.vs_layers)
    interfaces = ", ".join(f"{depth:.3g}" for depth in measures.interfaces_m)
    informed = (
        f"{measures.useful_depth_m:.3g} m" if measures.useful_depth_m is not None else "the bottom"
    )
    return (f"fits: {fits}; R-hat at most {max(rhat):.3g}" if rhat else f"fits: {fits}") + (
        f"; acceptance {', '.join(f'{value:.0f}' for value in measures.acceptance)} %; "
        f"{measures.samples_per_chain} samples a chain; at the bounds: {piled or 'none'}; "
        f"informed down to {informed} (models to {measures.depth_max_m:.3g} m); layered median "
        f"Vs {layers} m/s, interfaces at {interfaces or 'none'} m."
    )


def _band(band: BandFit) -> str:
    """A band's fit: its wavelengths and its misfit."""
    low, high = band.wavelength_m
    return f"λ {low:.3g}-{high:.3g} m {_misfit(band.misfit)}"


def _misfit(value: float | None) -> str:
    return f"{value:.2g}" if value is not None else "no mode"


def _draw(axes: Axes, image: DispersionImage, folder: Path, title: str) -> None:
    """A window's dispersion image on `axes`, with its saved pick and the lines λ = 2·dx (under
    it, the aliasing zone) and λ = 3·L (beyond it, the window's reach)."""
    fs, vs = image.fs, image.vs
    axes.pcolormesh(fs, vs, image.fv_map.T, shading="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    curve = saved_m0(folder / CURVES_FILE)
    if curve is not None and len(curve.fs):
        axes.plot(curve.fs, curve.vs, ".", color="white", markersize=3, label="M0 pick")
    for wavelength, style, name in (
        (min_resolvable_wavelength(image.acquisition), "--", "2·dx"),
        (longest_reached_wavelength(image.acquisition), ":", "3·L"),
    ):
        if wavelength:
            axes.plot(
                fs,
                wavelength * fs,
                style,
                color="red",
                linewidth=1,
                label=f"λ {name}, {wavelength:g} m",
            )
    axes.set_xlim(float(fs[0]), float(fs[-1]))
    axes.set_ylim(float(vs[0]), float(vs[-1]))
    axes.set_xlabel("frequency (Hz)", fontsize=8)
    axes.set_ylabel("phase velocity (m/s)", fontsize=8)
    axes.set_title(title, fontsize=8)
    axes.tick_params(labelsize=7)
    axes.legend(fontsize=6, loc="upper right")


def _png(figure: Figure) -> bytes:
    data = io.BytesIO()
    figure.savefig(data, format="png")
    return data.getvalue()


def _position(where: str) -> float:
    try:
        return float(where)
    except ValueError:
        raise ValueError(f"where: a window's xmid, m (not '{where}').") from None


def _span(where: str) -> tuple[float, float]:
    """`a-b`, m; one position for both ends."""
    low, _, high = where.partition("-")
    try:
        return (float(low), float(high or low))
    except ValueError:
        raise ValueError(f"where: the windows' span, a-b in m (not '{where}').") from None


def _nearest(folders: list[str], position: float) -> str:
    xmids = np.array([float(folder.removeprefix("xmid_")) for folder in folders])
    return folders[int(np.argmin(np.abs(xmids - position)))]
