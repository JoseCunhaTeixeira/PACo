"""look: figures for a model with vision, of the committed line, of an
experiment's variants, and before the line's first commit of the survey's raw records. A record's
gather or spectra, as its preprocessing and G1 drew them, a mute's window or velocities from the
shot drawn on a gather on request; a window's dispersion image with its pick and the wavelengths
that bound what it resolves (twice the receiver spacing, under which a ridge may be an alias; three
window lengths, its reach), as PAC shows it; a sheet of up to 12 windows' images; a knockout's
sheet of the search's variants, each at the same places, from their thumbnails; an experiment's
variants side by side, sheets of 4, their images and picks at the same windows, and on a passive or
passive-active line their stacked correlations; a window's model, as PAC's inversion card draws it: its Vs profile with
the curves' density, its fit on the image, its marginals, its chains. A passive or passive-active
window's stacked correlations (its virtual shot gather), and a passive window's FK selection of its
segments. Without vision, a window's ridge table stands in its image's stead, G1's measures in a
record's, G2's in its correlations', the selection's counts in its figure's, the panels' measures
in an experiment's, and the model's measures (G5's) in its figures'."""

import io
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict, Field
from sigpipe.algorithms.picking.dispersion.curve import (
    longest_reached_wavelength,
    min_resolvable_wavelength,
)
from sigpipe.base.dispersion_image import DispersionImage
from sigpipe.base.stream import Stream
from sigpipe.dataio.dispersion.plotting import draw_dispersion_image
from sigpipe.dataio.selection_plotting import load_selection
from sigpipe.dataio.stream.loading import load_stream
from sigpipe.dataio.stream.plotting import draw_stream, plot_stream
from sigpipe.masw.inversion.measuring import BandFit, InversionMeasures
from sigpipe.masw.inversion.window import (
    CHAINS_FIGURE,
    IMAGE_FIGURE,
    MARGINALS_FIGURE,
    WINDOW_FIGURE,
)
from sigpipe.masw.picks import CURVES_FILE
from sigpipe.masw.runs import RunError, RunManifest, load_image, xmid_of

from paco.autonomous.sensors.ridges import format_ridge_table, ridge_table
from paco.autonomous.sensors.scorecard import measure, window_image
from paco.autonomous.tools import grid as gridding
from paco.autonomous.tools.observe import describe_result
from paco.autonomous.tools.survey import SAVED, SURVEY, Survey
from paco.autonomous.tools.trials import (
    EXPERIMENTS,
    RESULT,
    TRIALS,
    Experiment,
    Study,
    TrialRun,
    Tried,
    WindowResult,
)
from paco.qc.coherence import trial_indices
from paco.qc.inverting import MEASURES_FILE
from paco.qc.judging import CORRELATION_MODES, saved_m0
from paco.qc.log import latest, read_attempts
from paco.qc.muting import MuteRules

type What = Literal[
    "gather",
    "spectra",
    "image",
    "correlations",
    "selection",
    "sheet",
    "experiment",
    "profile",
    "fit",
    "marginals",
    "chains",
]
# The most windows a sheet shows, and the figures' resolution.
SHEET = 12
DPI = 90
# An experiment's sheet: up to 4 variants (rows, the current settings first), at up to 4 windows
# (columns).
ROWS = 4
COLUMNS = 4
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


class Mute(BaseModel):
    """A mute to draw on a gather: it keeps from x/vmax to x/vmin plus its width (x the offset)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    vmin: float = Field(gt=0, description="m/s: its slow edge.")
    vmax: float = Field(gt=0, description="m/s: its fast edge.")
    width: float | None = Field(
        default=None, ge=0, description="s kept after the slow edge, the wave train's tail."
    )


@dataclass(frozen=True)
class Seen:
    """What look gives: its text, and its figures as PNG files."""

    text: str
    figures: tuple[bytes, ...] = ()


def look(
    run_folder: Path,
    what: What,
    where: str,
    vision: bool,
    mute: Mute | None = None,
    velocities: Sequence[float] = (),
) -> Seen:
    """`what` of the run in `run_folder`: a record's (`where`, its file's name) gather or
    spectra, `mute` or `velocities` drawn on a gather; a window's image (`where`, its xmid, m);
    a sheet of the windows `where` names (`a-b`, m). Without `vision`, text in the figures'
    stead."""
    manifest = RunManifest.model_validate_json((run_folder / "run.json").read_text())
    if what in RECORD_FIGURES:
        record = next((one for one in manifest.records if one.name == where), None)
        if record is None:
            names = ", ".join(one.name for one in manifest.records)
            raise ValueError(f"No record '{where}' on this run: {names}.")
        shots = str(manifest.preset.mode) != "passive"
        return _record(
            run_folder,
            run_folder / record.folder,
            record.name,
            what,
            vision,
            mute,
            velocities,
            shots,
            raw=False,
        )
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
        draw_image(figure.add_subplot(), image, run_folder / folder, folder)
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
        draw_image(
            figure.add_subplot(rows, 4, index + 1),
            load_image(run_folder / folder),
            run_folder / folder,
            folder,
        )
    return Seen(
        f"{len(shown)} of the {len(within)} windows from {low:g} to {high:g} m.", (_png(figure),)
    )


def look_line(run_folder: Path) -> Seen:
    """A sheet of the run's images and picks, up to SHEET windows spread along all of its line."""
    manifest = RunManifest.model_validate_json((run_folder / "run.json").read_text())
    xmids = [xmid_of(window.folder) for window in manifest.windows if window.status == "succeeded"]
    if not xmids:
        return Seen("The run has no window with an image.")
    return look(run_folder, "sheet", f"{min(xmids):g}-{max(xmids):g}", vision=True)


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


def look_survey(
    study: Study,
    what: What,
    where: str,
    vision: bool,
    mute: Mute | None = None,
    velocities: Sequence[float] = (),
) -> Seen:
    """A raw record of `study`'s survey, unmuted and unfiltered (`where`, the record's file
    name): its gather or spectra as the survey drew them, `mute` or `velocities` drawn on a
    gather; G1's measures without vision."""
    if what not in RECORD_FIGURES:
        raise ValueError(
            "Before the first commit, look shows the survey's raw records: their gather or "
            "spectra. A window's figures come with the line."
        )
    saved = study.folder / SAVED
    if not saved.exists():
        raise RunError("No survey yet in this study: call survey first.")
    shots = Survey.model_validate_json(saved.read_text()).kind != "passive"
    survey = study.folder / SURVEY
    folders = sorted(path.name for path in (survey / "records").iterdir() if path.is_dir())
    stem = Path(where).stem
    if stem not in folders:
        raise ValueError(f"No record '{where}' in the survey: {', '.join(folders)}.")
    attempts = read_attempts(survey)
    name = next((one.unit for one in attempts if Path(one.unit).stem == stem), where)
    folder = survey / "records" / stem
    return _record(survey, folder, name, what, vision, mute, velocities, shots, raw=True)


def _record(
    log_folder: Path,
    folder: Path,
    name: str,
    what: What,
    vision: bool,
    mute: Mute | None,
    velocities: Sequence[float],
    shots: bool,
    *,
    raw: bool,
) -> Seen:
    """A record's gather or spectra (its figures in `folder`, its G1 in `log_folder`'s log; `raw`
    when unmuted and unfiltered), a mute or velocities drawn on a gather, a line with `shots`;
    G1's measures without vision."""
    if (mute is not None or velocities) and not shots:
        raise ValueError("A passive line's records have no shot: no mute or velocities to draw.")
    if not vision:
        attempt = latest(read_attempts(log_folder), name, "preprocessing")
        result = attempt.results.get("G1") if attempt is not None else None
        said = describe_result(name, result) if result is not None else "not judged yet"
        return Seen(f"{name}, G1's measures in its {what}'s stead (vision is off): {said}")
    whose = "its raw" if raw else "its"
    if what == "gather" and (mute is not None or velocities):
        return _drawn(folder, f"{name}: {whose} gather", mute, velocities)
    if mute is not None or velocities:
        raise ValueError("A mute or velocities are drawn on a gather, not on spectra.")
    path = folder / RECORD_FIGURES[what]
    if not path.exists():
        raise ValueError(f"{name} has no {what} drawn.")
    return Seen(f"{name}: {whose} {what}.", (path.read_bytes(),))


def _drawn(folder: Path, said: str, mute: Mute | None, velocities: Sequence[float]) -> Seen:
    """A record's gather as sigpipe draws it (`said`, what it is), with lines from its shot
    (draw_lines)."""
    (stream,) = load_stream([folder / RECORD_FIGURES["gather"]])
    figure = plot_stream(stream)
    lines = draw_lines(figure.axes[0], stream, mute, velocities)
    png = _png(figure)
    plt.close(figure)
    return Seen(f"{said}, {'; '.join(lines)}.", (png,))


def draw_lines(
    axes: Axes, stream: Stream, mute: Mute | None, velocities: Sequence[float]
) -> list[str]:
    """On `axes`, which hold `stream`'s gather, lines from its shot, t = |x - x_shot| / v:
    `velocities` dashed, and `mute`'s window solid, from x/vmax to x/vmin plus its width (PACo's
    when it gives none), the time shown down to half again the window's end; what they are, in
    words."""
    source = stream.acquisition.source.x
    xs = np.array([receiver.x for receiver in stream.acquisition.receivers], dtype=float)
    span = np.linspace(min(float(xs.min()), source), max(float(xs.max()), source), 200)
    offsets = np.abs(span - source)
    said: list[str] = []
    if mute is not None:
        width = mute.width if mute.width is not None else MuteRules().width_s
        last = offsets / mute.vmin + width
        axes.plot(span, offsets / mute.vmax, color="red", linewidth=1.0)
        axes.plot(span, last, color="red", linewidth=1.0)
        axes.set_ylim(min(1.5 * float(last.max()), float(stream.ts[-1])), float(stream.ts[0]))
        said.append(
            f"the mute's window in red, from x/{mute.vmax:g} to x/{mute.vmin:g} + {width:g} s"
        )
    for velocity in velocities:
        axes.plot(span, offsets / velocity, "--", color="#2a78d6", linewidth=0.9)
    if velocities:
        speeds = ", ".join(f"{velocity:g} m/s" for velocity in velocities)
        said.append(f"lines from the shot at {speeds} in blue, dashed")
    return said


@dataclass(frozen=True)
class _Row:
    """A variant on an experiment's sheet: its changes, its windows' length, m, and its windows
    with an image (their folder and results), by xmid."""

    label: str
    changes: dict[str, Any]
    mode: str
    length_m: float
    windows: dict[float, tuple[Path, WindowResult]]

    def at(self, position: float) -> tuple[float, Path, WindowResult] | None:
        """Its window nearest `position`, m: its xmid, folder and results."""
        if not self.windows:
            return None
        xmid = min(self.windows, key=lambda one: abs(one - position))
        return (xmid, *self.windows[xmid])

    def score(self, position: float) -> float | None:
        """The image score of its window nearest `position`."""
        found = self.at(position)
        return window_image(found[2].g2, found[2].g3) if found is not None else None


type Panel = Callable[[Axes, str, float, Path, WindowResult], None]


def look_experiment(study: Study, where: str, vision: bool) -> Seen:
    """Experiment `where` (E<n>) side by side: its current settings and every variant, sheets of
    the current settings and three variants in the experiment's order, or the variants named
    after a colon (up to 4, the current settings `current`); their images and
    picks at the windows where their image scores differ most, then spread along the line, none
    overlapping another; on a passive or passive-active line the first sheet's stacked
    correlations at the same windows. Without vision, the panels' measures."""
    name, _, named = where.partition(":")
    path = study.folder / EXPERIMENTS / f"{name.strip()}.json"
    if not path.exists():
        raise ValueError(f"No experiment '{name.strip()}' in this study: where is E<n>[:labels].")
    experiment = Experiment.model_validate_json(path.read_text())
    tried = {one.label: one for one in experiment.tried}
    sheets = [[label.strip() for label in named.split(",")]] if named else _sheets(experiment)
    unknown = [label for label in sheets[0] if label not in tried]
    if unknown:
        raise ValueError(f"No variant {', '.join(unknown)} in {experiment.id}: {', '.join(tried)}.")
    if len(sheets[0]) > ROWS:
        raise ValueError(f"At most {ROWS} variants side by side.")
    made = {label: _row(study, tried[label]) for sheet in sheets for label in sheet}
    rows = list(made.values())
    columns = _columns(experiment.xmids, rows, max(row.length_m for row in rows))
    at = ", ".join(f"{xmid:g}" for xmid in columns)
    if not vision:
        lines = [f"{experiment.id} at {at} m, the panels' measures (vision is off):"]
        lines += [f"- {row.label}: {_measured(row, columns)}" for row in rows]
        return Seen("\n".join(lines))
    figures = [_sheet([made[label] for label in sheet], columns, _image_panel) for sheet in sheets]
    said = "; then ".join(
        "; ".join(
            f"{label} {json.dumps(made[label].changes)}" if made[label].changes else label
            for label in sheet
        )
        for sheet in sheets
    )
    many = f", in {len(sheets)} sheets" if len(sheets) > 1 else ""
    text = (
        f"{experiment.id} side by side{many}, by row: {said}; at the windows nearest {at} m, "
        "where their image scores differ most. Each image with its automatic pick (white dots), "
        "λ = 2·dx (dashed) and 3·L (dotted) in red"
    )
    correlated = [made[label] for label in sheets[0] if made[label].mode in CORRELATION_MODES]
    if correlated:
        figures.append(_sheet(correlated, columns, _correlations_panel))
        labels = ", ".join(row.label for row in correlated)
        text += f"; then the stacked correlations of {labels} there, their virtual shot gathers"
    return Seen(f"{text}.", tuple(figures))


def _sheets(experiment: Experiment) -> list[list[str]]:
    """Every variant, by sheets of the current settings and three, in the experiment's order: no
    score orders them, none reads an image's quality."""
    labels = [one.label for one in experiment.tried[1:]]
    step = ROWS - 1
    return [["current", *labels[start : start + step]] for start in range(0, len(labels), step)]


def _row(study: Study, tried: Tried) -> _Row:
    """`tried`'s trial as a sheet's row."""
    folder = study.folder / TRIALS / tried.trial
    if not (folder / RESULT).exists():
        raise ValueError(
            "This experiment's images were removed with the study's working files (at the "
            "freeze, or when a session ended); before the freeze, try_settings makes them again."
        )
    run = TrialRun.model_validate_json((folder / RESULT).read_text())
    found = {xmid_of(path.name): path for path in folder.glob("xmid_*")}
    windows = {
        result.xmid: (found[result.xmid], result)
        for result in run.windows
        if result.error is None and result.xmid in found
    }
    return _Row(tried.label, tried.changes, run.mode, run.length_m, windows)


def _columns(xmids: Sequence[float], rows: Sequence[_Row], length_m: float) -> list[float]:
    """Up to COLUMNS of `xmids`: half where `rows`' image scores differ most, then spread along
    the line, none within `length_m` of another (their windows would overlap); in order along
    it. Many variants differ most at one stretch: the spread ones show the rest of the line."""

    def spread(xmid: float) -> float:
        scores = [score for row in rows if (score := row.score(xmid)) is not None]
        return max(scores) - min(scores) if len(scores) > 1 else 0.0

    differing = sorted((xmid for xmid in xmids if spread(xmid) > 0), key=spread, reverse=True)
    even = [xmids[index] for index in trial_indices(len(xmids), COLUMNS)]
    half = COLUMNS // 2
    chosen: list[float] = []
    for xmid in [*differing[:half], *even, *differing[half:]]:
        if len(chosen) < COLUMNS and all(abs(xmid - other) >= length_m for other in chosen):
            chosen.append(xmid)
    return sorted(chosen)


def _measured(row: _Row, columns: Sequence[float]) -> str:
    """`row`'s panels in words: each window's image score, background and G3 verdict."""
    cells = []
    for position in columns:
        found = row.at(position)
        if found is None:
            cells.append(f"no image near {position:g} m")
            continue
        xmid, _, result = found
        verdict = result.g3.verdict if result.g3 is not None else "no pick"
        cells.append(f"{xmid:g} m: {_scored(result)}, G3 {verdict}")
    return "; ".join(cells)


def _scored(result: WindowResult) -> str:
    """A window's image score and background, as known."""
    score = window_image(result.g2, result.g3)
    level = measure(result.g2, "background")
    said = f"image {score:.2f}" if score is not None else "no image score"
    return f"{said}, background {level:.2f}" if level is not None else said


def _sheet(rows: Sequence[_Row], columns: Sequence[float], panel: Panel) -> bytes:
    """A figure of `rows` by `columns`, each panel drawn by `panel`."""
    figure = Figure(figsize=(3.2 * len(columns), 2.7 * len(rows)), dpi=DPI, layout="constrained")
    for index, (row, position) in enumerate(product(rows, columns)):
        axes = figure.add_subplot(len(rows), len(columns), index + 1)
        found = row.at(position)
        if found is None:
            axes.set_axis_off()
            axes.set_title(f"{row.label}: no image near {position:g} m", fontsize=8)
            continue
        panel(axes, row.label, *found)
    return _png(figure)


def _image_panel(
    axes: Axes,
    label: str,
    xmid: float,
    folder: Path,
    result: WindowResult,  # noqa: ARG001
) -> None:
    """A variant's image and pick at a window, with no score: none reads an image's quality."""
    draw_image(axes, load_image(folder), folder, f"{label}, {xmid:g} m", legend=False)


# A knockout sheet's layout: a variant a row, a panel for each of the places along the line; each
# panel's size, inches.
PANEL_IN = (1.4, 1.15)


def draw_sheet(
    folder: Path, labels: Sequence[str], lengths: Mapping[str, int], places: Sequence[float]
) -> bytes:
    """A knockout's sheet (search): its variants a row each, top to bottom, each thumbnail of
    `folder`'s (<label>.npz) under the one of `places` along the line its window is nearest, as
    PAC draws images (each frequency's row from its least value, yellow, to its largest, black),
    the automatic pick in white dots, λ = 2·dx dashed and 3·L dotted in red; each variant's label
    and receivers at its row's start; a place no window of it is nearest left empty; no score."""
    columns = max(1, len(places))
    figure = Figure(figsize=(PANEL_IN[0] * columns, PANEL_IN[1] * len(labels) + 0.1), dpi=DPI)
    spec = figure.add_gridspec(
        len(labels), columns, wspace=0.05, hspace=0.3,
        left=0.005, right=0.995, top=1 - 0.25 / (PANEL_IN[1] * len(labels)), bottom=0.005,
    )  # fmt: skip
    for row, label in enumerate(labels):
        thumbs = gridding.read_thumbs(folder / f"{label}.npz")
        at = {_nearest_place(one.xmid, places): one for one in thumbs} if places else {}
        for place in range(columns):
            axes = figure.add_subplot(spec[row, place])
            axes.set_xticks([])
            axes.set_yticks([])
            if place == 0:
                axes.set_title(
                    f"{label} · {lengths[label]} rx", fontsize=9, loc="left", fontweight="bold"
                )
            if place in at:
                _thumb_panel(axes, at[place])
            else:
                axes.set_frame_on(False)
    return _png(figure)


def _nearest_place(xmid: float, places: Sequence[float]) -> int:
    """The index of the place along the line `xmid` is nearest."""
    return min(range(len(places)), key=lambda index: abs(places[index] - xmid))


def _thumb_panel(axes: Axes, thumb: gridding.Thumb) -> None:
    """A thumbnail as PAC draws its image, with its pick and the lines λ = 2·dx and 3·L."""
    fmin, fmax, vmin, vmax = thumb.extent
    axes.imshow(
        thumb.image, origin="lower", aspect="auto", cmap="gist_stern_r", vmin=0, vmax=255,
        extent=(fmin, fmax, vmin, vmax), interpolation="nearest",
    )  # fmt: skip
    fs, vs = thumb.pick
    if len(fs):
        axes.plot(fs, vs, ".", color="white", markersize=2)
    span = np.linspace(fmin, fmax, 50)
    for wavelength, style in zip(thumb.lines_m, ("--", ":"), strict=True):
        if wavelength:
            axes.plot(span, wavelength * span, style, color="red", linewidth=0.8)
    axes.set_xlim(fmin, fmax)
    axes.set_ylim(vmin, vmax)


def _correlations_panel(
    axes: Axes, label: str, xmid: float, folder: Path, result: WindowResult
) -> None:
    """A window's stacked correlations (draw_correlations), with G2's lateral coherence and SNR
    of them."""
    path = folder / WINDOW_FIGURES["correlations"]
    if path.exists():
        draw_correlations(axes, load_stream([path])[0])
    else:
        axes.set_axis_off()
    measures = [
        f"{said} {value:.{digits}f}"
        for said, value, digits in (
            ("lateral", measure(result.g2, "lateral_coherence"), 2),
            ("SNR dB", measure(result.g2, "snr_db"), 0),
        )
        if value is not None
    ]
    axes.set_title(f"{label}, {xmid:g} m: {', '.join(measures) or 'no measure'}", fontsize=8)


def draw_correlations(axes: Axes, stream: Stream) -> None:
    """A window's stacked correlations, its virtual shot gather, on `axes`: down to half again the
    time by which 99 % of their energy has arrived."""
    draw_stream(axes, stream)
    axes.set_ylim(min(1.5 * _arrived(stream), float(stream.ts[-1])), float(stream.ts[0]))


def _arrived(stream: Stream) -> float:
    """The time by which 99 % of `stream`'s energy has arrived, each trace scaled to its peak: a
    clean gather's arrivals come early, noise fills the whole of it."""
    xt = np.asarray(stream.xt, dtype=float)
    xt = xt / (np.max(np.abs(xt), axis=1, keepdims=True) + 1e-12)
    energy = np.cumsum(np.sum(xt**2, axis=0))
    index = int(np.searchsorted(energy, 0.99 * energy[-1]))
    return float(stream.ts[min(index, len(stream.ts) - 1)])


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


def draw_image(
    axes: Axes, image: DispersionImage, folder: Path, title: str, legend: bool = True
) -> None:
    """A window's dispersion image on `axes` as PAC shows it (each frequency's row scaled from its
    least value to its largest: a short window's high floor, 1/√N, does not paint it bright),
    with its saved pick and the lines λ = 2·dx (under it, the aliasing zone) and λ = 3·L (beyond
    it, the window's reach); their legend unless the text says them."""
    fs, vs = image.fs, image.vs
    draw_dispersion_image(axes, image, normalize=True)
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
    if legend:
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
