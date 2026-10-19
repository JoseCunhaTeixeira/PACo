"""PACo's atlas drawn again (python -m paco.autonomous.atlas <input> <scratch> [<out>]): the
reference figures the host shows a model with vision (prompts.atlas), from PAC's test lines in
<input> (active_test_profile and passive_test_profile, which PAC ships) studied in <scratch>, each
sheet's figure written into <out>, by default the prompts' atlas folder, beside its caption. Each
panel is a window of those lines at the settings that show its case, found by trying them: a change
of the processing that moves a case asks to choose them again, and to read the captions again."""

import io
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from PIL import Image
from sigpipe.base.stream import Stream
from sigpipe.dataio.stream.loading import load_stream
from sigpipe.dataio.stream.plotting import draw_stream
from sigpipe.masw.runs import load_image, xmid_of

from paco.autonomous import prompts
from paco.autonomous.sensors.scorecard import measure
from paco.autonomous.tools.look import (
    DPI,
    RECORD_FIGURES,
    WINDOW_FIGURES,
    Mute,
    draw_correlations,
    draw_image,
    draw_lines,
)
from paco.autonomous.tools.survey import SURVEY, survey
from paco.autonomous.tools.trials import (
    RESULT,
    TRIALS,
    Study,
    TrialRun,
    WindowResult,
    new_study,
    try_settings,
)
from paco.qc.config import QCConfig
from paco.settings import Settings

ACTIVE = "active_test_profile"
PASSIVE = "passive_test_profile"
# Each line's settings, which the examples change: 11 receivers on the active line, 24 on the
# passive one.
LINES: dict[str, dict[str, Any]] = {
    ACTIVE: {"masw": {"length": 11}},
    PASSIVE: {"masw": {"length": 24}},
}
SHORT = {"masw": {"length": 5}}
CORRELATED = {"mode": "passive-active"}
MUTED = {"mode": "passive-active", "muting": {"method": "mute", "vmin": 50, "vmax": 1500}}
CONE = {"mode": "passive-active", "muting": {"method": "mute", "vmin": 80, "vmax": 1500}}
PASSIVE_SHORT = {"masw": {"length": 11}}
# The colours a sheet keeps: as sharp to the eye, a few times lighter, and the host sends the
# atlas with every call.
COLOURS = 128
# The gathers' mute, drawn: PACo's standard window, with the tail PACo gives it.
MUTE = Mute(vmin=80, vmax=1500)


@dataclass(frozen=True)
class Example:
    """A panel: its title, the line and the change of its settings whose window it draws, and
    that window's position, m."""

    title: str
    line: str
    changes: Mapping[str, Any]
    xmid: float

    @property
    def window(self) -> tuple[str, float]:
        """What its window depends on, besides its line: its change, and its position."""
        return json.dumps(dict(self.changes), sort_keys=True), self.xmid


# A panel's drawing: its axes, its example, its window's folder and results.
type Panel = Callable[[Axes, Example, Path, WindowResult], None]

IMAGES = (
    Example("1. Clean", ACTIVE, MUTED, 108.0),
    Example("2. Stripes", ACTIVE, CORRELATED, 52.5),
    Example("3. Smear", ACTIVE, SHORT, 42.0),
    Example("4. A weak ridge", ACTIVE, {}, 43.5),
    Example("5. A pick off the ridge", ACTIVE, SHORT, 139.5),
    Example("6. Aliases, and beyond reach", PASSIVE, PASSIVE_SHORT, 39.0),
)
CORRELATIONS = (
    Example("1. Clean", ACTIVE, CONE, 108.0),
    Example("2. Noisy", ACTIVE, CORRELATED, 52.5),
    Example("3. Too fast for surface waves", PASSIVE, PASSIVE_SHORT, 39.0),
)


def make(input_dir: Path, scratch: Path, out: Path) -> None:
    """The atlas's figures, from the test lines in `input_dir` studied in `scratch`, into
    `out`."""
    settings = Settings(input_dir=input_dir, output_dir=scratch, workers=4)  # pyright: ignore[reportCallIssue]
    config = QCConfig()
    studies = {line: new_study(line, settings) for line in LINES}
    cone = survey(studies[ACTIVE], settings, config).cone_m_s
    if cone is None:
        raise ValueError(f"{ACTIVE}'s survey found no surface waves' cone.")
    found = {
        line: _windows(
            study, [one for one in (*IMAGES, *CORRELATIONS) if one.line == line], settings, config
        )
        for line, study in studies.items()
    }
    out.mkdir(parents=True, exist_ok=True)
    figures = {
        "gathers": _gathers(studies[ACTIVE], cone),
        "images": _sheet(IMAGES, found, 2, _image),
        "correlations": _sheet(CORRELATIONS, found, 1, _correlations),
    }
    for name, figure in figures.items():
        drawn = io.BytesIO()
        figure.savefig(drawn, format="png")
        sheet = Image.open(drawn).convert("RGB").quantize(colors=COLOURS)
        sheet.save(out / f"{name}.png", optimize=True)


def _windows(
    study: Study, examples: Sequence[Example], settings: Settings, config: QCConfig
) -> dict[tuple[str, float], tuple[Path, WindowResult]]:
    """Each of `examples`' window, its folder and results, by what it depends on (Example.window):
    its line's settings with its change, tried on the windows nearest the examples' positions."""
    keys = list(dict.fromkeys(one.window[0] for one in examples))
    variants = [json.loads(key) for key in keys if key != "{}"]
    xmids = sorted({one.xmid for one in examples})
    made = try_settings(
        study,
        variants,
        f"xmids:{json.dumps(xmids)}",
        current=LINES[study.profile],
        settings=settings,
        config=config,
    )
    trials = {json.dumps(one.changes, sort_keys=True): one.trial for one in made.tried}
    found: dict[tuple[str, float], tuple[Path, WindowResult]] = {}
    for example in examples:
        folder = study.folder / TRIALS / trials[example.window[0]]
        run = TrialRun.model_validate_json((folder / RESULT).read_text())
        result = min(
            (one for one in run.windows if one.error is None),
            key=lambda one: abs(one.xmid - example.xmid),
        )
        window = next(path for path in folder.glob("xmid_*") if xmid_of(path.name) == result.xmid)
        found[example.window] = (window, result)
    return found


def _gathers(study: Study, cone: tuple[float, float, float]) -> Figure:
    """The gathers' sheet: 105.dat with its wavefield named, the survey's cone and a mute drawn;
    101.dat, another source's wave train; 149.dat, a ringing trace."""
    records = study.folder / SURVEY / "records"
    figure = Figure(figsize=(12, 9), dpi=DPI, layout="constrained")
    grid = figure.add_gridspec(2, 2)
    slow, median, fast = cone
    top = figure.add_subplot(grid[0, :])
    stream = _gather(top, records / "105")
    draw_lines(top, stream, MUTE, [slow, fast])
    source = stream.acquisition.source.x
    for x, t, said in (
        (62.0, (62.0 - source) / median + 0.12, "surface waves: slow, strong, dispersive"),
        (112.0, 0.03, "refractions: fast, straight"),
        (100.0, 2.05, "other sources: no moveout from the shot"),
    ):
        _label(top, x, t, said)
    top.set_title(
        f"1. 105.dat: the cone {slow:.0f} to {fast:.0f} m/s in blue; a mute, "
        f"{MUTE.vmin:g} to {MUTE.vmax:g} m/s, in red",
        fontsize=9,
    )
    for column, (name, title, x, t, said) in enumerate(
        (
            ("101", "2. 101.dat: another source's wave train", 75.0, 2.45, "another source"),
            ("149", "3. 149.dat: a ringing trace", 45.0, 1.2, "a bad receiver"),
        )
    ):
        axes = figure.add_subplot(grid[1, column])
        _gather(axes, records / name)
        _label(axes, x, t, said)
        axes.set_title(title, fontsize=9)
    return figure


def _gather(axes: Axes, folder: Path) -> Stream:
    (stream,) = load_stream([folder / RECORD_FIGURES["gather"]])
    draw_stream(axes, stream)
    return stream


def _label(axes: Axes, x: float, t: float, said: str) -> None:
    axes.text(
        x,
        t,
        said,
        fontsize=9,
        color="#b00020",
        ha="center",
        va="center",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2},
    )


def _sheet(
    examples: Sequence[Example],
    found: Mapping[str, Mapping[tuple[str, float], tuple[Path, WindowResult]]],
    rows: int,
    panel: Panel,
) -> Figure:
    """`examples` on `rows` rows of panels, each drawn by `panel`."""
    columns = -(-len(examples) // rows)
    figure = Figure(figsize=(4 * columns, 3.6 * rows), dpi=DPI, layout="constrained")
    for index, example in enumerate(examples):
        folder, result = found[example.line][example.window]
        panel(figure.add_subplot(rows, columns, index + 1), example, folder, result)
    return figure


def _image(axes: Axes, example: Example, folder: Path, result: WindowResult) -> None:
    level = measure(result.g2, "background")
    said = f", background {level:.2f}" if level is not None else ""
    draw_image(axes, load_image(folder), folder, f"{example.title}{said}", legend=False)
    axes.title.set_fontsize(9)


def _correlations(axes: Axes, example: Example, folder: Path, result: WindowResult) -> None:
    (stream,) = load_stream([folder / WINDOW_FIGURES["correlations"]])
    draw_correlations(axes, stream)
    measures = [
        f"{said} {value:.{digits}f}"
        for said, value, digits in (
            ("lateral", measure(result.g2, "lateral_coherence"), 2),
            ("SNR dB", measure(result.g2, "snr_db"), 0),
        )
        if value is not None
    ]
    axes.set_title(f"{example.title}: {', '.join(measures)}", fontsize=9)


def main(argv: Sequence[str] | None = None) -> None:
    """python -m paco.autonomous.atlas <input> <scratch> [<out>]."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) not in (2, 3):
        raise SystemExit(__doc__)
    out = Path(arguments[2]) if len(arguments) == 3 else Path(str(prompts.ATLAS_FOLDER))
    make(Path(arguments[0]), Path(arguments[1]), out)


if __name__ == "__main__":
    main()
