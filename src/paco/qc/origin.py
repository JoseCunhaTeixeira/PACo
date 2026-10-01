"""Whose each window's work is: the assistant's, which its gates judge, or a person's, made in
PAC's pages and verified by them: no gate judges it, nothing automatic changes it, the
inversion takes it as it is, and a step the user asks for that would change it asks them
first.

A window's image is a person's when the assistant made none of the window's (a run processed
in PAC's pages); its curves are by sigpipe's rule (masw.runs.origin: changed by hand after
their last automatic pick, or a higher mode); its model and its soil column are a person's
when PAC's Seismic or Petrophysical inversion page made them (without an attempt of the
assistant's). An automatic M0 the assistant's checks are not of (PAC's own automatic pick, or
picked again there after them) is unjudged: judged before the inversion takes it."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from sigpipe.base.dispersion_curve import Mode
from sigpipe.masw.picks import load_curves
from sigpipe.masw.runs import RunManifest
from sigpipe.masw.runs.finding import IMAGE_FILE
from sigpipe.masw.runs.origin import JUDGED, M0, checks_current, curve_origin

from paco.qc.log import attempts_of, read_attempts
from paco.qc.models import Attempt, Stage

type Maker = Literal["assistant", "user"]
# An automatic M0 its gates judged as it is now, a person's, or automatic and not judged since
# it last changed.
type M0State = Literal["judged", "user", "unjudged"]
# A window's model files and soil column's, as PAC lists them.
MODEL_FILES = "SeismicInversion_*"
SOIL_FILES = "PetroInversion_*"


@dataclass(frozen=True)
class WindowWork:
    """What window `unit` holds, and whose each part is (None: it holds none)."""

    unit: str
    image: Maker | None
    m0: M0State | None
    modes: tuple[Mode, ...]  # every mode picked, M0 first
    model: Maker | None
    soils: Maker | None = None

    @property
    def by_hand(self) -> tuple[str, ...]:
        """What a person made of it in PAC: "image", each curve's mode, "model", "soils"."""
        return (
            *(("image",) if self.image == "user" else ()),
            *(mode.label for mode in self.modes if mode != M0 or self.m0 == "user"),
            *(("model",) if self.model == "user" else ()),
            *(("soils",) if self.soils == "user" else ()),
        )

    @property
    def frozen(self) -> bool:
        """Whether a new image would make a person's work stale: the window holds some."""
        return bool(self.by_hand)


def window_work(run_folder: Path, unit: str, attempts: Sequence[Attempt]) -> WindowWork:
    """Whose each part of window `unit` of the run in `run_folder` is, by its QC log's
    `attempts`."""
    window = run_folder / unit
    image: Maker | None = None
    if (window / IMAGE_FILE).exists():
        image = "assistant" if attempts_of(attempts, unit, "phase_shift") else "user"
    saved = load_curves(window)
    modes = tuple(
        sorted(
            (curve.mode for curve in saved.dispersion_curves) if saved is not None else (),
            key=lambda mode: (mode.number, mode.wave),
        )
    )
    m0: M0State | None = None
    if M0 in modes:
        picks = [a for a in attempts_of(attempts, unit, "picking") if a.status == "succeeded"]
        picked = _ended(a for a in picks if a.triggered_by != JUDGED)
        if curve_origin(window, M0, picked) == "hand":
            m0 = "user"
        else:
            m0 = "judged" if checks_current(window, _ended(picks)) else "unjudged"
    return WindowWork(
        unit=unit,
        image=image,
        m0=m0,
        modes=modes,
        model=_maker(window, MODEL_FILES, attempts, "inversion"),
        soils=_maker(window, SOIL_FILES, attempts, "petro_inversion"),
    )


def _maker(window: Path, files: str, attempts: Sequence[Attempt], stage: Stage) -> Maker | None:
    """Who made the window's results of `stage` (`files`): the assistant, when one of its
    attempts at it succeeded; else a person, in PAC; None without any."""
    if not any(window.glob(files)):
        return None
    done = [a for a in attempts_of(attempts, window.name, stage) if a.status == "succeeded"]
    return "assistant" if done else "user"


def run_work(
    run_folder: Path, manifest: RunManifest, attempts: Sequence[Attempt] | None = None
) -> dict[str, WindowWork]:
    """Every window of the run, by folder, and whose each part of it is."""
    logged = read_attempts(run_folder) if attempts is None else attempts
    return {
        window.folder: window_work(run_folder, window.folder, logged) for window in manifest.windows
    }


def assistant_run(attempts: Sequence[Attempt]) -> bool:
    """Whether the assistant processed the run (it made its images): else PAC's pages did."""
    return any(attempt.stage == "phase_shift" for attempt in attempts)


def left_alone(work: Mapping[str, WindowWork], units: Iterable[str]) -> list[str]:
    """Those of `units` holding a person's work: a stage done again for them would change it."""
    return [unit for unit in units if unit in work and work[unit].frozen]


def _ended(attempts: Iterable[Attempt]) -> datetime | None:
    ends = [attempt.finished_at or attempt.started_at for attempt in attempts]
    return max(ends) if ends else None
