"""freeze: the one way from the curves to the inversion. It
checks that every window of the committed line is decided, that each accepted curve passes G3 with
no blocking flag and G4, and that each rejected window has its reason. Then it writes the
inversion's dataset, curves.json in the study's folder: each accepted curve's file and hash, its
band and wavelengths, its points, their median uncertainty, the share beyond the window's reach and
the notes it carries; each rejected window and why; and the line's facts the inversion needs. Once
it is written the study is past its curves: no processing or picking tool works on it
(trials.check_open)."""

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict
from sigpipe.masw.picks import CURVES_FILE
from sigpipe.masw.runs import RunError, window_length
from sigpipe.masw.runs.writing import write_atomic

from paco.autonomous.sensors.checks import median_known, shot_note, window_checks
from paco.autonomous.sensors.flags import flag_class
from paco.autonomous.tools.commit import Line
from paco.autonomous.tools.decide import acceptable, current
from paco.autonomous.tools.pick import manifest_of
from paco.autonomous.tools.survey import SAVED, Survey
from paco.autonomous.tools.trials import CURVES, Study, check_open
from paco.qc.judging import saved_m0
from paco.qc.log import latest, read_attempts
from paco.qc.report import stretches


class FrozenCurve(BaseModel):
    """An accepted window's curve, as the inversion takes it."""

    model_config = ConfigDict(frozen=True)

    unit: str
    xmid: float
    curve: str  # its file, in the run's folder (PAC's layout)
    sha256: str
    band_hz: tuple[float, float]
    wavelength_m: tuple[float, float]
    points: int
    uncertainty: float | None  # its points' median relative uncertainty (G3's)
    beyond: float | None  # the share of its points beyond three window lengths (G3's)
    notes: tuple[str, ...]  # G3's and G4's notes on it, as they say them
    reason: str | None  # the model's, accepting it


class RejectedWindow(BaseModel):
    """A rejected window, and why."""

    model_config = ConfigDict(frozen=True)

    unit: str
    xmid: float
    reason: str


class Frozen(BaseModel):
    """The frozen curves: the line they come from, its facts the inversion needs, each accepted
    curve and each rejected window."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    commit: int  # the line's last commit, L<n>
    settings: dict[str, Any]
    dx_m: float  # the receiver spacing
    length: int  # the windows' receivers
    length_m: float  # their length, from the first receiver to the last
    reach_m: float | None  # the line's reach, the survey's
    objective: dict[str, Any] | None  # the user's (P0), from the host: none yet
    curves: tuple[FrozenCurve, ...]
    rejected: tuple[RejectedWindow, ...]
    frozen_at: datetime


def frozen(study: Study) -> Frozen | None:
    """The study's frozen curves; none before the freeze."""
    path = study.folder / CURVES
    return Frozen.model_validate_json(path.read_text()) if path.exists() else None


def freeze(study: Study, run_folder: Path, line: Line) -> Frozen:
    """The committed `line` (its run in `run_folder`) frozen, once every window is decided;
    refused, saying what is missing, before."""
    check_open(study)
    manifest = manifest_of(run_folder)
    attempts = read_attempts(run_folder)
    decided = current(study, run_folder, manifest.run_id)
    problems: list[str] = []
    undecided = [window.xmid for window in manifest.windows if window.folder not in decided]
    if undecided:
        problems.append(f"{len(undecided)} windows are not decided, at {stretches(undecided)}")
    for unit, decision in decided.items():
        xmid = unit.removeprefix("xmid_")
        if decision.status == "accepted":
            problem, _ = acceptable(run_folder, attempts, unit)
            if problem is not None:
                problems.append(f"{xmid} is accepted, but {problem}: decide it again")
        elif not decision.reason:
            problems.append(f"{xmid} is rejected without a reason")
    if problems:
        raise RunError("Not frozen: " + "; ".join(problems) + ".")
    length_m = window_length(run_folder) or 0.0
    mode = str(manifest.preset.mode)
    curves: list[FrozenCurve] = []
    rejected: list[RejectedWindow] = []
    for window in manifest.windows:
        decision = decided[window.folder]
        if decision.status == "rejected":
            rejected.append(
                RejectedWindow(unit=window.folder, xmid=window.xmid, reason=decision.reason or "")
            )
            continue
        path = run_folder / window.folder / CURVES_FILE
        curve = saved_m0(path)
        picked = latest(attempts, window.folder, "picking")
        if curve is None or picked is None:  # acceptable() said it has both
            raise RunError(f"{window.folder} lost its curve while it was frozen: freeze again.")
        g3, g4 = picked.results["G3"], picked.results.get("G4")
        checks = window_checks(g3, length_m, mode != "passive")
        wavelengths = curve.vs / curve.fs
        curves.append(
            FrozenCurve(
                unit=window.folder,
                xmid=window.xmid,
                curve=str(path.relative_to(run_folder)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                band_hz=(round(float(curve.fs.min()), 3), round(float(curve.fs.max()), 3)),
                wavelength_m=(
                    round(float(wavelengths.min()), 3),
                    round(float(wavelengths.max()), 3),
                ),
                points=len(curve.fs),
                uncertainty=checks.uncertainty if checks is not None else None,
                beyond=checks.beyond if checks is not None else None,
                notes=tuple(
                    f"{result.gate} {flag.name}: {flag.message}"
                    for result in (g3, g4)
                    if result is not None
                    for flag in result.flags
                    if flag_class(flag) == "note" and not shot_note(flag.name, mode)
                ),
                reason=decision.reason,
            )
        )
    survey = study.folder / SAVED
    made = Frozen(
        run_id=manifest.run_id,
        commit=line.commits[-1].number,
        settings=line.settings,
        dx_m=manifest.profile.receiver_spacing_m,
        length=manifest.preset.masw.length,
        length_m=round(length_m, 3),
        reach_m=Survey.model_validate_json(survey.read_text()).reach_m if survey.exists() else None,
        objective=None,
        curves=tuple(curves),
        rejected=tuple(rejected),
        frozen_at=datetime.now(UTC),
    )
    write_atomic(study.folder / CURVES, made.model_dump_json(indent=2))
    return made


def describe_frozen(made: Frozen) -> str:
    """The freeze as the model reads it: the curves kept and the windows rejected, what the
    curves inform, and what follows."""
    total = len(made.curves) + len(made.rejected)
    depths = [curve.wavelength_m[1] / 2 for curve in made.curves]
    depth = median_known(depths)
    uncertainty = median_known(curve.uncertainty for curve in made.curves)
    lines = [
        f"Frozen, from L{made.commit} (run {made.run_id}): {len(made.curves)} curves accepted of "
        f"the line's {total} windows, {len(made.rejected)} rejected."
    ]
    if made.curves:
        lines.append(
            f"The curves inform down to {depth:.3g} m (half their longest wavelength, the median)"
            + (f", their points' uncertainty {uncertainty:.0%} (the median)" if uncertainty else "")
            + "."
        )
    if made.rejected:
        lines.append(f"Rejected: {stretches([one.xmid for one in made.rejected])}.")
    lines.append(
        "The inversion's dataset is the study's curves.json. No processing or picking now: the "
        "inversion's turn (P4)."
    )
    return "\n".join(lines)
