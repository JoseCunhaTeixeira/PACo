"""What the agent reads to know what is there, changing nothing (the inspect tool): the
profiles, one profile's acquisition, the runs, one run's settings and windows, one window.
Each part of a run says who made it: the assistant, whose gates judged it, or a person in PAC's
pages, verified by them. Compact text, a line a window: the model's context is short."""

from pathlib import Path

from pydantic import ValidationError
from sigpipe.masw.inversion.measuring import InversionMeasures
from sigpipe.masw.picks import load_curves
from sigpipe.masw.profiles import inspect_profile, list_profiles
from sigpipe.masw.runs import RunError, RunManifest, find_run, list_runs, load_manifest
from sigpipe.masw.runs.finding import window_length

from paco.qc.config import CONFIG_FILE, QCConfig, load_qc_config, read_qc_config
from paco.qc.inverting import MEASURES_FILE
from paco.qc.log import read_attempts
from paco.qc.origin import WindowWork, assistant_run, run_work
from paco.qc.positions import at_positions, shown_xmid
from paco.qc.report import QCReport, UnitReport, build_report, line_step, stretches
from paco.qc.used import processing_used
from paco.settings import Settings

# The runs listed, the newest first.
MAX_RUNS = 20
# The model whose fit the lines say, as PAC shows it.
FIT_MODEL = "median"


def profiles_text(settings: Settings) -> str:
    """The profiles to process, each with its runs."""
    names = list_profiles(settings)
    if not names:
        return "No profile in the input folder."
    runs = [run.split("/")[0] for run in list_runs(settings)]
    return "\n".join(
        f"{name}: a profile to process, {_runs_said(runs.count(name))}" for name in names
    )


def _runs_said(count: int) -> str:
    return "no run yet" if not count else f"{count} run{'s' if count > 1 else ''}"


def profile_text(name: str, settings: Settings) -> str:
    """One profile's acquisition, before any run."""
    summary = inspect_profile(name, settings)
    source = (
        f"; shots from {summary.source_x_range_m[0]:g} to {summary.source_x_range_m[1]:g} m"
        if summary.source_x_range_m is not None
        else ""
    )
    return (
        f"{summary.name}: {summary.kind}, {summary.n_records} records, {summary.n_receivers} "
        f"receivers every {summary.receiver_spacing_m:g} m from "
        f"{summary.receiver_x_range_m[0]:g} to {summary.receiver_x_range_m[1]:g} m{source}; "
        f"sampled at {summary.sampling_rate_hz:g} Hz (Nyquist {summary.nyquist_hz:g} Hz), "
        f"records {summary.record_duration_range_s[0]:g} to "
        f"{summary.record_duration_range_s[1]:g} s; modes: {', '.join(summary.modes)}."
    )


def runs_text(settings: Settings) -> str:
    """Every run, the newest first: who made it, its windows, what it holds."""
    found = list_runs(settings)
    if not found:
        return "No run yet."
    lines = [_run_line(run.split("/", 1)[1], settings) for run in found[:MAX_RUNS]]
    if len(found) > MAX_RUNS:
        lines.append(f"... and {len(found) - MAX_RUNS} older run(s).")
    return "\n".join(lines)


def run_text(run_id: str, settings: Settings) -> str:
    """One run: who made it, its windows and settings, the retries it spent, then its windows,
    those alike next to each other in one line (their image, curves and model, who made each,
    the checks; their models' misfits and depths informed as ranges): a line per window would
    fill the model's context."""
    run_folder = find_run(run_id, settings)
    manifest = load_manifest(run_id, settings)
    report, work = _judged(run_folder, manifest, settings)
    budget = report.budgets.per_xmid_of_the_run * report.n_xmids
    by_unit = {unit.unit: unit for unit in report.units}
    lines = [
        _header(manifest, run_folder, work),
        "Settings: " + "; ".join(processing_used(manifest)),
        f"Retries: {report.retries} of {budget} spent.",
    ]
    step = line_step(window.xmid for window in manifest.windows)
    groups: list[tuple[str, list[float], list[float], list[float]]] = []
    for window in manifest.windows:
        state, fit, depth = _window_state(run_folder, work[window.folder], by_unit)
        if not groups or groups[-1][0] != state:
            groups.append((state, [], [], []))
        _, xmids, fits, depths = groups[-1]
        xmids.append(window.xmid)
        fits += [fit] if fit is not None else []
        depths += [depth] if depth is not None else []
    for state, xmids, fits, depths in groups:
        lines.append(f"{stretches(xmids, step)}: {state}{_numbers(fits, depths)}.")
    return "\n".join(lines)


def window_text(run_id: str, position: float, settings: Settings) -> str:
    """The window nearest `position` (m): its image's checks, each curve (who picked it, its
    points, band and wavelengths, its checks), its model (misfit, depth informed, Vs)."""
    run_folder = find_run(run_id, settings)
    manifest = load_manifest(run_id, settings)
    (unit,), _ = at_positions(manifest, [position])
    report, work = _judged(run_folder, manifest, settings)
    own = next((one for one in report.units if one.unit == unit), None)
    one = work[unit]
    xmid = next(window.xmid for window in manifest.windows if window.folder == unit)
    nearest = "" if abs(xmid - position) < 1e-6 else f" (the nearest to {position:g} m)"
    head = f"xmid {shown_xmid(xmid)}{nearest}: "
    lines = [head + _window_line(run_folder, one, {unit: own} if own else {})]
    for gate in ("G2", "G3", "G4", "G5", "G6"):
        for flag in own.flags.get(gate, ()) if own is not None else ():
            lines.append(f"{gate} {flag.name}: {flag.message}")
    saved = load_curves(run_folder / unit)
    for curve in saved.dispersion_curves if saved is not None else ():
        fs, vs = curve.fs, curve.vs
        wavelengths = vs / fs
        who = "by hand" if curve.mode.label in one.by_hand else "automatic"
        lines.append(
            f"{curve.mode.label} ({who}): {fs.size} points, {fs.min():.3g}-{fs.max():.3g} Hz, "
            f"{vs.min():.0f}-{vs.max():.0f} m/s, wavelengths "
            f"{wavelengths.min():.3g}-{wavelengths.max():.3g} m."
        )
    measures = _measures(run_folder / unit)
    if measures is not None and measures.vs_at_depths:
        lines.append(
            "Vs: "
            + ", ".join(f"{vs:.0f} m/s at {depth:g} m" for depth, vs in measures.vs_at_depths)
            + "."
        )
    return "\n".join(lines)


def _judged(
    run_folder: Path, manifest: RunManifest, settings: Settings
) -> tuple[QCReport, dict[str, WindowWork]]:
    """The run's report as its log holds it now, and whose each window's work is: read, never
    written (a run the assistant never checked is read with the configuration it would use)."""
    config: QCConfig = (
        read_qc_config(run_folder)
        if (run_folder / CONFIG_FILE).exists()
        else load_qc_config(settings.qc_config)
    )
    attempts = read_attempts(run_folder)
    report = build_report(manifest.run_id, run_folder, config.budgets, len(manifest.windows))
    return report, run_work(run_folder, manifest, attempts)


def _header(manifest: RunManifest, run_folder: Path, work: dict[str, WindowWork]) -> str:
    """Who made the run, when, and its windows."""
    maker = "the assistant" if assistant_run(read_attempts(run_folder)) else "PAC's pages"
    masw = manifest.preset.masw
    spacing = manifest.profile.receiver_spacing_m
    span = window_length(run_folder)
    xmids = [window.xmid for window in manifest.windows]
    where = f", xmid {shown_xmid(min(xmids))}-{shown_xmid(max(xmids))} m" if xmids else ""
    every = "receiver" if masw.step == 1 else f"{masw.step} receivers"
    return (
        f"Run {manifest.run_id}: {manifest.profile.name} ({manifest.preset.mode}), processed by "
        f"{maker} on {manifest.started_at:%Y-%m-%d %H:%M}. {len(work)} windows of {masw.length} "
        f"receivers ({span:g} m), one every {every} ({masw.step * spacing:g} m){where}."
    )


def _run_line(run_id: str, settings: Settings) -> str:
    """A run in one line: who made it, its windows, and what it holds; a run whose manifest
    does not read, said as such."""
    try:
        run_folder = find_run(run_id, settings)
        manifest = load_manifest(run_id, settings)
    except RunError, ValidationError:
        return f"Run {run_id}: its manifest does not read (being written, or broken)."
    work = run_work(run_folder, manifest)
    images = sum(one.image is not None for one in work.values())
    m0 = [one.m0 for one in work.values() if one.m0 is not None]
    higher = sum(len(one.modes) - (one.m0 is not None) for one in work.values())
    models = [one.model for one in work.values() if one.model is not None]

    def of(count: int, what: str, *notes: tuple[int, str]) -> str:
        said = [f"{number} {note}" for number, note in notes if number]
        return f"{count} {what}" + (f" ({', '.join(said)})" if said else "")

    held = [
        of(images, "images"),
        of(
            len(m0),
            "M0 curves",
            (m0.count("user"), "by hand"),
            (m0.count("unjudged"), "not judged"),
        ),
        *([f"{higher} higher-mode curves (by hand)"] if higher else []),
        of(len(models), "models", (models.count("user"), "by hand")),
    ]
    return _header(manifest, run_folder, work) + f" Holds {', '.join(held)}."


def _window_line(run_folder: Path, work: WindowWork, by_unit: dict[str, UnitReport]) -> str:
    """A window in one line: its image, its curves, its model, each with who made it and its
    checks, and its model's misfit and depth informed."""
    state, fit, depth = _window_state(run_folder, work, by_unit)
    fits, depths = [fit] if fit is not None else [], [depth] if depth is not None else []
    return f"{state}{_numbers(fits, depths)}."


def _numbers(fits: list[float], depths: list[float]) -> str:
    """Models' misfits and depths informed, as ranges."""

    def span(values: list[float], form: str) -> str:
        low, high = format(min(values), form), format(max(values), form)
        return low if low == high else f"{low}-{high}"

    said = [f"misfit {span(fits, '.2f')}"] if fits else []
    said += [f"informed to {span(depths, '.3g')} m"] if depths else []
    return f" ({', '.join(said)})" if said else ""


def _window_state(
    run_folder: Path, work: WindowWork, by_unit: dict[str, UnitReport]
) -> tuple[str, float | None, float | None]:
    """A window's image, curves and model, each with who made it and its checks; and its
    model's misfit and depth informed, when it has them."""
    unit = by_unit.get(work.unit)
    verdicts = unit.verdicts if unit is not None else {}

    def checks(*gates: str) -> str:
        said = [f"{gate} {verdicts[gate]}" for gate in gates if gate in verdicts]
        return ", ".join(said) if said else "not checked"

    image = (
        "no image"
        if work.image is None
        else "image by hand"
        if work.image == "user"
        else f"image {checks('G2')}"
    )
    if work.m0 is None:
        curve = "no M0"
    elif work.m0 == "user":
        curve = "M0 by hand"
    elif work.m0 == "unjudged":
        curve = "M0 automatic, not judged"
    else:
        curve = f"M0 {checks('G3', 'G4')}"
    higher = [mode.label for mode in work.modes if mode.number > 0]
    if higher:
        curve += f"; {', '.join(higher)} by hand"
    if work.model is None:
        return f"{image}; {curve}; no model", None, None
    measures = _measures(run_folder / work.unit)
    if measures is None:
        fit, depth = None, None
    else:
        fit = next((one.misfit for one in measures.fits if one.model == FIT_MODEL), None)
        depth = measures.useful_depth_m
    model = "model by hand" if work.model == "user" else f"model {checks('G5', 'G6')}"
    return f"{image}; {curve}; {model}", fit, depth


def _measures(window: Path) -> InversionMeasures | None:
    path = window / MEASURES_FILE
    return InversionMeasures.model_validate_json(path.read_text()) if path.exists() else None
