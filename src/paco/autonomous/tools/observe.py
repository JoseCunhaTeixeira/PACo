"""observe (docs/autonomous_plan.md §6): the gates' results on the committed line, as text. A summary by
default, PACo's own (each verdict's count, each flag with its units as stretches of xmids, and the
change it suggests); with `where`, the rows of the units it names, in pages. On the line, a text
pseudo-section too: each curve's velocity at a few wavelengths, along the line."""

import re
from pathlib import Path
from typing import Literal

import numpy as np
from sigpipe.masw.picks import CURVES_FILE
from sigpipe.masw.runs import RunManifest

from paco.qc.config import QCConfig
from paco.qc.g4_profile import LINE
from paco.qc.judging import saved_m0
from paco.qc.log import latest, read_attempts
from paco.qc.models import Attempt, GateResult, Stage
from paco.qc.report import build_report, summarize_report, xmid_of

type Level = Literal["records", "images", "curves", "line", "models", "models_line"]

# Each level's gate, and the stage whose attempts hold its results.
GATES: dict[Level, tuple[str, Stage]] = {
    "records": ("G1", "preprocessing"),
    "images": ("G2", "phase_shift"),
    "curves": ("G3", "picking"),
    "line": ("G4", "picking"),
    "models": ("G5", "inversion"),
    "models_line": ("G6", "inversion"),
}
PAGE = 30  # rows at most a page
# The wavelengths the pseudo-section reads each curve's velocity at, m.
SECTION_WAVELENGTHS = (2.0, 5.0, 10.0, 20.0, 30.0)
WHERE = "xmids:a-b, flag:<name> or verdict:<pass|retry|reject>, joined by +"
_WHERE = re.compile(r"xmids:(-?[\d.]+)-(-?[\d.]+)|flag:(\w+)|verdict:(pass|retry|reject)")


def observe(
    run_folder: Path, level: Level, config: QCConfig, where: str | None = None, page: int = 1
) -> str:
    """What the gate of `level` said on the run in `run_folder`: its summary, or, with `where`,
    the rows of the units it names, page `page` of them."""
    gate, stage = GATES[level]
    attempts = read_attempts(run_folder)
    results = _results(attempts, gate, stage)
    if not results:
        return f"No {gate} result on this run yet: {_missing(level)}"
    if where is not None:
        return _rows(results, gate, where, page)
    manifest = RunManifest.model_validate_json((run_folder / "run.json").read_text())
    report = build_report(manifest.run_id, run_folder, config.budgets, len(manifest.windows))
    summary = summarize_report(report, (gate,), retries=False)
    if level == "line":
        summary += "\n" + pseudo_section(run_folder, [window.folder for window in manifest.windows])
    return summary


def pseudo_section(run_folder: Path, folders: list[str]) -> str:
    """Each window's saved M0 curve read at SECTION_WAVELENGTHS, along the line: a table of
    velocities, m/s ("-" where the curve does not reach)."""
    header = "xmid (m)  " + "  ".join(f"{wavelength:>5g} m" for wavelength in SECTION_WAVELENGTHS)
    lines = ["Pseudo-section, velocity (m/s) at each wavelength:", header]
    for folder in folders:
        curve = saved_m0(run_folder / folder / CURVES_FILE)
        if curve is None or len(curve.fs) == 0:
            continue
        wavelengths = curve.vs / curve.fs
        order = np.argsort(wavelengths)
        known, velocities = wavelengths[order], curve.vs[order]
        cells = [
            f"{float(np.interp(target, known, velocities)):7.0f}"
            if known[0] <= target <= known[-1]
            else "      -"
            for target in SECTION_WAVELENGTHS
        ]
        lines.append(f"{folder.removeprefix('xmid_'):>8}  " + "  ".join(cells))
    return "\n".join(lines)


def _results(attempts: tuple[Attempt, ...], gate: str, stage: Stage) -> dict[str, GateResult]:
    """Each unit's latest result of `gate`, on the latest attempt of `stage`."""
    found: dict[str, GateResult] = {}
    for unit in dict.fromkeys(attempt.unit for attempt in attempts):
        attempt = latest(attempts, unit, stage)
        if attempt is not None and (result := attempt.results.get(gate)) is not None:
            found[unit] = result
    return found


def _rows(results: dict[str, GateResult], gate: str, where: str, page: int) -> str:
    """The rows of the units `where` names, page `page`."""
    kept = [(unit, result) for unit, result in results.items() if _named(unit, result, where)]
    if not kept:
        return f"No {gate} result where {where}."
    kept.sort(key=lambda item: (xmid_of(item[0]) is None, xmid_of(item[0]) or 0.0, item[0]))
    pages = (len(kept) + PAGE - 1) // PAGE
    if not 1 <= page <= pages:
        raise ValueError(f"page: 1 to {pages}.")
    shown = kept[(page - 1) * PAGE : page * PAGE]
    lines = [f"{gate} where {where}: {len(kept)} units, page {page} of {pages}."]
    lines += [describe_result(unit, result) for unit, result in shown]
    return "\n".join(lines)


def _named(unit: str, result: GateResult, where: str) -> bool:
    """Whether `where`'s every part holds for `unit` and its `result`."""
    for text in (part.strip() for part in where.split("+")):
        found = _WHERE.fullmatch(text)
        if found is None:
            raise ValueError(f"Unknown filter '{text}': {WHERE}.")
        low, high, flag, verdict = found.groups()
        xmid = xmid_of(unit)
        if low is not None and (xmid is None or not float(low) <= xmid <= float(high)):
            return False
        if flag is not None and flag not in {one.name for one in result.flags}:
            return False
        if verdict is not None and result.verdict != verdict:
            return False
    return True


def describe_result(unit: str, result: GateResult) -> str:
    """A unit's result in one line: its verdict, its flags, what it kept, its measures."""
    name = "the line" if unit == LINE else unit.removeprefix("xmid_")
    flags = ", ".join(flag.name for flag in result.flags) or "no flag"
    kept = []
    if result.kept.wavelength_m is not None:
        kept.append(f"λ {result.kept.wavelength_m[0]:.3g}-{result.kept.wavelength_m[1]:.3g} m")
    if result.kept.band_hz is not None:
        kept.append(f"{result.kept.band_hz[0]:.3g}-{result.kept.band_hz[1]:.3g} Hz")
    measures = ", ".join(
        f"{metric.name} {metric.value:.3g}" for metric in result.metrics if metric.value is not None
    )
    return f"{name}: {result.verdict}; {flags}" + "".join(
        f"; {part}" for part in (*kept, measures) if part
    )


def _missing(level: Level) -> str:
    match level:
        case "records" | "images":
            return "commit a line first."
        case "curves" | "line":
            return "commit a line first; its windows are picked as it is made."
        case "models" | "models_line":
            return "invert the frozen curves first (P4)."
