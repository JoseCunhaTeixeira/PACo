"""The known-truth inversion set (docs/geo_plan.md §12): curves forward-modelled from known layered
models (sigpipe's rayleigh/seismic/forward.py), with noise drawn from the Lorentzian uncertainty
each pick carries, inverted by both agents from the freeze on. The geophysicist agent starts at P4
on them; PACo's agent finds a run whose curves are picked. Each is scored on the models it keeps
(the geophysicist agent's accepted, PACo's passing G5): their median Vs's error down to the depth
they inform, and how often the truth lies within their P10 to P90.

The curves stand on active_p2's own windows (90 receivers, one receiver apart: 7 windows),
processed and picked through the geophysicist's tools with no model. The windows whose pick passed
G3 and G4, those PACo's agent inverts, are frozen, the others rejected; each frozen window's
picked curve lends its frequencies and geometry to the synthetic one that replaces it, its noise
seeded by the truth and the window."""

import hashlib
import shutil
import zlib
from collections.abc import Collection
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from mcp import Client
from mcp.types import TextContent
from pydantic import BaseModel, ConfigDict
from sigpipe.algorithms.inversion.rayleigh.seismic.forward import fwd_seismic_phase
from sigpipe.algorithms.picking.dispersion.curve import lorentzian_uncertainty
from sigpipe.base import DispersionCurve
from sigpipe.masw.inversion.measuring import InversionMeasures
from sigpipe.masw.inversion.window import VP_VS_RATIO, load_vs_spread
from sigpipe.masw.picks import CURVES_FILE, save_pick
from sigpipe.masw.runs import find_run, load_image, window_length
from sigpipe.masw.runs.writing import write_atomic

from paco.geo import server as geo_server
from paco.geo.tools import journal, models
from paco.geo.tools.commit import line_of
from paco.geo.tools.freeze import Frozen, FrozenCurve, RejectedWindow
from paco.geo.tools.pick import manifest_of
from paco.geo.tools.survey import SAVED, Survey
from paco.geo.tools.trials import CURVES, Study, new_study
from paco.geo.upstream.running import server_settings
from paco.qc.config import read_qc_config
from paco.qc.inverting import MEASURES_FILE, investigation_depth
from paco.qc.judging import saved_m0
from paco.qc.log import latest, read_attempts
from paco.settings import Settings, get_settings

PROFILE = "active_p2"
# The line the synthetic curves stand on: 7 windows of 90 receivers, imaged on a coarse grid
# (their curves are replaced).
LINE = {"masw": {"length": 90}, "dispersion": {"nv": 250}}
REASONS = {
    "masw length": "the known-truth set: few windows, long curves",
    "dispersion nv": "a coarse grid: the curves are replaced by synthetic ones",
}
ACCEPTED = "the known-truth set: its curve is replaced by a synthetic one"
REJECTED = "the known-truth set: its pick did not pass G3 and G4"
# The objective the study starts from, as the host would have read it.
OBJECTIVE = {
    "target_depth_m": None,
    "priority": "balanced",
    "effort": "normal",
    "site_notes": None,
    "profile": PROFILE,
    "given": {},
}
# The noise's standard deviation, in the Lorentzian uncertainties: the uncertainty the curve
# carries is the noise it holds, so that a misfit near 1 fits it.
NOISE_SCALE = 1.0
HALF_SPACE_M = 1000.0  # the half-space's thickness, as sigpipe's forward model takes it


@dataclass(frozen=True)
class Truth:
    """A layered model: its layers' thicknesses above the half-space, and their Vs, the
    half-space's last."""

    name: str
    thicknesses_m: tuple[float, ...]
    vs_m_s: tuple[float, ...]

    def vs_at(self, depths: np.ndarray) -> np.ndarray:
        """The model's Vs at `depths`, in metres."""
        tops = np.concatenate(([0.0], np.cumsum(self.thicknesses_m)))
        return np.asarray(self.vs_m_s)[np.searchsorted(tops, depths, side="right") - 1]

    def curve(self, picked: DispersionCurve, seed: int) -> DispersionCurve:
        """The model's fundamental mode at `picked`'s frequencies, as a pick of `picked`'s window:
        its noise drawn from the Lorentzian uncertainty of the window's array, the uncertainty it
        carries the picker's of its velocities."""
        fs = np.asarray(picked.fs, dtype=float)
        modelled = fwd_seismic_phase(
            [*self.thicknesses_m, HALF_SPACE_M], list(self.vs_m_s), 0, fs, VP_VS_RATIO
        )
        vs = np.asarray(modelled.vs, dtype=float)
        kept = np.isfinite(vs)
        fs, vs = fs[kept], vs[kept]
        sigma = lorentzian_uncertainty(fs, vs, picked.acquisition)
        if sigma is not None:
            rng = np.random.default_rng(seed)
            vs = vs + NOISE_SCALE * sigma * rng.standard_normal(vs.size)
        return DispersionCurve(
            fs=fs,
            vs=vs,
            mode=picked.mode,
            acquisition=picked.acquisition,
            vs_err=lorentzian_uncertainty(fs, vs, picked.acquisition),
            type=picked.type,
        )


# Shallow sites: Vs 150 to 420 m/s over their first 20 m.
TRUTHS = (
    Truth("two_layers", (5.0,), (200.0, 350.0)),
    Truth("three_layers", (3.0, 7.0), (180.0, 280.0, 420.0)),
    # A softer layer under a stiffer one, 35 % slower: past the 20 % drop the bounds allow.
    Truth("soft_layer", (3.0, 4.0), (260.0, 170.0, 380.0)),
    Truth("gradient", (2.0, 3.0, 4.0, 5.0), (150.0, 200.0, 260.0, 320.0, 380.0)),
)


class TruthResult(BaseModel):
    """How close a play's kept models came to the truth."""

    model_config = ConfigDict(frozen=True)

    windows: int  # the models scored: those the agent kept
    error: float | None  # the median relative error of their median Vs, down to the depth informed
    in_band: float | None  # the share of those depths where the truth lies within P10 to P90


async def prepare(
    truth: Truth, outputs: Path, inputs: Path, cache: Path | None, qc_config: Path | None = None
) -> Study:
    """A study of `PROFILE` at P4 in `outputs`, its frozen curves `truth`'s: the line surveyed,
    processed and picked through the geophysicist's tools, then frozen (frozen_as). The run keeps
    `qc_config` (the defaults when None), the configuration its plays are judged by."""
    with server_settings(outputs, inputs, cache, qc_config):
        settings = get_settings()
        study = new_study(PROFILE, settings)
        journal.objective(study, OBJECTIVE, "balanced, no target depth, normal effort")
        async with Client(geo_server.server) as client:
            for name, arguments in (
                ("survey", {}),
                ("commit", {"changes": LINE, "reasons": REASONS}),
            ):
                result = await client.call_tool(name, {"profile": PROFILE, **arguments})
                if result.is_error:
                    said = " ".join(
                        block.text for block in result.content if isinstance(block, TextContent)
                    )
                    raise RuntimeError(f"The known-truth line could not be made: {said}")
        journal.freeze(study, frozen_as(study, truth, settings))
    return study


def frozen_as(study: Study, truth: Truth, settings: Settings) -> Frozen:
    """The study's line frozen with `truth`'s curves: the windows whose pick passed G3 and G4,
    each curve replaced by the truth's; the others rejected."""
    line = line_of(study)
    if line is None:
        raise RuntimeError("The known-truth line was not committed.")
    run_folder = find_run(line.run_id, settings)
    manifest = manifest_of(run_folder)
    attempts = read_attempts(run_folder)
    length_m = window_length(run_folder) or 0.0
    curves: list[FrozenCurve] = []
    rejected: list[RejectedWindow] = []
    for window in manifest.windows:
        folder = run_folder / window.folder
        picked = latest(attempts, window.folder, "picking")
        results = picked.results if picked is not None else {}
        passed = all(gate in results and results[gate].verdict == "pass" for gate in ("G3", "G4"))
        real = saved_m0(folder / CURVES_FILE)
        if not passed or real is None:
            rejected.append(RejectedWindow(unit=window.folder, xmid=window.xmid, reason=REJECTED))
            continue
        synthetic = truth.curve(real, zlib.crc32(f"{truth.name}/{window.folder}".encode()))
        save_pick(folder, load_image(folder), synthetic)
        path = folder / CURVES_FILE
        wavelengths = synthetic.vs / synthetic.fs
        relative = synthetic.vs_err / synthetic.vs if synthetic.vs_err is not None else None
        curves.append(
            FrozenCurve(
                unit=window.folder,
                xmid=window.xmid,
                curve=str(path.relative_to(run_folder)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                band_hz=(round(float(synthetic.fs.min()), 3), round(float(synthetic.fs.max()), 3)),
                wavelength_m=(
                    round(float(wavelengths.min()), 3),
                    round(float(wavelengths.max()), 3),
                ),
                points=len(synthetic.fs),
                uncertainty=float(np.median(relative)) if relative is not None else None,
                beyond=float(np.mean(wavelengths > 3 * length_m)),
                notes=(),
                reason=ACCEPTED,
            )
        )
    if not curves:
        raise RuntimeError("No window of the known-truth line passed G3 and G4.")
    surveyed = study.folder / SAVED
    made = Frozen(
        run_id=manifest.run_id,
        commit=line.commits[-1].number,
        settings=line.settings,
        dx_m=manifest.profile.receiver_spacing_m,
        length=manifest.preset.masw.length,
        length_m=round(length_m, 3),
        reach_m=Survey.model_validate_json(surveyed.read_text()).reach_m
        if surveyed.exists()
        else None,
        objective=None,
        curves=tuple(curves),
        rejected=tuple(rejected),
        frozen_at=datetime.now(UTC),
    )
    write_atomic(study.folder / CURVES, made.model_dump_json(indent=2))
    return made


def copy_prepared(prepared: Path, outputs: Path) -> None:
    """The prepared study and its run, copied into a play's `outputs`."""
    shutil.copytree(prepared, outputs, dirs_exist_ok=True)


def kept(run_folder: Path, study: Study | None) -> list[str]:
    """The windows whose model the agent kept: the geophysicist agent's accepted (`study`), PACo's
    passing G5."""
    run_id = run_folder.name
    if study is not None:
        statuses = models.statuses(study, run_folder, run_id)
        return sorted(unit for unit, status in statuses.items() if status == "accepted")
    attempts = read_attempts(run_folder)
    units = sorted(path.parent.name for path in run_folder.glob(f"*/{MEASURES_FILE}"))
    return [
        unit
        for unit in units
        if (inverted := latest(attempts, unit, "inversion")) is not None
        and (g5 := inverted.results.get("G5")) is not None
        and g5.verdict == "pass"
    ]


def score(truth: Truth, run_folder: Path, units: Collection[str]) -> TruthResult:
    """How close the models of `units` in `run_folder` came to `truth`: down to each one's depth
    informed (G5's, else its curve's depth of investigation), its median Vs's relative error, and
    whether the truth lies within its P10 to P90."""
    config = read_qc_config(run_folder)
    errors: list[float] = []
    shares: list[float] = []
    for unit in sorted(units):
        folder = run_folder / unit
        spread = load_vs_spread(folder)
        curve = saved_m0(folder / CURVES_FILE)
        path = folder / MEASURES_FILE
        if spread is None or curve is None or not path.exists():
            continue
        measures = InversionMeasures.model_validate_json(path.read_text())
        informed = measures.useful_depth_m
        if informed is None:
            informed = investigation_depth(curve, config)
        within = spread.depths <= informed
        if not within.any():
            continue
        true = truth.vs_at(spread.depths[within])
        errors.append(float(np.median(np.abs(spread.middle[within] - true) / true)))
        inside = (spread.low[within] <= true) & (true <= spread.high[within])
        shares.append(float(np.mean(inside)))
    return TruthResult(
        windows=len(errors),
        error=round(float(np.median(errors)), 3) if errors else None,
        in_band=round(float(np.mean(shares)), 3) if shares else None,
    )
