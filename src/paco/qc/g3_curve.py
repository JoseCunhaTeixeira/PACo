"""G3, the QC of a picked curve (docs/qc_workflow.md): paco.quality's metrics on the M0 pick,
said in the gates' language, and the curve's own rules: points under twice the spacing or over
three window lengths flagged (the picker follows its ridge as far as it holds, at either end), no
jump onto another mode, no air wave, the trend, enough points, uncertainties the inversion can
use."""

import math

import numpy as np
from pydantic import ConfigDict, Field
from sigpipe.algorithms.picking.dispersion.tracking import PickedMode, PickingParameters
from sigpipe.base import DispersionImage
from sigpipe.masw.quality.curve import CurveLimits, measure_curve
from sigpipe.masw.quality.pick import constant_wavelength_start

from paco.qc.models import Flag, GateResult, Keep, Kept, Metric, Override, Reject
from paco.quality import ImageQuality, QualityParameters, quality_of

GATE = "G3"


class CurveThresholds(CurveLimits):
    """G3's limits (sigpipe's CurveLimits: how a curve is measured, PAC's alike), and what its
    fixes use: provisional, measured on the demo profiles (rule 9)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # PACo's own, refusing a limit it does not know (a typo) as the rest of its configuration;
    # the models frozen, the narrower type is safe.
    metrics: QualityParameters = Field(  # pyright: ignore[reportIncompatibleVariableOverride]
        default_factory=QualityParameters,
        description="Sharpness, prominence, on_data and constant wavelength.",
    )
    mute_vmin: float = Field(default=80.0, gt=0, description="m/s, the slowest wave the mute keeps")
    finest_step: float = Field(
        default=0.1,
        gt=0,
        description="m: the finest wavelength step a pick is resampled at to keep min_points.",
    )


def judge_curve(
    unit: str,
    image: DispersionImage,
    m0: PickedMode | None,
    thresholds: CurveThresholds,
    band: tuple[float, float] | None = None,
    picking: PickingParameters | None = None,
    nearest_offset: float | None = None,
    mutable: bool = True,
) -> GateResult:
    """G3's verdict on one window's M0 pick, with the band G2 found coherent (or G1 usable)
    when known, the band a fix narrows to, the picking parameters the fixes start from, and
    the distance from the window's nearest shot to its receivers. Not `mutable` (a passive
    line, which has no muting, or records muted already), a flag only a mute would fix rejects."""
    picking = picking or PickingParameters()
    # The curve's measures, sigpipe's (PAC's alike), each saying what it covers.
    report = measure_curve(image, m0, thresholds, nearest_offset)
    quality = quality_of(report.pick, thresholds.metrics)
    metrics = [Metric(**one.model_dump()) for one in report.measures]
    band = band or quality.band_hz
    receivers = image.acquisition.receivers
    length = abs(receivers[-1].x - receivers[0].x)
    cut = _constant_wavelength_cut(m0, length)
    flags = [_flag(name, quality, band, thresholds, cut, mutable) for name in quality.flags]
    n_traces = len(receivers)
    if m0 is None or m0.curve is None:
        kept = Kept(band_hz=quality.band_hz, n_points=quality.n_points, n_traces=n_traces)
        return _result(unit, metrics, flags, kept)

    # The picker follows its ridge as far as it holds, at either end: its points under twice the
    # spacing (the aliasing zone, where the ridge may be its alias) and over three window lengths
    # (beyond the window's reach, where it resolves no velocity) are flagged and kept, for the
    # agent or G4 to judge. The lines PAC draws on the image.
    if report.shortest and report.aliased > 0:
        flags.append(
            Flag(
                name="aliasing_zone",
                message=f"{report.aliased:.0%} of the points lie under twice the receiver "
                f"spacing ({report.shortest:g} m): the aliasing zone, where the ridge may be its "
                "alias.",
                stage="picking",
                action=Keep(note="points in the aliasing zone"),
                fixable=False,
            )
        )
    if report.longest and report.beyond > 0:
        flags.append(
            Flag(
                name="beyond_reach",
                message=f"{report.beyond:.0%} of the points lie over three window lengths "
                f"({report.longest:.3g} m): beyond the window's reach, where the pick may be "
                "the tracker's.",
                stage="picking",
                action=Keep(note="points beyond the window's reach"),
                fixable=False,
            )
        )

    kept_fs, kept_vs, kept_wl = report.fs, report.vs, report.wavelengths
    n_points = int(kept_vs.size)
    if n_points < thresholds.min_points:
        flags.append(_too_few_points(m0, n_points, thresholds, picking))
    # However many its points, a narrow span resolves no layered model. Keeping more of the ridge
    # may widen it.
    if n_points and report.ratio < thresholds.min_wavelength_ratio:
        flags.append(
            Flag(
                name="narrow_span",
                message=f"The curve spans {kept_wl.min():.1f}-{kept_wl.max():.1f} m of wavelength "
                f"(ratio {report.ratio:.2f}): too narrow for a layered model. Keep more of the "
                "ridge.",
                stage="picking",
                action=Override(
                    stage="picking",
                    overrides={
                        "min_relative_coherence": round(picking.min_relative_coherence * 0.6, 2)
                    },
                ),
            )
        )
    if report.jump > thresholds.max_jump:
        flags.append(_mode_jump(report.jump, kept_fs, kept_vs, picking))
    if report.air > thresholds.max_air_share:
        # No mute parts them: one just under the air wave's speed cuts a fraction of a
        # millisecond of it a metre, and every surface wave faster than it.
        low, high = thresholds.air_wave_band
        flags.append(
            Flag(
                name="air_wave",
                message=f"{report.air:.0%} of the points sit at {low:g}-{high:g} m/s: the air "
                "wave, not the ground; no mute parts them.",
                stage="picking",
                action=Reject(reason="the air wave, not the ground"),
                fixable=False,
            )
        )
    # Normal dispersion: velocity rising with wavelength. The opposite is flagged, not rejected.
    if np.isfinite(report.trend) and report.trend < 0:
        flags.append(
            Flag(
                name="inverse_dispersion",
                message=f"Velocity falls with wavelength (rank correlation {report.trend:+.2f}): "
                "a stiff layer over a softer one, if the pick is right.",
                stage="picking",
                action=Keep(note="an inverse trend can be geology"),
            )
        )
    # The near field: the line leaves out a window's shots nearer than half the trial curves'
    # longest wavelength where it has farther ones (coherence.near_field_windows). Its own curve
    # may reach longer wavelengths, and a window with near shots only keeps them: reported.
    limit = report.near_limit
    if nearest_offset is not None and limit is not None and nearest_offset < limit:
        flags.append(
            Flag(
                name="near_field",
                message=f"The nearest shot is {nearest_offset:.2f} m from the window, under "
                f"half the longest wavelength kept ({limit:.2f} m): the long wavelengths may "
                "read slow (near field).",
                stage="phase_shift",
                action=Keep(
                    note="reported: the window has no farther shot, or its curve reaches "
                    "longer wavelengths than the line's trial curves"
                ),
            )
        )

    kept = Kept(
        band_hz=(float(kept_fs.min()), float(kept_fs.max())) if n_points else quality.band_hz,
        wavelength_m=(float(kept_wl.min()), float(kept_wl.max())) if n_points else None,
        n_points=n_points,
        n_traces=n_traces,
    )
    return _result(unit, metrics, flags, kept)


def _mode_jump(
    jump: float, frequencies: np.ndarray, velocities: np.ndarray, picking: PickingParameters
) -> Flag:
    """The flag of a jump onto another mode: first cut the band where the largest step between
    points consecutive in frequency sits (after a jump the wavelengths interleave), the side
    with fewer points going; once the band is cut, track with a narrower corridor."""
    message = f"A {jump:.0%} step between consecutive points: the pick jumped onto another mode."
    if picking.fmin is None and picking.fmax is None:
        order = np.argsort(frequencies)
        fs, vs = frequencies[order], velocities[order]
        at = int(np.argmax(np.abs(np.diff(vs)) / vs[:-1]))
        cut = round(float(fs[at] + fs[at + 1]) / 2, 1)
        side = "fmax" if fs.size - at - 1 < at + 1 else "fmin"
        return Flag(
            name="mode_jump",
            message=f"{message} Cut the band where it starts ({side} {cut:g} Hz).",
            stage="picking",
            action=Override(stage="picking", overrides={side: cut}),
        )
    return Flag(
        name="mode_jump",
        message=f"{message} Track with a narrower corridor.",
        stage="picking",
        action=Override(stage="picking", overrides={"corridor": round(picking.corridor / 2, 3)}),
    )


def _result(unit: str, metrics: list[Metric], flags: list[Flag], kept: Kept) -> GateResult:
    acted = [flag for flag in flags if not isinstance(flag.action, Keep)]
    if not acted:
        verdict = "pass"
    elif any(not flag.fixable for flag in acted):
        verdict = "reject"
    else:
        verdict = "retry"
    return GateResult(
        gate=GATE, unit=unit, verdict=verdict, metrics=tuple(metrics), flags=tuple(flags), kept=kept
    )


def _too_few_points(
    m0: PickedMode, n_points: int, thresholds: CurveThresholds, picking: PickingParameters
) -> Flag:
    """Too few points: the kept ridge resampled finer when its wavelengths span enough for
    `min_points` at a step no finer than `finest_step` (the resampling's grid starts and ends
    on whole steps: two more steps than points); else more of the ridge kept."""
    kept = m0.velocities[m0.kept] / m0.frequencies[m0.kept]
    span = float(kept.max() - kept.min()) if kept.size else 0.0
    step = math.floor(span / (thresholds.min_points + 1) / thresholds.finest_step) * (
        thresholds.finest_step
    )
    said = f"{n_points} point{'s' if n_points != 1 else ''} in the curve: too few"
    if step >= thresholds.finest_step and step < picking.wavelength_step:
        step = round(step, 3)
        return Flag(
            name="too_few_points",
            message=f"{said}. The ridge spans {span:.2f} m of wavelength: resample it every "
            f"{step:g} m.",
            stage="picking",
            action=Override(stage="picking", overrides={"wavelength_step": step}),
        )
    return Flag(
        name="too_few_points",
        message=f"{said}. Keep more of the ridge's points.",
        stage="picking",
        action=Override(
            stage="picking",
            overrides={"min_relative_coherence": round(picking.min_relative_coherence * 0.6, 2)},
        ),
    )


def _constant_wavelength_cut(m0: PickedMode | None, length: float) -> float | None:
    """The longest wavelength to search, in window lengths, that leaves out the stretch where
    the pick follows the edge of what the window resolves; None without such a stretch."""
    if m0 is None or length <= 0:
        return None
    kept = m0.kept & (m0.frequencies > 0)
    if kept.sum() < 2:
        return None
    start = constant_wavelength_start(m0.frequencies[kept], m0.velocities[kept])
    return None if start is None else math.floor(start / length * 100) / 100


def _flag(
    name: str,
    quality: ImageQuality,
    band: tuple[float, float] | None,
    thresholds: CurveThresholds,
    cut: float | None,
    mutable: bool = True,
) -> Flag:
    match name:
        case "no_ridge":
            return Flag(
                name="no_ridge",
                message=f"No ridge kept above the noise floor ({quality.n_points} points): pick "
                "with a looser mode rule before giving up.",
                stage="picking",
                action=Override(stage="picking", overrides={"mode_min_ratio": 1.2}),
            )
        case "sharpness":
            return Flag(
                name="sharpness",
                message=f"Peaks {quality.sharpness:.2f} times as narrow as the window resolves: "
                "not a propagating wave at this window length.",
                stage="phase_shift",
                action=Reject(reason="peaks narrower than the window can resolve"),
                fixable=False,
            )
        case "prominence":
            said = (
                f"The ridge stands {quality.prominence:.1f} times above the rest of the image: "
                "barely."
            )
            # A filter cannot change the image (the phase shift divides each trace's spectrum
            # by its own amplitude): a mute alone, on records not muted yet.
            if not mutable:
                return Flag(
                    name="prominence",
                    message=f"{said} Its records muted already, or a passive line: nothing to try.",
                    stage="preprocessing",
                    action=Reject(reason="the ridge barely stands out"),
                    fixable=False,
                )
            return Flag(
                name="prominence",
                message=f"{said} Mute its records to their surface waves.",
                stage="preprocessing",
                action=Override(
                    stage="preprocessing",
                    overrides={
                        "muting": {"method": "mute", "vmin": thresholds.mute_vmin, "vmax": 1500.0}
                    },
                ),
            )
        case "on_data":
            overrides = {"dispersion": {"fmin": _low(band), "fmax": _high(band)}} if band else {}
            return Flag(
                name="on_data",
                message=f"Only {quality.on_data:.0%} of the points sit on their column's brightest "
                "value: competing ridges. Narrow the dispersion band.",
                stage="phase_shift",
                action=Override(stage="phase_shift", overrides=overrides),
            )
        case _:
            message = (
                f"{quality.constant_wavelength:.0%} of the points follow the edge of what the "
                "window resolves, not a dispersion curve"
            )
            if cut is None or cut <= 0:
                return Flag(
                    name="constant_wavelength",
                    message=f"{message}, away from its long wavelengths: no cut removes them.",
                    stage="picking",
                    action=Keep(note="not at the long wavelengths: a cut would not remove it"),
                )
            return Flag(
                name="constant_wavelength",
                message=f"{message}: cut the long wavelengths where it starts, "
                f"{cut:g} window lengths.",
                stage="picking",
                action=Override(stage="picking", overrides={"max_wavelength": cut}),
            )


def _low(band: tuple[float, float]) -> float:
    return round(max(0.0, band[0]), 1)


def _high(band: tuple[float, float]) -> float:
    return round(band[1], 1)
