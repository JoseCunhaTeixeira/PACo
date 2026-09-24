"""G3, the QC of a picked curve (docs/qc_workflow.md): today's dispersion_quality metrics on
the M0 pick, said in the gates' language, and the curve's own rules: wavelengths within what
the window resolves (from twice the spacing to a few window lengths), no jump onto another
mode, no air wave, the trend, enough points, uncertainties the inversion can use."""

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import rankdata
from sigpipe.base import DispersionImage

from paco.picking import PickedMode, PickingParameters
from paco.qc.models import Flag, GateResult, Keep, Kept, Metric, Override, Reject
from paco.quality import ImageQuality, QualityParameters, measure_quality

GATE = "G3"


class CurveThresholds(BaseModel):
    """G3's limits: provisional, measured on the demo profiles (rule 9)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metrics: QualityParameters = Field(
        default_factory=QualityParameters,
        description="Sharpness, prominence, on_data and constant wavelength, as before.",
    )
    max_wavelength_lengths: float = Field(
        default=2.0,
        gt=0,
        description="Points beyond this many window lengths are cut (the picking's "
        "max_wavelength): the window does not resolve them.",
    )
    max_jump: float = Field(
        default=0.3,
        gt=0,
        description="Relative velocity step between consecutive points of the resampled "
        "curve, at most: beyond it the pick jumped onto another mode.",
    )
    air_wave_band: tuple[float, float] = Field(
        default=(330.0, 345.0), description="m/s, where the air wave sits"
    )
    max_air_share: float = Field(
        default=0.5, gt=0, le=1, description="Share of the points in the air wave's band, at most."
    )
    air_wave_mute_vmax: float = Field(
        default=320.0,
        gt=0,
        description="m/s, the mute that removes the air wave: faster arrivals go",
    )
    mute_vmin: float = Field(default=80.0, gt=0, description="m/s, the slowest wave the mute keeps")
    min_points: int = Field(default=5, ge=2, description="Points within the wavelength range.")
    near_offset_wavelengths: float = Field(
        default=0.5,
        gt=0,
        description="The nearest shot's offset, as a share of the longest wavelength kept, at "
        "least: closer, the long wavelengths may read slow (near field). Reported, not applied.",
    )
    max_uncertainty: float = Field(
        default=0.5,
        gt=0,
        description="Median Lorentzian uncertainty over the velocity, at most: it depends on the "
        "window's geometry, the same for every window of a line.",
    )


def judge_curve(
    unit: str,
    image: DispersionImage,
    m0: PickedMode | None,
    thresholds: CurveThresholds,
    band: tuple[float, float] | None = None,
    picking: PickingParameters | None = None,
    nearest_offset: float | None = None,
) -> GateResult:
    """G3's verdict on one window's M0 pick, with the band G2 found coherent (or G1 usable)
    when known, the band a fix narrows to, the picking parameters the fixes start from, and
    the distance from the window's nearest shot to its receivers."""
    picking = picking or PickingParameters()
    quality = measure_quality(image, m0, thresholds.metrics)
    metrics = _metrics(quality, thresholds)
    band = band or quality.band_hz
    flags = [_flag(name, quality, band, thresholds) for name in quality.flags]
    n_traces = len(image.acquisition.receivers)
    if m0 is None or m0.curve is None:
        kept = Kept(band_hz=quality.band_hz, n_points=quality.n_points, n_traces=n_traces)
        return _result(unit, metrics, flags, kept)

    receivers = image.acquisition.receivers
    length = abs(receivers[-1].x - receivers[0].x)
    spacing = abs(receivers[1].x - receivers[0].x) if len(receivers) > 1 else 0.0
    curve = m0.curve
    fs, vs = np.asarray(curve.fs, dtype=float), np.asarray(curve.vs, dtype=float)
    wavelengths = vs / fs
    order = np.argsort(wavelengths)
    fs, vs, wavelengths = fs[order], vs[order], wavelengths[order]
    longest = thresholds.max_wavelength_lengths * length
    within = wavelengths <= longest

    # Wavelengths: below twice the spacing they are aliased (the picker starts above), beyond a
    # few window lengths the window does not resolve them: cut them.
    long_share = float(np.mean(~within))
    metrics.append(
        Metric(
            name="long_wavelengths",
            value=round(long_share, 3),
            threshold=0,
            bound="max",
            passed=long_share == 0,
        )
    )
    if long_share > 0:
        flags.append(
            Flag(
                name="long_wavelengths",
                message=f"{long_share:.0%} of the points lie beyond {thresholds.max_wavelength_lengths:g} "
                f"window lengths ({longest:.1f} m), which the window does not resolve: cut them.",
                stage="picking",
                action=Override(
                    stage="picking", overrides={"max_wavelength": thresholds.max_wavelength_lengths}
                ),
            )
        )
    aliased = float(np.mean(wavelengths < 2 * spacing)) if spacing else 0.0
    metrics.append(
        Metric(
            name="aliased_points",
            value=round(aliased, 3),
            threshold=0,
            bound="max",
            passed=aliased == 0,
        )
    )

    kept_vs, kept_fs, kept_wl = vs[within], fs[within], wavelengths[within]
    n_points = int(within.sum())
    metrics.append(
        Metric(
            name="points_within",
            value=n_points,
            threshold=thresholds.min_points,
            bound="min",
            passed=n_points >= thresholds.min_points,
        )
    )
    if n_points < thresholds.min_points:
        flags.append(
            Flag(
                name="too_few_points",
                message=f"{n_points} point{'s' if n_points != 1 else ''} within {longest:.1f} m: "
                "too few for a curve. Keep more of the ridge's points.",
                stage="picking",
                action=Override(
                    stage="picking",
                    overrides={
                        "min_relative_coherence": round(picking.min_relative_coherence * 0.6, 2)
                    },
                ),
            )
        )

    # A step between consecutive points of the resampled curve: the pick jumped onto another mode.
    jump = float(np.max(np.abs(np.diff(kept_vs)) / kept_vs[:-1])) if n_points > 1 else 0.0
    metrics.append(
        Metric(
            name="max_jump",
            value=round(jump, 3),
            threshold=thresholds.max_jump,
            bound="max",
            passed=jump <= thresholds.max_jump,
        )
    )
    if jump > thresholds.max_jump:
        flags.append(
            Flag(
                name="mode_jump",
                message=f"A {jump:.0%} step between consecutive points: the pick jumped onto "
                "another mode. Track with a narrower corridor.",
                stage="picking",
                action=Override(
                    stage="picking", overrides={"corridor": round(picking.corridor / 2, 3)}
                ),
            )
        )

    low, high = thresholds.air_wave_band
    air = float(np.mean((kept_vs >= low) & (kept_vs <= high))) if n_points else 0.0
    metrics.append(
        Metric(
            name="air_wave_share",
            value=round(air, 3),
            threshold=thresholds.max_air_share,
            bound="max",
            passed=air <= thresholds.max_air_share,
        )
    )
    if air > thresholds.max_air_share:
        flags.append(
            Flag(
                name="air_wave",
                message=f"{air:.0%} of the points sit at {low:g}-{high:g} m/s: the air wave, not "
                "the ground. Mute the arrivals faster than the surface wave.",
                stage="preprocessing",
                action=Override(
                    stage="preprocessing",
                    overrides={
                        "muting": {
                            "method": "mute",
                            "vmin": thresholds.mute_vmin,
                            "vmax": thresholds.air_wave_mute_vmax,
                        }
                    },
                ),
            )
        )

    # Normal dispersion: velocity rising with wavelength. The opposite is flagged, not rejected.
    trend = _spearman(kept_wl, kept_vs) if n_points >= 3 else float("nan")
    inverse = np.isfinite(trend) and trend < 0
    metrics.append(
        Metric(
            name="trend",
            value=None if not np.isfinite(trend) else round(trend, 3),
            threshold=0,
            bound="min",
            passed=not inverse,
        )
    )
    if inverse:
        flags.append(
            Flag(
                name="inverse_dispersion",
                message=f"Velocity falls with wavelength (rank correlation {trend:+.2f}): a stiff "
                "layer over a softer one, if the pick is right.",
                stage="picking",
                action=Keep(note="an inverse trend can be geology"),
            )
        )

    uncertainty = (
        float(np.median(np.asarray(curve.vs_err, dtype=float)[order][within] / kept_vs))
        if curve.vs_err is not None and n_points
        else None
    )
    uncertain = uncertainty is not None and uncertainty > thresholds.max_uncertainty
    metrics.append(
        Metric(
            name="uncertainty",
            value=None if uncertainty is None else round(uncertainty, 3),
            threshold=thresholds.max_uncertainty,
            bound="max",
            passed=not uncertain,
        )
    )
    if uncertain:
        flags.append(
            Flag(
                name="uncertain",
                message=f"The array resolves velocity to {uncertainty:.0%} at best: the inversion "
                "cannot use the curve. Longer windows, for the whole profile.",
                stage="phase_shift",
                action=Reject(reason="uncertainties beyond what the inversion can use"),
                fixable=False,
            )
        )

    # The near field (the spec's near-offset rule): reported, never applied (a decision of
    # milestone 13): keeping only far shots lost a third of the demo line's curves.
    if nearest_offset is not None and n_points:
        limit = thresholds.near_offset_wavelengths * float(kept_wl.max())
        near = nearest_offset < limit
        metrics.append(
            Metric(
                name="near_offset",
                value=round(nearest_offset, 2),
                threshold=round(limit, 2),
                bound="min",
                passed=not near,
                unit="m",
            )
        )
        if near:
            flags.append(
                Flag(
                    name="near_field",
                    message=f"The nearest shot is {nearest_offset:.2f} m from the window, under "
                    f"half the longest wavelength kept ({limit:.2f} m): the long wavelengths may "
                    "read slow (near field).",
                    stage="phase_shift",
                    action=Keep(
                        note="reported, not applied: far shots only lost a third of the "
                        "demo's curves"
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


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman's rank correlation; NaN when either input is constant."""
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


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


def _metrics(quality: ImageQuality, thresholds: CurveThresholds) -> list[Metric]:
    limits = thresholds.metrics
    return [
        Metric(
            name="sharpness",
            value=quality.sharpness,
            threshold=limits.min_sharpness,
            bound="min",
            passed="sharpness" not in quality.flags,
        ),
        Metric(
            name="prominence",
            value=quality.prominence,
            threshold=limits.min_prominence,
            bound="min",
            passed="prominence" not in quality.flags,
        ),
        Metric(
            name="on_data",
            value=quality.on_data,
            threshold=limits.min_on_data,
            bound="min",
            passed="on_data" not in quality.flags,
        ),
        Metric(
            name="constant_wavelength",
            value=quality.constant_wavelength,
            threshold=limits.max_constant_wavelength,
            bound="max",
            passed="constant_wavelength" not in quality.flags,
        ),
        Metric(
            name="n_points",
            value=quality.n_points,
            threshold=2,
            bound="min",
            passed=quality.n_points >= 2,
        ),
    ]


def _flag(
    name: str, quality: ImageQuality, band: tuple[float, float] | None, thresholds: CurveThresholds
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
            overrides = (
                {"filtering": {"method": "iir", "fmin": _low(band), "fmax": _high(band)}}
                if band
                else {"muting": {"method": "mute", "vmin": thresholds.mute_vmin, "vmax": 1500.0}}
            )
            return Flag(
                name="prominence",
                message=f"The ridge stands {quality.prominence:.1f} times above the rest of the "
                "image: barely. Filter to the band, or mute.",
                stage="preprocessing",
                action=Override(stage="preprocessing", overrides=overrides),
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
            return Flag(
                name="constant_wavelength",
                message=f"{quality.constant_wavelength:.0%} of the points follow the edge of what "
                "the window resolves, not a dispersion curve: cut the long wavelengths.",
                stage="picking",
                action=Override(
                    stage="picking", overrides={"max_wavelength": thresholds.max_wavelength_lengths}
                ),
            )


def _low(band: tuple[float, float]) -> float:
    return round(max(0.0, band[0]), 1)


def _high(band: tuple[float, float]) -> float:
    return round(band[1], 1)
