"""G2, the QC of a dispersion image before picking (docs/qc_workflow.md): coherent energy
against the noise floor, the energy maximum on the grid's edges, competing ridges (a higher mode
or aliasing), and a coherent band much narrower than the record's usable band. The measures are
sigpipe's (sigpipe.masw.quality.image); G2 judges them."""

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sigpipe.base import DispersionImage
from sigpipe.masw.quality.image import aliased, coherent_columns, competing_ridges, edge_peaks

from paco.qc.models import Flag, GateResult, Keep, Kept, Metric, Override

GATE = "G2"


class ImageThresholds(BaseModel):
    """G2's limits: provisional, measured on the demo profiles (rule 9)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    coherent_level: float = Field(
        default=0.3,
        gt=0,
        lt=1,
        description="A column is coherent when its peak is this far from the noise floor\n"
        "1 / sqrt(N) towards 1, the most a plane wave gives: the same for every N.",
    )
    min_coherent_columns: float = Field(
        default=0.5, ge=0, le=1, description="Share of the image's columns that must be coherent."
    )
    edge_share: float = Field(
        default=0.02,
        gt=0,
        lt=0.5,
        description="Share of the velocity range that counts as an edge.",
    )
    max_edge_columns: float = Field(
        default=0.2, ge=0, le=1, description="Share of coherent columns peaking on an edge."
    )
    competing_ratio: float = Field(
        default=0.7, gt=0, lt=1, description="A second ridge at least this share of the first."
    )
    competing_separation: float = Field(
        default=0.15, gt=0, description="Its velocity at least this share away from the first's."
    )
    max_competing_columns: float = Field(
        default=0.7,
        ge=0,
        le=1,
        description="Share of coherent columns with a second ridge: good 24-receiver windows of the\n"
        "demo hold one in 52 to 60 % of theirs, 5-receiver ones in 87 to 95 %.",
    )
    min_band_share: float = Field(
        default=0.5, gt=0, le=1, description="Coherent band over the usable band, at least."
    )
    vmin_floor: float = Field(
        default=30.0,
        gt=0,
        description="m/s: energy peaking at a vmin below this is an artifact, not a slower wave.",
    )
    vg_min: float = Field(default=80.0, gt=0, description="m/s, for the mute a retry suggests")
    vg_max: float = Field(default=1500.0, gt=0, description="m/s, for the mute a retry suggests")


def judge_image(
    unit: str,
    image: DispersionImage,
    thresholds: ImageThresholds,
    usable_band: tuple[float, float] | None = None,
) -> GateResult:
    """G2's verdict on one window's image, with G1's usable band of its records when known."""
    fs = np.asarray(image.fs, dtype=float)
    vs = np.asarray(image.vs, dtype=float)
    coherent = coherent_columns(image, thresholds.coherent_level)
    n_coherent = int(coherent.sum())
    share = n_coherent / max(coherent.size, 1)
    metrics = [
        Metric(
            name="coherent_columns",
            value=round(share, 3),
            threshold=thresholds.min_coherent_columns,
            bound="min",
            passed=share >= thresholds.min_coherent_columns,
        )
    ]
    flags: list[Flag] = []
    mute = Override(
        stage="preprocessing",
        overrides={
            "muting": {"method": "mute", "vmin": thresholds.vg_min, "vmax": thresholds.vg_max}
        },
    )
    if n_coherent == 0:
        flags.append(
            Flag(
                name="no_coherent_energy",
                message=f"No column of the image rises {thresholds.coherent_level:.0%} of the way "
                "from the noise floor to 1: nothing to pick. Try a surface-wave mute.",
                stage="preprocessing",
                action=mute,
            )
        )
        return _result(
            unit, metrics, flags, Kept(band_hz=None, n_traces=len(image.acquisition.receivers))
        )

    band = (float(fs[coherent].min()), float(fs[coherent].max()))
    if share < thresholds.min_coherent_columns:
        flags.append(
            Flag(
                name="weak_coherence",
                message=f"Only {share:.0%} of the columns are coherent ({band[0]:.1f}-{band[1]:.1f} "
                "Hz): the image is mostly noise. Try a surface-wave mute.",
                stage="preprocessing",
                action=mute,
            )
        )

    low, high = edge_peaks(image, coherent, thresholds.edge_share, thresholds.vmin_floor)
    # A peak at a vmin below the floor is an artifact (a correlation's zero lag, a mute's edge):
    # start the range above it; at a real vmin, lower it.
    lower_vmin = (
        {"vmin": thresholds.vmin_floor}
        if vs.min() < thresholds.vmin_floor
        else {"vmin": max(1.0, round(vs.min() / 1.5, 1))}
    )
    for name, count, edge, override in (
        ("ridge_at_vmin", low, vs.min(), lower_vmin),
        ("ridge_at_vmax", high, vs.max(), {"vmax": round(vs.max() * 1.5, 1)}),
    ):
        edge_share = count / n_coherent
        metrics.append(
            Metric(
                name=name,
                value=round(edge_share, 3),
                threshold=thresholds.max_edge_columns,
                bound="max",
                passed=edge_share <= thresholds.max_edge_columns,
            )
        )
        if edge_share > thresholds.max_edge_columns:
            flags.append(
                Flag(
                    name=name,
                    message=f"{edge_share:.0%} of the coherent columns peak at {edge:g} m/s, the "
                    "grid's edge: "
                    + (
                        "an artifact below any surface wave; start the range above it."
                        if name == "ridge_at_vmin" and vs.min() < thresholds.vmin_floor
                        else "the velocity range is too narrow."
                    ),
                    stage="phase_shift",
                    action=Override(stage="phase_shift", overrides={"dispersion": override}),
                )
            )

    rows = np.flatnonzero(coherent)
    at_fmin, at_fmax = rows[0] == 0, rows[-1] == fs.size - 1
    # A band reaching fmax is kept, not widened: a wider band lets a second ridge compete, and
    # fewer curves pass G3. The coherence rules cap fmax at the records' usable band.
    for name, touches, action in (
        # Kept too: the picker stops where its ridge breaks or the window resolves no velocity.
        (
            "band_at_fmin",
            at_fmin,
            Keep(note="fmin stays: the picker stops where the ridge breaks"),
        ),
        ("band_at_fmax", at_fmax, Keep(note="fmax stays: a wider band let other ridges compete")),
    ):
        metrics.append(
            Metric(name=name, value=float(touches), threshold=0, bound="max", passed=not touches)
        )
        if touches:
            where = "lowest" if name == "band_at_fmin" else "highest"
            flags.append(
                Flag(
                    name=name,
                    message=f"The coherent band reaches the image's {where} frequency: the band "
                    "may go on beyond it.",
                    stage="phase_shift",
                    action=action,
                )
            )

    competing = competing_ridges(
        image,
        coherent,
        thresholds.competing_ratio,
        thresholds.competing_separation,
        thresholds.vmin_floor,
    )
    competing_share = int(competing.sum()) / n_coherent
    metrics.append(
        Metric(
            name="competing_ridges",
            value=round(competing_share, 3),
            threshold=thresholds.max_competing_columns,
            bound="max",
            passed=competing_share <= thresholds.max_competing_columns,
        )
    )
    # On a grid too narrow, a truncated ridge makes second ridges and aliases of its own: the
    # velocity range first.
    grid_first = any(flag.name in ("ridge_at_vmin", "ridge_at_vmax") for flag in flags)
    if competing_share > thresholds.max_competing_columns and not grid_first:
        alias = aliased(image, competing, thresholds.vmin_floor)
        if alias.sum() > competing.sum() / 2:
            first = float(fs[alias][0])
            flags.append(
                Flag(
                    name="aliasing",
                    message=f"{competing_share:.0%} of the coherent columns hold a second ridge, "
                    f"mostly below the aliasing limit 2 dx f: an alias from {first:.1f} Hz on.",
                    stage="phase_shift",
                    action=Override(
                        stage="phase_shift", overrides={"dispersion": {"fmax": round(first, 1)}}
                    ),
                )
            )
        else:
            flags.append(
                Flag(
                    name="competing_ridges",
                    message=f"{competing_share:.0%} of the coherent columns hold a second ridge of "
                    "similar strength: a higher mode may dominate. Pick two modes.",
                    stage="picking",
                    action=Override(stage="picking", overrides={"max_modes": 2}),
                )
            )

    if usable_band is not None:
        usable_width = usable_band[1] - usable_band[0]
        band_share = (band[1] - band[0]) / usable_width if usable_width > 0 else 1.0
        metrics.append(
            Metric(
                name="band_share_of_usable",
                value=round(band_share, 3),
                threshold=thresholds.min_band_share,
                bound="min",
                passed=band_share >= thresholds.min_band_share,
            )
        )
        if band_share < thresholds.min_band_share:
            flags.append(
                Flag(
                    name="narrower_than_usable",
                    message=f"The coherent band {band[0]:.1f}-{band[1]:.1f} Hz is {band_share:.0%} "
                    f"of the records' usable band {usable_band[0]:.1f}-{usable_band[1]:.1f} Hz: "
                    "the image does not use all the data offers.",
                    stage="phase_shift",
                    action=Keep(note="the band is capped, never widened: see band_at_fmax"),
                )
            )

    return _result(
        unit, metrics, flags, Kept(band_hz=band, n_traces=len(image.acquisition.receivers))
    )


def _result(unit: str, metrics: list[Metric], flags: list[Flag], kept: Kept) -> GateResult:
    # A kept flag is information: it never changes the verdict.
    verdict = "retry" if any(not isinstance(flag.action, Keep) for flag in flags) else "pass"
    return GateResult(
        gate=GATE, unit=unit, verdict=verdict, metrics=tuple(metrics), flags=tuple(flags), kept=kept
    )
