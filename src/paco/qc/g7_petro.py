"""G7, the petrophysical model QC per window (the user's decision of 2026-09-26: range, fit,
line): the curve the Silex model's soil column gives back against the pick, by band of
wavelength as G5 judges the seismic models, with G5's limit. No retry: a model predicts one
soil column per curve. A window G7 rejects is left out of the petrophysical sections; its flag
says what could change it: another model covering the curve, or the pick."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from paco.qc.models import Flag, GateResult, Kept, Metric, Reject

if TYPE_CHECKING:
    # Its module runs santiludo's rock physics: an extra (paco.qc.petro).
    from sigpipe.masw.petro.measuring import PetroMeasures

GATE = "G7"
BAND_NAMES = {3: ("short", "middle", "long")}


class PetroThresholds(BaseModel):
    """G7's limits: G5's for now, provisional (rule 9)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_misfit: float = Field(
        default=2.0,
        gt=0,
        description="RMS of the residuals over the curve's uncertainties, in any band, at most: "
        "about 1 is a fit within the errors.",
    )
    n_bands: int = Field(default=3, ge=1, description="Bands of wavelength the fit is judged in.")


def judge_petro(
    unit: str,
    measures: PetroMeasures,
    thresholds: PetroThresholds,
    other_models: Sequence[str] = (),
) -> GateResult:
    """G7's verdict on one window's petrophysical inversion; `other_models` are the other Silex
    models covering the window's curve."""
    fit = measures.fit
    names = BAND_NAMES.get(len(fit.bands), tuple(f"band{i + 1}" for i in range(len(fit.bands))))
    metrics = [
        Metric(
            name=f"misfit_{name}",
            value=band.misfit,
            threshold=thresholds.max_misfit,
            bound="max",
            # A band with no point to weigh says nothing: judged by the others.
            passed=band.misfit is None or band.misfit <= thresholds.max_misfit,
        )
        for name, band in zip(names, fit.bands, strict=True)
    ]
    metrics += [
        Metric(
            name=f"residual_{name}",
            value=None if band.residual is None else round(100 * band.residual, 1),
            passed=True,
            unit="%",
        )
        for name, band in zip(names, fit.bands, strict=True)
    ]
    metrics.append(Metric(name="water_table", value=measures.water_table_m, passed=True, unit="m"))

    instead = (
        f"invert_petro with {', '.join(other_models)}, which cover{'s' if len(other_models) == 1 else ''} this curve"
        if other_models
        else "no other model covers this curve; check its pick in PAC"
    )
    flags: list[Flag] = []
    if fit.n_missing:
        flags.append(
            Flag(
                name="no_mode",
                message=f"The soil column's curve has no fundamental mode at {fit.n_missing} of "
                f"the picked points: {instead}.",
                stage="petro_inversion",
                action=Reject(reason="the soil column gives no fundamental mode at picked points"),
                fixable=bool(other_models),
            )
        )
    measured = [
        (name, band, band.misfit)
        for name, band in zip(names, fit.bands, strict=True)
        if band.misfit is not None
    ]
    worst = max(measured, key=lambda found: found[2], default=None)
    if worst is not None and worst[2] > thresholds.max_misfit:
        name, band, _ = worst
        flags.append(
            Flag(
                name="misfit",
                message=f"The soil column's curve misfits the pick by {band.misfit:.1f} at {name} "
                f"wavelengths ({band.wavelength_m[0]:g}-{band.wavelength_m[1]:g} m): {instead}.",
                stage="petro_inversion",
                action=Reject(reason="the soil column does not give the picked curve back"),
                fixable=bool(other_models),
            )
        )
    if fit.misfit is None and not fit.n_missing:
        flags.append(
            Flag(
                name="no_uncertainty",
                message="No picked point carries an uncertainty: the fit cannot be weighed. "
                "Check the pick in PAC.",
                stage="picking",
                action=Reject(reason="no picked point to weigh the fit with"),
                fixable=False,
            )
        )
    return GateResult(
        gate=GATE,
        unit=unit,
        verdict="reject" if flags else "pass",
        metrics=tuple(metrics),
        flags=tuple(flags),
        kept=Kept(n_points=sum(band.n_points for band in fit.bands)),
    )
