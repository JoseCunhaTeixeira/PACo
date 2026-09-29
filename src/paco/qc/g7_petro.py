"""G7, the petrophysical model QC per window (docs/qc_workflow.md): the curve the Silex model's
soil column gives back against the pick, by band of wavelength as G5 judges the seismic models,
with G5's limit. No retry: a model predicts one soil column per curve. A window G7 rejects is
left out of the petrophysical sections; its flag says what could change it: another model
covering the curve, or the pick."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic import ConfigDict
from sigpipe.masw.quality.model import BAND_NAMES
from sigpipe.masw.quality.soil import SoilLimits, measure_soil

from paco.qc.models import Flag, GateResult, Kept, Metric, Reject

if TYPE_CHECKING:
    # Its module runs santiludo's rock physics: an extra (paco.qc.petro).
    from sigpipe.masw.petro.measuring import PetroMeasures

GATE = "G7"


class PetroThresholds(SoilLimits):
    """G7's limits (sigpipe's SoilLimits, PAC's alike): G5's, provisional (rule 9)."""

    model_config = ConfigDict(frozen=True, extra="forbid")


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
    # The soil column's measures, sigpipe's (PAC's alike), each saying what it covers.
    metrics = [
        Metric(**one.model_dump()) for one in measure_soil(fit, measures.water_table_m, thresholds)
    ]

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
