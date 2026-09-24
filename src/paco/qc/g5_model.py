"""G5, the model QC per window (docs/qc_workflow.md): the fit of the monitored smooth median to
the picked curve, by band of wavelength (a misfit only PAC's smoothing causes is kept: a
decision of milestone 13), whether the chains agree, whether the posterior piles at a bound of
the prior, and down to which depth the data inform the model (reported, never failed). The
cheapest fix first: sampling longer before widening a bound, both before another layer."""

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from paco.inversion import InversionParameters
from paco.inversion.measuring import InversionMeasures, ModelFit
from paco.inversion.models import SAMPLE_EVERY
from paco.qc.models import Flag, GateResult, Keep, Kept, Metric, Override, Reject

GATE = "G5"
BAND_NAMES = {3: ("short", "middle", "long")}
# How far a bound the posterior piles at moves out: the checks before S4's own margins.
WIDEN_LOW, WIDEN_HIGH = 0.8, 1.25


class ModelThresholds(BaseModel):
    """G5's limits: provisional, measured on the demo profiles (rule 9)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_misfit: float = Field(
        default=2.0,
        gt=0,
        description="RMS of the residuals over the curve's uncertainties, in any band, at most: "
        "about 1 is a fit within the errors.",
    )
    n_bands: int = Field(default=3, ge=1, description="Bands of wavelength the fit is judged in.")
    max_rhat: float = Field(
        default=1.1, gt=1, description="Split R-hat of any parameter, at most: the chains agree."
    )
    min_acceptance: float = Field(
        default=10.0, ge=0, description="Acceptance rate of every chain, at least (%)."
    )
    min_samples_per_chain: int = Field(
        default=100, ge=2, description="Models each chain keeps after the burn-in, at least."
    )
    bound_edge: float = Field(
        default=0.02,
        gt=0,
        lt=0.5,
        description="The edge of a prior's range watched at each bound, as a share of the range.",
    )
    max_at_bound: float = Field(
        default=0.1,
        gt=0,
        description="Share of a parameter's samples within the edge of a bound, at most: 5 times "
        "what a flat posterior puts there.",
    )
    useful_std_ratio: float = Field(
        default=0.5,
        gt=0,
        description="The useful depth ends where the spread of the sampled Vs reaches this share "
        "of the prior's.",
    )
    max_layers: int = Field(default=4, ge=2, description="Layers the loop may go up to.")


def judge_model(
    unit: str,
    measures: InversionMeasures,
    parameters: InversionParameters,
    thresholds: ModelThresholds,
    reach_m: float | None = None,
) -> GateResult:
    """G5's verdict on one window's inversion, measured with `thresholds`' bands, edge and
    ratio, from `parameters`. `reach_m` is how deep the half-space's top may go (the checks
    before S4): a thickness piled at a maximum shallower than that is widened."""
    smooth, layered = measures.fits
    names = BAND_NAMES.get(
        len(smooth.bands), tuple(f"band{i + 1}" for i in range(len(smooth.bands)))
    )
    metrics = [
        Metric(
            name=f"misfit_{name}",
            value=band.misfit,
            threshold=thresholds.max_misfit,
            bound="max",
            passed=band.misfit is not None and band.misfit <= thresholds.max_misfit,
        )
        for name, band in zip(names, smooth.bands, strict=True)
    ]
    metrics.append(
        Metric(
            name="misfit_layered",
            value=layered.misfit,
            threshold=thresholds.max_misfit,
            bound="max",
            passed=_fits(layered, thresholds),
        )
    )
    rhats = [value for value in measures.rhat.values() if value is not None]
    rhat = max(rhats) if rhats else None
    acceptance = min(measures.acceptance) if measures.acceptance else None
    piled = measures.at_bounds[0] if measures.at_bounds else None
    converged = (
        (rhat is not None and rhat <= thresholds.max_rhat)
        and (acceptance is not None and acceptance >= thresholds.min_acceptance)
        and measures.samples_per_chain >= thresholds.min_samples_per_chain
    )
    metrics += [
        Metric(
            name="rhat",
            value=rhat,
            threshold=thresholds.max_rhat,
            bound="max",
            passed=rhat is not None and rhat <= thresholds.max_rhat,
        ),
        Metric(
            name="acceptance",
            value=acceptance,
            threshold=thresholds.min_acceptance,
            bound="min",
            passed=acceptance is not None and acceptance >= thresholds.min_acceptance,
            unit="%",
        ),
        Metric(
            name="samples_per_chain",
            value=measures.samples_per_chain,
            threshold=thresholds.min_samples_per_chain,
            bound="min",
            passed=measures.samples_per_chain >= thresholds.min_samples_per_chain,
        ),
        Metric(
            name="at_bound",
            value=piled.share if piled else None,
            threshold=thresholds.max_at_bound,
            bound="max",
            passed=piled is None or piled.share <= thresholds.max_at_bound,
        ),
        Metric(name="useful_depth", value=measures.useful_depth_m, passed=True, unit="m"),
    ]

    flags: list[Flag] = []
    if layered.n_missing:
        flags.append(
            Flag(
                name="no_mode",
                message=f"The layered median has no fundamental mode at {layered.n_missing} of the "
                "picked points: they are faster than any mode of a model with a softer layer "
                "below. A higher mode may be picked there.",
                stage="picking",
                action=Reject(
                    reason="the curve holds points no fundamental mode of the model reaches"
                ),
                fixable=False,
            )
        )
    if not converged:
        iterations = 2 * parameters.n_iterations
        if measures.samples_per_chain < thresholds.min_samples_per_chain:
            # Enough models a chain at once (sigpipe keeps one every SAMPLE_EVERY iterations after
            # a burn-in of a tenth): doubling 2,000 iterations twice still left too few.
            enough = thresholds.min_samples_per_chain * SAMPLE_EVERY / 0.9
            iterations = max(iterations, math.ceil(enough / 1_000) * 1_000)
        flags.append(
            Flag(
                name="not_converged",
                message=f"The chains do not agree (R-hat {_value(rhat)}, acceptance "
                f"{_value(acceptance)} %, {measures.samples_per_chain} models a chain): sample "
                f"longer, {iterations} iterations.",
                stage="inversion",
                action=Override(
                    stage="inversion",
                    overrides={
                        "n_iterations": iterations,
                        "n_burnin_iterations": iterations // 10,
                    },
                ),
            )
        )
    widened = _widened_vs(measures, parameters, thresholds)
    if widened is not None:
        vs_layers, piled_names = widened
        flags.append(
            Flag(
                name="at_bound",
                message=f"The posterior piles at the prior's bound ({piled_names}): widen it.",
                stage="inversion",
                action=Override(stage="inversion", overrides={"vs_layers": vs_layers}),
            )
        )
    flags += _thickness_flags(measures, parameters, thresholds, reach_m)
    worst = _worst_band(smooth, names)
    if converged and widened is None and worst is not None and worst[1] > thresholds.max_misfit:
        name, value, band = worst
        where = f"{name} wavelengths ({band[0]:g}-{band[1]:g} m)"
        if _fits(layered, thresholds):
            flags.append(
                Flag(
                    name="smoothing_misfit",
                    message=f"The smooth median misfits {value:.1f} at {where} where the layered "
                    f"median fits ({layered.misfit:.1f}): PAC's smoothing spreads the layer "
                    "boundaries.",
                    stage="inversion",
                    action=Keep(note="re-inverting cannot fix a smoothing effect"),
                )
            )
        elif parameters.n_layers < thresholds.max_layers:
            flags.append(
                Flag(
                    name="underfit",
                    message=f"The model misfits {value:.1f} at {where}: try one more layer.",
                    stage="inversion",
                    action=Override(
                        stage="inversion", overrides={"n_layers": parameters.n_layers + 1}
                    ),
                )
            )
        else:
            flags.append(
                Flag(
                    name="underfit",
                    message=f"The model misfits {value:.1f} at {where} with "
                    f"{parameters.n_layers} layers, the most the loop tries.",
                    stage="inversion",
                    action=Reject(reason=f"not fitted with up to {thresholds.max_layers} layers"),
                    fixable=False,
                )
            )

    kinds = {type(flag.action) for flag in flags}
    verdict = "reject" if Reject in kinds else "retry" if Override in kinds else "pass"
    wavelengths = [band.wavelength_m for band in smooth.bands]
    return GateResult(
        gate=GATE,
        unit=unit,
        verdict=verdict,
        metrics=tuple(metrics),
        flags=tuple(flags),
        kept=Kept(
            wavelength_m=(wavelengths[0][0], wavelengths[-1][1]) if wavelengths else None,
            n_points=sum(band.n_points for band in smooth.bands),
        ),
    )


def _fits(fit: ModelFit, thresholds: ModelThresholds) -> bool:
    """Whether a model fits every point within the limit: it has a mode at each, and no band
    misfits beyond it."""
    return (
        fit.n_missing == 0
        and fit.misfit is not None
        and all(
            band.misfit is not None and band.misfit <= thresholds.max_misfit for band in fit.bands
        )
    )


def _worst_band(
    fit: ModelFit, names: tuple[str, ...]
) -> tuple[str, float, tuple[float, float]] | None:
    """The band the model fits worst: its name, misfit and wavelengths; a band without a mode
    counts as infinitely misfit."""
    scored = [
        (name, band.misfit if band.misfit is not None else float("inf"), band.wavelength_m)
        for name, band in zip(names, fit.bands, strict=True)
    ]
    return max(scored, key=lambda item: item[1]) if scored else None


def _widened_vs(
    measures: InversionMeasures, parameters: InversionParameters, thresholds: ModelThresholds
) -> tuple[list[dict[str, Any]], str] | None:
    """Every layer's Vs bounds, each bound the posterior piles at moved out (the lowest by
    WIDEN_LOW, the highest by WIDEN_HIGH), and the bounds named; None when no Vs bound holds too
    many samples."""
    layers = [layer.model_dump() for layer in parameters.vs_layers]
    named: list[str] = []
    for share in measures.at_bounds:
        if share.share <= thresholds.max_at_bound or not share.parameter.startswith("vs"):
            continue
        layer = layers[int(share.parameter[2:]) - 1]
        if share.bound == "min":
            layer["vs_min"] = round(layer["vs_min"] * WIDEN_LOW)
        else:
            layer["vs_max"] = round(layer["vs_max"] * WIDEN_HIGH)
        named.append(f"{share.parameter} {share.bound} {share.value:g} m/s, {share.share:.0%}")
    return (layers, "; ".join(named)) if named else None


def _thickness_flags(
    measures: InversionMeasures,
    parameters: InversionParameters,
    thresholds: ModelThresholds,
    reach_m: float | None,
) -> list[Flag]:
    """A layer piled at its thinnest is not resolved: one layer fewer, when there are more than
    two. A layer piled at its thickest is given room while the half-space's top stays within the
    curve's reach (`reach_m`); at the reach, the limit stays and the flag says so."""
    flags: list[Flag] = []
    deepest = sum(layer.thickness_max for layer in parameters.thickness_layers)
    for share in measures.at_bounds:
        if share.share <= thresholds.max_at_bound or not share.parameter.startswith("thick"):
            continue
        if share.bound == "min" and parameters.n_layers > 2:
            flags.append(
                Flag(
                    name="thin_layer",
                    message=f"{share.share:.0%} of {share.parameter}'s samples sit at its thinnest "
                    f"({share.value:g} m): the layer is not resolved.",
                    stage="inversion",
                    action=Override(
                        stage="inversion", overrides={"n_layers": parameters.n_layers - 1}
                    ),
                )
            )
        elif share.bound == "max" and reach_m is not None and deepest < 0.99 * reach_m:
            layers = [layer.model_dump() for layer in parameters.thickness_layers]
            layer = layers[int(share.parameter[5:]) - 1]
            layer["thickness_max"] = round(layer["thickness_max"] * WIDEN_HIGH, 2)
            flags.append(
                Flag(
                    name="at_bound",
                    message=f"{share.share:.0%} of {share.parameter}'s samples sit at its thickest "
                    f"({share.value:g} m), shallower than the curve reaches ({reach_m:g} m): "
                    "widen it.",
                    stage="inversion",
                    action=Override(stage="inversion", overrides={"thickness_layers": layers}),
                )
            )
        elif share.bound == "max":
            flags.append(
                Flag(
                    name="deep_interface",
                    message=f"{share.share:.0%} of {share.parameter}'s samples sit at its thickest "
                    f"({share.value:g} m): the interface would go deeper than the curve resolves.",
                    stage="inversion",
                    action=Keep(note="the depth limit stays: the curve does not reach deeper"),
                )
            )
    return flags


def _value(value: float | None) -> str:
    return "n/a" if value is None else f"{value:g}"
