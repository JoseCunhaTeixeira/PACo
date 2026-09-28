"""G5, the model QC per window (docs/qc_workflow.md): the fit of the monitored model (the
ensemble: the kept models' median Vs at each depth) to the picked curve, by band of wavelength
(a misfit only that median across the models causes is kept), whether
the chains agree on the models' Vs at the depths the curve resolves, whether the posterior piles
at a bound of the prior, down to which depth the data inform the model, and, when the layers are
given, whether two adjacent layers are one. The cheapest fix first: sampling longer, a bound
widened (more layers allowed, when the data choose them) or the depth shrunk to what the data
inform, before a layer more or fewer. A retry's changes are one set of parameters, whichever
flags ask them."""

import itertools
import math
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from sigpipe.algorithms.inversion.rayleigh.seismic.parameters import SAVE_EVERY
from sigpipe.masw.inversion import InversionParameters
from sigpipe.masw.inversion.measuring import InversionMeasures, ModelFit
from sigpipe.masw.inversion.priors import MIN_LAYERS, THICKNESS_STEP_SHARE, VS_STEP_SHARE

from paco.qc.models import Flag, GateResult, Keep, Kept, Metric, Override, Reject

GATE = "G5"
BAND_NAMES = {3: ("short", "middle", "long")}
# How far a bound the posterior piles at moves out: the checks before S4's own margins.
WIDEN_LOW, WIDEN_HIGH = 0.8, 1.25
# Layers more the data may choose, where their models pile at the most allowed.
MORE_LAYERS = 2
# The share of a chain kept after the burn-in: a quarter of the iterations burn in.
KEPT_SHARE = 0.75
# A range narrowed to its samples: their 5th to 95th percentiles, this share of that span added
# on each side (the span widened by a fifth), and at least this share of the range it had.
NARROW_MARGIN = 0.1
NARROW_LEAST = 0.1


class ModelThresholds(BaseModel):
    """G5's limits: provisional, measured on the demo profiles (rule 9)."""

    # The limits of runs saved before are read too: acceptance's, no longer judged (the chains'
    # moves follow the posterior; they need no step tuned to a share of their proposals).
    model_config = ConfigDict(frozen=True, extra="ignore")

    max_misfit: float = Field(
        default=2.0,
        gt=0,
        description="RMS of the residuals over the curve's uncertainties, in any band, at most: "
        "about 1 is a fit within the errors.",
    )
    n_bands: int = Field(default=3, ge=1, description="Bands of wavelength the fit is judged in.")
    max_rhat: float = Field(
        default=1.1,
        gt=1,
        description="Split R-hat of the models' Vs at any depth watched, at most: the chains "
        "agree.",
    )
    min_samples_per_chain: int = Field(
        default=100, ge=2, description="Models each chain keeps after the burn-in, at least."
    )
    min_ess: float = Field(
        default=200.0,
        gt=0,
        description="Effective samples of the models' Vs at any depth watched, over the chains, "
        "at least: a median and 5 to 95 % band to a few percent (a Vs that jumps between two "
        "values where an interface may be above or below holds fewer than a layer's value).",
    )
    max_vs_drop: float = Field(
        default=0.5,
        gt=0,
        lt=1,
        description="A layer of the layered median under this share of the Vs above it is "
        "reported (a strong low-velocity layer), never failed.",
    )
    plausible_vs: tuple[float, float] = Field(
        default=(50.0, 2_500.0),
        description="m/s: a layered median's Vs outside is reported, never failed.",
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
    min_useful_share: float = Field(
        default=0.8,
        gt=0,
        le=1,
        description="The useful depth, at least this share of the depth the half-space's top may "
        "reach: shallower, the model is shrunk to what the data inform.",
    )
    min_contrast: float = Field(
        default=0.05,
        ge=0,
        description="Adjacent layers of the layered median whose Vs differ less, relative to "
        "their mean, are one: a model that fits loses a layer.",
    )
    max_longer_runs: int = Field(
        default=2,
        ge=0,
        description="Times a window's chains are sampled longer when they disagree; after, the "
        "model is kept with a warning (several modes: the median over the chains).",
    )
    max_layers: int = Field(
        default=10,
        ge=3,
        description="Layers the loop may go up to, at most (given, or allowed the data): below "
        "it, what each curve resolves.",
    )


def judge_model(
    unit: str,
    measures: InversionMeasures,
    parameters: InversionParameters,
    thresholds: ModelThresholds,
    reach_m: float | None = None,
    resolved_layers: int | None = None,
    fewest_layers: int = MIN_LAYERS,
    longer_runs: int = 0,
    narrowed: bool = False,
) -> GateResult:
    """G5's verdict on one window's inversion, measured with `thresholds`' bands, edge and
    ratio, from `parameters`. `reach_m` is how deep the half-space's top may go (the checks
    before S4): a thickness piled at a maximum shallower than that is widened. `resolved_layers`
    is the most layers the curve resolves; `fewest_layers` the fewest the loop goes down to, one
    more than a count that misfit before; `longer_runs` the times the window was sampled longer
    already."""
    monitored, layered = measures.fits
    names = BAND_NAMES.get(
        len(monitored.bands), tuple(f"band{i + 1}" for i in range(len(monitored.bands)))
    )
    metrics = [
        Metric(
            name=f"misfit_{name}",
            value=band.misfit,
            threshold=thresholds.max_misfit,
            bound="max",
            passed=band.misfit is not None and band.misfit <= thresholds.max_misfit,
        )
        for name, band in zip(names, monitored.bands, strict=True)
    ]
    # PAC's residual, (modelled - picked) / modelled in %, by band: reported, never judged (the
    # misfit divides by the Lorentzian uncertainties).
    metrics += [
        Metric(
            name=f"residual_{name}",
            value=None if band.residual is None else round(100 * band.residual, 1),
            passed=True,
            unit="%",
        )
        for name, band in zip(names, monitored.bands, strict=True)
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
    judged = _judged(measures)
    rhats = [value for name, value in measures.rhat.items() if value is not None and name in judged]
    rhat = max(rhats) if rhats else None
    piled = measures.at_bounds[0] if measures.at_bounds else None
    esses = [value for name, value in measures.ess.items() if value is not None and name in judged]
    ess = min(esses) if esses else None
    correlations = [
        value
        for name, value in measures.autocorrelation.items()
        if value is not None and name in judged
    ]
    free = parameters.layering == "free"
    # The chains agree on one posterior: what it says of the data can be judged. Converged,
    # they also hold enough independent samples for its uncertainties.
    agree = (
        rhat is not None
        and rhat <= thresholds.max_rhat
        and measures.samples_per_chain >= thresholds.min_samples_per_chain
    )
    converged = agree and (ess is None or ess >= thresholds.min_ess)
    depth = model_depth(parameters)
    useful = measures.useful_depth_m
    informed = useful is None or useful >= thresholds.min_useful_share * depth
    contrast = _least_contrast(measures.vs_layers)
    metrics += [
        Metric(
            name="rhat",
            value=rhat,
            threshold=thresholds.max_rhat,
            bound="max",
            passed=rhat is not None and rhat <= thresholds.max_rhat,
        ),
        Metric(
            name="ess",
            value=ess,
            threshold=thresholds.min_ess,
            bound="min",
            passed=ess is None or ess >= thresholds.min_ess,
        ),
        Metric(
            name="autocorrelation",
            value=max(correlations) if correlations else None,
            passed=True,
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
        # When the data choose the layers, the deepest allowed is the curve's reach whatever the
        # data inform: reported, the model's depth is not shrunk.
        Metric(
            name="useful_depth",
            value=useful,
            threshold=round(thresholds.min_useful_share * depth, 2),
            bound="min",
            passed=informed or free,
            unit="m",
        ),
        Metric(
            name="contrast",
            value=None if contrast is None else round(100 * contrast[0], 1),
            threshold=round(100 * thresholds.min_contrast, 1),
            bound="min",
            passed=contrast is None or contrast[0] >= thresholds.min_contrast,
            unit="%",
        ),
    ]

    loop = _Loop(parameters)
    if layered.n_missing:
        loop.flags.append(_no_mode(layered))
    if not converged and longer_runs >= thresholds.max_longer_runs:
        said = (
            f"after sampling {longer_runs} times longer (R-hat {_value(rhat)}, {_value(ess)} "
            "effective samples)"
        )
        loop.flags.append(
            Flag(
                name="few_samples",
                message=f"The chains agree but hold few independent samples {said}: the "
                "model stands, its uncertainties are rough.",
                stage="inversion",
                action=Keep(note="kept with the warning, as the user chose"),
            )
            if agree
            else Flag(
                name="multimodal",
                message=f"The chains still disagree {said}: the posterior holds several "
                "modes, and the median over the chains is less reliable.",
                stage="inversion",
                action=Keep(note="kept with the warning, as the user chose"),
            )
        )
    elif (
        not converged
        and not free
        and not narrowed
        and measures.samples_per_chain >= thresholds.min_samples_per_chain
        and (ranges := loop.narrow(measures.quantiles))
    ):
        # Chains that wander a wide prior are given the part of it their samples found first
        # (with enough samples a chain to say where they lie). The depth the data inform is still
        # judged against the first, wide prior.
        loop.change(
            "not_converged",
            f"The chains do not agree or hold too few independent samples (R-hat "
            f"{_value(rhat)}, {_value(ess)} effective): narrow each range to where its samples "
            f"lie ({ranges}) before sampling longer.",
            "vs_layers",
            "thickness_layers",
        )
    elif not converged:
        iterations = 2 * parameters.n_iterations
        if measures.samples_per_chain < thresholds.min_samples_per_chain:
            # Enough models a chain at once (sigpipe keeps one every SAVE_EVERY iterations after
            # a burn-in of a quarter): doubling 2,000 iterations twice still leaves too few.
            enough = thresholds.min_samples_per_chain * SAVE_EVERY / KEPT_SHARE
            iterations = max(iterations, math.ceil(enough / 1_000) * 1_000)
        loop.change(
            "not_converged",
            f"The chains do not agree or hold too few independent samples (R-hat "
            f"{_value(rhat)}, {measures.samples_per_chain} models a chain, {_value(ess)} "
            f"effective): sample longer, {iterations} iterations.",
            n_iterations=iterations,
            n_burnin_iterations=iterations // 4,
        )
    widened = loop.widen_vs(measures, thresholds)
    if widened:
        loop.change(
            "at_bound",
            f"The posterior piles at the prior's bound ({widened}): widen it.",
            "free" if free else "vs_layers",
        )
    more = False
    if free:
        more = _free_flags(loop, measures, thresholds)
    else:
        _thickness_flags(loop, measures, thresholds, reach_m, informed, fewest_layers)
    tmin = (
        parameters.free.depth_min or 0.0
        if free
        else max(layer.thickness_min for layer in parameters.thickness_layers)
    )
    # What the posterior says is judged once the chains agree on it.
    if useful == 0 and agree:
        loop.flags.append(
            Flag(
                name="uninformed",
                message="The curve informs no depth of the model: the spread of the sampled Vs "
                "is at least half the prior's everywhere, the model is the prior's. The picks' "
                "uncertainties are too large for it: longer windows (run_processing again) give "
                "smaller ones.",
                stage="phase_shift",
                action=Reject(reason="the curve constrains no depth of the model"),
                fixable=False,
            )
        )
    shrunk = (
        agree
        and not free
        and useful is not None
        and useful > 0
        and not informed
        and _too_deep(loop, useful, depth, tmin, thresholds, fewest_layers)
    )
    # One layer more, up to what the curve resolves, never past the loop's own limit.
    ceiling = (
        thresholds.max_layers
        if free
        else min(
            thresholds.max_layers,
            resolved_layers or thresholds.max_layers,
            _layers_within(depth, tmin),
        )
    )
    # The layers are counted once the chains agree, no Vs bound holds the posterior and the
    # model ends where the data inform it.
    settled = agree and not widened and not shrunk and not more
    worst = _worst_band(monitored, names)
    if settled and worst is not None and worst[1] > thresholds.max_misfit:
        name, value, band = worst
        where = f"{name} wavelengths ({band[0]:g}-{band[1]:g} m)"
        if _fits(layered, thresholds):
            loop.flags.append(
                Flag(
                    name="smoothing_misfit",
                    message=f"The ensemble misfits {value:.1f} at {where} where the layered "
                    f"median fits ({layered.misfit:.1f}): the median across the kept models "
                    "blurs the interfaces they place apart.",
                    stage="inversion",
                    action=Keep(note="re-inverting cannot fix the models' median blurring"),
                )
            )
        elif free:
            loop.flags.append(
                Flag(
                    name="underfit",
                    message=f"The model misfits {value:.1f} at {where}, the layers chosen by "
                    f"the data from up to {parameters.free.max_layers}.",
                    stage="inversion",
                    action=Reject(
                        reason=f"not fitted by the data's own layers, up to "
                        f"{parameters.free.max_layers}"
                    ),
                    fixable=False,
                )
            )
        elif parameters.n_layers < ceiling:
            loop.change(
                "underfit",
                f"The model misfits {value:.1f} at {where}: try one more layer.",
                n_layers=parameters.n_layers + 1,
            )
        else:
            loop.flags.append(
                Flag(
                    name="underfit",
                    message=f"The model misfits {value:.1f} at {where} with "
                    f"{parameters.n_layers} layers, the most the curve resolves down to "
                    f"{depth:g} m.",
                    stage="inversion",
                    action=Reject(reason=f"not fitted with up to {ceiling} layers"),
                    fixable=False,
                )
            )
    elif (
        settled
        and not free
        and worst is not None
        and contrast is not None
        and contrast[0] < thresholds.min_contrast
        and "n_layers" not in loop.final
        and parameters.n_layers - 1 >= max(fewest_layers, MIN_LAYERS)
    ):
        share, index = contrast
        first, second = measures.vs_layers[index], measures.vs_layers[index + 1]
        loop.change(
            "alike_layers",
            f"Layers {index + 1} and {index + 2} of the layered median ({first:g} and "
            f"{second:g} m/s, {share:.0%} apart) are one: one layer fewer.",
            n_layers=parameters.n_layers - 1,
        )

    loop.flags += _plausibility(measures.vs_layers, thresholds)

    flags = loop.finished()
    kinds = {type(flag.action) for flag in flags}
    rejected = Reject in kinds or any(not flag.fixable for flag in flags)
    verdict = "reject" if rejected else "retry" if Override in kinds else "pass"
    wavelengths = [band.wavelength_m for band in monitored.bands]
    return GateResult(
        gate=GATE,
        unit=unit,
        verdict=verdict,
        metrics=tuple(metrics),
        flags=flags,
        kept=Kept(
            wavelength_m=(wavelengths[0][0], wavelengths[-1][1]) if wavelengths else None,
            n_points=sum(band.n_points for band in monitored.bands),
        ),
    )


class _Loop:
    """The flags of one judgement, and the next attempt's parameters they ask: every flag
    asking a list of layers carries the one all flags made together, so that the changes merged
    are one set of parameters."""

    def __init__(self, parameters: InversionParameters) -> None:
        self.layering = parameters.layering
        self.free = parameters.free.model_dump()
        self.vs_layers = [layer.model_dump() for layer in parameters.vs_layers]
        self.thickness_layers = [layer.model_dump() for layer in parameters.thickness_layers]
        self.flags: list[Flag] = []
        self.final: dict[str, Any] = {}
        self._asked: dict[int, tuple[str, ...]] = {}

    def change(self, name: str, message: str, *lists: str, **values: object) -> None:
        """A flag asking `values` and the final state of the layer `lists` it changed."""
        self.final.update(values)
        self._asked[len(self.flags)] = (*lists, *values)
        self.flags.append(
            Flag(
                name=name,
                message=message,
                stage="inversion",
                action=Override(stage="inversion", overrides={}),
            )
        )

    def depth(self) -> float:
        """The deepest the next attempt's half-space may start."""
        return _depth(layer["thickness_max"] for layer in self.thickness_layers)

    def narrow(self, quantiles: Mapping[str, tuple[float, float]]) -> str:
        """Every range narrowed to where its samples lay (their 5th to 95th percentiles, widened
        by NARROW_MARGIN of that span on each side, at least NARROW_LEAST of the range it had,
        within it), each step the same share of its range as sigpipe's; the ranges said, or ""
        when no parameter has its samples' percentiles."""
        said: list[str] = []
        for i, layer in enumerate(self.vs_layers):
            if (found := quantiles.get(f"vs{i + 1}")) is None:
                continue
            low, high = _narrowed(found, layer["vs_min"], layer["vs_max"], 1.0)
            layer["vs_min"], layer["vs_max"] = low, high
            layer["vs_perturb_std"] = significant((high - low) * VS_STEP_SHARE)
            said.append(f"Vs{i + 1} {low:g}-{high:g}")
        for i, layer in enumerate(self.thickness_layers):
            if (found := quantiles.get(f"thick{i + 1}")) is None:
                continue
            low, high = _narrowed(found, layer["thickness_min"], layer["thickness_max"], 0.01)
            layer["thickness_min"], layer["thickness_max"] = low, high
            layer["thickness_perturb_std"] = max(
                significant((high - low) * THICKNESS_STEP_SHARE), 0.01
            )
            said.append(f"H{i + 1} {low:g}-{high:g}")
        return ", ".join(said)

    def widen_vs(self, measures: InversionMeasures, thresholds: ModelThresholds) -> str:
        """Every Vs bound the posterior piles at moved out (the lowest by WIDEN_LOW, the
        highest by WIDEN_HIGH); the bounds named, or "" when none holds too many samples. When
        the data choose the layers, their one Vs range: the top layer piling at its least, the
        half-space at its most, widen it."""
        named: list[str] = []
        if self.layering == "free":
            for share in measures.at_bounds:
                if share.share <= thresholds.max_at_bound:
                    continue
                if share.parameter in ("top_vs", "half_space_vs") and share.bound == "min":
                    self.free["vs_min"] = round(self.free["vs_min"] * WIDEN_LOW)
                elif share.parameter in ("top_vs", "half_space_vs"):
                    self.free["vs_max"] = round(self.free["vs_max"] * WIDEN_HIGH)
                else:
                    continue
                named.append(
                    f"{share.parameter} {share.bound} {share.value:g} m/s, {share.share:.0%}"
                )
            return "; ".join(named)
        for share in measures.at_bounds:
            if share.share <= thresholds.max_at_bound or not share.parameter.startswith("vs"):
                continue
            layer = self.vs_layers[int(share.parameter[2:]) - 1]
            if share.bound == "min":
                layer["vs_min"] = round(layer["vs_min"] * WIDEN_LOW)
            else:
                layer["vs_max"] = round(layer["vs_max"] * WIDEN_HIGH)
            named.append(f"{share.parameter} {share.bound} {share.value:g} m/s, {share.share:.0%}")
        return "; ".join(named)

    def finished(self) -> tuple[Flag, ...]:
        lists = {
            "vs_layers": self.vs_layers,
            "thickness_layers": self.thickness_layers,
            "free": self.free,
        }
        final = {**self.final, **lists}
        return tuple(
            flag.model_copy(
                update={
                    "action": Override(
                        stage="inversion",
                        overrides={key: final[key] for key in self._asked[index]},
                    )
                }
            )
            if index in self._asked
            else flag
            for index, flag in enumerate(self.flags)
        )


def _narrowed(
    found: tuple[float, float], low: float, high: float, unit: float
) -> tuple[float, float]:
    """The range [`low`, `high`] narrowed to the samples' 5th to 95th percentiles `found`, a
    margin added, rounded to `unit`, at least NARROW_LEAST of it wide, within it."""
    first, last = found
    margin = NARROW_MARGIN * (last - first)
    least = NARROW_LEAST * (high - low)
    start, end = max(low, first - margin), min(high, last + margin)
    if end - start < least:
        middle = min(max((first + last) / 2, low + least / 2), high - least / 2)
        start, end = middle - least / 2, middle + least / 2
    start, end = round(start / unit) * unit, round(end / unit) * unit
    return round(max(low, start), 2), round(min(high, max(end, start + unit)), 2)


def _no_mode(layered: ModelFit) -> Flag:
    lowest = layered.lowest_missing_hz
    message = (
        f"The layered median has no fundamental mode at {layered.n_missing} of the picked "
        "points: they are faster than any mode of a model with a softer layer below. A "
        "higher mode may be picked there"
    )
    # The model is rejected either way; the suggested change is the agent's to make with
    # redo, a band stopping under those points.
    if lowest is not None:
        return Flag(
            name="no_mode",
            message=f"{message}: redo the phase shift with the band below {lowest:g} Hz.",
            stage="phase_shift",
            action=Override(
                stage="phase_shift",
                overrides={"dispersion": {"fmax": round(0.95 * lowest, 1)}},
            ),
            fixable=False,
        )
    return Flag(
        name="no_mode",
        message=f"{message}.",
        stage="picking",
        action=Reject(reason="the curve holds points no fundamental mode of the model reaches"),
        fixable=False,
    )


def _thickness_flags(
    loop: _Loop,
    measures: InversionMeasures,
    thresholds: ModelThresholds,
    reach_m: float | None,
    informed: bool,
    fewest_layers: int,
) -> None:
    """A layer piled at its thinnest is not resolved: one layer fewer, down to the fewest the
    loop allows. A layer piled at its thickest is given room while the data inform the model's
    whole depth and the half-space's top stays within the curve's reach (`reach_m`); at the
    reach, the limit stays and the flag says so."""
    n_layers = len(loop.vs_layers)
    deepest = loop.depth()
    for share in measures.at_bounds:
        if share.share <= thresholds.max_at_bound or not share.parameter.startswith("thick"):
            continue
        if share.bound == "min" and n_layers - 1 >= max(fewest_layers, MIN_LAYERS):
            loop.change(
                "thin_layer",
                f"{share.share:.0%} of {share.parameter}'s samples sit at its thinnest "
                f"({share.value:g} m): the layer is not resolved.",
                n_layers=n_layers - 1,
            )
        elif share.bound == "max" and not informed:
            continue  # the model is shrunk to what the data inform instead
        elif share.bound == "max" and reach_m is not None and deepest < 0.99 * reach_m:
            layer = loop.thickness_layers[int(share.parameter[5:]) - 1]
            layer["thickness_max"] = round(layer["thickness_max"] * WIDEN_HIGH, 2)
            loop.change(
                "at_bound",
                f"{share.share:.0%} of {share.parameter}'s samples sit at its thickest "
                f"({share.value:g} m), shallower than the curve reaches ({reach_m:g} m): "
                "widen it.",
                "thickness_layers",
            )
        elif share.bound == "max":
            loop.flags.append(
                Flag(
                    name="deep_interface",
                    message=f"{share.share:.0%} of {share.parameter}'s samples sit at its thickest "
                    f"({share.value:g} m): the interface would go deeper than the curve resolves.",
                    stage="inversion",
                    action=Keep(note="the depth limit stays: the curve does not reach deeper"),
                )
            )


def _free_flags(loop: _Loop, measures: InversionMeasures, thresholds: ModelThresholds) -> bool:
    """When the data choose the layers: models piled at the most layers allowed are allowed
    more, up to the loop's limit (then kept, said); piled at the deepest interface allowed, kept,
    said (the curve reaches no deeper). Whether more layers are asked."""
    more = False
    for share in measures.at_bounds:
        if share.share <= thresholds.max_at_bound:
            continue
        if share.parameter == "layers":
            most = int(loop.free["max_layers"])
            if most < thresholds.max_layers:
                loop.free["max_layers"] = min(thresholds.max_layers, most + MORE_LAYERS)
                loop.change(
                    "at_bound",
                    f"{share.share:.0%} of the models hold the most layers allowed ({most}): "
                    f"allow {loop.free['max_layers']}.",
                    "free",
                )
                more = True
            else:
                loop.flags.append(
                    Flag(
                        name="most_layers",
                        message=f"{share.share:.0%} of the models hold the most layers the "
                        f"loop allows ({most}).",
                        stage="inversion",
                        action=Keep(note="the loop's limit stays"),
                    )
                )
        elif share.parameter == "deepest_interface":
            loop.flags.append(
                Flag(
                    name="deep_interface",
                    message=f"{share.share:.0%} of the models put an interface at the deepest "
                    f"allowed ({share.value:g} m): the curve resolves no deeper.",
                    stage="inversion",
                    action=Keep(note="the depth limit stays: the curve does not reach deeper"),
                )
            )
    return more


def _too_deep(
    loop: _Loop,
    useful: float,
    depth: float,
    tmin: float,
    thresholds: ModelThresholds,
    fewest_layers: int,
) -> bool:
    """The data inform the model down to `useful` only, well above the `depth` its half-space
    may start at: the layers shrink to end there, as few as fit, never fewer than the loop
    allows; kept when that leaves too little room. Whether they shrink."""
    least = max(fewest_layers, MIN_LAYERS)
    # Each layer at least its thinnest, with room to move: a quarter more at the least.
    target = round(max(useful, (least - 1) * tmin * 1.25), 2)
    if target >= thresholds.min_useful_share * depth:
        loop.flags.append(
            Flag(
                name="too_deep",
                message=f"The data inform the model down to {useful:g} m of the {depth:g} m its "
                f"half-space may start at, too little for {least} layers of at least {tmin:g} m.",
                stage="inversion",
                action=Keep(note="the depth stays: fewer layers would not fit the curve"),
            )
        )
        return False
    count = len(loop.thickness_layers)
    for layer in loop.thickness_layers:
        # Every layer the same share of the new depth, its step in proportion.
        ratio = layer["thickness_perturb_std"] / (layer["thickness_max"] - layer["thickness_min"])
        layer["thickness_max"] = math.floor(target / count * 100) / 100  # never deeper
        layer["thickness_perturb_std"] = max(
            significant(ratio * (layer["thickness_max"] - layer["thickness_min"])), 0.01
        )
    within = _layers_within(target, tmin)
    fewer = {"n_layers": within} if within < count + 1 else {}
    loop.change(
        "too_deep",
        f"The data inform the model down to {useful:g} m, where its half-space may start as deep "
        f"as {depth:g} m: the layers end at {target:g} m"
        + (f", {within} of them." if fewer else "."),
        "thickness_layers",
        **fewer,
    )
    return True


def _plausibility(vs_layers: tuple[float, ...], thresholds: ModelThresholds) -> list[Flag]:
    """What of the layered median to look at, reported only: a strong low-velocity layer
    (possible, under a stiff crust), a Vs outside what soils and rocks near the surface have."""
    flags: list[Flag] = []
    drops = [
        (below / above, index)
        for index, (above, below) in enumerate(itertools.pairwise(vs_layers))
        if above > 0 and below < thresholds.max_vs_drop * above
    ]
    if drops:
        ratio, index = min(drops)
        flags.append(
            Flag(
                name="strong_inversion",
                message=f"Layer {index + 2} of the layered median ({vs_layers[index + 1]:g} m/s) "
                f"is {ratio:.0%} of the Vs above it ({vs_layers[index]:g} m/s): a strong "
                "low-velocity layer, possible under a stiff crust; worth a look.",
                stage="inversion",
                action=Keep(note="reported: plausibility is the user's to judge"),
            )
        )
    low, high = thresholds.plausible_vs
    outside = [vs for vs in vs_layers if not low <= vs <= high]
    if outside:
        flags.append(
            Flag(
                name="implausible_vs",
                message=f"The layered median has Vs of {', '.join(f'{vs:g}' for vs in outside)} "
                f"m/s, outside {low:g} to {high:g} m/s.",
                stage="inversion",
                action=Keep(note="reported: plausibility is the user's to judge"),
            )
        )
    return flags


def _layers_within(depth: float, thinnest: float) -> int:
    """The most layers, the half-space among them, whose layers above it each keep a range of
    thicknesses, all within `depth`: as the checks before S4 count them."""
    above = max(1, math.floor(round(depth / thinnest, 6)))
    while above > 1 and round(depth / above, 2) <= thinnest:
        above -= 1
    return above + 1


def _depth(maxima: Iterable[float]) -> float:
    """The deepest the half-space's top may be: every layer above it at its thickest."""
    return round(sum(maxima), 2)


def model_depth(parameters: InversionParameters) -> float:
    """The deepest the half-space's top may be: every layer given at its thickest; the deepest
    interface allowed when the data choose the layers."""
    if parameters.layering == "free":
        return round(parameters.free.depth_max or 0.0, 2)
    return _depth(layer.thickness_max for layer in parameters.thickness_layers)


def _judged(measures: InversionMeasures) -> set[str]:
    """The series the chains are judged on: the models' Vs at the depths watched; every one for
    windows measured before."""
    return set(measures.watched) or set(measures.rhat)


def _least_contrast(vs_layers: tuple[float, ...]) -> tuple[float, int] | None:
    """The smallest difference of Vs between adjacent layers, relative to their mean, and the
    upper layer's index; None with fewer than two layers."""
    pairs = [
        (abs(below - above) / ((above + below) / 2), index)
        for index, (above, below) in enumerate(itertools.pairwise(vs_layers))
        if above + below > 0
    ]
    return min(pairs) if pairs else None


def significant(value: float) -> float:
    """`value` to 3 significant digits: a step is never rounded to 0."""
    return float(f"{value:.3g}")


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


def _value(value: float | None) -> str:
    return "n/a" if value is None else f"{value:g}"
