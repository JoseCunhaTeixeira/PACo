"""G5 on measures with a known answer: a model that fits and chains that agree, chains that do
not, a posterior piled at a bound, a model deeper than the data inform, alike layers, a misfit
only the smoothing makes, an underfit, points no mode reaches; the layers chosen by the data;
and a retry's changes as one set of parameters."""

from typing import Any

import pytest
from sigpipe.masw.inversion import InversionParameters
from sigpipe.masw.inversion.measuring import (
    USEFUL_REFERENCE,
    BandFit,
    BoundShare,
    InversionMeasures,
    ModelFit,
)

from paco.qc.g5_model import ModelThresholds, judge_model
from paco.qc.models import Flag, GateResult

THRESHOLDS = ModelThresholds()
PARAMETERS = InversionParameters.model_validate(
    {
        "vs_layers": [{"vs_min": 130.0, "vs_max": 450.0, "vs_perturb_std": 7.1}] * 2,
        "thickness_layers": [{"thickness_min": 1.0, "thickness_max": 5.5}],
    }
)
BANDS = ((2.0, 4.0), (4.0, 7.0), (7.0, 11.0))


def _fit(
    model: str, misfits: tuple[float | None, ...] = (0.6, 0.4, 0.3), missing: int = 0
) -> ModelFit:
    known = [value for value in misfits if value is not None]
    return ModelFit(
        model=model,
        misfit=max(known) if known else None,
        n_missing=missing,
        bands=tuple(
            BandFit(wavelength_m=band, n_points=3, misfit=value, residual=0.01)
            for band, value in zip(BANDS, misfits, strict=True)
        ),
    )


def _measures(**changes: Any) -> InversionMeasures:  # noqa: ANN401
    values: dict[str, Any] = {
        "fits": (_fit("ensemble"), _fit("median")),
        "rhat": {"vs1": 1.01, "vs2": 1.02, "thick1": 1.01},
        "ess": {"vs1": 2_400.0, "vs2": 1_900.0, "thick1": 2_100.0},
        "autocorrelation": {"vs1": 0.1, "vs2": 0.2, "thick1": 0.15},
        "acceptance": (25.0, 24.0, 26.0, 23.0, 27.0),
        "samples_per_chain": 600,
        "at_bounds": (
            BoundShare(parameter="vs2", bound="min", value=130.0, share=0.03),
            BoundShare(parameter="thick1", bound="max", value=5.5, share=0.02),
        ),
        "useful_depth_m": 5.0,
        "useful_reference": USEFUL_REFERENCE,  # read by the current rule, as a run's are
        "depth_max_m": 6.5,
        "vs_at_depths": ((1.0, 230.0), (2.0, 228.0), (3.0, 210.0), (4.0, 200.0), (5.0, 200.0)),
        "vs_layers": (230.0, 200.0),
        "interfaces_m": (3.0,),
    }
    return InversionMeasures.model_validate(values | changes)


def _judge(measures: InversionMeasures, parameters: InversionParameters = PARAMETERS) -> GateResult:
    return judge_model("xmid_8.88", measures, parameters, THRESHOLDS)


def _flags(result: GateResult) -> dict[str, Flag]:
    return {flag.name: flag for flag in result.flags}


def test_a_model_that_fits_with_agreeing_chains_passes() -> None:
    result = _judge(_measures())

    assert (result.gate, result.verdict, result.flags) == ("G5", "pass", ())
    assert {metric.name: metric.value for metric in result.metrics} == {
        "misfit_short": 0.6,
        "misfit_middle": 0.4,
        "misfit_long": 0.3,
        # PAC's residual by band, in %: reported, never judged.
        "residual_short": 1.0,
        "residual_middle": 1.0,
        "residual_long": 1.0,
        "misfit_layered": 0.6,
        "rhat": 1.02,
        "ess": 1_900.0,  # the least of the parameters', against 200
        "autocorrelation": 0.2,  # reported
        "acceptance": 25.0,  # reported: the chains' median, %
        "samples_per_chain": 600,
        "at_bound": 0.03,
        "depth_informed": 5.0,
        "contrast": 14.0,  # 230 over 200 m/s
    }
    assert result.kept.wavelength_m == (2.0, 11.0) and result.kept.n_points == 9


def test_an_acceptance_outside_its_band_is_a_warning() -> None:
    # 15 % of their moves accepted, the median of the chains, when the data chose the layers:
    # under the 20 to 30 % their adapted steps aim at (the user, 2026-09-29: a warning, never a
    # failure).
    slow = (15.0, 16.0, 14.0, 15.0, 17.0)
    result = _judge(_measures(**WATCHED, acceptance=slow), FREE)

    assert "acceptance" in {flag.name for flag in result.flags}
    flag = next(flag for flag in result.flags if flag.name == "acceptance")
    assert flag.action.kind == "keep" and flag.fixable
    rows = [metric for metric in result.metrics if metric.name == "acceptance"]
    assert [(row.threshold, row.bound, row.passed) for row in rows] == [
        (20.0, "min", False),
        (30.0, "max", True),
    ]
    # The layers given (DREAM, near 5 % by design): reported only.
    fixed = _judge(_measures(acceptance=slow))
    assert fixed.verdict == "pass" and fixed.flags == ()
    assert [(row.threshold, row.passed) for row in fixed.metrics if row.name == "acceptance"] == [
        (None, True)
    ]


@pytest.mark.parametrize(
    "change",
    [
        {"rhat": {"vs1": 1.01, "vs2": 1.3, "thick1": 1.01}},
        {"samples_per_chain": 50},
        # Chains that agree, but each sample much like the one before: few independent ones.
        {"ess": {"vs1": 2_400.0, "vs2": 150.0, "thick1": 2_100.0}},
        {"rhat": {"vs1": None, "vs2": None, "thick1": None}},
    ],
)
def test_chains_that_do_not_agree_sample_twice_as_long(change: dict[str, Any]) -> None:
    result = _judge(_measures(**change))

    assert result.verdict == "retry"
    flag = _flags(result)["not_converged"]
    assert flag.action.model_dump() == {
        "kind": "override",
        "stage": "inversion",
        "overrides": {"n_iterations": 400_000, "n_burnin_iterations": 100_000},
    }


QUANTILES = {"vs1": (200.0, 260.0), "vs2": (150.0, 300.0), "thick1": (2.0, 3.0)}


def test_chains_that_do_not_agree_first_narrow_each_range_to_their_samples() -> None:
    measures = _measures(rhat={"vs1": 1.01, "vs2": 1.3, "thick1": 1.01}, quantiles=QUANTILES)

    result = _judge(measures)

    assert result.verdict == "retry"
    flag = _flags(result)["not_converged"]
    overrides = flag.action.model_dump()["overrides"]
    # The samples' 5th to 95th percentiles, widened by a tenth of that span on each side,
    # within the bounds they had; the iterations kept.
    assert [(layer["vs_min"], layer["vs_max"]) for layer in overrides["vs_layers"]] == [
        (194.0, 266.0),
        (135.0, 315.0),
    ]
    (thickness,) = overrides["thickness_layers"]
    assert (thickness["thickness_min"], thickness["thickness_max"]) == (1.9, 3.1)
    # The steps shrink with the ranges.
    assert overrides["vs_layers"][0]["vs_perturb_std"] < PARAMETERS.vs_layers[0].vs_perturb_std
    assert "n_iterations" not in overrides
    assert "narrow each range to where its samples lie (Vs1 194-266" in flag.message


def test_once_narrowed_chains_that_still_disagree_sample_longer() -> None:
    measures = _measures(rhat={"vs1": 1.01, "vs2": 1.3, "thick1": 1.01}, quantiles=QUANTILES)

    result = judge_model("xmid_8.88", measures, PARAMETERS, THRESHOLDS, narrowed=True)

    assert _flags(result)["not_converged"].action.model_dump()["overrides"] == {
        "n_iterations": 400_000,
        "n_burnin_iterations": 100_000,
    }


def test_a_narrowed_range_keeps_a_width() -> None:
    # Samples piled on one value: the range keeps a tenth of the one it had, about them.
    measures = _measures(
        rhat={"vs1": 1.3, "vs2": 1.01, "thick1": 1.01},
        quantiles={"vs1": (300.0, 300.0), "vs2": (150.0, 300.0), "thick1": (2.0, 3.0)},
    )

    overrides = _flags(_judge(measures))["not_converged"].action.model_dump()["overrides"]

    first = overrides["vs_layers"][0]
    assert (first["vs_min"], first["vs_max"]) == (284.0, 316.0)


def test_too_few_models_a_chain_ask_for_enough_iterations_at_once() -> None:
    # 2,000 iterations keep 12 models a chain: doubling twice would still leave 48. 100 models a
    # chain after a burn-in of a quarter: 20,000.
    short = PARAMETERS.model_copy(update={"n_iterations": 2_000, "n_burnin_iterations": 500})

    result = _judge(_measures(samples_per_chain=12), short)

    flag = _flags(result)["not_converged"]
    assert flag.action.model_dump()["overrides"] == {
        "n_iterations": 20_000,
        "n_burnin_iterations": 5_000,
    }
    assert flag.message.endswith("sample longer, 20000 iterations.")


def test_a_posterior_piled_at_a_vs_bound_widens_that_bound() -> None:
    piled = (
        BoundShare(parameter="vs2", bound="min", value=130.0, share=0.3),
        BoundShare(parameter="vs1", bound="max", value=450.0, share=0.15),
    )
    result = _judge(_measures(at_bounds=piled))

    assert result.verdict == "retry"
    flag = _flags(result)["at_bound"]
    assert "vs2 min 130 m/s, 30%" in flag.message and "vs1 max 450 m/s, 15%" in flag.message
    layers = flag.action.model_dump()["overrides"]["vs_layers"]
    assert [(layer["vs_min"], layer["vs_max"]) for layer in layers] == [(130.0, 562), (104, 450.0)]


def test_an_interface_piled_at_its_deepest_keeps_the_depth_limit() -> None:
    piled = (BoundShare(parameter="thick1", bound="max", value=5.5, share=0.3),)
    result = judge_model("xmid_8.88", _measures(at_bounds=piled), PARAMETERS, THRESHOLDS, 5.5)

    assert result.verdict == "pass"
    assert _flags(result)["deep_interface"].action.model_dump()["kind"] == "keep"


def test_a_thickness_bound_shallower_than_the_curve_reaches_is_widened() -> None:
    piled = (BoundShare(parameter="thick1", bound="max", value=5.5, share=0.3),)
    result = judge_model("xmid_8.88", _measures(at_bounds=piled), PARAMETERS, THRESHOLDS, 11.0)

    assert result.verdict == "retry"
    flag = _flags(result)["at_bound"]
    assert "shallower than the curve reaches (11 m)" in flag.message
    (layer,) = flag.action.model_dump()["overrides"]["thickness_layers"]
    assert (layer["thickness_min"], layer["thickness_max"]) == (1.0, 6.88)


def test_a_layer_piled_at_its_thinnest_is_dropped_above_three_layers() -> None:
    four = InversionParameters.model_validate(
        {
            "n_layers": 4,
            "vs_layers": [{"vs_min": 130.0, "vs_max": 450.0}] * 4,
            "thickness_layers": [{"thickness_min": 1.0, "thickness_max": 2.75}] * 3,
        }
    )
    piled = (BoundShare(parameter="thick2", bound="min", value=1.0, share=0.4),)

    result = _judge(_measures(at_bounds=piled), four)
    assert result.verdict == "retry"
    assert _flags(result)["thin_layer"].action.model_dump()["overrides"] == {"n_layers": 3}
    # Never fewer than 3 layers: with three, nothing to do.
    three = InversionParameters.model_validate(
        {
            "n_layers": 3,
            "vs_layers": [{"vs_min": 130.0, "vs_max": 450.0}] * 3,
            "thickness_layers": [{"thickness_min": 1.0, "thickness_max": 2.75}] * 2,
        }
    )
    assert _judge(_measures(at_bounds=piled), three).flags == ()


def test_a_misfit_only_the_models_median_makes_is_kept() -> None:
    # The kept models place a thin stiff layer apart: their median blurs it and misses the short
    # wavelengths, the layered median (one of them) fits them.
    fits = (_fit("ensemble", (2.6, 0.7, 0.3)), _fit("median", (1.1, 0.5, 0.3)))
    result = _judge(_measures(fits=fits))

    assert result.verdict == "pass"
    flag = _flags(result)["smoothing_misfit"]
    assert flag.message.startswith("The ensemble misfits 2.6 at short wavelengths (2-4 m)")
    assert flag.action.model_dump() == {
        "kind": "keep",
        "note": "re-inverting cannot fix the models' median blurring",
    }
    assert next(m for m in result.metrics if m.name == "misfit_short").passed is False


def test_a_model_that_misfits_gets_one_more_layer_up_to_the_limit() -> None:
    fits = (_fit("ensemble", (0.6, 0.4, 3.0)), _fit("median", (0.6, 0.4, 2.8)))
    result = _judge(_measures(fits=fits))

    assert result.verdict == "retry"
    flag = _flags(result)["underfit"]
    assert "long wavelengths (7-11 m)" in flag.message
    assert flag.action.model_dump()["overrides"] == {"n_layers": 3}
    four = InversionParameters.model_validate(
        {
            "n_layers": 4,
            "vs_layers": [{"vs_min": 130.0, "vs_max": 450.0}] * 4,
            "thickness_layers": [{"thickness_min": 1.0, "thickness_max": 1.8}] * 3,
        }
    )
    # Up to what the curve resolves (the checks before S4): here 4 layers.
    rejected = judge_model("xmid_8.88", _measures(fits=fits), four, THRESHOLDS, None, 4)
    assert rejected.verdict == "reject"
    assert _flags(rejected)["underfit"].action.model_dump() == {
        "kind": "reject",
        "reason": "not fitted with up to 4 layers",
    }
    # With no such limit, up to the loop's own 10.
    assert _judge(_measures(fits=fits), four).verdict == "retry"


def test_the_cheapest_fix_first_convergence_before_layers() -> None:
    fits = (_fit("ensemble", (0.6, 0.4, 3.0)), _fit("median", (0.6, 0.4, 2.8)))
    result = _judge(_measures(fits=fits, samples_per_chain=50))

    assert set(_flags(result)) == {"not_converged"}


def test_points_no_mode_of_the_model_reaches_reject_the_curve() -> None:
    fits = (_fit("ensemble", (None, 0.4, 0.3), 2), _fit("median", (None, 0.4, 0.3), 2))
    result = _judge(_measures(fits=fits))

    assert result.verdict == "reject"
    flag = _flags(result)["no_mode"]
    assert (flag.stage, flag.fixable) == ("picking", False)
    assert "at 2 of the picked points" in flag.message
    # Knowing where those points start, the flag suggests the band below them, for the agent's
    # redo; the model stays rejected.
    at = _fit("median", (None, 0.4, 0.3), 2).model_copy(update={"lowest_missing_hz": 60.0})
    result = _judge(_measures(fits=(fits[0], at)))
    assert result.verdict == "reject"
    flag = _flags(result)["no_mode"]
    assert (flag.stage, flag.fixable) == ("phase_shift", False)
    assert flag.action.model_dump() == {
        "kind": "override",
        "stage": "phase_shift",
        "overrides": {"dispersion": {"fmax": 57.0}},
    }
    assert flag.message.endswith("redo the phase shift with the band below 60 Hz.")


def test_thresholds_round_trip() -> None:
    thresholds = ModelThresholds(max_misfit=1.5, max_rhat=1.05)
    assert ModelThresholds.model_validate_json(thresholds.model_dump_json()) == thresholds


def _layers(n_layers: int, thickest: float, thinnest: float = 1.0) -> InversionParameters:
    return InversionParameters.model_validate(
        {
            "n_layers": n_layers,
            "vs_layers": [{"vs_min": 100.0, "vs_max": 1000.0}] * (n_layers - 1)
            + [{"vs_min": 100.0, "vs_max": 2000.0}],
            "thickness_layers": [
                {"thickness_min": thinnest, "thickness_max": thickest, "thickness_perturb_std": 1.0}
            ]
            * (n_layers - 1),
        }
    )


def test_a_model_deeper_than_the_data_inform_is_shrunk_to_them() -> None:
    # Four layers down to 3 x 13 = 39 m, the data informing the top 18 m only: the layers end
    # at 18 m, 6 m each at most, their steps in proportion.
    result = _judge(_measures(useful_depth_m=18.0), _layers(4, 13.0))

    assert result.verdict == "retry"
    flag = _flags(result)["too_deep"]
    assert flag.message == (
        "The data inform the model down to 18 m, where its half-space may start as deep as "
        "39 m: the layers end at 18 m."
    )
    layers = flag.action.model_dump()["overrides"]["thickness_layers"]
    assert [(layer["thickness_max"], layer["thickness_perturb_std"]) for layer in layers] == [
        (6.0, 0.417)
    ] * 3
    useful = next(metric for metric in result.metrics if metric.name == "depth_informed")
    assert (useful.threshold, useful.passed) == (31.2, False)
    # Shallower still, fewer layers fit: 3 m holds two layers of at least 1 m above the
    # half-space, with room.
    fewer = _flags(_judge(_measures(useful_depth_m=3.0), _layers(4, 13.0)))["too_deep"]
    assert fewer.action.model_dump()["overrides"]["n_layers"] == 3
    assert fewer.message.endswith("the layers end at 3 m, 3 of them.")
    # Too little room even for three layers: the depth stays, and the flag says so.
    kept = _flags(_judge(_measures(useful_depth_m=1.0), _layers(3, 1.5)))["too_deep"]
    assert kept.action.model_dump()["kind"] == "keep"


def test_the_layer_count_waits_for_the_depth() -> None:
    # A misfit and a model too deep: the depth first, the layers once it ends where the data
    # inform it.
    fits = (_fit("ensemble", (0.6, 0.4, 3.0)), _fit("median", (0.6, 0.4, 2.8)))
    result = _judge(_measures(fits=fits, useful_depth_m=18.0), _layers(4, 13.0))

    assert set(_flags(result)) == {"too_deep"}


def test_alike_layers_of_a_model_that_fits_are_one() -> None:
    measures = _measures(vs_layers=(210.0, 330.0, 340.0, 520.0), useful_depth_m=None)

    result = _judge(measures, _layers(4, 3.0))

    assert result.verdict == "retry"
    flag = _flags(result)["alike_layers"]
    assert flag.message == (
        "Layers 2 and 3 of the layered median (330 and 340 m/s, 3% apart) are one: one layer fewer."
    )
    assert flag.action.model_dump()["overrides"] == {"n_layers": 3}
    # Not below the fewest the loop allows: 3 layers misfit before.
    kept = judge_model("xmid_8.88", measures, _layers(4, 3.0), THRESHOLDS, fewest_layers=4)
    assert kept.verdict == "pass"
    # Nor while the model misfits: then it needs a layer more, not fewer.
    fits = (_fit("ensemble", (0.6, 0.4, 3.0)), _fit("median", (0.6, 0.4, 2.8)))
    assert set(_flags(_judge(measures.model_copy(update={"fits": fits}), _layers(4, 3.0)))) == {
        "underfit"
    }


def test_a_retrys_changes_are_one_set_of_parameters() -> None:
    # Two thicknesses piled at their thickest, well within the curve's reach: both flags carry
    # the same layers, each widening in.
    piled = (
        BoundShare(parameter="thick1", bound="max", value=3.0, share=0.3),
        BoundShare(parameter="thick2", bound="max", value=3.0, share=0.2),
    )
    result = judge_model(
        "xmid_8.88",
        _measures(at_bounds=piled, useful_depth_m=None),
        _layers(3, 3.0),
        THRESHOLDS,
        30.0,
    )

    first, second = (flag.action.model_dump()["overrides"] for flag in result.flags)
    assert first == second
    assert [layer["thickness_max"] for layer in first["thickness_layers"]] == [3.75, 3.75]


def test_a_model_the_curve_informs_nowhere_is_rejected() -> None:
    # The posterior as wide as the prior at every depth: nothing to shrink, nothing to retry.
    result = _judge(_measures(useful_depth_m=0.0), _layers(4, 13.0))

    assert result.verdict == "reject"
    assert set(_flags(result)) == {"uninformed"}
    flag = _flags(result)["uninformed"]
    assert (flag.stage, flag.fixable) == ("phase_shift", False)
    assert "longer windows" in flag.message


def test_implausible_profiles_are_reported_never_failed() -> None:
    # A layer at 40 % of the Vs above it, and a half-space faster than rocks near the surface.
    result = _judge(
        _measures(vs_layers=(320.0, 128.0, 2_700.0), useful_depth_m=None), _layers(3, 3.0)
    )

    assert result.verdict == "pass"
    flags = _flags(result)
    assert flags["strong_inversion"].message.startswith(
        "Layer 2 of the layered median (128 m/s) is 40% of the Vs above it (320 m/s)"
    )
    assert flags["implausible_vs"].action.model_dump()["kind"] == "keep"


def test_the_posterior_is_judged_once_the_chains_agree() -> None:
    # Chains that disagree say nothing yet of what the data inform: sample longer first.
    result = _judge(_measures(useful_depth_m=0.0, samples_per_chain=50), _layers(4, 13.0))

    assert set(_flags(result)) == {"not_converged"}
    assert (
        _judge(_measures(useful_depth_m=18.0, samples_per_chain=50), _layers(4, 13.0)).flags[0].name
        == "not_converged"
    )


def test_chains_that_still_disagree_are_kept_with_a_warning() -> None:
    # Sampled longer twice already: the model is kept, said to hold several modes.
    disagree = _measures(rhat={"vs1": 1.01, "vs2": 1.6, "thick1": 1.01})

    result = judge_model("xmid_8.88", disagree, PARAMETERS, THRESHOLDS, longer_runs=2)

    assert result.verdict == "pass"
    flag = _flags(result)["multimodal"]
    assert flag.action.model_dump()["kind"] == "keep"
    assert "after sampling 2 times longer (R-hat 1.6" in flag.message
    assert judge_model("xmid_8.88", disagree, PARAMETERS, THRESHOLDS, longer_runs=1).verdict == (
        "retry"
    )


def test_chains_that_agree_with_few_samples_are_kept_and_judged() -> None:
    # R-hat within the limit, 150 effective samples: sampled longer twice, kept as it stands.
    few = _measures(ess={"vs1": 2_400.0, "vs2": 150.0, "thick1": 2_100.0})

    result = judge_model("xmid_8.88", few, PARAMETERS, THRESHOLDS, longer_runs=2)

    assert (result.verdict, [flag.name for flag in result.flags]) == ("pass", ["few_samples"])
    # Agreeing chains are enough to judge what the data inform, whatever their samples.
    uninformed = few.model_copy(update={"useful_depth_m": 0.0})
    assert _flags(judge_model("xmid_8.88", uninformed, PARAMETERS, THRESHOLDS))["uninformed"]


# The layers chosen by the data: one Vs range, a depth, the most layers.
FREE = InversionParameters.model_validate(
    {
        "layering": "free",
        "free": {
            "vs_min": 100.0,
            "vs_max": 2_000.0,
            "depth_min": 1.0,
            "depth_max": 15.0,
            "max_layers": 8,
        },
    }
)
WATCHED = {
    "rhat": {"vs@1m": 1.01, "vs@3m": 1.02, "vs@5m": 1.01, "layers": 1.4, "noise": 1.3},
    "ess": {"vs@1m": 900.0, "vs@3m": 600.0, "vs@5m": 700.0, "layers": 20.0, "noise": 30.0},
    "watched": ("vs@1m", "vs@3m", "vs@5m"),
    "at_bounds": (),
}


def test_the_chains_are_judged_on_vs_at_the_depths_watched() -> None:
    # The number of layers and the noise factor mix slowly; the models' Vs is what is used.
    result = _judge(_measures(**WATCHED), FREE)

    assert result.verdict == "pass"
    metrics = {metric.name: metric.value for metric in result.metrics}
    assert (metrics["rhat"], metrics["ess"]) == (1.02, 600.0)
    # Chains that disagree at a depth watched: twice as long, no range to narrow.
    far = WATCHED | {"rhat": {"vs@1m": 1.01, "vs@3m": 1.3, "vs@5m": 1.01}}
    flag = _flags(_judge(_measures(**far), FREE))["not_converged"]
    assert flag.action.model_dump()["overrides"] == {
        "n_iterations": 400_000,
        "n_burnin_iterations": 100_000,
    }


def test_the_datas_layers_piled_at_a_bound_widen_it() -> None:
    piled = (
        BoundShare(parameter="half_space_vs", bound="max", value=2_000.0, share=0.3),
        BoundShare(parameter="layers", bound="max", value=8.0, share=0.4),
        BoundShare(parameter="deepest_interface", bound="max", value=15.0, share=0.5),
    )

    result = _judge(_measures(**(WATCHED | {"at_bounds": piled})), FREE)

    flags = _flags(result)
    assert result.verdict == "retry"
    # The Vs range and the most layers widened as one set; the depth kept: the curve's reach.
    free = flags["at_bound"].action.model_dump()["overrides"]["free"]
    assert (free["vs_max"], free["max_layers"], free["depth_max"]) == (2_500, 10, 15.0)
    assert flags["deep_interface"].action.kind == "keep"
    # At the loop's limit, the layers are kept, said.
    most = FREE.model_copy(update={"free": FREE.free.model_copy(update={"max_layers": 10})})
    again = _flags(_judge(_measures(**(WATCHED | {"at_bounds": piled[1:2]})), most))
    assert set(again) == {"most_layers"}


def test_the_datas_layers_that_misfit_are_rejected() -> None:
    misfit = _fit("ensemble", (3.0, 0.4, 0.3))
    result = _judge(
        _measures(**(WATCHED | {"fits": (misfit, _fit("median", (3.0, 0.4, 0.3)))})), FREE
    )

    assert result.verdict == "reject"
    assert "not fitted by the data's own layers, up to 8" in str(
        _flags(result)["underfit"].action.model_dump()
    )
    # Deeper than the data inform: reported, the depth is the curve's reach.
    shallow = _judge(_measures(**(WATCHED | {"useful_depth_m": 3.0})), FREE)
    assert shallow.verdict == "pass"
