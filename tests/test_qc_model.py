"""G5 on measures with a known answer: a model that fits and chains that agree, chains that do
not, a posterior piled at a bound, a misfit only the smoothing makes, an underfit, points no
mode reaches."""

from typing import Any

import pytest
from sigpipe.masw.inversion import InversionParameters
from sigpipe.masw.inversion.measuring import BandFit, BoundShare, InversionMeasures, ModelFit

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
        "fits": (_fit("smooth_median"), _fit("median")),
        "rhat": {"vs1": 1.01, "vs2": 1.02, "thick1": 1.01},
        "acceptance": (60.0, 62.0, 58.0, 61.0, 64.0),
        "samples_per_chain": 600,
        "at_bounds": (
            BoundShare(parameter="vs2", bound="min", value=130.0, share=0.03),
            BoundShare(parameter="thick1", bound="max", value=5.5, share=0.02),
        ),
        "useful_depth_m": 4.0,
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
        "acceptance": 58.0,
        "samples_per_chain": 600,
        "at_bound": 0.03,
        "useful_depth": 4.0,
    }
    assert result.kept.wavelength_m == (2.0, 11.0) and result.kept.n_points == 9


@pytest.mark.parametrize(
    "change",
    [
        {"rhat": {"vs1": 1.01, "vs2": 1.3, "thick1": 1.01}},
        {"acceptance": (60.0, 5.0, 58.0, 61.0, 64.0)},
        {"samples_per_chain": 50},
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
        "overrides": {"n_iterations": 200_000, "n_burnin_iterations": 20_000},
    }


def test_too_few_models_a_chain_ask_for_enough_iterations_at_once() -> None:
    # 2,000 iterations keep 12 models a chain: doubling twice would still leave 48.
    short = PARAMETERS.model_copy(update={"n_iterations": 2_000, "n_burnin_iterations": 200})

    result = _judge(_measures(samples_per_chain=12), short)

    flag = _flags(result)["not_converged"]
    assert flag.action.model_dump()["overrides"] == {
        "n_iterations": 17_000,
        "n_burnin_iterations": 1_700,
    }
    assert flag.message.endswith("sample longer, 17000 iterations.")


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
    # Never fewer than 3 layers (the user, 2026-09-25): with three, nothing to do.
    three = InversionParameters.model_validate(
        {
            "n_layers": 3,
            "vs_layers": [{"vs_min": 130.0, "vs_max": 450.0}] * 3,
            "thickness_layers": [{"thickness_min": 1.0, "thickness_max": 2.75}] * 2,
        }
    )
    assert _judge(_measures(at_bounds=piled), three).flags == ()


def test_a_misfit_only_the_smoothing_makes_is_kept() -> None:
    # PAC's smoothing spreads a thin stiff layer: the smooth median misses the short
    # wavelengths, the layered median it comes from fits them.
    fits = (_fit("smooth_median", (2.6, 0.7, 0.3)), _fit("median", (1.1, 0.5, 0.3)))
    result = _judge(_measures(fits=fits))

    assert result.verdict == "pass"
    flag = _flags(result)["smoothing_misfit"]
    assert flag.message.startswith("The smooth median misfits 2.6 at short wavelengths (2-4 m)")
    assert flag.action.model_dump() == {
        "kind": "keep",
        "note": "re-inverting cannot fix a smoothing effect",
    }
    assert next(m for m in result.metrics if m.name == "misfit_short").passed is False


def test_a_model_that_misfits_gets_one_more_layer_up_to_the_limit() -> None:
    fits = (_fit("smooth_median", (0.6, 0.4, 3.0)), _fit("median", (0.6, 0.4, 2.8)))
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
    fits = (_fit("smooth_median", (0.6, 0.4, 3.0)), _fit("median", (0.6, 0.4, 2.8)))
    result = _judge(_measures(fits=fits, samples_per_chain=50))

    assert set(_flags(result)) == {"not_converged"}


def test_points_no_mode_of_the_model_reaches_reject_the_curve() -> None:
    fits = (_fit("smooth_median", (None, 0.4, 0.3), 2), _fit("median", (None, 0.4, 0.3), 2))
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
