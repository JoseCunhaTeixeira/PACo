"""The petrophysical inversion the QC way: the range check, G7 on each window, G8 along the line,
on made-up measures, then the whole flow on the demo line, where the one bundled Silex model
covers a single curve of six (they end below the 43 Hz it needs). Needs sigpipe's silex and
santiludo extras."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from sigpipe.masw.quality.line import Series
from sigpipe.masw.runs import find_run, run_processing

if not all(importlib.util.find_spec(name) for name in ("santiludo", "keras", "keras_nlp")):
    pytest.skip("needs PACo's petro extra", allow_module_level=True)

from sigpipe.masw.inversion.measuring import BandFit, ModelFit
from sigpipe.masw.petro.measuring import PetroMeasures

from paco.qc import (
    Budgets,
    Flag,
    GateResult,
    Keep,
    PetroLineThresholds,
    PetroThresholds,
    QCConfig,
    archived_attempts,
    downstream,
    invert_petro_line,
    judge_petro,
    judge_petro_line,
    judge_run,
    petro_models,
    read_attempts,
)
from paco.qc.petro import MEASURES_FILE
from paco.settings import Settings

GRAND_EST = "grand_est_15-50hz_193-415mps"


def _measures(misfits: tuple[float | None, ...], n_missing: int = 0) -> PetroMeasures:
    bands = tuple(
        BandFit(wavelength_m=(2.0 * i + 1, 2.0 * i + 2), n_points=4, misfit=misfit, residual=0.01)
        for i, misfit in enumerate(misfits)
    )
    known = [misfit for misfit in misfits if misfit is not None]
    return PetroMeasures(
        silex_model=GRAND_EST,
        fit=ModelFit(
            model="petro",
            misfit=max(known) if known else None,
            n_missing=n_missing,
            bands=bands,
        ),
        soils=("silt", "sand"),
        thicknesses_m=(2.0, 6.0),
        ns=(8, 9),
        water_table_m=3.0,
        vs_at_depths=((1.0, 180.0), (2.0, 220.0)),
    )


def test_a_soil_column_that_gives_the_curve_back_passes_g7() -> None:
    result = judge_petro("xmid_4.00", _measures((0.8, 1.2, None)), PetroThresholds())

    assert result.verdict == "pass" and not result.flags
    metrics = {metric.name: metric for metric in result.metrics}
    assert metrics["misfit_middle"].value == 1.2 and metrics["misfit_long"].passed
    assert metrics["water_table"].value == 3.0


def test_g7_rejects_a_misfit_and_says_what_could_change_it() -> None:
    alone = judge_petro("xmid_4.00", _measures((0.8, 2.6, 1.0)), PetroThresholds())
    other = judge_petro("xmid_4.00", _measures((0.8, 2.6, 1.0)), PetroThresholds(), ["alps_x"])

    (flag,) = alone.flags
    assert alone.verdict == "reject" and flag.name == "misfit" and not flag.fixable
    assert "2.6 at middle wavelengths (3-4 m)" in flag.message
    assert "no other model covers this curve" in flag.message
    assert "invert_petro with alps_x" in other.flags[0].message and other.flags[0].fixable


def test_g7_rejects_points_without_a_mode_or_without_uncertainties() -> None:
    no_mode = judge_petro("xmid_4.00", _measures((0.8, 1.0, None), n_missing=2), PetroThresholds())
    unweighed = judge_petro("xmid_4.00", _measures((None, None, None)), PetroThresholds())

    assert no_mode.verdict == "reject" and no_mode.flags[0].name == "no_mode"
    assert unweighed.verdict == "reject" and unweighed.flags[0].name == "no_uncertainty"


def _profiles(off: float = 1.0) -> list[Series]:
    """Five windows of Vs by depth, the middle one `off` times the others."""
    depths = np.array([1.0, 2.0, 3.0])
    return [
        Series(f"xmid_{x:.2f}", x, depths, np.array([180.0, 220.0, 260.0]) * (off if x == 3 else 1))
        for x in (1.0, 2.0, 3.0, 4.0, 5.0)
    ]


def _tables(middle: float = 1.5) -> dict[str, float]:
    return {f"xmid_{x:.2f}": middle if x == 3 else 1.5 for x in (1.0, 2.0, 3.0, 4.0, 5.0)}


def test_g8_passes_a_line_whose_windows_agree() -> None:
    results = judge_petro_line(_profiles(), _tables(), {}, PetroLineThresholds())

    assert [result.verdict for result in results] == ["pass"] * 6
    assert results[-1].unit == "line"


def test_g8_leaves_out_an_outlier_the_curves_do_not_show() -> None:
    vs = judge_petro_line(_profiles(off=1.4), _tables(), {}, PetroLineThresholds())
    table = judge_petro_line(_profiles(), _tables(middle=4.0), {}, PetroLineThresholds())

    assert [result.verdict for result in vs[:5]] == ["pass", "pass", "reject", "pass", "pass"]
    assert vs[2].flags[0].name == "petro_outlier" and "rock physics' Vs" in vs[2].flags[0].message
    assert table[2].verdict == "reject" and "water table" in table[2].flags[0].message


def test_g8_compares_the_water_table_in_metres_and_only_close_windows() -> None:
    # 0.5 m among 1.5 m ones: one of the model's steps, 67 % relative.
    one_step = judge_petro_line(_profiles(), _tables(middle=0.5), {}, PetroLineThresholds())
    # The middle window 40 steps from the others, with a water table 5 m deeper.
    far = [
        Series(
            one.unit,
            40.0 if one.xmid == 3 else one.xmid,
            one.x,
            one.values * 1.4 ** (one.xmid == 3),
        )
        for one in _profiles()
    ]
    alone = judge_petro_line(far, _tables(middle=8.0), {}, PetroLineThresholds())

    assert one_step[2].verdict == "pass"
    water = {metric.name: metric for metric in one_step[2].metrics}["water_table_jump"]
    assert (water.value, water.unit) == (1.0, "m")
    (lone,) = [result for result in alone if result.unit == "xmid_3.00"]
    assert lone.verdict == "pass" and not lone.flags  # no neighbour within 3 steps


def test_g8_keeps_a_change_the_curves_show_too() -> None:
    curve = GateResult(
        gate="G4",
        unit="xmid_3.00",
        verdict="pass",
        flags=(
            Flag(
                name="shared_change",
                message="The curve changes with one side.",
                stage="picking",
                action=Keep(note="geology"),
            ),
        ),
    )

    results = judge_petro_line(
        _profiles(off=1.4), _tables(), {"xmid_3.00": curve}, PetroLineThresholds()
    )

    assert results[2].verdict == "pass" and results[2].flags[0].name == "petro_change"


def test_g8_needs_two_models_for_a_section() -> None:
    (window, line) = judge_petro_line(_profiles()[:1], _tables(), {}, PetroLineThresholds(), (2.0,))

    assert window.verdict == "pass" and line.verdict == "reject"
    assert line.flags[0].name == "gaps"


def test_redoing_the_seismic_inversion_keeps_the_petrophysical_one() -> None:
    assert "petro_inversion" not in downstream("inversion")
    assert downstream("picking") == ("picking", "inversion", "petro_inversion")


@pytest.fixture(scope="module")
def judged(
    demo_input_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Settings, str, Path]:
    """active_p1 in 24-receiver windows every 12, judged up to G4 with no retry."""
    root = tmp_path_factory.mktemp("petro")
    settings = Settings(input_dir=demo_input_dir, output_dir=root / "outputs", workers=2)
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(root)
        windows = {"masw": {"length": 24, "step": 12}}
        run_id = run_processing("active_p1", "active", windows, settings).run_id
        judge_run(run_id, settings, QCConfig(budgets=Budgets(per_gate_and_unit=0)))
    return settings, run_id, find_run(run_id, settings)


def test_a_model_says_how_many_curves_it_covers(judged: tuple[Settings, str, Path]) -> None:
    settings, run_id, _ = judged

    choice = petro_models(run_id, settings)

    (card,) = choice.models
    assert (card.name, card.n_covered) == (GRAND_EST, 1)
    assert card.covers == "1 of the 6 curves G4 passed; 5 end below 43 Hz"
    assert card.trained_on == (
        "15-50 Hz, 193-415 m/s; soils clay, loam, silt, sand; up to 4 layers down to 20 m; "
        "water tables 1-10 m"
    )
    assert choice.next.startswith(f"Call invert_petro with model {GRAND_EST}")


def test_the_covered_curves_are_inverted_and_judged(judged: tuple[Settings, str, Path]) -> None:
    settings, run_id, run_folder = judged

    report, described = invert_petro_line(run_id, GRAND_EST, settings)

    petro = [a for a in read_attempts(run_folder) if a.stage == "petro_inversion"]
    (window,) = [a for a in petro if a.unit != "line"]
    assert window.parameters == {"model": GRAND_EST} and window.status == "succeeded"
    assert {"G7", "G8"} <= set(window.results)
    assert (run_folder / window.unit / MEASURES_FILE).exists()
    line = next(unit for unit in report.units if unit.unit == "line")
    assert line.verdicts["G8"] == "reject"  # one model: no section
    assert described.startswith(f"Silex model {GRAND_EST} covers 1 of the 6 curves G4 passed.")
    assert "Left out, outside its range: 5 end below 43 Hz." in described
    assert "No section: fewer than two models passed." in described
    assert not list(run_folder.glob("PetroInversion_*"))

    # Again: the first one's files archived, a second attempt logged.
    invert_petro_line(run_id, GRAND_EST, settings)

    assert [path.name for path in archived_attempts(run_folder / window.unit)] == [
        "1_petro_inversion"
    ]
    again = [
        a for a in read_attempts(run_folder) if a.stage == "petro_inversion" and a.unit != "line"
    ]
    assert [a.attempt for a in again] == [1, 2]


def test_an_unknown_model_is_refused(judged: tuple[Settings, str, Path]) -> None:
    settings, run_id, _ = judged

    with pytest.raises(ValueError, match="Unknown bundled Silex model 'nowhere'"):
        invert_petro_line(run_id, "nowhere", settings)
