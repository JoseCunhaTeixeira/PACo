"""S4 on a judged run: the windows G4 passed inverted with bounds from their own curves, G5 on
each model, G6 over the line, every attempt in the log; the inversion done again for one window;
what is refused. Short chains: the sampler is not seeded, so only what does not depend on its
draws is checked."""

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sigpipe.base import DispersionCurve, Mode, VelocityType
from sigpipe.base.acquisition import UNKNOWN_ACQUISITION
from sigpipe.dataio.dispersion.loading import load_dispersion_curves
from sigpipe.masw.inversion import InversionParameters
from sigpipe.masw.inversion.measuring import InversionMeasures
from sigpipe.masw.inversion.priors import Derived
from sigpipe.masw.inversion.window import SAMPLES_FILE
from sigpipe.masw.picks import CURVES_FILE, load_curves
from sigpipe.masw.runs import RunError, find_run, run_processing

from paco.qc import (
    Attempt,
    Budgets,
    QCConfig,
    append_attempt,
    archived_attempts,
    inverting,
    judge_run,
    latest,
    read_attempts,
    read_report,
)
from paco.qc.inverting import MEASURES_FILE, judge_inversions, rerun_inversion
from paco.settings import Settings

SMALL_WINDOWS = {"masw": {"length": 24, "step": 24}}
# 9 models a chain after the burn-in: fast, and G5 must find the chains too short.
SHORT = {"n_iterations": 1_500}


@pytest.fixture(scope="module")
def inverted(
    demo_input_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Settings, str, Path]:
    """active_p1 in four 24-receiver windows, judged up to G4, then inverted."""
    root = tmp_path_factory.mktemp("inverting")
    settings = Settings(input_dir=demo_input_dir, output_dir=root / "outputs", workers=2)
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(root)
        run_id = run_processing("active_p1", "active", SMALL_WINDOWS, settings).run_id
        # No retry: the S4 flow itself, the gates' first verdicts; the loop is tested with the
        # job (test_inversion.py) and the tools (test_server.py).
        judge_run(
            run_id,
            settings,
            QCConfig(budgets=Budgets(per_gate_and_unit=0, inversion_per_window=0)),
        )
        judge_inversions(run_id, settings, SHORT)
    return settings, run_id, find_run(run_id, settings)


def test_the_windows_g4_passed_are_inverted_with_bounds_from_their_curves(
    inverted: tuple[Settings, str, Path],
) -> None:
    _, _, run_folder = inverted
    attempts = read_attempts(run_folder)

    # G3 sent xmid 20.88 back (2 points where its ridge holds): not a curve for the line, not
    # inverted.
    inversions = [a for a in attempts if a.stage == "inversion" and a.unit.startswith("xmid_")]
    assert sorted({a.unit for a in inversions}) == ["xmid_14.88", "xmid_2.88", "xmid_8.88"]
    for attempt in inversions:
        assert (attempt.attempt, attempt.status, attempt.triggered_by) == (
            1,
            "succeeded",
            "initial",
        )
        assert attempt.notes == ()  # nothing was given but the iterations
        parameters = attempt.parameters
        assert parameters["n_iterations"] == 1_500 and parameters["n_burnin_iterations"] == 375
        (curve,) = load_dispersion_curves([run_folder / attempt.unit / CURVES_FILE])[0]
        velocities = np.asarray(curve.vs, dtype=float)
        wavelengths = velocities / np.asarray(curve.fs, dtype=float)
        # The layers chosen by the data: Vs 100 to 2,000 m/s, as the curve (163 to 291 m/s)
        # needs no wider; interfaces from a third of its shortest wavelength to half its longest
        # (deeper, the curve resolves nothing); 8 layers at most.
        assert velocities.min() >= 100 and 1.09 * velocities.max() <= 2_000
        assert parameters["layering"] == "free"
        free = parameters["free"]
        assert (free["vs_min"], free["vs_max"], free["max_layers"]) == (100.0, 2_000.0, 8)
        assert free["depth_min"] == round(float(wavelengths.min()) / 3, 2)
        assert free["depth_max"] == round(float(wavelengths.max()) / 2, 2)
        # What G5 judged is kept next to PAC's files.
        folder = run_folder / attempt.unit
        assert (folder / SAMPLES_FILE).exists()
        measures = InversionMeasures.model_validate_json((folder / MEASURES_FILE).read_text())
        assert measures.samples_per_chain == 7  # (1,500 - 375) // 150
        # Round depths down to half the line's median longest wavelength (about 11 m here: the
        # picks go down to where their ridge breaks).
        assert [depth for depth, _ in measures.vs_at_depths] == [2.0, 4.0, 6.0, 8.0, 10.0]


def test_g5_finds_the_chains_too_short_and_g6_has_no_model(
    inverted: tuple[Settings, str, Path],
) -> None:
    _, _, run_folder = inverted
    report = read_report(run_folder)
    attempts = read_attempts(run_folder)

    for unit in ("xmid_2.88", "xmid_8.88", "xmid_14.88"):
        attempt = latest(attempts, unit, "inversion")
        assert attempt is not None
        g5 = attempt.results["G5"]
        # G5 asks the iterations 100 models a chain need; with no budget, the window is
        # rejected, its flags kept.
        assert g5.verdict == "reject"
        assert [flag.name for flag in g5.flags][:2] == ["budget_spent", "not_converged"]
        flag = next(flag for flag in g5.flags if flag.name == "not_converged")
        assert flag.action.model_dump()["overrides"] == {
            "n_iterations": 20_000,
            "n_burnin_iterations": 5_000,
        }
    # No model passed G5: the line has none, and says where.
    line = next(unit for unit in report.units if unit.unit == "line")
    assert line.verdicts == {"G4": "pass", "G6": "reject"}
    assert report.counts["G5"] == {"reject": 3}


def test_the_inversion_done_again_starts_from_the_windows_parameters(
    inverted: tuple[Settings, str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, run_id, run_folder = inverted
    before = latest(read_attempts(run_folder), "xmid_8.88", "inversion")
    assert before is not None
    monkeypatch.chdir(tmp_path)

    (result,) = rerun_inversion(run_id, ["xmid_8.88"], {"n_iterations": 3_000}, settings)

    assert result.unit == "xmid_8.88"
    again = latest(read_attempts(run_folder), "xmid_8.88", "inversion")
    assert again is not None and (again.attempt, again.triggered_by) == (2, "backtrack")
    # The bounds of the first attempt, the iterations of the override, the burn-in following.
    assert again.parameters["free"] == before.parameters["free"]
    assert (again.parameters["n_iterations"], again.parameters["n_burnin_iterations"]) == (
        3_000,
        750,
    )
    measures = json.loads((run_folder / "xmid_8.88" / MEASURES_FILE).read_text())
    assert measures["samples_per_chain"] == 15  # (3,000 - 750) // 150
    # The first attempt's files are archived; the other windows keep theirs.
    (archive,) = archived_attempts(run_folder / "xmid_8.88")
    assert archive.name == "1_inversion" and (archive / SAMPLES_FILE).exists()
    assert archived_attempts(run_folder / "xmid_2.88") == ()


def test_another_number_of_layers_spreads_the_ranges_over_the_same_depth(
    inverted: tuple[Settings, str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, run_id, run_folder = inverted
    monkeypatch.chdir(tmp_path)
    before = latest(read_attempts(run_folder), "xmid_2.88", "inversion")
    assert before is not None

    # Layers asked of a window whose data chose them: the layers given, from the curve.
    rerun_inversion(run_id, ["xmid_2.88"], {"n_layers": 4}, settings)
    given = latest(read_attempts(run_folder), "xmid_2.88", "inversion")
    assert given is not None and given.parameters["layering"] == "fixed"
    assert given.parameters["n_layers"] <= 4 and len(given.parameters["vs_layers"]) >= 3
    before = given

    rerun_inversion(run_id, ["xmid_2.88"], {"n_layers": 3}, settings)

    attempt = latest(read_attempts(run_folder), "xmid_2.88", "inversion")
    assert attempt is not None and attempt.parameters["n_layers"] == 3
    assert len(attempt.parameters["vs_layers"]) == 3
    assert len(attempt.parameters["thickness_layers"]) == 2
    assert attempt.parameters["n_iterations"] == 1_500  # kept from the attempt before

    def depth(parameters: dict[str, Any]) -> float:
        return sum(layer["thickness_max"] for layer in parameters["thickness_layers"])

    # The same depth, shared by fewer layers, as deep as the curve reaches at most (the first
    # inversion's wide layers go deeper); the half-space keeps its own Vs range.
    curves = load_curves(run_folder / "xmid_2.88")
    assert curves is not None
    m0 = next(curve for curve in curves.dispersion_curves if curve.mode.number == 0)
    reach = 0.5 * float(np.max(np.asarray(m0.vs) / np.asarray(m0.fs)))
    expected = min(depth(before.parameters), reach)
    assert depth(attempt.parameters) == pytest.approx(expected, abs=0.02)
    assert attempt.parameters["vs_layers"][-1] == before.parameters["vs_layers"][-1]


def test_what_cannot_be_inverted_is_refused(
    inverted: tuple[Settings, str, Path], demo_input_dir: Path, tmp_path: Path
) -> None:
    settings, run_id, _ = inverted

    with pytest.raises(RunError, match=r"has no window xmid_20\.88 that G4 passed"):
        rerun_inversion(run_id, ["xmid_20.88"], {}, settings)
    with pytest.raises(RunError, match=r"Unknown inversion parameter\(s\) iterations"):
        rerun_inversion(run_id, ["xmid_2.88"], {"iterations": 5}, settings)
    # A run G4 has not judged.
    fresh = Settings(input_dir=demo_input_dir, output_dir=tmp_path / "outputs", workers=1)
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(tmp_path)
        other = run_processing("active_p1", "active", SMALL_WINDOWS, fresh).run_id
    with pytest.raises(RunError, match="has not been judged up to G4"):
        judge_inversions(other, fresh, SHORT)


def test_a_failed_inversion_is_tried_once_more_with_the_same_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # sigpipe's sampler fails now and then (a chain keeping no predicted curve for some models);
    # it is not seeded, so the same parameters may well pass the next time.
    parameters = InversionParameters.model_validate(SHORT)
    started = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    for unit, failed in (("xmid_1.00", True), ("xmid_2.00", False)):
        append_attempt(
            tmp_path,
            Attempt(
                unit=unit,
                stage="inversion",
                attempt=1,
                parameters=parameters.model_dump(mode="json"),
                triggered_by="initial",
                started_at=started,
                status="failed" if failed else "succeeded",
                notes=("n_layers 2 kept.",),
            ),
        )
    inverted: list[tuple[dict[str, InversionParameters], str]] = []

    def invert(_: Path, jobs: Mapping[str, Derived], *args: object) -> tuple[()]:
        inverted.append(({unit: job.parameters for unit, job in jobs.items()}, str(args[2])))
        return ()

    monkeypatch.setattr(inverting, "_invert", invert)
    curve = DispersionCurve(
        fs=np.array([5.0, 12.0, 24.0, 43.0, 100.0]),
        vs=np.array([150.0, 180.0, 220.0, 260.0, 300.0]),
        mode=Mode("M", 0),
        type=VelocityType.PHASE,
        acquisition=UNKNOWN_ACQUISITION,
    )
    ready = {"xmid_1.00": curve, "xmid_2.00": curve}
    settings = Settings(input_dir=tmp_path, output_dir=tmp_path, workers=1)

    inverting.retry_failed(
        inverting.RetryBatch(tmp_path, QCConfig(), settings, ready, (1.0,), 2, None, None)
    )

    assert inverted == [({"xmid_1.00": parameters}, "S4:failed")]
    # No budget left for the window: it stays failed.
    monkeypatch.setattr(inverting, "_invert", invert)
    inverted.clear()
    # A failed inversion's retry is on the window's inversion budget.
    none_left = QCConfig(budgets=Budgets(inversion_per_window=0))
    inverting.retry_failed(
        inverting.RetryBatch(tmp_path, none_left, settings, ready, (1.0,), 2, None, None)
    )
    assert inverted == []


def test_the_runs_sampled_longer_count_the_one_being_judged() -> None:
    started = datetime(2026, 9, 26, tzinfo=UTC)
    # The first run, a retry that narrowed the ranges (the same iterations), then one that
    # sampled longer.
    runs = (
        ("initial", 100_000),
        ("G5:not_converged", 100_000),
        ("G5:not_converged", 200_000),
    )
    attempts = [
        Attempt(
            unit="xmid_1.00",
            stage="inversion",
            attempt=number,
            parameters={"n_iterations": iterations},
            triggered_by=trigger,
            started_at=started,
            status="succeeded",
        )
        for number, (trigger, iterations) in enumerate(runs, start=1)
    ]

    # Judging a run that doubled them again: two runs sampled longer, the cap.
    assert inverting.longer_runs(attempts, "xmid_1.00", {"n_iterations": 400_000}) == 2
    assert inverting.longer_runs(attempts, "xmid_1.00", {"n_iterations": 200_000}) == 1
    assert inverting.longer_runs(attempts[:2], "xmid_1.00") == 0
    assert inverting.longer_runs(attempts, "xmid_2.00") == 0
    # The narrowing was done: by the second run, or by the one being judged.
    assert inverting.narrowed(attempts, "xmid_1.00")
    assert not inverting.narrowed(attempts[:1], "xmid_1.00")
    assert inverting.narrowed(
        attempts[:1], "xmid_1.00", "G5:not_converged", {"n_iterations": 100_000}
    )
    assert not inverting.narrowed(
        attempts[:1], "xmid_1.00", "G5:too_deep", {"n_iterations": 100_000}
    )


@pytest.mark.parametrize(
    ("workers", "windows", "expected"),
    [(6, 1, 5), (6, 2, 3), (6, 3, 2), (6, 4, 1), (6, 12, 1), (1, 1, 1), (8, 1, 5)],
)
def test_idle_cores_run_a_windows_chains(workers: int, windows: int, expected: int) -> None:
    # Never more processes than workers, nor than a window's 5 chains.
    assert inverting.chain_jobs(workers, windows, chains=5) == expected
    assert min(workers, windows) * expected <= workers
