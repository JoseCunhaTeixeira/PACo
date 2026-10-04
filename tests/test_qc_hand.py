"""A person's work, made in PAC's pages: verified by them, taken as it is. No gate judges it,
nothing automatic changes it, and the inversion takes it; G4 compares the assistant's curves
with a person's, never the other way round."""

import shutil
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from sigpipe.base.dispersion_curve import DispersionCurve, Mode
from sigpipe.masw.picks import load_curves, save_pick
from sigpipe.masw.quality.line import Series
from sigpipe.masw.runs import RunError, find_run, load_image, load_manifest, run_processing
from sigpipe.masw.runs.origin import JUDGED, mark_auto, mark_edited

from paco import inspection
from paco.qc import (
    LINE,
    REPORT_FILE,
    Attempt,
    ProfileThresholds,
    QCConfig,
    QCReport,
    build_report,
    invertible,
    judge_profile,
    judge_run,
    latest,
    pick_line,
    read_attempts,
    read_report,
    rebuild_state,
    redo_stage,
    retries_in_run,
    submit_inversion,
    summarize_report,
)
from paco.qc.curves import imaged_windows, judge_curves
from paco.qc.inverting import window_modes
from paco.qc.judging import judge_line
from paco.qc.origin import run_work
from paco.qc.positions import at_positions, in_receivers
from paco.settings import Settings

SMALL_WINDOWS = {"masw": {"length": 24, "step": 24}}
M0, M1 = Mode("M", 0), Mode("M", 1)


@pytest.fixture(scope="module")
def made(demo_input_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str, str]:
    """active_p1 processed in four 24-receiver windows twice: as PAC's pages leave a run (no
    check), and judged by the assistant (G1 to G4, its picks). Their outputs, and each run."""
    root = tmp_path_factory.mktemp("hand")
    settings = Settings(input_dir=demo_input_dir, output_dir=root / "outputs", workers=2)
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(root)
        pac = run_processing("active_p1", "active", SMALL_WINDOWS, settings).run_id
        judged = run_processing("active_p1", "active", SMALL_WINDOWS, settings).run_id
        judge_run(judged, settings, QCConfig())
    return root / "outputs", pac, judged


@pytest.fixture
def outputs(
    made: tuple[Path, str, str], demo_input_dir: Path, tmp_path: Path
) -> tuple[Settings, str, str]:
    """A copy of both runs, for one test to change."""
    source, pac, judged = made
    shutil.copytree(source, tmp_path / "outputs")
    return (
        Settings(input_dir=demo_input_dir, output_dir=tmp_path / "outputs", workers=2),
        pac,
        judged,
    )


def _picked_by_hand(window: Path, mode: Mode = M0, scale: float = 1.0) -> DispersionCurve:
    """`mode`'s curve picked by hand in PAC (normally dispersive, its velocities scaled), saved
    in place of the mode's and marked a person's."""
    image = load_image(window)
    fs = np.linspace(10.0, 40.0, 16)
    curve = DispersionCurve(
        fs=fs, vs=scale * (350.0 - 4.0 * fs), mode=mode, acquisition=image.acquisition
    )
    save_pick(window, image, curve)
    mark_edited(window, mode)
    return curve


def _vs(window: Path, mode: Mode = M0) -> np.ndarray:
    saved = load_curves(window)
    assert saved is not None
    return next(np.asarray(one.vs) for one in saved.dispersion_curves if one.mode == mode)


def test_a_run_made_in_pac_is_inverted_as_the_user_picked_it(
    outputs: tuple[Settings, str, str],
) -> None:
    settings, run_id, _ = outputs
    run_folder = find_run(run_id, settings)
    manifest = load_manifest(run_id, settings)
    for window in manifest.windows:
        _picked_by_hand(run_folder / window.folder)

    work = run_work(run_folder, manifest)
    ready = invertible(run_folder, manifest)

    # Its images and curves a person's: every window with its curve, no gate asked.
    assert {one.by_hand for one in work.values()} == {("image", "M0")}
    assert imaged_windows(run_folder, manifest) == {w.folder: None for w in manifest.windows}
    assert set(ready) == {window.folder for window in manifest.windows}
    assert judge_line(run_folder, manifest, QCConfig()) == ()  # no automatic curve: no G4
    record = submit_inversion(run_id, None, settings)
    assert record.total == len(manifest.windows)
    summary = summarize_report(build_report(run_id, run_folder, QCConfig().budgets, 4))
    assert "Made by hand in PAC, verified by the user" in summary
    assert "images at xmid 2.88-20.88 (4); M0 curve at xmid 2.88-20.88 (4)." in summary


def test_a_curve_picked_by_hand_is_taken_as_it_is_and_never_picked_again(
    outputs: tuple[Settings, str, str],
) -> None:
    settings, _, run_id = outputs
    run_folder = find_run(run_id, settings)
    manifest = load_manifest(run_id, settings)
    window = run_folder / "xmid_8.88"
    # Far off its neighbours: an automatic curve like it would be an outlier.
    hand = _picked_by_hand(window, scale=1.6)

    pick_line(run_id, settings)

    np.testing.assert_allclose(_vs(window), hand.vs, rtol=1e-3)
    work = run_work(run_folder, manifest)
    assert work["xmid_8.88"].m0 == "user" and work["xmid_8.88"].by_hand == ("M0",)
    # The assistant's checks of the older curve are its history, not its verdicts.
    report = build_report(run_id, run_folder, QCConfig().budgets, 4)
    unit = next(one for one in report.units if one.unit == "xmid_8.88")
    assert "G3" not in unit.verdicts and "G4" not in unit.verdicts
    ready = invertible(run_folder, manifest)
    np.testing.assert_allclose(ready["xmid_8.88"].vs, hand.vs, rtol=1e-3)


def test_a_mode_added_by_hand_keeps_the_assistants_m0_and_its_checks(
    outputs: tuple[Settings, str, str],
) -> None:
    settings, _, run_id = outputs
    run_folder = find_run(run_id, settings)
    manifest = load_manifest(run_id, settings)
    window = run_folder / "xmid_8.88"
    before = invertible(run_folder, manifest)

    _picked_by_hand(window, M1, scale=1.8)

    work = run_work(run_folder, manifest)["xmid_8.88"]
    assert (work.m0, work.by_hand) == ("judged", ("M1",))
    # Its M0 inverted as the gates passed it, its M1 with it.
    assert set(invertible(run_folder, manifest)) == set(before)
    assert window_modes(window) == (M0, M1)


def test_pacs_automatic_pick_is_judged_before_the_inversion_takes_it(
    outputs: tuple[Settings, str, str],
) -> None:
    settings, _, run_id = outputs
    run_folder = find_run(run_id, settings)
    manifest = load_manifest(run_id, settings)
    ready = set(invertible(run_folder, manifest))
    unit = sorted(ready)[0]
    window = run_folder / unit
    spent = retries_in_run(read_attempts(run_folder))
    # PAC's own automatic pick: another automatic curve, its time recorded.
    image = load_image(window)
    auto = replace(invertible(run_folder, manifest)[unit], mode=M0)
    auto = replace(auto, vs=np.asarray(auto.vs) * 1.02)
    save_pick(window, image, auto)
    mark_auto(window)

    assert run_work(run_folder, manifest)[unit].m0 == "unjudged"
    assert set(invertible(run_folder, manifest)) == ready - {unit}

    judge_curves(run_id, settings)

    attempts = read_attempts(run_folder)
    judged = latest(attempts, unit, "picking")
    assert judged is not None and judged.triggered_by == JUDGED
    assert set(judged.results) == {"G3", "G4"}
    assert run_work(run_folder, manifest)[unit].m0 == "judged"
    np.testing.assert_allclose(_vs(window), auto.vs, rtol=1e-3)  # judged as it is, not picked
    assert retries_in_run(attempts) == spent  # a judgement is no retry
    passed = all(judged.results[gate].verdict == "pass" for gate in ("G3", "G4"))
    assert (unit in invertible(run_folder, manifest)) == passed


def test_g4_compares_the_assistants_curves_with_a_persons_never_the_reverse() -> None:
    wavelengths = np.linspace(2.0, 20.0, 12)
    velocities = 150.0 + 10.0 * wavelengths

    def curve(unit: str, xmid: float, scale: float = 1.0) -> Series:
        return Series(unit, xmid, wavelengths, velocities * scale)

    thresholds = ProfileThresholds()
    # A person's curve off its neighbours is never judged; the assistant's, among agreeing
    # curves (a person's with them), is an outlier.
    trusted = judge_profile(
        [
            curve("a", 0.0),
            curve("b", 1.0),
            curve("hand", 2.0, 1.6),
            curve("c", 3.0),
            curve("d", 4.0),
        ],
        thresholds,
        trusted=frozenset({"hand"}),
    )
    assert "hand" not in {result.unit for result in trusted}
    assert trusted[-1].unit == LINE
    judged = judge_profile(
        [
            curve("a", 0.0),
            curve("h1", 1.0),
            curve("off", 2.0, 1.6),
            curve("h2", 3.0),
            curve("d", 4.0),
        ],
        thresholds,
        trusted=frozenset({"h1", "h2"}),
    )
    verdicts = {result.unit: result.verdict for result in judged}
    assert verdicts["off"] == "retry" and "h1" not in verdicts and "h2" not in verdicts


def test_the_line_done_again_leaves_a_persons_work_as_it_is(
    outputs: tuple[Settings, str, str],
) -> None:
    settings, _, run_id = outputs
    run_folder = find_run(run_id, settings)
    windows = [window.folder for window in load_manifest(run_id, settings).windows]
    _picked_by_hand(run_folder / "xmid_8.88")
    image = (run_folder / "xmid_8.88" / "DispersionImage_0000.hdf5").read_bytes()

    # The images are the line's: one window's alone is no redo of them.
    with pytest.raises(RunError, match="the line's, the same for every window"):
        redo_stage(run_id, "phase_shift", ["xmid_8.88"], {"dispersion": {"vmax": 900}}, settings)
    redo_stage(run_id, "phase_shift", windows, {"dispersion": {"vmax": 900}}, settings)

    # The person's window keeps its image; the line's others are made with the new range.
    assert (run_folder / "xmid_8.88" / "DispersionImage_0000.hdf5").read_bytes() == image
    assert load_image(run_folder / "xmid_2.88").vs.max() == pytest.approx(900.0)
    np.testing.assert_allclose(load_image(run_folder / "xmid_8.88").vs.max(), 1000.0)
    # Every window a person's: nothing left to redo.
    for window in windows:
        _picked_by_hand(run_folder / window)
    with pytest.raises(RunError, match="made by hand in PAC"):
        redo_stage(run_id, "phase_shift", windows, {"dispersion": {"vmax": 800}}, settings)


def test_asked_stages_and_judgements_are_off_the_runs_budget() -> None:
    now = datetime.now(UTC)

    def attempt(trigger: str) -> Attempt:
        return Attempt(
            unit="xmid_1.00",
            stage="picking",
            attempt=1,
            parameters={},
            triggered_by=trigger,
            started_at=now,
            status="succeeded",
        )

    attempts = [attempt(one) for one in ("initial", "asked", JUDGED, "G3:no_ridge", "backtrack")]

    assert retries_in_run(attempts) == 2


def test_inspect_reads_each_run_and_who_made_it_changing_nothing(
    outputs: tuple[Settings, str, str],
) -> None:
    settings, pac, judged = outputs
    run_folder = find_run(pac, settings)
    _picked_by_hand(run_folder / "xmid_2.88")
    stamps = {
        path: path.stat().st_mtime for path in settings.output_dir.rglob("*") if path.is_file()
    }

    runs = inspection.runs_text(settings)
    one = inspection.run_text(judged, settings)
    window = inspection.window_text(pac, 3.0, settings)

    assert f"Run {pac}: active_p1 (active), processed by PAC's pages" in runs
    assert f"Run {judged}: active_p1 (active), processed by the assistant" in runs
    assert "1 M0 curves (1 by hand)" in runs
    # Alike windows next to each other in one line, their count said.
    assert "Settings: mode active" in one and "Retries: " in one
    assert any(line.startswith("xmid 2.88-") and "(" in line for line in one.splitlines()[3:])
    assert window.startswith("xmid 2.88 (the nearest to 3 m): image by hand; M0 by hand; no model.")
    assert "M0 (by hand): 16 points, 10-40 Hz" in window
    after = {
        path: path.stat().st_mtime for path in settings.output_dir.rglob("*") if path.is_file()
    }
    assert after == stamps


def test_positions_are_the_nearest_windows_the_line_holds(
    outputs: tuple[Settings, str, str],
) -> None:
    settings, pac, _ = outputs
    manifest = load_manifest(pac, settings)

    units, read = at_positions(manifest, [9.0, 3.1, 8.0])

    assert units == ["xmid_2.88", "xmid_8.88"]
    assert read == (
        "9 m: xmid 8.88, window 2 of 4; 3.1 m: xmid 2.88, window 1 of 4; "
        "8 m: xmid 8.88, window 2 of 4"
    )
    with pytest.raises(
        RunError, match=r"50 m is off the line of run .*: its windows are at xmid 2\.88 to 20\.88 m"
    ):
        at_positions(manifest, [50.0])


def test_a_window_given_in_metres_is_the_nearest_receiver_count() -> None:
    # 0.25 m between receivers: 15 m spans 61 receivers (60 gaps); a step of 3 m, 12 receivers.
    overrides, notes = in_receivers(
        {"masw": {"length_m": 15.0, "step_m": 3.0}, "mode": "active"}, 0.25
    )

    assert overrides == {"masw": {"length": 61, "step": 12}, "mode": "active"}
    assert notes == (
        "masw length_m 15 m: 61 receivers (15 m).",
        "masw step_m 3 m: 12 receivers (3 m).",
    )
    assert in_receivers({"masw": {"length": 24}}, 0.25) == ({"masw": {"length": 24}}, ())
    with pytest.raises(ValueError, match="length_m must be over"):
        in_receivers({"masw": {"length_m": 0.1}}, 0.25)


def test_the_state_is_rebuilt_from_the_files_a_page_of_pac_changed(
    outputs: tuple[Settings, str, str],
) -> None:
    # A curve picked in PAC changes the window's files, not the QC log: the state read after
    # it says so, the saved report, older, written again (S3).
    settings, _, run_id = outputs
    run_folder = find_run(run_id, settings)
    window = load_manifest(run_id, settings).windows[1].folder
    before = rebuild_state(run_folder)
    assert read_report(run_folder) == before

    _picked_by_hand(run_folder / window)

    after = rebuild_state(run_folder)
    unit = next(one for one in after.units if one.unit == window)
    assert "M0" in unit.by_hand and after != before
    assert read_report(run_folder) == after
    assert QCReport.model_validate_json((run_folder / REPORT_FILE).read_text()) == after
