import shutil
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from paco.picking import PickingParameters, pick_modes
from paco.picks import PickSummary, RunPicks, pick
from paco.quality import Flag, RunQuality, Verdict, dispersion_quality
from paco.runs import RunError, find_run, load_image, run_processing
from paco.settings import Settings
from sigpipe.base import DispersionCurvesImage, Mode
from sigpipe.dataio.dispersion.loading import load_dispersion_curves
from sigpipe.dataio.dispersion.saving import save_dispersion_curves

# Four 24-receiver windows along the active demo line, as in test_runs.py; all four are good.
SMALL_WINDOWS = {"masw": {"length": 24, "step": 24}}
WINDOW_LENGTH = 23 * 0.25  # m, 24 receivers 0.25 m apart


@dataclass(frozen=True)
class Picked:
    settings: Settings
    folder: Path  # the run folder
    working_dir: Path  # where the tools were called from
    figures: dict[str, bytes]  # each window's figure before pick, by window folder
    summary: PickSummary


# The real run is the slow part: active_p1 is processed, assessed and picked once. Tests that
# change the run work on a copy.
@pytest.fixture(scope="module")
def picked(demo_input_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Picked:
    root = tmp_path_factory.mktemp("picks")
    settings = Settings(input_dir=demo_input_dir, output_dir=root / "outputs", workers=2)
    working_dir = root / "working_dir"
    working_dir.mkdir()

    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(working_dir)
        run_id = run_processing("active_p1", "active", SMALL_WINDOWS, settings).run_id
        dispersion_quality(run_id, settings)
        folder = find_run(run_id, settings)
        figures = {
            path.parent.name: path.read_bytes()
            for path in folder.glob("xmid_*/DispersionImage_0000.png")
        }
        summary = pick(run_id, settings)
    return Picked(settings, folder, working_dir, figures, summary)


def _copy(picked: Picked, tmp_path: Path) -> tuple[Settings, Path]:
    """A copy of the picked run, to change."""
    settings = Settings(input_dir=picked.settings.input_dir, output_dir=tmp_path)
    folder = shutil.copytree(picked.folder, tmp_path / "active_p1" / picked.folder.name)
    return settings, folder


def _quality(folder: Path) -> RunQuality:
    return RunQuality.model_validate_json((folder / "quality.json").read_text())


def _judge(folder: Path, verdicts: dict[str, Verdict]) -> None:
    """Rewrite the run's quality.json with new verdicts for some windows, by window folder."""
    flags: dict[Verdict, tuple[Flag, ...]] = {
        "good": (),
        "doubtful": ("sharpness",),
        "bad": ("sharpness", "prominence"),
    }
    quality = _quality(folder)
    windows = tuple(
        window.model_copy(
            update={
                "quality": window.quality.model_copy(
                    update={
                        "verdict": verdicts[window.folder],
                        "flags": flags[verdicts[window.folder]],
                    }
                )
            }
        )
        if window.folder in verdicts
        else window
        for window in quality.windows
    )
    (folder / "quality.json").write_text(
        quality.model_copy(update={"windows": windows}).model_dump_json()
    )


# ---------------------------------------------------------------- a first pick


def test_every_good_window_is_picked(picked: Picked) -> None:
    assert picked.summary.model_dump(exclude={"run_id"}) == {
        "profile": "active_p1",
        "n_picked": 4,
        "picked_xmids": ("2.88-20.88 m (4)",),
        "n_skipped": 0,
        "n_replaced": 0,
    }


def test_curves_are_the_judged_picks_in_pacs_format(picked: Picked) -> None:
    quality = _quality(picked.folder)

    for window in quality.windows:
        folder = picked.folder / window.folder
        (curves,) = load_dispersion_curves([folder / "DispersionCurves_0000.csv"])
        (saved,) = curves
        expected = pick_modes(load_image(folder), quality.picking)[0].curve

        assert expected is not None
        assert saved.mode == Mode("M", 0)
        # The CSV keeps six decimals.
        assert saved.fs == pytest.approx(expected.fs, rel=1e-5)
        assert saved.vs == pytest.approx(expected.vs, rel=1e-5)
        assert saved.vs_err == pytest.approx(expected.vs_err, rel=1e-5)


def test_pick_json_records_the_picks(picked: Picked) -> None:
    quality = _quality(picked.folder)

    record = RunPicks.model_validate_json((picked.folder / "pick.json").read_text())

    assert record.picking == quality.picking
    assert [window.folder for window in record.windows] == [
        window.folder for window in quality.windows
    ]
    for window in record.windows:
        ((curve,),) = load_dispersion_curves(
            [picked.folder / window.folder / "DispersionCurves_0000.csv"]
        )
        assert window.n_points == curve.fs.size
        assert window.band_hz == pytest.approx((curve.fs.min(), curve.fs.max()), rel=1e-5)
        assert not window.replaced


def test_figures_are_redrawn_with_the_curves(picked: Picked) -> None:
    assert len(picked.figures) == 4
    for name, before in picked.figures.items():
        assert (picked.folder / name / "DispersionImage_0000.png").read_bytes() != before


def test_summary_stays_short(picked: Picked) -> None:
    # The summary is what the agent reads: it must stay far below the tool-output budget.
    assert len(picked.summary.model_dump_json()) < 1_000


def test_nothing_is_written_in_the_working_directory(picked: Picked) -> None:
    assert list(picked.working_dir.iterdir()) == []


# ---------------------------------------------------------------- picking again


def test_picking_again_replaces_m0_and_keeps_other_labels(picked: Picked, tmp_path: Path) -> None:
    settings, folder = _copy(picked, tmp_path)
    path = folder / "xmid_2.88" / "DispersionCurves_0000.csv"
    ((automatic,),) = load_dispersion_curves([path])
    # Corrections by hand in PAC's UI: M0 moved up 10 %, and an M1 curve added.
    by_hand = replace(automatic, vs=automatic.vs * 1.1)
    m1 = replace(automatic, vs=automatic.vs * 1.8, mode=Mode("M", 1))
    save_dispersion_curves(DispersionCurvesImage(dispersion_curves=(by_hand, m1)), path=path)

    summary = pick(folder.name, settings)

    ((m0_now, m1_now),) = load_dispersion_curves([path])
    assert (m0_now.mode, m1_now.mode) == (Mode("M", 0), Mode("M", 1))
    assert m0_now.vs == pytest.approx(automatic.vs, rel=1e-5)
    assert m1_now.vs == pytest.approx(m1.vs, rel=1e-5)
    # Every window had an M0 curve from the first pick.
    assert summary.n_replaced == 4


def test_only_good_windows_are_picked(picked: Picked, tmp_path: Path) -> None:
    settings, folder = _copy(picked, tmp_path)
    _judge(folder, {"xmid_8.88": "doubtful", "xmid_20.88": "bad"})
    left = (folder / "xmid_8.88" / "DispersionCurves_0000.csv").read_bytes()

    summary = pick(folder.name, settings)

    assert (summary.n_picked, summary.n_skipped) == (2, 2)
    assert summary.picked_xmids == ("2.88 m (1)", "14.88 m (1)")
    record = RunPicks.model_validate_json((folder / "pick.json").read_text())
    assert [window.folder for window in record.windows] == ["xmid_2.88", "xmid_14.88"]
    # Like PAC, a curve from an earlier pick stays where the window is no longer good.
    assert (folder / "xmid_8.88" / "DispersionCurves_0000.csv").read_bytes() == left


def test_the_runs_picking_parameters_are_used(picked: Picked, tmp_path: Path) -> None:
    settings, folder = _copy(picked, tmp_path)
    # Assessed again, with the search limited to twice the window length.
    dispersion_quality(folder.name, settings, picking=PickingParameters(max_wavelength=2.0))

    pick(folder.name, settings)

    record = RunPicks.model_validate_json((folder / "pick.json").read_text())
    assert record.picking == PickingParameters(max_wavelength=2.0)
    for window in record.windows:
        ((curve,),) = load_dispersion_curves([folder / window.folder / "DispersionCurves_0000.csv"])
        assert (curve.vs / curve.fs).max() <= 2 * WINDOW_LENGTH


# ---------------------------------------------------------------- runs refused


def test_a_run_must_be_assessed_first(picked: Picked, tmp_path: Path) -> None:
    settings, folder = _copy(picked, tmp_path)
    (folder / "quality.json").unlink()

    with pytest.raises(
        RunError,
        match=rf"^Run '{folder.name}' has no quality assessment yet: call dispersion_quality\.$",
    ):
        pick(folder.name, settings)


def test_a_run_without_good_windows_is_refused(picked: Picked, tmp_path: Path) -> None:
    settings, folder = _copy(picked, tmp_path)
    _judge(folder, {window.folder: "bad" for window in _quality(folder).windows})
    before = {path: path.read_bytes() for path in folder.rglob("*") if path.is_file()}

    with pytest.raises(RunError, match=rf"^Run '{folder.name}' has no good window to pick"):
        pick(folder.name, settings)

    # Nothing changed.
    assert {path: path.read_bytes() for path in folder.rglob("*") if path.is_file()} == before
