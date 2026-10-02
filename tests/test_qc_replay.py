"""A run made again from its inputs and its QC log (S7): the same records, images and curves
within the stated tolerance; a person's work and the inversions not replayed, and said."""

import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from sigpipe.base.dispersion_curve import DispersionCurve
from sigpipe.masw.picks import save_pick
from sigpipe.masw.profiles import Profile
from sigpipe.masw.runs import RunError, find_run, load_image, run_processing, window_folders
from sigpipe.masw.runs.caching import Cache, using
from sigpipe.masw.runs.history import forget
from sigpipe.masw.runs.models import InputFile
from sigpipe.masw.runs.origin import M0, mark_edited

from paco.qc import LINE, Attempt, QCConfig, judge_run
from paco.qc import replay as replaying
from paco.qc.judging import saved_m0
from paco.qc.models import Stage
from paco.qc.replay import REPLAYS, replay_run
from paco.settings import Settings

SMALL_WINDOWS = {"masw": {"length": 24, "step": 24}}


@pytest.fixture(scope="module")
def made(demo_input_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    """active_p1 in four 24-receiver windows, judged by the assistant: G1 to G4, its picks."""
    root = tmp_path_factory.mktemp("replay")
    settings = Settings(input_dir=demo_input_dir, output_dir=root / "outputs", workers=2)
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(root)
        run_id = run_processing("active_p1", "active", SMALL_WINDOWS, settings).run_id
        judge_run(run_id, settings, QCConfig())
    return root / "outputs", run_id


@pytest.fixture
def run(made: tuple[Path, str], demo_input_dir: Path, tmp_path: Path) -> tuple[Settings, str]:
    """A copy of the run, for one test to change."""
    source, run_id = made
    shutil.copytree(source, tmp_path / "outputs")
    return Settings(input_dir=demo_input_dir, output_dir=tmp_path / "outputs", workers=2), run_id


def test_a_run_replays_to_the_same_records_images_and_curves(run: tuple[Settings, str]) -> None:
    settings, run_id = run

    replay = replay_run(run_id, settings)

    stages = {one.stage for one in replay.compared}
    assert stages == {"preprocessing", "phase_shift", "picking"}
    assert replay.same and all(one.difference == 0.0 for one in replay.compared)
    assert sum(one.stage == "phase_shift" for one in replay.compared) == 4
    assert replay.lines()[-1] == f"Run {run_id} replayed: the same."
    # Its outputs gone, unless kept: the run's folder holds nothing of it.
    assert replay.folder is None and not any((settings.output_dir / REPLAYS).iterdir())


def test_a_replay_makes_every_image_again(run: tuple[Settings, str], tmp_path: Path) -> None:
    settings, run_id = run
    cache = Cache(tmp_path / "cache", max_bytes=1 << 30)

    with using(cache):
        replay = replay_run(run_id, settings)

    # Within a caller's cache, none taken from it nor kept in it.
    assert replay.same and not cache.folder.exists()


def test_an_output_the_log_does_not_make_is_found(run: tuple[Settings, str]) -> None:
    settings, run_id = run
    run_folder = find_run(run_id, settings)
    unit = window_folders(run_folder)[1]
    window = run_folder / unit
    picked = saved_m0(window / "DispersionCurves_0000.csv")
    assert picked is not None
    # The curve changed with nothing saying so: no person's mark, no attempt of the log.
    save_pick(window, load_image(window), DispersionCurve(
        fs=picked.fs, vs=picked.vs * 1.01, mode=M0, acquisition=picked.acquisition
    ))  # fmt: skip

    replay = replay_run(run_id, settings, keep=True)

    off = [one for one in replay.compared if not one.within(replay.tolerance)]
    assert [(one.unit, one.stage) for one in off] == [(unit, "picking")]
    # The run's curve 1 % faster: 1 % of the replay's, relative to the run's.
    difference = off[0].difference
    assert difference == pytest.approx(0.01 / 1.01, rel=1e-3)
    assert difference is not None and difference > settings.replay_tolerance == replay.tolerance
    assert f"  {unit}: differs by 9.9e-03" in replay.lines()
    assert replay.folder is not None and (replay.folder / unit).is_dir()


def test_a_persons_work_is_left_out_and_said(run: tuple[Settings, str]) -> None:
    settings, run_id = run
    run_folder = find_run(run_id, settings)
    first, *_, last = window_folders(run_folder)
    mark_edited(run_folder / first, M0)
    # An image made again in PAC: its window's history started afresh by a person.
    forget(run_folder, last, "phase_shift", results=False, actor="user")

    replay = replay_run(run_id, settings)

    assert replay.same
    skipped = {(one.unit, one.stage): one.reason for one in replay.skipped}
    assert skipped[(first, "picking")] == "changed in PAC after PACo picked it"
    assert skipped[(last, "phase_shift")] == "imaged again in PAC"
    assert last not in {one.unit for one in replay.compared}
    assert "Not replayed: picking of 1 unit(s), changed in PAC after PACo picked it." in (
        replay.lines()
    )


def test_a_run_whose_inputs_changed_is_not_replayed(
    run: tuple[Settings, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, run_id = run
    changed = InputFile(name="1.dat", bytes=1, sha256="0" * 64)

    def input_files(profile: Profile) -> tuple[InputFile, ...]:  # noqa: ARG001
        return (changed,)

    monkeypatch.setattr(replaying, "input_files", input_files)

    with pytest.raises(RunError, match=r"inputs changed since it ran: 1\.dat"):
        replay_run(run_id, settings)


def test_the_tolerance_is_relative_to_the_runs_largest_value() -> None:
    run = np.array([0.0, 2.0, -4.0])

    assert replaying._relative(run, run) == 0.0  # pyright: ignore[reportPrivateUsage]
    assert replaying._relative(run, run + 4e-5) == pytest.approx(1e-5)  # pyright: ignore[reportPrivateUsage]


def test_the_inversions_and_soils_are_said_not_replayed() -> None:
    def attempt(unit: str, stage: Stage) -> Attempt:
        at = datetime(2026, 10, 1, tzinfo=UTC)
        return Attempt(
            unit=unit, stage=stage, attempt=1, parameters={}, triggered_by="initial",
            started_at=at, status="succeeded",
        )  # fmt: skip

    attempts = [
        attempt("xmid_8.88", "inversion"),
        attempt("xmid_8.88", "inversion"),  # a retry: the window said once
        attempt("xmid_2.88", "petro_inversion"),
        attempt("xmid_2.88", "picking"),
        attempt(LINE, "inversion"),
    ]

    skipped = replaying._not_replayed(attempts)  # pyright: ignore[reportPrivateUsage]

    assert [(one.unit, one.stage) for one in skipped] == [
        ("xmid_2.88", "petro_inversion"),
        ("xmid_8.88", "inversion"),
    ]
    assert skipped[1].reason.startswith("a random search without a seed")
