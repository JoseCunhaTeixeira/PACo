import logging
import re
import shutil
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from paco.inversion import (
    InversionError,
    InversionParameters,
    InversionRecord,
    ThicknessLayer,
    VsLayer,
    WindowInversion,
    check_inversion,
    find_job,
    invert_run,
    read_record,
    submit_inversion,
    summarize_inversion,
    write_record,
)
from paco.jobs import JobManager
from paco.picks import pick
from paco.quality import dispersion_quality
from paco.runs import run_processing
from paco.settings import Settings

# Four 24-receiver windows along the active demo line, as in test_runs.py; all four are picked.
SMALL_WINDOWS = {"masw": {"length": 24, "step": 24}}
# A short sampler: every step of an inversion, in about a second per window.
SHORT = InversionParameters(n_iterations=500, n_burnin_iterations=50, n_chains=1)
# The files PAC's invert_position writes in a window folder (compared with PAC on 2026-09-23).
PAC_FILES = {
    "SeismicInversion_DensityCurves_0000.png",
    "SeismicInversion_DispersionCurves_0000_best.csv",
    "SeismicInversion_DispersionCurves_0000_ensemble.csv",
    "SeismicInversion_DispersionCurves_0000_median.csv",
    "SeismicInversion_DispersionCurves_0000_smooth_best.csv",
    "SeismicInversion_DispersionCurves_0000_smooth_median.csv",
    "SeismicInversion_DispersionImage_0000.png",
    "SeismicInversion_Log_0000.log",
    "SeismicInversion_Marginals_0000.png",
    "SeismicInversion_Model_0000_best.csv",
    "SeismicInversion_Model_0000_ensemble.csv",
    "SeismicInversion_Model_0000_median.csv",
    "SeismicInversion_Model_0000_smooth_best.csv",
    "SeismicInversion_Model_0000_smooth_median.csv",
}


@dataclass(frozen=True)
class Picked:
    settings: Settings
    run_id: str
    folder: Path


@dataclass(frozen=True)
class Inverted:
    record: InversionRecord
    folder: Path
    progress: list[tuple[int, int]]


# The real runs are the slow part: one run is processed, judged and picked, and inverted once
# with a short sampler. Tests that change a run work on a copy.
@pytest.fixture(scope="module")
def picked(demo_input_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Picked:
    root = tmp_path_factory.mktemp("inversion")
    settings = Settings(input_dir=demo_input_dir, output_dir=root / "outputs", workers=2)
    run_id = run_processing("active_p1", "active", SMALL_WINDOWS, settings).run_id
    dispersion_quality(run_id, settings)
    pick(run_id, settings)
    return Picked(settings, run_id, settings.output_dir / "active_p1" / run_id)


@pytest.fixture(scope="module")
def inverted(picked: Picked, tmp_path_factory: pytest.TempPathFactory) -> Inverted:
    settings, folder = _copy(picked, tmp_path_factory.mktemp("inverted"))
    progress: list[tuple[int, int]] = []
    record = submit_inversion(picked.run_id, SHORT, settings)
    record = invert_run(record, settings, lambda done, total: progress.append((done, total)))
    return Inverted(record, folder, progress)


def _copy(picked: Picked, root: Path) -> tuple[Settings, Path]:
    """A copy of the picked run, to change."""
    settings = picked.settings.model_copy(update={"output_dir": root})
    folder = shutil.copytree(picked.folder, root / "active_p1" / picked.run_id)
    return settings, folder


def _record(**changes: object) -> InversionRecord:
    """A record of a two-window job, with `changes`."""
    fields: dict[str, object] = {
        "job_id": "inv-20260923-100000-abcd",
        "run_id": "20260923-090000-abcd",
        "parameters": SHORT,
        "state": "running",
        "submitted_at": datetime(2026, 9, 23, 10, 0, tzinfo=UTC),
        "started_at": datetime(2026, 9, 23, 10, 0, tzinfo=UTC),
        "total": 2,
    }
    return InversionRecord.model_validate(fields | changes)


def _window(xmid: float, vs: tuple[float, ...], thicknesses: tuple[float, ...]) -> WindowInversion:
    return WindowInversion(
        xmid=xmid,
        folder=f"xmid_{xmid:.2f}",
        status="succeeded",
        vs_m_s=vs,
        thicknesses_m=thicknesses,
    )


# ---------------------------------------------------------------- parameters


def test_parameters_default_to_pacs_form() -> None:
    assert InversionParameters().model_dump() == {
        "n_layers": 2,
        "vs_layers": ({"vs_min": 100.0, "vs_max": 1_000.0, "vs_perturb_std": 20.0},) * 2,
        "thickness_layers": (
            {"thickness_min": 1.0, "thickness_max": 10.0, "thickness_perturb_std": 1.0},
        ),
        "n_iterations": 100_000,
        "n_burnin_iterations": 10_000,
        "n_chains": 5,
    }


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: InversionParameters(n_layers=3), r"vs_layers must have length n_layers \(3\)"),
        (
            lambda: InversionParameters(n_layers=3, vs_layers=(VsLayer(),) * 3),
            r"thickness_layers must have length n_layers - 1 \(2\)",
        ),
        (lambda: VsLayer(vs_min=500, vs_max=400), "vs_max must be greater than vs_min"),
        (
            lambda: ThicknessLayer(thickness_min=5, thickness_max=5),
            "thickness_max must be greater than thickness_min",
        ),
    ],
)
def test_parameters_are_checked(build: Callable[[], object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        build()


# ---------------------------------------------------------------- starting a job


def test_submit_records_a_queued_job(picked: Picked, tmp_path: Path) -> None:
    settings, folder = _copy(picked, tmp_path)

    record = submit_inversion(picked.run_id, SHORT, settings)

    assert re.fullmatch(r"inv-\d{8}-\d{6}-[0-9a-f]{4}", record.job_id)
    assert (record.run_id, record.state, record.total, record.windows) == (
        picked.run_id,
        "queued",
        4,
        (),
    )
    assert read_record(folder) == record


def test_an_inversion_needs_picks(picked: Picked, tmp_path: Path) -> None:
    settings, folder = _copy(picked, tmp_path)
    (folder / "pick.json").unlink()

    with pytest.raises(InversionError, match=r"has no picks yet: call pick first\.$"):
        check_inversion(picked.run_id, settings)


@pytest.mark.parametrize("state", ["queued", "running"])
def test_one_inversion_per_run_at_a_time(picked: Picked, tmp_path: Path, state: str) -> None:
    settings, folder = _copy(picked, tmp_path)
    write_record(folder, _record(run_id=picked.run_id, state=state))

    with pytest.raises(InversionError, match=f"is already being inverted: .* is {state}"):
        submit_inversion(picked.run_id, SHORT, settings)


# ---------------------------------------------------------------- running a job


def test_every_picked_window_is_inverted(inverted: Inverted) -> None:
    record = inverted.record

    assert (record.state, record.total, record.error) == ("succeeded", 4, None)
    assert record.started_at is not None and record.finished_at is not None
    for window in record.windows:
        assert window.status == "succeeded"
        assert window.vs_m_s is not None and len(window.vs_m_s) == 2
        assert window.thicknesses_m is not None and len(window.thicknesses_m) == 1
    assert inverted.progress == [(done, 4) for done in range(5)]
    # What job_status reads is what the job returned.
    assert read_record(inverted.folder) == record


def test_windows_get_pacs_files(inverted: Inverted) -> None:
    for window in inverted.record.windows:
        files = {path.name for path in (inverted.folder / window.folder).iterdir()}
        assert files >= PAC_FILES


def test_a_window_without_m0_fails_alone(picked: Picked, tmp_path: Path) -> None:
    settings, folder = _copy(picked, tmp_path)
    (folder / "xmid_8.88" / "DispersionCurves_0000.csv").unlink()

    record = invert_run(submit_inversion(picked.run_id, SHORT, settings), settings)

    assert record.state == "succeeded"
    statuses = {window.folder: window.status for window in record.windows}
    assert statuses == {
        "xmid_2.88": "succeeded",
        "xmid_8.88": "failed",
        "xmid_14.88": "succeeded",
        "xmid_20.88": "succeeded",
    }
    (failed,) = [window for window in record.windows if window.status == "failed"]
    assert (
        failed.error == "ValueError: No M0 curve in xmid_8.88: pick it, or pick it again by hand."
    )
    assert (folder / "xmid_8.88" / "inversion_error.log").exists()


def test_a_job_that_cannot_run_is_failed(picked: Picked, tmp_path: Path) -> None:
    settings, folder = _copy(picked, tmp_path)
    record = submit_inversion(picked.run_id, SHORT, settings)
    (folder / "pick.json").unlink()

    record = invert_run(record, settings)

    assert record.state == "failed"
    assert (
        record.error == f"InversionError: Run '{picked.run_id}' has no picks yet: call pick first."
    )
    assert read_record(folder) == record


def test_find_job(inverted: Inverted) -> None:
    settings = Settings(input_dir=inverted.folder.parents[2], output_dir=inverted.folder.parents[1])

    assert find_job(inverted.record.job_id, settings) == (inverted.folder, inverted.record)
    for job_id in ("inv-20260923-000000-0000", "../x", "*"):
        with pytest.raises(InversionError, match=f"^Unknown job '{re.escape(job_id)}'"):
            find_job(job_id, settings)


# ---------------------------------------------------------------- the summary on its own


@pytest.mark.parametrize(("state", "live"), [("queued", False), ("running", False)])
def test_a_job_a_stopped_server_left_is_interrupted(state: str, live: bool) -> None:
    status = summarize_inversion(_record(state=state), live=live)

    assert status.state == "interrupted"
    assert status.error == "The server stopped before the job finished: call invert again."


def test_a_live_job_keeps_its_state() -> None:
    assert summarize_inversion(_record(state="running"), live=True).state == "running"


def test_summary_gives_the_range_of_the_models() -> None:
    record = _record(
        state="succeeded",
        finished_at=datetime(2026, 9, 23, 10, 2, 5, tzinfo=UTC),
        total=3,
        windows=(
            _window(1.0, (200.0, 400.0), (3.0,)),
            _window(2.0, (220.0, 500.0), (5.0,)),
            WindowInversion(xmid=3.0, folder="xmid_3.00", status="failed", error="ValueError: x"),
        ),
    )

    status = summarize_inversion(record, live=False)

    assert status.model_dump(exclude={"job_id", "run_id"}) == {
        "state": "succeeded",
        "done": 3,
        "total": 3,
        "n_failed": 1,
        "elapsed_s": 125.0,
        "vs_m_s": ((200.0, 220.0), (400.0, 500.0)),
        "depths_m": ((3.0, 5.0),),
        "errors": ("xmid 3.00: ValueError: x",),
        "error": None,
    }


def test_depths_add_up_the_thicknesses() -> None:
    record = _record(windows=(_window(1.0, (150.0, 300.0, 600.0), (2.0, 4.0)),), total=1)

    assert summarize_inversion(record, live=True).depths_m == ((2.0, 2.0), (6.0, 6.0))


def test_summary_reports_a_few_distinct_errors() -> None:
    def failed(xmid: float, error: str) -> WindowInversion:
        return WindowInversion(xmid=xmid, folder=f"xmid_{xmid:.2f}", status="failed", error=error)

    windows = [failed(1.0, "A"), failed(2.0, "A"), failed(3.0, "B"), failed(4.0, "C")]
    record = _record(windows=(*windows, failed(5.0, "D")), total=5)

    assert summarize_inversion(record, live=True).errors == (
        "xmid 1.00: A",
        "xmid 3.00: B",
        "xmid 4.00: C",
    )


# ---------------------------------------------------------------- the job manager


def test_jobs_run_one_at_a_time_and_stay_live_until_done() -> None:
    jobs = JobManager()
    release, second_started = threading.Event(), threading.Event()
    jobs.submit("first", lambda: release.wait(timeout=10))
    jobs.submit("second", second_started.set)

    # The second job waits for the first.
    assert not second_started.wait(timeout=0.2)
    assert jobs.is_live("first") and jobs.is_live("second")

    release.set()
    assert second_started.wait(timeout=10)
    # A job queued after both: once it runs, both are done.
    finished = threading.Event()
    jobs.submit("marker", finished.set)
    assert finished.wait(timeout=10)
    assert not jobs.is_live("first")
    assert not jobs.is_live("second")


def test_a_crashing_job_is_logged_and_forgotten(caplog: pytest.LogCaptureFixture) -> None:
    jobs = JobManager()
    finished = threading.Event()

    def crash() -> None:
        raise RuntimeError("boom")

    with caplog.at_level(logging.ERROR, logger="paco.jobs.manager"):
        jobs.submit("broken", crash)
        jobs.submit("marker", finished.set)
        assert finished.wait(timeout=10)

    assert not jobs.is_live("broken")
    assert "Job broken crashed" in caplog.text
